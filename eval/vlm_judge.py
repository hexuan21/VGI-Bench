"""VLM-as-judge for generated benchmark outputs (agentic for videos).

For each task under `eval/eval_res/<model_dir>/`, looks up the rubric in
`data/3_verified/<task>/_judge_rubric_<task>.json` (with fallback to
other pipeline stages), shows the judge VLM the original input image,
the GT image, and the model's generated output (single image, OR a
video judged via an iterative agentic loop). For videos the per-instance
score is Final = Completeness x Rubric, and an instance counts as a
"success" only when BOTH Completeness and Rubric are perfect (== 1).

For videos the FULL METHOD loop is:
  Round 1:  sample at 4 fps across the whole clip, in 10-frame windows
            with 1-frame overlap. Ask the judge to evaluate frames and
            nominate frame indices that need closer inspection.
  Round 2:  re-sample small windows around those nominated frame indices
            at 8 fps and judge those windows. Round-2 is driven ONLY by
            the LLM's nominated frames.
  Polish:   with no images, hand the LLM all comments grouped by
            original-video frame range and ask it to deduplicate them
            into a per-rubric-item violation count (-> Rubric Score).
  Completeness: one separate low-fps (2 fps) all-frames call scoring how
            far the goal was reached (tier 0/1/2 -> 0/0.5/1).

All per-instance verdicts are aggregated into a single JSON at:
    eval/eval_res/<model_dir>/_vlm_judge_<model_dir>.json
A per-task detailed trace is also written next to the generated outputs as
    eval/eval_res/<model_dir>/<task>/_judge_<task>.json

Ablations (to validate the full method's components) are selected with
CLI flags; each non-default config writes its per-task JSON as
`_judge_<task>_abl_<config>.json` so it sits next to the full-method
result instead of clobbering it:
  --sampling even         one fixed fps over the whole clip, no round-2
                          zoom (vs. the adaptive round1+round2 schedule).
  --batching all_at_once  one LLM call sees every frame (vs. 10-frame
                          windows with 1-frame overlap).
  --even_fps {2,4,8}      the fixed fps for --sampling even.
The sampling and batching dimensions combine freely.

Both passes run by default (Final = Comp x Rub; success = Comp==1 and
Rub==1). Score a single pass with --no_completeness (rubric only, writes
`_abl_rub_only`) or --no_rubric (completeness only, writes `_abl_comp_only`);
the two flags cannot be combined.

Defaults to `google/gemini-3-flash-preview` via NetMind. Reads
GOOGLE_API_KEY, OPENROUTER_API_KEY, NETMIND_API_KEY and optionally
OPENAI_BASE_URL from `<repo>/const/_api_key.json`.

Usage:
    python vlm_judge.py veo3_1                         # judge everything
    python vlm_judge.py veo3_1 --task maze_square
    python vlm_judge.py veo3_1 --sampling even --even_fps 2
    python vlm_judge.py veo3_1 --sampling even --batching all_at_once
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# Single source of truth for GT signal routing — see vlm_judge_utils.py.
# A task's `task_gt_signals(name)` says whether to feed gt_image and/or
# gt_desc to the judge prompt; `gt_desc_for(...)` materialises the desc
# string (task-wide + per-instance concatenated).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_judge_utils import (  # noqa: E402
    gt_desc_for, task_gt_signals,
    FRAME_VLM_MAX_SIDE, REF_VLM_MAX_SIDE,
    image_to_data_url, _zoom_max_per_batch,
    AGENTIC_SYSTEM, COMPLETENESS_SYSTEM,
    _batch_user_message,
    _polish_messages, _completeness_messages,
    # LLM-call machinery (moved into vlm_judge_utils; re-exported here):
    load_judge_credentials, _post_chat, _post_chat_one,
    _post_chat_clawbench, _log_llm_usage,
    DEFAULT_JUDGE_MODEL, DEFAULT_BASE_URL, GOOGLE_BASE_URL,
    NETMIND_BASE_URL, NETMIND_MODELS,
    CLAWBENCH_BASE_URL, CLAWBENCH_MODELS, CLAWBENCH_MODEL_ID,
    API_KEY_FILE, USAGE_DIR, LLM_USAGE_LOG, OPENROUTER_USAGE_LOG,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = PROJECT_ROOT / "eval" / "eval_res"
# Judge results live in their own tree, mirrored per-model under
# eval/judge_res/<model>/, with `_judge_<task>_<model>.json` per task
# and `_all_judge_<model>.json` as the top-level summary. The old
# in-place layout (one judge json next to the model's mp4 outputs)
# is preserved only in eval/_discarded/_judge/ for historical reference.
JUDGE_RES_ROOT = PROJECT_ROOT / "eval" / "judge_res"
TASK_NAMES_YAML = PROJECT_ROOT / "data" / "task_names.yaml"


def load_task_classes() -> dict[str, str]:
    """Parse task_names.yaml: minimal grouped-list YAML (`class:` header
    followed by `- task_name` lines). Returns {task_name: class_name}."""
    classes: dict[str, str] = {}
    if not TASK_NAMES_YAML.is_file():
        return classes
    current: str | None = None
    for line in TASK_NAMES_YAML.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("- "):
            task = s[2:].strip()
            if current and task:
                classes[task] = current
            continue
        if ":" in s:
            key, _, val = s.partition(":")
            key, val = key.strip(), val.strip()
            if val:
                classes[key] = val            # legacy flat form
            else:
                current = key                 # grouped header
    return classes


def _relpath_from_project(p: Path) -> str:
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


DATA_ROOTS = [
    # Public layout: the HF dataset's bench_data/ dropped into data/. Absent in
    # the private tree, so it costs nothing there and needs no configuration in
    # a fresh clone of the public repo.
    PROJECT_ROOT / "data" / "bench_data",
    PROJECT_ROOT / "data" / "5_ok",
    PROJECT_ROOT / "data" / "4_checking",
    PROJECT_ROOT / "data" / "3_verified",
    PROJECT_ROOT / "data" / "3_gen_ok",
    PROJECT_ROOT / "data" / "2_auto_gen",
    PROJECT_ROOT / "data" / "2_gen",
    PROJECT_ROOT / "data" / "1_stack",
    PROJECT_ROOT / "data" / "1_processing",
]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv"}

# Outer loop has TWO tiers:
#   round 1 = AGENTIC_FPS_SCHEDULE[0] sampled across the whole clip;
#   round 2 = AGENTIC_FPS_SCHEDULE[1] sampled in zoom windows around
#            each round-1 frame the LLM asked to "inspect closely".
# Inner loop: chunk a round's frame list into windows of BATCH_SIZE
# with BATCH_OVERLAP frames shared with the previous window
# (stride = BATCH_SIZE - BATCH_OVERLAP). Each window is its own VLM
# call carrying rubric + GT image. Batches within ONE round run in
# parallel with up to AGENTIC_NUM_WORKERS threads. Round-2 zoom is driven
# solely by the frames the LLM nominated via `frames_to_inspect_closely`.
AGENTIC_FPS_SCHEDULE = (4, 8)
# The full method = adaptive fps + 10-frame window with 1-frame overlap.
AGENTIC_BATCH_SIZE = 10
AGENTIC_BATCH_OVERLAP = 1
AGENTIC_NUM_WORKERS = 2

# Completeness pass: ONE low-fps LLM call (all frames at once) that scores how
# far the task goal was carried out, against <task>/_completeness_judge.txt
# (three tiers 0/1/2 -> normalized to 0/0.5/1). It is INDEPENDENT of the
# rubric-checklist path. final_score = rubric_score (the arith aggregation)
# * completeness_score. ON by default; disable via --no_completeness (CLI)
# or ablation_cfg["completeness"]=False. An instance counts as a success
# only when BOTH completeness_score and rubric_score are 1.
COMPLETENESS_DEFAULT = True
COMPLETENESS_FPS = 2
COMPLETENESS_TIER_TO_SCORE = {0: 0.0, 1: 0.5, 2: 1.0}
COMPLETENESS_FILENAME = "_completeness_judge.txt"

# A batch_size large enough that _make_batches always yields a single
# batch — used by the `all_at_once` batching ablation.
ABLATION_ALL_AT_ONCE_BATCH = 100_000

# Per-checklist-item score from violation count: score = 1/(x+1),
# x = polish's final violation count for that item. Range [0, 1]. The
# instance `rubric_score` is the arithmetic mean over checklist items —
# see `_build_checklist_detail`.


def _viol_to_score(n: int) -> float:
    if n < 0:
        n = 0
    return 1.0 / (n + 1)


def _arith_mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _build_checklist_detail(rubric_checklist: list[str] | None,
                            raw_items) -> dict | None:
    """Normalise per-item judge output into a fixed-length structure aligned
    to `rubric_checklist`. Each item carries the raw `n_violations` count
    from the polish/judge LLM and a derived `score = 1/(n_violations+1)`
    in [0, 1]. Missing / malformed entries default to 1 violation with a
    placeholder evidence string. The instance-level rubric score is
    the arithmetic mean over checklist items."""
    if not rubric_checklist:
        return None
    by_index: dict[int, dict] = {}
    if isinstance(raw_items, list):
        for it in raw_items:
            if not isinstance(it, dict):
                continue
            idx = it.get("index")
            if isinstance(idx, int) and 1 <= idx <= len(rubric_checklist):
                by_index[idx] = it
    items_out: list[dict] = []
    n_total = len(rubric_checklist)
    n_satisfied = 0
    per_item_scores: list[float] = []
    for i, text in enumerate(rubric_checklist, 1):
        entry = by_index.get(i) or {}
        n_viol_raw = entry.get("n_violations", entry.get("violations"))
        try:
            n_viol = max(0, int(n_viol_raw))
        except (TypeError, ValueError):
            n_viol = 1
        score = _viol_to_score(n_viol)
        ev = str(entry.get("evidence", "")).strip()
        if not ev:
            ev = ("(model did not return an entry for this item)"
                  if i not in by_index else "(no evidence provided)")
        sat = (n_viol == 0)
        items_out.append({
            "index": i, "item": text,
            "n_violations": n_viol,
            "score": round(score, 4),
            "satisfied": sat, "evidence": ev,
        })
        per_item_scores.append(score)
        if sat:
            n_satisfied += 1
    return {
        "n_total": n_total,
        "n_satisfied": n_satisfied,
        "rubric_score_arith": round(_arith_mean(per_item_scores), 4),
        "items": items_out,
    }
# Per round-1 batch the LLM may emit up to ceil(batch_size/2)
# round-1-local frame indices to re-examine at round-2 fps. For each
# flagged frame we sample (2*y + 1) frames at round-2 fps (y before,
# the frame itself, y after), with y chosen so the zoom window stays
# shorter than batch_size. See _zoom_y().

_NAT_RE = re.compile(r"(\d+)")


def _natkey(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in _NAT_RE.split(str(s))]


# ---------- task / rubric / reference-image discovery ----------------------

def find_data_task_dir(task_name: str) -> Path | None:
    for root in DATA_ROOTS:
        for d in (root / task_name, root / f"v_{task_name}"):
            if d.is_dir():
                return d
    return None


_RUBRIC_ROOT_OVERRIDE: Path | None = None


def find_rubric_file(task_name: str) -> Path | None:
    # --rubric_root, if set, wins over DATA_ROOTS for the rubric lookup only.
    fname = f"_judge_rubric_{task_name}.json"
    roots = ([_RUBRIC_ROOT_OVERRIDE] if _RUBRIC_ROOT_OVERRIDE is not None
             else []) + list(DATA_ROOTS)
    for root in roots:
        for d in (root / task_name, root / f"v_{task_name}"):
            for cand in (d / "judge_spec" / fname, d / fname):  # new, then legacy
                if cand.is_file():
                    return cand
    return None


def find_video_prompt_file(task_name: str) -> Path | None:
    """Used as a backstop when the rubric is empty: the video prompt at
    least tells the judge what behaviour was requested."""
    for root in DATA_ROOTS:
        for d in (root / task_name, root / f"v_{task_name}"):
            for cand in (d / "prompt" / "video_gen_prompt" / f"{task_name}.txt",
                         d / f"_video_prompt_{task_name}.txt"):  # new, then legacy
                if cand.is_file():
                    return cand
    return None


def discover_tasks_from_data(tasks_filter: set[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for root in DATA_ROOTS:
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir(), key=lambda p: _natkey(p.name)):
            if not d.is_dir() or d.name.startswith(("_", ".")):
                continue
            task_name = d.name[2:] if d.name.startswith("v_") else d.name
            if task_name in seen:
                continue
            if tasks_filter and task_name not in tasks_filter:
                continue
            seen.add(task_name)
            out.append(task_name)
    return out


def load_rubric(task_name: str) -> dict[str, dict]:
    f = find_rubric_file(task_name)
    if not f:
        return {}
    try:
        # utf-8-sig: tolerate a UTF-8 BOM (PowerShell's default Set-Content
        # encoding) which would otherwise make json.loads choke.
        data = json.loads(f.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    if isinstance(data, list):
        return {e["id"]: e for e in data
                if isinstance(e, dict) and "id" in e}
    if isinstance(data, dict):
        if "rubric" in data or "answer" in data:
            return {"default": data}
        return data
    return {}


def select_rubric_entry(rubric_map: dict[str, dict], task_name: str,
                        level: str, iid: str) -> dict:
    if not rubric_map:
        return {}
    keys: list[str] = []
    if level != "no_lv":
        keys.append(f"{task_name}_{level}_{iid}")
    keys.extend([
        f"{task_name}_{iid}",
        f"{task_name}_default",
        "default",
        task_name,
        "rubric",
    ])
    for key in keys:
        entry = rubric_map.get(key)
        if isinstance(entry, dict):
            return entry

    # Some task rubrics are task-wide but only contain one legacy id. Prefer
    # that over treating the task as unrubriced.
    if len(rubric_map) == 1:
        only = next(iter(rubric_map.values()))
        return only if isinstance(only, dict) else {}
    return {}


def _rubric_items_raw(rubric_entry: dict) -> list[str]:
    """Returns the rubric list verbatim (overall + sub-items, in order).
    Returns [] if the rubric is not a list."""
    if not rubric_entry:
        return []
    rubric = rubric_entry.get("rubric")
    if isinstance(rubric, list):
        return [str(x).strip() for x in rubric if str(x).strip()]
    return []


_OVERALL_PREFIX_RE = re.compile(
    r"^\s*overall\s*task\s*[:：-]+\s*", re.IGNORECASE)


def rubric_items_for_persistence(rubric_entry: dict) -> list[str]:
    """Return the rubric checklist as a list of strings — the same order
    seen in the source `_judge_rubric_<task>.json` (item 0 = overall
    task, items 1.. = per-checklist sub-items). This is the form
    persisted under the per-instance `rubric` field in the per-task
    judge json; the joined-text form is only used as a fallback inside
    prompts when neither `rubric_overall` nor `rubric_checklist` resolve."""
    return _rubric_items_raw(rubric_entry)


def extract_rubric_overall(rubric_entry: dict) -> str:
    """First rubric item is the high-level task description, embedded in
    the judge prompt under a `Task main goal:` header. Returns "" if
    there is no rubric list.

    Strips the legacy `Overall task:` prefix (~95 of our rubric files
    open with it) — the surrounding header already labels this block,
    so the inline prefix would just produce `Task main goal:\nOverall
    task: ...` in the prompt."""
    items = _rubric_items_raw(rubric_entry)
    if not items:
        return ""
    return _OVERALL_PREFIX_RE.sub("", items[0]).strip()


def extract_rubric_checklist(rubric_entry: dict) -> list[str]:
    """Sub-rubric items (everything after the first overall-task item).
    These are the items polish counts violations against and that drive
    `rubric_score`. If only one item exists, treat it as both overall
    AND the lone checklist item so single-item rubrics keep working."""
    items = _rubric_items_raw(rubric_entry)
    if len(items) >= 2:
        return items[1:]
    return items[:]


def rubric_text_for(rubric_entry: dict, task_name: str | None = None) -> str:
    """Render a rubric for embedding in a judge prompt.

    Numbering must match the per-item `checklist` schema the judge is
    asked to fill in: rubric[0] is the overall-task description (the
    `Task main goal:` header, NOT counted as a checklist item) and
    rubric[1:] is the 1..K checklist. We therefore render the overall
    item separately ("Overall task: ...") and number the K checklist
    items from 1, so the indices the model sees inside the prompt
    line up with the indices it writes into the JSON output. (The
    earlier behaviour numbered ALL items including overall, leaving
    the model with TWO conflicting numberings for the same items.)"""
    parts = []
    have_rubric_content = False
    if rubric_entry:
        if "answer" in rubric_entry:
            parts.append(
                "Reference answer: " +
                json.dumps(rubric_entry["answer"], ensure_ascii=False))
            have_rubric_content = True
        rubric = rubric_entry.get("rubric")
        if isinstance(rubric, list) and rubric:
            if len(rubric) >= 2:
                parts.append("Overall task: " + str(rubric[0]).strip())
                parts.append("")
                parts.append("Checklist (1-based indices match the "
                             "`checklist[*].index` field below):")
                checklist_items = rubric[1:]
            else:
                # single-item rubric: that item IS the lone checklist
                # item; no separate overall description.
                parts.append("Checklist:")
                checklist_items = rubric
            for i, item in enumerate(checklist_items, 1):
                parts.append(f"  {i}. {item}")
            have_rubric_content = True
        elif isinstance(rubric, str) and rubric.strip():
            parts.append("Rubric: " + rubric.strip())
            have_rubric_content = True
    if not have_rubric_content and task_name:
        prompt_file = find_video_prompt_file(task_name)
        if prompt_file:
            parts.append("(No explicit rubric; falling back to the video "
                         "prompt as a description of the requested behaviour.)")
            parts.append("Video prompt:\n" + prompt_file.read_text(
                encoding="utf-8-sig").strip())
            have_rubric_content = True
    if not have_rubric_content:
        return "(no rubric available)"
    return "\n".join(parts)


def find_input_image(data_task_dir: Path | None, task: str,
                     level: str, iid: str) -> Path | None:
    if data_task_dir is None:
        return None
    inp = data_task_dir / "input_image"     # new layout, then legacy at task root
    for ext in ("png", "jpg", "jpeg"):
        if level == "no_lv":
            for cand in (inp / f"{task}_{iid}.{ext}",
                         data_task_dir / level / f"{task}_{iid}.{ext}",
                         data_task_dir / f"{task}_{iid}.{ext}"):
                if cand.is_file():
                    return cand
        else:
            for cand in (inp / level / f"{task}_{level}_{iid}.{ext}",
                         data_task_dir / level / f"{task}_{level}_{iid}.{ext}"):
                if cand.is_file():
                    return cand
    return None


def find_gt_image(data_task_dir: Path | None, task: str,
                  level: str, iid: str) -> Path | None:
    if data_task_dir is None:
        return None
    inp = data_task_dir / "input_image"     # new layout, then legacy at task root
    for ext in ("png", "jpg", "jpeg"):
        if level == "no_lv":
            for cand in (inp / f"{task}_{iid}_gt.{ext}",
                         data_task_dir / level / f"{task}_{iid}_gt.{ext}",
                         data_task_dir / f"{task}_{iid}_gt.{ext}"):
                if cand.is_file():
                    return cand
        else:
            for cand in (inp / level / f"{task}_{level}_{iid}_gt.{ext}",
                         data_task_dir / level / f"{task}_{level}_{iid}_gt.{ext}"):
                if cand.is_file():
                    return cand
    # Fallback: intermediate GT under _raw/<level>/. The intermediate
    # render suffix is task-dependent:
    #   `_sim`    Blender / physics renders
    #   `_line`   2D matplotlib / PIL line-art renders
    #   `_render` legacy unified suffix (pre-2026-05 stage-2 tasks still
    #             use this — keep as a fallback so find_gt_image doesn't
    #             return None on the older data roots).
    raw_dir = data_task_dir / "_raw"
    if raw_dir.is_dir():
        for ext in ("png", "jpg", "jpeg"):
            for suffix in ("sim", "line", "render"):
                if level == "no_lv":
                    cand = raw_dir / f"{task}_{iid}_{suffix}_gt.{ext}"
                else:
                    cand = (raw_dir / level /
                            f"{task}_{level}_{iid}_{suffix}_gt.{ext}")
                if cand.is_file():
                    return cand
    return None


# ---------- output filename parsing ---------------------------------------

def parse_output_filename(filename: str, model_dir: str, task: str
                          ) -> tuple[str, str, str] | None:
    # `level` may contain underscores (e.g. "no_lv"). The greedy match
    # backtracks so the trailing _<iid:3-digit>.<ext> still binds.
    pat = re.compile(
        rf"^{re.escape(model_dir)}_{re.escape(task)}_"
        rf"(?P<level>[A-Za-z0-9_]+)_(?P<iid>\d{{3}})\.(?P<ext>\w+)$",
        re.IGNORECASE,
    )
    m = pat.match(filename)
    if not m:
        return None
    return m.group("level"), m.group("iid"), m.group("ext").lower()


def guess_kind(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    return "unknown"


def collect_outputs(model_dir: Path, model_dir_name: str, task_name: str,
                    data_task_dir: Path | None
                    ) -> list[tuple[str, str, Path]]:
    task_subdir = model_dir / task_name
    out: list[tuple[str, str, Path]] = []
    if task_subdir.is_dir():
        for f in sorted(task_subdir.iterdir(),
                        key=lambda p: _natkey(p.name)):
            if not f.is_file() or f.name.startswith(("_", ".")):
                continue
            parsed = parse_output_filename(f.name, model_dir_name, task_name)
            if parsed:
                level, iid, _ = parsed
                out.append((level, iid, f))
    return out


# ---------- ffprobe / ffmpeg ----------------------------------------------

def probe_video(path: Path) -> tuple[float, float]:
    """Returns (duration_seconds, native_fps)."""
    out = subprocess.check_output(
        ["ffprobe", "-v", "error",
         "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1",
         str(path)], text=True).strip().splitlines()
    fr_str = out[0].strip()
    duration = float(out[1].strip())
    if "/" in fr_str:
        num, _, den = fr_str.partition("/")
        fps = float(num) / float(den) if float(den) else 0.0
    else:
        fps = float(fr_str)
    return duration, fps


def extract_frame_records_at_times(video_path: Path, times: list[float],
                                   out_dir: Path,
                                   prefix: str) -> list[tuple[int, Path]]:
    """Extract a frame per requested time.

    Returns (requested_index, path) for successful extractions so callers can
    keep frame metadata aligned even if an ffmpeg seek fails in the middle.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[tuple[int, Path]] = []
    for i, t in enumerate(times):
        out = out_dir / f"{prefix}_{i:03d}_t{t:.3f}.png"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 # Force single-threaded decode + small thread queue so
                 # 4 parallel vlm_judge processes don't blow Windows
                 # memory by each forking ffmpeg's default 4-8 decoder
                 # threads → 16-32 concurrent h264 decoders → OOM.
                 "-threads", "1", "-thread_queue_size", "8",
                 "-ss", f"{max(0.0, t):.3f}", "-i", str(video_path),
                 "-frames:v", "1", "-q:v", "2", str(out)],
                check=True)
        except subprocess.CalledProcessError as e:
            print(f"    [warn] ffmpeg failed at t={t:.3f}s: {e}", flush=True)
            continue
        if not out.is_file() or out.stat().st_size == 0:
            print(f"    [warn] no frame produced at t={t:.3f}s "
                  f"(seek past EOF?)", flush=True)
            continue
        records.append((i, out))
    return records


def _attach_extracted_frames(metas: list[dict],
                             frame_records: list[tuple[int, Path]]
                             ) -> list[dict]:
    aligned: list[dict] = []
    for src_idx, frame_path in frame_records:
        if not (0 <= src_idx < len(metas)):
            continue
        m = dict(metas[src_idx])
        m["round_local_idx"] = len(aligned)
        m["path"] = str(frame_path)
        aligned.append(m)
    return aligned


def _compact_agentic_trace(trace: dict) -> dict:
    """Slim a per-instance agentic trace before persisting:
      * Replace fully-uniform per-frame metadata lists with one summary string
        ("n=25 t=[0.00..6.00]s global=[0..143]") instead of a 25-item list of
        dicts.
      * Collapse parallel index arrays inside each batch evaluation
        (`batch_local_indices`, `round_local_indices`, `global_frames`,
        `time_s_list`) into bracketed inline strings so each batch reads on a
        few lines instead of dozens.
      * Render two-element ranges (`time_s_range`, `global_frame_range`,
        `round_local_range`) as `"a..b"` strings.
      * Drop empty / vacuous diagnostic fields (`flagged_frames`,
        `zoom_intervals_raw`, `zoom_intervals_merged` when empty;
        `frames_to_inspect_closely` when nothing was flagged).
      * Strip per-round duplicates of values already on the trace
        (`zoom_y`, `zoom_max_per_batch`, `batch_size`, `batch_overlap`,
        `num_workers`).

    Operates on a copy; original is not mutated. Pass either the agentic blob
    directly OR a wrapper dict containing `agentic`; both are handled."""
    if not isinstance(trace, dict):
        return trace
    if "agentic" in trace and isinstance(trace["agentic"], dict):
        return {**trace, "agentic": _compact_agentic_trace(trace["agentic"])}

    out = dict(trace)

    def _arr(xs, fmt: str = "{}") -> str:
        return "[" + ", ".join(fmt.format(x) for x in xs) + "]"

    def _frames_summary(fs: list) -> str:
        if not fs:
            return "n=0"
        ts = [f.get("time_s") for f in fs if isinstance(f, dict)]
        gs = [f.get("global_frame") for f in fs if isinstance(f, dict)]
        if ts and gs:
            return (f"n={len(fs)} t=[{ts[0]:.2f}..{ts[-1]:.2f}]s "
                    f"global=[{gs[0]}..{gs[-1]}]")
        return f"n={len(fs)}"

    rounds = out.get("rounds")
    if isinstance(rounds, list):
        new_rounds = []
        for r in rounds:
            if not isinstance(r, dict):
                new_rounds.append(r)
                continue
            r = dict(r)
            if isinstance(r.get("frames"), list):
                r["frames"] = _frames_summary(r["frames"])
            batches = r.get("batches")
            if isinstance(batches, list):
                new_batches = []
                for b in batches:
                    if not isinstance(b, dict):
                        new_batches.append(b)
                        continue
                    b = dict(b)
                    if isinstance(b.get("frames"), list):
                        b["frames"] = _frames_summary(b["frames"])
                    for fld in ("round_local_range", "global_frame_range",
                                "time_s_range"):
                        v = b.get(fld)
                        if isinstance(v, list) and len(v) == 2:
                            a, c = v
                            if fld == "time_s_range":
                                b[fld] = f"{a:.2f}..{c:.2f}"
                            else:
                                b[fld] = f"{a}..{c}"
                    evals = b.get("evaluations")
                    if isinstance(evals, list):
                        new_evals = []
                        for ev in evals:
                            if not isinstance(ev, dict):
                                new_evals.append(ev)
                                continue
                            ev = dict(ev)
                            for fld in ("batch_local_indices",
                                        "round_local_indices",
                                        "global_frames"):
                                v = ev.get(fld)
                                if isinstance(v, list):
                                    ev[fld] = _arr(v)
                            v = ev.get("time_s_list")
                            if isinstance(v, list):
                                ev["time_s_list"] = _arr(v, "{:.2f}")
                            new_evals.append(ev)
                        b["evaluations"] = new_evals
                    fic = b.get("frames_to_inspect_closely")
                    if isinstance(fic, dict):
                        if all(not v for v in fic.values()):
                            b["frames_to_inspect_closely"] = "(none)"
                        else:
                            b["frames_to_inspect_closely"] = {
                                k: (_arr(v, "{:.2f}" if k == "time_s" else "{}")
                                    if isinstance(v, list) else v)
                                for k, v in fic.items()
                            }
                    new_batches.append(b)
                r["batches"] = new_batches
            for empty_fld in ("flagged_frames", "zoom_intervals_raw",
                              "zoom_intervals_merged"):
                if empty_fld in r and not r[empty_fld]:
                    r.pop(empty_fld)
            for dup_fld in ("zoom_y", "zoom_max_per_batch", "batch_size",
                            "batch_overlap", "num_workers"):
                r.pop(dup_fld, None)
            new_rounds.append(r)
        out["rounds"] = new_rounds

    fe = out.get("frame_evaluations")
    if isinstance(fe, dict):
        new_fe: dict = {}
        for gf, entries in fe.items():
            if isinstance(entries, list):
                items = []
                for e in entries:
                    if isinstance(e, str):
                        items.append(e)            # already in new string form
                    elif isinstance(e, dict):
                        t = e.get("time_s")
                        items.append(
                            f"r{e.get('round')}b{e.get('batch_idx')} "
                            f"t={t:.2f}s: {e.get('comment', '')}"
                            if isinstance(t, (int, float))
                            else
                            f"r{e.get('round')}b{e.get('batch_idx')}: "
                            f"{e.get('comment', '')}"
                        )
                if items:
                    new_fe[gf] = items[0] if len(items) == 1 else items
                else:
                    new_fe[gf] = entries
            else:
                new_fe[gf] = entries
        out["frame_evaluations"] = new_fe

    return out


# Per-instance headline fields, in display order. _shape_instance floats
# these up to just before `fps_schedule` (or to the end of the dict for
# early-return traces that have no fps_schedule).
_INSTANCE_HEADLINE = ["final_score", "rubric_score", "completeness_score",
                      "completeness_tier", "success", "overall_reasoning",
                      "completeness_reasoning"]
# Per-instance fields dropped entirely from the persisted JSON. The
# agentic-config constants (batch_size / batch_overlap / num_workers /
# zoom_y / zoom_max_per_batch) are runtime parameters or derived module
# constants that don't vary per instance — the top-level summary
# already records the schedule, no need to repeat on every instance.
_INSTANCE_DROP = {"task", "level", "iid",
                  "batch_size", "batch_overlap",
                  "num_workers", "zoom_y", "zoom_max_per_batch",
                  # `frame_evaluations` is a reorganisation of the same
                  # eval comments already inside `rounds[*].batches[*]`
                  # — kept transiently to feed polish, not worth
                  # persisting (~4 KB/instance of duplicate text).
                  "frame_evaluations"}


def _slim_checklist_detail(cd: dict | None) -> dict | None:
    """Strip duplicated / derivable fields from a `checklist_detail`
    block so the per-instance JSON only carries the irreducible signal:
      * top-level `n_total` / `n_satisfied` (= counts) — derivable from items
      * top-level `rubric_score_arith` — already floated to the instance
        headline as `rubric_score`
      * each `items[i]` keeps only {index, n_violations, evidence}; the
        `item` text duplicates `rubric[i]`, `score` is `1/(n_violations+1)`,
        `satisfied` is `n_violations == 0`.
    Idempotent."""
    if not isinstance(cd, dict):
        return cd
    items = cd.get("items")
    if not isinstance(items, list):
        return {"items": []}
    slim_items = []
    for it in items:
        if not isinstance(it, dict):
            slim_items.append(it)
            continue
        slim_items.append({
            "index":         it.get("index"),
            "n_violations":  it.get("n_violations"),
            "evidence":      it.get("evidence", ""),
        })
    return {"items": slim_items}


def _level_from_id(inst_id: str, task_name: str | None = None) -> str:
    """Recover the level token from an instance id.

    Ids are built as `<task>_<level>_<NNN>` for leveled layouts and
    `<task>_<NNN>` for flat (no_lv) layouts. The level token is NOT
    restricted to `lvN` — it can be `3pcs`, `1x2`, `3x3`, `4x4`, ... so
    we cannot pattern-match it. With `task_name` we strip the known
    `<task>_` prefix and trailing `_<NNN>` to get the level exactly;
    without it we fall back to the second-to-last underscore group
    (correct only when the task name has no underscores)."""
    s = str(inst_id)
    if task_name and s.startswith(task_name + "_"):
        rest = s[len(task_name) + 1:]            # "<level>_<NNN>" or "<NNN>"
        m = re.match(r"^(.+)_(\d{3})$", rest)
        return m.group(1) if m else "no_lv"      # flat layout -> no level token
    m = re.match(r"^(.+)_(\d{3})$", s)
    if not m:
        return "no_lv"
    head = m.group(1)
    return head.rsplit("_", 1)[1] if "_" in head else "no_lv"


def _build_violation_summ(checklist_detail: dict | None) -> str | None:
    """Compact one-line summary of how many distinct substantive
    violations the judge counted per rubric checklist item, in the same
    1-based order shown to the model. Format: `"1:0 2:1 3:2 4:0"` (item
    1 had 0 violations, item 2 had 1, ...). Lives in the persisted
    instance dict right under `rubric` so a reader scanning the file
    sees the score breakdown immediately without expanding
    `checklist_detail`.

    Returns None when there is no checklist (e.g. failed-extract
    early-return traces, or completeness-only runs with no rubric)."""
    if not isinstance(checklist_detail, dict):
        return None
    items = checklist_detail.get("items")
    if not isinstance(items, list) or not items:
        return None
    parts: list[str] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        idx = it.get("index")
        nv  = it.get("n_violations")
        try:
            nv = max(0, int(nv))
        except (TypeError, ValueError):
            nv = "?"
        parts.append(f"{idx}:{nv}")
    return " ".join(parts) if parts else None


def _shape_instance(inst: dict) -> dict:
    """Final per-instance JSON shape for the per-task judge file:
      * drop task / level / iid (level is still recoverable from `id`);
      * float the 5 headline fields (rubric_score .. overall_reasoning)
        up to just before `fps_schedule`;
      * move `rubric` to immediately after `fps_schedule` (it's bulky and
        belongs near the agentic-schedule block, not next to `id`);
      * derive a compact `violation_summ` from `checklist_detail` and
        place it immediately after `rubric` so the per-item violation
        counts are visible without expanding the nested checklist block;
      * emit `checklist_detail` immediately BEFORE `rounds` so the
        final per-item evidence sits next to its headline scores rather
        than after the bulky agentic trace.
    Tolerant of missing keys — early-return and completeness-only traces
    that have no `fps_schedule` get headline fields and `rubric` appended at
    the end."""
    if not isinstance(inst, dict):
        return inst
    src = {k: v for k, v in inst.items() if k not in _INSTANCE_DROP}
    if "checklist_detail" in src:
        viol_summ = _build_violation_summ(src["checklist_detail"])
        src["checklist_detail"] = _slim_checklist_detail(
            src["checklist_detail"])
    else:
        viol_summ = None

    def _maybe_emit_viol(d: dict) -> None:
        if viol_summ is not None:
            d["violation_summ"] = viol_summ

    out: dict = {}
    for k, v in src.items():
        if k in _INSTANCE_HEADLINE:
            continue                       # placed explicitly below
        if k == "rubric":
            continue                       # placed after fps_schedule below
        if k == "violation_summ":
            continue                       # rebuilt + emitted below
        if k == "checklist_detail":
            continue                       # placed right before `rounds`
        if k == "fps_schedule":
            for hk in _INSTANCE_HEADLINE:
                if hk in src:
                    out[hk] = src[hk]
            out[k] = v
            if "rubric" in src:
                out["rubric"] = src["rubric"]
                _maybe_emit_viol(out)
            continue
        if k == "rounds":
            if "checklist_detail" in src:
                out["checklist_detail"] = src["checklist_detail"]
            out[k] = v
            continue
        out[k] = v
    for hk in _INSTANCE_HEADLINE:           # no fps_schedule -> append
        if hk in src and hk not in out:
            out[hk] = src[hk]
    if "rubric" in src and "rubric" not in out:
        out["rubric"] = src["rubric"]
        _maybe_emit_viol(out)
    elif "violation_summ" not in out and viol_summ is not None:
        # rubric absent but checklist present: still surface the summary
        # at the tail.
        out["violation_summ"] = viol_summ
    # If `rounds` was absent (e.g. an early-return trace), checklist_detail
    # still needs to be emitted somewhere — tail it on.
    if "checklist_detail" in src and "checklist_detail" not in out:
        out["checklist_detail"] = src["checklist_detail"]
    return out


def _per_task_score_breakdown(instances: list[dict],
                              task_name: str | None = None) -> dict:
    """Aggregate final_score / completeness_score / rubric_score /
    success_rate by level and overall. Levels are recovered from each
    instance id (pass `task_name` so non-`lvN` level tokens like 3x3 /
    4pcs split correctly).

    Each instance carries its own per-checklist `rubric_score` (arith
    mean over items); this function arithmetic-averages that across
    instances. Instances with a non-numeric / missing metric simply
    don't contribute to that metric's mean. `success_rate` counts
    success == 1 / True."""
    buckets: dict[str, dict] = {}

    def _bucket(name: str) -> dict:
        return buckets.setdefault(name, {
            "_rub_arith": [],
            "_compl": [], "_final": [], "_succ": [], "n": 0})

    for inst in instances:
        if not isinstance(inst, dict):
            continue
        lv = _level_from_id(inst.get("id", ""), task_name)
        for tgt in (_bucket(lv), _bucket("overall")):
            tgt["n"] += 1
            v = inst.get("rubric_score")
            if isinstance(v, (int, float)):
                tgt["_rub_arith"].append(float(v))
            cs = inst.get("completeness_score")
            if isinstance(cs, (int, float)):
                tgt["_compl"].append(float(cs))
            fs = inst.get("final_score")
            if isinstance(fs, (int, float)):
                tgt["_final"].append(float(fs))
            s = inst.get("success")
            if isinstance(s, bool):
                tgt["_succ"].append(1.0 if s else 0.0)
            elif isinstance(s, (int, float)):
                tgt["_succ"].append(1.0 if s else 0.0)

    def _mean(xs: list[float]):
        return round(sum(xs) / len(xs), 3) if xs else None

    out: dict = {}
    # overall first, then levels in natural order.
    for name in (["overall"] + sorted((b for b in buckets if b != "overall"),
                                      key=_natkey)):
        b = buckets[name]
        out[name] = {
            "n": b["n"],
            "final_score":       _mean(b["_final"]),
            "rubric_score":      _mean(b["_rub_arith"]),
            "completeness_score": _mean(b["_compl"]),
            "success_rate":      _mean(b["_succ"]),
        }
    return out


def _ablation_config_name(cfg: dict | None) -> str:
    """Short tag identifying the ablation config; '' for the full
    (default) method (adaptive fps + windowed batching), which keeps the
    plain `_judge_<task>.json` filename. Any other config gets a tag used
    as the `_abl_<tag>` filename suffix.

    NOTE: running BOTH passes (the default) is NOT part of the tag — the
    canonical `_judge_<task>.json` carries rubric + completeness + final
    together, and a default run can backfill completeness onto an existing
    rubric run in place. Running a SINGLE pass (`--no_completeness` =>
    rub_only, `--no_rubric` => comp_only) IS a variant and gets a tag so it
    sits in its own file instead of clobbering the canonical result."""
    if not cfg:
        return ""
    # Caller-supplied override wins (used by judge-model-swap ablations
    # whose differentiator isn't sampling/batching).
    if cfg.get("_suffix_override"):
        return str(cfg["_suffix_override"])
    sampling = cfg.get("sampling", "adaptive")
    batching = cfg.get("batching", "window")
    with_rubric = bool(cfg.get("rubric", True))
    with_completeness = bool(cfg.get("completeness", COMPLETENESS_DEFAULT))
    pass_tag = ""
    if with_rubric and not with_completeness:
        pass_tag = "rub_only"
    elif with_completeness and not with_rubric:
        pass_tag = "comp_only"
    if sampling == "adaptive" and batching == "window" and not pass_tag:
        return ""                               # the full method (default)
    parts: list[str] = []
    # sampling/batching pair only contributes when it deviates from the
    # default (adaptive + window); otherwise pass_tag alone names the file.
    if sampling == "even":
        parts.append(f"even_fps{cfg.get('even_fps')}")
        parts.append("allatonce" if batching == "all_at_once" else "window")
    elif batching == "all_at_once":
        parts.append("adaptive")
        parts.append("allatonce")
    if pass_tag:
        parts.append(pass_tag)
    return "_".join(parts)


def _parse_json_lenient(text: str, list_key: str | None = None) -> dict:
    """Parse a model reply into a dict, tolerating fenced / prefixed output.

    ALWAYS returns a dict: the judge prompts ask for a JSON object, but the
    model occasionally emits the bare top-level ARRAY instead (e.g. polish
    returning `[{index: 1, n_violations: 0}, ...]` rather than
    `{"checklist": [...]}`). Returning that list verbatim made every caller
    blow up on `.get` ('list' object has no attribute 'get'), which surfaced
    as an instance-level error that retries could not clear. When the reply
    is a top-level array, `list_key` (if given) says which field it was
    meant to fill, so the content is kept instead of discarded; without it
    the array is dropped and the caller takes its normal unparseable path.
    """
    def _coerce(obj):
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and list_key:
            return {list_key: obj}
        return None

    if not isinstance(text, str):
        return {}
    try:
        out = _coerce(json.loads(text))
        if out is not None:
            return out
    except json.JSONDecodeError:
        pass
    # Fall back to a {...} or [...] block embedded in prose. Candidates are
    # tried in order of where they START, so `... [{"index": 1}, ...] ...`
    # yields the whole array rather than the first object nested inside it
    # (which would silently truncate a checklist to its first entry).
    cands = [m for m in (re.search(p, text)
                         for p in (r"\{[\s\S]*\}", r"\[[\s\S]*\]")) if m]
    for m in sorted(cands, key=lambda m: m.start()):
        try:
            out = _coerce(json.loads(m.group(0)))
        except json.JSONDecodeError:
            continue
        if out is not None:
            return out
    return {}


def _coerce_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "yes", "y", "1", "pass", "passed", "success"}:
            return True
        if s in {"false", "no", "n", "0", "fail", "failed", "failure"}:
            return False
    return default


# ---------- per-instance judging ------------------------------------------


def _full_round_metas(fps: int, duration: float, native_fps: float
                      ) -> list[dict]:
    """Whole-clip uniform sampling at `fps`. Last sample is clamped to
    `duration - 1/native_fps` so ffmpeg's seek never lands past EOF."""
    safe_max = max(0.0, duration - (1.0 / native_fps if native_fps > 0
                                    else 0.04))
    n = int(round(duration * fps)) + 1
    seen: set[float] = set()
    metas: list[dict] = []
    for k in range(n):
        t = min(safe_max, k / fps)
        key = round(t, 6)
        if key in seen:
            continue
        seen.add(key)
        metas.append({
            "round_local_idx": len(metas), "time_s": t,
            "global_frame": int(round(t * native_fps)),
        })
    return metas


def _make_batches(metas: list[dict], batch_size: int, overlap: int
                  ) -> list[list[dict]]:
    """Slide a window of `batch_size` over `metas` with `overlap` shared
    frames between consecutive batches (stride = batch_size - overlap).
    The final batch may be shorter than `batch_size` when the list does
    not divide evenly."""
    if not metas:
        return []
    stride = max(1, batch_size - overlap)
    batches: list[list[dict]] = []
    i = 0
    n = len(metas)
    while i < n:
        j = min(i + batch_size, n)
        batches.append(metas[i:j])
        if j >= n:
            break
        i += stride
    return batches


def _merge_intervals(intervals: list[tuple[float, float]],
                     gap_tol: float = 1e-3) -> list[tuple[float, float]]:
    """Sort and merge overlapping or near-touching closed time intervals."""
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: x[0])
    merged: list[list[float]] = [list(sorted_iv[0])]
    for s, e in sorted_iv[1:]:
        if s <= merged[-1][1] + gap_tol:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _interval_metas(fps: int, intervals: list[tuple[float, float]],
                    duration: float, native_fps: float) -> list[dict]:
    """Sample at `fps` only inside the given (start, end) time intervals.
    Frames are sorted by time and round_local_idx numbered 0..N-1. Times
    are clamped to (duration - 1/native_fps) so ffmpeg seeks stay inside
    the clip."""
    safe_max = max(0.0, duration - (1.0 / native_fps if native_fps > 0
                                    else 0.04))
    step = 1.0 / fps
    seen: set[float] = set()
    times: list[float] = []
    for (a, b) in intervals:
        a = max(0.0, min(a, safe_max))
        b = max(0.0, min(b, safe_max))
        if b <= a:
            continue
        t = a
        while t < b - 1e-6:
            key = round(t, 6)
            if key not in seen:
                seen.add(key); times.append(t)
            t += step
        key = round(b, 6)
        if key not in seen:
            seen.add(key); times.append(b)
    times.sort()
    return [{
        "round_local_idx": k, "time_s": t,
        "global_frame": int(round(t * native_fps)),
    } for k, t in enumerate(times)]


def _zoom_y(window_size: int = AGENTIC_BATCH_SIZE) -> int:
    """Number of round-2 frames sampled BEFORE (and AFTER) a flagged
    round-1 frame. y is ceil(window_size/2) capped so 2*y + 1 <
    window_size (i.e. the zoom window stays shorter than one batch
    of the FULL METHOD).

    Always uses AGENTIC_BATCH_SIZE by default — NOT the runtime
    batch_size, which the `all_at_once` batching ablation inflates to
    100_000. Tying zoom_y to that would blow the round-2 interval out
    to ~minutes (clamped back to the whole clip) and silently mix two
    independent ablation dimensions."""
    initial = math.ceil(window_size / 2)
    cap = max(0, (window_size - 2) // 2)
    return min(initial, cap)


def _interval_groups(metas: list[dict], step_s: float
                     ) -> list[list[dict]]:
    """Split a flat metas list into contiguous groups where adjacent
    samples are at most `step_s + tolerance` apart in time. Used so a
    single batch never spans two disjoint zoom intervals."""
    if not metas:
        return []
    groups: list[list[dict]] = [[metas[0]]]
    tol = step_s + 0.05
    for m in metas[1:]:
        if m["time_s"] - groups[-1][-1]["time_s"] > tol:
            groups.append([m])
        else:
            groups[-1].append(m)
    return groups


_PRINT_LOCK = threading.Lock()


def _safe_print(*args, **kwargs):
    with _PRINT_LOCK:
        print(*args, **kwargs)


def _judge_one_batch(api_key: str, base_url: str, judge_model: str,
                     round_idx: int, fps: int,
                     b_idx: int, n_batches: int,
                     batch_metas: list[dict], round_n_frames: int,
                     rubric_text: str, task_name: str, level: str, iid: str,
                     input_image: Path | None, gt_image: Path | None,
                     rubric_overall: str = "",
                     rubric_checklist: list[str] | None = None,
                     gt_desc_text: str = ""
                     ) -> tuple[dict, list[dict]]:
    """One VLM call for one batch. Returns (batch_record, eval_records).
    Pure function: no shared mutable state, safe to call from parallel
    threads; the caller merges eval_records back in batch order. The
    batch returns per-frame evaluation comments and the frames it wants
    re-inspected at higher fps; it makes no success/verdict judgment."""
    user_msg = _batch_user_message(
        round_idx=round_idx, fps=fps,
        batch_idx=b_idx, n_batches=n_batches,
        batch_metas=batch_metas, round_n_frames=round_n_frames,
        rubric_text=rubric_text,
        task_name=task_name, level=level, iid=iid,
        input_image=input_image, gt_image=gt_image,
        rubric_overall=rubric_overall,
        rubric_checklist=rubric_checklist,
        gt_desc_text=gt_desc_text)
    messages = [{"role": "system", "content": AGENTIC_SYSTEM}, user_msg]
    batch_error: str | None = None
    try:
        response_text = _post_chat(api_key, base_url, judge_model,
                                   messages,
                                   response_format={"type": "json_object"})
    except Exception as e:
        msg = f"call failed: {e}"
        _safe_print(f"        [warn] batch {b_idx + 1} {msg}", flush=True)
        response_text = ""
        batch_error = str(msg)[:400]
    parsed = (_parse_json_lenient(response_text)
              if response_text else {})
    if not parsed:
        _safe_print(f"        [warn] batch {b_idx + 1} unparseable output",
                    flush=True)
        parsed = {}
        if batch_error is None:
            batch_error = "unparseable JSON output"

    # LLM now references frames by original-video global_frame numbers
    # (we used to give it batch-local indices). Build a global->local map
    # once so both `evaluations.frame_indices` and
    # `frames_to_inspect_closely` can be resolved back to the internal
    # batch-local index the rest of this function expects.
    gf_to_local = {m["global_frame"]: i for i, m in enumerate(batch_metas)}

    new_eval_records: list[dict] = []
    batch_evals_clean: list[dict] = []
    evs = parsed.get("evaluations") or []
    if isinstance(evs, list):
        for ev in evs:
            if not isinstance(ev, dict):
                continue
            raw_idxs = ev.get("frame_indices") or []
            if not isinstance(raw_idxs, list):
                raw_idxs = []
            valid = sorted({gf_to_local[gf] for gf in raw_idxs
                            if isinstance(gf, int) and gf in gf_to_local})
            comment = str(ev.get("comment", "")).strip()
            if not comment:
                continue
            rec = {
                "round": round_idx, "fps": fps,
                "batch_idx": b_idx,
                "batch_local_indices": valid,
                "round_local_indices":
                    [batch_metas[i]["round_local_idx"] for i in valid],
                "global_frames":
                    [batch_metas[i]["global_frame"] for i in valid],
                "time_s_list":
                    [batch_metas[i]["time_s"] for i in valid],
                "comment": comment,
            }
            new_eval_records.append(rec)
            batch_evals_clean.append({
                "batch_local_indices": valid,
                "round_local_indices": rec["round_local_indices"],
                "global_frames": rec["global_frames"],
                "time_s_list": rec["time_s_list"],
                "comment": comment,
            })

    # Parse frames_to_inspect_closely (round-1 only uses this; round-2
    # may emit it harmlessly; we record it either way). Like
    # frame_indices, the LLM now returns global_frame numbers.
    raw_zoom = parsed.get("frames_to_inspect_closely") or []
    zoom_local: list[int] = []
    if isinstance(raw_zoom, list):
        seen: set[int] = set()
        for gf in raw_zoom:
            if isinstance(gf, int) and gf in gf_to_local:
                local_i = gf_to_local[gf]
                if local_i in seen:
                    continue
                seen.add(local_i); zoom_local.append(local_i)
    cap = _zoom_max_per_batch(len(batch_metas))
    zoom_local = zoom_local[:cap]
    zoom_round_local = [batch_metas[i]["round_local_idx"]
                        for i in zoom_local]
    zoom_global = [batch_metas[i]["global_frame"] for i in zoom_local]
    zoom_time = [batch_metas[i]["time_s"] for i in zoom_local]

    batch_record = {
        "batch_idx": b_idx,
        "n_frames": len(batch_metas),
        "round_local_range": [
            batch_metas[0]["round_local_idx"],
            batch_metas[-1]["round_local_idx"]],
        "global_frame_range": [
            batch_metas[0]["global_frame"],
            batch_metas[-1]["global_frame"]],
        "time_s_range": [
            batch_metas[0]["time_s"],
            batch_metas[-1]["time_s"]],
        "frames": [{
            "batch_local_idx": k,
            "round_local_idx": m["round_local_idx"],
            "global_frame": m["global_frame"],
            "time_s": m["time_s"],
        } for k, m in enumerate(batch_metas)],
        "evaluations": batch_evals_clean,
        "frames_to_inspect_closely": {
            "batch_local": zoom_local,
            "round_local": zoom_round_local,
            "global_frames": zoom_global,
            "time_s": zoom_time,
        },
    }
    if batch_error is not None:
        batch_record["error"] = batch_error
    return batch_record, new_eval_records


def _run_batches_parallel(batches: list[list[dict]], num_workers: int,
                          api_key: str, base_url: str, judge_model: str,
                          round_idx: int, fps: int,
                          round_n_frames: int,
                          rubric_text: str, task_name: str, level: str,
                          iid: str,
                          input_image: Path | None,
                          gt_image: Path | None,
                          rubric_overall: str = "",
                          rubric_checklist: list[str] | None = None,
                          gt_desc_text: str = ""
                          ) -> tuple[list[dict], list[dict]]:
    """Run all batches of one round; up to `num_workers` in flight.
    Returns (batch_records_in_order, flat_eval_records_in_order)."""
    n = len(batches)
    results: list[tuple[dict, list[dict]] | None] = [None] * n
    if num_workers <= 1 or n <= 1:
        for b_idx, batch_metas in enumerate(batches):
            _safe_print(
                f"      batch {b_idx + 1}/{n}: "
                f"round_local {batch_metas[0]['round_local_idx']}.."
                f"{batch_metas[-1]['round_local_idx']}  "
                f"global {batch_metas[0]['global_frame']}-"
                f"{batch_metas[-1]['global_frame']}  "
                f"calling {judge_model}...", flush=True)
            results[b_idx] = _judge_one_batch(
                api_key, base_url, judge_model,
                round_idx=round_idx, fps=fps,
                b_idx=b_idx, n_batches=n,
                batch_metas=batch_metas, round_n_frames=round_n_frames,
                rubric_text=rubric_text,
                task_name=task_name, level=level, iid=iid,
                input_image=input_image, gt_image=gt_image,
                rubric_overall=rubric_overall,
                rubric_checklist=rubric_checklist,
                gt_desc_text=gt_desc_text)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            future_to_idx = {}
            for b_idx, batch_metas in enumerate(batches):
                _safe_print(
                    f"      submit batch {b_idx + 1}/{n}: "
                    f"round_local {batch_metas[0]['round_local_idx']}.."
                    f"{batch_metas[-1]['round_local_idx']}  "
                    f"global {batch_metas[0]['global_frame']}-"
                    f"{batch_metas[-1]['global_frame']}",
                    flush=True)
                fut = ex.submit(
                    _judge_one_batch,
                    api_key, base_url, judge_model,
                    round_idx, fps, b_idx, n, batch_metas,
                    round_n_frames, rubric_text,
                    task_name, level, iid,
                    input_image, gt_image,
                    rubric_overall, rubric_checklist,
                    gt_desc_text)
                future_to_idx[fut] = b_idx
            for fut in as_completed(future_to_idx):
                b_idx = future_to_idx[fut]
                results[b_idx] = fut.result()
                _safe_print(f"      done batch {b_idx + 1}/{n}",
                            flush=True)

    batch_records: list[dict] = []
    eval_records: list[dict] = []
    for r in results:
        if r is None:
            continue
        br, evs = r
        batch_records.append(br)
        eval_records.extend(evs)
    return batch_records, eval_records


_COMPLETENESS_ROOT_OVERRIDE: Path | None = None


def find_completeness_file(data_task_dir: Path | None,
                           task_name: str | None = None) -> Path | None:
    """Locate <task_dir>/_completeness_judge.txt. --completeness_root, if
    set, wins for per-style tier descriptions (e.g. line-art = blue ball
    instead of toy car)."""
    if _COMPLETENESS_ROOT_OVERRIDE is not None and task_name:
        for d in (_COMPLETENESS_ROOT_OVERRIDE / task_name,
                  _COMPLETENESS_ROOT_OVERRIDE / f"v_{task_name}"):
            for cand in (d / "judge_spec" / COMPLETENESS_FILENAME,
                         d / COMPLETENESS_FILENAME):
                if cand.is_file():
                    return cand
    if data_task_dir is None:
        return None
    for cand in (data_task_dir / "judge_spec" / COMPLETENESS_FILENAME,
                 data_task_dir / COMPLETENESS_FILENAME):  # new, then legacy
        if cand.is_file():
            return cand
    return None


def judge_completeness(api_key: str, base_url: str, judge_model: str,
                       task_name: str, level: str, iid: str,
                       video_path: Path, completeness_text: str,
                       input_image: Path | None, gt_image: Path | None,
                       work_dir: Path,
                       gt_desc_text: str = "",
                       fps: int = COMPLETENESS_FPS) -> dict:
    """Single-call completeness score: sample the whole clip at `fps`, feed
    every frame at once, return a tier 0/1/2 normalized to 0/0.5/1."""
    base = {"completeness_fps": fps, "completeness_tier": None,
            "completeness_score": None, "completeness_reasoning": "",
            "completeness_n_frames": 0, "completeness_n_llm_calls": 0}
    duration, native_fps = probe_video(video_path)
    metas = _full_round_metas(fps, duration, native_fps)
    if not metas:
        base["completeness_reasoning"] = "no frames could be sampled"
        return base
    cdir = work_dir / "completeness"
    recs = extract_frame_records_at_times(
        video_path, [m["time_s"] for m in metas], cdir, "c")
    metas = _attach_extracted_frames(metas, recs)
    if not metas:
        base["completeness_reasoning"] = "no frames could be extracted"
        return base
    messages = _completeness_messages(task_name, level, iid, metas,
                                      completeness_text, input_image, gt_image,
                                      gt_desc_text=gt_desc_text)
    text = _post_chat(api_key, base_url, judge_model, messages,
                      response_format={"type": "json_object"})
    base["completeness_n_llm_calls"] = 1
    base["completeness_n_frames"] = len(metas)
    parsed = _parse_json_lenient(text) or {}
    tier = parsed.get("tier")
    try:
        tier = int(tier)
    except (TypeError, ValueError):
        tier = None
    if tier not in COMPLETENESS_TIER_TO_SCORE:
        tier = None
    base["completeness_tier"] = tier
    base["completeness_score"] = (COMPLETENESS_TIER_TO_SCORE[tier]
                                  if tier is not None else None)
    base["completeness_reasoning"] = str(parsed.get("reasoning") or "").strip()
    return base


def agentic_judge_video(api_key: str, base_url: str, judge_model: str,
                        task_name: str, level: str, iid: str,
                        video_path: Path, rubric_text: str,
                        input_image: Path | None, gt_image: Path | None,
                        work_dir: Path,
                        fps_schedule: tuple[int, ...] = AGENTIC_FPS_SCHEDULE,
                        batch_size: int = AGENTIC_BATCH_SIZE,
                        batch_overlap: int = AGENTIC_BATCH_OVERLAP,
                        num_workers: int = AGENTIC_NUM_WORKERS,
                        rubric_checklist: list[str] | None = None,
                        rubric_overall: str = "",
                        gt_desc_text: str = "",
                        rubric_items: list[str] | None = None
                        ) -> dict:
    # `rubric_items` is the raw rubric list persisted in the trace;
    # `rubric_text` is the joined-text fallback used inside prompts. Caller
    # passes both so persistence and prompting can decouple cleanly.
    rubric_items = list(rubric_items) if rubric_items else []
    duration, native_fps = probe_video(video_path)
    rounds: list[dict] = []
    eval_records: list[dict] = []
    n_llm_calls = 0
    t0 = time.time()

    # ---------- Round 1: full video at fps_schedule[0] -----------------
    fps_r1 = fps_schedule[0]
    metas = _full_round_metas(fps_r1, duration, native_fps)
    if not metas:
        return {
            "video": _relpath_from_project(video_path),
            "task": task_name, "level": level, "iid": iid,
            "duration_s": duration, "native_fps": native_fps,
            "rubric": rubric_items,
            "fps_schedule": list(fps_schedule),
            "batch_size": batch_size, "batch_overlap": batch_overlap,
            "num_workers": num_workers,
            "n_llm_calls": n_llm_calls, "wall_time_s": 0.0,
            "rounds": [], "frame_evaluations": {},
            "checklist_detail": None,
            "rubric_score": 0.0,
            "success": False,
            "overall_reasoning": "no frames could be sampled",
        }
    round1_dir = work_dir / "round1"
    frame_records = extract_frame_records_at_times(
        video_path, [m["time_s"] for m in metas], round1_dir, "r1")
    metas = _attach_extracted_frames(metas, frame_records)
    if not metas:
        return {
            "video": _relpath_from_project(video_path),
            "task": task_name, "level": level, "iid": iid,
            "duration_s": duration, "native_fps": native_fps,
            "rubric": rubric_items,
            "fps_schedule": list(fps_schedule),
            "batch_size": batch_size, "batch_overlap": batch_overlap,
            "num_workers": num_workers,
            "n_llm_calls": n_llm_calls,
            "wall_time_s": time.time() - t0,
            "rounds": [], "frame_evaluations": {},
            "checklist_detail": None,
            "rubric_score": 0.0,
            "success": False,
            "overall_reasoning": "no frames could be extracted",
        }

    batches = _make_batches(metas, batch_size, batch_overlap)
    round1_record: dict = {
        "round": 1, "fps": fps_r1,
        "n_frames": len(metas),
        "n_batches": len(batches),
        "batch_size": batch_size,
        "batch_overlap": batch_overlap,
        "num_workers": num_workers,
        "frames": [{k: v for k, v in m.items() if k != "path"}
                   for m in metas],
        "batches": [],
    }
    _safe_print(f"    round 1: fps={fps_r1}, {len(metas)} frames, "
                f"{len(batches)} batches "
                f"(parallel workers={num_workers})",
                flush=True)
    batch_records, new_evals = _run_batches_parallel(
        batches, num_workers,
        api_key, base_url, judge_model,
        round_idx=1, fps=fps_r1,
        round_n_frames=len(metas),
        rubric_text=rubric_text,
        task_name=task_name, level=level, iid=iid,
        input_image=input_image, gt_image=gt_image,
        rubric_overall=rubric_overall,
        rubric_checklist=rubric_checklist,
        gt_desc_text=gt_desc_text)
    round1_record["batches"] = batch_records
    eval_records.extend(new_evals)
    n_llm_calls += len(batches)

    # ---------- Build round-2 zoom intervals -------------------------
    # Round-2 zoom is driven solely by LLM-flagged frames: each round-1
    # batch may have asked us to "inspect closely" some round-1-local
    # frames (capped at ceil(batch_size/2) per batch). For every flagged
    # frame at time t_k we sample the round-2 fps grid over
    # [t_k - y/fps_r2, t_k + y/fps_r2] (2y+1 samples around it).
    # Overlapping windows naturally merge when close.
    flagged_round_local: list[int] = []
    for br in batch_records:
        flagged_round_local.extend(br["frames_to_inspect_closely"]
                                   ["round_local"])
    flagged_round_local = sorted(set(flagged_round_local))
    fps_r2 = fps_schedule[1] if len(fps_schedule) >= 2 else 0
    # zoom_y is intentionally derived from the METHOD's window size, NOT
    # `batch_size` — see _zoom_y. The all_at_once ablation inflates
    # batch_size but must NOT inflate the zoom interval.
    y = _zoom_y() if fps_r2 else 0
    zoom_step = (1.0 / fps_r2) if fps_r2 else 0.0
    intervals_raw: list[tuple[float, float]] = []
    flagged_log: list[dict] = []
    for k in flagged_round_local:
        t_k = metas[k]["time_s"]
        a = max(0.0, t_k - y * zoom_step)
        b = t_k + y * zoom_step
        intervals_raw.append((a, b))
        flagged_log.append({
            "round_local_idx": k,
            "global_frame": metas[k]["global_frame"],
            "time_s": t_k,
            "zoom_window_t_s": [a, b],
        })
    intervals_merged = _merge_intervals(intervals_raw)
    round1_record["zoom_y"] = y
    # zoom_max_per_batch also derives from the method's window size, not
    # the inflated all_at_once batch_size.
    round1_record["zoom_max_per_batch"] = _zoom_max_per_batch(
        AGENTIC_BATCH_SIZE)
    round1_record["flagged_frames"] = flagged_log
    round1_record["zoom_intervals_raw"] = [
        {"t_start": a, "t_end": b} for (a, b) in intervals_raw]
    round1_record["zoom_intervals_merged"] = [
        {"t_start": a, "t_end": b} for (a, b) in intervals_merged]
    rounds.append(round1_record)

    # ---------- Round 2: zoom at fps_r2 over the merged windows -------
    round2_record: dict | None = None
    if intervals_merged and fps_r2 > 0:
        metas_r2 = _interval_metas(fps_r2, intervals_merged,
                                   duration, native_fps)
        if metas_r2:
            round2_dir = work_dir / "round2"
            records_r2 = extract_frame_records_at_times(
                video_path, [m["time_s"] for m in metas_r2],
                round2_dir, "r2")
            metas_r2 = _attach_extracted_frames(metas_r2, records_r2)
            groups = _interval_groups(metas_r2, 1.0 / fps_r2)
            all_batches_r2: list[list[dict]] = []
            for g in groups:
                for batch in _make_batches(g, batch_size, batch_overlap):
                    all_batches_r2.append(batch)

            round2_record = {
                "round": 2, "fps": fps_r2,
                "n_frames": len(metas_r2),
                "n_batches": len(all_batches_r2),
                "batch_size": batch_size,
                "batch_overlap": batch_overlap,
                "num_workers": num_workers,
                "intervals_merged": [{"t_start": a, "t_end": b}
                                     for (a, b) in intervals_merged],
                "frames": [{k: v for k, v in m.items() if k != "path"}
                           for m in metas_r2],
                "batches": [],
            }
            _safe_print(f"    round 2: fps={fps_r2}, "
                        f"{len(intervals_merged)} merged intervals, "
                        f"{len(metas_r2)} frames, "
                        f"{len(all_batches_r2)} batches "
                        f"(parallel workers={num_workers})", flush=True)
            batch_records_r2, new_evals_r2 = _run_batches_parallel(
                all_batches_r2, num_workers,
                api_key, base_url, judge_model,
                round_idx=2, fps=fps_r2,
                round_n_frames=len(metas_r2),
                rubric_text=rubric_text,
                task_name=task_name, level=level, iid=iid,
                input_image=input_image, gt_image=gt_image,
                rubric_overall=rubric_overall,
                rubric_checklist=rubric_checklist,
                gt_desc_text=gt_desc_text)
            round2_record["batches"] = batch_records_r2
            eval_records.extend(new_evals_r2)
            n_llm_calls += len(all_batches_r2)
            rounds.append(round2_record)
        else:
            _safe_print("    round 2: no frames sampled in zoom "
                        "intervals, skipping", flush=True)
    else:
        _safe_print("    round 2: skipped (no LLM-flagged frames in "
                    "round 1)", flush=True)

    # ---------- Aggregate evaluations keyed by batch frame range ------
    # Each LLM eval covers a contiguous span of global frames (the batch's
    # span, or a sub-span the LLM grouped). We key by "<gf_start>-<gf_end>"
    # so the JSON shows one entry per batch span instead of repeating the
    # comment under every individual frame index. Boundary frames are no
    # longer double-counted across overlapping batches because each batch
    # contributes exactly one entry under its own range key.
    frame_evals: dict[str, list[str]] = {}
    for ev in eval_records:
        gfs = ev["global_frames"]
        ts = ev["time_s_list"]
        if not gfs:
            continue
        key = f"{gfs[0]}-{gfs[-1]}"
        if len(ts) >= 2 and ts[0] != ts[-1]:
            t_part = f"t={ts[0]:.2f}-{ts[-1]:.2f}s"
        else:
            t_part = f"t={ts[0]:.2f}s"
        line = (f"r{ev['round']}b{ev['batch_idx']} {t_part}: "
                f"{ev['comment']}")
        frame_evals.setdefault(key, []).append(line)
    # Sort by (start, end) numerically.
    def _range_sort_key(k: str):
        a, _, b = k.partition("-")
        return (int(a), int(b))
    frame_evals = {k: frame_evals[k]
                   for k in sorted(frame_evals, key=_range_sort_key)}

    # ---------- Polish (no images) ------------------------------------
    batch_summaries: list[dict] = []
    for r in rounds:
        for b in r["batches"]:
            batch_summaries.append({
                "round": r["round"], "fps": r["fps"],
                "batch_idx": b["batch_idx"],
                "global_frame_range": b["global_frame_range"],
                "time_s_range": b["time_s_range"],
            })

    polish_msgs = _polish_messages(rubric_text, task_name, level, iid,
                                   duration, native_fps,
                                   frame_evals, batch_summaries,
                                   rubric_checklist=rubric_checklist,
                                   rubric_overall=rubric_overall,
                                   gt_desc_text=gt_desc_text)
    print(f"    polish step: calling {judge_model} (no images)...",
          flush=True)
    polish_text = _post_chat(api_key, base_url, judge_model, polish_msgs,
                             response_format={"type": "json_object"})
    n_llm_calls += 1
    polish = _parse_json_lenient(polish_text, list_key="checklist")
    if not polish:
        polish = {"overall_reasoning": "polish step produced unparseable JSON",
                  "_raw": polish_text[:600]}

    checklist_score = _build_checklist_detail(
        rubric_checklist, polish.get("checklist"))
    if checklist_score is not None:
        polish["checklist"] = checklist_score["items"]

    wall_time_s = time.time() - t0
    # Instance rubric_score = arith mean of per-item 1/(x+1) scores, in [0, 1].
    rubric_score_arith = float((checklist_score or {}).get(
        "rubric_score_arith", 0.0))
    # Polish no longer returns a success verdict. The rubric pass only
    # contributes whether the rubric is perfect; the final per-instance
    # success (rubric == 1 AND completeness == 1) is decided in
    # judge_one_real once the completeness pass has run. Here we expose a
    # provisional rubric-only success that judge_one_real overwrites.
    success_val: bool | None = bool(rubric_score_arith >= 1.0 - 1e-9)
    overall_reasoning_val = str(polish.get("overall_reasoning", "")).strip()
    # Any batch-call failure or unparseable batch output means polish
    # built its counts on incomplete evidence. Surface it as an
    # instance ERROR (success=None + `error` field) rather than letting
    # the (possibly bogus) counts leak through. We still keep
    # the full trace for debugging.
    batch_errors: list[dict] = []
    for r in rounds:
        for b in r.get("batches", []):
            if isinstance(b, dict) and b.get("error"):
                batch_errors.append({
                    "round": r.get("round"),
                    "batch_idx": b.get("batch_idx"),
                    "error": b["error"],
                })
    instance_error_msg: str | None = None
    if batch_errors:
        n_total_batches = sum(len(r.get("batches", []) or [])
                              for r in rounds)
        first = batch_errors[0]
        instance_error_msg = (
            f"{len(batch_errors)}/{n_total_batches} batch(es) failed "
            f"(e.g. round {first['round']} batch {first['batch_idx']}: "
            f"{first['error']!s:.200}); verdict unreliable")
        success_val = None
        overall_reasoning_val = (
            instance_error_msg + " | polish reasoning: "
            + overall_reasoning_val) if overall_reasoning_val \
            else instance_error_msg
    return {
        "video": _relpath_from_project(video_path),
        "task": task_name, "level": level, "iid": iid,
        "duration_s": duration,
        "native_fps": native_fps,
        "rubric": rubric_items,
        "fps_schedule": list(fps_schedule),
        "batch_size": batch_size,
        "batch_overlap": batch_overlap,
        "num_workers": num_workers,
        # Both come from the method's window size, not the (possibly
        # ablation-inflated) runtime batch_size — see _zoom_y.
        "zoom_y": _zoom_y(),
        "zoom_max_per_batch": _zoom_max_per_batch(AGENTIC_BATCH_SIZE),
        "n_llm_calls": n_llm_calls,
        "wall_time_s": wall_time_s,
        "rounds": rounds,
        "frame_evaluations": frame_evals,
        # ----- summary tail: detail block, then headline numbers -----
        "checklist_detail": checklist_score,
        "rubric_score":      round(rubric_score_arith, 4),
        "success": success_val,
        "overall_reasoning": overall_reasoning_val,
        **({"error": instance_error_msg,
            "batch_errors": batch_errors}
           if instance_error_msg else {}),
    }


def judge_one_real(api_key: str, base_url: str, judge_model: str,
                   task_name: str, level: str, iid: str, output_path: Path,
                   rubric_entry: dict, input_image: Path | None,
                   gt_image: Path | None,
                   num_workers: int = AGENTIC_NUM_WORKERS,
                   ablation_cfg: dict | None = None,
                   data_task_dir: Path | None = None) -> dict:
    if not output_path.is_file():
        reason = f"output file missing: {output_path.name}"
        return {"success": False,
                "reasoning": reason,
                "overall_reasoning": reason}
    kind = guess_kind(output_path)
    if kind == "unknown":
        reason = f"unknown output type: {output_path.suffix}"
        return {"success": False,
                "reasoning": reason,
                "overall_reasoning": reason}

    rubric_text = rubric_text_for(rubric_entry, task_name)
    rubric_checklist = extract_rubric_checklist(rubric_entry)
    rubric_overall = extract_rubric_overall(rubric_entry)
    rubric_items = rubric_items_for_persistence(rubric_entry)
    if not rubric_overall:
        # Fallback for tasks whose rubric does not yet have an overall
        # description as item 0; feed the joined rubric text instead.
        rubric_overall = rubric_text

    # GT signal routing (single source of truth in vlm_judge_utils):
    # tasks listed in TASKS_NEED_GT_IMAGE get gt_image fed into the
    # prompt; tasks in TASKS_NEED_GT_DESC get gt_desc_text fed. Tasks
    # in neither set are judged on the rubric checklist alone — we
    # explicitly drop the gt_image even if find_gt_image returned one
    # so the prompt geometry matches across runs.
    if data_task_dir is None:
        data_task_dir = find_data_task_dir(task_name)
    _sig = task_gt_signals(task_name)
    gt_image_use: Path | None = gt_image if _sig["image"] else None
    gt_desc_text: str = (gt_desc_for(task_name, level, iid, data_task_dir)
                         if _sig["desc"] else "")

    if kind != "video":
        # Image-output models are judged by vlm_judge_img.py — that script
        # owns the single-still-image judging path (input + gt + gt-json +
        # gen, binary success). vlm_judge.py is video-only.
        reason = (f"vlm_judge.py is video-only; image outputs "
                  f"({output_path.suffix}) must be judged via "
                  f"vlm_judge_img.py")
        return {"success": False,
                "reasoning": reason,
                "overall_reasoning": reason,
                "error": reason}

    # Video: agentic (or an ablation variant). Caller (judge_model)
    # aggregates per-task traces into one JSON; we just hand back the
    # full trace dict alongside the verdict summary.
    cfg = ablation_cfg or {}
    work_dir = Path(tempfile.mkdtemp(
        prefix=f"vlm_agentic_{task_name}_{level}_{iid}_"))
    try:
        # Which passes to run. `--no_rubric` => completeness-only;
        # `--no_completeness` => rubric-only. main() forbids turning both
        # off. completeness-only forces completeness on regardless of the
        # completeness flag (otherwise nothing would be judged).
        with_rubric = bool(cfg.get("rubric", True))
        with_completeness = (bool(cfg.get("completeness", COMPLETENESS_DEFAULT))
                             or not with_rubric)
        if with_rubric:
            # Sampling dimension: 'adaptive' keeps the round1 + round2-zoom
            # schedule; 'even' uses a single fixed fps over the whole clip
            # (a 1-element fps_schedule naturally skips round 2).
            sampling = cfg.get("sampling", "adaptive")
            if sampling == "even":
                even_fps = int(cfg.get("even_fps") or AGENTIC_FPS_SCHEDULE[0])
                fps_schedule: tuple[int, ...] = (even_fps,)
            else:
                fps_schedule = AGENTIC_FPS_SCHEDULE
            # Batching dimension: 'window' = 10-frame window w/ 1-frame
            # overlap; 'all_at_once' = one LLM call sees every frame.
            if cfg.get("batching") == "all_at_once":
                batch_size = ABLATION_ALL_AT_ONCE_BATCH
                batch_overlap = 0
            else:
                batch_size = AGENTIC_BATCH_SIZE
                batch_overlap = AGENTIC_BATCH_OVERLAP
            agentic = agentic_judge_video(
                api_key, base_url, judge_model,
                task_name, level, iid, output_path,
                rubric_text, input_image, gt_image_use, work_dir,
                fps_schedule=fps_schedule,
                batch_size=batch_size, batch_overlap=batch_overlap,
                num_workers=num_workers,
                rubric_checklist=rubric_checklist,
                rubric_overall=rubric_overall,
                gt_desc_text=gt_desc_text,
                rubric_items=rubric_items)
        else:
            # Completeness-only: skip the (expensive) rubric agentic loop.
            # Build a minimal trace so the completeness result still
            # persists with the usual shape; rubric_score stays None.
            duration, native_fps = probe_video(output_path)
            agentic = {
                "video": _relpath_from_project(output_path),
                "task": task_name, "level": level, "iid": iid,
                "duration_s": duration, "native_fps": native_fps,
                "rubric": rubric_items,
                "n_llm_calls": 0, "wall_time_s": 0.0,
                "rounds": [], "frame_evaluations": {},
                "checklist_detail": None,
                "rubric_score": None,
                "success": None,
                "overall_reasoning": "",
            }

        # ---------- Completeness pass (independent of rubric) ----------
        # One low-fps all-frames call scoring how far the goal was reached
        # (tier 0/1/2 -> 0/0.5/1). final_score = rubric_score (arith
        # aggregation) * completeness_score. Injected onto the
        # agentic trace so it persists in the per-task JSON + scores.
        cfile = (find_completeness_file(data_task_dir, task_name)
                 if with_completeness else None)
        if cfile is not None:
            # Completeness feeds in WHATEVER GT the task has — gt_image and/or
            # gt_desc — independent of the rubric path's gt routing signals.
            compl_gt_image = (gt_image if (gt_image
                              and Path(gt_image).is_file()) else None)
            compl_gt_desc = gt_desc_for(task_name, level, iid,
                                        data_task_dir) or ""
            try:
                ctext = cfile.read_text(encoding="utf-8-sig").strip()
                compl = judge_completeness(
                    api_key, base_url, judge_model,
                    task_name, level, iid, output_path, ctext,
                    input_image, compl_gt_image, work_dir,
                    gt_desc_text=compl_gt_desc)
            except Exception as ce:
                compl = {"completeness_tier": None, "completeness_score": None,
                         "completeness_reasoning": f"completeness error: {ce}",
                         "completeness_n_frames": 0,
                         "completeness_n_llm_calls": 0,
                         "completeness_fps": COMPLETENESS_FPS}
            for k, v in compl.items():
                agentic[k] = v
            rs = agentic.get("rubric_score")
            cs = compl.get("completeness_score")
            if isinstance(rs, (int, float)) and isinstance(cs, (int, float)):
                agentic["final_score"] = round(float(rs) * float(cs), 4)
            else:
                agentic["final_score"] = None
            agentic["n_llm_calls"] = (agentic.get("n_llm_calls", 0)
                                      + compl.get("completeness_n_llm_calls", 0))

        # ---------- Final per-instance success -------------------------
        # An instance succeeds ONLY when every requested pass is perfect:
        # completeness_score == 1 AND/OR rubric_score == 1. There is no
        # LLM "success" verdict anymore. Batch-call failures (agentic
        # `error`) keep success = None so they count as errors.
        #   - both passes:        rubric == 1 AND completeness == 1
        #   - rubric-only:        rubric == 1
        #   - completeness-only:  completeness == 1
        rub = agentic.get("rubric_score")
        cs = agentic.get("completeness_score")
        rub_ok = isinstance(rub, (int, float)) and rub >= 1.0 - 1e-9
        comp_ok = isinstance(cs, (int, float)) and cs >= 1.0 - 1e-9
        if not with_rubric and cs is None:
            # completeness-only but the completeness pass produced nothing
            # (no _completeness_judge.txt / call failed): mark as error.
            agentic["error"] = (agentic.get("error")
                                or "completeness-only run but completeness "
                                   "did not produce a score")
        if agentic.get("error"):
            agentic["success"] = None
        elif not with_rubric:
            agentic["success"] = bool(comp_ok)
        elif cs is None:
            agentic["success"] = bool(rub_ok)
        else:
            agentic["success"] = bool(rub_ok and comp_ok)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    return {
        "success": agentic["success"],
        "overall_reasoning": agentic.get("overall_reasoning", ""),
        # rubric_score = arith mean of per-item 1/(x+1) scores, in [0,1]
        # (set in agentic_judge_video).
        "rubric_score":      agentic.get("rubric_score", 0.0),
        "completeness_tier":  agentic.get("completeness_tier"),
        "completeness_score": agentic.get("completeness_score"),
        "completeness_reasoning": agentic.get("completeness_reasoning", ""),
        "final_score":       agentic.get("final_score"),
        "checklist_detail":  agentic.get("checklist_detail"),
        "n_rounds": len(agentic["rounds"]),
        # Surfaces batch-call failures: instance must be counted as an
        # error, not silently coerced to success=False.
        "error": agentic.get("error"),
        "agentic": agentic,
    }


# ---------- main driver ---------------------------------------------------

def _flush_per_task_json(per_task_path: Path, task_name: str,
                        model_dir_name: str, judge_model_name: str,
                        cfg_name: str,
                        new_traces: list[dict],
                        existing: dict[str, dict]) -> None:
    """Atomically write the per-task judge json after each instance.

    Mutates `existing` in place: for every trace in `new_traces`, builds the
    instance key and overwrites that entry. Then sorts, shapes, and writes
    the full blob to `per_task_path`. Safe to call after every instance —
    use case is incremental persistence so a killed run resumes cleanly."""
    for tr in new_traces:
        key = (f"{tr['task']}_{tr['iid']}" if tr["level"] == "no_lv"
               else f"{tr['task']}_{tr['level']}_{tr['iid']}")
        existing[key] = {"id": key, **_compact_agentic_trace(tr)}
    if not existing:
        return
    sorted_instances = [_shape_instance(existing[k])
                        for k in sorted(existing.keys(), key=_natkey)]
    total_calls = sum(int(inst.get("n_llm_calls") or 0)
                      for inst in sorted_instances)
    total_wall = sum(float(inst.get("wall_time_s") or 0.0)
                     for inst in sorted_instances)
    blob = {
        "task": task_name,
        "model_dir": model_dir_name,
        "judge_model": judge_model_name,
        "ablation_config": cfg_name or "full_method",
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_instances": len(sorted_instances),
        "scores": _per_task_score_breakdown(sorted_instances, task_name),
        "total_n_llm_calls": total_calls,
        "total_wall_time_s": round(total_wall, 2),
        "avg_n_llm_calls_per_instance":
            round(total_calls / len(sorted_instances), 2)
            if sorted_instances else 0,
        "avg_wall_time_s_per_instance":
            round(total_wall / len(sorted_instances), 2)
            if sorted_instances else 0.0,
        "instances": sorted_instances,
    }
    per_task_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = per_task_path.with_suffix(per_task_path.suffix + ".tmp")
    tmp.write_text(json.dumps(blob, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    tmp.replace(per_task_path)


def _run_completeness_only(api_key, base_url, judge_model: str,
                           task_name: str, level: str, iid: str,
                           output_path: Path,
                           input_image: Path | None, gt_image: Path | None,
                           data_task_dir: Path | None,
                           rubric_score) -> dict | None:
    """Run ONLY the completeness pass for an instance whose rubric verdict is
    already persisted (resume backfill). Returns the completeness_* fields +
    recomputed final_score to merge into the cached instance, or None when
    completeness can't run (no _completeness_judge.txt, or missing video)."""
    cfile = find_completeness_file(data_task_dir, task_name)
    if cfile is None or not output_path.is_file():
        return None
    compl_gt_image = (gt_image if (gt_image and Path(gt_image).is_file())
                      else None)
    compl_gt_desc = gt_desc_for(task_name, level, iid, data_task_dir) or ""
    work_dir = Path(tempfile.mkdtemp(
        prefix=f"vlm_compl_{task_name}_{level}_{iid}_"))
    try:
        ctext = cfile.read_text(encoding="utf-8-sig").strip()
        compl = judge_completeness(api_key, base_url, judge_model,
                                   task_name, level, iid, output_path, ctext,
                                   input_image, compl_gt_image, work_dir,
                                   gt_desc_text=compl_gt_desc)
    except Exception as ce:
        compl = {"completeness_tier": None, "completeness_score": None,
                 "completeness_reasoning": f"completeness error: {ce}",
                 "completeness_n_frames": 0, "completeness_n_llm_calls": 0,
                 "completeness_fps": COMPLETENESS_FPS}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    out = dict(compl)
    cs = compl.get("completeness_score")
    if isinstance(rubric_score, (int, float)) and isinstance(cs, (int, float)):
        out["final_score"] = round(float(rubric_score) * float(cs), 4)
    else:
        out["final_score"] = None
    return out


def judge_model(model_dir_name: str, judge_model_name: str,
                tasks_filter: set[str] | None,
                max_per_task: int | None,
                num_workers: int = AGENTIC_NUM_WORKERS,
                ablation_cfg: dict | None = None) -> dict:
    model_dir = EVAL_ROOT / model_dir_name
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    api_key, base_url, judge_model_name = load_judge_credentials(
        judge_model_name)

    tasks_in_dir: list[str] = []
    if model_dir.is_dir():
        tasks_in_dir = [d.name for d in sorted(model_dir.iterdir(),
                                                key=lambda p: _natkey(p.name))
                        if d.is_dir() and not d.name.startswith(("_", "."))]
    if tasks_filter:
        tasks_in_dir = [t for t in tasks_in_dir if t in tasks_filter]

    tasks_summary: dict = {}
    task_details: dict = {}
    overall_total = overall_succ = overall_fail = overall_err = 0

    for task_name in tasks_in_dir:
        rubric_map = load_rubric(task_name)
        data_task_dir = find_data_task_dir(task_name)
        outputs = collect_outputs(model_dir, model_dir_name, task_name,
                                  data_task_dir)
        if max_per_task is not None:
            outputs = outputs[:max_per_task]
        instances = []
        agentic_traces: list[dict] = []
        n_backfilled = 0

        # ----- resume: load existing per-task json once per task ----------
        # The persisted file is the source of truth for already-judged
        # instances; we'll re-judge anything with success==None (errored
        # earlier) and skip anything with success in (0, 1).
        task_subdir = model_dir / task_name
        cfg_name = _ablation_config_name(ablation_cfg)
        fname = (f"_judge_{task_name}_{model_dir_name}.json"
                 if not cfg_name
                 else f"_judge_{task_name}_{model_dir_name}_abl_{cfg_name}.json")
        judge_model_subdir = JUDGE_RES_ROOT / model_dir_name
        # Nested layout: <root>/<model>/<task>/<fname>. Opt-in via the
        # ablation_cfg flag (set by --nest_task_subdir at the CLI), used by
        # the ablation runners so per-task per-config JSONs sit in their own
        # task folder under the model dir.
        if ablation_cfg and ablation_cfg.get("_nest_task_subdir"):
            judge_model_subdir = judge_model_subdir / task_name
        judge_model_subdir.mkdir(parents=True, exist_ok=True)
        per_task_path = judge_model_subdir / fname
        existing: dict[str, dict] = {}
        existing_instances_by_id: dict[str, dict] = {}
        if per_task_path.is_file():
            try:
                prev = json.loads(per_task_path.read_text(encoding="utf-8"))
                for it in prev.get("instances", []):
                    if isinstance(it, dict) and it.get("id"):
                        existing[it["id"]] = it
                        existing_instances_by_id[it["id"]] = it
            except Exception:
                pass
        n_resume = sum(1 for v in existing.values()
                       if v.get("success") in (0, 1))
        if n_resume:
            print(f"  [{task_name}] resuming: {n_resume} instance(s) already "
                  f"judged in {_relpath_from_project(per_task_path)}",
                  flush=True)

        for level, iid, out_path in outputs:
            rubric_id = (f"{task_name}_{iid}" if level == "no_lv"
                         else f"{task_name}_{level}_{iid}")
            # Skip if a clean (non-error) verdict is already persisted.
            cached = existing_instances_by_id.get(rubric_id)
            if cached and cached.get("success") in (0, 1):
                # `_shape_instance` drops level from the persisted blob;
                # restore it (and a couple of fields the post-loop summary
                # expects) so the cached entry survives the summary stage.
                inst = dict(cached)
                inst.setdefault("level", level)
                inst.setdefault("rubric_score", cached.get("rubric_score"))
                inst.setdefault("overall_reasoning",
                                cached.get("overall_reasoning", ""))
                # Backfill: rubric is done but --completeness is on and this
                # instance has no completeness_score yet -> run ONLY the
                # completeness pass (one call) and merge it + final_score,
                # keeping the existing rubric verdict untouched.
                want_compl = bool((ablation_cfg or {}).get("completeness"))
                has_compl = isinstance(cached.get("completeness_score"),
                                       (int, float))
                if want_compl and not has_compl:
                    gt_image = find_gt_image(data_task_dir, task_name,
                                             level, iid)
                    input_image = find_input_image(data_task_dir, task_name,
                                                    level, iid)
                    print(f"  [{task_name}/{level}/{iid}] backfilling "
                          f"completeness (rubric kept) ...", flush=True)
                    merge = _run_completeness_only(
                        api_key, base_url, judge_model_name,
                        task_name, level, iid, out_path,
                        input_image, gt_image, data_task_dir,
                        cached.get("rubric_score"))
                    if merge is not None:
                        inst.update(merge)
                        existing[rubric_id].update(merge)
                        # Recompute success under the comp & rub rule now
                        # that this cached (rubric-only) instance has a
                        # completeness score.
                        rub = inst.get("rubric_score")
                        cs_m = merge.get("completeness_score")
                        new_succ = (1 if (
                            isinstance(rub, (int, float)) and rub >= 1.0 - 1e-9
                            and isinstance(cs_m, (int, float))
                            and cs_m >= 1.0 - 1e-9) else 0)
                        inst["success"] = new_succ
                        existing[rubric_id]["success"] = new_succ
                        n_backfilled += 1
                        cs = merge.get("completeness_score")
                        fs = merge.get("final_score")
                        cstr = (f"compl={cs:.2f}(t{merge.get('completeness_tier')})"
                                if isinstance(cs, (int, float)) else "compl=-")
                        fstr = (f"final={fs:.3f}"
                                if isinstance(fs, (int, float)) else "final=-")
                        print(f"    -> {cstr}  {fstr}", flush=True)
                else:
                    print(f"  [{task_name}/{level}/{iid}] skipping — already "
                          f"judged (success={cached.get('success')})",
                          flush=True)
                instances.append(inst)
                continue
            # Per-instance id wins; otherwise fall back to a `default`
            # entry so checklist-style task-wide rubrics work without
            # needing one entry per (level, iid).
            rubric_entry = select_rubric_entry(
                rubric_map, task_name, level, iid)
            gt_image = find_gt_image(data_task_dir, task_name, level, iid)
            input_image = find_input_image(data_task_dir, task_name, level, iid)

            print(f"  [{task_name}/{level}/{iid}] judging "
                  f"{out_path.name} ...", flush=True)
            try:
                v = judge_one_real(api_key, base_url, judge_model_name,
                                   task_name, level, iid, out_path,
                                   rubric_entry, input_image, gt_image,
                                   num_workers=num_workers,
                                   ablation_cfg=ablation_cfg,
                                   data_task_dir=data_task_dir)
                rs_raw = v.get("rubric_score")
                if rs_raw is None:
                    rs_round = None          # completeness-only run
                else:
                    try:
                        rs_round = round(float(rs_raw), 3)
                    except (TypeError, ValueError):
                        rs_round = 0.0
                # `success is None` means agentic_judge_video flagged
                # this instance as ERROR (e.g. one or more batch calls
                # failed and the polish verdict is unreliable). Don't
                # coerce to False — let it propagate as instance-level
                # error so it shows up in the n_error count.
                raw_success = v.get("success")
                instance_error_msg = v.get("error")
                if raw_success is None and instance_error_msg:
                    success_for_inst: int | None = None
                    success_bool = False
                else:
                    success_bool = _coerce_bool(raw_success, False)
                    success_for_inst = 1 if success_bool else 0
                reasoning_text = str(
                    v.get("overall_reasoning")
                    or v.get("reasoning")
                    or "").strip()
                inst = {
                    "id": rubric_id,
                    "level": level,
                    "output_file": _relpath_from_project(out_path),
                    "has_rubric": 1 if rubric_entry else 0,
                    "has_gt_ref": 1 if gt_image else 0,
                }
                if "n_rounds" in v:
                    inst["n_rounds"] = v["n_rounds"]
                cs_full = v.get("checklist_detail")
                if cs_full:
                    inst["checklist_detail"] = cs_full
                inst["rubric_score"] = rs_round
                for fld in ("completeness_score", "completeness_tier",
                            "final_score"):
                    if v.get(fld) is not None:
                        inst[fld] = v.get(fld)
                if v.get("completeness_reasoning"):
                    inst["completeness_reasoning"] = v["completeness_reasoning"]
                inst["success"] = success_for_inst
                inst["overall_reasoning"] = reasoning_text
                if instance_error_msg:
                    inst["error"] = instance_error_msg
                cscore = v.get("completeness_score")
                ctier = v.get("completeness_tier")
                fscore = v.get("final_score")
                compl_str = (f"compl={cscore:.2f}(t{ctier})"
                             if isinstance(cscore, (int, float)) else "")
                final_str = (f"final={fscore:.3f}"
                             if isinstance(fscore, (int, float)) else "")
                extra = "  ".join(s for s in (compl_str, final_str) if s)
                if cs_full:
                    cs = cs_full
                    print(f"    -> success={success_bool}  "
                          f"rubric={rs_round:.3f}"
                          f"{('  ' + extra) if extra else ''}",
                          flush=True)
                    for it in cs["items"]:
                        nv = it.get("n_violations", 0)
                        sc = it.get("score", _viol_to_score(nv))
                        mark = f"{sc:.2f} {nv}v"
                        print(f"       [{mark}] {it['index']}. "
                              f"{it['evidence']}", flush=True)
                else:
                    print(f"    -> success={success_bool}"
                          f"{('  ' + extra) if extra else ''}", flush=True)
                if v.get("agentic"):
                    agentic_traces.append(v["agentic"])
                new_traces = [v["agentic"]] if v.get("agentic") else []
                if not new_traces:
                    # No agentic trace (e.g. judge aborted polish): inject
                    # inst directly so the verdict still survives to disk.
                    # Without this branch the entry only lived in the
                    # in-memory list and was lost between runs, causing
                    # infinite resume retries.
                    existing[inst["id"]] = inst
                try:
                    _flush_per_task_json(
                        per_task_path, task_name, model_dir_name,
                        judge_model_name, cfg_name,
                        new_traces, existing)
                except Exception as flush_err:
                    print(f"    [warn] per-task json flush failed: "
                          f"{flush_err}", flush=True)
                instances.append(inst)
            except Exception as e:
                # Salvage: if `inst` was fully built before the failure
                # (e.g. judge call succeeded with a rubric_score and the
                # exception came from a downstream print/format step), keep
                # those scores instead of discarding them.
                salvage = locals().get("inst")
                if (isinstance(salvage, dict)
                        and salvage.get("rubric_score") is not None):
                    err_inst = {**salvage, "error": str(e)}
                else:
                    err_inst = {
                        "id": rubric_id,
                        "level": level,
                        "output_file": _relpath_from_project(out_path),
                        "has_rubric": 1 if rubric_entry else 0,
                        "has_gt_ref": 1 if gt_image else 0,
                        "success": None,
                        "error": str(e),
                    }
                instances.append(err_inst)
                existing[err_inst["id"]] = err_inst
                try:
                    _flush_per_task_json(
                        per_task_path, task_name, model_dir_name,
                        judge_model_name, cfg_name,
                        [], existing)
                except Exception as flush_err:
                    print(f"    [warn] err-entry json flush failed: "
                          f"{flush_err}", flush=True)

        # If we only backfilled completeness onto cached instances (no fresh
        # judgments to trigger an in-loop flush), persist the updated
        # `existing` blob once here so the merged completeness/final survive.
        if n_backfilled:
            try:
                _flush_per_task_json(per_task_path, task_name, model_dir_name,
                                     judge_model_name, cfg_name, [], existing)
                print(f"  [{task_name}] backfilled completeness on "
                      f"{n_backfilled} instance(s)", flush=True)
            except Exception as flush_err:
                print(f"    [warn] backfill flush failed: {flush_err}",
                      flush=True)

        # Per-task json is now written incrementally inside the loop via
        # _flush_per_task_json. Here we only annotate the in-memory
        # `instances` list with the on-disk path so the summary printer
        # can reference it.
        if existing:
            print(f"    per-task judge -> "
                  f"{_relpath_from_project(per_task_path)}  "
                  f"(n={len(existing)})", flush=True)
            rel_path = _relpath_from_project(per_task_path)
            for inst in instances:
                if inst.get("id") in existing:
                    inst["per_task_json"] = rel_path

        n_total = len(instances)
        n_succ = sum(1 for i in instances if i.get("success") == 1)
        n_fail = sum(1 for i in instances if i.get("success") == 0)
        n_err = sum(1 for i in instances if i.get("success") is None)

        # Average rubric_score across instances that have a numeric value.
        # Instances with errors / no rubric simply don't contribute; the
        # inter-instance step is a plain mean.
        def _avg_rub_variant(field: str):
            xs = [float(i[field]) for i in instances
                  if isinstance(i.get(field), (int, float))]
            return (sum(xs) / len(xs)) if xs else None

        avg_rubric = _avg_rub_variant("rubric_score")

        by_level: dict[str, dict] = {}
        for inst in instances:
            b = by_level.setdefault(
                inst["level"],
                {"n_total": 0, "n_success": 0, "n_failed": 0, "n_error": 0,
                 "_rs_arith": []})
            b["n_total"] += 1
            if inst.get("success") == 1:
                b["n_success"] += 1
            elif inst.get("success") == 0:
                b["n_failed"] += 1
            else:
                b["n_error"] += 1
            v = inst.get("rubric_score")
            if isinstance(v, (int, float)):
                b["_rs_arith"].append(float(v))
        levels_summary = {}
        for lvl, b in by_level.items():
            def _m(key):
                xs = b[key]
                return (sum(xs) / len(xs)) if xs else None
            levels_summary[lvl] = {
                "n_total": b["n_total"],
                "n_success": b["n_success"],
                "n_failed": b["n_failed"],
                "n_error": b["n_error"],
                "success_rate": (b["n_success"] / b["n_total"])
                                if b["n_total"] else 0.0,
                "avg_rubric_score": _m("_rs_arith"),
            }

        tasks_summary[task_name] = {
            "n_total": n_total,
            "n_success": n_succ,
            "n_failed": n_fail,
            "n_error": n_err,
            "success_rate": (n_succ / n_total) if n_total else 0.0,
            "avg_rubric_score": avg_rubric,
            "levels": levels_summary,
        }
        task_details[task_name] = instances
        overall_total += n_total
        overall_succ += n_succ
        overall_fail += n_fail
        overall_err += n_err

    overall_rate = (overall_succ / overall_total) if overall_total else 0.0

    # Overall avg rubric_score, weighted per-instance across all task_details.
    all_rs_arith: list[float] = []
    for insts in task_details.values():
        for i in insts:
            v = i.get("rubric_score")
            if isinstance(v, (int, float)):
                all_rs_arith.append(float(v))

    def _mean_or_none(xs):
        return (sum(xs) / len(xs)) if xs else None

    overall_avg_rubric = _mean_or_none(all_rs_arith)

    task_classes = load_task_classes()
    cls_acc: dict[str, dict] = {}
    for tname, ts in tasks_summary.items():
        cls = task_classes.get(tname, "other")
        bucket = cls_acc.setdefault(
            cls, {"n_total": 0, "n_success": 0, "n_failed": 0, "n_error": 0,
                  "_rs_arith": []})
        bucket["n_total"]   += ts["n_total"]
        bucket["n_success"] += ts["n_success"]
        bucket["n_failed"]  += ts["n_failed"]
        bucket["n_error"]   += ts["n_error"]
        for inst in task_details.get(tname, []):
            v = inst.get("rubric_score")
            if isinstance(v, (int, float)):
                bucket["_rs_arith"].append(float(v))

    classes_summary: dict = {}
    for cls, b in cls_acc.items():
        if b["n_total"] == 0:
            continue
        classes_summary[cls] = {
            "n_total": b["n_total"],
            "n_success": b["n_success"],
            "n_failed": b["n_failed"],
            "n_error": b["n_error"],
            "success_rate": b["n_success"] / b["n_total"],
            "avg_rubric_score": _mean_or_none(b["_rs_arith"]),
        }

    # Ablation runs must NOT clobber the full-method top-level summary.
    # The per-task JSON already uses an `_abl_<cfg>` suffix; mirror it on
    # the top-level file, and record the ablation config inside the blob
    # so consumers can't confuse an ablation run for the canonical one.
    summary_cfg_name = _ablation_config_name(ablation_cfg)
    out_summary: dict = {
        "gen_model": model_dir_name,
        "judge_model": judge_model_name,
        "ablation_config": summary_cfg_name or "full_method",
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "success_rate": overall_rate,
        "avg_rubric_score": overall_avg_rubric,
        "classes": classes_summary,
        "n_total": overall_total,
        "n_success": overall_succ,
        "n_failed": overall_fail,
        "n_error": overall_err,
        "tasks": tasks_summary,
    }

    summary_fname = (f"_all_judge_{model_dir_name}.txt"
                     if not summary_cfg_name
                     else f"_all_judge_{model_dir_name}_abl_"
                          f"{summary_cfg_name}.txt")
    out_dir = JUDGE_RES_ROOT / model_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / summary_fname
    out_path.write_text(
        _render_summary_txt(out_summary, task_details, task_classes),
        encoding="utf-8")
    print(f"\nWrote judge summary -> {out_path}")
    return out_summary


def _render_summary_txt(summary: dict, task_details: dict,
                        task_classes: dict) -> str:
    """Render the per-model judge summary as a plain-text report with
    three tables: (1) class × level success_rate + rubric_score arith;
    (2) per-task rollup; (3) per-task × level breakdown. Both metrics
    are computed by inter-instance arithmetic mean over instances that
    actually have the field; instances missing a metric simply don't
    contribute.

    `summary` is the same `out_summary` blob the JSON writer used to
    emit; `task_details` is the original {task: [instance, ...]} dict
    so we can recompute per-(class, level) buckets from raw instances."""
    lines: list[str] = []
    add = lines.append

    add(f"gen_model     : {summary.get('gen_model')}")
    add(f"judge_model   : {summary.get('judge_model')}")
    add(f"ablation      : {summary.get('ablation_config')}")
    add(f"generated_at  : {summary.get('generated_at')}")
    add(f"n_instances   : {summary.get('n_total')}"
        f"  (success={summary.get('n_success')}, "
        f"failed={summary.get('n_failed')}, "
        f"error={summary.get('n_error')})")
    sr = summary.get('success_rate')
    ar = summary.get('avg_rubric_score')

    # comp./final live only on instances that ran the completeness pass; the
    # tables and header pull them straight off raw instances and show '-' when
    # absent. _sc formats a score list into a fixed-width cell.
    def _sc(vals, w: int = 6) -> str:
        return (f"{(sum(vals) / len(vals)):>{w}.3f}" if vals
                else f"{'-':>{w}}")

    def _metrics(n: int, succ: int, rs, cs, fs) -> str:
        srate = (succ / n) if n else 0
        return (f"{n:>4d}  {srate*100:>10.1f}%   "
                f"{_sc(rs)}  {_sc(cs)}  {_sc(fs)}")

    _HDR = (f"{'n':>4s}  {'success%':>11s}   "
            f"{'rub.':>6s}  {'comp.':>6s}  {'final':>6s}")

    def _collect(insts):
        """(n, succ, rubric[], completeness[], final[]) over instances."""
        rs = [float(i["rubric_score"]) for i in insts
              if isinstance(i.get("rubric_score"), (int, float))]
        cs = [float(i["completeness_score"]) for i in insts
              if isinstance(i.get("completeness_score"), (int, float))]
        fs = [float(i["final_score"]) for i in insts
              if isinstance(i.get("final_score"), (int, float))]
        succ = sum(1 for i in insts if i.get("success") == 1)
        return len(insts), succ, rs, cs, fs

    _all_insts = [i for insts in task_details.values() for i in insts]
    _, _, _all_rs, _all_cs, _all_fs = _collect(_all_insts)
    add(f"overall       : success_rate = "
        f"{(sr*100 if sr is not None else 0):5.1f}%   "
        f"rubric = {(ar if ar is not None else 0):.3f}   "
        f"comp. = {_sc(_all_cs, 1).strip()}   "
        f"final = {_sc(_all_fs, 1).strip()}")
    add("")

    # ---- 1) class × level ----
    # Build buckets from raw instances so we don't depend on any
    # pre-computed structure.
    cls_lv: dict[tuple[str, str], list] = {}
    cls_overall: dict[str, list] = {}
    for task_name, insts in task_details.items():
        cls = task_classes.get(task_name, "other")
        for inst in insts:
            lv = _level_from_id(inst.get("id", ""), task_name)
            cls_overall.setdefault(cls, []).append(inst)
            cls_lv.setdefault((cls, lv), []).append(inst)

    def _fmt_row(label_cls, label_lv, insts):
        n, succ, rs, cs, fs = _collect(insts)
        return (f"  {label_cls:14s}  {label_lv:14s}  "
                + _metrics(n, succ, rs, cs, fs))

    add("=" * 60)
    add("class x level")
    add("-" * 60)
    add(f"  {'class':14s}  {'level':14s}  " + _HDR)
    add("-" * 60)
    classes_sorted = sorted(cls_overall.keys())
    for cls in classes_sorted:
        # per-level rows for this class, sorted by natural level key
        lvs = sorted([k[1] for k in cls_lv if k[0] == cls], key=_natkey)
        for lv in lvs:
            add(_fmt_row(cls, lv, cls_lv[(cls, lv)]))
        add(_fmt_row(cls, "(overall)", cls_overall[cls]))
        add("")

    # ---- 2) per-task rollup ----
    add("=" * 60)
    add("per-task rollup")
    add("-" * 60)
    add(f"  {'task':24s}  {'cls':14s}  " + _HDR)
    add("-" * 60)
    for task_name in sorted(task_details, key=_natkey):
        insts = task_details[task_name]
        n, succ, rs, cs, fs = _collect(insts)
        cls = task_classes.get(task_name, "other")
        add(f"  {task_name:24s}  {cls:14s}  "
            + _metrics(n, succ, rs, cs, fs))
    add("")

    # ---- 3) per-task x level breakdown ----
    add("=" * 60)
    add("per-task x level")
    add("-" * 60)
    add(f"  {'task':24s}  {'level':10s}  " + _HDR)
    add("-" * 60)
    for task_name in sorted(task_details, key=_natkey):
        insts = task_details[task_name]
        # group by level
        by_lv: dict[str, list[dict]] = {}
        for inst in insts:
            lv = _level_from_id(inst.get("id", ""), task_name)
            by_lv.setdefault(lv, []).append(inst)
        for lv in sorted(by_lv, key=_natkey):
            n, succ, rs, cs, fs = _collect(by_lv[lv])
            add(f"  {task_name:24s}  {lv:10s}  "
                + _metrics(n, succ, rs, cs, fs))
        add("")
    return "\n".join(lines)


def main() -> int:
    # Windows-default cp936 console crashes on common GPT output chars like
    # U+2011 (non-breaking hyphen). Force utf-8 + replace so a rogue char in
    # the rubric printout cannot abort an instance mid-stream.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, Exception):
            pass
    parser = argparse.ArgumentParser(
        description="VLM-as-judge over generated benchmark outputs.")
    parser.add_argument("model_dir",
                        help="Model directory under eval_res/ to judge "
                             "(e.g. 'kling_o1', 'veo3_1').")
    parser.add_argument("--judge_model", default=DEFAULT_JUDGE_MODEL,
                        help=f"Judge VLM (default: {DEFAULT_JUDGE_MODEL}).")
    parser.add_argument("--task", default=None,
                        help="Comma-separated subset of task names.")
    parser.add_argument("--max_per_task", type=int, default=None,
                        help="Cap instances judged per task (debugging).")
    parser.add_argument("--num_workers", type=int,
                        default=AGENTIC_NUM_WORKERS,
                        help=f"Parallel batch threads per round "
                             f"(default: {AGENTIC_NUM_WORKERS}).")
    parser.add_argument("--eval_root", default=None,
                        help="Override the eval root scanned for outputs "
                             "(default: eval/eval_res). For auto-gen runs "
                             "pass 'eval/auto_eval_res'.")
    parser.add_argument("--rubric_root", default=None,
                        help="Additional root to consult FIRST for the "
                             "rubric file lookup (does NOT affect images / "
                             "completeness file resolution, which keep using "
                             "the canonical data tree). Layout expected: "
                             "<rubric_root>/<task>/_judge_rubric_<task>.json.")
    parser.add_argument("--completeness_root", default=None,
                        help="Additional root to consult FIRST for the "
                             "_completeness_judge.txt lookup. Same shape as "
                             "--rubric_root: <completeness_root>/<task>/"
                             "_completeness_judge.txt. Used by the style "
                             "ablation to swap in style-specific tier "
                             "descriptions (line-art is a blue ball not a car, "
                             "sim is a low-poly vehicle, etc.).")
    parser.add_argument("--judge_res_root", default=None,
                        help="Override the directory where per-task and "
                             "aggregated judge results are written "
                             "(default: eval/judge_res).")
    # ---- ablation flags -------------------------------------------------
    # The full method = adaptive fps + 10-frame window (the defaults below).
    # Any non-default combination is an ablation; its per-task JSON is
    # written as `_judge_<task>_abl_<config>.json` so it sits next to the
    # full-method result without clobbering it.
    abl = parser.add_argument_group("ablation")
    abl.add_argument("--sampling", choices=["adaptive", "even"],
                     default="adaptive",
                     help="'adaptive': round-1 + round-2 zoom schedule "
                          "(the method). 'even': one fixed fps over the "
                          "whole clip, no zoom. Default: adaptive.")
    abl.add_argument("--batching", choices=["window", "all_at_once"],
                     default="window",
                     help="'window': 10-frame window w/ 1-frame overlap "
                          "(the method). 'all_at_once': one LLM call sees "
                          "every sampled frame. Default: window.")
    abl.add_argument("--even_fps", type=int, default=4,
                     help="Fixed sampling fps when --sampling even "
                          "(ablation sweeps 2/4/8). Default: 4.")
    abl.add_argument("--abl_suffix_override", default=None,
                     help="Force the `_abl_<tag>` filename suffix to this "
                          "exact string, bypassing the sampling-derived "
                          "auto-tag. Used by judge-model-swap ablations "
                          "(e.g. --abl_suffix_override claude_sonnet_4_6) "
                          "so the output sits next to sampling ablations in "
                          "the same `judge_res_abltn/<model>/` dir.")
    abl.add_argument("--nest_task_subdir", action="store_true",
                     help="Nest per-task JSON under `<root>/<model>/<task>/` "
                          "instead of writing flat under `<root>/<model>/`. "
                          "Used by ablation runners so per-config files sit "
                          "in their own task folder. The per-model summary "
                          "txt still goes to `<root>/<model>/`.")
    abl.add_argument("--no_completeness", dest="completeness",
                     action="store_false", default=True,
                     help="Disable the completeness pass (it is ON by "
                          "default). With completeness on, one 2-fps "
                          "all-frames call scores goal completion (tier "
                          "0/1/2 -> 0/0.5/1) against "
                          "<task>/_completeness_judge.txt and final_score = "
                          "rubric_score * completeness; an instance is a "
                          "success only when both == 1. Implies rubric-only "
                          "(writes `_abl_rub_only`).")
    abl.add_argument("--no_rubric", dest="rubric",
                     action="store_false", default=True,
                     help="Disable the rubric pass (it is ON by default), "
                          "i.e. score completeness ONLY. Skips the agentic "
                          "rubric loop entirely; rubric_score / final_score "
                          "are null and success = (completeness == 1). "
                          "Writes `_abl_comp_only`. Cannot be combined with "
                          "--no_completeness.")
    args = parser.parse_args()
    if not args.rubric and not args.completeness:
        parser.error("--no_rubric and --no_completeness cannot both be set "
                     "(nothing would be judged).")
    if args.eval_root:
        global EVAL_ROOT
        EVAL_ROOT = (PROJECT_ROOT / args.eval_root).resolve()
    if args.rubric_root:
        # Override ONLY rubric lookup — keep DATA_ROOTS untouched so the
        # completeness file, input image, and gt image resolution still hit
        # the canonical data tree (data/5_ok/<task>/...).
        global _RUBRIC_ROOT_OVERRIDE
        _RUBRIC_ROOT_OVERRIDE = (PROJECT_ROOT / args.rubric_root).resolve()
    if args.completeness_root:
        global _COMPLETENESS_ROOT_OVERRIDE
        _COMPLETENESS_ROOT_OVERRIDE = (PROJECT_ROOT / args.completeness_root).resolve()
    if args.judge_res_root:
        global JUDGE_RES_ROOT
        JUDGE_RES_ROOT = (PROJECT_ROOT / args.judge_res_root).resolve()

    tasks_filter: set[str] | None = None
    if args.task:
        tasks_filter = {t.strip() for t in args.task.split(",") if t.strip()}

    ablation_cfg = {
        "sampling": args.sampling,
        "batching": args.batching,
        "even_fps": args.even_fps,
        "completeness": args.completeness,
        "rubric": args.rubric,
    }
    if getattr(args, "abl_suffix_override", None):
        # Force the filename `_abl_<tag>` suffix to a caller-supplied tag,
        # bypassing the sampling/batching-derived auto-name. Used for
        # judge-model-swap ablations whose differentiator isn't sampling.
        ablation_cfg["_suffix_override"] = args.abl_suffix_override
    if getattr(args, "nest_task_subdir", False):
        ablation_cfg["_nest_task_subdir"] = True
    cfg_name = _ablation_config_name(ablation_cfg)
    if cfg_name:
        print(f"[ablation] config = {cfg_name}  "
              f"(per-task JSON suffix: _abl_{cfg_name})")
    else:
        print("[ablation] config = full_method (adaptive fps + 10-frame "
              "window)")

    summary = judge_model(args.model_dir, args.judge_model,
                          tasks_filter, args.max_per_task,
                          num_workers=args.num_workers,
                          ablation_cfg=ablation_cfg)
    ar = summary.get("avg_rubric_score")
    extra = ""
    if ar is not None:
        extra += f"  avg_rubric_score={ar:.3f}"
    print(f"\nOverall: {summary['n_success']}/{summary['n_total']} "
          f"succeeded ({summary['success_rate']:.1%}); "
          f"failed={summary['n_failed']} errors={summary['n_error']}"
          f"{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
