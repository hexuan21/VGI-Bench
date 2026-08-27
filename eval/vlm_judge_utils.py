"""GT signal router + judge-prompt builders for vlm_judge.

The bottom half of this module (see the `judge prompts` section) holds the
VLM prompt text and its leaf helpers (image encoding, frame downsizing,
system prompts, per-pass message builders), moved out of vlm_judge.py so
the driver / agentic-loop logic stays separable from the prompt strings.
vlm_judge.py re-imports them, so `from vlm_judge import ...` keeps working.

The top half is the GT signal router — single source of truth for two
things:

  (A) Which tasks expect which GT signal(s) — image and/or text — to be
      fed into the judge prompt. See the three `TASKS_GT_*` sets below
      and the derived helper `task_gt_signals(task_name)`.

  (B) Where to find the per-instance / task-wide GT *description* files
      on disk, and how to render their JSON payload into a single
      paragraph the judge prompt can embed. See `gt_desc_for(...)`.

GT *image* discovery still lives in vlm_judge.py (`find_gt_image`) — it
already handles top-level + `_raw/<lv>/*_{line|sim|render}_gt.*` fallback
correctly. This module only TELLS the caller whether to look for an
image at all (`task_gt_signals(task)['image']`).

GT *desc* sources, in order:
  1. Task-wide file: `<data_task_dir>/_gt_desc_<task>.json`
  2. Per-instance file:
     `<data_task_dir>/_raw/<level>/<task>_<level>_<iid>_gt_desc.json`
     (or `<data_task_dir>/_raw/<task>_<iid>_gt_desc.json` for flat
     `no_lv` layouts).

If both exist their rendered strings are concatenated with a blank line
between (task-wide first → per-instance second; general → specific). If
only one exists that one is used alone. If neither, "" is returned and
the caller falls back to the rubric checklist alone.

Each file is either `{"description": "<text>"}` (preferred), a top-level
string (verbatim), or any other dict (rendered as compact JSON as a
last-resort fallback).

The `v_` video-input task-name prefix is stripped before all lookups.
The companion `_solution.json` files (raw render output) are deliberately
NOT read here — that's the renderer's internal artefact, not a judge
input; use the `_raw/.../build_gt_desc.py` helpers to materialise per-
instance `_gt_desc.json` from solution data when needed.
"""
from __future__ import annotations

import base64
import io
import json
import math
import mimetypes
from pathlib import Path
from typing import Any

import datetime
import os
import re
import threading
import time

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ===================== LLM call machinery ============================
# Credential routing, the OpenAI-compat / clawbench POST helpers, usage
# logging, retry budgets and their constants -- all the judge-API
# plumbing, moved out of vlm_judge.py. vlm_judge.py re-imports these so
# `from vlm_judge import _post_chat, load_judge_credentials, ...` (and
# vlm_judge_img.py's imports) keep working.

API_KEY_FILE = PROJECT_ROOT / "const" / "_api_key.json"

# Every judge LLM call appends one usage record under eval/api_usage/,
# split by provider so each bills as its own line item (see _log_llm_usage):
#   OpenRouter calls            -> open_router.jsonl
#   QNAIGC calls                -> qnaigc_usage.jsonl
#   everything else (Google      -> gemini.jsonl
#   direct, NetMind, clawbench)
USAGE_DIR = PROJECT_ROOT / "eval" / "api_usage"
LLM_USAGE_LOG = USAGE_DIR / "gemini.jsonl"
OPENROUTER_USAGE_LOG = USAGE_DIR / "open_router.jsonl"
QNAIGC_USAGE_LOG = USAGE_DIR / "qnaigc_usage.jsonl"
_LLM_USAGE_LOCK = threading.Lock()

DEFAULT_JUDGE_MODEL = "google/gemini-3-flash-preview"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
NETMIND_BASE_URL = "https://api.netmind.ai/inference-api/openai/v1"

# Sufy / QNAIGC OpenAI-compat gateway. Serves Chinese-lab models (Qwen-VL,
# doubao-vision, GLM, DeepSeek, Kimi, MiniMax) plus anthropic/claude-fable-5;
# no openai/* or google/* slugs exist there. Client + model catalogue live in
# eval/models_supported/api_qnaigc.py. Reached via a `qnaigc/<model>` judge
# id, e.g. `qnaigc/qwen3-vl-30b-a3b-thinking`.
QNAIGC_BASE_URL = "https://api.qnaigc.com/v1"

# Models we know NetMind hosts under their `google/<id>` slug. NetMind is
# the only provider with these (Google's first-party API doesn't accept the
# `-preview` aliases for OpenAI-compat, and OpenRouter returns 404), so route
# them straight to NetMind with NETMIND_API_KEY.
NETMIND_MODELS = {"google/gemini-3-pro-preview",
                  "google/gemini-3-flash-preview",
                  "anthropic/claude-sonnet-4-6",
                  "Qwen/Qwen3.6-Plus",
                  "Qwen/Qwen3-VL-235B-A22B-Instruct"}

# clawbench api-wrap gateway (auth via T2V_API_KEY, X-API-Key header). NOT an
# OpenAI-compatible endpoint: POST {base}/chat with body {model, messages} and
# the reply text lives at `.content` (not `.choices[0].message.content`). It
# currently serves only `gemini31flash` (= gemini-3.1-flash); other model ids
# 502. Route any of these aliases through the gateway.
CLAWBENCH_BASE_URL = "https://t2v.clawbench.cc/v1"
CLAWBENCH_MODELS = {"gemini31flash", "clawbench/gemini31flash",
                    "clawbench/gemini-3.1-flash"}
# Map a user-facing alias -> the exact model id the gateway expects.
CLAWBENCH_MODEL_ID = {
    "gemini31flash": "gemini31flash",
    "clawbench/gemini31flash": "gemini31flash",
    "clawbench/gemini-3.1-flash": "gemini31flash",
}

# Endpoints to try after the primary endpoint hits 429. Populated by
# `load_judge_credentials()` based on the chosen model. Each entry is
# (api_key, base_url, model_id); _post_chat walks through them in order.
_FALLBACK_CHAIN: list[tuple[str, str, str]] = []

def load_judge_credentials(model: str = DEFAULT_JUDGE_MODEL
                           ) -> tuple[str, str, str]:
    """Returns (api_key, base_url, resolved_model_id) for the PRIMARY endpoint
    and registers any 429-fallback endpoints in module state.

    Any `gemini-*` model id (e.g. 'gemini-2.5-pro', 'gemini-3.0-pro') is
    routed through Google's OpenAI-compat endpoint with GOOGLE_API_KEY;
    OpenRouter (`google/<id>`) is registered as the 429 fallback. If no
    GOOGLE_API_KEY is present, falls through to OpenRouter directly under
    the `google/<id>` slug.

    Any other model name goes through OPENAI_BASE_URL / OPENROUTER_API_KEY (i.e.
    OpenRouter by default) with no fallback. OPENAI_BASE_URL in the key file
    overrides the OpenRouter default."""
    global _FALLBACK_CHAIN
    data = json.loads(API_KEY_FILE.read_text(encoding="utf-8"))
    google_key = data.get("GOOGLE_API_KEY")
    or_key = data.get("OPENROUTER_API_KEY")
    or_base = (data.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    nm_key = data.get("NETMIND_API_KEY")

    # Explicit OpenRouter route: judge_model "openrouter/<slug>" forces
    # OpenRouter (OPENROUTER_BASE_URL + OPENROUTER_API_KEY) with the bare
    # <slug>, bypassing the NetMind/clawbench/Google routing below. Used when
    # NetMind is out of balance / clawbench is down.
    if model.startswith("openrouter/"):
        slug = model[len("openrouter/"):]
        or_only_base = (data.get("OPENROUTER_BASE_URL")
                        or DEFAULT_BASE_URL).rstrip("/")
        if not or_key:
            raise RuntimeError(
                f"OPENROUTER_API_KEY not found in {API_KEY_FILE}")
        _FALLBACK_CHAIN = []
        return or_key, or_only_base, slug

    # Explicit QNAIGC route: judge_model "qnaigc/<model>" strips the prefix
    # and posts the bare id to the QNAIGC gateway. The prefix must be checked
    # before the generic "<provider>/<name>" -> OpenRouter branch below, since
    # a QNAIGC id may itself contain a slash (e.g. anthropic/claude-fable-5,
    # which as `qnaigc/anthropic/claude-fable-5` must not go to OpenRouter).
    if model.startswith("qnaigc/"):
        qn_key = data.get("QNAIGC_API_KEY")
        if not qn_key:
            raise RuntimeError(
                f"QNAIGC_API_KEY not found in {API_KEY_FILE}")
        qn_base = (data.get("QNAIGC_BASE_URL") or QNAIGC_BASE_URL).rstrip("/")
        _FALLBACK_CHAIN = []
        return qn_key, qn_base, model[len("qnaigc/"):]

    # clawbench api-wrap gateway (X-API-Key auth, non-OpenAI shape — handled
    # specially in _post_chat_one by detecting the clawbench base url).
    if model in CLAWBENCH_MODELS:
        t2v_key = data.get("T2V_API_KEY")
        if not t2v_key:
            raise RuntimeError(
                f"T2V_API_KEY not found in {API_KEY_FILE}; "
                f"required for clawbench model {model}")
        _FALLBACK_CHAIN = []
        return t2v_key, CLAWBENCH_BASE_URL, CLAWBENCH_MODEL_ID.get(model, model)

    # Models we route via NetMind's OpenAI-compat endpoint (currently the
    # only place serving gemini-3-{pro,flash}-preview reliably — Google's
    # spend cap may halt direct API; OpenRouter has no endpoint for them).
    if model in NETMIND_MODELS:
        if not nm_key:
            raise RuntimeError(
                f"NETMIND_API_KEY not found in {API_KEY_FILE}; "
                f"required for model {model}")
        _FALLBACK_CHAIN = []
        return nm_key, NETMIND_BASE_URL, model

    # Any bare gemini-* model id (e.g. gemini-2.5-pro, gemini-3.0-pro,
    # gemini-3-pro-preview) prefers Google's OpenAI-compat endpoint when
    # GOOGLE_API_KEY is present; OpenRouter (`google/<id>`) is registered
    # as the 429 fallback. Without a Google key we go straight to
    # OpenRouter under the `google/<id>` slug.
    if model.startswith("gemini-"):
        or_slug = f"google/{model}"
        if google_key:
            _FALLBACK_CHAIN = ([(or_key, or_base, or_slug)]
                               if or_key else [])
            return google_key, GOOGLE_BASE_URL, model
        if or_key:
            _FALLBACK_CHAIN = []
            return or_key, or_base, or_slug
        raise RuntimeError(
            f"Neither GOOGLE_API_KEY nor OPENROUTER_API_KEY found in {API_KEY_FILE}")

    if not or_key:
        raise RuntimeError(f"OPENROUTER_API_KEY not found in {API_KEY_FILE}")
    # Models with a "<provider>/<name>" slug (e.g. `openai/gpt-5-mini`,
    # `anthropic/claude-...`) are OpenRouter-namespaced — force them through
    # the OpenRouter endpoint instead of an OPENAI_BASE_URL override which
    # would (a) reject the slug and (b) send the wrong API key.
    if "/" in model:
        or_only_base = (data.get("OPENROUTER_BASE_URL")
                        or DEFAULT_BASE_URL).rstrip("/")
        _FALLBACK_CHAIN = []
        return or_key, or_only_base, model
    _FALLBACK_CHAIN = []
    return or_key, or_base, model


def _log_llm_usage(rec: dict) -> None:
    """Append one judge-call usage record under eval/api_usage/ (jsonl).

    Routes by serving endpoint so each gateway bills as its own provider:
    OpenRouter -> open_router.jsonl; QNAIGC -> qnaigc_usage.jsonl; everything
    else (Google direct, NetMind, clawbench) -> gemini.jsonl. Logs every call
    regardless of model id. Best-effort: a logging failure must never break
    the run."""
    try:
        rec = {"ts_utc": datetime.datetime.now(datetime.timezone.utc)
               .strftime("%Y-%m-%dT%H:%M:%SZ"), **rec}
        ep = str(rec.get("endpoint") or "").lower()
        if "openrouter" in ep:
            path = OPENROUTER_USAGE_LOG
        elif "qnaigc" in ep:
            path = QNAIGC_USAGE_LOG
        else:
            path = LLM_USAGE_LOG
        with _LLM_USAGE_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


_RETRY_IN_RE = re.compile(r"retry in ([\d.]+)\s*s", re.IGNORECASE)
_MAX_429_RETRIES = 3
_429_RETRY_BUFFER_S = 1.0
_429_RETRY_CAP_S = 65.0
# Transient server-side errors (NetMind/Google occasionally 503 "service
# unavailable" under load). Unlike 429 there's no Retry-After header, so we
# back off exponentially. Retried independently of the 429 budget.
_RETRY_5XX = {500, 502, 503, 504}
_MAX_5XX_RETRIES = 5
_5XX_BACKOFF_BASE_S = 2.0
_5XX_BACKOFF_CAP_S = 30.0


def _post_chat_clawbench(api_key: str, base_url: str, model: str,
                         messages: list[dict], timeout: int = 600) -> str:
    """Post to the clawbench api-wrap gateway, which is NOT OpenAI-compatible:
      - POST {base_url}/chat  (note: /chat, not /chat/completions)
      - auth via X-API-Key header (not Authorization: Bearer)
      - body {model, messages}; reply text at `.content`
    `response_format` isn't supported by the gateway, so it's dropped (the
    judge prompt already instructs JSON-only output)."""
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages}
    t0 = time.time()
    try:
        resp = requests.post(f"{base_url}/chat", headers=headers,
                             json=payload, timeout=timeout)
    except Exception as e:
        _log_llm_usage({"endpoint": base_url, "model": model, "status": "error",
                        "elapsed_s": round(time.time() - t0, 2),
                        "error": f"{type(e).__name__}: {e}"})
        raise
    elapsed = round(time.time() - t0, 2)
    if not resp.ok:
        _log_llm_usage({"endpoint": base_url, "model": model, "status": "error",
                        "elapsed_s": elapsed, "http_status": resp.status_code,
                        "error": resp.text[:600]})
        raise RuntimeError(f"chat call failed [{resp.status_code}]: "
                           f"{resp.text[:600]}")
    data = resp.json()
    usage = data.get("usage") or {}
    _log_llm_usage({
        "endpoint": base_url, "model": data.get("model") or model,
        "status": "ok", "elapsed_s": elapsed, "http_status": resp.status_code,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    })
    content = data.get("content")
    if content is None:
        raise RuntimeError(f"clawbench reply missing 'content': "
                           f"{json.dumps(data)[:400]}")
    return content


def _post_chat_one(api_key: str, base_url: str, model: str,
                   messages: list[dict],
                   response_format: dict | None = None,
                   timeout: int = 600) -> str:
    # clawbench gateway has a bespoke shape — handle it separately.
    if base_url.rstrip("/") == CLAWBENCH_BASE_URL:
        return _post_chat_clawbench(api_key, base_url, model, messages, timeout)
    # NetMind occasionally hangs a connection; cap its read timeout far below
    # the 600s default so a stalled call fails fast (~2.5 min), freeing the
    # worker slot. The instance is then retried in a later loop round instead
    # of blocking the slot for ~10 min. Env NETMIND_TIMEOUT_S overrides.
    if base_url.rstrip("/") == NETMIND_BASE_URL.rstrip("/"):
        import os as _os
        nm_to = int(_os.environ.get("NETMIND_TIMEOUT_S", "150"))
        timeout = min(timeout, nm_to)
    payload: dict = {"model": model, "messages": messages}
    if response_format:
        payload["response_format"] = response_format
    # Cap the per-call output budget. OpenRouter reserves credit up to the
    # model's max-output ceiling (e.g. gpt-5-mini defaults to ~65k) and
    # returns 402 if balance can't cover the reservation, even when the
    # actual response is far smaller. 8k is plenty for our rubric replies.
    payload.setdefault("max_tokens", 8000)
    # gpt-5* models reason by default (effort=medium). For judging we want
    # fast, non-thinking verdicts, so pin minimal reasoning effort. Scoped to
    # gpt-5 slugs so other judges are unaffected.
    if "gpt-5" in model.lower():
        payload["reasoning_effort"] = "minimal"
    # Qwen3.x are hybrid reasoners with thinking ON by default: ~2.7k output
    # tokens and ~40 s per call, vs ~200 tokens / ~5 s with it off. Judging
    # wants a verdict, not a chain of thought. OpenRouter takes the unified
    # `reasoning.enabled`; the Alibaba-style gateways take `enable_thinking`.
    # Set QWEN_THINKING=1 to restore the thinking behaviour.
    # Kimi k2.x reason by default too. `enable_thinking` is ignored for them;
    # the Moonshot-style `thinking: {"type": "disabled"}` is what works (the
    # answer then lands in `content` instead of `reasoning_content`).
    if "kimi" in model.lower() and os.environ.get("KIMI_THINKING") != "1":
        payload["thinking"] = {"type": "disabled"}
    if "qwen" in model.lower() and os.environ.get("QWEN_THINKING") != "1":
        if base_url.rstrip("/") == QNAIGC_BASE_URL.rstrip("/"):
            payload["enable_thinking"] = False
        else:
            payload["reasoning"] = {"enabled": False}
        # With thinking off a reply is ~230 tokens (longest observed: 1755), so
        # the 8k default only inflates OpenRouter's up-front credit reservation
        # — 16 concurrent workers reserve 16 x max_tokens and 402 out. 2k keeps
        # the reservation small without truncating a reply into invalid JSON.
        payload["max_tokens"] = 2000
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # OpenRouter likes these for billing attribution; harmless on OpenAI.
        "HTTP-Referer": "https://github.com/anthropics/claude-code",
        "X-Title": "VGenReason VLM Judge",
    }
    # Two independent retry budgets: 429 (rate-limit, honors Retry-After) and
    # 5xx (transient server errors, exponential backoff). A 5xx retry does not
    # consume the 429 budget and vice versa.
    attempt_429 = 0
    attempt_5xx = 0
    while True:
        t0 = time.time()
        try:
            resp = requests.post(f"{base_url}/chat/completions",
                                 headers=headers, json=payload, timeout=timeout)
        except Exception as e:
            _log_llm_usage({"endpoint": base_url, "model": model,
                            "status": "error",
                            "elapsed_s": round(time.time() - t0, 2),
                            "error": f"{type(e).__name__}: {e}"})
            raise
        elapsed = round(time.time() - t0, 2)
        if resp.status_code == 429 and attempt_429 < _MAX_429_RETRIES:
            attempt_429 += 1
            m = _RETRY_IN_RE.search(resp.text)
            wait = (float(m.group(1)) + _429_RETRY_BUFFER_S) if m else 10.0
            wait = min(wait, _429_RETRY_CAP_S)
            _log_llm_usage({"endpoint": base_url, "model": model,
                            "status": "error", "elapsed_s": elapsed,
                            "http_status": 429,
                            "error": f"[retry {attempt_429}/{_MAX_429_RETRIES} "
                                     f"after {wait:.1f}s] " + resp.text[:500]})
            print(f"  [judge] 429 on {base_url} ({model}); sleeping "
                  f"{wait:.1f}s then retrying "
                  f"(attempt {attempt_429}/{_MAX_429_RETRIES})", flush=True)
            time.sleep(wait)
            continue
        if resp.status_code in _RETRY_5XX and attempt_5xx < _MAX_5XX_RETRIES:
            attempt_5xx += 1
            wait = min(_5XX_BACKOFF_BASE_S * (2 ** (attempt_5xx - 1)),
                       _5XX_BACKOFF_CAP_S)
            _log_llm_usage({"endpoint": base_url, "model": model,
                            "status": "error", "elapsed_s": elapsed,
                            "http_status": resp.status_code,
                            "error": f"[5xx retry {attempt_5xx}/"
                                     f"{_MAX_5XX_RETRIES} after {wait:.1f}s] "
                                     + resp.text[:500]})
            print(f"  [judge] {resp.status_code} on {base_url} ({model}); "
                  f"sleeping {wait:.1f}s then retrying (attempt "
                  f"{attempt_5xx}/{_MAX_5XX_RETRIES})", flush=True)
            time.sleep(wait)
            continue
        if not resp.ok:
            _log_llm_usage({"endpoint": base_url, "model": model,
                            "status": "error", "elapsed_s": elapsed,
                            "http_status": resp.status_code,
                            "error": resp.text[:600]})
            raise RuntimeError(f"chat call failed [{resp.status_code}]: "
                               f"{resp.text[:600]}")
        data = resp.json()
        usage = data.get("usage") or {}
        _log_llm_usage({
            "endpoint": base_url, "model": data.get("model") or model,
            "status": "ok", "elapsed_s": elapsed,
            "http_status": resp.status_code,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        })
        msg = data["choices"][0]["message"]
        content = msg.get("content")
        # Some thinking models served via QNAIGC (e.g. moonshotai/kimi-k2.6)
        # return an empty `content` and put the entire final answer in
        # `reasoning_content`. Without this fallback the reply parses as {}
        # and a judge silently records success=False for every instance.
        if not (isinstance(content, str) and content.strip()):
            rc = msg.get("reasoning_content")
            if isinstance(rc, str) and rc.strip():
                return rc
        return content


def _post_chat(api_key: str, base_url: str, model: str,
               messages: list[dict],
               response_format: dict | None = None,
               timeout: int = 600) -> str:
    """Try (api_key, base_url, model); on 429 walk through any registered
    fallback endpoints from `_FALLBACK_CHAIN` in order. Other errors propagate
    immediately."""
    attempts = [(api_key, base_url, model), *_FALLBACK_CHAIN]
    last_err: Exception | None = None
    for i, (k, u, m) in enumerate(attempts):
        try:
            return _post_chat_one(k, u, m, messages, response_format, timeout)
        except RuntimeError as e:
            msg = str(e).upper()
            is_429 = "[429]" in msg or "RESOURCE_EXHAUSTED" in msg
            if is_429 and i + 1 < len(attempts):
                nxt = attempts[i + 1]
                print(f"  [judge] 429 on {u} ({m}); falling back to "
                      f"{nxt[1]} ({nxt[2]})", flush=True)
                last_err = e
                continue
            raise
    raise last_err or RuntimeError("all endpoints exhausted")


# ---------- task GT signal classification --------------------------------
#
# Three disjoint groups + two derived unions. Edit the three groups when
# a task gains or loses a GT signal; the unions and `task_gt_signals()`
# stay in sync automatically.

TASKS_GT_IMAGE_AND_DESC: set[str] = {
    "pick_unique", "clock_running", "gears",
}

TASKS_GT_IMAGE_ONLY: set[str] = {
    "polyform_tiling", "sorting", "maze_round", "maze_square",
    "sliding_puzzle", "hanoi_tower", "jigsaw_puzzle",
    "section_3d_figure", "recover_2d_net",
    "expand_3d_to_2d", "grouping_clustering",
}

TASKS_GT_DESC_ONLY: set[str] = {
    "leave_parking_lot", "unhook_ring", "untie_knot", "rope_untangle",
}

# Derived. Do not edit directly — change the three groups above instead.
TASKS_NEED_GT_IMAGE: set[str] = TASKS_GT_IMAGE_AND_DESC | TASKS_GT_IMAGE_ONLY
TASKS_NEED_GT_DESC:  set[str] = TASKS_GT_IMAGE_AND_DESC | TASKS_GT_DESC_ONLY


def _strip_v_prefix(task_name: str) -> str:
    return task_name[2:] if task_name.startswith("v_") else task_name


def task_gt_signals(task_name: str) -> dict[str, bool]:
    """Return {`image`: bool, `desc`: bool} telling vlm_judge which GT
    signals to feed this task's judge prompt. Tasks not listed in either
    set return both False (judged on the rubric checklist alone). The
    `v_` video-input prefix is stripped before lookup."""
    bare = _strip_v_prefix(task_name)
    return {
        "image": bare in TASKS_NEED_GT_IMAGE,
        "desc":  bare in TASKS_NEED_GT_DESC,
    }


def _read_json_lenient(p: Path) -> Any | None:
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _instance_id(task: str, level: str, iid: str) -> str:
    return f"{task}_{iid}" if level == "no_lv" else f"{task}_{level}_{iid}"


def _render(content: Any) -> str:
    """Coerce a parsed gt_desc.json payload to a single string.

    `{"description": "..."}` → that string.
    Bare string → the string verbatim.
    Any other dict → compact JSON dump (best-effort fallback so something
    still reaches the prompt).
    Anything else → ""."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        desc = content.get("description")
        if isinstance(desc, str) and desc.strip():
            return desc.strip()
        return json.dumps(content, ensure_ascii=False)
    return ""


def load_task_wide_gt_desc(task_name: str,
                           data_task_dir: Path | None) -> Any | None:
    """Read `<data_task_dir>/_gt_desc_<task>.json`; returns parsed payload
    or None if absent."""
    if data_task_dir is None:
        return None
    task = _strip_v_prefix(task_name)
    for cand in (data_task_dir / "judge_spec" / f"_gt_desc_{task}.json",
                 data_task_dir / f"_gt_desc_{task}.json"):  # new, then legacy
        if cand.is_file():
            return _read_json_lenient(cand)
    return None


def load_per_instance_gt_desc(task_name: str, level: str, iid: str,
                              data_task_dir: Path | None) -> Any | None:
    """Read the per-instance gt_desc.json under `_raw/<level>/` (or
    `_raw/` for `no_lv`); returns parsed payload or None if absent."""
    if data_task_dir is None:
        return None
    task = _strip_v_prefix(task_name)
    raw = data_task_dir / "_raw"
    if not raw.is_dir():
        return None
    inst_id = _instance_id(task, level, iid)
    search_dirs = [raw / level, raw] if level != "no_lv" else [raw]
    for d in search_dirs:
        cand = d / f"{inst_id}_gt_desc.json"
        if cand.is_file():
            return _read_json_lenient(cand)
    return None


def gt_desc_for(task_name: str, level: str, iid: str,
                data_task_dir: Path | None,
                rubric_entry: dict | None = None) -> str:
    """Return a normalized GT description string for one instance, or ""
    when no source exists.

    Combines the task-wide and per-instance sources (see module
    docstring) — task-wide goes first, blank line, then per-instance.

    `rubric_entry` is accepted for signature compatibility with the
    earlier per-task formatter API; it is currently unused — drop a
    `_gt_desc.json` next to the instance (or task dir) instead of
    stuffing GT signal into the rubric."""
    _ = rubric_entry            # reserved; see docstring
    task_wide = _render(load_task_wide_gt_desc(task_name, data_task_dir))
    per_inst = _render(load_per_instance_gt_desc(
        task_name, level, iid, data_task_dir))
    parts = [s for s in (task_wide, per_inst) if s]
    return "\n\n".join(parts)


# ===================== judge prompts =====================
# Prompt builders + their leaf helpers (image encoding, frame
# downsizing, system prompts). Moved out of vlm_judge.py so the
# driver/agentic-loop logic is separable from the prompt text.

FRAME_VLM_MAX_SIDE = 512   # downsize each frame before sending to the VLM
REF_VLM_MAX_SIDE = 768     # input/GT images sent slightly larger


def image_to_data_url(path: Path, max_side: int | None = None) -> str:
    if max_side is None:
        mime, _ = mimetypes.guess_type(str(path))
        if mime is None:
            mime = "image/png"
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{b64}"
    from PIL import Image
    img = Image.open(path)
    w, h = img.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))),
                         Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=88)
    return ("data:image/jpeg;base64," +
            base64.b64encode(buf.getvalue()).decode("ascii"))


def _zoom_max_per_batch(batch_size: int) -> int:
    """Maximum number of frames a single round-1 batch may flag for
    closer inspection (ceil(batch_size / 2))."""
    return math.ceil(batch_size / 2)


AGENTIC_SYSTEM = (
    "You are a visual rubric judge for video generation outputs. You "
    "are judging a VIDEO (delivered as a small batch of frames sampled "
    "at a given fps), NOT a single image. Each frame is preceded by a "
    "metadata line giving the time in seconds and the original-video "
    "global frame index.\n\n"
    "Rules:\n"
    "  - Reference frames by their global_frame number when describing "
    "what you see.\n"
    "  - EVERY frame in this batch must appear in at least one "
    "evaluation's frame_indices. If a frame is uneventful, group it "
    "with the nearest informative comment instead of skipping.\n"
    "  - Flag substantive rubric violations and ignore minor noise. "
    "Substantive means the kind of failure the rubric is actually "
    "targeting (for a maze task: the car clearly clipping through or "
    "driving over a wall, a teleport or a duplicate car, a missing "
    "required turn, etc.). IGNORE minor near-rubric deviations that "
    "are NOT the kind of failure the rubric is checking for: sub-pixel "
    "jitter, a small-part sliver of overlap during fast motion, faint "
    "compression artifacts, lighting flicker, and so on.\n"
    "  - DO NOT hallucinate events between adjacent sampled frames. At "
    "the current fps, consecutive frames in this batch are separated "
    "by ~1/fps seconds of native video that you cannot see. If you "
    "SUSPECT something happened in that gap (two objects collided / "
    "phased, a piece teleported, the car briefly crossed a wall, "
    "etc.) but you cannot see it actually, you MUST list both flanking "
    "global_frame numbers in \"frames_to_inspect_closely\", so the "
    "next round samples higher fps in that gap. Do NOT write comments "
    "like \"they collide between frame X and frame Y\" or \"the object "
    "phases through between X and Y\" based on inference, that is "
    "hallucination. Only flag a collision / phase / teleport as a "
    "violation when you can point to one or more sampled frames that "
    "visibly show the overlap, the impossible state, or the object in "
    "two places at once.\n\n"
    "Each turn you are also shown the original input image and the "
    "reference ground-truth image (when available) BEFORE the sampled "
    "frames; use them as references for what the input scene contains "
    "and what one acceptable output looks like.\n\n"
    "Always reply with valid JSON matching the schema in the user "
    "message."
)


COMPLETENESS_SYSTEM = (
    "You are a visual judge for task completeness in a generated VIDEO. "
    "You are "
    "shown the INPUT image (the video's first frame / starting state), "
    "optionally a GT image and/or a GT description (one acceptable final "
    "state / goal), and the whole clip sampled as still frames in time "
    "order. Using ONLY the supplied tiered standard, decide how far the "
    "task's GOAL was actually carried out and return a single integer "
    "tier. Judge the procedure and the final state you can see across "
    "the frames; do not invent events between frames. Reply ONLY with "
    "the JSON schema given in the user message."
)


def _batch_user_message(round_idx: int, fps: int,
                        batch_idx: int, n_batches: int,
                        batch_metas: list[dict],
                        round_n_frames: int,
                        rubric_text: str, task_name: str, level: str,
                        iid: str,
                        input_image: Path | None,
                        gt_image: Path | None,
                        rubric_overall: str = "",
                        rubric_checklist: list[str] | None = None,
                        gt_desc_text: str = "") -> dict:
    n = len(batch_metas)
    gf_lo = batch_metas[0]["global_frame"]
    gf_hi = batch_metas[-1]["global_frame"]
    intro_lines = [f"Task: {task_name}", ""]
    # Explicit two-level rubric split: the main goal (rubric[0]) frames
    # the high-level task; the detailed checklist (rubric[1:]) is what
    # the judge actually scores per-item against.
    if rubric_overall:
        intro_lines += [
            "Task main goal:",
            rubric_overall.strip(),
            "",
        ]
    if rubric_checklist:
        intro_lines.append(
            "Detailed checklist (each item is scored independently; "
            "reference these 1-based indices when you flag violations):")
        for i, item in enumerate(rubric_checklist, 1):
            intro_lines.append(f"  {i}. {item}")
        intro_lines.append("")
    elif not rubric_overall:
        # No structured rubric — fall back to the rendered text blob.
        intro_lines += ["Rubric / requested behaviour:", rubric_text, ""]
    # Inject textual GT description (routed through vlm_judge_utils) BEFORE
    # the batch metadata so it sits with the other prompt context. Renders
    # only when the task is in TASKS_NEED_GT_DESC and a `_gt_desc.json`
    # resolved to a non-empty string.
    if gt_desc_text:
        intro_lines += [
            "Ground-truth description (success / final state to reach):",
            gt_desc_text.strip(),
            "",
        ]
    intro_lines.append(
        f"fps: {fps}    Batch {batch_idx + 1}/{n_batches} "
        f"({n} frames, global {gf_lo} to {gf_hi}, "
        f"t {batch_metas[0]['time_s']:.3f}s to "
        f"{batch_metas[-1]['time_s']:.3f}s, "
        f"out of {round_n_frames} frames at this fps)")

    refs_present = []
    if input_image and Path(input_image).is_file():
        refs_present.append("the original input image")
    if gt_image and Path(gt_image).is_file():
        refs_present.append("the reference ground-truth image")
    if refs_present:
        below_line = (
            "Below: " + " then ".join(refs_present) +
            f", followed by THIS BATCH's sampled video frames "
            f"({n} frames at {fps} fps, each preceded by a metadata "
            f"block).")
    else:
        below_line = (
            f"Below: this batch's sampled video frames ({n} frames at "
            f"{fps} fps), each preceded by a metadata block giving "
            f"time (seconds) and original-video global frame index.")

    max_flag = _zoom_max_per_batch(n)
    schema_lines = [
        "Output STRICT JSON (no markdown, no prose outside JSON):",
        "{",
        '  "evaluations": [',
        '    {"frame_indices": [<global_frame>, ...], '
        '"comment": "<one or two sentences>"}',
        "  ],",
        f'  "frames_to_inspect_closely": [<global_frame>, ...]'
        f'   // 0..{max_flag} entries; can be []',
        "}",
    ]

    content: list[dict] = [{"type": "text", "text": "\n".join(intro_lines)}]
    # `Below: ...` is glued to the images it describes so the model sees
    # the description immediately followed by the input / GT picture(s)
    # and the sampled frames.
    content.append({"type": "text", "text": below_line})
    if input_image and Path(input_image).is_file():
        content.append({"type": "image_url",
                        "image_url": {"url": image_to_data_url(
                            Path(input_image), REF_VLM_MAX_SIDE)}})
    if gt_image and Path(gt_image).is_file():
        content.append({"type": "image_url",
                        "image_url": {"url": image_to_data_url(
                            Path(gt_image), REF_VLM_MAX_SIDE)}})
    for m in batch_metas:
        content.append({"type": "text",
                        "text": f"t={m['time_s']:.3f}s  "
                                f"global_frame={m['global_frame']}"})
        content.append({"type": "image_url",
                        "image_url": {"url": image_to_data_url(
                            Path(m["path"]), FRAME_VLM_MAX_SIDE)}})
    # Output schema + rules end up AFTER all the imagery so the model
    # answers with the format requirements freshest in context.
    content.append({"type": "text", "text": "\n".join(schema_lines)})
    return {"role": "user", "content": content}


def _polish_messages(rubric_text: str, task_name: str, level: str, iid: str,
                     duration_s: float, native_fps: float,
                     frame_evals: dict[str, list[str]],
                     batch_summaries: list[dict],
                     rubric_checklist: list[str] | None = None,
                     rubric_overall: str = "",
                     gt_desc_text: str = "") -> list[dict]:
    lines = [
        "You are finalising the rubric violation counts for a VIDEO. The "
        "per-batch evidence below was collected from multiple LLM calls, "
        "each shown a contiguous window of sampled frames (round 1 at low "
        "fps over the whole clip; round 2 at high fps zoomed into the "
        "frames round 1 flagged). Your job is to deduplicate that evidence "
        "into a final violation count per rubric checklist item.",
        "",
    ]
    if rubric_overall:
        lines += [
            "Task main goal:",
            rubric_overall.strip(),
            "",
        ]
    if rubric_checklist:
        lines.append(
            "Detailed checklist (reference these 1-based indices):")
        for i, item in enumerate(rubric_checklist, 1):
            lines.append(f"  {i}. {item}")
        lines.append("")
    elif not rubric_overall:
        # No structured rubric — fall back to rendered text.
        lines += ["Rubric / requested behaviour:", rubric_text, ""]
    # Merge each batch's preliminary verdict with that same batch's
    # per-range comments (looked up by the `r<R>b<B>` prefix the agentic
    # loop bakes into every comment line). Each comment line still
    # carries its `t=...s` annotation, so we strip the redundant prefix
    # before re-printing under the batch header.
    def _range_sort_key(k: str):
        a, _, b = k.partition("-")
        return (int(a), int(b))

    sorted_rngs = sorted(frame_evals, key=_range_sort_key)
    lines += [
        "Per-batch evidence (each batch saw a contiguous window at the "
        "given fps; consecutive batches share their boundary frame as "
        "overlap):",
    ]
    for bs in batch_summaries:
        gr = bs.get("global_frame_range") or [None, None]
        tr = bs.get("time_s_range") or [0.0, 0.0]
        lines.append(
            f"  - round {bs['round']} fps={bs['fps']} "
            f"batch {bs['batch_idx'] + 1} "
            f"(global {gr[0]}-{gr[1]}, "
            f"t {tr[0]:.3f}-{tr[1]:.3f}s):")
        prefix = f"r{bs['round']}b{bs['batch_idx']} "
        bc_lines: list[str] = []
        for rng in sorted_rngs:
            for c in frame_evals[rng]:
                if c.startswith(prefix):
                    bc_lines.append(
                        f"      - global_frames {rng}: "
                        f"{c[len(prefix):]}")
        if bc_lines:
            lines.append("      comments:")
            lines.extend(bc_lines)
        else:
            lines.append("      comments: (none)")
    lines += [
        "",
        "Tolerance: only count something as a violation if it is "
        "defensible to a human reviewer as a real failure; ignore minor "
        "near-rubric noise (sub-pixel jitter, a one-pixel sliver during "
        "fast motion, lighting flicker, micro-drift). A borderline / "
        "cosmetic deviation does NOT count.",
    ]
    if rubric_checklist:
        n_items = len(rubric_checklist)
        lines += [
            "",
            "Output STRICT JSON only:",
            "{",
            '  "by_range": [{"global_frames": "<a-b>", "summary": "<one '
            'sentence>"}, ...],',
            '  "checklist": [',
            f'    {{"index": <1..{n_items}>, "n_violations": <int >= 0>, '
            f'"evidence": "<one sentence citing time ranges>"}}',
            f'    // EXACTLY {n_items} entries, indexes 1..{n_items} in '
            f'order',
            "  ],",
            '  "overall_reasoning": "<two or three sentences>"',
            "}",
            "",
            "Per-item rule: n_violations is the FINAL deduplicated count "
            "of DISTINCT substantive violations of that item across the "
            "whole video, after integrating evidence from ALL batches "
            "(round 1 + round 2). Deduplication: (a) a persistent "
            "violation spanning a continuous window counts as ONE; (b) "
            "two reports in adjacent batches within 0.5 s across the "
            "boundary count as ONE; (c) the same incident in two "
            "overlapping batches is ONE. Per-item score = "
            "1/(n_violations+1); rubric_score is the arithmetic mean "
            "over items.",
        ]
    else:
        lines += [
            "Output STRICT JSON only:",
            "{",
            '  "by_frame": [{"global_frame": <int>, "time_s": <float>, '
            '"summary": "<one sentence>"}, ...],',
            '  "overall_reasoning": "<two or three sentences summarising '
            'the substantive violations found, with time ranges>"',
            "}",
        ]
    return [
        {"role": "system",
         "content": ("You are a careful video-judging assistant. "
                     "You are finalising the per-rubric-item violation "
                     "counts for a VIDEO based on sampled-frame evidence "
                     "collected by per-batch judges.")},
        {"role": "user", "content": "\n".join(lines)},
    ]


def _completeness_messages(task_name: str, level: str, iid: str,
                           frame_metas: list[dict], completeness_text: str,
                           input_image: Path | None,
                           gt_image: Path | None,
                           gt_desc_text: str = "") -> list[dict]:
    """One user message: input image, optional GT image and/or GT
    description, all sampled frames in order, and the tiered completeness
    standard. Asks for {tier:0|1|2, reasoning}."""
    has_input = bool(input_image and Path(input_image).is_file())
    has_gt_img = bool(gt_image and Path(gt_image).is_file())
    has_gt_desc = bool(gt_desc_text and gt_desc_text.strip())
    order = []
    if has_input:
        order.append("the INPUT image (starting state at t=0)")
    if has_gt_img:
        order.append("a GT image (one acceptable final state)")
    order.append("then the sampled video frames in time order")
    desc_block = ""
    if has_gt_desc:
        desc_block = ("\n\nGround-truth description (the goal / acceptable "
                      f"final state):\n\"\"\"\n{gt_desc_text.strip()}\n\"\"\"")
    head = (
        f"Task: {task_name}    Level: {level}    Instance: {iid}\n"
        f"You will be shown: {'; '.join(order)}.\n\n"
        f"Tiered standard (score by how far the goal was carried out):\n"
        f"\"\"\"\n{completeness_text.strip()}\n\"\"\""
        f"{desc_block}"
    )
    content: list[dict] = [{"type": "text", "text": head}]
    if has_input:
        content.append({"type": "image_url", "image_url": {"url":
                        image_to_data_url(Path(input_image), REF_VLM_MAX_SIDE)}})
    if has_gt_img:
        content.append({"type": "image_url", "image_url": {"url":
                        image_to_data_url(Path(gt_image), REF_VLM_MAX_SIDE)}})
    for m in frame_metas:
        content.append({"type": "text",
                        "text": f"t={m['time_s']:.3f}s  "
                                f"global_frame={m['global_frame']}"})
        content.append({"type": "image_url", "image_url": {"url":
                        image_to_data_url(Path(m["path"]), FRAME_VLM_MAX_SIDE)}})
    content.append({"type": "text", "text":
                    "Reply ONLY with valid JSON:\n"
                    '  {"tier": 0|1|2, "reasoning": "1-2 sentences"}\n'
                    "tier is the single integer from the tiered standard "
                    "above (0, 1, or 2)."})
    return [{"role": "system", "content": COMPLETENESS_SYSTEM},
            {"role": "user", "content": content}]


__all__ = [
    # task GT signal classification
    "TASKS_GT_IMAGE_AND_DESC", "TASKS_GT_IMAGE_ONLY", "TASKS_GT_DESC_ONLY",
    "TASKS_NEED_GT_IMAGE", "TASKS_NEED_GT_DESC",
    "task_gt_signals",
    # GT description loaders
    "gt_desc_for",
    "load_task_wide_gt_desc", "load_per_instance_gt_desc",
    # prompt builders + helpers
    "FRAME_VLM_MAX_SIDE", "REF_VLM_MAX_SIDE",
    "image_to_data_url",
    "AGENTIC_SYSTEM", "COMPLETENESS_SYSTEM",
]
