# VGI-Bench: Probing Visual Intelligence in Video Generation Models

<p align="center">
  <a href="https://hexuan21.github.io/VGI-Bench/"><img src="https://img.shields.io/badge/Project-Page-blue?logo=googlechrome&logoColor=white" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2608.19583"><img src="https://img.shields.io/badge/arXiv-2608.19583-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/hexuan21/VGI-Bench"><img src="https://img.shields.io/badge/HF-Data%20%26%20Res-yellow?logo=huggingface&logoColor=black" alt="HF Data & Res"></a>
  <a href="https://x.com/XuanHe21/status/2093029571343839603"><img src="https://img.shields.io/badge/X-Tweet-black?logo=x&logoColor=white" alt="X / Tweet"></a>
</p>

## Abstract

Recent studies suggest that video generative models can exhibit certain forms of
zero-shot visual reasoning through generated frames. Yet reliable evaluation
remains challenging: benchmarks should adopt inputs aligned with the visual
priors of current video models, require valid evolving processes rather than only
plausible final states, and calibrate task difficulty to remain challenging yet
partly feasible. To this end, we introduce **VGI-Bench**, containing 27 tasks and
810 instances, organized by a two-level taxonomy of task domains and skill tags
for fine-grained evaluation of visual reasoning capabilities of video generative
models. Our evaluations show that current generative systems can solve a subset
of visually grounded reasoning tasks, but remain far from reliable, with even the
strongest model, Seedance 2.0, achieving only 51.0% under our evaluation
criteria. Our analysis further explores the output failure modes, input condition
sensitivity, performance transfer boundary from synthetic fine-tuning, and
internal denoising perspective revealing limited self-correction, where later
steps mainly refine early hypotheses rather than correct reasoning errors. We
hope VGI-Bench will help stimulate the development of next-generation video
generative models.

## Leaderboard

Aggregated final score (`rubric_score × completeness_score`), reported overall
and per task domain (averaged over the easy / mid / hard buckets). Models are
ranked by overall score; **bold** marks the best score in each column. Full
per-difficulty numbers are on the
[project page](https://hexuan21.github.io/VGI-Bench/).

| # | Model | Type | Overall | Visual Org. | Spatiotemporal | Structured Puzzles | Physical Manip. |
|---|-------|------|:-------:|:-----------:|:--------------:|:------------------:|:---------------:|
| 1 | Seedance 2.0 | Commercial | **51.0** | **60.8** | **45.3** | 44.6 | **56.0** |
| 2 | MiniMax-H3 | Open Source | 44.4 | 41.6 | 40.3 | **50.6** | 44.4 |
| 3 | Kling 3.0 | Commercial | 44.0 | 52.9 | 36.5 | 37.5 | 52.5 |
| 4 | Sora 2 | Commercial | 36.7 | 43.9 | 34.8 | 29.5 | 40.8 |
| 5 | Gen-4.5 | Commercial | 36.6 | 46.9 | 34.8 | 22.9 | 45.0 |
| 6 | Wan 2.7 | Commercial | 35.7 | 33.9 | 36.8 | 24.5 | 47.1 |
| 7 | Veo 3.1 | Commercial | 32.0 | 45.9 | 22.3 | 22.4 | 42.5 |
| 8 | Wan 2.2 | Open Source | 21.6 | 21.4 | 30.2 | 10.4 | 24.4 |
| 9 | HunyuanVideo-1.5 | Open Source | 19.1 | 23.1 | 28.7 | 8.9 | 17.0 |

## Usage

The bulk data is not in git. Fetch the benchmark into `data/` and the model
generations into `eval/eval_res/` -- [`data/README.md`](data/README.md) and
[`eval/README.md`](eval/README.md) give the exact commands and layout:

```bash
pip install requests pillow           # plus ffmpeg/ffprobe on PATH, for video
hf download hexuan21/VGI-Bench --repo-type dataset \
    --include "bench_data/*" --local-dir data     # -> data/bench_data/<task>/
```

Run the judge model on each instance, reading your model's outputs from
`eval/eval_res/<model>/<task>/<model>_<task>_<lv>_<id>.mp4|png`:

```bash
python eval/vlm_judge.py <model>                    # judge every task
python eval/vlm_judge.py <model> --task maze_square # just one task
```

For each instance the judge sees the input image, the ground-truth image and the
model's output, and scores it against that task's
`judge_spec/_judge_rubric_<task>.json` checklist. Videos go through an agentic
loop: a first pass samples the clip at 4 fps in overlapping 10-frame windows and
nominates frames worth a closer look, a second pass re-samples those at 8 fps,
and a text-only pass deduplicates the findings into a rubric score. A separate
2 fps pass rates completeness (how far the goal was reached: 0 / 1 / 2).

`Final = Completeness x Rubric`, and an instance counts as a **success** only
when both are perfect. Per-task traces land in
`eval/judge_res/<model>/_judge_<task>_<model>.json` and the aggregate in
`eval/judge_res/<model>/_all_judge_<model>.json`.

Ground-truth images are deliberately not published; the judge treats them as
optional and scores against the rubric checklist alone.

The judge defaults to `google/gemini-3-flash-preview`; put your API key in
`const/_api_key.json`. Ablation flags (`--sampling even`, `--even_fps`,
`--batching all_at_once`, `--no_completeness`, `--no_rubric`) write to separate
`_abl_*` files so they never clobber the full-method result.

## Citation

```bibtex
@misc{he2026vgibenchprobingvisualintelligence,
      title={VGI-Bench: Probing Visual Intelligence in Video Generation Models},
      author={Xuan He and Cong Wei and Yuhao Cheng and Linrui Ma and Yuxuan Zhang and Zuojun Li and Yuhao Wen and Jize Jiang and Zeyi Liu and Yuren Hao and Songcheng Cai and Keming Wu and Penghui Du and Kai Zou and Rui Yang and Chenkai Sun and Ke Yang and Ping Nie and Kelsey R Allen and Chenglong Wang and Michel Galley and Jianfeng Gao and ChengXiang Zhai},
      year={2026},
      eprint={2608.19583},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2608.19583},
}
```
