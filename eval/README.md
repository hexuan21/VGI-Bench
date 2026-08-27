# `eval/` — model outputs and the VLM judge

## 0. Requirements

`pip install requests pillow`, plus **ffmpeg and ffprobe on your `PATH`** — the
judge shells out to them to pull frames from the generated videos. Image-only
models need neither.

## 1. Put the generations here

`eval_res/` holds what a model produced for every instance, one directory per
model:

```
eval/eval_res/<model>/<task>/<model>_<task>_<lv>_<id>.mp4     video models
eval/eval_res/<model>/<task>/<model>_<task>_<lv>_<id>.png     image models / UMM
```

Either run your own model over `data/bench_data/` and write the files out in
that shape, or pull the generations we evaluated from the Hugging Face dataset:

```bash
# from the repo root — one model, e.g. sora2
hf download hexuan21/VGI-Bench --repo-type dataset \
    --include "eval_res/video_model_res/sora2/*" --local-dir /tmp/vgi
mkdir -p eval/eval_res/sora2 && cp -r /tmp/vgi/eval_res/video_model_res/sora2/* eval/eval_res/sora2/
```

Note the filename difference: the published copies drop the `<model>_` prefix
(the model name is already the directory), while `vlm_judge.py` accepts the file
with or without it.

## 2. Add an API key

The judge calls a VLM. Put your key in `const/_api_key.json` at the repo root:

```json
{ "NETMIND_API_KEY": "...", "GOOGLE_API_KEY": "...", "OPENROUTER_API_KEY": "..." }
```

Only the key for the provider you actually use is required. The default judge is
`google/gemini-3-flash-preview` via NetMind.

## 3. Run the judge

```bash
python eval/vlm_judge.py <model>                    # every task
python eval/vlm_judge.py <model> --task maze_square # one task
```

Results:

```
eval/judge_res/<model>/_judge_<task>_<model>.json   per-task trace
eval/judge_res/<model>/_all_judge_<model>.json      aggregate
```

`Final = Completeness x Rubric`, and an instance counts as a success only when
both are perfect. See the top-level README for what the judge actually does and
for the ablation flags.
