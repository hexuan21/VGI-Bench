# `data/` — benchmark data

The benchmark itself is not in this git repo; it lives on the Hugging Face
dataset [`hexuan21/VGI-Bench`](https://huggingface.co/datasets/hexuan21/VGI-Bench),
under `bench_data/`. Download it into this directory:

```bash
# from the repo root -- note `--local-dir data`, so it lands as data/bench_data/
hf download hexuan21/VGI-Bench --repo-type dataset \
    --include "bench_data/*" --local-dir data
```

That leaves you with:

```
data/
  task_names.yaml                 (already in the repo — task -> domain map)
  bench_data/
    <task>/
      input_image/lv{1,2,3}/<task>_lv<N>_<id>.png    input image = first frame
      prompt/video_gen_prompt/<task>.txt             prompt for video-gen models
      prompt/image_gen_prompt/<task>.txt             prompt for image-gen models / UMM (optional)
      judge_spec/_judge_rubric_<task>.json           VLM-as-judge checklist
      judge_spec/_completeness_judge.txt             completeness scale (0 / 1 / 2)
```

27 tasks x 3 difficulty levels x 10 instances. An instance is one input image
(16:9, also the first frame for video models) plus one text prompt. Tasks whose
process cannot be expressed in a single image ship no `image_gen_prompt/`.

`vlm_judge.py` looks for task directories under `data/bench_data/`, so keep the
layout above and no configuration is needed.

## Ground-truth images

The reference answer images (`*_gt.png`) and the solution metadata are **not
published** — releasing them would make the benchmark trivially gameable. The
judge treats them as optional and scores against the rubric checklist alone when
they are absent, which is the intended public setup.
