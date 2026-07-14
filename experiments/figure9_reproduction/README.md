# Figure 9 Reproduction Experiment

This directory contains the independent experiment harness for reproducing the Figure 9 mechanism: whether dynamic Selected Blocks for adjacent decode steps in the same layer have high overlap.

This workspace now has a Qwen-2.5-7B Hugging Face runtime path for generating real selected-block traces. Llama-3.1-8B is still missing, so Qwen-only figures must not fabricate a Llama bar.

## Required Trace Schema

The model runner must emit one JSONL record per sample/head/layer/decode step:

```json
{"model":"Llama-3.1-8B","dataset":"gov_report","sample_id":"gov_report/0","head_id":0,"layer_id":0,"decode_step":1,"selected_blocks":[1,2],"init_blocks":[0],"local_blocks":[31]}
```

Only `selected_blocks` are used for the main overlap@K. `init_blocks` and `local_blocks` are only used for the supplemental all-active overlap.

## Commands

Generate the fixed RULER alternating multi-key task:

```bash
python3 experiments/figure9_reproduction/scripts/make_ruler_multikey_alternating.py \
  --output experiments/figure9_reproduction/ruler_multikey_alternating.jsonl \
  --samples 20 --seed 20260714 --context-length 16000 \
  --num-needle-k 8 --num-needle-q 8
```

Generate Qwen selected-block records:

```bash
.deps/figure9_qwen_venv/bin/python experiments/figure9_reproduction/scripts/run_qwen_selected_trace.py \
  --model-dir .deps/models/Qwen2.5-7B \
  --datasets all \
  --samples-per-dataset 3 \
  --max-new-tokens 32 \
  --context-budget 1024 \
  --block-size 32 \
  --top-k 16 \
  --output experiments/figure9_reproduction/results/selected_blocks_trace.jsonl
```

Compute similarity from the generated trace:

```bash
.deps/figure9_qwen_venv/bin/python experiments/figure9_reproduction/scripts/compute_similarity.py \
  --input experiments/figure9_reproduction/results/selected_blocks_trace.jsonl \
  --output-dir experiments/figure9_reproduction/results \
  --k 16 --bootstrap-rounds 2000 --seed 20260714 \
  --models Qwen-2.5-7B
```


Plot the Figure 9 anchor reproduction for the four paper workloads:

```bash
.deps/figure9_qwen_venv/bin/python experiments/figure9_reproduction/scripts/plot_selection_similarity.py \
  --csv experiments/figure9_reproduction/results/selection_similarity_values.csv \
  --output-dir experiments/figure9_reproduction/results --anchor-only \
  --models Qwen-2.5-7B
```

Plot the extended PPT figure and the CI figure:

```bash
.deps/figure9_qwen_venv/bin/python experiments/figure9_reproduction/scripts/plot_selection_similarity.py \
  --csv experiments/figure9_reproduction/results/selection_similarity_values.csv \
  --output-dir experiments/figure9_reproduction/results \
  --models Qwen-2.5-7B

.deps/figure9_qwen_venv/bin/python experiments/figure9_reproduction/scripts/plot_selection_similarity.py \
  --csv experiments/figure9_reproduction/results/selection_similarity_values.csv \
  --output-dir experiments/figure9_reproduction/results --with-ci \
  --models Qwen-2.5-7B
```

Validate outputs:

```bash
.deps/figure9_qwen_venv/bin/python experiments/figure9_reproduction/scripts/validate_selection_similarity.py \
  --results-dir experiments/figure9_reproduction/results \
  --models Qwen-2.5-7B
```

## Required Final Files

When the real model run is available, the commands above generate:

- `results/similarity_raw.jsonl`
- `results/similarity_summary.csv`
- `results/selection_similarity_values.csv`
- `results/selection_similarity_extended.png`
- `results/selection_similarity_extended.pdf`
- `results/selection_similarity_extended_with_ci.png`
- `results/selection_similarity_extended_with_ci.pdf`

`figure9_anchor_reproduction.png/pdf` should be produced by running the same plotting script on the four anchor workloads only.

## Current Status

Current workspace status:

- Qwen-2.5-7B checkpoint is present at `.deps/models/Qwen2.5-7B`.
- Python runtime is present at `.deps/figure9_qwen_venv`.
- LongBench datasets can be loaded with `datasets==2.21.0`.
- Llama-3.1-8B checkpoint is still missing.
- Qwen selected-block trace generation is implemented in `scripts/run_qwen_selected_trace.py`.
- A Qwen-only smoke run has completed with 3 samples per workload and `max_new_tokens=32`.
- The generated figures are valid Qwen-only mechanism evidence, not the original 2-model/20-sample main experiment.

The reference style image has been downloaded to `reference/selection_similarity_target_style.png` and verified as a 855x249 PNG, aspect ratio 3.434.
