# Figure 9 Results

This directory contains a Qwen-2.5-7B-only real selected-block similarity smoke run.

Actual run scale:

- model: Qwen-2.5-7B
- workloads: 8
- samples per workload: 3
- max_new_tokens: 32
- context budget: 1024 tokens
- block size: 32 tokens
- dynamic Top-K: 16 blocks
- recorded layers: 28
- recorded query heads: 28

Generated files:

- `similarity_summary.csv`
- `selection_similarity_values.csv`
- `selection_similarity_extended.png/pdf`
- `selection_similarity_extended_with_ci.png/pdf`
- `figure9_anchor_reproduction.png/pdf`
- `qwen_selected_blocks_run_manifest.json`
- `plot_validation.json`

Large local-only files `selected_blocks_trace.jsonl` and `similarity_raw.jsonl` were not uploaded to GitHub.
This is not the full Llama+Qwen 20-sample main experiment.
