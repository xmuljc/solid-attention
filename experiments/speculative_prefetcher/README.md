# Speculative Prefetcher Microbenchmark

This directory isolates the SolidAttention speculative prefetcher mechanism.
It does not modify `llama.cpp`, does not run Llama, and does not implement
I/O-Progressive Attention.

The experiment answers two separate questions:

1. Does `S[layer,t-1]` predict `S[layer,t]` better than random selection?
2. Do correctly predicted KV blocks arrive at GPU memory before the attention
   deadline, reducing correction blocking latency?

## Inputs

- Model: local `Qwen-2.5-7B`
- Trace: `experiments/figure9_reproduction/results/selected_blocks_trace.jsonl`
- Test file: existing non-sparse 2 GiB NVMe file from the previous I/O
  experiment
- Dynamic selected budget: `K=16`, block size `32` tokens

The available Qwen trace is a smoke trace: 8 workloads, 3 prompts per workload,
32 decode steps, 28 layers, 28 query heads. It is real autoregressive selected
block trace, but it is not the full 20-prompt Figure 9 run.

## Reproduction Commands

```bash
cd "/data/disk2/ljc/solid attention"

python3 experiments/speculative_prefetcher/scripts/audit_inputs.py \
  --config experiments/speculative_prefetcher/experiment_config.json

python3 experiments/speculative_prefetcher/scripts/compute_prediction_metrics.py \
  --config experiments/speculative_prefetcher/experiment_config.json

./experiments/speculative_prefetcher/build.sh

python3 experiments/speculative_prefetcher/scripts/make_replay_events.py \
  --config experiments/speculative_prefetcher/experiment_config.json \
  --mode smoke

experiments/speculative_prefetcher/bin/speculative_prefetch_replay \
  --file experiments/io_progressive_attention/data/qwen_kv_finite_2g.bin \
  --events experiments/speculative_prefetcher/results/replay_events_smoke.csv \
  --output experiments/speculative_prefetcher/results/raw_block_timing_smoke.jsonl \
  --queue-depth 4

python3 experiments/speculative_prefetcher/scripts/analyze_replay.py \
  --config experiments/speculative_prefetcher/experiment_config.json \
  --raw experiments/speculative_prefetcher/results/raw_block_timing_smoke.jsonl

python3 experiments/speculative_prefetcher/scripts/plot_results_chinese.py \
  --config experiments/speculative_prefetcher/experiment_config.json \
  --font experiments/io_progressive_attention/assets/fonts/NotoSansCJKsc-Regular.otf
```

## Boundary

The smoke replay is intended to validate the harness and produce preliminary
mechanism evidence. It must not be described as full paper-level performance
reproduction unless the formal 1000-trial matrix is run.
