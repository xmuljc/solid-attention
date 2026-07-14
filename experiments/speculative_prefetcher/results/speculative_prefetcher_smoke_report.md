# Speculative Prefetcher Smoke Report

## Scope

This run validates the SolidAttention speculative prefetcher mechanism on the
available Qwen-2.5-7B selected-block trace. It is an isolated microbenchmark:
no Llama, no llama.cpp runtime modification, and no I/O-Progressive Attention.

The available trace is a smoke trace with 8 workloads, 3 prompts per workload,
32 decode steps, 28 layers, and 28 query-head selection units. It does not cover
the newly requested Qasper, 2WikiMQA, TriviaQA, or HotpotQA traces.

## Qwen KV Shape

- num layers: 28
- query heads: 28
- KV heads: 4
- head dim: 128
- block tokens: 32
- KV dtype bytes: 2
- K+V block bytes: 65,536

## Question 1: Is History-Last1 Better Than Random?

Yes, on the available Qwen smoke trace.

History-Last1 semantic recall is about 83.8-86.3% across workloads, while
Random-Same-Volume is about 53.3%. Prompt-paired bootstrap differences are
positive for every available workload, about +30 to +33 percentage points.

Past-Frequency is slightly better than History-Last1 on this trace, by about
1.7 to 3.5 percentage points. This means the trace supports temporal locality,
but Last1 is not the best policy among the tested simple predictors.

## Question 2: Are Correct Predictions Timely?

Yes in this smoke timing replay, under the configured 1.0 ms prefetch lead and
QD=4.

For workload replay:

- No-Selected-Prefetch mean T_block: 0.369 ms
- Random-Same-Volume mean T_block: 0.204 ms
- History-Last1 mean T_block: 0.096 ms
- Past-Frequency mean T_block: 0.086 ms
- Oracle mean T_block: 0.000 ms

History-Last1 mean on-time recall is 85.1%, close to its semantic recall in the
sampled replay. Prompt-level on-time recall and latency saving correlation is
0.917.

## Boundary

This is a smoke result. It supports the mechanism on the available Qwen trace:
using the previous same-layer decode step for selected-block prefetching beats
random and reduces correction blocking latency when the prefetch lead is
sufficient.

It is not a full paper-level reproduction because the formal 1000-trial matrix
and the newly requested LongBench workload traces have not been run.
