# SolidAttention Harness Roadmap

The reproduction strategy is harness-first: create a measurable simulation
environment, validate each core mechanism in isolation, then attach real SSD I/O
and llama.cpp integration points.

The harness baseline is reproduced through `python -m tools.check fast`; Linux
I/O and llama.cpp work use the dedicated `io`, `liburing`, and `llama` profiles.
Generated run evidence belongs under ignored `outputs/` directories or CI
artifacts.

## Phase 1: Simulation Harness

Deliverables:

- KV block metadata and location state management.
- Simulated DRAM/VRAM/SSD block store.
- Trace recorder for state changes and future scheduler events.
- Metrics collector that exports JSON.
- Init/Local/Selected block selection.
- Representative-vector based block ranking.
- Speculative prefetch simulation and evaluation.
- SSD-aware scheduling simulation.
- Interleaved KV layout simulation.
- Executable smoke harness with JSON trace and metrics outputs.
- Unit tests for core behavior.

Out of scope:

- Real attention kernels.
- Large llama.cpp modifications.
- liburing integration.

## Phase 2: liburing-based SSD Block I/O Prototype

Deliverables:

- Standalone block read/write API.
- Build-time liburing detection with POSIX fallback for dependency-limited hosts.
- JSON latency metrics for block read/write smoke probes.
- Compatibility notes for future Python harness integration.

## Phase 3: llama.cpp KV Cache Integration

Deliverables:

- Integration scaffold for future llama.cpp KV instrumentation events.
- Probe for validating a local llama.cpp checkout before patching.
- KV cache mapping from llama.cpp-style events to harness block metadata.
- Opt-in `src/llama-kv-cache.cpp` JSONL trace patch verified with `git apply --check`.
- Synthetic patched-target trace, command-bridge, staging, mutation-gate, and
  attention-view residency verification at the pinned submodule revision.

## Phase 4: SolidAttention-style Scheduler Integration

Deliverables:

- Init Blocks, Local Blocks, and Selected Blocks scheduling.
- Representative-vector based block selection.
- Speculative prefetch.
- SSD-aware scheduling.
- Interleaved KV layout support.

Still pending:

- Scheduler-timed load and prefetch before attention consumes `get_k` / `get_v`
  views.
- Post-decode persistence and eviction of newly written KV rows.
- Real scheduler-selected KV contents and measured liburing latency in the
  runtime movement path.

## Phase 5: Ablation Experiments and Metrics

Deliverables:

- Reproducible experiment configs.
- JSON metrics and JSON traces for every run.
- Simulation ablation harness for memory-only, SSD-sync, SSD-async, prefetch-only, and full scheduler modes.
- Ablation tables for cache policy, prefetch, SSD scheduling, and layout.

Real-model work starts only after the harness baseline and a future
load-before-attention vertical slice are complete.
