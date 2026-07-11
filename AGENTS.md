# SolidAttention Reproduction Agent Rules

This project reproduces the core mechanisms of SolidAttention with a
harness-first engineering workflow.

## Scope Rules

- Do not start by implementing a full LLM runtime.
- Phase 1 focuses on simulation and harness-based reproduction.
- Do not make large llama.cpp kernel changes in Phase 1.
- Real model integration should later be based on llama.cpp.
- Real SSD I/O should later use liburing.
- Phase 1 may use mock I/O latency and simulated storage movement.
- Do not implement real attention in Phase 1.

## Engineering Rules

- Every core module must have unit tests.
- Every scheduler must emit trace events.
- Every experiment must emit JSON metrics.
- Prefer small, testable modules over large runtime changes.
- Keep state transitions explicit and observable.
- Keep outputs reproducible: record configs, traces, metrics, and run metadata.

## Verification Profiles

- Every change must pass `python -m tools.check fast` on its development host.
- Linux I/O changes must also pass `python -m tools.check io`.
- liburing changes must pass `python -m tools.check liburing`; fallback is not success.
- llama.cpp patch changes must pass `python -m tools.check llama` at the pinned submodule revision.
- Required profiles may not use skip or fallback to report success.
- Generated outputs belong under ignored `outputs/`; deterministic inputs belong under `tests/fixtures/`.
- Experiment entry points must emit `solidattention.run_manifest.v1`.

## Project Layout

```text
core/
  KV block and storage abstractions used by simulations and future runtimes.

harness/
  Trace recording, metrics collection, experiment execution, and reporting.

tests/
  Unit tests for core mechanisms and harness behavior.

configs/
  Experiment, hardware, model, and method configuration files.

outputs/
  Ignored generated traces, metrics, benchmark results, and reports.

docs/
  Design notes, reproduction roadmap, and implementation decisions.
```

## Phase Plan

### Phase 1: Simulation Harness

- Implement KV block metadata, locations, roles, and state transitions.
- Implement DRAM/VRAM/SSD block-store simulation.
- Implement trace recording for all state changes.
- Implement JSON metrics collection.
- Add unit tests for all core modules.

### Phase 2: liburing-based SSD Block I/O Prototype

- Add a standalone C/C++ SSD block I/O prototype using liburing.
- Measure read/write latency, queue depth effects, and block-size tradeoffs.
- Keep it separate from llama.cpp until the API is stable.

### Phase 3: llama.cpp KV Cache Integration

- Identify minimal llama.cpp KV cache integration points.
- Add instrumentation before behavioral changes.
- Integrate the block-store abstraction without rewriting the runtime.

### Phase 4: SolidAttention-style Scheduler Integration

- Add Init Blocks, Local Blocks, and Selected Blocks selection.
- Add representative-vector based block selection.
- Add speculative prefetch and SSD-aware scheduling.
- Add interleaved KV layout simulation, then map it to runtime storage.

### Phase 5: Ablation Experiments and Metrics

- Compare memory-only, SSD-sync, SSD-async, prefetch, and full scheduler modes.
- Produce JSON metrics for every experiment.
- Produce trace files for scheduler behavior.
- Summarize latency, bandwidth, cache residency, and movement costs.
