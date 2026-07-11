# Phase 4 Runtime Integration Map

This document records the current SolidAttention-to-llama.cpp runtime boundary.
It is deliberately narrower than a full production integration: the goal is to
identify patchable points, keep the command contract testable, and avoid large
attention-kernel changes until the movement path is verified.

## Verified Contract

The paths below are generated artifact locations, not tracked source files.
Counts and backend results are historical evidence from the imported snapshot;
current runs publish ignored profile outputs as CI artifacts.

- Command schema: `solidattention.llama_cpp.kv_command.v1`
- llama.cpp dry-run summary schema: `solidattention.llama_cpp.kv_command_dry_run.v1`
- llama.cpp live bridge schema: `solidattention.llama_cpp.kv_command_bridge.v1`
- Bridge trace summary schema: `solidattention.llama_cpp.kv_command_bridge_summary.v1`
- Integration map schema: `solidattention.llama_cpp.integration_map.v1`
- KV byte-range verifier schema: `solidattention.llama_cpp.kv_byte_ranges.v1`
- Command stream source: `outputs/phase4/runtime_replay_evict_trace_locations/runtime_kv_commands.jsonl`
- C consumer prototype: `io/runtime_kv_command_consumer`
- Historically verified backend: `liburing`
- Runtime dry-run output: `outputs/phase4/llama_command_dry_run/runtime_kv_command_dry_run_summary.json`
- Runtime bridge trace: `outputs/phase4/llama_command_bridge/runtime_kv_command_bridge_trace.jsonl`
- Runtime bridge summary: `outputs/phase4/llama_command_bridge/runtime_kv_command_bridge_summary.json`
- Runtime shadow bridge summary: `outputs/phase4/llama_shadow_bridge/runtime_kv_shadow_bridge_summary.json`
- Runtime movement intent bridge summary: `outputs/phase4/llama_movement_intent_bridge/runtime_kv_movement_intent_bridge_summary.json`
- Runtime movement intent plan: `outputs/phase4/llama_movement_intent_bridge/runtime_kv_movement_intent_plan.json`
- Runtime movement intent execution metrics: `outputs/phase4/movement_intent_execution/movement_execution_metrics.json`
- Runtime residency execution metrics: `outputs/phase4/runtime_residency_execution/runtime_residency_metrics.json`
- Runtime residency verification metrics: `outputs/phase4/runtime_residency_execution/verification_metrics.json`
- Runtime KV byte-range metrics: `outputs/phase4/kv_byte_range_verification/kv_byte_range_metrics.json`
- Runtime KV byte-range verification metrics: `outputs/phase4/kv_byte_range_verification/verification_metrics.json`
- Runtime executor stub summary: `outputs/phase4/llama_executor_stub/runtime_kv_executor_stub_summary.json`
- Runtime executor stub trace summary: `outputs/phase4/llama_executor_stub/runtime_kv_executor_stub_trace_summary.json`
- Runtime executor stub tensor dry-run trace: `outputs/phase4/llama_executor_stub/runtime_kv_tensor_dry_run_trace.jsonl`
- Runtime executor stub verification metrics: `outputs/phase4/llama_executor_stub/verification_metrics.json`
- Runtime liburing executor summary: `outputs/phase4/llama_executor_liburing/runtime_kv_executor_liburing_summary.json`
- Runtime liburing executor trace summary: `outputs/phase4/llama_executor_liburing/runtime_kv_executor_liburing_trace_summary.json`
- Runtime liburing executor tensor dry-run trace: `outputs/phase4/llama_executor_liburing/runtime_kv_tensor_dry_run_trace.jsonl`
- Runtime liburing executor verification metrics: `outputs/phase4/llama_executor_liburing/verification_metrics.json`
- Runtime tensor mutation gate summary: `outputs/phase4/llama_tensor_mutation_gate/runtime_kv_tensor_mutation_summary.json`
- Runtime tensor mutation gate trace: `outputs/phase4/llama_tensor_mutation_gate/runtime_kv_tensor_mutation_trace.jsonl`
- Runtime tensor mutation gate verification metrics: `outputs/phase4/llama_tensor_mutation_gate/verification_metrics.json`
- Runtime movement intent verification metrics: `outputs/phase4/llama_movement_intent_bridge/verification_metrics.json`

## Integration Points

The machine-readable source of truth is `integrations/llama_cpp/integration_map.py`.
It checks the local llama.cpp checkout for the anchors below and checks the
current trace patch for implemented instrumentation anchors.

### Implemented Instrumentation

- `trace_kv_add_apply_ubatch`
  - File: `src/llama-kv-cache.cpp`
  - Anchor: `llama_kv_cache::apply_ubatch`
  - Emits: `kv_add`
  - Purpose: records newly applied KV cache token ranges.

- `trace_kv_evict_seq_rm`
  - File: `src/llama-kv-cache.cpp`
  - Anchor: `llama_kv_cache::seq_rm`
  - Emits: `kv_evict`
  - Purpose: records runtime KV removals that can later become SSD evictions.

### Implemented Runtime Adapter

- `runtime_command_dry_run_adapter`
  - File: `src/llama-kv-cache.cpp`
  - Anchor: `llama_kv_cache::apply_ubatch`
  - Commands: `kv_load`, `kv_prefetch`, `kv_compute`, `kv_evict`
  - Purpose: validates exported command JSONL inside patched llama.cpp and writes JSON metrics while leaving real KV tensor residency unchanged.

- `runtime_command_live_bridge`
  - File: `src/llama-kv-cache.cpp`
  - Anchor: `llama_kv_cache::apply_ubatch`
  - Commands: `kv_load`, `kv_prefetch`, `kv_compute`, `kv_evict`
  - Purpose: matches live layer/block/token ranges to scheduler commands and emits runtime bridge trace/metrics while leaving real KV tensor residency unchanged.

- `runtime_shadow_residency_bridge`
  - File: `src/llama-kv-cache.cpp`
  - Anchor: `llama_kv_cache::apply_ubatch`
  - Commands: `kv_load`, `kv_prefetch`, `kv_compute`, `kv_evict`
  - Purpose: maintains per-block metadata-only shadow residency from matched command source/target transitions.

- `runtime_movement_intent_bridge`
  - File: `src/llama-kv-cache.cpp`
  - Anchor: `llama_kv_cache::apply_ubatch`
  - Commands: `kv_load`, `kv_prefetch`, `kv_compute`, `kv_evict`
  - Purpose: classifies matched commands as metadata-only movement intents, including blocking/nonblocking state and per-kind byte counters.

- `runtime_executor_stub`
  - File: `src/llama-kv-cache.cpp`
  - Anchor: `llama_kv_cache::apply_ubatch`
  - Commands: `kv_load`, `kv_prefetch`
  - Purpose: executes matched SSD-read intents against an opt-in backing file and stores bytes in DRAM staging buffers while leaving real KV tensor residency unchanged. Supports `stdio` and `liburing` backends.

- `runtime_executor_staging_residency`
  - File: `src/llama-kv-cache.cpp`
  - Anchor: `llama_kv_cache::apply_ubatch`
  - Commands: `kv_load`, `kv_prefetch`, `kv_compute`, `kv_evict`
  - Purpose: tracks explicit DRAM/VRAM staging buffers, lazy initial-resident source seeding, DRAM-to-VRAM staging copies, compute residency checks, and C++ byte-range counters before real KV tensors are mutated.

- `runtime_executor_tensor_copy_dry_run`
  - File: `src/llama-kv-cache.cpp`
  - Anchor: `llama_kv_cache::apply_ubatch`
  - Commands: `kv_load`, `kv_prefetch`, `kv_compute`
  - Purpose: emits opt-in tensor-copy dry-run trace events with staged K/V tensor byte offsets and FNV-1a checksums while real llama.cpp KV tensors remain unchanged.

- `runtime_executor_tensor_mutation_gate`
  - File: `src/llama-kv-cache.cpp`
  - Anchor: `llama_kv_cache::apply_ubatch`
  - Commands: `kv_load`, `kv_prefetch`, `kv_compute`
  - Purpose: when `LLAMA_SOLIDATTENTION_KV_TENSOR_MUTATION=1` is explicitly enabled, copies VRAM staging bytes into llama.cpp K/V tensors and verifies post-copy checksums. The gate is disabled in default executor baselines.

- `runtime_attention_residency_gate`
  - File: `src/llama-kv-cache.cpp`
  - Anchors: `llama_kv_cache::get_k`, `llama_kv_cache::get_v`
  - Commands: `kv_load`, `kv_prefetch`, `kv_compute`
  - Purpose: verifies that scheduler-required KV blocks have completed tensor
    mutation before K/V attention views are returned. This is a verification
    hook and does not perform scheduler-timed SSD I/O.

- `runtime_executor_liburing_build_hook`
  - File: `src/CMakeLists.txt`
  - Commands: `kv_load`, `kv_prefetch`
  - Purpose: optionally compiles and links patched llama.cpp with liburing for the executor backend when the local liburing prefix is available.

### Candidate Runtime Hooks

- `future_runtime_load_before_attention_views`
  - Candidate anchors: `get_k`, `get_v`
  - Commands: `kv_load`, `kv_prefetch`
  - Rationale: selected KV blocks must be resident before attention views are
    built. This needs careful synchronization because these functions are part
    of graph construction.

- `future_runtime_store_after_decode_copy`
  - Candidate anchors: `cpy_k`, `cpy_v`
  - Commands: metadata-side `kv_load` bookkeeping for newly written rows
  - Rationale: decode writes are graph nodes, so this is a metadata boundary,
    not an immediate host copy boundary.

- `future_runtime_block_io_byte_ranges`
  - Candidate anchors: `state_write_data`, `state_read_data`
  - Commands: `kv_load`, `kv_evict`
  - Rationale: these functions show the existing row-offset K/V byte range
    semantics and are useful for verifying block layout. They are not by
    themselves the final low-latency decode path.

### External Prototype

- `prototype_runtime_command_consumer`
  - File: `io/runtime_kv_command_consumer.c`
  - Commands: `kv_load`, `kv_prefetch`, `kv_compute`, `kv_evict`
  - Purpose: proves the exported command stream can drive liburing SSD movement
    and mock DRAM/VRAM movement before the same contract is wired into llama.cpp.

- `movement_intent_plan_export`
  - File: `integrations/llama_cpp/movement_intents.py`
  - Commands: `kv_load`, `kv_prefetch`, `kv_compute`, `kv_evict`
  - Purpose: turns metadata-only bridge movement intents into a MovementPlan-compatible artifact for executor verification.

- `runtime_residency_executor`
  - File: `harness/runtime_residency_executor.py`
  - Commands: `kv_load`, `kv_prefetch`, `kv_compute`, `kv_evict`
  - Purpose: executes movement-intent plans with explicit DRAM/VRAM buffer residency, lazy initial-resident source seeding, trace, and JSON metrics before real llama.cpp KV tensors are mutated.

- `runtime_kv_byte_range_verifier`
  - File: `integrations/llama_cpp/kv_byte_ranges.py`
  - Commands: `kv_load`, `kv_prefetch`, `kv_compute`, `kv_evict`
  - Purpose: maps interleaved staging bytes to llama.cpp row-major K/V tensor byte offsets, checks coverage and tensor bounds, and writes checksum-backed trace/metrics before real tensor mutation is enabled.

## Verification

```bash
git submodule update --init external/llama.cpp
python -m tools.check llama
```

Historical evidence records an earlier 16-point map with 13 implemented-style
points and 3 candidate runtime hooks. Adding the attention-residency gate
extended the map to 17 checked points with 14 implemented-style points and the
same 3 candidates. Current status is still Phase 4 scaffolding: scheduler
behavior is trace-derived and command-driven, default executor runs do not
mutate real KV tensors, and the explicit mutation and attention gates are
verified smoke paths rather than final scheduler-timed attention integration.
