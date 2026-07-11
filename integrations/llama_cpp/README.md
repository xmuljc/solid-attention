# Phase 3 llama.cpp Integration Scaffold

This directory prepares llama.cpp KV-cache integration without immediately
rewriting runtime behavior.

Implemented now:

- `adapter.py`: converts llama.cpp KV-cache instrumentation events into
  `KVBlock` and `BlockStore` state, including movement-style `kv_load`,
  `kv_prefetch`, and `kv_evict` events.
- `load_jsonl_events()` / `LlamaKVCacheAdapter.ingest_jsonl()`: consume one
  JSON event per line from a patched llama.cpp run.
- `duplicate_policy="replace"`: replays runtime traces where repeated decode
  passes overwrite the same KV cache slot.
- `runtime_trace.py`: summarizes runtime JSONL traces and writes harness trace
  JSON.
- `evict_smoke.py`: runs a patched synthetic llama runtime with
  `LLAMA_SOLIDATTENTION_KV_TRACE_EVICT_SMOKE=1` and requires real `kv_evict`
  events in the emitted JSONL.
- `runtime_commands.py`: validates token-aware movement plans and exports a
  llama.cpp-oriented KV command JSONL contract.
- `command_bridge.py`: summarizes patched llama.cpp live command bridge JSONL
  traces, including metadata-only shadow residency transitions and movement
  intent counts/bytes, and writes JSON metrics.
- `movement_intents.py`: exports bridge movement-intent traces into
  MovementPlan-compatible JSON for the existing liburing-backed movement executor.
- `harness.runtime_residency_executor`: replays movement-intent plans with explicit DRAM/VRAM buffer state, optional lazy seeding for initially resident blocks, and JSON trace/metrics.
- `kv_byte_ranges.py`: verifies interleaved staging bytes against llama.cpp K/V row-major tensor offsets with checksums, trace, and JSON metrics.
- Patched executor path: enabled by `LLAMA_SOLIDATTENTION_KV_EXECUTOR_BACKING_PATH`, reads matched SSD-read intents into DRAM staging buffers, copies DRAM blocks into VRAM staging buffers, verifies compute residency, and emits C++ byte-range counters. `LLAMA_SOLIDATTENTION_KV_EXECUTOR_BACKEND=liburing|stdio` selects the backend; the liburing backend is enabled by the optional CMake hook. Setting `LLAMA_SOLIDATTENTION_KV_TENSOR_DRY_RUN_TRACE_PATH` adds an opt-in tensor-copy dry-run JSONL trace with K/V tensor byte offsets and FNV-1a checksums. Real KV tensor writes stay disabled unless `LLAMA_SOLIDATTENTION_KV_TENSOR_MUTATION=1` is set; when enabled, `LLAMA_SOLIDATTENTION_KV_TENSOR_MUTATION_TRACE_PATH` records per-block post-copy checksum verification.
- `integration_map.py`: records and validates patchable llama.cpp KV cache
  integration points for trace capture, the in-process command dry-run adapter,
  the metadata-only movement intent bridge, future runtime load/store hooks, and the external command consumer prototype.
- `probe.py`: checks whether a local llama.cpp checkout has the expected files
  before integration work begins.
- `verify_patch.py`: runs `git apply --check` and also accepts the
  `already_applied` state for local experimentation.
- `patches/solidattention_kv_trace.patch`: adds an opt-in JSONL trace hook in
  `src/llama-kv-cache.cpp`, enabled by `LLAMA_SOLIDATTENTION_KV_TRACE_PATH`;
  emitted records include `source`, `target`, and `ssd_offset`, with a
  trace-only `kv_evict` hook in `seq_rm`. The same patch also adds opt-in
  command dry-run, live bridge, shadow residency, and movement intent adapters enabled by
  `LLAMA_SOLIDATTENTION_KV_COMMAND_PATH`,
  `LLAMA_SOLIDATTENTION_KV_COMMAND_SUMMARY_PATH`,
  `LLAMA_SOLIDATTENTION_KV_COMMAND_BRIDGE_TRACE_PATH`, and
  `LLAMA_SOLIDATTENTION_KV_COMMAND_BRIDGE_SUMMARY_PATH`. The executor extension also recognizes `LLAMA_SOLIDATTENTION_KV_TENSOR_DRY_RUN_TRACE_PATH`, `LLAMA_SOLIDATTENTION_KV_TENSOR_MUTATION`, and `LLAMA_SOLIDATTENTION_KV_TENSOR_MUTATION_TRACE_PATH`.

Current integration contract:

- Clean path: `external/llama.cpp`
- Commit: `bbebeec`
- Patch status: applies
- Historical verification evidence in the prior imported snapshot records
  successful patched `llama` and `test-llama-archs` builds, movement-schema and
  evict-smoke runs, and command dry-run and live-bridge runs. Their temporary
  build directories are intentionally not part of the clean-clone contract.

The generated paths and counts below are historical evidence from that imported
snapshot. Generated `outputs/` files are ignored and are not tracked in the
current baseline.
- Runtime trace: `outputs/phase3/runtime_kv_trace.jsonl`, 48 events from synthetic llama decode
- Movement-schema runtime trace: `outputs/phase3/runtime_kv_trace_movement_schema.jsonl`, 48 events with `source`, `target`, and `ssd_offset` fields
- Evict-smoke runtime trace: `outputs/phase3/runtime_kv_trace_evict_smoke.jsonl`, 60 events including 12 real `kv_evict` records
- Runtime movement plan: `outputs/phase3/runtime_replay_calibrated/runtime_movement_plan.json`, 12 ops over 4 SSD ranges
- Movement execution: `outputs/phase3/movement_execution/movement_execution_metrics.json`, 4 SSD reads via liburing probe
- Runtime command export: `outputs/phase4/runtime_replay_evict_trace_locations/runtime_kv_commands.jsonl`, 10 commands with token intervals
- Earlier runtime integration map: `outputs/phase4/runtime_integration_map_check.json`, 16 checked points, 13 implemented points, 3 candidate runtime hooks before the attention-residency gate was added
- Runtime command dry-run summary: `outputs/phase4/llama_command_dry_run/runtime_kv_command_dry_run_summary.json`, status ok with 10 valid commands
- Runtime command bridge summary: `outputs/phase4/llama_command_bridge/runtime_kv_command_bridge_summary.json`, status ok with 120 matched command events
- Runtime command bridge trace summary: `outputs/phase4/llama_command_bridge/runtime_kv_command_bridge_trace_summary.json`, 120 bridge trace events
- Runtime shadow bridge summary: `outputs/phase4/llama_shadow_bridge/runtime_kv_shadow_bridge_summary.json`, 120 shadow transitions with 0 source mismatches
- Runtime movement intent bridge summary: `outputs/phase4/llama_movement_intent_bridge/runtime_kv_movement_intent_bridge_summary.json`, 120 movement intents with 72 blocking and 48 nonblocking intents
- Runtime movement intent verification metrics: `outputs/phase4/llama_movement_intent_bridge/verification_metrics.json`
- Runtime movement intent plan: `outputs/phase4/llama_movement_intent_bridge/runtime_kv_movement_intent_plan.json`, 120 movement ops over 4 unique blocks
- Runtime movement intent execution: `outputs/phase4/movement_intent_execution/movement_execution_metrics.json`, 2 unique SSD reads through liburing probe
- Runtime residency execution: `outputs/phase4/runtime_residency_execution/runtime_residency_metrics.json`, 120 movement ops with 0 verification errors
- Runtime residency verification metrics: `outputs/phase4/runtime_residency_execution/verification_metrics.json`
- Runtime KV byte-range verification: `outputs/phase4/kv_byte_range_verification/kv_byte_range_metrics.json`, 4 blocks and 512 K/V row ranges with 0 verification errors
- Runtime KV byte-range verification metrics: `outputs/phase4/kv_byte_range_verification/verification_metrics.json`
- Runtime executor stub summary: `outputs/phase4/llama_executor_stub/runtime_kv_executor_stub_summary.json`, stdio backend with 24 SSD-read successes, 4 DRAM staging buffers, 4 VRAM staging buffers, 512 C++ byte ranges, 4 tensor dry-run events, and 0 missing sources
- Runtime executor stub tensor dry-run trace: `outputs/phase4/llama_executor_stub/runtime_kv_tensor_dry_run_trace.jsonl`
- Runtime executor stub verification metrics: `outputs/phase4/llama_executor_stub/verification_metrics.json`
- Runtime liburing executor summary: `outputs/phase4/llama_executor_liburing/runtime_kv_executor_liburing_summary.json`, liburing backend with 24 SSD-read successes, 4 DRAM staging buffers, 4 VRAM staging buffers, 512 C++ byte ranges, 4 tensor dry-run events, and 0 missing sources
- Runtime liburing executor tensor dry-run trace: `outputs/phase4/llama_executor_liburing/runtime_kv_tensor_dry_run_trace.jsonl`
- Runtime liburing executor verification metrics: `outputs/phase4/llama_executor_liburing/verification_metrics.json`
- Runtime tensor mutation gate summary: `outputs/phase4/llama_tensor_mutation_gate/runtime_kv_tensor_mutation_summary.json`, explicit gate enabled with 4 mutation attempts, 512 K/V ranges, 262144 copied bytes, and 0 verify errors
- Runtime tensor mutation gate trace: `outputs/phase4/llama_tensor_mutation_gate/runtime_kv_tensor_mutation_trace.jsonl`
- Runtime tensor mutation gate verification metrics: `outputs/phase4/llama_tensor_mutation_gate/verification_metrics.json`

Verify the current integration on Linux:

```bash
git submodule update --init external/llama.cpp
python -m tools.check llama
```

The profile owns temporary patch/build directories and writes structured check
evidence under ignored `outputs/checks/llama/` for CI artifact upload.

Not implemented yet:

- Running a user-provided or downloaded real GGUF model.
- Production scheduler-timed SSD/DRAM/VRAM block scheduling inside llama.cpp.
- Attention kernel changes.
