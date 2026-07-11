# Phase 3 Verification Status

Phase 3 prepares llama.cpp integration with a small, reviewable instrumentation
patch before behavioral runtime changes. The imported verified milestone was:
patch applies cleanly, compiles in isolated worktrees, emits runtime KV-cache
JSONL from an offline synthetic llama decode, and the emitted JSONL can be
replayed into the harness adapter.

## Implemented

- `integrations/llama_cpp/adapter.py`: maps llama.cpp-style KV-cache trace events to `KVBlock` and `BlockStore`, including movement-style `kv_load`, `kv_prefetch`, and `kv_evict` records.
- `load_jsonl_events()` and `LlamaKVCacheAdapter.ingest_jsonl()`: read patch-emitted JSONL traces into the harness.
- Duplicate runtime KV slot replay via explicit `duplicate_policy="replace"`.
- `integrations/llama_cpp/runtime_trace.py`: summarizes runtime JSONL traces and writes harness trace JSON.
- `integrations/llama_cpp/evict_smoke.py`: runs a patched synthetic llama runtime scenario that must emit real `kv_evict` JSONL records.
- `harness/runtime_replay.py`: replays runtime KV traces through the SSD-aware scheduler simulation.
- `core/movement_plan.py`: converts scheduler timelines into runtime-oriented `kv_load` / `kv_compute` movement plans with SSD file offsets.
- `harness/movement_executor.py`: executes movement-plan SSD `kv_load` groups through the Phase 2 block I/O probe and writes execution metrics/trace.
- `integrations/llama_cpp/probe.py`: verifies whether a local llama.cpp checkout has a minimal expected shape.
- `integrations/llama_cpp/verify_patch.py`: checks whether an integration patch applies cleanly or is already applied.
- `integrations/llama_cpp/patches/solidattention_kv_trace.patch`: adds an opt-in JSONL KV-cache trace hook to `src/llama-kv-cache.cpp`; events now carry `source`, `target`, and `ssd_offset`, and the patch includes a trace-only `kv_evict` hook in `seq_rm`.
- Unit tests for add/move/movement KV events, duplicate slot replay, JSONL ingestion, runtime trace summaries, eviction-smoke summarization, checkout probing, and patch verification.

## Checkout Contract

- Clean llama.cpp checkout: `external/llama.cpp`
- Checkout commit: `bbebeec`
- Clean checkout status during verification: clean
- Runtime files modified in clean checkout: no
- `python -m tools.check llama` owns isolated temporary patch and build
  directories; those directories are not part of the source contract.

The patch is controlled by `LLAMA_SOLIDATTENTION_KV_TRACE_PATH`. When applied
and run with this environment variable, llama.cpp appends JSONL events
containing layer id, cache-cell block id, token interval, estimated touched
bytes, location, `source`, `target`, `ssd_offset`, and metadata. The default synthetic decode emits `kv_add`; the optional evict-smoke
scenario sets `LLAMA_SOLIDATTENTION_KV_TRACE_EVICT_SMOKE=1` and calls
`llama_memory_seq_rm`, which has now been verified to emit real runtime
`kv_evict` events.

## Runtime Trace Verification

The patched `test-llama-archs` target was run for `-a llama -s 1`. This test
constructs a small synthetic llama model in-process and calls `llama_decode()`;
it does not download a model.

The generated paths and measurements below are historical evidence from the
imported snapshot. They are retained as verified mechanism history, but the
files themselves are ignored and no longer tracked.

Runtime trace outputs:

- `outputs/phase3/runtime_kv_trace.jsonl`
- `outputs/phase3/runtime_kv_trace_metrics.json`
- `outputs/phase3/runtime_kv_harness_trace.json`
- `outputs/phase3/runtime_kv_archs_stdout.txt`
- `outputs/phase3/runtime_kv_archs_stderr.txt`

Movement-schema runtime trace outputs:

- `outputs/phase3/runtime_kv_trace_movement_schema.jsonl`
- `outputs/phase3/runtime_kv_trace_movement_schema_metrics.json`
- `outputs/phase3/runtime_kv_movement_schema_harness_trace.json`

Evict-smoke runtime trace outputs:

- `outputs/phase3/runtime_kv_trace_evict_smoke.jsonl`
- `outputs/phase3/runtime_kv_trace_evict_smoke_metrics.json`
- `outputs/phase3/runtime_kv_evict_smoke_harness_trace.json`
- `outputs/phase3/runtime_kv_evict_smoke_stdout.txt`
- `outputs/phase3/runtime_kv_evict_smoke_stderr.txt`

Runtime scheduler replay outputs:

- `outputs/phase3/runtime_replay/runtime_replay_metrics.json`
- `outputs/phase3/runtime_replay/runtime_replay_trace.json`
- `outputs/phase3/runtime_replay/runtime_replay_summary.json`

Calibrated runtime scheduler replay outputs:

- `outputs/phase3/runtime_replay_calibrated/runtime_replay_metrics.json`
- `outputs/phase3/runtime_replay_calibrated/runtime_replay_trace.json`
- `outputs/phase3/runtime_replay_calibrated/runtime_replay_summary.json`
- `outputs/phase3/runtime_replay_calibrated/runtime_movement_plan.json`
- `outputs/phase3/runtime_replay_calibrated/runtime_replay_io_probe.bin`

Movement-schema calibrated runtime scheduler replay outputs:

- `outputs/phase3/runtime_replay_movement_schema_calibrated/runtime_replay_metrics.json`
- `outputs/phase3/runtime_replay_movement_schema_calibrated/runtime_replay_trace.json`
- `outputs/phase3/runtime_replay_movement_schema_calibrated/runtime_replay_summary.json`
- `outputs/phase3/runtime_replay_movement_schema_calibrated/runtime_movement_plan.json`

Evict-smoke calibrated runtime scheduler replay outputs:

- `outputs/phase3/runtime_replay_evict_smoke_calibrated/runtime_replay_metrics.json`
- `outputs/phase3/runtime_replay_evict_smoke_calibrated/runtime_replay_trace.json`
- `outputs/phase3/runtime_replay_evict_smoke_calibrated/runtime_replay_summary.json`
- `outputs/phase3/runtime_replay_evict_smoke_calibrated/runtime_movement_plan.json`

Movement execution outputs:

- `outputs/phase3/movement_execution/movement_execution_metrics.json`
- `outputs/phase3/movement_execution/movement_execution_trace.json`
- `outputs/phase3/movement_execution/movement_execution_summary.json`
- `outputs/phase3/movement_execution/movement_io_probe_65536.bin`

Movement-schema movement execution outputs:

- `outputs/phase3/movement_execution_movement_schema/movement_execution_metrics.json`
- `outputs/phase3/movement_execution_movement_schema/movement_execution_trace.json`
- `outputs/phase3/movement_execution_movement_schema/movement_execution_summary.json`

Evict-smoke movement execution outputs:

- `outputs/phase3/movement_execution_evict_smoke/movement_execution_metrics.json`
- `outputs/phase3/movement_execution_evict_smoke/movement_execution_trace.json`
- `outputs/phase3/movement_execution_evict_smoke/movement_execution_summary.json`

Historical runtime summary:

```text
event_count: 48
unique_block_count: 4
duplicate_event_count: 44
layers: [0, 1]
ops: ['kv_add']
location_stats: {'dram': {'block_count': 4, 'size_bytes': 262144}, 'ssd': {'block_count': 0, 'size_bytes': 0}, 'vram': {'block_count': 0, 'size_bytes': 0}}
```

Historical evict-smoke runtime summary:

```text
event_count: 60
kv_add: 48
kv_evict: 12
ops: ['kv_add', 'kv_evict']
location_stats: {'dram': {'block_count': 2, 'size_bytes': 131072}, 'ssd': {'block_count': 2, 'size_bytes': 131072}, 'vram': {'block_count': 0, 'size_bytes': 0}}
```

Historical runtime scheduler replay summary:

```text
initial_location: ssd
selected_block_count: 4
ssd_read_bytes: 262144
io_op_count: 4
gpu_wait_time_ms: 5.0
after_location_stats: {'dram': {'block_count': 0, 'size_bytes': 0}, 'ssd': {'block_count': 0, 'size_bytes': 0}, 'vram': {'block_count': 4, 'size_bytes': 262144}}
```

Historical calibrated runtime scheduler replay summary:

```text
io_backend: liburing
io_probe_block_size: 65536
io_probe_block_count: 4
ssd_read_latency_ms_per_block: 0.03888325
selected_block_count: 4
ssd_read_bytes: 262144
io_op_count: 4
after_location_stats: {'dram': {'block_count': 0, 'size_bytes': 0}, 'ssd': {'block_count': 0, 'size_bytes': 0}, 'vram': {'block_count': 4, 'size_bytes': 262144}}
movement_plan_ops: 12
ssd_layout_total_size_bytes: 262144
ssd_layout_alignment: 4096
```

Historical movement execution summary:

```text
io_backend: liburing
movement_op_count: 12
kv_load_op_count: 8
ssd_load_op_count: 4
unique_ssd_read_count: 4
ssd_read_bytes: 262144
io_probe_read_latency_ms_per_block: 0.037602
```

## Verified Commands

```bash
git submodule update --init external/llama.cpp
python -m tools.check llama
```

The profile validates the pinned checkout, applies the patch in an isolated
temporary tree, builds and runs the synthetic target, and executes the
llama-marked test contract. Profile output is ignored locally and uploaded by
CI as an artifact.

Historical observed results:

```text
patch status: applies
patched test-llama-archs target: built successfully
runtime trace replay: 48 events
movement-schema runtime trace: 48 events with source/target/ssd_offset fields
runtime kv_evict smoke: 60 events, 12 kv_evict events
calibrated runtime replay: liburing, 4 blocks, 262144 SSD read bytes
movement plan: 12 runtime ops, 4 SSD layout ranges, total size 262144 bytes
movement executor: liburing, 4 unique SSD reads, 262144 bytes
movement-schema movement executor: liburing, 4 unique SSD reads, 262144 bytes
evict-smoke movement executor: liburing, 4 unique SSD reads, 262144 bytes
full pytest: 85 passed in 0.80s
```

## Still Pending

- Run a real downloaded or user-provided GGUF model through patched llama.cpp.
- Add a production scheduler-owned load-before-attention path that moves real
  llama.cpp KV tensor blocks across SSD, DRAM, and VRAM.
- Replace simulated prefetch latency with scheduler-timed liburing reads of real
  scheduler-selected KV block contents.
- Integrate SolidAttention scheduling behavior into llama.cpp after instrumentation has been validated.
