# Phase 4 Verification Status

Phase 4 is currently implemented as trace-derived scheduler replay plus
patched llama.cpp runtime bridge/executor scaffolding. Default executor runs
still leave real KV tensors unchanged; an explicit
`LLAMA_SOLIDATTENTION_KV_TENSOR_MUTATION=1` gate now verifies a smoke path that
copies VRAM staging bytes into llama.cpp K/V tensors and checks post-copy
checksums. A verification-only attention-view residency gate now checks that scheduler-required blocks are mutation-ready before `get_k` / `get_v` return views. Full scheduler-timed attention integration remains pending.

## Implemented

- Init/Local/Selected block selection.
- Representative-vector selected-block ranking.
- Speculative prefetch simulation.
- SSD-aware scheduling simulation.
- Interleaved KV layout metadata.
- End-to-end smoke harness combining the above modules.
- Runtime KV trace replay into the SSD-aware scheduler simulation via `harness/runtime_replay.py`.
- Runtime movement plan generation with SSD offsets and token intervals via `core/movement_plan.py`.
- Token-aware movement plan schema with top-level `start_token` / `end_token` on each movement op.
- Trace-derived runtime scheduler replay that preserves llama.cpp `kv_add` / `kv_evict` residency before scheduling.
- llama.cpp runtime command JSONL export and validation via `integrations/llama_cpp/runtime_commands.py`.
- Runtime command C consumer prototype via `io/runtime_kv_command_consumer.c`, using liburing SSD reads/writes and mock DRAM/VRAM buffers.
- Patched llama.cpp command dry-run adapter that reads `solidattention.llama_cpp.kv_command.v1` JSONL and writes JSON summary metrics without mutating KV tensors.
- Patched llama.cpp live command bridge that matches `apply_ubatch` layer/block/token ranges to exported scheduler commands and emits JSONL trace plus JSON metrics without mutating KV tensors.
- Metadata-only per-block shadow residency inside the patched llama.cpp bridge, including source-match validation and final SSD/DRAM/VRAM shadow counts.
- Metadata-only movement intent classification inside the patched llama.cpp bridge, including blocking/nonblocking counts and per-kind bytes for `ssd_read`, `dram_to_vram`, `compute`, and `ssd_write`.
- Movement-intent plan export via `integrations/llama_cpp/movement_intents.py`, producing MovementPlan-compatible JSON from bridge traces.
- Opt-in patched llama.cpp executor path via `LLAMA_SOLIDATTENTION_KV_EXECUTOR_BACKING_PATH`, reading SSD-read intents into explicit DRAM staging buffers, copying DRAM blocks into VRAM staging buffers, verifying VRAM residency for compute intents, and still leaving real KV tensors unchanged. `LLAMA_SOLIDATTENTION_KV_EXECUTOR_BACKEND=liburing|stdio` selects the backend; liburing is compiled through an optional CMake hook when available.
- Machine-checkable llama.cpp integration map via `integrations/llama_cpp/integration_map.py`.
- Movement plan executor verification via `harness/movement_executor.py`.
- Runtime movement-intent residency execution via `harness/runtime_residency_executor.py`, tracking explicit DRAM/VRAM buffers, lazy initial-resident source seeding, trace, and JSON metrics.
- Runtime KV byte-range verifier via `integrations/llama_cpp/kv_byte_ranges.py`, mapping interleaved staging bytes to llama.cpp K/V row-major tensor offsets with checksums, trace, and JSON metrics.
- Patched llama.cpp executor summaries now emit C++ byte-range counters for staged K/V blocks, so runtime and Python verifier evidence can be compared directly.
- Opt-in patched llama.cpp tensor-copy dry-run via `LLAMA_SOLIDATTENTION_KV_TENSOR_DRY_RUN_TRACE_PATH`, emitting per-block K/V tensor byte offsets and FNV-1a checksums from VRAM staging buffers without mutating real KV tensors.
- Explicit patched llama.cpp tensor mutation gate via `LLAMA_SOLIDATTENTION_KV_TENSOR_MUTATION=1`, copying VRAM staging bytes into llama.cpp K/V tensors and emitting post-copy checksum verification in `LLAMA_SOLIDATTENTION_KV_TENSOR_MUTATION_TRACE_PATH`. The gate is disabled by default in stdio/liburing executor baselines.
- Verification-only patched llama.cpp attention-view residency gate via `LLAMA_SOLIDATTENTION_KV_ATTENTION_GATE_TRACE_PATH`, checking `get_k` / `get_v` view construction against scheduler-required blocks and reporting ready/missing/pending mutation counts.

## Verification Profiles

```bash
python -m tools.check fast
python -m tools.check io
python -m tools.check liburing
git submodule update --init external/llama.cpp
python -m tools.check llama
```

`fast` owns the platform-neutral scheduler and manifest contracts. The Linux
profiles own the POSIX, liburing, and pinned llama.cpp runtime boundaries; a
required backend may not pass through fallback or skip. Generated evidence is
written below ignored `outputs/checks/` directories and uploaded by CI.

The imported snapshot recorded the following historical result. The values are
retained as mechanism evidence, but the generated files are no longer tracked:

```text
113 passed in 0.86s
smoke block_count: 8
runtime replay: 4 blocks, 262144 SSD read bytes, all replayed blocks end in VRAM
calibrated runtime replay: liburing read latency drives scheduler SSD-read latency
runtime movement plan: 12 ops with SSD offsets for 4 KV blocks
movement executor: 4 SSD kv_load reads validated through liburing probe
runtime evict smoke: 12 kv_evict records emitted by patched llama.cpp seq_rm
trace-derived runtime replay: starts from 2 DRAM blocks and 2 SSD blocks, then schedules 2 SSD reads
trace-derived movement executor: 2 SSD kv_load reads, 131072 bytes, validated through liburing probe
runtime command export: 10 commands with token intervals, 2 SSD-read commands, 131072 SSD-read bytes
runtime command consumer: liburing backend, 10 commands, 2 SSD reads, 131072 SSD-read bytes, 0 verification errors
llama.cpp command dry-run: status ok, 10 valid commands, 2 SSD-read commands, 131072 SSD-read bytes, layer_count 2, cache_size 256
llama.cpp live command bridge: status ok, 48 runtime layer events, 120 matched command events, 0 unmatched layer events, 24 SSD-read command matches, 1572864 matched SSD-read bytes
bridge trace summarizer: 120 events over layers [0, 1], block IDs [0, 64], 4 unique runtime blocks
shadow residency bridge: 120 transitions, 120 source matches, 0 source mismatches, final shadow locations {vram: 4, dram: 0, ssd: 0}
movement intent bridge: 120 intents, 72 blocking, 48 nonblocking, 24 ssd_read intents (1572864 bytes), 48 dram_to_vram intents (3145728 bytes), 48 compute intents (3145728 bytes), 0 ssd_write intents
movement intent plan export: 120 movement ops, 4 unique blocks, 24 SSD-read intents, 1572864 repeated intent bytes
movement intent execution: liburing backend, 2 unique SSD kv_load reads, 131072 bytes, 0.07833 ms total read latency in probe
llama.cpp executor stub: stdio backend, 24 SSD-read attempts, 24 successes, 0 errors, 1572864 bytes read into DRAM staging buffers, 48 DRAM->VRAM staging copies, 48 compute residency checks, 2 lazy initial-resident source seeds, C++ byte-range counters {blocks: 4, ranges: 512, bytes: 262144, errors: 0}, tensor dry-run {events: 4, ranges: 512, bytes: 262144, checksum_count: 12, errors: 0}, tensor mutation {enabled: false, attempts: 0, bytes: 0}, final staging {dram: 4, vram: 4}
llama.cpp liburing executor: liburing linked in libllama, 24 SSD-read attempts, 24 successes, 0 errors, 0 backend errors, 1572864 bytes read into DRAM staging buffers, 48 DRAM->VRAM staging copies, 48 compute residency checks, 2 lazy initial-resident source seeds, C++ byte-range counters {blocks: 4, ranges: 512, bytes: 262144, errors: 0}, tensor dry-run {events: 4, ranges: 512, bytes: 262144, checksum_count: 12, errors: 0}, tensor mutation {enabled: false, attempts: 0, bytes: 0}, final staging {dram: 4, vram: 4}
llama.cpp tensor mutation gate: explicit gate enabled, 4 mutation attempts, 4 mutated blocks, 512 K/V ranges, 262144 copied bytes, checksum_count 16, 0 size/bounds/unsupported/verify errors; 4 JSONL mutation events have status ok and matching staging/post K/V checksums
llama.cpp attention residency gate: 168 get_k/get_v trace events, 144 required-view events status ok, 24 no-required-blocks events, 288 required block checks, 288 ready, 0 missing, 0 pending mutation
runtime residency execution: 120 movement ops, 24 SSD reads, 48 DRAM->VRAM copies, 48 compute ops, 2 lazy initial-resident source seeds, 0 verification errors, final residency {dram: 4, vram: 4}
KV byte-range verifier: 4 unique blocks, 512 K/V row ranges, 262144 mapped staging bytes, inferred 512-byte K rows and 512-byte V rows, KV-cell tensor offsets, 0 coverage/bounds/size/alignment errors
runtime integration map: 17 checked points, 14 implemented points, 3 candidate runtime hooks, status ok
```

## Still Pending

- Replace the verification-only attention residency gate with scheduler-timed `kv_load` / `kv_prefetch` that can block or prefetch before llama.cpp `get_k` / `get_v` attention views are consumed.
- Add real post-decode KV persistence and eviction for newly written rows after llama.cpp copy graph execution.
- Connect liburing reads to scheduler-selected KV block contents instead of synthetic backing-file bytes, then measure latency against the harness metrics.
