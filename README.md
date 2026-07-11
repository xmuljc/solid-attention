# SolidAttention Reproduction Harness

## Environment Setup

Windows PowerShell does not require script activation. Create the environment
and invoke its interpreter directly:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m tools.check fast
```

On a POSIX shell:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m tools.check fast
```

After activating the virtual environment, the shell-neutral verification entry
point is `python -m tools.check fast`.

| Profile | Owner platform | Prerequisites | Command |
| --- | --- | --- | --- |
| `fast` | Windows and Linux | Python 3.11+ and the `dev` extra | `python -m tools.check fast` |
| `io` | Linux | POSIX C toolchain and `make` | `python -m tools.check io` |
| `liburing` | Linux | C toolchain, `make`, and liburing development files | `python -m tools.check liburing` |
| `llama` | Linux | Git, CMake, C/C++ toolchain, and the pinned llama.cpp submodule | `python -m tools.check llama` |

The llama.cpp integration is optional for `fast`. Initialize its pinned
submodule before running the `llama` profile:

```bash
git submodule update --init external/llama.cpp
```

Generated experiment and check artifacts belong under ignored
`outputs/<run-id>/` directories. CI uploads profile evidence from `outputs/` as
workflow artifacts; generated output is not part of the source tree.

This repository is a harness-first reproduction of the core mechanisms from
SolidAttention. Phase 1 intentionally stays in simulation: it does not run a
real LLM, does not modify llama.cpp, does not use liburing, and does not
implement real attention.

## Phase 1 Status

Implemented:

- KV block metadata
- Simulated SSD/DRAM/VRAM block store
- Trace recorder
- Metrics collector
- Init/Local/Selected block selector
- Representative-vector based selected-block ranking
- Speculative prefetch simulation with hit/miss/wrong-prefetch metrics
- SSD-aware scheduler simulation with trace and metrics
- Interleaved KV layout metadata simulation
- Executable smoke simulation harness with JSON trace/metrics outputs

Current scope:

- Metadata-only KV cache block state management
- Simulated location transfers between `ssd`, `dram`, and `vram`
- JSON trace output for simulated operations
- JSON metrics output for latency, I/O, selection, and prefetch counters
- Trace-emitting Init/Local/Selected block selection simulation
- Metadata-only speculative prefetch execution and evaluation
- Metadata-only SSD/DRAM/VRAM scheduling timeline
- Token-level interleaved K/V offset mapping

Out of scope for Phase 1:

- Real attention computation
- CUDA kernels
- liburing-based SSD I/O
- llama.cpp runtime integration
- Model or dataset downloads

## Run Smoke Harness

```bash
python -m harness.smoke --config configs/smoke_base.yaml --output-dir outputs/smoke
```

The smoke harness writes:

- `outputs/smoke/smoke_trace.json`
- `outputs/smoke/smoke_metrics.json`
- `outputs/smoke/smoke_summary.json`
- `outputs/smoke/run_manifest.json`

## Phase 2 Block I/O Prototype

```bash
python -m tools.check io
python -m tools.check liburing
```

The `io` profile requires the POSIX fallback backend. The `liburing` profile
requires the real liburing backend and fails when it is unavailable.

## Phase 3 llama.cpp Integration Scaffold

The project includes `integrations/llama_cpp/` with an adapter, JSONL trace ingestion, runtime trace summarization, runtime movement-plan generation, checkout probe, patch verifier, and an opt-in KV-cache trace patch for llama.cpp. The clean checkout is `external/llama.cpp` at commit `bbebeec`; `python -m tools.check llama` defines the current clean-apply, build, and smoke contract for `solidattention_kv_trace.patch`. Imported historical evidence records a successful build, movement-schema KV trace fields (`source`, `target`, `ssd_offset`), and a runtime `kv_evict` smoke path through patched `test-llama-archs`; a fresh checkout must run the profile to produce current evidence. No llama.cpp runtime files are modified in-place in the clean checkout.

Run `python -m tools.check llama` on Linux to verify the pinned checkout, apply
the patch in an isolated temporary tree, build the synthetic target, and run the
llama-marked tests. Phase status documents retain historical counts from the
imported snapshot; their generated files are no longer tracked.

## Phase 4 Trace-Derived Scheduler Replay

`runtime_replay` can preserve llama.cpp trace-derived residency with `--initial-location trace`. Imported historical evict-smoke evidence records a replay that starts from 2 DRAM blocks and 2 SSD blocks, then performs only 2 SSD reads before all 4 blocks reach VRAM. `runtime_commands.py` exports the resulting token-aware movement plan as a C++-friendly llama.cpp KV command JSONL contract, `io/runtime_kv_command_consumer` executes that command stream through liburing-backed SSD reads plus mock DRAM/VRAM buffers, and the patched llama.cpp runtime now has opt-in command dry-run plus live command bridge adapters. The live bridge matches each `apply_ubatch` layer/block/token range to scheduler commands, maintains metadata-only per-block shadow residency, classifies matched commands as movement intents (`ssd_read`, `dram_to_vram`, `compute`, `ssd_write`) with blocking and byte-count metrics, and emits JSONL trace/metrics without mutating KV tensors. `movement_intents.py` exports those bridge events back into a MovementPlan-compatible JSON so the existing liburing-backed movement executor can run the SSD-read intent path outside llama.cpp. The patched llama.cpp bridge also has an opt-in executor path, enabled by `LLAMA_SOLIDATTENTION_KV_EXECUTOR_BACKING_PATH`, that reads SSD-read intents from a backing file into explicit DRAM staging buffers, copies DRAM-resident blocks into VRAM staging buffers for `dram_to_vram` intents, verifies VRAM residency for compute intents, and still leaves real KV tensors unchanged. It supports `LLAMA_SOLIDATTENTION_KV_EXECUTOR_BACKEND=liburing|stdio`, with liburing linked through the patched llama.cpp CMake hook when available, and its runtime summary reports C++ byte-range counters for the staged K/V blocks. When `LLAMA_SOLIDATTENTION_KV_TENSOR_DRY_RUN_TRACE_PATH` is set, the executor also emits a tensor-copy dry-run JSONL trace with K/V tensor byte offsets and FNV-1a checksums, still without mutating real KV tensors. A separate explicit gate, `LLAMA_SOLIDATTENTION_KV_TENSOR_MUTATION=1`, copies VRAM staging bytes into llama.cpp K/V tensors and verifies post-copy checksums; the stdio/liburing executor baselines keep this gate off by default. `LLAMA_SOLIDATTENTION_KV_ATTENTION_GATE_TRACE_PATH` adds a get_k/get_v attention-view residency gate that verifies required blocks are mutation-ready before attention views are returned. A harness-side runtime residency executor (`harness.runtime_residency_executor`) replays movement-intent plans with explicit DRAM/VRAM buffer state, optional lazy seeding for initially resident blocks, and JSON trace/metrics. `integrations.llama_cpp.kv_byte_ranges` then verifies that interleaved staging bytes map onto llama.cpp KV-cell K/V tensor byte offsets with checksums.

Use `python -m tools.check fast` for the harness-side replay contracts and
`python -m tools.check llama` for the pinned runtime patch and bridge. Imported
historical evidence recorded 120 movement intents, 24 SSD-read intents, 512 K/V
row ranges, and 288 ready attention-view block checks with no missing or pending
mutations. Those generated metrics are not tracked in the baseline; current CI
profile artifacts provide reproducible evidence for each run.

## Ablation Harness

```bash
python -m harness.ablation --config configs/smoke_base.yaml --output-dir outputs/ablation
```

Current simulation methods:

- `memory_only`
- `ssd_sync`
- `ssd_async`
- `prefetch_only`
- `full_scheduler`
- `solid_sim` compatibility alias for `full_scheduler`

Each method writes a JSON trace and metrics file under the selected ignored
output directory. The run writes one aggregate `ablation_summary.json` and one
`solidattention.run_manifest.v1` manifest.
