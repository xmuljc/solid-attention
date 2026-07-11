# Phase 2 Block I/O Prototype

This directory contains the standalone SSD block I/O prototype for Phase 2.

The Makefile supports three `BACKEND_MODE` values:

- `auto`: select liburing when its headers and library are available, otherwise
  build the POSIX fallback.
- `posix`: require the POSIX fallback backend.
- `liburing`: require liburing and fail the build if it is unavailable.

The fallback keeps the command, JSON output format, and block verification path
working so the harness can continue to evolve on dependency-limited hosts. It is
not a substitute for final liburing verification.

## Verification Profiles

```bash
python -m tools.check io
python -m tools.check liburing
```

The `io` profile builds with `BACKEND_MODE=posix`; the `liburing` profile builds
with `BACKEND_MODE=liburing`. Both profiles require Linux, `make`, and a POSIX C
toolchain. The liburing profile additionally requires liburing development
files and does not accept fallback as success. Generated probe evidence is
written under ignored `outputs/checks/<profile>/` directories and uploaded as
CI artifacts.

The legacy smoke target writes generated files under the ignored output tree:

- `outputs/phase2/io_probe.bin`
- `outputs/phase2/io_probe_metrics.json`

## Python Harness Bridge

`harness/io_probe.py` wraps `io/block_io_probe` and exposes read/write latency
metrics to Python. `harness/runtime_replay.py --calibrate-ssd-io` uses this path
to calibrate SSD-read latency from the real liburing probe before running the
scheduler replay.

## Runtime KV Command Consumer

`runtime_kv_command_consumer` is a Phase 4 bridge prototype. It reads llama.cpp-oriented KV command JSONL, seeds a backing SSD file for SSD-source blocks, executes `kv_load` / `kv_prefetch` / `kv_evict` through liburing when available, and uses mock DRAM/VRAM buffers for non-SSD movement and compute verification.

The Linux I/O profiles build this consumer together with `block_io_probe` and
exercise its marked integration tests.
