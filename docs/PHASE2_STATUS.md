# Phase 2 Verification Status

Phase 2 adds a standalone block I/O prototype under `io/`. The prototype is kept
separate from the Python simulation harness and from llama.cpp runtime changes.

## Implemented

- `io/block_io_probe.c`: fixed-size block write/read/verify probe.
- `io/Makefile`: build, backend detection, and smoke target.
- JSON metrics output with backend, bytes, block size/count, write latency, and read latency.
- Compile-time `HAVE_LIBURING` path.
- POSIX fallback path for dependency-limited hosts.

## Backend Profiles

`python -m tools.check io` forces the POSIX fallback backend on Linux.
`python -m tools.check liburing` forces liburing and fails if the headers or
library are unavailable. Fallback is not accepted as liburing evidence.

## Verified Commands

```bash
python -m tools.check io
python -m tools.check liburing
```

The imported snapshot recorded the following historical liburing evidence.
Generated probe files are no longer tracked; current runs write ignored profile
artifacts under `outputs/checks/`.

```text
make -C io backend -> liburing
make -C io smoke -> {"backend":"liburing","block_size":4096,"block_count":4,"bytes":16384,...}
full pytest -> 69 passed in 0.46s
```

The imported snapshot verified the standalone liburing block I/O prototype.

## Python Harness Bridge

- `harness/io_probe.py` wraps `io/block_io_probe` from Python and parses JSON metrics.
- `harness/runtime_replay.py --calibrate-ssd-io` can now run the real block I/O probe and use measured read latency as scheduler SSD-read latency.

Historical calibrated replay:

```text
backend: liburing
block_size: 65536
block_count: 4
read_latency_ms_per_block: 0.03888325
```

## Movement Executor Bridge

`harness/movement_executor.py` consumes `runtime_movement_plan.json` and executes
unique SSD `kv_load` ranges through `io/block_io_probe`.

Historical execution summary:

```text
backend: liburing
block_size: 65536
block_count: 4
ssd_read_bytes: 262144
read_latency_ms_per_block: 0.037602
```
