# Harness Baseline Status

## Scope

The baseline provides a clean-clone, harness-first reproduction of the
SolidAttention simulation, trace, scheduling, movement-plan, and runtime bridge
contracts. It does not claim a production LLM runtime or completed real-model
attention integration.

## Profiles

`python -m tools.check fast` is the Windows and Linux development gate. Linux
owns three additional profiles: `io` requires the POSIX C backend, `liburing`
requires the real liburing backend, and `llama` verifies the pinned llama.cpp
submodule with Git and CMake. Required profiles fail on missing prerequisites,
test skips, or backend fallback.

## Run Manifest

Experiment entry points write JSON trace, metrics, and summary artifacts plus a
`solidattention.run_manifest.v1` manifest. The manifest records the selected
profile, final status, configuration and revision metadata, and SHA-256 hashes
for declared outputs. Generated runs live under ignored `outputs/<run-id>/`
directories. Its top-level fields are `schema`, `run_id`, `profile`, `command`,
`started_at`, `ended_at`, `duration_ms`, `status`, `failure`, `git`, `runtime`,
`io_backend`, `config`, `inputs`, and `outputs`; the authoritative schema writer
is `harness/run_manifest.py`.

## Clean Clone Contract

Maintained Python packages, tests, configuration, documentation, and source live
at the Git root. The tree contains exactly one gitlink,
`external/llama.cpp`, pinned at
`bbebeec4a87355896e3faac0c2baca8130c91b6a`. The fast profile checks that
generated outputs, dependency caches, compiled I/O tools, model files, and
secrets are not tracked. Deterministic test inputs are kept under
`tests/fixtures/`.

## CI Evidence

The GitHub Actions workflow defines the fast profile on Windows and Linux for
Python 3.11 and 3.13, plus Linux POSIX I/O, liburing, and llama.cpp jobs. Each job
uploads its structured profile output even when a check fails. The first
`harness-candidate` run will provide the remote CI evidence for this baseline;
this document does not claim that candidate run has already passed. Artifact
names are `check-fast-<os>-py<version>`, `check-io-posix`,
`check-io-liburing`, and `check-llama`; a retried job overwrites its prior
artifact for the same workflow run.

## Next Runtime Slice

Scheduler-timed load-before-attention is not implemented in this baseline.
The next vertical slice must issue scheduler-owned `kv_load` and `kv_prefetch`
operations early enough to make selected K/V blocks resident before llama.cpp
attention consumes `get_k` and `get_v` views. Post-decode persistence and
eviction, real scheduler-selected backing contents, and measured liburing
latency remain subsequent runtime work.
