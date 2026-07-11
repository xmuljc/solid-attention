# Reproducible Harness Baseline Design

**Date:** 2026-07-11
**Status:** Approved

## Summary

SolidAttention already has a useful simulation and runtime-integration harness,
but the imported Git repository does not provide a trustworthy clean-clone
feedback loop. Most tracked paths are dependency or build caches, generated
outputs are committed, six gitlinks have no `.gitmodules` mapping, and the
default test suite fails on Windows for environment-shape reasons rather than
product behavior.

This design establishes a small, reproducible engineering harness before more
Phase 4 runtime work. The repository will become the system of record for setup,
verification, experiment metadata, and dependency boundaries. A Windows-only
developer can run the required fast checks locally; GitHub-hosted Linux runners
will own the Linux, liburing, and llama.cpp checks.

## Approved Decisions

- **A1:** Replace the current one-commit history with a clean root commit and
  update GitHub `main` with `--force-with-lease`.
- **P1:** Require the fast suite on Windows and Linux. Run Linux-only I/O,
  liburing, and llama.cpp verification on GitHub Actions.
- **D1:** Keep only `external/llama.cpp` as a real Git submodule, pinned to
  commit `bbebeec4a87355896e3faac0c2baca8130c91b6a`.
- **O1:** Do not track generated `outputs/`. Keep only small deterministic test
  data under `tests/fixtures/`; publish CI outputs as workflow artifacts.
- **H2:** Build a contract-first baseline: clean repository, one verification
  entry point, explicit test profiles, CI gates, deterministic fixtures, and a
  versioned run manifest.

## Current Evidence

- The repository contains one imported root commit.
- The default Windows run reports 105 passed, 6 failed, and 2 skipped.
- Four failures execute Python fixture scripts as native executables and fail
  with Windows error 193.
- Two failures treat an empty, uninitialized llama.cpp gitlink directory as a
  usable checkout.
- Six mode-160000 gitlinks exist, but no `.gitmodules` file maps them.
- `solid attention/.deps/` and `solid_attention_deps/` contain 7,956 tracked
  paths and about 258 MiB of dependency/build data.
- Generated `outputs/` and compiled I/O binaries are tracked.
- There is no Python dependency contract, unified check command, or CI workflow.

## Goals

1. Make a fresh clone small and understandable.
2. Make the required local Windows feedback loop deterministic and green.
3. Make Linux-only capabilities explicit, observable, and mechanically gated.
4. Ensure optional prerequisites cannot silently turn a required CI job green.
5. Preserve the existing Python module boundaries and Phase 4 implementation.
6. Give every experiment a versioned, machine-readable run manifest.
7. Keep the documentation aligned with commands that work from a fresh clone.

## Non-Goals

- No scheduler algorithm changes.
- No new llama.cpp runtime mutation behavior.
- No real GGUF download or model execution.
- No production SSD benchmark thresholds on shared CI hardware.
- No `src/` package-layout migration.
- No comprehensive lint, type-check, coverage, or pre-commit platform in this
  iteration.

## Repository Layout

The current `solid attention/` contents move to the Git repository root. The
space-bearing wrapper directory and `solid_attention_project` pointer disappear.

```text
AGENTS.md
README.md
pyproject.toml
.gitmodules
.github/workflows/ci.yml
core/
harness/
integrations/
io/
tests/
  fixtures/
configs/
docs/
  superpowers/specs/
tools/
external/
  llama.cpp/
outputs/                 # generated and ignored
```

The following content is excluded from the new root history:

- `.deps/`
- `solid_attention_deps/`
- temporary llama.cpp worktrees and their gitlinks
- generated `outputs/`
- compiled I/O binaries
- Python and pytest caches
- local model, checkpoint, key, and environment files

Existing imports such as `core`, `harness`, and `integrations` remain valid.

## History And Dependency Boundaries

Before rewriting `main`, the current commit is retained through a local-only
safety reference. That reference is not pushed because doing so would keep the
large imported objects reachable from normal remote clones.

The clean tree contains exactly one gitlink:

```text
external/llama.cpp -> bbebeec4a87355896e3faac0c2baca8130c91b6a
```

`.gitmodules` maps it to the official llama.cpp repository. Normal clones and
the fast suite do not initialize the submodule. The Linux llama profile performs
recursive initialization and verifies the exact revision before patch checks.

## Python Environment Contract

`pyproject.toml` declares the supported Python range and a development extra
containing pytest. Runtime modules continue to use the standard library unless
an existing feature requires otherwise.

A fresh developer setup is:

```text
python -m venv .venv
python -m pip install -e ".[dev]"
python -m tools.check fast
```

The README provides PowerShell and POSIX activation examples, but verification
commands remain shell-neutral Python module invocations.

## Verification Profiles

`python -m tools.check <profile>` is the single verification interface.

### `fast`

- Runs on Windows and Linux.
- Requires no compiler, liburing, model, or initialized submodule.
- Covers pure Python core/harness behavior, deterministic fixtures, manifest
  contracts, patch text contracts, and dependency probes.
- Must finish with zero failures and zero unexpected skips.

### `io`

- Runs on Linux.
- Builds the C I/O tools with the POSIX fallback and runs their integration
  tests.
- Reports the selected backend explicitly.

### `liburing`

- Runs on Linux with `liburing-dev` installed.
- Builds and tests the I/O tools.
- Fails unless the selected backend is exactly `liburing`.

### `llama`

- Runs on Linux.
- Initializes `external/llama.cpp`, verifies the pinned revision, applies
  `git apply --check`, validates the integration map, and builds the existing
  smoke target.
- Missing or wrong submodule state is a failure in this profile.

Profiles produce a concise console summary and a JSON check result suitable for
CI artifact upload.

## Test Classification And Prerequisites

Pytest markers classify environment-sensitive tests as `io`, `liburing`, or
`llama`; unmarked deterministic tests belong to `fast`. Optional local tests
may skip only when invoked outside their owning profile. A required profile
performs a prerequisite check first and treats an unavailable required
dependency as a profile failure, not a passing skip.

Cross-platform command execution is centralized. A `.py` executable is invoked
with the active `sys.executable`; native tools are invoked directly. This fixes
the current Windows error without weakening production executable checks.

The llama.cpp probe reports one of four states:

- `missing`
- `uninitialized`
- `invalid`
- `ready`

The state and missing anchors are serializable. Fast tests accept the first two
states for an optional checkout; the llama profile requires `ready`.

## CI Design

`.github/workflows/ci.yml` contains independently visible jobs:

1. `fast` matrix on `windows-latest` and `ubuntu-latest`.
2. `io-posix` on Ubuntu.
3. `io-liburing` on Ubuntu after installing `liburing-dev`.
4. `llama-integration` on Ubuntu with submodule initialization.

Each job invokes `python -m tools.check`, rather than duplicating verification
logic in YAML. Check JSON, manifests, traces, metrics, and relevant build logs
are uploaded as workflow artifacts even when a job fails where feasible.

Before rewriting `main`, the final parentless commit is pushed to the temporary
`harness-candidate` branch and must pass this same workflow. Only that verified
commit may replace `main`; the temporary branch is removed after the new
`main` run and fresh-clone checks pass.

Real SSD performance thresholds remain outside hosted CI because shared virtual
hardware does not provide a stable physical-device baseline.

## Run Manifest Contract

All experiment-style entry points create `run_manifest.json` using the schema
identifier `solidattention.run_manifest.v1`. Existing trace and metrics schemas
remain compatible.

For this baseline, "experiment-style entry points" means `smoke`, `ablation`,
`runtime_replay`, `movement_executor`, and `runtime_residency_executor`: they
own a run directory and produce trace/metrics/summary bundles. llama.cpp
adapters, schema converters, patch validators, and the evict smoke are
verification utilities; they remain covered by `solidattention.check_result.v1`
and their existing focused artifacts rather than creating nested run manifests.

Required manifest fields are:

- schema identifier and run ID
- UTC start/end timestamps, duration, and final status
- command as a complete argv list (executable/module plus arguments), and profile
- Git SHA and dirty-worktree flag
- Python version, operating system, and CPU architecture
- selected I/O backend (`mock`, `fake`, `posix_fallback`, or `liburing`)
- normalized configuration snapshot and content hash
- input paths and content hashes
- output trace, metrics, summary, and log paths
- failure type and concise failure message when status is `failed`

Tests inject a clock and run ID so manifests are deterministic. Output paths are
relative to the run directory when possible. Secrets and environment-variable
values are never copied into the manifest.

## Data Flow

```text
config + input artifacts
        |
        v
     RunContext
        |
        +--> smoke / ablation / replay / executor
        |          |
        |          +--> trace
        |          +--> metrics
        |          +--> summary
        |
        +--> run_manifest.json
```

`RunContext` records a failed manifest when an experiment raises after the run
directory is available, then re-raises so the CLI returns nonzero.

## Error Handling

- Invalid configuration and input contracts fail before expensive work.
- Required profile prerequisites fail with a short actionable diagnostic.
- subprocess errors retain the command, exit code, and bounded stdout/stderr in
  the check artifact.
- A failed experiment writes `status=failed` when possible and never reports a
  successful summary.
- CI does not convert missing liburing, wrong submodule revision, or an
  unbuildable llama patch into a passing skip.

## Documentation Updates

- README becomes the clean-clone setup and command entry point.
- AGENTS records the profile ownership rules and completion gates.
- ROADMAP and Phase status documents retain the existing research sequence but
  replace obsolete paths, historical output claims, and stale Phase 3 status.
- Generated results are referenced through reproducible commands and CI
  artifacts rather than committed output directories.
- This harness baseline is documented as a prerequisite to the next Phase 4
  `load-before-attention` vertical slice, not as a scheduler feature itself.

## Verification And Acceptance

The change is complete only when all of the following are proven:

1. Windows `python -m tools.check fast` reports zero failures and zero
   unexpected skips.
2. The new root tree has no cache directory, generated output, compiled binary,
   or temporary worktree gitlink.
3. `external/llama.cpp` is the only gitlink and has a valid `.gitmodules` entry.
4. The clean root commit is pushed with `--force-with-lease`.
5. A fresh non-recursive GitHub clone passes the fast profile without the
   llama.cpp submodule.
6. GitHub Actions fast, I/O POSIX, liburing, and llama jobs all pass under their
   explicit backend and prerequisite contracts.
7. Experiment entry points emit schema-valid manifests in focused tests.
8. README, AGENTS, and Phase documents describe paths and commands that match the
   final tree.

## Risks And Mitigations

- **History rewrite:** use `--force-with-lease`, verify the expected remote old
  SHA, gate the parentless commit on a temporary candidate branch, and keep a
  local-only safety reference until the new clone is proven.
- **Large llama.cpp CI cost:** initialize and build only in the dedicated job;
  keep fast jobs independent and cache build inputs only through CI caches.
- **Platform drift:** keep prerequisite detection and profile logic in tested
  Python code rather than shell-specific workflow steps.
- **Documentation drift:** make the unified check commands the only documented
  verification surface and update Phase docs in the same change.
- **Scope expansion:** defer scheduler behavior, real GGUF runs, static-analysis
  platform work, and performance thresholds to later designs.
