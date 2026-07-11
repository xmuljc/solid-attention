# Reproducible Harness Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the imported snapshot with a small, clean-clone repository whose Windows/Linux fast checks, Linux I/O checks, pinned llama.cpp integration, and experiment manifests are explicit and mechanically verifiable.

**Architecture:** Keep the existing `core`, `harness`, and `integrations` module boundaries, but move them to the Git root and remove generated/cache content from the rewritten history. Put all environment-sensitive verification behind `python -m tools.check <profile>`, keep llama.cpp as the only pinned submodule, and add a shared `RunContext` that writes `solidattention.run_manifest.v1` for experiment entry points.

**Tech Stack:** Python 3.11+, pytest, setuptools/`pyproject.toml`, C11/Make, liburing on Linux, llama.cpp at `bbebeec4a87355896e3faac0c2baca8130c91b6a`, GitHub Actions, JSON/JSONL.

---

## Path Convention

Tasks 0-1 run before the repository is flattened, so project paths begin with
`solid attention/`. Task 2 moves the project to the Git root. Every path in
Task 3 and later is relative to the new Git root.

Do not push the reviewable Task commits. They provide local review points and
are collapsed into one clean root commit in Task 14; only that parentless
commit is pushed to the temporary candidate branch before replacing `main`.

### Task 0: Isolate Work And Record The Baseline

**Files:**
- Read: `solid attention/docs/superpowers/specs/2026-07-11-harness-baseline-design.md`
- No repository file changes

- [ ] **Step 1: Create an isolated implementation worktree**

Before invoking the worktree skill, verify that this approved plan and design
are committed, the original workspace is clean, and record its path in local
Git configuration without hard-coding the Unicode user directory:

```powershell
$pending = (& git status --porcelain) -join "`n"
if ($LASTEXITCODE -ne 0) { throw "git status failed" }
if ($pending) { throw "original workspace must be clean before creating the implementation worktree" }
git cat-file -e 'HEAD:solid attention/docs/superpowers/specs/2026-07-11-harness-baseline-design.md'
if ($LASTEXITCODE -ne 0) { throw "approved design is not committed" }
git cat-file -e 'HEAD:solid attention/docs/superpowers/plans/2026-07-11-harness-baseline.md'
if ($LASTEXITCODE -ne 0) { throw "approved implementation plan is not committed" }
$original = (& git rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0) { throw "cannot resolve original worktree" }
$originalMain = (& git rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw "cannot resolve original local main" }
git update-ref refs/backup/pre-harness-local-main $originalMain
if ($LASTEXITCODE -ne 0) { throw "cannot preserve original local main" }
git config --local solidattention.original-worktree $original
if ($LASTEXITCODE -ne 0) { throw "cannot record original worktree" }
```

Invoke `superpowers:using-git-worktrees` and create a worktree for branch
`harness-baseline`. All commands through Task 13 run in that worktree.

Expected: the original workspace remains on `main`; the implementation
worktree is on `harness-baseline` with a clean status.

- [ ] **Step 2: Fetch and verify the remote rewrite lease**

Run from the implementation worktree:

```powershell
git fetch origin main
if ($LASTEXITCODE -ne 0) { throw "origin/main fetch failed" }
$remoteMain = (git rev-parse origin/main).Trim()
if ($LASTEXITCODE -ne 0) { throw "cannot resolve origin/main" }
if ($remoteMain -ne "d0d2943444d7184840ece2308d71b264e2e9c299") {
    throw "origin/main changed; expected d0d2943444d7184840ece2308d71b264e2e9c299, got $remoteMain"
}
git update-ref refs/backup/pre-harness-rewrite $remoteMain
if ($LASTEXITCODE -ne 0) { throw "cannot create pre-rewrite backup ref" }
```

Expected: `refs/backup/pre-harness-rewrite` points at `d0d2943`. Do not push
this local-only backup ref.

- [ ] **Step 3: Reproduce the Windows baseline**

Run:

```powershell
$baseline = (& python -m pytest -q 2>&1) -join "`n"
$baselineExit = $LASTEXITCODE
$baseline | Write-Host
if ($baselineExit -ne 1) { throw "baseline pytest exit was $baselineExit, expected 1" }
foreach ($expected in @('6 failed', '105 passed', '2 skipped', 'WinError 193')) {
    if (-not $baseline.Contains($expected)) { throw "baseline output missing: $expected" }
}
```

Expected on the current Windows host: `6 failed, 105 passed, 2 skipped`. Four
failures contain `WinError 193`; two report the empty llama.cpp checkout as
present but invalid.

- [ ] **Step 4: Confirm the imported-tree evidence**

Run:

```powershell
$tree = (& git ls-tree -r HEAD) -join "`n"
if ($LASTEXITCODE -ne 0) { throw "git ls-tree failed" }
$gitlinks = @($tree -split "`n" | Select-String '^160000')
if ($gitlinks.Count -ne 6) { throw "expected six imported gitlinks, got $($gitlinks.Count)" }
$submoduleOutput = (& git submodule status 2>&1) -join "`n"
$submoduleExit = $LASTEXITCODE
$submoduleOutput | Write-Host
if ($submoduleExit -eq 0) { throw "broken imported submodule metadata unexpectedly passed" }
if (-not $submoduleOutput.Contains('no submodule mapping found')) {
    throw "submodule failure did not report the expected missing mapping"
}
```

Expected: six gitlinks are listed and `git submodule status` fails because no
`.gitmodules` mapping exists. Save the output in the task notes; do not create a
tracked output file.

### Task 1: Add A Mechanized Repository Contract

**Files:**
- Create: `solid attention/tools/__init__.py`
- Create: `solid attention/tools/repository_contract.py`
- Create: `solid attention/tests/test_repository_contract.py`

- [ ] **Step 1: Write contract unit tests and a current-tree gate**

Create `solid attention/tests/test_repository_contract.py`:

```python
from pathlib import Path

from tools.repository_contract import RepoEntry, check_repository, validate_entries


def test_validate_entries_accepts_clean_tree() -> None:
    entries = [
        RepoEntry("100644", "README.md"),
        RepoEntry("100644", "AGENTS.md"),
        RepoEntry("100644", ".gitmodules"),
        RepoEntry("100644", "pyproject.toml"),
        RepoEntry("100644", "core/__init__.py"),
        RepoEntry("160000", "external/llama.cpp"),
    ]

    assert validate_entries(entries) == []


def test_validate_entries_reports_cache_binary_wrapper_and_gitlink() -> None:
    entries = [
        RepoEntry("100644", "solid attention/README.md"),
        RepoEntry("100644", ".deps/python/pytest.py"),
        RepoEntry("100755", "io/block_io_probe"),
        RepoEntry("100644", "io/runtime_kv_command_consumer.exe"),
        RepoEntry("100644", "io/runtime_kv_command_consumer.obj"),
        RepoEntry("160000", "external/temporary-worktree"),
    ]

    violations = validate_entries(entries)

    assert "wrapper path is tracked: solid attention/README.md" in violations
    assert "generated/cache path is tracked: .deps/python/pytest.py" in violations
    assert "compiled binary is tracked: io/block_io_probe" in violations
    assert "compiled binary is tracked: io/runtime_kv_command_consumer.exe" in violations
    assert "compiled binary is tracked: io/runtime_kv_command_consumer.obj" in violations
    assert "unexpected gitlink: external/temporary-worktree" in violations


def test_current_repository_contract_is_clean() -> None:
    violations = check_repository(Path(__file__).resolve())

    assert violations == []
```

- [ ] **Step 2: Run the contract tests to verify the repository gate fails**

Run:

```powershell
Set-Location 'solid attention'
python -m pytest -q tests/test_repository_contract.py
```

Expected: the two synthetic tests fail with `ModuleNotFoundError` until the
implementation exists. After Step 3, they pass and the current-tree test fails
with wrapper/cache/binary/unexpected-gitlink violations.

- [ ] **Step 3: Implement the repository contract**

Create an empty `solid attention/tools/__init__.py` and create
`solid attention/tools/repository_contract.py`:

```python
"""Mechanical source-tree boundaries for a clean SolidAttention clone."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FORBIDDEN_PREFIXES = (
    "solid attention/",
    "solid_attention_deps/",
    ".deps/",
    "outputs/",
)
COMPILED_BINARIES = {
    "io/block_io_probe",
    "io/block_io_probe.exe",
    "io/runtime_kv_command_consumer",
    "io/runtime_kv_command_consumer.exe",
}
COMPILED_SUFFIXES = {".o", ".obj"}
ALLOWED_GITLINKS = {"external/llama.cpp"}
REQUIRED_PATHS = {"README.md", "AGENTS.md", ".gitmodules", "pyproject.toml"}


@dataclass(frozen=True)
class RepoEntry:
    mode: str
    path: str


def find_git_root(start: Path) -> Path:
    completed = subprocess.run(
        ["git", "-C", str(start.parent if start.is_file() else start), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(completed.stdout.strip())


def read_index(root: Path) -> list[RepoEntry]:
    completed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--stage", "-z"],
        check=True,
        capture_output=True,
    )
    entries: list[RepoEntry] = []
    for raw_record in completed.stdout.split(b"\0"):
        if not raw_record:
            continue
        header, raw_path = raw_record.split(b"\t", 1)
        mode = header.split(b" ", 1)[0].decode("ascii")
        entries.append(RepoEntry(mode=mode, path=raw_path.decode("utf-8")))
    return entries


def validate_entries(entries: Iterable[RepoEntry]) -> list[str]:
    items = list(entries)
    paths = {item.path for item in items}
    violations: list[str] = []
    for item in items:
        if item.path.startswith("solid attention/"):
            violations.append(f"wrapper path is tracked: {item.path}")
        elif item.path.startswith(FORBIDDEN_PREFIXES[1:]):
            violations.append(f"generated/cache path is tracked: {item.path}")
        if item.path in COMPILED_BINARIES or (
            item.path.startswith("io/") and Path(item.path).suffix.lower() in COMPILED_SUFFIXES
        ):
            violations.append(f"compiled binary is tracked: {item.path}")
        if item.mode == "160000" and item.path not in ALLOWED_GITLINKS:
            violations.append(f"unexpected gitlink: {item.path}")
    for required in sorted(REQUIRED_PATHS - paths):
        violations.append(f"required path is missing: {required}")
    gitlinks = {item.path for item in items if item.mode == "160000"}
    if gitlinks != ALLOWED_GITLINKS:
        violations.append(
            f"gitlinks must be {sorted(ALLOWED_GITLINKS)}, got {sorted(gitlinks)}"
        )
    return sorted(set(violations))


def check_repository(start: Path) -> list[str]:
    root = find_git_root(start)
    return validate_entries(read_index(root))
```

- [ ] **Step 4: Re-run the focused tests**

Run:

```powershell
python -m pytest -q tests/test_repository_contract.py
```

Expected: two tests pass and `test_current_repository_contract_is_clean`
fails, listing the current imported-tree violations. This is the intended red
gate for Task 2.

- [ ] **Step 5: Commit the red repository contract**

Run from the Git root:

```powershell
git add -- 'solid attention/tools' 'solid attention/tests/test_repository_contract.py'
git commit -m 'test: define clean repository contract'
```

### Task 2: Flatten The Source Tree And Remove Imported Artifacts

**Files:**
- Move: `solid attention/{core,harness,integrations,io,tests,configs,docs,tools,external}` to Git root
- Move: `solid attention/{README.md,AGENTS.md}` to Git root
- Move: `solid attention/SolidAttention - Low-Latency SSD-based Serving on Memory-Constrained PCs.pdf` to `docs/papers/solidattention-paper.pdf`
- Delete: `solid attention/.deps/`
- Delete: `solid_attention_deps/`
- Delete: `solid attention/outputs/`
- Delete: `solid attention/io/block_io_probe`
- Delete: `solid attention/io/runtime_kv_command_consumer`
- Delete: `solid_attention_project`
- Delete: `solid attention/pytest.ini`
- Replace: `.gitignore`
- Create: `.gitmodules`
- Create: `pyproject.toml`
- Create: `tests/fixtures/README.md`

- [ ] **Step 1: Remove generated/cache trees and compiled binaries from the index**

Run from the Git root:

```powershell
git rm -r -- 'solid attention/.deps' 'solid attention/outputs' 'solid_attention_deps'
git rm -- 'solid attention/io/block_io_probe' 'solid attention/io/runtime_kv_command_consumer'
git rm -- 'solid attention/pytest.ini' 'solid_attention_project'
```

Expected: all imported build caches, generated outputs, compiled I/O binaries,
and temporary worktree gitlinks are staged for deletion.

- [ ] **Step 2: Move the maintained project to the Git root**

Run:

```powershell
git mv 'solid attention/core' 'core'
git mv 'solid attention/harness' 'harness'
git mv 'solid attention/integrations' 'integrations'
git mv 'solid attention/io' 'io'
git mv 'solid attention/tests' 'tests'
git mv 'solid attention/configs' 'configs'
git mv 'solid attention/docs' 'docs'
git mv 'solid attention/tools' 'tools'
git mv 'solid attention/external' 'external'
git mv 'solid attention/README.md' 'README.md'
git mv 'solid attention/AGENTS.md' 'AGENTS.md'
New-Item -ItemType Directory -Force -Path 'docs/papers' | Out-Null
git mv 'solid attention/SolidAttention - Low-Latency SSD-based Serving on Memory-Constrained PCs.pdf' 'docs/papers/solidattention-paper.pdf'
```

Expected: no tracked path begins with `solid attention/`.

- [ ] **Step 3: Replace `.gitignore` with source-boundary rules**

Replace `.gitignore` with:

```gitignore
# Python
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
dist/
build/
*.egg-info/

# Local tools and editors
.codex/
.agents/
.vscode/
.idea/

# Generated experiment and check artifacts
outputs/

# Dependency and build caches
.deps/
solid_attention_deps/
CMakeFiles/
CMakeCache.txt
cmake_install.cmake

# Compiled I/O tools
/io/block_io_probe
/io/block_io_probe.exe
/io/runtime_kv_command_consumer
/io/runtime_kv_command_consumer.exe
/io/*.o
/io/*.obj

# Models, checkpoints, and secrets
FlexGen/
*.gguf
*.pt
*.pth
*.ckpt
*.bin
*.safetensors
.env
*.key
*.pem
```

- [ ] **Step 4: Define the one valid submodule**

Create `.gitmodules`:

```ini
[submodule "external/llama.cpp"]
	path = external/llama.cpp
	url = https://github.com/ggerganov/llama.cpp.git
```

Run:

```powershell
git config -f .gitmodules --get submodule.external/llama.cpp.path
git ls-files --stage external/llama.cpp
```

Expected: the path is `external/llama.cpp` and its index mode is `160000` at
`bbebeec4a87355896e3faac0c2baca8130c91b6a`.

- [ ] **Step 5: Add the Python project contract**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools==80.9.0"]
build-backend = "setuptools.build_meta"

[project]
name = "solidattention-harness"
version = "0.1.0"
description = "Harness-first reproduction of SolidAttention mechanisms"
readme = "README.md"
requires-python = ">=3.11"
dependencies = []

[project.optional-dependencies]
dev = ["pytest==8.4.1"]

[tool.setuptools]
packages = [
  "core",
  "harness",
  "integrations",
  "integrations.llama_cpp",
  "tools",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = ["--strict-config", "--strict-markers"]
markers = [
  "io: requires a Linux C/POSIX toolchain",
  "liburing: requires Linux and liburing",
  "llama: requires the pinned llama.cpp submodule",
]
```

Create `tests/fixtures/README.md`:

```markdown
# Test Fixtures

Only small, deterministic inputs required by automated tests belong here.
Generated traces, metrics, manifests, models, and benchmark outputs belong in
the ignored `outputs/` directory or in CI artifacts.
```

- [ ] **Step 6: Run the repository contract green**

Run:

```powershell
git add -- .gitignore .gitmodules pyproject.toml tests/fixtures/README.md
python -m pytest -q tests/test_repository_contract.py
git submodule status
git ls-files --stage | Select-String '^160000'
```

Expected: all three repository contract tests pass. `git submodule status`
prints a single uninitialized `external/llama.cpp` entry (leading `-`), and the
staged index contains exactly that one gitlink.

- [ ] **Step 7: Commit the clean source layout**

Run:

```powershell
git add --all
git commit -m 'build: establish clean source boundaries'
```

### Task 3: Make Executable Invocation Cross-Platform

**Files:**
- Create: `harness/process.py`
- Create: `tests/test_process.py`
- Modify: `harness/io_probe.py`
- Modify: `harness/runtime_command_consumer.py`
- Modify: `integrations/llama_cpp/evict_smoke.py`

- [ ] **Step 1: Write failing command-construction tests**

Create `tests/test_process.py`:

```python
import sys
from pathlib import Path

from harness.process import build_executable_command


def test_python_script_uses_active_interpreter(tmp_path: Path) -> None:
    script = tmp_path / "probe.py"
    script.write_text("print('ok')\n", encoding="utf-8")

    command = build_executable_command(script, "output.bin", 4096, 2)

    assert command == [sys.executable, str(script), "output.bin", "4096", "2"]


def test_native_executable_is_invoked_directly(tmp_path: Path) -> None:
    executable = tmp_path / "probe.exe"
    executable.write_bytes(b"")

    command = build_executable_command(executable, "output.bin")

    assert command == [str(executable), "output.bin"]
```

- [ ] **Step 2: Run the new tests red**

Run:

```powershell
python -m pytest -q tests/test_process.py
```

Expected: collection fails because `harness.process` does not exist.

- [ ] **Step 3: Implement the shared command builder**

Create `harness/process.py`:

```python
"""Cross-platform subprocess command construction."""

from __future__ import annotations

import sys
from pathlib import Path


def build_executable_command(executable: str | Path, *args: object) -> list[str]:
    path = Path(executable)
    command = [sys.executable, str(path)] if path.suffix.lower() == ".py" else [str(path)]
    command.extend(str(arg) for arg in args)
    return command
```

In `harness/io_probe.py`, import `build_executable_command` and replace the
literal subprocess argument list with:

```python
build_executable_command(executable_path, output_path, block_size, block_count)
```

In `harness/runtime_command_consumer.py`, replace its argument list with:

```python
build_executable_command(executable_path, command_path, backing_path)
```

In `integrations/llama_cpp/evict_smoke.py`, replace its argument list with:

```python
build_executable_command(executable, *args)
```

Keep the existing `check`, `capture_output`, `text`, and `env` keyword
arguments unchanged in both calls.

- [ ] **Step 4: Verify the Windows regression is fixed**

Run:

```powershell
python -m pytest -q tests/test_process.py tests/test_io_probe.py tests/test_movement_executor.py tests/test_runtime_replay.py
```

Expected: all focused tests pass; no `WinError 193` appears.

- [ ] **Step 5: Commit the process boundary**

Run:

```powershell
git add harness/process.py harness/io_probe.py harness/runtime_command_consumer.py integrations/llama_cpp/evict_smoke.py tests/test_process.py
git commit -m 'fix: run Python probes portably'
```

### Task 4: Model llama.cpp Checkout State Explicitly

**Files:**
- Modify: `integrations/llama_cpp/probe.py`
- Modify: `tests/test_llama_cpp_integration.py`
- Modify: `tests/test_llama_cpp_integration_map.py`
- Modify: `tests/test_llama_cpp_patch.py`

- [ ] **Step 1: Add failing checkout-state tests**

Add to `tests/test_llama_cpp_integration.py`:

```python
def test_probe_reports_uninitialized_checkout(tmp_path) -> None:
    checkout = tmp_path / "llama.cpp"
    checkout.mkdir()

    result = probe_llama_cpp(checkout)

    assert result.state == "uninitialized"
    assert result.exists
    assert not result.ready


def test_probe_reports_invalid_nonempty_checkout(tmp_path) -> None:
    checkout = tmp_path / "llama.cpp"
    checkout.mkdir()
    (checkout / "unexpected.txt").write_text("partial\n", encoding="utf-8")

    result = probe_llama_cpp(checkout)

    assert result.state == "invalid"
    assert result.exists
    assert not result.ready
```

Extend existing assertions so missing returns `state == "missing"`, a minimal
valid checkout returns `state == "ready"`, and a partial checkout returns
`state == "invalid"`.

- [ ] **Step 2: Run the focused probe tests red**

Run:

```powershell
python -m pytest -q tests/test_llama_cpp_integration.py -k probe
```

Expected: failures report that `LlamaCppProbeResult` has no `state` attribute.

- [ ] **Step 3: Implement the four-state probe**

Replace the result model and `probe_llama_cpp` logic in
`integrations/llama_cpp/probe.py` with the following public contract while
retaining `EXPECTED_ANY_OF` and `EXPECTED_REQUIRED`:

```python
from typing import Literal

CheckoutState = Literal["missing", "uninitialized", "invalid", "ready"]


@dataclass(frozen=True)
class LlamaCppProbeResult:
    path: str
    state: CheckoutState
    missing_required: tuple[str, ...]
    missing_any_of: tuple[tuple[str, ...], ...]

    @property
    def exists(self) -> bool:
        return self.state != "missing"

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "state": self.state,
            "exists": self.exists,
            "missing_required": list(self.missing_required),
            "missing_any_of": [list(group) for group in self.missing_any_of],
            "ready": self.ready,
        }


def probe_llama_cpp(path: str | Path) -> LlamaCppProbeResult:
    root = Path(path)
    if not root.exists():
        return LlamaCppProbeResult(str(root), "missing", EXPECTED_REQUIRED, EXPECTED_ANY_OF)
    if root.is_dir() and not any(root.iterdir()):
        return LlamaCppProbeResult(str(root), "uninitialized", EXPECTED_REQUIRED, EXPECTED_ANY_OF)

    missing_required = tuple(item for item in EXPECTED_REQUIRED if not (root / item).exists())
    missing_any_of = tuple(
        group for group in EXPECTED_ANY_OF if not any((root / candidate).exists() for candidate in group)
    )
    state: CheckoutState = "ready" if not missing_required and not missing_any_of else "invalid"
    return LlamaCppProbeResult(str(root), state, missing_required, missing_any_of)
```

- [ ] **Step 4: Make optional real-checkout tests skip only missing/uninitialized states**

Change the real-checkout test in `tests/test_llama_cpp_integration.py` to:

```python
def test_real_llama_cpp_checkout_is_ready_when_initialized() -> None:
    result = probe_llama_cpp("external/llama.cpp")
    if result.state in {"missing", "uninitialized"}:
        pytest.skip(f"llama.cpp checkout is {result.state}")
    assert result.ready, result.to_dict()
```

In `tests/test_llama_cpp_integration_map.py`, start the real-checkout test with:

```python
probe = probe_llama_cpp(checkout)
if probe.state in {"missing", "uninitialized"}:
    pytest.skip(f"llama.cpp checkout is {probe.state}")
assert probe.ready, probe.to_dict()
```

Import `probe_llama_cpp` in that file. Make the identical change in
`tests/test_llama_cpp_patch.py` before calling `verify_patch`. An `invalid`
checkout reaches the `assert probe.ready` failure rather than skipping.

- [ ] **Step 5: Run the probe and former checkout failures green**

Run:

```powershell
python -m pytest -q tests/test_llama_cpp_integration.py tests/test_llama_cpp_integration_map.py tests/test_llama_cpp_patch.py
```

Expected on a non-recursive Windows clone: all pure tests pass and exactly the
three real-checkout tests skip as `uninitialized`; there are no failures.

- [ ] **Step 6: Commit the checkout-state contract**

Run:

```powershell
git add integrations/llama_cpp/probe.py tests/test_llama_cpp_integration.py tests/test_llama_cpp_integration_map.py tests/test_llama_cpp_patch.py
git commit -m 'fix: distinguish uninitialized llama checkout'
```

### Task 5: Separate Fast, I/O, liburing, And llama Tests

**Files:**
- Modify: `io/Makefile`
- Modify: `tests/test_phase2_io.py`
- Modify: `tests/test_runtime_command_consumer.py`
- Modify: `tests/test_llama_cpp_integration.py`
- Modify: `tests/test_llama_cpp_integration_map.py`
- Modify: `tests/test_llama_cpp_patch.py`

- [ ] **Step 1: Write a failing forced-backend test**

Add this pure contract test to `tests/test_phase2_io.py` before marking the
module as integration-only:

```python
def test_makefile_declares_explicit_backend_mode() -> None:
    makefile = Path("io/Makefile").read_text(encoding="utf-8")

    assert "BACKEND_MODE ?= auto" in makefile
    assert "BACKEND_MODE=liburing requested but liburing is unavailable" in makefile
    assert "/data/disk2/ljc" not in makefile
```

Run:

```powershell
python -m pytest -q tests/test_phase2_io.py::test_makefile_declares_explicit_backend_mode
```

Expected: FAIL because `BACKEND_MODE` does not yet exist.

- [ ] **Step 2: Add explicit Make backend selection**

In `io/Makefile`, remove `LOCAL_LIBURING_PREFIX` and its host-specific discovery
block. Replace it and the unconditional liburing availability assignment with:

```make
BACKEND_MODE ?= auto
LIBURING_CFLAGS ?=
LIBURING_LDFLAGS ?=

ifeq ($(BACKEND_MODE),posix)
  LIBURING_AVAILABLE := 0
else
  LIBURING_AVAILABLE := $(shell printf '#include <liburing.h>\nint main(void){return 0;}\n' | $(CC) $(LIBURING_CFLAGS) -x c - $(LIBURING_LDFLAGS) -luring -o /tmp/solid_attention_liburing_check >/dev/null 2>&1 && echo 1 || echo 0)
endif

ifeq ($(BACKEND_MODE),liburing)
  ifneq ($(LIBURING_AVAILABLE),1)
    $(error BACKEND_MODE=liburing requested but liburing is unavailable)
  endif
endif
```

Keep the existing `HAVE_LIBURING`, libraries, and `BACKEND` assignment. Add
validation near the top:

```make
ifneq ($(filter $(BACKEND_MODE),auto posix liburing),$(BACKEND_MODE))
  $(error BACKEND_MODE must be auto, posix, or liburing)
endif
```

- [ ] **Step 3: Make I/O tests obey the profile backend contract**

In `tests/test_phase2_io.py`, remove `LOCAL_LIBURING_PREFIX` and
`local_liburing_available`. Keep the Makefile contract test unmarked so it runs
in `fast`. Decorate both native tests
`test_phase2_make_backend_reports_supported_mode` and
`test_phase2_block_io_probe_builds_and_verifies_blocks` with both markers:

```python
@pytest.mark.io
@pytest.mark.liburing
```

Use these helpers for those native tests:

```python
import os

def expected_backend() -> str:
    value = os.environ.get("SOLIDATTENTION_EXPECTED_IO_BACKEND", "")
    if value not in {"posix_fallback", "liburing"}:
        pytest.skip("run through tools.check io or tools.check liburing")
    return value


def backend_mode() -> str:
    return "liburing" if expected_backend() == "liburing" else "posix"
```

Pass `f"BACKEND_MODE={backend_mode()}"` to every `make` invocation and replace
the current permissive backend assertions with exact equality to
`expected_backend()`.

Decorate only the native build test in
`tests/test_runtime_command_consumer.py`:

```python
@pytest.mark.io
@pytest.mark.liburing
def test_runtime_command_consumer_builds_and_executes_commands(tmp_path) -> None:
```

Define the same `expected_backend()` and `backend_mode()` helpers in that test
module. Pass `f"BACKEND_MODE={backend_mode()}"` to Make, but compare the
consumer result to `expected_backend()`. In particular, the reported
`posix_fallback` value maps to Make's accepted `BACKEND_MODE=posix`. Leave the
dataclass/unit test unmarked so it stays in the fast suite.

- [ ] **Step 4: Mark only real llama.cpp checkout tests**

Add `@pytest.mark.llama` to these three tests:

```text
tests/test_llama_cpp_integration.py::test_real_llama_cpp_checkout_is_ready_when_initialized
tests/test_llama_cpp_integration_map.py::test_real_checkout_integration_map_is_valid_when_present
tests/test_llama_cpp_patch.py::test_solidattention_kv_trace_patch_applies_to_real_checkout_when_present
```

All fixture-based patch, adapter, parser, and integration-map tests remain
unmarked and therefore remain part of `fast`.

- [ ] **Step 5: Verify the Windows fast selection has no skips**

Run:

```powershell
python -m pytest -q -m "not io and not liburing and not llama"
```

Expected: all selected tests pass and pytest reports no skipped tests.

- [ ] **Step 6: Commit test ownership and backend enforcement**

Run:

```powershell
git add io/Makefile tests/test_phase2_io.py tests/test_runtime_command_consumer.py tests/test_llama_cpp_integration.py tests/test_llama_cpp_integration_map.py tests/test_llama_cpp_patch.py
git commit -m 'test: separate platform verification profiles'
```

### Task 6: Define JSON Check Results And JUnit Accounting

**Files:**
- Create: `tools/check_result.py`
- Create: `tests/test_check_result.py`

- [ ] **Step 1: Write failing serialization and JUnit tests**

Create `tests/test_check_result.py`:

```python
import json
from pathlib import Path

from tools.check_result import CheckResult, CommandResult, TestCounts, parse_junit


def test_parse_junit_sums_nested_suites(tmp_path: Path) -> None:
    path = tmp_path / "junit.xml"
    path.write_text(
        '<testsuites name="pytest tests">'
        '<testsuite tests="2" failures="1" errors="0" skipped="0" time="0.3" />'
        '<testsuite tests="1" failures="0" errors="0" skipped="1" time="0.2" />'
        '</testsuites>',
        encoding="utf-8",
    )

    assert parse_junit(path) == TestCounts(tests=3, failures=1, errors=0, skipped=1, seconds=0.5)


def test_check_result_writes_bounded_json(tmp_path: Path) -> None:
    result = CheckResult(
        profile="fast",
        status="passed",
        platform="Windows",
        commands=(CommandResult("pytest", ("python", "-m", "pytest"), 0, "ok", "", 12),),
        tests=TestCounts(3, 0, 0, 0, 0.01),
        message="",
    )

    output = tmp_path / "check_result.json"
    result.save(output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema"] == "solidattention.check_result.v1"
    assert payload["profile"] == "fast"
    assert payload["tests"]["skipped"] == 0
```

- [ ] **Step 2: Run the new tests red**

Run:

```powershell
python -m pytest -q tests/test_check_result.py
```

Expected: collection fails because `tools.check_result` does not exist.

- [ ] **Step 3: Implement the result contract**

Create `tools/check_result.py` with these public types and behavior:

```python
"""Serializable results shared by local and CI verification profiles."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

CheckStatus = Literal["passed", "failed", "unavailable"]


@dataclass(frozen=True)
class TestCounts:
    tests: int
    failures: int
    errors: int
    skipped: int
    seconds: float


@dataclass(frozen=True)
class CommandResult:
    name: str
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_ms: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CheckResult:
    profile: str
    status: CheckStatus
    platform: str
    commands: tuple[CommandResult, ...]
    tests: TestCounts | None
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "solidattention.check_result.v1",
            "profile": self.profile,
            "status": self.status,
            "platform": self.platform,
            "commands": [command.to_dict() for command in self.commands],
            "tests": asdict(self.tests) if self.tests is not None else None,
            "message": self.message,
        }

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_junit(path: str | Path) -> TestCounts:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else [
        node
        for node in root.iter("testsuite")
        if not any(child.tag == "testsuite" for child in node)
    ]
    return TestCounts(
        tests=sum(int(node.attrib.get("tests", 0)) for node in suites),
        failures=sum(int(node.attrib.get("failures", 0)) for node in suites),
        errors=sum(int(node.attrib.get("errors", 0)) for node in suites),
        skipped=sum(int(node.attrib.get("skipped", 0)) for node in suites),
        seconds=sum(float(node.attrib.get("time", 0.0)) for node in suites),
    )


def bounded_output(text: str, limit: int = 8000) -> str:
    return text if len(text) <= limit else text[-limit:]
```

- [ ] **Step 4: Run the result tests green**

Run:

```powershell
python -m pytest -q tests/test_check_result.py
```

Expected: `2 passed`.

- [ ] **Step 5: Commit the check-result contract**

Run:

```powershell
git add tools/check_result.py tests/test_check_result.py
git commit -m 'feat: add machine-readable check results'
```

### Task 7: Build The Pinned llama.cpp Verification Helper

**Files:**
- Create: `tools/llama_check.py`
- Create: `tests/test_llama_check.py`
- Modify: `integrations/llama_cpp/patches/solidattention_kv_trace.patch`
- Modify: `tests/test_llama_cpp_patch.py`

- [ ] **Step 1: Write failing revision and command-plan tests**

Create `tests/test_llama_check.py`:

```python
from pathlib import Path

import pytest

from tools.llama_check import PINNED_LLAMA_COMMIT, llama_build_commands, require_pinned_revision


def test_require_pinned_revision_accepts_expected_hash() -> None:
    require_pinned_revision(PINNED_LLAMA_COMMIT + "\n")


def test_require_pinned_revision_rejects_wrong_hash() -> None:
    with pytest.raises(RuntimeError, match="wrong llama.cpp revision"):
        require_pinned_revision("0" * 40)


def test_llama_build_commands_include_patch_build_and_smoke(tmp_path: Path) -> None:
    commands = llama_build_commands(
        checkout=Path("external/llama.cpp"),
        patch=Path("integrations/llama_cpp/patches/solidattention_kv_trace.patch"),
        worktree=tmp_path / "worktree",
        build=tmp_path / "build",
        smoke_output=tmp_path / "smoke",
    )

    flattened = [" ".join(command) for command in commands]
    assert any("worktree add --detach" in command for command in flattened)
    assert any("apply" in command for command in flattened)
    assert any("cmake" in command and "LLAMA_BUILD_TESTS=ON" in command for command in flattened)
    assert any("--target test-llama-archs" in command for command in flattened)
    assert any("integrations.llama_cpp.evict_smoke" in command for command in flattened)
    assert any("--trace" in command and "runtime_kv_trace.jsonl" in command for command in flattened)
```

Add to `tests/test_llama_cpp_patch.py`:

```python
def test_patch_has_no_host_specific_liburing_default() -> None:
    patch = Path("integrations/llama_cpp/patches/solidattention_kv_trace.patch").read_text(
        encoding="utf-8"
    )

    assert 'set(LLAMA_SOLIDATTENTION_LIBURING_PREFIX "" CACHE PATH' in patch
    assert "/data/disk2/ljc" not in patch
```

- [ ] **Step 2: Run the helper tests red**

Run:

```powershell
python -m pytest -q tests/test_llama_check.py tests/test_llama_cpp_patch.py::test_patch_has_no_host_specific_liburing_default
```

Expected: collection fails because `tools.llama_check` does not exist, and the
patch contract fails because its liburing cache default is host-specific.

- [ ] **Step 3: Implement the pinned build plan and runner**

Create `tools/llama_check.py` with:

```python
"""Build and run the pinned patched llama.cpp smoke target on Linux."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from integrations.llama_cpp.probe import probe_llama_cpp

PINNED_LLAMA_COMMIT = "bbebeec4a87355896e3faac0c2baca8130c91b6a"


def require_pinned_revision(stdout: str) -> None:
    actual = stdout.strip()
    if actual != PINNED_LLAMA_COMMIT:
        raise RuntimeError(f"wrong llama.cpp revision: expected {PINNED_LLAMA_COMMIT}, got {actual}")


def llama_build_commands(
    *, checkout: Path, patch: Path, worktree: Path, build: Path, smoke_output: Path
) -> list[list[str]]:
    executable = build / "bin" / "test-llama-archs"
    trace = (smoke_output / "runtime_kv_trace.jsonl").resolve()
    return [
        ["git", "-C", str(checkout), "worktree", "add", "--detach", str(worktree), PINNED_LLAMA_COMMIT],
        ["git", "-C", str(worktree), "apply", str(patch.resolve())],
        ["cmake", "-S", str(worktree), "-B", str(build), "-DLLAMA_BUILD_TESTS=ON", "-DGGML_NATIVE=OFF", "-DCMAKE_BUILD_TYPE=Release"],
        ["cmake", "--build", str(build), "--target", "test-llama-archs", "--parallel", "2"],
        [
            sys.executable,
            "-m",
            "integrations.llama_cpp.evict_smoke",
            "--executable",
            str(executable),
            "--trace",
            str(trace),
            "--summary",
            str((smoke_output / "runtime_kv_trace_summary.json").resolve()),
            "--harness-trace",
            str((smoke_output / "harness_trace.jsonl").resolve()),
            "--stdout",
            str((smoke_output / "stdout.log").resolve()),
            "--stderr",
            str((smoke_output / "stderr.log").resolve()),
        ],
    ]


def run_llama_check(
    checkout: str | Path = "external/llama.cpp",
    patch: str | Path = "integrations/llama_cpp/patches/solidattention_kv_trace.patch",
    output_dir: str | Path = "outputs/checks/llama/llama-smoke",
) -> None:
    checkout_path = Path(checkout)
    patch_path = Path(patch)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    probe = probe_llama_cpp(checkout_path)
    if not probe.ready:
        raise RuntimeError(f"llama.cpp checkout is {probe.state}: {probe.to_dict()}")
    revision = subprocess.run(
        ["git", "-C", str(checkout_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    require_pinned_revision(revision.stdout)
    if shutil.which("cmake") is None:
        raise RuntimeError("cmake is required for the llama profile")
    with tempfile.TemporaryDirectory(prefix="solidattention-llama-") as temp:
        temp_path = Path(temp)
        worktree = temp_path / "worktree"
        build = temp_path / "build"
        try:
            for command in llama_build_commands(
                checkout=checkout_path,
                patch=patch_path,
                worktree=worktree,
                build=build,
                smoke_output=output_path,
            ):
                subprocess.run(command, check=True)
        finally:
            if worktree.exists():
                subprocess.run(
                    ["git", "-C", str(checkout_path), "worktree", "remove", "--force", str(worktree)],
                    check=False,
                )
```

In the patch's added CMake line, replace the developer-machine default:

```diff
-+set(LLAMA_SOLIDATTENTION_LIBURING_PREFIX "/data/disk2/ljc/solid_attention_deps/liburing-install" CACHE PATH "Optional liburing prefix for SolidAttention executor stub")
++set(LLAMA_SOLIDATTENTION_LIBURING_PREFIX "" CACHE PATH "Optional liburing prefix for SolidAttention executor stub")
```

The final command is deliberately the existing `evict_smoke` harness, not a
plain `test-llama-archs` invocation. It sets
`LLAMA_SOLIDATTENTION_KV_TRACE_PATH` and
`LLAMA_SOLIDATTENTION_KV_TRACE_EVICT_SMOKE=1`, then fails unless the patched
runtime emits a parseable trace containing a real `kv_evict` event.

Add:

```python
def main() -> int:
    run_llama_check()
    print(f"llama.cpp smoke passed at {PINNED_LLAMA_COMMIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Let exceptions propagate so the process is nonzero and `tools.check` captures
the diagnostic.

- [ ] **Step 4: Run the helper tests green**

Run:

```powershell
python -m pytest -q tests/test_llama_check.py tests/test_llama_cpp_patch.py::test_patch_has_no_host_specific_liburing_default
```

Expected: `4 passed`. Do not run the Linux-only real build on Windows.

- [ ] **Step 5: Commit the llama verification helper**

Run:

```powershell
git add tools/llama_check.py tests/test_llama_check.py tests/test_llama_cpp_patch.py integrations/llama_cpp/patches/solidattention_kv_trace.patch
git commit -m 'feat: add pinned llama integration check'
```

### Task 8: Implement The Unified Check CLI

**Files:**
- Create: `tools/check.py`
- Create: `tests/test_check.py`
- Modify: `tools/__init__.py`

- [ ] **Step 1: Write failing profile-availability and command tests**

Create `tests/test_check.py`:

```python
import json
import subprocess
from pathlib import Path

from tools.check import build_profile_commands, profile_availability, run_profile


def test_fast_profile_is_available_on_windows() -> None:
    assert profile_availability("fast", "Windows") == (True, "")


def test_linux_profiles_are_unavailable_on_windows() -> None:
    available, message = profile_availability("liburing", "Windows")
    assert not available
    assert "requires Linux" in message


def test_fast_profile_excludes_platform_tests(tmp_path: Path) -> None:
    commands = build_profile_commands("fast", tmp_path)
    pytest_command = next(command for name, command, _env in commands if name == "pytest")
    joined = " ".join(pytest_command)
    assert "not io and not liburing and not llama" in joined
    assert "--junitxml" in joined
    assert [name for name, _command, _env in commands] == ["pytest", "smoke", "ablation"]


def test_liburing_profile_forces_backend(tmp_path: Path) -> None:
    commands = build_profile_commands("liburing", tmp_path)
    assert any("BACKEND_MODE=liburing" in command for _name, args, _env in commands for command in args)
    pytest_env = next(env for name, _args, env in commands if name == "pytest")
    assert pytest_env["SOLIDATTENTION_EXPECTED_IO_BACKEND"] == "liburing"


def test_profile_setup_removes_stale_results(tmp_path: Path) -> None:
    stale = tmp_path / "outputs" / "checks" / "fast" / "smoke" / "run_manifest.json"
    stale.parent.mkdir(parents=True)
    stale.write_text('{"status":"passed"}\n', encoding="utf-8")

    build_profile_commands("fast", tmp_path)

    assert not stale.exists()


def test_nonzero_command_writes_failed_result_and_returns_one(tmp_path: Path) -> None:
    def runner(argv, **_kwargs):
        return subprocess.CompletedProcess(argv, 7, "", "boom")

    exit_code = run_profile(
        "fast",
        tmp_path,
        system="Windows",
        runner=runner,
        repository_checker=lambda _root: [],
    )

    payload = json.loads(
        (tmp_path / "outputs/checks/fast/check_result.json").read_text(encoding="utf-8")
    )
    assert exit_code == 1
    assert payload["status"] == "failed"
    assert payload["commands"][0]["returncode"] == 7


def test_junit_skip_fails_required_profile(tmp_path: Path) -> None:
    def runner(argv, **_kwargs):
        if "--junitxml" in argv:
            junit = Path(argv[argv.index("--junitxml") + 1])
            junit.write_text(
                '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="1" time="0.1" /></testsuites>',
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(argv, 0, "", "")

    exit_code = run_profile(
        "fast",
        tmp_path,
        system="Windows",
        runner=runner,
        repository_checker=lambda _root: [],
    )
    payload = json.loads(
        (tmp_path / "outputs/checks/fast/check_result.json").read_text(encoding="utf-8")
    )

    assert exit_code == 1
    assert payload["status"] == "failed"
    assert payload["tests"]["skipped"] == 1


def test_backend_mismatch_fails_io_profile(tmp_path: Path) -> None:
    def runner(argv, **_kwargs):
        stdout = "liburing\n" if "backend" in argv else ""
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    exit_code = run_profile(
        "io",
        tmp_path,
        system="Linux",
        runner=runner,
        which=lambda _name: "/usr/bin/tool",
    )
    payload = json.loads(
        (tmp_path / "outputs/checks/io/check_result.json").read_text(encoding="utf-8")
    )

    assert exit_code == 1
    assert payload["status"] == "failed"
    assert "expected posix_fallback" in payload["message"]
```

- [ ] **Step 2: Run the check tests red**

Run:

```powershell
python -m pytest -q tests/test_check.py
```

Expected: collection fails because `tools.check` does not exist.

- [ ] **Step 3: Implement profile definitions**

Create `tools/check.py`. Define:

```python
PROFILE_NAMES = ("fast", "io", "liburing", "llama")


def profile_availability(profile: str, system: str) -> tuple[bool, str]:
    if profile == "fast":
        return True, ""
    if system != "Linux":
        return False, f"profile {profile} requires Linux; run it in GitHub Actions"
    return True, ""
```

Expose a testable execution boundary:

```python
def run_profile(
    profile: str,
    root: str | Path = Path.cwd(),
    *,
    system: str | None = None,
    runner=subprocess.run,
    which=shutil.which,
    repository_checker=check_repository,
) -> int:
```

`main()` delegates to `run_profile`. Production uses the default subprocess and
tool lookup functions; tests inject deterministic fakes without changing
environment detection or touching native tools.

`build_profile_commands(profile, root)` returns `(name, argv, env)` tuples with
these exact command contracts. Its first lines are:

```python
output = root / "outputs" / "checks" / profile
if output.exists():
    shutil.rmtree(output)
output.mkdir(parents=True, exist_ok=True)
junit = output / "junit.xml"
```

Import `shutil`. Clearing this generated, profile-owned directory at the start
of every run is part of the contract: a failed command must never be followed
by validation of a stale JUnit file or run manifest.

The profile lists are:

```python
fast = [
    ("pytest", [sys.executable, "-m", "pytest", "-q", "-m", "not io and not liburing and not llama", "--junitxml", str(junit)], {}),
    ("smoke", [sys.executable, "-m", "harness.smoke", "--config", "configs/smoke_base.yaml", "--output-dir", str(output / "smoke")], {}),
    ("ablation", [sys.executable, "-m", "harness.ablation", "--config", "configs/smoke_base.yaml", "--output-dir", str(output / "ablation")], {}),
]
io = [
    ("make", ["make", "-C", "io", "clean", "all", "BACKEND_MODE=posix"], {}),
    ("backend", ["make", "-s", "-C", "io", "backend", "BACKEND_MODE=posix"], {}),
    ("pytest", [sys.executable, "-m", "pytest", "-q", "-m", "io", "--junitxml", str(junit)], {"SOLIDATTENTION_EXPECTED_IO_BACKEND": "posix_fallback"}),
]
liburing = [
    ("make", ["make", "-C", "io", "clean", "all", "BACKEND_MODE=liburing"], {}),
    ("backend", ["make", "-s", "-C", "io", "backend", "BACKEND_MODE=liburing"], {}),
    ("pytest", [sys.executable, "-m", "pytest", "-q", "-m", "liburing", "--junitxml", str(junit)], {"SOLIDATTENTION_EXPECTED_IO_BACKEND": "liburing"}),
]
llama = [
    ("llama-smoke", [sys.executable, "-m", "tools.llama_check"], {}),
    ("pytest", [sys.executable, "-m", "pytest", "-q", "-m", "llama", "--junitxml", str(junit)], {}),
]
```

For `fast`, call `check_repository(root)` before subprocess execution and turn
any violation into a failed result. For `io` and `liburing`, require `make` and
`cc`; for `llama`, require `git` and `cmake`. Wrong-platform invocation returns
structured `unavailable`; a missing prerequisite on the owning Linux platform
is a failed profile with exit code `1`.

- [ ] **Step 4: Implement execution, backend checks, JSON, and exit codes**

For each command:

1. Merge the command-specific environment into `os.environ`.
2. Run with `cwd=root`, captured text output, and `check=False`.
3. Record monotonic duration and bounded stdout/stderr in `CommandResult`.
4. Stop after the first nonzero command.
5. Require backend stdout to equal `posix_fallback` for `io` and `liburing` for
   `liburing`.
6. Parse JUnit after pytest. Fail every available required profile when
   `skipped != 0`; a skip is not success in `fast`, `io`, `liburing`, or
   `llama`.
7. Require smoke and ablation subprocesses to exit zero. Task 10 adds their
   manifest-schema gate after manifest support exists.

Always save `outputs/checks/<profile>/check_result.json`. Return exit code `0`
for `passed`, `1` for `failed`, and `2` for `unavailable`. Print one line:

```text
profile=<name> status=<status> tests=<n> failures=<n> errors=<n> skipped=<n>
```

The CLI parser accepts exactly one positional `profile` from `PROFILE_NAMES`.

- [ ] **Step 5: Run unit and real Windows fast checks**

Run:

```powershell
python -m pytest -q tests/test_check.py tests/test_check_result.py
python -m tools.check fast
```

Expected: focused tests pass. The real command exits `0`, reports no failures
or skips, and writes `outputs/checks/fast/check_result.json` with
`status=passed`.

- [ ] **Step 6: Verify Windows returns structured unavailable for Linux profiles**

Run:

```powershell
python -m tools.check liburing
$LASTEXITCODE
Get-Content -Raw outputs/checks/liburing/check_result.json
```

Expected: exit code `2`; JSON status is `unavailable` and the message says to
run the profile in GitHub Actions. No Python traceback appears.

- [ ] **Step 7: Commit the unified verification interface**

Run:

```powershell
git add tools/check.py tools/__init__.py tests/test_check.py
git commit -m 'feat: add explicit verification profiles'
```

### Task 9: Implement `solidattention.run_manifest.v1`

**Files:**
- Create: `harness/run_manifest.py`
- Create: `tests/test_run_manifest.py`

- [ ] **Step 1: Write deterministic success and failure tests**

Create `tests/test_run_manifest.py`:

```python
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from harness.run_manifest import RunContext


def fixed_clock():
    values = iter(
        [
            datetime(2026, 7, 11, 8, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 7, 11, 8, 0, 1, tzinfo=timezone.utc),
        ]
    )
    return lambda: next(values)


def test_run_context_writes_success_manifest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SECRET_TOKEN", "must-not-appear")
    config = tmp_path / "config.yaml"
    source = tmp_path / "trace.jsonl"
    config.write_text("blocks: 4\n", encoding="utf-8")
    source.write_text('{"op":"kv_add"}\n', encoding="utf-8")

    with RunContext(
        tmp_path / "run",
        command=("python", "-m", "harness.smoke"),
        profile="smoke",
        config_path=config,
        config_snapshot={"blocks": 4},
        inputs=(source,),
        io_backend="mock",
        run_id="run-fixed",
        clock=fixed_clock(),
    ) as run:
        metrics = tmp_path / "run" / "metrics.json"
        metrics.write_text("{}\n", encoding="utf-8")
        run.add_output("metrics", metrics)

    payload = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    serialized = json.dumps(payload)
    assert payload["schema"] == "solidattention.run_manifest.v1"
    assert payload["run_id"] == "run-fixed"
    assert payload["profile"] == "smoke"
    assert payload["io_backend"] == "mock"
    assert payload["status"] == "passed"
    assert payload["duration_ms"] == 1000
    assert payload["started_at"] == "2026-07-11T08:00:00+00:00"
    assert payload["ended_at"] == "2026-07-11T08:00:01+00:00"
    assert payload["command"] == ["python", "-m", "harness.smoke"]
    assert set(payload["runtime"]) == {"python", "os", "release", "machine"}
    assert set(payload["git"]) == {"sha", "dirty"}
    assert payload["config"]["snapshot"] == {"blocks": 4}
    assert payload["config"]["sha256"]
    assert payload["inputs"][0]["sha256"]
    assert payload["outputs"][0]["kind"] == "metrics"
    assert payload["outputs"][0]["path"] == "metrics.json"
    assert payload["outputs"][0]["sha256"]
    assert "must-not-appear" not in serialized


def test_run_context_records_failure_and_reraises(tmp_path: Path) -> None:
    run = RunContext(
        tmp_path / "failed",
        command=("python", "-m", "harness.smoke"),
        profile="smoke",
        run_id="run-failed",
        clock=fixed_clock(),
    )

    with pytest.raises(ValueError, match="bad config"):
        with run:
            raise ValueError("bad config")

    payload = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["failure"] == {"type": "ValueError", "message": "bad config"}


def test_run_context_records_non_exception_verification_failure(tmp_path: Path) -> None:
    with RunContext(
        tmp_path / "verification-failed",
        command=("python", "-m", "harness.runtime_residency_executor"),
        profile="runtime-residency-executor",
        run_id="run-verification-failed",
        clock=fixed_clock(),
    ) as run:
        run.mark_failed("VerificationError", "runtime residency verification errors: 2")

    payload = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["failure"]["type"] == "VerificationError"


def test_run_context_records_missing_input_failure_from_enter(tmp_path: Path) -> None:
    run = RunContext(
        tmp_path / "missing-input",
        command=("python", "-m", "harness.runtime_replay"),
        profile="runtime-replay",
        inputs=(tmp_path / "missing.jsonl",),
        run_id="run-missing-input",
        clock=fixed_clock(),
    )

    with pytest.raises(FileNotFoundError):
        with run:
            pass

    payload = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["failure"]["type"] == "FileNotFoundError"


def test_run_context_generates_timestamped_run_id(tmp_path: Path) -> None:
    with RunContext(
        tmp_path / "generated-id",
        command=("python", "-m", "harness.smoke"),
        profile="smoke",
        clock=fixed_clock(),
    ) as run:
        pass

    payload = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert re.fullmatch(r"20260711T080000Z-[0-9a-f]{8}", payload["run_id"])
```

- [ ] **Step 2: Run manifest tests red**

Run:

```powershell
python -m pytest -q tests/test_run_manifest.py
```

Expected: collection fails because `harness.run_manifest` does not exist.

- [ ] **Step 3: Implement hashing and environment metadata helpers**

Create `harness/run_manifest.py` with imports for `hashlib`, `json`, `platform`,
`subprocess`, `sys`, `uuid`, `datetime`, and `Path`. Implement:

```python
def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata(cwd: Path) -> dict[str, object]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=cwd, check=True, capture_output=True, text=True
            ).stdout.strip()
        )
        return {"sha": sha, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "dirty": None}
```

Input/config file records use
`{"path": str(path), "sha256": sha256_file(path)}`; output records use the
run-relative path rule in Step 4. Runtime metadata contains only Python
version, OS, release, and machine architecture; do not serialize environment
variables.

- [ ] **Step 4: Implement `RunContext`**

Use this public constructor:

```python
class RunContext:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        command: tuple[str, ...],
        profile: str,
        config_path: str | Path | None = None,
        config_snapshot: dict[str, object] | None = None,
        inputs: tuple[str | Path, ...] = (),
        io_backend: str = "mock",
        run_id: str | None = None,
        clock=None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.manifest_path = self.output_dir / "run_manifest.json"
        self.command = tuple(command)
        self.profile = profile
        self.config_path = Path(config_path) if config_path is not None else None
        self.config_snapshot = config_snapshot
        self.inputs = tuple(Path(path) for path in inputs)
        self.io_backend = io_backend
        self.run_id = run_id
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.outputs: list[dict[str, str]] = []
        self.started_at: datetime | None = None
        self.git: dict[str, object] = {"sha": None, "dirty": None}
        self.runtime: dict[str, str] = {}
        self.forced_failure: dict[str, str] | None = None
```

Implement these semantics exactly:

- `manifest_path` is `<output_dir>/run_manifest.json`.
- `__enter__` creates the output directory, captures start time, Git metadata,
  runtime metadata, config hash/snapshot, and input hashes, then returns `self`.
  Git metadata is queried from `Path.cwd()`, not from the output directory, so
  an output path outside the checkout still records the current source SHA.
- `__enter__` wraps metadata and input/config hashing in `try/except Exception`.
  If setup fails after the output directory and start time exist, it writes the
  failed manifest through the same finalization helper, then re-raises; it does
  not rely on `__exit__`, which Python will not call after a failed enter.
- Default clock is `datetime.now(timezone.utc)`.
- Default run ID is `<UTC YYYYmmddTHHMMSSZ>-<8 uuid hex chars>`.
- `add_output(kind, path)` appends a file record; call it only after the file
  exists. Store output paths relative to `output_dir` with POSIX separators
  when possible, otherwise store the supplied path string.
- `set_config_snapshot(value)` stores a parsed JSON-compatible configuration
  after config parsing succeeds.
- The constructor and `set_io_backend(value)` accept only `mock`, `fake`,
  `posix_fallback`, or `liburing`; the setter replaces the initial backend
  after an I/O probe reports its actual value.
- `mark_failed(failure_type, message)` records a non-exception verification
  failure that `__exit__` must serialize with `status=failed`.
- `__exit__` captures end time, sets `passed` or `failed`, writes the manifest,
  and returns `False` so exceptions are re-raised.
- Failure payload contains only exception class name and `str(exception)`.
- The top-level payload keys are `schema`, `run_id`, `profile`, `command`,
  `started_at`, `ended_at`, `duration_ms`, `status`, `failure`, `git`,
  `runtime`, `io_backend`, `config`, `inputs`, and `outputs`. `config` contains
  `path`, `sha256`, and `snapshot` when present.
- Output JSON is indented, key-sorted, UTF-8, and newline terminated.

- [ ] **Step 5: Run manifest tests green**

Run:

```powershell
python -m pytest -q tests/test_run_manifest.py
```

Expected: `5 passed`.

- [ ] **Step 6: Commit the run-manifest core**

Run:

```powershell
git add harness/run_manifest.py tests/test_run_manifest.py
git commit -m 'feat: add reproducible run manifests'
```

### Task 10: Add Manifests To Smoke And Ablation Runs

**Files:**
- Modify: `harness/smoke.py`
- Modify: `harness/ablation.py`
- Modify: `tools/check.py`
- Modify: `tests/test_smoke_harness.py`
- Modify: `tests/test_ablation_harness.py`
- Modify: `tests/test_check.py`

- [ ] **Step 1: Extend smoke and ablation tests red**

In `tests/test_smoke_harness.py`, after running smoke, assert:

```python
manifest_path = tmp_path / "run_manifest.json"
assert manifest_path.exists()
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
assert manifest["schema"] == "solidattention.run_manifest.v1"
assert manifest["profile"] == "smoke"
assert manifest["status"] == "passed"
assert manifest["io_backend"] == "mock"
assert {item["kind"] for item in manifest["outputs"]} == {"trace", "metrics", "summary"}
```

In `tests/test_ablation_harness.py`, assert its manifest has profile
`ablation`, status `passed`, and output kinds containing every method trace and
metrics plus the summary.

Add to `tests/test_check.py`:

```python
import json

import pytest

from tools.check import require_passed_manifest


def test_require_passed_manifest_accepts_expected_schema(tmp_path: Path) -> None:
    path = tmp_path / "run_manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema": "solidattention.run_manifest.v1",
                "profile": "smoke",
                "status": "passed",
                "outputs": [
                    {"kind": "trace", "path": "trace.jsonl", "sha256": "a" * 64},
                    {"kind": "metrics", "path": "metrics.json", "sha256": "b" * 64},
                    {"kind": "summary", "path": "summary.json", "sha256": "c" * 64},
                ],
            }
        ),
        encoding="utf-8",
    )
    require_passed_manifest(
        path,
        expected_profile="smoke",
        required_output_kinds={"trace", "metrics", "summary"},
    )


def test_require_passed_manifest_rejects_failed_run(tmp_path: Path) -> None:
    path = tmp_path / "run_manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema": "solidattention.run_manifest.v1",
                "profile": "smoke",
                "status": "failed",
                "outputs": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="manifest is not passed"):
        require_passed_manifest(
            path,
            expected_profile="smoke",
            required_output_kinds={"trace", "metrics", "summary"},
        )


def test_require_passed_manifest_rejects_wrong_profile_or_missing_hash(tmp_path: Path) -> None:
    path = tmp_path / "run_manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema": "solidattention.run_manifest.v1",
                "profile": "ablation",
                "status": "passed",
                "outputs": [{"kind": "summary", "path": "summary.json", "sha256": ""}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="profile mismatch"):
        require_passed_manifest(
            path,
            expected_profile="smoke",
            required_output_kinds={"summary"},
        )
    with pytest.raises(RuntimeError, match="missing sha256"):
        require_passed_manifest(
            path,
            expected_profile="ablation",
            required_output_kinds={"summary"},
        )
```

Run:

```powershell
python -m pytest -q tests/test_smoke_harness.py tests/test_ablation_harness.py
```

Expected: failures report missing `run_manifest.json`.

- [ ] **Step 2: Wrap `run_smoke` in `RunContext`**

Add keyword-only arguments without breaking existing callers:

```python
def run_smoke(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    run_id: str | None = None,
    clock=None,
) -> dict[str, Any]:
```

Create the context before parsing so invalid configuration still produces a
failed manifest. Parse and attach the snapshot inside it:

```python
with RunContext(
    output,
    command=("python", "-m", "harness.smoke", "--config", str(config_path), "--output-dir", str(output)),
    profile="smoke",
    config_path=config_path,
    inputs=(config_path,),
    io_backend="mock",
    run_id=run_id,
    clock=clock,
) as run:
    config = load_smoke_config(config_path)
    run.set_config_snapshot(config)
    run.add_output("trace", trace_path)
    run.add_output("metrics", metrics_path)
    run.add_output("summary", summary_path)
```

Move the existing statements beginning with `trace = TraceRecorder()` and
ending with the `smoke_summary.json` write into the context block without
changing their ordering or payload construction. Place the three
`run.add_output` calls immediately after all three files have been written.

Add `"manifest": str(run.manifest_path)` to `summary["outputs"]`. Keep existing
trace, metrics, summary names and payloads unchanged.

- [ ] **Step 3: Wrap `run_ablation` in `RunContext`**

Add the same `run_id` and `clock` keyword-only arguments. Enter a `RunContext`
with profile `ablation`, backend `mock`, and the config file as an input before
calling `load_smoke_config`; then call `run.set_config_snapshot(config)`. Move
the existing method loop and summary write into the context. Register each
method trace/metrics output and `ablation_summary.json`, then add
`manifest_path` to the returned summary.

In `tools/check.py`, implement
`require_passed_manifest(path, *, expected_profile, required_output_kinds)` by
loading JSON and requiring schema `solidattention.run_manifest.v1`, status
`passed`, the exact profile, every required output kind, and a nonempty SHA-256
for every output record. After the fast profile's smoke and ablation commands
succeed, validate them as follows:

```python
require_passed_manifest(
    output / "smoke" / "run_manifest.json",
    expected_profile="smoke",
    required_output_kinds={"trace", "metrics", "summary"},
)
require_passed_manifest(
    output / "ablation" / "run_manifest.json",
    expected_profile="ablation",
    required_output_kinds={"trace", "metrics", "summary"},
)
```

- [ ] **Step 4: Run focused manifest integrations green**

Run:

```powershell
python -m pytest -q tests/test_smoke_harness.py tests/test_ablation_harness.py tests/test_run_manifest.py tests/test_check.py
python -m harness.smoke --config configs/smoke_base.yaml --output-dir outputs/verification/smoke
python -m harness.ablation --config configs/smoke_base.yaml --output-dir outputs/verification/ablation
```

Expected: tests pass; both CLI output directories contain a passed manifest.

- [ ] **Step 5: Commit smoke and ablation manifests**

Run:

```powershell
git add harness/smoke.py harness/ablation.py tools/check.py tests/test_smoke_harness.py tests/test_ablation_harness.py tests/test_check.py
git commit -m 'feat: record simulation run manifests'
```

### Task 11: Add Manifests To Replay And Executors

**Files:**
- Modify: `harness/runtime_replay.py`
- Modify: `harness/movement_executor.py`
- Modify: `harness/runtime_residency_executor.py`
- Modify: `tests/test_runtime_replay.py`
- Modify: `tests/test_movement_executor.py`
- Modify: `tests/test_runtime_residency_executor.py`

- [ ] **Step 1: Add failing manifest assertions to each entry point**

In each focused test file, extend one successful tmp-path test to load
`<output_dir>/run_manifest.json` and assert:

```python
assert manifest["schema"] == "solidattention.run_manifest.v1"
assert manifest["status"] == "passed"
assert manifest["profile"] == expected_profile
assert manifest["inputs"][0]["sha256"]
```

Use profiles `runtime-replay`, `movement-executor`, and
`runtime-residency-executor`. For the calibrated replay and I/O movement test,
also assert `io_backend == "fake"` from the fake probe result.

Extend
`test_runtime_residency_executor_strict_mode_reports_missing_source` to assert
that a non-exception verification failure is represented in the manifest:

```python
manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
assert manifest["status"] == "failed"
assert manifest["failure"] == {
    "type": "VerificationError",
    "message": "runtime residency verification errors: 2",
}
```

Run:

```powershell
python -m pytest -q tests/test_runtime_replay.py tests/test_movement_executor.py tests/test_runtime_residency_executor.py
```

Expected: manifest assertions fail.

- [ ] **Step 2: Wrap runtime replay**

Add optional keyword-only `run_id=None` and `clock=None` to
`run_runtime_replay`. Create `RunContext` with the runtime trace and the
optional probe *executable* as inputs. Do not register `io_probe_path` as an
input: that file is created or overwritten during calibration. Start with
backend `mock`; after calibration call:

```python
run.set_io_backend(io_probe_result.backend)
```

Register replay metrics, scheduler trace, summary, movement plan, and the
generated calibration probe file as outputs when present.

- [ ] **Step 3: Wrap movement execution**

Add optional keyword-only `run_id=None` and `clock=None` to
`execute_movement_plan`. Register the plan and optional probe executable as
inputs, but register generated probe data files only as outputs. Start with
backend `mock`; if probe results exist, require that all report one backend and call
`run.set_io_backend(backend)`. Register metrics, trace, summary, and generated
probe files.

- [ ] **Step 4: Wrap runtime residency execution**

Add optional keyword-only `run_id=None` and `clock=None` to
`execute_runtime_residency_plan`. Register the movement plan and optional
backing file as inputs, use backend `mock`, and register metrics, trace, and
summary outputs. When `result.verification_error_count > 0`, call:

```python
run.mark_failed(
    "VerificationError",
    f"runtime residency verification errors: {result.verification_error_count}",
)
```

Preserve the existing nonzero CLI exit when verification errors occur. The
manifest and process exit now agree that such a run failed.

- [ ] **Step 5: Run all manifest integrations green**

Run:

```powershell
python -m pytest -q tests/test_runtime_replay.py tests/test_movement_executor.py tests/test_runtime_residency_executor.py tests/test_run_manifest.py
```

Expected: all selected tests pass, including fake backend assertions.

- [ ] **Step 6: Commit runtime manifests**

Run:

```powershell
git add harness/runtime_replay.py harness/movement_executor.py harness/runtime_residency_executor.py tests/test_runtime_replay.py tests/test_movement_executor.py tests/test_runtime_residency_executor.py
git commit -m 'feat: record runtime execution manifests'
```

### Task 12: Add Windows And Linux GitHub Actions Gates

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `tests/test_ci_contract.py`

- [ ] **Step 1: Write a failing workflow contract test**

Create `tests/test_ci_contract.py`:

```python
from pathlib import Path


def test_ci_exposes_all_required_profiles() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "windows-latest" in workflow
    assert "ubuntu-latest" in workflow
    assert "branches: [main, harness-candidate]" in workflow
    assert 'python: "3.11"' in workflow
    assert 'python: "3.13"' in workflow
    assert "python -m tools.check fast" in workflow
    assert "python -m tools.check io" in workflow
    assert "python -m tools.check liburing" in workflow
    assert "python -m tools.check llama" in workflow
    assert "submodules: recursive" in workflow
    assert "liburing-dev" in workflow
    assert "if-no-files-found: ignore" in workflow
```

Run:

```powershell
python -m pytest -q tests/test_ci_contract.py
```

Expected: FAIL because the workflow does not exist.

- [ ] **Step 2: Create the CI workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: ci

on:
  push:
    branches: [main, harness-candidate]
  pull_request:

permissions:
  contents: read

jobs:
  fast:
    name: fast (${{ matrix.label }})
    runs-on: ${{ matrix.os }}
    timeout-minutes: 15
    strategy:
      fail-fast: false
      matrix:
        include:
          - os: windows-latest
            python: "3.11"
            label: Windows / Python 3.11
          - os: windows-latest
            python: "3.13"
            label: Windows / Python 3.13
          - os: ubuntu-latest
            python: "3.11"
            label: Linux / Python 3.11
          - os: ubuntu-latest
            python: "3.13"
            label: Linux / Python 3.13
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v6
        with:
          python-version: ${{ matrix.python }}
          cache: pip
      - run: python -m pip install --disable-pip-version-check -e ".[dev]"
      - run: python -m tools.check fast
      - if: always()
        uses: actions/upload-artifact@v7
        with:
          name: check-fast-${{ matrix.os }}-py${{ matrix.python }}
          path: outputs/checks/fast
          if-no-files-found: ignore

  io-posix:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v6
        with:
          python-version: "3.13"
          cache: pip
      - run: python -m pip install --disable-pip-version-check -e ".[dev]"
      - run: python -m tools.check io
      - if: always()
        uses: actions/upload-artifact@v7
        with:
          name: check-io-posix
          path: outputs/checks/io
          if-no-files-found: ignore

  io-liburing:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v6
        with:
          python-version: "3.13"
          cache: pip
      - run: sudo apt-get update && sudo apt-get install -y liburing-dev
      - run: python -m pip install --disable-pip-version-check -e ".[dev]"
      - run: python -m tools.check liburing
      - if: always()
        uses: actions/upload-artifact@v7
        with:
          name: check-io-liburing
          path: outputs/checks/liburing
          if-no-files-found: ignore

  llama-integration:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v7
        with:
          fetch-depth: 0
          submodules: recursive
      - uses: actions/setup-python@v6
        with:
          python-version: "3.13"
          cache: pip
      - run: sudo apt-get update && sudo apt-get install -y build-essential cmake
      - run: python -m pip install --disable-pip-version-check -e ".[dev]"
      - run: python -m tools.check llama
      - if: always()
        uses: actions/upload-artifact@v7
        with:
          name: check-llama
          path: outputs/checks/llama
          if-no-files-found: ignore
```

The Action majors (`checkout@v7`, `setup-python@v6`, and
`upload-artifact@v7`) were verified against the official `actions/*` tags on
2026-07-11.

- [ ] **Step 3: Run workflow and fast-profile contract tests**

Run:

```powershell
python -m pytest -q tests/test_ci_contract.py tests/test_check.py
python -m tools.check fast
```

Expected: tests and the Windows fast profile pass.

- [ ] **Step 4: Commit CI**

Run:

```powershell
git add .github/workflows/ci.yml tests/test_ci_contract.py
git commit -m 'ci: gate explicit harness profiles'
```

### Task 13: Align README, Agent Rules, And Phase Status

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `io/README.md`
- Modify: `integrations/llama_cpp/README.md`
- Modify: `docs/ROADMAP.md`
- Modify: `docs/PHASE1_STATUS.md`
- Modify: `docs/PHASE2_STATUS.md`
- Modify: `docs/PHASE3_STATUS.md`
- Modify: `docs/PHASE4_STATUS.md`
- Modify: `docs/PHASE4_RUNTIME_INTEGRATION_MAP.md`
- Modify: `docs/PHASE5_STATUS.md`
- Create: `docs/HARNESS_BASELINE_STATUS.md`
- Create: `tests/test_documentation_contract.py`

- [ ] **Step 1: Write a failing documentation contract test**

Create `tests/test_documentation_contract.py`:

```python
from pathlib import Path


def test_primary_docs_use_clean_clone_commands_and_paths() -> None:
    paths = [
        Path("README.md"),
        Path("AGENTS.md"),
        Path("io/README.md"),
        Path("integrations/llama_cpp/README.md"),
        Path("docs/ROADMAP.md"),
        Path("docs/PHASE4_RUNTIME_INTEGRATION_MAP.md"),
        *Path("docs").glob("PHASE*_STATUS.md"),
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert "/data/disk2/ljc" not in text
    assert ".deps/" not in text
    assert "PYTHONPATH=.deps/python" not in text
    assert "solid attention/" not in text
    assert "python -m tools.check fast" in Path("README.md").read_text(encoding="utf-8")
    assert ".\\.venv\\Scripts\\python.exe -m pip" in Path("README.md").read_text(encoding="utf-8")
    assert "python -m tools.check liburing" in Path("AGENTS.md").read_text(encoding="utf-8")


def test_harness_baseline_status_records_scope_boundary() -> None:
    text = Path("docs/HARNESS_BASELINE_STATUS.md").read_text(encoding="utf-8")

    assert "solidattention.run_manifest.v1" in text
    assert "load-before-attention" in text
    assert "not implemented in this baseline" in text
```

- [ ] **Step 2: Run documentation tests red**

Run:

```powershell
python -m pytest -q tests/test_documentation_contract.py
```

Expected: failures show stale absolute paths and the missing baseline status.

- [ ] **Step 3: Rewrite the README setup and verification entry point**

Make the README begin with explicit environment setup for both supported
developer shells. The PowerShell path must use the virtual environment's
interpreter even when script activation is blocked by execution policy:

```text
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m tools.check fast
```

The POSIX example is:

```text
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m tools.check fast
```

After activation, document the shell-neutral verification form
`python -m tools.check fast` used throughout the rest of the repository.

Document the four profiles in a table with owner platforms and prerequisites.
Document optional submodule initialization:

```text
git submodule update --init external/llama.cpp
```

Replace tracked-output claims with ignored `outputs/<run-id>/` and CI artifact
instructions. Preserve the existing Phase 1-5 mechanism summaries and current
Phase 4 pending items, but remove `/data/disk2`, every runnable `.deps/` path,
and wrapper-path commands.

- [ ] **Step 4: Strengthen `AGENTS.md` completion gates**

Add these durable rules:

```markdown
## Verification Profiles

- Every change must pass `python -m tools.check fast` on its development host.
- Linux I/O changes must also pass `python -m tools.check io`.
- liburing changes must pass `python -m tools.check liburing`; fallback is not success.
- llama.cpp patch changes must pass `python -m tools.check llama` at the pinned submodule revision.
- Required profiles may not use skip or fallback to report success.
- Generated outputs belong under ignored `outputs/`; deterministic inputs belong under `tests/fixtures/`.
- Experiment entry points must emit `solidattention.run_manifest.v1`.
```

Keep the existing phase scope and trace/metrics rules.

- [ ] **Step 5: Align roadmap and Phase status documents**

For each Phase status file, `docs/PHASE4_RUNTIME_INTEGRATION_MAP.md`,
`io/README.md`, and `integrations/llama_cpp/README.md`:

- Preserve implemented mechanism descriptions and verified historical numbers.
- Label imported metrics as historical evidence when their generated file is no
  longer tracked.
- Replace reproduction commands with `tools.check` or current root-relative
  commands.
- Remove `.deps/` worktree/build commands; dedicated profiles now own those
  temporary paths internally.
- Keep the three Phase 4 pending items unchanged.
- State that real-model Phase 5 work begins only after the harness baseline and
  the future `load-before-attention` vertical slice.

Create `docs/HARNESS_BASELINE_STATUS.md` with sections `Scope`, `Profiles`,
`Run Manifest`, `Clean Clone Contract`, `CI Evidence`, and `Next Runtime Slice`.
State exactly: `Scheduler-timed load-before-attention is not implemented in this baseline.`

- [ ] **Step 6: Run documentation and full fast checks**

Run:

```powershell
python -m pytest -q tests/test_documentation_contract.py
python -m tools.check fast
rg -n "/data/disk2/ljc|\.deps/|solid attention/" README.md AGENTS.md io/README.md integrations/llama_cpp/README.md docs/ROADMAP.md docs/PHASE1_STATUS.md docs/PHASE2_STATUS.md docs/PHASE3_STATUS.md docs/PHASE4_STATUS.md docs/PHASE4_RUNTIME_INTEGRATION_MAP.md docs/PHASE5_STATUS.md docs/HARNESS_BASELINE_STATUS.md
```

Expected: documentation tests and fast profile pass. `rg` has no matches in
primary current-command documentation; any intentionally quoted migration
history must be rewritten without obsolete runnable paths.

- [ ] **Step 7: Commit documentation alignment**

Run:

```powershell
git add README.md AGENTS.md io/README.md integrations/llama_cpp/README.md docs tests/test_documentation_contract.py
git commit -m 'docs: align reproducible harness workflow'
```

### Task 14: Verify, Rewrite `main`, Fresh-Clone, And Audit CI

**Files:**
- No new implementation files
- Rewrites: Git history for `main`
- Updates: GitHub `refs/heads/main`

- [ ] **Step 1: Run the complete local Windows evidence gate**

Run from the implementation worktree:

```powershell
python -m tools.check fast
if ($LASTEXITCODE -ne 0) { throw "fast profile failed" }
python -m harness.smoke --config configs/smoke_base.yaml --output-dir outputs/final/smoke
if ($LASTEXITCODE -ne 0) { throw "final smoke run failed" }
python -m harness.ablation --config configs/smoke_base.yaml --output-dir outputs/final/ablation
if ($LASTEXITCODE -ne 0) { throw "final ablation run failed" }
git diff --check
if ($LASTEXITCODE -ne 0) { throw "git diff --check failed" }
$pending = (& git status --porcelain) -join "`n"
if ($LASTEXITCODE -ne 0) { throw "git status failed" }
if ($pending) { throw "implementation worktree is dirty: $pending" }
```

Expected: fast status is passed with zero failures/errors/skips; smoke and
ablation manifests have `status=passed`; `git diff --check` is empty; status is
clean because `outputs/` is ignored.

- [ ] **Step 2: Audit the exact final tree**

Run:

```powershell
python -m pytest -q tests/test_repository_contract.py tests/test_ci_contract.py tests/test_documentation_contract.py
if ($LASTEXITCODE -ne 0) { throw "contract tests failed" }
$tree = (& git ls-tree -r HEAD) -join "`n"
if ($LASTEXITCODE -ne 0) { throw "git ls-tree failed" }
$gitlinks = @($tree -split "`n" | Select-String '^160000')
if ($gitlinks.Count -ne 1) { throw "expected one gitlink, got $($gitlinks.Count)" }
$expectedGitlink = "160000 commit bbebeec4a87355896e3faac0c2baca8130c91b6a`texternal/llama.cpp"
if ($gitlinks[0].Line -ne $expectedGitlink) { throw "unexpected gitlink: $($gitlinks[0].Line)" }
$tracked = (& git ls-files) -join "`n"
if ($LASTEXITCODE -ne 0) { throw "git ls-files failed" }
$forbidden = @($tracked -split "`n" | Select-String '^(\.deps/|solid_attention_deps/|outputs/|solid attention/)|^io/(block_io_probe|runtime_kv_command_consumer)(\.exe)?$|^io/.*\.(o|obj)$')
if ($forbidden.Count -ne 0) { throw "forbidden tracked paths remain: $($forbidden -join ', ')" }
```

Expected: contract tests pass; exactly one gitlink is printed for
`external/llama.cpp`; the forbidden-path query prints nothing.

- [ ] **Step 3: Create a local backup of the reviewable commit series**

Run:

```powershell
$featureTip = (git rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw "cannot resolve feature tip" }
git update-ref refs/backup/harness-baseline-series $featureTip
if ($LASTEXITCODE -ne 0) { throw "cannot preserve reviewable series" }
git log --oneline refs/backup/pre-harness-rewrite..refs/backup/harness-baseline-series
if ($LASTEXITCODE -ne 0) { throw "cannot inspect reviewable series" }
```

Expected: the local backup ref retains every Task commit. Do not push this ref.

- [ ] **Step 4: Create a parentless commit from the verified final tree**

Run:

```powershell
$featureTip = (git rev-parse refs/backup/harness-baseline-series).Trim()
if ($LASTEXITCODE -ne 0) { throw "cannot resolve preserved feature tip" }
$tree = (git rev-parse "$featureTip`^{tree}").Trim()
if ($LASTEXITCODE -ne 0) { throw "cannot resolve verified tree" }
$cleanRoot = (git commit-tree $tree -m 'build: establish reproducible harness baseline').Trim()
if ($LASTEXITCODE -ne 0) { throw "cannot create parentless commit" }
git update-ref refs/staging/harness-baseline-clean-root $cleanRoot
if ($LASTEXITCODE -ne 0) { throw "cannot preserve clean-root candidate" }
$commit = (& git cat-file -p refs/staging/harness-baseline-clean-root) -join "`n"
if ($LASTEXITCODE -ne 0) { throw "cannot inspect clean-root candidate" }
if ($commit -match '(?m)^parent ') { throw "clean-root candidate unexpectedly has a parent" }
$count = (& git rev-list --count refs/staging/harness-baseline-clean-root).Trim()
if ($LASTEXITCODE -ne 0 -or $count -ne '1') { throw "clean-root revision count is $count, expected 1" }
$candidateTree = (& git rev-parse 'refs/staging/harness-baseline-clean-root^{tree}').Trim()
if ($LASTEXITCODE -ne 0 -or $candidateTree -ne $tree) { throw "clean-root tree differs from verified tree" }
```

Expected: commit output has a `tree` line and no `parent` line; revision count
is `1`.

- [ ] **Step 5: Preflight the original workspace and remote lease**

Run:

```powershell
$cleanRoot = (& git rev-parse refs/staging/harness-baseline-clean-root).Trim()
if ($LASTEXITCODE -ne 0) { throw "cannot resolve clean-root candidate" }
$featureTip = (& git rev-parse refs/backup/harness-baseline-series).Trim()
if ($LASTEXITCODE -ne 0) { throw "cannot resolve preserved feature series" }
$original = (& git config --local --get solidattention.original-worktree).Trim()
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $original)) {
    throw "recorded original worktree is unavailable: $original"
}
$originalStatus = (& git -C $original status --porcelain) -join "`n"
if ($LASTEXITCODE -ne 0) { throw "cannot inspect original workspace" }
if ($originalStatus) { throw "original workspace is not clean: $originalStatus" }
$originalBranch = (& git -C $original symbolic-ref --short HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $originalBranch -ne 'main') {
    throw "original workspace must still be on main, got $originalBranch"
}
$originalMain = (& git -C $original rev-parse refs/heads/main).Trim()
if ($LASTEXITCODE -ne 0) { throw "cannot resolve original main" }
git merge-base --is-ancestor $originalMain $featureTip
if ($LASTEXITCODE -ne 0) { throw "original main is not preserved by the reviewable series" }
git -c http.sslVersion=tlsv1.2 -c http.version=HTTP/1.1 fetch origin main
if ($LASTEXITCODE -ne 0) { throw "origin/main fetch failed" }
$remoteMain = (git rev-parse origin/main).Trim()
if ($LASTEXITCODE -ne 0) { throw "cannot resolve origin/main" }
if ($remoteMain -ne "d0d2943444d7184840ece2308d71b264e2e9c299") {
    throw "origin/main changed during implementation; refusing rewrite: $remoteMain"
}
$candidateRemote = (& git -c http.sslVersion=tlsv1.2 -c http.version=HTTP/1.1 ls-remote --heads origin refs/heads/harness-candidate) -join "`n"
if ($LASTEXITCODE -ne 0) { throw "cannot inspect remote candidate branch" }
if ($candidateRemote -and $candidateRemote -ne "$cleanRoot`trefs/heads/harness-candidate") {
    throw "remote harness-candidate points at an unexpected SHA: $candidateRemote"
}
if ($candidateRemote) {
    $ownedCandidate = (& git show-ref --verify --hash refs/staging/harness-candidate-owned 2>$null).Trim()
    if ($LASTEXITCODE -ne 0 -or $ownedCandidate -ne $cleanRoot) {
        throw "candidate exists without this task's local ownership proof"
    }
}
```

Expected: both worktrees are clean, the original `main` is preserved by the
reviewable series, the lease still matches the approved imported root, and the
candidate branch is absent or already equals this exact candidate on a safe
resume.

- [ ] **Step 6: Push the parentless commit to the candidate branch**

Run:

```powershell
$cleanRoot = (& git rev-parse refs/staging/harness-baseline-clean-root).Trim()
if ($LASTEXITCODE -ne 0) { throw "cannot resolve clean-root candidate" }
$candidateRemote = (& git -c http.sslVersion=tlsv1.2 -c http.version=HTTP/1.1 ls-remote --heads origin refs/heads/harness-candidate) -join "`n"
if ($LASTEXITCODE -ne 0) { throw "cannot inspect candidate branch before creation" }
if (-not $candidateRemote) {
    git -c http.sslVersion=tlsv1.2 -c http.version=HTTP/1.1 push "--force-with-lease=refs/heads/harness-candidate:" origin "$cleanRoot`:refs/heads/harness-candidate"
    if ($LASTEXITCODE -ne 0) { throw "atomic candidate creation failed" }
} elseif ($candidateRemote -ne "$cleanRoot`trefs/heads/harness-candidate") {
    throw "candidate branch changed before push: $candidateRemote"
}
$candidateRemote = (& git -c http.sslVersion=tlsv1.2 -c http.version=HTTP/1.1 ls-remote --heads origin refs/heads/harness-candidate) -join "`n"
if ($LASTEXITCODE -ne 0) { throw "cannot verify candidate branch" }
if ($candidateRemote -ne "$cleanRoot`trefs/heads/harness-candidate") {
    throw "candidate branch does not point at $cleanRoot`: $candidateRemote"
}
git update-ref refs/staging/harness-candidate-owned $cleanRoot
if ($LASTEXITCODE -ne 0) { throw "cannot persist candidate ownership proof" }
```

Expected: `harness-candidate` points exactly at the parentless commit and the
CI workflow starts without changing `main`.

- [ ] **Step 7: Require every candidate CI job to pass**

Poll the public Actions API at most once every two minutes. Across both
40-minute gates this consumes at most 46 run/job requests, leaving room under
GitHub's unauthenticated primary rate limit for the explicit revalidation
requests. Filter by workflow name, event, branch, and SHA so a run for another
ref cannot satisfy the gate. During execution, keep user updates flowing while
the 30-second sleep increments run:

```powershell
$cleanRoot = (& git rev-parse refs/staging/harness-baseline-clean-root).Trim()
if ($LASTEXITCODE -ne 0) { throw "cannot resolve clean-root candidate" }
$deadline = (Get-Date).AddMinutes(40)
$run = $null
do {
    $url = "https://api.github.com/repos/xmuljc/solid-attention/actions/runs?event=push&branch=harness-candidate&head_sha=$cleanRoot&per_page=10"
    $runsJson = (& curl.exe --fail-with-body --connect-timeout 10 --max-time 30 -L --silent --show-error --tlsv1.2 --tls-max 1.2 --http1.1 $url) -join "`n"
    if ($LASTEXITCODE -ne 0) { throw "GitHub Actions run query failed" }
    $runs = $runsJson | ConvertFrom-Json
    $run = $runs.workflow_runs |
        Where-Object {
            $_.name -eq 'ci' -and
            $_.event -eq 'push' -and
            $_.head_branch -eq 'harness-candidate' -and
            $_.head_sha -eq $cleanRoot
        } |
        Sort-Object created_at -Descending |
        Select-Object -First 1
    if ($null -eq $run -or $run.status -ne 'completed') {
        1..4 | ForEach-Object { Start-Sleep -Seconds 30 }
    }
} while (($null -eq $run -or $run.status -ne 'completed') -and (Get-Date) -lt $deadline)

if ($null -eq $run) { throw "no candidate CI run found for $cleanRoot" }
if ($run.status -ne 'completed') { throw "candidate CI timed out for $cleanRoot" }
if ($run.conclusion -ne 'success') { throw "candidate workflow concluded $($run.conclusion)" }
$jobsJson = (& curl.exe --fail-with-body --connect-timeout 10 --max-time 30 -L --silent --show-error --tlsv1.2 --tls-max 1.2 --http1.1 "$($run.jobs_url)?per_page=100") -join "`n"
if ($LASTEXITCODE -ne 0) { throw "GitHub Actions jobs query failed" }
$jobs = ($jobsJson | ConvertFrom-Json).jobs
$requiredJobs = @(
    'fast (Windows / Python 3.11)',
    'fast (Windows / Python 3.13)',
    'fast (Linux / Python 3.11)',
    'fast (Linux / Python 3.13)',
    'io-posix',
    'io-liburing',
    'llama-integration'
)
foreach ($name in $requiredJobs) {
    $matches = @($jobs | Where-Object { $_.name -eq $name })
    if ($matches.Count -ne 1) { throw "candidate CI job count for $name is $($matches.Count), expected 1" }
    $job = $matches[0]
    if ($job.conclusion -ne 'success') {
        throw "candidate CI job $name concluded $($job.conclusion)"
    }
}
git update-ref refs/verified/harness-candidate-ci $cleanRoot
if ($LASTEXITCODE -ne 0) { throw "cannot persist candidate CI proof ref" }
git config --local solidattention.candidate-ci-run-id $run.id
if ($LASTEXITCODE -ne 0) { throw "cannot persist candidate CI run id" }
```

Expected: all seven jobs conclude `success`. If any candidate job fails, leave
`main` untouched. Fix and commit on `harness-baseline`, update
`refs/backup/harness-baseline-series`, recreate and verify the parentless
commit, then update the owned candidate branch with an exact
`--force-with-lease=refs/heads/harness-candidate:<old-candidate-sha>` and repeat
this gate. Never move the local feature branch to the parentless commit before
candidate CI is green.

- [ ] **Step 8: Force-update GitHub `main` using the exact lease**

Re-run every local and remote precondition immediately before the destructive
ref update, then use the approved imported SHA as the exact lease:

```powershell
$cleanRoot = (& git rev-parse refs/staging/harness-baseline-clean-root).Trim()
if ($LASTEXITCODE -ne 0) { throw "cannot resolve candidate SHA" }
$featureTip = (& git rev-parse refs/backup/harness-baseline-series).Trim()
if ($LASTEXITCODE -ne 0) { throw "cannot resolve preserved feature series" }
$implementationBranch = (& git symbolic-ref --short HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $implementationBranch -ne 'harness-baseline') {
    throw "implementation worktree left harness-baseline during candidate CI"
}
$implementationHead = (& git rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $implementationHead -ne $featureTip) {
    throw "implementation HEAD changed after candidate creation"
}
$implementationStatus = (& git status --porcelain) -join "`n"
if ($LASTEXITCODE -ne 0 -or $implementationStatus) {
    throw "implementation worktree changed during candidate CI: $implementationStatus"
}
$featureTree = (& git rev-parse 'refs/backup/harness-baseline-series^{tree}').Trim()
if ($LASTEXITCODE -ne 0) { throw "cannot resolve feature tree" }
$cleanTree = (& git rev-parse 'refs/staging/harness-baseline-clean-root^{tree}').Trim()
if ($LASTEXITCODE -ne 0 -or $cleanTree -ne $featureTree) {
    throw "candidate tree no longer equals the reviewable feature tree"
}
$verifiedCandidate = (& git rev-parse refs/verified/harness-candidate-ci).Trim()
if ($LASTEXITCODE -ne 0 -or $verifiedCandidate -ne $cleanRoot) {
    throw "candidate CI proof ref is missing or stale"
}
$candidateRunId = (& git config --local --get solidattention.candidate-ci-run-id).Trim()
if ($LASTEXITCODE -ne 0 -or -not $candidateRunId) { throw "candidate CI run id is missing" }
$candidateRunJson = (& curl.exe --fail-with-body --connect-timeout 10 --max-time 30 -L --silent --show-error --tlsv1.2 --tls-max 1.2 --http1.1 "https://api.github.com/repos/xmuljc/solid-attention/actions/runs/$candidateRunId") -join "`n"
if ($LASTEXITCODE -ne 0) { throw "cannot revalidate candidate CI run" }
$candidateRun = $candidateRunJson | ConvertFrom-Json
if ($candidateRun.name -ne 'ci' -or $candidateRun.event -ne 'push' -or $candidateRun.status -ne 'completed' -or $candidateRun.head_branch -ne 'harness-candidate' -or $candidateRun.head_sha -ne $cleanRoot -or $candidateRun.conclusion -ne 'success') {
    throw "persisted candidate CI run no longer proves this clean root"
}
$original = (& git config --local --get solidattention.original-worktree).Trim()
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $original)) {
    throw "recorded original worktree is unavailable"
}
$originalStatus = (& git -C $original status --porcelain) -join "`n"
if ($LASTEXITCODE -ne 0 -or $originalStatus) { throw "original workspace changed during CI: $originalStatus" }
$originalBranch = (& git -C $original symbolic-ref --short HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $originalBranch -ne 'main') {
    throw "original workspace left main during candidate CI"
}
$originalMain = (& git -C $original rev-parse refs/heads/main).Trim()
if ($LASTEXITCODE -ne 0) { throw "cannot resolve original main" }
git merge-base --is-ancestor $originalMain $featureTip
if ($LASTEXITCODE -ne 0) { throw "original main is not preserved by the feature series" }
git -c http.sslVersion=tlsv1.2 -c http.version=HTTP/1.1 fetch origin main
if ($LASTEXITCODE -ne 0) { throw "origin/main fetch failed" }
$remoteMain = (& git rev-parse origin/main).Trim()
if ($LASTEXITCODE -ne 0 -or $remoteMain -ne 'd0d2943444d7184840ece2308d71b264e2e9c299') {
    throw "origin/main lease changed; refusing rewrite: $remoteMain"
}
$candidateRemote = (& git -c http.sslVersion=tlsv1.2 -c http.version=HTTP/1.1 ls-remote --heads origin refs/heads/harness-candidate) -join "`n"
if ($LASTEXITCODE -ne 0 -or $candidateRemote -ne "$cleanRoot`trefs/heads/harness-candidate") {
    throw "verified candidate branch changed: $candidateRemote"
}
git -c http.sslVersion=tlsv1.2 -c http.version=HTTP/1.1 -c http.postBuffer=524288000 push --force-with-lease=refs/heads/main:d0d2943444d7184840ece2308d71b264e2e9c299 origin "$cleanRoot`:refs/heads/main"
if ($LASTEXITCODE -ne 0) { throw "exact-lease main update failed" }
$remoteRefs = (& git -c http.sslVersion=tlsv1.2 -c http.version=HTTP/1.1 ls-remote --symref origin HEAD refs/heads/main) -join "`n"
if ($LASTEXITCODE -ne 0) { throw "cannot verify remote main" }
if (-not $remoteRefs.Contains("ref: refs/heads/main`tHEAD")) { throw "remote HEAD is not main" }
if (-not $remoteRefs.Contains("$cleanRoot`tHEAD")) { throw "remote HEAD does not resolve to clean root" }
if (-not $remoteRefs.Contains("$cleanRoot`trefs/heads/main")) { throw "remote main does not resolve to clean root" }
```

Expected: the exact lease succeeds only for the candidate that already passed
all seven jobs; remote `HEAD` and `main` both equal the clean root.

- [ ] **Step 9: Prove a fresh non-recursive GitHub clone works**

Create a unique new directory under `D:\tmp`; never delete or reuse an existing
directory. Use an isolated virtual environment and mechanically assert the
remote SHA, history shape, and submodule state:

```powershell
$implementationRoot = (& git rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0) { throw "cannot resolve implementation worktree" }
$cleanRoot = (& git rev-parse refs/staging/harness-baseline-clean-root).Trim()
if ($LASTEXITCODE -ne 0) { throw "cannot resolve clean-root SHA" }
$verifyPath = Join-Path 'D:\tmp' ("solid-attention-clean-clone-" + [guid]::NewGuid().ToString('N'))
if (Test-Path -LiteralPath $verifyPath) { throw "unique clone path unexpectedly exists: $verifyPath" }
git -c http.sslVersion=tlsv1.2 -c http.version=HTTP/1.1 clone --no-recurse-submodules https://github.com/xmuljc/solid-attention.git $verifyPath
if ($LASTEXITCODE -ne 0) { throw "fresh clone failed" }
git -C $implementationRoot config --local solidattention.verify-clone $verifyPath
if ($LASTEXITCODE -ne 0) { throw "cannot record fresh-clone path" }
Set-Location -LiteralPath $verifyPath -ErrorAction Stop
if ((Get-Location).Path -ne (Resolve-Path -LiteralPath $verifyPath).Path) { throw "failed to enter fresh clone" }
python -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "fresh-clone virtual environment creation failed" }
& .\.venv\Scripts\python.exe -m pip install --disable-pip-version-check -e ".[dev]"
if ($LASTEXITCODE -ne 0) { throw "fresh-clone dependency installation failed" }
& .\.venv\Scripts\python.exe -m tools.check fast
if ($LASTEXITCODE -ne 0) { throw "fresh-clone fast profile failed" }
& .\.venv\Scripts\python.exe -m pytest -q tests/test_repository_contract.py tests/test_run_manifest.py tests/test_ci_contract.py tests/test_documentation_contract.py
if ($LASTEXITCODE -ne 0) { throw "fresh-clone focused contracts failed" }
$cloneHead = (& git rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $cloneHead -ne $cleanRoot) { throw "fresh clone HEAD is $cloneHead, expected $cleanRoot" }
$commitCount = (& git rev-list --count HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $commitCount -ne '1') { throw "fresh clone commit count is $commitCount, expected 1" }
$submoduleState = (& git submodule status) -join "`n"
if ($LASTEXITCODE -ne 0) { throw "fresh-clone submodule status failed" }
if ($submoduleState -notmatch '^-bbebeec4a87355896e3faac0c2baca8130c91b6a external/llama\.cpp(?: |$)') {
    throw "unexpected fresh-clone submodule state: $submoduleState"
}
$cloneStatus = (& git status --porcelain) -join "`n"
if ($LASTEXITCODE -ne 0 -or $cloneStatus) { throw "fresh clone is dirty: $cloneStatus" }
Set-Location -LiteralPath $implementationRoot -ErrorAction Stop
if ((Get-Location).Path -ne (Resolve-Path -LiteralPath $implementationRoot).Path) { throw "failed to return to implementation worktree" }
git update-ref refs/verified/harness-fresh-clone $cleanRoot
if ($LASTEXITCODE -ne 0) { throw "cannot persist fresh-clone proof ref" }
```

Expected: fast passes without initializing llama.cpp; the only submodule has a
leading `-` at the pinned SHA; the clone is clean at the one-commit root.

- [ ] **Step 10: Verify all GitHub Actions jobs for `main`**

Read the public Actions API for `xmuljc/solid-attention` and require a distinct
`push` run for branch `main`. Poll at most once every two minutes so the earlier
candidate run for the same SHA cannot satisfy this gate while the total public
API request budget remains bounded:

```powershell
$cleanRoot = (& git rev-parse refs/staging/harness-baseline-clean-root).Trim()
if ($LASTEXITCODE -ne 0) { throw "cannot resolve clean-root SHA" }
$deadline = (Get-Date).AddMinutes(40)
$run = $null
do {
    $url = "https://api.github.com/repos/xmuljc/solid-attention/actions/runs?event=push&branch=main&head_sha=$cleanRoot&per_page=10"
    $runsJson = (& curl.exe --fail-with-body --connect-timeout 10 --max-time 30 -L --silent --show-error --tlsv1.2 --tls-max 1.2 --http1.1 $url) -join "`n"
    if ($LASTEXITCODE -ne 0) { throw "GitHub Actions main-run query failed" }
    $runs = $runsJson | ConvertFrom-Json
    $run = $runs.workflow_runs |
        Where-Object {
            $_.name -eq 'ci' -and
            $_.event -eq 'push' -and
            $_.head_branch -eq 'main' -and
            $_.head_sha -eq $cleanRoot
        } |
        Sort-Object created_at -Descending |
        Select-Object -First 1
    if ($null -eq $run -or $run.status -ne 'completed') {
        1..4 | ForEach-Object { Start-Sleep -Seconds 30 }
    }
} while (($null -eq $run -or $run.status -ne 'completed') -and (Get-Date) -lt $deadline)

if ($null -eq $run) { throw "no main CI run found for $cleanRoot" }
if ($run.status -ne 'completed') { throw "main CI timed out for $cleanRoot" }
if ($run.conclusion -ne 'success') { throw "main workflow concluded $($run.conclusion)" }

$jobsJson = (& curl.exe --fail-with-body --connect-timeout 10 --max-time 30 -L --silent --show-error --tlsv1.2 --tls-max 1.2 --http1.1 "$($run.jobs_url)?per_page=100") -join "`n"
if ($LASTEXITCODE -ne 0) { throw "GitHub Actions main-job query failed" }
$jobs = ($jobsJson | ConvertFrom-Json).jobs
$requiredJobs = @(
    'fast (Windows / Python 3.11)',
    'fast (Windows / Python 3.13)',
    'fast (Linux / Python 3.11)',
    'fast (Linux / Python 3.13)',
    'io-posix',
    'io-liburing',
    'llama-integration'
)
foreach ($name in $requiredJobs) {
    $matches = @($jobs | Where-Object { $_.name -eq $name })
    if ($matches.Count -ne 1) { throw "main CI job count for $name is $($matches.Count), expected 1" }
    $job = $matches[0]
    if ($job.conclusion -ne 'success') { throw "required main job $name concluded $($job.conclusion)" }
}
git update-ref refs/verified/harness-main-ci $cleanRoot
if ($LASTEXITCODE -ne 0) { throw "cannot persist main CI proof ref" }
git config --local solidattention.main-ci-run-id $run.id
if ($LASTEXITCODE -ne 0) { throw "cannot persist main CI run id" }
```

Required job conclusions are:

```text
fast (Windows / Python 3.11)   success
fast (Windows / Python 3.13)   success
fast (Linux / Python 3.11)     success
fast (Linux / Python 3.13)     success
io-posix                       success
io-liburing                    success
llama-integration              success
```

If the fresh-clone or `main` run fails despite candidate success, do not move
the local branches or delete the candidate branch. Inspect the exact failure.
If the clean root must be replaced, fix the preserved reviewable series and use
the current clean-root SHA as the next exact `main` lease. The rollback command,
when restoring the imported root is necessary, is:

```powershell
$cleanRoot = (& git rev-parse refs/staging/harness-baseline-clean-root).Trim()
$oldRoot = (& git rev-parse refs/backup/pre-harness-rewrite).Trim()
git -c http.sslVersion=tlsv1.2 -c http.version=HTTP/1.1 push --force-with-lease=refs/heads/main:$cleanRoot origin "$oldRoot`:refs/heads/main"
if ($LASTEXITCODE -ne 0) { throw "exact-lease rollback failed; do not retry without refetching" }
```

Do not claim completion while a required job is skipped, cancelled, missing,
or failing.

- [ ] **Step 11: Remove the candidate branch and align local branches**

Only after both candidate and `main` CI are green, delete the owned temporary
branch with an exact lease. Then move the implementation branch and original
`main` to the identical verified tree while preserving all three backup refs:

```powershell
$cleanRoot = (& git rev-parse refs/staging/harness-baseline-clean-root).Trim()
if ($LASTEXITCODE -ne 0) { throw "cannot resolve clean-root SHA" }
$featureTip = (& git rev-parse refs/backup/harness-baseline-series).Trim()
if ($LASTEXITCODE -ne 0) { throw "cannot resolve preserved feature series" }
$implementationHead = (& git rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $implementationHead -ne $featureTip) {
    throw "implementation HEAD changed after remote verification"
}
$featureTree = (& git rev-parse 'refs/backup/harness-baseline-series^{tree}').Trim()
if ($LASTEXITCODE -ne 0) { throw "cannot resolve feature tree" }
$cleanTree = (& git rev-parse 'refs/staging/harness-baseline-clean-root^{tree}').Trim()
if ($LASTEXITCODE -ne 0 -or $cleanTree -ne $featureTree) {
    throw "verified clean root no longer matches the feature series"
}
foreach ($proofRef in @(
    'refs/verified/harness-candidate-ci',
    'refs/verified/harness-fresh-clone',
    'refs/verified/harness-main-ci'
)) {
    $proofSha = (& git rev-parse $proofRef).Trim()
    if ($LASTEXITCODE -ne 0 -or $proofSha -ne $cleanRoot) {
        throw "verification proof is missing or stale: $proofRef"
    }
}
$ownedCandidate = (& git rev-parse refs/staging/harness-candidate-owned).Trim()
if ($LASTEXITCODE -ne 0 -or $ownedCandidate -ne $cleanRoot) {
    throw "candidate ownership proof is missing or stale"
}
$remoteMainLine = (& git -c http.sslVersion=tlsv1.2 -c http.version=HTTP/1.1 ls-remote origin refs/heads/main) -join "`n"
if ($LASTEXITCODE -ne 0 -or $remoteMainLine -ne "$cleanRoot`trefs/heads/main") {
    throw "remote main changed after CI: $remoteMainLine"
}
$mainRunId = (& git config --local --get solidattention.main-ci-run-id).Trim()
if ($LASTEXITCODE -ne 0 -or -not $mainRunId) { throw "main CI run id is missing" }
$mainRunJson = (& curl.exe --fail-with-body --connect-timeout 10 --max-time 30 -L --silent --show-error --tlsv1.2 --tls-max 1.2 --http1.1 "https://api.github.com/repos/xmuljc/solid-attention/actions/runs/$mainRunId") -join "`n"
if ($LASTEXITCODE -ne 0) { throw "cannot revalidate main CI run" }
$mainRun = $mainRunJson | ConvertFrom-Json
if ($mainRun.name -ne 'ci' -or $mainRun.event -ne 'push' -or $mainRun.status -ne 'completed' -or $mainRun.head_branch -ne 'main' -or $mainRun.head_sha -ne $cleanRoot -or $mainRun.conclusion -ne 'success') {
    throw "persisted main CI run no longer proves this clean root"
}
$original = (& git config --local --get solidattention.original-worktree).Trim()
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $original)) {
    throw "recorded original worktree is unavailable"
}
$implementationBranch = (& git symbolic-ref --short HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $implementationBranch -ne 'harness-baseline') {
    throw "implementation worktree is not on harness-baseline"
}
$implementationStatus = (& git status --porcelain) -join "`n"
if ($LASTEXITCODE -ne 0 -or $implementationStatus) { throw "implementation worktree is dirty" }
$originalStatus = (& git -C $original status --porcelain) -join "`n"
if ($LASTEXITCODE -ne 0 -or $originalStatus) { throw "original workspace is dirty" }
$originalBranch = (& git -C $original symbolic-ref --short HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $originalBranch -ne 'main') {
    throw "original workspace is no longer on main"
}
$originalMain = (& git -C $original rev-parse refs/heads/main).Trim()
if ($LASTEXITCODE -ne 0) { throw "cannot resolve original main" }
git merge-base --is-ancestor $originalMain $featureTip
if ($LASTEXITCODE -ne 0) { throw "original main is not preserved by the feature series" }
git -c http.sslVersion=tlsv1.2 -c http.version=HTTP/1.1 push --force-with-lease=refs/heads/harness-candidate:$cleanRoot origin ':refs/heads/harness-candidate'
if ($LASTEXITCODE -ne 0) { throw "exact-lease candidate cleanup failed" }
$candidateRemote = (& git -c http.sslVersion=tlsv1.2 -c http.version=HTTP/1.1 ls-remote --heads origin refs/heads/harness-candidate) -join "`n"
if ($LASTEXITCODE -ne 0 -or $candidateRemote) { throw "candidate branch still exists: $candidateRemote" }
git update-ref -d refs/staging/harness-candidate-owned $cleanRoot
if ($LASTEXITCODE -ne 0) { throw "cannot clear candidate ownership ref" }
git switch --detach $featureTip
if ($LASTEXITCODE -ne 0) { throw "cannot detach implementation worktree" }
git update-ref refs/heads/harness-baseline $cleanRoot $featureTip
if ($LASTEXITCODE -ne 0) { throw "cannot align harness-baseline branch" }
git switch harness-baseline
if ($LASTEXITCODE -ne 0) { throw "cannot restore implementation branch" }
git -C $original switch --detach
if ($LASTEXITCODE -ne 0) { throw "cannot detach original worktree" }
git -C $original update-ref refs/heads/main $cleanRoot $originalMain
if ($LASTEXITCODE -ne 0) { throw "cannot align local main" }
git -C $original switch main
if ($LASTEXITCODE -ne 0) { throw "cannot restore original worktree to main" }
$localMain = (& git -C $original rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $localMain -ne $cleanRoot) { throw "local main is $localMain, expected $cleanRoot" }
$finalOriginalStatus = (& git -C $original status --porcelain) -join "`n"
if ($LASTEXITCODE -ne 0 -or $finalOriginalStatus) { throw "aligned original workspace is dirty" }
```

Expected: the remote candidate branch is absent; both local branches point to
the clean root; the original workspace is clean on `main`. The imported root
and every review commit remain reachable through the three local backup refs.

- [ ] **Step 12: Run the final requirement-by-requirement audit**

Use the recorded fresh clone and its virtual environment; do not rely on the
implementation worktree's Python state:

```powershell
$cleanRoot = (& git rev-parse refs/staging/harness-baseline-clean-root).Trim()
if ($LASTEXITCODE -ne 0) { throw "cannot resolve final clean-root SHA" }
$implementationRoot = (& git rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0) { throw "cannot resolve implementation worktree" }
$original = (& git config --local --get solidattention.original-worktree).Trim()
if ($LASTEXITCODE -ne 0) { throw "cannot resolve original worktree" }
$verifyPath = (& git config --local --get solidattention.verify-clone).Trim()
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $verifyPath)) {
    throw "recorded fresh clone is unavailable"
}
$verifyPython = Join-Path $verifyPath '.venv\Scripts\python.exe'
Set-Location -LiteralPath $verifyPath -ErrorAction Stop
if ((Get-Location).Path -ne (Resolve-Path -LiteralPath $verifyPath).Path) { throw "failed to enter recorded fresh clone" }
& $verifyPython -m tools.check fast
if ($LASTEXITCODE -ne 0) { throw "final fresh-clone fast profile failed" }
& $verifyPython -m pytest -q tests/test_repository_contract.py tests/test_run_manifest.py tests/test_ci_contract.py tests/test_documentation_contract.py
if ($LASTEXITCODE -ne 0) { throw "final fresh-clone contracts failed" }
$verifyStatus = (& git -C $verifyPath status --porcelain) -join "`n"
if ($LASTEXITCODE -ne 0 -or $verifyStatus) { throw "fresh clone is dirty" }
$verifyHead = (& git -C $verifyPath rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $verifyHead -ne $cleanRoot) { throw "fresh clone SHA mismatch" }
Set-Location -LiteralPath $implementationRoot -ErrorAction Stop
if ((Get-Location).Path -ne (Resolve-Path -LiteralPath $implementationRoot).Path) { throw "failed to return to implementation worktree" }
$localMain = (& git -C $original rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $localMain -ne $cleanRoot) { throw "local main SHA mismatch" }
$remoteMainLine = (& git -c http.sslVersion=tlsv1.2 -c http.version=HTTP/1.1 ls-remote origin refs/heads/main) -join "`n"
if ($LASTEXITCODE -ne 0 -or $remoteMainLine -ne "$cleanRoot`trefs/heads/main") {
    throw "remote main SHA mismatch: $remoteMainLine"
}
$candidateRemote = (& git -c http.sslVersion=tlsv1.2 -c http.version=HTTP/1.1 ls-remote --heads origin refs/heads/harness-candidate) -join "`n"
if ($LASTEXITCODE -ne 0 -or $candidateRemote) { throw "candidate branch unexpectedly remains" }
$preRewrite = (& git rev-parse refs/backup/pre-harness-rewrite).Trim()
if ($LASTEXITCODE -ne 0 -or $preRewrite -ne 'd0d2943444d7184840ece2308d71b264e2e9c299') {
    throw "pre-rewrite backup ref is missing or wrong"
}
$preLocalMain = (& git rev-parse refs/backup/pre-harness-local-main).Trim()
if ($LASTEXITCODE -ne 0) { throw "original local-main backup ref is missing" }
$seriesTip = (& git rev-parse refs/backup/harness-baseline-series).Trim()
if ($LASTEXITCODE -ne 0) { throw "reviewable-series backup ref is missing" }
git merge-base --is-ancestor $preLocalMain $seriesTip
if ($LASTEXITCODE -ne 0) { throw "reviewable series no longer preserves original local main" }
$ciRunIds = @{}
foreach ($proof in @(
    @{ Key = 'solidattention.candidate-ci-run-id'; Branch = 'harness-candidate' },
    @{ Key = 'solidattention.main-ci-run-id'; Branch = 'main' }
)) {
    $runId = (& git config --local --get $proof.Key).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $runId) { throw "missing final CI run id: $($proof.Key)" }
    $ciRunIds[$proof.Branch] = $runId
    $runJson = (& curl.exe --fail-with-body --connect-timeout 10 --max-time 30 -L --silent --show-error --tlsv1.2 --tls-max 1.2 --http1.1 "https://api.github.com/repos/xmuljc/solid-attention/actions/runs/$runId") -join "`n"
    if ($LASTEXITCODE -ne 0) { throw "cannot perform final CI proof query for $($proof.Branch)" }
    $run = $runJson | ConvertFrom-Json
    if ($run.name -ne 'ci' -or $run.event -ne 'push' -or $run.status -ne 'completed' -or $run.head_branch -ne $proof.Branch -or $run.head_sha -ne $cleanRoot -or $run.conclusion -ne 'success') {
        throw "final CI proof is invalid for $($proof.Branch)"
    }
}
git config --local --unset-all solidattention.original-worktree
if ($LASTEXITCODE -ne 0) { throw "cannot clear original-worktree task metadata" }
git config --local --unset-all solidattention.verify-clone
if ($LASTEXITCODE -ne 0) { throw "cannot clear verify-clone task metadata" }
git config --local --unset-all solidattention.candidate-ci-run-id
if ($LASTEXITCODE -ne 0) { throw "cannot clear candidate CI task metadata" }
git config --local --unset-all solidattention.main-ci-run-id
if ($LASTEXITCODE -ne 0) { throw "cannot clear main CI task metadata" }
$ciRunIds | ConvertTo-Json | Write-Host
```

Expected: the fresh clone, local `main`, and remote `main` all equal the same
parentless SHA; the candidate branch is gone; fast and all focused contracts
pass; all three backup refs remain valid. Record the remote SHA, local SHA,
fresh-clone test counts, candidate/main run IDs, and all seven CI job
conclusions in the final handoff.
