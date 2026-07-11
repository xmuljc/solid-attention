import subprocess
from pathlib import Path

from pytest import MonkeyPatch

from tools.repository_contract import (
    RepoEntry,
    check_repository,
    find_git_root,
    read_index,
    validate_entries,
)


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
        RepoEntry("100644", "io/runtime_kv_command_consumer.OBJ"),
        RepoEntry("160000", "external/temporary-worktree"),
    ]
    violations = validate_entries(entries)
    assert "wrapper path is tracked: solid attention/README.md" in violations
    assert "generated/cache path is tracked: .deps/python/pytest.py" in violations
    assert "compiled binary is tracked: io/block_io_probe" in violations
    assert "compiled binary is tracked: io/runtime_kv_command_consumer.exe" in violations
    assert "compiled binary is tracked: io/runtime_kv_command_consumer.obj" in violations
    assert "compiled binary is tracked: io/runtime_kv_command_consumer.OBJ" in violations
    assert "unexpected gitlink: external/temporary-worktree" in violations


def test_validate_entries_reports_only_missing_required_path() -> None:
    entries = [
        RepoEntry("100644", "README.md"),
        RepoEntry("100644", "AGENTS.md"),
        RepoEntry("100644", ".gitmodules"),
        RepoEntry("160000", "external/llama.cpp"),
    ]
    assert validate_entries(entries) == [
        "required path is missing: pyproject.toml"
    ]


def test_validate_entries_reports_exact_gitlink_set_mismatch() -> None:
    entries = [
        RepoEntry("100644", "README.md"),
        RepoEntry("100644", "AGENTS.md"),
        RepoEntry("100644", ".gitmodules"),
        RepoEntry("100644", "pyproject.toml"),
        RepoEntry("160000", "external/temporary-worktree"),
    ]
    assert validate_entries(entries) == [
        "gitlink set mismatch: expected ['external/llama.cpp'], "
        "found ['external/temporary-worktree']",
        "unexpected gitlink: external/temporary-worktree",
    ]


def test_validate_entries_rejects_case_insensitive_forbidden_paths() -> None:
    entries = [
        RepoEntry("100644", "README.md"),
        RepoEntry("100644", "AGENTS.md"),
        RepoEntry("100644", ".gitmodules"),
        RepoEntry("100644", "pyproject.toml"),
        RepoEntry("100644", "Solid Attention/README.md"),
        RepoEntry("100644", "Solid_Attention_Deps/cache/x"),
        RepoEntry("100644", ".DEPS/cache/x"),
        RepoEntry("100644", "Outputs/run.json"),
        RepoEntry("160000", "external/llama.cpp"),
    ]
    assert validate_entries(entries) == [
        "generated/cache path is tracked: .DEPS/cache/x",
        "generated/cache path is tracked: Outputs/run.json",
        "generated/cache path is tracked: Solid_Attention_Deps/cache/x",
        "wrapper path is tracked: Solid Attention/README.md",
    ]


def test_validate_entries_rejects_case_insensitive_compiled_paths() -> None:
    entries = [
        RepoEntry("100644", "README.md"),
        RepoEntry("100644", "AGENTS.md"),
        RepoEntry("100644", ".gitmodules"),
        RepoEntry("100644", "pyproject.toml"),
        RepoEntry("100755", "IO/BLOCK_IO_PROBE.EXE"),
        RepoEntry("100644", "IO/helper.OBJ"),
        RepoEntry("160000", "external/llama.cpp"),
    ]
    assert validate_entries(entries) == [
        "compiled binary is tracked: IO/BLOCK_IO_PROBE.EXE",
        "compiled binary is tracked: IO/helper.OBJ",
    ]


def test_validate_entries_reports_case_insensitive_path_collision() -> None:
    entries = [
        RepoEntry("100644", "README.md"),
        RepoEntry("100644", "AGENTS.md"),
        RepoEntry("100644", ".gitmodules"),
        RepoEntry("100644", "pyproject.toml"),
        RepoEntry("100644", "core/x.py"),
        RepoEntry("100644", "Core/x.py"),
        RepoEntry("160000", "external/llama.cpp"),
    ]
    assert validate_entries(entries) == [
        "case-insensitive path collision: Core/x.py, core/x.py"
    ]


def test_validate_entries_rejects_non_regular_required_path() -> None:
    entries = [
        RepoEntry("120000", "README.md"),
        RepoEntry("100644", "AGENTS.md"),
        RepoEntry("100644", ".gitmodules"),
        RepoEntry("100755", "pyproject.toml"),
        RepoEntry("160000", "external/llama.cpp"),
    ]
    assert validate_entries(entries) == [
        "required path is not a regular file: README.md (mode 120000)"
    ]


def test_validate_entries_keeps_gitlink_matching_case_sensitive() -> None:
    entries = [
        RepoEntry("100644", "README.md"),
        RepoEntry("100644", "AGENTS.md"),
        RepoEntry("100644", ".gitmodules"),
        RepoEntry("100644", "pyproject.toml"),
        RepoEntry("160000", "external/Llama.cpp"),
    ]
    assert validate_entries(entries) == [
        "gitlink set mismatch: expected ['external/llama.cpp'], "
        "found ['external/Llama.cpp']",
        "unexpected gitlink: external/Llama.cpp",
    ]


def test_find_git_root_decodes_utf8_output(monkeypatch: MonkeyPatch) -> None:
    start = Path("D:/workspace/固态注意力")
    root = start
    calls = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(
            args,
            0,
            stdout=f"{root}\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert find_git_root(start) == root
    assert calls == [
        (
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
            {
                "check": True,
                "capture_output": True,
                "text": True,
                "encoding": "utf-8",
            },
        )
    ]


def test_read_index_preserves_tab_and_newline_in_path(
    monkeypatch: MonkeyPatch,
) -> None:
    root = Path("repo")
    path = "docs/name\twith\nline.txt"
    output = b"100644 " + (b"a" * 40) + b" 0\t" + path.encode("utf-8") + b"\0"
    calls = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout=output, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert read_index(root) == [RepoEntry("100644", path)]
    assert calls == [
        (
            ["git", "-C", str(root), "ls-files", "--stage", "-z"],
            {"check": True, "capture_output": True},
        )
    ]


def test_current_repository_contract_is_clean() -> None:
    violations = check_repository(Path(__file__).resolve())
    assert violations == []
