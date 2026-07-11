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
REGULAR_FILE_MODES = {"100644", "100755"}


@dataclass(frozen=True)
class RepoEntry:
    mode: str
    path: str


def find_git_root(start: Path) -> Path:
    start_dir = start.parent if start.is_file() else start
    result = subprocess.run(
        ["git", "-C", str(start_dir), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return Path(result.stdout.strip())


def read_index(root: Path) -> list[RepoEntry]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--stage", "-z"],
        check=True,
        capture_output=True,
    )
    entries = []
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        metadata, encoded_path = record.split(b"\t", 1)
        mode = metadata.split(b" ", 1)[0].decode("ascii")
        entries.append(RepoEntry(mode, encoded_path.decode("utf-8")))
    return entries


def validate_entries(entries: Iterable[RepoEntry]) -> list[str]:
    entry_list = list(entries)
    paths = {entry.path for entry in entry_list}
    gitlinks = {entry.path for entry in entry_list if entry.mode == "160000"}
    violations = []
    paths_by_casefold: dict[str, set[str]] = {}

    for entry in entry_list:
        path = entry.path
        casefolded_path = path.casefold()
        paths_by_casefold.setdefault(casefolded_path, set()).add(path)
        if casefolded_path.startswith(FORBIDDEN_PREFIXES[0]):
            violations.append(f"wrapper path is tracked: {path}")
        elif casefolded_path.startswith(FORBIDDEN_PREFIXES[1:]):
            violations.append(f"generated/cache path is tracked: {path}")

        if casefolded_path in COMPILED_BINARIES or (
            casefolded_path.startswith("io/")
            and Path(casefolded_path).suffix in COMPILED_SUFFIXES
        ):
            violations.append(f"compiled binary is tracked: {path}")

        if entry.mode == "160000" and path not in ALLOWED_GITLINKS:
            violations.append(f"unexpected gitlink: {path}")

        if path in REQUIRED_PATHS and entry.mode not in REGULAR_FILE_MODES:
            violations.append(
                f"required path is not a regular file: {path} (mode {entry.mode})"
            )

    for equivalent_paths in paths_by_casefold.values():
        if len(equivalent_paths) > 1:
            violations.append(
                "case-insensitive path collision: "
                + ", ".join(sorted(equivalent_paths))
            )

    for path in REQUIRED_PATHS - paths:
        violations.append(f"required path is missing: {path}")

    if gitlinks != ALLOWED_GITLINKS:
        violations.append(
            "gitlink set mismatch: "
            f"expected {sorted(ALLOWED_GITLINKS)}, found {sorted(gitlinks)}"
        )

    return sorted(set(violations))


def check_repository(start: Path) -> list[str]:
    root = find_git_root(start)
    return validate_entries(read_index(root))
