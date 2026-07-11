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
    revision = stdout.strip()
    if revision != PINNED_LLAMA_COMMIT:
        raise RuntimeError(
            f"wrong llama.cpp revision: expected {PINNED_LLAMA_COMMIT}, got {revision}"
        )


def llama_build_commands(
    *,
    checkout: Path,
    patch: Path,
    worktree: Path,
    build: Path,
    smoke_output: Path,
) -> list[list[str]]:
    executable = build / "bin" / "test-llama-archs"
    trace = (smoke_output / "runtime_kv_trace.jsonl").resolve()
    summary = (smoke_output / "runtime_kv_trace_summary.json").resolve()
    harness_trace = (smoke_output / "harness_trace.jsonl").resolve()
    stdout = (smoke_output / "stdout.log").resolve()
    stderr = (smoke_output / "stderr.log").resolve()

    return [
        [
            "git",
            "-C",
            str(checkout),
            "worktree",
            "add",
            "--detach",
            str(worktree),
            PINNED_LLAMA_COMMIT,
        ],
        ["git", "-C", str(worktree), "apply", str(patch.resolve())],
        [
            "cmake",
            "-S",
            str(worktree),
            "-B",
            str(build),
            "-DLLAMA_BUILD_TESTS=ON",
            "-DGGML_NATIVE=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        [
            "cmake",
            "--build",
            str(build),
            "--target",
            "test-llama-archs",
            "--parallel",
            "2",
        ],
        [
            sys.executable,
            "-m",
            "integrations.llama_cpp.evict_smoke",
            "--executable",
            str(executable),
            "--trace",
            str(trace),
            "--summary",
            str(summary),
            "--harness-trace",
            str(harness_trace),
            "--stdout",
            str(stdout),
            "--stderr",
            str(stderr),
        ],
    ]


def run_llama_check(
    checkout: str | Path = "external/llama.cpp",
    patch: str | Path = "integrations/llama_cpp/patches/solidattention_kv_trace.patch",
    output_dir: str | Path = "outputs/checks/llama/llama-smoke",
) -> None:
    checkout_path = Path(checkout)
    patch_path = Path(patch)
    smoke_output = Path(output_dir)
    smoke_output.mkdir(parents=True, exist_ok=True)

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

    with tempfile.TemporaryDirectory(prefix="solidattention-llama-") as temporary:
        worktree = Path(temporary) / "worktree"
        build = Path(temporary) / "build"
        try:
            commands = llama_build_commands(
                checkout=checkout_path,
                patch=patch_path,
                worktree=worktree,
                build=build,
                smoke_output=smoke_output,
            )
            for command in commands:
                subprocess.run(command, check=True)
        finally:
            if worktree.exists():
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(checkout_path),
                        "worktree",
                        "remove",
                        "--force",
                        str(worktree),
                    ],
                    check=False,
                )


def main() -> int:
    run_llama_check()
    print(f"llama.cpp smoke passed at {PINNED_LLAMA_COMMIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
