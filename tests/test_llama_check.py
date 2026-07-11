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
