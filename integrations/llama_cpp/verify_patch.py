"""Verify SolidAttention llama.cpp integration patches without applying them."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PatchVerification:
    checkout: str
    patch: str
    status: str
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.status in {"applies", "already_applied"}

    def to_dict(self) -> dict[str, object]:
        return {
            "checkout": self.checkout,
            "patch": self.patch,
            "status": self.status,
            "ok": self.ok,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


def _run_git_apply(checkout: Path, patch: Path, reverse: bool = False) -> subprocess.CompletedProcess[str]:
    args = ["git", "-C", str(checkout), "apply", "--check"]
    if reverse:
        args.append("--reverse")
    args.append(str(patch))
    return subprocess.run(args, check=False, capture_output=True, text=True)


def verify_patch(checkout: str | Path, patch: str | Path) -> PatchVerification:
    checkout_path = Path(checkout)
    patch_path = Path(patch)

    if shutil.which("git") is None:
        return PatchVerification(str(checkout_path), str(patch_path), "missing_git", "", "git not found")
    if not checkout_path.exists():
        return PatchVerification(str(checkout_path), str(patch_path), "missing_checkout", "", "checkout not found")
    if not patch_path.exists():
        return PatchVerification(str(checkout_path), str(patch_path), "missing_patch", "", "patch not found")

    patch_for_git = patch_path.resolve()

    apply_check = _run_git_apply(checkout_path, patch_for_git)
    if apply_check.returncode == 0:
        return PatchVerification(str(checkout_path), str(patch_path), "applies", apply_check.stdout, apply_check.stderr)

    reverse_check = _run_git_apply(checkout_path, patch_for_git, reverse=True)
    if reverse_check.returncode == 0:
        return PatchVerification(
            str(checkout_path),
            str(patch_path),
            "already_applied",
            reverse_check.stdout,
            reverse_check.stderr,
        )

    return PatchVerification(str(checkout_path), str(patch_path), "conflict", apply_check.stdout, apply_check.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", default="external/llama.cpp")
    parser.add_argument("--patch", default="integrations/llama_cpp/patches/solidattention_kv_trace.patch")
    args = parser.parse_args()

    result = verify_patch(args.checkout, args.patch)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
