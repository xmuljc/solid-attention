"""Probe helpers for a future local llama.cpp checkout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


EXPECTED_ANY_OF = (
    ("src/llama.cpp", "llama.cpp"),
    ("include/llama.h", "llama.h"),
)
EXPECTED_REQUIRED = ("CMakeLists.txt",)

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
