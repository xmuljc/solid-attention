"""Cross-platform subprocess command construction."""

from __future__ import annotations

import sys
from pathlib import Path


def build_executable_command(executable: str | Path, *args: object) -> list[str]:
    path = Path(executable)
    command = [sys.executable, str(path)] if path.suffix.lower() == ".py" else [str(path)]
    command.extend(str(arg) for arg in args)
    return command
