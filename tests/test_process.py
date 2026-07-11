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
