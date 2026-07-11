import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


def test_makefile_declares_explicit_backend_mode() -> None:
    makefile = Path("io/Makefile").read_text(encoding="utf-8")
    compile_command = (
        "$(CC) $(CFLAGS) $(BACKEND_CFLAGS) -o $@ $< "
        "$(LDFLAGS) $(BACKEND_LDFLAGS) $(LDLIBS) $(BACKEND_LDLIBS)"
    )

    assert "BACKEND_MODE ?= auto" in makefile
    assert "ifneq ($(words $(BACKEND_MODE)),1)" in makefile
    assert "ifeq ($(filter auto posix liburing,$(BACKEND_MODE)),)" in makefile
    assert "BACKEND_MODE=liburing requested but liburing is unavailable" in makefile
    assert "BACKEND_CFLAGS := -DHAVE_LIBURING $(LIBURING_CFLAGS)" in makefile
    assert "BACKEND_LDFLAGS := $(LIBURING_LDFLAGS)" in makefile
    assert "BACKEND_LDLIBS := -luring" in makefile
    assert "CFLAGS += -DHAVE_LIBURING" not in makefile
    assert compile_command in makefile
    assert makefile.count(compile_command) == 2
    assert "BACKEND_STAMP := ../build/io/backend-$(BACKEND)" in makefile
    assert "block_io_probe: block_io_probe.c $(BACKEND_STAMP)" in makefile
    assert "runtime_kv_command_consumer: runtime_kv_command_consumer.c $(BACKEND_STAMP)" in makefile
    assert "$(BACKEND_STAMP):" in makefile
    assert "rm -f $(TARGETS) ../build/io/backend-*" in makefile
    assert 'mktemp "$${TMPDIR:-/tmp}/solid_attention_liburing_check.XXXXXX"' in makefile
    assert "trap 'rm -f \"$$tmp\"' 0 1 2 15" in makefile
    assert '-o "$$tmp"' in makefile
    assert "-o /tmp/solid_attention_liburing_check" not in makefile
    assert "/data/disk2/ljc" not in makefile


def expected_backend() -> str:
    value = os.environ.get("SOLIDATTENTION_EXPECTED_IO_BACKEND", "")
    if value not in {"posix_fallback", "liburing"}:
        pytest.skip("run through tools.check io or tools.check liburing")
    return value


def backend_mode() -> str:
    return "liburing" if expected_backend() == "liburing" else "posix"


@pytest.mark.io
@pytest.mark.liburing
def test_phase2_make_backend_reports_supported_mode() -> None:
    if shutil.which("make") is None:
        pytest.skip("make is not available")
    result = subprocess.run(
        ["make", "--no-print-directory", "-C", "io", "backend", f"BACKEND_MODE={backend_mode()}"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == expected_backend()


@pytest.mark.io
@pytest.mark.liburing
def test_phase2_block_io_probe_builds_and_verifies_blocks(tmp_path) -> None:
    if shutil.which("make") is None or shutil.which("cc") is None:
        pytest.skip("C build toolchain is not available")

    subprocess.run(
        ["make", "-C", "io", "clean", "all", f"BACKEND_MODE={backend_mode()}"],
        check=True,
        capture_output=True,
        text=True,
    )

    probe_path = tmp_path / "probe.bin"
    result = subprocess.run(
        ["io/block_io_probe", str(probe_path), "1024", "3"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["backend"] == expected_backend()
    assert payload["block_size"] == 1024
    assert payload["block_count"] == 3
    assert payload["bytes"] == 3072
    assert payload["write_latency_ns"] >= 0
    assert payload["read_latency_ns"] >= 0
    assert probe_path.exists()
