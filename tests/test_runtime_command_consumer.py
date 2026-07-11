import json
import os
import shutil
import subprocess

import pytest

from harness.runtime_command_consumer import RuntimeCommandConsumerResult, run_runtime_command_consumer


def expected_backend() -> str:
    value = os.environ.get("SOLIDATTENTION_EXPECTED_IO_BACKEND", "")
    if value not in {"posix_fallback", "liburing"}:
        pytest.skip("run through tools.check io or tools.check liburing")
    return value


def backend_mode() -> str:
    return "liburing" if expected_backend() == "liburing" else "posix"


def write_commands(path) -> None:
    commands = [
        {
            "op": "kv_load",
            "layer_id": 0,
            "block_id": 0,
            "start_token": 0,
            "end_token": 64,
            "token_count": 64,
            "size_bytes": 4096,
            "location": "dram",
            "source": "ssd",
            "target": "dram",
            "ssd_offset": 0,
            "blocking": True,
            "metadata": {"runtime_contract": "solidattention.llama_cpp.kv_command.v1"},
        },
        {
            "op": "kv_load",
            "layer_id": 0,
            "block_id": 0,
            "start_token": 0,
            "end_token": 64,
            "token_count": 64,
            "size_bytes": 4096,
            "location": "vram",
            "source": "dram",
            "target": "vram",
            "ssd_offset": 0,
            "blocking": True,
            "metadata": {"runtime_contract": "solidattention.llama_cpp.kv_command.v1"},
        },
        {
            "op": "kv_compute",
            "layer_id": 0,
            "block_id": 0,
            "start_token": 0,
            "end_token": 64,
            "token_count": 64,
            "size_bytes": 4096,
            "location": "vram",
            "source": "vram",
            "target": "vram",
            "ssd_offset": 0,
            "blocking": False,
            "metadata": {"runtime_contract": "solidattention.llama_cpp.kv_command.v1"},
        },
        {
            "op": "kv_evict",
            "layer_id": 0,
            "block_id": 0,
            "start_token": 0,
            "end_token": 64,
            "token_count": 64,
            "size_bytes": 4096,
            "location": "ssd",
            "source": "vram",
            "target": "ssd",
            "ssd_offset": 0,
            "blocking": True,
            "metadata": {"runtime_contract": "solidattention.llama_cpp.kv_command.v1"},
        },
    ]
    path.write_text("".join(json.dumps(command, sort_keys=True) + "\n" for command in commands), encoding="utf-8")


def test_runtime_command_consumer_result_from_dict() -> None:
    result = RuntimeCommandConsumerResult.from_dict(
        {
            "backend": "fake",
            "command_count": 1,
            "kv_load_count": 1,
            "kv_prefetch_count": 0,
            "kv_compute_count": 0,
            "kv_evict_count": 0,
            "ssd_read_count": 1,
            "ssd_write_count": 0,
            "dram_to_vram_copy_count": 0,
            "vram_compute_count": 0,
            "lazy_init_count": 0,
            "ssd_seed_count": 1,
            "ssd_read_bytes": 4096,
            "ssd_write_bytes": 0,
            "dram_to_vram_bytes": 0,
            "lazy_init_bytes": 0,
            "ssd_seed_bytes": 4096,
            "verification_errors": 0,
            "ssd_read_latency_ns": 1000000,
            "ssd_write_latency_ns": 2000000,
            "copy_latency_ns": 0,
            "compute_verify_latency_ns": 0,
        }
    )

    assert result.ssd_read_latency_ms == 1.0
    assert result.ssd_write_latency_ms == 2.0
    assert result.to_dict()["backend"] == "fake"


@pytest.mark.io
@pytest.mark.liburing
def test_runtime_command_consumer_builds_and_executes_commands(tmp_path) -> None:
    if shutil.which("make") is None or shutil.which("cc") is None:
        pytest.skip("C build toolchain is not available")

    subprocess.run(
        ["make", "-C", "io", "runtime_kv_command_consumer", f"BACKEND_MODE={backend_mode()}"],
        check=True,
        capture_output=True,
        text=True,
    )
    command_path = tmp_path / "commands.jsonl"
    backing_path = tmp_path / "kv_backing.bin"
    write_commands(command_path)

    result = run_runtime_command_consumer("io/runtime_kv_command_consumer", command_path, backing_path)

    assert result.backend == expected_backend()
    assert result.command_count == 4
    assert result.kv_load_count == 2
    assert result.kv_compute_count == 1
    assert result.kv_evict_count == 1
    assert result.ssd_read_count == 1
    assert result.ssd_write_count == 1
    assert result.ssd_read_bytes == 4096
    assert result.ssd_write_bytes == 4096
    assert result.dram_to_vram_copy_count == 1
    assert result.vram_compute_count == 1
    assert result.verification_errors == 0
    assert backing_path.exists()
    assert backing_path.stat().st_size == 4096
