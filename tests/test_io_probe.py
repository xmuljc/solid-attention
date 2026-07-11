import json

import pytest

from harness.io_probe import BlockIOProbeResult, run_block_io_probe


def make_fake_probe(tmp_path, *, read_latency_ns=4_000_000, write_latency_ns=8_000_000):
    script = tmp_path / "fake_probe.py"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "path = pathlib.Path(sys.argv[1])\n"
        "block_size = int(sys.argv[2])\n"
        "block_count = int(sys.argv[3])\n"
        "path.parent.mkdir(parents=True, exist_ok=True)\n"
        "path.write_bytes(b'0' * (block_size * block_count))\n"
        f"print(json.dumps({{'backend': 'fake', 'path': str(path), 'block_size': block_size, 'block_count': block_count, 'bytes': block_size * block_count, 'write_latency_ns': {write_latency_ns}, 'read_latency_ns': {read_latency_ns}}}))\n"
    )
    script.chmod(0o755)
    return script


def test_block_io_probe_result_computes_latency_rates() -> None:
    result = BlockIOProbeResult.from_dict(
        {
            "backend": "fake",
            "path": "probe.bin",
            "block_size": 1024,
            "block_count": 4,
            "bytes": 4096,
            "write_latency_ns": 8_000_000,
            "read_latency_ns": 4_000_000,
        }
    )

    assert result.write_latency_ms == 8.0
    assert result.read_latency_ms == 4.0
    assert result.write_latency_ms_per_block == 2.0
    assert result.read_latency_ms_per_block == 1.0
    assert result.to_dict()["backend"] == "fake"


def test_run_block_io_probe_parses_json_stdout(tmp_path) -> None:
    executable = make_fake_probe(tmp_path, read_latency_ns=6_000_000, write_latency_ns=9_000_000)
    probe_path = tmp_path / "probe.bin"

    result = run_block_io_probe(executable, probe_path, block_size=2048, block_count=3)

    assert result.backend == "fake"
    assert result.block_size == 2048
    assert result.block_count == 3
    assert result.bytes == 6144
    assert result.read_latency_ms_per_block == 2.0
    assert result.write_latency_ms_per_block == 3.0
    assert probe_path.exists()


def test_run_block_io_probe_rejects_invalid_sizes(tmp_path) -> None:
    executable = make_fake_probe(tmp_path)

    with pytest.raises(ValueError):
        run_block_io_probe(executable, tmp_path / "probe.bin", block_size=0, block_count=1)
    with pytest.raises(ValueError):
        run_block_io_probe(executable, tmp_path / "probe.bin", block_size=1, block_count=0)
