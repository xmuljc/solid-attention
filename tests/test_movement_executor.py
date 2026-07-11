import json

import pytest

from harness.io_probe import BlockIOProbeResult
from harness.movement_executor import execute_movement_plan


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


def make_plan(path):
    plan = {
        "layout": {
            "base_offset": 0,
            "alignment": 4096,
            "total_size_bytes": 8192,
            "blocks": [
                {"layer_id": 0, "block_id": 0, "offset": 0, "size_bytes": 4096, "end_offset": 4096},
                {"layer_id": 0, "block_id": 1, "offset": 4096, "size_bytes": 4096, "end_offset": 8192},
            ],
        },
        "ops": [
            {
                "op": "kv_load",
                "layer_id": 0,
                "block_id": 0,
                "source": "ssd",
                "target": "dram",
                "ssd_offset": 0,
                "size_bytes": 4096,
                "start_ms": 0.0,
                "end_ms": 1.0,
                "duration_ms": 1.0,
                "blocking": True,
                "metadata": {"scheduler_op": "schedule_ssd_read"},
            },
            {
                "op": "kv_load",
                "layer_id": 0,
                "block_id": 0,
                "source": "dram",
                "target": "vram",
                "ssd_offset": 0,
                "size_bytes": 4096,
                "start_ms": 1.0,
                "end_ms": 1.25,
                "duration_ms": 0.25,
                "blocking": True,
                "metadata": {"scheduler_op": "schedule_dram_to_vram"},
            },
            {
                "op": "kv_load",
                "layer_id": 0,
                "block_id": 1,
                "source": "ssd",
                "target": "dram",
                "ssd_offset": 4096,
                "size_bytes": 4096,
                "start_ms": 1.25,
                "end_ms": 2.25,
                "duration_ms": 1.0,
                "blocking": True,
                "metadata": {"scheduler_op": "schedule_ssd_read"},
            },
            {
                "op": "kv_compute",
                "layer_id": 0,
                "block_id": 1,
                "source": "vram",
                "target": "vram",
                "ssd_offset": 4096,
                "size_bytes": 4096,
                "start_ms": 2.25,
                "end_ms": 2.35,
                "duration_ms": 0.1,
                "blocking": False,
                "metadata": {"scheduler_op": "schedule_compute"},
            },
        ],
    }
    path.write_text(json.dumps(plan))


def test_execute_movement_plan_runs_io_probe_for_unique_ssd_reads(tmp_path) -> None:
    plan_path = tmp_path / "movement_plan.json"
    output_dir = tmp_path / "exec"
    make_plan(plan_path)
    fake_probe = make_fake_probe(tmp_path, read_latency_ns=6_000_000, write_latency_ns=9_000_000)

    result = execute_movement_plan(plan_path, output_dir, io_probe_executable=fake_probe)

    assert result.movement_op_count == 4
    assert result.kv_load_op_count == 3
    assert result.ssd_load_op_count == 2
    assert result.unique_ssd_read_count == 2
    assert result.ssd_read_bytes == 8192
    assert len(result.io_probe_results) == 1
    assert result.io_probe_results[0]["backend"] == "fake"
    assert result.io_probe_results[0]["block_size"] == 4096
    assert result.io_probe_results[0]["block_count"] == 2

    metrics = json.loads((output_dir / "movement_execution_metrics.json").read_text())
    assert metrics["io_probe_total_read_latency_ms"] == 6.0
    trace = json.loads((output_dir / "movement_execution_trace.json").read_text())
    assert [event["op"] for event in trace][:2] == ["execute_kv_load", "execute_kv_load"]
    assert trace[0]["metadata"]["ssd_offset"] == 0
    assert (output_dir / "movement_execution_summary.json").exists()
    manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema"] == "solidattention.run_manifest.v1"
    assert manifest["status"] == "passed"
    assert manifest["profile"] == "movement-executor"
    assert manifest["inputs"][0]["sha256"]
    assert manifest["io_backend"] == "fake"
    assert [item["kind"] for item in manifest["outputs"]].count("io-probe") == 1


def test_execute_movement_plan_retains_probes_when_backends_are_mixed(
    tmp_path, monkeypatch
) -> None:
    plan_path = tmp_path / "movement_plan.json"
    output_dir = tmp_path / "exec"
    make_plan(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["ops"][2]["size_bytes"] = 8192
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    fake_executable = tmp_path / "fake_probe.py"
    fake_executable.write_text("# input only\n", encoding="utf-8")

    def run_mixed_probe(
        executable, probe_path, *, block_size, block_count
    ) -> BlockIOProbeResult:
        probe_path.write_bytes(bytes(block_size * block_count))
        backend = "fake" if block_size == 4096 else "posix_fallback"
        return BlockIOProbeResult(
            backend=backend,
            path=str(probe_path),
            block_size=block_size,
            block_count=block_count,
            bytes=block_size * block_count,
            write_latency_ns=1,
            read_latency_ns=1,
        )

    monkeypatch.setattr(
        "harness.movement_executor.run_block_io_probe", run_mixed_probe
    )

    with pytest.raises(RuntimeError, match="mixed backends"):
        execute_movement_plan(
            plan_path, output_dir, io_probe_executable=fake_executable
        )

    manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert manifest["failure"]["type"] == "RuntimeError"
    assert "mixed backends: fake, posix_fallback" in manifest["failure"]["message"]
    assert manifest["io_backend"] == "mock"
    assert [item["kind"] for item in manifest["outputs"]] == [
        "io-probe",
        "io-probe",
    ]
    assert {item["path"] for item in manifest["outputs"]} == {
        "movement_io_probe_4096.bin",
        "movement_io_probe_8192.bin",
    }
    assert all(item["sha256"] for item in manifest["outputs"])


def test_execute_movement_plan_can_skip_io(tmp_path) -> None:
    plan_path = tmp_path / "movement_plan.json"
    output_dir = tmp_path / "exec"
    make_plan(plan_path)

    result = execute_movement_plan(plan_path, output_dir, execute_io=False)

    assert result.io_probe_results == ()
    assert result.unique_ssd_read_count == 2
    assert json.loads((output_dir / "movement_execution_metrics.json").read_text())["io_probe_results"] == []


def test_execute_movement_plan_without_ssd_reads_does_not_require_probe(
    tmp_path,
) -> None:
    plan_path = tmp_path / "movement_plan.json"
    output_dir = tmp_path / "exec"
    plan_path.write_text('{"layout": {}, "ops": []}', encoding="utf-8")

    result = execute_movement_plan(
        plan_path,
        output_dir,
        io_probe_executable=tmp_path / "missing_probe",
    )

    manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert result.unique_ssd_read_count == 0
    assert manifest["status"] == "passed"
    assert manifest["io_backend"] == "mock"
    assert [item["path"] for item in manifest["inputs"]] == [str(plan_path)]


def test_execute_movement_plan_rejects_missing_ops(tmp_path) -> None:
    plan_path = tmp_path / "bad_plan.json"
    plan_path.write_text("{}")

    with pytest.raises(ValueError):
        execute_movement_plan(plan_path, tmp_path / "exec", execute_io=False)
