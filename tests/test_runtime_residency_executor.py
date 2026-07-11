import json

import pytest

from harness.runtime_residency_executor import execute_runtime_residency_plan


def make_op(
    op,
    *,
    source,
    target,
    block_id=0,
    offset=0,
    start_ms=0.0,
    end_ms=1.0,
):
    return {
        "op": op,
        "layer_id": 0,
        "block_id": block_id,
        "start_token": block_id * 16,
        "end_token": block_id * 16 + 16,
        "source": source,
        "target": target,
        "ssd_offset": offset,
        "size_bytes": 32,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_ms": end_ms - start_ms,
        "blocking": op != "kv_compute",
        "metadata": {"scheduler_op": f"schedule_{op}"},
    }


def write_plan(path, ops):
    plan = {
        "layout": {
            "base_offset": 0,
            "alignment": 32,
            "total_size_bytes": 64,
            "blocks": [
                {
                    "layer_id": 0,
                    "block_id": 0,
                    "offset": 0,
                    "size_bytes": 32,
                    "end_offset": 32,
                }
            ],
        },
        "ops": ops,
    }
    path.write_text(json.dumps(plan), encoding="utf-8")


def write_backing_file(path):
    path.write_bytes(bytes(range(64)))


def test_runtime_residency_executor_runs_full_lifecycle(tmp_path) -> None:
    plan_path = tmp_path / "plan.json"
    backing_file = tmp_path / "kv_backing.bin"
    output_dir = tmp_path / "out"
    write_backing_file(backing_file)
    write_plan(
        plan_path,
        [
            make_op("kv_load", source="ssd", target="dram", start_ms=0.0, end_ms=1.0),
            make_op("kv_load", source="dram", target="vram", start_ms=1.0, end_ms=1.2),
            make_op("kv_compute", source="vram", target="vram", start_ms=1.2, end_ms=1.3),
            make_op("kv_evict", source="vram", target="ssd", start_ms=1.3, end_ms=1.9),
        ],
    )

    result = execute_runtime_residency_plan(plan_path, output_dir, backing_file=backing_file)

    assert result.movement_op_count == 4
    assert result.ssd_read_count == 1
    assert result.dram_to_vram_copy_count == 1
    assert result.compute_count == 1
    assert result.evict_count == 1
    assert result.ssd_write_count == 1
    assert result.ssd_read_bytes == 32
    assert result.dram_to_vram_bytes == 32
    assert result.compute_bytes == 32
    assert result.ssd_write_bytes == 32
    assert result.lazy_source_seed_count == 0
    assert result.verification_error_count == 0
    assert result.final_residency["dram"] == {"block_count": 1, "size_bytes": 32}
    assert result.final_residency["vram"] == {"block_count": 0, "size_bytes": 0}

    metrics = json.loads((output_dir / "runtime_residency_metrics.json").read_text(encoding="utf-8"))
    trace = json.loads((output_dir / "runtime_residency_trace.json").read_text(encoding="utf-8"))
    assert metrics["verification_error_count"] == 0
    assert [event["op"] for event in trace] == [
        "residency_kv_load",
        "residency_kv_load",
        "residency_kv_compute",
        "residency_kv_evict",
    ]
    assert (output_dir / "runtime_residency_summary.json").exists()
    manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema"] == "solidattention.run_manifest.v1"
    assert manifest["status"] == "passed"
    assert manifest["profile"] == "runtime-residency-executor"
    assert manifest["inputs"][0]["sha256"]


def test_runtime_residency_executor_can_seed_initial_resident_dram(tmp_path) -> None:
    plan_path = tmp_path / "plan.json"
    output_dir = tmp_path / "out"
    write_plan(
        plan_path,
        [
            make_op("kv_load", source="dram", target="vram", start_ms=0.0, end_ms=0.2),
            make_op("kv_compute", source="vram", target="vram", start_ms=0.2, end_ms=0.3),
        ],
    )

    result = execute_runtime_residency_plan(plan_path, output_dir)

    assert result.lazy_source_seed_count == 1
    assert result.lazy_source_seed_bytes == 32
    assert result.dram_to_vram_copy_count == 1
    assert result.compute_count == 1
    assert result.missing_source_count == 0
    assert result.verification_error_count == 0
    assert result.final_residency["dram"] == {"block_count": 1, "size_bytes": 32}
    assert result.final_residency["vram"] == {"block_count": 1, "size_bytes": 32}


def test_runtime_residency_executor_strict_mode_reports_missing_source(tmp_path) -> None:
    plan_path = tmp_path / "plan.json"
    output_dir = tmp_path / "out"
    write_plan(
        plan_path,
        [
            make_op("kv_load", source="dram", target="vram", start_ms=0.0, end_ms=0.2),
            make_op("kv_compute", source="vram", target="vram", start_ms=0.2, end_ms=0.3),
        ],
    )

    result = execute_runtime_residency_plan(plan_path, output_dir, seed_missing_sources=False)

    assert result.lazy_source_seed_count == 0
    assert result.missing_source_count == 2
    assert result.verification_error_count == 2
    assert result.final_residency["dram"] == {"block_count": 0, "size_bytes": 0}
    assert result.final_residency["vram"] == {"block_count": 0, "size_bytes": 0}
    manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert manifest["failure"] == {
        "type": "VerificationError",
        "message": "runtime residency verification errors: 2",
    }


def test_runtime_residency_executor_rejects_missing_ops(tmp_path) -> None:
    plan_path = tmp_path / "bad_plan.json"
    plan_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError):
        execute_runtime_residency_plan(plan_path, tmp_path / "out")
