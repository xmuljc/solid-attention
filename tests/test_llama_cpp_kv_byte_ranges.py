import json

import pytest

from integrations.llama_cpp.kv_byte_ranges import verify_kv_byte_ranges


def make_plan(path, *, size_bytes=64, start_token=0, end_token=4, offset=0, block_id=0):
    plan = {
        "layout": {
            "base_offset": 0,
            "alignment": 16,
            "total_size_bytes": size_bytes,
            "blocks": [
                {
                    "layer_id": 0,
                    "block_id": block_id,
                    "offset": offset,
                    "size_bytes": size_bytes,
                    "end_offset": offset + size_bytes,
                }
            ],
        },
        "ops": [
            {
                "op": "kv_load",
                "layer_id": 0,
                "block_id": block_id,
                "source": "ssd",
                "target": "dram",
                "ssd_offset": offset,
                "size_bytes": size_bytes,
                "start_token": start_token,
                "end_token": end_token,
                "start_ms": 0.0,
                "end_ms": 1.0,
                "duration_ms": 1.0,
                "blocking": True,
                "metadata": {},
            },
            {
                "op": "kv_compute",
                "layer_id": 0,
                "block_id": block_id,
                "source": "vram",
                "target": "vram",
                "ssd_offset": offset,
                "size_bytes": size_bytes,
                "start_token": start_token,
                "end_token": end_token,
                "start_ms": 1.0,
                "end_ms": 1.1,
                "duration_ms": 0.1,
                "blocking": False,
                "metadata": {},
            },
        ],
    }
    path.write_text(json.dumps(plan), encoding="utf-8")


def test_verify_kv_byte_ranges_infers_equal_rows_and_saves_outputs(tmp_path) -> None:
    plan_path = tmp_path / "plan.json"
    output_dir = tmp_path / "out"
    backing_file = tmp_path / "kv.bin"
    runtime_summary = tmp_path / "runtime_summary.json"
    make_plan(plan_path)
    backing_file.write_bytes(bytes(range(64)))
    runtime_summary.write_text(json.dumps({"cache_size": 8}), encoding="utf-8")

    result = verify_kv_byte_ranges(
        plan_path,
        output_dir,
        runtime_summary_path=runtime_summary,
        backing_file=backing_file,
    )

    assert result.verification_error_count == 0
    assert result.block_count == 1
    assert result.range_count == 8
    assert result.mapped_bytes == 64
    assert result.key_bytes == 32
    assert result.value_bytes == 32
    assert result.inferred_row_size_count == 1
    mapping = result.mappings[0]
    assert mapping.key_row_size_bytes == 8
    assert mapping.value_row_size_bytes == 8
    assert mapping.ranges[0].kind == "k"
    assert mapping.ranges[0].staging_offset == 0
    assert mapping.ranges[0].tensor_offset == 0
    assert mapping.ranges[1].kind == "v"
    assert mapping.ranges[1].staging_offset == 8
    assert mapping.ranges[1].tensor_offset == 0
    assert mapping.block_sha256 is not None
    assert mapping.key_sha256 is not None
    assert mapping.value_sha256 is not None

    metrics = json.loads((output_dir / "kv_byte_range_metrics.json").read_text(encoding="utf-8"))
    trace = json.loads((output_dir / "kv_byte_range_trace.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "kv_byte_range_summary.json").read_text(encoding="utf-8"))
    assert metrics["contract"] == "solidattention.llama_cpp.kv_byte_ranges.v1"
    assert metrics["verification_error_count"] == 0
    assert trace[0]["op"] == "kv_byte_range_map"
    assert summary["mappings"][0]["range_count"] == 8


def test_verify_kv_byte_ranges_maps_tensor_offsets_by_block_cell(tmp_path) -> None:
    plan_path = tmp_path / "plan.json"
    output_dir = tmp_path / "out"
    make_plan(plan_path, block_id=8, start_token=100, end_token=104)

    result = verify_kv_byte_ranges(
        plan_path,
        output_dir,
        key_row_size_bytes=8,
        value_row_size_bytes=8,
        cache_size=16,
    )

    assert result.verification_error_count == 0
    mapping = result.mappings[0]
    assert mapping.block_id == 8
    assert mapping.ranges[0].token_id == 100
    assert [item.tensor_offset for item in mapping.ranges[:4]] == [64, 64, 72, 72]


def test_verify_kv_byte_ranges_accepts_explicit_asymmetric_rows(tmp_path) -> None:
    plan_path = tmp_path / "plan.json"
    output_dir = tmp_path / "out"
    make_plan(plan_path, size_bytes=80)

    result = verify_kv_byte_ranges(
        plan_path,
        output_dir,
        key_row_size_bytes=12,
        value_row_size_bytes=8,
        cache_size=4,
    )

    assert result.verification_error_count == 0
    assert result.key_bytes == 48
    assert result.value_bytes == 32
    assert result.mappings[0].ranges[-1].tensor_end_offset == 32


def test_verify_kv_byte_ranges_reports_tensor_bounds_errors(tmp_path) -> None:
    plan_path = tmp_path / "plan.json"
    output_dir = tmp_path / "out"
    make_plan(plan_path, block_id=6, start_token=100, end_token=104)

    result = verify_kv_byte_ranges(plan_path, output_dir, cache_size=8)

    assert result.tensor_bounds_error_count == 4
    assert result.verification_error_count == 4


def test_verify_kv_byte_ranges_rejects_missing_ops(tmp_path) -> None:
    plan_path = tmp_path / "bad_plan.json"
    plan_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="missing ops"):
        verify_kv_byte_ranges(plan_path, tmp_path / "out")


def test_verify_kv_byte_ranges_rejects_v_trans_for_now(tmp_path) -> None:
    plan_path = tmp_path / "plan.json"
    make_plan(plan_path)

    with pytest.raises(ValueError, match="v_trans"):
        verify_kv_byte_ranges(plan_path, tmp_path / "out", v_trans=True)
