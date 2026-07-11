import json

import pytest

from integrations.llama_cpp.runtime_commands import commands_from_movement_plan, export_runtime_commands


def make_plan() -> dict:
    return {
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
                "start_token": 0,
                "end_token": 64,
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
                "start_token": 0,
                "end_token": 64,
                "start_ms": 1.0,
                "end_ms": 1.25,
                "duration_ms": 0.25,
                "blocking": True,
                "metadata": {"scheduler_op": "schedule_dram_to_vram"},
            },
            {
                "op": "kv_compute",
                "layer_id": 0,
                "block_id": 0,
                "source": "vram",
                "target": "vram",
                "ssd_offset": 0,
                "size_bytes": 4096,
                "start_token": 0,
                "end_token": 64,
                "start_ms": 1.25,
                "end_ms": 1.35,
                "duration_ms": 0.1,
                "blocking": False,
                "metadata": {"scheduler_op": "schedule_compute"},
            },
            {
                "op": "kv_evict",
                "layer_id": 0,
                "block_id": 1,
                "source": "vram",
                "target": "ssd",
                "ssd_offset": 4096,
                "size_bytes": 4096,
                "start_token": 64,
                "end_token": 128,
                "start_ms": 1.35,
                "end_ms": 1.75,
                "duration_ms": 0.4,
                "blocking": True,
                "metadata": {"scheduler_op": "evict_to_ssd"},
            },
        ],
    }


def test_commands_from_movement_plan_exports_runtime_schema() -> None:
    commands = commands_from_movement_plan(make_plan())

    assert [command.op for command in commands] == ["kv_load", "kv_load", "kv_compute", "kv_evict"]
    first = commands[0].to_dict()
    assert first["location"] == "dram"
    assert first["start_token"] == 0
    assert first["end_token"] == 64
    assert first["token_count"] == 64
    assert first["metadata"]["runtime_contract"] == "solidattention.llama_cpp.kv_command.v1"


def test_export_runtime_commands_writes_jsonl_and_summary(tmp_path) -> None:
    plan_path = tmp_path / "movement_plan.json"
    jsonl_path = tmp_path / "commands.jsonl"
    summary_path = tmp_path / "summary.json"
    plan_path.write_text(json.dumps(make_plan()))

    result = export_runtime_commands(plan_path, jsonl_path=jsonl_path, summary_path=summary_path)

    assert result.command_count == 4
    assert result.command_counts_by_op == {"kv_compute": 1, "kv_evict": 1, "kv_load": 2}
    assert result.ssd_read_command_count == 1
    assert result.ssd_write_command_count == 1
    assert result.ssd_read_bytes == 4096
    assert result.ssd_write_bytes == 4096
    records = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    assert records[0]["source"] == "ssd"
    assert records[-1]["op"] == "kv_evict"
    assert json.loads(summary_path.read_text())["command_count"] == 4


def test_commands_from_movement_plan_rejects_missing_runtime_fields() -> None:
    plan = make_plan()
    del plan["ops"][0]["start_token"]

    with pytest.raises(ValueError, match="missing runtime command fields"):
        commands_from_movement_plan(plan)


def test_commands_from_movement_plan_rejects_layout_mismatch() -> None:
    plan = make_plan()
    plan["ops"][0]["ssd_offset"] = 128

    with pytest.raises(ValueError, match="SSD offset"):
        commands_from_movement_plan(plan)
