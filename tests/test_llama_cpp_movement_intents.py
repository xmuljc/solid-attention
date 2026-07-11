import json

import pytest

from harness.movement_executor import execute_movement_plan
from integrations.llama_cpp.movement_intents import export_movement_intent_plan


def make_event(
    *,
    command_op="kv_load",
    source="ssd",
    target="dram",
    intent_kind="ssd_read",
    layer_id=0,
    block_id=1,
    ssd_offset=4096,
    size_bytes=4096,
    start_token=64,
    end_token=128,
    movement_intent=True,
    blocking=True,
):
    return {
        "op": "kv_command_bridge",
        "layer_id": layer_id,
        "block_id": block_id,
        "start_token": start_token,
        "end_token": end_token,
        "metadata": {
            "command_op": command_op,
            "command_source": source,
            "command_target": target,
            "command_size_bytes": size_bytes,
            "command_start_token": start_token,
            "command_end_token": end_token,
            "ssd_offset": ssd_offset,
            "movement_intent": movement_intent,
            "intent_kind": intent_kind,
            "blocking": blocking,
            "shadow_before": source,
            "shadow_after": target,
            "shadow_source_match": True,
        },
    }


def write_trace(path, events):
    path.write_text("".join(json.dumps(event) + "\n" for event in events))


def test_export_movement_intent_plan_writes_movement_plan_and_summary(tmp_path) -> None:
    trace_path = tmp_path / "bridge.jsonl"
    plan_path = tmp_path / "intent_plan.json"
    summary_path = tmp_path / "intent_summary.json"
    write_trace(
        trace_path,
        [
            make_event(),
            make_event(command_op="kv_load", source="dram", target="vram", intent_kind="dram_to_vram"),
            make_event(command_op="kv_compute", source="vram", target="vram", intent_kind="compute", blocking=False),
            make_event(movement_intent=False),
        ],
    )

    result = export_movement_intent_plan(trace_path, plan_path=plan_path, summary_path=summary_path)

    assert result.movement_intent_count == 3
    assert result.skipped_non_intent_count == 1
    assert result.command_counts_by_op == {"kv_compute": 1, "kv_load": 2}
    assert result.intent_counts_by_kind == {"compute": 1, "dram_to_vram": 1, "ssd_read": 1}
    assert result.blocking_movement_intent_count == 2
    assert result.nonblocking_movement_intent_count == 1
    assert result.ssd_read_intent_count == 1
    assert result.ssd_read_bytes == 4096

    plan = json.loads(plan_path.read_text())
    assert plan["contract"] == "solidattention.llama_cpp.movement_intent_plan.v1"
    assert len(plan["layout"]["blocks"]) == 1
    assert [op["metadata"]["intent_kind"] for op in plan["ops"]] == ["ssd_read", "dram_to_vram", "compute"]
    assert json.loads(summary_path.read_text())["movement_intent_count"] == 3


def test_exported_movement_intent_plan_can_feed_movement_executor(tmp_path) -> None:
    trace_path = tmp_path / "bridge.jsonl"
    plan_path = tmp_path / "intent_plan.json"
    write_trace(trace_path, [make_event(), make_event(command_op="kv_compute", source="vram", target="vram", intent_kind="compute", blocking=False)])
    export_movement_intent_plan(trace_path, plan_path=plan_path)

    result = execute_movement_plan(plan_path, tmp_path / "execution", execute_io=False)

    assert result.movement_op_count == 2
    assert result.ssd_load_op_count == 1
    assert result.unique_ssd_read_count == 1
    assert result.ssd_read_bytes == 4096


def test_export_movement_intent_plan_rejects_conflicting_block_range(tmp_path) -> None:
    trace_path = tmp_path / "bridge.jsonl"
    write_trace(trace_path, [make_event(ssd_offset=0), make_event(ssd_offset=4096)])

    with pytest.raises(ValueError, match="conflicting SSD range"):
        export_movement_intent_plan(trace_path)
