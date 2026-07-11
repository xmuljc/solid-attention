import json

import pytest

from core.block_store import BlockStore
from core.kv_block import KVBlock
from core.movement_plan import SSDFileLayout, build_movement_plan
from core.scheduler import SSDAwareScheduler, SchedulerConfig, ScheduleOp


def make_block(block_id: int, layer_id: int = 0, size: int = 1024, location: str = "ssd") -> KVBlock:
    return KVBlock(
        block_id=block_id,
        layer_id=layer_id,
        start_token=block_id * 64,
        end_token=(block_id + 1) * 64,
        size_bytes=size,
        location=location,
    )


def test_ssd_file_layout_assigns_aligned_ranges() -> None:
    layout = SSDFileLayout(alignment=4096)
    first = layout.add_block(make_block(0, size=1000))
    second = layout.add_block(make_block(1, size=2000))

    assert first.offset == 0
    assert first.end_offset == 1000
    assert second.offset == 4096
    assert second.end_offset == 6096
    assert layout.total_size_bytes == 6096
    assert layout.to_list()[1]["offset"] == 4096


def test_ssd_file_layout_rejects_duplicates_and_zero_size() -> None:
    layout = SSDFileLayout()
    block = make_block(0)
    layout.add_block(block)

    with pytest.raises(KeyError):
        layout.add_block(block)
    with pytest.raises(ValueError):
        layout.add_block(make_block(1, size=0))


def test_build_movement_plan_maps_scheduler_ops_to_runtime_ops(tmp_path) -> None:
    store = BlockStore()
    blocks = [make_block(0, layer_id=0, size=1024), make_block(0, layer_id=1, size=2048)]
    for block in blocks:
        store.add_block(block)

    scheduler = SSDAwareScheduler(
        store,
        SchedulerConfig(ssd_read_latency_ms=1.0, dram_to_vram_latency_ms=0.25, compute_latency_ms=0.1),
    )
    schedule = scheduler.schedule(blocks, step=0)
    plan = build_movement_plan(blocks, schedule, alignment=4096)

    assert [op.op for op in plan.ops] == [
        "kv_load",
        "kv_load",
        "kv_compute",
        "kv_load",
        "kv_load",
        "kv_compute",
    ]
    assert plan.ops[0].source == "ssd"
    assert plan.ops[0].target == "dram"
    assert plan.ops[1].source == "dram"
    assert plan.ops[1].target == "vram"
    assert plan.ops[2].source == "vram"
    assert plan.ops[2].target == "vram"
    assert plan.ops[0].ssd_offset == 0
    assert plan.ops[0].start_token == 0
    assert plan.ops[0].end_token == 64
    assert plan.ops[3].ssd_offset == 4096
    assert plan.ops[3].start_token == 0
    assert plan.ops[3].end_token == 64

    output = tmp_path / "movement_plan.json"
    plan.save_json(output)
    payload = json.loads(output.read_text())
    assert payload["layout"]["alignment"] == 4096
    assert payload["layout"]["total_size_bytes"] == 6144
    assert payload["ops"][0]["metadata"]["scheduler_op"] == "schedule_ssd_read"
    assert payload["ops"][0]["start_token"] == 0
    assert payload["ops"][0]["end_token"] == 64


def test_build_movement_plan_rejects_unknown_scheduler_op() -> None:
    block = make_block(0)
    unknown = ScheduleOp(
        op="unknown",
        layer_id=0,
        block_id=0,
        device="ssd",
        start_ms=0.0,
        end_ms=1.0,
        metadata={},
    )

    with pytest.raises(ValueError):
        build_movement_plan([block], type("Schedule", (), {"ops": (unknown,)})())
