import pytest

from core.block_store import BlockStore
from core.kv_block import KVBlock
from core.scheduler import SSDAwareScheduler, SchedulerConfig
from harness.metrics import MetricsCollector
from harness.trace import TraceRecorder


def make_block(block_id: int, location: str) -> KVBlock:
    return KVBlock(
        block_id=block_id,
        layer_id=0,
        start_token=block_id * 32,
        end_token=(block_id + 1) * 32,
        size_bytes=4096,
        location=location,
    )


def make_store(blocks: list[KVBlock]) -> BlockStore:
    store = BlockStore()
    for block in blocks:
        store.add_block(block)
    return store


def test_scheduler_loads_ssd_and_dram_blocks_before_compute() -> None:
    blocks = [make_block(0, "ssd"), make_block(1, "dram"), make_block(2, "vram")]
    store = make_store(blocks)
    trace = TraceRecorder()
    metrics = MetricsCollector()
    scheduler = SSDAwareScheduler(
        store,
        SchedulerConfig(ssd_read_latency_ms=4.0, dram_to_vram_latency_ms=1.0, compute_latency_ms=0.5),
        trace=trace,
        metrics=metrics,
    )

    result = scheduler.schedule(blocks, step=1, start_ms=10.0)

    assert [op.op for op in result.ops] == [
        "schedule_ssd_read",
        "schedule_dram_to_vram",
        "schedule_compute",
        "schedule_dram_to_vram",
        "schedule_compute",
        "schedule_compute",
    ]
    assert [block.location.value for block in blocks] == ["vram", "vram", "vram"]
    assert result.total_latency_ms == 7.5
    assert metrics.ssd_read_bytes == 4096
    assert metrics.io_op_count == 1
    assert metrics.gpu_wait_time_ms == 6.0
    assert metrics.load_blocking_latency_ms == 6.0
    assert metrics.total_latency_ms == 7.5
    assert trace.to_list()[0]["op"] == "schedule_ssd_read"
    assert trace.to_list()[0]["metadata"]["to_location"] == "dram"


def test_scheduler_dedupes_blocks() -> None:
    block = make_block(0, "vram")
    store = make_store([block])
    scheduler = SSDAwareScheduler(store, SchedulerConfig(compute_latency_ms=0.25))

    result = scheduler.schedule([block, block], step=1)

    assert [op.op for op in result.ops] == ["schedule_compute"]
    assert result.total_latency_ms == 0.25


def test_scheduler_can_schedule_selection_result() -> None:
    blocks = [make_block(0, "vram"), make_block(1, "vram")]
    store = make_store(blocks)
    scheduler = SSDAwareScheduler(store)

    class FakeSelection:
        all_blocks = tuple(blocks)

    result = scheduler.schedule_selection_result(FakeSelection(), step=2)

    assert [op.block_id for op in result.ops] == [0, 1]
    assert all(op.op == "schedule_compute" for op in result.ops)


def test_scheduler_config_rejects_negative_latencies() -> None:
    with pytest.raises(ValueError):
        SchedulerConfig(ssd_read_latency_ms=-1)

    with pytest.raises(ValueError):
        SchedulerConfig(dram_to_vram_latency_ms=-1)

    with pytest.raises(ValueError):
        SchedulerConfig(compute_latency_ms=-1)
