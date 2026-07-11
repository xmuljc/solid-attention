from core.block_store import BlockStore
from core.kv_block import KVBlock
from core.prefetcher import PrefetchConfig, SpeculativePrefetcher
from harness.metrics import MetricsCollector
from harness.trace import TraceRecorder


def make_block(block_id: int, location: str = "ssd") -> KVBlock:
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


def test_prefetch_plan_dedupes_and_caps_blocks() -> None:
    blocks = [make_block(0), make_block(1), make_block(1), make_block(2)]
    store = make_store(blocks[:1] + blocks[1:2] + blocks[3:])
    trace = TraceRecorder()
    prefetcher = SpeculativePrefetcher(
        store,
        PrefetchConfig(target_location="dram", max_prefetch_blocks=2),
        trace=trace,
    )

    plan = prefetcher.plan(blocks, step=4, start_ms=10.0)

    assert [block.block_id for block in plan.blocks] == [0, 1]
    assert plan.to_dict()["target_location"] == "dram"
    assert [event["op"] for event in trace.to_list()] == ["prefetch_submit", "prefetch_submit"]
    assert trace.to_list()[0]["metadata"]["step"] == 4


def test_prefetch_execute_moves_blocks_and_records_metrics() -> None:
    blocks = [make_block(0, "ssd"), make_block(1, "dram")]
    store = make_store(blocks)
    trace = TraceRecorder()
    metrics = MetricsCollector()
    prefetcher = SpeculativePrefetcher(
        store,
        PrefetchConfig(target_location="vram", mock_latency_ms=2.5),
        trace=trace,
        metrics=metrics,
    )

    plan = prefetcher.plan(blocks, step=1)
    prefetcher.execute(plan, start_ms=20.0)

    assert store.get_block(0, 0).is_resident("vram")
    assert store.get_block(0, 1).is_resident("vram")
    assert metrics.ssd_read_bytes == 4096
    assert metrics.io_op_count == 1
    assert metrics.total_latency_ms == 2.5
    move_events = [event for event in trace.to_list() if event["op"] == "prefetch_move"]
    assert len(move_events) == 2
    assert move_events[0]["duration_ms"] == 2.5


def test_prefetch_evaluate_counts_hits_misses_and_wrong_prefetches() -> None:
    blocks = [make_block(0), make_block(1), make_block(2)]
    store = make_store(blocks)
    trace = TraceRecorder()
    metrics = MetricsCollector()
    prefetcher = SpeculativePrefetcher(
        store,
        PrefetchConfig(target_location="dram", mock_latency_ms=4.0),
        trace=trace,
        metrics=metrics,
    )

    plan = prefetcher.plan([blocks[0], blocks[1]], step=1)
    prefetcher.execute(plan)
    evaluation = prefetcher.evaluate(plan, [blocks[0], blocks[2]], step=2, start_ms=30.0)

    assert [block.block_id for block in evaluation.hits] == [0]
    assert [block.block_id for block in evaluation.misses] == [2]
    assert [block.block_id for block in evaluation.wrong_prefetches] == [1]
    assert metrics.prefetch_hit_count == 1
    assert metrics.prefetch_miss_count == 1
    assert metrics.wrong_prefetch_count == 1
    assert metrics.prefetch_hit_rate == 0.5
    assert metrics.load_blocking_latency_ms == 4.0
    assert [event["op"] for event in trace.to_list()[-3:]] == [
        "prefetch_hit",
        "prefetch_miss",
        "prefetch_wrong",
    ]


def test_prefetch_without_execute_is_a_miss_even_if_planned() -> None:
    block = make_block(0, "ssd")
    store = make_store([block])
    metrics = MetricsCollector()
    prefetcher = SpeculativePrefetcher(
        store,
        PrefetchConfig(target_location="dram", mock_latency_ms=1.0),
        metrics=metrics,
    )

    plan = prefetcher.plan([block], step=1)
    evaluation = prefetcher.evaluate(plan, [block], step=2)

    assert evaluation.hits == ()
    assert [miss.block_id for miss in evaluation.misses] == [0]
    assert metrics.prefetch_miss_count == 1


def test_prefetch_config_rejects_invalid_values() -> None:
    try:
        PrefetchConfig(mock_latency_ms=-1)
    except ValueError as exc:
        assert "mock_latency_ms" in str(exc)
    else:
        raise AssertionError("negative mock latency should fail")

    try:
        PrefetchConfig(max_prefetch_blocks=-1)
    except ValueError as exc:
        assert "max_prefetch_blocks" in str(exc)
    else:
        raise AssertionError("negative max_prefetch_blocks should fail")
