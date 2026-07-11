import pytest

from core.block_selector import BlockSelector, SelectionConfig, representative_score
from core.kv_block import BlockRole, KVBlock
from harness.trace import TraceRecorder


def make_block(block_id: int, layer_id: int = 0) -> KVBlock:
    return KVBlock(
        block_id=block_id,
        layer_id=layer_id,
        start_token=block_id * 32,
        end_token=(block_id + 1) * 32,
        size_bytes=4096,
        location="ssd",
    )


def test_selector_picks_init_local_and_selected_blocks() -> None:
    blocks = [make_block(block_id) for block_id in range(6)]
    blocks[2].set_representative([1.0, 0.0])
    blocks[3].set_representative([0.0, 1.0])

    selector = BlockSelector(
        SelectionConfig(init_block_count=2, local_block_count=1, selected_block_count=1)
    )

    result = selector.select(
        blocks,
        layer_id=0,
        current_token=192,
        query_vector=[0.9, 0.1],
    )

    assert [block.block_id for block in result.init_blocks] == [0, 1]
    assert [block.block_id for block in result.local_blocks] == [5]
    assert [block.block_id for block in result.selected_blocks] == [2]
    assert blocks[0].role == BlockRole.INIT
    assert blocks[1].role == BlockRole.INIT
    assert blocks[5].role == BlockRole.LOCAL
    assert blocks[2].role == BlockRole.SELECTED


def test_selector_filters_by_layer_and_current_token() -> None:
    blocks = [make_block(0, layer_id=0), make_block(1, layer_id=0), make_block(0, layer_id=1)]
    selector = BlockSelector(
        SelectionConfig(init_block_count=1, local_block_count=1, selected_block_count=1)
    )

    result = selector.select(blocks, layer_id=0, current_token=64, query_vector=[1.0])

    assert result.to_dict()["all_blocks"] == [0, 1]
    assert all(block.layer_id == 0 for block in result.all_blocks)


def test_selector_emits_trace_for_each_selected_group() -> None:
    blocks = [make_block(block_id) for block_id in range(4)]
    blocks[1].set_representative([1.0, 0.0])
    trace = TraceRecorder()
    selector = BlockSelector(
        SelectionConfig(init_block_count=1, local_block_count=1, selected_block_count=1),
        trace=trace,
    )

    selector.select(
        blocks,
        layer_id=0,
        current_token=128,
        query_vector=[1.0, 0.0],
        start_ms=7.5,
    )

    events = trace.to_list()
    assert [event["op"] for event in events] == [
        "select_init",
        "select_local",
        "select_representative",
    ]
    assert events[0]["metadata"]["role"] == "init"
    assert events[1]["metadata"]["role"] == "local"
    assert events[2]["metadata"]["role"] == "selected"
    assert events[2]["metadata"]["score"] == pytest.approx(1.0)


def test_representative_score_validates_dimensions() -> None:
    block = make_block(0)
    block.set_representative([1.0, 0.0])

    with pytest.raises(ValueError):
        representative_score(block, [1.0])


def test_selection_config_rejects_negative_counts() -> None:
    with pytest.raises(ValueError):
        SelectionConfig(init_block_count=-1)

    with pytest.raises(ValueError):
        SelectionConfig(local_block_count=-1)

    with pytest.raises(ValueError):
        SelectionConfig(selected_block_count=-1)
