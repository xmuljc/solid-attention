import pytest

from core.kv_block import KVBlock
from core.kv_layout import BlockLayout, InterleavedKVLayout, KVExtent


def make_block(block_id: int, start: int, end: int) -> KVBlock:
    return KVBlock(
        block_id=block_id,
        layer_id=0,
        start_token=start,
        end_token=end,
        size_bytes=(end - start) * 8,
        location="ssd",
    )


def test_block_layout_interleaves_key_and_value_extents() -> None:
    block = make_block(0, 10, 12)
    layout = BlockLayout(block, base_offset=100, key_size_bytes=4, value_size_bytes=6)

    k10, v10 = layout.token_extents(10)
    k11, v11 = layout.token_extents(11)

    assert k10 == KVExtent("k", 10, 100, 4)
    assert v10 == KVExtent("v", 10, 104, 6)
    assert k11 == KVExtent("k", 11, 110, 4)
    assert v11 == KVExtent("v", 11, 114, 6)
    assert layout.total_size_bytes == 20
    assert layout.end_offset == 120


def test_layout_rejects_out_of_range_token() -> None:
    block = make_block(0, 0, 2)
    layout = BlockLayout(block, base_offset=0, key_size_bytes=4, value_size_bytes=4)

    with pytest.raises(ValueError):
        layout.key_extent(2)


def test_interleaved_layout_assigns_contiguous_block_spans() -> None:
    layout = InterleavedKVLayout(key_size_bytes=4, value_size_bytes=4, base_offset=128)
    first = layout.add_block(make_block(0, 0, 2))
    second = layout.add_block(make_block(1, 2, 5))

    assert layout.block_span(0, 0) == (128, 144)
    assert layout.block_span(0, 1) == (144, 168)
    assert first.end_offset == second.base_offset
    assert layout.token_extents(0, 1, 3)[0].offset == 152
    assert [entry["block_id"] for entry in layout.to_list()] == [0, 1]


def test_interleaved_layout_rejects_duplicate_and_unknown_blocks() -> None:
    block = make_block(0, 0, 1)
    layout = InterleavedKVLayout(key_size_bytes=4, value_size_bytes=4)
    layout.add_block(block)

    with pytest.raises(KeyError):
        layout.add_block(block)

    with pytest.raises(KeyError):
        layout.get_block_layout(0, 99)


def test_layout_config_validation() -> None:
    with pytest.raises(ValueError):
        InterleavedKVLayout(key_size_bytes=0, value_size_bytes=4)

    with pytest.raises(ValueError):
        InterleavedKVLayout(key_size_bytes=4, value_size_bytes=0)

    with pytest.raises(ValueError):
        InterleavedKVLayout(key_size_bytes=4, value_size_bytes=4, base_offset=-1)
