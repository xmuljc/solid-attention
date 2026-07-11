import pytest

from core.block_store import BlockStore
from core.kv_block import KVBlock


def make_block(block_id: int, layer_id: int = 0, location: str = "dram", size: int = 1024) -> KVBlock:
    return KVBlock(
        block_id=block_id,
        layer_id=layer_id,
        start_token=block_id * 32,
        end_token=(block_id + 1) * 32,
        size_bytes=size,
        location=location,
    )


def test_block_store_add_and_query() -> None:
    store = BlockStore()
    block = make_block(0, layer_id=1, location="ssd")

    store.add_block(block)

    assert store.get_block(layer_id=1, block_id=0) is block


def test_block_store_rejects_duplicate_blocks() -> None:
    store = BlockStore()
    store.add_block(make_block(0))

    with pytest.raises(KeyError):
        store.add_block(make_block(0))


def test_block_store_rejects_unknown_lookup() -> None:
    store = BlockStore()

    with pytest.raises(KeyError):
        store.get_block(layer_id=0, block_id=99)


def test_block_store_move_block() -> None:
    store = BlockStore()
    store.add_block(make_block(0, location="ssd"))

    moved = store.move_block(layer_id=0, block_id=0, target_location="vram")

    assert moved.is_resident("vram")
    assert store.get_block(0, 0).is_resident("vram")


def test_block_store_list_blocks_filters_by_layer_and_location() -> None:
    store = BlockStore()
    store.add_block(make_block(0, layer_id=0, location="ssd"))
    store.add_block(make_block(1, layer_id=0, location="dram"))
    store.add_block(make_block(0, layer_id=1, location="dram"))

    assert [block.block_id for block in store.list_blocks(layer_id=0)] == [0, 1]
    assert [(block.layer_id, block.block_id) for block in store.list_blocks(location="dram")] == [
        (0, 1),
        (1, 0),
    ]


def test_block_store_location_stats() -> None:
    store = BlockStore()
    store.add_block(make_block(0, location="ssd", size=1024))
    store.add_block(make_block(1, location="ssd", size=2048))
    store.add_block(make_block(2, location="vram", size=4096))

    stats = store.location_stats()

    assert stats["ssd"] == {"block_count": 2, "size_bytes": 3072}
    assert stats["dram"] == {"block_count": 0, "size_bytes": 0}
    assert stats["vram"] == {"block_count": 1, "size_bytes": 4096}


def test_block_store_upsert_replaces_existing_block() -> None:
    store = BlockStore()
    store.add_block(make_block(0, location="ssd", size=1024))

    replacement = make_block(0, location="dram", size=2048)
    stored = store.upsert_block(replacement)

    assert stored is replacement
    assert store.get_block(0, 0) is replacement
    assert store.get_block(0, 0).is_resident("dram")
    assert store.location_stats()["dram"] == {"block_count": 1, "size_bytes": 2048}


def test_block_store_has_block() -> None:
    store = BlockStore()
    assert not store.has_block(0, 0)
    store.add_block(make_block(0))
    assert store.has_block(0, 0)
