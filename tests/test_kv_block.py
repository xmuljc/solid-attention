import pytest

from core.kv_block import BlockLocation, BlockRole, KVBlock


def test_kv_block_initialization_and_fields() -> None:
    block = KVBlock(
        block_id=1,
        layer_id=2,
        start_token=32,
        end_token=64,
        size_bytes=4096,
        location="dram",
        role="init",
    )

    assert block.block_id == 1
    assert block.layer_id == 2
    assert block.start_token == 32
    assert block.end_token == 64
    assert block.size_bytes == 4096
    assert block.location == BlockLocation.DRAM
    assert block.role == BlockRole.INIT
    assert block.token_count == 32


def test_kv_block_rejects_invalid_token_range() -> None:
    with pytest.raises(ValueError):
        KVBlock(
            block_id=0,
            layer_id=0,
            start_token=4,
            end_token=4,
            size_bytes=1024,
        )

    with pytest.raises(ValueError):
        KVBlock(
            block_id=0,
            layer_id=0,
            start_token=-1,
            end_token=4,
            size_bytes=1024,
        )


def test_kv_block_rejects_invalid_ids_and_size() -> None:
    with pytest.raises(ValueError):
        KVBlock(-1, 0, 0, 1, 1)

    with pytest.raises(ValueError):
        KVBlock(0, -1, 0, 1, 1)

    with pytest.raises(ValueError):
        KVBlock(0, 0, 0, 1, -1)


def test_kv_block_location_transfer_and_residency() -> None:
    block = KVBlock(
        block_id=3,
        layer_id=0,
        start_token=96,
        end_token=128,
        size_bytes=2048,
        location="ssd",
        role="prefetch",
    )

    assert block.is_resident("ssd")
    assert not block.is_resident("vram")

    block.move_to("vram")

    assert block.is_resident("vram")
    assert block.location == BlockLocation.VRAM


def test_kv_block_supports_all_required_roles() -> None:
    for role in ("init", "local", "selected", "current", "prefetch"):
        block = KVBlock(0, 0, 0, 1, 1, role=role)
        assert block.role.value == role
