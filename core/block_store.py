"""Simulated block store for SSD/DRAM/VRAM KV cache placement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from core.kv_block import BlockLocation, KVBlock


@dataclass(frozen=True)
class LocationStats:
    """Residency statistics for one simulated location."""

    block_count: int = 0
    size_bytes: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "block_count": self.block_count,
            "size_bytes": self.size_bytes,
        }


@dataclass
class BlockStore:
    """Tracks simulated KV blocks across SSD, DRAM, and VRAM.

    This class performs metadata-only state transitions. It does not perform
    real SSD I/O, GPU memory operations, or attention computation.
    """

    _blocks: dict[tuple[int, int], KVBlock] = field(default_factory=dict)

    def add_block(self, block: KVBlock) -> None:
        if block.key in self._blocks:
            raise KeyError(f"block already exists: {block.key}")
        self._blocks[block.key] = block

    def upsert_block(self, block: KVBlock) -> KVBlock:
        self._blocks[block.key] = block
        return block

    def has_block(self, layer_id: int, block_id: int) -> bool:
        return (layer_id, block_id) in self._blocks

    def get_block(self, layer_id: int, block_id: int) -> KVBlock:
        key = (layer_id, block_id)
        try:
            return self._blocks[key]
        except KeyError as exc:
            raise KeyError(f"unknown block: {key}") from exc

    def move_block(
        self,
        layer_id: int,
        block_id: int,
        target_location: BlockLocation | str,
    ) -> KVBlock:
        block = self.get_block(layer_id, block_id)
        block.move_to(target_location)
        return block

    def list_blocks(
        self,
        layer_id: int | None = None,
        location: BlockLocation | str | None = None,
    ) -> list[KVBlock]:
        normalized_location = BlockLocation(location) if location is not None else None
        blocks = []
        for block in self._blocks.values():
            if layer_id is not None and block.layer_id != layer_id:
                continue
            if normalized_location is not None and block.location != normalized_location:
                continue
            blocks.append(block)
        return sorted(blocks, key=lambda block: (block.layer_id, block.block_id))

    def location_stats(self) -> dict[str, dict[str, int]]:
        stats = {
            location.value: {"block_count": 0, "size_bytes": 0}
            for location in BlockLocation
        }
        for block in self._blocks.values():
            entry = stats[block.location.value]
            entry["block_count"] += 1
            entry["size_bytes"] += block.size_bytes
        return stats

    def total_size_bytes(self, location: BlockLocation | str | None = None) -> int:
        return sum(block.size_bytes for block in self.list_blocks(location=location))

    # Backward-compatible aliases from the first skeleton.
    def add(self, block: KVBlock) -> None:
        self.add_block(block)

    def get(self, layer_id: int, block_id: int) -> KVBlock:
        return self.get_block(layer_id, block_id)

    def move(
        self,
        layer_id: int,
        block_id: int,
        location: BlockLocation | str,
    ) -> KVBlock:
        return self.move_block(layer_id, block_id, location)

    def blocks(self) -> list[KVBlock]:
        return self.list_blocks()

    def by_location(self, location: BlockLocation | str) -> list[KVBlock]:
        return self.list_blocks(location=location)

    def stats(self) -> dict[str, dict[str, int]]:
        return self.location_stats()


def make_store(blocks: Iterable[KVBlock]) -> BlockStore:
    store = BlockStore()
    for block in blocks:
        store.add_block(block)
    return store
