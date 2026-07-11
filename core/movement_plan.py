"""Runtime KV movement plans derived from scheduler timelines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from core.kv_block import KVBlock


_VALID_MOVEMENT_OPS = {"kv_evict", "kv_load", "kv_prefetch", "kv_compute"}


@dataclass(frozen=True)
class SSDRange:
    """Byte range for one KV block in the backing SSD file."""

    layer_id: int
    block_id: int
    offset: int
    size_bytes: int

    def __post_init__(self) -> None:
        if self.layer_id < 0:
            raise ValueError("layer_id must be non-negative")
        if self.block_id < 0:
            raise ValueError("block_id must be non-negative")
        if self.offset < 0:
            raise ValueError("offset must be non-negative")
        if self.size_bytes <= 0:
            raise ValueError("size_bytes must be positive")

    @property
    def end_offset(self) -> int:
        return self.offset + self.size_bytes

    @property
    def key(self) -> tuple[int, int]:
        return (self.layer_id, self.block_id)

    def to_dict(self) -> dict[str, int]:
        return {
            "layer_id": self.layer_id,
            "block_id": self.block_id,
            "offset": self.offset,
            "size_bytes": self.size_bytes,
            "end_offset": self.end_offset,
        }


class SSDFileLayout:
    """Assigns aligned contiguous SSD byte ranges to KV blocks."""

    def __init__(self, *, base_offset: int = 0, alignment: int = 4096) -> None:
        if base_offset < 0:
            raise ValueError("base_offset must be non-negative")
        if alignment <= 0:
            raise ValueError("alignment must be positive")
        self.base_offset = base_offset
        self.alignment = alignment
        self._ranges: dict[tuple[int, int], SSDRange] = {}

    def add_block(self, block: KVBlock) -> SSDRange:
        if block.key in self._ranges:
            raise KeyError(f"SSD range already exists for block: {block.key}")
        if block.size_bytes <= 0:
            raise ValueError("block size_bytes must be positive for SSD layout")
        offset = self._align(self._next_offset())
        block_range = SSDRange(
            layer_id=block.layer_id,
            block_id=block.block_id,
            offset=offset,
            size_bytes=block.size_bytes,
        )
        self._ranges[block.key] = block_range
        return block_range

    def get_range(self, layer_id: int, block_id: int) -> SSDRange:
        key = (layer_id, block_id)
        try:
            return self._ranges[key]
        except KeyError as exc:
            raise KeyError(f"unknown SSD range: {key}") from exc

    @property
    def total_size_bytes(self) -> int:
        if not self._ranges:
            return 0
        return max(item.end_offset for item in self._ranges.values()) - self.base_offset

    def to_list(self) -> list[dict[str, int]]:
        return [
            item.to_dict()
            for item in sorted(self._ranges.values(), key=lambda value: (value.layer_id, value.block_id))
        ]

    def _next_offset(self) -> int:
        if not self._ranges:
            return self.base_offset
        return max(item.end_offset for item in self._ranges.values())

    def _align(self, offset: int) -> int:
        remainder = offset % self.alignment
        if remainder == 0:
            return offset
        return offset + self.alignment - remainder


@dataclass(frozen=True)
class MovementOp:
    """One executable runtime KV movement or compute event."""

    op: str
    layer_id: int
    block_id: int
    source: str
    target: str
    ssd_offset: int
    size_bytes: int
    start_token: int
    end_token: int
    start_ms: float
    end_ms: float
    blocking: bool
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.op not in _VALID_MOVEMENT_OPS:
            raise ValueError(f"unsupported movement op: {self.op}")
        if self.layer_id < 0:
            raise ValueError("layer_id must be non-negative")
        if self.block_id < 0:
            raise ValueError("block_id must be non-negative")
        if self.ssd_offset < 0:
            raise ValueError("ssd_offset must be non-negative")
        if self.size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        if self.start_token < 0:
            raise ValueError("start_token must be non-negative")
        if self.end_token <= self.start_token:
            raise ValueError("end_token must be greater than start_token")
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be >= start_ms")

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "layer_id": self.layer_id,
            "block_id": self.block_id,
            "source": self.source,
            "target": self.target,
            "ssd_offset": self.ssd_offset,
            "size_bytes": self.size_bytes,
            "start_token": self.start_token,
            "end_token": self.end_token,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "blocking": self.blocking,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MovementPlan:
    """Runtime-oriented movement plan plus SSD backing layout."""

    ops: tuple[MovementOp, ...]
    layout: SSDFileLayout

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout": {
                "base_offset": self.layout.base_offset,
                "alignment": self.layout.alignment,
                "total_size_bytes": self.layout.total_size_bytes,
                "blocks": self.layout.to_list(),
            },
            "ops": [op.to_dict() for op in self.ops],
        }

    def save_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_movement_plan(
    blocks: Iterable[KVBlock],
    schedule_result: object,
    *,
    base_offset: int = 0,
    alignment: int = 4096,
) -> MovementPlan:
    """Build a runtime movement plan from blocks and scheduler operations."""

    layout = SSDFileLayout(base_offset=base_offset, alignment=alignment)
    unique_blocks = _dedupe_blocks(blocks)
    for block in unique_blocks:
        layout.add_block(block)

    blocks_by_key = {block.key: block for block in unique_blocks}
    ops = []
    for schedule_op in getattr(schedule_result, "ops"):
        key = (int(schedule_op.layer_id), int(schedule_op.block_id))
        block_range = layout.get_range(schedule_op.layer_id, schedule_op.block_id)
        ops.append(_movement_from_schedule_op(schedule_op, block_range, blocks_by_key[key]))
    return MovementPlan(ops=tuple(ops), layout=layout)


def _movement_from_schedule_op(schedule_op: object, block_range: SSDRange, block: KVBlock) -> MovementOp:
    schedule_name = str(getattr(schedule_op, "op"))
    metadata = dict(getattr(schedule_op, "metadata"))
    metadata["scheduler_op"] = schedule_name

    if schedule_name == "schedule_ssd_read":
        movement_op = "kv_load"
        source = str(metadata.get("from_location", "ssd"))
        target = str(metadata.get("to_location", "dram"))
    elif schedule_name == "schedule_dram_to_vram":
        movement_op = "kv_load"
        source = str(metadata.get("from_location", "dram"))
        target = str(metadata.get("to_location", "vram"))
    elif schedule_name == "schedule_compute":
        movement_op = "kv_compute"
        source = str(metadata.get("location", "vram"))
        target = source
    elif schedule_name == "prefetch_move":
        movement_op = "kv_prefetch"
        source = str(metadata.get("from_location", "ssd"))
        target = str(metadata.get("to_location", "dram"))
    elif schedule_name == "evict_to_ssd":
        movement_op = "kv_evict"
        source = str(metadata.get("from_location", "vram"))
        target = str(metadata.get("to_location", "ssd"))
    else:
        raise ValueError(f"unsupported scheduler op for movement plan: {schedule_name}")

    return MovementOp(
        op=movement_op,
        layer_id=int(getattr(schedule_op, "layer_id")),
        block_id=int(getattr(schedule_op, "block_id")),
        source=source,
        target=target,
        ssd_offset=block_range.offset,
        size_bytes=block_range.size_bytes,
        start_token=int(block.start_token),
        end_token=int(block.end_token),
        start_ms=float(getattr(schedule_op, "start_ms")),
        end_ms=float(getattr(schedule_op, "end_ms")),
        blocking=bool(metadata.get("blocking", False)),
        metadata=metadata,
    )


def _dedupe_blocks(blocks: Iterable[KVBlock]) -> list[KVBlock]:
    unique: list[KVBlock] = []
    seen: set[tuple[int, int]] = set()
    for block in blocks:
        if block.key in seen:
            continue
        unique.append(block)
        seen.add(block.key)
    return sorted(unique, key=lambda block: (block.layer_id, block.block_id))
