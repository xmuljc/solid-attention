"""Interleaved KV layout simulation for SolidAttention Phase 1.

This module models token-level K/V interleaving as metadata offsets. It does not
allocate tensors and does not perform real SSD I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.kv_block import KVBlock


@dataclass(frozen=True)
class KVExtent:
    """Byte range for one K or V segment."""

    kind: str
    token_id: int
    offset: int
    size_bytes: int

    def __post_init__(self) -> None:
        if self.kind not in {"k", "v"}:
            raise ValueError("kind must be 'k' or 'v'")
        if self.token_id < 0:
            raise ValueError("token_id must be non-negative")
        if self.offset < 0:
            raise ValueError("offset must be non-negative")
        if self.size_bytes <= 0:
            raise ValueError("size_bytes must be positive")

    @property
    def end_offset(self) -> int:
        return self.offset + self.size_bytes

    def to_dict(self) -> dict[str, int | str]:
        return {
            "kind": self.kind,
            "token_id": self.token_id,
            "offset": self.offset,
            "size_bytes": self.size_bytes,
            "end_offset": self.end_offset,
        }


@dataclass(frozen=True)
class BlockLayout:
    """Interleaved layout for one KV block."""

    block: KVBlock
    base_offset: int
    key_size_bytes: int
    value_size_bytes: int

    def __post_init__(self) -> None:
        if self.base_offset < 0:
            raise ValueError("base_offset must be non-negative")
        if self.key_size_bytes <= 0:
            raise ValueError("key_size_bytes must be positive")
        if self.value_size_bytes <= 0:
            raise ValueError("value_size_bytes must be positive")

    @property
    def token_stride_bytes(self) -> int:
        return self.key_size_bytes + self.value_size_bytes

    @property
    def total_size_bytes(self) -> int:
        return self.block.token_count * self.token_stride_bytes

    @property
    def end_offset(self) -> int:
        return self.base_offset + self.total_size_bytes

    def key_extent(self, token_id: int) -> KVExtent:
        self._check_token(token_id)
        offset = self.base_offset + self._token_index(token_id) * self.token_stride_bytes
        return KVExtent("k", token_id, offset, self.key_size_bytes)

    def value_extent(self, token_id: int) -> KVExtent:
        self._check_token(token_id)
        offset = (
            self.base_offset
            + self._token_index(token_id) * self.token_stride_bytes
            + self.key_size_bytes
        )
        return KVExtent("v", token_id, offset, self.value_size_bytes)

    def token_extents(self, token_id: int) -> tuple[KVExtent, KVExtent]:
        return (self.key_extent(token_id), self.value_extent(token_id))

    def to_dict(self) -> dict[str, object]:
        return {
            "layer_id": self.block.layer_id,
            "block_id": self.block.block_id,
            "start_token": self.block.start_token,
            "end_token": self.block.end_token,
            "base_offset": self.base_offset,
            "end_offset": self.end_offset,
            "total_size_bytes": self.total_size_bytes,
            "token_stride_bytes": self.token_stride_bytes,
            "key_size_bytes": self.key_size_bytes,
            "value_size_bytes": self.value_size_bytes,
        }

    def _token_index(self, token_id: int) -> int:
        assert self.block.start_token is not None
        return token_id - self.block.start_token

    def _check_token(self, token_id: int) -> None:
        if not self.block.contains_token(token_id):
            raise ValueError(
                f"token {token_id} is outside block range "
                f"[{self.block.start_token}, {self.block.end_token})"
            )


class InterleavedKVLayout:
    """Assigns contiguous interleaved K/V byte ranges to blocks."""

    def __init__(self, key_size_bytes: int, value_size_bytes: int, base_offset: int = 0) -> None:
        if base_offset < 0:
            raise ValueError("base_offset must be non-negative")
        if key_size_bytes <= 0:
            raise ValueError("key_size_bytes must be positive")
        if value_size_bytes <= 0:
            raise ValueError("value_size_bytes must be positive")
        self.key_size_bytes = key_size_bytes
        self.value_size_bytes = value_size_bytes
        self.base_offset = base_offset
        self._layouts: dict[tuple[int, int], BlockLayout] = {}

    def add_block(self, block: KVBlock) -> BlockLayout:
        if block.key in self._layouts:
            raise KeyError(f"layout already exists for block: {block.key}")
        offset = self._next_offset()
        layout = BlockLayout(
            block=block,
            base_offset=offset,
            key_size_bytes=self.key_size_bytes,
            value_size_bytes=self.value_size_bytes,
        )
        self._layouts[block.key] = layout
        return layout

    def get_block_layout(self, layer_id: int, block_id: int) -> BlockLayout:
        key = (layer_id, block_id)
        try:
            return self._layouts[key]
        except KeyError as exc:
            raise KeyError(f"unknown block layout: {key}") from exc

    def token_extents(self, layer_id: int, block_id: int, token_id: int) -> tuple[KVExtent, KVExtent]:
        return self.get_block_layout(layer_id, block_id).token_extents(token_id)

    def block_span(self, layer_id: int, block_id: int) -> tuple[int, int]:
        layout = self.get_block_layout(layer_id, block_id)
        return (layout.base_offset, layout.end_offset)

    def to_list(self) -> list[dict[str, object]]:
        return [
            layout.to_dict()
            for layout in sorted(
                self._layouts.values(),
                key=lambda item: (item.block.layer_id, item.block.block_id),
            )
        ]

    def _next_offset(self) -> int:
        if not self._layouts:
            return self.base_offset
        return max(layout.end_offset for layout in self._layouts.values())
