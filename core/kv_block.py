"""KV cache block metadata for SolidAttention Phase 1 simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BlockLocation(str, Enum):
    """Simulated storage tier for a KV block."""

    SSD = "ssd"
    DRAM = "dram"
    VRAM = "vram"


class BlockRole(str, Enum):
    """Logical scheduler role assigned to a KV block."""

    INIT = "init"
    LOCAL = "local"
    SELECTED = "selected"
    CURRENT = "current"
    PREFETCH = "prefetch"


@dataclass(frozen=True)
class BlockInterval:
    """Half-open token interval covered by a KV block."""

    start_token: int
    end_token: int

    def __post_init__(self) -> None:
        if self.start_token < 0:
            raise ValueError("start_token must be non-negative")
        if self.end_token <= self.start_token:
            raise ValueError("end_token must be greater than start_token")

    @property
    def token_count(self) -> int:
        return self.end_token - self.start_token

    def contains(self, token_id: int) -> bool:
        return self.start_token <= token_id < self.end_token

    def to_dict(self) -> dict[str, int]:
        return {
            "start_token": self.start_token,
            "end_token": self.end_token,
            "token_count": self.token_count,
        }


@dataclass
class KVBlock:
    """Metadata for one simulated KV cache block.

    Phase 1 stores metadata only. It does not store real K/V tensors and does
    not implement attention.
    """

    block_id: int
    layer_id: int
    start_token: int | None = None
    end_token: int | None = None
    size_bytes: int = 0
    location: BlockLocation | str = BlockLocation.DRAM
    role: BlockRole | str = BlockRole.CURRENT
    representative: tuple[float, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    interval: BlockInterval | None = None

    def __post_init__(self) -> None:
        if self.interval is not None:
            if self.start_token is not None and self.start_token != self.interval.start_token:
                raise ValueError("start_token conflicts with interval")
            if self.end_token is not None and self.end_token != self.interval.end_token:
                raise ValueError("end_token conflicts with interval")
            self.start_token = self.interval.start_token
            self.end_token = self.interval.end_token
        elif self.start_token is not None and self.end_token is not None:
            self.interval = BlockInterval(self.start_token, self.end_token)
        else:
            raise ValueError("start_token and end_token are required")

        if self.block_id < 0:
            raise ValueError("block_id must be non-negative")
        if self.layer_id < 0:
            raise ValueError("layer_id must be non-negative")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")

        self.location = BlockLocation(self.location)
        self.role = BlockRole(self.role)

    @property
    def token_count(self) -> int:
        assert self.interval is not None
        return self.interval.token_count

    @property
    def key(self) -> tuple[int, int]:
        return (self.layer_id, self.block_id)

    def contains_token(self, token_id: int) -> bool:
        assert self.interval is not None
        return self.interval.contains(token_id)

    def is_resident(self, location: BlockLocation | str) -> bool:
        return self.location == BlockLocation(location)

    def move_to(self, location: BlockLocation | str) -> None:
        self.location = BlockLocation(location)

    def mark_role(self, role: BlockRole | str) -> None:
        self.role = BlockRole(role)

    def set_representative(self, vector: tuple[float, ...] | list[float]) -> None:
        self.representative = tuple(float(value) for value in vector)

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "layer_id": self.layer_id,
            "start_token": self.start_token,
            "end_token": self.end_token,
            "size_bytes": self.size_bytes,
            "location": self.location.value,
            "role": self.role.value,
            "representative": list(self.representative)
            if self.representative is not None
            else None,
            "metadata": dict(self.metadata),
        }
