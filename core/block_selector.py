"""Block selection simulation for SolidAttention Phase 1.

This module implements metadata-only selection of Init Blocks, Local Blocks, and
Selected Blocks. It does not compute real attention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from core.kv_block import BlockRole, KVBlock
from harness.trace import TraceRecorder


@dataclass(frozen=True)
class SelectionConfig:
    """Controls how many blocks each simulated policy may choose."""

    init_block_count: int = 1
    local_block_count: int = 1
    selected_block_count: int = 1

    def __post_init__(self) -> None:
        if self.init_block_count < 0:
            raise ValueError("init_block_count must be non-negative")
        if self.local_block_count < 0:
            raise ValueError("local_block_count must be non-negative")
        if self.selected_block_count < 0:
            raise ValueError("selected_block_count must be non-negative")


@dataclass(frozen=True)
class SelectionResult:
    """Selected block groups for one layer and decode step."""

    init_blocks: tuple[KVBlock, ...]
    local_blocks: tuple[KVBlock, ...]
    selected_blocks: tuple[KVBlock, ...]

    @property
    def all_blocks(self) -> tuple[KVBlock, ...]:
        ordered: list[KVBlock] = []
        seen: set[tuple[int, int]] = set()
        for group in (self.init_blocks, self.local_blocks, self.selected_blocks):
            for block in group:
                if block.key not in seen:
                    ordered.append(block)
                    seen.add(block.key)
        return tuple(ordered)

    def to_dict(self) -> dict[str, list[int]]:
        return {
            "init_blocks": [block.block_id for block in self.init_blocks],
            "local_blocks": [block.block_id for block in self.local_blocks],
            "selected_blocks": [block.block_id for block in self.selected_blocks],
            "all_blocks": [block.block_id for block in self.all_blocks],
        }


class BlockSelector:
    """Simulates SolidAttention-style block selection.

    Init blocks are earliest blocks in the layer. Local blocks are the latest
    blocks before the current token. Selected blocks are chosen from remaining
    candidates by representative-vector cosine similarity.
    """

    def __init__(self, config: SelectionConfig, trace: TraceRecorder | None = None) -> None:
        self.config = config
        self.trace = trace

    def select(
        self,
        blocks: Iterable[KVBlock],
        *,
        layer_id: int,
        current_token: int,
        query_vector: tuple[float, ...] | list[float] | None = None,
        start_ms: float = 0.0,
    ) -> SelectionResult:
        layer_blocks = sorted(
            [block for block in blocks if block.layer_id == layer_id and block.end_token <= current_token],
            key=lambda block: (block.start_token if block.start_token is not None else -1, block.block_id),
        )

        init_blocks = tuple(layer_blocks[: self.config.init_block_count])
        init_keys = {block.key for block in init_blocks}

        local_candidates = [block for block in layer_blocks if block.key not in init_keys]
        local_blocks = tuple(local_candidates[-self.config.local_block_count :]) if self.config.local_block_count else ()
        local_keys = {block.key for block in local_blocks}

        selected_candidates = [
            block
            for block in layer_blocks
            if block.key not in init_keys and block.key not in local_keys
        ]
        ranked = self._rank_by_representative(selected_candidates, query_vector)
        selected_blocks = tuple(block for block, _score in ranked[: self.config.selected_block_count])
        selected_scores = {block.key: score for block, score in ranked}

        for block in init_blocks:
            block.mark_role(BlockRole.INIT)
            self._trace(block, "select_init", "init", 1.0, start_ms)
        for block in local_blocks:
            block.mark_role(BlockRole.LOCAL)
            self._trace(block, "select_local", "local", 1.0, start_ms)
        for block in selected_blocks:
            block.mark_role(BlockRole.SELECTED)
            self._trace(block, "select_representative", "selected", selected_scores[block.key], start_ms)

        return SelectionResult(
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            selected_blocks=selected_blocks,
        )

    def _rank_by_representative(
        self,
        blocks: Iterable[KVBlock],
        query_vector: tuple[float, ...] | list[float] | None,
    ) -> list[tuple[KVBlock, float]]:
        scored = []
        for block in blocks:
            score = representative_score(block, query_vector)
            scored.append((block, score))
        return sorted(
            scored,
            key=lambda item: (-item[1], item[0].start_token if item[0].start_token is not None else 0, item[0].block_id),
        )

    def _trace(
        self,
        block: KVBlock,
        op: str,
        reason: str,
        score: float,
        start_ms: float,
    ) -> None:
        if self.trace is None:
            return
        self.trace.record(
            op=op,
            layer_id=block.layer_id,
            block_id=block.block_id,
            device=block.location.value,
            start_ms=start_ms,
            end_ms=start_ms,
            metadata={
                "role": block.role.value,
                "reason": reason,
                "score": score,
                "start_token": block.start_token,
                "end_token": block.end_token,
            },
        )


def representative_score(
    block: KVBlock,
    query_vector: tuple[float, ...] | list[float] | None,
) -> float:
    """Return cosine similarity between a block representative and query vector."""

    if query_vector is None or block.representative is None:
        return 0.0
    query = tuple(float(value) for value in query_vector)
    representative = tuple(float(value) for value in block.representative)
    if len(query) != len(representative):
        raise ValueError("query_vector and representative must have the same length")
    query_norm = math.sqrt(sum(value * value for value in query))
    rep_norm = math.sqrt(sum(value * value for value in representative))
    if query_norm == 0.0 or rep_norm == 0.0:
        return 0.0
    dot = sum(left * right for left, right in zip(query, representative))
    return dot / (query_norm * rep_norm)
