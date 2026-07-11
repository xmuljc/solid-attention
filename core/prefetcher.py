"""Speculative prefetch simulation for SolidAttention Phase 1.

The prefetcher is metadata-only: it moves KVBlock location state inside a
BlockStore and records trace/metrics. It does not perform real SSD I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from core.block_store import BlockStore
from core.kv_block import BlockLocation, BlockRole, KVBlock
from harness.metrics import MetricsCollector
from harness.trace import TraceRecorder


BlockKey = tuple[int, int]


@dataclass(frozen=True)
class PrefetchConfig:
    """Configuration for simulated speculative prefetch."""

    target_location: BlockLocation | str = BlockLocation.DRAM
    mock_latency_ms: float = 0.0
    max_prefetch_blocks: int | None = None

    def __post_init__(self) -> None:
        BlockLocation(self.target_location)
        if self.mock_latency_ms < 0:
            raise ValueError("mock_latency_ms must be non-negative")
        if self.max_prefetch_blocks is not None and self.max_prefetch_blocks < 0:
            raise ValueError("max_prefetch_blocks must be non-negative")

    @property
    def normalized_target(self) -> BlockLocation:
        return BlockLocation(self.target_location)


@dataclass(frozen=True)
class PrefetchPlan:
    """Blocks predicted for speculative prefetch at one decode step."""

    step: int
    target_location: BlockLocation
    blocks: tuple[KVBlock, ...]

    @property
    def block_keys(self) -> tuple[BlockKey, ...]:
        return tuple(block.key for block in self.blocks)

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "target_location": self.target_location.value,
            "block_keys": [list(key) for key in self.block_keys],
        }


@dataclass(frozen=True)
class PrefetchEvaluation:
    """Hit/miss/wrong-prefetch result for one required block set."""

    hits: tuple[KVBlock, ...]
    misses: tuple[KVBlock, ...]
    wrong_prefetches: tuple[KVBlock, ...]

    def to_dict(self) -> dict[str, list[list[int]]]:
        return {
            "hits": [list(block.key) for block in self.hits],
            "misses": [list(block.key) for block in self.misses],
            "wrong_prefetches": [list(block.key) for block in self.wrong_prefetches],
        }


class SpeculativePrefetcher:
    """Predicts, executes, and evaluates simulated block prefetches."""

    def __init__(
        self,
        store: BlockStore,
        config: PrefetchConfig | None = None,
        trace: TraceRecorder | None = None,
        metrics: MetricsCollector | None = None,
    ) -> None:
        self.store = store
        self.config = config or PrefetchConfig()
        self.trace = trace
        self.metrics = metrics

    def plan(
        self,
        predicted_blocks: Iterable[KVBlock],
        *,
        step: int,
        start_ms: float = 0.0,
    ) -> PrefetchPlan:
        blocks = self._dedupe(predicted_blocks)
        if self.config.max_prefetch_blocks is not None:
            blocks = blocks[: self.config.max_prefetch_blocks]
        plan = PrefetchPlan(
            step=step,
            target_location=self.config.normalized_target,
            blocks=tuple(blocks),
        )
        for block in plan.blocks:
            block.mark_role(BlockRole.PREFETCH)
            self._record(
                "prefetch_submit",
                block,
                start_ms,
                start_ms,
                {
                    "step": step,
                    "target_location": plan.target_location.value,
                },
            )
        return plan

    def execute(self, plan: PrefetchPlan, *, start_ms: float = 0.0) -> None:
        end_ms = start_ms + self.config.mock_latency_ms
        for block in plan.blocks:
            before = block.location
            self.store.move_block(block.layer_id, block.block_id, plan.target_location)
            self._record(
                "prefetch_move",
                block,
                start_ms,
                end_ms,
                {
                    "step": plan.step,
                    "from_location": before.value,
                    "to_location": plan.target_location.value,
                    "size_bytes": block.size_bytes,
                },
            )
            if self.metrics is not None and before == BlockLocation.SSD:
                self.metrics.add(
                    total_latency_ms=self.config.mock_latency_ms,
                    ssd_read_bytes=block.size_bytes,
                    io_op_count=1,
                )

    def evaluate(
        self,
        plan: PrefetchPlan,
        required_blocks: Iterable[KVBlock],
        *,
        step: int,
        start_ms: float = 0.0,
    ) -> PrefetchEvaluation:
        required = self._dedupe(required_blocks)
        planned_by_key = {block.key: block for block in plan.blocks}
        required_by_key = {block.key: block for block in required}

        hits = tuple(
            block
            for key, block in required_by_key.items()
            if key in planned_by_key and block.is_resident(plan.target_location)
        )
        misses = tuple(
            block
            for key, block in required_by_key.items()
            if key not in planned_by_key or not block.is_resident(plan.target_location)
        )
        wrong_prefetches = tuple(
            block for key, block in planned_by_key.items() if key not in required_by_key
        )

        for block in hits:
            self._record("prefetch_hit", block, start_ms, start_ms, {"step": step})
        for block in misses:
            self._record(
                "prefetch_miss",
                block,
                start_ms,
                start_ms + self.config.mock_latency_ms,
                {"step": step, "blocking_load": True},
            )
        for block in wrong_prefetches:
            self._record("prefetch_wrong", block, start_ms, start_ms, {"step": step})

        if self.metrics is not None:
            self.metrics.add(
                load_blocking_latency_ms=len(misses) * self.config.mock_latency_ms,
                prefetch_hit_count=len(hits),
                prefetch_miss_count=len(misses),
                wrong_prefetch_count=len(wrong_prefetches),
            )
        return PrefetchEvaluation(
            hits=hits,
            misses=misses,
            wrong_prefetches=wrong_prefetches,
        )

    def plan_from_selection_result(
        self,
        selection_result: object,
        *,
        step: int,
        start_ms: float = 0.0,
    ) -> PrefetchPlan:
        selected_blocks = getattr(selection_result, "selected_blocks")
        return self.plan(selected_blocks, step=step, start_ms=start_ms)

    @staticmethod
    def _dedupe(blocks: Iterable[KVBlock]) -> list[KVBlock]:
        seen: set[BlockKey] = set()
        unique: list[KVBlock] = []
        for block in blocks:
            if block.key in seen:
                continue
            unique.append(block)
            seen.add(block.key)
        return unique

    def _record(
        self,
        op: str,
        block: KVBlock,
        start_ms: float,
        end_ms: float,
        metadata: dict[str, object],
    ) -> None:
        if self.trace is None:
            return
        self.trace.record(
            op=op,
            layer_id=block.layer_id,
            block_id=block.block_id,
            device=block.location.value,
            start_ms=start_ms,
            end_ms=end_ms,
            metadata=metadata,
        )
