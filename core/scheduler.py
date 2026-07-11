"""SSD-aware scheduling simulation for SolidAttention Phase 1.

The scheduler creates a metadata-only timeline for loading KV blocks from SSD or
DRAM into VRAM before simulated compute. It records trace and metrics but does
not perform real I/O, GPU transfer, or attention computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from core.block_store import BlockStore
from core.kv_block import BlockLocation, KVBlock
from harness.metrics import MetricsCollector
from harness.trace import TraceRecorder


@dataclass(frozen=True)
class SchedulerConfig:
    """Latency model for the simulation scheduler."""

    ssd_read_latency_ms: float = 4.0
    dram_to_vram_latency_ms: float = 1.0
    compute_latency_ms: float = 0.5
    compute_location: BlockLocation | str = BlockLocation.VRAM

    def __post_init__(self) -> None:
        BlockLocation(self.compute_location)
        if self.ssd_read_latency_ms < 0:
            raise ValueError("ssd_read_latency_ms must be non-negative")
        if self.dram_to_vram_latency_ms < 0:
            raise ValueError("dram_to_vram_latency_ms must be non-negative")
        if self.compute_latency_ms < 0:
            raise ValueError("compute_latency_ms must be non-negative")

    @property
    def normalized_compute_location(self) -> BlockLocation:
        return BlockLocation(self.compute_location)


@dataclass(frozen=True)
class ScheduleOp:
    """One simulated scheduler operation."""

    op: str
    layer_id: int
    block_id: int
    device: str
    start_ms: float
    end_ms: float
    metadata: dict[str, object]

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict[str, object]:
        return {
            "op": self.op,
            "layer_id": self.layer_id,
            "block_id": self.block_id,
            "device": self.device,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ScheduleResult:
    """Timeline produced for one decode step."""

    step: int
    ops: tuple[ScheduleOp, ...]

    @property
    def total_latency_ms(self) -> float:
        if not self.ops:
            return 0.0
        return max(op.end_ms for op in self.ops) - min(op.start_ms for op in self.ops)

    def to_list(self) -> list[dict[str, object]]:
        return [op.to_dict() for op in self.ops]


class SSDAwareScheduler:
    """Builds a simple SSD/DRAM/VRAM load and compute schedule."""

    def __init__(
        self,
        store: BlockStore,
        config: SchedulerConfig | None = None,
        trace: TraceRecorder | None = None,
        metrics: MetricsCollector | None = None,
    ) -> None:
        self.store = store
        self.config = config or SchedulerConfig()
        self.trace = trace
        self.metrics = metrics

    def schedule(
        self,
        blocks: Iterable[KVBlock],
        *,
        step: int,
        start_ms: float = 0.0,
    ) -> ScheduleResult:
        now = float(start_ms)
        ops: list[ScheduleOp] = []
        unique_blocks = self._dedupe(blocks)

        for block in unique_blocks:
            if block.location == BlockLocation.SSD:
                op = self._schedule_ssd_read(block, now)
                ops.append(op)
                now = op.end_ms
            if block.location == BlockLocation.DRAM:
                op = self._schedule_dram_to_vram(block, now)
                ops.append(op)
                now = op.end_ms
            if block.location != self.config.normalized_compute_location:
                raise ValueError(
                    f"block {(block.layer_id, block.block_id)} is in {block.location.value}, "
                    f"expected {self.config.normalized_compute_location.value} before compute"
                )
            op = self._schedule_compute(block, now)
            ops.append(op)
            now = op.end_ms

        return ScheduleResult(step=step, ops=tuple(ops))

    def schedule_selection_result(
        self,
        selection_result: object,
        *,
        step: int,
        start_ms: float = 0.0,
    ) -> ScheduleResult:
        blocks = getattr(selection_result, "all_blocks")
        return self.schedule(blocks, step=step, start_ms=start_ms)

    def _schedule_ssd_read(self, block: KVBlock, start_ms: float) -> ScheduleOp:
        end_ms = start_ms + self.config.ssd_read_latency_ms
        self.store.move_block(block.layer_id, block.block_id, BlockLocation.DRAM)
        op = self._make_op(
            "schedule_ssd_read",
            block,
            device="ssd",
            start_ms=start_ms,
            end_ms=end_ms,
            metadata={
                "from_location": "ssd",
                "to_location": "dram",
                "blocking": True,
                "size_bytes": block.size_bytes,
            },
        )
        if self.metrics is not None:
            self.metrics.add(
                total_latency_ms=self.config.ssd_read_latency_ms,
                gpu_wait_time_ms=self.config.ssd_read_latency_ms,
                load_blocking_latency_ms=self.config.ssd_read_latency_ms,
                ssd_read_bytes=block.size_bytes,
                io_op_count=1,
            )
        return op

    def _schedule_dram_to_vram(self, block: KVBlock, start_ms: float) -> ScheduleOp:
        end_ms = start_ms + self.config.dram_to_vram_latency_ms
        before = block.location
        self.store.move_block(block.layer_id, block.block_id, BlockLocation.VRAM)
        op = self._make_op(
            "schedule_dram_to_vram",
            block,
            device="dram",
            start_ms=start_ms,
            end_ms=end_ms,
            metadata={
                "from_location": before.value,
                "to_location": "vram",
                "blocking": True,
                "size_bytes": block.size_bytes,
            },
        )
        if self.metrics is not None:
            self.metrics.add(
                total_latency_ms=self.config.dram_to_vram_latency_ms,
                gpu_wait_time_ms=self.config.dram_to_vram_latency_ms,
                load_blocking_latency_ms=self.config.dram_to_vram_latency_ms,
            )
        return op

    def _schedule_compute(self, block: KVBlock, start_ms: float) -> ScheduleOp:
        end_ms = start_ms + self.config.compute_latency_ms
        op = self._make_op(
            "schedule_compute",
            block,
            device=self.config.normalized_compute_location.value,
            start_ms=start_ms,
            end_ms=end_ms,
            metadata={
                "location": block.location.value,
                "blocking": False,
                "simulated_attention": False,
            },
        )
        if self.metrics is not None:
            self.metrics.add(total_latency_ms=self.config.compute_latency_ms)
        return op

    def _make_op(
        self,
        op: str,
        block: KVBlock,
        *,
        device: str,
        start_ms: float,
        end_ms: float,
        metadata: dict[str, object],
    ) -> ScheduleOp:
        schedule_op = ScheduleOp(
            op=op,
            layer_id=block.layer_id,
            block_id=block.block_id,
            device=device,
            start_ms=start_ms,
            end_ms=end_ms,
            metadata=metadata,
        )
        if self.trace is not None:
            self.trace.record(
                op=op,
                layer_id=block.layer_id,
                block_id=block.block_id,
                device=device,
                start_ms=start_ms,
                end_ms=end_ms,
                metadata=metadata,
            )
        return schedule_op

    @staticmethod
    def _dedupe(blocks: Iterable[KVBlock]) -> list[KVBlock]:
        unique: list[KVBlock] = []
        seen: set[tuple[int, int]] = set()
        for block in blocks:
            if block.key in seen:
                continue
            unique.append(block)
            seen.add(block.key)
        return unique
