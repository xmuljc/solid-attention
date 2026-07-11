"""JSON metrics collection for SolidAttention Phase 1 simulations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_METRIC_FIELDS = (
    "total_latency_ms",
    "gpu_wait_time_ms",
    "load_blocking_latency_ms",
    "ssd_read_bytes",
    "ssd_write_bytes",
    "io_op_count",
    "selected_block_count",
    "prefetch_hit_count",
    "prefetch_miss_count",
    "wrong_prefetch_count",
)


@dataclass
class MetricsCollector:
    """Collects fixed simulation metrics and derived prefetch rates."""

    metadata: dict[str, Any] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    gpu_wait_time_ms: float = 0.0
    load_blocking_latency_ms: float = 0.0
    ssd_read_bytes: int = 0
    ssd_write_bytes: int = 0
    io_op_count: int = 0
    selected_block_count: int = 0
    prefetch_hit_count: int = 0
    prefetch_miss_count: int = 0
    wrong_prefetch_count: int = 0

    def add(
        self,
        *,
        total_latency_ms: float = 0.0,
        gpu_wait_time_ms: float = 0.0,
        load_blocking_latency_ms: float = 0.0,
        ssd_read_bytes: int = 0,
        ssd_write_bytes: int = 0,
        io_op_count: int = 0,
        selected_block_count: int = 0,
        prefetch_hit_count: int = 0,
        prefetch_miss_count: int = 0,
        wrong_prefetch_count: int = 0,
    ) -> None:
        self.total_latency_ms += float(total_latency_ms)
        self.gpu_wait_time_ms += float(gpu_wait_time_ms)
        self.load_blocking_latency_ms += float(load_blocking_latency_ms)
        self.ssd_read_bytes += int(ssd_read_bytes)
        self.ssd_write_bytes += int(ssd_write_bytes)
        self.io_op_count += int(io_op_count)
        self.selected_block_count += int(selected_block_count)
        self.prefetch_hit_count += int(prefetch_hit_count)
        self.prefetch_miss_count += int(prefetch_miss_count)
        self.wrong_prefetch_count += int(wrong_prefetch_count)

    @property
    def prefetch_attempt_count(self) -> int:
        return self.prefetch_hit_count + self.prefetch_miss_count

    @property
    def prefetch_hit_rate(self) -> float:
        return self._rate(self.prefetch_hit_count, self.prefetch_attempt_count)

    @property
    def prefetch_miss_rate(self) -> float:
        return self._rate(self.prefetch_miss_count, self.prefetch_attempt_count)

    @property
    def wrong_prefetch_rate(self) -> float:
        return self._rate(self.wrong_prefetch_count, self.prefetch_attempt_count)

    def to_dict(self) -> dict[str, Any]:
        payload = {field_name: getattr(self, field_name) for field_name in _METRIC_FIELDS}
        payload.update(
            {
                "prefetch_attempt_count": self.prefetch_attempt_count,
                "prefetch_hit_rate": self.prefetch_hit_rate,
                "prefetch_miss_rate": self.prefetch_miss_rate,
                "wrong_prefetch_rate": self.wrong_prefetch_rate,
                "metadata": dict(self.metadata),
            }
        )
        return payload

    def save_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _rate(count: int, total: int) -> float:
        if total == 0:
            return 0.0
        return count / total

    # Compatibility helpers from the first skeleton.
    def set_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def set_scalar(self, key: str, value: float | int) -> None:
        if key not in _METRIC_FIELDS:
            raise KeyError(f"unknown metric field: {key}")
        setattr(self, key, float(value) if key.endswith("_ms") else int(value))

    def increment(self, key: str, amount: int = 1) -> None:
        if key not in _METRIC_FIELDS:
            raise KeyError(f"unknown metric field: {key}")
        setattr(self, key, getattr(self, key) + int(amount))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def write_json(self, path: str | Path) -> None:
        self.save_json(path)
