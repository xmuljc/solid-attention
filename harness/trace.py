"""Structured trace recording for SolidAttention Phase 1 simulations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceEvent:
    """One simulated operation trace event."""

    op: str
    layer_id: int
    block_id: int
    device: str
    start_ms: float
    end_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be greater than or equal to start_ms")

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict[str, Any]:
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


class TraceRecorder:
    """Collects trace events and saves them as JSON."""

    def __init__(self) -> None:
        self._events: list[TraceEvent] = []

    def record(
        self,
        op: str,
        layer_id: int,
        block_id: int,
        device: str,
        start_ms: float,
        end_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> TraceEvent:
        event = TraceEvent(
            op=op,
            layer_id=layer_id,
            block_id=block_id,
            device=device,
            start_ms=float(start_ms),
            end_ms=float(end_ms),
            metadata=dict(metadata or {}),
        )
        self._events.append(event)
        return event

    @property
    def events(self) -> list[TraceEvent]:
        return list(self._events)

    def to_list(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self._events]

    def save_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_list(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def clear(self) -> None:
        self._events.clear()

    # Compatibility helpers from the first skeleton.
    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(event, sort_keys=True) for event in self.to_list())

    def write_jsonl(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        text = self.to_jsonl()
        if text:
            text += "\n"
        output.write_text(text, encoding="utf-8")
