"""Adapter from llama.cpp KV-cache trace events to harness blocks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from core.block_store import BlockStore
from core.kv_block import BlockLocation, KVBlock
from harness.trace import TraceRecorder


DuplicatePolicy = Literal["strict", "replace"]
MOVEMENT_OPS = {"kv_load", "kv_prefetch", "kv_evict"}


@dataclass(frozen=True)
class LlamaKVEvent:
    """Minimal KV-cache event expected from llama.cpp instrumentation."""

    op: str
    layer_id: int
    block_id: int
    start_token: int
    end_token: int
    size_bytes: int
    location: str
    source: str | None = None
    target: str | None = None
    ssd_offset: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[int, int]:
        return (self.layer_id, self.block_id)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LlamaKVEvent":
        ssd_offset = payload.get("ssd_offset")
        return cls(
            op=str(payload["op"]),
            layer_id=int(payload["layer_id"]),
            block_id=int(payload["block_id"]),
            start_token=int(payload["start_token"]),
            end_token=int(payload["end_token"]),
            size_bytes=int(payload["size_bytes"]),
            location=str(payload["location"]),
            source=str(payload["source"]) if payload.get("source") is not None else None,
            target=str(payload["target"]) if payload.get("target") is not None else None,
            ssd_offset=int(ssd_offset) if ssd_offset is not None else None,
            metadata=dict(payload.get("metadata", {})),
        )


def load_jsonl_events(path: str | Path) -> list[LlamaKVEvent]:
    """Load one llama.cpp KV event per JSONL line."""

    events: list[LlamaKVEvent] = []
    trace_path = Path(path)
    for line_number, line in enumerate(trace_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("event line must contain a JSON object")
            events.append(LlamaKVEvent.from_dict(payload))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid llama.cpp KV trace event at line {line_number}: {exc}") from exc
    return events


class LlamaKVCacheAdapter:
    """Maintains a BlockStore from llama.cpp-style KV trace events."""

    def __init__(self, store: BlockStore | None = None, trace: TraceRecorder | None = None) -> None:
        self.store = store or BlockStore()
        self.trace = trace

    def ingest(
        self,
        payload: dict[str, Any] | LlamaKVEvent,
        *,
        duplicate_policy: DuplicatePolicy = "strict",
    ) -> KVBlock:
        event = payload if isinstance(payload, LlamaKVEvent) else LlamaKVEvent.from_dict(payload)
        if duplicate_policy not in {"strict", "replace"}:
            raise ValueError(f"unsupported duplicate policy: {duplicate_policy}")
        if event.op == "kv_add":
            replaced_existing = self.store.has_block(event.layer_id, event.block_id)
            block = KVBlock(
                block_id=event.block_id,
                layer_id=event.layer_id,
                start_token=event.start_token,
                end_token=event.end_token,
                size_bytes=event.size_bytes,
                location=event.location,
                metadata=self._event_metadata(event),
            )
            if duplicate_policy == "replace":
                self.store.upsert_block(block)
            else:
                self.store.add_block(block)
            self._trace("llama_kv_add", block, event, duplicate_policy, replaced_existing)
            return block
        if event.op == "kv_move":
            target_location = event.target or event.location
            block = self.store.move_block(event.layer_id, event.block_id, target_location)
            self._trace("llama_kv_move", block, event, duplicate_policy, False)
            return block
        if event.op in MOVEMENT_OPS:
            replaced_existing = self.store.has_block(event.layer_id, event.block_id)
            target_location = BlockLocation(event.target or event.location)
            block = KVBlock(
                block_id=event.block_id,
                layer_id=event.layer_id,
                start_token=event.start_token,
                end_token=event.end_token,
                size_bytes=event.size_bytes,
                location=target_location,
                metadata=self._event_metadata(event),
            )
            self.store.upsert_block(block)
            self._trace(f"llama_{event.op}", block, event, duplicate_policy, replaced_existing)
            return block
        raise ValueError(f"unsupported llama kv event op: {event.op}")

    def ingest_many(
        self,
        events: Iterable[dict[str, Any] | LlamaKVEvent],
        *,
        duplicate_policy: DuplicatePolicy = "strict",
    ) -> list[KVBlock]:
        return [self.ingest(event, duplicate_policy=duplicate_policy) for event in events]

    def ingest_jsonl(
        self,
        path: str | Path,
        *,
        duplicate_policy: DuplicatePolicy = "strict",
    ) -> list[KVBlock]:
        return self.ingest_many(load_jsonl_events(path), duplicate_policy=duplicate_policy)

    def _trace(
        self,
        op: str,
        block: KVBlock,
        event: LlamaKVEvent,
        duplicate_policy: DuplicatePolicy,
        replaced_existing: bool,
    ) -> None:
        if self.trace is None:
            return
        metadata = self._event_metadata(event)
        metadata.update(
            {
                "source": "llama.cpp",
                "llama_op": event.op,
                "start_token": event.start_token,
                "end_token": event.end_token,
                "size_bytes": event.size_bytes,
                "location": event.location,
                "duplicate_policy": duplicate_policy,
                "replaced_existing": replaced_existing,
            }
        )
        self.trace.record(
            op=op,
            layer_id=block.layer_id,
            block_id=block.block_id,
            device=block.location.value,
            start_ms=0.0,
            end_ms=0.0,
            metadata=metadata,
        )

    @staticmethod
    def _event_metadata(event: LlamaKVEvent) -> dict[str, Any]:
        metadata = {
            "source": "llama.cpp",
            "llama_op": event.op,
            "start_token": event.start_token,
            "end_token": event.end_token,
            "size_bytes": event.size_bytes,
            "location": event.location,
            "event_source": event.source,
            "event_target": event.target,
            "ssd_offset": event.ssd_offset,
        }
        metadata.update(event.metadata)
        return metadata
