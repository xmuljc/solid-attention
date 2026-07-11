"""Summarize patched llama.cpp KV-cache JSONL traces."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from integrations.llama_cpp.adapter import LlamaKVCacheAdapter, LlamaKVEvent, load_jsonl_events
from harness.trace import TraceRecorder


@dataclass(frozen=True)
class RuntimeTraceSummary:
    trace_path: str
    event_count: int
    unique_block_count: int
    duplicate_event_count: int
    total_event_size_bytes: int
    layers: tuple[int, ...]
    ops: tuple[str, ...]
    location_stats: dict[str, dict[str, int]]
    harness_trace_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "trace_path": self.trace_path,
            "event_count": self.event_count,
            "unique_block_count": self.unique_block_count,
            "duplicate_event_count": self.duplicate_event_count,
            "total_event_size_bytes": self.total_event_size_bytes,
            "layers": list(self.layers),
            "ops": list(self.ops),
            "location_stats": self.location_stats,
            "harness_trace_path": self.harness_trace_path,
        }


def summarize_runtime_trace(
    trace_path: str | Path,
    *,
    summary_path: str | Path | None = None,
    harness_trace_path: str | Path | None = None,
) -> RuntimeTraceSummary:
    trace_path = Path(trace_path)
    events = load_jsonl_events(trace_path)
    trace = TraceRecorder()
    adapter = LlamaKVCacheAdapter(trace=trace)
    adapter.ingest_many(events, duplicate_policy="replace")

    unique_keys = {event.key for event in events}
    summary = RuntimeTraceSummary(
        trace_path=str(trace_path),
        event_count=len(events),
        unique_block_count=len(unique_keys),
        duplicate_event_count=len(events) - len(unique_keys),
        total_event_size_bytes=sum(event.size_bytes for event in events),
        layers=tuple(sorted({event.layer_id for event in events})),
        ops=tuple(sorted({event.op for event in events})),
        location_stats=adapter.store.location_stats(),
        harness_trace_path=str(harness_trace_path) if harness_trace_path is not None else None,
    )

    if harness_trace_path is not None:
        trace.save_json(harness_trace_path)
    if summary_path is not None:
        output = Path(summary_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--harness-trace", default=None)
    args = parser.parse_args()

    summary = summarize_runtime_trace(
        args.trace,
        summary_path=args.summary,
        harness_trace_path=args.harness_trace,
    )
    print(json.dumps(summary.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
