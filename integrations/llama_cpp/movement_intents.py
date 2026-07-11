"""Export llama.cpp bridge movement intents as executable movement plans."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from integrations.llama_cpp.command_bridge import infer_intent_kind, load_bridge_trace_events


INTENT_PLAN_CONTRACT = "solidattention.llama_cpp.movement_intent_plan.v1"
INTENT_EXPORT_CONTRACT = "solidattention.llama_cpp.movement_intent_export.v1"
SSD_READ_OPS = {"kv_load", "kv_prefetch"}


@dataclass(frozen=True)
class MovementIntentPlanExport:
    """Summary for a movement-intent plan exported from a bridge trace."""

    trace_path: str
    movement_intent_count: int
    skipped_non_intent_count: int
    command_counts_by_op: dict[str, int]
    intent_counts_by_kind: dict[str, int]
    intent_bytes_by_kind: dict[str, int]
    blocking_movement_intent_count: int
    nonblocking_movement_intent_count: int
    ssd_read_intent_count: int
    ssd_read_bytes: int
    ssd_write_intent_count: int
    ssd_write_bytes: int
    unique_block_count: int
    total_size_bytes: int
    plan_path: str | None = None
    summary_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": INTENT_EXPORT_CONTRACT,
            "trace_path": self.trace_path,
            "plan_path": self.plan_path,
            "summary_path": self.summary_path,
            "movement_intent_count": self.movement_intent_count,
            "skipped_non_intent_count": self.skipped_non_intent_count,
            "command_counts_by_op": dict(self.command_counts_by_op),
            "intent_counts_by_kind": dict(self.intent_counts_by_kind),
            "intent_bytes_by_kind": dict(self.intent_bytes_by_kind),
            "blocking_movement_intent_count": self.blocking_movement_intent_count,
            "nonblocking_movement_intent_count": self.nonblocking_movement_intent_count,
            "ssd_read_intent_count": self.ssd_read_intent_count,
            "ssd_read_bytes": self.ssd_read_bytes,
            "ssd_write_intent_count": self.ssd_write_intent_count,
            "ssd_write_bytes": self.ssd_write_bytes,
            "unique_block_count": self.unique_block_count,
            "total_size_bytes": self.total_size_bytes,
        }


def export_movement_intent_plan(
    trace_path: str | Path,
    *,
    plan_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    base_offset: int = 0,
    alignment: int = 4096,
) -> MovementIntentPlanExport:
    """Convert bridge movement intents into a MovementPlan-compatible JSON file."""

    trace_path = Path(trace_path)
    if base_offset < 0:
        raise ValueError("base_offset must be non-negative")
    if alignment <= 0:
        raise ValueError("alignment must be positive")

    events = load_bridge_trace_events(trace_path)
    ops: list[dict[str, Any]] = []
    layout_ranges: dict[tuple[int, int], dict[str, int]] = {}
    skipped_non_intent_count = 0

    for event_index, event in enumerate(events):
        metadata = event["metadata"]
        if not _as_bool(metadata.get("movement_intent", True)):
            skipped_non_intent_count += 1
            continue
        op = _movement_op_from_event(event, event_index)
        _upsert_layout_range(layout_ranges, op)
        ops.append(op)

    plan = {
        "contract": INTENT_PLAN_CONTRACT,
        "trace_path": str(trace_path),
        "layout": _layout_payload(layout_ranges, base_offset=base_offset, alignment=alignment),
        "ops": ops,
    }

    if plan_path is not None:
        output = Path(plan_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = _summarize_export(
        trace_path=trace_path,
        ops=ops,
        skipped_non_intent_count=skipped_non_intent_count,
        layout=plan["layout"],
        plan_path=plan_path,
        summary_path=summary_path,
    )
    if summary_path is not None:
        output = Path(summary_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _movement_op_from_event(event: dict[str, Any], event_index: int) -> dict[str, Any]:
    metadata = event["metadata"]
    missing = [field for field in ("command_op", "command_source", "command_target", "command_size_bytes", "ssd_offset") if field not in metadata]
    if missing:
        raise ValueError(f"bridge movement intent missing metadata fields: {missing}")

    layer_id = int(event["layer_id"])
    block_id = int(event["block_id"])
    command_op = str(metadata["command_op"])
    source = str(metadata["command_source"])
    target = str(metadata["command_target"])
    size_bytes = int(metadata["command_size_bytes"])
    ssd_offset = int(metadata["ssd_offset"])
    start_token = int(metadata.get("command_start_token", event["start_token"]))
    end_token = int(metadata.get("command_end_token", event["end_token"]))
    blocking = _as_bool(metadata.get("blocking", False))
    intent_kind = str(metadata.get("intent_kind", infer_intent_kind(command_op, source, target)))

    if size_bytes <= 0:
        raise ValueError("command_size_bytes must be positive")
    if ssd_offset < 0:
        raise ValueError("ssd_offset must be non-negative")
    if end_token <= start_token:
        raise ValueError("movement intent end_token must be greater than start_token")

    # Bridge traces are metadata observations rather than measured transfer spans.
    # Use a zero-duration ordered timestamp so the plan stays executable.
    timestamp_ms = float(event_index)
    op_metadata = dict(metadata)
    op_metadata.update(
        {
            "bridge_event_index": event_index,
            "bridge_trace_op": event.get("op"),
            "intent_kind": intent_kind,
            "movement_intent_contract": INTENT_PLAN_CONTRACT,
        }
    )
    return {
        "op": command_op,
        "layer_id": layer_id,
        "block_id": block_id,
        "source": source,
        "target": target,
        "ssd_offset": ssd_offset,
        "size_bytes": size_bytes,
        "start_token": start_token,
        "end_token": end_token,
        "start_ms": timestamp_ms,
        "end_ms": timestamp_ms,
        "duration_ms": 0.0,
        "blocking": blocking,
        "metadata": op_metadata,
    }


def _upsert_layout_range(layout_ranges: dict[tuple[int, int], dict[str, int]], op: dict[str, Any]) -> None:
    key = (int(op["layer_id"]), int(op["block_id"]))
    block_range = {
        "layer_id": key[0],
        "block_id": key[1],
        "offset": int(op["ssd_offset"]),
        "size_bytes": int(op["size_bytes"]),
        "end_offset": int(op["ssd_offset"]) + int(op["size_bytes"]),
    }
    existing = layout_ranges.get(key)
    if existing is None:
        layout_ranges[key] = block_range
        return
    if existing["offset"] != block_range["offset"] or existing["size_bytes"] != block_range["size_bytes"]:
        raise ValueError(f"conflicting SSD range for movement intent block: {key}")


def _layout_payload(layout_ranges: dict[tuple[int, int], dict[str, int]], *, base_offset: int, alignment: int) -> dict[str, Any]:
    blocks = sorted(layout_ranges.values(), key=lambda item: (item["layer_id"], item["block_id"]))
    total_size_bytes = 0
    if blocks:
        total_size_bytes = max(item["end_offset"] for item in blocks) - base_offset
    return {
        "base_offset": base_offset,
        "alignment": alignment,
        "total_size_bytes": total_size_bytes,
        "blocks": blocks,
    }


def _summarize_export(
    *,
    trace_path: Path,
    ops: list[dict[str, Any]],
    skipped_non_intent_count: int,
    layout: dict[str, Any],
    plan_path: str | Path | None,
    summary_path: str | Path | None,
) -> MovementIntentPlanExport:
    command_counts: dict[str, int] = {}
    intent_counts: dict[str, int] = {}
    intent_bytes: dict[str, int] = {}
    blocking_count = 0
    nonblocking_count = 0
    ssd_read_count = 0
    ssd_write_count = 0
    ssd_read_bytes = 0
    ssd_write_bytes = 0

    for op in ops:
        command_op = str(op["op"])
        source = str(op["source"])
        target = str(op["target"])
        size_bytes = int(op["size_bytes"])
        intent_kind = str(op.get("metadata", {}).get("intent_kind", infer_intent_kind(command_op, source, target)))
        command_counts[command_op] = command_counts.get(command_op, 0) + 1
        intent_counts[intent_kind] = intent_counts.get(intent_kind, 0) + 1
        intent_bytes[intent_kind] = intent_bytes.get(intent_kind, 0) + size_bytes
        if _as_bool(op.get("blocking", False)):
            blocking_count += 1
        else:
            nonblocking_count += 1
        if intent_kind == "ssd_read" or (command_op in SSD_READ_OPS and source == "ssd"):
            ssd_read_count += 1
            ssd_read_bytes += size_bytes
        if intent_kind == "ssd_write" or (command_op == "kv_evict" and target == "ssd"):
            ssd_write_count += 1
            ssd_write_bytes += size_bytes

    return MovementIntentPlanExport(
        trace_path=str(trace_path),
        plan_path=str(plan_path) if plan_path is not None else None,
        summary_path=str(summary_path) if summary_path is not None else None,
        movement_intent_count=len(ops),
        skipped_non_intent_count=skipped_non_intent_count,
        command_counts_by_op=dict(sorted(command_counts.items())),
        intent_counts_by_kind=dict(sorted(intent_counts.items())),
        intent_bytes_by_kind=dict(sorted(intent_bytes.items())),
        blocking_movement_intent_count=blocking_count,
        nonblocking_movement_intent_count=nonblocking_count,
        ssd_read_intent_count=ssd_read_count,
        ssd_read_bytes=ssd_read_bytes,
        ssd_write_intent_count=ssd_write_count,
        ssd_write_bytes=ssd_write_bytes,
        unique_block_count=len(layout.get("blocks", [])),
        total_size_bytes=int(layout.get("total_size_bytes", 0)),
    )


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n", ""}:
            return False
    return bool(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--base-offset", type=int, default=0)
    parser.add_argument("--alignment", type=int, default=4096)
    args = parser.parse_args()

    result = export_movement_intent_plan(
        args.trace,
        plan_path=args.plan,
        summary_path=args.summary,
        base_offset=args.base_offset,
        alignment=args.alignment,
    )
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
