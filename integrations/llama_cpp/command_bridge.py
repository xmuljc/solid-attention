"""Summarize llama.cpp SolidAttention command bridge traces."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


BRIDGE_CONTRACT = "solidattention.llama_cpp.kv_command_bridge_summary.v1"
SSD_READ_OPS = {"kv_load", "kv_prefetch"}


@dataclass(frozen=True)
class CommandBridgeTraceSummary:
    """Metrics derived from kv_command_bridge runtime trace events."""

    trace_path: str
    event_count: int
    layers: tuple[int, ...]
    block_ids: tuple[int, ...]
    command_counts_by_op: dict[str, int]
    ssd_read_command_count: int
    ssd_write_command_count: int
    ssd_read_bytes: int
    ssd_write_bytes: int
    movement_intent_count: int
    blocking_movement_intent_count: int
    nonblocking_movement_intent_count: int
    intent_counts_by_kind: dict[str, int]
    intent_bytes_by_kind: dict[str, int]
    executor_enabled: bool
    executor_backend: str
    executor_metadata_only_count: int
    executor_ssd_read_attempt_count: int
    executor_ssd_read_success_count: int
    executor_ssd_read_error_count: int
    executor_open_error_count: int
    executor_seek_error_count: int
    executor_short_read_count: int
    executor_alloc_error_count: int
    executor_backend_error_count: int
    executor_ssd_read_bytes: int
    executor_staging_record_count: int
    executor_dram_buffer_count: int
    executor_dram_buffer_bytes: int
    executor_vram_buffer_count: int
    executor_vram_buffer_bytes: int
    executor_lazy_source_seed_count: int
    executor_lazy_source_seed_bytes: int
    executor_dram_to_vram_copy_count: int
    executor_dram_to_vram_copy_bytes: int
    executor_compute_resident_count: int
    executor_compute_resident_bytes: int
    executor_evict_count: int
    executor_evict_bytes: int
    executor_missing_source_count: int
    executor_size_mismatch_count: int
    executor_staging_overflow_count: int
    executor_byte_range_block_count: int
    executor_byte_range_count: int
    executor_byte_range_mapped_bytes: int
    executor_byte_range_key_bytes: int
    executor_byte_range_value_bytes: int
    executor_byte_range_inferred_row_size_count: int
    executor_byte_range_size_mismatch_count: int
    executor_byte_range_tensor_bounds_error_count: int
    executor_tensor_dry_run_enabled: bool
    executor_tensor_dry_run_block_count: int
    executor_tensor_dry_run_range_count: int
    executor_tensor_dry_run_bytes: int
    executor_tensor_dry_run_key_bytes: int
    executor_tensor_dry_run_value_bytes: int
    executor_tensor_dry_run_checksum_count: int
    executor_tensor_dry_run_checksum_xor: int
    executor_tensor_dry_run_size_mismatch_count: int
    executor_tensor_dry_run_tensor_bounds_error_count: int
    executor_tensor_mutation_enabled: bool
    executor_tensor_mutation_attempt_count: int
    executor_tensor_mutation_block_count: int
    executor_tensor_mutation_range_count: int
    executor_tensor_mutation_bytes: int
    executor_tensor_mutation_key_bytes: int
    executor_tensor_mutation_value_bytes: int
    executor_tensor_mutation_checksum_count: int
    executor_tensor_mutation_staging_checksum_xor: int
    executor_tensor_mutation_post_checksum_xor: int
    executor_tensor_mutation_verify_error_count: int
    executor_tensor_mutation_size_mismatch_count: int
    executor_tensor_mutation_tensor_bounds_error_count: int
    executor_tensor_mutation_unsupported_count: int
    attention_gate_enabled: bool
    attention_gate_view_count: int
    attention_gate_k_view_count: int
    attention_gate_v_view_count: int
    attention_gate_no_required_view_count: int
    attention_gate_required_block_count: int
    attention_gate_ready_block_count: int
    attention_gate_missing_block_count: int
    attention_gate_pending_mutation_count: int
    shadow_transition_count: int
    shadow_source_match_count: int
    shadow_source_mismatch_count: int
    shadow_final_locations: dict[str, int]
    unique_runtime_block_count: int
    summary_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": BRIDGE_CONTRACT,
            "trace_path": self.trace_path,
            "summary_path": self.summary_path,
            "event_count": self.event_count,
            "layers": list(self.layers),
            "block_ids": list(self.block_ids),
            "command_counts_by_op": dict(self.command_counts_by_op),
            "ssd_read_command_count": self.ssd_read_command_count,
            "ssd_write_command_count": self.ssd_write_command_count,
            "ssd_read_bytes": self.ssd_read_bytes,
            "ssd_write_bytes": self.ssd_write_bytes,
            "movement_intent_count": self.movement_intent_count,
            "blocking_movement_intent_count": self.blocking_movement_intent_count,
            "nonblocking_movement_intent_count": self.nonblocking_movement_intent_count,
            "intent_counts_by_kind": dict(self.intent_counts_by_kind),
            "intent_bytes_by_kind": dict(self.intent_bytes_by_kind),
            "executor_enabled": self.executor_enabled,
            "executor_backend": self.executor_backend,
            "executor_metadata_only_count": self.executor_metadata_only_count,
            "executor_ssd_read_attempt_count": self.executor_ssd_read_attempt_count,
            "executor_ssd_read_success_count": self.executor_ssd_read_success_count,
            "executor_ssd_read_error_count": self.executor_ssd_read_error_count,
            "executor_open_error_count": self.executor_open_error_count,
            "executor_seek_error_count": self.executor_seek_error_count,
            "executor_short_read_count": self.executor_short_read_count,
            "executor_alloc_error_count": self.executor_alloc_error_count,
            "executor_backend_error_count": self.executor_backend_error_count,
            "executor_ssd_read_bytes": self.executor_ssd_read_bytes,
            "executor_staging_record_count": self.executor_staging_record_count,
            "executor_dram_buffer_count": self.executor_dram_buffer_count,
            "executor_dram_buffer_bytes": self.executor_dram_buffer_bytes,
            "executor_vram_buffer_count": self.executor_vram_buffer_count,
            "executor_vram_buffer_bytes": self.executor_vram_buffer_bytes,
            "executor_lazy_source_seed_count": self.executor_lazy_source_seed_count,
            "executor_lazy_source_seed_bytes": self.executor_lazy_source_seed_bytes,
            "executor_dram_to_vram_copy_count": self.executor_dram_to_vram_copy_count,
            "executor_dram_to_vram_copy_bytes": self.executor_dram_to_vram_copy_bytes,
            "executor_compute_resident_count": self.executor_compute_resident_count,
            "executor_compute_resident_bytes": self.executor_compute_resident_bytes,
            "executor_evict_count": self.executor_evict_count,
            "executor_evict_bytes": self.executor_evict_bytes,
            "executor_missing_source_count": self.executor_missing_source_count,
            "executor_size_mismatch_count": self.executor_size_mismatch_count,
            "executor_staging_overflow_count": self.executor_staging_overflow_count,
            "executor_byte_range_block_count": self.executor_byte_range_block_count,
            "executor_byte_range_count": self.executor_byte_range_count,
            "executor_byte_range_mapped_bytes": self.executor_byte_range_mapped_bytes,
            "executor_byte_range_key_bytes": self.executor_byte_range_key_bytes,
            "executor_byte_range_value_bytes": self.executor_byte_range_value_bytes,
            "executor_byte_range_inferred_row_size_count": self.executor_byte_range_inferred_row_size_count,
            "executor_byte_range_size_mismatch_count": self.executor_byte_range_size_mismatch_count,
            "executor_byte_range_tensor_bounds_error_count": self.executor_byte_range_tensor_bounds_error_count,
            "executor_tensor_dry_run_enabled": self.executor_tensor_dry_run_enabled,
            "executor_tensor_dry_run_block_count": self.executor_tensor_dry_run_block_count,
            "executor_tensor_dry_run_range_count": self.executor_tensor_dry_run_range_count,
            "executor_tensor_dry_run_bytes": self.executor_tensor_dry_run_bytes,
            "executor_tensor_dry_run_key_bytes": self.executor_tensor_dry_run_key_bytes,
            "executor_tensor_dry_run_value_bytes": self.executor_tensor_dry_run_value_bytes,
            "executor_tensor_dry_run_checksum_count": self.executor_tensor_dry_run_checksum_count,
            "executor_tensor_dry_run_checksum_xor": self.executor_tensor_dry_run_checksum_xor,
            "executor_tensor_dry_run_size_mismatch_count": self.executor_tensor_dry_run_size_mismatch_count,
            "executor_tensor_dry_run_tensor_bounds_error_count": self.executor_tensor_dry_run_tensor_bounds_error_count,
            "executor_tensor_mutation_enabled": self.executor_tensor_mutation_enabled,
            "executor_tensor_mutation_attempt_count": self.executor_tensor_mutation_attempt_count,
            "executor_tensor_mutation_block_count": self.executor_tensor_mutation_block_count,
            "executor_tensor_mutation_range_count": self.executor_tensor_mutation_range_count,
            "executor_tensor_mutation_bytes": self.executor_tensor_mutation_bytes,
            "executor_tensor_mutation_key_bytes": self.executor_tensor_mutation_key_bytes,
            "executor_tensor_mutation_value_bytes": self.executor_tensor_mutation_value_bytes,
            "executor_tensor_mutation_checksum_count": self.executor_tensor_mutation_checksum_count,
            "executor_tensor_mutation_staging_checksum_xor": self.executor_tensor_mutation_staging_checksum_xor,
            "executor_tensor_mutation_post_checksum_xor": self.executor_tensor_mutation_post_checksum_xor,
            "executor_tensor_mutation_verify_error_count": self.executor_tensor_mutation_verify_error_count,
            "executor_tensor_mutation_size_mismatch_count": self.executor_tensor_mutation_size_mismatch_count,
            "executor_tensor_mutation_tensor_bounds_error_count": self.executor_tensor_mutation_tensor_bounds_error_count,
            "executor_tensor_mutation_unsupported_count": self.executor_tensor_mutation_unsupported_count,
            "attention_gate_enabled": self.attention_gate_enabled,
            "attention_gate_view_count": self.attention_gate_view_count,
            "attention_gate_k_view_count": self.attention_gate_k_view_count,
            "attention_gate_v_view_count": self.attention_gate_v_view_count,
            "attention_gate_no_required_view_count": self.attention_gate_no_required_view_count,
            "attention_gate_required_block_count": self.attention_gate_required_block_count,
            "attention_gate_ready_block_count": self.attention_gate_ready_block_count,
            "attention_gate_missing_block_count": self.attention_gate_missing_block_count,
            "attention_gate_pending_mutation_count": self.attention_gate_pending_mutation_count,
            "shadow_transition_count": self.shadow_transition_count,
            "shadow_source_match_count": self.shadow_source_match_count,
            "shadow_source_mismatch_count": self.shadow_source_mismatch_count,
            "shadow_final_locations": dict(self.shadow_final_locations),
            "unique_runtime_block_count": self.unique_runtime_block_count,
            "metadata": dict(self.metadata),
        }



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


def infer_intent_kind(command_op: str, command_source: str, command_target: str) -> str:
    """Infer movement intent kind for legacy bridge traces."""

    if command_op in SSD_READ_OPS and command_source == "ssd" and command_target == "dram":
        return "ssd_read"
    if command_op in SSD_READ_OPS and command_source == "dram" and command_target == "vram":
        return "dram_to_vram"
    if command_op == "kv_compute":
        return "compute"
    if command_op == "kv_evict" and command_target == "ssd":
        return "ssd_write"
    return "metadata"


def load_bridge_trace_events(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate bridge trace JSONL events."""

    trace_path = Path(path)
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(trace_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(event, dict):
            raise ValueError(f"bridge trace line {line_number} must be a JSON object")
        if event.get("op") != "kv_command_bridge":
            raise ValueError(f"bridge trace line {line_number} has unsupported op: {event.get('op')}")
        metadata = event.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"bridge trace line {line_number} missing metadata")
        for field_name in (
            "command_op",
            "command_source",
            "command_target",
            "command_size_bytes",
        ):
            if field_name not in metadata:
                raise ValueError(f"bridge trace line {line_number} missing metadata.{field_name}")
        events.append(event)
    return events


def summarize_command_bridge_trace(
    trace_path: str | Path,
    *,
    summary_path: str | Path | None = None,
    runtime_summary_path: str | Path | None = None,
) -> CommandBridgeTraceSummary:
    """Summarize live bridge trace events and optionally write JSON metrics."""

    trace_path = Path(trace_path)
    events = load_bridge_trace_events(trace_path)

    command_counts: dict[str, int] = {}
    ssd_read_command_count = 0
    ssd_write_command_count = 0
    ssd_read_bytes = 0
    ssd_write_bytes = 0
    runtime_blocks: set[tuple[int, int]] = set()
    shadow_locations: dict[tuple[int, int, int, int], str] = {}
    movement_intent_count = 0
    blocking_movement_intent_count = 0
    nonblocking_movement_intent_count = 0
    intent_counts_by_kind: dict[str, int] = {}
    intent_bytes_by_kind: dict[str, int] = {}
    shadow_transition_count = 0
    shadow_source_match_count = 0
    shadow_source_mismatch_count = 0
    layers: set[int] = set()
    block_ids: set[int] = set()

    for event in events:
        layer_id = int(event["layer_id"])
        block_id = int(event["block_id"])
        metadata = event["metadata"]
        command_op = str(metadata["command_op"])
        command_source = str(metadata["command_source"])
        command_target = str(metadata["command_target"])
        command_size_bytes = int(metadata["command_size_bytes"])
        shadow_after = str(metadata.get("shadow_after", command_target))
        shadow_source_match = bool(metadata.get("shadow_source_match", True))
        movement_intent = bool(metadata.get("movement_intent", True))
        intent_kind = str(metadata.get("intent_kind", infer_intent_kind(command_op, command_source, command_target)))
        blocking = bool(metadata.get("blocking", False))

        layers.add(layer_id)
        block_ids.add(block_id)
        runtime_blocks.add((layer_id, block_id))
        shadow_locations[(layer_id, block_id, int(event["start_token"]), int(event["end_token"]))] = shadow_after
        shadow_transition_count += 1
        if shadow_source_match:
            shadow_source_match_count += 1
        else:
            shadow_source_mismatch_count += 1
        command_counts[command_op] = command_counts.get(command_op, 0) + 1
        if movement_intent:
            movement_intent_count += 1
            if blocking:
                blocking_movement_intent_count += 1
            else:
                nonblocking_movement_intent_count += 1
            intent_counts_by_kind[intent_kind] = intent_counts_by_kind.get(intent_kind, 0) + 1
            intent_bytes_by_kind[intent_kind] = intent_bytes_by_kind.get(intent_kind, 0) + command_size_bytes

        if command_op in SSD_READ_OPS and command_source == "ssd":
            ssd_read_command_count += 1
            ssd_read_bytes += command_size_bytes
        if command_op == "kv_evict" and command_target == "ssd":
            ssd_write_command_count += 1
            ssd_write_bytes += command_size_bytes

    metadata: dict[str, Any] = {}
    runtime_summary: dict[str, Any] = {}
    if runtime_summary_path is not None:
        loaded_runtime_summary = json.loads(Path(runtime_summary_path).read_text(encoding="utf-8"))
        if isinstance(loaded_runtime_summary, dict):
            runtime_summary = loaded_runtime_summary
            metadata["runtime_summary"] = runtime_summary
            if int(runtime_summary.get("matched_command_count", len(events))) != len(events):
                raise ValueError("bridge trace event count does not match runtime matched_command_count")

    shadow_final_locations = {location: 0 for location in ("ssd", "dram", "vram")}
    for location in shadow_locations.values():
        shadow_final_locations[location] = shadow_final_locations.get(location, 0) + 1

    result = CommandBridgeTraceSummary(
        trace_path=str(trace_path),
        event_count=len(events),
        layers=tuple(sorted(layers)),
        block_ids=tuple(sorted(block_ids)),
        command_counts_by_op=dict(sorted(command_counts.items())),
        ssd_read_command_count=ssd_read_command_count,
        ssd_write_command_count=ssd_write_command_count,
        ssd_read_bytes=ssd_read_bytes,
        ssd_write_bytes=ssd_write_bytes,
        movement_intent_count=movement_intent_count,
        blocking_movement_intent_count=blocking_movement_intent_count,
        nonblocking_movement_intent_count=nonblocking_movement_intent_count,
        intent_counts_by_kind=dict(sorted(intent_counts_by_kind.items())),
        intent_bytes_by_kind=dict(sorted(intent_bytes_by_kind.items())),
        executor_enabled=_as_bool(runtime_summary.get("executor_enabled", False)),
        executor_backend=str(runtime_summary.get("executor_backend", "not_enabled")),
        executor_metadata_only_count=int(runtime_summary.get("executor_metadata_only_count", 0)),
        executor_ssd_read_attempt_count=int(runtime_summary.get("executor_ssd_read_attempt_count", 0)),
        executor_ssd_read_success_count=int(runtime_summary.get("executor_ssd_read_success_count", 0)),
        executor_ssd_read_error_count=int(runtime_summary.get("executor_ssd_read_error_count", 0)),
        executor_open_error_count=int(runtime_summary.get("executor_open_error_count", 0)),
        executor_seek_error_count=int(runtime_summary.get("executor_seek_error_count", 0)),
        executor_short_read_count=int(runtime_summary.get("executor_short_read_count", 0)),
        executor_alloc_error_count=int(runtime_summary.get("executor_alloc_error_count", 0)),
        executor_backend_error_count=int(runtime_summary.get("executor_backend_error_count", 0)),
        executor_ssd_read_bytes=int(runtime_summary.get("executor_ssd_read_bytes", 0)),
        executor_staging_record_count=int(runtime_summary.get("executor_staging_record_count", 0)),
        executor_dram_buffer_count=int(runtime_summary.get("executor_dram_buffer_count", 0)),
        executor_dram_buffer_bytes=int(runtime_summary.get("executor_dram_buffer_bytes", 0)),
        executor_vram_buffer_count=int(runtime_summary.get("executor_vram_buffer_count", 0)),
        executor_vram_buffer_bytes=int(runtime_summary.get("executor_vram_buffer_bytes", 0)),
        executor_lazy_source_seed_count=int(runtime_summary.get("executor_lazy_source_seed_count", 0)),
        executor_lazy_source_seed_bytes=int(runtime_summary.get("executor_lazy_source_seed_bytes", 0)),
        executor_dram_to_vram_copy_count=int(runtime_summary.get("executor_dram_to_vram_copy_count", 0)),
        executor_dram_to_vram_copy_bytes=int(runtime_summary.get("executor_dram_to_vram_copy_bytes", 0)),
        executor_compute_resident_count=int(runtime_summary.get("executor_compute_resident_count", 0)),
        executor_compute_resident_bytes=int(runtime_summary.get("executor_compute_resident_bytes", 0)),
        executor_evict_count=int(runtime_summary.get("executor_evict_count", 0)),
        executor_evict_bytes=int(runtime_summary.get("executor_evict_bytes", 0)),
        executor_missing_source_count=int(runtime_summary.get("executor_missing_source_count", 0)),
        executor_size_mismatch_count=int(runtime_summary.get("executor_size_mismatch_count", 0)),
        executor_staging_overflow_count=int(runtime_summary.get("executor_staging_overflow_count", 0)),
        executor_byte_range_block_count=int(runtime_summary.get("executor_byte_range_block_count", 0)),
        executor_byte_range_count=int(runtime_summary.get("executor_byte_range_count", 0)),
        executor_byte_range_mapped_bytes=int(runtime_summary.get("executor_byte_range_mapped_bytes", 0)),
        executor_byte_range_key_bytes=int(runtime_summary.get("executor_byte_range_key_bytes", 0)),
        executor_byte_range_value_bytes=int(runtime_summary.get("executor_byte_range_value_bytes", 0)),
        executor_byte_range_inferred_row_size_count=int(runtime_summary.get("executor_byte_range_inferred_row_size_count", 0)),
        executor_byte_range_size_mismatch_count=int(runtime_summary.get("executor_byte_range_size_mismatch_count", 0)),
        executor_byte_range_tensor_bounds_error_count=int(runtime_summary.get("executor_byte_range_tensor_bounds_error_count", 0)),
        executor_tensor_dry_run_enabled=_as_bool(runtime_summary.get("executor_tensor_dry_run_enabled", False)),
        executor_tensor_dry_run_block_count=int(runtime_summary.get("executor_tensor_dry_run_block_count", 0)),
        executor_tensor_dry_run_range_count=int(runtime_summary.get("executor_tensor_dry_run_range_count", 0)),
        executor_tensor_dry_run_bytes=int(runtime_summary.get("executor_tensor_dry_run_bytes", 0)),
        executor_tensor_dry_run_key_bytes=int(runtime_summary.get("executor_tensor_dry_run_key_bytes", 0)),
        executor_tensor_dry_run_value_bytes=int(runtime_summary.get("executor_tensor_dry_run_value_bytes", 0)),
        executor_tensor_dry_run_checksum_count=int(runtime_summary.get("executor_tensor_dry_run_checksum_count", 0)),
        executor_tensor_dry_run_checksum_xor=int(runtime_summary.get("executor_tensor_dry_run_checksum_xor", 0)),
        executor_tensor_dry_run_size_mismatch_count=int(runtime_summary.get("executor_tensor_dry_run_size_mismatch_count", 0)),
        executor_tensor_dry_run_tensor_bounds_error_count=int(runtime_summary.get("executor_tensor_dry_run_tensor_bounds_error_count", 0)),
        executor_tensor_mutation_enabled=_as_bool(runtime_summary.get("executor_tensor_mutation_enabled", False)),
        executor_tensor_mutation_attempt_count=int(runtime_summary.get("executor_tensor_mutation_attempt_count", 0)),
        executor_tensor_mutation_block_count=int(runtime_summary.get("executor_tensor_mutation_block_count", 0)),
        executor_tensor_mutation_range_count=int(runtime_summary.get("executor_tensor_mutation_range_count", 0)),
        executor_tensor_mutation_bytes=int(runtime_summary.get("executor_tensor_mutation_bytes", 0)),
        executor_tensor_mutation_key_bytes=int(runtime_summary.get("executor_tensor_mutation_key_bytes", 0)),
        executor_tensor_mutation_value_bytes=int(runtime_summary.get("executor_tensor_mutation_value_bytes", 0)),
        executor_tensor_mutation_checksum_count=int(runtime_summary.get("executor_tensor_mutation_checksum_count", 0)),
        executor_tensor_mutation_staging_checksum_xor=int(runtime_summary.get("executor_tensor_mutation_staging_checksum_xor", 0)),
        executor_tensor_mutation_post_checksum_xor=int(runtime_summary.get("executor_tensor_mutation_post_checksum_xor", 0)),
        executor_tensor_mutation_verify_error_count=int(runtime_summary.get("executor_tensor_mutation_verify_error_count", 0)),
        executor_tensor_mutation_size_mismatch_count=int(runtime_summary.get("executor_tensor_mutation_size_mismatch_count", 0)),
        executor_tensor_mutation_tensor_bounds_error_count=int(runtime_summary.get("executor_tensor_mutation_tensor_bounds_error_count", 0)),
        executor_tensor_mutation_unsupported_count=int(runtime_summary.get("executor_tensor_mutation_unsupported_count", 0)),
        attention_gate_enabled=_as_bool(runtime_summary.get("attention_gate_enabled", False)),
        attention_gate_view_count=int(runtime_summary.get("attention_gate_view_count", 0)),
        attention_gate_k_view_count=int(runtime_summary.get("attention_gate_k_view_count", 0)),
        attention_gate_v_view_count=int(runtime_summary.get("attention_gate_v_view_count", 0)),
        attention_gate_no_required_view_count=int(runtime_summary.get("attention_gate_no_required_view_count", 0)),
        attention_gate_required_block_count=int(runtime_summary.get("attention_gate_required_block_count", 0)),
        attention_gate_ready_block_count=int(runtime_summary.get("attention_gate_ready_block_count", 0)),
        attention_gate_missing_block_count=int(runtime_summary.get("attention_gate_missing_block_count", 0)),
        attention_gate_pending_mutation_count=int(runtime_summary.get("attention_gate_pending_mutation_count", 0)),
        shadow_transition_count=shadow_transition_count,
        shadow_source_match_count=shadow_source_match_count,
        shadow_source_mismatch_count=shadow_source_mismatch_count,
        shadow_final_locations=dict(sorted(shadow_final_locations.items())),
        unique_runtime_block_count=len(runtime_blocks),
        summary_path=str(summary_path) if summary_path is not None else None,
        metadata=metadata,
    )

    if summary_path is not None:
        output = Path(summary_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--runtime-summary", default=None)
    args = parser.parse_args()

    result = summarize_command_bridge_trace(args.trace, summary_path=args.summary, runtime_summary_path=args.runtime_summary)
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
