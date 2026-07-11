import json

import pytest

from integrations.llama_cpp.command_bridge import load_bridge_trace_events, summarize_command_bridge_trace


def write_jsonl(path, events) -> None:
    path.write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in events), encoding="utf-8")


def intent_kind_for(command_op: str, command_source: str, command_target: str) -> str:
    if command_op in {"kv_load", "kv_prefetch"} and command_source == "ssd" and command_target == "dram":
        return "ssd_read"
    if command_op in {"kv_load", "kv_prefetch"} and command_source == "dram" and command_target == "vram":
        return "dram_to_vram"
    if command_op == "kv_compute":
        return "compute"
    if command_op == "kv_evict" and command_target == "ssd":
        return "ssd_write"
    return "metadata"


def make_event(
    layer_id=0,
    block_id=64,
    command_op="kv_load",
    command_source="ssd",
    command_target="dram",
    blocking=True,
) -> dict:
    return {
        "op": "kv_command_bridge",
        "layer_id": layer_id,
        "block_id": block_id,
        "start_token": 64,
        "end_token": 128,
        "metadata": {
            "source": "llama.cpp",
            "event": "kv_cache_apply_ubatch_command_bridge",
            "command_index": 2,
            "command_op": command_op,
            "command_block_id": block_id,
            "command_start_token": 64,
            "command_end_token": 128,
            "command_source": command_source,
            "command_target": command_target,
            "command_size_bytes": 4096,
            "ssd_offset": 65536,
            "shadow_before": command_source,
            "shadow_after": command_target,
            "shadow_source_match": True,
            "movement_intent": True,
            "intent_kind": intent_kind_for(command_op, command_source, command_target),
            "blocking": blocking,
            "cache_size": 256,
            "n_tokens": 64,
            "n_stream": 1,
        },
    }


def test_summarize_command_bridge_trace_counts_commands_and_ssd_reads(tmp_path) -> None:
    trace_path = tmp_path / "bridge.jsonl"
    summary_path = tmp_path / "summary.json"
    runtime_summary_path = tmp_path / "runtime_summary.json"
    events = [
        make_event(command_op="kv_load", command_source="ssd", command_target="dram", blocking=True),
        make_event(command_op="kv_load", command_source="dram", command_target="vram", blocking=True),
        make_event(command_op="kv_compute", command_source="vram", command_target="vram", blocking=False),
    ]
    write_jsonl(trace_path, events)
    runtime_summary_path.write_text(
        json.dumps(
            {
                "matched_command_count": 3,
                "executor_enabled": True,
                "executor_backend": "stdio_backing_file_stub",
                "executor_metadata_only_count": 2,
                "executor_ssd_read_attempt_count": 1,
                "executor_ssd_read_success_count": 1,
                "executor_ssd_read_error_count": 0,
                "executor_open_error_count": 0,
                "executor_seek_error_count": 0,
                "executor_short_read_count": 0,
                "executor_alloc_error_count": 0,
                "executor_backend_error_count": 0,
                "executor_ssd_read_bytes": 4096,
                "executor_staging_record_count": 1,
                "executor_dram_buffer_count": 1,
                "executor_dram_buffer_bytes": 4096,
                "executor_vram_buffer_count": 1,
                "executor_vram_buffer_bytes": 4096,
                "executor_lazy_source_seed_count": 0,
                "executor_lazy_source_seed_bytes": 0,
                "executor_dram_to_vram_copy_count": 1,
                "executor_dram_to_vram_copy_bytes": 4096,
                "executor_compute_resident_count": 1,
                "executor_compute_resident_bytes": 4096,
                "executor_evict_count": 0,
                "executor_evict_bytes": 0,
                "executor_missing_source_count": 0,
                "executor_size_mismatch_count": 0,
                "executor_staging_overflow_count": 0,
                "executor_byte_range_block_count": 1,
                "executor_byte_range_count": 8,
                "executor_byte_range_mapped_bytes": 4096,
                "executor_byte_range_key_bytes": 2048,
                "executor_byte_range_value_bytes": 2048,
                "executor_byte_range_inferred_row_size_count": 1,
                "executor_byte_range_size_mismatch_count": 0,
                "executor_byte_range_tensor_bounds_error_count": 0,
                "executor_tensor_dry_run_enabled": True,
                "executor_tensor_dry_run_block_count": 1,
                "executor_tensor_dry_run_range_count": 8,
                "executor_tensor_dry_run_bytes": 4096,
                "executor_tensor_dry_run_key_bytes": 2048,
                "executor_tensor_dry_run_value_bytes": 2048,
                "executor_tensor_dry_run_checksum_count": 3,
                "executor_tensor_dry_run_checksum_xor": 123456,
                "executor_tensor_dry_run_size_mismatch_count": 0,
                "executor_tensor_dry_run_tensor_bounds_error_count": 0,
                "executor_tensor_mutation_enabled": True,
                "executor_tensor_mutation_attempt_count": 1,
                "executor_tensor_mutation_block_count": 1,
                "executor_tensor_mutation_range_count": 8,
                "executor_tensor_mutation_bytes": 4096,
                "executor_tensor_mutation_key_bytes": 2048,
                "executor_tensor_mutation_value_bytes": 2048,
                "executor_tensor_mutation_checksum_count": 4,
                "executor_tensor_mutation_staging_checksum_xor": 111,
                "executor_tensor_mutation_post_checksum_xor": 111,
                "executor_tensor_mutation_verify_error_count": 0,
                "executor_tensor_mutation_size_mismatch_count": 0,
                "executor_tensor_mutation_tensor_bounds_error_count": 0,
                "executor_tensor_mutation_unsupported_count": 0,
                "attention_gate_enabled": True,
                "attention_gate_view_count": 2,
                "attention_gate_k_view_count": 1,
                "attention_gate_v_view_count": 1,
                "attention_gate_no_required_view_count": 0,
                "attention_gate_required_block_count": 2,
                "attention_gate_ready_block_count": 2,
                "attention_gate_missing_block_count": 0,
                "attention_gate_pending_mutation_count": 0,
            }
        ),
        encoding="utf-8",
    )

    result = summarize_command_bridge_trace(
        trace_path,
        summary_path=summary_path,
        runtime_summary_path=runtime_summary_path,
    )

    assert result.event_count == 3
    assert result.layers == (0,)
    assert result.block_ids == (64,)
    assert result.command_counts_by_op == {"kv_compute": 1, "kv_load": 2}
    assert result.ssd_read_command_count == 1
    assert result.ssd_read_bytes == 4096
    assert result.movement_intent_count == 3
    assert result.blocking_movement_intent_count == 2
    assert result.nonblocking_movement_intent_count == 1
    assert result.intent_counts_by_kind == {"compute": 1, "dram_to_vram": 1, "ssd_read": 1}
    assert result.intent_bytes_by_kind == {"compute": 4096, "dram_to_vram": 4096, "ssd_read": 4096}
    assert result.executor_enabled is True
    assert result.executor_backend == "stdio_backing_file_stub"
    assert result.executor_metadata_only_count == 2
    assert result.executor_ssd_read_attempt_count == 1
    assert result.executor_ssd_read_success_count == 1
    assert result.executor_ssd_read_error_count == 0
    assert result.executor_backend_error_count == 0
    assert result.executor_ssd_read_bytes == 4096
    assert result.executor_staging_record_count == 1
    assert result.executor_dram_buffer_count == 1
    assert result.executor_dram_buffer_bytes == 4096
    assert result.executor_vram_buffer_count == 1
    assert result.executor_vram_buffer_bytes == 4096
    assert result.executor_dram_to_vram_copy_count == 1
    assert result.executor_dram_to_vram_copy_bytes == 4096
    assert result.executor_compute_resident_count == 1
    assert result.executor_compute_resident_bytes == 4096
    assert result.executor_missing_source_count == 0
    assert result.executor_staging_overflow_count == 0
    assert result.executor_byte_range_block_count == 1
    assert result.executor_byte_range_count == 8
    assert result.executor_byte_range_mapped_bytes == 4096
    assert result.executor_byte_range_key_bytes == 2048
    assert result.executor_byte_range_value_bytes == 2048
    assert result.executor_byte_range_inferred_row_size_count == 1
    assert result.executor_byte_range_size_mismatch_count == 0
    assert result.executor_byte_range_tensor_bounds_error_count == 0
    assert result.executor_tensor_dry_run_enabled is True
    assert result.executor_tensor_dry_run_block_count == 1
    assert result.executor_tensor_dry_run_range_count == 8
    assert result.executor_tensor_dry_run_bytes == 4096
    assert result.executor_tensor_dry_run_key_bytes == 2048
    assert result.executor_tensor_dry_run_value_bytes == 2048
    assert result.executor_tensor_dry_run_checksum_count == 3
    assert result.executor_tensor_dry_run_checksum_xor == 123456
    assert result.executor_tensor_dry_run_size_mismatch_count == 0
    assert result.executor_tensor_dry_run_tensor_bounds_error_count == 0
    assert result.executor_tensor_mutation_enabled is True
    assert result.executor_tensor_mutation_attempt_count == 1
    assert result.executor_tensor_mutation_block_count == 1
    assert result.executor_tensor_mutation_range_count == 8
    assert result.executor_tensor_mutation_bytes == 4096
    assert result.executor_tensor_mutation_key_bytes == 2048
    assert result.executor_tensor_mutation_value_bytes == 2048
    assert result.executor_tensor_mutation_checksum_count == 4
    assert result.executor_tensor_mutation_staging_checksum_xor == 111
    assert result.executor_tensor_mutation_post_checksum_xor == 111
    assert result.executor_tensor_mutation_verify_error_count == 0
    assert result.executor_tensor_mutation_size_mismatch_count == 0
    assert result.executor_tensor_mutation_tensor_bounds_error_count == 0
    assert result.executor_tensor_mutation_unsupported_count == 0
    assert result.attention_gate_enabled is True
    assert result.attention_gate_view_count == 2
    assert result.attention_gate_k_view_count == 1
    assert result.attention_gate_v_view_count == 1
    assert result.attention_gate_required_block_count == 2
    assert result.attention_gate_ready_block_count == 2
    assert result.attention_gate_missing_block_count == 0
    assert result.attention_gate_pending_mutation_count == 0
    assert result.shadow_transition_count == 3
    assert result.shadow_source_match_count == 3
    assert result.shadow_source_mismatch_count == 0
    assert result.shadow_final_locations == {"dram": 0, "ssd": 0, "vram": 1}
    assert result.unique_runtime_block_count == 1
    assert json.loads(summary_path.read_text())["contract"] == "solidattention.llama_cpp.kv_command_bridge_summary.v1"


def test_load_bridge_trace_events_rejects_wrong_op(tmp_path) -> None:
    trace_path = tmp_path / "bad.jsonl"
    write_jsonl(trace_path, [{"op": "kv_add", "metadata": {}}])

    with pytest.raises(ValueError, match="unsupported op"):
        load_bridge_trace_events(trace_path)


def test_summarize_command_bridge_trace_checks_runtime_summary_count(tmp_path) -> None:
    trace_path = tmp_path / "bridge.jsonl"
    runtime_summary_path = tmp_path / "runtime_summary.json"
    write_jsonl(trace_path, [make_event()])
    runtime_summary_path.write_text(json.dumps({"matched_command_count": 2}), encoding="utf-8")

    with pytest.raises(ValueError, match="matched_command_count"):
        summarize_command_bridge_trace(trace_path, runtime_summary_path=runtime_summary_path)

def test_summarize_command_bridge_trace_supports_legacy_events_without_shadow_fields(tmp_path) -> None:
    trace_path = tmp_path / "legacy.jsonl"
    event = make_event(command_op="kv_load", command_source="dram", command_target="vram")
    del event["metadata"]["shadow_before"]
    del event["metadata"]["shadow_after"]
    del event["metadata"]["shadow_source_match"]
    write_jsonl(trace_path, [event])

    result = summarize_command_bridge_trace(trace_path)

    assert result.movement_intent_count == 1
    assert result.intent_counts_by_kind == {"dram_to_vram": 1}
    assert result.executor_enabled is False
    assert result.executor_ssd_read_attempt_count == 0
    assert result.executor_staging_record_count == 0
    assert result.executor_dram_buffer_count == 0
    assert result.executor_vram_buffer_count == 0
    assert result.executor_byte_range_block_count == 0
    assert result.executor_byte_range_count == 0
    assert result.executor_byte_range_mapped_bytes == 0
    assert result.executor_tensor_dry_run_enabled is False
    assert result.executor_tensor_dry_run_block_count == 0
    assert result.executor_tensor_dry_run_range_count == 0
    assert result.executor_tensor_dry_run_bytes == 0
    assert result.executor_tensor_dry_run_checksum_count == 0
    assert result.executor_tensor_dry_run_checksum_xor == 0
    assert result.executor_tensor_dry_run_tensor_bounds_error_count == 0
    assert result.executor_tensor_mutation_enabled is False
    assert result.executor_tensor_mutation_attempt_count == 0
    assert result.executor_tensor_mutation_block_count == 0
    assert result.executor_tensor_mutation_range_count == 0
    assert result.executor_tensor_mutation_bytes == 0
    assert result.executor_tensor_mutation_checksum_count == 0
    assert result.executor_tensor_mutation_staging_checksum_xor == 0
    assert result.executor_tensor_mutation_post_checksum_xor == 0
    assert result.executor_tensor_mutation_verify_error_count == 0
    assert result.executor_tensor_mutation_unsupported_count == 0
    assert result.attention_gate_enabled is False
    assert result.attention_gate_view_count == 0
    assert result.attention_gate_required_block_count == 0
    assert result.attention_gate_ready_block_count == 0
    assert result.attention_gate_missing_block_count == 0
    assert result.attention_gate_pending_mutation_count == 0
    assert result.shadow_transition_count == 1
    assert result.shadow_source_match_count == 1
    assert result.shadow_final_locations == {"dram": 0, "ssd": 0, "vram": 1}

