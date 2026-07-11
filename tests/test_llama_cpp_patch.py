from pathlib import Path

import pytest

from integrations.llama_cpp.probe import probe_llama_cpp
from integrations.llama_cpp.verify_patch import verify_patch


def test_patch_has_no_host_specific_liburing_default() -> None:
    patch = Path("integrations/llama_cpp/patches/solidattention_kv_trace.patch").read_text(
        encoding="utf-8"
    )

    assert 'set(LLAMA_SOLIDATTENTION_LIBURING_PREFIX "" CACHE PATH' in patch
    assert "/data/disk2/ljc" not in patch


def test_solidattention_kv_trace_patch_exists_and_documents_runtime_switch() -> None:
    patch = Path("integrations/llama_cpp/patches/solidattention_kv_trace.patch")

    content = patch.read_text()

    assert "LLAMA_SOLIDATTENTION_KV_TRACE_PATH" in content
    assert "op" in content
    assert "%s" in content
    assert "source" in content
    assert "target" in content
    assert "ssd_offset" in content
    assert "kv_cache_apply_ubatch" in content
    assert "kv_cache_seq_rm" in content
    assert "kv_evict" in content
    assert "LLAMA_SOLIDATTENTION_KV_TRACE_EVICT_SMOKE" in content
    assert "LLAMA_SOLIDATTENTION_KV_COMMAND_PATH" in content
    assert "LLAMA_SOLIDATTENTION_KV_COMMAND_SUMMARY_PATH" in content
    assert "solidattention.llama_cpp.kv_command_dry_run.v1" in content
    assert "llama_solidattention_kv_command_dry_run_once" in content
    assert "valid_command_count" in content
    assert "ssd_read_command_count" in content
    assert "LLAMA_SOLIDATTENTION_KV_COMMAND_BRIDGE_TRACE_PATH" in content
    assert "LLAMA_SOLIDATTENTION_KV_COMMAND_BRIDGE_SUMMARY_PATH" in content
    assert "solidattention.llama_cpp.kv_command_bridge.v1" in content
    assert "llama_solidattention_kv_command_bridge_apply" in content
    assert "matched_command_count" in content
    assert "llama_solidattention_kv_shadow_record" in content
    assert "shadow_before" in content
    assert "shadow_after" in content
    assert "shadow_source_match_count" in content
    assert "llama_solidattention_json_bool" in content
    assert "movement_intent_count" in content
    assert "blocking_movement_intent_count" in content
    assert "intent_kind" in content
    assert "LLAMA_SOLIDATTENTION_KV_EXECUTOR_BACKING_PATH" in content
    assert "LLAMA_SOLIDATTENTION_KV_EXECUTOR_BACKEND" in content
    assert "LLAMA_SOLIDATTENTION_HAVE_LIBURING" in content
    assert "llama_solidattention_execute_movement_intent_stub" in content
    assert "executor_ssd_read_success_count" in content
    assert "executor_backend_error_count" in content
    assert "LLAMA_SOLIDATTENTION_KV_TENSOR_DRY_RUN_TRACE_PATH" in content
    assert "llama_solidattention_executor_tensor_copy_dry_run" in content
    assert "kv_tensor_copy_dry_run" in content
    assert "executor_tensor_dry_run_checksum_xor" in content
    assert "LLAMA_SOLIDATTENTION_KV_TENSOR_MUTATION" in content
    assert "LLAMA_SOLIDATTENTION_KV_TENSOR_MUTATION_TRACE_PATH" in content
    assert "llama_solidattention_executor_tensor_mutation_gate" in content
    assert "kv_tensor_mutation" in content
    assert "executor_tensor_mutation_verify_error_count" in content
    assert "LLAMA_SOLIDATTENTION_KV_ATTENTION_GATE_TRACE_PATH" in content
    assert "LLAMA_SOLIDATTENTION_KV_ATTENTION_GATE" in content
    assert "llama_solidattention_attention_residency_gate" in content
    assert "kv_attention_residency_gate" in content
    assert "attention_gate_pending_mutation_count" in content
    assert "ggml_backend_tensor_set" in content
    assert "stdio_backing_file_stub" in content
    assert "liburing_backing_file_stub" in content
    assert "io_uring_prep_read" in content
    assert "tests/test-llama-archs.cpp" in content
    assert "src/llama-kv-cache.cpp" in content


@pytest.mark.llama
def test_solidattention_kv_trace_patch_applies_to_real_checkout_when_present() -> None:
    checkout = Path("external/llama.cpp")
    patch = Path("integrations/llama_cpp/patches/solidattention_kv_trace.patch")

    probe = probe_llama_cpp(checkout)
    if probe.state in {"missing", "uninitialized"}:
        pytest.skip(f"llama.cpp checkout is {probe.state}")
    assert probe.ready, probe.to_dict()

    result = verify_patch(checkout, patch)

    assert result.ok, result.to_dict()
