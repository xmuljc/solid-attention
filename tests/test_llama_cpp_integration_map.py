from pathlib import Path

import pytest

from integrations.llama_cpp.integration_map import (
    IntegrationPoint,
    check_integration_map,
    default_integration_points,
    save_integration_map,
)
from integrations.llama_cpp.probe import probe_llama_cpp


def test_integration_map_detects_missing_source_anchor(tmp_path) -> None:
    checkout = tmp_path / "checkout"
    source = checkout / "src"
    source.mkdir(parents=True)
    (source / "llama-kv-cache.cpp").write_text("void llama_kv_cache::seq_rm() {}\n")
    point = IntegrationPoint(
        point_id="missing_anchor",
        phase="test",
        status="candidate_not_patched",
        purpose="test",
        file_path="src/llama-kv-cache.cpp",
        required_anchors=("not present",),
    )

    result = check_integration_map(checkout, points=(point,))

    assert not result.ok
    assert result.points[0].missing_anchors == ("not present",)


def test_integration_map_checks_patch_anchors(tmp_path) -> None:
    checkout = tmp_path / "checkout"
    source = checkout / "src"
    source.mkdir(parents=True)
    (source / "llama-kv-cache.cpp").write_text("void llama_kv_cache::apply_ubatch() {}\n")
    patch = tmp_path / "trace.patch"
    patch.write_text("kv_cache_apply_ubatch\n")
    point = IntegrationPoint(
        point_id="patch_anchor",
        phase="test",
        status="implemented_patch",
        purpose="test",
        file_path="src/llama-kv-cache.cpp",
        required_anchors=("apply_ubatch",),
        patch_required_anchors=("kv_cache_apply_ubatch",),
    )

    result = check_integration_map(checkout, patch=patch, points=(point,))

    assert result.ok
    assert result.points[0].missing_patch_anchors == ()


def test_save_integration_map_writes_contract(tmp_path) -> None:
    output = tmp_path / "map.json"

    save_integration_map(output)

    content = output.read_text()
    assert "solidattention.llama_cpp.integration_map.v1" in content
    assert "trace_kv_add_apply_ubatch" in content


def test_default_integration_map_covers_runtime_command_contract() -> None:
    points = default_integration_points()
    command_ops = {op for point in points for op in point.command_ops}

    assert {"kv_load", "kv_prefetch", "kv_compute", "kv_evict", "kv_add"}.issubset(command_ops)


def test_default_integration_map_includes_command_dry_run_adapter() -> None:
    points = {point.point_id: point for point in default_integration_points()}

    point = points["runtime_command_dry_run_adapter"]
    assert point.status == "implemented_patch"
    assert {"kv_load", "kv_prefetch", "kv_compute", "kv_evict"}.issubset(point.command_ops)
    assert "LLAMA_SOLIDATTENTION_KV_COMMAND_PATH" in point.patch_required_anchors
    assert "llama_solidattention_kv_command_dry_run_once" in point.patch_required_anchors


def test_default_integration_map_includes_command_live_bridge() -> None:
    points = {point.point_id: point for point in default_integration_points()}

    point = points["runtime_command_live_bridge"]
    assert point.status == "implemented_patch"
    assert {"kv_load", "kv_prefetch", "kv_compute", "kv_evict"}.issubset(point.command_ops)
    assert "LLAMA_SOLIDATTENTION_KV_COMMAND_BRIDGE_TRACE_PATH" in point.patch_required_anchors
    assert "solidattention.llama_cpp.kv_command_bridge.v1" in point.patch_required_anchors


def test_default_integration_map_includes_shadow_residency_bridge() -> None:
    points = {point.point_id: point for point in default_integration_points()}

    point = points["runtime_shadow_residency_bridge"]
    assert point.status == "implemented_patch"
    assert {"kv_load", "kv_prefetch", "kv_compute", "kv_evict"}.issubset(point.command_ops)
    assert "llama_solidattention_kv_shadow_record" in point.patch_required_anchors
    assert "shadow_source_match_count" in point.patch_required_anchors


def test_default_integration_map_includes_movement_intent_bridge() -> None:
    points = {point.point_id: point for point in default_integration_points()}

    point = points["runtime_movement_intent_bridge"]
    assert point.status == "implemented_patch"
    assert {"kv_load", "kv_prefetch", "kv_compute", "kv_evict"}.issubset(point.command_ops)
    assert "movement_intent_count" in point.patch_required_anchors
    assert "intent_kind" in point.patch_required_anchors


def test_default_integration_map_includes_runtime_executor_stub() -> None:
    points = {point.point_id: point for point in default_integration_points()}

    point = points["runtime_executor_stub"]
    assert point.status == "implemented_patch"
    assert {"kv_load", "kv_prefetch"}.issubset(point.command_ops)
    assert "LLAMA_SOLIDATTENTION_KV_EXECUTOR_BACKING_PATH" in point.patch_required_anchors
    assert "LLAMA_SOLIDATTENTION_KV_EXECUTOR_BACKEND" in point.patch_required_anchors
    assert "executor_ssd_read_success_count" in point.patch_required_anchors
    assert "liburing_backing_file_stub" in point.patch_required_anchors


def test_default_integration_map_includes_runtime_executor_staging_residency() -> None:
    points = {point.point_id: point for point in default_integration_points()}

    point = points["runtime_executor_staging_residency"]
    assert point.status == "implemented_patch"
    assert {"kv_load", "kv_prefetch", "kv_compute", "kv_evict"}.issubset(point.command_ops)
    assert "llama_solidattention_kv_staging_record" in point.patch_required_anchors
    assert "executor_dram_to_vram_copy_count" in point.patch_required_anchors
    assert "executor_compute_resident_count" in point.patch_required_anchors
    assert "executor_byte_range_block_count" in point.patch_required_anchors
    assert "executor_byte_range_tensor_bounds_error_count" in point.patch_required_anchors


def test_default_integration_map_includes_runtime_executor_tensor_copy_dry_run() -> None:
    points = {point.point_id: point for point in default_integration_points()}

    point = points["runtime_executor_tensor_copy_dry_run"]
    assert point.status == "implemented_patch"
    assert {"kv_load", "kv_prefetch", "kv_compute"}.issubset(point.command_ops)
    assert "LLAMA_SOLIDATTENTION_KV_TENSOR_DRY_RUN_TRACE_PATH" in point.patch_required_anchors
    assert "llama_solidattention_executor_tensor_copy_dry_run" in point.patch_required_anchors
    assert "kv_tensor_copy_dry_run" in point.patch_required_anchors
    assert "executor_tensor_dry_run_checksum_xor" in point.patch_required_anchors


def test_default_integration_map_includes_runtime_executor_tensor_mutation_gate() -> None:
    points = {point.point_id: point for point in default_integration_points()}

    point = points["runtime_executor_tensor_mutation_gate"]
    assert point.status == "implemented_patch"
    assert {"kv_load", "kv_prefetch", "kv_compute"}.issubset(point.command_ops)
    assert "LLAMA_SOLIDATTENTION_KV_TENSOR_MUTATION" in point.patch_required_anchors
    assert "LLAMA_SOLIDATTENTION_KV_TENSOR_MUTATION_TRACE_PATH" in point.patch_required_anchors
    assert "llama_solidattention_executor_tensor_mutation_gate" in point.patch_required_anchors
    assert "ggml_backend_tensor_set" in point.patch_required_anchors
    assert "executor_tensor_mutation_verify_error_count" in point.patch_required_anchors


def test_default_integration_map_includes_runtime_attention_residency_gate() -> None:
    points = {point.point_id: point for point in default_integration_points()}

    point = points["runtime_attention_residency_gate"]
    assert point.status == "implemented_patch"
    assert {"kv_load", "kv_prefetch", "kv_compute"}.issubset(point.command_ops)
    assert "LLAMA_SOLIDATTENTION_KV_ATTENTION_GATE_TRACE_PATH" in point.patch_required_anchors
    assert "LLAMA_SOLIDATTENTION_KV_ATTENTION_GATE" in point.patch_required_anchors
    assert "llama_solidattention_attention_residency_gate" in point.patch_required_anchors
    assert "kv_attention_residency_gate" in point.patch_required_anchors
    assert "attention_gate_pending_mutation_count" in point.patch_required_anchors


def test_default_integration_map_includes_runtime_kv_byte_range_verifier() -> None:
    points = {point.point_id: point for point in default_integration_points()}

    point = points["runtime_kv_byte_range_verifier"]
    assert point.status == "implemented_external_verifier"
    assert {"kv_load", "kv_prefetch", "kv_compute", "kv_evict"}.issubset(point.command_ops)
    assert "solidattention.llama_cpp.kv_byte_ranges.v1" in point.required_anchors
    assert "verify_kv_byte_ranges" in point.required_anchors
    assert point.file_path == "integrations/llama_cpp/kv_byte_ranges.py"


def test_default_integration_map_includes_runtime_executor_liburing_build_hook() -> None:
    points = {point.point_id: point for point in default_integration_points()}

    point = points["runtime_executor_liburing_build_hook"]
    assert point.status == "implemented_patch"
    assert {"kv_load", "kv_prefetch"}.issubset(point.command_ops)
    assert "LLAMA_SOLIDATTENTION_LIBURING_PREFIX" in point.patch_required_anchors
    assert "LLAMA_SOLIDATTENTION_HAVE_LIBURING" in point.patch_required_anchors


@pytest.mark.llama
def test_real_checkout_integration_map_is_valid_when_present() -> None:
    checkout = Path("external/llama.cpp")
    patch = Path("integrations/llama_cpp/patches/solidattention_kv_trace.patch")
    probe = probe_llama_cpp(checkout)
    if probe.state in {"missing", "uninitialized"}:
        pytest.skip(f"llama.cpp checkout is {probe.state}")
    assert probe.ready, probe.to_dict()

    result = check_integration_map(checkout, patch=patch, project_root=".")

    assert result.ok, result.to_dict()
