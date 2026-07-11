"""Machine-checkable map of SolidAttention llama.cpp integration points."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class IntegrationPoint:
    """One candidate or implemented llama.cpp integration point."""

    point_id: str
    phase: str
    status: str
    purpose: str
    file_path: str
    required_anchors: tuple[str, ...]
    command_ops: tuple[str, ...] = ()
    patch_required_anchors: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_id": self.point_id,
            "phase": self.phase,
            "status": self.status,
            "purpose": self.purpose,
            "file_path": self.file_path,
            "required_anchors": list(self.required_anchors),
            "command_ops": list(self.command_ops),
            "patch_required_anchors": list(self.patch_required_anchors),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class IntegrationPointCheck:
    point_id: str
    ok: bool
    file_path: str
    missing_anchors: tuple[str, ...]
    missing_patch_anchors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_id": self.point_id,
            "ok": self.ok,
            "file_path": self.file_path,
            "missing_anchors": list(self.missing_anchors),
            "missing_patch_anchors": list(self.missing_patch_anchors),
        }


@dataclass(frozen=True)
class IntegrationMapCheckResult:
    checkout: str
    patch: str | None
    ok: bool
    points: tuple[IntegrationPointCheck, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkout": self.checkout,
            "patch": self.patch,
            "ok": self.ok,
            "points": [point.to_dict() for point in self.points],
            "metadata": dict(self.metadata),
        }


DEFAULT_INTEGRATION_POINTS: tuple[IntegrationPoint, ...] = (
    IntegrationPoint(
        point_id="trace_kv_add_apply_ubatch",
        phase="phase3_instrumentation",
        status="implemented_patch",
        purpose="emit kv_add records when llama.cpp applies a decoded ubatch to KV cache metadata",
        file_path="src/llama-kv-cache.cpp",
        required_anchors=("void llama_kv_cache::apply_ubatch", "head = sinfo.idxs[s].back() + 1"),
        command_ops=("kv_add",),
        patch_required_anchors=("kv_cache_apply_ubatch", "llama_solidattention_kv_trace_event"),
    ),
    IntegrationPoint(
        point_id="trace_kv_evict_seq_rm",
        phase="phase3_instrumentation",
        status="implemented_patch",
        purpose="emit kv_evict records when llama.cpp removes KV cells via seq_rm",
        file_path="src/llama-kv-cache.cpp",
        required_anchors=("bool llama_kv_cache::seq_rm", "cells.seq_rm", "cells.rm"),
        command_ops=("kv_evict",),
        patch_required_anchors=("kv_cache_seq_rm", "kv_evict", "solidattention_removed_count"),
    ),
    IntegrationPoint(
        point_id="runtime_command_dry_run_adapter",
        phase="phase4_runtime_scheduler",
        status="implemented_patch",
        purpose="load and validate exported SolidAttention KV command JSONL inside llama.cpp without mutating KV tensors",
        file_path="src/llama-kv-cache.cpp",
        required_anchors=("void llama_kv_cache::apply_ubatch", "std::vector<uint32_t> llama_kv_cache::get_layer_ids() const"),
        command_ops=("kv_load", "kv_prefetch", "kv_compute", "kv_evict"),
        patch_required_anchors=(
            "LLAMA_SOLIDATTENTION_KV_COMMAND_PATH",
            "LLAMA_SOLIDATTENTION_KV_COMMAND_SUMMARY_PATH",
            "llama_solidattention_kv_command_dry_run_once",
        ),
        notes=("Dry-run adapter proves command contract compatibility before real KV movement hooks are added.",),
    ),
    IntegrationPoint(
        point_id="runtime_command_live_bridge",
        phase="phase4_runtime_scheduler",
        status="implemented_patch",
        purpose="match live llama.cpp apply_ubatch token/block ranges to exported SolidAttention KV commands and emit bridge trace metrics",
        file_path="src/llama-kv-cache.cpp",
        required_anchors=("void llama_kv_cache::apply_ubatch", "head = sinfo.idxs[s].back() + 1"),
        command_ops=("kv_load", "kv_prefetch", "kv_compute", "kv_evict"),
        patch_required_anchors=(
            "LLAMA_SOLIDATTENTION_KV_COMMAND_BRIDGE_TRACE_PATH",
            "LLAMA_SOLIDATTENTION_KV_COMMAND_BRIDGE_SUMMARY_PATH",
            "llama_solidattention_kv_command_bridge_apply",
            "solidattention.llama_cpp.kv_command_bridge.v1",
        ),
        notes=("Bridge emits scheduler-decision evidence at decode time while leaving KV tensor residency unchanged.",),
    ),
    IntegrationPoint(
        point_id="runtime_shadow_residency_bridge",
        phase="phase4_runtime_scheduler",
        status="implemented_patch",
        purpose="maintain per-block shadow residency while replaying matched SolidAttention KV commands inside llama.cpp",
        file_path="src/llama-kv-cache.cpp",
        required_anchors=("void llama_kv_cache::apply_ubatch", "head = sinfo.idxs[s].back() + 1"),
        command_ops=("kv_load", "kv_prefetch", "kv_compute", "kv_evict"),
        patch_required_anchors=(
            "llama_solidattention_kv_shadow_record",
            "shadow_before",
            "shadow_after",
            "shadow_source_match_count",
        ),
        notes=("Shadow residency is metadata-only evidence; it does not move llama.cpp KV tensors yet.",),
    ),
    IntegrationPoint(
        point_id="runtime_movement_intent_bridge",
        phase="phase4_runtime_scheduler",
        status="implemented_patch",
        purpose="classify matched KV commands as metadata-only movement intents with blocking and byte-count metrics inside llama.cpp",
        file_path="src/llama-kv-cache.cpp",
        required_anchors=("void llama_kv_cache::apply_ubatch", "head = sinfo.idxs[s].back() + 1"),
        command_ops=("kv_load", "kv_prefetch", "kv_compute", "kv_evict"),
        patch_required_anchors=(
            "llama_solidattention_json_bool",
            "movement_intent_count",
            "blocking_movement_intent_count",
            "intent_kind",
        ),
        notes=("Movement intents are metadata-only; they classify the future executor path before real KV tensor movement is enabled.",),
    ),
    IntegrationPoint(
        point_id="runtime_executor_stub",
        phase="phase4_runtime_scheduler",
        status="implemented_patch",
        purpose="execute matched SSD-read movement intents against an opt-in backing file without mutating llama.cpp KV tensors",
        file_path="src/llama-kv-cache.cpp",
        required_anchors=("void llama_kv_cache::apply_ubatch", "head = sinfo.idxs[s].back() + 1"),
        command_ops=("kv_load", "kv_prefetch"),
        patch_required_anchors=(
            "LLAMA_SOLIDATTENTION_KV_EXECUTOR_BACKING_PATH",
            "LLAMA_SOLIDATTENTION_KV_EXECUTOR_BACKEND",
            "llama_solidattention_execute_movement_intent_stub",
            "executor_ssd_read_success_count",
            "executor_backend_error_count",
            "stdio_backing_file_stub",
            "liburing_backing_file_stub",
            "io_uring_prep_read",
        ),
        notes=("Executor stub reads SSD ranges into DRAM staging buffers; it is a runtime evidence path before real KV tensor movement.",),
    ),
    IntegrationPoint(
        point_id="runtime_executor_staging_residency",
        phase="phase4_runtime_scheduler",
        status="implemented_patch",
        purpose="retain matched executor bytes in explicit DRAM/VRAM staging records and verify DRAM-to-VRAM plus compute residency before mutating real KV tensors",
        file_path="src/llama-kv-cache.cpp",
        required_anchors=("void llama_kv_cache::apply_ubatch", "head = sinfo.idxs[s].back() + 1"),
        command_ops=("kv_load", "kv_prefetch", "kv_compute", "kv_evict"),
        patch_required_anchors=(
            "llama_solidattention_kv_staging_record",
            "executor_dram_buffer_count",
            "executor_vram_buffer_count",
            "executor_dram_to_vram_copy_count",
            "executor_compute_resident_count",
            "executor_lazy_source_seed_count",
            "executor_missing_source_count",
            "executor_byte_range_block_count",
            "executor_byte_range_count",
            "executor_byte_range_tensor_bounds_error_count",
        ),
        notes=("Staging residency is still isolated from llama.cpp KV tensors; it verifies the lifecycle that the real tensor movement hook will later reuse.",),
    ),
    IntegrationPoint(
        point_id="runtime_executor_tensor_copy_dry_run",
        phase="phase4_runtime_scheduler",
        status="implemented_patch",
        purpose="emit opt-in dry-run trace and checksum metrics for copying VRAM staging bytes into llama.cpp K/V tensor byte ranges without mutating tensors",
        file_path="src/llama-kv-cache.cpp",
        required_anchors=("void llama_kv_cache::apply_ubatch", "head = sinfo.idxs[s].back() + 1"),
        command_ops=("kv_load", "kv_prefetch", "kv_compute"),
        patch_required_anchors=(
            "LLAMA_SOLIDATTENTION_KV_TENSOR_DRY_RUN_TRACE_PATH",
            "llama_solidattention_executor_tensor_copy_dry_run",
            "kv_tensor_copy_dry_run",
            "executor_tensor_dry_run_block_count",
            "executor_tensor_dry_run_checksum_xor",
            "executor_tensor_dry_run_tensor_bounds_error_count",
        ),
        notes=("Dry-run emits byte offsets and FNV-1a checksums from VRAM staging before any real llama.cpp KV tensor copy is enabled.",),
    ),
    IntegrationPoint(
        point_id="runtime_executor_tensor_mutation_gate",
        phase="phase4_runtime_scheduler",
        status="implemented_patch",
        purpose="optionally copy VRAM staging bytes into llama.cpp K/V tensors with post-copy checksum verification when an explicit mutation gate is enabled",
        file_path="src/llama-kv-cache.cpp",
        required_anchors=("void llama_kv_cache::apply_ubatch", "head = sinfo.idxs[s].back() + 1"),
        command_ops=("kv_load", "kv_prefetch", "kv_compute"),
        patch_required_anchors=(
            "LLAMA_SOLIDATTENTION_KV_TENSOR_MUTATION",
            "LLAMA_SOLIDATTENTION_KV_TENSOR_MUTATION_TRACE_PATH",
            "llama_solidattention_executor_tensor_mutation_gate",
            "kv_tensor_mutation",
            "ggml_backend_tensor_set",
            "executor_tensor_mutation_block_count",
            "executor_tensor_mutation_verify_error_count",
            "executor_tensor_mutation_unsupported_count",
        ),
        notes=("The mutation gate is disabled by default; it is the first verified path that can mutate llama.cpp KV tensors from SolidAttention staging buffers.",),
    ),
    IntegrationPoint(
        point_id="runtime_attention_residency_gate",
        phase="phase4_runtime_scheduler",
        status="implemented_patch",
        purpose="verify before llama.cpp get_k/get_v attention views that scheduler-required KV blocks have completed tensor mutation into real K/V tensors",
        file_path="src/llama-kv-cache.cpp",
        required_anchors=("ggml_tensor * llama_kv_cache::get_k", "ggml_tensor * llama_kv_cache::get_v", "ggml_view_4d"),
        command_ops=("kv_load", "kv_prefetch", "kv_compute"),
        patch_required_anchors=(
            "LLAMA_SOLIDATTENTION_KV_ATTENTION_GATE_TRACE_PATH",
            "LLAMA_SOLIDATTENTION_KV_ATTENTION_GATE",
            "llama_solidattention_attention_residency_gate",
            "kv_attention_residency_gate",
            "attention_gate_view_count",
            "attention_gate_ready_block_count",
            "attention_gate_pending_mutation_count",
        ),
        notes=("This gate is a verification hook at the attention-view boundary; it does not yet perform blocking SSD I/O inside get_k/get_v.",),
    ),
    IntegrationPoint(
        point_id="runtime_executor_liburing_build_hook",
        phase="phase4_runtime_scheduler",
        status="implemented_patch",
        purpose="optionally compile and link the patched llama.cpp executor stub with liburing",
        file_path="src/CMakeLists.txt",
        required_anchors=("add_library(llama", "target_link_libraries(llama PUBLIC ggml)"),
        command_ops=("kv_load", "kv_prefetch"),
        patch_required_anchors=(
            "LLAMA_SOLIDATTENTION_LIBURING_PREFIX",
            "LLAMA_SOLIDATTENTION_LIBURING_LIBRARY",
            "LLAMA_SOLIDATTENTION_HAVE_LIBURING",
        ),
        notes=("The build hook is optional; hosts without liburing still compile the stdio executor stub.",),
    ),
    IntegrationPoint(
        point_id="runtime_kv_byte_range_verifier",
        phase="phase4_runtime_scheduler",
        status="implemented_external_verifier",
        purpose="verify SolidAttention staging byte ranges against llama.cpp K/V row-major tensor offsets with checksums, trace, and JSON metrics",
        file_path="integrations/llama_cpp/kv_byte_ranges.py",
        required_anchors=(
            "solidattention.llama_cpp.kv_byte_ranges.v1",
            "verify_kv_byte_ranges",
            "KVByteRange",
            "kv_byte_range_metrics.json",
            "kv_byte_range_trace.json",
        ),
        command_ops=("kv_load", "kv_prefetch", "kv_compute", "kv_evict"),
        notes=("External verifier maps interleaved staging bytes to llama.cpp row offsets before real KV tensor mutation is enabled.",),
    ),
    IntegrationPoint(
        point_id="future_runtime_load_before_attention_views",
        phase="phase4_runtime_scheduler",
        status="candidate_not_patched",
        purpose="ensure selected KV blocks are resident before get_k/get_v return attention views",
        file_path="src/llama-kv-cache.cpp",
        required_anchors=("ggml_tensor * llama_kv_cache::get_k", "ggml_tensor * llama_kv_cache::get_v", "ggml_view_4d"),
        command_ops=("kv_load", "kv_prefetch"),
        notes=("Do not block inside graph-building without an explicit synchronization plan.",),
    ),
    IntegrationPoint(
        point_id="future_runtime_store_after_decode_copy",
        phase="phase4_runtime_scheduler",
        status="candidate_not_patched",
        purpose="associate newly written K/V rows with SolidAttention block metadata after cpy_k/cpy_v graph ops",
        file_path="src/llama-kv-cache.cpp",
        required_anchors=("ggml_tensor * llama_kv_cache::cpy_k", "ggml_tensor * llama_kv_cache::cpy_v", "ggml_set_rows"),
        command_ops=("kv_load",),
        notes=("This is metadata-sensitive; tensor writes are graph nodes, not immediate host copies.",),
    ),
    IntegrationPoint(
        point_id="future_runtime_block_io_byte_ranges",
        phase="phase4_runtime_scheduler",
        status="candidate_not_patched",
        purpose="reuse state_read_data/state_write_data row-offset logic as reference for K/V byte range mapping",
        file_path="src/llama-kv-cache.cpp",
        required_anchors=("void llama_kv_cache::state_write_data", "bool llama_kv_cache::state_read_data", "io.write_tensor", "io.read_tensor"),
        command_ops=("kv_load", "kv_evict"),
        notes=("Good reference for byte ranges; not the final low-latency decode path by itself.",),
    ),
    IntegrationPoint(
        point_id="prototype_runtime_command_consumer",
        phase="phase4_runtime_scheduler",
        status="implemented_external_prototype",
        purpose="execute exported SolidAttention KV commands with liburing SSD I/O and mock DRAM/VRAM buffers",
        file_path="io/runtime_kv_command_consumer.c",
        required_anchors=("kv_load", "kv_evict", "HAVE_LIBURING", "verification_errors"),
        command_ops=("kv_load", "kv_prefetch", "kv_compute", "kv_evict"),
    ),
)


def default_integration_points() -> tuple[IntegrationPoint, ...]:
    return DEFAULT_INTEGRATION_POINTS


def check_integration_map(
    checkout: str | Path,
    *,
    patch: str | Path | None = None,
    project_root: str | Path | None = None,
    points: Iterable[IntegrationPoint] = DEFAULT_INTEGRATION_POINTS,
) -> IntegrationMapCheckResult:
    checkout = Path(checkout)
    project_root = Path(project_root) if project_root is not None else Path.cwd()
    patch_text = None
    patch_path = Path(patch) if patch is not None else None
    if patch_path is not None:
        patch_text = patch_path.read_text(encoding="utf-8")

    checks: list[IntegrationPointCheck] = []
    for point in points:
        source_root = project_root if point.file_path.startswith(("core/", "harness/", "integrations/", "io/")) else checkout
        source_path = source_root / point.file_path
        if source_path.exists():
            content = source_path.read_text(encoding="utf-8", errors="replace")
            missing = tuple(anchor for anchor in point.required_anchors if anchor not in content)
        else:
            missing = point.required_anchors

        if patch_text is None:
            missing_patch = () if not point.patch_required_anchors else point.patch_required_anchors
        else:
            missing_patch = tuple(anchor for anchor in point.patch_required_anchors if anchor not in patch_text)

        checks.append(
            IntegrationPointCheck(
                point_id=point.point_id,
                ok=not missing and not missing_patch,
                file_path=str(source_path),
                missing_anchors=missing,
                missing_patch_anchors=missing_patch,
            )
        )

    return IntegrationMapCheckResult(
        checkout=str(checkout),
        patch=str(patch_path) if patch_path is not None else None,
        ok=all(check.ok for check in checks),
        points=tuple(checks),
        metadata={
            "integration_point_count": len(checks),
            "implemented_count": sum(1 for point in points if point.status.startswith("implemented")),
            "candidate_count": sum(1 for point in points if point.status.startswith("candidate")),
            "contract": "solidattention.llama_cpp.integration_map.v1",
        },
    )


def save_integration_map(path: str | Path, points: Iterable[IntegrationPoint] = DEFAULT_INTEGRATION_POINTS) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "contract": "solidattention.llama_cpp.integration_map.v1",
        "points": [point.to_dict() for point in points],
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--patch", default=None)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--map-output", default=None)
    parser.add_argument("--check-output", default=None)
    args = parser.parse_args()

    if args.map_output is not None:
        save_integration_map(args.map_output)
    result = check_integration_map(args.checkout, patch=args.patch, project_root=args.project_root)
    if args.check_output is not None:
        output = Path(args.check_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
