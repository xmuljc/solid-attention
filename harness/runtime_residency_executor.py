"""Execute runtime KV movement plans with explicit DRAM/VRAM residency state."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.run_manifest import RunContext
from harness.trace import TraceRecorder


@dataclass(frozen=True)
class RuntimeBuffer:
    """One simulated runtime buffer owned by DRAM or VRAM."""

    layer_id: int
    block_id: int
    start_token: int
    end_token: int
    size_bytes: int
    data: bytes

    @property
    def key(self) -> tuple[int, int, int, int]:
        return (self.layer_id, self.block_id, self.start_token, self.end_token)


@dataclass(frozen=True)
class RuntimeResidencyExecutionResult:
    plan_path: str
    output_dir: str
    movement_op_count: int
    ssd_read_count: int
    dram_to_vram_copy_count: int
    compute_count: int
    evict_count: int
    ssd_write_count: int
    ssd_read_bytes: int
    dram_to_vram_bytes: int
    compute_bytes: int
    ssd_write_bytes: int
    dram_alloc_count: int
    vram_alloc_count: int
    lazy_source_seed_count: int
    lazy_source_seed_bytes: int
    missing_source_count: int
    size_mismatch_count: int
    unsupported_op_count: int
    verification_error_count: int
    final_residency: dict[str, dict[str, int]]
    metrics_path: str
    trace_path: str
    summary_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_path": self.plan_path,
            "output_dir": self.output_dir,
            "movement_op_count": self.movement_op_count,
            "ssd_read_count": self.ssd_read_count,
            "dram_to_vram_copy_count": self.dram_to_vram_copy_count,
            "compute_count": self.compute_count,
            "evict_count": self.evict_count,
            "ssd_write_count": self.ssd_write_count,
            "ssd_read_bytes": self.ssd_read_bytes,
            "dram_to_vram_bytes": self.dram_to_vram_bytes,
            "compute_bytes": self.compute_bytes,
            "ssd_write_bytes": self.ssd_write_bytes,
            "dram_alloc_count": self.dram_alloc_count,
            "vram_alloc_count": self.vram_alloc_count,
            "lazy_source_seed_count": self.lazy_source_seed_count,
            "lazy_source_seed_bytes": self.lazy_source_seed_bytes,
            "missing_source_count": self.missing_source_count,
            "size_mismatch_count": self.size_mismatch_count,
            "unsupported_op_count": self.unsupported_op_count,
            "verification_error_count": self.verification_error_count,
            "final_residency": self.final_residency,
            "metrics_path": self.metrics_path,
            "trace_path": self.trace_path,
            "summary_path": self.summary_path,
        }


class RuntimeResidencyExecutor:
    """Metadata-plus-buffer executor for KV residency lifecycle experiments."""

    def __init__(self, *, backing_file: str | Path | None = None, seed_missing_sources: bool = True) -> None:
        self.backing_file = Path(backing_file) if backing_file is not None else None
        self.seed_missing_sources = seed_missing_sources
        self.dram: dict[tuple[int, int, int, int], RuntimeBuffer] = {}
        self.vram: dict[tuple[int, int, int, int], RuntimeBuffer] = {}
        self.trace = TraceRecorder()
        self.ssd_read_count = 0
        self.dram_to_vram_copy_count = 0
        self.compute_count = 0
        self.evict_count = 0
        self.ssd_write_count = 0
        self.ssd_read_bytes = 0
        self.dram_to_vram_bytes = 0
        self.compute_bytes = 0
        self.ssd_write_bytes = 0
        self.dram_alloc_count = 0
        self.vram_alloc_count = 0
        self.lazy_source_seed_count = 0
        self.lazy_source_seed_bytes = 0
        self.missing_source_count = 0
        self.size_mismatch_count = 0
        self.unsupported_op_count = 0

    @property
    def verification_error_count(self) -> int:
        return self.missing_source_count + self.size_mismatch_count + self.unsupported_op_count

    def execute(self, ops: list[dict[str, Any]]) -> None:
        for index, op in enumerate(ops):
            self.execute_op(op, index)

    def execute_op(self, op: dict[str, Any], index: int) -> None:
        movement_op = str(op.get("op"))
        source = str(op.get("source"))
        target = str(op.get("target"))
        if movement_op in {"kv_load", "kv_prefetch"} and source == "ssd" and target == "dram":
            self._ssd_to_dram(op)
        elif movement_op in {"kv_load", "kv_prefetch"} and source == "dram" and target == "vram":
            self._dram_to_vram(op)
        elif movement_op == "kv_compute":
            self._compute(op)
        elif movement_op == "kv_evict" and target == "ssd":
            self._evict_to_ssd(op)
        else:
            self.unsupported_op_count += 1
        self._record_trace(op, index)

    def final_residency(self) -> dict[str, dict[str, int]]:
        return {
            "dram": self._location_stats(self.dram),
            "vram": self._location_stats(self.vram),
        }

    def _ssd_to_dram(self, op: dict[str, Any]) -> None:
        data = self._read_backing_or_pattern(op)
        buffer = self._buffer_from_op(op, data)
        self.dram[buffer.key] = buffer
        self.ssd_read_count += 1
        self.ssd_read_bytes += buffer.size_bytes
        self.dram_alloc_count += 1

    def _dram_to_vram(self, op: dict[str, Any]) -> None:
        key = self._key_from_op(op)
        source = self.dram.get(key)
        if source is None:
            source = self._seed_source_buffer(op, "dram")
            if source is None:
                self.missing_source_count += 1
                return
        if source.size_bytes != int(op["size_bytes"]):
            self.size_mismatch_count += 1
            return
        copied = RuntimeBuffer(
            layer_id=source.layer_id,
            block_id=source.block_id,
            start_token=source.start_token,
            end_token=source.end_token,
            size_bytes=source.size_bytes,
            data=bytes(source.data),
        )
        self.vram[key] = copied
        self.dram_to_vram_copy_count += 1
        self.dram_to_vram_bytes += copied.size_bytes
        self.vram_alloc_count += 1

    def _compute(self, op: dict[str, Any]) -> None:
        key = self._key_from_op(op)
        source = self.vram.get(key)
        if source is None:
            source = self._seed_source_buffer(op, "vram")
            if source is None:
                self.missing_source_count += 1
                return
        if source.size_bytes != int(op["size_bytes"]):
            self.size_mismatch_count += 1
            return
        self.compute_count += 1
        self.compute_bytes += source.size_bytes

    def _evict_to_ssd(self, op: dict[str, Any]) -> None:
        key = self._key_from_op(op)
        source = self.vram.pop(key, None)
        if source is None:
            source = self.dram.pop(key, None)
        if source is None:
            source = self._seed_source_buffer(op, "vram")
            if source is None:
                self.missing_source_count += 1
                return
            self.vram.pop(key, None)
        self.evict_count += 1
        self.ssd_write_count += 1
        self.ssd_write_bytes += source.size_bytes

    def _read_backing_or_pattern(self, op: dict[str, Any]) -> bytes:
        size_bytes = int(op["size_bytes"])
        offset = int(op["ssd_offset"])
        if self.backing_file is not None:
            with self.backing_file.open("rb") as handle:
                handle.seek(offset)
                data = handle.read(size_bytes)
            if len(data) != size_bytes:
                self.size_mismatch_count += 1
                return data + bytes(size_bytes - len(data))
            return data
        seed = (
            int(op["layer_id"]) * 1315423911
            + int(op["block_id"]) * 2654435761
            + int(op["start_token"]) * 97
            + offset
        ) & 0xFF
        return bytes((seed + i) & 0xFF for i in range(size_bytes))

    def _buffer_from_op(self, op: dict[str, Any], data: bytes) -> RuntimeBuffer:
        return RuntimeBuffer(
            layer_id=int(op["layer_id"]),
            block_id=int(op["block_id"]),
            start_token=int(op["start_token"]),
            end_token=int(op["end_token"]),
            size_bytes=int(op["size_bytes"]),
            data=data,
        )

    def _key_from_op(self, op: dict[str, Any]) -> tuple[int, int, int, int]:
        return (int(op["layer_id"]), int(op["block_id"]), int(op["start_token"]), int(op["end_token"]))

    def _seed_source_buffer(self, op: dict[str, Any], location: str) -> RuntimeBuffer | None:
        if not self.seed_missing_sources:
            return None
        data = self._read_backing_or_pattern(op)
        buffer = self._buffer_from_op(op, data)
        if location == "dram":
            self.dram[buffer.key] = buffer
            self.dram_alloc_count += 1
        elif location == "vram":
            self.vram[buffer.key] = buffer
            self.vram_alloc_count += 1
        else:
            self.unsupported_op_count += 1
            return None
        self.lazy_source_seed_count += 1
        self.lazy_source_seed_bytes += buffer.size_bytes
        return buffer

    def _record_trace(self, op: dict[str, Any], index: int) -> None:
        metadata = dict(op.get("metadata", {}))
        metadata.update(
            {
                "executor": "runtime_residency_executor",
                "op_index": index,
                "movement_op": op.get("op"),
                "source": op.get("source"),
                "target": op.get("target"),
                "size_bytes": op.get("size_bytes"),
                "ssd_offset": op.get("ssd_offset"),
                "verification_error_count": self.verification_error_count,
            }
        )
        self.trace.record(
            op=f"residency_{op.get('op')}",
            layer_id=int(op["layer_id"]),
            block_id=int(op["block_id"]),
            device=str(op.get("target", "unknown")),
            start_ms=float(op.get("start_ms", index)),
            end_ms=float(op.get("end_ms", index)),
            metadata=metadata,
        )

    def _location_stats(self, buffers: dict[tuple[int, int, int, int], RuntimeBuffer]) -> dict[str, int]:
        return {
            "block_count": len(buffers),
            "size_bytes": sum(buffer.size_bytes for buffer in buffers.values()),
        }


def execute_runtime_residency_plan(
    plan_path: str | Path,
    output_dir: str | Path,
    *,
    backing_file: str | Path | None = None,
    seed_missing_sources: bool = True,
    run_id: str | None = None,
    clock=None,
) -> RuntimeResidencyExecutionResult:
    plan_path = Path(plan_path)
    output = Path(output_dir)
    backing_path = Path(backing_file) if backing_file is not None else None
    command = [
        "python",
        "-m",
        "harness.runtime_residency_executor",
        "--plan",
        str(plan_path),
        "--output-dir",
        str(output),
    ]
    inputs = [plan_path]
    if backing_path is not None:
        command.extend(["--backing-file", str(backing_path)])
        inputs.append(backing_path)
    if not seed_missing_sources:
        command.append("--no-seed-missing-sources")

    with RunContext(
        output,
        command=tuple(command),
        profile="runtime-residency-executor",
        inputs=tuple(inputs),
        io_backend="mock",
        run_id=run_id,
        clock=clock,
    ) as run:
        plan = _load_plan(plan_path)
        ops = list(plan.get("ops", []))
        if not isinstance(ops, list):
            raise ValueError("movement plan ops must be a list")

        executor = RuntimeResidencyExecutor(
            backing_file=backing_path, seed_missing_sources=seed_missing_sources
        )
        executor.execute(ops)

        metrics_path = output / "runtime_residency_metrics.json"
        trace_path = output / "runtime_residency_trace.json"
        summary_path = output / "runtime_residency_summary.json"
        result = RuntimeResidencyExecutionResult(
            plan_path=str(plan_path),
            output_dir=str(output),
            movement_op_count=len(ops),
            ssd_read_count=executor.ssd_read_count,
            dram_to_vram_copy_count=executor.dram_to_vram_copy_count,
            compute_count=executor.compute_count,
            evict_count=executor.evict_count,
            ssd_write_count=executor.ssd_write_count,
            ssd_read_bytes=executor.ssd_read_bytes,
            dram_to_vram_bytes=executor.dram_to_vram_bytes,
            compute_bytes=executor.compute_bytes,
            ssd_write_bytes=executor.ssd_write_bytes,
            dram_alloc_count=executor.dram_alloc_count,
            vram_alloc_count=executor.vram_alloc_count,
            lazy_source_seed_count=executor.lazy_source_seed_count,
            lazy_source_seed_bytes=executor.lazy_source_seed_bytes,
            missing_source_count=executor.missing_source_count,
            size_mismatch_count=executor.size_mismatch_count,
            unsupported_op_count=executor.unsupported_op_count,
            verification_error_count=executor.verification_error_count,
            final_residency=executor.final_residency(),
            metrics_path=str(metrics_path),
            trace_path=str(trace_path),
            summary_path=str(summary_path),
        )
        metrics_path.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        executor.trace.save_json(trace_path)
        summary = result.to_dict()
        summary["plan_layout"] = plan.get("layout", {})
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        run.add_output("metrics", metrics_path)
        run.add_output("trace", trace_path)
        run.add_output("summary", summary_path)
        if result.verification_error_count > 0:
            run.mark_failed(
                "VerificationError",
                "runtime residency verification errors: "
                f"{result.verification_error_count}",
            )
    return result


def _load_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("movement plan must be a JSON object")
    if "ops" not in payload:
        raise ValueError("movement plan missing ops")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backing-file", default=None)
    parser.add_argument("--no-seed-missing-sources", action="store_true")
    args = parser.parse_args()
    result = execute_runtime_residency_plan(
        args.plan,
        args.output_dir,
        backing_file=args.backing_file,
        seed_missing_sources=not args.no_seed_missing_sources,
    )
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0 if result.verification_error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
