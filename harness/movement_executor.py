"""Execute runtime KV movement plans through the Phase 2 I/O probe."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.io_probe import BlockIOProbeResult, run_block_io_probe
from harness.run_manifest import RunContext
from harness.trace import TraceRecorder


@dataclass(frozen=True)
class MovementExecutionResult:
    plan_path: str
    output_dir: str
    movement_op_count: int
    kv_load_op_count: int
    ssd_load_op_count: int
    unique_ssd_read_count: int
    ssd_read_bytes: int
    io_probe_results: tuple[dict[str, Any], ...]
    metrics_path: str
    trace_path: str
    summary_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_path": self.plan_path,
            "output_dir": self.output_dir,
            "movement_op_count": self.movement_op_count,
            "kv_load_op_count": self.kv_load_op_count,
            "ssd_load_op_count": self.ssd_load_op_count,
            "unique_ssd_read_count": self.unique_ssd_read_count,
            "ssd_read_bytes": self.ssd_read_bytes,
            "io_probe_results": list(self.io_probe_results),
            "io_probe_total_read_latency_ms": sum(
                float(result["read_latency_ms"]) for result in self.io_probe_results
            ),
            "io_probe_total_write_latency_ms": sum(
                float(result["write_latency_ms"]) for result in self.io_probe_results
            ),
            "metrics_path": self.metrics_path,
            "trace_path": self.trace_path,
            "summary_path": self.summary_path,
        }


def execute_movement_plan(
    plan_path: str | Path,
    output_dir: str | Path,
    *,
    io_probe_executable: str | Path = "io/block_io_probe",
    execute_io: bool = True,
    run_id: str | None = None,
    clock=None,
) -> MovementExecutionResult:
    plan_path = Path(plan_path)
    output = Path(output_dir)
    command = [
        "python",
        "-m",
        "harness.movement_executor",
        "--plan",
        str(plan_path),
        "--output-dir",
        str(output),
    ]
    inputs = [plan_path]
    if execute_io:
        command.extend(["--io-probe-executable", str(io_probe_executable)])
        if Path(io_probe_executable).is_file():
            inputs.append(Path(io_probe_executable))
    else:
        command.append("--skip-io")

    with RunContext(
        output,
        command=tuple(command),
        profile="movement-executor",
        inputs=tuple(inputs),
        io_backend="mock",
        run_id=run_id,
        clock=clock,
    ) as run:
        plan = _load_plan(plan_path)
        ops = list(plan.get("ops", []))
        if not isinstance(ops, list):
            raise ValueError("movement plan ops must be a list")

        trace = TraceRecorder()
        for op in ops:
            _record_execution_trace(trace, op)

        ssd_load_ops = [
            op
            for op in ops
            if op.get("op") == "kv_load" and op.get("source") == "ssd"
        ]
        unique_reads = _unique_ssd_reads(ssd_load_ops)
        io_probe_results: list[BlockIOProbeResult] = []
        if execute_io:
            for size_bytes, entries in _group_reads_by_size(unique_reads).items():
                probe_path = output / f"movement_io_probe_{size_bytes}.bin"
                probe_result = run_block_io_probe(
                    io_probe_executable,
                    probe_path,
                    block_size=size_bytes,
                    block_count=len(entries),
                )
                io_probe_results.append(probe_result)
                run.add_output("io-probe", probe_path)

        if io_probe_results:
            backends = {result.backend for result in io_probe_results}
            if len(backends) != 1:
                reported = ", ".join(sorted(backends))
                raise RuntimeError(
                    f"movement I/O probes reported mixed backends: {reported}"
                )
            run.set_io_backend(next(iter(backends)))

        metrics_path = output / "movement_execution_metrics.json"
        trace_path = output / "movement_execution_trace.json"
        summary_path = output / "movement_execution_summary.json"

        result = MovementExecutionResult(
            plan_path=str(plan_path),
            output_dir=str(output),
            movement_op_count=len(ops),
            kv_load_op_count=sum(1 for op in ops if op.get("op") == "kv_load"),
            ssd_load_op_count=len(ssd_load_ops),
            unique_ssd_read_count=len(unique_reads),
            ssd_read_bytes=sum(int(item["size_bytes"]) for item in unique_reads),
            io_probe_results=tuple(item.to_dict() for item in io_probe_results),
            metrics_path=str(metrics_path),
            trace_path=str(trace_path),
            summary_path=str(summary_path),
        )

        metrics_path.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        trace.save_json(trace_path)
        summary = result.to_dict()
        summary["plan_layout"] = plan.get("layout", {})
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        run.add_output("metrics", metrics_path)
        run.add_output("trace", trace_path)
        run.add_output("summary", summary_path)
    return result


def _load_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("movement plan must be a JSON object")
    if "ops" not in payload:
        raise ValueError("movement plan missing ops")
    return payload


def _unique_ssd_reads(ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, int, int, int]] = set()
    unique: list[dict[str, Any]] = []
    for op in ops:
        key = (
            int(op["layer_id"]),
            int(op["block_id"]),
            int(op["ssd_offset"]),
            int(op["size_bytes"]),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(
            {
                "layer_id": key[0],
                "block_id": key[1],
                "ssd_offset": key[2],
                "size_bytes": key[3],
            }
        )
    return sorted(unique, key=lambda item: (item["ssd_offset"], item["layer_id"], item["block_id"]))


def _group_reads_by_size(reads: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for read in reads:
        groups.setdefault(int(read["size_bytes"]), []).append(read)
    return groups


def _record_execution_trace(trace: TraceRecorder, op: dict[str, Any]) -> None:
    metadata = dict(op.get("metadata", {}))
    metadata.update(
        {
            "movement_op": op.get("op"),
            "source": op.get("source"),
            "target": op.get("target"),
            "ssd_offset": op.get("ssd_offset"),
            "size_bytes": op.get("size_bytes"),
            "blocking": op.get("blocking"),
        }
    )
    trace.record(
        op=f"execute_{op.get('op')}",
        layer_id=int(op["layer_id"]),
        block_id=int(op["block_id"]),
        device=str(op.get("source", "unknown")),
        start_ms=float(op.get("start_ms", 0.0)),
        end_ms=float(op.get("end_ms", 0.0)),
        metadata=metadata,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--io-probe-executable", default="io/block_io_probe")
    parser.add_argument("--skip-io", action="store_true")
    args = parser.parse_args()

    result = execute_movement_plan(
        args.plan,
        args.output_dir,
        io_probe_executable=args.io_probe_executable,
        execute_io=not args.skip_io,
    )
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
