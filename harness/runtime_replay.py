"""Replay runtime llama.cpp KV traces through the simulation scheduler."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.kv_block import BlockLocation
from harness.io_probe import BlockIOProbeResult, run_block_io_probe
from core.movement_plan import build_movement_plan
from core.scheduler import SSDAwareScheduler, SchedulerConfig
from harness.metrics import MetricsCollector
from harness.run_manifest import RunContext
from harness.trace import TraceRecorder
from integrations.llama_cpp.adapter import LlamaKVCacheAdapter, load_jsonl_events


@dataclass(frozen=True)
class RuntimeReplayResult:
    trace_path: str
    output_dir: str
    event_count: int
    unique_block_count: int
    initial_location: str
    before_location_stats: dict[str, dict[str, int]]
    after_location_stats: dict[str, dict[str, int]]
    metrics_path: str
    scheduler_trace_path: str
    summary_path: str
    movement_plan_path: str
    ssd_layout_total_size_bytes: int
    ssd_layout_alignment: int
    ssd_read_latency_ms: float
    ssd_read_latency_source: str
    io_probe_result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_path": self.trace_path,
            "output_dir": self.output_dir,
            "event_count": self.event_count,
            "unique_block_count": self.unique_block_count,
            "initial_location": self.initial_location,
            "before_location_stats": self.before_location_stats,
            "after_location_stats": self.after_location_stats,
            "metrics_path": self.metrics_path,
            "scheduler_trace_path": self.scheduler_trace_path,
            "summary_path": self.summary_path,
            "movement_plan_path": self.movement_plan_path,
            "ssd_layout_total_size_bytes": self.ssd_layout_total_size_bytes,
            "ssd_layout_alignment": self.ssd_layout_alignment,
            "ssd_read_latency_ms": self.ssd_read_latency_ms,
            "ssd_read_latency_source": self.ssd_read_latency_source,
            "io_probe_result": self.io_probe_result,
        }


def run_runtime_replay(
    trace_path: str | Path,
    output_dir: str | Path,
    *,
    initial_location: str = "ssd",
    ssd_read_latency_ms: float = 1.0,
    dram_to_vram_latency_ms: float = 0.25,
    compute_latency_ms: float = 0.1,
    calibrate_ssd_io: bool = False,
    io_probe_executable: str | Path = "io/block_io_probe",
    io_probe_path: str | Path | None = None,
    ssd_layout_alignment: int = 4096,
    run_id: str | None = None,
    clock=None,
) -> RuntimeReplayResult:
    trace_path = Path(trace_path)
    output = Path(output_dir)
    command = [
        "python",
        "-m",
        "harness.runtime_replay",
        "--trace",
        str(trace_path),
        "--output-dir",
        str(output),
        "--initial-location",
        initial_location,
        "--ssd-read-latency-ms",
        str(ssd_read_latency_ms),
        "--dram-to-vram-latency-ms",
        str(dram_to_vram_latency_ms),
        "--compute-latency-ms",
        str(compute_latency_ms),
        "--ssd-layout-alignment",
        str(ssd_layout_alignment),
    ]
    inputs = [trace_path]
    if calibrate_ssd_io:
        command.extend(
            ["--calibrate-ssd-io", "--io-probe-executable", str(io_probe_executable)]
        )
        if Path(io_probe_executable).is_file():
            inputs.append(Path(io_probe_executable))
        if io_probe_path is not None:
            command.extend(["--io-probe-path", str(io_probe_path)])

    with RunContext(
        output,
        command=tuple(command),
        profile="runtime-replay",
        inputs=tuple(inputs),
        io_backend="mock",
        run_id=run_id,
        clock=clock,
    ) as run:
        events = load_jsonl_events(trace_path)
        adapter = LlamaKVCacheAdapter()
        adapter.ingest_many(events, duplicate_policy="replace")
        blocks = adapter.store.list_blocks()

        normalized_initial_location = _normalize_initial_location(initial_location)
        if normalized_initial_location != "trace":
            for block in blocks:
                block.move_to(normalized_initial_location)

        before_stats = adapter.store.location_stats()

        io_probe_result: BlockIOProbeResult | None = None
        generated_probe_path: Path | None = None
        ssd_read_latency_source = "configured"
        ssd_blocks = [block for block in blocks if block.location == BlockLocation.SSD]
        if calibrate_ssd_io and ssd_blocks:
            calibration_block_size = max(1, max(block.size_bytes for block in ssd_blocks))
            calibration_block_count = max(1, len(ssd_blocks))
            generated_probe_path = (
                Path(io_probe_path)
                if io_probe_path is not None
                else output / "runtime_replay_io_probe.bin"
            )
            io_probe_result = run_block_io_probe(
                io_probe_executable,
                generated_probe_path,
                block_size=calibration_block_size,
                block_count=calibration_block_count,
            )
            run.set_io_backend(io_probe_result.backend)
            ssd_read_latency_ms = io_probe_result.read_latency_ms_per_block
            ssd_read_latency_source = "io_probe"
        elif calibrate_ssd_io:
            ssd_read_latency_source = "no_ssd_blocks"

        metrics = MetricsCollector(
            metadata={
                "source": "llama.cpp runtime trace",
                "trace_path": str(trace_path),
                "initial_location": normalized_initial_location,
                "event_count": len(events),
                "unique_block_count": len(blocks),
                "ssd_read_latency_ms": ssd_read_latency_ms,
                "ssd_read_latency_source": ssd_read_latency_source,
                "io_probe_result": io_probe_result.to_dict() if io_probe_result is not None else None,
                "ssd_calibration_block_count": len(ssd_blocks),
                "ssd_calibration_size_bytes": sum(block.size_bytes for block in ssd_blocks),
            }
        )
        metrics.add(selected_block_count=len(blocks))
        scheduler_trace = TraceRecorder()
        scheduler = SSDAwareScheduler(
            adapter.store,
            SchedulerConfig(
                ssd_read_latency_ms=ssd_read_latency_ms,
                dram_to_vram_latency_ms=dram_to_vram_latency_ms,
                compute_latency_ms=compute_latency_ms,
            ),
            trace=scheduler_trace,
            metrics=metrics,
        )
        schedule = scheduler.schedule(blocks, step=0, start_ms=0.0)
        movement_plan = build_movement_plan(blocks, schedule, alignment=ssd_layout_alignment)

        metrics_path = output / "runtime_replay_metrics.json"
        trace_output_path = output / "runtime_replay_trace.json"
        summary_path = output / "runtime_replay_summary.json"
        movement_plan_path = output / "runtime_movement_plan.json"

        metrics.metadata["movement_plan_path"] = str(movement_plan_path)
        metrics.metadata["ssd_layout_alignment"] = ssd_layout_alignment
        metrics.metadata["ssd_layout_total_size_bytes"] = movement_plan.layout.total_size_bytes
        metrics.save_json(metrics_path)
        scheduler_trace.save_json(trace_output_path)
        movement_plan.save_json(movement_plan_path)

        result = RuntimeReplayResult(
            trace_path=str(trace_path),
            output_dir=str(output),
            event_count=len(events),
            unique_block_count=len(blocks),
            initial_location=normalized_initial_location,
            before_location_stats=before_stats,
            after_location_stats=adapter.store.location_stats(),
            metrics_path=str(metrics_path),
            scheduler_trace_path=str(trace_output_path),
            summary_path=str(summary_path),
            movement_plan_path=str(movement_plan_path),
            ssd_layout_total_size_bytes=movement_plan.layout.total_size_bytes,
            ssd_layout_alignment=ssd_layout_alignment,
            ssd_read_latency_ms=ssd_read_latency_ms,
            ssd_read_latency_source=ssd_read_latency_source,
            io_probe_result=io_probe_result.to_dict() if io_probe_result is not None else None,
        )
        payload = result.to_dict()
        payload["schedule"] = schedule.to_list()
        payload["movement_plan"] = movement_plan.to_dict()
        summary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        run.add_output("metrics", metrics_path)
        run.add_output("trace", trace_output_path)
        run.add_output("summary", summary_path)
        run.add_output("movement-plan", movement_plan_path)
        if generated_probe_path is not None:
            run.add_output("io-probe", generated_probe_path)
    return result


def _normalize_initial_location(location: str) -> str:
    if location == "trace":
        return location
    return BlockLocation(location).value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--initial-location", default="ssd", choices=("trace", "ssd", "dram", "vram"))
    parser.add_argument("--ssd-read-latency-ms", type=float, default=1.0)
    parser.add_argument("--dram-to-vram-latency-ms", type=float, default=0.25)
    parser.add_argument("--compute-latency-ms", type=float, default=0.1)
    parser.add_argument("--calibrate-ssd-io", action="store_true")
    parser.add_argument("--io-probe-executable", default="io/block_io_probe")
    parser.add_argument("--io-probe-path", default=None)
    parser.add_argument("--ssd-layout-alignment", type=int, default=4096)
    args = parser.parse_args()

    result = run_runtime_replay(
        args.trace,
        args.output_dir,
        initial_location=args.initial_location,
        ssd_read_latency_ms=args.ssd_read_latency_ms,
        dram_to_vram_latency_ms=args.dram_to_vram_latency_ms,
        compute_latency_ms=args.compute_latency_ms,
        calibrate_ssd_io=args.calibrate_ssd_io,
        io_probe_executable=args.io_probe_executable,
        io_probe_path=args.io_probe_path,
        ssd_layout_alignment=args.ssd_layout_alignment,
    )
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
