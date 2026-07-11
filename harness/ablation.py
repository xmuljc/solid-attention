"""Simulation ablation harness for Phase 5-style experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core.block_selector import BlockSelector, SelectionConfig
from core.block_store import BlockStore
from core.kv_block import KVBlock
from core.prefetcher import PrefetchConfig, SpeculativePrefetcher
from core.scheduler import SSDAwareScheduler, SchedulerConfig
from harness.metrics import MetricsCollector
from harness.run_manifest import RunContext
from harness.smoke import DEFAULT_CONFIG, load_smoke_config
from harness.trace import TraceRecorder


METHODS = (
    "memory_only",
    "ssd_sync",
    "ssd_async",
    "prefetch_only",
    "full_scheduler",
    "solid_sim",
)
DEFAULT_OUTPUT_DIR = Path("outputs/ablation")
_SOLID_ALIAS = {"solid_sim": "full_scheduler"}


def run_ablation(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    run_id: str | None = None,
    clock=None,
) -> dict[str, Any]:
    output = Path(output_dir)
    with RunContext(
        output,
        command=(
            "python",
            "-m",
            "harness.ablation",
            "--config",
            str(config_path),
            "--output-dir",
            str(output),
        ),
        profile="ablation",
        config_path=config_path,
        inputs=(config_path,),
        io_backend="mock",
        run_id=run_id,
        clock=clock,
    ) as run:
        config = load_smoke_config(config_path)
        run.set_config_snapshot(config)

        method_summaries = {}
        for method in METHODS:
            result = _run_method(method, config, output, str(config_path))
            method_summaries[method] = result
            run.add_output("trace", result["trace"])
            run.add_output("metrics", result["metrics"])

        summary = {
            "config_path": str(config_path),
            "methods": list(METHODS),
            "results": method_summaries,
        }
        summary_path = output / "ablation_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary["summary_path"] = str(summary_path)
        summary["manifest_path"] = str(run.manifest_path)
        run.add_output("summary", summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simulation ablations.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    summary = run_ablation(args.config, args.output_dir)
    print(json.dumps({"methods": summary["methods"], "summary": summary["summary_path"]}, sort_keys=True))


def _run_method(method: str, config: dict[str, Any], output: Path, config_path: str) -> dict[str, Any]:
    method_kind = _method_kind(method)
    trace = TraceRecorder()
    metrics = MetricsCollector(metadata={"config_path": config_path, "method": method, "method_kind": method_kind})
    store = BlockStore()
    blocks = _make_blocks(config, method_kind)
    for block in blocks:
        store.add_block(block)

    selector = BlockSelector(
        SelectionConfig(init_block_count=1, local_block_count=1, selected_block_count=1),
        trace=trace,
    )
    scheduler = SSDAwareScheduler(
        store,
        SchedulerConfig(ssd_read_latency_ms=1.0, dram_to_vram_latency_ms=0.25, compute_latency_ms=0.1),
        trace=trace,
        metrics=metrics,
    )

    current_token = int(config["num_blocks_per_layer"]) * 32
    schedule_summaries = []
    for layer_id in range(int(config["num_layers"])):
        result = selector.select(
            blocks,
            layer_id=layer_id,
            current_token=current_token,
            query_vector=[1.0, 0.0],
            start_ms=float(layer_id),
        )
        metrics.add(selected_block_count=len(result.all_blocks))

        if method_kind == "ssd_async":
            _prefetch_exact_required_blocks(store, trace, metrics, result, layer_id)
        elif method_kind == "prefetch_only":
            _prefetch_selected_blocks(store, trace, metrics, result, layer_id)
        elif method_kind == "full_scheduler":
            _prefetch_selected_blocks(store, trace, metrics, result, layer_id)

        schedule = scheduler.schedule_selection_result(result, step=layer_id, start_ms=20.0 + layer_id)
        schedule_summaries.append(schedule.to_list())

    trace_path = output / f"{method}_trace.json"
    metrics_path = output / f"{method}_metrics.json"
    trace.save_json(trace_path)
    metrics.save_json(metrics_path)
    return {
        "trace": str(trace_path),
        "metrics": str(metrics_path),
        "metrics_payload": metrics.to_dict(),
        "schedules": schedule_summaries,
    }


def _prefetch_exact_required_blocks(
    store: BlockStore,
    trace: TraceRecorder,
    metrics: MetricsCollector,
    selection_result: object,
    step: int,
) -> None:
    prefetcher = SpeculativePrefetcher(
        store,
        PrefetchConfig(target_location="dram", mock_latency_ms=0.25, max_prefetch_blocks=None),
        trace=trace,
        metrics=metrics,
    )
    plan = prefetcher.plan(getattr(selection_result, "all_blocks"), step=step, start_ms=10.0 + step)
    prefetcher.execute(plan, start_ms=11.0 + step)


def _prefetch_selected_blocks(
    store: BlockStore,
    trace: TraceRecorder,
    metrics: MetricsCollector,
    selection_result: object,
    step: int,
) -> None:
    prefetcher = SpeculativePrefetcher(
        store,
        PrefetchConfig(target_location="dram", mock_latency_ms=0.25, max_prefetch_blocks=2),
        trace=trace,
        metrics=metrics,
    )
    plan = prefetcher.plan_from_selection_result(selection_result, step=step, start_ms=10.0 + step)
    prefetcher.execute(plan, start_ms=11.0 + step)
    prefetcher.evaluate(plan, getattr(selection_result, "all_blocks"), step=step, start_ms=12.0 + step)


def _method_kind(method: str) -> str:
    return _SOLID_ALIAS.get(method, method)


def _make_blocks(config: dict[str, Any], method: str) -> list[KVBlock]:
    num_layers = int(config["num_layers"])
    num_blocks_per_layer = int(config["num_blocks_per_layer"])
    block_size_bytes = int(config["block_size_bytes"])
    configured_locations = list(config["locations"])
    blocks: list[KVBlock] = []
    for layer_id in range(num_layers):
        for block_id in range(num_blocks_per_layer):
            if method == "memory_only":
                location = "vram"
            elif method in {"ssd_sync", "ssd_async", "prefetch_only"}:
                location = "ssd"
            else:
                location = configured_locations[(layer_id + block_id) % len(configured_locations)]
            block = KVBlock(
                block_id=block_id,
                layer_id=layer_id,
                start_token=block_id * 32,
                end_token=(block_id + 1) * 32,
                size_bytes=block_size_bytes,
                location=location,
            )
            block.set_representative([float(num_blocks_per_layer - block_id), float(block_id)])
            blocks.append(block)
    return blocks


if __name__ == "__main__":
    main()
