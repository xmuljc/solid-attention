"""Executable smoke harness for Phase 1 SolidAttention simulations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core.block_selector import BlockSelector, SelectionConfig
from core.block_store import BlockStore
from core.kv_block import KVBlock
from core.kv_layout import InterleavedKVLayout
from core.prefetcher import PrefetchConfig, SpeculativePrefetcher
from core.scheduler import SSDAwareScheduler, SchedulerConfig
from harness.metrics import MetricsCollector
from harness.run_manifest import RunContext
from harness.trace import TraceRecorder


DEFAULT_CONFIG = Path("configs/smoke_base.yaml")
DEFAULT_OUTPUT_DIR = Path("outputs/smoke")


def load_smoke_config(path: str | Path) -> dict[str, Any]:
    """Load the tiny smoke YAML subset used by this project.

    This parser intentionally supports only top-level scalars and string lists,
    enough for configs/smoke_base.yaml without adding PyYAML as a dependency.
    """

    config: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            if current_list_key is None:
                raise ValueError("list item without list key")
            config[current_list_key].append(line[1:].strip().strip('"\''))
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            config[key] = []
            current_list_key = key
        else:
            current_list_key = None
            config[key] = _parse_scalar(value)
    return config


def run_smoke(
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
            "harness.smoke",
            "--config",
            str(config_path),
            "--output-dir",
            str(output),
        ),
        profile="smoke",
        config_path=config_path,
        inputs=(config_path,),
        io_backend="mock",
        run_id=run_id,
        clock=clock,
    ) as run:
        config = load_smoke_config(config_path)
        run.set_config_snapshot(config)

        trace = TraceRecorder()
        metrics = MetricsCollector(
            metadata={"config_path": str(config_path), "harness": "smoke"}
        )
        store = BlockStore()
        blocks = _make_blocks(config)
        for block in blocks:
            store.add_block(block)

        layout = InterleavedKVLayout(key_size_bytes=64, value_size_bytes=64)
        for block in blocks:
            layout.add_block(block)

        selector = BlockSelector(
            SelectionConfig(
                init_block_count=1,
                local_block_count=1,
                selected_block_count=1,
            ),
            trace=trace,
        )
        selected_by_layer = []
        current_token = int(config["num_blocks_per_layer"]) * 32
        for layer_id in range(int(config["num_layers"])):
            result = selector.select(
                blocks,
                layer_id=layer_id,
                current_token=current_token,
                query_vector=[1.0, 0.0],
                start_ms=float(layer_id),
            )
            metrics.add(selected_block_count=len(result.all_blocks))
            selected_by_layer.append(result)

        prefetcher = SpeculativePrefetcher(
            store,
            PrefetchConfig(
                target_location="dram",
                mock_latency_ms=0.25,
                max_prefetch_blocks=4,
            ),
            trace=trace,
            metrics=metrics,
        )
        for step, result in enumerate(selected_by_layer):
            plan = prefetcher.plan_from_selection_result(
                result, step=step, start_ms=10.0 + step
            )
            prefetcher.execute(plan, start_ms=11.0 + step)
            prefetcher.evaluate(
                plan,
                result.all_blocks,
                step=step,
                start_ms=12.0 + step,
            )

        scheduler = SSDAwareScheduler(
            store,
            SchedulerConfig(
                ssd_read_latency_ms=1.0,
                dram_to_vram_latency_ms=0.25,
                compute_latency_ms=0.1,
            ),
            trace=trace,
            metrics=metrics,
        )
        schedules = [
            scheduler.schedule_selection_result(
                result, step=step, start_ms=20.0 + step
            )
            for step, result in enumerate(selected_by_layer)
        ]

        trace_path = output / "smoke_trace.json"
        metrics_path = output / "smoke_metrics.json"
        summary_path = output / "smoke_summary.json"
        trace.save_json(trace_path)
        metrics.save_json(metrics_path)

        summary = {
            "config": config,
            "block_count": len(blocks),
            "selection": [result.to_dict() for result in selected_by_layer],
            "layout": layout.to_list(),
            "schedules": [schedule.to_list() for schedule in schedules],
            "outputs": {
                "trace": str(trace_path),
                "metrics": str(metrics_path),
                "summary": str(summary_path),
                "manifest": str(run.manifest_path),
            },
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        run.add_output("trace", trace_path)
        run.add_output("metrics", metrics_path)
        run.add_output("summary", summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 1 smoke simulation harness.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    summary = run_smoke(args.config, args.output_dir)
    print(json.dumps({"outputs": summary["outputs"], "block_count": summary["block_count"]}, sort_keys=True))


def _make_blocks(config: dict[str, Any]) -> list[KVBlock]:
    num_layers = int(config["num_layers"])
    num_blocks_per_layer = int(config["num_blocks_per_layer"])
    block_size_bytes = int(config["block_size_bytes"])
    locations = list(config["locations"])
    blocks: list[KVBlock] = []
    for layer_id in range(num_layers):
        for block_id in range(num_blocks_per_layer):
            block = KVBlock(
                block_id=block_id,
                layer_id=layer_id,
                start_token=block_id * 32,
                end_token=(block_id + 1) * 32,
                size_bytes=block_size_bytes,
                location=locations[(layer_id + block_id) % len(locations)],
            )
            block.set_representative([float(num_blocks_per_layer - block_id), float(block_id)])
            blocks.append(block)
    return blocks


def _parse_scalar(value: str) -> int | float | str:
    value = value.strip().strip('"\'')
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


if __name__ == "__main__":
    main()
