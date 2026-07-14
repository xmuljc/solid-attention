#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from common import EXP_ROOT, load_config, load_qwen_shape, resolve, run_text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    results = EXP_ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)

    shape = load_qwen_shape(config)
    trace_path = resolve(config["paths"]["selected_trace"])
    manifest_path = resolve(config["paths"]["trace_manifest"])
    kv_file = resolve(config["paths"]["kv_test_file"])

    workload_events: Counter[str] = Counter()
    workload_prompts: dict[str, set[str]] = defaultdict(set)
    layers: set[int] = set()
    heads: set[int] = set()
    steps: set[int] = set()
    keys: list[str] | None = None
    selected_lengths: Counter[int] = Counter()
    selected_contains_fixed = 0
    init_blocks = set(config["selection"]["init_blocks"])
    local_blocks = set(config["selection"]["local_blocks"])

    with trace_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if keys is None:
                keys = sorted(rec)
            dataset = rec["dataset"]
            sample_id = rec["sample_id"]
            workload_events[dataset] += 1
            workload_prompts[dataset].add(sample_id)
            layers.add(int(rec["layer_id"]))
            heads.add(int(rec["head_id"]))
            steps.add(int(rec["decode_step"]))
            selected = set(int(x) for x in rec["selected_blocks"])
            selected_lengths[len(selected)] += 1
            if selected & (init_blocks | local_blocks):
                selected_contains_fixed += 1

    trace_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    kv_stat = kv_file.stat()
    allocated = kv_stat.st_blocks * 512

    env = {
        "contract": "solidattention.speculative_prefetcher.audit.v1",
        "status": "ok",
        "qwen_shape": shape,
        "trace": {
            "path": str(trace_path),
            "keys": keys,
            "event_count": sum(workload_events.values()),
            "workloads": dict(workload_events),
            "prompt_counts": {k: len(v) for k, v in workload_prompts.items()},
            "decode_step_min": min(steps) if steps else None,
            "decode_step_max": max(steps) if steps else None,
            "decode_step_count": len(steps),
            "layer_min": min(layers) if layers else None,
            "layer_max": max(layers) if layers else None,
            "layer_count": len(layers),
            "head_min": min(heads) if heads else None,
            "head_max": max(heads) if heads else None,
            "head_count": len(heads),
            "selected_lengths": dict(selected_lengths),
            "selected_events_containing_init_or_local": selected_contains_fixed,
            "available_workloads_note": "Available trace covers previous Figure 9 smoke workloads, not the newly requested Qasper/2WikiMQA/TriviaQA/HotpotQA set.",
        },
        "trace_manifest": {
            "model": trace_manifest.get("model"),
            "model_dir": trace_manifest.get("model_dir"),
            "load_dtype": trace_manifest.get("load_dtype"),
            "kv_trace_dtype": trace_manifest.get("kv_trace_dtype"),
            "datasets": trace_manifest.get("datasets"),
            "samples_per_dataset": trace_manifest.get("samples_per_dataset"),
            "max_new_tokens": trace_manifest.get("max_new_tokens"),
            "context_budget": trace_manifest.get("context_budget"),
            "block_size": trace_manifest.get("block_size"),
            "top_k": trace_manifest.get("top_k"),
            "recorded_layers": trace_manifest.get("recorded_layers"),
            "recorded_heads": trace_manifest.get("recorded_heads"),
        },
        "kv_test_file": {
            "path": str(kv_file),
            "size_bytes": kv_stat.st_size,
            "allocated_bytes": allocated,
            "appears_sparse": allocated < kv_stat.st_size,
        },
        "device_checks": {
            "findmnt": run_text(["findmnt", "-T", str(kv_file)]),
            "lsblk": run_text(["lsblk", "-o", "NAME,TYPE,SIZE,MOUNTPOINT,FSTYPE,MODEL"]),
            "nvme_list": run_text(["nvme", "list"]),
        },
    }

    (results / "environment.json").write_text(json.dumps(env, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (results / "trace_schema.json").write_text(json.dumps({"keys": keys, "selection_unit": "head_id"}, indent=2) + "\n", encoding="utf-8")

    md = [
        "# Existing Trace And Environment Audit",
        "",
        f"- Model: `{config['model']['name']}`",
        f"- Qwen config: `{config['model']['config_path']}`",
        f"- num layers: `{shape['num_hidden_layers']}`",
        f"- num query heads: `{shape['num_attention_heads']}`",
        f"- num KV heads: `{shape['num_key_value_heads']}`",
        f"- head dim: `{shape['head_dim']}`",
        f"- KV block bytes: `{shape['block_bytes']}`",
        f"- Trace path: `{trace_path}`",
        f"- Events: `{sum(workload_events.values())}`",
        f"- Workloads: `{', '.join(workload_events)}`",
        f"- Prompt counts: `{ {k: len(v) for k, v in workload_prompts.items()} }`",
        f"- Decode steps: `{min(steps)}..{max(steps)}`",
        f"- Layers: `{min(layers)}..{max(layers)}`",
        f"- Query heads / selection units: `{min(heads)}..{max(heads)}`",
        f"- Events where selected blocks contain init/local: `{selected_contains_fixed}`",
        f"- KV test file: `{kv_file}`",
        f"- KV file size: `{kv_stat.st_size}` bytes; allocated `{allocated}` bytes",
        "",
        "Important limitation: the available real trace covers the previous smoke workloads "
        "`gov_report`, `multifieldqa_en`, `musique`, `narrativeqa`, `qmsum`, `multi_news`, "
        "`repobench-p`, and `ruler_multikey_alternating`. It does not cover the newly listed "
        "`Qasper`, `2WikiMQA`, `TriviaQA`, and `HotpotQA` workloads. Those cannot be claimed "
        "without collecting new selected-block traces.",
        "",
        "The test file is on `/data/disk2`, which `findmnt` reports as `/dev/nvme1n1` ext4.",
    ]
    (results / "existing_trace_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "audit": str(results / "existing_trace_audit.md"), "block_bytes": shape["block_bytes"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
