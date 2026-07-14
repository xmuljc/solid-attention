#!/usr/bin/env python3
"""Compute Figure-9-style Selected Block similarity metrics.

Input JSONL schema, one record per sample/head/layer/decode step:
{
  "model": "Llama-3.1-8B",
  "dataset": "gov_report",
  "sample_id": "gov_report/0",
  "head_id": 0,
  "layer_id": 0,
  "decode_step": 1,
  "selected_blocks": [3, 8, ...],
  "init_blocks": [0],
  "local_blocks": [31]
}

Only selected_blocks are used for the primary overlap@K. init/local blocks are
only used for supplemental all-active overlap.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

WORKLOAD_ORDER = [
    ("gov_report", "GovReport"),
    ("multifieldqa_en", "MultiFieldQA"),
    ("musique", "MuSiQue"),
    ("narrativeqa", "NarrativeQA"),
    ("qmsum", "QMSum"),
    ("multi_news", "MultiNews"),
    ("repobench-p", "RepoBench-P"),
    ("ruler_multikey_alternating", "Alt-MultiKey"),
]
WORKLOAD_LABEL = dict(WORKLOAD_ORDER)
MODEL_ORDER = ["Llama-3.1-8B", "Qwen-2.5-7B"]
REQUIRED_K = 16


def norm_set(value: Any) -> set[int]:
    if value is None:
        return set()
    return {int(x) for x in value}


def overlap_percent(a: set[int], b: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(a & b) / k * 100.0


def jaccard_percent(a: set[int], b: set[int]) -> float:
    union = a | b
    if not union:
        return 100.0
    return len(a & b) / len(union) * 100.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def bootstrap_ci(values: list[float], *, seed: int, rounds: int = 2000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(rounds):
        means.append(mean(values[rng.randrange(n)] for _ in range(n)))
    return percentile(means, 0.025), percentile(means, 0.975)


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for key in ("model", "dataset", "sample_id", "layer_id", "decode_step", "selected_blocks"):
                if key not in rec:
                    raise ValueError(f"{path}:{lineno} missing {key}")
            records.append(rec)
    return records


def transition_rows(records: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        head_id = int(rec.get("head_id", 0))
        key = (str(rec["model"]), str(rec["dataset"]), str(rec["sample_id"]), head_id, int(rec["layer_id"]))
        grouped[key].append(rec)

    rows: list[dict[str, Any]] = []
    for (model, dataset, sample_id, head_id, layer_id), events in grouped.items():
        events.sort(key=lambda r: int(r["decode_step"]))
        prev = None
        for rec in events:
            if prev is None:
                prev = rec
                continue
            prev_sel = norm_set(prev.get("selected_blocks"))
            cur_sel = norm_set(rec.get("selected_blocks"))
            prev_all = prev_sel | norm_set(prev.get("init_blocks")) | norm_set(prev.get("local_blocks"))
            cur_all = cur_sel | norm_set(rec.get("init_blocks")) | norm_set(rec.get("local_blocks"))
            rows.append({
                "model": model,
                "dataset": dataset,
                "sample_id": sample_id,
                "head_id": head_id,
                "layer_id": layer_id,
                "decode_step_prev": int(prev["decode_step"]),
                "decode_step": int(rec["decode_step"]),
                "selected_prev": sorted(prev_sel),
                "selected_cur": sorted(cur_sel),
                "overlap": overlap_percent(prev_sel, cur_sel, k),
                "jaccard": jaccard_percent(prev_sel, cur_sel),
                "all_active_overlap": jaccard_percent(prev_all, cur_all),
            })
            prev = rec
    return rows


def summarize(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    bootstrap_rounds: int,
    model_order: list[str],
) -> list[dict[str, Any]]:
    sample_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sample_groups[(row["model"], row["dataset"], row["sample_id"])].append(row)

    sample_summaries: list[dict[str, Any]] = []
    for (model, dataset, sample_id), group in sample_groups.items():
        sample_summaries.append({
            "model": model,
            "dataset": dataset,
            "sample_id": sample_id,
            "valid_transitions": len(group),
            "mean_overlap": mean(r["overlap"] for r in group),
            "mean_jaccard": mean(r["jaccard"] for r in group),
            "mean_all_active_overlap": mean(r["all_active_overlap"] for r in group),
        })

    by_dataset: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sample_summaries:
        by_dataset[(row["model"], row["dataset"])].append(row)

    output = []
    for model in model_order:
        for dataset, label in WORKLOAD_ORDER:
            group = by_dataset.get((model, dataset), [])
            vals = [g["mean_overlap"] for g in group]
            jac = [g["mean_jaccard"] for g in group]
            all_active = [g["mean_all_active_overlap"] for g in group]
            low, high = bootstrap_ci(vals, seed=seed + abs(hash((model, dataset))) % 100000, rounds=bootstrap_rounds)
            output.append({
                "model": model,
                "dataset": dataset,
                "workload": label,
                "sample_count": len(group),
                "valid_transitions": sum(g["valid_transitions"] for g in group),
                "mean_overlap": mean(vals) if vals else 0.0,
                "std_overlap": pstdev(vals) if len(vals) > 1 else 0.0,
                "ci95_low": low,
                "ci95_high": high,
                "mean_jaccard": mean(jac) if jac else 0.0,
                "mean_all_active_overlap": mean(all_active) if all_active else 0.0,
            })
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model", "dataset", "workload", "sample_count", "valid_transitions", "mean_overlap",
        "std_overlap", "ci95_low", "ci95_high", "mean_jaccard", "mean_all_active_overlap",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="selected-block trace JSONL")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/figure9_reproduction/results"))
    parser.add_argument("--k", type=int, default=REQUIRED_K)
    parser.add_argument("--bootstrap-rounds", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--models",
        default="auto",
        help="comma-separated model order, or 'auto' to infer models from the trace",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.input)
    if args.models == "auto":
        present = {str(record["model"]) for record in records}
        model_order = [model for model in MODEL_ORDER if model in present]
        model_order.extend(sorted(present - set(model_order)))
    else:
        model_order = [item.strip() for item in args.models.split(",") if item.strip()]
    transitions = transition_rows(records, args.k)
    summary = summarize(transitions, seed=args.seed, bootstrap_rounds=args.bootstrap_rounds, model_order=model_order)

    raw_path = args.output_dir / "similarity_raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as fh:
        for row in transitions:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_csv(args.output_dir / "similarity_summary.csv", summary)
    write_csv(args.output_dir / "selection_similarity_values.csv", summary)
    print(json.dumps({"status": "ok", "models": model_order, "records": len(records), "transitions": len(transitions), "summary_rows": len(summary)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
