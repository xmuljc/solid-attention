#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import EXP_ROOT, bootstrap_ci, load_config, mean, percentile, stable_int, stdev, write_csv


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return math.nan
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else math.nan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    results = EXP_ROOT / "results"
    rows = read_jsonl(args.raw)
    if not rows:
        raise SystemExit("empty raw timing")

    # Policy latency summary for workload replay.
    summary_rows: list[dict[str, Any]] = []
    group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group[(row["mode"], row["policy"])].append(row)
    for (mode, policy), vals in sorted(group.items()):
        t = [float(r["t_block_ms"]) for r in vals]
        summary_rows.append({
            "mode": mode,
            "policy": policy,
            "count": len(vals),
            "mean_t_block_ms": mean(t),
            "p50_t_block_ms": percentile(t, 0.50),
            "p95_t_block_ms": percentile(t, 0.95),
            "p99_t_block_ms": percentile(t, 0.99),
            "pr_block_gt_0": sum(1 for x in t if x > 0.0) / len(t),
            "max_t_block_ms": max(t),
            "mean_on_time_recall": mean([float(r["on_time_recall"]) for r in vals]),
            "mean_semantic_recall": mean([float(r["semantic_recall"]) for r in vals]),
            "mean_read_amplification": mean([float(r["read_amplification"]) for r in vals]),
            "mean_actual_qd": mean([float(r["actual_qd_max"]) for r in vals]),
        })
    write_csv(results / "policy_latency_summary_smoke.csv", summary_rows, [
        "mode", "policy", "count", "mean_t_block_ms", "p50_t_block_ms", "p95_t_block_ms",
        "p99_t_block_ms", "pr_block_gt_0", "max_t_block_ms", "mean_on_time_recall",
        "mean_semantic_recall", "mean_read_amplification", "mean_actual_qd",
    ])

    # Workload-level metrics, prompt clustered where prompt ids exist.
    prompt_acc: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["mode"] != "workload":
            continue
        key = (row["dataset"], row["sample_id"], row["policy"])
        for metric in ["t_block_ms", "semantic_recall", "on_time_recall", "read_amplification"]:
            prompt_acc[key][metric].append(float(row[metric]))
        for metric in ["useful_on_time_bytes", "late_useful_bytes", "wrong_nvme_bytes", "wrong_h2d_bytes", "correction_bytes"]:
            prompt_acc[key][metric].append(float(row[metric]))

    prompt_rows = []
    for (dataset, sample_id, policy), vals in prompt_acc.items():
        prompt_rows.append({
            "dataset": dataset,
            "sample_id": sample_id,
            "policy": policy,
            "event_count": len(vals["t_block_ms"]),
            **{f"mean_{k}": mean(v) for k, v in vals.items()},
        })
    write_csv(results / "policy_prompt_latency_smoke.csv", prompt_rows, [
        "dataset", "sample_id", "policy", "event_count", "mean_t_block_ms", "mean_semantic_recall",
        "mean_on_time_recall", "mean_read_amplification", "mean_useful_on_time_bytes",
        "mean_late_useful_bytes", "mean_wrong_nvme_bytes", "mean_wrong_h2d_bytes", "mean_correction_bytes",
    ])

    workload_rows = []
    by_workload_policy: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in prompt_rows:
        by_workload_policy[(row["dataset"], row["policy"])].append(row)
    for (dataset, policy), vals in sorted(by_workload_policy.items()):
        out = {
            "dataset": dataset,
            "policy": policy,
            "sample_count": len(vals),
            "event_count": sum(int(v["event_count"]) for v in vals),
        }
        for metric in [
            "t_block_ms", "semantic_recall", "on_time_recall", "read_amplification",
            "useful_on_time_bytes", "late_useful_bytes", "wrong_nvme_bytes", "wrong_h2d_bytes", "correction_bytes",
        ]:
            data = [float(v[f"mean_{metric}"]) for v in vals]
            lo, hi = bootstrap_ci(data, stable_int([config["random"]["bootstrap_seed"], dataset, policy, metric]), rounds=1000)
            out[f"mean_{metric}"] = mean(data)
            out[f"std_{metric}"] = stdev(data)
            out[f"ci95_low_{metric}"] = lo
            out[f"ci95_high_{metric}"] = hi
        workload_rows.append(out)
    fields = ["dataset", "policy", "sample_count", "event_count"]
    for metric in [
        "t_block_ms", "semantic_recall", "on_time_recall", "read_amplification",
        "useful_on_time_bytes", "late_useful_bytes", "wrong_nvme_bytes", "wrong_h2d_bytes", "correction_bytes",
    ]:
        fields += [f"mean_{metric}", f"std_{metric}", f"ci95_low_{metric}", f"ci95_high_{metric}"]
    write_csv(results / "workload_policy_latency_smoke.csv", workload_rows, fields)

    # Controlled lead x hit-rate table: History relative to No.
    controlled_rows = []
    ctrl_group: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["mode"] == "controlled":
            hit = float(row["dataset"].replace("hit_", ""))
            ctrl_group[(row["policy"], row["sample_id"].split("/")[0], hit)].append(row)
    # sample_id begins with lead_X.
    history_by: dict[tuple[float, float], list[float]] = defaultdict(list)
    no_by: dict[tuple[float, float], list[float]] = defaultdict(list)
    oracle_by: dict[tuple[float, float], list[float]] = defaultdict(list)
    for row in rows:
        if row["mode"] != "controlled":
            continue
        hit = float(row["dataset"].replace("hit_", ""))
        lead = float(row["prediction_lead_ms"])
        key = (hit, lead)
        if row["policy"] == "History-Last1":
            history_by[key].append(float(row["t_block_ms"]))
        elif row["policy"] == "No-Selected-Prefetch":
            no_by[key].append(float(row["t_block_ms"]))
        elif row["policy"] == "Oracle":
            oracle_by[key].append(float(row["t_block_ms"]))
    for key in sorted(history_by):
        hit, lead = key
        h = history_by[key]
        n = no_by[key]
        o = oracle_by[key]
        h_mean = mean(h)
        n_mean = mean(n)
        o_mean = mean(o)
        controlled_rows.append({
            "hit_rate": hit,
            "lead_ms": lead,
            "history_mean_t_block_ms": h_mean,
            "no_mean_t_block_ms": n_mean,
            "oracle_mean_t_block_ms": o_mean,
            "history_p50_t_block_ms": percentile(h, 0.50),
            "no_p50_t_block_ms": percentile(n, 0.50),
            "oracle_p50_t_block_ms": percentile(o, 0.50),
            "speedup_no_over_history": n_mean / h_mean if h_mean > 0 else math.inf,
            "latency_saving_ms": n_mean - h_mean,
            "oracle_gap_closed": (n_mean - h_mean) / (n_mean - o_mean) if abs(n_mean - o_mean) > 1e-12 else math.nan,
        })
    write_csv(results / "lead_time_hit_rate_speedup_smoke.csv", controlled_rows, [
        "hit_rate", "lead_ms", "history_mean_t_block_ms", "no_mean_t_block_ms", "oracle_mean_t_block_ms",
        "history_p50_t_block_ms", "no_p50_t_block_ms", "oracle_p50_t_block_ms",
        "speedup_no_over_history", "latency_saving_ms", "oracle_gap_closed",
    ])

    # Prompt-level recall vs saving.
    scatter_rows = []
    by_prompt_policy = {(r["dataset"], r["sample_id"], r["policy"]): r for r in prompt_rows}
    for row in prompt_rows:
        if row["policy"] != "History-Last1":
            continue
        key_no = (row["dataset"], row["sample_id"], "No-Selected-Prefetch")
        if key_no not in by_prompt_policy:
            continue
        no = by_prompt_policy[key_no]
        saving = float(no["mean_t_block_ms"]) - float(row["mean_t_block_ms"])
        scatter_rows.append({
            "dataset": row["dataset"],
            "sample_id": row["sample_id"],
            "semantic_recall": row["mean_semantic_recall"],
            "on_time_recall": row["mean_on_time_recall"],
            "latency_saving_ms": saving,
        })
    write_csv(results / "hit_rate_vs_latency_saving_smoke.csv", scatter_rows, [
        "dataset", "sample_id", "semantic_recall", "on_time_recall", "latency_saving_ms",
    ])
    corr_summary = {
        "semantic_recall_vs_latency_saving": corr([float(r["semantic_recall"]) for r in scatter_rows], [float(r["latency_saving_ms"]) for r in scatter_rows]),
        "on_time_recall_vs_latency_saving": corr([float(r["on_time_recall"]) for r in scatter_rows], [float(r["latency_saving_ms"]) for r in scatter_rows]),
    }
    (results / "correlation_summary_smoke.json").write_text(json.dumps(corr_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = {
        "contract": "solidattention.speculative_prefetcher.replay_analysis.v1",
        "status": "ok",
        "raw": str(args.raw),
        "rows": len(rows),
        "outputs": [
            "policy_latency_summary_smoke.csv",
            "policy_prompt_latency_smoke.csv",
            "workload_policy_latency_smoke.csv",
            "lead_time_hit_rate_speedup_smoke.csv",
            "hit_rate_vs_latency_saving_smoke.csv",
            "correlation_summary_smoke.json",
        ],
        "correlation": corr_summary,
    }
    (results / "replay_analysis_manifest_smoke.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
