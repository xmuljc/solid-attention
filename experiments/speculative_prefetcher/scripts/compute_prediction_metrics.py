#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from common import EXP_ROOT, bootstrap_ci, load_config, load_qwen_shape, mean, resolve, stable_int, stdev, write_csv


POLICIES = [
    "No-Selected-Prefetch",
    "Random-Same-Volume",
    "History-Last1",
    "Past-Frequency",
    "Oracle",
]


def parse_trace(trace_path: Path) -> dict[tuple[str, str, int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    with trace_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            key = (rec["dataset"], rec["sample_id"], int(rec["layer_id"]), int(rec["head_id"]))
            grouped[key].append(
                {
                    "dataset": rec["dataset"],
                    "sample_id": rec["sample_id"],
                    "decode_step": int(rec["decode_step"]),
                    "layer_id": int(rec["layer_id"]),
                    "head_id": int(rec["head_id"]),
                    "selected": tuple(int(x) for x in rec["selected_blocks"]),
                    "init_blocks": tuple(int(x) for x in rec.get("init_blocks", [])),
                    "local_blocks": tuple(int(x) for x in rec.get("local_blocks", [])),
                    "prompt_tokens": int(rec.get("prompt_tokens", 0)),
                }
            )
    for events in grouped.values():
        events.sort(key=lambda r: r["decode_step"])
    return grouped


def legal_candidates(candidate_count: int, init_blocks: set[int], local_blocks: set[int]) -> list[int]:
    return [i for i in range(candidate_count) if i not in init_blocks and i not in local_blocks]


def random_prediction(event: dict[str, Any], legal: list[int], k: int, seed: int) -> set[int]:
    rng = random.Random(stable_int([seed, event["dataset"], event["sample_id"], event["decode_step"], event["layer_id"], event["head_id"]]))
    return set(rng.sample(legal, k))


def frequency_prediction(history: deque[set[int]], legal: list[int], k: int) -> set[int]:
    if not history:
        return set()
    freq: Counter[int] = Counter()
    recent_rank: dict[int, int] = {}
    # Larger recent_rank means more recent.
    for rank, selected in enumerate(history, start=1):
        for block in selected:
            freq[block] += 1
            recent_rank[block] = rank
    ranked = sorted(legal, key=lambda b: (-freq[b], -recent_rank.get(b, -1), b))
    return set(ranked[:k])


def score_event(policy: str, predicted: set[int], selected: set[int], block_bytes: int) -> dict[str, Any]:
    tp = len(predicted & selected)
    fp = len(predicted - selected)
    fn = len(selected - predicted)
    union = len(predicted | selected)
    return {
        "policy": policy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_blocks": len(predicted),
        "selected_blocks": len(selected),
        "precision": tp / len(predicted) if predicted else (1.0 if not selected else 0.0),
        "recall": tp / len(selected) if selected else 1.0,
        "jaccard": tp / union if union else 1.0,
        "missing_blocks": fn,
        "wasted_blocks": fp,
        "missing_bytes": fn * block_bytes,
        "wasted_bytes": fp * block_bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bootstrap-rounds", type=int, default=2000)
    args = parser.parse_args()

    config = load_config(args.config)
    results = EXP_ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    shape = load_qwen_shape(config)
    trace_path = resolve(config["paths"]["selected_trace"])
    grouped = parse_trace(trace_path)

    selected_k = int(config["selection"]["selected_k"])
    block_tokens = int(config["selection"]["block_tokens"])
    context_budget = int(config["selection"]["context_budget_tokens"])
    candidate_count = context_budget // block_tokens
    init_blocks = set(int(x) for x in config["selection"]["init_blocks"])
    local_blocks = set(int(x) for x in config["selection"]["local_blocks"])
    legal = legal_candidates(candidate_count, init_blocks, local_blocks)
    if len(legal) < selected_k:
        raise SystemExit(f"not enough legal candidates: {len(legal)} < K={selected_k}")

    random_seeds = [int(x) for x in config["random"]["random_same_volume_seeds"]]
    event_rows: list[dict[str, Any]] = []
    cold_rows: list[dict[str, Any]] = []

    for (dataset, sample_id, layer_id, head_id), events in grouped.items():
        prev: set[int] | None = None
        recent: deque[set[int]] = deque(maxlen=4)
        for event in events:
            selected = set(event["selected"]) - init_blocks - local_blocks
            base = {
                "dataset": dataset,
                "sample_id": sample_id,
                "decode_step": event["decode_step"],
                "layer_id": layer_id,
                "head_id": head_id,
                "candidate_count": candidate_count,
                "selected_k": len(selected),
                "boundary_step": (event["decode_step"] % block_tokens) == 0,
            }
            if prev is None:
                cold_rows.append({**base, "reason": "no_previous_step"})
                prev = selected
                recent.append(selected)
                continue

            policy_predictions: dict[str, set[int]] = {
                "No-Selected-Prefetch": set(),
                "History-Last1": set(prev),
                "Past-Frequency": frequency_prediction(recent, legal, selected_k),
                "Oracle": set(selected),
            }

            for policy in ["No-Selected-Prefetch", "History-Last1", "Past-Frequency", "Oracle"]:
                row = {**base, **score_event(policy, policy_predictions[policy], selected, shape["block_bytes"])}
                event_rows.append(row)

            random_scores = []
            for seed in random_seeds:
                pred = random_prediction(event, legal, selected_k, seed)
                random_scores.append(score_event("Random-Same-Volume", pred, selected, shape["block_bytes"]))
            avg = {k: mean([float(s[k]) for s in random_scores]) for k in [
                "tp", "fp", "fn", "predicted_blocks", "selected_blocks", "precision", "recall",
                "jaccard", "missing_blocks", "wasted_blocks", "missing_bytes", "wasted_bytes"
            ]}
            event_rows.append({**base, "policy": "Random-Same-Volume", **avg})

            prev = selected
            recent.append(selected)

    event_fields = [
        "dataset", "sample_id", "decode_step", "layer_id", "head_id", "candidate_count", "selected_k",
        "boundary_step", "policy", "tp", "fp", "fn", "predicted_blocks", "selected_blocks", "precision",
        "recall", "jaccard", "missing_blocks", "wasted_blocks", "missing_bytes", "wasted_bytes",
    ]
    write_csv(results / "prediction_event_metrics.csv", event_rows, event_fields)
    write_csv(results / "cold_start_events.csv", cold_rows, ["dataset", "sample_id", "decode_step", "layer_id", "head_id", "candidate_count", "selected_k", "boundary_step", "reason"])

    # Prompt macro average first; prompt is the bootstrap cluster.
    prompt_acc: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in event_rows:
        key = (row["dataset"], row["sample_id"], row["policy"])
        for metric in ["precision", "recall", "jaccard", "missing_blocks", "wasted_blocks", "missing_bytes", "wasted_bytes"]:
            prompt_acc[key][metric].append(float(row[metric]))

    prompt_rows: list[dict[str, Any]] = []
    for (dataset, sample_id, policy), vals in prompt_acc.items():
        prompt_rows.append({
            "dataset": dataset,
            "sample_id": sample_id,
            "policy": policy,
            "event_count": len(vals["recall"]),
            **{f"mean_{m}": mean(v) for m, v in vals.items()},
        })
    prompt_rows.sort(key=lambda r: (r["dataset"], r["sample_id"], r["policy"]))
    prompt_fields = ["dataset", "sample_id", "policy", "event_count"] + [f"mean_{m}" for m in ["precision", "recall", "jaccard", "missing_blocks", "wasted_blocks", "missing_bytes", "wasted_bytes"]]
    write_csv(results / "prediction_prompt_metrics.csv", prompt_rows, prompt_fields)

    summary_rows: list[dict[str, Any]] = []
    grouped_prompt: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in prompt_rows:
        grouped_prompt[(row["dataset"], row["policy"])].append(row)
    boot_seed = int(config["random"]["bootstrap_seed"])
    for (dataset, policy), rows in grouped_prompt.items():
        out: dict[str, Any] = {
            "dataset": dataset,
            "policy": policy,
            "sample_count": len(rows),
            "event_count": sum(int(r["event_count"]) for r in rows),
        }
        for metric in ["recall", "precision", "jaccard", "missing_blocks", "wasted_blocks", "missing_bytes", "wasted_bytes"]:
            values = [float(r[f"mean_{metric}"]) for r in rows]
            ci_lo, ci_hi = bootstrap_ci(values, stable_int([boot_seed, dataset, policy, metric]), rounds=args.bootstrap_rounds)
            out[f"mean_{metric}"] = mean(values)
            out[f"std_{metric}"] = stdev(values)
            out[f"ci95_low_{metric}"] = ci_lo
            out[f"ci95_high_{metric}"] = ci_hi
        summary_rows.append(out)
    summary_rows.sort(key=lambda r: (r["dataset"], POLICIES.index(r["policy"])))
    summary_fields = ["dataset", "policy", "sample_count", "event_count"]
    for metric in ["recall", "precision", "jaccard", "missing_blocks", "wasted_blocks", "missing_bytes", "wasted_bytes"]:
        summary_fields.extend([f"mean_{metric}", f"std_{metric}", f"ci95_low_{metric}", f"ci95_high_{metric}"])
    write_csv(results / "prediction_metrics.csv", summary_rows, summary_fields)

    # Paired prompt differences: History vs Random, History vs Past-Frequency.
    by_prompt = {(r["dataset"], r["sample_id"], r["policy"]): r for r in prompt_rows}
    diff_rows: list[dict[str, Any]] = []
    for dataset in sorted({r["dataset"] for r in prompt_rows}):
        prompts = sorted({r["sample_id"] for r in prompt_rows if r["dataset"] == dataset})
        for other in ["Random-Same-Volume", "Past-Frequency"]:
            diffs = []
            for prompt in prompts:
                h = by_prompt[(dataset, prompt, "History-Last1")]
                o = by_prompt[(dataset, prompt, other)]
                diffs.append(float(h["mean_recall"]) - float(o["mean_recall"]))
            ci_lo, ci_hi = bootstrap_ci(diffs, stable_int([boot_seed, dataset, "History-Last1", other, "recall_diff"]), rounds=args.bootstrap_rounds)
            diff_rows.append({
                "dataset": dataset,
                "comparison": f"History-Last1 minus {other}",
                "sample_count": len(diffs),
                "mean_recall_diff": mean(diffs),
                "ci95_low": ci_lo,
                "ci95_high": ci_hi,
            })
    write_csv(results / "prediction_paired_differences.csv", diff_rows, ["dataset", "comparison", "sample_count", "mean_recall_diff", "ci95_low", "ci95_high"])

    manifest = {
        "contract": "solidattention.speculative_prefetcher.offline_metrics.v1",
        "status": "ok",
        "trace": str(trace_path),
        "event_rows": len(event_rows),
        "cold_start_events": len(cold_rows),
        "candidate_count": candidate_count,
        "legal_dynamic_candidates": len(legal),
        "selected_k": selected_k,
        "block_bytes": shape["block_bytes"],
        "outputs": [
            "prediction_event_metrics.csv",
            "prediction_prompt_metrics.csv",
            "prediction_metrics.csv",
            "prediction_paired_differences.csv",
            "cold_start_events.csv",
        ],
    }
    (results / "prediction_metrics_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
