#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from common import EXP_ROOT, load_config, resolve, stable_int


POLICIES = [
    "No-Selected-Prefetch",
    "Random-Same-Volume",
    "History-Last1",
    "Past-Frequency",
    "Oracle",
]


def list_str(values: set[int] | list[int] | tuple[int, ...]) -> str:
    return ";".join(str(x) for x in sorted(values))


def load_trace_events(config: dict[str, Any]) -> list[dict[str, Any]]:
    trace_path = resolve(config["paths"]["selected_trace"])
    grouped: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    init = set(config["selection"]["init_blocks"])
    local = set(config["selection"]["local_blocks"])
    with trace_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            key = (rec["dataset"], rec["sample_id"], int(rec["layer_id"]), int(rec["head_id"]))
            grouped[key].append({
                "dataset": rec["dataset"],
                "sample_id": rec["sample_id"],
                "decode_step": int(rec["decode_step"]),
                "layer_id": int(rec["layer_id"]),
                "head_id": int(rec["head_id"]),
                "selected": set(int(x) for x in rec["selected_blocks"]) - init - local,
            })
    out: list[dict[str, Any]] = []
    for key, events in grouped.items():
        events.sort(key=lambda r: r["decode_step"])
        prev: set[int] | None = None
        recent: deque[set[int]] = deque(maxlen=4)
        for event in events:
            if prev is None:
                prev = set(event["selected"])
                recent.append(set(event["selected"]))
                continue
            event["history_last1"] = set(prev)
            event["history_recent"] = [set(x) for x in recent]
            out.append(event)
            prev = set(event["selected"])
            recent.append(set(event["selected"]))
    return out


def legal_candidates(candidate_count: int, init_blocks: set[int], local_blocks: set[int]) -> list[int]:
    return [i for i in range(candidate_count) if i not in init_blocks and i not in local_blocks]


def random_prediction(event: dict[str, Any], legal: list[int], k: int, seed: int) -> set[int]:
    rng = random.Random(stable_int([seed, event["dataset"], event["sample_id"], event["decode_step"], event["layer_id"], event["head_id"], "replay"]))
    return set(rng.sample(legal, k))


def frequency_prediction(history: list[set[int]], legal: list[int], k: int) -> set[int]:
    freq: dict[int, int] = defaultdict(int)
    recency: dict[int, int] = {}
    for rank, selected in enumerate(history, start=1):
        for b in selected:
            freq[b] += 1
            recency[b] = rank
    ranked = sorted(legal, key=lambda b: (-freq[b], -recency.get(b, -1), b))
    return set(ranked[:k])


def prediction_for_policy(policy: str, event: dict[str, Any], legal: list[int], k: int, seed: int) -> set[int]:
    selected = set(event["selected"])
    if policy == "No-Selected-Prefetch":
        return set()
    if policy == "Random-Same-Volume":
        return random_prediction(event, legal, k, seed)
    if policy == "History-Last1":
        return set(event["history_last1"])
    if policy == "Past-Frequency":
        return frequency_prediction(event["history_recent"], legal, k)
    if policy == "Oracle":
        return set(selected)
    raise ValueError(policy)


def write_events(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "event_id", "mode", "dataset", "sample_id", "decode_step", "layer_id", "head_id",
        "policy", "lead_ms", "slack_ms", "candidate_count", "selected_blocks",
        "predicted_blocks", "semantic_tp", "semantic_fp", "semantic_fn",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def make_workload_rows(config: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    timing = config["timing"]
    selected_k = int(config["selection"]["selected_k"])
    context_blocks = int(config["selection"]["context_budget_tokens"]) // int(config["selection"]["block_tokens"])
    init = set(config["selection"]["init_blocks"])
    local = set(config["selection"]["local_blocks"])
    legal = legal_candidates(context_blocks, init, local)
    events = load_trace_events(config)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_dataset[event["dataset"]].append(event)
    rng = random.Random(config["random"]["seed"])
    per_policy = int(timing["smoke_events_per_policy"] if mode == "smoke" else timing["formal_iterations_per_config"])
    per_dataset = max(1, per_policy // max(1, len(by_dataset)))
    sampled: list[dict[str, Any]] = []
    for dataset, rows in sorted(by_dataset.items()):
        sampled.extend(rng.sample(rows, min(per_dataset, len(rows))))
    if len(sampled) > per_policy:
        sampled = rng.sample(sampled, per_policy)

    out: list[dict[str, Any]] = []
    lead = float(timing["main_lead_ms"])
    slack = float(timing["correction_slack_ms"])
    seed = int(config["random"]["random_same_volume_seeds"][0])
    eid = 0
    for event in sampled:
        selected = set(event["selected"])
        for policy in POLICIES:
            pred = prediction_for_policy(policy, event, legal, selected_k, seed)
            out.append({
                "event_id": f"workload-{eid}",
                "mode": "workload",
                "dataset": event["dataset"],
                "sample_id": event["sample_id"],
                "decode_step": event["decode_step"],
                "layer_id": event["layer_id"],
                "head_id": event["head_id"],
                "policy": policy,
                "lead_ms": lead,
                "slack_ms": slack,
                "candidate_count": context_blocks,
                "selected_blocks": list_str(selected),
                "predicted_blocks": list_str(pred),
                "semantic_tp": len(selected & pred),
                "semantic_fp": len(pred - selected),
                "semantic_fn": len(selected - pred),
            })
            eid += 1
    return out


def make_controlled_rows(config: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    timing = config["timing"]
    selected_k = int(config["selection"]["selected_k"])
    # Controlled traces use a larger candidate universe so that 0% overlap is
    # mathematically possible for K=16. The real 1024-token smoke trace has only
    # 30 legal dynamic candidates after init/local removal.
    candidate_count = 128
    selected = set(range(10, 10 + selected_k))
    wrong_pool = [i for i in range(candidate_count) if i not in selected]
    repeats = int(timing["smoke_controlled_repeats"] if mode == "smoke" else timing["formal_iterations_per_config"])
    out: list[dict[str, Any]] = []
    eid = 0
    for hit_rate in timing["controlled_hit_rates"]:
        hit_count = int(round(selected_k * float(hit_rate) / 100.0))
        hit_blocks = sorted(selected)[:hit_count]
        wrong_blocks = wrong_pool[: selected_k - hit_count]
        history_pred = set(hit_blocks + wrong_blocks)
        for lead in timing["lead_times_ms"]:
            for rep in range(repeats):
                for policy, pred in [
                    ("No-Selected-Prefetch", set()),
                    ("History-Last1", history_pred),
                    ("Oracle", set(selected)),
                ]:
                    out.append({
                        "event_id": f"controlled-{eid}",
                        "mode": "controlled",
                        "dataset": f"hit_{hit_rate:g}",
                        "sample_id": f"lead_{lead:g}/rep_{rep}",
                        "decode_step": rep + 1,
                        "layer_id": 0,
                        "head_id": 0,
                        "policy": policy,
                        "lead_ms": float(lead),
                        "slack_ms": float(timing["correction_slack_ms"]),
                        "candidate_count": candidate_count,
                        "selected_blocks": list_str(selected),
                        "predicted_blocks": list_str(pred),
                        "semantic_tp": len(selected & pred),
                        "semantic_fp": len(pred - selected),
                        "semantic_fn": len(selected - pred),
                    })
                    eid += 1
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "formal"], default="smoke")
    args = parser.parse_args()
    config = load_config(args.config)
    results = EXP_ROOT / "results"
    rows = make_controlled_rows(config, args.mode) + make_workload_rows(config, args.mode)
    out_path = results / f"replay_events_{args.mode}.csv"
    write_events(out_path, rows)
    manifest = {
        "contract": "solidattention.speculative_prefetcher.replay_events.v1",
        "status": "ok",
        "mode": args.mode,
        "rows": len(rows),
        "output": str(out_path),
        "note": "Controlled rows use candidate_count=128 so 0% overlap is feasible; workload rows use real Qwen selected-block trace.",
    }
    (results / f"replay_events_{args.mode}_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
