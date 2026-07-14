#!/usr/bin/env python3
"""Plot grouped bar charts for Figure 9 reproduction.

Requires matplotlib. The repository environment used by Codex may not have it;
install it in the experiment runtime before plotting real results.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

WORKLOAD_ORDER = ["GovReport", "MultiFieldQA", "MuSiQue", "NarrativeQA", "QMSum", "MultiNews", "RepoBench-P", "Alt-MultiKey"]
ANCHOR_WORKLOAD_ORDER = ["GovReport", "MultiFieldQA", "MuSiQue", "NarrativeQA"]
MODEL_ORDER = ["Llama-3.1-8B", "Qwen-2.5-7B"]
COLORS = {"Llama-3.1-8B": "#1f77b4", "Qwen-2.5-7B": "#ff7f0e"}
HATCHES = {"Llama-3.1-8B": "//", "Qwen-2.5-7B": "\\\\"}


def load_rows(path: Path, workloads: list[str], models: list[str]) -> dict[tuple[str, str], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for row in rows:
        out[(row["model"], row["workload"])] = row
    missing = [(m, w) for w in workloads for m in models if (m, w) not in out]
    if missing:
        raise ValueError(f"missing model/workload rows: {missing}")
    return out


def plot(csv_path: Path, output_dir: Path, *, with_ci: bool, anchor_only: bool, models: list[str]) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("matplotlib is required to generate PNG/PDF plots from real CSV values") from exc

    workloads = ANCHOR_WORKLOAD_ORDER if anchor_only else WORKLOAD_ORDER
    rows = load_rows(csv_path, workloads, models)
    x = list(range(len(workloads)))
    width = 0.36 if len(models) > 1 else 0.42
    fig, ax = plt.subplots(figsize=(10.2, 3.0), dpi=300)
    for idx, model in enumerate(models):
        if len(models) == 1:
            offset = 0.0
        else:
            offset = (idx - (len(models) - 1) / 2) * width
        means = [float(rows[(model, w)]["mean_overlap"]) for w in workloads]
        yerr = None
        if with_ci:
            lows = [float(rows[(model, w)]["ci95_low"]) for w in workloads]
            highs = [float(rows[(model, w)]["ci95_high"]) for w in workloads]
            yerr = [[m - lo for m, lo in zip(means, lows)], [hi - m for m, hi in zip(means, highs)]]
        ax.bar([i + offset for i in x], means, width, label=model, color=COLORS[model], edgecolor="black", linewidth=0.8, hatch=HATCHES[model], yerr=yerr, capsize=2 if with_ci else 0)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Similarity(%)")
    ax.set_xlabel("Workload")
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=0)
    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=2, frameon=False)
    fig.tight_layout(pad=0.35)
    if anchor_only:
        stem = "figure9_anchor_reproduction"
    else:
        stem = "selection_similarity_extended_with_ci" if with_ci else "selection_similarity_extended"
    png = output_dir / f"{stem}.png"
    fig.savefig(png, dpi=300)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)
    return png


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("experiments/figure9_reproduction/results/selection_similarity_values.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/figure9_reproduction/results"))
    parser.add_argument("--with-ci", action="store_true")
    parser.add_argument("--anchor-only", action="store_true", help="plot only the four paper anchor workloads")
    parser.add_argument("--models", default="Llama-3.1-8B,Qwen-2.5-7B", help="comma-separated models to plot")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    png = plot(args.csv, args.output_dir, with_ci=args.with_ci, anchor_only=args.anchor_only, models=models)
    print(png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
