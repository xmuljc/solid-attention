#!/usr/bin/env python3
"""Validate Figure 9 reproduction outputs against CSV values."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

WORKLOAD_ORDER = ["GovReport", "MultiFieldQA", "MuSiQue", "NarrativeQA", "QMSum", "MultiNews", "RepoBench-P", "Alt-MultiKey"]
MODEL_ORDER = ["Llama-3.1-8B", "Qwen-2.5-7B"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("experiments/figure9_reproduction/results"))
    parser.add_argument("--models", default="Llama-3.1-8B,Qwen-2.5-7B", help="comma-separated models expected in the CSV")
    args = parser.parse_args()
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    csv_path = args.results_dir / "selection_similarity_values.csv"
    png = args.results_dir / "selection_similarity_extended.png"
    pdf = args.results_dir / "selection_similarity_extended.pdf"
    ci_png = args.results_dir / "selection_similarity_extended_with_ci.png"
    for path in (csv_path, png, pdf, ci_png):
        if not path.exists() or path.stat().st_size == 0:
            raise SystemExit(f"missing or empty output: {path}")
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8", newline="")))
    keys = {(r["model"], r["workload"]) for r in rows}
    missing = [(m, w) for w in WORKLOAD_ORDER for m in models if (m, w) not in keys]
    if missing:
        raise SystemExit(f"missing bars in CSV: {missing}")
    for row in rows:
        value = float(row["mean_overlap"])
        if not (0.0 <= value <= 100.0):
            raise SystemExit(f"mean_overlap out of 0-100 range: {row}")
        if row["model"] in models and row["workload"] in WORKLOAD_ORDER:
            if int(row["sample_count"]) <= 0:
                raise SystemExit(f"sample_count must be positive for plotted row: {row}")
            if int(row["valid_transitions"]) <= 0:
                raise SystemExit(f"valid_transitions must be positive for plotted row: {row}")
    report = {
        "status": "ok",
        "workload_count": len(WORKLOAD_ORDER),
        "model_count": len(models),
        "bar_count": len(keys),
        "y_axis_expected": [0, 25, 50, 75, 100],
        "note": "Programmatic image-object overlap inspection requires a rendering backend; CSV/range/file checks passed.",
    }
    (args.results_dir / "plot_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
