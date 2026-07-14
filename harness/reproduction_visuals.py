"""Generate SolidAttention reproduction status visualizations.

The figures are intentionally static SVG files backed by existing harness
artifacts. They are meant for project reports and repository review, not for
runtime scheduling.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any


COLORS = {
    "bg": "#f8fafc",
    "card": "#ffffff",
    "ink": "#111827",
    "muted": "#64748b",
    "grid": "#d9e2ec",
    "ok": "#16a34a",
    "partial": "#f59e0b",
    "gap": "#dc2626",
    "pending": "#94a3b8",
    "dram_to_vram": "#2563eb",
    "ssd_read": "#0f766e",
    "vram_to_dram_eviction": "#db2777",
    "latency": "#2563eb",
    "eviction": "#db2777",
}


DEFAULT_TRACE = (
    "outputs/phase4/tier_policy_handoff/"
    "llama_attention_tier_event_kind_parity/runtime_kv_attention_tier_trace.jsonl"
)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    return int(round(as_float(row, key, float(default))))


def tag(color: str, label: str, x: float, y: float) -> str:
    return (
        f'<circle cx="{x}" cy="{y}" r="5" fill="{color}"/>'
        f'<text x="{x + 11}" y="{y + 4}" font-size="12" fill="{COLORS["muted"]}">'
        f"{esc(label)}</text>"
    )


def svg(width: int, height: int, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="{width}" height="{height}" fill="{COLORS["bg"]}"/>
<style>
text {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
.title {{ font-size: 24px; font-weight: 700; fill: {COLORS["ink"]}; }}
.subtitle {{ font-size: 13px; fill: {COLORS["muted"]}; }}
.label {{ font-size: 13px; fill: {COLORS["ink"]}; }}
.small {{ font-size: 11px; fill: {COLORS["muted"]}; }}
</style>
{body}
</svg>
"""


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", rx: int = 6) -> str:
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'


def text(x: float, y: float, value: object, klass: str = "label", anchor: str = "start") -> str:
    return f'<text x="{x:.2f}" y="{y:.2f}" class="{klass}" text-anchor="{anchor}">{esc(value)}</text>'


def line(x1: float, y1: float, x2: float, y2: float, color: str, width: float = 1.0) -> str:
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{color}" stroke-width="{width:.2f}"/>'
    )


def short_name(name: str) -> str:
    replacements = {
        "baseline_init_local": "baseline",
        "representative_token_position": "rep-select",
        "prefetch_next_view": "prefetch",
        "representative_prefetch": "rep+prefetch",
        "representative_sidecar": "rep-sidecar",
    }
    if name in replacements:
        return replacements[name]
    return name.replace("dram_", "d").replace("_vram_", " v").replace("_depth_", " q").replace("_prefetch_", " p")


def load_data(root: Path) -> dict[str, Any]:
    ablation_csv = root / "outputs/phase5/live_attention_ablation_suite/live_attention_ablation_table.csv"
    trace_parity = root / "outputs/phase4/tier_policy_handoff/tier_trace_parity_verification.json"
    sequence_parity = root / "outputs/phase4/tier_policy_handoff/tier_sequence_parity_verification.json"
    prefetch = root / "outputs/phase4/attention_prefetch_timing/attention_prefetch_timing_verification.json"
    suite = root / "outputs/phase5/live_attention_ablation_suite/live_attention_ablation_suite_summary.json"
    trace = root / DEFAULT_TRACE

    rows = read_csv(ablation_csv)
    events = read_jsonl(trace)
    return {
        "ablation_csv": str(ablation_csv.relative_to(root)),
        "rows": rows,
        "trace_path": str(trace.relative_to(root)),
        "events": events,
        "trace_parity": read_json(trace_parity, {}),
        "sequence_parity": read_json(sequence_parity, {}),
        "prefetch": read_json(prefetch, {}),
        "suite": read_json(suite, {}),
    }


def render_coverage_matrix(data: dict[str, Any]) -> str:
    trace = data["trace_parity"]
    sequence = data["sequence_parity"]
    prefetch = data["prefetch"]
    rows = data["rows"]
    has_phase5 = bool(rows)

    items = [
        ("Phase 1", "KV metadata + simulation harness", "ok", "pytest-covered"),
        ("Phase 2", "SSD queue / latency harness", "ok", "mock + probe artifacts"),
        ("Phase 3", "llama.cpp KV trace bridge", "ok", "runtime KV traces"),
        ("Phase 4", "live scheduler handoff", "ok", "attention gate + command bridge"),
        (
            "Phase 4",
            "speculative prefetch timing",
            "ok" if prefetch.get("prefetch_gap_status") == "closed" else "partial",
            f"{prefetch.get('command_prefetch_count', 0)} commands",
        ),
        (
            "Phase 4",
            "tier event-kind parity",
            "ok" if trace.get("tier_trace_gap_status") == "closed" else "partial",
            f"{trace.get('cxx_tier_event_count', 0)} C++ events",
        ),
        (
            "Phase 4",
            "tier block-identity parity",
            "gap" if sequence.get("status") == "failed" else "ok",
            sequence.get("tier_sequence_gap_status", "unknown"),
        ),
        (
            "Phase 5",
            "ablation suite metrics",
            "ok" if has_phase5 else "pending",
            f"{len(rows)} rows",
        ),
        ("Benchmark", "paper-scale benchmark", "pending", "not started"),
    ]

    col_x = [40, 150, 575, 700]
    row_h = 54
    y0 = 118
    body = [
        text(40, 42, "SolidAttention Reproduction Coverage", "title"),
        text(40, 66, "Harness-first status from existing project artifacts", "subtitle"),
        rect(30, 88, 820, 574, COLORS["card"], COLORS["grid"], 8),
        text(col_x[0], 116, "Area", "small"),
        text(col_x[1], 116, "Mechanism", "small"),
        text(col_x[2], 116, "Status", "small"),
        text(col_x[3], 116, "Evidence", "small"),
        line(40, 128, 835, 128, COLORS["grid"], 1),
    ]
    status_label = {"ok": "done", "partial": "partial", "gap": "gap", "pending": "pending"}
    for i, (phase, mechanism, status, evidence) in enumerate(items):
        y = y0 + i * row_h
        if i % 2 == 1:
            body.append(rect(40, y + 18, 795, row_h - 8, "#f1f5f9", "none", 4))
        body.append(text(col_x[0], y + 48, phase))
        body.append(text(col_x[1], y + 48, mechanism))
        body.append(rect(col_x[2], y + 29, 86, 24, COLORS[status], "none", 12))
        body.append(
            f'<text x="{col_x[2] + 43}" y="{y + 46}" font-size="12" font-weight="700" '
            f'fill="#ffffff" text-anchor="middle">{esc(status_label[status])}</text>'
        )
        body.append(text(col_x[3], y + 48, evidence, "small"))
    body.append(tag(COLORS["ok"], "implemented", 43, 700))
    body.append(tag(COLORS["gap"], "known reproduction gap", 155, 700))
    body.append(tag(COLORS["pending"], "not yet reproduced", 330, 700))
    return svg(880, 740, "\n".join(body))


def render_parity_ladder(data: dict[str, Any]) -> str:
    trace = data["trace_parity"]
    sequence = data["sequence_parity"]
    prefetch = data["prefetch"]
    steps = [
        ("Python scheduler", "live decisions + tier capacity", "ok"),
        ("Command bridge", f"{prefetch.get('command_prefetch_count', 0)} prefetch commands", "ok"),
        ("C++ runtime gate", f"queue depth {prefetch.get('runtime_executor_target_queue_depth', 0)}", "ok"),
        ("Event-kind parity", f"{trace.get('cxx_tier_event_count', 0)} events match", "ok"),
        (
            "Block identity parity",
            f"prefix {sequence.get('matching_prefix_event_count', 0)} / {sequence.get('cxx_event_count', 0)}",
            "gap" if sequence.get("status") == "failed" else "ok",
        ),
        ("Paper benchmark", "pending", "pending"),
    ]

    body = [
        text(40, 42, "Python vs C++ Parity Ladder", "title"),
        text(40, 66, "What currently matches, and where reproduction still diverges", "subtitle"),
        rect(30, 88, 820, 460, COLORS["card"], COLORS["grid"], 8),
    ]
    x0 = 90
    gap = 140
    y = 260
    for i, (label, detail, status) in enumerate(steps):
        x = x0 + i * gap
        if i > 0:
            body.append(line(x - gap + 36, y, x - 36, y, COLORS["grid"], 3))
        color = COLORS[status]
        body.append(f'<circle cx="{x}" cy="{y}" r="34" fill="{color}"/>')
        body.append(
            f'<text x="{x}" y="{y + 6}" font-size="18" font-weight="700" fill="#ffffff" text-anchor="middle">'
            f"{i + 1}</text>"
        )
        body.append(text(x, y + 62, label, "label", "middle"))
        body.append(text(x, y + 84, detail, "small", "middle"))

    counts = trace.get("cxx_decision_kind_counts", {})
    body.extend(
        [
            rect(70, 390, 720, 100, "#f8fafc", COLORS["grid"], 6),
            text(92, 420, "C++ tier event-kind counts", "label"),
            tag(COLORS["ssd_read"], f"ssd_read: {counts.get('ssd_read', 0)}", 96, 455),
            tag(COLORS["dram_to_vram"], f"dram_to_vram: {counts.get('dram_to_vram', 0)}", 246, 455),
            tag(
                COLORS["vram_to_dram_eviction"],
                f"vram_to_dram_eviction: {counts.get('vram_to_dram_eviction', 0)}",
                462,
                455,
            ),
        ]
    )
    return svg(880, 590, "\n".join(body))


def render_tier_timeline(data: dict[str, Any]) -> str:
    events = data["events"]
    shown = events[:160]
    counts = Counter(
        event.get("metadata", {}).get("decision_kind", event.get("decision_kind", "unknown"))
        for event in events
    )
    body = [
        text(40, 42, "Tier Movement Timeline", "title"),
        text(40, 66, f"First {len(shown)} of {len(events)} C++ tier movement events", "subtitle"),
        rect(30, 88, 1000, 505, COLORS["card"], COLORS["grid"], 8),
    ]
    lanes = [("ssd_read", 160), ("dram_to_vram", 275), ("vram_to_dram_eviction", 390)]
    for kind, y in lanes:
        body.append(line(90, y, 960, y, COLORS["grid"], 1))
        body.append(text(42, y + 4, kind, "small"))
    if shown:
        max_idx = max(1, len(shown) - 1)
        lane_by_kind = {kind: y for kind, y in lanes}
        for i, event in enumerate(shown):
            meta = event.get("metadata", {})
            kind = meta.get("decision_kind", event.get("decision_kind", "unknown"))
            x = 95 + i * (855 / max_idx)
            y = lane_by_kind.get(kind, 505)
            color = COLORS.get(kind, COLORS["pending"])
            layer = event.get("layer_id", "?")
            block = event.get("block_id", "?")
            view = meta.get("view", "?")
            body.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color}">'
                f"<title>event {i + 1}: {esc(kind)} layer={esc(layer)} block={esc(block)} view={esc(view)}</title>"
                "</circle>"
            )
    body.append(line(95, 475, 950, 475, COLORS["grid"], 1))
    for t in range(0, 161, 40):
        x = 95 + t * (855 / max(1, len(shown) - 1))
        body.append(line(x, 470, x, 480, COLORS["grid"], 1))
        body.append(text(x, 500, t, "small", "middle"))
    body.extend(
        [
            text(95, 535, f"ssd_read: {counts.get('ssd_read', 0)}", "small"),
            text(240, 535, f"dram_to_vram: {counts.get('dram_to_vram', 0)}", "small"),
            text(430, 535, f"evictions: {counts.get('vram_to_dram_eviction', 0)}", "small"),
            text(720, 535, "Note: timeline shows event-kind parity, not block identity parity.", "small"),
        ]
    )
    return svg(1060, 630, "\n".join(body))


def render_bar_panel(
    rows: list[dict[str, str]],
    x: float,
    y: float,
    width: float,
    height: float,
    metric: str,
    color: str,
    title: str,
    unit: str,
) -> list[str]:
    body: list[str] = [text(x, y, title, "label")]
    values = [as_float(row, metric) for row in rows]
    max_value = max(values) if values else 1.0
    max_value = max(max_value, 1.0)
    plot_x = x + 120
    bar_h = 22
    gap = 12
    for i, row in enumerate(rows):
        yy = y + 32 + i * (bar_h + gap)
        value = as_float(row, metric)
        bar_w = (width - 185) * value / max_value
        body.append(text(x, yy + 16, short_name(row.get("name", "")), "small"))
        body.append(rect(plot_x, yy, width - 185, bar_h, "#e2e8f0", "none", 4))
        body.append(rect(plot_x, yy, bar_w, bar_h, color, "none", 4))
        label = f"{value:.3f} {unit}" if unit else f"{value:.0f}"
        body.append(text(plot_x + bar_w + 8, yy + 16, label, "small"))
    return body


def render_ablation_bars(data: dict[str, Any]) -> str:
    rows = data["rows"]
    single = [row for row in rows if row.get("section") == "single_run"]
    tier_all = [row for row in rows if row.get("section") == "tier_grid"]
    tier: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in tier_all:
        key = (row.get("dram_capacity_bytes", ""), row.get("vram_capacity_bytes", ""))
        if key in seen:
            continue
        seen.add(key)
        tier.append(row)
    tier = sorted(tier, key=lambda row: (as_int(row, "vram_capacity_bytes"), as_int(row, "dram_capacity_bytes")))

    body = [
        text(40, 42, "Ablation Metrics Snapshot", "title"),
        text(40, 66, "Latency, blocking decisions, and tier evictions from Phase 5 artifacts", "subtitle"),
        rect(30, 88, 1020, 600, COLORS["card"], COLORS["grid"], 8),
    ]
    body.extend(
        render_bar_panel(
            single,
            55,
            125,
            460,
            250,
            "total_estimated_latency_ms",
            COLORS["latency"],
            "Single scheduler latency",
            "ms",
        )
    )
    body.extend(
        render_bar_panel(
            single,
            55,
            390,
            460,
            210,
            "blocking_decision_count",
            COLORS["partial"],
            "Blocking decisions",
            "",
        )
    )
    body.extend(
        render_bar_panel(
            tier,
            560,
            125,
            440,
            250,
            "total_estimated_latency_ms",
            COLORS["latency"],
            "Tier capacity latency",
            "ms",
        )
    )
    body.extend(
        render_bar_panel(
            tier,
            560,
            390,
            440,
            210,
            "tier_eviction_count",
            COLORS["eviction"],
            "Tier evictions",
            "",
        )
    )
    return svg(1080, 725, "\n".join(body))


def render_dashboard(output_names: list[str]) -> str:
    cards = "\n".join(
        f'<section><h2>{esc(name[3:-4].replace("_", " ").title())}</h2>'
        f'<img src="{esc(name)}" alt="{esc(name)}"/></section>'
        for name in output_names
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>SolidAttention Reproduction Status</title>
<style>
body {{ margin: 0; background: #e5e7eb; color: #111827; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 32px 20px; }}
h1 {{ margin: 0 0 8px; font-size: 30px; }}
p {{ color: #475569; margin: 0 0 24px; }}
section {{ background: white; border: 1px solid #cbd5e1; border-radius: 8px; margin: 0 0 24px; padding: 16px; }}
h2 {{ font-size: 18px; margin: 0 0 12px; }}
img {{ width: 100%; height: auto; display: block; }}
</style>
</head>
<body>
<main>
<h1>SolidAttention Reproduction Status</h1>
<p>Static figures generated from harness trace, parity verification, and ablation metrics artifacts.</p>
{cards}
</main>
</body>
</html>
"""


def summary_payload(data: dict[str, Any], output_names: list[str]) -> dict[str, Any]:
    trace = data["trace_parity"]
    sequence = data["sequence_parity"]
    prefetch = data["prefetch"]
    rows = data["rows"]
    return {
        "contract": "solidattention.reproduction_visuals.v1",
        "status": "ok",
        "figures": output_names,
        "source_artifacts": {
            "trace": data["trace_path"],
            "ablation_csv": data["ablation_csv"],
            "tier_trace_parity": "outputs/phase4/tier_policy_handoff/tier_trace_parity_verification.json",
            "tier_sequence_parity": "outputs/phase4/tier_policy_handoff/tier_sequence_parity_verification.json",
            "prefetch_timing": "outputs/phase4/attention_prefetch_timing/attention_prefetch_timing_verification.json",
        },
        "headline_metrics": {
            "tier_trace_gap_status": trace.get("tier_trace_gap_status"),
            "tier_sequence_gap_status": sequence.get("tier_sequence_gap_status"),
            "prefetch_gap_status": prefetch.get("prefetch_gap_status"),
            "cxx_tier_event_count": trace.get("cxx_tier_event_count"),
            "ablation_row_count": len(rows),
        },
    }


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("docs/reproduction_status"))
    args = parser.parse_args(argv)

    root = args.project_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(root)
    figures = {
        "01_reproduction_coverage_matrix.svg": render_coverage_matrix(data),
        "02_python_cpp_parity_ladder.svg": render_parity_ladder(data),
        "03_tier_movement_timeline.svg": render_tier_timeline(data),
        "04_ablation_latency_eviction.svg": render_ablation_bars(data),
    }
    for name, content in figures.items():
        write_text(output_dir / name, content)
    write_text(output_dir / "index.html", render_dashboard(list(figures)))

    summary = summary_payload(data, list(figures))
    (output_dir / "reproduction_visuals_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "output_dir": str(output_dir), "figures": list(figures)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
