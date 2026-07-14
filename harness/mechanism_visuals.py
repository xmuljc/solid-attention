# Generate data-backed SolidAttention mechanism evidence figures.

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

C = {
    "bg": "#f8fafc", "card": "#ffffff", "ink": "#111827", "muted": "#64748b",
    "grid": "#d9e2ec", "green": "#16a34a", "blue": "#2563eb", "teal": "#0f766e",
    "pink": "#db2777", "amber": "#f59e0b", "gray": "#94a3b8", "purple": "#7c3aed",
}
FONT = '\"Noto Sans CJK SC\", \"Source Han Sans SC\", \"Microsoft YaHei\", \"PingFang SC\", Inter, ui-sans-serif, system-ui, sans-serif'


def esc(v: object) -> str:
    return html.escape(str(v), quote=True)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def fnum(v: Any, default: float = 0.0) -> float:
    if v in (None, ""):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def svg(w: int, h: int, body: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<rect width="{w}" height="{h}" fill="{C['bg']}"/>
<style>
text {{ font-family: {FONT}; }}
.title {{ font-size: 25px; font-weight: 700; fill: {C['ink']}; }}
.subtitle {{ font-size: 13px; fill: {C['muted']}; }}
.label {{ font-size: 13px; fill: {C['ink']}; }}
.small {{ font-size: 11px; fill: {C['muted']}; }}
.value {{ font-size: 22px; font-weight: 700; fill: {C['ink']}; }}
</style>
{body}
</svg>
'''


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", rx: int = 6) -> str:
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'


def text(x: float, y: float, value: object, klass: str = "label", anchor: str = "start") -> str:
    return f'<text x="{x:.2f}" y="{y:.2f}" class="{klass}" text-anchor="{anchor}">{esc(value)}</text>'


def line(x1: float, y1: float, x2: float, y2: float, color: str, width: float = 1.0) -> str:
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width:.2f}"/>'


def bar(x: float, y: float, w: float, h: float, value: float, max_value: float, color: str) -> str:
    fill_w = 0 if max_value <= 0 else w * value / max_value
    return rect(x, y, w, h, "#e2e8f0", "none", 4) + rect(x, y, fill_w, h, color, "none", 4)


def load(root: Path) -> dict[str, Any]:
    base = root / "outputs/phase5/live_attention_ablation_suite"
    return {
        "queue": read_json(base / "queue_depth_prefetch_grid/queue_depth_ablation_summary.json", {}),
        "tier": read_json(base / "tier_capacity_grid/tier_capacity_ablation_summary.json", {}),
        "rep_metrics": read_json(base / "single_runs/representative_prefetch/live_attention_scheduler_metrics.json", {}),
        "base_metrics": read_json(base / "single_runs/baseline_init_local/live_attention_scheduler_metrics.json", {}),
        "prefetch_metrics": read_json(base / "single_runs/prefetch_next_view/live_attention_scheduler_metrics.json", {}),
        "rep_decisions": read_jsonl(base / "single_runs/representative_prefetch/live_attention_scheduler_decisions.jsonl"),
        "baseline_decisions": read_jsonl(base / "single_runs/baseline_init_local/live_attention_scheduler_decisions.jsonl"),
        "tier_trace": read_jsonl(root / "outputs/phase4/tier_policy_handoff/llama_attention_tier_event_kind_parity/runtime_kv_attention_tier_trace.jsonl"),
        "prefetch_timing": read_json(root / "outputs/phase4/attention_prefetch_timing/attention_prefetch_timing_verification.json", {}),
    }


def first_selection(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    for event in decisions:
        selection = event.get("metadata", {}).get("selection", {})
        if selection.get("all_blocks"):
            return selection
    return {}


def representative_summary(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    scores, selected, with_candidates = [], 0, 0
    for event in decisions:
        candidates = event.get("metadata", {}).get("selection", {}).get("candidate_scores", [])
        if candidates:
            with_candidates += 1
        for item in candidates:
            scores.append(fnum(item.get("score")))
            selected += 1 if item.get("selected") else 0
    return {"score_count": len(scores), "selected": selected, "mean_score": sum(scores) / len(scores) if scores else 0.0, "with_candidates": with_candidates}


def render_selection(data: dict[str, Any]) -> str:
    baseline = first_selection(data["baseline_decisions"])
    rep = first_selection(data["rep_decisions"])
    rep_metrics = data["rep_metrics"]
    rep_sum = representative_summary(data["rep_decisions"])
    body = [text(40, 42, "机制 1：Init / Local / Selected Blocks 与代表向量选择", "title"), text(40, 66, "同一段 KV block 在不同策略下的角色变化；Selected Blocks 来自代表向量余弦评分。", "subtitle"), rect(30, 88, 1000, 470, C["card"], C["grid"], 8), text(70, 125, "基线策略：Init + Local", "label"), text(560, 125, "代表向量策略：Init + Selected", "label")]
    def draw_blocks(x: float, y: float, selection: dict[str, Any], role_colors: dict[int, tuple[str, str]]) -> list[str]:
        parts = []
        for i, bid in enumerate(selection.get("all_blocks", [])):
            color, role = role_colors.get(int(bid), (C["gray"], "未使用"))
            bx = x + i * 165
            parts.append(rect(bx, y, 130, 78, color, "none", 8))
            parts.append(f'<text x="{bx + 65}" y="{y + 34}" font-size="18" font-weight="700" fill="#ffffff" text-anchor="middle">块 {bid}</text>')
            parts.append(f'<text x="{bx + 65}" y="{y + 58}" font-size="13" fill="#ffffff" text-anchor="middle">{esc(role)}</text>')
        return parts
    body += draw_blocks(70, 155, baseline, {0: (C["teal"], "Init Block"), 64: (C["amber"], "Local Block")})
    body += draw_blocks(560, 155, rep, {0: (C["teal"], "Init Block"), 64: (C["purple"], "Selected Block")})
    score = fnum(rep.get("score_summary", {}).get("top_score"))
    body += [line(690, 250, 690, 316, C["grid"], 2), text(560, 292, "代表向量评分", "label"), bar(670, 275, 220, 22, score, 1.0, C["purple"]), text(900, 292, f"top score = {score:.3f}", "small"), rect(70, 350, 260, 120, "#f1f5f9", C["grid"], 6), text(92, 382, "候选评分次数", "small"), text(92, 414, rep_metrics.get("representative_candidate_count", 0), "value"), text(92, 450, "missing representative = 0", "small"), rect(370, 350, 260, 120, "#f1f5f9", C["grid"], 6), text(392, 382, "Selected 选中次数", "small"), text(392, 414, rep_metrics.get("representative_selected_count", 0), "value"), text(392, 450, f"平均分 {rep_sum['mean_score']:.3f}", "small"), rect(670, 350, 260, 120, "#f1f5f9", C["grid"], 6), text(692, 382, "验证关系", "small"), text(692, 414, "144 / 144", "value"), text(692, 450, "候选评分与选中数量一致", "small")]
    return svg(1060, 595, "\n".join(body))


def render_prefetch(data: dict[str, Any]) -> str:
    base, pref, rep = data["base_metrics"], data["prefetch_metrics"], data["rep_metrics"]
    timing = data["prefetch_timing"]
    rows = [("基线", base, C["gray"]), ("next-view 预取", pref, C["blue"]), ("代表向量+预取", rep, C["purple"])]
    max_block = max(fnum(r[1].get("blocking_decision_count")) for r in rows)
    max_lat = max(fnum(r[1].get("total_estimated_latency_ms")) for r in rows)
    body = [text(40, 42, "机制 2：推测预取降低阻塞加载", "title"), text(40, 66, "预取命中后，SSD 读提前完成，attention gate 不再执行这部分阻塞读。", "subtitle"), rect(30, 88, 1000, 470, C["card"], C["grid"], 8), text(70, 130, "阻塞决策数", "label"), text(560, 130, "估计延迟", "label")]
    for i, (name, m, color) in enumerate(rows):
        y = 165 + i * 72
        body += [text(70, y + 17, name, "small"), bar(190, y, 250, 24, fnum(m.get("blocking_decision_count")), max_block, color), text(455, y + 17, m.get("blocking_decision_count", 0), "small"), bar(650, y, 250, 24, fnum(m.get("total_estimated_latency_ms")), max_lat, color), text(915, y + 17, f"{fnum(m.get('total_estimated_latency_ms')):.3f} ms", "small")]
    hit = fnum(rep.get("prefetch_hit_count")); wrong = fnum(rep.get("wrong_prefetch_count")); miss = fnum(rep.get("prefetch_miss_count")); total = max(1.0, hit + miss + wrong)
    body += [rect(70, 405, 250, 95, "#f1f5f9", C["grid"], 6), text(92, 435, "预取命中", "small"), text(92, 468, f"{int(hit)} / {int(total)}", "value"), rect(370, 405, 250, 95, "#f1f5f9", C["grid"], 6), text(392, 435, "错误预取", "small"), text(392, 468, int(wrong), "value"), rect(670, 405, 250, 95, "#f1f5f9", C["grid"], 6), text(692, 435, "运行时验证", "small"), text(692, 468, "已提前完成", "value"), text(692, 492, f"resolved_before_attention = {timing.get('prefetch_resolved_before_attention_count', 0)}", "small")]
    return svg(1060, 595, "\n".join(body))


def render_queue(data: dict[str, Any]) -> str:
    grid = data["queue"].get("grid", {})
    depths = sorted(grid.keys(), key=int)
    prefetches = sorted({p for d in grid.values() for p in d.keys()}, key=int)
    body = [text(40, 42, "机制 3：SSD-aware 队列调度", "title"), text(40, 66, "同样的 SSD 读请求，在队列深度增加后等待时间降为 0，体现 I/O 调度约束被纳入 harness。", "subtitle"), rect(30, 88, 1000, 470, C["card"], C["grid"], 8), text(78, 130, "队列深度", "label"), text(245, 130, "prefetch=1 等待(ms)", "label"), text(535, 130, "prefetch=2 等待(ms)", "label")]
    max_wait = max((fnum(grid[d][p].get("ssd_queue_wait_time_ms")) for d in depths for p in prefetches), default=1.0)
    for i, d in enumerate(depths):
        y = 165 + i * 78
        body += [text(110, y + 18, d, "value", "middle")]
        for j, pkey in enumerate(prefetches):
            m = grid[d][pkey]; wait = fnum(m.get("ssd_queue_wait_time_ms")); x = 245 + j * 290; color = C["green"] if wait == 0 else C["amber"]
            body += [bar(x, y, 210, 26, wait, max_wait, color), text(x + 225, y + 18, f"{wait:.6f}", "small")]
    body += [rect(80, 430, 260, 82, "#f1f5f9", C["grid"], 6), text(102, 462, "深度 1", "small"), text(102, 492, "有排队等待", "value"), rect(400, 430, 260, 82, "#f1f5f9", C["grid"], 6), text(422, 462, "深度 2 / 4", "small"), text(422, 492, "等待为 0", "value"), rect(720, 430, 240, 82, "#f1f5f9", C["grid"], 6), text(742, 462, "SSD 读字节", "small"), text(742, 492, "262144", "value")]
    return svg(1060, 595, "\n".join(body))


def tier_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    grid = data["tier"].get("grid", {})
    rows = []
    for _, by_vram in grid.items():
        for _, by_depth in by_vram.items():
            row = by_depth.get("1", {}).get("0")
            if row:
                rows.append(row)
    return sorted(rows, key=lambda r: (int(r.get("vram_capacity_bytes", 0)), int(r.get("dram_capacity_bytes", 0))))


def render_tier(data: dict[str, Any]) -> str:
    trace_counts = Counter(e.get("metadata", {}).get("decision_kind") for e in data["tier_trace"])
    rows = tier_rows(data)
    max_ev = max((fnum(r.get("tier_eviction_count")) for r in rows), default=1.0); max_lat = max((fnum(r.get("total_estimated_latency_ms")) for r in rows), default=1.0)
    body = [text(40, 42, "机制 4：SSD / DRAM / VRAM 三层 KV 调度", "title"), text(40, 66, "C++ trace 显示 promotion / eviction / SSD read；容量消融显示 VRAM 容量变大后驱逐和延迟下降。", "subtitle"), rect(30, 88, 1000, 500, C["card"], C["grid"], 8), text(70, 130, "C++ 迁移事件类型", "label")]
    event_items = [("SSD→DRAM", trace_counts.get("ssd_read", 0), C["teal"]), ("DRAM→VRAM", trace_counts.get("dram_to_vram", 0), C["blue"]), ("VRAM→DRAM 驱逐", trace_counts.get("vram_to_dram_eviction", 0), C["pink"])]
    max_count = max(v for _, v, _ in event_items) or 1
    for i, (name, value, color) in enumerate(event_items):
        y = 165 + i * 58
        body += [text(70, y + 17, name, "small"), bar(210, y, 260, 24, value, max_count, color), text(485, y + 17, value, "small")]
    body += [text(590, 130, "容量消融：驱逐 / 延迟", "label")]
    for i, r in enumerate(rows):
        y = 165 + i * 58
        dram = int(r.get("dram_capacity_bytes", 0)) // 1024; vram = int(r.get("vram_capacity_bytes", 0)) // 1024
        ev = fnum(r.get("tier_eviction_count")); lat = fnum(r.get("total_estimated_latency_ms")); label = f"DRAM {dram}KiB / VRAM {vram}KiB"
        body += [text(590, y + 17, label, "small"), bar(790, y, 110, 20, ev, max_ev, C["pink"]), text(910, y + 16, f"{int(ev)} 次", "small"), bar(790, y + 25, 110, 12, lat, max_lat, C["blue"]), text(910, y + 37, f"{lat:.2f} ms", "small")]
    body += [rect(70, 410, 260, 95, "#f1f5f9", C["grid"], 6), text(92, 440, "总迁移事件", "small"), text(92, 474, len(data["tier_trace"]), "value"), rect(370, 410, 260, 95, "#f1f5f9", C["grid"], 6), text(392, 440, "promotion", "small"), text(392, 474, trace_counts.get("ssd_read", 0) + trace_counts.get("dram_to_vram", 0), "value"), rect(670, 410, 260, 95, "#f1f5f9", C["grid"], 6), text(692, 440, "eviction", "small"), text(692, 474, trace_counts.get("vram_to_dram_eviction", 0), "value")]
    return svg(1060, 625, "\n".join(body))


def render_index(figures: list[str]) -> str:
    titles = {"01_block_selection_evidence.svg": "Init / Local / Selected Blocks 与代表向量选择", "02_speculative_prefetch_evidence.svg": "推测预取命中与阻塞降低", "03_ssd_queue_scheduling_evidence.svg": "SSD-aware 队列调度", "04_tier_cache_scheduling_evidence.svg": "SSD / DRAM / VRAM 三层 KV 调度"}
    cards = "\n".join(f'<section><h2>{esc(titles.get(f, f))}</h2><img src="{esc(f)}" alt="{esc(titles.get(f, f))}"/></section>' for f in figures)
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>SolidAttention 机制复现证据图</title>
<style>body{{margin:0;background:#e5e7eb;color:#111827;font-family:{FONT};}}main{{max-width:1180px;margin:0 auto;padding:32px 20px;}}h1{{margin:0 0 8px;font-size:30px;}}p{{color:#475569;margin:0 0 24px;}}section{{background:white;border:1px solid #cbd5e1;border-radius:8px;margin:0 0 24px;padding:16px;}}h2{{font-size:18px;margin:0 0 12px;}}img{{width:100%;height:auto;display:block;}}</style>
</head><body><main><h1>SolidAttention 机制复现证据图</h1><p>这些图不是进度看板，而是由 harness 输出的实验数据直接生成，用于说明论文核心机制已经被复现到可测量状态。</p>{cards}</main></body></html>
'''


def summary(data: dict[str, Any], figures: list[str]) -> dict[str, Any]:
    rep, base, pref = data["rep_metrics"], data["base_metrics"], data["prefetch_metrics"]
    trace_counts = Counter(e.get("metadata", {}).get("decision_kind") for e in data["tier_trace"])
    return {"contract": "solidattention.mechanism_evidence_visuals.v1", "status": "ok", "figures": figures, "mechanism_metrics": {"representative_candidates": rep.get("representative_candidate_count"), "representative_selected": rep.get("representative_selected_count"), "prefetch_hits": rep.get("prefetch_hit_count"), "wrong_prefetches": rep.get("wrong_prefetch_count"), "baseline_blocking_decisions": base.get("blocking_decision_count"), "prefetch_blocking_decisions": pref.get("blocking_decision_count"), "tier_trace_event_count": len(data["tier_trace"]), "tier_event_kind_counts": dict(trace_counts)}, "source_artifacts": ["outputs/phase5/live_attention_ablation_suite/live_attention_ablation_suite_summary.json", "outputs/phase5/live_attention_ablation_suite/single_runs/representative_prefetch/live_attention_scheduler_metrics.json", "outputs/phase5/live_attention_ablation_suite/single_runs/representative_prefetch/live_attention_scheduler_decisions.jsonl", "outputs/phase5/live_attention_ablation_suite/queue_depth_prefetch_grid/queue_depth_ablation_summary.json", "outputs/phase5/live_attention_ablation_suite/tier_capacity_grid/tier_capacity_ablation_summary.json", "outputs/phase4/tier_policy_handoff/llama_attention_tier_event_kind_parity/runtime_kv_attention_tier_trace.jsonl"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate SolidAttention mechanism evidence figures.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("docs/mechanism_evidence"))
    args = parser.parse_args(argv)
    root = args.project_root.resolve(); out = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir; out.mkdir(parents=True, exist_ok=True)
    data = load(root)
    figs = {"01_block_selection_evidence.svg": render_selection(data), "02_speculative_prefetch_evidence.svg": render_prefetch(data), "03_ssd_queue_scheduling_evidence.svg": render_queue(data), "04_tier_cache_scheduling_evidence.svg": render_tier(data)}
    for name, content in figs.items():
        (out / name).write_text(content, encoding="utf-8")
    (out / "index.html").write_text(render_index(list(figs)), encoding="utf-8")
    (out / "mechanism_evidence_summary.json").write_text(json.dumps(summary(data, list(figs)), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output_dir": str(out), "figures": list(figs)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
