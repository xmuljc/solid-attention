#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from common import EXP_ROOT, percentile, write_csv


RGB = tuple[int, int, int]
WHITE: RGB = (255, 255, 255)
BLACK: RGB = (0, 0, 0)
GRAY: RGB = (100, 100, 100)
LIGHT: RGB = (235, 235, 235)
ORANGE: RGB = (255, 127, 14)
ORANGE_DARK: RGB = (190, 84, 0)
ORANGE_LIGHT: RGB = (255, 190, 115)
BLUE: RGB = (31, 119, 180)
GREEN: RGB = (44, 160, 44)
RED: RGB = (214, 74, 64)
PURPLE: RGB = (148, 103, 189)

WORKLOAD_ORDER = [
    "gov_report",
    "multifieldqa_en",
    "musique",
    "narrativeqa",
    "qmsum",
    "multi_news",
    "repobench-p",
    "ruler_multikey_alternating",
]
WORKLOAD_LABELS = {
    "gov_report": "GovReport",
    "multifieldqa_en": "MultiFieldQA",
    "musique": "MuSiQue",
    "narrativeqa": "NarrativeQA",
    "qmsum": "QMSum",
    "multi_news": "MultiNews",
    "repobench-p": "RepoBench-P",
    "ruler_multikey_alternating": "Alt-MultiKey",
}
POLICY_LABELS = {
    "No-Selected-Prefetch": "无 Selected 预取",
    "Random-Same-Volume": "随机等量",
    "History-Last1": "上一 step",
    "Past-Frequency": "过去频率",
    "Oracle": "Oracle",
}
POLICY_COLORS = {
    "No-Selected-Prefetch": (150, 150, 150),
    "Random-Same-Volume": ORANGE_LIGHT,
    "History-Last1": ORANGE,
    "Past-Frequency": ORANGE_DARK,
    "Oracle": GREEN,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def text_box(draw: ImageDraw.ImageDraw, text: str, f: ImageFont.ImageFont) -> tuple[int, int]:
    b = draw.textbbox((0, 0), text, font=f)
    return b[2] - b[0], b[3] - b[1]


def center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, f: ImageFont.ImageFont, fill: RGB = BLACK) -> None:
    w, h = text_box(draw, text, f)
    draw.text((xy[0] - w / 2, xy[1] - h / 2), text, font=f, fill=fill)


def right(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, f: ImageFont.ImageFont, fill: RGB = BLACK) -> None:
    w, h = text_box(draw, text, f)
    draw.text((xy[0] - w, xy[1] - h / 2), text, font=f, fill=fill)


def rotate_label(img: Image.Image, center_xy: tuple[int, int], text: str, f: ImageFont.ImageFont) -> None:
    tmp = Image.new("RGBA", (10, 10), (255, 255, 255, 0))
    d = ImageDraw.Draw(tmp)
    w, h = text_box(d, text, f)
    label = Image.new("RGBA", (w + 12, h + 12), (255, 255, 255, 0))
    ld = ImageDraw.Draw(label)
    ld.text((6, 6), text, font=f, fill=BLACK)
    label = label.rotate(90, expand=True)
    img.alpha_composite(label, (center_xy[0] - label.width // 2, center_xy[1] - label.height // 2))


def scale(v: float, lo: float, hi: float, a: float, b: float) -> float:
    if hi == lo:
        return (a + b) / 2
    return a + (v - lo) / (hi - lo) * (b - a)


def save(img: Image.Image, png: Path, pdf: Path) -> None:
    png.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(png, format="PNG", dpi=(300, 300))
    img.convert("RGB").save(pdf, format="PDF", resolution=300.0)


def draw_axes(
    img: Image.Image,
    left: int,
    top: int,
    right_margin: int,
    bottom: int,
    y_ticks: list[float],
    ylo: float,
    yhi: float,
    y_label: str,
    x_label: str,
    font_path: Path,
) -> tuple[ImageDraw.ImageDraw, int, int, int, int]:
    draw = ImageDraw.Draw(img)
    f_tick = font(font_path, 22)
    f_label = font(font_path, 27)
    w, h = img.size
    x0, y0 = left, h - bottom
    x1, y1 = w - right_margin, top
    draw.line((x0, y0, x1, y0), fill=BLACK, width=3)
    draw.line((x0, y0, x0, y1), fill=BLACK, width=3)
    for tick in y_ticks:
        y = scale(tick, ylo, yhi, y0, y1)
        draw.line((x0 - 8, y, x0, y), fill=BLACK, width=2)
        right(draw, (x0 - 14, y), f"{tick:g}", f_tick)
        draw.line((x0, y, x1, y), fill=LIGHT, width=1)
    center(draw, ((x0 + x1) / 2, h - 48), x_label, f_label)
    rotate_label(img, (55, (y0 + y1) // 2), y_label, f_label)
    return draw, x0, y0, x1, y1


def header(draw: ImageDraw.ImageDraw, title: str, subtitle: str, font_path: Path) -> None:
    draw.text((85, 28), title, font=font(font_path, 34), fill=BLACK)
    draw.text((85, 74), subtitle, font=font(font_path, 21), fill=(75, 75, 75))


def plot_selection_locality(results: Path, plots: Path, font_path: Path) -> None:
    pred = read_csv(results / "prediction_metrics.csv")
    latency = read_csv(results / "workload_policy_latency_smoke.csv")
    h_pred = {r["dataset"]: r for r in pred if r["policy"] == "History-Last1"}
    h_lat = {r["dataset"]: r for r in latency if r["policy"] == "History-Last1"}
    rows = []
    for ds in WORKLOAD_ORDER:
        if ds not in h_pred or ds not in h_lat:
            continue
        rows.append({
            "workload": WORKLOAD_LABELS[ds],
            "semantic_recall_pct": float(h_pred[ds]["mean_recall"]) * 100.0,
            "semantic_ci_low_pct": float(h_pred[ds]["ci95_low_recall"]) * 100.0,
            "semantic_ci_high_pct": float(h_pred[ds]["ci95_high_recall"]) * 100.0,
            "jaccard_pct": float(h_pred[ds]["mean_jaccard"]) * 100.0,
            "jaccard_ci_low_pct": float(h_pred[ds]["ci95_low_jaccard"]) * 100.0,
            "jaccard_ci_high_pct": float(h_pred[ds]["ci95_high_jaccard"]) * 100.0,
            "on_time_recall_pct": float(h_lat[ds]["mean_on_time_recall"]) * 100.0,
            "on_time_ci_low_pct": float(h_lat[ds]["ci95_low_on_time_recall"]) * 100.0,
            "on_time_ci_high_pct": float(h_lat[ds]["ci95_high_on_time_recall"]) * 100.0,
        })
    write_csv(plots / "selection_locality_by_workload.csv", rows, list(rows[0].keys()))

    img = Image.new("RGBA", (1800, 940), WHITE + (255,))
    draw = ImageDraw.Draw(img)
    header(draw, "Speculative Prefetcher：同层上一 step 的选块局部性", "柱高来自真实 Qwen selected-block trace；误差线为 prompt bootstrap 95% CI（smoke：每任务 3 prompts）", font_path)
    draw, x0, y0, x1, y1 = draw_axes(img, 150, 145, 70, 155, [0, 25, 50, 75, 100], 0, 100, "比例（%）", "Workload", font_path)
    f_tick = font(font_path, 21)
    f_leg = font(font_path, 22)
    group_w = (x1 - x0) / len(rows)
    bar_w = group_w / 5.0
    colors = [ORANGE, BLUE, GREEN]
    labels = ["semantic recall", "Jaccard", "on-time recall"]
    keys = ["semantic_recall_pct", "jaccard_pct", "on_time_recall_pct"]
    ci_keys = [
        ("semantic_ci_low_pct", "semantic_ci_high_pct"),
        ("jaccard_ci_low_pct", "jaccard_ci_high_pct"),
        ("on_time_ci_low_pct", "on_time_ci_high_pct"),
    ]
    for i, row in enumerate(rows):
        gx = x0 + i * group_w + group_w / 2
        for j, key in enumerate(keys):
            val = row[key]
            bx0 = gx - 1.5 * bar_w + j * bar_w
            bx1 = bx0 + bar_w * 0.82
            by = scale(val, 0, 100, y0, y1)
            draw.rectangle((bx0, by, bx1, y0), fill=colors[j], outline=BLACK, width=1)
            lo, hi = row[ci_keys[j][0]], row[ci_keys[j][1]]
            ylo, yhi = scale(lo, 0, 100, y0, y1), scale(hi, 0, 100, y0, y1)
            cx = (bx0 + bx1) / 2
            draw.line((cx, ylo, cx, yhi), fill=BLACK, width=2)
            draw.line((cx - 5, ylo, cx + 5, ylo), fill=BLACK, width=2)
            draw.line((cx - 5, yhi, cx + 5, yhi), fill=BLACK, width=2)
        center(draw, (gx, y0 + 35), row["workload"], f_tick)
    lx = 920
    for i, label in enumerate(labels):
        draw.rectangle((lx + i * 245, 112, lx + i * 245 + 32, 136), fill=colors[i], outline=BLACK)
        draw.text((lx + i * 245 + 42, 107), label, font=f_leg, fill=BLACK)
    save(img, plots / "selection_locality_by_workload.png", plots / "selection_locality_by_workload.pdf")


def plot_policy_latency(results: Path, plots: Path, font_path: Path) -> None:
    rows = [r for r in read_csv(results / "policy_latency_summary_smoke.csv") if r["mode"] == "workload"]
    order = ["No-Selected-Prefetch", "Random-Same-Volume", "History-Last1", "Past-Frequency", "Oracle"]
    by = {r["policy"]: r for r in rows}
    out_rows = []
    for p in order:
        r = by[p]
        out_rows.append({
            "policy": p,
            "label": POLICY_LABELS[p],
            "p50_t_block_ms": r["p50_t_block_ms"],
            "p95_t_block_ms": r["p95_t_block_ms"],
            "p99_t_block_ms": r["p99_t_block_ms"],
            "mean_t_block_ms": r["mean_t_block_ms"],
            "pr_block_gt_0": r["pr_block_gt_0"],
        })
    write_csv(plots / "prefetch_policy_blocking_latency.csv", out_rows, list(out_rows[0].keys()))

    img = Image.new("RGBA", (1500, 900), WHITE + (255,))
    draw = ImageDraw.Draw(img)
    header(draw, "不同预取策略的 correction blocking latency", "真实 NVMe→DRAM→VRAM replay；显示 workload smoke 的 P50/P95/P99", font_path)
    maxv = max(float(r["p99_t_block_ms"]) for r in out_rows) * 1.18
    draw, x0, y0, x1, y1 = draw_axes(img, 150, 145, 90, 150, [0, round(maxv / 2, 2), round(maxv, 2)], 0, maxv, "T_block（ms）", "Policy", font_path)
    f_tick = font(font_path, 22)
    group_w = (x1 - x0) / len(out_rows)
    sub_w = group_w / 5
    metrics = [("p50_t_block_ms", "P50"), ("p95_t_block_ms", "P95"), ("p99_t_block_ms", "P99")]
    colors = [ORANGE_LIGHT, ORANGE, ORANGE_DARK]
    for i, row in enumerate(out_rows):
        gx = x0 + i * group_w + group_w / 2
        for j, (key, _) in enumerate(metrics):
            val = float(row[key])
            bx0 = gx - 1.5 * sub_w + j * sub_w
            bx1 = bx0 + sub_w * 0.82
            by = scale(val, 0, maxv, y0, y1)
            draw.rectangle((bx0, by, bx1, y0), fill=colors[j], outline=BLACK, width=1)
        center(draw, (gx, y0 + 35), row["label"], f_tick)
    for j, (_, label) in enumerate(metrics):
        draw.rectangle((1030 + j * 120, 112, 1062 + j * 120, 136), fill=colors[j], outline=BLACK)
        draw.text((1072 + j * 120, 107), label, font=font(font_path, 22), fill=BLACK)
    save(img, plots / "prefetch_policy_blocking_latency.png", plots / "prefetch_policy_blocking_latency.pdf")


def heat_color(v: float, lo: float, hi: float) -> RGB:
    if v >= 1.0:
        t = min(max((v - 1.0) / max(hi - 1.0, 1e-9), 0.0), 1.0)
        return (255 - int(70 * t), 190 - int(50 * t), 115 - int(90 * t))
    t = min(max((1.0 - v) / max(1.0 - lo, 1e-9), 0.0), 1.0)
    return (220 + int(30 * t), 130 - int(60 * t), 120 - int(55 * t))


def plot_heatmap(results: Path, plots: Path, font_path: Path) -> None:
    rows = read_csv(results / "lead_time_hit_rate_speedup_smoke.csv")
    write_csv(plots / "lead_time_hit_rate_speedup_heatmap.csv", rows, list(rows[0].keys()))
    hits = sorted({float(r["hit_rate"]) for r in rows})
    leads = sorted({float(r["lead_ms"]) for r in rows})
    vals = {(float(r["hit_rate"]), float(r["lead_ms"])): float(r["speedup_no_over_history"]) for r in rows}
    finite_vals = [v for v in vals.values() if math.isfinite(v)]
    lo = min(finite_vals)
    hi = max(finite_vals)
    cap_hi = hi * 1.15

    img = Image.new("RGBA", (1550, 940), WHITE + (255,))
    draw = ImageDraw.Draw(img)
    header(draw, "Controlled replay：命中率 / lead time 对延迟收益的影响", "单元格 = No-Prefetch / History 的平均 T_block；大于 1 表示 History 更快", font_path)
    f_tick = font(font_path, 22)
    f_cell = font(font_path, 20)
    f_label = font(font_path, 27)
    left, top = 190, 165
    cw, ch = 120, 64
    for i, lead in enumerate(leads):
        center(draw, (left + i * cw + cw / 2, top - 35), f"{lead:g}", f_tick)
    for j, hit in enumerate(hits):
        right(draw, (left - 18, top + j * ch + ch / 2), f"{hit:g}", f_tick)
    center(draw, (left + len(leads) * cw / 2, top - 82), "预取 lead time（ms）", f_label)
    rotate_label(img, (68, top + len(hits) * ch // 2), "semantic hit rate（%）", f_label)
    for j, hit in enumerate(hits):
        for i, lead in enumerate(leads):
            v = vals[(hit, lead)]
            v_draw = min(v, cap_hi) if math.isfinite(v) else cap_hi
            label = f"{v:.2f}" if math.isfinite(v) else f">{hi:.1f}"
            x, y = left + i * cw, top + j * ch
            draw.rectangle((x, y, x + cw - 6, y + ch - 5), fill=heat_color(v_draw, lo, cap_hi), outline=BLACK, width=1)
            center(draw, (x + (cw - 6) / 2, y + (ch - 5) / 2), label, f_cell)
    draw.rectangle((1120, 210, 1155, 235), fill=heat_color(0.9, lo, hi), outline=BLACK)
    draw.text((1170, 204), "负收益", font=font(font_path, 22), fill=BLACK)
    draw.rectangle((1120, 260, 1155, 285), fill=heat_color(max(hi, 1.2), lo, cap_hi), outline=BLACK)
    draw.text((1170, 254), "History 更快", font=font(font_path, 22), fill=BLACK)
    save(img, plots / "lead_time_hit_rate_speedup_heatmap.png", plots / "lead_time_hit_rate_speedup_heatmap.pdf")


def plot_bytes(results: Path, plots: Path, font_path: Path) -> None:
    rows = read_csv(results / "workload_policy_latency_smoke.csv")
    policies = ["Random-Same-Volume", "History-Last1", "Past-Frequency", "Oracle"]
    categories = [
        ("mean_useful_on_time_bytes", "按时有用", GREEN),
        ("mean_late_useful_bytes", "迟到有用", BLUE),
        ("mean_wrong_nvme_bytes", "错误预取", RED),
        ("mean_correction_bytes", "补读缺失", ORANGE),
    ]
    out_rows = []
    for p in policies:
        vals = [r for r in rows if r["policy"] == p]
        item = {"policy": p, "label": POLICY_LABELS[p]}
        for key, _, _ in categories:
            item[key] = sum(float(r[key]) for r in vals) / len(vals)
        out_rows.append(item)
    write_csv(plots / "useful_late_wasted_missing_bytes.csv", out_rows, ["policy", "label"] + [c[0] for c in categories])

    img = Image.new("RGBA", (1500, 900), WHITE + (255,))
    draw = ImageDraw.Draw(img)
    header(draw, "预取字节去向：按时 / 迟到 / 错误 / 补读", "workload smoke 平均值；错误预取会占用 SSD/H2D 资源，迟到有用块不计为 timely hit", font_path)
    maxv = max(sum(float(r[c[0]]) for c in categories) for r in out_rows) / 1024 / 1024 * 1.25
    draw, x0, y0, x1, y1 = draw_axes(img, 150, 145, 330, 150, [0, round(maxv / 2, 2), round(maxv, 2)], 0, maxv, "字节（MiB）", "Policy", font_path)
    f_tick = font(font_path, 23)
    group_w = (x1 - x0) / len(out_rows)
    bar_w = group_w * 0.42
    for i, row in enumerate(out_rows):
        gx = x0 + i * group_w + group_w / 2
        y = y0
        for key, _, color in categories:
            val = float(row[key]) / 1024 / 1024
            bh = scale(val, 0, maxv, 0, y0 - y1)
            draw.rectangle((gx - bar_w / 2, y - bh, gx + bar_w / 2, y), fill=color, outline=BLACK, width=1)
            y -= bh
        center(draw, (gx, y0 + 35), row["label"], f_tick)
    for i, (_, label, color) in enumerate(categories):
        y = 205 + i * 50
        draw.rectangle((1190, y, 1225, y + 25), fill=color, outline=BLACK)
        draw.text((1240, y - 5), label, font=font(font_path, 22), fill=BLACK)
    save(img, plots / "useful_late_wasted_missing_bytes.png", plots / "useful_late_wasted_missing_bytes.pdf")


def plot_scatter(results: Path, plots: Path, font_path: Path) -> None:
    rows = read_csv(results / "hit_rate_vs_latency_saving_smoke.csv")
    write_csv(plots / "hit_rate_vs_latency_saving.csv", rows, list(rows[0].keys()))
    xs = [float(r["on_time_recall"]) * 100 for r in rows]
    ys = [float(r["latency_saving_ms"]) for r in rows]
    ylo, yhi = min(ys) - 0.04, max(ys) + 0.04
    img = Image.new("RGBA", (1400, 900), WHITE + (255,))
    draw = ImageDraw.Draw(img)
    header(draw, "timely hit 与真实 latency saving 的关系", "每个点是 workload prompt；saving = No-Prefetch - History 的平均 T_block", font_path)
    draw, x0, y0, x1, y1 = draw_axes(img, 145, 145, 80, 150, [round(ylo, 2), 0, round(yhi, 2)], ylo, yhi, "latency saving（ms）", "on-time recall（%）", font_path)
    f_small = font(font_path, 20)
    for row in rows:
        x = scale(float(row["on_time_recall"]) * 100, 0, 100, x0, x1)
        y = scale(float(row["latency_saving_ms"]), ylo, yhi, y0, y1)
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=ORANGE, outline=BLACK)
    corr = json.loads((results / "correlation_summary_smoke.json").read_text(encoding="utf-8"))
    draw.text((920, 170), f"相关性 on-time vs saving: {corr['on_time_recall_vs_latency_saving']:.3f}", font=f_small, fill=BLACK)
    draw.text((920, 205), f"相关性 semantic vs saving: {corr['semantic_recall_vs_latency_saving']:.3f}", font=f_small, fill=BLACK)
    save(img, plots / "hit_rate_vs_latency_saving.png", plots / "hit_rate_vs_latency_saving.pdf")


def plot_cdf(results: Path, plots: Path, font_path: Path) -> None:
    raw = [r for r in read_jsonl(results / "raw_block_timing_smoke.jsonl") if r["mode"] == "workload" and r["policy"] in {"No-Selected-Prefetch", "History-Last1", "Past-Frequency", "Oracle"}]
    by: dict[str, list[float]] = defaultdict(list)
    for r in raw:
        by[r["policy"]].append(float(r["t_block_ms"]))
    cdf_rows = []
    for policy, vals in by.items():
        vals = sorted(vals)
        for i, v in enumerate(vals):
            cdf_rows.append({"policy": policy, "t_block_ms": v, "cdf": (i + 1) / len(vals)})
    write_csv(plots / "blocking_latency_cdf.csv", cdf_rows, ["policy", "t_block_ms", "cdf"])

    maxx = max(r["t_block_ms"] for r in cdf_rows) * 1.08
    img = Image.new("RGBA", (1400, 900), WHITE + (255,))
    draw = ImageDraw.Draw(img)
    header(draw, "每步 correction blocking latency CDF", "真实 workload smoke replay；越靠左越好", font_path)
    draw, x0, y0, x1, y1 = draw_axes(img, 145, 145, 100, 150, [0, 0.25, 0.5, 0.75, 1.0], 0, 1.0, "CDF", "T_block（ms）", font_path)
    f_leg = font(font_path, 22)
    order = ["No-Selected-Prefetch", "History-Last1", "Past-Frequency", "Oracle"]
    for policy in order:
        pts = sorted((float(r["t_block_ms"]), float(r["cdf"])) for r in cdf_rows if r["policy"] == policy)
        color = POLICY_COLORS[policy]
        last = None
        for xval, yval in pts:
            x = scale(xval, 0, maxx, x0, x1)
            y = scale(yval, 0, 1, y0, y1)
            if last:
                draw.line((last[0], last[1], x, last[1]), fill=color, width=4)
                draw.line((x, last[1], x, y), fill=color, width=4)
            last = (x, y)
    for i, policy in enumerate(order):
        y = 190 + i * 42
        draw.line((980, y, 1030, y), fill=POLICY_COLORS[policy], width=5)
        draw.text((1045, y - 15), POLICY_LABELS[policy], font=f_leg, fill=BLACK)
    save(img, plots / "blocking_latency_cdf.png", plots / "blocking_latency_cdf.pdf")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    args = parser.parse_args()
    if not args.font.exists():
        raise SystemExit(f"font not found: {args.font}")
    results = EXP_ROOT / "results"
    plots = results / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    plot_selection_locality(results, plots, args.font)
    plot_policy_latency(results, plots, args.font)
    plot_heatmap(results, plots, args.font)
    plot_bytes(results, plots, args.font)
    plot_scatter(results, plots, args.font)
    plot_cdf(results, plots, args.font)
    manifest = {
        "contract": "solidattention.speculative_prefetcher.plots.v1",
        "status": "ok",
        "font": str(args.font),
        "outputs": sorted(p.name for p in plots.iterdir() if p.is_file()),
        "note": "Chinese PNG/PDF plots generated from smoke CSV/JSONL. Not a formal 1000-trial paper-level run.",
    }
    (plots / "plot_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
