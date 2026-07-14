#!/usr/bin/env python3
"""Generate Chinese smoke plots with real text rendered into PNG/PDF.

The original dependency-free smoke plotter only drew shapes into PNG and wrote
text into PDF, so PNG previews looked incomplete. This script uses Pillow plus
an explicit CJK font to render Chinese labels directly into the image.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


RGB = tuple[int, int, int]

BLUE: RGB = (31, 119, 180)
ORANGE: RGB = (255, 127, 14)
GREEN: RGB = (44, 160, 44)
GRAY: RGB = (110, 110, 110)
BLACK: RGB = (0, 0, 0)
WHITE: RGB = (255, 255, 255)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_font(font_path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(font_path), size)


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def draw_center(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: RGB = BLACK,
) -> None:
    tw, th = text_size(draw, text, font)
    draw.text((xy[0] - tw / 2, xy[1] - th / 2), text, font=font, fill=fill)


def draw_right(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: RGB = BLACK,
) -> None:
    tw, th = text_size(draw, text, font)
    draw.text((xy[0] - tw, xy[1] - th / 2), text, font=font, fill=fill)


def draw_rotated_label(
    image: Image.Image,
    center: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: RGB = BLACK,
) -> None:
    dummy = Image.new("RGBA", (10, 10), (255, 255, 255, 0))
    d = ImageDraw.Draw(dummy)
    tw, th = text_size(d, text, font)
    label = Image.new("RGBA", (tw + 8, th + 8), (255, 255, 255, 0))
    ld = ImageDraw.Draw(label)
    ld.text((4, 4), text, font=font, fill=fill)
    label = label.rotate(90, expand=True)
    image.alpha_composite(label, (center[0] - label.width // 2, center[1] - label.height // 2))


def scale(value: float, lo: float, hi: float, start: float, end: float) -> float:
    if hi == lo:
        return (start + end) / 2
    return start + (value - lo) / (hi - lo) * (end - start)


def save_outputs(image: Image.Image, png_path: Path, pdf_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(png_path, format="PNG", dpi=(300, 300))
    image.convert("RGB").save(pdf_path, format="PDF", resolution=300.0)


def draw_line_legend(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    color: RGB,
    label: str,
    font: ImageFont.ImageFont,
) -> None:
    draw.line((x, y, x + 44, y), fill=color, width=5)
    draw.ellipse((x + 18, y - 7, x + 32, y + 7), fill=color, outline=BLACK, width=1)
    draw.text((x + 58, y - 14), label, font=font, fill=BLACK)


def plot_hit_rate(csv_path: Path, output_dir: Path, font_path: Path) -> None:
    rows = [r for r in read_csv(csv_path) if r["scheme"] in {"B3", "B5"}]
    hits = sorted({float(r["prefetch_hit_rate"]) for r in rows})
    vals = [float(r["wall_p50_ms"]) for r in rows]
    lo = min(vals + [float(r["wall_p95_ms"]) for r in rows])
    hi = max(vals + [float(r["wall_p95_ms"]) for r in rows])
    pad = max((hi - lo) * 0.22, 0.04)
    ylo, yhi = lo - pad * 0.35, hi + pad

    w, h = 1600, 900
    left, top, right, bottom = 150, 130, 80, 150
    plot_w, plot_h = w - left - right, h - top - bottom
    img = Image.new("RGBA", (w, h), WHITE + (255,))
    draw = ImageDraw.Draw(img)
    f_title = load_font(font_path, 34)
    f_label = load_font(font_path, 28)
    f_tick = load_font(font_path, 23)
    f_small = load_font(font_path, 22)

    draw.text((left, 28), "预取命中率下降会增加 Attention 前阻塞延迟（Qwen smoke）", font=f_title, fill=BLACK)
    draw.text((left, 72), "指标：selection-known-to-output P50；误差线到 P95；数据来自本次 O_DIRECT/io_uring smoke CSV", font=f_small, fill=(70, 70, 70))

    # Axes.
    x0, y0 = left, h - bottom
    x1, y1 = w - right, top
    draw.line((x0, y0, x1, y0), fill=BLACK, width=3)
    draw.line((x0, y0, x0, y1), fill=BLACK, width=3)

    # Y ticks.
    tick_count = 5
    for i in range(tick_count):
        val = ylo + (yhi - ylo) * i / (tick_count - 1)
        y = scale(val, ylo, yhi, y0, y1)
        draw.line((x0 - 8, y, x0, y), fill=BLACK, width=2)
        draw_right(draw, (x0 - 14, y), f"{val:.2f}", f_tick)
        draw.line((x0, y, x1, y), fill=(232, 232, 232), width=1)

    # X ticks.
    for hit in hits:
        x = scale(hit, min(hits), max(hits), x0, x1)
        draw.line((x, y0, x, y0 + 8), fill=BLACK, width=2)
        draw_center(draw, (x, y0 + 34), f"{hit:g}", f_tick)

    draw_center(draw, ((x0 + x1) / 2, h - 50), "预取命中率（%）", f_label)
    draw_rotated_label(img, (55, (y0 + y1) // 2), "阻塞延迟（ms）", f_label)

    colors = {"B3": BLUE, "B5": ORANGE}
    labels = {"B3": "B3 屏障异步", "B5": "B5 到块即算流水"}
    for scheme in ("B3", "B5"):
        pts: list[tuple[float, float, float]] = []
        for hit in hits:
            row = next(r for r in rows if r["scheme"] == scheme and float(r["prefetch_hit_rate"]) == hit)
            p50 = float(row["wall_p50_ms"])
            p95 = float(row["wall_p95_ms"])
            x = scale(hit, min(hits), max(hits), x0, x1)
            y = scale(p50, ylo, yhi, y0, y1)
            y95 = scale(p95, ylo, yhi, y0, y1)
            pts.append((x, y, y95))
        for (xa, ya, _), (xb, yb, _) in zip(pts, pts[1:]):
            draw.line((xa, ya, xb, yb), fill=colors[scheme], width=5)
        for x, y, y95 in pts:
            draw.line((x, y, x, y95), fill=colors[scheme], width=2)
            draw.line((x - 8, y95, x + 8, y95), fill=colors[scheme], width=2)
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=colors[scheme], outline=BLACK, width=1)

    # Mark 81.25%.
    x81 = scale(81.25, min(hits), max(hits), x0, x1)
    draw.line((x81, y1, x81, y0), fill=GRAY, width=2)
    draw.text((x81 + 12, y1 + 12), "论文约 81.25%", font=f_small, fill=GRAY)

    draw_line_legend(draw, 1040, 42, BLUE, "B3 屏障异步", f_small)
    draw_line_legend(draw, 1260, 42, ORANGE, "B5 到块即算流水", f_small)

    save_outputs(
        img,
        output_dir / "barrier_vs_progressive_hit_rate_smoke.png",
        output_dir / "barrier_vs_progressive_hit_rate_smoke.pdf",
    )


def speed_color(value: float, lo: float, hi: float) -> RGB:
    if value >= 1.0:
        denom = max(hi - 1.0, 1e-6)
        t = min(max((value - 1.0) / denom, 0.0), 1.0)
        return (120 - int(50 * t), 190 + int(45 * t), 135 - int(35 * t))
    denom = max(1.0 - lo, 1e-6)
    t = min(max((1.0 - value) / denom, 0.0), 1.0)
    return (230 + int(20 * t), 155 - int(65 * t), 130 - int(45 * t))


def plot_heatmap(csv_path: Path, output_dir: Path, font_path: Path) -> None:
    rows = read_csv(csv_path)
    b3 = {(int(r["queue_depth"]), int(r["chunk_blocks"])): float(r["wall_p50_ms"]) for r in rows if r["scheme"] == "B3"}
    b5 = {(int(r["queue_depth"]), int(r["chunk_blocks"])): float(r["wall_p50_ms"]) for r in rows if r["scheme"] == "B5"}
    qds = sorted({q for q, _ in b3})
    chunks = sorted({ch for _, ch in b3})
    speeds = {(q, ch): b3[(q, ch)] / b5[(q, ch)] for q in qds for ch in chunks}
    lo, hi = min(speeds.values()), max(speeds.values())

    w, h = 1350, 900
    img = Image.new("RGBA", (w, h), WHITE + (255,))
    draw = ImageDraw.Draw(img)
    f_title = load_font(font_path, 34)
    f_label = load_font(font_path, 28)
    f_tick = load_font(font_path, 24)
    f_cell = load_font(font_path, 23)
    f_small = load_font(font_path, 22)

    draw.text((95, 34), "81.25% 命中率下的流水加速比热力图（Qwen smoke）", font=f_title, fill=BLACK)
    draw.text((95, 78), "单元格 = B3 屏障异步 P50 / B5 到块即算 P50；低于 1 表示负收益", font=f_small, fill=(70, 70, 70))

    left, top = 210, 170
    cell_w, cell_h = 150, 105
    for i, q in enumerate(qds):
        draw_center(draw, (left + i * cell_w + cell_w / 2, top - 34), str(q), f_tick)
    for j, ch in enumerate(chunks):
        draw_right(draw, (left - 24, top + j * cell_h + cell_h / 2), str(ch), f_tick)
    draw_center(draw, (left + len(qds) * cell_w / 2, top - 82), "io_uring 队列深度（QD）", f_label)
    draw_rotated_label(img, (65, top + len(chunks) * cell_h // 2), "micro-batch 块数", f_label)

    for i, q in enumerate(qds):
        for j, ch in enumerate(chunks):
            x = left + i * cell_w
            y = top + j * cell_h
            v = speeds[(q, ch)]
            color = speed_color(v, lo, hi)
            draw.rounded_rectangle((x, y, x + cell_w - 8, y + cell_h - 8), radius=4, fill=color, outline=BLACK, width=2)
            draw_center(draw, (x + (cell_w - 8) / 2, y + (cell_h - 8) / 2), f"{v:.3f}", f_cell)

    # Compact legend.
    lx, ly = 1030, 220
    draw.rectangle((lx, ly, lx + 34, ly + 24), fill=(245, 90, 85), outline=BLACK)
    draw.text((lx + 46, ly - 4), "负收益", font=f_small, fill=BLACK)
    draw.rectangle((lx, ly + 48, lx + 34, ly + 72), fill=(70, 235, 100), outline=BLACK)
    draw.text((lx + 46, ly + 44), "有加速", font=f_small, fill=BLACK)
    draw.text((95, 805), f"范围：{lo:.3f} - {hi:.3f}；此 smoke 结果中 B5 未稳定击败 B3。", font=f_small, fill=(70, 70, 70))

    save_outputs(
        img,
        output_dir / "speedup_heatmap_81pct_smoke.png",
        output_dir / "speedup_heatmap_81pct_smoke.pdf",
    )


def plot_breakdown(csv_path: Path, output_dir: Path, font_path: Path) -> None:
    rows = [
        r
        for r in read_csv(csv_path)
        if float(r["prefetch_hit_rate"]) == 81.25
        and int(r["queue_depth"]) == 4
        and int(r["chunk_blocks"]) == 1
        and r["scheme"] in {"B3", "B4", "B5"}
    ]
    rows.sort(key=lambda r: r["scheme"])
    parts = [
        ("ssd_to_dram_mean_ms", "SSD→DRAM", BLUE),
        ("h2d_mean_ms", "H2D", ORANGE),
        ("attention_mean_ms", "Attention/合并", GREEN),
    ]
    maxv = max(sum(float(row[k]) for k, _, _ in parts) for row in rows)

    w, h = 1350, 900
    img = Image.new("RGBA", (w, h), WHITE + (255,))
    draw = ImageDraw.Draw(img)
    f_title = load_font(font_path, 34)
    f_label = load_font(font_path, 28)
    f_tick = load_font(font_path, 23)
    f_small = load_font(font_path, 22)

    draw.text((95, 34), "81.25% 命中率下的延迟分解（Qwen smoke）", font=f_title, fill=BLACK)
    draw.text((95, 78), "堆叠项为诊断用均值，不等同于最终重叠区间核算；柱高来自 qd_chunk CSV", font=f_small, fill=(70, 70, 70))

    left, top, right, bottom = 150, 150, 370, 150
    x0, y0 = left, h - bottom
    x1, y1 = w - right, top
    draw.line((x0, y0, x1, y0), fill=BLACK, width=3)
    draw.line((x0, y0, x0, y1), fill=BLACK, width=3)
    for i in range(6):
        val = maxv * i / 5
        y = scale(val, 0, maxv, y0, y1)
        draw.line((x0 - 8, y, x0, y), fill=BLACK, width=2)
        draw_right(draw, (x0 - 14, y), f"{val:.2f}", f_tick)
        draw.line((x0, y, x1, y), fill=(232, 232, 232), width=1)

    draw_center(draw, ((x0 + x1) / 2, h - 55), "方案", f_label)
    draw_rotated_label(img, (55, (y0 + y1) // 2), "延迟（ms）", f_label)

    bar_w = 120
    gap = 145
    start_x = x0 + 120
    for i, row in enumerate(rows):
        x = start_x + i * (bar_w + gap)
        y = y0
        total = 0.0
        for key, _, color in parts:
            val = float(row[key])
            total += val
            bh = scale(val, 0, maxv, 0, y0 - y1)
            draw.rectangle((x, y - bh, x + bar_w, y), fill=color, outline=BLACK, width=1)
            y -= bh
        draw.rectangle((x, y, x + bar_w, y0), outline=BLACK, width=2)
        draw_center(draw, (x + bar_w / 2, y0 + 35), row["scheme"], f_tick)
        draw_center(draw, (x + bar_w / 2, y - 20), f"{total:.2f}", f_tick)

    lx, ly = 1000, 205
    for idx, (_, label, color) in enumerate(parts):
        yy = ly + idx * 48
        draw.rectangle((lx, yy, lx + 34, yy + 24), fill=color, outline=BLACK)
        draw.text((lx + 48, yy - 5), label, font=f_small, fill=BLACK)
    draw.text((95, 805), "结论边界：smoke 图仅说明流水框架可测；未达到正式论文级性能结论。", font=f_small, fill=(70, 70, 70))

    save_outputs(
        img,
        output_dir / "latency_breakdown_81pct_smoke.png",
        output_dir / "latency_breakdown_81pct_smoke.pdf",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hit-rate-csv", type=Path, required=True)
    parser.add_argument("--qd-chunk-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    args = parser.parse_args()

    if not args.font.exists():
        raise FileNotFoundError(f"Chinese font not found: {args.font}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_hit_rate(args.hit_rate_csv, args.output_dir, args.font)
    plot_heatmap(args.qd_chunk_csv, args.output_dir, args.font)
    plot_breakdown(args.qd_chunk_csv, args.output_dir, args.font)

    manifest = {
        "status": "ok",
        "type": "chinese_smoke_plots",
        "font": str(args.font),
        "note": "Chinese labels are rendered into PNG/PDF with Pillow. These are smoke plots, not formal paper-level results.",
        "outputs": [
            "barrier_vs_progressive_hit_rate_smoke.png",
            "barrier_vs_progressive_hit_rate_smoke.pdf",
            "speedup_heatmap_81pct_smoke.png",
            "speedup_heatmap_81pct_smoke.pdf",
            "latency_breakdown_81pct_smoke.png",
            "latency_breakdown_81pct_smoke.pdf",
        ],
    }
    (args.output_dir / "smoke_plot_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
