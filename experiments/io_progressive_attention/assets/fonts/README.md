中文绘图需要 CJK 字体。当前重绘使用：

- NotoSansCJKsc-Regular.otf
- 来源：https://github.com/googlefonts/noto-cjk
- 许可证：SIL Open Font License 1.1

字体文件未随仓库提交。重新生成图片时，请下载该字体或指定任意支持简体中文的字体：

```bash
python3 experiments/io_progressive_attention/scripts/plot_smoke_chinese.py \
  --hit-rate-csv experiments/io_progressive_attention/results/hit_rate_latency_smoke_v3_paired.csv \
  --qd-chunk-csv experiments/io_progressive_attention/results/qd_chunk_grid_smoke_v2.csv \
  --font /path/to/NotoSansCJKsc-Regular.otf \
  --output-dir experiments/io_progressive_attention/results/plots_smoke_v3
```
