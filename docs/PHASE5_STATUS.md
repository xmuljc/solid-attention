# Phase 5 Verification Status

Phase 5 provides simulation ablation experiments with JSON trace and metrics for
every method. These are still harness-level experiments: no real model, no real
attention, and no runtime llama.cpp KV movement is performed here.

## Implemented Methods

- `memory_only`: all selected blocks are already resident in VRAM.
- `ssd_sync`: selected blocks start on SSD and are synchronously scheduled into VRAM.
- `ssd_async`: required blocks are accurately preloaded SSD->DRAM before the blocking VRAM schedule, reducing simulated GPU wait.
- `prefetch_only`: only Selected Blocks are speculatively prefetched before the scheduler handles remaining required blocks.
- `full_scheduler`: representative selection, speculative prefetch, prefetch hit/miss evaluation, and SSD-aware scheduling.
- `solid_sim`: compatibility alias for `full_scheduler`.

## Outputs

Each method writes generated artifacts under an ignored output directory:

- `<method>_trace.json`
- `<method>_metrics.json`

The harness also writes:

- `ablation_summary.json`

Example output directory:

- `outputs/ablation/`

## Historical Metric Shape

The imported snapshot's smoke ablation produced the expected qualitative
differences. These values are historical evidence; generated files are no
longer tracked:

- `memory_only`: `ssd_read_bytes = 0`
- `ssd_sync`: blocking SSD reads and highest simulated GPU wait
- `ssd_async`: SSD reads with lower simulated GPU wait than `ssd_sync`
- `prefetch_only`: prefetch attempts with hit/miss counts
- `full_scheduler`: prefetch attempts plus SSD-aware scheduling

## Verified Commands

```bash
python -m tools.check fast
```

Historical observed result:

```text
2 passed in 0.04s
methods: memory_only, ssd_sync, ssd_async, prefetch_only, full_scheduler, solid_sim
```

Real-model work begins only after the harness baseline and a future
load-before-attention vertical slice are complete. It then requires runtime KV
events from an actual model invocation and scheduler-owned SSD/DRAM/VRAM block
movement.
