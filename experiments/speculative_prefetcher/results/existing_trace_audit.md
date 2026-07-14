# Existing Trace And Environment Audit

- Model: `Qwen-2.5-7B`
- Qwen config: `.deps/models/Qwen2.5-7B/config.json`
- num layers: `28`
- num query heads: `28`
- num KV heads: `4`
- head dim: `128`
- KV block bytes: `65536`
- Trace path: `/data/disk2/ljc/solid attention/experiments/figure9_reproduction/results/selected_blocks_trace.jsonl`
- Events: `602112`
- Workloads: `gov_report, multifieldqa_en, musique, narrativeqa, qmsum, multi_news, repobench-p, ruler_multikey_alternating`
- Prompt counts: `{'gov_report': 3, 'multifieldqa_en': 3, 'musique': 3, 'narrativeqa': 3, 'qmsum': 3, 'multi_news': 3, 'repobench-p': 3, 'ruler_multikey_alternating': 3}`
- Decode steps: `0..31`
- Layers: `0..27`
- Query heads / selection units: `0..27`
- Events where selected blocks contain init/local: `528320`
- KV test file: `/data/disk2/ljc/solid attention/experiments/io_progressive_attention/data/qwen_kv_finite_2g.bin`
- KV file size: `2147483648` bytes; allocated `2147487744` bytes

Important limitation: the available real trace covers the previous smoke workloads `gov_report`, `multifieldqa_en`, `musique`, `narrativeqa`, `qmsum`, `multi_news`, `repobench-p`, and `ruler_multikey_alternating`. It does not cover the newly listed `Qasper`, `2WikiMQA`, `TriviaQA`, and `HotpotQA` workloads. Those cannot be claimed without collecting new selected-block traces.

The test file is on `/data/disk2`, which `findmnt` reports as `/dev/nvme1n1` ext4.
