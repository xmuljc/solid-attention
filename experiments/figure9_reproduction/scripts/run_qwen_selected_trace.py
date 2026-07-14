#!/usr/bin/env python3
"""Generate real Qwen selected-block traces for Figure 9 style analysis.

This runner does not implement SSD I/O, prefetching, or attention replacement.
It runs greedy decode with Hugging Face Qwen2.5, captures real q_proj/k_proj
activations, builds per-block representative key vectors, and records dynamic
Top-K selected blocks using the same cosine-ranking semantics as
core.block_selector.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


LONG_BENCH_DATASETS = [
    "gov_report",
    "multifieldqa_en",
    "musique",
    "narrativeqa",
    "qmsum",
    "multi_news",
    "repobench-p",
]
RULER_DATASET = "ruler_multikey_alternating"


@dataclass
class Sample:
    dataset: str
    sample_id: str
    prompt: str
    source_id: str | None


class ProjectionCapture:
    def __init__(self, model: Any) -> None:
        self.q: dict[int, torch.Tensor] = {}
        self.k: dict[int, torch.Tensor] = {}
        self._handles = []
        for layer_idx, layer in enumerate(model.model.layers):
            self._handles.append(layer.self_attn.q_proj.register_forward_hook(self._hook(self.q, layer_idx)))
            self._handles.append(layer.self_attn.k_proj.register_forward_hook(self._hook(self.k, layer_idx)))

    @staticmethod
    def _hook(store: dict[int, torch.Tensor], layer_idx: int):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            store[layer_idx] = output.detach()

        return hook

    def clear(self) -> None:
        self.q.clear()
        self.k.clear()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    left = x[..., : x.shape[-1] // 2]
    right = x[..., x.shape[-1] // 2 :]
    return torch.cat((-right, left), dim=-1)


def apply_rope(
    model: Any,
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    positions: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = int(q_raw.shape[1])
    q = q_raw[0].view(seq_len, num_heads, head_dim)
    k = k_raw[0].view(seq_len, num_kv_heads, head_dim)
    dummy = torch.empty((1, seq_len, 1), device=q.device, dtype=q.dtype)
    cos, sin = model.model.rotary_emb(dummy, positions.unsqueeze(0))
    cos = cos[0].unsqueeze(1)
    sin = sin[0].unsqueeze(1)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q.float(), k.float()


def parse_dataset_list(value: str) -> list[str]:
    if value == "all":
        return LONG_BENCH_DATASETS + [RULER_DATASET]
    return [item.strip() for item in value.split(",") if item.strip()]


def format_longbench_prompt(dataset: str, row: dict[str, Any]) -> str:
    context = str(row.get("context", "")).strip()
    question = str(row.get("input", "")).strip()
    if dataset in {"gov_report", "multi_news"}:
        return f"{context}\n\nSummarize the above text."
    if dataset == "qmsum":
        return f"{context}\n\n{question}\nSummary:"
    if dataset == "repobench-p":
        return f"{context}\n\nComplete the following code:\n{question}"
    return f"{context}\n\nQuestion: {question}\nAnswer:"


def load_samples(
    datasets: list[str],
    *,
    samples_per_dataset: int,
    sample_offset: int,
    seed: int,
    ruler_path: Path,
) -> list[Sample]:
    random.seed(seed)
    samples: list[Sample] = []
    for dataset in datasets:
        if dataset == RULER_DATASET:
            with ruler_path.open("r", encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            for idx, row in enumerate(rows[sample_offset : sample_offset + samples_per_dataset], start=sample_offset):
                samples.append(
                    Sample(
                        dataset=dataset,
                        sample_id=f"{dataset}/{idx}",
                        prompt=str(row["prompt"]),
                        source_id=None,
                    )
                )
            continue

        loaded = load_dataset("THUDM/LongBench", dataset, split="test", trust_remote_code=True)
        for idx in range(sample_offset, min(sample_offset + samples_per_dataset, len(loaded))):
            row = loaded[idx]
            samples.append(
                Sample(
                    dataset=dataset,
                    sample_id=f"{dataset}/{idx}",
                    prompt=format_longbench_prompt(dataset, row),
                    source_id=str(row.get("_id", idx)),
                )
            )
    return samples


def encode_prompt(tokenizer: Any, prompt: str, *, context_budget: int) -> list[int]:
    ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    if len(ids) <= context_budget:
        return ids
    head = context_budget // 2
    tail = context_budget - head
    return ids[:head] + ids[-tail:]


def update_block_sums(
    block_sums: list[torch.Tensor],
    block_counts: list[int],
    keys: torch.Tensor,
    start_position: int,
    block_size: int,
) -> None:
    for offset in range(keys.shape[0]):
        position = start_position + offset
        block_id = position // block_size
        while len(block_sums) <= block_id:
            block_sums.append(torch.zeros_like(keys[0]))
            block_counts.append(0)
        block_sums[block_id] += keys[offset]
        block_counts[block_id] += 1


def selected_for_head(
    query: torch.Tensor,
    reps: torch.Tensor,
    counts: torch.Tensor,
    *,
    head_id: int,
    num_key_value_groups: int,
    top_k: int,
    init_block_count: int,
    local_block_count: int,
) -> tuple[list[int], list[int], list[int]]:
    block_count = int(reps.shape[0])
    if block_count == 0:
        return [], [], []

    init_blocks = list(range(min(init_block_count, block_count)))
    init_set = set(init_blocks)
    local_start = max(0, block_count - local_block_count)
    local_blocks = [idx for idx in range(local_start, block_count) if idx not in init_set]
    excluded = init_set | set(local_blocks)
    candidate_ids = [idx for idx in range(block_count) if idx not in excluded]
    if not candidate_ids or top_k <= 0:
        return [], init_blocks, local_blocks

    kv_head = head_id // num_key_value_groups
    candidate_tensor = torch.tensor(candidate_ids, device=reps.device, dtype=torch.long)
    means = reps[candidate_tensor, kv_head, :] / counts[candidate_tensor].view(-1, 1).clamp_min(1.0)
    q_norm = torch.linalg.vector_norm(query).clamp_min(1e-12)
    rep_norm = torch.linalg.vector_norm(means, dim=1).clamp_min(1e-12)
    scores = torch.mv(means, query) / (rep_norm * q_norm)
    take = min(top_k, len(candidate_ids))
    _, order = torch.topk(scores, k=take, largest=True, sorted=True)
    selected = [candidate_ids[int(pos)] for pos in order]
    return selected, init_blocks, local_blocks


def write_selection_events(
    fh: Any,
    *,
    model_name: str,
    sample: Sample,
    decode_step: int,
    query_by_layer: dict[int, torch.Tensor],
    block_sums_by_layer: dict[int, list[torch.Tensor]],
    block_counts_by_layer: dict[int, list[int]],
    recorded_layers: list[int],
    recorded_heads: list[int],
    top_k: int,
    init_block_count: int,
    local_block_count: int,
    num_key_value_groups: int,
    prompt_tokens: int,
    generated_tokens_seen: int,
) -> int:
    event_count = 0
    for layer_id in recorded_layers:
        block_sums = block_sums_by_layer[layer_id]
        if not block_sums:
            continue
        reps = torch.stack(block_sums, dim=0)
        counts = torch.tensor(block_counts_by_layer[layer_id], device=reps.device, dtype=torch.float32)
        query = query_by_layer[layer_id]
        for head_id in recorded_heads:
            selected, init_blocks, local_blocks = selected_for_head(
                query[head_id],
                reps,
                counts,
                head_id=head_id,
                num_key_value_groups=num_key_value_groups,
                top_k=top_k,
                init_block_count=init_block_count,
                local_block_count=local_block_count,
            )
            fh.write(
                json.dumps(
                    {
                        "model": model_name,
                        "dataset": sample.dataset,
                        "sample_id": sample.sample_id,
                        "source_id": sample.source_id,
                        "head_id": head_id,
                        "layer_id": layer_id,
                        "decode_step": decode_step,
                        "selected_blocks": selected,
                        "init_blocks": init_blocks,
                        "local_blocks": local_blocks,
                        "prompt_tokens": prompt_tokens,
                        "generated_tokens_seen": generated_tokens_seen,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            event_count += 1
    return event_count


def run_sample(
    *,
    model: Any,
    tokenizer: Any,
    capture: ProjectionCapture,
    sample: Sample,
    output_fh: Any,
    args: argparse.Namespace,
    recorded_layers: list[int],
    recorded_heads: list[int],
) -> dict[str, Any]:
    ids = encode_prompt(tokenizer, sample.prompt, context_budget=args.context_budget)
    input_ids = torch.tensor([ids], device=model.device, dtype=torch.long)
    prompt_tokens = int(input_ids.shape[1])
    cfg = model.config
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
    num_heads = int(cfg.num_attention_heads)
    num_kv_heads = int(cfg.num_key_value_heads)
    num_key_value_groups = num_heads // num_kv_heads
    block_sums_by_layer = {layer_id: [] for layer_id in recorded_layers}
    block_counts_by_layer = {layer_id: [] for layer_id in recorded_layers}

    capture.clear()
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True)
    past_key_values = out.past_key_values
    next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    positions = torch.arange(prompt_tokens, device=model.device, dtype=torch.long)
    query_by_layer: dict[int, torch.Tensor] = {}
    for layer_id in recorded_layers:
        q, k = apply_rope(
            model,
            capture.q[layer_id],
            capture.k[layer_id],
            positions,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        update_block_sums(block_sums_by_layer[layer_id], block_counts_by_layer[layer_id], k, 0, args.block_size)
        query_by_layer[layer_id] = q[-1]

    events = write_selection_events(
        output_fh,
        model_name=args.model_name,
        sample=sample,
        decode_step=0,
        query_by_layer=query_by_layer,
        block_sums_by_layer=block_sums_by_layer,
        block_counts_by_layer=block_counts_by_layer,
        recorded_layers=recorded_layers,
        recorded_heads=recorded_heads,
        top_k=args.top_k,
        init_block_count=args.init_block_count,
        local_block_count=args.local_block_count,
        num_key_value_groups=num_key_value_groups,
        prompt_tokens=prompt_tokens,
        generated_tokens_seen=0,
    )

    generated = [int(next_token.item())]
    current_position = prompt_tokens
    eos_id = tokenizer.eos_token_id
    stop = bool(args.stop_on_eos and eos_id is not None and generated[-1] == int(eos_id))

    for decode_step in range(1, args.max_new_tokens):
        if stop:
            break
        token_in = next_token
        capture.clear()
        with torch.inference_mode():
            out = model(input_ids=token_in, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

        positions = torch.tensor([current_position], device=model.device, dtype=torch.long)
        query_by_layer = {}
        for layer_id in recorded_layers:
            q, k = apply_rope(
                model,
                capture.q[layer_id],
                capture.k[layer_id],
                positions,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )
            update_block_sums(
                block_sums_by_layer[layer_id],
                block_counts_by_layer[layer_id],
                k,
                current_position,
                args.block_size,
            )
            query_by_layer[layer_id] = q[-1]

        current_position += 1
        events += write_selection_events(
            output_fh,
            model_name=args.model_name,
            sample=sample,
            decode_step=decode_step,
            query_by_layer=query_by_layer,
            block_sums_by_layer=block_sums_by_layer,
            block_counts_by_layer=block_counts_by_layer,
            recorded_layers=recorded_layers,
            recorded_heads=recorded_heads,
            top_k=args.top_k,
            init_block_count=args.init_block_count,
            local_block_count=args.local_block_count,
            num_key_value_groups=num_key_value_groups,
            prompt_tokens=prompt_tokens,
            generated_tokens_seen=decode_step,
        )

        generated.append(int(next_token.item()))
        stop = bool(args.stop_on_eos and eos_id is not None and generated[-1] == int(eos_id))

    return {
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": len(generated),
        "trace_events": events,
        "stopped_on_eos": stop,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path(".deps/models/Qwen2.5-7B"))
    parser.add_argument("--model-name", default="Qwen-2.5-7B")
    parser.add_argument("--datasets", default="all", help="comma-separated list or 'all'")
    parser.add_argument("--samples-per-dataset", type=int, default=3)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--context-budget", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--init-block-count", type=int, default=1)
    parser.add_argument("--local-block-count", type=int, default=1)
    parser.add_argument("--max-layers", type=int, default=0, help="0 means all layers")
    parser.add_argument("--max-heads", type=int, default=0, help="0 means all query heads")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ruler-path", type=Path, default=Path("experiments/figure9_reproduction/ruler_multikey_alternating.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("experiments/figure9_reproduction/results/selected_blocks_trace.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("experiments/figure9_reproduction/results/qwen_selected_blocks_run_manifest.json"))
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--stop-on-eos", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    datasets = parse_dataset_list(args.datasets)
    samples = load_samples(
        datasets,
        samples_per_dataset=args.samples_per_dataset,
        sample_offset=args.sample_offset,
        seed=args.seed,
        ruler_path=args.ruler_path,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        dtype=torch.float16,
        device_map={"": args.device},
        low_cpu_mem_usage=True,
    )
    model.eval()

    num_layers = int(model.config.num_hidden_layers)
    num_heads = int(model.config.num_attention_heads)
    recorded_layers = list(range(num_layers if args.max_layers == 0 else min(args.max_layers, num_layers)))
    recorded_heads = list(range(num_heads if args.max_heads == 0 else min(args.max_heads, num_heads)))
    capture = ProjectionCapture(model)

    started = time.time()
    mode = "a" if args.append else "w"
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with args.output.open(mode, encoding="utf-8") as fh:
        for idx, sample in enumerate(samples, start=1):
            sample_start = time.time()
            try:
                result = run_sample(
                    model=model,
                    tokenizer=tokenizer,
                    capture=capture,
                    sample=sample,
                    output_fh=fh,
                    args=args,
                    recorded_layers=recorded_layers,
                    recorded_heads=recorded_heads,
                )
                result["elapsed_sec"] = round(time.time() - sample_start, 3)
                results.append(result)
                print(json.dumps({"status": "sample_ok", "index": idx, "total": len(samples), **result}, sort_keys=True), flush=True)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                failure = {
                    "sample_id": sample.sample_id,
                    "dataset": sample.dataset,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failures.append(failure)
                print(json.dumps({"status": "sample_failed", **failure}, sort_keys=True), flush=True)
    capture.close()

    manifest = {
        "contract": "solidattention.figure9.qwen_selected_trace.v1",
        "status": "ok" if not failures else "partial",
        "model": args.model_name,
        "model_dir": str(args.model_dir),
        "load_dtype": "float16",
        "kv_trace_dtype": "float16_projection_to_float32_representative",
        "datasets": datasets,
        "samples_per_dataset": args.samples_per_dataset,
        "sample_offset": args.sample_offset,
        "max_new_tokens": args.max_new_tokens,
        "context_budget": args.context_budget,
        "block_size": args.block_size,
        "top_k": args.top_k,
        "init_block_count": args.init_block_count,
        "local_block_count": args.local_block_count,
        "recorded_layers": recorded_layers,
        "recorded_heads": recorded_heads,
        "trace_output": str(args.output),
        "sample_results": results,
        "failures": failures,
        "elapsed_sec": round(time.time() - started, 3),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "samples": len(results), "failures": len(failures), "manifest": str(args.manifest)}, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
