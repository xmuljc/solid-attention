#!/usr/bin/env python3
"""Create a deterministic RULER-style alternating multi-key NIAH prompt.

This is a fixed generator specification for the controlled pressure task. It
is not tuned against results.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def make_sample(seed: int, context_length: int, num_needle_k: int, num_needle_q: int) -> dict[str, object]:
    rng = random.Random(seed)
    keys = [f"key_{i}" for i in range(1, num_needle_k + 1)]
    values = [f"value_{rng.randrange(10**8):08d}" for _ in keys]
    order = []
    lo, hi = 0, num_needle_q - 1
    while lo <= hi:
        order.append(lo)
        if lo != hi:
            order.append(hi)
        lo += 1
        hi -= 1
    order = order[:num_needle_q]
    filler = "The following context contains independent key value facts. "
    slots = [f"Needle fact: {k} => {v}." for k, v in zip(keys, values)]
    segment = max(1, context_length // max(1, len(slots)))
    parts = []
    for i, needle in enumerate(slots):
        parts.append((filler * max(1, segment // len(filler)))[:segment])
        parts.append(needle)
    context = "\n".join(parts)[:context_length]
    query_keys = [keys[i] for i in order]
    expected = [values[i] for i in order]
    prompt = context + "\n\nReturn the values for these keys in this exact order: " + ", ".join(query_keys)
    return {"seed": seed, "context_length": context_length, "num_needle_k": num_needle_k, "num_needle_q": num_needle_q, "query_order_1_indexed": [i + 1 for i in order], "prompt": prompt, "expected_values": expected}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("experiments/figure9_reproduction/ruler_multikey_alternating.jsonl"))
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--context-length", type=int, default=16000)
    parser.add_argument("--num-needle-k", type=int, default=8)
    parser.add_argument("--num-needle-q", type=int, default=8)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for i in range(args.samples):
            fh.write(json.dumps(make_sample(args.seed + i, args.context_length, args.num_needle_k, args.num_needle_q), ensure_ascii=False) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
