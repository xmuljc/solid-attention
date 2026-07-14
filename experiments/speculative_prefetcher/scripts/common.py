#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def load_qwen_shape(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = json.loads(resolve(config["model"]["config_path"]).read_text(encoding="utf-8"))
    num_heads = int(model_cfg["num_attention_heads"])
    hidden_size = int(model_cfg["hidden_size"])
    head_dim = int(model_cfg.get("head_dim") or hidden_size // num_heads)
    num_kv_heads = int(model_cfg["num_key_value_heads"])
    block_tokens = int(config["selection"]["block_tokens"])
    dtype_bytes = int(config["model"]["dtype_bytes"])
    block_bytes = block_tokens * num_kv_heads * head_dim * 2 * dtype_bytes
    return {
        "model_type": model_cfg.get("model_type"),
        "num_hidden_layers": int(model_cfg["num_hidden_layers"]),
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "torch_dtype": model_cfg.get("torch_dtype"),
        "load_kv_dtype": config["model"]["kv_dtype"],
        "block_tokens": block_tokens,
        "dtype_bytes": dtype_bytes,
        "block_bytes": block_bytes,
    }


def run_text(cmd: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return {"cmd": cmd, "returncode": proc.returncode, "output": proc.stdout}
    except FileNotFoundError as exc:
        return {"cmd": cmd, "returncode": 127, "output": str(exc), "status": "unavailable"}


def stable_int(parts: Iterable[Any]) -> int:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return int.from_bytes(h.digest()[:8], "big")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def bootstrap_ci(values: list[float], seed: int, rounds: int = 2000) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(rounds):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    return percentile(means, 0.025), percentile(means, 0.975)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def group_rows(rows: Iterable[dict[str, Any]], key_fields: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    out: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[tuple(row[k] for k in key_fields)].append(row)
    return out
