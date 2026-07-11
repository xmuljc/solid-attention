"""Python wrapper for the runtime KV command consumer prototype."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.process import build_executable_command


@dataclass(frozen=True)
class RuntimeCommandConsumerResult:
    backend: str
    command_count: int
    kv_load_count: int
    kv_prefetch_count: int
    kv_compute_count: int
    kv_evict_count: int
    ssd_read_count: int
    ssd_write_count: int
    dram_to_vram_copy_count: int
    vram_compute_count: int
    lazy_init_count: int
    ssd_seed_count: int
    ssd_read_bytes: int
    ssd_write_bytes: int
    dram_to_vram_bytes: int
    lazy_init_bytes: int
    ssd_seed_bytes: int
    verification_errors: int
    ssd_read_latency_ns: int
    ssd_write_latency_ns: int
    copy_latency_ns: int
    compute_verify_latency_ns: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RuntimeCommandConsumerResult":
        return cls(
            backend=str(payload["backend"]),
            command_count=int(payload["command_count"]),
            kv_load_count=int(payload["kv_load_count"]),
            kv_prefetch_count=int(payload["kv_prefetch_count"]),
            kv_compute_count=int(payload["kv_compute_count"]),
            kv_evict_count=int(payload["kv_evict_count"]),
            ssd_read_count=int(payload["ssd_read_count"]),
            ssd_write_count=int(payload["ssd_write_count"]),
            dram_to_vram_copy_count=int(payload["dram_to_vram_copy_count"]),
            vram_compute_count=int(payload["vram_compute_count"]),
            lazy_init_count=int(payload["lazy_init_count"]),
            ssd_seed_count=int(payload["ssd_seed_count"]),
            ssd_read_bytes=int(payload["ssd_read_bytes"]),
            ssd_write_bytes=int(payload["ssd_write_bytes"]),
            dram_to_vram_bytes=int(payload["dram_to_vram_bytes"]),
            lazy_init_bytes=int(payload["lazy_init_bytes"]),
            ssd_seed_bytes=int(payload["ssd_seed_bytes"]),
            verification_errors=int(payload["verification_errors"]),
            ssd_read_latency_ns=int(payload["ssd_read_latency_ns"]),
            ssd_write_latency_ns=int(payload["ssd_write_latency_ns"]),
            copy_latency_ns=int(payload["copy_latency_ns"]),
            compute_verify_latency_ns=int(payload["compute_verify_latency_ns"]),
        )

    @property
    def ssd_read_latency_ms(self) -> float:
        return self.ssd_read_latency_ns / 1_000_000.0

    @property
    def ssd_write_latency_ms(self) -> float:
        return self.ssd_write_latency_ns / 1_000_000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "command_count": self.command_count,
            "kv_load_count": self.kv_load_count,
            "kv_prefetch_count": self.kv_prefetch_count,
            "kv_compute_count": self.kv_compute_count,
            "kv_evict_count": self.kv_evict_count,
            "ssd_read_count": self.ssd_read_count,
            "ssd_write_count": self.ssd_write_count,
            "dram_to_vram_copy_count": self.dram_to_vram_copy_count,
            "vram_compute_count": self.vram_compute_count,
            "lazy_init_count": self.lazy_init_count,
            "ssd_seed_count": self.ssd_seed_count,
            "ssd_read_bytes": self.ssd_read_bytes,
            "ssd_write_bytes": self.ssd_write_bytes,
            "dram_to_vram_bytes": self.dram_to_vram_bytes,
            "lazy_init_bytes": self.lazy_init_bytes,
            "ssd_seed_bytes": self.ssd_seed_bytes,
            "verification_errors": self.verification_errors,
            "ssd_read_latency_ns": self.ssd_read_latency_ns,
            "ssd_write_latency_ns": self.ssd_write_latency_ns,
            "copy_latency_ns": self.copy_latency_ns,
            "compute_verify_latency_ns": self.compute_verify_latency_ns,
            "ssd_read_latency_ms": self.ssd_read_latency_ms,
            "ssd_write_latency_ms": self.ssd_write_latency_ms,
        }


def run_runtime_command_consumer(
    executable: str | Path,
    command_jsonl: str | Path,
    backing_file: str | Path,
) -> RuntimeCommandConsumerResult:
    executable_path = Path(executable)
    if not executable_path.exists():
        raise FileNotFoundError(f"runtime command consumer executable not found: {executable_path}")
    command_path = Path(command_jsonl)
    if not command_path.exists():
        raise FileNotFoundError(f"runtime command JSONL not found: {command_path}")
    backing_path = Path(backing_file)
    backing_path.parent.mkdir(parents=True, exist_ok=True)

    completed = subprocess.run(
        build_executable_command(executable_path, command_path, backing_path),
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"runtime command consumer did not emit JSON: {completed.stdout!r}") from exc
    if not isinstance(payload, dict):
        raise ValueError("runtime command consumer JSON must be an object")
    return RuntimeCommandConsumerResult.from_dict(payload)
