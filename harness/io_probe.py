"""Python wrapper for the Phase 2 block I/O probe."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.process import build_executable_command


@dataclass(frozen=True)
class BlockIOProbeResult:
    backend: str
    path: str
    block_size: int
    block_count: int
    bytes: int
    write_latency_ns: int
    read_latency_ns: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BlockIOProbeResult":
        return cls(
            backend=str(payload["backend"]),
            path=str(payload["path"]),
            block_size=int(payload["block_size"]),
            block_count=int(payload["block_count"]),
            bytes=int(payload["bytes"]),
            write_latency_ns=int(payload["write_latency_ns"]),
            read_latency_ns=int(payload["read_latency_ns"]),
        )

    @property
    def read_latency_ms(self) -> float:
        return self.read_latency_ns / 1_000_000.0

    @property
    def write_latency_ms(self) -> float:
        return self.write_latency_ns / 1_000_000.0

    @property
    def read_latency_ms_per_block(self) -> float:
        if self.block_count <= 0:
            return 0.0
        return self.read_latency_ms / self.block_count

    @property
    def write_latency_ms_per_block(self) -> float:
        if self.block_count <= 0:
            return 0.0
        return self.write_latency_ms / self.block_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "path": self.path,
            "block_size": self.block_size,
            "block_count": self.block_count,
            "bytes": self.bytes,
            "write_latency_ns": self.write_latency_ns,
            "read_latency_ns": self.read_latency_ns,
            "write_latency_ms": self.write_latency_ms,
            "read_latency_ms": self.read_latency_ms,
            "write_latency_ms_per_block": self.write_latency_ms_per_block,
            "read_latency_ms_per_block": self.read_latency_ms_per_block,
        }


def run_block_io_probe(
    executable: str | Path,
    probe_path: str | Path,
    *,
    block_size: int,
    block_count: int,
) -> BlockIOProbeResult:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if block_count <= 0:
        raise ValueError("block_count must be positive")

    executable_path = Path(executable)
    if not executable_path.exists():
        raise FileNotFoundError(f"block I/O probe executable not found: {executable_path}")

    output_path = Path(probe_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = subprocess.run(
        build_executable_command(executable_path, output_path, block_size, block_count),
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"block I/O probe did not emit JSON: {completed.stdout!r}") from exc
    if not isinstance(payload, dict):
        raise ValueError("block I/O probe JSON must be an object")
    return BlockIOProbeResult.from_dict(payload)
