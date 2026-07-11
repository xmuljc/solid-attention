"""Export runtime KV movement plans as llama.cpp command JSONL."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


MOVEMENT_COMMAND_OPS = {"kv_load", "kv_prefetch", "kv_evict", "kv_compute"}
STORAGE_LOCATIONS = {"ssd", "dram", "vram"}


@dataclass(frozen=True)
class LlamaKVCommand:
    """C++-friendly command for future llama.cpp KV movement integration."""

    op: str
    layer_id: int
    block_id: int
    start_token: int
    end_token: int
    size_bytes: int
    source: str
    target: str
    ssd_offset: int
    blocking: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.op not in MOVEMENT_COMMAND_OPS:
            raise ValueError(f"unsupported command op: {self.op}")
        if self.layer_id < 0:
            raise ValueError("layer_id must be non-negative")
        if self.block_id < 0:
            raise ValueError("block_id must be non-negative")
        if self.start_token < 0:
            raise ValueError("start_token must be non-negative")
        if self.end_token <= self.start_token:
            raise ValueError("end_token must be greater than start_token")
        if self.size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        if self.source not in STORAGE_LOCATIONS:
            raise ValueError(f"unsupported source location: {self.source}")
        if self.target not in STORAGE_LOCATIONS:
            raise ValueError(f"unsupported target location: {self.target}")
        if self.ssd_offset < 0:
            raise ValueError("ssd_offset must be non-negative")
        if self.op == "kv_compute" and (self.source != "vram" or self.target != "vram"):
            raise ValueError("kv_compute commands must run on vram")
        if self.op == "kv_evict" and self.target != "ssd":
            raise ValueError("kv_evict commands must target ssd")
        if self.op in {"kv_load", "kv_prefetch"} and self.source == self.target:
            raise ValueError(f"{self.op} commands must move between different locations")

    @property
    def location(self) -> str:
        return self.target

    @property
    def token_count(self) -> int:
        return self.end_token - self.start_token

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "layer_id": self.layer_id,
            "block_id": self.block_id,
            "start_token": self.start_token,
            "end_token": self.end_token,
            "token_count": self.token_count,
            "size_bytes": self.size_bytes,
            "location": self.location,
            "source": self.source,
            "target": self.target,
            "ssd_offset": self.ssd_offset,
            "blocking": self.blocking,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RuntimeCommandExport:
    plan_path: str
    command_count: int
    command_counts_by_op: dict[str, int]
    ssd_read_command_count: int
    ssd_write_command_count: int
    ssd_read_bytes: int
    ssd_write_bytes: int
    jsonl_path: str | None = None
    summary_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_path": self.plan_path,
            "command_count": self.command_count,
            "command_counts_by_op": dict(self.command_counts_by_op),
            "ssd_read_command_count": self.ssd_read_command_count,
            "ssd_write_command_count": self.ssd_write_command_count,
            "ssd_read_bytes": self.ssd_read_bytes,
            "ssd_write_bytes": self.ssd_write_bytes,
            "jsonl_path": self.jsonl_path,
            "summary_path": self.summary_path,
        }


def commands_from_movement_plan(plan: dict[str, Any]) -> list[LlamaKVCommand]:
    """Convert a movement-plan payload to validated runtime commands."""

    _validate_layout(plan)
    layout_ranges = _layout_ranges(plan)
    ops = plan.get("ops")
    if not isinstance(ops, list):
        raise ValueError("movement plan ops must be a list")

    commands: list[LlamaKVCommand] = []
    for index, op in enumerate(ops):
        if not isinstance(op, dict):
            raise ValueError(f"movement op {index} must be a JSON object")
        command = _command_from_op(op)
        block_range = layout_ranges.get((command.layer_id, command.block_id))
        if block_range is None:
            raise ValueError(f"command references unknown layout block: {(command.layer_id, command.block_id)}")
        if command.ssd_offset != int(block_range["offset"]):
            raise ValueError(f"command SSD offset does not match layout for {(command.layer_id, command.block_id)}")
        if command.size_bytes != int(block_range["size_bytes"]):
            raise ValueError(f"command size does not match layout for {(command.layer_id, command.block_id)}")
        commands.append(command)
    return commands


def export_runtime_commands(
    plan_path: str | Path,
    *,
    jsonl_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> RuntimeCommandExport:
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("movement plan must be a JSON object")
    commands = commands_from_movement_plan(plan)

    if jsonl_path is not None:
        output = Path(jsonl_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(json.dumps(command.to_dict(), sort_keys=True) + "\n" for command in commands), encoding="utf-8")

    result = _summarize_commands(plan_path, commands, jsonl_path=jsonl_path, summary_path=summary_path)
    if summary_path is not None:
        output = Path(summary_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _command_from_op(op: dict[str, Any]) -> LlamaKVCommand:
    missing = [field for field in ("op", "layer_id", "block_id", "start_token", "end_token", "size_bytes", "source", "target", "ssd_offset") if field not in op]
    if missing:
        raise ValueError(f"movement op missing runtime command fields: {missing}")
    metadata = dict(op.get("metadata", {}))
    metadata.update(
        {
            "runtime_contract": "solidattention.llama_cpp.kv_command.v1",
            "movement_start_ms": op.get("start_ms"),
            "movement_end_ms": op.get("end_ms"),
            "movement_duration_ms": op.get("duration_ms"),
        }
    )
    return LlamaKVCommand(
        op=str(op["op"]),
        layer_id=int(op["layer_id"]),
        block_id=int(op["block_id"]),
        start_token=int(op["start_token"]),
        end_token=int(op["end_token"]),
        size_bytes=int(op["size_bytes"]),
        source=str(op["source"]),
        target=str(op["target"]),
        ssd_offset=int(op["ssd_offset"]),
        blocking=bool(op.get("blocking", False)),
        metadata=metadata,
    )


def _validate_layout(plan: dict[str, Any]) -> None:
    layout = plan.get("layout")
    if not isinstance(layout, dict):
        raise ValueError("movement plan missing layout")
    blocks = layout.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError("movement plan layout blocks must be a list")


def _layout_ranges(plan: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    blocks = plan["layout"]["blocks"]
    ranges: dict[tuple[int, int], dict[str, Any]] = {}
    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            raise ValueError(f"layout block {index} must be a JSON object")
        key = (int(block["layer_id"]), int(block["block_id"]))
        if key in ranges:
            raise ValueError(f"duplicate layout block: {key}")
        ranges[key] = block
    return ranges


def _summarize_commands(
    plan_path: Path,
    commands: Iterable[LlamaKVCommand],
    *,
    jsonl_path: str | Path | None,
    summary_path: str | Path | None,
) -> RuntimeCommandExport:
    commands = list(commands)
    counts: dict[str, int] = {op: 0 for op in sorted(MOVEMENT_COMMAND_OPS)}
    for command in commands:
        counts[command.op] += 1
    ssd_reads = [command for command in commands if command.op in {"kv_load", "kv_prefetch"} and command.source == "ssd"]
    ssd_writes = [command for command in commands if command.op == "kv_evict" and command.target == "ssd"]
    return RuntimeCommandExport(
        plan_path=str(plan_path),
        command_count=len(commands),
        command_counts_by_op={key: value for key, value in counts.items() if value},
        ssd_read_command_count=len(ssd_reads),
        ssd_write_command_count=len(ssd_writes),
        ssd_read_bytes=sum(command.size_bytes for command in ssd_reads),
        ssd_write_bytes=sum(command.size_bytes for command in ssd_writes),
        jsonl_path=str(jsonl_path) if jsonl_path is not None else None,
        summary_path=str(summary_path) if summary_path is not None else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--jsonl", default=None)
    parser.add_argument("--summary", default=None)
    args = parser.parse_args()

    result = export_runtime_commands(args.plan, jsonl_path=args.jsonl, summary_path=args.summary)
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
