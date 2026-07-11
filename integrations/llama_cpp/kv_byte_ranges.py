"""Verify SolidAttention staging bytes against llama.cpp KV row byte ranges."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.trace import TraceRecorder


@dataclass(frozen=True)
class KVByteRange:
    """One staging-to-llama.cpp tensor byte-range mapping."""

    kind: str
    layer_id: int
    block_id: int
    token_id: int
    staging_offset: int
    backing_offset: int
    tensor_offset: int
    size_bytes: int
    sha256: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"k", "v"}:
            raise ValueError("kind must be 'k' or 'v'")
        if self.layer_id < 0:
            raise ValueError("layer_id must be non-negative")
        if self.block_id < 0:
            raise ValueError("block_id must be non-negative")
        if self.token_id < 0:
            raise ValueError("token_id must be non-negative")
        if self.staging_offset < 0:
            raise ValueError("staging_offset must be non-negative")
        if self.backing_offset < 0:
            raise ValueError("backing_offset must be non-negative")
        if self.tensor_offset < 0:
            raise ValueError("tensor_offset must be non-negative")
        if self.size_bytes <= 0:
            raise ValueError("size_bytes must be positive")

    @property
    def staging_end_offset(self) -> int:
        return self.staging_offset + self.size_bytes

    @property
    def tensor_end_offset(self) -> int:
        return self.tensor_offset + self.size_bytes

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "kind": self.kind,
            "layer_id": self.layer_id,
            "block_id": self.block_id,
            "token_id": self.token_id,
            "staging_offset": self.staging_offset,
            "staging_end_offset": self.staging_end_offset,
            "backing_offset": self.backing_offset,
            "tensor_offset": self.tensor_offset,
            "tensor_end_offset": self.tensor_end_offset,
            "size_bytes": self.size_bytes,
        }
        if self.sha256 is not None:
            payload["sha256"] = self.sha256
        return payload


@dataclass(frozen=True)
class KVBlockByteMapping:
    """All K/V byte ranges for one runtime KV block."""

    layer_id: int
    block_id: int
    start_token: int
    end_token: int
    ssd_offset: int
    size_bytes: int
    key_row_size_bytes: int
    value_row_size_bytes: int
    ranges: tuple[KVByteRange, ...]
    block_sha256: str | None = None
    key_sha256: str | None = None
    value_sha256: str | None = None

    @property
    def token_count(self) -> int:
        return self.end_token - self.start_token

    @property
    def mapped_bytes(self) -> int:
        return sum(item.size_bytes for item in self.ranges)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "block_id": self.block_id,
            "start_token": self.start_token,
            "end_token": self.end_token,
            "token_count": self.token_count,
            "ssd_offset": self.ssd_offset,
            "size_bytes": self.size_bytes,
            "mapped_bytes": self.mapped_bytes,
            "key_row_size_bytes": self.key_row_size_bytes,
            "value_row_size_bytes": self.value_row_size_bytes,
            "range_count": len(self.ranges),
            "block_sha256": self.block_sha256,
            "key_sha256": self.key_sha256,
            "value_sha256": self.value_sha256,
            "ranges": [item.to_dict() for item in self.ranges],
        }


@dataclass(frozen=True)
class KVByteRangeVerificationResult:
    """Machine-readable byte-range verification metrics."""

    plan_path: str
    output_dir: str
    runtime_summary_path: str | None
    backing_file: str | None
    cache_size: int | None
    v_trans: bool
    block_count: int
    range_count: int
    mapped_bytes: int
    key_bytes: int
    value_bytes: int
    inferred_row_size_count: int
    size_mismatch_count: int
    coverage_error_count: int
    tensor_bounds_error_count: int
    backing_short_read_count: int
    ssd_alignment_error_count: int
    verification_error_count: int
    metrics_path: str
    trace_path: str
    summary_path: str
    mappings: tuple[KVBlockByteMapping, ...] = field(default_factory=tuple)

    def to_dict(self, *, include_mappings: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contract": "solidattention.llama_cpp.kv_byte_ranges.v1",
            "plan_path": self.plan_path,
            "output_dir": self.output_dir,
            "runtime_summary_path": self.runtime_summary_path,
            "backing_file": self.backing_file,
            "cache_size": self.cache_size,
            "v_trans": self.v_trans,
            "block_count": self.block_count,
            "range_count": self.range_count,
            "mapped_bytes": self.mapped_bytes,
            "key_bytes": self.key_bytes,
            "value_bytes": self.value_bytes,
            "inferred_row_size_count": self.inferred_row_size_count,
            "size_mismatch_count": self.size_mismatch_count,
            "coverage_error_count": self.coverage_error_count,
            "tensor_bounds_error_count": self.tensor_bounds_error_count,
            "backing_short_read_count": self.backing_short_read_count,
            "ssd_alignment_error_count": self.ssd_alignment_error_count,
            "verification_error_count": self.verification_error_count,
            "metrics_path": self.metrics_path,
            "trace_path": self.trace_path,
            "summary_path": self.summary_path,
        }
        if include_mappings:
            payload["mappings"] = [item.to_dict() for item in self.mappings]
        return payload


def verify_kv_byte_ranges(
    plan_path: str | Path,
    output_dir: str | Path,
    *,
    runtime_summary_path: str | Path | None = None,
    backing_file: str | Path | None = None,
    key_row_size_bytes: int | None = None,
    value_row_size_bytes: int | None = None,
    cache_size: int | None = None,
    v_trans: bool = False,
) -> KVByteRangeVerificationResult:
    """Verify staging K/V byte ranges against llama.cpp row-major tensor offsets."""

    if v_trans:
        raise ValueError("v_trans=True is not supported by the first byte-range verifier")

    plan_path = Path(plan_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plan = _load_plan(plan_path)
    runtime_summary_path = Path(runtime_summary_path) if runtime_summary_path is not None else None
    backing_path = Path(backing_file) if backing_file is not None else None
    runtime_summary = _load_runtime_summary(runtime_summary_path)
    if cache_size is None and runtime_summary is not None:
        raw_cache_size = runtime_summary.get("cache_size")
        cache_size = int(raw_cache_size) if raw_cache_size is not None else None

    blocks = _unique_blocks(plan)
    layout_by_key = {
        (int(item["layer_id"]), int(item["block_id"])): item
        for item in plan.get("layout", {}).get("blocks", [])
    }
    alignment = int(plan.get("layout", {}).get("alignment", 1))

    trace = TraceRecorder()
    mappings: list[KVBlockByteMapping] = []
    inferred_row_size_count = 0
    size_mismatch_count = 0
    coverage_error_count = 0
    tensor_bounds_error_count = 0
    backing_short_read_count = 0
    ssd_alignment_error_count = 0

    for index, block in enumerate(blocks):
        layer_id = int(block["layer_id"])
        block_id = int(block["block_id"])
        start_token = int(block["start_token"])
        end_token = int(block["end_token"])
        token_count = end_token - start_token
        size_bytes = int(block["size_bytes"])
        ssd_offset = int(block["ssd_offset"])
        if token_count <= 0:
            size_mismatch_count += 1
            continue
        if alignment > 1 and ssd_offset % alignment != 0:
            ssd_alignment_error_count += 1
        layout_range = layout_by_key.get((layer_id, block_id))
        if layout_range is not None and int(layout_range.get("size_bytes", size_bytes)) != size_bytes:
            size_mismatch_count += 1

        k_row, v_row, inferred = _resolve_row_sizes(
            size_bytes,
            token_count,
            key_row_size_bytes=key_row_size_bytes,
            value_row_size_bytes=value_row_size_bytes,
        )
        if inferred:
            inferred_row_size_count += 1
        if (k_row + v_row) * token_count != size_bytes:
            size_mismatch_count += 1

        block_bytes = _read_block_bytes(backing_path, ssd_offset, size_bytes)
        if block_bytes is not None and len(block_bytes) != size_bytes:
            backing_short_read_count += 1
            block_bytes = block_bytes + bytes(size_bytes - len(block_bytes))

        ranges = _build_block_ranges(
            layer_id=layer_id,
            block_id=block_id,
            start_token=start_token,
            end_token=end_token,
            ssd_offset=ssd_offset,
            key_row_size_bytes=k_row,
            value_row_size_bytes=v_row,
            block_bytes=block_bytes,
        )
        coverage_error_count += _coverage_errors(ranges, size_bytes)
        if cache_size is not None:
            tensor_bounds_error_count += _tensor_bounds_errors(ranges, cache_size, k_row, v_row)
        key_digest, value_digest = _kind_digests(ranges, block_bytes)
        mapping = KVBlockByteMapping(
            layer_id=layer_id,
            block_id=block_id,
            start_token=start_token,
            end_token=end_token,
            ssd_offset=ssd_offset,
            size_bytes=size_bytes,
            key_row_size_bytes=k_row,
            value_row_size_bytes=v_row,
            ranges=tuple(ranges),
            block_sha256=_sha256(block_bytes) if block_bytes is not None else None,
            key_sha256=key_digest,
            value_sha256=value_digest,
        )
        mappings.append(mapping)
        trace.record(
            op="kv_byte_range_map",
            layer_id=layer_id,
            block_id=block_id,
            device="staging",
            start_ms=float(index),
            end_ms=float(index),
            metadata={
                "start_token": start_token,
                "end_token": end_token,
                "ssd_offset": ssd_offset,
                "size_bytes": size_bytes,
                "key_row_size_bytes": k_row,
                "value_row_size_bytes": v_row,
                "range_count": len(ranges),
                "mapped_bytes": mapping.mapped_bytes,
                "block_sha256": mapping.block_sha256,
            },
        )

    mapped_bytes = sum(item.mapped_bytes for item in mappings)
    key_bytes = sum(item.key_row_size_bytes * item.token_count for item in mappings)
    value_bytes = sum(item.value_row_size_bytes * item.token_count for item in mappings)
    verification_error_count = (
        size_mismatch_count
        + coverage_error_count
        + tensor_bounds_error_count
        + backing_short_read_count
        + ssd_alignment_error_count
    )
    metrics_path = output / "kv_byte_range_metrics.json"
    trace_path = output / "kv_byte_range_trace.json"
    summary_path = output / "kv_byte_range_summary.json"
    result = KVByteRangeVerificationResult(
        plan_path=str(plan_path),
        output_dir=str(output),
        runtime_summary_path=str(runtime_summary_path) if runtime_summary_path is not None else None,
        backing_file=str(backing_path) if backing_path is not None else None,
        cache_size=cache_size,
        v_trans=v_trans,
        block_count=len(mappings),
        range_count=sum(len(item.ranges) for item in mappings),
        mapped_bytes=mapped_bytes,
        key_bytes=key_bytes,
        value_bytes=value_bytes,
        inferred_row_size_count=inferred_row_size_count,
        size_mismatch_count=size_mismatch_count,
        coverage_error_count=coverage_error_count,
        tensor_bounds_error_count=tensor_bounds_error_count,
        backing_short_read_count=backing_short_read_count,
        ssd_alignment_error_count=ssd_alignment_error_count,
        verification_error_count=verification_error_count,
        metrics_path=str(metrics_path),
        trace_path=str(trace_path),
        summary_path=str(summary_path),
        mappings=tuple(mappings),
    )
    metrics_path.write_text(json.dumps(result.to_dict(include_mappings=False), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    trace.save_json(trace_path)
    summary_path.write_text(json.dumps(result.to_dict(include_mappings=True), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _load_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("movement plan must be a JSON object")
    if "ops" not in payload:
        raise ValueError("movement plan missing ops")
    return payload


def _load_runtime_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("runtime summary must be a JSON object")
    return payload


def _unique_blocks(plan: dict[str, Any]) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for op in plan.get("ops", []):
        if not isinstance(op, dict):
            raise ValueError("movement plan ops must contain JSON objects")
        key = (int(op["layer_id"]), int(op["block_id"]), int(op["start_token"]), int(op["end_token"]))
        current = {
            "layer_id": int(op["layer_id"]),
            "block_id": int(op["block_id"]),
            "start_token": int(op["start_token"]),
            "end_token": int(op["end_token"]),
            "ssd_offset": int(op["ssd_offset"]),
            "size_bytes": int(op["size_bytes"]),
        }
        previous = by_key.get(key)
        if previous is not None and (
            previous["ssd_offset"] != current["ssd_offset"] or previous["size_bytes"] != current["size_bytes"]
        ):
            raise ValueError(f"inconsistent byte range for block {key}")
        by_key[key] = current
    return [by_key[key] for key in sorted(by_key)]


def _resolve_row_sizes(
    size_bytes: int,
    token_count: int,
    *,
    key_row_size_bytes: int | None,
    value_row_size_bytes: int | None,
) -> tuple[int, int, bool]:
    if key_row_size_bytes is not None and key_row_size_bytes <= 0:
        raise ValueError("key_row_size_bytes must be positive")
    if value_row_size_bytes is not None and value_row_size_bytes <= 0:
        raise ValueError("value_row_size_bytes must be positive")
    row_sum, remainder = divmod(size_bytes, token_count)
    if remainder != 0:
        return (key_row_size_bytes or 1, value_row_size_bytes or 1, False)
    if key_row_size_bytes is None and value_row_size_bytes is None:
        if row_sum % 2 != 0:
            raise ValueError("cannot infer equal K/V row sizes from odd per-token byte count")
        return (row_sum // 2, row_sum // 2, True)
    if key_row_size_bytes is None:
        assert value_row_size_bytes is not None
        return (row_sum - value_row_size_bytes, value_row_size_bytes, True)
    if value_row_size_bytes is None:
        return (key_row_size_bytes, row_sum - key_row_size_bytes, True)
    return (key_row_size_bytes, value_row_size_bytes, False)


def _build_block_ranges(
    *,
    layer_id: int,
    block_id: int,
    start_token: int,
    end_token: int,
    ssd_offset: int,
    key_row_size_bytes: int,
    value_row_size_bytes: int,
    block_bytes: bytes | None,
) -> list[KVByteRange]:
    ranges: list[KVByteRange] = []
    stride = key_row_size_bytes + value_row_size_bytes
    for token_id in range(start_token, end_token):
        token_index = token_id - start_token
        tensor_cell = block_id + token_index
        key_staging_offset = token_index * stride
        value_staging_offset = key_staging_offset + key_row_size_bytes
        ranges.append(
            KVByteRange(
                kind="k",
                layer_id=layer_id,
                block_id=block_id,
                token_id=token_id,
                staging_offset=key_staging_offset,
                backing_offset=ssd_offset + key_staging_offset,
                tensor_offset=tensor_cell * key_row_size_bytes,
                size_bytes=key_row_size_bytes,
                sha256=_slice_sha256(block_bytes, key_staging_offset, key_row_size_bytes),
            )
        )
        ranges.append(
            KVByteRange(
                kind="v",
                layer_id=layer_id,
                block_id=block_id,
                token_id=token_id,
                staging_offset=value_staging_offset,
                backing_offset=ssd_offset + value_staging_offset,
                tensor_offset=tensor_cell * value_row_size_bytes,
                size_bytes=value_row_size_bytes,
                sha256=_slice_sha256(block_bytes, value_staging_offset, value_row_size_bytes),
            )
        )
    return ranges


def _coverage_errors(ranges: list[KVByteRange], size_bytes: int) -> int:
    expected = 0
    for item in sorted(ranges, key=lambda value: value.staging_offset):
        if item.staging_offset != expected:
            return 1
        expected = item.staging_end_offset
    return 0 if expected == size_bytes else 1


def _tensor_bounds_errors(
    ranges: list[KVByteRange],
    cache_size: int,
    key_row_size_bytes: int,
    value_row_size_bytes: int,
) -> int:
    errors = 0
    key_limit = cache_size * key_row_size_bytes
    value_limit = cache_size * value_row_size_bytes
    for item in ranges:
        limit = key_limit if item.kind == "k" else value_limit
        if item.tensor_end_offset > limit:
            errors += 1
    return errors


def _read_block_bytes(backing_file: Path | None, offset: int, size_bytes: int) -> bytes | None:
    if backing_file is None:
        return None
    with backing_file.open("rb") as handle:
        handle.seek(offset)
        return handle.read(size_bytes)


def _kind_digests(ranges: list[KVByteRange], block_bytes: bytes | None) -> tuple[str | None, str | None]:
    if block_bytes is None:
        return None, None
    key_hash = hashlib.sha256()
    value_hash = hashlib.sha256()
    for item in ranges:
        data = block_bytes[item.staging_offset : item.staging_end_offset]
        if item.kind == "k":
            key_hash.update(data)
        else:
            value_hash.update(data)
    return key_hash.hexdigest(), value_hash.hexdigest()


def _slice_sha256(data: bytes | None, offset: int, size_bytes: int) -> str | None:
    if data is None:
        return None
    return _sha256(data[offset : offset + size_bytes])


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-summary", default=None)
    parser.add_argument("--backing-file", default=None)
    parser.add_argument("--key-row-size-bytes", type=int, default=None)
    parser.add_argument("--value-row-size-bytes", type=int, default=None)
    parser.add_argument("--cache-size", type=int, default=None)
    parser.add_argument("--v-trans", action="store_true")
    args = parser.parse_args()
    result = verify_kv_byte_ranges(
        args.plan,
        args.output_dir,
        runtime_summary_path=args.runtime_summary,
        backing_file=args.backing_file,
        key_row_size_bytes=args.key_row_size_bytes,
        value_row_size_bytes=args.value_row_size_bytes,
        cache_size=args.cache_size,
        v_trans=args.v_trans,
    )
    print(json.dumps(result.to_dict(include_mappings=False), sort_keys=True))
    return 0 if result.verification_error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
