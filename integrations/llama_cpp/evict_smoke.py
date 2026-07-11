"""Run a patched llama.cpp runtime scenario that must emit kv_evict events."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from harness.process import build_executable_command
from integrations.llama_cpp.adapter import load_jsonl_events
from integrations.llama_cpp.runtime_trace import summarize_runtime_trace


DEFAULT_ARGS = ("-a", "llama", "-s", "1")


@dataclass(frozen=True)
class EvictSmokeResult:
    executable: str | None
    trace_path: str
    summary_path: str | None
    harness_trace_path: str | None
    stdout_path: str | None
    stderr_path: str | None
    returncode: int | None
    event_count: int
    add_event_count: int
    evict_event_count: int
    ops: tuple[str, ...]
    location_stats: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, object]:
        return {
            "executable": self.executable,
            "trace_path": self.trace_path,
            "summary_path": self.summary_path,
            "harness_trace_path": self.harness_trace_path,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
            "returncode": self.returncode,
            "event_count": self.event_count,
            "add_event_count": self.add_event_count,
            "evict_event_count": self.evict_event_count,
            "ops": list(self.ops),
            "location_stats": self.location_stats,
        }


def summarize_evict_trace(
    trace_path: str | Path,
    *,
    summary_path: str | Path | None = None,
    harness_trace_path: str | Path | None = None,
    executable: str | Path | None = None,
    returncode: int | None = None,
    stdout_path: str | Path | None = None,
    stderr_path: str | Path | None = None,
    require_evict: bool = True,
) -> EvictSmokeResult:
    trace_path = Path(trace_path)
    events = load_jsonl_events(trace_path)
    evict_event_count = sum(1 for event in events if event.op == "kv_evict")
    if require_evict and evict_event_count == 0:
        raise RuntimeError(f"trace did not contain kv_evict events: {trace_path}")

    summary = summarize_runtime_trace(
        trace_path,
        summary_path=summary_path,
        harness_trace_path=harness_trace_path,
    )
    return EvictSmokeResult(
        executable=str(executable) if executable is not None else None,
        trace_path=str(trace_path),
        summary_path=str(summary_path) if summary_path is not None else None,
        harness_trace_path=str(harness_trace_path) if harness_trace_path is not None else None,
        stdout_path=str(stdout_path) if stdout_path is not None else None,
        stderr_path=str(stderr_path) if stderr_path is not None else None,
        returncode=returncode,
        event_count=len(events),
        add_event_count=sum(1 for event in events if event.op == "kv_add"),
        evict_event_count=evict_event_count,
        ops=summary.ops,
        location_stats=summary.location_stats,
    )


def run_evict_smoke(
    executable: str | Path,
    *,
    trace_path: str | Path,
    summary_path: str | Path | None = None,
    harness_trace_path: str | Path | None = None,
    stdout_path: str | Path | None = None,
    stderr_path: str | Path | None = None,
    args: Sequence[str] = DEFAULT_ARGS,
) -> EvictSmokeResult:
    executable = Path(executable)
    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text("", encoding="utf-8")

    env = os.environ.copy()
    env["LLAMA_SOLIDATTENTION_KV_TRACE_PATH"] = str(trace_path)
    env["LLAMA_SOLIDATTENTION_KV_TRACE_EVICT_SMOKE"] = "1"

    completed = subprocess.run(
        build_executable_command(executable, *args),
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    if stdout_path is not None:
        output = Path(stdout_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(completed.stdout, encoding="utf-8")
    if stderr_path is not None:
        output = Path(stderr_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(completed.stderr, encoding="utf-8")

    if completed.returncode != 0:
        raise RuntimeError(
            f"evict smoke executable failed with code {completed.returncode}: {executable}"
        )

    return summarize_evict_trace(
        trace_path,
        summary_path=summary_path,
        harness_trace_path=harness_trace_path,
        executable=executable,
        returncode=completed.returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--harness-trace", default=None)
    parser.add_argument("--stdout", default=None)
    parser.add_argument("--stderr", default=None)
    parser.add_argument("--arg", action="append", default=None, help="argument passed to test-llama-archs; repeatable")
    args = parser.parse_args()

    result = run_evict_smoke(
        args.executable,
        trace_path=args.trace,
        summary_path=args.summary,
        harness_trace_path=args.harness_trace,
        stdout_path=args.stdout,
        stderr_path=args.stderr,
        args=tuple(args.arg) if args.arg else DEFAULT_ARGS,
    )
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
