import json

import pytest

from integrations.llama_cpp.evict_smoke import summarize_evict_trace


def write_jsonl(path, events) -> None:
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


def test_summarize_evict_trace_requires_kv_evict(tmp_path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    write_jsonl(
        trace_path,
        [
            {
                "op": "kv_add",
                "layer_id": 0,
                "block_id": 0,
                "start_token": 0,
                "end_token": 64,
                "size_bytes": 1024,
                "location": "dram",
                "source": "runtime",
                "target": "dram",
                "ssd_offset": -1,
                "metadata": {"event": "kv_cache_apply_ubatch"},
            }
        ],
    )

    with pytest.raises(RuntimeError, match="kv_evict"):
        summarize_evict_trace(trace_path)


def test_summarize_evict_trace_counts_add_and_evict_events(tmp_path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    summary_path = tmp_path / "summary.json"
    harness_trace_path = tmp_path / "harness_trace.json"
    write_jsonl(
        trace_path,
        [
            {
                "op": "kv_add",
                "layer_id": 0,
                "block_id": 64,
                "start_token": 64,
                "end_token": 128,
                "size_bytes": 2048,
                "location": "dram",
                "source": "runtime",
                "target": "dram",
                "ssd_offset": -1,
                "metadata": {"event": "kv_cache_apply_ubatch"},
            },
            {
                "op": "kv_evict",
                "layer_id": 0,
                "block_id": 64,
                "start_token": 64,
                "end_token": 128,
                "size_bytes": 2048,
                "location": "ssd",
                "source": "dram",
                "target": "ssd",
                "ssd_offset": -1,
                "metadata": {"event": "kv_cache_seq_rm"},
            },
        ],
    )

    result = summarize_evict_trace(
        trace_path,
        summary_path=summary_path,
        harness_trace_path=harness_trace_path,
        executable="test-llama-archs",
        returncode=0,
    )

    assert result.event_count == 2
    assert result.add_event_count == 1
    assert result.evict_event_count == 1
    assert result.ops == ("kv_add", "kv_evict")
    assert result.location_stats["ssd"] == {"block_count": 1, "size_bytes": 2048}
    assert summary_path.exists()
    assert harness_trace_path.exists()
    assert json.loads(summary_path.read_text())["ops"] == ["kv_add", "kv_evict"]
