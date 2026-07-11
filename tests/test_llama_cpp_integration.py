import pytest

from integrations.llama_cpp.adapter import LlamaKVCacheAdapter, load_jsonl_events
from integrations.llama_cpp.probe import probe_llama_cpp
from integrations.llama_cpp.runtime_trace import summarize_runtime_trace
from harness.trace import TraceRecorder


def test_llama_adapter_ingests_add_and_move_events() -> None:
    trace = TraceRecorder()
    adapter = LlamaKVCacheAdapter(trace=trace)

    block = adapter.ingest(
        {
            "op": "kv_add",
            "layer_id": 1,
            "block_id": 2,
            "start_token": 64,
            "end_token": 96,
            "size_bytes": 4096,
            "location": "ssd",
        }
    )

    assert block.layer_id == 1
    assert block.block_id == 2
    assert block.is_resident("ssd")

    moved = adapter.ingest(
        {
            "op": "kv_move",
            "layer_id": 1,
            "block_id": 2,
            "start_token": 64,
            "end_token": 96,
            "size_bytes": 4096,
            "location": "dram",
        }
    )

    assert moved.is_resident("dram")
    assert [event["op"] for event in trace.to_list()] == ["llama_kv_add", "llama_kv_move"]
    assert trace.to_list()[0]["metadata"]["source"] == "llama.cpp"


def test_llama_adapter_ingests_movement_schema_events() -> None:
    trace = TraceRecorder()
    adapter = LlamaKVCacheAdapter(trace=trace)

    adapter.ingest(
        {
            "op": "kv_add",
            "layer_id": 0,
            "block_id": 4,
            "start_token": 128,
            "end_token": 192,
            "size_bytes": 8192,
            "location": "ssd",
            "source": "runtime",
            "target": "ssd",
            "ssd_offset": 32768,
            "metadata": {"event": "seed"},
        }
    )
    loaded = adapter.ingest(
        {
            "op": "kv_load",
            "layer_id": 0,
            "block_id": 4,
            "start_token": 128,
            "end_token": 192,
            "size_bytes": 8192,
            "location": "dram",
            "source": "ssd",
            "target": "dram",
            "ssd_offset": 32768,
        }
    )
    prefetched = adapter.ingest(
        {
            "op": "kv_prefetch",
            "layer_id": 0,
            "block_id": 4,
            "start_token": 128,
            "end_token": 192,
            "size_bytes": 8192,
            "location": "vram",
            "source": "dram",
            "target": "vram",
            "ssd_offset": 32768,
        }
    )
    evicted = adapter.ingest(
        {
            "op": "kv_evict",
            "layer_id": 0,
            "block_id": 4,
            "start_token": 128,
            "end_token": 192,
            "size_bytes": 8192,
            "location": "ssd",
            "source": "vram",
            "target": "ssd",
            "ssd_offset": 32768,
        }
    )

    assert loaded.is_resident("dram")
    assert prefetched.is_resident("vram")
    assert evicted.is_resident("ssd")
    assert adapter.store.get_block(0, 4).is_resident("ssd")

    events = trace.to_list()
    assert [event["op"] for event in events] == [
        "llama_kv_add",
        "llama_kv_load",
        "llama_kv_prefetch",
        "llama_kv_evict",
    ]
    assert events[1]["metadata"]["event_source"] == "ssd"
    assert events[1]["metadata"]["event_target"] == "dram"
    assert events[1]["metadata"]["ssd_offset"] == 32768
    assert events[-1]["device"] == "ssd"


def test_llama_adapter_movement_event_can_create_missing_block() -> None:
    adapter = LlamaKVCacheAdapter()

    block = adapter.ingest(
        {
            "op": "kv_load",
            "layer_id": 2,
            "block_id": 9,
            "start_token": 288,
            "end_token": 320,
            "size_bytes": 4096,
            "location": "dram",
            "source": "ssd",
            "target": "dram",
            "ssd_offset": 65536,
        }
    )

    assert block.is_resident("dram")
    assert adapter.store.get_block(2, 9).metadata["event_source"] == "ssd"


def test_llama_adapter_ingests_jsonl_trace_events(tmp_path) -> None:
    trace_path = tmp_path / "kv_trace.jsonl"
    trace_path.write_text(
        '{"op":"kv_add","layer_id":0,"block_id":8,"start_token":32,"end_token":36,"size_bytes":2048,"location":"dram","metadata":{"event":"kv_cache_apply_ubatch","cache_size":128}}\n'
        '{"op":"kv_add","layer_id":1,"block_id":8,"start_token":32,"end_token":36,"size_bytes":2048,"location":"dram","metadata":{"event":"kv_cache_apply_ubatch","cache_size":128}}\n'
    )

    trace = TraceRecorder()
    adapter = LlamaKVCacheAdapter(trace=trace)

    blocks = adapter.ingest_jsonl(trace_path)

    assert [(block.layer_id, block.block_id) for block in blocks] == [(0, 8), (1, 8)]
    assert adapter.store.location_stats()["dram"] == {"block_count": 2, "size_bytes": 4096}
    assert trace.to_list()[0]["metadata"]["event"] == "kv_cache_apply_ubatch"
    assert trace.to_list()[0]["metadata"]["cache_size"] == 128


def test_llama_adapter_replace_policy_replays_duplicate_runtime_slots(tmp_path) -> None:
    trace_path = tmp_path / "runtime_trace.jsonl"
    trace_path.write_text(
        '{"op":"kv_add","layer_id":0,"block_id":0,"start_token":0,"end_token":64,"size_bytes":1024,"location":"dram","metadata":{"event":"kv_cache_apply_ubatch"}}\n'
        '{"op":"kv_add","layer_id":0,"block_id":0,"start_token":64,"end_token":128,"size_bytes":2048,"location":"dram","metadata":{"event":"kv_cache_apply_ubatch"}}\n'
    )

    strict_adapter = LlamaKVCacheAdapter()
    with pytest.raises(KeyError):
        strict_adapter.ingest_jsonl(trace_path)

    trace = TraceRecorder()
    adapter = LlamaKVCacheAdapter(trace=trace)
    blocks = adapter.ingest_jsonl(trace_path, duplicate_policy="replace")

    assert len(blocks) == 2
    assert adapter.store.get_block(0, 0).start_token == 64
    assert adapter.store.get_block(0, 0).size_bytes == 2048
    assert trace.to_list()[1]["metadata"]["replaced_existing"] is True


def test_runtime_trace_summary_counts_duplicates_and_writes_outputs(tmp_path) -> None:
    trace_path = tmp_path / "runtime_trace.jsonl"
    summary_path = tmp_path / "runtime_summary.json"
    harness_trace_path = tmp_path / "harness_trace.json"
    trace_path.write_text(
        '{"op":"kv_add","layer_id":0,"block_id":0,"start_token":0,"end_token":64,"size_bytes":1024,"location":"dram","metadata":{"event":"kv_cache_apply_ubatch"}}\n'
        '{"op":"kv_add","layer_id":1,"block_id":0,"start_token":0,"end_token":64,"size_bytes":1024,"location":"dram","metadata":{"event":"kv_cache_apply_ubatch"}}\n'
        '{"op":"kv_add","layer_id":0,"block_id":0,"start_token":64,"end_token":128,"size_bytes":2048,"location":"dram","metadata":{"event":"kv_cache_apply_ubatch"}}\n'
    )

    summary = summarize_runtime_trace(trace_path, summary_path=summary_path, harness_trace_path=harness_trace_path)

    assert summary.event_count == 3
    assert summary.unique_block_count == 2
    assert summary.duplicate_event_count == 1
    assert summary.total_event_size_bytes == 4096
    assert summary.layers == (0, 1)
    assert summary.ops == ("kv_add",)
    assert summary.location_stats["dram"] == {"block_count": 2, "size_bytes": 3072}
    assert summary_path.exists()
    assert harness_trace_path.exists()


def test_load_jsonl_events_reports_line_number_for_invalid_payload(tmp_path) -> None:
    trace_path = tmp_path / "bad_trace.jsonl"
    trace_path.write_text(
        '{"op":"kv_add","layer_id":0,"block_id":0,"start_token":0,"end_token":1,"size_bytes":1,"location":"dram"}\n'
        'not-json\n'
    )

    with pytest.raises(ValueError, match="line 2"):
        load_jsonl_events(trace_path)


def test_llama_adapter_rejects_unknown_events() -> None:
    adapter = LlamaKVCacheAdapter()

    with pytest.raises(ValueError):
        adapter.ingest(
            {
                "op": "unknown",
                "layer_id": 0,
                "block_id": 0,
                "start_token": 0,
                "end_token": 1,
                "size_bytes": 1,
                "location": "ssd",
            }
        )


def test_probe_reports_missing_checkout() -> None:
    result = probe_llama_cpp("/definitely/not/a/llama.cpp/path")

    assert result.state == "missing"
    assert not result.exists
    assert not result.ready
    assert "CMakeLists.txt" in result.missing_required


def test_probe_reports_uninitialized_checkout(tmp_path) -> None:
    checkout = tmp_path / "llama.cpp"
    checkout.mkdir()

    result = probe_llama_cpp(checkout)

    assert result.state == "uninitialized"
    assert result.exists
    assert not result.ready


def test_probe_reports_invalid_nonempty_checkout(tmp_path) -> None:
    checkout = tmp_path / "llama.cpp"
    checkout.mkdir()
    (checkout / "unexpected.txt").write_text("partial\n", encoding="utf-8")

    result = probe_llama_cpp(checkout)

    assert result.state == "invalid"
    assert result.exists
    assert not result.ready


def test_probe_accepts_minimal_checkout_shape(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "include").mkdir()
    (tmp_path / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.16)\n")
    (tmp_path / "src" / "llama.cpp").write_text("// placeholder\n")
    (tmp_path / "include" / "llama.h").write_text("// placeholder\n")

    result = probe_llama_cpp(tmp_path)

    assert result.state == "ready"
    assert result.exists
    assert result.ready
    assert result.to_dict()["ready"] is True


def test_probe_reports_partial_checkout_shape(tmp_path) -> None:
    (tmp_path / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.16)\n")

    result = probe_llama_cpp(tmp_path)

    assert result.state == "invalid"
    assert result.exists
    assert not result.ready
    assert result.missing_required == ()
    assert result.missing_any_of


@pytest.mark.llama
def test_real_llama_cpp_checkout_is_ready_when_initialized() -> None:
    result = probe_llama_cpp("external/llama.cpp")
    if result.state in {"missing", "uninitialized"}:
        pytest.skip(f"llama.cpp checkout is {result.state}")
    assert result.ready, result.to_dict()
