import json

from harness.runtime_replay import run_runtime_replay


def make_fake_probe(tmp_path, *, read_latency_ns=4_000_000, write_latency_ns=8_000_000):
    script = tmp_path / "fake_probe.py"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "path = pathlib.Path(sys.argv[1])\n"
        "block_size = int(sys.argv[2])\n"
        "block_count = int(sys.argv[3])\n"
        "path.parent.mkdir(parents=True, exist_ok=True)\n"
        "path.write_bytes(b\'0\' * (block_size * block_count))\n"
        f"print(json.dumps({{'backend': 'fake', 'path': str(path), 'block_size': block_size, 'block_count': block_count, 'bytes': block_size * block_count, 'write_latency_ns': {write_latency_ns}, 'read_latency_ns': {read_latency_ns}}}))\n"
    )
    script.chmod(0o755)
    return script


def test_runtime_replay_writes_scheduler_metrics_and_trace(tmp_path) -> None:
    trace_path = tmp_path / "runtime_trace.jsonl"
    output_dir = tmp_path / "replay"
    trace_path.write_text(
        '{"op":"kv_add","layer_id":0,"block_id":0,"start_token":0,"end_token":64,"size_bytes":1024,"location":"dram","metadata":{"event":"kv_cache_apply_ubatch"}}\n'
        '{"op":"kv_add","layer_id":1,"block_id":0,"start_token":0,"end_token":64,"size_bytes":1024,"location":"dram","metadata":{"event":"kv_cache_apply_ubatch"}}\n'
        '{"op":"kv_add","layer_id":0,"block_id":0,"start_token":64,"end_token":128,"size_bytes":2048,"location":"dram","metadata":{"event":"kv_cache_apply_ubatch"}}\n'
    )

    result = run_runtime_replay(trace_path, output_dir, initial_location="ssd")

    assert result.event_count == 3
    assert result.unique_block_count == 2
    assert result.before_location_stats["ssd"] == {"block_count": 2, "size_bytes": 3072}
    assert result.after_location_stats["vram"] == {"block_count": 2, "size_bytes": 3072}
    assert result.ssd_read_latency_source == "configured"
    assert result.io_probe_result is None
    assert result.movement_plan_path == str(output_dir / "runtime_movement_plan.json")
    assert result.ssd_layout_alignment == 4096

    metrics = json.loads((output_dir / "runtime_replay_metrics.json").read_text())
    assert metrics["metadata"]["source"] == "llama.cpp runtime trace"
    assert metrics["selected_block_count"] == 2
    assert metrics["ssd_read_bytes"] == 3072
    assert metrics["io_op_count"] == 2

    trace = json.loads((output_dir / "runtime_replay_trace.json").read_text())
    assert [event["op"] for event in trace][:2] == ["schedule_ssd_read", "schedule_dram_to_vram"]
    movement_plan = json.loads((output_dir / "runtime_movement_plan.json").read_text())
    assert movement_plan["ops"][0]["op"] == "kv_load"
    assert movement_plan["ops"][0]["source"] == "ssd"
    assert movement_plan["ops"][0]["target"] == "dram"
    assert movement_plan["layout"]["total_size_bytes"] >= 3072
    summary = json.loads((output_dir / "runtime_replay_summary.json").read_text())
    assert summary["movement_plan"]["ops"][0]["op"] == "kv_load"
    manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema"] == "solidattention.run_manifest.v1"
    assert manifest["status"] == "passed"
    assert manifest["profile"] == "runtime-replay"
    assert manifest["inputs"][0]["sha256"]


def test_runtime_replay_can_keep_trace_locations(tmp_path) -> None:
    trace_path = tmp_path / "runtime_trace.jsonl"
    output_dir = tmp_path / "replay"
    trace_path.write_text(
        '{"op":"kv_add","layer_id":0,"block_id":0,"start_token":0,"end_token":64,"size_bytes":1024,"location":"dram","metadata":{}}\n'
    )

    result = run_runtime_replay(trace_path, output_dir, initial_location="trace")

    metrics = json.loads((output_dir / "runtime_replay_metrics.json").read_text())
    assert result.before_location_stats["dram"] == {"block_count": 1, "size_bytes": 1024}
    assert metrics["ssd_read_bytes"] == 0
    assert metrics["load_blocking_latency_ms"] > 0


def test_runtime_replay_calibration_without_ssd_does_not_require_probe(
    tmp_path,
) -> None:
    trace_path = tmp_path / "runtime_trace.jsonl"
    output_dir = tmp_path / "replay"
    trace_path.write_text(
        '{"op":"kv_add","layer_id":0,"block_id":0,"start_token":0,"end_token":64,"size_bytes":1024,"location":"dram","metadata":{}}\n'
    )

    result = run_runtime_replay(
        trace_path,
        output_dir,
        initial_location="trace",
        calibrate_ssd_io=True,
        io_probe_executable=tmp_path / "missing_probe",
    )

    manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert result.ssd_read_latency_source == "no_ssd_blocks"
    assert manifest["status"] == "passed"
    assert manifest["io_backend"] == "mock"
    assert [item["path"] for item in manifest["inputs"]] == [str(trace_path)]


def test_runtime_replay_trace_locations_respect_runtime_evict_events(tmp_path) -> None:
    trace_path = tmp_path / "runtime_trace.jsonl"
    output_dir = tmp_path / "replay"
    trace_path.write_text(
        '{"op":"kv_add","layer_id":0,"block_id":0,"start_token":0,"end_token":64,"size_bytes":1024,"location":"dram","source":"runtime","target":"dram","ssd_offset":-1,"metadata":{"event":"kv_cache_apply_ubatch"}}\n'
        '{"op":"kv_add","layer_id":0,"block_id":64,"start_token":64,"end_token":128,"size_bytes":2048,"location":"dram","source":"runtime","target":"dram","ssd_offset":-1,"metadata":{"event":"kv_cache_apply_ubatch"}}\n'
        '{"op":"kv_evict","layer_id":0,"block_id":64,"start_token":64,"end_token":128,"size_bytes":2048,"location":"ssd","source":"dram","target":"ssd","ssd_offset":-1,"metadata":{"event":"kv_cache_seq_rm"}}\n'
    )
    fake_probe = make_fake_probe(tmp_path, read_latency_ns=3_000_000, write_latency_ns=7_000_000)

    result = run_runtime_replay(
        trace_path,
        output_dir,
        initial_location="trace",
        calibrate_ssd_io=True,
        io_probe_executable=fake_probe,
    )

    metrics = json.loads((output_dir / "runtime_replay_metrics.json").read_text())
    trace = json.loads((output_dir / "runtime_replay_trace.json").read_text())
    movement_plan = json.loads((output_dir / "runtime_movement_plan.json").read_text())

    assert result.before_location_stats["dram"] == {"block_count": 1, "size_bytes": 1024}
    assert result.before_location_stats["ssd"] == {"block_count": 1, "size_bytes": 2048}
    assert result.io_probe_result["block_count"] == 1
    assert result.io_probe_result["block_size"] == 2048
    assert metrics["metadata"]["ssd_calibration_block_count"] == 1
    assert metrics["metadata"]["ssd_calibration_size_bytes"] == 2048
    assert metrics["ssd_read_bytes"] == 2048
    assert metrics["io_op_count"] == 1
    assert [event["op"] for event in trace].count("schedule_ssd_read") == 1
    assert [op["op"] for op in movement_plan["ops"]].count("kv_load") == 3
    assert movement_plan["ops"][2]["source"] == "ssd"


def test_runtime_replay_can_calibrate_ssd_latency_with_io_probe(tmp_path) -> None:
    trace_path = tmp_path / "runtime_trace.jsonl"
    output_dir = tmp_path / "replay"
    trace_path.write_text(
        '{"op":"kv_add","layer_id":0,"block_id":0,"start_token":0,"end_token":64,"size_bytes":1024,"location":"dram","metadata":{}}\n'
        '{"op":"kv_add","layer_id":1,"block_id":0,"start_token":0,"end_token":64,"size_bytes":2048,"location":"dram","metadata":{}}\n'
    )
    fake_probe = make_fake_probe(tmp_path, read_latency_ns=4_000_000, write_latency_ns=8_000_000)

    result = run_runtime_replay(
        trace_path,
        output_dir,
        initial_location="ssd",
        calibrate_ssd_io=True,
        io_probe_executable=fake_probe,
    )

    metrics = json.loads((output_dir / "runtime_replay_metrics.json").read_text())
    assert result.ssd_read_latency_source == "io_probe"
    assert result.ssd_read_latency_ms == 2.0
    assert result.io_probe_result["backend"] == "fake"
    assert result.io_probe_result["block_size"] == 2048
    assert result.io_probe_result["block_count"] == 2
    assert result.ssd_layout_total_size_bytes >= 3072
    assert metrics["metadata"]["ssd_read_latency_source"] == "io_probe"
    assert metrics["metadata"]["io_probe_result"]["read_latency_ms_per_block"] == 2.0
    assert metrics["gpu_wait_time_ms"] == 4.5
    assert metrics["ssd_read_bytes"] == 3072
    manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["io_backend"] == "fake"
