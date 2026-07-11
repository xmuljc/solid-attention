import json

from harness.smoke import load_smoke_config, run_smoke


def test_load_smoke_config_reads_minimal_yaml() -> None:
    config = load_smoke_config("configs/smoke_base.yaml")

    assert config["num_layers"] == 2
    assert config["num_blocks_per_layer"] == 4
    assert config["block_size_bytes"] == 4096
    assert config["locations"] == ["ssd", "dram", "vram"]


def test_run_smoke_writes_trace_metrics_and_summary(tmp_path) -> None:
    summary = run_smoke("configs/smoke_base.yaml", tmp_path)

    trace_path = tmp_path / "smoke_trace.json"
    metrics_path = tmp_path / "smoke_metrics.json"
    summary_path = tmp_path / "smoke_summary.json"
    manifest_path = tmp_path / "run_manifest.json"

    assert trace_path.exists()
    assert metrics_path.exists()
    assert summary_path.exists()
    assert manifest_path.exists()
    assert summary["block_count"] == 8
    assert len(summary["layout"]) == 8
    assert len(summary["selection"]) == 2
    assert len(summary["schedules"]) == 2

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    ops = {event["op"] for event in trace}
    assert "select_init" in ops
    assert "prefetch_submit" in ops
    assert "schedule_compute" in ops
    assert metrics["selected_block_count"] > 0
    assert metrics["io_op_count"] >= 0
    assert summary_payload["outputs"]["trace"] == str(trace_path)
    assert manifest["schema"] == "solidattention.run_manifest.v1"
    assert manifest["profile"] == "smoke"
    assert manifest["status"] == "passed"
    assert manifest["io_backend"] == "mock"
    assert {record["kind"] for record in manifest["outputs"]} == {
        "trace",
        "metrics",
        "summary",
    }
