import json

from harness.ablation import METHODS, run_ablation


def test_run_ablation_writes_metrics_for_all_methods(tmp_path) -> None:
    summary = run_ablation("configs/smoke_base.yaml", tmp_path)
    manifest_path = tmp_path / "run_manifest.json"

    assert summary["methods"] == list(METHODS)
    assert set(summary["methods"]) == {
        "memory_only",
        "ssd_sync",
        "ssd_async",
        "prefetch_only",
        "full_scheduler",
        "solid_sim",
    }
    for method in METHODS:
        result = summary["results"][method]
        trace_path = tmp_path / f"{method}_trace.json"
        metrics_path = tmp_path / f"{method}_metrics.json"
        assert result["trace"] == str(trace_path)
        assert result["metrics"] == str(metrics_path)
        assert trace_path.exists()
        assert metrics_path.exists()
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert payload["metadata"]["method"] == method
        assert payload["selected_block_count"] > 0

    assert manifest_path.exists()
    assert summary["manifest_path"] == str(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "solidattention.run_manifest.v1"
    assert manifest["profile"] == "ablation"
    assert manifest["status"] == "passed"
    assert {record["kind"] for record in manifest["outputs"]} == {
        "trace",
        "metrics",
        "summary",
    }
    assert len(manifest["outputs"]) == 2 * len(METHODS) + 1
    output_records = {
        (record["kind"], record["path"]) for record in manifest["outputs"]
    }
    for method in METHODS:
        assert ("trace", f"{method}_trace.json") in output_records
        assert ("metrics", f"{method}_metrics.json") in output_records
    assert ("summary", "ablation_summary.json") in output_records


def test_ablation_methods_capture_expected_io_differences(tmp_path) -> None:
    summary = run_ablation("configs/smoke_base.yaml", tmp_path)

    memory_metrics = summary["results"]["memory_only"]["metrics_payload"]
    sync_metrics = summary["results"]["ssd_sync"]["metrics_payload"]
    async_metrics = summary["results"]["ssd_async"]["metrics_payload"]
    prefetch_metrics = summary["results"]["prefetch_only"]["metrics_payload"]
    full_metrics = summary["results"]["full_scheduler"]["metrics_payload"]
    solid_metrics = summary["results"]["solid_sim"]["metrics_payload"]

    assert memory_metrics["ssd_read_bytes"] == 0
    assert sync_metrics["ssd_read_bytes"] > 0
    assert async_metrics["ssd_read_bytes"] > 0
    assert async_metrics["gpu_wait_time_ms"] < sync_metrics["gpu_wait_time_ms"]

    assert prefetch_metrics["prefetch_attempt_count"] > 0
    assert prefetch_metrics["prefetch_hit_count"] > 0
    assert prefetch_metrics["prefetch_miss_count"] > 0

    assert full_metrics["prefetch_attempt_count"] > 0
    assert full_metrics["prefetch_hit_count"] > 0
    assert full_metrics["prefetch_miss_count"] > 0
    assert solid_metrics["metadata"]["method_kind"] == "full_scheduler"
    assert solid_metrics["prefetch_attempt_count"] == full_metrics["prefetch_attempt_count"]
