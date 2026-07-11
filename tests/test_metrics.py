import json

from harness.metrics import MetricsCollector


def test_metrics_collector_accumulates_required_metrics() -> None:
    metrics = MetricsCollector(metadata={"config": "smoke"})

    metrics.add(
        total_latency_ms=10.0,
        gpu_wait_time_ms=1.5,
        load_blocking_latency_ms=2.0,
        ssd_read_bytes=4096,
        ssd_write_bytes=2048,
        io_op_count=2,
        selected_block_count=3,
        prefetch_hit_count=1,
        prefetch_miss_count=1,
        wrong_prefetch_count=1,
    )
    metrics.add(total_latency_ms=5.0, prefetch_hit_count=1)

    payload = metrics.to_dict()

    assert payload["total_latency_ms"] == 15.0
    assert payload["gpu_wait_time_ms"] == 1.5
    assert payload["load_blocking_latency_ms"] == 2.0
    assert payload["ssd_read_bytes"] == 4096
    assert payload["ssd_write_bytes"] == 2048
    assert payload["io_op_count"] == 2
    assert payload["selected_block_count"] == 3
    assert payload["prefetch_hit_count"] == 2
    assert payload["prefetch_miss_count"] == 1
    assert payload["wrong_prefetch_count"] == 1
    assert payload["prefetch_hit_rate"] == 2 / 3
    assert payload["prefetch_miss_rate"] == 1 / 3
    assert payload["wrong_prefetch_rate"] == 1 / 3
    assert payload["metadata"] == {"config": "smoke"}


def test_metrics_rates_are_zero_without_prefetch_attempts() -> None:
    metrics = MetricsCollector()

    assert metrics.prefetch_hit_rate == 0.0
    assert metrics.prefetch_miss_rate == 0.0
    assert metrics.wrong_prefetch_rate == 0.0


def test_metrics_collector_saves_json(tmp_path) -> None:
    metrics = MetricsCollector()
    metrics.add(ssd_read_bytes=4096, io_op_count=1, selected_block_count=2)

    output = tmp_path / "metrics.json"
    metrics.save_json(output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ssd_read_bytes"] == 4096
    assert payload["io_op_count"] == 1
    assert payload["selected_block_count"] == 2
