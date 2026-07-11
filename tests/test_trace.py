import json

import pytest

from harness.trace import TraceRecorder


def test_trace_recorder_records_required_fields() -> None:
    recorder = TraceRecorder()

    event = recorder.record(
        op="ssd_read",
        layer_id=1,
        block_id=2,
        device="ssd",
        start_ms=10.0,
        end_ms=12.5,
        metadata={"queue_depth": 4},
    )

    assert event.duration_ms == 2.5
    assert recorder.to_list() == [
        {
            "op": "ssd_read",
            "layer_id": 1,
            "block_id": 2,
            "device": "ssd",
            "start_ms": 10.0,
            "end_ms": 12.5,
            "duration_ms": 2.5,
            "metadata": {"queue_depth": 4},
        }
    ]


def test_trace_recorder_rejects_negative_duration() -> None:
    recorder = TraceRecorder()

    with pytest.raises(ValueError):
        recorder.record("bad", 0, 0, "ssd", 2.0, 1.0)


def test_trace_recorder_saves_json(tmp_path) -> None:
    recorder = TraceRecorder()
    recorder.record("move", 0, 1, "dram", 0.0, 1.0, {"target": "vram"})

    output = tmp_path / "trace.json"
    recorder.save_json(output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload[0]["op"] == "move"
    assert payload[0]["duration_ms"] == 1.0
    assert payload[0]["metadata"] == {"target": "vram"}
