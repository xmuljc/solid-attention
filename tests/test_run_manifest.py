import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from harness.run_manifest import RunContext


def fixed_clock():
    moments = iter(
        [
            datetime(2026, 7, 11, 8, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 7, 11, 8, 0, 1, tzinfo=timezone.utc),
        ]
    )
    return lambda: next(moments)


def assert_manifest_write_note(error: BaseException) -> None:
    notes = getattr(error, "__notes__", ())
    assert any(
        "could not write run manifest" in note
        and ("IsADirectoryError" in note or "PermissionError" in note)
        for note in notes
    )


def test_run_context_records_reproducible_success_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SECRET_TOKEN", "manifest-must-not-contain-this")
    config_path = tmp_path / "config.json"
    config_path.write_text('{"blocks": 4}\n', encoding="utf-8")
    source_path = tmp_path / "source.jsonl"
    source_path.write_text('{"token": 1}\n', encoding="utf-8")
    output_dir = tmp_path / "run"

    with RunContext(
        output_dir,
        command=("python", "-m", "harness.smoke"),
        profile="smoke",
        config_path=config_path,
        config_snapshot={"blocks": 4},
        inputs=(source_path,),
        io_backend="mock",
        run_id="run-fixed",
        clock=fixed_clock(),
    ) as run:
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text("{}\n", encoding="utf-8")
        run.add_output("metrics", metrics_path)

    serialized = run.manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(serialized)

    assert set(manifest) == {
        "schema",
        "run_id",
        "profile",
        "command",
        "started_at",
        "ended_at",
        "duration_ms",
        "status",
        "failure",
        "git",
        "runtime",
        "io_backend",
        "config",
        "inputs",
        "outputs",
    }
    assert manifest["schema"] == "solidattention.run_manifest.v1"
    assert manifest["run_id"] == "run-fixed"
    assert manifest["profile"] == "smoke"
    assert manifest["command"] == ["python", "-m", "harness.smoke"]
    assert manifest["started_at"] == "2026-07-11T08:00:00+00:00"
    assert manifest["ended_at"] == "2026-07-11T08:00:01+00:00"
    assert manifest["duration_ms"] == 1000
    assert manifest["status"] == "passed"
    assert manifest["failure"] is None
    assert set(manifest["runtime"]) == {"python", "os", "release", "machine"}
    assert set(manifest["git"]) == {"sha", "dirty"}
    assert manifest["io_backend"] == "mock"
    assert manifest["config"]["path"] == str(config_path)
    assert manifest["config"]["snapshot"] == {"blocks": 4}
    assert re.fullmatch(r"[0-9a-f]{64}", manifest["config"]["sha256"])
    assert manifest["inputs"][0]["path"] == str(source_path)
    assert re.fullmatch(r"[0-9a-f]{64}", manifest["inputs"][0]["sha256"])
    assert manifest["outputs"][0]["kind"] == "metrics"
    assert manifest["outputs"][0]["path"] == "metrics.json"
    assert re.fullmatch(r"[0-9a-f]{64}", manifest["outputs"][0]["sha256"])
    assert "SECRET_TOKEN" not in serialized
    assert "manifest-must-not-contain-this" not in serialized


def test_run_context_records_exception_failure(tmp_path: Path) -> None:
    run = RunContext(
        tmp_path / "failed",
        command=("python", "-m", "harness.smoke"),
        profile="failed",
        run_id="run-failed",
        clock=fixed_clock(),
    )

    with pytest.raises(ValueError, match="bad config"):
        with run:
            raise ValueError("bad config")

    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failure"] == {"type": "ValueError", "message": "bad config"}


def test_run_context_records_nonexception_verification_failure(tmp_path: Path) -> None:
    run = RunContext(
        tmp_path / "verification-failed",
        command=("python", "-m", "harness.smoke"),
        profile="failed",
        run_id="run-verification-failed",
        clock=fixed_clock(),
    )

    with run:
        run.mark_failed(
            "VerificationError", "runtime residency verification errors: 2"
        )

    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failure"] == {
        "type": "VerificationError",
        "message": "runtime residency verification errors: 2",
    }


def test_run_context_records_missing_input_enter_failure(tmp_path: Path) -> None:
    run = RunContext(
        tmp_path / "missing-input",
        command=("python", "-m", "harness.smoke"),
        profile="failed",
        inputs=(tmp_path / "missing.jsonl",),
        run_id="run-missing-input",
        clock=fixed_clock(),
    )

    with pytest.raises(FileNotFoundError):
        with run:
            pass

    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failure"]["type"] == "FileNotFoundError"


def test_run_context_generates_timestamped_run_id(tmp_path: Path) -> None:
    run = RunContext(
        tmp_path / "generated-id",
        command=("python", "-m", "harness.smoke"),
        profile="smoke",
        clock=fixed_clock(),
    )

    with run:
        pass

    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert re.fullmatch(r"20260711T080000Z-[0-9a-f]{8}", manifest["run_id"])


def test_run_context_preserves_enter_error_when_manifest_write_fails(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "enter-error"
    (output_dir / "run_manifest.json").mkdir(parents=True)
    run = RunContext(
        output_dir,
        command=("python", "-m", "harness.smoke"),
        profile="failed",
        inputs=(tmp_path / "missing.jsonl",),
        run_id="run-enter-error",
        clock=fixed_clock(),
    )

    with pytest.raises(FileNotFoundError) as caught:
        with run:
            pass

    assert_manifest_write_note(caught.value)


def test_run_context_preserves_body_error_when_manifest_write_fails(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "body-error"
    (output_dir / "run_manifest.json").mkdir(parents=True)
    run = RunContext(
        output_dir,
        command=("python", "-m", "harness.smoke"),
        profile="failed",
        run_id="run-body-error",
        clock=fixed_clock(),
    )

    with pytest.raises(ValueError, match="bad config") as caught:
        with run:
            raise ValueError("bad config")

    assert_manifest_write_note(caught.value)
