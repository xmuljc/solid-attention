import json
import subprocess
from pathlib import Path

import pytest

import tools.check as check_module
from tools.check import (
    build_profile_commands,
    profile_availability,
    require_passed_manifest,
    run_profile,
)


def test_fast_profile_is_available_on_windows() -> None:
    assert profile_availability("fast", "Windows") == (True, "")


def test_liburing_profile_is_unavailable_on_windows() -> None:
    available, message = profile_availability("liburing", "Windows")

    assert available is False
    assert "requires Linux" in message


def test_fast_profile_builds_pytest_smoke_and_ablation_commands(tmp_path: Path) -> None:
    commands = build_profile_commands("fast", tmp_path)

    assert [name for name, _, _ in commands] == ["pytest", "smoke", "ablation"]
    pytest_argv = commands[0][1]
    assert "not io and not liburing and not llama" in pytest_argv
    assert "--junitxml" in pytest_argv


def test_liburing_profile_pins_backend_for_build_and_tests(tmp_path: Path) -> None:
    commands = build_profile_commands("liburing", tmp_path)

    for _, argv, _ in commands[:2]:
        assert "BACKEND_MODE=liburing" in argv
    assert commands[2][2] == {"SOLIDATTENTION_EXPECTED_IO_BACKEND": "liburing"}


def test_build_profile_commands_removes_stale_profile_output(tmp_path: Path) -> None:
    stale = tmp_path / "outputs" / "checks" / "fast" / "smoke" / "run_manifest.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale", encoding="utf-8")

    build_profile_commands("fast", tmp_path)

    assert not stale.exists()


def test_require_passed_manifest_accepts_valid_outputs(tmp_path: Path) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "solidattention.run_manifest.v1",
                "profile": "smoke",
                "status": "passed",
                "outputs": [
                    {"kind": kind, "sha256": character * 64}
                    for kind, character in (
                        ("trace", "a"),
                        ("metrics", "b"),
                        ("summary", "c"),
                    )
                ],
            }
        ),
        encoding="utf-8",
    )

    require_passed_manifest(
        manifest_path,
        expected_profile="smoke",
        required_output_kinds={"trace", "metrics", "summary"},
    )


def test_require_passed_manifest_rejects_failed_run(tmp_path: Path) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "solidattention.run_manifest.v1",
                "profile": "smoke",
                "status": "failed",
                "outputs": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="manifest is not passed"):
        require_passed_manifest(
            manifest_path,
            expected_profile="smoke",
            required_output_kinds={"trace", "metrics", "summary"},
        )


def test_require_passed_manifest_checks_profile_before_output_hashes(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "solidattention.run_manifest.v1",
                "profile": "ablation",
                "status": "passed",
                "outputs": [
                    {"kind": "trace", "sha256": "a" * 64},
                    {"kind": "metrics", "sha256": "b" * 64},
                    {"kind": "summary", "sha256": ""},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="profile mismatch"):
        require_passed_manifest(
            manifest_path,
            expected_profile="smoke",
            required_output_kinds={"trace", "metrics", "summary"},
        )
    with pytest.raises(RuntimeError, match="missing sha256"):
        require_passed_manifest(
            manifest_path,
            expected_profile="ablation",
            required_output_kinds={"trace", "metrics", "summary"},
        )


def test_run_profile_records_first_command_failure(tmp_path: Path) -> None:
    def runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 7, stdout="", stderr="boom")

    exit_code = run_profile(
        "fast",
        tmp_path,
        system="Windows",
        runner=runner,
        repository_checker=lambda root: [],
    )

    payload = json.loads(
        (tmp_path / "outputs" / "checks" / "fast" / "check_result.json").read_text(
            encoding="utf-8"
        )
    )
    assert exit_code == 1
    assert payload["status"] == "failed"
    assert payload["commands"][0]["returncode"] == 7


def test_run_profile_fails_when_pytest_skips_tests(tmp_path: Path) -> None:
    def runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "--junitxml" in argv:
            junit = Path(argv[argv.index("--junitxml") + 1])
            junit.write_text(
                '<testsuite tests="1" failures="0" errors="0" skipped="1" time="0.1" />',
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    exit_code = run_profile(
        "fast",
        tmp_path,
        system="Windows",
        runner=runner,
        repository_checker=lambda root: [],
    )

    payload = json.loads(
        (tmp_path / "outputs" / "checks" / "fast" / "check_result.json").read_text(
            encoding="utf-8"
        )
    )
    assert exit_code == 1
    assert payload["status"] == "failed"
    assert payload["tests"]["skipped"] == 1


def test_run_profile_rejects_unexpected_io_backend(tmp_path: Path) -> None:
    def runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        stdout = "liburing\n" if "backend" in argv else ""
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    exit_code = run_profile(
        "io",
        tmp_path,
        system="Linux",
        runner=runner,
        which=lambda name: f"/usr/bin/{name}",
    )

    payload = json.loads(
        (tmp_path / "outputs" / "checks" / "io" / "check_result.json").read_text(
            encoding="utf-8"
        )
    )
    assert exit_code == 1
    assert payload["status"] == "failed"
    assert "expected posix_fallback" in payload["message"]


def test_run_profile_resolves_relative_root_for_cwd_and_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    calls: list[tuple[list[str], Path]] = []

    def runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        cwd = Path(kwargs["cwd"])
        calls.append((argv, cwd))
        if "--junitxml" in argv:
            junit = Path(argv[argv.index("--junitxml") + 1])
            effective_junit = junit if junit.is_absolute() else cwd / junit
            effective_junit.parent.mkdir(parents=True, exist_ok=True)
            effective_junit.write_text(
                '<testsuite tests="1" failures="0" errors="0" skipped="0" time="0.1" />',
                encoding="utf-8",
            )
        if "--output-dir" in argv:
            profile = "smoke" if "harness.smoke" in argv else "ablation"
            manifest_dir = Path(argv[argv.index("--output-dir") + 1])
            manifest_dir.mkdir(parents=True, exist_ok=True)
            (manifest_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "solidattention.run_manifest.v1",
                        "profile": profile,
                        "status": "passed",
                        "outputs": [
                            {"kind": kind, "sha256": "a" * 64}
                            for kind in ("trace", "metrics", "summary")
                        ],
                    }
                ),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    exit_code = run_profile(
        "fast",
        Path("checkout"),
        system="Windows",
        runner=runner,
        repository_checker=lambda root: [],
    )

    resolved_checkout = checkout.resolve()
    pytest_argv = calls[0][0]
    junit = Path(pytest_argv[pytest_argv.index("--junitxml") + 1])
    assert exit_code == 0
    assert all(cwd == resolved_checkout for _, cwd in calls)
    assert junit == resolved_checkout / "outputs" / "checks" / "fast" / "junit.xml"
    assert (resolved_checkout / "outputs" / "checks" / "fast" / "check_result.json").exists()


def test_run_profile_records_runner_start_failure(tmp_path: Path) -> None:
    def runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError("spawn failed")

    exit_code = run_profile(
        "fast",
        tmp_path,
        system="Windows",
        runner=runner,
        repository_checker=lambda root: [],
    )

    payload = json.loads(
        (tmp_path / "outputs" / "checks" / "fast" / "check_result.json").read_text(
            encoding="utf-8"
        )
    )
    assert exit_code == 1
    assert payload["status"] == "failed"
    assert payload["commands"] == []
    assert "spawn failed" in payload["message"]


def test_run_profile_records_repository_check_failure(tmp_path: Path) -> None:
    def repository_checker(root: Path) -> list[str]:
        raise RuntimeError("git index unavailable")

    exit_code = run_profile(
        "fast",
        tmp_path,
        system="Windows",
        runner=lambda argv, **kwargs: None,
        repository_checker=repository_checker,
    )

    payload = json.loads(
        (tmp_path / "outputs" / "checks" / "fast" / "check_result.json").read_text(
            encoding="utf-8"
        )
    )
    assert exit_code == 1
    assert payload["status"] == "failed"
    assert payload["commands"] == []
    assert "repository check" in payload["message"]
    assert "git index unavailable" in payload["message"]


def test_run_profile_records_prerequisite_check_failure(tmp_path: Path) -> None:
    def which(name: str) -> str | None:
        raise OSError("PATH unavailable")

    exit_code = run_profile(
        "io",
        tmp_path,
        system="Linux",
        runner=lambda argv, **kwargs: None,
        which=which,
    )

    payload = json.loads(
        (tmp_path / "outputs" / "checks" / "io" / "check_result.json").read_text(
            encoding="utf-8"
        )
    )
    assert exit_code == 1
    assert payload["status"] == "failed"
    assert payload["commands"] == []
    assert "prerequisite check" in payload["message"]
    assert "PATH unavailable" in payload["message"]


def test_run_profile_records_output_cleanup_failure(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "outputs" / "checks" / "fast"
    output.mkdir(parents=True)

    def rmtree(path: Path) -> None:
        raise PermissionError("cleanup denied")

    monkeypatch.setattr(check_module.shutil, "rmtree", rmtree)

    exit_code = run_profile(
        "fast",
        tmp_path,
        system="Windows",
        repository_checker=lambda root: [],
    )

    payload = json.loads((output / "check_result.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert payload["status"] == "failed"
    assert payload["commands"] == []
    assert "could not prepare profile output" in payload["message"]
    assert "cleanup denied" in payload["message"]


def test_run_profile_returns_failure_when_result_cannot_be_saved(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    def save(result: object, path: Path) -> None:
        raise PermissionError("artifact denied")

    monkeypatch.setattr(check_module.CheckResult, "save", save)

    exit_code = run_profile("liburing", tmp_path, system="Windows")
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "could not save check result" in captured.err
    assert "artifact denied" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == (
        "profile=liburing status=unavailable tests=0 failures=0 errors=0 skipped=0\n"
    )
