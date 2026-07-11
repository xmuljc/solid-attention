"""Explicit local and CI verification profiles."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence
from xml.etree.ElementTree import ParseError

from tools.check_result import (
    CheckResult,
    CommandResult,
    TestCounts,
    bounded_output,
    parse_junit,
)
from tools.repository_contract import check_repository


PROFILE_NAMES = ("fast", "io", "liburing", "llama")

CommandSpec = tuple[str, list[str], dict[str, str]]


def require_passed_manifest(
    path: str | Path,
    *,
    expected_profile: str,
    required_output_kinds: set[str],
) -> None:
    """Require a passed run manifest with complete hashed outputs."""
    manifest_path = Path(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"could not read run manifest {manifest_path}: {error}"
        ) from error

    if not isinstance(manifest, dict):
        raise RuntimeError("run manifest must contain a JSON object")
    if manifest.get("schema") != "solidattention.run_manifest.v1":
        raise RuntimeError("run manifest schema mismatch")
    if manifest.get("status") != "passed":
        raise RuntimeError("manifest is not passed")
    if manifest.get("profile") != expected_profile:
        raise RuntimeError(
            "run manifest profile mismatch: "
            f"expected {expected_profile!r}, got {manifest.get('profile')!r}"
        )

    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise RuntimeError("run manifest outputs must be a list")

    output_kinds: set[str] = set()
    records: list[dict[str, object]] = []
    for index, record in enumerate(outputs):
        if not isinstance(record, dict):
            raise RuntimeError(
                f"run manifest output record {index} must be an object"
            )
        kind = record.get("kind")
        if not isinstance(kind, str) or not kind:
            raise RuntimeError(
                f"run manifest output record {index} has invalid kind"
            )
        output_kinds.add(kind)
        records.append(record)

    missing_kinds = required_output_kinds - output_kinds
    if missing_kinds:
        raise RuntimeError(
            "run manifest missing required output kinds: "
            + ", ".join(sorted(missing_kinds))
        )

    for index, record in enumerate(records):
        digest = record.get("sha256")
        if not isinstance(digest, str) or not digest:
            raise RuntimeError(
                f"run manifest output record {index} missing sha256"
            )


def profile_availability(profile: str, system: str) -> tuple[bool, str]:
    """Report whether a profile is owned by the current platform."""
    if profile not in PROFILE_NAMES:
        raise ValueError(f"unknown profile: {profile}")
    if profile == "fast" or system == "Linux":
        return True, ""
    return False, f"profile {profile} requires Linux; run it in GitHub Actions"


def build_profile_commands(profile: str, root: str | Path) -> list[CommandSpec]:
    """Create a fresh output directory and return a profile's commands."""
    root_path = Path(root).resolve()
    output = root_path / "outputs" / "checks" / profile
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    junit = output / "junit.xml"

    if profile == "fast":
        return [
            (
                "pytest",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-m",
                    "not io and not liburing and not llama",
                    "--junitxml",
                    str(junit),
                ],
                {},
            ),
            (
                "smoke",
                [
                    sys.executable,
                    "-m",
                    "harness.smoke",
                    "--config",
                    "configs/smoke_base.yaml",
                    "--output-dir",
                    str(output / "smoke"),
                ],
                {},
            ),
            (
                "ablation",
                [
                    sys.executable,
                    "-m",
                    "harness.ablation",
                    "--config",
                    "configs/smoke_base.yaml",
                    "--output-dir",
                    str(output / "ablation"),
                ],
                {},
            ),
        ]
    if profile in {"io", "liburing"}:
        backend_mode = "posix" if profile == "io" else "liburing"
        expected_backend = "posix_fallback" if profile == "io" else "liburing"
        return [
            (
                "make",
                [
                    "make",
                    "-C",
                    "io",
                    "clean",
                    "all",
                    f"BACKEND_MODE={backend_mode}",
                ],
                {},
            ),
            (
                "backend",
                [
                    "make",
                    "-s",
                    "-C",
                    "io",
                    "backend",
                    f"BACKEND_MODE={backend_mode}",
                ],
                {},
            ),
            (
                "pytest",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-m",
                    profile,
                    "--junitxml",
                    str(junit),
                ],
                {"SOLIDATTENTION_EXPECTED_IO_BACKEND": expected_backend},
            ),
        ]
    if profile == "llama":
        return [
            ("llama-smoke", [sys.executable, "-m", "tools.llama_check"], {}),
            (
                "pytest",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-m",
                    "llama",
                    "--junitxml",
                    str(junit),
                ],
                {},
            ),
        ]
    raise ValueError(f"unknown profile: {profile}")


def run_profile(
    profile: str,
    root: str | Path = Path.cwd(),
    *,
    system: str | None = None,
    runner=subprocess.run,
    which=shutil.which,
    repository_checker=check_repository,
) -> int:
    """Run one verification profile and persist its structured result."""
    if profile not in PROFILE_NAMES:
        raise ValueError(f"unknown profile: {profile}")

    root_path = Path(root).resolve()
    output = root_path / "outputs" / "checks" / profile
    result_path = output / "check_result.json"
    system_name = system if system is not None else platform.system()

    try:
        command_specs = build_profile_commands(profile, root_path)
    except Exception as error:
        return _failed(
            profile,
            system_name,
            (),
            None,
            f"could not prepare profile output: {error}",
            result_path,
        )

    available, availability_message = profile_availability(profile, system_name)
    if not available:
        return _finish(
            CheckResult(
                profile=profile,
                status="unavailable",
                platform=system_name,
                commands=(),
                tests=None,
                message=availability_message,
            ),
            result_path,
        )

    if profile == "fast":
        try:
            violations = repository_checker(root_path)
        except Exception as error:
            return _failed(
                profile,
                system_name,
                (),
                None,
                f"repository check failed: {error}",
                result_path,
            )
        if violations:
            return _finish(
                CheckResult(
                    profile=profile,
                    status="failed",
                    platform=system_name,
                    commands=(),
                    tests=None,
                    message="repository contract violations: " + "; ".join(violations),
                ),
                result_path,
            )
    else:
        required_tools = ("make", "cc") if profile in {"io", "liburing"} else ("git", "cmake")
        try:
            missing_tools = [tool for tool in required_tools if which(tool) is None]
        except Exception as error:
            return _failed(
                profile,
                system_name,
                (),
                None,
                f"prerequisite check failed: {error}",
                result_path,
            )
        if missing_tools:
            return _finish(
                CheckResult(
                    profile=profile,
                    status="failed",
                    platform=system_name,
                    commands=(),
                    tests=None,
                    message="missing required tools: " + ", ".join(missing_tools),
                ),
                result_path,
            )

    command_results: list[CommandResult] = []
    test_counts: TestCounts | None = None
    for name, argv, command_env in command_specs:
        merged_env = os.environ.copy()
        merged_env.update(command_env)
        started = time.monotonic()
        try:
            completed = runner(
                argv,
                cwd=root_path,
                capture_output=True,
                text=True,
                check=False,
                env=merged_env,
            )
        except Exception as error:
            return _failed(
                profile,
                system_name,
                command_results,
                test_counts,
                f"command {name} could not start: {error}",
                result_path,
            )
        duration_ms = int((time.monotonic() - started) * 1000)
        command_result = CommandResult(
            name=name,
            command=tuple(argv),
            returncode=completed.returncode,
            stdout=bounded_output(completed.stdout or ""),
            stderr=bounded_output(completed.stderr or ""),
            duration_ms=duration_ms,
        )
        command_results.append(command_result)

        if completed.returncode != 0:
            return _failed(
                profile,
                system_name,
                command_results,
                test_counts,
                f"command {name} failed with exit code {completed.returncode}",
                result_path,
            )

        if name == "backend":
            expected_backend = "posix_fallback" if profile == "io" else "liburing"
            actual_backend = (completed.stdout or "").strip()
            if actual_backend != expected_backend:
                return _failed(
                    profile,
                    system_name,
                    command_results,
                    test_counts,
                    f"backend reported {actual_backend!r}; expected {expected_backend}",
                    result_path,
                )

        if name == "pytest":
            try:
                test_counts = parse_junit(output / "junit.xml")
            except (OSError, ParseError, ValueError) as error:
                return _failed(
                    profile,
                    system_name,
                    command_results,
                    None,
                    f"could not parse pytest JUnit XML: {error}",
                    result_path,
                )
            if test_counts.failures or test_counts.errors or test_counts.skipped:
                return _failed(
                    profile,
                    system_name,
                    command_results,
                    test_counts,
                    "pytest reported "
                    f"failures={test_counts.failures}, errors={test_counts.errors}, "
                    f"skipped={test_counts.skipped}",
                    result_path,
                )

        if profile == "fast" and name in {"smoke", "ablation"}:
            try:
                require_passed_manifest(
                    output / name / "run_manifest.json",
                    expected_profile=name,
                    required_output_kinds={"trace", "metrics", "summary"},
                )
            except Exception as error:
                return _failed(
                    profile,
                    system_name,
                    command_results,
                    test_counts,
                    f"{name} run manifest validation failed: {error}",
                    result_path,
                )

    return _finish(
        CheckResult(
            profile=profile,
            status="passed",
            platform=system_name,
            commands=tuple(command_results),
            tests=test_counts,
            message="",
        ),
        result_path,
    )


def _failed(
    profile: str,
    system: str,
    commands: Sequence[CommandResult],
    tests: TestCounts | None,
    message: str,
    result_path: Path,
) -> int:
    return _finish(
        CheckResult(
            profile=profile,
            status="failed",
            platform=system,
            commands=tuple(commands),
            tests=tests,
            message=message,
        ),
        result_path,
    )


def _finish(result: CheckResult, path: Path) -> int:
    saved = True
    try:
        result.save(path)
    except Exception as error:
        saved = False
        print(f"could not save check result to {path}: {error}", file=sys.stderr)
    tests = result.tests or TestCounts(0, 0, 0, 0, 0.0)
    print(
        f"profile={result.profile} status={result.status} tests={tests.tests} "
        f"failures={tests.failures} errors={tests.errors} skipped={tests.skipped}"
    )
    if not saved:
        return 1
    return {"passed": 0, "failed": 1, "unavailable": 2}[result.status]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a SolidAttention verification profile.")
    parser.add_argument("profile", choices=PROFILE_NAMES)
    args = parser.parse_args(argv)
    return run_profile(args.profile)


if __name__ == "__main__":
    raise SystemExit(main())
