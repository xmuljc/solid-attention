import json
from pathlib import Path

from tools.check_result import CheckResult, CommandResult, TestCounts, parse_junit


def test_parse_junit_sums_nested_suites(tmp_path: Path) -> None:
    path = tmp_path / "junit.xml"
    path.write_text(
        '<testsuites name="pytest tests">'
        '<testsuite tests="2" failures="1" errors="0" skipped="0" time="0.3" />'
        '<testsuite tests="1" failures="0" errors="0" skipped="1" time="0.2" />'
        '</testsuites>',
        encoding="utf-8",
    )

    assert parse_junit(path) == TestCounts(tests=3, failures=1, errors=0, skipped=1, seconds=0.5)


def test_check_result_writes_bounded_json(tmp_path: Path) -> None:
    result = CheckResult(
        profile="fast",
        status="passed",
        platform="Windows",
        commands=(CommandResult("pytest", ("python", "-m", "pytest"), 0, "ok", "", 12),),
        tests=TestCounts(3, 0, 0, 0, 0.01),
        message="",
    )

    output = tmp_path / "check_result.json"
    result.save(output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema"] == "solidattention.check_result.v1"
    assert payload["profile"] == "fast"
    assert payload["tests"]["skipped"] == 0
