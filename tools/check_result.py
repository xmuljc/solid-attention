"""Serializable results shared by local and CI verification profiles."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

CheckStatus = Literal["passed", "failed", "unavailable"]


@dataclass(frozen=True)
class TestCounts:
    tests: int
    failures: int
    errors: int
    skipped: int
    seconds: float


@dataclass(frozen=True)
class CommandResult:
    name: str
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_ms: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CheckResult:
    profile: str
    status: CheckStatus
    platform: str
    commands: tuple[CommandResult, ...]
    tests: TestCounts | None
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "solidattention.check_result.v1",
            "profile": self.profile,
            "status": self.status,
            "platform": self.platform,
            "commands": [command.to_dict() for command in self.commands],
            "tests": asdict(self.tests) if self.tests is not None else None,
            "message": self.message,
        }

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_junit(path: str | Path) -> TestCounts:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else [
        node
        for node in root.iter("testsuite")
        if not any(child.tag == "testsuite" for child in node)
    ]
    return TestCounts(
        tests=sum(int(node.attrib.get("tests", 0)) for node in suites),
        failures=sum(int(node.attrib.get("failures", 0)) for node in suites),
        errors=sum(int(node.attrib.get("errors", 0)) for node in suites),
        skipped=sum(int(node.attrib.get("skipped", 0)) for node in suites),
        seconds=sum(float(node.attrib.get("time", 0.0)) for node in suites),
    )


def bounded_output(text: str, limit: int = 8000) -> str:
    return text if len(text) <= limit else text[-limit:]
