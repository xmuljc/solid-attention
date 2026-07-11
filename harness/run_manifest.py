"""Reproducible run metadata and artifact hashing."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path


_ALLOWED_IO_BACKENDS = frozenset(
    {"mock", "fake", "posix_fallback", "liburing"}
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata(cwd: Path) -> dict[str, str | bool | None]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "dirty": None}
    return {"sha": sha, "dirty": dirty}


class RunContext:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        command: tuple[str, ...],
        profile: str,
        config_path: str | Path | None = None,
        config_snapshot: dict[str, object] | None = None,
        inputs: tuple[str | Path, ...] = (),
        io_backend: str = "mock",
        run_id: str | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.command = tuple(command)
        self.profile = profile
        self.config_path = Path(config_path) if config_path is not None else None
        self.config_snapshot = config_snapshot
        self.inputs = tuple(Path(path) for path in inputs)
        self._validate_io_backend(io_backend)
        self.io_backend = io_backend
        self.run_id = run_id
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.manifest_path = self.output_dir / "run_manifest.json"
        self.outputs: list[dict[str, str]] = []
        self.started_at: datetime | None = None
        self.git: dict[str, str | bool | None] = {"sha": None, "dirty": None}
        self.runtime: dict[str, str] = {}
        self.forced_failure: dict[str, str] | None = None
        self._config_hash: str | None = None
        self._input_records: list[dict[str, str]] = []

    def __enter__(self) -> RunContext:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = self.clock()
        if self.run_id is None:
            timestamp = self.started_at.astimezone(timezone.utc).strftime(
                "%Y%m%dT%H%M%SZ"
            )
            self.run_id = f"{timestamp}-{uuid.uuid4().hex[:8]}"

        try:
            self.git = git_metadata(Path.cwd())
            self.runtime = {
                "python": sys.version.split()[0],
                "os": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            }
            if self.config_path is not None:
                self._config_hash = sha256_file(self.config_path)
            self._input_records = [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in self.inputs
            ]
        except Exception as exc:
            try:
                self._finalize(
                    status="failed",
                    failure={"type": type(exc).__name__, "message": str(exc)},
                    ended_at=self.clock(),
                )
            except Exception as error:
                self._add_manifest_error_note(exc, error)
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> bool:
        if exc_type is not None:
            failure = {
                "type": exc_type.__name__,
                "message": str(exc_value),
            }
        else:
            failure = self.forced_failure
        try:
            self._finalize(
                status="failed" if failure is not None else "passed",
                failure=failure,
                ended_at=self.clock(),
            )
        except Exception as error:
            if exc_value is None:
                raise
            self._add_manifest_error_note(exc_value, error)
        return False

    def add_output(self, kind: str, path: str | Path) -> None:
        output_path = Path(path)
        digest = sha256_file(output_path)
        try:
            manifest_path = output_path.relative_to(self.output_dir).as_posix()
        except ValueError:
            manifest_path = str(output_path)
        self.outputs.append(
            {"kind": kind, "path": manifest_path, "sha256": digest}
        )

    def set_config_snapshot(self, snapshot: object) -> None:
        self.config_snapshot = snapshot

    def set_io_backend(self, io_backend: str) -> None:
        self._validate_io_backend(io_backend)
        self.io_backend = io_backend

    def mark_failed(self, failure_type: str, message: str) -> None:
        self.forced_failure = {"type": failure_type, "message": message}

    @staticmethod
    def _validate_io_backend(io_backend: str) -> None:
        if io_backend not in _ALLOWED_IO_BACKENDS:
            allowed = ", ".join(sorted(_ALLOWED_IO_BACKENDS))
            raise ValueError(
                f"unsupported io_backend {io_backend!r}; expected one of: {allowed}"
            )

    @staticmethod
    def _add_manifest_error_note(
        exception: BaseException, error: Exception
    ) -> None:
        exception.add_note(
            "could not write run manifest: "
            f"{type(error).__name__}: {error}"
        )

    def _finalize(
        self,
        *,
        status: str,
        failure: dict[str, str] | None,
        ended_at: datetime,
    ) -> None:
        started_at = self.started_at
        duration_ms = (
            int((ended_at - started_at).total_seconds() * 1000)
            if started_at is not None
            else 0
        )
        config = None
        if self.config_path is not None and self._config_hash is not None:
            config = {
                "path": str(self.config_path),
                "sha256": self._config_hash,
                "snapshot": self.config_snapshot,
            }
        payload = {
            "schema": "solidattention.run_manifest.v1",
            "run_id": self.run_id,
            "profile": self.profile,
            "command": list(self.command),
            "started_at": started_at.isoformat() if started_at is not None else None,
            "ended_at": ended_at.isoformat(),
            "duration_ms": duration_ms,
            "status": status,
            "failure": failure,
            "git": self.git,
            "runtime": self.runtime,
            "io_backend": self.io_backend,
            "config": config,
            "inputs": self._input_records,
            "outputs": self.outputs,
        }
        self.manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
