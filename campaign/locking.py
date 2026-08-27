"""Advisory campaign locks with auditable owner/process-start identity."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, IO


class LockError(RuntimeError):
    """Raised when a campaign lock is active or its recovery is unsafe."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def process_start_identity(pid: int) -> str | None:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart="], text=True, capture_output=True
    )
    value = result.stdout.strip()
    return value or None


def read_metadata(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text().strip()
    except FileNotFoundError:
        return None
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LockError(f"invalid lock metadata: {path}") from exc


def audit_lock(path: Path) -> dict[str, Any]:
    """Read-only stale audit; never removes or overwrites lock metadata."""
    metadata = read_metadata(path)
    if not path.exists():
        return {"path": str(path), "state": "absent", "metadata": metadata}
    with path.open("r") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            locked = True
        else:
            locked = False
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    process_matches = False
    if metadata and metadata.get("pid") and metadata.get("process_start_identity"):
        process_matches = (
            metadata.get("hostname") == platform.node()
            and process_start_identity(int(metadata["pid"])) == metadata["process_start_identity"]
        )
    state = "active" if locked else ("stale_metadata" if metadata and metadata.get("state") == "held" else "available")
    return {"path": str(path), "state": state, "advisory_lock_held": locked,
            "metadata_process_matches": process_matches, "metadata": metadata}


class CampaignLock:
    def __init__(self, path: Path, run_id: str):
        self.path = path
        self.run_id = run_id
        self.handle: IO[str] | None = None

    def __enter__(self) -> "CampaignLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close(); self.handle = None
            raise LockError(f"campaign lock held: {json.dumps(audit_lock(self.path), sort_keys=True)}") from exc
        metadata = {
            "schema_version": 1, "state": "held", "hostname": platform.node(),
            "pid": os.getpid(), "process_start_identity": process_start_identity(os.getpid()),
            "run_id": self.run_id, "acquired_at_utc": utc_now(),
        }
        self._write(metadata)
        return self

    def _write(self, value: dict[str, Any]) -> None:
        assert self.handle is not None
        self.handle.seek(0); self.handle.truncate()
        self.handle.write(json.dumps(value, sort_keys=True) + "\n")
        self.handle.flush(); os.fsync(self.handle.fileno())

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is None:
            return
        try:
            self._write({
                "schema_version": 1, "state": "released", "hostname": platform.node(),
                "pid": os.getpid(), "process_start_identity": process_start_identity(os.getpid()),
                "run_id": self.run_id, "released_at_utc": utc_now(),
                "exception": None if exc is None else f"{type(exc).__name__}: {exc}",
            })
        finally:
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            finally:
                self.handle.close(); self.handle = None


def recover_lock(path: Path, operator_note: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LockError("refusing recovery: advisory lock is active") from exc
        prior = read_metadata(path)
        if prior and prior.get("state") == "held" and prior.get("hostname") == platform.node():
            current_start = process_start_identity(int(prior.get("pid", -1)))
            if current_start and current_start == prior.get("process_start_identity"):
                raise LockError("refusing recovery: recorded owner process/start identity is alive")
        value = {"schema_version": 1, "state": "recovered", "recovered_at_utc": utc_now(),
                 "hostname": platform.node(), "operator_note": operator_note, "prior": prior}
        handle.seek(0); handle.truncate(); handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno()); fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit"); audit.add_argument("path", type=Path)
    recover = sub.add_parser("recover"); recover.add_argument("path", type=Path); recover.add_argument("--operator-note", required=True)
    args = parser.parse_args()
    value = audit_lock(args.path) if args.command == "audit" else recover_lock(args.path, args.operator_note)
    print(json.dumps(value, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
