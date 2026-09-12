"""Atomic artifacts, run identity, active-time accounting, and exclusive run locks."""

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import re
import time
import uuid

from .config import digest, from_dict


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_manifest():
    root = Path(__file__).parent
    files = {str(path.relative_to(root)): file_hash(path) for path in sorted(root.rglob("*.py"))}
    return {"files": files, "fingerprint": digest(files)}


class Run:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.config = from_dict(read_json(self.path / "config.json"))
        self.state = read_json(self.path / "state.json")
        self._started = None

    @classmethod
    def create(cls, config, run_id=None):
        run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", run_id):
            raise ValueError("Run ID must contain only letters, numbers, '.', '_' or '-'")
        path = Path(config.runtime.run_root).expanduser().resolve() / run_id
        path.mkdir(parents=True, exist_ok=False)
        atomic_json(path / "config.json", config.to_dict())
        atomic_json(path / "source_manifest.json", source_manifest())
        atomic_json(path / "state.json", {"config_fingerprint": config.fingerprint,
                    "completed": [], "spent_seconds": 0.0, "created_utc": datetime.now(timezone.utc).isoformat()})
        return cls(path)

    def start_clock(self):
        if self._started is None:
            self._started = time.monotonic()

    @property
    def elapsed(self):
        return self.state["spent_seconds"] + (time.monotonic() - self._started if self._started else 0)

    @property
    def remaining(self):
        return max(0.0, self.config.runtime.budget_seconds + self.state.get("extra_seconds", 0) - self.elapsed)

    def heartbeat(self):
        self.state["spent_seconds"] = self.elapsed
        if self._started is not None:
            self._started = time.monotonic()
        atomic_json(self.path / "state.json", self.state)

    def complete(self, stage):
        if stage not in self.state["completed"]:
            self.state["completed"].append(stage)
        self.heartbeat()

    def log(self, event, **values):
        record = {"event": event, "elapsed_seconds": round(self.elapsed, 3), **values}
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)


@contextmanager
def locked_run(run):
    """Kernel-managed lock, automatically released after a crashed WSL process."""
    if os.name != "posix":
        raise RuntimeError("Run pipeline stages from WSL; Windows supports inspection and configuration")
    import fcntl
    with (run.path / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process owns this run: {run.path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
