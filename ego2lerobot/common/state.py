"""Idempotent task output handling shared by conversion workers.

Layout used by a task::

    <output_path>.partial/   # written first, never read by aggregation
    <output_path>/           # committed atomically with os.replace
    <output_path>/_SUCCESS   # written after finalize
    <output_path>.failed.json

A worker must call :func:`is_task_done` first, then :func:`prepare_output`,
write into :func:`partial_path`, and finally :func:`commit_output`.  If the
worker crashes, the next attempt removes the stale partial/final directory and
starts clean.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import time
from pathlib import Path
from typing import Any

from .types import ConversionTask

SUCCESS_MARKER = "_SUCCESS"


def success_marker(output_path: str | Path) -> Path:
    return Path(output_path) / SUCCESS_MARKER


def is_task_done(task: ConversionTask) -> bool:
    return success_marker(task.output_path).is_file()


def partial_path(output_path: str | Path) -> Path:
    return Path(str(output_path) + ".partial")


def failure_path(output_path: str | Path) -> Path:
    return Path(str(output_path) + ".failed.json")


def safe_name(value: str, max_len: int = 120) -> str:
    """Turn a task id into a filesystem-safe, collision-resistant name."""

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    if not cleaned:
        cleaned = "task"
    if len(cleaned) <= max_len:
        return cleaned
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[:max_len]}-{digest}"


def prepare_output(task: ConversionTask) -> None:
    """Remove stale partial output and any non-committed final output."""

    partial = partial_path(task.output_path)
    shutil.rmtree(partial, ignore_errors=True)

    # A final directory without _SUCCESS is the remnant of a crashed run.
    if task.output_path.exists() and not is_task_done(task):
        shutil.rmtree(task.output_path, ignore_errors=True)

    failure_path(task.output_path).unlink(missing_ok=True)


def commit_output(task: ConversionTask) -> None:
    """Atomically move ``<output>.partial`` to ``<output>`` and mark success."""

    final = task.output_path
    partial = partial_path(final)

    if not partial.is_dir():
        raise FileNotFoundError(f"partial output does not exist: {partial}")

    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        if is_task_done(task):
            raise FileExistsError(f"task already committed: {final}")
        shutil.rmtree(final)

    os.replace(partial, final)

    marker = success_marker(final)
    marker.write_text(
        json.dumps(
            {
                "task_id": task.task_id,
                "committed_at": time.time(),
                "host": socket.gethostname(),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_failure(task: ConversionTask, error: str) -> Path:
    """Persist a per-task failure next to the output, atomically."""

    path = failure_path(task.output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "task_id": task.task_id,
        "input_path": str(task.input_path),
        "error": error,
        "failed_at": time.time(),
        "host": socket.gethostname(),
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def write_task_log(
    log_dir: str | Path | None,
    task: ConversionTask,
    status: str,
    *,
    frames: int = 0,
    episodes: int = 0,
    error: str | None = None,
) -> Path | None:
    """Write one JSON line per task under ``log_dir`` (best effort)."""

    if not log_dir:
        return None
    path = Path(log_dir) / f"{safe_name(task.task_id)}.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "task_id": task.task_id,
            "input_path": str(task.input_path),
            "output_path": str(task.output_path),
            "status": status,
            "frames": frames,
            "episodes": episodes,
            "host": socket.gethostname(),
            "error": error,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path
    except OSError:
        # Logging must never turn a successful conversion into a failure.
        return None
