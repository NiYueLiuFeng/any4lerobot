"""Idempotent task output handling shared by conversion workers.

Layout used by a task::

    <output_path>.partial/   # written first, never read by aggregation
    <output_path>/           # committed with os.replace where supported
    <output_path>/_SUCCESS   # written after finalize
    <output_path>.failed.json

A worker must call :func:`is_task_done` first, then :func:`prepare_output`,
write into :func:`staging_path`, and finally :func:`commit_output`.  If the
worker crashes, the next attempt removes the stale partial/final directory and
starts clean.

Some network file systems reject ``rename(2)`` with ``EPERM`` even for files
the caller just created (Huawei SFS Turbo exports do this, for example).  There
:func:`staging_path` returns the final directory itself and ``_SUCCESS`` is the
only commit signal; the data is never copied a second time.
"""

from __future__ import annotations

import errno
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

# Errors a file system uses to report that rename is unsupported.
_RENAME_ERRNOS = frozenset(
    {errno.EPERM, errno.EXDEV, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS}
)
# Probe result per parent directory, so a run pays for it once per worker.
_RENAME_SUPPORT: dict[str, bool] = {}


def success_marker(output_path: str | Path) -> Path:
    return Path(output_path) / SUCCESS_MARKER


def is_task_done(task: ConversionTask) -> bool:
    return success_marker(task.output_path).is_file()


def partial_path(output_path: str | Path) -> Path:
    return Path(str(output_path) + ".partial")


def failure_path(output_path: str | Path) -> Path:
    return Path(str(output_path) + ".failed.json")


def rename_supported(directory: str | Path) -> bool:
    """Whether ``rename(2)`` works inside ``directory``.

    Probed once per directory and cached for the life of the process.  The
    probe uses the same operation the commit path needs, so the answer is
    reliable even on exports with unusual permission mapping.
    """

    directory = Path(directory)
    key = str(directory)
    cached = _RENAME_SUPPORT.get(key)
    if cached is not None:
        return cached

    token = f"{os.getpid()}-{time.time_ns()}"
    probe = directory / f".rename_probe-{token}"
    moved = directory / f".rename_probe-{token}.done"
    supported = False
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe.write_text("", encoding="utf-8")
        try:
            os.replace(probe, moved)
            supported = True
        except OSError:
            supported = False
    except OSError:
        supported = False
    finally:
        for path in (probe, moved):
            try:
                path.unlink()
            except OSError:
                pass

    _RENAME_SUPPORT[key] = supported
    return supported


def staging_directory(output_path: str | Path) -> Path:
    """Directory to build an output in before it is committed.

    Returns ``<output>.partial`` when the parent supports rename (the usual
    atomic path), and ``<output>`` itself when it does not.  In the second
    case callers treat ``_SUCCESS`` as the only completion marker, so no extra
    copy of a multi-gigabyte dataset is needed.
    """

    final = Path(output_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    return partial_path(final) if rename_supported(final.parent) else final


def staging_path(task: ConversionTask) -> Path:
    """Directory a worker must write the dataset into (see staging_directory)."""

    return staging_directory(task.output_path)


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

    failure = failure_path(task.output_path)
    failure.unlink(missing_ok=True)
    # A crash while writing the failure record leaves its temp copy behind.
    failure.with_name(failure.name + ".tmp").unlink(missing_ok=True)


def commit_output(task: ConversionTask) -> None:
    """Move the staged output into place and write ``_SUCCESS``.

    The move is a rename when possible.  On file systems without rename the
    worker already wrote into the final directory (see :func:`staging_path`),
    so only the success marker is added.
    """

    final = Path(task.output_path)
    partial = partial_path(final)
    final.parent.mkdir(parents=True, exist_ok=True)

    if partial.is_dir():
        if final.exists() and is_task_done(task):
            raise FileExistsError(f"task already committed: {final}")
        move_into_place(partial, final)
    elif not final.is_dir():
        raise FileNotFoundError(f"no staged output: {partial} or {final}")

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


def move_into_place(src: str | Path, dst: str | Path) -> None:
    """Rename ``src`` to ``dst``, copying when rename is unsupported.

    An existing ``dst`` is replaced.  The copy fallback is not atomic; callers
    that need crash safety rely on the ``_SUCCESS`` marker written afterwards.
    """

    src, dst = Path(src), Path(dst)
    if dst.is_dir():
        shutil.rmtree(dst)
    elif dst.exists():
        dst.unlink()

    try:
        os.replace(src, dst)
        return
    except OSError as exc:
        if exc.errno not in _RENAME_ERRNOS:
            raise

    # Keep the data even where rename(2) is unavailable.  The destination is
    # not usable until the caller writes _SUCCESS, and prepare_output removes
    # it if this copy dies halfway.
    shutil.copytree(src, dst, symlinks=True)
    shutil.rmtree(src)


def write_text_atomic(path: str | Path, text: str) -> Path:
    """Write ``text`` to ``path`` via a temp file, with a direct-write fallback."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    try:
        os.replace(tmp, path)
    except OSError as exc:
        if exc.errno not in _RENAME_ERRNOS:
            raise
        # Same rename limitation as above: overwrite the file in place.
        path.write_text(text, encoding="utf-8")
        tmp.unlink(missing_ok=True)
    return path


def write_failure(task: ConversionTask, error: str) -> Path:
    """Persist a per-task failure next to the output, atomically if possible."""

    payload: dict[str, Any] = {
        "task_id": task.task_id,
        "input_path": str(task.input_path),
        "error": error,
        "failed_at": time.time(),
        "host": socket.gethostname(),
    }
    return write_text_atomic(
        failure_path(task.output_path),
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


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
