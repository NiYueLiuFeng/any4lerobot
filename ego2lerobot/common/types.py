"""Small, dependency-free data structures shared by the execution layer."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

TaskStatus = Literal["ok", "skipped", "failed"]


@dataclass(frozen=True)
class ConversionTask:
    """One independently convertible raw input.

    ``task_id`` must be unique and stable across runs.  Use a path relative to
    the dataset root plus a short hash, not just the file stem: different
    sub-directories can contain files with the same name.
    """

    task_id: str
    input_path: Path
    output_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_path", Path(self.input_path))
        object.__setattr__(self, "output_path", Path(self.output_path))


@dataclass(frozen=True)
class TaskResult:
    """Small result returned by a worker.

    Never put frames, images or arrays in here: results travel through the Ray
    object store and the driver's memory.
    """

    task_id: str
    status: TaskStatus
    episodes: int = 0
    frames: int = 0
    elapsed_s: float = 0.0
    output_path: str = ""
    error: str | None = None
    host: str | None = None


def summarize_results(results: Iterable[TaskResult]) -> dict[str, Any]:
    """Return counters suitable for logs/JSON, preserving task order upstream."""

    results = list(results)
    summary = {
        "total": len(results),
        "ok": 0,
        "skipped": 0,
        "failed": 0,
        "episodes": 0,
        "frames": 0,
        "task_elapsed_s": 0.0,
    }
    for result in results:
        if result.status in ("ok", "skipped", "failed"):
            summary[result.status] += 1
        summary["episodes"] += result.episodes
        summary["frames"] += result.frames
        summary["task_elapsed_s"] += result.elapsed_s
    summary["task_elapsed_s"] = round(summary["task_elapsed_s"], 3)
    return summary
