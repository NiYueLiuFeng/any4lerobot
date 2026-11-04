"""Demo worker: read JSON episodes, process them, write atomically.

In a real converter, replace :func:`_process_one` with your h5/mcap reader and
LeRobot writer.  The execution layer does not need to change.
"""

from __future__ import annotations

import json
import socket
import time
import traceback
from typing import Any

from ego2lerobot.common.state import (
    commit_output,
    is_task_done,
    partial_path,
    prepare_output,
    write_failure,
    write_task_log,
)
from ego2lerobot.common.types import ConversionTask, TaskResult


def process_batch(batch: list[ConversionTask], config: dict[str, Any]) -> list[TaskResult]:
    """Entry point called by the Ray executor for one batch of tasks."""

    return [_process_one(task, config) for task in batch]


def _process_one(task: ConversionTask, config: dict[str, Any]) -> TaskResult:
    started = time.perf_counter()
    host = socket.gethostname()
    log_dir = (config.get("_executor") or {}).get("log_dir")

    if is_task_done(task):
        write_task_log(log_dir, task, "skipped")
        return TaskResult(
            task_id=task.task_id,
            status="skipped",
            output_path=str(task.output_path),
            elapsed_s=round(time.perf_counter() - started, 3),
            host=host,
        )

    try:
        prepare_output(task)

        payload = json.loads(task.input_path.read_text(encoding="utf-8"))
        frames = payload["frames"]
        if not isinstance(frames, list) or not frames:
            raise ValueError("frames must be a non-empty list")

        # Simulate CPU work (decoding / resampling / encoding).  This makes the
        # effect of Ray parallelism visible in the demo.
        compute_ms = int(config.get("simulate_compute_ms", 0))
        if compute_ms > 0:
            time.sleep(compute_ms / 1000.0)

        result = {
            "task_id": task.task_id,
            "relative_path": task.metadata.get("relative_path", ""),
            "episode_id": payload.get("episode_id", task.input_path.stem),
            "frames": len(frames),
            "state_mean": _mean([frame["state"] for frame in frames]),
            "action_mean": _mean([frame["action"] for frame in frames]),
            "host": host,
        }

        partial = partial_path(task.output_path)
        partial.mkdir(parents=True, exist_ok=True)
        (partial / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        # Atomic commit: <output>.partial -> <output>, then write _SUCCESS.
        commit_output(task)

        elapsed = round(time.perf_counter() - started, 3)
        write_task_log(log_dir, task, "ok", frames=len(frames), episodes=1)
        return TaskResult(
            task_id=task.task_id,
            status="ok",
            episodes=1,
            frames=len(frames),
            elapsed_s=elapsed,
            output_path=str(task.output_path),
            host=host,
        )
    except Exception as exc:
        # Data errors are returned, not raised: one bad file must not kill the
        # whole batch or trigger endless Ray retries.  Transient IO errors can
        # be re-raised in a real worker to let Ray retry them.
        write_failure(task, traceback.format_exc())
        write_task_log(log_dir, task, "failed", error=f"{type(exc).__name__}: {exc}")
        return TaskResult(
            task_id=task.task_id,
            status="failed",
            elapsed_s=round(time.perf_counter() - started, 3),
            output_path=str(task.output_path),
            error=f"{type(exc).__name__}: {exc}",
            host=host,
        )


def _mean(rows: list[list[float]]) -> list[float]:
    if not rows:
        return []
    width = len(rows[0])
    for row in rows:
        if len(row) != width:
            raise ValueError("inconsistent vector length across frames")
    return [round(sum(float(row[i]) for row in rows) / len(rows), 6) for i in range(width)]
