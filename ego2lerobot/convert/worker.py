"""Ray worker: read one raw episode, map fields, write a LeRobot dataset.

The worker is deliberately small.  All format-specific code lives in
``ego2lerobot.readers`` and all field naming lives in the YAML config, so the
same worker handles h5 and mcap files.

Contract with :class:`ego2lerobot.common.RayExecutor`::

    process_batch(batch, {"config_path": ".../pipeline.yaml", "_executor": {...}})
"""

from __future__ import annotations

import json
import os
import socket
import time
import traceback
from pathlib import Path
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
from ego2lerobot.config import PipelineConfig

_CONFIG_CACHE: dict[str, tuple[float, PipelineConfig]] = {}


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load and cache a config in the worker process.

    The cache key includes the file mtime, so editing the YAML while a long job
    is running is picked up by newly started workers.
    """

    resolved = str(Path(path).expanduser().resolve())
    try:
        mtime = os.path.getmtime(resolved)
    except OSError as exc:
        raise FileNotFoundError(f"pipeline config not visible on this node: {resolved}") from exc

    cached = _CONFIG_CACHE.get(resolved)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    config = PipelineConfig.load(resolved)
    _CONFIG_CACHE[resolved] = (mtime, config)
    return config


def process_batch(batch: list[ConversionTask], config: dict[str, Any]) -> list[TaskResult]:
    config_path = config.get("config_path")
    if not config_path:
        raise ValueError("worker config is missing 'config_path'")
    pipeline = load_pipeline_config(config_path)
    log_dir = (config.get("_executor") or {}).get("log_dir")
    return [process_task(task, pipeline, log_dir=log_dir) for task in batch]


def process_task(
    task: ConversionTask,
    pipeline: PipelineConfig,
    *,
    log_dir: str | Path | None = None,
) -> TaskResult:
    """Convert one task with an already-loaded pipeline.

    ``process_batch`` is the Ray entry point and reloads the config from disk.
    This variant takes the in-memory pipeline, so demos and notebooks can tweak
    the reader window or output root without writing a temporary config file.
    """
    started = time.perf_counter()
    host = socket.gethostname()

    if is_task_done(task):
        write_task_log(log_dir, task, "skipped")
        return TaskResult(
            task_id=task.task_id,
            status="skipped",
            output_path=str(task.output_path),
            elapsed_s=round(time.perf_counter() - started, 3),
            host=host,
        )

    prepare_output(task)
    dataset = None
    frames = 0
    try:
        pipeline.reader.validate(task.input_path)

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        partial = partial_path(task.output_path)
        # LeRobotDataset.create() requires a non-existing root and creates it
        # itself, so do not pre-create the partial directory here.
        partial.parent.mkdir(parents=True, exist_ok=True)

        dataset = LeRobotDataset.create(
            repo_id=pipeline.repo_id,
            fps=pipeline.fps,
            features=pipeline.mapping.lerobot_features(),
            root=partial,
            robot_type=pipeline.mapping.robot_type,
            **pipeline.writer_kwargs(),
        )

        for record in pipeline.reader.iter_frames(task.input_path):
            frame = pipeline.mapping.resolve(record)
            task_text = frame.get("task")
            if not isinstance(task_text, str) or not task_text:
                raise ValueError("mapping produced an empty 'task' value")
            dataset.add_frame(frame)
            frames += 1

        if frames == 0:
            raise ValueError(f"no frames decoded from {task.input_path}")

        dataset.save_episode()
        dataset.finalize()

        _write_conversion_meta(task, pipeline, partial, frames)

        # Atomic commit: <output>.partial -> <output>, then write _SUCCESS.
        commit_output(task)

        elapsed = round(time.perf_counter() - started, 3)
        write_task_log(log_dir, task, "ok", frames=frames, episodes=1)
        return TaskResult(
            task_id=task.task_id,
            status="ok",
            episodes=1,
            frames=frames,
            elapsed_s=elapsed,
            output_path=str(task.output_path),
            host=host,
        )
    except Exception as exc:
        # Release file handles and async image writers.  The partial directory
        # is not committed and is removed on the next attempt.
        if dataset is not None:
            try:
                dataset.finalize()
            except Exception:
                pass
        write_failure(task, traceback.format_exc())
        write_task_log(log_dir, task, "failed", error=f"{type(exc).__name__}: {exc}")
        return TaskResult(
            task_id=task.task_id,
            status="failed",
            episodes=0,
            frames=frames,
            elapsed_s=round(time.perf_counter() - started, 3),
            output_path=str(task.output_path),
            error=f"{type(exc).__name__}: {exc}",
            host=host,
        )


def _write_conversion_meta(
    task: ConversionTask, pipeline: PipelineConfig, partial: Path, frames: int
) -> None:
    """Record how this dataset was produced; harmless extra file in the root."""

    payload = {
        "task_id": task.task_id,
        "input_path": str(task.input_path),
        "frames": frames,
        "config_path": str(pipeline.source_path) if pipeline.source_path else None,
        "repo_id": pipeline.repo_id,
        "fps": pipeline.fps,
        "features": sorted(pipeline.mapping.features),
        "reader": pipeline.reader.describe(),
        "host": socket.gethostname(),
    }
    (partial / "_conversion.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


__all__ = ["load_pipeline_config", "process_batch", "process_task"]
