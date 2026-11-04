"""Driver utilities: discovery -> tasks -> Ray -> optional aggregation.

These functions are intentionally free of CLI concerns so they can be called
from a notebook or a scheduler.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from ego2lerobot.common import ConversionTask, RayExecutor, RayExecutorConfig, TaskResult
from ego2lerobot.common.discovery import (
    DEFAULT_EXCLUDE,
    DiscoveredFile,
    DiscoveryConfig,
    build_output_path,
    discover_files,
    summarize_discovery,
    write_manifest,
)
from ego2lerobot.common.state import SUCCESS_MARKER, is_task_done, partial_path
from ego2lerobot.config import PipelineConfig


def reader_includes(pipeline: PipelineConfig) -> tuple[str, ...]:
    """Default file globs derived from the configured reader."""

    if pipeline.reader.kind == "h5":
        return ("**/*.h5", "**/*.hdf5", "**/*.hdf")
    if pipeline.reader.kind == "mcap":
        return ("**/*.mcap",)
    if pipeline.reader.kind == "ego_egoverse":
        # One sub-dataset (episode) has exactly one zarr.json anchor.
        return ("**/zarr.json",)
    return ("**/*",)


@dataclass
class ConversionPlan:
    files: list[DiscoveredFile]
    tasks: list[ConversionTask]
    roots: list[Path] = field(default_factory=list)
    manifest_path: Path | None = None

    def summary(self) -> dict[str, Any]:
        payload = summarize_discovery(self.files)
        payload["tasks"] = len(self.tasks)
        payload["manifest"] = str(self.manifest_path) if self.manifest_path else None
        return payload


def plan_conversion(
    pipeline: PipelineConfig,
    roots: Sequence[str | Path],
    *,
    output_root: str | Path | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    follow_symlinks: bool = False,
    max_depth: int | None = None,
    max_files: int | None = None,
    manifest_path: str | Path | None = None,
) -> ConversionPlan:
    """Discover files and build one task per file."""

    discovery = DiscoveryConfig(
        roots=list(roots),
        include=tuple(include) if include else reader_includes(pipeline),
        exclude=tuple(exclude) if exclude else DEFAULT_EXCLUDE,
        follow_symlinks=follow_symlinks,
        max_depth=max_depth,
        max_files=max_files,
        # Readers can reject look-alike files (e.g. Zarr array-node zarr.json).
        file_filter=pipeline.reader.accepts_input,
    )
    files = discover_files(discovery)
    tasks = build_tasks(files, output_root or pipeline.output_root)

    manifest = Path(manifest_path) if manifest_path else None
    if manifest is not None:
        write_manifest(manifest, files)

    return ConversionPlan(
        files=files,
        tasks=tasks,
        roots=[Path(root).expanduser().resolve() for root in roots],
        manifest_path=manifest,
    )


def build_tasks(
    files: Sequence[DiscoveredFile], output_root: str | Path
) -> list[ConversionTask]:
    """Map discovered files to unique tasks.

    The relative path is preserved under ``output_root`` and a short hash is
    appended, so equal stems in different sub-directories cannot collide.
    """

    root_labels = _root_labels(files)
    tasks: list[ConversionTask] = []

    for item in files:
        label = root_labels[item.root]
        relative = f"{label}/{item.relative_path}" if label else item.relative_path
        key = PurePosixPath(relative).with_suffix("").as_posix()
        digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:10]
        output = build_output_path(
            DiscoveredFile(
                path=item.path,
                root=item.root,
                relative_path=relative,
                size=item.size,
                mtime=item.mtime,
                inode=item.inode,
            ),
            output_root,
        )
        tasks.append(
            ConversionTask(
                task_id=f"{key}-{digest}",
                input_path=item.path,
                output_path=output,
                metadata={
                    "relative_path": relative,
                    "root": str(item.root),
                    "source_size": item.size,
                    "source_mtime": item.mtime,
                    "suffix": item.suffix,
                },
            )
        )

    ids = [task.task_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("task ids are not unique; check overlapping roots")
    return tasks


def _root_labels(files: Sequence[DiscoveredFile]) -> dict[Path, str]:
    roots = sorted({item.root for item in files}, key=str)
    counts: dict[str, int] = {}
    for root in roots:
        counts[root.name] = counts.get(root.name, 0) + 1
    labels: dict[Path, str] = {}
    for index, root in enumerate(roots):
        if len(roots) == 1:
            labels[root] = ""
        elif counts[root.name] > 1:
            labels[root] = f"{root.name}-{index}"
        else:
            labels[root] = root.name
    return labels


def make_executor_config(
    *,
    pipeline: PipelineConfig,
    plan: ConversionPlan,
    output_root: str | Path,
    ray_address: str | None = "auto",
    cpus_per_task: float = 1.0,
    gpus_per_task: float = 0.0,
    workers: int = -1,
    tasks_per_job: int = 1,
    max_retries: int = 1,
    preflight: bool = True,
    preflight_ffmpeg: bool = False,
    working_dir: str | Path | None = None,
    shutdown_after_run: bool = True,
) -> RayExecutorConfig:
    output_root = Path(output_root)

    imports = ["ego2lerobot", "lerobot"]
    if pipeline.reader.kind == "h5":
        imports.append("h5py")
    elif pipeline.reader.kind == "ego_egoverse":
        # Decoding is lazy, but every node must have these before the run.
        imports.extend(["zarr", "PIL"])

    return RayExecutorConfig(
        address=ray_address,
        cpus_per_task=cpus_per_task,
        gpus_per_task=gpus_per_task,
        workers=workers,
        tasks_per_job=tasks_per_job,
        max_retries=max_retries,
        working_dir=working_dir,
        preflight=preflight,
        preflight_imports=imports,
        preflight_distributions=["lerobot"],
        preflight_ffmpeg=preflight_ffmpeg,
        probe_dir=output_root / ".ray_probe",
        sample_input=plan.files[0].path if plan.files else None,
        log_dir=output_root / "_logs",
        shutdown_after_run=shutdown_after_run,
    )


def run_conversion(
    pipeline: PipelineConfig,
    config_path: str | Path,
    plan: ConversionPlan,
    *,
    output_root: str | Path | None = None,
    ray_address: str | None = "auto",
    cpus_per_task: float = 1.0,
    gpus_per_task: float = 0.0,
    workers: int = -1,
    tasks_per_job: int = 1,
    max_retries: int = 1,
    preflight: bool = True,
    preflight_ffmpeg: bool = False,
    working_dir: str | Path | None = None,
) -> tuple[list[TaskResult], RayExecutor]:
    """Submit all tasks that are not already committed."""

    from ego2lerobot.convert.worker import process_batch

    output_root = Path(output_root or pipeline.output_root)

    # Skip already-finished tasks on the driver: this keeps the in-flight
    # window full of real work instead of trivial "skipped" results.
    pending = [task for task in plan.tasks if not is_task_done(task)]
    skipped = [
        TaskResult(
            task_id=task.task_id,
            status="skipped",
            output_path=str(task.output_path),
        )
        for task in plan.tasks
        if is_task_done(task)
    ]

    executor_config = make_executor_config(
        pipeline=pipeline,
        plan=plan,
        output_root=output_root,
        ray_address=ray_address,
        cpus_per_task=cpus_per_task,
        gpus_per_task=gpus_per_task,
        workers=workers,
        tasks_per_job=tasks_per_job,
        max_retries=max_retries,
        preflight=preflight,
        preflight_ffmpeg=preflight_ffmpeg,
        working_dir=working_dir,
    )
    executor = RayExecutor(executor_config)

    if pending:
        results = executor.run(
            process_batch,
            pending,
            {"config_path": str(Path(config_path).expanduser().resolve())},
        )
    else:
        executor.last_summary = {
            "total": 0,
            "ok": 0,
            "skipped": 0,
            "failed": 0,
            "episodes": 0,
            "frames": 0,
            "task_elapsed_s": 0.0,
            "wall_elapsed_s": 0.0,
        }
        results = []

    merged = _merge_results(plan.tasks, results, skipped)
    return merged, executor


def _merge_results(
    tasks: Sequence[ConversionTask],
    results: Sequence[TaskResult],
    skipped: Sequence[TaskResult],
) -> list[TaskResult]:
    by_id = {result.task_id: result for result in [*skipped, *results]}
    merged: list[TaskResult] = []
    for task in tasks:
        merged.append(
            by_id.get(
                task.task_id,
                TaskResult(
                    task_id=task.task_id,
                    status="failed",
                    output_path=str(task.output_path),
                    error="no result for task",
                ),
            )
        )
    return merged


def write_run_summary(
    path: str | Path,
    pipeline: PipelineConfig,
    plan: ConversionPlan,
    results: Sequence[TaskResult],
    *,
    wall_elapsed_s: float | None = None,
) -> dict[str, Any]:
    from ego2lerobot.common.types import summarize_results

    summary = summarize_results(results)
    if wall_elapsed_s is not None:
        summary["wall_elapsed_s"] = wall_elapsed_s
    payload = {
        "generated_at": time.time(),
        "config": pipeline.describe(),
        "discovery": plan.summary(),
        "summary": summary,
        "episode_order": [
            {
                "task_id": result.task_id,
                "output_path": result.output_path,
                "frames": result.frames,
            }
            for result in results
            if result.status == "ok"
        ],
        "skipped": [
            {
                "task_id": result.task_id,
                "output_path": result.output_path,
                "frames": result.frames,
            }
            for result in results
            if result.status == "skipped"
        ],
        "failures": [
            {"task_id": result.task_id, "error": result.error}
            for result in results
            if result.status == "failed"
        ],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return payload


def committed_roots(results: Sequence[TaskResult]) -> list[Path]:
    """Output roots of the tasks that were committed (``_SUCCESS`` present)."""

    return [
        Path(result.output_path)
        for result in results
        if result.status in {"ok", "skipped"}
        and (Path(result.output_path) / SUCCESS_MARKER).is_file()
    ]


def aggregate(
    pipeline: PipelineConfig,
    results: Sequence[TaskResult],
    *,
    output_root: str | Path | None = None,
    repo_id: str | None = None,
    force: bool = False,
) -> Path | None:
    """Merge per-task LeRobot datasets in deterministic task order.

    Only committed datasets (``_SUCCESS``) are included.  Aggregation is
    skipped when every task failed, or when the target already exists and
    ``force`` is false.

    This is the sequential LeRobot backend; see
    :mod:`ego2lerobot.convert.aggregate_raydata` for the Ray Data one.
    """

    roots = committed_roots(results)
    if not roots:
        return None

    target = Path(output_root or pipeline.aggregate_output_root)
    marker = target / SUCCESS_MARKER
    if marker.is_file() and not force:
        return target

    from lerobot.datasets.aggregate import aggregate_datasets

    partial = partial_path(target)
    shutil.rmtree(partial, ignore_errors=True)
    # aggregate_datasets() creates aggr_root itself and requires it to be absent.
    partial.parent.mkdir(parents=True, exist_ok=True)

    aggregate_datasets(
        repo_ids=[pipeline.repo_id] * len(roots),
        aggr_repo_id=repo_id or pipeline.aggregate_repo_id,
        roots=roots,
        aggr_root=partial,
    )

    marker_tmp = partial / SUCCESS_MARKER
    marker_tmp.write_text(
        json.dumps(
            {
                "datasets": len(roots),
                "repo_id": repo_id or pipeline.aggregate_repo_id,
                "created_at": time.time(),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    os.replace(partial, target)
    return target


__all__ = [
    "ConversionPlan",
    "aggregate",
    "build_tasks",
    "make_executor_config",
    "plan_conversion",
    "reader_includes",
    "run_conversion",
    "write_run_summary",
]
