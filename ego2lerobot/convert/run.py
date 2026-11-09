"""Driver utilities for the conversion pipeline.

The driver is a sequence of small, CLI-free stages so it can also be called
from a notebook or a scheduler::

    plan_conversion()      # discover raw files -> one ConversionTask per file
    make_executor_config() # describe the Ray runtime (imports, preflight, ...)
    run_conversion()       # submit pending tasks, skip committed ones
    write_run_summary()    # JSON report for logs and downstream tooling
    aggregate()            # merge per-task LeRobot datasets into one dataset
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from ego2lerobot.common import (
    ConversionTask,
    RayExecutor,
    RayExecutorConfig,
    TaskResult,
    summarize_results,
)
from ego2lerobot.common.discovery import (
    DEFAULT_EXCLUDE,
    DiscoveredFile,
    DiscoveryConfig,
    build_output_path,
    discover_files,
    summarize_discovery,
    write_manifest,
)
from ego2lerobot.common.state import (
    SUCCESS_MARKER,
    is_task_done,
    move_into_place,
    staging_directory,
    write_text_atomic,
)
from ego2lerobot.config import PipelineConfig


# ---------------------------------------------------------------------------
# File globs and Ray preflight imports per reader kind
# ---------------------------------------------------------------------------

# Each reader has an anchor file that identifies one episode.  For EgoVerse,
# an episode is a Zarr v3 group whose anchor is ``zarr.json``.
_READER_INCLUDES: dict[str, tuple[str, ...]] = {
    "h5": ("**/*.h5", "**/*.hdf5", "**/*.hdf"),
    "mcap": ("**/*.mcap",),
    "ego_egoverse": ("**/zarr.json",),
}

# Extra imports every Ray node must provide before a run starts.
_READER_PREFLIGHT_IMPORTS: dict[str, tuple[str, ...]] = {
    "h5": ("h5py",),
    # Decoding is lazy, but every node needs these when a task starts.
    "ego_egoverse": ("zarr", "PIL"),
}


def reader_includes(pipeline: PipelineConfig) -> tuple[str, ...]:
    """Default file globs derived from the configured reader."""

    return _READER_INCLUDES.get(pipeline.reader.kind, ("**/*",))


# ---------------------------------------------------------------------------
# Stage 1: planning - discovery -> tasks
# ---------------------------------------------------------------------------


@dataclass
class ConversionPlan:
    """Discovered inputs plus the tasks built from them."""

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
    """Discover raw files and build one conversion task per file."""

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
        # Prefix the relative path with a root label when several roots are
        # scanned, so files from different roots stay distinguishable.
        label = root_labels[item.root]
        relative = f"{label}/{item.relative_path}" if label else item.relative_path
        digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:10]
        task_id = f"{PurePosixPath(relative).with_suffix('').as_posix()}-{digest}"
        output_path = build_output_path(replace(item, relative_path=relative), output_root)

        tasks.append(
            ConversionTask(
                task_id=task_id,
                input_path=item.path,
                output_path=output_path,
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
    """Give each discovery root a stable prefix for task/output names."""

    roots = sorted({item.root for item in files}, key=str)
    counts = Counter(root.name for root in roots)

    labels: dict[Path, str] = {}
    for index, root in enumerate(roots):
        if len(roots) == 1:
            labels[root] = ""
        elif counts[root.name] > 1:
            # Two roots share a basename: disambiguate with a position suffix.
            labels[root] = f"{root.name}-{index}"
        else:
            labels[root] = root.name
    return labels


# ---------------------------------------------------------------------------
# Stage 2: execution - configure Ray and submit pending tasks
# ---------------------------------------------------------------------------


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
    """Describe the Ray runtime used for one conversion run."""

    output_root = Path(output_root)
    imports = [
        "ego2lerobot",
        "lerobot",
        *_READER_PREFLIGHT_IMPORTS.get(pipeline.reader.kind, ()),
    ]

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
    """Submit every task that is not already committed."""

    from ego2lerobot.convert.worker import process_batch

    output_root = Path(output_root or pipeline.output_root)

    # Partition on the driver: finished tasks never reach Ray, which keeps the
    # in-flight window full of real work instead of trivial "skipped" results.
    pending: list[ConversionTask] = []
    skipped: list[TaskResult] = []
    for task in plan.tasks:
        if is_task_done(task):
            skipped.append(_skipped_result(task))
        else:
            pending.append(task)

    executor = RayExecutor(
        make_executor_config(
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
    )

    if pending:
        results = executor.run(
            process_batch,
            pending,
            {"config_path": str(Path(config_path).expanduser().resolve())},
        )
    else:
        executor.last_summary = {**summarize_results([]), "wall_elapsed_s": 0.0}
        results = []

    return _merge_results(plan.tasks, results, skipped), executor


def _skipped_result(task: ConversionTask) -> TaskResult:
    return TaskResult(
        task_id=task.task_id,
        status="skipped",
        output_path=str(task.output_path),
    )


def _merge_results(
    tasks: Sequence[ConversionTask],
    results: Sequence[TaskResult],
    skipped: Sequence[TaskResult],
) -> list[TaskResult]:
    """Return one result per task in plan order, flagging missing results."""

    by_id = {result.task_id: result for result in (*skipped, *results)}
    merged: list[TaskResult] = []
    for task in tasks:
        result = by_id.get(task.task_id)
        if result is None:
            result = TaskResult(
                task_id=task.task_id,
                status="failed",
                output_path=str(task.output_path),
                error="no result for task",
            )
        merged.append(result)
    return merged


# ---------------------------------------------------------------------------
# Stage 3: reporting - write a machine-readable run summary
# ---------------------------------------------------------------------------


def write_run_summary(
    path: str | Path,
    pipeline: PipelineConfig,
    plan: ConversionPlan,
    results: Sequence[TaskResult],
    *,
    wall_elapsed_s: float | None = None,
) -> dict[str, Any]:
    """Write the run report atomically and return its payload."""

    summary = summarize_results(results)
    if wall_elapsed_s is not None:
        summary["wall_elapsed_s"] = wall_elapsed_s

    payload = {
        "generated_at": time.time(),
        "config": pipeline.describe(),
        "discovery": plan.summary(),
        "summary": summary,
        "episode_order": _result_rows(results, "ok"),
        "skipped": _result_rows(results, "skipped"),
        "failures": [
            {"task_id": result.task_id, "error": result.error}
            for result in results
            if result.status == "failed"
        ],
    }

    write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return payload


def _result_rows(results: Sequence[TaskResult], status: str) -> list[dict[str, Any]]:
    """Compact task rows for the ``ok`` and ``skipped`` report sections."""

    return [
        {
            "task_id": result.task_id,
            "output_path": result.output_path,
            "frames": result.frames,
        }
        for result in results
        if result.status == status
    ]


# ---------------------------------------------------------------------------
# Stage 4: aggregation - merge committed per-task datasets
# ---------------------------------------------------------------------------


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
    if (target / SUCCESS_MARKER).is_file() and not force:
        return target

    from lerobot.datasets.aggregate import aggregate_datasets

    # Build into ``<target>.partial`` and swap it into place only after the
    # success marker is written, so a crash never leaves a half-merged dataset.
    # On file systems without rename the staging directory *is* the target and
    # _SUCCESS is the completion signal.
    staging = staging_directory(target)
    shutil.rmtree(staging, ignore_errors=True)
    # aggregate_datasets() creates aggr_root itself and requires it to be absent.

    aggregate_datasets(
        repo_ids=[pipeline.repo_id] * len(roots),
        aggr_repo_id=repo_id or pipeline.aggregate_repo_id,
        roots=roots,
        aggr_root=staging,
    )

    (staging / SUCCESS_MARKER).write_text(
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

    if staging != target:
        move_into_place(staging, target)
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
