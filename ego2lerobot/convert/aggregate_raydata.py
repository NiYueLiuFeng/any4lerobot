"""Ray Data implementation of LeRobot dataset aggregation.

Why
---
``lerobot.datasets.aggregate.aggregate_datasets`` merges sources sequentially on
a single process: for every source it reads one parquet file, rewrites the
indices and appends the result to the destination.  With thousands of episodes
this is the slowest stage of the conversion pipeline.

This module produces the same dataset with ``ray.data`` by turning the
sequential merge into a fan-out over destination files:

1. driver: read every source's metadata, validate it, compute the cumulative
   frame/episode offsets and assign one destination file per source file;
2. ``ray.data.map``: rewrite every source data file in parallel;
3. ``ray.data.map``: rewrite every source episodes file in parallel;
4. driver: write ``tasks.jsonl`` / ``info.json`` / ``stats.json`` with LeRobot's
   own ``finalize_aggregation``.

The per-file transforms are LeRobot's own ``update_data_df`` /
``update_meta_data`` / ``to_parquet_with_hf_images``; only the *scheduling*
changes, so the two backends cannot drift in their index arithmetic.

Constraints
-----------
* No video features.  Video concatenation is inherently sequential (ffmpeg
  concat of growing files), so the driver refuses to run and callers should
  fall back to ``lerobot``'s implementation.
* One destination file per source file.  Rows and their order are identical to
  ``aggregate_datasets``; the file layout differs (LeRobot packs by size, this
  backend keeps the 1:1 mapping to stay embarrassingly parallel).  Use
  ``chunks_size`` / ``data_files_size_in_mb`` of the destination metadata to
  control the chunk directory, not the file size.
* Sources must contain exactly one data file and one episodes file.  Datasets
  written by ``ego2lerobot.convert.worker`` always satisfy this (one episode
  per task).
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ego2lerobot.common.ray_executor import RayExecutorConfig, build_runtime_env
from ego2lerobot.common.state import SUCCESS_MARKER, partial_path
from ego2lerobot.common.types import TaskResult
from ego2lerobot.config import PipelineConfig


class AggregateNotSupported(RuntimeError):
    """The inputs cannot be aggregated with the Ray Data backend."""


@dataclass(frozen=True)
class _SourcePlan:
    index: int
    root: Path
    frame_offset: int
    episode_offset: int
    data_file: tuple[int, int]
    episodes_file: tuple[int, int]
    dst_data_file: tuple[int, int]
    dst_episodes_file: tuple[int, int]


def aggregate_with_ray_data(
    pipeline: PipelineConfig,
    results: Sequence[TaskResult],
    *,
    output_root: str | Path | None = None,
    repo_id: str | None = None,
    force: bool = False,
    ray_address: str | None = None,
    num_cpus: float = 1.0,
    working_dir: str | Path | None = None,
) -> Path | None:
    """Merge committed per-task datasets with ``ray.data``.

    Returns the aggregated root, or ``None`` when there is nothing to merge.
    Raises :class:`AggregateNotSupported` when the sources need the sequential
    LeRobot backend (videos or multi-file sources).
    """

    from ego2lerobot.convert.run import committed_roots

    roots = committed_roots(results)
    if not roots:
        return None

    target = Path(output_root or pipeline.aggregate_output_root)
    if (target / SUCCESS_MARKER).is_file() and not force:
        return target

    from lerobot.datasets.aggregate import finalize_aggregation, validate_all_metadata
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.utils import DEFAULT_DATA_PATH, DEFAULT_EPISODES_PATH

    repo_ids = [pipeline.repo_id] * len(roots)
    metas = [
        LeRobotDatasetMetadata(repo_id, root=root)
        for repo_id, root in zip(repo_ids, roots)
    ]
    fps, robot_type, features = validate_all_metadata(metas)

    video_keys = [key for key, spec in features.items() if spec.get("dtype") == "video"]
    if video_keys:
        raise AggregateNotSupported(
            f"video features {sorted(video_keys)} need sequential concatenation; "
            "use the lerobot backend"
        )

    partial = partial_path(target)
    shutil.rmtree(partial, ignore_errors=True)
    # LeRobotDatasetMetadata.create() creates the root itself and requires it
    # to be absent, so only the parent is created here.
    partial.parent.mkdir(parents=True, exist_ok=True)

    dst_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id or pipeline.aggregate_repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        root=partial,
        use_videos=False,
    )

    unique_tasks = _unique_tasks(metas)
    dst_meta.tasks = _tasks_frame(unique_tasks, list(range(len(unique_tasks))))

    plans = _plan_sources(metas, roots, dst_meta.chunks_size)
    data_items, episodes_items = _build_items(
        plans,
        metas,
        pipeline.repo_id,
        unique_tasks,
        partial,
        DEFAULT_DATA_PATH,
        DEFAULT_EPISODES_PATH,
    )

    stats = _run_stage(
        "data",
        data_items,
        _write_data_file,
        ray_address=ray_address,
        num_cpus=num_cpus,
        working_dir=working_dir,
    )
    stats += _run_stage(
        "episodes",
        episodes_items,
        _write_episodes_file,
        ray_address=ray_address,
        num_cpus=num_cpus,
        working_dir=working_dir,
    )

    finalize_aggregation(dst_meta, metas)

    marker = partial / SUCCESS_MARKER
    marker.write_text(
        json.dumps(
            {
                "backend": "ray-data",
                "datasets": len(roots),
                "repo_id": repo_id or pipeline.aggregate_repo_id,
                "files_written": stats,
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


# ------------------------------------------------------------------- planning


def _unique_tasks(metas: Sequence[Any]) -> list[str]:
    """Union of task names in first-seen order (same as pandas ``unique``)."""

    import pandas as pd

    combined = pd.concat([meta.tasks for meta in metas])
    return [str(name) for name in combined.index.unique()]


def _tasks_frame(names: Sequence[str], indices: Sequence[int]):
    import pandas as pd

    return pd.DataFrame(
        {"task_index": list(indices)},
        index=pd.Index(list(names), name="task"),
    )


def _chunk_files(episodes, prefix: str) -> list[tuple[int, int]]:
    pairs = {
        (int(chunk), int(file))
        for chunk, file in zip(
            episodes[f"{prefix}/chunk_index"],
            episodes[f"{prefix}/file_index"],
            strict=False,
        )
    }
    return sorted(pairs)


def _plan_sources(
    metas: Sequence[Any], roots: Sequence[Path], chunk_size: int
) -> list[_SourcePlan]:
    plans: list[_SourcePlan] = []
    frame_offset = 0
    episode_offset = 0

    for index, (meta, root) in enumerate(zip(metas, roots)):
        data_files = _chunk_files(meta.episodes, "data")
        episodes_files = _chunk_files(meta.episodes, "meta/episodes")
        if len(data_files) != 1 or len(episodes_files) != 1:
            raise AggregateNotSupported(
                f"{root}: expected one data file and one episodes file, got "
                f"{data_files} and {episodes_files}; use the lerobot backend"
            )

        plans.append(
            _SourcePlan(
                index=index,
                root=Path(root),
                frame_offset=frame_offset,
                episode_offset=episode_offset,
                data_file=data_files[0],
                episodes_file=episodes_files[0],
                dst_data_file=divmod(index, chunk_size),
                dst_episodes_file=divmod(index, chunk_size),
            )
        )
        frame_offset += int(meta.total_frames)
        episode_offset += int(meta.total_episodes)

    return plans


def _build_items(
    plans: Sequence[_SourcePlan],
    metas: Sequence[Any],
    repo_id: str,
    unique_tasks: Sequence[str],
    partial: Path,
    data_pattern: str,
    episodes_pattern: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    global_index = {name: index for index, name in enumerate(unique_tasks)}
    data_items: list[dict[str, Any]] = []
    episodes_items: list[dict[str, Any]] = []

    for plan, meta in zip(plans, metas):
        # ``update_data_df`` indexes src_meta.tasks positionally by task_index.
        ordered = sorted(meta.tasks.index.tolist(), key=lambda name: meta.tasks.loc[name, "task_index"])
        task_names = [str(name) for name in ordered]
        task_indices = [global_index[name] for name in task_names]

        src_chunk, src_file = plan.data_file
        dst_chunk, dst_file = plan.dst_data_file
        data_items.append(
            {
                "repo_id": repo_id,
                "src_root": str(plan.root),
                "src_rel": data_pattern.format(chunk_index=src_chunk, file_index=src_file),
                "dst_path": str(
                    partial / data_pattern.format(chunk_index=dst_chunk, file_index=dst_file)
                ),
                "frame_offset": plan.frame_offset,
                "episode_offset": plan.episode_offset,
                "task_names": task_names,
                "task_indices": task_indices,
            }
        )

        src_chunk, src_file = plan.episodes_file
        dst_chunk, dst_file = plan.dst_episodes_file
        episodes_items.append(
            {
                "src_root": str(plan.root),
                "src_rel": episodes_pattern.format(chunk_index=src_chunk, file_index=src_file),
                "dst_path": str(
                    partial / episodes_pattern.format(chunk_index=dst_chunk, file_index=dst_file)
                ),
                "frame_offset": plan.frame_offset,
                "episode_offset": plan.episode_offset,
                "src_chunk": src_chunk,
                "src_file": src_file,
                "dst_data_chunk": plan.dst_data_file[0],
                "dst_data_file": plan.dst_data_file[1],
                "meta_chunk": dst_chunk,
                "meta_file": dst_file,
            }
        )

    return data_items, episodes_items


# -------------------------------------------------------------------- workers


class _MetaView:
    """Minimal stand-in for ``LeRobotDatasetMetadata``.

    LeRobot's update helpers only touch ``info`` and ``tasks``; passing a view
    lets us give every source its own offsets without mutating shared state.
    """

    def __init__(self, info: dict[str, int], tasks):
        self.info = info
        self.tasks = tasks


def _write_data_file(item: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd
    from lerobot.datasets.aggregate import update_data_df
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.feature_utils import get_hf_features_from_features
    from lerobot.datasets.io_utils import to_parquet_with_hf_images

    src_meta = LeRobotDatasetMetadata(item["repo_id"], root=item["src_root"])
    src_path = Path(item["src_root"]) / item["src_rel"]

    if src_meta.image_keys:
        import datasets

        frame = datasets.Dataset.from_parquet(str(src_path)).to_pandas()
    else:
        frame = pd.read_parquet(src_path)

    view = _MetaView(
        {
            "total_episodes": item["episode_offset"],
            "total_frames": item["frame_offset"],
        },
        _tasks_frame(item["task_names"], item["task_indices"]),
    )
    frame = update_data_df(frame, src_meta, view)

    out = Path(item["dst_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    if src_meta.image_keys:
        to_parquet_with_hf_images(
            frame, out, features=get_hf_features_from_features(src_meta.features)
        )
    else:
        frame.to_parquet(out)
    return {"kind": "data", "rows": int(len(frame)), "path": str(out)}


def _write_episodes_file(item: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd
    from lerobot.datasets.aggregate import update_meta_data

    src_path = Path(item["src_root"]) / item["src_rel"]
    frame = pd.read_parquet(src_path)

    view = _MetaView(
        {
            "total_episodes": item["episode_offset"],
            "total_frames": item["frame_offset"],
        },
        None,
    )
    data_idx = {
        "chunk": 0,
        "file": 0,
        "src_to_dst": {
            (item["src_chunk"], item["src_file"]): (
                item["dst_data_chunk"],
                item["dst_data_file"],
            )
        },
    }
    frame = update_meta_data(
        frame,
        view,
        {"chunk": item["meta_chunk"], "file": item["meta_file"]},
        data_idx,
        {},
    )

    out = Path(item["dst_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out)
    return {"kind": "episodes", "rows": int(len(frame)), "path": str(out)}


# ------------------------------------------------------------------ execution


def _run_stage(
    name: str,
    items: Sequence[dict[str, Any]],
    fn,
    *,
    ray_address: str | None,
    num_cpus: float,
    working_dir: str | Path | None,
) -> int:
    if not items:
        return 0

    import ray
    import ray.data

    if not ray.is_initialized():
        runtime_env = build_runtime_env(
            RayExecutorConfig(address=ray_address, working_dir=working_dir)
        )
        ray.init(
            address=_resolve_address(ray_address),
            ignore_reinit_error=True,
            include_dashboard=False,
            runtime_env=runtime_env,
        )

    started = time.perf_counter()
    dataset = ray.data.from_items(list(items))
    rows = dataset.map(fn, num_cpus=num_cpus).take_all()
    written = sum(int(row.get("rows", 0)) for row in rows)
    print(
        f"[aggregate:ray-data] {name}: {len(items)} file(s), "
        f"{written} row(s) in {time.perf_counter() - started:.2f}s",
        flush=True,
    )
    return len(items)


def _resolve_address(address: str | None) -> str | None:
    if address is None:
        return None
    if address.strip().lower() in {"", "local", "none"}:
        return None
    return address


__all__ = ["AggregateNotSupported", "aggregate_with_ray_data"]
