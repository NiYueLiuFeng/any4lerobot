"""A/B benchmark: sequential LeRobot aggregation vs Ray Data aggregation.

Both backends read the same committed per-task datasets and must produce a
dataset with identical rows, in identical order.  The script runs both, times
them and compares the result column by column::

    python -m ego2lerobot.demo.aggregate_ab --num-files 64 --frames 40

Existing ``data-dir/output`` datasets are reused; pass ``--regenerate`` to
rebuild the raw episodes and conversion output.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ego2lerobot.common.types import TaskResult  # noqa: E402
from ego2lerobot.config import PipelineConfig  # noqa: E402
from ego2lerobot.convert.aggregate_raydata import (  # noqa: E402
    AggregateNotSupported,
    aggregate_with_ray_data,
)
from ego2lerobot.convert.run import (  # noqa: E402
    aggregate,
    plan_conversion,
    run_conversion,
)

DEFAULT_CONFIG = REPO_ROOT / "ego2lerobot" / "configs" / "ego_h5.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=Path("/tmp/ego_agg_ab"))
    parser.add_argument("--num-files", type=int, default=64)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--convert-workers", type=int, default=4)
    parser.add_argument("--ray-address", default="local")
    parser.add_argument("--num-cpus", type=float, default=1.0)
    parser.add_argument(
        "--include-ray-startup",
        action="store_true",
        help="also charge Ray/Ray-Data initialisation to the ray-data backend",
    )
    return parser.parse_args()


def _load_results(output_dir: Path) -> list[TaskResult]:
    results: list[TaskResult] = []
    for marker in sorted(output_dir.rglob("_SUCCESS")):
        root = marker.parent
        if (root / "meta" / "info.json").is_file():
            results.append(
                TaskResult(
                    task_id=str(root.relative_to(output_dir)),
                    status="ok",
                    episodes=1,
                    output_path=str(root),
                )
            )
    return results


def _read_frames(root: Path):
    import pandas as pd

    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no data parquet under {root}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def _read_episodes(root: Path):
    import pandas as pd

    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def _cell(value):
    if isinstance(value, dict) and "bytes" in value:
        return ("bytes", value.get("bytes"), value.get("path"))
    if isinstance(value, np.ndarray):
        return ("array", value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, (list, tuple)):
        return tuple(_cell(item) for item in value)
    return value


def _compare_frames(left, right) -> list[str]:
    issues: list[str] = []
    if list(left.columns) != list(right.columns):
        issues.append(f"columns differ: {list(left.columns)} vs {list(right.columns)}")
        return issues
    if len(left) != len(right):
        issues.append(f"row count differs: {len(left)} vs {len(right)}")
        return issues

    for column in left.columns:
        lvalues = left[column].to_numpy()
        rvalues = right[column].to_numpy()
        if lvalues.dtype.kind in "iufb":
            if not np.array_equal(lvalues, rvalues):
                mismatches = int(np.sum(lvalues != rvalues))
                issues.append(f"{column}: {mismatches} value(s) differ")
            continue

        for index, (lvalue, rvalue) in enumerate(zip(left[column], right[column], strict=False)):
            if _cell(lvalue) != _cell(rvalue):
                issues.append(f"{column}: first difference at row {index}")
                break
    return issues


def _compare_episodes(left, right) -> list[str]:
    issues: list[str] = []
    columns = ["episode_index", "dataset_from_index", "dataset_to_index", "length"]
    if len(left) != len(right):
        return [f"episode count differs: {len(left)} vs {len(right)}"]
    for column in columns:
        if column not in left.columns or column not in right.columns:
            continue
        if not np.array_equal(left[column].to_numpy(), right[column].to_numpy()):
            issues.append(f"episodes/{column} differs")
    return issues


def _summary(root: Path) -> dict:
    import pandas as pd

    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    episodes = _read_episodes(root)
    return {
        "root": str(root),
        "total_frames": info.get("total_frames"),
        "total_episodes": info.get("total_episodes"),
        "total_tasks": info.get("total_tasks"),
        "data_files": len(list((root / "data").rglob("*.parquet"))),
        "frames_df": len(_read_frames(root)),
        "episodes_df": len(episodes),
        "tasks": len(pd.read_json(root / "meta" / "tasks.jsonl", lines=True))
        if (root / "meta" / "tasks.jsonl").is_file()
        else None,
    }


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    raw_dir = data_dir / "raw"
    output_dir = data_dir / "output"
    lerobot_dir = data_dir / "aggregated_lerobot"
    ray_dir = data_dir / "aggregated_ray"

    config_path = args.config.expanduser().resolve()
    pipeline = PipelineConfig.load(config_path)

    if args.regenerate or not any(output_dir.rglob("_SUCCESS")):
        from ego2lerobot.demo.h5_demo import generate_h5

        if args.regenerate:
            shutil.rmtree(raw_dir, ignore_errors=True)
            shutil.rmtree(output_dir, ignore_errors=True)
        if not raw_dir.exists() or args.regenerate:
            generate_h5(
                raw_dir,
                num_files=args.num_files,
                frames=args.frames,
                corrupt=0,
                seed=args.seed,
            )
        plan = plan_conversion(pipeline, [raw_dir], output_root=output_dir)
        run_conversion(
            pipeline,
            config_path,
            plan,
            output_root=output_dir,
            ray_address=args.ray_address,
            workers=args.convert_workers,
            preflight=False,
        )

    results = _load_results(output_dir)
    if not results:
        print("[ab] no committed datasets", file=sys.stderr)
        return 1
    print(f"[ab] aggregating {len(results)} committed dataset(s)", flush=True)

    if not args.include_ray_startup:
        # In production the cluster is already running; charging Ray's ~5s
        # start-up to the aggregation would flatter the sequential backend.
        from ego2lerobot.common.ray_executor import RayExecutorConfig, build_runtime_env

        import ray

        if not ray.is_initialized():
            ray.init(
                address=None
                if (args.ray_address or "").lower() in {"", "local", "none"}
                else args.ray_address,
                ignore_reinit_error=True,
                include_dashboard=False,
                runtime_env=build_runtime_env(RayExecutorConfig(address=args.ray_address)),
            )

    shutil.rmtree(lerobot_dir, ignore_errors=True)
    started = time.perf_counter()
    aggregate(pipeline, results, output_root=lerobot_dir, force=True)
    lerobot_s = time.perf_counter() - started

    try:
        shutil.rmtree(ray_dir, ignore_errors=True)
        started = time.perf_counter()
        aggregate_with_ray_data(
            pipeline,
            results,
            output_root=ray_dir,
            force=True,
            ray_address=args.ray_address,
            num_cpus=args.num_cpus,
        )
        ray_s = time.perf_counter() - started
    except AggregateNotSupported as exc:
        print(f"[ab] ray-data backend not applicable: {exc}", file=sys.stderr)
        return 2

    left = _summary(lerobot_dir)
    right = _summary(ray_dir)
    print("\n=== summary ===")
    for key in ("total_frames", "total_episodes", "total_tasks", "data_files", "frames_df", "episodes_df"):
        print(f"  {key:15s} lerobot={left[key]!s:>8}  ray-data={right[key]!s:>8}")

    print("\n=== timing ===")
    print(f"  lerobot  : {lerobot_s:.2f}s")
    print(f"  ray-data : {ray_s:.2f}s")
    if ray_s > 0:
        print(f"  speedup  : {lerobot_s / ray_s:.2f}x")

    print("\n=== correctness ===")
    issues = _compare_frames(_read_frames(lerobot_dir), _read_frames(ray_dir))
    issues += _compare_episodes(_read_episodes(lerobot_dir), _read_episodes(ray_dir))
    for key in ("total_frames", "total_episodes", "total_tasks"):
        if left[key] != right[key]:
            issues.append(f"info/{key}: {left[key]} vs {right[key]}")

    if issues:
        print("  MISMATCH")
        for issue in issues:
            print(f"    - {issue}")
        return 1

    print("  identical rows, order, indices, tasks and image bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
