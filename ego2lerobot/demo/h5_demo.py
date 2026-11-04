"""End-to-end HDF5 -> LeRobot demo on top of the distributed converter.

It generates nested ``batch_XX/scene_Y/episode_NNNN.h5`` files, then runs the
same discovery + Ray + mapping + LeRobot pipeline used for production data::

    python -m ego2lerobot.demo.h5_demo --regenerate --workers 4

The only difference from real data is :func:`generate_h5`; replace it with your
raw episodes and keep the rest.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ego2lerobot.config import PipelineConfig  # noqa: E402
from ego2lerobot.convert.run import (  # noqa: E402
    plan_conversion,
    run_conversion,
    write_run_summary,
)

DEFAULT_CONFIG = REPO_ROOT / "ego2lerobot" / "configs" / "ego_h5.yaml"


def generate_h5(
    raw_dir: Path,
    *,
    num_files: int,
    frames: int,
    corrupt: int,
    seed: int,
) -> None:
    import h5py

    rng = np.random.default_rng(seed)
    raw_dir.mkdir(parents=True, exist_ok=True)

    for index in range(num_files):
        group = raw_dir / f"batch_{index // 8:02d}" / f"scene_{index % 3}"
        group.mkdir(parents=True, exist_ok=True)
        path = group / f"episode_{index:04d}.h5"

        if index < corrupt:
            path.write_bytes(b"not an hdf5 file\n")
            continue

        qpos = rng.normal(0.0, 0.1, size=(frames, 7)).astype(np.float32)
        cmd = rng.normal(0.0, 0.1, size=(frames, 7)).astype(np.float32)
        images = rng.integers(0, 256, size=(frames, 64, 96, 3), dtype=np.uint8)

        with h5py.File(path, "w") as handle:
            handle.create_dataset("observations/qpos", data=qpos, compression="gzip")
            handle.create_dataset("action", data=cmd, compression="gzip")
            handle.create_dataset("observations/images/head", data=images, compression="gzip")
            handle.attrs["instruction"] = "pick up the cube"
            handle.attrs["episode_id"] = path.stem

    print(f"[demo] generated {num_files} h5 file(s) under {raw_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=Path("/tmp/ego_h5_demo"))
    parser.add_argument("--num-files", type=int, default=8)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--corrupt", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--generate-only", action="store_true")

    parser.add_argument("--ray-address", default="local")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tasks-per-job", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--cpus-per-task", type=float, default=1.0)
    parser.add_argument("--no-preflight", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_files <= 0:
        raise SystemExit("--num-files must be > 0")
    if args.frames <= 0:
        raise SystemExit("--frames must be > 0")

    data_dir = args.data_dir.expanduser().resolve()
    raw_dir = data_dir / "raw"
    output_dir = data_dir / "output"
    aggregate_dir = data_dir / "aggregated"

    if args.clean:
        for path in (output_dir, aggregate_dir):
            shutil.rmtree(path, ignore_errors=True)
        (data_dir / "_run_summary.json").unlink(missing_ok=True)

    existing = sorted(raw_dir.rglob("*.h5")) if raw_dir.exists() else []
    if args.regenerate or not existing:
        if raw_dir.exists() and args.regenerate:
            shutil.rmtree(raw_dir)
        generate_h5(
            raw_dir,
            num_files=args.num_files,
            frames=args.frames,
            corrupt=args.corrupt,
            seed=args.seed,
        )
    elif args.corrupt:
        print("[demo] --corrupt ignored because data exists; use --regenerate", flush=True)

    config_path = args.config.expanduser().resolve()
    pipeline = PipelineConfig.load(config_path)
    pipeline = replace(pipeline, output_root=output_dir)

    plan = plan_conversion(pipeline, [raw_dir], output_root=output_dir)
    print(f"[demo] discovered {len(plan.tasks)} task(s)", flush=True)
    if args.generate_only:
        return 0

    results, executor = run_conversion(
        pipeline,
        config_path,
        plan,
        output_root=output_dir,
        ray_address=args.ray_address,
        cpus_per_task=args.cpus_per_task,
        workers=args.workers,
        tasks_per_job=args.tasks_per_job,
        max_retries=args.max_retries,
        preflight=not args.no_preflight,
    )

    summary_path = data_dir / "_run_summary.json"
    payload = write_run_summary(
        summary_path,
        pipeline,
        plan,
        results,
        wall_elapsed_s=(executor.last_summary or {}).get("wall_elapsed_s"),
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"[demo] summary: {summary_path}", flush=True)

    failures = [result for result in results if result.status == "failed"]
    if failures:
        print(f"[demo] {len(failures)} failed task(s):", flush=True)
        for result in failures[:5]:
            print(f"  - {result.task_id}: {result.error}", flush=True)
        if not args.allow_partial:
            return 1

    if args.aggregate:
        from ego2lerobot.convert.run import aggregate

        target = aggregate(pipeline, results, output_root=aggregate_dir)
        print(f"[demo] aggregated dataset: {target}", flush=True)

    print(
        "[demo] rerun the same command to see committed tasks skipped",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
