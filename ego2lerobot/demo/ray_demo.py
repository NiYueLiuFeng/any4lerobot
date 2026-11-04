"""End-to-end demo for the Ray execution layer.

It generates a nested tree of synthetic episode JSON files, recursively
discovers them, converts them in parallel with Ray, and writes a summary.  The
data format is intentionally trivial: only the worker (``demo/worker.py``) needs
to change when you plug in h5 or mcap.

Run from the repository root::

    python -m ego2lerobot.demo.ray_demo --ray-address local --workers 4

Or directly::

    python ego2lerobot/demo/ray_demo.py --ray-address local --workers 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ego2lerobot.common import (  # noqa: E402
    ConversionTask,
    RayExecutor,
    RayExecutorConfig,
    summarize_results,
)
from ego2lerobot.demo.worker import process_batch  # noqa: E402


def generate_demo_data(
    raw_dir: Path,
    num_files: int,
    frames_per_file: int,
    corrupt: int,
    seed: int,
) -> None:
    """Create ``batch_XX/scene_Y/episode_NNNN.json`` files."""

    rng = random.Random(seed)
    raw_dir.mkdir(parents=True, exist_ok=True)

    for index in range(num_files):
        group = raw_dir / f"batch_{index // 8:02d}" / f"scene_{index % 3}"
        group.mkdir(parents=True, exist_ok=True)
        episode_path = group / f"episode_{index:04d}.json"

        if index < corrupt:
            episode_path.write_text("{ this is not valid json\n", encoding="utf-8")
            continue

        frames = [
            {
                "state": [round(rng.uniform(-1.0, 1.0), 4) for _ in range(7)],
                "action": [round(rng.uniform(-1.0, 1.0), 4) for _ in range(7)],
            }
            for _ in range(frames_per_file)
        ]
        episode_path.write_text(
            json.dumps({"episode_id": episode_path.stem, "frames": frames}),
            encoding="utf-8",
        )


def discover_episode_files(raw_dir: Path) -> list[Path]:
    """Recursively find files, with deterministic ordering and no symlink loops."""

    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(raw_dir, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames if not name.startswith("."))
        for filename in sorted(filenames):
            if filename.endswith(".json") and not filename.startswith("."):
                files.append(Path(dirpath) / filename)
    return sorted(files)


def build_tasks(files: list[Path], raw_dir: Path, output_dir: Path) -> list[ConversionTask]:
    """Map every raw file to a unique task id and output directory."""

    tasks: list[ConversionTask] = []
    for path in files:
        relative = path.relative_to(raw_dir)
        digest = hashlib.sha1(relative.as_posix().encode("utf-8")).hexdigest()[:10]
        task_id = f"{relative.with_suffix('').as_posix()}-{digest}"
        output_path = output_dir / relative.parent / f"{path.stem}-{digest}"
        tasks.append(
            ConversionTask(
                task_id=task_id,
                input_path=path,
                output_path=output_path,
                metadata={"relative_path": relative.as_posix()},
            )
        )
    return tasks


def write_aggregate(results, data_dir: Path, wall_elapsed_s: float | None = None) -> dict:
    """Demo-level aggregation.

    A real converter should call LeRobot's ``aggregate_datasets`` on the
    committed per-task datasets instead of writing this JSON summary.
    ``results`` is already in deterministic task order, so episode order is
    reproducible regardless of completion order.
    """

    summary = summarize_results(results)
    if wall_elapsed_s is not None:
        summary["wall_elapsed_s"] = wall_elapsed_s
    payload = {
        "summary": summary,
        "episode_order": [result.task_id for result in results if result.status == "ok"],
        "failures": [
            {"task_id": result.task_id, "error": result.error}
            for result in results
            if result.status == "failed"
        ],
    }
    (data_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/tmp/ego_ray_demo"))
    parser.add_argument("--num-files", type=int, default=24)
    parser.add_argument("--frames-per-file", type=int, default=50)
    parser.add_argument(
        "--compute-ms",
        type=int,
        default=100,
        help="simulated per-episode CPU work, in milliseconds",
    )
    parser.add_argument(
        "--corrupt",
        type=int,
        default=0,
        help="make the first N files invalid JSON to demo failure isolation",
    )
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--ray-address",
        default="local",
        help="local (start local Ray) | auto (existing cluster) | host:port",
    )
    parser.add_argument("--cpus-per-task", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument(
        "--tasks-per-job",
        type=int,
        default=1,
        help="number of ConversionTasks processed by one Ray task",
    )
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--no-preflight", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="exit 0 even if some tasks failed",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="delete and regenerate the synthetic raw data",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="delete outputs/logs before running, keeping raw data",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="generate data and exit (useful before a multi-node run on shared FS)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_files <= 0:
        raise SystemExit("--num-files must be > 0")
    if args.frames_per_file <= 0:
        raise SystemExit("--frames-per-file must be > 0")
    if args.compute_ms < 0:
        raise SystemExit("--compute-ms must be >= 0")
    if args.corrupt < 0:
        raise SystemExit("--corrupt must be >= 0")

    data_dir = args.data_dir.expanduser().resolve()
    raw_dir = data_dir / "raw"
    output_dir = data_dir / "output"
    log_dir = data_dir / "logs"
    probe_dir = data_dir / "probe"

    if args.clean:
        for path in (output_dir, log_dir, probe_dir):
            shutil.rmtree(path, ignore_errors=True)
        (data_dir / "summary.json").unlink(missing_ok=True)

    files = discover_episode_files(raw_dir) if raw_dir.exists() else []
    should_generate = args.regenerate or not files
    if should_generate:
        if raw_dir.exists() and args.regenerate:
            shutil.rmtree(raw_dir)
        print(f"[demo] generating {args.num_files} episode files under {raw_dir}", flush=True)
        generate_demo_data(
            raw_dir,
            num_files=args.num_files,
            frames_per_file=args.frames_per_file,
            corrupt=args.corrupt,
            seed=args.seed,
        )
    elif args.corrupt:
        print(
            "[demo] --corrupt ignored because data already exists; "
            "use --regenerate to recreate it",
            flush=True,
        )

    files = discover_episode_files(raw_dir)
    if not files:
        raise SystemExit(f"no .json files found under {raw_dir}")

    if args.generate_only:
        print(f"[demo] generated/found {len(files)} files; exiting (--generate-only)")
        return 0

    tasks = build_tasks(files, raw_dir, output_dir)
    print(f"[demo] discovered {len(tasks)} tasks under {raw_dir}", flush=True)
    print(f"[demo] outputs will be written under {output_dir}", flush=True)

    executor_config = RayExecutorConfig(
        address=args.ray_address,
        cpus_per_task=args.cpus_per_task,
        workers=args.workers,
        tasks_per_job=args.tasks_per_job,
        max_retries=args.max_retries,
        preflight=not args.no_preflight,
        preflight_imports=["ego2lerobot"],
        probe_dir=probe_dir,
        sample_input=files[0],
        log_dir=log_dir,
        progress=True,
    )
    executor = RayExecutor(executor_config)
    results = executor.run(
        process_batch,
        tasks,
        {"simulate_compute_ms": args.compute_ms},
    )

    payload = write_aggregate(
        results,
        data_dir,
        wall_elapsed_s=(executor.last_summary or {}).get("wall_elapsed_s"),
    )
    print(f"[demo] summary written to {data_dir / 'summary.json'}", flush=True)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)

    if payload["failures"]:
        print(f"[demo] {len(payload['failures'])} task(s) failed:", flush=True)
        for failure in payload["failures"][:10]:
            print(f"  - {failure['task_id']}: {failure['error']}", flush=True)
        if not args.allow_partial:
            print("[demo] rerun with --allow-partial to exit 0 despite failures")
            return 1
        print("[demo] --allow-partial set; continuing with partial output")
        return 0

    print("[demo] rerun the same command to see _SUCCESS-based skipping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
