"""Command line entry point.

Examples:
  python -m ego2lerobot discover -c ego2lerobot/configs/ego_h5.yaml --root /data/ego
  python -m ego2lerobot convert  -c ego2lerobot/configs/ego_h5.yaml --root /data/ego \\
      --ray-address auto --workers 16 --tasks-per-job 2 --aggregate
  python -m ego2lerobot aggregate -c ego2lerobot/configs/ego_h5.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from ego2lerobot.config import PipelineConfig, PipelineConfigError
from ego2lerobot.convert.run import (
    ConversionPlan,
    aggregate,
    plan_conversion,
    run_conversion,
    write_run_summary,
)


def _add_plan_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=Path,
        help="pipeline YAML (reader + mapping + output)",
    )
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help="raw data root; repeat for several roots",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="glob relative to each root; defaults to the reader's extension",
    )
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--follow-symlinks", action="store_true")
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--manifest", type=Path, default=None)


def _add_ray_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ray-address",
        default="auto",
        help="auto (existing cluster) | local | host:port",
    )
    parser.add_argument("--cpus-per-task", type=float, default=1.0)
    parser.add_argument("--gpus-per-task", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--tasks-per-job", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--no-preflight", action="store_true")
    parser.add_argument("--preflight-ffmpeg", action="store_true")
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=None,
        help="upload this directory with runtime_env instead of relying on PYTHONPATH",
    )
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def _build_plan(pipeline: PipelineConfig, args: argparse.Namespace) -> ConversionPlan:
    if not args.root:
        raise SystemExit("--root is required (repeat it for several data roots)")
    roots = args.root
    return plan_conversion(
        pipeline,
        roots,
        output_root=args.output_root or pipeline.output_root,
        include=args.include or None,
        exclude=args.exclude or None,
        follow_symlinks=args.follow_symlinks,
        max_depth=args.max_depth,
        max_files=args.max_files,
        manifest_path=args.manifest,
    )


def _cmd_discover(args: argparse.Namespace) -> int:
    pipeline = PipelineConfig.load(args.config)
    plan = _build_plan(pipeline, args)
    payload = plan.summary()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.manifest:
        print(f"[discover] manifest written to {args.manifest}")
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    pipeline = PipelineConfig.load(args.config)
    plan = _build_plan(pipeline, args)
    payload = plan.summary()
    print(
        f"[convert] {payload['files']} file(s) -> {payload['tasks']} task(s), "
        f"{payload['total_gib']} GiB",
        flush=True,
    )
    print(f"[convert] output root: {args.output_root or pipeline.output_root}", flush=True)

    if args.dry_run:
        for task in plan.tasks[:20]:
            print(f"  {task.task_id} <- {task.input_path}")
        if len(plan.tasks) > 20:
            print(f"  ... and {len(plan.tasks) - 20} more")
        return 0

    if not plan.tasks:
        print("[convert] nothing to do", flush=True)
        return 0

    output_root = Path(args.output_root or pipeline.output_root)
    results, executor = run_conversion(
        pipeline,
        args.config,
        plan,
        output_root=output_root,
        ray_address=args.ray_address,
        cpus_per_task=args.cpus_per_task,
        gpus_per_task=args.gpus_per_task,
        workers=args.workers,
        tasks_per_job=args.tasks_per_job,
        max_retries=args.max_retries,
        preflight=not args.no_preflight,
        preflight_ffmpeg=args.preflight_ffmpeg,
        working_dir=args.working_dir,
    )

    summary_path = output_root / "_run_summary.json"
    summary = write_run_summary(
        summary_path,
        pipeline,
        plan,
        results,
        wall_elapsed_s=(executor.last_summary or {}).get("wall_elapsed_s"),
    )
    print(f"[convert] summary written to {summary_path}", flush=True)

    failed = summary["summary"]["failed"]
    if failed and not args.allow_partial:
        print(f"[convert] {failed} task(s) failed; see {summary_path}", file=sys.stderr)
        return 1

    if args.aggregate:
        target = aggregate(pipeline, results, output_root=args.aggregate_output)
        if target is None:
            print("[convert] nothing to aggregate (no committed datasets)", file=sys.stderr)
            return 1 if failed else 0
        print(f"[convert] aggregated dataset: {target}", flush=True)

    return 0


def _cmd_aggregate(args: argparse.Namespace) -> int:
    pipeline = PipelineConfig.load(args.config)
    output_root = Path(args.output_root or pipeline.output_root)

    from ego2lerobot.common.types import TaskResult

    summary_path = args.summary or output_root / "_run_summary.json"
    results: list[TaskResult] = []
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        for item in payload.get("episode_order", []):
            results.append(
                TaskResult(
                    task_id=item["task_id"],
                    status="ok",
                    episodes=1,
                    frames=int(item.get("frames", 0)),
                    output_path=item["output_path"],
                )
            )
        for item in payload.get("skipped", []):
            if isinstance(item, dict):
                results.append(
                    TaskResult(
                        task_id=item.get("task_id", ""),
                        status="skipped",
                        frames=int(item.get("frames", 0)),
                        output_path=item.get("output_path", ""),
                    )
                )
            else:
                results.append(TaskResult(task_id=str(item), status="skipped"))
    else:
        print(
            f"[aggregate] {summary_path} not found; scanning {output_root} for committed datasets",
            file=sys.stderr,
        )
        for marker in sorted(output_root.rglob("_SUCCESS")):
            dataset_root = marker.parent
            if (dataset_root / "meta" / "info.json").is_file():
                results.append(
                    TaskResult(
                        task_id=str(dataset_root.relative_to(output_root)),
                        status="ok",
                        episodes=1,
                        output_path=str(dataset_root),
                    )
                )

    if args.engine == "ray-data":
        from ego2lerobot.convert.aggregate_raydata import (
            AggregateNotSupported,
            aggregate_with_ray_data,
        )

        try:
            target = aggregate_with_ray_data(
                pipeline,
                results,
                output_root=args.aggregate_output,
                repo_id=args.aggregate_repo_id,
                force=args.force,
                ray_address=args.ray_address,
                num_cpus=args.num_cpus,
            )
        except AggregateNotSupported as exc:
            print(f"[aggregate] ray-data backend not applicable: {exc}", file=sys.stderr)
            return 2
    else:
        target = aggregate(
            pipeline,
            results,
            output_root=args.aggregate_output,
            repo_id=args.aggregate_repo_id,
            force=args.force,
        )

    if target is None:
        print("[aggregate] no committed datasets found", file=sys.stderr)
        return 1
    print(f"[aggregate] wrote {target}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ego2lerobot",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    discover = sub.add_parser("discover", help="scan roots and write a manifest")
    _add_plan_args(discover)
    discover.set_defaults(func=_cmd_discover)

    convert = sub.add_parser("convert", help="run distributed conversion with Ray")
    _add_plan_args(convert)
    _add_ray_args(convert)
    convert.add_argument("--aggregate", action="store_true", help="aggregate after success")
    convert.add_argument("--aggregate-output", type=Path, default=None)
    convert.set_defaults(func=_cmd_convert)

    aggregate_parser = sub.add_parser("aggregate", help="merge committed per-task datasets")
    aggregate_parser.add_argument("-c", "--config", required=True, type=Path)
    aggregate_parser.add_argument("--output-root", type=Path, default=None)
    aggregate_parser.add_argument("--aggregate-output", type=Path, default=None)
    aggregate_parser.add_argument("--aggregate-repo-id", default=None)
    aggregate_parser.add_argument("--summary", type=Path, default=None)
    aggregate_parser.add_argument("--force", action="store_true")
    aggregate_parser.add_argument(
        "--engine",
        choices=["lerobot", "ray-data"],
        default="lerobot",
        help="lerobot: sequential aggregate_datasets; ray-data: parallel per-file rewrite",
    )
    aggregate_parser.add_argument("--ray-address", default="local")
    aggregate_parser.add_argument("--num-cpus", type=float, default=1.0)
    aggregate_parser.set_defaults(func=_cmd_aggregate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Older scopes used ``args.aggregate_output`` only in convert; make the
    # attribute available to both commands without duplicating parser code.
    if not hasattr(args, "aggregate_output"):
        args.aggregate_output = None

    try:
        return args.func(args)
    except PipelineConfigError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("[interrupted]", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
