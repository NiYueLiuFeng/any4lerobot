"""A small, reusable Ray execution layer for file/episode conversion tasks.

Design goals
------------
* Ray is an optional dependency: importing this module does not import Ray.
* The caller supplies a module-level ``worker_fn(batch, config) -> list[TaskResult]``.
* Tasks are submitted with a bounded in-flight window, never all at once.
* The driver receives only small :class:`TaskResult` objects.
* Preflight checks run once per cluster node so multi-node failures are found
  before the conversion starts.

Typical use::

    from ego2lerobot.common import ConversionTask, RayExecutor, RayExecutorConfig

    results = RayExecutor(RayExecutorConfig(address="auto")).run(
        decide_worker_fn,
        tasks,
        {"fps": 30},
    )
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import socket
import subprocess
import sys
import time
import uuid
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import ConversionTask, TaskResult, summarize_results

WorkerFn = Callable[[list[ConversionTask], dict[str, Any]], list[TaskResult] | TaskResult]


class RayExecutionError(RuntimeError):
    """Raised when the Ray execution layer cannot run a job."""


class RayPreflightError(RayExecutionError):
    """Raised when one or more cluster nodes fail environment checks."""


@dataclass
class RayExecutorConfig:
    """Configuration for :class:`RayExecutor`.

    ``address`` semantics:

    * ``None`` / ``"local"``: start a local Ray instance.
    * ``"auto"``: connect to an existing cluster (``ray start --head``).
    * ``"host:port"`` or ``"ray://host:10001"``: explicit address.
    """

    address: str | None = None

    cpus_per_task: float = 1.0
    gpus_per_task: float = 0.0
    workers: int = -1
    tasks_per_job: int = 1
    max_retries: int = 1

    # Distributing code.  Prefer installing the package on every node.  If that
    # is not possible, set ``working_dir`` to upload the project directory.
    working_dir: str | Path | None = None
    code_excludes: list[str] = field(
        default_factory=lambda: [
            ".git",
            ".venv",
            "data",
            "outputs",
            "_temp",
            "logs",
            "__pycache__",
            "*.partial",
            "*.pyc",
        ]
    )
    env_vars: dict[str, str] = field(
        default_factory=lambda: {
            "HDF5_USE_FILE_LOCKING": "FALSE",
            "HF_DATASETS_DISABLE_PROGRESS_BARS": "TRUE",
            "RAY_DEDUP_LOGS": "0",
        }
    )
    runtime_env: dict[str, Any] = field(default_factory=dict)

    # Preflight: one tiny task on every alive node.
    preflight: bool = True
    preflight_imports: list[str] = field(default_factory=lambda: ["ego2lerobot"])
    preflight_distributions: list[str] = field(default_factory=list)
    preflight_ffmpeg: bool = False
    preflight_timeout_s: float = 60.0
    probe_dir: str | Path | None = None
    sample_input: str | Path | None = None
    fail_fast_preflight: bool = True

    log_dir: str | Path | None = None
    progress: bool = True
    shutdown_after_run: bool = True

    def __post_init__(self) -> None:
        if self.cpus_per_task <= 0:
            raise ValueError("cpus_per_task must be > 0")
        if self.gpus_per_task < 0:
            raise ValueError("gpus_per_task must be >= 0")
        if self.tasks_per_job < 1:
            raise ValueError("tasks_per_job must be >= 1")
        if self.workers == 0 or self.workers < -1:
            raise ValueError("workers must be -1 or >= 1")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")


class RayExecutor:
    """Submit conversion tasks to a Ray cluster with bounded concurrency."""

    def __init__(self, config: RayExecutorConfig | None = None):
        self.config = config or RayExecutorConfig()
        self._ray: Any | None = None
        self._initialized_here = False
        self.last_summary: dict[str, Any] | None = None

    # ------------------------------------------------------------------ public

    def run(
        self,
        worker_fn: WorkerFn,
        tasks: Sequence[ConversionTask],
        config: dict[str, Any] | None = None,
    ) -> list[TaskResult]:
        """Run ``worker_fn`` over ``tasks`` and return results in task order.

        ``worker_fn`` is called as ``worker_fn(batch, config)`` where ``batch``
        is a list of tasks (length up to ``tasks_per_job``).  It must return a
        :class:`TaskResult` or a list of them, one per task in the batch.
        """

        tasks = list(tasks)
        if not tasks:
            return []

        self._validate_tasks(tasks)
        self._validate_worker_fn(worker_fn)

        ray = self._init_ray()
        started = time.perf_counter()
        try:
            self._print_cluster_info(ray)
            self._run_preflight(ray)
            results = self._run_tasks(ray, worker_fn, tasks, config or {})
        finally:
            if self._initialized_here and self.config.shutdown_after_run:
                self.close()

        self.last_summary = {
            **summarize_results(results),
            "wall_elapsed_s": round(time.perf_counter() - started, 3),
        }
        self._print_summary(results)
        return results

    def close(self) -> None:
        if self._ray is not None and self._initialized_here and self._ray.is_initialized():
            self._ray.shutdown()
        self._initialized_here = False

    # --------------------------------------------------------------- Ray setup

    def _init_ray(self) -> Any:
        try:
            import ray
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RayExecutionError(
                'Ray is not installed. Install it with: pip install -U "ray[default]"'
            ) from exc

        self._ray = ray
        if ray.is_initialized():
            return ray

        address = _resolve_address(self.config.address)
        runtime_env = build_runtime_env(self.config)
        ray.init(
            address=address,
            runtime_env=runtime_env,
            ignore_reinit_error=True,
        )
        self._initialized_here = True
        return ray

    def _resolve_workers(self, ray: Any) -> int:
        if self.config.workers > 0:
            return self.config.workers
        cpus = float(ray.cluster_resources().get("CPU", 1.0))
        return max(1, int(cpus // self.config.cpus_per_task))

    def _print_cluster_info(self, ray: Any) -> None:
        nodes = [node for node in ray.nodes() if node.get("Alive")]
        resources = ray.cluster_resources()
        workers = self._resolve_workers(ray)
        gpus = resources.get("GPU", 0)
        print(
            f"[ray] nodes={len(nodes)} CPUs={resources.get('CPU', 0)} "
            f"GPUs={gpus} workers={workers} "
            f"cpus_per_task={self.config.cpus_per_task}",
            flush=True,
        )

    def _validate_tasks(self, tasks: Sequence[ConversionTask]) -> None:
        seen: set[str] = set()
        for task in tasks:
            if not task.task_id:
                raise ValueError("ConversionTask.task_id must not be empty")
            if task.task_id in seen:
                raise ValueError(f"duplicate task_id: {task.task_id}")
            seen.add(task.task_id)

    @staticmethod
    def _validate_worker_fn(worker_fn: WorkerFn) -> None:
        if not callable(worker_fn):
            raise TypeError("worker_fn must be callable")
        if getattr(worker_fn, "__module__", None) == "__main__":
            warnings.warn(
                "worker_fn is defined in __main__; multi-node workers may not be "
                "able to import it. Put it in an importable module.",
                stacklevel=3,
            )

    # ---------------------------------------------------------------- preflight

    def _run_preflight(self, ray: Any) -> None:
        if not self.config.preflight:
            return

        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        payload = {
            "imports": list(self.config.preflight_imports),
            "distributions": list(self.config.preflight_distributions),
            "check_ffmpeg": self.config.preflight_ffmpeg,
            "probe_dir": str(self.config.probe_dir) if self.config.probe_dir else None,
            "sample_input": str(self.config.sample_input) if self.config.sample_input else None,
        }

        preflight_remote = ray.remote(_preflight_checks)
        refs = []
        hosts: list[str] = []
        for node in ray.nodes():
            if not node.get("Alive"):
                continue
            node_id = node.get("NodeID")
            if not node_id:
                continue
            host = (
                node.get("NodeManagerHostname")
                or node.get("NodeManagerAddress")
                or node.get("NodeName")
                or node_id[:8]
            )
            hosts.append(host)
            # Ray >= 2.5x returns a one-shot FuncWrapper from .options(); pass
            # every option in a single call instead of chaining .options().
            refs.append(
                preflight_remote.options(
                    num_cpus=0,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=False
                    ),
                ).remote(payload)
            )

        if not refs:
            raise RayPreflightError("no alive Ray nodes found")

        try:
            reports = ray.get(refs, timeout=self.config.preflight_timeout_s)
        except Exception as exc:
            for ref in refs:
                try:
                    ray.cancel(ref, force=True)
                except Exception:
                    pass
            raise RayPreflightError(f"preflight did not complete: {exc}") from exc

        errors: list[str] = []
        pythons: set[str] = set()
        versions_by_dist: dict[str, set[str]] = {
            dist: set() for dist in self.config.preflight_distributions
        }
        for host, report in zip(hosts, reports):
            pythons.add(report.get("python", "unknown"))
            for dist, version in report.get("versions", {}).items():
                versions_by_dist.setdefault(dist, set()).add(str(version))
            for error in report.get("errors", []):
                errors.append(f"{host}: {error}")
            status = "ok" if report.get("ok") else "FAILED"
            print(
                f"[ray] preflight {host}: {status} "
                f"python={report.get('python')} "
                f"ffmpeg={report.get('ffmpeg') or 'n/a'}",
                flush=True,
            )

        if len(pythons) > 1:
            errors.append(f"Python major.minor differs across nodes: {sorted(pythons)}")
        for dist, versions in versions_by_dist.items():
            if len(versions) > 1:
                errors.append(f"{dist} version differs across nodes: {sorted(versions)}")

        if errors:
            message = "Ray preflight failed:\n  - " + "\n  - ".join(errors)
            if self.config.fail_fast_preflight:
                raise RayPreflightError(message)
            print(f"[ray] WARNING: {message}", flush=True)

    # -------------------------------------------------------------- task loop

    def _run_tasks(
        self,
        ray: Any,
        worker_fn: WorkerFn,
        tasks: list[ConversionTask],
        config: dict[str, Any],
    ) -> list[TaskResult]:
        remote_fn = ray.remote(worker_fn).options(
            num_cpus=self.config.cpus_per_task,
            num_gpus=self.config.gpus_per_task,
            max_retries=self.config.max_retries,
            retry_exceptions=True,
        )

        payload = _build_worker_payload(config, self.config)
        config_ref = ray.put(payload)

        workers = self._resolve_workers(ray)
        window = max(1, workers * 2)
        batches = list(_batched(tasks, self.config.tasks_per_job))

        results_by_id: dict[str, TaskResult] = {}
        pending: dict[Any, list[ConversionTask]] = {}
        batch_iter = iter(batches)

        try:
            while True:
                while len(pending) < window:
                    batch = next(batch_iter, None)
                    if batch is None:
                        break
                    ref = remote_fn.remote(batch, config_ref)
                    pending[ref] = batch

                if not pending:
                    break

                ready, _ = ray.wait(list(pending), num_returns=1)
                for ref in ready:
                    batch = pending.pop(ref)
                    for result in self._collect_batch(ray, ref, batch):
                        results_by_id[result.task_id] = result
                self._print_progress(results_by_id, len(tasks), len(pending))
        except KeyboardInterrupt:
            print("\n[ray] interrupted; cancelling pending tasks...", flush=True)
            for ref in pending:
                try:
                    ray.cancel(ref, force=True)
                except Exception:
                    pass
            raise

        # Defensive: guarantee one result per task and preserve input order.
        ordered: list[TaskResult] = []
        for task in tasks:
            result = results_by_id.get(task.task_id)
            if result is None:
                result = TaskResult(
                    task_id=task.task_id,
                    status="failed",
                    output_path=str(task.output_path),
                    error="task was submitted but no result was returned",
                )
            ordered.append(result)
        if self.config.progress and tasks:
            print(flush=True)
        return ordered

    def _collect_batch(self, ray: Any, ref: Any, batch: list[ConversionTask]) -> list[TaskResult]:
        try:
            value = ray.get(ref)
        except Exception as exc:
            message = _short_error(exc)
            return [
                TaskResult(
                    task_id=task.task_id,
                    status="failed",
                    output_path=str(task.output_path),
                    error=f"Ray task failed: {message}",
                )
                for task in batch
            ]

        if isinstance(value, TaskResult):
            value = [value]
        if not isinstance(value, (list, tuple)):
            message = (
                f"worker returned {type(value).__name__}, expected TaskResult "
                "or a list of TaskResult"
            )
            return [
                TaskResult(
                    task_id=task.task_id,
                    status="failed",
                    output_path=str(task.output_path),
                    error=message,
                )
                for task in batch
            ]

        returned = {result.task_id: result for result in value if isinstance(result, TaskResult)}
        collected: list[TaskResult] = []
        for task in batch:
            result = returned.get(task.task_id)
            if result is None:
                result = TaskResult(
                    task_id=task.task_id,
                    status="failed",
                    output_path=str(task.output_path),
                    error="worker returned no result for this task",
                )
            collected.append(result)
        return collected

    def _print_progress(
        self, results_by_id: dict[str, TaskResult], total: int, pending: int
    ) -> None:
        if not self.config.progress:
            return
        ok = sum(1 for result in results_by_id.values() if result.status == "ok")
        skipped = sum(1 for result in results_by_id.values() if result.status == "skipped")
        failed = sum(1 for result in results_by_id.values() if result.status == "failed")
        print(
            f"\r[ray] {len(results_by_id)}/{total} "
            f"ok={ok} skipped={skipped} failed={failed} in_flight={pending}",
            end="",
            flush=True,
        )

    def _print_summary(self, results: list[TaskResult]) -> None:
        summary = self.last_summary or summarize_results(results)
        print(
            f"[ray] done tasks={summary['total']} ok={summary['ok']} "
            f"skipped={summary['skipped']} failed={summary['failed']} "
            f"episodes={summary['episodes']} frames={summary['frames']} "
            f"task_time={summary['task_elapsed_s']}s "
            f"wall={summary.get('wall_elapsed_s', 'n/a')}s",
            flush=True,
        )
        failures = [result for result in results if result.status == "failed"]
        if failures:
            print(f"[ray] {len(failures)} failed task(s):", flush=True)
            for result in failures[:10]:
                print(f"  - {result.task_id}: {result.error}", flush=True)
            if len(failures) > 10:
                print(f"  ... and {len(failures) - 10} more", flush=True)


# --------------------------------------------------------------------- helpers


def _resolve_address(address: str | None) -> str | None:
    if address is None:
        return None
    if address.strip().lower() in {"", "local", "none"}:
        return None
    return address


def _build_pythonpath() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    candidates: list[str] = [str(repo_root), str(Path.cwd())]
    candidates.extend(os.environ.get("PYTHONPATH", "").split(os.pathsep))
    paths: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            path = Path(candidate).expanduser().resolve()
        except OSError:
            continue
        if path.exists() and str(path) not in paths:
            paths.append(str(path))
    return os.pathsep.join(paths)


def build_runtime_env(config: RayExecutorConfig) -> dict[str, Any]:
    """Build a Ray runtime_env without importing Ray."""

    env_vars = {key: str(value) for key, value in config.env_vars.items()}
    runtime_env: dict[str, Any] = {"env_vars": env_vars}

    if config.working_dir is not None:
        working_dir = Path(config.working_dir).expanduser().resolve()
        if not working_dir.is_dir():
            raise FileNotFoundError(f"working_dir does not exist: {working_dir}")
        runtime_env["working_dir"] = str(working_dir)
        runtime_env["excludes"] = list(config.code_excludes)
    else:
        pythonpath = _build_pythonpath()
        if pythonpath:
            runtime_env["env_vars"]["PYTHONPATH"] = pythonpath

    for key, value in config.runtime_env.items():
        if key == "env_vars":
            runtime_env.setdefault("env_vars", {}).update({k: str(v) for k, v in value.items()})
        elif key == "working_dir":
            runtime_env["working_dir"] = str(value)
        else:
            runtime_env[key] = value

    return runtime_env


def _build_worker_payload(config: dict[str, Any], executor_config: RayExecutorConfig) -> dict[str, Any]:
    payload = dict(config)
    executor_meta = payload.get("_executor")
    if executor_meta is not None and not isinstance(executor_meta, dict):
        raise TypeError("config['_executor'] must be a dict")
    executor_meta = dict(executor_meta or {})
    executor_meta.update(
        {
            "log_dir": str(executor_config.log_dir) if executor_config.log_dir else None,
            "cpus_per_task": executor_config.cpus_per_task,
            "tasks_per_job": executor_config.tasks_per_job,
            "max_retries": executor_config.max_retries,
        }
    )
    payload["_executor"] = executor_meta
    return payload


def _batched(items: Sequence[ConversionTask], size: int) -> Iterable[list[ConversionTask]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def _short_error(exc: BaseException, limit: int = 4000) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= limit else text[:limit] + "..."


# NOTE: keep this function at module level so Ray can serialize it by reference.
def _preflight_checks(payload: dict[str, Any]) -> dict[str, Any]:
    """Run on one node; must not import Ray."""

    errors: list[str] = []
    versions: dict[str, str] = {}

    for module in payload.get("imports", []):
        try:
            importlib.import_module(module)
        except Exception as exc:
            errors.append(f"import {module}: {type(exc).__name__}: {exc}")

    for distribution in payload.get("distributions", []):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except Exception as exc:
            errors.append(f"distribution {distribution}: {type(exc).__name__}: {exc}")

    ffmpeg: str | None = None
    if payload.get("check_ffmpeg"):
        try:
            completed = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if completed.returncode != 0:
                errors.append(f"ffmpeg exited with {completed.returncode}")
            else:
                ffmpeg = (completed.stdout.splitlines() or ["unknown"])[0]
        except FileNotFoundError:
            errors.append("ffmpeg not found")
        except Exception as exc:
            errors.append(f"ffmpeg check: {type(exc).__name__}: {exc}")

    probe_dir = payload.get("probe_dir")
    if probe_dir:
        try:
            directory = Path(probe_dir)
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / f".ray_probe_{socket.gethostname()}_{uuid.uuid4().hex}"
            probe.write_text("ok", encoding="utf-8")
            if probe.read_text(encoding="utf-8") != "ok":
                errors.append(f"shared filesystem read mismatch: {probe}")
            probe.unlink(missing_ok=True)
        except Exception as exc:
            errors.append(f"probe_dir {probe_dir}: {type(exc).__name__}: {exc}")

    sample_input = payload.get("sample_input")
    if sample_input and not os.access(sample_input, os.R_OK):
        errors.append(f"sample input is not readable: {sample_input}")

    # ``platform`` is imported for the full version in the report.
    return {
        "host": socket.gethostname(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "python_full": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        "ffmpeg": ffmpeg,
        "errors": errors,
        "ok": not errors,
    }


__all__ = [
    "RayExecutionError",
    "RayExecutor",
    "RayExecutorConfig",
    "RayPreflightError",
    "WorkerFn",
    "build_runtime_env",
]
