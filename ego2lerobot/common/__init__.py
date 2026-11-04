"""Reusable infrastructure shared by ego dataset converters."""

from .discovery import (
    DiscoveredFile,
    DiscoveryConfig,
    build_output_path,
    discover_files,
    read_manifest,
    summarize_discovery,
    write_manifest,
)
from .ray_executor import (
    RayExecutionError,
    RayExecutor,
    RayExecutorConfig,
    RayPreflightError,
    build_runtime_env,
    resolve_address,
)
from .state import (
    commit_output,
    is_task_done,
    partial_path,
    prepare_output,
    safe_name,
    success_marker,
    write_failure,
    write_task_log,
)
from .types import ConversionTask, TaskResult, summarize_results

__all__ = [
    "ConversionTask",
    "DiscoveredFile",
    "DiscoveryConfig",
    "RayExecutionError",
    "RayExecutor",
    "RayExecutorConfig",
    "RayPreflightError",
    "TaskResult",
    "build_output_path",
    "build_runtime_env",
    "commit_output",
    "discover_files",
    "is_task_done",
    "partial_path",
    "prepare_output",
    "read_manifest",
    "resolve_address",
    "safe_name",
    "summarize_discovery",
    "summarize_results",
    "success_marker",
    "write_failure",
    "write_manifest",
    "write_task_log",
]
