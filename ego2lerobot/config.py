"""Pipeline configuration: one YAML file describes reading, mapping and output.

Example::

    version: 1
    repo_id: ego/ego_demo
    output_root: /data/lerobot/ego_demo
    fps: 30
    robot_type: ego

    task:
      source: attrs.instruction
      default: ego manipulation

    features:
      observation.state:
        dtype: float32
        shape: [7]
        source: proprio.qpos
      action:
        dtype: float32
        shape: [7]
        source: proprio.cmd

    reader:
      type: h5
      arrays:
        proprio.qpos: /observations/qpos
        proprio.cmd: /action

    writer:
      use_videos: true
      image_writer_threads: 4

The file is reloaded inside every worker, so it only needs to live on a
filesystem visible to all nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .mapping.config import MappingConfig
from .readers import EpisodeReader, make_reader

_PIPELINE_KEYS = {
    "version",
    "fps",
    "robot_type",
    "task",
    "features",
    "reader",
    "repo_id",
    "output_root",
    "writer",
    "aggregate",
}


class PipelineConfigError(ValueError):
    """Raised when a pipeline configuration file is invalid."""


def _optional_mapping(value: Any, name: str) -> dict[str, Any]:
    """Return a shallow copy of an optional config section."""

    value = value or {}
    if not isinstance(value, dict):
        raise PipelineConfigError(f"{name} must be a mapping")
    return dict(value)


@dataclass(frozen=True)
class PipelineConfig:
    mapping: MappingConfig
    reader: EpisodeReader
    repo_id: str
    output_root: Path
    writer: dict[str, Any] = field(default_factory=dict)
    aggregate: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PipelineConfigError(f"cannot read config {path}: {exc}") from exc

        if path.suffix.lower() == ".json":
            import json

            payload = json.loads(text)
        else:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - depends on env
                raise PipelineConfigError("PyYAML is required for YAML configs") from exc
            payload = yaml.safe_load(text)

        if not isinstance(payload, dict):
            raise PipelineConfigError(f"config must contain a mapping at the top level: {path}")
        return cls.from_dict(payload, source_path=path)

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], *, source_path: Path | None = None
    ) -> "PipelineConfig":
        unknown = set(payload) - _PIPELINE_KEYS
        if unknown:
            raise PipelineConfigError(f"unknown top-level keys: {sorted(unknown)}")

        repo_id = payload.get("repo_id")
        if not isinstance(repo_id, str) or not repo_id.strip():
            raise PipelineConfigError("repo_id must be a non-empty string")

        output_root = payload.get("output_root", "./lerobot_output")
        if not isinstance(output_root, str) or not output_root.strip():
            raise PipelineConfigError("output_root must be a non-empty string")

        mapping_payload = {
            key: payload[key]
            for key in ("version", "fps", "robot_type", "task", "features")
            if key in payload
        }
        try:
            mapping = MappingConfig.from_dict(mapping_payload)
        except Exception as exc:
            raise PipelineConfigError(f"invalid mapping: {exc}") from exc

        try:
            reader = make_reader(payload.get("reader"))
        except Exception as exc:
            raise PipelineConfigError(f"invalid reader: {exc}") from exc

        writer = _optional_mapping(payload.get("writer"), "writer")
        aggregate = _optional_mapping(payload.get("aggregate"), "aggregate")
        unknown_aggregate = set(aggregate) - {"repo_id", "output_root"}
        if unknown_aggregate:
            raise PipelineConfigError(f"unknown aggregate keys: {sorted(unknown_aggregate)}")

        return cls(
            mapping=mapping,
            reader=reader,
            repo_id=repo_id.strip(),
            output_root=Path(output_root).expanduser(),
            writer=writer,
            aggregate=aggregate,
            source_path=source_path,
        )

    # ------------------------------------------------------------------- use

    @property
    def fps(self) -> int:
        return self.mapping.fps

    @property
    def aggregate_repo_id(self) -> str:
        return str(self.aggregate.get("repo_id") or f"{self.repo_id}-aggregated")

    @property
    def aggregate_output_root(self) -> Path:
        value = self.aggregate.get("output_root")
        return Path(value).expanduser() if value else self.output_root.parent / (
            self.output_root.name + "_aggregated"
        )

    def writer_kwargs(self) -> dict[str, Any]:
        """Keyword arguments forwarded to ``LeRobotDataset.create``."""

        allowed = {
            "use_videos",
            "tolerance_s",
            "image_writer_processes",
            "image_writer_threads",
            "video_backend",
            "batch_encoding_size",
            "vcodec",
            "metadata_buffer_size",
            "streaming_encoding",
            "encoder_queue_maxsize",
            "encoder_threads",
        }
        unknown = set(self.writer) - allowed
        if unknown:
            raise PipelineConfigError(f"unknown writer keys: {sorted(unknown)}")
        return dict(self.writer)

    def describe(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path) if self.source_path else None,
            "repo_id": self.repo_id,
            "output_root": str(self.output_root),
            "fps": self.mapping.fps,
            "robot_type": self.mapping.robot_type,
            "features": sorted(self.mapping.features),
            "reader": self.reader.describe(),
            "writer": self.writer_kwargs(),
            "aggregate": {
                "repo_id": self.aggregate_repo_id,
                "output_root": str(self.aggregate_output_root),
            },
        }


__all__ = ["PipelineConfig", "PipelineConfigError"]
