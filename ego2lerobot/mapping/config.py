"""Config-driven mapping from raw episode records to LeRobot features.

The mapping file is the only place that knows how source fields are named.  It
is intentionally small and explicit::

    version: 1
    fps: 30
    robot_type: ego

    task:
      source: meta.instruction
      default: ego manipulation

    features:
      observation.state:
        dtype: float32
        shape: [7]
        source: proprio.qpos
        key_name: [x, y, z, qx, qy, qz, qw]   # per-dimension meaning

      observation.images.head:
        dtype: image
        shape: [480, 640, 3]
        source: cameras.head
        transforms:
          - op: as_image
            dtype: uint8

      action:
        dtype: float32
        shape: [7]
        source: [proprio.cmd, action]

A source may be a single dotted path or a list of candidates; the first one
found wins.  A feature without ``source`` and without ``default`` is rejected
at load time.  ``required: false`` turns a missing value into ``None`` instead
of an error.

``key_name`` names every dimension of the feature and is written to LeRobot's
``names`` metadata.  Numeric features take one name per element (nested lists
matching ``shape`` are flattened row-major); image/video features take one name
per axis (e.g. ``[height, width, channels]``).  The length is validated against
``shape`` at load time.

Two composition options extend that rule:

``combine: concat``
    Concatenate every entry of ``sources`` in order instead of picking the
    first one found.  Useful for ``action`` built from several streams.

``lookahead: N``
    Resolve the source(s) from the frame ``N`` steps ahead instead of the
    current one (``action[t] = state[t+1]``).  At the end of the episode the
    window is clamped to the last frame, so the last action reuses the last
    state.

The same shift can be written as a named transform, which reads better in a
config::

    transforms:
      - op: state_to_action    # action[t] = state[t+1], clamped at the end

``state_to_action`` is a temporal directive: the mapping layer applies it when
picking the source frame, so it cannot be called on a single value.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

_MISSING = object()

# LeRobot's feature dtypes.  numpy spellings are accepted and normalised.
_NUMPY_DTYPES = {
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bool",
}
_SPECIAL_DTYPES = {"image", "video", "string"}

#: Transforms that select *which frame* to read instead of editing a value.
#: They are resolved by :meth:`FeatureSpec.resolve`, not by ``_TRANSFORMS``.
_TEMPORAL_OPS = {"state_to_action"}


class MappingError(ValueError):
    """Raised for an invalid mapping file or an unresolvable field."""


@dataclass(frozen=True)
class TransformSpec:
    op: str
    params: dict[str, Any] = field(default_factory=dict)

    def apply(self, value: Any) -> Any:
        return apply_transform(self.op, value, self.params)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dtype: str
    shape: tuple[int, ...] | None
    sources: tuple[str, ...]
    default: Any = _MISSING
    required: bool = True
    transforms: tuple[TransformSpec, ...] = ()
    #: Raw LeRobot ``names`` metadata, passed through unchanged.
    names: Any = None
    #: Per-dimension names (``key_name``), flattened row-major and validated
    #: against ``shape``; written to LeRobot's ``names`` metadata.
    key_name: tuple[str, ...] | None = None
    #: ``"first"`` (first source found wins) or ``"concat"`` (join all sources).
    combine: str = "first"
    #: Resolve sources from this many frames ahead; clamped at the episode end.
    lookahead: int = 0

    def resolve(
        self,
        record: Mapping[str, Any],
        *,
        future: Sequence[Mapping[str, Any]] = (),
    ) -> Any:
        source_record = self._source_record(record, future)

        if self.combine == "concat":
            value = self._concat(source_record)
            used_source: str | None = ", ".join(self.sources)
        else:
            value = _MISSING
            used_source = None
            for source in self.sources:
                value = _lookup(source_record, source)
                if value is not _MISSING:
                    used_source = source
                    break

            if value is _MISSING:
                if self.default is not _MISSING:
                    value = self.default
                elif self.required:
                    tried = ", ".join(self.sources) or "(no source)"
                    raise MappingError(f"feature {self.name!r}: value not found; tried {tried}")
                else:
                    return None

        try:
            for transform in self.transforms:
                if transform.op in _TEMPORAL_OPS:
                    continue  # applied when selecting the source frame
                value = transform.apply(value)
        except Exception as exc:
            origin = f" from {used_source!r}" if used_source else ""
            raise MappingError(
                f"feature {self.name!r}: transform {transform.op!r} failed{origin}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return value

    @property
    def lookahead_frames(self) -> int:
        """Effective lookahead, including ``state_to_action`` directives."""

        frames = self.lookahead
        for transform in self.transforms:
            if transform.op in _TEMPORAL_OPS:
                frames = max(frames, int(transform.params.get("lookahead", 1)))
        return frames

    def _source_record(
        self, record: Mapping[str, Any], future: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        """Pick the frame a lookahead feature reads its sources from."""

        lookahead = self.lookahead_frames
        if lookahead <= 0:
            return record
        # Clamp at the end of the episode: late frames reuse the last frame.
        index = min(lookahead, len(future)) - 1
        return future[index] if index >= 0 else record

    def _concat(self, record: Mapping[str, Any]) -> Any:
        """Join every source into one array (axis 0), in declaration order."""

        np = _np()
        values = []
        for source in self.sources:
            item = _lookup(record, source)
            if item is _MISSING:
                raise MappingError(
                    f"feature {self.name!r}: source {source!r} is required for concat"
                )
            values.append(np.atleast_1d(item))
        return np.concatenate(values, axis=0)

    def to_lerobot(self) -> dict[str, Any]:
        # LeRobot validates ``value.shape != feature["shape"]``; numpy shapes are
        # tuples, so a list here would make every frame fail validation.
        feature: dict[str, Any] = {
            "dtype": self.dtype,
            "shape": tuple(self.shape or (1,)),
        }
        if self.key_name is not None:
            feature["names"] = list(self.key_name)
        elif self.names is not None:
            feature["names"] = self.names
        return feature


@dataclass(frozen=True)
class MappingConfig:
    fps: int
    robot_type: str | None
    features: Mapping[str, FeatureSpec]
    task: FeatureSpec
    version: int = 1

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, path: str | Path) -> "MappingConfig":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".json"}:
            payload = json.loads(text)
        else:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - depends on env
                raise MappingError("PyYAML is required for YAML mapping files") from exc
            payload = yaml.safe_load(text)
        if not isinstance(payload, dict):
            raise MappingError(f"mapping file must contain a mapping at the top level: {path}")
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MappingConfig":
        unknown = set(payload) - {"version", "fps", "robot_type", "features", "task"}
        if unknown:
            raise MappingError(f"unknown top-level keys: {sorted(unknown)}")

        fps = payload.get("fps")
        if not isinstance(fps, int) or isinstance(fps, bool) or fps <= 0:
            raise MappingError("fps must be a positive integer")

        raw_features = payload.get("features")
        if not isinstance(raw_features, dict) or not raw_features:
            raise MappingError("features must be a non-empty mapping")

        features: dict[str, FeatureSpec] = {}
        for name, raw in raw_features.items():
            if not isinstance(name, str) or not name:
                raise MappingError(f"feature name must be a non-empty string: {name!r}")
            features[name] = _parse_feature(name, raw)

        task = _parse_task(payload.get("task"))

        return cls(
            version=int(payload.get("version", 1)),
            fps=fps,
            robot_type=payload.get("robot_type"),
            features=features,
            task=task,
        )

    # ------------------------------------------------------------------- use

    def resolve(
        self,
        record: Mapping[str, Any],
        *,
        future: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        """Map one raw record to a LeRobot ``add_frame`` dictionary.

        ``future`` holds the records that follow ``record``; features with a
        ``lookahead`` read from it.  Streaming callers should pass at least
        :attr:`max_lookahead` records, and the empty tail is clamped.
        """

        frame = {
            name: spec.resolve(record, future=future)
            for name, spec in self.features.items()
        }
        frame["task"] = self.task.resolve(record, future=future)
        return frame

    @property
    def max_lookahead(self) -> int:
        """Largest lookahead any feature asks for, in frames."""

        return max((spec.lookahead_frames for spec in self.features.values()), default=0)

    def lerobot_features(self) -> dict[str, dict[str, Any]]:
        """Features argument for ``LeRobotDataset.create``."""

        return {name: spec.to_lerobot() for name, spec in self.features.items()}

    def missing_fields(self, record: Mapping[str, Any]) -> list[str]:
        """Diagnostics helper: features that would fail on ``record``."""

        missing: list[str] = []
        for name, spec in self.features.items():
            if spec.combine == "concat":
                # Every source is used, so a single missing one is fatal.
                if all(_lookup(record, source) is not _MISSING for source in spec.sources):
                    continue
                missing.append(name)
                continue
            if any(_lookup(record, source) is not _MISSING for source in spec.sources):
                continue
            if spec.default is not _MISSING or not spec.required:
                continue
            missing.append(name)
        return missing


def _parse_feature(name: str, raw: Any) -> FeatureSpec:
    if not isinstance(raw, dict):
        raise MappingError(f"feature {name!r}: expected a mapping, got {type(raw).__name__}")

    allowed = {
        "dtype",
        "shape",
        "source",
        "sources",
        "default",
        "required",
        "transforms",
        "names",
        "key_name",
        "combine",
        "lookahead",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise MappingError(f"feature {name!r}: unknown keys {sorted(unknown)}")

    dtype = str(raw.get("dtype", "")).lower()
    if dtype not in _NUMPY_DTYPES | _SPECIAL_DTYPES:
        raise MappingError(
            f"feature {name!r}: unsupported dtype {dtype!r}; "
            f"use one of {sorted(_NUMPY_DTYPES | _SPECIAL_DTYPES)}"
        )

    shape = raw.get("shape")
    if shape is not None:
        if not isinstance(shape, (list, tuple)) or not all(
            isinstance(dim, int) and dim > 0 for dim in shape
        ):
            raise MappingError(f"feature {name!r}: shape must be a list of positive integers")
        shape = tuple(shape)

    sources = raw.get("sources", raw.get("source"))
    if sources is None:
        sources = ()
    elif isinstance(sources, str):
        sources = (sources,)
    elif isinstance(sources, (list, tuple)):
        if not all(isinstance(item, str) and item for item in sources):
            raise MappingError(f"feature {name!r}: source entries must be non-empty strings")
        sources = tuple(sources)
    else:
        raise MappingError(f"feature {name!r}: source must be a string or a list of strings")

    if not sources and "default" not in raw and raw.get("required", True):
        raise MappingError(f"feature {name!r}: needs a source or a default value")

    combine = str(raw.get("combine", "first")).lower()
    if combine not in {"first", "concat"}:
        raise MappingError(
            f"feature {name!r}: combine must be 'first' or 'concat', got {combine!r}"
        )
    if combine == "concat" and not sources:
        raise MappingError(f"feature {name!r}: combine 'concat' needs at least one source")
    if combine == "concat" and "default" in raw:
        raise MappingError(f"feature {name!r}: combine 'concat' cannot use a default value")

    lookahead = raw.get("lookahead", 0)
    if not isinstance(lookahead, int) or isinstance(lookahead, bool) or lookahead < 0:
        raise MappingError(f"feature {name!r}: lookahead must be a non-negative integer")

    if "key_name" in raw and "names" in raw:
        raise MappingError(
            f"feature {name!r}: use either key_name or names, not both "
            "(key_name is validated and flattened)"
        )
    key_name = _parse_key_name(name, raw.get("key_name"), shape, dtype)

    transforms = tuple(_parse_transform(name, item) for item in raw.get("transforms", []) or [])
    for transform in transforms:
        if transform.op not in _TEMPORAL_OPS:
            continue
        offset = transform.params.get("lookahead", 1)
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 1:
            raise MappingError(
                f"feature {name!r}: {transform.op} lookahead must be a positive integer"
            )

    return FeatureSpec(
        name=name,
        dtype=dtype,
        shape=shape,
        sources=sources,
        default=raw.get("default", _MISSING),
        required=bool(raw.get("required", True)),
        transforms=transforms,
        names=raw.get("names"),
        key_name=key_name,
        combine=combine,
        lookahead=lookahead,
    )


def _parse_key_name(
    feature: str,
    raw: Any,
    shape: tuple[int, ...] | None,
    dtype: str,
) -> tuple[str, ...] | None:
    """Validate ``key_name`` and flatten it row-major.

    Numeric features need one name per element (``prod(shape)``); image/video
    features need one name per axis (``len(shape)``, e.g. height/width/channels).
    Nested lists matching ``shape`` are accepted and flattened.
    """
    if raw is None:
        return None
    if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
        raise MappingError(f"feature {feature!r}: key_name must be a list of strings")
    if shape is None:
        raise MappingError(f"feature {feature!r}: key_name requires an explicit shape")

    names: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        elif isinstance(value, str) and value:
            names.append(value)
        else:
            raise MappingError(
                f"feature {feature!r}: key_name entries must be non-empty strings"
            )

    walk(raw)
    if not names:
        raise MappingError(f"feature {feature!r}: key_name is empty")

    expected = len(shape) if dtype in {"image", "video"} else _prod(shape)
    if len(names) != expected:
        raise MappingError(
            f"feature {feature!r}: key_name has {len(names)} entries but "
            f"shape {list(shape)} needs {expected}"
        )
    return tuple(names)


def _prod(shape: Sequence[int]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def _task_spec(**kwargs: Any) -> FeatureSpec:
    """``task`` is a LeRobot metadata field, not a dataset feature."""

    return FeatureSpec(name="task", dtype="string", shape=(1,), **kwargs)


def _parse_task(raw: Any) -> FeatureSpec:
    if raw is None:
        return _task_spec(sources=(), default="convert", required=False)
    if isinstance(raw, str):
        return _task_spec(sources=(), default=raw, required=False)
    if isinstance(raw, dict):
        spec = _parse_feature("task", {"dtype": "string", "shape": [1], **raw})
        return _task_spec(
            sources=spec.sources,
            default=spec.default,
            required=spec.required,
            transforms=spec.transforms,
        )
    raise MappingError("task must be a string, a mapping, or omitted")


def _parse_transform(feature: str, raw: Any) -> TransformSpec:
    if isinstance(raw, dict):
        if "op" not in raw:
            raise MappingError(f"feature {feature!r}: every transform needs an 'op'")
        params = {key: value for key, value in raw.items() if key != "op"}
        op = str(raw["op"])
    elif isinstance(raw, str):
        op, params = raw, {}
    else:
        raise MappingError(f"feature {feature!r}: transform must be a string or a mapping")
    if op not in _TRANSFORMS and op not in _TEMPORAL_OPS:
        available = sorted(set(_TRANSFORMS) | _TEMPORAL_OPS)
        raise MappingError(
            f"feature {feature!r}: unknown transform {op!r}; available: {available}"
        )
    return TransformSpec(op=op, params=params)


# ------------------------------------------------------------------ lookup


def _lookup(record: Any, path: str) -> Any:
    """Resolve a source path against nested dicts/lists.

    A literal key wins over dotted traversal, so a reader that emits flat keys
    such as ``{"proprio.qpos": ...}`` works without nesting its records.
    Returns a private sentinel on any missing segment; ``None`` is a real value.
    """

    if isinstance(record, Mapping) and path in record:
        return record[path]

    current = record
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return _MISSING
            current = current[part]
        elif isinstance(current, (list, tuple)):
            try:
                index = int(part)
            except ValueError:
                return _MISSING
            if index < 0 or index >= len(current):
                return _MISSING
            current = current[index]
        else:
            # Support light objects (e.g. h5 attrs namespaces wrapped as objects).
            if not hasattr(current, part):
                return _MISSING
            current = getattr(current, part)
    return current


# --------------------------------------------------------------- transforms


def _np():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy ships with LeRobot
        raise MappingError("numpy is required for numeric transforms") from exc
    return np


def _identity(value: Any, **_: Any) -> Any:
    return value


def _astype(value: Any, dtype: str, **_: Any) -> Any:
    np = _np()
    return np.asarray(value, dtype=dtype)


def _reshape(value: Any, shape: Sequence[int], **_: Any) -> Any:
    np = _np()
    return np.asarray(value).reshape(tuple(shape))


def _scale(value: Any, factor: float = 1.0, offset: float = 0.0, **_: Any) -> Any:
    np = _np()
    return np.asarray(value, dtype=np.float32) * factor + offset


def _clip(value: Any, min: float | None = None, max: float | None = None, **_: Any) -> Any:
    np = _np()
    return np.clip(np.asarray(value), min, max)


def _slice(
    value: Any,
    start: int | None = None,
    stop: int | None = None,
    step: int | None = None,
    **_: Any,
) -> Any:
    np = _np()
    return np.asarray(value)[slice(start, stop, step)]


def _index(value: Any, index: int, **_: Any) -> Any:
    np = _np()
    return np.asarray(value)[index]


def _transpose(value: Any, axes: Sequence[int] | None = None, **_: Any) -> Any:
    np = _np()
    return np.transpose(np.asarray(value), axes=tuple(axes) if axes else None)


def _squeeze(value: Any, axis: int | None = None, **_: Any) -> Any:
    np = _np()
    return np.squeeze(np.asarray(value), axis=axis)


def _expand_dims(value: Any, axis: int = 0, **_: Any) -> Any:
    np = _np()
    return np.expand_dims(np.asarray(value), axis=axis)


def _to_bool(value: Any, **_: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _to_string(value: Any, **_: Any) -> str:
    return value if isinstance(value, str) else str(value)


def _as_image(value: Any, dtype: str = "uint8", channel_order: str = "HWC", **_: Any) -> Any:
    np = _np()
    array = np.asarray(value)
    if array.ndim not in (2, 3):
        raise ValueError(f"image must be 2D or 3D, got shape {array.shape}")
    if array.ndim == 3:
        if channel_order.upper() == "CHW":
            array = np.transpose(array, (1, 2, 0))
        if array.shape[2] not in (1, 3, 4):
            raise ValueError(f"image must have 1, 3 or 4 channels, got {array.shape[2]}")
    if np.issubdtype(array.dtype, np.floating):
        if float(np.nanmin(array)) >= 0.0 and float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255)
    return array.astype(dtype)


def apply_transform(op: str, value: Any, params: Mapping[str, Any] | None = None) -> Any:
    """Apply one named transform.  Exposed for tests and custom pipelines."""

    if op in _TEMPORAL_OPS:
        raise MappingError(
            f"transform {op!r} is a temporal directive; list it in a feature's "
            "'transforms' so the mapping layer can pick the source frame"
        )
    try:
        fn = _TRANSFORMS[op]
    except KeyError as exc:
        raise MappingError(f"unknown transform {op!r}") from exc
    return fn(value, **dict(params or {}))


_TRANSFORMS: dict[str, Callable[..., Any]] = {
    "identity": _identity,
    "astype": _astype,
    "reshape": _reshape,
    "scale": _scale,
    "clip": _clip,
    "slice": _slice,
    "index": _index,
    "transpose": _transpose,
    "squeeze": _squeeze,
    "expand_dims": _expand_dims,
    "to_bool": _to_bool,
    "to_string": _to_string,
    "as_image": _as_image,
}


__all__ = [
    "FeatureSpec",
    "MappingConfig",
    "MappingError",
    "TransformSpec",
    "apply_transform",
]
