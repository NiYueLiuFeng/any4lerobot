"""EgoVerse (Zarr v3) episode reader.

An EgoVerse *sub-dataset* is one Zarr v3 group directory::

    2025-09-20-17-42-51-000000/
      zarr.json                  # group attrs: embodiment/fps/task_name/features
      left.obs_ee_pose/          # (T, 7) float64
      right.obs_ee_pose/         # (T, 7) float64
      obs_rgb_timestamps_ns/     # (T,) int64
      images.front_1/            # (T,) variable-length JPEG bytes
      annotations/               # (N,) JSON strings (may be empty)

One sub-dataset is one episode, so discovery should match its unique anchor::

    python -m ego2lerobot convert -c ego2lerobot/configs/ego_egoverse.yaml \\
        --root /data/egoverse --include '**/zarr.json'

Config::

    reader:
      type: ego_egoverse
      arrays:                    # optional; null means auto-detect every numeric member
        obs.ee_pose: right.obs_ee_pose
        obs.timestamps_ns: obs_rgb_timestamps_ns
      images:                    # optional; null means auto-detect image members
        cameras.front: images.front_1
      annotations: false         # parse the "annotations" member into attrs
      start: 0
      stop: null
      stride: 1
      decode_images: true        # false hands back raw encoded bytes

Both mappings are ``source -> zarr member name``.  A source is the key used by
the YAML mapping (``features.*.source``), so the two layers stay independent:
renaming a source here does not rename the member, and vice versa.

Numeric members keep their stored dtype; add an ``astype`` transform in the
mapping if LeRobot expects a different dtype.  Image members are decoded lazily
to ``(H, W, 3)`` uint8, one frame at a time, so a long episode is never held in
memory.  Group attributes are attached to every record as ``record["attrs"]``.

The group's ``attrs.total_frames`` is advisory: the frame count is taken from
the selected arrays, because real EgoVerse samples can disagree with it.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from .base import EpisodeReader

_IMAGE_DTYPES = {"jpeg", "jpg", "png", "image", "video", "webp"}
_JSON_DTYPES = {"json", "annotation_v1", "annotations"}

_NUMERIC = "numeric"
_IMAGE = "image"


@dataclass(frozen=True)
class _Stream:
    source: str
    member: str
    kind: str


class EgoEgoverseReader(EpisodeReader):
    kind = "ego_egoverse"

    def __init__(
        self,
        *,
        arrays: Mapping[str, str] | None = None,
        images: Mapping[str, str] | None = None,
        annotations: bool = False,
        start: int = 0,
        stop: int | None = None,
        stride: int = 1,
        decode_images: bool = True,
    ):
        if start < 0:
            raise ValueError("reader.start must be >= 0")
        if stop is not None and stop <= start:
            raise ValueError("reader.stop must be greater than reader.start")
        if stride < 1:
            raise ValueError("reader.stride must be >= 1")

        self.arrays = _parse_mapping("arrays", arrays)
        self.images = _parse_mapping("images", images)
        self.annotations = bool(annotations)
        self.start = int(start)
        self.stop = stop if stop is None else int(stop)
        self.stride = int(stride)
        self.decode_images = bool(decode_images)

        if self.arrays is not None and self.images is not None:
            overlap = set(self.arrays) & set(self.images)
            if overlap:
                raise ValueError(f"reader source names used twice: {sorted(overlap)}")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "EgoEgoverseReader":
        allowed = {
            "type",
            "arrays",
            "images",
            "annotations",
            "start",
            "stop",
            "stride",
            "decode_images",
        }
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"unknown ego_egoverse reader keys: {sorted(unknown)}")
        return cls(
            arrays=config.get("arrays"),
            images=config.get("images"),
            annotations=bool(config.get("annotations", False)),
            start=int(config.get("start", 0)),
            stop=None if config.get("stop") is None else int(config["stop"]),
            stride=int(config.get("stride", 1)),
            decode_images=bool(config.get("decode_images", True)),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "type": self.kind,
            # None means "auto-detect"; an empty mapping means "use none".
            "arrays": None if self.arrays is None else dict(self.arrays),
            "images": None if self.images is None else dict(self.images),
            "annotations": self.annotations,
            "start": self.start,
            "stop": self.stop,
            "stride": self.stride,
            "decode_images": self.decode_images,
        }

    # ------------------------------------------------------------------ checks

    def accepts_input(self, path: str | Path) -> bool:
        """Only accept group roots, not Zarr array nodes.

        Every Zarr v3 array directory also contains a ``zarr.json``, so the
        ``**/zarr.json`` discovery glob matches member nodes too.  A group
        root has ``"node_type": "group"``; an array node has ``"array"``.
        """

        path = Path(path)
        if path.name != "zarr.json":
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return payload.get("node_type") == "group"

    def validate(self, path: str | Path) -> None:
        base = self._base(path)
        _require_zarr(base)
        group = _open_group(base)

        streams = self._resolve_streams(group)
        lengths = self._lengths(group, streams)
        _check_lengths(lengths, base)

    # ------------------------------------------------------------------ frames

    def iter_frames(self, path: str | Path) -> Iterator[dict[str, Any]]:
        base = self._base(path)
        _require_zarr(base)
        group = _open_group(base)

        streams = self._resolve_streams(group)
        lengths = self._lengths(group, streams)
        total = _check_lengths(lengths, base)

        attrs = _to_jsonable(group.attrs)
        if self.annotations:
            annotations = _load_annotations(group)
            if annotations is not None:
                attrs["annotations"] = annotations

        stop = total if self.stop is None else min(self.stop, total)
        if self.start >= stop:
            raise ValueError(
                f"empty frame range: start={self.start} stop={stop} total={total}"
            )

        nodes = {stream.source: group[stream.member] for stream in streams}
        for index in range(self.start, stop, self.stride):
            record: dict[str, Any] = {"frame_index": index, "attrs": attrs}
            for stream in streams:
                node = nodes[stream.source]
                if stream.kind == _IMAGE:
                    value = node[index]
                    record[stream.source] = (
                        _decode_image(value, stream.source)
                        if self.decode_images
                        else _raw_image(value)
                    )
                elif getattr(node, "ndim", None) == 0:
                    # A scalar member is constant for every frame.
                    record[stream.source] = np.array(node[()], copy=True)
                else:
                    record[stream.source] = np.array(node[index], copy=True)
            yield record

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _base(path: str | Path) -> Path:
        """Accept either the episode directory or its ``zarr.json`` anchor."""

        path = Path(path).expanduser()
        if path.is_file():
            if path.name != "zarr.json":
                raise ValueError(f"expected a zarr.json anchor or group directory: {path}")
            return path.parent
        return path

    def _resolve_streams(self, group: Any) -> list[_Stream]:
        specs = _feature_specs(group)
        auto_arrays: dict[str, str] = {}
        auto_images: dict[str, str] = {}

        for name, node in group.members():
            if getattr(node, "shape", None) is None:
                continue  # nested group, not a feature
            kind = _classify(name, node, specs.get(name))
            if kind == "annotation":
                continue
            if kind == _IMAGE:
                auto_images[name] = name
            else:
                auto_arrays[name] = name

        arrays = auto_arrays if self.arrays is None else dict(self.arrays)
        images = auto_images if self.images is None else dict(self.images)

        streams: dict[str, _Stream] = {}
        for source, member in arrays.items():
            _require_array(group, member, source)
            streams[source] = _Stream(source, member, _NUMERIC)
        for source, member in images.items():
            if source in streams:
                raise ValueError(f"reader source {source!r} is defined twice")
            _require_array(group, member, source)
            streams[source] = _Stream(source, member, _IMAGE)

        if not streams:
            raise ValueError(
                "no streams selected; add reader.arrays/reader.images or check the group members"
            )
        return list(streams.values())

    @staticmethod
    def _lengths(group: Any, streams: list[_Stream]) -> dict[str, int | None]:
        lengths: dict[str, int | None] = {}
        for stream in streams:
            node = group[stream.member]
            shape = getattr(node, "shape", None) or ()
            lengths[stream.source] = int(shape[0]) if shape else None
        return lengths


# --------------------------------------------------------------------- helpers


def _parse_mapping(name: str, raw: Any) -> dict[str, str] | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"reader.{name} must be a mapping or null")
    parsed: dict[str, str] = {}
    for source, member in raw.items():
        if not isinstance(source, str) or not source:
            raise ValueError(f"reader.{name} keys must be non-empty strings")
        if not isinstance(member, str) or not member:
            raise ValueError(f"reader.{name}[{source!r}] must be a non-empty string")
        parsed[source] = member
    return parsed


def _require_zarr(base: Path) -> None:
    if not (base / "zarr.json").is_file():
        raise FileNotFoundError(f"not an EgoVerse Zarr group (missing zarr.json): {base}")
    try:
        import zarr  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError("zarr is required for the ego_egoverse reader") from exc


def _open_group(base: Path) -> Any:
    import zarr

    return zarr.open_group(base, mode="r")


def _feature_specs(group: Any) -> dict[str, Any]:
    specs = group.attrs.get("features")
    return dict(specs) if isinstance(specs, Mapping) else {}


def _classify(name: str, node: Any, spec: Any) -> str:
    dtype = ""
    if isinstance(spec, Mapping):
        dtype = str(spec.get("dtype") or "").lower()
    if name == "annotations" or dtype in _JSON_DTYPES:
        return "annotation"
    if dtype in _IMAGE_DTYPES or name.startswith("images.") or ".images." in name:
        return _IMAGE
    if getattr(node, "dtype", None) is not None and node.dtype.kind == "O":
        # Unknown variable-length bytes: treat them as a lazy image stream.
        return _IMAGE
    return _NUMERIC


def _require_array(group: Any, member: str, source: str) -> None:
    try:
        node = group[member]
    except KeyError as exc:
        raise KeyError(
            f"reader source {source!r}: zarr member {member!r} not found"
        ) from exc
    if getattr(node, "shape", None) is None:
        raise TypeError(f"reader source {source!r}: {member!r} is a group, not an array")


def _check_lengths(lengths: Mapping[str, int | None], base: Path) -> int:
    timed = {source: length for source, length in lengths.items() if length is not None}
    if not timed:
        raise ValueError(f"no time-indexed streams in {base}")
    distinct = set(timed.values())
    if len(distinct) > 1:
        detail = ", ".join(f"{source}:{length}" for source, length in timed.items())
        raise ValueError(f"streams disagree on frame count in {base} ({detail})")
    total = next(iter(distinct))
    if total <= 0:
        raise ValueError(f"empty episode: {base}")
    return total


def _unwrap(value: Any) -> Any:
    """Zarr 3 scalar selection of a vlen array wraps bytes in a 0-d object array."""

    while isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, np.bytes_):
        value = bytes(value)
    return value


def _decode_image(value: Any, source: str) -> np.ndarray:
    from PIL import Image

    value = _unwrap(value)
    if isinstance(value, Image.Image):
        return np.asarray(value.convert("RGB"), dtype=np.uint8)
    if isinstance(value, (bytes, bytearray)):
        image = Image.open(io.BytesIO(bytes(value)))
        image.load()
        return np.asarray(image.convert("RGB"), dtype=np.uint8)

    array = np.asarray(value)
    if array.ndim == 3 and array.shape[-1] in (1, 3, 4):
        return array.astype(np.uint8, copy=False)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        return np.transpose(array, (1, 2, 0)).astype(np.uint8, copy=False)
    raise TypeError(
        f"image stream {source!r}: cannot decode frame of type "
        f"{type(value).__name__} with shape {getattr(array, 'shape', None)}"
    )


def _raw_image(value: Any) -> Any:
    value = _unwrap(value)
    return value if isinstance(value, (bytes, bytearray, np.ndarray)) else np.asarray(value)


def _load_annotations(group: Any) -> list[Any] | None:
    try:
        node = group["annotations"]
    except KeyError:
        return None
    if getattr(node, "shape", None) is None:
        return None

    raw = node[:]
    if getattr(raw, "size", 0) == 0:
        return []
    values: list[Any] = []
    for item in raw:
        item = _unwrap(item)
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                pass
        values.append(item)
    return values


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


__all__ = ["EgoEgoverseReader"]
