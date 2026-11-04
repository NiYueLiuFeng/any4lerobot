"""EgoVerse (Zarr v3) episode reader.

An EgoVerse *sub-dataset* is one Zarr v3 group directory::

    2025-09-20-17-42-51-000000/
      zarr.json                  # group attrs: embodiment/fps/task_name/features
      left.obs_ee_pose/          # (T, 7) float64
      right.obs_ee_pose/         # (T, 7) float64
      obs_rgb_timestamps_ns/     # (T,) int64
      images.front_1/            # (T,) variable-length JPEG bytes
      annotations/               # (N,) JSON strings (may be empty)

One sub-dataset is one episode, so discovery matches its unique anchor::

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
      annotations: false         # attach the "annotations" member to attrs
      start: 0
      stop: null
      stride: 1
      decode_images: true        # false hands back raw encoded bytes

Both mappings are ``source -> zarr member name``.  A source is the key used by
the YAML mapping (``features.*.source``), so renaming a source here never
renames the stored member.

Numeric members keep their stored dtype; add an ``astype`` transform in the
mapping when LeRobot expects something else.  Images are decoded one frame at
a time, so a long episode is never held in memory.  Group attributes are
attached to every record as ``record["attrs"]``.

``attrs.total_frames`` is advisory: the frame count comes from the selected
arrays, because real EgoVerse samples can disagree with it.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from .base import EpisodeReader

# Feature dtypes declared in the group's ``features`` attribute.
_IMAGE_DTYPES = {"jpeg", "jpg", "png", "image", "video", "webp"}
_JSON_DTYPES = {"json", "annotation_v1", "annotations"}

# Internal stream kinds returned by ``_classify``.
_NUMERIC = "numeric"
_IMAGE = "image"
_ANNOTATION = "annotation"


class EgoEgoverseReader(EpisodeReader):
    """Read one EgoVerse Zarr v3 sub-dataset as a stream of frames.

    The group is opened read-only and every member is read lazily by row, so
    memory use is bounded by one frame regardless of episode length.
    """

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
        """Validate and store the reader settings.

        ``arrays`` and ``images`` map mapping-layer source names to zarr
        member names.  ``None`` means "auto-detect from the group members";
        an empty mapping means "use none".
        """
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

    # --------------------------------------------------------------- contract

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "EgoEgoverseReader":
        """Build a reader from the ``reader:`` section of a pipeline config."""
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
        """Return a JSON-serialisable snapshot of the reader settings."""
        return {
            "type": self.kind,
            # None means "auto-detect"; {} means "use none".
            "arrays": None if self.arrays is None else dict(self.arrays),
            "images": None if self.images is None else dict(self.images),
            "annotations": self.annotations,
            "start": self.start,
            "stop": self.stop,
            "stride": self.stride,
            "decode_images": self.decode_images,
        }

    def accepts_input(self, path: str | Path) -> bool:
        """Only accept group roots, not Zarr array nodes.

        Every Zarr v3 array directory also contains a ``zarr.json``, so a
        ``**/zarr.json`` glob matches member nodes too.  Only the group root
        has ``"node_type": "group"``.
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
        """Raise if the episode cannot be read with these settings."""
        self._prepare(path)

    def iter_frames(self, path: str | Path) -> Iterator[dict[str, Any]]:
        """Yield ``{"frame_index", "attrs", <source>: value}`` once per frame."""
        group, streams, total = self._prepare(path)

        attrs = _to_jsonable(group.attrs)
        if self.annotations:
            loaded = _load_annotations(group)
            if loaded is not None:
                attrs["annotations"] = loaded

        stop = total if self.stop is None else min(self.stop, total)
        if self.start >= stop:
            raise ValueError(
                f"empty frame range: start={self.start} stop={stop} total={total}"
            )

        for index in range(self.start, stop, self.stride):
            record: dict[str, Any] = {"frame_index": index, "attrs": attrs}
            for source, (node, kind) in streams.items():
                if kind == _IMAGE:
                    value = node[index]
                    record[source] = (
                        _decode_image(value, source)
                        if self.decode_images
                        else _unwrap(value)
                    )
                else:
                    record[source] = _read_numeric_row(node, index)
            yield record

    # ---------------------------------------------------------------- helpers

    def _prepare(self, path: str | Path) -> tuple[Any, dict[str, tuple[Any, str]], int]:
        """Open the episode and return ``(group, streams, frame_count)``.

        Both ``validate()`` and ``iter_frames()`` go through here, so they
        always apply exactly the same checks.
        """
        base = _episode_dir(path)
        group = _open_group(base)
        streams = self._resolve(group)
        return group, streams, _frame_count(streams, base)

    def _resolve(self, group: Any) -> dict[str, tuple[Any, str]]:
        """Return ``{source: (array_node, kind)}`` for this episode.

        Explicit ``arrays``/``images`` win; a ``None`` section is filled by
        scanning the group: encoded/object members become images, every other
        array becomes a numeric stream, and ``annotations`` is skipped because
        it is not a per-frame stream.
        """
        raw_specs = group.attrs.get("features")
        specs = dict(raw_specs) if isinstance(raw_specs, Mapping) else {}

        detected: dict[str, dict[str, str]] = {_NUMERIC: {}, _IMAGE: {}}
        for name, node in group.members():
            if getattr(node, "shape", None) is None:
                continue  # nested group, not a feature
            kind = _classify(name, node, specs.get(name))
            if kind in detected:
                detected[kind][name] = name

        arrays = detected[_NUMERIC] if self.arrays is None else self.arrays
        images = detected[_IMAGE] if self.images is None else self.images

        streams = {
            source: (_member_node(group, member, source), _NUMERIC)
            for source, member in arrays.items()
        }
        for source, member in images.items():
            if source in streams:
                raise ValueError(f"reader source {source!r} is defined twice")
            streams[source] = (_member_node(group, member, source), _IMAGE)

        if not streams:
            raise ValueError(
                "no streams selected; add reader.arrays/reader.images or check the group members"
            )
        return streams


# --------------------------------------------------------------------- helpers


def _parse_mapping(name: str, raw: Any) -> dict[str, str] | None:
    """Validate a ``source -> member`` mapping from the config."""
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"reader.{name} must be a mapping or null")
    for source, member in raw.items():
        if not isinstance(source, str) or not source:
            raise ValueError(f"reader.{name} keys must be non-empty strings")
        if not isinstance(member, str) or not member:
            raise ValueError(f"reader.{name}[{source!r}] must be a non-empty string")
    return dict(raw)


def _episode_dir(path: str | Path) -> Path:
    """Return the episode directory for a group dir or its ``zarr.json``."""
    path = Path(path).expanduser()
    if path.is_file():
        if path.name != "zarr.json":
            raise ValueError(f"expected a zarr.json anchor or group directory: {path}")
        return path.parent
    return path


def _open_group(base: Path) -> Any:
    """Open ``base`` as a read-only Zarr group.

    The anchor is checked first so a wrong path fails with a clear message,
    and zarr is imported lazily to keep this module importable without it.
    """
    if not (base / "zarr.json").is_file():
        raise FileNotFoundError(f"not an EgoVerse Zarr group (missing zarr.json): {base}")
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError("zarr is required for the ego_egoverse reader") from exc
    return zarr.open_group(base, mode="r")


def _classify(name: str, node: Any, spec: Any) -> str:
    """Decide whether a member is an annotation, an image or numeric data."""
    dtype = ""
    if isinstance(spec, Mapping):
        dtype = str(spec.get("dtype") or "").lower()
    if name == "annotations" or dtype in _JSON_DTYPES:
        return _ANNOTATION
    if dtype in _IMAGE_DTYPES or name.startswith("images.") or ".images." in name:
        return _IMAGE
    if getattr(node, "dtype", None) is not None and node.dtype.kind == "O":
        # Unknown variable-length bytes: treat them as a lazy image stream.
        return _IMAGE
    return _NUMERIC


def _member_node(group: Any, member: str, source: str) -> Any:
    """Fetch a member array, failing with a readable error on a wrong name."""
    try:
        node = group[member]
    except KeyError as exc:
        raise KeyError(
            f"reader source {source!r}: zarr member {member!r} not found"
        ) from exc
    if getattr(node, "shape", None) is None:
        raise TypeError(f"reader source {source!r}: {member!r} is a group, not an array")
    return node


def _frame_count(streams: Mapping[str, tuple[Any, str]], base: Path) -> int:
    """Return the episode length, requiring all time-indexed streams to agree."""
    lengths = {}
    for source, (node, _) in streams.items():
        shape = getattr(node, "shape", None) or ()
        if shape:
            lengths[source] = int(shape[0])

    if not lengths:
        raise ValueError(f"no time-indexed streams in {base}")
    distinct = set(lengths.values())
    if len(distinct) > 1:
        detail = ", ".join(f"{source}:{length}" for source, length in lengths.items())
        raise ValueError(f"streams disagree on frame count in {base} ({detail})")

    total = next(iter(distinct))
    if total <= 0:
        raise ValueError(f"empty episode: {base}")
    return total


def _read_numeric_row(node: Any, index: int) -> np.ndarray:
    """Return one numeric row as a copy; a 0-d member is constant per frame."""
    value = node[()] if getattr(node, "ndim", None) == 0 else node[index]
    return np.array(value, copy=True)


def _unwrap(value: Any) -> Any:
    """Unwrap zarr 3's 0-d object array around variable-length bytes."""
    while isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, np.bytes_):
        value = bytes(value)
    return value


def _decode_image(value: Any, source: str) -> np.ndarray:
    """Decode one encoded frame (or an HWC/CHW array) to ``(H, W, 3)`` uint8."""
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


def _load_annotations(group: Any) -> list[Any] | None:
    """Load the ``annotations`` member as parsed JSON (``None`` if absent)."""
    try:
        node = group["annotations"]
    except KeyError:
        return None
    if getattr(node, "shape", None) is None:
        return None

    values: list[Any] = []
    for item in node[:]:
        item = _unwrap(item)
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                pass
        values.append(item)
    return values


def _to_jsonable(value: Any) -> Any:
    """Convert zarr attrs (numpy scalars/arrays, bytes) into JSON-safe values."""
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
