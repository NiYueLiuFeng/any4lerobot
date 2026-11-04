"""EgoVerse (Zarr v3) demo.

An EgoVerse *sub-dataset* is a directory that contains a ``zarr.json`` group
with ``embodiment`` / ``fps`` / ``task_name`` / ``features`` attributes, plus one
Zarr array per feature::

    2025-09-20-17-42-51-000000/
      zarr.json                     # group attributes + feature specs
      left.obs_ee_pose/             # (T, 7) float64
      left.obs_keypoints/           # (T, 63) float64
      obs_rgb_timestamps_ns/        # (T,) int64
      images.front_1/               # (T,) variable-length JPEG bytes
      annotations/                  # (N,) JSON strings (may be empty)
      ...

:func:`generate_egoverse_demo` reads one sub-dataset and returns an
:class:`EgoVerseReader` that owns the loaded metadata and arrays.  Images are
kept lazy by default (a 2900-frame 640x480 dataset is ~2.6 GB decoded); pass
``load_images=True`` or call :meth:`EgoVerseReader.decode_image` when needed.

Run from the repository root::

    python -m ego2lerobot.demo.egoverse_demo

    # inspect one real sub-dataset without loading every image
    python -m ego2lerobot.demo.egoverse_demo \
        --sub-dataset 2025-09-20-17-42-51-000000 --max-frames 5 --show-frame 0
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Default location of the EgoVerse sample used by this demo.
DEFAULT_EGOVERSE_ROOT = Path("/mnt/cloudrobo/sfs_turbo_wam/cloudrobo-wam/tmp")
DEFAULT_SUB_DATASET = "2025-09-20-17-42-51-000000"

_IMAGE_DTYPES = {"jpeg", "jpg", "png", "image", "video"}
_JSON_DTYPES = {"json", "annotation_v1", "annotations"}


class EgoVerseReader:
    """In-memory view over one EgoVerse Zarr sub-dataset.

    Numeric features (``left.obs_ee_pose``, ``obs_head_pose``, ...) are loaded
    as trimmed numpy arrays.  Image features stay as Zarr arrays and are decoded
    frame by frame, unless ``load_images=True``.

    ``stride`` / ``max_frames`` select a window of the episode without reading
    the rest; ``frame_indices`` records the source frames that were kept.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        stride: int = 1,
        max_frames: int | None = None,
        load_images: bool = False,
    ):
        import zarr  # imported here so the rest of ego2lerobot has no hard dep

        if stride < 1:
            raise ValueError("stride must be >= 1")
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be > 0 or None")

        self.path = Path(path).expanduser().resolve()
        if not (self.path / "zarr.json").is_file():
            raise FileNotFoundError(f"not an EgoVerse Zarr group (missing zarr.json): {self.path}")

        self._group = zarr.open_group(self.path, mode="r")
        self.attrs: dict[str, Any] = dict(self._group.attrs)
        self.feature_specs: dict[str, Any] = dict(self.attrs.get("features") or {})

        self.warnings: list[str] = []
        self.arrays: dict[str, np.ndarray] = {}
        self.image_arrays: dict[str, Any] = {}
        self.annotations: list[Any] = []
        self.images: dict[str, np.ndarray] | None = None
        self.load_images = bool(load_images)

        self._collect_streams()
        self.frame_indices = self._select_frames(stride=stride, max_frames=max_frames)
        self._load_numeric_arrays()
        if self.load_images:
            self.images = {
                name: self._decode_all(name) for name in self.image_arrays
            }

    # --------------------------------------------------------------- loading

    def _collect_streams(self) -> None:
        """Split group members into numeric / image / annotation streams."""

        lengths: dict[str, int] = {}
        for name, node in self._group.members():
            shape = getattr(node, "shape", None)
            if shape is None:
                continue  # nested group, not a feature
            if _is_annotation(name, node, self.feature_specs.get(name)):
                self.annotations = _load_annotations(node)
                continue
            if _is_image(name, node, self.feature_specs.get(name)):
                self.image_arrays[name] = node
                lengths[name] = int(shape[0])
                continue
            if getattr(node, "dtype", None) is not None and node.dtype.kind == "O":
                # Unknown vlen stream: keep it as raw frames rather than failing.
                self.image_arrays[name] = node
                lengths[name] = int(shape[0])
                self.warnings.append(f"{name}: object dtype treated as lazy byte stream")
                continue
            if shape and np.prod(shape) > 0:
                lengths[name] = int(shape[0])

        if lengths:
            self.num_frames = min(lengths.values())
            disagree = {name: length for name, length in lengths.items() if length != self.num_frames}
            if disagree:
                self.warnings.append(
                    f"streams disagree on frame count; using {self.num_frames}: {disagree}"
                )
        else:
            self.num_frames = 0

        declared = self.attrs.get("total_frames")
        if declared is not None and int(declared) != self.num_frames:
            self.warnings.append(
                f"attrs.total_frames={declared} but arrays hold {self.num_frames} frames"
            )

    def _select_frames(self, *, stride: int, max_frames: int | None) -> np.ndarray:
        indices = np.arange(0, self.num_frames, stride, dtype=np.int64)
        if max_frames is not None:
            indices = indices[:max_frames]
        if indices.size == 0:
            raise ValueError(f"no frames selected from {self.path} (stride={stride})")
        return indices

    def _load_numeric_arrays(self) -> None:
        for name, node in self._group.members():
            if name in self.image_arrays:
                continue
            shape = getattr(node, "shape", None)
            if shape is None or name not in self._group.array_keys():
                continue
            if getattr(node, "dtype", None) is not None and node.dtype.kind == "O":
                continue
            if not shape or np.prod(shape) == 0:
                continue
            if node.ndim == 0:
                self.arrays[name] = np.asarray(node[()])
            else:
                self.arrays[name] = np.asarray(node[self.frame_indices])

    # ------------------------------------------------------------------ info

    @property
    def fps(self) -> int:
        return int(self.attrs.get("fps", 0) or 0)

    @property
    def embodiment(self) -> str | None:
        return self.attrs.get("embodiment")

    @property
    def task_name(self) -> str | None:
        return self.attrs.get("task_name")

    @property
    def task_description(self) -> str | None:
        return self.attrs.get("task_description")

    @property
    def feature_names(self) -> list[str]:
        return list(self.arrays) + list(self.image_arrays)

    def describe(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "embodiment": self.embodiment,
            "fps": self.fps,
            "task_name": self.task_name,
            "task_description": self.task_description,
            "total_frames": self.num_frames,
            "selected_frames": int(self.frame_indices.size),
            "frame_indices": [int(self.frame_indices[0]), int(self.frame_indices[-1])]
            if self.frame_indices.size
            else [],
            "features": {
                name: {
                    "shape": tuple(np.shape(values)),
                    "dtype": str(np.asarray(values).dtype),
                }
                for name, values in self.arrays.items()
            },
            "image_features": {
                name: {"shape": tuple(node.shape), "dtype": str(node.dtype)}
                for name, node in self.image_arrays.items()
            },
            "annotations": len(self.annotations),
            "warnings": list(self.warnings),
        }

    # ------------------------------------------------------------- accessing

    def __len__(self) -> int:
        return int(self.frame_indices.size)

    def __getitem__(self, position: int) -> dict[str, Any]:
        return self.frame_at(position, decode_images=False)

    def iter_frames(self, *, decode_images: bool = True) -> Iterator[dict[str, Any]]:
        """Yield one record per selected frame, in time order."""

        for position in range(len(self)):
            yield self.frame_at(position, decode_images=decode_images)

    def frame_at(self, position: int, *, decode_images: bool = False) -> dict[str, Any]:
        if position < 0:
            position += len(self)
        if not 0 <= position < len(self):
            raise IndexError(f"frame position {position} out of range (0..{len(self) - 1})")

        source_index = int(self.frame_indices[position])
        record: dict[str, Any] = {
            "frame_index": source_index,
            "embodiment": self.embodiment,
            "task_name": self.task_name,
            "task_description": self.task_description,
        }
        for name, values in self.arrays.items():
            record[name] = values[position]
        if "obs_rgb_timestamps_ns" in self.arrays:
            record["timestamp_ns"] = int(self.arrays["obs_rgb_timestamps_ns"][position])
        for name in self.image_arrays:
            if self.images is not None:
                record[name] = self.images[name][position]
            elif decode_images:
                record[name] = self.decode_image(name, position)
            else:
                # Keep the key present but hand back the raw encoded bytes.
                source_index = int(self.frame_indices[position])
                record[name] = self.image_arrays[name][source_index:source_index + 1][0]
        return record

    def decode_image(self, name: str, position: int) -> np.ndarray:
        """Decode one image feature frame into an ``(H, W, 3)`` uint8 array."""

        from PIL import Image

        if name not in self.image_arrays:
            raise KeyError(f"unknown image feature {name!r}; available: {sorted(self.image_arrays)}")
        if not 0 <= position < len(self):
            raise IndexError(f"frame position {position} out of range")

        node = self.image_arrays[name]
        source_index = int(self.frame_indices[position])
        value = node[source_index:source_index + 1][0]
        return _decode_image_value(value, name=name)

    def _decode_all(self, name: str) -> np.ndarray:
        node = self.image_arrays[name]
        raw = node[self.frame_indices]
        return np.stack([_decode_image_value(value, name=name) for value in raw])


# ----------------------------------------------------------------- helpers


def _is_image(name: str, node: Any, spec: Any) -> bool:
    dtype = str((spec or {}).get("dtype", "")).lower() if isinstance(spec, dict) else ""
    if dtype in _IMAGE_DTYPES:
        return True
    if name.startswith("images.") or ".images." in name:
        return True
    return False


def _is_annotation(name: str, node: Any, spec: Any) -> bool:
    if name == "annotations":
        return True
    dtype = str((spec or {}).get("dtype", "")).lower() if isinstance(spec, dict) else ""
    return dtype in _JSON_DTYPES


def _load_annotations(node: Any) -> list[Any]:
    raw = node[:]
    if getattr(raw, "size", 0) == 0:
        return []
    values = []
    for item in raw:
        if isinstance(item, (bytes, bytearray, np.bytes_)):
            item = bytes(item).decode("utf-8", errors="replace")
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                pass
        values.append(item)
    return values


def _decode_image_value(value: Any, *, name: str) -> np.ndarray:
    from PIL import Image

    # Zarr 3 scalar selection of a vlen-bytes array wraps the value in 0-d
    # object arrays; unwrap until a real value is reached.
    while isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, np.bytes_):
        value = bytes(value)

    if isinstance(value, (bytes, bytearray)):
        image = Image.open(io.BytesIO(bytes(value)))
        image.load()
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    if isinstance(value, Image.Image):
        return np.asarray(value.convert("RGB"), dtype=np.uint8)

    array = np.asarray(value)
    if array.ndim == 3 and array.shape[-1] == 3:
        return array.astype(np.uint8, copy=False)
    raise TypeError(f"image feature {name!r}: cannot decode frame of type {type(value).__name__}")


def _resolve_sub_dataset(sub_dataset: str | Path, root: str | Path | None) -> Path:
    path = Path(sub_dataset).expanduser()
    if path.is_dir():
        return path.resolve()
    if root is not None:
        candidate = Path(root).expanduser() / sub_dataset
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"sub-dataset {str(sub_dataset)!r} not found"
        + (f" under {root}" if root is not None else "")
    )


def generate_egoverse_demo(
    sub_dataset: str | Path = DEFAULT_SUB_DATASET,
    *,
    root: str | Path | None = DEFAULT_EGOVERSE_ROOT,
    stride: int = 1,
    max_frames: int | None = None,
    load_images: bool = False,
    verbose: bool = True,
) -> EgoVerseReader:
    """Read one EgoVerse sub-dataset and return an :class:`EgoVerseReader`.

    ``sub_dataset`` may be an absolute/relative directory, or a name resolved
    under ``root`` (defaults to the local EgoVerse sample root).

    The returned reader owns the metadata, numeric arrays and lazy image
    handles; the caller can inspect ``describe()``, iterate ``iter_frames()``
    or slice frames.  Images are decoded on demand unless ``load_images=True``.
    """

    path = _resolve_sub_dataset(sub_dataset, root)
    reader = EgoVerseReader(
        path,
        stride=stride,
        max_frames=max_frames,
        load_images=load_images,
    )
    if verbose:
        _print_summary(reader)
    return reader


def _print_summary(reader: EgoVerseReader, *, stream: Any = None) -> None:
    stream = stream or sys.stdout
    print(f"[egoverse] {reader.path}", file=stream)
    print(
        f"[egoverse] task={reader.task_name!r} embodiment={reader.embodiment!r} "
        f"fps={reader.fps} frames={reader.num_frames} selected={len(reader)}",
        file=stream,
    )
    for name, values in reader.arrays.items():
        print(f"  array  {name:<28} shape={tuple(np.shape(values))} dtype={np.asarray(values).dtype}", file=stream)
    for name, node in reader.image_arrays.items():
        print(f"  image  {name:<28} frames={node.shape[0]} dtype={node.dtype}", file=stream)
    print(f"  annotations={len(reader.annotations)}", file=stream)
    for warning in reader.warnings:
        print(f"  warning: {warning}", file=stream)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sub-dataset", default=DEFAULT_SUB_DATASET)
    parser.add_argument("--root", type=Path, default=DEFAULT_EGOVERSE_ROOT)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--load-images", action="store_true")
    parser.add_argument(
        "--show-frame",
        type=int,
        default=None,
        help="print one decoded frame (arrays + image shapes) and exit",
    )
    parser.add_argument("--json", action="store_true", help="print describe() as JSON")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    reader = generate_egoverse_demo(
        args.sub_dataset,
        root=args.root,
        stride=args.stride,
        max_frames=args.max_frames,
        load_images=args.load_images,
        verbose=not args.json,
    )

    if args.json:
        print(json.dumps(reader.describe(), ensure_ascii=False, indent=2))
        return 0

    if args.show_frame is not None:
        record = reader.frame_at(args.show_frame, decode_images=True)
        print(f"[egoverse] frame {record['frame_index']}:", flush=True)
        for name, value in record.items():
            if isinstance(value, np.ndarray):
                print(f"  {name:<28} shape={value.shape} dtype={value.dtype}")
            else:
                print(f"  {name:<28} {value!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
