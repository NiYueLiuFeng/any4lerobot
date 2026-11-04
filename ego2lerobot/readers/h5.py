"""HDF5 episode reader.

Expected layout (configurable, this is only an example)::

    /observations/qpos          (T, D)    float32
    /observations/images/head   (T, H, W, 3)
    /action                     (T, D)

    reader:
      type: h5
      time_axis: 0
      arrays:
        proprio.qpos: /observations/qpos
        proprio.cmd: /action
        cameras.head: /observations/images/head
      attrs: true

The time axis can differ per dataset::

    arrays:
      proprio.qpos:
        path: /qpos
        axis: 1
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from .base import EpisodeReader

_H5_SUFFIXES = {".h5", ".hdf5", ".hdf", ".he5"}


@dataclass(frozen=True)
class H5Array:
    source: str
    path: str
    axis: int | None = None  # None -> use reader default


class H5Reader(EpisodeReader):
    kind = "h5"

    def __init__(
        self,
        arrays: Mapping[str, Any],
        *,
        time_axis: int = 0,
        attrs: bool = True,
        start: int = 0,
        stop: int | None = None,
        stride: int = 1,
    ):
        if not arrays:
            raise ValueError("reader.arrays must not be empty for an h5 reader")
        if stride < 1:
            raise ValueError("reader.stride must be >= 1")
        if start < 0:
            raise ValueError("reader.start must be >= 0")
        if stop is not None and stop <= start:
            raise ValueError("reader.stop must be greater than reader.start")

        self.arrays: list[H5Array] = []
        for source, raw in arrays.items():
            if not isinstance(source, str) or not source:
                raise ValueError(f"reader.arrays key must be a non-empty string: {source!r}")
            if isinstance(raw, str):
                self.arrays.append(H5Array(source=source, path=raw))
            elif isinstance(raw, dict):
                path = raw.get("path", raw.get("dataset"))
                if not isinstance(path, str) or not path:
                    raise ValueError(f"reader.arrays[{source!r}] needs a 'path' string")
                axis = raw.get("axis")
                if axis is not None and not isinstance(axis, int):
                    raise ValueError(f"reader.arrays[{source!r}].axis must be an integer")
                self.arrays.append(H5Array(source=source, path=path, axis=axis))
            else:
                raise ValueError(
                    f"reader.arrays[{source!r}] must be a dataset path or a mapping"
                )
        self.time_axis = int(time_axis)
        self.attrs = bool(attrs)
        self.start = int(start)
        self.stop = stop
        self.stride = int(stride)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "H5Reader":
        allowed = {"type", "arrays", "time_axis", "attrs", "start", "stop", "stride"}
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"unknown h5 reader keys: {sorted(unknown)}")
        return cls(
            config.get("arrays") or {},
            time_axis=int(config.get("time_axis", 0)),
            attrs=bool(config.get("attrs", True)),
            start=int(config.get("start", 0)),
            stop=None if config.get("stop") is None else int(config["stop"]),
            stride=int(config.get("stride", 1)),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "type": self.kind,
            "time_axis": self.time_axis,
            "start": self.start,
            "stop": self.stop,
            "stride": self.stride,
            "attrs": self.attrs,
            "arrays": {
                item.source: {"path": item.path, "axis": item.axis} for item in self.arrays
            },
        }

    def validate(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix.lower() not in _H5_SUFFIXES:
            raise ValueError(f"not an HDF5 file (suffix {path.suffix!r}): {path}")
        import h5py

        with h5py.File(path, "r") as handle:
            datasets = self._datasets(handle, h5py, path)
            for item, (_, dataset, axis_override) in zip(self.arrays, datasets):
                axis = self.time_axis if axis_override is None else axis_override
                if dataset.ndim > 0 and (axis >= dataset.ndim or axis < -dataset.ndim):
                    raise IndexError(
                        f"{item.path!r} has shape {dataset.shape}; axis {axis} is invalid"
                    )

    def _datasets(self, handle: Any, h5py: Any, path: Path) -> list[tuple[str, Any, int | None]]:
        """Resolve every configured dataset, with a readable error on a miss."""

        resolved: list[tuple[str, Any, int | None]] = []
        for item in self.arrays:
            if item.path not in handle:
                raise KeyError(f"dataset {item.path!r} not found in {path}")
            dataset = handle[item.path]
            if not isinstance(dataset, h5py.Dataset):
                raise TypeError(f"{item.path!r} in {path} is a group, not a dataset")
            resolved.append((item.source, dataset, item.axis))
        return resolved

    def iter_frames(self, path: str | Path) -> Iterator[dict[str, Any]]:
        import h5py
        import numpy as np

        path = Path(path)
        with h5py.File(path, "r") as handle:
            datasets = self._datasets(handle, h5py, path)
            frame_attrs = _decode_attrs(handle.attrs) if self.attrs else None

            lengths: list[int] = []
            for _, dataset, axis_override in datasets:
                if dataset.ndim == 0:
                    continue
                axis = self.time_axis if axis_override is None else axis_override
                if axis >= dataset.ndim or axis < -dataset.ndim:
                    raise IndexError(
                        f"time axis {axis} is invalid for dataset with shape {dataset.shape}"
                    )
                lengths.append(int(dataset.shape[axis]))
            if not lengths:
                # All scalar datasets: emit a single frame.
                total = 1
            else:
                if len(set(lengths)) > 1:
                    detail = ", ".join(
                        f"{source}:{dataset.shape}" for source, dataset, _ in datasets
                    )
                    raise ValueError(f"datasets disagree on frame count ({detail})")
                total = lengths[0]

            stop = total if self.stop is None else min(int(self.stop), total)
            if self.start >= stop:
                raise ValueError(
                    f"empty frame range: start={self.start} stop={stop} total={total}"
                )

            for index in range(self.start, stop, self.stride):
                record: dict[str, Any] = {}
                for source, dataset, axis_override in datasets:
                    record[source] = _take(dataset, index, axis_override, self.time_axis, np)
                if frame_attrs is not None:
                    record["attrs"] = frame_attrs
                yield record


def _take(dataset: Any, index: int, axis_override: int | None, default_axis: int, np: Any) -> Any:
    if dataset.ndim == 0:
        return dataset[()]
    axis = default_axis if axis_override is None else axis_override
    if axis < 0:
        axis += dataset.ndim
    if axis == 0:
        return dataset[index]
    # Slice one frame without materialising the whole dataset.
    selection = (slice(None),) * axis + (index,)
    return np.asarray(dataset[selection])


def _decode_attrs(attrs: Any) -> dict[str, Any]:
    return {str(key): _decode_value(value) for key, value in attrs.items()}


def _decode_value(value: Any) -> Any:
    import numpy as np

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = ["H5Array", "H5Reader"]
