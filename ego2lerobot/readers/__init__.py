"""Reader backends.

``make_reader`` is the only entry point the converter needs::

    from ego2lerobot.readers import make_reader
    reader = make_reader({"type": "h5", "arrays": {...}})
"""

from __future__ import annotations

from typing import Any, Mapping

from .base import ArraySpec, EpisodeReader, ReaderStats
from .ego_egoverse import EgoEgoverseReader
from .h5 import H5Reader
from .mcap import McapReader

_READERS = {
    "h5": H5Reader,
    "hdf5": H5Reader,
    "mcap": McapReader,
    "ego_egoverse": EgoEgoverseReader,
}


def make_reader(config: Mapping[str, Any] | None) -> EpisodeReader:
    """Build a reader from the ``reader`` section of a pipeline config."""

    if not config:
        raise ValueError("pipeline config is missing the 'reader' section")
    if not isinstance(config, Mapping):
        raise ValueError("reader section must be a mapping")
    kind = str(config.get("type") or "").lower()
    if not kind:
        raise ValueError("reader.type is required (h5, mcap or ego_egoverse)")
    try:
        factory = _READERS[kind]
    except KeyError as exc:
        raise ValueError(
            f"unknown reader type {kind!r}; available: {sorted(_READERS)}"
        ) from exc
    return factory.from_config(config)


__all__ = [
    "ArraySpec",
    "EgoEgoverseReader",
    "EpisodeReader",
    "H5Reader",
    "McapReader",
    "ReaderStats",
    "make_reader",
]
