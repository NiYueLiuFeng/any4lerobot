"""Reader interface shared by the h5 and mcap backends.

A reader yields one *record* per frame.  A record is a plain dictionary whose
keys are the ``source`` names used by the mapping file; the mapping layer has no
knowledge of file formats.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class ArraySpec:
    """One scalar/vector/image stream inside a file."""

    source: str
    path: str | None = None
    axis: int | None = None
    transform: str | None = None  # reserved for future per-stream conversion


class EpisodeReader(ABC):
    """Backend that turns one raw file into a stream of frame records."""

    #: Short backend name used in logs and config validation.
    kind: str = "base"

    @abstractmethod
    def iter_frames(self, path: str | Path) -> Iterator[dict[str, Any]]:
        """Yield frame records in time order."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable description for logs/manifests."""

    def validate(self, path: str | Path) -> None:
        """Optional cheap check performed before conversion starts."""

    def accepts_input(self, path: str | Path) -> bool:
        """Return False to keep a discovered file from becoming a task.

        Discovery calls this for every candidate file.  The default accepts
        everything; a reader whose container stores metadata files in nested
        members (for example Zarr v3, whose array directories also contain
        ``zarr.json``) overrides it to select only real episode roots.
        """

        return True


@dataclass
class ReaderStats:
    """Optional counters a reader can fill while iterating."""

    frames: int = 0
    warnings: list[str] = field(default_factory=list)
