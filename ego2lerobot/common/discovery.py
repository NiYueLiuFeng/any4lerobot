"""Recursive, deterministic discovery of raw episode files.

The discovery layer is deliberately independent from Ray and LeRobot: it only
turns a set of roots into a stable list of :class:`DiscoveredFile` objects.  The
converter then builds one ``ConversionTask`` per file.

Design points
-------------
* Deterministic: directories and files are sorted, so two runs over the same
  tree produce the same order (important for reproducible aggregation).
* Prunable: excluded directories are never descended into.
* Deduplicated: the same inode reachable through several roots or symlinks is
  reported once.
* Includes a small JSON manifest so a large scan can be reviewed before the
  expensive conversion starts.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Iterator, Sequence

DEFAULT_INCLUDE = (
    "**/*.h5",
    "**/*.hdf5",
    "**/*.mcap",
)

DEFAULT_EXCLUDE = (
    "**/.*",
    "**/.*/**",
    "**/__pycache__/**",
    "**/_temp/**",
    "**/tmp/**",
    "**/logs/**",
)


@dataclass(frozen=True)
class DiscoveredFile:
    """One raw input file found under a discovery root."""

    path: Path
    root: Path
    relative_path: str
    size: int
    mtime: float
    inode: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def suffix(self) -> str:
        return self.path.suffix.lower()


@dataclass
class DiscoveryConfig:
    """Inputs for :func:`discover_files`.

    ``include``/``exclude`` are globs matched against the path relative to the
    current root (POSIX separators).  ``**`` crosses directory separators.
    A leading ``**/`` is also tried without the prefix, so ``**/*.h5`` matches
    a top-level file as well.
    """

    roots: Sequence[str | Path]
    include: Sequence[str] = DEFAULT_INCLUDE
    exclude: Sequence[str] = DEFAULT_EXCLUDE
    follow_symlinks: bool = False
    max_depth: int | None = None
    max_files: int | None = None
    dedupe: bool = True
    # Prune these directory names anywhere in the tree without a glob.
    skip_dirs: Sequence[str] = (".git", ".venv", "__pycache__", "node_modules")
    # Optional extra predicate applied after include/exclude.  Readers use it
    # to reject look-alike files (e.g. Zarr array nodes also contain zarr.json).
    file_filter: Callable[[Path], bool] | None = None

    def __post_init__(self) -> None:
        self.roots = [Path(root).expanduser() for root in self.roots]
        if not self.roots:
            raise ValueError("DiscoveryConfig.roots must not be empty")
        if self.max_depth is not None and self.max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        if self.max_files is not None and self.max_files <= 0:
            raise ValueError("max_files must be > 0")


def _matches(relative: PurePosixPath, pattern: str) -> bool:
    """Match a relative path against a glob.

    ``PurePath.match`` does not let ``**/`` match zero directories on all
    supported Python versions, and ``fnmatch`` lets ``*`` cross ``/``.  Trying
    both (plus the stripped prefix) gives intuitive behaviour on 3.10+.
    """

    text = relative.as_posix()
    if fnmatch.fnmatchcase(text, pattern):
        return True
    if relative.match(pattern):
        return True
    if pattern.startswith("**/") and fnmatch.fnmatchcase(text, pattern[3:]):
        return True
    # ``**/logs/**`` describes files inside a directory; it should also prune
    # the directory itself instead of walking a potentially huge tree.
    if pattern.endswith("/**") and fnmatch.fnmatchcase(text, pattern[:-3]):
        return True
    return False


def _matches_any(relative: PurePosixPath, patterns: Iterable[str]) -> bool:
    return any(_matches(relative, pattern) for pattern in patterns)


def _matches_dir(relative: PurePosixPath, patterns: Iterable[str]) -> bool:
    """Directory pruning: ``**/logs/**`` also excludes the ``logs`` directory."""

    for pattern in patterns:
        if _matches(relative, pattern):
            return True
        if pattern.endswith("/**") and _matches(relative, pattern[:-3]):
            return True
    return False


def _accept(config: DiscoveryConfig, path: Path) -> bool:
    return config.file_filter is None or bool(config.file_filter(path))


def _join(relative_dir: PurePosixPath, name: str) -> PurePosixPath:
    """Append a name to a relative directory, where ``"."`` means the root."""

    return PurePosixPath(name) if str(relative_dir) == "." else relative_dir / name


def _stat_key(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_dev, stat.st_ino)


def _walk(
    root: Path,
    config: DiscoveryConfig,
    seen_inodes: set[tuple[int, int]],
    seen_paths: set[Path],
) -> Iterator[DiscoveredFile]:
    root = root.resolve()
    if not root.exists():
        return
    if root.is_file():
        # Allow a single file to be passed as a root.
        relative = PurePosixPath(root.name)
        if (
            _matches_any(relative, config.include)
            and not _matches_any(relative, config.exclude)
            and _accept(config, root)
        ):
            key = _stat_key(root) if config.dedupe else None
            if key is not None and key in seen_inodes:
                return
            if key is not None:
                seen_inodes.add(key)
            if root in seen_paths:
                return
            seen_paths.add(root)
            stat = root.stat()
            yield DiscoveredFile(
                path=root,
                root=root.parent,
                relative_path=root.name,
                size=stat.st_size,
                mtime=stat.st_mtime,
                inode=key,
            )
        return

    for dirpath, dirnames, filenames in os.walk(root, followlinks=config.follow_symlinks):
        current = Path(dirpath)
        depth = len(current.relative_to(root).parts) if config.max_depth is not None else 0
        try:
            relative_dir = PurePosixPath(current.relative_to(root).as_posix())
        except ValueError:
            relative_dir = PurePosixPath(".")

        # Prune before descending.  Sorting makes traversal deterministic.
        kept_dirs: list[str] = []
        for name in sorted(dirnames):
            if name in config.skip_dirs:
                continue
            if _matches_dir(_join(relative_dir, name), config.exclude):
                continue
            if config.max_depth is not None and depth + 1 > config.max_depth:
                continue
            kept_dirs.append(name)
        dirnames[:] = kept_dirs

        for filename in sorted(filenames):
            if filename in config.skip_dirs:
                continue
            path = current / filename
            if path.is_symlink() and not config.follow_symlinks:
                continue
            relative = _join(relative_dir, filename)
            if not _matches_any(relative, config.include):
                continue
            if _matches_any(relative, config.exclude):
                continue
            if not _accept(config, path):
                continue

            try:
                # One stat call supplies both the dedupe key and the metadata.
                stat = path.stat()
            except OSError:
                # Broken link or unreadable entry: skip it instead of failing
                # the whole scan.
                continue
            key = (stat.st_dev, stat.st_ino) if config.dedupe else None
            if key is not None and key in seen_inodes:
                continue
            if key is not None:
                seen_inodes.add(key)

            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)

            yield DiscoveredFile(
                path=resolved,
                root=root,
                relative_path=relative.as_posix(),
                size=stat.st_size,
                mtime=stat.st_mtime,
                inode=key,
            )

            if config.max_files is not None and len(seen_paths) >= config.max_files:
                return


def discover_files(config: DiscoveryConfig) -> list[DiscoveredFile]:
    """Return every matching file under every root, in stable order."""

    seen_inodes: set[tuple[int, int]] = set()
    seen_paths: set[Path] = set()
    discovered: list[DiscoveredFile] = []

    for root in config.roots:
        for item in _walk(Path(root), config, seen_inodes, seen_paths):
            discovered.append(item)
            if config.max_files is not None and len(discovered) >= config.max_files:
                return discovered

    # Two roots may overlap.  inode dedupe already handles hard/symlinks; this
    # keeps the output stable even when dedupe is disabled.
    unique: dict[Path, DiscoveredFile] = {}
    for item in discovered:
        unique.setdefault(item.path, item)
    return sorted(
        unique.values(),
        key=lambda item: (str(item.root), item.relative_path),
    )


def summarize_discovery(files: Sequence[DiscoveredFile]) -> dict[str, Any]:
    """Small counters for logs."""

    by_suffix = Counter(item.suffix for item in files)
    total_size = sum(item.size for item in files)
    return {
        "files": len(files),
        "roots": sorted({str(item.root) for item in files}),
        "by_suffix": dict(sorted(by_suffix.items())),
        "total_bytes": total_size,
        "total_gib": round(total_size / (1024**3), 3),
    }


def write_manifest(path: str | Path, files: Sequence[DiscoveredFile]) -> Path:
    """Write a JSON manifest atomically."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.time(),
        "summary": summarize_discovery(files),
        "files": [
            {
                "path": str(item.path),
                "root": str(item.root),
                "relative_path": item.relative_path,
                "size": item.size,
                "mtime": item.mtime,
                "inode": list(item.inode) if item.inode else None,
                "metadata": item.metadata,
            }
            for item in files
        ],
    }
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return target


def read_manifest(path: str | Path) -> list[DiscoveredFile]:
    """Read a manifest written by :func:`write_manifest`."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    files: list[DiscoveredFile] = []
    for item in payload.get("files", []):
        inode = item.get("inode")
        files.append(
            DiscoveredFile(
                path=Path(item["path"]),
                root=Path(item["root"]),
                relative_path=item["relative_path"],
                size=int(item.get("size", 0)),
                mtime=float(item.get("mtime", 0.0)),
                inode=tuple(inode) if inode else None,
                metadata=dict(item.get("metadata") or {}),
            )
        )
    return files


def build_output_path(
    item: DiscoveredFile,
    output_root: str | Path,
    *,
    suffix: str = "",
    max_stem: int = 80,
) -> Path:
    """Derive a unique output directory from a discovered file.

    The relative directory structure is preserved, and a short path hash is
    appended so that equal stems under different sub-directories never collide.
    """

    relative = PurePosixPath(item.relative_path)
    stem = relative.stem or "episode"
    stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)[:max_stem]
    digest = hashlib.sha1(relative.as_posix().encode("utf-8")).hexdigest()[:10]
    directory = Path(output_root) / Path(relative.parent.as_posix())
    return directory / f"{stem}-{digest}{suffix}"


__all__ = [
    "DEFAULT_EXCLUDE",
    "DEFAULT_INCLUDE",
    "DiscoveredFile",
    "DiscoveryConfig",
    "build_output_path",
    "discover_files",
    "read_manifest",
    "summarize_discovery",
    "write_manifest",
]
