"""MCAP episode reader with topic synchronisation.

MCAP files store independent topics at independent timestamps.  This reader
decodes the configured topics, merges them onto the timeline of a reference
topic, and yields one record per output frame::

    reader:
      type: mcap
      topics:
        proprio.qpos: /joint_states
        proprio.cmd: /command
        cameras.head: /camera/image
      fields:
        proprio.qpos: position
        cameras.head: data
      sync:
        mode: nearest        # nearest | hold | exact
        max_gap_s: 0.1
        reference: proprio.qpos

Decoding requires either ``mcap-ros2-support`` (``pip install mcap-ros2-support``)
or ``rosbags``.  Both are imported lazily so the rest of the package works
without ROS installed.

Note on memory: synchronisation needs all messages of the configured topics
before the first frame can be emitted, so one episode is buffered in memory.
Keep ``start_s`` / ``stop_s`` and ``stride`` for very long recordings, or cap
episode length at the conversion layer.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from .base import EpisodeReader


@dataclass(frozen=True)
class _Sample:
    t: float
    value: Any


class McapReader(EpisodeReader):
    kind = "mcap"

    def __init__(
        self,
        topics: Mapping[str, str],
        *,
        fields: Mapping[str, str] | None = None,
        sync: str = "nearest",
        max_gap_s: float | None = None,
        reference: str | None = None,
        start_s: float | None = None,
        stop_s: float | None = None,
        stride: int = 1,
        decoder: str = "auto",
    ):
        if not topics:
            raise ValueError("reader.topics must not be empty for an mcap reader")
        if sync not in {"nearest", "hold", "exact"}:
            raise ValueError("reader.sync.mode must be one of: nearest, hold, exact")
        if max_gap_s is not None and max_gap_s < 0:
            raise ValueError("reader.sync.max_gap_s must be >= 0")
        if stride < 1:
            raise ValueError("reader.stride must be >= 1")
        if reference is not None and reference not in topics:
            raise ValueError(
                f"reader.sync.reference {reference!r} is not one of the configured topics"
            )
        if decoder not in {"auto", "mcap_ros2", "rosbags"}:
            raise ValueError("reader.decoder must be auto, mcap_ros2 or rosbags")

        self.topics = dict(topics)
        # Reverse lookup for the topic name emitted by the decoder.  First
        # configured source wins if several map to the same topic.
        self._sources_by_topic: dict[str, str] = {}
        for name, topic in self.topics.items():
            self._sources_by_topic.setdefault(topic, name)
        self.fields = dict(fields or {})
        self.sync = sync
        self.max_gap_s = max_gap_s
        self.reference = reference or next(iter(topics))
        self.start_s = start_s
        self.stop_s = stop_s
        self.stride = stride
        self.decoder = decoder

        unknown_fields = set(self.fields) - set(self.topics)
        if unknown_fields:
            raise ValueError(f"reader.fields refers to unknown topics: {sorted(unknown_fields)}")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "McapReader":
        allowed = {"type", "topics", "fields", "sync", "start_s", "stop_s", "stride", "decoder"}
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"unknown mcap reader keys: {sorted(unknown)}")

        sync = config.get("sync") or {}
        if isinstance(sync, str):
            sync = {"mode": sync}
        if not isinstance(sync, dict):
            raise ValueError("reader.sync must be a string or a mapping")
        unknown_sync = set(sync) - {"mode", "max_gap_s", "reference"}
        if unknown_sync:
            raise ValueError(f"unknown reader.sync keys: {sorted(unknown_sync)}")

        return cls(
            config.get("topics") or {},
            fields=config.get("fields") or {},
            sync=str(sync.get("mode", "nearest")),
            max_gap_s=sync.get("max_gap_s"),
            reference=sync.get("reference"),
            start_s=config.get("start_s"),
            stop_s=config.get("stop_s"),
            stride=int(config.get("stride", 1)),
            decoder=str(config.get("decoder", "auto")),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "type": self.kind,
            "topics": self.topics,
            "fields": self.fields,
            "sync": self.sync,
            "max_gap_s": self.max_gap_s,
            "reference": self.reference,
            "start_s": self.start_s,
            "stop_s": self.stop_s,
            "stride": self.stride,
            "decoder": self.decoder,
        }

    def validate(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix.lower() != ".mcap":
            raise ValueError(f"not an mcap file: {path}")
        _check_decoder(self.decoder)

    def iter_frames(self, path: str | Path) -> Iterator[dict[str, Any]]:
        topics = list(self.topics.values())
        streams: dict[str, list[_Sample]] = {name: [] for name in self.topics}

        for topic, timestamp_ns, message in _iter_messages(Path(path), topics, self.decoder):
            name = self._name_for_topic(topic)
            if name is None:
                continue
            value = _extract(message, self.fields.get(name))
            streams[name].append(_Sample(t=timestamp_ns / 1e9, value=value))

        reference = self.reference
        if not streams[reference]:
            raise ValueError(f"reference topic {self.topics[reference]!r} has no messages in {path}")

        for name, samples in streams.items():
            if not samples:
                raise ValueError(f"topic {self.topics[name]!r} has no messages in {path}")
            samples.sort(key=lambda sample: sample.t)

        ref_samples = streams[reference]
        times = {name: [sample.t for sample in samples] for name, samples in streams.items()}
        start = self.start_s if self.start_s is not None else ref_samples[0].t
        stop = self.stop_s if self.stop_s is not None else ref_samples[-1].t

        emitted = 0
        for position, sample in enumerate(ref_samples):
            if sample.t < start:
                continue
            if sample.t > stop:
                break
            if position % self.stride != 0:
                continue

            record: dict[str, Any] = {}
            ok = True
            for name in self.topics:
                match = _pick(streams[name], times[name], sample.t, self.sync)
                if match is None or (
                    self.max_gap_s is not None and abs(match.t - sample.t) > self.max_gap_s
                ):
                    ok = False
                    break
                record[name] = match.value
            if not ok:
                continue

            record["timestamp"] = sample.t
            record["timestamp_ns"] = int(round(sample.t * 1e9))
            emitted += 1
            yield record

        if emitted == 0:
            raise ValueError(
                f"no synchronised frames in {path} (reference topic {self.topics[reference]!r})"
            )

    def _name_for_topic(self, topic: str) -> str | None:
        return self._sources_by_topic.get(topic)


def _pick(samples: list[_Sample], times: list[float], t: float, mode: str) -> _Sample | None:
    if not samples:
        return None
    position = bisect.bisect_left(times, t)

    if mode == "hold":
        if position == 0:
            return samples[0] if abs(samples[0].t - t) == 0 else None
        return samples[position - 1]

    if position == 0:
        return samples[0]
    if position >= len(samples):
        return samples[-1]
    before = samples[position - 1]
    after = samples[position]
    if mode == "exact":
        return after if abs(after.t - t) <= abs(before.t - t) else before
    return after if (after.t - t) <= (t - before.t) else before


def _extract(message: Any, field: str | None) -> Any:
    if field is None:
        return _to_builtin(message)
    value = message
    for part in field.split("."):
        if isinstance(value, Mapping):
            value = value[part]
        elif isinstance(value, (list, tuple)):
            value = value[int(part)]
        else:
            value = getattr(value, part)
    return _to_builtin(value)


def _to_builtin(value: Any) -> Any:
    """Convert ROS message objects to plain dicts/arrays, lazily and shallowly."""

    import numpy as np

    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (str, bytes, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _to_builtin(item) for key, item in value.items()}

    # dataclasses and ROS message classes expose fields via __dict__/__slots__.
    fields: dict[str, Any] = {}
    if hasattr(value, "__dict__"):
        fields.update(vars(value))
    for slot in getattr(value, "__slots__", ()) or ():
        if hasattr(value, slot):
            fields[slot] = getattr(value, slot)
    if fields:
        return {key: _to_builtin(item) for key, item in fields.items() if not key.startswith("_")}

    # Unknown object: leave it to the mapping layer to extract a field from it.
    return value


def _check_decoder(decoder: str) -> str:
    """Return the decoder that will actually be used."""

    if decoder in {"auto", "mcap_ros2"}:
        try:
            import mcap_ros2.reader  # noqa: F401

            return "mcap_ros2"
        except ImportError:
            if decoder == "mcap_ros2":
                raise ImportError(
                    "mcap-ros2-support is not installed; run: pip install mcap-ros2-support"
                ) from None
    if decoder in {"auto", "rosbags"}:
        try:
            import rosbags.highlevel  # noqa: F401

            return "rosbags"
        except ImportError:
            if decoder == "rosbags":
                raise ImportError("rosbags is not installed; run: pip install rosbags") from None
    raise ImportError(
        "no MCAP ROS decoder found; install one of:\n"
        "  pip install mcap-ros2-support   (ROS 2 files)\n"
        "  pip install rosbags             (ROS 1 and ROS 2 files)"
    )


def _iter_messages(
    path: Path, topics: list[str], decoder: str
) -> Iterator[tuple[str, int, Any]]:
    chosen = _check_decoder(decoder)
    if chosen == "mcap_ros2":
        yield from _iter_mcap_ros2(path, topics)
    else:
        yield from _iter_rosbags(path, topics)


def _iter_mcap_ros2(path: Path, topics: list[str]) -> Iterator[tuple[str, int, Any]]:
    from mcap_ros2.reader import read_ros2_messages

    wanted = set(topics)
    for item in read_ros2_messages(str(path), topics=topics):
        topic = getattr(getattr(item, "channel", None), "topic", None)
        if topic is None or topic not in wanted:
            continue
        yield topic, int(item.log_time), item.ros_msg


def _iter_rosbags(path: Path, topics: list[str]) -> Iterator[tuple[str, int, Any]]:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([path], default_typestore=typestore) as reader:
        for connection, timestamp, rawdata in reader.messages(topics=topics):
            message = reader.deserialize(rawdata, connection.msgtype)
            yield connection.topic, int(timestamp), message


__all__ = ["McapReader"]
