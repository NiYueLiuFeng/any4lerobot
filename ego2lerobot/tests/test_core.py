"""Tests for discovery, mapping, readers and the conversion worker.

Run with::

    python -m unittest discover -s ego2lerobot/tests -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ego2lerobot.common.discovery import (
    DiscoveryConfig,
    build_output_path,
    discover_files,
    read_manifest,
    write_manifest,
)
from ego2lerobot.common.state import is_task_done
from ego2lerobot.common.types import ConversionTask
from ego2lerobot.config import PipelineConfig, PipelineConfigError
from ego2lerobot.mapping import MappingConfig, MappingError


class DiscoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for relative in (
            "scene_a/episode_0.h5",
            "scene_a/sub/episode_1.hdf5",
            "scene_b/episode_0.h5",
            "scene_b/ignore.mcap",
            ".hidden/x.h5",
            "logs/y.h5",
        ):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_recursive_and_excluded(self) -> None:
        files = discover_files(
            DiscoveryConfig(roots=[self.root], include=("**/*.h5", "**/*.hdf5"))
        )
        relatives = [item.relative_path for item in files]
        self.assertEqual(
            relatives,
            [
                "scene_a/episode_0.h5",
                "scene_a/sub/episode_1.hdf5",
                "scene_b/episode_0.h5",
            ],
        )

    def test_include_single_extension(self) -> None:
        files = discover_files(
            DiscoveryConfig(roots=[self.root], include=("**/*.hdf5",))
        )
        self.assertEqual([item.relative_path for item in files], ["scene_a/sub/episode_1.hdf5"])

    def test_dedupe_and_max_files(self) -> None:
        scope = ("**/*.h5", "**/*.hdf5")
        files = discover_files(
            DiscoveryConfig(roots=[self.root, self.root / "scene_a"], include=scope)
        )
        self.assertEqual(len(files), 3)

        limited = discover_files(
            DiscoveryConfig(roots=[self.root], include=scope, max_files=2)
        )
        self.assertEqual(len(limited), 2)

    def test_manifest_roundtrip(self) -> None:
        files = discover_files(DiscoveryConfig(roots=[self.root]))
        manifest = self.root / "manifest.json"
        write_manifest(manifest, files)
        loaded = read_manifest(manifest)
        self.assertEqual([item.relative_path for item in loaded], [f.relative_path for f in files])

    def test_output_path_is_unique_per_relative_path(self) -> None:
        files = discover_files(DiscoveryConfig(roots=[self.root]))
        outputs = {build_output_path(item, self.root / "out") for item in files}
        self.assertEqual(len(outputs), len(files))


class MappingTest(unittest.TestCase):
    def test_nested_candidates_defaults_and_transforms(self) -> None:
        config = MappingConfig.from_dict(
            {
                "fps": 30,
                "features": {
                    "observation.state": {
                        "dtype": "float32",
                        "shape": [2],
                        "source": ["proprio.missing", "proprio.qpos"],
                        "transforms": [{"op": "slice", "start": 0, "stop": 2}],
                    },
                    "action": {"dtype": "float32", "shape": [2], "default": [0.0, 0.0]},
                },
                "task": {"source": "attrs.instruction", "default": "fallback"},
            }
        )
        frame = config.resolve(
            {"proprio": {"qpos": [1.0, 2.0, 3.0]}, "attrs": {"instruction": "pick"}}
        )
        np.testing.assert_allclose(frame["observation.state"], [1.0, 2.0])
        self.assertEqual(frame["action"], [0.0, 0.0])
        self.assertEqual(frame["task"], "pick")

        fallback = config.resolve({"proprio": {"qpos": [1.0, 2.0]}})
        self.assertEqual(fallback["task"], "fallback")

    def test_flat_key_with_dots_wins(self) -> None:
        config = MappingConfig.from_dict(
            {
                "fps": 10,
                "features": {
                    "x": {"dtype": "float32", "shape": [2], "source": "proprio.qpos"}
                },
            }
        )
        frame = config.resolve({"proprio.qpos": np.array([1.0, 2.0], dtype=np.float32)})
        np.testing.assert_allclose(frame["x"], [1.0, 2.0])

    def test_missing_required_feature_raises(self) -> None:
        config = MappingConfig.from_dict(
            {
                "fps": 10,
                "features": {"x": {"dtype": "float32", "shape": [1], "source": "a"}},
            }
        )
        with self.assertRaises(MappingError):
            config.resolve({})

    def test_invalid_configs_rejected(self) -> None:
        with self.assertRaises(MappingError):
            MappingConfig.from_dict({"fps": 10, "features": {}})
        with self.assertRaises(MappingError):
            MappingConfig.from_dict(
                {
                    "fps": 10,
                    "features": {"x": {"dtype": "not-a-dtype", "source": "a"}},
                }
            )
        with self.assertRaises(MappingError):
            MappingConfig.from_dict(
                {
                    "fps": 10,
                    "features": {"x": {"dtype": "float32", "source": "a", "transforms": [{"op": "nope"}]}},
                }
            )

    def test_lerobot_shapes_are_tuples(self) -> None:
        config = MappingConfig.from_dict(
            {
                "fps": 10,
                "features": {"x": {"dtype": "float32", "shape": [3], "source": "a"}},
            }
        )
        self.assertEqual(config.lerobot_features()["x"]["shape"], (3,))


class ReaderAndWorkerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import h5py  # noqa: F401
            import lerobot  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest(f"optional dependency missing: {exc}") from exc

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.h5_path = self.root / "episode.h5"
        self._write_h5(self.h5_path)
        self.config_path = self.root / "pipeline.json"
        self.output = self.root / "out"
        self.config_path.write_text(
            json.dumps(
                {
                    "repo_id": "ego/test",
                    "output_root": str(self.output),
                    "fps": 10,
                    "robot_type": "ego",
                    "features": {
                        "observation.state": {
                            "dtype": "float32",
                            "shape": [3],
                            "source": "proprio.qpos",
                        },
                        "action": {"dtype": "float32", "shape": [3], "source": "proprio.cmd"},
                    },
                    "reader": {
                        "type": "h5",
                        "arrays": {
                            "proprio.qpos": "/qpos",
                            "proprio.cmd": "/cmd",
                        },
                        "attrs": True,
                    },
                    "writer": {"use_videos": False},
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @staticmethod
    def _write_h5(path: Path, frames: int = 5) -> None:
        import h5py

        with h5py.File(path, "w") as handle:
            handle.create_dataset("qpos", data=np.arange(frames * 3, dtype=np.float32).reshape(frames, 3))
            handle.create_dataset("cmd", data=np.ones((frames, 3), dtype=np.float32))
            handle.attrs["instruction"] = "pick up"

    def test_h5_reader_frame_count(self) -> None:
        pipeline = PipelineConfig.load(self.config_path)
        records = list(pipeline.reader.iter_frames(self.h5_path))
        self.assertEqual(len(records), 5)
        np.testing.assert_allclose(records[0]["proprio.qpos"], [0, 1, 2])
        self.assertEqual(records[0]["attrs"]["instruction"], "pick up")

    def test_h5_reader_time_axis(self) -> None:
        import h5py

        path = self.root / "time_first.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("qpos", data=np.arange(12, dtype=np.float32).reshape(3, 4))
            handle.create_dataset("cmd", data=np.arange(12, dtype=np.float32).reshape(3, 4))

        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        config["reader"] = {
            "type": "h5",
            "arrays": {"proprio.qpos": "/qpos", "proprio.cmd": "/cmd"},
            "time_axis": 1,
            "attrs": False,
        }
        config_path = self.root / "time_first.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        pipeline = PipelineConfig.load(config_path)
        records = list(pipeline.reader.iter_frames(path))
        self.assertEqual(len(records), 4)
        np.testing.assert_allclose(records[1]["proprio.qpos"], [1, 5, 9])

    def test_worker_commits_and_skips(self) -> None:
        from ego2lerobot.convert.worker import process_batch

        task = ConversionTask(
            task_id="episode",
            input_path=self.h5_path,
            output_path=self.output / "episode",
        )
        params = {"config_path": str(self.config_path)}

        first = process_batch([task], params)
        self.assertEqual(first[0].status, "ok")
        self.assertEqual(first[0].frames, 5)
        self.assertTrue(is_task_done(task))

        second = process_batch([task], params)
        self.assertEqual(second[0].status, "skipped")

    def test_worker_records_failures(self) -> None:
        from ego2lerobot.convert.worker import process_batch

        broken = self.root / "broken.h5"
        broken.write_bytes(b"not hdf5")
        task = ConversionTask("broken", broken, self.output / "broken")
        results = process_batch([task], {"config_path": str(self.config_path)})
        self.assertEqual(results[0].status, "failed")
        self.assertFalse(is_task_done(task))
        self.assertTrue((self.output / "broken.failed.json").is_file())

    def test_worker_writes_directly_when_rename_is_rejected(self) -> None:
        """SFS/NFS exports without rename(2) must still commit a dataset."""

        from unittest import mock

        from ego2lerobot.common import state
        from ego2lerobot.convert.worker import process_batch

        task = ConversionTask(
            task_id="no_rename",
            input_path=self.h5_path,
            output_path=self.output / "no_rename",
        )
        with mock.patch.object(state, "rename_supported", return_value=False):
            results = process_batch([task], {"config_path": str(self.config_path)})

        self.assertEqual(results[0].status, "ok", results[0].error)
        self.assertTrue(state.is_task_done(task))
        self.assertFalse(state.partial_path(task.output_path).exists())
        self.assertTrue((Path(task.output_path) / "_conversion.json").is_file())


class StateTest(unittest.TestCase):
    """Commit behaviour on file systems that reject rename(2)."""

    def setUp(self) -> None:
        from ego2lerobot.common import state

        self.state = state
        state._RENAME_SUPPORT.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.output = self.root / "out"

    def tearDown(self) -> None:
        self.state._RENAME_SUPPORT.clear()
        self.tmp.cleanup()

    def _task(self, name: str = "episode") -> ConversionTask:
        return ConversionTask(name, self.root / f"{name}.h5", self.output / name)

    def test_staging_path_uses_partial_when_rename_works(self) -> None:
        from unittest import mock

        with mock.patch.object(self.state, "rename_supported", return_value=True):
            staging = self.state.staging_path(self._task())
        self.assertEqual(staging, self.state.partial_path(self.output / "episode"))

    def test_staging_path_uses_final_when_rename_is_rejected(self) -> None:
        from unittest import mock

        with mock.patch.object(self.state, "rename_supported", return_value=False):
            staging = self.state.staging_path(self._task())
        self.assertEqual(staging, self.output / "episode")
        self.assertTrue(self.output.is_dir())

    def test_commit_marks_direct_output_successful(self) -> None:
        task = self._task("direct")
        task.output_path.mkdir(parents=True)
        (task.output_path / "result.json").write_text("{}", encoding="utf-8")

        self.state.commit_output(task)

        self.assertTrue(self.state.is_task_done(task))
        self.assertTrue((task.output_path / "result.json").is_file())

    def test_commit_copies_when_replace_fails(self) -> None:
        from unittest import mock

        task = self._task("copy")
        partial = self.state.partial_path(task.output_path)
        partial.mkdir(parents=True)
        (partial / "result.json").write_text("{}", encoding="utf-8")

        denied = PermissionError(1, "Operation not permitted")
        with mock.patch.object(self.state.os, "replace", side_effect=denied):
            self.state.commit_output(task)

        self.assertTrue(self.state.is_task_done(task))
        self.assertTrue((task.output_path / "result.json").is_file())
        self.assertFalse(partial.exists())

    def test_write_failure_falls_back_without_replace(self) -> None:
        from unittest import mock

        task = self._task("broken")
        denied = PermissionError(1, "Operation not permitted")
        with mock.patch.object(self.state.os, "replace", side_effect=denied):
            path = self.state.write_failure(task, "boom")

        self.assertTrue(path.is_file())
        self.assertIn("boom", path.read_text(encoding="utf-8"))
        self.assertFalse(path.with_name(path.name + ".tmp").exists())


class PipelineConfigTest(unittest.TestCase):
    def test_example_configs_load(self) -> None:
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        for name in ("ego_h5.yaml", "ego_mcap.yaml", "ego_egoverse.yaml"):
            pipeline = PipelineConfig.load(config_dir / name)
            self.assertGreater(pipeline.fps, 0)
            self.assertTrue(pipeline.mapping.features)
            self.assertIn("reader", pipeline.describe())

    def test_unknown_keys_rejected(self) -> None:
        with self.assertRaises(PipelineConfigError):
            PipelineConfig.from_dict(
                {
                    "repo_id": "ego/test",
                    "fps": 10,
                    "features": {"x": {"dtype": "float32", "source": "a"}},
                    "reader": {"type": "h5", "arrays": {"a": "/a"}},
                    "unexpected": True,
                }
            )


if __name__ == "__main__":
    unittest.main()
