"""Tests for the EgoVerse (Zarr v3) reader."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

try:
    import zarr
    from PIL import Image
    from zarr.core.dtype import VariableLengthBytes

    _HAS_DEPS = True
except Exception:  # pragma: no cover - depends on env
    _HAS_DEPS = False

from ego2lerobot.config import PipelineConfig
from ego2lerobot.readers import EgoEgoverseReader, make_reader


def _jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.full((4, 4, 3), 128, dtype=np.uint8)).save(buffer, format="JPEG")
    return buffer.getvalue()


@unittest.skipUnless(_HAS_DEPS, "zarr v3 and Pillow are required")
class EgoEgoverseReaderTest(unittest.TestCase):
    frames = 5

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.episode = self.root / "2025-09-20-17-42-51-000000"
        self._write_episode(self.episode, self.frames)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @staticmethod
    def _write_episode(path: Path, frames: int) -> None:
        group = zarr.open_group(path, mode="w")
        group.attrs.update(
            {
                "embodiment": "aria_bimanual",
                "fps": 10,
                "task_name": "fold_clothes",
                "task_description": "folding clothes",
                "features": {
                    "right.obs_ee_pose": {"dtype": "float64", "shape": [7]},
                    "images.front_1": {"dtype": "jpeg", "shape": [4, 4, 3]},
                    "annotations": {"dtype": "json", "shape": [0]},
                },
            }
        )
        group.create_array(
            "right.obs_ee_pose",
            data=np.arange(frames * 7, dtype=np.float64).reshape(frames, 7),
        )
        group.create_array(
            "obs_rgb_timestamps_ns",
            data=np.arange(frames, dtype=np.int64) * 1_000,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            images = group.create_array(
                "images.front_1", shape=(frames,), dtype=VariableLengthBytes()
            )
            images[:] = np.array([_jpeg_bytes()] * frames, dtype=object)
            group.create_array(
                "annotations", shape=(0,), dtype=VariableLengthBytes()
            )

    # ------------------------------------------------------------- basic API

    def test_explicit_mapping_and_decode(self) -> None:
        reader = make_reader(
            {
                "type": "ego_egoverse",
                "arrays": {"obs.state": "right.obs_ee_pose"},
                "images": {"cameras.front": "images.front_1"},
            }
        )
        self.assertIsInstance(reader, EgoEgoverseReader)
        reader.validate(self.episode / "zarr.json")  # anchor file also works

        records = list(reader.iter_frames(self.episode))
        self.assertEqual(len(records), self.frames)
        np.testing.assert_allclose(records[0]["obs.state"], np.arange(7))
        self.assertEqual(records[0]["cameras.front"].shape, (4, 4, 3))
        self.assertEqual(records[0]["cameras.front"].dtype, np.uint8)
        self.assertEqual(records[0]["attrs"]["task_name"], "fold_clothes")
        self.assertEqual(records[0]["frame_index"], 0)

    def test_auto_detect_and_stride(self) -> None:
        reader = make_reader(
            {"type": "ego_egoverse", "start": 1, "stop": self.frames, "stride": 2}
        )
        records = list(reader.iter_frames(self.episode))

        self.assertEqual([record["frame_index"] for record in records], [1, 3])
        self.assertIn("right.obs_ee_pose", records[0])
        self.assertIn("obs_rgb_timestamps_ns", records[0])
        self.assertIn("images.front_1", records[0])
        # `annotations` is not a per-frame stream and must be skipped.
        self.assertNotIn("annotations", records[0])

    def test_decode_images_false_returns_bytes(self) -> None:
        reader = make_reader(
            {
                "type": "ego_egoverse",
                "arrays": {},
                "images": {"cam": "images.front_1"},
                "decode_images": False,
            }
        )
        record = next(reader.iter_frames(self.episode))
        self.assertIsInstance(record["cam"], (bytes, bytearray))

    def test_describe_is_json_serialisable(self) -> None:
        reader = make_reader({"type": "ego_egoverse"})
        json.dumps(reader.describe())

    def test_accepts_input_rejects_array_nodes(self) -> None:
        reader = make_reader({"type": "ego_egoverse"})
        self.assertTrue(reader.accepts_input(self.episode / "zarr.json"))
        self.assertFalse(
            reader.accepts_input(self.episode / "right.obs_ee_pose" / "zarr.json")
        )

    def test_discovery_keeps_one_task_per_sub_dataset(self) -> None:
        from ego2lerobot.convert.run import plan_conversion

        pipeline = PipelineConfig.from_dict(
            {
                "repo_id": "ego/test",
                "output_root": str(self.root / "out"),
                "fps": 10,
                "features": {
                    "observation.state": {
                        "dtype": "float32",
                        "shape": [7],
                        "source": "obs.state",
                    }
                },
                "reader": {
                    "type": "ego_egoverse",
                    "arrays": {"obs.state": "right.obs_ee_pose"},
                },
            }
        )
        plan = plan_conversion(pipeline, [self.root], output_root=self.root / "out")

        self.assertEqual(len(plan.tasks), 1)
        self.assertEqual(plan.tasks[0].input_path, self.episode / "zarr.json")

    # ---------------------------------------------------------------- errors

    def test_unknown_config_key_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_reader({"type": "ego_egoverse", "bogus": 1})

    def test_length_mismatch_is_rejected(self) -> None:
        bad = self.root / "bad_episode"
        group = zarr.open_group(bad, mode="w")
        group.create_array("a", data=np.zeros((3, 2), dtype=np.float32))
        group.create_array("b", data=np.zeros((4, 2), dtype=np.float32))

        reader = make_reader(
            {"type": "ego_egoverse", "arrays": {"a": "a", "b": "b"}}
        )
        with self.assertRaises(ValueError):
            reader.validate(bad)

    def test_missing_member_is_rejected(self) -> None:
        reader = make_reader(
            {"type": "ego_egoverse", "arrays": {"x": "does.not.exist"}}
        )
        with self.assertRaises(KeyError):
            reader.validate(self.episode)

    # ----------------------------------------------------------- integration

    def test_pipeline_mapping_resolves_record(self) -> None:
        config_path = self.root / "pipeline.json"
        config_path.write_text(
            json.dumps(
                {
                    "repo_id": "ego/test",
                    "output_root": str(self.root / "out"),
                    "fps": 10,
                    "robot_type": "ego_egoverse",
                    "task": {
                        "source": "attrs.task_description",
                        "default": "ego manipulation",
                    },
                    "features": {
                        "observation.state": {
                            "dtype": "float32",
                            "shape": [7],
                            "source": "obs.state",
                            "transforms": [{"op": "astype", "dtype": "float32"}],
                        },
                        "observation.images.front_1": {
                            "dtype": "image",
                            "shape": [4, 4, 3],
                            "source": "cameras.front",
                            "transforms": [
                                {
                                    "op": "as_image",
                                    "dtype": "uint8",
                                    "channel_order": "HWC",
                                }
                            ],
                        },
                    },
                    "reader": {
                        "type": "ego_egoverse",
                        "arrays": {"obs.state": "right.obs_ee_pose"},
                        "images": {"cameras.front": "images.front_1"},
                    },
                    "writer": {"use_videos": False},
                }
            ),
            encoding="utf-8",
        )

        pipeline = PipelineConfig.load(config_path)
        record = next(pipeline.reader.iter_frames(self.episode))
        frame = pipeline.mapping.resolve(record)

        self.assertEqual(frame["task"], "folding clothes")
        self.assertEqual(frame["observation.state"].shape, (7,))
        self.assertEqual(frame["observation.state"].dtype, np.float32)
        self.assertEqual(frame["observation.images.front_1"].shape, (4, 4, 3))
        self.assertEqual(frame["observation.images.front_1"].dtype, np.uint8)

    def test_demo_convert_writes_lerobot_dataset(self) -> None:
        try:
            import lerobot  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on env
            self.skipTest(f"lerobot missing: {exc}")

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        from ego2lerobot.demo.egoverse_demo import convert_to_lerobot

        config_path = self.root / "demo_pipeline.json"
        config_path.write_text(
            json.dumps(
                {
                    "repo_id": "ego/demo",
                    "output_root": str(self.root / "lerobot"),
                    "fps": 10,
                    "robot_type": "ego_egoverse",
                    "task": {
                        "source": "attrs.task_description",
                        "default": "ego manipulation",
                    },
                    "features": {
                        "observation.state": {
                            "dtype": "float32",
                            "shape": [7],
                            "source": "obs.state",
                            "transforms": [{"op": "astype", "dtype": "float32"}],
                        },
                        "action": {
                            "dtype": "float32",
                            "shape": [7],
                            "default": [0.0] * 7,
                            "transforms": [{"op": "astype", "dtype": "float32"}],
                        },
                        "observation.images.front_1": {
                            "dtype": "image",
                            "shape": [4, 4, 3],
                            "source": "cameras.front",
                            "transforms": [
                                {"op": "as_image", "dtype": "uint8", "channel_order": "HWC"}
                            ],
                        },
                    },
                    "reader": {
                        "type": "ego_egoverse",
                        "arrays": {"obs.state": "right.obs_ee_pose"},
                        "images": {"cameras.front": "images.front_1"},
                    },
                    "writer": {"use_videos": False},
                }
            ),
            encoding="utf-8",
        )

        result = convert_to_lerobot(
            self.episode,
            config=config_path,
            output_root=self.root / "lerobot",
            max_frames=3,
            overwrite=True,
            verbose=False,
        )
        self.assertEqual(result.status, "ok", result.error)
        self.assertEqual(result.frames, 3)

        dataset = LeRobotDataset(repo_id="ego/demo", root=Path(result.output_path))
        self.assertEqual(dataset.num_frames, 3)
        self.assertEqual(dataset.num_episodes, 1)
        item = dataset[0]
        self.assertEqual(tuple(item["observation.images.front_1"].shape), (3, 4, 4))
        self.assertEqual(item["task"], "folding clothes")


if __name__ == "__main__":
    unittest.main()
