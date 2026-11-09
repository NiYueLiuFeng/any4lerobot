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

中文说明
--------
本文件演示如何读取一个 EgoVerse 子数据集，分三层：

1. ``EgoVerseReader``：打开一个 Zarr v3 group，把成员分成数值流、图像流和
   标注流；数值数组只加载选中的帧，图像保持懒加载、按需解码。
2. ``generate_egoverse_demo``：对外入口，解析路径后构造 reader，并打印摘要。
3. ``convert_to_lerobot``：把一集子数据集写成真正的 LeRobot 数据集，复用
   生产链路（PipelineConfig + worker.process_task），产出可直接用
   ``LeRobotDataset(repo_id, root=...)`` 读取的目录。
4. ``parse_args`` / ``main``：命令行封装，支持 ``--to-lerobot`` 转换以及
   ``--json``、``--show-frame`` 两种检查方式。

Run from the repository root::

    python -m ego2lerobot.demo.egoverse_demo

    # inspect one real sub-dataset without loading every image
    python -m ego2lerobot.demo.egoverse_demo \
        --sub-dataset 2025-09-20-17-42-51-000000 --max-frames 5 --show-frame 0

    # convert one sub-dataset into a LeRobot dataset
    python -m ego2lerobot.demo.egoverse_demo --to-lerobot \
        --sub-dataset 2025-09-20-17-42-51-000000 --max-frames 5

The converted dataset is written to ``<output_root>/<sub-dataset>-<hash>`` and
is verified by reopening it with ``LeRobotDataset`` before the script returns.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Sequence

import numpy as np

if TYPE_CHECKING:
    # 仅用于类型标注；运行时不导入，保持检查模式的启动开销不变。
    from ego2lerobot.common.types import ConversionTask, TaskResult
    from ego2lerobot.config import PipelineConfig

# 把仓库根目录加入 sys.path，保证脚本被直接执行时也能 import ego2lerobot。
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: 默认样本根目录（本地 EgoVerse 数据）。
DEFAULT_EGOVERSE_ROOT = Path("/mnt/cloudrobo/sfs_turbo_wam/cloudrobo-wam/tmp")
#: 默认读取的子数据集：一个目录就是一集。
DEFAULT_SUB_DATASET = "2025-09-20-17-42-51-000000"
#: 转换模式使用的 pipeline 配置（reader + 字段映射 + writer 参数）。
DEFAULT_PIPELINE_CONFIG = REPO_ROOT / "ego2lerobot" / "configs" / "ego_egoverse.yaml"

# group attrs 里 features.<name>.dtype 的取值，用来判断成员类别。
_IMAGE_DTYPES = {"jpeg", "jpg", "png", "image", "video"}
_JSON_DTYPES = {"json", "annotation_v1", "annotations"}


class EgoVerseReader:
    """In-memory view over one EgoVerse Zarr sub-dataset.

    Numeric features (``left.obs_ee_pose``, ``obs_head_pose``, ...) are loaded
    as trimmed numpy arrays.  Image features stay as Zarr arrays and are decoded
    frame by frame, unless ``load_images=True``.

    ``stride`` / ``max_frames`` select a window of the episode without reading
    the rest; ``frame_indices`` records the source frames that were kept.

    中文：一个子数据集的内存视图。数值特征按选中帧加载为 numpy 数组；
    图像默认只保存 Zarr 句柄，用到的帧才解码，避免整集常驻内存。
    """

    def __init__(
        self,
        path: str | Path,
        *,
        stride: int = 1,
        max_frames: int | None = None,
        load_images: bool = False,
    ):
        """打开子数据集：分类成员、选帧、加载数值数组。

        参数：
            path: 子数据集目录，必须包含 zarr.json。
            stride: 帧抽样步长，1 表示逐帧。
            max_frames: 最多保留多少帧，None 表示不限制。
            load_images: True 时一次性解码所有选中帧的图像（很占内存）。
        """
        # 延迟导入：未安装 zarr 时不影响 ego2lerobot 的其他模块。
        import zarr

        # 1. 参数校验，尽早暴露配置错误。
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be > 0 or None")

        # 2. 定位并打开 group：先确认锚点文件存在，再只读打开。
        self.path = Path(path).expanduser().resolve()
        if not (self.path / "zarr.json").is_file():
            raise FileNotFoundError(f"not an EgoVerse Zarr group (missing zarr.json): {self.path}")

        self._group = zarr.open_group(self.path, mode="r")
        self.attrs: dict[str, Any] = dict(self._group.attrs)
        self.feature_specs: dict[str, Any] = dict(self.attrs.get("features") or {})

        # 3. 初始化结果容器。
        self.warnings: list[str] = []
        self.arrays: dict[str, np.ndarray] = {}
        self.image_arrays: dict[str, Any] = {}
        self.annotations: list[Any] = []
        self.images: dict[str, np.ndarray] | None = None
        self.load_images = bool(load_images)

        # 4. 三步装载：先分类成员并算出总帧数，再选帧，最后只读选中的数值。
        self._collect_streams()
        self.frame_indices = self._select_frames(stride=stride, max_frames=max_frames)
        self._load_numeric_arrays()
        # 5. 可选：一次性解码全部选中帧的图像。
        if self.load_images:
            self.images = {
                name: self._decode_all(name) for name in self.image_arrays
            }

    # --------------------------------------------------------------- loading

    def _collect_streams(self) -> None:
        """Split group members into numeric / image / annotation streams.

        中文：遍历 group 成员，分成数值流、图像流、标注流，并用最短的
        时间流长度作为总帧数。长度不一致时只记 warning，不中断读取。
        """

        lengths: dict[str, int] = {}
        for name, node in self._group.members():
            shape = getattr(node, "shape", None)
            if shape is None:
                continue  # 嵌套 group，不是逐帧特征
            if _is_annotation(name, node, self.feature_specs.get(name)):
                # 标注不是逐帧流，整体解析一次即可。
                self.annotations = _load_annotations(node)
                continue
            if _is_image(name, node, self.feature_specs.get(name)):
                # 图像只保存句柄，读取到具体帧时才解码。
                self.image_arrays[name] = node
                lengths[name] = int(shape[0])
                continue
            if getattr(node, "dtype", None) is not None and node.dtype.kind == "O":
                # 未知 object 流：按懒加载字节流处理，而不是直接报错。
                self.image_arrays[name] = node
                lengths[name] = int(shape[0])
                self.warnings.append(f"{name}: object dtype treated as lazy byte stream")
                continue
            if shape and np.prod(shape) > 0:
                # 普通数值流：记录长度，参与总帧数计算。
                lengths[name] = int(shape[0])

        if lengths:
            # 取最短流，防止越界；长度不一致时给出告警而不是失败。
            self.num_frames = min(lengths.values())
            disagree = {name: length for name, length in lengths.items() if length != self.num_frames}
            if disagree:
                self.warnings.append(
                    f"streams disagree on frame count; using {self.num_frames}: {disagree}"
                )
        else:
            self.num_frames = 0

        # attrs.total_frames 只是声明值，以数组实际长度为准。
        declared = self.attrs.get("total_frames")
        if declared is not None and int(declared) != self.num_frames:
            self.warnings.append(
                f"attrs.total_frames={declared} but arrays hold {self.num_frames} frames"
            )

    def _select_frames(self, *, stride: int, max_frames: int | None) -> np.ndarray:
        """Return the source frame indices to keep.

        中文：先按 stride 抽样，再截断到 max_frames 个；返回的是源帧号。
        """
        indices = np.arange(0, self.num_frames, stride, dtype=np.int64)
        if max_frames is not None:
            indices = indices[:max_frames]
        if indices.size == 0:
            raise ValueError(f"no frames selected from {self.path} (stride={stride})")
        return indices

    def _load_numeric_arrays(self) -> None:
        """Load the selected rows of every numeric stream.

        中文：把选中帧的数值数据一次性读进内存；图像、标注、0 维标量
        和空数组不在这里处理。
        """
        for name, node in self._group.members():
            if name in self.image_arrays:
                continue  # 图像流保持懒加载
            shape = getattr(node, "shape", None)
            if shape is None or name not in self._group.array_keys():
                continue  # 嵌套 group，不是数组
            if getattr(node, "dtype", None) is not None and node.dtype.kind == "O":
                continue  # object 流已在 _collect_streams 归为图像
            if not shape or np.prod(shape) == 0:
                continue  # 空数组没有可读内容
            if node.ndim == 0:
                # 0 维标量对所有帧都相同，只取一次。
                self.arrays[name] = np.asarray(node[()])
            else:
                # 只读选中的帧，长序列也不会整段载入。
                self.arrays[name] = np.asarray(node[self.frame_indices])

    # ------------------------------------------------------------------ info

    # 以下属性直接透出 group attrs 里的元数据。
    @property
    def fps(self) -> int:
        """采集帧率；缺失时返回 0。"""
        return int(self.attrs.get("fps", 0) or 0)

    @property
    def embodiment(self) -> str | None:
        """具身配置名，例如 ``aria_bimanual``。"""
        return self.attrs.get("embodiment")

    @property
    def task_name(self) -> str | None:
        """任务短名，例如 ``fold_clothes``。"""
        return self.attrs.get("task_name")

    @property
    def task_description(self) -> str | None:
        """任务自然语言描述。"""
        return self.attrs.get("task_description")

    @property
    def feature_names(self) -> list[str]:
        """所有数值特征名 + 图像特征名。"""
        return list(self.arrays) + list(self.image_arrays)

    def describe(self) -> dict[str, Any]:
        """Summarise metadata and shapes as a JSON-serialisable dict.

        中文：汇总路径、元数据、帧数、特征形状和告警，供 ``--json`` 使用。
        """
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
        """选中窗口内的帧数（不是源数据总帧数）。"""
        return int(self.frame_indices.size)

    def __getitem__(self, position: int) -> dict[str, Any]:
        """``reader[i]``：返回第 i 个选中帧，不解码图像。"""
        return self.frame_at(position, decode_images=False)

    def iter_frames(self, *, decode_images: bool = True) -> Iterator[dict[str, Any]]:
        """Yield one record per selected frame, in time order.

        中文：按时间顺序逐帧产出；默认解码图像。
        """

        for position in range(len(self)):
            yield self.frame_at(position, decode_images=decode_images)

    def frame_at(self, position: int, *, decode_images: bool = False) -> dict[str, Any]:
        """Return one selected frame by position.

        中文：按「选中窗口内的位置」取一帧，支持负索引。返回 dict 含源帧号、
        元数据、数值特征切片和图像（按需解码或原始 bytes）。
        """
        # 支持负索引，并做越界检查。
        if position < 0:
            position += len(self)
        if not 0 <= position < len(self):
            raise IndexError(f"frame position {position} out of range (0..{len(self) - 1})")

        # 源帧号：用于和原始数据/时间戳对齐。
        source_index = int(self.frame_indices[position])
        record: dict[str, Any] = {
            "frame_index": source_index,
            "embodiment": self.embodiment,
            "task_name": self.task_name,
            "task_description": self.task_description,
        }
        # 数值特征：直接取已加载数组的第 position 行。
        for name, values in self.arrays.items():
            record[name] = values[position]
        # 时间戳单独透出一个便捷字段。
        if "obs_rgb_timestamps_ns" in self.arrays:
            record["timestamp_ns"] = int(self.arrays["obs_rgb_timestamps_ns"][position])
        # 图像：已全量加载则直接取；否则按需解码或返回原始编码字节。
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
        """Decode one image feature frame into an ``(H, W, 3)`` uint8 array.

        中文：解码指定图像特征的第 position 帧；名字或位置非法时报错。
        """

        from PIL import Image

        # 校验特征名和位置，报错时给出可用特征列表。
        if name not in self.image_arrays:
            raise KeyError(f"unknown image feature {name!r}; available: {sorted(self.image_arrays)}")
        if not 0 <= position < len(self):
            raise IndexError(f"frame position {position} out of range")

        node = self.image_arrays[name]
        source_index = int(self.frame_indices[position])
        # 用切片取一个元素，兼容 Zarr 3 对 vlen bytes 的标量包装。
        value = node[source_index:source_index + 1][0]
        return _decode_image_value(value, name=name)

    def _decode_all(self, name: str) -> np.ndarray:
        """Decode every selected frame of one image feature.

        中文：一次性解码该图像特征的所有选中帧，堆叠成 (N, H, W, 3)。
        """
        node = self.image_arrays[name]
        raw = node[self.frame_indices]
        return np.stack([_decode_image_value(value, name=name) for value in raw])


# ----------------------------------------------------------------- helpers


def _is_image(name: str, node: Any, spec: Any) -> bool:
    """判断成员是否为图像流：看名称前缀或 features 里的 dtype。"""
    dtype = str((spec or {}).get("dtype", "")).lower() if isinstance(spec, dict) else ""
    if dtype in _IMAGE_DTYPES:
        return True
    if name.startswith("images.") or ".images." in name:
        return True
    return False


def _is_annotation(name: str, node: Any, spec: Any) -> bool:
    """判断成员是否为标注流：名字叫 annotations 或 dtype 属于 JSON 类。"""
    if name == "annotations":
        return True
    dtype = str((spec or {}).get("dtype", "")).lower() if isinstance(spec, dict) else ""
    return dtype in _JSON_DTYPES


def _load_annotations(node: Any) -> list[Any]:
    """读取标注数组：空数组返回 ``[]``，JSON 字符串会尝试解析。"""
    raw = node[:]
    # 空数组直接返回，避免无意义遍历。
    if getattr(raw, "size", 0) == 0:
        return []
    values = []
    for item in raw:
        # Zarr 可能返回 bytes/np.bytes_，先按 UTF-8 解码。
        if isinstance(item, (bytes, bytearray, np.bytes_)):
            item = bytes(item).decode("utf-8", errors="replace")
        # 能解析成 JSON 就用结构化结果，否则保留原始字符串。
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                pass
        values.append(item)
    return values


def _decode_image_value(value: Any, *, name: str) -> np.ndarray:
    """把 Zarr 里的一个元素解码为 ``(H, W, 3)`` uint8。"""
    from PIL import Image

    # Zarr 3 对标量选择可能返回 0-d object 数组，先逐层拆到真实值。
    while isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, np.bytes_):
        value = bytes(value)

    # 三种输入形态：编码后的 bytes / PIL Image / 已经是 ndarray。
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
    """把 ``--sub-dataset`` 解析成真实目录。

    中文：先按路径判断；不是现有目录就拼到 root 下再找；都找不到时报错。
    """
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

    中文：对外主入口。解析路径 → 构造 reader → 默认打印摘要 → 返回 reader，
    方便在 notebook 或脚本里一行拿到可检查对象。
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
    """打印元数据、特征形状和告警。"""
    stream = stream or sys.stdout
    # 第一行：数据位置和任务级元数据。
    print(f"[egoverse] {reader.path}", file=stream)
    print(
        f"[egoverse] task={reader.task_name!r} embodiment={reader.embodiment!r} "
        f"fps={reader.fps} frames={reader.num_frames} selected={len(reader)}",
        file=stream,
    )
    # 逐个列出数值流和图像流。
    for name, values in reader.arrays.items():
        print(f"  array  {name:<28} shape={tuple(np.shape(values))} dtype={np.asarray(values).dtype}", file=stream)
    for name, node in reader.image_arrays.items():
        print(f"  image  {name:<28} frames={node.shape[0]} dtype={node.dtype}", file=stream)
    # 标注数量和读取过程中的告警。
    print(f"  annotations={len(reader.annotations)}", file=stream)
    for warning in reader.warnings:
        print(f"  warning: {warning}", file=stream)


def convert_to_lerobot(
    sub_dataset: str | Path = DEFAULT_SUB_DATASET,
    *,
    root: str | Path | None = DEFAULT_EGOVERSE_ROOT,
    config: str | Path = DEFAULT_PIPELINE_CONFIG,
    output_root: str | Path | None = None,
    repo_id: str | None = None,
    stride: int = 1,
    max_frames: int | None = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> "TaskResult":
    """Convert one EgoVerse sub-dataset into a LeRobot dataset.

    中文：把一集子数据集真正转换成 LeRobot 格式。复用生产链路
    ``PipelineConfig -> reader -> mapping -> LeRobotDataset -> 原子提交``，
    只是把配置换成内存对象，方便在调试器里单集运行；产物与
    ``python -m ego2lerobot convert`` 完全一致。

    参数：
        sub_dataset: 子数据集名字（在 root 下查找）或直接给目录路径。
        root: 子数据集根目录，仅在 sub_dataset 是名字时使用。
        config: pipeline 配置，决定 reader、字段映射和 writer 行为。
        output_root: 输出根目录；None 表示用配置里的 output_root。
        repo_id: 覆盖配置里的 repo_id，None 表示不覆盖。
        stride: 帧抽样步长，1 表示逐帧。
        max_frames: 最多转换多少帧；None 表示整集。
        overwrite: True 时先删掉已有产物（包括 ``_SUCCESS``）再重转。
        verbose: 是否打印转换结果和数据集摘要。

    返回：
        ``TaskResult``（status=ok/skipped/failed、frames、elapsed_s、error）。
    """
    import shutil
    from dataclasses import replace

    from ego2lerobot.config import PipelineConfig
    from ego2lerobot.convert.worker import process_task

    if stride < 1:
        raise ValueError("stride must be >= 1")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be > 0 or None")

    path = _resolve_sub_dataset(sub_dataset, root)
    pipeline = PipelineConfig.load(config)
    if pipeline.reader.kind != "ego_egoverse":
        raise ValueError(
            f"convert_to_lerobot needs an ego_egoverse config, got {pipeline.reader.kind!r}"
        )

    # 1. 用命令行的 stride / max_frames 覆盖配置里的取帧窗口。
    pipeline = _with_reader_window(pipeline, stride=stride, max_frames=max_frames)
    # 2. 应用输出位置和 repo_id 覆盖。
    if repo_id:
        pipeline = replace(pipeline, repo_id=repo_id)
    if output_root is not None:
        pipeline = replace(pipeline, output_root=Path(output_root).expanduser().resolve())

    # 3. 构造与 CLI 一致的 ConversionTask（相对路径 + 哈希，避免重名）。
    task = _build_conversion_task(path, pipeline.output_root)
    if overwrite:
        # process_task 只会清理未提交的目录；已提交的要先手动删掉。
        shutil.rmtree(task.output_path, ignore_errors=True)
    task.output_path.parent.mkdir(parents=True, exist_ok=True)

    # 4. 单集直接调用 worker，避免启动 Ray。
    result = process_task(task, pipeline)

    if verbose:
        print(
            f"[egoverse] -> lerobot status={result.status} frames={result.frames} "
            f"elapsed={result.elapsed_s}s",
            flush=True,
        )
        print(f"[egoverse] output: {result.output_path}", flush=True)
        if result.error:
            print(f"[egoverse] error: {result.error}", flush=True)
        elif result.status in {"ok", "skipped"}:
            _print_lerobot_summary(Path(result.output_path), pipeline.repo_id)
    return result


def _with_reader_window(
    pipeline: "PipelineConfig", *, stride: int, max_frames: int | None
) -> "PipelineConfig":
    """Override the reader's frame window with the CLI stride / max_frames.

    中文：``max_frames`` 是「帧数」，reader 的 ``stop`` 是「帧号」，
    所以换算成 ``start + max_frames * stride`` 才能拿到正好 N 帧。
    """
    from dataclasses import replace

    from ego2lerobot.readers import make_reader

    if stride == pipeline.reader.stride and max_frames is None:
        return pipeline

    settings = dict(pipeline.reader.describe())
    start = int(settings.get("start") or 0)
    settings["stride"] = int(stride)
    if max_frames is not None:
        settings["stop"] = start + int(max_frames) * int(stride)
    return replace(pipeline, reader=make_reader(settings))


def _build_conversion_task(path: Path, output_root: Path) -> "ConversionTask":
    """Build the same ``ConversionTask`` the CLI would create for this episode.

    中文：用 ``build_tasks`` 复用生产命名规则，输出目录名带短哈希，
    同一子数据集重复转换时总是落在同一个位置，便于断点续跑。
    """
    from ego2lerobot.common.discovery import DiscoveredFile
    from ego2lerobot.convert.run import build_tasks

    anchor = path / "zarr.json"
    if not anchor.is_file():
        raise FileNotFoundError(f"not an EgoVerse Zarr group (missing zarr.json): {path}")

    stat = anchor.stat()
    item = DiscoveredFile(
        path=anchor,
        root=path.parent,
        # 用子数据集目录名做相对路径，输出目录更易读（不含 "zarr"）。
        relative_path=path.name,
        size=stat.st_size,
        mtime=stat.st_mtime,
    )
    return build_tasks([item], output_root)[0]


def _print_lerobot_summary(root: Path, repo_id: str) -> None:
    """Reopen the converted dataset and print its metadata.

    中文：用 ``LeRobotDataset`` 读回产物，确认帧数、特征和 dtype 都正确。
    这一步失败说明转换产物不可用，会直接把异常抛给调用方。
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=repo_id, root=root)
    print(
        f"[egoverse] dataset repo_id={repo_id} fps={dataset.fps} "
        f"episodes={dataset.num_episodes} frames={dataset.num_frames}",
        flush=True,
    )
    for name, feature in dataset.features.items():
        print(
            f"  feature {name:<32} dtype={feature['dtype']} "
            f"shape={feature.get('shape')}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """定义命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    # 数据定位：子数据集名字 + 根目录。
    parser.add_argument("--sub-dataset", default=DEFAULT_SUB_DATASET)
    parser.add_argument("--root", type=Path, default=DEFAULT_EGOVERSE_ROOT)
    # 读取窗口：步长、帧数上限、是否一次性解码图像。
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--load-images", action="store_true")
    # 转换模式：把选中窗口写成真正的 LeRobot 数据集。
    parser.add_argument(
        "--to-lerobot",
        action="store_true",
        help="convert the sub-dataset into a LeRobot dataset instead of only printing stats",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PIPELINE_CONFIG,
        help="pipeline config used by --to-lerobot (default: ego_egoverse.yaml)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="LeRobot output root; defaults to the config's output_root",
    )
    parser.add_argument("--repo-id", default=None, help="override the config's repo_id")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="delete an existing converted dataset before writing",
    )
    # 检查模式：--json 打印元数据，--show-frame 打印单帧，都不给则打印摘要。
    parser.add_argument(
        "--show-frame",
        type=int,
        default=None,
        help="print one decoded frame (arrays + image shapes) and exit",
    )
    parser.add_argument("--json", action="store_true", help="print describe() as JSON")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：按参数选择「转换成 LeRobot」或「读取检查」。"""
    args = parse_args(argv)

    # 1. --to-lerobot：写出 LeRobot 数据集；退出码反映转换是否成功。
    if args.to_lerobot:
        result = convert_to_lerobot(
            args.sub_dataset,
            root=args.root,
            config=args.config,
            output_root=args.output_root,
            repo_id=args.repo_id,
            stride=args.stride,
            max_frames=args.max_frames,
            overwrite=args.overwrite,
        )
        return 0 if result.status in {"ok", "skipped"} else 1

    # 2. 检查模式：加载子数据集并打印摘要。
    reader = generate_egoverse_demo(
        args.sub_dataset,
        root=args.root,
        stride=args.stride,
        max_frames=args.max_frames,
        load_images=args.load_images,
        verbose=not args.json,
    )

    # 3. --json：只输出可序列化的元数据。
    if args.json:
        print(json.dumps(reader.describe(), ensure_ascii=False, indent=2))
        return 0

    # 4. --show-frame：解码并打印一帧的每个字段。
    if args.show_frame is not None:
        record = reader.frame_at(args.show_frame, decode_images=True)
        print(f"[egoverse] frame {record['frame_index']}:", flush=True)
        for name, value in record.items():
            if isinstance(value, np.ndarray):
                print(f"  {name:<28} shape={value.shape} dtype={value.dtype}")
            else:
                print(f"  {name:<28} {value!r}")
    # 5. 默认：摘要已在 generate_egoverse_demo 中打印，这里直接返回。
    return 0


if __name__ == "__main__":
    # 作为脚本或模块执行时，用 main() 的返回值作为进程退出码。
    raise SystemExit(main())
