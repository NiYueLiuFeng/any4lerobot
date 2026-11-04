# Ego 数据读取实现指南

本文说明在 `ego2lerobot` 中接入一种**新的 ego 数据格式**需要做什么，重点回答两个问题：

1. 数据字段映射（`features` / `source` / `transforms`）怎么配置；
2. 非 mcap、非 h5 的格式（尤其是**一个 episode 由多个零散文件组成**）是否要重写 reader，以及怎么实现。

先给结论：

| 数据情况 | 只改 YAML | 新写 Reader | 改 discovery / worker |
|---|---|---|---|
| 容器还是 h5，只是 dataset 路径不同 | ✅ | ❌ | ❌ |
| 容器还是 mcap，只是 topic / field 不同 | ✅ | ❌ | ❌ |
| 新容器：csv、parquet、zarr、目录散文件、私有二进制等 | 部分 | ✅ 必须 | 通常要改文件发现粒度 |
| 一个文件里含多个 episode | 部分 | ✅ | ✅ 需要自定义 worker（当前 worker 每题只 `save_episode()` 一次） |

核心分工是固定的：

```text
原始文件/目录
   │
   ▼
Reader（唯一懂格式的层）          readers/*.py
   │  iter_frames() -> 每帧一个 dict
   ▼
Mapping（只懂字段名，不懂格式）   mapping/config.py + YAML
   │  features / task / transforms
   ▼
LeRobotDataset.add_frame()       convert/worker.py
   │
   ▼
每个 task 一个 LeRobot 数据集 + _SUCCESS
```

**Mapping 层完全不知道文件格式**，所以“怎么把 csv 某一列读成向量”无法只靠 YAML 表达；只要不是 h5/mcap 这类已经实现的后端，就必须新增一个 `EpisodeReader` 子类。

---

## 1. 字段映射配置

映射配置就是 pipeline YAML 里的 `features` / `task`（以及 `fps`、`robot_type`）。加载入口是 `mapping/config.py:132` 的 `MappingConfig`，实际解析 `features` 的是 `_parse_feature()`（`mapping/config.py:214`）。

### 1.1 完整骨架

```yaml
version: 1
repo_id: ego/my_dataset              # 输出数据集的 repo_id
output_root: /data/lerobot/my_ego    # 每个 task 的输出根目录
fps: 30                              # LeRobot 时间轴按此常量生成
robot_type: ego

# task 是 LeRobot 的 metadata 字段，不是 feature
task:
  source: attrs.instruction          # 每帧记录里的路径
  default: ego manipulation          # 找不到时用这个常量

features:
  observation.state:
    dtype: float32
    shape: [7]
    source: proprio.qpos             # reader 输出的 key

  action:
    dtype: float32
    shape: [7]
    source: [proprio.cmd, action]    # 候选列表：第一个存在的胜出
    required: false

  observation.images.head:
    dtype: image
    shape: [480, 640, 3]
    source: cameras.head
    transforms:
      - op: as_image
        dtype: uint8
        channel_order: HWC

  observation.images.wrist:
    dtype: image
    shape: [480, 640, 3]
    source: cameras.wrist
    transforms:
      - op: as_image
        dtype: uint8
        channel_order: CHW           # 源数据是 CHW 时

reader:
  # 见第 2 节；不同格式的 reader 配置项不同
  type: h5
  arrays:
    proprio.qpos: /observations/qpos
    proprio.cmd: /action
    cameras.head: /observations/images/head

writer:
  use_videos: false
  image_writer_threads: 2
```

`PipelineConfig` 允许的顶层键是固定的（`config.py:48`）：
`version`、`fps`、`robot_type`、`task`、`features`、`reader`、`repo_id`、`output_root`、`writer`、`aggregate`。
多写一个未知键会直接报错，这是有意的严格校验。

### 1.2 `features` 的每个字段

`features` 的 key 就是**最终写进 LeRobot 的列名**。常用命名：

| key | 含义 |
|---|---|
| `observation.state` | 本体状态 |
| `action` | 动作（ego 数据没有 action 时需要构造，见 1.6） |
| `observation.images.<camera>` | 图像/视频特征，LeRobot 依赖这个前缀识别相机 |
| 其它自定义列 | 会作为普通列写入 |

每个 feature 支持：

| 键 | 必填 | 说明 |
|---|---|---|
| `dtype` | ✅ | NumPy dtype（`float32`、`int64`、`uint8`…）或特殊值 `image` / `video` / `string` |
| `shape` | 建议 | 单帧形状（**不含时间维**），如 `[7]`、`[480, 640, 3]`；LeRobot 会逐帧校验 |
| `source` | 二选一 | 在 reader 产出的 record 里查找的路径，字符串或候选列表 |
| `sources` | 二选一 | `source` 的别名，等价写法 |
| `default` | 可选 | 所有 `source` 都找不到时使用的常量 |
| `required` | 可选 | 默认 `true`；设为 `false` 时缺失会得到 `None` 而不是报错 |
| `transforms` | 可选 | 按顺序应用的变换列表 |
| `names` | 可选 | 维度名元信息，原样写入 LeRobot feature |

约束（加载期就会校验）：

- `dtype` 必须在允许集合内；`shape` 必须是正整数列表；
- 既没有 `source` 也没有 `default` 且 `required: true` 的 feature 会被拒绝；
- `source` 和 `sources` 不要同时写；
- feature 名不能为空。

### 1.3 `source` 的解析规则

实现见 `_lookup()`（`mapping/config.py:324`），按以下顺序解析：

1. **字面 key 优先**：如果 record 直接有 `{"proprio.qpos": ...}`，直接命中。
   因此 reader 推荐输出扁平的点号 key，映射无需嵌套结构。
2. 否则按 `.` 逐层下钻：dict 用 key，list/tuple 用整数下标。
3. 最后回退到对象属性（适配轻量对象/h5 attrs 命名空间）。

示例：

```yaml
source: attrs.instruction        # record["attrs"]["instruction"]
source: cameras.head.data        # record["cameras"]["head"]["data"]
source: [proprio.cmd, action]    # 候选：先试 proprio.cmd，再试 action
```

注意：

- **`None` 是合法值**，不会被当成“没找到”；只有完全不存在的路径才返回缺失哨兵。
- `required: false` 缺失时返回 `None`，但 LeRobot 的数值/图像列通常不接受 `None`。它主要用于诊断或确实允许空值的元数据列，不要给 `observation.state` / `action` 用。

### 1.4 `transforms` 变换清单

注册表在 `mapping/config.py:460`，解析在 `_parse_transform()`（`mapping/config.py:304`）。每一项写成字符串或 `{op: ..., 参数...}`。

| op | 参数 | 用途 |
|---|---|---|
| `identity` | — | 原样返回 |
| `astype` | `dtype` | 类型转换，最常用于把 reader 的 float64/int 转成声明的 dtype |
| `reshape` | `shape` | 重排形状 |
| `scale` | `factor`, `offset` | 线性变换：`x * factor + offset`，如毫米→米 |
| `clip` | `min`, `max` | 截断 |
| `slice` | `start`, `stop`, `step` | 取子向量，如只保留前 7 维 |
| `index` | `index` | 取单个元素或某一维 |
| `transpose` | `axes` | 轴置换 |
| `squeeze` | `axis` | 去长度为 1 的维 |
| `expand_dims` | `axis` | 加长度为 1 的维 |
| `to_bool` | — | 字符串/数值转 bool |
| `to_string` | — | 转字符串 |
| `as_image` | `dtype`, `channel_order` | 图像规范化：float 0~1 自动乘 255、裁剪、CHW→HWC、转 uint8 |

典型写法：

```yaml
# 毫米转米
transforms:
  - op: scale
    factor: 0.001

# 只取第 0、1、2 维
transforms:
  - op: slice
    start: 0
    stop: 3

# 源图像是 CHW float32 [0,1]
transforms:
  - op: as_image
    dtype: uint8
    channel_order: CHW
```

变换按列表顺序执行；某个变换抛错时，报错会带上 feature 名和命中的 source，便于定位。

### 1.5 `task` 的写法

三种形式：

```yaml
task: ego manipulation                  # 常量
task:
  source: attrs.instruction             # 每 episode 的指令
  default: ego manipulation
task:
  source: [attrs.instruction, env.task_name]
  default: convert
```

`task` 会作为 `frame["task"]` 写入。`convert/worker.py:107-109` 会拒绝空字符串，所以 reader 要么每帧给出 `attrs.instruction`，要么配置里必须有非空 `default`。

### 1.6 关于 `action`

LeRobot 数据集要求有 `action`。ego 数据如果本身没有动作，需要在 reader 或映射阶段构造，常见策略：

- 下一帧状态（`state[t+1]`）；
- 未来轨迹的某一段；
- 相邻状态差分；
- 显式空 action（固定维度全零），并在数据卡里说明。

`transforms` 目前只能做逐帧的数组运算，**不能引用其它帧**。所以“下一帧状态”这类跨帧构造应在 reader 的 `iter_frames()` 里完成，再以一个新 key（如 `proprio.next_qpos`）输出。

---

## 2. 是否需要重写 Reader

### 2.1 结论

**只要容器格式不是 h5/mcap，就必须新增一个 `EpisodeReader` 子类。** 字段映射只描述“从 record 的哪个 key 取值”，无法描述“如何打开 parquet / 如何拼接散文件 / 如何解码私有二进制”。

以下情况**不需要**新 reader，只改 YAML：

- HDF5 里 dataset 路径不同 → 改 `reader.arrays`；
- HDF5 的时间轴不在第 0 维 → 改 `reader.time_axis` 或每个 stream 的 `axis`；
- mcap 的 topic、消息字段、同步方式不同 → 改 `reader.topics` / `fields` / `sync`。

以下情况**需要**新 reader：

- csv / parquet / jsonl / npy / zarr / rosbag / 私有二进制；
- 一个 episode 是**目录**，由图像、状态、元数据等零散文件组成；
- 需要自定义时间对齐、跨帧动作构造、图像解码逻辑。

### 2.2 Reader 契约

接口在 `readers/base.py:26`：

```python
class EpisodeReader(ABC):
    kind: str = "base"

    @abstractmethod
    def iter_frames(self, path) -> Iterator[dict[str, Any]]: ...

    @abstractmethod
    def describe(self) -> dict[str, Any]: ...

    def validate(self, path) -> None: ...
```

约定：

- `path` 是**一个 ConversionTask 的输入路径**。它可以是文件，也可以是目录（由你的 reader 和 discovery 决定）。
- `iter_frames()` 必须**逐帧流式**产出 record；每个 record 是一次 `add_frame` 的输入。
- 同一 episode 的所有帧应具有相同的 key 集合。
- record 的 key 就是 YAML 里 `source` 使用的名字。推荐扁平点号风格（`"proprio.qpos"`），映射层支持字面 key 优先。
- 需要每帧携带元数据时，放一个 `record["attrs"] = {...}`，YAML 里用 `source: attrs.instruction` 取。
- 图像直接返回 `(H, W, 3)` 的 `uint8` NumPy 数组即可；源数据是 CHW 或浮点时交给 `as_image` 转换。
- `describe()` 的返回值会写进 `_conversion.json` 和 `_run_summary.json`，**必须可 JSON 序列化**（Path 要转 `str`）。
- `validate()` 做便宜的预检即可：后缀/目录存在、关键文件存在、各流帧数一致。真正解码放在 `iter_frames()`。
- `accepts_input()`（`readers/base.py`）默认接受所有候选文件。若容器内部也有同名元数据文件（Zarr v3 的每个 array 目录都含 `zarr.json`），重写它只接受真正的 episode 根，否则 discovery 会把数组节点也当成 episode。
- 失败时抛带路径的异常；`convert/worker.py` 会写 `<output>.failed.json` 并继续其它任务。

### 2.3 注册新 reader

工厂表在 `readers/__init__.py:17`：

```python
# readers/__init__.py
from .ego_dir import EgoDirReader

_READERS = {
    "h5": H5Reader,
    "hdf5": H5Reader,
    "mcap": McapReader,
    "ego_dir": EgoDirReader,   # 新增
}
```

`make_reader()` 会调用类的 `from_config(config)`，所以新类要实现这个 classmethod，并像 `H5Reader` / `McapReader` 一样校验未知键。

### 2.4 配套但容易漏的改动

| 位置 | 为什么 |
|---|---|
| `convert/run.py:32` `reader_includes()` | 未知 `kind` 会退化成 `("**/*",)`，把目录里**每个文件**都变成一个 task。必须为新 kind 返回每 episode 唯一的锚点 glob，或运行时显式传 `--include`。 |
| `readers/base.py` `accepts_input()` | 锚点文件名可能在容器内部重复出现（Zarr v3 的 array 目录也有 `zarr.json`）。重写它过滤掉非 episode 根；`plan_conversion()` 会把它作为 `DiscoveryConfig.file_filter`。 |
| `convert/run.py:176-178` `make_executor_config()` | 默认只给 h5 加 `h5py` 预检。新 reader 依赖 PIL/pandas/pyarrow 等时，把模块名加进 `preflight_imports`，包名加进 `preflight_distributions`。 |
| 新 YAML | 见第 4 节。 |
| `tests/test_core.py` | 至少覆盖 `validate`、帧数、`start/stop/stride`、映射一次。 |

`reader_includes()` 示例：

```python
def reader_includes(pipeline: PipelineConfig) -> tuple[str, ...]:
    if pipeline.reader.kind == "h5":
        return ("**/*.h5", "**/*.hdf5", "**/*.hdf")
    if pipeline.reader.kind == "mcap":
        return ("**/*.mcap",)
    if pipeline.reader.kind == "ego_dir":
        return ("**/episode.json",)      # 每个 episode 唯一的锚点文件
    return ("**/*",)
```

---

## 3. 多个零散文件组成的 Episode

### 3.1 问题在哪

当前执行链是**一个文件 = 一个 task = 一个 episode**：

```text
discover_files()            common/discovery.py:237   # 只产出文件
  -> build_tasks()          convert/run.py:93          # 每个文件一个 ConversionTask
  -> process_batch()        convert/worker.py:58       # 每个 task 独立处理
  -> iter_frames(input_path)
  -> dataset.save_episode() convert/worker.py:116      # 每题只调用一次
```

如果直接把 `--root` 指向散文件目录并匹配 `**/*`，那么 `head/000001.jpg`、`proprio.csv`、`meta.json` 会各自变成一个 task，全部失败。

所以关键不是“reader 能不能读多文件”，而是**先让 discovery 以 episode 为单位产生 task**。有三种做法。

### 3.2 方案 A（推荐）：每个 episode 放一个唯一锚点文件

让每个 episode 目录里有一个唯一文件（`episode.json` / `meta.json` / `index.jsonl`），discovery 只匹配它：

```text
/data/ego/
  episode_0001/
    episode.json      <-- 锚点，也是元数据
    proprio.csv
    action.npy
    images/head/000000.jpg ...
  episode_0002/
    episode.json
    ...
```

```bash
python -m ego2lerobot convert -c ego2lerobot/configs/ego_dir.yaml \
  --root /data/ego \
  --include '**/episode.json' \
  --ray-address auto --workers 16 --aggregate
```

`reader.iter_frames(anchor_path)` 拿到的就是 `episode.json` 的路径；reader 用 `Path(anchor_path).parent` 找到整个 episode 目录，自行拼接兄弟文件。**discovery、task_id、输出路径、断点续跑全部不用改**：输出目录和 task_id 由锚点的相对路径决定，天然每个 episode 一个。

这是改动最小、与现有 Ray/断点续跑/聚合完全兼容的方案。只要原始数据能接受“加一个锚点文件”，就选它。

### 3.3 方案 B：自定义 discovery，直接产出 episode 目录

如果数据不能加锚点文件（例如上游严格只给 `episode_xxx/` 目录），就在**驱动端**把目录包装成 `DiscoveredFile`：

```python
from pathlib import Path

from ego2lerobot.common.discovery import DiscoveredFile
from ego2lerobot.common.types import ConversionTask
from ego2lerobot.convert.run import ConversionPlan, build_tasks, run_conversion

root = Path("/data/ego").resolve()

files = []
for episode_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    stat = episode_dir.stat()
    files.append(
        DiscoveredFile(
            path=episode_dir.resolve(),
            root=root,
            relative_path=episode_dir.name,
            size=stat.st_size,
            mtime=stat.st_mtime,
            metadata={"kind": "episode_dir"},
        )
    )

plan = ConversionPlan(
    files=files,
    tasks=build_tasks(files, output_root=Path("/data/lerobot/my_ego")),
    roots=[root],
)

results, executor = run_conversion(
    pipeline, config_path, plan, output_root=..., ray_address="auto", workers=16
)
```

相应地，`EgoDirReader.iter_frames()` / `validate()` 的第一句要同时接受文件或目录：

```python
base = Path(path)
base = base if base.is_dir() else base.parent
```

注意：

- `DiscoveredFile.path` 是目录时，`suffix` 为空；`build_output_path()` 会用目录名生成输出路径，仍然唯一。
- CLI 的 `discover/convert` 子命令仍走 `discover_files()`，方案 B 需要你在脚本/notebook 里调用上面的 API，或者给 `discovery.py` 增加 `discover_episode_dirs()` 并接到 `plan_conversion`。
- preflight 的 `sample_input` 会变成目录，`os.access(dir, os.R_OK)` 仍然可用。

### 3.4 方案 C：manifest 索引

适用于 episode 定义在外部清单里的场景：先生成 `episodes.jsonl`，每行一个 episode 及成员文件路径、元数据；再写一个 reader 读取该清单并按 episode 产出。

- 若“一个 manifest 文件 = 一个 episode”：把 manifest 当锚点，方案 A 直接可用；
- 若“一个 manifest 覆盖全部 episode”：reader 需要从指定 episode 切片，建议仍按方案 B 在驱动端拆成多个 task，避免一个 task 里写多个 episode。

### 3.5 多路数据对齐

散文件通常长度或采样率不一致。reader 必须自己决定对齐策略：

| 场景 | 做法 |
|---|---|
| 所有流都是等长数组 | 直接按帧下标 join（最简单，推荐） |
| 各流有 `timestamp_ns`，频率不同 | 选一个 reference 流，其它流用最近邻/保持/精确匹配；可参考 `readers/mcap.py` 的 `_pick()` |
| 变帧率 | **在 reader 里重采样到固定的 `fps`**，再逐帧输出 |
| 某流缺失尾部 | 明确策略：截断到最短（严格）、补齐（hold last）、或报错 |

必须注意：**LeRobot 时间轴由 YAML 的 `fps` 常量生成**，reader 输出里的 `timestamp_ns` 不会自动变成 LeRobot 时间戳。变帧率数据若不重采样，训练侧会看到错误的速度。

---

## 4. 完整示例：`ego_dir` reader

下面实现“一个 episode 一个目录、多路零散文件”的最小可用 reader。把它保存为 `ego2lerobot/readers/ego_dir.py`。

### 4.1 数据布局

```text
episode_0001/
  episode.json          # {"instruction": "...", "fps": 30}
  proprio.csv           # 表头 qpos_0..qpos_6，一行一帧
  action.npy            # (T, 7) float32
  images/
    head/
      000000.jpg
      000001.jpg
      ...
```

### 4.2 reader 实现

```python
"""ego_dir: one episode = a directory of scattered files.

Config::

    reader:
      type: ego_dir
      meta: episode.json
      streams:
        proprio.qpos:
          file: proprio.csv
          columns: [qpos_0, qpos_1, qpos_2, qpos_3, qpos_4, qpos_5, qpos_6]
        proprio.cmd:
          file: action.npy
        cameras.head:
          dir: images/head
          pattern: "*.jpg"
      start: 0
      stop: null
      stride: 1
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from .base import EpisodeReader

_CSV = "csv"
_NPY = "npy"
_IMAGES = "images"


@dataclass(frozen=True)
class _Stream:
    source: str
    kind: str
    path: str                     # 相对 episode 目录
    columns: tuple[str, ...] = ()
    pattern: str = "*.jpg"


class EgoDirReader(EpisodeReader):
    kind = "ego_dir"

    def __init__(
        self,
        streams: Mapping[str, Any],
        *,
        meta: str | None = "episode.json",
        start: int = 0,
        stop: int | None = None,
        stride: int = 1,
    ):
        if not streams:
            raise ValueError("reader.streams must not be empty for an ego_dir reader")
        if stride < 1:
            raise ValueError("reader.stride must be >= 1")
        if start < 0:
            raise ValueError("reader.start must be >= 0")
        if stop is not None and stop <= start:
            raise ValueError("reader.stop must be greater than reader.start")

        self.streams: list[_Stream] = []
        for source, raw in streams.items():
            if not isinstance(raw, dict):
                raise ValueError(f"reader.streams[{source!r}] must be a mapping")

            if "file" in raw:
                path = str(raw["file"])
                suffix = Path(path).suffix.lower()
                if suffix in {".csv", ".tsv"}:
                    columns = raw.get("columns")
                    if not isinstance(columns, list) or not columns or not all(
                        isinstance(item, str) and item for item in columns
                    ):
                        raise ValueError(
                            f"reader.streams[{source!r}] csv needs a non-empty 'columns' list"
                        )
                    self.streams.append(
                        _Stream(source, _CSV, path, tuple(columns))
                    )
                elif suffix == ".npy":
                    self.streams.append(_Stream(source, _NPY, path))
                else:
                    raise ValueError(
                        f"reader.streams[{source!r}] unsupported file suffix {suffix!r}"
                    )
            elif "dir" in raw:
                self.streams.append(
                    _Stream(
                        source,
                        _IMAGES,
                        str(raw["dir"]),
                        pattern=str(raw.get("pattern", "*.jpg")),
                    )
                )
            else:
                raise ValueError(
                    f"reader.streams[{source!r}] needs 'file' or 'dir'"
                )

        self.meta = meta
        self.start = int(start)
        self.stop = stop if stop is None else int(stop)
        self.stride = int(stride)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "EgoDirReader":
        allowed = {"type", "streams", "meta", "start", "stop", "stride"}
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"unknown ego_dir reader keys: {sorted(unknown)}")
        return cls(
            config.get("streams") or {},
            meta=config.get("meta", "episode.json"),
            start=int(config.get("start", 0)),
            stop=None if config.get("stop") is None else int(config["stop"]),
            stride=int(config.get("stride", 1)),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "type": self.kind,
            "meta": self.meta,
            "start": self.start,
            "stop": self.stop,
            "stride": self.stride,
            "streams": {
                item.source: {
                    "kind": item.kind,
                    "path": item.path,
                    "columns": list(item.columns),
                    "pattern": item.pattern,
                }
                for item in self.streams
            },
        }

    def validate(self, path: str | Path) -> None:
        base = self._base(path)
        if not base.is_dir():
            raise FileNotFoundError(f"episode directory not found: {base}")
        if self.meta and not (base / self.meta).is_file():
            raise FileNotFoundError(f"episode metadata not found: {base / self.meta}")

        lengths = [self._length(item, base) for item in self.streams]
        if len(set(lengths)) > 1:
            detail = ", ".join(
                f"{item.source}:{length}"
                for item, length in zip(self.streams, lengths)
            )
            raise ValueError(f"streams disagree on frame count ({detail})")
        if not lengths or lengths[0] == 0:
            raise ValueError(f"empty episode: {base}")

    def iter_frames(self, path: str | Path) -> Iterator[dict[str, Any]]:
        base = self._base(path)

        meta: dict[str, Any] = {}
        if self.meta:
            meta = json.loads((base / self.meta).read_text(encoding="utf-8"))

        handles = [(item, self._open(item, base)) for item in self.streams]
        lengths = [self._length(item, base) for item in self.streams]
        if len(set(lengths)) > 1:
            detail = ", ".join(
                f"{item.source}:{length}"
                for item, length in zip(self.streams, lengths)
            )
            raise ValueError(f"streams disagree on frame count ({detail})")

        total = lengths[0]
        stop = total if self.stop is None else min(self.stop, total)
        if self.start >= stop:
            raise ValueError(
                f"empty frame range: start={self.start} stop={stop} total={total}"
            )

        for index in range(self.start, stop, self.stride):
            record: dict[str, Any] = {
                item.source: self._take(item, handle, index)
                for item, handle in handles
            }
            record["attrs"] = meta      # 供 task: source: attrs.instruction 使用
            yield record

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _base(path: str | Path) -> Path:
        path = Path(path)
        return path if path.is_dir() else path.parent

    @staticmethod
    def _length(item: _Stream, base: Path) -> int:
        if item.kind == _CSV:
            with (base / item.path).open(newline="", encoding="utf-8") as handle:
                return max(0, sum(1 for _ in handle) - 1)   # 减表头
        if item.kind == _NPY:
            array = np.load(base / item.path, mmap_mode="r")
            return int(array.shape[0])
        return len(sorted((base / item.path).glob(item.pattern)))

    @staticmethod
    def _open(item: _Stream, base: Path) -> Any:
        path = base / item.path
        if item.kind == _CSV:
            with path.open(newline="", encoding="utf-8") as handle:
                return list(csv.DictReader(handle))
        if item.kind == _NPY:
            return np.load(path, mmap_mode="r")
        return sorted(path.glob(item.pattern))

    def _take(self, item: _Stream, handle: Any, index: int) -> Any:
        if item.kind == _CSV:
            row = handle[index]
            return np.asarray(
                [float(row[column]) for column in item.columns], dtype=np.float32
            )
        if item.kind == _NPY:
            # copy=True: 返回的帧不能依赖 mmap 句柄的生命周期
            return np.array(handle[index], dtype=np.float32, copy=True)

        from PIL import Image

        image = Image.open(handle[index])
        image.load()
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
```

### 4.3 注册

```python
# readers/__init__.py
from .ego_dir import EgoDirReader

_READERS = {
    "h5": H5Reader,
    "hdf5": H5Reader,
    "mcap": McapReader,
    "ego_dir": EgoDirReader,
}
```

### 4.4 YAML

```yaml
version: 1
repo_id: ego/ego_dir_demo
output_root: /data/lerobot/ego_dir_demo
fps: 30
robot_type: ego

task:
  source: attrs.instruction
  default: ego manipulation

features:
  observation.state:
    dtype: float32
    shape: [7]
    source: proprio.qpos

  action:
    dtype: float32
    shape: [7]
    source: proprio.cmd

  observation.images.head:
    dtype: image
    shape: [480, 640, 3]
    source: cameras.head
    transforms:
      - op: as_image
        dtype: uint8
        channel_order: HWC

reader:
  type: ego_dir
  meta: episode.json
  streams:
    proprio.qpos:
      file: proprio.csv
      columns: [qpos_0, qpos_1, qpos_2, qpos_3, qpos_4, qpos_5, qpos_6]
    proprio.cmd:
      file: action.npy
    cameras.head:
      dir: images/head
      pattern: "*.jpg"
  start: 0
  stop: null
  stride: 1

writer:
  use_videos: false
  image_writer_threads: 2
```

### 4.5 运行

```bash
# 先确认发现粒度：每个 episode 只应出现一次
python -m ego2lerobot discover \
  -c ego2lerobot/configs/ego_dir.yaml \
  --root /data/ego \
  --include '**/episode.json'

# 转换 + 聚合
python -m ego2lerobot convert \
  -c ego2lerobot/configs/ego_dir.yaml \
  --root /data/ego \
  --include '**/episode.json' \
  --ray-address auto --workers 16 --aggregate
```

如果已按 2.4 更新 `reader_includes()`，`--include` 可以省略。但**首次接入时建议显式传**，避免未知 reader kind 回退到 `**/*` 把散文件逐个当 task。

---

## 5. 自测清单

新增 reader 后，建议按下面顺序验证：

1. **配置能加载**
   ```python
   from ego2lerobot.config import PipelineConfig
   PipelineConfig.load("ego2lerobot/configs/ego_dir.yaml")
   ```
2. **`validate()` 正确**
   - 正常 episode 通过；
   - 缺 `episode.json`、csv 缺列、各流帧数不一致时报错且信息包含路径。
3. **帧数与切片**
   - `list(reader.iter_frames(anchor))` 长度等于总帧数；
   - `start/stop/stride` 生效；
   - 第一帧数值与原始 csv/npy 对得上。
4. **映射一次**
   ```python
   record = next(reader.iter_frames(anchor))
   frame = pipeline.mapping.resolve(record)
   assert frame["observation.state"].shape == (7,)
   assert frame["observation.images.head"].dtype.name == "uint8"
   assert isinstance(frame["task"], str) and frame["task"]
   ```
5. **端到端**
   ```python
   from ego2lerobot.convert.worker import process_batch
   from ego2lerobot.common.types import ConversionTask

   task = ConversionTask(
       task_id="episode_0001",
       input_path=anchor,
       output_path=output_root / "episode_0001",
   )
   result = process_batch([task], {"config_path": "ego2lerobot/configs/ego_dir.yaml"})
   assert result[0].status == "ok", result[0].error
   ```
6. **断点续跑**：重复执行同一 task 应返回 `skipped`。
7. **单元测试**：把上面的最小用例加到 `ego2lerobot/tests/test_core.py`，然后
   ```bash
   python -m unittest discover -s ego2lerobot/tests -v
   ```

---

## 6. 常见陷阱

- **`--include` 没设对**：未知 reader kind 默认 `("**/*",)`，目录里每个文件都会变成 task。务必用锚点 glob 或更新 `reader_includes()`。
- **一个 task 写多个 episode**：`convert/worker.py:116` 每个 task 只调用一次 `save_episode()`。多 episode 文件必须先在 reader/上游拆分，或自定义 worker。
- **把整段 episode 读进内存**：`iter_frames()` 必须是生成器。图像逐帧解码，npy 用 `mmap_mode="r"`，不要一次性 `np.load()` 全部数组再 `tolist()`。
- **返回的数组是 mmap 视图**：worker 可能在句柄失效后使用它；数值帧用 `copy=True` 或 `astype(..., copy=True)`。
- **reader 返回的 dtype/shape 与 YAML 声明不一致**：LeRobot 逐帧校验会报错。用 `astype`/`reshape`/`as_image` 在映射层修正，或在 reader 里直接返回目标格式。
- **图像通道顺序**：ROS/部分相机是 CHW，LeRobot 要 HWC；用 `as_image.channel_order: CHW`。
- **文件名自然序**：`000001.jpg` 这种零填充文件名可以用 `sorted(glob)`；若文件名不零填充（`1.jpg`、`10.jpg`、`2.jpg`），必须自己做自然排序。
- **变帧率**：在 reader 内重采样到 YAML 的 `fps`；不要指望 `timestamp_ns` 自动影响 LeRobot 时间轴。
- **`describe()` 不可序列化**：不要返回 `Path`、NumPy 标量、自定义对象；统一转换成 `str`/`int`/`float`/`list`/`dict`。
- **依赖预检**：新 reader 用了 PIL/pandas/pyarrow，记得更新 `make_executor_config()` 的 `preflight_imports`，否则多机环境可能在跑到一半才报 ImportError。
- **路径共享**：原始数据、输出、`probe_dir`、日志必须位于所有 Ray 节点可见的共享文件系统（与 h5/mcap 一致）。

---

## 7. 参考实现

- Reader 契约：`readers/base.py:26`
- HDF5 后端（时间轴、attrs、局部读取）：`readers/h5.py`
- MCAP 后端（topic 解码、nearest/hold/exact 同步）：`readers/mcap.py`
- EgoVerse Zarr v3 后端（多零散成员、图像惰性解码、`accepts_input` 过滤）：`readers/ego_egoverse.py`
- Reader 工厂与注册：`readers/__init__.py:17`
- 映射解析与 transform：`mapping/config.py:132`、`mapping/config.py:214`、`mapping/config.py:324`、`mapping/config.py:460`
- worker（读取 → 映射 → 写入 → 提交）：`convert/worker.py:66`
- 文件发现与 task 构建：`common/discovery.py:237`、`convert/run.py:56`
- 可直接参考的非 h5/mcap 读取示例（Zarr）：`demo/egoverse_demo.py`
- 示例配置：`configs/ego_h5.yaml`、`configs/ego_mcap.yaml`、`configs/ego_egoverse.yaml`


## demo代码测试

```
DEFAULT_EGOVERSE_ROOT = Path("/mnt/cloudrobo/sfs_turbo_wam/cloudrobo-wam/tmp")
DEFAULT_SUB_DATASET = "2025-09-20-17-42-51-000000"

python -m ego2lerobot.demo.egoverse_demo --max-frames 5 --show-frame 0
```