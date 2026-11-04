# Ego 数据分布式转换 LeRobot 方案（基于 Any4LeRobot）

> 目标：将 ego 数据（mcap / h5）分布式读取并转换为 LeRobot v3 格式。
> 结论日期：2026-11-02

## 1. 结论

**复用 `generic_converter` 的任务模型与执行语义，但不直接依赖其 `pipeline.py`；在自己的 `ego2lerobot` 包内实现轻量 runner。**

具体分工：

- **复用/复制**：`ConversionTask`、`BaseAdapter` 契约、`local/ray` 执行参数语义（`generic_converter/adapter.py` + `utils.py` 共约 100 行，MIT）。
- **自行实现**：递归发现、配置驱动的字段映射、可选后端 runner、聚合封装。
- **不直接 import** `generic_converter.pipeline`：它在模块顶层强依赖 `datatrove`，并 monkeypatch LeRobot 的私有聚合函数，版本耦合较重。详见第 9 节。

不要从零用 Ray 重写调度层，但也不要为了复用而把整个 `generic_converter` 拖进来。

仓库当前没有任何 mcap / ego 相关实现（搜索 `mcap|ego|rosbag|ros2` 仅命中 README 中的无关链接），因此：

- **需要新写的部分**：mcap/h5 的读取、解码、时间对齐、schema 映射。
- **可以直接复用的部分**：DataTrove/Ray 的任务调度、任务隔离、断点续跑、临时数据集管理、最终合并。

AgiBot 与 LIBERO 已经采用该模式，可直接作为参考：

- `agibot2lerobot/agibot_h5.py:494`
- `libero2lerobot/libero_h5.py:143`

## 2. 现有代码能力评估

### 2.1 generic_converter 已具备的能力

`generic_converter/pipeline.py` 使用 DataTrove 作为执行层，支持两种执行器：

| 执行器 | 说明 | 代码位置 |
|---|---|---|
| `local` | `LocalPipelineExecutor`，多进程并发；`workers=-1` 时按 CPU 核数自动计算 | `generic_converter/pipeline.py:98-112` |
| `ray` | `RayPipelineExecutor`，可单机多核或多节点分布式 | `generic_converter/pipeline.py:113-128` |

设计模型：

- `ConversionTask` 表示一个独立可转换单元（`generic_converter/utils.py:11-19`）。
- 每个 task 在 `SaveLeRobotDataset.run()` 中按 `rank` 独立处理，并写入自己的临时 LeRobotDataset（`generic_converter/pipeline.py:22-63`）。
- 全部结束后由 `aggregate_tasks()` 合并为最终数据集（`generic_converter/pipeline.py:220-247`）。

能力与 ego 场景的对应关系：

| generic_converter 能力 | ego 场景 |
|---|---|
| `ConversionTask` = 独立转换单元 | 一个 mcap 文件 / 一个 h5 episode |
| 每个 task 独立建临时数据集 | 文件之间无写冲突 |
| `local` 多进程 / `ray` 多机 | 多节点并行解码 mcap/h5 |
| `--workers`、`--cpus-per-task`、`--tasks-per-job` | 控制解码并发与 CPU 配额 |
| `resume_dir` + DataTrove 日志 | 中断后跳过已完成文件 |
| `aggregate_tasks()` | 多个临时数据集合成一个 LeRobot 数据集 |
| Ray 环境自动注入 `HDF5_USE_FILE_LOCKING=FALSE`（`pipeline.py:161-170`） | 并行读 h5 必需 |

### 2.2 其它转换器的并发情况（参考）

| 转换器 | 并发/分布式支持 | 说明 |
|---|---|---|
| AgiBot | ✅ Ray（默认） | 支持 `--episodes-per-task` 细粒度切分（`agibot_h5.py:312-330`） |
| LIBERO | ✅ local / ray | 一个 HDF5 文件一个 task（`libero_h5.py:64-90`） |
| RoboMIND | ✅ Ray（直接 `ray.remote`） | 每个 task 独立输出，**没有最终合并**（`robomind_h5.py:352-367`） |
| OpenX | ❌ | 单进程、单数据集顺序处理（`openx_rlds.py:195-228`） |
| RoboCasa | ❌ | 单进程顺序遍历（`robocasa_h5.py:50+`） |
| LeRobot → RLDS | ✅ Beam | 方向相反，不适用于本任务（`lerobot2rlds.py:197-226`） |
| v20 → v21 版本转换 | ⚠️ 部分 | 仅 stats 计算使用 `ProcessPoolExecutor`（`convert_stats.py:41-46`） |

## 3. 方案选择：复用 generic_converter vs 基于 Ray 重写

### 推荐：复用 generic_converter

理由：

1. 分布式转换中最繁琐的是调度、隔离、重试/续跑和合并，这些已有实现。
2. mcap/h5 通常天然可按文件或 episode 切分，与 task 模型匹配。
3. ego 数据的特殊性全部集中在 adapter 内：topic 解码、时间同步、坐标系、相机映射、state/action 定义。

### 仅在以下情况才考虑 Ray Data / 自研 Ray 流水线

1. **文件内并行且无法切分**：单个 mcap 长达数小时，无法拆成独立 task。
2. **需要 record/stream 级流水线**：从对象存储流式读取、动态 batching、多模态流 join/shuffle 等。
3. **任务数量极大**：几十万级 task 导致 pipeline 序列化开销成为瓶颈（见 6.2）。

补充：第 1 点通常也有解。`load_tasks()` 允许多个 task 指向同一个文件，用 `metadata` 记录 episode key 或 mcap 的 `[start_time, end_time]` 区间，worker 各自做 indexed read。AgiBot 的 `--episodes-per-task 1` 即此思路（`agibot_h5.py:312-330`）。mcap 本身支持按时间索引，因此通常**不需要**重写执行层。

## 4. 推荐架构

新建 `ego2lerobot/`，一个 adapter 抽象 + 两个 reader：

```python
class EgoAdapter(BaseAdapter):
    dataset_type = "ego"
    fps = 30                      # 统一重采样后的目标 fps
    robot_type = "ego_head"       # 采集设备/机器人名称
    features = {...}              # observation.images.* / observation.state / action

    def __init__(self, src_paths, output_path, ...):
        super().__init__(output_path)
        self.src_paths = src_paths

    def load_tasks(self):
        # h5:   每个文件 = 1 task；或按 episode 拆 task
        # mcap: 每个文件 = 1 task；多 episode 时用 metadata 存时间区间
        ...

    def load_subset(self, task):
        # 必须用生成器逐帧 yield，不要整段读入内存
        reader = read_mcap if task.input_path.suffix == ".mcap" else read_h5
        for frame in reader(task.input_path, task.metadata):
            yield {**frame, "task": task.metadata.get("task", "")}
```

### 运行方式

```bash
# 单机多进程
python ego_h5.py ... --executor local --workers -1 --cpus-per-task 2

# 多机
ray start --head --port=6379           # 节点 1
ray start --address='node1_ip:6379'    # 节点 2
python ego_h5.py ... --executor ray --workers -1 --cpus-per-task 2 --tasks-per-job 1
```

## 5. 实现要点

1. **解码逻辑放在 `load_subset` 内部，adapter 保持可 pickle。**
   DataTrove 会把 pipeline/adapter 序列化到 worker；不要在 `__init__` 中持有打开的文件句柄、ROS node 或连接对象。

2. **覆盖 `create_dataset()`。**
   `BaseAdapter.create_dataset()` 未暴露 `use_videos`、`image_writer_threads`、`image_writer_processes`（`generic_converter/adapter.py:33-45`）。ego 数据为高分辨率多相机，通常必须传入这些参数，否则视频编码会成为单 worker 瓶颈。

3. **先确定时间对齐策略。**
   mcap 各 topic 频率通常不一致，而 LeRobot 要求固定 fps。建议：
   - 先按目标 fps 建立时间轴；
   - 各 topic 做最近邻或线性插值；
   - 多相机按最近时间戳取帧。
   这是 ego 转换中最容易出错的部分。

4. **明确 action 语义。**
   ego 数据往往没有 action。LeRobot 数据集需要 action feature，必须明确使用“下一帧状态 / 未来轨迹 / 关节增量”中的哪一种，或构造空 action。

5. **选择解码依赖。**
   - mcap：`mcap` + `mcap-ros2-support`（有 embedded msg definition 时）或 `rosbags`；
   - h5：`h5py`，逐 episode 流式读取。

6. **异构 schema 拆分 adapter。**
   `features` / `robot_type` 是类属性；不同采集设备或相机布局应拆 adapter 或分多次运行。generic pipeline 只合并同 schema 的 task。

7. **确认 LeRobot 版本。**
   本仓库面向 LeRobot v3.0；需确认安装版本与 `LeRobotDataset.create` / `aggregate_datasets` API 匹配。

## 6. 风险与坑

### 6.1 所有 Ray 节点依赖必须一致

`_build_ray_env_vars()` 只传递 `PYTHONPATH`，不传递 pip 包。`mcap`、`h5py`、`ffmpeg`、解码库必须在每个节点安装。可选方案：

- 统一使用共享 conda/venv；
- 或给 `ray.init` 增加 `runtime_env={"pip": ...}`。

### 6.2 任务列表会被序列化到每个 job

`SaveLeRobotDataset` 持有整个 `tasks` 列表（`pipeline.py:22-26`），DataTrove 按 job 序列化 pipeline。几百到几千个文件没问题；若按 episode 拆成几十万 task，序列化与 driver 内存会成为瓶颈。此时应：

- 改为“task 清单落盘 + worker 按 rank 读取清单”；或
- 退化为每个文件一个 task。

### 6.3 最终 aggregate 是单机串行瓶颈

`aggregate_tasks()` 在 driver 上合并 parquet/index/stats，只有视频合并使用了线程池（`pipeline.py:367-406`）。多相机高码率 ego 视频会使该步骤很慢，并且要求：

- 所有 worker 的 `_temp` 目录位于共享文件系统；
- driver 有足够磁盘（临时 + 最终约 2 倍）和 IO；
- 数据量极大时考虑保留 per-task 数据集，再分批树状 aggregate，而不是一次性合并。

### 6.4 临时目录写满

`cleanup_temp=True` 仅在成功后删除；失败或磁盘不足时会保留中间结果，需要监控。

### 6.5 任务粒度

并行粒度是 task，无法自动拆分单个 task。若单文件过大：

- h5：在 `load_tasks()` 中按 episode 生成 task；
- mcap：按 `[start_time, end_time]` 生成 task，每个 worker 做 indexed read。

## 7. 递归发现与整体聚合

### 7.1 是否支持

支持。但**文件发现不属于 generic_converter**，它是 adapter 的职责：`load_tasks()` 返回的 `ConversionTask` 列表，就是最终参与转换与聚合的全集。

现有实现中：

- `robocasa2lerobot/robocasa_h5.py:50` 已使用 `glob("**/*.hdf5")` 递归；
- LIBERO（`libero_h5.py:64`）和 AgiBot（`agibot_h5.py:312`）是非递归的，只扫描固定层级；
- 新 adapter 需要自己实现“多 root + 递归 + 去重 + 过滤”。

### 7.2 递归发现实现

推荐用 `os.walk` 而不是 `Path.rglob`，因为需要剪枝、跳过软链接环、去重和确定性排序：

```python
def discover(roots, include, exclude, follow_symlinks=False):
    seen = set()
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
            dirnames[:] = sorted(
                d for d in dirnames
                if not is_excluded(Path(dirpath, d).relative_to(root), exclude)
            )
            for name in sorted(filenames):
                path = Path(dirpath) / name
                rel = path.relative_to(root)
                if not match(rel, include, exclude):
                    continue
                st = path.stat()
                key = (st.st_dev, st.st_ino)
                if key in seen:
                    continue
                seen.add(key)
                yield root, path, rel
```

要点：

- `include` / `exclude` 从配置读取，例如 `**/*.mcap`、`**/*.h5`、`**/*.hdf5`、`**/.cache/**`。
- `follow_symlinks=False`，避免软链接环和重复文件。
- 按相对路径排序，保证 episode 顺序稳定、可复现。
- 用 `(st_dev, st_ino)` 去重（多个 root 相互重叠时只处理一次）。
- 过滤 0 字节文件、隐藏目录、`.partial` / `.tmp`、未完成采集目录。
- 目录名或路径长度不能直接作为唯一 ID（见 7.4）。

### 7.3 manifest：大规模发现与续跑

文件达到万级时，不要把“发现”和“转换”耦合在一次运行里：

```bash
python -m ego2lerobot discover --config configs/ego.yaml --emit manifest.jsonl
python -m ego2lerobot convert  --config configs/ego.yaml --manifest manifest.jsonl
```

manifest 每条记录至少包含：`relative_path`、`size`、`mtime_ns`、`format`、`group`、可选内容哈希。好处：

- 发现结果可审计、可 diff、可人工剔除坏文件；
- runner 可按 manifest 分片（`--shard-index/--shard-count`）；
- 避免每次启动重复扫描共享文件系统；
- 便于实现“已完成清单”，支持断点续跑。

### 7.4 临时路径与 repo id 必须唯一

不同子目录下可能有同名文件（`a/cam.h5`、`b/cam.h5`）。临时输出目录和 `local_repo_id` 必须由**相对路径 + 哈希**生成，不能只用 stem：

```python
digest = hashlib.sha1(str(rel).encode()).hexdigest()[:10]
task.output_path = temp_root / group / f"{path.stem}-{digest}"
task.local_repo_id = f"ego/{group}/{path.stem}-{digest}"
```

同时注意路径长度上限：截断 stem，保留哈希。

### 7.5 聚合语义

- **默认（所有子路径同一个 schema）**：全部 task 聚合成一个 LeRobot 数据集，即“递归读取 + 整体聚合”。
- **按组聚合（`group_by`）**：按子目录、深度或正则分组，每组分别聚合为一个数据集。不同组 schema 不同时必须这样做。
- **schema 不一致**：LeRobot 数据集的 feature schema 是固定的，不能直接合并。必须用第 8 节的映射配置把各子路径统一成**并集 schema**；缺失字段用 `on_missing: fill` 填默认值，或明确 `skip_episode`。
- **失败任务不能静默丢失**：`generic_converter` 的 `aggregate_tasks()` 只聚合已存在的临时目录（`pipeline.py:230`），失败 task 会被静默跳过。自研 runner 应记录失败清单，并在聚合前要求 `--allow-partial` 才继续。
- **不要静默覆盖**：`aggregate_tasks()` 会无条件 `shutil.rmtree(output_dir)`（`pipeline.py:228`）。自研实现默认应拒绝覆盖非空目录，除非显式 `--overwrite`。
- **原子性**：每个 task 先写 `xxx.partial/`，`finalize()` 成功后再写 `_SUCCESS` 标记并原子 rename。重启时跳过带 `_SUCCESS` 的 task，避免把崩溃留下的半成品聚合进去。
- **聚合仍是单机瓶颈**：见 6.3。十万级 episode 时应分批树状合并，或保留 per-group 数据集。

## 8. 配置驱动的字段映射

### 8.1 原则

- 不再把 schema 硬编码为 adapter 的类属性；`features` 从 YAML 生成。
- **target** 直接决定 LeRobot feature 名；**source** 按数据格式声明（h5 路径 / mcap topic + 字段）。
- 同一 target 可同时给出 `h5` 和 `mcap` 两种来源，reader 按实际文件格式选择。
- 颜色转换、类型转换、单位换算、切片、重采样都实现为可组合的 `transform`。
- 配置必须可校验、可由样本自动生成草稿、可随数据集一起存档。

### 8.2 示例配置

```yaml
dataset:
  name: ego
  fps: 30
  robot_type: ego_head
  output: /data/lerobot/ego

discovery:
  roots:
    - /data/ego/2026-01
    - /data/ego/2026-02
  include: ["**/*.mcap", "**/*.h5", "**/*.hdf5"]
  exclude: ["**/.cache/**", "**/tmp/**", "**/*.partial/**"]
  follow_symlinks: false
  group_by: relative_dir          # none | relative_dir | depth:N | regex:...

source:
  format: auto                    # auto | mcap | h5
  timestamp:
    h5: "timestamp"               # dataset 路径，或 attrs:xxx
    mcap: {topic: "/clock", field: "stamp", unit: "ns"}
  task:
    h5: "attrs/instruction"
    mcap: {topic: "/task", field: "data"}
    default: "ego"

fields:
  observation.images.head:
    source:
      h5: "obs/head_rgb"
      mcap: {topic: "/camera/head/image", field: "data"}
    dtype: video
    shape: [480, 640, 3]
    names: [height, width, rgb]
    transforms:
      - {op: bgr_to_rgb}

  observation.images.left_wrist:
    source:
      h5: "obs/left_wrist_rgb"
      mcap: {topic: "/camera/left_wrist/image", field: "data"}
    dtype: video
    shape: [480, 640, 3]
    names: [height, width, rgb]

  observation.state:
    source:
      h5: "obs/state"
      mcap: {topic: "/proprio", field: "qpos"}
    dtype: float32
    shape: [7]
    names: {motors: [j1, j2, j3, j4, j5, j6, j7]}
    resample: linear

  action:
    source:
      h5: "actions"
      mcap: {topic: "/cmd", field: "action"}
    dtype: float32
    shape: [7]
    names: {motors: [j1, j2, j3, j4, j5, j6, j7]}
    resample: hold

resample:
  default: nearest
  tolerance_ms: 20

on_missing: error                 # error | skip_frame | skip_episode | fill
```

### 8.3 Resolver 设计

- **H5Resolver**：解析 `group/subgroup/dataset`，支持 `attrs:path`；支持标量、1-D 向量、`(T, H, W, C)` 图像序列；时间戳可来自 dataset 或 attrs；episode 划分由配置决定。
- **McapResolver**：解析 `{topic, field}`，用 `mcap` + `mcap-ros2-support` 或 `rosbags` 解码；`field` 支持点路径和索引，如 `pose.position.x`、`data[0]`；时间戳可取 `log_time` 或 `header.stamp`。
- **统一字段路径解析**：实现一个小工具 `resolve_path(obj, "a.b[0].c")`，同时适用于 ROS message 与嵌套 dict。
- **FieldMapper**：输入时间戳和原始消息，输出 `{target: value}`，再补上 `task`，形成 `add_frame` 可接受的 frame dict。
- **features 生成**：由 `fields` 配置直接构造 LeRobot `features`，不依赖 adapter 类属性。

### 8.4 Transform 注册表

```yaml
transforms:
  - {op: bgr_to_rgb}
  - {op: cast, dtype: float32}
  - {op: scale, factor: 0.001}       # mm -> m
  - {op: slice, start: 0, end: 7}
  - {op: quat_to_euler, order: wxyz}
  - {op: custom, fn: "my_pkg.transforms:normalize_quat"}
```

内置常用的少量 op，同时允许 `module:function` 形式接入自己的实现。这样新增字段只需改配置，不必改转换主流程。

### 8.5 自动扫描生成草稿配置

数据源字段很多时，手写配置容易漏。提供 `scan` 命令：

```bash
python -m ego2lerobot scan \
  --input /data/ego/2026-01/task_001/episode_0001.mcap \
  --emit-config configs/ego_draft.yaml
```

- mcap：列出所有 channel/topic、schema 名称、可解析字段、消息频率、时间戳范围；
- h5：列出 dataset 树、shape、dtype、attrs；
- 输出一份带 `# TODO` 的草稿 YAML，把可自动推断的字段（图像、state、action、timestamp）预填，剩下的由人工确认。

### 8.6 校验与 dry-run

- `--validate-config`：检查 YAML 结构、target 唯一性、dtype/shape/names 合法性、sample 文件上 source 是否可解析。
- `--dry-run --limit 2`：真实转换 1~2 个 task 到临时目录，再用 `LeRobotDataset` 加载，校验 episode 数、帧数、features、stats。
- 源字段不存在时，报错并列出该文件可用字段，避免默默生成空数据。
- 转换完成后把最终配置副本和哈希写入输出目录（如 `_conversion/config.yaml` + `config.sha256`），保证数据集可复现。
- `on_missing` 策略必须显式：`error`（默认）、`skip_frame`、`skip_episode`、`fill`（需提供默认值）。

### 8.7 多组 / 异构字段

不同子路径字段不一致时，支持“基础配置 + 分组覆盖”：

```yaml
groups:
  "2026-01":
    fields:
      observation.images.head:
        source: {h5: "obs/head_rgb_v2"}
```

合并规则：基础 `fields` 与组内 `fields` 深合并，组内优先。

若最终要聚合为一个数据集，所有组必须收敛到**同一并集 schema**：

- 某组缺失的字段 → `fill` 默认值（如全零数组）；
- 或者选择不聚合，按组输出多个数据集。

### 8.8 已实现的映射文件格式

当前实现位于 `ego2lerobot/mapping/config.py`，一个 YAML 同时描述 reader 和字段映射（完整示例见 `ego2lerobot/configs/ego_h5.yaml`、`ego_mcap.yaml`）：

```yaml
version: 1
repo_id: ego/ego_demo
output_root: /data/lerobot/ego_demo
fps: 30
robot_type: ego

task:                       # 可写常量字符串，或 {source: attrs.instruction, default: ...}
  source: attrs.instruction
  default: ego manipulation

features:
  observation.state:
    dtype: float32
    shape: [7]
    source: proprio.qpos    # 单个路径或候选列表，先找到先用
  action:
    dtype: float32
    shape: [7]
    source: [proprio.cmd, action]
    transforms:             # 可组合；顺序执行
      - op: scale
        factor: 0.001

reader:
  type: h5                  # h5 | mcap
  time_axis: 0
  arrays:
    proprio.qpos: /observations/qpos
    proprio.cmd: /action
  attrs: true               # 文件 attrs 暴露为 record["attrs"][...]

writer:
  use_videos: false
```

要点：

- `source` 解析规则：**字面 key 优先**（支持 reader 输出的 `"proprio.qpos"` 扁平 key），否则按 `.` 逐层解析 dict/list；数字段可索引 list。
- 缺失值：有 `default` 用默认值；`required: false` 返回 `None`；否则抛 `MappingError` 并列出尝试过的路径。
- transform 注册表：`identity / astype / reshape / scale / clip / slice / index / transpose / squeeze / expand_dims / to_bool / to_string / as_image`。
- `FeatureSpec.to_lerobot()` 的 `shape` 必须输出 tuple：LeRobot 0.5.1 用 `value.shape != feature["shape"]` 校验，list 会导致每一帧都校验失败。
- 配置在 worker 内按 `(path, mtime)` 缓存；修改配置后新启动的 task 会重新加载。

## 9. 弱依赖复用策略

### 9.1 generic_converter 的实际耦合点

| 耦合点 | 位置 | 风险 |
|---|---|---|
| 模块顶层 `from datatrove...` | `pipeline.py:7-8` | 不装 datatrove 就无法 import |
| 模块顶层 `from lerobot...` | `pipeline.py:9-10` | LeRobot 版本必须匹配 |
| monkeypatch LeRobot 私有聚合函数 | `pipeline.py:248-270` | 依赖 `aggregate_videos`、`pd.read_parquet`、`to_parquet_one_row_group_per_episode`、`update_meta_data` 等内部实现，LeRobot 升级易碎 |
| 无条件删除输出目录 | `pipeline.py:228` | 误删风险 |
| 失败任务被静默跳过 | `pipeline.py:230` | 数据静默丢失 |

因此：**不要直接 `import generic_converter.pipeline`**。它适合作为参考实现，不适合作为弱依赖的库。

### 9.2 建议的复用边界

**可以复制/复用的部分（很小、稳定）：**

- `generic_converter/utils.py:11-19` 的 `ConversionTask` 数据结构；
- `generic_converter/adapter.py` 的 `BaseAdapter` 契约（约 62 行）；
- CLI 参数语义：`--executor local|ray`、`--workers`、`--cpus-per-task`、`--tasks-per-job`。

**自己实现的部分：**

- 递归发现 + manifest（第 7 节）；
- 配置解析与字段映射（第 8 节）；
- runner：`local` 用 `concurrent.futures.ProcessPoolExecutor`；`ray` 用**延迟 import**，未安装时给出清晰报错；
- 聚合封装：优先调用 LeRobot 公共 API；只有确实需要兼容性修复时，才把 `_recompute_aggregated_index_stats`、`_normalize_array_values`、`_aggregate_videos_by_key_parallel` 等函数**复制**到自己的 `aggregate.py`，并固定 LeRobot 版本。

**不建议引入 datatrove。** 自研 runner 只需一百多行，就能覆盖 task 调度、并发、续跑；引入 datatrove 会重新把依赖变重，并失去对任务状态和失败处理的控制权。

### 9.3 依赖分层

| 组件 | 必需性 | 说明 |
|---|---|---|
| `lerobot` | 必需 | 数据格式本体；固定并测试版本 |
| `pyyaml` | 必需 | 映射配置 |
| `h5py` | 按数据源 | 仅 h5 需要；延迟 import |
| `mcap` + `mcap-ros2-support` 或 `rosbags` | 按数据源 | 仅 mcap 需要；延迟 import |
| `ray` | 可选 | 仅 `--executor ray` 需要 |
| `datatrove` | 不需要 | 自研 runner 后不再依赖 |
| any4lerobot 源码 | 不 import | 仅参考/复制少量 MIT 代码 |

建议在 `pyproject.toml` 中用 optional extras 表达：

```toml
[project.optional-dependencies]
h5 = ["h5py"]
mcap = ["mcap", "mcap-ros2-support"]
ray = ["ray"]
all = ["h5py", "mcap", "mcap-ros2-support", "ray"]
```

再加一个“最小依赖导入测试”：只安装 `lerobot + pyyaml + h5py`（不装 ray / datatrove），运行 `--executor local --debug`，确保没有任何顶层强依赖。

复制 any4lerobot 代码时保留 MIT 许可证和版权声明。

## 10. Ray 分布式处理：需要新增的功能

### 10.1 总体结构

新增一个可插拔执行层，发现 / 映射 / 写数据集的业务代码保持不变：

```text
cli.py
  └─ runner.py                 # 统一入口、进度、退出语义
       ├─ local_executor.py    # ProcessPoolExecutor
       └─ ray_executor.py      # ray.remote + ray.wait + 限流
            ├─ preflight.py    # 每节点环境 / 共享盘 / 样本可读检查
            └─ state.py        # manifest、_SUCCESS、failed、断点续跑
```

统一接口：

```python
class Executor(Protocol):
    def map(
        self,
        worker_fn: Callable[[ConversionTask, dict], TaskResult],
        tasks: list[ConversionTask],
        config: dict,
    ) -> list[TaskResult]: ...
```

`worker_fn` 必须是**模块级函数**（可序列化），输入一个 `ConversionTask` 和配置，返回一个**小体积** `TaskResult`：

```python
@dataclass
class TaskResult:
    task_id: str
    status: Literal["ok", "skipped", "failed"]
    episodes: int
    frames: int
    elapsed_s: float
    output_path: str
    error: str | None
```

绝对不要从 Ray task 返回图像、视频或大数组，否则会打爆 object store。

### 10.2 必须新增的功能清单

| 功能 | 说明 | 主要位置 |
|---|---|---|
| Ray 初始化 / 集群连接 | `--ray-address local\|auto\|host:port\|ray://...`，支持 `RAY_ADDRESS`；`ray.init(ignore_reinit_error=True)` | `ray_executor.py` |
| 代码分发 | 共享文件系统 + `PYTHONPATH`，或 `runtime_env.working_dir` 上传；用 `excludes` 排除数据/输出/日志 | `ray_executor.py` |
| 资源声明 | `num_cpus`（`--cpus-per-task`）、可选 `num_gpus`、`memory`（调度提示） | `ray_executor.py` |
| 并发限流 | 每 task 一个 Ray task，但只保持 `workers × 2` 个 in-flight，禁止一次提交全部 task | `ray_executor.py` |
| 任务批处理 | `--tasks-per-job N`：一个 Ray task 内顺序处理 N 个 task，摊薄 import/启动开销 | `ray_executor.py` |
| 配置共享 | `config_ref = ray.put(config)`，传给所有 task，避免每个 task 重复序列化大配置 | `ray_executor.py` |
| 失败隔离 | worker 内捕获可预期数据错误，返回 `failed`，不影响其它 task | worker |
| 自动重试 | Ray `max_retries` + `retry_exceptions`，配合幂等 `_SUCCESS` 标记 | `ray_executor.py` + worker |
| 断点续跑 | 启动前跳过带 `_SUCCESS` 的 task；失败写 `failed.jsonl` | `state.py` |
| 原子提交 | 写 `xxx.partial/` → `finalize()` → 写 `_SUCCESS` → rename | worker |
| 进度 / 日志 | `ray.wait` 驱动进度条；每 task 日志落共享盘；结束打印 ok/skipped/failed 汇总 | `runner.py` |
| 聚合顺序确定 | 按排序后的 task 顺序构造 aggregate roots，**不是**按完成顺序 | `aggregate.py` |
| 聚合前校验 | schema/fps/robot_type 一致；输出目录非空需 `--overwrite`；失败未处理需 `--allow-partial` | `aggregate.py` |
| 集群预检 | 每个节点验证 import、ffmpeg、共享盘可写、样本可读、版本一致 | `preflight.py` |
| 中断处理 | Ctrl-C 时 `ray.cancel` 未完成任务、写状态、必要时 `ray.shutdown()` | `runner.py` |
| 可选：数据本地性 | 数据不在共享盘时，按节点资源 / `NodeAffinitySchedulingStrategy` 调度 | `ray_executor.py` |

### 10.3 主要执行流程

```python
import ray

@ray.remote
def convert_one(task: dict, config: dict) -> TaskResult:
    if success_marker_exists(task):
        return TaskResult.skipped(task)
    clean_partial(task)                       # 清理上次崩溃残留
    try:
        run_reader_and_writer(task, config)   # 流式读 → 映射 → 写临时数据集
        mark_success(task)                    # finalize 后写 _SUCCESS
        return TaskResult.ok(task)
    except TransientError as e:               # OSError、临时 IO、超时
        raise                                 # 交给 Ray 重试
    except Exception as e:                    # 数据损坏、schema 不匹配
        mark_failed(task, e)
        return TaskResult.failed(task, e)


def run_ray(tasks, config):
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    ray.init(
        address=resolve_address(config),      # None=本地, "auto"=现有集群
        runtime_env=build_runtime_env(config),
        ignore_reinit_error=True,
    )
    preflight_all_nodes(config)               # 失败则直接退出

    convert = ray.remote(convert_one).options(
        num_cpus=config.cpus_per_task,
        num_gpus=config.gpus_per_task,
        max_retries=config.max_retries,
        retry_exceptions=True,
    )
    config_ref = ray.put(config)
    window = resolve_workers(config) * 2

    pending, results = [], []
    for batch in batched(tasks, config.tasks_per_job):
        pending.append(convert.remote(batch, config_ref))
        if len(pending) >= window:
            ready, pending = ray.wait(pending, num_returns=1)
            results.extend(collect(ready))    # 小结果，可安全 ray.get
    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        results.extend(collect(ready))

    return results                            # 由 runner 决定聚合 / 退出码
```

关键点：

- `ray.put(config)` 后把 ref 作为参数传入，Ray 会按引用传递；worker 拿到的是配置值，不会每个 task 重新序列化。
- `tasks_per_job > 1` 时，一个 Ray task 处理多个 ConversionTask；如果 worker 崩溃，整批重试，但已写 `_SUCCESS` 的 task 会被跳过，因此仍然幂等。
- 可预期数据错误（坏文件、字段缺失）应返回 `failed`；只有临时性错误才 `raise` 交给 Ray 重试，避免坏数据被反复重试。
- `workers=-1` 时按 `floor(cluster_cpu / cpus_per_task)` 计算并发；同时打印集群节点数、CPU、GPU、内存。

### 10.4 每节点预检（preflight）

多机最常见的失败不是代码，而是环境不一致。启动转换前，在每个 alive 节点上跑一次：

```python
@ray.remote(num_cpus=0)
def preflight(config: dict) -> dict:
    # 1. import 检查：ego2lerobot / lerobot / h5py / mcap / rosbags / yaml
    # 2. 版本检查：python、numpy、pyarrow、lerobot、解码库
    # 3. ffmpeg -version（LeRobot 视频编码依赖）
    # 4. 共享盘写读探针：temp_root/.probe-<uuid>
    # 5. 样本输入可读：config.sample_input
    # 6. 返回 hostname、versions、ok/errors
    ...
```

调度方式：

```python
for node in ray.nodes():
    if not node["Alive"]:
        continue
    refs.append(
        preflight.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node["NodeID"], soft=False
            )
        ).remote(config)
    )
```

检查失败时打印节点 hostname 和具体原因并退出，不要等到跑到一半才发现某个 worker 没装 `mcap`。

### 10.5 配置与 CLI 新增项

```yaml
execution:
  executor: ray
  workers: -1
  cpus_per_task: 2
  gpus_per_task: 0
  tasks_per_job: 1
  max_retries: 1
  ray:
    address: auto                 # local | auto | host:port | ray://...
    code_mode: shared             # shared（共享盘 + PYTHONPATH）| upload（working_dir）
    runtime_env_excludes:
      - ".git"
      - "data"
      - "outputs"
      - "_temp"
      - "logs"
      - "**/*.partial"
      - "__pycache__"
    env_vars:
      HDF5_USE_FILE_LOCKING: "FALSE"
      HF_DATASETS_DISABLE_PROGRESS_BARS: "TRUE"
      RAY_DEDUP_LOGS: "0"

preflight:
  enabled: true
  check_shared_fs: true
  check_ffmpeg: true
  sample_input: /data/ego/sample.mcap

aggregate:
  allow_partial: false
  overwrite: false
  parallel_groups: 1              # >1 时先并行按组聚合，再最终合并
```

```bash
python -m ego2lerobot convert \
  --config configs/ego.yaml \
  --manifest manifest.jsonl \
  --executor ray --ray-address auto \
  --cpus-per-task 2 --workers -1 --tasks-per-job 2 \
  --max-retries 1 --allow-partial --overwrite
```

`--cpus-per-task` 要覆盖“解码线程 + 视频编码线程”，否则单节点会过度订阅，实际并发低于预期。

### 10.6 集群运行方式

与仓库现有 AgiBot / LIBERO 文档一致：

```bash
# 节点 1
ray start --head --port=6379 --dashboard-host=0.0.0.0

# 节点 2..N
ray start --address='node1_ip:6379'

# 任意节点
ray status
python -m ego2lerobot convert ... --executor ray --ray-address auto
```

补充建议：

- 若 `/tmp` 较小，`ray start --temp-dir=/shared/ray_tmp`。
- 代码用 `pip install -e .` 装到所有节点最稳；否则用 `code_mode: shared` + `PYTHONPATH`，或 `code_mode: upload`。
- 数据、`_temp`、输出目录必须位于**所有节点可见的共享文件系统**；否则需要启用数据本地性调度（见 10.7）。
- 从集群外提交时使用 Ray Client：`--ray-address ray://head:10001`（需安装 `ray[client]`）。

### 10.7 可选：数据不在共享盘时的本地性调度

默认假设共享文件系统。如果数据分散在各节点本地盘：

1. 发现阶段按节点执行（`discovery.per_node: true`）；
2. task 携带 `node_name` / `node_id`；
3. 提交时用 `NodeAffinitySchedulingStrategy(node_id, soft=True/False)`。

这属于第二阶段能力；第一版先保证共享盘路径跑通。

### 10.8 明确不要做的反模式

- ❌ 所有 worker 同时写同一个 `LeRobotDataset` —— 每个 task 必须写自己的临时数据集。
- ❌ 每个 task 内部做一次 aggregate —— 聚合只在全部转换完成后做一次。
- ❌ 把整个 `tasks` 列表塞进每个 Ray task —— task 粒度要小，配置用 `ray.put` 共享。
- ❌ 用 `ray.data` 处理整文件转换 —— 除非需要 record/stream 级流水线，否则只会增加复杂度。
- ❌ 按完成顺序聚合 —— 会导致 episode 顺序不可复现。
- ❌ 在 worker 内返回视频帧 / 大数组。
- ❌ 失败后静默聚合 —— 必须 `--allow-partial` 显式确认，并输出失败清单。

### 10.9 当前已实现的代码

执行层放在 `ego2lerobot/common/`（共享基础设施，而不是通用 `utils`）：

- `ego2lerobot/common/types.py`：`ConversionTask`、`TaskResult`、`summarize_results`
- `ego2lerobot/common/state.py`：`_SUCCESS`、`.partial` 原子提交、失败标记、任务日志
- `ego2lerobot/common/discovery.py`：递归发现
  - 多 root、include/exclude glob、`max_depth`、`max_files`
  - `(st_dev, st_ino)` 去重、确定性排序、软链接环保护
  - JSON manifest 读写、`build_output_path`（相对路径 + 短哈希）
- `ego2lerobot/common/ray_executor.py`：`RayExecutor` / `RayExecutorConfig`
  - Ray 延迟导入（`import ego2lerobot.common` 不会加载 Ray，已验证）
  - 有界提交、`ray.put` 配置共享、`tasks_per_job`、自动重试
  - 每节点 preflight（import / 版本 / ffmpeg / 共享盘 / 样本可读）
- `ego2lerobot/mapping/config.py`：YAML 映射（见 8.8）+ transform 注册表
- `ego2lerobot/readers/`：`H5Reader`、`McapReader`（ROS2 解码延迟导入）
- `ego2lerobot/config.py`：`PipelineConfig`（reader + mapping + writer + aggregate）
- `ego2lerobot/convert/worker.py`：真实 worker：读取 → 映射 → 写 `LeRobotDataset` → 原子提交
- `ego2lerobot/convert/run.py`：discovery → task → Ray → summary → `aggregate_datasets`
- `ego2lerobot/cli.py`：`discover` / `convert` / `aggregate` 子命令
- `ego2lerobot/demo/worker.py`、`ray_demo.py`：不依赖 h5/LeRobot 的最小 demo
- `ego2lerobot/demo/h5_demo.py`：h5 → LeRobot 端到端 demo（单机 Ray、失败隔离、续跑、聚合）
- `ego2lerobot/tests/test_core.py`：16 个单元测试（发现 / 映射 / h5 reader / worker / 配置）

已在本环境验证：

- 本地 Ray 全量成功（JSON demo 与 h5 demo 均通过）；
- 重复运行通过 `_SUCCESS` 全部跳过；
- 坏文件只失败对应 task，其余任务正常；
- worker 抛异常时按 `max_retries` 重试，最终返回 failed；
- preflight 能发现样本路径不可读并在启动转换前退出；
- h5 → LeRobot 单 task 数据集可被 `LeRobotDataset` 加载，`aggregate_datasets` 后 3 episode / 30 帧 / 图像字段读取正常；
- `python -m unittest discover -s ego2lerobot/tests`：16 passed。

Demo 命令：

```bash
# 不需要 h5 / LeRobot 的执行层 demo
python -m ego2lerobot.demo.ray_demo --ray-address local --workers 4

# 真实 h5 -> LeRobot demo
python -m ego2lerobot.demo.h5_demo --regenerate --workers 4 --aggregate

# 生产 CLI
python -m ego2lerobot discover -c ego2lerobot/configs/ego_h5.yaml --root /data/ego
python -m ego2lerobot convert  -c ego2lerobot/configs/ego_h5.yaml --root /data/ego \
  --ray-address auto --workers 16 --tasks-per-job 2 --aggregate
python -m ego2lerobot aggregate -c ego2lerobot/configs/ego_h5.yaml --output-root /data/lerobot/ego
```

## 11. 推荐的目录与模块划分

```text
ego2lerobot/
  common/                   # ✅ 已实现：共享执行层
    __init__.py
    types.py                # ConversionTask / TaskResult
    state.py                # partial / _SUCCESS / failed / task log
    discovery.py            # 递归发现 / 去重 / manifest / 唯一输出路径
    ray_executor.py         # RayExecutor: init / 限流 / 批处理 / 重试 / preflight
  mapping/                  # ✅ 已实现：配置驱动字段映射
    __init__.py
    config.py               # MappingConfig / FeatureSpec / transforms
  readers/                  # ✅ 已实现：格式 reader
    __init__.py             # make_reader(type) 工厂
    base.py                 # EpisodeReader 接口
    h5.py                   # HDF5 reader（time_axis / attrs / 局部读取）
    mcap.py                 # MCAP reader（topic 同步 / nearest|hold|exact）
  convert/                  # ✅ 已实现：真实转换
    __init__.py
    worker.py               # process_batch：读取 -> 映射 -> LeRobot -> 原子提交
    run.py                  # discovery -> Ray -> summary -> aggregate_datasets
  configs/                  # ✅ 示例配置
    ego_h5.yaml
    ego_mcap.yaml
  demo/                     # ✅ 可运行 demo
    __init__.py
    worker.py               # 不依赖 h5/LeRobot 的执行层 demo worker
    ray_demo.py             # JSON 数据生成 / 递归发现 / 提交 / 汇总
    h5_demo.py              # h5 -> LeRobot 端到端 demo
    README.md
  tests/                    # ✅ unittest（无需 pytest）
    test_core.py
  cli.py                    # ✅ discover / convert / aggregate
  config.py                 # ✅ PipelineConfig
  __main__.py               # python -m ego2lerobot
```

仍建议后续补充：

- `scripts/ray_cluster.sh`：head / worker 启动示例（含 Slurm 版本）；
- manifest 分片（`--shard-index/--shard-count`）；
- 分批 / 树状聚合，避免十万级 episode 时单机聚合成为瓶颈；
- `--overwrite` 覆盖保护与失败清单的更强约束。

## 12. 落地顺序

1. ✅ `config.py` + `discovery.py`：递归发现、去重、manifest、唯一路径。
2. ✅ 最小 h5 reader + 配置映射 + worker，端到端跑通；`aggregate` 后数据集可被 `LeRobotDataset` 正常加载。
3. ⏳ mcap reader：解码路径与 topic 同步已实现，需在安装 `mcap-ros2-support` 或 `rosbags` 的节点上用真实数据验证（当前环境两者都未安装）。
4. ⏳ 单机压测：确认单文件解码耗时、内存峰值、临时目录占用。
5. ✅ `RayExecutor` 最小闭环：单机 Ray、并发限流、`ray.put(config)`、`_SUCCESS` 幂等、失败返回。
6. ✅ 每节点 preflight；⏳ 多机共享盘小批量验证（需要真实集群）。
7. ⏳ manifest 分片、`--tasks-per-job` 调优、聚合分批、断点续跑与失败重试的规模化验证。

## 13. 一句话总结

**复用 any4lerobot 的“任务模型 + 执行语义”，不复制它的重依赖 pipeline；递归发现与配置映射自己掌控；执行层以 Ray 分布式为主、local 为调试后备，datatrove 不需要；LeRobot 只用公共 API 并固定版本。**
