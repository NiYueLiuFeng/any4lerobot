# Ray 分布式读取/处理 Demo

这个 demo 用最简单的 JSON 数据演示 `ego2lerobot.common` 中的 Ray 执行层：

- 递归发现嵌套目录下的数据；
- 每个文件一个 `ConversionTask`，任务 id 由相对路径 + 哈希生成；
- Ray 并发处理，驱动端只接收小结果；
- `_SUCCESS` 标记实现断点续跑；
- 单文件失败不影响其它文件；
- 确定性顺序汇总，不依赖任务完成顺序。

真实 h5 / mcap 转换只需要替换 `demo/worker.py` 里的 `_process_one`，执行层不用改。

## 目录

```text
ego2lerobot/
  common/
    types.py          # ConversionTask / TaskResult
    state.py          # _SUCCESS、partial、失败标记
    discovery.py      # 递归发现、去重、manifest、唯一输出路径
    ray_executor.py   # Ray 执行层：init、限流、批处理、重试、preflight
  mapping/
    config.py         # YAML 字段映射 + transform 注册表
  readers/
    h5.py             # HDF5 reader
    mcap.py           # MCAP reader（topic 同步）
  convert/
    worker.py         # 读取 -> 映射 -> LeRobotDataset -> 原子提交
    run.py            # discovery -> Ray -> summary -> aggregate
  configs/
    ego_h5.yaml       # 可直接跑的 h5 配置
    ego_mcap.yaml     # mcap 配置模板
  demo/
    worker.py         # 不依赖 h5/LeRobot 的 demo worker
    ray_demo.py       # 执行层 demo（JSON 数据）
    h5_demo.py        # h5 -> LeRobot 端到端 demo
    egoverse_demo.py  # EgoVerse Zarr v3 -> EgoVerseReader demo
    README.md
  cli.py              # discover / convert / aggregate
  config.py           # PipelineConfig
  tests/test_core.py  # 16 个单元测试
```

## 快速开始（单机 Ray）

从仓库根目录执行：

```bash
python -m ego2lerobot.demo.ray_demo \
  --ray-address local \
  --workers 4 \
  --tasks-per-job 1
```

默认会在 `/tmp/ego_ray_demo` 下生成 24 个嵌套 episode：

```text
/tmp/ego_ray_demo/raw/
  batch_00/scene_0/episode_0000.json
  batch_00/scene_1/episode_0001.json
  ...
```

每个文件模拟 50 帧、7 维 state/action，并人为加入 100ms 计算延迟。

预期日志：

```text
[ray] nodes=1 CPUs=... GPUs=0 workers=4 cpus_per_task=1.0
[ray] preflight <host>: ok python=3.12 ffmpeg=n/a
[demo] discovered 24 tasks under /tmp/ego_ray_demo/raw
[ray] 24/24 ok=24 skipped=0 failed=0 in_flight=0
[ray] done tasks=24 ok=24 skipped=0 failed=0 episodes=24 frames=1200 task_time=2.4s wall=1.1s
```

可以对比并发效果：

```bash
# 串行（1 个 worker）
python -m ego2lerobot.demo.ray_demo --clean --workers 1 --compute-ms 200

# 4 个 worker
python -m ego2lerobot.demo.ray_demo --clean --workers 4 --compute-ms 200
```

> Ray 本地集群启动本身有 2~5 秒开销，小数据量下总耗时可能被启动开销掩盖；把 `--num-files` 调大更容易看出差异。

## Demo 1：断点续跑

第一次运行成功后，直接再跑一次：

```bash
python -m ego2lerobot.demo.ray_demo --workers 4
```

每个 task 发现输出目录里已有 `_SUCCESS`，直接返回 `skipped`：

```text
[ray] 24/24 ok=0 skipped=24 failed=0 in_flight=0
```

强制重跑：

```bash
python -m ego2lerobot.demo.ray_demo --clean --workers 4
```

`--clean` 只删输出/日志，不删原始数据。

## Demo 2：失败隔离

生成 2 个损坏的 JSON：

```bash
python -m ego2lerobot.demo.ray_demo \
  --regenerate --corrupt 2 --clean --workers 4
```

预期：

```text
[ray] 24/24 ok=22 skipped=0 failed=2
[demo] 2 task(s) failed:
  - batch_00/scene_0/episode_0000-...: JSONDecodeError: ...
```

- 坏文件会写 `<output>.failed.json`，不会生成 `_SUCCESS`；
- 其它 22 个文件正常完成；
- 默认退出码为 1；加 `--allow-partial` 可忽略失败并返回 0；
- 修复坏文件后重跑，已成功的 22 个会跳过，只重试失败的 2 个。

## Demo 3：多机 Ray

前提：`--data-dir` 位于**所有节点可见的共享文件系统**，且每个节点安装了相同依赖。

```bash
# 1) 在共享盘上生成数据（任选一个节点执行一次）
python -m ego2lerobot.demo.ray_demo \
  --data-dir /shared/ego_ray_demo \
  --generate-only

# 2) 启动 Ray 集群
#    node1:
ray start --head --port=6379
#    node2..N:
ray start --address='node1_ip:6379'

# 3) 在任意节点连接现有集群并转换
python -m ego2lerobot.demo.ray_demo \
  --data-dir /shared/ego_ray_demo \
  --ray-address auto \
  --workers -1 \
  --tasks-per-job 2
```

`--ray-address auto` 要求已经有集群；单机想临时起 Ray 时用 `local`。

多机时 preflight 会在每个节点上检查：

- 能否 `import ego2lerobot`；
- Python 版本是否一致；
- `probe_dir` 是否可写（共享盘验证）；
- 样本输入是否可读；
- 可选：`ffmpeg` 是否存在。

任何一个节点失败都会在开始转换前报错，并打印节点 hostname。

## 如何换成 h5 / mcap

执行层 demo 只改 `demo/worker.py` 即可。仓库里已经有一套真实实现，可以直接用：

### h5 → LeRobot 端到端 demo

```bash
# 生成 8 个嵌套 h5，转换并聚合
python -m ego2lerobot.demo.h5_demo --regenerate --workers 4 --aggregate

# 失败隔离：第 1 个文件写成非法 h5
python -m ego2lerobot.demo.h5_demo --regenerate --corrupt 1 --clean

# 重复运行：已提交的 task 直接 skipped
python -m ego2lerobot.demo.h5_demo --workers 4
```

生成的目录：

```text
/tmp/ego_h5_demo/
  raw/batch_00/scene_0/episode_0000.h5   # qpos / cmd / images + attrs.instruction
  output/.../episode_0000-<hash>/        # 每个 task 一个 LeRobotDataset + _SUCCESS
  aggregated/                            # aggregate_datasets 结果
  _run_summary.json
```

### EgoVerse（Zarr v3）demo

EgoVerse 的子数据集是一个 Zarr v3 group：根 `zarr.json` 带
`embodiment / fps / task_name / features`，每个 feature 一个数组；
图像特征是 `variable_length_bytes`（JPEG）。

```bash
# 默认读取示例子数据集，打印元信息与数组形状
python -m ego2lerobot.demo.egoverse_demo

# 只看前 5 帧、并解码第 0 帧
python -m ego2lerobot.demo.egoverse_demo \
  --sub-dataset 2025-09-20-17-42-51-000000 --max-frames 5 --show-frame 0

# 指定自己的 EgoVerse 根目录 / 子数据集（也可直接传绝对路径）
python -m ego2lerobot.demo.egoverse_demo \
  --root /path/to/egoverse --sub-dataset 2025-09-20-17-42-51-000000 --json
```

把一集子数据集真正转换成 LeRobot 数据集（复用了生产链路的
`PipelineConfig` + `convert/worker.py`，不是另写一份写盘逻辑）：

```bash
# 转换前 5 帧；默认写到 config 的 output_root 下
python -m ego2lerobot.demo.egoverse_demo --to-lerobot --max-frames 5 --overwrite

# 指定输出位置 / repo_id / 帧窗口
python -m ego2lerobot.demo.egoverse_demo --to-lerobot \
  --sub-dataset 2025-09-20-17-42-51-000000 \
  --config ego2lerobot/configs/ego_egoverse.yaml \
  --output-root /tmp/ego_lerobot_demo \
  --repo-id ego/egoverse_demo \
  --stride 1 --max-frames 5 --overwrite
```

产物（默认位置，即 `ego_egoverse.yaml` 的 `output_root`）：

```text
/tmp/ego_egoverse_demo/output/
  2025-09-20-17-42-51-000000-<hash>/
    _SUCCESS                  # 原子提交标记；重跑时任务直接 skipped
    meta/info.json            # fps / features / total_frames
    data/... parquet
    images/...                # writer.use_videos=false 时的 PNG 帧
```

脚本转换完成后会用 `LeRobotDataset(repo_id, root=...)` 读回产物并打印
fps / episodes / frames / features；退出码 0 表示 `ok` 或 `skipped`，
1 表示失败。

- 已有的 `_SUCCESS` 会让重复运行直接 `skipped`；`--overwrite` 才会删除重转；
- `--max-frames` 是帧数，内部换算成 reader 的 `stop = start + max_frames * stride`；
- 只想检查原始数据时不要加 `--to-lerobot`，那条路径不会写任何文件；
- 输出目录不支持 `rename(2)` 时（例如华为 SFS Turbo 的某些导出会返回 `EPERM`），
  `staging_path()` 会直接往最终目录写，`_SUCCESS` 仍是唯一的完成标记，不会多复制一份数据。

核心 API：

```python
from ego2lerobot.demo.egoverse_demo import generate_egoverse_demo

reader = generate_egoverse_demo(
    "2025-09-20-17-42-51-000000",   # 子数据集目录或 root 下的名字
    root="/path/to/egoverse",
    max_frames=100,
)

reader.describe()          # 元信息、feature 形状、warnings
reader.arrays              # 数值 feature -> numpy 数组
reader.decode_image("images.front_1", 0)   # 按需解码单帧 (480, 640, 3) uint8
for frame in reader.iter_frames(decode_images=False):
    ...                    # 每帧一个 dict；图像键为原始 JPEG bytes
```

- 数值数组按 `stride` / `max_frames` 只读需要的帧；
- 图像默认惰性解码，`load_images=True` 才一次性载入 `reader.images`；
- 数组帧数与 `attrs.total_frames` 不一致时记录在 `reader.warnings`，不会静默截断；
- `annotations`（`json` / `annotation_v1`）解析后放在 `reader.annotations`。

### 生产 CLI

```bash
python -m ego2lerobot discover -c ego2lerobot/configs/ego_h5.yaml --root /data/ego
python -m ego2lerobot convert  -c ego2lerobot/configs/ego_h5.yaml --root /data/ego \
  --ray-address auto --workers 16 --tasks-per-job 2 --aggregate
python -m ego2lerobot aggregate -c ego2lerobot/configs/ego_h5.yaml \
  --output-root /data/lerobot/ego
```

### 聚合后端：lerobot vs ray-data

`convert/run.py` 的 `aggregate()` 直接调用 LeRobot 的 `aggregate_datasets`，是单进程顺序合并；
`convert/aggregate_raydata.py` 用 `ray.data` 把同样的合并在**文件级**并行化：

| | lerobot | ray-data |
|---|---|---|
| 调度 | 逐个 source 读→改写→追加 | 每个源文件一个任务并行改写 |
| 索引更新 | `update_data_df` / `update_meta_data` | **同一批函数**，只是每个 source 带自己的 offset |
| 输出文件布局 | 按大小打包（会反复重读已写文件） | 一个源文件对应一个目标文件 |
| 视频特征 | 支持（ffmpeg 顺序拼接） | 不支持，抛 `AggregateNotSupported` |
| 源数据集要求 | 任意 | 每个源恰好 1 个 data 文件 + 1 个 episodes 文件 |

选择后端：

```bash
python -m ego2lerobot aggregate -c ego2lerobot/configs/ego_h5.yaml --engine ray-data --force
```

A/B 基准（内容逐列对比 + 计时）：

```bash
python -m ego2lerobot.demo.aggregate_ab --num-files 64 --frames 40
```

脚本会把两种后端的结果都读回来，比较 `data` 的每一列（含图像字节）、
`meta/episodes` 的索引列和 `info.json` 的计数，一致才返回 0。

字段映射全部在 YAML 里，示例见 `ego2lerobot/configs/ego_h5.yaml`：

```yaml
features:
  observation.state:
    dtype: float32
    shape: [7]
    source: proprio.qpos          # 对应 reader.arrays 的 key
  observation.images.head:
    dtype: image
    shape: [64, 96, 3]
    source: cameras.head
    transforms:
      - op: as_image
        dtype: uint8

reader:
  type: h5
  arrays:
    proprio.qpos: /observations/qpos
    proprio.cmd: /action
    cameras.head: /observations/images/head
```

### mcap

`readers/mcap.py` 已实现 topic 解码与同步，但需要节点安装解码器之一：

```bash
pip install mcap-ros2-support    # ROS 2
pip install rosbags              # ROS 1 + ROS 2
```

配置模板见 `ego2lerobot/configs/ego_mcap.yaml`。当前开发环境未安装两者，因此 mcap 路径只验证了配置校验与错误提示，需要在目标集群上用真实数据跑一遍。

### 直接替换 demo worker

如果不使用上面的 `convert/worker.py`，只需按同样约定修改 `demo/worker.py`：

```python
def _process_one(task, config):
    if is_task_done(task):
        return TaskResult(task.task_id, "skipped", ...)

    prepare_output(task)
    try:
        # 1) 读取：根据 task.input_path 后缀选择 h5 或 mcap reader
        # 2) 映射：按配置把源字段映射成 LeRobot features
        # 3) 写入：root 用 staging_path(task)，由它决定临时目录还是最终目录
        #    LeRobotDataset.create(root=staging_path(task), ...)
        #    dataset.add_frame(...); dataset.save_episode(); dataset.finalize()
        # 4) finalize 后调用 commit_output(task) 提交并写 _SUCCESS
        ...
    except TransientIOError:
        raise          # 交给 Ray 重试（max_retries）
    except Exception as exc:
        write_failure(task, traceback.format_exc())
        return TaskResult(..., status="failed", error=str(exc))
```

注意：

- 每个 task 写自己的临时 LeRobotDataset，绝不能多个 worker 写同一个数据集；
- 流式逐帧读取，不要整段 episode 读进内存；
- 只返回 `TaskResult`，不要返回图像/数组；
- 数据错误返回 `failed`，临时性 IO 错误 `raise` 让 Ray 重试。

## 执行层关键点

| 能力 | 位置 |
|---|---|
| 有界提交（`workers × 2` in-flight） | `common/ray_executor.py` `_run_tasks` |
| 配置共享 `ray.put` | `_run_tasks` |
| `--tasks-per-job` 批处理 | `_batched` |
| `max_retries` + `retry_exceptions` | `ray.remote(...).options(...)` |
| 每节点 preflight | `_run_preflight` / `_preflight_checks` |
| 结果按 task 原始顺序返回 | `_run_tasks` 末尾 |
| 失败隔离/幂等 | `demo/worker.py` + `common/state.py` |
| 原子提交 `_SUCCESS` | `common/state.py` `commit_output` |

## 调参建议

| 参数 | 建议 |
|---|---|
| `--workers` | 单机先用 4；多机设 `-1`，由集群 CPU 自动计算 |
| `--cpus-per-task` | 要覆盖解码线程 + ffmpeg 编码线程，否则会过度订阅 |
| `--tasks-per-job` | 文件多且单文件小（如 h5）时用 2~4，摊薄 Ray 任务开销；单文件很大时用 1 |
| `--max-retries` | 数据错误不应重试；共享盘偶发 IO 错误可用 1~2 |
| `--compute-ms` | 仅 demo 用，模拟真实解码/编码耗时 |

## 注意事项

- 代码必须能在每个 worker 上 import。仓库没有 `pyproject.toml` 时，`RayExecutor` 会自动把仓库根目录和当前工作目录加入 worker 的 `PYTHONPATH`（要求仓库在共享盘上路径一致）；否则用 `convert --working-dir <repo>` 让 Ray 上传代码。
- 数据、临时输出、日志、`probe_dir` 必须位于共享文件系统。
- `--ray-address local` 只适合单机调试；多机必须 `ray start` 后用 `auto` 或具体地址。
- `ray_demo.py` 的“聚合”只是写 `summary.json`；`convert/worker.py` + `convert/run.py` 走真实 `aggregate_datasets`，并按 task 原始顺序构造输入列表。
- 每个 task 的数据集根目录里会多写一个 `_conversion.json`（来源文件、reader、config 路径），用于审计；LeRobot 读取时会忽略它。
- `writer.use_videos: true` 需要可用的 ffmpeg 与编码器；当前示例默认 `false`（PNG 帧），确认集群编码器后再打开。
- 单机 Ray 启动有 2~5 秒固定开销；小数据量下并行收益会被启动时间掩盖。
- 单 task 解码失败会写 `<output>.failed.json`；修复源文件后重跑会跳过已成功的 task。

## 测试

```bash
python -m unittest discover -s ego2lerobot/tests -v
```

覆盖：递归发现 / include-exclude / 去重 / manifest、映射（候选路径、默认值、transform、错误校验）、h5 reader（time_axis）、worker 原子提交与失败标记、示例配置加载。
