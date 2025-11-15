# htb_environment

- pipeline/pipeline.py：知识图谱→先验知识→RL→生成调度计划→更新kg→下一轮..

  1. 创建并连接任务一的 KG 客户端 Dataset_KG

  2. 用 KG 构造先验适配器 T1KGPriorAdapter，并 attach 到环境

  3. rollout -> buffer -> agent.train

  4. 选出本 epoch 最优/最后一组调度，转三元组 -> 调用任务一接口 update_with_triples 回写

  5. 下一 epoch 自动使用更新后的图谱统计
- pipeline/kg_bridge.py：生成先验（plane_prior、site_prior，在environment.py attach_prior()），调度转换为三元组方便更新
- environment.py：仿真环境
- main.py：主流程入口，默认带有--use_task1_kg调用pipeline，若无该参数，只进行调度

- utils/schedule_converter.py：绘图工具类
- MARL/arguments.py：各类参数设置
- MARL/runner.py：调度算法逻辑

> 测试场景：给定初始飞机布设场景下，某飞行日早上07:00开始，单批按2分钟间隔连续着陆12架飞机，共5批，批次间隔60分钟(前批最后一架着陆时刻与后批第一架着陆时刻间隔时长)。

```bash
python main.py --learn False --load_model True --n_agents 60 --batch_mode --batch_size_per_batch 12 --batches_count 5 --intra_gap_min 2 --inter_batch_gap_min 60 --batch_start_time_min 420 --result_name my_multi_batch_run --evaluate_epoch 1
```

## 扰动事件配置

- `--enable_disturbance`：开启扰动事件逻辑，环境会在事件起止时间内自动封锁对应停机位，并对受到影响的飞机进行暂停处理，记录当时的全局状态快照。
- `--disturbance_events`：JSON 字符串或 JSON 文件路径，描述一组扰动事件。每个事件包含 `start`、`end`（分钟）以及 `stands`（如 `[5,6,7]` 或 `"5-10,12"`）。

示例：

```bash
python main.py --learn False --load_model False --n_agents 12 --batch_mode --batch_size_per_batch 12 --batches_count 1 --intra_gap_min 2 --inter_batch_gap_min 60 --batch_start_time_min 420 --evaluate_epoch 1 --result_name disturbance_test --enable_disturbance --disturbance_events '[{"start":450,"end":550,"stands":"5-10"}]'
```

上述命令会在 450~550 分钟内封锁 5~10 号停机位，被迫停止保障的飞机会记录剩余作业进度并在新站位继续未完成的作业，事件结束后自动恢复站位可用性。

## 调度状态输入与快照推理指南

本文档总结如何在任意时间点注入全局状态、继续运行调度推理，并说明 `--snapshot_json` / `snapshot_scheduler.py` 的使用方式。

### 1. 关键状态字段

- **飞机**（`utils/plane.py`）  
  `plane_id`、`status`、`current_site_id`、`position`、`eta_move_end`、`eta_proc_end`、`finished_codes`、`paused_jobs`、`_active_jobs` 等字段直接影响 `get_obs`/`get_state`（`environment.py:548-646`）和调度主循环（`environment.py:1064-1494`）。
- **作业/依赖**（`utils/job.py`、`utils/task.py`）  
  `required_resources`、`pre`、`any_pre`、`mutex` 由 `TaskGraph.enabled`、`pack_parallel` 使用，用于识别可执行作业。
- **站位与占用**（`utils/site.py`、`environment.py`）  
  `stand_current_occupancy`、`last_leave_time_by_stand/runway`、`disturbance_blocked_stands` 控制站位可用性和干涉；`Site.unavailable_windows` 用于封控。
- **设备**（`environment.py:22-424`）  
  `FixedDevice` 记录 `in_use`、`capacity`，`MobileDevice` 记录 `loc_stand`、`busy_until_min`、`locked_by`；`_alloc_resources_for_job` 依据这些状态分配资源。
- **快照**（`environment.py:834-905`）  
  `_capture_global_state` 输出 `{"time", "planes", "stand_occupancy", "blocked_stands", ...}`，是与外部项目互通的统一字段格式。

#### 1.1 必填 / 可选字段说明

- **必填**：
  - `planes`（至少提供 `plane_id`、`status` 和 `current_site_id`）。快照恢复时会根据 `current_site_id` 自动回填站位坐标与占用情况。
  - 扰动重放场景建议提供 `disturbance_events`（含 `start`/`end`/`stands`），这样快照和多批次模式都会复用相同的扰动注入逻辑（强制迁移、封控等）。
- **自动推断**：
  - 若 `finished_codes` 为空，调度器会依据当前 `active_job`/`paused_jobs` 及其前驱依赖补全；与当前作业可并行且未写入的依赖作业默认视为已完成。
  - `position` 缺省或为占位值（如 `[0, 0]`）时，会自动取停机位坐标。
- **可选**：`stand_occupancy`（若提供则作为校验，缺省时会从飞机位置汇总）、`time`、`arrival_plan`、`devices`、`site_unavailable`、`blocked_stands`（额外封控）等增强信息按需提供即可。

### 2. 快照 JSON 示例

```json
{
  "time": 485.0,
  "planes": [
    {
      "plane_id": 0,
      "status": "PROCESSING",
      "current_site_id": 5,
      "active_job": "ZY02",
      "active_remaining": { "ZY02": 1.5 }
    },
    {
      "plane_id": 1,
      "status": "PROCESSING",
      "current_site_id": 6,
      "active_job": "ZY03",
      "active_remaining": { "ZY03": 2.0 }
    },
    {
      "plane_id": 2,
      "status": "PROCESSING",
      "current_site_id": 7,
      "active_job": "ZY04",
      "active_remaining": { "ZY04": 2.8 }
    },
    {
      "plane_id": 3,
      "status": "PROCESSING",
      "current_site_id": 8,
      "active_job": "ZY05",
      "active_remaining": { "ZY05": 2.2 }
    },
    {
      "plane_id": 4,
      "status": "IDLE",
      "current_site_id": 9,
      "paused_jobs": [
        { "code": "ZY06", "remaining": 1.0 }
      ]
    },
    {
      "plane_id": 5,
      "status": "IDLE",
      "current_site_id": 10
    },
    {
      "plane_id": 6,
      "status": "IDLE"
    },
    {
      "plane_id": 7,
      "status": "IDLE"
    }
  ],
  "blocked_stands": [5,6,7,8,9],
  "arrival_plan": { "6": 500.0, "7": 505.0 },
  "devices": {
    "fixed": {
      "FR5": { "in_use": [2], "capacity": 2 }
    },
    "mobile": {
      "MR05": {
        "loc_stand": 7,
        "busy_until_min": 492.0,
        "locked_by": 2,
        "speed_m_s": 3.0
      }
    }
  },
  "disturbance_events": [
    { "id": 0, "start": 500.0, "end": 540.0, "stands": [5,6,7,8,9] }
  ]
}
```

字段说明：

- `time`：全局时间（分钟）。
- `planes`：每架飞机的状态（`active_job` 支持 `ZY04+ZY05` 并行写法；`paused_jobs` 记录剩余作业）。
- `stand_occupancy`：站位是否被占用。
- `blocked_stands`：当前封锁站位。
- `arrival_plan`（可选）：覆盖 `env.arrival_plan`，用于尚未落地的飞机。
- `devices`（可选）：固定/移动设备的占用、容量、位置。
- `site_unavailable`（可选）：额外的封控时间窗。

### 3. 注入步骤

1. `env = ScheduleEnv(args); env.reset(n_agents)` 初始化结构。
2. `snapshot_scheduler.restore_env_from_snapshot(env, snapshot_dict)` 将快照写入环境。
3. 可选：打印 `env.get_state()`、`env.stand_current_occupancy` 校验。
4. 使用自有策略 `policy_fn -> env.step(actions)` 继续推理，或调用 `snapshot_scheduler.infer_schedule_from_snapshot`。

### 4. 快照调度模式

#### 通过 main.py

`main.py` 新增 `--snapshot_json` 参数（文件路径或 JSON 字符串）。示例：

```bash
python main.py --snapshot_json my_snapshot.json --result_name snapshot_run --enable_disturbance --learn False --evaluate_epoch 1 2>&1 | Select-Object -Last 30
```

- 程序会跳过训练/评估，直接调用 `snapshot_scheduler` 继续推理。
- 结果写入 `result/<result_name>/snapshot/snapshot_YYYYmmdd_HHMMSS.json`，其中包含 `time`、`reward`、`episodes_situation`、`devices_situation` 以及扰动信息。
- 其它参数（如 `--enable_disturbance`、`--batch_mode`）仍可设置，用于控制环境行为。

### 5. 调度推理流程回顾

1. 自动落地（`environment.py:1164-1194`）。
2. 解析动作、分配滑行/牵引（`environment.py:1196-1274`）。
3. 按最小剩余作业时间推进时钟（扰动会截断步长，`environment.py:1276-1336`）。
4. 完工释放资源并为 IDLE 飞机重新申请作业或恢复暂停（`environment.py:1336-1494`）。
5. 记录奖励与 `episodes_situation` / `devices_situation`（`environment.py:1496-1520`）。

### 6. snapshot_scheduler 入口

- `restore_env_from_snapshot(env, snapshot)`：依据快照重建 `ScheduleEnv`。
- `greedy_idle_policy(env)`：演示策略，挑选编号最小的可用站位。
- `infer_schedule_from_snapshot(args, snapshot, policy_fn=None, max_steps=None)`：重置环境、注入快照、循环调用 `policy_fn`（默认 `greedy_idle_policy`），返回最后一次 `env.step` 的 `info` 用于生成新的调度计划。

结合 `--snapshot_json` 或自定义策略，即可把任意时间点的状态导入系统，在不重新训练的情况下完成一次完整的后续调度推理。
