# schedule_converter.py
# 从 info.json 选出"最佳"一组 episodes_situation，生成 plan.json，并绘制甘特图与训练曲线
import json
import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
from utils.job import Jobs

# --------- 公共工具 ---------

try:
    _JOB_ID_TO_CODE = {
        job.index_id: job.code for job in Jobs().jobs_object_list
    }
except Exception:
    _JOB_ID_TO_CODE = {}


def _minutes_to_time_str(minutes: float) -> str:
    """将分钟时间转换为 HH:MM:SS 格式

    Args:
        minutes: 从午夜（00:00）算起的分钟数，例如 420 = 7:00 AM

    Returns:
        格式为 "HH:MM:SS" 的时间字符串
    """
    # 确保分钟在合理范围内（0-1440 为一天）
    total_minutes = float(minutes) % 1440
    hours = int(total_minutes // 60)
    mins = int(total_minutes % 60)
    secs = int((total_minutes % 1) * 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d}"


def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _calc_makespan(episodes: List[List[float]]) -> float:
    """episodes: [(time, job_id, site_id, plane_id, proc_min, move_min), ...]
       以 max(time + max(proc, move)) 估算 makespan
    """
    if not episodes:
        return float("inf")
    m = 0.0
    for t, jid, sid, pid, pmin, mmin in episodes:
        m = max(m, float(t) + float(max(pmin, mmin)))
    return m


def _pick_best_group(info: Dict[str, Any]) -> Tuple[List[List[float]], Optional[List[Any]], int, str]:
    """从 info.json 中选择“最佳”的一组 episodes_situation.
    返回: (episodes, devices, idx, criterion_desc)
    优先规则：
      1) 若 schedule_results 多组，按“本组 makespan 最小”选；
      2) 若无法计算，且存在 evaluate_reward/average_reward，则按 reward 最大；
      3) 否则退化：取最后一组（与官方默认一致）。
    """
    sched_groups: List[List[List[float]]] = info.get("schedule_results", [])
    dev_groups: Optional[List[List[Any]]] = info.get("devices_results", None)

    if not sched_groups:
        return [], None, -1, "empty"

    # 1) 尝试按 makespan 最小
    best_i, best_ms = -1, float("inf")
    for i, g in enumerate(sched_groups):
        ms = _calc_makespan(g)
        if ms < best_ms:
            best_ms, best_i = ms, i
    if best_i >= 0 and best_ms < float("inf"):
        devs = dev_groups[best_i] if (
            dev_groups and best_i < len(dev_groups)) else None
        return sched_groups[best_i], devs, best_i, f"min_makespan={best_ms:.2f}"

    # 2) 尝试按 reward 最大
    eval_rewards = info.get("evaluate_reward", []
                            ) or info.get("average_reward", [])
    if eval_rewards:
        max_r, max_i = max((r, i) for i, r in enumerate(eval_rewards))
        if 0 <= max_i < len(sched_groups):
            devs = dev_groups[max_i] if (
                dev_groups and max_i < len(dev_groups)) else None
            return sched_groups[max_i], devs, max_i, f"max_reward={max_r:.2f}"

    # 3) 退化：取最后一组
    devs = dev_groups[-1] if dev_groups and len(
        dev_groups) == len(sched_groups) else None
    return sched_groups[-1], devs, len(sched_groups) - 1, "last_group"


# schedule_converter.py —— _attach_devices_to_plan 调整
def _attach_devices_to_plan(plan, episodes, devices):
    if not devices or len(devices) != len(episodes):
        for row in plan:
            row["FixedDevices"] = []
            row["MobileDevices"] = []
        return
    for i, dev in enumerate(devices):
        fixed_ids, mobile_ids = [], []
        if isinstance(dev, dict):
            # 直接保留字符串 ID
            fixed_ids = [str(x) for x in dev.get("FixedDevices", [])]
            mobile_ids = [str(x) for x in dev.get("MobileDevices", [])]
        plan[i]["FixedDevices"] = fixed_ids
        plan[i]["MobileDevices"] = mobile_ids


# --------- 主转换函数 ---------


def convert_schedule_with_fixed_logic(info_json_path: str,
                                      plan_json_path: str,
                                      n_agent: int,
                                      out_dir: Optional[str] = None,
                                      also_plot: bool = True,
                                      move_job_id: int = 1,
                                      batch_size_per_batch: Optional[int] = None):
    """读取 info.json，选出“最佳”一组，生成 plan.json，并绘图（甘特 + 训练曲线）
    - out_dir 为空时，与 plan_json_path 同目录
    """
    _ensure_dir(plan_json_path)
    out_dir = out_dir or os.path.dirname(plan_json_path) or "."

    # 1) 读 info
    with open(info_json_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    # 2) 选出最佳 episodes
    episodes, devices, best_idx, criterion = _pick_best_group(info)

    # 3) 转成官方规范的 plan.json
    #    统一排序：按 Time, Plane_ID
    def _key(e):
        t, jid, _, pid, _, _ = e
        # 移动作业排在前面（相同时刻同飞机）
        return (float(t), 0 if int(jid) == int(move_job_id) else 1, int(pid))

    episodes_sorted = sorted(episodes, key=_key)
    # per-plane 上一次 End_Site
    last_end: Dict[int, int] = {}
    plan: List[Dict[str, Any]] = []
    runway_usage: List[Dict[str, Any]] = []
    for t, jid, end_site, pid, pmin, mmin in episodes_sorted:
        start_site = last_end.get(int(pid), int(end_site))
        row = dict(
            Time=_minutes_to_time_str(t),  # 用 HH:MM:SS 格式替代分钟数
            Plane_ID=int(pid),
            Start_Site=int(start_site),
            End_Site=int(end_site),
            Job_ID=int(jid),
            process_time=float(pmin),
            move_time=float(mmin),
            _time_minutes=float(t),  # 内部字段用于绘图，不会显示在 JSON 中
        )
        plan.append(row)
        last_end[int(pid)] = int(end_site)
        end_site_int = int(end_site)
        if end_site_int in (29, 30, 31):
            entry = dict(
                Time=row["Time"],
                time_minutes=row["_time_minutes"],
                Plane_ID=row["Plane_ID"],
                Job_ID=row["Job_ID"],
                Job_Code=_JOB_ID_TO_CODE.get(row["Job_ID"]),
                Start_Site=row["Start_Site"],
                End_Site=row["End_Site"],
                process_time=row["process_time"],
                move_time=row["move_time"],
            )
            runway_usage.append(entry)

    # 注入设备信息（若 info.json 提供）
    _attach_devices_to_plan(plan, episodes_sorted, devices)

    # 保存 JSON 前，提取 _time_minutes 字段用于绘图，然后清除
    plan_for_json = []
    for row in plan:
        row_copy = dict(row)
        row_copy.pop('_time_minutes', None)  # 移除内部字段
        plan_for_json.append(row_copy)

    output_payload = {
        "plan": plan_for_json,
        "selected_group_index": best_idx,
        "criterion": criterion,
        "runway_usage": runway_usage,
    }
    with open(plan_json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=4)

    # 4) 绘图（使用原始 plan，包含 _time_minutes）
    if also_plot:
        try:
            _plot_gantt(plan, os.path.join(out_dir, "gantt.png"))
            # 额外输出：停机位使用情况甘特图（按飞机/批次着色，并显示图例）
            try:
                _plot_stand_usage(plan, os.path.join(
                    out_dir, "gantt_stand_usage.png"), batch_size_per_batch=batch_size_per_batch)
            except Exception:
                pass
            # 不再输出训练/评估曲线图（evaluate_span / evaluate_metrics），按用户要求去掉
        except Exception as e:
            print(f"[schedule_converter] plotting failed: {e}")

# --------- 甘特图 ---------


def _job_color(job_id: int) -> str:
    """简单的 job 调色盘；可按你的 job_id→code 字典改进"""
    base = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc949',
            '#af7aa1', '#ff9da7', '#9c755f', '#bab0ac', '#b6992d', '#a17c6b',
            '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
            '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4', '#ff6b6b']
    return base[job_id % len(base)]


def _plot_gantt(plan: List[Dict[str, Any]], png_path: str):
    """甘特图：按飞机分行；加工用色块，移动用浅色边框"""
    if not plan:
        return
    _ensure_dir(png_path)
    # 聚合到每个 plane 的时间片
    by_plane: Dict[int, List[Dict[str, Any]]] = {}
    for row in plan:
        by_plane.setdefault(int(row["Plane_ID"]), []).append(row)
    for lst in by_plane.values():
        # 按 _time_minutes（用于绘图的内部分钟字段）或 Time（兼容性）排序
        lst.sort(key=lambda r: r.get("_time_minutes", r.get("Time", 0)))

    fig = plt.figure(figsize=(12, max(4, len(by_plane)*0.6)))
    ax = plt.gca()
    yticks, ylabels = [], []
    y = 0
    for pid in sorted(by_plane.keys()):
        seq = by_plane[pid]
        for r in seq:
            # 使用 _time_minutes（如果有）作为绘图的时间基准
            t0 = r.get("_time_minutes", None)
            if t0 is None:
                # 兼容性：如果没有 _time_minutes，尝试从 Time 解析或跳过
                continue
            # 移动段（若 move_time > 0）画为浅色边框
            if r["move_time"] > 0:
                ax.broken_barh([(t0, r["move_time"])], (y-0.35, 0.25),
                               facecolors='none', edgecolors='#888', linewidth=1.2)
                t0 += r["move_time"]
            # 加工段（若 process_time > 0）用实心色块
            if r["process_time"] > 0:
                ax.broken_barh([(t0, r["process_time"])],
                               (y-0.2, 0.4), facecolors=_job_color(r["Job_ID"]))
        yticks.append(y)
        ylabels.append(f"Plane {pid}")
        y += 1

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Time (min)")
    ax.set_title("Schedule Gantt (best group)")
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    # 构造图例：每个 Job_ID 对应一个颜色，添加移动段说明
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    job_ids = sorted(set(r["Job_ID"] for r in plan))
    legend_handles = []
    for jid in job_ids:
        col = _job_color(jid)
        patch = mpatches.Patch(color=col, label=f"Job {jid}")
        legend_handles.append(patch)
    move_line = mlines.Line2D([], [], color='#888',
                              linewidth=1.2, label='Move (travel)')
    legend_handles.append(move_line)
    if legend_handles:
        ncol = max(1, len(legend_handles))
        ax.legend(
            handles=legend_handles,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),
            ncol=ncol,
            frameon=True,
            framealpha=0.95,
        )
        bottom_margin = 0.22
    else:
        bottom_margin = 0.08

    plt.tight_layout(rect=[0, bottom_margin, 1, 1])
    plt.savefig(png_path, dpi=160)
    plt.close(fig)

# --------- 训练曲线 ---------


def _moving_avg(arr: List[float], k: int) -> List[float]:
    if not arr or k <= 1:
        return arr
    out = []
    s = 0.0
    for i, v in enumerate(arr):
        s += v
        if i >= k:
            s -= arr[i-k]
        out.append(s / min(k, i+1))
    return out


def _plot_makespan_progress(info: dict, out_dir: str):
    train_ms = info.get("train_makespan") or []
    eval_ms = info.get("evaluate_makespan") or []
    if not train_ms:
        return
    epi_per_epoch = int(info.get("episodes_per_epoch") or 0) or None
    eval_cycle = int(info.get("evaluate_cycle") or 0) or None
    eval_epoch = int(info.get("evaluate_epoch") or 1)

    x = np.arange(1, len(train_ms)+1)
    y = np.asarray(train_ms, dtype=float)

    # 运行最优与平滑
    running_min = np.minimum.accumulate(y)
    k = max(5, len(y)//50)
    smooth = np.convolve(y, np.ones(k)/k, mode="same") if k > 1 else y

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x, y, alpha=0.25, label="Train makespan (per episode)")
    ax.plot(x, smooth, lw=2, label="Moving avg")
    ax.plot(x, running_min, lw=2, ls="--", label="Best-so-far (running min)")

    # 标记每次 evaluate 的位置，并叠加当次评估的最优 makespan
    if epi_per_epoch and eval_cycle:
        eval_marks = []
        # 每 eval_cycle 个 epoch 评估一次，落在该 epoch 的结尾
        for k_epoch in range(eval_cycle, (len(y)//epi_per_epoch)+1, eval_cycle):
            eval_ep = k_epoch * epi_per_epoch
            if eval_ep <= len(y):
                eval_marks.append(eval_ep)
        for ep in eval_marks:
            ax.axvline(ep, color="gray", ls=":", lw=1, alpha=0.5)

        if eval_ms:
            xs = np.array(eval_marks[:len(eval_ms)], dtype=int)
            ax.scatter(xs, eval_ms[:len(xs)], zorder=5, marker="o",
                       label="Eval best makespan", s=25)

    ax.set_title(
        "Makespan progress (training trial-and-error → improved schedules)")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Makespan (min)")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "makespan_progress.png"), dpi=160)
    plt.close(fig)


def _plot_stand_usage(plan: List[Dict[str, Any]], png_path: str, batch_size_per_batch: Optional[int] = None):
    """
    绘制停机位使用情况甘特图：按停机位分行显示占用时段（仅展示 End_Site 为停机位的加工段）。

    需求：
    - 颜色只按“基础飞机编号”区分：若设置 batch_size_per_batch=N，则使用 N 种颜色，
      每架飞机的颜色由 (Plane_ID % N) 决定。
      例如 N=12 时，plane0 与 plane12 使用相同颜色。
    - 图例中只展示基础飞机编号 0 ~ (N-1) 与颜色的对应关系，不展示所有全局 plane_id。
    - 纵轴固定显示所有停机位 0~31（包含 0、29、30、31），即使某些停机位未被使用也显示出来。
    - 横轴适当拉长（figsize 宽度增大）。

    Args:
        plan: 由 plan.json 生成的调度序列（内部包含 _time_minutes 字段）。
        png_path: 输出图片路径。
        batch_size_per_batch: 每批次飞机数量，用于确定基础飞机编号数量；
                              若为 12，则图例只展示 Plane 0~11，颜色按 plane_id % 12 复用。
    """
    if not plan:
        return
    _ensure_dir(png_path)

    # -------------------------
    # 1. 聚合到每个停机位（仅统计 0~31 号站位）
    # -------------------------
    STAND_MIN_ID = 0
    STAND_MAX_ID = 31
    by_stand: Dict[int, List[Dict[str, Any]]] = {}

    for row in plan:
        sid = int(row.get("End_Site", -1))
        # 只关注 0~31 号站位；其他 site_id 在本图中忽略
        if sid < STAND_MIN_ID or sid > STAND_MAX_ID:
            continue
        by_stand.setdefault(sid, []).append(row)

    # 每个停机位内按时间排序
    for lst in by_stand.values():
        lst.sort(key=lambda r: r.get("_time_minutes", r.get("Time", 0)))

    # -------------------------
    # 2. 确定基础飞机编号数量 & 颜色映射
    # -------------------------
    # 所有出现的全局 plane_id（可能很多批次）
    plane_ids_all = sorted(set(int(r["Plane_ID"]) for r in plan))

    # 基础飞机编号数量：优先使用 batch_size_per_batch；否则根据数据推一个上限
    if batch_size_per_batch is not None and batch_size_per_batch > 0:
        base_plane_num = int(batch_size_per_batch)
    else:
        # 未显式给出时：若总飞机数 <= 12，则用实际数量；否则固定用 12 种颜色
        base_plane_num = len(plane_ids_all) if plane_ids_all else 1
        if base_plane_num > 12:
            base_plane_num = 12

    # 基础飞机编号：0 ~ base_plane_num-1
    base_plane_ids = list(range(base_plane_num))

    # 调色盘：对基础飞机编号分配颜色
    cmap = plt.get_cmap('tab20')
    base_colors = {pid: cmap(i % cmap.N)
                   for i, pid in enumerate(base_plane_ids)}

    # 图例中的“飞机数”就是基础编号数（而不是全局 plane_id 数量）
    n_planes_for_legend = base_plane_num

    # --- 修改点 开始 ---
    # 原有的多行计算逻辑 (if/elif/else) 被替换
    # 强制图例为单排显示：
    ncol = n_planes_for_legend  # 列数 = 图例项总数
    legend_rows = 1             # 行数 = 1
    # --- 修改点 结束 ---

    # -------------------------
    # 3. 图尺寸：纵轴按 0~31 号站位，横轴拉长
    # -------------------------
    stand_ids = list(range(STAND_MIN_ID, STAND_MAX_ID + 1))
    # 每个停机位占 0.4，高度 = 停机位数 * 0.4；图例按行数增加高度
    gantt_height = max(4, len(stand_ids) * 0.4)
    legend_height = legend_rows * 0.25 + 0.5  # 图例区高度
    total_height = gantt_height + legend_height

    # 横向拉长：宽度 18（比默认 12/14 更长）
    fig = plt.figure(figsize=(18, total_height))
    ax = plt.gca()

    # -------------------------
    # 4. 绘制每个停机位的甘特条
    # -------------------------
    yticks, ylabels = [], []
    y = 0
    for sid in stand_ids:
        seq = by_stand.get(sid, [])
        for r in seq:
            t0 = r.get("_time_minutes", None)
            if t0 is None:
                continue
            if r.get("process_time", 0) > 0:
                pid = int(r["Plane_ID"])
                # 颜色只由“基础飞机编号”决定：plane_id % base_plane_num
                base_pid = pid % base_plane_num
                ax.broken_barh(
                    [(t0, r["process_time"])],
                    (y - 0.2, 0.4),
                    facecolors=base_colors.get(base_pid, "#999999"),
                )
        yticks.append(y)
        ylabels.append(f"Stand {sid}")
        y += 1

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Time (min)")
    ax.set_title("Stand Usage Gantt")
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)

    # -------------------------
    # 5. 图例：只展示基础飞机编号 0~(base_plane_num-1)
    # -------------------------
    import matplotlib.patches as mpatches
    legend_handles = []
    for pid in base_plane_ids:
        lbl = f"Plane {pid}"  # 只展示 0~(base_plane_num-1)
        patch = mpatches.Patch(color=base_colors[pid], label=lbl)
        legend_handles.append(patch)

    # 图例放在下方，不遮住甘特图
    # (注意：bbox_to_anchor 的 Y 值计算依赖于 legend_rows，
    # 因为 legend_rows 现固定为 1，Y 值也会固定)
    ax.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.05 - 0.04 * legend_rows),
        ncol=ncol,
        frameon=True,
        framealpha=0.95,
    )

    # 为图例预留空间
    # (注意：bottom 的计算也依赖于 legend_rows，
    # 因为 legend_rows 现固定为 1，bottom 值也会固定)
    plt.subplots_adjust(bottom=0.1 + 0.04 * legend_rows)

    plt.savefig(png_path, dpi=160, bbox_inches='tight')
    plt.close(fig)



