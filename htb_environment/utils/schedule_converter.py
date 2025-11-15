# schedule_converter.py
# 从 info.json 选出"最佳"一组 episodes_situation，生成 plan.json，并绘制甘特图与训练曲线
import json
import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt

# --------- 公共工具 ---------


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

    # 注入设备信息（若 info.json 提供）
    _attach_devices_to_plan(plan, episodes_sorted, devices)

    # 保存 JSON 前，提取 _time_minutes 字段用于绘图，然后清除
    plan_for_json = []
    for row in plan:
        row_copy = dict(row)
        row_copy.pop('_time_minutes', None)  # 移除内部字段
        plan_for_json.append(row_copy)

    with open(plan_json_path, "w", encoding="utf-8") as f:
        json.dump({"plan": plan_for_json, "selected_group_index": best_idx, "criterion": criterion},
                  f, ensure_ascii=False, indent=4)

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
    ax.legend(handles=legend_handles, loc='upper right',
              bbox_to_anchor=(1.02, 1.0))

    plt.tight_layout()
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
    """绘制停机位使用情况甘特图：按停机位分行显示占用时段（仅展示 End_Site 为停机位的加工段）。
    使用每个飞机一种颜色，并在图例中标注 Plane / Batch 信息（若提供 batch_size_per_batch）。
    图例放在图表下方并分多行显示，不遮盖甘特图。"""
    if not plan:
        return
    _ensure_dir(png_path)
    # 聚合到每个停机位
    by_stand: Dict[int, List[Dict[str, Any]]] = {}
    for row in plan:
        sid = int(row.get("End_Site", -1))
        # 仅考虑站位（假定站位 id <= 28 且 >0）
        if sid <= 0:
            continue
        by_stand.setdefault(sid, []).append(row)
    for lst in by_stand.values():
        lst.sort(key=lambda r: r.get("_time_minutes", r.get("Time", 0)))

    # 颜色分配：按 Plane_ID 分配 distinct colors
    plane_ids = sorted(set(r["Plane_ID"] for r in plan))
    cmap = plt.get_cmap('tab20')
    colors = {pid: cmap(i % cmap.N) for i, pid in enumerate(plane_ids)}

    # 计算合适的图例列数和行数（根据飞机数量动态调整）
    n_planes = len(plane_ids)
    if n_planes <= 10:
        ncol = 5
    elif n_planes <= 20:
        ncol = 5
    else:
        ncol = 6

    # 计算图例需要的行数
    legend_rows = (n_planes + ncol - 1) // ncol

    # 根据停机位数和图例大小动态调整图表高度
    # 每个停机位占 0.4，图例每行占 0.25
    gantt_height = max(4, len(by_stand) * 0.4)
    legend_height = legend_rows * 0.25 + 0.5  # 加 0.5 的上下边距
    total_height = gantt_height + legend_height

    fig = plt.figure(figsize=(14, total_height))
    ax = plt.gca()
    yticks, ylabels = [], []
    y = 0
    for sid in sorted(by_stand.keys()):
        seq = by_stand[sid]
        for r in seq:
            # 使用 _time_minutes 作为绘图时间
            t0 = r.get("_time_minutes", None)
            if t0 is None:
                continue
            # 停机位上的加工段
            if r["process_time"] > 0:
                pid = int(r["Plane_ID"])
                ax.broken_barh([(t0, r["process_time"])],
                               (y-0.2, 0.4), facecolors=colors[pid])
        yticks.append(y)
        ylabels.append(f"Stand {sid}")
        y += 1

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Time (min)")
    ax.set_title("Stand Usage Gantt")
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)

    # 构造图例：每个 Plane 对应一个颜色，若提供 batch_size_per_batch，标注批次
    import matplotlib.patches as mpatches
    legend_handles = []
    for pid in plane_ids:
        batch_label = ''
        if batch_size_per_batch and batch_size_per_batch > 0:
            batch_idx = int(pid) // int(batch_size_per_batch)
            batch_label = f" (Batch {batch_idx})"
        lbl = f"Plane {pid}{batch_label}"
        patch = mpatches.Patch(color=colors[pid], label=lbl)
        legend_handles.append(patch)

    # 将图例放在图表下方，不遮盖甘特图
    # loc='upper center' + bbox_to_anchor=(0.5, -0.1) 将图例放在轴下方
    ax.legend(handles=legend_handles, loc='upper center',
              bbox_to_anchor=(0.5, -0.05 - 0.04 * legend_rows),
              ncol=ncol, frameon=True, framealpha=0.95)

    # 使用 subplots_adjust 为图例预留空间
    plt.subplots_adjust(bottom=0.1 + 0.04 * legend_rows)

    plt.savefig(png_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def _plot_metrics(info: Dict[str, Any], out_dir: str, best_idx: int):
    _ensure_dir(os.path.join(out_dir, "dummy"))
    # 1) 训练奖励（每 episode）
    train_r = info.get("train_reward", [])
    if train_r:
        fig = plt.figure(figsize=(10, 4))
        ax = plt.gca()
        ax.plot(train_r, label="train_reward", linewidth=1.0)
        ax.plot(_moving_avg(train_r, 50),
                label="moving_avg(50)", linewidth=1.5)
        ax.set_title("Training Reward per Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "train_reward.png"), dpi=160)
        plt.close(fig)

    # 2) 损失（每次优化步）
    loss = info.get("loss", [])
    if loss:
        fig = plt.figure(figsize=(10, 4))
        ax = plt.gca()
        ax.plot(loss, label="loss", linewidth=1.0)
        ax.set_title("Training Loss")
        ax.set_xlabel("Update Step")
        ax.set_ylabel("Loss")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "loss.png"), dpi=160)
        plt.close(fig)

    # 3) 评估工期（每次 evaluate）
    eval_ms = info.get("evaluate_makespan", []) or info.get(
        "average_makespan", [])
    if eval_ms:
        fig = plt.figure(figsize=(10, 4))
        ax = plt.gca()
        ax.plot(eval_ms, label="evaluate_makespan", linewidth=1.0)
        ax.set_title("Evaluate Makespan")
        ax.set_xlabel("Evaluate Round")
        ax.set_ylabel("Makespan (min)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "evaluate_makespan.png"), dpi=160)
        plt.close(fig)

    # 4) 胜率/平均奖励
    win = info.get("win_rates", [])
    avg_r = info.get("average_reward", [])
    if win or avg_r:
        fig = plt.figure(figsize=(10, 4))
        ax = plt.gca()
        if avg_r:
            ax.plot(avg_r, label="average_reward", linewidth=1.0)
        if win:
            ax2 = ax.twinx()
            ax2.plot(win, label="win_rate", linestyle='--',
                     linewidth=1.0, color='#e15759')
            ax2.set_ylabel("Win Rate")
        ax.set_title("Evaluation Metrics")
        ax.set_xlabel("Evaluate Round")
        ax.set_ylabel("Reward")
        lines, labels = ax.get_legend_handles_labels()
        if win:
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines += lines2
            labels += labels2
        plt.legend(lines, labels, loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "evaluation_metrics.png"), dpi=160)
        plt.close(fig)
