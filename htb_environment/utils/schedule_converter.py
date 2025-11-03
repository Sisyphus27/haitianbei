# schedule_converter.py
# 从 info.json 选出“最佳”一组 episodes_situation，生成 plan.json，并绘制甘特图与训练曲线
import json
import os
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt

# --------- 公共工具 ---------


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
                                      move_job_id: int = 1):
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
        return(float(t), 0 if int(jid) == int(move_job_id) else 1, int(pid))
    
    episodes_sorted = sorted(episodes, key=_key)
    # per-plane 上一次 End_Site
    last_end: Dict[int, int] = {}
    plan: List[Dict[str, Any]] = []
    for t, jid, end_site, pid, pmin, mmin in episodes_sorted:
        start_site = last_end.get(int(pid), int(end_site))
        row = dict(
            Time=float(t),
            Plane_ID=int(pid),
            Start_Site=int(start_site),
            End_Site=int(end_site),
            Job_ID=int(jid),
            process_time=float(pmin),
            move_time=float(mmin),
        )
        plan.append(row)
        last_end[int(pid)] = int(end_site)

    # 注入设备信息（若 info.json 提供）
    _attach_devices_to_plan(plan, episodes_sorted, devices)

    with open(plan_json_path, "w", encoding="utf-8") as f:
        json.dump({"plan": plan, "selected_group_index": best_idx, "criterion": criterion},
                  f, ensure_ascii=False, indent=4)

    # 4) 绘图
    if also_plot:
        try:
            _plot_gantt(plan, os.path.join(out_dir, "gantt.png"))
            _plot_metrics(info, out_dir, best_idx)
        except Exception as e:
            print(f"[schedule_converter] plotting failed: {e}")

# --------- 甘特图 ---------


def _job_color(job_id: int) -> str:
    """简单的 job 调色盘；可按你的 job_id→code 字典改进"""
    base = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc949',
            '#af7aa1', '#ff9da7', '#9c755f', '#bab0ac']
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
        lst.sort(key=lambda r: r["Time"])

    fig = plt.figure(figsize=(12, max(4, len(by_plane)*0.6)))
    ax = plt.gca()
    yticks, ylabels = [], []
    y = 0
    for pid in sorted(by_plane.keys()):
        seq = by_plane[pid]
        for r in seq:
            t0 = r["Time"]
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
