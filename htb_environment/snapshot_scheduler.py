"""快照注入与单轮调度推理辅助模块。

- snapshot['time']：当前时间（分钟）
- snapshot['planes']：飞机状态列表
- snapshot['stand_occupancy']：站位占用
- snapshot['blocked_stands']：封锁站位
- snapshot['devices']：固定/移动设备状态（可选）

示例：

    from argparse import Namespace
    args = Namespace(n_agents=12, batch_mode=False, arrival_gap_min=2, ...)
    info = infer_schedule_from_snapshot(args, snapshot_payload)
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from environment import ScheduleEnv
from utils.job import Job

PolicyFn = Callable[[ScheduleEnv], List[int]]


def _site_index_by_id(env: ScheduleEnv, site_id: Optional[int]) -> Optional[int]:
    if site_id is None:
        return None
    for idx, site in enumerate(env.sites):
        if site.site_id == site_id:
            return idx
    return None


def _job_from_code(env: ScheduleEnv, code: Optional[str]) -> Optional[Job]:
    if not code:
        return None
    try:
        return env.jobs_obj.get_job(code)
    except Exception:
        return None


def _build_paused_entries(env: ScheduleEnv, raw_entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for entry in raw_entries or []:
        code = entry.get("code")
        job = _job_from_code(env, code)
        if job is None:
            continue
        result.append({
            "job": job,
            "code": code,
            "remaining": float(entry.get("remaining", 0.0))
        })
    return result


def _apply_device_snapshot(env: ScheduleEnv, device_state: Optional[Dict[str, Any]]) -> None:
    if not device_state:
        return
    fixed_index = {d.dev_id: d for d in env.fixed_devices}
    for dev_id, payload in (device_state.get("fixed") or {}).items():
        dev = fixed_index.get(dev_id)
        if dev is None:
            continue
        if "capacity" in payload:
            dev.capacity = int(payload["capacity"])
        in_use = payload.get("in_use")
        if in_use is not None:
            dev.in_use = set(int(pid) for pid in in_use)

    mobile_index = {d.dev_id: d for d in env.mobile_devices}
    for dev_id, payload in (device_state.get("mobile") or {}).items():
        dev = mobile_index.get(dev_id)
        if dev is None:
            continue
        if "loc_stand" in payload:
            dev.loc_stand = int(payload["loc_stand"])
        if "busy_until_min" in payload:
            dev.busy_until_min = float(payload["busy_until_min"])
        if "locked_by" in payload:
            locked = payload["locked_by"]
            dev.locked_by = None if locked is None else int(locked)
        if "speed_m_s" in payload:
            dev.speed_m_s = float(payload["speed_m_s"])


def restore_env_from_snapshot(env: ScheduleEnv, snapshot: Dict[str, Any]) -> None:
    """
    根据快照恢复 ScheduleEnv 内部状态（需先执行 env.reset）。
    """
    env.current_time = float(snapshot.get("time", env.current_time))
    # Stand occupancy
    stand_state = snapshot.get("stand_occupancy", {})
    env.stand_current_occupancy = {
        sid: None for sid in range(1, 29)
    }
    for sid_str, plane_id in stand_state.items():
        sid = int(sid_str)
        env.stand_current_occupancy[sid] = None if plane_id is None else int(plane_id)
    env.disturbance_blocked_stands = set(int(s) for s in snapshot.get("blocked_stands", []))

    # Optional site unavailable windows
    for sid_str, windows in (snapshot.get("site_unavailable") or {}).items():
        sid = int(sid_str)
        for window in windows or []:
            if len(window) != 2:
                continue
            start, end = float(window[0]), float(window[1])
            env.sites_obj.mark_unavailable(sid, start, end)

    # Arrival plan override
    for pid_str, arrive_min in (snapshot.get("arrival_plan") or {}).items():
        pid = int(pid_str)
        env.arrival_plan[pid] = (float(arrive_min), env.landing_runway_site_for_pos)

    # Planes
    plane_states = snapshot.get("planes", [])
    for plane_state in plane_states:
        pid = int(plane_state["plane_id"])
        if pid >= len(env.planes):
            continue
        plane = env.planes[pid]
        plane.finished_codes = set(plane_state.get("finished_codes", []))
        plane.status = plane_state.get("status", "IDLE")
        site_id = plane_state.get("current_site_id")
        idx = _site_index_by_id(env, site_id)
        plane.current_site_id = idx

        if "position" in plane_state and plane_state["position"] is not None:
            plane.position = np.array(plane_state["position"], dtype=float)
        elif idx is not None:
            plane.position = env.sites[idx].absolute_position.copy()

        plane.fuel_percent = float(plane_state.get("fuel_percent", plane.fuel_percent))
        plane.eta_move_end = float(plane_state.get("eta_move_end", 0.0))
        plane.eta_proc_end = float(plane_state.get("eta_proc_end", 0.0))
        plane.move_code = plane_state.get("move_code")
        plane.paused_jobs = _build_paused_entries(env, plane_state.get("paused_jobs", []))

        active_label = plane_state.get("active_job")
        active_remaining = {
            code: float(val) for code, val in (plane_state.get("active_remaining") or {}).items()
        }
        if active_label:
            active_codes = [c for c in str(active_label).split("+") if c]
        else:
            active_codes = []

        if active_codes and plane.status == "PROCESSING":
            jobs = [job for code in active_codes if (job := _job_from_code(env, code)) is not None]
            plane._active_jobs = jobs if jobs else None
            plane.current_job_code = "+".join(active_codes)
            plane.active_job_progress = {code: active_remaining.get(code, plane.eta_proc_end) for code in active_codes}
            if plane.active_job_progress:
                plane.eta_proc_end = max(plane.active_job_progress.values())
            plane._job_total_durations = copy.deepcopy(plane.active_job_progress)
            plane._active_event_indices = [(code, -1) for code in active_codes]
        else:
            plane._active_jobs = None
            plane.current_job_code = None
            plane.active_job_progress = {}
            plane._job_total_durations = {}
            plane._active_event_indices = []

        plane._last_handles = []
        plane._move_handles = None
        env.arrival_plan[pid] = (env.current_time, env.landing_runway_site_for_pos)

    # Devices (optional)
    _apply_device_snapshot(env, snapshot.get("devices"))


def greedy_idle_policy(env: ScheduleEnv) -> List[int]:
    """
    演示策略：选择编号最小的可用站位，否则返回 WAIT/BUSY/DONE。
    """
    actions: List[int] = []
    for pid in range(len(env.planes)):
        avail = env.get_avail_agent_actions(pid)
        candidates = [idx for idx, flag in enumerate(avail) if flag == 1]
        if not candidates:
            actions.append(len(env.sites))  # WAIT fallback
            continue
        chosen = min(candidates)
        if chosen < len(env.sites):
            env.has_chosen_action(chosen, pid)
        actions.append(chosen)
    return actions


def infer_schedule_from_snapshot(
    args: Any,
    snapshot: Dict[str, Any],
    policy_fn: Optional[PolicyFn] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """
    重置环境、注入快照并运行单轮调度，返回最后一次 env.step 的 info。
    """
    env = ScheduleEnv(args)
    env.reset(args.n_agents)
    restore_env_from_snapshot(env, snapshot)

    policy = policy_fn or greedy_idle_policy
    final_info: Dict[str, Any] = {"episodes_situation": [], "devices_situation": [], "time": env.current_time}
    max_iters = max_steps or env.episode_limit

    for _ in range(max_iters):
        actions = policy(env)
        reward, terminated, info = env.step(actions)
        final_info = {
            "reward": reward,
            **info,
        }
        if terminated:
            break
    return final_info


__all__ = [
    "restore_env_from_snapshot",
    "infer_schedule_from_snapshot",
    "greedy_idle_policy",
]
