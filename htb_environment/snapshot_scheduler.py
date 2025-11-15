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
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from environment import ScheduleEnv
from MARL.runner import Runner
from MARL.common.arguments import get_mixer_args
from utils.job import Job
from utils.knowledgeGraph_test import KGPrior

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


def _normalize_finished_codes(raw_entries: Optional[Sequence[Any]]) -> Set[str]:
    codes: Set[str] = set()
    for entry in raw_entries or []:
        if isinstance(entry, str):
            tokens = entry.replace(";", ",").split(",")
            codes.update(token.strip()
                         for token in tokens if token and token.strip())
        elif entry is not None:
            codes.add(str(entry))
    return codes


def _dependency_closure(env: ScheduleEnv, seeds: Iterable[str]) -> Set[str]:
    closure: Set[str] = set()
    stack: List[str] = [code for code in seeds if code]
    while stack:
        code = stack.pop()
        job = _job_from_code(env, code)
        if job is None:
            continue
        for pre in job.predecessors:
            if pre not in closure:
                closure.add(pre)
                stack.append(pre)
    return closure


def _infer_finished_codes(
        env: ScheduleEnv,
        plane_state: Dict[str, Any],
        explicit: Set[str],
        active_codes: Sequence[str],
        paused_codes: Sequence[str]) -> Set[str]:
    finished = set(explicit)
    if finished:
        finished.update(_dependency_closure(env, list(finished)))

    status = str(plane_state.get("status", "") or "").upper()
    if status == "DONE":
        finished.update(env.jobs_obj.code2id().keys())
        return finished

    hints: List[str] = list(active_codes) + list(paused_codes)
    if hints:
        finished.update(_dependency_closure(env, hints))

    prereq_reference = set(finished)
    if prereq_reference:
        paused_set = set(paused_codes)
        active_set = set(active_codes)
        for job in env.jobs_obj.jobs_object_list:
            if job.code in prereq_reference:
                continue
            if job.code in active_set or job.code in paused_set:
                continue
            if not job.predecessors:
                continue
            if set(job.predecessors).issubset(prereq_reference):
                finished.add(job.code)
    return finished


def summarize_plane_completion(env: ScheduleEnv) -> Dict[str, Any]:
    """
    给出当前环境下各飞机的完成情况、未完作业等信息，帮助诊断快照是否覆盖完备。
    """
    summary: Dict[str, Any] = {
        "total_planes": len(env.planes),
        "completed": [],
        "unfinished": []
    }
    for plane in env.planes:
        finished = set(getattr(plane, "finished_codes", set()) or [])
        done = env.task_obj.graph.all_finished(finished)
        entry = {
            "plane_id": plane.plane_id,
            "status": plane.status,
            "current_site_id": None if plane.current_site_id is None or plane.current_site_id >= len(env.sites)
            else env.sites[plane.current_site_id].site_id,
            "finished_codes": sorted(finished)
        }
        remaining = [job.code for job in env.jobs_obj.jobs_object_list
                     if job.code not in finished]
        if done:
            summary["completed"].append(entry)
        else:
            entry["remaining_jobs"] = remaining
            ready = env.task_obj.graph.enabled(
                finished, getattr(plane, "ongoing_mutex", set()))
            entry["next_candidates"] = [job.code for job in ready]
            summary["unfinished"].append(entry)
    return summary


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
            loc_stand = payload["loc_stand"]
            # 处理 None 值：如果为 None，跳过设置（保持设备当前状态）
            if loc_stand is not None:
                dev.loc_stand = int(loc_stand)
        if "busy_until_min" in payload:
            dev.busy_until_min = float(payload["busy_until_min"])
        if "locked_by" in payload:
            locked = payload["locked_by"]
            dev.locked_by = None if locked is None else int(locked)
        if "speed_m_s" in payload:
            dev.speed_m_s = float(payload["speed_m_s"])


def _restore_disturbance_events(env: ScheduleEnv, snapshot: Dict[str, Any]) -> None:
    raw_events = snapshot.get("disturbance_events")
    if not raw_events:
        env.disturbance_events = []
        return
    try:
        iterable = list(raw_events)
    except Exception:
        env.disturbance_events = []
        return
    restored: List[Dict[str, Any]] = []
    for idx, evt in enumerate(iterable):
        try:
            start = float(evt.get("start"))
            end = float(evt.get("end"))
        except Exception:
            continue
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            continue
        stands_payload = evt.get("stands") or evt.get("sites") or evt.get("site_ids")
        stands: List[int] = []
        for token in stands_payload or []:
            try:
                sid = int(token)
            except Exception:
                continue
            if 1 <= sid <= 31:
                stands.append(sid)
        if not stands:
            continue
        restored.append({
            "id": evt.get("id", idx),
            "start": start,
            "end": end,
            "stands": stands,
            "started": False,
            "completed": False,
            "snapshot": evt.get("snapshot"),
            "affected_planes": list(evt.get("affected_planes", [])),
            "meta": evt.get("meta", evt)
        })
    restored.sort(key=lambda e: e["start"])
    env.disturbance_events = restored


def _reapply_disturbance_state(env: ScheduleEnv, blocked_fallback: Iterable[int]) -> None:
    fallback = set(int(s) for s in blocked_fallback or [])
    if not env.enable_disturbance or not getattr(env, "disturbance_events", []):
        env.disturbance_blocked_stands = fallback
        return
    env.disturbance_blocked_stands = set()
    env.disturbance_forced_planes.clear()
    env.disturbance_forced_from.clear()
    env.disturbance_history = list(getattr(env, "disturbance_history", []))
    env._process_disturbance_timeline(initial=True)
    env.disturbance_blocked_stands.update(fallback)


def restore_env_from_snapshot(env: ScheduleEnv, snapshot: Dict[str, Any]) -> None:
    """
    根据快照恢复 ScheduleEnv 内部状态（需先执行 env.reset）。
    如果快照包含历史 episodes_situation，也会一并恢复。
    """
    env.current_time = float(snapshot.get("time", env.current_time))
    # Stand occupancy
    stand_state = snapshot.get("stand_occupancy") or {}
    env.stand_current_occupancy = {
        sid: None for sid in range(1, 29)
    }
    occupied_by_plane: Dict[int, int] = {}
    for sid_str, plane_id in stand_state.items():
        try:
            sid = int(sid_str)
        except Exception:
            continue
        if sid not in env.stand_current_occupancy:
            env.stand_current_occupancy[sid] = None
        occupant = None if plane_id is None else int(plane_id)
        env.stand_current_occupancy[sid] = occupant
        if occupant is not None:
            occupied_by_plane[occupant] = sid
    env.disturbance_history = list(snapshot.get("disturbance_history", []))
    _restore_disturbance_events(env, snapshot)

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
        env.arrival_plan[pid] = (
            float(arrive_min), env.landing_runway_site_for_pos)

    # 恢复历史 episodes_situation（如果存在）
    hist_episodes = snapshot.get("episodes_situation", [])
    if hist_episodes:
        for evt in hist_episodes:
            # evt 格式: (time, jid, sid, pid, proc_min, move_min)
            if isinstance(evt, (list, tuple)) and len(evt) >= 6:
                env.episodes_situation.append(tuple(evt))

    # 恢复历史设备情况（如果存在）
    hist_devices = snapshot.get("devices_situation", [])
    if hist_devices:
        for dev_entry in hist_devices:
            if isinstance(dev_entry, dict):
                env.episode_devices.append(dev_entry)

    # Planes
    plane_states = snapshot.get("planes", [])
    for plane_state in plane_states:
        pid = int(plane_state["plane_id"])
        if pid >= len(env.planes):
            continue
        plane = env.planes[pid]
        plane.status = plane_state.get("status", "IDLE")
        site_id_raw = plane_state.get("current_site_id")
        if site_id_raw is None:
            site_id_raw = occupied_by_plane.get(pid)
        try:
            site_id = None if site_id_raw is None else int(site_id_raw)
        except (TypeError, ValueError):
            site_id = None
        idx = _site_index_by_id(env, site_id)
        plane.current_site_id = idx
        site_obj = env.sites[idx] if idx is not None and 0 <= idx < len(
            env.sites) else None

        use_site_position = False
        if "position" in plane_state:
            pos_payload = plane_state.get("position")
            if pos_payload is not None:
                try:
                    pos_arr = np.array(pos_payload, dtype=float)
                except Exception:
                    use_site_position = idx is not None
                else:
                    if idx is not None and np.allclose(pos_arr, 0.0):
                        use_site_position = True
                    else:
                        plane.position = pos_arr
            else:
                use_site_position = idx is not None
        elif idx is not None:
            use_site_position = True
        if use_site_position and idx is not None:
            plane.position = env.sites[idx].absolute_position.copy()

        plane.fuel_percent = float(plane_state.get(
            "fuel_percent", plane.fuel_percent))
        plane.eta_move_end = float(plane_state.get("eta_move_end", 0.0))
        plane.eta_proc_end = float(plane_state.get("eta_proc_end", 0.0))
        plane.move_code = plane_state.get("move_code")
        paused_payload = plane_state.get("paused_jobs", [])
        plane.paused_jobs = _build_paused_entries(
            env, paused_payload)
        paused_codes: List[str] = []
        for entry in paused_payload or []:
            if isinstance(entry, dict):
                code = entry.get("code")
            else:
                code = entry
            if code:
                paused_codes.append(str(code))

        active_label = plane_state.get("active_job")
        active_remaining = {
            code: float(val) for code, val in (plane_state.get("active_remaining") or {}).items()
        }
        if active_label:
            active_codes = [c for c in str(active_label).split("+") if c]
        else:
            active_codes = []

        explicit_finished = _normalize_finished_codes(
            plane_state.get("finished_codes"))
        plane.finished_codes = _infer_finished_codes(
            env, plane_state, explicit_finished, active_codes, paused_codes)

        if active_codes and plane.status == "PROCESSING":
            jobs = [job for code in active_codes if (
                job := _job_from_code(env, code)) is not None]
            plane._active_jobs = jobs if jobs else None
            plane.current_job_code = "+".join(active_codes)
            plane.active_job_progress = {code: active_remaining.get(
                code, plane.eta_proc_end) for code in active_codes}
            if plane.active_job_progress:
                plane.eta_proc_end = max(plane.active_job_progress.values())
            plane._job_total_durations = copy.deepcopy(
                plane.active_job_progress)
            plane._active_event_indices = [(code, -1) for code in active_codes]
        else:
            plane._active_jobs = None
            plane.current_job_code = None
            plane.active_job_progress = {}
            plane._job_total_durations = {}
            plane._active_event_indices = []

        plane._last_handles = []
        plane._move_handles = None

        # If the snapshot places the plane on any physical site (including a
        # runway) or indicates it has started non-landing work, ensure the
        # landing job ZY_Z is recorded as finished so the env will not try to
        # auto-trigger a brand new landing sequence again.
        inferred_grounded = site_obj is not None
        if not inferred_grounded and plane.finished_codes:
            inferred_grounded = any(code and code != "ZY_Z"
                                    for code in plane.finished_codes)
        if not inferred_grounded and active_codes:
            inferred_grounded = any(code != "ZY_Z" for code in active_codes)
        if inferred_grounded and "ZY_Z" not in plane.finished_codes:
            plane.finished_codes.add("ZY_Z")

        # If snapshot provides explicit arrival time for this plane, keep it.
        # Otherwise, mark as "arrived now" only when the snapshot indicates
        # the plane has already performed the landing job (ZY_Z) or is located
        # on a concrete site (current_site_id not None).
        if pid in (snapshot.get("arrival_plan") or {}):
            # arrival_plan already set above from snapshot; keep it
            pass
        else:
            # If the plane appears to be on-site (has site id) or finished ZY_Z,
            # consider it already arrived at current_time; otherwise leave arrival
            # plan untouched so in-air planes still obey their future arrival times.
            finished = set(plane.finished_codes or [])
            if ("ZY_Z" in finished) or (plane.current_site_id is not None):
                env.arrival_plan[pid] = (
                    env.current_time, env.landing_runway_site_for_pos)

    for plane in env.planes:
        idx = plane.current_site_id
        if idx is None:
            continue
        site_id = env.sites[idx].site_id
        if 1 <= site_id <= 28:
            env.stand_current_occupancy[site_id] = plane.plane_id

    # Devices (optional)
    _apply_device_snapshot(env, snapshot.get("devices"))
    _reapply_disturbance_state(
        env, snapshot.get("blocked_stands", []))


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
    # init env with agent count / devices
    env.reset(args.n_agents)
    if getattr(args, "use_prior", False):
        prior = KGPrior(ds=getattr(args, "prior_dim_site", 8),
                        dp=getattr(args, "prior_dim_plane", 3))
        env.attach_prior(prior, getattr(args, "prior_dim_site", 8),
                         getattr(args, "prior_dim_plane", 3))
    restore_env_from_snapshot(env, snapshot)

    # If user requested to use the trained agent (e.g. --load_model), run
    # the MARL Runner starting from this restored env (skip the internal reset)
    use_agent = bool(getattr(args, "load_model", False)
                     or getattr(args, "snapshot_use_agent", False))

    # Ensure agent args/state dims are set
    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]
    # populate other default mixer args
    args = get_mixer_args(args)

    if use_agent:
        runner = Runner(env, args)
        # Collect evaluation episodes by calling rolloutWorker.generate_episode
        # with skip_reset=True so it runs from the restored env state.
        agg_reward = 0.0
        agg_time = 0.0
        agg_move = 0.0
        win_count = 0
        last_info = {"episodes_situation": [],
                     "devices_situation": [], "time": env.current_time}
        eval_epoch = int(getattr(args, 'evaluate_epoch', 1))
        for ep in range(eval_epoch):
            _, ep_reward, ep_time, win_tag, for_gant, move_time, for_devices = runner.rolloutWorker.generate_episode(
                ep, evaluate=True, skip_reset=True)
            agg_reward += float(ep_reward)
            agg_time += float(ep_time)
            agg_move += float(move_time)
            if win_tag:
                win_count += 1
            last_info = {"episodes_situation": for_gant,
                         "devices_situation": for_devices or [], "time": float(ep_time)}

        avg_reward = agg_reward / max(1, eval_epoch)
        avg_time = agg_time / max(1, eval_epoch)
        avg_move = agg_move / max(1, eval_epoch)
        win_rate = win_count / max(1, eval_epoch)

        final_info = {
            "reward": avg_reward,
            "time": avg_time,
            "devices_situation": last_info.get("devices_situation", []),
            "episodes_situation": last_info.get("episodes_situation", []),
            "win_rate": win_rate,
            "move_time": avg_move,
            "completion": summarize_plane_completion(env)
        }
        if env.enable_disturbance:
            final_info["disturbance"] = {
                "active_stands": sorted(env.disturbance_blocked_stands),
                "history": copy.deepcopy(env.disturbance_history)
            }
        return final_info

    # Fallback: run greedy policy loop (existing behaviour)
    policy = policy_fn or greedy_idle_policy
    final_info: Dict[str, Any] = {"episodes_situation": [
    ], "devices_situation": [], "time": env.current_time}
    max_iters = max_steps or env.episode_limit
    total_reward = 0.0

    for _ in range(max_iters):
        actions = policy(env)
        reward, terminated, info = env.step(actions)
        total_reward += float(reward)
        final_info = {**info}
        if terminated:
            break
    final_info["reward"] = total_reward
    final_info["completion"] = summarize_plane_completion(env)
    if env.enable_disturbance:
        final_info["disturbance"] = {
            "active_stands": sorted(env.disturbance_blocked_stands),
            "history": copy.deepcopy(env.disturbance_history)
        }
    return final_info


__all__ = [
    "restore_env_from_snapshot",
    "infer_schedule_from_snapshot",
    "greedy_idle_policy",
    "summarize_plane_completion",
]
