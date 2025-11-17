# environment.py
import ast
import copy
import json
import os
import re

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from utils.util import count_path_on_road
from utils.job import Jobs, Job
from utils.task import Task
from utils.plane import Planes, Plane
from utils.site import Sites, Site

# 固定/移动设备


# LOG：固定资源实现逻辑
@dataclass
class FixedDevice:
    dev_id: str
    rtype: str
    cover_stands: Set[int]
    # FIXME:按照勘误修改为2
    capacity: int = 2
    in_use: Set[int] = None
    is_down: bool = False

    def __post_init__(self):
        if self.in_use is None:
            self.in_use = set()

    def can_serve(self, stand_id: int) -> bool:
        if self.is_down:
            return False
        return (stand_id in self.cover_stands) and (len(self.in_use) < self.capacity)


# LOG：移动资源实现逻辑
# LOG：路径规划实现逻辑
@dataclass
class MobileDevice:
    dev_id: str
    rtype: str
    loc_stand: int
    speed_m_s: float = 3.0
    busy_until_min: float = 0.0
    locked_by: Optional[int] = None
    is_down: bool = False

    def eta_to_min(self, from_pos: np.ndarray, to_pos: np.ndarray, now_min: float) -> float:
        speed_m_per_min = float(self.speed_m_s) * 60.0
        travel_min = count_path_on_road(from_pos, to_pos, speed_m_per_min)
        return max(now_min, self.busy_until_min) + travel_min


class ScheduleEnv:
    """海天杯对齐的调度环境（动作=选择站位/跑道或 WAIT/BUSY/DONE；时间以分钟计）"""

    def __init__(self, args=None):
        self.args = args
        # 作业/任务库
        self.jobs_obj = Jobs()
        self.task_obj = Task()
        # 站位/跑道集合（根据 jobs 构造默认布局，可替换为技术资料地图）
        self.sites_obj = Sites(self.jobs_obj)
        self.sites: List[Site] = self.sites_obj.sites_object_list

        # 统一时间：分钟
        self.current_time: float = 0.0
        self.min_time_unit: float = 0.1

        # 先验与 obs/state 预留
        self.prior = None
        self.prior_dim_site = 8
        self.prior_dim_plane = 3
        self.obs_pad = getattr(args, "obs_pad", 32) if args is not None else 32

        # 课程开关
        self.enable_deps = getattr(
            args, "enable_deps", True) if args is not None else True
        self.enable_mutex = getattr(
            args, "enable_mutex", True) if args is not None else True
        self.enable_dynres = getattr(
            args, "enable_dynres", True) if args is not None else True
        self.enable_space = getattr(
            args, "enable_space", True) if args is not None else True
        self.enable_long_occupy = getattr(
            args, "enable_long_occupy", False) if args is not None else False
        self.enable_disturbance = getattr(
            args, "enable_disturbance", False) if args is not None else False
        self.penalty_idle_per_min = float(getattr(
            args, "penalty_idle_per_min", 0.05) if args is not None else 0.05)
        self.disturbance_events: List[Dict[str, Any]] = []
        self.disturbance_blocked_stands: Set[int] = set()
        self.disturbance_blocked_resource_types: Set[str] = set()
        self.disturbance_blocked_device_ids: Set[str] = set()
        self.disturbance_snapshots: Dict[int, Dict[str, Any]] = {}
        self.disturbance_history: List[Dict[str, Any]] = []
        self.disturbance_forced_planes: Set[int] = set()
        self.disturbance_forced_from: Dict[int, Optional[int]] = {}
        # 记录当前step内被动作提前占用的停机位，避免同一步出现多机同位
        self._reserved_stands: Dict[int, int] = {}
        # 记录已被分配为“在途目标”的停机位，阻止其它飞机在其空闲期间再次分配
        self._incoming_stand_reserved: Dict[int, int] = {}
        self._incoming_target_by_plane: Dict[int, int] = {}

        # 干涉表（技术资料）
        self.inbound_block: Dict[int, Tuple[int, float]] = {
            2: (3, 2), 6: (5, 2), 8: (9, 2), 15: (16, 2), 18: (17, 2), 27: (26, 2)}
        self.runway_block: Dict[int, List[Tuple[int, float]]] = {
            10: [(29, 2), (30, 2)], 11: [(29, 2), (30, 2)], 12: [(29, 2), (30, 2)],
            13: [(29, 2), (30, 2)], 14: [(29, 2), (30, 2)], 23: [(30, 2), (31, 2)],
            24: [(30, 2), (31, 2)], 29: [(30, 0.5)], 30: [(29, 0.5)]
        }
        # FIXME:暂时用29号跑道作为着陆点
        self.landing_runway_site_for_pos = 29              # 用 29 的坐标来放置飞机位置即可
        # 记录到日志时用“虚拟ID=0”，避免被当成 29/30/31
        self.landing_virtual_site_id = 0
        # LOG:着陆间隔时间
        self.landing_sep_min = getattr(self.args, "landing_sep_min", 0)
        self.landing_busy_until = -1e9

        self.last_leave_time_by_stand: Dict[int, float] = {}
        self.last_leave_time_by_runway: Dict[int, float] = {
            29: -1e9, 30: -1e9, 31: -1e9}
        # LOG:到达计划
        self.arrival_gap_min = getattr(self.args, "arrival_gap_min", 0)
        # plane_id -> (eta_min, stand_id)
        self.arrival_plan: Dict[int, Tuple[float, int]] = {}

        # 设备池 & 站位占用
        self.fixed_devices: List[FixedDevice] = []
        self.mobile_devices: List[MobileDevice] = []
        self.stand_current_occupancy: Dict[int, Optional[int]] = {}

        # 飞机群
        self.planes_obj: Optional[Planes] = None
        self.planes: List[Plane] = []

        # 奖励参数
        self.penalty_early_runway = getattr(
            self.args, "penalty_early_runway", -2.0)    # NEW
        self.bonus_parallel_per_job = getattr(
            self.args, "bonus_parallel_per_job", 0.5)  # NEW

        # MARL接口所需
        self.n_agents = 0
        self.n_actions = len(self.sites) + 3  # 所有站位/跑道 + WAIT + BUSY + DONE
        self.episode_limit = 2000  # 可按需要调整
        self._solve_reserved = []

        # 记录（时刻, job_id, site_id, plane_id, proc_min, move_min）
        self.episodes_situation: List[Tuple[float,
                                            int, int, int, float, float]] = []
        self.episode_devices: List[Dict[str, list]] = []

        self.disturbance_events = self._load_disturbance_events()

    # ============ 基础时间计算 ============
    def _sec_to_min(self, sec: float) -> float:
        return sec / 60.0

    def _speed_m_per_min(self) -> float:
        sp_m_s = getattr(self.planes_obj, "plane_speed", 5.0)
        return float(sp_m_s) * 60.0

    def _proc_time_minutes(self, job: Job, plane: Plane,
                           from_site: Optional[Site], to_site: Optional[Site]) -> float:
        code = job.code
        speed_m_per_min = self._speed_m_per_min()
        if code == "ZY_M":   # 滑行
            assert to_site is not None
            return count_path_on_road(plane.position, to_site.absolute_position, speed_m_per_min)
        if code == "ZY_T":   # 转运
            assert from_site is not None and to_site is not None
            return count_path_on_road(from_site.absolute_position, to_site.absolute_position, speed_m_per_min)
        if code == "ZY10":   # 加燃油（每分钟5%）
            need = max(0.0, 100.0 - float(getattr(plane, "fuel_percent", 20.0)))
            return need / 5.0
        if job.time_span > 0:
            return float(job.time_span)
        return 1.0

    # ============ 干涉与批次门控 ============
    def _is_stand_occupied(self, stand_id: int) -> bool:
        return self.stand_current_occupancy.get(stand_id, None) is not None

    def _reserve_stand_for_step(self, stand_id: int, plane_id: int):
        """
        在当前决策步中预留停机位，防止其它飞机在本步再次选择同一位置。
        仅处理 1..28 号停机位。
        """
        if stand_id not in self.stand_current_occupancy:
            return
        holder = self._reserved_stands.get(stand_id)
        if holder is not None and holder != plane_id:
            return
        self._reserved_stands[stand_id] = plane_id

    def _reserve_incoming_stand(self, stand_id: int, plane_id: int):
        """
        记录 plane_id 正在赶往 stand_id，在到达前阻止其它飞机再次分配。
        """
        if stand_id not in self.stand_current_occupancy:
            return
        prev = self._incoming_target_by_plane.get(plane_id)
        if prev is not None and prev != stand_id:
            self._incoming_stand_reserved.pop(prev, None)
        self._incoming_stand_reserved[stand_id] = plane_id
        self._incoming_target_by_plane[plane_id] = stand_id

    def _release_incoming_stand(self, plane_id: int, stand_id: Optional[int] = None):
        """
        解除 plane_id 的在途停机位预约（到达或被打断时调用）。
        """
        tgt = stand_id
        if tgt is None:
            tgt = self._incoming_target_by_plane.pop(plane_id, None)
        else:
            if self._incoming_target_by_plane.get(plane_id) == tgt:
                self._incoming_target_by_plane.pop(plane_id, None)
        if tgt is None:
            return
        holder = self._incoming_stand_reserved.get(tgt)
        if holder == plane_id:
            self._incoming_stand_reserved.pop(tgt, None)

    def _occupy_stand(self, stand_id: int, plane_id: int):
        """
        标记某个停机位被 plane_id 占用（物理占位）。
        若已被其他飞机占用则抛出异常，方便暴露逻辑错误。
        """
        if stand_id not in self.stand_current_occupancy:
            # 只管理 1~28 号停机位，跑道 29~31 不在此表中
            return
        cur = self.stand_current_occupancy.get(stand_id)
        if cur is None:
            self.stand_current_occupancy[stand_id] = plane_id
        elif cur != plane_id:
            # 如果触发这里，说明逻辑确实产生了“多机同位”，直接抛错
            raise AssertionError(
                f"Stand {stand_id} already occupied by plane {cur}, cannot occupy by plane {plane_id}"
            )

    def _release_stand(self, stand_id: int, plane_id: Optional[int] = None):
        """
        释放某个停机位占用，并记录 last_leave_time_by_stand。
        plane_id 若不为 None，则只在当前占用者等于 plane_id 时才释放。
        """
        if stand_id not in self.stand_current_occupancy:
            return
        cur = self.stand_current_occupancy.get(stand_id)
        if cur is None:
            return
        if (plane_id is not None) and (cur != plane_id):
            # 当前记录的是别的飞机，占用不归这个 plane_id 管，跳过
            return
        self.stand_current_occupancy[stand_id] = None
        self.last_leave_time_by_stand[stand_id] = self.current_time


        # LOG：碰撞冲突检测及避让实现逻辑
    def _stand_available_inbound(self, stand_id: int) -> bool:
        if not self.enable_space:
            return True
        if stand_id not in self.inbound_block:
            return True
        interferer, wait_min = self.inbound_block[stand_id]
        if self._is_stand_occupied(interferer):
            return False
        if self.current_time < self.last_leave_time_by_stand.get(interferer, -1e9) + wait_min:
            return False
        return True

    # LOG：碰撞冲突检测及避让实现逻辑
    def _runway_available_outbound(self, runway_id: int) -> bool:
        if not self.enable_space:
            return True
        for stand, blocks in self.runway_block.items():
            for (rw, wait_min) in blocks:
                if rw == runway_id:
                    if self._is_stand_occupied(stand):
                        return False
                    if self.current_time < self.last_leave_time_by_stand.get(stand, -1e9) + wait_min:
                        return False
        if runway_id in [29, 30]:
            other = 30 if runway_id == 29 else 29
            if self.current_time < self.last_leave_time_by_runway.get(other, -1e9) + 0.5:
                return False
        return True

    def _batch_ready_for_takeoff(self) -> bool:
        return all(("ZY18" in p.finished_codes) for p in self.planes)

    def _plane_support_done(self, p) -> bool:
        # “保障组”全部完成 = 不再有 group=='保障' 的作业未完成
        for j in self.jobs_obj.jobs_object_list:
            if getattr(j, "group", "") == "保障" and j.code not in p.finished_codes:
                return False
        return True

    def _plane_has_support_left(self, p) -> bool:
        for j in self.jobs_obj.jobs_object_list:
            if getattr(j, "group", "") == "保障" and j.code not in p.finished_codes:
                return True
        return False

    def _outbound_ready_for(self, plane):
        """当该机支持完成且已解固时，返回可执行的出场作业（ZY_S / ZY_F），用于跑道位。"""
        out = []
        if (self._plane_support_done(plane) and ("ZY_L" in plane.finished_codes)):
            for code in ("ZY_S", "ZY_F"):
                j = self.jobs_obj.get_job(code)
                if self.task_obj.graph._deps_satisfied(code, plane.finished_codes):
                    out.append(j)
        return out

    def _batch_support_done(self) -> bool:
        return all(self._plane_support_done(p) for p in self.planes)

    def _batch_support_done_for_plane(self, plane_id: int) -> bool:
        """
        Return True if the support tasks for the batch that `plane_id` belongs to
        have all been finished. In non-batch mode, always True.
        """
        if not getattr(self.args, 'batch_mode', False):
            return True
        batch_size = int(getattr(self.args, 'batch_size_per_batch', 1))
        if batch_size <= 0:
            return True
        batch_idx = int(plane_id) // batch_size
        start = batch_idx * batch_size
        end = min(start + batch_size, len(self.planes))
        for pid in range(start, end):
            p = self.planes[pid]
            if not self._plane_support_done(p):
                return False
        return True

    # ADD:设备池

    # LOG：固定资源实现逻辑
    def _build_fixed_devices(self):
        cover = {
            "FR1": ("R001", range(1, 7)),  "FR2": ("R001", range(7, 15)), "FR3": ("R001", range(15, 23)), "FR4": ("R001", range(23, 29)),
            "FR5": ("R002", range(1, 7)),  "FR6": ("R002", range(7, 15)), "FR7": ("R002", range(15, 23)), "FR8": ("R002", range(23, 29)),
            "FR9": ("R003", range(1, 7)), "FR10": ("R003", range(7, 15)), "FR11": ("R003", range(15, 23)), "FR12": ("R003", range(23, 29)),
            "FR13": ("R005", range(1, 7)), "FR14": ("R005", range(7, 15)), "FR15": ("R005", range(15, 23)), "FR16": ("R005", range(23, 29)),
            "FR17": ("R006", range(1, 7)), "FR18": ("R006", range(7, 15)), "FR19": ("R006", range(15, 23)), "FR20": ("R006", range(23, 29)),
            "FR21": ("R007", range(1, 7)), "FR22": ("R007", range(7, 15)), "FR23": ("R007", range(15, 23)), "FR24": ("R007", range(23, 29)),
            "FR25": ("R008", range(1, 7)), "FR26": ("R008", range(7, 15)), "FR27": ("R008", range(15, 23)), "FR28": ("R008", range(23, 29)),
            "FR29": ("R008", range(1, 7)), "FR30": ("R008", range(7, 15)), "FR31": ("R008", range(15, 23)), "FR32": ("R008", range(23, 29)),
        }
        self.fixed_devices = []
        for k, (rtype, rng) in cover.items():
            self.fixed_devices.append(FixedDevice(
                k, rtype, set(list(rng)), capacity=2))

    # LOG：移动资源实现逻辑
    def _build_mobile_devices(self):
        init = [
            ("MR01", "R002", 5), ("MR02", "R003", 6), ("MR03",
                                                       "R005", 13), ("MR04", "R007", 14), ("MR05", "R008", 15),
            ("MR06", "R011", 1), ("MR07", "R011",
                                  7), ("MR08", "R011", 15), ("MR09", "R011", 23),
            ("MR10", "R012", 1), ("MR11", "R012",
                                  7), ("MR12", "R012", 15), ("MR13", "R012", 23),
            ("MR14", "R013", 1), ("MR15", "R013",
                                  7), ("MR16", "R013", 15), ("MR17", "R013", 23),
            ("MR18", "R014", 1), ("MR19", "R014",
                                  7), ("MR20", "R014", 15), ("MR21", "R014", 23),
            ("MR22", "R014", 29), ("MR23", "R014", 29), ("MR24", "R014",
                                                         29), ("MR25", "R014", 29), ("MR26", "R014", 29), ("MR27", "R014", 29),
        ]
        self.mobile_devices = [MobileDevice(*x) for x in init]

    def _eta_fixed_available(self, rtype: str, stand_id: int) -> float:
        now = self.current_time
        pools = [d for d in self.fixed_devices if d.rtype ==
                 rtype and d.can_serve(stand_id)]
        if not pools:
            return float('inf')
        for d in pools:
            if len(d.in_use) < d.capacity:
                return now
        id2plane = {p.plane_id: p for p in self.planes}
        best = float('inf')
        for d in pools:
            for pid in d.in_use:
                p = id2plane[pid]
                if p is not None and p.status == "PROCESSING" and p.eta_proc_end > 0:
                    best = min(best, now + p.eta_proc_end)
        return best

    # LOG：固定资源实现逻辑
    # LOG：移动资源实现逻辑
    # LOG：路径规划实现逻辑
    def _alloc_resources_for_job(self, job: Job, stand_id: int, plane: Plane) -> Tuple[bool, float, List[object]]:
        """
        为即将开工的 job 选择设备：
        - 若有固定设备立刻可用 → 直接占用（fixed_now）
        - 否则比较“移动设备到达时间 vs 固定设备最早可用时间”：
            * 移动更早/不晚 → 锁一台移动设备（mobile），等待到达时间
            * 固定更早       → 选择等待固定设备（fixed_wait），但当前不占用设备（避免锁死产能）
        若两者都不可行（固定=∞ 且没有可锁的移动设备），直接返回 can=False，避免 ∞ 等待。
        返回: (can_start:bool, extra_wait:float, handles:list[FixedDevice|MobileDevice|None])
        其中 fixed_wait 的句柄用 None 占位，释放/记录时会被忽略。
        """
        need = job.required_resources
        if not need:
            return True, 0.0, []
        for rt in need:
            if self._is_resource_type_blocked(rt):
                return False, 0.0, []

        plans: List[Tuple[str, object, float]] = []
        now = self.current_time
        site_pos = self.sites[stand_id - 1].absolute_position
        extra_wait = 0.0

        # —— 第一步：规划（杜绝 ∞ 等待） —— #
        for rt in need:
            eta_fix = self._eta_fixed_available(rt, stand_id)

            best_m, eta_mob = None, float("inf")
            for m in self.mobile_devices:
                if m.rtype == rt and (not m.is_down) and m.locked_by is None:
                    from_pos = self.sites[m.loc_stand - 1].absolute_position
                    eta = m.eta_to_min(from_pos, site_pos, now)
                    if eta < eta_mob:
                        eta_mob, best_m = eta, m

            if eta_fix <= now:
                plans.append(("fixed_now", rt, now))
            elif (best_m is not None) and (eta_mob <= eta_fix):
                plans.append(("mobile", best_m, eta_mob))
                extra_wait = max(extra_wait, eta_mob - now)
            elif eta_fix < float("inf"):
                plans.append(("fixed_wait", rt, eta_fix))
                extra_wait = max(extra_wait, eta_fix - now)
            else:
                # 固定=∞ 且没有可用移动设备：当前作业不可行，直接失败，避免 ∞ 等待
                return False, 0.0, []

        # —— 第二步：执行占用/锁定（含 fixed_now 失败时的“二选一比较”） —— #
        handles: List[object] = []
        for kind, obj, _ in plans:
            if kind == "fixed_now":
                rt = obj  # obj 是 rtype 字符串
                pool = next(
                    (d for d in self.fixed_devices if d.rtype ==
                     rt and d.can_serve(stand_id)),
                    None
                )
                if pool is not None:
                    pool.in_use.add(plane.plane_id)
                    handles.append(pool)
                else:
                    # 重新评估：等固定 vs 用移动 → 取等待时间更短的（平手偏向移动）
                    eta_fix2 = self._eta_fixed_available(rt, stand_id)
                    best_m2, eta_mob2 = None, float("inf")
                    for m in self.mobile_devices:
                        if m.rtype == rt and (not m.is_down) and m.locked_by is None:
                            from_pos = self.sites[m.loc_stand -
                                                  1].absolute_position
                            eta = m.eta_to_min(from_pos, site_pos, now)
                            if eta < eta_mob2:
                                eta_mob2, best_m2 = eta, m

                    dt_fix = (
                        eta_fix2 - now) if eta_fix2 < float("inf") else float("inf")
                    dt_mob = (
                        eta_mob2 - now) if best_m2 is not None else float("inf")

                    if dt_mob <= dt_fix:
                        if best_m2 is None:
                            return False, 0.0, []
                        best_m2.locked_by = plane.plane_id
                        extra_wait = max(extra_wait, max(0.0, dt_mob))
                        handles.append(best_m2)
                    elif dt_fix < float("inf"):
                        extra_wait = max(extra_wait, max(0.0, dt_fix))
                        handles.append(None)  # 等固定，不立即占用
                    else:
                        return False, 0.0, []

            elif kind == "mobile":
                m: MobileDevice = obj
                if m.locked_by is None:
                    m.locked_by = plane.plane_id
                    handles.append(m)
                else:
                    # 被并发锁走了 → 尝试同类替代
                    best_m2, eta2 = None, float("inf")
                    for mm in self.mobile_devices:
                        if mm.rtype == m.rtype and (not mm.is_down) and mm.locked_by is None:
                            from_pos = self.sites[mm.loc_stand -
                                                  1].absolute_position
                            eta = mm.eta_to_min(from_pos, site_pos, now)
                            if eta < eta2:
                                eta2, best_m2 = eta, mm
                    if best_m2 is None:
                        return False, 0.0, []
                    best_m2.locked_by = plane.plane_id
                    extra_wait = max(extra_wait, eta2 - now)
                    handles.append(best_m2)

            elif kind == "fixed_wait":
                handles.append(None)

        return True, extra_wait, handles

    def _release_resources(self, handles: List[object], plane: Plane, stand_id: int):
        for h in handles or []:
            if isinstance(h, FixedDevice):
                h.in_use.discard(plane.plane_id)
            elif isinstance(h, MobileDevice):
                h.busy_until_min = self.current_time
                h.locked_by = None
                h.loc_stand = stand_id

    # ============ 先验
    def attach_prior(self, prior, prior_dim_site=8, prior_dim_plane=3):
        self.prior = prior
        self.prior_dim_site = prior_dim_site
        self.prior_dim_plane = prior_dim_plane

    def _pad_obs_tail(self, base_obs_list: List[np.ndarray]) -> List[np.ndarray]:
        if self.prior is None:
            return [np.concatenate([ob, np.zeros(self.obs_pad, dtype=np.float32)], -1) for ob in base_obs_list]
        site_ids = [s.site_id for s in self.sites]
        site_pos = [s.absolute_position for s in self.sites]
        site_caps = [dict(zip(s.resource_ids_list, s.resource_number))
                     for s in self.sites]
        site_pri = self.prior.site_prior(
            site_ids, site_pos, site_caps)  # [S, ds]
        plane_pri = self.prior.plane_prior(
            self.planes)                  # [A, dp]
        site_stats = np.concatenate(
            [site_pri.mean(0), site_pri.min(0), site_pri.max(0)], -1)
        out = []
        for i, ob in enumerate(base_obs_list):
            agg = np.concatenate([plane_pri[i], site_stats], -1)
            pad = np.zeros(
                max(0, self.obs_pad - agg.shape[-1]), dtype=np.float32)
            out.append(np.concatenate([ob, agg, pad], -1))
        return out

    def _pad_state_tail(self, base_state: np.ndarray) -> np.ndarray:
        if self.prior is None:
            return np.concatenate([base_state, np.zeros(self.obs_pad, dtype=np.float32)], -1)
        site_ids = [s.site_id for s in self.sites]
        site_pos = [s.absolute_position for s in self.sites]
        site_caps = [dict(zip(s.resource_ids_list, s.resource_number))
                     for s in self.sites]
        site_pri = self.prior.site_prior(
            site_ids, site_pos, site_caps)  # [S, ds]
        plane_pri = self.prior.plane_prior(
            self.planes)                  # [A, dp]
        agg = np.concatenate([
            site_pri.mean(0), site_pri.min(0), site_pri.max(0),
            plane_pri.mean(0), plane_pri.min(0), plane_pri.max(0)
        ], -1)
        pad = np.zeros(max(0, self.obs_pad - agg.shape[-1]), dtype=np.float32)
        return np.concatenate([base_state, agg, pad], -1)

    # ============ Env 基本接口 ============
    def reset(self, n_agents: int):
        # 飞机群
        self.planes_obj = Planes(n_agents)
        self.planes = self.planes_obj.planes_object_list
        self.n_agents = n_agents
        for p in self.planes:
            p.fuel_percent = 20.0  # 外部进场；若一开始就在停机位，可设为100
            p._last_handles = []
            p._active_jobs = None

        # 站位占用与干涉时钟
        self.stand_current_occupancy = {sid: None for sid in range(1, 29)}
        self.last_leave_time_by_stand = {sid: -1e9 for sid in range(1, 29)}
        self.last_leave_time_by_runway = {29: -1e9, 30: -1e9, 31: -1e9}

        # 设备池
        self._build_fixed_devices()
        self._build_mobile_devices()

        # 时间与轨迹
        # 默认从 0 开始；若启用了批次模式，可把 current_time 重设为批次起始时间
        self.current_time = 0.0
        self.episodes_situation = []
        self.episode_devices = []
        self.disturbance_blocked_stands.clear()
        self.disturbance_blocked_resource_types.clear()
        self.disturbance_blocked_device_ids.clear()
        self.disturbance_snapshots = {}
        self.disturbance_history = []
        self.disturbance_forced_planes.clear()
        self.disturbance_forced_from.clear()
        self.disturbance_events = self._load_disturbance_events()
        self._reserved_stands.clear()
        self._incoming_stand_reserved.clear()
        self._incoming_target_by_plane.clear()

        # 更新动作数 —— 构造到达计划
        self.arrival_plan.clear()
        # 如果传入 args 并启用了 batch_mode，则构造分批到达时间表；否则按线性 gap
        if getattr(self.args, 'batch_mode', False):
            # 参数：batch_start_time_min, batch_size, batches_count, intra_gap_min, inter_batch_gap_min
            start_time = float(
                getattr(self.args, 'batch_start_time_min', 7*60))
            batch_size = int(getattr(self.args, 'batch_size_per_batch', 12))
            # batches_count 可以用来检查或作为提示（实际 n_agents 决定实际飞机数）
            batches_count = int(getattr(self.args, 'batches_count', 1))
            intra_gap = float(
                getattr(self.args, 'intra_gap_min', self.arrival_gap_min))
            inter_gap = float(getattr(self.args, 'inter_batch_gap_min', 0))

            # 将仿真时间起点移到首批起始时间（符合“某飞行日早上07:00开始”需求）
            self.current_time = start_time

            # 计算每架飞机的到达时间：
            # batch_start_k = start_time + k * ((batch_size-1)*intra_gap + inter_gap)
            # plane_in_batch j: t = batch_start_k + j * intra_gap
            for pid in range(n_agents):
                batch_idx = pid // batch_size
                pos_in_batch = pid % batch_size
                batch_start = start_time + batch_idx * \
                    ((max(0, batch_size - 1) * intra_gap) + inter_gap)
                t_arr = batch_start + pos_in_batch * intra_gap
                self.arrival_plan[pid] = (
                    t_arr, self.landing_runway_site_for_pos)
        else:
            for pid in range(n_agents):
                t_arr = pid * self.arrival_gap_min
                self.arrival_plan[pid] = (
                    t_arr, self.landing_runway_site_for_pos)  # 全部用“落地跑道”的位置
        self.landing_busy_until = -1e9

        # 清空已完成集，飞机不在任何站位（表示“在空中等待落地”）
        for p in self.planes:
            p.finished_codes.clear()
            p.current_site_id = None
            p.position = np.array([0.0, 0.0], dtype=float)  # 仅占位，无实际意义
            p.status = "IDLE"  # 仍用 IDLE，但通过到达计划限制动作
            p.paused_jobs = []
            p.active_job_progress = {}
            p._job_total_durations = {}
            p._active_event_indices = []
            p._handles_by_job = {}
            # 返回初始观测（若 MARL 框架不需要，可忽略）
        self._process_disturbance_timeline(initial=True)
        return self.get_obs()

    def _pick_stage_runway_index(self, pid: int) -> Optional[int]:
        preferred = getattr(self.args, "stage_runway_order", [29, 30, 31])
        for rid in preferred:
            idx = next((i for i, s in enumerate(self.sites)
                        if s.site_id == rid), None)
            if idx is None:
                continue
            if self._runway_available_outbound(rid):
                return idx
        return None

    def _auto_stage_if_ready(self, pid: int) -> Tuple[bool, float]:
        plane = self.planes[pid]
        if plane.status != "IDLE":
            return False, 0.0
        if plane.current_site_id is None or not (0 <= plane.current_site_id < len(self.sites)):
            return False, 0.0
        site_cur = self.sites[plane.current_site_id]
        if site_cur.is_runway:
            return False, 0.0
        if not self._plane_support_done(plane):
            return False, 0.0
        if "ZY_L" not in plane.finished_codes:
            return False, 0.0
        runway_idx = self._pick_stage_runway_index(pid)
        if runway_idx is None:
            return False, 0.0
        moved, penalty = self._handle_move_action(
            pid, runway_idx, allow_early_runway=True)
        return moved, penalty

    def get_env_info(self):
        # 先构造 state 用来取维度（我们刚刚实现了 get_state，未 reset 也能返回占位）
        state = self.get_state()

        # obs：若未 reset 返回空，则构造一个“单机占位 obs”计算维度
        obs = self.get_obs()
        if isinstance(obs, list) and len(obs) == 0:
            dummy = self._pad_obs_tail([np.zeros(8, dtype=np.float32)])[0]
            obs_shape = int(dummy.shape[-1])
        else:
            obs_shape = (len(obs[0]) if isinstance(
                obs, list) else int(obs.shape[-1]))

        n_agents_info = self.n_agents if self.n_agents > 0 else int(
            getattr(self.args, "n_agents", 0))

        return dict(
            n_actions=self.n_actions,
            n_agents=n_agents_info,
            state_shape=int(
                state.shape[-1]) if isinstance(state, np.ndarray) else int(len(state)),
            obs_shape=obs_shape,
            episode_limit=self.episode_limit
        )

    # 观测/状态（示例：把关键状态拼接成向量；可按你们原实现保留）

    def get_obs(self) -> List[np.ndarray]:
        obs_list = []
        for p in self.planes:
            status_oh = np.array([
                1.0 if p.status == "IDLE" else 0.0,
                1.0 if p.status == "MOVING" else 0.0,
                1.0 if p.status == "PROCESSING" else 0.0,
                1.0 if p.status == "DONE" else 0.0,
            ], dtype=np.float32)
            left = float(p.left_jobs_count)
            feat = np.concatenate([
                p.position.astype(np.float32),            # 2
                status_oh,                               # 4
                np.array([p.fuel_percent/100.0], np.float32),  # 1
                np.array([left], np.float32),             # 1
            ], -1)  # 8 dims
            obs_list.append(feat)
        return self._pad_obs_tail(obs_list)

    def get_state(self) -> np.ndarray:
        """
        全局状态向量：
        - 每架飞机 8 维（与 get_obs 的单机特征一致）：pos(2)+status onehot(4)+fuel(1)+left_jobs(1)
        - 站位占用标志（len(self.sites) 维，跑道位默认 0）
        - 当前时间（1 维，归一化可选）
        并通过 _pad_state_tail(...) 拼接先验/统计。
        兼容“未 reset 前先 get_env_info”：此时按 args.n_agents 构造零向量占位。
        """
        # 确定 agent 数（reset 前为 0，则回退到 args.n_agents）
        n_agents_eff = self.n_agents if self.n_agents > 0 else int(
            getattr(self.args, "n_agents", 0))

        # 1) 飞机特征：若未 reset，则用全 0 占位
        plane_feats = []
        if self.planes and len(self.planes) == n_agents_eff:
            for p in self.planes:
                status_oh = np.array([
                    1.0 if p.status == "IDLE" else 0.0,
                    1.0 if p.status == "MOVING" else 0.0,
                    1.0 if p.status == "PROCESSING" else 0.0,
                    1.0 if p.status == "DONE" else 0.0,
                ], dtype=np.float32)
                left = float(p.left_jobs_count)
                feat = np.concatenate([
                    p.position.astype(np.float32),                  # 2
                    status_oh,                                      # 4
                    np.array([p.fuel_percent/100.0], np.float32),   # 1
                    np.array([left], np.float32),                   # 1
                ], -1)  # 共 8 维
                plane_feats.append(feat)
        else:
            # 未 reset 或数量不一致：构造零向量
            for _ in range(n_agents_eff):
                plane_feats.append(np.zeros(8, dtype=np.float32))
        plane_feats = np.concatenate(
            plane_feats, axis=-1) if len(plane_feats) > 0 else np.zeros(0, dtype=np.float32)

        # 2) 站位占用（停机位 1..28 使用占用表；跑道 29..31 置 0）
        occ = []
        for s in self.sites:
            if s.site_id <= 28:
                occ.append(1.0 if self.stand_current_occupancy.get(
                    s.site_id, None) is not None else 0.0)
            else:
                occ.append(0.0)
        occ = np.array(occ, dtype=np.float32)

        # 3) 当前时间
        t = np.array([float(self.current_time)], dtype=np.float32)

        base_state = np.concatenate(
            [plane_feats, occ, t], -1).astype(np.float32)
        return self._pad_state_tail(base_state)

    # ============ 扰动事件工具 ============

    def _normalize_stand_list(self, value) -> List[int]:
        stands: Set[int] = set()
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            for v in value:
                try:
                    stands.add(int(v))
                except Exception:
                    continue
        elif isinstance(value, str):
            parts = [p.strip() for p in value.split(',') if p.strip()]
            for token in parts:
                if '-' in token:
                    a, b = token.split('-', 1)
                    try:
                        start = int(a)
                        end = int(b)
                    except Exception:
                        continue
                    lo, hi = sorted((start, end))
                    for sid in range(lo, hi + 1):
                        stands.add(sid)
                else:
                    try:
                        stands.add(int(token))
                    except Exception:
                        continue
        else:
            try:
                stands.add(int(value))
            except Exception:
                pass
        return sorted(s for s in stands if 1 <= s <= 28)

    def _normalize_equipment_tokens(self, value) -> List[str]:
        tokens: List[str] = []
        if value is None:
            return tokens
        if isinstance(value, str):
            parts = re.split(r'[;,\\s]+', value.strip())
            tokens.extend([p for p in parts if p])
            return tokens
        if isinstance(value, (list, tuple, set)):
            for entry in value:
                entry_tokens = self._normalize_equipment_tokens(entry)
                tokens.extend(entry_tokens)
            return tokens
        if isinstance(value, dict):
            for entry in value.values():
                tokens.extend(self._normalize_equipment_tokens(entry))
            return tokens
        tokens.append(str(value).strip())
        return [t for t in tokens if t]

    def _normalize_equipment_scope(self, event_payload: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        resource_tokens: List[str] = []
        for key in ("resource_types", "resources", "reqs", "req_resources"):
            resource_tokens.extend(
                self._normalize_equipment_tokens(event_payload.get(key)))
        device_tokens: List[str] = []
        for key in ("devices", "device_ids", "equipment"):
            device_tokens.extend(
                self._normalize_equipment_tokens(event_payload.get(key)))
        tokens = resource_tokens + device_tokens
        resource_types: Set[str] = set()
        device_ids: Set[str] = set()
        for token in tokens:
            up = token.upper()
            if not up:
                continue
            if re.match(r'^R\d+$', up):
                resource_types.add(up)
            else:
                device_ids.add(up)
        return sorted(resource_types), sorted(device_ids)

    def _is_resource_type_blocked(self, rtype: str) -> bool:
        return rtype.upper() in self.disturbance_blocked_resource_types

    def _set_device_down(self, dev_id: str, down: bool, purge_usage: bool = True) -> bool:
        target = None
        for d in self.fixed_devices:
            if d.dev_id.upper() == dev_id.upper():
                target = d
                break
        if target is None:
            for d in self.mobile_devices:
                if d.dev_id.upper() == dev_id.upper():
                    target = d
                    break
        if target is None:
            return False
        target.is_down = down
        if not down:
            return True
        # When down, release locks/usage immediately
        if purge_usage:
            if isinstance(target, FixedDevice):
                target.in_use.clear()
            elif isinstance(target, MobileDevice):
                target.locked_by = None
                target.busy_until_min = self.current_time
        return True

    def _refresh_handles_cache(self, plane: Plane):
        handles_map = getattr(plane, "_handles_by_job", {})
        flat: List[object] = []
        for hlist in handles_map.values():
            flat.extend(hlist or [])
        plane._last_handles = flat

    def _assign_handles_to_jobs(self, plane: Plane, mapping: Dict[str, List[object]]):
        plane._handles_by_job = {}
        for code, handles in (mapping or {}).items():
            plane._handles_by_job[code] = list(handles or [])
        self._refresh_handles_cache(plane)

    def _load_disturbance_events(self) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        if not self.enable_disturbance:
            return events
        cfg = getattr(self.args, "disturbance_events",
                      "") if self.args is not None else ""
        if not cfg:
            return events
        raw = None
        try:
            if os.path.exists(cfg):
                with open(cfg, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
            else:
                raw = self._parse_disturbance_text(cfg)
        except Exception as exc:
            print(f"[Disturbance] Failed to parse disturbance_events: {exc}")
            return []
        if isinstance(raw, dict):
            raw = [raw]
        try:
            iterable = list(raw)
        except Exception:
            return []

        for idx, evt in enumerate(iterable):
            try:
                start = float(evt.get("start"))
                end = float(evt.get("end"))
            except Exception:
                continue
            if not np.isfinite(start) or not np.isfinite(end) or end <= start:
                continue
            stands = self._normalize_stand_list(
                evt.get("stands") or evt.get("sites") or evt.get("site_ids"))
            resource_types, device_ids = self._normalize_equipment_scope(evt)
            if not stands and not resource_types and not device_ids:
                continue
            events.append({
                "id": idx,
                "start": float(start),
                "end": float(end),
                "stands": stands,
                "resource_types": resource_types,
                "device_ids": device_ids,
                "started": False,
                "completed": False,
                "snapshot": None,
                "affected_planes": [],
                "meta": evt
            })
        events.sort(key=lambda x: x["start"])
        return events

    def _parse_disturbance_text(self, text: str):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        sanitized = text.strip()
        if sanitized.count("\"") == 0 and "'" in sanitized:
            sanitized = sanitized.replace("'", '"')
            try:
                return json.loads(sanitized)
            except json.JSONDecodeError:
                pass
        key_fixed = re.sub(
            r'(?<!["\\])([A-Za-z_][A-Za-z0-9_]*)\s*:', r'"\1":', sanitized)
        try:
            return json.loads(key_fixed)
        except json.JSONDecodeError:
            pass
        try:
            return ast.literal_eval(text)
        except Exception as exc:
            raise exc

    def _next_disturbance_transition(self) -> Optional[float]:
        nxt = None
        for evt in self.disturbance_events:
            target = None
            if not evt.get("started"):
                target = evt.get("start")
            elif not evt.get("completed"):
                target = evt.get("end")
            if target is None:
                continue
            if target <= self.current_time + 1e-9:
                continue
            if nxt is None or float(target) < nxt:
                nxt = float(target)
        return nxt

    def _limit_step_to_disturbance(self, delta_t: float) -> float:
        nxt = self._next_disturbance_transition()
        if nxt is None:
            return delta_t
        remaining = float(nxt - self.current_time)
        if remaining <= 1e-9:
            return delta_t
        return min(delta_t, remaining)

    def _process_disturbance_timeline(self, initial: bool = False):
        if not self.enable_disturbance:
            return
        for evt in self.disturbance_events:
            if (not evt.get("started")) and self.current_time >= evt.get("start", 0.0) - 1e-9:
                self._activate_disturbance(evt)
            if evt.get("started") and (not evt.get("completed")) and self.current_time >= evt.get("end", 0.0) - 1e-9:
                self._complete_disturbance(evt)

    def _activate_disturbance(self, event: Dict[str, Any]):
        event["started"] = True
        snapshot = self._capture_global_state()
        event["snapshot"] = snapshot
        if isinstance(event.get("id"), int):
            self.disturbance_snapshots[event["id"]] = snapshot
        stands = set(event.get("stands", []))
        resources = {r.upper()
                     for r in event.get("resource_types", []) or []}
        device_ids = {d.upper()
                      for d in event.get("device_ids", []) or []}
        self.disturbance_blocked_stands.update(stands)
        affected: List[int] = []
        for pid, plane in enumerate(self.planes):
            site_id = self._plane_site_id(plane)
            if site_id is None or site_id not in stands:
                continue
            if plane.status == "PROCESSING":
                paused = self._pause_plane_due_to_disturbance(
                    pid, plane, site_id)
                if paused:
                    affected.append(pid)
            else:
                self._evacuate_plane_from_blocked(pid, plane, site_id)
                affected.append(pid)
        if resources or device_ids:
            affected_eq = self._activate_equipment_disturbance(
                resources, device_ids)
            affected.extend(affected_eq)
        affected = sorted(set(affected))
        event["affected_planes"] = affected
        self.disturbance_history.append({
            "event_id": event.get("id"),
            "action": "start",
            "time": float(self.current_time),
            "stands": sorted(stands),
            "resources": sorted(resources),
            "devices": sorted(device_ids),
            "affected": affected
        })
        meta = []
        if stands:
            meta.append(f"stands {sorted(stands)}")
        if resources:
            meta.append(f"resources {sorted(resources)}")
        if device_ids:
            meta.append(f"devices {sorted(device_ids)}")
        desc = ", ".join(meta) if meta else "no scope"
        print(
            f"[Disturbance] Event {event.get('id')} activated at {self.current_time:.2f} min, {desc}")

    def _complete_disturbance(self, event: Dict[str, Any]):
        event["completed"] = True
        stands = set(event.get("stands", []))
        self.disturbance_blocked_stands.difference_update(stands)
        resources = {r.upper()
                     for r in event.get("resource_types", []) or []}
        if resources:
            self.disturbance_blocked_resource_types.difference_update(
                resources)
        devices = {d.upper() for d in event.get("device_ids", []) or []}
        for dev_id in devices:
            self._set_device_down(dev_id, False)
            self.disturbance_blocked_device_ids.discard(dev_id)
        self.disturbance_history.append({
            "event_id": event.get("id"),
            "action": "end",
            "time": float(self.current_time),
            "stands": sorted(stands),
            "resources": sorted(resources),
            "devices": sorted(devices)
        })
        meta = []
        if stands:
            meta.append(f"stands {sorted(stands)}")
        if resources:
            meta.append(f"resources {sorted(resources)}")
        if devices:
            meta.append(f"devices {sorted(devices)}")
        desc = ", ".join(meta) if meta else "no scope"
        print(
            f"[Disturbance] Event {event.get('id')} completed at {self.current_time:.2f} min, {desc}")

    def _activate_equipment_disturbance(self, resources: Set[str], device_ids: Set[str]) -> List[int]:
        affected: List[int] = []
        if resources:
            self.disturbance_blocked_resource_types.update(resources)
        if device_ids:
            for dev_id in device_ids:
                self.disturbance_blocked_device_ids.add(dev_id)
                self._set_device_down(dev_id, True)
        for pid, plane in enumerate(self.planes):
            paused = self._pause_plane_jobs_for_equipment(
                pid, plane, resources, device_ids)
            if paused:
                affected.append(pid)
        return affected

    def _capture_global_state(self) -> Dict[str, Any]:
        snapshot = {
            "time": float(self.current_time),
            "planes": [],
            "stand_occupancy": copy.deepcopy(self.stand_current_occupancy),
            "blocked_stands": sorted(self.disturbance_blocked_stands),
            "blocked_resources": sorted(
                self.disturbance_blocked_resource_types),
            "down_devices": sorted(self.disturbance_blocked_device_ids)
        }
        for plane in self.planes:
            plane_state = {
                "plane_id": plane.plane_id,
                "status": plane.status,
                "current_site_id": self.sites[plane.current_site_id].site_id if plane.current_site_id is not None and 0 <= plane.current_site_id < len(self.sites) else None,
                "finished_codes": sorted(list(plane.finished_codes)),
                "paused_jobs": [
                    {"code": entry.get("code"), "remaining": entry.get(
                        "remaining", 0.0)}
                    for entry in getattr(plane, "paused_jobs", [])
                ],
                "active_job": plane.current_job_code,
                "active_remaining": copy.deepcopy(getattr(plane, "active_job_progress", {}))
            }
            snapshot["planes"].append(plane_state)
        return snapshot

    def _plane_site_id(self, plane: Plane) -> Optional[int]:
        if plane.current_site_id is None or not (0 <= plane.current_site_id < len(self.sites)):
            return None
        return self.sites[plane.current_site_id].site_id

    def _pause_plane_jobs_for_equipment(self, pid: int, plane: Plane,
                                        blocked_resources: Set[str],
                                        blocked_devices: Set[str]) -> bool:
        if plane.status != "PROCESSING":
            return False
        if not blocked_resources and not blocked_devices:
            return False
        jobs: List[Job] = []
        if getattr(plane, "_active_jobs", None):
            jobs = list(plane._active_jobs)
        elif plane.current_job_code:
            parts = str(plane.current_job_code).split("+")
            job_obj = self.jobs_obj.get_job(parts[0])
            if job_obj:
                jobs = [job_obj]
        if not jobs:
            return False
        blocked_res = {r.upper() for r in blocked_resources or []}
        blocked_dev_ids = {d.upper() for d in blocked_devices or []}
        evt_idx_map = dict(getattr(plane, "_active_event_indices", []))
        progress_map = dict(getattr(plane, "active_job_progress", {}))
        total_map = dict(getattr(plane, "_job_total_durations", {}))
        keep_jobs: List[Job] = []
        paused_entries: List[Dict[str, Any]] = []
        handles_map = getattr(plane, "_handles_by_job", {})
        stand_id = self._plane_site_id(plane) or self.landing_virtual_site_id
        for job in jobs:
            req_hit = bool(blocked_res.intersection(
                [r.upper() for r in job.required_resources or []]))
            dev_hit = False
            if not req_hit and blocked_dev_ids:
                for handle in handles_map.get(job.code, []):
                    dev = getattr(handle, "dev_id", None)
                    if dev and dev.upper() in blocked_dev_ids:
                        dev_hit = True
                        break
            if not req_hit and not dev_hit:
                keep_jobs.append(job)
                continue
            remaining = float(progress_map.get(
                job.code, plane.eta_proc_end))
            total = float(total_map.get(job.code, remaining))
            if remaining <= 1e-6:
                continue
            paused_entry = {
                "job": job,
                "remaining": remaining,
                "code": job.code,
                "reason": "device_unavailable"
            }
            paused_entries.append(paused_entry)
            evt_idx = evt_idx_map.get(job.code)
            if evt_idx is not None and 0 <= evt_idx < len(self.episodes_situation):
                t, jid, sid, pid_evt, proc_min, move_min = self.episodes_situation[evt_idx]
                elapsed = max(0.0, total - remaining)
                self.episodes_situation[evt_idx] = (
                    t, jid, sid, pid_evt, float(elapsed), move_min)
            handles = handles_map.pop(job.code, [])
            if handles:
                self._release_resources(handles, plane, stand_id)
        if not paused_entries:
            return False
        plane.paused_jobs = paused_entries + plane.paused_jobs
        if keep_jobs:
            if len(keep_jobs) > 1:
                plane._active_jobs = keep_jobs
                plane.current_job_code = "+".join(j.code for j in keep_jobs)
            else:
                plane._active_jobs = None
                plane.current_job_code = keep_jobs[0].code
            plane.active_job_progress = {
                job.code: progress_map.get(job.code, 0.0) for job in keep_jobs}
            plane._job_total_durations = {
                job.code: total_map.get(
                    job.code, plane.active_job_progress[job.code])
                for job in keep_jobs}
            plane._active_event_indices = [
                (code, idx) for code, idx in evt_idx_map.items() if code in plane.active_job_progress]
            plane.eta_proc_end = max(
                plane.active_job_progress.values()) if plane.active_job_progress else 0.0
            plane.status = "PROCESSING"
        else:
            plane._active_jobs = None
            plane.current_job_code = None
            plane.active_job_progress = {}
            plane._job_total_durations = {}
            plane._active_event_indices = []
            plane.eta_proc_end = 0.0
            plane.status = "IDLE"
        self._refresh_handles_cache(plane)
        return True

    def _pause_plane_due_to_disturbance(self, pid: int, plane: Plane, stand_id: int) -> bool:
        if plane.status != "PROCESSING":
            if not plane.paused_jobs:
                plane.paused_jobs = []
            return False
        handles = getattr(plane, "_last_handles", None)
        if handles:
            self._release_resources(handles, plane, stand_id)
            plane._last_handles = []
            plane._handles_by_job = {}
        if stand_id in self.stand_current_occupancy:
            self._release_stand(stand_id, pid)
        paused_entries = []
        jobs: List[Job] = []
        if getattr(plane, "_active_jobs", None):
            jobs = list(plane._active_jobs)
        elif plane.current_job_code:
            job_obj = self.jobs_obj.get_job(plane.current_job_code)
            if job_obj:
                jobs = [job_obj]
        if not jobs:
            return False
        evt_idx_map = dict(getattr(plane, "_active_event_indices", []))
        for job in jobs:
            remaining = float(plane.active_job_progress.get(
                job.code, plane.eta_proc_end))
            total = float(plane._job_total_durations.get(job.code, remaining))
            if remaining <= 1e-6:
                continue
            paused_entries.append({
                "job": job,
                "remaining": remaining,
                "code": job.code
            })
            evt_idx = evt_idx_map.get(job.code)
            if evt_idx is not None and 0 <= evt_idx < len(self.episodes_situation):
                t, jid, sid, pid_evt, proc_min, move_min = self.episodes_situation[evt_idx]
                elapsed = max(0.0, total - remaining)
                self.episodes_situation[evt_idx] = (
                    t, jid, sid, pid_evt, float(elapsed), move_min)
        if not paused_entries:
            return False
        plane.paused_jobs.extend(paused_entries)
        plane._active_jobs = None
        plane.current_job_code = None
        plane.active_job_progress = {}
        plane._job_total_durations = {}
        plane._active_event_indices = []
        plane.status = "IDLE"
        plane.eta_proc_end = 0.0
        self._register_forced_plane(pid, plane, stand_id)
        return True

    def _evacuate_plane_from_blocked(self, pid: int, plane: Plane, stand_id: int):
        if getattr(plane, "_move_handles", None):
            self._release_resources(plane._move_handles, plane, stand_id)
            plane._move_handles = None
        self._register_forced_plane(pid, plane, stand_id)

    def _register_forced_plane(self, pid: int, plane: Plane, stand_id: Optional[int]):
        if stand_id is not None and stand_id in self.stand_current_occupancy:
            self._release_stand(stand_id, pid)
        plane.current_site_id = None
        self.disturbance_forced_planes.add(pid)
        self.disturbance_forced_from[pid] = stand_id
        self._auto_move_plane_to_available(pid, stand_id)

    def _clear_disturbance_force(self, pid: int):
        self.disturbance_forced_planes.discard(pid)
        self.disturbance_forced_from.pop(pid, None)

    def _retry_forced_relocations(self):
        for pid in list(self.disturbance_forced_planes):
            plane = self.planes[pid]
            if plane.status != "IDLE":
                continue
            from_sid = self.disturbance_forced_from.get(pid)
            self._auto_move_plane_to_available(pid, from_sid)

    def _auto_move_plane_to_available(self, pid: int, from_site_id: Optional[int] = None) -> bool:
        if not self.enable_disturbance:
            return False
        plane = self.planes[pid]
        if plane.status == "MOVING":
            return True
        target_idx = self._select_backup_stand(plane)
        if target_idx is None:
            holding = self.sites[self.landing_runway_site_for_pos -
                                 1].absolute_position
            plane.position = holding.copy()
            return False
        site_to = self.sites[target_idx]
        self._reserve_stand_for_step(site_to.site_id, pid)
        from_site = None
        if from_site_id is not None:
            for s in self.sites:
                if s.site_id == from_site_id:
                    from_site = s
                    break
        elif plane.current_site_id is not None and 0 <= plane.current_site_id < len(self.sites):
            from_site = self.sites[plane.current_site_id]
            from_site_id = from_site.site_id
        move_code = "ZY_M"
        handles = None
        wait_min = 0.0
        if from_site is not None:
            jobT = self.jobs_obj.get_job("ZY_T")
            if jobT is not None:
                can, wait_min, handles = self._alloc_resources_for_job(
                    jobT, from_site.site_id, plane)
                if can:
                    move_code = "ZY_T"
                else:
                    handles = None
                    wait_min = 0.0
        speed = self._speed_m_per_min()
        from_pos = from_site.absolute_position.copy(
        ) if from_site is not None else plane.position.copy()
        travel_min = count_path_on_road(
            from_pos, site_to.absolute_position, speed)
        total_move = float(wait_min + travel_min)
        jid = self.jobs_obj.code2id().get(move_code)
        if jid is None:
            return False
        self.episodes_situation.append(
            (self.current_time, jid, site_to.site_id, pid, 0.0, total_move))
        fix_ids, mob_ids = [], []
        for h in handles or []:
            if isinstance(h, MobileDevice):
                mob_ids.append(h.dev_id)
            elif isinstance(h, FixedDevice):
                fix_ids.append(h.dev_id)
        self.episode_devices.append(
            {"FixedDevices": fix_ids, "MobileDevices": mob_ids})
        if handles:
            plane._move_handles = handles
        plane.start_move(to_site_id=site_to.site_id - 1, move_min=total_move)
        plane.position = from_pos
        self._clear_disturbance_force(pid)
        return True

    def _select_backup_stand(self, plane: Plane) -> Optional[int]:
        candidates = []
        for idx, site in enumerate(self.sites):
            if site.is_runway:
                continue
            if site.site_id in self.disturbance_blocked_stands:
                continue
            if self.stand_current_occupancy.get(site.site_id, None) is not None:
                continue
            reserved_by = self._reserved_stands.get(site.site_id)
            if reserved_by is not None and reserved_by != plane.plane_id:
                continue
            reserved_future = self._incoming_stand_reserved.get(site.site_id)
            if reserved_future is not None and reserved_future != plane.plane_id:
                continue
            if not site.is_available(self.current_time):
                continue
            if not self._stand_available_inbound(site.site_id):
                continue
            dist = float(np.linalg.norm(
                plane.position - site.absolute_position))
            candidates.append((dist, idx))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def _try_resume_paused_jobs(self, pid: int, plane: Plane, site_cur: Site) -> bool:
        if not plane.paused_jobs:
            return False
        idx = 0
        while idx < len(plane.paused_jobs):
            entry = plane.paused_jobs[idx]
            job = entry.get("job")
            if job is None:
                idx += 1
                continue
            j_id = self.jobs_obj.code2id().get(job.code)
            if j_id is None or j_id not in site_cur.resource_ids_list:
                idx += 1
                continue
            can, wait_min, handles = self._alloc_resources_for_job(
                job, site_cur.site_id, plane)
            if not can:
                idx += 1
                continue
            remaining = max(0.0, float(entry.get("remaining", 0.0)))
            proc_total = float(wait_min + remaining)
            if proc_total <= 0.0:
                plane.paused_jobs.pop(idx)
                continue
            jid = self.jobs_obj.code2id()[job.code]
            self.episodes_situation.append(
                (self.current_time, jid, site_cur.site_id, pid, proc_total, 0.0))
            fix_ids, mob_ids = [], []
            for h in handles or []:
                if isinstance(h, FixedDevice):
                    fix_ids.append(h.dev_id)
                elif isinstance(h, MobileDevice):
                    mob_ids.append(h.dev_id)
            self.episode_devices.append(
                {"FixedDevices": fix_ids, "MobileDevices": mob_ids})

            plane.start_job(job, proc_total)
            plane._active_event_indices = [
                (job.code, len(self.episodes_situation) - 1)]
            if not site_cur.is_runway:
                self._occupy_stand(site_cur.site_id, pid)
            self._assign_handles_to_jobs(
                plane, {job.code: handles or []})

            plane.paused_jobs.pop(idx)
            return True
        return False

    def get_avail_agent_actions(self, i: int):
        mask = np.zeros(self.n_actions, dtype=np.int32)
        plane: Plane = self.planes[i]

        # 若还没到达（ZY_Z 未做且当前时刻 < 到达时刻）：只能 WAIT
        if "ZY_Z" not in plane.finished_codes:
            arr_t, _ = self.arrival_plan.get(i, (0.0, 29))
            if self.current_time < arr_t:
                mask[-3] = 1  # WAIT
                return mask
            # 到达了也不交给 RL 决策落地，交给 step() 自动触发
            mask[-3] = 1
            return mask

        # DONE
        if self.task_obj.graph.all_finished(plane.finished_codes):
            mask[-1] = 1
            return mask
        # BUSY
        if plane.status == "PROCESSING":
            mask[-2] = 1
            return mask

        # 候选作业（去掉 ZY_Z）
        ready_jobs: List[Job] = self.task_obj.graph.enabled(plane.finished_codes, plane.ongoing_mutex) \
            if (self.enable_deps or self.enable_mutex) else self.task_obj.jobs.jobs_object_list[:1]

        outbound_jobs = self._outbound_ready_for(plane)
        cand_jobs: List[Job] = list(ready_jobs)

        batch_ready = self._batch_support_done_for_plane(i)
        if batch_ready:
            cand_jobs.extend(outbound_jobs)

        cand_jobs = [j for j in cand_jobs if j.code != "ZY_Z"]

        can_stage_on_runway = self._plane_support_done(
            plane) and ("ZY_L" in plane.finished_codes)

        if len(cand_jobs) == 0 and (not can_stage_on_runway):
            mask[-3] = 1
            return mask

        # 站位枚举（加入“被占用则不可选”的过滤）
        for idx, site in enumerate(self.sites):
            if not site.is_runway:
                # 站位已被占用，则跳过（容量>1 时请改为“容量计数”）
                if self.stand_current_occupancy.get(site.site_id, None) is not None:
                    continue
                reserved_by = self._reserved_stands.get(site.site_id)
                if reserved_by is not None and reserved_by != i:
                    continue
                reserved_future = self._incoming_stand_reserved.get(site.site_id)
                if reserved_future is not None and reserved_future != i:
                    continue
            if self.enable_disturbance and site.site_id in self.disturbance_blocked_stands:
                continue
            if not site.is_available(self.current_time):
                continue

            ok = False
            for j in cand_jobs:
                j_id = self.jobs_obj.code2id()[j.code]
                if site.is_runway:
                    if batch_ready and (j.code in ("ZY_S", "ZY_F")) and self._runway_available_outbound(site.site_id):
                        ok = True

                else:
                    if j_id not in site.resource_ids_list:
                        continue
                    pos = site.resource_ids_list.index(j_id)
                    if site.resource_number[pos] <= 0:
                        continue
                    # 入位相关动作的邻位干涉
                    if j.code in ("ZY_M", "ZY01") and not self._stand_available_inbound(site.site_id):
                        continue
                    ok = True
                if ok:
                    break
            if (not ok) and site.is_runway and can_stage_on_runway:
                if self._runway_available_outbound(site.site_id):
                    ok = True
            if ok:
                mask[idx] = 1

        if mask[:-3].sum() == 0:
            mask[-3] = 1
        return mask

    # 当步预占（静态站位资源）

    def has_chosen_action(self, site_id: int, agent_id: int):
        """
        提前标记在当前 decision step 中被 agent 选择的停机位，防止其它 agent
        在掩码判断阶段继续看到该站位空闲。site_id 是 self.sites 的索引。
        """
        if site_id < 0 or site_id >= len(self.sites):
            return
        site = self.sites[site_id]
        if site.is_runway:
            return
        self._reserve_stand_for_step(site.site_id, agent_id)

    # 关键逻辑：解析动作→ETA推进→到位/开工/完工→释放/干涉时钟
    # LOG:核心step()

    def _handle_move_action(self, pid: int, target_idx: int, allow_early_runway: bool = False) -> Tuple[bool, float]:
        plane: Plane = self.planes[pid]
        site_to: Site = self.sites[target_idx]
        penalty = 0.0

        if site_to.is_runway and (not allow_early_runway) and self._plane_has_support_left(plane):
            penalty += self.penalty_early_runway
            return False, penalty

        speed_m_per_min = self._speed_m_per_min()
        move_min = count_path_on_road(
            plane.position, site_to.absolute_position, speed_m_per_min)

        mv = None
        if site_to.is_runway:
            if "ZY_L" in plane.finished_codes:  # 去跑道需先解固
                mv = "ZY_T"
        else:
            if ("ZY_Z" in plane.finished_codes) and ("ZY01" not in plane.finished_codes):
                mv = "ZY_M"
            elif ("ZY_L" in plane.finished_codes) and ("ZY01" not in plane.finished_codes):
                mv = "ZY_T"

        plane.move_code = mv
        cur_sid = None
        if plane.current_site_id is not None and 0 <= plane.current_site_id < len(self.sites):
            cur_sid = self.sites[plane.current_site_id].site_id

        if mv is None or (cur_sid == site_to.site_id) or not (float(move_min) > 1e-9):
            return False, penalty

        if mv == "ZY_T":
            jobT = self.jobs_obj.get_job("ZY_T")
            from_site = self.sites[self.planes[pid].current_site_id] if self.planes[
                pid].current_site_id is not None else None
            if from_site is None:
                return False, penalty
            can, wait_min, handles = self._alloc_resources_for_job(
                jobT, from_site.site_id, plane)
            if not can:
                return False, penalty

            if not site_to.is_runway:
                self._reserve_incoming_stand(site_to.site_id, pid)
            if cur_sid is not None and cur_sid in self.stand_current_occupancy:
                self._release_stand(cur_sid, pid)

            tow_speed_m_per_min = self._speed_m_per_min()
            for h in handles or []:
                if isinstance(h, MobileDevice) and h.rtype == "R014":
                    tow_speed_m_per_min = float(h.speed_m_s) * 60.0
                    break
            tow_min = count_path_on_road(
                from_site.absolute_position, site_to.absolute_position, tow_speed_m_per_min)
            total_move = float(wait_min + tow_min)

            jid = self.jobs_obj.code2id()[mv]
            self.episodes_situation.append(
                (self.current_time, jid, site_to.site_id, pid, 0.0, total_move))
            fix_ids, mob_ids = [], []
            for h in handles or []:
                if isinstance(h, MobileDevice):
                    mob_ids.append(h.dev_id)
                elif isinstance(h, FixedDevice):
                    fix_ids.append(h.dev_id)
            self.episode_devices.append(
                {"FixedDevices": fix_ids, "MobileDevices": mob_ids})

            plane._move_handles = handles
            plane.start_move(
                to_site_id=site_to.site_id - 1, move_min=total_move)
            return True, penalty
        else:
            if cur_sid is not None and cur_sid in self.stand_current_occupancy:
                self._release_stand(cur_sid, pid)

            if not site_to.is_runway:
                self._reserve_incoming_stand(site_to.site_id, pid)
            jid = self.jobs_obj.code2id()[mv]
            self.episodes_situation.append(
                (self.current_time, jid, site_to.site_id, pid, 0.0, float(move_min)))
            self.episode_devices.append(
                {"FixedDevices": [], "MobileDevices": []})
            plane.start_move(
                to_site_id=site_to.site_id - 1, move_min=move_min)
            return True, penalty

    def step(self, actions: List[int]):
        prev_evt_len = len(self.episodes_situation)
        inst_reward = 0.0
        if self.enable_disturbance:
            self._process_disturbance_timeline()
            self._retry_forced_relocations()
        # (1) 自动落地：不交给 RL 决策
        just_landed = [False] * len(self.planes)
        for pid, plane in enumerate(self.planes):
            if "ZY_Z" in plane.finished_codes:
                continue
            arr_t, rw = self.arrival_plan.get(
                pid, (0.0, self.landing_runway_site_for_pos))
            if self.current_time >= arr_t and self.current_time >= self.landing_busy_until and plane.status in ("IDLE",):
                runway_idx = next(k for k, s in enumerate(self.sites)
                                  if s.site_id == self.landing_runway_site_for_pos)
                plane.current_site_id = runway_idx
                plane.position = self.sites[runway_idx].absolute_position.copy(
                )
                job = self.jobs_obj.get_job("ZY_Z")
                proc_min = self._proc_time_minutes(
                    job, plane, self.sites[runway_idx], self.sites[runway_idx])

                # 事件：site_id 用虚拟 0，避免被误解为 29/30/31
                jid = self.jobs_obj.code2id()["ZY_Z"]
                self.episodes_situation.append(
                    (self.current_time, jid, self.landing_virtual_site_id, pid, float(proc_min), 0.0))
                self.episode_devices.append(
                    {"FixedDevices": [], "MobileDevices": []})

                plane.start_job(job, proc_min)
                self.landing_busy_until = self.current_time + \
                    proc_min + self.landing_sep_min  # 落地占道+间隔
                just_landed[pid] = True

        # (2) 解析动作：开始移动时立即记“移动事件”
        for pid, act in enumerate(actions):
            plane: Plane = self.planes[pid]
            if "ZY_Z" not in plane.finished_codes or just_landed[pid]:
                continue  # 未落地/刚落地，本步不执行动作

            moved_auto, penalty_auto = self._auto_stage_if_ready(pid)
            inst_reward += penalty_auto
            if moved_auto:
                continue

            if act < len(self.sites):
                moved, penalty = self._handle_move_action(pid, act)
                inst_reward += penalty
                if moved:
                    continue
            # 其它动作（WAIT/BUSY/DONE）保持当前逻辑，无需额外处理

        # (3) 时间推进
        etas = []
        for p in self.planes:
            if p.status == "MOVING" and np.isfinite(p.eta_move_end) and p.eta_move_end > 0:
                etas.append(p.eta_move_end)
            if p.status == "PROCESSING" and np.isfinite(p.eta_proc_end) and p.eta_proc_end > 0:
                etas.append(p.eta_proc_end)
        delta_t = min(etas) if len(etas) > 0 else self.min_time_unit
        if self.enable_disturbance:
            delta_t = self._limit_step_to_disturbance(delta_t)
        self.last_dt = delta_t
        self.current_time += delta_t

        # (4) 到位&完工
        idle_candidates: List[int] = []
        for pid, plane in enumerate(self.planes):
            site_cur = None
            if plane.current_site_id is not None and 0 <= plane.current_site_id < len(self.sites):
                site_cur = self.sites[plane.current_site_id]
            site_pos = site_cur.absolute_position if site_cur is not None else None

            # 推进 ETA
            if plane.status == "MOVING":
                plane.eta_move_end = max(0.0, plane.eta_move_end - delta_t)
                if plane.eta_move_end == 0.0 and site_pos is not None:
                    plane.position = site_pos
                    plane.status = "IDLE"
                    # 释放牵引车等移动设备；把它们的位置更新为“目的站位”
                    if getattr(plane, "_move_handles", None):
                        self._release_resources(
                            plane._move_handles, plane, site_cur.site_id)
                        plane._move_handles = None
                    plane.move_code = None

                    # ★ 到达停机位时标记物理占位
                    if site_cur is not None and (not site_cur.is_runway):
                        self._occupy_stand(site_cur.site_id, pid)
                        self._release_incoming_stand(pid, site_cur.site_id)

            if plane.status == "PROCESSING":
                plane.eta_proc_end = max(0.0, plane.eta_proc_end - delta_t)
                if getattr(plane, "active_job_progress", None):
                    for code in list(plane.active_job_progress.keys()):
                        plane.active_job_progress[code] = max(
                            0.0, plane.active_job_progress[code] - delta_t)
                    if plane.active_job_progress:
                        plane.eta_proc_end = max(
                            plane.active_job_progress.values())

            if plane.status == "IDLE" and site_cur is not None:
                idle_candidates.append(pid)

            # 完工：仅做完工统一释放/收尾
            # LOG:完工逻辑
            if plane.status == "PROCESSING" and plane.eta_proc_end == 0.0:
                # 先释放设备（句柄可能不存在/为空，统一容错）
                cur_site_id = self.sites[plane.current_site_id].site_id if plane.current_site_id is not None else 1
                self._release_resources(
                    getattr(plane, "_last_handles", None), plane, cur_site_id)
                plane._handles_by_job = {}

                # 这一步完成的作业集合（并行 or 单作业）
                finished_codes = []
                if getattr(plane, "_active_jobs", None):
                    finished_codes = [j.code for j in plane._active_jobs]
                elif plane.current_job_code:
                    finished_codes = [plane.current_job_code]

                # 特殊逻辑 + 完成入账
                any_outbound = False
                for code in finished_codes:
                    # 特殊：解固撤销 ZY01
                    if code == "ZY_L" and "ZY01" in plane.finished_codes:
                        plane.finished_codes.discard("ZY01")
                    # 特殊：加油置满
                    if code == "ZY10":
                        plane.fuel_percent = 100.0

                    job_obj = self.jobs_obj.get_job(code)
                    if job_obj and getattr(job_obj, "group", "") == "出场":
                        any_outbound = True
                    plane.finish_job(job_obj)

                plane._active_jobs = None
                plane._last_handles = []  # 统一清空
                plane._handles_by_job = {}

                # 清占用&互锁时间
                if plane.current_site_id is not None:
                    sid = self.sites[plane.current_site_id].site_id
                    if any_outbound and sid in (29, 30, 31):
                        self.last_leave_time_by_runway[sid] = self.current_time

                # 长占清理
                if self.enable_long_occupy and ("ZY15" in finished_codes):
                    plane.long_occupy.clear()

                # 终态 or 继续
                if self.task_obj.graph.all_finished(plane.finished_codes):
                    plane.status = "DONE"
                else:
                    plane.status = "IDLE"

        if self.enable_disturbance:
            self._process_disturbance_timeline()
            self._retry_forced_relocations()

        for pid in idle_candidates:
            plane = self.planes[pid]
            if plane.status != "IDLE" or plane.current_site_id is None:
                continue
            site_cur = self.sites[plane.current_site_id]
            if site_cur.site_id in self.disturbance_blocked_stands:
                continue
            if self._try_resume_paused_jobs(pid, plane, site_cur):
                continue

            batch_ready = self._batch_support_done_for_plane(pid)
            ready_iter = self.task_obj.graph.enabled(plane.finished_codes, plane.ongoing_mutex) \
                if (self.enable_deps or self.enable_mutex) else self.task_obj.jobs.jobs_object_list[:1]
            ready = list(ready_iter)
            if site_cur.is_runway:
                if batch_ready:
                    ready += self._outbound_ready_for(plane)
                else:
                    ready = [j for j in ready if j.group != "出场"]
            site_cap_ids = set(site_cur.resource_ids_list)
            pack = self.task_obj.graph.pack_parallel(ready, site_cap_ids)
            pack = [j for j in pack if (
                j.group != "出场" or site_cur.is_runway)]

            accepted = []
            for j in pack:
                can, wait_min, handles = self._alloc_resources_for_job(
                    j, site_cur.site_id, plane)
                if can:
                    accepted.append((j, wait_min, handles))

            if not accepted:
                cand_iter = self.task_obj.graph.enabled(plane.finished_codes, plane.ongoing_mutex) \
                    if (self.enable_deps or self.enable_mutex) else self.task_obj.jobs.jobs_object_list[:1]
                cand_jobs = list(cand_iter)
                if site_cur.is_runway:
                    if batch_ready:
                        cand_jobs += self._outbound_ready_for(plane)
                    else:
                        cand_jobs = [j for j in cand_jobs if j.group != "出场"]
                job_for_site = None
                for j in cand_jobs:
                    j_id = self.jobs_obj.code2id()[j.code]
                    if j_id in site_cur.resource_ids_list:
                        job_for_site = j
                        break
                if job_for_site is not None and job_for_site.group == "出场" and (not site_cur.is_runway):
                    job_for_site = None

                if job_for_site is not None:
                    can, wait_min, handles = self._alloc_resources_for_job(
                        job_for_site, site_cur.site_id, plane)
                    if can:
                        proc_min = float(
                            wait_min + self._proc_time_minutes(job_for_site, plane, site_cur, site_cur))
                        if (not np.isfinite(proc_min)) or (proc_min < 0.0):
                            continue
                        jid = self.jobs_obj.code2id()[
                            job_for_site.code]
                        self.episodes_situation.append(
                            (self.current_time, jid, site_cur.site_id, pid, proc_min, 0.0))
                        fix_ids, mob_ids = [], []
                        for h in handles or []:
                            if isinstance(h, FixedDevice):
                                fix_ids.append(h.dev_id)
                            elif isinstance(h, MobileDevice):
                                mob_ids.append(h.dev_id)
                        self.episode_devices.append(
                            {"FixedDevices": fix_ids, "MobileDevices": mob_ids})

                        plane.start_job(job_for_site, proc_min)
                        plane._active_event_indices = [
                            (job_for_site.code, len(self.episodes_situation) - 1)]
                        if not site_cur.is_runway:
                            self._occupy_stand(site_cur.site_id, pid)
                        if self.enable_long_occupy and job_for_site.code in ("ZY03", "ZY02") and job_for_site.required_resources:
                            plane.long_occupy.add(
                                job_for_site.required_resources[0])
                        self._assign_handles_to_jobs(
                            plane, {job_for_site.code: handles or []})
            else:
                durations = []
                event_indices = []
                valid_jobs = []
                job_handle_map: Dict[str, List[object]] = {}
                for (j, wait_min, hlist) in accepted:
                    d = float(wait_min + self._proc_time_minutes(j,
                                                                 plane, site_cur, site_cur))
                    if (not np.isfinite(d)) or (d < 0.0):
                        continue
                    durations.append(d)
                    valid_jobs.append(j)
                    jid = self.jobs_obj.code2id()[j.code]
                    self.episodes_situation.append(
                        (self.current_time, jid, site_cur.site_id, pid, d, 0.0))

                    fix_ids, mob_ids = [], []
                    for h in (hlist or []):
                        if isinstance(h, FixedDevice):
                            fix_ids.append(h.dev_id)
                        elif isinstance(h, MobileDevice):
                            mob_ids.append(h.dev_id)
                    self.episode_devices.append(
                        {"FixedDevices": fix_ids, "MobileDevices": mob_ids})
                    job_handle_map[j.code] = list(hlist or [])
                    event_indices.append(
                        (j.code, len(self.episodes_situation) - 1))

                if durations:
                    plane._active_jobs = valid_jobs
                    self._assign_handles_to_jobs(plane, job_handle_map)
                    plane.current_job_code = "+".join(
                        [j.code for j in plane._active_jobs])
                    plane.eta_proc_end = max(durations)
                    plane.status = "PROCESSING"
                    plane.active_job_progress = {
                        job.code: dur for job, dur in zip(valid_jobs, durations)}
                    plane._job_total_durations = plane.active_job_progress.copy()
                    plane._active_event_indices = event_indices
                    if not site_cur.is_runway:
                        self._occupy_stand(site_cur.site_id, pid)
                    inst_reward += self.bonus_parallel_per_job * \
                        (len(accepted) - 1)

        # (5) 奖励/终止（保持你当前逻辑，略）
        step_time_cost = getattr(self, "last_dt", self.min_time_unit)
        if (not np.isfinite(step_time_cost)) or (step_time_cost <= 0.0):
            step_time_cost = self.min_time_unit
        reward = -0.1 * step_time_cost

        new_events = self.episodes_situation[prev_evt_len:]
        for (_, job_id, _, _, _, _) in new_events:
            code = self.jobs_obj.id2code()[job_id]
            if code == "ZY_F":
                reward += 10.0
            elif code not in ("ZY_M", "ZY_T"):
                reward += 1.0
        terminated = all(p.status == "DONE" for p in self.planes)
        if terminated:
            makespan = self.current_time
            reward += max(0.0, 200.0 - makespan)
        reward += inst_reward

        # NEW: 把设备并行数组也放进 info
        info = {
            "episodes_situation": self.episodes_situation,
            "devices_situation": self.episode_devices,
            "time": self.current_time
        }
        if self.enable_disturbance:
            info["disturbance"] = {
                "active_stands": sorted(self.disturbance_blocked_stands),
                "blocked_resources": sorted(self.disturbance_blocked_resource_types),
                "down_devices": sorted(self.disturbance_blocked_device_ids),
                "history": copy.deepcopy(self.disturbance_history)
            }

        for site, idx in self._solve_reserved:
            site.resource_number[idx] += 1
        self._solve_reserved.clear()
        self._reserved_stands.clear()
        return reward, terminated, info
    # 扰动注入

    def apply_disturbance(self, event: Dict):
        tp = event.get("type")
        if tp == "site_down":
            s = next(s for s in self.sites if s.site_id == event["site_id"])
            s.unavailable_windows.append((event["start"], event["end"]))
        elif tp == "capacity_drop":
            s = next(s for s in self.sites if s.site_id == event["site_id"])
            for idx in range(len(s.resource_number)):
                s.resource_number[idx] = max(
                    0, s.resource_number[idx] + int(event.get("delta", -1)))
        elif tp == "proc_scale":
            code = event["job_code"]
            scale = float(event.get("scale", 1.0))
            for j in self.jobs_obj.jobs_object_list:
                if j.code == code and j.time_span > 0:
                    j.time_span *= scale
        elif tp == "device_fail":
            dev_id = event["dev_id"]
            self._set_device_down(dev_id, True)
            self.disturbance_blocked_device_ids.add(dev_id.upper())
    
