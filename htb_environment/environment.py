# environment.py
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set

from utils.util import count_path_on_road
from utils.job import Jobs, Job
from utils.task import Task
from utils.plane import Planes, Plane
from utils.site import Sites, Site

# 固定/移动设备


@dataclass
class FixedDevice:
    dev_id: str
    rtype: str
    cover_stands: Set[int]
    # FIXME:按照勘误修改为2 
    capacity: int = 2
    in_use: Set[int] = None

    def __post_init__(self):
        if self.in_use is None:
            self.in_use = set()

    def can_serve(self, stand_id: int) -> bool:
        return (stand_id in self.cover_stands) and (len(self.in_use) < self.capacity)


@dataclass
class MobileDevice:
    dev_id: str
    rtype: str
    loc_stand: int
    speed_m_s: float = 3.0
    busy_until_min: float = 0.0
    locked_by: Optional[int] = None

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

        # 干涉表（技术资料）
        self.inbound_block: Dict[int, Tuple[int, float]] = {
            2: (3, 2), 6: (5, 2), 8: (9, 2), 15: (16, 2), 18: (17, 2), 27: (26, 2)}
        self.runway_block: Dict[int, List[Tuple[int, float]]] = {
            10: [(29, 2), (30, 2)], 11: [(29, 2), (30, 2)], 12: [(29, 2), (30, 2)],
            13: [(29, 2), (30, 2)], 14: [(29, 2), (30, 2)], 23: [(30, 2), (31, 2)],
            24: [(30, 2), (31, 2)], 29: [(30, 0.5)], 30: [(29, 0.5)]
        }
        self.last_leave_time_by_stand: Dict[int, float] = {}
        self.last_leave_time_by_runway: Dict[int, float] = {
            29: -1e9, 30: -1e9, 31: -1e9}
        
        self.arrival_gap_min = getattr(self.args, "arrival_gap_min", 0.5)
        self.arrival_plan: Dict[int, Tuple[float, int]] = {}  # plane_id -> (eta_min, stand_id)

        # 设备池 & 站位占用
        self.fixed_devices: List[FixedDevice] = []
        self.mobile_devices: List[MobileDevice] = []
        self.stand_current_occupancy: Dict[int, Optional[int]] = {}

        # 飞机群
        self.planes_obj: Optional[Planes] = None
        self.planes: List[Plane] = []

        # MARL接口所需
        self.n_agents = 0
        self.n_actions = len(self.sites) + 3  # 所有站位/跑道 + WAIT + BUSY + DONE
        self.episode_limit = 1000  # 可按需要调整
        self._solve_reserved = []

        # 记录（时刻, job_id, site_id, plane_id, proc_min, move_min）
        self.episodes_situation: List[Tuple[float,
                                            int, int, int, float, float]] = []
        self.episode_devices: List[Dict[str, list]] = []

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

    # ADD:设备池
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

    # def _alloc_resources_for_job(self, job: Job, stand_id: int, plane: Plane) -> Tuple[bool, float, List[object]]:
    #     need = job.required_resources
    #     if not need:
    #         return True, 0.0, []
    #     selected, extra_wait = [], 0.0
    #     # 固定设备优先
    #     for rt in need:
    #         fix = next((d for d in self.fixed_devices if d.rtype ==
    #                    rt and d.can_serve(stand_id)), None)
    #         selected.append(fix)
    #     if all(h is not None for h in selected):
    #         for h in selected:
    #             h.in_use.add(plane.plane_id)
    #         return True, 0.0, selected
    #     # 移动设备最近可达
    #     site_pos = self.sites[stand_id -
    #                           1].absolute_position if 1 <= stand_id <= 31 else self.sites[stand_id].absolute_position
    #     for idx, rt in enumerate(need):
    #         if selected[idx] is not None:
    #             continue
    #         cands = [m for m in self.mobile_devices if m.rtype ==
    #                  rt and m.locked_by is None]
    #         if not cands:
    #             return False, 0.0, []
    #         best, best_eta = None, 1e9
    #         for m in cands:
    #             from_pos = self.sites[m.loc_stand -
    #                                   1].absolute_position if 1 <= m.loc_stand <= 31 else site_pos
    #             eta = m.eta_to_min(from_pos, site_pos, self.current_time)
    #             if eta < best_eta:
    #                 best_eta, best = eta, m
    #         extra_wait = max(extra_wait, best_eta - self.current_time)
    #         best.locked_by = plane.plane_id
    #         selected[idx] = best
    #     return True, extra_wait, selected

    def _alloc_resources_for_job(self, job: Job, stand_id: int, plane: Plane) -> Tuple[bool, float, List[object]]:
        """为即将开工的 job 挑设备：
        - 固定设备能立刻用 → 直接占
        - 否则比较：移动设备到达时间 vs 固定设备最早可用时间
        - 移动更快 → 锁一台移动设备，extra_wait 为到达等待
        - 固定更快 → 返回 False（暂不执行，回合推进后再尝试）
        说明：之所以不“预占未来的固定设备”，是为了避免把产能提前锁死导致全局阻塞。
        """
        need = job.required_resources
        if not need:
            return True, 0.0, []

        # ———— 第一遍：只做“规划”，不真正占用 ————
        # (kind, handle/rt, extra_eta_abs)
        plans: List[Tuple[str, object, float]] = []
        now = self.current_time
        site_pos = self.sites[stand_id -
                            1].absolute_position if 1 <= stand_id <= 31 else self.sites[stand_id].absolute_position
        extra_wait = 0.0

        for rt in need:
            # 1) 固定设备最早可用时间
            eta_fix = self._eta_fixed_available(rt, stand_id)

            # 2) 移动设备最早到达时间
            best_m, eta_mob = None, float("inf")
            for m in self.mobile_devices:
                if m.rtype == rt and m.locked_by is None:
                    from_pos = self.sites[m.loc_stand - 1].absolute_position
                    eta = m.eta_to_min(from_pos, site_pos, now)  # 已含 busy_until
                    if eta < eta_mob:
                        eta_mob, best_m = eta, m

            # 3) 决策
            if eta_fix <= now:
                plans.append(("fixed_now", rt, now))
            elif best_m is not None and eta_mob <= eta_fix:
                plans.append(("mobile", best_m, eta_mob))
                extra_wait = max(extra_wait, eta_mob - now)
            else:
                # 固定更快，但不是“立刻”可用；为了不提前锁产能，这里选择“暂不执行”
                return False, 0.0, []

        # ———— 第二遍：执行“占用/锁定” ————
        handles: List[object] = []
        for kind, obj, _ in plans:
            if kind == "fixed_now":
                rt = obj  # 这里 obj 是 rtype
                # 在可用池里找一个真的“有空位”的设备并占用
                pool = next(d for d in self.fixed_devices if d.rtype ==
                            rt and stand_id in d.cover_stands and len(d.in_use) < d.capacity)
                pool.in_use.add(plane.plane_id)
                handles.append(pool)
            else:
                # 锁定移动设备
                m: MobileDevice = obj
                m.locked_by = plane.plane_id
                handles.append(m)

        return True, extra_wait, handles

    def _release_resources(self, handles: List[object], plane: Plane, stand_id: int):
        for h in handles or []:
            if isinstance(h, FixedDevice):
                h.in_use.discard(plane.plane_id)
            elif isinstance(h, MobileDevice):
                h.busy_until_min = self.current_time
                h.locked_by = None
                h.loc_stand = stand_id

    # ============ 先验 & obs/state 预留 ============
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

        # 站位占用与干涉时钟
        self.stand_current_occupancy = {sid: None for sid in range(1, 29)}
        self.last_leave_time_by_stand = {sid: -1e9 for sid in range(1, 29)}
        self.last_leave_time_by_runway = {29: -1e9, 30: -1e9, 31: -1e9}

        # 设备池
        self._build_fixed_devices()
        self._build_mobile_devices()

        # 时间与轨迹
        self.current_time = 0.0
        self.episodes_situation = []
        self.episode_devices = []

        # 更新动作数
        self.n_actions = len(self.sites) + 3
        self.arrival_plan.clear()
        for pid in range(n_agents):
            t_arr = pid * self.arrival_gap_min
            rw = 29 + (pid % 3)  # 29/30/31
            self.arrival_plan[pid] = (t_arr, rw)

        # 清空已完成集，飞机不在任何站位（表示“在空中等待落地”）
        for p in self.planes:
            p.finished_codes.clear()
            p.current_site_id = None
            p.position = np.array([0.0, 0.0], dtype=float)  # 仅占位，无实际意义
            p.status = "IDLE"  # 仍用 IDLE，但通过到达计划限制动作
            # 返回初始观测（若 MARL 框架不需要，可忽略）
        return self.get_obs()


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

        base_state = np.concatenate([plane_feats, occ, t], -1).astype(np.float32)
        return self._pad_state_tail(base_state)

    
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
        cand_jobs: List[Job] = self.task_obj.graph.enabled(plane.finished_codes, plane.ongoing_mutex) \
            if (self.enable_deps or self.enable_mutex) else self.task_obj.jobs.jobs_object_list[:1]
        cand_jobs = [j for j in cand_jobs if j.code != "ZY_Z"]

        # 批次离场门控
        if not self._batch_ready_for_takeoff():
            cand_jobs = [j for j in cand_jobs if j.group != "出场"]
        if len(cand_jobs) == 0:
            mask[-3] = 1
            return mask

        # 站位枚举（加入“被占用则不可选”的过滤）
        for idx, site in enumerate(self.sites):
            if not site.is_runway:
                # 站位已被占用，则跳过（容量>1 时请改为“容量计数”）
                if self.stand_current_occupancy.get(site.site_id, None) is not None:
                    continue
            if not site.is_available(self.current_time):
                continue

            ok = False
            for j in cand_jobs:
                j_id = self.jobs_obj.code2id()[j.code]
                if site.is_runway:
                    # 跑道只用于出场组（且满足互锁），ZY_Z 已移交 step() 自动触发
                    if j.group == "出场" and self._runway_available_outbound(site.site_id):
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
            if ok:
                mask[idx] = 1

        if mask[:-3].sum() == 0:
            mask[-3] = 1
        return mask



    # 当步预占（静态站位资源）
    def has_chosen_action(self, site_id: int, agent_id: int):
        site = next(s for s in self.sites if s.site_id ==
                    (site_id if site_id >= 1 else site_id+1))
        for idx, cnt in enumerate(site.resource_number):
            if cnt > 0:
                site.resource_number[idx] -= 1
                self._solve_reserved.append((site, idx))
                break

    
    # def step(self, actions: List[int]):
    #     prev_evt_len = len(self.episodes_situation)

    #     # 1) 先处理自动落地（外生事件），且本步落地后不再处理该机的动作
    #     just_landed = [False] * len(self.planes)
    #     for pid, plane in enumerate(self.planes):
    #         if "ZY_Z" in plane.finished_codes:
    #             continue
    #         arr_t, rw = self.arrival_plan.get(pid, (0.0, 29))
    #         if self.current_time >= arr_t and plane.status in ("IDLE",):
    #             runway_idx = next(k for k, s in enumerate(
    #                 self.sites) if s.site_id == rw)
    #             plane.current_site_id = runway_idx
    #             plane.position = self.sites[runway_idx].absolute_position.copy()
    #             job = self.jobs_obj.get_job("ZY_Z")
    #             proc_min = self._proc_time_minutes(
    #                 job, plane, self.sites[runway_idx], self.sites[runway_idx])
    #             plane.start_job(job, proc_min)
    #             just_landed[pid] = True
    #             # 注意：不在落地开始时写 last_leave_time_by_runway，释放时再写

    #     # 2) 解析动作（仅对“已落地且未在本步刚落地”的飞机）
    #     for pid, act in enumerate(actions):
    #         plane: Plane = self.planes[pid]
    #         if "ZY_Z" not in plane.finished_codes or just_landed[pid]:
    #             continue  # 未落地或本步刚触发落地：本步不执行动作

    #         if act < len(self.sites):
    #             site_to: Site = self.sites[act]
    #             speed_m_per_min = self._speed_m_per_min()
    #             move_min = count_path_on_road(
    #                 plane.position, site_to.absolute_position, speed_m_per_min)

    #             mv = None
    #             if site_to.is_runway:
    #                 # 出场前应已解固（ZY_L），去跑道视作转运 ZY_T（牵引）
    #                 if "ZY_L" in plane.finished_codes:
    #                     mv = "ZY_T"
    #             else:
    #                 # 进场滑行/站间转运
    #                 if ("ZY_Z" in plane.finished_codes) and ("ZY01" not in plane.finished_codes):
    #                     mv = "ZY_M"
    #                 elif ("ZY_L" in plane.finished_codes) and ("ZY01" not in plane.finished_codes):
    #                     mv = "ZY_T"

    #             plane.move_code = mv
    #             plane.start_move(to_site_id=site_to.site_id - 1, move_min=move_min)
    #         else:
    #             pass

    #     # 2) 推进
    #     etas = []
    #     for p in self.planes:
    #         if p.status == "MOVING" and p.eta_move_end > 0:
    #             etas.append(p.eta_move_end)
    #         if p.status == "PROCESSING" and p.eta_proc_end > 0:
    #             etas.append(p.eta_proc_end)
    #     delta_t = min(etas) if etas else self.min_time_unit
    #     self.last_dt = delta_t
    #     self.current_time += delta_t

    #     # 3) 到位&完工
    #     for pid, plane in enumerate(self.planes):
    #         site_cur = None
    #         if plane.current_site_id is not None and 0 <= plane.current_site_id < len(self.sites):
    #             site_cur = self.sites[plane.current_site_id]
    #         site_pos = site_cur.absolute_position if site_cur is not None else None

    #         # 推进 ETA
    #         if plane.status == "MOVING":
    #             plane.eta_move_end = max(0.0, plane.eta_move_end - delta_t)
    #             if plane.eta_move_end == 0.0 and site_pos is not None:
    #                 plane.position = site_pos

    #         if plane.status == "PROCESSING":
    #             plane.eta_proc_end = max(0.0, plane.eta_proc_end - delta_t)


    #         # FIXME到位：尝试开工
    #         # 到位：尝试开工（已在你的代码中定位到此块）
    #         if plane.status == "MOVING" and plane.eta_move_end == 0.0 and site_cur is not None:
    #             # 若打过移动标签，先记一条“移动事件”
    #             if plane.move_code in ("ZY_M", "ZY_T"):
    #                 jid = self.jobs_obj.code2id()[plane.move_code]
    #                 self.episodes_situation.append((
    #                     self.current_time, jid, site_cur.site_id, pid,
    #                     0.0, float(plane.move_last_min)
    #                 ))
    #                 self.episode_devices.append({})
    #                 plane.finished_codes.add(plane.move_code)
    #             plane.move_code = None
    #             plane.status = "IDLE"

    #             # 【新增】二次占用拦截：若该站位刚被其他飞机抢占，则本步不再开工（避免同刻多机）
    #             if (not site_cur.is_runway) and (self.stand_current_occupancy.get(site_cur.site_id) is not None):
    #                 # 留在 IDLE，下一步再尝试
    #                 continue

    #             # 之后再走“挑作业+资源分配”的逻辑（你已有）
    #             cand_jobs = self.task_obj.graph.enabled(plane.finished_codes, plane.ongoing_mutex) \
    #                 if (self.enable_deps or self.enable_mutex) else self.task_obj.jobs.jobs_object_list[:1]
    #             job_for_site = None
    #             for j in cand_jobs:
    #                 j_id = self.jobs_obj.code2id()[j.code]
    #                 if j_id in site_cur.resource_ids_list:
    #                     job_for_site = j
    #                     break
    #             if job_for_site is None:
    #                 plane.status = "IDLE"
    #             else:
    #                 can, wait_min, handles = self._alloc_resources_for_job(
    #                     job_for_site, site_cur.site_id, plane)
    #                 if not can:
    #                     plane.status = "IDLE"
    #                 else:
    #                     proc_min = self._proc_time_minutes(
    #                         job_for_site, plane, site_cur, site_cur)
    #                     proc_min = float(wait_min + proc_min)
    #                     plane.start_job(job_for_site, proc_min)
    #                     # 占用站位（仅停机位）
    #                     self.stand_current_occupancy[site_cur.site_id] = pid
    #                     if self.enable_long_occupy and job_for_site.code in ("ZY03", "ZY02"):
    #                         if job_for_site.required_resources:
    #                             plane.long_occupy.add(job_for_site.required_resources[0])
    #                     plane._last_handles = handles


    #         # 完工：释放
    #         # environment.py -- 在完工释放处替换为如下结构（只示意关键几行）
    #         if plane.status == "PROCESSING" and plane.eta_proc_end == 0.0:
    #             job_code = plane.current_job_code
    #             if job_code is None:
    #                 # 防御：如果出现这种情况，直接退回空闲，避免 get_job(None)
    #                 plane.status = "IDLE"
    #                 plane.eta_proc_end = 0.0
    #                 continue

    #             job = self.jobs_obj.get_job(job_code)
    #             if job.code == "ZY_L" and "ZY01" in plane.finished_codes:
    #                 plane.finished_codes.discard("ZY01")
    #             # 加油置满
    #             if job.code == "ZY10":
    #                 plane.fuel_percent = 100.0

    #             job_id = self.jobs_obj.code2id()[job.code]
    #             move_min = getattr(plane, "move_last_min", 0.0)
    #             # self.episodes_situation.append((self.current_time, job_id, plane.current_site_id, pid,
    #             #                                 float(max(job.time_span, 0.0)
    #             #                                     if job.time_span > 0 else 0.0),
    #             #                                 float(move_min)))
    #             # FIXME
    #             proc_min = float(getattr(plane, "proc_last_min", 0.0))
    #             code = job.code
    #             # environment.py :: 完工记录处（去掉 ZY_Z 特判）
    #             site_id_for_record = (
    #                 self.sites[plane.current_site_id].site_id
    #                 if plane.current_site_id is not None else 1
    #             )
    #             self.episodes_situation.append((
    #                 self.current_time, job_id, site_id_for_record, pid,
    #                 proc_min, 0.0
    #             ))

    #             plane.proc_last_min = 0.0
    #             plane.move_last_min = 0.0

    #             # 释放设备
    #             self._release_resources(getattr(plane, "_last_handles", None), plane,
    #                                     plane.current_site_id+1 if plane.current_site_id is not None else 1)
    #             plane._last_handles = None

    #             # 清占用与离位（停机位才在占用表）
    #             if plane.current_site_id is not None:
    #                 sid = self.sites[plane.current_site_id].site_id
    #                 if sid in self.stand_current_occupancy:     # 跑道(29..31)跳过
    #                     self.stand_current_occupancy[sid] = None
    #                     self.last_leave_time_by_stand[sid] = self.current_time
    #                 if job.group == "出场" and sid in (29, 30, 31):
    #                     self.last_leave_time_by_runway[sid] = self.current_time

    #             # 完成作业（plane.finish_job 现已把状态设回 IDLE）
    #             plane.finish_job(job)
    #             if self.enable_long_occupy and job.code == "ZY15":
    #                 plane.long_occupy.clear()

    #             # 若该机已完成全流程，置 DONE（否则保持 IDLE）
    #             if self.task_obj.graph.all_finished(plane.finished_codes):
    #                 plane.status = "DONE"


    #     # 4) 奖励与终止（此处只返回0；终局奖励在外部评估）
    #     step_time_cost = getattr(self, "last_dt", self.min_time_unit)
    #     reward = -0.1*step_time_cost
    #     # 只统计本步新追加的事件
    #     new_events = self.episodes_situation[prev_evt_len:]
    #     for (_, job_id, _, _, _, _) in new_events:
    #         code = self.jobs_obj.id2code()[job_id]
    #         if code == "ZY_F":
    #             reward += 10.0
    #         elif code not in ("ZY_M", "ZY_T"):   # 移动不加分，防止“移动完成=作业完成”膨胀
    #             reward += 1.0

    #     terminated = all(p.status == "DONE" for p in self.planes)
    #     if terminated:
    #         makespan = self.current_time
    #         reward += max(0.0, 200.0 - makespan)

    #     info = {"episodes_situation": self.episodes_situation, "time": self.current_time}
    #     for site, idx in self._solve_reserved:
    #         site.resource_number[idx] += 1
    #     self._solve_reserved.clear()
    #     return reward, terminated, info
    
    
    # 关键逻辑：解析动作→ETA推进→到位/开工/完工→释放/干涉时钟
    # FIXME
    def step(self, actions: List[int]):
        prev_evt_len = len(self.episodes_situation)

        # (1) 自动落地：不交给 RL 决策
        just_landed = [False] * len(self.planes)
        for pid, plane in enumerate(self.planes):
            if "ZY_Z" in plane.finished_codes:
                continue
            arr_t, rw = self.arrival_plan.get(pid, (0.0, 29))
            if self.current_time >= arr_t and plane.status in ("IDLE",):
                runway_idx = next(k for k, s in enumerate(
                    self.sites) if s.site_id == rw)
                plane.current_site_id = runway_idx
                plane.position = self.sites[runway_idx].absolute_position.copy(
                )
                job = self.jobs_obj.get_job("ZY_Z")
                proc_min = self._proc_time_minutes(
                    job, plane, self.sites[runway_idx], self.sites[runway_idx])

                # NEW: 在“开始加工”时记事件（ZY_Z）
                jid = self.jobs_obj.code2id()["ZY_Z"]
                self.episodes_situation.append(
                    (self.current_time, jid, rw, pid, float(proc_min), 0.0))
                self.episode_devices.append(
                    {"FixedDevices": [], "MobileDevices": []})

                plane.start_job(job, proc_min)
                just_landed[pid] = True

        # (2) 解析动作：开始移动时立即记“移动事件”
        for pid, act in enumerate(actions):
            plane: Plane = self.planes[pid]
            if "ZY_Z" not in plane.finished_codes or just_landed[pid]:
                continue  # 未落地/刚落地，本步不执行动作

            if act < len(self.sites):
                site_to: Site = self.sites[act]
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
                if mv is not None:
                    jid = self.jobs_obj.code2id()[mv]
                    # NEW: 移动在开始时刻入账，proc=0，move=move_min
                    self.episodes_situation.append(
                        (self.current_time, jid, site_to.site_id, pid, 0.0, float(move_min)))
                    self.episode_devices.append(
                        {"FixedDevices": [], "MobileDevices": []})

                plane.start_move(to_site_id=site_to.site_id -
                                 1, move_min=move_min)
            else:
                pass

        # (3) 时间推进
        etas = []
        for p in self.planes:
            if p.status == "MOVING" and p.eta_move_end > 0:
                etas.append(p.eta_move_end)
            if p.status == "PROCESSING" and p.eta_proc_end > 0:
                etas.append(p.eta_proc_end)
        delta_t = min(etas) if etas else self.min_time_unit
        self.last_dt = delta_t
        self.current_time += delta_t

        # (4) 到位&完工
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
                    plane.status = "IDLE"          # NEW: 到位回到 IDLE
                    plane.move_code = None

            if plane.status == "PROCESSING":
                plane.eta_proc_end = max(0.0, plane.eta_proc_end - delta_t)

            # NEW：到位后“尝试开工”，在“开始加工时刻”记事件 + 占位 + 设备绑定
            if plane.status == "IDLE" and site_cur is not None:
                # 若该站位刚被别人占了，本步不再开工（避免同刻多机）
                if (not site_cur.is_runway) and (self.stand_current_occupancy.get(site_cur.site_id) is not None):
                    continue

                cand_jobs = self.task_obj.graph.enabled(plane.finished_codes, plane.ongoing_mutex) \
                    if (self.enable_deps or self.enable_mutex) else self.task_obj.jobs.jobs_object_list[:1]
                # 挑一个此站位能做的
                job_for_site = None
                for j in cand_jobs:
                    j_id = self.jobs_obj.code2id()[j.code]
                    if j_id in site_cur.resource_ids_list:
                        job_for_site = j
                        break

                if job_for_site is not None:
                    can, wait_min, handles = self._alloc_resources_for_job(
                        job_for_site, site_cur.site_id, plane)
                    if can:
                        proc_min = float(
                            wait_min + self._proc_time_minutes(job_for_site, plane, site_cur, site_cur))
                        # NEW: 在“开始加工时刻”记事件（proc>0, move=0），并附设备
                        jid = self.jobs_obj.code2id()[job_for_site.code]
                        self.episodes_situation.append(
                            (self.current_time, jid, site_cur.site_id, pid, proc_min, 0.0))
                        fix_ids = []
                        mob_ids = []
                        for h in handles or []:
                            if isinstance(h, FixedDevice):
                                fix_ids.append(h.dev_id)     # 保留字串 ID（如 FR1）
                            elif isinstance(h, MobileDevice):
                                mob_ids.append(h.dev_id)
                        self.episode_devices.append(
                            {"FixedDevices": fix_ids, "MobileDevices": mob_ids})

                        plane.start_job(job_for_site, proc_min)
                        # 占位（停机位）
                        if not site_cur.is_runway:
                            self.stand_current_occupancy[site_cur.site_id] = pid
                        if self.enable_long_occupy and job_for_site.code in ("ZY03", "ZY02") and job_for_site.required_resources:
                            plane.long_occupy.add(
                                job_for_site.required_resources[0])
                        plane._last_handles = handles

            # 完工：仅做释放与状态清理（不再记事件，事件已在“开始时”记过）
            if plane.status == "PROCESSING" and plane.eta_proc_end == 0.0:
                job_code = plane.current_job_code
                if job_code is None:
                    plane.status = "IDLE"
                    continue
                job = self.jobs_obj.get_job(job_code)
                if job.code == "ZY_L" and "ZY01" in plane.finished_codes:
                    plane.finished_codes.discard("ZY01")
                if job.code == "ZY10":
                    plane.fuel_percent = 100.0

                # 释放设备
                self._release_resources(getattr(plane, "_last_handles", None), plane,
                                        plane.current_site_id+1 if plane.current_site_id is not None else 1)
                plane._last_handles = None

                # 清占用&互锁时间
                if plane.current_site_id is not None:
                    sid = self.sites[plane.current_site_id].site_id
                    if sid in self.stand_current_occupancy:
                        self.stand_current_occupancy[sid] = None
                        self.last_leave_time_by_stand[sid] = self.current_time
                    if job.group == "出场" and sid in (29, 30, 31):
                        self.last_leave_time_by_runway[sid] = self.current_time

                plane.finish_job(job)
                if self.enable_long_occupy and job.code == "ZY15":
                    plane.long_occupy.clear()
                if self.task_obj.graph.all_finished(plane.finished_codes):
                    plane.status = "DONE"

        # (5) 奖励/终止（保持你当前逻辑，略）
        step_time_cost = getattr(self, "last_dt", self.min_time_unit)
        reward = -0.1*step_time_cost
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

        # NEW: 把设备并行数组也放进 info
        info = {
            "episodes_situation": self.episodes_situation,
            "devices_situation": self.episode_devices,
            "time": self.current_time
        }

        for site, idx in self._solve_reserved:
            site.resource_number[idx] += 1
        self._solve_reserved.clear()
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
            for d in self.fixed_devices:
                if d.dev_id == dev_id:
                    d.capacity = 0
