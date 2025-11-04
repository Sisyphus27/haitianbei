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
        # FIXME:暂时用29号跑道作为着陆点
        self.landing_runway_site_for_pos = 29              # 用 29 的坐标来放置飞机位置即可
        # 记录到日志时用“虚拟ID=0”，避免被当成 29/30/31
        self.landing_virtual_site_id = 0
        self.landing_sep_min = getattr(self.args, "landing_sep_min", 0.5)
        self.landing_busy_until = -1e9

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

        # 奖励参数
        self.penalty_early_runway = getattr(self.args, "penalty_early_runway", -2.0)    # NEW
        self.bonus_parallel_per_job = getattr(self.args, "bonus_parallel_per_job", 0.5) # NEW


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
    
    def _plane_support_done(self, p) -> bool:
        # “保障组”全部完成 = 不再有 group=='保障' 的作业未完成
        for j in self.jobs_obj.jobs_object_list:
            if getattr(j, "group", "") == "保障" and j.code not in p.finished_codes:
                return False
        return True

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

        plans: List[Tuple[str, object, float]] = []
        now = self.current_time
        site_pos = self.sites[stand_id - 1].absolute_position
        extra_wait = 0.0

        # —— 第一步：规划（杜绝 ∞ 等待） —— #
        for rt in need:
            eta_fix = self._eta_fixed_available(rt, stand_id)

            best_m, eta_mob = None, float("inf")
            for m in self.mobile_devices:
                if m.rtype == rt and m.locked_by is None:
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
                        if m.rtype == rt and m.locked_by is None:
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
                        if mm.rtype == m.rtype and mm.locked_by is None:
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
        self.current_time = 0.0
        self.episodes_situation = []
        self.episode_devices = []


        # 更新动作数
        self.arrival_plan.clear()
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

        # 动态注入出场作业（只允许在跑道位被选择）
        cand_jobs = list(cand_jobs) + self._outbound_ready_for(plane)

        cand_jobs = [j for j in cand_jobs if j.code != "ZY_Z"]

        # 批次离场门控
        if not self._batch_support_done():
            cand_jobs = [j for j in cand_jobs if (j.group != "出场")]
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
                    if (j.code in ("ZY_S", "ZY_F")) and self._runway_available_outbound(site.site_id):
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

    
    
    # 关键逻辑：解析动作→ETA推进→到位/开工/完工→释放/干涉时钟
    # LOG:核心step()
    def step(self, actions: List[int]):
        prev_evt_len = len(self.episodes_situation)
        inst_reward = 0.0
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
                plane.position = self.sites[runway_idx].absolute_position.copy()
                job = self.jobs_obj.get_job("ZY_Z")
                proc_min = self._proc_time_minutes(
                    job, plane, self.sites[runway_idx], self.sites[runway_idx])

                # 事件：site_id 用虚拟 0，避免被误解为 29/30/31
                jid = self.jobs_obj.code2id()["ZY_Z"]
                self.episodes_situation.append(
                    (self.current_time, jid, self.landing_virtual_site_id, pid, float(proc_min), 0.0))
                self.episode_devices.append({"FixedDevices": [], "MobileDevices": []})

                plane.start_job(job, proc_min)
                self.landing_busy_until = self.current_time + \
                    proc_min + self.landing_sep_min  # 落地占道+间隔
                just_landed[pid] = True


        # (2) 解析动作：开始移动时立即记“移动事件”
        for pid, act in enumerate(actions):
            plane: Plane = self.planes[pid]
            if "ZY_Z" not in plane.finished_codes or just_landed[pid]:
                continue  # 未落地/刚落地，本步不执行动作

            if act < len(self.sites):
                site_to: Site = self.sites[act]

                def _has_support_left(p):
                    # 支持：除 ZY_S/ZY_F/ZY_Z/ZY_L 之外的“保障组”是否还有未完成
                    for j in self.jobs_obj.jobs_object_list:
                        if j.group == "保障" and j.code not in p.finished_codes:
                            return True
                    return False
                if site_to.is_runway and (_has_support_left(plane) or (not self._batch_support_done())):
                    inst_reward += self.penalty_early_runway   # 惩罚
                    continue
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
                if mv is not None and (cur_sid != site_to.site_id) and (float(move_min) > 1e-9):
                    if mv == "ZY_T":
                        jobT = self.jobs_obj.get_job("ZY_T")
                        # 牵引车先到“当前站位”再牵引到“目标站位”
                        from_site = self.sites[self.planes[pid].current_site_id] if self.planes[pid].current_site_id is not None else None
                        if from_site is None:
                            # 没有明确的 from_site（极端情况），退化为不可执行
                            continue
                        # 为 ZY_T 分配设备（会返回等待牵引车到位的 extra_wait）
                        can, wait_min, handles = self._alloc_resources_for_job(
                            jobT, from_site.site_id, plane)
                        if not can:
                            # 分配不到牵引车，取消本步移动
                            continue
                        # 牵引“行驶时间”用移动设备速度估计：取被锁定的 R014 中的第一台
                        tow_speed_m_per_min = self._speed_m_per_min()
                        for h in handles or []:
                            if isinstance(h, MobileDevice) and h.rtype == "R014":
                                tow_speed_m_per_min = float(h.speed_m_s) * 60.0
                                break
                        tow_min = count_path_on_road(
                            from_site.absolute_position, site_to.absolute_position, tow_speed_m_per_min)
                        total_move = float(wait_min + tow_min)

                        # 记录移动事件（move_min=等待+牵引）
                        jid = self.jobs_obj.code2id()[mv]
                        self.episodes_situation.append(
                            (self.current_time, jid, site_to.site_id, pid, 0.0, total_move))
                        # 记录设备ID（方便你在 plan.json / 甘特上查看）
                        fix_ids, mob_ids = [], []
                        for h in handles or []:
                            if isinstance(h, MobileDevice):
                                mob_ids.append(h.dev_id)
                            elif isinstance(h, FixedDevice):
                                fix_ids.append(h.dev_id)
                        self.episode_devices.append(
                            {"FixedDevices": fix_ids, "MobileDevices": mob_ids})

                        plane._move_handles = handles
                        plane.start_move(to_site_id=site_to.site_id - 1, move_min=total_move)
                    else:
                        # ZY_M 保持原有逻辑
                        jid = self.jobs_obj.code2id()[mv]
                        self.episodes_situation.append(
                            (self.current_time, jid, site_to.site_id, pid, 0.0, float(move_min)))
                        self.episode_devices.append({"FixedDevices": [], "MobileDevices": []})
                        plane.start_move(to_site_id=site_to.site_id - 1, move_min=move_min)
                    continue

        # (3) 时间推进
        etas = []
        for p in self.planes:
            if p.status == "MOVING" and np.isfinite(p.eta_move_end) and p.eta_move_end > 0:
                etas.append(p.eta_move_end)
            if p.status == "PROCESSING" and np.isfinite(p.eta_proc_end) and p.eta_proc_end > 0:
                etas.append(p.eta_proc_end)
        delta_t = min(etas) if len(etas) > 0 else self.min_time_unit
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
                    plane.status = "IDLE"
                    # 释放牵引车等移动设备；把它们的位置更新为“目的站位”
                    if getattr(plane, "_move_handles", None):
                        self._release_resources(
                            plane._move_handles, plane, site_cur.site_id)
                        plane._move_handles = None
                    plane.move_code = None

            if plane.status == "PROCESSING":
                plane.eta_proc_end = max(0.0, plane.eta_proc_end - delta_t)
            
            # ===== 到位后尝试开工 =====
            if plane.status == "IDLE" and site_cur is not None:
                # 1) 计算就绪集（依赖/互斥）
                ready = self.task_obj.graph.enabled(plane.finished_codes, plane.ongoing_mutex) \
                    if (self.enable_deps or self.enable_mutex) else self.task_obj.jobs.jobs_object_list[:1]
                if site_cur.is_runway:
                    ready = list(ready) + self._outbound_ready_for(plane)
                # 2) 站位能力过滤 + 贪心打包
                site_cap_ids = set(site_cur.resource_ids_list)
                pack = self.task_obj.graph.pack_parallel(ready, site_cap_ids)

                # 2) 过滤：出场作业必须在跑道位
                pack = [j for j in pack if (j.group != "出场" or site_cur.is_runway)]

                # 3) 尝试为包内每个作业分配设备
                accepted = []
                for j in pack:
                    can, wait_min, handles = self._alloc_resources_for_job(
                        j, site_cur.site_id, plane)
                    if can:
                        accepted.append((j, wait_min, handles))

                if not accepted:
                    # ===== 兜底：单作业开工 =====
                    cand_jobs = self.task_obj.graph.enabled(plane.finished_codes, plane.ongoing_mutex) \
                        if (self.enable_deps or self.enable_mutex) else self.task_obj.jobs.jobs_object_list[:1]
                    if site_cur.is_runway:
                        cand_jobs = list(cand_jobs) + self._outbound_ready_for(plane)
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
                            if(not np.isfinite(proc_min)) or (proc_min < 0.0):
                                pass
                            else:
                                jid = self.jobs_obj.code2id()[job_for_site.code]
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
                                if not site_cur.is_runway:
                                    self.stand_current_occupancy[site_cur.site_id] = pid
                                if self.enable_long_occupy and job_for_site.code in ("ZY03", "ZY02") and job_for_site.required_resources:
                                    plane.long_occupy.add(job_for_site.required_resources[0])
                                plane._last_handles = handles
                else:
                    # ===== 并行开工成功 =====
                    durations, flat_handles = [], []
                    for (j, wait_min, hlist) in accepted:
                        d = float(wait_min + self._proc_time_minutes(j,
                                plane, site_cur, site_cur))
                        if(not np.isfinite(d)) or (d < 0.0):
                            continue
                        durations.append(d)
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
                        flat_handles.extend(hlist or [])

                    plane._active_jobs = [j for (j, _, _) in accepted]
                    plane._last_handles = flat_handles
                    plane.current_job_code = "+".join([j.code for j in plane._active_jobs])
                    plane.eta_proc_end = max(durations)
                    plane.status = "PROCESSING"
                    if not site_cur.is_runway:
                        # ★ 并行也要占位！
                        self.stand_current_occupancy[site_cur.site_id] = pid
                    inst_reward += self.bonus_parallel_per_job * (len(accepted) - 1)


            # 完工：仅做完工统一释放/收尾
            # LOG:完工逻辑
            if plane.status == "PROCESSING" and plane.eta_proc_end == 0.0:
                # 先释放设备（句柄可能不存在/为空，统一容错）
                cur_site_id = self.sites[plane.current_site_id].site_id if plane.current_site_id is not None else 1
                self._release_resources(
                    getattr(plane, "_last_handles", None), plane, cur_site_id)

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

                # 清占用&互锁时间
                if plane.current_site_id is not None:
                    sid = self.sites[plane.current_site_id].site_id
                    if sid in self.stand_current_occupancy:
                        self.stand_current_occupancy[sid] = None
                        self.last_leave_time_by_stand[sid] = self.current_time
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
