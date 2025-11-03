# utils/site.py
import numpy as np
from typing import List, Dict, Tuple


class Site:
    """站位/跑道。is_runway=True 表示跑道（29/30/31）。"""

    def __init__(self, site_id: int, absolute_position: np.ndarray,
                 resource_ids_list: List[int], resource_number: List[int],
                 is_runway: bool = False):
        self.site_id = site_id
        self.absolute_position = absolute_position.astype(float)
        # 该站位能执行的作业（用 job.index_id 表示）
        self.resource_ids_list = list(resource_ids_list)
        # 对应作业的并发产能
        self.resource_number = list(resource_number)
        self.is_runway = is_runway
        # 扰动不可用时间窗 [(start_min, end_min), ...]
        self.unavailable_windows: List[Tuple[float, float]] = []

    def is_available(self, now_min: float) -> bool:
        for s, e in self.unavailable_windows:
            if s <= now_min < e:
                return False
        return True


class Sites:
    """
    官方 31 个位置：
      - 1..28 为停机位
      - 29/30/31 为跑道位（仅执行 ZY_S / ZY_F / ZY_Z）
    """

    def __init__(self, jobs):
        self.sites_object_list: List[Site] = []
        self._build_layout(jobs)

    def _build_layout(self, jobs):
        J = jobs.jobs_object_list
        # 正确写法：变量注解或直接赋值，不能给 typing.Dict 赋值
        code2id: Dict[str, int] = {j.code: j.index_id for j in J}

        # 官方坐标（单位与你环境中速度/距离的约定一致；必要时配合 coord_scale）
        positions = [
            [45, 110], [35, 110], [25, 100], [25, 90], [35, 80], [45, 80],
            [95, 120], [85, 120], [75, 110], [85, 100], [
                95, 100], [105, 90], [95, 80], [85, 80],
            [45, 60], [35, 60], [25, 50], [35, 40], [
                45, 40], [55, 30], [45, 20], [35, 20],
            [95, 50], [85, 50], [75, 40], [75, 30], [85, 20], [95, 20],
            [130, 80], [130, 70], [130, 60]
        ]
        positions = [np.array(p, dtype=float) for p in positions]
        assert len(positions) == 31, "官方站位应为 31 个坐标"

        # 跑道位仅负责起飞流程相关作业
        runway_codes = [c for c in ("ZY_S", "ZY_F", "ZY_Z") if c in code2id]
        runway_ids = [code2id[c] for c in runway_codes]

        # 停机位可执行除上述起飞流程外的其他作业
        apron_cap_codes = [c for c in code2id.keys() if c not in runway_codes]
        apron_cap_ids = [code2id[c] for c in apron_cap_codes]

        # 1..28 停机位
        for sid in range(1, 29):
            pos = positions[sid - 1]
            res_ids = apron_cap_ids
            res_num = [1] * len(res_ids)   # 默认并发 1；需细化可在环境里按站位覆盖
            self.sites_object_list.append(
                Site(site_id=sid, absolute_position=pos,
                     resource_ids_list=res_ids, resource_number=res_num, is_runway=False)
            )

        # 29/30/31 跑道位（仅 ZY_S/ZY_F/ZY_Z）
        for sid in (29, 30, 31):  # 修正：不能用 range(29,30,31)
            pos = positions[sid - 1]
            res_ids = runway_ids
            res_num = [10**6] * len(res_ids)
            self.sites_object_list.append(
                Site(site_id=sid, absolute_position=pos,
                     resource_ids_list=res_ids, resource_number=res_num, is_runway=True)
            )

    # ==== 工具方法：供环境/可视化/扰动注入使用 ====
    def id2pos(self) -> Dict[int, np.ndarray]:
        return {s.site_id: s.absolute_position.copy() for s in self.sites_object_list}

    def mark_unavailable(self, site_id: int, start_min: float, end_min: float):
        """把某站位在时间窗内标记为不可用（环境 apply_disturbance 会调用）"""
        for s in self.sites_object_list:
            if s.site_id == site_id:
                s.unavailable_windows.append(
                    (float(start_min), float(end_min)))
                break
