"""
战位的相关信息，被环境调用
A-R 20个战位
每个站位拥有:站位id,绝对位置,可以进行的作业对象列表,可以进行作业id列表
"""
# utils/site.py
import numpy as np
from typing import List, Dict, Tuple


class Site:
    """站位/跑道。若 is_runway=True 表示跑道（29/30/31）。"""

    def __init__(self, site_id: int, absolute_position: np.ndarray,
                 resource_ids_list: List[int], resource_number: List[int],
                 is_runway: bool = False):
        self.site_id = site_id
        self.absolute_position = absolute_position.astype(float)
        self.resource_ids_list = list(resource_ids_list)  # job.id 列表：该站位能执行的作业
        self.resource_number = list(resource_number)      # 对应作业的并发产能
        self.is_runway = is_runway
        # [(start,end),...]
        self.unavailable_windows: List[Tuple[float, float]] = []

    def is_available(self, now_min: float) -> bool:
        for s, e in self.unavailable_windows:
            if s <= now_min < e:
                return False
        return True


class Sites:
    """默认构造 28 个停机位 + 3 条跑道。位置与能力可按技术资料地图替换。"""

    def __init__(self, jobs):
        self.sites_object_list: List[Site] = []
        self._build_default(jobs)

    def _build_default(self, jobs):
        J = jobs.jobs_object_list
        code2id = {j.code: j.index_id for j in J}
        # 简化布局：28 个停位坐标均匀散布；最后三位为跑道
        positions = []
        for i in range(28):
            positions.append(
                np.array([10 + (i % 7)*10, 70 - (i//7)*10], dtype=float))
        runways = [
            np.array([120.0, 10.0], dtype=float),  # 29
            np.array([140.0, 10.0], dtype=float),  # 30
            np.array([160.0, 10.0], dtype=float),  # 31
        ]
        # 能力：每个停位可执行大部分“保障/进场/转移”作业；产能默认 1（可按站位类型细化）
        # 这里用 job.id 作为“能力标识”，与 environment 的掩码/匹配逻辑一致
        capability_ids = [code2id[c]
                          for c in code2id.keys() if c not in ("ZY_S", "ZY_F", "ZY_Z")]
        for i in range(28):
            res_ids = capability_ids
            res_num = [1]*len(res_ids)
            self.sites_object_list.append(Site(site_id=i+1, absolute_position=positions[i],
                                               resource_ids_list=res_ids, resource_number=res_num, is_runway=False))
        # FIXME:跑道仅负责 ZY_S/ZY_F/ZY_Z 作业
        for rid, pos in enumerate(runways, start=29):
            res_ids = [code2id["ZY_S"], code2id["ZY_F"], code2id["ZY_Z"]]
            res_num = [1, 1, 1]
            self.sites_object_list.append(Site(site_id=rid, absolute_position=pos,
                                               resource_ids_list=res_ids, resource_number=res_num, is_runway=True))



