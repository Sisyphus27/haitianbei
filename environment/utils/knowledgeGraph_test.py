# utils/kg_prior.py
import numpy as np
from typing import List, Dict, Any


class KGPrior:
    """
    占位用的KG先验编码器：
    - 假设有三类飞机(0/1/2)，用 one-hot + 简单度量构成先验
    - 站位先验：按站位id分簇（1-7, 8-14, 15-21, 22-28）与跑道(29/30/31)生成簇嵌入 + 资源统计
    - 返回的向量维度: site_prior -> [ds], plane_prior -> [dp]
    注：真实场景可由Task1产出的图（资源-站位-飞机-作业 关系图）喂入GNN得到这两个先验。
    """

    def __init__(self, ds: int = 8, dp: int = 3, seed: int = 42):
        self.ds = ds
        self.dp = dp
        rs = np.random.RandomState(seed)
        # 四个站位簇 + 3条跑道的随机向量（可固定以保证复现）
        self.cluster_embed = {
            "c1": rs.normal(0, 0.5, size=ds).astype(np.float32),
            "c2": rs.normal(0, 0.5, size=ds).astype(np.float32),
            "c3": rs.normal(0, 0.5, size=ds).astype(np.float32),
            "c4": rs.normal(0, 0.5, size=ds).astype(np.float32),
            "r29": rs.normal(0, 0.5, size=ds).astype(np.float32),
            "r30": rs.normal(0, 0.5, size=ds).astype(np.float32),
            "r31": rs.normal(0, 0.5, size=ds).astype(np.float32),
        }

    def _stand_cluster(self, site_id: int) -> str:
        if 1 <= site_id <= 7:
            return "c1"
        if 8 <= site_id <= 14:
            return "c2"
        if 15 <= site_id <= 21:
            return "c3"
        if 22 <= site_id <= 28:
            return "c4"
        if site_id == 29:
            return "r29"
        if site_id == 30:
            return "r30"
        return "r31"

    def site_prior(self, site_ids: List[int], site_pos: List[np.ndarray],
                   site_caps: List[Dict[int, int]]) -> np.ndarray:
        """
        返回 [S, ds]，每个站位的先验。
        这里演示：簇嵌入 + 站位资源数量求和（归一化）拼接/截断到 ds。
        """
        S = len(site_ids)
        out = np.zeros((S, self.ds), dtype=np.float32)
        for i, sid in enumerate(site_ids):
            base = self.cluster_embed[self._stand_cluster(sid)]
            # 简单把资源总量的归一化加入一个标量（演示）
            cap_sum = float(sum(site_caps[i].values())) if len(
                site_caps[i]) > 0 else 0.0
            cap_feat = np.array([cap_sum/10.0], dtype=np.float32)
            vec = np.concatenate([base[:-1], cap_feat], -
                                 1) if self.ds >= len(base) else base[:self.ds]
            out[i] = vec.astype(np.float32)
        return out

    def plane_prior(self, planes: List[Any]) -> np.ndarray:
        """
        返回 [A, dp]，每架飞机的先验。
        演示：用 fuel_percent 大小离散到三类（0/1/2）做 one-hot。
        """
        A = len(planes)
        out = np.zeros((A, self.dp), dtype=np.float32)
        for i, p in enumerate(planes):
            c = 0 if getattr(p, "fuel_percent", 50.0) < 40.0 else (
                1 if p.fuel_percent < 80.0 else 2)
            oh = np.zeros(self.dp, np.float32)
            oh[min(c, self.dp-1)] = 1.0
            out[i] = oh
        return out
