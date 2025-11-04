# kg_bridge.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import os
import numpy as np

from data_provider.data_loader import Dataset_KG  

class T1KGPriorAdapter:
    """
    把任务一的 Dataset_KG 作为知识先验来源，提供 env.attach_prior 期望的两个接口：
      - site_prior(site_ids, site_pos, site_caps) -> np.ndarray [#sites, ds]
      - plane_prior(planes) -> np.ndarray [#planes, dp]
    ds/dp 由 args.prior_dim_site / args.prior_dim_plane 控制。
    """

    def __init__(self, kg: Dataset_KG, ds: int = 8, dp: int = 3):
        self.kg = kg
        self.ds = ds
        self.dp = dp

    # 站位/跑道的先验，使用任务一图谱中的“到达/分配/使用频次、入度/出度”等简单统计作为启发
    def site_prior(self, site_ids: List[int], site_pos, site_caps) -> np.ndarray:
        feats = []
        tmp = []
        totals = {"in": 0, "out": 0, "assign": 0, "arrive": 0, "use_rw": 0}

        def site_name(sid: int) -> str:
            if sid == 0:  # 虚拟着陆位
                return "跑道Z"
            if 1 <= sid <= 28:
                return f"停机位{sid}"
            return f"跑道{sid}"  # 29/30/31

        # 先把每个站位的度与关键关系频次扫一遍
        for sid in site_ids:
            nm = site_name(sid)
            # {"out":[(s,p,o),...], "in":[(s,p,o),...]}
            nb = self.kg.neighbors(nm)
            ins, outs = nb.get("in", []), nb.get("out", [])
            deg_in, deg_out = len(ins), len(outs)
            cnt_assign = sum(1 for (_, p, _) in ins if p == "分配停机位")
            cnt_arrive = sum(1 for (_, p, _) in ins if p in ("到达停机位", "当前停机位"))
            cnt_use_rw = sum(1 for (_, p, _) in ins if p == "使用跑道")

            tmp.append((sid, deg_in, deg_out, cnt_assign,
                    cnt_arrive, cnt_use_rw))
            totals["in"] += deg_in
            totals["out"] += deg_out
            totals["assign"] += cnt_assign
            totals["arrive"] += cnt_arrive
            totals["use_rw"] += cnt_use_rw

        def norm(x, denom): return 0.0 if denom <= 0 else float(
            x) / float(denom)

        for (sid, di, do, ca, car, crw) in tmp:
            v = [
                1.0 if sid in (29, 30, 31) else 0.0,   # is_runway
                norm(di, totals["in"]),
                norm(do, totals["out"]),
                norm(ca, totals["assign"]),
                norm(car, totals["arrive"]),
                norm(crw, totals["use_rw"]),
            ]
            try:
                idx = site_ids.index(sid)
                cap_sum = float(sum(site_caps[idx].values())) if isinstance(
                    site_caps[idx], dict) else 0.0
            except Exception:
                cap_sum = 0.0
            v.append(cap_sum / 10.0)  # 经验归一
            v.append(1.0)             # bias

            # pad/trim 到 ds
            if len(v) < self.ds:
                v += [0.0] * (self.ds - len(v))
            else:
                v = v[:self.ds]
            feats.append(np.array(v, dtype=np.float32))

        return np.stack(feats, axis=0)

    # 飞机先验，是否有当前停机位、是否使用过跑道、是否有历史分配记录
    def plane_prior(self, planes) -> np.ndarray:
        feats = []
        for p in planes:
            nm = f"飞机{p.plane_id}"
            nb = self.kg.neighbors(nm)
            outs = nb.get("out", [])
            cur_gate = next(
                (o for (_, pred, o) in outs if pred == "当前停机位"), None)
            on_gate = 0.0
            if isinstance(cur_gate, str) and cur_gate.startswith("停机位"):
                try:
                    gid = int(cur_gate.replace("停机位", ""))
                    on_gate = gid / 31.0
                except Exception:
                    on_gate = 0.0
            used_rw = sum(1 for (_, pred, _) in outs if pred == "使用跑道")
            assigned = sum(1 for (_, pred, _) in outs if pred == "分配停机位")
            v = [on_gate, float(assigned > 0), float(used_rw > 0)]

            if len(v) < self.dp:
                v += [0.0] * (self.dp - len(v))
            else:
                v = v[:self.dp]
            feats.append(np.array(v, dtype=np.float32))
        return np.stack(feats, axis=0)


# 把调度生成的数据转成三元组 ===============
def schedule_to_kg_triples(for_gantt: List[tuple], env) -> List[tuple[str, str, str]]:
    """
    for_gantt: List[(time_min, job_id, site_id, plane_id, proc_min, move_min)]
    将其转换为任务一 PREDICATE_MAP 支持的中文谓词三元组，以便 Dataset_KG.update_with_triples 写入。
    """
    triples = []
    id2code = env.jobs_obj.id2code()  # job_id -> "ZY_*"
    for (t, jid, sid, pid, pmin, mmin) in sorted(for_gantt, key=lambda x: x[0]):
        code = id2code[jid]
        plane = f"飞机{pid}"
        site = "跑道Z" if sid == 0 else (
            f"停机位{sid}" if 1 <= sid <= 28 else f"跑道{sid}")

        # 基本事实
        triples.append((plane, "动作", code))
        triples.append((plane, "时间", f"{t:.2f}"))

        # 站位/跑道使用
        if sid == 0 or sid >= 29:
            # 着陆点或跑道作业
            triples.append((plane, "着陆跑道" if code == "ZY_Z" else "使用跑道", site))
        else:
            # 进入或停靠某个停机位
            if mmin and mmin > 0:
                triples.append((plane, "到达停机位", site))
            else:
                triples.append((plane, "当前停机位", site))
    return triples
