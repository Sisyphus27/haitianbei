"""
这个文件书写作业类，即抽象保障资源类
每个作业拥有:作业id,作业编码,作业名,作业时长
"""
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Job:
    index_id: int
    code: str
    name: str
    group: str
    time_span: float  # 分钟；-1 表示按距离/状态计算
    required_resources: List[str] = field(default_factory=list)
    predecessors: List[str] = field(default_factory=list)
    mutex: List[str] = field(default_factory=list)


class Jobs:
    def __init__(self):
        self.jobs_object_list: List[Job] = []
        self._build_from_spec()
        # 建立多种映射，供 environment.py 使用
        self._code2id: Dict[str, int] = {
            j.code: j.index_id for j in self.jobs_object_list}
        self._id2code: Dict[int, str] = {
            j.index_id: j.code for j in self.jobs_object_list}
        self._code2job: Dict[str, Job] = {
            j.code: j for j in self.jobs_object_list}

    def _build_from_spec(self):
        S = []
        # 进场
        S += [dict(code="ZY_Z", name="着陆", group="进场", time=1, pre=[], req=[])]
        S += [dict(code="ZY_M", name="滑行", group="进场",
                   time=-1, pre=["ZY_Z"], req=[])]

        # FIXME
        S += [dict(code="ZY01", name="固定", group="保障",
                time=1, pre=[], req=[])]
        
        S += [dict(code="ZY02", name="供液压", group="保障",
                   time=1, pre=["ZY01"], req=["R008"])]
        S += [dict(code="ZY03", name="供电",   group="保障",
                   time=1, pre=["ZY01"], req=["R002"])]
        S += [dict(code="ZY04", name="加氧",   group="保障", time=3,
                   pre=["ZY01"], req=["R005", "R013"], mutex=["ZY10"])]
        S += [dict(code="ZY05", name="加氮",   group="保障",
                   time=3, pre=["ZY01"], req=["R003", "R012"])]
        S += [dict(code="ZY06", name="污水操作", group="保障",
                   time=4, pre=["ZY01"], req=["R007"])]
        S += [dict(code="ZY07", name="加气",   group="保障",
                   time=2, pre=["ZY02"], req=["R006", "R011"])]
        S += [dict(code="ZY08", name="开舱",   group="保障",
                   time=1, pre=["ZY02", "ZY03"], req=[])]
        S += [dict(code="ZY09", name="清水操作", group="保障",
                   time=2, pre=["ZY06"], req=["R007"])]
        S += [dict(code="ZY10", name="加燃油", group="保障", time=-1,
                   pre=["ZY03"], req=["R001"], mutex=["ZY04"])]
        S += [dict(code="ZY11", name="放梯",   group="保障",
                   time=1, pre=["ZY02", "ZY08"], req=[])]
        S += [dict(code="ZY12", name="装卸货/上下客", group="保障",
                   time=20, pre=["ZY03", "ZY08"], req=[])]
        S += [dict(code="ZY13", name="空调",   group="保障",
                   time=1, pre=["ZY03"], req=[])]
        S += [dict(code="ZY14", name="机组检查", group="保障", time=2,
                   pre=["ZY02", "ZY03", "ZY04", "ZY05", "ZY07", "ZY09", "ZY10", "ZY11", "ZY12", "ZY13"], req=[])]
        S += [dict(code="ZY15", name="关舱",   group="保障", time=5,
                   pre=["ZY02", "ZY03", "ZY14"], req=[])]
        S += [dict(code="ZY16", name="飞行员登机", group="保障",
                   time=1, pre=["ZY15"], req=[])]
        S += [dict(code="ZY17", name="启动发动机", group="保障",
                   time=1, pre=["ZY16"], req=[])]
        S += [dict(code="ZY18", name="暖机自检", group="保障",
                   time=3, pre=["ZY17"], req=[])]

        # 解固/转移
        S += [dict(code="ZY_L", name="解固",  group="转移", time=1,  pre=["ZY01"], req=[])]

        # FIXME
        S += [dict(code="ZY_T", name="转运",  group="转移", time=-1,
                pre=["ZY_L"], req=["R014"])]
        # 出场
        S += [dict(code="ZY_S", name="调整姿态", group="出场",
                   time=3, pre=["ZY_L"], req=[])]
        S += [dict(code="ZY_F", name="起飞",    group="出场",
                   time=1, pre=["ZY_S"], req=[])]

        self.jobs_object_list.clear()
        for i, x in enumerate(S):
            self.jobs_object_list.append(
                Job(index_id=i, code=x["code"], name=x["name"], group=x["group"], time_span=x["time"],
                    required_resources=x.get("req", []), predecessors=x.get("pre", []), mutex=x.get("mutex", []))
            )

    # === 提供 environment.py 依赖的接口 ===
    def code2id(self) -> Dict[str, int]:
        return self._code2id

    def id2code(self) -> Dict[int, str]:
        return self._id2code

    def get_job(self, code: str) -> Job:
        """按作业编码返回 Job 对象"""
        return self._code2job[code]
