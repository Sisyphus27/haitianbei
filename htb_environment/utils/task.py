"""
这个文件用于描述task的内容

"""
from typing import List, Set, Dict
from utils.job import Jobs, Job


class TaskGraph:
    def __init__(self, jobs: Jobs):
        self.jobs = jobs
        self.pre: Dict[str, Set[str]] = {j.code: set(
            j.predecessors) for j in jobs.jobs_object_list}
        self.mutex: Dict[str, Set[str]] = {
            j.code: set(j.mutex) for j in jobs.jobs_object_list}
        self.any_pre: Dict[str, Set[str]] = {j.code: set(
            getattr(j, "any_pre", [])) for j in jobs.jobs_object_list}

    # LOG：依赖关系实现逻辑
    def _deps_satisfied(self, code: str, finished: Set[str]) -> bool:
        # 所有 pre 都完成，且任一 any_pre 完成
        if not self.pre.get(code, set()).issubset(finished):
            return False
        anyset = self.any_pre.get(code, set())
        return True if not anyset else anyset.intersection(finished)

    # LOG：依赖关系实现逻辑
    # LOG：互斥关系实现逻辑
    def enabled(self, finished: Set[str], ongoing_mutex: Set[str]) -> List[Job]:
        """返回所有"入度为0且不触发互斥"的就绪作业"""
        cand: List[Job] = []
        for j in self.jobs.jobs_object_list:
            c = j.code
            if c in finished:              # 已完成
                continue
            if j.group == "出场":          # 出场交由批次门控统一放行
                continue
            # 互斥：正在进行的互斥标签不允许再起
            if self.mutex.get(c, set()).intersection(ongoing_mutex):
                continue
            if self._deps_satisfied(c, finished):
                cand.append(j)
        return cand

    # LOG：互斥关系实现逻辑
    def pack_parallel(self, ready: List[Job], site_job_ids: Set[int]) -> List[Job]:
        """
        从就绪集里按"贪心"挑一组两两不互斥、且该站位能做的作业（并行执行）。
        """
        pack: List[Job] = []
        used_mutex: Set[str] = set()
        for j in ready:
            j_id = self.jobs.code2id()[j.code]
            if j_id not in site_job_ids:
                continue
            # 与已选包互不互斥
            if self.mutex.get(j.code, set()).intersection(used_mutex):
                continue
            pack.append(j)
            used_mutex.update(self.mutex.get(j.code, set()))
        return pack

    def all_finished(self, finished: Set[str]) -> bool:
        return "ZY_F" in finished


class Task:
    def __init__(self):
        self.jobs = Jobs()
        self.graph = TaskGraph(self.jobs)
