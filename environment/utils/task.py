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
        self.any_pre: Dict[str, Set[str]] = {
            "ZY01": [{"ZY_T"}, {"ZY_M"}],  # FIXME
        }

    def enabled(self, finished: Set[str], ongoing_mutex: Set[str]) -> List[Job]:
        cand = []
        for j in self.jobs.jobs_object_list:
            if j.code in finished:
                continue
            if not self.pre[j.code].issubset(finished):
                continue
            if j.code in self.any_pre:
                groups = self.any_pre[j.code]
                if not any(group.issubset(finished) for group in groups):
                    continue
            if self.mutex[j.code] & ongoing_mutex:
                continue
            cand.append(j)
        return cand

    def all_finished(self, finished: Set[str]) -> bool:
        return "ZY_F" in finished


class Task:
    def __init__(self):
        self.jobs = Jobs()
        self.graph = TaskGraph(self.jobs)



