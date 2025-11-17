"""
飞机类
包含:飞机位置,飞机id,静态作业对象列表(每个飞机一样),已完成作业对象列表,剩余作业对象列表,已用时间,进行作业的历史站位id列表

"""
# utils/plane.py
import numpy as np
from typing import Any, Dict, List, Set, Optional, Tuple
from utils.task import Task, Job


class Planes:
    def __init__(self, numbers: int):
        self.plane_speed = 5.0  # m/s
        initial_position = np.array([10.0, 70.0], dtype=float)
        self.planes_object_list: List[Plane] = []
        for i in range(numbers):
            self.planes_object_list.append(
                Plane(i, Task(), initial_position.copy()))

    def count_jobs(self):
        left, allj = 0, 0
        for p in self.planes_object_list:
            left += p.left_jobs_count
            allj += len(p.task.jobs.jobs_object_list)
        return left, allj


class Plane:
    def __init__(self, plane_id: int, task: Task, initial_position: np.ndarray):
        self.plane_id = plane_id
        self.task = task
        self.position = initial_position
        self.status = "IDLE"       # "IDLE" | "MOVING" | "PROCESSING" | "DONE"
        self.current_job_code: Optional[str] = None
        self.current_site_id: Optional[int] = None
        self.eta_move_end: float = 0.0
        self.eta_proc_end: float = 0.0
        self.finished_codes: Set[str] = set()
        self.ongoing_mutex: Set[str] = set()
        self.time_spent = 0.0
        self.site_history: List[int] = []
        self.fuel_percent = 20.0
        self.long_occupy = set()   # eg. {"R002"} 当供电/液压开启长占开关时使用
        self.move_last_min: float = 0.0
        self.proc_last_min: float = 0.0
        self.move_code: Optional[str] = None
        self.paused_jobs: List[Dict[str, Any]] = []
        self.active_job_progress: Dict[str, float] = {}
        self._job_total_durations: Dict[str, float] = {}
        self._active_event_indices: List[Tuple[str, int]] = []
        self._handles_by_job: Dict[str, List[Any]] = {}

    @property
    def left_jobs_count(self) -> int:
        total = len(self.task.jobs.jobs_object_list)
        done = len(self.finished_codes)
        return max(0, total - done)

    def start_move(self, to_site_id: int, move_min: float):
        self.status = "MOVING"
        self.current_site_id = to_site_id
        self.eta_move_end = move_min
        self.move_last_min = move_min

    # LOG：互斥关系实现逻辑
    def start_job(self, job: Job, process_min: float):
        self.status = "PROCESSING"
        self.current_job_code = job.code
        self.eta_proc_end = process_min
        self.proc_last_min = float(process_min)
        self.active_job_progress = {job.code: float(process_min)}
        self._job_total_durations = {job.code: float(process_min)}
        self._active_event_indices = []
        for m in job.mutex:
            self.ongoing_mutex.add(m)

    def advance(self, delta_t: float, site_position: Optional[np.ndarray] = None):
        if self.status == "MOVING":
            self.eta_move_end = max(0.0, self.eta_move_end - delta_t)
            if self.eta_move_end == 0.0 and site_position is not None:
                self.position = site_position
        elif self.status == "PROCESSING":
            self.eta_proc_end = max(0.0, self.eta_proc_end - delta_t)

# utils/plane.py
    # LOG：互斥关系实现逻辑
    def finish_job(self, job: Job):
        if job.time_span > 0:
            self.time_spent += job.time_span
        for m in job.mutex:
            self.ongoing_mutex.discard(m)
        self.finished_codes.add(job.code)
        if self.current_site_id is not None:
            self.site_history.append(self.current_site_id)
        self.current_job_code = None
        self.status = "IDLE"
        self.eta_proc_end = 0.0
        self.active_job_progress = {}
        self._job_total_durations = {}
        self._active_event_indices = []
        self._handles_by_job = {}
