"""
Author: zy
Date: 2025-10-22 17:39:16
LastEditTime: 2025-10-22 17:39:20
LastEditors: zy
Description:
FilePath: haitianbei/data_provider/data_loader.py

"""

import os
import re
import json
import logging
from urllib.parse import quote

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# 知识图谱库
from neo4j import GraphDatabase, basic_auth

# 三元组抽取器
from models.triples_extraction import extract_triples

from htb_environment.utils.job import Jobs

import warnings

warnings.filterwarnings("ignore")

_logger = logging.getLogger(__name__)

class Dataset_KG(Dataset):
    """基于 Neo4j 的知识图谱构建与查询。

    - 初始化：连接 Neo4j，并创建固定资源（停机位1-28、跑道Z/29/30/31）。
    - 静态拓扑：构建停机位/跑道链路图与“作业-固定设备-停机位”保障拓扑。
    - 动态更新：可通过三元组增量更新（自动实体规范化与类型推断）。
    - 查询：按主体/关系检索三元组，邻居与图谱快照。
    """

    _STAND_CHAIN = ["跑道Z"] + [f"停机位{i}" for i in range(1, 29)] + ["跑道29", "跑道30", "跑道31"]
    # 固定设备覆盖表：来源于海天杯技术资料表4（与环境仿真配置保持一致）
    _FIXED_DEVICE_LAYOUT = {
        "FR1": ("R001", tuple(range(1, 7))),
        "FR2": ("R001", tuple(range(7, 15))),
        "FR3": ("R001", tuple(range(15, 23))),
        "FR4": ("R001", tuple(range(23, 29))),
        "FR5": ("R002", tuple(range(1, 7))),
        "FR6": ("R002", tuple(range(7, 15))),
        "FR7": ("R002", tuple(range(15, 23))),
        "FR8": ("R002", tuple(range(23, 29))),
        "FR9": ("R003", tuple(range(1, 7))),
        "FR10": ("R003", tuple(range(7, 15))),
        "FR11": ("R003", tuple(range(15, 23))),
        "FR12": ("R003", tuple(range(23, 29))),
        "FR13": ("R005", tuple(range(1, 7))),
        "FR14": ("R005", tuple(range(7, 15))),
        "FR15": ("R005", tuple(range(15, 23))),
        "FR16": ("R005", tuple(range(23, 29))),
        "FR17": ("R006", tuple(range(1, 7))),
        "FR18": ("R006", tuple(range(7, 15))),
        "FR19": ("R006", tuple(range(15, 23))),
        "FR20": ("R006", tuple(range(23, 29))),
        "FR21": ("R007", tuple(range(1, 7))),
        "FR22": ("R007", tuple(range(7, 15))),
        "FR23": ("R007", tuple(range(15, 23))),
        "FR24": ("R007", tuple(range(23, 29))),
        "FR25": ("R008", tuple(range(1, 7))),
        "FR26": ("R008", tuple(range(7, 15))),
        "FR27": ("R008", tuple(range(15, 23))),
        "FR28": ("R008", tuple(range(23, 29))),
        "FR29": ("R008", tuple(range(1, 7))),
        "FR30": ("R008", tuple(range(7, 15))),
        "FR31": ("R008", tuple(range(15, 23))),
        "FR32": ("R008", tuple(range(23, 29))),
    }

    # 约束初始化类级一次性标志，防止多实例重复创建触发大量 SCHEMA 日志
    _constraints_initialized: bool = False

    def __init__(self, root_path, flag="train", size=None,
                 data_path="海天杯-ST_Job_训练集.csv", load_data: bool = True,
                 neo4j_uri: str | None = None,
                 neo4j_user: str | None = None,
                 neo4j_password: str | None = None,
                 neo4j_database: str | None = None,
                 create_constraints: bool = True) -> None:
        super().__init__()
        self.root_path = root_path
        self.data_path = data_path
        if load_data:
            self.__read_data__()

        # Neo4j 连接参数（支持环境变量覆盖）
        self.neo4j_uri = neo4j_uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.environ.get("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.environ.get("NEO4J_PASSWORD", "neo4j")
        self.neo4j_database = neo4j_database or os.environ.get("NEO4J_DATABASE", None)

        # 驱动
        self.driver = GraphDatabase.driver(self.neo4j_uri, auth=basic_auth(self.neo4j_user, self.neo4j_password))
        # 试连
        try:
            with self.driver.session(database=self.neo4j_database) as sess:
                sess.run("RETURN 1 AS ok").single()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"无法连接 Neo4j: {e}")

        # 定义类标签与关系/属性映射（可控制是否尝试创建唯一约束）
        self._init_schema(create_constraints=create_constraints)
        # 初始化固定资源
        self._init_static_graph()

    # ----------------------------
    # 数据读取（保持原有 Dataset 行为）
    # ----------------------------
    def __read_data__(self):
        data = pd.read_csv(os.path.join(self.root_path, self.data_path))
        self.data = data

    def __getitem__(self, index):
        return self.data.iloc[index]

    def __len__(self):
        return len(self.data)

    # ----------------------------
    # 知识图谱：构建与更新
    # ----------------------------
    # FIXME: 初始化图有问题
    def _init_schema(self, create_constraints: bool = True):
        """定义类标签与关系/属性映射（Neo4j 推荐使用英文大写关系类型）。"""
        # 节点标签
        self.Aircraft = "Aircraft"
        self.Gate = "Stand"
        self.Runway = "Runway"
        self.Device = "Device"
        self.Job = "Job"
        self.Resource = "Resource"
        self.FixedDevice = "FixedDevice"
        # 新增：作业实例（按 任务_车编号 区分）
        self.JobInstance = "JobInstance"

        # 将中文谓词映射到关系类型或节点属性
        # rel: 使用关系，prop: 使用节点属性
        self.PREDICATE_MAP = {
            "使用跑道": {"type": "rel", "rel": "USES_RUNWAY"},
            # 同义：着陆跑道 视为 使用跑道
            "着陆跑道": {"type": "rel", "rel": "USES_RUNWAY"},
            "分配停机位": {"type": "rel", "rel": "ASSIGNED_GATE"},
            "到达停机位": {"type": "rel", "rel": "ARRIVED_GATE"},
            "目标停机位": {"type": "rel", "rel": "TARGET_GATE"},
            # 同义：滑至 -> 目标停机位（业务上表达为移动到目标位）
            "滑至": {"type": "rel", "rel": "TARGET_GATE"},
            "待命位置": {"type": "rel", "rel": "STANDBY_AT"},
            "当前停机位": {"type": "rel", "rel": "HAS_CURRENT_GATE"},
            # 设备-飞机 牵引关系
            "牵引": {"type": "rel", "rel": "TOWS"},
            # 以下为属性型谓词，同时补充“值节点关系”，便于在图中直观看到与主体的连接
            # 在写入时会：1) 将值写入主体属性；2) MERGE 值节点(label 见 val_label，name 为文本)；3) 创建 HAS_* 关系
            "动作": {"type": "prop", "prop": "action", "rel": "HAS_ACTION", "val_label": "Action"},
            "时间": {"type": "prop", "prop": "time", "rel": "HAS_TIME", "val_label": "Time"},
            "坐标": {"type": "prop", "prop": "coordinate", "rel": "HAS_COORDINATE", "val_label": "Coordinate"},
            "速度": {"type": "prop", "prop": "speed", "rel": "HAS_SPEED", "val_label": "Speed"},
            "固定不动": {"type": "prop", "prop": "isFixed"},
            # 新增：当文本包含作业编号（如 ZY03）时，可将其视为“执行作业”
            # 执行作业：兼容旧三元组，作为属性捕获原始文本 + 值节点；实例化逻辑在 _add_object 中处理
            "执行作业": {"type": "prop", "prop": "job_action", "rel": "HAS_JOB_ACTION", "val_label": "JobAction"},
        }

        # 关系类型到中文名（用于可视化/查询展示）
        self.REL_TO_CN = {
            "USES_RUNWAY": "使用跑道",
            # 同义谓词（着陆跑道）也映射到相同中文，展示一致
            "ASSIGNED_GATE": "分配停机位",
            "ARRIVED_GATE": "到达停机位",
            "TARGET_GATE": "目标停机位",
            "STANDBY_AT": "待命位置",
            "HAS_CURRENT_GATE": "当前停机位",
            "TOWS": "牵引",
            # 值节点关系（用于可视化/查询展示）
            "HAS_ACTION": "动作",
            "HAS_TIME": "时间",
            "HAS_COORDINATE": "坐标",
            "HAS_SPEED": "速度",
            "CHAIN_NEXT": "邻接",
            "REQUIRES_RESOURCE": "需要资源",
            "SERVES_RESOURCE": "服务资源",
            "HAS_DEVICE": "可用设备",
            "COVERS_STAND": "覆盖停机位",
            "AVAILABLE_AT": "可用停机位",
            "HAS_JOB_ACTION": "执行作业",
            "JOB_CAN_USE_DEVICE": "可用固定设备",
            # 新增：作业实例相关关系
            "PERFORMS_JOB_INSTANCE": "执行作业实例",
            "JOB_INSTANCE_OF": "作业类型",
            "ASSIGNED_TO_JOB_INSTANCE": "分配作业实例",
        }

        # 指定哪些关系在业务上是“单值”的：同一主体该关系同时最多指向一个客体
        # 避免出现冲突（重复/多值）时越积越多，这些关系在插入新三元组时会自动清理旧客体关系
        # 单值关系集合
        self.SINGLE_VALUED_RELS = {
            "USES_RUNWAY",        # 使用跑道（通常一次只使用一个跑道）
            "ASSIGNED_GATE",      # 分配停机位（分配应唯一）
            "TARGET_GATE",        # 目标停机位（目标应唯一）
            "STANDBY_AT",         # 待命位置（唯一）
            "HAS_CURRENT_GATE",   # 当前停机位（唯一，已在到达时同步维护）
            # 新增：同一时间一个飞机只执行一个作业实例；一台车只分配到一个作业实例
            "PERFORMS_JOB_INSTANCE",
            "ASSIGNED_TO_JOB_INSTANCE",
        }

        # 可选：为常用标签创建 name 唯一约束/索引，提升 MATCH 性能并避免重复节点
        if create_constraints and not Dataset_KG._constraints_initialized:
            try:
                with self.driver.session(database=self.neo4j_database) as sess:
                    labels_with_unique = [
                        self.Aircraft, self.Gate, self.Runway, self.Device,
                        self.Job, self.Resource, self.FixedDevice, self.JobInstance,
                        "Action", "Time", "Coordinate", "Speed", "Entity",
                    ]
                    for lbl in labels_with_unique:
                        # 约束命名：<label>_name_unique，避免冲突
                        cname = f"{lbl.lower()}_name_unique"
                        sess.run(
                            f"CREATE CONSTRAINT {cname} IF NOT EXISTS FOR (n:{lbl}) REQUIRE n.name IS UNIQUE"  # type: ignore[arg-type]
                        )
                Dataset_KG._constraints_initialized = True
            except Exception:
                # 无权限或旧版本语法不支持时，忽略，不影响正常运行
                pass

    def _init_static_graph(self):
        """构建静态拓扑：跑道-停机位链与作业-资源-设备-停机位保障图。"""

        jobs_spec = Jobs()
        resource_from_jobs = {res for job in jobs_spec.jobs_object_list for res in job.required_resources}
        all_resources = sorted(resource_from_jobs)

        with self.driver.session(database=self.neo4j_database) as sess:
            # --- 跑道/停机位链 ---
            for node_name in self._STAND_CHAIN:
                label = self._class_for_entity(node_name) or "Entity"
                sess.run(
                    f"MERGE (n:{label} {{name:$name}}) SET n.isFixed = true",  # type: ignore[arg-type]
                    {"name": node_name},
                )

            for src, dst in zip(self._STAND_CHAIN, self._STAND_CHAIN[1:]):
                src_label = self._class_for_entity(src) or "Entity"
                dst_label = self._class_for_entity(dst) or "Entity"
                sess.run(
                    f"MATCH (s:{src_label} {{name:$src}}) MATCH (t:{dst_label} {{name:$dst}}) "  # type: ignore[arg-type]
                    "MERGE (s)-[:CHAIN_NEXT]->(t) "
                    "MERGE (t)-[:CHAIN_NEXT]->(s)",
                    {"src": src, "dst": dst},
                )

            # --- 资源与作业（图2：作业->固定设备->支持停机位）---
            for res in all_resources:
                sess.run(
                    "MERGE (r:Resource {name:$name}) SET r.isFixed = true",  # type: ignore[arg-type]
                    {"name": res},
                )

            for job in jobs_spec.jobs_object_list:
                sess.run(
                    "MERGE (j:Job {name:$name}) "
                    "SET j.displayName = $display, j.group = $group, "
                    "j.duration = $duration, j.isFixed = true",  # type: ignore[arg-type]
                    {
                        "name": job.code,
                        "display": job.name,
                        "group": job.group,
                        "duration": float(job.time_span),
                    },
                )
                for res in job.required_resources:
                    sess.run(
                        "MATCH (j:Job {name:$job}) MATCH (r:Resource {name:$res}) "
                        "MERGE (j)-[:REQUIRES_RESOURCE]->(r)",
                        {"job": job.code, "res": res},
                    )

            # --- 固定设备覆盖 ---
            for device_name, (resource_code, stand_ids) in self._FIXED_DEVICE_LAYOUT.items():
                stand_names = [f"停机位{sid}" for sid in stand_ids]
                sess.run(
                    "MERGE (d:Device:FixedDevice {name:$name}) "
                    "SET d.resource = $resource, d.coverage = $coverage, d.isFixed = true",  # type: ignore[arg-type]
                    {
                        "name": device_name,
                        "resource": resource_code,
                        "coverage": stand_names,
                    },
                )

                sess.run(
                    "MATCH (d:FixedDevice {name:$device}) MATCH (r:Resource {name:$resource}) "
                    "MERGE (d)-[:SERVES_RESOURCE]->(r) "
                    "MERGE (r)-[:HAS_DEVICE]->(d)",
                    {"device": device_name, "resource": resource_code},
                )

                # Job -> FixedDevice 静态可用性（不与停机位直接连通，避免拥挤）
                base_jobs = [j for j in jobs_spec.jobs_object_list if resource_code in j.required_resources]
                for base in base_jobs:
                    sess.run(
                        "MATCH (j:Job {name:$job}) MATCH (d:FixedDevice {name:$dev}) "
                        "MERGE (j)-[:JOB_CAN_USE_DEVICE]->(d)",
                        {"job": base.code, "dev": device_name},
                    )

        _logger.info(
            "Static KG ready: %d chain nodes, %d resources, %d fixed devices",  # noqa: G004
            len(self._STAND_CHAIN),
            len(all_resources),
            len(self._FIXED_DEVICE_LAYOUT),
        )

    def fixed_nodes(self):
        """返回被标注为“固定不动”的实体标签列表及计数。"""
        with self.driver.session(database=self.neo4j_database) as sess:
            rs = sess.run("MATCH (n) WHERE n.isFixed = true RETURN n.name AS name")
            labels = sorted([r["name"] for r in rs])
            return {"count": len(labels), "labels": labels}

    # ----------------------------
    # RDF 基础：实体与三元组
    # ----------------------------
    def _ensure_entity(self, name: str, label: str | None = None):
        """确保实体存在，按标签与 name 唯一。返回实体名（name）。"""
        if label is None:
            label = self._class_for_entity(name) or "Entity"
        with self.driver.session(database=self.neo4j_database) as sess:
            sess.run(f"MERGE (n:{label} {{name:$name}})", {"name": name})  # type: ignore[arg-type]
        return name
    # FIXME: 将资源车上时和任务分离了。
    def _add_object(self, subject_name: str, predicate_str: str, obj_value: str, obj_type_hint: str | None = None):
        """依据谓词写入 Neo4j：关系或属性。"""
        meta = self.PREDICATE_MAP.get(predicate_str)
        if not meta:
            # 未知谓词，作为普通属性存储在 subject 上（字符串）
            with self.driver.session(database=self.neo4j_database) as sess:
                sess.run("MATCH (s {name:$name}) SET s += $props",
                         {"name": subject_name, "props": {predicate_str: str(obj_value)}})
            return

        if meta["type"] == "rel":
            rel_type = meta["rel"]
            # 目标实体规范化与类型推断
            obj_norm = self._canon_entity(obj_value, predicate_str)
            obj_label = self._class_for_entity(obj_norm) or "Entity"
            s_label = self._class_for_entity(subject_name)
            with self.driver.session(database=self.neo4j_database) as sess:
                # 特例：执行作业（当 subject 是飞机 且 obj 是作业编号）
                if rel_type == "PERFORMS_JOB":  # 已不再使用（保留兼容，不会触发）
                    # 支持两种写法：
                    # 1) PERFORMS_JOB 对象为作业编号（如 ZY03）-> 维持原有飞机 -> 作业关系
                    # 2) 对象为“作业_车编号”（如 ZY03_MR01/ZY10_FR5）-> 创建作业实例并绑定具体车辆
                    m_inst = re.fullmatch(r"(ZY[A-Z0-9]+)_(FR\d{1,3}|MR\d{2})", obj_norm)
                    if m_inst:
                        base_job = m_inst.group(1)
                        device_code = m_inst.group(2)
                        # 确保基础节点
                        sess.run("MERGE (j:Job {name:$n})", {"n": base_job})
                        sess.run("MERGE (d:Device {name:$n})", {"n": device_code})
                        # 作业实例（按 任务_车编号 命名）
                        sess.run(
                            "MERGE (ji:JobInstance {name:$name}) "
                            "SET ji.job=$job, ji.device=$dev, ji.isFixed=false",
                            {"name": obj_norm, "job": base_job, "dev": device_code},
                        )
                        # 关联：实例 -> 作业类型
                        sess.run(
                            "MATCH (ji:JobInstance {name:$ji}) MATCH (j:Job {name:$j}) "
                            "MERGE (ji)-[:JOB_INSTANCE_OF]->(j)",
                            {"ji": obj_norm, "j": base_job},
                        )
                        # 单值化：同一主体只保留一个作业实例
                        if s_label:
                            sess.run(
                                f"MATCH (s:{s_label} {{name:$s}})-[r:PERFORMS_JOB_INSTANCE]->(x:JobInstance) "  # type: ignore[arg-type]
                                "WHERE x.name <> $ji DELETE r",
                                {"s": subject_name, "ji": obj_norm},
                            )
                        else:
                            sess.run(
                                "MATCH (s {name:$s})-[r:PERFORMS_JOB_INSTANCE]->(x:JobInstance) "
                                "WHERE x.name <> $ji DELETE r",
                                {"s": subject_name, "ji": obj_norm},
                            )
                        # 单值化：车辆只分配到一个作业实例
                        sess.run(
                            "MATCH (d:Device {name:$d})-[r:ASSIGNED_TO_JOB_INSTANCE]->(x:JobInstance) "
                            "WHERE x.name <> $ji DELETE r",
                            {"d": device_code, "ji": obj_norm},
                        )
                        # 合入关系：主体(通常为飞机) 执行 -> 作业实例
                        if s_label:
                            sess.run(
                                f"MATCH (s:{s_label} {{name:$s}}) MATCH (ji:JobInstance {{name:$ji}}) "  # type: ignore[arg-type]
                                "MERGE (s)-[:PERFORMS_JOB_INSTANCE]->(ji)",
                                {"s": subject_name, "ji": obj_norm},
                            )
                        else:
                            sess.run(
                                "MATCH (s {name:$s}) MATCH (ji:JobInstance {name:$ji}) "
                                "MERGE (s)-[:PERFORMS_JOB_INSTANCE]->(ji)",
                                {"s": subject_name, "ji": obj_norm},
                            )
                        # 合入关系：车辆 分配 -> 作业实例
                        sess.run(
                            "MATCH (d:Device {name:$d}) MATCH (ji:JobInstance {name:$ji}) "
                            "MERGE (d)-[:ASSIGNED_TO_JOB_INSTANCE]->(ji)",
                            {"d": device_code, "ji": obj_norm},
                        )
                        # 兼容保留：主体 -> 作业（便于已有查询）
                        if s_label:
                            sess.run(
                                f"MATCH (s:{s_label} {{name:$s}}) MATCH (j:Job {{name:$j}}) "  # type: ignore[arg-type]
                                "MERGE (s)-[:PERFORMS_JOB]->(j)",
                                {"s": subject_name, "j": base_job},
                            )
                        else:
                            sess.run(
                                "MATCH (s {name:$s}) MATCH (j:Job {name:$j}) MERGE (s)-[:PERFORMS_JOB]->(j)",
                                {"s": subject_name, "j": base_job},
                            )
                        return
                    else:
                        # 若主体是设备 + 对象为作业编号 -> 自动实例化为 任务_车编号
                        if s_label == self.Device and re.fullmatch(r"ZY[A-Z0-9]+", obj_norm):
                            base_job = obj_norm
                            device_code = subject_name
                            inst_name = f"{base_job}_{device_code}"
                            sess.run("MERGE (j:Job {name:$n})", {"n": base_job})
                            sess.run(
                                "MERGE (ji:JobInstance {name:$name}) SET ji.job=$job, ji.device=$dev, ji.isFixed=false",
                                {"name": inst_name, "job": base_job, "dev": device_code},
                            )
                            sess.run(
                                "MATCH (ji:JobInstance {name:$ji}) MATCH (j:Job {name:$j}) MERGE (ji)-[:JOB_INSTANCE_OF]->(j)",
                                {"ji": inst_name, "j": base_job},
                            )
                            # 单值化：该设备只分配一个作业实例
                            sess.run(
                                "MATCH (d:Device {name:$d})-[r:ASSIGNED_TO_JOB_INSTANCE]->(x:JobInstance) WHERE x.name <> $ji DELETE r",
                                {"d": device_code, "ji": inst_name},
                            )
                            sess.run(
                                "MATCH (d:Device {name:$d}) MATCH (ji:JobInstance {name:$ji}) MERGE (d)-[:ASSIGNED_TO_JOB_INSTANCE]->(ji)",
                                {"d": device_code, "ji": inst_name},
                            )
                            return
                        # 旧行为：仅基于作业编号（主体通常为飞机）
                        sess.run("MERGE (j:Job {name:$name})", {"name": obj_norm})
                        if s_label:
                            sess.run(
                                f"MATCH (s:{s_label} {{name:$s}}) MATCH (j:Job {{name:$j}}) MERGE (s)-[:PERFORMS_JOB]->(j)",  # type: ignore[arg-type]
                                {"s": subject_name, "j": obj_norm},
                            )
                        else:
                            sess.run(
                                "MATCH (s {name:$s}) MATCH (j:Job {name:$j}) MERGE (s)-[:PERFORMS_JOB]->(j)",
                                {"s": subject_name, "j": obj_norm},
                            )
                        return
                # 确保目标存在
                sess.run(f"MERGE (o:{obj_label} {{name:$obj}})", {"obj": obj_norm})  # type: ignore[arg-type]
                # 单值状态维护：当前停机位在到达时更新
                if predicate_str == "到达停机位":
                    # 先删除已有 HAS_CURRENT_GATE
                    if s_label:
                        sess.run(
                            f"MATCH (s:{s_label} {{name:$s}})-[r:HAS_CURRENT_GATE]->(:Stand) DELETE r",  # type: ignore[arg-type]
                            {"s": subject_name},
                        )
                    else:
                        sess.run(
                            "MATCH (s {name:$s})-[r:HAS_CURRENT_GATE]->(:Stand) DELETE r",
                            {"s": subject_name},
                        )
                    # 补充 HAS_CURRENT_GATE
                    if s_label:
                        sess.run(
                            f"MATCH (s:{s_label} {{name:$s}}) MATCH (o:Stand {{name:$o}}) MERGE (s)-[:HAS_CURRENT_GATE]->(o)",  # type: ignore[arg-type]
                            {"s": subject_name, "o": obj_norm},
                        )
                    else:
                        sess.run(
                            "MATCH (s {name:$s}) MATCH (o:Stand {name:$o}) MERGE (s)-[:HAS_CURRENT_GATE]->(o)",
                            {"s": subject_name, "o": obj_norm},
                        )
                # 冲突处理：对于单值关系，若已存在指向其它客体的边，先删除后再合入
                if rel_type in self.SINGLE_VALUED_RELS:
                    if s_label:
                        sess.run(
                            f"MATCH (s:{s_label} {{name:$s}})-[r:{rel_type}]->(x) WHERE x.name <> $o DELETE r",  # type: ignore[arg-type]
                            {"s": subject_name, "o": obj_norm},
                        )
                    else:
                        sess.run(
                            f"MATCH (s {{name:$s}})-[r:{rel_type}]->(x) WHERE x.name <> $o DELETE r",  # type: ignore[arg-type]
                            {"s": subject_name, "o": obj_norm},
                        )
                # 写入（或保持）指定关系
                if s_label:
                    sess.run(
                        f"MATCH (s:{s_label} {{name:$s}}) MATCH (o:{obj_label} {{name:$o}}) MERGE (s)-[r:{rel_type}]->(o)",  # type: ignore[arg-type]
                        {"s": subject_name, "o": obj_norm},
                    )
                else:
                    sess.run(
                        f"MATCH (s {{name:$s}}) MATCH (o:{obj_label} {{name:$o}}) MERGE (s)-[r:{rel_type}]->(o)",  # type: ignore[arg-type]
                        {"s": subject_name, "o": obj_norm},
                    )
        else:
            # 属性更新为单值 +（可选）值节点关系
            prop = meta["prop"]
            value = str(obj_value).replace("（", "(").replace("）", ")").replace("，", ",")
            # 规范化特殊值（如 坐标）
            if predicate_str == "坐标":
                value = value.replace(" ", "")
            with self.driver.session(database=self.neo4j_database) as sess:
                # 1) 更新主体属性
                sess.run("MATCH (s {name:$name}) SET s[$prop] = $val",
                         {"name": subject_name, "prop": prop, "val": value})
                # 2) 若定义了 rel 与值节点标签，则补充值节点与关系
                rel_type = meta.get("rel")
                val_label = meta.get("val_label")
                if rel_type and val_label:
                    # 单值化：确保同一主体在该 HAS_* 关系下只连向该值
                    s_label = self._class_for_entity(subject_name)
                    if s_label:
                        sess.run(
                            f"MATCH (s:{s_label} {{name:$s}})-[r:{rel_type}]->(x:{val_label}) WHERE x.name <> $o DELETE r",  # type: ignore[arg-type]
                            {"s": subject_name, "o": value},
                        )
                    else:
                        sess.run(
                            f"MATCH (s {{name:$s}})-[r:{rel_type}]->(x:{val_label}) WHERE x.name <> $o DELETE r",  # type: ignore[arg-type]
                            {"s": subject_name, "o": value},
                        )
                    # 合入值节点与关系
                    sess.run(
                        f"MERGE (o:{val_label} {{name:$o}})",  # type: ignore[arg-type]
                        {"o": value},
                    )
                    if s_label:
                        sess.run(
                            f"MATCH (s:{s_label} {{name:$s}}) MATCH (o:{val_label} {{name:$o}}) MERGE (s)-[:{rel_type}]->(o)",  # type: ignore[arg-type]
                            {"s": subject_name, "o": value},
                        )
                    else:
                        sess.run(
                            f"MATCH (s {{name:$s}}) MATCH (o:{val_label} {{name:$o}}) MERGE (s)-[:{rel_type}]->(o)",  # type: ignore[arg-type]
                            {"s": subject_name, "o": value},
                        )
            # 3) 特判：若“动作”值形如 ZYxx 或 ZYxx_车编号，则等价映射为一次“执行作业”写入
            if predicate_str == "动作" or predicate_str == "执行作业":
                # 动作或执行作业文本中若检测到 任务 或 任务_车编号，进行实例化
                action_val = value
                pattern = r"ZY[A-Z0-9]+(?:_(FR\d{1,3}|MR\d{2}))?"
                m = re.fullmatch(pattern, action_val)
                if m:
                    base_and_device = action_val
                    m_inst = re.fullmatch(r"(ZY[A-Z0-9]+)_(FR\d{1,3}|MR\d{2})", base_and_device)
                    with self.driver.session(database=self.neo4j_database) as sess:
                        if m_inst:
                            base_job = m_inst.group(1)
                            device_code = m_inst.group(2)
                            inst_name = base_and_device
                            # 节点
                            sess.run("MERGE (j:Job {name:$n})", {"n": base_job})
                            sess.run("MERGE (d:Device {name:$n})", {"n": device_code})
                            sess.run("MERGE (ji:JobInstance {name:$n}) SET ji.job=$job, ji.device=$dev", {"n": inst_name, "job": base_job, "dev": device_code})
                            sess.run("MATCH (ji:JobInstance {name:$ji}) MATCH (j:Job {name:$j}) MERGE (ji)-[:JOB_INSTANCE_OF]->(j)", {"ji": inst_name, "j": base_job})
                            # 单值：设备分配实例
                            sess.run("MATCH (d:Device {name:$d})-[r:ASSIGNED_TO_JOB_INSTANCE]->(x:JobInstance) WHERE x.name <> $ji DELETE r", {"d": device_code, "ji": inst_name})
                            sess.run("MATCH (d:Device {name:$d}) MATCH (ji:JobInstance {name:$ji}) MERGE (d)-[:ASSIGNED_TO_JOB_INSTANCE]->(ji)", {"d": device_code, "ji": inst_name})
                            # 主体（飞机或设备）执行实例
                            s_label = self._class_for_entity(subject_name)
                            if s_label:
                                sess.run(f"MATCH (s:{s_label} {{name:$s}})-[r:PERFORMS_JOB_INSTANCE]->(x:JobInstance) WHERE x.name <> $ji DELETE r", {"s": subject_name, "ji": inst_name})  # type: ignore[arg-type]
                                sess.run(f"MATCH (s:{s_label} {{name:$s}}) MATCH (ji:JobInstance {{name:$ji}}) MERGE (s)-[:PERFORMS_JOB_INSTANCE]->(ji)", {"s": subject_name, "ji": inst_name})  # type: ignore[arg-type]
                            else:
                                sess.run("MATCH (s {name:$s})-[r:PERFORMS_JOB_INSTANCE]->(x:JobInstance) WHERE x.name <> $ji DELETE r", {"s": subject_name, "ji": inst_name})
                                sess.run("MATCH (s {name:$s}) MATCH (ji:JobInstance {name:$ji}) MERGE (s)-[:PERFORMS_JOB_INSTANCE]->(ji)", {"s": subject_name, "ji": inst_name})
                            # 兼容：主体 -> 作业
                            if s_label:
                                sess.run(f"MATCH (s:{s_label} {{name:$s}}) MATCH (j:Job {{name:$j}}) MERGE (s)-[:PERFORMS_JOB]->(j)", {"s": subject_name, "j": base_job})  # type: ignore[arg-type]
                            else:
                                sess.run("MATCH (s {name:$s}) MATCH (j:Job {name:$j}) MERGE (s)-[:PERFORMS_JOB]->(j)", {"s": subject_name, "j": base_job})
                        else:
                            # 仅作业编号（主体执行作业）
                            base_job = base_and_device
                            sess.run("MERGE (j:Job {name:$n})", {"n": base_job})
                            s_label = self._class_for_entity(subject_name)
                            if s_label:
                                sess.run(f"MATCH (s:{s_label} {{name:$s}}) MATCH (j:Job {{name:$j}}) MERGE (s)-[:PERFORMS_JOB]->(j)", {"s": subject_name, "j": base_job})  # type: ignore[arg-type]
                            else:
                                sess.run("MATCH (s {name:$s}) MATCH (j:Job {name:$j}) MERGE (s)-[:PERFORMS_JOB]->(j)", {"s": subject_name, "j": base_job})

    # 规范化/类型推断
    def _canon_entity(self, ent: str, predicate: str | None = None) -> str:
        ent = str(ent).strip()
        # 统一中文括号/逗号
        ent = ent.replace("（", "(").replace("）", ")").replace("，", ",")

        # 停机位：如 “14号”/“14号停机位” -> “停机位14”
        m_gate_num = re.fullmatch(r"(\d+)号(?:停机位)?", ent)
        if m_gate_num and (predicate in {"到达停机位", "目标停机位", "分配停机位"} or ent.endswith("停机位") or True):
            return f"停机位{int(m_gate_num.group(1))}"

        # 跑道：Z / 29 / 30 / 31 -> 跑道Z/跑道29...
        # 同义谓词："着陆跑道" 与 "使用跑道" 一致
        if predicate in {"使用跑道", "着陆跑道"}:
            if ent.upper() == "Z":
                return "跑道Z"
            if re.fullmatch(r"\d+", ent):
                return f"跑道{ent}"
            # 若文本即为“着陆跑道”，统一归一为“跑道Z”
            if ent in {"着陆跑道", "着陆跑道Z", "着陆跑道z"}:
                return "跑道Z"

        # 统一“坐标(60,260)”
        if predicate == "坐标":
            ent = ent.replace(" ", "")

        return ent

    def _class_for_entity(self, name: str) -> str | None:
        if name.startswith("飞机"):
            return self.Aircraft
        if name.startswith("停机位"):
            return self.Gate
        if name.startswith("跑道"):
            return self.Runway
        if re.search(r"牵引车|加氧车|加氮车|空气终端|氧气终端|氮气终端|压缩空气终端|清洗装置", name):
            return self.Device
        if re.fullmatch(r"FR\d{1,3}", name):
            return self.Device
        if re.fullmatch(r"MR\d{2}", name):
            return self.Device
        if re.fullmatch(r"R\d{3}", name):
            return self.Resource
        # 作业/作业实例
        if re.fullmatch(r"ZY[A-Z0-9]+_(FR\d{1,3}|MR\d{2})", name):
            return self.JobInstance
        if re.fullmatch(r"ZY[A-Z0-9]+", name):
            return self.Job
        return None

    def _get_label(self, node_name_or_record) -> str:
        return str(node_name_or_record)

    def update_with_triples(self, triples: list[tuple[str, str, str]]):
        """将三元组增量合入 Neo4j 图谱。"""
        for s, p, o in triples:
            s_c = self._canon_entity(s, None)
            # 确保主体存在
            self._ensure_entity(s_c, self._class_for_entity(s_c))
            # 写入关系/属性
            self._add_object(s_c, p, o)

    def extract_and_update(self, text: str) -> list[tuple[str, str, str]]:
        """对输入文本抽取并更新图谱，返回抽取到的三元组（原始值）。"""
        triples = extract_triples(text)
        self.update_with_triples(triples)
        return triples

    # ----------------------------
    # 查询接口
    # ----------------------------
    def query(self, subject: str, predicate: str | None = None):
        """按主体（和可选关系）检索三元组，返回 (subject, predicate, object_str)。"""
        s_c = self._canon_entity(subject)
        res: list[tuple[str, str, str]] = []
        with self.driver.session(database=self.neo4j_database) as sess:
            # 关系
            if predicate is None:
                rs = sess.run(
                    "MATCH (s {name:$name})-[r]->(o) RETURN type(r) AS rel, o.name AS oname",
                    {"name": s_c},
                )
                for r in rs:
                    rel_cn = str(self.REL_TO_CN.get(r.get("rel"), r.get("rel", "")))
                    res.append((s_c, rel_cn, str(r.get("oname", ""))))
                # 属性
                props = sess.run("MATCH (s {name:$name}) RETURN s AS s", {"name": s_c}).single()
                if props and props.get("s"):
                    sprops = props["s"]
                    for cn, meta in self.PREDICATE_MAP.items():
                        if meta["type"] == "prop":
                            key = meta["prop"]
                            if key in sprops and sprops[key] is not None:
                                res.append((s_c, cn, str(sprops[key])))
                return res
            else:
                meta = self.PREDICATE_MAP.get(predicate)
                if meta and meta["type"] == "rel":
                    rs = sess.run(
                        f"MATCH (s {{name:$name}})-[r:{meta['rel']}]->(o) RETURN o.name AS oname",  # type: ignore[arg-type]
                        {"name": s_c},
                    )
                    for r in rs:
                        res.append((s_c, predicate, str(r.get("oname", ""))))
                elif meta and meta["type"] == "prop":
                    row = sess.run("MATCH (s {name:$name}) RETURN s[$key] AS v", {"name": s_c, "key": meta["prop"]}).single()
                    if row and row.get("v") is not None:
                        res.append((s_c, predicate, str(row["v"])))
                return res

    def neighbors(self, entity: str):
        e_c = self._canon_entity(entity)
        outs, ins = [], []
        with self.driver.session(database=self.neo4j_database) as sess:
            rs_out = sess.run("MATCH (s {name:$name})-[r]->(o) RETURN type(r) AS rel, o.name AS oname",
                              {"name": e_c})
            for r in rs_out:
                outs.append((e_c, str(self.REL_TO_CN.get(r.get("rel"), r.get("rel", ""))), str(r.get("oname", ""))))
            rs_in = sess.run("MATCH (s)-[r]->(o {name:$name}) RETURN type(r) AS rel, s.name AS sname",
                             {"name": e_c})
            for r in rs_in:
                ins.append((str(r.get("sname", "")), str(self.REL_TO_CN.get(r.get("rel"), r.get("rel", ""))), e_c))
        return {"out": outs, "in": ins}

    def graph_snapshot(self):
        with self.driver.session(database=self.neo4j_database) as sess:
            row_n = sess.run("MATCH (n) RETURN count(n) AS c").single()
            row_r = sess.run("MATCH ()-[r]->() RETURN count(r) AS c").single()
            n = int(row_n["c"]) if row_n and "c" in row_n else 0
            r = int(row_r["c"]) if row_r and "c" in row_r else 0
            return {"nodes_count": n, "edges_count": r}

    # ----------------------------
    # 文本化上下文与冲突检测（用于训练数据准备）
    # ----------------------------
    def text_context(self, focus_entities: list[str] | None = None, max_edges: int = 200) -> str:
        """将当前 KG 的局部状态渲染为可读文本，供模型作为上下文。

        - 若指定 focus_entities，则优先输出这些实体的一阶出入边；否则输出全局快照信息。
        """
        lines: list[str] = []
        try:
            if focus_entities:
                for ent in focus_entities:
                    nb = self.neighbors(ent)
                    if nb.get('out'):
                        for s, p, o in nb['out'][: max_edges // 2]:
                            lines.append(f"{s} -[{p}]-> {o}")
                    if nb.get('in'):
                        for s, p, o in nb['in'][: max_edges // 2]:
                            lines.append(f"{s} -[{p}]-> {o}")
            else:
                snap = self.graph_snapshot()
                lines.append(f"[SNAPSHOT] nodes={snap.get('nodes_count',0)} edges={snap.get('edges_count',0)}")
        except Exception:
            pass
        ctx = "\n".join(lines[:max_edges])
        if not ctx:
            ctx = "(当前图为空或仅有固定节点)"
        return "【KG状态】\n" + ctx

    # ---- 冲突检测基础查询 ----
    def _other_aircraft_at_gate(self, gate_name: str, exclude_aircraft: str | None = None) -> list[str]:
        """返回在指定停机位上（当前/分配）的其它飞机名称列表。"""
        names: set[str] = set()
        with self.driver.session(database=self.neo4j_database) as sess:
            # 当前停机位
            rs1 = sess.run(
                "MATCH (a:Aircraft)-[:HAS_CURRENT_GATE]->(g:Stand {name:$g}) RETURN a.name AS name",
                {"g": gate_name},
            )
            for r in rs1:
                n = str(r.get("name", ""))
                if n:
                    names.add(n)
            # 分配停机位
            rs2 = sess.run(
                "MATCH (a:Aircraft)-[:ASSIGNED_GATE]->(g:Stand {name:$g}) RETURN a.name AS name",
                {"g": gate_name},
            )
            for r in rs2:
                n = str(r.get("name", ""))
                if n:
                    names.add(n)
        if exclude_aircraft:
            names.discard(exclude_aircraft)
        return sorted(names)

    def _other_aircraft_using_runway(self, runway_name: str, exclude_aircraft: str | None = None) -> list[str]:
        names: set[str] = set()
        with self.driver.session(database=self.neo4j_database) as sess:
            rs = sess.run(
                "MATCH (a:Aircraft)-[:USES_RUNWAY]->(r:Runway {name:$r}) RETURN a.name AS name",
                {"r": runway_name},
            )
            for r in rs:
                n = str(r.get("name", ""))
                if n:
                    names.add(n)
        if exclude_aircraft:
            names.discard(exclude_aircraft)
        return sorted(names)

    def _other_devices_towing_aircraft(self, aircraft_name: str, exclude_device: str | None = None) -> list[str]:
        names: set[str] = set()
        with self.driver.session(database=self.neo4j_database) as sess:
            rs = sess.run(
                "MATCH (d:Device)-[:TOWS]->(a:Aircraft {name:$a}) RETURN d.name AS name",
                {"a": aircraft_name},
            )
            for r in rs:
                n = str(r.get("name", ""))
                if n:
                    names.add(n)
        if exclude_device:
            names.discard(exclude_device)
        return sorted(names)

    def check_event_conflicts(self, event_text: str) -> dict:
        """对单条事件进行简单规则的冲突检测（只读；不会写回图谱）。

        返回：{"is_conflict": bool, "reasons": [str,...], "triples": [(s,p,o), ...]}
        规则示例（启发式，便于自动生成标注）：
        - 到达/目标/分配 停机位G：若KG中已有其他飞机的当前或分配为G，则判为冲突（占用冲突）。
        - 使用/着陆 跑道R：若KG中已有其他飞机正在使用R，则可能冲突（跑道占用）。
        - 设备 牵引 飞机A：若KG中已有其它设备正牵引A，则判为冲突（多设备冲突）。
        """
        reasons: list[str] = []
        conflict = False
        triples = extract_triples(event_text)

        for s, p, o in triples:
            s = str(s)
            p = str(p)
            o = str(o)
            # 规范化用于匹配（与构图时一致）
            s_label = self._class_for_entity(self._canon_entity(s)) or "Entity"
            if p in {"到达停机位", "目标停机位", "分配停机位"}:
                gate = self._canon_entity(o, p)
                others = self._other_aircraft_at_gate(gate, exclude_aircraft=s)
                if others:
                    conflict = True
                    reasons.append(f"停机位占用冲突：{gate} 已有关联飞机 {', '.join(others)}")
            elif p in {"使用跑道", "着陆跑道"}:
                runway = self._canon_entity(o, p)
                others = self._other_aircraft_using_runway(runway, exclude_aircraft=s)
                if others:
                    conflict = True
                    reasons.append(f"跑道占用冲突：{runway} 已有飞机 {', '.join(others)} 正在使用")
            elif p == "牵引":
                # s 可能是设备，o 是飞机
                aircraft = self._canon_entity(o)
                device = self._canon_entity(s)
                other_devices = self._other_devices_towing_aircraft(aircraft, exclude_device=device)
                if other_devices:
                    conflict = True
                    reasons.append(f"牵引冲突：飞机 {aircraft} 已被 {', '.join(other_devices)} 牵引")

        return {"is_conflict": conflict, "reasons": reasons, "triples": triples}

    # ----------------------------
    # 维护操作：重置图谱（可保留固定节点）
    # ----------------------------
    def reset_graph(self, keep_fixed: bool = True):
        """清理 Neo4j 中的动态数据。

        keep_fixed=True 时仅删除非固定节点（n.isFixed!=true 或缺失），并清除固定节点之间可能的边；
        keep_fixed=False 时清空整个图谱。
        """
        with self.driver.session(database=self.neo4j_database) as sess:
            if keep_fixed:
                # 删除非固定节点
                sess.run("""
                MATCH (n)
                WHERE NOT coalesce(n.isFixed, false)
                DETACH DELETE n
                """)
                # 清理固定节点之间的边（若存在历史关系）
                sess.run("""
                MATCH (a)-[r]->(b)
                WHERE coalesce(a.isFixed, false) AND coalesce(b.isFixed, false)
                DELETE r
                """)
            else:
                sess.run("MATCH (n) DETACH DELETE n")
        # 重新确保固定资源存在
        if keep_fixed:
            self._init_static_graph()

    # ----------------------------
    # 可选：SPARQL 查询
    # ----------------------------
    def sparql(self, query: str):
        """占位：Neo4j 不支持 SPARQL。"""
        return {"error": "SPARQL 不适用于 Neo4j（请使用 Cypher）"}

    def export_ttl(self, path: str):
        """占位：Neo4j 无直接 TTL 导出。可使用 APOC 导出或 CSV。这里写入说明文件。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Neo4j 暂不支持直接导出 TTL。可安装 APOC，使用 apoc.export.graphml / apoc.export.csv 等功能。\n")

    def export_png(
        self,
        path: str,
        max_edges: int | None = 400,
        *,
        figsize: str | tuple[float, float] | None = None,
        dpi: int = 200,
        layout: str = "spring",
        hide_edge_labels: bool = False,
        node_label_font_size: int = 8,
        edge_label_font_size: int = 7,
        highlight_triples: list[tuple[str, str, str]] | None = None,
    ):
        """从 Neo4j 查询全图并导出 PNG（增强版本）。

        新增参数用于缓解『图片太拥挤看不清』问题：
        - figsize: 图尺寸，支持 "14,10" / "14x10" / (14,10) 形式；默认 12x8。
        - dpi: 输出分辨率（默认 200）。
        - layout: spring|kamada|circular|spectral|shell 影响节点布局。
        - hide_edge_labels: 隐藏边标签以减少遮挡。
        - node_label_font_size / edge_label_font_size: 字体大小可调。
        - highlight_triples: 将本次新增的 (s,p,o) 三元组的边高亮（红色）。

        兼容：保持原默认逻辑与返回结构不变；旧调用不受影响。
        """
        import math
        import random
        import traceback

        try:
            import matplotlib  # type: ignore
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore
            from matplotlib import font_manager as fm  # type: ignore
        except Exception as e:  # noqa: BLE001
            return {"error": f"可视化依赖缺失: {e}", "trace": traceback.format_exc()}

        # networkx 可选导入：失败则降级为纯 matplotlib 绘制
        nx = None
        try:
            import networkx as _nx  # type: ignore
            nx = _nx
        except Exception:
            nx = None

        # 配置中文字体，优先使用系统常见中文字体
        candidates = [
            "Microsoft YaHei",  # 微软雅黑（Windows 常见）
            "SimHei",           # 黑体
            "SimSun",           # 宋体
            "Noto Sans CJK SC", # 思源黑体
            "Arial Unicode MS",
        ]
        chosen_family = None
        for name in candidates:
            try:
                fp = fm.FontProperties(family=name)
                # 若字体不存在且禁止回退，将抛异常
                fm.findfont(fp, fallback_to_default=False)
                chosen_family = name
                break
            except Exception:
                continue
        if chosen_family:
            matplotlib.rcParams['font.sans-serif'] = [chosen_family]
        # 解决负号显示成方块的问题
        matplotlib.rcParams['axes.unicode_minus'] = False

        # 从 Neo4j 拉取节点与边
        names: list[str] = []
        edges: list[tuple[str, str, str]] = []
        with self.driver.session(database=self.neo4j_database) as sess:
            # 先加入按类型的独立节点（即便没有边也可展示）
            for lbl in [self.Gate, self.Runway, self.Aircraft, self.Device]:
                rs = sess.run(f"MATCH (n:{lbl}) RETURN n.name AS name")  # type: ignore[arg-type]
                for r in rs:
                    names.append(r["name"])  # 可能重复，稍后去重

            # 收集边（只展示主要业务关系）
            allow_rels = list(self.REL_TO_CN.keys())
            rs = sess.run(
                "MATCH (s)-[r]->(o) WHERE type(r) IN $rels RETURN s.name AS s, type(r) AS rel, o.name AS o",
                {"rels": allow_rels},
            )
            edges = [(r["s"], r["o"], str(self.REL_TO_CN.get(r["rel"], r["rel"]))) for r in rs]

        # 合并边上的节点并去重
        for u, v, _ in edges:
            names.append(u)
            names.append(v)
        unique_nodes = list(dict.fromkeys(names))

        if max_edges is not None and len(edges) > max_edges:
            edges = random.sample(edges, max_edges)

        # 节点颜色按类型区分
        def node_color(name: str) -> str:
            if name.startswith("飞机"):
                return "#1f77b4"  # 蓝
            if name.startswith("停机位"):
                return "#2ca02c"  # 绿
            if name.startswith("跑道"):
                return "#ff7f0e"  # 橙
            return "#7f7f7f"      # 灰

        # 解析 figsize
        fig_w, fig_h = 12.0, 8.0
        if figsize:
            try:
                if isinstance(figsize, tuple):
                    fig_w, fig_h = float(figsize[0]), float(figsize[1])
                elif isinstance(figsize, str):
                    txt = figsize.lower().replace("x", ",")
                    parts = [p.strip() for p in txt.split(",") if p.strip()]
                    if len(parts) == 2:
                        fig_w, fig_h = float(parts[0]), float(parts[1])
            except Exception:  # noqa: BLE001
                pass

        plt.figure(figsize=(fig_w, fig_h))

        if nx is not None:
            # 使用 networkx 绘制
            G = nx.DiGraph()
            for n in unique_nodes:
                G.add_node(n)
            for u, v, p_str in edges:
                G.add_edge(u, v, label=p_str)
            # 根据布局类型选择
            try:
                if layout == "kamada":
                    pos = nx.kamada_kawai_layout(G)
                elif layout == "circular":
                    pos = nx.circular_layout(G)
                elif layout == "spectral":
                    pos = nx.spectral_layout(G)
                elif layout == "shell":
                    pos = nx.shell_layout(G)
                else:  # spring 默认，加自适应 k 缓解重叠
                    # k 越大距离越远；经验缩放：基础 0.9 * log(N+1)
                    k_base = 0.9 * math.log(len(unique_nodes) + 1, 2)
                    pos = nx.spring_layout(G, k=k_base, seed=42)
            except Exception:
                pos = nx.spring_layout(G, k=0.9, seed=42)
            import matplotlib.pyplot as _plt  # type: ignore
            # 节点尺寸自适应：节点越多减小
            node_size = max(150, 9000 // max(1, len(unique_nodes)))
            nx.draw_networkx_nodes(
                G,
                pos,
                node_color=[node_color(n) for n in G.nodes()],
                node_size=node_size,
            )
            nx.draw_networkx_labels(
                G,
                pos,
                font_family=(chosen_family or "sans-serif"),
                font_size=node_label_font_size,
            )
            # 高亮新增边：分两次绘制
            highlight_set = set()
            if highlight_triples:
                for s, p, o in highlight_triples:
                    highlight_set.add((s, o))
            normal_edges = []
            highlight_edges = []
            for u, v in G.edges():
                if (u, v) in highlight_set:
                    highlight_edges.append((u, v))
                else:
                    normal_edges.append((u, v))
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=normal_edges,
                arrows=True,
                arrowstyle='-|>',
                width=1.0,
                alpha=0.6,
            )
            if highlight_edges:
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=highlight_edges,
                    arrows=True,
                    arrowstyle='-|>',
                    width=1.4,
                    alpha=0.9,
                    edge_color='#d62728',
                )
            if not hide_edge_labels:
                edge_labels = {(u, v): d.get('label', '') for u, v, d in G.edges(data=True)}
                nx.draw_networkx_edge_labels(
                    G,
                    pos,
                    edge_labels=edge_labels,
                    font_size=edge_label_font_size,
                    font_family=(chosen_family or "sans-serif"),
                )
            nodes_count = G.number_of_nodes()
            edges_count = G.number_of_edges()
        else:
            # 纯 matplotlib 绘制（环形布局）
            N = max(len(unique_nodes), 1)
            radius = 5.0
            angle_step = (2 * math.pi) / N
            pos = {}
            for i, name in enumerate(unique_nodes):
                theta = i * angle_step
                pos[name] = (radius * math.cos(theta), radius * math.sin(theta))

            xs = [pos[n][0] for n in unique_nodes]
            ys = [pos[n][1] for n in unique_nodes]
            colors = [node_color(n) for n in unique_nodes]
            node_size = max(150, 9000 // max(1, len(unique_nodes)))
            plt.scatter(xs, ys, c=colors, s=node_size, edgecolors='k', linewidths=0.5)

            # 绘制标签
            for n in unique_nodes:
                plt.text(
                    pos[n][0],
                    pos[n][1],
                    n,
                    fontsize=node_label_font_size,
                    ha='center',
                    va='center',
                    fontfamily=(chosen_family or "sans-serif"),
                )

            # 绘制有向边
            for u, v, p_str in edges:
                if u not in pos or v not in pos:
                    continue
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                plt.annotate(
                    '', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='-|>', lw=1.0, alpha=0.8, color='#333333')
                )
                # 边标签放在中点稍偏移
                if not hide_edge_labels:
                    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                    plt.text(
                        mx,
                        my,
                        p_str,
                        fontsize=edge_label_font_size,
                        color='#444444',
                        fontfamily=(chosen_family or "sans-serif"),
                    )

            plt.axis('equal')
            plt.axis('off')
            nodes_count = len(unique_nodes)
            edges_count = len(edges)

        import time
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            plt.tight_layout()
            plt.savefig(path, dpi=dpi)
            plt.close()
        except Exception as e:  # noqa: BLE001
            err = f"savefig failed: {e}"
            return {"error": err, "trace": traceback.format_exc(), "path": path}

        # 等待文件系统落盘（在某些 Windows 环境下可能需要极短延迟）
        for _ in range(10):
            if os.path.exists(path) and os.path.getsize(path) > 0:
                break
            time.sleep(0.05)

        size = os.path.getsize(path) if os.path.exists(path) else 0
        return {"nodes": int(nodes_count), "edges": int(edges_count), "path": path, "size": int(size)}


# ----------------------------
# 规则学习：训练数据构造（SFT）
# ----------------------------
def load_instruction_jsonl(path: str) -> list[dict]:
    """读取 JSONL（每行包含 instruction/input/output）。非法行将被跳过。"""
    arr: list[dict] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if all(k in obj for k in ("instruction", "input", "output")):
                    arr.append(obj)
            except Exception:
                continue
    return arr


def _read_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def _clean_markdown_images(md: str) -> str:
    # 去除图片/空白行
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)
    md = "\n".join([ln.rstrip() for ln in md.splitlines() if ln.strip() != ""])  # 去空白行
    return md


def build_rules_sft_samples_from_md(rules_md_path: str, *, max_samples: int = 50, chunk_chars: int = 2000) -> list[dict]:
    """基于规则 Markdown 文档构造一个极简的SFT训练集（占位用）。

    返回的每个样本形如：{"instruction", "input", "output"}
    - instruction: 固定为“提炼冲突判断检查要点”
    - input: 规则文档片段（加上【规则文档】前缀）
    - output: 占位答案（建议替换为人工标注）
    """
    if not os.path.isfile(rules_md_path):
        raise FileNotFoundError(rules_md_path)
    md = _read_text(rules_md_path)
    md = _clean_markdown_images(md)
    prefix = "【规则文档】\n"
    chunks = [md[i:i+chunk_chars] for i in range(0, len(md), chunk_chars)]
    samples: list[dict] = []
    for i, ch in enumerate(chunks[:max_samples]):
        samples.append({
            "instruction": "请从规则文档片段中提炼面向冲突判断的检查要点（5-10条）",
            "input": prefix + ch,
            "output": "检查要点：1）… 2）… 3）…（此为占位样本，建议替换为人工标注数据）",
        })
    return samples


# ----------------------------
# 训练数据准备：结合 KG 状态的指令微调样本
# ----------------------------
def _build_rules_prompt_text(rules_md_path: str | None) -> str:
    if not rules_md_path or not os.path.isfile(rules_md_path):
        return ""
    md = _read_text(rules_md_path)
    md = _clean_markdown_images(md)
    return "【规则文档】\n" + md


def build_kg_sft_samples_from_events(
    kg: Dataset_KG | None,
    events: list[str],
    *,
    rules_md_path: str | None = None,
    focus_entities: list[str] | None = None,
    max_edges: int = 200,
    auto_label: bool = True,
) -> list[dict]:
    """基于 KG 状态与事件文本，构造包含上下文的 SFT 样本。

    每个样本形如：{"instruction", "input", "output"}
    - instruction：固定任务描述
    - input：规则文档文本（可选） + KG状态文本 + 事件文本
    - output：
        * auto_label=True：使用启发式冲突检测自动生成“结论/依据/建议”占位答案
        * 否则：生成待人工标注的模板
    """
    rules_text = _build_rules_prompt_text(rules_md_path)
    instr = "任务：判断以下事件是否与当前状态或规则冲突。请先给出结论（合规/冲突），再给出1-3条依据，最后给出可操作建议。"
    samples: list[dict] = []
    for ev in events:
        if not isinstance(ev, str) or not ev.strip():
            continue
        ev = ev.strip()
        kg_text = (
            kg.text_context(focus_entities=focus_entities, max_edges=max_edges)
            if kg is not None else "【KG状态】\n(离线模式，未加载图谱)"
        )
        prompt_parts = [rules_text, kg_text, "【事件】\n" + ev, "输出格式：结论+依据+建议"]
        input_text = "\n\n".join([p for p in prompt_parts if p])

        if auto_label:
            if kg is None:
                # 离线模式无法做基于KG的自动冲突检测，输出温和的占位弱标签
                concl = "合规"
                reasons = ["离线模式未加载KG，无法自动冲突检测"]
                advice = "请结合实时KG状态进行复核。"
                output = (
                    f"结论：{concl}\n"
                    f"依据：\n- " + "\n- ".join(reasons[:3]) + "\n"
                    f"建议：{advice}"
                )
            else:
                chk = kg.check_event_conflicts(ev)
                concl = "冲突" if chk.get("is_conflict") else "合规"
                reasons = chk.get("reasons") or ["未发现与当前KG状态明显冲突"]
                # 简要建议模板
                if concl == "冲突":
                    advice = "请及时协调资源（调整跑道/停机位/设备），确保安全与流程合规。"
                else:
                    advice = "按计划执行，保持与现场状态一致并持续监控。"
                output = (
                    f"结论：{concl}\n"
                    f"依据：\n- " + "\n- ".join(reasons[:3]) + "\n"
                    f"建议：{advice}"
                )
        else:
            output = (
                "结论：<合规/冲突>\n"
                "依据：\n- <依据1>\n- <依据2>\n"
                "建议：<可操作建议>"
            )

        samples.append({
            "instruction": instr,
            "input": input_text,
            "output": output,
        })
    return samples


def save_jsonl(samples: list[dict], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for obj in samples:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_events_from_file(path: str) -> list[str]:
    """读取事件列表：支持 .txt（每行一条）或 .jsonl（优先使用 text/event 字段）。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    events: list[str] = []
    if path.lower().endswith('.txt'):
        with open(path, 'r', encoding='utf-8') as f:
            for ln in f:
                s = ln.strip()
                if s:
                    events.append(s)
    else:
        with open(path, 'r', encoding='utf-8') as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                    val = obj.get('text') or obj.get('event') or obj.get('input')
                    if isinstance(val, str) and val.strip():
                        events.append(val.strip())
                except Exception:
                    continue
    return events


