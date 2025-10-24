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

import warnings

warnings.filterwarnings("ignore")

class Dataset_KG(Dataset):
    """基于 Neo4j 的知识图谱构建与查询。

    - 初始化：连接 Neo4j，并创建固定资源（停机位1-28、跑道Z/29/30/31）。
    - 动态更新：可通过三元组增量更新（自动实体规范化与类型推断）。
    - 查询：按主体/关系检索三元组，邻居与图谱快照。
    """

    def __init__(self, root_path, flag="train", size=None,
                 data_path="海天杯-ST_Job_训练集.csv", load_data: bool = True,
                 neo4j_uri: str | None = None,
                 neo4j_user: str | None = None,
                 neo4j_password: str | None = None,
                 neo4j_database: str | None = None) -> None:
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

        # 定义类标签与关系/属性映射
        self._init_schema()
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
    def _init_schema(self):
        """定义类标签与关系/属性映射（Neo4j 推荐使用英文大写关系类型）。"""
        # 节点标签
        self.Aircraft = "Aircraft"
        self.Gate = "Stand"
        self.Runway = "Runway"
        self.Device = "Device"

        # 将中文谓词映射到关系类型或节点属性
        # rel: 使用关系，prop: 使用节点属性
        self.PREDICATE_MAP = {
            "使用跑道": {"type": "rel", "rel": "USES_RUNWAY"},
            "分配停机位": {"type": "rel", "rel": "ASSIGNED_GATE"},
            "到达停机位": {"type": "rel", "rel": "ARRIVED_GATE"},
            "目标停机位": {"type": "rel", "rel": "TARGET_GATE"},
            "待命位置": {"type": "rel", "rel": "STANDBY_AT"},
            "当前停机位": {"type": "rel", "rel": "HAS_CURRENT_GATE"},
            "动作": {"type": "prop", "prop": "action"},
            "时间": {"type": "prop", "prop": "time"},
            "坐标": {"type": "prop", "prop": "coordinate"},
            "速度": {"type": "prop", "prop": "speed"},
            "固定不动": {"type": "prop", "prop": "isFixed"},
        }

        # 关系类型到中文名（用于可视化/查询展示）
        self.REL_TO_CN = {
            "USES_RUNWAY": "使用跑道",
            "ASSIGNED_GATE": "分配停机位",
            "ARRIVED_GATE": "到达停机位",
            "TARGET_GATE": "目标停机位",
            "STANDBY_AT": "待命位置",
            "HAS_CURRENT_GATE": "当前停机位",
        }

    def _init_static_graph(self):
        """初始化固定资源到 Neo4j：停机位1-28，跑道Z，跑道29/30/31。"""
        with self.driver.session(database=self.neo4j_database) as sess:
            # 停机位 1-28
            for i in range(1, 29):
                name = f"停机位{i}"
                sess.run(
                    f"MERGE (n:{self.Gate} {{name:$name}}) SET n.isFixed = true",
                    {"name": name},
                )
            # 跑道 Z
            sess.run(
                f"MERGE (n:{self.Runway} {{name:$name}}) SET n.isFixed = true",
                {"name": "跑道Z"},
            )
            # 跑道 29/30/31
            for r in (29, 30, 31):
                sess.run(
                    f"MERGE (n:{self.Runway} {{name:$name}}) SET n.isFixed = true",
                    {"name": f"跑道{r}"},
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
            sess.run(f"MERGE (n:{label} {{name:$name}})", {"name": name})
        return name

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
            with self.driver.session(database=self.neo4j_database) as sess:
                # 确保目标存在
                sess.run(f"MERGE (o:{obj_label} {{name:$obj}})", {"obj": obj_norm})
                # 单值状态维护：当前停机位在到达时更新
                if predicate_str == "到达停机位":
                    # 先删除已有 HAS_CURRENT_GATE
                    sess.run(
                        "MATCH (s {name:$s})-[r:HAS_CURRENT_GATE]->(:Stand) DELETE r",
                        {"s": subject_name},
                    )
                    # 补充 HAS_CURRENT_GATE
                    sess.run(
                        "MATCH (s {name:$s}),(o {name:$o}) MERGE (s)-[:HAS_CURRENT_GATE]->(o)",
                        {"s": subject_name, "o": obj_norm},
                    )
                # 写入指定关系
                sess.run(
                    f"MATCH (s {{name:$s}}),(o {{name:$o}}) MERGE (s)-[r:{rel_type}]->(o)",
                    {"s": subject_name, "o": obj_norm},
                )
        else:
            # 属性更新为单值
            prop = meta["prop"]
            value = str(obj_value).replace("（", "(").replace("）", ")").replace("，", ",")
            with self.driver.session(database=self.neo4j_database) as sess:
                sess.run("MATCH (s {name:$name}) SET s[$prop] = $val",
                         {"name": subject_name, "prop": prop, "val": value})

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
        if predicate == "使用跑道":
            if ent.upper() == "Z":
                return "跑道Z"
            if re.fullmatch(r"\d+", ent):
                return f"跑道{ent}"

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
                        f"MATCH (s {{name:$name}})-[r:{meta['rel']}]->(o) RETURN o.name AS oname",
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
            n = sess.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            r = sess.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            return {"nodes_count": int(n), "edges_count": int(r)}

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

    def export_png(self, path: str, max_edges: int | None = 400):
        """从 Neo4j 查询全图并导出 PNG。
        优先使用 networkx（若可用），否则自动回退到纯 matplotlib 绘制，确保在较新的依赖版本下也能稳定出图。
        - 为避免过密，可限制最大边数；超出时随机抽样部分边。
        - 强制使用 Agg 后端，无需显示器环境。
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

        plt.figure(figsize=(12, 8))

        if nx is not None:
            # 使用 networkx 绘制
            G = nx.DiGraph()
            for n in unique_nodes:
                G.add_node(n)
            for u, v, p_str in edges:
                G.add_edge(u, v, label=p_str)
            pos = nx.spring_layout(G, k=0.9, seed=42)
            import matplotlib.pyplot as _plt  # type: ignore
            nx.draw_networkx_nodes(G, pos, node_color=[node_color(n) for n in G.nodes()], node_size=600)
            nx.draw_networkx_labels(G, pos, font_family=(chosen_family or "sans-serif"), font_size=8)
            nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', width=1.0, alpha=0.8)
            edge_labels = {(u, v): d.get('label', '') for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, font_family=(chosen_family or "sans-serif"))
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
            plt.scatter(xs, ys, c=colors, s=600, edgecolors='k', linewidths=0.5)

            # 绘制标签
            for n in unique_nodes:
                plt.text(pos[n][0], pos[n][1], n, fontsize=8, ha='center', va='center', fontfamily=(chosen_family or "sans-serif"))

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
                mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                plt.text(mx, my, p_str, fontsize=7, color='#444444', fontfamily=(chosen_family or "sans-serif"))

            plt.axis('equal')
            plt.axis('off')
            nodes_count = len(unique_nodes)
            edges_count = len(edges)

        import time
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            plt.tight_layout()
            plt.savefig(path, dpi=200)
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


