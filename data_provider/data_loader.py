"""
Author: zy
Date: 2025-10-22 17:39:16
LastEditTime: 2025-10-22 17:39:20
LastEditors: zy
Description:
FilePath: \haitianbei\data_provider\data_loader.py

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
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS

# 三元组抽取器
from models.triples_extraction import extract_triples

import warnings

warnings.filterwarnings("ignore")

class Dataset_KG(Dataset):
    """构建并维护一个可动态更新的知识图谱。

    - 初始化：加入固定资源（停机位1-28、着陆跑道Z、起飞跑道29-31）。
    - 动态更新：可通过传入文本，先用 extract_triples 抽取三元组，再增量更新到图谱。
    - 查询：提供简单的查找接口（按主体/关系等）。
    """

    def __init__(self, root_path, flag="train", size=None,
                 data_path="海天杯-ST_Job_训练集.csv", load_data: bool = True) -> None:
        super().__init__()
        self.root_path = root_path
        self.data_path = data_path
        if load_data:
            self.__read_data__()

        # RDF 图谱（rdflib）
        self.g = Graph()
        self.EX = Namespace("http://example.org/htb/")
        self.g.bind("ex", self.EX)
        self._init_schema()
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
        """定义最小本体（类与属性）。"""
        EX = self.EX
        # 定义类
        self.Aircraft = EX.Aircraft
        self.Gate = EX.Stand
        self.Runway = EX.Runway
        self.Device = EX.Device

        # 定义属性（谓词）
        self.PROP = {
            "动作": EX.hasAction,
            "时间": EX.hasTime,
            "坐标": EX.hasCoordinate,
            "速度": EX.hasSpeed,
            "使用跑道": EX.usesRunway,
            "分配停机位": EX.assignedGate,
            "到达停机位": EX.arrivedGate,
            "目标停机位": EX.targetGate,
            "待命位置": EX.standbyAt,
            "当前停机位": EX.hasCurrentGate,
            "固定不动": EX.isFixed,
        }

    def _init_static_graph(self):
        """初始化固定资源到图谱：停机位1-28，跑道Z（着陆），跑道29/30/31（起飞）。"""
        # 停机位 1-28（固定）
        for i in range(1, 29):
            uri = self._ensure_entity(f"停机位{i}", self.Gate)
            # 标注为固定不动
            self.g.add((uri, self.PROP["固定不动"], Literal(True)))
        # 着陆跑道 Z（固定）
        z_uri = self._ensure_entity("跑道Z", self.Runway)
        self.g.add((z_uri, self.PROP["固定不动"], Literal(True)))
        # 起飞跑道 29/30/31（固定）
        for r in (29, 30, 31):
            r_uri = self._ensure_entity(f"跑道{r}", self.Runway)
            self.g.add((r_uri, self.PROP["固定不动"], Literal(True)))

    def fixed_nodes(self):
        """返回被标注为“固定不动”的实体标签列表及计数。"""
        res = []
        for s, _, _ in self.g.triples((None, self.PROP["固定不动"], None)):
            if isinstance(s, URIRef):
                res.append(self._get_label(s))
        return {"count": len(res), "labels": sorted(res)}

    # ----------------------------
    # RDF 基础：实体与三元组
    # ----------------------------
    def _uri_for_entity(self, name: str) -> URIRef:
        # 使用 URL 安全编码确保 URI 合法
        return URIRef(self.EX["ent/" + quote(name, safe="")])

    def _ensure_entity(self, label: str, clazz: URIRef | None = None) -> URIRef:
        uri = self._uri_for_entity(label)
        # rdfs:label
        if (uri, RDFS.label, None) not in self.g:
            self.g.add((uri, RDFS.label, Literal(label)))
        # rdf:type
        if clazz is not None and (uri, RDF.type, clazz) not in self.g:
            self.g.add((uri, RDF.type, clazz))
        return uri

    def _add_object(self, subject_uri: URIRef, predicate_str: str, obj_value: str, obj_type_hint: str | None = None):
        pred = self.PROP.get(predicate_str, self.EX[predicate_str])

        # 决定对象是资源还是字面量
        if predicate_str in {"使用跑道", "分配停机位", "到达停机位", "目标停机位", "待命位置", "当前停机位"}:
            # 作为实体对象
            obj_norm = self._canon_entity(obj_value, predicate_str)
            # 类型推断
            clazz = self._class_for_entity(obj_norm)
            obj_uri = self._ensure_entity(obj_norm, clazz)
            # 单值状态维护
            if predicate_str == "到达停机位":
                # 维护 hasCurrentGate
                self.g.remove((subject_uri, self.PROP["当前停机位"], None))
                self.g.add((subject_uri, self.PROP["当前停机位"], obj_uri))
            self.g.add((subject_uri, pred, obj_uri))
        else:
            # 作为字面量
            lit = Literal(obj_value)
            # 坐标/速度/时间等属于单值状态，可选择叠加 has* 状态属性
            if predicate_str == "坐标":
                self.g.remove((subject_uri, self.PROP["坐标"], None))
            if predicate_str == "速度":
                self.g.remove((subject_uri, self.PROP["速度"], None))
            if predicate_str == "时间":
                self.g.remove((subject_uri, self.PROP["时间"], None))
            self.g.add((subject_uri, pred, lit))

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

    def _class_for_entity(self, name: str) -> URIRef | None:
        if name.startswith("飞机"):
            return self.Aircraft
        if name.startswith("停机位"):
            return self.Gate
        if name.startswith("跑道"):
            return self.Runway
        if re.search(r"牵引车|加氧车|加氮车|空气终端|氧气终端|氮气终端|压缩空气终端|清洗装置", name):
            return self.Device
        return None

    def _get_label(self, node) -> str:
        """获取节点的人类可读标签；若无则回退为简化的 URI 末段或直接字符串。"""
        try:
            lab = self.g.value(node, RDFS.label)
            if lab is not None:
                return str(lab)
        except Exception:
            pass
        s = str(node)
        return s.split("/")[-1]

    def update_with_triples(self, triples: list[tuple[str, str, str]]):
        """将三元组增量合入 RDF 图谱（rdflib）。"""
        for s, p, o in triples:
            s_c = self._canon_entity(s, None)
            subj_uri = self._ensure_entity(s_c, self._class_for_entity(s_c))
            self._add_object(subj_uri, p, o)

    def extract_and_update(self, text: str) -> list[tuple[str, str, str]]:
        """对输入文本抽取并更新图谱，返回抽取到的三元组（规范化前的原始值）。"""
        triples = extract_triples(text)
        self.update_with_triples(triples)
        return triples

    # ----------------------------
    # 查询接口
    # ----------------------------
    def query(self, subject: str, predicate: str | None = None):
        """按主体（和可选关系）检索三元组，返回 (subject, predicate, object_str)。"""
        s_c = self._canon_entity(subject)
        s_uri = self._uri_for_entity(s_c)
        res = []
        if predicate is None:
            for _, p, o in self.g.triples((s_uri, None, None)):
                p_str = next((k for k, v in self.PROP.items() if v == p), str(p).split("/")[-1])
                if isinstance(o, URIRef):
                    res.append((s_c, p_str, self._get_label(o)))
                else:
                    res.append((s_c, p_str, str(o)))
            return res
        # 指定关系
        p_uri = self.PROP.get(predicate, self.EX[predicate])
        for _, _, o in self.g.triples((s_uri, p_uri, None)):
            if isinstance(o, URIRef):
                res.append((s_c, predicate, self._get_label(o)))
            else:
                res.append((s_c, predicate, str(o)))
        return res

    def neighbors(self, entity: str):
        e_c = self._canon_entity(entity)
        e_uri = self._uri_for_entity(e_c)
        outs = []
        ins = []
        for _, p, o in self.g.triples((e_uri, None, None)):
            p_str = next((k for k, v in self.PROP.items() if v == p), str(p).split("/")[-1])
            outs.append((e_c, p_str, self._get_label(o) if isinstance(o, URIRef) else str(o)))
        for s, p, _ in self.g.triples((None, None, e_uri)):
            p_str = next((k for k, v in self.PROP.items() if v == p), str(p).split("/")[-1])
            ins.append((self._get_label(s), p_str, e_c))
        return {"out": outs, "in": ins}

    def graph_snapshot(self):
        # 返回节点、边计数与若干示例
        nodes = set(s for s, _, _ in self.g) | set(o for _, _, o in self.g if isinstance(o, URIRef))
        edges = list(self.g.triples((None, None, None)))
        return {"nodes_count": len(nodes), "edges_count": len(edges)}

    # ----------------------------
    # 可选：SPARQL 查询
    # ----------------------------
    def sparql(self, query: str):
        """运行 SPARQL 查询（字符串），返回结果列表。需要熟悉 rdflib 语法。"""
        try:
            qres = self.g.query(query)
            rows = []
            for row in qres:
                if isinstance(row, (tuple, list)):
                    rows.append(tuple(str(x) for x in row))
                else:
                    # 例如 ASK 查询可能返回布尔或单值
                    rows.append((str(row),))
            return rows
        except Exception as e:
            return {"error": str(e)}

    def export_ttl(self, path: str):
        """将图谱以 Turtle 格式导出到指定路径。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.g.serialize(destination=path, format="turtle")

    def export_png(self, path: str, max_edges: int | None = 400):
        """将图谱以 PNG 可视化导出（基于 networkx + matplotlib）。
        - 为避免过密，可限制最大边数；超出时随机抽样部分边。
        - 使用 Agg 后端，无需显示器环境。
        """
        import random
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib import font_manager as fm
            import networkx as nx
        except Exception as e:  # noqa: BLE001
            return {"error": f"可视化依赖缺失: {e}"}

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

        # 构建 networkx 图
        G = nx.DiGraph()

        # 先加入按类型的独立节点（即便没有业务边也可展示）
        typed_classes = [self.Gate, self.Runway, self.Aircraft, self.Device]
        for clazz in typed_classes:
            for node in self.g.subjects(RDF.type, clazz):
                if isinstance(node, URIRef):
                    G.add_node(self._get_label(node))

        # 收集边（尽量过滤非业务谓词冗余）
        edges = []
        for s, p, o in self.g.triples((None, None, None)):
            # 仅展示带 label 的主体与对象
            s_label = self._get_label(s) if isinstance(s, URIRef) else str(s)
            if isinstance(o, URIRef):
                o_label = self._get_label(o)
            else:
                # 字面量可跳过，避免噪声太多
                continue
            # 谓词转中文名（若有）
            p_str = next((k for k, v in self.PROP.items() if v == p), str(p).split("/")[-1])
            # 只展示主要业务谓词
            if p_str in {"使用跑道", "分配停机位", "到达停机位", "当前停机位", "目标停机位", "待命位置"}:
                edges.append((s_label, o_label, p_str))

        if max_edges is not None and len(edges) > max_edges:
            edges = random.sample(edges, max_edges)

        # 加入点与边
        for u, v, p_str in edges:
            G.add_node(u)
            G.add_node(v)
            G.add_edge(u, v, label=p_str)

        # 节点颜色按类型区分
        def node_color(name: str) -> str:
            if name.startswith("飞机"):
                return "#1f77b4"  # 蓝
            if name.startswith("停机位"):
                return "#2ca02c"  # 绿
            if name.startswith("跑道"):
                return "#ff7f0e"  # 橙
            return "#7f7f7f"      # 灰

        pos = nx.spring_layout(G, k=0.9, seed=42)
        plt.figure(figsize=(12, 8))
        nx.draw_networkx_nodes(G, pos, node_color=[node_color(n) for n in G.nodes()], node_size=600)
        nx.draw_networkx_labels(G, pos, font_family=(chosen_family or "sans-serif"), font_size=8)
        nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', width=1.0, alpha=0.8)
        edge_labels = {(u, v): d.get('label', '') for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, font_family=(chosen_family or "sans-serif"))

        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()}


