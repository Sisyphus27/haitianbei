r'''
Author: zy
Date: 2025-10-22 17:35:46
LastEditTime: 2025-10-23 19:49:44
LastEditors: zy
Description: 
FilePath: haitianbei/exp/exp_main.py

'''
# 兼容直接"运行本文件"的调试方式：确保项目根目录在 sys.path
# 这样即使用 "python exp/exp_main.py" 启动，也能导入顶层包（exp、utils、data_provider）。
import os as _os
import sys as _sys
if __package__ is None or __package__ == "":
    _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))

from exp.exp_basic import Exp_Basic
from utils.origindata_preprocessing import convert_csv_to_jsonl
from utils.batch_extract_triples import run as run_extract
from utils.pack_training_json import run as run_pack
from data_provider.data_loader import Dataset_KG
import os
import json
from typing import Optional

class Exp_main(Exp_Basic):
    def __init__(self, args):
        super(Exp_main, self).__init__(args)
        # 根路径
        self.root = getattr(args, 'root', os.getcwd())
        # 输入 CSV（原始数据）
        self.input_csv = getattr(args, 'input_csv', os.path.join(self.root, 'data_provider', '海天杯-ST_Job_训练集.csv'))
        # 中间与输出
        self.out_dir = getattr(args, 'out_dir', os.path.join(self.root, 'data_provider'))
        # 以下产物路径在步骤完成后再赋值（初始为 None）
        self.texts_jsonl = None
        self.triples_jsonl = None
        self.train_jsonl = None
        self.ttl_out = None
        self.png_out = None
        # 限制用于构建 KG 的记录数（可选，便于快速调试）
        self.limit_kg = getattr(args, 'limit_kg', None)
        # 可视化开关与参数
        # 每多少条记录导出一次快照（0 表示不导出）
        self.visualize_every = int(getattr(args, 'visualize_every', 0) or 0)
        self.visualize_dir = getattr(args, 'visualize_dir', os.path.join(self.root, 'resoterd', 'output', 'kg_steps'))
        self.visualize_max_edges = getattr(args, 'visualize_max_edges', 300)
        self.visualize_limit = getattr(args, 'visualize_limit', 50)  # 最多导出多少张快照，避免过多文件
        self.visualize_clean = bool(getattr(args, 'visualize_clean', False))
        # Neo4j 连接参数
        self.neo4j_uri = getattr(args, 'neo4j_uri', None)
        self.neo4j_user = getattr(args, 'neo4j_user', None)
        self.neo4j_password = getattr(args, 'neo4j_password', None)
        self.neo4j_database = getattr(args, 'neo4j_database', None)
        # 离线模式：跳过 KG 构建
        self.skip_kg = getattr(args, 'skip_kg', False)
        # 是否在构建前重置 KG（保留固定节点）
        self.reset_kg = bool(getattr(args, 'reset_kg', False))

    def _build_model(self):
        """占位，满足基类在构造时的要求（torch 可选）。"""
        try:
            import torch.nn as nn  # type: ignore
            return nn.Identity(), []
        except Exception:
            class DummyModel:
                def to(self, device):
                    return self
            return DummyModel(), []

    # 读取 JSONL 行迭代器
    def _iter_jsonl(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def run(self):
        """从原始 CSV 开始，一键跑通：预处理 -> 批量抽取 -> 打包训练 JSON -> 构建 KG -> 导出 TTL"""
        # 1) 原始 CSV -> JSONL 文本
        texts_jsonl_path = getattr(self.args, 'texts_jsonl', None) or os.path.join(self.out_dir, 'train_texts.jsonl')
        total_texts = convert_csv_to_jsonl(self.input_csv, texts_jsonl_path)
        self.texts_jsonl = texts_jsonl_path
        print(f"[1/6] 预处理完成: {total_texts} 行 -> {self.texts_jsonl}")

        # 2) 批量抽取三元组
        triples_jsonl_path = getattr(self.args, 'triples_jsonl', None) or os.path.join(self.out_dir, 'train_triples.jsonl')
        total_triples = run_extract(self.texts_jsonl, triples_jsonl_path)
        self.triples_jsonl = triples_jsonl_path
        print(f"[2/6] 三元组抽取完成: {total_triples} 行 -> {self.triples_jsonl}")

        # 3) 打包训练 JSON（可供后续轻量模型使用）
        train_jsonl_path = getattr(self.args, 'train_jsonl', None) or os.path.join(self.out_dir, 'train_for_model.jsonl')
        total_packed = run_pack(self.texts_jsonl, self.triples_jsonl, train_jsonl_path)
        self.train_jsonl = train_jsonl_path
        print(f"[3/6] 训练 JSON 打包完成: {total_packed} 行 -> {self.train_jsonl}")

        # 4) 构建知识图谱（Neo4j）或离线跳过
        if self.skip_kg:
            print('[4/6] 跳过知识图谱构建（离线模式）')
            snap = {'nodes_count': 0, 'edges_count': 0}
            self.ttl_out = None
            self.png_out = None
        else:
            kg = Dataset_KG(
                self.root, load_data=False,
                neo4j_uri=self.neo4j_uri,
                neo4j_user=self.neo4j_user,
                neo4j_password=self.neo4j_password,
                neo4j_database=self.neo4j_database,
            )
            # 按需重置 KG（避免上一轮历史数据影响本次快照）
            if self.reset_kg:
                try:
                    kg.reset_graph(keep_fixed=True)
                    print("[4.0] 已重置 KG：保留固定节点，清理历史动态数据")
                except Exception as _e:
                    print("[4.0] 重置 KG 失败，继续构建：", _e)
            used = 0
            snap_count = 0
            # 调试：打印可视化相关参数
            print(f"[4.0] 可视化参数: every={self.visualize_every}, limit={self.visualize_limit}, max_edges={self.visualize_max_edges}, dir={self.visualize_dir}")
            if self.visualize_every > 0:
                os.makedirs(self.visualize_dir, exist_ok=True)
                # 可选：清理旧的 PNG，避免目录里混有历史文件
                if self.visualize_clean:
                    try:
                        import glob
                        for fp in glob.glob(os.path.join(self.visualize_dir, '*.png')):
                            try:
                                os.remove(fp)
                            except Exception:
                                pass
                        print(f"[4.0] 已清理旧快照: {self.visualize_dir}")
                    except Exception as _:
                        pass
            snapshot_disabled = False
            for obj in self._iter_jsonl(self.triples_jsonl):
                triples = obj.get('triples', [])
                if triples:
                    kg.update_with_triples([tuple(t) for t in triples])
                used += 1
                # 动态可视化快照
                if self.visualize_every > 0 and not snapshot_disabled:
                    if (used % self.visualize_every == 0) and (snap_count < int(self.visualize_limit)):
                        step_png = os.path.join(self.visualize_dir, f"step_{used:05d}.png")
                        ret = kg.export_png(step_png, max_edges=int(self.visualize_max_edges))
                        snap_count += 1
                        # 调试输出，便于确认快照数量与文件是否落盘
                        if isinstance(ret, dict) and 'error' in ret:
                            print(f"[4.x] 可视化快照失败 {snap_count}/{self.visualize_limit}: {step_png} -> {ret['error']}")
                            # 若导入或保存依赖失败，禁用后续快照，避免重复噪声
                            snapshot_disabled = True
                            trace = ret.get('trace') if isinstance(ret, dict) else None
                            if trace:
                                first_line = trace.strip().splitlines()[-1] if trace.strip().splitlines() else ''
                                print(f"[4.x] 失败追踪: {first_line}")
                        else:
                            size = 0
                            try:
                                size = int(ret.get('size', 0)) if isinstance(ret, dict) else (os.path.getsize(step_png) if os.path.exists(step_png) else 0)
                            except Exception:
                                size = 0
                            print(f"[4.x] 可视化快照 {snap_count}/{self.visualize_limit}: {step_png} (size={size} bytes)")
                if self.limit_kg is not None and isinstance(self.limit_kg, int) and used >= self.limit_kg:
                    break
            snap = kg.graph_snapshot()
            print(f"[4/6] 知识图谱构建完成: 节点数={snap['nodes_count']}, 边数={snap['edges_count']}")

            # 5) 导出（占位 TTL 文件，说明如何从 Neo4j 导出）
            ttl_out_path = getattr(self.args, 'ttl_out', None) or os.path.join(self.root, 'resoterd', 'output', 'kg.ttl')
            kg.export_ttl(ttl_out_path)
            self.ttl_out = ttl_out_path
            print(f"[5/6] 图谱 TTL 已导出: {self.ttl_out}")

            # 6) 可视化 PNG 导出
            png_out_path = getattr(self.args, 'png_out', None) or os.path.join(self.root, 'resoterd', 'output', 'kg.png')
            final_ret = kg.export_png(png_out_path)
            self.png_out = png_out_path
            if isinstance(final_ret, dict) and 'error' in final_ret:
                print(f"[6/6] 图谱 PNG 导出失败: {final_ret['error']}")
            else:
                print(f"[6/6] 图谱 PNG 已导出: {self.png_out}")

        return {
            'texts_jsonl': self.texts_jsonl,
            'triples_jsonl': self.triples_jsonl,
            'train_jsonl': self.train_jsonl,
            'ttl_out': self.ttl_out,
            'png_out': self.png_out,
            'kg_snapshot': snap,
        }
    
