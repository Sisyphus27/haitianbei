'''
Author: zy
Date: 2025-10-22 17:35:46
LastEditTime: 2025-10-23 19:49:44
LastEditors: zy
Description: 
FilePath: \haitianbei\exp\exp_main.py

'''
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

    def _build_model(self):
        """占位，满足基类在构造时的要求。"""
        import torch.nn as nn
        return nn.Identity(), []

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

        # 4) 构建知识图谱（RDF）
        kg = Dataset_KG(self.root, load_data=False)
        used = 0
        for obj in self._iter_jsonl(self.triples_jsonl):
            triples = obj.get('triples', [])
            if triples:
                kg.update_with_triples([tuple(t) for t in triples])
            used += 1
            if self.limit_kg is not None and isinstance(self.limit_kg, int) and used >= self.limit_kg:
                break
        snap = kg.graph_snapshot()
        print(f"[4/6] 知识图谱构建完成: 节点数={snap['nodes_count']}, 边数={snap['edges_count']}")

        # 5) 导出 TTL
        ttl_out_path = getattr(self.args, 'ttl_out', None) or os.path.join(self.root, 'resoterd', 'output', 'kg.ttl')
        kg.export_ttl(ttl_out_path)
        self.ttl_out = ttl_out_path
        print(f"[5/6] 图谱 TTL 已导出: {self.ttl_out}")

        # 6) 可视化 PNG 导出
        png_out_path = getattr(self.args, 'png_out', None) or os.path.join(self.root, 'resoterd', 'output', 'kg.png')
        kg.export_png(png_out_path)
        self.png_out = png_out_path
        print(f"[6/6] 图谱 PNG 已导出: {self.png_out}")

        return {
            'texts_jsonl': self.texts_jsonl,
            'triples_jsonl': self.triples_jsonl,
            'train_jsonl': self.train_jsonl,
            'ttl_out': self.ttl_out,
            'png_out': self.png_out,
            'kg_snapshot': snap,
        }
    
