#!/usr/bin/env python
"""
一键运行端到端流程：
1) 原始 CSV 预处理 -> JSONL 文本
2) 批量三元组抽取
3) 打包训练 JSON
4) 构建 RDF 知识图谱并导出 TTL
"""

import os
import argparse
import random
import numpy as np
import torch

from exp.exp_main import Exp_main


def main():
	fix_seed = 42
	random.seed(fix_seed)
	np.random.seed(fix_seed)
	torch.manual_seed(fix_seed)

	cwd = os.getcwd()
	default_root = cwd
	default_input_csv = os.path.join(default_root, 'data_provider', '海天杯-ST_Job_训练集.csv')
	default_out_dir = os.path.join(default_root, 'data_provider')
	default_texts_jsonl = os.path.join(default_out_dir, 'train_texts.jsonl')
	default_triples_jsonl = os.path.join(default_out_dir, 'train_triples.jsonl')
	default_train_jsonl = os.path.join(default_out_dir, 'train_for_model.jsonl')
	default_ttl_out = os.path.join(default_root, 'resoterd', 'output', 'kg.ttl')
	default_png_out = os.path.join(default_root, 'resoterd', 'output', 'kg.png')

	parser = argparse.ArgumentParser(description='Run KG Pipeline')
	parser.add_argument('--root', default=default_root, help='项目根路径')
	parser.add_argument('--input_csv', default=default_input_csv, help='原始 CSV 文件路径')
	parser.add_argument('--out_dir', default=default_out_dir, help='中间/输出目录')
	parser.add_argument('--texts_jsonl', default=default_texts_jsonl, help='文本 JSONL 输出路径')
	parser.add_argument('--triples_jsonl', default=default_triples_jsonl, help='三元组 JSONL 输出路径')
	parser.add_argument('--train_jsonl', default=default_train_jsonl, help='训练 JSONL 输出路径')
	parser.add_argument('--ttl_out', default=default_ttl_out, help='TTL 导出路径')
	parser.add_argument('--png_out', default=default_png_out, help='PNG 可视化导出路径')
	parser.add_argument('--limit_kg', type=int, default=None, help='用于构建 KG 的最大记录数（可选）')

	# 与 Exp_Basic 兼容的占位参数
	parser.add_argument('--use_gpu', action='store_true', help='是否使用 GPU（可选）')
	parser.add_argument('--gpu', type=int, default=0)
	parser.add_argument('--use_multi_gpu', action='store_true')
	parser.add_argument('--devices', type=str, default='0')

	args = parser.parse_args()

	exp = Exp_main(args)
	result = exp.run()
	print('\n=== Pipeline Finished ===')
	print('texts_jsonl :', result['texts_jsonl'])
	print('triples_jsonl:', result['triples_jsonl'])
	print('train_jsonl  :', result['train_jsonl'])
	print('ttl_out      :', result['ttl_out'])
	print('png_out      :', result['png_out'])
	print('kg_snapshot  :', result['kg_snapshot'])


if __name__ == '__main__':
	main()

