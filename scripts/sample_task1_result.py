# -*- coding: utf-8 -*-
"""
从 task1_result.json 中随机抽样指定数量的样本。

默认处理路径：
- 输入: results/task1/task1_result.json
- 输出: results/task1/task1_result_sample_100.json

可通过命令行参数自定义：
  python scripts/sample_task1_result.py --input <input_json> --output <output_json> --size <sample_size>

说明：
- 使用 random.sample() 进行无重复随机抽样
- 支持设置随机种子以便复现结果
- 如果抽样数量大于总样本数，将采样所有样本
"""
import argparse
import json
import os
import random
import sys

# 确保可以 import 顶层包
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="从 task1_result.json 中随机抽样指定数量的样本"
    )
    parser.add_argument(
        "--input",
        "-i",
        dest="input_path",
        type=str,
        default=os.path.join(ROOT, "results", "task1", "task1_result.json"),
        help="输入 JSON 文件路径",
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="output_path",
        type=str,
        default=os.path.join(ROOT, "results", "task1", "task1_result_sample_100.json"),
        help="输出 JSON 文件路径",
    )
    parser.add_argument(
        "--size",
        "-n",
        dest="sample_size",
        type=int,
        default=100,
        help="抽样数量（默认: 100）",
    )
    parser.add_argument(
        "--seed",
        dest="random_seed",
        type=int,
        default=None,
        help="随机种子（可选，用于复现结果）",
    )
    args = parser.parse_args()

    # 设置随机种子
    if args.random_seed is not None:
        random.seed(args.random_seed)
        print(f"[抽样] 设置随机种子: {args.random_seed}")

    # 读取输入文件
    if not os.path.exists(args.input_path):
        print(f"[错误] 输入文件不存在: {args.input_path}")
        sys.exit(1)

    print(f"[抽样] 正在读取文件: {args.input_path}")
    try:
        with open(args.input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[错误] JSON 文件格式错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[错误] 读取文件失败: {e}")
        sys.exit(1)

    total_samples = len(data)
    print(f"[抽样] 总样本数: {total_samples}")

    # 确定实际抽样数量
    actual_sample_size = min(args.sample_size, total_samples)
    if args.sample_size > total_samples:
        print(
            f"[警告] 抽样数量 ({args.sample_size}) 大于总样本数 ({total_samples})，"
            f"将采样所有 {total_samples} 条样本"
        )

    # 随机抽样
    if actual_sample_size == total_samples:
        sampled_data = data
        print(f"[抽样] 采样所有 {total_samples} 条样本")
    else:
        sampled_data = random.sample(data, actual_sample_size)
        print(f"[抽样] 随机采样 {actual_sample_size} 条样本")

    # 保存结果
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    try:
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(sampled_data, f, ensure_ascii=False, indent=2)
        print(f"[抽样] 保存完成 -> {args.output_path}")
    except Exception as e:
        print(f"[错误] 保存文件失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

