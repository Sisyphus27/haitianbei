# -*- coding: utf-8 -*-
"""
快速将 judge 流式输出（JSONL）转换为“任务一评分规则”格式的 JSON 文件。

默认处理路径：
- 源:   D:\\WorkSpace\\haitianbei\\results\\model_outputs\\judge\\stream_20251114_231005.jsonl
- 目标: D:\\WorkSpace\\haitianbei\\results\\task1\\stream_20251114_231005.json

可通过命令行参数自定义：
  python scripts/export_task1_from_judge.py --src <source_jsonl> --out <output_json>

说明：
- 直接复用 Exp_main.export_task1_results 方法，但避免执行 Exp_main.__init__ 的重型初始化，
  通过 __new__ 创建实例并手动设置必要属性。
"""
import argparse
import os
import sys

# 确保可以 import 顶层包
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from exp.exp_main import Exp_main  # type: ignore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        dest="source_jsonl",
        type=str,
        default=os.path.join(
            ROOT,
            "results",
            "model_outputs",
            "judge",
            "stream_20251114_231005.jsonl",
        ),
        help="judge 阶段的 JSONL 源文件路径",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        type=str,
        default=os.path.join(
            ROOT,
            "results",
            "task1",
            "stream_20251114_231005.json",
        ),
        help="任务一结果输出 JSON 文件路径",
    )
    args = parser.parse_args()

    # 以最小代价构造一个 Exp_main 实例：跳过 __init__ 重型初始化
    exp = Exp_main.__new__(Exp_main)  # type: ignore
    # export_task1_results 仅依赖 results_out_dir（以及 _judge_out_file 在未传 src 时），这里设置为目标目录
    exp.results_out_dir = os.path.join(ROOT, "results", "task1")

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    out_file = Exp_main.export_task1_results(
        exp,
        source_jsonl=args.source_jsonl,
        out_path=args.out_path,
    )
    if not out_file:
        print("[TASK1] 导出失败，请检查源文件是否存在且格式正确:", args.source_jsonl)
        sys.exit(1)
    print("[TASK1] 导出完成 ->", out_file)


if __name__ == "__main__":
    main()
