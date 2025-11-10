#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: zy
# Date: 2025-11-04 20:40:03
# LastEditTime: 2025-11-04 20:55:00
# Description: 生成分解器SFT数据的脚本

"""
生成“分解器”小模型的指令微调训练集（SFT）。

- 输入：事件文本（支持 .txt 每行一条，或 .jsonl 的 text/event/input 字段）
- 输出：instruction/input/output 结构的 JSONL，output 为 JSON 字符串，包含
  entities / applicable_rules / potential_conflicts / notes

示例：
    python scripts/prepare_decompose_sft.py \
        --events_file ./tests/events_sample.txt \
        --out_jsonl ./data_provider/sft_decompose.jsonl \
        --print_preview 3

可选：
- 自定义 instruction： --instruction "..."
- 限制最大事件条数： --max_events 100
"""

from __future__ import annotations

import os
import sys
import logging as _logging
import argparse


def _ensure_sys_path():
    # 允许从仓库根目录或 scripts/ 运行
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_sys_path()

from data_provider.data_loader import load_events_from_file  # noqa: E402
from utils import build_decomposition_samples_from_events, save_jsonl  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT data for Decomposer (split event into sub-problems)")
    parser.add_argument("--events_file", required=True, help="事件文件 .txt(每行一条) 或 .jsonl(text/event/input 字段)")
    parser.add_argument("--out_jsonl", default=None, help="输出 SFT JSONL 路径，默认写入 data_provider/sft_decompose.jsonl")
    parser.add_argument("--instruction", default=None, help="覆盖默认的 instruction 文本（可选）")
    parser.add_argument("--max_events", type=int, default=None, help="最多处理的事件条数（可选）")
    parser.add_argument("--print_preview", type=int, default=0, help="在控制台预览前N条样本（中文建议先设置UTF-8控制台）")

    args = parser.parse_args()

    if not os.path.isfile(args.events_file):
        raise FileNotFoundError(args.events_file)

    events = load_events_from_file(args.events_file)
    if not events:
        raise RuntimeError(f"未从 {args.events_file} 读取到事件文本")
    if args.max_events is not None:
        events = list(events)[: max(0, int(args.max_events))]
    if not _logging.getLogger().handlers:
        _logging.basicConfig(level=_logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _logging.info(f"[DATA] 事件条数: {len(events)}")

    out_path = args.out_jsonl or os.path.join("data_provider", "sft_decompose.jsonl")

    samples = build_decomposition_samples_from_events(events, instruction=args.instruction)
    save_jsonl(samples, out_path)

    _logging.info(f"[OUT] 写出: {out_path}")
    _logging.info(f"[OUT] 样本数: {len(samples)}")

    if args.print_preview and len(samples) > 0:
        n = max(1, int(args.print_preview))
        _logging.info(f"\n[PREVIEW] 仅展示前 {n} 条 input/output（如乱码，请先 chcp 65001 或设置 UTF-8）：")
        for i, s in enumerate(samples[:n], 1):
            inp = s.get("input", "")
            out = s.get("output", "")
            if len(inp) > 800:
                inp = inp[:800] + "... (truncated)"
            if len(out) > 800:
                out = out[:800] + "... (truncated)"
            _logging.info(f"\n--- SAMPLE #{i} INPUT ---\n" + inp)
            _logging.info(f"\n--- SAMPLE #{i} OUTPUT(JSON) ---\n" + out)
    _logging.info("done.")


if __name__ == "__main__":
    main()
