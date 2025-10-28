#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成包含 KG 上下文的指令微调训练集（SFT）。

功能：
- 连接 Neo4j，读取当前 KG 状态（可选聚焦若干实体），将其文本化；
- 从事件文件读取事件文本；
- 拼接 规则文档（可选）+ KG 状态 + 事件文本 为 input；
- 可选择启发式自动生成输出（结论/依据/建议）作为弱标注，或输出人工标注模板；
- 保存为符合 instruction/input/output 的 JSONL，可直接用于 run.py --mode train。

示例：
  python scripts/prepare_kg_sft.py \
      --events_file ./resoterd/prompts/test.txt \
      --rules_md_path ./resoterd/prompts/answer.txt \
      --out_jsonl ./data_provider/sft_with_kg.jsonl \
      --focus_entities 飞机A001,停机位14 \
      --max_edges 200

注意：
- 默认启用自动弱标注（启发式冲突检测），如需人工标注模板请添加 --no_auto_label。
- Neo4j 连接参数可通过命令行或环境变量/ .env 注入（NEO4J_URI/USER/PASSWORD/DATABASE）。
"""

from __future__ import annotations

import os
import sys
import argparse


def _ensure_sys_path():
    # 允许从仓库根目录或 scripts/ 运行
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_sys_path()

from data_provider.data_loader import (  # noqa: E402
    Dataset_KG,
    load_events_from_file,
    build_kg_sft_samples_from_events,
    save_jsonl,
)


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT data with KG context for conflict judgment")
    parser.add_argument("--events_file", required=True, help="事件文件：.txt(每行一条) 或 .jsonl(text/event/input 字段)")
    parser.add_argument("--out_jsonl", default=os.path.join("data_provider", "sft_with_kg.jsonl"), help="输出 SFT JSONL 路径")
    parser.add_argument("--rules_md_path", default=None, help="规则文档 Markdown 路径（可选）")
    parser.add_argument("--focus_entities", default=None, help="关注实体列表，逗号分隔，例如：飞机A001,停机位14")
    parser.add_argument("--max_edges", type=int, default=200, help="KG上下文最多绘制的边数（文本行数）")
    parser.add_argument("--no_auto_label", action="store_true", help="禁用启发式弱标注，输出人工标注模板")
    parser.add_argument("--allow_offline", action="store_true", help="允许在无法连接 Neo4j 时离线生成（KG上下文置为空占位）")
    parser.add_argument("--print_preview", type=int, default=0, help="在控制台预览前N条样本的 input 段（中文控制台建议先设置UTF-8编码）")

    # Neo4j 连接参数（支持环境变量覆盖）
    parser.add_argument("--neo4j-uri", dest="neo4j_uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", dest="neo4j_user", default=os.environ.get("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", dest="neo4j_password", default=os.environ.get("NEO4J_PASSWORD", "neo4j"))
    parser.add_argument("--neo4j-database", dest="neo4j_database", default=os.environ.get("NEO4J_DATABASE", "neo4j"))

    args = parser.parse_args()

    focus = None
    if args.focus_entities:
        focus = [s.strip() for s in args.focus_entities.split(",") if s.strip()]

    # 兼容 NEO4J_AUTH=neo4j/<password>
    auth = os.environ.get("NEO4J_AUTH")
    if auth and (args.neo4j_user == "neo4j" and args.neo4j_password == "neo4j") and "/" in auth:
        u, p = auth.split("/", 1)
        if u:
            args.neo4j_user = u
        if p:
            args.neo4j_password = p

    print("[KG] connecting:", args.neo4j_uri)
    kg = None
    try:
        kg = Dataset_KG(
            root_path=os.getcwd(),
            load_data=False,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            neo4j_database=args.neo4j_database,
        )
    except Exception as e:
        if args.allow_offline:
            print("[KG] 连接失败，进入离线模式：", e)
            kg = None
        else:
            raise

    events = load_events_from_file(args.events_file)
    if not events:
        raise RuntimeError(f"未从 {args.events_file} 读取到事件文本")
    print(f"[DATA] 事件条数: {len(events)}")

    samples = build_kg_sft_samples_from_events(
        kg=kg,
        events=events,
        rules_md_path=args.rules_md_path,
        focus_entities=focus,
        max_edges=max(10, int(args.max_edges)),
        auto_label=(not args.no_auto_label),
    )
    save_jsonl(samples, args.out_jsonl)
    print("[OUT] 写出:", args.out_jsonl)
    print("[OUT] 样本数:", len(samples))
    if args.print_preview and len(samples) > 0:
        n = max(1, int(args.print_preview))
        print("\n[PREVIEW] 仅展示前", n, "条 input（如出现乱码，请先在 PowerShell 执行 chcp 65001 或设置 $OutputEncoding 为 UTF-8）：")
        for i, s in enumerate(samples[:n], 1):
            inp = s.get("input", "")
            # 适度截断避免控制台过长
            if len(inp) > 1000:
                inp = inp[:1000] + "... (truncated)"
            print(f"\n--- SAMPLE #{i} INPUT ---\n" + inp)
    print("done.")


if __name__ == "__main__":
    main()
