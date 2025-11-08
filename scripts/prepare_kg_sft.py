#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成指令微调训练集（SFT）。支持两种来源：

模式 A：基于 KG 上下文 + 事件文本（原有功能）
- 连接 Neo4j，读取当前 KG 状态（可选聚焦若干实体），将其文本化；
- 从事件文件读取事件文本；
- 拼接 规则文档（可选）+ KG 状态 + 事件文本 为 input；
- 可选择启发式自动生成输出（结论/依据/建议）作为弱标注，或输出人工标注模板；

模式 B：基于 train_triples.jsonl 的资源可用性对照样本（新功能）
- 从 train_triples.jsonl 读取每条文本及其三元组；
- 抽取文本涉及的资源（停机位/跑道/设备），构造正/负样本：
    * 正样本：所有涉及的资源均可用；
    * 负样本：按 per_resource（逐个资源置不可用）或 all_resources（全部置不可用）生成；

通用：保存为符合 instruction/input/output 的 JSONL，可直接用于训练。

示例（模式A，基于KG）：
    python scripts/prepare_kg_sft.py \
            --events_file ./tests/events_sample.txt \
            --rules_md_path ./tests/rules_sample.md \
            --out_jsonl ./data_provider/sft_with_kg.jsonl \
            --focus_entities 飞机A001,停机位14 \
            --max_edges 200

示例（模式B，基于triples）：
    python scripts/prepare_kg_sft.py \
            --from_triples ./data_provider/train_triples.jsonl \
            --negative_mode per_resource \
            --out_jsonl ./data_provider/sft_from_triples.jsonl

注意：
- 模式A默认启用自动弱标注（启发式冲突检测），如需人工标注模板请添加 --no_auto_label。
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

# 基于 triples 的训练数据生成（资源可用性对照样本）
from utils import (  # noqa: E402
    build_samples_from_train_triples,
)


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT data (with KG context or from triples)")
    # 互斥输入：A) 基于事件+KG；B) 基于 triples
    parser.add_argument("--events_file", default=None, help="模式A：事件文件 .txt(每行一条) 或 .jsonl(text/event/input 字段)")
    parser.add_argument("--from_triples", default=None, help="模式B：train_triples.jsonl 的路径，基于资源可用性构造样本")

    parser.add_argument("--out_jsonl", default=None, help="输出 SFT JSONL 路径；若省略，将根据模式给出默认文件名")
    parser.add_argument("--rules_md_path", default=None, help="规则文档 Markdown 路径（可选）")
    parser.add_argument("--focus_entities", default=None, help="关注实体列表，逗号分隔，例如：飞机A001,停机位14")
    parser.add_argument("--max_edges", type=int, default=200, help="KG上下文最多绘制的边数（文本行数）")
    parser.add_argument("--no_auto_label", action="store_true", help="禁用启发式弱标注，输出人工标注模板")
    parser.add_argument("--allow_offline", action="store_true", help="允许在无法连接 Neo4j 时离线生成（KG上下文置为空占位）")
    parser.add_argument("--print_preview", type=int, default=0, help="在控制台预览前N条样本的 input 段（中文控制台建议先设置UTF-8编码）")

    # 模式B（triples）相关参数
    parser.add_argument("--negative_mode", choices=["per_resource", "all_resources"], default="per_resource",
                        help="负样本构造策略：per_resource 为逐个资源置不可用，all_resources 为全部资源不可用")
    parser.add_argument("--no_positive", action="store_true", help="不生成正样本（默认会包含正样本）")
    parser.add_argument("--max_negatives_per_event", type=int, default=None,
                        help="每条事件最多生成的负样本数量（仅 per_resource 生效）")
    parser.add_argument("--instruction", default=None, help="覆盖默认的 instruction 文本（可选）")

    # Neo4j 连接参数（支持环境变量覆盖）
    parser.add_argument("--neo4j-uri", dest="neo4j_uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", dest="neo4j_user", default=os.environ.get("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", dest="neo4j_password", default=os.environ.get("NEO4J_PASSWORD", "neo4j"))
    parser.add_argument("--neo4j-database", dest="neo4j_database", default=os.environ.get("NEO4J_DATABASE", "neo4j"))

    args = parser.parse_args()

    # 分支：模式B（from_triples）或 模式A（events+KG）
    if args.from_triples:
        src = os.path.abspath(args.from_triples)
        if not os.path.isfile(src):
            raise FileNotFoundError(src)
        out_path = args.out_jsonl or os.path.join("data_provider", "sft_from_triples.jsonl")
        print("[TRIPLES] 源:", src)
        print("[TRIPLES] 模式:", args.negative_mode, " include_positive=", (not args.no_positive))
        samples = build_samples_from_train_triples(
            src,
            include_positive=(not args.no_positive),
            negative_mode=args.negative_mode,
            max_negatives_per_event=args.max_negatives_per_event,
            instruction=args.instruction,
        )
        save_jsonl(samples, out_path)
        print("[OUT] 写出:", out_path)
        print("[OUT] 样本数:", len(samples))
        if args.print_preview and len(samples) > 0:
            n = max(1, int(args.print_preview))
            print("\n[PREVIEW] 仅展示前", n, "条 input（如出现乱码，请先在 PowerShell 执行 chcp 65001 或设置 $OutputEncoding 为 UTF-8）：")
            for i, s in enumerate(samples[:n], 1):
                inp = s.get("input", "")
                if len(inp) > 1000:
                    inp = inp[:1000] + "... (truncated)"
                print(f"\n--- SAMPLE #{i} INPUT ---\n" + inp)
        print("done.")
        return

    # 模式A（events + KG 上下文）
    if not args.events_file:
        raise RuntimeError("请提供 --events_file 或 --from_triples 其中之一")

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

    out_path = args.out_jsonl or os.path.join("data_provider", "sft_with_kg.jsonl")
    samples = build_kg_sft_samples_from_events(
        kg=kg,
        events=events,
        rules_md_path=args.rules_md_path,
        focus_entities=focus,
        max_edges=max(10, int(args.max_edges)),
        auto_label=(not args.no_auto_label),
    )
    save_jsonl(samples, out_path)
    print("[OUT] 写出:", out_path)
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
