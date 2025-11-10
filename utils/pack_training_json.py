r'''
Author: zy
Date: 2025-10-23 10:43:28
LastEditTime: 2025-10-23 10:43:32
LastEditors: zy
Description: 
FilePath: \haitianbei\utils\pack_training_json.py

'''
from __future__ import annotations

# 将文本与抽取到的三元组合并，生成可用于训练的 JSONL：
# {id, text, text_norm, triples}
#
# - text: 原始文本（来自 train_texts.jsonl）
# - text_norm: 规范化标点（全角->半角），便于后续匹配/训练
# - triples: [[subject, predicate, object], ...]（来自规则抽取器批处理输出）
#
# 用法：
# python -m utils.pack_training_json \
#   --texts d:/WorkSpace/haitianbei/data_provider/train_texts.jsonl \
#   --triples d:/WorkSpace/haitianbei/data_provider/train_triples.jsonl \
#   --output d:/WorkSpace/haitianbei/data_provider/train_for_model.jsonl

import argparse
import json
from typing import Dict, List
import logging as _logging


REPL = {
    "（": "(",
    "）": ")",
    "，": ",",
    "：": ":",
    "；": ";",
}


def normalize_text_punct(s: str) -> str:
    t = s
    for k, v in REPL.items():
        t = t.replace(k, v)
    # 统一空白
    t = " ".join(t.split())
    return t


def load_jsonl(path: str) -> List[dict]:
    arr = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            arr.append(json.loads(line))
    return arr


def run(texts_path: str, triples_path: str, output_path: str) -> int:
    texts = load_jsonl(texts_path)
    triples = load_jsonl(triples_path)

    triples_map: Dict[int, List[List[str]]] = {}
    for obj in triples:
        _id = obj.get("id")
        if _id is None:
            continue
        try:
            key = int(_id)
        except Exception:  # noqa: BLE001
            continue
        tps = obj.get("triples", [])
        triples_map[key] = tps

    total = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for obj in texts:
            _id_raw = obj.get("id")
            if _id_raw is None:
                continue
            try:
                _id = int(_id_raw)
            except Exception:  # noqa: BLE001
                continue
            text = obj.get("text", "")
            tps = triples_map.get(_id, [])
            rec = {
                "id": _id,
                "text": text,
                "text_norm": normalize_text_punct(text),
                "triples": tps,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += 1
    return total


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="打包训练 JSON")
    parser.add_argument("--texts", required=True, help="文本 JSONL: 每行 {id, text}")
    parser.add_argument("--triples", required=True, help="三元组 JSONL: 每行 {id, text, triples}")
    parser.add_argument("--output", required=True, help="输出 JSONL: 每行 {id, text, text_norm, triples}")
    args = parser.parse_args(argv)

    total = run(args.texts, args.triples, args.output)
    if not _logging.getLogger().handlers:
        _logging.basicConfig(level=_logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _logging.info(f"Wrote {total} lines to {args.output}")


if __name__ == "__main__":
    main()
