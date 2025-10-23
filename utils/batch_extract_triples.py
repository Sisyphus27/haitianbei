r'''
Author: zy
Date: 2025-10-23 10:42:22
LastEditTime: 2025-10-23 10:42:26
LastEditors: zy
Description: 
FilePath: \haitianbei\utils\batch_extract_triples.py

'''
from __future__ import annotations

# 批量抽取三元组：读取 JSONL (id,text)，调用规则抽取器，写出 (id,text,triples)。
#
# 用法：
# python -m utils.batch_extract_triples \
#   --input d:/WorkSpace/haitianbei/data_provider/train_texts.jsonl \
#   --output d:/WorkSpace/haitianbei/data_provider/train_triples.jsonl

import argparse
import json
from typing import Iterable

from models.triples_extraction import extract_triples


def run(input_path: str, output_path: str) -> int:
    count = 0
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            _id = obj.get("id")
            triples = extract_triples(text)
            out = {"id": _id, "text": text, "triples": [list(t) for t in triples]}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1
    return count


message = "批量抽取三元组"

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=message)
    parser.add_argument("--input", required=True, help="输入 JSONL: 每行 {id, text}")
    parser.add_argument("--output", required=True, help="输出 JSONL: 每行 {id, text, triples}")
    args = parser.parse_args(argv)

    total = run(args.input, args.output)
    print(f"Wrote {total} lines to {args.output}")


if __name__ == "__main__":
    main()
