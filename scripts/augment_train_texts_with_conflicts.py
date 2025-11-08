'''
Author: zy
Date: 2025-11-07 16:08:40
LastEditTime: 2025-11-07 16:08:44
LastEditors: zy
Description: 
FilePath: \haitianbei\scripts\augment_train_texts_with_conflicts.py

'''
#!/usr/bin/env python
r"""
将原始非冲突样本中约10%增强为“包含潜在冲突”的样本。

输入：JSONL，每行一个对象，至少包含一个可用于抽取的文本字段。
优先字段顺序：text > event > input > raw > content。若对象中无这些字段，则跳过。
冲突生成策略：
 1. 正则/三元组抽取资源实体（飞机/跑道/停机位/牵引车等）。
 2. 在抽取结果中随机选1个资源，构造一个附加片段：
      "；系统提示：<资源> 当前不可用，可能导致调度冲突"
 3. 对于跑道或停机位，随机改变其占用状态文本；对于飞机，附加“已被其它设备占用”；对牵引设备，加“已锁定”。
 4. 保持原文本在前，附加片段在后，避免破坏原始结构。
 5. 写入新字段 conflict_augmented=true 标记被修改。

用法示例：
  python scripts/augment_train_texts_with_conflicts.py \
      --input data_provider/train_texts.jsonl \
      --output data_provider/train_texts_conflict_aug.jsonl \
      --ratio 0.1 --seed 42

参数：
  --input   输入 JSONL 文件路径
  --output  输出 JSONL 文件路径（默认在输入同目录加 _conflict_aug 后缀）
  --ratio   增强比例（默认0.1）
  --seed    随机种子（默认42）
  --dry_run 不写文件，只打印统计

不会修改原文件；若输出文件已存在需显式 --force 覆盖。

兼容：若安装了 models.triples_extraction.extract_triples 则优先使用；否则退化到简易正则。
"""
import os
import json
import random
import argparse
from typing import List, Dict, Any, Optional

# 尝试导入三元组抽取器（可选）
try:
    from models.triples_extraction import extract_triples  # type: ignore
except Exception:
    extract_triples = None  # type: ignore

RESOURCE_PREFIXES = ["飞机", "跑道", "停机位", "牵引车"]
TEXT_FIELDS_ORDER = ["text", "event", "input", "raw", "content"]

def _find_text_field(obj: Dict[str, Any]) -> Optional[str]:
    for k in TEXT_FIELDS_ORDER:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return k
    return None

def _simple_regex_extract(text: str) -> List[str]:
    import re
    ents: List[str] = []
    # 飞机ID：飞机A001 / 飞机B12
    ents += ["飞机" + m for m in re.findall(r"飞机([A-Za-z0-9]+)", text)]
    # 停机位：14号停机位 / 停机位14 / 14号
    for m in re.findall(r"停机位\s*(\d+)", text):
        ents.append(f"停机位{m}")
    for m in re.findall(r"(\d+)号停机位", text):
        ents.append(f"停机位{m}")
    # 纯 “14号” 但后面跟停机位语义的（简单粗略）
    for m in re.findall(r"(\d+)号(?!跑道)", text):
        ents.append(f"停机位{m}")
    # 跑道：跑道Z / 跑道29 / 着陆跑道Z
    for m in re.findall(r"跑道([Zz]|\d+)", text):
        mm = m.upper()
        ents.append("跑道" + mm)
    # 牵引车：数字+号牵引车
    for m in re.findall(r"(\d+)号牵引车", text):
        ents.append(f"{m}号牵引车")
    # 去重保持顺序
    out: List[str] = []
    seen = set()
    for e in ents:
        if e not in seen:
            out.append(e)
            seen.add(e)
    return out

def _extract_resources(text: str) -> List[str]:
    if extract_triples is not None:
        try:
            trips = extract_triples(text)
            ents: List[str] = []
            for s, p, o in trips:
                for t in (s, o):
                    t = str(t)
                    if any(t.startswith(pref) for pref in RESOURCE_PREFIXES):
                        ents.append(t)
            # 去重
            uniq: List[str] = []
            seen = set()
            for e in ents:
                if e not in seen:
                    uniq.append(e)
                    seen.add(e)
            return uniq
        except Exception:
            pass
    # 回退：正则
    return _simple_regex_extract(text)

def _conflict_suffix(resource: str) -> str:
    # 针对不同资源类型定制提示语
    if resource.startswith("跑道"):
        return f"；系统提示：{resource} 当前被其它航班占用，可能导致冲突"
    if resource.startswith("停机位"):
        return f"；系统提示：{resource} 已被占用，等待释放后再调度"
    if resource.startswith("飞机"):
        return f"；系统提示：{resource} 当前处于锁定状态，相关作业需重新排程"
    if resource.endswith("牵引车"):
        return f"；系统提示：{resource} 已锁定或维护中，无法继续牵引"
    return f"；系统提示：{resource} 暂时不可用"


def augment_conflicts(records: List[Dict[str, Any]], ratio: float, seed: int) -> List[Dict[str, Any]]:
    random.seed(seed)
    n = len(records)
    target = max(1, int(n * ratio))
    idxs = list(range(n))
    random.shuffle(idxs)
    chosen = set(idxs[:target])

    augmented: List[Dict[str, Any]] = []
    for i, obj in enumerate(records):
        new_obj = dict(obj)
        if i in chosen:
            field = _find_text_field(obj)
            if field:
                text = str(obj[field])
                resources = _extract_resources(text)
                if resources:
                    res = random.choice(resources)
                else:
                    res = "停机位13"  # 兜底一个常见资源
                new_text = text.rstrip() + _conflict_suffix(res)
                new_obj[field] = new_text
                new_obj["conflict_augmented"] = True
        augmented.append(new_obj)
    return augmented


def load_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """读取 JSONL 文件，可选仅返回前 limit 条记录。"""
    arr: List[Dict[str, Any]] = []
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict):
                    arr.append(obj)
                    count += 1
                    if limit is not None and count >= limit:
                        break
            except Exception:
                continue
    return arr


def save_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description="将约10%原始样本增强为潜在冲突样本")
    ap.add_argument("--input", required=True, help="输入 JSONL 文件路径")
    ap.add_argument("--output", default=None, help="输出 JSONL 文件路径 (默认自动命名 *_conflict_aug.jsonl)")
    ap.add_argument("--ratio", type=float, default=0.1, help="增强比例 (默认0.1)")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--dry_run", action="store_true", help="只打印统计不写文件")
    ap.add_argument("--limit", type=int, default=None, help="仅读取前 N 条记录进行增强（用于快速验证流程）")
    ap.add_argument("--force", action="store_true", help="已存在输出文件时覆盖")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"输入文件不存在: {args.input}")
    out_path = args.output
    if not out_path:
        base, ext = os.path.splitext(args.input)
        out_path = base + "_conflict_aug.jsonl"
    if os.path.exists(out_path) and not args.force and not args.dry_run:
        raise SystemExit(f"输出文件已存在: {out_path} (使用 --force 覆盖或 --output 指定新文件)")

    records = load_jsonl(args.input, limit=args.limit)
    if not records:
        raise SystemExit("输入文件为空或格式不合法 (没有有效 JSON 对象行)")

    augmented = augment_conflicts(records, ratio=args.ratio, seed=args.seed)
    total = len(records)
    modified = sum(1 for r in augmented if r.get("conflict_augmented"))
    extra = f" | 仅读取前 {args.limit} 条" if args.limit is not None else ""
    print(f"总样本: {total}{extra} | 增强比例目标: {args.ratio:.2f} | 实际修改: {modified} ({modified/total:.2%})")

    if args.dry_run:
        print("[dry_run] 不写输出文件。示例修改后样本：")
        for r in augmented[:3]:
            if r.get("conflict_augmented"):
                field = _find_text_field(r)
                if field:
                    print("---")
                    print(r[field])
        return

    save_jsonl(augmented, out_path)
    print(f"已写出: {out_path}")


if __name__ == "__main__":
    main()
