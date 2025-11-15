#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
插入脚本：将故障事件插入train_texts.jsonl

将以下两个事件按时间顺序插入到train_texts.jsonl中：
1. 08:35:00 - 编号 10-15 等 6 个停机位故障不可用
2. 10:15:00 - 下一架降落飞机将因故障必须停在 5 号停机位修理
"""

import json
import re
from datetime import datetime
from pathlib import Path


def parse_time_from_text(text: str) -> datetime:
    """从文本中解析时间"""
    # 匹配格式：时间：2025年7月1日 HH:MM:SS
    pattern = r"时间：(\d{4})年(\d{1,2})月(\d{1,2})日\s+(\d{2}):(\d{2}):(\d{2})"
    match = re.search(pattern, text)
    if match:
        year, month, day, hour, minute, second = map(int, match.groups())
        return datetime(year, month, day, hour, minute, second)
    raise ValueError(f"无法从文本中解析时间: {text[:50]}...")


def create_event(time_str: str, message: str) -> dict:
    """创建标准格式的事件"""
    return {
        "text": f"时间：2025年7月1日 {time_str}，信息：{message}"
    }


def insert_task3_events():
    """主函数：插入故障事件到train_texts.jsonl"""
    # 文件路径
    input_file = Path("data_provider/train_texts.jsonl")
    output_file = Path("data_provider/train_texts_task3.jsonl")
    
    # 读取原始文件
    print(f"读取文件: {input_file}")
    events = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                events.append(event)
            except json.JSONDecodeError as e:
                print(f"警告：跳过无效JSON行: {e}")
                continue
    
    print(f"读取了 {len(events)} 条原始事件")
    
    # 创建两个新事件
    new_event1 = create_event(
        "08:35:00",
        "编号 10-15 等 6 个停机位故障不可用，不可用时间均为30分钟"
    )
    new_event2 = create_event(
        "10:15:00",
        "下一架降落飞机将因故障必须停在 5 号停机位修理，修理时间为 30分钟"
    )
    
    # 解析所有事件的时间并排序
    print("解析时间并排序...")
    event_with_time = []
    
    for event in events:
        try:
            dt = parse_time_from_text(event["text"])
            event_with_time.append((dt, event))
        except ValueError as e:
            print(f"警告：跳过无法解析时间的事件: {e}")
            continue
    
    # 添加新事件
    dt1 = parse_time_from_text(new_event1["text"])
    dt2 = parse_time_from_text(new_event2["text"])
    event_with_time.append((dt1, new_event1))
    event_with_time.append((dt2, new_event2))
    
    # 按时间排序
    event_with_time.sort(key=lambda x: x[0])
    
    print(f"排序后共有 {len(event_with_time)} 条事件（包含2条新事件）")
    
    # 重新分配id并写入文件
    print(f"写入文件: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, (dt, event) in enumerate(event_with_time):
            # 重新分配id，保持字段顺序与原始文件一致（id在前，text在后）
            ordered_event = {"id": idx, "text": event["text"]}
            # 写入JSON行
            f.write(json.dumps(ordered_event, ensure_ascii=False) + "\n")
    
    # 验证插入位置
    print("\n验证插入位置:")
    for idx, (dt, event) in enumerate(event_with_time):
        if "编号 10-15" in event["text"]:
            print(f"  事件1 (08:35:00) 插入位置: id={idx}, 时间={dt.strftime('%H:%M:%S')}")
        if "下一架降落飞机将因故障" in event["text"]:
            print(f"  事件2 (10:15:00) 插入位置: id={idx}, 时间={dt.strftime('%H:%M:%S')}")
    
    print(f"\n完成！已生成文件: {output_file}")
    print(f"总事件数: {len(event_with_time)} (原始: {len(events)}, 新增: 2)")


if __name__ == "__main__":
    insert_task3_events()

