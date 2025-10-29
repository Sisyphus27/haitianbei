'''
Author: zy
Date: 2025-10-23 10:11:42
LastEditTime: 2025-10-23 10:11:45
LastEditors: zy
Description: 
FilePath: \haitianbei\tests\run_triples_demo.py

'''
import os
import sys

# 将项目根目录加入 sys.path 以便脚本直接运行
CURR = os.path.dirname(__file__)
ROOT = os.path.dirname(CURR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.triples_extraction import extract_triples

text = (
    "时间：2025年7月1日 08:00:00，信息：飞机A001开始着陆，使用着陆跑道Z，坐标(60，260)，速度15.2米/秒；"
    "系统检测到5号牵引车待命于着陆跑道。"
)
triples = extract_triples(text)
print(triples)

expected = {
    ("飞机A001", "时间", "2025-07-01 08:00:00"),
    ("飞机A001", "动作", "开始着陆"),
    ("飞机A001", "使用跑道", "Z"),
    ("飞机A001", "坐标", "(60,260)"),
    ("飞机A001", "速度", "15.2米/秒"),
    ("5号牵引车", "待命位置", "着陆跑道"),
}

missing = [t for t in expected if t not in set(triples)]
if missing:
    raise SystemExit(f"Missing triples: {missing}; got: {triples}")
