'''
Author: zy
Date: 2025-10-23 10:10:58
LastEditTime: 2025-10-23 10:11:02
LastEditors: zy
Description: 
FilePath: /haitianbei/tests/test_triples_extraction.py

'''
from models.triples_extraction import extract_triples


def test_extract_triples_sample():
    text = (
        "时间：2025年7月1日 08:00:00，信息：飞机A001开始着陆，使用着陆跑道Z，坐标(60，260)，速度15.2米/秒；"
        "系统检测到5号牵引车待命于着陆跑道。"
    )
    triples = extract_triples(text)
    s = set(triples)

    expected = {
        ("飞机A001", "时间", "2025-07-01 08:00:00"),
        ("飞机A001", "动作", "开始着陆"),
        ("飞机A001", "使用跑道", "Z"),
        ("飞机A001", "坐标", "(60,260)"),
        ("飞机A001", "速度", "15.2米/秒"),
        ("5号牵引车", "待命位置", "着陆跑道"),
    }

    for t in expected:
        assert t in s, f"缺失三元组: {t}; 实际: {triples}"
