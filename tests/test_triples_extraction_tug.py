from models.triples_extraction import extract_triples

def test_extract_triples_tug_scenario():
    text = (
        "时间：2025年7月1日 08:01:00，信息：飞机A001着陆完成；"
        "5号牵引车开始牵引飞机A001滑行至14号停机位，滑行速度5米/秒。"
    )
    triples = extract_triples(text)
    s = set(triples)

    expected = {
        ("飞机A001", "动作", "着陆完成"),
        ("5号牵引车", "牵引", "飞机A001"),
        ("5号牵引车", "滑至", "14号停机位"),
        ("飞机A001", "滑至", "14号停机位"),
        ("5号牵引车", "速度", "5米/秒"),
        ("飞机A001", "速度", "5米/秒"),
    }

    for t in expected:
        assert t in s, f"缺失三元组: {t}; 实际: {triples}"
