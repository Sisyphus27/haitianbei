import os, sys
ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.triples_extraction import extract_triples

def run_case():
    text = "时间：2025年7月1日 08:01:00，信息：飞机A001着陆完成；5号牵引车开始牵引飞机A001滑行至14号停机位，滑行速度5米/秒。"
    res = extract_triples(text)
    for t in res:
        print(t)

if __name__ == "__main__":
    run_case()
