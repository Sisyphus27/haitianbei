r'''
验证 Dataset_KG 的单值关系冲突处理：
- 对于 SINGLE_VALUED_RELS（如 分配停机位/ASSIGNED_GATE、使用跑道/USES_RUNWAY、待命位置/STANDBY_AT、当前停机位/HAS_CURRENT_GATE），
    当同一主体再次写入不同客体时，应删除旧边，仅保留新边。
- 对于 到达停机位（ARRIVED_GATE），默认允许保留历史多条，但 HAS_CURRENT_GATE 应仅指向最新到达的停机位。

运行方式：
    python tests/test_kg_conflicts.py

提示：本测试会调用 reset_graph(keep_fixed=True) 清理动态数据，仅保留固定节点。
'''

import os
import sys

# 确保项目根目录在 sys.path，便于脚本方式运行本测试
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_provider.data_loader import Dataset_KG

ROOT = os.getcwd()
SUBJ = "飞机T-测试冲突"


def collect_objs(kg: Dataset_KG, predicate_cn: str):
    triples = kg.query(SUBJ, predicate_cn)
    return sorted([o for (_, _, o) in triples])


def make_kg() -> Dataset_KG:
    """构造带凭据的 Dataset_KG，优先读取环境变量。
    支持 NEO4J_AUTH=USER/PASSWORD 快速指定认证。
    """
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    pwd = os.environ.get("NEO4J_PASSWORD", "neo4j")
    auth = os.environ.get("NEO4J_AUTH")
    if auth and "/" in auth:
        u, p = auth.split("/", 1)
        user = u or user
        pwd = p or pwd
    db = os.environ.get("NEO4J_DATABASE")
    return Dataset_KG(ROOT, load_data=False, neo4j_uri=uri, neo4j_user=user, neo4j_password=pwd, neo4j_database=db)


def try_make_kg():
    try:
        return make_kg()
    except RuntimeError as e:
        # 无法连接 Neo4j 或认证失败时跳过测试（不视为失败），打印提示
        print("[SKIP] Neo4j 不可用或认证失败，跳过冲突测试:", e)
        raise SystemExit(0)


def test_assigned_gate_conflict():
    kg = try_make_kg()
    # 清理历史动态，避免干扰
    kg.reset_graph(keep_fixed=True)

    # 第一次分配 14 号
    kg.update_with_triples([(SUBJ, "分配停机位", "14号")])
    objs1 = collect_objs(kg, "分配停机位")
    assert objs1 == ["停机位14"], f"期望仅有 停机位14，实际: {objs1}"

    # 第二次分配 15 号，应覆盖旧边
    kg.update_with_triples([(SUBJ, "分配停机位", "15号")])
    objs2 = collect_objs(kg, "分配停机位")
    assert objs2 == ["停机位15"], f"期望仅有 停机位15，实际: {objs2}"


def test_uses_runway_conflict():
    kg = try_make_kg()
    # 使用跑道 Z，然后切换到 29，应只保留 跑道29
    kg.update_with_triples([(SUBJ, "使用跑道", "Z")])
    objs1 = collect_objs(kg, "使用跑道")
    assert objs1 == ["跑道Z"], f"期望 跑道Z，实际: {objs1}"

    kg.update_with_triples([(SUBJ, "使用跑道", "29")])
    objs2 = collect_objs(kg, "使用跑道")
    assert objs2 == ["跑道29"], f"期望 跑道29，实际: {objs2}"


def test_arrived_gate_history_and_current():
    kg = try_make_kg()
    # 到达 14，再到达 15：历史允许多条 ARRIVED_GATE，但 HAS_CURRENT_GATE 应仅指向 15
    kg.update_with_triples([(SUBJ, "到达停机位", "14号")])
    kg.update_with_triples([(SUBJ, "到达停机位", "15号")])

    arrived = collect_objs(kg, "到达停机位")
    current = collect_objs(kg, "当前停机位")

    # 历史可包含 14 与 15（若你希望到达也单值，可把 ARRIVED_GATE 加入 SINGLE_VALUED_RELS）
    assert "停机位15" in arrived, f"到达历史缺失 停机位15: {arrived}"
    assert current == ["停机位15"], f"当前停机位应为 15，实际: {current}"


if __name__ == "__main__":
    # 以脚本方式运行时，按顺序执行测试并打印结果
    failed = []
    try:
        test_assigned_gate_conflict()
        print("[PASS] test_assigned_gate_conflict")
    except AssertionError as e:
        print("[FAIL] test_assigned_gate_conflict:", e)
        failed.append("test_assigned_gate_conflict")

    try:
        test_uses_runway_conflict()
        print("[PASS] test_uses_runway_conflict")
    except AssertionError as e:
        print("[FAIL] test_uses_runway_conflict:", e)
        failed.append("test_uses_runway_conflict")

    try:
        test_arrived_gate_history_and_current()
        print("[PASS] test_arrived_gate_history_and_current")
    except AssertionError as e:
        print("[FAIL] test_arrived_gate_history_and_current:", e)
        failed.append("test_arrived_gate_history_and_current")

    if failed:
        raise SystemExit(1)
    else:
        print("All tests passed.")
