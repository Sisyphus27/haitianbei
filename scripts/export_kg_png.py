"""
测试脚本：验证图谱更新逻辑
- 测试飞机编号是否正确更新（如C011）
- 测试固定资源的isDamaged属性
- 测试离散节点清理功能

## 使用测试容器（推荐）

### 1. 启动测试容器
```bash
docker run -d \
  --name neo4j-test-kg \
  -p 7688:7687 \
  -p 7475:7474 \
  -e NEO4J_AUTH=neo4j/test123456 \
  neo4j:latest
```

### 2. 等待容器启动（约10-30秒）
```bash
# 检查容器状态
docker ps | grep neo4j-test-kg

# 查看容器日志
docker logs neo4j-test-kg
```

### 3. 运行测试
```bash
# 使用默认测试容器连接
python scripts/export_kg_png.py

# 或使用自定义连接参数
python scripts/export_kg_png.py --test-uri bolt://localhost:7688 --test-user neo4j --test-password test123456

# 或使用环境变量
export NEO4J_TEST_URI=bolt://localhost:7688
export NEO4J_TEST_USER=neo4j
export NEO4J_TEST_PASSWORD=test123456
python scripts/export_kg_png.py
```

### 4. 清理测试容器（可选）
```bash
# 停止并删除测试容器
docker rm -f neo4j-test-kg
```

## 测试容器配置

- 容器名称: `neo4j-test-kg`
- Bolt端口: 7688（主机）-> 7687（容器）
- HTTP端口: 7475（主机）-> 7474（容器）
- 认证: 用户名 `neo4j`, 密码 `test123456`
- 数据卷: 不持久化（测试完成后可删除）

## 注意事项

1. 确保端口7688和7475未被占用
2. 测试容器与生产容器完全隔离（不同端口）
3. 如果测试容器已存在，需要先删除: `docker rm -f neo4j-test-kg`
4. 确保Docker已安装并运行
"""

import os
import sys
import argparse

# 确保项目根目录在 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_provider.data_loader import Dataset_KG
from exp.kg_service import KGServiceLocal
import logging
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_kg_instance(neo4j_uri=None, neo4j_user=None, neo4j_password=None, neo4j_database=None):
    """创建KG实例，使用指定的Neo4j连接参数"""
    try:
        kg = Dataset_KG(
            root_path=project_root,
            load_data=False,
            create_constraints=True,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            neo4j_database=neo4j_database,
        )
        # 测试连接
        with kg.driver.session(database=kg.neo4j_database) as sess:
            sess.run("RETURN 1 AS ok").single()
        logger.info(f"成功连接到Neo4j: {neo4j_uri or '默认'}")
        return kg
    except Exception as e:
        logger.error(f"连接Neo4j失败: {e}")
        logger.error("请确保测试容器已启动，命令：")
        logger.error("docker run -d --name neo4j-test-kg -p 7688:7687 -p 7475:7474 -e NEO4J_AUTH=neo4j/test123456 neo4j:latest")
        raise


def test_aircraft_update(neo4j_uri=None, neo4j_user=None, neo4j_password=None, neo4j_database=None):
    """测试飞机编号更新：验证C011等编号是否正确更新到图谱"""
    logger.info("=" * 60)
    logger.info("测试1: 飞机编号更新")
    logger.info("=" * 60)
    
    # 初始化KG
    kg = create_kg_instance(neo4j_uri, neo4j_user, neo4j_password, neo4j_database)
    kg_service = KGServiceLocal(kg)
    
    # 重置图谱（保留固定节点）
    logger.info("重置图谱...")
    kg_service.reset_graph(keep_fixed=True)
    
    # 测试事件：包含飞机C011
    test_event = "时间：2025年7月1日 10:39:32，信息：飞机C011开始固定作业，使用17号停机位锚点装置。"
    logger.info(f"处理事件: {test_event}")
    
    # 更新图谱
    triples = kg_service.extract_and_update(test_event)
    logger.info(f"抽取的三元组: {triples}")
    
    # 查询飞机C011是否存在
    with kg.driver.session(database=kg.neo4j_database) as sess:
        result = sess.run("""
            MATCH (a:Aircraft {name: "飞机C011"})
            RETURN a.name AS name, labels(a) AS labels
        """).single()
        
        if result:
            logger.info(f"✓ 飞机C011已成功添加到图谱: {result['name']}")
            logger.info(f"  标签: {result['labels']}")
        else:
            logger.error("✗ 飞机C011未找到！")
            # 查找所有飞机
            all_aircraft = sess.run("""
                MATCH (a:Aircraft)
                RETURN a.name AS name
                ORDER BY a.name
            """).data()
            logger.info(f"当前图谱中的所有飞机: {[a['name'] for a in all_aircraft]}")
    
    # 清理离散节点（静默）
    kg_service.cleanup_isolated_nodes()
    
    return result is not None


def test_isDamaged_attribute(neo4j_uri=None, neo4j_user=None, neo4j_password=None, neo4j_database=None):
    """测试固定资源的isDamaged属性"""
    logger.info("=" * 60)
    logger.info("测试2: 固定资源isDamaged属性")
    logger.info("=" * 60)
    
    # 初始化KG
    kg = create_kg_instance(neo4j_uri, neo4j_user, neo4j_password, neo4j_database)
    kg_service = KGServiceLocal(kg)
    
    # 重置图谱（保留固定节点）
    logger.info("重置图谱...")
    kg_service.reset_graph(keep_fixed=True)
    
    # 检查跑道、停机位、资源、设备的isDamaged属性
    with kg.driver.session(database=kg.neo4j_database) as sess:
        # 检查跑道
        runways = sess.run("""
            MATCH (r:Runway)
            WHERE r.isFixed = true
            RETURN r.name AS name, r.isDamaged AS isDamaged
            LIMIT 5
        """).data()
        
        logger.info(f"跑道isDamaged属性检查:")
        for rw in runways:
            has_attr = rw.get("isDamaged") is not None
            status = "✓" if has_attr else "✗"
            logger.info(f"  {status} {rw['name']}: isDamaged={rw.get('isDamaged')}")
        
        # 检查停机位
        gates = sess.run("""
            MATCH (g:Gate)
            WHERE g.isFixed = true
            RETURN g.name AS name, g.isDamaged AS isDamaged
            LIMIT 5
        """).data()
        
        logger.info(f"停机位isDamaged属性检查:")
        for gate in gates:
            has_attr = gate.get("isDamaged") is not None
            status = "✓" if has_attr else "✗"
            logger.info(f"  {status} {gate['name']}: isDamaged={gate.get('isDamaged')}")
        
        # 检查资源
        resources = sess.run("""
            MATCH (r:Resource)
            WHERE r.isFixed = true
            RETURN r.name AS name, r.isDamaged AS isDamaged
            LIMIT 5
        """).data()
        
        logger.info(f"资源isDamaged属性检查:")
        for res in resources:
            has_attr = res.get("isDamaged") is not None
            status = "✓" if has_attr else "✗"
            logger.info(f"  {status} {res['name']}: isDamaged={res.get('isDamaged')}")
        
        # 检查设备
        devices = sess.run("""
            MATCH (d:FixedDevice)
            WHERE d.isFixed = true
            RETURN d.name AS name, d.isDamaged AS isDamaged
            LIMIT 5
        """).data()
        
        logger.info(f"设备isDamaged属性检查:")
        for dev in devices:
            has_attr = dev.get("isDamaged") is not None
            status = "✓" if has_attr else "✗"
            logger.info(f"  {status} {dev['name']}: isDamaged={dev.get('isDamaged')}")
    
    # 验证所有固定资源都有isDamaged属性
    all_ok = (
        all(r.get("isDamaged") is not None for r in runways) and
        all(g.get("isDamaged") is not None for g in gates) and
        all(r.get("isDamaged") is not None for r in resources) and
        all(d.get("isDamaged") is not None for d in devices)
    )
    
    return all_ok


def test_isolated_nodes_cleanup(neo4j_uri=None, neo4j_user=None, neo4j_password=None, neo4j_database=None):
    """测试离散节点清理功能"""
    logger.info("=" * 60)
    logger.info("测试3: 离散节点清理")
    logger.info("=" * 60)
    
    # 初始化KG
    kg = create_kg_instance(neo4j_uri, neo4j_user, neo4j_password, neo4j_database)
    kg_service = KGServiceLocal(kg)
    
    # 重置图谱（保留固定节点）
    logger.info("重置图谱...")
    kg_service.reset_graph(keep_fixed=True)
    
    # 添加一些测试事件（包含时间节点）
    test_events = [
        "时间：2025年7月1日 08:00:00，信息：飞机A001开始着陆，使用着陆跑道Z。",
        "时间：2025年7月1日 08:01:00，信息：飞机A001着陆完成。",
        "时间：2025年7月1日 10:39:32，信息：飞机C011开始固定作业，使用17号停机位锚点装置。",
    ]
    
    # 更新图谱
    for event in test_events:
        kg_service.extract_and_update(event)
    
    # 统计清理前的节点数
    with kg.driver.session(database=kg.neo4j_database) as sess:
        before_result = sess.run("MATCH (n) RETURN count(n) AS cnt").single()
        before_count = before_result["cnt"] if before_result else 0
        logger.info(f"清理前节点数: {before_count}")
        
        # 查找时间节点
        time_nodes = sess.run("""
            MATCH (n)
            WHERE n.name IS NOT NULL
            AND (
                n.name =~ '^\\d{4}-\\d{2}-\\d{2}.*' OR
                n.name =~ '.*\\d{4}年\\d{1,2}月\\d{1,2}日.*'
            )
            RETURN n.name AS name, labels(n) AS labels
        """).data()
        logger.info(f"清理前时间节点数: {len(time_nodes)}")
        for tn in time_nodes[:5]:  # 只显示前5个
            logger.info(f"  时间节点: {tn['name']} ({tn['labels']})")
    
    # 执行清理
    deleted_count = kg_service.cleanup_isolated_nodes()
    logger.info(f"清理删除的节点数: {deleted_count}")
    
    # 统计清理后的节点数
    with kg.driver.session(database=kg.neo4j_database) as sess:
        after_result = sess.run("MATCH (n) RETURN count(n) AS cnt").single()
        after_count = after_result["cnt"] if after_result else 0
        logger.info(f"清理后节点数: {after_count}")
        
        # 查找时间节点
        time_nodes_after = sess.run("""
            MATCH (n)
            WHERE n.name IS NOT NULL
            AND (
                n.name =~ '^\\d{4}-\\d{2}-\\d{2}.*' OR
                n.name =~ '.*\\d{4}年\\d{1,2}月\\d{1,2}日.*'
            )
            RETURN n.name AS name, labels(n) AS labels
        """).data()
        logger.info(f"清理后时间节点数: {len(time_nodes_after)}")
    
    # 验证必要节点仍然存在
    with kg.driver.session(database=kg.neo4j_database) as sess:
        aircraft_result = sess.run("MATCH (a:Aircraft) RETURN count(a) AS cnt").single()
        aircraft_count = aircraft_result["cnt"] if aircraft_result else 0
        gate_result = sess.run("MATCH (g:Gate) RETURN count(g) AS cnt").single()
        gate_count = gate_result["cnt"] if gate_result else 0
        runway_result = sess.run("MATCH (r:Runway) RETURN count(r) AS cnt").single()
        runway_count = runway_result["cnt"] if runway_result else 0
        device_result = sess.run("MATCH (d:Device) RETURN count(d) AS cnt").single()
        device_count = device_result["cnt"] if device_result else 0
        resource_result = sess.run("MATCH (r:Resource) RETURN count(r) AS cnt").single()
        resource_count = resource_result["cnt"] if resource_result else 0
        job_result = sess.run("MATCH (j:Job) RETURN count(j) AS cnt").single()
        job_count = job_result["cnt"] if job_result else 0
        
        logger.info(f"必要节点统计:")
        logger.info(f"  飞机: {aircraft_count}")
        logger.info(f"  停机位: {gate_count}")
        logger.info(f"  跑道: {runway_count}")
        logger.info(f"  设备: {device_count}")
        logger.info(f"  资源: {resource_count}")
        logger.info(f"  任务: {job_count}")
    
    return deleted_count > 0 and len(time_nodes_after) < len(time_nodes)


def test_full_process(neo4j_uri=None, neo4j_user=None, neo4j_password=None, neo4j_database=None):
    """测试完整流程：从train_texts.jsonl文件读取前500行事件并更新图谱"""
    logger.info("=" * 60)
    logger.info("测试4: 完整流程测试（从train_texts.jsonl前500行）")
    logger.info("=" * 60)
    
    # 初始化KG
    kg = create_kg_instance(neo4j_uri, neo4j_user, neo4j_password, neo4j_database)
    kg_service = KGServiceLocal(kg)
    
    # 重置图谱（保留固定节点）
    logger.info("重置图谱...")
    kg_service.reset_graph(keep_fixed=True)
    
    # 读取train_texts.jsonl文件
    jsonl_file = os.path.join(project_root, "data_provider", "train_texts.jsonl")
    if not os.path.exists(jsonl_file):
        logger.warning(f"文件不存在: {jsonl_file}")
        return False
    
    # 读取事件（限制前500行）
    events = []
    max_lines = 500
    try:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if line_num > max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    # 提取text字段
                    text = obj.get("text") or obj.get("event") or obj.get("input")
                    if isinstance(text, str) and text.strip():
                        events.append(text.strip())
                except json.JSONDecodeError:
                    # 如果不是有效的JSON，跳过该行
                    logger.warning(f"跳过无效的JSON行 {line_num}: {line[:50]}...")
                    continue
                except Exception as e:
                    logger.warning(f"解析第 {line_num} 行时出错: {e}")
                    continue
    except Exception as e:
        logger.error(f"读取文件失败: {e}")
        return False
    
    logger.info(f"读取到 {len(events)} 个事件（前{max_lines}行）")
    
    # 处理事件
    for i, event in enumerate(events, 1):
        logger.info(f"处理事件 {i}/{len(events)}: {event}")
        try:
            kg_service.extract_and_update(event)
            # 静默清理离散节点（不显示日志）
            kg_service.cleanup_isolated_nodes()
        except Exception as e:
            logger.error(f"处理事件失败: {e}")
    
    # 验证飞机编号
    with kg.driver.session(database=kg.neo4j_database) as sess:
        # 查找所有飞机
        all_aircraft = sess.run("""
            MATCH (a:Aircraft)
            RETURN a.name AS name
            ORDER BY a.name
        """).data()
        
        logger.info(f"图谱中的所有飞机:")
        for ac in all_aircraft:
            logger.info(f"  - {ac['name']}")
        
        # 检查是否包含C011
        c011_exists = any(ac["name"] == "飞机C011" for ac in all_aircraft)
        if c011_exists:
            logger.info("✓ 飞机C011已成功添加到图谱")
        else:
            logger.warning("✗ 飞机C011未找到（可能事件中没有C011）")
    
    # 导出图谱PNG
    output_dir = os.path.join(project_root, "results", "kg_vis")
    os.makedirs(output_dir, exist_ok=True)
    output_png = os.path.join(output_dir, "test_kg_export.png")
    
    try:
        kg_service.export_png(output_png)
        logger.info(f"✓ 图谱已导出到: {output_png}")
    except Exception as e:
        logger.error(f"导出图谱失败: {e}")
    
    return True


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="测试脚本：验证图谱更新逻辑",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用测试容器（推荐）:
  1. 启动测试容器:
     docker run -d --name neo4j-test-kg -p 7688:7687 -p 7475:7474 -e NEO4J_AUTH=neo4j/test123456 neo4j:latest
  
  2. 运行测试:
     python scripts/export_kg_png.py
  
  3. 清理测试容器（可选）:
     docker rm -f neo4j-test-kg
        """
    )
    
    # 测试容器连接参数
    parser.add_argument(
        "--test-uri",
        type=str,
        default=None,
        help="测试Neo4j URI（默认: bolt://localhost:7688）"
    )
    parser.add_argument(
        "--test-user",
        type=str,
        default=None,
        help="测试Neo4j用户名（默认: neo4j）"
    )
    parser.add_argument(
        "--test-password",
        type=str,
        default=None,
        help="测试Neo4j密码（默认: test123456）"
    )
    parser.add_argument(
        "--test-database",
        type=str,
        default=None,
        help="测试数据库名称（默认: neo4j）"
    )
    
    return parser.parse_args()


def get_test_connection_params(args):
    """获取测试容器连接参数（优先级：命令行参数 > 环境变量 > 默认值）"""
    # 默认值（测试容器配置）
    default_uri = "bolt://localhost:7688"
    default_user = "neo4j"
    default_password = "test123456"
    default_database = "neo4j"
    
    # 从命令行参数或环境变量获取
    neo4j_uri = (
        args.test_uri or
        os.environ.get("NEO4J_TEST_URI") or
        default_uri
    )
    neo4j_user = (
        args.test_user or
        os.environ.get("NEO4J_TEST_USER") or
        default_user
    )
    neo4j_password = (
        args.test_password or
        os.environ.get("NEO4J_TEST_PASSWORD") or
        default_password
    )
    neo4j_database = (
        args.test_database or
        os.environ.get("NEO4J_TEST_DATABASE") or
        default_database
    )
    
    return neo4j_uri, neo4j_user, neo4j_password, neo4j_database


def main():
    """主测试函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 获取测试容器连接参数
    neo4j_uri, neo4j_user, neo4j_password, neo4j_database = get_test_connection_params(args)
    
    logger.info("开始测试图谱更新逻辑")
    logger.info("=" * 60)
    logger.info(f"测试容器连接:")
    logger.info(f"  URI: {neo4j_uri}")
    logger.info(f"  用户: {neo4j_user}")
    logger.info(f"  数据库: {neo4j_database}")
    logger.info("=" * 60)
    
    results = {}
    
    # 测试1: 飞机编号更新
    try:
        results["aircraft_update"] = test_aircraft_update(neo4j_uri, neo4j_user, neo4j_password, neo4j_database)
    except Exception as e:
        logger.error(f"测试1失败: {e}", exc_info=True)
        results["aircraft_update"] = False
    
    # 测试2: isDamaged属性
    try:
        results["isDamaged"] = test_isDamaged_attribute(neo4j_uri, neo4j_user, neo4j_password, neo4j_database)
    except Exception as e:
        logger.error(f"测试2失败: {e}", exc_info=True)
        results["isDamaged"] = False
    
    # 测试3: 离散节点清理
    try:
        results["isolated_cleanup"] = test_isolated_nodes_cleanup(neo4j_uri, neo4j_user, neo4j_password, neo4j_database)
    except Exception as e:
        logger.error(f"测试3失败: {e}", exc_info=True)
        results["isolated_cleanup"] = False
    
    # 测试4: 完整流程
    try:
        results["full_process"] = test_full_process(neo4j_uri, neo4j_user, neo4j_password, neo4j_database)
    except Exception as e:
        logger.error(f"测试4失败: {e}", exc_info=True)
        results["full_process"] = False
    
    # 输出测试结果
    logger.info("=" * 60)
    logger.info("测试结果汇总:")
    logger.info("=" * 60)
    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"  {test_name}: {status}")
    
    # 总结
    all_passed = all(results.values())
    if all_passed:
        logger.info("=" * 60)
        logger.info("所有测试通过！")
        logger.info("=" * 60)
    else:
        logger.warning("=" * 60)
        logger.warning("部分测试失败，请检查上述日志")
        logger.warning("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

