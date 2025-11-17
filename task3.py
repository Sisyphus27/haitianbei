#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务三：冲突检测与重调度系统

功能：
1. 读取train_texts_task3.jsonl事件流
2. 使用关键词初步检测潜在冲突（减少大模型调用）
3. 当检测到冲突时，调用大模型进行冲突判定
4. 生成强化学习重调度所需的快照配置文件
5. 在整个流程中持续更新KG

涉及的文件：

直接导入的文件：
- exp/exp_main.py: 主实验类，提供stream-judge模式和KG服务
- exp/exp_basic.py: 基础实验类
- data_provider/data_loader.py: 数据加载器
- utils/utils.py: 工具函数（detect_potential_conflict, generate_snapshot_from_kg）

从htb_environment导入的文件：
- htb_environment/snapshot_scheduler.py: 强化学习重调度入口（infer_schedule_from_snapshot）
- htb_environment/environment.py: 强化学习环境类（ScheduleEnv）
- htb_environment/pipeline/kg_bridge.py: KG桥接，将调度结果转换为三元组（schedule_to_kg_triples）
- htb_environment/utils/util.py: 强化学习工具函数

间接使用的文件：
- exp/kg_service.py: 通过exp.kg_service使用，进行KG查询和更新
- models/triples_extraction.py: 通过kg_service间接使用，提取三元组

输入输出文件：
- data_provider/train_texts_task3.jsonl: 输入事件文件（默认）
- results/task2/conflict_results_*.jsonl: 输出结果文件
- results/task2/snapshots/*.json: 快照配置文件
- results/task2/任务三测试场景{X}_A_{team_name}.txt: A文件（冲突判定输出）
- results/task2/任务三测试场景{X}_S_{team_name}.txt: S文件（快照配置）
- results/task2/任务三测试场景{X}_P_{team_name}.json: P文件（重调度计划）
"""

# 事件ID白名单：只有这些事件ID会触发大模型调度和强化学习重调度
SCHEDULE_TRIGGER_EVENT_IDS = [195, 777]

# 仅生成甘特图的事件ID：这些事件只生成甘特图，不进行大模型判定和KG写回
GANTT_ONLY_EVENT_IDS = [194, 776]  # 事件194对应场景1的前一个事件，事件776对应场景2的前一个事件

# 场景编号映射：事件ID -> 场景编号
EVENT_SCENARIO_MAP = {
    195: 1,  # 事件ID 195 -> 场景1
    777: 2,  # 事件ID 777 -> 场景2
    194: 1,  # 事件ID 194 -> 场景1（前一个事件）
    776: 2   # 事件ID 776 -> 场景2（前一个事件）
}

def get_scenario_number(event_id: int) -> int:
    """获取事件ID对应的场景编号。
    
    参数:
        event_id: 事件ID
        
    返回:
        int: 场景编号（1或2），如果不在映射中则返回0
    """
    return EVENT_SCENARIO_MAP.get(event_id, 0)

import os
import sys
import importlib
import json
import re
import logging
import argparse
from datetime import datetime
from typing import Optional, List
from pathlib import Path

# 确保项目根目录在sys.path中
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from exp.exp_main import Exp_main
from exp.exp_basic import Exp_Basic
from data_provider.data_loader import load_events_from_file
# 注意：utils.utils 的导入移到强化学习模块导入之后，避免模块冲突

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
_logger = logging.getLogger(__name__)

# 导入强化学习重调度模块
_htb_env_path_saved = None  # 初始化，供后续使用
try:
    import sys as _sys
    # 使用绝对路径，确保路径正确
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    _htb_env_path = os.path.abspath(os.path.join(_base_dir, "htb_environment"))
    _logger.info(f"[INIT] 尝试导入强化学习模块，路径: {_htb_env_path}")
    
    if not os.path.isdir(_htb_env_path):
        raise ImportError(f"htb_environment 目录不存在: {_htb_env_path}")
    
    # 验证关键文件是否存在
    _required_files = [
        os.path.join(_htb_env_path, "snapshot_scheduler.py"),
        os.path.join(_htb_env_path, "environment.py"),
        os.path.join(_htb_env_path, "pipeline", "kg_bridge.py"),
        os.path.join(_htb_env_path, "utils", "util.py")
    ]
    for _file in _required_files:
        if not os.path.isfile(_file):
            raise ImportError(f"必需文件不存在: {_file}")
    
    # 确保 htb_environment 在 sys.path 的最前面，这样相对导入才能工作
    if _htb_env_path in _sys.path:
        _sys.path.remove(_htb_env_path)
    _sys.path.insert(0, _htb_env_path)
    _logger.debug(f"[INIT] 已将 {_htb_env_path} 添加到 sys.path (位置: 0)")
    
    # 清除 Python 的模块缓存，确保新的路径生效
    importlib.invalidate_caches()
    
    # 如果项目根目录的 utils 模块已经被导入，需要先卸载它
    # 这样 Python 才能找到 htb_environment/utils
    if 'utils' in sys.modules:
        utils_module = sys.modules['utils']
        utils_file = getattr(utils_module, '__file__', '')
        # 如果导入的是项目根目录的 utils，而不是 htb_environment/utils，则卸载它
        if utils_file and _htb_env_path not in utils_file:
            _logger.debug(f"[INIT] 卸载已导入的 utils 模块: {utils_file}")
            del sys.modules['utils']
            # 同时卸载 utils 的子模块
            modules_to_remove = [k for k in sys.modules.keys() if k.startswith('utils.')]
            for mod_name in modules_to_remove:
                del sys.modules[mod_name]
            importlib.invalidate_caches()
    
    # 先测试导入 utils 模块，确保包结构正确
    try:
        import utils  # type: ignore
        utils_file = utils.__file__ if hasattr(utils, '__file__') else 'N/A'
        _logger.debug(f"[INIT] 成功导入 utils 模块: {utils_file}")
        # 验证导入的是 htb_environment/utils，而不是项目根目录的 utils
        if _htb_env_path not in str(utils_file):
            raise ImportError(f"导入的 utils 模块路径不正确: {utils_file}，期望包含 {_htb_env_path}")
        from utils import util  # type: ignore
        _logger.debug(f"[INIT] 成功导入 utils.util 模块")
    except Exception as e:
        _logger.warning(f"[INIT] 预导入 utils 模块失败，但继续尝试: {e}")
    
    # 现在可以导入模块了
    from snapshot_scheduler import infer_schedule_from_snapshot  # type: ignore
    _logger.debug("[INIT] 成功导入 snapshot_scheduler.infer_schedule_from_snapshot")
    
    from pipeline.kg_bridge import schedule_to_kg_triples  # type: ignore
    _logger.debug("[INIT] 成功导入 pipeline.kg_bridge.schedule_to_kg_triples")
    
    from environment import ScheduleEnv  # type: ignore
    _logger.debug("[INIT] 成功导入 environment.ScheduleEnv")
    
    from utils.schedule_converter import convert_schedule_with_fixed_logic  # type: ignore
    _logger.debug("[INIT] 成功导入 utils.schedule_converter.convert_schedule_with_fixed_logic")
    
    # 验证导入的函数是否可调用
    if not callable(infer_schedule_from_snapshot):
        raise ImportError("infer_schedule_from_snapshot 不是可调用对象")
    if not callable(schedule_to_kg_triples):
        raise ImportError("schedule_to_kg_triples 不是可调用对象")
    if ScheduleEnv is None:
        raise ImportError("ScheduleEnv 为 None")
    
    _RL_AVAILABLE = True
    _logger.info("[INIT] 强化学习模块导入成功")
    # 保存 _htb_env_path 供后续使用
    _htb_env_path_saved = _htb_env_path
except Exception as e:  # 捕获所有异常，不仅仅是 ImportError
    _logger.error(f"[INIT] 强化学习模块导入失败: {type(e).__name__}: {e}", exc_info=True)
    _RL_AVAILABLE = False
    infer_schedule_from_snapshot = None  # type: ignore
    schedule_to_kg_triples = None  # type: ignore
    ScheduleEnv = None  # type: ignore
    convert_schedule_with_fixed_logic = None  # type: ignore
    # _htb_env_path_saved 已在 try 块外初始化为 None，如果导入失败则保持为 None

# 导入项目根目录的 utils.utils（在强化学习模块导入之后）
# 需要临时调整 sys.path，确保项目根目录在 htb_environment 之前
try:
    _base_dir_for_root_utils = os.path.dirname(os.path.abspath(__file__))
    # 临时将项目根目录移到 sys.path 最前面
    if _base_dir_for_root_utils in sys.path:
        sys.path.remove(_base_dir_for_root_utils)
    sys.path.insert(0, _base_dir_for_root_utils)
    
    # 如果 htb_environment 的 utils 已被导入，需要先卸载它
    if 'utils' in sys.modules:
        utils_module = sys.modules['utils']
        utils_file = getattr(utils_module, '__file__', '')
        # 如果导入的是 htb_environment/utils，则卸载它
        if utils_file and 'htb_environment' in utils_file:
            _logger.debug(f"[INIT] 临时卸载 htb_environment/utils 模块: {utils_file}")
            del sys.modules['utils']
            # 同时卸载 utils 的子模块
            modules_to_remove = [k for k in sys.modules.keys() if k.startswith('utils.')]
            for mod_name in modules_to_remove:
                del sys.modules[mod_name]
            importlib.invalidate_caches()
    
    from utils.utils import detect_potential_conflict, generate_snapshot_from_kg  # type: ignore
    _logger.debug("[INIT] 成功导入项目根目录的 utils.utils")
    
    # 恢复 sys.path 顺序（将项目根目录移回，htb_environment 保持在最前面）
    if _base_dir_for_root_utils in sys.path:
        sys.path.remove(_base_dir_for_root_utils)
    # 将项目根目录添加到 sys.path，但不在最前面（在 htb_environment 之后）
    if _base_dir_for_root_utils not in sys.path:
        # 找到 htb_environment 的位置，在其后插入
        if _htb_env_path_saved and _htb_env_path_saved in sys.path:
            htb_index = sys.path.index(_htb_env_path_saved)
            sys.path.insert(htb_index + 1, _base_dir_for_root_utils)
        else:
            sys.path.append(_base_dir_for_root_utils)
except Exception as e:
    _logger.error(f"[INIT] 导入项目根目录的 utils.utils 失败: {type(e).__name__}: {e}", exc_info=True)
    # 如果导入失败，定义占位函数避免后续错误
    def detect_potential_conflict(event_text: str) -> bool:
        _logger.warning("detect_potential_conflict 函数不可用，返回 False")
        return False
    def generate_snapshot_from_kg(kg_service, current_time_min: float) -> tuple[dict, dict, dict]:
        _logger.warning("generate_snapshot_from_kg 函数不可用，返回空字典和空映射")
        return ({}, {}, {"aircraft": {}, "devices": {}, "resources": {}})


def parse_time_from_event(event_text: str) -> Optional[float]:
    """从事件文本中解析时间，转换为分钟数（从00:00:00开始计算）。
    
    参数:
        event_text: 事件文本，格式如"时间：2025年7月1日 08:00:00，信息：..."
        
    返回:
        float: 时间（分钟），如果解析失败返回None
    """
    if not isinstance(event_text, str):
        return None
    
    # 匹配格式：时间：2025年7月1日 HH:MM:SS
    pattern = r"时间：\d{4}年\d{1,2}月\d{1,2}日\s+(\d{2}):(\d{2}):(\d{2})"
    match = re.search(pattern, event_text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        second = int(match.group(3))
        # 转换为从00:00:00开始的分钟数
        total_minutes = hour * 60 + minute + second / 60.0
        return total_minutes
    
    return None


def save_snapshot(snapshot: dict, output_dir: str, event_id: int, timestamp: str, additional_dir: Optional[str] = None) -> str:
    """保存快照配置文件。
    
    参数:
        snapshot: 快照配置字典
        output_dir: 输出目录
        event_id: 事件ID
        timestamp: 时间戳字符串
        additional_dir: 额外的保存目录（可选），如果提供，会同时保存到该目录
        
    返回:
        str: 保存的文件路径（主目录中的路径）
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"snapshot_event_{event_id:05d}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # 保存到主目录
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    
    _logger.info(f"[SNAPSHOT] 已保存快照配置: {filepath}")
    
    # 如果指定了额外目录，也保存一份
    if additional_dir and additional_dir != output_dir:
        os.makedirs(additional_dir, exist_ok=True)
        additional_filepath = os.path.join(additional_dir, filename)
        with open(additional_filepath, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
        _logger.info(f"[SNAPSHOT] 已保存快照配置副本: {additional_filepath}")
    
    return filepath


def main():
    """主函数：实现冲突检测与重调度流程。"""
    parser = argparse.ArgumentParser(description="任务三：冲突检测与重调度系统")
    parser.add_argument(
        "--events_file",
        type=str,
        default="data_provider/train_texts_task3.jsonl",
        help="事件文件路径（JSONL格式）"
    )
    parser.add_argument(
        "--rules_md_path",
        type=str,
        default="rules_sample.md",
        help="规则文档路径（Markdown格式）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/task2",
        help="输出目录（冲突检测结果和快照配置）"
    )
    parser.add_argument(
        "--snapshot_dir",
        type=str,
        default="results/task2/snapshots",
        help="快照配置文件保存目录"
    )
    parser.add_argument(
        "--base_model_dir",
        type=str,
        default=r"C:\Users\zy\.ssh\haitianbei\models\GLM-4-9B-0414",
        help="大模型目录路径（判定模型）"
    )
    parser.add_argument(
        "--decomp_base_model_dir",
        type=str,
        default=r"C:\Users\zy\.ssh\haitianbei\models\Qwen2_5-3B",
        help="分解模型目录路径"
    )
    parser.add_argument(
        "--judge_base_model_dir",
        type=str,
        default=r"C:\Users\zy\.ssh\haitianbei\models\Qwen2_5-14B-Instruct",
        help="冲突判定模型目录路径（大模型，用于冲突判定推理）"
    )
    parser.add_argument(
        "--neo4j_uri",
        type=str,
        default="bolt://localhost:7687",
        help="Neo4j URI"
    )
    parser.add_argument(
        "--neo4j_user",
        type=str,
        default="neo4j",
        help="Neo4j 用户名"
    )
    parser.add_argument(
        "--neo4j_password",
        type=str,
        default="test123456",
        help="Neo4j 密码"
    )
    parser.add_argument(
        "--neo4j_database",
        type=str,
        default="neo4j",
        help="Neo4j 数据库名"
    )
    parser.add_argument(
        "--reset_kg",
        action="store_true",
        help="重置KG（保留固定节点）"
    )
    parser.add_argument(
        "--skip_kg",
        action="store_true",
        help="跳过KG初始化（仅用于测试）"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="批量处理大小（建议为1，每条事件立即处理）"
    )
    parser.add_argument(
        "--team_name",
        type=str,
        default="default",
        help="队名（用于输出文件命名）"
    )
    
    args = parser.parse_args()
    
    # 1.2 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.snapshot_dir, exist_ok=True)
    
    # 1.3 初始化Exp_main（主实验类，提供模型加载和KG服务）
    _logger.info("[INIT] 初始化Exp_main...")
    try:
        # 创建args对象，设置必要的属性
        exp_args = argparse.Namespace()
        exp_args.root = os.getcwd()
        exp_args.mode = "stream-judge"
        exp_args.events_file = args.events_file
        exp_args.rules_md_path = args.rules_md_path
        exp_args.base_model_dir = args.base_model_dir
        exp_args.decomp_base_model_dir = args.decomp_base_model_dir
        exp_args.judge_base_model_dir = args.judge_base_model_dir
        exp_args.neo4j_uri = args.neo4j_uri
        exp_args.neo4j_user = args.neo4j_user
        exp_args.neo4j_password = args.neo4j_password
        exp_args.neo4j_database = args.neo4j_database
        exp_args.reset_kg = args.reset_kg
        exp_args.skip_kg = args.skip_kg
        exp_args.batch_size = args.batch_size
        exp_args.conflict_judge = True  # 启用冲突判定模式
        exp_args.simple_output = False
        exp_args.enable_decomposer = True  # 启用分解模型（用于冲突判定）
        exp_args.print_decomposition = False
        exp_args.disable_kg_vis = True  # 禁用KG可视化以提高性能
        exp_args.task2_output_dir = args.output_dir  # 设置 task2 输出目录
        exp_args.judge_quant = "int4"  # 使用4bit量化以适配4090显存（14B模型即使8bit量化仍然较大）
        
        # 在初始化前检查强化学习模块状态
        _logger.info("=" * 80)
        _logger.info("任务三：冲突检测与重调度系统")
        _logger.info("=" * 80)
        _logger.info(f"[INIT] 强化学习模块状态: _RL_AVAILABLE={_RL_AVAILABLE}")
        if _RL_AVAILABLE:
            _logger.info(f"[INIT] infer_schedule_from_snapshot 可用: {infer_schedule_from_snapshot is not None}")
            _logger.info(f"[INIT] schedule_to_kg_triples 可用: {schedule_to_kg_triples is not None}")
            _logger.info(f"[INIT] ScheduleEnv 可用: {ScheduleEnv is not None}")
            if infer_schedule_from_snapshot is not None:
                _logger.info(f"[INIT] infer_schedule_from_snapshot 可调用: {callable(infer_schedule_from_snapshot)}")
        else:
            _logger.warning("[INIT] 强化学习模块导入失败，将无法进行重调度")
        
        exp = Exp_main(exp_args)
        _logger.info("[INIT] Exp_main初始化完成")
        
        # 1.4 预加载模型（避免在处理事件时才加载，确保模型只加载一次）
        _logger.info("[INIT] 预加载模型...")
        try:
            exp._build_model()
            _logger.info("[INIT] 模型预加载完成")
        except Exception as e:
            _logger.warning(f"[INIT] 模型预加载失败，将在首次使用时加载: {e}")
    except Exception as e:
        _logger.error(f"[INIT] Exp_main初始化失败: {e}")
        return 1
    
    # ==================== 2. 数据加载阶段 ====================
    # 读取事件文件（JSONL格式，每行一个事件对象，包含id和text字段）
    _logger.info(f"[LOAD] 读取事件文件: {args.events_file}")
    try:
        events_data = []
        with open(args.events_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    event_id = obj.get('id', -1)
                    event_text = obj.get('text', '')
                    if event_text:
                        events_data.append((event_id, event_text))
                except Exception as e:
                    _logger.warning(f"[LOAD] 跳过无效行: {e}")
                    continue
        
        _logger.info(f"[LOAD] 共读取 {len(events_data)} 条事件")
    except Exception as e:
        _logger.error(f"[LOAD] 读取事件文件失败: {e}")
        return 1
    
    # ==================== 3. 预处理阶段 ====================
    # 3.1 准备输出文件（结果文件、快照文件等）
    import time as _time
    timestamp = _time.strftime("%Y%m%d_%H%M%S", _time.localtime())
    conflict_results_file = os.path.join(args.output_dir, f"conflict_results_{timestamp}.jsonl")
    
    # 统计信息
    stats = {
        "total_events": len(events_data),
        "potential_conflicts": 0,
        "confirmed_conflicts": 0,
        "snapshots_generated": 0
    }
    
    # 3.2 重置KG（清理历史动态数据，仅保留固定节点，确保从干净状态开始）
    _logger.info("[INIT] 重置KG，清理历史动态数据，仅保留固定节点...")
    try:
        if exp.kg_service:
            exp.kg_service.reset_graph(keep_fixed=True)
            _logger.info("[INIT] KG重置完成，已清理历史动态关系，仅保留固定节点")
        else:
            _logger.warning("[INIT] KG服务不可用，跳过KG重置")
    except Exception as e:
        _logger.warning(f"[INIT] KG重置失败，将继续处理事件: {e}")
    
    # ==================== 4. 事件处理循环 ====================
    # 对每个事件进行处理：KG更新 -> 白名单检查 -> 冲突检测 -> 大模型判定 -> 重调度 -> KG写回
    _logger.info("[PROCESS] 开始处理事件...")
    for idx, (event_id, event_text) in enumerate(events_data):
        try:
            _logger.info(f"[#{idx+1}/{len(events_data)}] 处理事件 ID={event_id}")
            
            # 解析时间
            current_time_min = parse_time_from_event(event_text)
            if current_time_min is None:
                _logger.warning(f"[#{idx+1}] 无法解析事件时间，跳过时间相关处理")
                current_time_min = 0.0
            
            # 0. 先更新KG（对所有事件，这是必须的，因为后续调度依赖KG状态）
            try:
                if exp.kg_service:
                    # 从事件文本中提取三元组并更新KG
                    exp.kg_service.extract_and_update(event_text)
                    _logger.debug(f"[#{idx+1}] KG已更新")
                    
                    # 4.2.1 清理离散节点（每次更新后都清理，保持KG干净，避免孤立节点）
                    try:
                        deleted_count = exp.kg_service.cleanup_isolated_nodes()
                        if deleted_count > 0:
                            _logger.debug(f"[#{idx+1}] 清理离散节点: 删除了 {deleted_count} 个节点")
                    except Exception as e:
                        _logger.warning(f"[#{idx+1}] 清理离散节点失败: {e}")
                else:
                    _logger.warning(f"[#{idx+1}] KG服务不可用，跳过KG更新")
            except Exception as e:
                _logger.warning(f"[#{idx+1}] KG更新失败: {e}")
            
            # 4.3 自动修复过期的损坏（基于持续时间的自动修复，例如停机位损坏30分钟后自动恢复）
            try:
                if exp.kg_service and current_time_min is not None and current_time_min > 0:
                    repair_result = exp.kg_service.check_and_repair_expired_failures(current_time_min)
                    if repair_result.get("total_repaired", 0) > 0:
                        _logger.info(f"[#{idx+1}] 自动修复统计: {repair_result.get('total_repaired', 0)} 个损坏已修复（停机位: {len(repair_result.get('repaired_stands', []))}, 设备: {len(repair_result.get('repaired_devices', []))}）")
            except Exception as e:
                _logger.warning(f"[#{idx+1}] 自动修复过期损坏时出错: {e}")
            
            # 4.3.5 检查是否是仅生成甘特图的事件（在白名单检查之前）
            is_gantt_only = event_id in GANTT_ONLY_EVENT_IDS
            
            if is_gantt_only:
                # 仅生成甘特图的事件：生成快照 -> 调用强化学习 -> 生成甘特图（跳过冲突判定和KG写回）
                scenario_num = get_scenario_number(event_id)
                _logger.info(f"[#{idx+1}] 事件 ID={event_id} 是仅生成甘特图的事件（场景{scenario_num}的前一个事件），开始生成甘特图流程")
                
                result_record = {
                    "event_id": event_id,
                    "event_text": event_text,
                    "time_min": current_time_min,
                    "potential_conflict": False,
                    "confirmed_conflict": False,
                    "compliance": None,
                    "reasons": [],
                    "snapshot_file": None,
                    "reschedule_file": None,
                    "reschedule_makespan": None,
                    "reschedule_reward": None,
                    "output_a_file": None,
                    "output_s_file": None,
                    "output_p_file": None,
                    "judge_output_file": None,
                    "gantt_only": True
                }
                
                # 生成快照配置
                snapshot = None
                try:
                    if not exp.kg_service:
                        _logger.error(f"[#{idx+1}] KG服务不可用，无法生成快照")
                    else:
                        snapshot, id_to_name_map, all_mappings = generate_snapshot_from_kg(
                            exp.kg_service,
                            current_time_min
                        )
                        
                        if snapshot:
                            snapshot["_aircraft_id_mapping"] = id_to_name_map
                            snapshot["_aircraft_names"] = list(id_to_name_map.values())
                            snapshot["_all_mappings"] = all_mappings
                            _logger.info(f"[#{idx+1}] 快照配置已生成（用于甘特图和S文件）")
                            
                            # 4.3.5.1 保存S文件（快照配置的JSON字符串，按照事件编号命名）
                            try:
                                output_s_filename = f"事件{event_id}_S_{args.team_name}.txt"
                                output_s_filepath = os.path.join(args.output_dir, output_s_filename)
                                os.makedirs(args.output_dir, exist_ok=True)
                                with open(output_s_filepath, 'w', encoding='utf-8') as f:
                                    json.dump(snapshot, f, ensure_ascii=False, indent=2)
                                _logger.info(f"[#{idx+1}] S文件已保存: {output_s_filepath}")
                                result_record["output_s_file"] = output_s_filepath
                            except Exception as e:
                                _logger.warning(f"[#{idx+1}] 保存S文件失败: {e}")
                                result_record["output_s_file"] = None
                        else:
                            _logger.error(f"[#{idx+1}] 快照生成返回None")
                except Exception as e:
                    _logger.error(f"[#{idx+1}] 生成快照配置失败: {e}", exc_info=True)
                
                # 调用强化学习重调度（仅用于生成甘特图）
                if snapshot and _RL_AVAILABLE and infer_schedule_from_snapshot is not None and callable(infer_schedule_from_snapshot):
                    try:
                        num_planes = len(snapshot.get("planes", []))
                        from argparse import Namespace
                        rl_args = Namespace(
                            n_agents=num_planes,
                            batch_mode=False,
                            arrival_gap_min=2,
                            result_dir=args.output_dir,
                            result_name="rl_reschedule",
                            alg="qmix",
                            n_actions=0,
                            state_shape=0,
                            obs_shape=0,
                            episode_limit=1000,
                            enable_deps=True,
                            enable_mutex=True,
                            enable_dynres=True,
                            enable_space=True,
                            enable_long_occupy=False,
                            enable_disturbance=False,
                            penalty_idle_per_min=0.05,
                            epsilon_start=None,
                            epsilon_end=None,
                            epsilon_anneal_steps=None,
                            epsilon_anneal_scale=None
                        )
                        
                        _logger.info(f"[#{idx+1}] ========== 强化学习重调度开始（仅生成甘特图） ==========")
                        
                        reschedule_info = infer_schedule_from_snapshot(
                            rl_args,
                            snapshot,
                            policy_fn=None,
                            max_steps=None
                        )
                        
                        # 生成甘特图
                        episodes_situation_for_gantt = reschedule_info.get("episodes_situation", [])
                        devices_situation_for_gantt = reschedule_info.get("devices_situation", [])
                        
                        if episodes_situation_for_gantt and convert_schedule_with_fixed_logic is not None and callable(convert_schedule_with_fixed_logic):
                            # 创建临时目录和文件
                            temp_dir = os.path.join(args.output_dir, "temp_p_file_generation")
                            os.makedirs(temp_dir, exist_ok=True)
                            
                            temp_evaluate_json = os.path.join(temp_dir, f"temp_evaluate_{event_id:05d}_{timestamp}.json")
                            temp_plan_eval_json = os.path.join(temp_dir, f"temp_plan_eval_{event_id:05d}_{timestamp}.json")
                            
                            try:
                                # 创建临时环境对象以获取 move_job_id
                                if ScheduleEnv is None:
                                    raise ImportError("ScheduleEnv 不可用")
                                temp_env_for_converter = ScheduleEnv(rl_args)
                                temp_env_for_converter.reset(rl_args.n_agents)
                                try:
                                    move_jid = int(temp_env_for_converter.jobs_obj.code2id().get("ZY_M", 1))
                                except Exception:
                                    move_jid = 1
                                
                                # 创建临时的 evaluate.json 文件
                                temp_evaluate_data = {
                                    "schedule_results": [episodes_situation_for_gantt],
                                    "devices_results": [devices_situation_for_gantt] if devices_situation_for_gantt else [None],
                                    "evaluate_reward": [reschedule_info.get("reward", 0.0)],
                                    "evaluate_makespan": [reschedule_info.get("time", 0.0)]
                                }
                                
                                with open(temp_evaluate_json, 'w', encoding='utf-8') as f:
                                    json.dump(temp_evaluate_data, f, ensure_ascii=False, indent=2)
                                
                                # 创建甘特图输出目录
                                gantt_output_dir = os.path.join(args.output_dir, "gantt")
                                os.makedirs(gantt_output_dir, exist_ok=True)
                                
                                # 调用 convert_schedule_with_fixed_logic 生成甘特图
                                convert_schedule_with_fixed_logic(
                                    temp_evaluate_json,
                                    temp_plan_eval_json,
                                    rl_args.n_agents,
                                    out_dir=gantt_output_dir,
                                    also_plot=True,
                                    move_job_id=move_jid,
                                    batch_size_per_batch=None
                                )
                                
                                # 将生成的甘特图文件重命名为包含场景编号和事件ID的版本
                                original_gantt_png = os.path.join(gantt_output_dir, "gantt.png")
                                original_gantt_stand_usage_png = os.path.join(gantt_output_dir, "gantt_stand_usage.png")
                                
                                gantt_png_path = os.path.join(gantt_output_dir, f"gantt_scenario_{scenario_num}_event_{event_id}.png")
                                gantt_stand_usage_png_path = os.path.join(gantt_output_dir, f"gantt_stand_usage_scenario_{scenario_num}_event_{event_id}.png")
                                
                                # 重命名文件
                                try:
                                    if os.path.exists(original_gantt_png):
                                        if os.path.exists(gantt_png_path):
                                            os.remove(gantt_png_path)
                                        os.rename(original_gantt_png, gantt_png_path)
                                        _logger.info(f"[#{idx+1}] 甘特图已生成: {gantt_png_path}")
                                    
                                    if os.path.exists(original_gantt_stand_usage_png):
                                        if os.path.exists(gantt_stand_usage_png_path):
                                            os.remove(gantt_stand_usage_png_path)
                                        os.rename(original_gantt_stand_usage_png, gantt_stand_usage_png_path)
                                        _logger.info(f"[#{idx+1}] 停机位使用甘特图已生成: {gantt_stand_usage_png_path}")
                                except Exception as rename_e:
                                    _logger.warning(f"[#{idx+1}] 重命名甘特图文件失败: {rename_e}")
                                
                                # 清理临时文件
                                try:
                                    if os.path.exists(temp_evaluate_json):
                                        os.remove(temp_evaluate_json)
                                    if os.path.exists(temp_plan_eval_json):
                                        os.remove(temp_plan_eval_json)
                                    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                                        os.rmdir(temp_dir)
                                except Exception:
                                    pass
                                
                                _logger.info(f"[#{idx+1}] ========== 甘特图生成完成 ==========")
                                
                                # 4.3.5.2 保存P文件（强化学习重调度计划的JSON，按照事件编号命名）
                                try:
                                    output_p_filename = f"事件{event_id}_P_{args.team_name}.json"
                                    output_p_filepath = os.path.join(args.output_dir, output_p_filename)
                                    os.makedirs(args.output_dir, exist_ok=True)
                                    
                                    # 读取生成的 plan_eval.json 内容
                                    if os.path.exists(temp_plan_eval_json):
                                        with open(temp_plan_eval_json, 'r', encoding='utf-8') as f:
                                            plan_eval_data = json.load(f)
                                        
                                        # 写入到 P 文件
                                        with open(output_p_filepath, 'w', encoding='utf-8') as f:
                                            json.dump(plan_eval_data, f, ensure_ascii=False, indent=2)
                                        
                                        _logger.info(f"[#{idx+1}] P文件已保存（plan_eval格式）: {output_p_filepath}")
                                        result_record["output_p_file"] = output_p_filepath
                                    else:
                                        _logger.warning(f"[#{idx+1}] 临时 plan_eval.json 文件不存在，使用原始格式")
                                        # 回退到原始格式
                                        reschedule_output = {
                                            "time": reschedule_info.get("time"),
                                            "reward": reschedule_info.get("reward"),
                                            "episodes_situation": reschedule_info.get("episodes_situation", []),
                                            "devices_situation": reschedule_info.get("devices_situation", []),
                                            "event_id": event_id
                                        }
                                        with open(output_p_filepath, 'w', encoding='utf-8') as f:
                                            json.dump(reschedule_output, f, ensure_ascii=False, indent=2)
                                        _logger.info(f"[#{idx+1}] P文件已保存（原始格式）: {output_p_filepath}")
                                        result_record["output_p_file"] = output_p_filepath
                                except Exception as p_file_e:
                                    _logger.error(f"[#{idx+1}] 保存P文件失败: {p_file_e}", exc_info=True)
                                    result_record["output_p_file"] = None
                                
                            except Exception as gantt_e:
                                _logger.error(f"[#{idx+1}] 生成甘特图失败: {gantt_e}", exc_info=True)
                        else:
                            _logger.warning(f"[#{idx+1}] 无法生成甘特图：episodes_situation为空或convert_schedule_with_fixed_logic不可用")
                        
                    except Exception as rl_e:
                        _logger.error(f"[#{idx+1}] 强化学习重调度失败: {rl_e}", exc_info=True)
                else:
                    _logger.warning(f"[#{idx+1}] 无法进行强化学习重调度：快照为空或强化学习模块不可用")
                
                # 保存结果记录
                try:
                    with open(conflict_results_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
                except Exception as e:
                    _logger.warning(f"[#{idx+1}] 保存结果失败: {e}")
                
                continue  # 跳过后续的冲突检测和大模型调度流程
            
            # 4.4 白名单检查（只有白名单中的事件ID才会触发冲突检测和大模型调度）
            is_in_whitelist = event_id in SCHEDULE_TRIGGER_EVENT_IDS
            
            if not is_in_whitelist:
                # 不在白名单中：只更新KG，跳过冲突检测和大模型调用（节省计算资源）
                _logger.info(f"[#{idx+1}] 事件 ID={event_id} 不在调度触发白名单中，仅更新KG，跳过冲突检测和大模型调度")
                result_record = {
                    "event_id": event_id,
                    "event_text": event_text,
                    "time_min": current_time_min,
                    "potential_conflict": False,
                    "confirmed_conflict": False,
                    "compliance": None,
                    "reasons": [],
                    "snapshot_file": None,
                    "reschedule_file": None,
                    "reschedule_makespan": None,
                    "reschedule_reward": None,
                    "skipped_reason": "not_in_whitelist"
                }
                # 保存结果并继续下一个事件
                try:
                    with open(conflict_results_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
                except Exception as e:
                    _logger.warning(f"[#{idx+1}] 保存结果失败: {e}")
                continue
            
            # ==================== 4.5 冲突检测与大模型调度流程（白名单事件） ====================
            _logger.info(f"[#{idx+1}] 事件 ID={event_id} 在白名单中，开始冲突检测和大模型调度流程")
            
            # 4.5.1 初步冲突检测（使用关键词匹配，快速筛选可能冲突的事件，减少大模型调用）
            has_potential_conflict = detect_potential_conflict(event_text)
            
            result_record = {
                "event_id": event_id,
                "event_text": event_text,
                "time_min": current_time_min,
                "potential_conflict": has_potential_conflict,
                "confirmed_conflict": False,
                "compliance": None,
                "reasons": [],
                "snapshot_file": None,
                "reschedule_file": None,
                "reschedule_makespan": None,
                "reschedule_reward": None,
                "output_a_file": None,
                "output_s_file": None,
                "output_p_file": None,
                "judge_output_file": None
            }
            
            if has_potential_conflict:
                stats["potential_conflicts"] += 1
                _logger.info(f"[#{idx+1}] 检测到潜在冲突，调用大模型判定...")
                
                # 4.5.2 调用大模型进行冲突判定（结合KG上下文和规则文档，判定是否确实存在冲突）
                try:
                    # 使用stream_judge_conflicts进行判定（抽取时空信息并判定冲突）
                    for ev, output in exp.stream_judge_conflicts(
                        events_iter=[event_text],
                        rules_md_path=args.rules_md_path,
                        batch_size=1,
                        simple_output=False,
                        show_decomposition=False
                    ):
                        # 4.5.2.1 记录大模型原始输出（用于调试和问题定位）
                        output_preview = str(output)[:500] if output else "None"
                        _logger.debug(f"[#{idx+1}] 大模型原始输出（前500字符）: {output_preview}")
                        
                        # 4.5.2.2 保存独立输出文件（每次调用都保存，无论是否判定为冲突，用于后续分析）
                        try:
                            judge_output_filename = f"judge_output_event_{event_id:05d}_{timestamp}.txt"
                            judge_output_filepath = os.path.join(args.output_dir, judge_output_filename)
                            os.makedirs(args.output_dir, exist_ok=True)
                            with open(judge_output_filepath, 'w', encoding='utf-8') as f:
                                f.write(str(output) if output else "")
                            _logger.info(f"[#{idx+1}] 大模型输出已保存（独立文件）: {judge_output_filepath}")
                            result_record["judge_output_file"] = judge_output_filepath
                        except Exception as e:
                            _logger.warning(f"[#{idx+1}] 保存大模型独立输出文件失败: {e}")
                            result_record["judge_output_file"] = None
                        
                        # 4.5.2.3 解析大模型输出（提取compliance、reason、suggest等字段）
                        parsed = exp._parse_judge_output(output)
                        compliance = parsed.get("compliance")
                        reasons_len = parsed.get("reasons_len", 0)
                        parsed_success = parsed.get("parsed", False)
                        
                        # 记录解析结果
                        _logger.info(f"[#{idx+1}] 解析结果: parsed={parsed_success}, compliance={compliance}, reasons_len={reasons_len}")
                        
                        # 如果解析失败，记录警告和原始输出
                        if not parsed_success or compliance is None:
                            _logger.warning(f"[#{idx+1}] 大模型输出解析失败或compliance为None")
                            _logger.warning(f"[#{idx+1}] 原始输出（完整）: {output}")
                            _logger.warning(f"[#{idx+1}] 解析结果详情: {parsed}")
                        
                        result_record["compliance"] = compliance
                        result_record["confirmed_conflict"] = (compliance == "冲突")
                        
                        # 4.5.2.4 保存A文件（冲突判定大模型的完整输出，用于评分）
                        if result_record["confirmed_conflict"]:
                            scenario_num = get_scenario_number(event_id)
                            if scenario_num > 0:
                                try:
                                    output_a_filename = f"任务三测试场景{scenario_num}_A_{args.team_name}.txt"
                                    output_a_filepath = os.path.join(args.output_dir, output_a_filename)
                                    os.makedirs(args.output_dir, exist_ok=True)
                                    with open(output_a_filepath, 'w', encoding='utf-8') as f:
                                        f.write(str(output) if output else "")
                                    _logger.info(f"[#{idx+1}] A文件已保存: {output_a_filepath}")
                                    result_record["output_a_file"] = output_a_filepath
                                except Exception as e:
                                    _logger.warning(f"[#{idx+1}] 保存A文件失败: {e}")
                                    result_record["output_a_file"] = None
                        
                        # ==================== 4.6 确认冲突后的重调度流程 ====================
                        if result_record["confirmed_conflict"]:
                            stats["confirmed_conflicts"] += 1
                            _logger.info(f"[#{idx+1}] 确认冲突，生成快照配置并调用强化学习重调度...")
                            
                            # 4.6.1 生成快照配置（从KG中提取当前状态，转换为强化学习环境需要的格式）
                            snapshot = None
                            snapshot_file = None
                            try:
                                if not exp.kg_service:
                                    _logger.error(f"[#{idx+1}] KG服务不可用，无法生成快照")
                                    result_record["snapshot_file"] = None
                                else:
                                    # 生成快照（包含飞机、停机位、设备、资源等状态，以及映射关系）
                                    snapshot, id_to_name_map, all_mappings = generate_snapshot_from_kg(
                                        exp.kg_service,
                                        current_time_min
                                    )
                                    
                                    # 保存映射信息到快照配置中（飞机ID->名称、设备名称、资源名称等，用于后续写回KG时恢复原始名称）
                                    snapshot["_aircraft_id_mapping"] = id_to_name_map
                                    snapshot["_aircraft_names"] = list(id_to_name_map.values())  # 所有飞机的原始名称列表
                                    snapshot["_all_mappings"] = all_mappings  # 完整的映射关系（包含飞机、设备、资源）
                                    
                                    if not snapshot:
                                        _logger.error(f"[#{idx+1}] 快照生成返回None")
                                        result_record["snapshot_file"] = None
                                    else:
                                        # 保存快照（同时保存到snapshot_dir和output_dir）
                                        # 冲突时的快照使用特殊前缀
                                        conflict_snapshot_filename = f"conflict_snapshot_{event_id:05d}_{timestamp}.json"
                                        conflict_snapshot_filepath = os.path.join(args.output_dir, conflict_snapshot_filename)
                                        
                                        # 确保输出目录存在
                                        os.makedirs(args.output_dir, exist_ok=True)
                                        
                                        with open(conflict_snapshot_filepath, 'w', encoding='utf-8') as f:
                                            json.dump(snapshot, f, ensure_ascii=False, indent=2)
                                        
                                        # 验证文件是否成功创建
                                        if os.path.exists(conflict_snapshot_filepath):
                                            file_size = os.path.getsize(conflict_snapshot_filepath)
                                            _logger.info(f"[#{idx+1}] 冲突快照配置已生成: {conflict_snapshot_filepath}")
                                            _logger.info(f"[#{idx+1}] 文件大小: {file_size} 字节")
                                        else:
                                            _logger.error(f"[#{idx+1}] 快照文件创建失败: {conflict_snapshot_filepath}")
                                        
                                        # 同时保存到snapshot_dir
                                        snapshot_file = save_snapshot(
                                            snapshot,
                                            args.snapshot_dir,
                                            event_id,
                                            timestamp,
                                            additional_dir=None  # 已经在output_dir保存了
                                        )
                                        
                                        result_record["snapshot_file"] = conflict_snapshot_filepath
                                        stats["snapshots_generated"] += 1
                                        
                                        # 4.6.1.1 保存S文件（快照配置的JSON字符串，用于评分）
                                        # 注意：虽然内容是JSON格式，但文件扩展名是 .txt（按评分规则要求）
                                        scenario_num = get_scenario_number(event_id)
                                        if scenario_num > 0:
                                            try:
                                                output_s_filename = f"任务三测试场景{scenario_num}_S_{args.team_name}.txt"
                                                output_s_filepath = os.path.join(args.output_dir, output_s_filename)
                                                os.makedirs(args.output_dir, exist_ok=True)
                                                with open(output_s_filepath, 'w', encoding='utf-8') as f:
                                                    json.dump(snapshot, f, ensure_ascii=False, indent=2)
                                                _logger.info(f"[#{idx+1}] S文件已保存: {output_s_filepath}")
                                                result_record["output_s_file"] = output_s_filepath
                                            except Exception as e:
                                                _logger.warning(f"[#{idx+1}] 保存S文件失败: {e}")
                                                result_record["output_s_file"] = None
                            except Exception as e:
                                _logger.error(f"[#{idx+1}] 生成快照配置失败: {e}", exc_info=True)
                                result_record["snapshot_file"] = None
                            
                            # 4.6.2 调用强化学习重调度（基于快照配置，使用RL算法生成新的调度方案）
                            # 详细检查每个条件（确保所有必需的模块和函数都可用）
                            _rl_check_snapshot = snapshot is not None
                            _rl_check_available = _RL_AVAILABLE
                            _rl_check_function = infer_schedule_from_snapshot is not None
                            _rl_check_callable = callable(infer_schedule_from_snapshot) if infer_schedule_from_snapshot is not None else False
                            
                            _logger.debug(f"[#{idx+1}] 强化学习模块检查: snapshot={_rl_check_snapshot}, _RL_AVAILABLE={_rl_check_available}, function_exists={_rl_check_function}, callable={_rl_check_callable}")
                            
                            if snapshot and _RL_AVAILABLE and infer_schedule_from_snapshot is not None and callable(infer_schedule_from_snapshot):
                                try:
                                    # 4.6.2.1 记录快照基本信息（用于日志和调试）
                                    num_planes = len(snapshot.get("planes", []))
                                    num_occupied_stands = sum(1 for v in snapshot.get("stand_occupancy", {}).values() if v is not None)
                                    num_blocked_stands = len(snapshot.get("blocked_stands", []))
                                    num_fixed_devices = len(snapshot.get("devices", {}).get("fixed", {}))
                                    num_mobile_devices = len(snapshot.get("devices", {}).get("mobile", {}))
                                    current_time = snapshot.get("time", current_time_min)
                                    
                                    _logger.info(f"[#{idx+1}] ========== 强化学习重调度开始 ==========")
                                    _logger.info(f"[#{idx+1}] 快照信息:")
                                    _logger.info(f"[#{idx+1}]   - 当前时间: {current_time:.2f} 分钟")
                                    _logger.info(f"[#{idx+1}]   - 飞机数量: {num_planes}")
                                    _logger.info(f"[#{idx+1}]   - 占用停机位: {num_occupied_stands}")
                                    _logger.info(f"[#{idx+1}]   - 封锁停机位: {num_blocked_stands}")
                                    _logger.info(f"[#{idx+1}]   - 固定设备: {num_fixed_devices}")
                                    _logger.info(f"[#{idx+1}]   - 移动设备: {num_mobile_devices}")
                                    
                                    # 创建args对象
                                    from argparse import Namespace
                                    rl_args = Namespace(
                                        n_agents=num_planes,
                                        batch_mode=False,
                                        arrival_gap_min=2,
                                        result_dir=args.output_dir,
                                        result_name="rl_reschedule",
                                        alg="qmix",
                                        n_actions=0,  # 占位值，环境会重新计算
                                        state_shape=0,  # 占位值
                                        obs_shape=0,  # 占位值
                                        episode_limit=1000,  # 最大步数
                                        enable_deps=True,
                                        enable_mutex=True,
                                        enable_dynres=True,
                                        enable_space=True,
                                        enable_long_occupy=False,
                                        enable_disturbance=False,
                                        penalty_idle_per_min=0.05,
                                        # epsilon 相关属性（设置为 None，让 get_mixer_args 使用默认值）
                                        epsilon_start=None,
                                        epsilon_end=None,
                                        epsilon_anneal_steps=None,
                                        epsilon_anneal_scale=None
                                    )
                                    
                                    _logger.info(f"[#{idx+1}] 强化学习参数:")
                                    _logger.info(f"[#{idx+1}]   - n_agents: {rl_args.n_agents}")
                                    _logger.info(f"[#{idx+1}]   - episode_limit: {rl_args.episode_limit}")
                                    _logger.info(f"[#{idx+1}]   - enable_deps: {rl_args.enable_deps}")
                                    _logger.info(f"[#{idx+1}]   - enable_mutex: {rl_args.enable_mutex}")
                                    _logger.info(f"[#{idx+1}]   - enable_dynres: {rl_args.enable_dynres}")
                                    _logger.info(f"[#{idx+1}]   - enable_space: {rl_args.enable_space}")
                                    
                                    # 4.6.2.2 调用强化学习重调度（使用QMIX算法，基于快照配置生成新的调度方案）
                                    reschedule_info = infer_schedule_from_snapshot(
                                        rl_args,
                                        snapshot,
                                        policy_fn=None,  # 使用默认的greedy_idle_policy
                                        max_steps=None  # 使用环境默认的episode_limit
                                    )
                                    
                                    # 4.6.2.3 保存重调度结果（包含makespan、reward、episodes_situation、devices_situation等）
                                    reschedule_output = {
                                        "time": reschedule_info.get("time"),
                                        "reward": reschedule_info.get("reward"),
                                        "episodes_situation": reschedule_info.get("episodes_situation", []),
                                        "devices_situation": reschedule_info.get("devices_situation", []),
                                        "event_id": event_id,
                                        "original_snapshot_file": snapshot_file
                                    }
                                    
                                    reschedule_filename = f"reschedule_event_{event_id:05d}_{timestamp}.json"
                                    reschedule_filepath = os.path.join(args.output_dir, reschedule_filename)
                                    with open(reschedule_filepath, 'w', encoding='utf-8') as f:
                                        json.dump(reschedule_output, f, ensure_ascii=False, indent=2)
                                    
                                    result_record["reschedule_file"] = reschedule_filepath
                                    result_record["reschedule_makespan"] = reschedule_info.get("time")
                                    result_record["reschedule_reward"] = reschedule_info.get("reward")
                                    
                                    # 4.6.2.4 保存P文件（强化学习重调度计划的JSON，用于评分）
                                    scenario_num = get_scenario_number(event_id)
                                    if scenario_num > 0:
                                        try:
                                            output_p_filename = f"任务三测试场景{scenario_num}_P_{args.team_name}.json"
                                            output_p_filepath = os.path.join(args.output_dir, output_p_filename)
                                            os.makedirs(args.output_dir, exist_ok=True)
                                            
                                            # 使用 convert_schedule_with_fixed_logic 生成 plan_eval.json 格式
                                            if convert_schedule_with_fixed_logic is not None and callable(convert_schedule_with_fixed_logic):
                                                # 创建临时目录和文件
                                                temp_dir = os.path.join(args.output_dir, "temp_p_file_generation")
                                                os.makedirs(temp_dir, exist_ok=True)
                                                
                                                temp_evaluate_json = os.path.join(temp_dir, f"temp_evaluate_{event_id:05d}_{timestamp}.json")
                                                temp_plan_eval_json = os.path.join(temp_dir, f"temp_plan_eval_{event_id:05d}_{timestamp}.json")
                                                
                                                try:
                                                    # 创建临时环境对象以获取 move_job_id
                                                    if ScheduleEnv is None:
                                                        raise ImportError("ScheduleEnv 不可用")
                                                    temp_env_for_converter = ScheduleEnv(rl_args)
                                                    temp_env_for_converter.reset(rl_args.n_agents)
                                                    try:
                                                        move_jid = int(temp_env_for_converter.jobs_obj.code2id().get("ZY_M", 1))
                                                    except Exception:
                                                        move_jid = 1
                                                    
                                                    # 创建临时的 evaluate.json 文件
                                                    episodes_situation_for_converter = reschedule_info.get("episodes_situation", [])
                                                    devices_situation_for_converter = reschedule_info.get("devices_situation", [])
                                                    temp_evaluate_data = {
                                                        "schedule_results": [episodes_situation_for_converter],
                                                        "devices_results": [devices_situation_for_converter] if devices_situation_for_converter else [None],
                                                        "evaluate_reward": [reschedule_info.get("reward", 0.0)],
                                                        "evaluate_makespan": [reschedule_info.get("time", 0.0)]
                                                    }
                                                    
                                                    with open(temp_evaluate_json, 'w', encoding='utf-8') as f:
                                                        json.dump(temp_evaluate_data, f, ensure_ascii=False, indent=2)
                                                    
                                                    _logger.debug(f"[#{idx+1}] 已创建临时 evaluate.json: {temp_evaluate_json}")
                                                    
                                                    # 创建甘特图输出目录（固定路径：results/task2/gantt）
                                                    # 使用 args.output_dir 作为基础路径，添加 gantt 子目录
                                                    gantt_output_dir = os.path.join(args.output_dir, "gantt")
                                                    os.makedirs(gantt_output_dir, exist_ok=True)
                                                    _logger.debug(f"[#{idx+1}] 甘特图输出目录: {gantt_output_dir}")
                                                    
                                                    # 调用 convert_schedule_with_fixed_logic 生成 plan_eval.json 和甘特图
                                                    convert_schedule_with_fixed_logic(
                                                        temp_evaluate_json,
                                                        temp_plan_eval_json,
                                                        rl_args.n_agents,
                                                        out_dir=gantt_output_dir,  # 甘特图保存到专门的目录
                                                        also_plot=True,  # 生成甘特图
                                                        move_job_id=move_jid,
                                                        batch_size_per_batch=None
                                                    )
                                                    
                                                    # 将生成的甘特图文件重命名为包含场景编号的版本
                                                    original_gantt_png = os.path.join(gantt_output_dir, "gantt.png")
                                                    original_gantt_stand_usage_png = os.path.join(gantt_output_dir, "gantt_stand_usage.png")
                                                    
                                                    gantt_png_path = os.path.join(gantt_output_dir, f"gantt_scenario_{scenario_num}.png")
                                                    gantt_stand_usage_png_path = os.path.join(gantt_output_dir, f"gantt_stand_usage_scenario_{scenario_num}.png")
                                                    
                                                    # 重命名文件（如果原文件存在）
                                                    try:
                                                        if os.path.exists(original_gantt_png):
                                                            if os.path.exists(gantt_png_path):
                                                                os.remove(gantt_png_path)  # 如果目标文件已存在，先删除
                                                            os.rename(original_gantt_png, gantt_png_path)
                                                            _logger.info(f"[#{idx+1}] 甘特图已生成: {gantt_png_path}")
                                                        else:
                                                            _logger.warning(f"[#{idx+1}] 甘特图文件不存在: {original_gantt_png}")
                                                        
                                                        if os.path.exists(original_gantt_stand_usage_png):
                                                            if os.path.exists(gantt_stand_usage_png_path):
                                                                os.remove(gantt_stand_usage_png_path)  # 如果目标文件已存在，先删除
                                                            os.rename(original_gantt_stand_usage_png, gantt_stand_usage_png_path)
                                                            _logger.info(f"[#{idx+1}] 停机位使用甘特图已生成: {gantt_stand_usage_png_path}")
                                                        else:
                                                            _logger.warning(f"[#{idx+1}] 停机位使用甘特图文件不存在: {original_gantt_stand_usage_png}")
                                                    except Exception as rename_e:
                                                        _logger.warning(f"[#{idx+1}] 重命名甘特图文件失败: {rename_e}")
                                                        # 如果重命名失败，至少记录原始文件路径
                                                        if os.path.exists(original_gantt_png):
                                                            _logger.info(f"[#{idx+1}] 甘特图文件（未重命名）: {original_gantt_png}")
                                                        if os.path.exists(original_gantt_stand_usage_png):
                                                            _logger.info(f"[#{idx+1}] 停机位使用甘特图文件（未重命名）: {original_gantt_stand_usage_png}")
                                                    
                                                    _logger.debug(f"[#{idx+1}] 已生成临时 plan_eval.json: {temp_plan_eval_json}")
                                                    
                                                    # 读取生成的 plan_eval.json 内容
                                                    if os.path.exists(temp_plan_eval_json):
                                                        with open(temp_plan_eval_json, 'r', encoding='utf-8') as f:
                                                            plan_eval_data = json.load(f)
                                                        
                                                        # 写入到 P 文件
                                                        with open(output_p_filepath, 'w', encoding='utf-8') as f:
                                                            json.dump(plan_eval_data, f, ensure_ascii=False, indent=2)
                                                        
                                                        _logger.info(f"[#{idx+1}] P文件已保存（plan_eval格式）: {output_p_filepath}")
                                                        result_record["output_p_file"] = output_p_filepath
                                                    else:
                                                        _logger.warning(f"[#{idx+1}] 临时 plan_eval.json 文件不存在，使用原始格式")
                                                        # 回退到原始格式
                                                        with open(output_p_filepath, 'w', encoding='utf-8') as f:
                                                            json.dump(reschedule_output, f, ensure_ascii=False, indent=2)
                                                        result_record["output_p_file"] = output_p_filepath
                                                    
                                                    # 清理临时文件
                                                    try:
                                                        if os.path.exists(temp_evaluate_json):
                                                            os.remove(temp_evaluate_json)
                                                        if os.path.exists(temp_plan_eval_json):
                                                            os.remove(temp_plan_eval_json)
                                                        # 如果临时目录为空，尝试删除它（但可能失败，因为可能有其他文件）
                                                        try:
                                                            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                                                                os.rmdir(temp_dir)
                                                        except Exception:
                                                            pass  # 忽略删除目录失败
                                                        _logger.debug(f"[#{idx+1}] 已清理临时文件")
                                                    except Exception as cleanup_e:
                                                        _logger.warning(f"[#{idx+1}] 清理临时文件失败: {cleanup_e}")
                                                
                                                except Exception as converter_e:
                                                    _logger.error(f"[#{idx+1}] 使用 convert_schedule_with_fixed_logic 生成 P 文件失败: {converter_e}", exc_info=True)
                                                    # 回退到原始格式
                                                    with open(output_p_filepath, 'w', encoding='utf-8') as f:
                                                        json.dump(reschedule_output, f, ensure_ascii=False, indent=2)
                                                    _logger.info(f"[#{idx+1}] P文件已保存（原始格式，转换失败）: {output_p_filepath}")
                                                    result_record["output_p_file"] = output_p_filepath
                                            else:
                                                # convert_schedule_with_fixed_logic 不可用，使用原始格式
                                                _logger.warning(f"[#{idx+1}] convert_schedule_with_fixed_logic 不可用，使用原始格式")
                                                with open(output_p_filepath, 'w', encoding='utf-8') as f:
                                                    json.dump(reschedule_output, f, ensure_ascii=False, indent=2)
                                                _logger.info(f"[#{idx+1}] P文件已保存（原始格式）: {output_p_filepath}")
                                                result_record["output_p_file"] = output_p_filepath
                                        
                                        except Exception as e:
                                            _logger.warning(f"[#{idx+1}] 保存P文件失败: {e}", exc_info=True)
                                            result_record["output_p_file"] = None
                                    
                                    # 记录调度结果摘要
                                    makespan = reschedule_info.get("time", 0.0)
                                    reward = reschedule_info.get("reward", 0.0)
                                    episodes_situation = reschedule_info.get("episodes_situation", [])
                                    devices_situation = reschedule_info.get("devices_situation", [])
                                    num_episodes = len(episodes_situation)
                                    num_device_events = len(devices_situation)
                                    
                                    _logger.info(f"[#{idx+1}] ========== 强化学习重调度完成 ==========")
                                    _logger.info(f"[#{idx+1}] 调度结果摘要:")
                                    _logger.info(f"[#{idx+1}]   - Makespan: {makespan:.2f} 分钟")
                                    _logger.info(f"[#{idx+1}]   - Reward: {reward:.4f}")
                                    _logger.info(f"[#{idx+1}]   - 调度事件数: {num_episodes}")
                                    _logger.info(f"[#{idx+1}]   - 设备事件数: {num_device_events}")
                                    _logger.info(f"[#{idx+1}]   - 结果文件: {reschedule_filepath}")
                                    
                                    # 统计受影响的飞机数量
                                    if episodes_situation:
                                        affected_planes = set()
                                        for ep in episodes_situation:
                                            if len(ep) >= 4:
                                                plane_id = ep[3]  # episodes_situation格式: (time, job_id, site_id, plane_id, ...)
                                                if plane_id is not None:
                                                    affected_planes.add(plane_id)
                                        _logger.info(f"[#{idx+1}]   - 受影响飞机数: {len(affected_planes)}")
                                    
                                    # ==================== 4.7 将重调度结果写回KG ====================
                                    # 将强化学习生成的调度方案转换为三元组，更新KG，形成闭环
                                    try:
                                        episodes_situation = reschedule_info.get("episodes_situation", [])
                                        if episodes_situation and schedule_to_kg_triples is not None and ScheduleEnv is not None:
                                            _logger.info(f"[#{idx+1}] 开始将重调度结果写回KG...")
                                            
                                            # 4.7.1 从快照中获取飞机ID映射（RL环境使用数字ID，KG使用原始名称，需要转换）
                                            id_to_name_map = snapshot.get("_aircraft_id_mapping", {})
                                            
                                            if not id_to_name_map:
                                                _logger.warning(f"[#{idx+1}] 未找到飞机ID映射信息，将使用数字ID作为飞机名称")
                                            
                                            # 4.7.2 将episodes_situation中的数字ID转换为原始飞机名称（飞机0 -> "飞机A001"）
                                            converted_episodes = []
                                            for ep in episodes_situation:
                                                if len(ep) >= 4:
                                                    time_min, job_id, site_id, plane_id = ep[0], ep[1], ep[2], ep[3]
                                                    # 将数字ID转换为原始名称
                                                    original_name = id_to_name_map.get(plane_id, f"飞机{plane_id}")
                                                    # 重新构建元组，使用原始名称
                                                    if len(ep) >= 6:
                                                        converted_ep = (time_min, job_id, site_id, original_name, ep[4], ep[5])
                                                    else:
                                                        converted_ep = (time_min, job_id, site_id, original_name) + tuple(ep[4:])
                                                    converted_episodes.append(converted_ep)
                                                else:
                                                    converted_episodes.append(ep)
                                            
                                            _logger.debug(f"[#{idx+1}] 已将 {len(converted_episodes)} 个调度事件中的飞机ID转换为原始名称")
                                            
                                            # 4.7.3 创建临时环境对象用于转换三元组（获取job_id到code的映射等）
                                            # 注意：这里使用rl_args，因为env需要这些参数来初始化
                                            temp_env = ScheduleEnv(rl_args)
                                            temp_env.reset(rl_args.n_agents)
                                            
                                            # 4.7.4 将转换后的episodes_situation转换为三元组（飞机名称、作业、停机位等）
                                            # schedule_to_kg_triples 期望 plane_id 是数字，但我们已经转换为名称字符串
                                            # 需要创建一个包装函数来处理名称格式
                                            def convert_episodes_with_names(episodes, env):
                                                """将包含名称的episodes转换为三元组（主语-谓词-宾语格式）"""
                                                triples = []
                                                id2code = env.jobs_obj.id2code()  # job_id -> "ZY_*"
                                                for ep in sorted(episodes, key=lambda x: x[0]):
                                                    if len(ep) < 4:
                                                        continue
                                                    time_min, job_id, site_id, plane_name = ep[0], ep[1], ep[2], ep[3]
                                                    code = id2code.get(job_id, f"ZY_{job_id}")
                                                    
                                                    # plane_name 已经是 "飞机A001" 格式，直接使用
                                                    plane = plane_name if isinstance(plane_name, str) and plane_name.startswith("飞机") else f"飞机{plane_name}"
                                                    
                                                    site = "跑道Z" if site_id == 0 else (
                                                        f"停机位{site_id}" if 1 <= site_id <= 28 else f"跑道{site_id}")
                                                    
                                                    # 基本事实
                                                    triples.append((plane, "动作", code))
                                                    triples.append((plane, "时间", f"{time_min:.2f}"))
                                                    
                                                    # 站位/跑道使用
                                                    if site_id == 0 or site_id >= 29:
                                                        # 着陆点或跑道作业
                                                        triples.append((plane, "着陆跑道" if code == "ZY_Z" else "使用跑道", site))
                                                    else:
                                                        # 进入或停靠某个停机位
                                                        move_min = ep[5] if len(ep) > 5 else 0
                                                        if move_min and move_min > 0:
                                                            triples.append((plane, "到达停机位", site))
                                                        else:
                                                            triples.append((plane, "当前停机位", site))
                                                    
                                                    # 作业相关
                                                    if code.startswith("ZY_"):
                                                        triples.append((plane, "当前作业", code))
                                                
                                                return triples
                                            
                                            triples = convert_episodes_with_names(converted_episodes, temp_env)
                                            
                                            # 4.7.5 将三元组写回KG（更新飞机的状态、位置、作业等信息）
                                            if triples and exp.kg_service:
                                                exp.kg_service.kg.update_with_triples(triples)
                                                
                                                # 统计更新的信息
                                                num_triples = len(triples)
                                                updated_planes = set()
                                                updated_resources = set()
                                                for s, p, o in triples:
                                                    # 提取飞机信息
                                                    if isinstance(s, str) and s.startswith("飞机"):
                                                        updated_planes.add(s)
                                                    # 提取资源信息（停机位、设备等）
                                                    if isinstance(o, str):
                                                        if "停机位" in o or "跑道" in o:
                                                            updated_resources.add(o)
                                                        elif o.startswith("FR") or o.startswith("MR"):
                                                            updated_resources.add(o)
                                                
                                                _logger.info(f"[#{idx+1}] ========== KG写回完成 ==========")
                                                _logger.info(f"[#{idx+1}] 更新统计:")
                                                _logger.info(f"[#{idx+1}]   - 三元组数量: {num_triples}")
                                                _logger.info(f"[#{idx+1}]   - 更新飞机数: {len(updated_planes)}")
                                                _logger.info(f"[#{idx+1}]   - 更新资源数: {len(updated_resources)}")
                                                if updated_planes:
                                                    _logger.info(f"[#{idx+1}]   - 更新的飞机: {', '.join(sorted(list(updated_planes)[:5]))}{'...' if len(updated_planes) > 5 else ''}")
                                            else:
                                                _logger.warning(f"[#{idx+1}] 未生成有效的三元组或KG服务不可用")
                                        else:
                                            _logger.warning(f"[#{idx+1}] 重调度结果为空或转换函数不可用，跳过KG写回")
                                    except Exception as e:
                                        _logger.error(f"[#{idx+1}] 将重调度结果写回KG失败: {e}", exc_info=True)
                                except Exception as e:
                                    _logger.error(f"[#{idx+1}] 强化学习重调度失败: {e}", exc_info=True)
                                    result_record["reschedule_file"] = None
                            elif not _RL_AVAILABLE:
                                _logger.warning(f"[#{idx+1}] 强化学习模块不可用，跳过重调度")
                                _logger.debug(f"[#{idx+1}] 详细状态: _RL_AVAILABLE={_RL_AVAILABLE}, infer_schedule_from_snapshot={infer_schedule_from_snapshot is not None if 'infer_schedule_from_snapshot' in globals() else 'N/A'}")
                            elif not snapshot:
                                _logger.warning(f"[#{idx+1}] 快照配置生成失败，跳过重调度")
                        
                        break  # 单条事件，处理完即退出（batch_size=1，一次只处理一条）
                except Exception as e:
                    _logger.error(f"[#{idx+1}] 大模型冲突判定失败: {e}")
                    result_record["compliance"] = "ERROR"
            
            # 注意：KG更新已在事件处理开始时就完成（步骤4.2），这里不需要重复更新
            
            # 4.8 保存处理结果（每条事件处理完成后，立即保存到结果文件）
            try:
                with open(conflict_results_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
            except Exception as e:
                _logger.warning(f"[#{idx+1}] 保存结果失败: {e}")
            
        except Exception as e:
            _logger.error(f"[#{idx+1}] 处理事件失败: {e}")
            continue
    
    # ==================== 5. 后处理阶段 ====================
    # 输出统计信息（总事件数、潜在冲突数、确认冲突数、生成快照数等）
    _logger.info("=" * 60)
    _logger.info("[STATS] 处理完成统计:")
    _logger.info(f"  总事件数: {stats['total_events']}")
    _logger.info(f"  潜在冲突: {stats['potential_conflicts']}")
    _logger.info(f"  确认冲突: {stats['confirmed_conflicts']}")
    _logger.info(f"  生成快照: {stats['snapshots_generated']}")
    _logger.info(f"  结果文件: {conflict_results_file}")
    _logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

