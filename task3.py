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
"""

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
    def generate_snapshot_from_kg(kg_service, current_time_min: float) -> tuple[dict, dict]:
        _logger.warning("generate_snapshot_from_kg 函数不可用，返回空字典和空映射")
        return ({}, {})


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
    parser = argparse.ArgumentParser(description="任务二：冲突检测与重调度系统")
    parser.add_argument(
        "--events_file",
        type=str,
        default="data_provider/train_texts_task3.jsonl",
        help="事件文件路径（JSONL格式）"
    )
    parser.add_argument(
        "--rules_md_path",
        type=str,
        default="海天杯-技术资料.md",
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
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.snapshot_dir, exist_ok=True)
    
    # 初始化Exp_main
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
        
        # 在初始化前检查强化学习模块状态
        _logger.info("=" * 80)
        _logger.info("任务二：冲突检测与重调度系统")
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
        
        # 显式触发模型加载，避免在处理事件时才加载（确保模型只加载一次）
        _logger.info("[INIT] 预加载模型...")
        try:
            exp._build_model()
            _logger.info("[INIT] 模型预加载完成")
        except Exception as e:
            _logger.warning(f"[INIT] 模型预加载失败，将在首次使用时加载: {e}")
    except Exception as e:
        _logger.error(f"[INIT] Exp_main初始化失败: {e}")
        return 1
    
    # 读取事件文件
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
    
    # 准备输出文件
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
    
    # 事件处理循环
    _logger.info("[PROCESS] 开始处理事件...")
    for idx, (event_id, event_text) in enumerate(events_data):
        try:
            _logger.info(f"[#{idx+1}/{len(events_data)}] 处理事件 ID={event_id}")
            
            # 解析时间
            current_time_min = parse_time_from_event(event_text)
            if current_time_min is None:
                _logger.warning(f"[#{idx+1}] 无法解析事件时间，跳过时间相关处理")
                current_time_min = 0.0
            
            # 1. 初步冲突检测
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
                "reschedule_reward": None
            }
            
            if has_potential_conflict:
                stats["potential_conflicts"] += 1
                _logger.info(f"[#{idx+1}] 检测到潜在冲突，调用大模型判定...")
                
                # 2. 调用大模型进行冲突判定
                try:
                    # 使用stream_judge_conflicts进行判定
                    for ev, output in exp.stream_judge_conflicts(
                        events_iter=[event_text],
                        rules_md_path=args.rules_md_path,
                        batch_size=1,
                        simple_output=False,
                        show_decomposition=False
                    ):
                        # 记录大模型原始输出（用于调试）
                        output_preview = str(output)[:500] if output else "None"
                        _logger.debug(f"[#{idx+1}] 大模型原始输出（前500字符）: {output_preview}")
                        
                        # 解析输出
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
                        
                        if result_record["confirmed_conflict"]:
                            stats["confirmed_conflicts"] += 1
                            _logger.info(f"[#{idx+1}] 确认冲突，生成快照配置并调用强化学习重调度...")
                            
                            # 3. 生成快照配置（冲突时使用更详细的文件名）
                            snapshot = None
                            snapshot_file = None
                            try:
                                if not exp.kg_service:
                                    _logger.error(f"[#{idx+1}] KG服务不可用，无法生成快照")
                                    result_record["snapshot_file"] = None
                                else:
                                    snapshot, id_to_name_map = generate_snapshot_from_kg(
                                        exp.kg_service,
                                        current_time_min
                                    )
                                    
                                    # 保存飞机名称映射信息到快照配置中（用于后续写回KG时恢复名称）
                                    snapshot["_aircraft_id_mapping"] = id_to_name_map
                                    snapshot["_aircraft_names"] = list(id_to_name_map.values())  # 所有飞机的原始名称列表
                                    
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
                            except Exception as e:
                                _logger.error(f"[#{idx+1}] 生成快照配置失败: {e}", exc_info=True)
                                result_record["snapshot_file"] = None
                            
                            # 4. 调用强化学习重调度
                            # 详细检查每个条件
                            _rl_check_snapshot = snapshot is not None
                            _rl_check_available = _RL_AVAILABLE
                            _rl_check_function = infer_schedule_from_snapshot is not None
                            _rl_check_callable = callable(infer_schedule_from_snapshot) if infer_schedule_from_snapshot is not None else False
                            
                            _logger.debug(f"[#{idx+1}] 强化学习模块检查: snapshot={_rl_check_snapshot}, _RL_AVAILABLE={_rl_check_available}, function_exists={_rl_check_function}, callable={_rl_check_callable}")
                            
                            if snapshot and _RL_AVAILABLE and infer_schedule_from_snapshot is not None and callable(infer_schedule_from_snapshot):
                                try:
                                    # 记录快照基本信息
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
                                    
                                    # 调用重调度
                                    reschedule_info = infer_schedule_from_snapshot(
                                        rl_args,
                                        snapshot,
                                        policy_fn=None,  # 使用默认的greedy_idle_policy
                                        max_steps=None  # 使用环境默认的episode_limit
                                    )
                                    
                                    # 保存重调度结果
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
                                    
                                    # 5. 将重调度结果写回KG
                                    try:
                                        episodes_situation = reschedule_info.get("episodes_situation", [])
                                        if episodes_situation and schedule_to_kg_triples is not None and ScheduleEnv is not None:
                                            _logger.info(f"[#{idx+1}] 开始将重调度结果写回KG...")
                                            
                                            # 从快照中获取飞机ID映射（数字ID -> 原始名称）
                                            id_to_name_map = snapshot.get("_aircraft_id_mapping", {})
                                            
                                            if not id_to_name_map:
                                                _logger.warning(f"[#{idx+1}] 未找到飞机ID映射信息，将使用数字ID作为飞机名称")
                                            
                                            # 将episodes_situation中的数字ID转换为原始飞机名称
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
                                            
                                            # 创建临时环境对象用于转换三元组
                                            # 注意：这里使用rl_args，因为env需要这些参数来初始化
                                            temp_env = ScheduleEnv(rl_args)
                                            temp_env.reset(rl_args.n_agents)
                                            
                                            # 将转换后的episodes_situation转换为三元组
                                            # schedule_to_kg_triples 期望 plane_id 是数字，但我们已经转换为名称字符串
                                            # 需要创建一个包装函数来处理名称格式
                                            def convert_episodes_with_names(episodes, env):
                                                """将包含名称的episodes转换为三元组"""
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
                                            
                                            if triples and exp.kg_service:
                                                # 写回KG
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
                        
                        break  # 单条事件，处理完即退出
                except Exception as e:
                    _logger.error(f"[#{idx+1}] 大模型冲突判定失败: {e}")
                    result_record["compliance"] = "ERROR"
            
            # 4. 更新KG（无论是否冲突）
            try:
                if exp.kg_service:
                    exp.kg_service.extract_and_update(event_text)
                    _logger.debug(f"[#{idx+1}] KG已更新")
            except Exception as e:
                _logger.warning(f"[#{idx+1}] KG更新失败: {e}")
            
            # 保存结果
            try:
                with open(conflict_results_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
            except Exception as e:
                _logger.warning(f"[#{idx+1}] 保存结果失败: {e}")
            
        except Exception as e:
            _logger.error(f"[#{idx+1}] 处理事件失败: {e}")
            continue
    
    # 输出统计信息
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

