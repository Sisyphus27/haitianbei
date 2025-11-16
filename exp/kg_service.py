"""
知识图谱服务封装：提供KG查询和更新的统一接口

功能：
1. 上下文查询：根据关注的实体查询相关的KG上下文信息
2. 缓存机制：使用TTL缓存减少重复查询
3. 结果更新：将LLM处理结果写回知识图谱
4. 图谱管理：提供重置、清理等管理功能

在stream-judge模式中的作用：
- 为每个事件查询相关的KG上下文（如飞机的当前状态、停机位占用情况等）
- 将抽取的三元组和判定结果更新到KG中
- 通过缓存机制优化查询性能
"""
import time
import logging
from typing import List, Optional, Tuple, Dict, Any

try:
    from data_provider.data_loader import Dataset_KG  # type: ignore
except Exception:  # pragma: no cover
    Dataset_KG = None  # type: ignore


class KGServiceLocal:
    """
    轻量封装 Dataset_KG，提供统一的上下文文本/校验/写回接口，并带有简单 TTL 缓存。
    """

    def __init__(self, kg: Any, *, ctx_ttl_sec: int = 2):
        self.kg = kg
        self.ctx_ttl_sec = max(0, int(ctx_ttl_sec))
        # 缓存: key=(tuple(focus) or None, limit) -> (ts, text)
        self._ctx_cache: Dict[Tuple[Optional[Tuple[str, ...]], int], Tuple[float, str]] = {}
        self._log = logging.getLogger(__name__)

    def _make_key(self, focus_entities: Optional[List[str]], limit: int) -> Tuple[Optional[Tuple[str, ...]], int]:
        key_focus: Optional[Tuple[str, ...]]
        if focus_entities:
            # 规范化：去重保持顺序
            seen = set()
            ordered: List[str] = []
            for x in focus_entities:
                x = str(x).strip()
                if x and x not in seen:
                    ordered.append(x)
                    seen.add(x)
            key_focus = tuple(ordered)
        else:
            key_focus = None
        return (key_focus, int(limit))

    def get_context_text(self, *, focus_entities: Optional[List[str]] = None, limit: int = 200) -> str:
        """
        返回可读 KG 上下文文本，尽量复用缓存，结构与原 _kg_text_context 一致。
        优化：减少查询次数，限制返回信息量。
        """
        key = self._make_key(focus_entities, limit)
        now = time.time()
        if self.ctx_ttl_sec > 0:
            item = self._ctx_cache.get(key)
            if item and (now - item[0] <= self.ctx_ttl_sec):
                self._log.debug("[KGCTX] cache=HIT focus=%s limit=%d", key[0], limit)
                return item[1]

        self._log.debug("[KGCTX] cache=MISS focus=%s limit=%d", key[0], limit)
        lines: List[str] = []
        seen: set[str] = set()
        
        # 统计各部分的数量
        stats = {
            "node_props": 0,  # 节点属性
            "out_edges": 0,   # 出边关系
            "in_edges": 0,    # 入边关系
            "related_props": 0,  # 关联节点属性
            "snapshot": 0,    # 快照信息
        }

        def _push(line: str, category: str = "other") -> None:
            txt = str(line).strip()
            if not txt or txt in seen:
                return
            seen.add(txt)
            lines.append(txt)
            # 统计
            if category in stats:
                stats[category] += 1
            # 提前停止：达到limit就返回
            if len(lines) >= limit:
                return

        # 优化：减少limit，提升速度
        limit_int = max(1, min(int(limit), 100))  # 最多100条信息

        try:
            if focus_entities:
                # 只处理前5个实体，避免过多查询
                for ent in focus_entities[:5]:
                    if len(lines) >= limit_int:
                        break
                    ent_name = str(ent).strip()
                    if not ent_name:
                        continue
                    try:
                        nb = self.kg.neighbors(ent_name)
                    except Exception:
                        nb = {}

                    # 优先查询节点属性（特别是资源和飞机的状态属性：损坏状态、占用状态等）
                    try:
                        node_props = self._get_node_properties(ent_name)
                        if node_props:
                            _push(node_props, "node_props")
                            if len(lines) >= limit_int:
                                break
                    except Exception:
                        pass

                    out_edges = nb.get("out") or []
                    in_edges = nb.get("in") or []
                    
                    # 大幅减少边关系：只保留关键业务关系，跳过邻接关系
                    # 关键业务关系：CURRENT_JOB, PERFORMS_JOB, REQUIRES_RESOURCE, USING_DEVICE等
                    key_relations = {"CURRENT_JOB", "PERFORMS_JOB", "REQUIRES_RESOURCE", "USING_DEVICE", 
                                    "ASSIGNED_TO_JOB_INSTANCE", "PERFORMS_JOB_INSTANCE"}
                    
                    # 只保留关键业务关系的出边（最多2条）
                    key_edge_count = 0
                    for s, p, o in out_edges:
                        if len(lines) >= limit_int or key_edge_count >= 2:
                            break
                        if p in key_relations:
                            _push(f"{s} -[{p}]-> {o}", "out_edges")
                            key_edge_count += 1
                    
                    # 入边：只保留关键业务关系（最多1条）
                    for s, p, o in in_edges:
                        if len(lines) >= limit_int:
                            break
                        if p in key_relations:
                            _push(f"{s} -[{p}]-> {o}", "in_edges")
                            break  # 只保留第一条关键关系
                    
                    # 查询关联资源节点的属性（优先保留状态属性）
                    # 从关键边中提取资源节点（停机位、跑道、设备等）
                    try:
                        related_nodes = set()
                        # 从关键边中提取资源节点
                        for s, p, o in out_edges:
                            if isinstance(o, str) and (
                                o.startswith("停机位") or o.startswith("跑道") or 
                                o.startswith("设备") or o.startswith("MR") or o.startswith("牵引车")
                            ):
                                related_nodes.add(o)
                                if len(related_nodes) >= 3:  # 最多提取3个资源节点
                                    break
                        
                        # 优先查询这些资源节点的状态属性
                        for related_node in list(related_nodes)[:3]:
                            if len(lines) >= limit_int:
                                break
                            try:
                                related_props = self._get_node_properties(related_node)
                                if related_props:
                                    _push(related_props, "related_props")
                                    if len(lines) >= limit_int:
                                        break
                            except Exception:
                                pass
                    except Exception:
                        pass
            else:
                snap = self.kg.graph_snapshot()
                snapshot_text = f"[SNAPSHOT] nodes={snap.get('nodes_count',0)} edges={snap.get('edges_count',0)}"
                _push(snapshot_text, "snapshot")
        except Exception as _e:  # pragma: no cover
            pass
        if not lines:
            _push("(当前图为空或仅有固定节点)")

        ctx = "\n".join(lines[:limit_int])
        if not ctx:
            ctx = "(当前图为空或仅有固定节点)"
        
        # 添加KG上下文组成分析日志
        total_items = len(lines)
        if total_items > 0:
            node_props_pct = (stats["node_props"] / total_items) * 100
            out_edges_pct = (stats["out_edges"] / total_items) * 100
            in_edges_pct = (stats["in_edges"] / total_items) * 100
            related_props_pct = (stats["related_props"] / total_items) * 100
            snapshot_pct = (stats["snapshot"] / total_items) * 100
            
            self._log.info("[KGCTX] KG上下文组成分析：")
            self._log.info(f"[KGCTX]   总条目数: {total_items}")
            self._log.info(f"[KGCTX]   - 节点属性: {stats['node_props']} 条 ({node_props_pct:.1f}%)")
            self._log.info(f"[KGCTX]   - 出边关系: {stats['out_edges']} 条 ({out_edges_pct:.1f}%)")
            self._log.info(f"[KGCTX]   - 入边关系: {stats['in_edges']} 条 ({in_edges_pct:.1f}%)")
            self._log.info(f"[KGCTX]   - 关联节点属性: {stats['related_props']} 条 ({related_props_pct:.1f}%)")
            self._log.info(f"[KGCTX]   - 快照信息: {stats['snapshot']} 条 ({snapshot_pct:.1f}%)")
            self._log.info(f"[KGCTX]   - 总字符数: {len(ctx)} 字符")
            
            # 警告：如果边关系占比过高，说明可能有冗余
            edges_pct = out_edges_pct + in_edges_pct
            if edges_pct > 60:
                self._log.warning(f"[KGCTX] 警告：边关系占比过高（{edges_pct:.1f}%），可能包含冗余信息")
        
        # 改进KG文本格式：添加明确的标记，区分输入和输出，避免格式混淆
        # 注意：这个标记会在_format_conflict_prompt_with_mode中进一步处理，这里保持原格式以便兼容
        text = "【KG状态】\n" + ctx
        if self.ctx_ttl_sec > 0:
            self._ctx_cache[key] = (now, text)
        return text

    def _get_node_properties(self, node_name: str) -> Optional[str]:
        """查询节点的属性信息，特别是损坏状态。
        
        参数:
            node_name: 节点名称
            
        返回:
            节点属性字符串，格式：节点名称 {属性1: 值1, 属性2: 值2}
        """
        if not node_name or not isinstance(node_name, str):
            return None
        
        try:
            # 尝试从KG获取节点信息
            # Dataset_KG可能有_driver属性（Neo4j驱动）
            driver = None
            if hasattr(self.kg, '_driver'):
                driver = self.kg._driver
            elif hasattr(self.kg, 'driver'):
                driver = self.kg.driver
            elif hasattr(self.kg, 'get_driver'):
                driver = self.kg.get_driver()
            
            if driver:
                with driver.session() as sess:
                    # 查询节点及其属性
                    result = sess.run(
                        """
                        MATCH (n {name: $name})
                        RETURN labels(n) as labels, properties(n) as props
                        LIMIT 1
                        """,
                        {"name": node_name}
                    )
                    record = result.single()
                    if record:
                        props = record.get("props", {})
                        labels = record.get("labels", [])
                        
                        # 过滤重要属性（特别是状态相关的）
                        important_props = {}
                        for key, value in props.items():
                            # 包含损坏、状态、不可用等关键词的属性
                            key_lower = key.lower()
                            if any(keyword in key_lower for keyword in ["damaged", "damage", "损坏", "状态", "status", "available", "可用", "failure", "故障"]):
                                important_props[key] = value
                            # 时间相关的属性
                            elif any(keyword in key_lower for keyword in ["time", "时间", "duration", "持续"]):
                                important_props[key] = value
                        
                        if important_props:
                            # 格式化输出
                            prop_strs = []
                            for k, v in important_props.items():
                                prop_strs.append(f"{k}={v}")
                            props_str = ", ".join(prop_strs)
                            label_str = ":".join(labels) if labels else "Node"
                            return f"{node_name} ({label_str}) {{{props_str}}}"
        except Exception as e:
            self._log.debug(f"[KGCTX] 查询节点属性失败 {node_name}: {e}")
        
        return None

    # 透传与简单封装
    def extract_and_update(self, event_text: str) -> Any:
        return self.kg.extract_and_update(event_text)

    def check_event_conflicts(self, event_text: str) -> Dict[str, Any]:
        try:
            out = self.kg.check_event_conflicts(event_text) or {}
            return out
        except Exception:
            return {}

    def export_png(self, out_png: str) -> dict | None:
        """导出 PNG，或在导出模式为 cypher 时导出 Cypher 查询文件；同时支持通过环境变量控制绘制参数。

        环境变量（可选）：
        - KG_EXPORT_FIGSIZE: 例如 "16,10" 或 "16x10"
        - KG_EXPORT_DPI: 如 300
        - KG_EXPORT_LAYOUT: spring|kamada|circular|spectral|shell
        - KG_EXPORT_HIDE_EDGE_LABELS: "1"/"true" 隐藏边标签
        - KG_EXPORT_NODE_FONT: 节点字体大小（int）
        - KG_EXPORT_EDGE_FONT: 边字体大小（int）
        - KG_EXPORT_MAX_EDGES: 单图最大边数抽样（int）
        """
        import os

        if not hasattr(self.kg, "export_png"):
            return None

        figsize = os.environ.get("KG_EXPORT_FIGSIZE")
        dpi = os.environ.get("KG_EXPORT_DPI")
        layout = os.environ.get("KG_EXPORT_LAYOUT") or "spring"
        hide_edge_labels = os.environ.get("KG_EXPORT_HIDE_EDGE_LABELS", "0").lower() in {"1", "true", "yes"}
        node_font = os.environ.get("KG_EXPORT_NODE_FONT")
        edge_font = os.environ.get("KG_EXPORT_EDGE_FONT")
        max_edges = os.environ.get("KG_EXPORT_MAX_EDGES")

        kw = {}
        if figsize:
            kw["figsize"] = figsize
        if dpi and dpi.isdigit():
            kw["dpi"] = int(dpi)
        if layout:
            kw["layout"] = layout
        kw["hide_edge_labels"] = hide_edge_labels
        if node_font and node_font.isdigit():
            kw["node_label_font_size"] = int(node_font)
        if edge_font and edge_font.isdigit():
            kw["edge_label_font_size"] = int(edge_font)
        if max_edges and max_edges.isdigit():
            kw["max_edges"] = int(max_edges)

        return self.kg.export_png(out_png, **kw)

    def graph_snapshot(self) -> Dict[str, Any]:
        try:
            return self.kg.graph_snapshot() or {}
        except Exception:
            return {}

    def reset_graph(self, keep_fixed: bool = True) -> None:
        """重置底层图谱（透传）。

        参数
        ----
        keep_fixed: 是否保留固定/初始节点。与底层 `Dataset_KG.reset_graph` 语义保持一致。
        """
        if hasattr(self.kg, "reset_graph"):
            try:  # pragma: no cover - 仅防御性容错
                self.kg.reset_graph(keep_fixed=keep_fixed)
            except Exception:
                pass

    # ========== 设备占用/释放（可选） ==========
    def occupy_device(self, device_name: str, *, aircraft: str | None = None, job_code: str | None = None) -> None:
        """标记设备占用，并可选绑定飞机与作业。

        - device_name: FRxx/MRxx
        - aircraft: 可选，如 "飞机A001"
        - job_code: 可选，如 "ZY10"
        """
        if hasattr(self.kg, "occupy_device"):
            try:
                self.kg.occupy_device(device_name, aircraft=aircraft, job_code=job_code)
            except Exception:
                pass

    def release_device(self, device_name: str, *, aircraft: str | None = None) -> None:
        """释放设备占用。
        
        参数:
        - device_name: 设备名称
        - aircraft: 可选，指定释放该设备的飞机；如果为None，则释放所有飞机对该设备的使用
        """
        if hasattr(self.kg, "release_device"):
            try:
                self.kg.release_device(device_name, aircraft=aircraft)
            except Exception:
                pass

    # ========== 跑道占用/释放（可选） ==========
    def occupy_runway(self, runway_name: str, *, aircraft: str | None = None) -> None:
        if hasattr(self.kg, "occupy_runway"):
            try:
                self.kg.occupy_runway(runway_name, aircraft=aircraft)
            except Exception:
                pass

    def release_runway(self, runway_name: str) -> None:
        if hasattr(self.kg, "release_runway"):
            try:
                self.kg.release_runway(runway_name)
            except Exception:
                pass

    def cleanup_isolated_nodes(self) -> int:
        """清理不连通的节点（如时间节点），但保留必要的节点类型。
        
        返回:
        -----
        int: 删除的节点数量
        """
        if hasattr(self.kg, "cleanup_isolated_nodes"):
            try:
                deleted_count = self.kg.cleanup_isolated_nodes()
                if deleted_count > 0:
                    self._log.debug(f"[KG] 清理离散节点: 删除了 {deleted_count} 个节点")
                return deleted_count
            except Exception as e:
                self._log.warning(f"[KG] 清理离散节点失败: {e}")
                return 0
        return 0

    def check_and_repair_expired_failures(self, current_time_min: float) -> Dict[str, Any]:
        """检查并自动修复过期的损坏。
        
        参数:
            current_time_min: 当前时间（分钟，从00:00:00开始计算）
            
        返回:
            dict: 包含修复统计信息的字典
                - repaired_stands: 修复的停机位列表
                - repaired_devices: 修复的设备列表
                - total_repaired: 总修复数量
        """
        if hasattr(self.kg, "check_and_repair_expired_failures"):
            try:
                return self.kg.check_and_repair_expired_failures(current_time_min)
            except Exception as e:
                self._log.warning(f"检查并修复过期损坏失败: {e}")
                return {
                    "repaired_stands": [],
                    "repaired_devices": [],
                    "total_repaired": 0
                }
        return {
            "repaired_stands": [],
            "repaired_devices": [],
            "total_repaired": 0
        }
