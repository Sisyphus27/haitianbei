"""
KG 本地服务封装：对 Dataset_KG 提供统一访问接口与轻缓存。
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

        def _push(line: str) -> None:
            txt = str(line).strip()
            if not txt or txt in seen:
                return
            seen.add(txt)
            lines.append(txt)
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

                    out_edges = nb.get("out") or []
                    in_edges = nb.get("in") or []
                    # 大幅减少每条实体的边数量
                    per_direction_cap = max(1, limit_int // (len(focus_entities[:5]) * 2 + 1))
                    for s, p, o in out_edges[:per_direction_cap]:
                        _push(f"{s} -[{p}]-> {o}")
                        if len(lines) >= limit_int:
                            break
                    for s, p, o in in_edges[:per_direction_cap]:
                        _push(f"{s} -[{p}]-> {o}")
                        if len(lines) >= limit_int:
                            break
                    # 跳过属性查询（减少耗时）
            else:
                snap = self.kg.graph_snapshot()
                _push(
                    f"[SNAPSHOT] nodes={snap.get('nodes_count',0)} edges={snap.get('edges_count',0)}"
                )
        except Exception as _e:  # pragma: no cover
            pass
        if not lines:
            _push("(当前图为空或仅有固定节点)")

        ctx = "\n".join(lines[:limit_int])
        if not ctx:
            ctx = "(当前图为空或仅有固定节点)"
        text = "【KG状态】\n" + ctx
        if self.ctx_ttl_sec > 0:
            self._ctx_cache[key] = (now, text)
        return text

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
        """清理不连通的节点（如时间节点），但保留必要的节点类型。"""
        if hasattr(self.kg, "cleanup_isolated_nodes"):
            try:
                return self.kg.cleanup_isolated_nodes()
            except Exception:
                return 0
        return 0
