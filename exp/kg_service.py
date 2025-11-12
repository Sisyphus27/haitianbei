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
        try:
            if focus_entities:
                for ent in focus_entities:
                    nb = self.kg.neighbors(ent)
                    if nb.get("out"):
                        for s, p, o in nb["out"][: max(1, limit) // 2]:
                            lines.append(f"{s} -[{p}]-> {o}")
                    if nb.get("in"):
                        for s, p, o in nb["in"][: max(1, limit) // 2]:
                            lines.append(f"{s} -[{p}]-> {o}")
            else:
                snap = self.kg.graph_snapshot()
                lines.append(
                    f"[SNAPSHOT] nodes={snap.get('nodes_count',0)} edges={snap.get('edges_count',0)}"
                )
        except Exception as _e:  # pragma: no cover
            pass
        ctx = "\n".join(lines[:limit])
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
        """导出 PNG，并支持通过环境变量控制绘制参数。

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

        try:
            return self.kg.export_png(out_png, **kw)
        except TypeError:
            # 兼容旧签名
            return self.kg.export_png(out_png)

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
