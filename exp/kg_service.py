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

    def export_png(self, out_png: str) -> None:
        if hasattr(self.kg, "export_png"):
            self.kg.export_png(out_png)

    def graph_snapshot(self) -> Dict[str, Any]:
        try:
            return self.kg.graph_snapshot() or {}
        except Exception:
            return {}
