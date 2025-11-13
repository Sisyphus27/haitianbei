"""重放事件文本并逐条写入知识图谱，同时导出图谱快照。

Author: zy
Date: 2025-11-11 10:46:58
LastEditTime: 2025-11-11 10:47:02
LastEditors: zy
Description:
FilePath: \\haitianbei\\scripts\\replay_events_to_kg.py

Usage
-----
python scripts/replay_events_to_kg.py \
    --events-file data_provider/train_texts_conflict_aug.jsonl \
    --output-dir results/kg_replay \
    --neo4j-uri bolt://localhost:7687 \
    --neo4j-user neo4j \
    --neo4j-password neo4j

脚本特性
~~~~~~~~
* 支持 JSONL 事件文件，优先读取 `text`/`event`/`input` 字段。
* 每条事件使用 `Dataset_KG.extract_and_update` 进行抽取+写回。
* 每次写回后调用 `export_png` 生成图谱快照，便于目视检查。
* 可通过 `--reset-kg` 在导入前清空历史动态数据，仅保留固定节点。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Iterable
from pathlib import Path
import shutil
import getpass

# 可选：自动读取项目根部的 .env（不依赖 python-dotenv，避免新增包）
def _load_local_env(env_path: str) -> None:
    """读取简单的 KEY=VALUE 格式 .env，将未设置的键注入到 os.environ。

    规则：
    - 忽略以 # 开头的行与空行。
    - 仅解析首次出现的 KEY（避免覆盖已在系统环境中设置的变量）。
    - VALUE 首尾空白去除；保留原始大小写。
    - 不支持多行值与 export 语法，足够满足当前 NEO4J_* 配置。
    """
    try:
        if not os.path.isfile(env_path):
            return
        with open(env_path, 'r', encoding='utf-8') as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' not in line:
                    continue
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip()
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception as e:  # noqa: BLE001
        logging.getLogger("kg_replay").debug("忽略 .env 读取错误: %s", e)

# --- 解决相对运行时找不到顶层包的问题 ---
# 当以 "python scripts/replay_events_to_kg.py" 方式运行时, sys.path 只包含 scripts 目录,
# 导致同级的顶层包 data_provider 无法被发现。将项目根插入到 sys.path 前置位置。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from data_provider.data_loader import Dataset_KG
from models.triples_extraction import extract_triples
from exp.kg_service import KGServiceLocal


class OfflineKG:
    """离线占位 KG：不连接 Neo4j，只做三元组抽取与统计。

    提供与 Dataset_KG 最小兼容的接口：
    - extract_and_update(text) -> triples
    - graph_snapshot() -> {nodes_count, edges_count}
    - export_png(path) -> {error: str}
    - reset_graph(keep_fixed=True) -> None
    """

    def __init__(self) -> None:
        self._all: list[tuple[str, str, str]] = []

    def extract_and_update(self, text: str):
        triples = extract_triples(text)
        self._all.extend(triples)
        return triples

    def graph_snapshot(self):
        # 离线模式不维护节点/边，返回0占位
        return {"nodes_count": 0, "edges_count": 0}

    def export_png(self, path: str):
        return {"error": "offline mode: no graph"}

    def reset_graph(self, keep_fixed: bool = True):  # noqa: D401
        self._all.clear()

_LOG = logging.getLogger("kg_replay")


def _iter_event_texts(path: str) -> Iterable[str]:
    """读取 JSONL 文件，依次产出事件文本。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"事件文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            ln = raw.strip()
            if not ln:
                continue
            text: str | None = None
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict):
                    for key in ("text", "event", "input"):
                        val = obj.get(key)
                        if isinstance(val, str) and val.strip():
                            text = val.strip()
                            break
                elif isinstance(obj, str) and obj.strip():
                    text = obj.strip()
            except json.JSONDecodeError:
                text = ln

            if not text:
                _LOG.warning("[line %d] 未找到可用字段，跳过", line_no)
                continue

            yield text


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """解析命令行参数，支持：
    1. 自动加载项目根 `.env` （可关闭）
    2. NEO4J_AUTH=用户名/密码 快捷设置
    3. 默认优先使用端口 8687（常见 Docker 映射），失败后再尝试 7687
    """

    # 预加载 .env（允许通过 --no-load-env 关闭）
    # 为确保参数覆盖优先级：先加载 .env -> 再解析命令行 -> 命令行覆盖环境
    if "--no-load-env" not in argv:
        _load_local_env(str(PROJECT_ROOT / ".env"))

    # 兼容 NEO4J_AUTH=neo4j/secret
    auth_env = os.environ.get("NEO4J_AUTH")
    if auth_env and "/" in auth_env:
        env_user, env_pwd = auth_env.split("/", 1)
    else:
        env_user = os.environ.get("NEO4J_USER", "neo4j")
        env_pwd = os.environ.get("NEO4J_PASSWORD", "neo4j")

    # 默认 URI：优先环境变量；否则优先 8687 回退 7687
    default_uri = os.environ.get("NEO4J_URI") or "bolt://localhost:8687"

    parser = argparse.ArgumentParser(description="事件回放并更新知识图谱")
    parser.add_argument("--no-load-env", action="store_true", help="禁用自动加载项目根 .env")
    parser.add_argument(
        "--events-file",
        default=os.path.join("data_provider", "train_texts_conflict_aug.jsonl"),
        help="事件 JSONL 文件路径",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("results", "kg_replay"),
        help="图谱快照输出目录",
    )
    parser.add_argument(
        "--neo4j-uri",
        default=default_uri,
        help="Neo4j Bolt URI (默认优先使用 8687 -> .env -> 环境变量)",
    )
    parser.add_argument(
        "--neo4j-user",
        default=env_user,
        help="Neo4j 用户名 (可通过环境变量 NEO4J_USER 或 NEO4J_AUTH 覆盖)",
    )
    parser.add_argument(
        "--neo4j-password",
        default=env_pwd,
        help="Neo4j 密码 (可通过环境变量 NEO4J_PASSWORD 或 NEO4J_AUTH 覆盖)",
    )
    parser.add_argument(
        "--neo4j-database",
        default=os.environ.get("NEO4J_DATABASE", None),
        help="Neo4j 数据库 (可选)",
    )
    parser.add_argument("--reset-kg", action="store_true", help="导入前重置图谱，仅保留固定节点 (默认开启)")
    # 默认开启重置；如未来需要关闭，可再添加 --no-reset-kg 选项
    parser.set_defaults(reset_kg=True)
    parser.add_argument("--offline", action="store_true", help="离线模式：不连接 Neo4j，仅做抽取与简单统计")
    parser.add_argument("--auto-offline", action="store_true", help="当连接 Neo4J 失败时自动降级为离线模式")
    parser.add_argument("--prompt-password", action="store_true", help="运行时交互输入 Neo4j 密码（覆盖 --neo4j-password）")
    parser.add_argument("--connect-retries", type=int, default=3, help="连接 Neo4j 的重试次数（默认 3）")
    parser.add_argument("--connect-retry-interval", type=float, default=1.0, help="连接重试间隔秒数（默认 1.0）")
    parser.add_argument("--limit", type=int, default=50, help="最多处理的事件数量 (默认全部)")
    # --- 新增可视化调节参数 ---
    parser.add_argument("--png-max-edges", type=int, default=400, help="单张图最大边数抽样（缓解过密，默认 400，不等于限制事件数量）")
    parser.add_argument("--png-figsize", type=str, default=None, help="PNG 图尺寸，例如 16,10 / 16x10；默认 12x8")
    parser.add_argument("--png-dpi", type=int, default=200, help="PNG 分辨率 DPI，默认 200，可提高到 300 获得更清晰文字")
    parser.add_argument("--png-layout", type=str, default="spring", choices=["spring","kamada","circular","spectral","shell"], help="布局算法 (networkx)，默认 spring")
    parser.add_argument("--png-hide-edge-labels", action="store_true", help="隐藏边标签减轻遮挡 (仍不限制边/节点数量)")
    parser.add_argument("--png-node-font", type=int, default=8, help="节点字体大小")
    parser.add_argument("--png-edge-font", type=int, default=7, help="边字体大小")
    parser.add_argument("--snapshot-after", action="store_true", help="改为事件写入后再导出快照，并高亮本次新增边")
    args = parser.parse_args([a for a in argv if a != "--no-load-env"])  # 去除标志防止 .env 再次加载
    if args.prompt_password and not args.offline:
        args.neo4j_password = getpass.getpass("Neo4j 密码: ")
    return args


def main(argv: list[str]) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args(argv)


    # 准备输出目录：先清空再创建，确保每次运行得到干净的 PNG 序列
    if os.path.isdir(args.output_dir):
        removed = 0
        for name in os.listdir(args.output_dir):
            p = os.path.join(args.output_dir, name)
            try:
                if os.path.isfile(p) or os.path.islink(p):
                    os.remove(p)
                else:
                    shutil.rmtree(p)
                removed += 1
            except Exception as e:  # noqa: BLE001
                _LOG.warning("清理输出目录项失败: %s (%s)", p, e)
        _LOG.info("已清空输出目录 %s, 删除项目数=%d", args.output_dir, removed)
    os.makedirs(args.output_dir, exist_ok=True)
    _LOG.info("事件文件: %s", args.events_file)
    _LOG.info("快照输出(已清空): %s", args.output_dir)

    def _connect_dataset_kg() -> Dataset_KG:
        last_err: Exception | None = None
        for i in range(max(1, int(args.connect_retries))):
            try:
                _LOG.info("尝试连接 Neo4j (%s) [第%d次]", args.neo4j_uri, i + 1)
                return Dataset_KG(
                    root_path=os.getcwd(),
                    load_data=False,
                    neo4j_uri=args.neo4j_uri,
                    neo4j_user=args.neo4j_user,
                    neo4j_password=args.neo4j_password,
                    neo4j_database=args.neo4j_database,
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                if i < int(args.connect_retries) - 1:
                    time.sleep(max(0.0, float(args.connect_retry_interval)))
        # 端口自动回退：优先 8687 -> 7687 或 7687 -> 8687
        if last_err and isinstance(args.neo4j_uri, str) and "localhost" in args.neo4j_uri:
            alt_uri: str | None = None
            if ":8687" in args.neo4j_uri:
                alt_uri = args.neo4j_uri.replace(":8687", ":7687")
            elif ":7687" in args.neo4j_uri:
                alt_uri = args.neo4j_uri.replace(":7687", ":8687")
            if alt_uri:
                _LOG.warning("主端口连接失败，尝试回退端口: %s", alt_uri)
                try:
                    return Dataset_KG(
                        root_path=os.getcwd(),
                        load_data=False,
                        neo4j_uri=alt_uri,
                        neo4j_user=args.neo4j_user,
                        neo4j_password=args.neo4j_password,
                        neo4j_database=args.neo4j_database,
                    )
                except Exception as e2:  # noqa: BLE001
                    last_err = e2
        assert last_err is not None
        raise last_err

    if args.offline:
        _LOG.warning("使用离线模式：不会连接 Neo4j，PNG 导出将被跳过。")
        kg_raw = OfflineKG()
    else:
        try:
            kg_raw = _connect_dataset_kg()
        except Exception as e:  # noqa: BLE001
            err_msg = str(e)
            auth_hint = ""
            if "Unauthorized" in err_msg or "authentication" in err_msg.lower():
                auth_hint = (
                    "\n[认证失败排查] 1) 确认用户名/密码是否正确; 2) 若使用 Docker 官方镜像, 可通过设置环境变量 NEO4J_AUTH=neo4j/你的密码; "
                    "3) 如需临时跳过请使用 --offline 或 --auto-offline; 4) 可使用 --prompt-password 交互输入避免明文。"
                )
            if args.auto_offline:
                _LOG.error("连接 Neo4j 失败，将自动降级为离线模式：%s%s", err_msg, auth_hint)
                kg_raw = OfflineKG()
            else:
                _LOG.error("连接 Neo4j 失败。如需忽略，请加 --auto-offline 或使用 --offline。%s", auth_hint)
                raise

    # 用本地服务封装，贴近真实运行路径（带轻缓存与统一接口）
    kg = KGServiceLocal(kg_raw)

    if args.reset_kg:
        _LOG.info("重置图谱 (保留固定节点)...")
        # 透传到包装底层；KGServiceLocal 提供 reset_graph
        kg.reset_graph(keep_fixed=True)

    processed = 0
    t0 = time.time()
    for text in _iter_event_texts(args.events_file):
        processed += 1

        if not args.snapshot_after:
            # 导出写入前状态（临时禁用导出，仅记录快照数）
            snap_before = kg.graph_snapshot()
            _LOG.info(
                "[%03d] BEFORE nodes=%s edges=%s",
                processed,
                snap_before.get("nodes_count"),
                snap_before.get("edges_count"),
            )
            _LOG.info("跳过导出（已临时禁用 PNG/Cypher 导出）")

        # 抽取 + 写回
        triples = kg.extract_and_update(text)
        _LOG.info("[%03d] triples=%d", processed, len(triples))

        if args.snapshot_after:
            snap_after = kg.graph_snapshot()
            _LOG.info(
                "[%03d] AFTER nodes=%s edges=%s",
                processed,
                snap_after.get("nodes_count"),
                snap_after.get("edges_count"),
            )
            _LOG.info("跳过导出（已临时禁用 PNG/Cypher 导出）")

        if args.limit is not None and processed >= args.limit:
            break

    _LOG.info("处理结束，总计 %d 条事件，耗时 %.2fs", processed, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
