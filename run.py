#!/usr/bin/env python
"""
一键运行：支持多种模式
 pipeline: 原始CSV -> 流式抽取 -> 可选构建KG -> 导出产物
 train:    对 Qwen3-4B 进行 LoRA 训练（学习航保规则/冲突判定）
 judge:    单条事件判冲突（外挂KG+规则提示）
 stream-judge: 事件流判冲突（小批量流式推理）
"""

import os
import argparse
import random
import numpy as np
from typing import Optional, Tuple

from exp.exp_main import Exp_main
try:
    import torch
except ImportError:
    torch = None



def _try_connect_neo4j(
    uri: str, user: str, password: str, database: Optional[str]
) -> Tuple[bool, Optional[str], Optional[Exception]]:
    """尝试用多种 URI 方案连接 Neo4j，以处理加密/自签名证书等差异。

    返回: (是否成功, 实际使用的URI, 异常)
    """
    try:
        from neo4j import GraphDatabase, basic_auth
    except Exception as e:  # noqa: BLE001
        return False, None, e

    # 解析 host:port，确保优先尝试 bolt://
    from urllib.parse import urlsplit

    def normalize_uri(u: str) -> str:
        if u.startswith(
            (
                "bolt://",
                "neo4j://",
                "bolt+ssc://",
                "neo4j+ssc://",
                "bolt+s://",
                "neo4j+s://",
            )
        ):
            return u
        # 无 scheme 时默认 bolt
        return f"bolt://{u}"

    def host_port(u: str) -> Tuple[str, int]:
        parts = urlsplit(u)
        host = parts.hostname or "localhost"
        port = parts.port or 7687
        return host, port

    base_uri = normalize_uri(uri)
    host, port = host_port(base_uri)

    # 构造候选列表（bolt 优先，其次自签名变体，再到 neo4j 路由）
    variants = [
        f"bolt://{host}:{port}",
        f"bolt+ssc://{host}:{port}",
        f"neo4j://{host}:{port}",
        f"neo4j+ssc://{host}:{port}",
        f"bolt+s://{host}:{port}",
        f"neo4j+s://{host}:{port}",
    ]

    # 若用户传入的是不同主机或端口的 URI，也加入作为候选（避免被覆盖）
    if base_uri not in variants:
        variants.insert(0, base_uri)

    # 去重，保持顺序
    seen = set()
    uniq_candidates = []
    for c in variants:
        if c not in seen:
            uniq_candidates.append(c)
            seen.add(c)

    last_err: Optional[Exception] = None
    for cand in uniq_candidates:
        try:
            driver = GraphDatabase.driver(cand, auth=basic_auth(user, password))
            with driver.session(database=database) as s:
                s.run("RETURN 1 AS ok").single()
            return True, cand, None
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    return False, None, last_err


def _load_dotenv_to_environ(root: str) -> None:
    """从项目根目录读取 .env（若存在），将 NEO4J_* 变量注入 os.environ（不覆盖已有值）。"""
    dotenv_path = os.path.join(root, ".env")
    if not os.path.isfile(dotenv_path):
        return
    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k.upper().startswith("NEO4J_") and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        # 安静失败，不影响主流程
        pass


def main():
    fix_seed = 42
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    if torch is not None:
        torch.manual_seed(fix_seed)

    cwd = os.getcwd()
    default_root = cwd
    default_input_csv = os.path.join(
        default_root, "data_provider", "海天杯-ST_Job_训练集.csv"
    )
    default_out_dir = os.path.join(default_root, "data_provider")
    default_texts_jsonl = os.path.join(default_out_dir, "train_texts.jsonl")
    default_triples_jsonl = os.path.join(default_out_dir, "train_triples.jsonl")
    default_train_jsonl = os.path.join(default_out_dir, "train_for_model.jsonl")
    default_ttl_out = os.path.join(default_root, "output", "kg.ttl")
    default_png_out = os.path.join(default_root, "output", "kg.png")
    default_steps_dir = os.path.join(default_root, "output", "kg_steps")

    parser = argparse.ArgumentParser(description="Run haitianbei modes: pipeline/train/judge/stream-judge")
    parser.add_argument("--root", default=default_root, help="项目根路径")
    parser.add_argument(
        "--input_csv", default=default_input_csv, help="原始 CSV 文件路径"
    )
    parser.add_argument("--out_dir", default=default_out_dir, help="中间/输出目录")
    parser.add_argument(
        "--texts_jsonl", default=default_texts_jsonl, help="文本 JSONL 输出路径"
    )
    parser.add_argument(
        "--triples_jsonl", default=default_triples_jsonl, help="三元组 JSONL 输出路径"
    )
    parser.add_argument(
        "--train_jsonl", default=default_train_jsonl, help="训练 JSONL 输出路径"
    )
    parser.add_argument("--ttl_out", default=default_ttl_out, help="TTL 导出路径")
    parser.add_argument("--png_out", default=default_png_out, help="PNG 可视化导出路径")
    # 动态可视化（可选）
    parser.add_argument(
        "--visualize_every",
        type=int,
        default=1,
        help="构建 KG 时每 N 条记录导出一次快照（0 表示不导出，示例：10 表示每 10 条导出一次）",
    )
    parser.add_argument(
        "--visualize_dir", default=default_steps_dir, help="动态快照输出目录"
    )
    parser.add_argument(
        "--visualize_max_edges",
        type=int,
        default=300,
        help="动态快照中最多绘制边的数量，避免图片过密",
    )
    parser.add_argument(
        "--visualize_limit", type=int, default=50, help="最多导出多少张动态快照"
    )
    parser.add_argument(
        "--visualize_clean",
        action="store_true",
        default=True,
        help="在开始构建前清空快照目录中的旧 PNG 文件",
    )
    parser.add_argument(
        "--limit_kg", type=int, default=None, help="用于构建 KG 的最大记录数（可选）"
    )

    # 先加载 .env（若存在）
    _load_dotenv_to_environ(default_root)

    # Neo4j 连接参数（支持从 NEO4J_AUTH 解析）
    def _defaults_from_env():
        # 允许从 NEO4J_AUTH=neo4j/<password> 解析默认用户名与密码
        user_default = os.environ.get("NEO4J_USER", "neo4j")
        pwd_default = os.environ.get("NEO4J_PASSWORD", "neo4j")
        auth = os.environ.get("NEO4J_AUTH")
        if auth and "/" in auth:
            u, p = auth.split("/", 1)
            user_default = u or user_default
            pwd_default = p or pwd_default
        return user_default, pwd_default

    # 模式选择
    parser.add_argument("--mode", choices=["pipeline", "train", "judge", "stream-judge"], default="pipeline",
                        help="运行模式：pipeline/train/judge/stream-judge")

    # 模型相关
    parser.add_argument("--base_model_dir", default=os.path.join(default_root, "models", "Qwen3-4B"),
                        help="底座模型目录（Qwen3-4B）")
    parser.add_argument("--lora_out_dir", default=os.path.join(default_root, "results_entity_judge", "lora"),
                        help="LoRA 训练输出目录")
    parser.add_argument("--lora_adapter_dir", default=None, help="推理时加载的 LoRA 适配器目录")

    # 规则文档与训练数据
    parser.add_argument("--rules_md_path", default=None, help="规则文档（海天杯-技术资料.md）路径")

    # 训练超参
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="梯度累积步数（建议CPU先设为1以便看到进度）")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="训练时的最大序列长度（越小越快）")
    parser.add_argument("--log_steps", type=int, default=1, help="每多少个优化步打印一次训练日志（越小越详细）")
    parser.add_argument("--use_4bit", action="store_true", default=False, help="开启4bit量化（Windows 常不建议）")
    parser.add_argument("--no_fp16", action="store_true", default=False, help="关闭fp16，改用bf16/float32")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="训练设备优先级：auto/cuda/cpu")

    # 推理（judge/stream-judge）
    parser.add_argument("--event_text", default=None, help="单条事件文本（judge 模式）")
    parser.add_argument("--events_file", default=None, help="事件文本文件（每行一条，stream-judge 模式）")
    parser.add_argument("--focus_entities", default=None, help="逗号分隔的关注实体，例如: 飞机A001,停机位14")
    parser.add_argument("--no_vllm", action="store_true", default=False, help="禁用 vLLM，回退 transformers 推理")
    parser.add_argument("--batch_size", type=int, default=4, help="stream-judge 小批量大小")
    parser.add_argument("--simple_output", action="store_true", default=False, help="推理仅输出‘合规/冲突’，不带依据与建议")

    # 训练时动态拼接KG上下文（基于样本事件自动检索邻接）
    parser.add_argument("--no_augment_train_with_kg", action="store_true", default=False, help="禁用：为训练样本动态拼接KG上下文")

    _env_user, _env_pwd = _defaults_from_env()
    parser.add_argument(
        "--neo4j-uri",
        dest="neo4j_uri",
        default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j URI，例如 bolt://localhost:7687（也支持 neo4j://、bolt+ssc:// 等）",
    )
    parser.add_argument(
        "--neo4j-user",
        dest="neo4j_user",
        default=_env_user,
        help="Neo4j 用户名（默认可从 NEO4J_AUTH 推断）",
    )
    parser.add_argument(
        "--neo4j-password",
        dest="neo4j_password",
        default=_env_pwd,
        help="Neo4j 密码（默认可从 NEO4J_AUTH 推断）",
    )
    parser.add_argument(
        "--neo4j-database",
        dest="neo4j_database",
        default=os.environ.get("NEO4J_DATABASE", "neo4j"),
        help="Neo4j 数据库名称，默认 neo4j",
    )

    # 与 Exp_Basic 兼容的占位参数
    parser.add_argument("--use_gpu", action="store_true", help="是否使用 GPU（可选）")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_multi_gpu", action="store_true")
    parser.add_argument("--devices", type=str, default="0")

    # 允许离线模式：Neo4j 不可达时跳过图谱阶段，仍完成数据处理
    parser.add_argument(
        "--allow_offline",
        action="store_true",
        help="当 Neo4j 连接失败时，跳过 KG 阶段，仅运行前 3 步",
    )
    parser.add_argument(
        "--reset_kg",
        dest="reset_kg",
        action="store_true",
        default=True,
        help="在构建前清理 Neo4j 中的历史动态数据（保留固定节点），避免首张快照包含上一轮数据（默认开启）",
    )
    parser.add_argument(
        "--no-reset_kg",
        dest="reset_kg",
        action="store_false",
        help="禁用构建前的图谱重置，保留历史动态数据",
    )

    args = parser.parse_args()

    # 运行前进行 Neo4j 连接性检测（尝试多种 URI 方案）
    ok, used_uri, err = _try_connect_neo4j(
        args.neo4j_uri, args.neo4j_user, args.neo4j_password, args.neo4j_database
    )
    if ok:
        print(f"[Neo4j] 连接测试: PASS ({used_uri})")
    else:
        print("[Neo4j] 连接测试失败:", err)
        if args.mode == "pipeline":
            if args.allow_offline:
                print("[Neo4j] 启用离线模式：跳过图谱阶段，仅运行前 3 步。")
                setattr(args, "skip_kg", True)
            else:
                print("排查建议:")
                print("  1) 确认 Docker 容器状态为 Up/healthy，且端口映射包含 7687->7687 与 7474->7474")
                print("  2) 若启用了加密或自签名证书，请将 --neo4j-uri 设为 neo4j+ssc://localhost:7687 或 bolt+ssc://localhost:7687")
                print("  3) 若端口 7687 被错误映射到 7474（HTTP），会出现 handshake 错误，请重新创建容器并修正 -p 7687:7687")
                print("  4) 也可先使用 --allow_offline 运行前 3 步，稍后再接入数据库")
                return
        else:
            # 训练/推理模式下若 Neo4j 不可用，自动跳过 KG
            print("[Neo4j] 在当前模式下将自动跳过 KG（仍可进行训练/推理）")
            setattr(args, "skip_kg", True)

    # 统一放入供 Exp_main 读取
    setattr(args, "base_model_dir", args.base_model_dir)
    setattr(args, "lora_out_dir", args.lora_out_dir)

    exp = Exp_main(args)

    if args.mode == "pipeline":
        result = exp.run()
        print("\n=== Pipeline Finished ===")
        print("texts_jsonl :", result["texts_jsonl"])
        print("triples_jsonl:", result["triples_jsonl"])
        print("train_jsonl  :", result["train_jsonl"])
        print("ttl_out      :", result["ttl_out"])
        print("png_out      :", result["png_out"])
        print("kg_snapshot  :", result["kg_snapshot"])
    elif args.mode == "train":
        fp16 = (not args.no_fp16)
        info = exp.train_rules_lora(
            train_jsonl=args.train_jsonl,
            rules_md_path=args.rules_md_path,
            output_dir=args.lora_out_dir,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            use_4bit=args.use_4bit,
            fp16=fp16,
            augment_train_with_kg=(not args.no_augment_train_with_kg),
            gradient_accumulation_steps=max(1, int(args.grad_accum_steps)),
            max_seq_len=max(128, int(args.max_seq_len)),
            prefer_device=args.device,
            log_steps=max(1, int(args.log_steps)),
        )
        print("\n=== Train Finished ===")
        print("adapter_dir :", info.get("adapter_dir"))
        print("samples     :", info.get("samples"))
    elif args.mode == "judge":
        if not args.event_text:
            print("[judge] 需要 --event_text")
            return
        focus = [s.strip() for s in (args.focus_entities or "").split(",") if s.strip()]
        res = exp.judge_conflict(
            event_text=args.event_text,
            focus_entities=(focus or None),
            rules_md_path=args.rules_md_path,
            lora_adapter_dir=args.lora_adapter_dir,
            use_vllm=(not args.no_vllm),
            simple_output=bool(args.simple_output),
        )
        print("\n=== Judge Result ===")
        print(res["output"])
    elif args.mode == "stream-judge":
        if not args.events_file or not os.path.isfile(args.events_file):
            print("[stream-judge] 需要 --events_file (每行一条事件)")
            return
        with open(args.events_file, "r", encoding="utf-8") as f:
            events = [ln.strip() for ln in f if ln.strip()]
        focus = [s.strip() for s in (args.focus_entities or "").split(",") if s.strip()]
        print("\n=== Stream Judge Start ===")
        count = 0
        for ev, out in exp.stream_judge_conflicts(
            events_iter=events,
            focus_entities=(focus or None),
            rules_md_path=args.rules_md_path,
            lora_adapter_dir=args.lora_adapter_dir,
            use_vllm=(not args.no_vllm),
            batch_size=max(1, int(args.batch_size)),
            simple_output=bool(args.simple_output),
        ):
            count += 1
            print(f"[#{count}] 事件: {ev}")
            print(out)
            print("-" * 60)
        print("=== Stream Judge End ===")


if __name__ == "__main__":
    main()
