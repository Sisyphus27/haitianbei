#!/usr/bin/env python
"""
简洁入口：所有模式统一从 Exp_main.run() 进入
保留功能：
 - train:         对 Qwen3-4B 进行 LoRA 训练
 - stream-judge:  事件流判冲突（读取文本 -> 抽取三元组 -> 构建/更新KG -> LLM 判冲突）
"""

import os
import logging
import argparse
import random
import numpy as np
from typing import Optional

from exp.exp_main import Exp_main
try:
    import torch
except ImportError:
    torch = None


def main():
    fix_seed = 42
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    if torch is not None:
        torch.manual_seed(fix_seed)

    cwd = os.getcwd()
    default_root = cwd

    parser = argparse.ArgumentParser(description="haitianbei: train / stream-judge / marl-train")
    parser.add_argument("--root", default=default_root, help="项目根路径")
    # 可选：训练数据（仅 train 用）
    parser.add_argument("--train_jsonl", default=None, help="SFT 训练数据(JSONL) 可选")

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
    parser.add_argument("--mode", choices=["train", "stream-judge", "marl-train"], default="stream-judge",
                        help="运行模式：train / stream-judge / marl-train")

    # 日志等级
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default=os.environ.get("HTB_LOG_LEVEL", "INFO"),
                        help="日志等级，默认 INFO，可用环境变量 HTB_LOG_LEVEL 覆盖")

    # 模型相关
    parser.add_argument("--base_model_dir", default=os.path.join(default_root, "models", "Qwen3-4B"),
                        help="底座模型目录（Qwen3-4B）")
    parser.add_argument("--lora_out_dir", default=os.path.join(default_root, "results_entity_judge", "lora"),
                        help="LoRA 训练输出目录")
    # 可选：分别为 judge 与 decomposer 指定独立的输出目录（若不指定，将使用 --lora_out_dir 并在内部做默认分目录隔离）
    parser.add_argument("--judge_lora_out_dir", default=None,
                        help="(可选) 主判定LoRA的输出目录，覆盖 --lora_out_dir 的默认值")
    parser.add_argument("--decomp_lora_out_dir", default=None,
                        help="(可选) 分解器LoRA的输出目录，覆盖 --lora_out_dir 的默认值")
    parser.add_argument("--lora_adapter_dir", default=None, help="推理时加载的 LoRA 适配器目录")
    # 小LLM 分解器相关（推理/训练均可用）
    parser.add_argument("--enable_decomposer", action="store_true", default=False, help="启用：先用小LLM进行问题分解")
    parser.add_argument("--decomp_lora_adapter_dir", default=None, help="分解器 LoRA 适配器目录（可选）")
    parser.add_argument(
        "--decomp_base_model_dir",
        default=os.path.join(default_root, "models", "Qwen3-4B"),
        help="分解器基座模型目录（默认 Qwen3-0.6B）",
    )

    # 规则文档与训练数据
    parser.add_argument("--rules_md_path", default=None, help="规则文档（海天杯-技术资料.md）路径")

    # 训练超参
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="训练时的最大序列长度")
    parser.add_argument("--log_steps", type=int, default=1, help="每多少个优化步打印一次训练日志（越小越详细）")
    parser.add_argument("--use_4bit", action="store_true", default=False, help="开启4bit量化（Windows 常不建议）")
    parser.add_argument("--no_fp16", action="store_true", default=False, help="关闭fp16，改用bf16/float32")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="训练设备优先级：auto/cuda/cpu")

    # 训练后端：hf（Transformers+PEFT）或 llama-factory
    parser.add_argument("--train_backend", choices=["hf", "llama-factory"], default="hf",
                        help="选择训练后端：hf（内置 Transformers+PEFT）或 llama-factory（外部训练器）")
    parser.add_argument("--train_task", choices=["judge", "decompose"], default="judge",
                        help="训练任务：judge（规则学习）或 decompose（问题分解）")
    parser.add_argument("--decomp_events_file", default=None, help="用于分解器训练的事件文本文件（每行一条）")

    # 推理（judge/stream-judge）
    parser.add_argument("--events_file", default=None, help="事件文本文件（每行一条，stream-judge 模式）")
    parser.add_argument("--focus_entities", default=None, help="逗号分隔的关注实体，例如: 飞机A001,停机位14")
    parser.add_argument("--no_vllm", action="store_true", default=False, help="禁用 vLLM，回退 transformers 推理")
    parser.add_argument("--batch_size", type=int, default=4, help="stream-judge 小批量大小")
    parser.add_argument("--simple_output", action="store_true", default=False, help="推理仅输出‘合规/冲突’，不带依据与建议")
    parser.add_argument("--print_decomposition", action="store_true", default=False, help="在判定前打印分解器输出(JSON)")

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
    # KG 控制
    parser.add_argument("--skip_kg", action="store_true", default=False, help="推理时跳过连接/使用KG，仅离线文本模式")
    parser.add_argument("--reset_kg", action="store_true", default=True,
                        help="运行前重置KG（保留固定节点），清理历史动态关系，避免残留占用/分配影响判定")

    # 与 Exp_Basic 兼容的占位参数（最小化保留）
    parser.add_argument("--use_gpu", action="store_true", help="是否使用 GPU（可选）")
    parser.add_argument("--gpu", type=int, default=0)

    # ================= MARL 训练相关参数（复用 htb_environment） =================
    parser.add_argument("--marl_use_task1_kg", action="store_true", default=False,
                        help="使用任务一的 Dataset_KG 形成先验闭环（需可用的 Neo4j 连接）")
    parser.add_argument("--marl_n_agents", type=int, default=8, help="智能体数量（飞机数）")
    parser.add_argument("--marl_result_dir", default=os.path.join(default_root, "htb_environment", "result"),
                        help="MARL 结果输出目录（info.json / plan.json / 图表）")
    parser.add_argument("--marl_result_name", default="exp", help="结果子目录名")
    parser.add_argument("--marl_n_epoch", type=int, default=5)
    parser.add_argument("--marl_n_episodes", type=int, default=5)
    parser.add_argument("--marl_train_steps", type=int, default=2)
    parser.add_argument("--marl_evaluate_cycle", type=int, default=5)
    parser.add_argument("--marl_evaluate_epoch", type=int, default=20)
    parser.add_argument("--marl_batch_size", type=int, default=32)
    parser.add_argument("--marl_buffer_size", type=int, default=1000)
    parser.add_argument("--marl_target_update_cycle", type=int, default=200)
    parser.add_argument("--marl_save_cycle", type=int, default=50)
    parser.add_argument("--marl_lr", type=float, default=5e-4)
    parser.add_argument("--marl_cuda", action="store_true", default=True, help="启用CUDA（若可用）")
    parser.add_argument("--marl_use_prior", action="store_true", default=True, help="在观测/状态中拼接先验特征")
    parser.add_argument("--marl_prior_dim_site", type=int, default=8)
    parser.add_argument("--marl_prior_dim_plane", type=int, default=3)
    parser.add_argument("--marl_obs_pad", type=int, default=32, help="先验拼接后尾部补零到该维度")
    parser.add_argument("--marl_no_export_csv", action="store_true", default=False, help="评估时不导出CSV")
    parser.add_argument("--marl_eval_only", action="store_true", default=False, help="仅评估已训练模型（不训练）")

    args = parser.parse_args()
    # 配置日志
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    # 统一入口：直接交给 Exp_main.run()
    exp = Exp_main(args)
    exp.run()


if __name__ == "__main__":
    main()
