"""
Author: zy
Date: 2025-10-22 17:35:46
LastEditTime: 2025-10-23 19:49:44
LastEditors: zy
Description:
FilePath: haitianbei/exp/exp_main.py

"""

# 兼容直接"运行本文件"的调试方式：确保项目根目录在 sys.path
# 这样即使用 "python exp/exp_main.py" 启动，也能导入顶层包（exp、utils、data_provider）。
import os as _os
import sys as _sys

if __package__ is None or __package__ == "":
    _sys.path.insert(
        0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
    )

import os
import json
import shutil
import logging as _logging
from typing import Any, Optional
from typing import Dict, List, Tuple
import time as _time

from exp.exp_basic import Exp_Basic
from exp.kg_service import KGServiceLocal
from models.triples_extraction import extract_triples as _extract_triples
from data_provider.data_loader import Dataset_KG
from data_provider.data_loader import (
    load_instruction_jsonl,
    build_rules_sft_samples_from_md,
)
from data_provider.data_loader import load_events_from_file  # type: ignore

# 可选依赖：transformers / peft / datasets / bitsandbytes
try:
    import transformers  # type: ignore
except Exception:  # pragma: no cover
    transformers = None  # type: ignore
try:
    import peft  # type: ignore
except Exception:  # pragma: no cover
    peft = None  # type: ignore
try:
    import datasets as _datasets  # type: ignore
except Exception:  # pragma: no cover
    _datasets = None  # type: ignore
try:
    import bitsandbytes as _bnb  # type: ignore
except Exception:  # pragma: no cover
    _bnb = None  # type: ignore


class _MemDS:
    """极简内存数据集，兼容 Trainer 的 __len__/__getitem__ 接口。"""

    def __init__(self, arr: List[dict]):
        self.arr = arr

    def __len__(self) -> int:  # pragma: no cover - 简单容器
        return len(self.arr)

    def __getitem__(self, i: int) -> dict:  # pragma: no cover
        return self.arr[i]


class Exp_main(Exp_Basic):
    def __init__(self, args):
        # 在父类构造前开启延迟模型初始化，避免父类调用 _build_model 访问未就绪属性
        self._defer_model_init = True
        # 初始化一些成员，防止父类访问异常
        self._judge_model = None
        self._judge_tokenizer = None
        self._judge_device = None
        self._decomp_model = None
        self._decomp_tokenizer = None
        self._decomp_device = None
        self._kg_initialized = False

        # 提前设置 root，避免父类 __init__ 或早期调用 _ensure_kg_service 时访问缺失属性
        try:
            self.root = getattr(args, "root", os.getcwd())
        except Exception:
            self.root = os.getcwd()

        # 提前初始化 Neo4j 连接相关属性，防止父类或其他早期调用访问缺失导致 AttributeError
        # 使用 getattr 兜底 None，保持与后续逻辑一致
        try:
            self.neo4j_uri = getattr(args, "neo4j_uri", None)
            self.neo4j_user = getattr(args, "neo4j_user", None)
            self.neo4j_password = getattr(args, "neo4j_password", None)
            self.neo4j_database = getattr(args, "neo4j_database", None)
        except Exception:
            self.neo4j_uri = None
            self.neo4j_user = None
            self.neo4j_password = None
            self.neo4j_database = None
        # 提前初始化 skip_kg / reset_kg 避免 _ensure_kg_service 使用时缺失
        try:
            self.skip_kg = getattr(args, "skip_kg", False)
            self.reset_kg = bool(getattr(args, "reset_kg", False))
        except Exception:
            self.skip_kg = False
            self.reset_kg = False

        # 优先使用 CUDA：若检测到可用 GPU 且用户未强制指定 CPU，则自动切换到 CUDA，并补齐所需参数
        torch_mod = None
        try:
            import torch as _torch_mod  # type: ignore

            torch_mod = _torch_mod
            cuda_available = _torch_mod.cuda.is_available()
        except Exception:
            cuda_available = False

        # 确保必需的属性存在，避免父类 _acquire_device 访问失败
        if not hasattr(args, "use_gpu"):
            setattr(args, "use_gpu", False)
        if not hasattr(args, "gpu"):
            setattr(args, "gpu", 0)
        if not hasattr(args, "use_multi_gpu"):
            setattr(args, "use_multi_gpu", False)
        if not hasattr(args, "devices"):
            setattr(args, "devices", str(getattr(args, "gpu", 0)))

        if cuda_available:
            device_arg = str(getattr(args, "device", "auto")).lower()
            if device_arg in ("auto", "cuda", "gpu", ""):
                setattr(args, "device", "cuda")
            setattr(args, "use_gpu", True)
            if getattr(args, "use_multi_gpu", None) is None:
                setattr(args, "use_multi_gpu", False)
            if getattr(args, "use_multi_gpu", False):
                devices_val = getattr(args, "devices", None)
                if not devices_val:
                    try:
                        visible = torch_mod.cuda.device_count() if torch_mod else 1
                        setattr(args, "devices", ",".join(str(i) for i in range(visible)))
                    except Exception:
                        setattr(args, "devices", str(getattr(args, "gpu", 0)))
        else:
            # 无 CUDA 时确保 device 参数为 cpu
            if str(getattr(args, "device", "auto")).lower() == "auto":
                setattr(args, "device", "cpu")
            setattr(args, "use_gpu", False)
            setattr(args, "use_multi_gpu", False)

        super(Exp_main, self).__init__(args)
        # 父类完成基础属性初始化后，允许 _build_model 进入真实加载流程
        self._defer_model_init = False
        # 根路径已在 super 调用前设置，这里不再重复，但若用户在运行期修改 args.root，可同步刷新
        try:
            if hasattr(args, "root") and args.root and args.root != self.root:
                self.root = args.root
        except Exception:
            pass
        # Neo4j 连接参数
        self.neo4j_uri = getattr(args, "neo4j_uri", None)
        self.neo4j_user = getattr(args, "neo4j_user", None)
        self.neo4j_password = getattr(args, "neo4j_password", None)
        self.neo4j_database = getattr(args, "neo4j_database", None)
        # 离线模式：跳过 KG 构建
        self.skip_kg = getattr(args, "skip_kg", False)
        # 是否在构建前重置 KG（保留固定节点）（流推理中通常不需要）
        self.reset_kg = bool(getattr(args, "reset_kg", False))
        # 确保 base_model_dir/decomp_base_model_dir 在 _build_model 调用前可用
        raw_base_model_dir = getattr(args, "base_model_dir", None)
        if isinstance(raw_base_model_dir, str) and raw_base_model_dir:
            base_model_dir = raw_base_model_dir
        else:
            base_model_dir = os.path.join(self.root, "models", "Qwen3-4B")
            _logging.info(
                f"[MODEL] base_model_dir 未传入，使用默认目录: {base_model_dir}"
            )
        setattr(args, "base_model_dir", base_model_dir)
        self.base_model_dir = base_model_dir
        decomp_arg = getattr(args, "decomp_base_model_dir", None)
        decomp_dir = decomp_arg if isinstance(decomp_arg, str) and decomp_arg else base_model_dir
        setattr(args, "decomp_base_model_dir", decomp_dir)
        self.decomp_base_model_dir = decomp_dir
        # LoRA 输出目录
        self.lora_out_dir = getattr(
            args,
            "lora_out_dir",
            os.path.join(self.root, "results_entity_judge", "lora"),
        )
        # 小LLM（问题分解）相关配置：优先使用 args.decomp_base_model_dir，否则回退到主模型目录
        self.decomp_base_model_dir = getattr(args, "decomp_base_model_dir", None) or self.base_model_dir
        self.generate_timeout_sec = float(getattr(args, "generate_timeout_sec", 120.0))
        # 提升 CPU 场景的生成长度上限，避免输出被截断导致 JSON 不完整
        self.cpu_max_new_tokens = int(getattr(args, "cpu_max_new_tokens", 256))
        # 重试/降级策略配置
        self.retry_max_new_tokens = int(getattr(args, "retry_max_new_tokens", 1024))
        self.no_simple_fallback = bool(getattr(args, "no_simple_fallback", False))

        try:
            if not os.path.isdir(self.decomp_base_model_dir):
                _logging.warning(
                    f"[DECOMP] 模型目录不存在，回退到 base_model_dir: {self.decomp_base_model_dir} -> {self.base_model_dir}"
                )
                self.decomp_base_model_dir = self.base_model_dir
        except Exception:
            pass

        # KG 可视化输出目录与计数器
        self.kg_vis_dir = os.path.join(self.root, "results", "kg_vis")
        self.kg_vis_idx = 0
        # 输出保存目录（区分大小模型）
        self.results_out_dir = os.path.join(self.root, "results", "model_outputs")
        self.judge_out_dir = os.path.join(self.results_out_dir, "judge")
        self.decomp_out_dir = os.path.join(self.results_out_dir, "decomp")
        try:
            os.makedirs(self.judge_out_dir, exist_ok=True)
            os.makedirs(self.decomp_out_dir, exist_ok=True)
        except Exception:
            pass
        # 流式会话级文件
        self._session_ts = None
        self._judge_out_file = None
        self._decomp_out_file = None
        # 统一：仅通过 KGServiceLocal 封装访问 KG；保留 self.kg 引用以兼容旧代码，但不直接使用。
        self.kg_service = None
        self.kg = None  # 兼容旧逻辑；后续请使用 self.kg_service

        self._ensure_kg_service()

    def _ensure_kg_service(self) -> None:
        """惰性初始化 KG 服务，确保在需要时可用。"""
        if getattr(self, "_kg_initialized", False):
            return

        self._kg_initialized = True
        if getattr(self, "skip_kg", False):
            return

        try:
            _raw_kg = Dataset_KG(
                getattr(self, "root", os.getcwd()),
                load_data=False,
                neo4j_uri=self.neo4j_uri,
                neo4j_user=self.neo4j_user,
                neo4j_password=self.neo4j_password,
                neo4j_database=self.neo4j_database,
            )
            try:
                self.kg_service = KGServiceLocal(_raw_kg)
                self.kg = self.kg_service.kg
            except Exception:
                self.kg_service = None

            if self.reset_kg and self.kg_service is not None:
                try:
                    self.kg_service.reset_graph(keep_fixed=True)
                    _logging.info(
                        "[KG] (init) reset_graph 已执行，已清理历史动态关系，仅保留固定节点。"
                    )
                except Exception:
                    pass

                try:
                    if os.path.isdir(self.kg_vis_dir):
                        for _f in os.listdir(self.kg_vis_dir):
                            _fp = os.path.join(self.kg_vis_dir, _f)
                            if os.path.isfile(_fp):
                                os.remove(_fp)
                except Exception:
                    pass

            try:
                os.makedirs(self.kg_vis_dir, exist_ok=True)
            except Exception:
                pass

            try:
                _snap = self.kg_service.graph_snapshot() if self.kg_service else {}
                _logging.info(
                    f"[KG] (init) snapshot nodes={_snap.get('nodes_count')} edges={_snap.get('edges_count')}"
                )
            except Exception:
                pass
        except Exception as _e:
            _logging.info(f"[KG] 初始化失败，进入跳过模式：{_e}")
            self.kg_service = None

    def _build_model(self):
        """根据当前模式加载判决/分解模型。"""
        if getattr(self, "_defer_model_init", False):
            return {"judge": None, "decomp": None}, []

        self._ensure_kg_service()

        if getattr(self, "_models_ready", False):
            return self._model_handles, []

        if transformers is None:
            raise RuntimeError("需要安装 transformers 才能加载推理模型")

        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("需要安装 torch 才能加载推理模型") from exc

        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        def _resolve_device(prefer: str) -> str:
            prefer = (prefer or "auto").lower()
            cuda_ok = torch.cuda.is_available()
            if prefer == "cpu":
                return "cpu"
            if prefer == "cuda":
                return "cuda" if cuda_ok else "cpu"
            return "cuda" if cuda_ok else "cpu"

        prefer_device = str(getattr(self.args, "device", "auto"))
        target_device = _resolve_device(prefer_device)

        # 统一模型路径来源：完全依赖 __init__ 中从 args 注入的 base_model_dir / decomp_base_model_dir
        base_model_dir = self.base_model_dir

        # 校验 Transformers 模型目录（避免误传 GGUF 目录导致报错）
        def _is_transformers_dir(path: str) -> bool:
            try:
                if not path or not os.path.isdir(path):
                    return False
                # GGUF 目录通常含 .gguf 文件或名称包含 GGUF
                if "GGUF" in os.path.basename(path).upper():
                    return False
                cfg_path = os.path.join(path, "config.json")
                if not os.path.isfile(cfg_path):
                    return False
                import json as _json

                with open(cfg_path, "r", encoding="utf-8") as _f:
                    cfg = _json.load(_f)
                return isinstance(cfg, dict) and bool(cfg.get("model_type"))
            except Exception:
                return False

        if not _is_transformers_dir(base_model_dir):
            # 自动回退到仓库内可能的 Transformers 检查点
            candidates = [
                os.path.join(self.root, "models", "Qwen3-4B"),
                os.path.join(self.root, "models", "Qwen2_5-3B"),
            ]
            fallback = None
            for c in candidates:
                if _is_transformers_dir(c):
                    fallback = c
                    break
            if fallback is None:
                raise RuntimeError(
                    f"[MODEL] 非法的 Transformers 模型目录: {base_model_dir}。请提供包含 config.json 且含 model_type 的目录（非 GGUF）。"
                )
            _logging.warning(
                f"[MODEL] 检测到非 Transformers 目录或缺少 config: {base_model_dir}，自动回退到: {fallback}"
            )
            base_model_dir = fallback
            self.base_model_dir = fallback
            try:
                setattr(self.args, "base_model_dir", fallback)
            except Exception:
                pass

        decomp_model_dir = self.decomp_base_model_dir or base_model_dir
        if not _is_transformers_dir(decomp_model_dir):
            _logging.warning(
                f"[MODEL] decomp_base_model_dir 无效或为 GGUF: {decomp_model_dir}，回退到主模型目录"
            )
            decomp_model_dir = base_model_dir
        self.decomp_base_model_dir = decomp_model_dir

        judge_lora_dir = getattr(self.args, "lora_adapter_dir", None)
        decomp_lora_dir = getattr(self.args, "decomp_lora_adapter_dir", None)

        shared_cache: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {}

        def _load_handle(model_dir: str, lora_dir: Optional[str], tag: str) -> Dict[str, Any]:
            if not model_dir or not os.path.isdir(model_dir):
                raise RuntimeError(f"[MODEL] {tag} 模型目录不存在: {model_dir}")

            key = (model_dir, lora_dir or None)
            if key in shared_cache:
                return shared_cache[key]

            _logging.info(f"[MODEL] 加载 {tag} 基座: {model_dir}")
            tok = AutoTokenizer.from_pretrained(
                model_dir, trust_remote_code=True, use_fast=False
            )
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

            load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
            use_half = False
            quantized = False
            quant_tag = "none"
            # 配置 judge 量化：auto/none/int8/int4；decomp 维持默认（CUDA优先fp16）
            judge_quant = str(getattr(self.args, "judge_quant", "auto")).lower()
            if tag == "judge":
                if judge_quant in ("int8", "auto"):
                    if _bnb is not None:
                        try:
                            from transformers import BitsAndBytesConfig  # type: ignore

                            if judge_quant == "int8" or judge_quant == "auto":
                                bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
                                load_kwargs["quantization_config"] = bnb_cfg
                                quantized = True
                                quant_tag = "int8"
                                _logging.info("[MODEL] judge 使用 8bit 量化 (bitsandbytes)")
                        except Exception as _qe:
                            _logging.warning(f"[MODEL] 8bit 量化初始化失败，将按非量化路径: {_qe}")
                    elif judge_quant == "int8":
                        _logging.warning("[MODEL] 请求 int8 但未安装 bitsandbytes，忽略量化请求")

                if (not quantized) and judge_quant == "int4":
                    if _bnb is not None:
                        try:
                            from transformers import BitsAndBytesConfig  # type: ignore

                            # 常用 nf4 配置；计算精度使用 float16 以兼顾显存与稳定性
                            bnb_cfg = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.float16,
                            )
                            load_kwargs["quantization_config"] = bnb_cfg
                            quantized = True
                            quant_tag = "int4"
                            _logging.info("[MODEL] judge 使用 4bit 量化 (nf4)")
                        except Exception as _qe:
                            _logging.warning(f"[MODEL] 4bit 量化初始化失败，将按非量化路径: {_qe}")
                    else:
                        _logging.warning("[MODEL] 请求 int4 但未安装 bitsandbytes，忽略量化请求")

                # judge_quant==none：显式不量化，走下方 half 路径

            # 非量化（或量化失败）时，如为 CUDA 则优先使用 float16
            if (not quantized) and isinstance(target_device, str) and target_device.startswith("cuda"):
                try:
                    if torch.cuda.is_available():
                        load_kwargs["torch_dtype"] = torch.float16
                        use_half = True
                except Exception:
                    pass

            model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)

            if lora_dir:
                if not os.path.isdir(lora_dir):
                    _logging.warning(f"[MODEL] LoRA 目录不存在({tag}): {lora_dir}")
                else:
                    if peft is None:
                        raise RuntimeError("需要安装 peft 才能加载 LoRA 适配器")
                    from peft import PeftModel  # type: ignore

                    _logging.info(f"[MODEL] 应用 {tag} LoRA: {lora_dir}")
                    model = PeftModel.from_pretrained(model, lora_dir)

            model.eval()
            # 如果已经量化（int8/int4），不再强制 dtype 转换；否则按原逻辑将模型移到目标设备
            if not quantized:
                try:
                    if use_half:
                        try:
                            model = model.to(device=target_device, dtype=torch.float16)  # type: ignore[arg-type]
                        except Exception:
                            model = model.to(target_device)  # type: ignore[call-arg]
                            try:
                                model = model.to(torch.float16)  # type: ignore[arg-type]
                            except Exception:
                                pass
                    else:
                        model = model.to(target_device)  # type: ignore[call-arg]
                except Exception:
                    _logging.warning(
                        f"[MODEL] 无法将 {tag} 模型移动到 {target_device}，保留当前设备"
                    )

            try:
                model.requires_grad_(False)
            except Exception:
                try:
                    for _param in model.parameters():
                        _param.requires_grad = False
                except Exception:
                    pass

            try:
                model_device = next(model.parameters()).device
            except Exception:
                model_device = torch.device(target_device)

            # 运行时摘要日志：明确实际采用的量化/精度与设备
            try:
                try:
                    _dtype = str(next(model.parameters()).dtype)
                except Exception:
                    _dtype = "unknown"
                _quant = quant_tag if tag == "judge" else "none (decomp)"
                _logging.info(
                    f"[MODEL] {tag} ready: quant={_quant} dtype={_dtype} device={model_device}"
                )
            except Exception:
                pass

            handle = {
                "tokenizer": tok,
                "model": model,
                "device": model_device,
                "model_dir": model_dir,
                "lora_dir": lora_dir,
            }
            shared_cache[key] = handle
            return handle

        handles: Dict[str, Dict[str, Any]] = {}
        handles["judge"] = _load_handle(base_model_dir, judge_lora_dir, "judge")

        # 仅在明确需要时加载分解器：启用标志或要求打印分解结果
        _need_decomp = bool(
            getattr(self.args, "enable_decomposer", False)
            or getattr(self.args, "print_decomposition", False)
        )
        if _need_decomp:
            try:
                handles["decomp"] = _load_handle(
                    decomp_model_dir, decomp_lora_dir, "decomp"
                )
            except Exception as exc:  # noqa: BLE001
                _logging.warning(f"[MODEL] 分解模型加载失败，跳过：{exc}")
                handles["decomp"] = {
                    "tokenizer": None,
                    "model": None,
                    "device": None,
                    "model_dir": decomp_model_dir,
                    "lora_dir": decomp_lora_dir,
                }
        else:
            _logging.info(
                "[MODEL] 跳过 decomp 加载：未启用 enable_decomposer/print_decomposition"
            )
            handles["decomp"] = {
                "tokenizer": None,
                "model": None,
                "device": None,
                "model_dir": decomp_model_dir,
                "lora_dir": decomp_lora_dir,
            }

        self._model_handles = handles
        self._models_ready = True

        judge_handle = handles.get("judge", {})
        decomp_handle = handles.get("decomp", {})
        self._judge_tokenizer = judge_handle.get("tokenizer")
        self._judge_model = judge_handle.get("model")
        self._judge_device = judge_handle.get("device")
        self._decomp_tokenizer = decomp_handle.get("tokenizer")
        self._decomp_model = decomp_handle.get("model")
        self._decomp_device = decomp_handle.get("device")

        self.model = handles
        self.parameters1 = []

        # 设备可见性日志，便于确认是否在 CUDA 上运行
        try:
            _logging.info(
                f"[MODEL] devices -> judge: {self._judge_device} | decomp: {self._decomp_device}"
            )
        except Exception:
            pass

        return handles, []

    def run(self):
        """统一入口：根据 mode 执行 LoRA/MARL 训练或事件流冲突判定。"""
        import logging as _logging

        mode = getattr(self.args, "mode", "stream-judge")

        # ============ 新增：MARL 训练入口（复用 htb_environment，无需修改其源码） ============
        if mode == "marl-train":
            _logging.info("[MARL] 开始训练/评估 (mode=marl-train)")
            default_event_jsonl = os.path.join(
                self.root, "data_provider", "train_texts_conflict_aug.jsonl"
            )
            event_jsonl = getattr(self.args, "marl_event_jsonl", None) or default_event_jsonl
            if not os.path.isfile(event_jsonl):
                _logging.warning(
                    f"[MARL] 指定的事件JSONL文件不存在: {event_jsonl}"
                )
            info = self.train_marl(
                use_task1_kg=bool(getattr(self.args, "marl_use_task1_kg", False)),
                n_agents=int(getattr(self.args, "marl_n_agents", 8)),
                result_dir=str(
                    getattr(
                        self.args,
                        "marl_result_dir",
                        os.path.join(self.root, "htb_environment", "result"),
                    )
                ),
                result_name=str(getattr(self.args, "marl_result_name", "exp")),
                n_epoch=int(getattr(self.args, "marl_n_epoch", 5)),
                n_episodes=int(getattr(self.args, "marl_n_episodes", 5)),
                train_steps=int(getattr(self.args, "marl_train_steps", 2)),
                evaluate_cycle=int(getattr(self.args, "marl_evaluate_cycle", 5)),
                evaluate_epoch=int(getattr(self.args, "marl_evaluate_epoch", 20)),
                batch_size=int(getattr(self.args, "marl_batch_size", 32)),
                buffer_size=int(getattr(self.args, "marl_buffer_size", 1000)),
                target_update_cycle=int(
                    getattr(self.args, "marl_target_update_cycle", 200)
                ),
                save_cycle=int(getattr(self.args, "marl_save_cycle", 50)),
                lr=float(getattr(self.args, "marl_lr", 5e-4)),
                cuda=bool(getattr(self.args, "marl_cuda", True)),
                use_prior=bool(getattr(self.args, "marl_use_prior", True)),
                prior_dim_site=int(getattr(self.args, "marl_prior_dim_site", 8)),
                prior_dim_plane=int(getattr(self.args, "marl_prior_dim_plane", 3)),
                obs_pad=int(getattr(self.args, "marl_obs_pad", 32)),
                export_csv=not bool(getattr(self.args, "marl_no_export_csv", False)),
                eval_only=bool(getattr(self.args, "marl_eval_only", False)),
                event_jsonl=event_jsonl,
            )
            _logging.info("=== MARL Train Finished ===")
            _logging.info(f"result_dir  : {info.get('result_dir')}")
            _logging.info(f"result_name : {info.get('result_name')}")
            _logging.info(f"use_task1_kg: {info.get('use_task1_kg')}")
            _logging.info(f"event_jsonl : {info.get('event_jsonl')}")
            return info

        if mode == "train":
            # 读取训练超参
            fp16 = not bool(getattr(self.args, "no_fp16", False))
            train_backend = str(
                getattr(self.args, "train_backend", "hf") or "hf"
            ).lower()
            train_task = str(
                getattr(self.args, "train_task", "judge") or "judge"
            ).lower()
            _logging.info(f"[TRAIN] backend={train_backend} task={train_task}")
            # 如选择 llama-factory，用其执行 LoRA 训练
            if train_backend in ("llama-factory", "llamafactory", "lf"):
                if train_task in ("decompose", "decomp"):
                    # 优先使用专用输出目录；否则回退到通用目录
                    _out = getattr(self.args, "decomp_lora_out_dir", None) or getattr(
                        self.args, "lora_out_dir", self.lora_out_dir
                    )
                    info = self.train_decomposer_with_llamafactory(
                        # 数据来源：优先 decomp_events_file；否则从 train_jsonl 推断
                        decomp_events_file=getattr(
                            self.args, "decomp_events_file", None
                        ),
                        train_jsonl=getattr(self.args, "train_jsonl", None),
                        rules_md_path=getattr(self.args, "rules_md_path", None),
                        output_dir=_out,
                        num_train_epochs=int(getattr(self.args, "num_train_epochs", 1)),
                        learning_rate=float(getattr(self.args, "learning_rate", 2e-5)),
                        per_device_train_batch_size=int(
                            getattr(self.args, "per_device_train_batch_size", 1)
                        ),
                        gradient_accumulation_steps=int(
                            getattr(self.args, "grad_accum_steps", 1)
                        ),
                        max_seq_len=int(getattr(self.args, "max_seq_len", 1024)),
                        fp16=fp16,
                    )
                else:
                    _out = getattr(self.args, "judge_lora_out_dir", None) or getattr(
                        self.args, "lora_out_dir", self.lora_out_dir
                    )
                    info = self.train_rules_lora_with_llamafactory(
                        train_jsonl=getattr(self.args, "train_jsonl", None),
                        rules_md_path=getattr(self.args, "rules_md_path", None),
                        output_dir=_out,
                        num_train_epochs=int(getattr(self.args, "num_train_epochs", 1)),
                        learning_rate=float(getattr(self.args, "learning_rate", 2e-5)),
                        per_device_train_batch_size=int(
                            getattr(self.args, "per_device_train_batch_size", 1)
                        ),
                        gradient_accumulation_steps=int(
                            getattr(self.args, "grad_accum_steps", 1)
                        ),
                        max_seq_len=int(getattr(self.args, "max_seq_len", 1024)),
                        fp16=fp16,
                    )
                _logging.info("=== Train Finished (LLaMA-Factory) ===")
                _logging.info(f"adapter_dir : {info.get('adapter_dir')}")
                _logging.info(f"samples     : {info.get('samples')}")
                return info
            # Transformers+PEFT 路径
            _out = getattr(self.args, "judge_lora_out_dir", None) or getattr(
                self.args, "lora_out_dir", self.lora_out_dir
            )
            info = self.train_rules_lora(
                train_jsonl=getattr(self.args, "train_jsonl", None),
                rules_md_path=getattr(self.args, "rules_md_path", None),
                output_dir=_out,
                num_train_epochs=int(getattr(self.args, "num_train_epochs", 1)),
                learning_rate=float(getattr(self.args, "learning_rate", 2e-5)),
                per_device_train_batch_size=int(
                    getattr(self.args, "per_device_train_batch_size", 1)
                ),
                use_4bit=bool(getattr(self.args, "use_4bit", False)),
                fp16=fp16,
                augment_train_with_kg=not bool(
                    getattr(self.args, "no_augment_train_with_kg", False)
                ),
                gradient_accumulation_steps=int(
                    getattr(self.args, "grad_accum_steps", 1)
                ),
                max_seq_len=int(getattr(self.args, "max_seq_len", 1024)),
                prefer_device=str(getattr(self.args, "device", "auto")),
                log_steps=int(getattr(self.args, "log_steps", 1)),
            )
            _logging.info("=== Train Finished ===")
            _logging.info(f"adapter_dir : {info.get('adapter_dir')}")
            _logging.info(f"samples     : {info.get('samples')}")
            return info

        # 默认：stream-judge
        if mode != "stream-judge":
            return  # 已在上面返回
        try:
            model_handles, _ = self._build_model()
        except Exception as exc:  # noqa: BLE001
            _logging.error(f"[STREAM] 模型加载失败: {exc}")
            return {"error": "model_init_failed", "message": str(exc)}

        if not isinstance(model_handles, dict):
            _logging.error("[STREAM] 模型句柄返回异常类型")
            return {"error": "invalid_model_handles"}

        judge_entry = model_handles.get("judge")
        if not isinstance(judge_entry, dict) or judge_entry.get("model") is None:
            _logging.error("[STREAM] 判定模型未正确初始化")
            return {"error": "missing_judge_model"}

        events_file = getattr(self.args, "events_file", None)
        # 若未显式提供，尝试回退到 tests/events_sample.txt
        if not events_file or not os.path.isfile(events_file):
            _default_ev = os.path.join(self.root, "tests", "events_sample.txt")
            if os.path.isfile(_default_ev):
                _logging.warning(
                    f"[STREAM] 未提供 --events_file，使用默认样例: {_default_ev}"
                )
                events_file = _default_ev
            else:
                _logging.error(
                    "[STREAM] 缺失 --events_file。示例: python run.py --mode stream-judge --events_file tests/events_sample.txt --rules_md_path tests/rules_sample.txt"
                )
                return {"error": "missing_events_file"}
        # 统一通过 loader 读取事件：支持 .txt（每行一条）与 .jsonl（优先取 text/event/input 字段）
        try:
            events = load_events_from_file(events_file)[:5]
        except Exception:
            # 兜底：按纯文本逐行
            with open(events_file, "r", encoding="utf-8") as f:
                events = [ln.strip() for ln in f if ln.strip()]
        rules_md_path = getattr(self.args, "rules_md_path", None)
        batch_size = max(1, int(getattr(self.args, "batch_size", 4)))
        simple_output = bool(getattr(self.args, "simple_output", False))

        # 预热（可通过 --no_warmup 关闭）
        if not bool(getattr(self.args, "no_warmup", False)):
            try:
                _order = str(getattr(self.args, "warmup_order", "judge_first"))
                _decomp_skip = bool(getattr(self.args, "no_warmup_decomp", False))
                _timeout = int(getattr(self.args, "warmup_timeout_sec", 300))
                self.warmup_models(
                    judge=True,
                    decomp=not _decomp_skip,
                    order=_order,
                    timeout_sec=_timeout,
                )
            except Exception:
                pass

        _logging.info("=== Stream Judge Start ===")
        # 为当前会话准备输出文件（区分大小模型）
        try:
            import time as __t

            self._session_ts = __t.strftime("%Y%m%d_%H%M%S", __t.localtime())
            self._judge_out_file = os.path.join(
                self.judge_out_dir, f"stream_{self._session_ts}.jsonl"
            )
            self._decomp_out_file = os.path.join(
                self.decomp_out_dir, f"stream_{self._session_ts}.jsonl"
            )
            _logging.info(f"[SAVE] judge-> {self._judge_out_file}")
            _logging.info(f"[SAVE] decomp-> {self._decomp_out_file}")
        except Exception:
            pass
        count = 0
        for ev, out in self.stream_judge_conflicts(
            events_iter=events,
            rules_md_path=rules_md_path,
            batch_size=batch_size,
            simple_output=simple_output,
            show_decomposition=bool(getattr(self.args, "print_decomposition", False)),
        ):
            count += 1
            _logging.info(f"[#{count}] 事件: {ev}")
            # 不输出模型内容，改为提示保存路径
            try:
                _logging.info("[SAVE] 模型输出已写入对应 JSONL 文件（judge/decomp）")
            except Exception:
                pass
            _logging.info("-" * 60)
        _logging.info("=== Stream Judge End ===")
        # 导出任务一格式（若指定路径）
        try:
            _export_path = getattr(self.args, "export_task1_json", None)
            if _export_path:
                self.export_task1_results(out_path=_export_path)
        except Exception as _e:
            try:
                _logging.warning(f"[EXPORT-TASK1] 导出失败: {_e}")
            except Exception:
                pass
        return {"count": count}

    # ----------------------------
    # 规则学习与冲突判断（训练/推理）
    # ----------------------------
    def build_rules_prompt(self, rules_md_path: str) -> str:
        """读取技术资料 Markdown，构造基础规则提示词。

        返回一个可作为系统/前置提示的长文本，用于指导大模型基于规则进行判断。
        """
        if not os.path.isfile(rules_md_path):
            return ""
        with open(rules_md_path, "r", encoding="utf-8") as f:
            md = f.read()
        # 粗略清洗：去图片/多余空白
        import re

        md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)
        md = "\n".join(
            [ln.rstrip() for ln in md.splitlines() if ln.strip() != ""]
        )  # 去空白行
        # 前缀提示：
        prefix = (
            "你是一名航保作业助手。你的主要任务是根据知识图谱和文本信息来整理航保作业状态"
            # "当输入一段新事件文本时，需要结合当前知识图谱状态，回答是否与现有状态或规则冲突，并给出依据。\n\n"
            "【规则文档】\n"
        )
        return prefix + md

    def _kg_text_context(
        self,
        kg: Dataset_KG,
        focus_entities: Optional[List[str]] = None,
        max_edges: int = 200,
    ) -> str:
        """从 Neo4j 组织一段可读的上下文文本，供大模型检索。

        - 若给定 focus_entities，则优先输出这些实体的一阶邻居；否则输出全局少量边。
        """
        lines: List[str] = []
        try:
            if focus_entities:
                for ent in focus_entities:
                    nb = kg.neighbors(ent)
                    if nb.get("out"):
                        for s, p, o in nb["out"][: max_edges // 2]:
                            lines.append(f"{s} -[{p}]-> {o}")
                    if nb.get("in"):
                        for s, p, o in nb["in"][: max_edges // 2]:
                            lines.append(f"{s} -[{p}]-> {o}")
            else:
                # 回退：扫描部分关系（通过导出 PNG 逻辑里的 allow_rels）
                snap = kg.graph_snapshot()
                lines.append(
                    f"[SNAPSHOT] nodes={snap.get('nodes_count',0)} edges={snap.get('edges_count',0)}"
                )
        except Exception as _e:
            pass
        ctx = "\n".join(lines[:max_edges])
        if not ctx:
            ctx = "(当前图为空或仅有固定节点)"
        return "【KG状态】\n" + ctx

    def _format_conflict_prompt(
        self, event_text: str, rules_text: str, kg_text: str
    ) -> str:
        """将规则文本、KG上下文与事件文本格式化为单轮对话提示。"""
        instruction = (
            "任务：判断以下事件是否与当前状态或规则冲突。\n"
            "输出格式：先给出结论（合规/冲突），再给出1-3条依据，最后给出可操作建议。\n"
        )
        # 将指令置顶，其次是规则、KG与事件，避免模型先复述上下文
        parts = [instruction, rules_text, kg_text, "【事件】\n" + event_text]
        return "\n\n".join([p for p in parts if p])
    
    def _format_conflict_prompt_with_mode(
    self,
    event_text: str,
    rules_text: str,
    kg_text: str,
    *,
    simple: bool = False,
    conflict_judge: bool = True,
        ) -> str:
        """根据模式构造提示。

        参数:
        - simple: 仅在 conflict_judge=True 时可用；输出“合规”或“冲突”二字之一。
        - conflict_judge=True: 抽取时空信息 + 判定冲突；需输出 判定/reason/suggest。
        - conflict_judge=False: 用于任务一时空解析输出，不做冲突判断，按“时空信息/历史统计/态势预测”句式输出。
        """
        if not conflict_judge:
            # === 任务一：仅做时空/历史/态势解析，输出标准 JSON（支持多条） ===
            instruction = (
                "你是一个严格的时空信息解析器。\n"
                "【任务】依据提供的事件文本与知识图谱，抽取该时刻的作业与状态，并以固定 JSON 结构输出。\n\n"
                "【输出要求（非常重要）】\n"
                "1. 只能输出一个 JSON 对象，且必须位于 JSON_START 与 JSON_END 两行之间。\n"
                "2. 禁止在 JSON 外输出任何文字、解释、分析、空行或 Markdown 代码块（例如 ```）。\n"
                "3. 所有键名必须使用半角双引号，JSON 可被标准解析器解析。\n\n"
                "【字段与多条规则】\n"
                "1) \"time\"：归一化时间，格式 \"YYYY-MM-DD HH:MM:SS\"；无法确定时填空字符串 \"\"。\n"
                "2) \"时空信息 01..NN\"：按出现顺序编号（两位补零，名称与编号之间有空格）。每条为一个完整中文句子，结尾以全角分号 \"；\" 收尾。\n"
                "   - 句式示例A：\"飞机 A001，正在开展作业，着陆作业，推理耗时 0.1秒，对应能力项（飞机ID，作业关系）；\"\n"
                "   - 句式示例B：\"飞机 A001，处于，移动中（速度 15.2米/秒），推理耗时 0.2秒，对应能力项（飞机ID，飞机状态、状态关系）；\"\n"
                "   - 如存在速度、位置、牵引车/停机位等要素，请自然融入句中（速度以括号形式，例如（速度 5米/秒））。\n"
                "3) \"历史信息 01..NN\"（尽量）：两位补零编号的中文句子，用于作业历史统计，如：\n"
                "   - \"飞机 A001 牵引作业 已作业 1分钟，推理耗时 0.02秒，对应能力项（作业时长）；\"\n"
                "   - 亦可用同义键名 \"历史统计 01..NN\"（推荐统一输出为 历史信息）。\n"
                "4) \"态势感知 01..NN\"（尽量）：两位补零编号的中文句子，用于当前/后续态势，如：\n"
                "   - \"牵引至 14号停机位 剩余作业时长 预计 9分钟完成，推理耗时 0.03秒，对应能力项（预计剩余作业时长）；\"\n"
                "   - 亦可用同义键名 \"态势预测 01..NN\"（推荐统一输出为 态势感知）。\n\n"
                "【严格约束】\n"
                "- 必须至少包含键 \"time\" 与 \"时空信息 01\"。若无法抽取内容，\"时空信息 01\" 的值使用空字符串。\n"
                "- 每一条中文句子均以全角分号结束，且包含 \"对应能力项（...）\" 子句，能力项从：飞机ID、作业关系、飞机状态、状态关系、位置关系、停机位ID 中择有即列。\n"
                "- 严禁出现 \"判定\"、\"reason\"、\"suggest\" 等与冲突判定相关的键。\n"
                "- 不要输出示例、说明文字或任何与 JSON 无关的内容。\n\n"
                "【最终输出格式（仅以下三行，注意不要加其他内容）】\n"
                "JSON_START\n"
                "{...}\n"
                "JSON_END"
            )
            # 抽取模式不拼接规则文本，减少干扰；只给 KG + 事件
            parts = [
                instruction,
                kg_text.strip() if kg_text else "",
                "【事件】\n" + event_text.strip(),
            ]
            return "\n\n".join([p for p in parts if p])

        # === 以下是冲突判定模式，结构可以保持不变，仅略微强化“只输出 JSON”约束 ===
        if simple:
            instruction = (
                "任务：判断以下事件是否与当前状态或规则冲突。\n"
                "仅输出下列两者之一（不得包含任何额外文字/标点/代码块）：\n"
                "合规\n"
                "冲突\n"
            )
        else:
            instruction = (
                "任务：解析事件中的所有时空作业信息，并基于规则+KG判定是否冲突。\n"
                "只输出一个 JSON 对象，且必须放在 JSON_START 与 JSON_END 之间；不得出现任何额外文字/Markdown。\n"
                "键与顺序（必须按照此顺序排列）：\n"
                "1) time: 归一化时间 YYYY-MM-DD HH:MM:SS；若事件含多个时刻，可填首个关键时刻；缺失可填空字符串。\n"
                "2) 时空信息01..NN: 按出现顺序编号的对象，每个对象尽可能包含以下键（缺失可省略）：\n"
                "   - 时间、飞机ID、停机位ID、作业ID、飞机状态、停机位状态、作业持续时间、预测剩余作业时间\n"
                "3) 判定: 仅 '合规' 或 '冲突'。\n"
                "4) reason: 冲突/合规的精炼依据，1-3 条短句组成的数组；无则 [].\n"
                "5) suggest: 操作性建议；若合规填 '无'。\n"
                "严格要求：\n"
                "- 时空信息01..NN 必须全部在前；判定/reason/suggest 放在最后。\n"
                "- 必须至少包含 '时空信息01' 键；若文本无法抽取任何要素，'时空信息01' 的值使用空对象 {}。\n"
                "- 不得输出除上述以外的键；不得输出示例、解释或代码围栏以外文本。\n"
                "- 时间需归一化为 YYYY-MM-DD HH:MM:SS；无法确定具体日期可采用同一日期占位。\n"
                "输出格式（仅以下三行）：\n"
                "JSON_START\n"
                "{...}\n"
                "JSON_END"
            )
        parts = [instruction, rules_text, kg_text, "【事件】\n" + event_text]
        return "\n\n".join([p for p in parts if p])

    def _format_conflict_prompt_with_mode1(
        self,
        event_text: str,
        rules_text: str,
        kg_text: str,
        *,
        simple: bool = False,
        conflict_judge: bool = True,
    ) -> str:
        """根据模式构造提示。

        参数:
        - simple: 仅在 conflict_judge=True 时可用；输出“合规”或“冲突”二字之一。
        - conflict_judge=True: 抽取时空信息 + 判定冲突；需输出 判定/reason/suggest。
        - conflict_judge=False: 仅抽取时空信息（不拼接规则文本，不输出 判定/reason/suggest）。
        """
        if not conflict_judge:
            # 抽取模式（无冲突判断）——与任务示例格式一致，但省略判定/reason/suggest。
            instruction = (
                "任务：解析事件中的所有时空作业信息。\n"
                "只输出一个 JSON，对象须位于 JSON_START 与 JSON_END 之间；不得出现任何额外文字/Markdown。\n"
                "键与顺序（必须按照此顺序排列）：\n"
                "1) time: 归一化时间 YYYY-MM-DD HH:MM:SS；若事件含多个时刻，可填首个关键时刻；缺失填空字符串。\n"
                "2) 时空信息01..NN: 按出现顺序编号的对象，每个对象尽可能包含以下键（缺失可省略）：\n"
                "   - 时间、飞机ID、停机位ID、作业ID、飞机状态、停机位状态、作业持续时间、预测剩余作业时间\n"
                "严格要求：\n"
                "- 必须至少包含 '时空信息01' 键；若无法抽取任何要素，值使用空对象 {}。\n"
                "- 不得出现判定/reason/suggest 等键。\n"
                "- 不得输出示例解释或多余文本。\n"
                "示例（仅用于理解，不要照抄）：\n"
                "JSON_START\n"
                "{\n"
                "  \"time\": \"2025-07-01 08:00:00\",\n"
                "  \"时空信息01\": {\"时间\": \"2025-07-01 08:00:00\", \"飞机ID\": \"A001\"},\n"
                "  \"时空信息02\": {\"时间\": \"2025-07-01 08:05:00\", \"停机位ID\": \"08\"}\n"
                "}\n"
                "JSON_END\n"
                "输出格式（仅以下三行）：\n"
                "JSON_START\n"
                "{...}\n"
                "JSON_END"
            )
            # 抽取模式不拼接规则文本，减少干扰
            parts = [instruction, kg_text, "【事件】\n" + event_text]
            return "\n\n".join([p for p in parts if p])

        # 以下为冲突判定模式
        if simple:
            instruction = (
                "任务：判断以下事件是否与当前状态或规则冲突。\n"
                "仅输出下列两者之一（不得包含任何额外文字/标点/代码块）：\n"
                "合规\n"
                "冲突\n"
            )
        else:
            instruction = (
                "任务：解析事件中的所有时空作业信息，并基于规则+KG判定是否冲突。\n"
                "只输出一个 JSON 对象，且必须放在 JSON_START 与 JSON_END 之间；不得出现任何额外文字/Markdown。\n"
                "键与顺序（必须按照此顺序排列）：\n"
                "1) time: 归一化时间 YYYY-MM-DD HH:MM:SS；若事件含多个时刻，可填首个关键时刻；缺失可填空字符串。\n"
                "2) 时空信息01..NN: 按出现顺序编号的对象，每个对象尽可能包含以下键（缺失可省略）：\n"
                "   - 时间、飞机ID、停机位ID、作业ID、飞机状态、停机位状态、作业持续时间、预测剩余作业时间\n"
                "3) 判定: 仅 '合规' 或 '冲突'。\n"
                "4) reason: 冲突/合规的精炼依据，1-3 条短句组成的数组；无则 [].\n"
                "5) suggest: 操作性建议；若合规填 '无'。\n"
                "严格要求：\n"
                "- 时空信息01..NN 必须全部在前；判定/reason/suggest 放在最后。\n"
                "- 必须至少包含 '时空信息01' 键；若文本无法抽取任何要素，'时空信息01' 的值使用空对象 {}，不要省略该键。\n"
                "- 不得输出除上述以外的键；不得输出示例、解释或代码围栏以外文本。\n"
                "- 时间需归一化为 YYYY-MM-DD HH:MM:SS；无法确定具体日期可采用同一日期占位。\n"
                "示例（仅用于理解，不要照抄内容）：\n"
                "JSON_START\n"
                "{\n"
                "  \"time\": \"2025-07-01 08:00:00\",\n"
                "  \"时空信息01\": {\n"
                "    \"时间\": \"2025-07-01 08:00:00\", \"飞机ID\": \"A001\", \"停机位ID\": \"08\"\n"
                "  },\n"
                "  \"时空信息02\": {\n"
                "    \"时间\": \"2025-07-01 08:02:00\", \"飞机状态\": \"到位\"\n"
                "  },\n"
                "  \"判定\": \"冲突\",\n"
                "  \"reason\": [\"停机位08已占用\"],\n"
                "  \"suggest\": \"改派空位\"\n"
                "}\n"
                "JSON_END\n"
                "输出格式（仅以下三行）：\n"
                "JSON_START\n"
                "{...}\n"
                "JSON_END"
            )
        parts = [instruction, rules_text, kg_text, "【事件】\n" + event_text]
        return "\n\n".join([p for p in parts if p])

    def _format_decompose_prompt(
        self, event_text: str, rules_text: str, kg_text: str
    ) -> str:
        """为小LLM构造问题分解提示，要求输出 JSON 结构，聚焦三类冲突。

        目标输出示例（说明用，不要抄写内容）：
        JSON_START
        {"entities": ["飞机A001", "跑道Z"], "applicable_rules": ["……"], "potential_conflicts": [], "evidence": [], "notes": "none"}
        JSON_END

        说明：小模型参数量较小，移除 KG 文本上下文，仅保留规则文档与事件文本，以降低提示长度并减少幻觉概率。
        """
        # 强化版分解指令：将指令置顶，严格禁止代码块与额外文本，加入 JSON_START/JSON_END 约束
        instruction = (
            "你是一名严格的时空作业分解器。\n"
            "只做结构化分解，不做最终合规/冲突判定。\n"
            "输出要求（必须全部满足）：\n"
            "- 仅输出一个 JSON 对象，且必须位于 JSON_START 与 JSON_END 两行之间。\n"
            "- 严禁在 JSON 外输出任意文字、提示、说明、前后缀或空行。\n"
            "- 严禁使用任何 Markdown 代码块（例如 ``` 或 ```json）。出现即视为错误输出。\n"
            "- 所有键名与字符串值均使用英文双引号。\n"
            "JSON 字段与约束：\n"
            "{\n"
            '  "entities": [str,...],            // 事件中出现的关键实体（飞机/跑道/停机位/牵引车）；按出现顺序去重；最多10个\n'
            '  "applicable_rules": [str,...],    // 可能相关的规则要点，涵盖互斥/占用/许可/依赖/状态一致性/转运约束/设备范围；每条≤40字；最多8条\n'
            '  "potential_conflicts": [str,...], // 基于 KG 状态推测的潜在冲突（如：跑道Z 已被占用）；最多5条；若无则 []\n'
            '  "evidence": [str,...],            // (可缺省) 支撑片段：KG 三元组或节点状态，如 "飞机A001 -[占用]-> 跑道Z"；可空\n'
            '  "notes": ""                  // 当 potential_conflicts 为空填 "none"，否则填空字符串\n'
            "}\n"
            "操作步骤提示（不要输出这些步骤本身）：\n"
            "1. 抽取实体：仅使用事件文本中出现的名词，不合并不臆造。\n"
            "2. 匹配规则：提炼短句，不粘贴整段原文，不生成训练外条目。\n"
            "3. 列举潜在冲突：仅在 KG 提示存在占用/互斥/依赖未满足/等待未达/范围不足/时序违法时填写；否则置空。\n"
            "最终仅按如下三行输出：\n"
            "JSON_START\n{...}\nJSON_END"
        )
        # 将指令置顶，减少模型在阅读上下文后先行复述的可能
        # 仅保留 instruction + rules_text + 事件文本；不再拼接 kg_text
        parts = [instruction, rules_text, "【事件】\n" + event_text]
        return "\n\n".join([p for p in parts if p])

    def train_rules_lora(
        self,
        train_jsonl: Optional[str] = None,
        rules_md_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        num_train_epochs: int = 1,
        learning_rate: float = 2e-5,
        per_device_train_batch_size: int = 1,
        use_4bit: bool = True,
        fp16: bool = True,
        augment_train_with_kg: bool = True,
        gradient_accumulation_steps: int = 8,
        max_seq_len: int = 4096,
        prefer_device: str = "auto",
        log_steps: int = 1,
    ) -> dict:
        """使用 Transformers+PEFT(Q-LoRA) 对 Qwen3-4B 进行指令微调。

        说明：vLLM 专注推理，不提供训练能力。这里采用 PEFT 进行 LoRA 微调，产出的适配器可在 vLLM 推理时加载。

        训练数据格式（JSONL 每行一个样本）：
        {"instruction": str, "input": str, "output": str}

        - 若未提供 train_jsonl，将基于 rules_md_path 生成一个极简自监督样本集（总结/提炼任务），
          仅用于跑通流程；建议后续提供高质量标注数据以提升冲突判定能力。
        """
        if transformers is None or peft is None:
            raise RuntimeError(
                "需要安装 transformers 与 peft 方可训练。请先 pip install transformers peft accelerate datasets bitsandbytes"
            )

        from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
        from transformers import Trainer, TrainingArguments, TrainerCallback  # type: ignore
        from peft import LoraConfig, get_peft_model, TaskType  # type: ignore
        import torch as _torch  # type: ignore

        model_dir = self.base_model_dir
        out_dir = output_dir or self.lora_out_dir
        # 若未显式区分目录，默认将判定任务输出到 lora/judge，避免与分解器互相覆盖
        try:
            if (output_dir is None) or (
                os.path.abspath(output_dir) == os.path.abspath(self.lora_out_dir)
            ):
                out_dir = os.path.join(self.lora_out_dir, "judge")
        except Exception:
            pass
        os.makedirs(out_dir, exist_ok=True)

        # 构造训练数据集（模块化迁移到 data_loader）
        samples = []
        if train_jsonl and os.path.isfile(train_jsonl):
            samples = load_instruction_jsonl(train_jsonl)
            # 兼容性兜底：若给定的 JSONL 不是 instruction/input/output 结构（例如项目中用于抽取的 train_for_model.jsonl），
            # 则回退到基于规则文档自动构造一批占位样本，避免出现空数据集导致的 num_samples=0 错误。
            if len(samples) == 0 and rules_md_path:
                _logging.info(
                    f"[TRAIN] 检测到 {os.path.basename(train_jsonl)} 非指令微调格式(或为空)，改用规则文档生成占位样本。"
                )
                samples = build_rules_sft_samples_from_md(
                    rules_md_path, max_samples=50, chunk_chars=2000
                )
        if len(samples) == 0:
            if not rules_md_path:
                raise RuntimeError(
                    "未提供 instruction/input/output 训练数据，且未提供规则文档用于自动构造样本。请传入 --rules_md_path 或提供符合SFT格式的 JSONL。"
                )
            samples = build_rules_sft_samples_from_md(
                rules_md_path, max_samples=50, chunk_chars=2000
            )
        _logging.info(f"[TRAIN] 使用训练样本数: {len(samples)}")

        # 可选：动态拼接 KG 上下文（基于样本中的事件文本自动检索实体邻接）
        if augment_train_with_kg:
            kg_service = self.kg_service  # 统一使用已初始化的服务
            if kg_service is None:
                _logging.info("[TRAIN] 未初始化 kg_service，跳过训练样本的KG上下文拼接。")

            def _extract_event_text(inp: str) -> str:
                if not isinstance(inp, str):
                    return ""
                import re

                m = re.search(r"【事件】\n(.+)$", inp, flags=re.S)
                if m:
                    return m.group(1).strip()
                return inp.strip()

            def _auto_focus_entities(ev: str) -> List[str]:
                ents: List[str] = []
                try:
                    trips = _extract_triples(ev)
                    for s, p, o in trips:
                        for t in (s, o):
                            t = str(t)
                            if (
                                t.startswith("飞机")
                                or t.startswith("停机位")
                                or t.startswith("跑道")
                            ):
                                ents.append(t)
                except Exception:
                    pass
                # 去重
                seen = set()
                out: List[str] = []
                for x in ents:
                    if x not in seen:
                        out.append(x)
                        seen.add(x)
                return out[:8]

            if kg_service is not None:
                rules_text = (
                    self.build_rules_prompt(rules_md_path) if rules_md_path else ""
                )
                aug: List[dict] = []
                for ex in samples:
                    inp0 = ex.get("input", "")
                    ev = _extract_event_text(inp0)
                    focus = _auto_focus_entities(ev)
                    try:
                        kg_text = kg_service.get_context_text(
                            focus_entities=(focus or None), limit=200
                        )
                    except Exception:
                        kg_text = "【KG状态】\n(离线模式，服务异常)"
                    new_input = "\n\n".join(
                        [
                            p
                            for p in (
                                rules_text,
                                kg_text,
                                ("【事件】\n" + ev if ev else inp0),
                                "输出格式：结论+依据+建议",
                            )
                            if p
                        ]
                    )
                    ex2 = dict(ex)
                    ex2["input"] = new_input
                    aug.append(ex2)
                samples = aug

        # 简单的数据集包装：拼接成 prompt -> target 的监督微调
        def build_text(ex: dict) -> Tuple[str, str]:
            ins, inp, out = (
                ex.get("instruction", ""),
                ex.get("input", ""),
                ex.get("output", ""),
            )
            prompt = f"指令：{ins}\n输入：{inp}\n回答："
            return prompt, out

        # tokenizer / model（开启4bit量化以节省显存）
        tok = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True, use_fast=False
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        quant_config = None
        if use_4bit and _bnb is not None:
            from transformers import BitsAndBytesConfig  # type: ignore

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="float16" if fp16 else "bfloat16",
            )

        # 设备选择
        _cuda_ok = hasattr(_torch, "cuda") and _torch.cuda.is_available()
        _device_map = "auto"
        if prefer_device == "cpu":
            _device_map = None
        elif prefer_device == "cuda" and _cuda_ok:
            _device_map = None

        # 缓解显存碎片：尽量启用可扩展分段分配
        try:
            import os as __os

            if _cuda_ok:
                __os.environ.setdefault(
                    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
                )
                # 新变量名（旧名已弃用），提前兼容，减少告警
                __os.environ.setdefault(
                    "PYTORCH_ALLOC_CONF", "expandable_segments:True"
                )
        except Exception:
            pass

        # 指定权重量化/精度，进一步降低显存
        _torch_dtype = None
        try:
            if fp16:
                _torch_dtype = _torch.float16
            else:
                _torch_dtype = _torch.bfloat16
        except Exception:
            _torch_dtype = None

        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            device_map=_device_map,
            quantization_config=quant_config,
            dtype=_torch_dtype,
        )

        # 动态获取 LoraConfig 以规避静态类型检查在不同版本上的签名差异
        try:
            import peft as __peft  # type: ignore
            _LoraCfgKls = getattr(__peft, "LoraConfig", None)
        except Exception:
            _LoraCfgKls = None
        if _LoraCfgKls is None:
            raise RuntimeError("需要安装 peft 方可构建 LoRA 配置")
        lora_cfg = _LoraCfgKls(  # type: ignore[call-arg]
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=None,  # 让 PEFT 自动选择常见投影层；必要时可指定
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        # 训练时关闭缓存节省显存
        try:
            _cfg = getattr(model, "config", None)
            if _cfg is not None and hasattr(_cfg, "use_cache"):
                setattr(_cfg, "use_cache", False)
        except Exception:
            pass
        # 将模型放到期望设备
        if prefer_device == "cuda" and _cuda_ok:
            model.to("cuda")
        elif prefer_device == "cpu":
            model.to("cpu")
        model.print_trainable_parameters()

        # 打印设备与显存信息
        try:
            if _cuda_ok:
                gname = _torch.cuda.get_device_name(0)
                mem_alloc = _torch.cuda.memory_allocated(0) / (1024**3)
                mem_resv = _torch.cuda.memory_reserved(0) / (1024**3)
                _logging.info(
                    f"[TRAIN] 使用GPU: {gname} | alloc={mem_alloc:.2f}GB reserved={mem_resv:.2f}GB"
                )
            else:
                _logging.info("[TRAIN] 使用CPU训练（建议在有GPU时设置 --device cuda）")
        except Exception:
            pass

        # 将样本 tokenize
        def _tok_map(ex):
            prompt, target = build_text(ex)
            full = prompt + target
            enc = tok(full, padding=False, truncation=True, max_length=int(max_seq_len))
            # labels：与 input_ids 相同，训练时计算全损失
            enc["labels"] = enc["input_ids"].copy()
            return enc

        if _datasets is None:
            # 先在主进程中完成 tokenize，避免在 worker 里调用局部闭包
            _encoded = [_tok_map(x) for x in samples]
            # 为满足 Trainer 类型要求，优先使用 torch.utils.data.Dataset 适配
            try:
                from torch.utils.data import Dataset as _TorchDS  # type: ignore

                class _TorchDataset(_TorchDS):  # type: ignore
                    def __init__(self, arr):
                        self.arr = arr

                    def __len__(self):
                        return len(self.arr)

                    def __getitem__(self, i):
                        return self.arr[i]

                ds_train = _TorchDataset(_encoded)
            except Exception:
                ds_train = _MemDS(_encoded)
        else:
            ds_train = _datasets.Dataset.from_list(samples).map(
                _tok_map, remove_columns=list(samples[0].keys())
            )

        args = TrainingArguments(  # type: ignore[call-arg]
            output_dir=out_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=int(gradient_accumulation_steps),
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            logging_steps=max(1, int(log_steps)),
            save_steps=200,
            save_total_limit=2,
            fp16=fp16,
            bf16=(not fp16),
            report_to=[],
            dataloader_num_workers=0,
            gradient_checkpointing=False,
        )

        # 自定义进度显示回调：展示 step/total、loss、steps/sec、ETA、GPU显存
        class _ProgressCB(TrainerCallback):
            def __init__(self, log_every: int = 1):
                self.start = None
                self.last = None
                self.log_every = max(1, int(log_every))

            def on_train_begin(self, args, state, control, **kwargs):
                self.start = self.last = _time.time()

            def on_log(self, args, state, control, **kwargs):
                # 依赖 Trainer 的 logging_steps 触发
                now = _time.time()
                total_elapsed = now - (self.start or now)
                self.last = now
                gs = state.global_step or 0
                ms = state.max_steps or 0
                steps_per_sec = (gs / total_elapsed) if total_elapsed > 0 else 0.0
                eta_sec = ((ms - gs) / steps_per_sec) if steps_per_sec > 0 else 0.0

                def fmt_sec(x):
                    m, s = divmod(int(x), 60)
                    h, m = divmod(m, 60)
                    return f"{h:02d}:{m:02d}:{s:02d}"

                logs = kwargs.get("logs", {}) if kwargs else {}
                loss = logs.get("loss", None)
                lr = logs.get("learning_rate", None)
                # 显存
                mem_info = ""
                try:
                    if _cuda_ok:
                        a = _torch.cuda.memory_allocated(0) / (1024**3)
                        r = _torch.cuda.memory_reserved(0) / (1024**3)
                        mem_info = f" | GPU {a:.2f}G/{r:.2f}G"
                except Exception:
                    pass

                # 安全格式化（避免 None 导致格式化异常）
                def _fmt_loss(x):
                    try:
                        return f"{float(x):.4f}"
                    except Exception:
                        return "-"

                def _fmt_lr(x):
                    try:
                        return f"{float(x):.2e}"
                    except Exception:
                        return str(x) if x is not None else "-"

                _logging.info(
                    f"[STEP] {gs}/{ms} | loss={_fmt_loss(loss)} | lr={_fmt_lr(lr)} | {steps_per_sec:.2f} steps/s | ETA {fmt_sec(eta_sec)}{mem_info}"
                )

        trainer = Trainer(model=model, args=args, train_dataset=ds_train, callbacks=[_ProgressCB(log_steps)])  # type: ignore[arg-type]
        trainer.train()
        # 保存 LoRA 适配器
        adapter_dir = os.path.join(out_dir, "adapter")
        os.makedirs(adapter_dir, exist_ok=True)
        model.save_pretrained(adapter_dir)
        if hasattr(tok, "save_pretrained"):
            tok.save_pretrained(out_dir)

        _logging.info(f"[TRAIN] LoRA 适配器已保存: {adapter_dir}")
        return {"adapter_dir": adapter_dir, "samples": len(samples)}

    # ----------------------------
    # 使用 LLaMA-Factory 进行 LoRA 训练（外部训练器）
    # ----------------------------
    def train_rules_lora_with_llamafactory(
        self,
        train_jsonl: Optional[str] = None,
        rules_md_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        num_train_epochs: int = 1,
        learning_rate: float = 2e-5,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        max_seq_len: int = 4096,
        fp16: bool = True,
    ) -> dict:
        """使用 LLaMA-Factory 执行 LoRA 训练，并将产物保存到 output_dir。

        约定：
        - 数据采用 Alpaca 格式（instruction/input/output）；若传入 rules_md_path 而无 JSONL，则自动生成占位样本。
        - 依赖外部包 llamafactory（未安装则报错提示）。
        - 通过 CLI 子进程调用：python -m llamafactory.cli train ...
        """
        import json as _json
        import os as _os
        import sys as _sys
        import subprocess as _sp
        from typing import List as _List

        # 检查依赖
        try:
            import importlib.util as _ilu

            if _ilu.find_spec("llamafactory") is None:
                raise ImportError("llamafactory not installed")
        except Exception:
            raise RuntimeError(
                "未检测到 LLaMA-Factory。请先在当前环境安装：pip install -U llama-factory"
            )

        # 准备样本：优先使用 JSONL；否则基于规则文档构造占位样本
        samples = []
        if train_jsonl and _os.path.isfile(train_jsonl):
            try:
                samples = load_instruction_jsonl(train_jsonl)
            except Exception:
                samples = []
        if len(samples) == 0:
            if not rules_md_path:
                raise RuntimeError(
                    "未提供可用的训练数据（train_jsonl）或规则文档（rules_md_path）。"
                )
            samples = build_rules_sft_samples_from_md(
                rules_md_path, max_samples=50, chunk_chars=2000
            )

        # 将样本写为 Alpaca JSON（list而非jsonl），便于 LLaMA-Factory 直接消费
        out_dir = output_dir or self.lora_out_dir
        # 若未显式区分目录，默认将“判定任务”输出到 lora/judge，避免与分解器互相覆盖
        try:
            if (output_dir is None) or (
                os.path.abspath(output_dir) == os.path.abspath(self.lora_out_dir)
            ):
                out_dir = os.path.join(self.lora_out_dir, "judge")
        except Exception:
            pass
        data_dir = _os.path.join(out_dir, "llama_factory", "data")
        _os.makedirs(data_dir, exist_ok=True)
        data_file = _os.path.join(data_dir, "alpaca_sft.json")
        # 为提升兼容性，补充 prompt/query/response 字段（映射自 instruction/input/output）
        samples_out = []
        for ex in samples:
            ins = str(ex.get("instruction", "") or "")
            inp = str(ex.get("input", "") or "")
            out = str(ex.get("output", "") or "")
            samples_out.append(
                {
                    "instruction": ins,
                    "input": inp,
                    "output": out,
                    # 兼容某些版本的数据列命名
                    "prompt": ins,
                    "query": inp,
                    "response": out,
                }
            )
        with open(data_file, "w", encoding="utf-8") as f:
            _json.dump(samples_out, f, ensure_ascii=False, indent=2)

        # 写入 LLaMA-Factory 所需的数据集配置 dataset_info.json
        ds_info_path = _os.path.join(data_dir, "dataset_info.json")
        ds_info = {
            "alpaca_sft": {
                "file_name": "alpaca_sft.json",
                "formatting": "alpaca",
                # 明确列映射，尽量覆盖不同版本的键名
                "columns": {
                    "instruction": "instruction",
                    "input": "input",
                    "output": "output",
                    "prompt": "prompt",
                    "query": "query",
                    "response": "response",
                },
            }
        }
        try:
            with open(ds_info_path, "w", encoding="utf-8") as f:
                _json.dump(ds_info, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # 目标输出目录（LoRA 适配器会写在这里）
        _os.makedirs(out_dir, exist_ok=True)

        # 组装 LLaMA-Factory CLI 命令
        cmd: _List[str] = [
            _sys.executable,
            "-m",
            "llamafactory.cli",
            "train",
            "--stage",
            "sft",
            "--do_train",
            "--model_name_or_path",
            self.base_model_dir,
            "--finetuning_type",
            "lora",
            # 数据集：通过 dataset_dir + dataset 名称（文件名不含扩展）
            "--dataset_dir",
            data_dir,
            "--dataset",
            "alpaca_sft",
            # 训练超参
            "--output_dir",
            out_dir,
            "--num_train_epochs",
            str(int(num_train_epochs)),
            "--per_device_train_batch_size",
            str(int(per_device_train_batch_size)),
            "--learning_rate",
            str(float(learning_rate)),
            "--cutoff_len",
            str(int(max_seq_len)),
            "--gradient_accumulation_steps",
            str(int(gradient_accumulation_steps)),
            # 常用保存/日志（适度）
            "--save_steps",
            "200",
            "--logging_steps",
            "20",
            # 指定模板，减少自动解析不一致
            "--template",
            "qwen",
            "--overwrite_output_dir",
        ]
        # 精度/设备策略：
        # - 若无GPU：不传 --fp16/--bf16，默认 float32，避免 "Your setup doesn't support bf16/gpu" 报错。
        # - 若有GPU：fp16=True 则传 --fp16；否则在支持时传 --bf16，不支持则也不传。
        try:
            import torch as __torch

            _has_cuda = __torch.cuda.is_available()
            _bf16_ok = False
            try:
                _bf16_ok = bool(
                    getattr(__torch.cuda, "is_bf16_supported", lambda: False)()
                )
            except Exception:
                # 兼容旧版：通过算力粗略判断（Ampere>=8.0 通常支持 bfloat16）
                try:
                    cc = __torch.cuda.get_device_capability(0) if _has_cuda else (0, 0)
                    _bf16_ok = cc[0] >= 8
                except Exception:
                    _bf16_ok = False
            if _has_cuda:
                if fp16:
                    cmd += ["--fp16"]
                else:
                    if _bf16_ok:
                        cmd += ["--bf16"]
                    # 否则不加精度开关，使用默认精度
            # 无GPU：不添加精度开关（默认fp32）
        except Exception:
            # 回退：不设置精度开关
            pass

        _logging.info("[LLaMA-Factory] 开始训练： " + " ".join(cmd))
        # 运行子进程
        proc = _sp.run(cmd, cwd=self.root)
        if proc.returncode != 0:
            raise RuntimeError("LLaMA-Factory 训练失败，请检查终端输出与环境依赖。")

        _logging.info(f"[LLaMA-Factory] 训练完成。输出目录: {out_dir}")
        return {
            "adapter_dir": out_dir,
            "samples": len(samples),
            "backend": "llama-factory",
        }

    # ----------------------------
    # 小LLM（问题分解）训练：优先 LLaMA-Factory
    # ----------------------------
    def train_decomposer_with_llamafactory(
        self,
        decomp_events_file: Optional[str] = None,
        train_jsonl: Optional[str] = None,
        rules_md_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        num_train_epochs: int = 1,
        learning_rate: float = 2e-5,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        max_seq_len: int = 4096,
        fp16: bool = True,
    ) -> dict:
        """训练一个用于问题分解的小LLM（LoRA），让主LLM更聚焦冲突规则。优先使用 LLaMA-Factory。"""
        import json as _json
        import os as _os
        import sys as _sys
        import subprocess as _sp
        from typing import List as _List

        # 依赖检查
        try:
            import importlib.util as _ilu

            if _ilu.find_spec("llamafactory") is None:
                raise ImportError("llamafactory not installed")
        except Exception:
            raise RuntimeError(
                "未检测到 LLaMA-Factory。请先 pip install -U llama-factory"
            )

        # 构造样本：优先从事件文件生成；否则从现有 JSONL 推断并生成分解输出
        samples: list[dict] = []

        def _rules_text():
            return self.build_rules_prompt(rules_md_path) if rules_md_path else ""

        def _kg_text_for_ev(ev: str) -> str:
            svc = self.kg_service
            if svc is None:
                return "【KG状态】\n(离线模式，未加载图谱)"
            ents: List[str] = []
            try:
                for s, p, o in _extract_triples(ev):
                    ents += [str(s), str(o)]
            except Exception:
                pass
            ordered: List[str] = []
            seen = set()
            for x in ents:
                if x and x not in seen:
                    ordered.append(x)
                    seen.add(x)
            try:
                return svc.get_context_text(focus_entities=(ordered or None), limit=200)
            except Exception:
                return "【KG状态】\n(离线模式，服务异常)"

        def _build_one(ev: str) -> dict:
            rtxt = _rules_text()
            ktxt = _kg_text_for_ev(ev)
            inp = "\n\n".join([x for x in (rtxt, ktxt, "【事件】\n" + ev) if x])
            # 生成一个启发式的"分解输出"：使用规则+KG的简单检查
            conflicts: List[str] = []
            try:
                if self.kg_service is not None:
                    chk = self.kg_service.check_event_conflicts(ev) or {}
                    conflicts = chk.get("reasons", []) or []
            except Exception:
                conflicts = []
            # 目标是指导小LLM输出 JSON，但这里做一个占位示例输出作为监督
            out_lines = [
                "{",
                '  "entities": [提示：抽取飞机/跑道/停机位/牵引车],',
                '  "applicable_rules": [提示：列出与跑道互斥/停机位互斥/牵引唯一相关的规则要点],',
                f'  "potential_conflicts": { _json.dumps(conflicts, ensure_ascii=False) }',
                "}",
            ]
            return {
                "instruction": "将规则+KG+事件分解为JSON（entities/applicable_rules/potential_conflicts）",
                "input": inp,
                "output": "\n".join(out_lines),
            }

        # 首选：事件文件
        if decomp_events_file and _os.path.isfile(decomp_events_file):
            evs = load_events_from_file(decomp_events_file)
            for ev in evs:
                if isinstance(ev, str) and ev.strip():
                    samples.append(_build_one(ev.strip()))
        # 备选：从现有 JSONL 的 input 字段提取事件行
        if (not samples) and train_jsonl and _os.path.isfile(train_jsonl):
            arr = load_instruction_jsonl(train_jsonl)
            import re as _re

            for ex in arr:
                inp = str(ex.get("input", "") or "")
                m = _re.search(r"【事件】\n(.+)$", inp, flags=_re.S)
                ev = m.group(1).strip() if m else inp.strip()
                if ev:
                    samples.append(_build_one(ev))
        if not samples:
            raise RuntimeError(
                "未找到用于分解训练的样本（请提供 --decomp_events_file 或有效的 --train_jsonl）"
            )

        out_dir = output_dir or self.lora_out_dir
        data_dir = _os.path.join(out_dir, "llama_factory", "data_decomp")
        _os.makedirs(data_dir, exist_ok=True)
        data_file = _os.path.join(data_dir, "alpaca_decomp.json")
        with open(data_file, "w", encoding="utf-8") as f:
            _json.dump(samples, f, ensure_ascii=False, indent=2)
        ds_info_path = _os.path.join(data_dir, "dataset_info.json")
        ds_info = {
            "alpaca_decomp": {
                "file_name": "alpaca_decomp.json",
                "formatting": "alpaca",
                "columns": {
                    "instruction": "instruction",
                    "input": "input",
                    "output": "output",
                },
            }
        }
        with open(ds_info_path, "w", encoding="utf-8") as f:
            _json.dump(ds_info, f, ensure_ascii=False, indent=2)

        _os.makedirs(out_dir, exist_ok=True)
        cmd: _List[str] = [
            _sys.executable,
            "-m",
            "llamafactory.cli",
            "train",
            "--stage",
            "sft",
            "--do_train",
            "--model_name_or_path",
            self.decomp_base_model_dir,
            "--finetuning_type",
            "lora",
            "--dataset_dir",
            data_dir,
            "--dataset",
            "alpaca_decomp",
            "--output_dir",
            out_dir,
            "--num_train_epochs",
            str(int(num_train_epochs)),
            "--per_device_train_batch_size",
            str(int(per_device_train_batch_size)),
            "--learning_rate",
            str(float(learning_rate)),
            "--cutoff_len",
            str(int(max_seq_len)),
            "--gradient_accumulation_steps",
            str(int(gradient_accumulation_steps)),
            "--save_steps",
            "200",
            "--logging_steps",
            "20",
            "--template",
            "qwen",
            "--overwrite_output_dir",
        ]
        # 精度同前
        try:
            import torch as __torch

            _has_cuda = __torch.cuda.is_available()
            _bf16_ok = False
            try:
                _bf16_ok = bool(
                    getattr(__torch.cuda, "is_bf16_supported", lambda: False)()
                )
            except Exception:
                try:
                    cc = __torch.cuda.get_device_capability(0) if _has_cuda else (0, 0)
                    _bf16_ok = cc[0] >= 8
                except Exception:
                    _bf16_ok = False
            if _has_cuda:
                if fp16:
                    cmd += ["--fp16"]
                else:
                    if _bf16_ok:
                        cmd += ["--bf16"]
        except Exception:
            pass

        _logging.info("[LLaMA-Factory][Decomposer] 开始训练： " + " ".join(cmd))
        proc = _sp.run(cmd, cwd=self.root)
        if proc.returncode != 0:
            raise RuntimeError("LLaMA-Factory 分解器训练失败，请检查输出与环境。")
        _logging.info(f"[LLaMA-Factory][Decomposer] 训练完成。输出目录: {out_dir}")
        return {
            "adapter_dir": out_dir,
            "samples": len(samples),
            "backend": "llama-factory:decompose",
        }

    def _parse_judge_output(self, text: str) -> dict:
        """解析判定输出，提取合规标签/JSON。

        返回字典：{parsed: bool, compliance: str|None, reasons_len: int, raw: str}
        规则：
        1. 首选：提取最后一个可解析 JSON，对其中 compliance 字段进行判断。
        2. 回退：若文本只包含单独的“合规”或“冲突”两字（去除空白），即视为解析成功。
        3. 再回退：若在输出前 64 个字符中出现“合规/冲突/不冲突/无冲突”，按最早判定标签处理（容忍后续赘述）。
        不直接用“字符串是否包含 冲突”来判定，避免指令/示例污染导致误触发。
        """
        import json as _json
        import re as _re

        res = {"parsed": False, "compliance": None, "reasons_len": 0, "raw": text or ""}
        if not isinstance(text, str) or not text.strip():
            return res
        s = text.strip()
        # 去掉常见的代码围栏
        s = _re.sub(r"^```(?:json)?\s*", "", s)
        s = _re.sub(r"```\s*$", "", s)

        def _scan_last_json(t: str):
            starts = [i for i, ch in enumerate(t) if ch == "{"]
            last_obj = None
            for st in starts:
                depth = 0
                for j in range(st, len(t)):
                    ch = t[j]
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            frag = t[st : j + 1]
                            try:
                                obj = _json.loads(frag)
                                last_obj = obj
                            except Exception:
                                pass
                            break
            return last_obj

        # 优先解析 JSON_START/JSON_END 之间的内容
        m = _re.search(r"JSON_START\s*(\{[\s\S]*?\})\s*JSON_END", s)
        if m:
            try:
                obj = _json.loads(m.group(1))
            except Exception:
                obj = None
        else:
            obj = _scan_last_json(s)
        # 0) 新版（任务1）：两类顶层结构兼容：
        #   A) 评分示例结构：{"time":..., "时空信息01":{...}, "时空信息02":{...}, ..., "判定":..., "reason":[], "suggest":...}
        #   B) 我们早期结构：{"时空信息": [...], "判定": "合规|冲突", "依据": [...], "建议": "..."}
        if isinstance(obj, dict) and ("判定" in obj or "时空信息" in obj or any(isinstance(k, str) and k.startswith("时空信息") for k in obj.keys())):
            try:
                comp_cn = str(obj.get("判定", "")).strip()
                if comp_cn in ("合规", "冲突"):
                    res["parsed"] = True
                    res["compliance"] = comp_cn
                    rs = obj.get("依据", [])
                    if not isinstance(rs, list):
                        # 兼容英文键 reason
                        rs = obj.get("reason", [])
                    if isinstance(rs, str):
                        rs = [rs] if rs.strip() else []
                    if isinstance(rs, list):
                        res["reasons_len"] = len(rs)
                    return res
            except Exception:
                pass
        # 1) 兼容旧版：平面 JSON，包含全局 compliance
        if isinstance(obj, dict):
            comp = str(obj.get("compliance", "")).strip()
            if comp in ("合规", "冲突"):
                res["parsed"] = True
                res["compliance"] = comp
                rs = obj.get("reasons", [])
                if isinstance(rs, list):
                    res["reasons_len"] = len(rs)
                return res
            # 2) 新版：按时刻聚合的对象（含 time 与 多个“时空信息xx”）；若无顶层判定，则通过子项推断
            # 统计所有“时空信息xx”中的合规与依据条数（若子结构也带有 compliance/reasons）
            try:
                if "time" in obj:
                    any_conflict = False
                    reasons_cnt = 0
                    for k, v in obj.items():
                        if isinstance(k, str) and k.startswith("时空信息") and isinstance(v, dict):
                            c = str(v.get("compliance", "")).strip()
                            if c == "冲突":
                                any_conflict = True
                            rs = v.get("reasons", [])
                            if isinstance(rs, list):
                                reasons_cnt += len(rs)
                    # 若检测到该结构，则据此给出全局判断
                    if reasons_cnt > 0 or any_conflict or any(
                        isinstance(k, str) and k.startswith("时空信息") for k in obj.keys()
                    ):
                        res["parsed"] = True
                        res["compliance"] = "冲突" if any_conflict else "合规"
                        res["reasons_len"] = reasons_cnt
                        return res
            except Exception:
                pass
        # 3) 新版数组：[{time, 时空信息..}, ...]
        if isinstance(obj, list):
            try:
                any_conflict = False
                reasons_cnt = 0
                seen_any = False
                for item in obj:
                    if isinstance(item, dict):
                        # 直接兼容平面 JSON
                        c0 = str(item.get("compliance", "")).strip()
                        if c0 in ("合规", "冲突"):
                            seen_any = True
                            any_conflict = any_conflict or (c0 == "冲突")
                            rs0 = item.get("reasons", [])
                            if isinstance(rs0, list):
                                reasons_cnt += len(rs0)
                        # 兼容“时空信息xx”结构
                        for k, v in item.items():
                            if isinstance(k, str) and k.startswith("时空信息") and isinstance(v, dict):
                                seen_any = True
                                c = str(v.get("compliance", "")).strip()
                                if c == "冲突":
                                    any_conflict = True
                                rs = v.get("reasons", [])
                                if isinstance(rs, list):
                                    reasons_cnt += len(rs)
                if seen_any:
                    res["parsed"] = True
                    res["compliance"] = "冲突" if any_conflict else "合规"
                    res["reasons_len"] = reasons_cnt
                    return res
            except Exception:
                pass

        only = s.replace("\u3000", " ").strip()
        if only in ("合规", "冲突"):
            res["parsed"] = True
            res["compliance"] = only
            return res

        if ("不冲突" in s) or ("无冲突" in s):
            res["parsed"] = True
            res["compliance"] = "合规"
            return res
        # 进一步放宽：仅扫描前 64 字符，若出现单侧标签则采纳
        head = s[:64]
        # 若只出现“合规”而不出现“冲突”或出现“不冲突/无冲突” -> 合规
        if (
            ("不冲突" in head)
            or ("无冲突" in head)
            or ("合规" in head and "冲突" not in head)
        ):
            res["parsed"] = True
            res["compliance"] = "合规"
            return res
        # 若只出现“冲突”而不出现“合规” -> 冲突
        if (
            ("冲突" in head)
            and ("合规" not in head)
            and ("不冲突" not in head)
            and ("无冲突" not in head)
        ):
            res["parsed"] = True
            res["compliance"] = "冲突"
            return res
        return res

    # 统一的 JSONL 追加保存
    def _append_jsonl(self, path: Optional[str], obj: dict) -> None:
        if not path:
            return
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _build_minimal_judge_json(self, event_text: str, kg_check: Optional[dict] = None) -> dict:
        """在 LLM 输出不可解析时构造一个最小可用的判定 JSON（降级兜底）。

        结构：{"时空信息": [], "判定": "合规|冲突", "依据": [..<=3], "建议": "..."}
        - 时空信息为空数组（仅兜底，不做文本抽取以避免二次幻觉）
        - 判定/依据 来自 KG 检查（若可用），否则判定为合规、依据为空
        - 建议：冲突则给出通用操作建议，合规则“无”
        """
        reasons: list[str] = []
        label = "合规"
        if isinstance(kg_check, dict):
            rs = kg_check.get("reasons", []) or []
            if isinstance(rs, list):
                reasons = [str(x) for x in rs][:3]
            if len(reasons) > 0:
                label = "冲突"
        advice = "无" if label == "合规" else "调整资源或等待释放后再执行"
        # 为对齐评分示例，同时输出中文与英文键（reason/suggest）以增强兼容性
        return {
            "time": "",
            "时空信息01": {},
            "判定": label,
            "依据": reasons,
            "建议": advice,
            "reason": reasons,
            "suggest": advice,
        }

    def _generate_with_handle(
        self,
        model_key: str,
        prompts: List[str],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        timeout_sec: Optional[float] = None,
    ) -> List[str]:
        """使用已缓存的模型句柄执行批量生成。"""
        if not isinstance(prompts, list):
            raise TypeError("prompts 必须为 list[str]")
        if not prompts:
            return []

        if not getattr(self, "_models_ready", False):
            self._build_model()

        handle = self._model_handles.get(model_key)
        if not handle or handle.get("model") is None or handle.get("tokenizer") is None:
            raise RuntimeError(f"模型 {model_key} 尚未初始化")

        tok = handle["tokenizer"]
        model = handle["model"]
        device = handle.get("device")

        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover - torch 为强依赖
            raise RuntimeError("需要安装 torch 才能执行推理") from exc

        if device is None:
            try:
                device = next(model.parameters()).device
            except Exception:
                device = torch.device("cpu")

        if isinstance(device, str):
            device_obj = torch.device(device)
        else:
            device_obj = device

        timeout = (
            self.generate_timeout_sec if timeout_sec is None else max(0.0, float(timeout_sec))
        )

        max_new_tokens = max(1, int(max_tokens))
        if isinstance(device_obj, torch.device) and device_obj.type == "cpu":
            cpu_cap = max(1, getattr(self, "cpu_max_new_tokens", 64))
            if max_new_tokens > cpu_cap:
                _logging.warning(
                    f"[GEN] CPU 设备生成 {model_key} 时将 max_new_tokens 从 {max_new_tokens} 限制为 {cpu_cap}"
                )
                max_new_tokens = cpu_cap

        do_sample = temperature is not None and float(temperature) > 0.0
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tok.pad_token_id,
            "eos_token_id": tok.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)

        stopping_builder = None
        if timeout > 0:
            try:
                from transformers import StoppingCriteria, StoppingCriteriaList  # type: ignore

                class _TimeoutStopping(StoppingCriteria):
                    def __init__(self, limit: float) -> None:
                        self._limit = float(limit)
                        self._start = _time.monotonic()

                    def __call__(self, input_ids, scores, **kwargs):  # type: ignore[override]
                        return (_time.monotonic() - self._start) >= self._limit

                def _build_timeout_list() -> "StoppingCriteriaList":
                    return StoppingCriteriaList([_TimeoutStopping(timeout)])

                stopping_builder = _build_timeout_list
            except Exception:
                gen_kwargs["max_time"] = timeout

        outputs: List[str] = []

        with torch.no_grad():
            for prompt in prompts:
                text = prompt if isinstance(prompt, str) else str(prompt)
                inputs = tok(text, return_tensors="pt")
                try:
                    inputs = inputs.to(device_obj)
                except Exception:
                    inputs = {k: v.to(device_obj) for k, v in inputs.items()}

                start_ts = _time.monotonic()
                local_stopping = stopping_builder() if stopping_builder else None
                try:
                    if local_stopping is not None:
                        generated = model.generate(
                            **inputs, **gen_kwargs, stopping_criteria=local_stopping
                        )
                    else:
                        generated = model.generate(**inputs, **gen_kwargs)
                except Exception as err:  # noqa: BLE001
                    _logging.error(f"[GEN] {model_key} 生成失败: {err}")
                    raise
                duration = _time.monotonic() - start_ts
                if timeout > 0 and duration > timeout + 1:
                    _logging.warning(
                        f"[GEN] {model_key} 生成耗时 {duration:.1f}s，超过阈值 {timeout:.1f}s，输出可能被截断"
                    )

                prompt_len = inputs["input_ids"].shape[1]
                seq = generated[0].detach().cpu()
                outputs.append(
                    tok.decode(seq[prompt_len:], skip_special_tokens=True)
                )

        return outputs

    # ====== 新增：预热与卸载接口 ======
    def warmup_models(
        self,
        *,
        judge: bool = True,
        decomp: bool = True,
        order: str = "judge_first",
        timeout_sec: int = 300,
    ) -> None:
        """预加载常用模型，减少首次请求等待时间。"""
        _logging.info("[WARMUP] 触发模型预热")

        try:
            self._build_model()
        except Exception as exc:  # noqa: BLE001
            _logging.warning(f"[WARMUP] 模型加载失败: {exc}")
            return

        plan: List[str] = []
        if order == "decomp_first":
            if decomp:
                plan.append("decomp")
            if judge:
                plan.append("judge")
        else:
            if judge:
                plan.append("judge")
            if decomp:
                plan.append("decomp")

        for key in plan:
            try:
                handle = self._model_handles.get(key, {})
                if key == "decomp" and not handle.get("model"):
                    _logging.info("[WARMUP] 跳过 decomp（未加载分解模型）")
                    continue
                self._generate_with_handle(
                    key,
                    ["warmup"],
                    max_tokens=1,
                    temperature=0.0,
                    timeout_sec=min(float(timeout_sec), self.generate_timeout_sec),
                )
                _logging.info(f"[WARMUP] {key} 完成预热")
            except Exception as exc:  # noqa: BLE001
                _logging.warning(f"[WARMUP] {key} 预热失败: {exc}")

        _logging.info("[WARMUP] 预热流程结束")

    def unload_model(
        self, model_dir: str, lora_adapter_dir: Optional[str] = None
    ) -> None:
        """释放已经缓存的模型句柄。"""
        if not getattr(self, "_models_ready", False):
            _logging.info("[UNLOAD] 当前没有已加载的模型")
            return

        remove_keys = []
        for key, handle in list(self._model_handles.items()):
            if handle.get("model_dir") == model_dir and handle.get("lora_dir") == lora_adapter_dir:
                remove_keys.append(key)

        if not remove_keys:
            _logging.info(
                f"[UNLOAD] 未找到匹配模型: {model_dir} lora={lora_adapter_dir}"
            )
            return

        try:
            import torch  # type: ignore
        except Exception:
            torch = None  # type: ignore

        for key in remove_keys:
            handle = self._model_handles.pop(key, None)
            if handle and handle.get("model") is not None:
                mdl = handle.get("model")
                try:
                    del mdl
                except Exception:
                    pass
        if torch is not None and hasattr(torch, "cuda"):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        if not self._model_handles:
            self._models_ready = False
            self._judge_model = None
            self._judge_tokenizer = None
            self._judge_device = None
            self._decomp_model = None
            self._decomp_tokenizer = None
            self._decomp_device = None

        _logging.info(f"[UNLOAD] 已卸载模型: {remove_keys}")

    # 删除单条 judge 接口，统一通过 stream_judge_conflicts/Exp_main.run 路由
    def _marl_rectify_and_update_kg(
        self, kg: Dataset_KG, *, result_name_suffix: str = "auto"
    ) -> None:
        """检测到冲突时触发：优先用训练好的 MARL 模型进行一次评估并将最佳调度写回 KG；
        若未找到模型则回退到一次极小步的 KG 闭环训练（1 epoch），同样会写回 KG。
        同时不修改 htb_environment 源码。
        """
        import os as _os
        import sys as _sys
        from types import SimpleNamespace as _NS

        # 保障 htb_environment 的导入优先级与 utils 绑定
        try:
            _htb_dir = os.path.join(self.root, "htb_environment")
            if _htb_dir not in _sys.path:
                _sys.path.insert(0, _htb_dir)
            else:
                try:
                    _sys.path.remove(_htb_dir)
                except Exception:
                    pass
                _sys.path.insert(0, _htb_dir)
            # 清掉顶层 utils 缓存并绑定到 htb_environment/utils
            try:
                for _k in list(_sys.modules.keys()):
                    if _k == "utils" or _k.startswith("utils."):
                        _sys.modules.pop(_k, None)
            except Exception:
                pass
            try:
                import types as _types, importlib.util as _ilu

                _utils_dir = os.path.join(_htb_dir, "utils")
                if os.path.isdir(_utils_dir):
                    _pkg = _types.ModuleType("utils")
                    _pkg.__path__ = [_utils_dir]
                    _sys.modules["utils"] = _pkg
                    for _name in (
                        "util",
                        "job",
                        "task",
                        "plane",
                        "site",
                        "knowledgeGraph_test",
                        "schedule_converter",
                    ):
                        _file = os.path.join(_utils_dir, f"{_name}.py")
                        if os.path.isfile(_file):
                            _spec = _ilu.spec_from_file_location(
                                f"utils.{_name}", _file
                            )
                            if _spec and _spec.loader:
                                _mod = _ilu.module_from_spec(_spec)
                                _spec.loader.exec_module(_mod)
                                _sys.modules[f"utils.{_name}"] = _mod
            except Exception:
                pass
        except Exception:
            pass

        # 导入需要的组件
        from htb_environment.environment import ScheduleEnv  # type: ignore
        from htb_environment.MARL.runner import Runner  # type: ignore
        from htb_environment.MARL.common.arguments import get_mixer_args  # type: ignore
        from htb_environment.pipeline.kg_bridge import T1KGPriorAdapter, schedule_to_kg_triples  # type: ignore
        from htb_environment.pipeline.pipeline import run_kg_epoch_pipeline  # type: ignore

        _result_dir = os.path.join(self.root, "htb_environment", "result")
        _model_dir = os.path.join(self.root, "htb_environment", "MARL", "model")
        default_event_jsonl = os.path.join(
            self.root, "data_provider", "train_texts_conflict_aug.jsonl"
        )
        rectify_event_jsonl = (
            getattr(self.args, "marl_event_jsonl", None) or default_event_jsonl
        )
        rname = f"rectify_{result_name_suffix}"
        args = _NS(
            map="boatschedule",
            n_agents=int(getattr(self.args, "marl_n_agents", 8)),
            seed=123,
            alg="qmix",
            last_action=True,
            reuse_network=True,
            gamma=0.99,
            optimizer="RMS",
            evaluate_epoch=int(getattr(self.args, "marl_evaluate_epoch", 5)),
            model_dir=_model_dir,
            result_dir=_result_dir,
            result_name=rname,
            learn=False,
            cuda=bool(getattr(self.args, "marl_cuda", True)),
            replay_dir="",
            use_prior=True,
            prior_dim_site=int(getattr(self.args, "marl_prior_dim_site", 8)),
            prior_dim_plane=int(getattr(self.args, "marl_prior_dim_plane", 3)),
            obs_pad=int(getattr(self.args, "marl_obs_pad", 32)),
            n_epoch=None,
            n_episodes=None,
            train_steps=None,
            evaluate_cycle=None,
            batch_size=None,
            buffer_size=None,
            save_cycle=None,
            target_update_cycle=None,
            lr=None,
        )
        args = get_mixer_args(args)

        # 优先：评估已训练模型
        try:
            env = ScheduleEnv(args)
            prior = T1KGPriorAdapter(
                kg, ds=args.prior_dim_site, dp=args.prior_dim_plane
            )
            env.attach_prior(prior, args.prior_dim_site, args.prior_dim_plane)
            runner = Runner(env, args)
            _ = runner.evaluate()
            if runner.results.get("schedule_results"):
                gantt = runner.results["schedule_results"][-1]
                triples = schedule_to_kg_triples(gantt, env)
                kg.update_with_triples(triples)
            return
        except Exception:
            # 兜底：一次极小步的 KG 闭环训练（内部会写回 KG）
            try:
                small = _NS(**vars(args))
                small.learn = True
                small.load_model = False
                small.n_epoch = 1
                small.n_episodes = 1
                small.train_steps = 1
                small.evaluate_cycle = None
                # 统一传递现有 kg_service，保持与主流程一致
                run_kg_epoch_pipeline(
                    small,
                    kg_service=self.kg_service,
                    event_jsonl=rectify_event_jsonl,
                )
            except Exception:
                pass

    def stream_judge_conflicts(
        self,
        events_iter,
        focus_entities: Optional[List[str]] = None,
        rules_md_path: Optional[str] = None,
        batch_size: int = 4,
        simple_output: bool = False,
        show_decomposition: bool = False,
    ):
        """对事件流进行逐条/小批量判冲突（流式在推理时执行，而非训练时）。

        - events_iter: 可迭代的事件文本序列（例如生成器/列表）。
        - 每批最多 batch_size 条，生成对应输出后立即 yield，适合持续到来的实时日志。
        - 返回迭代器：每次 yield (event_text, output_str)
        """
        if not getattr(self, "_models_ready", False):
            self._build_model()

        judge_handle = self._model_handles.get("judge") if hasattr(self, "_model_handles") else None
        if not judge_handle or judge_handle.get("model") is None:
            raise RuntimeError("判定模型未初始化，无法执行流式判定")

        decomp_handle = self._model_handles.get("decomp") if hasattr(self, "_model_handles") else None
        decomp_available = bool(decomp_handle and decomp_handle.get("model"))

        # 规则提示一次构建，复用
        rules_text = self.build_rules_prompt(rules_md_path) if rules_md_path else ""
        # 小LLM分解器配置
        use_decomposer = bool(getattr(self.args, "enable_decomposer", False))

        _kg_vis_dir = self.kg_vis_dir
        _kg_vis_idx = self.kg_vis_idx

        batch_events: List[str] = []
        batch_prompts: List[str] = []
        for ev in events_iter:
            if not (isinstance(ev, str) and ev.strip()):
                continue
            ev = ev.strip()
            # 针对每条事件，使用“当前”KG状态生成 prompt（先判后更）
            # 仅关注“当前事件文本”中出现的实体：从抽取的三元组(subject/object)收集实体作为查询焦点。
            # 注意：不再合并全局 --focus_entities，严格限定上下文查询范围到本事件。
            auto_focus: List[str] = []
            try:
                trips = _extract_triples(ev)
                for s, p, o in trips:
                    for t in (s, o):
                        t = str(t).strip()
                        if t:
                            auto_focus.append(t)
            except Exception:
                pass
            # 去重：仅使用当前文本实体
            if auto_focus:
                seen = set()
                cur_focus = []
                for x in auto_focus:
                    if x and x not in seen:
                        cur_focus.append(x)
                        seen.add(x)
            else:
                cur_focus = None
            if self.kg_service is not None:
                try:
                    kg_text = self.kg_service.get_context_text(focus_entities=cur_focus, limit=200)
                except Exception:
                    kg_text = "【KG状态】\n(离线模式，服务异常)"
            else:
                kg_text = "【KG状态】\n(离线模式，未加载图谱)"
            # 可选：先调用小LLM做问题分解
            decomp_text = None
            if use_decomposer and decomp_available:
                # TODO: before training, using base model as default
                try:
                    d_prompt = self._format_decompose_prompt(ev, rules_text, kg_text)
                    d_outs = self._generate_with_handle(
                        "decomp",
                        [d_prompt],
                        max_tokens=1024,
                        temperature=0.0,
                        timeout_sec=self.generate_timeout_sec,
                    )
                    decomp_text = (d_outs[0] if d_outs else "") or ""
                    # 保存小模型输出到 JSONL
                    try:
                        import json as __json
                        import time as __t

                        payload = {
                            "ts": __t.strftime("%Y-%m-%d %H:%M:%S", __t.localtime()),
                            "event": ev,
                            "model": "decomposer",
                        }
                        try:
                            payload["output"] = __json.loads(decomp_text)
                        except Exception:
                            payload["output_raw"] = decomp_text
                        self._append_jsonl(self._decomp_out_file, payload)
                    except Exception:
                        pass
                except Exception as _e:
                    decomp_text = None
            elif use_decomposer and not decomp_available:
                try:
                    _logging.warning("[DECOMP] 已启用分解器但未加载分解模型，跳过该步骤")
                except Exception:
                    pass
            # 可选：打印分解器输出，便于人工复核
            # 按要求不输出具体内容
            if show_decomposition and decomp_text:
                try:
                    _logging.info("[SAVE] 分解器输出已保存到 JSONL（不在控制台展示）")
                except Exception:
                    pass

            # 主提示：附加分解结果（若有）
            # 注：为提高主模型对最终 JSON 结构的遵循度，不再将分解器输出注入主提示，避免多余JSON干扰。
            prompt = self._format_conflict_prompt_with_mode(
                ev,
                rules_text,
                kg_text,
                simple=simple_output,
                conflict_judge=bool(getattr(self.args, "conflict_judge", 1)),
            )
            batch_events.append(ev)
            batch_prompts.append(prompt)

            if len(batch_events) >= max(1, int(batch_size)):
                outs: Optional[List[str]] = None
                try:
                    outs = self._generate_with_handle(
                        "judge",
                        batch_prompts,
                        max_tokens=1024,
                        temperature=0.0,
                        timeout_sec=self.generate_timeout_sec,
                    )
                except (RuntimeError, OSError, MemoryError) as err:
                    try:
                        _logging.warning(f"[JUDGE] 主模型推理失败，尝试 simple 降级：{err}")
                    except Exception:
                        pass
                    outs = None

                if outs is None:
                    try:
                        if self.no_simple_fallback:
                            # 不进行 simple 降级；直接按原始完整提示重试一次，保持较大输出上限
                            outs = self._generate_with_handle(
                                "judge",
                                batch_prompts,
                                max_tokens=int(self.retry_max_new_tokens),
                                temperature=0.0,
                                timeout_sec=self.generate_timeout_sec,
                            )
                        else:
                            # simple 降级（保持原逻辑，但 max_tokens 改为可配置上限）
                            retry_prompts: List[str] = []
                            for ev_i in batch_events:
                                try:
                                    _auto = []
                                    try:
                                        _tr = _extract_triples(ev_i)
                                        for s, p, o in _tr:
                                            for t in (s, o):
                                                t = str(t).strip()
                                                if t:
                                                    _auto.append(t)
                                    except Exception:
                                        pass
                                    if _auto:
                                        _seen = set()
                                        _focus = []
                                        for x in _auto:
                                            if x and x not in _seen:
                                                _focus.append(x)
                                                _seen.add(x)
                                    else:
                                        _focus = None
                                    if self.kg_service is not None:
                                        try:
                                            _kg_txt = self.kg_service.get_context_text(
                                                focus_entities=_focus, limit=200
                                            )
                                        except Exception:
                                            _kg_txt = "【KG状态】\n(离线模式，服务异常)"
                                    else:
                                        _kg_txt = "【KG状态】\n(离线模式，未加载图谱)"
                                except Exception:
                                    _kg_txt = "【KG状态】\n(离线模式，未加载图谱)"
                                retry_prompts.append(
                                    self._format_conflict_prompt_with_mode(
                                        ev_i,
                                        rules_text,
                                        _kg_txt,
                                        simple=True,
                                        conflict_judge=bool(getattr(self.args, "conflict_judge", 1)),
                                    )
                                )
                            fallback_key = "decomp" if decomp_available else "judge"
                            outs = self._generate_with_handle(
                                fallback_key,
                                retry_prompts,
                                max_tokens=int(self.retry_max_new_tokens),
                                temperature=0.0,
                                timeout_sec=self.generate_timeout_sec,
                            )
                    except Exception:
                        outs = [""] * len(batch_prompts)
                # 若非 simple_output，则对无法解析的样本进行一次降级重试（simple 模式，仅输出“合规/冲突”）
                if not simple_output:
                    try:
                        parsed_list = [self._parse_judge_output(o) for o in outs]
                        retry_idx = [
                            i
                            for i, p in enumerate(parsed_list)
                            if not bool(p.get("parsed"))
                        ]
                        if retry_idx:
                            if self.no_simple_fallback:
                                # 使用原完整提示对失败样本重试
                                retry_prompts = [batch_prompts[i] for i in retry_idx]
                                try:
                                    re_outs = self._generate_with_handle(
                                        "judge",
                                        retry_prompts,
                                        max_tokens=int(self.retry_max_new_tokens),
                                        temperature=0.0,
                                        timeout_sec=self.generate_timeout_sec,
                                    )
                                except Exception:
                                    re_outs = [""] * len(retry_prompts)
                            else:
                                retry_prompts: List[str] = []
                                for i in retry_idx:
                                    ev_i = batch_events[i]
                                    # 复用当前KG快速构造 simple 提示
                                    try:
                                        _auto: List[str] = []
                                        try:
                                            _tr = _extract_triples(ev_i)
                                            for s, p, o in _tr:
                                                for t in (s, o):
                                                    t = str(t).strip()
                                                    if t:
                                                        _auto.append(t)
                                        except Exception:
                                            pass
                                        if _auto:
                                            _seen = set()
                                            _focus = []
                                            for x in _auto:
                                                if x and x not in _seen:
                                                    _focus.append(x)
                                                    _seen.add(x)
                                        else:
                                            _focus = None
                                        if self.kg_service is not None:
                                            try:
                                                _kg_txt = self.kg_service.get_context_text(focus_entities=_focus, limit=200)
                                            except Exception:
                                                _kg_txt = "【KG状态】\n(离线模式，服务异常)"
                                        else:
                                            _kg_txt = "【KG状态】\n(离线模式，未加载图谱)"
                                    except Exception:
                                        _kg_txt = "【KG状态】\n(离线模式，未加载图谱)"
                                    retry_prompts.append(
                                        self._format_conflict_prompt_with_mode(
                                            ev_i,
                                            rules_text,
                                            _kg_txt,
                                            simple=True,
                                            conflict_judge=bool(getattr(self.args, "conflict_judge", 1)),
                                        )
                                    )
                                # 执行一次小步重试（simple 提示，但使用可配置上限）
                                try:
                                    fallback_key = "decomp" if decomp_available else "judge"
                                    re_outs = self._generate_with_handle(
                                        fallback_key,
                                        retry_prompts,
                                        max_tokens=int(self.retry_max_new_tokens),
                                        temperature=0.0,
                                        timeout_sec=self.generate_timeout_sec,
                                    )
                                except Exception:
                                    re_outs = [""] * len(retry_prompts)
                            for j, idx in enumerate(retry_idx):
                                if (
                                    isinstance(re_outs, list)
                                    and j < len(re_outs)
                                    and re_outs[j]
                                ):
                                    outs[idx] = re_outs[j]
                    except Exception:
                        pass
                # 若 simple_output，仅保留首个“合规/冲突”标签
                if simple_output:

                    def _to_label(t: str) -> str:
                        if not isinstance(t, str):
                            return ""
                        if "合规" in t:
                            return "合规"
                        if "冲突" in t:
                            return "冲突"
                        return t.strip().splitlines()[0] if t.strip() else ""

                    outs = [_to_label(o) for o in outs]
                # 逐条回传，并在判定后更新 KG（冲突 -> 触发 MARL 矫正并回写；否则按事件增量回写）
                for e, o in zip(batch_events, outs):
                    # 统一：保存大模型（judge）输出（无论是否启用KG）
                    parsed = self._parse_judge_output(o)
                    try:
                        _logging.info(
                            f"[JUDGE] compliance={parsed.get('compliance')} parsed={parsed.get('parsed')} reasons={parsed.get('reasons_len')}"
                        )
                    except Exception:
                        pass
                    try:
                        import json as __json
                        import time as __t

                        payload = {
                            "ts": __t.strftime("%Y-%m-%d %H:%M:%S", __t.localtime()),
                            "event": e,
                            "model": "judge",
                            "compliance": parsed.get("compliance"),
                            "reasons_len": parsed.get("reasons_len"),
                        }
                        # 若解析失败，尝试使用 KG 结果构造最小 JSON 兜底
                        if not bool(parsed.get("parsed")):
                            _kg_chk = None
                            if self.kg_service is not None:
                                try:
                                    _kg_chk = self.kg_service.check_event_conflicts(e) or {}
                                except Exception:
                                    _kg_chk = None
                            _fallback = self._build_minimal_judge_json(e, _kg_chk)
                            payload["output"] = _fallback
                            payload["compliance"] = _fallback.get("判定")
                            payload["reasons_len"] = len(_fallback.get("依据", []) or [])
                        else:
                            try:
                                payload["output"] = __json.loads(o) if isinstance(o, str) else o
                            except Exception:
                                payload["output_raw"] = o
                        self._append_jsonl(self._judge_out_file, payload)
                    except Exception:
                        pass
                    if self.kg_service is not None:
                        # KG 交叉验证：仅当 KG 也能给出冲突理由时才触发 MARL
                        kg_reasons_cnt = 0
                        try:
                            _kg_chk = self.kg_service.check_event_conflicts(e) or {}
                            _kg_rs = _kg_chk.get("reasons", []) or []
                            kg_reasons_cnt = len(_kg_rs)
                        except Exception:
                            kg_reasons_cnt = 0
                        try:
                            _logging.info(f"[KGCHK] reasons={kg_reasons_cnt}")
                        except Exception:
                            pass
                        trigger_policy = getattr(self.args, "marl_trigger_policy", "llm_and_kg")
                        llm_conflict = bool(parsed.get("compliance") == "冲突")
                        kg_conflict = bool(kg_reasons_cnt > 0)
                        if trigger_policy == "llm_and_kg":
                            is_conflict = llm_conflict and kg_conflict
                        elif trigger_policy == "llm_or_kg":
                            is_conflict = llm_conflict or kg_conflict
                        elif trigger_policy == "llm_only":
                            is_conflict = llm_conflict
                        elif trigger_policy == "kg_only":
                            is_conflict = kg_conflict
                        else:
                            is_conflict = llm_conflict and kg_conflict
                        try:
                            _logging.info(
                                f"[MARL-TRIGGER] policy={trigger_policy} llm={llm_conflict} kg={kg_conflict} -> {is_conflict}"
                            )
                        except Exception:
                            pass
                        if is_conflict:
                            try:
                                if getattr(self.kg_service, "kg", None) is not None:
                                    self._marl_rectify_and_update_kg(
                                        self.kg_service.kg, result_name_suffix="auto"
                                    )
                            except Exception:
                                # 兜底：若 MARL 失败，至少将事件本身回写
                                try:
                                    self.kg_service.extract_and_update(e)
                                except Exception:
                                    pass
                        else:
                            try:
                                self.kg_service.extract_and_update(e)
                            except Exception:
                                pass
                        # 每次更新后都导出一张 KG 可视化图片
                        try:
                            import time as __t

                            _kg_vis_idx += 1
                            ts = int(__t.time())
                            out_png = _os.path.join(
                                _kg_vis_dir, f"kg_{ts}_{_kg_vis_idx:04d}.png"
                            )
                            self.kg_service.export_png(out_png)
                        except Exception:
                            pass
                    yield (e, o)
                batch_events, batch_prompts = [], []

        # 处理尾批
        if batch_events:
            outs: Optional[List[str]] = None
            try:
                outs = self._generate_with_handle(
                    "judge",
                    batch_prompts,
                    max_tokens=1024,
                    temperature=0.0,
                    timeout_sec=self.generate_timeout_sec,
                )
            except (RuntimeError, OSError, MemoryError) as err:
                try:
                    _logging.warning(f"[JUDGE] 主模型推理失败，尝试 simple 降级：{err}")
                except Exception:
                    pass
                outs = None

            if outs is None:
                try:
                    if self.no_simple_fallback:
                        # 不进行 simple 降级；直接按原始完整提示重试一次
                        outs = self._generate_with_handle(
                            "judge",
                            batch_prompts,
                            max_tokens=int(self.retry_max_new_tokens),
                            temperature=0.0,
                            timeout_sec=self.generate_timeout_sec,
                        )
                    else:
                        retry_prompts: List[str] = []
                        for ev_i in batch_events:
                            try:
                                _auto = []
                                try:
                                    _tr = _extract_triples(ev_i)
                                    for s, p, o in _tr:
                                        for t in (s, o):
                                            t = str(t).strip()
                                            if t:
                                                _auto.append(t)
                                except Exception:
                                    pass
                                if _auto:
                                    _seen = set()
                                    _focus = []
                                    for x in _auto:
                                        if x and x not in _seen:
                                            _focus.append(x)
                                            _seen.add(x)
                                else:
                                    _focus = None
                                if self.kg_service is not None:
                                    try:
                                        _kg_txt = self.kg_service.get_context_text(
                                            focus_entities=_focus, limit=200
                                        )
                                    except Exception:
                                        _kg_txt = "【KG状态】\n(离线模式，服务异常)"
                                else:
                                    _kg_txt = "【KG状态】\n(离线模式，未加载图谱)"
                            except Exception:
                                _kg_txt = "【KG状态】\n(离线模式，未加载图谱)"
                            retry_prompts.append(
                                self._format_conflict_prompt_with_mode(
                                    ev_i,
                                    rules_text,
                                    _kg_txt,
                                    simple=True,
                                    conflict_judge=bool(getattr(self.args, "conflict_judge", 1)),
                                )
                            )
                        fallback_key = "decomp" if decomp_available else "judge"
                        outs = self._generate_with_handle(
                            fallback_key,
                            retry_prompts,
                            max_tokens=int(self.retry_max_new_tokens),
                            temperature=0.0,
                            timeout_sec=self.generate_timeout_sec,
                        )
                except Exception:
                    outs = [""] * len(batch_prompts)
            # 若非 simple_output，则对无法解析的样本进行一次降级重试（simple 模式，仅输出“合规/冲突”）
            if not simple_output:
                try:
                    parsed_list = [self._parse_judge_output(o) for o in outs]
                    retry_idx = [
                        i
                        for i, p in enumerate(parsed_list)
                        if not bool(p.get("parsed"))
                    ]
                    if retry_idx:
                        if self.no_simple_fallback:
                            retry_prompts = [batch_prompts[i] for i in retry_idx]
                            try:
                                re_outs = self._generate_with_handle(
                                    "judge",
                                    retry_prompts,
                                    max_tokens=int(self.retry_max_new_tokens),
                                    temperature=0.0,
                                    timeout_sec=self.generate_timeout_sec,
                                )
                            except Exception:
                                re_outs = [""] * len(retry_prompts)
                        else:
                            retry_prompts: List[str] = []
                            for i in retry_idx:
                                ev_i = batch_events[i]
                                try:
                                    _auto: List[str] = []
                                    try:
                                        _tr = _extract_triples(ev_i)
                                        for s, p, o in _tr:
                                            for t in (s, o):
                                                t = str(t).strip()
                                                if t:
                                                    _auto.append(t)
                                    except Exception:
                                        pass
                                    if _auto:
                                        _seen = set()
                                        _focus = []
                                        for x in _auto:
                                            if x and x not in _seen:
                                                _focus.append(x)
                                                _seen.add(x)
                                    else:
                                        _focus = None
                                    if self.kg_service is not None:
                                        try:
                                            _kg_txt = self.kg_service.get_context_text(
                                                focus_entities=_focus, limit=200
                                            )
                                        except Exception:
                                            _kg_txt = "【KG状态】\n(离线模式，服务异常)"
                                    else:
                                        _kg_txt = "【KG状态】\n(离线模式，未加载图谱)"
                                except Exception:
                                    _kg_txt = "【KG状态】\n(离线模式，未加载图谱)"
                                retry_prompts.append(
                                    self._format_conflict_prompt_with_mode(
                                        ev_i,
                                        rules_text,
                                        _kg_txt,
                                        simple=True,
                                        conflict_judge=bool(getattr(self.args, "conflict_judge", 1)),
                                    )
                                )
                            try:
                                retry_key = "decomp" if decomp_available else "judge"
                                re_outs = self._generate_with_handle(
                                    retry_key,
                                    retry_prompts,
                                    max_tokens=int(self.retry_max_new_tokens),
                                    temperature=0.0,
                                    timeout_sec=self.generate_timeout_sec,
                                )
                            except Exception:
                                re_outs = [""] * len(retry_prompts)
                        for j, idx in enumerate(retry_idx):
                            if (
                                isinstance(re_outs, list)
                                and j < len(re_outs)
                                and re_outs[j]
                            ):
                                outs[idx] = re_outs[j]
                except Exception:
                    pass
            # 若 simple_output，仅保留首个“合规/冲突”标签
            if simple_output:

                def _to_label(t: str) -> str:
                    if not isinstance(t, str):
                        return ""
                    if "合规" in t:
                        return "合规"
                    if "冲突" in t:
                        return "冲突"
                    return t.strip().splitlines()[0] if t.strip() else ""

                outs = [_to_label(o) for o in outs]
            if batch_events:
                for e, o in zip(batch_events, outs):
                    parsed = self._parse_judge_output(o)
                    try:
                        _logging.info(
                            f"[JUDGE] compliance={parsed.get('compliance')} parsed={parsed.get('parsed')} reasons={parsed.get('reasons_len')}"
                        )
                    except Exception:
                        pass
                    # 保存大模型（judge）输出
                    try:
                        import json as __json
                        import time as __t

                        payload = {
                            "ts": __t.strftime("%Y-%m-%d %H:%M:%S", __t.localtime()),
                            "event": e,
                            "model": "judge",
                            "compliance": parsed.get("compliance"),
                            "reasons_len": parsed.get("reasons_len"),
                        }
                        if not bool(parsed.get("parsed")):
                            _kg_chk = None
                            if self.kg_service is not None:
                                try:
                                    _kg_chk = self.kg_service.check_event_conflicts(e) or {}
                                except Exception:
                                    _kg_chk = None
                            _fallback = self._build_minimal_judge_json(e, _kg_chk)
                            payload["output"] = _fallback
                            payload["compliance"] = _fallback.get("判定")
                            payload["reasons_len"] = len(_fallback.get("依据", []) or [])
                        else:
                            try:
                                payload["output"] = __json.loads(o) if isinstance(o, str) else o
                            except Exception:
                                payload["output_raw"] = o
                        self._append_jsonl(self._judge_out_file, payload)
                    except Exception:
                        pass
                    if self.kg_service is not None:
                        kg_reasons_cnt = 0
                        try:
                            _kg_chk = self.kg_service.check_event_conflicts(e) or {}
                            _kg_rs = _kg_chk.get("reasons", []) or []
                            kg_reasons_cnt = len(_kg_rs)
                        except Exception:
                            kg_reasons_cnt = 0
                        try:
                            _logging.info(f"[KGCHK] reasons={kg_reasons_cnt}")
                        except Exception:
                            pass
                        trigger_policy = getattr(self.args, "marl_trigger_policy", "llm_and_kg")
                        llm_conflict = bool(parsed.get("compliance") == "冲突")
                        kg_conflict = bool(kg_reasons_cnt > 0)
                        if trigger_policy == "llm_and_kg":
                            is_conflict = llm_conflict and kg_conflict
                        elif trigger_policy == "llm_or_kg":
                            is_conflict = llm_conflict or kg_conflict
                        elif trigger_policy == "llm_only":
                            is_conflict = llm_conflict
                        elif trigger_policy == "kg_only":
                            is_conflict = kg_conflict
                        else:
                            is_conflict = llm_conflict and kg_conflict
                        try:
                            _logging.info(
                                f"[MARL-TRIGGER] policy={trigger_policy} llm={llm_conflict} kg={kg_conflict} -> {is_conflict}"
                            )
                        except Exception:
                            pass
                        if is_conflict:
                            try:
                                if getattr(self.kg_service, "kg", None) is not None:
                                    self._marl_rectify_and_update_kg(
                                        self.kg_service.kg, result_name_suffix="auto"
                                    )
                            except Exception:
                                try:
                                    self.kg_service.extract_and_update(e)
                                except Exception:
                                    pass
                        else:
                            try:
                                self.kg_service.extract_and_update(e)
                            except Exception:
                                pass
                        # 可视化
                        try:
                            import time as __t

                            _kg_vis_idx += 1
                            ts = int(__t.time())
                            out_png = os.path.join(
                                _kg_vis_dir, f"kg_{ts}_{_kg_vis_idx:04d}.png"
                            )
                            self.kg_service.export_png(out_png)
                        except Exception:
                            pass
                    yield (e, o)
        # 更新实例中的可视化计数器，便于后续继续追加
        self.kg_vis_idx = _kg_vis_idx

    # ----------------------------
    # 任务一结果格式导出
    # ----------------------------
    def export_task1_results(self, source_jsonl: Optional[str] = None, out_path: Optional[str] = None) -> Optional[str]:
        """将判定阶段的 judge JSONL 文件转换为任务一评分规则格式。

        输出为 JSON 数组：每元素至少包含顶层键：time 与若干条信息字段。
        - 信息字段类别：
            - 时空信息 01..NN（必有）
            - 历史信息 01..NN（可选，兼容键名：历史统计01..NN）
            - 态势感知 01..NN（可选，兼容键名：态势预测01..NN）
        - 键名带空格，编号两位补零；值为中文描述句，以全角分号结尾。
        - 若缺失 time，则尝试从子项或记录时间戳 ts 推断。
        - 对于对象值（而非字符串值）的时空信息，将基于常见键拼装句子并附“对应能力项（…）”。
        """
        import json, os, logging, re
        logger = logging.getLogger("task1_export")
        src = source_jsonl or getattr(self, "_judge_out_file", None)
        if not src or not os.path.isfile(src):
            logger.warning(f"[TASK1] 源文件不存在: {src}")
            return None
        out_path = out_path or os.path.join(self.results_out_dir, "task1_result.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        def _norm_key(k: str) -> str:
            if not isinstance(k, str):
                return k
            # 统一三类键名：时空信息/历史信息(历史统计)/态势感知(态势预测)
            def _norm_with(prefixes, target_prefix):
                for p in prefixes:
                    if k.startswith(p):
                        num = k[len(p):]
                        num = re.sub(r"[^0-9]", "", num)
                        if num:
                            return f"{target_prefix} {num.zfill(2)}"
                        return f"{target_prefix} 01"
                return None
            hit = (
                _norm_with(["时空信息 ", "时空信息"], "时空信息") or
                _norm_with(["历史信息 ", "历史信息", "历史统计 ", "历史统计"], "历史信息") or
                _norm_with(["态势感知 ", "态势感知", "态势预测 ", "态势预测"], "态势感知")
            )
            if hit:
                return hit
            return k

        def _build_sentence(obj: dict) -> str:
            if not isinstance(obj, dict) or not obj:
                return ""
            plane_id = obj.get("飞机ID") or obj.get("飞机")
            op_rel = obj.get("作业关系") or obj.get("作业ID")
            plane_state = obj.get("飞机状态")
            speed = obj.get("速度") or obj.get("速率")
            state_rel = obj.get("状态关系")
            loc_rel = obj.get("位置关系")
            gate_id = obj.get("停机位ID")
            parts = []
            if plane_id:
                parts.append(f"飞机 {plane_id}")
            if op_rel:
                parts.append(f"正在开展作业，{op_rel}")
            if plane_state:
                parts.append(f"处于，{plane_state}")
            if speed:
                parts.append(f"（速度{speed}）")
            handled = {"飞机ID", "飞机", "作业关系", "作业ID", "飞机状态", "速度", "速率", "状态关系", "位置关系", "停机位ID"}
            for k, v in obj.items():
                if k in handled:
                    continue
                if isinstance(v, (str, int, float)) and str(v):
                    parts.append(f"{k}{v}")
            ability_items = []
            for k in ["飞机ID", "作业关系", "飞机状态", "状态关系", "位置关系", "停机位ID"]:
                if k in obj and obj.get(k):
                    ability_items.append(k)
            seen = set()
            ability_items = [x for x in ability_items if not (x in seen or seen.add(x))]
            if ability_items:
                parts.append(f"对应能力项（{'，'.join(能力项 for 能力项 in ability_items)}）")
            sentence = "，".join([p for p in parts if p])
            if sentence and sentence[-1] not in "；。":
                sentence += "；"
            return sentence

        results = []
        with open(src, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                out_obj = rec.get("output")
                if not isinstance(out_obj, dict):
                    # 兼容当前 stream 仅保存 output_raw 的情况：从 output_raw 中提取最后一个 JSON
                    raw = rec.get("output_raw")
                    if isinstance(raw, str) and raw.strip():
                        # === 新增：先进行全角字符与注释的规范化，提升解析率 ===
                        def _normalize_fullwidth(s: str) -> str:
                            # 去除行尾注释 (# ...)
                            import re as _re
                            s = _re.sub(r"#[^\n]*", "", s)
                            # 全角括号/引号/冒号替换
                            trans_map = {
                                "｛": "{", "｝": "}", "“": '"', "”": '"', "：": ":",
                            }
                            # 统一逗号格式：中文逗号保留，不影响 JSON 解析
                            for k, v in trans_map.items():
                                s = s.replace(k, v)
                            # 双层包裹 "{{ ... }}" -> "{ ... }"
                            s = _re.sub(r"\{\s*\{", "{", s)
                            s = _re.sub(r"\}\s*\}", "}", s)
                            return s
                        raw_norm = _normalize_fullwidth(raw)
                        # 若用户输出是近似 JSON 但键值全为中文字符，这里尝试快速补逗号/分号处理
                        # 将可能的 "；" 视为结束标点，但不影响 JSON；保留在字符串内部
                        # 尝试匹配 JSON_START/JSON_END
                        m = re.search(r"JSON_START\s*(\{[\s\S]*?\})\s*JSON_END", raw_norm)
                        frag = None
                        if m:
                            frag = m.group(1)
                        else:
                            # 扫描最后一个 { .. }
                            sraw = raw_norm
                            starts = [i for i, ch in enumerate(sraw) if ch == "{"]
                            for st in starts:
                                depth = 0
                                for j in range(st, len(sraw)):
                                    ch = sraw[j]
                                    if ch == "{":
                                        depth += 1
                                    elif ch == "}":
                                        depth -= 1
                                        if depth == 0:
                                            cand = sraw[st:j+1]
                                            try:
                                                _ = json.loads(cand)
                                                frag = cand
                                            except Exception:
                                                pass
                                            break
                        if frag:
                            try:
                                out_obj = json.loads(frag)
                            except Exception:
                                out_obj = None
                        # 二次回退：若仍无法解析，尝试手工抽取 三类信息 与 time
                        if not isinstance(out_obj, dict):
                            # 手工模式：按行解析 "\"时空信息 01\":\"...\"" 或 全角形式
                            fw = raw_norm
                            # 简单时间提取
                            import re as _re2
                            time_match = _re2.search(r'"time"\s*:\s*"([0-9:\-\s]+)"', fw)
                            if not time_match:
                                time_match = _re2.search(r'"时间"\s*:\s*"([0-9:\-\s]+)"', fw)
                            # 提取三类信息键值对（支持多条，含同义键名）
                            kv_pairs = []
                            for pat in [
                                r'"(时空信息\s*\d{1,2})"\s*:\s*"([^"]+?)"',
                                r'"(历史信息\s*\d{1,2})"\s*:\s*"([^"]+?)"',
                                r'"(历史统计\s*\d{1,2})"\s*:\s*"([^"]+?)"',
                                r'"(态势感知\s*\d{1,2})"\s*:\s*"([^"]+?)"',
                                r'"(态势预测\s*\d{1,2})"\s*:\s*"([^"]+?)"',
                            ]:
                                kv_pairs.extend(_re2.findall(pat, fw))
                            if kv_pairs:
                                tmp_obj: Dict[str, Any] = {}
                                if time_match:
                                    tmp_obj["time"] = time_match.group(1)
                                for k, v in kv_pairs:
                                    tmp_obj[k] = v
                                out_obj = tmp_obj if tmp_obj else None
                if not isinstance(out_obj, dict):
                    continue
                time_val = out_obj.get("time") or out_obj.get("时间")
                if not time_val:
                    for k, v in out_obj.items():
                        if isinstance(k, str) and k.startswith("时空信息") and isinstance(v, dict):
                            time_val = v.get("时间") or v.get("time")
                            if time_val:
                                break
                if not time_val:
                    time_val = rec.get("ts")
                pack = {"time": time_val}
                # 收集三类信息并归一化
                all_items: List[Tuple[str, Any]] = []
                for raw_k, v in out_obj.items():
                    if not isinstance(raw_k, str):
                        continue
                    nk = _norm_key(raw_k)
                    if isinstance(nk, str) and (
                        nk.startswith("时空信息 ") or nk.startswith("历史信息 ") or nk.startswith("态势感知 ")
                    ):
                        all_items.append((nk, v))
                # 排序：时空信息→历史信息→态势感知，各自编号升序
                def _sort_key(item: Tuple[str, Any]) -> Tuple[int, int]:
                    name = item[0]
                    if name.startswith("时空信息 "):
                        cat = 0
                    elif name.startswith("历史信息 "):
                        cat = 1
                    else:
                        cat = 2
                    m = re.search(r" (\d{2})$", name)
                    num = int(m.group(1)) if m else 99
                    return (cat, num)
                for nk, v in sorted(all_items, key=_sort_key):
                    if isinstance(v, dict):
                        pack[nk] = _build_sentence(v)
                    elif isinstance(v, str):
                        pack[nk] = v if v.endswith("；") or v.endswith("。") else v + "；"
                    else:
                        pack[nk] = ""
                results.append(pack)
        try:
            with open(out_path, "w", encoding="utf-8") as wf:
                json.dump(results, wf, ensure_ascii=False, indent=2)
            logger.info(f"[TASK1] 导出完成 -> {out_path} (records={len(results)})")
        except Exception as e:
            logger.warning(f"[TASK1] 写文件失败: {e}")
            return None
        return out_path

    # =====================================================================
    # MARL 训练
    # =====================================================================
    def train_marl(
        self,
        *,
        use_task1_kg: bool = True,
        n_agents: int = 8,
        result_dir: Optional[str] = None,
        result_name: str = "exp",
        n_epoch: int = 5,
        n_episodes: int = 5,
        train_steps: int = 2,
        evaluate_cycle: int = 5,
        evaluate_epoch: int = 20,
        batch_size: int = 32,
        buffer_size: int = 1000,
        target_update_cycle: int = 200,
        save_cycle: int = 50,
        lr: float = 5e-4,
        cuda: bool = True,
        use_prior: bool = True,
        prior_dim_site: int = 8,
        prior_dim_plane: int = 3,
        obs_pad: int = 32,
        export_csv: bool = True,
        eval_only: bool = False,
        event_jsonl: Optional[str] = None,
    ) -> dict:
        
        # 先确保优先搜索 htb_environment 下的 utils（避免被项目根目录下的 utils 覆盖）
        try:
            _htb_dir = os.path.join(self.root, "htb_environment")
            # 1) 将 htb_environment 放到 sys.path 最前
            if _htb_dir not in _sys.path:
                _sys.path.insert(0, _htb_dir)
            else:
                # 移到最前，确保优先级
                try:
                    _sys.path.remove(_htb_dir)
                except Exception:
                    pass
                _sys.path.insert(0, _htb_dir)

            # 2) 将项目根目录移到更靠后位置，降低与 htb_environment/utils 的冲突概率
            _proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            if _proj_root in _sys.path:
                try:
                    _sys.path.remove(_proj_root)
                    _sys.path.append(_proj_root)
                except Exception:
                    pass

            # 3) 主动清理已加载的顶层 'utils' 相关模块，避免被缓存占用
            try:
                for _k in list(_sys.modules.keys()):
                    if _k == "utils" or _k.startswith("utils."):
                        _sys.modules.pop(_k, None)
            except Exception:
                pass

            # 4) 绑定一个指向 htb_environment/utils 的“合成”包，确保 from utils.xxx 导向正确位置
            try:
                import importlib.util as _ilu
                import types as _types

                _utils_dir = os.path.join(_htb_dir, "utils")
                if os.path.isdir(_utils_dir):
                    _pkg = _types.ModuleType("utils")
                    _pkg.__path__ = [_utils_dir]  # 作为包目录
                    _sys.modules["utils"] = _pkg
                    # 预加载关键子模块，避免后续再次落到顶层 utils
                    for _name in (
                        "util",
                        "job",
                        "task",
                        "plane",
                        "site",
                        "knowledgeGraph_test",
                    ):
                        _file = os.path.join(_utils_dir, f"{_name}.py")
                        if os.path.isfile(_file):
                            _spec = _ilu.spec_from_file_location(
                                f"utils.{_name}", _file
                            )
                            if _spec and _spec.loader:
                                _mod = _ilu.module_from_spec(_spec)
                                _spec.loader.exec_module(_mod)
                                _sys.modules[f"utils.{_name}"] = _mod
            except Exception:
                pass
        except Exception:
            pass

        # 动态导入，避免非 MARL 场景的硬依赖
        from types import SimpleNamespace as _NS

        try:
            from htb_environment.environment import ScheduleEnv  # type: ignore
            from htb_environment.MARL.runner import Runner  # type: ignore
            from htb_environment.MARL.common.arguments import get_mixer_args  # type: ignore
            from htb_environment.pipeline.pipeline import run_kg_epoch_pipeline  # type: ignore
            from htb_environment.utils.knowledgeGraph_test import KGPrior  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"无法导入 htb_environment 组件，请确认路径与依赖可用：{e}"
            )

        # 组织参数对象（仿照 htb_environment 的 args 结构）
        _result_dir = result_dir or os.path.join(self.root, "htb_environment", "result")
        _model_dir = os.path.join(self.root, "htb_environment", "MARL", "model")
        args = _NS(
            # 通用
            map="boatschedule",
            n_agents=int(n_agents),
            seed=123,
            alg="qmix",
            last_action=True,
            reuse_network=True,
            gamma=0.99,
            optimizer="RMS",
            evaluate_epoch=int(evaluate_epoch),
            model_dir=_model_dir,
            result_dir=_result_dir,
            result_name=str(result_name),
            load_model=False,
            learn=not bool(eval_only),
            cuda=bool(cuda),
            # 先验
            replay_dir="",
            use_prior=bool(use_prior),
            prior_dim_site=int(prior_dim_site),
            prior_dim_plane=int(prior_dim_plane),
            obs_pad=int(obs_pad),
            # 循环与缓冲
            n_epoch=int(n_epoch),
            n_episodes=int(n_episodes),
            train_steps=int(train_steps),
            evaluate_cycle=int(evaluate_cycle),
            batch_size=int(batch_size),
            buffer_size=int(buffer_size),
            save_cycle=int(save_cycle),
            target_update_cycle=int(target_update_cycle),
            lr=float(lr),
            # ε-greedy（用默认 None 让 get_mixer_args 自动填充）
            epsilon_start=None,
            epsilon_end=None,
            epsilon_anneal_steps=None,
            epsilon_anneal_scale=None,
            # 其他
            use_task1_kg=bool(use_task1_kg),
            export_csv=bool(export_csv),
            # Neo4j 连接，用于 KG 模式
            neo4j_uri=self.neo4j_uri,
            neo4j_user=self.neo4j_user,
            neo4j_password=self.neo4j_password,
            neo4j_database=self.neo4j_database,
        )

        # KG 闭环：直接调用 htb_environment.pipeline
        if use_task1_kg:
            # 统一传递已有 kg_service，确保图谱增量更新一致
            run_kg_epoch_pipeline(
                args,
                kg_service=self.kg_service,
                event_jsonl=event_jsonl,
            )
            return {
                "use_task1_kg": True,
                "result_dir": _result_dir,
                "result_name": result_name,
                "learn": not bool(eval_only),
                "event_jsonl": event_jsonl,
            }

        # 纯 MARL（无 KG）：复用其环境与 Runner
        # 先 attach mixer 缺省参数
        args = get_mixer_args(args)

        env = ScheduleEnv(args)
        if bool(use_prior):
            try:
                prior = KGPrior(ds=int(prior_dim_site), dp=int(prior_dim_plane))
                env.attach_prior(prior, int(prior_dim_site), int(prior_dim_plane))
            except Exception:
                # 忽略先验失败，继续
                pass

        # 初始化并同步维度
        env.reset(args.n_agents)
        info = env.get_env_info()
        args.n_actions = int(info["n_actions"])  # type: ignore[attr-defined]
        args.state_shape = int(info["state_shape"])  # type: ignore[attr-defined]
        args.obs_shape = int(info["obs_shape"])  # type: ignore[attr-defined]
        args.episode_limit = int(info["episode_limit"])  # type: ignore[attr-defined]

        runner = Runner(env, args)
        # 设备诊断：明确当前训练所用设备
        try:
            import torch as _torch  # type: ignore

            _dev = getattr(runner.agents.policy, "device", None)
            _logging.info(
                f"[MARL] torch.cuda.is_available={_torch.cuda.is_available()} | args.cuda={args.cuda} | policy.device={_dev}"
            )
        except Exception:
            pass
        if bool(eval_only):
            _wr, _r, _t, _mv = runner.evaluate()
        else:
            runner.run(args.alg)
        return {
            "use_task1_kg": False,
            "result_dir": _result_dir,
            "result_name": result_name,
            "learn": not bool(eval_only),
            "event_jsonl": event_jsonl,
        }
