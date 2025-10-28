r'''
Author: zy
Date: 2025-10-22 17:35:46
LastEditTime: 2025-10-23 19:49:44
LastEditors: zy
Description: 
FilePath: haitianbei/exp/exp_main.py

'''
# 兼容直接"运行本文件"的调试方式：确保项目根目录在 sys.path
# 这样即使用 "python exp/exp_main.py" 启动，也能导入顶层包（exp、utils、data_provider）。
import os as _os
import sys as _sys
if __package__ is None or __package__ == "":
    _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))

from exp.exp_basic import Exp_Basic
from models.triples_extraction import extract_triples as _extract_triples
from data_provider.data_loader import Dataset_KG
from data_provider.data_loader import load_instruction_jsonl, build_rules_sft_samples_from_md
import os
import json
from typing import Optional
from typing import List, Tuple
import time as _time

# 训练/推理相关的可选依赖（惰性导入）
try:
    import transformers  # type: ignore
except Exception:
    transformers = None  # type: ignore
try:
    import peft  # type: ignore
except Exception:
    peft = None  # type: ignore
try:
    import datasets as _datasets  # type: ignore
except Exception:
    _datasets = None  # type: ignore
try:
    import bitsandbytes as _bnb  # noqa: F401  # type: ignore
except Exception:
    _bnb = None  # type: ignore
try:
    import vllm  # type: ignore
except Exception:
    vllm = None  # type: ignore

# 顶层内存数据集（可被 DataLoader 多进程 pickling）
class _MemDS:
    def __init__(self, arr):
        # arr 应为已编码好的字典列表（包含 input_ids/labels 等）
        self.arr = list(arr)
    def __len__(self):
        return len(self.arr)
    def __getitem__(self, i):
        return self.arr[i]

class Exp_main(Exp_Basic):
    def __init__(self, args):
        super(Exp_main, self).__init__(args)
        # 根路径
        self.root = getattr(args, 'root', os.getcwd())
        # Neo4j 连接参数
        self.neo4j_uri = getattr(args, 'neo4j_uri', None)
        self.neo4j_user = getattr(args, 'neo4j_user', None)
        self.neo4j_password = getattr(args, 'neo4j_password', None)
        self.neo4j_database = getattr(args, 'neo4j_database', None)
        # 离线模式：跳过 KG 构建
        self.skip_kg = getattr(args, 'skip_kg', False)
        # 是否在构建前重置 KG（保留固定节点）（流推理中通常不需要）
        self.reset_kg = bool(getattr(args, 'reset_kg', False))
        # 基座模型目录（Qwen3-4B）
        self.base_model_dir = getattr(args, 'base_model_dir', os.path.join(self.root, 'models', 'Qwen3-4B'))
        # LoRA 输出目录
        self.lora_out_dir = getattr(args, 'lora_out_dir', os.path.join(self.root, 'results_entity_judge', 'lora'))

    def _build_model(self):
        """占位，满足基类在构造时的要求（torch 可选）。"""
        try:
            import torch.nn as nn  # type: ignore
            return nn.Identity(), []
        except Exception:
            class DummyModel:
                def to(self, device):
                    return self
            return DummyModel(), []

    def run(self):
        """统一入口：根据 mode 执行 LoRA 训练或事件流冲突判定。"""
        mode = getattr(self.args, 'mode', 'stream-judge')

        if mode == 'train':
            # 读取训练超参
            fp16 = not bool(getattr(self.args, 'no_fp16', False))
            info = self.train_rules_lora(
                train_jsonl=getattr(self.args, 'train_jsonl', None),
                rules_md_path=getattr(self.args, 'rules_md_path', None),
                output_dir=getattr(self.args, 'lora_out_dir', self.lora_out_dir),
                num_train_epochs=int(getattr(self.args, 'num_train_epochs', 1)),
                learning_rate=float(getattr(self.args, 'learning_rate', 2e-5)),
                per_device_train_batch_size=int(getattr(self.args, 'per_device_train_batch_size', 1)),
                use_4bit=bool(getattr(self.args, 'use_4bit', False)),
                fp16=fp16,
                augment_train_with_kg=not bool(getattr(self.args, 'no_augment_train_with_kg', False)),
                gradient_accumulation_steps=int(getattr(self.args, 'grad_accum_steps', 1)),
                max_seq_len=int(getattr(self.args, 'max_seq_len', 1024)),
                prefer_device=str(getattr(self.args, 'device', 'auto')),
                log_steps=int(getattr(self.args, 'log_steps', 1)),
            )
            print("\n=== Train Finished ===")
            print("adapter_dir :", info.get("adapter_dir"))
            print("samples     :", info.get("samples"))
            return info

        # 默认：stream-judge
        events_file = getattr(self.args, 'events_file', None)
        # 若未显式提供，尝试回退到 tests/events_sample.txt
        if not events_file or not os.path.isfile(events_file):
            _default_ev = os.path.join(self.root, 'tests', 'events_sample.txt')
            if os.path.isfile(_default_ev):
                print(f"[stream-judge] 未提供 --events_file，使用默认样例: {_default_ev}")
                events_file = _default_ev
            else:
                print("[stream-judge] 需要 --events_file (每行一条事件)。示例:")
                print("  python run.py --mode stream-judge --events_file tests/events_sample.txt --rules_md_path tests/rules_sample.md")
                return {"error": "missing_events_file"}
        with open(events_file, 'r', encoding='utf-8') as f:
            events = [ln.strip() for ln in f if ln.strip()]
        focus = [s.strip() for s in str(getattr(self.args, 'focus_entities', '') or '').split(',') if s.strip()]
        rules_md_path = getattr(self.args, 'rules_md_path', None)
        lora_adapter_dir = getattr(self.args, 'lora_adapter_dir', None)
        use_vllm = not bool(getattr(self.args, 'no_vllm', False))
        batch_size = max(1, int(getattr(self.args, 'batch_size', 4)))
        simple_output = bool(getattr(self.args, 'simple_output', False))

        print("\n=== Stream Judge Start ===")
        count = 0
        for ev, out in self.stream_judge_conflicts(
            events_iter=events,
            focus_entities=(focus or None),
            rules_md_path=rules_md_path,
            lora_adapter_dir=lora_adapter_dir,
            use_vllm=use_vllm,
            batch_size=batch_size,
            simple_output=simple_output,
        ):
            count += 1
            print(f"[#{count}] 事件: {ev}")
            print(out)
            print("-" * 60)
        print("=== Stream Judge End ===")
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
        with open(rules_md_path, 'r', encoding='utf-8') as f:
            md = f.read()
        # 粗略清洗：去图片/多余空白
        import re
        md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)
        md = "\n".join([ln.rstrip() for ln in md.splitlines() if ln.strip() != ""])  # 去空白行
        # 前缀提示：
        prefix = (
            "你是一名航保作业规则判定助手。基于以下规则文档进行严谨的合规性判断，"
            "当输入一段新事件文本时，需要结合当前知识图谱状态，回答是否与现有状态或规则冲突，并给出依据。\n\n"
            "【规则文档】\n"
        )
        return prefix + md

    def _kg_text_context(self, kg: Dataset_KG, focus_entities: Optional[List[str]] = None, max_edges: int = 200) -> str:
        """从 Neo4j 组织一段可读的上下文文本，供大模型检索。

        - 若给定 focus_entities，则优先输出这些实体的一阶邻居；否则输出全局少量边。
        """
        lines: List[str] = []
        try:
            if focus_entities:
                for ent in focus_entities:
                    nb = kg.neighbors(ent)
                    if nb.get('out'):
                        for s, p, o in nb['out'][: max_edges // 2]:
                            lines.append(f"{s} -[{p}]-> {o}")
                    if nb.get('in'):
                        for s, p, o in nb['in'][: max_edges // 2]:
                            lines.append(f"{s} -[{p}]-> {o}")
            else:
                # 回退：扫描部分关系（通过导出 PNG 逻辑里的 allow_rels）
                snap = kg.graph_snapshot()
                lines.append(f"[SNAPSHOT] nodes={snap.get('nodes_count',0)} edges={snap.get('edges_count',0)}")
        except Exception as _e:
            pass
        ctx = "\n".join(lines[:max_edges])
        if not ctx:
            ctx = "(当前图为空或仅有固定节点)"
        return "【KG状态】\n" + ctx

    def _format_conflict_prompt(self, event_text: str, rules_text: str, kg_text: str) -> str:
        """将规则文本、KG上下文与事件文本格式化为单轮对话提示。"""
        instruction = (
            "任务：判断以下事件是否与当前状态或规则冲突。\n"
            "输出格式：先给出结论（合规/冲突），再给出1-3条依据，最后给出可操作建议。\n"
        )
        parts = [rules_text, kg_text, "【事件】\n" + event_text, instruction]
        return "\n\n".join([p for p in parts if p])

    def _format_conflict_prompt_with_mode(self, event_text: str, rules_text: str, kg_text: str, *, simple: bool = False) -> str:
        """根据 simple 模式切换输出要求：
        - simple=False：结论+依据+建议
        - simple=True：仅输出“合规”或“冲突”二字之一
        """
        if simple:
            instruction = (
                "任务：判断以下事件是否与当前状态或规则冲突。\n"
                "请仅输出一个词：‘合规’ 或 ‘冲突’。不需要解释或建议。\n"
            )
        else:
            instruction = (
                "任务：判断以下事件是否与当前状态或规则冲突。\n"
                "输出格式：先给出结论（合规/冲突），再给出1-3条依据，最后给出可操作建议。\n"
            )
        parts = [rules_text, kg_text, "【事件】\n" + event_text, instruction]
        return "\n\n".join([p for p in parts if p])

    def train_rules_lora(self,
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
              log_steps: int = 1) -> dict:
        """使用 Transformers+PEFT(Q-LoRA) 对 Qwen3-4B 进行指令微调。

        说明：vLLM 专注推理，不提供训练能力。这里采用 PEFT 进行 LoRA 微调，产出的适配器可在 vLLM 推理时加载。

        训练数据格式（JSONL 每行一个样本）：
        {"instruction": str, "input": str, "output": str}

        - 若未提供 train_jsonl，将基于 rules_md_path 生成一个极简自监督样本集（总结/提炼任务），
          仅用于跑通流程；建议后续提供高质量标注数据以提升冲突判定能力。
        """
        if transformers is None or peft is None:
            raise RuntimeError("需要安装 transformers 与 peft 方可训练。请先 pip install transformers peft accelerate datasets bitsandbytes")

        from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
        from transformers import Trainer, TrainingArguments, TrainerCallback  # type: ignore
        from peft import LoraConfig, get_peft_model, TaskType  # type: ignore
        import torch as _torch  # type: ignore

        model_dir = self.base_model_dir
        out_dir = output_dir or self.lora_out_dir
        os.makedirs(out_dir, exist_ok=True)

        # 构造训练数据集（模块化迁移到 data_loader）
        samples = []
        if train_jsonl and os.path.isfile(train_jsonl):
            samples = load_instruction_jsonl(train_jsonl)
            # 兼容性兜底：若给定的 JSONL 不是 instruction/input/output 结构（例如项目中用于抽取的 train_for_model.jsonl），
            # 则回退到基于规则文档自动构造一批占位样本，避免出现空数据集导致的 num_samples=0 错误。
            if len(samples) == 0 and rules_md_path:
                print(f"[TRAIN] 检测到 {os.path.basename(train_jsonl)} 非指令微调格式(或为空)，改用规则文档生成占位样本。")
                samples = build_rules_sft_samples_from_md(rules_md_path, max_samples=50, chunk_chars=2000)
        if len(samples) == 0:
            if not rules_md_path:
                raise RuntimeError("未提供 instruction/input/output 训练数据，且未提供规则文档用于自动构造样本。请传入 --rules_md_path 或提供符合SFT格式的 JSONL。")
            samples = build_rules_sft_samples_from_md(rules_md_path, max_samples=50, chunk_chars=2000)
        print(f"[TRAIN] 使用训练样本数: {len(samples)}")

        # 可选：动态拼接 KG 上下文（基于样本中的事件文本自动检索实体邻接）
        if augment_train_with_kg:
            kg = None
            try:
                kg = Dataset_KG(
                    self.root, load_data=False,
                    neo4j_uri=self.neo4j_uri,
                    neo4j_user=self.neo4j_user,
                    neo4j_password=self.neo4j_password,
                    neo4j_database=self.neo4j_database,
                )
            except Exception as _e:
                print("[TRAIN] 动态KG拼接不可用（连接失败），将跳过：", _e)

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
                            if t.startswith("飞机") or t.startswith("停机位") or t.startswith("跑道"):
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

            if kg is not None:
                rules_text = self.build_rules_prompt(rules_md_path) if rules_md_path else ""
                aug: List[dict] = []
                for ex in samples:
                    inp0 = ex.get("input", "")
                    ev = _extract_event_text(inp0)
                    focus = _auto_focus_entities(ev)
                    kg_text = self._kg_text_context(kg, focus_entities=(focus or None), max_edges=200)
                    new_input = "\n\n".join([p for p in (
                        rules_text,
                        kg_text,
                        ("【事件】\n" + ev if ev else inp0),
                        "输出格式：结论+依据+建议",
                    ) if p])
                    ex2 = dict(ex)
                    ex2["input"] = new_input
                    aug.append(ex2)
                samples = aug

        # 简单的数据集包装：拼接成 prompt -> target 的监督微调
        def build_text(ex: dict) -> Tuple[str, str]:
            ins, inp, out = ex.get("instruction", ""), ex.get("input", ""), ex.get("output", "")
            prompt = f"指令：{ins}\n输入：{inp}\n回答："
            return prompt, out

        # tokenizer / model（开启4bit量化以节省显存）
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
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
                __os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                # 新变量名（旧名已弃用），提前兼容，减少告警
                __os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
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
            torch_dtype=_torch_dtype,
        )

        lora_cfg = LoraConfig(
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
            _cfg = getattr(model, 'config', None)
            if _cfg is not None and hasattr(_cfg, 'use_cache'):
                setattr(_cfg, 'use_cache', False)
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
                print(f"[TRAIN] 使用GPU: {gname} | alloc={mem_alloc:.2f}GB reserved={mem_resv:.2f}GB")
            else:
                print("[TRAIN] 使用CPU训练（建议在有GPU时设置 --device cuda）")
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
            ds_train = _MemDS(_encoded)
        else:
            ds_train = _datasets.Dataset.from_list(samples).map(_tok_map, remove_columns=list(samples[0].keys()))

        args = TrainingArguments(
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
                print(f"[STEP] {gs}/{ms} | loss={_fmt_loss(loss)} | lr={_fmt_lr(lr)} | {steps_per_sec:.2f} steps/s | ETA {fmt_sec(eta_sec)}{mem_info}")

        trainer = Trainer(model=model, args=args, train_dataset=ds_train, callbacks=[_ProgressCB(log_steps)])
        trainer.train()
        # 保存 LoRA 适配器
        adapter_dir = os.path.join(out_dir, "adapter")
        os.makedirs(adapter_dir, exist_ok=True)
        model.save_pretrained(adapter_dir)
        if hasattr(tok, 'save_pretrained'):
            tok.save_pretrained(out_dir)

        print(f"[TRAIN] LoRA 适配器已保存: {adapter_dir}")
        return {"adapter_dir": adapter_dir, "samples": len(samples)}

    def generate_with_vllm(self, prompts: List[str], lora_adapter_dir: Optional[str] = None,
                            max_tokens: int = 256, temperature: float = 0.2) -> List[str]:
        """使用 vLLM 进行批量推理。若提供 LoRA 适配器，尝试加载。"""
        if vllm is None:
            raise RuntimeError("需要安装 vllm 才能进行推理：pip install vllm")
        from vllm import LLM, SamplingParams  # type: ignore

        llm = LLM(model=self.base_model_dir, trust_remote_code=True)
        # 尝试加载 LoRA（不同 vLLM 版本 API 差异较大，这里做最小兼容处理）
        if lora_adapter_dir:
            try:
                loader = getattr(llm, 'load_lora_modules', None)
                if callable(loader):
                    loader({"default": lora_adapter_dir})
                    set_active = getattr(llm, 'set_active_lora', None)
                    if callable(set_active):
                        set_active("default")
                else:
                    print("[vLLM] 当前版本不支持在 Python API 中动态加载 LoRA，忽略适配器。")
            except Exception as e:  # noqa: BLE001
                print("[vLLM] 加载 LoRA 失败，忽略：", e)

        sam = SamplingParams(max_tokens=max_tokens, temperature=temperature, stop=["\n\n【"],
                              n=1)
        outs = llm.generate(prompts, sam)
        texts = []
        for out in outs:
            if out and out.outputs:
                texts.append(out.outputs[0].text)
            else:
                texts.append("")
        return texts

    # 删除单条 judge 接口，统一通过 stream_judge_conflicts/Exp_main.run 路由

    def stream_judge_conflicts(self,
                               events_iter,
                               focus_entities: Optional[List[str]] = None,
                               rules_md_path: Optional[str] = None,
                               lora_adapter_dir: Optional[str] = None,
                               use_vllm: bool = True,
                               batch_size: int = 4,
                               simple_output: bool = False):
        """对事件流进行逐条/小批量判冲突（流式在推理时执行，而非训练时）。

        - events_iter: 可迭代的事件文本序列（例如生成器/列表）。
        - 每批最多 batch_size 条，生成对应输出后立即 yield，适合持续到来的实时日志。
        - 返回迭代器：每次 yield (event_text, output_str)
        """
        # 规则提示一次构建，复用
        rules_text = self.build_rules_prompt(rules_md_path) if rules_md_path else ""

        kg = None
        if not self.skip_kg:
            try:
                kg = Dataset_KG(
                    self.root, load_data=False,
                    neo4j_uri=self.neo4j_uri,
                    neo4j_user=self.neo4j_user,
                    neo4j_password=self.neo4j_password,
                    neo4j_database=self.neo4j_database,
                )
            except Exception:
                kg = None

        batch_events: List[str] = []
        batch_prompts: List[str] = []
        for ev in events_iter:
            if not (isinstance(ev, str) and ev.strip()):
                continue
            ev = ev.strip()
            # 针对每条事件，使用“当前”KG状态生成 prompt（先判后更）
            kg_text = self._kg_text_context(kg, focus_entities) if (kg is not None) else "【KG状态】\n(离线模式，未加载图谱)"
            prompt = self._format_conflict_prompt_with_mode(ev, rules_text, kg_text, simple=simple_output)
            batch_events.append(ev)
            batch_prompts.append(prompt)

            if len(batch_events) >= max(1, int(batch_size)):
                # 推理
                if use_vllm:
                    outs = self.generate_with_vllm(batch_prompts, lora_adapter_dir=lora_adapter_dir)
                else:
                    if transformers is None:
                        raise RuntimeError("需要安装 transformers 或 vllm 进行推理")
                    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
                    tok = AutoTokenizer.from_pretrained(self.base_model_dir, trust_remote_code=True, use_fast=False)
                    if tok.pad_token is None:
                        tok.pad_token = tok.eos_token
                    mdl = AutoModelForCausalLM.from_pretrained(self.base_model_dir, trust_remote_code=True)
                    try:
                        import torch  # type: ignore
                        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
                    except Exception:
                        device_str = 'cpu'
                    try:
                        if device_str == 'cuda':
                            mdl = getattr(mdl.__class__, 'cuda')(mdl)
                        else:
                            mdl = getattr(mdl.__class__, 'cpu')(mdl)
                    except Exception:
                        pass
                    # 加载 LoRA 适配器（若提供）
                    if lora_adapter_dir:
                        try:
                            from peft import PeftModel  # type: ignore
                            mdl = PeftModel.from_pretrained(mdl, lora_adapter_dir)
                            try:
                                if device_str == 'cuda' and hasattr(mdl, 'cuda'):
                                    mdl = mdl.cuda()
                                elif hasattr(mdl, 'cpu'):
                                    mdl = mdl.cpu()
                            except Exception:
                                pass
                        except Exception as _e:
                            print("[infer] 加载 LoRA 适配器失败，改用基座模型：", _e)
                    outs = []
                    for p in batch_prompts:
                        ids = tok(p, return_tensors='pt')
                        try:
                            moved = {}
                            for k, v in ids.items():
                                if device_str == 'cuda':
                                    moved[k] = getattr(v.__class__, 'cuda')(v)
                                else:
                                    moved[k] = getattr(v.__class__, 'cpu')(v)
                            ids = moved
                        except Exception:
                            pass
                        gen = mdl.generate(**ids, max_new_tokens=256, do_sample=False)
                        out = tok.decode(gen[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)
                        outs.append(out)
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
                # 逐条回传，并在判定后更新 KG（流式推进）
                if kg is not None:
                    for e in batch_events:
                        try:
                            kg.extract_and_update(e)
                        except Exception:
                            pass
                for e, o in zip(batch_events, outs):
                    yield (e, o)
                batch_events, batch_prompts = [], []

        # 处理尾批
        if batch_events:
            if use_vllm:
                outs = self.generate_with_vllm(batch_prompts, lora_adapter_dir=lora_adapter_dir)
            else:
                if transformers is None:
                    raise RuntimeError("需要安装 transformers 或 vllm 进行推理")
                from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
                tok = AutoTokenizer.from_pretrained(self.base_model_dir, trust_remote_code=True, use_fast=False)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                mdl = AutoModelForCausalLM.from_pretrained(self.base_model_dir, trust_remote_code=True)
                try:
                    import torch  # type: ignore
                    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
                except Exception:
                    device_str = 'cpu'
                try:
                    if device_str == 'cuda':
                        mdl = getattr(mdl.__class__, 'cuda')(mdl)
                    else:
                        mdl = getattr(mdl.__class__, 'cpu')(mdl)
                except Exception:
                    pass
                if lora_adapter_dir:
                    try:
                        from peft import PeftModel  # type: ignore
                        mdl = PeftModel.from_pretrained(mdl, lora_adapter_dir)
                        try:
                            if device_str == 'cuda':
                                mdl = getattr(mdl.__class__, 'cuda')(mdl)
                            else:
                                mdl = getattr(mdl.__class__, 'cpu')(mdl)
                        except Exception:
                            pass
                    except Exception as _e:
                        print("[infer] 加载 LoRA 适配器失败，改用基座模型：", _e)
                outs = []
                for p in batch_prompts:
                    ids = tok(p, return_tensors='pt')
                    try:
                        moved = {}
                        for k, v in ids.items():
                            if device_str == 'cuda':
                                moved[k] = getattr(v.__class__, 'cuda')(v)
                            else:
                                moved[k] = getattr(v.__class__, 'cpu')(v)
                        ids = moved
                    except Exception:
                        pass
                    gen = mdl.generate(**ids, max_new_tokens=256, do_sample=False)
                    out = tok.decode(gen[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)
                    outs.append(out)
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
            if kg is not None:
                for e in batch_events:
                    try:
                        kg.extract_and_update(e)
                    except Exception:
                        pass
            for e, o in zip(batch_events, outs):
                yield (e, o)
    
