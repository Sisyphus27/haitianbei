'''
Author: zy
Date: 2025-11-04 14:09:37
LastEditTime: 2025-11-04 18:05:00
LastEditors: zy
Description: 通用工具函数（训练样本构建等）
FilePath: /haitianbei/utils/utils.py

'''

from __future__ import annotations

import json
import os
import re
from typing import Iterable


# ------------------------------------------------------------
# 读取/写入 JSONL
# ------------------------------------------------------------
def read_jsonl(path: str) -> list[dict]:
	arr: list[dict] = []
	with open(path, 'r', encoding='utf-8') as f:
		for ln in f:
			ln = ln.strip()
			if not ln:
				continue
			try:
				obj = json.loads(ln)
			except Exception:
				continue
			if isinstance(obj, dict):
				arr.append(obj)
	return arr


def save_jsonl(rows: Iterable[dict], out_path: str):
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	with open(out_path, 'w', encoding='utf-8') as f:
		for obj in rows:
			f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ------------------------------------------------------------
# 资源抽取与规范化（与 data_loader 中的类型判定保持一致的启发式）
# ------------------------------------------------------------
_DEVICE_PAT = re.compile(r"牵引车|加氧车|加氮车|空气终端|氧气终端|氮气终端|压缩空气终端|清洗装置|供氧站|供氮站|供气站|加油站|液压站|供电站")


def _canon_gate(name: str) -> str | None:
	name = str(name).strip()
	m = re.fullmatch(r"(\d+)号(?:停机位)?", name)
	if m:
		return f"停机位{int(m.group(1))}"
	if name.startswith("停机位"):
		return name
	return None


def _canon_runway(name: str) -> str | None:
	s = str(name).strip()
	if s.upper() == 'Z' or s in {"着陆跑道", "着陆跑道Z", "着陆跑道z"}:
		return "跑道Z"
	if re.fullmatch(r"\d+", s):
		return f"跑道{s}"
	if s.startswith("跑道"):
		return s
	return None


def _is_device(name: str) -> bool:
	return _DEVICE_PAT.search(str(name)) is not None


def detect_resources_from_triples(triples: Iterable[Iterable[str]]):
	"""从三元组列表中抽取涉及的“资源”（停机位/跑道/设备）。

	返回：去重后的资源名称列表（已尽量规范化）
	"""
	res: list[str] = []
	def add(x: str):
		nonlocal res
		x = str(x).strip()
		if not x:
			return
		if x not in res:
			res.append(x)

	for t in triples:
		if not isinstance(t, (list, tuple)) or len(t) != 3:
			continue
		s, p, o = t[0], t[1], t[2]
		# 设备
		if _is_device(s):
			add(str(s))
		if _is_device(o):
			add(str(o))
		# 停机位（常见出现在客体）
		g1 = _canon_gate(s)
		if g1:
			add(g1)
		g2 = _canon_gate(o)
		if g2:
			add(g2)
		# 跑道
		r1 = _canon_runway(s)
		if r1:
			add(r1)
		r2 = _canon_runway(o)
		if r2:
			add(r2)
	return res


def _format_availability(avails: dict[str, bool]) -> str:
	lines = ["【资源可用性】"]
	for k in sorted(avails.keys()):
		lines.append(f"- {k}：{'可用' if avails[k] else '不可用'}")
	return "\n".join(lines)


def _mk_positive_output() -> str:
	return (
		"结论：合规\n"
		"依据：\n- 文本涉及的资源均可用\n"
		"建议：按计划执行，并持续监控资源状态。"
	)


def _mk_negative_output(unavail: list[str]) -> str:
	reasons = [f"资源 {x} 当前不可用，与文本中的使用需求冲突" for x in unavail]
	return (
		"结论：冲突\n"
		"依据：\n- " + "\n- ".join(reasons[:3]) + "\n"
		"建议：更换或调度可用的替代资源，或调整作业计划。"
	)


# ------------------------------------------------------------
# 分解器（小模型）训练样本生成
# 目标：将“事件文本”分解为结构化 JSON（实体/适用规则/潜在冲突/备注）
# 与 exp.exp_main._format_decompose_prompt 的约定保持一致
# ------------------------------------------------------------

_AIRCRAFT_PAT = re.compile(r"飞机[0-9A-Za-z]+")
_TUG_PAT = re.compile(r"(\d+)号牵引车")


def _extract_decompose_entities(ev_text: str) -> list[str]:
	"""启发式抽取用于分解任务的关键实体：

	- 飞机：匹配诸如“飞机A001”
	- 跑道：规范化为 “跑道Z”/“跑道N”
	- 停机位：规范化为 “停机位N”
	- 牵引车：保留原文形态 “N号牵引车”

	返回：去重后的实体文本列表。
	"""
	ev = str(ev_text or "")
	ents: list[str] = []

	def add(x: str):
		x = (x or "").strip()
		if not x:
			return
		if x not in ents:
			ents.append(x)

	# 飞机
	for m in _AIRCRAFT_PAT.finditer(ev):
		add(m.group(0))

	# 牵引车
	for m in _TUG_PAT.finditer(ev):
		add(f"{int(m.group(1))}号牵引车")

	# 跑道/停机位（依赖已有规范化函数）
	# 简单切分获取可能的候选词（中文环境下，直接在整句上规范化也可）
	tokens = re.findall(r"[\w\u4e00-\u9fa5]+", ev)
	for tk in tokens:
		g = _canon_gate(tk)
		if g:
			add(g)
			continue
		r = _canon_runway(tk)
		if r:
			add(r)

	return ents


def _infer_applicable_rules(entities: list[str]) -> list[str]:
	"""根据实体类型推断适用的规则要点（仅作为训练弱标注的提示）。"""
	rs: list[str] = []
	has_runway = any(x.startswith("跑道") for x in entities)
	has_gate = any(x.startswith("停机位") for x in entities)
	has_tug = any(x.endswith("号牵引车") or x.endswith("牵引车") for x in entities)
	if has_runway:
		rs.append("跑道同一时刻仅允许一架飞机占用")
	if has_gate:
		rs.append("停机位一次仅能停放一架飞机")
	if has_tug:
		rs.append("同一时刻同一牵引车仅服务一架飞机")
	return rs


def _infer_potential_conflicts(entities: list[str]) -> list[str]:
	"""基于实体给出通用的潜在冲突提示（弱标注，不依赖实时KG）。

	为了训练分解器学习结构与关注点，这里只给出“可能”冲突点模板，
	实际是否冲突在主判定流程中由 KG/规则再判断。
	"""
	items: list[str] = []
	for x in entities:
		if x.startswith("跑道"):
			items.append(f"{x} 可能被其他飞机占用")
		elif x.startswith("停机位"):
			items.append(f"{x} 可能已有关联飞机")
		elif x.endswith("牵引车"):
			items.append(f"{x} 可能正在服务其他飞机")
	# 控制数量，保持 1-3 条
	return items[:3]


def build_decomposition_samples_from_events(
	events: Iterable[str],
	*,
	instruction: str | None = None,
) -> list[dict]:
	"""从事件文本构造“分解器（小模型）”的 SFT 样本。

	输出样本结构：{"instruction": str, "input": str, "output": str, "meta": {...}}

	- instruction：与 exp_main._format_decompose_prompt 中一致的任务描述，要求仅输出 JSON。
	- input：包含“【事件】\n<原文>”。（如需在上层拼接规则/KG，可在组装时附加）
	- output：启发式生成的 JSON 字符串，形如：
		{
		  "entities": [...],
		  "applicable_rules": [...],
		  "potential_conflicts": [...],
		  "notes": "仅围绕跑道/停机位/牵引三类冲突分解"
		}
	"""
	ins = instruction or (
		"请将以下判冲突任务分解为结构化步骤，并只输出一个JSON对象：\n"
		"- 第一步：提取关键实体（飞机/跑道/停机位/牵引车）。\n"
		"- 第二步：匹配可能相关的规则要点（仅围绕‘跑道互斥’、‘停机位互斥’、‘牵引唯一’）。\n"
		"- 第三步：基于当前KG状态列出潜在冲突点（用简短文本描述即可）。\n"
		"不要输出多余解释或格式。"
	)

	out_rows: list[dict] = []
	for idx, ev in enumerate(events):
		ev_text = str(ev or "").strip()
		if not ev_text:
			continue
		ents = _extract_decompose_entities(ev_text)
		rules = _infer_applicable_rules(ents)
		pot = _infer_potential_conflicts(ents)
		out_json_obj = {
			"entities": ents,
			"applicable_rules": rules,
			"potential_conflicts": pot,
			"notes": "仅围绕跑道/停机位/牵引三类冲突分解",
		}
		out_rows.append({
			"instruction": ins,
			"input": "【事件】\n" + ev_text,
			"output": json.dumps(out_json_obj, ensure_ascii=False, separators=(",", ":")),
			"meta": {"label": "decompose", "id": idx},
		})

	return out_rows


def build_samples_from_train_triples(
	triples_jsonl_path: str,
	*,
	include_positive: bool = True,
	negative_mode: str = "per_resource",  # per_resource | all_resources
	max_negatives_per_event: int | None = None,
	instruction: str | None = None,
) -> list[dict]:
	"""基于 data_provider/train_triples.jsonl 构建 SFT 训练样本。

	- 正样本：默认提供（资源全部可用）。
	- 负样本：将该条文本涉及的资源标记为不可用，引导模型学习“文本 vs 资源可用性”的冲突检测。

	参数：
	- negative_mode='per_resource'：对每个资源各生成一条负样本（逐一置为不可用）。
	- negative_mode='all_resources'：将该事件涉及的所有资源同时置为不可用，生成一条综合负样本。
	- max_negatives_per_event：可选，限制每条事件生成的负样本数量（仅对 per_resource 模式生效）。
	- instruction：自定义任务描述；默认提供通用版本。
	"""
	rows = read_jsonl(triples_jsonl_path)
	samples: list[dict] = []
	instr = instruction or (
		"任务：判断以下事件与给定的资源可用性是否冲突。请先给出结论（合规/冲突），再给出1-3条依据，最后给出可操作建议。"
	)

	for obj in rows:
		text = str(obj.get('text', '')).strip()
		triples = obj.get('triples') or []
		if not text or not isinstance(triples, (list, tuple)):
			continue
		resources = detect_resources_from_triples(triples)  # 已规范化的资源名

		# 正样本：资源全部可用
		if include_positive:
			pos_av = {r: True for r in resources}
			input_text = "\n\n".join([
				f"【事件】\n{text}",
				_format_availability(pos_av) if pos_av else "【资源可用性】\n- (未涉及资源)"
			])
			samples.append({
				"instruction": instr,
				"input": input_text,
				"output": _mk_positive_output(),
				"meta": {"label": "positive", "id": obj.get("id")},
			})

		# 负样本：资源不可用
		if resources:
			if negative_mode == "all_resources":
				neg_av = {r: False for r in resources}
				input_text = "\n\n".join([
					f"【事件】\n{text}",
					_format_availability(neg_av)
				])
				samples.append({
					"instruction": instr,
					"input": input_text,
					"output": _mk_negative_output(resources),
					"meta": {"label": "negative", "mode": "all_resources", "id": obj.get("id")},
				})
			else:
				count = 0
				for r in resources:
					if max_negatives_per_event is not None and count >= max_negatives_per_event:
						break
					neg_av = {x: True for x in resources}
					neg_av[r] = False
					input_text = "\n\n".join([
						f"【事件】\n{text}",
						_format_availability(neg_av)
					])
					samples.append({
						"instruction": instr,
						"input": input_text,
						"output": _mk_negative_output([r]),
						"meta": {"label": "negative", "mode": "per_resource", "id": obj.get("id"), "resource": r},
					})
					count += 1

	return samples


__all__ = [
	"read_jsonl",
	"save_jsonl",
	"detect_resources_from_triples",
	"build_samples_from_train_triples",
	"build_decomposition_samples_from_events",
]
