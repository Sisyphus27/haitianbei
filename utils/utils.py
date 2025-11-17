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
import logging
import os
import re
from typing import Iterable

_logger = logging.getLogger(__name__)


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


# ------------------------------------------------------------
# 任务二：冲突检测与重调度相关工具
# ------------------------------------------------------------

# 冲突关键词模式
_CONFLICT_KEYWORDS = [
	"冲突", "不可用", "故障", "封锁", "暂停", "中断", "异常", 
	"损坏", "维修", "修理", "故障不可用", "不可用时间", "必须停在.*修理"
]

_CONFLICT_PATTERN = re.compile("|".join(_CONFLICT_KEYWORDS))


def detect_potential_conflict(event_text: str) -> bool:
	"""初步检测事件文本中是否包含潜在冲突关键词。
	
	参数:
		event_text: 事件文本字符串
		
	返回:
		bool: True表示检测到潜在冲突，需要进一步调用大模型判定；False表示未检测到明显冲突关键词
	"""
	if not isinstance(event_text, str) or not event_text.strip():
		return False
	
	# 检测是否包含冲突关键词
	return bool(_CONFLICT_PATTERN.search(event_text))


def generate_snapshot_from_kg(kg_service, current_time_min: float) -> tuple[dict, dict, dict]:
	"""从KG全局状态提取信息，生成符合readme.md快照格式的JSON配置。
	
	参数:
		kg_service: KGServiceLocal实例
		current_time_min: 当前时间（分钟）
		
	返回:
		tuple[dict, dict, dict]: (快照配置字典, 飞机ID映射字典, 完整映射字典)
			- 快照配置字典：包含time, planes, stand_occupancy, blocked_stands, disturbance_events等字段
			- 飞机ID映射字典：{数字ID: 原始名称}，例如 {0: "飞机A001", 1: "飞机A002"}
			- 完整映射字典：包含所有映射关系，结构如下：
				{
					"aircraft": {plane_id: name},  # 飞机ID到名称的映射
					"devices": {normalized: original},  # 规范化设备名称到原始KG名称的映射（用于写回KG）
					"resources": {normalized: original}  # 规范化资源名称到原始KG名称的映射（用于写回KG）
				}
			
	注意:
		- 设备名称可能从中文（如"移动加氧车"）映射到MR格式（如"MR02"）
		- 资源名称通常已经是R格式（如"R005"），但为了统一性仍需记录映射
		- 映射字典用于后续将强化学习结果写回KG时，能够正确找到原始KG节点名称
	"""
	if not kg_service or not hasattr(kg_service, 'kg'):
		return ({
			"time": current_time_min,
			"planes": [],
			"stand_occupancy": {},
			"blocked_stands": [],
			"arrival_plan": {},
			"devices": {"fixed": {}, "mobile": {}},
			"disturbance_events": []
		}, {}, {
			"aircraft": {},
			"devices": {},
			"resources": {}
		})
	
	kg = kg_service.kg
	if not hasattr(kg, 'driver') or not hasattr(kg, 'neo4j_database'):
		return ({
			"time": current_time_min,
			"planes": [],
			"stand_occupancy": {},
			"blocked_stands": [],
			"arrival_plan": {},
			"devices": {"fixed": {}, "mobile": {}},
			"disturbance_events": []
		}, {}, {
			"aircraft": {},
			"devices": {},
			"resources": {}
		})
	
	snapshot = {
		"time": float(current_time_min),
		"planes": [],
		"stand_occupancy": {},
		"blocked_stands": [],
		"arrival_plan": {},
		"devices": {"fixed": {}, "mobile": {}},
		"disturbance_events": []  # 初始化扰动事件列表
	}
	
	aircraft_map = {}  # 飞机名称 -> 数字ID映射（在try块外初始化，以便在异常时也能返回）
	device_name_map = {}  # 规范化设备名称 -> 原始KG名称映射（在try块外初始化，以便在异常时也能返回）
	resource_name_map = {}  # 规范化资源名称 -> 原始KG名称映射（在try块外初始化，以便在异常时也能返回）
	
	try:
		with kg.driver.session(database=kg.neo4j_database) as sess:
			# 1. 提取飞机状态
			# 查询所有飞机及其当前状态
			aircraft_query = """
			MATCH (a:Aircraft)
			OPTIONAL MATCH (a)-[:HAS_CURRENT_GATE]->(s:Stand)
			OPTIONAL MATCH (a)-[:CURRENT_JOB]->(j:Job)
			OPTIONAL MATCH (a)-[:PERFORMS_JOB]->(pj:Job)
			RETURN a.name AS name, 
			       s.name AS current_site,
			       collect(DISTINCT j.name) AS current_jobs,
			       collect(DISTINCT pj.name) AS finished_jobs,
			       a.isDamaged AS is_damaged
			ORDER BY a.name
			"""
			
			plane_id_counter = 0
			
			for record in sess.run(aircraft_query):
				ac_name = record.get("name", "")
				if not ac_name:
					continue
				
				current_site = record.get("current_site")
				current_jobs = [j for j in (record.get("current_jobs") or []) if j]
				finished_jobs = [j for j in (record.get("finished_jobs") or []) if j]
				is_damaged = bool(record.get("is_damaged", False))
				
				# 提取飞机编号（如"飞机A001" -> 0, "飞机A002" -> 1）
				aircraft_map[ac_name] = plane_id_counter
				
				# 构建active_job字符串（支持并行作业，如"ZY04+ZY05"）
				active_job = "+".join(current_jobs) if current_jobs else None
				
				# 确定状态：根据当前作业推断
				status = "IDLE"
				if current_jobs:
					if any("ZY_Z" in j or "ZY_M" in j for j in current_jobs):
						status = "MOVING"
					elif any("ZY_T" in j for j in current_jobs):
						status = "MOVING"
					else:
						status = "PROCESSING"
				
				# 提取停机位编号（如"停机位14" -> 14）
				current_site_id = None
				if current_site:
					m = re.search(r"停机位(\d+)", str(current_site))
					if m:
						current_site_id = int(m.group(1))
				
				# 查询暂停的作业（如果有的话）
				# 注意：当前KG可能不直接存储暂停状态，这里先尝试查询
				# 如果找不到暂停信息，则保持空列表，但格式要正确
				paused_jobs_list = []
				
				# 严格按照 readme.md 示例格式构建 plane_obj
				# 只包含示例中出现的字段：plane_id, status, current_site_id, active_job, active_remaining, paused_jobs
				plane_obj = {
					"plane_id": plane_id_counter,
					"status": status
				}
				
				# 仅在存在时添加可选字段
				if current_site_id is not None:
					plane_obj["current_site_id"] = current_site_id
				
				if active_job:
					plane_obj["active_job"] = active_job
					# 为每个作业设置剩余时间
					active_remaining = {}
					for job in current_jobs:
						active_remaining[job] = 5.0  # 默认5分钟
					plane_obj["active_remaining"] = active_remaining
				
				# paused_jobs 仅在非空时添加（根据 readme 示例，空列表也可以包含）
				if paused_jobs_list:
					plane_obj["paused_jobs"] = paused_jobs_list
				
				snapshot["planes"].append(plane_obj)
				plane_id_counter += 1
			
			# 2. 提取停机位占用情况
			stand_query = """
			MATCH (s:Stand)
			OPTIONAL MATCH (a:Aircraft)-[:HAS_CURRENT_GATE]->(s)
			RETURN s.name AS name, 
			       s.isOccupied AS is_occupied,
			       s.isDamaged AS is_damaged,
			       collect(DISTINCT a.name) AS aircraft_names
			ORDER BY s.name
			"""
			
			for record in sess.run(stand_query):
				stand_name = record.get("name", "")
				if not stand_name:
					continue
				
				m = re.search(r"停机位(\d+)", stand_name)
				if not m:
					continue
				
				stand_num = m.group(1)
				is_occupied = bool(record.get("is_occupied", False))
				is_damaged = bool(record.get("is_damaged", False))
				aircraft_names = [a for a in (record.get("aircraft_names") or []) if a]
				
				# stand_occupancy: 占用则记录飞机ID，否则为null
				if is_occupied and aircraft_names:
					# 找到对应的飞机ID
					ac_id = aircraft_map.get(aircraft_names[0])
					if ac_id is not None:
						snapshot["stand_occupancy"][stand_num] = ac_id
					else:
						snapshot["stand_occupancy"][stand_num] = None
				else:
					snapshot["stand_occupancy"][stand_num] = None
				
				# blocked_stands: 损坏的停机位
				if is_damaged:
					try:
						stand_id = int(stand_num)
						if stand_id not in snapshot["blocked_stands"]:
							snapshot["blocked_stands"].append(stand_id)
					except (ValueError, TypeError):
						pass
			
			# 3. 提取设备状态（固定设备）
			# 注意：FixedDevice节点同时有Device和FixedDevice标签
			# 简化查询：只通过USING_DEVICE关系获取设备使用情况，避免ASSIGNED_TO_JOB_INSTANCE关系不存在的警告
			fixed_device_query = """
			MATCH (d:Device:FixedDevice)
			OPTIONAL MATCH (a:Aircraft)-[:USING_DEVICE]->(d)
			RETURN d.name AS name,
			       d.coverage AS coverage,
			       d.isOccupied AS is_occupied,
			       d.isReserved AS is_reserved,
			       collect(DISTINCT a.name) AS aircraft_names
			ORDER BY d.name
			"""
			
			for record in sess.run(fixed_device_query):
				dev_name = record.get("name", "")
				if not dev_name:
					continue
				
				coverage = record.get("coverage") or []
				# capacity可以从coverage（覆盖的停机位数量）推断，或使用默认值
				# 对于固定设备，通常capacity等于覆盖的停机位数量，但这里简化处理
				capacity = len(coverage) if isinstance(coverage, list) else 1
				if capacity == 0:
					capacity = 1  # 默认至少为1
				
				aircraft_names = [a for a in (record.get("aircraft_names") or []) if a]
				is_occupied = bool(record.get("is_occupied", False))
				is_reserved = bool(record.get("is_reserved", False))
				
				# 转换为设备ID列表（从飞机名称映射到ID）
				in_use = []
				for ac_name in aircraft_names:
					ac_id = aircraft_map.get(ac_name)
					if ac_id is not None:
						in_use.append(ac_id)
				
				# 如果设备被占用或保留，或者有飞机在使用，则记录
				if in_use or is_occupied or is_reserved or capacity > 0:
					snapshot["devices"]["fixed"][dev_name] = {
						"in_use": in_use,
						"capacity": int(capacity)
					}
			
			# 4. 提取设备状态（移动设备）
			# 注意：MobileDevice节点可能同时有Device和MobileDevice标签
			mobile_device_query = """
			MATCH (d:Device:MobileDevice)
			OPTIONAL MATCH (d)-[:INITIAL_AT]->(s:Stand)
			OPTIONAL MATCH (a:Aircraft)-[:USING_DEVICE]->(d)
			RETURN d.name AS name,
			       s.name AS loc_stand,
			       collect(DISTINCT a.name) AS aircraft_names
			ORDER BY d.name
			"""
			
			for record in sess.run(mobile_device_query):
				dev_name = record.get("name", "")
				if not dev_name:
					continue
				
				loc_stand = record.get("loc_stand", "")
				aircraft_names = [a for a in (record.get("aircraft_names") or []) if a]
				
				# 提取停机位编号
				loc_stand_id = None
				if loc_stand:
					m = re.search(r"停机位(\d+)", str(loc_stand))
					if m:
						loc_stand_id = int(m.group(1))
				
				# 确定locked_by（正在使用的飞机ID）
				locked_by = None
				if aircraft_names:
					ac_id = aircraft_map.get(aircraft_names[0])
					if ac_id is not None:
						locked_by = ac_id
				
				# 构建移动设备状态字典，只有当 loc_stand_id 不为 None 时才包含该字段
				mobile_device_state = {
					"busy_until_min": current_time_min + 10.0 if locked_by else current_time_min,  # 简化：默认10分钟后空闲
					"locked_by": locked_by,
					"speed_m_s": 3.0  # 默认速度
				}
				# 只有当 loc_stand_id 不为 None 时才添加该字段
				if loc_stand_id is not None:
					mobile_device_state["loc_stand"] = loc_stand_id
				
				snapshot["devices"]["mobile"][dev_name] = mobile_device_state
			
			# 5. 提取到达计划（从着陆事件推断）
			# 查询尚未着陆的飞机（没有 HAS_CURRENT_GATE 关系但有计划着陆时间）
			# arrival_plan 格式: {str(plane_id): arrival_time_min}（字符串键）
			arrival_plan_query = """
			MATCH (a:Aircraft)
			WHERE NOT (a)-[:HAS_CURRENT_GATE]->(:Stand)
			OPTIONAL MATCH (a)-[:HAS_TIME]->(t:Time)
			OPTIONAL MATCH (a)-[:USES_RUNWAY]->(r:Runway)
			WHERE r.name = "跑道Z"
			RETURN a.name AS name, t.name AS time_str
			ORDER BY a.name
			"""
			
			for record in sess.run(arrival_plan_query):
				ac_name = record.get("name", "")
				time_str = record.get("time_str", "")
				if not ac_name:
					continue
				
				# 获取飞机ID
				ac_id = aircraft_map.get(ac_name)
				if ac_id is None:
					continue
				
				# 尝试从时间字符串解析时间（分钟）
				# 如果无法解析，则跳过
				arrival_time = None
				if time_str:
					# 尝试解析时间字符串（格式可能是 "485.0" 或类似）
					try:
						arrival_time = float(time_str)
					except (ValueError, TypeError):
						# 如果无法解析为浮点数，尝试其他格式
						pass
				
				# 如果找到了到达时间，添加到 arrival_plan
				if arrival_time is not None:
					snapshot["arrival_plan"][str(ac_id)] = float(arrival_time)
			
			# 6. 生成 disturbance_events 字段（从KG的isDamaged属性提取）
			# 查询所有损坏的节点（停机位、设备、资源），提取failureTime和failureDuration
			disturbance_events = []
			event_id_counter = 0
			
			# 映射字典已在try块外初始化，这里只需确保它们已存在
			
			# 6.1 查询损坏的停机位（只查询在当前时间点仍然有效的，且未被修复的）
			# 注意：isDamaged = true 且 failureTime IS NOT NULL 表示停机位仍然损坏
			# 如果 isDamaged = false，说明已经被修复，不应该出现在 disturbance_events 中
			damaged_stands_query = """
			MATCH (s:Stand)
			WHERE s.isDamaged = true
			  AND s.failureTime IS NOT NULL
			  AND (s.failureDuration IS NULL OR s.failureTime + COALESCE(s.failureDuration, 60.0) >= $current_time)
			RETURN s.name AS name,
			       s.failureTime AS failure_time,
			       s.failureDuration AS failure_duration,
			       s.isDamaged AS is_damaged
			ORDER BY s.name
			"""
			
			damaged_stands_by_time = {}  # {(start, end): [stand_ids]}
			damaged_stands_found = []  # 用于调试日志
			
			for record in sess.run(damaged_stands_query, {"current_time": current_time_min}):
				stand_name = record.get("name", "")
				if not stand_name:
					continue
				
				# 提取停机位编号（如"停机位14" -> 14）
				m = re.search(r"停机位(\d+)", stand_name)
				if not m:
					continue
				
				stand_id = int(m.group(1))
				failure_time = record.get("failure_time")
				failure_duration = record.get("failure_duration")
				is_damaged = record.get("is_damaged", True)
				
				# 双重检查：确保 isDamaged 仍然是 true（防止在查询和生成快照之间被修复）
				if not is_damaged:
					_logger.debug(f"[SNAPSHOT] 停机位 {stand_name} 已被修复，跳过（isDamaged={is_damaged}）")
					continue
				
				# 确定开始和结束时间
				start_time = float(failure_time) if failure_time is not None else float(current_time_min)
				duration = float(failure_duration) if failure_duration is not None else 60.0
				end_time = start_time + duration
				
				# 验证时间有效性：结束时间应该 >= 当前时间
				if end_time < current_time_min:
					_logger.debug(f"[SNAPSHOT] 停机位 {stand_name} 故障已过期（结束时间 {end_time:.2f} < 当前时间 {current_time_min:.2f}），跳过")
					continue
				
				# 按时间段分组
				time_key = (start_time, end_time)
				if time_key not in damaged_stands_by_time:
					damaged_stands_by_time[time_key] = []
				damaged_stands_by_time[time_key].append(stand_id)
				damaged_stands_found.append((stand_id, start_time, end_time))
			
			# 为每个时间段的停机位创建事件
			for (start, end), stand_ids in damaged_stands_by_time.items():
				disturbance_events.append({
					"id": event_id_counter,
					"start": start,
					"end": end,
					"stands": sorted(stand_ids)
				})
				event_id_counter += 1
			
			# 记录调试信息
			if damaged_stands_found:
				_logger.debug(f"[SNAPSHOT] 查询到 {len(damaged_stands_found)} 个损坏的停机位: {[s[0] for s in damaged_stands_found]}")
			else:
				_logger.debug(f"[SNAPSHOT] 未查询到损坏的停机位（当前时间: {current_time_min:.2f}分钟）")
			
			# 6.2 查询损坏的设备（固定设备，只查询在当前时间点仍然有效的）
			damaged_fixed_devices_query = """
			MATCH (d:Device:FixedDevice)
			WHERE d.isDamaged = true
			  AND d.failureTime IS NOT NULL
			  AND (d.failureDuration IS NULL OR d.failureTime + COALESCE(d.failureDuration, 60.0) >= $current_time)
			RETURN d.name AS name,
			       d.failureTime AS failure_time,
			       d.failureDuration AS failure_duration
			ORDER BY d.name
			"""
			
			damaged_devices_by_time = {}  # {(start, end): [device_names]}
			
			for record in sess.run(damaged_fixed_devices_query, {"current_time": current_time_min}):
				device_name = record.get("name", "")
				if not device_name:
					continue
				
				# 规范化设备名称（确保是FR格式）
				device_norm = device_name.strip()
				original_name = device_name.strip()  # 保存原始名称
				# 如果设备名称不是FR格式，尝试映射到标准格式
				if not re.match(r"FR\d{1,3}$", device_norm):
					# 尝试提取数字：如"FR5" -> "FR05"，但保持原格式（强化学习环境可能接受"FR5"）
					# 如果包含中文，尝试查找对应的FR设备（简化处理，保持原名称）
					# KG中的设备名称应该已经是FR格式，这里主要处理可能的格式不一致
					if re.search(r"FR", device_norm, re.IGNORECASE):
						# 如果包含FR但不匹配标准格式，尝试规范化（如"FR5"保持原样，"FR05"也保持原样）
						# 强化学习环境接受"FR5"和"FR05"两种格式
						pass
					# 如果完全不是FR格式，保持原名称（可能在KG中使用中文名称）
				
				# 记录设备名称映射（规范化名称 -> 原始名称）
				if device_norm != original_name or device_norm not in device_name_map:
					device_name_map[device_norm] = original_name
				
				failure_time = record.get("failure_time")
				failure_duration = record.get("failure_duration")
				
				# 确定开始和结束时间
				start_time = float(failure_time) if failure_time is not None else float(current_time_min)
				duration = float(failure_duration) if failure_duration is not None else 60.0
				end_time = start_time + duration
				
				# 按时间段分组
				time_key = (start_time, end_time)
				if time_key not in damaged_devices_by_time:
					damaged_devices_by_time[time_key] = []
				damaged_devices_by_time[time_key].append(device_norm)
			
			# 6.3 查询损坏的设备（移动设备，只查询在当前时间点仍然有效的）
			damaged_mobile_devices_query = """
			MATCH (d:Device:MobileDevice)
			WHERE d.isDamaged = true
			  AND d.failureTime IS NOT NULL
			  AND (d.failureDuration IS NULL OR d.failureTime + COALESCE(d.failureDuration, 60.0) >= $current_time)
			RETURN d.name AS name,
			       d.failureTime AS failure_time,
			       d.failureDuration AS failure_duration
			ORDER BY d.name
			"""
			
			for record in sess.run(damaged_mobile_devices_query, {"current_time": current_time_min}):
				device_name = record.get("name", "")
				if not device_name:
					continue
				
				# 规范化设备名称（确保是MR格式）
				device_norm = device_name.strip()
				original_name = device_name.strip()  # 保存原始名称
				# 如果设备名称不是MR格式，尝试映射
				if not re.match(r"MR\d{2}$", device_norm):
					# 尝试使用映射函数将中文名称映射到MR格式
					try:
						from models.triples_extraction import _map_chinese_to_mobile_device
						mapped_name = _map_chinese_to_mobile_device(device_norm)
						if mapped_name:
							device_norm = mapped_name
						elif re.search(r"MR", device_norm, re.IGNORECASE):
							# 如果包含MR但不匹配标准格式，尝试规范化（如"MR5" -> "MR05"）
							m = re.search(r"MR(\d{1,2})", device_norm, re.IGNORECASE)
							if m:
								num = int(m.group(1))
								device_norm = f"MR{num:02d}"
					except Exception:
						# 如果映射失败，保持原名称
						pass
				
				# 记录设备名称映射（规范化名称 -> 原始名称）
				if device_norm != original_name or device_norm not in device_name_map:
					device_name_map[device_norm] = original_name
				
				failure_time = record.get("failure_time")
				failure_duration = record.get("failure_duration")
				
				# 确定开始和结束时间
				start_time = float(failure_time) if failure_time is not None else float(current_time_min)
				duration = float(failure_duration) if failure_duration is not None else 60.0
				end_time = start_time + duration
				
				# 按时间段分组（与固定设备合并）
				time_key = (start_time, end_time)
				if time_key not in damaged_devices_by_time:
					damaged_devices_by_time[time_key] = []
				damaged_devices_by_time[time_key].append(device_norm)
			
			# 为每个时间段的设备创建事件
			for (start, end), device_names in damaged_devices_by_time.items():
				disturbance_events.append({
					"id": event_id_counter,
					"start": start,
					"end": end,
					"devices": sorted(device_names)
				})
				event_id_counter += 1
			
			# 6.4 查询损坏的资源（只查询在当前时间点仍然有效的）
			damaged_resources_query = """
			MATCH (r:Resource)
			WHERE r.isDamaged = true
			  AND r.failureTime IS NOT NULL
			  AND (r.failureDuration IS NULL OR r.failureTime + COALESCE(r.failureDuration, 60.0) >= $current_time)
			RETURN r.name AS name,
			       r.failureTime AS failure_time,
			       r.failureDuration AS failure_duration
			ORDER BY r.name
			"""
			
			damaged_resources_by_time = {}  # {(start, end): [resource_types]}
			
			for record in sess.run(damaged_resources_query, {"current_time": current_time_min}):
				resource_name = record.get("name", "")
				if not resource_name:
					continue
				
				# 资源名称应该已经是R格式（如"R005"）
				resource_type = resource_name
				original_name = resource_name  # 保存原始名称（通常相同）
				
				# 记录资源名称映射（规范化名称 -> 原始名称）
				# 通常资源名称已经是规范化格式，但为了统一性和后续写回KG，仍需记录映射
				if resource_type not in resource_name_map:
					resource_name_map[resource_type] = original_name
				
				failure_time = record.get("failure_time")
				failure_duration = record.get("failure_duration")
				
				# 确定开始和结束时间
				start_time = float(failure_time) if failure_time is not None else float(current_time_min)
				duration = float(failure_duration) if failure_duration is not None else 60.0
				end_time = start_time + duration
				
				# 按时间段分组
				time_key = (start_time, end_time)
				if time_key not in damaged_resources_by_time:
					damaged_resources_by_time[time_key] = []
				damaged_resources_by_time[time_key].append(resource_type)
			
			# 为每个时间段的资源创建事件
			for (start, end), resource_types in damaged_resources_by_time.items():
				disturbance_events.append({
					"id": event_id_counter,
					"start": start,
					"end": end,
					"resource_types": sorted(resource_types)
				})
				event_id_counter += 1
			
			snapshot["disturbance_events"] = disturbance_events
			
			# 记录最终的 disturbance_events 信息
			if disturbance_events:
				total_stands = sum(len(evt.get("stands", [])) for evt in disturbance_events)
				total_devices = sum(len(evt.get("devices", [])) for evt in disturbance_events)
				total_resources = sum(len(evt.get("resource_types", [])) for evt in disturbance_events)
				_logger.info(f"[SNAPSHOT] 生成的 disturbance_events: {len(disturbance_events)} 个事件，涉及 {total_stands} 个停机位、{total_devices} 个设备、{total_resources} 个资源")
			else:
				_logger.info(f"[SNAPSHOT] 未生成任何 disturbance_events（当前时间: {current_time_min:.2f}分钟）")
			
			# 注意：readme.md 示例中没有 site_unavailable 字段，因此不包含此字段
		
	except Exception as e:
		# 发生错误时返回基础结构
		import logging
		logging.warning(f"[SNAPSHOT] KG查询失败: {e}")
		pass
	
	# 创建反向映射：数字ID -> 原始名称
	id_to_name_map = {plane_id: name for name, plane_id in aircraft_map.items()}
	
	# 创建包含所有映射关系的字典
	all_mappings = {
		"aircraft": id_to_name_map,  # {plane_id: name} - 飞机ID到名称的映射
		"devices": device_name_map,  # {normalized_name: original_name} - 规范化设备名称到原始KG名称的映射
		"resources": resource_name_map  # {normalized_name: original_name} - 规范化资源名称到原始KG名称的映射
	}
	
	return snapshot, id_to_name_map, all_mappings


__all__ = [
	"read_jsonl",
	"save_jsonl",
	"detect_resources_from_triples",
	"build_samples_from_train_triples",
	"build_decomposition_samples_from_events",
	"detect_potential_conflict",
	"generate_snapshot_from_kg",
]
