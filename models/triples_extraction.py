"""
三元组提取模块：从事件文本中抽取结构化信息

功能：
1. 实体识别：识别飞机、停机位、跑道、设备等实体
2. 关系抽取：提取实体之间的关系（如"使用"、"位于"、"执行"等）
3. 时间解析：提取和规范化时间信息
4. 状态提取：识别飞机的动作状态（着陆、起飞、滑行等）
5. 资源提取：提取资源分配信息（停机位、设备使用等）

在stream-judge模式中的作用：
- 从事件文本中提取三元组（主语-谓词-宾语）
- 为知识图谱更新提供结构化输入
- 支持多种航空地面调度场景的文本模式识别

使用规则基线方法，适用于术语稳定的领域文本，具有轻量、可解释、零训练的优势。
"""

from __future__ import annotations

import logging as _logging
if not _logging.getLogger().handlers:
	_logging.basicConfig(level=_logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

import re
from typing import List, Tuple, Optional, Set

Triple = Tuple[str, str, str]


def _norm_datetime(s: str) -> str:
	"""将“2025年7月1日 08:00:00”规范为“2025-07-01 08:00:00”。保守处理，缺位不补。"""
	# 匹配：YYYY年M月D日 可选空格 HH:MM:SS
	m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日\s*(\d{1,2}:\d{2}:\d{2})?", s)
	if not m:
		return s.strip()
	y, mo, d, t = m.group(1), m.group(2), m.group(3), m.group(4)
	mo = mo.zfill(2)
	d = d.zfill(2)
	if t:
		# 也容忍 8:00:00 -> 08:00:00 的情况
		parts = t.split(":")
		if len(parts) == 3:
			hh = parts[0].zfill(2)
			t = f"{hh}:{parts[1]}:{parts[2]}"
		return f"{y}-{mo}-{d} {t}"
	return f"{y}-{mo}-{d}"


def _canon_runway_token(tok: str) -> str:
	"""将各种跑道写法规范为“跑道Z/跑道29/30/31”。

	规则：
	- 去除空格与“号”字；
	- 去除前缀“着陆跑道”；
	- Z/z -> 跑道Z；纯数字 -> 跑道{数字}；已带“跑道”前缀则规范大小写；
	- 起飞跑道Q1/Q2/Q3 -> 跑道29/30/31（根据技术资料，Q1对应29，Q2对应30，Q3对应31）；
	- 空字符串或异常回退 -> 跑道Z。
	"""
	if tok is None:
		return "跑道Z"
	s = str(tok).strip().replace(" ", "")
	s = s.replace("号", "")
	# 去掉“着陆跑道”前缀
	if s.startswith("着陆跑道"):
		s = s[len("着陆跑道"):]
		s = s.strip()
	# 去掉“起飞跑道”前缀
	if s.startswith("起飞跑道"):
		s = s[len("起飞跑道"):]
		s = s.strip()
	# 若还带有“跑道”前缀，去掉后按主体处理
	if s.startswith("跑道"):
		s = s[len("跑道"):]
		s = s.strip()
	if not s:
		return "跑道Z"
	if s.upper() == "Z":
		return "跑道Z"
	# 起飞跑道映射：Q1->29, Q2->30, Q3->31
	if s.upper() == "Q1":
		return "跑道29"
	if s.upper() == "Q2":
		return "跑道30"
	if s.upper() == "Q3":
		return "跑道31"
	if s.isdigit():
		return f"跑道{s}"
	# 兜底：若仍有“跑道X”形式，尽量保持
	if tok.startswith("跑道") and len(tok) > 2:
		return tok.replace(" ", "")
	return "跑道Z"


def _map_chinese_to_mobile_device(chinese_name: str) -> str | None:
	"""将中文设备名称映射到移动设备编号。
	
	映射规则：
	- "移动加氧车" -> MR02
	- "移动加氮车" -> MR03
	- "X号压缩空气终端" -> MR06-MR09（根据编号顺序：1号->MR06, 2号->MR07, 3号->MR08, 4号->MR09）
	- "X号氧气终端" -> MR10-MR13（1号->MR10, 2号->MR11, 3号->MR12, 4号->MR13）
	- "X号氮气终端" -> MR14-MR17（1号->MR14, 2号->MR15, 3号->MR16, 4号->MR17）
	- "X号牵引车" -> MR18-MR27（1号->MR18, 2号->MR19, ..., 10号->MR27）
	"""
	name = str(chinese_name).strip()
	
	# 移动加氧车/加氮车（无编号）
	if name == "移动加氧车":
		return "MR02"
	if name == "移动加氮车":
		return "MR03"
	
	# 带编号的设备
	m_num = re.match(r"(\d+)号(压缩空气终端|氧气终端|氮气终端|牵引车)", name)
	if m_num:
		num = int(m_num.group(1))
		dev_type = m_num.group(2)
		
		if dev_type == "压缩空气终端":
			if 1 <= num <= 4:
				return f"MR{5 + num:02d}"  # MR06-MR09
		elif dev_type == "氧气终端":
			if 1 <= num <= 4:
				return f"MR{9 + num:02d}"  # MR10-MR13
		elif dev_type == "氮气终端":
			if 1 <= num <= 4:
				return f"MR{13 + num:02d}"  # MR14-MR17
		elif dev_type == "牵引车":
			if 1 <= num <= 10:
				return f"MR{17 + num:02d}"  # MR18-MR27
	
	return None


def extract_triples(text: str) -> List[Triple]:
	# 将“着陆跑道”统一规范为“跑道Z”，并规范其他跑道写法
	"""基于正则与启发式的轻量三元组抽取。

	规则覆盖要点：
	- 时间：时间：2025年7月1日 08:00:00
	- 飞机：飞机A001 / 飞机B12 等
	- 动作：开始着陆/开始起飞/降落/起飞/滑行
	- 跑道：使用着陆跑道Z / 着陆跑道Z
	- 坐标：坐标(60，260) 或 坐标(60,260)
	- 速度：速度15.2米/秒
	- 牵引车：5号牵引车待命于着陆跑道[Z]

	返回：List[(subject, predicate, object)]
	"""
	if not text:
		return []

	triples: List[Triple] = []
	seen: Set[Triple] = set()

	# 1) 时间
	# 支持前缀“时间：”可有可无
	time_match = re.search(r"(?:时间[:：]\s*)?(\d{4}年\d{1,2}月\d{1,2}日\s*\d{1,2}:\d{2}:\d{2})", text)
	norm_time: Optional[str] = None
	if time_match:
		norm_time = _norm_datetime(time_match.group(1))

	# 2) 飞机ID（可多架）
	aircraft_ids = [f"飞机{m}" for m in re.findall(r"飞机([A-Za-z0-9]+)", text)]
	# 若未显式出现“飞机X”，尝试兜底匹配裸露的飞机代号（如“A001”）
	# 为避免误匹配设备编号（如 MR10），此处仅匹配以字母 A 开头且后随 3-4 位数字的代号。
	if not aircraft_ids:
		# 注意：
		# - 飞机编号范围 AXXX..EXXX（字母 A-E），XXX 为数字（常为 3 位或 4 位）。
		# - 为避免匹配到设备编号（如 MR10），仅匹配首字母 A-E 后跟 3-4 位数字。
		# - 在构造实体名时统一为大写形式（飞机A001），但保留原文别名作为桥接。
		bare_ac = re.findall(r"[A-Ea-e]\d{3,4}", text)
		for ac in bare_ac:
			ac_norm = ac.upper()
			_add(triples, seen, (f"飞机{ac_norm}", "别名", ac))  # 添加别名以便下游规范化
			aircraft_ids.append(f"飞机{ac_norm}")

	# 3) 动作关键词（根据语料扩充）
	# 说明：为尽量覆盖训练集中的地面保障动作，这里列出常见关键字；
	# 命中即为该飞机增加一个“动作”三元组。
	action_map = [
		# 飞行阶段
		(r"开始着陆", "动作", "开始着陆"),
		(r"着陆完成", "动作", "着陆完成"),
		(r"开始起飞", "动作", "开始起飞"),
		(r"起飞", "动作", "起飞"),
		(r"滑行结束", "动作", "滑行结束"),
		(r"滑行至", "动作", "滑行"),  # 与目标停机位组合
		(r"滑行", "动作", "滑行"),

		# 固定/放梯/开舱门/空调
		(r"开始固定作业", "动作", "固定作业开始"),
		(r"固定作业完成", "动作", "固定作业完成"),
		(r"放梯完成", "动作", "放梯完成"),
		(r"打开舱门开始|打开舱门", "动作", "打开舱门"),
		(r"打开空调", "动作", "打开空调"),

		# 保障作业（供液压/供电/清洁/污水/加燃油/气源）
		(r"供液压完成", "动作", "供液压完成"),
		(r"供电完成", "动作", "供电完成"),
		(r"污水处理完成", "动作", "污水处理完成"),
		(r"清洁处理完成", "动作", "清洁处理完成"),
		(r"加燃油", "动作", "加燃油"),
		(r"加(氧|氮)作业|加氧|加氮", "动作", "气体补给"),
		(r"加压缩空气", "动作", "加压缩空气"),

		# 牵引
	(r"开始牵引", "动作", "牵引"),
	(r"牵引(?!车)", "动作", "牵引"),
	]

	# 4) 跑道使用
	# 例：使用着陆跑道Z / 使用跑道Z / 着陆跑道Z / 起飞跑道Q1 / 到达起飞跑道Q1
	# 匹配模式：使用(着陆|起飞)?跑道 / (着陆|起飞)?跑道 / 到达起飞跑道
	_runway_raw = []
	# 匹配"使用着陆跑道Z"、"使用起飞跑道Q1"、"着陆跑道Z"、"起飞跑道Q1"、"跑道Z"等
	_runway_raw.extend(re.findall(r"(?:使用)?(?:着陆|起飞)?跑道\s*([A-Za-z0-9号]+)", text))
	# 匹配"到达起飞跑道Q1"这种格式
	_runway_raw.extend(re.findall(r"到达起飞跑道\s*([A-Za-z0-9号]+)", text))
	# 去重
	_runway_raw = list(set(_runway_raw))
	runway_use = [_canon_runway_token(x) for x in _runway_raw]

	# 5) 坐标
	coord = None
	mc = re.search(r"坐标[（(]\s*([0-9]+(?:\.[0-9]+)?)\s*[，,]\s*([0-9]+(?:\.[0-9]+)?)\s*[）)]", text)
	if mc:
		coord = f"({mc.group(1)},{mc.group(2)})"

	# 6) 速度（全局匹配，稍后按上下文决定归属）
	ms = re.search(r"(?:滑行)?速度\s*([0-9]+(?:\.[0-9]+)?)\s*米/秒", text)
	speed = f"{ms.group(1)}米/秒" if ms else None

	# 6.1) 牵引相关模式：牵引车牵引飞机 -> 飞机/牵引车 滑至 停机位
	# 示例：5号牵引车开始牵引飞机A001滑行至14号停机位
	tug_trip_found = False
	for tug_no, ac_id, gate_no in re.findall(r"(\d+号牵引车).*?牵引.*?飞机([A-Za-z0-9]+).*?(?:滑行至|滑至)\s*(\d+)号停机位", text):
		mapped_tug = _map_chinese_to_mobile_device(tug_no)
		if mapped_tug:
			_add(triples, seen, (mapped_tug, "牵引", f"飞机{ac_id}"))
			_add(triples, seen, (mapped_tug, "滑至", f"{gate_no}号停机位"))
		else:
			_add(triples, seen, (tug_no, "牵引", f"飞机{ac_id}"))
			_add(triples, seen, (tug_no, "滑至", f"{gate_no}号停机位"))
		_add(triples, seen, (f"飞机{ac_id}", "滑至", f"{gate_no}号停机位"))
		tug_trip_found = True

	# 6.2) 若有"牵引车…速度X米/秒"或"滑行速度X米/秒"，把速度赋给牵引车与句中出现的具体飞机ID
	if speed:
		m_tug_speed = re.search(r"(\d+号牵引车).*?(?:滑行速度|速度)\s*[0-9]+(?:\.[0-9]+)?\s*米/秒", text)
		if m_tug_speed:
			mapped_tug = _map_chinese_to_mobile_device(m_tug_speed.group(1))
			if mapped_tug:
				_add(triples, seen, (mapped_tug, "速度", speed))
			else:
				_add(triples, seen, (m_tug_speed.group(1), "速度", speed))
			# 同步给出现的具体飞机ID（若文本中有多架，全部赋值）
			for ac in aircraft_ids:
				_add(triples, seen, (ac, "速度", speed))

	# 7) 牵引车待命位置
	# 例：系统检测到5号牵引车待命于着陆跑道Z / 着陆跑道
	for veh_no, loc in re.findall(r"(\d+号牵引车).*?待命于(着陆跑道\s*[A-Za-z0-9号]?)", text):
		mapped_tug = _map_chinese_to_mobile_device(veh_no)
		if mapped_tug:
			_add(triples, seen, (mapped_tug, "待命位置", _canon_runway_token(loc)))
		else:
			_add(triples, seen, (veh_no, "待命位置", _canon_runway_token(loc)))

	# 7.1) 牵引车释放连接（显式记录为设备动作，便于可视化与审计）
	# 例：5号牵引车释放连接。
	for veh_no in re.findall(r"(\d+号牵引车).*?释放连接", text):
		mapped_tug = _map_chinese_to_mobile_device(veh_no)
		if mapped_tug:
			_add(triples, seen, (mapped_tug, "动作", "释放连接"))
		else:
			_add(triples, seen, (veh_no, "动作", "释放连接"))

	# 8) 设备到达停机位（移动加氧车/移动加氮车/压缩空气终端/氧气终端/氮气终端 等）
	for device, gate in re.findall(r"((?:\d+号)?(?:移动)?(?:加氧车|加氮车|压缩空气终端|氧气终端|氮气终端))到达(\d+)号停机位", text):
		mapped_dev = _map_chinese_to_mobile_device(device)
		if mapped_dev:
			_add(triples, seen, (mapped_dev, "到达停机位", f"{gate}号"))
		else:
			_add(triples, seen, (device, "到达停机位", f"{gate}号"))

	# 8.1) 设备释放（飞机或系统释放移动/终端设备）
	# 示例：A001释放移动加氧车/加氮车；A001释放2号压缩空气终端
	if re.search(r"释放移动加氧车/加氮车", text):
		_add(triples, seen, ("MR02", "动作", "释放"))
		_add(triples, seen, ("MR03", "动作", "释放"))
	for dev in re.findall(r"释放\s*((?:\d+号)?(?:移动)?(?:加氧车|加氮车|压缩空气终端|氧气终端|氮气终端))", text):
		mapped_dev = _map_chinese_to_mobile_device(dev)
		if mapped_dev:
			_add(triples, seen, (mapped_dev, "动作", "释放"))
		else:
			_add(triples, seen, (dev, "动作", "释放"))

	# 9) 分配停机位 -> 飞机
	for gate, ac_id in re.findall(r"分配(\d+)号停机位给飞机([A-Za-z0-9]+)", text):
		_add(triples, seen, (f"飞机{ac_id}", "分配停机位", f"{gate}号"))

	# 8.2) 移动设备调度（与飞机无关，应在飞机循环外抽取）
	# 示例：调度移动加氧车从14号停机位前往7号停机位(距离208.8米，预计70秒)
	for dev, src_gate, dst_gate in re.findall(r"(移动加氧车|移动加氮车|\d+号压缩空气终端|\d+号氧气终端|\d+号氮气终端)从(\d+)号停机位前往(\d+)号停机位", text):
		mapped_dev = _map_chinese_to_mobile_device(dev)
		if mapped_dev:
			_add(triples, seen, (mapped_dev, "滑至", f"{dst_gate}号停机位"))
			_add(triples, seen, (mapped_dev, "到达停机位", f"{dst_gate}号"))
		else:
			_add(triples, seen, (dev, "滑至", f"{dst_gate}号停机位"))
			_add(triples, seen, (dev, "到达停机位", f"{dst_gate}号"))

	# 将“动作/跑道/坐标/速度/时间”赋予到每一架匹配到的飞机；若无飞机但存在动作等，则跳过与飞机绑定的项
	for ac in aircraft_ids:
		# 时间
		if norm_time:
			_add(triples, seen, (ac, "时间", norm_time))

		# 动作（按首次命中的关键词）
		# 多动作可能并存。若句子中存在牵引车牵引该飞机或存在“滑(行)?至”指向停机位，则抑制把“牵引/滑行”作为飞机的动作，避免歧义。
		same_clause_to_gate = bool(re.search(fr"{re.escape(ac)}[^。；;]*?(?:滑行至|滑至)\s*\d+号停机位", text))
		has_tug_context = bool(re.search(r"\d+号牵引车.*?牵引.*?" + re.escape(ac), text))
		for pat, pred, obj in action_map:
			if re.search(pat, text):
				if obj in {"牵引", "滑行"} and (same_clause_to_gate or has_tug_context):
					# 抑制 (飞机, 动作, 牵引/滑行)
					continue
				_add(triples, seen, (ac, pred, obj))

		# 3.1) 保障作业实例模式 (任务_设备) 捕获：例如 "ZY03_MR01" 或 "ZY04_FR02"
		# 设计意图：当日志中直接出现具体任务与具体保障车编码的组合时，为后续 KG 建模 JobInstance 提供输入。
		# 模式说明：
		#   - 任务编码：ZY 后接 1-3 个字母/数字 (兼容 ZY_Z, ZY_M, ZY01, ZY18 等)
		#   - 分隔符：下划线 "_"
		#   - 设备类型前缀：FR (固定设备车) / MR (移动设备车) / 可未来扩展 TR, DR 等；后接 1-3 位数字。
		# 扩展：也允许形如 "ZY03 FR01" 有空格分隔的写法，统一规范为 "ZY03_FR01"。
		instance_tokens: Set[str] = set()
		# 下划线形式
		for m in re.findall(r"(ZY[A-Z0-9]{1,3})_((?:FR|MR)[0-9]{1,3})", text):
			instance_tokens.add("_".join(m))
		# 空格分隔形式（例如 ZY03 FR01）
		for m in re.findall(r"(ZY[A-Z0-9]{1,3})\s+((?:FR|MR)[0-9]{1,3})", text):
			instance_tokens.add("_".join(m))
		for tok in instance_tokens:
			_add(triples, seen, (ac, "动作", tok))

		# 跑道
		for rwy in runway_use:
			_add(triples, seen, (ac, "使用跑道", rwy))

		# 坐标
		if coord:
			_add(triples, seen, (ac, "坐标", coord))

		# 速度：若非牵引语境，也给该飞机绑定速度
		if speed and not re.search(r"\d+号牵引车.*?(?:滑行速度|速度)\s*[0-9]+(?:\.[0-9]+)?\s*米/秒", text):
			_add(triples, seen, (ac, "速度", speed))

		# 到达/目标停机位（仅当该飞机名与模式在同一子句内时）
		m_arr = re.search(fr"{re.escape(ac)}[^。；；]*?到达(\d+)号停机位", text)
		if m_arr:
			_add(triples, seen, (ac, "到达停机位", f"{m_arr.group(1)}号"))
		m_to = re.search(fr"{re.escape(ac)}[^。；;]*?(?:滑行至|滑至)\s*(\d+)号停机位", text)
		if m_to:
			_add(triples, seen, (ac, "滑至", f"{m_to.group(1)}号停机位"))

		# 启动类动作中常见关键词单独兜底（当句子出现“启动X”但前面正则未命中时生效）
		start_keywords = [
			("加燃油", "加燃油"),
			("放梯", "放梯"),
			("装卸货", "装卸货"),
			("上下客", "上下客"),
			("供液压", "供液压"),
			("供电", "供电"),
			("污水处理", "污水处理"),
			("清洁处理", "清洁处理"),
		]
		if re.search(r"启动", text):
			for kw, act in start_keywords:
				if kw in text:
					_add(triples, seen, (ac, "动作", act))

		# 设备使用抽取：供液压(使用2号液压站)、供电(使用2号供电站) 等
		# 规则：识别 “使用N号<类型>” 其中类型映射到固定设备 FRxx；生成 (飞机, 使用设备, FRxx) 三元组
		# 类型与 FR 列表映射（按技术资料固定设备覆盖 1-6 / 7-14 / 15-22 / 23-28 的顺序）
		resource_type_map = {
			"液压站": ["FR25", "FR26", "FR27", "FR28"],  # R008
			"供电站": ["FR5", "FR6", "FR7", "FR8"],      # R002
			"清洗装置": ["FR21", "FR22", "FR23", "FR24"],  # R007
			"供氮站": ["FR13", "FR14", "FR15", "FR16"],    # R005 / R013 混合氮气按固定加氮设备
			"供氧站": ["FR9", "FR10", "FR11", "FR12"],     # R003
			"供气站": ["FR17", "FR18", "FR19", "FR20"],    # R006
			"加油站": ["FR1", "FR2", "FR3", "FR4"],        # R001
		}
		for num, typ in re.findall(r"使用(\d+)号(液压站|供电站|清洗装置|供氮站|供氧站|供气站|加油站)", text):
			arr = resource_type_map.get(typ)
			if not arr:
				continue
			idx = int(num)
			# 1->0, 2->1 ... 超出映射长度则取最后一个
			fr_code = arr[min(max(idx - 1, 0), len(arr) - 1)]
			_add(triples, seen, (ac, "使用设备", fr_code))
		
		# 设备使用抽取：处理移动设备（移动加氧车、移动加氮车、X号压缩空气终端等）
		# 匹配模式：使用2号供氮站和移动加氮车 / 使用2号供气站和2号压缩空气终端
		# 注意：这里只处理移动设备部分，固定设备已在上面处理
		# 匹配"使用...和移动加氧车"、"使用...和移动加氮车"、"使用...和2号压缩空气终端"等
		use_segment = re.search(r"使用([^；。]*)", text)
		if use_segment:
			segment = use_segment.group(1)
			# 切分"和/、/,"
			tokens = re.split(r"[和、,，]\s*", segment)
			for tk in tokens:
				tk = tk.strip()
				if not tk:
					continue
				# 检查是否是移动设备（移动加氧车、移动加氮车、X号压缩空气终端等）
				if re.search(r"(?:移动)?(?:加氧车|加氮车)|(?:\d+号)?(?:压缩空气终端|氧气终端|氮气终端)", tk):
					mapped_dev = _map_chinese_to_mobile_device(tk)
					if mapped_dev:
						_add(triples, seen, (ac, "使用设备", mapped_dev))
					else:
						# 如果无法映射，仍然添加原始名称（兼容性）
						_add(triples, seen, (ac, "使用设备", tk))

		# （已前移到飞机循环外）

	# 10) 故障相关三元组抽取
	# 10.1) 设备故障：{停机位号}号停机位{设备名称}{故障描述}；{作业}暂停/无法启动
	# 示例：5号停机位供电接口异常；作业暂停
	#      24号停机位供氮终端连接失败；加氮作业无法启动
	device_failure_pattern = r"(\d+)号停机位(供电接口|污水处理装置|供氮终端|清洗装置|供气|供氧阀门|液压软管|液压泵|供氮管路)(异常|连接失败|压力异常|水压不足|短路|卡滞|破裂|过热报警|堵塞|故障)"
	for stand_num, device_name, failure_type in re.findall(device_failure_pattern, text):
		stand_name = f"停机位{stand_num}"
		# 规范化设备名称（映射到固定设备编号或保持原样）
		device_normalized = device_name
		# 尝试映射到固定设备（如果需要）
		if device_name == "供电接口":
			device_normalized = f"{stand_num}号供电接口"  # 可能需要进一步映射到FR设备
		elif device_name == "污水处理装置":
			device_normalized = f"{stand_num}号污水处理装置"
		elif device_name == "供氮终端":
			device_normalized = f"{stand_num}号供氮终端"
		elif device_name == "清洗装置":
			device_normalized = f"{stand_num}号清洗装置"
		elif device_name == "供气":
			device_normalized = f"{stand_num}号供气"
		elif device_name == "供氧阀门":
			device_normalized = f"{stand_num}号供氧阀门"
		elif device_name == "液压软管":
			device_normalized = f"{stand_num}号液压软管"
		elif device_name == "液压泵":
			device_normalized = f"{stand_num}号液压泵"
		elif device_name == "供氮管路":
			device_normalized = f"{stand_num}号供氮管路"
		
		_add(triples, seen, (stand_name, "设备故障", device_normalized))
		_add(triples, seen, (device_normalized, "故障类型", failure_type))
	
	# 10.2) 停机位故障：编号 {范围} 等 {数量} 个停机位故障不可用，不可用时间均为{时间}分钟
	# 示例：编号 10-15 等 6 个停机位故障不可用，不可用时间均为30分钟
	stand_failure_pattern = r"编号\s*(\d+)[-~](\d+)\s*等\s*(\d+)\s*个停机位故障不可用[，,]?\s*不可用时间均为\s*(\d+)\s*分钟"
	for start_num, end_num, count, duration in re.findall(stand_failure_pattern, text):
		start_id = int(start_num)
		end_id = int(end_num)
		duration_min = int(duration)
		# 为范围内的每个停机位添加故障三元组
		for stand_id in range(start_id, end_id + 1):
			stand_name = f"停机位{stand_id}"
			_add(triples, seen, (stand_name, "停机位故障", str(duration_min)))
	
	# 10.3) 故障恢复：{设备名称}修复完成/恢复；{飞机}{作业}恢复
	# 示例：供电接口修复完成；飞机A008供电作业恢复
	#      污水处理装置恢复；飞机A010污水处理作业恢复
	recovery_pattern = r"((?:供电接口|污水处理装置|供氮终端|清洗装置|供气|供氧阀门|液压软管|液压泵|供氮管路))(修复完成|恢复)[；;]\s*(?:飞机([A-Za-z0-9]+))?([^恢复]*作业)?恢复"
	for device_name, recovery_type, aircraft_id, job_type in re.findall(recovery_pattern, text):
		# 设备恢复
		device_normalized = device_name  # 可能需要进一步规范化
		_add(triples, seen, (device_normalized, "故障恢复", recovery_type))
		# 如果有飞机信息，添加飞机相关的恢复三元组
		if aircraft_id:
			_add(triples, seen, (f"飞机{aircraft_id}", "作业恢复", job_type if job_type else "作业"))
	
	# 10.4) 特殊故障：下一架降落飞机将因故障必须停在 {停机位} 号停机位修理，修理时间为 {时间}分钟
	# 示例：下一架降落飞机将因故障必须停在 5 号停机位修理，修理时间为 30分钟
	special_failure_pattern = r"下一架降落飞机将因故障必须停在\s*(\d+)\s*号停机位修理[，,]?\s*修理时间为\s*(\d+)\s*分钟"
	for stand_num, repair_duration in re.findall(special_failure_pattern, text):
		stand_name = f"停机位{stand_num}"
		repair_min = int(repair_duration)
		_add(triples, seen, (stand_name, "停机位故障", str(repair_min)))
		_add(triples, seen, (stand_name, "故障类型", "修理"))

	return triples


def _add(triples: List[Triple], seen: Set[Triple], t: Triple) -> None:
	if t not in seen:
		triples.append(t)
		seen.add(t)


if __name__ == "__main__":
	demo = (
		"时间：2025年7月1日 08:00:00，信息：飞机A001开始着陆，使用着陆跑道Z，坐标(60，260)，速度15.2米/秒；"
		"系统检测到5号牵引车待命于着陆跑道。"
	)
	for s, p, o in extract_triples(demo):
		_logging.info(f"({s}, {p}, {o})")

