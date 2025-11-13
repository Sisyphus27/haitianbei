"""
轻量级三元组抽取器（规则基线）

功能
- 从中文运行日志中抽取多三元组，覆盖常见航空地面调度要素：飞机ID、动作、跑道、坐标、速度、车辆待命等。
- 一句文本可产生多个 (主语, 谓词, 宾语) 三元组。

为什么先上规则基线
- 您的数据中术语较稳定（如“飞机A001”“着陆跑道Z”“坐标(x，y)”等），规则法轻量、可解释、零训练即可运行。
- 后续可在此基础上平滑替换为轻量模型（DistilBERT 等）做关系判别。

用法
>>> from models.triples_extraction import extract_triples
>>> text = "时间：2025年7月1日 08:00:00，信息：飞机A001开始着陆，使用着陆跑道Z，坐标(60，260)，速度15.2米/秒；系统检测到5号牵引车待命于着陆跑道。"
>>> extract_triples(text)

日期: 2025-10-23
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
	# 若还带有“跑道”前缀，去掉后按主体处理
	if s.startswith("跑道"):
		s = s[len("跑道"):]
		s = s.strip()
	if not s:
		return "跑道Z"
	if s.upper() == "Z":
		return "跑道Z"
	if s.isdigit():
		return f"跑道{s}"
	# 兜底：若仍有“跑道X”形式，尽量保持
	if tok.startswith("跑道") and len(tok) > 2:
		return tok.replace(" ", "")
	return "跑道Z"


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
	# 若未显式出现“飞机X”，可为空

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
	# 例：使用着陆跑道Z / 使用跑道Z / 着陆跑道Z
	_runway_raw = re.findall(r"使用?(?:着陆)?跑道\s*([A-Za-z0-9号]+)", text)
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
		_add(triples, seen, (tug_no, "牵引", f"飞机{ac_id}"))
		_add(triples, seen, (tug_no, "滑至", f"{gate_no}号停机位"))
		_add(triples, seen, (f"飞机{ac_id}", "滑至", f"{gate_no}号停机位"))
		tug_trip_found = True

	# 6.2) 若有“牵引车…速度X米/秒”或“滑行速度X米/秒”，把速度赋给牵引车与句中出现的具体飞机ID
	if speed:
		m_tug_speed = re.search(r"(\d+号牵引车).*?(?:滑行速度|速度)\s*[0-9]+(?:\.[0-9]+)?\s*米/秒", text)
		if m_tug_speed:
			_add(triples, seen, (m_tug_speed.group(1), "速度", speed))
			# 同步给出现的具体飞机ID（若文本中有多架，全部赋值）
			for ac in aircraft_ids:
				_add(triples, seen, (ac, "速度", speed))

	# 7) 牵引车待命位置
	# 例：系统检测到5号牵引车待命于着陆跑道Z / 着陆跑道
	for veh_no, loc in re.findall(r"(\d+号牵引车).*?待命于(着陆跑道\s*[A-Za-z0-9号]?)", text):
		_add(triples, seen, (veh_no, "待命位置", _canon_runway_token(loc)))

	# 7.1) 牵引车释放连接（显式记录为设备动作，便于可视化与审计）
	# 例：5号牵引车释放连接。
	for veh_no in re.findall(r"(\d+号牵引车).*?释放连接", text):
		_add(triples, seen, (veh_no, "动作", "释放连接"))

	# 8) 设备到达停机位（移动加氧车/移动加氮车/压缩空气终端/氧气终端/氮气终端 等）
	for device, gate in re.findall(r"((?:\d+号)?(?:移动)?(?:加氧车|加氮车|压缩空气终端|氧气终端|氮气终端))到达(\d+)号停机位", text):
		_add(triples, seen, (device, "到达停机位", f"{gate}号"))

	# 9) 分配停机位 -> 飞机
	for gate, ac_id in re.findall(r"分配(\d+)号停机位给飞机([A-Za-z0-9]+)", text):
		_add(triples, seen, (f"飞机{ac_id}", "分配停机位", f"{gate}号"))

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

		# 移动设备调度："移动加氧车从6号停机位前往14号停机位" -> (移动加氧车, 滑至, 停机位14)
		for dev, src_gate, dst_gate in re.findall(r"(移动加氧车|移动加氮车|\d+号压缩空气终端|\d+号氧气终端|\d+号氮气终端)从(\d+)号停机位前往(\d+)号停机位", text):
			_add(triples, seen, (dev, "滑至", f"{dst_gate}号停机位"))
			# 可选：标记目标作为到达（将由 update_with_triples 处理占用）
			_add(triples, seen, (dev, "到达停机位", f"{dst_gate}号"))

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

