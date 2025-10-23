"""
将原始 CSV 文本预处理为可训练的 JSONL。

问题背景
- 原始 CSV 中，行内会有坐标，如：坐标(60,260)。由于英文逗号 "," 未被转义/引用，
  导致 CSV 读取时该行被拆成多个单元格（多列）。

目标
- 将每一行所有列重新拼接回一条完整文本，并修复坐标中的逗号，
  输出 JSON Lines（每行一个 JSON 对象）：{"id": <int>, "text": <str>}。

用法
python -m utils.origindata_preprocessing \
  --input "d:/WorkSpace/haitianbei/data_provider/海天杯-ST_Job_训练集.csv" \
  --output "d:/WorkSpace/haitianbei/data_provider/train_texts.jsonl"

注意
- 自动尝试多种常见编码（utf-8-sig, utf-16, gbk）。
- 自动将坐标中的英文逗号统一为中文逗号“，”，并规范中英文括号。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Iterable, List

import pandas as pd


PREFERRED_ENCODINGS = [
	"utf-8-sig",
	"utf-16",
	"gbk",
]


def _read_csv_any_encoding(path: str) -> pd.DataFrame:
	last_err = None
	for enc in PREFERRED_ENCODINGS:
		try:
			# 使用 python 引擎更宽容，header=None 表示无表头
			df = pd.read_csv(
				path,
				encoding=enc,
				header=None,
				dtype=str,
				keep_default_na=False,  # 空串别转 NaN
				engine="python",
			)
			# 基于抽样检查是否出现明显 mojibake（可选）。此处略。
			return df
		except Exception as e:  # noqa: BLE001
			last_err = e
	raise RuntimeError(f"无法读取 CSV：{path}; 尝试编码={PREFERRED_ENCODINGS}; 最后错误: {last_err}")


def _normalize_brackets(s: str) -> str:
	# 全角括号转半角，便于匹配
	return (
		s.replace("（", "(")
		.replace("）", ")")
		.replace("[", "[")
		.replace("]", "]")
	)


def _fix_coordinates_commas(s: str) -> str:
	"""将 (num, num) 统一转为 (num，num) 并恢复中文括号。
	同时容忍空格/全角逗号/半角逗号混排。
	"""
	import re

	t = _normalize_brackets(s)

	def repl(m: re.Match[str]) -> str:
		a = m.group(1)
		b = m.group(2)
		return f"（{a}，{b}）"  # 输出为中文括号+中文逗号

	# 匹配 (num , num) 的形式，支持小数
	t = re.sub(r"\((\d+(?:\.\d+)?)[\s,，]+(\d+(?:\.\d+)?)\)", repl, t)

	return t


def _merge_row_cells(cells: List[str]) -> str:
	"""将一行多个单元格合并为一条文本。

	策略：
	- 过滤空单元格后，用英文逗号连接，再统一修复坐标内的逗号为中文逗号。
	- 保留原有中文标点（如“，”）。
	- 去除两端空白与多余逗号。
	"""
	parts = [c.strip() for c in cells if isinstance(c, str) and c.strip() != ""]
	if not parts:
		return ""
	raw = ",".join(parts)
	# 去掉结尾多余逗号（若存在）
	raw = raw.rstrip(",")
	# 修复坐标内逗号
	raw = _fix_coordinates_commas(raw)
	# 统一一些可见空白
	raw = " ".join(raw.split())
	return raw


def convert_csv_to_jsonl(input_path: str, output_path: str) -> int:
	df = _read_csv_any_encoding(input_path)
	# 将所有列视为文本片段，行级合并
	texts: List[str] = []
	for _, row in df.iterrows():
		cells = [str(v) if not pd.isna(v) else "" for v in row.tolist()]
		text = _merge_row_cells(cells)
		if text:
			texts.append(text)

	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	with open(output_path, "w", encoding="utf-8") as f:
		for i, t in enumerate(texts):
			obj = {"id": i, "text": t}
			f.write(json.dumps(obj, ensure_ascii=False) + "\n")
	return len(texts)


def main(argv: list[str] | None = None) -> None:
	parser = argparse.ArgumentParser(description="原始 CSV 转 JSONL 预处理")
	parser.add_argument("--input", required=True, help="输入 CSV 文件路径")
	parser.add_argument("--output", required=True, help="输出 JSONL 文件路径")
	args = parser.parse_args(argv)

	total = convert_csv_to_jsonl(args.input, args.output)
	print(f"Wrote {total} lines to {args.output}")


if __name__ == "__main__":
	main()
