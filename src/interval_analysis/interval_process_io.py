"""Strict, GUI-free parsing of the process files accepted by the interval tool."""
from __future__ import annotations

import csv
import io
import math
import re
from pathlib import Path

import pandas as pd


class ProcessInfoFormatError(ValueError):
    """The process file cannot be interpreted without guessing its contents."""


_DIRECT = ("N", "ap", "ae", "F", "MRR", "s", "S", "G")
_EXPORT = ("N", "S", "ap", "ae", "F", "s", "MRR", "G")
_OUTPUT_COLUMNS = (
    "source_index", "line_no_raw", "N_str", "ap", "ae", "F_program", "S",
    "s", "gcode_content", "MRR", "source_file_line", "source_sequence",
    "source_mrr", "source_mrr_unit", "source_state_code",
)


def _decode_file(path: Path) -> tuple[str, str]:
    content = path.read_bytes()
    if content.startswith((b"\xff\xfe", b"\xfe\xff")):
        encodings = ("utf-16",)
    elif content.startswith(b"\xef\xbb\xbf"):
        encodings = ("utf-8-sig",)
    else:
        encodings = ("utf-8", "gb18030")
    for encoding in encodings:
        try:
            text = content.decode(encoding, errors="strict")
        except UnicodeDecodeError:
            continue
        if "\x00" not in text:
            return text, encoding
    raise ProcessInfoFormatError("工艺文件编码无效；支持 UTF-8、带 BOM 的 UTF-16 和 GB18030")


def _header_name(token: str) -> str:
    value = token.strip().lstrip("\ufeff")
    normalized = value.replace(" ", "").replace("³", "3").replace("^3", "3")
    names = {
        "序号": "index", "index": "index", "N": "N", "n": "N",
        "ap": "ap", "ap(mm)": "ap", "ae": "ae", "ae(mm)": "ae",
        "F": "F", "F(mm/min)": "F", "S": "S", "S(r/min)": "S",
        "s": "s", "s(mm)": "s", "G": "G", "g": "G",
        "MRR": "MRR", "MRR(mm3/min)": "MRR", "MRR(mm3/s)": "MRR",
        "state_code": "state_code",
    }
    if normalized not in names:
        raise ProcessInfoFormatError(f"不支持的工艺表列名：{value!r}")
    return names[normalized]


def _read_header(tokens: list[str], *, csv_mode: bool) -> tuple[list[str], str]:
    names = [_header_name(token) for token in tokens]
    if len(set(names)) != len(names):
        raise ProcessInfoFormatError("工艺表包含重复列名")
    body = list(names)
    if body and body[0] == "index":
        body.pop(0)
    if csv_mode and body and body[-1] == "state_code":
        body.pop()
    if tuple(body) not in (_DIRECT, _EXPORT):
        raise ProcessInfoFormatError("工艺表列顺序不符合 AfoMilling 或 ProcessInfo 格式")
    if not csv_mode and names[-1] != "G":
        raise ProcessInfoFormatError("空白分隔工艺文件必须以 G 列结束")
    mrr_token = tokens[names.index("MRR")].replace(" ", "").replace("³", "3").replace("^3", "3")
    unit = "mm3/s" if mrr_token == "MRR(mm3/s)" else "mm3/min" if mrr_token == "MRR(mm3/min)" else "unspecified"
    return names, unit


def _nonnegative(value: str, name: str, file_line: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProcessInfoFormatError(f"工艺文件第 {file_line} 行的 {name} 不是数值") from exc
    if not math.isfinite(number) or number < 0:
        raise ProcessInfoFormatError(f"工艺文件第 {file_line} 行的 {name} 必须为有限非负数")
    return number


def _integer(value: str, name: str, file_line: int, *, minimum: int) -> int:
    number = _nonnegative(value, name, file_line)
    if number < minimum or not number.is_integer() or number > 2**53 - 1:
        raise ProcessInfoFormatError(f"工艺文件第 {file_line} 行的 {name} 必须为不小于 {minimum} 的精确整数")
    return int(number)


def read_interval_process(path: str | Path) -> pd.DataFrame:
    """Read one process table without inferring NC geometry, timing or samples.

    The N column denotes one-based physical NC lines. ``line_no_raw`` is N-1;
    ``source_index`` numbers the retained data rows from zero. Repeated NC
    lines and original row order are preserved. ``s`` remains the original
    distance value, with cumulative/incremental semantics left unspecified;
    the caller must obtain that declaration from its verified input contract.
    MRR is recalculated in mm3/s from ap, ae and F.
    Input state_code, when present, is provenance only and is never reused as
    a new segmentation result.
    """
    source_path = Path(path).resolve()
    text, encoding = _decode_file(source_path)
    physical_lines = text.splitlines()
    header_index = next((i for i, line in enumerate(physical_lines) if line.strip()), None)
    if header_index is None:
        raise ProcessInfoFormatError("工艺文件为空")
    csv_mode = "," in physical_lines[header_index]
    if csv_mode:
        reader = csv.reader(io.StringIO(text), strict=True)
        records: list[tuple[int, list[str]]] = []
        previous_line = 0
        try:
            for cells in reader:
                start_line = previous_line + 1
                previous_line = reader.line_num
                if cells and any(cell.strip() for cell in cells):
                    records.append((start_line, cells))
        except csv.Error as exc:
            raise ProcessInfoFormatError(f"工艺 CSV 第 {reader.line_num} 行格式错误：{exc}") from exc
        names, mrr_unit = _read_header(records[0][1], csv_mode=True)
        data_records = records[1:]
    else:
        names, mrr_unit = _read_header(physical_lines[header_index].split(), csv_mode=False)
        # The G column is the entire remainder, including spaces between NC words.
        data_records = [(i + 1, line.split(maxsplit=len(names) - 1))
                        for i, line in enumerate(physical_lines)
                        if i > header_index and line.strip()]
    output = []
    for file_line, cells in data_records:
        if len(cells) != len(names):
            raise ProcessInfoFormatError(f"工艺文件第 {file_line} 行有 {len(cells)} 列，应有 {len(names)} 列")
        values = dict(zip(names, cells))
        line_token = re.sub(r"^[Nn]", "", values["N"].strip())
        nc_line = _integer(line_token, "N", file_line, minimum=1)
        numbers = {key: _nonnegative(values[key], key, file_line)
                   for key in ("ap", "ae", "F", "S", "s", "MRR")}
        sequence = _integer(values["index"], "序号", file_line, minimum=0) if "index" in values else None
        previous_state = _integer(values["state_code"], "state_code", file_line, minimum=0) if "state_code" in values else None
        if previous_state is not None and previous_state > 5:
            raise ProcessInfoFormatError(f"工艺文件第 {file_line} 行的 state_code 必须在 0..5 内")
        mrr = numbers["ap"] * numbers["ae"] * numbers["F"] / 60.0
        if not math.isfinite(mrr):
            raise ProcessInfoFormatError(f"工艺文件第 {file_line} 行计算 MRR 时溢出")
        output.append({
            "source_index": len(output), "line_no_raw": nc_line - 1, "N_str": f"N{nc_line}",
            "ap": numbers["ap"], "ae": numbers["ae"], "F_program": numbers["F"],
            "S": numbers["S"], "s": numbers["s"], "gcode_content": values["G"].strip(),
            "MRR": mrr, "source_file_line": file_line, "source_sequence": sequence,
            "source_mrr": numbers["MRR"], "source_mrr_unit": mrr_unit,
            "source_state_code": previous_state,
        })
    if not output:
        raise ProcessInfoFormatError("工艺文件只有表头，没有数据行")
    frame = pd.DataFrame.from_records(output, columns=_OUTPUT_COLUMNS)
    frame.attrs.update(source_path=str(source_path), encoding=encoding,
                       line_number_semantics="one_based_physical_nc_line",
                       s_semantics="source_value_unspecified", mrr_unit="mm3/s")
    return frame
