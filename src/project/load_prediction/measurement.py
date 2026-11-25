"""Strict streaming reader for the Scope CSV contract v1.1."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .common import Cancel, checkpoint

RULE_VERSION = "1.2"
UNITS = {"time_s": "s", "spindle_power_w": "W", "spindle_speed_rpm": "r/min",
         "actual_feed_mm_min": "mm/min", "source_sample_index": "zero_based_sample"}


def detect_encoding(path):
    with open(path, "rb") as stream:
        prefix = stream.read(65536)
    if prefix.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if prefix.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    # Use complete lines; a chunk can end inside a multibyte character.
    prefix = prefix.rsplit(b"\n", 1)[0]
    for encoding in ("utf-8", "gb18030"):
        try:
            prefix.decode(encoding, errors="strict")
            return encoding
        except UnicodeDecodeError:
            pass
    raise ValueError(f"无法严格解码：{path}")


def tokens(row):
    row = [v.strip().removeprefix("<").removesuffix(">") for v in row]
    if row and row[-1] == "":
        row.pop()
    return row


def channel_key(row):
    if len(row) < 6:
        raise ValueError("ChannelInfo 字段不足")
    signal, axis = row[3], row[5]
    extra = row[6] if len(row) > 6 else ""
    if (signal, axis, extra) == ("G寄存器", "X轴", "432"):
        return "spindle_power_w"
    if signal == "实际速度" and axis == "SP轴":
        return "spindle_speed_rpm"
    if signal == "程序行号" and axis in ("AX", "X轴"):
        return "program_line_raw"
    if signal in ("实际进给", "实际进给速度") and axis in ("AX", "通道", "路径", "合成", ""):
        return "actual_feed_mm_min"
    if signal == "负载电流" and axis == "SP轴":
        return "spindle_current_raw"
    if axis in ("X轴", "Y轴", "Z轴"):
        a = axis[0].lower()
        if signal in ("指令位置", "实际位置"):
            return f"{'command' if signal == '指令位置' else 'actual'}_{a}_mm"
        if signal == "实际速度":
            return f"actual_v{a}_mm_min"
    return None


LEGACY_POSITION_POLICY = "actual_preferred_v1"
COMMAND_POSITION_POLICY = "command_only_v1"


def resolve_position_policy(metadata=None, position_policy=None):
    """Resolve a versioned label-location policy without overriding provenance."""
    declared = (metadata or {}).get("position_policy")
    if position_policy is not None and declared is not None and position_policy != declared:
        raise ValueError("显式位置策略与元数据不一致")
    policy = position_policy if position_policy is not None else declared
    if policy is None:
        policy = LEGACY_POSITION_POLICY
    if policy not in (LEGACY_POSITION_POLICY, COMMAND_POSITION_POLICY):
        raise ValueError(f"未知位置策略：{policy}")
    return policy


def select_position_channels(values, *, position_policy=LEGACY_POSITION_POLICY):
    """Choose one XYZ group used only to locate simultaneous measured labels.

    ``command_only_v1`` binds labels to the NC command path. The historical
    default prefers actual XYZ and then command XYZ. Neither policy mixes axes,
    switches groups at invalid samples, or derives measured feed from commands.
    Missing command XYZ under command-only policy never falls back to actual.
    """
    policy = resolve_position_policy(position_policy=position_policy)
    sources = ("command",) if policy == COMMAND_POSITION_POLICY else ("actual", "command")
    for source in sources:
        columns = tuple(f"{source}_{axis}_mm" for axis in "xyz")
        if all(column in values for column in columns):
            return source, columns
    return None, ()


@dataclass
class Measurement:
    source: str
    time_s: np.ndarray
    values: dict[str, np.ndarray]
    valid: np.ndarray
    metadata: dict = field(default_factory=dict)
    source_indices: np.ndarray | None = None
    source_lines: np.ndarray | None = None

    def require(self, *names):
        missing = [n for n in names if n not in self.values]
        if missing:
            raise ValueError(f"缺少通道：{', '.join(missing)}")


def read_header(path, sample_period_s=None):
    encoding = detect_encoding(path)
    channels, mapping, duplicates, sample_code = [], {}, {}, None
    with open(path, encoding=encoding, errors="strict", newline="") as stream:
        for row in csv.reader(stream):
            row = tokens(row)
            if not row:
                continue
            if row[0] == "ChannelInfo":
                key = channel_key(row)
                if key is not None:
                    if key in mapping:
                        duplicates.setdefault(key, [mapping[key]]).append(len(channels))
                    else:
                        mapping[key] = len(channels)
                channels.append(row)
            elif row[0] == "SmplInfo":
                sample_code = row[2] if len(row) > 2 else None
            elif row[0] == "ChannelData":
                break
    if not channels:
        raise ValueError("未找到 ChannelInfo")
    if sample_period_s is None:
        if sample_code != "1000":
            raise ValueError(f"未确认的采样字段 {sample_code}，需要显式 sample_period_s")
        sample_period_s = .001
    if not np.isfinite(sample_period_s) or sample_period_s <= 0:
        raise ValueError("采样周期必须为正数")
    return dict(encoding=encoding, channels=channels, mapping=mapping, duplicate_channels=duplicates,
                sample_period_s=float(sample_period_s), sample_code=sample_code,
                rule_version=RULE_VERSION, units=UNITS, scaling="exported physical units; no 10/360 rescale")


def iter_measurement(path, *, chunk_size=65536, sample_period_s=None, cancel: Cancel = None):
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size 必须为正整数")
    header = read_header(path, sample_period_s)
    width = len(header["channels"])
    rows, indices, row_valid, offset, issues = [], [], [], 0, []

    def build():
        array = np.asarray(rows, dtype=float)
        values = {k: array[:, i] for k, i in header["mapping"].items()}
        valid = np.asarray(row_valid)
        finite = np.isfinite(array).all(axis=1)
        for i in np.flatnonzero(~finite)[:max(0, 100 - len(issues))]:
            issues.append(f"行 {indices[i]}：存在无效数值")
        valid &= finite
        duplicate_conflicts = {}
        for key, columns in header["duplicate_channels"].items():
            same = np.all(np.isclose(array[:, columns], array[:, columns[0], None], rtol=0, atol=1e-8, equal_nan=True), axis=1)
            values[key] = np.where(same, array[:, columns[0]], np.nan)
            for column in columns:
                values[f"{key}__channel{column}"] = array[:, column]
            duplicate_conflicts[key] = int((~same).sum())
            if key in ("spindle_power_w", "spindle_speed_rpm", "program_line_raw"):
                valid &= same
        return Measurement(str(path), np.arange(offset, offset + len(rows)) * header["sample_period_s"],
                           values, valid, {**header, "sample_offset": offset, "duplicate_conflicts": duplicate_conflicts,
                           "source_lines": indices.copy(), "issues": issues.copy()},
                           source_indices=np.arange(offset, offset + len(rows), dtype=np.int64),
                           source_lines=np.asarray(indices, dtype=np.int64))

    with open(path, encoding=header["encoding"], errors="strict", newline="") as stream:
        for physical_line, row in enumerate(csv.reader(stream), 1):
            if not row or row[0].strip() != "<ChannelData>":
                continue
            row = tokens(row)[1:]
            ok = len(row) == width
            if not ok and len(issues) < 100:
                issues.append(f"行 {physical_line}：期望 {width} 列，实际 {len(row)}")
            try:
                values = [float(value) for value in row[:width]]
            except ValueError:
                values = []
                for value in row[:width]:
                    try:
                        values.append(float(value))
                    except ValueError:
                        values.append(float("nan"))
            values.extend([float("nan")] * (width - len(values)))
            rows.append(values)
            indices.append(physical_line)
            row_valid.append(ok)
            if len(rows) >= chunk_size:
                checkpoint(cancel)
                yield build()
                offset += len(rows)
                rows, indices, row_valid, issues = [], [], [], []
        if rows:
            checkpoint(cancel)
            yield build()


def read_measurement(path, *, sample_period_s=None, cancel: Cancel = None):
    chunks = list(iter_measurement(path, sample_period_s=sample_period_s, cancel=cancel))
    if not chunks:
        raise ValueError("没有 ChannelData 样本")
    result = Measurement(str(path), np.concatenate([c.time_s for c in chunks]),
                         {k: np.concatenate([c.values[k] for c in chunks]) for k in chunks[0].values},
                         np.concatenate([c.valid for c in chunks]), dict(chunks[0].metadata),
                         source_indices=np.concatenate([c.source_indices for c in chunks]),
                         source_lines=np.concatenate([c.source_lines for c in chunks]))
    result.metadata["issues"] = [v for c in chunks for v in c.metadata["issues"]][:100]
    result.metadata["invalid_samples"] = int((~result.valid).sum())
    result.metadata["duplicate_conflicts"] = {k: sum(c.metadata["duplicate_conflicts"].get(k, 0) for c in chunks) for k in chunks[0].metadata["duplicate_channels"]}
    result.metadata.pop("source_lines", None)
    result.metadata["source_index_semantics"] = "source_indices: original zero-based ChannelData index; source_lines: original one-based CSV row"
    return result


def inspect_dataset(root, *, deep=False, cancel: Cancel = None):
    root = Path(root)
    if not root.is_dir():
        raise ValueError("实验数据目录不存在")
    files, rounds = [], {}
    mapping_path = root / "负载预测实验G代码及工件模型" / "90轮G代码映射.csv"
    experiment = {}
    if mapping_path.is_file():
        with mapping_path.open(encoding=detect_encoding(mapping_path), newline="") as stream:
            for row in csv.DictReader(stream):
                key = (int(row["轮次"].removeprefix("R")), row["工况代码"][0])
                experiment[key] = row
    process_candidates = {}
    process_root = root / "工艺信息文件"
    if process_root.is_dir():
        for p in process_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".txt", ".csv") and not p.name.startswith("旧"):
                for feature in ("外形", "曲面", "混合", "斜面", "台阶"):
                    if feature in str(p.relative_to(process_root)):
                        process_candidates.setdefault(feature, []).append(str(p))
    data_dirs = [root / name for name in ("空载数据采集", "实际加工采样数据") if (root / name).is_dir()]
    paths = [path for directory in (data_dirs or [root]) for path in directory.rglob("*")]
    for path in sorted(paths):
        if path.suffix.lower() != ".csv":
            continue
        checkpoint(cancel)
        item = {"path": str(path), "status": "header_checked", "samples": None}
        match = re.search(r"_R(\d+)_", path.name, re.I)
        if match:
            round_id = int(match[1])
            feature = next((f for f in ("外形", "曲面", "混合", "斜面", "台阶") if f in path.name), "未知")
            rounds.setdefault(round_id, []).append(feature)
            item.update(round_id=round_id, feature=feature)
            code = {"外形": "O", "曲面": "C", "混合": "M", "斜面": "S", "台阶": "T"}.get(feature)
            expected = experiment.get((round_id, code))
            if expected:
                nc_path = mapping_path.parent / expected["NC文件夹"] / expected["NC文件"]
                stamp = re.search(r"_(S\dT\d)_F(\d+)S(\d+)", path.stem)
                item["companions"] = dict(nc_path=str(nc_path), nc_exists=nc_path.is_file(),
                    process_candidates=process_candidates.get(feature, []), parameter_group=expected["参数组"],
                    expected_combination=expected["组合"],
                    filename_matches=bool(stamp and stamp[1] == expected["组合"] and float(stamp[2]) == float(expected["进给速度"]) and float(stamp[3]) == float(expected["主轴转速"])))
            else:
                item["companions"] = {"status": "mapping_missing"}
        try:
            header = read_header(path)
            item["channels"] = list(header["mapping"])
            item["duplicate_channels"] = header["duplicate_channels"]
            item["position_channels"] = select_position_channels(header["mapping"])[0] or "none"
            missing = {"spindle_power_w", "spindle_speed_rpm"} - header["mapping"].keys()
            if missing:
                raise ValueError(f"缺少功率或转速通道：{sorted(missing)}")
            if deep:
                count, invalid, conflicts = 0, 0, {}
                for chunk in iter_measurement(path, cancel=cancel):
                    count += len(chunk.time_s)
                    invalid += int((~chunk.valid).sum())
                    for k, v in chunk.metadata["duplicate_conflicts"].items():
                        conflicts[k] = conflicts.get(k, 0) + v
                item.update(samples=count, invalid_samples=invalid,
                            duplicate_conflicts=conflicts, status="checked" if count and not invalid else "invalid")
        except ValueError as exc:
            item.update(status="error", error=str(exc))
        files.append(item)
    required = {"外形", "曲面", "混合", "斜面", "台阶"}
    return {"rule_version": RULE_VERSION, "files": files, "csv_count": len(files),
            "round_count": len(rounds), "inspection": "full" if deep else "headers_only",
            "missing_rounds": sorted(set(range(1, 91)) - rounds.keys()),
            "missing_features": {str(r): sorted(required - set(fs)) for r, fs in rounds.items() if required - set(fs)},
            "duplicate_features": {str(r): sorted({f for f in fs if fs.count(f) > 1}) for r, fs in rounds.items() if len(fs) != len(set(fs))}}
