#!/usr/bin/env python
"""用 ProcessInfo 文本检查六态划分的完整性和批次无关性。"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd

from project.segmentation import SegmentationConfig, SegmentationPipeline


PROCESS_COLUMNS = (
    "source_index",
    "line_no_raw",
    "ap",
    "ae",
    "F_program",
    "s",
)
DEFAULT_BATCH_SIZES = (32, 48, 64)
STRUCTURAL_DIAGNOSTICS = (
    "coverage_rate",
    "gap_count",
    "overlap_count",
    "illegal_transition_count",
    "postprocess_illegal_transition_count",
    "postprocess_validation_passed",
    "fallback_used",
    "fallback_scope",
    "fallback_validated",
    "entry_missing_interval_count",
    "entry_peak_boundary_violation_count",
    "transition_overlaps_entry_count",
    "steady_precedence_entry_clipped_point_count",
    "transition_outside_steady_parent_count",
    "strict_transition_semantics_valid",
)


def _finite_number(value: str, *, path: Path, line_number: int, column: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(
            f"{path} 第 {line_number} 行的 {column} 不是数值: {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"{path} 第 {line_number} 行的 {column} 不是有限数")
    return number


def _path_increment(value: str, *, path: Path, line_number: int) -> float:
    """保留原表中的缺失行程，让标准化层按正式规则选择回退路径。"""

    normalized = value.strip().lower()
    missing_tokens = {
        "nan",
        "+nan",
        "-nan",
        "nan(ind)",
        "+nan(ind)",
        "-nan(ind)",
    }
    if normalized in missing_tokens:
        return float("nan")
    number = _finite_number(
        value,
        path=path,
        line_number=line_number,
        column="s",
    )
    if number < 0.0:
        raise ValueError(f"{path} 第 {line_number} 行的 s 为负数")
    return number


def read_process_text(path: Path) -> pd.DataFrame:
    """读取固定六列：序号、程序行、ap、ae、F、单点行程 s。"""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            text = raw_line.strip()
            if not text:
                continue
            fields = text.split(maxsplit=6)
            if len(fields) < 6:
                raise ValueError(
                    f"{path} 第 {line_number} 行少于六列，无法按 ProcessInfo 固定格式解析"
                )
            source_index = _finite_number(
                fields[0], path=path, line_number=line_number, column="序号"
            )
            line_no_raw = _finite_number(
                fields[1], path=path, line_number=line_number, column="程序行"
            )
            ap = _finite_number(
                fields[2], path=path, line_number=line_number, column="ap"
            )
            ae = _finite_number(
                fields[3], path=path, line_number=line_number, column="ae"
            )
            feed = _finite_number(
                fields[4], path=path, line_number=line_number, column="F"
            )
            path_increment = _path_increment(
                fields[5], path=path, line_number=line_number
            )
            if any(value < 0.0 for value in (ap, ae, feed)):
                raise ValueError(f"{path} 第 {line_number} 行含负工艺量")
            rows.append(
                {
                    "source_index": int(round(source_index)),
                    "line_id": int(round(line_no_raw)),
                    "line_no_raw": int(round(line_no_raw)),
                    "ap": ap,
                    "ae": ae,
                    "F_program": feed,
                    "s": path_increment,
                }
            )
    if not rows:
        raise ValueError(f"{path} 没有可用工艺点")
    frame = pd.DataFrame.from_records(rows)
    frame.attrs["invalid_path_value_count"] = int(frame["s"].isna().sum())
    if frame["source_index"].duplicated().any():
        raise ValueError(f"{path} 的序号列存在重复值")
    return frame


def _selected_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    selected = {
        name: diagnostics[name]
        for name in STRUCTURAL_DIAGNOSTICS
        if name in diagnostics
    }
    path = dict(diagnostics.get("path") or {})
    if path:
        selected["path"] = {
            name: path.get(name)
            for name in (
                "source",
                "is_valid",
                "is_physical",
                "used_nonphysical_fallback",
                "reason",
            )
        }
    return selected


def _process_signature(result) -> str:
    diagnostics = dict(result.diagnostics or {})
    repeat = dict(diagnostics.get("repeat_run_consistency") or {})
    return str(
        diagnostics.get("process_signature")
        or repeat.get("process_signature")
        or repeat.get("input_signature")
        or ""
    )


def _state_counts(point_labels: pd.DataFrame) -> dict[str, int]:
    counts = point_labels["segment_type"].astype(str).value_counts()
    return {str(name): int(count) for name, count in sorted(counts.items())}


def _validate_run(result, required_states: Iterable[str]) -> list[str]:
    diagnostics = dict(result.diagnostics or {})
    failures: list[str] = []
    if not math.isclose(
        float(diagnostics.get("coverage_rate", 0.0)),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        failures.append("覆盖率不是 100%")
    for name in (
        "gap_count",
        "overlap_count",
        "illegal_transition_count",
        "postprocess_illegal_transition_count",
    ):
        if int(diagnostics.get(name, 0)) != 0:
            failures.append(f"{name}={diagnostics.get(name)}")
    if not bool(diagnostics.get("postprocess_validation_passed", False)):
        failures.append("后处理结构校验未通过")
    if bool(diagnostics.get("fallback_used", False)):
        fallback_scope = str(diagnostics.get("fallback_scope") or "")
        fallback_validated = bool(diagnostics.get("fallback_validated", False))
        if fallback_scope != "local_verified" or not fallback_validated:
            failures.append(
                f"使用了不可写入正式结果的回退: {fallback_scope or 'unknown'}"
            )
    for name in (
        "entry_missing_interval_count",
        "entry_peak_boundary_violation_count",
        "transition_overlaps_entry_count",
        "transition_outside_steady_parent_count",
    ):
        if name in diagnostics and int(diagnostics[name]) != 0:
            failures.append(f"{name}={diagnostics[name]}")
    if not bool(diagnostics.get("strict_transition_semantics_valid", False)):
        failures.append("稳态父平台内部过渡段结构校验未通过")
    labels = set(result.point_labels["segment_type"].astype(str))
    missing_states = sorted(set(required_states).difference(labels))
    if missing_states:
        failures.append(f"缺少要求的状态: {', '.join(missing_states)}")
    return failures


def verify_case(
    path: Path,
    batch_sizes: tuple[int, ...],
    required_states: tuple[str, ...],
) -> tuple[dict[str, Any], list[str]]:
    frame = read_process_text(path)
    base_config = SegmentationConfig()
    runs: list[tuple[int, Any, float]] = []
    failures: list[str] = []
    for batch_size in batch_sizes:
        config = replace(base_config, max_segment_atoms=int(batch_size))
        started = time.perf_counter()
        result = SegmentationPipeline(config).run(frame)
        elapsed = time.perf_counter() - started
        runs.append((batch_size, result, elapsed))
        failures.extend(
            f"max_segment_atoms={batch_size}: {message}"
            for message in _validate_run(result, required_states)
        )

    reference_size, reference, _ = runs[0]
    reference_signature = _process_signature(reference)
    if not reference_signature:
        failures.append("结果没有 process_signature")
    reference_labels = reference.point_labels[
        ["source_index", "segment_type", "state_code", "interval_id"]
    ].reset_index(drop=True)
    reference_intervals = reference.intervals[
        ["start_idx", "end_idx", "segment_type", "state_code"]
    ].reset_index(drop=True)
    for batch_size, result, _ in runs[1:]:
        candidate_labels = result.point_labels[
            ["source_index", "segment_type", "state_code", "interval_id"]
        ].reset_index(drop=True)
        candidate_intervals = result.intervals[
            ["start_idx", "end_idx", "segment_type", "state_code"]
        ].reset_index(drop=True)
        if not reference_labels.equals(candidate_labels):
            failures.append(
                f"max_segment_atoms={reference_size} 与 {batch_size} 的逐点标签不一致"
            )
        if not reference_intervals.equals(candidate_intervals):
            failures.append(
                f"max_segment_atoms={reference_size} 与 {batch_size} 的区间边界不一致"
            )
        if _process_signature(result) != reference_signature:
            failures.append(
                f"max_segment_atoms={reference_size} 与 {batch_size} 的过程签名不一致"
            )

    report = {
        "path": str(path),
        "process_point_count": int(len(frame)),
        "invalid_path_value_count": int(
            frame.attrs.get("invalid_path_value_count", 0)
        ),
        "passed": not failures,
        "runs": [
            {
                "max_segment_atoms": int(batch_size),
                "elapsed_seconds": round(float(elapsed), 6),
                "interval_count": int(len(result.intervals)),
                "process_signature": _process_signature(result),
                "state_point_counts": _state_counts(result.point_labels),
                "diagnostics": _selected_diagnostics(dict(result.diagnostics or {})),
            }
            for batch_size, result, elapsed in runs
        ],
        "failures": failures,
    }
    return report, failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="验证 ProcessInfo 六态划分覆盖率、结构约束和 32/48/64 批次一致性。"
    )
    parser.add_argument(
        "process_files",
        nargs="+",
        type=Path,
        help="固定列格式的 ProcessInfo .txt 文件，可传入多个。",
    )
    parser.add_argument(
        "--max-segment-atoms",
        nargs="+",
        type=int,
        default=list(DEFAULT_BATCH_SIZES),
        metavar="N",
        help="待比较的候选评分批次大小，默认 32 48 64。",
    )
    parser.add_argument(
        "--require-states",
        default="",
        help="要求出现的状态，使用逗号分隔，例如 entry,steady,transition,exit。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    batch_sizes = tuple(dict.fromkeys(int(value) for value in args.max_segment_atoms))
    if not batch_sizes or any(value < 1 for value in batch_sizes):
        print("max_segment_atoms 必须是正整数", file=sys.stderr)
        return 2
    required_states = tuple(
        value.strip()
        for value in str(args.require_states).split(",")
        if value.strip()
    )
    allowed_states = {"idle", "entry", "steady", "transition", "nonsteady", "exit"}
    unknown_states = sorted(set(required_states).difference(allowed_states))
    if unknown_states:
        print(f"未知状态: {', '.join(unknown_states)}", file=sys.stderr)
        return 2

    failed = False
    for raw_path in args.process_files:
        path = raw_path.expanduser().resolve()
        try:
            if not path.is_file():
                raise FileNotFoundError(f"文件不存在: {path}")
            report, failures = verify_case(path, batch_sizes, required_states)
            failed = failed or bool(failures)
        except Exception as exc:
            failed = True
            report = {
                "path": str(path),
                "passed": False,
                "failures": [f"{type(exc).__name__}: {exc}"],
            }
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
