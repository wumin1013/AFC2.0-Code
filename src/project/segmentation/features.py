from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

from .schemas import AtomicSegment, PathDiagnostics, SegmentationConfig


STANDARD_INPUT_COLUMNS = (
    "point_id",
    "source_index",
    "s",
    "path_start",
    "path_end",
    "path_source",
    "path_is_physical",
    "line_id",
    "line_no_raw",
    "N_str",
    "line_point_index",
    "point_label",
    "ap",
    "ae",
    "F_program",
    "P_pred",
    "P_idle",
    "power_gate_valid",
)


def _as_frame(input_frame) -> pd.DataFrame:
    if isinstance(input_frame, pd.DataFrame):
        return input_frame.copy(deep=False).reset_index(drop=True)
    if input_frame is None:
        return pd.DataFrame()
    if isinstance(input_frame, dict):
        return pd.DataFrame(input_frame)
    return pd.DataFrame.from_records(list(input_frame))


def _first_column(frame: pd.DataFrame, names: Sequence[str], default) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name].reset_index(drop=True)
    return pd.Series([default] * len(frame))


def _numeric(frame: pd.DataFrame, names: Sequence[str]) -> np.ndarray:
    values = _first_column(frame, names, np.nan)
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def _bool_value(value, default: bool) -> bool:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "false", "no", "n", "nonphysical", "non_physical"}:
            return False
        if normalized in {"1", "true", "yes", "y", "physical"}:
            return True
    return bool(value)


def _source_metadata(frame: pd.DataFrame, fallback: str) -> Tuple[str, bool]:
    source = fallback
    if "path_source" in frame.columns:
        for value in frame["path_source"]:
            text = str(value or "").strip()
            if text and text.lower() not in {"nan", "none"}:
                source = text
                break
    is_physical = not any(token in source.lower() for token in ("fallback", "nonphysical", "non_physical"))
    if "path_is_physical" in frame.columns:
        present = [value for value in frame["path_is_physical"] if not pd.isna(value)]
        if present:
            is_physical = all(_bool_value(value, is_physical) for value in present)
    return source, bool(is_physical)


def _validate_cumulative(values: np.ndarray, tolerance: float) -> Tuple[bool, np.ndarray, str]:
    if len(values) == 0:
        return False, values, "缺少累计行程"
    if not np.all(np.isfinite(values)):
        return False, values, "累计行程包含非有限值"
    if values[0] < -tolerance:
        return False, values, "累计行程起点为负"
    diffs = np.diff(values)
    regression_tolerance = tolerance * np.maximum.reduce(
        (
            np.ones(len(diffs), dtype=float),
            np.abs(values[:-1]),
            np.abs(values[1:]),
        )
    )
    if np.any(diffs < -regression_tolerance):
        return False, values, "累计行程非单调不减"
    corrected = np.maximum.accumulate(np.maximum(values, 0.0))
    # 显式累计值的语义是“从程序起点累计”。因此首个值本身也可能
    # 对应第一条记录的正行程，物理总跨度应从 0 计算，而不是减去首值。
    span = float(corrected[-1])
    if span <= tolerance:
        return False, corrected, "累计行程无有效正跨度（包括全零列）"
    return True, corrected, ""


def _validate_incremental(values: np.ndarray, tolerance: float) -> Tuple[bool, np.ndarray, str]:
    """校验逐点行程增量，并转换为内部累计坐标。"""
    if len(values) == 0:
        return False, values, "缺少行程增量"
    if not np.all(np.isfinite(values)):
        return False, values, "行程增量包含非有限值"
    value_tolerance = tolerance * np.maximum(1.0, np.abs(values))
    if np.any(values < -value_tolerance):
        return False, values, "行程增量包含负值"
    corrected = np.maximum(values, 0.0)
    cumulative = np.cumsum(corrected)
    if float(cumulative[-1]) <= tolerance:
        return False, cumulative, "行程增量无有效正总和（包括全零列）"
    return True, cumulative, ""


def _validate_bounds(
    starts: np.ndarray,
    ends: np.ndarray,
    tolerance: float,
) -> Tuple[bool, np.ndarray, np.ndarray, str]:
    if len(starts) == 0 or len(starts) != len(ends):
        return False, starts, ends, "缺少行程边界"
    if not (np.all(np.isfinite(starts)) and np.all(np.isfinite(ends))):
        return False, starts, ends, "行程边界包含非有限值"
    bound_tolerance = tolerance * np.maximum.reduce(
        (np.ones(len(starts), dtype=float), np.abs(starts), np.abs(ends))
    )
    if np.any(ends < starts - bound_tolerance):
        return False, starts, ends, "存在终点早于起点的行程边界"
    start_sequence_tolerance = tolerance * np.maximum.reduce(
        (
            np.ones(max(len(starts) - 1, 0), dtype=float),
            np.abs(starts[:-1]),
            np.abs(starts[1:]),
        )
    )
    end_sequence_tolerance = tolerance * np.maximum.reduce(
        (
            np.ones(max(len(ends) - 1, 0), dtype=float),
            np.abs(ends[:-1]),
            np.abs(ends[1:]),
        )
    )
    if (
        np.any(np.diff(starts) < -start_sequence_tolerance)
        or np.any(np.diff(ends) < -end_sequence_tolerance)
    ):
        return False, starts, ends, "行程边界非单调不减"
    corrected_starts = np.maximum.accumulate(np.maximum(starts, 0.0))
    corrected_ends = np.maximum.accumulate(np.maximum(ends, corrected_starts))
    span = float(corrected_ends[-1] - corrected_starts[0])
    if span <= tolerance:
        return False, corrected_starts, corrected_ends, "行程边界无有效正跨度"
    return True, corrected_starts, corrected_ends, ""


def standardize_input(input_frame, config: SegmentationConfig) -> Tuple[pd.DataFrame, PathDiagnostics]:
    """
    只读取 alpha 允许的工艺字段及追溯元数据，并优先使用有效累计行程。

    G 代码/NC 几何计算属于上游职责；本函数只接收上游已生成的
    ``path_start``/``path_end``。两种物理来源都不可用时才进入集中配置的顺序回退。
    """

    source = _as_frame(input_frame)
    count = len(source)
    if count == 0:
        empty = pd.DataFrame(columns=STANDARD_INPUT_COLUMNS)
        diagnostics = PathDiagnostics(
            source="empty",
            is_valid=False,
            is_physical=False,
            used_nonphysical_fallback=False,
            span_mm=0.0,
            reason="输入为空",
        )
        return empty, diagnostics

    tolerance = float(config.path_tolerance_mm)
    # processing_core 与 ProcessInfo.csv 中 ``s``/``s(mm)`` 是单点段长，
    # 不是累计位置。因此先读显式累计列；没有显式累计列时，将 s 行程
    # 增量转换为内部累计坐标。直接传入 processing_core rows 时已有
    # ``path_cumulative``，仍可保持上游生成的精确边界。
    explicit_cumulative_names = tuple(
        name
        for name in ("input_path_cumulative", "s_cumulative", "path_cumulative")
        if name in source.columns
    )
    cumulative_valid = False
    cumulative = np.full(count, np.nan, dtype=float)
    cumulative_origin = ""
    cumulative_reasons = []
    path_value_semantics = "cumulative"
    if explicit_cumulative_names:
        for candidate_name in explicit_cumulative_names:
            candidate_raw = _numeric(source, (candidate_name,))
            candidate_valid, candidate_values, candidate_reason = _validate_cumulative(
                candidate_raw,
                tolerance,
            )
            if candidate_valid:
                cumulative_valid = True
                cumulative = candidate_values
                cumulative_origin = candidate_name
                break
            cumulative_reasons.append(f"{candidate_name}: {candidate_reason}")
    else:
        cumulative_reasons.append("缺少显式累计行程")

    # 显式累计列存在但无效时，仍须继续尝试契约明确为增量语义的 s/s(mm)。
    # 两类输入行程均无效后，才允许进入上游边界或顺序回退。
    if not cumulative_valid:
        incremental_names = tuple(
            name for name in ("s", "s(mm)") if name in source.columns
        )
        if incremental_names:
            for candidate_name in incremental_names:
                candidate_raw = _numeric(source, (candidate_name,))
                candidate_valid, candidate_values, candidate_reason = _validate_incremental(
                    candidate_raw,
                    tolerance,
                )
                if candidate_valid:
                    cumulative_valid = True
                    cumulative = candidate_values
                    cumulative_origin = candidate_name
                    path_value_semantics = "incremental"
                    break
                cumulative_reasons.append(f"{candidate_name}: {candidate_reason}")
        else:
            cumulative_reasons.append("缺少行程增量 s/s(mm)")
    cumulative_reason = "；".join(cumulative_reasons)
    starts_raw = _numeric(source, ("path_start",))
    ends_raw = _numeric(source, ("path_end",))
    bounds_valid, starts, ends, bounds_reason = _validate_bounds(starts_raw, ends_raw, tolerance)

    if cumulative_valid:
        path_end = cumulative
        path_start = np.empty(count, dtype=float)
        path_start[0] = 0.0
        path_start[1:] = cumulative[:-1]
        fallback_source = "input_incremental" if path_value_semantics == "incremental" else "input_cumulative"
        source_name, path_is_physical = _source_metadata(source, fallback_source)
        source_is_incremental = "increment" in str(source_name).lower()
        if path_value_semantics == "incremental" or source_is_incremental:
            origin_label = "s(mm)" if source_is_incremental else cumulative_origin
            reason = f"优先使用有效输入行程增量并累计（{origin_label}）"
        else:
            reason = f"优先使用有效累计行程（{cumulative_origin}）"
        used_fallback = False
    elif bounds_valid:
        path_start = starts
        path_end = ends
        source_name, path_is_physical = _source_metadata(source, "provided_path_bounds")
        reason = f"输入行程不可用（{cumulative_reason}），使用上游物理行程边界"
        used_fallback = not path_is_physical
    else:
        step = float(config.sequential_fallback_step_mm)
        path_start = np.arange(count, dtype=float) * step
        path_end = path_start + step
        source_name = "sequential_fallback"
        path_is_physical = False
        used_fallback = True
        reason = (
            f"输入行程不可用（{cumulative_reason}），"
            f"且上游行程边界不可用（{bounds_reason}）"
        )

    ap_raw = _numeric(source, ("ap", "ap(mm)"))
    ae_raw = _numeric(source, ("ae", "ae(mm)"))
    feed_raw = _numeric(source, ("F_program", "feed_effective", "F(mm/min)"))
    predicted_power_names = ("P_pred", "predicted_load", "P")
    idle_power_names = ("P_idle", "predicted_idle_power")
    if not any(name in source.columns for name in predicted_power_names):
        raise ValueError("六态划分缺少预测负载 P_pred，无法执行 idle 功率硬门控")
    if not any(name in source.columns for name in idle_power_names):
        raise ValueError("六态划分缺少预测空载功率 P_idle，无法执行 idle 功率硬门控")
    predicted_power = _numeric(source, predicted_power_names)
    idle_power = _numeric(source, idle_power_names)
    power_gate_valid = np.isfinite(predicted_power) & np.isfinite(idle_power)
    if not np.all(power_gate_valid):
        invalid_count = int(np.sum(~power_gate_valid))
        raise ValueError(
            f"P_pred/P_idle 含 {invalid_count} 个非有限值，idle 功率硬门控已安全停止"
        )
    invalid_process_value = (
        ~np.isfinite(ap_raw)
        | ~np.isfinite(ae_raw)
        | ~np.isfinite(feed_raw)
        | (ap_raw < 0.0)
        | (ae_raw < 0.0)
        | (feed_raw < 0.0)
    )
    ap = np.where(np.isfinite(ap_raw) & (ap_raw >= 0.0), ap_raw, 0.0)
    ae = np.where(np.isfinite(ae_raw) & (ae_raw >= 0.0), ae_raw, 0.0)
    feed = np.where(np.isfinite(feed_raw) & (feed_raw >= 0.0), feed_raw, 0.0)

    raw_line_id = _numeric(source, ("line_id", "line_no_aligned", "line_no_raw"))
    line_series = pd.Series(raw_line_id).ffill().bfill()
    missing_line = ~np.isfinite(line_series.to_numpy(dtype=float))
    line_values = line_series.to_numpy(dtype=float)
    line_values[missing_line] = np.arange(count, dtype=float)[missing_line]
    line_id = np.rint(line_values).astype(int)

    line_no_raw = _numeric(source, ("line_no_raw",))
    n_str = _first_column(source, ("N_str",), "").fillna("").astype(str).to_numpy(dtype=object)

    point_values = _first_column(source, ("point_id",), None)
    if point_values.isna().any() or point_values.astype(str).duplicated().any():
        point_id = np.arange(1, count + 1, dtype=int)
    else:
        point_id = point_values.to_numpy(copy=True)

    source_index_raw = _numeric(source, ("source_index",))
    source_index_rounded = np.rint(source_index_raw)
    source_index_valid = bool(
        np.all(np.isfinite(source_index_raw))
        and np.all(source_index_rounded >= 0.0)
        and np.all(np.abs(source_index_raw - source_index_rounded) <= 1e-9)
        and len(np.unique(source_index_rounded)) == count
    )
    source_index = (
        source_index_rounded.astype(int)
        if source_index_valid
        else np.arange(count, dtype=int)
    )

    line_point_index = np.zeros(count, dtype=int)
    within_line = 0
    for idx in range(count):
        if idx == 0 or line_id[idx] != line_id[idx - 1]:
            within_line = 0
        line_point_index[idx] = within_line
        within_line += 1

    display_line = np.where(np.isfinite(line_no_raw), np.rint(line_no_raw), line_id).astype(int)
    point_label = np.asarray(
        [f"{int(display_line[idx])}.{int(line_point_index[idx]) + 1}" for idx in range(count)],
        dtype=object,
    )

    standardized = pd.DataFrame(
        {
            "point_id": point_id,
            "source_index": source_index,
            "s": path_end,
            "path_start": path_start,
            "path_end": path_end,
            "path_source": np.full(count, source_name, dtype=object),
            "path_is_physical": np.full(count, bool(path_is_physical), dtype=bool),
            "line_id": line_id,
            "line_no_raw": line_no_raw,
            "N_str": n_str,
            "line_point_index": line_point_index,
            "point_label": point_label,
            "ap": ap,
            "ae": ae,
            "F_program": feed,
            "P_pred": predicted_power,
            "P_idle": idle_power,
            "power_gate_valid": power_gate_valid,
            "input_invalid": invalid_process_value,
        }
    )
    diagnostics = PathDiagnostics(
        source=source_name,
        is_valid=True,
        is_physical=bool(path_is_physical),
        used_nonphysical_fallback=bool(used_fallback),
        span_mm=float(path_end[-1] - path_start[0]),
        input_cumulative_valid=bool(
            cumulative_valid
            and path_value_semantics == "cumulative"
            and "increment" not in str(source_name).strip().lower()
        ),
        input_incremental_valid=bool(
            cumulative_valid
            and (
                path_value_semantics == "incremental"
                or "increment" in str(source_name).strip().lower()
            )
        ),
        input_bounds_valid=bool(bounds_valid),
        reason=reason,
    )
    return standardized, diagnostics


def _window_bounds(x: np.ndarray, window_size: float) -> Tuple[np.ndarray, np.ndarray]:
    half_window = max(float(window_size), 0.0) / 2.0
    left = np.searchsorted(x, x - half_window, side="left")
    right = np.searchsorted(x, x + half_window, side="right")
    return left.astype(int), right.astype(int)


def _prefix(values: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(values, dtype=float)))


def _local_statistics(
    x: np.ndarray,
    values: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    tolerance: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.maximum(right - left, 1).astype(float)
    sum_y = _prefix(values)
    sum_y2 = _prefix(values * values)
    sum_x = _prefix(x)
    sum_x2 = _prefix(x * x)
    sum_xy = _prefix(x * values)
    sy = sum_y[right] - sum_y[left]
    sy2 = sum_y2[right] - sum_y2[left]
    sx = sum_x[right] - sum_x[left]
    sx2 = sum_x2[right] - sum_x2[left]
    sxy = sum_xy[right] - sum_xy[left]
    mean = sy / counts
    variance = np.maximum(sy2 / counts - mean * mean, 0.0)
    denominator = counts * sx2 - sx * sx
    trend = np.zeros(len(values), dtype=float)
    valid = np.abs(denominator) > tolerance
    trend[valid] = (counts[valid] * sxy[valid] - sx[valid] * sy[valid]) / denominator[valid]
    return mean, np.sqrt(variance), trend


def _local_relative_path_slope(
    path_x: np.ndarray,
    values: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    absolute_floor: float,
    tolerance: float,
) -> np.ndarray:
    """返回局部线性拟合在实际行程跨度上的相对 MRR 漂移。"""

    counts = np.maximum(right - left, 1).astype(float)
    sum_y = _prefix(values)
    sum_x = _prefix(path_x)
    sum_x2 = _prefix(path_x * path_x)
    sum_xy = _prefix(path_x * values)
    sy = sum_y[right] - sum_y[left]
    sx = sum_x[right] - sum_x[left]
    sx2 = sum_x2[right] - sum_x2[left]
    sxy = sum_xy[right] - sum_xy[left]
    denominator = counts * sx2 - sx * sx
    slope = np.zeros(len(values), dtype=float)
    valid = np.abs(denominator) > tolerance
    slope[valid] = (counts[valid] * sxy[valid] - sx[valid] * sy[valid]) / denominator[valid]
    mean = sy / counts
    last_indices = np.maximum(right - 1, left)
    span = np.maximum(path_x[last_indices] - path_x[left], 0.0)
    scale = np.maximum(np.abs(mean), max(float(absolute_floor), 1e-12))
    return np.abs(slope) * span / scale


def _build_machining_segment_features(
    machining_active: np.ndarray,
    non_idle: np.ndarray,
    mrr: np.ndarray,
) -> dict[str, np.ndarray]:
    """按有效 idle 重置边界切分加工段，并建立独立 MRR 峰值参照。

    ``machining_active`` 会跨越不具备重置资格的短 idle 脉冲，但
    ``non_idle`` 仍保留功率门控真值。因此短脉冲前后共享加工段和峰值，
    而 idle 点自身仍只能被解码为 idle。此时还没有合法稳态锚点，
    entry/exit 先建立完整峰值阶段，之后由评分器收紧。
    """

    count = len(non_idle)
    segment_id = np.zeros(count, dtype=np.int32)
    segment_start = np.full(count, -1, dtype=np.int32)
    segment_end = np.full(count, -1, dtype=np.int32)
    peak_idx = np.full(count, -1, dtype=np.int32)
    first_peak_idx = np.full(count, -1, dtype=np.int32)
    last_peak_idx = np.full(count, -1, dtype=np.int32)
    peak_mrr = np.full(count, np.nan, dtype=float)
    relative_position = np.zeros(count, dtype=float)
    phase = np.zeros(count, dtype=np.int8)
    first_steady_idx = np.full(count, -1, dtype=np.int32)
    last_steady_idx = np.full(count, -1, dtype=np.int32)
    entry_eligible = np.zeros(count, dtype=bool)
    exit_eligible = np.zeros(count, dtype=bool)

    starts = np.flatnonzero(
        machining_active & np.concatenate(([True], ~machining_active[:-1]))
    )
    ends = np.flatnonzero(
        machining_active & np.concatenate((~machining_active[1:], [True]))
    )
    for current_id, (start_idx, end_idx) in enumerate(zip(starts, ends), 1):
        start_idx = int(start_idx)
        end_idx = int(end_idx)
        indices = np.arange(start_idx, end_idx + 1, dtype=int)
        cutting_indices = indices[non_idle[indices]]
        if cutting_indices.size == 0:
            continue
        local_mrr = mrr[cutting_indices]
        peak_value = float(np.max(local_mrr))
        peak_tolerance = max(abs(peak_value) * 1e-12, 1e-12)
        local_peak_offsets = np.flatnonzero(
            np.abs(local_mrr - peak_value) <= peak_tolerance
        )
        # 同高峰或平台峰值分别保留首末位置，避免最早峰值把振荡尾部
        # 全部纳入 exit，给中间高波动区留下 nonsteady 候选空间。
        current_first_peak_idx = int(cutting_indices[local_peak_offsets[0]])
        current_last_peak_idx = int(cutting_indices[local_peak_offsets[-1]])
        segment_id[indices] = current_id
        segment_start[indices] = start_idx
        segment_end[indices] = end_idx
        peak_idx[indices] = current_first_peak_idx
        first_peak_idx[indices] = current_first_peak_idx
        last_peak_idx[indices] = current_last_peak_idx
        peak_mrr[indices] = peak_value
        if end_idx > start_idx:
            relative_position[indices] = (
                (indices - start_idx).astype(float) / float(end_idx - start_idx)
            )
        phase[indices[indices < current_first_peak_idx]] = -1
        phase[indices[indices > current_last_peak_idx]] = 1
        entry_end = current_first_peak_idx
        if entry_end >= start_idx:
            entry_eligible[start_idx:entry_end + 1] = True

        exit_start = current_last_peak_idx
        if exit_start <= end_idx:
            exit_eligible[exit_start:end_idx + 1] = True

    return {
        "machining_segment_id": segment_id,
        "machining_segment_start_idx": segment_start,
        "machining_segment_end_idx": segment_end,
        "machining_segment_peak_idx": peak_idx,
        "machining_segment_first_peak_idx": first_peak_idx,
        "machining_segment_last_peak_idx": last_peak_idx,
        "machining_segment_peak_mrr": peak_mrr,
        "machining_segment_relative_position": relative_position,
        "machining_phase": phase,
        "machining_first_steady_idx": first_steady_idx,
        "machining_last_steady_idx": last_steady_idx,
        "entry_phase_eligible": entry_eligible,
        "exit_phase_eligible": exit_eligible,
    }


def _idle_reset_features(
    idle_gate: np.ndarray,
    path_start: np.ndarray,
    path_end: np.ndarray,
    config: SegmentationConfig,
) -> dict[str, np.ndarray]:
    """区分 idle 硬标签与可以重置进/退刀阶段的 idle 区间。"""

    count = len(idle_gate)
    qualified = np.zeros(count, dtype=bool)
    run_id = np.zeros(count, dtype=np.int32)
    run_length = np.zeros(count, dtype=float)
    starts = np.flatnonzero(idle_gate & np.concatenate(([True], ~idle_gate[:-1])))
    ends = np.flatnonzero(idle_gate & np.concatenate((~idle_gate[1:], [True])))
    minimum = float(config.min_idle_reset_mm)
    tolerance = float(config.path_tolerance_mm)
    for current_id, (start_idx, end_idx) in enumerate(zip(starts, ends), 1):
        start_idx = int(start_idx)
        end_idx = int(end_idx)
        length_mm = max(float(path_end[end_idx] - path_start[start_idx]), 0.0)
        sample = slice(start_idx, end_idx + 1)
        run_id[sample] = current_id
        run_length[sample] = length_mm
        # 行程首尾的 idle 天然是工件边界；内部 idle 则必须
        # 达到集中配置的物理长度，才能重置 entry/exit。
        if (
            start_idx == 0
            or end_idx == count - 1
            or length_mm >= minimum - tolerance
        ):
            qualified[sample] = True

    non_idle = ~idle_gate
    entry_boundary = non_idle & np.concatenate(([False], qualified[:-1]))
    exit_boundary = non_idle & np.concatenate((qualified[1:], [False]))
    return {
        "idle_reset_run_id": run_id,
        "idle_reset_run_length_mm": run_length,
        "is_idle_reset_qualified": qualified,
        "entry_boundary_candidate": entry_boundary,
        "exit_boundary_candidate": exit_boundary,
        # 非合格内部 idle 不结束当前加工阶段，但仍保留
        # is_idle_gate=True，不会被任何非 idle 候选跨越。
        "machining_phase_active": ~qualified,
    }


def compute_point_features(frame: pd.DataFrame, config: SegmentationConfig) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    features = frame.copy()
    point_position = np.arange(len(features), dtype=float)
    ap = features["ap"].to_numpy(dtype=float)
    ae = features["ae"].to_numpy(dtype=float)
    feed = features["F_program"].to_numpy(dtype=float)
    mrr = ap * ae * feed / 60.0
    cutting = mrr > config.mrr_cutting_epsilon
    predicted_power = features["P_pred"].to_numpy(dtype=float)
    idle_power = features["P_idle"].to_numpy(dtype=float)
    power_gate_valid = (
        features["power_gate_valid"].to_numpy(dtype=bool)
        & np.isfinite(predicted_power)
        & np.isfinite(idle_power)
    )
    idle_gate = power_gate_valid & (
        predicted_power <= idle_power + float(config.idle_power_tolerance)
    )
    non_idle = ~idle_gate

    features["MRR_program"] = mrr
    features["is_effective_cutting"] = cutting
    features["power_gate_valid"] = power_gate_valid
    features["is_idle_gate"] = idle_gate
    features["is_non_idle"] = non_idle
    reset_features = _idle_reset_features(
        idle_gate,
        features["path_start"].to_numpy(dtype=float),
        features["path_end"].to_numpy(dtype=float),
        config,
    )
    for name, values in reset_features.items():
        features[name] = values
    left, right = _window_bounds(point_position, config.local_window_points)
    mrr_local_mean, mrr_local_std, mrr_local_trend = _local_statistics(
        point_position,
        mrr,
        left,
        right,
        np.finfo(float).eps,
    )
    features["MRR_program_local_mean"] = mrr_local_mean
    features["MRR_program_local_std"] = mrr_local_std
    features["MRR_program_local_trend"] = mrr_local_trend
    path_center = (
        features["path_start"].to_numpy(dtype=float)
        + features["path_end"].to_numpy(dtype=float)
    ) / 2.0
    mrr_local_relative_slope = _local_relative_path_slope(
        path_center,
        mrr,
        left,
        right,
        max(float(config.mrr_cutting_epsilon), 1e-12),
        max(float(config.path_tolerance_mm), np.finfo(float).eps),
    )
    features["MRR_program_local_relative_slope"] = mrr_local_relative_slope
    mrr_relative_std = mrr_local_std / np.maximum(
        np.abs(mrr_local_mean),
        max(float(config.mrr_cutting_epsilon), 1e-12),
    )
    steady_point_candidate = (
        non_idle
        & cutting
        & (mrr_relative_std <= float(config.steady_mrr_relative_std_max))
        & (
            mrr_local_relative_slope
            <= float(config.steady_mrr_relative_slope_max)
        )
    )
    features["MRR_program_local_relative_std"] = mrr_relative_std
    features["steady_point_candidate"] = steady_point_candidate
    trend_threshold = config.mrr_trend_relative_per_point * np.maximum(np.abs(mrr_local_mean), 1e-9)
    mrr_trend_sign = np.zeros(len(features), dtype=np.int8)
    mrr_trend_sign[mrr_local_trend > trend_threshold] = 1
    mrr_trend_sign[mrr_local_trend < -trend_threshold] = -1

    features["MRR_trend_sign"] = mrr_trend_sign
    for name, values in _build_machining_segment_features(
        features["machining_phase_active"].to_numpy(dtype=bool),
        non_idle,
        mrr,
    ).items():
        features[name] = values
    return features


def build_atomic_segments(frame: pd.DataFrame, config: SegmentationConfig) -> Tuple[AtomicSegment, ...]:
    if frame.empty:
        return tuple()
    count = len(frame)
    # transition 的比例边界以有效 ProcessInfo 点数定义，因此原子边界
    # 必须与输入点一一对应。这里只细化已有点，不插值也不制造工艺点。
    starts = np.arange(count, dtype=int)
    ends = starts.copy()
    cutting = frame["is_effective_cutting"].to_numpy(dtype=bool)
    idle_gate = frame["is_idle_gate"].to_numpy(dtype=bool)
    machining_segment_id = frame["machining_segment_id"].to_numpy(dtype=np.int32)
    trend_sign = frame["MRR_trend_sign"].to_numpy(dtype=np.int8)
    mrr = frame["MRR_program"].to_numpy(dtype=float)
    path_start = frame["path_start"].to_numpy(dtype=float)
    path_end = frame["path_end"].to_numpy(dtype=float)
    line_id = frame["line_id"].to_numpy(dtype=int)
    atoms = []
    for atom_id, (start_idx, end_idx) in enumerate(zip(starts, ends), 1):
        sample = slice(int(start_idx), int(end_idx) + 1)
        atoms.append(
            AtomicSegment(
                atom_id=int(atom_id),
                start_idx=int(start_idx),
                end_idx=int(end_idx),
                start_s=float(path_start[start_idx]),
                end_s=float(path_end[end_idx]),
                length_mm=max(float(path_end[end_idx] - path_start[start_idx]), 0.0),
                start_line_id=int(line_id[start_idx]),
                end_line_id=int(line_id[end_idx]),
                point_count=int(end_idx - start_idx + 1),
                cutting_fraction=float(np.mean(cutting[sample])),
                mrr_mean=float(np.mean(mrr[sample])),
                mrr_std=float(np.std(mrr[sample], ddof=0)),
                mrr_trend_sign=int(np.sign(np.mean(trend_sign[sample]))),
                idle_fraction=float(np.mean(idle_gate[sample])),
                non_idle_fraction=float(np.mean(~idle_gate[sample])),
                machining_segment_id=int(machining_segment_id[start_idx]),
            )
        )
    return tuple(atoms)
