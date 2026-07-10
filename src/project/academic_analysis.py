from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


F_COLUMN_ALIASES = [
    "F",
    "f",
    "feed",
    "feed_rate",
    "feedrate",
    "进给",
    "进给率",
    "进给速度",
]
P_COLUMN_ALIASES = [
    "P",
    "p",
    "power",
    "load",
    "load_power",
    "actual_load",
    "实际负载",
    "功率",
    "主轴功率",
]
PRED_COLUMN_ALIASES = [
    "predicted_load",
    "prediction",
    "posterior_load",
    "后验负载",
    "预测负载",
    "预测功率",
]


@dataclass(frozen=True)
class AcademicInterval:
    interval_id: str
    start_idx: int
    end_idx: int
    point_count: int
    duration_ms: int
    duration_s: float
    baseline_mean_f: float
    opt1_mean_f: float
    delta_f: float


def normalize_column_name(name: object) -> str:
    return re.sub(r"[\s\-_()/\\\[\]{}%]+", "", str(name or "").strip().lower())


def infer_matching_column(df: pd.DataFrame, aliases: Sequence[str], fallback_index: Optional[int] = None) -> Optional[str]:
    normalized_aliases = [normalize_column_name(alias) for alias in aliases]
    normalized_map = {col: normalize_column_name(col) for col in df.columns}

    for col, normalized in normalized_map.items():
        if normalized in normalized_aliases:
            return str(col)

    for col, normalized in normalized_map.items():
        for alias in normalized_aliases:
            if alias and alias in normalized:
                return str(col)

    numeric_candidates = []
    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        if int(series.notna().sum()) >= 3:
            numeric_candidates.append(str(col))
    if fallback_index is not None and len(numeric_candidates) > fallback_index:
        return numeric_candidates[fallback_index]
    return None


def normalize_role(role: object) -> str:
    text = str(role or "").strip().lower()
    if not text:
        return "run"
    if any(token in text for token in ("baseline", "首次", "首刀", "初始", "原始", "首次加工", "base")):
        return "baseline"
    match = re.search(r"(\d+)", text)
    if "优化" in text or "opt" in text or "run" in text:
        if match:
            return f"optimization_{int(match.group(1))}"
        return "optimization"
    return re.sub(r"[^a-z0-9_]+", "_", text).strip("_") or "run"


def role_sort_key(role: object) -> Tuple[int, int, str]:
    normalized = normalize_role(role)
    if normalized == "baseline":
        return (0, 0, normalized)
    match = re.fullmatch(r"optimization_(\d+)", normalized)
    if match:
        return (1, int(match.group(1)), normalized)
    if normalized == "optimization":
        return (1, 9999, normalized)
    return (2, 9999, normalized)


def build_run_label(role: object, file_name: str, position: int) -> str:
    normalized = normalize_role(role)
    if normalized == "baseline":
        return "首次加工"
    match = re.fullmatch(r"optimization_(\d+)", normalized)
    if match:
        return f"第{int(match.group(1))}次优化"
    if normalized == "optimization":
        return f"优化{position}"
    return file_name or f"Run {position}"


def prepare_run_frame(
    df: pd.DataFrame,
    *,
    run_id: str,
    role: object,
    file_name: str,
    f_column: str,
    p_column: str,
    predicted_column: Optional[str] = None,
    feed_column: Optional[str] = None,
) -> pd.DataFrame:
    frame = pd.DataFrame({
        "sample_index": np.arange(len(df), dtype=int),
        "time_ms": np.arange(len(df), dtype=int),
        "time_s": np.arange(len(df), dtype=float) / 1000.0,
        "F": pd.to_numeric(df[f_column], errors="coerce"),
        "P": pd.to_numeric(df[p_column], errors="coerce"),
    })
    if predicted_column and predicted_column in df.columns:
        frame["P_pred"] = pd.to_numeric(df[predicted_column], errors="coerce")
    else:
        frame["P_pred"] = np.nan

    if feed_column and feed_column in df.columns:
        frame["feed"] = pd.to_numeric(df[feed_column], errors="coerce")
    else:
        frame["feed"] = frame["F"]

    normalized_role = normalize_role(role)
    frame["run_id"] = str(run_id)
    frame["role"] = normalized_role
    frame["file_name"] = str(file_name)
    frame["run_label"] = build_run_label(normalized_role, str(file_name), 0)
    return frame


def contiguous_segments(mask: Sequence[bool]) -> List[Tuple[int, int]]:
    mask_arr = np.asarray(mask, dtype=bool)
    segments: List[Tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(mask_arr):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            segments.append((int(start), int(idx - 1)))
            start = None
    if start is not None:
        segments.append((int(start), int(len(mask_arr) - 1)))
    return segments


def merge_segments(segments: Sequence[Tuple[int, int]], merge_gap: int) -> List[Tuple[int, int]]:
    if not segments:
        return []
    merged: List[List[int]] = [[int(segments[0][0]), int(segments[0][1])]]
    for start_idx, end_idx in segments[1:]:
        gap = int(start_idx) - int(merged[-1][1]) - 1
        if gap <= int(merge_gap):
            merged[-1][1] = int(end_idx)
        else:
            merged.append([int(start_idx), int(end_idx)])
    return [(start_idx, end_idx) for start_idx, end_idx in merged]


def identify_shared_intervals(
    baseline_frame: pd.DataFrame,
    opt1_frame: pd.DataFrame,
    *,
    threshold: float,
    min_segment_len: int,
    merge_gap: int,
) -> pd.DataFrame:
    length = int(min(len(baseline_frame), len(opt1_frame)))
    if length <= 0:
        raise ValueError("用于区间识别的 baseline / 第一次优化 数据为空")

    base_f = pd.to_numeric(baseline_frame["F"].iloc[:length], errors="coerce").to_numpy(dtype=float)
    opt1_f = pd.to_numeric(opt1_frame["F"].iloc[:length], errors="coerce").to_numpy(dtype=float)
    valid_mask = np.isfinite(base_f) & np.isfinite(opt1_f)
    diff = np.zeros(length, dtype=float)
    diff[valid_mask] = np.abs(opt1_f[valid_mask] - base_f[valid_mask])
    change_mask = valid_mask & (diff > float(threshold))

    segments = contiguous_segments(change_mask)
    segments = merge_segments(segments, int(merge_gap))
    segments = [(start_idx, end_idx) for start_idx, end_idx in segments if (end_idx - start_idx + 1) >= int(min_segment_len)]

    interval_rows = []
    for idx, (start_idx, end_idx) in enumerate(segments, 1):
        base_segment = base_f[start_idx:end_idx + 1]
        opt_segment = opt1_f[start_idx:end_idx + 1]
        point_count = int(end_idx - start_idx + 1)
        interval = AcademicInterval(
            interval_id=f"INT-{idx:03d}",
            start_idx=int(start_idx),
            end_idx=int(end_idx),
            point_count=point_count,
            duration_ms=point_count,
            duration_s=point_count / 1000.0,
            baseline_mean_f=float(np.nanmean(base_segment)) if base_segment.size else math.nan,
            opt1_mean_f=float(np.nanmean(opt_segment)) if opt_segment.size else math.nan,
            delta_f=float(np.nanmean(opt_segment) - np.nanmean(base_segment)) if base_segment.size and opt_segment.size else math.nan,
        )
        interval_rows.append(interval.__dict__)

    return pd.DataFrame(interval_rows)


def slice_interval(frame: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
    if frame.empty:
        return frame.iloc[0:0].copy()
    safe_start = max(0, int(start_idx))
    safe_end = min(int(end_idx), len(frame) - 1)
    if safe_end < safe_start:
        return frame.iloc[0:0].copy()
    return frame.iloc[safe_start:safe_end + 1].copy()


def add_interval_flags(frame: pd.DataFrame, intervals_df: pd.DataFrame) -> pd.DataFrame:
    tagged = frame.copy()
    tagged["interval_id"] = ""
    tagged["is_interval"] = False
    for row in intervals_df.itertuples(index=False):
        safe_start = max(0, int(row.start_idx))
        safe_end = min(int(row.end_idx), len(tagged) - 1)
        if safe_end < safe_start:
            continue
        tagged.loc[safe_start:safe_end, "interval_id"] = str(row.interval_id)
        tagged.loc[safe_start:safe_end, "is_interval"] = True
    return tagged


def get_relative_error_floor(actual: Iterable[float]) -> float:
    actual_arr = np.asarray(list(actual), dtype=float)
    valid_actual = np.abs(actual_arr[np.isfinite(actual_arr)])
    positive_actual = valid_actual[valid_actual > 1e-9]
    if positive_actual.size == 0:
        return math.nan
    return float(max(50.0, np.percentile(positive_actual, 95) * 0.05))


def compute_error_metrics(actual: Iterable[float], predicted: Iterable[float]) -> Dict[str, float]:
    actual_arr = np.asarray(list(actual), dtype=float)
    predicted_arr = np.asarray(list(predicted), dtype=float)
    valid_mask = np.isfinite(actual_arr) & np.isfinite(predicted_arr)
    if not np.any(valid_mask):
        return {
            "count": 0,
            "mae": math.nan,
            "rmse": math.nan,
            "mape": math.nan,
            "wmape": math.nan,
            "raw_mape": math.nan,
            "relative_error_floor": math.nan,
            "relative_valid_count": 0,
            "max_abs_error": math.nan,
        }

    actual_valid = actual_arr[valid_mask]
    predicted_valid = predicted_arr[valid_mask]
    error = predicted_valid - actual_valid
    abs_error = np.abs(error)
    squared_error = error ** 2

    abs_actual = np.abs(actual_valid)
    relative_error_floor = get_relative_error_floor(actual_valid)

    raw_mask = abs_actual > 1e-9
    if np.any(raw_mask):
        raw_mape = float(np.mean(np.abs(error[raw_mask] / actual_valid[raw_mask])) * 100.0)
    else:
        raw_mape = math.nan

    filtered_mask = raw_mask.copy()
    if np.isfinite(relative_error_floor):
        filtered_mask &= abs_actual >= float(relative_error_floor)
    if np.any(filtered_mask):
        mape = float(np.mean(np.abs(error[filtered_mask] / actual_valid[filtered_mask])) * 100.0)
    else:
        mape = math.nan

    actual_sum = float(np.sum(abs_actual))
    if actual_sum > 1e-9:
        wmape = float(np.sum(abs_error) / actual_sum * 100.0)
    else:
        wmape = math.nan

    return {
        "count": int(valid_mask.sum()),
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(squared_error))),
        "mape": mape,
        "wmape": wmape,
        "raw_mape": raw_mape,
        "relative_error_floor": relative_error_floor,
        "relative_valid_count": int(np.sum(filtered_mask)),
        "max_abs_error": float(np.max(abs_error)),
    }


def _calc_response_metrics(frame: pd.DataFrame, start_idx: int, end_idx: int) -> Dict[str, float]:
    segment = slice_interval(frame, start_idx, end_idx)
    if segment.empty:
        return {
            "response_delay_ms": math.nan,
            "settling_time_ms": math.nan,
            "overshoot": math.nan,
            "oscillation_score": math.nan,
        }

    power_arr = pd.to_numeric(segment["P"], errors="coerce").to_numpy(dtype=float)
    pre_window = slice_interval(frame, max(0, int(start_idx) - 30), max(0, int(start_idx) - 1))
    pre_power = pd.to_numeric(pre_window["P"], errors="coerce").to_numpy(dtype=float)
    pre_power = pre_power[np.isfinite(pre_power)]
    if pre_power.size == 0:
        pre_power = power_arr[np.isfinite(power_arr)][: min(10, len(power_arr))]

    post_power = power_arr[np.isfinite(power_arr)]
    if post_power.size == 0:
        return {
            "response_delay_ms": math.nan,
            "settling_time_ms": math.nan,
            "overshoot": math.nan,
            "oscillation_score": math.nan,
        }

    pre_mean = float(np.mean(pre_power)) if pre_power.size else float(np.mean(post_power))
    target_window = max(5, int(round(len(post_power) * 0.2)))
    target_mean = float(np.mean(post_power[-target_window:]))
    amplitude = target_mean - pre_mean
    base_std = float(np.std(pre_power)) if pre_power.size > 1 else 0.0
    trigger = max(abs(amplitude) * 0.15, base_std * 2.0, 1e-6)

    response_delay_ms = math.nan
    for idx, value in enumerate(power_arr):
        if np.isfinite(value) and abs(value - pre_mean) >= trigger:
            response_delay_ms = float(idx)
            break

    settling_tol = max(abs(amplitude) * 0.1, float(np.std(post_power)) * 0.25, 1e-6)
    settling_time_ms = math.nan
    start_from = int(response_delay_ms) if np.isfinite(response_delay_ms) else 0
    for idx in range(start_from, len(power_arr)):
        tail = power_arr[idx:]
        tail = tail[np.isfinite(tail)]
        if tail.size and np.all(np.abs(tail - target_mean) <= settling_tol):
            settling_time_ms = float(idx)
            break

    if amplitude >= 0.0:
        overshoot = float(np.nanmax(power_arr) - target_mean)
    else:
        overshoot = float(target_mean - np.nanmin(power_arr))
    diff_arr = np.diff(post_power)
    oscillation_score = float(np.std(diff_arr)) if diff_arr.size > 1 else 0.0
    return {
        "response_delay_ms": response_delay_ms,
        "settling_time_ms": settling_time_ms,
        "overshoot": overshoot,
        "oscillation_score": oscillation_score,
    }


def _calc_boundary_rows(frame: pd.DataFrame, start_idx: int, end_idx: int, interval_id: str, window: int) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for boundary_name, center_idx in (("start", int(start_idx)), ("end", int(end_idx))):
        left = slice_interval(frame, center_idx - int(window), center_idx - 1)
        right = slice_interval(frame, center_idx, center_idx + int(window) - 1)
        left_power = pd.to_numeric(left["P"], errors="coerce").to_numpy(dtype=float)
        right_power = pd.to_numeric(right["P"], errors="coerce").to_numpy(dtype=float)
        left_power = left_power[np.isfinite(left_power)]
        right_power = right_power[np.isfinite(right_power)]

        if left_power.size == 0 or right_power.size == 0:
            rows.append({
                "interval_id": interval_id,
                "boundary": boundary_name,
                "window_ms": int(window),
                "load_jump": math.nan,
                "load_peak": math.nan,
                "load_std": math.nan,
                "spike_score": math.nan,
                "is_spiky": False,
            })
            continue

        load_jump = float(np.mean(right_power) - np.mean(left_power))
        combined = np.concatenate([left_power, right_power])
        load_peak = float(np.max(combined) - np.min(combined))
        load_std = float(np.std(right_power))
        spike_score = abs(load_jump) + load_peak * 0.3 + load_std * 0.2
        rows.append({
            "interval_id": interval_id,
            "boundary": boundary_name,
            "window_ms": int(window),
            "load_jump": load_jump,
            "load_peak": load_peak,
            "load_std": load_std,
            "spike_score": spike_score,
            "is_spiky": bool(spike_score > max(abs(load_jump) * 1.2, 1e-6)),
        })
    return rows


def _calc_steady_metrics(frame: pd.DataFrame, start_idx: int, end_idx: int) -> Dict[str, float]:
    segment = slice_interval(frame, start_idx, end_idx)
    if segment.empty:
        return {
            "p_slope": math.nan,
            "half_mean_gap": math.nan,
            "steady_score": math.nan,
            "is_slow_drift": False,
        }
    power_arr = pd.to_numeric(segment["P"], errors="coerce").to_numpy(dtype=float)
    valid_mask = np.isfinite(power_arr)
    if int(valid_mask.sum()) < 3:
        return {
            "p_slope": math.nan,
            "half_mean_gap": math.nan,
            "steady_score": math.nan,
            "is_slow_drift": False,
        }

    x = np.arange(valid_mask.sum(), dtype=float)
    y = power_arr[valid_mask]
    slope = float(np.polyfit(x, y, 1)[0]) if y.size >= 2 else 0.0
    mid = max(1, y.size // 2)
    left_mean = float(np.mean(y[:mid]))
    right_mean = float(np.mean(y[mid:])) if y[mid:].size else left_mean
    half_mean_gap = right_mean - left_mean
    scale = max(abs(float(np.mean(y))), 1.0)
    normalized_drift = abs(slope) * y.size / scale + abs(half_mean_gap) / scale
    steady_score = max(0.0, 100.0 - normalized_drift * 100.0)
    return {
        "p_slope": slope,
        "half_mean_gap": half_mean_gap,
        "steady_score": steady_score,
        "is_slow_drift": bool(normalized_drift > 0.08),
    }


def _calc_strength_level(delta_f: float, thresholds: Tuple[float, float]) -> str:
    low_threshold, high_threshold = thresholds
    magnitude = abs(float(delta_f))
    if magnitude <= low_threshold:
        return "小"
    if magnitude <= high_threshold:
        return "中"
    return "大"


def _pick_strength_thresholds(control_rows: List[Dict[str, float]]) -> Tuple[float, float]:
    values = [abs(float(row.get("delta_F", 0.0))) for row in control_rows if np.isfinite(row.get("delta_F", math.nan))]
    if not values:
        return (0.0, 0.0)
    arr = np.asarray(values, dtype=float)
    return (
        float(np.quantile(arr, 0.33)),
        float(np.quantile(arr, 0.66)),
    )


def _safe_mean(series: pd.Series) -> float:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else math.nan


def _safe_std(series: pd.Series) -> float:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


def build_interval_frame_from_records(records: Sequence[dict]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for idx, record in enumerate(records or [], 1):
        if not isinstance(record, dict):
            continue
        try:
            start_idx = int(record.get("sample_start_idx"))
            end_idx = int(record.get("sample_end_idx"))
        except Exception:
            continue
        if end_idx < start_idx:
            continue
        point_count = int(record.get("sample_count", record.get("point_count", end_idx - start_idx + 1)))
        interval_id = str(record.get("zone_id") or record.get("interval_id") or f"Z{idx:03d}")
        segment_type = str(record.get("segment_type") or "steady").strip() or "steady"
        kc_source = str(record.get("kc_source") or "").strip().lower()
        is_idle_interval = bool(record.get("is_idle_interval")) or kc_source == "idle"
        rows.append({
            "interval_id": interval_id,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "point_count": max(point_count, end_idx - start_idx + 1),
            "duration_ms": max(point_count, end_idx - start_idx + 1),
            "duration_s": max(point_count, end_idx - start_idx + 1) / 1000.0,
            "start_line": record.get("start_line"),
            "end_line": record.get("end_line"),
            "start_label": record.get("start_label", ""),
            "end_label": record.get("end_label", ""),
            "display_start_x": record.get("display_start_x", math.nan),
            "display_end_x": record.get("display_end_x", math.nan),
            "display_start_t": record.get("display_start_t", math.nan),
            "display_end_t": record.get("display_end_t", math.nan),
            "segment_type": segment_type,
            "is_idle_interval": is_idle_interval,
            "steady_subtype": "idle" if segment_type == "steady" and is_idle_interval else ("cutting" if segment_type == "steady" else "nonsteady"),
            "p_idle": float(record.get("p_idle", math.nan)) if pd.notna(record.get("p_idle", math.nan)) else math.nan,
        })
    return pd.DataFrame(rows)


def classify_intervals(intervals_df: pd.DataFrame) -> pd.DataFrame:
    classified = intervals_df.copy()
    length_classes = []
    interval_classes = []
    steady_subtypes = []
    for row in classified.itertuples(index=False):
        segment_type = str(getattr(row, "segment_type", "steady") or "steady").strip().lower()
        point_count = int(getattr(row, "point_count", 0) or 0)
        if segment_type != "steady":
            length_classes.append("nonsteady")
            interval_classes.append("非稳态区间")
            steady_subtypes.append("nonsteady")
            continue
        is_idle_interval = bool(getattr(row, "is_idle_interval", False))
        if point_count < 100:
            length_class = "极短"
        elif point_count < 1000:
            length_class = "短"
        else:
            length_class = "长"
        length_classes.append(length_class)
        steady_subtypes.append("idle" if is_idle_interval else "cutting")
        interval_classes.append("空载段" if is_idle_interval else f"{length_class}稳态区间")
    classified["length_class"] = length_classes
    classified["interval_class"] = interval_classes
    classified["steady_subtype"] = steady_subtypes
    return classified


def _strategy_weights(strategy_mode: object) -> Tuple[float, float, float]:
    mode = str(strategy_mode or "折中策略").strip()
    if mode == "均衡优先":
        return 0.70, 0.20, 0.10
    if mode == "效率优先":
        return 0.30, 0.60, 0.10
    return 0.50, 0.35, 0.15


def build_target_table(
    intervals_df: pd.DataFrame,
    average_df: pd.DataFrame,
    fluctuation_df: pd.DataFrame,
    control_df: pd.DataFrame,
    *,
    strategy_mode: object,
) -> pd.DataFrame:
    if intervals_df.empty:
        return pd.DataFrame()

    avg_by_interval = average_df.groupby("interval_id", as_index=False).agg(
        pred_mean=("avg_predicted_load", "mean"),
        actual_mean=("avg_actual_load", "mean"),
    ) if not average_df.empty else pd.DataFrame(columns=["interval_id", "pred_mean", "actual_mean"])
    fluct_by_interval = fluctuation_df.groupby("interval_id", as_index=False).agg(
        load_std=("P_std", "mean"),
        residual_rms=("residual_rms", "mean"),
    ) if not fluctuation_df.empty else pd.DataFrame(columns=["interval_id", "load_std", "residual_rms"])
    control_by_interval = control_df.groupby("interval_id", as_index=False).agg(
        delta_f_mean=("delta_F", "mean"),
        delta_p_mean=("delta_P", "mean"),
        hard_to_reach=("is_hard_to_reach", "max"),
    ) if not control_df.empty else pd.DataFrame(columns=["interval_id", "delta_f_mean", "delta_p_mean", "hard_to_reach"])

    merged = intervals_df.merge(avg_by_interval, on="interval_id", how="left")
    merged = merged.merge(fluct_by_interval, on="interval_id", how="left")
    merged = merged.merge(control_by_interval, on="interval_id", how="left")

    steady_pred = merged.loc[
        merged["segment_type"].astype(str).eq("steady") & np.isfinite(pd.to_numeric(merged["pred_mean"], errors="coerce")),
        "pred_mean",
    ].to_numpy(dtype=float)
    global_ref = float(np.nanmedian(steady_pred)) if steady_pred.size else 0.0
    w_balance, w_efficiency, w_risk = _strategy_weights(strategy_mode)

    target_rows: List[Dict[str, object]] = []
    for row in merged.itertuples(index=False):
        segment_type = str(getattr(row, "segment_type", "steady") or "steady")
        length_class = str(getattr(row, "length_class", "") or "")
        steady_subtype = str(getattr(row, "steady_subtype", "cutting") or "cutting")
        pred_mean = float(getattr(row, "pred_mean", math.nan))
        actual_mean = float(getattr(row, "actual_mean", math.nan))
        p_idle = float(getattr(row, "p_idle", math.nan))
        load_std = float(getattr(row, "load_std", math.nan))
        residual_rms = float(getattr(row, "residual_rms", math.nan))
        delta_f_mean = float(getattr(row, "delta_f_mean", math.nan))
        hard_to_reach = bool(getattr(row, "hard_to_reach", False))

        base_value = pred_mean if np.isfinite(pred_mean) else actual_mean
        if not np.isfinite(base_value):
            base_value = global_ref

        balance_target = base_value + 0.55 * (global_ref - base_value)
        efficiency_gain = 0.04
        if str(strategy_mode or "") == "效率优先":
            efficiency_gain = 0.10
        elif str(strategy_mode or "") == "折中策略":
            efficiency_gain = 0.07
        efficiency_target = base_value * (1.0 + efficiency_gain)

        risk_penalty = 0.0
        if np.isfinite(load_std):
            risk_penalty += abs(load_std) * 0.10
        if np.isfinite(residual_rms):
            risk_penalty += abs(residual_rms) * 0.05
        if hard_to_reach:
            risk_penalty += abs(base_value) * 0.08

        raw_target = w_balance * balance_target + w_efficiency * efficiency_target - w_risk * risk_penalty
        max_delta_ratio = 0.12
        if length_class == "短":
            max_delta_ratio = 0.08
        elif length_class == "极短":
            max_delta_ratio = 0.04
        if hard_to_reach:
            max_delta_ratio *= 0.5
        max_delta = max(abs(base_value) * max_delta_ratio, 5.0)
        target_load = base_value + max(min(raw_target - base_value, max_delta), -max_delta)
        if segment_type == "steady" and steady_subtype == "idle":
            target_load = p_idle if np.isfinite(p_idle) else base_value

        reachability = 1.0
        if np.isfinite(delta_f_mean):
            reachability = 1.0 / (1.0 + abs(delta_f_mean))
        transition_cost = 0.0 if not np.isfinite(load_std) else float(abs(load_std))

        target_rows.append({
            "interval_id": str(getattr(row, "interval_id")),
            "segment_type": segment_type,
            "steady_subtype": steady_subtype,
            "length_class": length_class,
            "interval_class": getattr(row, "interval_class", ""),
            "pred_mean": pred_mean,
            "actual_mean": actual_mean,
            "p_idle": p_idle,
            "load_std": load_std,
            "reachability": reachability,
            "transition_cost": transition_cost,
            "strategy_mode": str(strategy_mode or "折中策略"),
            "target_load": float(target_load) if segment_type == "steady" else math.nan,
            "target_segments": [],
        })
    return pd.DataFrame(target_rows)


def build_multi_run_analysis(
    run_frames: Sequence[pd.DataFrame],
    *,
    intervals: Sequence[dict],
    strategy_mode: object = "折中策略",
    boundary_window: int = 50,
) -> Dict[str, object]:
    if len(run_frames) < 2:
        raise ValueError("至少需要 baseline 和第一次优化两个 run 才能进行数据分析")

    ordered_runs = sorted(run_frames, key=lambda frame: role_sort_key(frame["role"].iloc[0]))
    baseline_frame = next((frame for frame in ordered_runs if str(frame["role"].iloc[0]) == "baseline"), None)
    if baseline_frame is None:
        raise ValueError("缺少 baseline 数据，无法识别区间")

    opt_candidates = [frame for frame in ordered_runs if str(frame["role"].iloc[0]).startswith("optimization")]
    if not opt_candidates:
        raise ValueError("缺少第一次优化数据，无法识别区间")
    opt1_frame = sorted(opt_candidates, key=lambda frame: role_sort_key(frame["role"].iloc[0]))[0]

    intervals_df = build_interval_frame_from_records(intervals)
    intervals_df = classify_intervals(intervals_df)
    if intervals_df.empty:
        raise ValueError("当前没有可用的工艺信息页面稳态区间")
    tagged_runs = [add_interval_flags(frame, intervals_df) for frame in ordered_runs]

    average_rows: List[Dict[str, object]] = []
    fluctuation_rows: List[Dict[str, object]] = []
    control_rows: List[Dict[str, object]] = []
    response_rows: List[Dict[str, object]] = []
    boundary_rows: List[Dict[str, object]] = []
    steady_rows: List[Dict[str, object]] = []

    baseline_interval_lookup: Dict[str, Dict[str, float]] = {}
    for interval in intervals_df.itertuples(index=False):
        baseline_segment = slice_interval(baseline_frame, interval.start_idx, interval.end_idx)
        baseline_interval_lookup[str(interval.interval_id)] = {
            "avg_F": _safe_mean(baseline_segment["F"]),
            "avg_P": _safe_mean(baseline_segment["P"]),
        }

    for order_idx, frame in enumerate(tagged_runs):
        run_id = str(frame["run_id"].iloc[0])
        role = str(frame["role"].iloc[0])
        file_name = str(frame["file_name"].iloc[0])
        run_label = build_run_label(role, file_name, order_idx + 1)

        for interval in intervals_df.itertuples(index=False):
            segment = slice_interval(frame, interval.start_idx, interval.end_idx)
            sample_count = int(len(segment))
            avg_p = _safe_mean(segment["P"])
            avg_f = _safe_mean(segment["F"])
            avg_pred = _safe_mean(segment["P_pred"])
            avg_feed = _safe_mean(segment["feed"])
            baseline_ref = baseline_interval_lookup.get(str(interval.interval_id), {})
            delta_f = avg_f - baseline_ref.get("avg_F", math.nan)
            delta_p = avg_p - baseline_ref.get("avg_P", math.nan)
            residual_arr = (
                pd.to_numeric(segment["P"], errors="coerce").to_numpy(dtype=float)
                - pd.to_numeric(segment["P_pred"], errors="coerce").to_numpy(dtype=float)
            )
            residual_arr = residual_arr[np.isfinite(residual_arr)]
            residual_rms = float(np.sqrt(np.mean(residual_arr ** 2))) if residual_arr.size else math.nan

            average_rows.append({
                "run_id": run_id,
                "run_label": run_label,
                "file_name": file_name,
                "role": role,
                "interval_id": str(interval.interval_id),
                "point_count": sample_count,
                "duration_ms": sample_count,
                "duration_s": sample_count / 1000.0,
                "avg_actual_load": avg_p,
                "avg_F": avg_f,
                "avg_predicted_load": avg_pred,
                "avg_feed": avg_feed,
            })
            fluctuation_rows.append({
                "run_id": run_id,
                "run_label": run_label,
                "file_name": file_name,
                "role": role,
                "interval_id": str(interval.interval_id),
                "P_std": _safe_std(segment["P"]),
                "P_peak_to_peak": float(
                    np.nanmax(pd.to_numeric(segment["P"], errors="coerce").to_numpy(dtype=float))
                    - np.nanmin(pd.to_numeric(segment["P"], errors="coerce").to_numpy(dtype=float))
                ) if sample_count else math.nan,
                "residual_rms": residual_rms,
                "F_std": _safe_std(segment["F"]),
            })
            control_rows.append({
                "run_id": run_id,
                "run_label": run_label,
                "file_name": file_name,
                "role": role,
                "interval_id": str(interval.interval_id),
                "baseline_avg_F": baseline_ref.get("avg_F", math.nan),
                "current_avg_F": avg_f,
                "delta_F": delta_f,
                "baseline_avg_P": baseline_ref.get("avg_P", math.nan),
                "current_avg_P": avg_p,
                "delta_P": delta_p,
            })
            response_metrics = _calc_response_metrics(frame, int(interval.start_idx), int(interval.end_idx))
            response_rows.append({
                "run_id": run_id,
                "run_label": run_label,
                "file_name": file_name,
                "role": role,
                "interval_id": str(interval.interval_id),
                **response_metrics,
            })
            for boundary_row in _calc_boundary_rows(frame, int(interval.start_idx), int(interval.end_idx), str(interval.interval_id), int(boundary_window)):
                boundary_rows.append({
                    "run_id": run_id,
                    "run_label": run_label,
                    "file_name": file_name,
                    "role": role,
                    **boundary_row,
                })
            steady_metrics = _calc_steady_metrics(frame, int(interval.start_idx), int(interval.end_idx))
            steady_rows.append({
                "run_id": run_id,
                "run_label": run_label,
                "file_name": file_name,
                "role": role,
                "interval_id": str(interval.interval_id),
                **steady_metrics,
            })

    average_df = pd.DataFrame(average_rows)
    fluctuation_df = pd.DataFrame(fluctuation_rows)
    control_df = pd.DataFrame(control_rows)
    response_df = pd.DataFrame(response_rows)
    boundary_df = pd.DataFrame(boundary_rows)
    steady_df = pd.DataFrame(steady_rows)

    if not fluctuation_df.empty:
        p_std_threshold = float(fluctuation_df["P_std"].median() + fluctuation_df["P_std"].std(ddof=0))
        fluctuation_df["is_high_variation"] = fluctuation_df["P_std"].fillna(0.0) >= p_std_threshold
    else:
        fluctuation_df["is_high_variation"] = []

    strength_thresholds = _pick_strength_thresholds(control_rows)
    if not control_df.empty:
        control_df["strength_level"] = control_df["delta_F"].apply(lambda value: _calc_strength_level(value, strength_thresholds))
        efficiency_ratio = np.abs(control_df["delta_P"]) / np.maximum(np.abs(control_df["delta_F"]), 1e-6)
        hard_threshold = float(np.nanquantile(efficiency_ratio.to_numpy(dtype=float), 0.3)) if len(efficiency_ratio) else math.nan
        control_df["is_hard_to_reach"] = (
            control_df["strength_level"].isin(["中", "大"])
            & (efficiency_ratio <= hard_threshold if np.isfinite(hard_threshold) else False)
        )
    else:
        control_df["strength_level"] = []
        control_df["is_hard_to_reach"] = []

    drift_rows: List[Dict[str, object]] = []
    if not average_df.empty and len(tagged_runs) >= 3:
        run_order_map = {
            str(frame["run_id"].iloc[0]): idx
            for idx, frame in enumerate(tagged_runs)
        }
        for interval_id, group in average_df.groupby("interval_id", sort=False):
            if len(group) < 3:
                continue
            ordered_group = group.assign(run_order=group["run_id"].map(run_order_map)).sort_values("run_order")
            x = ordered_group["run_order"].to_numpy(dtype=float)
            for metric_name in ("avg_actual_load", "avg_F"):
                y = ordered_group[metric_name].to_numpy(dtype=float)
                valid_mask = np.isfinite(x) & np.isfinite(y)
                if int(valid_mask.sum()) < 3:
                    continue
                slope = float(np.polyfit(x[valid_mask], y[valid_mask], 1)[0])
                scale = max(abs(float(np.nanmean(y[valid_mask]))), 1.0)
                is_drift = abs(slope) >= scale * 0.03
                drift_rows.append({
                    "interval_id": str(interval_id),
                    "metric": metric_name,
                    "slope": slope,
                    "is_drift": bool(is_drift),
                    "trend": "上升" if slope > 0 else "下降",
                })
    drift_df = pd.DataFrame(drift_rows)

    similarity_rows: List[Dict[str, object]] = []
    baseline_average_df = average_df[average_df["role"] == "baseline"].copy()
    baseline_average_df = baseline_average_df.sort_values("interval_id")
    interval_ids = baseline_average_df["interval_id"].tolist()
    for left_id, right_id in zip(interval_ids, interval_ids[1:]):
        left_group = average_df[average_df["interval_id"] == left_id].sort_values("role")
        right_group = average_df[average_df["interval_id"] == right_id].sort_values("role")
        if left_group.empty or right_group.empty:
            continue
        merged = left_group[["run_id", "avg_actual_load", "avg_F"]].merge(
            right_group[["run_id", "avg_actual_load", "avg_F"]],
            on="run_id",
            suffixes=("_left", "_right"),
        )
        if merged.empty:
            continue
        load_gap = float(np.mean(np.abs(merged["avg_actual_load_left"] - merged["avg_actual_load_right"])))
        feed_gap = float(np.mean(np.abs(merged["avg_F_left"] - merged["avg_F_right"])))
        load_scale = max(
            abs(float(np.nanmean(np.r_[merged["avg_actual_load_left"], merged["avg_actual_load_right"]]))),
            1.0,
        )
        feed_scale = max(
            abs(float(np.nanmean(np.r_[merged["avg_F_left"], merged["avg_F_right"]]))),
            1.0,
        )
        similarity_score = max(0.0, 100.0 - (load_gap / load_scale + feed_gap / feed_scale) * 50.0)
        similarity_rows.append({
            "left_interval": str(left_id),
            "right_interval": str(right_id),
            "similarity_score": similarity_score,
            "mean_load_gap": load_gap,
            "mean_F_gap": feed_gap,
            "merge_suggestion": bool(similarity_score >= 85.0),
        })
    similarity_df = pd.DataFrame(similarity_rows)
    target_df = build_target_table(
        intervals_df,
        average_df,
        fluctuation_df,
        control_df,
        strategy_mode=strategy_mode,
    )

    summary_lines: List[str] = []
    summary_lines.append(f"共读取 {len(intervals_df)} 个工艺页区间。")
    summary_lines.append(f"当前策略: {str(strategy_mode or '折中策略')}")

    if not average_df.empty:
        mean_load_by_interval = average_df.groupby("interval_id", as_index=False)["avg_actual_load"].mean()
        high_load_threshold = float(mean_load_by_interval["avg_actual_load"].quantile(0.75))
        high_load_intervals = mean_load_by_interval.loc[
            mean_load_by_interval["avg_actual_load"] >= high_load_threshold,
            "interval_id",
        ].tolist()
        if high_load_intervals:
            summary_lines.append(f"高负载区间: {', '.join(high_load_intervals[:5])}")

    if not fluctuation_df.empty:
        high_var_intervals = fluctuation_df.loc[
            fluctuation_df["is_high_variation"],
            "interval_id",
        ].drop_duplicates().tolist()
        if high_var_intervals:
            summary_lines.append(f"高波动区间: {', '.join(high_var_intervals[:5])}")

    if not control_df.empty:
        hard_intervals = control_df.loc[
            control_df["is_hard_to_reach"],
            "interval_id",
        ].drop_duplicates().tolist()
        if hard_intervals:
            summary_lines.append(f"疑似难达区: {', '.join(hard_intervals[:5])}")

    if not drift_df.empty:
        drift_intervals = drift_df.loc[drift_df["is_drift"], "interval_id"].drop_duplicates().tolist()
        if drift_intervals:
            summary_lines.append(f"存在 run-to-run 漂移的区间: {', '.join(drift_intervals[:5])}")

    if not similarity_df.empty:
        merge_pairs = similarity_df.loc[similarity_df["merge_suggestion"]].head(5)
        if not merge_pairs.empty:
            pair_text = [f"{row.left_interval}+{row.right_interval}" for row in merge_pairs.itertuples(index=False)]
            summary_lines.append(f"建议合并的相邻区间: {', '.join(pair_text)}")

    return {
        "runs": tagged_runs,
        "intervals": intervals_df,
        "average_table": average_df,
        "fluctuation_table": fluctuation_df,
        "control_table": control_df,
        "response_table": response_df,
        "boundary_table": boundary_df,
        "steady_table": steady_df,
        "drift_table": drift_df,
        "similarity_table": similarity_df,
        "target_table": target_df,
        "summary_lines": summary_lines,
        "baseline_run_id": str(baseline_frame["run_id"].iloc[0]),
        "opt1_run_id": str(opt1_frame["run_id"].iloc[0]),
        "strategy_mode": str(strategy_mode or "折中策略"),
    }


def export_table_to_csv(df: pd.DataFrame, output_path: str) -> str:
    target_path = os.path.abspath(output_path)
    df.to_csv(target_path, index=False, encoding="utf-8-sig")
    return target_path
