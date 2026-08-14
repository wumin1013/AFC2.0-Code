from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd


def _solve_nonnegative_scalar(feature_values, target_values) -> float:
    feature = np.asarray(feature_values, dtype=float).reshape(-1)
    target = np.asarray(target_values, dtype=float).reshape(-1)
    denominator = float(np.dot(feature, feature))
    if denominator <= 1e-12:
        return 0.0
    return max(float(np.dot(feature, target) / denominator), 0.0)


def _solve_nonnegative_kc_ke(mrr_values, ap_values, target_values):
    """用二维边界枚举实现无 SciPy 依赖的非负最小二乘。"""
    mrr_arr = np.asarray(mrr_values, dtype=float).reshape(-1)
    ap_arr = np.asarray(ap_values, dtype=float).reshape(-1)
    target_arr = np.asarray(target_values, dtype=float).reshape(-1)
    design_matrix = np.column_stack([mrr_arr, ap_arr])

    candidates = []
    try:
        coefficients, _, _, _ = np.linalg.lstsq(design_matrix, target_arr, rcond=None)
        coefficients = np.asarray(coefficients, dtype=float).reshape(-1)
        if (
            coefficients.size == 2
            and np.all(np.isfinite(coefficients))
            and np.all(coefficients >= 0.0)
        ):
            candidates.append(coefficients)
    except Exception:
        pass

    candidates.extend(
        [
            np.array([_solve_nonnegative_scalar(mrr_arr, target_arr), 0.0], dtype=float),
            np.array([0.0, _solve_nonnegative_scalar(ap_arr, target_arr)], dtype=float),
            np.array([0.0, 0.0], dtype=float),
        ]
    )
    best_coefficients = candidates[-1]
    best_rss = float("inf")
    for coefficients in candidates:
        residuals = target_arr - design_matrix @ coefficients
        rss = float(np.dot(residuals, residuals))
        if rss < best_rss:
            best_rss = rss
            best_coefficients = coefficients
    return float(best_coefficients[0]), float(best_coefficients[1])


def fit_nonnegative_kc_ke(
    mrr_values,
    ap_values,
    target_values,
    *,
    source_label: str = "切削锚点",
    minimum_samples: int = 2,
    maximum_condition_number: float = 3e3,
):
    """按主代码口径联合辨识非负 ``Kc/Ke``，并返回诊断信息。"""
    mrr_arr = pd.to_numeric(pd.Series(mrr_values), errors="coerce").to_numpy(dtype=float)
    ap_arr = pd.to_numeric(pd.Series(ap_values), errors="coerce").to_numpy(dtype=float)
    target_arr = pd.to_numeric(pd.Series(target_values), errors="coerce").to_numpy(dtype=float)
    finite_mask = (
        np.isfinite(mrr_arr)
        & np.isfinite(ap_arr)
        & np.isfinite(target_arr)
        & (mrr_arr > 1e-12)
        & (ap_arr > 1e-12)
    )
    mrr_arr = mrr_arr[finite_mask]
    ap_arr = ap_arr[finite_mask]
    target_arr = target_arr[finite_mask]

    required_samples = max(int(minimum_samples), 2)
    sample_count = int(target_arr.size)
    if sample_count < required_samples:
        raise ValueError(
            f"{source_label}有效样本不足 {required_samples} 个，无法辨识 K_c / K_e"
        )

    design_matrix = np.column_stack([mrr_arr, ap_arr])
    scale_mrr = max(float(np.median(np.abs(mrr_arr))), 1.0)
    scale_ap = max(float(np.median(np.abs(ap_arr))), 1.0)
    scaled_design = np.column_stack([mrr_arr / scale_mrr, ap_arr / scale_ap])
    matrix_rank = int(np.linalg.matrix_rank(scaled_design))
    if matrix_rank < 2:
        specific_mrr = np.divide(
            mrr_arr,
            ap_arr,
            out=np.full_like(mrr_arr, np.nan, dtype=float),
            where=np.abs(ap_arr) > 1e-12,
        )
        excitation_span = (
            float(np.nanmax(specific_mrr) - np.nanmin(specific_mrr))
            if specific_mrr.size
            else 0.0
        )
        excitation_scale = max(
            float(np.nanmedian(np.abs(specific_mrr))) if specific_mrr.size else 0.0,
            1.0,
        )
        if excitation_span <= 1e-6 or excitation_span / excitation_scale < 0.02:
            raise ValueError(
                f"{source_label}的 a_e·F/60 变化不足，无法分离 K_c / K_e"
            )
        raise ValueError(f"{source_label}的 MRR 与 a_p 高度共线，无法稳定分离 K_c / K_e")

    condition_number = float(np.linalg.cond(scaled_design))
    if (
        not np.isfinite(condition_number)
        or condition_number > float(maximum_condition_number)
    ):
        raise ValueError(
            f"{source_label}病态过强（条件数={condition_number:.1f}），无法稳定辨识 K_c / K_e"
        )

    kc_value, ke_value = _solve_nonnegative_kc_ke(mrr_arr, ap_arr, target_arr)
    residuals = target_arr - design_matrix @ np.asarray([kc_value, ke_value], dtype=float)
    if sample_count > 2:
        residual_variance = float(
            np.sum(residuals ** 2) / max(sample_count - 2, 1)
        )
        covariance = residual_variance * np.linalg.pinv(design_matrix.T @ design_matrix)
        kc_sigma = float(np.sqrt(max(float(covariance[0, 0]), 0.0)))
    else:
        kc_sigma = 0.0
    return {
        "kc_value": float(kc_value),
        "ke_value": float(ke_value),
        "kc_sigma": max(float(kc_sigma), 0.0),
        "sample_count": sample_count,
        "matrix_rank": matrix_rank,
        "condition_number": condition_number,
        "residual_std": float(np.std(residuals)) if residuals.size else 0.0,
    }


def clip_nonnegative_numeric_array(values) -> np.ndarray:
    """将有限负数截断为 0，保留 NaN/Inf 供上层判定有效性。"""
    array = np.asarray(values, dtype=float).copy()
    negative_mask = np.isfinite(array) & (array < 0.0)
    if np.any(negative_mask):
        array[negative_mask] = 0.0
    return array


def robust_sigma(values) -> float:
    """与原反解链路一致的 MAD/样本标准差尺度。"""
    finite = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if finite.size == 0:
        return float("nan")
    if finite.size == 1:
        return 0.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    sigma = 1.4826 * mad
    if sigma > 1e-12:
        return float(sigma)
    return float(np.std(finite, ddof=1))


def summarize_interval_kc_mode_statistics(values, precision: int = 6):
    """
    返回稳态区间代表 Kc。

    先指定精度取众数；多众数时取最接近中位数者，距离再并列时
    由 ``numpy.unique`` 的升序顺序稳定地取较小值。
    """
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    valid = arr[np.isfinite(arr) & (arr >= 0.0)]
    if valid.size == 0:
        return float("nan"), float("nan"), valid

    rounded = np.round(valid, int(precision))
    unique_values, counts = np.unique(rounded, return_counts=True)
    candidates = unique_values[counts == int(np.max(counts))]
    center = float(np.median(valid))
    kc_hat = float(candidates[np.argmin(np.abs(candidates - center))])
    sigma_kc = robust_sigma(valid) if valid.size > 1 else 0.0
    if not np.isfinite(sigma_kc):
        sigma_kc = 0.0
    return max(kc_hat, 0.0), max(float(sigma_kc), 0.0), valid


def _numeric_column(frame: pd.DataFrame, name: str, default=0.0) -> np.ndarray:
    if name in frame.columns:
        series = frame[name]
    else:
        series = pd.Series(np.full(len(frame), default), index=frame.index)
    numeric = pd.to_numeric(series, errors="coerce")
    if default is not None:
        numeric = numeric.fillna(float(default))
    return numeric.to_numpy(dtype=float)


def estimate_idle_noise_and_mrr_gate(
    sample_df: pd.DataFrame,
    *,
    kc_reference: float | None = None,
):
    """按原反解规则评估空载噪声与 MRR 门限。"""
    mrr_values = _numeric_column(sample_df, "mrr", 0.0)
    ap_values = _numeric_column(sample_df, "ap", 0.0)
    ae_values = _numeric_column(sample_df, "ae", 0.0)
    feed_values = _numeric_column(sample_df, "feed_speed", 0.0)
    actual_load = _numeric_column(sample_df, "actual_load", None)
    idle_power = _numeric_column(sample_df, "idle_power", None)

    idle_mask = (
        np.isfinite(actual_load)
        & np.isfinite(idle_power)
        & (
            (actual_load <= idle_power + 1e-9)
            | (mrr_values <= 1e-9)
            | (ap_values <= 1e-9)
            | (ae_values <= 1e-9)
            | (feed_values <= 1e-9)
        )
    )
    if int(np.sum(idle_mask)) < 20:
        finite_mrr = mrr_values[np.isfinite(mrr_values)]
        if finite_mrr.size:
            low_mrr_cutoff = float(np.percentile(finite_mrr, 10))
            idle_mask = (
                np.isfinite(actual_load)
                & np.isfinite(idle_power)
                & (
                    (actual_load <= idle_power + 1e-9)
                    | (mrr_values <= max(low_mrr_cutoff, 1e-9))
                )
            )

    residuals = actual_load[idle_mask] - idle_power[idle_mask]
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size >= 5:
        low = float(np.percentile(residuals, 5))
        high = float(np.percentile(residuals, 95))
        trimmed = residuals[(residuals >= low) & (residuals <= high)]
        sigma_idle = robust_sigma(trimmed if trimmed.size >= 3 else residuals)
    elif residuals.size >= 2:
        sigma_idle = robust_sigma(residuals)
    else:
        sigma_idle = 0.0
    if not np.isfinite(sigma_idle) or sigma_idle < 0.0:
        sigma_idle = 0.0

    try:
        kc_ref = abs(float(kc_reference))
    except (TypeError, ValueError):
        kc_ref = 0.0
    if kc_ref <= 1e-12:
        kc_ref = 1.0
    delta_mrr = 3.0 * float(sigma_idle) / kc_ref if sigma_idle > 0.0 else 0.0
    return float(sigma_idle), float(delta_mrr), int(residuals.size), idle_mask


def append_inverse_prediction_channels(
    sample_df: pd.DataFrame,
    *,
    sigma_idle: float,
    delta_mrr: float,
    idle_mask,
    ke_value: float,
) -> pd.DataFrame:
    """逐点反解 Kc，并用当前 Ke/P_idle 重建负载。"""
    sample_df = sample_df.copy()
    row_count = len(sample_df)
    if row_count == 0:
        return sample_df

    ap_values = _numeric_column(sample_df, "ap", 0.0)
    mrr_values = _numeric_column(sample_df, "mrr", 0.0)
    actual_load = _numeric_column(sample_df, "actual_load", None)
    idle_power = _numeric_column(sample_df, "idle_power", None)
    prediction_valid = (
        np.asarray(sample_df["prediction_valid"], dtype=bool)
        if "prediction_valid" in sample_df.columns
        else np.ones(row_count, dtype=bool)
    )
    anchor_mask = (
        np.asarray(sample_df["process_anchor_mask"], dtype=bool)
        if "process_anchor_mask" in sample_df.columns
        else np.ones(row_count, dtype=bool)
    )
    supplied_idle_mask = np.asarray(idle_mask, dtype=bool)
    if supplied_idle_mask.size != row_count:
        supplied_idle_mask = np.zeros(row_count, dtype=bool)

    ke_value = max(float(ke_value), 0.0)
    kc_numerator = actual_load - idle_power - ke_value * ap_values
    process_idle_mask = (mrr_values <= 1e-12) | (ap_values <= 1e-12)
    actual_idle_mask = (
        np.isfinite(actual_load)
        & np.isfinite(idle_power)
        & (actual_load <= idle_power + 1e-9)
    )
    idle_point_mask = process_idle_mask | actual_idle_mask | supplied_idle_mask

    actual_series = pd.Series(actual_load, dtype=float)
    rolling_center = actual_series.rolling(window=7, center=True, min_periods=1).median().to_numpy(dtype=float)
    residual_from_center = np.abs(actual_load - rolling_center)
    prev_load = np.roll(actual_load, 1)
    next_load = np.roll(actual_load, -1)
    prev_load[0] = actual_load[0]
    next_load[-1] = actual_load[-1]
    max_jump = np.maximum(np.abs(actual_load - prev_load), np.abs(next_load - actual_load))
    cutting_ref = np.maximum(np.abs(rolling_center - idle_power), 0.0)
    spike_level_tol = np.maximum(6.0 * float(sigma_idle or 0.0), np.maximum(80.0, 0.35 * cutting_ref))
    spike_jump_tol = np.maximum(8.0 * float(sigma_idle or 0.0), np.maximum(120.0, 0.45 * cutting_ref))
    transition_spike_mask = (
        np.isfinite(actual_load)
        & np.isfinite(rolling_center)
        & (~idle_point_mask)
        & ((residual_from_center > spike_level_tol) | (max_jump > spike_jump_tol))
    )

    base_valid = (
        prediction_valid
        & np.isfinite(kc_numerator)
        & np.isfinite(mrr_values)
        & (mrr_values > 1e-12)
        & anchor_mask
        & ~idle_point_mask
        & ~transition_spike_mask
    )
    kc_valid_mask = base_valid & (mrr_values >= max(float(delta_mrr), 0.0))
    sample_kc_valid_mask = (
        prediction_valid
        & np.isfinite(kc_numerator)
        & np.isfinite(mrr_values)
        & (mrr_values > 1e-12)
        & ~idle_point_mask
    )
    sample_kc_values = np.full(row_count, np.nan, dtype=float)
    sample_kc_values[sample_kc_valid_mask] = (
        kc_numerator[sample_kc_valid_mask] / mrr_values[sample_kc_valid_mask]
    )
    sample_kc_values = clip_nonnegative_numeric_array(sample_kc_values)
    kc_point_values = np.full(row_count, np.nan, dtype=float)
    kc_point_values[kc_valid_mask] = sample_kc_values[kc_valid_mask]

    sample_df["idle_window"] = supplied_idle_mask
    sample_df["is_idle_point"] = idle_point_mask
    sample_df["transition_spike"] = transition_spike_mask
    sample_df["kc_numerator"] = kc_numerator
    sample_df["sample_kc"] = sample_kc_values
    sample_df["sample_kc_valid"] = sample_kc_valid_mask
    sample_df["kc_point"] = kc_point_values
    sample_df["kc_valid"] = kc_valid_mask
    sample_df["kc_gated_out"] = base_valid & ~kc_valid_mask

    predicted_source = (
        sample_df["predicted_kc_source"].to_numpy(dtype=object)
        if "predicted_kc_source" in sample_df.columns
        else np.full(row_count, "", dtype=object)
    )
    use_direct_sample_kc = np.asarray(
        [str(value or "").strip() == "" for value in predicted_source],
        dtype=bool,
    )
    if "predicted_kc" not in sample_df.columns:
        sample_df["predicted_kc"] = np.nan
    if "predicted_kc_source" not in sample_df.columns:
        sample_df["predicted_kc_source"] = ""
    if "predicted_load" not in sample_df.columns:
        sample_df["predicted_load"] = np.nan

    if np.any(use_direct_sample_kc):
        predicted_load = np.asarray(idle_power, dtype=float).copy()
        predicted_load[sample_kc_valid_mask] = (
            idle_power[sample_kc_valid_mask]
            + sample_kc_values[sample_kc_valid_mask] * mrr_values[sample_kc_valid_mask]
            + ke_value * ap_values[sample_kc_valid_mask]
        )
        predicted_load = np.maximum(predicted_load, 0.0)
        sample_df.loc[use_direct_sample_kc, "predicted_kc"] = sample_kc_values[use_direct_sample_kc]
        sample_df.loc[use_direct_sample_kc, "predicted_load"] = predicted_load[use_direct_sample_kc]
        sample_df.loc[
            use_direct_sample_kc & sample_kc_valid_mask,
            "predicted_kc_source",
        ] = "measurement_point_kc"
    sample_df.loc[idle_point_mask, "predicted_load"] = np.maximum(idle_power[idle_point_mask], 0.0)
    sample_df["sigma_idle"] = float(sigma_idle)
    sample_df["delta_mrr"] = float(delta_mrr)
    return sample_df


def apply_steady_representative_prediction(
    sample_df: pd.DataFrame,
    intervals: Iterable[dict],
    *,
    ke_value: float,
    mask_resolver: Callable[[dict, int], np.ndarray] | None = None,
    authoritative_interval_kc: bool = False,
) -> pd.DataFrame:
    """稳态区间改用代表 Kc，其他样本保留逐点反解结果。"""
    if sample_df is None or sample_df.empty:
        return sample_df
    interval_records = [dict(item) for item in (intervals or []) if isinstance(item, dict)]
    if not interval_records:
        return sample_df

    frame = sample_df.copy()
    row_count = len(frame)
    predicted_kc = _numeric_column(frame, "predicted_kc", None)
    predicted_load = _numeric_column(frame, "predicted_load", None)
    predicted_source = (
        frame["predicted_kc_source"].astype(str).to_numpy(dtype=object)
        if "predicted_kc_source" in frame.columns
        else np.full(row_count, "", dtype=object)
    )
    interval_summary_kc = (
        _numeric_column(frame, "interval_summary_kc", None)
        if "interval_summary_kc" in frame.columns
        else predicted_kc.copy()
    )
    interval_summary_load = (
        _numeric_column(frame, "interval_summary_load", None)
        if "interval_summary_load" in frame.columns
        else predicted_load.copy()
    )
    interval_summary_source = (
        frame["interval_summary_source"].astype(str).to_numpy(dtype=object)
        if "interval_summary_source" in frame.columns
        else predicted_source.copy()
    )
    kc_point = _numeric_column(frame, "kc_point", None)
    sample_kc = _numeric_column(frame, "sample_kc", None)
    prediction_valid = (
        np.asarray(frame["prediction_valid"], dtype=bool)
        if "prediction_valid" in frame.columns
        else np.ones(row_count, dtype=bool)
    )
    kc_valid = (
        np.asarray(frame["kc_valid"], dtype=bool)
        if "kc_valid" in frame.columns
        else np.isfinite(kc_point)
    )
    sample_kc_valid = (
        np.asarray(frame["sample_kc_valid"], dtype=bool)
        if "sample_kc_valid" in frame.columns
        else np.isfinite(sample_kc)
    )
    idle_point_mask = (
        np.asarray(frame["is_idle_point"], dtype=bool)
        if "is_idle_point" in frame.columns
        else np.zeros(row_count, dtype=bool)
    )
    idle_power = _numeric_column(frame, "idle_power", 0.0)
    ap_values = _numeric_column(frame, "ap", 0.0)
    if "forward_prediction_mrr" in frame.columns:
        mrr_values = _numeric_column(frame, "forward_prediction_mrr", 0.0)
    elif "process_mrr" in frame.columns:
        mrr_values = _numeric_column(frame, "process_mrr", 0.0)
    else:
        mrr_values = _numeric_column(frame, "mrr", 0.0)
    ke_value = max(float(ke_value), 0.0)

    for interval in interval_records:
        interval_mask = np.zeros(row_count, dtype=bool)
        if callable(mask_resolver):
            try:
                resolved_mask = np.asarray(mask_resolver(interval, row_count), dtype=bool)
            except Exception:
                resolved_mask = np.zeros(row_count, dtype=bool)
            if resolved_mask.size == row_count:
                interval_mask = resolved_mask
        if not np.any(interval_mask):
            try:
                start_idx = int(interval.get("sample_start_idx"))
                end_idx = int(interval.get("sample_end_idx"))
            except (TypeError, ValueError):
                continue
            if end_idx < start_idx:
                start_idx, end_idx = end_idx, start_idx
            if start_idx < 0 or end_idx < start_idx or start_idx >= row_count:
                continue
            interval_mask[start_idx:min(end_idx, row_count - 1) + 1] = True

        candidate_mask = interval_mask & prediction_valid & (~idle_point_mask)
        valid_interval_kc = clip_nonnegative_numeric_array(
            kc_point[candidate_mask & kc_valid & np.isfinite(kc_point)]
        )
        if valid_interval_kc.size == 0:
            valid_interval_kc = clip_nonnegative_numeric_array(
                sample_kc[candidate_mask & sample_kc_valid & np.isfinite(sample_kc)]
            )
        if valid_interval_kc.size == 0:
            valid_interval_kc = clip_nonnegative_numeric_array(
                interval_summary_kc[candidate_mask & np.isfinite(interval_summary_kc)]
            )
        runtime_kc_hat, _runtime_sigma_kc, _ = summarize_interval_kc_mode_statistics(valid_interval_kc)
        try:
            saved_kc_hat = float(interval.get("K_c_hat"))
        except (TypeError, ValueError):
            saved_kc_hat = float("nan")
        if authoritative_interval_kc:
            kc_hat = saved_kc_hat
            kc_source = "profile_interval_mode"
        else:
            kc_hat = runtime_kc_hat if np.isfinite(runtime_kc_hat) else saved_kc_hat
            kc_source = "measurement_mode" if np.isfinite(runtime_kc_hat) else str(interval.get("kc_source") or "")
        if not np.isfinite(kc_hat):
            continue
        kc_hat = max(float(kc_hat), 0.0)

        apply_mask = (
            interval_mask
            & prediction_valid
            & (~idle_point_mask)
            & np.isfinite(mrr_values)
            & np.isfinite(ap_values)
            & (mrr_values > 1e-12)
        )
        if not np.any(apply_mask):
            continue
        computed_load = (
            idle_power[apply_mask]
            + kc_hat * mrr_values[apply_mask]
            + ke_value * ap_values[apply_mask]
        )
        computed_load = np.maximum(computed_load, 0.0)
        predicted_kc[apply_mask] = kc_hat
        predicted_load[apply_mask] = computed_load
        predicted_source[apply_mask] = kc_source
        interval_summary_kc[apply_mask] = kc_hat
        interval_summary_load[apply_mask] = computed_load
        interval_summary_source[apply_mask] = kc_source

    frame["predicted_kc"] = predicted_kc
    frame["predicted_load"] = predicted_load
    frame["predicted_kc_source"] = predicted_source
    frame["interval_summary_kc"] = interval_summary_kc
    frame["interval_summary_load"] = interval_summary_load
    frame["interval_summary_source"] = interval_summary_source
    frame["display_predicted_kc"] = predicted_kc
    frame["display_predicted_load"] = predicted_load
    frame["display_prediction_source"] = predicted_source
    return frame


__all__ = [
    "append_inverse_prediction_channels",
    "apply_steady_representative_prediction",
    "clip_nonnegative_numeric_array",
    "estimate_idle_noise_and_mrr_gate",
    "fit_nonnegative_kc_ke",
    "robust_sigma",
    "summarize_interval_kc_mode_statistics",
]
