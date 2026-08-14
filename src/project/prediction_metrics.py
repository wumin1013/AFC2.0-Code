from __future__ import annotations

import math
import os
from typing import Dict, Iterable

import numpy as np
import pandas as pd


def get_relative_error_floor(actual: Iterable[float]) -> float:
    actual_array = np.asarray(list(actual), dtype=float)
    valid_actual = np.abs(actual_array[np.isfinite(actual_array)])
    positive_actual = valid_actual[valid_actual > 1e-9]
    if positive_actual.size == 0:
        return math.nan
    return float(max(50.0, np.percentile(positive_actual, 95) * 0.05))


def compute_error_metrics(actual: Iterable[float], predicted: Iterable[float]) -> Dict[str, float]:
    actual_array = np.asarray(list(actual), dtype=float)
    predicted_array = np.asarray(list(predicted), dtype=float)
    valid_mask = np.isfinite(actual_array) & np.isfinite(predicted_array)
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

    actual_valid = actual_array[valid_mask]
    predicted_valid = predicted_array[valid_mask]
    error = predicted_valid - actual_valid
    absolute_error = np.abs(error)
    absolute_actual = np.abs(actual_valid)
    relative_error_floor = get_relative_error_floor(actual_valid)

    raw_mask = absolute_actual > 1e-9
    raw_mape = (
        float(np.mean(np.abs(error[raw_mask] / actual_valid[raw_mask])) * 100.0)
        if np.any(raw_mask)
        else math.nan
    )
    filtered_mask = raw_mask.copy()
    if np.isfinite(relative_error_floor):
        filtered_mask &= absolute_actual >= float(relative_error_floor)
    mape = (
        float(np.mean(np.abs(error[filtered_mask] / actual_valid[filtered_mask])) * 100.0)
        if np.any(filtered_mask)
        else math.nan
    )
    actual_sum = float(np.sum(absolute_actual))
    wmape = float(np.sum(absolute_error) / actual_sum * 100.0) if actual_sum > 1e-9 else math.nan

    return {
        "count": int(valid_mask.sum()),
        "mae": float(np.mean(absolute_error)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mape": mape,
        "wmape": wmape,
        "raw_mape": raw_mape,
        "relative_error_floor": relative_error_floor,
        "relative_valid_count": int(np.sum(filtered_mask)),
        "max_abs_error": float(np.max(absolute_error)),
    }


def export_table_to_csv(frame: pd.DataFrame, output_path: str) -> str:
    target_path = os.path.abspath(output_path)
    frame.to_csv(target_path, index=False, encoding="utf-8-sig")
    return target_path
