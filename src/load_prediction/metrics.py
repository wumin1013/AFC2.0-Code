"""One metric definition for development reports and all user interfaces."""
from __future__ import annotations

import numpy as np

QUANTILES = (.05, .5, .95)
METRIC_DEFINITIONS = {
    "peak_error_w": "abs(max(predicted median) - max(target)) per record",
    "high_load_threshold_w": "90th percentile of training high-confidence steady cutting labels",
    "high_load_underestimate_rate": "mean(predicted median < target) where target >= training threshold",
    "low_load_bias_w": "mean(predicted median - target) where target <= training low threshold; W",
    "aggregation": "equal category within round, then equal round",
    "interval_coverage": "mean(q05 <= target <= q95); direct quantiles, not calibrated coverage guarantee",
}


def quantile_loss(prediction, target, *, quantiles=QUANTILES, weights=None, valid=None):
    """Torch pinball loss; point weights are summed, otherwise valid points averaged."""
    import torch
    if prediction.shape[:-1] != target.shape or prediction.shape[-1] != len(quantiles):
        raise ValueError("分位输出与标签维数不匹配")
    mask = torch.ones_like(target, dtype=torch.bool) if valid is None else valid.bool()
    if mask.shape != target.shape or not mask.any():
        raise ValueError("分位损失没有有效标签")
    if not torch.isfinite(prediction[mask]).all() or not torch.isfinite(target[mask]).all():
        raise ValueError("分位损失包含非有限值")
    error = torch.where(mask, target, torch.zeros_like(target))[..., None] - prediction
    q = torch.as_tensor(quantiles, device=prediction.device, dtype=prediction.dtype)
    losses = torch.maximum(q * error, (q - 1) * error).mean(dim=-1)
    losses = torch.where(mask, losses, torch.zeros_like(losses))
    if weights is None:
        return losses.sum() / mask.sum()
    weights = torch.as_tensor(weights, device=prediction.device, dtype=prediction.dtype)
    if weights.shape != target.shape or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("分位损失点权重无效")
    return (losses * weights).sum()


def regression_metrics(target, prediction, *, lower=None, upper=None, scale_w=1.,
                       high_threshold_w=None, low_threshold_w=None,
                       high_load_threshold_w=None, quantiles=None, allow_empty=False):
    target, prediction = np.asarray(target, float), np.asarray(prediction, float)
    if target.ndim != 1 or target.shape != prediction.shape or not np.isfinite([target, prediction]).all():
        raise ValueError("评价需要等长非空有限标签和预测")
    if not len(target):
        if not allow_empty:
            raise ValueError("评价需要等长非空有限标签和预测")
        return {**{key: None for key in ("mae_w", "rmse_w", "r2", "peak_error_w", "p95_absolute_error_w",
                    "nrmse", "high_load_underestimate_rate", "low_load_overestimate_rate", "low_load_bias_w",
                    "high_upper_exceed_rate", "coverage", "width", "interval_score", "interval_coverage", "interval_width_w")},
                "points": 0, "count": 0, "high_count": 0, "low_count": 0}
    error = prediction - target
    denominator = float(np.square(target - target.mean()).sum())
    result = {"mae_w": float(np.abs(error).mean()), "rmse_w": float(np.sqrt(np.square(error).mean())),
              "r2": float(1 - np.square(error).sum() / denominator) if denominator > 0 else None,
              "peak_error_w": float(abs(prediction.max() - target.max())),
              "p95_absolute_error_w": float(np.quantile(np.abs(error), .95)),
              "high_load_underestimate_rate": None, "low_load_overestimate_rate": None, "low_load_bias_w": None,
              "high_upper_exceed_rate": None, "coverage": None, "width": None, "interval_score": None,
              "points": len(target), "count": len(target), "high_count": 0, "low_count": 0}
    if not np.isfinite(scale_w) or scale_w <= 0:
        raise ValueError("NRMSE尺度必须为训练组确定的正有限值")
    result["nrmse"] = result["rmse_w"] / scale_w
    if high_load_threshold_w is None:
        high_load_threshold_w = high_threshold_w
    if high_load_threshold_w is not None:
        if not np.isfinite(high_load_threshold_w):
            raise ValueError("高负载阈值必须有限")
        high = target >= high_load_threshold_w
        result["high_count"] = int(high.sum())
        result["high_load_underestimate_rate"] = float(np.mean(error[high] < 0)) if high.any() else None
    if low_threshold_w is not None:
        if not np.isfinite(low_threshold_w):
            raise ValueError("低负载阈值必须有限")
        low = target <= low_threshold_w
        result["low_count"] = int(low.sum())
        result["low_load_overestimate_rate"] = float(np.mean(error[low] > 0)) if low.any() else None
        result["low_load_bias_w"] = float(np.mean(error[low])) if low.any() else None
    if (lower is None) != (upper is None):
        raise ValueError("上下分位需同时提供")
    if quantiles is None and lower is not None:
        quantiles = np.column_stack((lower, prediction, upper))
    if quantiles is not None:
        values = np.asarray(quantiles, float)
        if values.shape != (len(target), 3) or not np.isfinite(values).all() or (np.diff(values, axis=1) < 0).any():
            raise ValueError("分位预测必须有限且有序")
        result.update(interval_coverage=float(np.mean((values[:, 0] <= target) & (target <= values[:, 2]))),
                      interval_width_w=float(np.mean(values[:, 2] - values[:, 0])))
        lo, hi = values[:, 0], values[:, 2]
        result.update(coverage=result["interval_coverage"], width=result["interval_width_w"],
                      interval_score=float(np.mean(hi - lo + 20 * np.maximum(lo - target, 0) + 20 * np.maximum(target - hi, 0))))
        if high_load_threshold_w is not None:
            high = target >= high_load_threshold_w
            result["high_upper_exceed_rate"] = float(np.mean(target[high] > hi[high])) if high.any() else None
    return result


def aggregate_metrics(records):
    """Records carry round_id and feature; duplicate records do not gain weight."""
    groups = {}
    for record in records:
        groups.setdefault(record["round_id"], {}).setdefault(record["feature"], []).append(record)
    keys = {key for record in records for key in record} - {"round_id", "feature", "points", "record_id"}
    result = {}
    for key in keys:
        rounds = []
        for categories in groups.values():
            category_means = []
            for rows in categories.values():
                values = [row[key] for row in rows if isinstance(row.get(key), (int, float)) and np.isfinite(row[key])]
                if values:
                    category_means.append(float(np.mean(values)))
            if category_means:
                rounds.append(float(np.mean(category_means)))
        result[key] = float(np.mean(rounds)) if rounds else None
    return result
