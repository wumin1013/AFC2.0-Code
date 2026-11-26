"""Versioned per-record feed evaluation on an NC arclength grid.

Thresholds are supplied/frozen during development; this module never fits them
from validation or independent test trajectories. Region masks may overlap.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .sample_rules import continuous_ranges

FEED_METRICS_VERSION = "afc.feed_metrics.1"


@dataclass(frozen=True)
class FeedMetricConfig:
    gradient_threshold_mm_min_per_mm: float = 20.
    minimum_deceleration_distance_mm: float = 1.
    minimum_deceleration_points: int = 3
    onset_match_distance_mm: float = 5.
    corner_angle_deg: float = 30.
    corner_window_mm: float = 2.
    short_block_mm: float = 2.

    def validate(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"进给评价配置 {key} 必须为正有限数")
        if type(self.minimum_deceleration_points) is not int or self.minimum_deceleration_points < 2:
            raise ValueError("减速事件至少需要两个空间点")
        if self.corner_angle_deg > 180:
            raise ValueError("拐角阈值不能超过180度")


def _point_metrics(target, prediction):
    if not len(target):
        return {"mae_mm_min": None, "rmse_mm_min": None, "minimum_feed_error_mm_min": None, "points": 0}
    error = prediction - target
    return {"mae_mm_min": float(np.abs(error).mean()), "rmse_mm_min": float(np.sqrt(np.square(error).mean())),
            "minimum_feed_error_mm_min": float(abs(prediction.min() - target.min())), "points": len(target)}


def _dynamics(frame, values, valid, config):
    acceleration, deceleration, events = np.zeros(len(frame), bool), np.zeros(len(frame), bool), []
    position = frame.path_mm.to_numpy(float)
    for start, stop in continuous_ranges(frame, valid):
        if stop - start < 2:
            continue
        s, feed = position[start:stop], values[start:stop]
        ds = np.diff(s)
        gradient = np.divide(np.diff(feed), ds, out=np.full(len(ds), np.nan), where=ds > 1e-8)
        up = gradient > config.gradient_threshold_mm_min_per_mm
        down = gradient < -config.gradient_threshold_mm_min_per_mm
        for mask, output in ((up, acceleration), (down, deceleration)):
            indices = np.flatnonzero(mask) + start
            output[indices] = True
            output[indices + 1] = True
        boundaries = np.flatnonzero(np.diff(np.r_[False, down, False].astype(int)))
        for first, last in zip(boundaries[::2], boundaries[1::2]):
            if (last - first + 1 >= config.minimum_deceleration_points
                    and s[last] - s[first] >= config.minimum_deceleration_distance_mm):
                events.append({"path_mm": float(s[first]), "row": int(start + first),
                               "sequence_start": int(start), "distance_mm": float(s[last] - s[first])})
    return acceleration, deceleration, events


def _match_events(actual, predicted, config):
    candidates = sorted((abs(a["path_mm"] - p["path_mm"]), i, j)
                        for i, a in enumerate(actual) for j, p in enumerate(predicted)
                        if a["sequence_start"] == p["sequence_start"]
                        and abs(a["path_mm"] - p["path_mm"]) <= config.onset_match_distance_mm)
    used_actual, used_predicted, matches = set(), set(), []
    for distance, i, j in candidates:
        if i not in used_actual and j not in used_predicted:
            used_actual.add(i)
            used_predicted.add(j)
            matches.append({"actual_path_mm": actual[i]["path_mm"], "predicted_path_mm": predicted[j]["path_mm"],
                            "absolute_error_mm": float(distance),
                            "signed_error_mm": predicted[j]["path_mm"] - actual[i]["path_mm"]})
    return {"deceleration_onset_error_mm": float(np.mean([m["absolute_error_mm"] for m in matches])) if matches else None,
            "actual_deceleration_events": len(actual), "predicted_deceleration_events": len(predicted),
            "matched_deceleration_events": len(matches), "missed_deceleration_events": len(actual) - len(matches),
            "false_deceleration_events": len(predicted) - len(matches), "deceleration_matches": matches}


def evaluate_feed_record(frame, predicted, *, config=None, target_column="feed_actual_mm_min", valid=None):
    config = config or FeedMetricConfig()
    config.validate()
    if "path_mm" not in frame or target_column not in frame:
        raise ValueError("进给评价需要累计行程和实测实际进给")
    target, predicted = frame[target_column].to_numpy(float), np.asarray(predicted, float)
    if target.shape != predicted.shape or target.ndim != 1:
        raise ValueError("实测与预测进给需要等长一维数组")
    if valid is None:
        valid = frame.valid_label.fillna(False).to_numpy(bool) if "valid_label" in frame else np.ones(len(frame), bool)
    valid = np.asarray(valid, bool)
    if valid.shape != target.shape or not valid.any():
        raise ValueError("进给评价没有有效标签")
    if not np.isfinite([target[valid], predicted[valid], frame.path_mm.to_numpy(float)[valid]]).all() or np.any(target[valid] < 0) or np.any(predicted[valid] < 0):
        raise ValueError("有效位置上的实际/预测进给和行程必须有限，进给不得为负")
    acceleration, deceleration, actual_events = _dynamics(frame, target, valid, config)
    _, _, predicted_events = _dynamics(frame, predicted, valid, config)
    dynamic = acceleration | deceleration
    corners, short = np.zeros(len(frame), bool), np.zeros(len(frame), bool)
    if {"segment_mm", "block_length_mm"} <= set(frame):
        local, length = frame.segment_mm.to_numpy(float), frame.block_length_mm.to_numpy(float)
        short = np.isfinite(length) & (length <= config.short_block_mm)
        threshold = np.deg2rad(config.corner_angle_deg)
        if "entry_angle_rad" in frame:
            corners |= (frame.entry_angle_rad.to_numpy(float) >= threshold) & (local <= config.corner_window_mm)
        if "exit_angle_rad" in frame:
            corners |= (frame.exit_angle_rad.to_numpy(float) >= threshold) & (length - local <= config.corner_window_mm)
    steady = frame.steady.eq(True).to_numpy(bool) & ~dynamic if "steady" in frame else np.zeros(len(frame), bool)
    masks = {"steady": steady, "corners": corners, "short_blocks": short,
             "acceleration_deceleration": dynamic, "acceleration": acceleration, "deceleration": deceleration}
    groups = {name: _point_metrics(target[valid & mask], predicted[valid & mask]) for name, mask in masks.items()}
    result = _point_metrics(target[valid], predicted[valid])
    result.update(_match_events(actual_events, predicted_events, config))
    result.update(groups=groups, config=asdict(config), metric_version=FEED_METRICS_VERSION,
                  configuration_source="frozen_development_configuration", region_groups_overlap=True,
                  actual_events=actual_events, predicted_events=predicted_events,
                  minimum_feed_definition="abs(min(predicted)-min(actual)) on valid positions",
                  onset_definition="start of sustained negative spatial feed gradient; unmatched events reported separately")
    return result


def aggregate_feed_metrics(records):
    """Equal complete records within feature, then equal features and rounds."""
    records = list(records)
    groups = {}
    for index, record in enumerate(records):
        groups.setdefault(record.get("round_id", f"record-{index}"), {}).setdefault(record.get("feature", "record"), []).append(record["metrics"])
    keys = ("mae_mm_min", "rmse_mm_min", "minimum_feed_error_mm_min", "deceleration_onset_error_mm")

    def reduce(region=None):
        result = {}
        for key in keys:
            round_values, available = [], 0
            for features in groups.values():
                feature_values = []
                for metrics in features.values():
                    selected = [m if region is None else m.get("groups", {}).get(region, {}) for m in metrics]
                    values = [m[key] for m in selected if isinstance(m.get(key), (int, float)) and np.isfinite(m[key])]
                    available += len(values)
                    if values:
                        feature_values.append(float(np.mean(values)))
                if feature_values:
                    round_values.append(float(np.mean(feature_values)))
            result[key] = float(np.mean(round_values)) if round_values else None
            result[f"{key}_effective_records"] = available
        return result

    result = reduce()
    names = sorted({name for record in records for name in record["metrics"].get("groups", {})})
    result.update(groups={name: reduce(name) for name in names}, records=len(records), rounds=len(groups),
                  metric_version=FEED_METRICS_VERSION, weighting="equal_record_within_feature_then_equal_feature_then_equal_round",
                  missed_deceleration_events=sum(r["metrics"].get("missed_deceleration_events", 0) for r in records),
                  false_deceleration_events=sum(r["metrics"].get("false_deceleration_events", 0) for r in records))
    return result
