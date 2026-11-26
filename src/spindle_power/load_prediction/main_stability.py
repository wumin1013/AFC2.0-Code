"""Power-independent, time-scale-defined spindle screening for main labels.

This is a retrospective label mask, never an input to prospective prediction.
The existing base model and record-local idle labels are not recalculated.
"""
from __future__ import annotations

from collections import Counter
import hashlib

import numpy as np

from .base_power import CalibrationConfig, speed_features
from .feed_samples import _blocks, _period

VERSION = "afc.main_stability.1"
POLICY = dict(version=VERSION, speed_smoothing_s=.101, max_abs_rate_rpm_s=80.,
              speed_tolerance_abs_rpm=30., speed_tolerance_relative=.01,
              stable_interpolation_threshold=.999,
              filter="median3_then_SG2_smooth_then_SG2_derivative",
              support_radius="1+2*(nominal_window//2)=nominal_window_samples",
              minimum_block_samples="2*nominal_window+1",
              transition_guard="abs(rate)>=80_on_complete_support_dilated_by_nominal_window_within_block",
              short_blocks="unsupported_no_shrunk_window_or_raw_gradient_fallback",
              timing="contiguous_original_indices_and_uniform_time_no_resampling",
              interpolation="linear_interpolation_of_binary_qualification_not_window_fraction",
              spatial_support="support_interpolated==1_and_guard_interpolated==0",
              uses_power_for_selection=False, prediction_input=False)

REASONS = {0: "supported", 1: "invalid_measurement_or_speed", 2: "short_complete_support",
           3: "filter_boundary", 4: "nonuniform_time", 5: "transition_guard"}
SPATIAL_COLUMNS = ("spindle_rate_rpm_s", "spindle_rate_support_interpolated",
                   "spindle_rate_support_valid", "spindle_transition_guard_interpolated",
                   "spindle_stable_interpolated")


def _array_hash(values):
    values = np.asarray(values)
    identity = hashlib.sha256(str(values.dtype).encode("ascii"))
    identity.update(str(values.shape).encode("ascii"))
    identity.update(np.ascontiguousarray(values).tobytes())
    return identity.hexdigest()


def speed_support(measurement, *, break_before=None):
    """Calculate rates on full acquisition blocks and explicitly mask support.

    Existing measurement.valid remains the external row-quality boundary. It
    can reflect missing channels; finite power amplitudes are never inspected.
    Ordinary NC line changes are not acquisition discontinuities.
    """
    measurement.require("spindle_speed_rpm")
    time = np.asarray(measurement.time_s, float)
    speed = np.asarray(measurement.values["spindle_speed_rpm"], float)
    valid = np.asarray(measurement.valid, bool)
    if speed.shape != time.shape or valid.shape != time.shape or time.ndim != 1:
        raise ValueError("主趋势稳速输入长度不一致")
    if measurement.source_indices is None:
        raise ValueError("主趋势稳速需要原始采样索引，不能用压缩后行号替代")
    indices = np.asarray(measurement.source_indices)
    if indices.shape != time.shape or indices.dtype.kind not in "iu" or np.any(indices < 0):
        raise ValueError("原始采样索引必须为同长度整数数组")
    breaks = np.zeros(len(time), bool) if break_before is None else np.asarray(break_before)
    if breaks.shape != time.shape or breaks.dtype.kind != "b":
        raise ValueError("显式执行断点必须为同长度布尔数组")
    period = _period(time, measurement.metadata)
    if period is None or not np.isfinite(period) or period <= 0:
        raise ValueError("主趋势稳速缺少有效采样周期")
    rate = np.full(len(time), np.nan)
    support, guard = np.zeros(len(time), bool), np.zeros(len(time), bool)
    reason = np.full(len(time), 1, np.uint8)
    valid = valid & np.isfinite(time) & np.isfinite(speed)
    blocks = []
    for first, last in _blocks(time, valid, period):
        split = np.flatnonzero((np.diff(indices[first:last]) != 1) | breaks[first + 1:last]) + first + 1
        for start, stop in zip(np.r_[first, split], np.r_[split, last]):
            start, stop = int(start), int(stop)
            dt = float(np.median(np.diff(time[start:stop]))) if stop - start > 1 else float(period)
            window = max(5, int(POLICY["speed_smoothing_s"] / dt) // 2 * 2 + 1)
            row = dict(start=int(indices[start]), stop=int(indices[stop - 1]) + 1,
                       array_start=start, array_stop=stop, dt_s=dt, nominal_window=window,
                       support_radius=window, minimum_block_samples=2 * window + 1)
            if not np.allclose(np.diff(time[start:stop]), dt, rtol=.001, atol=1e-9):
                reason[start:stop] = 4
                row["status"] = "nonuniform_time"
            elif stop - start < 2 * window + 1:
                reason[start:stop] = 2
                row["status"] = "short_complete_support"
            else:
                _, derivative = speed_features(time[start:stop], speed[start:stop],
                                               CalibrationConfig(smooth_s=POLICY["speed_smoothing_s"]))
                rate[start:stop] = derivative
                reason[start:stop] = 3
                support[start + window:stop - window] = True
                reason[start + window:stop - window] = 0
                # Dilate only trusted dynamic detections; a missing block must
                # never receive a neighbour's transition protection.
                dynamic = support[start:stop] & (np.abs(derivative) >= POLICY["max_abs_rate_rpm_s"])
                changes = np.zeros(stop - start + 1, np.int64)
                marked = np.flatnonzero(dynamic)
                np.add.at(changes, np.maximum(0, marked - window), 1)
                np.add.at(changes, np.minimum(stop - start, marked + window + 1), -1)
                guard[start:stop] = np.cumsum(changes[:-1]) > 0
                reason[start:stop][support[start:stop] & guard[start:stop]] = 5
                row.update(status="processed", numerical_support_points=int(support[start:stop].sum()),
                           dynamic_points=int(dynamic.sum()), transition_guard_points=int(guard[start:stop].sum()))
            blocks.append(row)
    arrays = dict(source_indices=indices.copy(), time_s=time.copy(), rate_rpm_s=rate,
                  numerical_support_valid=support, transition_guard=guard, reason_code=reason)
    if measurement.source_lines is not None:
        arrays["source_lines"] = np.asarray(measurement.source_lines).copy()
    audit = dict(version=VERSION, policy=dict(POLICY), raw_samples=len(time), blocks=blocks,
                 reason_codes=REASONS, reason_counts={REASONS[k]: int(v) for k, v in Counter(reason).items()})
    return arrays, audit


def stability_channels(frame, measurement, *, line_map=None, break_before=None):
    arrays, audit = speed_support(measurement, break_before=break_before)
    speed = np.asarray(measurement.values["spindle_speed_rpm"], float)
    raw_lines = np.asarray(measurement.values["program_line_raw"], float)
    command = np.full(len(speed), np.nan)
    for raw_line in np.unique(raw_lines[np.isfinite(raw_lines)]):
        if raw_line != round(raw_line):
            continue
        line = (line_map or {}).get(int(raw_line), int(raw_line))
        commands = frame.loc[frame.line == line, "spindle_cmd_rpm"].unique()
        if len(commands) == 1 and np.isfinite(commands[0]):
            command[raw_lines == raw_line] = float(commands[0])
    band = np.abs(speed - command) <= np.maximum(POLICY["speed_tolerance_abs_rpm"],
                                                POLICY["speed_tolerance_relative"] * command)
    stable = (arrays["numerical_support_valid"] & ~arrays["transition_guard"] & band
              & (np.abs(arrays["rate_rpm_s"]) < POLICY["max_abs_rate_rpm_s"]))
    arrays.update(command_rpm=command, raw_speed_band_valid=band, raw_speed_qualified=stable)
    audit.update(raw_speed_band_points=int(band.sum()), raw_speed_qualified_points=int(stable.sum()),
                 array_sha256={key: _array_hash(value) for key, value in arrays.items()})
    channels = dict(spindle_rate_rpm_s=arrays["rate_rpm_s"],
                    spindle_rate_support_interpolated=arrays["numerical_support_valid"].astype(float),
                    spindle_transition_guard_interpolated=arrays["transition_guard"].astype(float),
                    spindle_stable_interpolated=stable.astype(float))
    return channels, arrays, audit


def apply_stability_columns(frame, aligned):
    """Attach diagnostics without changing geometry, labels or alignment."""
    for key in SPATIAL_COLUMNS:
        if key != "spindle_rate_support_valid":
            frame[key] = aligned[key].to_numpy(float)
    frame["spindle_rate_support_valid"] = (aligned.spindle_rate_support_interpolated.eq(1.)
                                            & frame.aligned).to_numpy(bool)
    frame["spindle_stable"] = (frame.spindle_rate_support_valid
                               & frame.spindle_transition_guard_interpolated.eq(0.)
                               & (frame.spindle_stable_interpolated >= POLICY["stable_interpolation_threshold"]))


def verify_stability_frame(sample):
    from pandas.api.types import is_bool_dtype
    frame, meta = sample.frame, sample.metadata
    if meta.get("main_stability_rules_version") != VERSION or set(SPATIAL_COLUMNS) - set(frame):
        raise ValueError("负载快照缺少当前主趋势稳速规则和中间量")
    for key in ("spindle_rate_support_valid", "spindle_stable", "valid_main", "valid_residual", "steady"):
        if not is_bool_dtype(frame[key]) or frame[key].isna().any():
            raise ValueError("主趋势稳速及监督遮罩必须为完整布尔值")
    if not frame.spindle_rate_support_valid.equals(frame.spindle_rate_support_interpolated.eq(1.) & frame.aligned):
        raise ValueError("完整导数支撑遮罩与空间插值及对齐不一致")
    for column in ("spindle_rate_support_interpolated", "spindle_transition_guard_interpolated", "spindle_stable_interpolated"):
        values = frame.loc[frame.aligned, column].to_numpy(float)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("稳速空间插值必须在[0,1]内且对齐处有限")
    expected = (frame.spindle_rate_support_valid & frame.spindle_transition_guard_interpolated.eq(0.)
                & (frame.spindle_stable_interpolated >= POLICY["stable_interpolation_threshold"]))
    if not frame.spindle_stable.equals(expected):
        raise ValueError("稳速遮罩与版本化空间中间量不一致")
    if not frame.valid_main.equals(frame.valid_residual & frame.steady & frame.spindle_stable):
        raise ValueError("主趋势遮罩与残差、工艺稳态和稳速条件不一致")
    if not np.isfinite(frame.loc[frame.spindle_rate_support_valid, "spindle_rate_rpm_s"]).all():
        raise ValueError("有效导数支撑缺少有限导数")
