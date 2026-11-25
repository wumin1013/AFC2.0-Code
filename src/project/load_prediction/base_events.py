"""Non-neural event calibration: steady curve, extrema, timing and phase shapes.

Profiles are conditioned on a settled 0 -> S or S -> 0 transition. Timing is
relative to observed response onset, not an unmeasured controller command.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib

import numpy as np
from scipy.ndimage import median_filter

from .base_power import (BasePowerModel, CalibrationConfig, _event_arrays, _runs,
                         _time, identify_events, speed_features)
from .common import checkpoint, version_of
from .measurement import Measurement


PHASE_GRID = np.linspace(0., 4., 65)
FEATURES = ("steady_w", "extremum_w", "peak_time_s", "settling_time_s", "motion_time_s")
TIMING = "relative_to_observed_response_onset; controller_delay_not_identified"


def validate_profiles(profiles):
    if not profiles:
        raise ValueError("事件模型缺少标定波形")
    keys = set()
    for p in profiles:
        key = (p["kind"], p["level_rpm"])
        if key in keys or key[0] not in ("startup", "braking") or not np.isfinite(key[1]) or key[1] <= 0:
            raise ValueError("事件档位或方向无效/重复")
        keys.add(key)
        times = np.asarray(p["phase_times_s"], float)
        power, speed = np.asarray(p["power_w"], float), np.asarray(p["speed_rpm"], float)
        if (times.shape != (5,) or times[0] != 0 or np.any(np.diff(times) <= 0)
                or power.shape != PHASE_GRID.shape or speed.shape != PHASE_GRID.shape
                or not np.all(np.isfinite(np.concatenate([times, power, speed])))):
            raise ValueError("事件分阶段波形无效")
        scalars = [p[k] for k in ("steady_w", "extremum_w", "peak_time_s", "settling_time_s", "motion_time_s")]
        if not np.isfinite(scalars).all() or min(scalars[2:]) <= 0 or p["settling_time_s"] < p["peak_time_s"]:
            raise ValueError("事件特征参数无效")
        if (not np.isclose(times[1], p["peak_time_s"]) or not np.isclose(times[3], p["settling_time_s"])
                or not np.isclose(power[16], p["extremum_w"]) or not np.isclose(power[-1], p["steady_w"])):
            raise ValueError("事件参数与波形锚点不一致")
        for name in FEATURES:
            span = np.asarray(p["observed_ranges"][name], float)
            if span.shape != (2,) or not np.isfinite(span).all() or span[0] > span[1]:
                raise ValueError("事件实测范围无效")


def _filtered(power, dt, config):
    width = max(1, int(round(config.power_smooth_s / dt)) // 2 * 2 + 1)
    return median_filter(power, size=width, mode="nearest")


def _settling(time, power, target, after, config):
    """First entry into a FINAL steady-power band held for the configured time."""
    tolerance = max(config.settling_absolute_band_w, config.settling_relative_band * abs(target))
    if len(time) < 2:
        return None
    for a, b in _runs((np.abs(power - target) <= tolerance) & (time >= after)):
        if time[b - 1] - time[a] + (time[1] - time[0]) >= config.settling_hold_s:
            return float(time[a])
    return None  # Right-censored: never substitute the end of the record.


def _nominal(value, config):
    return max(0., round(float(value) / config.nominal_step_rpm) * config.nominal_step_rpm)


def _steady_level(record, event):
    # Late plateau avoids contaminating the steady curve with relaxation tails.
    a = max(event["start_s"], event["end_s"] - 1.)
    use = (record.time_s >= a) & (record.time_s <= event["end_s"]) & record.valid
    if not use.any():
        raise ValueError("稳态窗口没有有效功率数据")
    return float(np.median(record.values["spindle_power_w"][use]))


def extract_events(record, events, config, cancel=None):
    """Extract features only from this record; callers split files BEFORE fitting."""
    _event_arrays(record, events)
    time, dt = _time(record.time_s)
    speed, power = record.values["spindle_speed_rpm"], record.values["spindle_power_w"]
    initial = _nominal(speed[0], config)
    output, skipped, levels = [], [], {}
    for i, event in enumerate(events):
        checkpoint(cancel)
        level = float(event["speed_rpm"])
        steady = _steady_level(record, event)
        levels.setdefault(level, []).append(steady)
        origin = initial if i == 0 else float(events[i - 1]["speed_rpm"])
        if origin == level:
            continue
        kind = "startup" if origin == 0 and level > 0 else "braking" if level == 0 and origin > 0 else None
        onset = event.get("command_time_s", event["start_s"])
        a = int(np.searchsorted(time, onset))
        b = int(np.searchsorted(time, event["end_s"], side="right"))
        reason = None
        if kind is None:
            reason = "非零转速直接切换，不属于当前标定范围"
        elif b - a < 5 or not np.all(record.valid[a:b]) or not np.isfinite(power[a:b]).all():
            reason = "事件含缺失/无效数据"
        if reason:
            skipped.append(dict(index=i, reason=reason))
            continue
        t = time[a:b] - time[a]
        y = _filtered(power[a:b], dt, config)
        peak = int(np.argmax(y) if kind == "startup" else np.argmin(y))
        settle = _settling(t, y, steady, t[peak], config)
        if peak == 0 or settle is None or settle <= t[peak] or settle + config.settling_hold_s > t[-1] + dt:
            skipped.append(dict(index=i, reason="峰值或稳态恢复未完整观测（右删失）"))
            continue
        # A distinct fast recovery phase resolves the rapid post-peak fall.
        tail = np.flatnonzero((np.arange(len(t)) > peak) & (np.abs(y - steady) <= .15 * abs(y[peak] - steady)))
        fast = float(t[tail[0]]) if len(tail) else (float(t[peak]) + settle) / 2
        end = min(float(t[-1]), settle + config.settling_hold_s)
        fast = min(max(fast, float(t[peak]) + dt / 4), settle - dt / 4)
        phase_times = [0., float(t[peak]), fast, settle, end]
        grid_time = np.interp(PHASE_GRID, np.arange(5), phase_times)
        shape = np.interp(grid_time, t, y)
        start_power = _steady_level(record, events[i - 1]) if i else float(y[0])
        shape[0], shape[-1] = start_power, steady
        speed_shape = np.interp(grid_time, t, speed[a:b])
        speed_shape[0], speed_shape[-1] = origin, level
        motion = max(dt, event["start_s"] - config.plateau_trim_s - onset)
        output.append(dict(index=i, kind=kind, level_rpm=max(origin, level), onset_s=float(time[a]),
                           source=event.get("source", "estimated"), steady_w=steady,
                           extremum_w=float(y[peak]), peak_time_s=float(t[peak]), settling_time_s=settle,
                           motion_time_s=float(motion), phase_times_s=phase_times,
                           power_w=shape.tolist(), speed_rpm=speed_shape.tolist()))
    return output, skipped, {s: float(np.median(p)) for s, p in levels.items()}


def _interpolate_profile(model, origin, target):
    if not np.isfinite([origin, target]).all() or min(origin, target) < 0:
        raise ValueError("初始和目标转速必须为非负有限数")
    if origin == 0 and target > 0:
        kind, level = "startup", target
    elif origin > 0 and target == 0:
        kind, level = "braking", origin
    else:
        raise ValueError("事件模型仅支持从已稳定停转启动，或从已稳定转速制动到零；非零档位切换待标定")
    profiles = sorted((p for p in model.event_profiles if p["kind"] == kind), key=lambda p: p["level_rpm"])
    levels = [p["level_rpm"] for p in profiles]
    if not levels or not levels[0] <= level <= levels[-1]:
        raise ValueError(f"{kind} 转速超出标定范围，不外推：{levels}")
    result = dict(kind=kind, level_rpm=float(level))
    for name in FEATURES:
        result[name] = float(np.interp(level, levels, [p[name] for p in profiles]))
    for name in ("phase_times_s", "power_w", "speed_rpm"):
        result[name] = [float(np.interp(level, levels, v)) for v in np.array([p[name] for p in profiles]).T]
    result["observed_ranges"] = {name: [float(np.interp(level, levels, v))
                                      for v in np.array([p["observed_ranges"][name] for p in profiles]).T]
                                 for name in FEATURES}
    _condition_recovery(result, CalibrationConfig(**model.config))
    return result


def _condition_recovery(profile, config):
    """A monotone relaxation envelope anchored to the fitted settling time.

Small electrical ripples are not reproduced. The pre-settling envelope stays
    outside the final-power band; the final phase stays inside it. This makes the
    independently fitted time a constraint of the generated curve itself.
    """
    values = np.asarray(profile["power_w"], float)
    steady = profile["steady_w"]
    sign = 1. if profile["extremum_w"] >= steady else -1.
    tolerance = max(config.settling_absolute_band_w, config.settling_relative_band * abs(steady))
    before = (PHASE_GRID > 1) & (PHASE_GRID < 3)
    deviation = sign * (values[before] - steady)
    deviation = np.minimum.accumulate(np.maximum(deviation, tolerance + .001))
    values[before] = steady + sign * deviation
    values[PHASE_GRID == 3] = steady + sign * tolerance
    after = PHASE_GRID > 3
    # Linear final approach is continuous, bounded, and cannot re-enter a ripple.
    values[after] = steady + sign * tolerance * (4 - PHASE_GRID[after])
    profile["power_w"] = values.tolist()


def _evaluate(time, profile):
    grid_time = np.interp(PHASE_GRID, np.arange(5), profile["phase_times_s"])
    return (np.interp(time, grid_time, profile["power_w"]),
            np.interp(time, grid_time, profile["speed_rpm"]))


def predict_event(target_speed_rpm, model, *, initial_speed_rpm=0., sample_period_s=.001, reference=None, cancel=None):
    checkpoint(cancel)
    if model.schema != "afc.base.2":
        raise ValueError("旧版基础模型没有事件参数，请用空载数据重新标定")
    if not np.isfinite(sample_period_s) or sample_period_s <= 0:
        raise ValueError("输出采样周期必须为有限正数")
    profile = _interpolate_profile(model, float(initial_speed_rpm), float(target_speed_rpm))
    from .base_reference import assess_repeatability, correct_profile
    adjustment = correct_profile(profile, model, reference) if reference is not None else None
    end = profile["phase_times_s"][-1] + CalibrationConfig(**model.config).settling_hold_s
    if end / sample_period_s > 2_000_000:
        raise ValueError("单事件输出超过200万点，请增大采样周期")
    if end / sample_period_s < 2:
        raise ValueError("输出至少需要3个采样点，请减小采样周期")
    time = np.arange(0., end + sample_period_s / 2, sample_period_s)
    power, speed = _evaluate(time, profile)
    # Derive the displayed settling time from the displayed curve using exactly
    # the same final-band/hold definition as validation, not an unrelated median.
    settle = _settling(time, power, profile["steady_w"], profile["peak_time_s"], CalibrationConfig(**model.config))
    characteristics = dict(kind=profile["kind"], initial_speed_rpm=float(initial_speed_rpm),
                           target_speed_rpm=float(target_speed_rpm), steady_power_w=profile["steady_w"],
                           extremum_power_w=profile["extremum_w"], peak_time_s=profile["peak_time_s"],
                           settling_time_s=settle,
                           peak_to_settle_s=settle - profile["peak_time_s"] if settle is not None else None)
    # Braking reports a trough explicitly, rather than calling a negative trough a peak.
    characteristics["peak_power_w" if profile["kind"] == "startup" else "trough_power_w"] = profile["extremum_w"]
    return dict(time_s=time, power_w=power, speed_rpm=speed, characteristics=characteristics,
                timing=TIMING, model_version=model.version,
                status=model.status if adjustment is None else "reference_adjusted_candidate",
                parent_model_status=model.status,
                observed_ranges=profile["observed_ranges"],
                range_interpretation="标定重复事件的最小/最大值；中间档位的范围同样来自插值，不是置信区间或保证范围",
                validation_components=model.report.get("validation_components", {}) if adjustment is None else {},
                reference_adjustment=adjustment,
                repeatability=[r for r in assess_repeatability(model)
                               if r["kind"] == profile["kind"] and r["speed_rpm"] == profile["level_rpm"]],
                assumptions=["同机床、相同升降速设置和热状态；事件开始前已稳定", "目标转速间插值尚不等于独立档位验证"])


def _result(time, power, speed, model, valid, warnings, **metadata):
    power[~valid] = np.nan
    return dict(time_s=time, power_w=power, speed_rpm=speed,
                steady_w=np.interp(speed, model.speeds_rpm, model.powers_w), valid=valid,
                model_version=model.version, status=model.status, warnings=sorted(set(warnings)), **metadata)


def _in_steady_domain(speed, model):
    positive = [s for s in model.speeds_rpm if s > 0]
    return (np.abs(speed) <= 30) | ((speed >= min(positive) - 30) & (speed <= max(positive) + 30))


def predict_measured(time_s, speed_rpm, model, *, initial_dynamic_w=None):
    time, dt = _time(time_s)
    if initial_dynamic_w is not None and (not np.isfinite(initial_dynamic_w) or initial_dynamic_w != 0):
        raise ValueError("事件模型暂不支持非零/无效的初始动态功率，请提供完整启动片段或使用未知初态")
    config = CalibrationConfig(**model.config)
    speed, derivative = speed_features(time, speed_rpm, config)
    raw = np.asarray(speed_rpm, float)
    record = Measurement("speed_only", time, {"spindle_speed_rpm": raw, "spindle_power_w": np.zeros(len(time))}, np.ones(len(time), bool))
    events = identify_events(record, config)
    power = np.interp(speed, model.speeds_rpm, model.powers_w)
    valid = _in_steady_domain(raw, model) & (np.abs(derivative) < config.derivative_threshold_rpm_s)
    warnings, predictions = [], []
    initial = _nominal(raw[0], config)
    initial_known = (initial == 0 and abs(raw[0]) <= 30) or initial_dynamic_w == 0
    recovery = max(p["phase_times_s"][-1] for p in model.event_profiles)
    if not initial_known:
        valid[time < time[0] + recovery] = False
        warnings.append("记录开始时动态状态未知，屏蔽保守恢复窗口；其后仅采用稳态假设")
    previous_recovery = 0.
    for i, event in enumerate(events):
        origin = initial if i == 0 else events[i - 1]["speed_rpm"]
        target = event["speed_rpm"]
        if origin == target:
            continue
        onset = event["command_time_s"]
        end = events[i + 1]["command_time_s"] if i + 1 < len(events) else time[-1] + dt
        use = (time >= onset) & (time < end)
        try:
            if i == 0 and not initial_known:
                raise ValueError("记录起始事件缺少已知稳定初态")
            p = _interpolate_profile(model, origin, target)
            motion = max(dt, event["start_s"] - config.plateau_trim_s - onset)
            if abs(motion / p["motion_time_s"] - 1) > config.maximum_motion_relative_error:
                raise ValueError("实测升降速持续时间偏离标定范围")
            if i and onset - events[i - 1]["command_time_s"] < previous_recovery:
                raise ValueError("上次事件尚未稳定，不支持直接拼接")
            power[use], _ = _evaluate(time[use] - onset, p)
            valid[use] = True
            previous_recovery = p["phase_times_s"][-1]
            predictions.append(dict(onset_s=onset, kind=p["kind"], target_speed_rpm=target))
        except ValueError as exc:
            invalid = use & (time < event["start_s"] + recovery)
            valid[invalid] = False
            previous_recovery = recovery
            warnings.append(f"{onset:.3f}s：{exc}")
    # A final ramp with no subsequent stable plateau has no identifiable target.
    dynamic = np.flatnonzero(np.abs(derivative) >= config.derivative_threshold_rpm_s)
    if len(dynamic) and (not events or time[dynamic[-1]] > events[-1]["end_s"]):
        last_covered = events[-1]["end_s"] if events else time[0]
        valid[time > last_covered] = False
        warnings.append("末尾变速未观察到目标平台，无法识别完整事件")
    valid &= (raw >= -30) & (raw <= model.speeds_rpm[-1] + 30)
    return _result(time, power, raw, model, valid, warnings, timing=TIMING, events=predictions,
                   initial_state_source="explicit_or_stopped" if initial_known else "unknown_then_assumed_settled")


def predict_commands(time_s, commands_rpm, model, *, initial_speed_rpm=0., source="estimated", cancel=None):
    checkpoint(cancel)
    time, dt = _time(time_s)
    commands = np.asarray(commands_rpm, float)
    if (commands.shape != time.shape or not np.isfinite(commands).all() or np.any(commands < 0)
            or not np.isfinite(initial_speed_rpm) or initial_speed_rpm < 0):
        raise ValueError("主轴命令必须为与时间等长的非负有限序列，初始转速须有效")
    changes = np.flatnonzero(np.r_[commands[0] != initial_speed_rpm, np.diff(commands) != 0])
    speed = np.full(len(time), initial_speed_rpm, float)
    power = np.full(len(time), np.interp(initial_speed_rpm, model.speeds_rpm, model.powers_w))
    valid = _in_steady_domain(speed, model)
    origin, ready = float(initial_speed_rpm), float(time[0])
    warnings = ["未采集控制器延迟：暂假定给定指令时刻即响应起点，绝对时序未验证"]
    for j, a in enumerate(changes):
        checkpoint(cancel)
        b = changes[j + 1] if j + 1 < len(changes) else len(time)
        target = float(commands[a])
        elapsed = time[a:b] - time[a]
        try:
            p = _interpolate_profile(model, origin, target)
            if time[a] < ready:
                raise ValueError("前一事件尚未稳定，不支持中途切换")
            power[a:b], speed[a:b] = _evaluate(elapsed, p)
            valid[a:b] = True
            ready = time[a] + p["phase_times_s"][-1]
        except ValueError as exc:
            valid[a:b] = False
            speed[a:b] = np.nan
            ready = np.inf
            warnings.append(f"{time[a]:.3f}s：{exc}")
        origin = target
    return _result(time, power, speed, model, valid, warnings, command_source=source,
                   timing="assumed_zero_controller_delay; response_relative_model", initial_state_source="assumed_settled")


def _fit(extracted, config):
    groups, knots = {}, {}
    for observations, skipped, levels in extracted:
        for level, value in levels.items():
            knots.setdefault(level, []).append(value)
        for item in observations:
            groups.setdefault((item["kind"], item["level_rpm"]), []).append(item)
    if len(knots) < 2 or not any(k[0] == "startup" for k in groups) or not any(k[0] == "braking" for k in groups):
        raise ValueError("至少需要两个稳态档位及完整启动、制动事件")
    profiles = []
    for (kind, level), items in sorted(groups.items()):
        p = dict(kind=kind, level_rpm=level, event_count=len(items),
                 observed_ranges={name: [min(item[name] for item in items), max(item[name] for item in items)] for name in FEATURES})
        for name in ("steady_w", "extremum_w", "peak_time_s", "settling_time_s", "motion_time_s",
                     "phase_times_s", "power_w", "speed_rpm"):
            p[name] = np.median([item[name] for item in items], axis=0).tolist()
        _condition_recovery(p, config)
        profiles.append(p)
    validate_profiles(profiles)
    levels = sorted(knots)
    # Legacy rate-model fields are retained solely for schema-1 file compatibility.
    model = BasePowerModel(levels, [float(np.median(knots[s])) for s in levels],
                           0., 0., 1., 1., 1., 1., asdict(config), schema="afc.base.2", event_profiles=profiles)
    model.version = version_of(asdict(model))
    return model


def _metrics(record, events, extracted, model, config):
    result = predict_measured(record.time_s, record.values["spindle_speed_rpm"], model)
    stable, _ = _event_arrays(record, events)
    valid = record.valid & result["valid"]
    actual = record.values["spindle_power_w"]
    # Speed stability and power stability are different. A relaxing tail belongs
    # to the transient evaluation window until the actual power has settled.
    steady_power_mask = stable.copy()
    for item in extracted[0]:
        steady_power_mask[(record.time_s >= item["onset_s"]) &
                          (record.time_s < item["onset_s"] + item["settling_time_s"])] = False
    report = dict(events=[], coverage_fraction=float(valid.mean()), warnings=result["warnings"])
    for name, selection in (("steady", stable), ("transient", ~stable)):
        use = valid & selection
        error = result["power_w"][use] - actual[use]
        report[name] = None
        if use.any():
            report[name] = dict(mae_w=float(np.mean(abs(error))), rmse_w=float(np.sqrt(np.mean(error ** 2))),
                                steady_only_rmse_w=float(np.sqrt(np.mean((result["steady_w"][use] - actual[use]) ** 2))))
    # Retain the original speed-based windows above for comparison with v1,
    # and report the full transient including relaxation separately.
    for name, selection in (("power_steady", steady_power_mask), ("full_transient", ~steady_power_mask)):
        use = valid & selection
        report[name] = None
        if use.any():
            report[name] = dict(mae_w=float(np.mean(np.abs(result["power_w"][use] - actual[use]))),
                                rmse_w=float(np.sqrt(np.mean((result["power_w"][use] - actual[use]) ** 2))),
                                steady_only_rmse_w=float(np.sqrt(np.mean((result["steady_w"][use] - actual[use]) ** 2))))
    legacy_use = valid & ~stable
    if events:
        legacy_use &= record.time_s >= events[0]["end_s"]
    report["legacy_comparable_transient_rmse_w"] = float(np.sqrt(np.mean((result["power_w"][legacy_use] - actual[legacy_use]) ** 2))) if legacy_use.any() else None
    for item in extracted[0]:
        origin, target = (0., item["level_rpm"]) if item["kind"] == "startup" else (item["level_rpm"], 0.)
        try:
            profile = _interpolate_profile(model, origin, target)
        except ValueError as exc:
            report["events"].append(dict(kind=item["kind"], level_rpm=item["level_rpm"], passed=False, reason=str(exc)))
            continue
        event = events[item["index"]]
        use = (record.time_s >= item["onset_s"]) & (record.time_s <= event["end_s"])
        t = record.time_s[use] - item["onset_s"]
        y, _ = _evaluate(t, profile)
        raw = actual[use]
        raw_peak = float(raw.max() if item["kind"] == "startup" else raw.min())
        peak_error = abs(profile["extremum_w"] - item["extremum_w"])
        raw_error = abs(profile["extremum_w"] - raw_peak)
        scale = max(float(np.ptp(raw)), 50.)
        predicted_settling = _settling(t, y, profile["steady_w"], profile["peak_time_s"], config)
        # Validate returned characteristics AND the waveform's actual threshold crossing.
        settle_error = abs(predicted_settling - item["settling_time_s"]) if predicted_settling is not None else None
        peak_time_error = abs(profile["peak_time_s"] - item["peak_time_s"])
        steady_error = abs(profile["steady_w"] - item["steady_w"])
        passed = (max(peak_error, raw_error) / scale <= config.maximum_peak_relative_error
                  and peak_time_error <= config.maximum_peak_time_error_s
                  and settle_error is not None and settle_error <= config.maximum_settling_time_error_s
                  and steady_error <= config.maximum_steady_error_w)
        report["events"].append(dict(kind=item["kind"], level_rpm=item["level_rpm"], onset_s=item["onset_s"],
            peak_error_w=peak_error, raw_peak_error_w=raw_error, peak_relative_error=max(peak_error, raw_error) / scale,
            peak_time_error_s=peak_time_error, settling_time_error_s=settle_error, steady_error_w=steady_error,
            measured={k: item[k] for k in ("extremum_w", "peak_time_s", "settling_time_s", "steady_w")},
            predicted={**{k: profile[k] for k in ("extremum_w", "peak_time_s", "steady_w")},
                       "settling_time_s": predicted_settling}, passed=bool(passed)))
    report["excluded_events"] = extracted[1]
    return report


def fit_event_model(records, *, events=None, config=None, cancel=None):
    config = config or CalibrationConfig()
    config.validate()
    checkpoint(cancel)
    if len(records) < 2:
        raise ValueError("至少两份独立空载记录用于留出验证；当前实验应导入三份")
    all_events = events if events is not None else [identify_events(r, config) for r in records]
    if len(all_events) != len(records):
        raise ValueError("事件清单与空载文件数量不匹配")
    extracted = [extract_events(r, e, config, cancel) for r, e in zip(records, all_events)]
    folds = []
    for i, record in enumerate(records):
        checkpoint(cancel)
        model = _fit([x for j, x in enumerate(extracted) if i != j], config)
        folds.append(dict(held_out=record.source, **_metrics(record, all_events[i], extracted[i], model, config)))
    checkpoint(cancel)
    model = _fit(extracted, config)
    peak_passed = all(f["events"] and all(e.get("peak_relative_error", np.inf) <= config.maximum_peak_relative_error for e in f["events"]) for f in folds)
    passed = all(f["events"] and not f["excluded_events"] and all(e["passed"] for e in f["events"])
                 and f["full_transient"] and f["full_transient"]["rmse_w"] < f["full_transient"]["steady_only_rmse_w"]
                 and f["coverage_fraction"] >= .99 for f in folds)
    model.status = "validated_in_sample_domain" if passed else "calibrated_unvalidated"
    fingerprints = []
    for record in records:
        digest = hashlib.sha256()
        for v in (record.time_s, record.values["spindle_speed_rpm"], record.values["spindle_power_w"]):
            digest.update(np.asarray(v, dtype="<f8").tobytes())
        fingerprints.append(digest.hexdigest())
    model.report = dict(folds=folds, sources=[r.source for r in records], events=all_events,
        event_observations=[x[0] for x in extracted], data_sha256=fingerprints,
        generated_utc=datetime.now(timezone.utc).isoformat(), validation_passed=bool(passed),
        peak_validation_passed=bool(peak_passed), method="steady_curve_and_phase_aligned_event_profiles",
        command_timing=TIMING, rule_version="1.1", acceptance=asdict(config),
        applicability="同机床/热状态/升降速设置的稳定0→S与稳定S→0；非零转速直接切换、未稳定重启和范围外不支持；中间档位仅插值，未独立验证",
        settling_definition="进入最终稳态功率±max(absolute_band,relative_band*abs(Pss))并持续hold_s；相对观测响应起点",
        split="整文件留出；档位、波形和参数只从训练文件生成；留出功率仅用于评分")
    scored = [e for f in folds for e in f["events"]]
    model.report["validation_components"] = dict(
        peak=bool(peak_passed),
        peak_time=all(e.get("peak_time_error_s", np.inf) <= config.maximum_peak_time_error_s for e in scored),
        settling_time=all(e.get("settling_time_error_s") is not None and e["settling_time_error_s"] <= config.maximum_settling_time_error_s for e in scored),
        steady_power=all(e.get("steady_error_w", np.inf) <= config.maximum_steady_error_w for e in scored))
    from .base_reference import assess_repeatability
    model.report["repeatability"] = assess_repeatability(model)
    return model
