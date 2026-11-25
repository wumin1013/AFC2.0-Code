"""Non-neural steady curve and transient power identification."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import lfilter, savgol_filter

from .common import Cancel, checkpoint, save_json


@dataclass
class CalibrationConfig:
    smooth_s: float = .101
    derivative_threshold_rpm_s: float = 80.
    minimum_plateau_s: float = .5
    plateau_trim_s: float = .2
    nominal_step_rpm: float = 100.
    fit_period_s: float = .01
    max_nfev: int = 180
    maximum_peak_relative_error: float = .25
    power_smooth_s: float = .021
    settling_relative_band: float = .05
    settling_absolute_band_w: float = 5.
    settling_hold_s: float = .5
    maximum_peak_time_error_s: float = .15
    maximum_settling_time_error_s: float = .5
    maximum_steady_error_w: float = 15.
    maximum_motion_relative_error: float = .2

    def validate(self):
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("标定配置必须为有限正数")


@dataclass
class BasePowerModel:
    speeds_rpm: list
    powers_w: list
    k_up: float
    k_down: float
    tau_power_s: float
    tau_spindle_s: float
    accel_up_rpm_s: float
    accel_down_rpm_s: float
    config: dict
    report: dict = field(default_factory=dict)
    version: str = ""
    status: str = "calibrated_unvalidated"
    schema: str = "afc.base.1"
    event_profiles: list = field(default_factory=list)

    def save(self, path):
        save_json(path, self)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("schema") not in ("afc.base.1", "afc.base.2"):
            raise ValueError("不支持的基础功率模型版本")
        model = cls(**payload)
        if len(model.speeds_rpm) < 2 or len(model.speeds_rpm) != len(model.powers_w):
            raise ValueError("稳态曲线无效")
        if np.any(np.diff(model.speeds_rpm) <= 0) or not np.all(np.isfinite(model.speeds_rpm + model.powers_w)):
            raise ValueError("稳态曲线非有限或转速非递增")
        parameters = [model.k_up, model.k_down, model.tau_power_s, model.tau_spindle_s, model.accel_up_rpm_s, model.accel_down_rpm_s]
        if not np.isfinite(parameters).all() or min(model.k_up, model.k_down) < 0 or min(parameters[2:]) <= 0:
            raise ValueError("动态参数无效")
        CalibrationConfig(**model.config).validate()
        if model.schema == "afc.base.2":
            from .base_events import validate_profiles
            validate_profiles(model.event_profiles)
        return model


def _time(time):
    time = np.asarray(time, float)
    if len(time) < 3 or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError("时间至少3点、有限且严格递增")
    dt = float(np.median(np.diff(time)))
    if not np.allclose(np.diff(time), dt, rtol=.001, atol=1e-9):
        raise ValueError("基础功率输入需要等间隔采样，不可压缩缺失点时间")
    return time, dt


def speed_features(time, speed, config):
    time, dt = _time(time)
    speed = np.asarray(speed, float)
    if speed.shape != time.shape or not np.all(np.isfinite(speed)):
        raise ValueError("转速含无效点或长度不匹配，先分割连续有效片段")
    window = min(len(speed) // 2 * 2 - 1, max(5, int(config.smooth_s / dt) // 2 * 2 + 1))
    if window < 5:
        return speed.copy(), np.gradient(speed, time)
    smoothed = savgol_filter(median_filter(speed, size=3), window, 2)
    return smoothed, savgol_filter(smoothed, window, 2, deriv=1, delta=dt)


def _runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)))
    return list(zip(edges[::2], edges[1::2]))


def identify_events(record, config=None):
    config = config or CalibrationConfig()
    config.validate()
    record.require("spindle_speed_rpm", "spindle_power_w")
    speed, derivative = speed_features(record.time_s, record.values["spindle_speed_rpm"], config)
    dt = record.time_s[1] - record.time_s[0]
    stable = (np.abs(derivative) < config.derivative_threshold_rpm_s) & record.valid
    events = []
    for start, stop in _runs(stable):
        if (stop - start) * dt < config.minimum_plateau_s:
            continue
        trim = min(int(config.plateau_trim_s / dt), (stop - start) // 4)
        a, b = start + trim, stop - trim
        nominal = round(float(np.median(speed[a:b])) / config.nominal_step_rpm) * config.nominal_step_rpm
        events.append(dict(start_s=float(record.time_s[a]), end_s=float(record.time_s[b - 1]),
                           speed_rpm=max(0., nominal), kind="steady", source="estimated"))
    # Timing is the observed motion onset, not a measured controller command.
    previous_end = 0
    for event in events:
        a = int(np.searchsorted(record.time_s, event["start_s"]))
        dynamic = np.flatnonzero(np.abs(derivative[previous_end:a]) >= config.derivative_threshold_rpm_s)
        onset = previous_end + int(dynamic[0]) if len(dynamic) else a
        event["command_time_s"] = float(record.time_s[onset])
        previous_end = int(np.searchsorted(record.time_s, event["end_s"]))
    return events


def _event_arrays(record, events):
    time = record.time_s
    stable = np.zeros(len(time), bool)
    command = np.full(len(time), record.values["spindle_speed_rpm"][0], float)
    last_end = -np.inf
    for event in events:
        start, end = float(event["start_s"]), float(event["end_s"])
        level = float(event["speed_rpm"])
        onset = float(event.get("command_time_s", start))
        if not np.all(np.isfinite([start, end, level, onset])) or onset < last_end or end < start or onset > start or onset < time[0] or end > time[-1] or level < 0:
            raise ValueError("审核事件必须按时间排序、不重叠且位于记录范围内")
        if event.get("source", "estimated") not in ("estimated", "measured", "nc_aligned"):
            raise ValueError("未知指令来源")
        last_end = end
        stable |= (time >= start) & (time <= end)
        command[time >= onset] = level
    return stable, command


def _response(derivative, dt, up, down, tau, initial=0.):
    decay = np.exp(-dt / tau)
    drive = up * np.maximum(derivative, 0) - down * np.maximum(-derivative, 0)
    # q[0] is an explicit initial state; forcing of sample i acts over [i-1,i].
    return np.r_[initial, lfilter([1 - decay], [1, -decay], drive[1:], zi=[decay * initial])[0]]


def predict_base_from_measured_speed(time_s, speed_rpm, model, *, initial_dynamic_w=None,
                                     reference=None, session=None, record_source=None,
                                     event_timeline=None, cancel=None,
                                     qualification=None, record_context=None):
    """Keep speed-only compatibility; explicitly opt in to audited NC timing.

    ``event_timeline`` is an AuditedEventTimeline bound to time/speed observations.
    It uses no target power and does not upgrade independent validation status.
    """
    if event_timeline is not None:
        from .base_timeline import predict_audited_measured
        if initial_dynamic_w is not None and initial_dynamic_w != 0:
            raise ValueError("审核事件接口仅支持已确认的稳定初态，不支持非零初始动态功率")
        result = predict_audited_measured(time_s, speed_rpm, model, event_timeline,
                                         record_source=record_source, cancel=cancel,
                                         qualification=qualification, record_context=record_context)
        if reference is not None:
            from .base_reference import apply_idle_reference
            result = apply_idle_reference(result, model, reference, session=session, record_source=record_source)
        return result
    if model.schema == "afc.base.2":
        from .base_events import predict_measured
        from .base_timeline import SPEED_ONLY_INTERFACE_VERSION
        result = predict_measured(time_s, speed_rpm, model, initial_dynamic_w=initial_dynamic_w)
        result["interface_version"] = SPEED_ONLY_INTERFACE_VERSION
        if reference is not None:
            from .base_reference import apply_idle_reference
            result = apply_idle_reference(result, model, reference, session=session, record_source=record_source)
        return result
    if reference is not None:
        raise ValueError("空载参考校正需要事件基础模型")
    time, dt = _time(time_s)
    config = CalibrationConfig(**model.config)
    speed, derivative = speed_features(time, speed_rpm, config)
    steady = np.interp(speed, model.speeds_rpm, model.powers_w)
    power = steady + _response(derivative, dt, model.k_up, model.k_down, model.tau_power_s,
                              0. if initial_dynamic_w is None else initial_dynamic_w)
    raw = np.asarray(speed_rpm)
    valid = (raw >= model.speeds_rpm[0] - 30) & (raw <= model.speeds_rpm[-1] + 30)
    state_source = "explicit" if initial_dynamic_w is not None else "inferred_initial_steady"
    if initial_dynamic_w is None and abs(derivative[0]) >= config.derivative_threshold_rpm_s:
        state_source = "unknown_until_first_steady_window"
        settled = next((b for a, b in _runs(np.abs(derivative) < config.derivative_threshold_rpm_s)
                        if (b - a) * dt >= max(config.minimum_plateau_s, 5 * model.tau_power_s)), len(valid))
        valid[:settled] = False
    power[~valid] = np.nan
    return dict(time_s=time, power_w=power, steady_w=steady, valid=valid,
                model_version=model.version, status=model.status,
                interface_version="afc.base.rate_measured_speed.1",
                initial_state_source=state_source,
                warnings=["低速无平台区间仅作过渡插值；超范围点不外推"])


def spindle_response(time_s, commands, tau, up, down, initial_speed=0., cancel=None):
    time, _ = _time(time_s)
    commands = np.asarray(commands, float)
    if commands.shape != time.shape or not np.all(np.isfinite(commands)) or np.any(commands < 0):
        raise ValueError("主轴命令必须为与时间等长的非负有限序列")
    speed = np.empty(len(time))
    speed[0] = initial_speed
    for i, dt in enumerate(np.diff(time), 1):
        if i % 10000 == 0:
            checkpoint(cancel)
        # Exact first-order step capped by physical acceleration limits.
        delta = (commands[i - 1] - speed[i - 1]) * (1 - np.exp(-dt / tau))
        speed[i] = speed[i - 1] + min(up * dt, max(-down * dt, delta))
    return speed


def predict_base_from_commands(time_s, commands_rpm, model, *, initial_speed_rpm=0., source="estimated",
                                reference=None, session=None, record_source=None, cancel=None):
    if model.schema == "afc.base.2":
        from .base_events import predict_commands
        result = predict_commands(time_s, commands_rpm, model, initial_speed_rpm=initial_speed_rpm, source=source, cancel=cancel)
        if reference is not None:
            from .base_reference import apply_idle_reference
            result = apply_idle_reference(result, model, reference, session=session, record_source=record_source)
        checkpoint(cancel)
        return result
    if reference is not None:
        raise ValueError("空载参考校正需要事件基础模型")
    speed = spindle_response(time_s, commands_rpm, model.tau_spindle_s,
                             model.accel_up_rpm_s, model.accel_down_rpm_s, initial_speed_rpm, cancel)
    result = predict_base_from_measured_speed(time_s, speed, model, initial_dynamic_w=0.)
    result.update(speed_rpm=speed, command_source=source, timing="relative_to_supplied_command_timeline")
    return result


def fit_base_power(records, *, events=None, config=None, cancel: Cancel = None):
    """Calibrate phase-aligned startup/braking profiles, without neural training."""
    from .base_events import fit_event_model
    return fit_event_model(records, events=events, config=config, cancel=cancel)


def predict_base_event(target_speed_rpm, model, *, initial_speed_rpm=0., sample_period_s=.001, reference=None,
                       session=None, record_source=None, onset_s=None, cancel=None):
    """Query a settled stop/start event and its peak/settling characteristics."""
    from .base_events import predict_event
    if reference is not None and (reference.session is not None or session is not None):
        from .base_reference import reference_use_mask
        if onset_s is None:
            raise ValueError("带会话参考的事件查询需填写目标响应起点onset_s")
        reference_use_mask([onset_s], model, reference, session=session, record_source=record_source)
    result = predict_event(target_speed_rpm, model, initial_speed_rpm=initial_speed_rpm,
                           sample_period_s=sample_period_s, reference=reference, cancel=cancel)
    if reference is not None:
        result["reference_usage"] = (dict(session=session, source=record_source, onset_s=onset_s)
                                      if reference.session is not None else
                                      dict(scope="无会话绑定的离线波形查询；不得用于整段预测或切削标签"))
    return result
