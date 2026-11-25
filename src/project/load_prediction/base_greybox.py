"""Low-order grey-box power identification; no neural runtime or GUI imports.

P = f(S) + b*S/1000 + d_up*u+ - d_down*u- + sum(x)
tau_j * dx_j/dt + x_j = g_up,j*u+ - g_down,j*u-
u is d(S/1000)/dt, optionally multiplied by S/1000 (inertial proxy).
Time constants/gains are shared. Each calibration run has its own b and x(0).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter

from .base_power import BasePowerModel, CalibrationConfig, _time, identify_events, speed_features
from .base_events import (_fit, _filtered, _interpolate_profile, _evaluate,
                          _settling, extract_events, validate_profiles, TIMING)
from .common import checkpoint, save_json, version_of
from .measurement import Measurement


@dataclass
class GreyBoxConfig:
    fit_period_s: float = .02
    max_nfev: int = 120
    robust_scale_w: float = 5.
    offset_bound_w_per_krpm: float = 10.
    initial_state_bound_w: float = 200.
    gain_bound: float = 2000.
    tau_min_s: float = .005
    tau_max_s: float = 20.

    def validate(self):
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("灰箱配置必须为有限正数")
        if not isinstance(self.max_nfev, int) or self.tau_min_s >= self.tau_max_s:
            raise ValueError("灰箱迭代次数或时间常数范围无效")


@dataclass
class GreyBoxModel:
    order: int
    drive: str
    taus_s: list
    gains_up: list  # direct feedthrough, followed by the state gains
    gains_down: list
    baseline: dict
    fitting: dict
    version: str = ""
    status: str = "research_candidate_unvalidated"
    report: dict = field(default_factory=dict)
    schema: str = "afc.greybox.1"

    def payload(self):
        return {k: v for k, v in asdict(self).items() if k not in ("version", "status", "report")}

    def validate(self):
        cfg = GreyBoxConfig(**self.fitting)
        cfg.validate()
        if self.schema != "afc.greybox.1" or self.order not in (1, 2) or self.drive not in ("acceleration", "inertial"):
            raise ValueError("不支持的灰箱模型结构")
        if len(self.taus_s) != self.order or len(self.gains_up) != self.order + 1 or len(self.gains_down) != self.order + 1:
            raise ValueError("灰箱参数维度错误")
        values = self.taus_s + self.gains_up + self.gains_down
        if (not np.isfinite(values).all() or min(self.gains_up + self.gains_down) < 0
                or max(self.gains_up + self.gains_down) > cfg.gain_bound + 1e-8
                or min(self.taus_s) < cfg.tau_min_s or max(self.taus_s) > cfg.tau_max_s + 1e-8
                or np.any(np.diff(self.taus_s) < 0)):
            raise ValueError("灰箱参数非有限、越界或时间常数未排序")
        base = BasePowerModel(**self.baseline)
        CalibrationConfig(**base.config).validate()
        if (base.schema != "afc.base.2" or len(base.speeds_rpm) < 2
                or len(base.speeds_rpm) != len(base.powers_w)
                or not np.isfinite(base.speeds_rpm + base.powers_w).all()
                or np.any(np.diff(base.speeds_rpm) <= 0)):
            raise ValueError("灰箱稳态曲线无效")
        validate_profiles(base.event_profiles)
        if self.version != version_of(self.payload()):
            raise ValueError("灰箱模型版本与参数不匹配")

    def save(self, path):
        self.validate()
        save_json(path, self)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as stream:
            model = cls(**json.load(stream))
        model.validate()
        return model


@dataclass(frozen=True)
class GreyBoxState:
    model_version: str
    source: str
    end_s: float
    speed_rpm: float
    offset_w_per_krpm: float
    components_w: list
    prefix_sha256: str
    no_load_confirmed: bool = True

    def validate(self, model):
        cfg = GreyBoxConfig(**model.fitting)
        if (self.model_version != model.version or self.no_load_confirmed is not True
                or len(self.components_w) != model.order
                or not np.isfinite([self.end_s, self.speed_rpm, self.offset_w_per_krpm] + self.components_w).all()
                or abs(self.offset_w_per_krpm) > cfg.offset_bound_w_per_krpm + 1e-8
                or self.speed_rpm < -30 or self.speed_rpm > max(model.baseline["speeds_rpm"]) + 30):
            raise ValueError("灰箱初始状态无效或模型版本不匹配")


def _inputs(time, speed, config, drive):
    smooth, rate = speed_features(time, speed, config)
    # Do not turn stationary encoder ripple into a rectified energy source.
    rate[np.abs(rate) < config.derivative_threshold_rpm_s] = 0.
    u = rate / 1000.
    if drive == "inertial":
        u = u * np.maximum(smooth, 0.) / 1000.
    return np.maximum(smooth, 0.), np.maximum(u, 0.), np.maximum(-u, 0.)


def _filter(drive, dt, tau, initial=0.):
    decay = np.exp(-dt / tau)
    return np.r_[initial, lfilter([1. - decay], [1., -decay], drive[1:], zi=[decay * initial])[0]]


def _simulate(time, speed, up, down, base, taus, gu, gd, offset=0., initial=None):
    dt = time[1] - time[0]
    initial = np.zeros(len(taus)) if initial is None else initial
    states = np.asarray([_filter(gu[j + 1] * up - gd[j + 1] * down, dt, tau, initial[j])
                         for j, tau in enumerate(taus)])
    steady = np.interp(speed, base.speeds_rpm, base.powers_w) + offset * speed / 1000.
    return steady + gu[0] * up - gd[0] * down + states.sum(axis=0), states, steady


def _check_record(record):
    record.require("spindle_speed_rpm", "spindle_power_w")
    _time(record.time_s)
    if (np.asarray(record.valid).shape != record.time_s.shape or not record.valid.all()
            or any(np.asarray(record.values[k]).shape != record.time_s.shape or not np.isfinite(record.values[k]).all()
                   for k in ("spindle_speed_rpm", "spindle_power_w"))):
        raise ValueError("灰箱标定需要连续有效记录；不可删除异常点后压缩时间")
    if np.any(record.values["spindle_speed_rpm"] < -30):
        raise ValueError("当前灰箱标定不支持反向主轴转速")


def fit_greybox(records, *, order=1, drive="inertial", config=None, fitting=None,
                events=None, cancel=None):
    """Joint fit on the supplied TRAINING files. One file is allowed for diagnostics."""
    config, fitting = config or CalibrationConfig(), fitting or GreyBoxConfig()
    config.validate()
    fitting.validate()
    if not records or order not in (1, 2) or drive not in ("acceleration", "inertial"):
        raise ValueError("需要空载记录；阶数为1或2，驱动为acceleration或inertial")
    if len({r.source for r in records}) != len(records):
        raise ValueError("不能把同一个采集文件重复计入联合辨识")
    for record in records:
        _check_record(record)
    all_events = events if events is not None else [identify_events(r, config) for r in records]
    if len(all_events) != len(records):
        raise ValueError("事件与记录数量不匹配")
    extracted = [extract_events(r, e, config, cancel) for r, e in zip(records, all_events)]
    base = _fit(extracted, config)
    data = []
    for record in records:
        checkpoint(cancel)
        t, dt = _time(record.time_s)
        stride = max(1, round(fitting.fit_period_s / dt))
        s, up, down = _inputs(t, record.values["spindle_speed_rpm"], config, drive)
        y = _filtered(record.values["spindle_power_w"], dt, config)
        data.append((t[::stride] - t[0], s[::stride], up[::stride], down[::stride], y[::stride]))
    # Equal weight per file. The robust residual scale remains in approximately W.
    weights = [np.sqrt(np.mean([len(d[0]) for d in data]) / len(d[0])) for d in data]
    n_gain = order + 1
    shared = order + 2 * n_gain

    def unpack(p):
        return np.exp(p[:order]), p[order:order + n_gain], p[order + n_gain:shared], p[shared:].reshape(len(data), order + 1)

    def residual(p):
        checkpoint(cancel)
        taus, gu, gd, runs = unpack(p)
        errors = []
        for d, w, run in zip(data, weights, runs):
            pred = _simulate(*d[:4], base, taus, gu, gd, run[0], run[1:])[0]
            errors.append((pred - d[4]) * w)
        # Weak regularisation identifies the least-complex run state when the
        # record starts after its initial response has already disappeared.
        errors.append((runs[:, 1:] / fitting.initial_state_bound_w).ravel())
        return np.concatenate(errors)

    lower = [np.log(fitting.tau_min_s)] * order + [0.] * (2 * n_gain)
    upper = [np.log(fitting.tau_max_s)] * order + [fitting.gain_bound] * (2 * n_gain)
    lower += [-fitting.offset_bound_w_per_krpm, *([-fitting.initial_state_bound_w] * order)] * len(data)
    upper += [fitting.offset_bound_w_per_krpm, *([fitting.initial_state_bound_w] * order)] * len(data)
    attempts = []
    best = None
    for start in ((.05,), (.5,)) if order == 1 else ((.03, 1.), (.15, 4.)):
        p0 = [*np.log(start), *([10.] * (2 * n_gain)), *([0.] * ((order + 1) * len(data)))]
        p0 = np.clip(p0, np.asarray(lower) + 1e-10, np.asarray(upper) - 1e-10)
        result = least_squares(residual, p0, bounds=(lower, upper), loss="soft_l1",
                               f_scale=fitting.robust_scale_w, max_nfev=fitting.max_nfev, x_scale="jac")
        attempts.append(dict(success=bool(result.success), cost=float(result.cost), nfev=result.nfev,
                             message=result.message, active_bounds=result.active_mask.tolist()))
        if best is None or result.cost < best.cost:
            best = result
    taus, gu, gd, runs = unpack(best.x)
    permutation = np.argsort(taus)
    model = GreyBoxModel(order, drive, taus[permutation].tolist(),
                         np.r_[gu[0], gu[1:][permutation]].tolist(), np.r_[gd[0], gd[1:][permutation]].tolist(),
                         asdict(base), asdict(fitting))
    model.version = version_of(model.payload())
    model.report = dict(method="joint_shared_dynamics_with_run_offset_and_initial_states", attempts=attempts,
        convergence=bool(best.success), sources=[r.source for r in records],
        fitted_run_states=[dict(source=r.source, offset_w_per_krpm=float(v[0]), initial_components_w=v[1:][permutation].tolist())
                           for r, v in zip(records, runs)],
        training_curve_rmse_w=[float(np.sqrt(np.mean((_simulate(*d[:4], base, taus, gu, gd, v[0], v[1:])[0] - d[4]) ** 2)))
                               for d, v in zip(data, runs)],
        excluded_events=[x[1] for x in extracted], validation_passed=False,
        units="power W; speed r/min; tau s; acceleration drive krpm/s; inertial drive krpm^2/s",
        identifiability="时间常数触界、状态增益接近零或相近时不能解释为独立物理过程；初始状态仅为运行条件")
    model.report["parameter_diagnostics"] = dict(
        tau_at_bound=bool(np.any(np.isclose(model.taus_s, fitting.tau_min_s, rtol=.01))
                          or np.any(np.isclose(model.taus_s, fitting.tau_max_s, rtol=.01))),
        nearly_equal_time_constants=bool(order == 2 and model.taus_s[1] / model.taus_s[0] < 1.5),
        inactive_state_up=[j for j, g in enumerate(model.gains_up[1:]) if g < 1e-4],
        inactive_state_down=[j for j, g in enumerate(model.gains_down[1:]) if g < 1e-4])
    model.validate()
    return model


def predict_greybox(time_s, speed_rpm, model, *, initial_components_w=None,
                    offset_w_per_krpm=0., state=None, source=None, cancel=None):
    """Offline measured-speed prediction; no power data enter this function.

    With a prefix state, input must start AT its endpoint (include that sample).
    This forces the caller to supply the intervening speed history rather than
    silently skipping an unknown motion. Smoothing uses speed only, offline.
    """
    checkpoint(cancel)
    model.validate()
    time, dt = _time(time_s)
    config = CalibrationConfig(**model.baseline["config"])
    raw = np.asarray(speed_rpm, float)
    speed, up, down = _inputs(time, raw, config, model.drive)
    if state is not None:
        state.validate(model)
        if (source != state.source or not np.isclose(time[0], state.end_s, atol=1e-9, rtol=0)
                or abs(raw[0] - state.speed_rpm) > 30 or initial_components_w is not None or offset_w_per_krpm != 0):
            raise ValueError("参考须来自同会话、预测起点须等于参考终点并提供完整后续转速；不可重复指定初态")
        initial_components_w, offset_w_per_krpm = state.components_w, state.offset_w_per_krpm
    initial = np.zeros(model.order) if initial_components_w is None else np.asarray(initial_components_w, float)
    if (initial.shape != (model.order,) or not np.isfinite(initial).all()
            or not np.isfinite(offset_w_per_krpm) or abs(offset_w_per_krpm) > model.fitting["offset_bound_w_per_krpm"] + 1e-8):
        raise ValueError("初始状态或偏移无效")
    base = BasePowerModel(**model.baseline)
    power, components, steady = _simulate(time, speed, up, down, base, model.taus_s, model.gains_up, model.gains_down,
                                         offset_w_per_krpm, initial)
    valid = (raw >= -30) & (raw <= max(base.speeds_rpm) + 30)
    minimum = min(s for s in base.speeds_rpm if s > 0)
    valid &= (np.abs(raw) <= 30) | (raw >= minimum - 30) | (up + down > 0)
    warnings = ["离线实测转速驱动；未验证任意变速、控制器延迟或实时因果滤波"]
    if initial_components_w is None:
        valid[time < time[0] + 5 * max(model.taus_s)] = False
        warnings.append("初始动态状态未知，前5倍最大时间常数屏蔽；其后仍属状态衰减近似")
    power[~valid] = np.nan
    return dict(time_s=time, speed_rpm=raw, power_w=power, steady_w=steady, valid=valid,
                components_w=components, model_version=model.version, status=model.status,
                initial_state_source="prior_idle_prefix" if state else "explicit" if initial_components_w is not None else "unknown",
                warnings=warnings, timing=TIMING)


def initialize_greybox_state(record, model, *, end_s, no_load_confirmed=False, cancel=None):
    """Estimate run conditions from ONLY an already-completed idle prefix."""
    checkpoint(cancel)
    model.validate()
    if no_load_confirmed is not True or not np.isfinite(end_s):
        raise ValueError("必须确认此前片段为空载，且指定有限的结束时间")
    if end_s < record.time_s[0] + 1 or end_s > record.time_s[-1]:
        raise ValueError("空载前缀至少1秒且必须位于记录内")
    stop = np.searchsorted(record.time_s, end_s, side="right")
    prefix = Measurement(record.source, record.time_s[:stop].copy(),
        {k: record.values[k][:stop].copy() for k in ("spindle_speed_rpm", "spindle_power_w")}, record.valid[:stop].copy())
    _check_record(prefix)
    cfg, fit = CalibrationConfig(**model.baseline["config"]), GreyBoxConfig(**model.fitting)
    t, dt = _time(prefix.time_s)
    s, up, down = _inputs(t, prefix.values["spindle_speed_rpm"], cfg, model.drive)
    if max(s) < min(x for x in model.baseline["speeds_rpm"] if x > 0) - 30:
        raise ValueError("空载前缀没有覆盖已标定正转速，无法辨识会话偏移")
    base = BasePowerModel(**model.baseline)
    # Anchor the baseline on an already stable window, so transient model error
    # cannot be absorbed into a spurious run offset. The shared dynamics stay fixed.
    from .base_reference import calibrate_idle_reference
    reference = calibrate_idle_reference(prefix, base, start_s=float(t[-1] - 1.), end_s=float(t[-1]),
                                         no_load_confirmed=True, cancel=cancel)
    offset = reference.offset_w_per_krpm
    if abs(offset) > fit.offset_bound_w_per_krpm:
        raise ValueError("此前空载平台偏移超过灰箱初态适用范围")
    nominal, components, _ = _simulate(t, s, up, down, base, model.taus_s, model.gains_up, model.gains_down)
    matrix = np.column_stack([np.exp(-(t - t[0]) / tau) for tau in model.taus_s])
    y = _filtered(prefix.values["spindle_power_w"], dt, cfg) - nominal - offset * s / 1000.
    stride = max(1, round(fit.fit_period_s / dt))
    a, b = matrix[::stride], y[::stride]
    bounds = np.full(model.order, fit.initial_state_bound_w)
    def residual(p):
        checkpoint(cancel)
        return a @ p - b
    result = least_squares(residual, np.zeros(model.order), jac=lambda p: a.copy(), bounds=(-bounds, bounds),
                           loss="soft_l1", f_scale=fit.robust_scale_w, max_nfev=fit.max_nfev)
    if not result.success:
        raise ValueError("空载前缀状态辨识未收敛")
    import hashlib
    digest = hashlib.sha256()
    for value in (t, prefix.values["spindle_speed_rpm"], prefix.values["spindle_power_w"]):
        digest.update(np.asarray(value, dtype="<f8").tobytes())
    state = GreyBoxState(model.version, record.source, float(t[-1]), float(prefix.values["spindle_speed_rpm"][-1]),
        float(offset), (components[:, -1] + matrix[-1] * result.x).tolist(), digest.hexdigest())
    state.validate(model)
    return state


def predict_greybox_event(target_speed_rpm, model, *, initial_speed_rpm=0., sample_period_s=.001, cancel=None):
    """Prospective query using TRAINING-only nominal speed profiles, settled start.

    No measured target-event speed or power is used. The spindle template and
    zero controller delay are additional assumptions, scored separately.
    """
    model.validate()
    checkpoint(cancel)
    if not np.isfinite(sample_period_s) or sample_period_s <= 0:
        raise ValueError("采样周期必须为有限正数")
    base = BasePowerModel(**model.baseline)
    profile = _interpolate_profile(base, initial_speed_rpm, target_speed_rpm)
    end = max(profile["phase_times_s"][-1], profile["motion_time_s"] + 7 * max(model.taus_s)) + 1.
    if not 3 <= end / sample_period_s <= 2_000_000:
        raise ValueError("输出采样点数须为3到200万")
    t = np.arange(0, end + sample_period_s / 2, sample_period_s)
    _, speed = _evaluate(t, profile)
    result = predict_greybox(t, speed, model, initial_components_w=np.zeros(model.order), cancel=cancel)
    power = result["power_w"]
    peak = int(np.argmax(power) if target_speed_rpm > initial_speed_rpm else np.argmin(power))
    steady = float(np.interp(target_speed_rpm, base.speeds_rpm, base.powers_w))
    settle = _settling(t, power, steady, t[peak], CalibrationConfig(**base.config))
    result["characteristics"] = dict(initial_speed_rpm=initial_speed_rpm, target_speed_rpm=target_speed_rpm,
        extremum_power_w=float(power[peak]), peak_time_s=float(t[peak]), steady_power_w=steady,
        settling_time_s=settle, peak_to_settle_s=settle - t[peak] if settle is not None else None)
    result["speed_source"] = "training_file_event_template; unmeasured_controller_delay_assumed_zero"
    result["warnings"].append("仅支持稳定0→S和S→0；当前查询使用零动态初态与总体平均基线")
    return result
