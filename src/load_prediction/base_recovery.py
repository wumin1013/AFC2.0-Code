"""Speed-scheduled recovery candidates, separate from the active event model.

The early event template is retained. Only the tail after the fast-recovery
anchor changes: either a time-warped template or a first-order exponential.
These are bounded, non-neural models; a quadratic schedule is NOT a second-
order dynamical system. Optional session adaptation sees a completed prefix.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit, logit

from .base_power import BasePowerModel, CalibrationConfig, _time, identify_events
from .base_events import (_fit, _filtered, _evaluate, _interpolate_profile,
                          _settling, extract_events, validate_profiles, TIMING)
from .base_reference import IdleReference, calibrate_idle_reference, correct_profile
from .measurement import Measurement
from .common import checkpoint, save_json, version_of


METHODS = ("phase_warp", "exponential")
PARAMETER_MIN_S, PARAMETER_MAX_S = .01, 30.
MIN_REFERENCE_TAIL_S = .3
SCALE_BOUNDS = (.5, 2.)


def _basis(speed, domain):
    x = 2 * (np.asarray(speed) - domain[0]) / (domain[1] - domain[0]) - 1
    return np.array([np.ones_like(x), x, x * x]).T


def _parameter(coefficients, speed, domain):
    return PARAMETER_MIN_S + (PARAMETER_MAX_S - PARAMETER_MIN_S) * expit(_basis(speed, domain) @ coefficients)


@dataclass
class RecoveryModel:
    method: str
    baseline: dict
    domains: dict
    coefficients: dict
    version: str = ""
    status: str = "research_candidate_unvalidated"
    report: dict = field(default_factory=dict)
    schema: str = "afc.recovery.1"

    def payload(self):
        return {k: v for k, v in asdict(self).items() if k not in ("version", "status", "report")}

    def validate(self):
        if self.schema != "afc.recovery.1" or self.method not in METHODS:
            raise ValueError("不支持的恢复候选结构")
        base = BasePowerModel(**self.baseline)
        CalibrationConfig(**base.config).validate()
        validate_profiles(base.event_profiles)
        if (base.schema != "afc.base.2" or len(base.speeds_rpm) != len(base.powers_w)
                or not np.isfinite(base.speeds_rpm + base.powers_w).all()
                or np.any(np.diff(base.speeds_rpm) <= 0)):
            raise ValueError("恢复候选的基础曲线无效")
        for kind in ("startup", "braking"):
            domain = np.asarray(self.domains[kind], float)
            coefficients = np.asarray(self.coefficients[kind], float)
            levels = [p["level_rpm"] for p in base.event_profiles if p["kind"] == kind]
            if (domain.shape != (2,) or not np.isfinite(domain).all() or domain[0] >= domain[1]
                    or coefficients.shape != (3,) or not np.isfinite(coefficients).all()
                    or np.max(np.abs(coefficients)) > 12.000001
                    or domain.tolist() != [min(levels), max(levels)]):
                raise ValueError("恢复参数或标定域无效")
        if self.version != version_of(self.payload()):
            raise ValueError("恢复模型版本与参数不匹配")

    def save(self, path):
        self.validate()
        save_json(path, self)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as stream:
            model = cls(**json.load(stream))
        model.validate()
        return model


def _tail_arrays(record, event, observation, config):
    use = (record.time_s >= observation["onset_s"]) & (record.time_s <= event["end_s"])
    t = record.time_s[use] - observation["onset_s"]
    y = _filtered(record.values["spindle_power_w"][use], t[1] - t[0], config)
    fast = observation["phase_times_s"][2]
    stride = max(1, int(round(.02 / (t[1] - t[0]))))
    indices = np.flatnonzero(t >= fast)[::stride]
    return t[indices] - fast, y[indices] - observation["steady_w"]


def fit_recovery(records, *, method="phase_warp", config=None, events=None, cancel=None, exclude_level=None):
    """Joint three-coefficient schedule per direction, equally weighted events.

    exclude_level removes that level from profiles, steady knots AND fitting
    targets before any fitting. It supports crossed file/level validation.
    """
    checkpoint(cancel)
    config = config or CalibrationConfig()
    config.validate()
    if method not in METHODS or not records or len({r.source for r in records}) != len(records):
        raise ValueError("恢复拟合需要独立记录与支持的方法")
    events = events if events is not None else [identify_events(r, config) for r in records]
    if len(events) != len(records):
        raise ValueError("事件与记录数量不匹配")
    extracted = [extract_events(r, e, config, cancel) for r, e in zip(records, events)]
    if exclude_level is not None:
        extracted = [([o for o in obs if o["level_rpm"] != exclude_level], skipped,
                      {s: p for s, p in levels.items() if s != exclude_level}) for obs, skipped, levels in extracted]
    base = _fit(extracted, config)
    coefficients, domains, fits = {}, {}, {}
    for kind in ("startup", "braking"):
        samples = []
        for record, event_list, (observations, _, _) in zip(records, events, extracted):
            for o in observations:
                if o["kind"] != kind:
                    continue
                x, y = _tail_arrays(record, event_list[o["index"]], o, config)
                samples.append((o, x, y))
        levels = sorted({o["level_rpm"] for o, _, _ in samples})
        if len(levels) < 3:
            raise ValueError("平滑恢复标定每个方向至少需要三个转速档位")
        domain = [levels[0], levels[-1]]
        domains[kind] = domain
        def residual(beta):
            checkpoint(cancel)
            errors = []
            for o, x, y in samples:
                parameter = _parameter(beta, o["level_rpm"], domain)
                if method == "phase_warp":
                    duration = o["settling_time_s"] - o["phase_times_s"][2]
                    errors.append(np.array([np.log(parameter / duration)]))
                else:
                    amplitude = o["power_w"][32] - o["steady_w"]
                    errors.append((amplitude * np.exp(-x / parameter) - y) / max(abs(amplitude), 5.) / np.sqrt(len(y)))
            # Fixed mild curvature penalty, not tuned on held-out scores.
            return np.r_[np.concatenate(errors), .03 * beta[2]]
        initial = [float(logit((1. - PARAMETER_MIN_S) / (PARAMETER_MAX_S - PARAMETER_MIN_S))), 0., 0.]
        solution = least_squares(residual, initial, bounds=(-12., 12.), loss="soft_l1", f_scale=.2, max_nfev=150)
        coefficients[kind] = solution.x.tolist()
        fits[kind] = dict(converged=bool(solution.success), evaluations=solution.nfev, cost=float(solution.cost),
                          coefficients_at_bound=bool(np.any(np.abs(solution.x) > 11.99)), events=len(samples))
    model = RecoveryModel(method, asdict(base), domains, coefficients)
    model.version = version_of(model.payload())
    model.report = dict(sources=[r.source for r in records], fits=fits, excluded_level_rpm=exclude_level,
                        convergence=all(f["converged"] for f in fits.values()),
                        speed_levels={k: sorted({o["level_rpm"] for obs, _, _ in extracted for o in obs if o["kind"] == k}) for k in domains},
                        parameter="fast_to_band_duration_s" if method == "phase_warp" else "tail_tau_s",
                        method="bounded quadratic speed schedule; shared across training events/files")
    model.validate()
    return model


@dataclass(frozen=True)
class RecoveryReference:
    model_version: str
    source: str
    end_s: float
    idle_reference: dict
    scales: dict
    accepted_events: list
    rejected_events: list
    prefix_sha256: str
    no_load_confirmed: bool = True

    def validate(self, model):
        if (self.model_version != model.version or self.no_load_confirmed is not True
                or not np.isfinite(self.end_s) or not self.source
                or set(self.scales) != {"startup", "braking"}
                or any(not np.isfinite(v) or not SCALE_BOUNDS[0] <= v <= SCALE_BOUNDS[1] for v in self.scales.values())
                or any(e["completed_s"] > self.end_s for e in self.accepted_events)):
            raise ValueError("恢复参考状态无效或版本不匹配")
        ref = IdleReference(**self.idle_reference)
        ref.validate(BasePowerModel(**model.baseline))
        if ref.source != self.source or ref.end_s > self.end_s:
            raise ValueError("空载参考会话或完成时间不匹配")


def initialize_recovery_reference(record, model, *, end_s, no_load_confirmed=False, cancel=None):
    """Only a completed, confirmed idle prefix; no full-record preprocessing.

    Up to the last three informative events per direction estimate a median
    time multiplier. Short tails (<0.3 s) cannot adapt the recovery schedule.
    An unsuitable event is explicitly rejected, never silently clipped.
    """
    checkpoint(cancel)
    model.validate()
    if no_load_confirmed is not True or not np.isfinite(end_s):
        raise ValueError("必须确认此前整个参考前缀为空载，并指定已完成时刻")
    if end_s <= record.time_s[0] or end_s > record.time_s[-1]:
        raise ValueError("参考前缀结束时刻超出记录")
    # Slice first: later invalid samples and later power must have no effect.
    use = record.time_s <= end_s + 1e-10
    prefix = Measurement(record.source, record.time_s[use].copy(),
                         {k: v[use].copy() for k, v in record.values.items()}, record.valid[use].copy())
    _time(prefix.time_s)
    base = BasePowerModel(**model.baseline)
    config = CalibrationConfig(**base.config)
    event_list = identify_events(prefix, config)
    observations, excluded, _ = extract_events(prefix, event_list, config, cancel)
    # Stable power reference from the latest completed positive plateau.
    windows = [e for e in reversed(event_list) if e["speed_rpm"] > 0 and e["end_s"] - e["start_s"] >= 1.]
    if not windows:
        raise ValueError("此前前缀缺少完整正转速空载平台")
    accepted, rejected = [], list(excluded)
    idle = None
    for window in windows:
        # identify_events trims both ends of a plateau. At an explicitly
        # supplied completed prefix boundary do not trim the end a second time.
        endpoint = window["end_s"]
        if window is event_list[-1] and abs(prefix.values["spindle_speed_rpm"][-1] - window["speed_rpm"]) <= 30:
            endpoint = float(prefix.time_s[-1])
        try:
            idle = calibrate_idle_reference(prefix, base, start_s=endpoint - 1., end_s=endpoint,
                                            no_load_confirmed=True, cancel=cancel)
            break
        except ValueError as exc:
            rejected.append(dict(kind="steady_reference", completed_s=endpoint, reason=str(exc)))
    if idle is None:
        raise ValueError("此前前缀没有通过检查的稳态空载参考窗口")
    for o in observations:
        checkpoint(cancel)
        kind, level = o["kind"], o["level_rpm"]
        duration = o["settling_time_s"] - o["phase_times_s"][2]
        row = dict(kind=kind, level_rpm=level, completed_s=event_list[o["index"]]["end_s"])
        domain = model.domains[kind]
        if not domain[0] <= level <= domain[1] or duration < MIN_REFERENCE_TAIL_S:
            rejected.append({**row, "reason": "转速超出标定域或慢恢复尾段不足0.3秒"})
            continue
        parameter = _parameter(model.coefficients[kind], level, domain)
        if model.method == "phase_warp":
            ratio = duration / parameter
        else:
            x, y = _tail_arrays(prefix, event_list[o["index"]], o, config)
            amplitude = o["power_w"][32] - o["steady_w"]
            def residual(log_scale):
                checkpoint(cancel)
                return (amplitude * np.exp(-x / (parameter * np.exp(log_scale[0]))) - y) / max(abs(amplitude), 5.)
            solution = least_squares(residual, [0.], bounds=(np.log(.25), np.log(4.)), loss="soft_l1", f_scale=.1)
            if not solution.success:
                rejected.append({**row, "reason": "参考时间尺度辨识未收敛"})
                continue
            ratio = float(np.exp(solution.x[0]))
        if not SCALE_BOUNDS[0] <= ratio <= SCALE_BOUNDS[1]:
            rejected.append({**row, "reason": "参考时间尺度超出0.5～2倍适用范围", "ratio": float(ratio)})
            continue
        accepted.append({**row, "ratio": float(ratio)})
    scales = {}
    for kind in ("startup", "braking"):
        ratios = [e["ratio"] for e in accepted if e["kind"] == kind][-3:]
        scales[kind] = float(np.exp(np.median(np.log(ratios)))) if ratios else 1.
    digest = hashlib.sha256()
    for value in (prefix.time_s, prefix.values["spindle_speed_rpm"], prefix.values["spindle_power_w"], prefix.valid):
        digest.update(np.asarray(value, dtype="<f8").tobytes())
    reference = RecoveryReference(model.version, record.source, float(prefix.time_s[-1]), asdict(idle),
                                  scales, accepted, rejected, digest.hexdigest())
    reference.validate(model)
    return reference


def predict_recovery_event(target_speed_rpm, model, *, initial_speed_rpm=0., sample_period_s=.001,
                           reference=None, source=None, onset_s=None, cancel=None):
    checkpoint(cancel)
    model.validate()
    if not np.isfinite(sample_period_s) or sample_period_s <= 0:
        raise ValueError("输出采样周期必须为有限正数")
    base = BasePowerModel(**model.baseline)
    config = CalibrationConfig(**base.config)
    profile = _interpolate_profile(base, float(initial_speed_rpm), float(target_speed_rpm))
    scale = 1.
    if reference is not None:
        reference.validate(model)
        if (source != reference.source or onset_s is None or not np.isfinite(onset_s) or onset_s <= reference.end_s):
            raise ValueError("恢复参考只能用于同会话中参考前缀结束之后的事件")
        correct_profile(profile, base, IdleReference(**reference.idle_reference))
        scale = reference.scales[profile["kind"]]
    kind, level = profile["kind"], profile["level_rpm"]
    parameter = float(_parameter(model.coefficients[kind], level, model.domains[kind]) * scale)
    fast = profile["phase_times_s"][2]
    steady, amplitude = profile["steady_w"], profile["power_w"][32] - profile["steady_w"]
    tolerance = max(config.settling_absolute_band_w, config.settling_relative_band * abs(steady))
    if model.method == "phase_warp":
        original = np.array(profile["phase_times_s"], float)
        duration = original[3] - fast
        profile["phase_times_s"][3] = fast + parameter
        profile["phase_times_s"][4] = fast + parameter + (original[4] - original[3]) * parameter / duration
        expected_settle = fast + parameter
        end = max(profile["phase_times_s"][-1], expected_settle) + config.settling_hold_s
    else:
        expected_settle = fast + parameter * max(0., np.log(max(abs(amplitude) / tolerance, 1.)))
        end = max(expected_settle + config.settling_hold_s, fast + 8 * parameter)
    if not 2 <= end / sample_period_s <= 2_000_000:
        raise ValueError("单事件输出需要3～200万点，请调整采样周期")
    t = np.arange(0., end + sample_period_s / 2, sample_period_s)
    power, _ = _evaluate(t, profile)
    # Speed retains the original template; power relaxation is not speed motion.
    original_profile = _interpolate_profile(base, float(initial_speed_rpm), float(target_speed_rpm))
    _, speed = _evaluate(t, original_profile)
    if model.method == "exponential":
        tail = t > fast
        power[tail] = steady + amplitude * np.exp(-(t[tail] - fast) / parameter)
    peak = int(np.argmax(power) if kind == "startup" else np.argmin(power))
    settle = _settling(t, power, steady, t[peak], config)
    features = dict(kind=kind, initial_speed_rpm=float(initial_speed_rpm), target_speed_rpm=float(target_speed_rpm),
                    steady_power_w=steady, extremum_power_w=float(power[peak]), peak_time_s=float(t[peak]),
                    settling_time_s=settle, peak_to_settle_s=settle - float(t[peak]) if settle is not None else None)
    features["peak_power_w" if kind == "startup" else "trough_power_w"] = float(power[peak])
    return dict(time_s=t, power_w=power, speed_rpm=speed, characteristics=features, method=model.method,
                status="research_candidate_unvalidated", model_version=model.version, timing=TIMING,
                recovery_parameter_s=parameter, time_multiplier=scale, reference=asdict(reference) if reference else None,
                warnings=["仅支持已稳定0→S或S→0；中途切换及范围外不支持", "时间相对响应起点，控制器延迟未标定",
                          "此前空载时间尺度不等于温度辨识；研究候选未自动替换基础模型"])
