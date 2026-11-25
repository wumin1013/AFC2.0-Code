"""Optional session reference from an explicitly confirmed, completed idle window.

This is a one-parameter baseline correction, not temperature identification or
network training. Reference data are kept separate from the population model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .base_power import CalibrationConfig, _time, speed_features
from .common import checkpoint, save_json, version_of


@dataclass(frozen=True)
class IdleReference:
    model_version: str
    source: str
    start_s: float
    end_s: float
    speed_rpm: float
    power_w: float
    expected_power_w: float
    offset_w_per_krpm: float
    sample_period_s: float
    window_sha256: str
    no_load_confirmed: bool
    schema: str = "afc.idle_reference.1"
    session: dict | None = None

    def validate(self, model=None):
        if self.session is not None:
            validate_reference_session(self.session)
        numbers = [self.start_s, self.end_s, self.speed_rpm, self.power_w,
                   self.expected_power_w, self.offset_w_per_krpm, self.sample_period_s]
        if (self.schema != "afc.idle_reference.1" or self.no_load_confirmed is not True
                or not np.isfinite(numbers).all() or self.sample_period_s <= 0
                or self.end_s - self.start_s + self.sample_period_s < 1. - 1e-8
                or self.speed_rpm <= 0 or self.power_w <= 0 or self.expected_power_w <= 0):
            raise ValueError("空载参考参数无效，必须来自已确认且至少1秒的稳定空载窗口")
        expected = (self.power_w - self.expected_power_w) / (self.speed_rpm / 1000.)
        if (not np.isclose(self.offset_w_per_krpm, expected)
                or abs(self.power_w - self.expected_power_w) > .2 * max(50., self.expected_power_w)):
            raise ValueError("参考偏移无效或超过20%适用范围，需要重新检查工况")
        if model is not None:
            if model.schema != "afc.base.2" or self.model_version != model.version:
                raise ValueError("空载参考与基础模型版本不匹配，请重新提取参考")
            positive = [s for s in model.speeds_rpm if s > 0]
            if not min(positive) <= self.speed_rpm <= max(positive):
                raise ValueError("参考转速超出模型范围")
            predicted = np.interp(self.speed_rpm, model.speeds_rpm, model.powers_w)
            if not np.isclose(predicted, self.expected_power_w):
                raise ValueError("参考基线与模型不一致")

    def save(self, path):
        self.validate()
        save_json(path, self)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as stream:
            result = cls(**json.load(stream))
        result.validate()
        return result


def idle_platform_quality(time_s, power_w):
    """Existing platform drift/ripple rule, independent of a model's offset."""
    time, power = np.asarray(time_s, float), np.asarray(power_w, float)
    if (time.shape != power.shape or time.ndim != 1 or len(time) < 4
            or not np.isfinite(time).all() or not np.isfinite(power).all() or np.any(np.diff(time) <= 0)):
        raise ValueError("平台质量检查需要至少4个有序、有限采样点")
    measured = float(np.median(power))
    groups = np.array_split(power, 4)
    drift = abs(float(np.median(groups[-1]) - np.median(groups[0]))) / (.75 * (time[-1] - time[0]))
    ripple = float(np.quantile(np.abs(power - measured), .95))
    drift_limit, ripple_limit = max(1., .01 * abs(measured)), max(5., .05 * abs(measured))
    return dict(median_w=measured, drift_w_per_s=drift, ripple_p95_w=ripple,
                drift_limit_w_per_s=drift_limit, ripple_limit_w=ripple_limit,
                passed=drift <= drift_limit and ripple <= ripple_limit,
                version="afc.idle_platform_quality.1")


def calibrate_idle_reference(record, model, *, start_s, end_s, no_load_confirmed=False, session=None, cancel=None):
    """Inspect ONLY the requested window, never later samples in the record.

    The caller must ensure the window precedes the event being predicted and is
    from the same machine/session. A stationary cutting signal is NOT an idle
    reference; this cannot be determined from power and speed alone.
    """
    checkpoint(cancel)
    if no_load_confirmed is not True:
        raise ValueError("必须明确确认参考窗口为空载，不能用切削功率校正基础功率")
    if model.schema != "afc.base.2":
        raise ValueError("空载参考校正需要事件基础模型")
    if not np.isfinite([start_s, end_s]).all() or end_s - start_s < 1. - 1e-8:
        raise ValueError("参考窗口至少需要1秒")
    if start_s < record.time_s[0] or end_s > record.time_s[-1]:
        raise ValueError("参考窗口超出记录范围")
    record.require("spindle_speed_rpm", "spindle_power_w")
    use = (record.time_s >= start_s - 1e-10) & (record.time_s <= end_s + 1e-10)
    time, dt = _time(record.time_s[use])
    speed = record.values["spindle_speed_rpm"][use]
    power = record.values["spindle_power_w"][use]
    if not record.valid[use].all() or not np.isfinite(power).all():
        raise ValueError("参考窗口含缺失或异常数据")
    config = CalibrationConfig(**model.config)
    _, rate = speed_features(time, speed, config)
    nominal = round(float(np.median(speed)) / config.nominal_step_rpm) * config.nominal_step_rpm
    levels = [s for s in model.speeds_rpm if s > 0]
    if not min(levels) <= nominal <= max(levels):
        raise ValueError("参考转速不在正转速标定范围内")
    if np.max(np.abs(rate)) > config.derivative_threshold_rpm_s or np.ptp(speed) > max(30., .01 * nominal):
        raise ValueError("参考窗口主轴尚未稳速")
    measured = float(np.median(power))
    if not idle_platform_quality(time, power)["passed"]:
        raise ValueError("参考功率仍在恢复或波动过大，请选择更晚的稳定窗口")
    baseline = float(np.interp(nominal, model.speeds_rpm, model.powers_w))
    digest = hashlib.sha256()
    for values in (time, speed, power):
        digest.update(np.asarray(values, dtype="<f8").tobytes())
    reference = IdleReference(model.version, record.source, float(time[0]), float(time[-1]),
                              nominal, measured, baseline, (measured - baseline) / (nominal / 1000.),
                              dt, digest.hexdigest(), True,
                              session=validate_reference_session(session) if session is not None else None)
    reference.validate(model)
    return reference


def validate_reference_session(session):
    """User-declared identity and a shared clock; never infer time from file order."""
    if not isinstance(session, dict):
        raise ValueError("空载参考需补充机床、会话和文件时间原点")
    for key in ("machine_id", "session_id"):
        if not isinstance(session.get(key), str) or not session[key].strip():
            raise ValueError("空载参考需填写机床标识和连续运行会话标识")
    origin = session.get("time_origin_s")
    if isinstance(origin, bool) or not isinstance(origin, (int, float)) or not np.isfinite(origin) or origin < 0:
        raise ValueError("time_origin_s须为非负有限秒数：文件0秒在本会话时间轴上的时刻")
    return dict(machine_id=session["machine_id"].strip(), session_id=session["session_id"].strip(),
                time_origin_s=float(origin))


def same_reference_source(source, reference):
    return bool(source) and Path(source).resolve() == Path(reference.source).resolve()


def verify_reference_record(record, reference):
    """When reusing the source file, ensure the saved window still matches it."""
    if not same_reference_source(record.source, reference):
        return
    record.require("spindle_speed_rpm", "spindle_power_w")
    use = (record.time_s >= reference.start_s - 1e-10) & (record.time_s <= reference.end_s + 1e-10)
    digest = hashlib.sha256()
    for values in (record.time_s[use], record.values["spindle_speed_rpm"][use], record.values["spindle_power_w"][use]):
        digest.update(np.asarray(values, dtype="<f8").tobytes())
    if not record.valid[use].all() or digest.hexdigest() != reference.window_sha256:
        raise ValueError("参考来源文件的窗口已变化，请重新提取空载参考")


def reference_use_mask(time_s, model, reference, *, session, record_source):
    reference.validate(model)
    ref_session = validate_reference_session(reference.session)
    target = validate_reference_session(session)
    if any(ref_session[k] != target[k] for k in ("machine_id", "session_id")):
        raise ValueError("空载参考与目标机床或会话不匹配；重启或改变工况后应重新提取")
    if not isinstance(record_source, str) or not record_source.strip():
        raise ValueError("必须记录目标数据或程序来源")
    time = np.asarray(time_s, float)
    if time.ndim != 1 or not len(time) or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("参考应用时间必须有限且递增")
    same_source = same_reference_source(record_source, reference)
    if same_source and target["time_origin_s"] != ref_session["time_origin_s"]:
        raise ValueError("同一来源文件的会话时间原点必须一致")
    end = ref_session["time_origin_s"] + reference.end_s
    eligible = time + target["time_origin_s"] > end + 1e-9
    if not eligible.any() or (not same_source and not eligible.all()):
        raise ValueError("空载参考必须在目标数据或预测事件开始前完成，禁止读取未来参考")
    return eligible


def apply_idle_reference(result, model, reference, *, session, record_source):
    """Apply a causal, auditable b*S/1000 correction to a complete prediction."""
    if result.get("reference_adjustment") is not None:
        raise ValueError("基础功率已应用参考，不能重复校正")
    eligible = reference_use_mask(result["time_s"], model, reference,
                                  session=session, record_source=record_source)
    valid = np.asarray(result["valid"], bool) & eligible
    speed = np.asarray(result["speed_rpm"], float)
    delta = reference.offset_w_per_krpm * np.where(np.abs(speed) <= 30., 0., speed) / 1000.
    steady = np.asarray(result["steady_w"]) + delta
    if np.any(valid & (speed > 30) & (steady <= 0)):
        raise ValueError("参考校正导致非正稳态功率，超出适用范围")
    context = validate_reference_session(session)
    adjustment = dict(reference=asdict(reference), target_session=context, target_source=record_source,
                      session_source="user_declared", coefficient_units="W/(1000 r/min)",
                      timing_adjusted=False, status="reference_adjusted_candidate")
    version = version_of({"model": model.version, **adjustment})
    warnings = list(result.get("warnings", [])) + ["空载参考只校正功率，不证明恢复时间或真实控制器时序已验证"]
    if not eligible.all():
        warnings.append("参考窗口及此前样点不应用该参考，已屏蔽以防止未来信息进入标签")
    return {**result, "power_unadjusted_w": np.asarray(result["power_w"]).copy(),
            "power_w": np.where(valid, np.asarray(result["power_w"]) + delta, np.nan),
            "steady_w": np.where(valid, steady, np.nan), "reference_correction_w": np.where(valid, delta, np.nan),
            "reference_applied": valid, "valid": valid, "reference_adjustment": adjustment,
            "prediction_version": version, "parent_model_status": model.status,
            "status": "reference_adjusted_candidate", "warnings": warnings}


def correct_profile(profile, model, reference):
    """Apply delta P = b * S/1000; zero-speed power is unchanged."""
    reference.validate(model)
    beta = reference.offset_w_per_krpm
    delta = beta * np.asarray(profile["speed_rpm"]) / 1000.
    values = np.asarray(profile["power_w"]) + delta
    target = profile["level_rpm"] if profile["kind"] == "startup" else 0.
    steady = profile["steady_w"] + beta * target / 1000.
    if target and steady <= 0:
        raise ValueError("参考校正导致非正稳态功率，超出适用范围")
    profile["power_w"] = values.tolist()
    profile["steady_w"] = steady
    profile["extremum_w"] = float(values[16])
    # Timing is not adapted from a baseline measurement. Retain the independent
    # event timing estimate; the existing recovery envelope enforces it.
    from .base_events import _condition_recovery
    _condition_recovery(profile, CalibrationConfig(**model.config))
    return dict(reference=asdict(reference), coefficient_units="W/(1000 r/min)",
                prediction_version=version_of({"model": model.version, "reference": asdict(reference)}),
                status="reference_adjusted_candidate", timing_adjusted=False,
                applicability="同机床、同会话、参考窗口已完成之后；仅修正功率平台，不自动修正恢复时间",
                range_interpretation="observed_ranges仍是原始标定记录范围，不是参考校正后的条件区间")


def assess_repeatability(model):
    """Empirical lower bound for ANY speed-only single-valued time predictor.

    At an identical input, min_c max_i |c-y_i| = (max(y)-min(y))/2.
    This is a property of observed repeats, not a future confidence interval.
    """
    groups = {}
    for observations in model.report.get("event_observations", []):
        per_file = {}
        for event in observations:
            per_file.setdefault((event["kind"], event["level_rpm"]), []).append(event["settling_time_s"])
        for key, times in per_file.items():
            groups.setdefault(key, []).append(float(np.median(times)))
    tolerance = CalibrationConfig(**model.config).maximum_settling_time_error_s
    return [dict(kind=kind, speed_rpm=speed, independent_files=len(times),
                 observed_settling_range_s=[min(times), max(times)],
                 best_possible_worst_error_s=(max(times) - min(times)) / 2 if len(times) >= 2 else None,
                 exceeds_current_tolerance=(max(times) - min(times)) / 2 > tolerance + 1e-9 if len(times) >= 2 else None,
                 tolerance_s=tolerance, scope="仅对现有重复记录、仅转速输入的单值预测成立；加入状态输入后须重新验证")
            for (kind, speed), times in sorted(groups.items())]
