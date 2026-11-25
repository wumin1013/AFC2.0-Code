"""Measured-speed prediction assisted by a separately audited NC event timeline.

This interface accepts no power channel. NC labels retain their exact values;
response onset and NC dwell entry are different observations, never controller
command timestamps. An engineering replay does not upgrade model qualification.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json

import numpy as np

from .base_power import CalibrationConfig, _time, speed_features
from .common import checkpoint


AUDITED_INTERFACE_VERSION = "afc.base.measured_audited.1"
SPEED_ONLY_INTERFACE_VERSION = "afc.base.measured_speed_only.1"
AUDITED_TIMELINE_SCHEMA = "afc.base.audited_timeline.1"


def speed_timeline_fingerprint(time_s, speed_rpm):
    """Bind observations without reading or hashing the target power channel."""
    time, _ = _time(time_s)
    speed = np.asarray(speed_rpm, float)
    if speed.shape != time.shape or not np.isfinite(speed).all():
        raise ValueError("转速必须与时间等长且全部有限；不可删除缺失点后压缩时间")
    digest = hashlib.sha256()
    for values in (time, speed):
        digest.update(np.asarray(values, dtype="<f8").tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class AuditedEventTimeline:
    events: list[dict]
    program_version: str
    record_source: str
    time_speed_sha256: str
    initial_speed_rpm: float | None = None
    initial_state_verified: bool = False
    record_integrity_verified: bool = False
    event_alignment_verified: bool = False
    timing_source: str = "observed_response_onset"
    plateau_timing_source: str = "nc_dwell_entry_before_trim"
    schema: str = AUDITED_TIMELINE_SCHEMA
    provenance: dict = field(default_factory=dict)

    def fingerprint(self):
        serialized = json.dumps(asdict(self), ensure_ascii=False, sort_keys=True,
                                allow_nan=False, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _speed_band(level):
    return max(30., .01 * abs(level))


def _timeline_events(timeline, time, raw, config):
    if not isinstance(timeline, AuditedEventTimeline) or timeline.schema != AUDITED_TIMELINE_SCHEMA:
        raise ValueError("需要明确版本的 AuditedEventTimeline，不能将旧自动识别静默当作审核接口")
    if not timeline.program_version or not timeline.record_source:
        raise ValueError("审核时间线需绑定程序版本和实测记录来源")
    if (timeline.initial_state_verified is not True or timeline.record_integrity_verified is not True
            or timeline.event_alignment_verified is not True):
        raise ValueError("初态、记录完整性和事件对齐必须明确通过审核；未知状态不能自动放行")
    initial = timeline.initial_speed_rpm
    if initial is None or not np.isfinite(initial) or initial < 0:
        raise ValueError("审核接口需要已确认的有限非负初始转速")
    if (timeline.timing_source != "observed_response_onset"
            or timeline.plateau_timing_source != "nc_dwell_entry_before_trim"):
        raise ValueError("时间线必须区分观测响应起点和NC驻留入口，不能混用速度平台或控制器指令时刻")
    if speed_timeline_fingerprint(time, raw) != timeline.time_speed_sha256:
        raise ValueError("审核时间线与当前时间/转速指纹不匹配")
    events = deepcopy(timeline.events)
    if not events:
        raise ValueError("审核事件清单为空，不能构造覆盖率")
    previous_end, origin, transitions = -np.inf, float(initial), []
    for index, event in enumerate(events):
        try:
            start, end, onset, target = [float(event[key]) for key in
                                         ("start_s", "end_s", "command_time_s", "speed_rpm")]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("审核事件需有 start_s/end_s/command_time_s/speed_rpm") from exc
        if (not np.isfinite([start, end, onset, target]).all() or target < 0
                or onset < time[0] or end > time[-1] or onset > start or end < start
                or onset < previous_end):
            raise ValueError("审核事件必须按时间排序、不重叠且位于记录范围内")
        if event.get("source") not in ("nc_aligned", "measured") or not event.get("timing_source"):
            raise ValueError("事件需要已审核的来源和响应时间提取依据")
        if event.get("ambiguity") is not None:
            raise ValueError("事件存在未解决的对齐歧义")
        previous_end = end
        if target == origin:
            if event.get("direction") is not None or transitions:
                raise ValueError("重复同档事件不能伪装成已观察的启动/制动")
            continue  # Optional initial platform; it is not a transition.
        event.update(index=index, start_s=start, end_s=end, command_time_s=onset,
                     speed_rpm=target, initial_speed_rpm=origin)
        transitions.append(event)
        origin = target
    if not transitions:
        raise ValueError("没有审核过的启动/制动事件，不能使用动态接口")
    prefix = time < transitions[0]["command_time_s"]
    dt = time[1] - time[0]
    if (prefix.sum() * dt < config.minimum_plateau_s
            or np.max(np.abs(raw[prefix] - initial)) > _speed_band(initial)):
        raise ValueError("首事件前未观察到足够的已确认稳定初态；可能缺少启动事件")
    return transitions


def predict_audited_measured(time_s, speed_rpm, model, timeline, *, record_source=None, cancel=None,
                             qualification=None, record_context=None):
    """Predict on every original time point, keeping unsupported intervals NaN.

    Timeline integrity/alignment approvals are necessary provenance, not a claim
    of independent interface validation. Power is never an input to this call.
    The original model, acceptance thresholds and speed-only path are retained.
    """
    from .base_events import TIMING, _evaluate, _in_steady_domain, _interpolate_profile, _result

    checkpoint(cancel)
    if model.schema != "afc.base.2":
        raise ValueError("审核事件接口需要 afc.base.2 事件模型；旧速率模型不兼容")
    time, dt = _time(time_s)
    raw = np.asarray(speed_rpm, float)
    config = CalibrationConfig(**model.config)
    config.validate()
    _, derivative = speed_features(time, raw, config)
    events = _timeline_events(timeline, time, raw, config)
    if record_source is not None and record_source != timeline.record_source:
        raise ValueError("审核时间线绑定的实测记录与当前记录来源不同")
    if not _in_steady_domain(np.asarray([timeline.initial_speed_rpm]), model)[0]:
        raise ValueError("初始转速不在模型已标定范围内")
    power = np.full(len(time), np.nan)
    valid = np.zeros(len(time), bool)
    prefix = time < events[0]["command_time_s"]
    power[prefix] = float(np.interp(timeline.initial_speed_rpm, model.speeds_rpm, model.powers_w))
    valid[prefix] = True
    ready, details, warnings = float(time[0]), [], []
    for index, event in enumerate(events):
        checkpoint(cancel)
        onset, target, origin = event["command_time_s"], event["speed_rpm"], event["initial_speed_rpm"]
        end = events[index + 1]["command_time_s"] if index + 1 < len(events) else time[-1] + dt
        use = (time >= onset) & (time < end)
        item = dict(index=event["index"], onset_s=onset, end_s=end,
                    initial_speed_rpm=origin, target_speed_rpm=target, valid=False,
                    source=event["source"], timing_source="observed_response_onset",
                    timing_evidence=event["timing_source"],
                    plateau_timing_source=timeline.plateau_timing_source)
        for key in ("segment", "physical_line", "raw_program_line", "response_start_index"):
            if key in event:
                item[key] = event[key]
        try:
            profile = _interpolate_profile(model, origin, target)
            if event.get("direction", profile["kind"]) != profile["kind"]:
                raise ValueError("事件方向与精确初始/目标转速不一致")
            item["kind"] = profile["kind"]
            if onset < ready - dt / 2:
                raise ValueError("上次事件尚未恢复完成，不支持未稳定重启或直接拼接")
            # Exclude the speed-filter leading edge while retaining the same
            # 0.5 s prior-state observation used in the five-run audit.
            prior = (time >= onset - config.minimum_plateau_s) & (time < onset - config.smooth_s / 2)
            if (prior.sum() * dt < config.minimum_plateau_s - config.smooth_s / 2 - dt
                    or np.max(abs(raw[prior] - origin)) > _speed_band(origin)):
                raise ValueError("事件前实测转速未确认稳定，或遗漏了先前事件")
            platform = (time >= event["start_s"]) & (time <= event["end_s"])
            if platform.sum() * dt < config.minimum_plateau_s:
                raise ValueError("目标平台观测不足，恢复状态右删失")
            if np.max(abs(raw[platform] - target)) > _speed_band(target):
                raise ValueError("审核目标平台与精确NC转速标签不一致")
            if np.max(abs(derivative[platform])) >= config.derivative_threshold_rpm_s:
                raise ValueError("审核目标平台仍有变速，不能确认稳定目标状态")
            # Both fit and prediction now mean onset -> NC dwell entry. A
            # speed-threshold platform must never be substituted for this time.
            motion = event["start_s"] - config.plateau_trim_s - onset
            if motion <= 0 or abs(motion / profile["motion_time_s"] - 1) > config.maximum_motion_relative_error:
                raise ValueError("审核NC驻留入口等待时长偏离同定义的标定范围")
            ready = onset + profile["phase_times_s"][-1]
            if event["end_s"] + dt < ready:
                raise ValueError("事件未完整观察到模型恢复及保持窗口，恢复右删失")
            power[use], _ = _evaluate(time[use] - onset, profile)
            valid[use] = True
            item.update(valid=True, motion_time_s=motion, steady_w=profile["steady_w"], ready_s=ready)
            # A missing final event or an unregistered mid-platform transition
            # is not repaired by carrying forward the prior steady prediction.
            # Smoothing then differentiating uses two centred windows. Exclude
            # their full combined leading support before a registered onset.
            tail_guard = config.smooth_s if index + 1 < len(events) else 0.
            tail = (time > event["end_s"]) & (time < end - tail_guard)
            unexpected = np.flatnonzero(tail & ((abs(raw - target) > _speed_band(target))
                                               | (abs(derivative) >= config.derivative_threshold_rpm_s)))
            if len(unexpected):
                valid[use & (time >= time[unexpected[0]])] = False
                item.update(valid=False, reason="审核平台后存在未登记的变速/末尾事件", unsupported_from_s=float(time[unexpected[0]]))
                ready = np.inf
                warnings.append(f"{time[unexpected[0]]:.3f}s：{item['reason']}")
        except ValueError as exc:
            valid[use] = False
            ready = np.inf  # Failed dynamics cannot establish the next initial state.
            item["reason"] = str(exc)
            warnings.append(f"{onset:.3f}s：{exc}")
        details.append(item)
    valid &= (raw >= -30) & (raw <= model.speeds_rpm[-1] + 30)
    qualification_check = dict(valid=False, issues=["未提供审核接口的正式标签资格凭据"])
    if qualification is not None:
        from .base_qualification import qualify_audited_prediction
        qualification_check = qualify_audited_prediction(qualification, model, timeline, record_context)
    released = bool(qualification_check["valid"] and valid.all())
    result = _result(time, power, raw.copy(), model, valid, warnings,
        interface_version=AUDITED_INTERFACE_VERSION, timing=TIMING,
        timing_source="observed_response_onset", events=details,
        initial_state_source="verified_settled_with_observed_prior", timeline_schema=timeline.schema,
        timeline_sha256=timeline.fingerprint(), time_speed_sha256=timeline.time_speed_sha256,
        program_version=timeline.program_version, record_source=timeline.record_source,
        plateau_timing_source=timeline.plateau_timing_source,
        coverage_fraction=float(valid.mean()), formal_training_release=released,
        qualification_report_sha256=qualification_check.get("report_sha256"),
        qualification_issues=qualification_check["issues"],
        interface_validation_status=("qualified_by_frozen_confirmation_and_equivalence" if released else
                                     "engineering_interface_requires_independent_validation"))
    if released:
        result["status"] = "validated_in_sample_domain"
        result["parent_model_status"] = model.status
    return result
