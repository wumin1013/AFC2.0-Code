"""Qualify optional base-power addition without changing base identification.

Evidence certifies either a settled pre-cutting estimate or the supplied
command timeline. Research/event-only validation never qualifies either mode.
The reported interval retains cutting-component uncertainty only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .base_power import BasePowerModel, predict_base_from_commands
from .common import file_hash

POSITION_KEYS = ["line", "execution_instance", "segment_mm"]


def _bound_file(path, expected, label):
    if not path or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label}缺少文件或SHA-256")
    path = Path(path)
    if not path.is_file() or file_hash(path) != expected:
        raise ValueError(f"{label}文件不存在或SHA-256不匹配")
    return path


def _domain_speeds(values, domain, model):
    speeds = np.asarray(values, dtype=float)
    low, high = domain["spindle_rpm_min"], domain["spindle_rpm_max"]
    if (not speeds.size or not np.isfinite(speeds).all() or (speeds < 0).any()
            or (speeds < max(low, model.speeds_rpm[0])).any()
            or (speeds > min(high, model.speeds_rpm[-1])).any()):
        raise ValueError("主轴转速超出已验证基础功率适用域，禁止外推")
    return speeds


def _evidence(request, known):
    evidence = request.get("base_validation") or {}
    if not isinstance(evidence, dict):
        raise ValueError("基础功率凭据必须为结构化对象")
    mode = evidence.get("mode")
    if mode not in ("steady_precutting", "commanded_dynamic"):
        raise ValueError("未提供稳态加工前或完整指令动态模式的基础功率凭据")
    if evidence.get("validated") is not True or evidence.get("applicable") is not True:
        raise ValueError("基础功率凭据未明确通过验证及适用性审核")
    model_path = _bound_file(evidence.get("model_path"), evidence.get("model_sha256"), "凭据模型")
    requested = _bound_file(request.get("base_model_path"), evidence.get("model_sha256"), "请求模型")
    report_path = _bound_file(evidence.get("report_path"), evidence.get("report_sha256"), "验证报告")
    model = BasePowerModel.load(requested)
    if not evidence.get("version") or model.version != evidence["version"]:
        raise ValueError("基础功率模型版本与凭据不一致")
    allowed = {"validated_in_sample_domain"}
    if mode == "steady_precutting":
        allowed.add("validated_steady_in_sample_domain")
    if model.status not in allowed:
        raise ValueError("基础功率模型仍未验证，研究候选不能放行总功率")
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    if not isinstance(report, dict):
        raise ValueError("基础功率验证报告必须为结构化对象")
    for key, expected in (("model_sha256", evidence["model_sha256"]), ("model_version", model.version),
                          ("mode", mode), ("domain", evidence.get("domain")), ("status", "validated")):
        if report.get(key) != expected:
            raise ValueError(f"基础功率验证报告的{key}与凭据不匹配")
    flag = "eligible_for_steady_prediction" if mode == "steady_precutting" else "eligible_for_dynamic_prediction"
    if report.get(flag) is not True:
        raise ValueError("报告尚未放行对应总功率预测模式")
    domain = evidence.get("domain")
    if not isinstance(domain, dict):
        raise ValueError("基础功率适用域缺失")
    for key in ("machine_id", "power_channel", "machine_settings_version"):
        if not domain.get(key) or known.get(key) != domain[key]:
            raise ValueError(f"当前{key}与已验证基础功率适用域不一致")
    if domain["power_channel"] != "spindle_power_w":
        raise ValueError("总功率仅支持同一主轴功率通道")
    bounds = [domain.get("spindle_rpm_min"), domain.get("spindle_rpm_max")]
    if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in bounds)
            or bounds[0] < 0 or bounds[0] > bounds[1]):
        raise ValueError("基础功率适用转速区间无效")
    return model, domain, mode, dict(model_path=str(model_path.resolve()),
        model_sha256=evidence["model_sha256"], report_path=str(report_path.resolve()),
        report_sha256=evidence["report_sha256"], model_version=model.version, mode=mode, domain=dict(domain))


def _dynamic_base(frame, request, domain, model):
    if request.get("complete_nc_timeline_verified") is not True:
        raise ValueError("完整NC指令和等待时间线尚未核验")
    if request.get("initial_state_verified") is not True:
        raise ValueError("初始主轴状态未知，不能生成动态总功率")
    initial = request.get("initial_speed_rpm")
    if isinstance(initial, bool) or not isinstance(initial, (int, float)) or not np.isfinite(initial):
        raise ValueError("动态总功率需要显式有限初始转速")
    _domain_speeds([initial], domain, model)
    timeline_path = _bound_file(request.get("command_timeline"), request.get("command_timeline_sha256"), "完整指令时间线")
    mapping_path = _bound_file(request.get("time_mapping"), request.get("time_mapping_sha256"), "位置时间映射")
    timeline, mapping = pd.read_csv(timeline_path), pd.read_csv(mapping_path)
    if not {"time_s", "commands_rpm"} <= set(timeline):
        raise ValueError("完整指令时间线需要time_s和commands_rpm")
    time = timeline.time_s.to_numpy(float)
    if time.ndim != 1 or len(time) < 3 or not np.isfinite(time).all() or (np.diff(time) <= 0).any():
        raise ValueError("指令时间线至少3点且有限、严格递增")
    steps = np.diff(time)
    if not np.allclose(steps, np.median(steps), rtol=.001, atol=1e-9):
        raise ValueError("完整指令时间线必须均匀采样，不能由空间网格假定时间")
    commands = _domain_speeds(timeline.commands_rpm.to_numpy(float), domain, model)
    if not set(POSITION_KEYS + ["time_s"]) <= set(mapping) or not set(POSITION_KEYS) <= set(frame):
        raise ValueError("位置时间映射缺少line/execution_instance/segment_mm/time_s")
    if (frame.duplicated(POSITION_KEYS).any() or mapping.duplicated(POSITION_KEYS).any()
            or not np.isfinite(mapping[POSITION_KEYS + ["time_s"]].to_numpy(float)).all()):
        raise ValueError("位置时间映射存在重复或无效位置")
    joined = frame[POSITION_KEYS].merge(mapping[POSITION_KEYS + ["time_s"]],
                                      on=POSITION_KEYS, how="left", validate="one_to_one", sort=False)
    position_time = joined.time_s.to_numpy(float)
    if (not np.isfinite(position_time).all() or (position_time < time[0]).any()
            or (position_time > time[-1]).any() or (np.diff(position_time) < 0).any()):
        raise ValueError("位置时间映射缺失、逆序或超出完整时间线")
    command_at_position = commands[np.searchsorted(time, position_time, side="right") - 1]
    if not np.allclose(command_at_position, frame.spindle_rpm.to_numpy(float), rtol=0, atol=1e-6):
        raise ValueError("位置指令转速与独立NC时间线不一致")
    prediction = predict_base_from_commands(time, commands, model, initial_speed_rpm=float(initial), source="nc_aligned")
    valid, power = np.asarray(prediction["valid"], dtype=bool), np.asarray(prediction["power_w"], dtype=float)
    if valid.shape != time.shape or power.shape != time.shape or not valid.all() or not np.isfinite(power).all():
        raise ValueError("完整动态时间线含模型不支持的过渡或未知状态")
    _domain_speeds(prediction["speed_rpm"], domain, model)
    return np.interp(position_time, time, power), dict(
        command_timeline_sha256=request["command_timeline_sha256"], time_mapping_sha256=request["time_mapping_sha256"],
        initial_speed_rpm=float(initial), initial_state="verified_settled", timing=prediction.get("timing"),
        warnings=list(prediction.get("warnings", [])))


def add_total_power(result, frame, request, known):
    """Return a new result; unavailable evidence preserves cutting predictions.

``initial_state_verified`` certifies the settled initial state expected by the
existing command-response model. Dynamic time comes only from hash-bound CSVs.
No base-model uncertainty is inferred from the cutting-power quantiles.
"""
    output = {key: value for key, value in result.items()
              if not key.startswith("total_") and key not in ("base_power_w", "base_model_version", "base_prediction_mode")}
    output.update(total_power_status="total_power_unavailable", total_power_reason="缺少总功率预测资格证据")
    try:
        if frame.empty or "spindle_rpm" not in frame:
            raise ValueError("总功率预测缺少位置及主轴指令转速")
        model, domain, mode, provenance = _evidence(request, known)
        speeds = _domain_speeds(frame.spindle_rpm.to_numpy(float), domain, model)
        if mode == "steady_precutting":
            if known.get("spindle_stable_before_cut") is not True or not np.allclose(speeds, speeds[0], rtol=0, atol=1e-6):
                raise ValueError("稳态总功率要求切削前已稳定且当前预测段主轴转速恒定")
            base = np.interp(speeds, model.speeds_rpm, model.powers_w)
            status = "steady_total_power_estimate"
        else:
            base, timing = _dynamic_base(frame, request, domain, model)
            provenance.update(timing)
            status = "commanded_dynamic_total_power_prediction"
        quantiles = np.asarray(result["cutting_quantiles_w"], dtype=float)
        if quantiles.shape != (len(frame), 3) or not np.isfinite(quantiles).all() or (np.diff(quantiles, axis=1) < 0).any():
            raise ValueError("总功率叠加需要等长有序切削功率三分位")
        if "calibrated_lower_w" in result or "calibrated_upper_w" in result:
            lower, upper = result["calibrated_lower_w"], result["calibrated_upper_w"]
        else:
            lower, upper = result.get("lower_w", quantiles[:, 0]), result.get("upper_w", quantiles[:, 2])
        lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        if lower.shape != base.shape or upper.shape != base.shape or not np.isfinite([lower, upper]).all() or (lower > upper).any():
            raise ValueError("切削功率有效区间无效")
        total = base[:, None] + quantiles
        output.update(total_power_status=status, total_power_reason=None, base_power_w=base,
            base_model_version=model.version, base_prediction_mode=mode, total_power_w=total[:, 1],
            total_quantiles_w=total, total_q05_w=total[:, 0], total_q50_w=total[:, 1], total_q95_w=total[:, 2],
            total_lower_w=base + lower, total_upper_w=base + upper,
            total_uncertainty_scope="cutting_component_only_base_estimate_added",
            total_power_provenance=provenance)
    except (ValueError, TypeError, KeyError, OSError, pd.errors.ParserError) as exc:
        output["total_power_reason"] = str(exc)
    return output
