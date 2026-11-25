"""Record-local measured idle labels; never a prospective power predictor.

The report certifies explicit record exclusions and observed no-contact
platforms. It does not certify the frozen dynamic model or another record.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .common import file_hash, json_value

MODE = "record_idle_labels"
VERSION = "afc.record_idle_labels.1"
SCHEMA = "afc.record_idle.label_qualification.1"
MASK_VERSION = "afc.record_idle.constant_speed_mask.1"
POLICY_FIELDS = ("speed_tolerance_abs_rpm", "speed_tolerance_relative",
                 "max_abs_acceleration_rpm_s", "stable_hold_s", "minimum_platform_duration_s")
POLICY_VALUES = dict(zip(POLICY_FIELDS, (30., .01, 80., .5, 1.)))
LABEL_SOURCES = ("record_idle.py", "base_qualification.py", "base_reference.py", "samples.py")


def _digest(value):
    return hashlib.sha256(json.dumps(json_value(value), ensure_ascii=False, sort_keys=True,
                                    allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _ranges(value, *, single=True):
    if not isinstance(value, list) or not value or (single and len(value) != 1):
        raise ValueError("平台必须是单个连续原始采样索引半开区间，不能拼接离散样本")
    previous = -1
    for pair in value:
        if (not isinstance(pair, list) or len(pair) != 2 or any(type(v) is not int for v in pair)
                or pair[0] < 0 or pair[0] >= pair[1] or pair[0] < previous):
            raise ValueError("平台原始采样索引范围无效、重叠或无序")
        previous = pair[1]
    return value


def verify_record_idle_evidence(evidence, records=(), *, base_dir=None,
                                historical_consumer_source=None, artifact_relocations=None):
    """Check a self-contained certificate without inventing record approvals."""
    errors, files = [], {}
    report = None
    try:
        from .artifact_paths import ArtifactResolver
        resolver = ArtifactResolver(artifact_relocations, base_dir=base_dir)
        if (evidence.get("mode") != MODE or evidence.get("version") != VERSION
                or evidence.get("validated") is not True or evidence.get("applicable") is not True):
            raise ValueError("缺少当前版本的本记录空载标签资格")
        domain = evidence.get("domain", {})
        if (not domain.get("machine_id") or domain.get("power_channel") != "spindle_power_w"
                or domain.get("scope") != "record_local_constant_speed_only"):
            raise ValueError("本记录空载标签适用域必须限定机床、功率通道和同记录恒速范围")
        path = resolver.resolve(evidence["report_path"])
        if not path.is_file() or file_hash(path) != evidence.get("report_sha256"):
            raise ValueError("本记录空载标签报告不存在或SHA-256不匹配")
        files["report"] = dict(path=str(path), sha256=evidence["report_sha256"])
        report = json.loads(path.read_text(encoding="utf-8"))
        expected = dict(schema=SCHEMA, mode=MODE, version=VERSION, domain=domain,
                        status="qualified", eligible_for_load_training=True, interface_version=MASK_VERSION)
        if any(report.get(key) != value for key, value in expected.items()):
            raise ValueError("本记录空载标签报告尚未审核放行，或版本/适用域不匹配")
        policy = report.get("selection_policy", {})
        from .base_qualification import _embedded
        if _embedded(report.get("selection_policy_artifact"), "平台选择规则") != policy:
            raise ValueError("平台选择规则与嵌入的冻结规则不一致")
        source_directory = (resolver.resolve(historical_consumer_source) / "src/project/load_prediction"
                            if historical_consumer_source is not None else Path(__file__).parent)
        for name in LABEL_SOURCES:
            if report.get("production_source_sha256", {}).get(name) != file_hash(source_directory / name):
                role = "历史消费者源码" if historical_consumer_source is not None else "当前接口源码"
                raise ValueError(f"本记录标签资格与{role}不一致：{name}")
        if not policy.get("version") or policy.get("uses_power_for_selection") is not False:
            raise ValueError("平台必须按NC无接触、转速与连续性选择，不能按低功率选择")
        for name in POLICY_FIELDS:
            value = policy.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"平台/恒速掩码规则缺少正有限参数：{name}")
            if value != POLICY_VALUES[name]:
                raise ValueError(f"当前标签版本不允许改变已确认规则：{name}")
        if policy.get("speed_smoothing_s", .101) != .101:
            raise ValueError("当前恒速掩码必须使用既有0.101秒平滑转速导数")
        audits = report.get("record_audits", [])
        indexed = {row.get("record_id"): row for row in audits if isinstance(row, dict)}
        if not indexed or len(indexed) != len(audits) or None in indexed:
            raise ValueError("本记录空载报告必须包含唯一逐记录审核")
        for record in records:
            meta = {**record.get("metadata", {}), **record}
            key = meta.get("record_id")
            audit = indexed.get(key)
            if not audit:
                errors.append(f"记录 {key} 缺少平台审核或显式排除")
                continue
            if meta.get("machine_id") != domain["machine_id"]:
                errors.append(f"记录 {key} 的机床不在本记录标签适用域")
            hashes = meta.get("file_sha256", {})
            for kind in ("measurement", "nc", "process", "machine"):
                if not hashes.get(kind) or audit.get("file_sha256", {}).get(kind) != hashes[kind]:
                    errors.append(f"记录 {key} 的{kind}内容未绑定平台审核")
            status = audit.get("status")
            if type(audit.get("sample_count")) is not int or audit["sample_count"] < 1:
                errors.append(f"记录 {key} 缺少完整原始采样数量")
            if status == "excluded":
                if not audit.get("qualification_reasons"):
                    errors.append(f"记录 {key} 排除未说明原因")
                continue
            if status != "qualified":
                errors.append(f"记录 {key} 平台审核尚未完成")
                continue
            for name in ("no_contact_verified", "same_speed_verified", "continuity_verified"):
                if audit.get(name) is not True:
                    errors.append(f"记录 {key} 尚未核验 {name}")
            for name in ("commanded_rpm", "idle_power_w"):
                value = audit.get(name)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    errors.append(f"记录 {key} 的{name}无效")
            _ranges(audit.get("platform_ranges"))
            _ranges(audit.get("derivative_context_ranges"), single=False)
            selection = _embedded(audit.get("selection_artifact"), "平台选择")
            if (audit.get("selection_sha256") != audit["selection_artifact"]["sha256"]
                    or any(selection.get(k) != audit.get(k) for k in ("record_id", "platform_ranges", "commanded_rpm"))
                    or selection.get("measurement_sha256") != hashes.get("measurement")):
                errors.append(f"记录 {key} 平台选择证据与当前记录不一致")
            if meta.get("dataset_role") == "development":
                from .experiment_protocol import round_spec
                if audit.get("commanded_rpm") != round_spec(meta["round_id"])["spindle_cmd_rpm"]:
                    errors.append(f"记录 {key} 平台转速与明确参数组不一致")
    except (ValueError, OSError, TypeError, KeyError, AttributeError) as exc:
        errors.append(f"本记录空载标签证据不可用：{exc}")
    return dict(valid=not errors, issues=errors, files=files, report=report)


def _continuous_runs(time, indices, valid, period):
    from .feed_samples import _blocks
    for start, stop in _blocks(time, valid, period):
        edges = np.r_[start, np.flatnonzero(np.diff(indices[start:stop]) != 1) + start + 1, stop]
        for first, last in zip(edges[:-1], edges[1:]):
            yield int(first), int(last)


def prepare_record_idle_power(measurement, evidence, metadata):
    """Recheck observed platform and derive a provenance-bound raw label mask."""
    check = verify_record_idle_evidence(evidence, [metadata])
    if not check["valid"]:
        raise ValueError("；".join(check["issues"]))
    report = check["report"]
    audit = next(row for row in report["record_audits"] if row["record_id"] == metadata["record_id"])
    policy = report["selection_policy"]
    if file_hash(measurement.source) != audit["file_sha256"]["measurement"]:
        raise ValueError("当前实测来源与本记录平台资格SHA-256不一致")
    return _record_idle_power(measurement, audit, policy, evidence["report_sha256"])


def _record_idle_power(measurement, audit, policy, report_sha256, *, after_reference=True):
    """Pure numeric platform/mask operation; callers verify source and protocol."""
    time = np.asarray(measurement.time_s, float)
    if len(time) != audit["sample_count"]:
        raise ValueError("当前完整实测样本数量与平台审核绑定不一致")
    indices = np.asarray(measurement.source_indices)
    if (indices.shape != time.shape or indices.dtype.kind not in "iu" or np.any(indices < 0)
            or np.any(np.diff(indices) <= 0)):
        raise ValueError("本记录标签必须保留有序、唯一的原始采样索引")
    valid = np.zeros(len(time), bool)
    power_w = np.full(len(time), np.nan)
    label = dict(schema=MASK_VERSION, method=MODE, version=VERSION, record_id=audit["record_id"],
                 report_sha256=report_sha256, measurement_sha256=audit["file_sha256"]["measurement"],
                 status=audit["status"], qualification_reasons=audit.get("qualification_reasons", []),
                 source_index_semantics="zero_based_ChannelData_half_open", raw_sample_count=len(time),
                 platform_ranges=audit.get("platform_ranges", []), valid_sample_ranges=[],
                 policy=policy, prediction_input=False)
    if audit["status"] == "qualified":
        from .feed_samples import _period
        from .base_reference import idle_platform_quality
        from .base_power import CalibrationConfig, speed_features
        measurement.require("spindle_speed_rpm", "spindle_power_w")
        speed = np.asarray(measurement.values["spindle_speed_rpm"], float)
        power = np.asarray(measurement.values["spindle_power_w"], float)
        period = _period(time, measurement.metadata)
        if period is None or not math.isfinite(period) or period <= 0:
            raise ValueError("本记录标签缺少有效采样周期")
        usable = np.asarray(measurement.valid, bool) & np.isfinite(time) & np.isfinite(speed) & np.isfinite(power)
        nominal = audit["commanded_rpm"]
        tolerance = max(policy["speed_tolerance_abs_rpm"], policy["speed_tolerance_relative"] * nominal)
        platform = np.zeros(len(time), bool)
        for first, last in audit["platform_ranges"]:
            selection = (indices >= first) & (indices < last)
            selected = np.flatnonzero(selection)
            if (len(selected) != last - first or not usable[selection].all()
                    or not np.all(np.diff(indices[selection]) == 1)
                    or np.any(np.diff(time[selection]) <= 0) or np.any(np.diff(time[selection]) > 1.5 * period)):
                raise ValueError("平台区间包含缺失、无效值或采样断点")
            if time[selected[-1]] - time[selected[0]] + period < policy["minimum_platform_duration_s"] - 1e-8:
                raise ValueError("平台区间不足已冻结最小时长")
            platform |= selection
        if np.any(np.abs(speed[platform] - nominal) > tolerance):
            raise ValueError("平台实测转速与该记录指令转速不一致")
        # Reproduce the audited prefix context. Filtering just the 1-second
        # platform or extending its right edge changes Savitzky-Golay edges.
        platform_stable = np.zeros(len(time), bool)
        for first, last in audit["derivative_context_ranges"]:
            use = (indices >= first) & (indices < last)
            if (int(use.sum()) != last - first or int(use.sum()) < 3 or not usable[use].all()
                    or np.any(np.diff(indices[use]) != 1) or np.any(np.diff(time[use]) <= 0)
                    or np.any(np.diff(time[use]) > 1.5 * period)):
                raise ValueError("平台导数上下文含缺口或无效值")
            _, rate = speed_features(time[use], speed[use], CalibrationConfig(smooth_s=.101))
            platform_stable[use] = np.abs(rate) < policy["max_abs_acceleration_rpm_s"]
        if not platform_stable[platform].all():
            raise ValueError("平台原审核上下文的平滑转速导数未通过")
        quality = idle_platform_quality(time[platform], power[platform])
        if not quality["passed"]:
            raise ValueError("本记录空载平台仍在恢复或波动超出既定规则")
        idle = float(np.median(power[platform]))
        if not np.isclose(idle, audit["idle_power_w"], atol=1e-9, rtol=1e-12):
            raise ValueError("平台中位功率与资格报告不一致")
        stable = np.zeros(len(time), bool)
        for start, stop in _continuous_runs(time, indices, usable, period):
            if stop - start < 3:
                continue
            _, derivative = speed_features(time[start:stop], speed[start:stop], CalibrationConfig(smooth_s=.101))
            rate = np.abs(derivative)
            stable[start:stop] = (rate < policy["max_abs_acceleration_rpm_s"]) & (np.abs(speed[start:stop] - nominal) <= tolerance)
        for start, stop in _continuous_runs(time, indices, stable, period):
            valid[start:stop] = time[start:stop] - time[start] >= policy["stable_hold_s"] - 1e-10
        if after_reference:
            valid &= indices >= max(last for _, last in audit["platform_ranges"])
        for start, stop in _continuous_runs(time, indices, valid, period):
            label["valid_sample_ranges"].append([int(indices[start]), int(indices[stop - 1]) + 1])
        power_w[:] = idle
        label.update(idle_power_w=idle, commanded_rpm=nominal, platform_quality=quality,
                     selection_sha256=audit["selection_sha256"],
                     derivative="base_power.speed_features_on_complete_valid_contiguous_measurement_block",
                     speed_smoothing_s=.101, hold_semantics="retain_after_0.5_seconds_continuously_speed_qualified",
                     derivative_context_ranges=audit["derivative_context_ranges"],
                     rejection_counts=dict(invalid_or_nonfinite=int((~usable).sum()),
                         speed_or_acceleration=int((usable & ~stable).sum()),
                         not_retained_by_hold_or_platform_end=int((stable & ~valid).sum())))
    label["valid_samples"] = int(valid.sum())
    label["mask_sha256"] = _digest(dict(valid_sample_ranges=label["valid_sample_ranges"],
        valid_samples=label["valid_samples"], raw_sample_count=len(time), method=MASK_VERSION,
        report_sha256=report_sha256, measurement_sha256=audit["file_sha256"]["measurement"]))
    return dict(power_w=power_w, valid=valid, status="validated_in_sample_domain" if audit["status"] == "qualified" else "excluded_record_idle_labels",
                formal_training_release=True, model_version=VERSION, prediction_version=VERSION,
                interface_version=MASK_VERSION, qualification_report_sha256=report_sha256,
                qualification_issues=audit.get("qualification_reasons", []),
                timing_source="observed_record_samples_post_machining_only", label_metadata=label)


def verify_prepared_record_idle(sample, evidence):
    """Check the versioned mask receipt after snapshot round-trip, without raw I/O."""
    label = sample.metadata.get("record_idle_label", {})
    required = dict(schema=MASK_VERSION, method=MODE, version=VERSION,
                    record_id=sample.metadata.get("record_id"), report_sha256=evidence.get("report_sha256"),
                    measurement_sha256=sample.metadata.get("file_sha256", {}).get("measurement"),
                    prediction_input=False)
    if any(label.get(key) != value for key, value in required.items()):
        raise ValueError("本记录标签快照缺少匹配的方法、来源或资格绑定")
    digest_value = _digest(dict(valid_sample_ranges=label["valid_sample_ranges"],
        valid_samples=label["valid_samples"], raw_sample_count=label["raw_sample_count"], method=MASK_VERSION,
        report_sha256=evidence["report_sha256"], measurement_sha256=label["measurement_sha256"]))
    if label.get("mask_sha256") != digest_value:
        raise ValueError("本记录恒速标签掩码指纹不一致")
    if label.get("status") == "excluded":
        if (label["valid_samples"] or label["valid_sample_ranges"]
                or sample.frame.valid_main.any() or sample.frame.valid_residual.any()):
            raise ValueError("显式排除记录不能包含任何有效负载标签")
    elif label.get("status") != "qualified":
        raise ValueError("本记录平台资格状态尚未明确")
