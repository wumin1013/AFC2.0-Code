"""Explicitly authorized mixed label estimation, not metrological release."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np

from .common import file_hash
from .record_idle import _continuous_runs, _digest, _ranges, _record_idle_power, POLICY_VALUES

MODE = "hybrid_estimated_labels"
VERSION = "afc.hybrid_estimated_labels.1"
SCHEMA = "afc.hybrid.label_protocol.1"
STATUS = "authorized_estimation_protocol"
MEASURED = "record_idle_measured"
ESTIMATED = "frozen_dynamic_estimated"
AFTER = "after_reference_end"
ENTIRE = "entire_same_speed_stage_with_background_representativeness_assumption"
LABEL_SOURCES = ("hybrid_labels.py", "record_idle.py", "base_qualification.py", "base_reference.py",
                 "base_power.py", "base_events.py", "samples.py", "measurement.py", "sample_rules.py", "training_service.py")
SOURCE_NAMES = {MEASURED: "postoperative_measured_idle", ESTIMATED: "frozen_model_estimate"}
PLATFORM_FIELDS = ("platform_ranges", "derivative_context_ranges", "idle_power_w", "commanded_rpm",
                   "selection_sha256", "sample_count", "file_sha256")


def verify_hybrid_evidence(evidence, records=(), *, base_dir=None,
                           historical_consumer_source=None, artifact_relocations=None):
    """Validate authorization and exact inputs without promoting old qualification."""
    from .base_power import BasePowerModel
    from .base_qualification import _embedded
    from .artifact_paths import ArtifactResolver
    errors, files, report = [], {}, None
    try:
        resolver = ArtifactResolver(artifact_relocations, base_dir=base_dir)
        required = dict(mode=MODE, protocol_version=VERSION, status=STATUS,
                        authorized_for_training=True, formal_metrological_qualification=False)
        if any(evidence.get(k) != v for k, v in required.items()):
            raise ValueError("缺少明确授权的混合估计协议；不能继承旧动态计量资格")
        for kind in ("model", "report"):
            path = resolver.resolve(evidence[kind + "_path"])
            if not path.is_file() or file_hash(path) != evidence[kind + "_sha256"]:
                raise ValueError(f"估计协议{kind}文件SHA-256不一致")
            files[kind] = dict(path=str(path.resolve()), sha256=evidence[kind + "_sha256"])
        report = json.loads(Path(files["report"]["path"]).read_text(encoding="utf-8"))
        if "decision_artifact" in report:
            _embedded(report["decision_artifact"], "作者授权决定")
        if any(report.get(k) != v for k, v in {**required, "schema": SCHEMA}.items()):
            raise ValueError("混合报告尚未授权或协议状态不一致")
        domain = evidence.get("domain", {})
        if (domain != report.get("domain") or not domain.get("machine_id")
                or domain.get("power_channel") != "spindle_power_w" or domain.get("scope") != "record_local_hybrid_estimation"):
            raise ValueError("混合估计域与报告不一致")
        model = BasePowerModel.load(files["model"]["path"])
        if (model.schema != "afc.base.2" or model.version != evidence.get("model_version")
                or any(report.get("model", {}).get(k) != v for k, v in
                       dict(version=model.version, sha256=evidence["model_sha256"], original_status=model.status).items())):
            raise ValueError("混合协议必须保留原冻结模型身份和状态")
        source_directory = (resolver.resolve(historical_consumer_source) / "src/project/load_prediction"
                            if historical_consumer_source is not None else Path(__file__).parent)
        for name in LABEL_SOURCES:
            if report.get("production_source_sha256", {}).get(name) != file_hash(source_directory / name):
                role = "历史消费者源码" if historical_consumer_source is not None else "当前源码"
                raise ValueError(f"混合协议与{role}不一致：{name}")
        policy = report.get("selection_policy", {})
        if (_embedded(report.get("selection_policy_artifact"), "平台规则") != policy
                or policy.get("uses_power_for_selection") is not False
                or any(policy.get(k) != v for k, v in POLICY_VALUES.items())
                or policy.get("speed_smoothing_s", .101) != .101):
            raise ValueError("混合协议不得改变已有平台数值规则")
        parent = _embedded(report.get("parent_record_idle_report"), "父平台报告")
        if report.get("parent_source_sha256") != parent.get("production_source_sha256"):
            raise ValueError("父平台源码身份未保留；历史身份不等于当前r4源码")
        parent_rows = {r["record_id"]: r for r in parent["record_audits"]}
        rows = report.get("records")
        if not isinstance(rows, dict) or not rows or any(k != row.get("record_id") for k, row in rows.items()):
            raise ValueError("混合协议需要唯一逐记录分发表")
        for item in records:
            meta = {**item.get("metadata", {}), **item}
            rid = meta.get("record_id")
            row = rows.get(rid)
            if not row or row.get("route") not in SOURCE_NAMES:
                raise ValueError(f"记录{rid}缺少唯一标签来源")
            if meta.get("machine_id") != domain["machine_id"] or meta.get("dataset_role") != "development":
                raise ValueError(f"记录{rid}超出本轮开发域")
            hashes = meta.get("file_sha256", {})
            if any(not hashes.get(k) or row.get("file_sha256", {}).get(k) != hashes[k] for k in ("measurement", "nc", "process", "machine")):
                raise ValueError(f"记录{rid}来源SHA不一致")
            if type(row.get("sample_count")) is not int or row["sample_count"] < 3:
                raise ValueError(f"记录{rid}缺少完整原始样本数")
            from .experiment_protocol import round_spec
            if row.get("commanded_rpm") != round_spec(meta["round_id"])["spindle_cmd_rpm"]:
                raise ValueError(f"记录{rid}转速与参数组不一致")
            if row["route"] == ESTIMATED:
                if (row.get("original_model_status") != model.status or not row.get("qualification_reasons")
                        or row.get("status") == "qualified"):
                    raise ValueError(f"记录{rid}必须保留模型估计等级，不能伪造动态资格")
                continue
            if (row.get("status") != "qualified" or any(row.get(k) is not True for k in
                    ("no_contact_verified", "same_speed_verified", "continuity_verified"))):
                raise ValueError(f"记录{rid}平台证据未核实")
            if not np.isfinite(row.get("idle_power_w", np.nan)) or row["idle_power_w"] <= 0:
                raise ValueError(f"记录{rid}平台功率无效")
            _ranges(row["platform_ranges"])
            _ranges(row["derivative_context_ranges"], single=False)
            selection = _embedded(row["selection_artifact"], "平台选择")
            if (row.get("selection_sha256") != row["selection_artifact"]["sha256"]
                    or any(selection.get(k) != row.get(k) for k in ("record_id", "platform_ranges", "commanded_rpm"))
                    or selection.get("measurement_sha256") != hashes["measurement"]):
                raise ValueError(f"记录{rid}平台选择未绑定")
            application = row.get("reference_application")
            if application == AFTER:
                original = parent_rows.get(rid, {})
                if original.get("status") != "qualified" or any(row.get(k) != original.get(k) for k in PLATFORM_FIELDS):
                    raise ValueError(f"记录{rid}改变了父平台数值证据")
            elif application != ENTIRE or row.get("representativeness_assumed") is not True:
                raise ValueError(f"记录{rid}缺少同转速阶段背景代表性假设")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        errors.append(str(exc))
    return dict(valid=not errors, issues=errors, files=files, report=report)


def label_metadata(row, evidence, *, valid_ranges, sample_count, warnings=(), original_status=None, parent=None):
    label = dict(schema=VERSION, mode=MODE, status=STATUS, authorized_for_training=True,
        formal_metrological_qualification=False, record_id=row["record_id"], route=row["route"],
        label_source=SOURCE_NAMES[row["route"]], reference_kind=SOURCE_NAMES[row["route"]],
        report_sha256=evidence["report_sha256"], measurement_sha256=row["file_sha256"]["measurement"],
        raw_sample_count=sample_count, valid_sample_ranges=valid_ranges,
        valid_samples=sum(b-a for a, b in valid_ranges), warnings=list(warnings),
        original_model_status=original_status, model_version=evidence.get("model_version") if row["route"] == ESTIMATED else None,
        model_sha256=evidence.get("model_sha256") if row["route"] == ESTIMATED else None,
        reference_application=row.get("reference_application"), idle_power_w=row.get("idle_power_w"),
        representativeness_assumed=row.get("representativeness_assumed", False),
        evidence_grade="model_estimated" if row["route"] == ESTIMATED else "observed_platform_with_background_assumption",
        prediction_input=False, parent_computation=parent)
    label["mask_sha256"] = _digest(label)
    return label


def prepare_hybrid_power(measurement, model, evidence, metadata):
    check = verify_hybrid_evidence(evidence, [metadata])
    if not check["valid"]:
        raise ValueError("；".join(check["issues"]))
    from .base_power import BasePowerModel
    if asdict(model) != asdict(BasePowerModel.load(check["files"]["model"]["path"])):
        raise ValueError("内存基础模型与混合协议不一致")
    row = check["report"]["records"][metadata["record_id"]]
    if file_hash(measurement.source) != row["file_sha256"]["measurement"]:
        raise ValueError("混合标签当前实测文件SHA不一致")
    if len(measurement.time_s) != row["sample_count"]:
        raise ValueError("混合标签完整样本数量不一致")
    if row["route"] == MEASURED:
        result = _record_idle_power(measurement, row, check["report"]["selection_policy"],
            evidence["report_sha256"], after_reference=row["reference_application"] == AFTER)
        warnings = (["同记录后段平台代表整个同转速阶段的背景功率是假设；初始不匹配转速仍被屏蔽"]
                    if row["reference_application"] == ENTIRE else [])
        ranges = result["label_metadata"]["valid_sample_ranges"]
        original_status = None  # This route has no dynamic-model status to promote.
    else:
        from .base_power import predict_base_from_measured_speed
        from .feed_samples import _period
        time, indices = np.asarray(measurement.time_s), np.asarray(measurement.source_indices)
        if (indices.shape != time.shape or indices.dtype.kind not in "iu" or np.any(indices < 0)
                or np.any(np.diff(indices) <= 0)):
            raise ValueError("动态估计必须保留有序原始采样索引")
        speed = np.asarray(measurement.values["spindle_speed_rpm"])
        usable = measurement.valid & np.isfinite(time) & np.isfinite(speed) & np.isfinite(measurement.values["spindle_power_w"])
        period = _period(time, measurement.metadata)
        if period is None or not math.isfinite(period) or period <= 0:
            raise ValueError("动态估计标签缺少有效采样周期")
        power, valid, warnings = np.full(len(time), np.nan), np.zeros(len(time), bool), []
        for a, b in _continuous_runs(time, indices, usable, period):
            if b-a < 3:
                continue
            predicted = predict_base_from_measured_speed(time[a:b], speed[a:b], model)
            power[a:b], valid[a:b] = predicted["power_w"], predicted["valid"]
            warnings.extend(f"[{int(indices[a])},{int(indices[b-1])+1}): {w}" for w in predicted.get("warnings", []))
        ranges = [[int(indices[a]), int(indices[b-1])+1] for a, b in _continuous_runs(time, indices, valid, period)]
        result = dict(power_w=power, valid=valid)
        original_status = model.status
    label = label_metadata(row, evidence, valid_ranges=ranges, sample_count=row["sample_count"],
                           warnings=warnings, original_status=original_status)
    result.update(status=STATUS, authorized_for_training=True, formal_training_release=False,
        formal_metrological_qualification=False, original_model_status=original_status,
        model_version=evidence["model_version"] if row["route"] == ESTIMATED else VERSION,
        interface_version=VERSION, prediction_version=VERSION, qualification_report_sha256=evidence["report_sha256"],
        qualification_issues=row.get("qualification_reasons", []), warnings=warnings,
        timing_source="postoperative_hybrid_label_estimation", hybrid_label_metadata=label)
    return result


def verify_prepared_hybrid(sample, evidence, row=None):
    label = sample.metadata.get("hybrid_label", {})
    if (label.get("schema") != VERSION or label.get("status") != STATUS or label.get("prediction_input") is not False
            or label.get("record_id") != sample.metadata.get("record_id")
            or label.get("measurement_sha256") != sample.metadata.get("file_sha256", {}).get("measurement")
            or label.get("report_sha256") != evidence.get("report_sha256")
            or label.get("formal_metrological_qualification") is not False
            or label.get("authorized_for_training") is not True
            or label.get("mask_sha256") != _digest({k: v for k, v in label.items() if k != "mask_sha256"})):
        raise ValueError("混合标签快照来源、授权或掩码回执不一致")
    if (label.get("route") not in SOURCE_NAMES or label.get("label_source") != SOURCE_NAMES[label["route"]]
            or label.get("reference_kind") != label.get("label_source")):
        raise ValueError("混合标签来源等级不一致")
    ranges = label.get("valid_sample_ranges")
    count = label.get("raw_sample_count")
    if (type(count) is not int or count < 3 or not isinstance(ranges, list)
            or any(not isinstance(r, list) or len(r) != 2 or any(type(v) is not int for v in r)
                   or not 0 <= r[0] < r[1] <= count for r in ranges)
            or any(a[1] > b[0] for a, b in zip(ranges, ranges[1:]))
            or label.get("valid_samples") != sum(b-a for a, b in ranges)):
        raise ValueError("混合标签原始有效范围或计数不一致")
    if row is not None and (row["record_id"] != label["record_id"] or row["route"] != label["route"]
            or row["sample_count"] != count or row.get("reference_application") != label.get("reference_application")
            or row.get("idle_power_w") != label.get("idle_power_w")):
        raise ValueError("混合标签与逐记录分发表不一致")
    if label["route"] == ESTIMATED and (label.get("model_version") != evidence.get("model_version")
            or label.get("model_sha256") != evidence.get("model_sha256")
            or row is not None and label.get("original_model_status") != row.get("original_model_status")):
        raise ValueError("模型估计标签丢失冻结模型身份或原资格状态")
    for name in ("label_source", "reference_kind"):
        if name not in sample.frame or not sample.frame[name].eq(label[name]).all():
            raise ValueError("逐点标签来源丢失或改变")
    if not label["valid_samples"] and (sample.frame.valid_main.any() or sample.frame.valid_residual.any()):
        raise ValueError("空原始有效域不能产生有效训练标签")
