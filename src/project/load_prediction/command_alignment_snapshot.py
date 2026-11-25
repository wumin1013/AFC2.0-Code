"""Verify command-position supervision independently of historical load routes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import file_hash
from .training_checkpoint import digest, source_fingerprints

SCHEMA = "afc.command_alignment_derivation.1"
RECEIPT_SCHEMA = "afc.command_alignment_dataset_receipt.1"
RELEASE_SCHEMA = "afc.command_alignment_release.1"
POSITION_POLICY = "command_only_v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_reference(reference, *, read_json=True):
    _require(isinstance(reference, dict) and set(("path", "sha256")) <= set(reference),
             "指令位置证据缺少路径或SHA-256")
    _require(isinstance(reference["path"], str) and bool(reference["path"])
             and isinstance(reference["sha256"], str) and len(reference["sha256"]) == 64,
             "指令位置证据引用无效")
    path = Path(reference["path"])
    _require(path.is_file() and file_hash(path) == reference["sha256"], "指令位置证据文件身份不符")
    if not read_json:
        return path
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(file_hash(path) == reference["sha256"], "指令位置证据读取期间改变")
    return value


def verify_command_record(sample, entry, audit, directory, *, verified_geometry=None):
    """Bind the source, exact prepared values and separately reported domains."""
    from .calibration_evidence import measured_frame_digest
    from .main_stability import verify_stability_frame
    from .process_coordinates import PROCESS_COORDINATE_RULES_VERSION
    from .sample_rules import continuous_ids

    meta, frame = sample.metadata, sample.frame
    rid = meta["record_id"]
    _require(meta == entry["metadata"] and audit.get("record_id") == rid
             and audit.get("file") == entry["file"] and audit.get("sha256") == entry["sha256"],
             "指令位置逐记录证据与快照不一致")
    path = (Path(directory) / entry["file"]).resolve()
    _require(path.is_relative_to(Path(directory).resolve()) and file_hash(path) == entry["sha256"],
             "指令位置样本越出目录或文件身份不符")
    _require(entry["dtypes"] == {name: str(kind) for name, kind in frame.dtypes.items()}, "指令位置样本列类型不符")
    frozen = pd.read_csv(path, float_precision="round_trip")
    _require(frozen.equals(frame), "内存样本与冻结CSV数值不一致")
    _require(meta.get("position_policy") == POSITION_POLICY and meta.get("position_source") == "command"
             and meta.get("position_columns") == ["command_x_mm", "command_y_mm", "command_z_mm"]
             and meta.get("process_mapping_rules_version") == PROCESS_COORDINATE_RULES_VERSION,
             "指令位置策略、实际通道或工艺定位版本不符")
    _require(all(meta.get(key) is True for key in ("steady_qualification_rebuilt", "supervision_rebuilt",
                                                   "prospective_sequence_rebuilt")), "指令位置监督或稳态资格未重建")
    _require(meta.get("prepared_measurement_sha256") == measured_frame_digest(frame), "新标签数值身份不符")
    alignment = checked_reference(audit.get("alignment_audit"))
    _require(meta.get("alignment_evidence") == audit["alignment_audit"]
             and alignment.get("record_id") == rid and alignment.get("position_policy") == POSITION_POLICY
             and alignment.get("uses_power_for_selection") is False
             and alignment.get("uses_actual_xyz_for_selection") is False, "位置选择使用了标签或未绑定指令通道证据")
    selected = alignment.get("selected", {})
    _require(selected.get("line_offset") == meta.get("measurement_line_offset")
             and selected.get("position_offset_mm") == meta.get("position_offset_mm"), "实际采用的位置变换与审计不一致")
    geometry_ref = audit.get("geometry_receipt")
    geometry = checked_reference(geometry_ref)
    _require(geometry.get("complete") is True and geometry.get("source_identity_preserved") is True
             and geometry.get("geometry_unknown") == 0 and geometry.get("version") == PROCESS_COORDINATE_RULES_VERSION
             and geometry.get("source_sha256") == meta["file_sha256"]["process"]
             and geometry.get("nc_sha256") == meta["file_sha256"]["nc"], "工艺几何重建证据不完整")
    key = geometry_ref["sha256"]
    if verified_geometry is None or key not in verified_geometry:
        base = Path(geometry_ref["path"]).parent
        for name, checksum in geometry["output_sha256"].items():
            output = (base / name).resolve()
            _require(output.is_relative_to(base.resolve()) and file_hash(output) == checksum, "重建工艺或稳态证据文件改变")
        geometry_frame = pd.read_csv(base / "geometry.csv", float_precision="round_trip")
        if verified_geometry is not None:
            verified_geometry[key] = geometry_frame
    else:
        geometry_frame = verified_geometry[key]
    columns = [name for name in geometry_frame if name not in ("feed_input_mm_min", "feed_source")]
    _require(set(columns) <= set(frame) and geometry_frame[columns].equals(frame[columns]),
             "标签准备改变了加工前几何、稳态或上下文")
    derived = meta.get("derived_process", {})
    _require(derived.get("receipt_path") == geometry_ref["path"]
             and derived.get("receipt_sha256") == key
             and derived.get("sha256") == geometry["output_sha256"].get("source.csv"), "样本未绑定重建工艺来源")
    checked_reference(derived, read_json=False)
    coverage = checked_reference(audit.get("coverage_audit"))
    _require(coverage.get("record_id") == rid and coverage.get("passed") is True
             and coverage.get("no_unexplained_loss") is True and coverage.get("grid_points") == len(frame)
             and coverage.get("geometry_unknown") == 0
             and coverage.get("valid_main") == int(frame.valid_main.sum())
             and coverage.get("valid_residual") == int(frame.valid_residual.sum()), "逐点覆盖审计未通过或计数不一致")
    checked_reference(coverage.get("lost_points"), read_json=False)
    domains = coverage.get("parent_domains", {})
    _require(set(domains) == {"aligned", "steady", "valid_main", "valid_residual", "spindle_stable"},
             "覆盖审计缺少原固定域")
    for name, domain in domains.items():
        _require(domain["old"] == domain["common"] + domain["lost"]
                 and domain["new"] == domain["common"] + domain["gained"]
                 and domain["new"] == int(frame[name].sum())
                 and domain["lost"] == sum(domain["lost_reasons"].values())
                 and not any("unexplained" in key or key.startswith("recomputed_") for key in domain["lost_reasons"]),
                 "原固定域丢点原因或计数未完整核对")
    platform = checked_reference(audit.get("platform_geometry"))
    _require(platform.get("record_id") == rid and platform.get("passed") is True
             and platform.get("measurement_sha256") == meta["file_sha256"]["measurement"]
             and platform.get("derived_source_sha256") == derived["sha256"], "原空载平台未绑定新几何审计")
    if meta.get("hybrid_label", {}).get("route") == "record_idle_measured":
        _require(platform.get("no_contact_reverified") is True and bool(platform.get("checks")),
                 "原空载平台在新几何上未确认无接触")
    else:
        _require(platform.get("original_estimation_grade_preserved") is True, "估计标签来源等级未保留")
    _require(frame.process_geometry_known.all(), "切削网格仍有缺工艺输入点")
    _require(np.array_equal(frame.sequence_id.to_numpy(), continuous_ids(frame, known_only=True)),
             "加工前上下文边界与工艺几何不一致")
    _require(np.array_equal(frame.training_sequence_id.to_numpy(),
                           continuous_ids(frame, frame.valid_residual.to_numpy(bool), known_only=True)),
             "监督序列未按新标签域重建")
    for name in ("power_cut_w", "feed_actual_mm_min"):
        _require(np.isfinite(frame.loc[frame.valid_residual, name]).all(), "有效监督含非有限标签")
    verify_stability_frame(sample)


def _verify_release(service, snapshot, directory, receipt_reference):
    release = json.loads((service.output_dir / "command_alignment_release.json").read_text(encoding="utf-8"))
    reference = dict(path=str((Path(directory) / "snapshot.json").resolve()),
                     sha256=file_hash(Path(directory) / "snapshot.json"))
    _require(release.get("schema") == RELEASE_SCHEMA and release.get("snapshot_id") == snapshot["snapshot_id"]
             and release.get("candidate_snapshot") == reference and release.get("dataset_receipt") == receipt_reference
             and release.get("training_released") is True, "指令位置训练尚未放行或身份不符")
    for field, schema in (("window_audit", "paper1.command_alignment_windows.1"),
                          ("execution_recheck", "paper1.command_alignment_execution.1")):
        report = checked_reference(release.get(field))
        _require(report.get("schema") == schema and report.get("status") == "passed" and report.get("passed") is True
                 and report.get("candidate_snapshot") == reference and report.get("source") == source_fingerprints(),
                 f"指令位置训练缺少通过的{field}验证")
        artifacts = report.get("artifact_evidence")
        _require(isinstance(artifacts, list) and bool(artifacts), "执行或窗口检查缺少实际产物")
        for artifact in artifacts:
            checked_reference(artifact, read_json=False)
        if field == "window_audit":
            expected = {row["metadata"]["record_id"] for row in snapshot["records"]}
            records = report.get("records", [])
            _require(report.get("checked_records") == len(records) == 450 and report.get("failures") == []
                     and {row["record_id"] for row in records} == expected and all(row.get("passed") is True for row in records)
                     and bool(report.get("representative_reports")), "窗口检查未覆盖全部450条及实际部署前向")
            for item in report["representative_reports"]:
                forward = checked_reference(item)
                _require(forward.get("passed") is True and forward.get("train_deployment_context_exact") is True
                         and forward.get("deployment_outputs_label_invariant") is True, "加工前前向上下文检查不完整")
        else:
            _verify_execution_details(report)
    return release


def _verify_execution_details(report):
    required = ("initialization_exact", "fixed_weight_outputs_exact", "weighted_loss_equivalent",
                "accumulated_gradients_equivalent", "adam_update_equivalent", "resume_weights_exact",
                "resume_adam_exact", "resume_rng_exact", "resume_history_exact", "changed_data_resume_rejected")
    _require(all(report.get("checks", {}).get(key) is True for key in required)
             and report.get("losses_checked") == ["quantiles", "point"]
             and report.get("stages_checked") == ["main", "residual"] and report.get("updates_per_measurement") == 3
             and report.get("train_records") == 400 and report.get("validation_records") == 50
             and report.get("engineering_weights_reused_for_training") is False
             and report.get("scientific_training_started") is False, "全F1执行、数值或恢复检查不完整")
    expected = dict(device="cpu", cpu_threads=10, cpu_interop_threads=1, main_batch_points=32768,
                    residual_batch_windows=128, residual_batch_strategy="padded", loss_cache_mode="cpu",
                    loss_cache_max_bytes=1073741824, seed=42, learning_rate=.001, gradient_clip=1.)
    _require(all(report.get("execution_config", {}).get(key) == value for key, value in expected.items())
             and report.get("window_policy") == "prospective_v1", "执行检查偏离已冻结配置")
    for kind, loss in (("Q", "quantiles"), ("P", "point")):
        comparison = report.get("numerical_equivalence", {}).get(kind, {})
        recovery = report.get("recovery", {}).get(kind, {})
        _require(comparison.get("passed") is True and recovery.get("passed") is True
                 and recovery.get("fresh_process_resume") is True and recovery.get("changed_data_resume_rejected") is True,
                 "Q/P数值或新进程恢复证据缺失")
        process_ids = [checked_reference(recovery.get(name)).get("process_id") for name in ("continuous", "resumed")]
        for stage in ("main", "residual"):
            interrupted = checked_reference(recovery.get(stage + "_interrupted"))
            _require(interrupted.get("expected") is True and interrupted.get("interrupted_stage") == stage
                     and interrupted.get("checkpointed_epoch") == 1 and interrupted.get("completed") is False,
                     "恢复检查必须在主趋势和残差各自第一轮存档后真实中断")
            checked_reference(interrupted.get("checkpoint"), read_json=False)
            process_ids.append(interrupted.get("process_id"))
        _require(all(type(pid) is int and pid > 0 for pid in process_ids) and len(set(process_ids)) == 4,
                 "连续、中断和恢复必须使用四个独立进程")
        for stage in ("main", "residual"):
            pair = comparison.get("stages", {}).get(stage, {})
            restored = recovery.get("stages", {}).get(stage, {})
            _require(pair.get("passed") is True and pair.get("initial_weights", {}).get("exact") is True
                     and pair.get("initial_rng_exact") is True
                     and all(pair.get(key, {}).get("passed") is True for key in ("raw_gradients", "one_adam_weights"))
                     and all(pair.get("losses", {}).get(key, {}).get("passed") is True for key in ("train_loss", "validation_loss")),
                     "主趋势或残差缺少完整数值对等结果")
            _require(all(restored.get(key) is True for key in
                         ("passed", "bitwise_exact", "weights_exact", "adam_exact", "rng_exact", "history_exact"))
                     and restored.get("optimizer_updates") == 2, "中断恢复未精确还原完整训练状态")
            checked_reference(restored.get("continuous"), read_json=False)
            checked_reference(restored.get("resumed"), read_json=False)
        fixed = checked_reference(report.get("fixed_weight_outputs", {}).get(loss))
        _require(fixed.get("passed") is True and all(fixed.get("stages", {}).get(stage, {}).get("bitwise_equal_outputs") is True
                 for stage in ("main", "residual")), "固定权重前向输出不一致")


def verify_command_alignment_snapshot(service, snapshot, directory, *, loads, require_release=True):
    """Accept only an audited new dataset; historical route checks remain intact."""
    from .dataset_manifest import verify_base_evidence
    from .hybrid_labels import verify_prepared_hybrid
    from .process_coordinates import PROCESS_COORDINATE_RULES_VERSION
    from .support_fill_snapshot import _development_entries
    from .samples import NOMINAL_ROUTE

    _require(snapshot.get("synthetic") is False and snapshot.get("snapshot_id") == digest(
        {key: value for key, value in snapshot.items() if key != "snapshot_id"}), "指令位置快照内容身份不符")
    source = snapshot["source_snapshot"]
    derivation = source.get("command_alignment_derivation", {})
    _require(derivation.get("schema") == SCHEMA, "未登记指令位置派生")
    reference = dict(path=derivation.get("receipt"), sha256=derivation.get("receipt_sha256"))
    receipt = checked_reference(reference)
    _require(receipt.get("schema") == RECEIPT_SCHEMA and receipt.get("status") == "completed"
             and receipt.get("position_policy") == POSITION_POLICY
             and receipt.get("process_mapping_rules_version") == PROCESS_COORDINATE_RULES_VERSION
             and receipt.get("source_sha256") == source_fingerprints() and receipt.get("training_started") is False,
             "指令位置准备未完成或消费者源码身份已改变")
    parent = checked_reference(receipt.get("parent_snapshot"))
    _require(parent.get("snapshot_id") == receipt["parent_snapshot"].get("snapshot_id") == digest(
        {key: value for key, value in parent.items() if key != "snapshot_id"}), "指令位置父快照身份不符")
    entries = _development_entries(snapshot, load_only=True)
    parents = {row["metadata"]["record_id"]: row for row in _development_entries(parent)}
    rows = receipt.get("records", [])
    audits = {row["record_id"]: row for row in rows}
    samples = {sample.metadata["record_id"]: sample for sample in loads}
    _require(len(loads) == len(samples) == len(rows) == len(audits) == 450 and set(audits) == set(parents) == set(samples),
             "指令位置450条记录身份不完整或重复")
    _require(source.get("route") == NOMINAL_ROUTE, "指令位置负载快照必须使用已核验的名义进给路线")
    evidence = source.get("base_validation", {})
    _require(evidence.get("mode") == "hybrid_estimated_labels", "位置修订不能改换既有标签方案")
    check = verify_base_evidence(evidence, [sample.metadata for sample in loads], base_dir=directory)
    _require(check["valid"], "；".join(check["issues"]))
    base = receipt.get("base_evidence", {})
    _require(all(base.get(key) == evidence.get(key) for key in ("report_path", "report_sha256", "model_path", "model_sha256")),
             "指令位置派生标签证据未绑定")
    old_report = checked_reference(dict(path=base.get("parent_report_path"), sha256=base.get("parent_report_sha256")))
    _require(old_report["records"] == check["report"]["records"], "位置修订改变了旧标签来源或平台分发表")
    review = checked_reference(dict(path=base.get("source_review_path"), sha256=base.get("source_review_sha256")))
    _require(review.get("passed") is True and review.get("status") == "passed"
             and review.get("label_numeric_implementation_unchanged") is True
             and review.get("records_unchanged") is True and review.get("current_source") == source_fingerprints()
             and review.get("parent_report") == dict(path=base["parent_report_path"], sha256=base["parent_report_sha256"]),
             "旧标签数值实现及分发表未通过派生审查")
    geometry_cache = {}
    count = 0
    for entry in entries:
        rid = entry["metadata"]["record_id"]
        meta, previous = entry["metadata"], parents[rid]["metadata"]
        _require(all(meta.get(key) == previous.get(key) for key in
                     ("file_sha256", "measurement_content_sha256", "round_id", "feature", "dataset_role")),
                 "位置修订更换了原始记录身份")
        verify_command_record(samples[rid], entry, audits[rid], directory, verified_geometry=geometry_cache)
        verify_prepared_hybrid(samples[rid], evidence, check["report"]["records"][rid])
        count += len(samples[rid].frame)
    _require(receipt.get("totals", {}).get("records") == 450 and receipt["totals"].get("grid_points") == count
             and receipt["totals"].get("geometry_unknown") == 0, "指令位置全量覆盖汇总不一致")
    if require_release:
        _verify_release(service, snapshot, directory, reference)
    return receipt
