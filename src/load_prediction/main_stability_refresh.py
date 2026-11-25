"""Derive only main-stability masks from an immutable mixed-label snapshot.

The current consumer protocol is validated normally. A separate, strict chain
binds unchanged labels to their historical producer; no old receipt is edited.
"""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
import ast
import hashlib
import json
from pathlib import Path
import shutil
import traceback
import uuid

import numpy as np
import pandas as pd

from .artifact_paths import ArtifactResolver, archived_load_directory
from .base_qualification import _embedded, embedded_json_artifact
from .common import checkpoint, file_hash, save_json
from .feed_samples import align_spatial_channels
from .hybrid_labels import LABEL_SOURCES, MODE, verify_prepared_hybrid
from .main_stability import (VERSION, POLICY, SPATIAL_COLUMNS, stability_channels,
                             apply_stability_columns, verify_stability_frame)
from .measurement import Measurement, iter_measurement, read_header, resolve_position_policy, select_position_channels
from .samples import SampleSet
from .training_checkpoint import digest

SCHEMA = "afc.main_stability_derivation.1"
ROLE = "current_main_stability_consumer_of_immutable_historical_labels"
CHANGED_COLUMNS = ("spindle_stable", "valid_main")
ADDED_METADATA = ("main_stability_rules_version", "main_stability_audit", "main_stability_refresh")
RULE_FILES = ("main_stability.py", "main_stability_refresh.py", "samples.py", "training_service.py")
ALIGNMENT_TOLERANCES = dict(position_tolerance_mm=.1, length_tolerance_fraction=.05)


def _require(condition, text):
    if not condition:
        raise ValueError(text)


def _checked_json(path, expected=None):
    path = Path(path)
    if expected is not None:
        _require(path.is_file() and file_hash(path) == expected, f"证据文件改变：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _snapshot(path, expected_id):
    snapshot = _checked_json(path)
    _require(snapshot.get("snapshot_id") == expected_id
             and digest({k: v for k, v in snapshot.items() if k != "snapshot_id"}) == expected_id,
             "父准备快照元数据身份不一致")
    return snapshot


def _inside(directory, name):
    directory = Path(directory).resolve()
    path = ArtifactResolver(base_dir=directory).resolve(name)
    _require(path.is_relative_to(directory), "派生回执路径越界")
    return path


def _frame(directory, item):
    path = _inside(directory, item["file"])
    _require(file_hash(path) == item["sha256"], "父准备CSV哈希改变")
    frame = pd.read_csv(path, float_precision="round_trip")
    for column, dtype in item["dtypes"].items():
        if dtype == "bool":
            frame[column] = frame[column].astype(bool)
    return frame


def frame_digest(frame, columns):
    selected = frame.loc[:, list(columns)]
    value = hashlib.sha256(digest(dict(columns=list(columns), dtypes=[str(d) for d in selected.dtypes])).encode("ascii"))
    value.update(pd.util.hash_pandas_object(selected, index=True).to_numpy().tobytes())
    return value.hexdigest()


def _sources(source_root=None):
    directory = archived_load_directory(source_root) if source_root is not None else Path(__file__).parent
    return {name: file_hash(directory / name) for name in RULE_FILES}


def verify_parent_alignment_defaults(producer_source):
    """The historical service supplied no custom alignment tolerance arguments."""
    folder = archived_load_directory(producer_source)
    tree = ast.parse((folder / "samples.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "prepare_load_samples")
    defaults = {arg.arg: ast.literal_eval(value) for arg, value in zip(function.args.kwonlyargs, function.args.kw_defaults)
                if arg.arg in ALIGNMENT_TOLERANCES}
    _require(defaults == ALIGNMENT_TOLERANCES, "父源码对齐默认容差不同，需显式新审查")
    service_tree = ast.parse((folder / "training_service.py").read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(service_tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "prepare_load_samples"]
    _require(bool(calls) and all(not any(k.arg is None or k.arg in ALIGNMENT_TOLERANCES for k in call.keywords) for call in calls),
             "父服务覆盖过对齐容差，不能假定默认值")
    return dict(**ALIGNMENT_TOLERANCES, basis="verified_historical_service_default_arguments")


def read_stability_measurement(path, *, sample_period_s=None, cancel=None, position_policy=None):
    """Reuse the strict 65,536-row parser; retain only screening/alignment inputs.

    Its full-row quality mask is deliberately preserved. Power is retained only
    to reproduce the original finite-value alignment boundary, never amplitude.
    """
    header = read_header(path, sample_period_s)
    header["position_policy"] = resolve_position_policy(position_policy=position_policy)
    _, position = select_position_channels(header["mapping"], position_policy=header["position_policy"])
    _require(bool(position), "稳速刷新缺少完整实际或指令XYZ")
    columns = ("spindle_speed_rpm", "spindle_power_w", "program_line_raw", *position)
    _require(set(columns) <= set(header["mapping"]), "稳速刷新缺少既有对齐依赖通道")
    pieces, values = [], {key: [] for key in columns}
    for chunk in iter_measurement(path, chunk_size=65536, sample_period_s=sample_period_s, cancel=cancel):
        pieces.append((chunk.time_s, chunk.valid, chunk.source_indices, chunk.source_lines))
        for key in columns:
            values[key].append(chunk.values[key].copy())
    _require(bool(pieces), "稳速刷新原始记录为空")
    time, valid, indices, lines = (np.concatenate([p[i] for p in pieces]) for i in range(4))
    return Measurement(str(path), time, {k: np.concatenate(v) for k, v in values.items()}, valid,
                       header, indices, lines)


def refresh_record(sample, measurement, *, break_before=None):
    """Reproject new screening channels using exactly the old label support."""
    old, meta = sample.frame, sample.metadata
    label = meta["hybrid_label"]
    _require(measurement.source_indices is not None and len(measurement.time_s) == label["raw_sample_count"]
             and np.array_equal(measurement.source_indices, np.arange(label["raw_sample_count"])),
             "原始采样索引或总数与冻结标签不一致")
    _require(meta.get("position_offset_mm") is not None, "刷新必须复用既有统一坐标偏移")
    _require(all(meta.get(key, value) == value for key, value in ALIGNMENT_TOLERANCES.items()),
             "父元数据有不同对齐容差，不能静默回退默认")
    indices = measurement.source_indices
    valid = np.zeros(len(indices), bool)
    for start, stop in label["valid_sample_ranges"]:
        valid |= (indices >= start) & (indices < stop)
    valid &= np.isfinite(measurement.values["spindle_power_w"]) & np.isfinite(measurement.values["spindle_speed_rpm"])
    offset = meta.get("measurement_line_offset", 0)
    mapping = {int(v): int(v) + offset for v in np.unique(measurement.values["program_line_raw"]) if np.isfinite(v)}
    channels, arrays, audit = stability_channels(old, measurement, line_map=mapping, break_before=break_before)
    aligned, info = align_spatial_channels(old, measurement, channels, valid=valid, line_map=mapping,
        position_offset_mm=meta["position_offset_mm"], position_policy=resolve_position_policy(meta),
        **ALIGNMENT_TOLERANCES)
    expected_alignment = aligned.aligned.to_numpy(bool) & old.process_geometry_known.to_numpy(bool)
    _require(np.array_equal(expected_alignment, old.aligned.to_numpy(bool)), "仅稳速刷新改变了既有空间对齐支撑")
    np.testing.assert_array_equal(aligned.alignment_error_mm.to_numpy(), old.alignment_error_mm.to_numpy())
    _require(info.get("position_source") == meta.get("position_source"), "刷新改变了实际/指令定位来源")
    frame = old.copy()
    apply_stability_columns(frame, aligned)
    frame["valid_main"] = frame.valid_residual & frame.steady & frame.spindle_stable
    preserved = [c for c in old if c not in CHANGED_COLUMNS]
    pd.testing.assert_frame_equal(old[preserved], frame[preserved], check_exact=True)
    result = SampleSet(frame, {**deepcopy(meta), "main_stability_rules_version": VERSION,
                             "main_stability_audit": audit})
    verify_stability_frame(result)
    return result, arrays, dict(before_main=int(old.valid_main.sum()), after_main=int(frame.valid_main.sum()),
        residual_points=int(old.valid_residual.sum()), unchanged_columns=preserved,
        unchanged_frame_sha256=frame_digest(old, preserved), aligned_exact=True, alignment_error_exact=True,
        raw_rejection_reasons=audit["reason_counts"],
        spatial_counts=dict(numerical_support=int(frame.spindle_rate_support_valid.sum()),
            transition_guard=int((frame.spindle_transition_guard_interpolated > 0).sum()),
            speed_qualified=int(frame.spindle_stable.sum())))


def verify_derivation(snapshot, samples, *, directory, consumer_report,
                      artifact_relocations=None, historical_consumer_source=None):
    """Validate current consumer, immutable producer and every retained value."""
    directory = Path(directory)
    resolver = ArtifactResolver(artifact_relocations)
    consumer_source = (resolver.resolve(historical_consumer_source)
                       if historical_consumer_source is not None else None)
    consumer_directory = (archived_load_directory(consumer_source)
                          if consumer_source is not None else Path(__file__).parent)
    rule_sources = _sources(consumer_source)
    link = snapshot["source_snapshot"].get("main_stability_refresh", {})
    _require(link.get("schema") == SCHEMA and link.get("rule_version") == VERSION, "缺少主趋势派生链")
    receipt = _checked_json(_inside(directory, link["receipt_file"]), link["receipt_sha256"])
    _require(receipt.get("schema") == SCHEMA and receipt.get("rule_sources") == rule_sources
             and receipt.get("policy") == POLICY, "派生规则/消费者源码改变")
    parent_ref = receipt["parent_snapshot"]
    parent_dir = resolver.resolve(parent_ref["directory"])
    _checked_json(parent_dir / "snapshot.json", parent_ref["sha256"])
    parent = _snapshot(parent_dir / "snapshot.json", parent_ref["snapshot_id"])
    _require(parent.get("synthetic") is False, "主趋势派生链不能使用合成父快照")
    evidence = parent["source_snapshot"]["base_validation"]
    producer_report = _embedded(consumer_report.get("main_stability_derivation", {}).get("producer_report"), "历史标签producer")
    _require(digest(producer_report) == digest(_checked_json(_inside(parent_dir, evidence["report_path"]), evidence["report_sha256"])),
             "历史标签producer报告与父快照不一致")
    _require(consumer_report.get("production_source_role") == ROLE, "派生报告未区分消费者与历史producer")
    expected_report = deepcopy(producer_report)
    expected_report.update(production_source_sha256={n: file_hash(consumer_directory / n) for n in LABEL_SOURCES},
                           production_source_role=ROLE, main_stability_derivation=consumer_report["main_stability_derivation"])
    _require(consumer_report == expected_report, "消费者报告改变了历史协议的非允许字段")
    derivation = consumer_report["main_stability_derivation"]
    _require(derivation.get("schema") == SCHEMA and derivation.get("receipt_sha256") == link["receipt_sha256"]
             and derivation.get("parent_snapshot") == parent_ref and derivation.get("rule_sources") == rule_sources,
             "消费者报告与派生回执不一致")
    for name, expected in producer_report["production_source_sha256"].items():
        path = _inside(directory, f"parent_producer/{name}")
        _require(path.is_file() and file_hash(path) == expected, f"历史producer源码改变：{name}")
    _require(receipt["decision"] == derivation["decision"], "预先固定规则决策不一致")
    _embedded(receipt["decision"], "主趋势筛选决策")
    original_source = deepcopy(parent["source_snapshot"])
    current_source = deepcopy(snapshot["source_snapshot"])
    current_source.pop("main_stability_refresh", None)
    current_source["base_validation"] = original_source["base_validation"]
    _require(current_source == original_source, "刷新改变了清单成员、工艺、质量或原始来源")
    previous = {(r["kind"], r["metadata"]["record_id"]): r for r in parent["records"]}
    current = {(r["kind"], r["metadata"]["record_id"]): r for r in snapshot["records"]}
    _require(len(previous) == len(parent["records"]) and set(previous) == set(current)
             and len(current) == len(snapshot["records"]), "派生快照成员改变或重复")
    rows = {r["record_id"]: r for r in receipt["records"]}
    _require(set(rows) == {rid for kind, rid in previous if kind == "load"}, "逐记录刷新回执不齐全")
    for key, old_item in previous.items():
        if key[0] == "feed":
            _require(current[key] == old_item, "进给CSV、metadata或dtype登记改变")
            _require(file_hash(_inside(directory, old_item["file"])) == old_item["sha256"], "复制进给CSV改变")
    for sample in samples:
        rid = sample.metadata["record_id"]
        old_item, row = previous["load", rid], rows[rid]
        old = _frame(parent_dir, old_item)
        _require(row["parent_csv_sha256"] == old_item["sha256"], "逐记录父CSV身份改变")
        preserved = [c for c in old if c not in CHANGED_COLUMNS]
        _require(row["unchanged_columns"] == preserved, "允许变动列扩大")
        added_order = [c for c in SPATIAL_COLUMNS if c != "spindle_rate_support_valid"] + ["spindle_rate_support_valid"]
        _require(list(sample.frame.columns) == list(old.columns) + added_order, "刷新添加了未声明的列")
        pd.testing.assert_frame_equal(old[preserved], sample.frame[preserved], check_exact=True)
        _require(frame_digest(old, preserved) == row["unchanged_frame_sha256"], "逐列保留证明不符")
        old_meta = deepcopy(sample.metadata)
        for name in ADDED_METADATA:
            old_meta.pop(name, None)
        _require(old_meta == old_item["metadata"], "刷新改变了历史标签或非允许元数据")
        _require(sample.metadata["main_stability_refresh"] == row, "样本与逐记录派生回执不一致")
        _require(row["before_main"] == int(old.valid_main.sum()) and row["after_main"] == int(sample.frame.valid_main.sum()),
                 "主趋势监督计数过期")
        audit = row["raw_audit"]
        _require(file_hash(_inside(directory, audit["file"])) == audit["sha256"], "原始稳速中间量改变")
        verify_stability_frame(sample)
        verify_prepared_hybrid(sample, evidence, producer_report["records"][rid])
    return dict(valid=True, producer_evidence=evidence, producer_report=producer_report, receipt=receipt,
                historical_consumer_source=str(consumer_source) if consumer_source is not None else None,
                verified_rule_sources=rule_sources, artifact_relocations=resolver.reference)


def refresh_main_stability(service, *, parent_service_output, parent_snapshot_id, producer_source,
                           decision_path, training_request, artifact_relocations=None, execute=False,
                           cancel=None, progress=None):
    """Create a derived snapshot and pending request; never launch or issue ready."""
    if not execute:
        return dict(status="planned_not_executed", rule_version=VERSION, policy=POLICY,
                    stages=["verify_parent", "refresh_main_only", "verify_preserved_columns", "publish_pending"])
    resolver = ArtifactResolver(artifact_relocations)
    parent_output = resolver.resolve(parent_service_output, base_dir=Path.cwd())
    parent_dir = parent_output / "snapshots" / parent_snapshot_id
    producer_source = resolver.resolve(producer_source, base_dir=Path.cwd())
    decision_path = resolver.resolve(decision_path, base_dir=Path.cwd())
    _require(service.output_dir.resolve() != parent_output, "刷新必须使用独立目标服务")
    parent = _snapshot(parent_dir / "snapshot.json", parent_snapshot_id)
    _require(parent.get("protocol") == service.protocol(), "父快照协议与统一服务不一致")
    _require(parent.get("synthetic") is False and parent["source_snapshot"].get("files_verified") is True,
             "只能刷新真实且已核验的父快照")
    _require(len(parent["records"]) == 900 and Counter(r["kind"] for r in parent["records"]) == {"feed": 450, "load": 450},
             "正式刷新需要450进给与450负载组件")
    _require(not parent["source_snapshot"].get("main_stability_refresh"), "不能重复刷新已有派生快照")
    parent_evidence = parent["source_snapshot"]["base_validation"]
    _require(parent_evidence.get("mode") == MODE, "本次刷新只接受冻结混合标签")
    producer_report_path = _inside(parent_dir, parent_evidence["report_path"])
    producer_report = _checked_json(producer_report_path, parent_evidence["report_sha256"])
    for name in LABEL_SOURCES:
        _require(file_hash(archived_load_directory(producer_source) / name)
                 == producer_report["production_source_sha256"][name], "原标签producer源码身份不符")
    alignment_contract = verify_parent_alignment_defaults(producer_source)
    decision = embedded_json_artifact(decision_path)
    decision_value = _embedded(decision, "预先固定规则")
    _require(decision_value.get("status") == "fixed_before_r6_real_data_recomputation", "筛选规则未预先固定")
    _require(training_request.get("snapshot_id") == parent_snapshot_id and training_request.get("mode") == "all",
             "待训练请求不属于父快照")
    operation = service.output_dir / "operations" / ("main_stability_" + uuid.uuid4().hex)
    staging = service.output_dir / "snapshots" / ("refresh_" + uuid.uuid4().hex)
    operation.mkdir(parents=True)
    staging.mkdir(parents=True)
    save_json(operation / "state.json", dict(status="running", stage="main_stability_refresh", rule_version=VERSION))
    rule_sources = _sources()
    try:
        producer_dir = staging / "parent_producer"
        producer_dir.mkdir()
        for name in LABEL_SOURCES:
            shutil.copyfile(archived_load_directory(producer_source) / name, producer_dir / name)
        items, receipts, dependencies = [], [], {}
        for item in parent["records"]:
            checkpoint(cancel)
            rid = item["metadata"]["record_id"]
            _require(isinstance(rid, str) and Path(rid).name == rid and rid not in (".", ".."), "不安全的记录标识")
            original = _inside(parent_dir, item["file"])
            _require(file_hash(original) == item["sha256"], "父CSV变化，刷新拒绝")
            if item["kind"] == "feed":
                shutil.copyfile(original, staging / item["file"])
                _require(file_hash(staging / item["file"]) == item["sha256"], "进给字节复制失败")
                items.append(deepcopy(item))
                continue
            old = SampleSet(_frame(parent_dir, item), deepcopy(item["metadata"]))
            verify_prepared_hybrid(old, parent_evidence, producer_report["records"][rid])
            _require(not (set(SPATIAL_COLUMNS) & set(old.frame)), "父数据已含另一稳速派生版本")
            files = old.metadata.get("resolved_files", {})
            source_record = next(r for r in parent["source_snapshot"]["records"] if r["record_id"] == rid)
            for kind in ("measurement", "nc", "process", "machine"):
                path = resolver.resolve(files.get(kind) or source_record.get("resolved_files", {}).get(kind) or source_record[kind])
                expected = old.metadata["file_sha256"][kind]
                if str(path) not in dependencies:
                    _require(file_hash(path) == expected, f"原始依赖变化：{rid}/{kind}")
                    dependencies[str(path)] = expected
                _require(dependencies[str(path)] == expected, "同路径依赖哈希冲突")
                if kind == "measurement":
                    measurement_path = path
            measurement = read_stability_measurement(measurement_path,
                sample_period_s=old.metadata.get("sample_period_s"), cancel=cancel,
                position_policy=resolve_position_policy(old.metadata))
            sample, arrays, row = refresh_record(old, measurement)
            _require(file_hash(measurement_path) == old.metadata["file_sha256"]["measurement"], "读取期间原始记录变化")
            audit_name = f"main_stability_audit/{rid}.npz"
            audit_path = staging / audit_name
            audit_path.parent.mkdir(exist_ok=True)
            np.savez_compressed(audit_path, **arrays)
            row.update(record_id=rid, parent_csv_sha256=item["sha256"],
                measurement_sha256=old.metadata["file_sha256"]["measurement"], rule_version=VERSION,
                raw_audit=dict(file=audit_name, sha256=file_hash(audit_path)))
            sample.metadata["main_stability_refresh"] = row
            sample.frame.to_csv(staging / item["file"], index=False, encoding="utf-8")
            new_item = dict(kind="load", file=item["file"], sha256=file_hash(staging / item["file"]),
                metadata=sample.metadata, dtypes={c: str(t) for c, t in sample.frame.dtypes.items()})
            restored = _frame(staging, new_item)
            pd.testing.assert_frame_equal(sample.frame, restored, check_exact=True)
            items.append(new_item)
            receipts.append(row)
            if progress:
                progress(dict(stage="main_stability_refresh", record_id=rid, completed=len(receipts), total=450))
        _require(rule_sources == _sources(), "刷新期间R6规则源码改变")
        for path, expected in dependencies.items():
            _require(file_hash(path) == expected, "刷新结束前原始依赖改变")
        parent_ref = dict(directory=str(parent_dir), snapshot_id=parent_snapshot_id, sha256=file_hash(parent_dir / "snapshot.json"))
        receipt = dict(schema=SCHEMA, policy=POLICY, rule_version=VERSION, rule_sources=rule_sources,
            parent_snapshot=parent_ref, decision=decision, records=receipts, dependencies=dependencies,
            alignment_contract=alignment_contract,
            feed_csv_copy="byte_identical_metadata_and_dtypes_identical", load_label_recomputed=False,
            residual_recomputed=False, ready_issued=False, training_started=False)
        save_json(staging / "main_stability_receipt.json", receipt)
        link = dict(schema=SCHEMA, rule_version=VERSION, receipt_file="main_stability_receipt.json",
                    receipt_sha256=file_hash(staging / "main_stability_receipt.json"))
        consumer = deepcopy(producer_report)
        consumer.update(production_source_role=ROLE,
            production_source_sha256={n: file_hash(Path(__file__).with_name(n)) for n in LABEL_SOURCES},
            main_stability_derivation=dict(schema=SCHEMA, producer_report=embedded_json_artifact(producer_report_path),
                parent_snapshot=parent_ref, rule_sources=rule_sources, receipt_sha256=link["receipt_sha256"], decision=decision))
        save_json(staging / "base_evidence/report.json", consumer)
        model_source = _inside(parent_dir, parent_evidence["model_path"])
        _require(file_hash(model_source) == parent_evidence["model_sha256"], "父冻结基础模型改变")
        shutil.copyfile(model_source, staging / "base_evidence/model.json")
        evidence = {**deepcopy(parent_evidence), "report_path": "base_evidence/report.json",
                    "report_sha256": file_hash(staging / "base_evidence/report.json"), "model_path": "base_evidence/model.json"}
        source = {**deepcopy(parent["source_snapshot"]), "base_validation": evidence, "main_stability_refresh": link}
        from .dataset_manifest import verify_base_evidence
        def validate(manifest, directory):
            checked = verify_base_evidence(evidence, [r["metadata"] for r in items if r["kind"] == "load"], base_dir=directory)
            _require(checked["valid"], "R6消费者协议门禁拒绝：" + ";".join(checked["issues"]))
            loaded = (SampleSet(_frame(directory, r), r["metadata"]) for r in items if r["kind"] == "load")
            verify_derivation(manifest, loaded, directory=directory, consumer_report=checked["report"],
                              artifact_relocations=artifact_relocations)
        saved = service._finalize_prepared_snapshot(staging, source, items, synthetic=False, validator=validate)
        destination = Path(saved["directory"])
        request = deepcopy(training_request)
        request["snapshot_id"] = saved["snapshot_id"]
        save_json(operation / "pending_train_request.json", request)
        result = dict(status="prepared_not_launched", snapshot_id=saved["snapshot_id"], directory=str(destination),
            operation=str(operation), pending_train_request=str(operation / "pending_train_request.json"),
            records=900, before_main=sum(r["before_main"] for r in receipts), after_main=sum(r["after_main"] for r in receipts),
            ready_issued=False, training_started=False, feed_model_reuse_pending=True)
        save_json(operation / "state.json", result)
        return result
    except Exception:
        save_json(operation / "state.json", dict(status="failed", traceback=traceback.format_exc(), staging=str(staging)))
        raise
