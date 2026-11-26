"""Paper-1 launch preparation; existing services own qualification and training.

The default action only reads manifest metadata. Neither preflight, activation,
nor preparation trains a network. Only explicit launch/resume starts the public
CLI, whose JobService owns checkpoints and recovery.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
for _name, _value in {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
                      "MKL_THREADING_LAYER": "SEQUENTIAL", "OMP_NUM_THREADS": "1",
                      "OPENBLAS_NUM_THREADS": "1", "MPLBACKEND": "Agg"}.items():
    os.environ[_name] = _value

from spindle_power.load_prediction.common import file_hash, json_value
from spindle_power.load_prediction.experiment_protocol import PROGRAM_FEATURES, protocol_folds, validate_round_metadata
from spindle_power.load_prediction.feed_model import FeedModelConfig
from spindle_power.load_prediction.feed_training import FeedTrainingConfig, train_feed_fold
from spindle_power.load_prediction.models import ModelConfig
from spindle_power.load_prediction.sample_rules import SAMPLE_RULES_VERSION
from spindle_power.load_prediction.training import TrainingConfig, train_two_stage
from spindle_power.load_prediction.training_checkpoint import source_fingerprints
from spindle_power.load_prediction.training_service import TrainingService, validate_paper1_fixed_conditions


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(json_value(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def load_plan(path):
    path = Path(path).resolve()
    plan = read_json(path)
    if plan.get("schema") != "paper1.launch_plan.1":
        raise ValueError("不支持的启动计划版本")
    for key in ("manifest", "candidates", "service_output", "python_executable", "base_evidence", "base_event_timelines",
                "source_feed_service_output"):
        if plan.get(key):
            value = Path(plan[key])
            plan[key] = str((value if value.is_absolute() else path.parent / value).resolve())
    if any(plan.get(key) for key in ("source_prepared_service_output", "historical_consumer_source")):
        resolver = artifact_resolver(plan)
        for key in ("source_prepared_service_output", "historical_consumer_source"):
            if plan.get(key):
                plan[key] = str(resolver.resolve(plan[key], base_dir=path.parent))
    return plan


def metadata(record):
    return {**record.get("metadata", {}), **{k: v for k, v in record.items() if k != "metadata"}}


def candidate_list(path, mode):
    candidates = deepcopy(read_json(path)["candidates"])
    if not candidates or len({item["id"] for item in candidates}) != len(candidates):
        raise ValueError("候选必须非空且ID唯一")
    seen, selected = set(), []
    for item in candidates:
        name = item["id"]
        if not isinstance(name, str) or not name or name in (".", "..") or Path(name).name != name:
            raise ValueError("候选ID必须为安全目录名")
        classes = (("load_model", ModelConfig), ("load_training", TrainingConfig)) if mode == "load" else (
            ("feed_model", FeedModelConfig), ("feed_training", FeedTrainingConfig),
            ("load_model", ModelConfig), ("load_training", TrainingConfig))
        for key, cls in classes:
            cls(**item.get(key, {})).validate()
        if mode == "load":
            from spindle_power.load_prediction.training_service import _nominal_candidates
            _nominal_candidates([item])
        if mode == "feed":
            item = {key: item[key] for key in ("id", "feed_model", "feed_training")}
            identity = json.dumps({k: v for k, v in item.items() if k != "id"}, sort_keys=True)
            if identity in seen:
                continue
            seen.add(identity)
        selected.append(item)
    return selected


def replacement_review_bound(record, base_dir, artifact_relocations=None):
    """Receipt-only check; current source bytes are checked by verify_review_sources."""
    review = record.get("review_evidence", {})
    if any(review.get("decisions", {}).get(key) is not True
           for key in ("complete", "alignment_reliable", "process_aligned")):
        return False
    from spindle_power.load_prediction.artifact_paths import ArtifactResolver
    resolver = ArtifactResolver(artifact_relocations, base_dir=base_dir)
    reports = {}
    try:
        for kind in ("numeric", "alignment", "preparation"):
            ref = review[kind]
            path = resolver.resolve(ref["path"])
            if file_hash(path) != ref["sha256"]:
                return False
            reports[kind] = read_json(path)
            if reports[kind].get("record_id") != record["record_id"]:
                return False
        measurement = record["recollection"]["measurement_sha256"]
        nc = record["recollection"]["expected_nc_sha256"]
        sources = reports["preparation"]["source_files"]
        return (len(measurement) == 64 and len(nc) == 64
                and reports["numeric"].get("file_sha256") == measurement
                and reports["alignment"].get("measurement_sha256") == measurement
                and reports["alignment"].get("nc_sha256") == nc
                and sources["measurement"]["sha256"] == measurement and sources["nc"]["sha256"] == nc)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def superseded_input_batches(plan, records):
    """Check the nearest artifact batch, never inherit a container's old marker."""
    base = Path(plan["manifest"]).resolve().parent
    resolver = artifact_resolver(plan)
    directories = {base}
    for record in records:
        for key in ("process", "segmentation_evidence_path"):
            if not record.get(key):
                continue
            path = resolver.resolve(record[key])
            if path.parent.parent.name == "combinations":
                directory = path.parents[2]
            else:
                directory = next((parent for parent in path.parents
                                  if any((parent / name).is_file() for name in ("summary.json", "segmentation_config.json"))),
                                 path.parent)
            directories.add(directory)
    return sorted(str(directory / "SUPERSEDED.json") for directory in directories
                  if (directory / "SUPERSEDED.json").is_file())


def artifact_resolver(plan):
    from spindle_power.load_prediction.artifact_paths import ArtifactResolver
    reference = plan.get("artifact_relocations")
    if reference is None:
        reference = read_json(plan["manifest"]).get("artifact_relocations")
    return ArtifactResolver(reference, base_dir=Path(plan["manifest"]).parent)


def verify_review_sources(plan):
    """Bind every active record's audit receipts to current source file bytes.

    Hash shared NC/process/config files once per invocation. Preparation calls
    this independently before and after reading samples, without reusing hashes.
    """
    base = Path(plan["manifest"]).parent
    hashes, reports = {}, {}
    resolver = artifact_resolver(plan)
    resolve = resolver.resolve

    def current_hash(value):
        path = resolve(value)
        if path not in hashes:
            before = path.stat()
            digest = file_hash(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"校验期间源文件改变：{path}")
            hashes[path] = digest
        return hashes[path]

    def receipt(ref):
        path = resolve(ref["path"])
        if current_hash(path) != ref["sha256"]:
            raise ValueError(f"审核回执已改变：{path}")
        if path not in reports:
            reports[path] = read_json(path)
        return reports[path]

    records = [metadata(record) for record in read_json(plan["manifest"]).get("records", [])]
    active = [record for record in records if record.get("status", "active") == "active"
              and record.get("dataset_role") == "development"]
    superseded = superseded_input_batches(plan, active)
    if superseded:
        raise ValueError("输入批次已标记SUPERSEDED，必须使用新审核产物：" + "；".join(superseded))
    for record in active:
        rid = record.get("record_id")
        try:
            review = record["review_evidence"]
            if any(review.get("decisions", {}).get(key) is not True
                   for key in ("complete", "alignment_reliable", "process_aligned")):
                raise ValueError("逐记录审核结论未通过")
            numeric, aligned, prepared = [receipt(review[kind]) for kind in ("numeric", "alignment", "preparation")]
            if any(report.get("record_id") != rid for report in (numeric, aligned, prepared)):
                raise ValueError("审核回执record_id不匹配")
            sources = prepared["source_files"]
            raw_process = record["source_process"]
            for field in ("measurement", "nc", "machine", "process"):
                value = raw_process["path"] if field == "process" else record[field]
                source = sources[field]
                if resolve(value) != resolve(source["path"]) or current_hash(value) != source["sha256"]:
                    raise ValueError(f"{field}当前文件与审核冻结源不一致")
            measurement, nc = current_hash(record["measurement"]), current_hash(record["nc"])
            if (numeric.get("file_sha256") != measurement
                    or aligned.get("measurement_sha256") != measurement or aligned.get("nc_sha256") != nc):
                raise ValueError("数值/对齐回执未绑定当前实测与NC")
            if record.get("replaces") and (record["recollection"]["measurement_sha256"] != measurement
                    or record["recollection"]["expected_nc_sha256"] != nc):
                raise ValueError("补采登记未绑定当前实测与NC")
            if raw_process["sha256"] != current_hash(raw_process["path"]):
                raise ValueError("原工艺来源哈希不匹配")
            segmentation = receipt({"path": record["segmentation_evidence_path"],
                                    "sha256": record["segmentation_evidence_sha256"]})
            canonical = current_hash(record["process"])
            if (record["process_version"] != canonical
                    or segmentation["output_sha256"]["canonical_process.csv"] != canonical
                    or resolve(record["process"]) != resolve(record["segmentation_evidence_path"]).parent / "canonical_process.csv"
                    or resolve(segmentation["nc_path"]) != resolve(record["nc"])
                    or segmentation["nc_sha256"] != nc
                    or resolve(segmentation["source_process_path"]) != resolve(raw_process["path"])
                    or segmentation["source_process_sha256"] != raw_process["sha256"]
                    or segmentation.get("program_version") != record["program_version"]
                    or segmentation.get("process_mapping_rules_version") != record["process_mapping_rules_version"]
                    or segmentation.get("source_points_preserved") is not True):
                raise ValueError("canonical工艺/分段证据与当前源、版本不一致")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"当前源内容校验失败：{rid}：{exc}") from exc
    binding = None
    if plan.get("process_library_binding"):
        from spindle_power.load_prediction.process_library import verify_process_library_binding
        binding = verify_process_library_binding(records, manifest_base=base, reference=plan["process_library_binding"],
                                                  artifact_relocations=resolver.reference)
    elif any(r.get("process_library_binding") for r in active):
        raise ValueError("清单含工艺库绑定但计划未绑定回执")
    return {"current_source_content_verified": True, "records_verified": len(active),
            "process_library_binding": binding,
            "unique_files_hashed": len(hashes), "raw_data_read": True,
            "verification": "sha256_current_sources_against_frozen_audit_receipts"}


def preflight(plan, mode="feed"):
    """Fast structural check, explicitly not a replacement for strict inspect."""
    manifest = read_json(plan["manifest"])
    relocation = artifact_resolver(plan).reference
    records = [metadata(r) for r in manifest.get("records", [])]
    active = [r for r in records if r.get("status", "active") == "active"
              and r.get("dataset_role") == "development"]
    blockers = []
    superseded = superseded_input_batches(plan, active)
    if superseded:
        blockers.append("输入批次已标记SUPERSEDED，必须使用新审核产物：" + "；".join(superseded))
    slots = [(r.get("round_id"), r.get("feature")) for r in active]
    expected = {(r, f) for r in range(1, 91) for f in PROGRAM_FEATURES}
    if len(slots) != 450 or set(slots) != expected:
        blockers.append("必须补齐90轮×5类的450个唯一当前有效位置")
    mapping = validate_round_metadata(active)
    blockers.extend(item["message"] for item in mapping["issues"])
    unresolved = []
    for record in active:
        failed = [key for key in ("complete", "round_complete", "alignment_reliable", "process_aligned",
                                  "no_alarm", "tool_healthy") if record.get(key) is not True]
        if failed or record.get("recollection_pending") is True:
            unresolved.append({"record_id": record.get("record_id"), "failed": failed,
                               "recollection_pending": record.get("recollection_pending", False)})
    if unresolved:
        blockers.append(f"{len(unresolved)}份记录仍有未完成的逐记录/整轮审核或补采")
    contract = plan.get("input_contract", {})
    required_metadata = contract.get("required_metadata", {})
    if (contract.get("status") != "frozen" or not required_metadata
            or any(value is None for value in required_metadata.values())):
        blockers.append("新工艺空间映射规则与清单尚未重新冻结；旧累计s定位产物不能启动")
    elif contract.get("manifest_sha256") != file_hash(plan["manifest"]):
        blockers.append("清单与新工艺映射冻结指纹不一致，必须完成该清单的重审核")
    elif required_metadata.get("sample_rules_version") != SAMPLE_RULES_VERSION:
        blockers.append("冻结的样本规则与当前源码版本不一致，必须重新准备")
    elif any(record.get(key) != value for record in active for key, value in required_metadata.items()):
        blockers.append("存在未绑定新工艺空间映射规则的有效记录")
    required = plan.get("required_replacement")
    if required:
        old = [r for r in records if r.get("record_id") == required and r.get("status") == "superseded"]
        new = [r for r in active if required in r.get("replaces", [])
               and r.get("recollection", {}).get("round_review_snapshot_id")]
        if len(old) != 1 or len(new) != 1:
            blockers.append(f"{required}必须经activate_record_replacement审核替换，不能直接把旧异常记录放行")
        elif new[0].get("replaces") != [required] or old[0].get("superseded_by") != [new[0]["record_id"]]:
            blockers.append("补采必须逐槽唯一关联原记录，不接受整轮交叉替代关系")
        elif not replacement_review_bound(new[0], Path(plan["manifest"]).parent, relocation):
            blockers.append("补采记录缺少绑定自身record_id、实测/NC哈希的独立审核证据，不能继承旧R74审核")
    if any(not record.get("recollection", {}).get("round_review_snapshot_id")
           or not replacement_review_bound(record, Path(plan["manifest"]).parent, relocation)
           for record in active if record.get("replaces")):
        blockers.append("存在未绑定自身独立审核回执的补采记录，不能冻结整轮替换")
    if any("O1203" in str(r.get(key, "")).upper() for r in active
           for key in ("record_id", "program_id", "nc", "measurement", "process")):
        blockers.append("O1203只能登记保留，不允许进入本轮开发清单")
    if mode in ("all", "load") and not plan.get("base_evidence"):
        blockers.append("负载训练仍缺所选标签模式的正式证据文件")
    candidates = candidate_list(plan["candidates"], mode)
    return {"schema": "paper1.launch_preflight.1", "mode": mode,
            "formal_training_release": False, "raw_data_read": False, "training_started": False,
            "current_source_content_verified": False,
            "structurally_ready_for_strict_prepare": not blockers,
            "active_development_records": len(active), "candidate_ids": [c["id"] for c in candidates],
            "unresolved_records": unresolved, "blockers": blockers,
            "manifest_sha256": file_hash(plan["manifest"]), "candidates_sha256": file_hash(plan["candidates"]),
            "next_required_check": "refreeze/prepare rehash current sources against audit receipts; prepare checks before and after strict service preparation, then validates immutable numerical snapshot and folds."}


def runtime_identity():
    return {"python_executable": str(Path(sys.executable).resolve()), "python_version": platform.python_version(),
            "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "torch")},
            "environment": {name: os.environ.get(name) for name in ("MKL_THREADING_LAYER", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MPLBACKEND")},
            "code": source_fingerprints(), "cli_sha256": file_hash(ROOT / "scripts/load_prediction.py"),
            "launcher_sha256": file_hash(__file__)}


def assert_runtime(plan):
    if Path(plan["python_executable"]).resolve() != Path(sys.executable).resolve():
        raise ValueError("请使用启动计划中已核验的AFC解释器；不接受conda run解析到其他环境")


def activate(plan, replacement_path, destination, *, whole_round=False):
    from spindle_power.load_prediction.dataset_manifest import activate_record_replacement, activate_round_replacement
    manifest = read_json(plan["manifest"])
    payload = read_json(replacement_path)
    base = Path(plan["manifest"]).parent
    if whole_round:
        if not isinstance(payload, dict) or not isinstance(payload.get("replacements"), list):
            raise ValueError("整轮激活输入必须包含 replacements 数组")
        updated = activate_round_replacement(manifest["records"], payload["replacements"], base_dir=base)
        new_ids = {metadata(record)["record_id"] for record in payload["replacements"]}
        if any(not replacement_review_bound(metadata(record), base) for record in updated
               if metadata(record).get("record_id") in new_ids):
            raise ValueError("整轮补采缺少绑定自身record_id、实测/NC哈希的独立审核证据")
    else:
        updated = activate_record_replacement(manifest["records"], payload, base_dir=base)
    directory = Path(destination).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    # Preserve the original path base when saving elsewhere; do not rewrite facts.
    for record in updated:
        target = record.get("metadata", record)
        for key in ("nc", "measurement", "machine", "process"):
            if target.get(key) and not Path(target[key]).is_absolute():
                target[key] = str((base / target[key]).resolve())
    output = directory / "activated_manifest.json"
    write_new(output, {**manifest, "records": updated})
    next_plan = {**plan, "manifest": str(output),
        "note": "补采激活后的清单；需确认新记录独立审核证据，然后refreeze并严格prepare。未知事实保持原值，未放行训练。",
        "input_contract": {
        **plan.get("input_contract", {}), "status": "awaiting_refreeze", "manifest_sha256": None}}
    write_new(directory / "launch_plan.json", next_plan)
    return {"manifest": str(output), "plan": str(directory / "launch_plan.json"), "training_started": False}


def refreeze(plan, destination):
    """Bind an audited replacement manifest; strict numerical qualification follows."""
    frozen = deepcopy(plan)
    frozen["input_contract"] = {**plan.get("input_contract", {}), "status": "frozen",
                                "manifest_sha256": file_hash(plan["manifest"])}
    frozen["note"] = "补采清单输入身份已冻结；通过结构/版本/审核回执与当前源内容校验，正式资格仍由严格prepare和数值预检决定。"
    mode = plan.get("mode", "feed")
    check = preflight(frozen, mode)
    if check["blockers"]:
        raise ValueError("补采清单尚不能冻结：" + "；".join(check["blockers"]))
    if mode == "load":
        check["source_verification"] = verify_nominal_parent(frozen)
        check.update(current_source_content_verified=True, raw_data_read=False,
                     source_basis="verified_immutable_parent_snapshot_and_historical_consumer")
    else:
        check["source_verification"] = verify_review_sources(frozen)
        check.update(current_source_content_verified=True, raw_data_read=True)
    if file_hash(plan["manifest"]) != frozen["input_contract"]["manifest_sha256"]:
        raise ValueError("源内容校验期间清单改变，不能冻结")
    destination = Path(destination).resolve()
    write_new(destination, frozen)
    write_new(destination.with_name(destination.stem + "_preflight.json"), check)
    return {"plan": str(destination), "manifest_sha256": frozen["input_contract"]["manifest_sha256"],
            "formal_training_release": False, "training_started": False,
            "next_required_check": "prepare runs strict full-manifest inspect and numerical fold validation"}


def bind_process_library(plan, library_manifest, process_root, destination):
    from spindle_power.load_prediction.process_library import build_process_library_binding, attach_process_binding
    manifest = read_json(plan["manifest"])
    receipt = build_process_library_binding(manifest["records"], manifest_base=Path(plan["manifest"]).parent,
        library_manifest=library_manifest, process_root=process_root)
    directory = Path(destination).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    receipt_path = directory / "process_library_binding.json"
    write_new(receipt_path, receipt)
    output = directory / "bound_manifest.json"
    write_new(output, {**manifest, "records": attach_process_binding(manifest["records"], receipt_path)})
    next_plan = {**plan, "manifest": str(output),
        "process_library_binding": {"path": str(receipt_path), "sha256": file_hash(receipt_path)},
        "input_contract": {**plan.get("input_contract", {}), "status": "awaiting_refreeze", "manifest_sha256": None},
        "note": "450个活动位置已绑定54组派生工艺；保留原来源与稳态证据，等待最终refreeze和严格prepare。"}
    write_new(directory / "launch_plan.json", next_plan)
    return {"plan": str(directory / "launch_plan.json"), "manifest": str(output),
            "binding": str(receipt_path), "records": receipt["active_records"], "training_started": False}


def baseline_candidate(request):
    if request.get("id") != "fixed_physical" or request.get("execute") is not True:
        raise ValueError("本阶段独立实验启动仅支持fixed_physical")
    model = request["model_config"]
    if model.get("physical_mode") != "fixed" or model.get("use_condition_effects") is not False:
        raise ValueError("固定物理基线只拟合两个正系数，不允许附加工况倍率")
    return dict(id="fixed_physical", feed_model=request["feed_model"], feed_training=request["feed_training"],
                load_model=model, load_training=request["training_config"])


def verify_baseline_feed_cache(service, request, fold, train, validation, *, feed_source=None):
    """Require the selected fold's already frozen BiLSTM, never silently refit it."""
    from spindle_power.load_prediction.training_checkpoint import digest
    from spindle_power.load_prediction.feed_identity import feed_training_identity
    from spindle_power.load_prediction.feed_model import load_feed_model
    source = service.output_dir / "runs" / request["source_development_job_id"] / "development.json"
    if file_hash(source) != request["source_development_sha256"]:
        raise ValueError("来源九折选型结果已改变")
    development = read_json(source)
    selected = next(item for item in development["candidates"] if item["candidate"] == development["selected"])
    chosen_fold = next(item for item in selected["folds"] if item["fold"] == fold["fold"])
    if chosen_fold["validation_rounds"] != fold["validation_rounds"]:
        raise ValueError("来源选型结果的留出轮次不符合当前九折协议")
    if feed_source is not None:
        from spindle_power.load_prediction.training_service import _load_feed_reference
        reference = feed_source["folds"][fold["fold"]]
        model = _load_feed_reference(reference, train, validation,
            FeedTrainingConfig(**request["feed_training"]), FeedModelConfig(**request["feed_model"]),
            artifact_relocations=request.get("artifact_relocations"))
        if model.model_version != chosen_fold["feed_metrics"]["model_version"]:
            raise ValueError("来源BiLSTM与获选配置该折使用的模型版本不一致")
        return {"reference": reference, "model_version": model.model_version,
                "source_feed_job_id": feed_source["source_feed_job_id"]}
    identity = feed_training_identity(train, validation,
        training_config=FeedTrainingConfig(**request["feed_training"]),
        model_config=FeedModelConfig(**request["feed_model"]), full=False)
    directory = service.output_dir / "feed_cache" / "feed_reuse_v1" / digest(identity)
    audit = read_json(directory / "model.json")
    weights = directory / "feed.pt"
    if audit.get("identity") != json_value(identity) or audit.get("sha256") != file_hash(weights):
        raise ValueError("固定基线需要同一折、同一快照和配置的完整冻结进给缓存")
    model = load_feed_model(weights)
    if model.model_version != chosen_fold["feed_metrics"]["model_version"]:
        raise ValueError("缓存BiLSTM与获选配置该折使用的模型版本不一致")
    return {"directory": str(directory), "model_version": model.model_version, "sha256": file_hash(weights)}


def validate_numerical(service, request, required_metadata=None):
    """Call existing dry-run validators; no optimizer is ever invoked here."""
    rejected = service.output_dir / "snapshots" / request["snapshot_id"] / "launcher_rejected.json"
    if rejected.exists():
        raise ValueError("该快照在准备后源内容校验中被拒绝，不能启动；请重新审核和准备")
    if request["mode"] == "load":
        return validate_nominal_numerical(service, request, required_metadata)
    baseline = request["mode"] == "experiment"
    if baseline:
        candidates = [baseline_candidate(request)]
        preview = service.experiment({**request, "execute": False})
        if preview.get("status") != "prepared_not_executed":
            raise ValueError(f"固定物理基线预检未通过：{preview}")
    else:
        candidates = request["candidates"]
        service.preview(request["snapshot_id"], candidates, request["mode"])
    feeds, loads, snapshot = service.load_prepared(request["snapshot_id"])
    feed_source = None
    if request.get("source_feed_job_id"):
        feed_source = service.validate_feed_source_job(request["source_feed_job_id"], request["snapshot_id"], candidates,
            source_feed_service_output=request.get("source_feed_service_output"),
            artifact_relocations=request.get("artifact_relocations"))
    if snapshot.get("synthetic") is not False:
        raise ValueError("论文1启动包只接受真实数据快照")
    expected = {**(required_metadata or {}), "sample_rules_version": SAMPLE_RULES_VERSION}
    relevant = feeds + (loads if request["mode"] != "feed" else [])
    if any(sample.metadata.get(key) != value for sample in relevant for key, value in expected.items()):
        raise ValueError("准备快照未绑定当前样本/工艺映射规则；禁止静默兼容旧产物")
    conditions = validate_paper1_fixed_conditions(loads or feeds)
    source_records = {r["record_id"]: r for r in snapshot["source_snapshot"]["records"]
                      if r.get("status") == "active" and r.get("dataset_role") == "development"}
    if len(feeds) != 450 or {s.metadata["record_id"] for s in feeds} != set(source_records):
        raise ValueError("进给数值快照必须对应450个唯一活动记录")
    for sample in feeds:
        meta = sample.metadata
        if meta.get("measured_feed_source") != "actual_axis_velocity_norm":
            raise ValueError("本阶段实际进给标签必须来自实测XYZ轴速度范数")
        if meta.get("process_library_binding") != source_records[meta["record_id"]].get("process_library_binding"):
            raise ValueError("进给快照未保留源清单工艺绑定")
    feed_summary = dict(records=len(feeds), nc_grid_points=sum(len(s.frame) for s in feeds),
        valid_label_points=sum(int(s.frame.valid_label.fillna(False).sum()) for s in feeds),
        label_sources=dict(Counter(s.metadata.get("measured_feed_source") for s in feeds)),
        position_sources=dict(Counter(s.metadata.get("position_source") for s in feeds)),
        process_binding_preserved=True)
    summaries = []
    for candidate in candidates:
        for fold in protocol_folds():
            selected = lambda data, key: [s for s in data if s.metadata["round_id"] in fold[key]]
            check = train_feed_fold(selected(feeds, "train_rounds"), selected(feeds, "validation_rounds"),
                                    config=FeedTrainingConfig(**candidate["feed_training"]),
                                    model_config=FeedModelConfig(**candidate["feed_model"]), execute=False)
            if not check["ready_to_train"]:
                raise ValueError(f"{fold['fold']}进给监督不合格：{check['blockers']}")
            summary = {"candidate": candidate["id"], "fold": fold["fold"], "feed_ready": True}
            if baseline:
                summary["reused_feed"] = verify_baseline_feed_cache(service, request, fold,
                    selected(feeds, "train_rounds"), selected(feeds, "validation_rounds"), feed_source=feed_source)
            if request["mode"] != "feed":
                for group in ("train_rounds", "validation_rounds"):
                    check_load = train_two_stage(selected(loads, group),
                        config=TrainingConfig(**candidate["load_training"]),
                        model_config=ModelConfig(**candidate["load_model"]), execute=False, main_only=baseline)
                    if not check_load["main_samples"] or not check_load["residual_samples"]:
                        raise ValueError(f"{fold['fold']} {group}缺少合格主趋势或残差标签")
                    summary[group] = {k: check_load[k] for k in ("main_samples", "residual_samples")}
            summaries.append(summary)
    return {"trained": False, "fold_checks": summaries, "snapshot_id": request["snapshot_id"],
            "fixed_condition_check": conditions, "feed_summary": feed_summary}


def verify_nominal_parent(plan):
    """Check the immutable preparation instead of reparsing historical measurements."""
    resolver = artifact_resolver(plan)
    parent_service = TrainingService(resolver.resolve(plan["source_prepared_service_output"]))
    feeds, loads, snapshot = parent_service.load_prepared(plan["source_prepared_snapshot_id"])
    parent_service._validate_prepared(feeds, loads, snapshot, mode="all",
        historical_consumer_source=str(resolver.resolve(plan["historical_consumer_source"])),
        artifact_relocations=resolver.reference)
    records = {r["record_id"]: r for r in snapshot["source_snapshot"]["records"] if r.get("status") == "active"}
    current = [metadata(r) for r in read_json(plan["manifest"])["records"] if metadata(r).get("status", "active") == "active"]
    if len(current) != 450 or {r["record_id"] for r in current} != set(records):
        raise ValueError("当前清单与父快照成员不一致")
    for row in current:
        parent = records[row["record_id"]]
        for key in ("round_id", "feature", "parameter_group", "program_id", "process_library_binding",
                    "complete", "round_complete", "alignment_reliable", "process_aligned", "no_alarm", "tool_healthy"):
            if row.get(key) != parent.get(key):
                raise ValueError(f"当前清单改变父快照已审核字段：{row['record_id']} {key}")
        for key in ("nc", "measurement", "machine", "process"):
            # Immutable source names remain provenance on a remote machine;
            # equal original names do not require transferring raw sampling.
            original = parent.get(key, parent.get("resolved_files", {}).get(key))
            if row[key] != original and resolver.resolve(row[key]) != resolver.resolve(original):
                raise ValueError(f"当前清单改变父来源路径：{row['record_id']} {key}")
    return dict(parent_snapshot_id=snapshot["snapshot_id"], records_verified=len(current), raw_data_read=False,
                parent_snapshot_sha256=file_hash(parent_service.output_dir / "snapshots" / snapshot["snapshot_id"] / "snapshot.json"))


def validate_nominal_numerical(service, request, required_metadata=None):
    from spindle_power.load_prediction.training_service import _nominal_candidates
    from spindle_power.load_prediction.samples import load_input_view, path_features
    candidates = _nominal_candidates(request["candidates"])
    if request.get("source_feed_job_id") or any(c.get("load_training", {}).get("device") not in ("cpu", "cuda") for c in candidates):
        raise ValueError("正式名义负载请求必须无feed来源并显式使用cpu或cuda")
    feeds, loads, snapshot = service.load_prepared(request["snapshot_id"])
    service._validate_prepared(feeds, loads, snapshot, mode="load")
    if snapshot.get("synthetic") is not False:
        raise ValueError("论文1启动包只接受真实数据快照")
    expected = {**(required_metadata or {}), "sample_rules_version": SAMPLE_RULES_VERSION}
    if any(s.metadata.get(k) != v for s in loads for k, v in expected.items()):
        raise ValueError("新负载快照未绑定required_metadata")
    import numpy as np
    for sample in loads:
        view = load_input_view(sample, "programmed")
        good = view.frame.valid_residual.to_numpy(bool)
        path_features(view.frame.loc[good], np.zeros(int(good.sum())), 0., 1., 2000.,
                      route="nominal_programmed_residual_v1")
    summaries = []
    for candidate in candidates:
        for fold in protocol_folds():
            row = dict(candidate=candidate["id"], fold=fold["fold"], feed_required=False)
            for group in ("train_rounds", "validation_rounds"):
                subset = [s for s in loads if s.metadata["round_id"] in fold[group]]
                checked = train_two_stage(subset, model_config=ModelConfig(**candidate["load_model"]),
                    config=TrainingConfig(**candidate["load_training"]), execute=False)
                if not checked["main_samples"] or not checked["residual_samples"]:
                    raise ValueError(f"{fold['fold']} {group}缺少监督")
                row[group] = {k: checked[k] for k in ("main_samples", "residual_samples")}
            summaries.append(row)
    return dict(trained=False, snapshot_id=request["snapshot_id"], fold_checks=summaries,
                fixed_condition_check=validate_paper1_fixed_conditions(loads), feed_network_used=False,
                inherited_feed_records=len(feeds), load_records=len(loads))


def save_ready(service, request, destination, *, provenance=None, required_metadata=None):
    before = runtime_identity()
    execution = service._run_identity(request) if request.get("mode") == "load" else None
    numerical = validate_numerical(service, request, required_metadata)
    if runtime_identity() != before or (execution is not None and service._run_identity(request) != execution):
        raise ValueError("预检期间源码或运行环境变化，必须重新准备")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    request_path = destination / "train_request.json"
    write_new(request_path, request)
    write_new(destination / "numerical_preflight.json", numerical)
    ready = {"schema": "paper1.ready_request.1", "request": str(request_path),
             "request_sha256": file_hash(request_path), "service_output": str(service.output_dir.resolve()),
             "runtime": before, "provenance": provenance or {}, "required_metadata": required_metadata or {},
             "training_started": False,
             "created_at": datetime.now(timezone.utc).isoformat()}
    if execution is not None:
        ready["run_identity"] = execution
    write_new(destination / "ready.json", ready)
    return {"ready": str(destination / "ready.json"), "snapshot_id": request["snapshot_id"], "training_started": False}


def prepare(plan, mode, destination, *, source_feed_job_id=None, source_feed_service_output=None,
            source_prepared_service_output=None, source_prepared_snapshot_id=None):
    assert_runtime(plan)
    if mode == "load":
        if source_feed_job_id or source_feed_service_output:
            raise ValueError("名义负载任务不接受feed来源")
        selected = {**plan}
        if source_prepared_service_output is not None:
            selected["source_prepared_service_output"] = str(source_prepared_service_output)
        if source_prepared_snapshot_id is not None:
            selected["source_prepared_snapshot_id"] = source_prepared_snapshot_id
        check = preflight(selected, mode)
        if check["blockers"]:
            raise ValueError("；".join(check["blockers"]))
        check["parent_verification"] = verify_nominal_parent(selected)
        before = runtime_identity()
        service = TrainingService(plan["service_output"])
        prepared = service.refresh_load_route(parent_service_output=selected["source_prepared_service_output"],
            parent_snapshot_id=selected["source_prepared_snapshot_id"],
            historical_consumer_source=selected["historical_consumer_source"],
            artifact_relocations=artifact_resolver(selected).reference, execute=True)
        if (before != runtime_identity() or file_hash(plan["manifest"]) != check["manifest_sha256"]
                or file_hash(plan["candidates"]) != check["candidates_sha256"]):
            raise ValueError("新路线准备期间源码、环境、清单或配置改变，不能签发ready")
        request = dict(mode="load", snapshot_id=prepared["snapshot_id"], candidates=candidate_list(plan["candidates"], mode),
                       analysis_mode="numbered_round_pooled_tool", analysis_tool_group="200C2_D10_Z3_pooled")
        return save_ready(service, request, destination, provenance=check,
                          required_metadata=plan["input_contract"]["required_metadata"])
    if mode == "feed" and (source_feed_job_id or source_feed_service_output):
        raise ValueError("source_feed_job_id只适用于all复用，不用于独立feed重训")
    if source_feed_service_output and not source_feed_job_id:
        raise ValueError("指定来源服务目录必须同时指定source_feed_job_id")
    if bool(source_prepared_service_output) != bool(source_prepared_snapshot_id):
        raise ValueError("增量准备必须同时指定来源服务目录和快照ID")
    if mode == "feed" and source_prepared_snapshot_id:
        raise ValueError("独立feed准备不接受负载标签增量来源")
    check = preflight(plan, mode)
    if check["blockers"]:
        raise ValueError("；".join(check["blockers"]))
    check["source_verification_before_prepare"] = verify_review_sources(plan)
    service = TrainingService(plan["service_output"])
    before = runtime_identity()
    if mode == "feed":
        prepared = service.prepare_feed(plan["manifest"])
    else:
        evidence = read_json(plan["base_evidence"])
        source_prepared = (dict(source_prepared_service_output=str(Path(source_prepared_service_output).resolve()),
                                source_prepared_snapshot_id=source_prepared_snapshot_id)
                           if source_prepared_snapshot_id else {})
        prepared = service.prepare(plan["manifest"], base_model_path=evidence.get("model_path"), base_validation=evidence,
                                   base_event_timelines=read_json(plan["base_event_timelines"]) if plan.get("base_event_timelines") else None,
                                   **source_prepared)
    try:
        check["source_verification_after_prepare"] = verify_review_sources(plan)
        check.update(current_source_content_verified=True, raw_data_read=True)
        if (before != runtime_identity() or file_hash(plan["manifest"]) != check["manifest_sha256"]
                or file_hash(plan["candidates"]) != check["candidates_sha256"]):
            raise ValueError("准备过程中源码、环境、清单或配置改变；快照不能直接启动，请重新准备")
    except (OSError, ValueError) as exc:
        rejected = service.output_dir / "snapshots" / prepared["snapshot_id"] / "launcher_rejected.json"
        if not rejected.exists():
            write_new(rejected, {"snapshot_id": prepared["snapshot_id"], "training_started": False,
                "reason": str(exc), "created_at": datetime.now(timezone.utc).isoformat()})
        raise ValueError(f"准备后校验拒绝该快照，保留产物但不生成ready：{exc}") from exc
    request = {"snapshot_id": prepared["snapshot_id"], "mode": mode, "candidates": candidate_list(plan["candidates"], mode)}
    if mode == "all":
        request.update(analysis_mode="numbered_round_pooled_tool", analysis_tool_group="200C2_D10_Z3_pooled")
        if source_feed_job_id:
            request["source_feed_job_id"] = source_feed_job_id
            if source_feed_service_output:
                request["source_feed_service_output"] = str(Path(source_feed_service_output).resolve())
        if plan.get("artifact_relocations"):
            request["artifact_relocations"] = plan["artifact_relocations"]
    return save_ready(service, request, destination, provenance=check,
                      required_metadata=plan["input_contract"]["required_metadata"])


def prepare_baseline(plan, source_job_id, destination):
    assert_runtime(plan)
    if not source_job_id.isalnum():
        raise ValueError("来源任务ID不合法")
    service = TrainingService(plan["service_output"])
    job_dir = service.output_dir / "jobs" / source_job_id
    state, source_request = read_json(job_dir / "state.json"), read_json(job_dir / "request.json")
    if state.get("status") != "succeeded" or state.get("result", {}).get("status") != "frozen" or source_request.get("mode", "all") != "all":
        raise ValueError("固定基线须等待完整all任务成功并取得已完成九折选型结果")
    run = service.output_dir / "runs" / source_job_id
    if read_json(run / "run_fingerprint.json") != json_value(service._run_identity(source_request)):
        raise ValueError("来源训练与当前源码/环境不一致，不能作为同条件固定基线")
    development_path = run / "development.json"
    development = read_json(development_path)
    chosen = development["selected"]
    if development.get("status") != "developed" or chosen not in source_request["candidates"]:
        raise ValueError("来源选型结果不完整或未绑定原候选配置")
    model = {**chosen["load_model"], "physical_mode": "fixed", "use_condition_effects": False,
             "use_residual_modulation": False}
    ModelConfig(**model).validate()
    request = dict(mode="experiment", id="fixed_physical", execute=True,
        snapshot_id=source_request["snapshot_id"], model_config=model, training_config=chosen["load_training"],
        feed_model=chosen["feed_model"], feed_training=chosen["feed_training"],
        source_development_job_id=source_job_id, source_development_sha256=file_hash(development_path))
    request.update({key: source_request[key] for key in
        ("source_feed_job_id", "source_feed_service_output", "artifact_relocations") if key in source_request})
    return save_ready(service, request, destination,
        provenance={"source_job_id": source_job_id, "selection": chosen["id"],
                    "comparison": "same_nine_folds_same_scored_positions_predicted_feed_no_memory",
                    "residual_training": False, "prediction_interval": None},
        required_metadata=plan.get("input_contract", {}).get("required_metadata"))


def cli_command(service_output, request_path, action="train"):
    return [sys.executable, "-u", str(ROOT / "scripts/load_prediction.py"), "--output-dir",
            str(Path(service_output).resolve()), action, str(Path(request_path).resolve())]


def launch(ready_path):
    ready = read_json(ready_path)
    if ready.get("schema") != "paper1.ready_request.1" or ready.get("runtime") != runtime_identity():
        raise ValueError("启动凭据缺失或源码/解释器/运行环境变化；重新准备，不能修改凭据绕过")
    if file_hash(ready["request"]) != ready["request_sha256"]:
        raise ValueError("训练请求已改变，必须重新生成并预检")
    request = read_json(ready["request"])
    service = TrainingService(ready["service_output"])
    if request.get("mode") == "load" and ready.get("run_identity") != json_value(service._run_identity(request)):
        raise ValueError("负载执行身份改变：设备、恢复策略或运行依赖不同，必须在目标环境重新准备")
    service.validate_recovery(request)
    validate_numerical(service, request, ready.get("required_metadata"))
    # Exclusive claim prevents accidental double-click/duplicate training.
    claim = Path(ready_path).with_name("launch_claim.json")
    write_new(claim, {"time": datetime.now(timezone.utc).isoformat(), "request_sha256": ready["request_sha256"]})
    directory = Path(ready_path).parent / "background"
    directory.mkdir(exist_ok=False)
    command = cli_command(ready["service_output"], ready["request"],
                          "experiment" if request["mode"] == "experiment" else "train")
    with (directory / "stdout.log").open("xb") as stdout, (directory / "stderr.log").open("xb") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=stderr, close_fds=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    receipt = {"pid": process.pid, "command": command, "service_output": ready["service_output"],
               "stdout": str(directory / "stdout.log"), "stderr": str(directory / "stderr.log"),
               "status": "cli_started_check_job_state_for_outcome", "request": ready["request"]}
    write_new(directory / "launch.json", receipt)
    return receipt


def resume(plan, job_id, destination):
    assert_runtime(plan)
    if not job_id.isalnum():
        raise ValueError("任务ID不合法")
    service = TrainingService(plan["service_output"])
    original = read_json(service.output_dir / "jobs" / job_id / "request.json")
    request = {k: v for k, v in original.items() if k not in ("job_id", "service_output", "resume_from_job_id")}
    request["resume_from_job_id"] = job_id
    service.validate_recovery(request)
    # A new explicit launch is required; the source run is never modified.
    return save_ready(service, request, destination, provenance={"resume_from_job_id": job_id},
                      required_metadata=plan.get("input_contract", {}).get("required_metadata"))


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    sub = parser.add_subparsers(dest="action")
    p = sub.add_parser("preflight")
    p.add_argument("--mode", choices=("feed", "all", "load"), default="feed")
    p.add_argument("--output", type=Path)
    p = sub.add_parser("activate")
    p.add_argument("--replacement", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    p = sub.add_parser("activate-round")
    p.add_argument("--replacements", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--mode", choices=("feed", "all", "load"), required=True)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--source-feed-job-id")
    p.add_argument("--source-feed-service-output", type=Path,
                   help="已完成独立feed任务的只读服务目录；须与--source-feed-job-id一起使用")
    p.add_argument("--source-prepared-service-output", type=Path,
                   help="已完成且只读的数值快照服务目录，用于共享服务的增量标签准备")
    p.add_argument("--source-prepared-snapshot-id")
    p = sub.add_parser("bind-process-library")
    p.add_argument("--library-manifest", type=Path, required=True)
    p.add_argument("--process-root", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    p = sub.add_parser("refreeze")
    p.add_argument("--destination", type=Path, required=True, help="新的启动计划JSON路径，不覆盖原计划")
    p = sub.add_parser("launch")
    p.add_argument("--ready", type=Path, required=True)
    p = sub.add_parser("prepare-baseline")
    p.add_argument("--source-job-id", required=True)
    p.add_argument("--destination", type=Path, required=True)
    p = sub.add_parser("resume")
    p.add_argument("--job-id", required=True)
    p.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = load_plan(args.plan)
    if args.action in (None, "preflight"):
        result = preflight(plan, getattr(args, "mode", "feed"))
        if getattr(args, "output", None):
            write_new(args.output, result)
    elif args.action == "activate":
        result = activate(plan, args.replacement, args.destination)
    elif args.action == "activate-round":
        result = activate(plan, args.replacements, args.destination, whole_round=True)
    elif args.action == "prepare":
        result = prepare(plan, args.mode, args.destination, source_feed_job_id=args.source_feed_job_id,
                         source_feed_service_output=args.source_feed_service_output,
                         source_prepared_service_output=args.source_prepared_service_output,
                         source_prepared_snapshot_id=args.source_prepared_snapshot_id)
    elif args.action == "bind-process-library":
        result = bind_process_library(plan, args.library_manifest, args.process_root, args.destination)
    elif args.action == "refreeze":
        result = refreeze(plan, args.destination)
    elif args.action == "launch":
        assert_runtime(plan)
        result = launch(args.ready)
    elif args.action == "prepare-baseline":
        result = prepare_baseline(plan, args.source_job_id, args.destination)
    else:
        result = resume(plan, args.job_id, args.destination)
    print(json.dumps(json_value(result), ensure_ascii=False, indent=2, allow_nan=False))
    return 2 if result.get("blockers") else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(json.dumps({"status": "blocked", "training_started": False, "reason": str(exc)}, ensure_ascii=False))
        raise SystemExit(2)
