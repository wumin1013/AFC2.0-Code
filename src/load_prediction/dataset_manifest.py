"""Read-only manifest qualification and immutable replacement provenance.

This service never trains a network or upgrades an unknown review fact to true.
Prepared numerical samples are frozen separately by the training service.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from .common import file_hash, json_value, version_of
from .experiment_protocol import (DATASET_ROLES, DEVELOPMENT_ROUNDS, PROGRAM_FEATURES,
                                  PROTOCOL_VERSION, SPACING_MM, protocol_folds, validate_round_metadata)

MANIFEST_SCHEMA = "afc.dataset_manifest.1"
SNAPSHOT_SCHEMA = "afc.dataset_snapshot.1"


def _digest(value):
    return hashlib.sha256(json.dumps(json_value(value), ensure_ascii=False, sort_keys=True,
                                    allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def measurement_content_hash(path):
    """Ignore Scope header timestamps when identifying copied sampled content."""
    path = Path(path)
    if path.suffix.lower() not in (".csv", ".txt"):
        return file_hash(path)
    from .measurement import detect_encoding
    scope = hashlib.sha256()
    count = 0
    with path.open(encoding=detect_encoding(path), errors="strict") as stream:
        for line in stream:
            if line.lstrip().startswith("<ChannelData>"):
                fields = [part.strip() for part in line.strip().rstrip(",").split(",")]
                normalized = []
                for field in fields[1:]:
                    try:
                        normalized.append(format(float(field), ".17g"))
                    except ValueError:
                        normalized.append(field)
                scope.update((",".join(normalized) + "\n").encode("utf-8"))
                count += 1
    return scope.hexdigest() if count else file_hash(path)


def _metadata(record):
    return {**record.get("metadata", {}), **{k: v for k, v in record.items() if k != "metadata"}}


def verify_base_evidence(evidence, records=(), *, base_dir=None,
                         historical_consumer_source=None, artifact_relocations=None):
    """Shared label qualification; retains the public manifest interface."""
    from .base_qualification import verify_label_evidence
    return verify_label_evidence(evidence, records, base_dir=base_dir,
        historical_consumer_source=historical_consumer_source, artifact_relocations=artifact_relocations)


def inspect_manifest(manifest_or_records, *, base_dir=None, verify_files=True,
                     base_validation=None, require_complete=True, selected_role="development",
                     artifact_relocations=None):
    """Return JSON-safe qualification and a fingerprinted fixed input snapshot.

    Blocker ``scope`` is feed/load/common/memory or reserved. The caller
    must never derive eligibility from only the absence of frontend warnings.
    """
    if isinstance(manifest_or_records, (str, Path)):
        source = Path(manifest_or_records).resolve()
        manifest = json.loads(source.read_text(encoding="utf-8-sig"))
        base = source.parent
    else:
        manifest = (deepcopy(manifest_or_records) if isinstance(manifest_or_records, dict)
                    else {"records": deepcopy(list(manifest_or_records))})
        configured_base = base_dir or manifest.get("base_dir")
        base = Path(configured_base).resolve() if configured_base else None
    if selected_role not in DATASET_ROLES:
        raise ValueError("未登记的清单准备用途")
    from .artifact_paths import ArtifactResolver
    resolver = ArtifactResolver(artifact_relocations or manifest.get("artifact_relocations"), base_dir=base)
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("数据清单 records 必须为列表")
    blockers, normalized, identities, role_programs, slots = [], [], {}, {}, {}
    record_ids, geometry_cache, version_roles = set(), {}, {}

    def block(code, message, record=None, scope="common"):
        record = record or {}
        blockers.append({"code": code, "message": message, "record_id": record.get("record_id"),
                         "round_id": record.get("round_id"), "feature": record.get("feature"), "scope": scope})

    if manifest.get("schema") not in (None, MANIFEST_SCHEMA, "afc.feed_manifest.1"):
        block("manifest_schema", "不支持的数据清单版本")
    for index, raw in enumerate(records):
        if not isinstance(raw, dict):
            block("invalid_record", f"第{index + 1}条记录不是对象")
            continue
        meta = _metadata(raw)
        meta.setdefault("record_id", _digest({"index": index, "record": raw})[:20])
        meta.setdefault("status", "active")
        if meta["status"] not in ("active", "superseded"):
            block("record_status", "记录状态必须为 active 或 superseded", meta)
        if meta["status"] == "superseded":
            normalized.append(meta)
            continue
        if meta["record_id"] in record_ids:
            block("duplicate_record_id", "有效记录 record_id 重复", meta)
        record_ids.add(meta["record_id"])
        role = meta.get("dataset_role")
        if role not in DATASET_ROLES:
            block("dataset_role", "数据用途缺失或不受支持", meta)
        development = role == "development"
        focused = role == selected_role
        scope = "feed" if focused else "reserved"
        paths, hashes = {}, {}
        for field in ("nc", "measurement", "machine", "process"):
            value = meta.get(field)
            required = field != "process"
            if not value:
                block("missing_file", f"缺少 {field} 文件", meta, scope if required else ("load" if focused else "reserved"))
                continue
            path = Path(value)
            if not path.is_absolute() and base is None:
                block("relative_path", f"{field} 为相对路径，必须提供 base_dir 或清单文件路径", meta, scope)
                continue
            path = resolver.resolve(path)
            paths[field] = str(path)
            file_scope = scope if required else ("load" if focused else "reserved")
            if not verify_files:
                continue
            if not path.is_file():
                block("missing_file", f"文件不存在：{path}", meta, file_scope)
                continue
            try:
                hashes[field] = file_hash(path)
                if field == "measurement":
                    meta["measurement_content_sha256"] = measurement_content_hash(path)
            except (OSError, ValueError, UnicodeError) as exc:
                block("unreadable_file", str(exc), meta, file_scope)
        meta["resolved_files"], meta["file_sha256"] = paths, hashes
        if "nc" in hashes and "machine" in hashes:
            try:
                from dataclasses import asdict
                from .feed import MachineConfig, parse_nc
                coordinate_mode = meta.get("coordinate_mode", "machine")
                cache_key = (hashes["nc"], hashes["machine"], coordinate_mode)
                if cache_key not in geometry_cache:
                    machine = MachineConfig.load(paths["machine"])
                    parsed = parse_nc(paths["nc"], machine,
                                      coordinate_mode=coordinate_mode)
                    geometry_cache[cache_key] = (parsed, version_of(asdict(machine)))
                parsed, machine_version = geometry_cache[cache_key]
                if parsed["issues"] or not parsed["segments"]:
                    block("nc_geometry", "NC几何或控制器语义未核实", meta, scope)
                if meta.get("program_version") != parsed["program_version"]:
                    block("program_version", "声明的程序版本与实际NC内容不匹配", meta, scope)
                if meta.get("machine_version", machine_version) != machine_version:
                    block("machine_version", "声明的机床配置版本与实际文件不匹配", meta, scope)
                meta["machine_version"] = machine_version
            except (ValueError, OSError, TypeError, KeyError) as exc:
                block("nc_geometry", f"NC/机床配置不能用于空间特征：{exc}", meta, scope)
        if not verify_files:
            block("files_not_verified", "尚未实际核对输入文件，禁止正式训练", meta, scope)
        if "measurement" in hashes:
            content = meta["measurement_content_sha256"]
            previous = identities.get(content)
            if previous is not None:
                block("duplicate_measurement", f"实测内容与 {previous['record_id']} 重复，复制记录不能算作新实验", meta,
                      "common" if focused or previous.get("dataset_role") == selected_role else "reserved")
            identities[content] = meta
            try:
                from .measurement import read_measurement, resolve_position_policy, select_position_channels
                measurement = read_measurement(paths["measurement"], sample_period_s=meta.get("sample_period_s"))
                channels = set(measurement.values)
                actual = "actual_feed_mm_min" in channels or {f"actual_v{a}_mm_min" for a in "xyz"} <= channels
                derivative = meta.get("position_derivative_verified") is True and {f"actual_{a}_mm" for a in "xyz"} <= channels
                if not (actual or derivative):
                    block("feed_channel", "缺少实际进给或经核验的实际XYZ差分标签", meta, scope)
                meta["position_policy"] = resolve_position_policy(meta)
                position_source, position_columns = select_position_channels(channels,
                    position_policy=meta["position_policy"])
                meta["position_source"] = position_source
                if "program_line_raw" not in channels or not position_columns:
                    block("alignment_channels", "空间对齐缺少程序行号或完整XYZ位置（实际或采集指令）", meta, scope)
                if not len(measurement.time_s) or not measurement.valid.any():
                    block("empty_measurement", "实测记录为空或没有有效采样", meta, scope)
                if focused and not {"spindle_power_w", "spindle_speed_rpm"} <= channels:
                    block("power_channels", "负载训练缺少主轴功率或转速通道", meta, "load")
            except (ValueError, OSError, UnicodeError, KeyError, TypeError) as exc:
                block("measurement_format", f"实测读取失败：{exc}", meta, scope)
        family = meta.get("program_id")
        if not isinstance(family, str) or not family.strip():
            block("program_id", "缺少明确程序族标识", meta, scope)
        else:
            role_programs.setdefault(family, set()).add(role)
        if meta.get("program_version"):
            version_roles.setdefault(meta["program_version"], set()).add(role)
        for flag in ("complete", "alignment_reliable", "no_alarm", "tool_healthy", "round_complete"):
            if meta.get(flag) is not True:
                block("quality_unknown", f"{flag} 尚未明确通过审核", meta, scope)
        for key in ("machine_settings_version", "machining_mode", "program_version"):
            if not isinstance(meta.get(key), str) or not meta[key].strip():
                block("missing_context", f"缺少 {key}", meta, scope)
        memory_scope = "memory" if focused else "reserved"
        if not isinstance(meta.get("tool_session"), str) or not meta["tool_session"].strip():
            block("missing_context", "缺少真实刀具会话；不能进行同位置记忆评价", meta, memory_scope)
        value = meta.get("feed_override")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            block("feed_override", "加工前已知进给倍率无效或缺失", meta, scope)
        if meta.get("spacing_mm", SPACING_MM) != SPACING_MM:
            block("spacing", "正式实验 NC 网格必须为 0.5 mm", meta, scope)
        event = meta.get("actual_event_order")
        if type(event) is not int or event < 0:
            block("actual_event_order", "缺少已核对的实际采集顺序，不能用计划轮次代替", meta, memory_scope)
        if not meta.get("acquired_at"):
            block("acquired_at", "缺少实际采集时间", meta, memory_scope)
        if meta.get("cutting_contact_confirmed") is not True and not (meta.get("process_aligned") is True and "process" in hashes):
            block("contact_unknown", "切削接触未确认，仅有G1/G2/G3不能放行进给监督", meta, scope)
        if development:
            feature = meta.get("feature")
            if feature not in PROGRAM_FEATURES:
                block("program_feature", "程序类别必须为外形/曲面/混合/斜面/台阶", meta)
            slot = (meta.get("round_id"), feature)
            if slot in slots:
                block("duplicate_slot", f"同一程序位置存在多份有效记录：{slots[slot]}", meta)
            slots[slot] = meta["record_id"]
        if focused and (meta.get("process_aligned") is not True or not meta.get("process_version")):
            block("process_alignment", "工艺资料版本和可靠对齐尚未确认", meta, "load")
        normalized.append(meta)
    for family, roles in role_programs.items():
        # Repeated programs across development folds are intended. Reserved
        # adaptation/calibration/test programs remain whole and isolated.
        isolated = roles - {"memory_validation"}
        if len(isolated) > 1:
            block("program_role_leakage", f"程序族 {family} 跨越开发/适配/区间校准/测试用途")
    for version, roles in version_roles.items():
        if len(roles - {"memory_validation"}) > 1:
            block("program_content_role_leakage", f"同一NC内容 {version} 不能改名后跨开发/适配/区间校准/测试用途")
    active = [r for r in normalized if r.get("status") == "active"]
    development = [r for r in active if r.get("dataset_role") == "development"]
    mapping = validate_round_metadata(development, require_complete=require_complete and selected_role == "development")
    for issue in mapping["issues"]:
        block(issue["code"], issue["message"], issue)
    expected = {(r, f) for r in DEVELOPMENT_ROUNDS for f in PROGRAM_FEATURES}
    if not require_complete:
        expected = {(r.get("round_id"), f) for r in development for f in PROGRAM_FEATURES}
    if selected_role != "development":
        expected = set()
    missing = [{"round_id": r, "feature": f} for r, f in sorted(expected - set(slots), key=lambda x: (str(x[0]), x[1]))]
    if missing:
        block("missing_slots", f"缺少 {len(missing)} 个有效程序位置（完整开发集需要450个）")
    if not any(r.get("dataset_role") == selected_role for r in active):
        block("empty_development" if selected_role == "development" else "empty_role", f"没有可用的{selected_role}记录")
    evidence = deepcopy(base_validation or manifest.get("base_validation") or {})
    base_check = verify_base_evidence(evidence, [r for r in active if r.get("dataset_role") == selected_role], base_dir=base)
    if not base_check["valid"]:
        for message in base_check["issues"]:
            block("base_not_validated", message, scope="load")
    for kind, bound in base_check["files"].items():
        evidence[f"{kind}_path"] = bound["path"]
    snapshot = {"schema": SNAPSHOT_SCHEMA, "protocol_version": PROTOCOL_VERSION,
                "records": normalized, "base_validation": deepcopy(evidence),
                "base_bound_files": base_check["files"], "files_verified": bool(verify_files)}
    if resolver.reference is not None:
        snapshot["artifact_relocations"] = resolver.reference
    snapshot_id = _digest(snapshot)
    return {"schema": MANIFEST_SCHEMA, "protocol_version": PROTOCOL_VERSION, "blockers": blockers,
            "feed_eligible": not any(x["scope"] in ("common", "feed") for x in blockers),
            "load_eligible": not any(x["scope"] in ("common", "feed", "load") for x in blockers),
            "memory_eligible": not any(x["scope"] in ("common", "feed", "load", "memory") for x in blockers),
            "snapshot": snapshot, "snapshot_id": snapshot_id, "records": normalized,
            "missing_slots": missing, "missing_rounds": mapping["missing_rounds"], "folds": protocol_folds(),
            "development_records": len(development), "expected_slots": 450}


def verify_snapshot(snapshot, *, snapshot_id=None):
    issues = []
    if snapshot.get("schema") != SNAPSHOT_SCHEMA or snapshot.get("protocol_version") != PROTOCOL_VERSION:
        issues.append({"code": "snapshot_schema", "message": "数据快照版本不兼容"})
    if snapshot_id is not None and _digest(snapshot) != snapshot_id:
        issues.append({"code": "snapshot_changed", "message": "快照元数据已改变"})
    if snapshot.get("files_verified") is not True:
        issues.append({"code": "files_not_verified", "message": "快照未实际验证文件"})
    for value in snapshot.get("base_bound_files", {}).values():
        if not Path(value["path"]).is_file() or file_hash(value["path"]) != value["sha256"]:
            issues.append({"code": "base_evidence_changed", "message": "基础功率模型或验证报告已变化"})
    for record in snapshot.get("records", []):
        if record.get("status") != "active":
            continue
        for field, expected in record.get("file_sha256", {}).items():
            path = record["resolved_files"][field]
            if not Path(path).is_file() or file_hash(path) != expected:
                issues.append({"code": "input_changed", "record_id": record["record_id"], "field": field,
                               "message": f"任务锁定后输入已变化：{path}"})
    return {"valid": not issues, "blockers": issues, "snapshot_id": _digest(snapshot)}


def activate_round_replacement(records, replacements, *, base_dir=None):
    """Pure transaction: audit five recollections, then switch each original slot."""
    replacements = deepcopy(list(replacements))
    replacement_meta = [_metadata(r) for r in replacements]
    rounds = {r.get("round_id") for r in replacement_meta}
    if len(rounds) != 1 or len(replacements) != 5 or {r.get("feature") for r in replacement_meta} != set(PROGRAM_FEATURES):
        raise ValueError("整轮替换必须包含同一轮次的五类完整程序位置")
    if len({r.get("record_id") for r in replacement_meta}) != 5:
        raise ValueError("整轮补采必须使用五个唯一新 record_id")
    original = deepcopy(list(records))
    round_id = next(iter(rounds))
    previous = [_metadata(r) for r in original if _metadata(r).get("round_id") == round_id
                and _metadata(r).get("status", "active") == "active"]
    if len(previous) != 5 or {r.get("feature") for r in previous} != set(PROGRAM_FEATURES):
        raise ValueError("整轮替换必须对应已有五类唯一 active 程序位置")
    overwritten_ids = {r["record_id"] for r in previous}
    candidates, indices, availability, audit_cache = [], [], {}, {}
    for replacement in replacements:
        new, index, observations = _review_recollection(original, replacement, base_dir=base_dir,
                                                        overwritten_record_ids=overwritten_ids, audit_cache=audit_cache)
        candidates.append(new)
        indices.append(index)
        availability.update(observations)
    if len(set(indices)) != 5:
        raise ValueError("整轮替换必须逐槽唯一对应五条旧记录")
    complete = all(r.get("complete") is True for r in candidates)
    for candidate in candidates:
        candidate["round_complete"] = complete
        candidate["round_completeness_evidence"] = dict(
            rule="five_unique_active_slots_each_individually_complete",
            record_ids=[r["record_id"] for r in candidates], recomputed_after_round_replacement=round_id)
    report = inspect_manifest(candidates, base_dir=base_dir, require_complete=False)
    blockers = [b for b in report["blockers"] if b["scope"] in ("common", "feed")]
    if blockers:
        raise ValueError("整轮替换未通过检查：" + "；".join(b["message"] for b in blockers[:8]))
    for index, candidate in zip(indices, candidates):
        original[index]["status"] = "superseded"
        original[index]["superseded_by"] = [candidate["record_id"]]
    for index, observation in availability.items():
        original[index]["source_availability"] = observation
    for record in report["records"]:
        record["recollection"]["round_review_snapshot_id"] = report["snapshot_id"]
        if record.get("actual_event_order") is None or not record.get("tool_session"):
            record["memory_eligible"] = False
    return original + report["records"]


def _review_recollection(records, replacement, *, base_dir=None, overwritten_record_ids=(), audit_cache=None):
    """Shared identity/provenance checks, without activating or writing sources."""
    from .experiment_protocol import round_spec

    original = deepcopy(list(records))
    audit_cache = {} if audit_cache is None else audit_cache
    if not isinstance(replacement, dict):
        raise ValueError("单记录补采必须提供记录对象")
    new = deepcopy(_metadata(replacement))
    replaced_ids = new.get("replaces")
    if isinstance(replaced_ids, str):
        replaced_ids = [replaced_ids]
    if not isinstance(replaced_ids, list) or len(replaced_ids) != 1 or not isinstance(replaced_ids[0], str):
        raise ValueError("单记录补采 replaces 必须明确指定一个旧 record_id")
    matches = [(i, _metadata(r)) for i, r in enumerate(original)
               if _metadata(r).get("record_id") == replaced_ids[0]]
    if len(matches) != 1 or matches[0][1].get("status", "active") != "active":
        raise ValueError("单记录补采必须指向唯一的现有 active 记录")
    index, old = matches[0]
    if new.get("status", "active") != "active" or new.get("dataset_role") != "development" or old.get("dataset_role") != "development":
        raise ValueError("单记录补采仅适用于 active 开发记录")
    if not isinstance(new.get("record_id"), str) or not new["record_id"].strip() or any(
            _metadata(r).get("record_id") == new["record_id"] for r in original):
        raise ValueError("补采必须使用历史清单中未出现的新 record_id")
    if any(new.get(key) != old.get(key) for key in ("round_id", "feature", "program_id")):
        raise ValueError("单记录补采必须保持原 round_id、feature 和 program_id")
    if any(old.get(key) is not None and new.get(key) != old[key]
           for key in ("machine_id", "diameter_mm", "teeth")):
        raise ValueError("固定工况补采不能改变已知机床、刀径或齿数")
    spec = round_spec(new.get("round_id"))
    expected_slot = spec["program_slots"].get(new.get("feature"))
    if expected_slot is None or new.get("program_slot") != expected_slot or old.get("program_slot", expected_slot) != expected_slot:
        raise ValueError("补采 program_slot 必须显式匹配该轮的预定NC位置")

    def resolve(value, label):
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError(f"缺少{label}文件")
        path = Path(value)
        if not path.is_absolute():
            if base_dir is None:
                raise ValueError(f"{label}为相对路径，必须提供 base_dir")
            path = Path(base_dir) / path
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"{label}文件不存在：{path}")
        return path

    planned_nc = resolve(old.get("nc"), "原记录预定NC")
    replacement_nc = resolve(new.get("nc"), "补采NC")
    expected_nc_sha256 = file_hash(planned_nc)
    if file_hash(replacement_nc) != expected_nc_sha256 or new.get("program_version") != old.get("program_version"):
        raise ValueError("补采NC内容或程序版本与原记录预定NC不一致；不能以相近几何替代预定程序")
    frozen_nc_hash = old.get("file_sha256", {}).get("nc")
    if frozen_nc_hash and frozen_nc_hash != expected_nc_sha256:
        raise ValueError("原记录预定NC已变化，不能据变化后的文件激活补采")
    acquired_at = new.get("acquired_at")
    try:
        if not isinstance(acquired_at, str) or not acquired_at.strip():
            raise ValueError
        new_time = datetime.fromisoformat(acquired_at)
    except ValueError as exc:
        raise ValueError("补采必须提供真实 acquired_at，格式为ISO日期或时间") from exc
    try:
        old_time = datetime.fromisoformat(old.get("acquired_at", ""))
    except (TypeError, ValueError):
        old_time = None
    if old_time is not None and (new_time == old_time or new_time.date() < old_time.date()):
        raise ValueError("补采 acquired_at 不得继承旧记录或早于原采集日期")
    for key in ("acquired_at", "tool_session", "actual_event_order"):
        new.setdefault(key, None)
        if new[key] is not None and old.get(key) is not None and new[key] == old[key]:
            raise ValueError(f"补采 {key} 不得继承旧记录；未知刀具会话或实际顺序应保留 null")
    event = new["actual_event_order"]
    if event is not None:
        if type(event) is not int or event < 0:
            raise ValueError("补采实际采集顺序必须为审核后的非负整数或 null")
        known_events = [_metadata(r).get("actual_event_order") for r in original]
        if any(type(previous) is int and event <= previous for previous in known_events):
            raise ValueError("补采实际采集顺序必须追加在已有真实顺序之后；不能按计划轮次插入")
    recollection = new.get("recollection")
    if not isinstance(recollection, dict) or not isinstance(recollection.get("reason"), str) or not recollection["reason"].strip():
        raise ValueError("补采必须提供 recollection.reason 和真实采集记录 evidence_path")
    evidence_path = resolve(recollection.get("evidence_path"), "补采采集记录")
    measurement_path = resolve(new.get("measurement"), "补采实测")
    new_content = measurement_content_hash(measurement_path)

    def current_hash(path):
        key = ("file", path)
        if key not in audit_cache:
            audit_cache[key] = file_hash(path)
        return audit_cache[key]

    def numeric_receipt(meta):
        ref = meta.get("review_evidence", {}).get("numeric")
        if not ref:
            return None
        path = resolve(ref.get("path"), "数值审核回执")
        if current_hash(path) != ref.get("sha256"):
            raise ValueError("数值审核回执哈希不匹配")
        key = ("numeric", path)
        if key not in audit_cache:
            audit_cache[key] = json.loads(path.read_text(encoding="utf-8-sig"))
        report = audit_cache[key]
        if report.get("record_id") != meta.get("record_id"):
            raise ValueError("数值审核回执 record_id 不匹配")
        return dict(path=str(path), sha256=ref["sha256"], file_sha256=report.get("file_sha256"),
                    content_sha256=report.get("content_sha256"), content_hash_schema=report.get("content_hash_schema"))

    new_numeric = numeric_receipt(new)
    new_file_hash = current_hash(measurement_path)
    if new_numeric and new_numeric["file_sha256"] != new_file_hash:
        raise ValueError("补采数值审核未绑定当前实测文件")
    availability = {}
    for prior_index, record in enumerate(original):
        meta = _metadata(record)
        prior_path = resolve(meta.get("measurement"), "历史实测")
        if prior_path == measurement_path:
            raise ValueError("补采必须使用独立原始文件，不能覆盖或重用历史实测路径")
        prior_numeric = numeric_receipt(meta)
        expected_hash = meta.get("file_sha256", {}).get("measurement") or (prior_numeric or {}).get("file_sha256")
        observed_hash = current_hash(prior_path)
        cached_content = meta.get("measurement_content_sha256")
        comparable_numeric = (prior_numeric and new_numeric
            and prior_numeric.get("content_hash_schema") == "canonical_channel_float64_sha256_v1"
            and new_numeric.get("content_hash_schema") == prior_numeric["content_hash_schema"]
            and all(isinstance(v, str) and len(v) == 64 for v in
                    (prior_numeric.get("content_sha256"), new_numeric.get("content_sha256"))))
        if comparable_numeric and prior_numeric["content_sha256"] == new_numeric["content_sha256"]:
            raise ValueError(f"补采实测内容与历史记录 {meta.get('record_id')} 重复，不能以复制文件替代新采集")
        if expected_hash and expected_hash != observed_hash:
            if (meta.get("record_id") not in overwritten_record_ids or not prior_numeric or not new_numeric
                    or prior_numeric.get("file_sha256") != expected_hash
                    or not comparable_numeric):
                raise ValueError("历史实测源已改变；仅本次明确替换槽可凭绑定的原数值回执保留历史内容身份")
            availability[prior_index] = dict(status="externally_overwritten_before_recollection_activation",
                path=str(prior_path), expected_file_sha256=expected_hash, observed_file_sha256=observed_hash,
                original_bytes_available_at_path=False, historical_numeric_receipt=prior_numeric)
            prior_content = None
        else:
            if comparable_numeric and prior_numeric["file_sha256"] == observed_hash:
                prior_content = None  # Compared above using the same audited canonical schema.
            else:
                key = ("content", prior_path)
                if key not in audit_cache:
                    audit_cache[key] = (cached_content if cached_content and expected_hash == observed_hash
                                        else measurement_content_hash(prior_path))
                prior_content = audit_cache[key]
        if prior_content is not None and prior_content == new_content:
            raise ValueError(f"补采实测内容与历史记录 {meta.get('record_id')} 重复，不能以复制文件替代新采集")

    new["status"] = "active"
    new.pop("memory_eligible", None)
    new["replaces"] = [old["record_id"]]
    new["recollection"] = {**deepcopy(recollection), "evidence_path": str(evidence_path),
                           "evidence_sha256": file_hash(evidence_path), "acquired_at": acquired_at,
                           "expected_program_slot": expected_slot, "expected_nc_sha256": expected_nc_sha256,
                           "measurement_sha256": new_file_hash,
                           "measurement_content_sha256": new_content}
    return new, index, availability


def activate_record_replacement(records, replacement, *, base_dir=None):
    """Return a new manifest after one recollected slot passes its complete round."""
    original = deepcopy(list(records))
    new, index, _ = _review_recollection(original, replacement, base_dir=base_dir)
    original[index]["status"] = "superseded"
    original[index]["superseded_by"] = [new["record_id"]]
    same_round = [r for r in original if _metadata(r).get("round_id") == new["round_id"]
                  and _metadata(r).get("dataset_role") == "development"] + [new]
    active_round = [r for r in same_round if _metadata(r).get("status", "active") == "active"]
    # Round completeness is derived from the five current slots. It must be
    # recomputed after recollection, not inherited from the incomplete round.
    round_complete = (len(active_round) == len(PROGRAM_FEATURES)
                      and {_metadata(r).get("feature") for r in active_round} == set(PROGRAM_FEATURES)
                      and all(_metadata(r).get("complete") is True for r in active_round))
    for record in active_round:
        if _metadata(record).get("round_complete") is not round_complete:
            record["round_complete"] = round_complete
            record["round_completeness_evidence"] = dict(
                rule="five_unique_active_slots_each_individually_complete",
                record_ids=[_metadata(r)["record_id"] for r in active_round],
                recomputed_after_replacement=new["record_id"])
    report = inspect_manifest(same_round, base_dir=base_dir, require_complete=False)
    blockers = [b for b in report["blockers"] if b["scope"] in ("common", "feed")]
    if blockers:
        raise ValueError("单记录补采未通过同轮五类检查：" + "；".join(b["message"] for b in blockers[:8]))
    checked = next(r for r in report["records"] if r.get("record_id") == new["record_id"])
    # This diagnostic cannot grant load or memory qualification: callers still
    # inspect the complete manifest with their exact base-power evidence.
    checked["recollection"]["round_review_snapshot_id"] = report["snapshot_id"]
    if checked.get("actual_event_order") is None or not checked.get("tool_session"):
        checked["memory_eligible"] = False
    return original + [checked]
