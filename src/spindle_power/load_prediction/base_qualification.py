"""Hash-bound qualification of measured-speed labels without changing models.

A bridge can retain an original independent event confirmation after a purely
semantic interface repair. It is not a new independent experiment. Qualification
also needs record-level applicability evidence; engineering replay alone fails.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from .common import file_hash

QUALIFICATION_SCHEMA = "afc.base.label_qualification.1"
PAIRED_SETTINGS_QUALIFICATION_SCHEMA = "afc.base.label_qualification.2"
BRIDGE_SCHEMA = "afc.base.interface_bridge.1"
APPLICABILITY_SCHEMA = "afc.base.label_applicability.1"
PAIRED_SETTINGS_APPLICABILITY_SCHEMA = "afc.base.label_applicability.2"
BRIDGE_KIND = "frozen_event_confirmation_with_interface_equivalence"
SETTINGS_KEYS = ("machine_settings_version", "measurement_settings_version")
APPLICABILITY_CHECKS = ("power_channel_identity_verified", "machine_settings_compatible",
                        "measurement_settings_compatible", "baseline_on_machining_idle_verified")


def embedded_json_artifact(path):
    """Preserve exact JSON bytes inside the report copied into a fixed snapshot."""
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8-sig")
    # Published scientific artifacts are UTF-8 without BOM. Do not silently
    # discard a BOM while claiming the original byte hash for the embedded text.
    if raw != text.encode("utf-8"):
        raise ValueError("嵌入证据必须为UTF-8无BOM；原文件保持只读")
    json.loads(text)
    return dict(source_path=str(Path(path).resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                json_text=text)


def _embedded(artifact, label):
    if not isinstance(artifact, dict) or not isinstance(artifact.get("json_text"), str):
        raise ValueError(f"缺少嵌入的{label}证据")
    if hashlib.sha256(artifact["json_text"].encode("utf-8")).hexdigest() != artifact.get("sha256"):
        raise ValueError(f"嵌入的{label}证据SHA-256不匹配")
    value = json.loads(artifact["json_text"])
    if not isinstance(value, dict):
        raise ValueError(f"{label}证据必须为JSON对象")
    return value


def _settings_domains(domain, *, paired):
    """Return explicit setting pairs, never a Cartesian product or wildcard."""
    if not paired:
        if "settings_domains" in domain:
            raise ValueError("单域资格不能包含多域列表；请显式使用新版配对资格")
        if any(not isinstance(domain.get(key), str) or not domain[key].strip() for key in SETTINGS_KEYS):
            raise ValueError("桥接适用域缺少机床和采样设置版本")
        return None
    if any(key in domain for key in SETTINGS_KEYS):
        raise ValueError("配对资格不能同时包含旧版单域设置字段")
    values = domain.get("settings_domains")
    if not isinstance(values, list) or not values:
        raise ValueError("配对资格需要非空的明确设置域列表")
    indexed, pairs = {}, set()
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("设置域必须为包含标识和设置配对的对象")
        if any(not isinstance(value.get(key), str) or not value[key].strip()
               or value[key] != value[key].strip() or value[key] == "*"
               for key in ("settings_domain_id", *SETTINGS_KEYS)):
            raise ValueError("设置域标识和设置版本必须是明确的非空字符串，不能使用通配符")
        key = value["settings_domain_id"]
        pair = tuple(value[name] for name in SETTINGS_KEYS)
        if key in indexed or pair in pairs:
            raise ValueError("设置域标识或设置配对重复，无法唯一绑定记录")
        indexed[key] = value
        pairs.add(pair)
    return indexed


def _settings_audit_errors(applicability, domains):
    """Every declared pair needs its own completed applicability audit."""
    audits = applicability.get("settings_audits")
    if not isinstance(audits, list) or any(not isinstance(row, dict) for row in audits):
        return ["配对资格缺少逐设置域适用性审核"]
    indexed = {row.get("settings_domain_id"): row for row in audits}
    if len(indexed) != len(audits) or set(indexed) != set(domains):
        return ["逐设置域审核必须唯一且恰好覆盖声明的设置域"]
    errors = []
    for key, domain in domains.items():
        audit = indexed[key]
        if any(audit.get(name) != domain[name] for name in SETTINGS_KEYS):
            errors.append(f"设置域 {key} 的审核与声明配对不一致")
        for name in APPLICABILITY_CHECKS:
            if audit.get(name) is not True:
                errors.append(f"设置域 {key} 尚未核实：{name}")
    return errors


def _bridge_errors(report, evidence, records):
    errors = []
    from .base_timeline import AUDITED_INTERFACE_VERSION
    domain = evidence["domain"]
    if (report.get("schema") not in (QUALIFICATION_SCHEMA, PAIRED_SETTINGS_QUALIFICATION_SCHEMA)
            or report.get("qualification_kind") != BRIDGE_KIND
            or report.get("interface_version") != AUDITED_INTERFACE_VERSION):
        return ["缺少当前审核接口的明确版本与桥接资格类型"]
    if report.get("new_independent_interface_confirmation") is not False:
        errors.append("桥接不能宣称新增独立接口确认")
    paired = report["schema"] == PAIRED_SETTINGS_QUALIFICATION_SCHEMA
    try:
        settings = _settings_domains(domain, paired=paired)
    except ValueError as exc:
        return errors + [str(exc)]
    if domain.get("initial_state") != "settled_with_observed_prior":
        errors.append("桥接适用域必须限定为已观察且稳定的初态")
    if domain.get("transitions") != ["settled0_to_target", "settled_target_to0"]:
        errors.append("桥接不能扩大已确认的启停转换范围")
    try:
        chain = report.get("evidence_chain", {})
        freeze = _embedded(chain.get("freeze"), "原冻结")
        confirmation = _embedded(chain.get("confirmation"), "原独立事件确认")
        bridge = _embedded(chain.get("bridge"), "接口等价性")
        applicability = _embedded(chain.get("applicability"), "标签适用域")
        if str(freeze.get("model_sha256", "")).lower() != evidence["model_sha256"]:
            errors.append("原冻结模型哈希与标签模型不一致")
        development = set(freeze.get("development_rounds", []))
        held = set(freeze.get("confirmation_rounds", []))
        if (not development or not held or development & held
                or freeze.get("confirmation_target_power_examined") is not False):
            errors.append("原冻结未证明开发与确认隔离")
        if str(confirmation.get("frozen_manifest_sha256", "")).lower() != chain["freeze"]["sha256"]:
            errors.append("原确认未绑定原冻结清单")
        summary = confirmation.get("summary", {})
        if (summary.get("passed") is not True or not summary.get("event_count")
                or any(summary.get(k) != 0 for k in
                       ("failed_events", "censored_or_invalid_events", "excluded_events"))):
            errors.append("原独立事件确认未完整通过")
        expected = dict(schema=BRIDGE_SCHEMA, status="equivalent_in_audited_domain",
                        model_sha256=evidence["model_sha256"], model_version=evidence["version"],
                        interface_version=AUDITED_INTERFACE_VERSION,
                        confirmation_sha256=chain["confirmation"]["sha256"],
                        freeze_sha256=chain["freeze"]["sha256"])
        if any(bridge.get(k) != v for k, v in expected.items()):
            errors.append("接口桥接未通过或未绑定当前模型、冻结与独立确认")
        checks = bridge.get("checks", {})
        for key in ("model_unchanged", "core_unchanged", "thresholds_unchanged",
                    "alignment_frozen_before_confirmation", "alignment_uses_no_power",
                    "event_curves_equivalent", "settling_grid_difference_explained",
                    "full_record_valid", "raw_inputs_unchanged"):
            if checks.get(key) is not True:
                errors.append(f"接口桥接检查未通过：{key}")
        for name in ("base_events.py", "base_timeline.py"):
            if bridge.get("production_source_sha256", {}).get(name) != file_hash(Path(__file__).with_name(name)):
                errors.append(f"接口桥接与当前执行源码不一致：{name}")
        applicability_schema = PAIRED_SETTINGS_APPLICABILITY_SCHEMA if paired else APPLICABILITY_SCHEMA
        if (applicability.get("schema") != applicability_schema
                or applicability.get("status") != "verified"
                or applicability.get("model_sha256") != evidence["model_sha256"]
                or applicability.get("interface_version") != AUDITED_INTERFACE_VERSION
                or applicability.get("domain") != domain):
            errors.append("缺少已核实且与模型/接口/适用域一致的真实标签适用性报告")
        for key in APPLICABILITY_CHECKS:
            if applicability.get(key) is not True:
                errors.append(f"标签适用性尚未核实：{key}")
        if paired:
            errors.extend(_settings_audit_errors(applicability, settings))
        audits = applicability.get("record_audits", [])
        indexed = {row.get("record_id"): row for row in audits if isinstance(row, dict)}
        if not indexed or len(indexed) != len(audits) or None in indexed:
            errors.append("适用性报告缺少唯一的逐记录审核")
        for record in records:
            meta = {**record.get("metadata", {}), **record}
            audit = indexed.get(meta.get("record_id"), {})
            expected_hash = meta.get("file_sha256", {}).get("measurement")
            if (not audit or not expected_hash or audit.get("measurement_sha256") != expected_hash
                    or audit.get("initial_state_verified") is not True
                    or audit.get("timeline_verified") is not True):
                errors.append(f"记录 {meta.get('record_id')} 缺少哈希绑定的初态/时间线适用性审核")
            record_domain = settings.get(audit.get("settings_domain_id")) if paired else domain
            if paired and (record_domain is None or any(audit.get(key) != record_domain[key] for key in SETTINGS_KEYS)):
                errors.append(f"记录 {meta.get('record_id')} 缺少明确一致的设置域审核绑定")
                continue
            for key in SETTINGS_KEYS:
                if meta.get(key) != record_domain.get(key):
                    errors.append(f"记录 {meta.get('record_id')} 的{key}不在标签适用域")
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        errors.append(f"基础功率桥接证据不可用：{exc}")
    return errors


def verify_label_evidence(evidence, records=(), *, base_dir=None,
                          historical_consumer_source=None, artifact_relocations=None):
    """Verify existing formal certificates or a fully evidenced limited bridge."""
    if isinstance(evidence, dict) and evidence.get("mode") == "hybrid_estimated_labels":
        from .hybrid_labels import verify_hybrid_evidence
        return verify_hybrid_evidence(evidence, records, base_dir=base_dir,
            historical_consumer_source=historical_consumer_source, artifact_relocations=artifact_relocations)
    if isinstance(evidence, dict) and evidence.get("mode") == "record_idle_labels":
        from .record_idle import verify_record_idle_evidence
        return verify_record_idle_evidence(evidence, records, base_dir=base_dir,
            historical_consumer_source=historical_consumer_source, artifact_relocations=artifact_relocations)
    errors, bound_files = [], {}
    evidence = evidence or {}
    records = list(records)
    if historical_consumer_source is not None:
        return dict(valid=False, issues=["历史消费者源码仅适用于版本化本记录或混合标签凭据"], files={})
    required = ("version", "model_path", "model_sha256", "report_path", "report_sha256", "domain")
    if (not isinstance(evidence, dict) or any(not evidence.get(k) for k in required)
            or evidence.get("validated") is not True or evidence.get("applicable") is not True
            or evidence.get("mode") != "measured_speed_labels"):
        return dict(valid=False, issues=["缺少基础功率模型/报告哈希、适用域或实测转速标签模式的正式验证凭据"], files={})
    for kind in ("model", "report"):
        try:
            from .artifact_paths import ArtifactResolver
            path = ArtifactResolver(artifact_relocations, base_dir=base_dir).resolve(evidence[f"{kind}_path"])
            if not path.is_file() or file_hash(path) != evidence[f"{kind}_sha256"]:
                raise ValueError(f"基础功率{kind}文件不存在或SHA-256不匹配")
            bound_files[kind] = dict(path=str(path), sha256=evidence[f"{kind}_sha256"])
        except (OSError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    domain = evidence["domain"]
    if not isinstance(domain, dict) or not domain.get("machine_id") or domain.get("power_channel") != "spindle_power_w":
        errors.append("基础功率适用域缺少机床身份或功率通道")
    else:
        lower, upper = domain.get("spindle_rpm_min"), domain.get("spindle_rpm_max")
        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in (lower, upper)) or lower > upper):
            errors.append("基础功率适用转速区间无效")
        else:
            from .experiment_protocol import round_spec
            for record in records:
                meta = {**record.get("metadata", {}), **record}
                if meta.get("machine_id") != domain["machine_id"]:
                    errors.append(f"记录 {meta.get('record_id')} 的机床不在基础功率适用域")
                if meta.get("dataset_role") == "development":
                    try:
                        speed = round_spec(meta.get("round_id"))["spindle_cmd_rpm"]
                        if not lower <= speed <= upper:
                            errors.append(f"R{meta['round_id']:02d} 转速超出基础功率适用域")
                    except ValueError:
                        errors.append("记录轮次无效，无法验证基础功率域")
    report = None
    if len(bound_files) == 2:
        try:
            from .base_power import BasePowerModel
            model = BasePowerModel.load(bound_files["model"]["path"])
            report = json.loads(Path(bound_files["report"]["path"]).read_text(encoding="utf-8-sig"))
            if not isinstance(report, dict):
                raise ValueError("验证报告必须为JSON对象")
            bridge = report.get("qualification_kind") == BRIDGE_KIND
            if model.version != evidence["version"] or (not bridge and model.status != "validated_in_sample_domain"):
                errors.append("基础功率模型版本或自身验证状态与凭据不一致")
            for key, expected in (("model_sha256", evidence["model_sha256"]), ("model_version", evidence["version"]),
                                  ("mode", "measured_speed_labels"), ("domain", domain)):
                if report.get(key) != expected:
                    errors.append(f"验证报告与凭据的 {key} 不匹配")
            if report.get("eligible_for_load_training") is not True or report.get("status") != "validated":
                errors.append("报告未正式放行实测转速模式负载标签；事件模式通过不能替代")
            if bridge and isinstance(domain, dict):
                errors.extend(_bridge_errors(report, evidence, records))
        except (ValueError, OSError, TypeError, KeyError) as exc:
            errors.append(f"基础功率验证报告不可读取：{exc}")
    return dict(valid=not errors, issues=errors, files=bound_files, report=report)


def qualify_audited_prediction(evidence, model, timeline, record_context, *, base_dir=None):
    """Bind a certificate to this in-memory model and this exact measurement."""
    context = record_context or {}
    check = verify_label_evidence(evidence, [context], base_dir=base_dir)
    errors = list(check["issues"])
    report = check.get("report") or {}
    if report.get("qualification_kind") != BRIDGE_KIND:
        errors.append("审核时间线接口需要自身的桥接资格，不能继承另一接口的模型状态")
    if check["files"].get("model"):
        from .base_power import BasePowerModel
        bound_model = BasePowerModel.load(check["files"]["model"]["path"])
        if asdict(bound_model) != asdict(model):
            errors.append("当前内存模型与资格绑定模型不一致")
    try:
        applicability = _embedded(report.get("evidence_chain", {}).get("applicability"), "标签适用域")
        matches = [r for r in applicability.get("record_audits", []) if r.get("record_id") == context.get("record_id")]
        if len(matches) != 1:
            errors.append("当前记录缺少唯一适用性审核")
        else:
            audit = matches[0]
            if (audit.get("timeline_sha256") != timeline.fingerprint()
                    or audit.get("time_speed_sha256") != timeline.time_speed_sha256
                    or audit.get("program_version") != timeline.program_version):
                errors.append("资格与当前审核时间线、程序或时间/转速指纹不一致")
        source = context.get("resolved_files", {}).get("measurement")
        if not source or str(Path(source).resolve()) != str(Path(timeline.record_source).resolve()):
            errors.append("资格与当前实测来源不同")
        if not source or not Path(source).is_file() or file_hash(source) != context.get("file_sha256", {}).get("measurement"):
            errors.append("当前实测文件与资格SHA-256不一致")
        domain = (evidence or {}).get("domain", {})
        lower, upper = domain.get("spindle_rpm_min"), domain.get("spindle_rpm_max")
        levels = [timeline.initial_speed_rpm] + [event.get("speed_rpm") for event in timeline.events]
        if any(level != 0 and not lower <= level <= upper for level in levels):
            errors.append("当前初态或审核事件转速超出标签资格适用域")
    except (ValueError, TypeError, KeyError, OSError) as exc:
        errors.append(f"逐记录基础标签资格不可用：{exc}")
    return dict(valid=not errors, issues=errors, report_sha256=(evidence or {}).get("report_sha256"))
