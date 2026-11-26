"""Immutable, load-only support-fill derivations with a verified parent chain."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import numpy as np
import pandas as pd

from .artifact_paths import ArtifactResolver, archived_load_directory
from .common import checkpoint, file_hash
from .samples import SampleSet
from .support_fill import SUPPORT_FILL_RULES_VERSION, fill_sample_from_process_source, fill_missing_process_support
from .training_checkpoint import digest, source_fingerprints


SCHEMA = "afc.support_fill_derivation.1"
ENDPOINT_SCHEMA = "afc.support_fill_derivation.2"
ENDPOINT_POLICY = "terminal_uniform_ulp_v1"
RELEASE_SCHEMA = "afc.support_fill_release.1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def archived_route_fingerprints(source):
    """Read the same fingerprint fields from an explicit source archive."""
    directory = archived_load_directory(source)
    _require(directory.is_dir(), "缺少归档负载消费者源码目录")
    return source_fingerprints(directory)


def implementation_identity():
    directory = Path(__file__).parent
    names = ("support_fill.py", "support_fill_snapshot.py", "samples.py", "common.py")
    return {**{name: file_hash(directory / name) for name in names},
            "shared/process_path.py": file_hash(directory.parents[1] / "afc_common" / "process_path.py")}


def _inside(directory, filename):
    directory = Path(directory).resolve()
    result = (directory / filename).resolve()
    _require(result.is_relative_to(directory), "补缺快照文件越出快照目录")
    return result


def _snapshot(path):
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(value.get("snapshot_id") == digest({key: item for key, item in value.items() if key != "snapshot_id"}),
             "补缺链快照内容身份不符")
    _require(value.get("synthetic") is False, "正式补缺链不接受合成快照")
    return value


def _development_entries(snapshot, *, load_only=False):
    from .experiment_protocol import PROGRAM_FEATURES, validate_round_metadata
    entries = [item for item in snapshot["records"] if item["kind"] == "load"]
    _require(not load_only or len(entries) == len(snapshot["records"]), "补缺候选只能包含load记录")
    metadata = [item["metadata"] for item in entries]
    _require(len(entries) == 450 and validate_round_metadata(metadata)["valid"], "补缺需要完整450条开发记录")
    expected = {(round_id, feature) for round_id in range(1, 91) for feature in PROGRAM_FEATURES}
    _require({(row.get("round_id"), row.get("feature")) for row in metadata} == expected,
             "补缺开发轮次/类别重复或不完整")
    for field in ("record_id", "measurement_content_sha256"):
        values = [row.get(field) for row in metadata]
        _require(all(isinstance(value, str) and value for value in values) and len(set(values)) == 450,
                 f"补缺记录身份重复或缺失：{field}")
    _require(all(row.get("dataset_role") == "development" and row.get("group_id") is None for row in metadata),
             "补缺派生禁止混入事件或盲测记录")
    return entries


def _frame(directory, item):
    path = _inside(directory, item["file"])
    _require(file_hash(path) == item["sha256"], "补缺链CSV哈希不符")
    frame = pd.read_csv(path, float_precision="round_trip")
    _require(file_hash(path) == item["sha256"], "补缺链CSV在读取期间改变")
    _require(list(item["dtypes"]) == list(frame.columns), "补缺链CSV列登记不符")
    for column, dtype in item["dtypes"].items():
        _require(str(frame[column].dtype) == dtype, f"补缺链CSV类型登记不符：{column}")
    return frame


def normalize_source_endpoints(process, parent, *, diameter_mm):
    """Correct only a proven one-ULP-short final uniform cell; never expand gaps."""
    required = {"line", "execution_instance", "segment_start_mm", "segment_mm", "ap_mm", "ae_mm",
                "point_index_in_line", "points_in_line", "source_point_index", "source_file_line", "mapping_status"}
    _require(required <= set(process), "端点规范化缺少完整工艺来源字段")
    _require({"line", "execution_instance", "block_length_mm"} <= set(parent), "端点规范化缺少父NC段长")
    result = process.copy(deep=True)
    changes = []
    for (line, instance), group in process.groupby(["line", "execution_instance"], sort=False):
        grid = parent.loc[(parent.line == line) & (parent.execution_instance == instance)]
        if grid.empty:
            continue  # Unmapped source groups do not acquire an invented NC owner.
        lengths = grid.block_length_mm.unique()
        _require(len(lengths) == 1 and np.isfinite(lengths[0]) and lengths[0] >= 0, "端点规范化NC段长不唯一或非法")
        length = float(lengths[0])
        if length == 0:
            continue
        ordered = group.sort_values("point_index_in_line", kind="stable")
        n = len(ordered)
        last = ordered.iloc[-1]
        end = float(last.segment_mm)
        if end != np.nextafter(length, -np.inf) or end != length * n / n:
            continue
        _require(np.array_equal(ordered.point_index_in_line.to_numpy(float), np.arange(1, n + 1))
                 and np.all(ordered.points_in_line.to_numpy(float) == n), "末端规范化要求唯一完整来源点序列")
        starts, ends = ordered.segment_start_mm.to_numpy(float), ordered.segment_mm.to_numpy(float)
        _require(np.isfinite(starts).all() and np.isfinite(ends).all() and starts[0] == 0.
                 and np.all(ends > starts) and np.array_equal(starts[1:], ends[:-1]),
                 "末端规范化不能跨越来源缺口或重叠")
        original_bounds = np.asarray([length * index / n for index in range(n + 1)])
        _require(ordered.mapping_status.eq("mapped_uniform_nc").all()
                 and np.array_equal(starts, original_bounds[:-1]) and np.array_equal(ends, original_bounds[1:]),
                 "末端规范化要求完整原均分来源，内部边界不可重新解释")
        values = ordered[["ap_mm", "ae_mm"]].to_numpy(float)
        _require(np.isfinite(values).all() and (values >= 0).all()
                 and np.isfinite(diameter_mm) and diameter_mm > 0 and (values[:, 1] <= diameter_mm).all(),
                 "末端规范化来源几何非法")
        for column, minimum in (("source_point_index", 0), ("source_file_line", 1)):
            ids = ordered[column].to_numpy(float)
            _require(np.isfinite(ids).all() and np.all(ids == np.round(ids)) and np.all(ids >= minimum)
                     and len(np.unique(ids)) == n, "末端规范化来源行标识不完整或重复")
        index = ordered.index[-1]
        _require(result.index.is_unique, "工艺源行索引重复")
        result.loc[index, "segment_mm"] = length
        changes.append(dict(source_row_position=int(process.index.get_loc(index)), line=int(line),
            execution_instance=int(instance), source_point_index=int(last.source_point_index),
            source_file_line=int(last.source_file_line), points_in_line=n, original_end_mm=end,
            normalized_end_mm=length, terminal_arithmetic_mm=length * n / n, delta_ulp=1))
    return result, changes


def endpoint_normalized_sample(sample, process_path):
    """Verify frozen source bytes, derive canonical bytes and attach an additive receipt."""
    from .samples import read_process_table
    _require("support_fill" not in sample.metadata, "父样本已有补缺回执，拒绝重复派生")
    meta = deepcopy(sample.metadata)
    expected = meta.get("file_sha256", {}).get("process")
    path = Path(process_path).resolve()
    _require(isinstance(expected, str) and file_hash(path) == expected, "工艺源文件与父登记SHA-256不符")
    plan = SimpleNamespace(frame=sample.frame, program_version=meta["program_version"], machine_version=meta["machine_version"])
    # v2 consumes frozen canonical intervals, not reparsed TXT with a changed reader.
    with path.open(encoding="utf-8-sig") as stream:
        _require(not stream.readline().lstrip().startswith("序号"), "端点v2仅接受父已冻结canonical CSV")
    process = read_process_table(path, plan, line_offset=meta.get("process_line_offset", 0), steady_lines=meta.get("steady_lines", []))
    normalized, changes = normalize_source_endpoints(process, sample.frame, diameter_mm=meta["diameter_mm"])
    content = normalized.to_csv(index=False).encode("utf-8")
    checksum = hashlib.sha256(content).hexdigest()
    normalized_ref = dict(file=f"process_sources/{checksum}.csv", sha256=checksum)
    source = dict(path=str(path), sha256=expected, parent_registered_path=meta.get("resolved_files", {}).get("process"))
    normalization = dict(schema="afc.support_fill_source_normalization.1", policy=ENDPOINT_POLICY,
        original_source=source, normalized_source=normalized_ref, source_rows=len(process), changed_cells=changes,
        changed_columns=["segment_mm"], internal_boundaries_preserved=True, source_geometry_preserved=True)
    frame, receipt = fill_missing_process_support(sample.frame, normalized, diameter_mm=meta["diameter_mm"])
    _require(file_hash(path) == expected, "工艺源文件在端点规范化期间发生变化")
    receipt.update(source_file_verified=True, source=source, source_normalization=normalization,
                   record_id=meta.get("record_id"), implementation_sha256=file_hash(Path(__file__).with_name("support_fill.py")))
    meta["support_fill"] = receipt
    return SampleSet(frame, meta), normalized, content


def checked_endpoint_source(sample, process_path, directory, candidate_metadata):
    """Recompute the contract; neither a resigned canonical hash nor receipt can hide edits."""
    expected, process, content = endpoint_normalized_sample(sample, process_path)
    normalization = expected.metadata["support_fill"]["source_normalization"]
    actual = candidate_metadata.get("support_fill", {}).get("source_normalization")
    _require(actual == normalization, "端点来源规范化回执与原来源重算不一致")
    reference = normalization["normalized_source"]
    location = _inside(directory, reference["file"])
    _require(file_hash(location) == reference["sha256"] == hashlib.sha256(content).hexdigest(),
             "端点规范化canonical文件哈希或原来源重算不一致")
    return expected, process


def _verify_parent_chain(parent_service, parent, directory, historical_route_source):
    _require(not parent["source_snapshot"].get("support_fill_derivation"), "不允许重复补缺派生")
    _development_entries(parent)
    return parent_service._verify_load_route_derivation(parent, directory,
        historical_route_source=historical_route_source)


def _checked_reference(reference):
    _require(isinstance(reference, dict) and set(("path", "sha256")) <= set(reference), "门禁缺少文件引用")
    path = Path(reference["path"]).resolve()
    _require(file_hash(path) == reference["sha256"], "门禁证据文件哈希不符")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(file_hash(path) == reference["sha256"], "门禁证据文件在读取期间改变")
    return value


def _verify_release(service, snapshot, directory):
    """Incomplete or future gate receipts fail closed; no boolean-only release."""
    path = service.output_dir / "support_fill_release.json"
    _require(path.is_file(), "补缺候选尚无正式训练放行回执")
    release = json.loads(path.read_text(encoding="utf-8"))
    reference = dict(path=str((Path(directory) / "snapshot.json").resolve()),
                     sha256=file_hash(Path(directory) / "snapshot.json"))
    _require(release.get("schema") == RELEASE_SCHEMA and release.get("snapshot_id") == snapshot["snapshot_id"]
             and release.get("snapshot_sha256") == reference["sha256"] and release.get("training_released") is True,
             "补缺训练放行身份不符或仍未放行")
    data = _checked_reference(release.get("data_audit"))
    prediction = _checked_reference(release.get("prediction_audit"))
    _require(data.get("schema") == "paper1.support_fill_data_audit.1"
             and data.get("status") == "data_passed_prediction_pending" and data.get("data_gate_passed") is True
             and data.get("checked_records") == 450 and data.get("failures") == []
             and data.get("candidate_snapshot") == reference, "独立补缺数据门禁未通过或指向其他快照")
    parent_reference = snapshot["source_snapshot"]["support_fill_derivation"]["parent_snapshot"]
    parent_reference = {key: parent_reference[key] for key in ("path", "sha256")}
    _require(data.get("parent_snapshot") == parent_reference, "独立补缺数据门禁父快照不一致")
    _require(prediction.get("schema") == "paper1.support_fill_prediction_audit.1"
             and prediction.get("status") == "passed" and prediction.get("passed") is True
             and prediction.get("checked_records") == 450 and prediction.get("failures") == []
             and prediction.get("candidate_snapshot") == reference and prediction.get("parent_snapshot") == parent_reference
             and prediction.get("data_audit") == release["data_audit"], "冻结模型逐点门禁未通过或证据链不一致")
    for field, schema in (("window_audit", "paper1.prospective_window_audit.1"),
                          ("execution_recheck", "paper1.load_execution_recheck.1")):
        evidence = _checked_reference(release.get(field))
        _require(evidence.get("schema") == schema and evidence.get("status") == "passed"
                 and evidence.get("passed") is True and evidence.get("candidate_snapshot") == reference,
                 f"正式训练缺少通过的{field}门禁")
    return release


def verify_support_fill_snapshot(service, snapshot, directory, *, require_release=True, cancel=None, loads=None):
    """Verify immutable parent evidence and each candidate/source pair in order."""
    from .training_service import TrainingService
    directory = Path(directory)
    _require(snapshot.get("snapshot_id") == digest({key: value for key, value in snapshot.items() if key != "snapshot_id"}),
             "补缺候选快照内容身份不符")
    link = snapshot["source_snapshot"].get("support_fill_derivation", {})
    _require(link.get("schema") in (SCHEMA, ENDPOINT_SCHEMA) and link.get("rules_version") == SUPPORT_FILL_RULES_VERSION
             and link.get("implementation_sha256") == implementation_identity(), "补缺派生规则或实现身份改变")
    endpoint = link.get("schema") == ENDPOINT_SCHEMA
    _require(link.get("endpoint_policy") == ENDPOINT_POLICY if endpoint else not link.get("endpoint_policy"),
             "补缺端点策略与派生schema不一致")
    if require_release:
        _verify_release(service, snapshot, directory)
    resolver = ArtifactResolver(link.get("artifact_relocations"))
    parent_service = TrainingService(resolver.resolve(link["parent_service_output"]))
    parent_reference = link["parent_snapshot"]
    parent_path = resolver.resolve(parent_reference["path"])
    _require(parent_path == parent_service.output_dir / "snapshots" / parent_reference["snapshot_id"] / "snapshot.json",
             "补缺父快照路径与服务/ID不一致")
    _require(file_hash(parent_path) == parent_reference["sha256"], "补缺父快照文件身份改变")
    parent = _snapshot(parent_path)
    _require(parent["snapshot_id"] == parent_reference["snapshot_id"], "补缺父快照ID不符")
    historical = resolver.resolve(link["historical_route_source"])
    _require(link.get("parent_historical_consumer_source") == parent["source_snapshot"]["load_route_derivation"]["historical_consumer_source"],
             "补缺改变了父基础标签消费者源码声明")
    _verify_parent_chain(parent_service, parent, parent_path.parent, str(historical))
    _require({key: value for key, value in snapshot["source_snapshot"].items() if key != "support_fill_derivation"}
             == parent["source_snapshot"], "补缺改变了父来源、标签证据或全局规则声明")
    previous = _development_entries(parent)
    current = _development_entries(snapshot, load_only=True)
    _require([item["metadata"]["record_id"] for item in current] == [item["metadata"]["record_id"] for item in previous],
             "补缺候选记录成员或顺序改变")
    in_memory = list(loads) if loads is not None else None
    if in_memory is not None:
        _require([sample.metadata.get("record_id") for sample in in_memory]
                 == [item["metadata"]["record_id"] for item in current], "内存补缺样本成员或顺序与快照不一致")
    bindings = []
    for number, (old_item, new_item) in enumerate(zip(previous, current)):
        checkpoint(cancel)
        old = SampleSet(_frame(parent_path.parent, old_item), old_item["metadata"])
        source = resolver.resolve(old.metadata["resolved_files"]["process"])
        expected = (checked_endpoint_source(old, source, directory, new_item["metadata"])[0] if endpoint
                    else fill_sample_from_process_source(old, source))
        actual = _frame(directory, new_item)
        _require(new_item["metadata"] == expected.metadata, "补缺改变了原元数据或来源回执")
        try:
            pd.testing.assert_frame_equal(actual, expected.frame, check_exact=True)
        except AssertionError as exc:
            raise ValueError("补缺候选改变了父保护列或不符合有来源补缺结果") from exc
        if in_memory is not None:
            memory_sample = in_memory[number]
            _require(memory_sample.metadata == new_item["metadata"], "内存补缺样本元数据与快照不一致")
            try:
                pd.testing.assert_frame_equal(memory_sample.frame, actual, check_exact=True)
            except AssertionError as exc:
                raise ValueError("内存补缺样本数值与已验证CSV不一致") from exc
        binding = dict(record_id=old.metadata["record_id"], parent_csv_sha256=old_item["sha256"],
            candidate_csv_sha256=new_item["sha256"], source_sha256=old.metadata["file_sha256"]["process"])
        if endpoint:
            binding["normalized_source"] = expected.metadata["support_fill"]["source_normalization"]["normalized_source"]
        bindings.append(binding)
    _require(bindings == link.get("record_bindings"), "补缺逐记录身份绑定不一致")
    _require(file_hash(parent_path) == parent_reference["sha256"], "补缺父快照在验证期间改变")
    return dict(verified=True, checked_records=len(bindings), training_released=require_release,
                parent_snapshot_id=parent["snapshot_id"])


def derive_support_fill(service, *, parent_service_output, parent_snapshot_id, historical_route_source,
                        artifact_relocations=None, endpoint_policy=None, execute=False, cancel=None, progress=None):
    """Stream a new load-only snapshot; no labels, source evidence or model is signed anew."""
    _require(endpoint_policy in (None, ENDPOINT_POLICY), "未登记的端点规范化策略")
    if not execute:
        return dict(status="planned_not_executed", rules_version=SUPPORT_FILL_RULES_VERSION,
                    training_released=False, training_started=False)
    from .training_service import TrainingService
    resolver = ArtifactResolver(artifact_relocations)
    parent_service = TrainingService(resolver.resolve(parent_service_output))
    output = service.output_dir.resolve()
    _require(not output.is_relative_to(parent_service.output_dir) and not parent_service.output_dir.is_relative_to(output),
             "补缺服务必须位于独立输出目录")
    _require(not output.exists() or not any(output.iterdir()), "补缺输出目录已存在内容，拒绝覆盖")
    parent_path = parent_service.output_dir / "snapshots" / parent_snapshot_id / "snapshot.json"
    parent_hash = file_hash(parent_path)
    parent = _snapshot(parent_path)
    _require(parent["snapshot_id"] == parent_snapshot_id, "请求父快照ID与内容不符")
    historical = resolver.resolve(historical_route_source)
    _verify_parent_chain(parent_service, parent, parent_path.parent, str(historical))
    entries = _development_entries(parent)
    directory = output / "snapshots" / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    records, bindings = [], []
    for number, item in enumerate(entries):
        checkpoint(cancel)
        sample = SampleSet(_frame(parent_path.parent, item), item["metadata"])
        source_path = resolver.resolve(sample.metadata["resolved_files"]["process"])
        if endpoint_policy:
            derived, _, content = endpoint_normalized_sample(sample, source_path)
            normalized_ref = derived.metadata["support_fill"]["source_normalization"]["normalized_source"]
            normalized_path = _inside(directory, normalized_ref["file"])
            normalized_path.parent.mkdir(exist_ok=True)
            if normalized_path.exists():
                _require(file_hash(normalized_path) == normalized_ref["sha256"], "共享规范化来源文件身份冲突")
            else:
                normalized_path.write_bytes(content)
        else:
            derived = fill_sample_from_process_source(sample, source_path)
        name = f"load_{number:04d}.csv"
        derived.frame.to_csv(directory / name, index=False, encoding="utf-8")
        checksum = file_hash(directory / name)
        records.append(dict(kind="load", file=name, sha256=checksum, metadata=derived.metadata,
                            dtypes={key: str(value) for key, value in derived.frame.dtypes.items()}))
        binding = dict(record_id=sample.metadata["record_id"], parent_csv_sha256=item["sha256"],
            candidate_csv_sha256=checksum, source_sha256=sample.metadata["file_sha256"]["process"])
        if endpoint_policy:
            binding["normalized_source"] = normalized_ref
        bindings.append(binding)
        if progress:
            progress(dict(stage="support_fill", record_id=sample.metadata["record_id"], completed=number + 1,
                          total=len(entries), restored_points=derived.metadata["support_fill"]["restored_points"]))
    _require(file_hash(parent_path) == parent_hash, "父快照在补缺派生期间改变")
    source = deepcopy(parent["source_snapshot"])
    source["support_fill_derivation"] = dict(schema=ENDPOINT_SCHEMA if endpoint_policy else SCHEMA, rules_version=SUPPORT_FILL_RULES_VERSION,
        parent_service_output=str(parent_service.output_dir),
        parent_snapshot=dict(path=str(parent_path), snapshot_id=parent_snapshot_id, sha256=parent_hash),
        historical_route_source=str(historical),
        parent_historical_consumer_source=source["load_route_derivation"]["historical_consumer_source"],
        artifact_relocations=resolver.reference, implementation_sha256=implementation_identity(), record_bindings=bindings)
    if endpoint_policy:
        source["support_fill_derivation"]["endpoint_policy"] = endpoint_policy
    result = service._finalize_prepared_snapshot(directory, source, records, synthetic=False,
        validator=lambda manifest, location: verify_support_fill_snapshot(service, manifest, location,
                                                                         require_release=False, cancel=cancel))
    return {**result, "status": "derived_data_verified_release_pending", "training_released": False,
            "training_started": False, "fixed_weight_prediction_audit_required": True,
            "independent_data_audit_required": True}
