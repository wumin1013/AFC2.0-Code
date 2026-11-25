"""Bind derived F/S process files to reviewed canonical inputs without rewriting provenance."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import file_hash
from .experiment_protocol import PARAMETER_GROUPS, PROGRAM_FEATURES, validate_round_metadata
from .feed import FeedPlan
from .samples import read_process_table

BINDING_SCHEMA = "paper1.process_library_binding.1"
GEOMETRY_COLUMNS = ["line", "execution_instance", "segment_mm", "segment_start_mm", "nc_path_mm",
    "ap_mm", "ae_mm", "source_point_index", "source_path_mm", "source_nc_line", "source_file_line",
    "point_index_in_line", "points_in_line", "mapping_status"]


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _meta(record):
    return {**record.get("metadata", {}), **{k: v for k, v in record.items() if k != "metadata"}}


def _active(records):
    return [r for r in map(_meta, records) if r.get("status", "active") == "active"
            and r.get("dataset_role") == "development"]


def _resolve(path, base):
    path = Path(path)
    return (path if path.is_absolute() else Path(base) / path).resolve()


def _check(path, expected, hashes):
    path = Path(path).resolve()
    if str(path) not in hashes:
        before = path.stat()
        actual = file_hash(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"工艺绑定期间文件改变：{path}")
        hashes[str(path)] = actual
    if hashes[str(path)] != expected:
        raise ValueError(f"工艺绑定来源哈希不符：{path}")
    return path


def relocated_process_path(value, process_root):
    parts = Path(value).parts
    if not parts or parts[0] != "工艺信息" or any(p in ("..", ".") for p in parts):
        raise ValueError("工艺库路径必须以原工艺信息包装目录开头且不含上级路径")
    root = Path(process_root).resolve()
    result = root.joinpath(*parts[1:]).resolve()
    if not result.is_relative_to(root) or result.suffix.lower() != ".txt":
        raise ValueError("派生工艺必须是当前工艺根目录内的TXT")
    return result


def build_process_library_binding(records, *, manifest_base, library_manifest, process_root):
    """Use the production reader to establish exact equivalence; do not rerun segmentation."""
    library_path = Path(library_manifest).resolve()
    library = _read(library_path)
    entries = library.get("records", [])
    if library.get("schema") != "paper1.fs_process_library.1" or len(entries) != 54:
        raise ValueError("必须提供已生成的54组合工艺库")
    if Counter(e["parameter_group"] for e in entries) != Counter({g: 9 for g in PARAMETER_GROUPS}):
        raise ValueError("工艺库六参数组各需九路径")
    active = _active(records)
    expected = {(r, f) for r in range(1, 91) for f in PROGRAM_FEATURES}
    if len(active) != 450 or {(r["round_id"], r["feature"]) for r in active} != expected:
        raise ValueError("工艺绑定要求450个唯一活动位置")
    checked_rounds = validate_round_metadata(active)
    if not checked_rounds["valid"]:
        raise ValueError(str(checked_rounds["issues"]))
    hashes = {str(library_path): file_hash(library_path)}
    combinations = {}
    for item in entries:
        group = item["parameter_group"]
        spec = PARAMETER_GROUPS[group]
        if item["feed_mm_min"] != spec["feed_program_mm_min"] or item["spindle_rpm"] != spec["spindle_cmd_rpm"]:
            raise ValueError("工艺库F/S与明确参数组不一致")
        if "O1203" in str(item).upper():
            raise ValueError("O1203不得进入工艺绑定")
        nc = _check(item["nc_file"], item["nc_sha256"], hashes)
        source = _check(item["source_process_file"], item["source_process_sha256"], hashes)
        if item["feature"] == "外形" and source.name != "外形_工艺信息文件.txt":
            raise ValueError("外形基础工艺来源不符")
        derived = _check(relocated_process_path(item["process_file"], process_root), item["process_sha256"], hashes)
        canonical_path = _check(item["canonical_process"], item["canonical_sha256"], hashes)
        evidence_path = _check(item["steady_evidence"], item["steady_evidence_sha256"], hashes)
        evidence = _read(evidence_path)
        candidate_path = _check(Path(evidence["candidate_geometry_source"]) / "candidate.csv",
                                evidence["candidate_geometry_sha256"], hashes)
        grid = pd.read_csv(candidate_path, float_precision="round_trip", usecols=["line", "execution_instance",
            "segment_mm", "path_mm", "block_length_mm", "feed_program_mm_min"])
        plan = FeedPlan(grid, evidence["program_version"], "process_binding_geometry", False, {})
        reread = read_process_table(derived, plan, line_offset=item["line_offset"])
        canonical = pd.read_csv(canonical_path, float_precision="round_trip")
        pd.testing.assert_frame_equal(reread[GEOMETRY_COLUMNS], canonical[GEOMETRY_COLUMNS], check_exact=True)
        if len(reread) != item["points"] or len(reread) != evidence["raw_process_points"]:
            raise ValueError("工艺点数改变")
        np.testing.assert_array_equal(reread.source_feed_mm_min, np.full(len(reread), item["feed_mm_min"]))
        np.testing.assert_array_equal(reread.source_spindle_rpm, np.full(len(reread), item["spindle_rpm"]))
        mapped = canonical.execution_instance >= 0
        if not canonical.loc[mapped, "feed_program_mm_min"].eq(item["feed_mm_min"]).all():
            raise ValueError("既有稳态输入的编程F不等价")
        # Exact ap/ae and NC F equality also proves contact and recalculated MRR equality.
        key = group + ":" + item["nc_sha256"]
        if key in combinations:
            raise ValueError("工艺库存在重复NC参数组合")
        combinations[key] = {**item, "process_file": str(derived), "nc_file": str(nc),
            "candidate_geometry": str(candidate_path), "candidate_geometry_sha256": file_hash(candidate_path),
            "geometry_point_order_contact_exact": True, "steady_algorithm_inputs_equivalent": True,
            "canonical_source_columns_preserved": True}
    assignments = []
    for record in active:
        nc = _resolve(record["nc"], manifest_base)
        key = record["parameter_group"] + ":" + file_hash(nc)
        item = combinations.get(key)
        if item is None or item["feature"] != record["feature"]:
            raise ValueError(f"记录无唯一工艺组合：{record['record_id']}")
        pairs = [(nc, item["nc_file"]), (_resolve(record["process"], manifest_base), item["canonical_process"]),
                 (_resolve(record["source_process"]["path"], manifest_base), item["source_process_file"]),
                 (_resolve(record["segmentation_evidence_path"], manifest_base), item["steady_evidence"])]
        if any(Path(a).resolve() != Path(b).resolve() for a, b in pairs):
            raise ValueError(f"记录审核链与工艺库不一致：{record['record_id']}")
        assignments.append({"record_id": record["record_id"], "round_id": record["round_id"],
            "feature": record["feature"], "combination_key": key,
            "process_txt": item["process_file"], "process_txt_sha256": item["process_sha256"]})
    for path, expected_hash in hashes.items():
        if file_hash(path) != expected_hash:
            raise ValueError(f"等价验证期间来源改变：{path}")
    return {"schema": BINDING_SCHEMA, "library_manifest": str(library_path),
        "library_manifest_sha256": hashes[str(library_path)], "combinations": combinations,
        "assignments": assignments, "source_hashes": hashes, "active_records": len(assignments),
        "combinations_checked": len(combinations), "process_points": sum(e["points"] for e in entries),
        "steady_results_reused": True, "raw_measurements_read": False, "training_started": False}


def attach_process_binding(records, receipt_path):
    receipt_path = Path(receipt_path).resolve()
    receipt = _read(receipt_path)
    refs = {a["record_id"]: {**a, "path": str(receipt_path), "sha256": file_hash(receipt_path)}
            for a in receipt["assignments"]}
    result = deepcopy(records)
    for record in result:
        meta = _meta(record)
        if meta.get("status", "active") == "active" and meta.get("dataset_role") == "development":
            record.get("metadata", record)["process_library_binding"] = refs[meta["record_id"]]
    return result


def verify_process_library_binding(records, *, manifest_base, reference, artifact_relocations=None):
    from .artifact_paths import ArtifactResolver
    resolver = ArtifactResolver(artifact_relocations, base_dir=manifest_base)
    hashes = {}
    path = _check(resolver.resolve(reference["path"]), reference["sha256"], hashes)
    receipt = _read(path)
    if receipt.get("schema") != BINDING_SCHEMA:
        raise ValueError("工艺绑定回执版本不符")
    for source, expected in receipt["source_hashes"].items():
        _check(resolver.resolve(source), expected, hashes)
    expected_records = attach_process_binding(records, path)
    active = _active(records)
    if {r["record_id"] for r in active} != {a["record_id"] for a in receipt["assignments"]}:
        raise ValueError("工艺绑定活动成员不符")
    expected_refs = {r["record_id"]: r["process_library_binding"] for r in _active(expected_records)}
    for record in active:
        ref = record.get("process_library_binding")
        normalized_ref = deepcopy(ref)
        if isinstance(normalized_ref, dict) and normalized_ref.get("path"):
            normalized_ref["path"] = str(resolver.resolve(normalized_ref["path"]))
        if normalized_ref != expected_refs[record["record_id"]]:
            raise ValueError(f"记录工艺绑定改变：{record['record_id']}")
        item = receipt["combinations"][ref["combination_key"]]
        if (record["parameter_group"] != item["parameter_group"] or record["feature"] != item["feature"]
                or resolver.resolve(record["nc"]) != resolver.resolve(item["nc_file"])
                or resolver.resolve(record["process"]) != resolver.resolve(item["canonical_process"])):
            raise ValueError(f"当前记录与工艺组合身份不符：{record['record_id']}")
    return {"schema": BINDING_SCHEMA, "records_verified": len(active),
        "combinations_verified": len(receipt["combinations"]), "files_verified": len(hashes),
        "receipt_sha256": reference["sha256"]}
