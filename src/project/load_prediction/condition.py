"""Training-only physical descriptor encoding; identities never enter a network.

Records may be descriptor dictionaries or metadata containing
``condition_descriptors``. Encoded arrays end in ``encoded_dim``; missing
values use training centers and an additional missingness indicator.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .common import version_of

BRANCH_FIELDS = {
    "machine": ("category", "rated_power_w", "rated_torque_nm", "torque_speed_characteristic",
                "transmission", "drive_type", "stiffness_proxy", "power_channel", "calibration_version"),
    "tool": ("material", "coating", "type", "rake_deg", "clearance_deg", "helix_deg", "edge_type",
             "overhang_mm", "holder", "connection"),
    "workpiece": ("material_family", "grade", "hardness", "heat_treatment", "source", "state",
                  "strength", "thermal_conductivity"),
    "clamping": ("fixture_type", "overhang_mm", "clamp_position", "stiffness_proxy"),
    "cooling": ("mode", "fluid", "pressure", "flow"),
}
# Version/source fields document provenance; never treat them as predictors.
PROVENANCE_FIELDS = {("machine", "calibration_version"), ("workpiece", "source")}
SCHEMA = "afc.condition.1"


def descriptors(record):
    record = getattr(record, "metadata", record)
    if not isinstance(record, dict):
        raise ValueError("工况记录必须为字典")
    source = record.get("condition_descriptors", record.get("condition", record))
    if not isinstance(source, dict):
        raise ValueError("condition_descriptors必须为物理描述字典")
    source = dict(source)
    if "material" in source:
        if "workpiece" in source and source["workpiece"] != source["material"]:
            raise ValueError("material与workpiece物理描述冲突")
        source["workpiece"] = source.pop("material")
    result = {}
    for branch, fields in BRANCH_FIELDS.items():
        values = source.get(branch, {})
        if values is None:
            values = {}
        if not isinstance(values, dict):
            raise ValueError(f"{branch}需要物理描述，不能使用工况ID")
        extra = set(values) - set(fields)
        if extra:
            raise ValueError(f"{branch}含未声明物理字段（不允许ID特征）：{sorted(extra)}")
        result[branch] = {key: value for key, value in values.items()
                          if value is not None and not (isinstance(value, str) and value == "")}
    return result


def _numeric(value):
    if isinstance(value, bool) or isinstance(value, (str, dict)):
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim > 2 or not array.size or not np.isfinite(array).all():
        raise ValueError("连续工况描述必须为有限标量或固定维数数值数组")
    return array.reshape(-1)


def fit_condition_spec(records, *, reference=None):
    """Fit centers/vocabularies on training records only; freeze single levels."""
    records = [descriptors(record) for record in records]
    if not records:
        raise ValueError("工况编码需要非空训练记录")
    reference = descriptors(reference) if reference is not None else records[0]
    branches, offset = {}, 0
    for branch, names in BRANCH_FIELDS.items():
        fields, varying = [], False
        begin = offset
        for name in names:
            if (branch, name) in PROVENANCE_FIELDS:
                continue
            observed = [row[branch][name] for row in records if name in row[branch]]
            item = {"name": name, "offset": offset}
            if not observed:
                item.update(kind="unobserved", width=0, center=[])
            elif all(isinstance(value, str) for value in observed):
                counts = Counter(observed)
                levels = sorted(counts)
                center = [counts[value] / len(observed) for value in levels]
                item.update(kind="categorical", levels=levels, width=len(levels), center=center)
                varying |= len(levels) > 1
            else:
                arrays = [_numeric(value) for value in observed]
                if any(value is None for value in arrays) or len({len(value) for value in arrays}) != 1:
                    raise ValueError(f"{branch}.{name}混合类型或数组维度变化")
                matrix = np.stack(arrays)
                scale = matrix.std(axis=0)
                item.update(kind="numeric", width=matrix.shape[1], center=matrix.mean(axis=0).tolist(),
                            scale=np.where(scale > 1e-8, scale, 1.).tolist(),
                            minimum=matrix.min(axis=0).tolist(), maximum=matrix.max(axis=0).tolist())
                varying |= bool(np.any(np.ptp(matrix, axis=0) > 1e-8))
            offset += item["width"] + 1
            fields.append(item)
        branches[branch] = {"offset": begin, "width": offset - begin, "active": bool(varying), "fields": fields}
    spec = {"schema": SCHEMA, "branches": branches, "encoded_dim": offset,
            "reference_descriptors": reference, "training_records": len(records),
            "interactions": False, "provenance_fields_excluded": [list(x) for x in sorted(PROVENANCE_FIELDS)]}
    spec["reference_encoded"] = encode_conditions([reference], spec)[0].tolist()
    encoded = encode_conditions(records, spec)
    active = [name for name, branch in branches.items() if branch["active"]]
    confounded = []
    for left_index, left in enumerate(active):
        for right in active[left_index + 1:]:
            a, b = branches[left], branches[right]
            left_rows = [tuple(x[a["offset"]:a["offset"] + a["width"]]) for x in encoded]
            right_rows = [tuple(x[b["offset"]:b["offset"] + b["width"]]) for x in encoded]
            pairs = set(zip(left_rows, right_rows))
            if len(pairs) == len(set(left_rows)) == len(set(right_rows)):
                confounded.append([left, right])
    spec["confounded_pairs"] = confounded
    spec["version"] = version_of(spec)
    return spec


def validate_condition_spec(spec):
    if not isinstance(spec, dict) or spec.get("schema") != SCHEMA or set(spec.get("branches", {})) != set(BRANCH_FIELDS):
        raise ValueError("工况描述schema不匹配")
    if spec.get("interactions") is not False:
        raise ValueError("当前工况编码不启用交互项")
    if "version" in spec and spec["version"] != version_of({k: v for k, v in spec.items() if k != "version"}):
        raise ValueError("工况编码版本与内容不一致")
    dimension = spec.get("encoded_dim")
    if type(dimension) is not int or dimension < 1:
        raise ValueError("工况编码维数无效")
    if "reference_encoded" in spec:
        values = np.asarray(spec["reference_encoded"], dtype=float)
        if values.shape != (dimension,) or not np.isfinite(values).all():
            raise ValueError("参考工况编码无效")
    return spec


def encode_conditions(records, spec, *, allow_unknown=False):
    """Return N×encoded_dim; only explicit low-dimensional calibration may
    represent unseen fields/categories by the training center plus missing mask.
    """
    validate_condition_spec(spec)
    if spec.get("encoding_mode") == "identity_only_diagnostic":
        from .comparison_models import encode_identity_conditions
        return encode_identity_conditions(records, spec)
    rows = [descriptors(record) for record in records]
    result = np.zeros((len(rows), spec["encoded_dim"]), dtype=np.float32)
    for row_index, row in enumerate(rows):
        for branch, branch_spec in spec["branches"].items():
            for field in branch_spec["fields"]:
                name, width, start = field["name"], field["width"], field["offset"]
                value = row[branch].get(name)
                missing = value is None
                result[row_index, start + width] = float(missing)
                if field["kind"] == "unobserved":
                    if not missing:
                        if not allow_unknown:
                            raise ValueError(f"{branch}.{name}未在训练中观察，需验证或校准新工况")
                        result[row_index, start + width] = 1.
                elif missing:
                    if field["kind"] == "categorical":
                        result[row_index, start:start + width] = field["center"]
                elif field["kind"] == "numeric":
                    numeric = _numeric(value)
                    if numeric is None or numeric.shape != (width,):
                        raise ValueError(f"{branch}.{name}维数或类型不匹配")
                    result[row_index, start:start + width] = (numeric - field["center"]) / field["scale"]
                elif field["kind"] == "categorical":
                    if value not in field["levels"]:
                        if not allow_unknown:
                            raise ValueError(f"{branch}.{name}类别未在训练中出现：{value}")
                        result[row_index, start:start + width] = field["center"]
                        result[row_index, start + width] = 1.
                    else:
                        result[row_index, start + field["levels"].index(value)] = 1.
    return result


def condition_coverage(spec):
    validate_condition_spec(spec)
    return {name: {"active": branch["active"], "status": "learned_levels" if branch["active"] else "fixed_reference",
                   "interpretation": "combination_only" if any(name in pair for pair in spec.get("confounded_pairs", [])) else "requires_matched_external_validation",
                   "unobserved": [f["name"] for f in branch["fields"] if f["kind"] == "unobserved"]}
            for name, branch in spec["branches"].items()}
