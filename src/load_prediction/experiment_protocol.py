"""Versioned experimental truth shared by all research entry points.

The explicit table preserves the executed R10 and the revised future rounds.
Identical NC programs may occur in different folds; measured records may not.
"""
from __future__ import annotations

from copy import deepcopy

PROTOCOL_VERSION = "afc.experiment.20261009.1"
DEVELOPMENT_ROUNDS = tuple(range(1, 91))
PROGRAM_FEATURES = ("外形", "曲面", "混合", "斜面", "台阶")
DATASET_ROLES = ("development", "memory_validation", "target_adaptation",
                 "adaptation_validation", "interval_calibration", "independent_test")
SPACING_MM = .5
VALIDATION_ROUNDS = {
    "S1T1": (1, 11, 26, 32, 40, 46, 55, 64, 73, 82),
    "S1T2": (7, 12, 27, 33, 38, 47, 56, 65, 74, 83),
    "S1T3": (4, 13, 25, 28, 42, 48, 57, 66, 75, 84),
    "S2T1": (6, 14, 20, 35, 44, 49, 58, 67, 76, 85),
    "S2T2": (3, 10, 21, 36, 39, 50, 59, 68, 77, 86),
    "S2T3": (9, 16, 22, 34, 37, 51, 60, 69, 78, 87),
    "S3T1": (8, 17, 23, 29, 43, 52, 61, 70, 79, 88),
    "S3T2": (5, 18, 24, 30, 41, 53, 62, 71, 80, 89),
    "S3T3": (2, 15, 19, 31, 45, 54, 63, 72, 81, 90),
}
ROUND_COMBINATIONS = {r: combo for combo, rounds in VALIDATION_ROUNDS.items() for r in rounds}
PARAMETER_GROUPS = {
    "P0": {"rounds": list(range(1, 46)), "feed_program_mm_min": 2000., "spindle_cmd_rpm": 8000.},
    "P1": {"rounds": list(range(46, 55)), "feed_program_mm_min": 1500., "spindle_cmd_rpm": 8000.},
    "P2": {"rounds": list(range(55, 64)), "feed_program_mm_min": 2500., "spindle_cmd_rpm": 8000.},
    "P3": {"rounds": list(range(64, 73)), "feed_program_mm_min": 2000., "spindle_cmd_rpm": 6000.},
    "P4": {"rounds": list(range(73, 82)), "feed_program_mm_min": 2000., "spindle_cmd_rpm": 10000.},
    "P5": {"rounds": list(range(82, 91)), "feed_program_mm_min": 2500., "spindle_cmd_rpm": 10000.},
}


def round_spec(round_id):
    if type(round_id) is not int or round_id not in ROUND_COMBINATIONS:
        raise ValueError("开发轮次必须为 R01—R90 的整数编号")
    group = next(key for key, value in PARAMETER_GROUPS.items() if round_id in value["rounds"])
    combo = ROUND_COMBINATIONS[round_id]
    return {"round_id": round_id, "combination": combo, "parameter_group": group,
            "feed_program_mm_min": PARAMETER_GROUPS[group]["feed_program_mm_min"],
            "spindle_cmd_rpm": PARAMETER_GROUPS[group]["spindle_cmd_rpm"],
            "program_slots": {"外形": "O0", "曲面": "C0", "混合": "M0", "斜面": combo[:2], "台阶": combo[2:]}}


def protocol_folds():
    return [{"fold": f"F{i}", "combination": combo, "validation_rounds": list(rounds),
             "train_rounds": [r for r in DEVELOPMENT_ROUNDS if r not in rounds],
             "protocol_version": PROTOCOL_VERSION}
            for i, (combo, rounds) in enumerate(VALIDATION_ROUNDS.items(), 1)]


def validate_round_metadata(records, *, require_complete=True):
    """Validate exact identity, not just the number of rounds in each fold."""
    issues, seen = [], set()
    for index, record in enumerate(records):
        meta = {**record.get("metadata", {}), **{k: v for k, v in record.items() if k != "metadata"}}
        if meta.get("dataset_role", "development") != "development":
            continue
        try:
            spec = round_spec(meta.get("round_id"))
        except ValueError as exc:
            issues.append({"code": "invalid_round", "record_index": index, "message": str(exc)})
            continue
        seen.add(spec["round_id"])
        for key in ("combination", "parameter_group"):
            if meta.get(key) != spec[key]:
                issues.append({"code": "round_mapping", "record_index": index, "round_id": spec["round_id"],
                               "message": f"R{spec['round_id']:02d} 的 {key} 应为 {spec[key]}"})
        feature = meta.get("feature")
        if "program_slot" in meta and feature in PROGRAM_FEATURES and meta["program_slot"] != spec["program_slots"][feature]:
            issues.append({"code": "program_slot", "record_index": index, "round_id": spec["round_id"],
                           "message": f"{feature} 程序位置与组合不匹配"})
    missing = sorted(set(DEVELOPMENT_ROUNDS) - seen)
    if require_complete and missing:
        issues.append({"code": "missing_rounds", "message": f"正式九折缺少开发轮次：{missing}", "rounds": missing})
    return {"valid": not issues, "issues": issues, "rounds": sorted(seen), "missing_rounds": missing,
            "protocol_version": PROTOCOL_VERSION}


def protocol_definition():
    return {"protocol_version": PROTOCOL_VERSION, "spacing_mm": SPACING_MM,
            "rounds": [round_spec(r) for r in DEVELOPMENT_ROUNDS], "folds": protocol_folds(),
            "roles": list(DATASET_ROLES), "parameter_groups": deepcopy(PARAMETER_GROUPS),
            "stages": ["feed", "freeze_feed", "main", "freeze_main", "residual", "freeze_residual", "evaluate_then_memory"],
            "independent_test_from_development": False}
