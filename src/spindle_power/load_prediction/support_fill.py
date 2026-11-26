"""Derive missing process inputs from explicit source cells, preserving supervision.

This opt-in derivation never rebuilds labels, masks, or sequence boundaries. The
caller must still compare fixed-weight predictions and their original contexts
before release. No legacy mapping or model-package version is changed here.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


SUPPORT_FILL_RULES_VERSION = "afc.support_fill.1"
MUTABLE_COLUMNS = ("ap_mm", "ae_mm", "process_geometry_known", "cutting_contact")
SOURCE_COLUMNS = {
    "support_fill_source_point_index": "source_point_index",
    "support_fill_source_file_line": "source_file_line",
    "support_fill_interval_start_mm": "segment_start_mm",
    "support_fill_interval_end_mm": "segment_mm",
}
ADDED_COLUMNS = ("support_fill_restored", *SOURCE_COLUMNS)
GRID_COLUMNS = ("line", "execution_instance", "segment_mm", "path_mm")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _boolean(frame, column):
    values = frame[column]
    _require(pd.api.types.is_bool_dtype(values.dtype) and not values.isna().any(),
             f"补缺需要完整布尔列：{column}")
    return values.to_numpy(dtype=bool)


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def fill_missing_process_support(parent, process, *, diameter_mm):
    """Preserve the historical supervised-snapshot contract."""
    required = {"steady", "sequence_id", "training_sequence_id", "power_cut_w", "valid_main", "valid_residual"}
    _require(required <= set(parent), f"父样本缺少字段：{sorted(required - set(parent))}")
    return fill_missing_geometry_support(parent, process, diameter_mm=diameter_mm)


def fill_missing_geometry_support(parent, process, *, diameter_mm):
    """Return ``(new_frame, receipt)`` without modifying either input frame.

    Only originally unknown geometry with a unique valid source cell is filled.
    Cells own exactly ``(start, end]``; the first cell also owns exact NC zero
    when its start is zero. No tolerance expands a cell across a real gap.
    Source row identifiers are mandatory for every restored point.
    """
    required = set(GRID_COLUMNS) | set(MUTABLE_COLUMNS) | {"process_geometry_conflict_support"}
    _require(required <= set(parent), f"父样本缺少字段：{sorted(required - set(parent))}")
    _require(parent.columns.is_unique and process.columns.is_unique, "样本或工艺源存在重复列名")
    _require(not set(ADDED_COLUMNS).intersection(parent), "父样本已含补缺字段，拒绝重复派生")
    source_required = {"line", "execution_instance", "segment_mm", "ap_mm", "ae_mm"}
    _require(source_required <= set(process), f"工艺源缺少字段：{sorted(source_required - set(process))}")
    diameter = float(diameter_mm)
    _require(np.isfinite(diameter) and diameter > 0, "补缺需要有限正刀具直径")
    grid = parent[list(GRID_COLUMNS)].to_numpy(float)
    _require(np.isfinite(grid).all() and not parent.duplicated(list(GRID_COLUMNS)).any(),
             "父NC网格必须完整、有限且唯一")
    _require(np.all(grid[:, :2] == np.round(grid[:, :2])) and np.all(grid[:, 1:] >= 0),
             "父NC行号、访问实例或行程无效")
    if "diameter_mm" in parent:
        _require(parent.diameter_mm.eq(diameter).all(), "父样本刀具直径与补缺声明不一致")
    known = _boolean(parent, "process_geometry_known")
    _boolean(parent, "cutting_contact")
    conflict = _boolean(parent, "process_geometry_conflict_support")
    geometry = parent[["ap_mm", "ae_mm"]].to_numpy(float)
    _require(np.array_equal(known, np.isfinite(geometry).all(axis=1)), "父几何支持声明与原值不一致")
    _require(np.isnan(geometry[~known]).all(), "未知父几何须为成对NaN，不能覆盖部分已知值或无穷值")

    result = parent.copy(deep=True)
    count = len(parent)
    restored = np.zeros(count, dtype=bool)
    reasons = np.full(count, "no_process_row", dtype=object)
    sources = {column: np.full(count, np.nan) for column in SOURCE_COLUMNS}
    for (line, instance), indices in parent.groupby(["line", "execution_instance"], sort=False).indices.items():
        indices = np.asarray(indices, dtype=int)
        rows = process.loc[(process.line == line) & (process.execution_instance == instance)]
        if rows.empty:
            continue
        if "segment_start_mm" not in rows:
            reasons[indices] = "no_declared_source_interval"
            continue
        rows = rows.sort_values("segment_mm", kind="stable")
        starts, ends = rows.segment_start_mm.to_numpy(float), rows.segment_mm.to_numpy(float)
        _require(np.isfinite(starts).all() and np.isfinite(ends).all() and np.all(starts >= 0)
                 and np.all(ends > starts) and np.all(starts[1:] >= ends[:-1]),
                 f"工艺源区间非法或重叠：line={line}, execution_instance={instance}")
        positions = parent.iloc[indices].segment_mm.to_numpy(float)
        owners = np.clip(np.searchsorted(ends, positions, side="left"), 0, len(ends) - 1)
        inside = (positions > starts[owners]) & (positions <= ends[owners])
        origin = (starts[0] == 0.) & (positions == 0.)
        owners[origin], inside[origin] = 0, True
        values = rows[["ap_mm", "ae_mm"]].to_numpy(float)
        legal = np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1) & (values[:, 1] <= diameter)
        supported = inside & legal[owners]
        reasons[indices] = np.where(~inside, "outside_or_missing_source_interval",
                                   np.where(legal[owners], "source_supported", "invalid_source_geometry"))
        use = ~known[indices] & ~conflict[indices] & supported
        selected = indices[use]
        if not len(selected):
            continue
        source_rows = rows.iloc[owners[use]]
        for column in ("source_point_index", "source_file_line"):
            _require(column in source_rows, f"可补工艺区间缺少来源标识：{column}")
            identity = source_rows[column].to_numpy(float)
            minimum = 1 if column == "source_file_line" else 0
            _require(np.isfinite(identity).all() and np.all(identity == np.round(identity))
                     and np.all(identity >= minimum), f"工艺来源标识无效：{column}")
        for column in ("ap_mm", "ae_mm"):
            result.iloc[selected, result.columns.get_loc(column)] = source_rows[column].to_numpy(float)
        result.iloc[selected, result.columns.get_loc("process_geometry_known")] = True
        result.iloc[selected, result.columns.get_loc("cutting_contact")] = (values[owners[use]] > 0).all(axis=1)
        restored[selected] = True
        for output_column, source_column in SOURCE_COLUMNS.items():
            sources[output_column][selected] = source_rows[source_column].to_numpy(float)
    reasons[conflict] = "preserved_source_conflict"
    result["support_fill_restored"] = restored
    for column, values in sources.items():
        result[column] = values
    remaining = ~known & ~restored
    binding_columns = [*GRID_COLUMNS, "ap_mm", "ae_mm", *SOURCE_COLUMNS]
    binding = result.loc[restored, binding_columns].to_dict("list")
    binding["parent_row_position"] = np.flatnonzero(restored).tolist()
    receipt = dict(schema="afc.support_fill_record.1", rules_version=SUPPORT_FILL_RULES_VERSION,
        interval_convention="(start,end]; exact_zero_uses_first_cell; no_extrapolation",
        mutable_columns=list(MUTABLE_COLUMNS), added_columns=list(ADDED_COLUMNS),
        grid_points=count, old_known_points=int(known.sum()), restored_points=int(restored.sum()),
        remaining_missing_points=int(remaining.sum()),
        remaining_missing_reasons=dict(Counter(reasons[remaining].tolist())),
        restored_source_binding_sha256=_digest(binding),
        parent_columns_preserved_except_declared_fill=True, labels_and_masks_preserved=True,
        sequence_columns_preserved=[name for name in ("sequence_id", "training_sequence_id", "nc_sequence_id") if name in parent],
        source_file_verified=False, fixed_weight_prediction_audit_required=True, training_released=False)
    return result, receipt


def fill_sample_from_process_source(sample, process_path):
    """Verify the parent's registered process bytes and return a derived SampleSet.

    ``process_path`` may be resolved through an existing relocation receipt.
    Original metadata, including prepared_measurement_sha256, stays unchanged;
    only the independent support_fill receipt is added.
    """
    from .common import file_hash
    from .samples import SampleSet, read_process_table

    _require("support_fill" not in sample.metadata, "父样本已有补缺回执，拒绝重复派生")
    meta = deepcopy(sample.metadata)
    expected = meta.get("file_sha256", {}).get("process")
    _require(isinstance(expected, str) and len(expected) == 64, "父样本缺少工艺源SHA-256")
    path = Path(process_path).resolve()
    _require(file_hash(path) == expected, "工艺源文件与父登记SHA-256不符")
    plan = SimpleNamespace(frame=sample.frame, program_version=meta["program_version"],
                           machine_version=meta["machine_version"])
    process = read_process_table(path, plan, line_offset=meta.get("process_line_offset", 0),
                                 steady_lines=meta.get("steady_lines", []))
    frame, receipt = fill_missing_process_support(sample.frame, process, diameter_mm=meta["diameter_mm"])
    _require(file_hash(path) == expected, "工艺源文件在补缺期间发生变化")
    receipt.update(source_file_verified=True, source=dict(path=str(path), sha256=expected,
        parent_registered_path=meta.get("resolved_files", {}).get("process")),
        record_id=meta.get("record_id"), implementation_sha256=file_hash(__file__))
    meta["support_fill"] = receipt
    return SampleSet(frame, meta)
