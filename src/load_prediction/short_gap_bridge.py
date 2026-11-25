"""Prediction-only, auditable connection of short gaps in NC geometry.

This module deliberately has no training, model, or measured-label dependency.
Call it after the original source mapper, never to create supervision labels.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import math
from numbers import Real


RULE_VERSION = "afc.prediction_short_gap_bridge.1"
MAX_ANCHOR_SPAN_MM = 0.05
MAPPING_METHOD = "short_gap_same_sequence_linear_interpolation"
POWER_FIELDS = {
    "base_power_w": "base_power_w",
    "cutting_power_w": "cutting_power_w",
    "total_power_w": "spindle_power_w",
    "total_lower_w": "spindle_lower_w",
    "total_upper_w": "spindle_upper_w",
}


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _identity(value):
    if _finite(value) and float(value).is_integer():
        return int(value)
    return None


def _sequence(value):
    if isinstance(value, str) and value:
        return value
    if _finite(value):
        return float(value)
    return None


def _supported(point):
    return (point.get("prediction_status") == "research_prediction"
            and all(_finite(point.get(field)) for field in POWER_FIELDS.values()))


def _anchor(point):
    return {
        "source": "prediction_nc_grid",
        "grid_row_index": point["_bridge_grid_row_index"],
        "line": int(point["line"]),
        "execution_instance": int(point["execution_instance"]),
        "sequence_id": _sequence(point["sequence_id"]),
        "segment_mm": float(point["segment_mm"]),
        "nc_path_mm": float(point["path_mm"]) if _finite(point.get("path_mm")) else None,
        "prediction_status": point["prediction_status"],
        "powers_w": {field: float(point[grid_field]) for field, grid_field in POWER_FIELDS.items()},
    }


def bridge_short_prediction_gaps(mapped_rows, grid):
    """Return copied source rows and an audit summary; preserve valid rows.

    ``grid`` may be records or a DataFrame from the original prediction.csv.
    Both anchors must be strictly on opposite sides of the missing position.
    Every grid point between them must belong to the same NC sequence. Contact
    state may change, but prediction values, model weights and labels do not.
    The exclusive threshold uses the decimal representation of NC geometry to
    reject a nominal 0.05 mm span despite binary subtraction roundoff.
    """
    records = grid.to_dict("records") if hasattr(grid, "to_dict") else list(grid)
    groups = {}
    for index, record in enumerate(records):
        key = (_identity(record.get("line")), _identity(record.get("execution_instance")))
        if None in key or not _finite(record.get("segment_mm")):
            continue
        point = dict(record, _bridge_grid_row_index=index)
        groups.setdefault(key, []).append(point)
    for group in groups.values():
        group.sort(key=lambda point: point["segment_mm"])

    rows = deepcopy(mapped_rows)
    bridged = []
    threshold = Decimal(str(MAX_ANCHOR_SPAN_MM))
    for row_index, row in enumerate(rows):
        # The original mapping owns all successful results, even if their
        # contact state or numerical value differs from this fallback.
        if row.get("prediction_status") != "unavailable_input":
            continue
        position = row.get("segment_mm")
        if not _finite(position):
            continue
        group = groups.get((_identity(row.get("line")), _identity(row.get("execution_instance"))), [])
        left = [point for point in group if point["segment_mm"] < position and _supported(point)]
        right = [point for point in group if point["segment_mm"] > position and _supported(point)]
        if not left or not right:
            continue
        lower, upper = left[-1], right[0]
        start, end = float(lower["segment_mm"]), float(upper["segment_mm"])
        span_decimal = Decimal(str(end)) - Decimal(str(start))
        if not Decimal(0) < span_decimal < threshold:
            continue
        sequence = _sequence(lower.get("sequence_id"))
        if sequence is None or _sequence(upper.get("sequence_id")) != sequence:
            continue
        source_sequence = row.get("sequence_id")
        if source_sequence is not None and _sequence(source_sequence) != sequence:
            continue
        between = [point for point in group if start <= point["segment_mm"] <= end]
        if any(_sequence(point.get("sequence_id")) != sequence for point in between):
            continue
        # Duplicate coordinates with different predictions are ambiguous.
        if any(_supported(point) and any(point[field] != anchor[field] for field in POWER_FIELDS.values())
               for anchor in (lower, upper) for point in between
               if point["segment_mm"] == anchor["segment_mm"]):
            continue
        fraction = (float(position) - start) / (end - start)
        values = {field: float(lower[grid_field]) + fraction * (float(upper[grid_field]) - float(lower[grid_field]))
                  for field, grid_field in POWER_FIELDS.items()}
        if not all(math.isfinite(value) for value in values.values()):
            continue
        row.update(values)
        row.update(prediction_status="research_prediction", status="research_prediction",
                   mapping_method=MAPPING_METHOD, prediction_segment_mm=float(position))
        row["short_gap_bridge"] = {
            "rule_version": RULE_VERSION,
            "anchor_span_mm": float(span_decimal),
            "max_anchor_span_mm_exclusive": MAX_ANCHOR_SPAN_MM,
            "interpolation_fraction": fraction,
            "geometry": "segment_mm_from_nc_supported_geometry",
            "left_anchor": _anchor(lower),
            "right_anchor": _anchor(upper),
        }
        bridged.append(row.get("source_index", row_index))

    metadata = {
        "enabled": True,
        "rule_version": RULE_VERSION,
        "max_anchor_span_mm_exclusive": MAX_ANCHOR_SPAN_MM,
        "geometry": "segment_mm_from_nc_supported_geometry",
        "same_nc_line_execution_sequence_required": True,
        "contact_transition_allowed": True,
        "extrapolation_allowed": False,
        "applies_to": "prediction_only_not_measured_labels",
        "bridged_row_count": len(bridged),
        "bridged_source_indices": bridged,
    }
    return rows, metadata
