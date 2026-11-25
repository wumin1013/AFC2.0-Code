"""Opt-in NC coordinates supported by the development process exports.

This module does not change the historical reader or rebuild a training set.
The rounded source distance validates the declared reconstruction; it is not
used as an exact global coordinate. No measured response enters this rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
import math

import numpy as np


PROCESS_COORDINATE_RULES_VERSION = "afc.data_supported_process_coordinates.2"
SHORT_ARC_ENDPOINT_POLICY = "last_source_cell_to_exact_nc_endpoint_v1"


@dataclass(frozen=True)
class ProcessCoordinates:
    rule: str
    positions_mm: tuple[float, ...]
    cell_starts_mm: tuple[float, ...]
    cumulative_source_distance_mm: tuple[float, ...]
    unsupported_tail_mm: float
    terminal_support_end_mm: float | None = None

    @property
    def cell_ends_mm(self):
        terminal = self.positions_mm[-1] if self.terminal_support_end_mm is None else self.terminal_support_end_mm
        return (*self.positions_mm[:-1], terminal)

    def __post_init__(self):
        positions, starts, cumulative = (np.asarray(values, dtype=float) for values in
            (self.positions_mm, self.cell_starts_mm, self.cumulative_source_distance_mm))
        if (positions.ndim != 1 or not len(positions) or starts.shape != positions.shape
                or cumulative.shape != positions.shape
                or not all(np.isfinite(values).all() for values in (positions, starts, cumulative))
                or positions[0] <= 0 or cumulative[0] <= 0 or starts[0] != 0
                or not np.array_equal(starts[1:], positions[:-1])
                or np.any(np.diff(positions) <= 0) or np.any(np.diff(cumulative) <= 0)
                or not math.isfinite(self.unsupported_tail_mm) or self.unsupported_tail_mm < 0):
            raise ValueError("重建坐标、来源累计距离与支持区间必须有限、有序、等长且一致")
        terminal = self.terminal_support_end_mm
        if terminal is not None and (not math.isfinite(terminal) or terminal < positions[-1]
                or self.rule != "small_arc_interior_n_plus_one" or len(positions) != 9
                or not 0 < terminal < .5 or self.unsupported_tail_mm != 0.
                or not math.isclose(positions[-1], terminal * 9 / 10, rel_tol=0., abs_tol=1e-12)):
            raise ValueError("末端闭合仅适用于已核验短弧的本段精确NC终点")


def reconstruct_process_coordinates(*, length_mm, point_count, motion,
                                    radius_mm=0., sweep_rad=0., z_change_mm=0.):
    """Reconstruct one supported geometry family, rejecting unstudied families.

    A point owns the preceding cell (start, end]; exact zero belongs to the
    first cell. This is an explicit input interpolation convention. For the
    verified short arcs, the last source cell ends at exact NC length. Source
    knots remain unchanged; closing the cell does not invent a source sample.
    """
    length, radius, sweep, dz = map(float, (length_mm, radius_mm, sweep_rad, z_change_mm))
    terminal_support_end = None
    if (not all(math.isfinite(x) for x in (length, radius, sweep, dz))
            or length <= 0 or type(point_count) is not int or point_count < 1
            or type(motion) is not int or motion not in (1, 2, 3)):
        raise ValueError("坐标重建需要正有限段长、正整数点数及G01/G02/G03运动")
    if motion == 1:
        rule = "linear_uniform_n"
        positions = [length * k / point_count for k in range(1, point_count)] + [length]
        cumulative = positions.copy()
        tail = 0.
    else:
        # This geometric tolerance only checks the already parsed NC circle;
        # it never expands source support or validates rounded source s.
        if (radius <= 0 or not 0 < abs(sweep) <= 2 * math.pi or dz != 0.
                or not math.isclose(length, radius * abs(sweep), rel_tol=0., abs_tol=1e-10)):
            raise ValueError("当前数据位置规则仅支持已核验的平面圆弧")
        if (point_count >= 2 and point_count == math.floor(length / .5)
                and 0 < (point_count - 1) * .5 < length):
            rule = "arc_fixed_half_plus_end"
            positions = [k * .5 for k in range(1, point_count)] + [length]
            tail = 0.
        elif (motion == 3 and point_count == 9 and length < .5
              and math.isclose(radius, .2, rel_tol=0., abs_tol=1e-10)
              and math.isclose(abs(sweep), math.pi / 2, rel_tol=0., abs_tol=1e-10)):
            rule = "small_arc_interior_n_plus_one"
            positions = [length * k / (point_count + 1) for k in range(1, point_count + 1)]
            tail = 0.
            terminal_support_end = length
        else:
            raise ValueError("圆弧点数或几何不属于已核验的位置规则；不得回退为均分")
        increments = np.diff([0., *positions])
        cumulative = np.cumsum(2 * radius * np.sin(increments / (2 * radius))).tolist()
    return ProcessCoordinates(rule, tuple(positions), tuple([0., *positions[:-1]]),
                              tuple(cumulative), tail, terminal_support_end)


def source_distance_compatible(tokens, coordinates):
    """Check a common origin under the declared six-significant-digit format.

    Returns an origin interval, not an identified physical origin. Extra
    decimal places may be omitted by the writer; repeated rounded values are
    retained. Higher-precision exports require their own format definition.
    """
    if len(tokens) != len(coordinates.positions_mm) or not tokens:
        raise ValueError("原始行程token与工艺点数不一致")
    lower, upper = [], []
    with localcontext() as context:
        context.prec = 60
        for token, distance in zip(tokens, coordinates.cumulative_source_distance_mm):
            try:
                value = Decimal(str(token))
            except InvalidOperation as exc:
                raise ValueError("原始行程token不是有限十进制数") from exc
            if not value.is_finite() or value < 0 or len(value.as_tuple().digits) > 6:
                raise ValueError("原始行程不符合本版本的非负、至多六有效位格式")
            # At a positive decade (10, 100, ...), the lower neighbour has a
            # ten-times finer quantization step. A symmetric half-step would
            # incorrectly accept values that cannot round to this token.
            # Six significant digits do not round a positive real to zero.
            if value:
                step = Decimal(10) ** (value.adjusted() - 5)
                lower_half = step / (20 if value == Decimal(10) ** value.adjusted() else 2)
                upper_half = step / 2
            else:
                lower_half = upper_half = Decimal(0)
            center = value - Decimal(str(distance))
            lower.append(center - lower_half)
            upper.append(center + upper_half)
        lo, hi = max(lower), min(upper)
        return dict(compatible=lo <= hi, origin_lower_mm=str(lo), origin_upper_mm=str(hi),
                    incompatible_gap_mm=str(max(Decimal(0), lo - hi)),
                    precision="six_significant_digits", absolute_origin_identified=False)


def source_cell_indices(positions_mm, coordinates):
    """Return source indices for the declared cells; unsupported points are -1."""
    targets = np.asarray(positions_mm, dtype=float)
    if targets.ndim != 1:
        raise ValueError("查询位置必须为一维数组")
    ends = np.asarray(coordinates.cell_ends_mm)
    starts = np.asarray(coordinates.cell_starts_mm)
    indices = np.minimum(np.searchsorted(ends, targets, side="left"), len(ends) - 1)
    supported = np.isfinite(targets) & (targets > starts[indices]) & (targets <= ends[indices])
    at_zero = targets == 0.
    indices[at_zero] = 0
    supported |= at_zero
    return np.where(supported, indices, -1)


def prepare_data_supported_geometry(process, plan, metadata, *, source_distance_tokens):
    """Prepare NC-grid geometry without reading or manufacturing any labels.

    ``source_distance_tokens`` maps original file-line IDs to original s text.
    Returns the geometry sample and a source table retaining reconstructed
    knots separately from the computational source-cell ends. Steady labels
    and training sequence eligibility must be rebuilt for a new dataset.
    """
    from .samples import _prepare_process_features
    from .support_fill import fill_missing_geometry_support

    required = {"line", "execution_instance", "source_file_line", "source_point_index",
                "segment_mm", "segment_start_mm", "ap_mm", "ae_mm"}
    if (not required <= set(process) or not process.columns.is_unique
            or not process.index.is_unique or not process.source_file_line.is_unique):
        raise ValueError("位置重建需要完整且唯一的工艺来源身份")
    source = process.copy(deep=True)
    source["source_cell_end_mm"] = source.segment_mm
    source["source_position_rule"] = "unmapped"
    source["endpoint_support_policy"] = "none"
    source["process_mapping_rules_version"] = PROCESS_COORDINATE_RULES_VERSION
    closed_groups = []
    for (line, instance), grid in plan.frame.groupby(["line", "execution_instance"], sort=False):
        indices = source.loc[(source.line == line) & (source.execution_instance == instance)].sort_values(
            "source_file_line", kind="stable").index
        if not len(indices):
            continue
        shape = grid.iloc[0]
        coordinates = reconstruct_process_coordinates(length_mm=float(shape.block_length_mm),
            point_count=len(indices), motion=int(shape.motion), radius_mm=float(shape.arc_radius_mm),
            sweep_rad=float(shape.arc_sweep_rad), z_change_mm=float(shape.end_z_mm-shape.start_z_mm))
        tokens = [source_distance_tokens[int(value)] for value in source.loc[indices, "source_file_line"]]
        if not source_distance_compatible(tokens, coordinates)["compatible"]:
            raise ValueError(f"NC行{line}实例{instance}的原行程不兼容固定位置规则")
        source.loc[indices, "segment_mm"] = coordinates.positions_mm
        source.loc[indices, "segment_start_mm"] = coordinates.cell_starts_mm
        if "nc_path_mm" in source:
            source.loc[indices, "nc_path_mm"] = (float(shape.path_mm - shape.segment_mm)
                                                  + np.asarray(coordinates.positions_mm))
        if "mapping_status" in source:
            source.loc[indices, "mapping_status"] = "mapped_data_supported_nc"
        source.loc[indices, "source_cell_end_mm"] = coordinates.cell_ends_mm
        source.loc[indices, "source_position_rule"] = coordinates.rule
        if coordinates.terminal_support_end_mm is not None:
            source.loc[indices[-1], "endpoint_support_policy"] = SHORT_ARC_ENDPOINT_POLICY
            closed_groups.append(dict(line=int(line), execution_instance=int(instance),
                source_file_line=int(source.loc[indices[-1], "source_file_line"]),
                reconstructed_source_position_mm=coordinates.positions_mm[-1],
                nc_endpoint_mm=coordinates.terminal_support_end_mm,
                support_extension_mm=coordinates.terminal_support_end_mm - coordinates.positions_mm[-1]))
    sample = _prepare_process_features(source, plan, metadata)
    # Only the cell lookup receives closed endpoints; interpolation above must
    # continue using original reconstructed knots, including the last 9L/10.
    cells = source.copy(deep=True)
    cells["segment_mm"] = cells.source_cell_end_mm
    sample.frame, receipt = fill_missing_geometry_support(sample.frame, cells,
        diameter_mm=metadata["diameter_mm"])
    sample.metadata.update(process_mapping_rules_version=PROCESS_COORDINATE_RULES_VERSION,
        endpoint_support_policy=SHORT_ARC_ENDPOINT_POLICY, closed_short_arc_groups=closed_groups,
        geometry_support_receipt=receipt, steady_qualification_rebuilt=False,
        training_released=False)
    return sample, source
