"""NC-only features, measured labels and optional program-holdout diagnostics.

``program_id`` is a caller-reviewed program-family identity: repeated machining
and F/S variants of one program must keep the same identity. Round numbers are
local to that program; different parts collected in one experiment round are
not claimed to be independent round holdouts by this module.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .common import checkpoint, version_of
from .feed import parse_nc
from .measurement import detect_encoding, resolve_position_policy, select_position_channels
from .sample_rules import SAMPLE_RULES_VERSION, contact_mask, continuous_ids, motion_mask, quality_pass
from afc_common.process_path import PROCESS_MAPPING_RULES_VERSION


FEED_SAMPLE_SCHEMA = "afc.feed_samples.2"
SPATIAL_ALIGNMENT_VERSION = "afc.spatial_alignment.2"
FEED_FEATURE_COLUMNS = (
    "motion_g0", "motion_g1", "motion_g2", "motion_g3",
    "x_mm", "y_mm", "z_mm", "end_x_mm", "end_y_mm", "end_z_mm",
    "arc_i_mm", "arc_j_mm", "arc_k_mm", "arc_radius_mm", "arc_sweep_rad",
    "feed_program_mm_min", "feed_override", "block_length_mm", "segment_fraction",
    "direction_x", "direction_y", "direction_z", "entry_angle_rad", "exit_angle_rad",
    "curvature_per_mm", "has_previous_block", "has_next_block",
    "previous_block_length_mm", "next_block_length_mm",
    "previous_feed_program_mm_min", "next_feed_program_mm_min",
    "previous_curvature_per_mm", "next_curvature_per_mm",
    "previous_direction_x", "previous_direction_y", "previous_direction_z",
    "next_direction_x", "next_direction_y", "next_direction_z",
)


@dataclass
class FeedSample:
    frame: pd.DataFrame
    metadata: dict


def _identity(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}必须是明确的非空标识")
    return value.strip()


def _tangent(segment, u):
    u = np.asarray(u, dtype=float)
    if segment.center is None:
        delta = np.broadcast_to(segment.end - segment.start, (*u.shape, 3)).copy()
    else:
        radius = np.linalg.norm(segment.start[:2] - segment.center[:2])
        angle = np.arctan2(segment.start[1] - segment.center[1], segment.start[0] - segment.center[0]) + segment.sweep * u
        delta = np.stack((-radius * np.sin(angle) * segment.sweep,
                          radius * np.cos(angle) * segment.sweep,
                          np.full(u.shape, segment.end[2] - segment.start[2])), axis=-1)
    return delta / np.linalg.norm(delta, axis=-1, keepdims=True)


def _curvature(segment):
    if segment.center is None:
        return 0.
    radius = float(np.linalg.norm(segment.start[:2] - segment.center[:2]))
    pitch = (segment.end[2] - segment.start[2]) / segment.sweep
    return float(radius / (radius * radius + pitch * pitch))


def _angle(a, b):
    return float(np.arccos(np.clip(np.dot(a, b), -1., 1.)))


def _reject_unparsed_full_circles(path):
    """The shared parser only emits motion when an XYZ endpoint is present."""
    mode = 0
    text = Path(path).read_text(encoding=detect_encoding(path), errors="strict")
    for number, raw in enumerate(text.splitlines(), 1):
        line = re.sub(r"\([^)]*\)", "", raw).split(";", 1)[0].upper()
        for motion in re.findall(r"G\s*0?([0123])(?=[^\d.]|$)", line):
            mode = int(motion)
        if mode in (2, 3) and re.search(r"[IJR]\s*[-+\d.#]", line) and not re.search(r"[XYZ]\s*[-+\d.#]", line):
            raise ValueError(f"行{number}的无XYZ终点整圆尚未被几何解析器支持")
        if re.search(r"M\s*30(?=\D|$)", line):
            break


def prepare_feed_features(nc_path, machine, *, program_id, feed_override,
                          machining_mode, machine_settings_version, spacing_mm=.5,
                          max_points=1_000_000, cancel=None, coordinate_mode="machine") -> FeedSample:
    """Sample exact supported NC geometry without a motion plan or time axis.

    The compatibility column ``feed_cmd_mm_min`` is deliberately NaN: NC-only
    inputs do not contain the controller's actual pre-override feed command.
    It is not a model feature. Known programmed F and override stay separate.
    """
    program_id = _identity(program_id, "program_id（完整程序族，包含其F/S变体）")
    machining_mode = _identity(machining_mode, "machining_mode")
    settings = _identity(machine_settings_version, "machine_settings_version")
    if isinstance(feed_override, bool) or not np.isfinite(feed_override) or feed_override <= 0:
        raise ValueError("feed_override必须为加工前已知的正倍率")
    if not np.isfinite(spacing_mm) or spacing_mm <= 0:
        raise ValueError("spacing_mm必须为正数")
    if isinstance(max_points, bool) or int(max_points) != max_points or max_points < 2:
        raise ValueError("max_points必须为不小于2的整数")
    initial = np.asarray(machine.initial_position_mm, dtype=float)
    if coordinate_mode == "machine" and (initial.shape != (3,) or not np.isfinite(initial).all()):
        raise ValueError("生成实际进给特征前必须明确初始XYZ位置")
    _reject_unparsed_full_circles(nc_path)
    parsed = parse_nc(nc_path, machine, coordinate_mode=coordinate_mode)
    if parsed["issues"]:
        detail = "; ".join(f"行{x['line']}：{x['message']}" for x in parsed["issues"])
        raise ValueError(f"NC几何/控制器语义未确认：{detail}")
    segments = parsed["segments"]
    if not segments:
        raise ValueError("NC程序没有可用的G0/G1/G2/G3空间运动")
    counts = [max(2, int(np.ceil(segment.length / spacing_mm)) + 1) for segment in segments]
    if sum(counts) > max_points:
        raise ValueError("NC特征点数超过max_points；请增大空间采样间距")
    frames, visits, distance = [], defaultdict(int), 0.
    for index, (segment, count) in enumerate(zip(segments, counts)):
        checkpoint(cancel)
        u = np.linspace(0., 1., count)
        xyz = segment.points(u)
        tangent = _tangent(segment, u)
        previous = segments[index - 1] if index and segments[index - 1].group == segment.group and (segments[index - 1].motion == 0) == (segment.motion == 0) else None
        following = segments[index + 1] if index + 1 < len(segments) and segments[index + 1].group == segment.group and (segments[index + 1].motion == 0) == (segment.motion == 0) else None
        before = _tangent(previous, 1.) if previous else np.zeros(3)
        after = _tangent(following, 0.) if following else np.zeros(3)
        arc = segment.center - segment.start if segment.center is not None else np.zeros(3)
        radius = float(np.linalg.norm(arc[:2]))
        frame = pd.DataFrame(dict(line=segment.line, execution_instance=visits[segment.line],
                                  sequence_id=segment.group, path_mm=distance + u * segment.length,
                                  segment_mm=u * segment.length, segment_fraction=u,
                                  block_length_mm=segment.length, motion=segment.motion,
                                  feed_cmd_mm_min=np.nan, feed_program_mm_min=segment.feed_mm_min,
                                  feed_override=float(feed_override), spindle_cmd_rpm=segment.spindle_rpm,
                                  curvature_per_mm=_curvature(segment), arc_radius_mm=radius,
                                  arc_sweep_rad=segment.sweep, entry_angle_rad=_angle(before, tangent[0]) if previous else 0.,
                                  exit_angle_rad=_angle(tangent[-1], after) if following else 0.,
                                  has_previous_block=float(previous is not None), has_next_block=float(following is not None)))
        for axis_index, axis in enumerate("xyz"):
            frame[f"{axis}_mm"] = xyz[:, axis_index]
            frame[f"start_{axis}_mm"] = segment.start[axis_index]
            frame[f"end_{axis}_mm"] = segment.end[axis_index]
            frame[f"direction_{axis}"] = tangent[:, axis_index]
            frame[f"previous_direction_{axis}"] = before[axis_index]
            frame[f"next_direction_{axis}"] = after[axis_index]
        for axis_index, axis in enumerate("ijk"):
            frame[f"arc_{axis}_mm"] = arc[axis_index]
        for motion in range(4):
            frame[f"motion_g{motion}"] = float(segment.motion == motion)
        for name, neighbor in (("previous", previous), ("next", following)):
            frame[f"{name}_block_length_mm"] = neighbor.length if neighbor else 0.
            frame[f"{name}_feed_program_mm_min"] = neighbor.feed_mm_min if neighbor else 0.
            frame[f"{name}_curvature_per_mm"] = _curvature(neighbor) if neighbor else 0.
        frames.append(frame)
        visits[segment.line] += 1
        distance += segment.length
    frame = pd.concat(frames, ignore_index=True)
    frame["sequence_id"] = continuous_ids(frame, known_only=True)
    if not np.isfinite(frame[list(FEED_FEATURE_COLUMNS)].to_numpy()).all():
        raise ValueError("NC几何特征包含无效数值")
    return FeedSample(frame, dict(schema=FEED_SAMPLE_SCHEMA, feature_columns=list(FEED_FEATURE_COLUMNS),
                       program_id=program_id, program_version=parsed["program_version"],
                       program_id_scope="complete_program_family_including_parameter_variants",
                       round_scope="experiment_round", sample_rules_version=SAMPLE_RULES_VERSION,
                       process_mapping_rules_version=PROCESS_MAPPING_RULES_VERSION,
                       machine_version=version_of(asdict(machine)),
                       machine_settings_version=settings, machining_mode=machining_mode,
                       coordinate_mode=coordinate_mode, controller_settings=parsed["controller_settings"],
                       rapid_override="unobserved_excluded" if coordinate_mode == "workpiece_cutting" else "not_separately_recorded",
                       feed_override=float(feed_override), spacing_mm=float(spacing_mm),
                       dataset_role="development", source="nc_geometry_and_known_override",
                       command_feed_source="unavailable_without_controller_command",
                       nc_path=str(Path(nc_path).resolve()), non_motion_events=parsed["events"]))


def _measurement_arrays(measurement):
    time = np.asarray(measurement.time_s, dtype=float)
    valid = np.asarray(measurement.valid, dtype=bool)
    if time.ndim != 1 or valid.shape != time.shape:
        raise ValueError("实测时间与有效性数组长度不一致")
    return time, valid & np.isfinite(time)


def _channel(measurement, name, count):
    value = np.asarray(measurement.values[name], dtype=float)
    if value.shape != (count,):
        raise ValueError(f"实测通道{name}与时间长度不一致")
    return value


def _period(time, metadata):
    value = metadata.get("sample_period_s")
    if value is not None and np.isfinite(value) and value > 0:
        return float(value)
    delta = np.diff(time)
    good = delta[np.isfinite(delta) & (delta > 0)]
    return float(np.median(good)) if len(good) else None


def _blocks(time, valid, period):
    if not len(time):
        return
    delta = np.diff(time)
    breaks = ~np.isfinite(delta) | (delta <= 0)
    if period is not None:
        breaks |= delta > 1.5 * period
    boundaries = np.flatnonzero(np.r_[True, ~valid[1:] | ~valid[:-1] | breaks, True])
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if valid[start:end].all() and end > start:
            yield int(start), int(end)


def extract_measured_feed(measurement, *, allow_position_derivative=False):
    """Return actual mm/min only; an explicit derivative flag asserts verification.

    Missing sources produce NaN/False/``missing`` so callers can still display
    other channels. Invalid high-priority channels are never silently replaced.
    """
    time, base_valid = _measurement_arrays(measurement)
    count = len(time)
    result = np.full(count, np.nan)
    if "actual_feed_mm_min" in measurement.values:
        result = _channel(measurement, "actual_feed_mm_min", count).copy()
        source = "actual_feed_mm_min"
    elif all(f"actual_v{axis}_mm_min" in measurement.values for axis in "xyz"):
        axes = np.column_stack([_channel(measurement, f"actual_v{axis}_mm_min", count) for axis in "xyz"])
        finite = np.isfinite(axes).all(axis=1)
        result[finite] = np.linalg.norm(axes[finite], axis=1)
        source = "actual_axis_velocity_norm"
    elif allow_position_derivative is True and all(f"actual_{axis}_mm" in measurement.values for axis in "xyz"):
        xyz = np.column_stack([_channel(measurement, f"actual_{axis}_mm", count) for axis in "xyz"])
        valid = base_valid & np.isfinite(xyz).all(axis=1)
        for start, end in _blocks(time, valid, _period(time, measurement.metadata)):
            if end - start >= 2:
                derivative = np.gradient(xyz[start:end], time[start:end], axis=0, edge_order=1)
                result[start:end] = np.linalg.norm(derivative, axis=1) * 60.
        source = "verified_actual_position_derivative"
    else:
        return result, np.zeros(count, dtype=bool), "missing"
    valid = base_valid & np.isfinite(result) & (result >= 0)
    result[~valid] = np.nan
    return result, valid, source


def _project(rows, xyz):
    first = rows.iloc[0]
    start = first[[f"start_{axis}_mm" for axis in "xyz"]].to_numpy(float)
    end = first[[f"end_{axis}_mm" for axis in "xyz"]].to_numpy(float)
    if first.arc_radius_mm == 0:
        delta = end - start
        fraction = (xyz - start) @ delta / np.dot(delta, delta)
        expected = start + fraction[:, None] * delta
    else:
        center = start + first[[f"arc_{axis}_mm" for axis in "ijk"]].to_numpy(float)
        angle0 = np.arctan2(start[1] - center[1], start[0] - center[0])
        signed = np.unwrap(np.arctan2(xyz[:, 1] - center[1], xyz[:, 0] - center[0]) - angle0) * np.sign(first.arc_sweep_rad)
        if len(signed) and signed[0] < -.01:
            signed += 2 * np.pi * np.ceil(-signed[0] / (2 * np.pi))
        fraction = signed / abs(first.arc_sweep_rad)
        angles = angle0 + fraction * first.arc_sweep_rad
        expected = np.column_stack((center[0] + first.arc_radius_mm * np.cos(angles),
                                    center[1] + first.arc_radius_mm * np.sin(angles),
                                    start[2] + fraction * (end[2] - start[2])))
    return fraction * first.block_length_mm, np.linalg.norm(xyz - expected, axis=1)


def _row_tail(rows, xyz, time, valid, period, tolerance):
    """Return the contiguous forward projection anchored at the last sample of a visit."""
    length = float(rows.block_length_mm.iloc[0])
    distance, error = _project(rows, xyz)
    accepted = valid & np.isfinite(distance) & np.isfinite(error) & (error <= tolerance)
    if not accepted[0] or not 0 <= distance[0] < length:
        return distance[:0], error[:0]
    end = next(_blocks(time, accepted, period))[1]
    advances = np.flatnonzero(np.diff(distance[:end]) <= 1e-8)
    if len(advances):
        end = advances[0] + 1
    reaches_end = np.flatnonzero(distance[:end] >= length)
    if len(reaches_end):
        end = reaches_end[0] + 1
    return distance[:end], error[:end]


def align_spatial_channels(frame, measurement, channels, *, valid=None, line_map=None,
                           position_offset_mm=None, position_tolerance_mm=.1,
                           length_tolerance_fraction=.05, support_provenance=False, position_policy=None,
                           recover_row_tails=False):
    """Project channels onto one NC grid without redefining visits at gaps.

    Channels may include NaN (e.g. unavailable actual feed); interpolation then
    preserves that channel's missing support. ``valid`` additionally breaks all
    interpolation, such as when a measured-speed base prediction is invalid.
    ``recover_row_tails`` pairs uncovered points using adjacent forward samples
    when the controller reports the next line before reaching the NC endpoint.
    """
    if not np.isfinite(position_tolerance_mm) or position_tolerance_mm <= 0 or not 0 <= length_tolerance_fraction < 1:
        raise ValueError("位置及长度容差无效")
    frame = frame.reset_index(drop=True)
    count = len(measurement.time_s)
    arrays = {name: np.asarray(values, float) for name, values in channels.items()}
    if any(values.shape != (count,) for values in arrays.values()):
        raise ValueError("待对齐通道与时间长度不一致")
    result = pd.DataFrame({name: np.full(len(frame), np.nan) for name in arrays})
    result["alignment_error_mm"] = np.nan
    result["aligned"] = False
    position_policy = resolve_position_policy(position_policy=position_policy)
    position_source, position_columns = select_position_channels(measurement.values, position_policy=position_policy)
    info = dict(position_policy=position_policy, position_source=position_source, position_columns=list(position_columns),
                spatial_alignment_version=SPATIAL_ALIGNMENT_VERSION, issues=[], aligned_points=0)
    if support_provenance:
        source_indices = np.asarray(measurement.source_indices)
        if (source_indices.shape != (count,) or source_indices.dtype.kind not in "iu"
                or np.any(source_indices < 0) or np.any(np.diff(source_indices) != 1)):
            raise ValueError("参考附件要求完整连续原始索引；不能用分段输出掩盖已跨原始缺口的插值")
        result["support_segment_id"] = -1
        info["support_segments"] = []
    if not count or "program_line_raw" not in measurement.values or not position_columns:
        info["issues"].append("空间对齐缺少有效采样、程序行号或完整实际/指令XYZ位置")
        return result, info
    time, record_valid = _measurement_arrays(measurement)
    raw = _channel(measurement, "program_line_raw", count)
    xyz = np.column_stack([_channel(measurement, name, count) for name in position_columns])
    period = _period(time, measurement.metadata)
    good = record_valid & np.isfinite(xyz).all(axis=1) & np.isfinite(raw) & (raw == np.round(raw))
    if valid is not None:
        extra = np.asarray(valid, bool)
        if extra.shape != (count,):
            raise ValueError("待对齐有效性遮罩与时间长度不一致")
        good &= extra
    # Visits are controller line events, independent of channel/base validity.
    visits = defaultdict(list)
    boundaries = np.flatnonzero(np.r_[True, raw[1:] != raw[:-1], True])
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start or not np.isfinite(raw[start]) or raw[start] != round(raw[start]):
            continue
        mapped = (line_map or {}).get(int(raw[start]), int(raw[start]))
        if isinstance(mapped, bool) or int(mapped) != mapped:
            raise ValueError("line_map必须显式映射到整数物理行号")
        visits[int(mapped)].append((int(start), int(end)))
    line_instances = {line: list(frame.loc[frame.line == line, "execution_instance"].drop_duplicates()) for line in frame.line.unique()}
    matched = []
    for (line, instance), rows in frame.groupby(["line", "execution_instance"], sort=False):
        candidates = visits.get(int(line), [])
        if len(candidates) != len(line_instances[line]):
            info["issues"].append(f"行{line}的实测访问次数不匹配；不跨访问拼接")
            continue
        required_geometry = {f"{end}_{axis}_mm" for end in ("start", "end") for axis in "xyz"} | {
            "arc_radius_mm", "arc_sweep_rad", "arc_i_mm", "arc_j_mm", "arc_k_mm"}
        if required_geometry - set(rows):
            # Legacy NC grids can establish a straight segment only when they
            # contain both exact endpoints and the whole grid is collinear.
            positions = rows.segment_mm.to_numpy(float)
            points = rows[["x_mm", "y_mm", "z_mm"]].to_numpy(float)
            length = float(rows.block_length_mm.iloc[0])
            if (len(rows) < 2 or length <= 0 or not np.isclose(positions[0], 0., atol=1e-8)
                    or not np.isclose(positions[-1], length, atol=1e-8)
                    or not np.isclose(np.linalg.norm(points[-1] - points[0]), length, atol=1e-8)
                    or not np.allclose(points, points[0] + positions[:, None] / length * (points[-1] - points[0]), atol=1e-8)):
                info["issues"].append(f"行{line}缺少可核对的NC投影几何")
                continue
            rows = rows.copy()
            for axis, index in zip("xyz", range(3)):
                rows[f"start_{axis}_mm"], rows[f"end_{axis}_mm"] = points[0, index], points[-1, index]
            for name in ("arc_radius_mm", "arc_sweep_rad", "arc_i_mm", "arc_j_mm", "arc_k_mm"):
                rows[name] = 0.
        start, end = candidates[line_instances[line].index(instance)]
        matched.append((rows, start, end))
    offset = position_offset_mm
    if offset is not None:
        offset = np.asarray(offset, dtype=float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("position_offset_mm必须为三个有限数值")
    else:
        for rows, start, end in matched:
            if end - start < 2 or not good[start:end].all() or len(list(_blocks(time[start:end], good[start:end], period))) != 1:
                continue
            steps = np.linalg.norm(np.diff(xyz[start:end], axis=0), axis=1)
            length = float(rows.block_length_mm.iloc[0])
            if np.any(steps <= 1e-8) or abs(steps.sum() - length) > max(position_tolerance_mm, length * length_tolerance_fraction):
                continue
            candidate = xyz[start] - rows.iloc[0][[f"start_{axis}_mm" for axis in "xyz"]].to_numpy(float)
            position, error = _project(rows, xyz[start:end] - candidate)
            if np.all(error <= position_tolerance_mm) and np.all(np.diff(position) > 1e-8):
                offset = candidate
                break
    if offset is None:
        info["issues"].append("没有完整连续访问可确定坐标平移；请提供经过核对的position_offset_mm")
        return result, info
    info["position_offset_mm"] = offset.tolist()
    info["observed_visit_count"] = sum(len(values) for values in visits.values())
    info["matched_visit_count"] = len(matched)
    assignment_counts = np.zeros(len(frame), dtype=int)
    if recover_row_tails:
        info["row_tail_recovered_points"] = 0
        next_visits = {start: (line, stop) for line, spans in visits.items() for start, stop in spans}

    def assign(rows, indices, targets, distance, error, sample_start, *, recovered=False):
        sample_end = sample_start + len(distance)
        assignment_counts[indices] += 1
        if support_provenance:
            segment_id = len(info["support_segments"])
            result.loc[indices, "support_segment_id"] = segment_id
            entry = dict(support_segment_id=segment_id, line=int(rows.line.iloc[0]),
                         execution_instance=int(rows.execution_instance.iloc[0]),
                         source_start_index=int(source_indices[sample_start]),
                         source_end_index_exclusive=int(source_indices[sample_end - 1]) + 1,
                         local_start_mm=float(distance[0]), local_end_mm=float(distance[-1]))
            if recovered:
                entry["row_tail_recovery"] = True
            info["support_segments"].append(entry)
        for name, values in arrays.items():
            result.loc[indices, name] = np.interp(targets, distance, values[sample_start:sample_end])
        result.loc[indices, "alignment_error_mm"] = np.interp(targets, distance, error)

    for rows, start, end in matched:
        targets = rows.segment_mm.to_numpy(float)
        row_indices = rows.index.to_numpy()
        for local_start, local_end in _blocks(time[start:end], good[start:end], period):
            a, b = start + local_start, start + local_end
            if b - a < 2:
                continue
            position, error = _project(rows, xyz[a:b] - offset)
            accepted = np.isfinite(position) & np.isfinite(error) & (error <= position_tolerance_mm)
            accepted &= (position >= -1e-8) & (position <= float(rows.block_length_mm.iloc[0]) + 1e-8)
            nonmoving = np.flatnonzero(np.diff(position) <= 1e-8)
            accepted[nonmoving] = False
            accepted[nonmoving + 1] = False
            for left, right in _blocks(time[a:b], accepted, period):
                if right - left < 2:
                    continue
                local_s = position[left:right]
                inside = (targets >= local_s[0]) & (targets <= local_s[-1])
                indices = row_indices[inside]
                if not len(indices):
                    continue
                assign(rows, indices, targets[inside], local_s, error[left:right], a + left)
        if not recover_row_tails or end not in next_visits:
            continue
        next_line, next_end = next_visits[end]
        if next_line <= rows.line.iloc[0] or not np.any(assignment_counts[row_indices] == 0):
            continue
        first = end - 1
        pos, err = _row_tail(rows, xyz[first:next_end] - offset, time[first:next_end],
                             good[first:next_end], period, position_tolerance_mm)
        if len(pos) < 2:
            continue
        missing = (assignment_counts[row_indices] == 0) & (targets >= pos[0]) & (targets <= pos[-1])
        if missing.any():
            assign(rows, row_indices[missing], targets[missing], pos, err, first, recovered=True)
            info["row_tail_recovered_points"] += int(missing.sum())
    unique = assignment_counts == 1
    result.loc[~unique, list(arrays)] = np.nan
    result["aligned"] = unique
    if support_provenance:
        result.loc[~unique, "support_segment_id"] = -1
    info["aligned_points"] = int(unique.sum())
    if np.any(assignment_counts > 1):
        info["issues"].append("同一NC位置被多段实测重复覆盖，标签已置无效")
    return result, info


def prepare_feed_samples(features, measurement, metadata, *, line_map=None,
                         position_tolerance_mm=.1, length_tolerance_fraction=.05,
                         allow_position_derivative=False, position_policy=None) -> FeedSample:
    """Attach actual-feed labels using NC visits and recorded spatial position.

    Invalid gaps and stationary positions split interpolation support. The NC
    context/sequence stays unchanged; only label masks are affected. A reviewed
    ``position_offset_mm`` may specify machine minus NC coordinates. Otherwise
    one complete, continuous matched visit must establish the translation.
    ``command_only_v1`` locates labels on the command path; the legacy default
    prefers actual XYZ. Recorded positions do not enter prospective NC features
    and never supply an actual-feed label by command differentiation.
    """
    if features.metadata.get("schema") != FEED_SAMPLE_SCHEMA:
        raise ValueError("实际进给特征schema不匹配")
    if not np.isfinite(position_tolerance_mm) or position_tolerance_mm <= 0 or not 0 <= length_tolerance_fraction < 1:
        raise ValueError("位置及长度容差无效")
    identity_keys = ("program_id", "program_version", "machine_version", "machine_settings_version", "machining_mode", "feed_override", "coordinate_mode")
    for key in identity_keys:
        if key in metadata and metadata[key] != features.metadata[key]:
            raise ValueError(f"实测与NC特征的{key}不一致")
    info = {**features.metadata, **metadata}
    position_policy = resolve_position_policy(metadata, position_policy)
    position_source, position_columns = select_position_channels(measurement.values, position_policy=position_policy)
    info.update(position_policy=position_policy, schema=FEED_SAMPLE_SCHEMA, feature_columns=list(FEED_FEATURE_COLUMNS),
                measurement_source=measurement.source, position_source=position_source,
                position_columns=list(position_columns), issues=[])
    frame = features.frame.copy().reset_index(drop=True)
    frame["feed_actual_mm_min"] = np.nan
    frame["valid_label"] = False
    frame["alignment_error_mm"] = np.nan
    measured_feed, feed_valid, source = extract_measured_feed(measurement, allow_position_derivative=allow_position_derivative)
    info["measured_feed_source"] = source
    if source == "verified_actual_position_derivative":
        info["position_derivative_verified"] = True
    quality = quality_pass(info)
    info["quality_pass"] = quality
    if not len(measurement.time_s):
        info["issues"].append("实测记录为空，未生成标签")
        return FeedSample(frame, info)
    if source == "missing":
        info["issues"].append("缺少实际进给通道；未使用指令位置或指令速度替代")
        return FeedSample(frame, info)
    if "program_line_raw" not in measurement.values or not position_columns:
        info["issues"].append("标签空间对齐缺少程序行号或完整实际/指令XYZ位置；候选保留，标签无效")
        return FeedSample(frame, info)
    aligned, alignment_info = align_spatial_channels(frame, measurement,
        {"feed_actual_mm_min": measured_feed}, valid=feed_valid, line_map=line_map,
        position_offset_mm=info.get("position_offset_mm"), position_tolerance_mm=position_tolerance_mm,
        length_tolerance_fraction=length_tolerance_fraction, position_policy=position_policy)
    info.update(alignment_info)
    frame["feed_actual_mm_min"] = aligned.feed_actual_mm_min
    frame["alignment_error_mm"] = aligned.alignment_error_mm
    unique = aligned.aligned.to_numpy(bool)
    frame["valid_label"] = (unique & quality & np.isfinite(frame.feed_actual_mm_min)
                            & motion_mask(frame) & contact_mask(frame, info))
    frame["training_sequence_id"] = continuous_ids(frame, frame.valid_label.to_numpy(bool), known_only=True)
    if not quality:
        info["issues"].append("complete、alignment_reliable、no_alarm、tool_healthy 均需明确为True")
    if not contact_mask(frame, info).any():
        info["issues"].append("切削接触尚未确认，仅保留候选运动，不把全部非G0视为切削")
    info["aligned_points"] = int(unique.sum())
    info["valid_label_points"] = int(frame.valid_label.sum())
    return FeedSample(frame, info)


def program_split(samples, assignment):
    """Split complete program families; identical source hashes cannot cross.

    A global experiment round can contain several different part programs.
    Their records follow program-family assignment, not the numeric round ID.
    Repeated rounds of one family and all of its F/S variants remain together.
    """
    groups = {name: [] for name in ("train", "validation", "test")}
    versions, sources = {}, {}
    for sample in samples:
        program_id = _identity(sample.metadata.get("program_id"), "program_id")
        version = _identity(sample.metadata.get("program_version"), "program_version")
        if program_id not in assignment or assignment[program_id] not in groups:
            raise ValueError(f"程序族{program_id}需要明确的train/validation/test划分")
        split = assignment[program_id]
        if version in versions and versions[version] != split:
            raise ValueError("相同program_version不能以不同程序族标识跨集合")
        versions[version] = split
        source = sample.metadata.get("measurement_source")
        if source:
            source = str(Path(source).resolve()).casefold()
            if source in sources and sources[source] != split:
                raise ValueError("同一实测来源不能跨完整程序集合")
            sources[source] = split
        groups[split].append(sample)
    return groups
