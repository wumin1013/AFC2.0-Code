"""Position alignment, physical quantities and leakage-safe sample contracts."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base_power import predict_base_from_measured_speed
from .common import version_of
from .measurement import detect_encoding, resolve_position_policy, select_position_channels
from .sample_rules import SAMPLE_RULES_VERSION, continuous_ids, task_masks
from .experiment_protocol import protocol_folds, validate_round_metadata
from afc_common.process_path import uniform_path_boundaries, PROCESS_MAPPING_RULES_VERSION

PHYSICS_COLUMNS = ["ap_mm", "ae_mm", "feed_input_mm_min", "spindle_rpm", "diameter_mm", "teeth"]
PATH_COLUMNS = ["normalized_main", "ap_over_d", "ae_over_d", "feed_over_reference", "curvature_times_d", "block_over_d"]
NOMINAL_ROUTE = "nominal_programmed_residual_v1"
NOMINAL_PATH_COLUMNS = PATH_COLUMNS + [
    "entry_angle_over_pi", "exit_angle_over_pi", "direction_x", "direction_y", "direction_z",
    "segment_fraction", "previous_block_over_d", "next_block_over_d",
    "previous_feed_over_reference", "next_feed_over_reference", "has_previous_block", "has_next_block",
]
NOMINAL_GEOMETRY_COLUMNS = [
    "entry_angle_rad", "exit_angle_rad", "direction_x", "direction_y", "direction_z",
    "segment_fraction", "previous_block_length_mm", "next_block_length_mm",
    "previous_feed_program_mm_min", "next_feed_program_mm_min", "has_previous_block", "has_next_block",
]
QUALITY_FIELDS = ["tool_healthy", "no_alarm", "complete", "alignment_reliable"]
LOAD_PROCESS_RULES_VERSION = "afc.load_process.3"
PROCESS_MAPPING_VERSION = PROCESS_MAPPING_RULES_VERSION


def read_process_table(path, plan, *, line_offset=0, steady_lines=()):
    """Read canonical CSV or the current 序号 N ap ae F MRR s S G text export.

    Raw s is provenance only. Points of each continuous NC instruction group
    divide that instruction's known length equally, with positions k*L/n.
    Original point order and unmatched rows are retained, never deduplicated.
    Steady labels must be explicitly supplied from reviewed segmentation.
    """
    encoding = detect_encoding(path)
    with open(path, encoding=encoding) as stream:
        header = stream.readline().strip()
    if header.startswith("序号") and "MRR" in header and "s(mm)" in header:
        columns = ["index", "line", "ap_mm", "ae_mm", "feed", "mrr", "path_mm", "speed"]
        # G is the complete remainder of each row, with a variable number of NC
        # words. Parse exactly eight numeric fields; never infer an index from G.
        values, file_lines = [], []
        with open(path, encoding=encoding) as stream:
            next(stream)
            for file_line, text in enumerate(stream, 2):
                if not text.strip():
                    continue
                fields = text.split(maxsplit=8)
                try:
                    if len(fields) < 8:
                        raise ValueError("missing numeric columns")
                    row = [float(value) for value in fields[:8]]
                    if not np.isfinite(row).all():
                        raise ValueError("non-finite numeric value")
                except ValueError as exc:
                    raise ValueError(f"工艺表文件行{file_line}的前八列必须是有限数值") from exc
                values.append(row)
                file_lines.append(file_line)
        raw = pd.DataFrame(values, columns=columns)
        if not np.isfinite(raw.to_numpy(float)).all() or np.any(raw.line != np.round(raw.line)):
            raise ValueError("工艺表数值或NC行号无效")
        raw.rename(columns={"index": "source_point_index", "path_mm": "source_path_mm",
                            "feed": "source_feed_mm_min", "mrr": "source_mrr_mm3_min",
                            "speed": "source_spindle_rpm"}, inplace=True)
        raw["source_file_line"] = file_lines
        raw["source_nc_line"] = raw.line.astype(int)
        raw["line"] += int(line_offset)
        raw["line"] = raw.line.astype(int)
        raw["segment_mm"] = np.nan
        raw["segment_start_mm"] = np.nan
        raw["nc_path_mm"] = np.nan
        raw["point_index_in_line"] = 0
        raw["points_in_line"] = 0
        raw["execution_instance"] = -1
        raw["mapping_status"] = "unmapped_nc_group"
        raw["process_mapping_rules_version"] = PROCESS_MAPPING_RULES_VERSION
        group_ids = raw.line.ne(raw.line.shift()).cumsum()
        groups = list(raw.groupby(group_ids, sort=False))
        group_counts = raw.groupby(group_ids, sort=False).line.first().value_counts().to_dict()
        seen = {}
        for _, rows in groups:
            line = int(rows.line.iloc[0])
            count = len(rows)
            raw.loc[rows.index, "point_index_in_line"] = np.arange(1, count + 1)
            raw.loc[rows.index, "points_in_line"] = count
            choices = plan.frame.loc[plan.frame.line == line, "execution_instance"].drop_duplicates().tolist()
            occurrence = seen.get(line, 0)
            seen[line] = occurrence + 1
            if len(choices) != group_counts[line]:
                raw.loc[rows.index, "mapping_status"] = "unmapped_execution_count" if choices else "unmapped_nc_group"
                continue
            instance = choices[occurrence]
            grid = plan.frame.loc[(plan.frame.line == line) & (plan.frame.execution_instance == instance)]
            lengths = grid.block_length_mm.unique()
            if len(lengths) != 1 or not np.isfinite(lengths[0]) or lengths[0] < 0:
                raise ValueError(f"NC行{line}实例{instance}缺少唯一有限段长")
            bounds = uniform_path_boundaries(lengths[0], count)
            raw.loc[rows.index, "segment_start_mm"] = bounds[:-1]
            raw.loc[rows.index, "segment_mm"] = bounds[1:]
            raw.loc[rows.index, "execution_instance"] = instance
            raw.loc[rows.index, "mapping_status"] = "mapped_uniform_nc" if lengths[0] > 0 else "mapped_zero_length"
            if "path_mm" in grid:
                origin = grid.path_mm.to_numpy(float) - grid.segment_mm.to_numpy(float)
                if not np.allclose(origin, origin[0], rtol=0., atol=1e-8):
                    raise ValueError(f"NC行{line}实例{instance}全局与局部行程不一致")
                raw.loc[rows.index, "nc_path_mm"] = float(origin[0]) + np.asarray(bounds[1:])
        raw["steady"] = raw.line.isin(steady_lines)
        first = ["line", "execution_instance", "segment_mm", "ap_mm", "ae_mm", "steady"]
        return raw[first + [name for name in raw if name not in first]].reset_index(drop=True)
    return pd.read_csv(path, encoding=encoding, float_precision="round_trip")


def physical_quantities(physics):
    values = np.asarray(physics, float)
    if values.shape[-1] != 6 or not np.all(np.isfinite(values)):
        raise ValueError("物理输入需六列有限数值")
    ap, ae, feed, speed, diameter, teeth = np.moveaxis(values, -1, 0)
    if np.any(diameter <= 0) or np.any(teeth <= 0) or np.any(teeth != np.round(teeth)) or np.any(ap < 0) or np.any(ae < 0) or np.any(ae > diameter) or np.any(feed < 0) or np.any(speed < 0):
        raise ValueError("切削几何、齿数或进给/转速不合法")
    contact = (ap > 0) & (ae > 0) & (feed > 0) & (speed > 0)
    vc = np.pi * diameter * speed / 60000
    fz = np.divide(feed, teeth * speed, out=np.zeros_like(feed), where=speed > 0)
    phi = np.arccos(np.clip(1 - 2 * ae / diameter, -1, 1))
    return dict(mrr_mm3_s=np.where(contact, ap * ae * feed / 60, 0), vc_m_s=vc,
                fz_mm=fz, equivalent_edge_mm=np.where(contact, teeth * ap * phi / (2 * np.pi), 0), contact=contact)


def physical_power(physics, ks, ke):
    q = physical_quantities(physics)
    return np.asarray(ks) * q["mrr_mm3_s"] / 1000 + np.asarray(ke) * q["vc_m_s"] * q["equivalent_edge_mm"]


def path_feature_names(route="predicted_feed_v1"):
    if route not in ("predicted_feed_v1", NOMINAL_ROUTE):
        raise ValueError("未登记的负载路径特征路线")
    return list(NOMINAL_PATH_COLUMNS if route == NOMINAL_ROUTE else PATH_COLUMNS)


def path_features(frame, main_power, mean, scale, feed_reference, *, route="predicted_feed_v1"):
    names = path_feature_names(route)
    if not np.isfinite(scale) or scale <= 0 or feed_reference <= 0:
        raise ValueError("归一化尺度和参考进给必须为正数")
    d = frame.diameter_mm.to_numpy(float)
    if not np.isfinite(d).all() or np.any(d <= 0):
        raise ValueError("路径特征需要正有限刀具直径")
    x = np.column_stack([(np.asarray(main_power) - mean) / scale, frame.ap_mm / d, frame.ae_mm / d,
                         frame.feed_input_mm_min / feed_reference, np.abs(frame.curvature_per_mm) * d,
                         frame.block_length_mm / d])
    if route == NOMINAL_ROUTE:
        missing = set(NOMINAL_GEOMETRY_COLUMNS) - set(frame)
        if missing:
            raise ValueError(f"18维路径输入缺少加工前几何：{sorted(missing)}")
        previous = frame.has_previous_block.to_numpy(float)
        following = frame.has_next_block.to_numpy(float)
        if not np.isin(previous, (0., 1.)).all() or not np.isin(following, (0., 1.)).all():
            raise ValueError("邻段存在标记必须为0或1")
        geometry = frame[NOMINAL_GEOMETRY_COLUMNS].to_numpy(float)
        if not np.isfinite(geometry).all():
            raise ValueError("加工前路径几何存在无效值")
        if ((frame.segment_fraction < 0) | (frame.segment_fraction > 1)).any():
            raise ValueError("段内位置比例必须在[0,1]内")
        for side, exists in (("previous", previous), ("next", following)):
            for suffix in ("block_length_mm", "feed_program_mm_min"):
                value = frame[f"{side}_{suffix}"].to_numpy(float)
                if np.any(value < 0) or np.any(value[exists == 0] != 0):
                    raise ValueError("缺失邻段数值须置零，邻段长度和进给须非负")
        x = np.column_stack([x, frame.entry_angle_rad / np.pi, frame.exit_angle_rad / np.pi,
            frame.direction_x, frame.direction_y, frame.direction_z, frame.segment_fraction,
            frame.previous_block_length_mm / d, frame.next_block_length_mm / d,
            frame.previous_feed_program_mm_min / feed_reference, frame.next_feed_program_mm_min / feed_reference,
            previous, following])
    if x.shape != (len(frame), len(names)):
        raise ValueError("路径特征维度或点数不一致")
    if not np.all(np.isfinite(x)):
        raise ValueError("路径特征存在无效值")
    return x


@dataclass
class SampleSet:
    frame: pd.DataFrame
    metadata: dict


def load_input_view(sample, mode):
    """Select a physical input without changing labels or source metadata.

    ``sequence_id`` is the reviewed NC/process boundary in R6 snapshots;
    ``training_sequence_id`` and observed masks never define this view.
    """
    if mode not in ("measured", "programmed"):
        raise ValueError("负载输入视图只支持measured或programmed")
    frame = sample.frame.copy(deep=True)
    column = "feed_actual_mm_min" if mode == "measured" else "feed_program_mm_min"
    if column not in frame:
        raise ValueError(f"负载输入视图缺少{column}，不允许由其他来源替代")
    frame["feed_input_mm_min"] = frame[column].to_numpy(copy=True)
    frame["feed_source"] = "measured_actual_feed" if mode == "measured" else "nc_programmed_feed"
    if "spindle_cmd_rpm" in frame:
        frame["spindle_rpm"] = frame.spindle_cmd_rpm.to_numpy(copy=True)
    if "nc_sequence_id" in frame:
        frame["sequence_id"] = frame.nc_sequence_id.to_numpy(copy=True)
    frame["sequence_id"] = continuous_ids(frame, known_only=True)
    return SampleSet(frame, dict(sample.metadata))


def assert_feed_inputs(frame, mode):
    """Keep prospective BiLSTM inputs separate from measured diagnostics."""
    expected = {"predicted": ("bilstm_actual_feed",), "measured": ("measured_actual_feed",),
                "programmed": ("programmed_feed_diagnostic", "nc_programmed_feed")}
    if mode not in expected:
        raise ValueError("进给输入模式必须为 predicted 或 measured")
    if not {"feed_input_mm_min", "feed_source"} <= set(frame):
        raise ValueError("缺少实际进给及来源；旧规划进给不能用于负载模型")
    if frame.empty or frame.feed_source.nunique(dropna=False) != 1 or not frame.feed_source.isin(expected[mode]).all():
        raise ValueError(f"进给来源不匹配：{mode} 需要 {expected[mode]}")
    values = frame.feed_input_mm_min.to_numpy(float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("实际进给存在缺失或无效值")
    if mode == "predicted" and ("feed_model_version" not in frame or
            frame.feed_model_version.isna().any() or frame.feed_model_version.eq("").any()):
        raise ValueError("预测进给缺少 BiLSTM 模型版本")


def _prepare_process_features(process, plan, metadata):
    """Map process geometry only; this function does not estimate actual feed."""
    if metadata.get("program_version") != plan.program_version:
        raise ValueError("工艺对应程序版本与计划不匹配")
    if metadata.get("diameter_mm") is None or metadata.get("teeth") is None:
        raise ValueError("缺少刀具直径或齿数")
    required = {"line", "execution_instance", "segment_mm", "ap_mm", "ae_mm"}
    if required - set(process):
        raise ValueError(f"工艺表缺少标准字段：{sorted(required - set(process))}")
    output = plan.frame.reset_index(drop=True).copy()
    output["ap_mm"], output["ae_mm"], output["steady"] = np.nan, np.nan, False
    output["process_geometry_known"] = False
    output["process_geometry_conflict_support"] = False
    issues, mapping_report = [], []
    for (line, instance), rows in output.groupby(["line", "execution_instance"], sort=False):
        source = process[(process.line == line) & (process.execution_instance == instance)].sort_values("segment_mm")
        if len(source) < 2:
            issues.append(f"行 {line} 实例 {instance} 工艺信息不足两个位置点")
            continue
        x = source.segment_mm.to_numpy(float)
        if not np.isfinite(x).all():
            issues.append(f"行 {line} 实例 {instance} 工艺位置含非有限值")
            continue
        geometry = source[["ap_mm", "ae_mm"]].to_numpy(float)
        starts = np.flatnonzero(np.r_[True, np.diff(x) > 0])
        low, high = np.minimum.reduceat(geometry, starts), np.maximum.reduceat(geometry, starts)
        agrees = np.isfinite(low).all(axis=1) & (low == high).all(axis=1)
        counts = np.diff(np.r_[starts, len(x)])
        conflicts = (counts > 1) & ~agrees
        valid = agrees & (low >= 0).all(axis=1) & (low[:, 1] <= float(metadata["diameter_mm"]))
        x, geometry = x[starts], geometry[starts]
        position = rows.segment_mm.to_numpy(float)
        right = np.clip(np.searchsorted(x, position, side="left"), 0, len(x) - 1)
        left = np.clip(right - 1, 0, len(x) - 1)
        exact = np.isclose(position, x[right], rtol=0., atol=1e-10)
        inside = np.isfinite(position) & (position >= x[0]) & (position <= x[-1])
        contact = (geometry > 0).all(axis=1)
        same_contact = contact[left] == contact[right]
        supported = inside & np.where(exact, valid[right], valid[left] & valid[right] & same_contact)
        mapped = np.column_stack([np.interp(position, x, geometry[:, i]) for i in range(2)])
        # Two distinct zero-contact geometries must not interpolate into a
        # positive-contact state, and unknown knots must not be bridged.
        supported &= exact | ((mapped > 0).all(axis=1) == contact[right])
        conflict_support = inside & np.where(exact, conflicts[right], conflicts[left] | conflicts[right])
        output.loc[rows.index[supported], ["ap_mm", "ae_mm"]] = mapped[supported]
        output.loc[rows.index, "process_geometry_known"] = supported
        output.loc[rows.index, "process_geometry_conflict_support"] = conflict_support
        if "steady" in source:
            steady = np.logical_and.reduceat(source.steady.eq(True).fillna(False).to_numpy(bool), starts)
            output.loc[rows.index, "steady"] = supported & np.where(exact, steady[right], steady[left] & steady[right])
        mapping_report.append(dict(line=int(line), execution_instance=int(instance),
            duplicate_positions=int((counts > 1).sum()), conflicting_positions=x[conflicts].tolist(),
            conflict_support_points=int(conflict_support.sum()), known_geometry_points=int(supported.sum()),
            unsupported_geometry_points=int((~supported).sum())))
        if conflicts.any():
            issues.append(f"行 {line} 实例 {instance} 的{int(conflicts.sum())}个工艺位置几何冲突；"
                          f"屏蔽{int(conflict_support.sum())}个依赖点，不跨冲突插值")
    output["diameter_mm"], output["teeth"] = float(metadata["diameter_mm"]), float(metadata["teeth"])
    output["spindle_rpm"] = output.spindle_cmd_rpm
    output["feed_input_mm_min"] = np.nan
    output["feed_source"] = "missing"
    output["cutting_contact"] = np.isfinite(output.ap_mm) & np.isfinite(output.ae_mm) & (output.ap_mm > 0) & (output.ae_mm > 0)
    output["sequence_id"] = continuous_ids(output, known_only=True)
    return SampleSet(output, {**metadata, "machine_version": plan.machine_version,
                    "source": "process_and_nc_only", "issues": issues,
                    "load_process_rules_version": LOAD_PROCESS_RULES_VERSION,
                    "process_mapping_rules_version": PROCESS_MAPPING_RULES_VERSION,
                    "process_mapping_report": mapping_report})


def prepare_forward_features(process, plan, metadata, *, feed_prediction=None, prepared_geometry=None):
    """Prepare process preview; load inference additionally requires BiLSTM output.

    The old FeedPlan remains readable, but its motion-planner feed is never
    substituted for actual feed. Missing BiLSTM results leave a missing input.
    A data-supported coordinate version must supply the same verified geometry
    used for training; it cannot silently return to the historical mapping.
    """
    result = _select_process_geometry(process, plan, metadata, prepared_geometry)
    result.metadata["position_policy"] = resolve_position_policy({**result.metadata, **metadata})
    output = result.frame
    if feed_prediction is None:
        result.metadata["issues"].append("实际进给模型未接入，仅可查看工艺")
        return result
    info = feed_prediction.metadata
    if info.get("source") != "bilstm_actual_feed" or info.get("trained") is not True or not info.get("model_version"):
        raise ValueError("需要已训练 BiLSTM 的实际进给预测")
    if info.get("program_version") != plan.program_version or info.get("machine_version") != plan.machine_version:
        raise ValueError("预测进给的程序或机床版本不匹配")
    keys = ["line", "execution_instance", "segment_mm"]
    source = feed_prediction.frame
    if set(keys + ["path_mm", "feed_predicted_mm_min"]) - set(source):
        raise ValueError("预测进给缺少程序位置或数值")
    if source.duplicated(keys).any() or output.duplicated(keys).any():
        raise ValueError("预测进给存在重复程序位置，需要消歧")
    matched = output[keys + ["path_mm"]].merge(
        source[keys + ["path_mm", "feed_predicted_mm_min"]], on=keys,
        how="left", validate="one_to_one", suffixes=("", "_feed"), sort=False)
    if not np.allclose(matched.path_mm, matched.path_mm_feed, rtol=0, atol=1e-8):
        raise ValueError("预测进给与工艺采样位置不一致；请使用同一 NC 路径网格")
    output["feed_input_mm_min"] = matched.feed_predicted_mm_min.to_numpy()
    output["feed_source"] = "bilstm_actual_feed"
    output["feed_model_version"] = info["model_version"]
    assert_feed_inputs(output, "predicted")
    finite = np.isfinite(output[PHYSICS_COLUMNS]).all(axis=1)
    if finite.any():
        physical_quantities(output.loc[finite, PHYSICS_COLUMNS].to_numpy())
    result.metadata.update(feed_source="bilstm_actual_feed", feed_model_version=info["model_version"],
                           schema="afc.samples.2")
    return result


def prepare_predicted_load_features(process, feed_prediction, metadata, *, prepared_geometry=None):
    """Use the BiLSTM path grid directly, without invoking the old planner."""
    from .feed import FeedPlan
    info = feed_prediction.metadata
    if feed_prediction.frame.duplicated(["line", "execution_instance", "segment_mm"]).any():
        raise ValueError("预测进给存在重复程序位置，需要消歧")
    plan = FeedPlan(feed_prediction.frame, info["program_version"], info["machine_version"], False,
                    {"source": "bilstm_path_grid", "time_mapping": "unavailable"})
    return prepare_forward_features(process, plan, metadata, feed_prediction=feed_prediction,
                                    prepared_geometry=prepared_geometry)


def _checked_prepared_geometry(sample, plan, metadata):
    """Copy a geometry-only grid after checking its NC and tooling identity."""
    required = {"ap_mm", "ae_mm", "steady", "process_geometry_known",
                "process_geometry_conflict_support", "cutting_contact", "sequence_id",
                "diameter_mm", "teeth", "spindle_rpm", "feed_input_mm_min", "feed_source"}
    if required - set(sample.frame):
        raise ValueError("预备几何缺少工艺字段")
    if {"power_cut_w", "base_power_w", "valid_main", "valid_residual", "aligned"} & set(sample.frame):
        raise ValueError("预备几何不能包含旧监督标签或资格")
    for key, expected in (("program_version", plan.program_version), ("machine_version", plan.machine_version),
                          ("diameter_mm", metadata["diameter_mm"]), ("teeth", metadata["teeth"])):
        if sample.metadata.get(key) != expected:
            raise ValueError(f"预备几何的{key}与当前样本不一致")
    # Process mapping may replace process/context fields, never NC geometry.
    nc_columns = [column for column in plan.frame if column not in required | {"feed_predicted_mm_min"}]
    try:
        pd.testing.assert_frame_equal(sample.frame[nc_columns].reset_index(drop=True),
                                      plan.frame[nc_columns].reset_index(drop=True), check_exact=True)
    except (AssertionError, KeyError) as exc:
        raise ValueError("预备几何与当前NC网格不一致") from exc
    if not sample.frame.diameter_mm.eq(metadata["diameter_mm"]).all() or not sample.frame.teeth.eq(metadata["teeth"]).all():
        raise ValueError("预备几何刀具字段不一致")
    if sample.frame.feed_input_mm_min.notna().any() or not sample.frame.feed_source.eq("missing").all():
        raise ValueError("预备几何不能包含观测或预测进给")
    from copy import deepcopy
    return SampleSet(sample.frame.copy(deep=True), deepcopy(sample.metadata))


def _select_process_geometry(process, plan, metadata, prepared_geometry):
    declared = str(metadata.get("process_mapping_rules_version", ""))
    if prepared_geometry is not None:
        if (declared.startswith("afc.data_supported_process_coordinates.")
                and prepared_geometry.metadata.get("process_mapping_rules_version") != declared):
            raise ValueError("预备几何坐标规则版本与当前样本不一致")
        return _checked_prepared_geometry(prepared_geometry, plan, metadata)
    if declared.startswith("afc.data_supported_process_coordinates."):
        raise ValueError("数据支持坐标规则需要传入相同版本的预备几何，不能退回旧工艺映射")
    return _prepare_process_features(process, plan, metadata)


def prepare_load_samples(process, plan, measurement, base_model, metadata, *, line_map=None,
                         position_tolerance_mm=.1, length_tolerance_fraction=.05,
                         reference=None, session=None, event_timeline=None,
                         base_qualification=None, position_policy=None, prepared_geometry=None):
    """Process table uses exact NC line/instance/local arclength, not XYZ keys.

    Process fields: line, execution_instance, segment_mm, ap_mm, ae_mm, steady.
    Metadata holds provenance and explicit quality facts; absent facts remain
    unknown. A whole-record candidate can be exported without qualifying for
    training. ``command_only_v1`` assigns simultaneous measured labels to the
    command path; the legacy default retains actual-first alignment. Recorded
    XYZ never become prospective features. Measured actual feed is required for
    load training and never inferred from command positions. ``prepared_geometry``
    can supply a verified NC/process-only grid without repeating its mapping.
    """
    required_meta = ["combination", "parameter_group", "feature", "diameter_mm", "teeth", "program_version"]
    if metadata.get("group_id") is None:
        required_meta.append("round_id")
    else:
        from .evaluation_policy import group_key
        group_key(metadata)
    missing = [k for k in required_meta if k not in metadata or metadata[k] is None]
    if missing:
        raise ValueError(f"样本元数据缺项：{missing}")
    if metadata["program_version"] != plan.program_version:
        raise ValueError("工艺/实测对应的程序版本与进给计划不匹配")
    required_process = {"line", "execution_instance", "segment_mm", "ap_mm", "ae_mm", "steady"}
    if required_process - set(process):
        raise ValueError(f"工艺表缺少标准字段：{sorted(required_process - set(process))}")
    position_policy = resolve_position_policy(metadata, position_policy)
    position_source, position_columns = select_position_channels(measurement.values, position_policy=position_policy)
    measurement.require("program_line_raw", "spindle_power_w", "spindle_speed_rpm")
    if reference is not None:
        from .base_reference import verify_reference_record
        verify_reference_record(measurement, reference)
    if event_timeline is not None and event_timeline.program_version != plan.program_version:
        raise ValueError("基础功率审核事件时间线与当前NC程序版本不同")
    hybrid = (base_qualification or {}).get("mode") == "hybrid_estimated_labels"
    if hybrid:
        if any(value is not None for value in (reference, session, event_timeline)):
            raise ValueError("混合估计协议不允许额外偏置校正或替代事件时间线")
        from .hybrid_labels import prepare_hybrid_power
        base = prepare_hybrid_power(measurement, base_model, base_qualification, metadata)
        base_version = base["model_version"]
    elif (base_qualification or {}).get("mode") == "record_idle_labels":
        if any(value is not None for value in (base_model, reference, session, event_timeline)):
            raise ValueError("本记录空载标签不能同时使用动态模型、参考校正或事件时间线")
        from .record_idle import prepare_record_idle_power
        base = prepare_record_idle_power(measurement, base_qualification, metadata)
        base_version = base["model_version"]
    else:
        base = predict_base_from_measured_speed(measurement.time_s, measurement.values["spindle_speed_rpm"], base_model,
                                                reference=reference, session=session, record_source=measurement.source,
                                                event_timeline=event_timeline,
                                                qualification=base_qualification, record_context=metadata)
        base_version = base_model.version
    correction = base.get("reference_correction_w")
    if correction is None:
        correction = np.zeros(len(measurement.time_s))
    from .feed_samples import align_spatial_channels, extract_measured_feed
    from .main_stability import VERSION as MAIN_STABILITY_VERSION, stability_channels, apply_stability_columns
    actual_feed, feed_valid, feed_label_source = extract_measured_feed(
        measurement, allow_position_derivative=metadata.get("position_derivative_verified") is True)
    actual_feed = np.where(feed_valid, actual_feed, np.nan)
    forward = _select_process_geometry(process, plan, metadata, prepared_geometry)
    output = forward.frame
    speed = np.asarray(measurement.values["spindle_speed_rpm"], float)
    power = np.asarray(measurement.values["spindle_power_w"], float)
    base_power = np.asarray(base["power_w"], float)
    usable = (np.asarray(base["valid"], bool) & np.isfinite(power) & np.isfinite(base_power)
              & np.isfinite(speed) & np.isfinite(correction))
    speed_channels, _, speed_audit = stability_channels(output, measurement, line_map=line_map)
    aligned, alignment_info = align_spatial_channels(output, measurement,
        {"power_cut_w": power - base_power, "base_power_w": base_power,
         "base_reference_correction_w": correction, "feed_input_mm_min": actual_feed,
         **speed_channels},
        valid=usable, line_map=line_map, position_offset_mm=metadata.get("position_offset_mm"),
        position_tolerance_mm=position_tolerance_mm, length_tolerance_fraction=length_tolerance_fraction,
        position_policy=position_policy)
    geometry = output.process_geometry_known.to_numpy(bool)
    for column in ("power_cut_w", "base_power_w", "base_reference_correction_w", "feed_input_mm_min"):
        output[column] = aligned[column].where(geometry)
    output["alignment_error_mm"] = aligned.alignment_error_mm
    output["aligned"] = aligned.aligned.to_numpy(bool) & geometry
    apply_stability_columns(output, aligned)
    output["quality_pass"] = all(metadata.get(k) is True for k in QUALITY_FIELDS)
    issues = list(forward.metadata["issues"]) + alignment_info["issues"]
    if position_source is None:
        issues.append("缺少完整实际或指令XYZ位置，无法对齐负载标签")
    output["feed_actual_mm_min"] = output.feed_input_mm_min
    output["feed_source"] = "measured_actual_feed"
    masks = task_masks(output, metadata)
    finite = np.isfinite(output[PHYSICS_COLUMNS]).all(axis=1)
    output["valid_residual"] = masks["valid_residual"] & finite
    output["valid_main"] = masks["valid_main"] & finite
    estimated_authorized = hybrid and base.get("status") == "authorized_estimation_protocol" and base.get("authorized_for_training") is True
    if base["status"] != "validated_in_sample_domain" and not estimated_authorized:
        output[["valid_main", "valid_residual"]] = False
        issues.append("基础功率或参考校正尚未通过完整验证；只生成候选样本")
    if base.get("formal_training_release") is False and not estimated_authorized:
        output[["valid_main", "valid_residual"]] = False
        issues.append("审核事件辅助接口尚待独立标签验证；工程覆盖率不能自动放行正式训练")
    # Keep the prospective NC/process boundaries. Observed missing labels may
    # split supervised windows, but must not change a pre-machining prediction.
    output["training_sequence_id"] = continuous_ids(output, output.valid_residual.to_numpy(bool), known_only=True)
    if hybrid:
        label = base["hybrid_label_metadata"]
        for key in ("label_source", "reference_kind", "evidence_grade", "model_version", "original_model_status"):
            output[key if key != "model_version" else "label_model_version"] = label.get(key)
    return SampleSet(output, {**forward.metadata, **metadata, "base_version": base_version,
                    "main_stability_rules_version": MAIN_STABILITY_VERSION,
                    "main_stability_audit": speed_audit,
                    "base_prediction_version": base.get("prediction_version", base_version),
                    "base_interface_version": base.get("interface_version"),
                    "base_formal_training_release": base.get("formal_training_release"),
                    "base_qualification_report_sha256": base.get("qualification_report_sha256"),
                    "base_qualification_issues": base.get("qualification_issues", []),
                    "base_event_timeline_sha256": base.get("timeline_sha256"),
                    "base_timing_source": base.get("timing_source", base.get("timing")),
                    "base_status": base["status"], "base_reference": base.get("reference_adjustment"),
                    **({"hybrid_label": base["hybrid_label_metadata"], "base_authorized_estimation_training": True,
                        "formal_metrological_qualification": False} if hybrid else {}),
                    **({"record_idle_label": base["label_metadata"]} if "label_metadata" in base else {}),
                    "machine_version": plan.machine_version, "position_policy": position_policy,
                    "position_source": position_source, "position_columns": list(position_columns), "issues": issues,
                    "load_process_rules_version": LOAD_PROCESS_RULES_VERSION,
                    "process_mapping_rules_version": forward.metadata["process_mapping_rules_version"],
                    "process_mapping_report": forward.metadata["process_mapping_report"],
                    "spatial_alignment_version": alignment_info["spatial_alignment_version"],
                    "position_offset_mm": alignment_info.get("position_offset_mm"),
                    "feed_source": "measured_actual_feed", "feed_label_source": feed_label_source,
                    "quality_unknown": [k for k in QUALITY_FIELDS if k not in metadata],
                    "schema": "afc.samples.2", "sample_rules_version": SAMPLE_RULES_VERSION})


def grouped_folds(metadata, *, require_complete=True):
    metadata = list(metadata)
    result = validate_round_metadata(metadata, require_complete=require_complete)
    if result["issues"]:
        raise ValueError("完整九折协议检查失败：" + "；".join(i["message"] for i in result["issues"][:12]))
    folds = protocol_folds()
    if not require_complete:
        known = set(result["rounds"])
        for fold in folds:
            for key in ("train_rounds", "validation_rounds"):
                fold[key] = [r for r in fold[key] if r in known]
    return folds


def residual_labels(samples, main_prediction, main_version):
    labels = samples.frame.power_cut_w.to_numpy() - np.asarray(main_prediction)
    return labels, {"main_version": main_version, "sample_version": version_of(samples.metadata)}
