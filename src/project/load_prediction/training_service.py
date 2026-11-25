"""One business service for CLI, HTTP and desktop research workflows."""
from __future__ import annotations

from dataclasses import asdict, replace
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import hashlib
import json
import math
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd

from .common import checkpoint, file_hash, json_value, save_json, version_of
from .experiment_protocol import PROTOCOL_VERSION, protocol_folds
from .feed_samples import FEED_FEATURE_COLUMNS, FeedSample
from .samples import PHYSICS_COLUMNS, PATH_COLUMNS, SampleSet
from .evaluation_policy import group_key, scoring_parameters

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "load_prediction" / "service"
SERVICE_SCHEMA = "afc.training_service.1"


NOMINAL_ROUTE = "nominal_programmed_residual_v1"


def protocol(route=None):
    from .experiment_protocol import PARAMETER_GROUPS, round_spec
    result = dict(version=PROTOCOL_VERSION, folds=protocol_folds(), rounds=[round_spec(r) for r in range(1, 91)],
                parameter_groups=PARAMETER_GROUPS, feed_columns=list(FEED_FEATURE_COLUMNS),
                physics_columns=PHYSICS_COLUMNS, path_columns=PATH_COLUMNS,
                spacing_mm=.5, quantiles=[.05, .5, .95],
                stage_order=["feed", "freeze_feed", "main", "freeze_main", "residual_quantiles", "freeze_residual"],
                selection_order=["rmse_w", "mae_w", "high_load_underestimate_rate"],
                final_training="reinitialize_all_90_median_best_epochs", independent_test="new_cases_only")
    if route == NOMINAL_ROUTE:
        from .samples import path_feature_names
        result.update(route=route, feed_columns=[], path_columns=path_feature_names(route),
                      stage_order=["main", "freeze_main", "residual_quantiles", "freeze_residual"])
    elif route not in (None, "predicted_feed_v1"):
        raise ValueError("未登记的模型路线")
    return result


def balanced_metrics(records):
    """Reduce complete records -> five feature means -> round means, equally."""
    groups = {}
    for record in records:
        groups.setdefault(group_key(record), {}).setdefault(record["feature"], []).append(record["metrics"])
    keys = {k for r in records for k in r["metrics"] if k not in ("points", "count", "high_count", "low_count")}
    result = {}
    for key in keys:
        rounds = []
        for features in groups.values():
            values = []
            for items in features.values():
                available = [m[key] for m in items if isinstance(m.get(key), (int, float)) and np.isfinite(m[key])]
                if available:
                    values.append(float(np.mean(available)))
            if values:
                rounds.append(float(np.mean(values)))
        result[key] = float(np.mean(rounds)) if rounds else None
    result.update(records=len(records), rounds=len(groups), effective_counts={
        k: sum(int(r["metrics"].get(k, 0) or 0) for r in records) for k in ("count", "high_count", "low_count")})
    return result


def metrics_by_reference(records):
    """Report estimator-reference error separately from observed-idle-reference error."""
    names = sorted({r.get("reference_kind") for r in records if r.get("reference_kind")})
    return {name: balanced_metrics([r for r in records if r.get("reference_kind") == name]) for name in names}


def _merge_feature_metadata(features, metadata):
    """Input provenance cannot overwrite facts derived by the shared NC parser."""
    derived = features.metadata.copy()
    for key in ("program_version", "machine_version", "spacing_mm", "feed_override", "machining_mode", "machine_settings_version", "coordinate_mode"):
        if key in metadata and metadata[key] != derived[key]:
            raise ValueError(f"清单 {key} 与实际NC/机床配置不一致")
    features.metadata = {**metadata, **derived, "protocol_version": PROTOCOL_VERSION}
    return features


def _predicted_frame(sample, feed_sample, feed_model):
    from .feed_model import predict_actual_feed
    from .sample_rules import continuous_ids
    prediction = predict_actual_feed(feed_model, feed_sample)
    keys = ["line", "execution_instance", "segment_mm"]
    if sample.frame.duplicated(keys).any() or prediction.frame.duplicated(keys).any():
        raise ValueError("进给与负载路径位置必须唯一")
    frame = sample.frame.copy()
    matched = frame[keys + ["path_mm"]].merge(prediction.frame[keys + ["path_mm", "sequence_id", "feed_predicted_mm_min"]],
                                               on=keys, how="left", validate="one_to_one", suffixes=("", "_feed"), sort=False)
    if not np.allclose(matched.path_mm, matched.path_mm_feed, atol=1e-8, rtol=0):
        raise ValueError("进给与负载不使用同一空间网格")
    frame["feed_input_mm_min"] = matched.feed_predicted_mm_min.to_numpy()
    frame["feed_source"], frame["feed_model_version"] = "bilstm_actual_feed", feed_model.model_version
    frame["sequence_id"] = matched.sequence_id.to_numpy()
    frame["sequence_id"] = continuous_ids(frame, known_only=True)
    return frame


def _memory_exclusion(metadata):
    if metadata.get("replaces") or metadata.get("recollection"):
        return "paused_recollection_not_in_historical_order"
    if metadata.get("memory_eligible") is False:
        return "paused_memory_ineligible"
    return None


def evaluate_records(bundle, load_samples, *, feed_samples=None, feed_model=None, feed_mode=None, memory=None,
                     main_only=False, output_dir=None, cancel=None,
                     analysis_mode="audited_acquisition", analysis_tool_group=None, metric_parameters=None,
                     update_memory=True):
    from .metrics import regression_metrics
    from .models import predict_cutting_power
    from .sample_rules import continuous_ids
    index = {s.metadata["record_id"]: s for s in (feed_samples or [])}
    records = []
    score_parameters = scoring_parameters(bundle, metric_parameters)
    nominal = getattr(bundle.config, "route", "predicted_feed_v1") == NOMINAL_ROUTE
    feed_mode = feed_mode or ("programmed" if nominal else "predicted")
    if nominal and (feed_mode == "predicted" or feed_model is not None):
        raise ValueError("名义负载路线不得调用实际进给网络")
    if feed_mode not in ("predicted", "measured", "programmed"):
        raise ValueError("未登记的进给评价模式")
    if analysis_mode not in ("audited_acquisition", "numbered_round_pooled_tool"):
        raise ValueError("未登记的记忆分析模式")
    assumed = analysis_mode == "numbered_round_pooled_tool"
    if assumed and (not isinstance(analysis_tool_group, str) or not analysis_tool_group.strip()):
        raise ValueError("编号轮次敏感性分析必须显式登记 analysis_tool_group")
    assumption = (dict(source="explicit_analysis_request", order="ascending (round_id, PROGRAM_FEATURES order)",
                       tool_group=analysis_tool_group, interpretation="sensitivity_analysis_only",
                       limitation="Numbered rounds and a pooled tool group do not establish actual acquisition order or a tool session.")
                  if assumed else None)
    # Unknown order permits cold-start evaluation, never a memory update.
    ordered = sorted(load_samples, key=lambda s: (s.metadata.get("actual_event_order") is None,
                     s.metadata.get("actual_event_order") if s.metadata.get("actual_event_order") is not None else 0,
                     s.metadata["record_id"]))
    if assumed:
        from .experiment_protocol import PROGRAM_FEATURES
        feature_order = {name: i for i, name in enumerate(PROGRAM_FEATURES)}
        ordered = sorted(load_samples, key=lambda s: (s.metadata["round_id"], feature_order[s.metadata["feature"]], s.metadata["record_id"]))
    interval_available = bundle.config.residual_output == "quantiles" and not main_only
    analysis_order = 0
    for order, sample in enumerate(ordered):
        checkpoint(cancel)
        if nominal:
            from .samples import load_input_view
            frame = load_input_view(sample, feed_mode).frame
        elif feed_mode == "predicted":
            if feed_model is None or sample.metadata["record_id"] not in index:
                raise ValueError("正式端到端评价需要该折实际进给预测")
            frame = _predicted_frame(sample, index[sample.metadata["record_id"]], feed_model)
        else:
            frame = sample.frame.copy()
            if sample.metadata["record_id"] in index:
                known = index[sample.metadata["record_id"]].frame
                keys = ["line", "execution_instance", "segment_mm"]
                joined = frame[keys].merge(known[keys + ["sequence_id"]], on=keys, how="left", validate="one_to_one", sort=False)
                if joined.sequence_id.isna().any():
                    raise ValueError("评价位置缺少加工前NC边界")
                frame["sequence_id"] = joined.sequence_id.to_numpy()
            frame["sequence_id"] = continuous_ids(frame, known_only=True)
            if feed_mode == "programmed":
                frame["feed_input_mm_min"] = frame.feed_program_mm_min
                frame["feed_source"] = "programmed_feed_diagnostic"
        # Predictions depend only on available NC/process/input-feed values.
        # Observed validity is used AFTER inference, solely for scoring/updating.
        if bundle.execution_window_policy() == "prospective_v1":
            from .neural import prospective_residual_context
            selected, source_indices, _ = prospective_residual_context(frame, bundle.config.window_points)
            known_valid = np.zeros(len(frame), dtype=bool)
            known_valid[source_indices] = True
        else:
            known_valid = np.isfinite(frame[PHYSICS_COLUMNS + ["curvature_per_mm", "block_length_mm"]].to_numpy(float)).all(axis=1)
            if nominal:
                from .samples import NOMINAL_GEOMETRY_COLUMNS
                known_valid &= np.isfinite(frame[NOMINAL_GEOMETRY_COLUMNS].to_numpy(float)).all(axis=1)
            frame["sequence_id"] = continuous_ids(frame, known_valid, known_only=True)
            selected = frame.loc[known_valid].reset_index(drop=True)
        valid = frame.valid_residual.to_numpy(bool)
        if np.any(valid & ~known_valid):
            raise ValueError("有效评价标签缺少完整加工前输入")
        scored = selected.valid_residual.to_numpy(bool)
        context = {**sample.metadata, "protocol_version": PROTOCOL_VERSION,
                   "feed_model_version": feed_model.model_version if feed_model is not None else sample.metadata.get("feed_model_version")}
        recollected = bool(context.get("replaces") or context.get("recollection"))
        if assumed and not recollected:
            # Local compatibility context only; the immutable source metadata is never rewritten.
            context.update(analysis_tool_group=analysis_tool_group, tool_session="analysis:" + analysis_tool_group,
                           actual_event_order=analysis_order, memory_analysis_mode=analysis_mode)
            analysis_order += 1
        quality_fields = ("tool_healthy", "no_alarm", "complete", "alignment_reliable")
        memory_status = ("disabled" if memory is None else
                         _memory_exclusion(context) if _memory_exclusion(context) else
                         "paused_quality" if any(context.get(k) is not True for k in quality_fields) else
                         "paused_unknown_session_or_order" if not context.get("tool_session") or context.get("actual_event_order") is None else
                         "ready")
        active_memory = memory if memory_status == "ready" else None
        memory_context = {**context, "main_version": bundle.main_version, "residual_version": bundle.residual_version}
        history_counts = np.zeros(len(selected), dtype=int)
        if active_memory is not None:
            history_counts = np.asarray([active_memory.entries.get(
                active_memory._key(memory_context, row.line, row.segment_mm, row.execution_instance), {}).get("count", 0)
                for row in selected[["line", "segment_mm", "execution_instance"]].itertuples(index=False)], dtype=int)
        if selected.empty:
            result = {key: np.empty(0) for key in ("main_w", "residual_w", "memory_w", "cutting_power_w", "q05_w", "q50_w", "q95_w")}
        elif main_only:
            if active_memory is not None:
                raise ValueError("纯物理基线不使用残差记忆")
            from .training import _main_values
            from .samples import assert_feed_inputs
            from .neural import state_version
            assert_feed_inputs(selected, mode=feed_mode)
            if not bundle.main_trained or bundle.main_version != state_version(bundle.main):
                raise ValueError("物理基线主趋势尚未冻结")
            power = _main_values(bundle, SampleSet(selected, sample.metadata))
            result = dict(main_w=power, residual_w=np.zeros(len(power)), memory_w=np.zeros(len(power)), cutting_power_w=power)
        else:
            result = predict_cutting_power(bundle, selected, context=context, memory=active_memory,
                                           round_id=(sample.metadata.get("round_id") if sample.metadata.get("round_id") is not None
                                                     else sample.metadata.get("group_id")), feed_mode=feed_mode)
        if any(not np.isfinite(result[key]).all() for key in ("main_w", "residual_w", "memory_w", "cutting_power_w")):
            raise ValueError("可预测位置产生非有限输出，不能通过评分交集隐藏失败")
        metrics = regression_metrics(selected.power_cut_w.to_numpy()[scored], result["cutting_power_w"][scored],
                                     lower=result["q05_w"][scored] if interval_available else None, upper=result["q95_w"][scored] if interval_available else None,
                                     **score_parameters, allow_empty=True)
        record = dict(record_id=sample.metadata["record_id"], round_id=sample.metadata.get("round_id"),
                      group_id=sample.metadata.get("group_id"),
                      feature=sample.metadata["feature"], feed_mode=feed_mode, metrics=metrics,
                      label_source=sample.metadata.get("hybrid_label", {}).get("label_source",
                          "record_local_measured_idle" if sample.metadata.get("record_idle_label") else None),
                      reference_kind=sample.metadata.get("hybrid_label", {}).get("reference_kind",
                          "postoperative_measured_idle" if sample.metadata.get("record_idle_label") else None),
                      predicted_points=len(selected), scored_points=int(scored.sum()),
                      status="unpredictable_known_inputs" if selected.empty else "predicted_without_scored_labels" if not scored.any() else "scored",
                      coverage=dict(nc_grid_points=len(frame), predictable_points=int(known_valid.sum()),
                                    predicted_points=len(selected), label_eligible_points=int(valid.sum()), native_scored_points=int(scored.sum()),
                                    excluded_counts={"invalid_known_inputs": int((~known_valid).sum()),
                                                     "ineligible_label": int((known_valid & ~valid).sum())}),
                      metric_parameters=dict(score_parameters),
                      prediction_fingerprint=version_of(result["cutting_power_w"]),
                      acquisition=dict(tool_session=sample.metadata.get("tool_session"), actual_event_order=sample.metadata.get("actual_event_order")),
                      memory_analysis=dict(mode=analysis_mode, analysis_tool_group=analysis_tool_group if assumed else None,
                                           event_order=context.get("actual_event_order"), assumption_source=assumption),
                      memory_update={"status": memory_status, "updated": 0})
        baseline = result["main_w"] + result["residual_w"]
        baseline_error = baseline - selected.power_cut_w.to_numpy()
        prediction_error = result["cutting_power_w"] - selected.power_cut_w.to_numpy()
        if memory is not None:
            baseline_metrics = regression_metrics(selected.power_cut_w.to_numpy()[scored], baseline[scored],
                lower=(result["q05_w"] - result["memory_w"])[scored] if interval_available else None,
                upper=(result["q95_w"] - result["memory_w"])[scored] if interval_available else None,
                **score_parameters, allow_empty=True)
            record["paired_memory_comparison"] = dict(without_memory_metrics=baseline_metrics,
                with_memory_metrics=metrics, comparison="same_record_same_scored_points_before_update",
                error_sign="prediction_minus_actual", delta_sign="with_memory_minus_without_memory",
                mae_delta_w=metrics["mae_w"] - baseline_metrics["mae_w"] if scored.any() else None,
                rmse_delta_w=metrics["rmse_w"] - baseline_metrics["rmse_w"] if scored.any() else None,
                prior_history_count_min=int(history_counts[scored].min()) if scored.any() else None,
                prior_history_count_max=int(history_counts[scored].max()) if scored.any() else None,
                prior_history_count_mean=float(history_counts[scored].mean()) if scored.any() else None,
                prior_history_supported_points=int((history_counts[scored] >= memory.minimum_repeats).sum()),
                main_version=bundle.main_version, residual_version=bundle.residual_version,
                improvement_guaranteed=False)
            if active_memory is None:
                record["memory_update"]["rejected_fields"] = (
                    [key for key in quality_fields if context.get(key) is not True] if memory_status == "paused_quality" else
                    ["recollection"] if recollected else ["memory_eligible"] if context.get("memory_eligible") is False else
                    [key for key in ("tool_session", "actual_event_order") if context.get(key) is None or context.get(key) == ""])
        if output_dir is not None:
            directory = Path(output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            artifact = frame[["line", "execution_instance", "segment_mm", "path_mm", "feed_input_mm_min", "power_cut_w"]].copy().reset_index(drop=True)
            artifact["record_id"] = sample.metadata["record_id"]
            artifact["known_input_valid"] = known_valid
            artifact["prediction_available"] = known_valid
            artifact["label_eligible"] = valid
            artifact["scored"] = artifact["native_scored"] = known_valid & valid
            artifact["score_mask"] = known_valid & valid
            for key in ("label_source", "reference_kind"):
                artifact[key] = record[key]
            artifact["exclusion_reason"] = np.where(~known_valid, "invalid_known_inputs", np.where(~valid, "ineligible_label", ""))
            if memory is not None:
                for key, values in dict(memory_history_count_before_update=history_counts, without_memory_power_w=baseline,
                    without_memory_error_w=np.where(scored, baseline_error, np.nan),
                    with_memory_error_w=np.where(scored, prediction_error, np.nan),
                    memory_absolute_error_delta_w=np.where(scored, np.abs(prediction_error) - np.abs(baseline_error), np.nan)).items():
                    artifact[key] = np.nan
                    artifact.loc[known_valid, key] = values
            for key in ("main_w", "residual_w", "memory_w", "cutting_power_w", "q05_w", "q50_w", "q95_w"):
                if key in result and (interval_available or not key.startswith("q")):
                    artifact[key] = np.nan
                    artifact.loc[known_valid, key] = result[key]
            filename = f"record_{len(records):04d}.csv"
            artifact.to_csv(directory / filename, index=False, encoding="utf-8")
            record.update(prediction_file=filename, prediction_sha256=file_hash(directory / filename))
        if active_memory is not None and not update_memory:
            record["memory_update"] = {"status": "prediction_only", "updated": 0}
        elif active_memory is not None and scored.any():
            meta = sample.metadata
            record["memory_update"] = active_memory.update_round(memory_context,
                                selected.loc[scored], (selected.power_cut_w.to_numpy() - result["main_w"] - result["residual_w"])[scored],
                                (meta.get("round_id") if meta.get("round_id") is not None else meta.get("group_id")),
                                healthy=meta.get("tool_healthy"), no_alarm=meta.get("no_alarm"),
                                complete=meta.get("complete"), aligned=meta.get("alignment_reliable"),
                                event_order=context.get("actual_event_order"))
            if record["memory_update"]["status"] == "paused_quality":
                record["memory_update"]["rejected_fields"] = [key for key in ("tool_healthy", "no_alarm", "complete", "alignment_reliable")
                                                              if meta.get(key) is not True]
        elif active_memory is not None:
            record["memory_update"] = {"status": "paused_no_scored_labels", "updated": 0}
        records.append(record)
    report = dict(records=records, equal_weight_metrics=balanced_metrics(records),
                  metrics_by_reference=metrics_by_reference(records),
                  interval_status="evaluated_model_prediction_interval" if bundle.config.residual_output == "quantiles" and not main_only else "point_regression_no_interval", feed_mode=feed_mode,
                  nominal_coverage=.9 if bundle.config.residual_output == "quantiles" and not main_only else None,
                  coverage_guarantee=False,
                  interpretation=("postmachining_diagnostic" if feed_mode == "measured" else
                                  "numbered_round_pooled_tool_sensitivity_analysis" if assumed and memory is not None else "development_evaluation"))
    rejected = [r for r in records if memory is not None and r["memory_update"]["status"] != "updated"]
    report.update(memory_analysis=dict(mode=analysis_mode, assumption=assumption, update_order="score_before_update",
                      attempted_records=len(records) if memory is not None else 0, rejected_records=len(rejected),
                      rejection_counts={status: sum(r["memory_update"]["status"] == status for r in rejected)
                                        for status in sorted({r["memory_update"]["status"] for r in rejected})}))
    if output_dir is not None:
        if memory is not None:
            memory.save(Path(output_dir) / "memory.json")
        save_json(Path(output_dir) / "evaluation.json", report)
    return report


def evaluate_feed_records(model, samples, *, output_dir=None, cancel=None):
    from .feed_metrics import evaluate_feed_record, aggregate_feed_metrics
    from .feed_neural import predict_values
    records = []
    for sample in samples:
        checkpoint(cancel)
        predicted = predict_values(model, sample, cancel=cancel)
        record = dict(record_id=sample.metadata["record_id"], round_id=sample.metadata["round_id"],
                      feature=sample.metadata["feature"], metrics=evaluate_feed_record(sample.frame, predicted))
        if output_dir is not None:
            directory = Path(output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            artifact = sample.frame[["line", "execution_instance", "segment_mm", "path_mm", "feed_actual_mm_min", "valid_label"]].copy()
            artifact["feed_predicted_mm_min"] = predicted
            filename = f"record_{len(records):04d}.csv"
            artifact.to_csv(directory / filename, index=False, encoding="utf-8")
            record.update(prediction_file=filename, prediction_sha256=file_hash(directory / filename))
        records.append(record)
    result = dict(records=records, equal_weight_metrics=aggregate_feed_metrics(records), units="mm/min",
                  model_version=model.model_version)
    if output_dir is not None:
        save_json(Path(output_dir) / "evaluation.json", result)
    return result


def compare_evaluation_artifacts(directories, *, output_dir=None):
    """Re-score saved comparisons on exact common positions; retain native coverage.

    Call once per fold with all six comparison directories. Source predictions
    are never edited. The caller may aggregate these fold reports afterwards.
    """
    from .metrics import regression_metrics
    if len(directories) < 2:
        raise ValueError("共同评分至少需要两项已保存评价")
    reports, records_by_name = {}, {}
    for name, directory in directories.items():
        reports[name] = json.loads((Path(directory) / "evaluation.json").read_text(encoding="utf-8"))
        rows = reports[name]["records"]
        records_by_name[name] = {row["record_id"]: row for row in rows}
        if len(records_by_name[name]) != len(rows):
            raise ValueError("评价记录ID重复")
    names = list(directories)
    members = set(records_by_name[names[0]])
    if any(set(rows) != members for rows in records_by_name.values()):
        raise ValueError("比较记录成员不一致，不能通过取交集隐藏缺失记录")
    output = {name: [] for name in names}
    coverage_rows = []
    keys = ["line", "execution_instance", "segment_mm", "path_mm"]
    for number, record_id in enumerate(records_by_name[names[0]]):
        frames = {}
        for name in names:
            meta = records_by_name[name][record_id]
            path = Path(directories[name]) / meta["prediction_file"]
            if file_hash(path) != meta["prediction_sha256"]:
                raise ValueError("逐点预测已改变，拒绝共同评分")
            frame = pd.read_csv(path, float_precision="round_trip")
            if frame.duplicated(keys).any():
                raise ValueError("比较位置键重复")
            frames[name] = frame.set_index(keys)
        reference = frames[names[0]]
        for name in names[1:]:
            if records_by_name[name][record_id]["metric_parameters"] != records_by_name[names[0]][record_id]["metric_parameters"]:
                raise ValueError("比较评分参数不一致；必须共用冻结尺度与阈值")
            if len(frames[name]) != len(reference) or set(frames[name].index) != set(reference.index):
                raise ValueError("比较NC网格不一致；必须保留原始完整网格")
            frames[name] = frames[name].reindex(reference.index)
            if not np.array_equal(reference.power_cut_w.to_numpy(), frames[name].power_cut_w.to_numpy(), equal_nan=True):
                raise ValueError("相同位置的比较标签不一致")
        common = np.logical_and.reduce([frame.native_scored.to_numpy(bool) for frame in frames.values()])
        mask_frame = reference.reset_index()[keys].copy()
        mask_frame["record_id"], mask_frame["common_scored"] = record_id, common
        for name, frame in frames.items():
            meta = records_by_name[name][record_id]
            native = frame.native_scored.to_numpy(bool)
            available = frame.prediction_available.to_numpy(bool)
            if not np.isfinite(frame.loc[available, "cutting_power_w"]).all():
                raise ValueError("可预测位置存在非有限输出，不能在共同评分中静默排除")
            interval = reports[name].get("nominal_coverage") is not None
            metrics = regression_metrics(frame.power_cut_w.to_numpy()[common], frame.cutting_power_w.to_numpy()[common],
                lower=frame.q05_w.to_numpy()[common] if interval else None,
                upper=frame.q95_w.to_numpy()[common] if interval else None,
                **meta["metric_parameters"], allow_empty=True)
            coverage = {**meta["coverage"], "common_scored_points": int(common.sum()),
                        "excluded_from_native_for_common": int((native & ~common).sum())}
            output[name].append(dict(record_id=record_id, round_id=meta["round_id"], group_id=meta.get("group_id"), feature=meta["feature"],
                                     label_source=meta.get("label_source"), reference_kind=meta.get("reference_kind"),
                                     metrics=metrics, native_metrics=meta["metrics"], coverage=coverage))
            mask_frame[name + "_native_scored"] = native
            coverage_rows.append(dict(comparison=name, record_id=record_id, **coverage))
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            mask_frame.to_csv(Path(output_dir) / f"record_{number:04d}_common_mask.csv", index=False, encoding="utf-8")
    comparisons = {name: dict(records=rows, equal_weight_metrics=balanced_metrics(rows),
                              metrics_by_reference=metrics_by_reference(rows),
                              native_equal_weight_metrics=reports[name]["equal_weight_metrics"]) for name, rows in output.items()}
    paired = []
    for first, left in enumerate(names):
        for right in names[first + 1:]:
            for a, b in zip(output[left], output[right]):
                deltas = {key: b["metrics"][key] - value for key, value in a["metrics"].items()
                          if key not in ("points", "count", "high_count", "low_count")
                          and isinstance(value, (int, float)) and isinstance(b["metrics"].get(key), (int, float))}
                paired.append(dict(left=left, right=right, record_id=a["record_id"], round_id=a["round_id"],
                                   feature=a["feature"], delta_sign="right_minus_left", deltas=deltas))
    result = dict(schema="afc.common_evaluation.1", comparisons=comparisons, paired_differences=paired,
                  coverage=coverage_rows, position_rule="exact_unrounded_NC_grid", source_directories={k: str(v) for k, v in directories.items()})
    if output_dir is not None:
        save_json(Path(output_dir) / "comparison.json", result)
    return result


def train_feed_cached(train, validation=None, *, training_config, model_config, cache_dir=None,
                      checkpoint_path=None, resume=False, cancel=None, progress=None):
    """Reuse only identical frozen fold models; partial work needs explicit resume."""
    from .feed_training import train_feed_fold
    from .feed_model import load_feed_model
    from .training_checkpoint import digest
    from .feed_identity import feed_training_identity
    # Recheck roles, duplicate content and masks even when weights are reusable.
    check = train_feed_fold(train, validation, config=training_config, model_config=model_config)
    if not check["ready_to_train"]:
        raise ValueError("进给复用前置检查失败：" + "；".join(check["blockers"]))
    identity = feed_training_identity(train, validation, training_config=training_config,
        model_config=model_config, full=validation is None) if cache_dir else None
    directory = Path(cache_dir) / "feed_reuse_v1" / digest(identity) if cache_dir else None
    if directory is not None and (directory / "model.json").exists():
        audit = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        if audit.get("identity") != json_value(identity) or audit.get("sha256") != file_hash(directory / "feed.pt"):
            raise ValueError("进给复用缓存的指纹或权重哈希不一致")
        model = load_feed_model(directory / "feed.pt")
        if progress:
            progress(dict(phase="reused_frozen_feed", model_version=model.model_version, cache=str(directory)))
        return model
    model = train_feed_fold(train, validation, config=training_config, model_config=model_config,
                            execute=True, cancel=cancel, progress=progress, checkpoint_path=checkpoint_path,
                            resume=resume)
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
        model.save(directory / "feed.pt")
        save_json(directory / "model.json", dict(identity=identity, sha256=file_hash(directory / "feed.pt")))
    return model


def _feed_reference(path, model, train, validation, training_config, model_config):
    from .feed_identity import feed_training_identity
    return dict(path=str(Path(path).resolve()), sha256=file_hash(path), model_version=model.model_version,
                best_epoch=model.metadata["best_epoch"], identity=feed_training_identity(train, validation,
                    training_config=training_config, model_config=model_config, full=validation is None))


def _load_feed_reference(reference, train, validation, training_config, model_config, *, artifact_relocations=None):
    """An explicit source never falls back to an optimizer or another cache."""
    from .feed_identity import feed_training_identity
    from .feed_model import load_feed_model
    from .feed_training import train_feed_fold
    from .artifact_paths import ArtifactResolver
    check = train_feed_fold(train, validation, config=training_config, model_config=model_config)
    if not check["ready_to_train"]:
        raise ValueError("来源进给复用前置不合格：" + "；".join(check["blockers"]))
    expected = feed_training_identity(train, validation, training_config=training_config,
                                      model_config=model_config, full=validation is None)
    path = ArtifactResolver(artifact_relocations).resolve(reference["path"])
    if (reference.get("identity") != json_value(expected) or not path.is_file()
            or file_hash(path) != reference.get("sha256")):
        raise ValueError("来源进给数值、配置、依赖或文件哈希不一致；禁止静默重训")
    model = load_feed_model(path)
    if model.model_version != reference.get("model_version") or model.metadata["best_epoch"] != reference.get("best_epoch"):
        raise ValueError("来源进给模型版本或最佳epoch不一致")
    if validation is None and (model.metadata.get("training_mode") != "fixed_epochs_full"
                               or model.metadata["best_epoch"] != training_config.epochs):
        raise ValueError("全量进给来源不是对应固定epoch的重新初始化训练")
    return model


def _inherit_hybrid_parent_records(current, parent_source, feeds, loads):
    """Bind structural inspection to frozen inputs and observed contact masks.

    Measurement bytes for unchanged labels remain inherited from the verified
    parent snapshot. NC/process/machine files are small and rehashed once each;
    the changed-label branch separately verifies all four current input files.
    No author-confirmation field or sample mask is manufactured here.
    """
    from .sample_rules import contact_mask, motion_mask

    if parent_source.get("files_verified") is not True:
        raise ValueError("父快照来源文件尚未核验，不能继承接触资格")
    source = {m["record_id"]: m for m in parent_source["records"] if m.get("status") == "active"}
    feed_index = {s.metadata["record_id"]: s for s in feeds}
    load_index = {s.metadata["record_id"]: s for s in loads}
    required = {"nc", "process", "measurement", "machine"}
    checked_files, receipts = {}, []
    for meta in current:
        rid = meta["record_id"]
        old, feed, load = source[rid], feed_index[rid], load_index[rid]
        hashes = old.get("file_sha256", {})
        if (set(hashes) != required or any(not isinstance(v, str) or len(v) != 64 for v in hashes.values())
                or feed.metadata.get("file_sha256") != hashes or load.metadata.get("file_sha256") != hashes):
            raise ValueError(f"父记录四项来源哈希未一致绑定：{rid}")
        for key, value in meta.items():
            if key == "file_sha256":
                if value and value != hashes:
                    raise ValueError(f"增量清单改变了来源哈希：{rid}")
            elif key == "resolved_files":
                if set(value) != required or any(
                        Path(path).resolve() != Path(old[key][kind]).resolve() for kind, path in value.items()):
                    raise ValueError(f"增量清单改变了输入路径：{rid}")
            elif old.get(key) != value:
                raise ValueError(f"增量清单改变了既有输入/审核：{rid} {key}")
        for kind in ("nc", "process", "machine"):
            path = Path(old["resolved_files"][kind]).resolve()
            if path not in checked_files:
                checked_files[path] = file_hash(path)
            if checked_files[path] != hashes[kind]:
                raise ValueError(f"继承接触资格的来源文件改变：{rid} {kind}")
        if (any(m.get("process_aligned") is not True for m in (old, feed.metadata, load.metadata))
                or not old.get("process_version") or "cutting_contact" not in feed.frame
                or "valid_label" not in feed.frame):
            raise ValueError(f"父记录缺少已对齐工艺和实际接触证据：{rid}")
        contact = contact_mask(feed.frame)
        valid = feed.frame.valid_label.eq(True).fillna(False).to_numpy(bool)
        if not contact.any() or np.any(valid & ~(contact & motion_mask(feed.frame))):
            raise ValueError(f"父记录接触证据未证实或有效标签越出接触范围：{rid}")
        meta.update(deepcopy(old))
        receipts.append(dict(record_id=rid, file_sha256=deepcopy(hashes),
            basis="verified_parent_process_alignment_and_frozen_feed_contact_mask",
            contact_points=int(contact.sum()), valid_feed_points=int(valid.sum()),
            measurement_hash_basis="verified_parent_snapshot_not_raw_reread",
            geometry_hash_basis="current_nc_process_machine_sha256"))
    return receipts


def validate_paper1_fixed_conditions(samples):
    """Paper 1 may only use frozen single-level/unknown condition branches."""
    from .condition import fit_condition_spec
    spec = fit_condition_spec([sample.metadata if hasattr(sample, "metadata") else sample for sample in samples])
    if spec["interactions"] is not False or any(branch["active"] for branch in spec["branches"].values()):
        raise ValueError("论文1固定工况要求五分支均冻结且交互关闭，不能拟合不可辨识工况效应")
    return dict(condition_version=spec["version"], branches={key: "fixed_reference" for key in spec["branches"]},
                interactions=False, confounded_pairs=spec["confounded_pairs"])


def _nominal_candidates(candidates):
    from .models import ModelConfig
    from .training import TrainingConfig
    if not candidates or len({c["id"] for c in candidates}) != len(candidates):
        raise ValueError("名义负载候选必须非空且ID唯一")
    for candidate in candidates:
        name = candidate["id"]
        if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("候选ID必须为安全目录名")
        if candidate.get("feed_model") or candidate.get("feed_training"):
            raise ValueError("名义负载候选不能包含实际进给网络配置")
        config = ModelConfig(**candidate.get("load_model", {}))
        config.validate()
        TrainingConfig(**candidate.get("load_training", {})).validate()
        if config.route != NOMINAL_ROUTE:
            raise ValueError("load任务只接受明确的名义负载路线")
    from .load_device import request_cpu_threads
    request_cpu_threads({"mode": "load", "candidates": candidates})
    return candidates


def _fold_metric_summary(folds, key="without_memory"):
    rows = [fold[key]["equal_weight_metrics"] for fold in folds]
    names = {name for row in rows for name, value in row.items()
             if name not in ("records", "rounds") and isinstance(value, (int, float))}
    result = {}
    for name in sorted(names):
        values = [row[name] for row in rows if isinstance(row.get(name), (int, float)) and np.isfinite(row[name])]
        result[name] = dict(mean=float(np.mean(values)) if values else None,
                           sample_std=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                           effective_folds=len(values))
    return result


def _run_nominal_development(load_samples, candidates, *, execute=False, cancel=None, progress=None,
                             output_dir=None, resume=False, analysis_mode="audited_acquisition", analysis_tool_group=None):
    """Use the shared physical/residual trainer and evaluator; no feed model exists."""
    from .condition import fit_condition_spec
    from .memory import PositionMemory
    from .models import ModelConfig
    from .training import TrainingConfig, train_two_stage, save_main_baseline
    candidates = _nominal_candidates(candidates)
    folds = protocol_folds()
    if not execute:
        return dict(status="prepared_not_trained", route=NOMINAL_ROUTE, folds=folds, candidates=candidates,
                    phases=[dict(candidate=c["id"], fold=f["fold"], stages=["main", "residual"])
                            for c in candidates for f in folds], final_stages=["main", "residual"], feed_network_used=False)
    results = []
    for candidate in candidates:
        fold_results = []
        for fold in folds:
            checkpoint(cancel)
            train = [s for s in load_samples if s.metadata["round_id"] in fold["train_rounds"]]
            val = [s for s in load_samples if s.metadata["round_id"] in fold["validation_rounds"]]
            directory = Path(output_dir) / candidate["id"] / fold["fold"] if output_dir else None
            def report(value):
                if progress:
                    progress(dict(candidate=candidate["id"], fold=fold["fold"], **value))
            cfg = {**candidate["load_model"], "condition_spec": fit_condition_spec([s.metadata for s in train])}
            training = TrainingConfig(**candidate.get("load_training", {}))
            bundle = train_two_stage(train, validation=val, model_config=ModelConfig(**cfg), config=training,
                execute=True, cancel=cancel, progress=report, resume=resume,
                checkpoint_dir=directory / "checkpoints" if directory else None)
            if directory:
                directory.mkdir(parents=True, exist_ok=True)
                bundle.save(directory / "load.pt")
                save_json(directory / "load_history.json", bundle.metadata)
            destinations = {name: directory / name if directory else None for name in
                            ("fixed_physical", "physical_reused_main", "nominal_no_memory", "with_memory", "measured_feed_diagnostic")}
            no_memory = evaluate_records(bundle, val, feed_mode="programmed", output_dir=destinations["nominal_no_memory"], cancel=cancel)
            physical = evaluate_records(bundle, val, feed_mode="programmed", main_only=True,
                                        output_dir=destinations["physical_reused_main"], cancel=cancel)
            diagnostic = evaluate_records(bundle, val, feed_mode="measured", output_dir=destinations["measured_feed_diagnostic"], cancel=cancel)
            with_memory = evaluate_records(bundle, val, feed_mode="programmed", memory=PositionMemory(),
                output_dir=destinations["with_memory"], cancel=cancel, analysis_mode=analysis_mode, analysis_tool_group=analysis_tool_group)
            with_memory["memory_analysis"].update(history_source="current_fold_validation_only", initial_memory="empty",
                allowed_record_ids=[s.metadata["record_id"] for s in val], validation_rounds=fold["validation_rounds"])
            baseline_cfg = {**cfg, "physical_mode": "fixed", "use_condition_effects": False, "use_residual_modulation": False}
            baseline = train_two_stage(train, validation=val, model_config=ModelConfig(**baseline_cfg), config=training,
                execute=True, main_only=True, cancel=cancel, resume=resume,
                progress=lambda value: report({**value, "comparison": "fixed_physical"}),
                checkpoint_dir=directory / "baseline_checkpoints" if directory else None)
            baseline_result = evaluate_records(baseline, val, feed_mode="programmed", main_only=True,
                                              output_dir=destinations["fixed_physical"], cancel=cancel)
            comparison = None
            if directory:
                save_json(destinations["with_memory"] / "evaluation.json", with_memory)
                save_main_baseline(baseline, directory / "fixed_physical.pt")
                comparison = compare_evaluation_artifacts(destinations, output_dir=directory / "common_comparison")
            result = dict(fold=fold["fold"], validation_rounds=fold["validation_rounds"],
                without_memory=no_memory, physical_reused_main=physical, with_memory=with_memory,
                measured_feed_diagnostic=diagnostic, fixed_physical=baseline_result, common_comparison=comparison,
                best_epochs={stage: bundle.metadata["stages"][stage]["best_epoch"] for stage in ("main", "residual")})
            fold_results.append(result)
            if directory:
                save_json(directory / "validation.json", result)
        summary = _fold_metric_summary(fold_results)
        results.append(dict(candidate=candidate, folds=fold_results, fold_summary=summary,
                            metrics={name: value["mean"] for name, value in summary.items()}))
    chosen = min(results, key=lambda r: tuple(r["metrics"].get(k) if r["metrics"].get(k) is not None else float("inf")
                                            for k in ("rmse_w", "mae_w", "high_load_underestimate_rate")))
    epochs = {stage: math.ceil(float(np.median([f["best_epochs"][stage] for f in chosen["folds"]])))
              for stage in ("main", "residual")}
    return dict(status="developed", route=NOMINAL_ROUTE, selected=chosen["candidate"], full_epochs=epochs,
                candidates=results, protocol_version=PROTOCOL_VERSION, independent_test_used=False, feed_network_used=False)


def run_development(load_samples, *, feed_samples=None, candidates=None, model_config=None,
                    training_config=None, execute=False, cancel=None, progress=None, output_dir=None, main_only=False,
                    feed_cache_dir=None, resume=False, analysis_mode="audited_acquisition", analysis_tool_group=None,
                    feed_source=None):
    """Nine folds select a configuration; all networks are newly initialized each fold."""
    from .condition import fit_condition_spec
    from .feed_model import FeedModelConfig
    from .feed_training import FeedTrainingConfig, train_feed_fold
    from .models import ModelConfig
    from .training import TrainingConfig, train_two_stage
    candidates = candidates or [{"id": "default", "load_model": asdict(model_config or ModelConfig()),
                                  "load_training": asdict(training_config or TrainingConfig())}]
    if any(c.get("load_model", {}).get("route") == NOMINAL_ROUTE for c in candidates):
        if feed_source is not None or main_only:
            raise ValueError("名义负载路线不接受feed来源或独立旧对照调度")
        return _run_nominal_development(load_samples, candidates, execute=execute, cancel=cancel, progress=progress,
            output_dir=output_dir, resume=resume, analysis_mode=analysis_mode, analysis_tool_group=analysis_tool_group)
    if len({c["id"] for c in candidates}) != len(candidates):
        raise ValueError("候选配置ID必须唯一")
    if any(not isinstance(c["id"], str) or not c["id"] or Path(c["id"]).name != c["id"] or c["id"] in (".", "..") for c in candidates):
        raise ValueError("候选配置ID必须为安全的单个目录名")
    folds = protocol_folds()
    if not execute:
        return dict(status="prepared_not_trained", folds=folds, candidates=candidates,
                    phases=[dict(candidate=c["id"], fold=f["fold"], stages=["feed", "main"] if main_only else ["feed", "main", "residual"])
                            for c in candidates for f in folds], final_stages=["feed", "main"] if main_only else ["feed", "main", "residual"])
    if not feed_samples:
        raise ValueError("完整流程需要实际进给监督样本")
    results = []
    for candidate in candidates:
        fold_results = []
        for fold in folds:
            checkpoint(cancel)
            train = [s for s in load_samples if s.metadata["round_id"] in fold["train_rounds"]]
            val = [s for s in load_samples if s.metadata["round_id"] in fold["validation_rounds"]]
            ft = [s for s in feed_samples if s.metadata["round_id"] in fold["train_rounds"]]
            fv = [s for s in feed_samples if s.metadata["round_id"] in fold["validation_rounds"]]
            directory = Path(output_dir) / candidate["id"] / fold["fold"] if output_dir else None
            def report(value):
                if progress:
                    progress(dict(candidate=candidate["id"], fold=fold["fold"], **value))
            if feed_source is not None:
                feed = _load_feed_reference(feed_source["folds"][fold["fold"]], ft, fv,
                    FeedTrainingConfig(**candidate.get("feed_training", {})), FeedModelConfig(**candidate.get("feed_model", {})),
                    artifact_relocations=feed_source.get("artifact_relocations"))
                report(dict(stage="feed", phase="reused_frozen_feed", source_feed_job_id=feed_source["source_feed_job_id"],
                            model_version=feed.model_version))
            else:
                feed = train_feed_cached(ft, fv, training_config=FeedTrainingConfig(**candidate.get("feed_training", {})),
                                   model_config=FeedModelConfig(**candidate.get("feed_model", {})),
                                   cache_dir=feed_cache_dir,
                                   checkpoint_path=directory / "checkpoints" / "feed.pt" if directory else None,
                                   resume=resume, cancel=cancel, progress=lambda x: report(dict(stage="feed", **x)))
            if directory is not None:
                directory.mkdir(parents=True, exist_ok=True)
                feed.save(directory / "feed.pt")
                save_json(directory / "feed_history.json", feed.metadata)
            cfg = dict(candidate.get("load_model", {}))
            if cfg.get("condition_input") == "identity_only_diagnostic":
                from .comparison_models import fit_identity_condition_spec
                cfg["condition_spec"] = fit_identity_condition_spec([s.metadata for s in train])
            else:
                cfg["condition_spec"] = fit_condition_spec([s.metadata for s in train])
            bundle = train_two_stage(train, validation=val, model_config=ModelConfig(**cfg),
                                     config=TrainingConfig(**candidate.get("load_training", {})),
                                     execute=True, cancel=cancel, progress=report, main_only=main_only,
                                     checkpoint_dir=directory / "checkpoints" if directory else None,
                                     resume=resume)
            if directory is not None:
                if main_only:
                    from .training import save_main_baseline
                    save_main_baseline(bundle, directory / "main_baseline.pt")
                else:
                    bundle.save(directory / "load.pt")
                save_json(directory / "load_history.json", bundle.metadata)
            def destination(name):
                return directory / name if directory else None
            no_memory = evaluate_records(bundle, val, feed_samples=fv, feed_model=feed, main_only=main_only,
                                         output_dir=destination("predicted_feed"), cancel=cancel)
            physical = no_memory if main_only else evaluate_records(bundle, val, feed_samples=fv, feed_model=feed,
                main_only=True, output_dir=destination("physical_reused_main"), cancel=cancel)
            physical["comparison"] = dict(kind="same_fold_frozen_main_ablation", network_retrained=False,
                main_version=bundle.main_version, feed_model_version=feed.model_version,
                validation_rounds=fold["validation_rounds"], source_load_file=None if main_only or directory is None else "../load.pt")
            if directory is not None and not main_only:
                save_json(directory / "physical_reused_main" / "evaluation.json", physical)
            feed_details = evaluate_feed_records(feed, fv, output_dir=destination("feed_predictions"), cancel=cancel)
            diagnostic = evaluate_records(bundle, val, feed_samples=fv, feed_mode="measured", main_only=main_only,
                                          output_dir=destination("measured_feed_diagnostic"), cancel=cancel)
            programmed = evaluate_records(bundle, val, feed_samples=fv, feed_mode="programmed", main_only=main_only,
                                          output_dir=destination("programmed_feed"), cancel=cancel)
            from .memory import PositionMemory
            with_memory = (dict(status="not_applicable_main_only", records=[]) if main_only else
                           evaluate_records(bundle, val, feed_samples=fv, feed_model=feed, memory=PositionMemory(),
                                            output_dir=destination("with_memory"), cancel=cancel,
                                            analysis_mode=analysis_mode, analysis_tool_group=analysis_tool_group))
            if not main_only:
                with_memory["memory_analysis"].update(history_source="current_fold_validation_only", initial_memory="empty",
                    allowed_record_ids=[sample.metadata["record_id"] for sample in val], validation_rounds=fold["validation_rounds"])
                if directory is not None:
                    save_json(directory / "with_memory" / "evaluation.json", with_memory)
            result = dict(fold=fold["fold"], validation_rounds=fold["validation_rounds"],
                          without_memory=no_memory, measured_feed_diagnostic=diagnostic, programmed_feed=programmed,
                          physical_reused_main=physical,
                          with_memory=with_memory, feed_metrics=feed_details,
                          best_epochs=dict(feed=feed.metadata["best_epoch"],
                                           main=bundle.metadata["stages"]["main"]["best_epoch"],
                                           residual=bundle.metadata["stages"]["residual"]["best_epoch"]))
            fold_results.append(result)
            if output_dir:
                checkpoint(cancel)
                save_json(directory / "validation.json", result)
        all_records = [r for f in fold_results for r in f["without_memory"]["records"]]
        results.append(dict(candidate=candidate, folds=fold_results, metrics=balanced_metrics(all_records)))
    def rank(result):
        return tuple(float(result["metrics"].get(k)) if result["metrics"].get(k) is not None else float("inf")
                     for k in ("rmse_w", "mae_w", "high_load_underestimate_rate"))
    chosen = min(results, key=rank)
    epochs = {stage: math.ceil(float(np.median([f["best_epochs"][stage] for f in chosen["folds"]])))
              for stage in ("feed", "main", "residual")}
    return dict(status="developed", selected=chosen["candidate"], full_epochs=epochs, candidates=results,
                protocol_version=PROTOCOL_VERSION, independent_test_used=False)


def train_full(load_samples, feed_samples, development, *, output_dir, cancel=None, progress=None,
               checkpoint_dir=None, feed_cache_dir=None, resume=False, feed_source=None):
    from .condition import fit_condition_spec
    from .feed_model import FeedModelConfig
    from .feed_training import FeedTrainingConfig, train_feed_fold
    from .models import ModelConfig
    from .training import TrainingConfig, train_two_stage
    candidate, epochs = development["selected"], development["full_epochs"]
    if development.get("route") == NOMINAL_ROUTE:
        if feed_source is not None:
            raise ValueError("名义负载全量训练不接受feed来源")
        _nominal_candidates([candidate])
        cfg = {**candidate["load_model"], "condition_spec": fit_condition_spec([s.metadata for s in load_samples])}
        config = replace(TrainingConfig(**candidate.get("load_training", {})),
                         main_epochs=epochs["main"], residual_epochs=epochs["residual"])
        bundle = train_two_stage(load_samples, model_config=ModelConfig(**cfg), config=config, execute=True,
                                 cancel=cancel, progress=progress, checkpoint_dir=checkpoint_dir, resume=resume)
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=resume)
        bundle.save(directory / "load.pt")
        save_json(directory / "load_history.json", bundle.metadata)
        return None, bundle
    feed_config = FeedTrainingConfig(**candidate.get("feed_training", {}))
    if feed_source is not None:
        feed = _load_feed_reference(feed_source["full"], feed_samples, None,
            replace(feed_config, epochs=epochs["feed"]), FeedModelConfig(**candidate.get("feed_model", {})),
            artifact_relocations=feed_source.get("artifact_relocations"))
        if progress:
            progress(dict(stage="feed", phase="reused_frozen_full_feed", source_feed_job_id=feed_source["source_feed_job_id"],
                          model_version=feed.model_version))
    else:
        feed = train_feed_cached(feed_samples, training_config=replace(feed_config, epochs=epochs["feed"]),
                           model_config=FeedModelConfig(**candidate.get("feed_model", {})),
                           checkpoint_path=Path(checkpoint_dir) / "feed.pt" if checkpoint_dir else None,
                           cache_dir=None, resume=resume,
                           cancel=cancel, progress=(lambda x: progress(dict(stage="feed", **x))) if progress else None)
    cfg = dict(candidate.get("load_model", {}))
    cfg["condition_spec"] = fit_condition_spec([s.metadata for s in load_samples])
    bundle = train_two_stage(load_samples, model_config=ModelConfig(**cfg),
                             config=replace(TrainingConfig(**candidate.get("load_training", {})),
                                            main_epochs=epochs["main"], residual_epochs=epochs["residual"]),
                             execute=True, cancel=cancel, progress=progress,
                             checkpoint_dir=checkpoint_dir, resume=resume)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=resume)
    checkpoint(cancel)
    feed.save(directory / "feed.pt")
    bundle.save(directory / "load.pt")
    return feed, bundle


class TrainingService:
    def __init__(self, output_dir=None):
        self.output_dir = Path(output_dir or DEFAULT_OUTPUT).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def protocol(self):
        return protocol()

    def validate_feed_source_job(self, source_job_id, snapshot_id, candidates, *,
                                 source_feed_service_output=None, artifact_relocations=None):
        """Verify an explicit successful feed run for prepare, launch and execute."""
        from .feed_model import FeedModelConfig
        from .feed_training import FeedTrainingConfig
        from .artifact_paths import ArtifactResolver
        if not isinstance(source_job_id, str) or not source_job_id or not all(c.isalnum() or c in "_-" for c in source_job_id):
            raise ValueError("来源进给任务ID不合法")
        resolver = ArtifactResolver(artifact_relocations, base_dir=self.output_dir)
        source_output = source_feed_service_output or self.output_dir
        resolved_output = resolver.resolve(source_output)
        job = resolved_output / "jobs" / source_job_id
        state = json.loads((job / "state.json").read_text(encoding="utf-8"))
        source_request = json.loads((job / "request.json").read_text(encoding="utf-8"))
        result = state.get("result", {})
        if state.get("status") != "succeeded" or result.get("status") != "frozen" or source_request.get("mode") != "feed":
            raise ValueError("来源必须是已成功冻结的独立进给任务")
        run = resolved_output / "runs" / source_job_id
        receipt_path = run / "feed_reuse.json"
        if file_hash(receipt_path) != result.get("feed_reuse_sha256"):
            raise ValueError("来源任务缺少匹配的新进给复用回执；不能改写旧指纹")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt.get("schema") != "afc.feed_source.1" or receipt.get("source_feed_job_id") != source_job_id
                or receipt.get("development_sha256") != file_hash(run / "development.json")
                or receipt.get("run_identity_sha256") != file_hash(run / "run_fingerprint.json")):
            raise ValueError("来源进给任务的开发结果或完整运行来源被修改")
        original = json.loads((run / "run_fingerprint.json").read_text(encoding="utf-8"))
        if original["request"] != {k: v for k, v in source_request.items() if k not in ("job_id", "service_output", "resume_from_job_id")}:
            raise ValueError("来源进给任务请求与冻结运行不一致")
        feeds, _, snapshot = self.load_prepared(snapshot_id)
        self._validate_prepared(feeds, [], snapshot, mode="feed")
        if snapshot["synthetic"] != receipt["synthetic"] or receipt["source_snapshot_id"] != source_request["snapshot_id"]:
            raise ValueError("来源进给任务数据性质或快照引用不一致")
        development = json.loads((run / "development.json").read_text(encoding="utf-8"))
        chosen = development["selected"]
        selection = next(item for item in development["candidates"] if item["candidate"] == chosen)
        folds = protocol_folds()
        if len(selection["folds"]) != len(folds) or set(receipt["folds"]) != {f["fold"] for f in folds}:
            raise ValueError("来源九折成员不完整")
        source_training = FeedTrainingConfig(**chosen.get("feed_training", {}))
        source_model = FeedModelConfig(**chosen.get("feed_model", {}))
        for candidate in candidates or [{"id": "default"}]:
            if (FeedTrainingConfig(**candidate.get("feed_training", {})) != source_training
                    or FeedModelConfig(**candidate.get("feed_model", {})) != source_model):
                raise ValueError("负载候选的进给配置与来源任务不一致")
        epochs = []
        for fold, evidence in zip(folds, selection["folds"]):
            ref = receipt["folds"][fold["fold"]]
            if evidence["fold"] != fold["fold"] or evidence["validation_rounds"] != fold["validation_rounds"]:
                raise ValueError("来源九折映射不一致")
            model = _load_feed_reference(ref, [s for s in feeds if s.metadata["round_id"] in fold["train_rounds"]],
                [s for s in feeds if s.metadata["round_id"] in fold["validation_rounds"]], source_training, source_model,
                artifact_relocations=resolver.reference)
            if model.metadata["best_epoch"] != evidence["best_epoch"] or model.model_version != evidence["model_version"]:
                raise ValueError("来源折模型与九折结果不一致")
            epochs.append(model.metadata["best_epoch"])
        full_epochs = math.ceil(float(np.median(epochs)))
        if full_epochs != development["epochs"] or full_epochs != receipt["full_epochs"]:
            raise ValueError("来源全量epoch不等于九折最佳epoch中位数向上取整")
        _load_feed_reference(receipt["full"], feeds, None, replace(source_training, epochs=full_epochs), source_model,
                             artifact_relocations=resolver.reference)
        return {**receipt, "receipt_path": str(receipt_path), "receipt_sha256": file_hash(receipt_path),
                "source_feed_service_output": str(source_output),
                "resolved_source_feed_service_output": str(resolved_output),
                "artifact_relocations": resolver.reference, "target_snapshot_id": snapshot_id}

    def inspect(self, manifest, **kwargs):
        from .dataset_manifest import inspect_manifest
        if isinstance(manifest, str) and not manifest.strip():
            raise ValueError("请填写数据清单JSON文件路径或导入清单")
        if isinstance(manifest, (str, Path)) and not Path(manifest).is_file():
            raise ValueError("数据清单必须是存在的JSON文件")
        return inspect_manifest(manifest, **kwargs)

    def discover(self, directory, *, limit=1000):
        root = Path(directory).expanduser().resolve()
        if not root.is_dir() or type(limit) is not int or not 1 <= limit <= 5000:
            raise ValueError("输入目录不存在或扫描上限无效")
        extensions = {".nc", ".csv", ".json", ".txt", ".fxt"}
        found = []
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                found.append(dict(path=str(path), bytes=path.stat().st_size, status="unreviewed"))
                if len(found) >= limit:
                    break
        return dict(directory=str(root), files=found, truncated=len(found) == limit,
                    message="文件发现不等于数据资格通过；请导入已登记轮次、工况和审核信息的清单")

    def save_prepared(self, feed_samples, load_samples, *, source_snapshot, synthetic=False, frozen_feed_files=None):
        """Persist values, masks and metadata. Jobs never reread mutable original files."""
        snapshot_id = uuid.uuid4().hex
        directory = self.output_dir / "snapshots" / snapshot_id
        directory.mkdir(parents=True)
        source_snapshot = deepcopy(source_snapshot)
        # Evidence is part of the immutable preparation, not a mutable global pointer.
        evidence = source_snapshot.get("base_validation", {})
        for kind in ("model", "report"):
            path = evidence.get(kind + "_path")
            if path and Path(path).is_file():
                frozen = directory / "base_evidence" / (kind + Path(path).suffix)
                frozen.parent.mkdir(exist_ok=True)
                shutil.copyfile(path, frozen)
                if file_hash(frozen) != evidence.get(kind + "_sha256"):
                    raise ValueError("准备过程中基础功率凭据发生变化")
                evidence[kind + "_path"] = str(frozen.relative_to(directory))
        records = []
        for kind, samples in (("feed", feed_samples), ("load", load_samples)):
            for number, sample in enumerate(samples):
                name = f"{kind}_{number:04d}.csv"
                frozen_feed = (frozen_feed_files or {}).get(sample.metadata.get("record_id")) if kind == "feed" else None
                if frozen_feed is not None:
                    shutil.copyfile(frozen_feed["path"], directory / name)
                    if file_hash(directory / name) != frozen_feed["sha256"]:
                        raise ValueError("继承的进给CSV发生变化")
                else:
                    sample.frame.to_csv(directory / name, index=False, encoding="utf-8")
                metadata = dict(sample.metadata)
                if kind == "load":
                    from .calibration_evidence import measured_frame_digest
                    metadata["prepared_measurement_sha256"] = measured_frame_digest(sample.frame)
                records.append(dict(kind=kind, file=name, sha256=file_hash(directory / name), metadata=metadata,
                                    dtypes={k: str(v) for k, v in sample.frame.dtypes.items()}))
        return self._finalize_prepared_snapshot(directory, source_snapshot, records, synthetic=synthetic)

    def _finalize_prepared_snapshot(self, directory, source_snapshot, records, *, synthetic=False, validator=None):
        """Single snapshot identity/publication boundary, including streamed derivations."""
        directory = Path(directory)
        manifest = dict(schema=SERVICE_SCHEMA, synthetic=synthetic,
                        source_snapshot=source_snapshot, records=records, protocol=protocol(source_snapshot.get("route")),
                        created_at=datetime.now(timezone.utc).isoformat())
        digest = hashlib.sha256(json.dumps(json_value(manifest), ensure_ascii=False, sort_keys=True,
                                allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        manifest["snapshot_id"] = digest
        save_json(directory / "snapshot.json", manifest)
        if validator is not None:
            validator(manifest, directory)
        fixed_directory = directory.parent / digest
        directory.rename(fixed_directory)
        directory, snapshot_id = fixed_directory, digest
        return dict(snapshot_id=snapshot_id, synthetic=synthetic, records=len(records), directory=str(directory))

    def load_prepared(self, snapshot_id):
        if not isinstance(snapshot_id, str) or len(snapshot_id) != 64 or any(c not in "0123456789abcdef" for c in snapshot_id):
            raise ValueError("快照ID不合法")
        directory = self.output_dir / "snapshots" / snapshot_id
        manifest = json.loads((directory / "snapshot.json").read_text(encoding="utf-8"))
        contents = {k: v for k, v in manifest.items() if k != "snapshot_id"}
        digest = hashlib.sha256(json.dumps(contents, ensure_ascii=False, sort_keys=True,
                                allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        if manifest.get("snapshot_id") != snapshot_id or digest != snapshot_id:
            raise ValueError("准备结果快照元数据被修改")
        if manifest["schema"] != SERVICE_SCHEMA or manifest["protocol"]["version"] != PROTOCOL_VERSION:
            raise ValueError("准备结果协议不兼容")
        samples = {"feed": [], "load": []}
        for item in manifest["records"]:
            path = directory / item["file"]
            if file_hash(path) != item["sha256"]:
                raise ValueError("准备结果快照被修改")
            frame = pd.read_csv(path, float_precision="round_trip")
            for column, dtype in item["dtypes"].items():
                if dtype == "bool":
                    frame[column] = frame[column].astype(bool)
            cls = FeedSample if item["kind"] == "feed" else SampleSet
            samples[item["kind"]].append(cls(frame, item["metadata"]))
        return samples["feed"], samples["load"], manifest

    def refresh_main_stability(self, **request):
        """Versioned main-mask derivation; default is a nonexecuting preview."""
        from .main_stability_refresh import refresh_main_stability
        return refresh_main_stability(self, **request)

    def derive_support_fill(self, **request):
        """Explicit missing-input derivation; creates no training release or job."""
        from .support_fill_snapshot import derive_support_fill
        return derive_support_fill(self, **request)

    def refresh_load_route(self, *, parent_service_output, parent_snapshot_id, historical_consumer_source,
                           artifact_relocations=None, execute=False):
        """Derive a new feature contract from verified immutable R6 values."""
        from .artifact_paths import ArtifactResolver
        from .samples import load_input_view, path_features, path_feature_names
        from .training_checkpoint import source_fingerprints
        resolver = ArtifactResolver(artifact_relocations)
        parent_service = TrainingService(resolver.resolve(parent_service_output))
        if parent_service.output_dir.resolve() == self.output_dir.resolve():
            raise ValueError("新路线需要独立服务输出目录")
        historical = resolver.resolve(historical_consumer_source)
        parent_directory = parent_service.output_dir / "snapshots" / parent_snapshot_id
        parent_path = parent_directory / "snapshot.json"
        preview = dict(route=NOMINAL_ROUTE, parent_snapshot_id=parent_snapshot_id,
                       parent_snapshot_sha256=file_hash(parent_path), training_started=False)
        if not execute:
            return {**preview, "status": "prepared_not_executed"}
        feeds, loads, parent = parent_service.load_prepared(parent_snapshot_id)
        if parent["synthetic"] or not parent["source_snapshot"].get("main_stability_refresh"):
            raise ValueError("正式新路线须继承已审核R6主趋势派生快照")
        parent_service._validate_prepared(feeds, loads, parent, mode="all",
            historical_consumer_source=str(historical), artifact_relocations=artifact_relocations)
        checks = []
        for sample in loads:
            for mode in ("measured", "programmed"):
                view = load_input_view(sample, mode)
                selected = view.frame.valid_main if mode == "measured" else view.frame.valid_residual
                if selected.any():
                    from .samples import assert_feed_inputs
                    assert_feed_inputs(view.frame.loc[selected], mode)
                    path_features(view.frame.loc[selected], np.zeros(int(selected.sum())), 0., 1., 2000., route=NOMINAL_ROUTE)
            checks.append(dict(record_id=sample.metadata["record_id"], main_points=int(sample.frame.valid_main.sum()),
                               residual_points=int(sample.frame.valid_residual.sum())))
        directory = self.output_dir / "snapshots" / uuid.uuid4().hex
        directory.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(parent_directory, directory)
        source = deepcopy(parent["source_snapshot"])
        source["route"] = NOMINAL_ROUTE
        source["load_route_derivation"] = dict(schema="afc.load_route_derivation.1", route=NOMINAL_ROUTE,
            feature_version="afc.nominal_path_features.1", route_feature_names=path_feature_names(NOMINAL_ROUTE),
            parent_service_output=str(parent_service.output_dir.resolve()), parent_snapshot_id=parent_snapshot_id,
            parent_snapshot_sha256=file_hash(parent_path), historical_consumer_source=str(historical),
            artifact_relocations=artifact_relocations, consumer_source_sha256=source_fingerprints(),
            inherited_components="byte_identical_feed_load_csv_metadata_and_dtypes", records=checks)
        result = self._finalize_prepared_snapshot(directory, source, deepcopy(parent["records"]),
            validator=lambda manifest, location: self._verify_load_route_derivation(manifest, location, feeds=feeds, loads=loads))
        return {**result, "route": NOMINAL_ROUTE, "training_started": False, "raw_measurements_read": 0,
                "feed_network_used": False, "records_with_main": sum(r["main_points"] > 0 for r in checks),
                "records_with_residual": sum(r["residual_points"] > 0 for r in checks)}

    def _verify_load_route_derivation(self, snapshot, directory, *, feeds=None, loads=None,
                                      historical_route_source=None):
        from .artifact_paths import ArtifactResolver
        from .samples import path_feature_names
        from .training_checkpoint import source_fingerprints
        source = snapshot["source_snapshot"]
        link = source.get("load_route_derivation", {})
        resolver = ArtifactResolver(link.get("artifact_relocations"))
        if historical_route_source is None:
            route_sources = source_fingerprints()
        else:
            from .support_fill_snapshot import archived_route_fingerprints
            route_sources = archived_route_fingerprints(resolver.resolve(historical_route_source))
        if (link.get("schema") != "afc.load_route_derivation.1" or link.get("route") != NOMINAL_ROUTE
                or source.get("route") != NOMINAL_ROUTE or link.get("feature_version") != "afc.nominal_path_features.1"
                or link.get("route_feature_names") != path_feature_names(NOMINAL_ROUTE)
                or link.get("consumer_source_sha256") != route_sources):
            raise ValueError("新路线来源、18维特征或消费者源码身份不一致")
        parent_service = TrainingService(resolver.resolve(link["parent_service_output"]))
        parent_dir = parent_service.output_dir / "snapshots" / link["parent_snapshot_id"]
        if file_hash(parent_dir / "snapshot.json") != link["parent_snapshot_sha256"]:
            raise ValueError("新路线父快照身份改变")
        parent = json.loads((parent_dir / "snapshot.json").read_text(encoding="utf-8"))
        contents = {key: value for key, value in parent.items() if key != "snapshot_id"}
        computed = hashlib.sha256(json.dumps(contents, ensure_ascii=False, sort_keys=True,
                    allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        if parent.get("snapshot_id") != link["parent_snapshot_id"] or computed != link["parent_snapshot_id"]:
            raise ValueError("父快照内容身份不符")
        inherited_source = {key: value for key, value in source.items() if key not in ("route", "load_route_derivation")}
        if inherited_source != parent["source_snapshot"] or snapshot["records"] != parent["records"]:
            raise ValueError("新路线改变了父标签、元数据、来源或掩码登记")
        for item in parent["records"]:
            if (file_hash(Path(directory) / item["file"]) != item["sha256"]
                    or file_hash(parent_dir / item["file"]) != item["sha256"]):
                raise ValueError("新路线继承的数值文件改变")
        if feeds is None or loads is None:
            feeds, loads, _ = parent_service.load_prepared(link["parent_snapshot_id"])
        parent_service._validate_prepared(feeds, loads, parent, mode="all",
            historical_consumer_source=link["historical_consumer_source"], artifact_relocations=link.get("artifact_relocations"))
        expected = [dict(record_id=s.metadata["record_id"], main_points=int(s.frame.valid_main.sum()),
                         residual_points=int(s.frame.valid_residual.sum())) for s in loads]
        if link.get("records") != expected:
            raise ValueError("新路线监督覆盖记录不一致")
        return dict(verified=True, parent_snapshot_id=link["parent_snapshot_id"], components=len(parent["records"]))

    def prepare(self, manifest, *, base_model_path=None, base_validation, dataset_role="development",
                base_event_timelines=None, source_prepared_service_output=None, source_prepared_snapshot_id=None):
        source_pair = (source_prepared_service_output is not None, source_prepared_snapshot_id is not None)
        if any(source_pair):
            if not all(source_pair) or base_validation.get("mode") != "hybrid_estimated_labels":
                raise ValueError("增量准备必须同时指定父服务和父快照，且仅限混合估计协议")
            if dataset_role != "development" or base_event_timelines:
                raise ValueError("混合增量准备仅限开发集且不接受替代事件时间线")
            return self.prepare_hybrid_from_prepared(manifest, base_model_path=base_model_path,
                base_validation=base_validation, source_prepared_service_output=source_prepared_service_output,
                source_prepared_snapshot_id=source_prepared_snapshot_id)
        from .base_power import BasePowerModel
        from .feed import FeedPlan, MachineConfig
        from .feed_samples import prepare_feed_features, prepare_feed_samples
        from .measurement import read_measurement
        from .samples import prepare_load_samples, read_process_table
        from .experiment_protocol import DATASET_ROLES
        if dataset_role not in DATASET_ROLES:
            raise ValueError("未登记的数据用途")
        report = self.inspect(manifest, base_validation=base_validation, require_complete=dataset_role == "development", selected_role=dataset_role)
        if not report["feed_eligible"] or not report["load_eligible"]:
            raise ValueError(json.dumps(report["blockers"], ensure_ascii=False))
        from .base_timeline import AuditedEventTimeline
        # CLI and HTTP both send this record-keyed JSON to the same service.
        # Timeline values, not mutable file pointers, enter the frozen snapshot.
        supplied = deepcopy({} if base_event_timelines is None else base_event_timelines)
        if not isinstance(supplied, dict):
            raise ValueError("base_event_timelines 必须按 record_id 登记审核时间线")
        if any(not isinstance(value, (dict, AuditedEventTimeline)) for value in supplied.values()):
            raise ValueError("每份审核时间线必须为 AuditedEventTimeline 或同字段JSON对象")
        active_ids = {meta.get("record_id") for meta in report["records"]
                      if meta.get("status") == "active" and meta.get("dataset_role") == dataset_role}
        if set(supplied) - active_ids:
            raise ValueError("基础功率审核时间线包含非当前用途的记录ID")
        timelines = {key: value if isinstance(value, AuditedEventTimeline) else AuditedEventTimeline(**value)
                     for key, value in supplied.items()}
        # Resolved report paths come from the shared manifest verifier, including
        # any explicit runtime artifact relocation. Original evidence stays intact.
        base_validation = report["snapshot"]["base_validation"]
        record_idle = base_validation.get("mode") == "record_idle_labels"
        if record_idle:
            if base_model_path is not None or timelines:
                raise ValueError("本记录空载标签准备不接受动态模型或事件时间线")
            base = None
        else:
            base_model_path = base_model_path or base_validation.get("model_path")
            base = BasePowerModel.load(base_model_path)
            if file_hash(base_model_path) != base_validation.get("model_sha256"):
                raise ValueError("基础功率凭据未绑定当前模型文件")
        feeds, loads = [], []
        for meta in report["records"]:
            if meta.get("status") != "active" or meta.get("dataset_role") != dataset_role:
                continue
            files = meta["resolved_files"]
            machine = MachineConfig.load(files["machine"])
            features = prepare_feed_features(files["nc"], machine, program_id=meta["program_id"],
                feed_override=meta["feed_override"], machining_mode=meta["machining_mode"],
                machine_settings_version=meta["machine_settings_version"], coordinate_mode=meta.get("coordinate_mode", "machine"))
            _merge_feature_metadata(features, meta)
            plan = FeedPlan(features.frame, features.metadata["program_version"], features.metadata["machine_version"],
                            False, {"source": "nc_geometry", "time_mapping": "unavailable"})
            process = read_process_table(files["process"], plan, line_offset=meta.get("process_line_offset", 0),
                                         steady_lines=meta.get("steady_lines", []))
            record = read_measurement(files["measurement"], sample_period_s=meta.get("sample_period_s"))
            mapping = {int(v): int(v) + meta.get("measurement_line_offset", 0)
                       for v in np.unique(record.values["program_line_raw"]) if np.isfinite(v)}
            load = prepare_load_samples(process, plan, record, base, meta, line_map=mapping,
                                        event_timeline=timelines.get(meta.get("record_id")), base_qualification=base_validation)
            # Feed contact is a separate, weaker claim than unique ap/ae. Keep
            # unknown states nullable and preserve agreed contact at conflicts.
            from .sample_rules import apply_process_contact
            features.frame = apply_process_contact(features.frame, process)
            feed = prepare_feed_samples(features, record, meta, line_map=mapping,
                                         allow_position_derivative=meta.get("position_derivative_verified") is True)
            feeds.append(feed)
            loads.append(load)
        source_snapshot = deepcopy(report["snapshot"])
        if timelines:
            source_snapshot["base_event_timelines"] = {key: asdict(value) for key, value in timelines.items()}
        return self.save_prepared(feeds, loads, source_snapshot=source_snapshot)

    def prepare_hybrid_from_prepared(self, manifest, *, base_validation, source_prepared_service_output,
                                     source_prepared_snapshot_id, base_model_path=None):
        """Reuse frozen feed and unchanged label calculations; reread changed records only."""
        from .base_power import BasePowerModel
        from .base_qualification import _embedded
        from .feed import FeedPlan, MachineConfig
        from .feed_identity import feed_sample_identity
        from .feed_samples import prepare_feed_features
        from .hybrid_labels import (AFTER, MEASURED, MODE, STATUS, VERSION, label_metadata,
                                    verify_hybrid_evidence, verify_prepared_hybrid)
        from .measurement import read_measurement
        from .record_idle import verify_prepared_record_idle
        from .samples import prepare_load_samples, read_process_table

        if base_validation.get("mode") != MODE:
            raise ValueError("父快照增量准备仅支持混合估计协议")
        parent_service = TrainingService(source_prepared_service_output)
        if parent_service.output_dir.resolve() == self.output_dir.resolve():
            raise ValueError("混合协议必须写入新的服务根目录")
        feeds, old_loads, parent = parent_service.load_prepared(source_prepared_snapshot_id)
        if parent.get("synthetic"):
            raise ValueError("混合正式增量准备不接受合成父快照")
        # Full metadata and every parent CSV hash were checked by load_prepared.
        parent_service._validate_prepared(feeds, old_loads, parent, mode="feed")
        parent_source = parent["source_snapshot"]
        parent_evidence = parent_source.get("base_validation", {})
        if parent_evidence.get("mode") != "record_idle_labels":
            raise ValueError("混合增量准备需要原同记录平台父快照")
        parent_directory = parent_service.output_dir / "snapshots" / source_prepared_snapshot_id
        parent_report_path = Path(parent_evidence["report_path"])
        if not parent_report_path.is_absolute():
            parent_report_path = parent_directory / parent_report_path
        if file_hash(parent_report_path) != parent_evidence["report_sha256"]:
            raise ValueError("父平台报告SHA-256不一致")
        original_report = json.loads(parent_report_path.read_text(encoding="utf-8"))
        load_index = {s.metadata["record_id"]: s for s in old_loads}
        if len(load_index) != 450 or set(load_index) != {s.metadata["record_id"] for s in feeds}:
            raise ValueError("父快照必须完整保存450份进给和负载样本")
        # Inspect structure without hashing/parsing 450 mutable raw files again.
        # Deferred file/contact checks require exact parent binding below; the
        # new protocol is separately checked before any label is prepared.
        report = self.inspect(manifest, verify_files=False, base_validation=base_validation)
        current = [m for m in report["records"] if m.get("status") == "active"]
        if len(current) != 450 or {m["record_id"] for m in current} != set(load_index):
            raise ValueError("增量清单成员必须与父快照的450条开发记录一致")
        contact_receipts = _inherit_hybrid_parent_records(current, parent_source, feeds, old_loads)
        inherited_contact = {row["record_id"] for row in contact_receipts}
        blockers = [b for b in report["blockers"] if b["scope"] in ("common", "feed", "load")
                    and b["code"] not in ("files_not_verified", "base_not_validated")
                    and not (b["code"] == "contact_unknown" and b.get("record_id") in inherited_contact)]
        if blockers:
            raise ValueError(json.dumps(blockers, ensure_ascii=False))
        base_dir = Path(manifest).resolve().parent if isinstance(manifest, (str, Path)) else None
        checked = verify_hybrid_evidence(base_validation, current, base_dir=base_dir)
        if not checked["valid"]:
            raise ValueError("；".join(checked["issues"]))
        protocol_report = checked["report"]
        if set(protocol_report["records"]) != set(load_index):
            raise ValueError("混合协议分发表必须与450条父快照记录完全一致")
        if (protocol_report["parent_record_idle_report"]["sha256"] != parent_evidence["report_sha256"]
                or _embedded(protocol_report["parent_record_idle_report"], "父平台报告") != original_report):
            raise ValueError("新协议引用的父平台报告与父快照不同")
        evidence = deepcopy(base_validation)
        for kind, bound in checked["files"].items():
            evidence[kind + "_path"] = bound["path"]
        if base_model_path is not None and file_hash(base_model_path) != evidence["model_sha256"]:
            raise ValueError("指定基础模型与混合协议不一致")
        base = BasePowerModel.load(evidence["model_path"])
        parent_items = {(r["kind"], r["metadata"]["record_id"]): r for r in parent["records"]}
        frozen_feed_files = {s.metadata["record_id"]: dict(
            path=str(parent_directory / parent_items["feed", s.metadata["record_id"]]["file"]),
            sha256=parent_items["feed", s.metadata["record_id"]]["sha256"]) for s in feeds}
        parent_feed_identity = feed_sample_identity(feeds)
        loads, reuse_rows, changed = [], [], []
        for meta in current:
            rid = meta["record_id"]
            old, row = load_index[rid], protocol_report["records"][rid]
            verify_prepared_record_idle(old, parent_evidence)
            if old.metadata.get("file_sha256") != meta["file_sha256"]:
                raise ValueError(f"父样本四项来源与父清单不一致：{rid}")
            if (old.metadata["record_idle_label"]["status"] == "qualified"
                    and not (row["route"] == MEASURED and row["reference_application"] == AFTER)):
                raise ValueError(f"已合格父平台必须沿用原冻结计算：{rid}")
            if row["route"] == MEASURED and row["reference_application"] == AFTER:
                previous = old.metadata["record_idle_label"]
                if previous["status"] != "qualified":
                    raise ValueError(f"不能复用未合格的旧平台：{rid}")
                receipt = dict(parent_snapshot_id=source_prepared_snapshot_id,
                    parent_csv_sha256=parent_items["load", rid]["sha256"],
                    parent_report_sha256=parent_evidence["report_sha256"],
                    parent_mask_sha256=previous["mask_sha256"],
                    parent_source_sha256=original_report["production_source_sha256"],
                    computation="accepted_frozen_parent_values_and_masks_not_recomputed_under_new_source")
                label = label_metadata(row, evidence, valid_ranges=previous["valid_sample_ranges"],
                    sample_count=previous["raw_sample_count"], parent=receipt)
                frame = old.frame.copy()
                for key in ("label_source", "reference_kind", "evidence_grade", "model_version", "original_model_status"):
                    frame[key if key != "model_version" else "label_model_version"] = label.get(key)
                sample = SampleSet(frame, {**deepcopy(old.metadata), "hybrid_label": label,
                    "base_status": STATUS, "base_formal_training_release": False,
                    "base_authorized_estimation_training": True, "formal_metrological_qualification": False,
                    "base_qualification_report_sha256": evidence["report_sha256"],
                    "base_prediction_version": VERSION, "base_interface_version": VERSION,
                    "base_timing_source": "inherited_postoperative_record_idle_label"})
                pd.testing.assert_frame_equal(old.frame, frame[old.frame.columns], check_exact=True)
                reuse_rows.append(dict(record_id=rid, **receipt))
            else:
                files = meta["resolved_files"]
                for kind, path in files.items():
                    if file_hash(path) != meta["file_sha256"][kind]:
                        raise ValueError(f"重算记录原文件改变：{rid} {kind}")
                features = prepare_feed_features(files["nc"], MachineConfig.load(files["machine"]), program_id=meta["program_id"],
                    feed_override=meta["feed_override"], machining_mode=meta["machining_mode"],
                    machine_settings_version=meta["machine_settings_version"], coordinate_mode=meta.get("coordinate_mode", "machine"))
                _merge_feature_metadata(features, meta)
                plan = FeedPlan(features.frame, features.metadata["program_version"], features.metadata["machine_version"],
                                False, {"source": "nc_geometry", "time_mapping": "unavailable"})
                process = read_process_table(files["process"], plan, line_offset=meta.get("process_line_offset", 0),
                                             steady_lines=meta.get("steady_lines", []))
                measurement = read_measurement(files["measurement"], sample_period_s=meta.get("sample_period_s"))
                mapping = {int(v): int(v) + meta.get("measurement_line_offset", 0)
                           for v in np.unique(measurement.values["program_line_raw"]) if np.isfinite(v)}
                sample = prepare_load_samples(process, plan, measurement, base, meta, line_map=mapping,
                                               base_qualification=evidence)
                changed.append(rid)
            verify_prepared_hybrid(sample, evidence, row)
            loads.append(sample)
        if len(reuse_rows) != 427 or len(changed) != 23:
            raise ValueError("本阶段增量准备必须保留427条既有平台并只重算23条")
        source_snapshot = deepcopy(report["snapshot"])
        source_snapshot.update(records=current, base_validation=evidence, base_bound_files=checked["files"],
            files_verified=True, file_verification_basis="frozen_parent_source_and_csv_hashes_plus_changed_record_raw_hashes",
            incremental_preparation=dict(schema="afc.hybrid.incremental_preparation.1",
                parent_service_output=str(parent_service.output_dir.resolve()), parent_snapshot_id=source_prepared_snapshot_id,
                parent_snapshot_file_sha256=file_hash(parent_directory / "snapshot.json"),
                parent_source_sha256=original_report["production_source_sha256"],
                feed_sample_identity=parent_feed_identity, feed_csv_copy="byte_identical_all_records",
                inherited_contact=contact_receipts,
                unchanged_load_csv="round_trip_exact_original_columns_with_added_provenance_columns",
                unchanged_loads=reuse_rows, recomputed_loads=changed,
                raw_measurements_parsed=len(changed), feed_labels_recomputed=0,
                parent_csv_sha256=[dict(kind=r["kind"], record_id=r["metadata"]["record_id"], sha256=r["sha256"]) for r in parent["records"]]))
        result = self.save_prepared(feeds, loads, source_snapshot=source_snapshot, frozen_feed_files=frozen_feed_files)
        new_feeds, new_loads, snapshot = self.load_prepared(result["snapshot_id"])
        if feed_sample_identity(new_feeds) != parent_feed_identity:
            raise ValueError("增量准备改变了专用进给样本身份")
        reused_ids = {r["record_id"] for r in reuse_rows}
        for sample in new_loads:
            old = load_index[sample.metadata["record_id"]]
            if sample.metadata["record_id"] in reused_ids:
                pd.testing.assert_frame_equal(old.frame, sample.frame[old.frame.columns], check_exact=True)
        self._validate_prepared(new_feeds, new_loads, snapshot)
        result.update(feed_records_copied=len(new_feeds), unchanged_load_records=len(reuse_rows),
                      recomputed_load_records=len(changed), feed_identity_unchanged=True,
                      unchanged_load_original_columns_round_trip_equal=True)
        return result

    def prepare_feed(self, manifest):
        from .feed import MachineConfig
        from .feed_samples import prepare_feed_features, prepare_feed_samples
        from .measurement import read_measurement
        report = self.inspect(manifest)
        if not report["feed_eligible"]:
            raise ValueError(json.dumps([b for b in report["blockers"] if b["scope"] in ("feed", "common")], ensure_ascii=False))
        feeds = []
        for meta in report["records"]:
            if meta.get("status") != "active" or meta.get("dataset_role") != "development":
                continue
            files = meta["resolved_files"]
            features = prepare_feed_features(files["nc"], MachineConfig.load(files["machine"]), program_id=meta["program_id"],
                feed_override=meta["feed_override"], machining_mode=meta["machining_mode"],
                machine_settings_version=meta["machine_settings_version"], coordinate_mode=meta.get("coordinate_mode", "machine"))
            _merge_feature_metadata(features, meta)
            if files.get("process"):
                from .feed import FeedPlan
                from .samples import read_process_table
                from .sample_rules import apply_process_contact
                plan = FeedPlan(features.frame, features.metadata["program_version"], features.metadata["machine_version"], False, {})
                process = read_process_table(files["process"], plan, line_offset=meta.get("process_line_offset", 0))
                features.frame = apply_process_contact(features.frame, process)
            record = read_measurement(files["measurement"], sample_period_s=meta.get("sample_period_s"))
            mapping = {int(v): int(v) + meta.get("measurement_line_offset", 0)
                       for v in np.unique(record.values["program_line_raw"]) if np.isfinite(v)}
            feeds.append(prepare_feed_samples(features, record, meta, line_map=mapping,
                allow_position_derivative=meta.get("position_derivative_verified") is True))
        return self.save_prepared(feeds, [], source_snapshot=report["snapshot"])

    def preview(self, snapshot_id, candidates=None, mode="all"):
        feed, load, snapshot = self.load_prepared(snapshot_id)
        self._validate_prepared(feed, load, snapshot, mode=mode)
        if mode == "feed":
            return dict(folds=protocol_folds(), candidates=candidates or [{"id": "default"}],
                        phases=[dict(fold=f["fold"], stages=["feed"]) for f in protocol_folds()], final_stages=["feed"])
        if mode == "load":
            _nominal_candidates(candidates)
        return run_development(load, feed_samples=feed, candidates=candidates)

    def execute(self, request, *, cancel=None, progress=None):
        feeds, loads, snapshot = self.load_prepared(request["snapshot_id"])
        mode = request.get("mode", "all")
        self._validate_prepared(feeds, loads, snapshot, mode="all" if mode == "experiment" else mode)
        if mode == "load":
            candidates = _nominal_candidates(request.get("candidates"))
            if request.get("source_feed_job_id") or request.get("source_feed_service_output"):
                raise ValueError("名义负载任务不接受feed模型来源")
            if not snapshot["synthetic"] and any(c.get("load_training", {}).get("device") not in ("cpu", "cuda") for c in candidates):
                raise ValueError("正式load训练必须显式使用cpu或cuda")
        elif any(c.get("load_model", {}).get("route") == NOMINAL_ROUTE for c in request.get("candidates") or []):
            raise ValueError("名义负载路线必须使用mode=load")
        if request.get("source_feed_job_id") and mode not in ("all", "experiment"):
            raise ValueError("显式进给来源仅用于完整all任务或开发对照实验")
        feed_source = (self.validate_feed_source_job(request["source_feed_job_id"], request["snapshot_id"], request.get("candidates"),
            source_feed_service_output=request.get("source_feed_service_output"),
            artifact_relocations=request.get("artifact_relocations"))
            if request.get("source_feed_job_id") and mode == "all" else None)
        run, resume = self._prepare_run(request)
        if feed_source is not None:
            save_json(run / "feed_source_reference.json", feed_source)
        if mode == "experiment":
            result = self.experiment({**request, "execute": True}, cancel=cancel, progress=progress,
                                     output_dir=run / "experiment", resume=resume)
            save_json(run / "experiment.json", result)
            return result
        if request.get("mode", "all") == "feed":
            return self._execute_feed(request, feeds, snapshot, run=run, resume=resume, cancel=cancel, progress=progress)
        if any(c.get("load_model", {}).get("residual_output", "quantiles") != "quantiles" for c in request.get("candidates") or []):
            raise ValueError("正式训练必须输出三分位；单点版本请使用独立论文对照任务")
        if any(c.get("load_model", {}).get("condition_input", "descriptors") != "descriptors" or
               c.get("load_model", {}).get("coefficient_condition", "explicit") != "explicit" for c in request.get("candidates") or []):
            raise ValueError("正式模型使用物理描述和显式倍率；编号/隐式结构用于独立论文对照")
        if not snapshot["synthetic"]:
            from .experiment_protocol import validate_round_metadata
            validation = validate_round_metadata([s.metadata for s in loads], require_complete=True)
            if validation.get("issues") or validation.get("blockers"):
                raise ValueError(str(validation))
        development = run_development(loads, feed_samples=feeds, candidates=request.get("candidates"),
                                      execute=True, cancel=cancel, progress=progress, output_dir=run / "folds",
                                      resume=resume, feed_cache_dir=self.output_dir / "feed_cache",
                                      analysis_mode=request.get("analysis_mode", "audited_acquisition"),
                                      analysis_tool_group=request.get("analysis_tool_group"), feed_source=feed_source)
        save_json(run / "development.json", development)
        checkpoint(cancel)
        staging = run / "full_model"
        feed, bundle = train_full(loads, feeds, development, output_dir=staging, cancel=cancel, progress=progress,
                                  checkpoint_dir=run / "full_checkpoints", feed_cache_dir=self.output_dir / "feed_cache", resume=resume,
                                  feed_source=feed_source)
        checkpoint(cancel)
        model_id = uuid.uuid4().hex
        from .training_checkpoint import source_fingerprints
        metadata = dict(model_id=model_id, status="frozen", synthetic=snapshot["synthetic"],
                        snapshot_id=request["snapshot_id"], protocol=protocol(bundle.config.route), route=bundle.config.route,
                        interval_status="model_prediction_interval", full_epochs=development["full_epochs"],
                        dependencies={k: importlib.metadata.version(k) for k in ("numpy", "scipy", "pandas", "torch")},
                        code_fingerprints=source_fingerprints(),
                        files={p.name: file_hash(p) for p in staging.glob("*.pt")},
                        base_validation=snapshot["source_snapshot"].get("base_validation"),
                        source_feed_job_id=request.get("source_feed_job_id"),
                        feed_source_receipt_sha256=None if feed_source is None else feed_source["receipt_sha256"],
                        created_at=datetime.now(timezone.utc).isoformat())
        if snapshot["source_snapshot"].get("command_alignment_derivation"):
            metadata["input_contract"] = {key: loads[0].metadata[key] for key in
                                          ("position_policy", "process_mapping_rules_version")}
        save_json(staging / "model.json", metadata)
        if feed is not None:
            feed.assert_ready()
        bundle.assert_ready()
        destination = self.output_dir / ("synthetic_models" if snapshot["synthetic"] else "models") / model_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        checkpoint(cancel)
        shutil.move(str(staging), str(destination))
        return dict(status="frozen", model_id=model_id, synthetic=snapshot["synthetic"], directory=str(destination),
                    source_feed_job_id=request.get("source_feed_job_id"),
                    development=str(run / "development.json"), full_epochs=development["full_epochs"])

    def _run_identity(self, request):
        from .training_checkpoint import source_fingerprints
        from .feed_device import feed_execution_identity
        import platform
        if request.get("mode") == "load":
            from .load_device import request_load_execution
            return dict(schema="afc.training_run.2", request={k: v for k, v in request.items()
                        if k not in ("job_id", "service_output", "resume_from_job_id")}, code=source_fingerprints(),
                        environment=dict(python=platform.python_version(),
                        dependencies={k: importlib.metadata.version(k) for k in ("numpy", "scipy", "pandas", "torch")},
                        load_execution=request_load_execution(request)))
        devices = sorted({candidate.get("feed_training", {}).get("device", "cpu")
                          for candidate in request.get("candidates") or [{"feed_training": request.get("feed_training", {})}]})
        feed_execution = {device: feed_execution_identity(device) for device in devices}
        return dict(schema="afc.training_run.1", request={k: v for k, v in request.items()
                    if k not in ("job_id", "service_output", "resume_from_job_id")},
                    code=source_fingerprints(), environment=dict(python=platform.python_version(),
                    dependencies={k: importlib.metadata.version(k) for k in ("numpy", "scipy", "pandas", "torch")},
                    feed_execution=feed_execution))

    def validate_recovery(self, request):
        """Require a terminal source job and identical data/config/code before enqueue."""
        source = request.get("resume_from_job_id")
        if source is None:
            return None
        if not isinstance(source, str) or not source or not all(c.isalnum() or c in "_-" for c in source):
            raise ValueError("恢复来源任务ID不合法")
        state_path = self.output_dir / "jobs" / source / "state.json"
        if not state_path.is_file():
            raise ValueError("恢复来源任务不存在")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") not in ("failed", "cancelled", "interrupted"):
            raise ValueError("只有失败、取消或中断任务可以显式恢复")
        directory = self.output_dir / "runs" / source
        audit = directory / "run_fingerprint.json"
        if not audit.is_file() or json.loads(audit.read_text(encoding="utf-8")) != json_value(self._run_identity(request)):
            raise ValueError("恢复指纹不一致：快照、配置、源码或运行环境已变化")
        return directory

    def _prepare_run(self, request):
        job_id = request.get("job_id")
        if not isinstance(job_id, str) or not job_id or not all(c.isalnum() or c in "_-" for c in job_id):
            raise ValueError("任务ID不合法")
        source = self.validate_recovery(request)
        run = self.output_dir / "runs" / job_id
        if source is None:
            run.mkdir(parents=True, exist_ok=False)
        else:
            shutil.copytree(source, run)
            save_json(run / "recovery.json", dict(source_job_id=request["resume_from_job_id"],
                resumed_job_id=job_id, copied_at=datetime.now(timezone.utc).isoformat(), source_preserved=True))
        save_json(run / "run_fingerprint.json", self._run_identity(request))
        return run, source is not None

    def _execute_feed(self, request, feeds, snapshot, *, run, resume=False, cancel=None, progress=None):
        from .feed_model import FeedModelConfig
        from .feed_training import FeedTrainingConfig
        from .feed_metrics import aggregate_feed_metrics
        candidates, results = request.get("candidates") or [{"id": "default"}], []
        if len({c["id"] for c in candidates}) != len(candidates) or any(
                not isinstance(c["id"], str) or not c["id"] or Path(c["id"]).name != c["id"] or c["id"] in (".", "..") for c in candidates):
            raise ValueError("进给候选配置ID必须唯一且为安全目录名")
        for candidate in candidates:
            metrics, epochs, fold_results = [], [], []
            for fold in protocol_folds():
                train = [s for s in feeds if s.metadata["round_id"] in fold["train_rounds"]]
                val = [s for s in feeds if s.metadata["round_id"] in fold["validation_rounds"]]
                directory = run / "folds" / candidate["id"] / fold["fold"]
                model = train_feed_cached(train, val, training_config=FeedTrainingConfig(**candidate.get("feed_training", {})),
                    model_config=FeedModelConfig(**candidate.get("feed_model", {})), cancel=cancel,
                    cache_dir=self.output_dir / "feed_cache", checkpoint_path=directory / "checkpoints" / "feed.pt", resume=resume,
                    progress=(lambda x: progress(dict(candidate=candidate["id"], fold=fold["fold"], stage="feed", **x))) if progress else None)
                epochs.append(model.metadata["best_epoch"])
                directory.mkdir(parents=True, exist_ok=True)
                model.save(directory / "feed.pt")
                save_json(directory / "history.json", model.metadata)
                evaluation = evaluate_feed_records(model, val, output_dir=directory / "predictions", cancel=cancel)
                metrics.extend(evaluation["records"])
                detail = dict(fold=fold["fold"], validation_rounds=fold["validation_rounds"],
                              best_epoch=model.metadata["best_epoch"], model_version=model.model_version, evaluation=evaluation)
                save_json(directory / "validation.json", detail)
                fold_results.append(detail)
            results.append(dict(candidate=candidate, metrics=aggregate_feed_metrics(metrics), best_epochs=epochs, folds=fold_results))
        best = min(results, key=lambda r: (r["metrics"]["rmse_mm_min"], r["metrics"]["mae_mm_min"]))
        epochs = math.ceil(float(np.median(best["best_epochs"])))
        candidate = best["candidate"]
        development = dict(candidates=results, selected=candidate, epochs=epochs, units="mm/min",
                           protocol_version=PROTOCOL_VERSION, selection_order=["rmse_mm_min", "mae_mm_min"],
                           epoch_selection="ceil_median_nine_fold_best_epochs", independent_test_used=False)
        save_json(run / "development.json", development)
        model = train_feed_cached(feeds, training_config=replace(FeedTrainingConfig(**candidate.get("feed_training", {})), epochs=epochs),
            model_config=FeedModelConfig(**candidate.get("feed_model", {})), cancel=cancel, progress=progress,
            checkpoint_path=run / "full_checkpoints" / "feed.pt", cache_dir=None, resume=resume)
        staging = run / "feed_model"
        staging.mkdir(exist_ok=resume)
        checkpoint(cancel)
        model.save(staging / "feed.pt")
        save_json(staging / "history.json", model.metadata)
        save_json(staging / "model.json", dict(status="frozen", snapshot_id=request["snapshot_id"],
            mode="feed", synthetic=snapshot["synthetic"], model_version=model.model_version, full_epochs=epochs,
            file_sha256=file_hash(staging / "feed.pt"), protocol=protocol(),
            run_identity=self._run_identity(request)))
        checkpoint(cancel)
        destination = self.output_dir / ("synthetic_feed_models" if snapshot["synthetic"] else "feed_models") / uuid.uuid4().hex
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging), str(destination))
        from .feed_model import load_feed_model
        source_training = FeedTrainingConfig(**candidate.get("feed_training", {}))
        source_model = FeedModelConfig(**candidate.get("feed_model", {}))
        fold_references = {}
        for fold in protocol_folds():
            path = run / "folds" / candidate["id"] / fold["fold"] / "feed.pt"
            fold_model = load_feed_model(path)
            fold_references[fold["fold"]] = _feed_reference(path, fold_model,
                [s for s in feeds if s.metadata["round_id"] in fold["train_rounds"]],
                [s for s in feeds if s.metadata["round_id"] in fold["validation_rounds"]], source_training, source_model)
        receipt = dict(schema="afc.feed_source.1", source_feed_job_id=request["job_id"],
            source_snapshot_id=request["snapshot_id"], synthetic=snapshot["synthetic"],
            development_sha256=file_hash(run / "development.json"), run_identity_sha256=file_hash(run / "run_fingerprint.json"),
            full_epochs=epochs, folds=fold_references,
            full=_feed_reference(destination / "feed.pt", model, feeds, None,
                                 replace(source_training, epochs=epochs), source_model))
        receipt_path = run / "feed_reuse.json"
        save_json(receipt_path, receipt)
        return dict(status="frozen", mode="feed", directory=str(destination), epochs=epochs, synthetic=snapshot["synthetic"],
                    development=str(run / "development.json"), feed_reuse=str(receipt_path), feed_reuse_sha256=file_hash(receipt_path))

    def _validate_prepared(self, feeds, loads, snapshot, *, mode="all", historical_consumer_source=None,
                           artifact_relocations=None):
        if mode not in ("all", "feed", "load"):
            raise ValueError("任务mode必须为all、feed或load")
        if (mode != "load" and not feeds) or (mode != "feed" and not loads):
            raise ValueError("快照缺少任务所需监督样本")
        if snapshot["synthetic"]:
            return
        from .experiment_protocol import PROGRAM_FEATURES, validate_round_metadata
        from .sample_rules import SAMPLE_RULES_VERSION
        from ..process_path import PROCESS_MAPPING_RULES_VERSION
        source = snapshot["source_snapshot"]
        command_alignment = bool(source.get("command_alignment_derivation"))
        if command_alignment:
            if mode != "load":
                raise ValueError("指令位置重建快照仅用于独立名义负载训练")
            from .process_coordinates import PROCESS_COORDINATE_RULES_VERSION
            mapping_version = PROCESS_COORDINATE_RULES_VERSION
        else:
            mapping_version = PROCESS_MAPPING_RULES_VERSION
        if source.get("files_verified") is not True:
            raise ValueError("正式准备结果缺少原文件核验依据")
        for samples in ([feeds, loads] if mode == "all" else [loads] if mode == "load" else [feeds]):
            metadata = [s.metadata for s in samples]
            if not validate_round_metadata(metadata)["valid"]:
                raise ValueError("正式训练轮次和协议不一致")
            slots = [(m["round_id"], m["feature"]) for m in metadata]
            expected = {(r, f) for r in range(1, 91) for f in PROGRAM_FEATURES}
            if len(slots) != 450 or set(slots) != expected:
                raise ValueError("正式训练需要450个唯一当前有效程序位置")
            identities = [m.get("measurement_content_sha256") for m in metadata]
            if None in identities or len(set(identities)) != len(identities):
                raise ValueError("正式样本缺少内容指纹或存在复制实测")
            for m in metadata:
                if (m.get("sample_rules_version") != SAMPLE_RULES_VERSION
                        or m.get("process_mapping_rules_version") != mapping_version):
                    raise ValueError("样本工艺点定位规则已更新；旧累计s快照必须重新准备")
                if m.get("dataset_role") != "development" or any(m.get(k) is not True for k in
                        ("complete", "round_complete", "alignment_reliable", "no_alarm", "tool_healthy")):
                    raise ValueError("正式样本质量或用途尚未明确通过")
        if mode == "load":
            if command_alignment:
                from .command_alignment_snapshot import verify_command_alignment_snapshot
                verify_command_alignment_snapshot(self, snapshot, self.output_dir / "snapshots" / snapshot["snapshot_id"],
                                                  loads=loads)
                return
            if source.get("support_fill_derivation"):
                from .support_fill_snapshot import verify_support_fill_snapshot
                verify_support_fill_snapshot(self, snapshot, self.output_dir / "snapshots" / snapshot["snapshot_id"],
                                             require_release=True, loads=loads)
                return
            self._verify_load_route_derivation(snapshot, self.output_dir / "snapshots" / snapshot["snapshot_id"], feeds=feeds, loads=loads)
            return
        if mode == "all":
            from .main_stability import verify_stability_frame
            for sample in loads:
                verify_stability_frame(sample)
            hybrid = source.get("base_validation", {}).get("mode") == "hybrid_estimated_labels"
            if not hybrid and any(sample.metadata.get("base_formal_training_release") is False for sample in loads):
                raise ValueError("审核事件辅助接口尚待独立正式标签验证；工程覆盖率不能替代训练资格")
            from .dataset_manifest import verify_base_evidence
            check = verify_base_evidence(source.get("base_validation", {}), [s.metadata for s in loads],
                                          base_dir=self.output_dir / "snapshots" / snapshot["snapshot_id"],
                                          historical_consumer_source=historical_consumer_source,
                                          artifact_relocations=artifact_relocations)
            if not check["valid"]:
                raise ValueError("；".join(check["issues"]))
            if hybrid:
                if source.get("main_stability_refresh"):
                    from .main_stability_refresh import verify_derivation
                    verify_derivation(snapshot, loads,
                        directory=self.output_dir / "snapshots" / snapshot["snapshot_id"], consumer_report=check["report"],
                        historical_consumer_source=historical_consumer_source, artifact_relocations=artifact_relocations)
                else:
                    from .hybrid_labels import verify_prepared_hybrid
                    for sample in loads:
                        verify_prepared_hybrid(sample, source["base_validation"], check["report"]["records"][sample.metadata["record_id"]])
            elif source.get("base_validation", {}).get("mode") == "record_idle_labels":
                from .record_idle import verify_prepared_record_idle
                for sample in loads:
                    verify_prepared_record_idle(sample, source["base_validation"])

    def calibrate(self, request):
        from .calibration import CalibrationResult, IntervalCalibrator, calibrate_model
        from .calibration_evidence import load_calibration_sample, checked_interval_record, assert_independent
        from .models import predict_cutting_power
        from .metrics import regression_metrics
        from .neural import prediction_policy_version
        _, bundle, registry = self.load_model(request["model_id"], allow_synthetic=request.get("allow_synthetic", False))
        synthetic = registry.get("synthetic") is True
        records = request.get("records", [])
        validation = request.get("validation_records", [])
        if not validation:
            raise ValueError("校准需要未参加拟合的适配验证记录")
        training = bundle.metadata.get("samples", [])
        directory = self.output_dir / "calibrations" / uuid.uuid4().hex
        if request.get("kind", "condition") == "interval":
            calibration = CalibrationResult.load(request["condition_calibration"]) if request.get("condition_calibration") else None
            if calibration is not None:
                calibration.validate_for(bundle, request.get("metadata", calibration.target_condition))
            if request.get("memory"):
                from .memory import PositionMemory
                memory = PositionMemory.load(request["memory"])
            else:
                memory = None
            prediction_version = prediction_policy_version(bundle, calibration, memory)
            fit_metadata = [checked_interval_record(r, prediction_version, synthetic=synthetic) for r in records]
            validation_metadata = [checked_interval_record(r, prediction_version, synthetic=synthetic) for r in validation]
            assert_independent([("区间拟合", fit_metadata), ("区间验证", validation_metadata)], training)
            excluded = [m.get("record_id") for m in bundle.metadata.get("samples", [])]
            excluded += [r.get("group_id") for r in validation]
            result = IntervalCalibrator.fit(records, model_version=prediction_version, excluded_group_ids=excluded)
            metrics = []
            for r in validation:
                if r.get("dataset_role") != "adaptation_validation":
                    raise ValueError("区间外部验证用途必须单独登记")
                if "median_w" not in r:
                    raise ValueError("区间验证必须保留原模型中位数")
                low, high = result.apply(r["lower_w"], r["upper_w"], model_version=prediction_version)
                metrics.append(regression_metrics(r["target_w"], r["median_w"],
                                                  lower=low, upper=high))
            report = dict(status="independently_calibrated_prediction_interval", validation=metrics,
                          version=result.version, prediction_version=prediction_version, group_ids=result.group_ids, directory=str(directory))
            result.save(directory / "interval.json")
        else:
            fitted = [load_calibration_sample(r, synthetic=synthetic) for r in records]
            checked = [load_calibration_sample(r, synthetic=synthetic) for r in validation]
            assert_independent([("工况拟合", [s.metadata for s in fitted]), ("适配验证", [s.metadata for s in checked])], training)
            if any(s.metadata.get("dataset_role") != "adaptation_validation" for s in checked):
                raise ValueError("适配验证不能复用拟合记录")
            try:
                result = calibrate_model(bundle, fitted)
            except ValueError as exc:
                report = dict(status="not_identifiable" if "辨识" in str(exc) else "rejected", reason=str(exc))
                save_json(directory / "report.json", report)
                return report
            metrics = []
            for s in checked:
                selected = s.frame.loc[s.frame.valid_residual].reset_index(drop=True)
                prediction = predict_cutting_power(bundle, selected, context=s.metadata, feed_mode="measured", calibration=result)
                metrics.append(regression_metrics(selected.power_cut_w, prediction["cutting_power_w"],
                                                   lower=prediction["q05_w"], upper=prediction["q95_w"]))
            report = dict(status="calibrated_with_independent_adaptation_validation", validation=metrics,
                          version=result.version, identifiability=result.report, directory=str(directory))
            result.save(directory / "condition.json")
        save_json(directory / "report.json", report)
        return report

    def experiments(self, snapshot_id=None):
        from .experiments import list_experiments
        context = {"supported_experiments": self._supported_experiments()}
        if snapshot_id:
            feed, load, snapshot = self.load_prepared(snapshot_id)
            self._validate_prepared(feed, load, snapshot)
            from .condition import fit_condition_spec
            spec = fit_condition_spec([s.metadata for s in load])
            context.update(development_data_ready=True,
                           multicondition_data_ready=any(b["active"] for b in spec["branches"].values()))
        return list_experiments(context)

    @staticmethod
    def _supported_experiments():
        return ["fixed_physical", "local_physical", "no_condition", "descriptor_condition", "explicit_multipliers",
                "id_condition", "implicit_condition",
                "no_residual_modulation", "residual_modulation", "no_memory", "memory", "point_output", "quantile_output",
                "programmed_feed", "measured_feed", "predicted_feed", "zero_shot", "four_parameter_calibration",
                "physical_refit", "output_finetune", "full_retrain"]

    def _target_snapshot(self, snapshot_id, role):
        feeds, loads, snapshot = self.load_prepared(snapshot_id)
        if not feeds or not loads or any(s.metadata.get("dataset_role") != role for s in loads + feeds):
            raise ValueError(f"目标快照必须有同网格进给与负载样本并登记为{role}")
        if {s.metadata["record_id"] for s in feeds} != {s.metadata["record_id"] for s in loads}:
            raise ValueError("目标进给与负载记录不匹配")
        if not snapshot["synthetic"]:
            if snapshot["source_snapshot"].get("files_verified") is not True:
                raise ValueError("目标快照缺少原始文件核验")
            if any(s.metadata.get(k) is not True for s in loads for k in
                   ("complete", "alignment_reliable", "no_alarm", "tool_healthy")):
                raise ValueError("目标记录质量未通过")
            from .dataset_manifest import verify_base_evidence
            check = verify_base_evidence(snapshot["source_snapshot"].get("base_validation", {}), [s.metadata for s in loads],
                                         base_dir=self.output_dir / "snapshots" / snapshot_id)
            if not check["valid"]:
                raise ValueError("；".join(check["issues"]))
        return feeds, loads, snapshot

    def experiment(self, request, *, cancel=None, progress=None, output_dir=None, resume=False):
        from .experiments import run_experiment
        from .condition import fit_condition_spec
        feeds, loads, snapshot = self.load_prepared(request["snapshot_id"])
        self._validate_prepared(feeds, loads, snapshot)
        spec = fit_condition_spec([s.metadata for s in loads])
        directory = Path(output_dir) if output_dir else self.output_dir / "experiments" / uuid.uuid4().hex
        feed_source = (self.validate_feed_source_job(request["source_feed_job_id"], request["snapshot_id"],
            [dict(feed_model=request.get("feed_model", {}), feed_training=request.get("feed_training", {}))],
            source_feed_service_output=request.get("source_feed_service_output"),
            artifact_relocations=request.get("artifact_relocations")) if request.get("source_feed_job_id") else None)
        if feed_source is not None and request.get("execute", False):
            save_json(directory / "feed_source_reference.json", feed_source)
        target = {}
        for key, role in (("target_adaptation_snapshot", "target_adaptation"),
                          ("adaptation_validation_snapshot", "adaptation_validation"),
                          ("independent_test_snapshot", "independent_test")):
            if request.get(key):
                target[key] = self._target_snapshot(request[key], role)
                if target[key][2]["synthetic"] != snapshot["synthetic"]:
                    raise ValueError("不能混用合成与真实实验快照")
        context = dict(snapshot_id=request["snapshot_id"], development_data_ready=True,
                       multicondition_data_ready=any(b["active"] for b in spec["branches"].values()),
                       supported_experiments=self._supported_experiments(), source_model_id=request.get("model_id"),
                       frozen_source_model_ready=bool(request.get("model_id")), model_config=request.get("model_config", {}),
                       training_config=request.get("training_config", {}),
                       independent_target_test_ready="independent_test_snapshot" in target,
                       target_adaptation_ready="target_adaptation_snapshot" in target and "adaptation_validation_snapshot" in target,
                       target_adaptation_snapshot=request.get("target_adaptation_snapshot"),
                       target_budget=request.get("target_budget"),
                       independent_test_snapshot=request.get("independent_test_snapshot"))
        def runner(item, cancel=None, progress=None):
            name, ablation = item["experiment"], item["ablation"]
            from .experiment_adapters import TARGET_EXPERIMENTS, run_target_experiment
            if name in TARGET_EXPERIMENTS:
                feed, bundle, _ = self.load_model(item["source_model_id"], allow_synthetic=snapshot["synthetic"])
                def target_loads(key):
                    return target[key][1] if key in target else []
                if name != "zero_shot":
                    budget = item.get("target_budget")
                    if not isinstance(budget, dict) or budget.get("records") != len(target_loads("target_adaptation_snapshot")):
                        raise ValueError("target_budget.records必须等于冻结适配快照的完整记录数")
                return run_target_experiment(name, feed_model=feed, bundle=bundle, source_samples=loads,
                    adaptation_samples=target_loads("target_adaptation_snapshot"),
                    validation_samples=target_loads("adaptation_validation_snapshot"),
                    test_samples=target_loads("independent_test_snapshot"),
                    feed_samples=[s for values in target.values() for s in values[0]],
                    training_config=item["training_config"], execute=True, synthetic=snapshot["synthetic"],
                    qualification=dict(source_data_verified=True, target_data_verified=True, base_labels_verified=True),
                    cancel=cancel, progress=progress, output_dir=directory / "target")
            if name in ("programmed_feed", "measured_feed", "predicted_feed"):
                feed, bundle, _ = self.load_model(item["source_model_id"], allow_synthetic=snapshot["synthetic"])
                mode = {"programmed_feed": "programmed", "measured_feed": "measured", "predicted_feed": "predicted"}[name]
                score = evaluate_records(bundle, loads, feed_samples=feeds, feed_model=feed, feed_mode=mode, memory=None,
                                         output_dir=directory / name, cancel=cancel)
                return dict(status="evaluated", applied_experiment=name, results=score, memory_updates=0,
                            interpretation="in_sample_diagnostic_frozen_load_model_feed_error_propagation",
                            development_evidence=False,
                            out_of_fold_alternative="run_development saves three feed inputs per held-out fold using that fold's frozen models")
            model = dict(item["model_config"])
            if ablation.get("main") == "fixed_coefficients":
                model["physical_mode"] = "fixed"
            if ablation.get("condition") == "none":
                model.update(use_condition_effects=False, use_residual_modulation=False)
            if "residual_modulation" in ablation:
                model["use_residual_modulation"] = ablation["residual_modulation"]
            if ablation.get("output") == "point":
                model["residual_output"] = "point"
            if ablation.get("condition") == "identity_only_diagnostic":
                model["condition_input"] = "identity_only_diagnostic"
            if ablation.get("multipliers") == "implicit":
                model["coefficient_condition"] = "implicit"
            candidate = dict(id=name, load_model=model, load_training=item["training_config"],
                             feed_model=request.get("feed_model", {}), feed_training=request.get("feed_training", {}))
            result = run_development(loads, feed_samples=feeds, candidates=[candidate], execute=True, cancel=cancel,
                                     progress=progress, main_only=ablation.get("residual") is False,
                                     output_dir=directory / "folds", resume=resume, feed_cache_dir=self.output_dir / "feed_cache",
                                     analysis_mode=request.get("analysis_mode", "audited_acquisition"),
                                     analysis_tool_group=request.get("analysis_tool_group"), feed_source=feed_source)
            key = "with_memory" if ablation.get("memory") is True else "without_memory"
            scores = [r for f in result["candidates"][0]["folds"] for r in f[key]["records"]]
            return dict(status="evaluated", applied_experiment=name, development=result,
                        comparison_metrics=balanced_metrics(scores), final_model_published=False,
                        synthetic=snapshot["synthetic"])
        result = run_experiment(request["id"], runner, context, execute=request.get("execute", False),
                                cancel=cancel, progress=progress)
        if request.get("execute", False):
            save_json(directory / "experiment.json", result)
        return result

    def models(self, *, include_synthetic=False):
        result = []
        for kind in (["models", "synthetic_models"] if include_synthetic else ["models"]):
            for path in (self.output_dir / kind).glob("*/model.json"):
                metadata = json.loads(path.read_text(encoding="utf-8"))
                result.append({**metadata, "directory": str(path.parent)})
        return result

    def load_model(self, model_id, *, allow_synthetic=False, device="cpu"):
        from .feed_model import load_feed_model
        from .models import load_models
        candidates = [m for m in self.models(include_synthetic=allow_synthetic) if m["model_id"] == model_id]
        if len(candidates) != 1:
            raise ValueError("正式模型不存在或仅为合成验收模型")
        metadata = candidates[0]
        if metadata.get("status") != "frozen" or metadata.get("protocol", {}).get("version") != PROTOCOL_VERSION:
            raise ValueError("模型冻结状态或实验协议不兼容")
        route = metadata.get("route", "predicted_feed_v1")
        expected = protocol(route)
        if (metadata.get("protocol", {}).get("feed_columns") != expected["feed_columns"]
                or metadata["protocol"].get("path_columns") != expected["path_columns"]):
            raise ValueError("模型输入特征定义不兼容")
        path = Path(metadata["directory"])
        if any(file_hash(path / k) != v for k, v in metadata["files"].items()):
            raise ValueError("模型文件与冻结指纹不一致")
        feed = None if route == NOMINAL_ROUTE else load_feed_model(path / "feed.pt")
        load = load_models(path / "load.pt", device=device)
        if load.config.route != route:
            raise ValueError("模型权重路线与发布身份不一致")
        if feed is not None:
            feed.assert_ready()
        load.assert_ready()
        return feed, load, metadata

    def predict(self, request):
        """NC/process/known conditions only; no measured file argument is accepted."""
        from .feed import FeedPlan, MachineConfig
        from .feed_samples import prepare_feed_features
        from .feed_model import predict_actual_feed
        from .models import predict_cutting_power
        from .sample_rules import continuous_ids
        from .samples import prepare_forward_features, prepare_predicted_load_features, read_process_table
        if any(k in request for k in ("measurement", "actual_power", "actual_feed", "vibration", "alarm")):
            raise ValueError("加工前预测不得传入当前实测通道")
        feed, bundle, metadata = self.load_model(request["model_id"], allow_synthetic=request.get("allow_synthetic", False))
        contract = metadata.get("input_contract", {})
        training_contracts = [row for row in getattr(bundle, "metadata", {}).get("samples", [])
                              if str(row.get("process_mapping_rules_version", "")).startswith(
                                  "afc.data_supported_process_coordinates.")]
        data_supported = (str(contract.get("process_mapping_rules_version", "")).startswith(
                              "afc.data_supported_process_coordinates.") or bool(training_contracts))
        known = dict(request["metadata"])
        prepared_geometry = None
        if data_supported:
            from .process_coordinates import PROCESS_COORDINATE_RULES_VERSION
            expected = dict(position_policy="command_only_v1",
                            process_mapping_rules_version=PROCESS_COORDINATE_RULES_VERSION)
            for declared in ([contract] if contract else []) + training_contracts:
                if any(declared.get(key) != value for key, value in expected.items()):
                    raise ValueError("模型训练输入位置或工艺坐标规则不兼容")
            if any(key in known and known[key] != value for key, value in expected.items()):
                raise ValueError("预测请求与模型冻结的输入位置规则不一致")
            known.update(expected)
            artifact = request.get("prepared_geometry")
            if not isinstance(artifact, dict) or not isinstance(artifact.get("metadata"), dict):
                raise ValueError("新坐标模型需要经过核验的prepared_geometry，不能退回旧工艺映射")
            reference = artifact.get("frame", {})
            path = Path(reference.get("path", ""))
            checksum = reference.get("sha256")
            if not checksum or not path.is_file() or file_hash(path) != checksum:
                raise ValueError("预备几何文件缺少有效SHA-256身份")
            frame = pd.read_csv(path, float_precision="round_trip")
            if file_hash(path) != checksum:
                raise ValueError("预备几何文件读取期间改变")
            geometry_meta = artifact["metadata"]
            if geometry_meta.get("process_mapping_rules_version") != expected["process_mapping_rules_version"]:
                raise ValueError("预备几何与模型的坐标规则版本不一致")
            hashes = geometry_meta.get("file_sha256", {})
            if any(not hashes.get(key) or hashes[key] != file_hash(request[key])
                   for key in ("nc", "process", "machine")):
                raise ValueError("预备几何未绑定当前NC、工艺及机床输入文件")
            prepared_geometry = SampleSet(frame, deepcopy(geometry_meta))
        machine = MachineConfig.load(request["machine"])
        features = prepare_feed_features(request["nc"], machine, program_id=known["program_id"],
            feed_override=known["feed_override"], machining_mode=known["machining_mode"],
            machine_settings_version=known["machine_settings_version"], coordinate_mode=known.get("coordinate_mode", "machine"))
        plan = FeedPlan(features.frame, features.metadata["program_version"], features.metadata["machine_version"], False, {})
        process = read_process_table(request["process"], plan, line_offset=known.get("process_line_offset", 0))
        from .sample_rules import apply_process_contact
        features.frame = apply_process_contact(features.frame, process)
        if bundle.config.route == NOMINAL_ROUTE:
            from .samples import _prepare_process_features, load_input_view
            known = {**known, "program_version": features.metadata["program_version"], "protocol_version": PROTOCOL_VERSION}
            geometry = (prepare_forward_features(process, plan, known, prepared_geometry=prepared_geometry)
                        if data_supported else _prepare_process_features(process, plan, known))
            sample = load_input_view(geometry, "programmed")
        else:
            prediction = predict_actual_feed(feed, features)
            known = {**known, "program_version": prediction.metadata["program_version"],
                     "feed_model_version": feed.model_version, "protocol_version": PROTOCOL_VERSION}
            sample = prepare_predicted_load_features(process, prediction, known, prepared_geometry=prepared_geometry)
        frame = sample.frame
        if bundle.execution_window_policy() == "prospective_v1":
            from .neural import prospective_residual_context
            selected, _, _ = prospective_residual_context(frame, bundle.config.window_points)
        else:
            valid = np.isfinite(frame[PHYSICS_COLUMNS]).all(axis=1).to_numpy()
            if bundle.config.route == NOMINAL_ROUTE:
                from .samples import NOMINAL_GEOMETRY_COLUMNS
                valid &= np.isfinite(frame[NOMINAL_GEOMETRY_COLUMNS].to_numpy(float)).all(axis=1)
            frame["sequence_id"] = continuous_ids(frame, valid, known_only=True)
            selected = frame.loc[valid].reset_index(drop=True)
        if selected.empty:
            raise ValueError("工艺映射没有可预测的位置")
        # First release supports constant commanded spindle speed cutting only.
        cutting = (selected.ap_mm > 0) & (selected.ae_mm > 0) & (selected.motion != 0)
        variable = selected.loc[cutting, "spindle_rpm"].nunique() > 1
        if variable:
            raise ValueError("含切削主轴变速，缺少已验证的加工前转速预测")
        memory = None
        if request.get("memory") and _memory_exclusion(known) is None:
            from .memory import PositionMemory
            memory = PositionMemory.load(request["memory"])
        calibration = interval = None
        if request.get("condition_calibration"):
            from .calibration import CalibrationResult
            calibration = CalibrationResult.load(request["condition_calibration"])
        if request.get("interval_calibration"):
            from .calibration import IntervalCalibrator
            interval = IntervalCalibrator.load(request["interval_calibration"])
        result = predict_cutting_power(bundle, selected, memory=memory, context=known, round_id=known.get("round_id"),
                                       calibration=calibration, interval_calibrator=interval)
        from .total_power import add_total_power
        result = add_total_power(result, selected, request, known)
        prediction_id = uuid.uuid4().hex
        directory = self.output_dir / "predictions" / prediction_id
        directory.mkdir(parents=True)
        for key, values in result.items():
            if isinstance(values, np.ndarray) and values.shape == (len(selected),):
                selected[key] = values
        selected.to_csv(directory / "prediction.csv", index=False, encoding="utf-8")
        status = dict(prediction_id=prediction_id, model_id=request["model_id"], metadata=known,
                      interval_status=result.get("interval_status", "model_prediction_interval"), total_power_status=result["total_power_status"],
                      total_power_reason=result.get("total_power_reason"),
                      total_power_provenance=result.get("total_power_provenance"),
                      base_model_version=result.get("base_model_version"), base_prediction_mode=result.get("base_prediction_mode"),
                      total_uncertainty_scope=result.get("total_uncertainty_scope"),
                      memory_version=file_hash(request["memory"]) if memory is not None else "cold_start",
                      memory_status=_memory_exclusion(known) or ("enabled" if memory is not None else "cold_start"),
                      condition_calibration=request.get("condition_calibration"), interval_calibration=request.get("interval_calibration"),
                      prediction_version=result.get("prediction_version"),
                      main_memory_version=bundle.main_version if calibration is None else version_of([bundle.main_version, calibration.version]),
                      residual_memory_version=bundle.residual_version if calibration is None else version_of([bundle.residual_version, calibration.version]),
                      created_at=datetime.now(timezone.utc).isoformat(), frame_sha256=file_hash(directory / "prediction.csv"))
        save_json(directory / "prediction.json", status)
        return json_value({**status, "columns": selected.to_dict(orient="list"), "directory": directory})

    def evaluate(self, request):
        from .metrics import regression_metrics
        directory = self.output_dir / "predictions" / request["prediction_id"]
        saved = json.loads((directory / "prediction.json").read_text(encoding="utf-8"))
        if file_hash(directory / "prediction.csv") != saved["frame_sha256"]:
            raise ValueError("预测记录被修改")
        prediction = pd.read_csv(directory / "prediction.csv", float_precision="round_trip")
        actual = pd.read_csv(request["aligned_measurement"], float_precision="round_trip")
        keys = ["line", "execution_instance", "segment_mm"]
        if actual.duplicated(keys).any():
            raise ValueError("加工后实测位置不唯一")
        joined = prediction.merge(actual[keys + ["power_cut_w"]], on=keys, validate="one_to_one", how="left")
        _, bundle, _ = self.load_model(saved["model_id"], allow_synthetic=request.get("allow_synthetic", False))
        parameters = scoring_parameters(bundle, request.get("metric_parameters"))
        metrics = regression_metrics(joined.power_cut_w, joined.cutting_power_w,
            lower=joined.lower_w if "lower_w" in joined else joined.q05_w,
            upper=joined.upper_w if "upper_w" in joined else joined.q95_w,
            **parameters)
        result = dict(prediction_id=saved["prediction_id"], metrics=metrics, metric_parameters=parameters,
                      interval_status="evaluated_calibrated_prediction_interval" if saved.get("interval_calibration") else "evaluated_model_prediction_interval",
                      memory_update="not_requested")
        if request.get("update_memory"):
            from .memory import PositionMemory
            metadata = {**saved["metadata"], **{k: v for k, v in request.get("quality", {}).items() if k in
                        ("tool_healthy", "no_alarm", "complete", "alignment_reliable", "actual_event_order")},
                        "main_version": saved.get("main_memory_version", bundle.main_version),
                        "residual_version": saved.get("residual_memory_version", bundle.residual_version)}
            excluded = _memory_exclusion(metadata)
            if excluded:
                result["memory_update"] = {"status": excluded, "updated": 0}
            else:
                memory = PositionMemory.load(request["memory"]) if request.get("memory") else PositionMemory()
                result["memory_update"] = memory.update_round(metadata, joined,
                    joined.power_cut_w.to_numpy() - joined.main_w.to_numpy() - joined.residual_w.to_numpy(),
                    metadata["round_id"], healthy=metadata.get("tool_healthy"), no_alarm=metadata.get("no_alarm"),
                    complete=metadata.get("complete"), aligned=metadata.get("alignment_reliable"),
                    event_order=metadata.get("actual_event_order"))
                destination = directory / ("memory_" + uuid.uuid4().hex + ".json")
                memory.save(destination)
                result["memory_file"] = str(destination)
        save_json(directory / "evaluation.json", result)
        return result
