"""Frozen-model compatibility inference for a source-supported input fill.

Old points are computed again in their complete parent context. New points use
the candidate context. This explicit two-domain policy does not assert that a
single candidate-context prediction preserves the old predictions. Source-cell
derivation and population membership are verified by the calling data audit.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import checkpoint, file_hash, json_value, save_json
from .metrics import regression_metrics
from .sample_rules import continuous_ids
from .samples import (NOMINAL_GEOMETRY_COLUMNS, NOMINAL_ROUTE, PHYSICS_COLUMNS,
                      SampleSet, load_input_view, path_features)
from .training_service import evaluate_records

POLICY = "support_fill_parent_context_v1"
KEYS = ["line", "execution_instance", "segment_mm", "path_mm"]
MUTABLE = {"ap_mm", "ae_mm", "process_geometry_known", "cutting_contact"}
EXTRA_COLUMNS = {"reference_power_cut_w", "reference_aligned", "support_fill_restored",
                 "support_fill_source_point_index", "support_fill_source_file_line",
                 "support_fill_interval_start_mm", "support_fill_interval_end_mm"}
COMPONENTS = ["main_w", "residual_w", "memory_w", "cutting_power_w"]
QUANTILES = ["q05_w", "q50_w", "q95_w"]
INPUTS = PHYSICS_COLUMNS + ["curvature_per_mm", "block_length_mm"] + NOMINAL_GEOMETRY_COLUMNS


class SupportPredictionError(ValueError):
    """A compatibility-inference gate failed; no release is authorized."""


def _require(condition, message):
    if not condition:
        raise SupportPredictionError(message)


def _digest(value):
    return hashlib.sha256(json.dumps(json_value(value), ensure_ascii=False, sort_keys=True,
                                    allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _array_digest(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((value.dtype.str, value.shape)).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _bool(frame, column):
    values = frame[column]
    _require(not values.isna().any() and values.isin([True, False]).all(),
             f"Invalid boolean column: {column}")
    return values.to_numpy(bool)


def prediction_view(sample):
    """Mirror the public nominal evaluator before inference, without labels."""
    frame = load_input_view(sample, "programmed").frame.reset_index(drop=True)
    known = np.isfinite(frame[INPUTS].to_numpy(float)).all(axis=1)
    frame["sequence_id"] = continuous_ids(frame, known, known_only=True)
    return frame, known, frame.loc[known].reset_index(drop=True)


def _validate_pair(parent, candidate):
    old, new = parent.frame.reset_index(drop=True), candidate.frame.reset_index(drop=True)
    _require(len(old) == len(new) and len(old) > 0, "Full NC grid length changed or is empty")
    _require(set(old) <= set(new), "Candidate dropped parent columns")
    _require(set(new) - set(old) <= EXTRA_COLUMNS, "Unreviewed candidate columns")
    _require(set(KEYS + ["valid_residual", "power_cut_w", "sequence_id"]) | MUTABLE <= set(old),
             "Parent input contract is incomplete")
    _require(not old.duplicated(KEYS).any() and not new.duplicated(KEYS).any(), "Duplicate NC grid keys")
    for key, value in parent.metadata.items():
        _require(key in candidate.metadata and _digest(value) == _digest(candidate.metadata[key]),
                 f"Parent metadata changed: {key}")
    for column in set(old) - MUTABLE:
        _require(old[column].equals(new[column]), f"Protected parent column changed: {column}")
    was_known, is_known = _bool(old, "process_geometry_known"), _bool(new, "process_geometry_known")
    for frame, mask in ((old, was_known), (new, is_known)):
        _require(np.array_equal(mask, np.isfinite(frame[["ap_mm", "ae_mm"]].to_numpy(float)).all(axis=1)),
                 "Geometry support flag disagrees with finite values")
    _require(not (was_known & ~is_known).any(), "Previously known geometry lost")
    restored = ~was_known & is_known
    for column in MUTABLE:
        _require(old.loc[~restored, column].equals(new.loc[~restored, column]),
                 f"Input changed outside restored geometry: {column}")
    if "support_fill_restored" in new:
        _require(np.array_equal(_bool(new, "support_fill_restored"), restored), "Restored receipt mask differs")
    _require(np.array_equal(_bool(new, "cutting_contact")[restored],
                           (new.loc[restored, ["ap_mm", "ae_mm"]].to_numpy(float) > 0).all(axis=1)),
             "Restored contact disagrees with geometry")
    old_view, old_known, _ = prediction_view(parent)
    new_view, new_known, _ = prediction_view(candidate)
    _require(not (old_known & ~new_known).any(), "Previously predictable input lost")
    _require(not (_bool(old_view, "valid_residual") & ~old_known).any(), "Parent score lacks known inputs")
    _require(not (_bool(new_view, "valid_residual") & ~new_known).any(), "Candidate score lacks known inputs")
    return restored, old_known, new_known


def context_manifest(bundle, sample, artifact):
    """Complete actual window/batch membership and the model input digests.

The main values come from this pass's computed artifact, not another pass.
Casting back to float32 reproduces the main array used by path_features.
"""
    from .neural import residual_windows, window_batches
    from .experiment_protocol import PROTOCOL_VERSION

    frame, known, selected = prediction_view(sample)
    original_indices = np.flatnonzero(known)
    context = {**sample.metadata, "protocol_version": PROTOCOL_VERSION,
               "feed_model_version": sample.metadata.get("feed_model_version")}
    condition = bundle.condition_values(context, len(selected)) if len(selected) else None
    norm = bundle.normalization
    features = (path_features(selected, artifact.loc[known, "main_w"].to_numpy(np.float32),
                             norm["main_mean_w"], norm["main_scale_w"], bundle.config.feed_reference_mm_min,
                             route=bundle.config.route).astype(np.float32) if len(selected)
                else np.empty((0, 18), np.float32))
    physics = selected[PHYSICS_COLUMNS].to_numpy(np.float32)
    windows = list(residual_windows(selected, bundle.config.window_points, known_boundaries=True))
    groups = window_batches(windows, bundle.execution_batch_windows(), length=len,
                            strategy=bundle.execution_batch_strategy())
    items, seen = [], []
    for batch_index, group in enumerate(groups):
        padded_width = max(map(len, group))
        for batch_slot, indices in enumerate(group):
            source = original_indices[indices]
            seen.extend(source.tolist())
            items.append(dict(original_indices=source.tolist(), batch_index=batch_index, batch_slot=batch_slot,
                batch_size=len(group), padded_width=padded_width,
                physics_sha256=_array_digest(physics[indices]),
                features_sha256=_array_digest(features[indices]),
                condition_sha256=None if condition is None else _array_digest(np.asarray(condition, np.float32)[indices])))
    _require(sorted(seen) == original_indices.tolist(), "Window manifest dropped or duplicated predictable points")
    return dict(schema="afc.support_fill_context.1", grid_points=len(frame),
                grid_sha256=_digest(frame[KEYS].to_dict("list")),
                selected_original_indices=original_indices.tolist(),
                model_version=bundle.current_version(), normalization_sha256=_digest(norm),
                window_points=bundle.config.window_points, batch_windows=bundle.execution_batch_windows(),
                batch_strategy=bundle.execution_batch_strategy(), windows=items)


def _read_artifact(path, report, sample, known, components):
    _require(len(report.get("records", [])) == 1, "Missing or extra evaluation record")
    row = report["records"][0]
    _require(row["record_id"] == sample.metadata["record_id"], "Evaluation record identity changed")
    file = path / row["prediction_file"]
    _require(file.resolve().is_relative_to(path.resolve()), "Prediction artifact escapes its directory")
    _require(file_hash(file) == row["prediction_sha256"], "Prediction artifact hash mismatch")
    frame = pd.read_csv(file, float_precision="round_trip")
    _require(len(frame) == len(sample.frame), "Prediction dropped NC grid points")
    _require(np.array_equal(frame[KEYS].to_numpy(), sample.frame[KEYS].to_numpy()), "Prediction grid changed")
    for name in ("known_input_valid", "prediction_available"):
        _require(np.array_equal(_bool(frame, name), known), f"Prediction coverage differs: {name}")
    valid = _bool(sample.frame, "valid_residual")
    for name in ("scored", "native_scored", "score_mask"):
        _require(np.array_equal(_bool(frame, name), known & valid), f"Scoring domain changed: {name}")
    _require(np.array_equal(_bool(frame, "label_eligible"), valid), "Label eligibility changed")
    _require(frame.power_cut_w.equals(sample.frame.power_cut_w.reset_index(drop=True)), "Prediction target changed")
    for name in components:
        _require(name in frame, f"Missing prediction component: {name}")
        _require(np.isfinite(frame.loc[known, name].to_numpy(float)).all(), f"Nonfinite predictable output: {name}")
        _require(frame.loc[~known, name].isna().all(), f"Output fabricated outside known domain: {name}")
    _require(np.all(frame.loc[known, "memory_w"].to_numpy(float) == 0), "Memory was used in support audit")
    return frame, dict(path=str(file), sha256=file_hash(file), evaluation_path=str(path / "evaluation.json"),
                       evaluation_sha256=file_hash(path / "evaluation.json"))


def _difference(left, right, mask):
    error = left[mask] - right[mask]
    return dict(points=int(mask.sum()), max_abs_w=float(np.abs(error).max()) if len(error) else 0.,
                rmse_w=float(np.sqrt(np.mean(error ** 2))) if len(error) else 0.,
                changed_points=int(np.count_nonzero(error)))


def _metrics(frame, mask, parameters, components):
    return regression_metrics(frame.power_cut_w.to_numpy(float)[mask], frame.cutting_power_w.to_numpy(float)[mask],
        lower=frame.q05_w.to_numpy(float)[mask] if "q05_w" in components else None,
        upper=frame.q95_w.to_numpy(float)[mask] if "q95_w" in components else None,
        allow_empty=True, **parameters)


def evaluate_support_fill_record(bundle, parent, candidate, *, output_dir, metric_parameters, cancel=None):
    """Run four independent evaluations and fail closed on any unmet gate."""
    import torch

    output = Path(output_dir).resolve()
    _require(not output.exists(), "Refusing to overwrite support prediction evidence")
    _require(bundle.config.route == NOMINAL_ROUTE, "Support compatibility requires the nominal route")
    restored, old_known, new_known = _validate_pair(parent, candidate)
    components = COMPONENTS + (QUANTILES if bundle.config.residual_output == "quantiles" else [])
    output.mkdir(parents=True)
    identity = bundle.current_version()
    report = dict(schema="afc.support_fill_prediction.1", policy=POLICY, status="running", passed=False,
        fixed_weight_prediction_gate_passed=False, record_id=parent.metadata["record_id"],
        interpretation="Parent points are recomputed in parent context; restored points use candidate context. "
                       "This is compatibility inference, not candidate single-pass invariance.",
        training_performed=False, memory_used=False, model_version=identity,
        execution=dict(device=str(bundle.device), cpu_threads=torch.get_num_threads(),
                       cpu_interop_threads=torch.get_num_interop_threads()), artifacts={}, manifest_digests={})
    try:
        outputs, manifests = {}, {}
        passes = (("baseline_1", parent, old_known), ("baseline_2", parent, old_known),
                  ("parent_context", parent, old_known), ("candidate_context", candidate, new_known))
        for name, sample, known in passes:
            checkpoint(cancel)
            path = output / name
            evaluation = evaluate_records(bundle, [SampleSet(sample.frame.copy(deep=True), deepcopy(sample.metadata))],
                feed_mode="programmed", memory=None, update_memory=False, output_dir=path,
                metric_parameters=metric_parameters, cancel=cancel)
            _require(bundle.current_version() == identity, "Frozen model changed during inference")
            artifact, receipt = _read_artifact(path, evaluation, sample, known, components)
            outputs[name] = artifact
            manifests[name] = context_manifest(bundle, sample, artifact)
            report["artifacts"][name] = receipt
            report["manifest_digests"][name] = _digest(manifests[name])
        parent_manifest = output / "parent_context_manifest.json"
        candidate_manifest = output / "candidate_context_manifest.json"
        save_json(parent_manifest, manifests["baseline_1"])
        save_json(candidate_manifest, manifests["candidate_context"])
        report["context_manifests"] = dict(parent=str(parent_manifest), candidate=str(candidate_manifest))
        report["context_manifest_sha256"] = dict(parent=file_hash(parent_manifest), candidate=file_hash(candidate_manifest))
        report["prediction_version"] = _digest(dict(policy=POLICY, model=identity,
            parent_context=report["manifest_digests"]["parent_context"],
            candidate_context=report["manifest_digests"]["candidate_context"]))
        _require(all(report["manifest_digests"][name] == report["manifest_digests"]["baseline_1"]
                     for name in ("baseline_2", "parent_context")), "Parent full window/batch/input manifest changed")
        first, repeat, kept, full = (outputs[name] for name, _, _ in passes)
        merged = full.copy(deep=True)
        merged.loc[old_known, components] = kept.loc[old_known, components].to_numpy()
        added = new_known & ~old_known
        merged["prediction_policy"] = POLICY
        merged["prediction_version"] = report["prediction_version"]
        merged["prediction_context"] = np.where(old_known, "parent", np.where(added, "candidate", "unavailable"))
        merged["support_fill_restored"] = restored
        old_scored = _bool(first, "scored")
        _require(np.array_equal(_bool(merged, "scored"), old_scored), "Merged score domain changed")
        details, checks = {}, []
        for name in components:
            a, b, c = (frame[name].to_numpy(float) for frame in (first, repeat, kept))
            tolerance = np.abs(a[old_known] - b[old_known])
            error = np.abs(a[old_known] - c[old_known])
            passed = bool(np.all(error <= tolerance))
            checks.append(passed)
            details[name] = dict(passed=passed, violations=int(np.sum(error > tolerance)),
                reproduction=_difference(a, b, old_known), compatibility=_difference(a, c, old_known),
                candidate_single_pass=_difference(a, full[name].to_numpy(float), old_known))
            _require(np.isfinite(merged.loc[new_known, name].to_numpy(float)).all(), "Merged predictable output is nonfinite")
        report["component_checks"] = details
        report["old_pointwise_within_reproduction"] = all(checks)
        report["parent_context_exact"] = all(item["compatibility"]["changed_points"] == 0 for item in details.values())
        report["old_prediction_difference"] = _difference(first.cutting_power_w.to_numpy(float),
                                                          merged.cutting_power_w.to_numpy(float), old_known)
        original_metrics = _metrics(first, old_scored, metric_parameters, components)
        merged_metrics = _metrics(merged, old_scored, metric_parameters, components)
        delta = (abs(original_metrics["rmse_w"] - merged_metrics["rmse_w"]) if old_scored.any() else None)
        report["fixed_parent_scored"] = dict(points=int(old_scored.sum()), baseline=original_metrics,
            merged=merged_metrics, rmse_absolute_change_w=delta, limit_w=1e-3, passed=delta is None or delta <= 1e-3)
        report.update(metrics=merged_metrics, old_domain_metrics=merged_metrics,
                      new_domain_metrics=_metrics(merged, added & old_scored, metric_parameters, components))
        finite_reference = np.isfinite(parent.frame.power_cut_w.to_numpy(float))
        display = (np.isfinite(candidate.frame.reference_power_cut_w.to_numpy(float)) & _bool(candidate.frame, "reference_aligned")
                   if {"reference_power_cut_w", "reference_aligned"} <= set(candidate.frame) else finite_reference)
        report["coverage"] = dict(nc_grid_points=len(merged), parent_predicted_points=int(old_known.sum()),
            candidate_predicted_points=int(new_known.sum()), merged_predicted_points=int(new_known.sum()),
            restored_geometry_points=int(restored.sum()), restored_prediction_points=int(added.sum()),
            remaining_missing_points=int((~new_known).sum()),
            restored_geometry_still_unpredictable_points=int((restored & ~new_known).sum()),
            parent_label_eligible_points=int(_bool(parent.frame, "valid_residual").sum()),
            fixed_parent_scored_points=int(old_scored.sum()), new_scored_points=int((added & old_scored).sum()),
            old_domain_reference_points=int((old_known & finite_reference).sum()),
            new_domain_reference_points=int((added & finite_reference).sum()),
            old_domain_display_reference_points=int((old_known & display).sum()),
            new_domain_display_reference_points=int((added & display).sum()))
        _, _, _ = _validate_pair(parent, candidate)
        _require(bundle.current_version() == identity, "Frozen model changed before publication")
        merged_path = output / "merged_predictions.csv"
        merged.to_csv(merged_path, index=False, encoding="utf-8")
        report["artifacts"]["merged"] = dict(path=str(merged_path), sha256=file_hash(merged_path))
        report.update(prediction_file=merged_path.name, prediction_sha256=file_hash(merged_path),
                      restored_prediction_points=int(added.sum()), remaining_missing_points=int((~new_known).sum()))
        _require(all(checks), "Old component differences exceed same-configuration pointwise reproduction error")
        _require(report["old_prediction_difference"]["rmse_w"] <= 1e-3, "Old-domain prediction RMSE changed")
        _require(report["fixed_parent_scored"]["passed"], "Fixed parent scoring-domain RMSE changed by more than 1e-3 W")
        report.update(status="passed", passed=True, fixed_weight_prediction_gate_passed=True,
                      parent_full_window_batch_manifest_preserved=True, all_predictable_points_retained=True)
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        save_json(output / "report.json", report)
        raise
    save_json(output / "report.json", report)
    return report
