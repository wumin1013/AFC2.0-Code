"""Development shape diagnostics on a fixed, complete point-error domain.

The caller freezes the F1 training manifest and protocol before evaluating either
arm. Missing shape evidence never removes scored points or prevents RMSE-based
training. Cross-record/category/group aggregation belongs to the caller, which
must retain every expected record, including records without evaluable shapes.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import numpy as np


class ShapeProtocolNotReady(ValueError):
    """Supplementary shape gates are incomplete; point-error training may continue."""


@dataclass(frozen=True)
class ShapeProtocol:
    noise_sigma_w: float
    grid_spacing_mm: float
    noise_differences: int
    minimum_sustained_span_mm: float | None = None
    minimum_segment_points: int = 3
    minimum_peak_width_points: int = 3
    prominence_sigma: float = 3.
    minimum_match_grid_steps: int = 3
    reference_width_match_fraction: float = .5

    def validate(self):
        if (not np.isfinite(self.noise_sigma_w) or self.noise_sigma_w <= 0
                or not np.isfinite(self.grid_spacing_mm) or self.grid_spacing_mm <= 0
                or self.noise_differences < 30):
            raise ValueError("Positive training fluctuation scale and spacing are required")
        for value in (self.minimum_segment_points, self.minimum_peak_width_points,
                      self.minimum_match_grid_steps):
            if type(value) is not int or value < 1:
                raise ValueError("Shape protocol counts must be positive integers")
        if self.minimum_segment_points < 3:
            raise ValueError("Three points is the computational floor for sustained segments")
        if (not np.isfinite(self.prominence_sigma) or self.prominence_sigma <= 0
                or not np.isfinite(self.reference_width_match_fraction)
                or self.reference_width_match_fraction <= 0):
            raise ValueError("Shape protocol thresholds must be positive and finite")
        if self.minimum_sustained_span_mm is None:
            raise ShapeProtocolNotReady("Freeze minimum_sustained_span_mm before the pilot")
        if (not np.isfinite(self.minimum_sustained_span_mm)
                or self.minimum_sustained_span_mm <= 0):
            raise ValueError("The predeclared sustained spatial span must be positive")

    def to_dict(self):
        self.validate()
        return dict(schema="paper1.shape_protocol.3", **asdict(self),
            noise_definition="pooled training steady first-difference MAD / (0.6744897501960817 * sqrt(2))",
            noise_interpretation="steady high-frequency fluctuation scale; not an independent-noise estimate or confidence bound",
            sustained_definition="continuous valid_main AND fixed scored, with predeclared minimum spatial span; 3 points is only a computational floor",
            dynamic_definition="fixed scored AND NOT valid_main: nonsteady supervised domain, not a physical-event classification",
            amplitude_definition="absolute cutting power at the peak top, not prominence",
            width_definition="spatial width at half prominence; sample-count floor is not a physical duration",
            matching="maximum-cardinality one-to-one, then minimum distance normalized by reference width/local spacing",
            endpoints="boundary-touching bases and endpoint maxima are explicitly censored; no width claim from them",
            peak_comparison_scope="credible reference interior peaks; not all potential peaks; zero credible reference peaks is not_applicable",
            peak_comparison_readiness="all credible reference peaks must be matched by both arms; other censored peaks and false positives are reported separately",
            aggregation="within-record diagnostics only; caller retains all expected records for balanced aggregation",
            scientific_status="development_screening_protocol_not_an_independent_effect_claim")


def _support_reasons(frame, mask):
    """Require the actual attachment fields, without intersecting the score mask."""
    reasons = []
    for key in ("grid_index", "reference_support_segment_id"):
        if key not in frame:
            reasons.append(f"missing_{key}")
            continue
        try:
            values = frame[key].to_numpy(float)
        except (TypeError, ValueError):
            reasons.append(f"invalid_{key}")
            continue
        checked = values if key == "grid_index" else values[mask]
        if not (np.isfinite(checked) & (checked >= 0) & (checked == np.floor(checked))).all():
            reasons.append(f"invalid_{key}")
        if key == "grid_index" and np.any(np.diff(values) <= 0):
            reasons.append("grid_index_not_unique_in_original_order")
    if not np.isfinite(frame.path_mm.to_numpy(float)[mask]).all():
        reasons.append("nonfinite_scored_position")
    return reasons


def continuous_segments(frame, mask):
    """Split true support changes and original grid gaps, including adjacent valid rows."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(frame),):
        raise ValueError("Mask does not cover the complete original grid")
    reasons = _support_reasons(frame, mask)
    if reasons:
        raise ShapeProtocolNotReady("; ".join(reasons))
    position = frame.path_mm.to_numpy(float)
    breaks = np.ones(len(frame), dtype=bool)
    if len(frame) > 1:
        breaks[1:] = (~mask[:-1] | ~mask[1:] | ~np.isfinite(position[:-1])
                      | ~np.isfinite(position[1:]) | (np.diff(position) <= 0)
                      | (np.diff(frame.grid_index.to_numpy(float)) != 1))
        for key in ("line", "execution_instance", "nc_sequence_id", "sequence_id",
                    "training_sequence_id", "reference_support_segment_id"):
            if key in frame:
                values = frame[key].to_numpy()
                breaks[1:] |= values[1:] != values[:-1]
    starts = np.flatnonzero(mask & breaks)
    for start in starts:
        end = start + 1
        while end < len(frame) and mask[end] and not breaks[end]:
            end += 1
        yield np.arange(start, end)


def fit_shape_protocol(samples, *, minimum_sustained_span_mm, expected_record_ids):
    """Fit only the complete caller-bound F1 training set; never skip unsupported records.

    Catch ShapeProtocolNotReady in the pilot driver: leave supplementary gates
    incomplete while retaining scientific training and full-domain RMSE/MAE.
    The expected IDs must come from the frozen F1 training manifest, not samples
    filtered according to available support or evaluation outcomes.
    """
    samples = list(samples)
    expected = list(expected_record_ids)
    actual = [sample.metadata["record_id"] for sample in samples]
    if (not expected or len(set(expected)) != len(expected) or len(set(actual)) != len(actual)
            or set(actual) != set(expected)):
        raise ValueError("The complete frozen F1 training record set is required")
    failures = []
    for sample in samples:
        scored = sample.frame.valid_residual.to_numpy(bool)
        failures.extend(f"{sample.metadata['record_id']}: {reason}"
                        for reason in _support_reasons(sample.frame, scored))
    if failures:
        raise ShapeProtocolNotReady("; ".join(failures))
    differences, spacings = [], []
    for sample in samples:
        frame = sample.frame
        good = frame.valid_main.to_numpy(bool) & frame.valid_residual.to_numpy(bool)
        target = frame.power_cut_w.to_numpy(float)
        if not np.isfinite(target[good]).all():
            raise ValueError("Non-finite qualified training labels")
        for indices in continuous_segments(frame, good):
            if len(indices) >= 3:
                differences.append(np.diff(target[indices]))
                spacings.append(np.diff(frame.path_mm.to_numpy(float)[indices]))
    if not differences:
        raise ShapeProtocolNotReady("Insufficient continuous training labels")
    values = np.concatenate(differences)
    noise = float(np.median(np.abs(values - np.median(values))) / (.6744897501960817 * np.sqrt(2.)))
    if not np.isfinite(noise) or noise <= 0 or len(values) < 30:
        raise ShapeProtocolNotReady("Insufficient positive training fluctuation scale")
    protocol = ShapeProtocol(noise, float(np.median(np.concatenate(spacings))), len(values),
                             minimum_sustained_span_mm=minimum_sustained_span_mm)
    protocol.validate()
    return protocol


def _peaks(values, positions, protocol):
    """Return complete interior peaks and explicit conservative boundary censoring."""
    from scipy.signal import find_peaks
    if len(values) < 3:
        return [], []
    indices, properties = find_peaks(values, prominence=(None, None), width=(None, None))
    threshold = protocol.prominence_sigma * protocol.noise_sigma_w
    complete, censored = [], []
    grid = np.arange(len(values))
    local_spacing = float(np.median(np.diff(positions)))
    for n, index in enumerate(indices):
        left, right = int(properties["left_bases"][n]), int(properties["right_bases"][n])
        prominence = float(properties["prominences"][n])
        reasons = (["left_prominence_base_touches_support_boundary"] if left == 0 else [])
        reasons += (["right_prominence_base_touches_support_boundary"] if right == len(values) - 1 else [])
        possible_prominence = float(values[index] - min(values[left], values[right]))
        if prominence < threshold and (not reasons or possible_prominence < threshold):
            continue
        peak = dict(index=int(index), position_mm=float(positions[index]),
                    amplitude_w=float(values[index]), prominence_w=prominence,
                    local_grid_spacing_mm=local_spacing)
        if reasons:
            censored.append(dict(**peak, width_mm=None, censor_reasons=reasons))
        elif properties["widths"][n] >= protocol.minimum_peak_width_points:
            width = (np.interp(properties["right_ips"][n], grid, positions)
                     - np.interp(properties["left_ips"][n], grid, positions))
            complete.append(dict(**peak, width_mm=float(width)))
    # A summit on a support endpoint has no two-sided neighbourhood. Keep a
    # visible candidate if its observed drop reaches the frozen scale threshold.
    for endpoint, direction in ((0, 1), (len(values) - 1, -1)):
        remaining = values[::direction]
        different = np.flatnonzero(remaining != remaining[0])
        if (len(different) and remaining[0] > remaining[different[0]]
                and remaining[0] - remaining.min() >= threshold):
            censored.append(dict(index=endpoint, position_mm=float(positions[endpoint]),
                amplitude_w=float(values[endpoint]), prominence_w=None, width_mm=None,
                local_grid_spacing_mm=local_spacing, censor_reasons=["summit_at_support_endpoint"]))
    return complete, censored


def match_peaks(reference, predicted, protocol):
    """Return all matched errors and explicit misses/false positives."""
    from scipy.optimize import linear_sum_assignment
    if not reference or not predicted:
        return [], len(reference), len(predicted)
    distance = np.abs(np.array([x["position_mm"] for x in reference])[:, None]
                      - np.array([x["position_mm"] for x in predicted])[None, :])
    radius = np.array([max(protocol.minimum_match_grid_steps * x.get("local_grid_spacing_mm", protocol.grid_spacing_mm),
                          protocol.reference_width_match_fraction * x["width_mm"]) for x in reference])[:, None]
    allowed = distance <= radius
    penalty = 10. * (max(len(reference), len(predicted)) + 1)
    rows, cols = linear_sum_assignment(np.where(allowed, distance / radius, penalty))
    pairs = [(i, j) for i, j in zip(rows, cols) if allowed[i, j]]
    matches = [dict(reference=reference[i], predicted=predicted[j],
        amplitude_absolute_error_w=abs(predicted[j]["amplitude_w"] - reference[i]["amplitude_w"]),
        position_error_mm=predicted[j]["position_mm"] - reference[i]["position_mm"],
        width_error_mm=predicted[j]["width_mm"] - reference[i]["width_mm"]) for i, j in pairs]
    return matches, len(reference) - len(pairs), len(predicted) - len(pairs)


def evaluate_shapes(frame, prediction_frame, protocol, *, record_id, unavailable_reasons=()):
    import pandas as pd
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("A frozen record identity is required")
    if len(frame) != len(prediction_frame):
        raise ValueError("Prediction must retain the complete grid")
    identity_columns = ["line", "execution_instance", "segment_mm", "path_mm", "power_cut_w"]
    for column in identity_columns:
        pd.testing.assert_series_equal(frame[column].reset_index(drop=True),
            prediction_frame[column].reset_index(drop=True), check_names=False,
            check_dtype=False, check_exact=True)
    scored = frame.valid_residual.to_numpy(bool)
    if (not np.array_equal(scored, prediction_frame.scored.to_numpy(bool))
            or np.any(scored & ~prediction_frame.prediction_available.to_numpy(bool))):
        raise ValueError("The fixed score domain lost or changed points")
    target, prediction = frame.power_cut_w.to_numpy(float), prediction_frame.cutting_power_w.to_numpy(float)
    available = prediction_frame.prediction_available.to_numpy(bool)
    if not np.isfinite(target[scored]).all() or not np.isfinite(prediction[available]).all():
        raise ValueError("Non-finite target/prediction cannot be hidden by filtering")
    error = prediction - target
    steady = scored & frame.valid_main.to_numpy(bool)
    dynamic = scored & ~frame.valid_main.to_numpy(bool)
    domain_columns = identity_columns + ["valid_main", "valid_residual"]
    domain_columns += [key for key in ("grid_index", "reference_support_segment_id", "nc_sequence_id",
                      "sequence_id", "training_sequence_id") if key in frame]
    domain_hash = hashlib.sha256(pd.util.hash_pandas_object(frame[domain_columns], index=False).to_numpy().tobytes()).hexdigest()
    result = dict(record_id=record_id, fixed_domain_sha256=domain_hash,
        scored_points=int(scored.sum()), rmse_w=float(np.sqrt(np.mean(error[scored] ** 2))) if scored.any() else None,
        mae_w=float(np.mean(np.abs(error[scored]))) if scored.any() else None,
        dynamic_points=int(dynamic.sum()), dynamic_rmse_w=float(np.sqrt(np.mean(error[dynamic] ** 2))) if dynamic.any() else None,
        dynamic_definition="nonsteady supervised domain (fixed scored AND NOT valid_main)",
        shape_status="not_ready", shape_not_ready_reasons=[], shape_protocol=None,
        sustained_segments=[], sustained_absolute_bias_w=None, sustained_points=0,
        peak_matches=[], reference_peak_catalog=[], predicted_peak_catalog=[],
        censored_reference_peaks=[], censored_predicted_peaks=[],
        missed_reference_peak_ids=[], false_positive_peak_ids=[],
        reference_peaks=None, predicted_peaks=None, peak_misses=None, peak_false_positives=None,
        peak_amplitude_absolute_error_w=None, peak_position_absolute_error_mm=None,
        peak_width_absolute_error_mm=None, peak_width_comparison_ready=False,
        peak_amplitude_applicability="not_ready",
        peak_amplitude_scope="credible_reference_interior_peaks",
        peak_width_scope="matched_credible_reference_interior_peaks; no width claim for censored peaks",
        amplitude_definition="absolute cutting power at the peak top",
        aggregation_scope="within_record; caller must retain all expected records")
    reasons = list(unavailable_reasons) + _support_reasons(frame, scored)
    if protocol is None:
        reasons.append("training_shape_protocol_not_frozen")
    else:
        try:
            protocol.validate()
        except ShapeProtocolNotReady as exc:
            reasons.append(str(exc))
    if reasons:
        result["shape_not_ready_reasons"] = list(dict.fromkeys(reasons))
        return result
    result.update(shape_status="ready", shape_protocol=protocol.to_dict())
    sustained = []
    for indices in continuous_segments(frame, steady):
        span = float(frame.path_mm.iloc[indices[-1]] - frame.path_mm.iloc[indices[0]])
        if len(indices) >= protocol.minimum_segment_points and span >= protocol.minimum_sustained_span_mm:
            sustained.append(dict(start_index=int(indices[0]), end_index=int(indices[-1]), points=len(indices),
                                  bias_w=float(error[indices].mean()), span_mm=span))
    result.update(sustained_segments=sustained, sustained_points=sum(row["points"] for row in sustained),
        sustained_absolute_bias_w=float(np.mean([abs(row["bias_w"]) for row in sustained])) if sustained else None)
    for indices in continuous_segments(frame, scored):
        positions = frame.path_mm.to_numpy(float)[indices]
        refs, censored_refs = _peaks(target[indices], positions, protocol)
        preds, censored_preds = _peaks(prediction[indices], positions, protocol)
        for peaks, kind in ((refs, "reference"), (censored_refs, "reference"),
                            (preds, "predicted"), (censored_preds, "predicted")):
            for peak in peaks:
                original = int(indices[peak["index"]])
                peak.update(original_grid_index=int(frame.grid_index.iloc[original]),
                            peak_id=f"{record_id}:{kind}:g{int(frame.grid_index.iloc[original])}")
        paired, _, _ = match_peaks(refs, preds, protocol)
        result["peak_matches"].extend(paired)
        for key, peaks in (("reference_peak_catalog", refs), ("predicted_peak_catalog", preds),
                           ("censored_reference_peaks", censored_refs), ("censored_predicted_peaks", censored_preds)):
            result[key].extend(peaks)
    matches = result["peak_matches"]
    for catalog, matched_key, output in (("reference_peak_catalog", "reference", "missed_reference_peak_ids"),
                                        ("predicted_peak_catalog", "predicted", "false_positive_peak_ids")):
        matched_ids = {row[matched_key]["peak_id"] for row in matches}
        result[output] = [row["peak_id"] for row in result[catalog] if row["peak_id"] not in matched_ids]
    result.update(reference_peaks=len(result["reference_peak_catalog"]), predicted_peaks=len(result["predicted_peak_catalog"]),
        peak_misses=len(result["missed_reference_peak_ids"]), peak_false_positives=len(result["false_positive_peak_ids"]))
    for output, key in (("peak_amplitude_absolute_error_w", "amplitude_absolute_error_w"),
                        ("peak_position_absolute_error_mm", "position_error_mm"),
                        ("peak_width_absolute_error_mm", "width_error_mm")):
        result[output] = float(np.mean([abs(row[key]) for row in matches])) if matches else None
    result["peak_amplitude_applicability"] = "applicable" if result["reference_peak_catalog"] else "not_applicable"
    result["peak_width_comparison_ready"] = bool(matches) and not result["missed_reference_peak_ids"]
    return result


def compare_peak_results(baseline, candidate):
    """Pair the fixed credible-reference catalog; missed credible peaks block its gate.

    This returns comparison data, never an overall Q/P decision. The caller must
    apply the frozen thresholds and aggregate every expected record, without
    dropping any expected record. No-reference-peak records remain visible with
    status=not_applicable; their applicability never depends on either model.
    Additional censored peaks and false positives remain diagnostics, not extra
    amplitude thresholds. Width claims cover only complete matched peaks.
    """
    for key in ("record_id", "fixed_domain_sha256", "scored_points"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"Q/P fixed domain differs: {key}")
    result = dict(status="not_ready", reasons=[], diagnostic_flags=[], reference_peak_ids=[], common_matched_reference_ids=[],
                  paired_errors=[], full_reference_amplitude_comparison=None, conditional_amplitude_comparison=None,
                  amplitude_gate_evaluable=False, width_gate_evaluable=False,
                  amplitude_applicability="not_ready", amplitude_scope="credible_reference_interior_peaks",
                  width_scope="matched_credible_reference_interior_peaks; no width claim for censored peaks",
                  detection={name: {key: row[key] for key in ("reference_peaks", "predicted_peaks", "peak_misses",
                             "peak_false_positives", "missed_reference_peak_ids", "false_positive_peak_ids")}
                             for name, row in (("baseline", baseline), ("candidate", candidate))},
                  censoring={name: {key: row[key] for key in ("censored_reference_peaks", "censored_predicted_peaks")}
                             for name, row in (("baseline", baseline), ("candidate", candidate))})
    if baseline["shape_status"] != "ready" or candidate["shape_status"] != "ready":
        result["reasons"].append("shape_support_or_training_protocol_not_ready")
        return result
    for key in ("shape_protocol", "reference_peak_catalog", "censored_reference_peaks"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"Q/P must share the same frozen reference definition: {key}")
    reference_ids = [row["peak_id"] for row in baseline["reference_peak_catalog"]]
    result["reference_peak_ids"] = reference_ids
    result["amplitude_applicability"] = "applicable" if reference_ids else "not_applicable"
    if any(row[key] for row in (baseline, candidate) for key in ("censored_reference_peaks", "censored_predicted_peaks")):
        result["diagnostic_flags"].append("censored_peaks_outside_credible_reference_scope")
    if candidate["peak_false_positives"] > baseline["peak_false_positives"]:
        result["diagnostic_flags"].append("candidate_false_positives_increased")
    if not reference_ids:
        result.update(status="not_applicable", reasons=["no_credible_reference_interior_peaks"])
        return result
    a = {row["reference"]["peak_id"]: row for row in baseline["peak_matches"]}
    b = {row["reference"]["peak_id"]: row for row in candidate["peak_matches"]}
    common = [key for key in reference_ids if key in a and key in b]
    result["common_matched_reference_ids"] = common
    result["paired_errors"] = [dict(reference_peak_id=key,
        baseline={field: a[key][field] for field in ("amplitude_absolute_error_w", "position_error_mm", "width_error_mm")},
        candidate={field: b[key][field] for field in ("amplitude_absolute_error_w", "position_error_mm", "width_error_mm")})
        for key in common]
    if common:
        result["conditional_amplitude_comparison"] = dict(
            baseline_w=float(np.mean([a[key]["amplitude_absolute_error_w"] for key in common])),
            candidate_w=float(np.mean([b[key]["amplitude_absolute_error_w"] for key in common])),
            interpretation="conditional on peaks matched by both arms; cannot alone pass the peak gate")
    if len(common) != len(reference_ids):
        result["reasons"].append("not_all_fixed_reference_peaks_matched_by_both_arms")
    if not result["reasons"]:
        result.update(status="ready", amplitude_gate_evaluable=True, width_gate_evaluable=True,
                      full_reference_amplitude_comparison={key: value for key, value in
                          result["conditional_amplitude_comparison"].items() if key != "interpretation"})
    return result
