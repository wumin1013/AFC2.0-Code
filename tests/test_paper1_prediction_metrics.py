from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import sys

import numpy as np
import pandas as pd
import pytest

spec = spec_from_file_location("paper1_prediction_metrics", Path(__file__).parents[1] /
                              "scripts/verification/paper1_prediction_metrics.py")
metrics = module_from_spec(spec)
sys.modules[spec.name] = metrics
spec.loader.exec_module(metrics)


def frames():
    x = np.arange(101) * .1
    y = 10. + 30. * np.exp(-((x - 5.) / .5) ** 2)
    frame = pd.DataFrame(dict(line=1, execution_instance=0, segment_mm=x, path_mm=x,
                             sequence_id=0, power_cut_w=y, valid_main=x < 3,
                             valid_residual=True, grid_index=np.arange(len(x)),
                             reference_support_segment_id=0))
    predicted = frame.copy()
    predicted["cutting_power_w"] = y.copy()
    predicted["scored"] = True
    predicted["prediction_available"] = True
    return frame, predicted


def protocol():
    return metrics.ShapeProtocol(.2, .1, 100, minimum_sustained_span_mm=.5)


def evaluate(frame, predicted, frozen=None):
    return metrics.evaluate_shapes(frame, predicted, frozen or protocol(), record_id="synthetic_training_record")


def test_shapes_perfect_predictions_have_zero_errors_and_one_peak():
    frame, predicted = frames()
    result = evaluate(frame, predicted)
    assert result["scored_points"] == 101
    assert result["dynamic_rmse_w"] == result["sustained_absolute_bias_w"] == 0
    assert result["reference_peaks"] == result["predicted_peaks"] == 1
    assert result["peak_misses"] == result["peak_false_positives"] == 0
    assert result["peak_amplitude_absolute_error_w"] == 0


def test_missing_peak_is_not_zero_amplitude_error():
    frame, predicted = frames()
    predicted["cutting_power_w"] = 10.
    result = evaluate(frame, predicted)
    assert result["peak_misses"] == 1
    assert result["peak_amplitude_absolute_error_w"] is None


@pytest.mark.parametrize("failure", ["scored", "available", "nan", "short", "grid"])
def test_fixed_domain_cannot_hide_bad_predictions(failure):
    frame, predicted = frames()
    if failure == "scored":
        predicted.loc[50, "scored"] = False
    elif failure == "available":
        predicted.loc[50, "prediction_available"] = False
    elif failure == "nan":
        predicted.loc[50, "cutting_power_w"] = np.nan
    elif failure == "short":
        predicted = predicted.iloc[:-1]
    else:
        predicted.loc[50, "segment_mm"] += .01
    with pytest.raises((ValueError, AssertionError)):
        evaluate(frame, predicted)


def test_gaps_and_reference_support_boundaries_are_not_bridged():
    frame, _ = frames()
    mask = np.ones(len(frame), bool)
    mask[50] = False
    frame["reference_support_segment_id"] = np.where(frame.index < 80, 0, 1)
    segments = list(metrics.continuous_segments(frame, mask))
    assert [(int(x[0]), int(x[-1])) for x in segments] == [(0, 49), (51, 79), (80, 100)]


def test_training_protocol_requires_actual_noise_and_uses_training_labels():
    frame, _ = frames()
    frame["valid_main"] = True
    frame["power_cut_w"] = np.random.default_rng(42).normal(0., 2., len(frame))
    sample = SimpleNamespace(frame=frame, metadata={"record_id": "train1"})
    frozen = metrics.fit_shape_protocol([sample], minimum_sustained_span_mm=.5, expected_record_ids=["train1"])
    assert frozen.noise_differences == 100
    assert frozen.noise_sigma_w > 0
    assert frozen.grid_spacing_mm == pytest.approx(.1)
    frame["power_cut_w"] = 1.
    with pytest.raises(metrics.ShapeProtocolNotReady, match="positive training fluctuation"):
        metrics.fit_shape_protocol([sample], minimum_sustained_span_mm=.5, expected_record_ids=["train1"])


def test_matching_is_one_to_one_and_reports_unmatched_peaks():
    ref = [dict(position_mm=1., width_mm=.4, amplitude_w=10.)]
    pred = [dict(position_mm=1.01, width_mm=.4, amplitude_w=9.),
            dict(position_mm=1.02, width_mm=.4, amplitude_w=11.)]
    matches, missing, extra = metrics.match_peaks(ref, pred, protocol())
    assert len(matches) == 1 and missing == 0 and extra == 1
    assert matches[0]["predicted"]["position_mm"] == 1.01


def test_adjacent_valid_rows_split_at_actual_support_and_original_grid_gaps():
    frame, _ = frames()
    frame["reference_support_segment_id"] = np.where(frame.index < 50, 0, 1)
    frame.loc[75:, "grid_index"] += 1
    segments = list(metrics.continuous_segments(frame, np.ones(len(frame), bool)))
    assert [(int(x[0]), int(x[-1])) for x in segments] == [(0, 49), (50, 74), (75, 100)]


@pytest.mark.parametrize("failure", ["missing_support", "old_field", "invalid_support", "malformed_support", "missing_grid", "nan_grid"])
def test_incomplete_shape_evidence_never_shrinks_point_error_domain(failure):
    frame, predicted = frames()
    if failure == "missing_support":
        frame = frame.drop(columns="reference_support_segment_id")
    elif failure == "old_field":
        frame = frame.rename(columns={"reference_support_segment_id": "reference_segment_id"})
    elif failure == "invalid_support":
        frame.loc[50, "reference_support_segment_id"] = -1
    elif failure == "malformed_support":
        frame["reference_support_segment_id"] = "not_a_support_id"
    elif failure == "missing_grid":
        frame = frame.drop(columns="grid_index")
    else:
        frame.loc[50, "grid_index"] = np.nan
    predicted["cutting_power_w"] += 2.
    result = evaluate(frame, predicted)
    assert result["shape_status"] == "not_ready"
    assert result["shape_not_ready_reasons"]
    assert result["scored_points"] == len(frame)
    assert result["rmse_w"] == result["mae_w"] == 2.
    assert result["peak_amplitude_absolute_error_w"] is None
    assert not metrics.compare_peak_results(result, result)["amplitude_gate_evaluable"]


def test_missing_protocol_or_unfrozen_spatial_span_is_not_ready_not_training_failure():
    frame, predicted = frames()
    for frozen in (None, metrics.ShapeProtocol(.2, .1, 100)):
        result = metrics.evaluate_shapes(frame, predicted, frozen, record_id="train1")
        assert result["scored_points"] == 101 and result["rmse_w"] == 0
        assert result["shape_status"] == "not_ready"


def test_training_fit_cannot_silently_drop_a_record_or_use_partial_support():
    frame, _ = frames()
    sample = SimpleNamespace(frame=frame, metadata={"record_id": "train1"})
    with pytest.raises(ValueError, match="complete frozen F1"):
        metrics.fit_shape_protocol([sample], minimum_sustained_span_mm=.5, expected_record_ids=["train1", "train2"])
    bad = SimpleNamespace(frame=frame.drop(columns="reference_support_segment_id"), metadata={"record_id": "train2"})
    with pytest.raises(metrics.ShapeProtocolNotReady, match="train2: missing_reference_support"):
        metrics.fit_shape_protocol([sample, bad], minimum_sustained_span_mm=.5, expected_record_ids=["train1", "train2"])


def test_training_fluctuation_fit_does_not_cross_actual_reference_support_boundary():
    frame, _ = frames()
    frame["valid_main"] = True
    frame["power_cut_w"] = np.random.default_rng(42).normal(0., 2., len(frame))
    frame["reference_support_segment_id"] = np.where(frame.index < 50, 0, 1)
    sample = SimpleNamespace(frame=frame, metadata={"record_id": "train1"})
    frozen = metrics.fit_shape_protocol([sample], minimum_sustained_span_mm=.5, expected_record_ids=["train1"])
    assert frozen.noise_differences == 99
    assert "not an independent-noise estimate" in frozen.to_dict()["noise_interpretation"]


def test_sustained_bias_requires_predeclared_spatial_span_not_just_three_points():
    frame, predicted = frames()
    frame["valid_main"] = False
    frame.loc[:2, "valid_main"] = True
    predicted["cutting_power_w"] -= 3.
    result = evaluate(frame, predicted)
    assert result["sustained_segments"] == []
    assert result["sustained_absolute_bias_w"] is None
    assert result["scored_points"] == 101
    small = metrics.ShapeProtocol(.2, .1, 100, minimum_sustained_span_mm=.1)
    result = evaluate(frame, predicted, small)
    assert result["sustained_segments"][0]["bias_w"] == -3.
    assert result["sustained_segments"][0]["span_mm"] == .2


def test_boundary_truncated_peak_is_explicitly_censored_without_width_claim():
    frame, predicted = frames()
    frame, predicted = frame.iloc[:54].copy(), predicted.iloc[:54].copy()
    result = evaluate(frame, predicted)
    assert result["scored_points"] == 54 and result["rmse_w"] == 0
    assert result["reference_peaks"] == 0
    censored = result["censored_reference_peaks"]
    assert len(censored) == 1 and censored[0]["original_grid_index"] == 50
    assert "right_prominence_base_touches_support_boundary" in censored[0]["censor_reasons"]
    assert censored[0]["width_mm"] is None
    comparison = metrics.compare_peak_results(result, result)
    assert comparison["status"] == comparison["amplitude_applicability"] == "not_applicable"
    assert not comparison["width_gate_evaluable"]
    assert not comparison["amplitude_gate_evaluable"]
    assert "censored_peaks_outside_credible_reference_scope" in comparison["diagnostic_flags"]
    assert comparison["censoring"]["baseline"]["censored_reference_peaks"] == censored


def test_endpoint_summit_is_visible_as_censored_and_kept_in_point_errors():
    frame, predicted = frames()
    frame, predicted = frame.iloc[50:].copy(), predicted.iloc[50:].copy()
    result = evaluate(frame, predicted)
    assert result["scored_points"] == 51 and result["rmse_w"] == 0
    assert result["censored_reference_peaks"][0]["original_grid_index"] == 50
    assert result["censored_reference_peaks"][0]["censor_reasons"] == ["summit_at_support_endpoint"]


def two_peak_results(*, miss_second=False, extra_peak=False):
    x = np.arange(301) * .1
    peak1, peak2 = np.exp(-((x - 10) / .5) ** 2), np.exp(-((x - 20) / .5) ** 2)
    y = 10 + 30 * peak1 + 25 * peak2
    frame = pd.DataFrame(dict(line=1, execution_instance=0, segment_mm=x, path_mm=x,
        power_cut_w=y, valid_main=False, valid_residual=True,
        grid_index=np.arange(len(x)), reference_support_segment_id=0))
    a, b = frame.copy(), frame.copy()
    a["cutting_power_w"] = y + 2 * peak1 + 2 * peak2
    b["cutting_power_w"] = 10 + 30 * peak1 if miss_second else y + peak1 + peak2
    if extra_peak:
        b["cutting_power_w"] += 10 * np.exp(-((x - 15) / .5) ** 2)
    for predicted in (a, b):
        predicted["scored"] = True
        predicted["prediction_available"] = True
    return evaluate(frame, a), evaluate(frame, b)


def test_missed_difficult_peak_cannot_pass_via_lower_conditional_amplitude_error():
    q, p = two_peak_results(miss_second=True)
    assert q["peak_misses"] == 0 and p["peak_misses"] == 1
    assert q["peak_amplitude_absolute_error_w"] == 2
    assert p["peak_amplitude_absolute_error_w"] == 0
    result = metrics.compare_peak_results(q, p)
    assert len(result["reference_peak_ids"]) == 2
    assert len(result["common_matched_reference_ids"]) == 1
    assert result["conditional_amplitude_comparison"]["candidate_w"] == 0
    assert result["full_reference_amplitude_comparison"] is None
    assert not result["amplitude_gate_evaluable"]
    assert result["status"] == "not_ready" and result["amplitude_applicability"] == "applicable"
    assert len(result["detection"]["candidate"]["missed_reference_peak_ids"]) == 1


def test_complete_reference_peak_comparison_is_paired_and_not_an_overall_success():
    q, p = two_peak_results()
    result = metrics.compare_peak_results(q, p)
    assert result["status"] == "ready" and result["amplitude_gate_evaluable"]
    assert result["full_reference_amplitude_comparison"] == {"baseline_w": 2., "candidate_w": 1.}
    assert len(result["paired_errors"]) == 2
    assert "passed" not in result


def test_extra_false_positive_is_reported_and_cannot_be_hidden_by_amplitude_mean():
    q, p = two_peak_results(extra_peak=True)
    result = metrics.compare_peak_results(q, p)
    assert result["detection"]["candidate"]["peak_false_positives"] == 1
    assert result["status"] == "ready" and result["amplitude_gate_evaluable"]
    assert result["full_reference_amplitude_comparison"] == {"baseline_w": 2., "candidate_w": 1.}
    assert "candidate_false_positives_increased" in result["diagnostic_flags"]
    assert result["reasons"] == []


def test_no_reference_peak_is_not_applicable_independently_of_false_positives():
    frame, predicted = frames()
    frame["power_cut_w"] = 10.
    predicted["power_cut_w"] = 10.
    baseline = predicted.copy()
    baseline["cutting_power_w"] = 10.
    q, p = evaluate(frame, baseline), evaluate(frame, predicted)
    assert q["scored_points"] == p["scored_points"] == len(frame)
    assert q["peak_amplitude_applicability"] == p["peak_amplitude_applicability"] == "not_applicable"
    result = metrics.compare_peak_results(q, p)
    assert result["status"] == result["amplitude_applicability"] == "not_applicable"
    assert result["full_reference_amplitude_comparison"] is None
    assert not result["amplitude_gate_evaluable"] and not result["width_gate_evaluable"]
    assert result["detection"]["candidate"]["peak_false_positives"] == 1
    assert "candidate_false_positives_increased" in result["diagnostic_flags"]


def test_censored_peaks_do_not_block_complete_interior_peak_comparison():
    x = np.arange(301) * .1
    peak1, peak2 = np.exp(-((x - 10) / .5) ** 2), np.exp(-((x - 20) / .5) ** 2)
    y = 10 + 30 * peak1 + 25 * peak2
    frame = pd.DataFrame(dict(line=1, execution_instance=0, segment_mm=x, path_mm=x,
        power_cut_w=y, valid_main=False, valid_residual=True,
        grid_index=np.arange(len(x)), reference_support_segment_id=np.where(x < 20.3, 0, 1)))
    predictions = []
    for error in (2., 1.):
        predicted = frame.copy()
        predicted["cutting_power_w"] = y + error * (peak1 + peak2)
        predicted["scored"] = predicted["prediction_available"] = True
        predictions.append(evaluate(frame, predicted))
    q, p = predictions
    assert q["scored_points"] == p["scored_points"] == 301
    assert q["reference_peaks"] == 1 and q["censored_reference_peaks"]
    assert q["peak_width_comparison_ready"] and p["peak_width_comparison_ready"]
    result = metrics.compare_peak_results(q, p)
    assert result["status"] == "ready" and result["amplitude_applicability"] == "applicable"
    assert result["amplitude_gate_evaluable"] and result["width_gate_evaluable"]
    assert result["full_reference_amplitude_comparison"] == {"baseline_w": 2., "candidate_w": 1.}
    assert result["amplitude_scope"] == "credible_reference_interior_peaks"
    assert "no width claim for censored peaks" in result["width_scope"]
    assert result["censoring"]["baseline"]["censored_reference_peaks"] == q["censored_reference_peaks"]
    assert all(row["width_mm"] is None for row in q["censored_reference_peaks"])


@pytest.mark.parametrize("field", ["record_id", "fixed_domain_sha256", "shape_protocol", "reference_peak_catalog"])
def test_comparison_rejects_different_records_domains_protocols_and_reference_sets(field):
    q, p = two_peak_results()
    p = deepcopy(p)
    if field == "shape_protocol":
        p[field]["prominence_sigma"] = 4
    elif field == "reference_peak_catalog":
        p[field] = p[field][:-1]
    else:
        p[field] = "different"
    with pytest.raises(ValueError):
        metrics.compare_peak_results(q, p)
