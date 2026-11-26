"""Small synthetic inference only: no training, optimization, or real records."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np
import pandas as pd
import pytest
import torch

from verify_unified_load_prediction import SMALL_CANDIDATE, synthetic_records
from spindle_power.load_prediction import support_prediction as module
from spindle_power.load_prediction.common import Cancelled, file_hash
from spindle_power.load_prediction.models import ModelConfig, build_load_models
from spindle_power.load_prediction.neural import state_version


@pytest.fixture
def case():
    torch.set_num_threads(1)
    torch.manual_seed(816)
    _, loads = synthetic_records((1,), points=12)
    parent = loads[0]
    frame = parent.frame
    frame["process_geometry_known"] = True
    frame["process_geometry_conflict_support"] = False
    frame["training_sequence_id"] = frame.sequence_id
    frame.loc[[0, 5, 11], ["ap_mm", "ae_mm", "power_cut_w"]] = np.nan
    frame.loc[[0, 5, 11], ["process_geometry_known", "cutting_contact", "valid_main", "valid_residual"]] = False
    candidate = deepcopy(parent)
    candidate.frame.loc[[0, 5], ["ap_mm", "ae_mm"]] = [1., 2.]
    candidate.frame.loc[[0, 5], ["process_geometry_known", "cutting_contact"]] = True
    candidate.frame["support_fill_restored"] = False
    candidate.frame.loc[[0, 5], "support_fill_restored"] = True
    candidate.metadata["support_fill"] = {"policy": "synthetic_source_cells"}
    config = dict(SMALL_CANDIDATE["load_model"], route=module.NOMINAL_ROUTE)
    config["window_points"] = 4
    bundle = build_load_models(ModelConfig(**config), device="cpu")
    # A deterministic random fixture is marked ready only for inference tests.
    # No optimizer or training entry point is called.
    bundle.main_trained = bundle.residual_trained = True
    bundle.main_version = bundle.residual_main_version = state_version(bundle.main)
    bundle.residual_version = state_version(bundle.residual)
    bundle.normalization = dict(main_mean_w=50., main_scale_w=20., residual_scale_w=10.)
    bundle.metadata["config"] = dict(residual_batch_windows=2, residual_batch_strategy="padded")
    return bundle, parent, candidate


def run(case, tmp_path, **options):
    return module.evaluate_support_fill_record(*case, output_dir=tmp_path / "audit",
        metric_parameters={"scale_w": 100., "high_threshold_w": 100., "low_threshold_w": 30.}, **options)


def test_four_real_passes_preserve_parent_context_and_report_candidate_change(case, tmp_path, monkeypatch):
    original = module.evaluate_records
    calls = []
    def tracked(*args, **kwargs):
        assert kwargs["memory"] is None and kwargs["update_memory"] is False
        calls.append(kwargs["output_dir"].name)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "evaluate_records", tracked)
    report = run(case, tmp_path)
    assert calls == ["baseline_1", "baseline_2", "parent_context", "candidate_context"]
    assert report["passed"] and report["parent_context_exact"]
    assert report["coverage"]["restored_prediction_points"] == 2
    assert report["coverage"]["remaining_missing_points"] == 1
    assert report["metrics"]["count"] == 9 and report["new_domain_metrics"]["count"] == 0
    assert report["component_checks"]["cutting_power_w"]["candidate_single_pass"]["changed_points"] > 0
    assert len(set(report["manifest_digests"][name] for name in calls[:3])) == 1
    assert report["manifest_digests"]["candidate_context"] != report["manifest_digests"]["parent_context"]
    output = tmp_path / "audit"
    result = pd.read_csv(output / report["prediction_file"], float_precision="round_trip")
    assert len(result) == 12 and result.prediction_available.sum() == 11
    assert result.loc[[0, 5], "prediction_context"].eq("candidate").all()
    assert result.loc[11, "prediction_context"] == "unavailable" and pd.isna(result.loc[11, "cutting_power_w"])
    assert file_hash(output / report["prediction_file"]) == report["prediction_sha256"]
    for artifact in report["artifacts"].values():
        assert file_hash(artifact["path"]) == artifact["sha256"]
    manifest = json.loads((output / "parent_context_manifest.json").read_text(encoding="utf-8"))
    indices = [index for row in manifest["windows"] for index in row["original_indices"]]
    assert sorted(indices) == [1, 2, 3, 4, 6, 7, 8, 9, 10]
    assert all(row["padded_width"] >= len(row["original_indices"]) for row in manifest["windows"])


@pytest.mark.parametrize("column,value", [("ap_mm", 3.), ("sequence_id", 7), ("training_sequence_id", 7),
    ("valid_residual", False), ("power_cut_w", 99.), ("feed_program_mm_min", 500.)])
def test_protected_old_values_fail_before_inference(case, tmp_path, monkeypatch, column, value):
    case[2].frame.loc[2, column] = value
    monkeypatch.setattr(module, "evaluate_records", lambda *args, **kwargs: pytest.fail("Unexpected inference"))
    with pytest.raises(module.SupportPredictionError, match="changed"):
        run(case, tmp_path)


def test_changed_metadata_or_new_boundary_is_rejected(case, tmp_path):
    case[2].metadata["tool_session"] = "other"
    with pytest.raises(module.SupportPredictionError, match="metadata"):
        run(case, tmp_path)
    case[2].metadata["tool_session"] = case[1].metadata["tool_session"]
    case[2].frame["nc_sequence_id"] = 0
    with pytest.raises(module.SupportPredictionError, match="Unreviewed"):
        run(case, tmp_path)


def test_missing_full_grid_point_is_rejected(case, tmp_path):
    case[2].frame = case[2].frame.iloc[:-1].copy()
    with pytest.raises(module.SupportPredictionError, match="grid length"):
        run(case, tmp_path)


def test_observed_fields_do_not_define_manifest(case, tmp_path):
    bundle, parent, candidate = case
    report = run(case, tmp_path)
    artifact = pd.read_csv(report["artifacts"]["baseline_1"]["path"], float_precision="round_trip")
    before = module.context_manifest(bundle, parent, artifact)
    changed = deepcopy(parent)
    changed.frame["power_cut_w"] = 9876.
    changed.frame["valid_residual"] = False
    changed.frame["training_sequence_id"] = np.arange(len(changed.frame))
    changed.frame["time_s"] = np.arange(len(changed.frame))[::-1]
    assert module.context_manifest(bundle, changed, artifact) == before


@pytest.mark.parametrize("change", ["nonfinite", "missing_record", "drop_prediction", "third_pass_drift"])
def test_failure_is_not_hidden_by_scoring_or_merge(case, tmp_path, monkeypatch, change):
    original = module.evaluate_records
    def altered(*args, **kwargs):
        report = original(*args, **kwargs)
        name = kwargs["output_dir"].name
        target = "parent_context" if change == "third_pass_drift" else "candidate_context"
        if name == target:
            if change == "missing_record":
                report["records"] = []
                return report
            row = report["records"][0]
            path = kwargs["output_dir"] / row["prediction_file"]
            data = pd.read_csv(path, float_precision="round_trip")
            if change == "nonfinite":
                data.loc[0, "cutting_power_w"] = np.nan  # New, unscored point must still fail.
            elif change == "drop_prediction":
                data.loc[0, "prediction_available"] = False
            else:
                data.loc[2, "residual_w"] += 1e-6  # Even tiny drift fails when repeated baseline is exact.
            data.to_csv(path, index=False, encoding="utf-8")
            row["prediction_sha256"] = file_hash(path)
        return report
    monkeypatch.setattr(module, "evaluate_records", altered)
    with pytest.raises(module.SupportPredictionError):
        run(case, tmp_path)
    report = json.loads((tmp_path / "audit/report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed" and not report["passed"]
    assert (tmp_path / "audit/baseline_1/record_0000.csv").exists()


def test_cancelled_and_existing_output_never_publish_pass(case, tmp_path):
    with pytest.raises(Cancelled):
        run(case, tmp_path, cancel=lambda: True)
    with pytest.raises(module.SupportPredictionError, match="overwrite"):
        run(case, tmp_path)


def test_no_new_points_and_no_scored_targets_remain_explicit(case, tmp_path):
    bundle, parent, _ = case
    parent.frame["valid_residual"] = False
    report = run((bundle, parent, deepcopy(parent)), tmp_path)
    assert report["passed"] and report["restored_prediction_points"] == 0
    assert report["old_domain_metrics"]["count"] == 0
    assert report["fixed_parent_scored"]["rmse_absolute_change_w"] is None


def test_point_output_keeps_the_same_coverage_without_interval_fields(case, tmp_path):
    bundle, parent, candidate = case
    bundle.config.residual_output = "point"
    report = run((bundle, parent, candidate), tmp_path)
    assert report["passed"] and report["restored_prediction_points"] == 2
    assert set(report["component_checks"]) == set(module.COMPONENTS)
    data = pd.read_csv(tmp_path / "audit" / report["prediction_file"])
    assert not any(key in data for key in module.QUANTILES)


def test_full_parent_input_domain_can_be_empty_without_dropping_new_points(case, tmp_path):
    bundle, parent, candidate = case
    parent.frame[["ap_mm", "ae_mm", "power_cut_w"]] = np.nan
    parent.frame[["process_geometry_known", "cutting_contact", "valid_main", "valid_residual"]] = False
    candidate = deepcopy(parent)
    candidate.frame.loc[[0, 5], ["ap_mm", "ae_mm"]] = [1., 2.]
    candidate.frame.loc[[0, 5], ["process_geometry_known", "cutting_contact"]] = True
    report = run((bundle, parent, candidate), tmp_path)
    assert report["passed"] and report["coverage"]["parent_predicted_points"] == 0
    assert report["coverage"]["merged_predicted_points"] == 2
