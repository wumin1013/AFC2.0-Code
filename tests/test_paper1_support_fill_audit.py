"""Synthetic audit contracts only; no model, optimizer, or real data is used."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("support_fill_audit", ROOT / "scripts/verification/audit_paper1_support_fill.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def case():
    old = pd.DataFrame(dict(line=[14] * 5, execution_instance=0, segment_mm=[0., .5, 1., 1.5, 2.],
        path_mm=[0., .5, 1., 1.5, 2.], ap_mm=[np.nan, np.nan, 1., 1.5, 2.],
        ae_mm=[np.nan, np.nan, 2., 2., 2.], process_geometry_known=[False, False, True, True, True],
        process_geometry_conflict_support=False, cutting_contact=[False, False, True, True, True],
        steady=[False, False, True, True, True], sequence_id=[0, 0, 1, 1, 1],
        training_sequence_id=[0, 1, 2, 2, 2], power_cut_w=[np.nan, np.nan, 10., 15., 20.],
        valid_main=[False, False, True, True, True], valid_residual=[False, False, True, True, True],
        feed_program_mm_min=2000., spindle_cmd_rpm=8000., diameter_mm=10., teeth=3))
    process = pd.DataFrame(dict(line=[14, 14], execution_instance=0, segment_start_mm=[0., 1.],
                               segment_mm=[1., 2.], ap_mm=[1., 2.], ae_mm=2.))
    new = old.copy(deep=True)
    new.loc[:1, ["ap_mm", "ae_mm"]] = [1., 2.]
    new.loc[:1, ["process_geometry_known", "cutting_contact"]] = True
    return old, new, process, {"diameter_mm": 10.}


def test_fill_preserves_old_interpolated_values_and_all_labels():
    old, new, process, meta = case()
    result = audit.audit_frames(old, new, process, meta)
    assert result["restored_points"] == 2
    assert result["old_scoring_domain_preserved"] is True
    assert new.ap_mm.iloc[3] == 1.5  # Must not replace the old interpolation by the cell value 2.


@pytest.mark.parametrize("column,value", [
    ("ap_mm", 2.25), ("ae_mm", 2.25), ("power_cut_w", 30.),
    ("valid_main", False), ("valid_residual", False), ("steady", False),
    ("sequence_id", 9), ("training_sequence_id", 9), ("feed_program_mm_min", 4000.),
])
def test_old_data_changes_fail(column, value):
    old, new, process, meta = case()
    new.loc[3, column] = value
    with pytest.raises(audit.AuditError):
        audit.audit_frames(old, new, process, meta)


def test_new_supervision_is_a_separate_intervention():
    old, new, process, meta = case()
    new.loc[0, ["power_cut_w", "valid_residual"]] = [12., True]
    with pytest.raises(audit.AuditError, match="Protected column"):
        audit.audit_frames(old, new, process, meta)


def test_missing_supported_gap_fails():
    old, _, process, meta = case()
    with pytest.raises(audit.AuditError, match="Missing or unsupported"):
        audit.audit_frames(old, old.copy(), process, meta)


def test_no_source_point_remains_missing_without_fabrication():
    old, new, process, meta = case()
    old.loc[0, "line"] = new.loc[0, "line"] = 99
    for column in audit.MUTABLE:
        new.loc[0, column] = old.loc[0, column]
    result = audit.audit_frames(old, new, process, meta)
    assert result["remaining_missing_reasons"] == {"no_process_row": 1}


def test_source_conflict_cannot_be_overridden():
    old, new, process, meta = case()
    old.loc[0, "process_geometry_conflict_support"] = True
    new.loc[0, "process_geometry_conflict_support"] = True
    with pytest.raises(audit.AuditError, match="Missing or unsupported"):
        audit.audit_frames(old, new, process, meta)


def test_overlapping_cells_fail_even_when_candidate_values_look_plausible():
    old, new, process, meta = case()
    process.loc[1, "segment_start_mm"] = .5
    with pytest.raises(audit.AuditError, match="overlapping"):
        audit.audit_frames(old, new, process, meta)


def test_gap_between_declared_cells_is_not_filled():
    old, new, process, meta = case()
    process.loc[0, "segment_start_mm"] = .25
    for column in audit.MUTABLE:
        new.loc[0, column] = old.loc[0, column]
    result = audit.audit_frames(old, new, process, meta)
    assert result["remaining_missing_reasons"] == {"outside_or_missing_source_interval": 1}


def test_micro_gap_and_source_tail_are_not_expanded_by_tolerance():
    old, _, process, _ = case()
    old = old.iloc[[0, 1, 2, 3, 4]].copy()
    old["segment_mm"] = old["path_mm"] = [0., 1., 1. + 2e-11, 1. + 5e-11, 2. + 1e-11]
    process.loc[1, "segment_start_mm"] = 1. + 5e-11
    supported, values, *_ = audit.source_support(old, process, 10.)
    assert supported.tolist() == [True, True, False, False, False]
    assert values[:2, 0].tolist() == [1., 1.]


def test_source_boundary_uses_left_cell_without_nearby_rounding():
    old, _, process, _ = case()
    old.loc[0, "segment_mm"] = 1.
    old.loc[1, "segment_mm"] = np.nextafter(1., 2.)
    supported, values, *_ = audit.source_support(old, process, 10.)
    assert supported[:2].all()
    assert values[:2, 0].tolist() == [1., 2.]


def test_display_reference_never_upgrades_score_mask():
    old, new, process, meta = case()
    new["reference_power_cut_w"] = [9., 9.5, 10., 15., 20.]
    new["reference_aligned"] = True
    result = audit.audit_frames(old, new, process, meta)
    assert result["old_residual_points"] == 3
    new.loc[2, "reference_power_cut_w"] = 11.
    with pytest.raises(audit.AuditError, match="existing observed target"):
        audit.audit_frames(old, new, process, meta)


def test_new_window_boundary_column_is_not_a_fill_only_change():
    old, new, process, meta = case()
    new["nc_sequence_id"] = 0
    with pytest.raises(audit.AuditError, match="Unreviewed new"):
        audit.audit_frames(old, new, process, meta)


def manifest():
    from spindle_power.load_prediction.experiment_protocol import PROGRAM_FEATURES, round_spec

    rows = []
    for number in range(1, 91):
        for feature in PROGRAM_FEATURES:
            rid = f"R{number:02d}_{feature}"
            rows.append(dict(kind="load", metadata={**round_spec(number), "feature": feature,
                "record_id": rid, "measurement_content_sha256": rid, "dataset_role": "development"}))
    return {"records": rows}


def test_all_450_unique_development_records_are_required():
    value = manifest()
    assert len(audit.development_records(value)) == 450
    with pytest.raises(audit.AuditError):
        audit.development_records({"records": value["records"][:-1]})


@pytest.mark.parametrize("field,value", [("record_id", "duplicate"),
    ("measurement_content_sha256", "duplicate"), ("dataset_role", "independent_test"), ("group_id", "O3")])
def test_duplicate_or_event_identity_is_rejected_before_csv_reads(field, value):
    data = manifest()
    data["records"][0]["metadata"][field] = value
    data["records"][1]["metadata"][field] = value
    with pytest.raises(audit.AuditError):
        audit.development_records(data)


def test_reference_grade_cannot_be_promoted():
    old = dict(record_id="test", hybrid_label=dict(reference_kind="frozen_model_estimate",
               formal_metrological_qualification=False, valid_sample_ranges=[[0, 10]]))
    new = deepcopy(old)
    new["hybrid_label"]["formal_metrological_qualification"] = True
    with pytest.raises(audit.AuditError, match="Frozen label evidence"):
        audit.check_metadata(old, new)
