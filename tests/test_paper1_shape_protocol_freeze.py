from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

path = Path(__file__).parents[1] / "scripts/verification/freeze_paper1_shape_protocol.py"
spec = spec_from_file_location("freeze_paper1_shape_protocol", path)
freeze = module_from_spec(spec)
sys.modules[spec.name] = freeze
spec.loader.exec_module(freeze)


def frames():
    old = pd.DataFrame(dict(line=[1, 1, 1, 1], execution_instance=0,
        path_mm=np.arange(4) * .5, segment_mm=np.arange(4) * .5,
        power_cut_w=[10., 12., 11., 13.], valid_main=True, valid_residual=True,
        sequence_id=0, ap_mm=[1., 1., np.nan, 1.], ae_mm=2.,
        process_geometry_known=[True, True, False, True], cutting_contact=True))
    new = old.copy()
    new.loc[2, ["ap_mm", "process_geometry_known"]] = [1., True]
    new["support_fill_source"] = "audited_source"
    attachment = old[list(freeze.GRID_COLUMNS)].copy()
    attachment["grid_index"] = np.arange(4)
    attachment["reference_support_segment_id"] = [0, 0, 1, 1]
    attachment["reference_power_cut_w"] = old.power_cut_w.to_numpy()
    attachment["reference_aligned"] = True
    return new, old, attachment


def test_only_support_is_joined_not_reference_labels_or_fill_inputs():
    new, old, attachment = frames()
    result = freeze.join_support(new, old, attachment)
    assert result.power_cut_w.equals(old.power_cut_w)
    assert result.reference_support_segment_id.tolist() == [0, 0, 1, 1]
    assert "reference_power_cut_w" not in result
    assert "ap_mm" not in result


@pytest.mark.parametrize("change", ["label", "mask", "boundary", "drop", "duplicate", "order", "shift", "support",
                                   "reference_value", "reference_hidden", "reference_nan", "reference_absent"])
def test_bad_join_cannot_redefine_grid_or_scoring(change):
    new, old, attachment = frames()
    if change == "label":
        new.loc[1, "power_cut_w"] += .01
    elif change == "mask":
        new.loc[1, "valid_residual"] = False
    elif change == "boundary":
        new.loc[1, "sequence_id"] += 1
    elif change == "drop":
        attachment = attachment.iloc[:-1]
    elif change == "duplicate":
        attachment.loc[1, "grid_index"] = 0
    elif change == "order":
        attachment = attachment.iloc[::-1]
    elif change == "shift":
        attachment.loc[1, "path_mm"] += .01
    elif change == "support":
        attachment = attachment.drop(columns="reference_support_segment_id")
    elif change == "reference_value":
        attachment.loc[1, "reference_power_cut_w"] += 1
    elif change == "reference_hidden":
        attachment.loc[1, "reference_aligned"] = False
    elif change == "reference_nan":
        attachment.loc[1, "reference_power_cut_w"] = np.nan
    else:
        attachment = attachment.drop(columns="reference_power_cut_w")
    with pytest.raises(ValueError):
        freeze.join_support(new, old, attachment)


def manifest():
    parent = dict(path="parent.json", sha256="frozen")
    value = dict(schema="afc.reference_display_candidate_manifest.2", mode="development",
        preflight_only=False, full_attachment_generation_complete=True,
        development_snapshot=parent, expected_records=2, completed_records=2,
        expected_record_ids=["R1", "R2"], records=[dict(record_id="R1"), dict(record_id="R2")],
        O3_read=False, training_performed=False)
    return value, parent


def test_full_manifest_retains_both_roles():
    value, parent = manifest()
    assert set(freeze.attachment_index(value, ["R1", "R2"], parent)) == {"R1", "R2"}


@pytest.mark.parametrize("change", ["preflight", "missing", "duplicate", "foreign_parent", "event", "expected"])
def test_partial_or_different_manifest_is_not_release(change):
    value, parent = manifest()
    if change == "preflight":
        value["preflight_only"] = True
    elif change == "missing":
        value["records"].pop()
    elif change == "duplicate":
        value["records"][1] = deepcopy(value["records"][0])
    elif change == "foreign_parent":
        value["development_snapshot"] = dict(path="other.json", sha256="other")
    elif change == "event":
        value["O3_read"] = True
    else:
        value["expected_record_ids"] = ["R1", "R1"]
    with pytest.raises(ValueError):
        freeze.attachment_index(value, ["R1", "R2"], parent)


def test_file_reference_rejects_late_change(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text("first", encoding="utf-8")
    reference = freeze.ref(path)
    assert freeze.checked(dict(**reference, snapshot_id="optional")) == reference
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError):
        freeze.checked(reference)


def test_missing_shape_support_does_not_remove_any_scored_point():
    new, old, attachment = frames()
    attachment.loc[1, "reference_support_segment_id"] = -1
    result = freeze.join_support(new, old, attachment)
    assert result.valid_residual.equals(old.valid_residual)
    assert len(result) == len(old)
    assert result.reference_support_segment_id.tolist() == [0, -1, 1, 1]


def receipt():
    value, parent = manifest()
    value.update(generator_sha256="generator_hash", candidate_source_sha256={"alignment.py": "bound_source"})
    row = dict(schema="afc.reference_display_candidate.2", status="candidate_verified_not_formally_released",
        record_id="R1", snapshot=parent, mode="development", preflight_only=False,
        source_and_score_files_unchanged=True, training_performed=False, prediction_metrics_evaluated=False,
        base_validity_exact=True, label_metadata_exact_except_declared_lineage=True,
        verification=dict(historical_alignment_exact=True, finite_parent_reference_exact=True),
        generator_sha256=value["generator_sha256"], candidate_source_sha256=value["candidate_source_sha256"],
        attachment=dict(path="reference.csv", sha256="bound"), support_segments=dict(path="support.json", sha256="bound"))
    return row, value, parent


def test_generated_development_receipt_matches_generator_contract():
    row, value, parent = receipt()
    freeze.validate_receipt(row, "R1", parent, value)


@pytest.mark.parametrize("field", ["schema", "status", "preflight_only", "generator_sha256",
                                   "candidate_source_sha256", "support_segments", "label_metadata_exact_except_declared_lineage"])
def test_preflight_mixed_execution_or_missing_support_receipts_are_rejected(field):
    row, value, parent = receipt()
    row[field] = True if field == "preflight_only" else None
    with pytest.raises(ValueError):
        freeze.validate_receipt(row, "R1", parent, value)
