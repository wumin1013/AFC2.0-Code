"""Small source-grounded fill tests; no model, training or real records."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spindle_power.load_prediction.common import file_hash
from spindle_power.load_prediction.samples import SampleSet, read_process_table
from spindle_power.load_prediction.support_fill import (
    ADDED_COLUMNS, MUTABLE_COLUMNS, fill_missing_process_support, fill_sample_from_process_source,
)


def example():
    parent = pd.DataFrame(dict(line=[14] * 5, execution_instance=0,
        segment_mm=[0., .5, 1., 1.5, 2.], path_mm=[0., .5, 1., 1.5, 2.],
        ap_mm=[np.nan, np.nan, 1., 1.5, 2.], ae_mm=[np.nan, np.nan, 2., 2., 2.],
        process_geometry_known=[False, False, True, True, True],
        process_geometry_conflict_support=False, cutting_contact=[False, False, True, True, True],
        steady=[False, False, True, True, True], sequence_id=[0, 0, 1, 1, 1],
        training_sequence_id=[0, 1, 2, 2, 2], nc_sequence_id=0,
        power_cut_w=[np.nan, np.nan, 10., 15., 20.],
        valid_main=[False, False, True, True, True], valid_residual=[False, False, True, True, True],
        feed_input_mm_min=[np.nan, np.nan, 1900., 2000., 2000.],
        feed_actual_mm_min=[np.nan, np.nan, 1900., 2000., 2000.],
        feed_program_mm_min=2000., feed_source="measured_actual_feed", diameter_mm=10.))
    process = pd.DataFrame(dict(line=[14, 14], execution_instance=0,
        segment_start_mm=[0., 1.], segment_mm=[1., 2.], ap_mm=[1., 2.], ae_mm=2.,
        source_point_index=[0., 1.], source_file_line=[2, 3]))
    return parent, process


def unknown_grid(positions):
    parent, process = example()
    parent = pd.concat([parent.iloc[[0]]] * len(positions), ignore_index=True)
    parent["segment_mm"], parent["path_mm"] = positions, positions
    return parent, process


def test_only_missing_geometry_changes_and_all_parent_context_is_preserved():
    parent, process = example()
    parent.index = pd.Index([11, 11, 8, 3, 19], name="original_index")
    old, old_process = parent.copy(deep=True), process.copy(deep=True)
    result, receipt = fill_missing_process_support(parent, process, diameter_mm=10.)
    pd.testing.assert_frame_equal(parent, old)
    pd.testing.assert_frame_equal(process, old_process)
    pd.testing.assert_frame_equal(result[list(set(parent) - set(MUTABLE_COLUMNS))],
                                  parent[list(set(parent) - set(MUTABLE_COLUMNS))])
    pd.testing.assert_frame_equal(result.loc[parent.process_geometry_known, list(parent)],
                                  parent.loc[parent.process_geometry_known])
    assert result.ap_mm.tolist() == [1., 1., 1., 1.5, 2.]
    assert result.support_fill_restored.tolist() == [True, True, False, False, False]
    assert result.support_fill_source_file_line.iloc[:2].tolist() == [2., 2.]
    assert result.loc[~result.support_fill_restored, list(ADDED_COLUMNS)[1:]].isna().all().all()
    assert receipt["restored_points"] == 2 and receipt["remaining_missing_points"] == 0
    assert receipt["source_file_verified"] is False and receipt["training_released"] is False
    assert receipt["sequence_columns_preserved"] == ["sequence_id", "training_sequence_id", "nc_sequence_id"]


def test_interval_boundaries_origin_and_micro_gaps_are_exact():
    end = 1.
    start = 1. + 5e-11
    parent, process = unknown_grid([0., end, end + 2e-11, start, start + 1e-11, 2., 2. + 1e-11])
    process.loc[1, "segment_start_mm"] = start
    result, receipt = fill_missing_process_support(parent, process, diameter_mm=10.)
    assert result.support_fill_restored.tolist() == [True, True, False, False, True, True, False]
    assert result.ap_mm.iloc[[0, 1, 4, 5]].tolist() == [1., 1., 2., 2.]
    assert receipt["remaining_missing_reasons"] == {"outside_or_missing_source_interval": 3}


@pytest.mark.parametrize("length,count,legacy_restored", [
    (13.163390767199763, 26, [False, False, False]),
    (50.17331351316155, 100, [True, True, False]),
])
def test_uniform_source_endpoint_repairs_only_supported_point_not_true_gaps(
        tmp_path, length, count, legacy_restored):
    source_path = tmp_path / "process.txt"
    source_path.write_text("序号 N ap(mm) ae(mm) F(mm/min) MRR(mm3/min) s(mm) S(r/min) G\n" +
        "".join(f"{index} 14 1 2 2000 4000 {index} 8000 G1\n" for index in range(count)),
        encoding="utf-8")
    parent, _ = unknown_grid([length, np.nextafter(length, np.inf), length])
    parent.loc[2, "line"] = 15  # A separate NC line with no process source.
    parent["block_length_mm"] = length
    process = read_process_table(source_path, SimpleNamespace(frame=parent))
    legacy = process.copy(deep=True)
    legacy.loc[legacy.index[-1], "segment_mm"] = length * count / count
    old_result, _ = fill_missing_process_support(parent, legacy, diameter_mm=10.)
    assert old_result.support_fill_restored.tolist() == legacy_restored
    result, receipt = fill_missing_process_support(parent, process, diameter_mm=10.)
    assert result.support_fill_restored.tolist() == [True, False, False]
    assert result.ap_mm.iloc[0] == 1. and result.ae_mm.iloc[0] == 2.
    assert result.loc[1:, ["ap_mm", "ae_mm"]].isna().all().all()
    assert result.support_fill_source_point_index.iloc[0] == count - 1
    assert result.support_fill_interval_end_mm.iloc[0] == length
    assert receipt["remaining_missing_reasons"] == {"outside_or_missing_source_interval": 1, "no_process_row": 1}
    preserved = [column for column in parent if column not in MUTABLE_COLUMNS]
    pd.testing.assert_frame_equal(result[preserved], parent[preserved], check_exact=True)


def test_contiguous_boundary_belongs_to_left_cell_and_zero_contact_is_legal():
    parent, process = unknown_grid([0., 1., np.nextafter(1., 2.), 2.])
    process.loc[0, "ap_mm"] = 0.
    result, _ = fill_missing_process_support(parent, process, diameter_mm=10.)
    assert result.ap_mm.tolist() == [0., 0., 2., 2.]
    assert result.cutting_contact.tolist() == [False, False, True, True]
    assert result.process_geometry_known.all()


def test_conflicts_and_missing_visits_stay_unknown():
    parent, process = example()
    parent.loc[0, "process_geometry_conflict_support"] = True
    parent.loc[1, "execution_instance"] = 1
    result, receipt = fill_missing_process_support(parent, process, diameter_mm=10.)
    pd.testing.assert_frame_equal(result[list(parent)], parent)
    assert receipt["remaining_missing_reasons"] == {"preserved_source_conflict": 1, "no_process_row": 1}


def test_repeated_nc_line_never_borrows_another_execution_source():
    parent, process = unknown_grid([0., .5])
    parent["execution_instance"] = [0, 1]
    second = process.copy()
    second["execution_instance"], second["ap_mm"] = 1, 3.
    second["source_file_line"] += 2
    result, _ = fill_missing_process_support(parent, pd.concat([process, second]), diameter_mm=10.)
    assert result.ap_mm.tolist() == [1., 3.]
    assert result.support_fill_source_file_line.tolist() == [2., 4.]


@pytest.mark.parametrize("value", [-1., 11., np.nan, np.inf])
def test_invalid_source_geometry_is_not_filled(value):
    parent, process = example()
    process.loc[0, "ae_mm"] = value
    result, receipt = fill_missing_process_support(parent, process, diameter_mm=10.)
    pd.testing.assert_frame_equal(result[list(parent)], parent)
    assert receipt["remaining_missing_reasons"] == {"invalid_source_geometry": 2}


@pytest.mark.parametrize("start,end", [(.5, 2.), (1., 1.), (np.nan, 2.), (-1., 2.)])
def test_malformed_or_overlapping_source_is_rejected(start, end):
    parent, process = example()
    process.loc[1, ["segment_start_mm", "segment_mm"]] = [start, end]
    with pytest.raises(ValueError, match="区间非法或重叠"):
        fill_missing_process_support(parent, process, diameter_mm=10.)


def test_absent_declared_intervals_are_not_inferred_from_neighbor_points():
    parent, process = example()
    result, receipt = fill_missing_process_support(parent, process.drop(columns="segment_start_mm"), diameter_mm=10.)
    pd.testing.assert_frame_equal(result[list(parent)], parent)
    assert receipt["remaining_missing_reasons"] == {"no_declared_source_interval": 2}


@pytest.mark.parametrize("column,value", [("source_file_line", np.nan), ("source_file_line", 0),
                                         ("source_point_index", .5)])
def test_restored_points_require_traceable_source_identifiers(column, value):
    parent, process = example()
    process[column] = process[column].astype(float)
    process.loc[0, column] = value
    with pytest.raises(ValueError, match="来源标识无效"):
        fill_missing_process_support(parent, process, diameter_mm=10.)


def test_partial_parent_geometry_is_rejected_instead_of_overwritten():
    parent, process = example()
    parent.loc[0, "ap_mm"] = 7.
    with pytest.raises(ValueError, match="部分已知"):
        fill_missing_process_support(parent, process, diameter_mm=10.)


def test_repeated_derivation_is_rejected():
    parent, process = example()
    result, _ = fill_missing_process_support(parent, process, diameter_mm=10.)
    with pytest.raises(ValueError, match="重复派生"):
        fill_missing_process_support(result, process, diameter_mm=10.)


def source_sample(tmp_path):
    parent, process = example()
    path = tmp_path / "canonical_process.csv"
    process.to_csv(path, index=False, encoding="utf-8")
    sample = SampleSet(parent, dict(record_id="synthetic", program_version="p1", machine_version="m1",
        diameter_mm=10., file_sha256={"process": file_hash(path)}, resolved_files={"process": str(path)},
        prepared_measurement_sha256="original-label-evidence", hybrid_label={"evidence_grade": "frozen"}))
    return sample, path


def test_verified_source_wrapper_adds_only_independent_receipt(tmp_path):
    sample, path = source_sample(tmp_path)
    original = deepcopy(sample.metadata)
    result = fill_sample_from_process_source(sample, path)
    assert sample.metadata == original
    assert {key: value for key, value in result.metadata.items() if key != "support_fill"} == original
    receipt = result.metadata["support_fill"]
    assert receipt["source_file_verified"] is True
    assert receipt["source"]["sha256"] == file_hash(path)
    assert receipt["source"]["parent_registered_path"] == str(path)
    assert receipt["restored_points"] == 2
    assert result.metadata["prepared_measurement_sha256"] == "original-label-evidence"


def test_source_hash_mismatch_is_rejected_before_parsing(tmp_path, monkeypatch):
    sample, path = source_sample(tmp_path)
    path.write_text("different-source", encoding="utf-8")
    def forbidden(*args, **kwargs):
        pytest.fail("Unverified source must not be parsed")
    monkeypatch.setattr("spindle_power.load_prediction.samples.read_process_table", forbidden)
    with pytest.raises(ValueError, match="SHA-256不符"):
        fill_sample_from_process_source(sample, path)


def test_source_changed_during_parsing_is_rejected(tmp_path, monkeypatch):
    from spindle_power.load_prediction import samples
    sample, path = source_sample(tmp_path)
    original = samples.read_process_table
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        path.write_text("changed-during-read", encoding="utf-8")
        return result
    monkeypatch.setattr(samples, "read_process_table", changed)
    with pytest.raises(ValueError, match="期间发生变化"):
        fill_sample_from_process_source(sample, path)
