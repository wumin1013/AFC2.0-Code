"""Real-export column semantics and arclength mapping, using small fixtures."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spindle_power.load_prediction.samples import read_process_table, _prepare_process_features, LOAD_PROCESS_RULES_VERSION
from spindle_power.load_prediction.sample_rules import apply_process_contact


def test_variable_trailing_nc_words_do_not_shift_numeric_columns(tmp_path):
    path = tmp_path / "process.txt"
    path.write_text("序号 N ap(mm) ae(mm) F(mm/min) MRR(mm3/min) s(mm) S(r/min) G\n\n"
                    "0 20 0 0 2000 0 10 8000 G00 X10. Y0. Z3.\n"
                    "1 21 1 2 2000 4000 10.5 8000 G01 X10.5 F2000.\n"
                    "2 21 1 2 2000 4000 11 8000 X11.\n"
                    "3 22 2 3 2000 12000 11.5 8000 G01 X11.5 Y0.\n"
                    "4 22 2 3 2000 12000 12 8000 X12.\n", encoding="utf-8")
    grid = pd.DataFrame(dict(line=[21, 21, 22, 22], execution_instance=0,
                             segment_mm=[.5, 1., .5, 1.], path_mm=[.5, 1., 1.5, 2.],
                             sequence_id=0, motion=1, block_length_mm=1.))
    result = read_process_table(path, SimpleNamespace(frame=grid), steady_lines=[22])
    assert result.line.tolist() == [20, 21, 21, 22, 22]
    assert result.mapping_status.iloc[0] == "unmapped_nc_group"
    assert np.isnan(result.segment_mm.iloc[0])
    assert result.ap_mm.tolist() == [0, 1, 1, 2, 2]
    assert result.ae_mm.tolist() == [0, 2, 2, 3, 3]
    np.testing.assert_allclose(result.segment_mm.iloc[1:], [.5, 1., .5, 1.])
    assert result.steady.tolist() == [False, False, False, True, True]
    contact = apply_process_contact(grid, result)
    assert contact.cutting_contact.all()


@pytest.mark.parametrize("length", [1.3223992664849737, .9800984644412057, .9723304595146649])
def test_canonical_csv_round_trip_preserves_real_nc_endpoint_support(tmp_path, length):
    # Segment lengths observed in the real outer/surface/step NC paths expose
    # last-bit rounding at strict interpolation-support boundaries.
    raw_path = tmp_path / "process.txt"
    raw_path.write_text("序号 N ap(mm) ae(mm) F(mm/min) MRR(mm3/min) s(mm) S(r/min) G\n"
        "0 21 1 2 2000 4000 0.5 8000 G1\n"
        "1 21 1 2 2000 4000 1.0 8000 G1\n", encoding="utf-8")
    grid = pd.DataFrame(dict(line=21, execution_instance=0, segment_mm=[length / 2, length],
        path_mm=[length / 2, length], block_length_mm=length, sequence_id=0, motion=1,
        spindle_cmd_rpm=8000))
    plan = SimpleNamespace(frame=grid, program_version="nc", machine_version="machine")
    raw = read_process_table(raw_path, plan, steady_lines=[21])
    canonical_path = tmp_path / "canonical.csv"
    raw.to_csv(canonical_path, index=False, encoding="utf-8")
    restored = read_process_table(canonical_path, plan)
    pd.testing.assert_frame_equal(raw, restored, check_exact=True)
    expected_contact = apply_process_contact(grid, raw)
    restored_contact = apply_process_contact(grid, restored)
    pd.testing.assert_frame_equal(expected_contact, restored_contact, check_exact=True)
    meta = dict(program_version="nc", diameter_mm=10, teeth=4)
    pd.testing.assert_frame_equal(_prepare_process_features(raw, plan, meta).frame,
        _prepare_process_features(restored, plan, meta).frame, check_exact=True)
    assert restored_contact.cutting_contact.eq(True).all()


def test_historical_canonical_retains_its_endpoint_and_mapping_version(tmp_path):
    length, count = 13.163390767199763, 26
    legacy_end = length * count / count
    original = pd.DataFrame(dict(line=[21], execution_instance=[0],
        segment_start_mm=[0.], segment_mm=[legacy_end], ap_mm=[1.], ae_mm=[2.],
        process_mapping_rules_version=["afc.process_mapping.2"]))
    path = tmp_path / "historical_canonical.csv"
    original.to_csv(path, index=False, encoding="utf-8")
    original_bytes = path.read_bytes()
    restored = read_process_table(path, SimpleNamespace(frame=pd.DataFrame()))
    pd.testing.assert_frame_equal(restored, original, check_exact=True)
    assert restored.segment_mm.iloc[0] != length
    assert restored.process_mapping_rules_version.iloc[0] == "afc.process_mapping.2"
    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize("bad", ["nan", "invalid"])
def test_non_numeric_or_nonfinite_required_process_values_are_rejected(tmp_path, bad):
    path = tmp_path / "process.txt"
    path.write_text("序号 N ap(mm) ae(mm) F(mm/min) MRR(mm3/min) s(mm) S(r/min) G\n"
                    f"0 21 1 2 2000 4000 {bad} 8000 G01 X10\n", encoding="utf-8")
    grid = pd.DataFrame(dict(line=[21], execution_instance=[0]))
    with pytest.raises(ValueError):
        read_process_table(path, SimpleNamespace(frame=grid))


def test_repeated_raw_s_keeps_distinct_points_at_uniform_nc_positions(tmp_path):
    path = tmp_path / "process.txt"
    path.write_text("序号 N ap(mm) ae(mm) F(mm/min) MRR(mm3/min) s(mm) S(r/min) G\n"
        "0 21 1 2 2000 4000 100.0 8000 G1 X2\n"
        "1 21 2 2 2000 8000 100.0 8000 G1 X2\n"
        "2 21 2 2 2000 8000 100.0 8000 G1 X2\n"
        "3 21 3 2 2000 12000 100.1 8000 G1 X2\n", encoding="utf-8")
    grid = pd.DataFrame(dict(line=21, execution_instance=0, segment_mm=[0., 2.], path_mm=[0., 2.], block_length_mm=2.))
    result = read_process_table(path, SimpleNamespace(frame=grid))
    assert len(result) == 4
    assert result.source_point_index.tolist() == [0., 1., 2., 3.]
    assert result.source_file_line.tolist() == [2, 3, 4, 5]
    assert result.source_path_mm.tolist() == [100., 100., 100., 100.1]
    assert result.ap_mm.tolist() == [1., 2., 2., 3.]
    np.testing.assert_allclose(result.segment_mm, [.5, 1., 1.5, 2.])
    np.testing.assert_allclose(result.segment_start_mm, [0., .5, 1., 1.5])
    assert not result.duplicated(["line", "execution_instance", "segment_mm"]).any()


def test_repeated_instruction_groups_map_to_distinct_execution_instances(tmp_path):
    path = tmp_path / "process.txt"
    path.write_text("序号 N ap(mm) ae(mm) F(mm/min) MRR(mm3/min) s(mm) S(r/min) G\n"
        "0 21 1 2 2000 4000 1 8000 G1 X2\n"
        "1 22 1 2 2000 4000 2 8000 G1 Y2\n"
        "2 21 2 2 2000 8000 3 8000 G1 X2\n", encoding="utf-8")
    grid = pd.DataFrame(dict(line=[21, 22, 21], execution_instance=[0, 0, 1],
        segment_mm=[2., 3., 4.], path_mm=[2., 5., 9.], block_length_mm=[2., 3., 4.]))
    result = read_process_table(path, SimpleNamespace(frame=grid))
    assert result.execution_instance.tolist() == [0, 0, 1]
    assert result.segment_mm.tolist() == [2., 3., 4.]
    assert result.nc_path_mm.tolist() == [2., 5., 9.]
    mismatched = read_process_table(path, SimpleNamespace(frame=grid.iloc[:2]))
    assert mismatched.loc[mismatched.line == 21, "mapping_status"].eq("unmapped_execution_count").all()
    assert mismatched.loc[mismatched.line == 21, "segment_mm"].isna().all()


def test_contact_duplicates_merge_only_agreeing_boolean_states_and_preserve_geometry():
    grid = pd.DataFrame(dict(line=21, execution_instance=0, segment_mm=np.arange(9) * .5,
                             path_mm=np.arange(9) * .5, sequence_id=0, motion=1))
    process = pd.DataFrame(dict(line=21, execution_instance=0,
        segment_mm=[0., 0., 1., 2., 2., 3., 4.],
        ap_mm=[1., 2., 1., 0., 1., 0., 0.], ae_mm=[2., 3., 2., 0., 2., 0., 0.]))
    original = process.copy(deep=True)
    result = apply_process_contact(grid, process)
    assert result.cutting_contact.iloc[:3].tolist() == [True, True, True]
    assert result.cutting_contact.iloc[3:6].isna().all()
    assert result.cutting_contact.iloc[6:].tolist() == [False, False, False]
    pd.testing.assert_frame_equal(process, original)


def test_contact_transition_is_unknown_and_zero_axes_cannot_interpolate_into_contact():
    grid = pd.DataFrame(dict(line=21, execution_instance=0, segment_mm=[0., .5, 1., 1.5, 2.],
                             path_mm=[0., .5, 1., 1.5, 2.], sequence_id=0, motion=1))
    process = pd.DataFrame(dict(line=21, execution_instance=0, segment_mm=[0., 1., 2.],
                                ap_mm=[1., 0., 1.], ae_mm=[0., 1., 1.]))
    result = apply_process_contact(grid, process)
    assert result.cutting_contact.iloc[:3].tolist() == [False, False, False]
    assert pd.isna(result.cutting_contact.iloc[3])
    assert bool(result.cutting_contact.iloc[4]) is True


def _load_mapping(process):
    position = np.arange(9) * .5
    grid = pd.DataFrame(dict(line=21, execution_instance=0, segment_mm=position,
                            path_mm=position, sequence_id=0, motion=1, spindle_cmd_rpm=6000))
    plan = SimpleNamespace(frame=grid, program_version="nc", machine_version="machine")
    return _prepare_process_features(process, plan, dict(program_version="nc", diameter_mm=10, teeth=4))


def test_load_geometry_conflict_masks_only_dependent_support_without_averaging():
    process = pd.DataFrame(dict(line=21, execution_instance=0,
        segment_mm=[0., 0., 1., 2., 2., 3., 4.],
        ap_mm=[1., 1., 1., 1., 3., 2., 2.], ae_mm=2., steady=True))
    original = process.copy(deep=True)
    result = _load_mapping(process)
    assert result.frame.process_geometry_known.tolist() == [True, True, True, False, False, False, True, True, True]
    assert result.frame.process_geometry_conflict_support.tolist() == [False, False, False, True, True, True, False, False, False]
    assert result.frame.ap_mm.iloc[3:6].isna().all()
    assert result.frame.ap_mm.iloc[:3].tolist() == [1., 1., 1.]
    assert result.frame.ap_mm.iloc[6:].tolist() == [2., 2., 2.]
    assert result.frame.steady.tolist() == result.frame.process_geometry_known.tolist()
    assert result.metadata["load_process_rules_version"] == LOAD_PROCESS_RULES_VERSION
    assert result.metadata["process_mapping_report"][0]["conflicting_positions"] == [2.]
    pd.testing.assert_frame_equal(process, original)


def test_load_geometry_preserves_zero_contact_and_conservative_steady_support():
    process = pd.DataFrame(dict(line=21, execution_instance=0,
        segment_mm=[0., 1., 2., 3., 3., 4.],
        ap_mm=[1., 0., 1., 1., 1., 1.], ae_mm=[0., 1., 1., 1., 1., 1.],
        steady=[True, True, True, True, False, True]))
    result = _load_mapping(process).frame
    assert result.process_geometry_known.tolist() == [True, False, True, False, True, True, True, True, True]
    assert result.cutting_contact.iloc[:4].tolist() == [False, False, False, False]
    assert result.steady.tolist() == [True, False, True, False, True, False, False, False, True]
