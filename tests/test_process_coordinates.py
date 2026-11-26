"""Source-grounded coordinates and exact short-arc endpoint support."""
from pathlib import Path
import math
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spindle_power.load_prediction.process_coordinates import (
    ProcessCoordinates, reconstruct_process_coordinates, source_cell_indices, source_distance_compatible,
    prepare_data_supported_geometry, SHORT_ARC_ENDPOINT_POLICY,
)


def test_linear_endpoint_is_exact_and_next_float_is_not_supported():
    length = 13.163390767199763
    coordinates = reconstruct_process_coordinates(length_mm=length, point_count=26, motion=1)
    assert coordinates.positions_mm[-1] == length
    assert coordinates.unsupported_tail_mm == 0.
    np.testing.assert_array_equal(source_cell_indices(
        [0., coordinates.positions_mm[0], length, np.nextafter(length, np.inf), np.nan], coordinates),
        [0, 0, 25, -1, -1])


def test_arc_has_fixed_steps_and_a_distinct_terminal_remainder():
    length, radius = 7.881539, 5.
    coordinates = reconstruct_process_coordinates(length_mm=length, point_count=15,
        motion=2, radius_mm=radius, sweep_rad=-length / radius)
    assert coordinates.positions_mm[:14] == tuple(np.arange(1, 15) * .5)
    assert coordinates.positions_mm[-1] == length
    assert coordinates.cumulative_source_distance_mm[-1] < length
    assert coordinates.rule == "arc_fixed_half_plus_end"


def test_short_arc_closes_last_cell_without_moving_its_nine_knots():
    length = .2 * math.pi / 2
    coordinates = reconstruct_process_coordinates(length_mm=length, point_count=9,
        motion=3, radius_mm=.2, sweep_rad=math.pi / 2)
    expected = tuple(length * k / 10 for k in range(1, 10))
    assert coordinates.positions_mm == expected
    assert coordinates.cell_ends_mm == (*expected[:-1], length)
    assert coordinates.terminal_support_end_mm == length
    assert coordinates.unsupported_tail_mm == 0.
    np.testing.assert_array_equal(coordinates.cumulative_source_distance_mm,
        np.cumsum(.4 * np.sin(np.diff([0., *expected]) / .4)))
    np.testing.assert_array_equal(source_cell_indices(
        [0., length * 9 / 10, length, np.nextafter(length, np.inf)], coordinates), [0, 8, 8, -1])


def short_arc_geometry():
    length = .2 * math.pi / 2
    positions = np.array([0., length * .85, length * .9, length, np.nextafter(length, np.inf)])
    grid = pd.DataFrame(dict(line=21, execution_instance=0, segment_mm=positions,
        path_mm=20. + positions, block_length_mm=length, motion=3, arc_radius_mm=.2,
        arc_sweep_rad=math.pi / 2, start_z_mm=0., end_z_mm=0., spindle_cmd_rpm=8000.))
    source = pd.DataFrame(dict(line=21, execution_instance=0, source_file_line=np.arange(2, 11),
        source_point_index=np.arange(9), segment_mm=np.arange(1, 10) * length / 9,
        segment_start_mm=np.arange(9) * length / 9, ap_mm=np.arange(1., 10.), ae_mm=2.,
        nc_path_mm=20. + np.arange(1, 10) * length / 9, mapping_status="mapped_uniform_nc"))
    knots = [length * k / 10 for k in range(1, 10)]
    cumulative = np.cumsum(.4 * np.sin(np.diff([0., *knots]) / .4))
    tokens = {index + 2: format(100. + value, ".6g") for index, value in enumerate(cumulative)}
    source["raw_s"] = list(tokens.values())
    return source, SimpleNamespace(frame=grid, program_version="nc", machine_version="machine"), tokens


def prepare_geometry(source, plan, tokens):
    return prepare_data_supported_geometry(source, plan,
        dict(program_version="nc", diameter_mm=10., teeth=4), source_distance_tokens=tokens)


def test_geometry_entry_preserves_source_knots_and_uses_last_real_value_at_endpoint():
    source, plan, tokens = short_arc_geometry()
    original, original_grid = source.copy(deep=True), plan.frame.copy(deep=True)
    sample, rebuilt = prepare_geometry(source, plan, tokens)
    length = plan.frame.block_length_mm.iloc[0]
    knots = np.array([length * k / 10 for k in range(1, 10)])
    assert len(rebuilt) == len(original) == 9
    pd.testing.assert_frame_equal(source, original, check_exact=True)
    pd.testing.assert_frame_equal(plan.frame, original_grid, check_exact=True)
    preserved = ["source_file_line", "source_point_index", "raw_s", "ap_mm", "ae_mm"]
    pd.testing.assert_frame_equal(rebuilt[preserved], original[preserved], check_exact=True)
    np.testing.assert_array_equal(rebuilt.segment_mm, knots)
    np.testing.assert_array_equal(rebuilt.nc_path_mm, 20. + knots)
    assert rebuilt.mapping_status.eq("mapped_data_supported_nc").all()
    assert rebuilt.source_cell_end_mm.iloc[-1] == length
    assert rebuilt.endpoint_support_policy.tolist() == ["none"] * 8 + [SHORT_ARC_ENDPOINT_POLICY]
    # The inner interpolation is exactly the old nine-knot interpolation.
    np.testing.assert_array_equal(sample.frame.ap_mm.iloc[1:3],
        np.interp(plan.frame.segment_mm.iloc[1:3], knots, original.ap_mm))
    assert sample.frame.process_geometry_known.tolist() == [True, True, True, True, False]
    assert sample.frame.ap_mm.iloc[3] == 9. and sample.frame.ae_mm.iloc[3] == 2.
    assert sample.frame.support_fill_source_file_line.iloc[3] == 10.
    assert sample.frame.support_fill_interval_end_mm.iloc[3] == length
    assert not {"power_cut_w", "valid_main", "valid_residual", "training_sequence_id"} & set(sample.frame)
    assert sample.metadata["training_released"] is False
    assert sample.metadata["steady_qualification_rebuilt"] is False


def test_endpoint_support_does_not_cross_nc_lines_or_execution_instances():
    source, plan, tokens = short_arc_geometry()
    unbound = pd.concat([plan.frame.iloc[[3]]] * 2, ignore_index=True)
    unbound.loc[0, "line"] = 22
    unbound.loc[1, "execution_instance"] = 1
    plan.frame = pd.concat([plan.frame, unbound], ignore_index=True)
    sample, _ = prepare_geometry(source, plan, tokens)
    assert sample.frame.ap_mm.iloc[-2:].isna().all()
    assert not sample.frame.process_geometry_known.iloc[-2:].any()
    # A different segment has its own values; it cannot change NC21's endpoint.
    adjacent = source.copy(deep=True)
    adjacent["line"] = 22
    adjacent["source_file_line"] += 9
    adjacent["source_point_index"] += 9
    adjacent["ap_mm"] = 42.
    both = pd.concat([source, adjacent], ignore_index=True)
    both_tokens = {**tokens, **{key + 9: value for key, value in tokens.items()}}
    changed, _ = prepare_geometry(both, plan, both_tokens)
    assert changed.frame.ap_mm.iloc[3] == sample.frame.ap_mm.iloc[3] == 9.
    assert changed.frame.ap_mm.iloc[-2] == 42.
    assert np.isnan(changed.frame.ap_mm.iloc[-1])


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -1., 11.])
def test_short_arc_closure_does_not_make_invalid_terminal_source_legal(invalid):
    source, plan, tokens = short_arc_geometry()
    source.loc[8, "ae_mm"] = invalid
    sample, _ = prepare_geometry(source, plan, tokens)
    assert not sample.frame.process_geometry_known.iloc[3]
    assert np.isnan(sample.frame.ap_mm.iloc[3])


def test_geometry_entry_rejects_ambiguous_source_index_and_incompatible_distance():
    source, plan, tokens = short_arc_geometry()
    source.index = [0] * len(source)
    with pytest.raises(ValueError, match="唯一"):
        prepare_geometry(source, plan, tokens)
    source = source.reset_index(drop=True)
    tokens[10] = "200"
    with pytest.raises(ValueError, match="不兼容"):
        prepare_geometry(source, plan, tokens)


def test_source_format_checks_rounded_values_without_using_them_as_coordinates():
    coordinates = reconstruct_process_coordinates(length_mm=1., point_count=2, motion=1)
    assert source_distance_compatible(["10.5", "11"], coordinates)["compatible"]
    assert not source_distance_compatible(["10.5", "11.1"], coordinates)["compatible"]
    repeated = reconstruct_process_coordinates(length_mm=.02, point_count=2, motion=1)
    assert source_distance_compatible(["100000", "100000"], repeated)["compatible"]
    assert repeated.positions_mm == (.01, .02)


def test_power_of_ten_uses_its_finer_lower_rounding_interval():
    coordinates = reconstruct_process_coordinates(length_mm=1.00016, point_count=2, motion=1)
    assert not source_distance_compatible(["10", "10.5"], coordinates)["compatible"]


def test_inconsistent_coordinate_arrays_cannot_silently_skip_a_source_token():
    with pytest.raises(ValueError):
        ProcessCoordinates("invalid", (.5, 1.), (0., .5), (.5,), 0.)


@pytest.mark.parametrize("tokens", [["1.000001", "1.5"], ["nan", "1.5"], ["-1", "1.5"], ["x", "1.5"], []])
def test_unsupported_source_formats_are_not_silently_relaxed(tokens):
    coordinates = reconstruct_process_coordinates(length_mm=1., point_count=2, motion=1)
    with pytest.raises(ValueError):
        source_distance_compatible(tokens, coordinates)


@pytest.mark.parametrize("changes", [dict(point_count=8), dict(radius_mm=.21),
    dict(z_change_mm=.001), dict(motion=2), dict(length_mm=0.), dict(point_count=True)])
def test_unstudied_arc_families_fail_closed(changes):
    arguments = dict(length_mm=.2 * math.pi / 2, point_count=9, motion=3,
                     radius_mm=.2, sweep_rad=math.pi / 2)
    arguments.update(changes)
    with pytest.raises(ValueError):
        reconstruct_process_coordinates(**arguments)
