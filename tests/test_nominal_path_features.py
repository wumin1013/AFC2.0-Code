"""Physical scaling and information-boundary checks for the 18-D route."""
import numpy as np
import pandas as pd
import pytest

from spindle_power.load_prediction.samples import (
    NOMINAL_ROUTE, SampleSet, assert_feed_inputs, load_input_view,
    path_feature_names, path_features,
)


def frame():
    return pd.DataFrame({
        "diameter_mm": [10., 10.], "ap_mm": [2., 2.], "ae_mm": [3., 3.],
        "feed_input_mm_min": [1000., 1000.], "feed_actual_mm_min": [1000., 1000.],
        "feed_program_mm_min": [2000., 2000.], "spindle_cmd_rpm": [8000., 8000.],
        "spindle_rpm": [9000., 9000.], "teeth": [3., 3.],
        "curvature_per_mm": [.02, .02], "block_length_mm": [20., 20.],
        "entry_angle_rad": [0., np.pi / 2], "exit_angle_rad": [0., np.pi],
        "direction_x": [1., 0.], "direction_y": [0., 1.], "direction_z": [0., 0.],
        "segment_fraction": [.25, .75], "previous_block_length_mm": [0., 5.],
        "next_block_length_mm": [5., 0.], "previous_feed_program_mm_min": [0., 1500.],
        "next_feed_program_mm_min": [2500., 0.], "has_previous_block": [0., 1.],
        "has_next_block": [1., 0.], "sequence_id": [0, 0],
        "training_sequence_id": [0, 4], "execution_instance": [0, 0],
        "path_mm": [0., 1.], "power_cut_w": [5., 8.], "valid_residual": [True, False],
    })


def test_feature_order_units_and_distinguishable_geometry():
    f = frame()
    x = path_features(f, [20., 20.], 10., 5., 2000., route=NOMINAL_ROUTE)
    assert len(path_feature_names(NOMINAL_ROUTE)) == 18
    np.testing.assert_allclose(x[0], [2., .2, .3, .5, .2, 2., 0., 0., 1., 0., 0.,
                                     .25, 0., .5, 0., 1.25, 0., 1.])
    # Identical legacy six features can describe different turning geometry.
    np.testing.assert_array_equal(x[0, :6], x[1, :6])
    assert not np.array_equal(x[0], x[1])
    assert path_features(f, [20., 20.], 10., 5., 2000.).shape == (2, 6)


def test_no_observed_values_enter_programmed_features():
    original = frame()
    changed = original.copy()
    changed["feed_actual_mm_min"] = [999999., np.nan]
    changed["power_cut_w"] = [-99999., np.nan]
    changed["valid_residual"] = [False, True]
    changed["training_sequence_id"] = [10, 200]
    changed["time_s"] = [1., np.nan]
    left = load_input_view(SampleSet(original, {"id": "same"}), "programmed")
    right = load_input_view(SampleSet(changed, {"id": "same"}), "programmed")
    np.testing.assert_array_equal(left.frame.sequence_id, right.frame.sequence_id)
    np.testing.assert_array_equal(path_features(left.frame, [20., 20.], 0., 1., 2000., route=NOMINAL_ROUTE),
                                  path_features(right.frame, [20., 20.], 0., 1., 2000., route=NOMINAL_ROUTE))
    np.testing.assert_array_equal(left.frame.feed_input_mm_min, [2000., 2000.])
    np.testing.assert_array_equal(left.frame.spindle_rpm, [8000., 8000.])
    np.testing.assert_array_equal(original.feed_input_mm_min, [1000., 1000.])
    assert_feed_inputs(left.frame, "programmed")


def test_measured_view_never_falls_back_to_programmed():
    f = frame().drop(columns="feed_actual_mm_min")
    with pytest.raises(ValueError, match="feed_actual_mm_min"):
        load_input_view(SampleSet(f, {}), "measured")


@pytest.mark.parametrize("kind", ["missing", "neighbor", "nan", "marker"])
def test_invalid_geometry_is_rejected(kind):
    f = frame()
    if kind == "missing":
        f = f.drop(columns="entry_angle_rad")
    elif kind == "neighbor":
        f.loc[0, "previous_block_length_mm"] = 100.
    elif kind == "nan":
        f.loc[0, "direction_x"] = np.nan
    else:
        f.loc[0, "has_previous_block"] = .5
    with pytest.raises(ValueError):
        path_features(f, [1., 1.], 0., 1., 2000., route=NOMINAL_ROUTE)
