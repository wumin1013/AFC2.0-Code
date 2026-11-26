"""Durable regressions for the optional production reference-support output."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from spindle_power.load_prediction.feed_samples import align_spatial_channels
from spindle_power.load_prediction.measurement import Measurement


def fixture(x=None, times=None, raw=None):
    x = np.arange(7.) if x is None else np.asarray(x, float)
    count = len(x)
    times = np.arange(count) * .001 if times is None else np.asarray(times)
    raw = np.ones(count) if raw is None else np.asarray(raw)
    frame = pd.DataFrame(dict(line=1, execution_instance=0, segment_mm=np.arange(7.),
        path_mm=np.arange(7.), block_length_mm=6., x_mm=np.arange(7.), y_mm=0., z_mm=0.,
        start_x_mm=0., end_x_mm=6., start_y_mm=0., end_y_mm=0., start_z_mm=0., end_z_mm=0.,
        arc_radius_mm=0., arc_sweep_rad=0., arc_i_mm=0., arc_j_mm=0., arc_k_mm=0.))
    measurement = Measurement('synthetic', times, dict(program_line_raw=raw, actual_x_mm=x,
        actual_y_mm=np.zeros(count), actual_z_mm=np.zeros(count)), np.ones(count, bool),
        {'sample_period_s': .001}, source_indices=np.arange(count))
    return frame, measurement, dict(reference_power_cut_w=x + 10.)


def aligned(frame, measurement, channels, valid=None):
    args = dict(valid=valid, position_offset_mm=[0., 0., 0.])
    default, default_info = align_spatial_channels(frame, measurement, channels, **args)
    result, info = align_spatial_channels(frame, measurement, channels, support_provenance=True, **args)
    pd.testing.assert_frame_equal(result[default.columns], default, check_exact=True)
    assert {key: value for key, value in info.items() if key != 'support_segments'} == default_info
    return result, info


def test_default_api_keeps_historical_fixed_vector_and_excludes_support_fields():
    frame, measurement, channels = fixture()
    result, info = align_spatial_channels(frame, measurement, channels, position_offset_mm=[0., 0., 0.])
    assert list(result) == ['reference_power_cut_w', 'alignment_error_mm', 'aligned']
    np.testing.assert_array_equal(result.reference_power_cut_w, np.arange(7.) + 10.)
    np.testing.assert_array_equal(result.alignment_error_mm, np.zeros(7))
    assert result.aligned.all() and info['aligned_points'] == 7
    assert 'support_segments' not in info


def test_time_gap_separates_support_between_adjacent_valid_grid_points():
    frame, measurement, channels = fixture(times=[0., .001, .002, .100, .101, .102, .103])
    result, info = aligned(frame, measurement, channels)
    assert result.aligned.all()
    np.testing.assert_array_equal(result.support_segment_id, [0, 0, 0, 1, 1, 1, 1])
    assert [(r['source_start_index'], r['source_end_index_exclusive']) for r in info['support_segments']] == [(0, 3), (3, 7)]


def test_invalid_base_point_leaves_a_real_gap_and_disjoint_support_ranges():
    frame, measurement, channels = fixture()
    result, info = aligned(frame, measurement, channels, valid=np.array([1, 1, 1, 0, 1, 1, 1], bool))
    assert not result.aligned.iloc[3] and result.support_segment_id.iloc[3] == -1
    assert np.isnan(result.reference_power_cut_w.iloc[3])
    assert [(r['source_start_index'], r['source_end_index_exclusive']) for r in info['support_segments']] == [(0, 3), (4, 7)]


def test_projection_rejection_splits_support_without_joining_its_neighbours():
    frame, measurement, channels = fixture()
    measurement.values['actual_y_mm'][3] = 1.
    result, info = aligned(frame, measurement, channels)
    assert not result.aligned.iloc[3] and result.support_segment_id.iloc[3] == -1
    assert result.support_segment_id.iloc[2] != result.support_segment_id.iloc[4]
    assert len(info['support_segments']) == 2


def test_duplicate_coverage_is_unknown_in_both_values_and_support():
    frame, measurement, channels = fixture(x=[0., 1., 2., 1., 0., 1., 2., 3.])
    result, info = aligned(frame, measurement, channels)
    assert not result.aligned.iloc[1] and result.support_segment_id.iloc[1] == -1
    assert np.isnan(result.reference_power_cut_w.iloc[1])
    assert any('重复覆盖' in message for message in info['issues'])


def test_visits_keep_execution_instance_and_original_source_ranges():
    frame, measurement, channels = fixture(x=[0., 1., 9., 9.5, 0., 1.], raw=[1, 1, 2, 2, 1, 1])
    frame = pd.concat([frame.iloc[:2], frame.iloc[:2].assign(execution_instance=1, path_mm=[10., 11.])], ignore_index=True)
    result, info = aligned(frame, measurement, channels)
    assert result.aligned.all()
    assert [row['execution_instance'] for row in info['support_segments']] == [0, 1]
    assert result.support_segment_id.iloc[1] != result.support_segment_id.iloc[2]


def test_incomplete_original_indices_are_refused_only_by_opted_in_provenance():
    frame, measurement, channels = fixture()
    measurement.source_indices[3:] += 1
    default, _ = align_spatial_channels(frame, measurement, channels, position_offset_mm=[0., 0., 0.])
    assert list(default) == ['reference_power_cut_w', 'alignment_error_mm', 'aligned']
    with pytest.raises(ValueError, match='完整连续原始索引'):
        align_spatial_channels(frame, measurement, channels, position_offset_mm=[0., 0., 0.], support_provenance=True)
