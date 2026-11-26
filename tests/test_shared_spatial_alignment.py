"""Feed/power share spatial support; channel gaps never redefine NC visits."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spindle_power.load_prediction.feed import FeedPlan, MachineConfig
from spindle_power.load_prediction.feed_samples import align_spatial_channels, prepare_feed_features, prepare_feed_samples
from spindle_power.load_prediction.measurement import Measurement
from spindle_power.load_prediction.samples import prepare_load_samples


def _case(tmp_path):
    nc = tmp_path / "synthetic.nc"
    nc.write_text("G1 X10 F600 S1000\n", encoding="utf-8")
    features = prepare_feed_features(nc, MachineConfig(initial_position_mm=[0., 0., 0.]),
        program_id="synthetic", feed_override=1., machining_mode="synthetic", machine_settings_version="synthetic")
    features.frame["cutting_contact"] = True
    positions = np.arange(-.65, 9.35001, .05)
    values = dict(program_line_raw=np.ones(len(positions)), actual_x_mm=positions,
        actual_y_mm=np.zeros(len(positions)), actual_z_mm=np.zeros(len(positions)),
        actual_vx_mm_min=420 + 3 * positions, actual_vy_mm_min=np.zeros(len(positions)),
        actual_vz_mm_min=np.zeros(len(positions)), spindle_speed_rpm=np.full(len(positions), 1000.),
        spindle_power_w=180 + 2 * positions)
    record = Measurement("synthetic_only", np.arange(len(positions)) * .001, values, np.ones(len(positions), bool),
                         source_indices=np.arange(len(positions)))
    meta = {**features.metadata, "round_id": 1, "combination": "S1T1", "parameter_group": "P0", "feature": "外形",
        "diameter_mm": 10., "teeth": 4, "position_offset_mm": [0., 0., 0.],
        "tool_healthy": True, "no_alarm": True, "complete": True, "alignment_reliable": True}
    process = pd.DataFrame(dict(line=[1, 1], execution_instance=0, segment_mm=[0., 10.], ap_mm=1., ae_mm=2., steady=True))
    plan = FeedPlan(features.frame, meta["program_version"], meta["machine_version"], False, {})
    base = dict(power_w=np.full(len(positions), 80.), valid=np.ones(len(positions), bool),
                status="validated_in_sample_domain", formal_training_release=False)
    return features, plan, record, meta, process, base


def test_phase_lag_uses_actual_nc_projection_and_keeps_unobserved_tail_empty(tmp_path):
    features, plan, record, meta, process, base = _case(tmp_path)
    original = deepcopy(record)
    feed = prepare_feed_samples(features, record, meta)
    with patch("spindle_power.load_prediction.samples.predict_base_from_measured_speed", return_value=base):
        load = prepare_load_samples(process, plan, record, SimpleNamespace(version="synthetic"), meta)
    candidate = feed.frame.feed_actual_mm_min.notna()
    assert candidate.equals(load.frame.feed_input_mm_min.notna())
    np.testing.assert_allclose(load.frame.loc[candidate, "feed_input_mm_min"], feed.frame.loc[candidate, "feed_actual_mm_min"])
    np.testing.assert_allclose(load.frame.loc[candidate, "power_cut_w"], 100 + 2 * load.frame.loc[candidate, "segment_mm"])
    assert not load.frame.loc[load.frame.segment_mm > 9.35, "aligned"].any()
    assert not load.frame.valid_main.any() and not load.frame.valid_residual.any()
    for name in record.values:
        np.testing.assert_array_equal(record.values[name], original.values[name])


def test_row_tail_uses_actual_brackets_after_controller_line_change(tmp_path):
    features, _, record, _, _, _ = _case(tmp_path)
    x = np.array([0., 4., 7., 8.5, 9.7, 10.1, 12., 18.])
    record = Measurement(
        record.source, np.arange(len(x)) * .001,
        dict(program_line_raw=np.array([1,1,1,2,2,2,2,2]), command_x_mm=x,
             command_y_mm=np.zeros(len(x)), command_z_mm=np.zeros(len(x))),
        np.ones(len(x), dtype=bool), source_indices=np.arange(len(x)))
    options = dict(position_offset_mm=[0., 0., 0.], position_policy="command_only_v1", support_provenance=True)
    channels = {"power": 100 + 5*x}
    old, _ = align_spatial_channels(features.frame, record, channels, **options)
    new, audit = align_spatial_channels(features.frame, record, channels, recover_row_tails=True, **options)
    tail = features.frame.segment_mm > 7.
    assert not old.loc[tail, "aligned"].any()
    np.testing.assert_allclose(new.power, 100 + 5*features.frame.segment_mm)
    assert audit["row_tail_recovered_points"] == tail.sum()
    support = audit["support_segments"][-1]
    assert support["row_tail_recovery"] and support["source_end_index_exclusive"] == 6

    record.valid[3] = False
    gapped, audit = align_spatial_channels(features.frame, record, channels, recover_row_tails=True, **options)
    assert audit["row_tail_recovered_points"] == 0
    pd.testing.assert_series_equal(gapped.power, old.power)


def test_base_gap_cuts_supervision_without_splitting_visit_or_prospective_context(tmp_path):
    features, plan, record, meta, process, base = _case(tmp_path)
    with patch("spindle_power.load_prediction.samples.predict_base_from_measured_speed", return_value=base):
        complete = prepare_load_samples(process, plan, record, SimpleNamespace(version="synthetic"), meta)
    base["valid"] = (record.values["actual_x_mm"] < 4.) | (record.values["actual_x_mm"] > 5.)
    with patch("spindle_power.load_prediction.samples.predict_base_from_measured_speed", return_value=base):
        gapped = prepare_load_samples(process, plan, record, SimpleNamespace(version="synthetic"), meta)
    gap = gapped.frame.segment_mm.between(4., 5.)
    assert not gapped.frame.loc[gap, "aligned"].any()
    assert gapped.frame.loc[gapped.frame.segment_mm.between(.5, 3.5), "aligned"].all()
    assert gapped.frame.loc[gapped.frame.segment_mm.between(5.5, 9.), "aligned"].all()
    pd.testing.assert_series_equal(complete.frame.sequence_id, gapped.frame.sequence_id)
    assert not any("访问次数" in message for message in gapped.metadata["issues"])
    assert gapped.frame.loc[gap, "power_cut_w"].isna().all()


def test_command_policy_uses_same_path_labels_for_feed_and_load_without_changing_context(tmp_path):
    features, plan, record, meta, process, base = _case(tmp_path)
    for axis in "xyz":
        record.values[f"command_{axis}_mm"] = record.values[f"actual_{axis}_mm"].copy()
    record.values["command_x_mm"] += .5
    meta["position_policy"] = "command_only_v1"
    feed = prepare_feed_samples(features, record, meta)
    with patch("spindle_power.load_prediction.samples.predict_base_from_measured_speed", return_value=base):
        load = prepare_load_samples(process, plan, record, SimpleNamespace(version="synthetic"), meta)
    present = load.frame.aligned
    assert present.any()
    assert feed.metadata["position_policy"] == load.metadata["position_policy"] == "command_only_v1"
    assert feed.metadata["position_source"] == load.metadata["position_source"] == "command"
    assert load.metadata["position_columns"] == [f"command_{axis}_mm" for axis in "xyz"]
    np.testing.assert_allclose(load.frame.loc[present, "power_cut_w"],
                               100 + 2 * (load.frame.loc[present, "segment_mm"] - .5))
    np.testing.assert_allclose(load.frame.loc[present, "feed_input_mm_min"],
                               feed.frame.loc[present, "feed_actual_mm_min"])
    pd.testing.assert_frame_equal(feed.frame[list(features.frame)], features.frame)
    # Actual positions are diagnostics under this policy and cannot affect labels.
    for axis in "xyz":
        record.values[f"actual_{axis}_mm"][:] = np.nan
    changed = prepare_feed_samples(features, record, meta)
    pd.testing.assert_frame_equal(changed.frame, feed.frame)


def test_command_only_never_falls_back_and_unknown_or_conflicting_policy_is_rejected(tmp_path):
    import pytest
    from spindle_power.load_prediction.measurement import select_position_channels
    features, plan, record, meta, process, base = _case(tmp_path)
    command_meta = {**meta, "position_policy": "command_only_v1"}
    feed = prepare_feed_samples(features, record, command_meta)
    assert feed.metadata["position_source"] is None
    assert not feed.frame.valid_label.any()
    with patch("spindle_power.load_prediction.samples.predict_base_from_measured_speed", return_value=base):
        load = prepare_load_samples(process, plan, record, SimpleNamespace(version="synthetic"), command_meta)
    assert not load.frame.aligned.any()
    with pytest.raises(ValueError, match="未知位置策略"):
        select_position_channels(record.values, position_policy="command_first_typo")
    with pytest.raises(ValueError, match="不一致"):
        prepare_feed_samples(features, record, command_meta, position_policy="actual_preferred_v1")
    with pytest.raises(ValueError, match="不一致"):
        prepare_load_samples(process, plan, record, None, command_meta, position_policy="actual_preferred_v1")


def test_command_invalid_sample_splits_labels_without_falling_back_or_changing_context(tmp_path):
    features, _, record, meta, _, _ = _case(tmp_path)
    for axis in "xyz":
        record.values[f"command_{axis}_mm"] = record.values[f"actual_{axis}_mm"].copy()
    meta["position_policy"] = "command_only_v1"
    before = prepare_feed_samples(features, record, meta)
    record.values["command_x_mm"][record.values["command_x_mm"] > 5.] = np.nan
    after = prepare_feed_samples(features, record, meta)
    assert not after.frame.loc[after.frame.segment_mm > 5., "valid_label"].any()
    assert after.frame.loc[after.frame.segment_mm > 5., "feed_actual_mm_min"].isna().all()
    pd.testing.assert_series_equal(before.frame.sequence_id, after.frame.sequence_id)
    assert after.metadata["position_source"] == "command"


def test_verified_geometry_is_used_once_and_is_not_modified_by_label_preparation(tmp_path):
    import pytest
    from spindle_power.load_prediction.samples import _prepare_process_features
    features, plan, record, meta, process, base = _case(tmp_path)
    prepared = _prepare_process_features(process, plan, meta)
    prepared.metadata["process_mapping_rules_version"] = "verified_geometry_test_v2"
    prepared.frame["ap_mm"] = 3.
    original = deepcopy(prepared)
    with patch("spindle_power.load_prediction.samples.predict_base_from_measured_speed", return_value=base), \
            patch("spindle_power.load_prediction.samples._prepare_process_features", side_effect=AssertionError("must reuse geometry")):
        result = prepare_load_samples(process, plan, record, SimpleNamespace(version="synthetic"), meta,
                                      prepared_geometry=prepared)
    assert result.frame.ap_mm.eq(3.).all()
    assert result.metadata["process_mapping_rules_version"] == "verified_geometry_test_v2"
    pd.testing.assert_frame_equal(prepared.frame, original.frame)
    assert prepared.metadata == original.metadata
    for column, value in (("x_mm", -1.), ("diameter_mm", 7.), ("feed_input_mm_min", 12.), ("power_cut_w", 9.)):
        broken = deepcopy(prepared)
        broken.frame[column] = value
        with patch("spindle_power.load_prediction.samples.predict_base_from_measured_speed", return_value=base), \
                pytest.raises(ValueError, match="预备几何"):
            prepare_load_samples(process, plan, record, SimpleNamespace(version="synthetic"), meta,
                                 prepared_geometry=broken)
