from __future__ import annotations

import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spindle_power.load_prediction.feed import MachineConfig
from spindle_power.load_prediction.feed_samples import (
    FEED_FEATURE_COLUMNS, FEED_SAMPLE_SCHEMA, FeedSample,
    extract_measured_feed, prepare_feed_features, prepare_feed_samples, program_split,
)
from spindle_power.load_prediction.measurement import Measurement


def machine(initial=(0., 0., 0.)):
    return MachineConfig(initial_position_mm=list(initial), work_offsets_mm={"G54": [0., 0., 0.]},
                         spindle_wait_s=0., optional_stop="skip")


def measured(position=None, *, valid=None):
    position = np.linspace(0., 1., 101) if position is None else np.asarray(position, float)
    count = len(position)
    values = dict(program_line_raw=np.ones(count), actual_x_mm=position + 100.,
                  actual_y_mm=np.full(count, 20.), actual_z_mm=np.zeros(count),
                  actual_vx_mm_min=np.full(count, 600.), actual_vy_mm_min=np.zeros(count),
                  actual_vz_mm_min=np.zeros(count))
    return Measurement("synthetic.csv", np.arange(count) * .001, values,
                       np.ones(count, bool) if valid is None else np.asarray(valid, bool),
                       dict(sample_period_s=.001))


QUALITY = dict(round_id=1, complete=True, alignment_reliable=True, no_alarm=True,
               tool_healthy=True, cutting_contact_confirmed=True)


class FeedFeatureTests(unittest.TestCase):
    def test_workpiece_cutting_geometry_needs_no_fabricated_offsets_or_rapid_time(self):
        code = "G54 G90\nG91 G28 Z0\nG90 G53\nG0 A0 C0\nG05.1 P2 E.0600\nG0 X10 Y20\nG43 Z5 H3\nG1 X12 F600\nG0 X15\nG1 Y22\n"
        path = Path(self.directory.name) / "workpiece.nc"
        path.write_text(code, encoding="utf-8")
        sample = prepare_feed_features(path, MachineConfig(), program_id="workpiece", feed_override=1.,
            machining_mode="recorded_fixed_directives", machine_settings_version="recorded-v1", coordinate_mode="workpiece_cutting")
        self.assertTrue(sample.frame.motion.eq(1).all())
        np.testing.assert_allclose(sample.frame[["x_mm", "y_mm", "z_mm"]].iloc[0], [10,20,5])
        self.assertEqual(sample.frame.sequence_id.nunique(), 2)
        self.assertEqual(sample.metadata["controller_settings"][0]["semantics"], "uninterpreted_execution_setting")
        self.assertNotIn("time_s", sample.frame)
        path.write_text("G54 G90\nG1 X1 F600\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "初始位置"):
            prepare_feed_features(path, MachineConfig(), program_id="workpiece", feed_override=1.,
                machining_mode="recorded_fixed_directives", machine_settings_version="recorded-v1", coordinate_mode="workpiece_cutting")

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def features(self, code="G1 X1 F600\n", config=None, **kwargs):
        path = Path(self.directory.name) / "part.nc"
        path.write_text(code, encoding="utf-8")
        return prepare_feed_features(path, config or machine(), program_id="part-family",
                                     feed_override=1., machining_mode="normal",
                                     machine_settings_version="settings-v1", spacing_mm=.1, **kwargs)

    def test_nc_only_context_and_no_fabricated_command_or_time(self):
        with patch("spindle_power.load_prediction.feed.plan_feed", side_effect=AssertionError("planner called")):
            sample = self.features("G1 X1 F600\nG1 Y1\nM0\nG1 X2\n")
        frame = sample.frame
        self.assertEqual(sample.metadata["schema"], FEED_SAMPLE_SCHEMA)
        self.assertEqual(sample.metadata["feature_columns"], list(FEED_FEATURE_COLUMNS))
        self.assertTrue(np.isfinite(frame[list(FEED_FEATURE_COLUMNS)].to_numpy()).all())
        self.assertTrue(frame.feed_cmd_mm_min.isna().all())
        self.assertNotIn("time_s", frame)
        self.assertNotIn("feed_actual_mm_min", frame)
        self.assertEqual(frame.loc[frame.line == 1, "sequence_id"].iloc[0], frame.loc[frame.line == 2, "sequence_id"].iloc[0])
        self.assertNotEqual(frame.loc[frame.line == 2, "sequence_id"].iloc[0], frame.loc[frame.line == 4, "sequence_id"].iloc[0])
        self.assertAlmostEqual(frame.loc[frame.line == 2, "entry_angle_rad"].iloc[0], np.pi / 2)
        self.assertEqual(frame.path_mm.iloc[-1], 3.)

    def test_arc_geometry_and_modal_features(self):
        sample = self.features("G3 X0 Y1 I-1 J0 F600\nG1 X-1\n", machine((1., 0., 0.)))
        arc = sample.frame.loc[sample.frame.line == 1]
        np.testing.assert_allclose(arc[["x_mm", "y_mm"]].iloc[[0, -1]], [[1, 0], [0, 1]], atol=1e-12)
        self.assertTrue((arc.motion_g3 == 1.).all())
        np.testing.assert_allclose(arc.curvature_per_mm, 1.)
        self.assertAlmostEqual(arc.block_length_mm.iloc[0], np.pi / 2)
        self.assertTrue((sample.frame.feed_program_mm_min == 600.).all())

    def test_unknown_semantics_initial_position_and_budget_rejected(self):
        for code, config, pattern in (("G1 X1 F600\n", MachineConfig(), "初始"),
                                      ("G18 G2 X1 Z1 I1 K0 F600\n", machine(), "语义"),
                                      ("G2 I1 J0 F600\n", machine(), "整圆")):
            with self.subTest(code=code), self.assertRaisesRegex(ValueError, pattern):
                self.features(code, config)
        with self.assertRaisesRegex(ValueError, "max_points"):
            self.features(max_points=2)

    def test_labels_use_actual_channels_and_preserve_nc_context(self):
        features = self.features()
        before = features.frame.copy(deep=True)
        result = prepare_feed_samples(features, measured(), QUALITY)
        self.assertTrue(result.frame.valid_label.all())
        np.testing.assert_allclose(result.frame.feed_actual_mm_min, 600.)
        np.testing.assert_allclose(result.metadata["position_offset_mm"], [100., 20., 0.])
        pd.testing.assert_frame_equal(features.frame, before)
        pd.testing.assert_series_equal(result.frame.sequence_id, features.frame.sequence_id)
        self.assertEqual(result.metadata["measured_feed_source"], "actual_axis_velocity_norm")
        self.assertEqual(result.metadata["position_source"], "actual")
        unknown = prepare_feed_samples(features, measured(), {"round_id": 1})
        self.assertFalse(unknown.frame.valid_label.any())
        with self.assertRaisesRegex(ValueError, "program_id"):
            prepare_feed_samples(features, measured(), {**QUALITY, "program_id": "wrong"})

    def test_invalid_gaps_never_interpolated_across(self):
        features = self.features()
        valid = np.ones(101, bool)
        valid[45:56] = False
        result = prepare_feed_samples(features, measured(valid=valid), {**QUALITY, "position_offset_mm": [100., 20., 0.]})
        gap = np.isclose(result.frame.segment_mm, .5)
        self.assertFalse(result.frame.loc[gap, "valid_label"].any())
        self.assertTrue(result.frame.loc[result.frame.segment_mm < .4, "valid_label"].all())
        self.assertTrue(result.frame.loc[result.frame.segment_mm > .6, "valid_label"].all())
        self.assertTrue(result.frame.loc[gap, "feed_actual_mm_min"].isna().all())

    def test_time_gap_stationary_region_and_channel_nan(self):
        features = self.features()
        record = measured(np.r_[np.linspace(0, .4, 50), np.linspace(.6, 1, 51)])
        record.time_s[50:] += .1
        timed = prepare_feed_samples(features, record, {**QUALITY, "position_offset_mm": [100., 20., 0.]})
        self.assertTrue(timed.frame.valid_label.any())
        self.assertFalse(timed.frame.loc[np.isclose(timed.frame.segment_mm, .5), "valid_label"].any())
        stopped = measured(np.r_[np.linspace(0, .4, 41), np.full(19, .5), np.linspace(.6, 1, 41)])
        result = prepare_feed_samples(features, stopped, {**QUALITY, "position_offset_mm": [100., 20., 0.]})
        self.assertFalse(result.frame.loc[np.isclose(result.frame.segment_mm, .5), "valid_label"].any())
        broken = measured()
        broken.values["actual_x_mm"][40:61] = np.nan
        result = prepare_feed_samples(features, broken, {**QUALITY, "position_offset_mm": [100., 20., 0.]})
        self.assertFalse(result.frame.loc[np.isclose(result.frame.segment_mm, .5), "valid_label"].any())

    def test_repeated_visits_not_assumed_to_be_one_execution(self):
        record = measured(np.r_[np.linspace(0, 1, 51), 0., np.linspace(0, 1, 51)])
        record.values["program_line_raw"][51] = 99.
        result = prepare_feed_samples(self.features(), record, QUALITY)
        self.assertFalse(result.frame.valid_label.any())
        self.assertTrue(any("访问次数" in issue for issue in result.metadata["issues"]))

    def test_incomplete_position_sources_are_not_mixed(self):
        record = measured()
        record.values["command_x_mm"] = record.values.pop("actual_x_mm")
        result = prepare_feed_samples(self.features(), record, QUALITY)
        self.assertFalse(result.frame.valid_label.any())
        self.assertIsNone(result.metadata["position_source"])
        self.assertEqual(result.metadata["position_columns"], [])
        self.assertTrue(any("完整实际/指令XYZ" in issue for issue in result.metadata["issues"]))

    def test_command_position_aligns_actual_velocity_labels(self):
        record = measured()
        for axis in "xyz":
            record.values[f"command_{axis}_mm"] = record.values.pop(f"actual_{axis}_mm")
        record.values["actual_vx_mm_min"][:] = 123.
        result = prepare_feed_samples(self.features(), record, QUALITY)
        self.assertTrue(result.frame.valid_label.all())
        self.assertEqual(result.metadata["position_source"], "command")
        self.assertEqual(result.metadata["position_columns"], [f"command_{axis}_mm" for axis in "xyz"])
        self.assertEqual(result.metadata["measured_feed_source"], "actual_axis_velocity_norm")
        np.testing.assert_allclose(result.frame.feed_actual_mm_min, 123.)
        np.testing.assert_allclose(result.metadata["position_offset_mm"], [100., 20., 0.])

    def test_actual_position_has_priority_over_complete_command_position(self):
        record = measured()
        for axis in "xyz":
            record.values[f"command_{axis}_mm"] = np.full(101, 999.)
        result = prepare_feed_samples(self.features(), record, QUALITY)
        self.assertTrue(result.frame.valid_label.all())
        self.assertEqual(result.metadata["position_source"], "actual")
        self.assertEqual(result.metadata["position_columns"], [f"actual_{axis}_mm" for axis in "xyz"])
        np.testing.assert_allclose(result.metadata["position_offset_mm"], [100., 20., 0.])

    def test_incomplete_actual_position_uses_complete_command_group(self):
        record = measured()
        for axis in "xyz":
            record.values[f"command_{axis}_mm"] = record.values[f"actual_{axis}_mm"].copy()
        record.values.pop("actual_z_mm")
        record.values["actual_x_mm"][:] = 999.
        result = prepare_feed_samples(self.features(), record, QUALITY)
        self.assertTrue(result.frame.valid_label.all())
        self.assertEqual(result.metadata["position_source"], "command")
        np.testing.assert_allclose(result.metadata["position_offset_mm"], [100., 20., 0.])

    def test_actual_nan_does_not_fall_back_to_command_position(self):
        record = measured()
        for axis in "xyz":
            record.values[f"command_{axis}_mm"] = record.values[f"actual_{axis}_mm"].copy()
        record.values["actual_x_mm"][45:56] = np.nan
        result = prepare_feed_samples(self.features(), record,
                                      {**QUALITY, "position_offset_mm": [100., 20., 0.]})
        self.assertEqual(result.metadata["position_source"], "actual")
        gap = np.isclose(result.frame.segment_mm, .5)
        self.assertFalse(result.frame.loc[gap, "valid_label"].any())
        self.assertTrue(result.frame.loc[gap, "feed_actual_mm_min"].isna().all())
        self.assertTrue(result.frame.loc[result.frame.segment_mm < .4, "valid_label"].all())
        self.assertTrue(result.frame.loc[result.frame.segment_mm > .6, "valid_label"].all())

    def test_command_position_never_generates_actual_feed_labels(self):
        record = measured()
        for axis in "xyz":
            record.values[f"command_{axis}_mm"] = record.values.pop(f"actual_{axis}_mm")
            record.values.pop(f"actual_v{axis}_mm_min")
        result = prepare_feed_samples(self.features(), record, QUALITY, allow_position_derivative=True)
        self.assertEqual(result.metadata["measured_feed_source"], "missing")
        self.assertEqual(result.metadata["position_source"], "command")
        self.assertFalse(result.frame.valid_label.any())
        self.assertTrue(result.frame.feed_actual_mm_min.isna().all())

    def test_empty_measurement_and_nondefault_frame_index(self):
        features = self.features()
        features.frame.index = np.arange(len(features.frame)) * 3 + 10
        result = prepare_feed_samples(features, measured(), QUALITY)
        self.assertTrue(result.frame.valid_label.all())
        result = prepare_feed_samples(features, measured([]), QUALITY)
        self.assertFalse(result.frame.valid_label.any())
        self.assertTrue(any("为空" in issue for issue in result.metadata["issues"]))


class ActualFeedSourceTests(unittest.TestCase):
    def test_source_priority_units_and_invalid_channels(self):
        record = measured()
        record.values["actual_vx_mm_min"][:] = 300.
        record.values["actual_vy_mm_min"][:] = 400.
        values, valid, source = extract_measured_feed(record)
        np.testing.assert_allclose(values, 500.)
        self.assertTrue(valid.all())
        record.values["actual_vz_mm_min"][4] = np.nan
        self.assertFalse(extract_measured_feed(record)[1][4])
        record.values["actual_feed_mm_min"] = np.full(101, 123.)
        record.values["actual_feed_mm_min"][3] = -1.
        values, valid, source = extract_measured_feed(record)
        self.assertEqual(source, "actual_feed_mm_min")
        self.assertEqual(values[4], 123.)
        self.assertFalse(valid[3])
        self.assertTrue(np.isnan(values[3]))

    def test_no_command_fallback_and_derivative_needs_explicit_verification(self):
        t = np.array([0., .1, .2, 1., 1.1, 1.2])
        xyz = np.array([0., 1., 2., 100., 101., 102.])
        record = Measurement("positions", t, dict(actual_x_mm=xyz, actual_y_mm=np.zeros(6), actual_z_mm=np.zeros(6)), np.ones(6, bool), dict(sample_period_s=.1))
        values, valid, source = extract_measured_feed(record)
        self.assertEqual(source, "missing")
        self.assertFalse(valid.any())
        values, valid, source = extract_measured_feed(record, allow_position_derivative=True)
        np.testing.assert_allclose(values, 600., atol=1e-10)
        self.assertTrue(valid.all())
        record.values = {name.replace("actual", "command"): value for name, value in record.values.items()}
        self.assertEqual(extract_measured_feed(record, allow_position_derivative=True)[2], "missing")

    def test_invalid_positions_and_short_blocks_do_not_create_derivatives(self):
        record = Measurement("gaps", np.arange(5) * .1,
                             dict(actual_x_mm=np.arange(5.), actual_y_mm=np.zeros(5), actual_z_mm=np.zeros(5)),
                             np.array([True, False, True, False, True]))
        self.assertFalse(extract_measured_feed(record, allow_position_derivative=True)[1].any())


class ProgramSplitTests(unittest.TestCase):
    def sample(self, program, version, round_id=1):
        return FeedSample(pd.DataFrame(), dict(program_id=program, program_version=version, round_id=round_id))

    def test_families_include_parameter_variants_and_repeated_rounds(self):
        samples = [self.sample("outer", "f1500", 1), self.sample("outer", "f2500", 2),
                   self.sample("step", "other", 1)]
        result = program_split(samples, {"outer": "train", "step": "test"})
        self.assertEqual(len(result["train"]), 2)
        self.assertEqual(len(result["test"]), 1)
        self.assertEqual(result["validation"], [])

    def test_same_program_hash_cannot_bypass_groups(self):
        with self.assertRaisesRegex(ValueError, "program_version"):
            program_split([self.sample("a", "same"), self.sample("alias", "same")], {"a": "train", "alias": "test"})
        with self.assertRaisesRegex(ValueError, "划分"):
            program_split([self.sample("a", "one")], {})
        a, b = self.sample("a", "one"), self.sample("b", "two")
        a.metadata["measurement_source"] = b.metadata["measurement_source"] = "same_record.csv"
        with self.assertRaisesRegex(ValueError, "实测来源"):
            program_split([a, b], {"a": "train", "b": "test"})


if __name__ == "__main__":
    unittest.main()
