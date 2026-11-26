"""Audit-assisted power replay: exact labels, no target power, no unsafe fill."""
from copy import deepcopy
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spindle_power.load_prediction import (AuditedEventTimeline, BasePowerModel, CalibrationConfig,
                                     predict_base_from_measured_speed, speed_timeline_fingerprint)
from spindle_power.load_prediction.base_events import FEATURES, PHASE_GRID
from spindle_power.load_prediction.base_timeline import AUDITED_INTERFACE_VERSION, SPEED_ONLY_INTERFACE_VERSION
from spindle_power.load_prediction.common import Cancelled
from spindle_power.load_prediction.common import file_hash
from spindle_power.load_prediction.feed import FeedPlan
from spindle_power.load_prediction.feed_samples import FeedSample
from spindle_power.load_prediction.measurement import Measurement
from spindle_power.load_prediction.samples import prepare_load_samples
from spindle_power.load_prediction.training_service import TrainingService


def fixture():
    profiles = []
    for level, steady in ((3000., 100.), (3020., 200.)):
        for kind in ("startup", "braking"):
            start, end, extreme = (20., steady, steady + 100.) if kind == "startup" else (steady, 20., -50.)
            profile = dict(kind=kind, level_rpm=level, event_count=3,
                steady_w=end, extremum_w=extreme, peak_time_s=.08, settling_time_s=.4, motion_time_s=.3,
                phase_times_s=[0., .08, .14, .4, .9],
                power_w=np.interp(PHASE_GRID, np.arange(5), [start, extreme, end + 20., end, end]).tolist(),
                speed_rpm=np.interp(PHASE_GRID, np.arange(5),
                    [0., level / 3, level / 2, level, level] if kind == "startup" else
                    [level, level * 2 / 3, level / 2, 0., 0.]).tolist())
            profile["observed_ranges"] = {name: [profile[name], profile[name]] for name in FEATURES}
            profiles.append(profile)
    model = BasePowerModel([0., 3000., 3020.], [20., 100., 200.], 0., 0., 1., 1., 1., 1.,
                          asdict(CalibrationConfig()), version="synthetic_exact_labels", schema="afc.base.2",
                          event_profiles=profiles)
    time = np.arange(7000) * .001
    speed = np.interp(time, [0., 1., 1.3, 4., 4.3, 7.], [0., 0., 3020., 3020., 0., 0.])
    events = [dict(start_s=1.5, end_s=3.7, command_time_s=1., speed_rpm=3020.,
                   source="nc_aligned", direction="startup", timing_source="joint_line_speed_review"),
              dict(start_s=4.5, end_s=6.7, command_time_s=4., speed_rpm=0.,
                   source="nc_aligned", direction="braking", timing_source="joint_line_speed_review")]
    timeline = AuditedEventTimeline(events, "synthetic_nc_sha", "synthetic_record",
        speed_timeline_fingerprint(time, speed), initial_speed_rpm=0., initial_state_verified=True,
        record_integrity_verified=True, event_alignment_verified=True)
    return time, speed, model, timeline


def sample_fixture():
    time, speed, model, timeline = fixture()
    # A formally validated MODEL alone must not qualify this new INTERFACE.
    model = replace(model, status="validated_in_sample_domain")
    position = np.clip((time - 2.) / 1.499, 0., 1.)
    line = np.where((time >= 2.) & (time < 3.5), 1., 0.)
    predicted = predict_base_from_measured_speed(time, speed, model, event_timeline=timeline)
    record = Measurement("synthetic_record", time, dict(
        spindle_speed_rpm=speed, spindle_power_w=predicted["power_w"] + 80., program_line_raw=line,
        actual_x_mm=position, actual_y_mm=np.zeros(len(time)), actual_z_mm=np.zeros(len(time)),
        actual_feed_mm_min=np.full(len(time), 40.)), np.ones(len(time), bool))
    grid = np.linspace(0., 1., 21)
    frame = pd.DataFrame(dict(line=1, execution_instance=0, segment_mm=grid, path_mm=grid,
        block_length_mm=1., x_mm=grid, y_mm=0., z_mm=0., motion=1,
        feed_cmd_mm_min=40., spindle_cmd_rpm=3020., curvature_per_mm=0.))
    plan = FeedPlan(frame, timeline.program_version, "synthetic_machine", False, {})
    process = pd.DataFrame(dict(line=[1, 1], execution_instance=[0, 0], segment_mm=[0., 1.],
        ap_mm=[1., 1.], ae_mm=[2., 2.], steady=[True, True]))
    metadata = dict(record_id="synthetic_1", status="active", dataset_role="development",
        round_id=1, combination="S1T1", parameter_group="P0", feature="外形", tool_session="synthetic_session",
        diameter_mm=10., teeth=4, program_version=timeline.program_version, tool_healthy=True,
        no_alarm=True, complete=True, alignment_reliable=True, program_id="synthetic_program",
        feed_override=1., machining_mode="synthetic_mode", machine_settings_version="synthetic_settings")
    return model, timeline, record, plan, process, metadata


class AuditedTimelineTests(unittest.TestCase):
    def setUp(self):
        self.time, self.speed, self.model, self.timeline = fixture()

    def predict(self, timeline=None, **kwargs):
        return predict_base_from_measured_speed(self.time, self.speed, self.model,
            event_timeline=timeline or self.timeline, record_source="synthetic_record", **kwargs)

    def test_full_original_1ms_grid_and_exact_3020_label_without_model_mutation(self):
        before_model, before_events = asdict(self.model), deepcopy(self.timeline.events)
        with patch("spindle_power.load_prediction.base_events.identify_events", side_effect=AssertionError("speed-only used")):
            result = self.predict()
        np.testing.assert_array_equal(result["time_s"], self.time)
        self.assertEqual(len(result["power_w"]), 7000)
        self.assertTrue(result["valid"].all())
        self.assertTrue(np.isfinite(result["power_w"]).all())
        self.assertEqual(result["coverage_fraction"], 1.)
        self.assertEqual(result["events"][0]["target_speed_rpm"], 3020.)
        self.assertEqual(result["power_w"][3000], 200.)
        self.assertEqual(asdict(self.model), before_model)
        self.assertEqual(self.timeline.events, before_events)

    def test_same_definition_nc_dwell_motion_and_explicit_response_timing(self):
        result = self.predict()
        self.assertAlmostEqual(result["events"][0]["motion_time_s"], .3)
        self.assertEqual(result["interface_version"], AUDITED_INTERFACE_VERSION)
        self.assertEqual(result["timing_source"], "observed_response_onset")
        self.assertEqual(result["plateau_timing_source"], "nc_dwell_entry_before_trim")
        self.assertIn("controller_delay_not_identified", result["timing"])
        self.assertFalse(result["formal_training_release"])
        self.assertEqual(result["status"], "calibrated_unvalidated")

    def test_unknown_initial_integrity_or_alignment_and_record_binding_rejected(self):
        for key in ("initial_state_verified", "record_integrity_verified", "event_alignment_verified"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "审核"):
                self.predict(replace(self.timeline, **{key: False}))
        with self.assertRaisesRegex(ValueError, "初始转速"):
            self.predict(replace(self.timeline, initial_speed_rpm=None))
        with self.assertRaisesRegex(ValueError, "来源不同"):
            self.predict(replace(self.timeline, record_source="other_record"))

    def test_target_power_is_not_an_input_and_reference_cannot_bypass_initial_state(self):
        import inspect
        from spindle_power.load_prediction.base_timeline import predict_audited_measured
        self.assertFalse(any("power" in name for name in inspect.signature(predict_audited_measured).parameters))
        with self.assertRaisesRegex(ValueError, "初始动态功率"):
            self.predict(initial_dynamic_w=3.)
        with self.assertRaisesRegex(ValueError, "初态"):
            self.predict(replace(self.timeline, initial_state_verified=False), initial_dynamic_w=0.)

    def test_missing_startup_cannot_be_repaired_by_steady_filling(self):
        with self.assertRaisesRegex(ValueError, "同档|没有审核过"):
            self.predict(replace(self.timeline, events=self.timeline.events[1:]))
        with self.assertRaisesRegex(ValueError, "清单为空"):
            self.predict(replace(self.timeline, events=[]))

    def test_missing_final_event_masks_unregistered_motion(self):
        result = self.predict(replace(self.timeline, events=self.timeline.events[:1]))
        self.assertTrue(result["valid"][self.time < 3.7].all())
        self.assertFalse(result["valid"][self.time > 4.3].any())
        self.assertTrue(np.isnan(result["power_w"][self.time > 4.3]).all())
        self.assertTrue(any("未登记" in warning for warning in result["warnings"]))

    def test_even_a_short_unregistered_ramp_at_the_record_end_is_not_filled(self):
        speed = self.speed.copy()
        speed[self.time >= 6.95] = np.linspace(0., 1000., (self.time >= 6.95).sum())
        timeline = replace(self.timeline, time_speed_sha256=speed_timeline_fingerprint(self.time, speed))
        result = predict_base_from_measured_speed(self.time, speed, self.model, event_timeline=timeline)
        self.assertFalse(result["valid"][-1])
        self.assertTrue(np.isnan(result["power_w"][-1]))

    def test_nonzero_direct_switch_and_unknown_recovery_are_masked(self):
        events = deepcopy(self.timeline.events)
        events[1].update(speed_rpm=3000., direction="startup")
        result = self.predict(replace(self.timeline, events=events))
        self.assertFalse(result["valid"][self.time >= 4.].any())
        self.assertTrue(any("非零档位" in warning for warning in result["warnings"]))
        events = deepcopy(self.timeline.events)
        events[0]["end_s"] = 1.7  # Before the model's .9 s recovery has completed.
        result = self.predict(replace(self.timeline, events=events))
        self.assertFalse(result["valid"][self.time >= 1.].any())
        self.assertTrue(any("右删失" in warning for warning in result["warnings"]))

    def test_target_outside_calibration_and_incomplete_plateau_remain_invalid(self):
        events = deepcopy(self.timeline.events)
        events[0]["speed_rpm"] = 4000.
        result = self.predict(replace(self.timeline, events=events))
        self.assertFalse(result["valid"][self.time >= 1.].any())
        self.assertTrue(any("不外推" in warning for warning in result["warnings"]))
        events = deepcopy(self.timeline.events)
        events[1]["end_s"] = 4.6
        result = self.predict(replace(self.timeline, events=events))
        self.assertFalse(result["valid"][self.time >= 4.].any())
        self.assertTrue(any("观测不足" in warning for warning in result["warnings"]))

    def test_invalid_timeline_semantics_time_and_speed_fingerprint_rejected(self):
        with self.assertRaisesRegex(ValueError, "指纹"):
            self.predict(replace(self.timeline, time_speed_sha256="0" * 64))
        for fields in (dict(timing_source="controller_command"), dict(plateau_timing_source="speed_threshold_platform")):
            with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, "区分"):
                self.predict(replace(self.timeline, **fields))
        events = deepcopy(self.timeline.events)
        events[1]["command_time_s"] = 3.6
        with self.assertRaisesRegex(ValueError, "不重叠"):
            self.predict(replace(self.timeline, events=events))
        speed = self.speed.copy()
        speed[10] = np.nan
        with self.assertRaisesRegex(ValueError, "无效"):
            predict_base_from_measured_speed(self.time, speed, self.model, event_timeline=self.timeline)
        with self.assertRaises(Cancelled):
            self.predict(cancel=lambda: True)

    def test_legacy_speed_only_path_stays_distinct(self):
        result = predict_base_from_measured_speed(self.time, self.speed, self.model)
        self.assertEqual(result["interface_version"], SPEED_ONLY_INTERFACE_VERSION)
        self.assertNotIn("timeline_sha256", result)


class AuditedSampleServiceTests(unittest.TestCase):
    def test_power_subtraction_uses_shared_interface_and_keeps_formal_masks_off(self):
        model, timeline, record, plan, process, metadata = sample_fixture()
        sample = prepare_load_samples(process, plan, record, model, metadata, event_timeline=timeline)
        self.assertTrue(sample.frame.aligned.all())
        np.testing.assert_allclose(sample.frame.power_cut_w, 80.)
        self.assertFalse(sample.frame.valid_main.any())
        self.assertFalse(sample.frame.valid_residual.any())
        self.assertEqual(sample.metadata["base_interface_version"], AUDITED_INTERFACE_VERSION)
        self.assertEqual(sample.metadata["base_event_timeline_sha256"], timeline.fingerprint())
        self.assertFalse(sample.metadata["base_formal_training_release"])
        with self.assertRaisesRegex(ValueError, "程序版本不同"):
            prepare_load_samples(process, plan, record, model, metadata,
                                 event_timeline=replace(timeline, program_version="wrong_nc"))

    def test_service_json_timeline_reaches_shared_prediction_and_is_frozen_in_snapshot(self):
        model, timeline, record, plan, process, metadata = sample_fixture()
        features = FeedSample(plan.frame.copy(), dict(program_version=plan.program_version,
            machine_version=plan.machine_version, spacing_mm=.5, feed_override=1.,
            machining_mode="synthetic_mode", machine_settings_version="synthetic_settings"))
        metadata["resolved_files"] = dict(machine="synthetic_machine.json", nc="synthetic.nc",
                                          process="synthetic_process.csv", measurement="synthetic_record")
        report = dict(feed_eligible=True, load_eligible=True, records=[metadata], snapshot={"synthetic_fixture": True})
        with tempfile.TemporaryDirectory(prefix="afc_audited_timeline_synthetic_") as directory, ExitStack() as mocks:
            model_path = Path(directory) / "synthetic_model.json"
            model.save(model_path)
            service = TrainingService(Path(directory) / "service")
            mocks.enter_context(patch.object(service, "inspect", return_value=report))
            mocks.enter_context(patch("spindle_power.load_prediction.feed.MachineConfig.load", return_value=object()))
            mocks.enter_context(patch("spindle_power.load_prediction.feed_samples.prepare_feed_features", return_value=features))
            mocks.enter_context(patch("spindle_power.load_prediction.feed_samples.prepare_feed_samples", return_value=features))
            mocks.enter_context(patch("spindle_power.load_prediction.samples.read_process_table", return_value=process))
            mocks.enter_context(patch("spindle_power.load_prediction.measurement.read_measurement", return_value=record))
            supplied = {metadata["record_id"]: asdict(timeline)}
            saved = service.prepare("synthetic_fixture_manifest", base_model_path=model_path,
                base_validation={"model_sha256": file_hash(model_path)}, base_event_timelines=supplied)
            supplied[metadata["record_id"]]["events"][0]["speed_rpm"] = 3000.
            _, loads, snapshot = service.load_prepared(saved["snapshot_id"])
            self.assertEqual(loads[0].metadata["base_interface_version"], AUDITED_INTERFACE_VERSION)
            np.testing.assert_allclose(loads[0].frame.power_cut_w, 80.)
            self.assertFalse(loads[0].frame.valid_residual.any())
            frozen = snapshot["source_snapshot"]["base_event_timelines"][metadata["record_id"]]
            self.assertEqual(frozen["events"][0]["speed_rpm"], 3020.)
            self.assertEqual(frozen["time_speed_sha256"], timeline.time_speed_sha256)
            with self.assertRaisesRegex(ValueError, "记录ID"):
                service.prepare("synthetic_fixture_manifest", base_model_path=model_path,
                    base_validation={"model_sha256": file_hash(model_path)},
                    base_event_timelines={"wrong_record": asdict(timeline)})


if __name__ == "__main__":
    unittest.main()
