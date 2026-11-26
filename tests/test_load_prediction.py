from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spindle_power.load_prediction import (BasePowerModel, CalibrationConfig, FeedPlan, MachineConfig, ModelConfig,
    PositionMemory, build_load_models, fit_base_power, grouped_folds, identify_events, inspect_dataset,
    load_models, parse_nc, plan_feed, predict_base_from_commands, predict_base_from_measured_speed, predict_base_event,
    predict_cutting_power, prepare_load_samples, read_measurement)
from spindle_power.load_prediction import IdleReference, calibrate_idle_reference, assess_repeatability
from spindle_power.load_prediction.common import Cancelled
from spindle_power.load_prediction.measurement import Measurement
from spindle_power.load_prediction.samples import PHYSICS_COLUMNS, SampleSet, physical_power, physical_quantities, prepare_forward_features
from spindle_power.load_prediction.training import train_two_stage


def machine():
    return MachineConfig(name="Synthetic test only", max_velocity_mm_min=[3000.] * 3,
        max_acceleration_mm_s2=[500.] * 3, max_jerk_mm_s3=[5000.] * 3,
        interpolation_period_s=.01, contour_tolerance_mm=.05, corner_tolerance_mm=.05,
        initial_position_mm=[0., 0., 0.], travel_min_mm=[-1000.] * 3, travel_max_mm=[1000.] * 3,
        work_offsets_mm={"G54": [0., 0., 0.]}, provenance="test", spindle_wait_s=0.,
        optional_stop="skip", home_position_mm=[0., 0., 0.])


def base_model():
    return BasePowerModel([0., 1000., 2000.], [20., 80., 150.], .05, .035, .12, .15, 5000., 4000.,
                          vars(CalibrationConfig()), version="test", status="validated_in_sample_domain")


def synthetic_idle():
    time = np.arange(0, 22., .01)
    speed = np.interp(time, [0, 2, 3, 6, 7, 10, 12, 15, 17, 22], [0, 0, 1000, 1000, 0, 0, 2000, 2000, 0, 0])
    power = predict_base_from_measured_speed(time, speed, base_model())["power_w"]
    return Measurement("synthetic", time, {"spindle_speed_rpm": speed, "spindle_power_w": power}, np.ones(len(time), bool))


class ScopeReaderTests(unittest.TestCase):
    def test_semantic_order_units_and_invalid_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scope.csv"
            content = "\n".join(["<Scope>", "<SmplInfo>,<0>,<1000>",
                "<ChannelInfo>,<0>,<0>,<G寄存器>,<0>,<X轴>,<432>,<10>",
                "<ChannelInfo>,<1>,<0>,<实际速度>,<0>,<SP轴>,<0>,<360>",
                "<ChannelData>,-200,2993,", "<ChannelData>,oops,3000,", "<ChannelData>,201,", "<ChannelData>,250,3010,"])
            path.write_bytes(content.encode("gb18030"))
            before = path.read_bytes()
            record = read_measurement(path)
            np.testing.assert_allclose(record.time_s, [0, .001, .002, .003])
            self.assertEqual(record.values["spindle_power_w"][0], -200)
            self.assertEqual(record.values["spindle_speed_rpm"][0], 2993)
            self.assertEqual(record.valid.tolist(), [True, False, False, True])
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(inspect_dataset(directory, deep=True)["files"][0]["invalid_samples"], 2)

    def test_unknown_sampling_and_duplicate_channels_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scope.csv"
            header = "<SmplInfo>,<0>,<500>\n<ChannelInfo>,<0>,<0>,<实际速度>,<0>,<SP轴>\n"
            path.write_text(header + "<ChannelData>,2000,\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "采样"):
                read_measurement(path)
            self.assertEqual(len(read_measurement(path, sample_period_s=.002).time_s), 1)
            path.write_text(header.replace("500", "1000") + header.splitlines()[1] + "\n<ChannelData>,2,2,\n", encoding="utf-8")
            record = read_measurement(path)
            self.assertEqual(record.metadata["duplicate_conflicts"], {"spindle_speed_rpm": 0})
            self.assertIn("spindle_speed_rpm__channel1", record.values)
            path.write_text(path.read_text(encoding="utf-8").replace("<ChannelData>,2,2", "<ChannelData>,2,3"), encoding="utf-8")
            self.assertFalse(read_measurement(path).valid[0])


class BasePowerTests(unittest.TestCase):
    def test_event_identification_and_validation(self):
        records = [synthetic_idle(), synthetic_idle(), synthetic_idle()]
        model = fit_base_power(records, config=CalibrationConfig(max_nfev=100))
        self.assertEqual(model.schema, "afc.base.2")
        self.assertEqual(len(model.event_profiles), 4)
        self.assertEqual(len(model.report["folds"]), 3)
        self.assertTrue(model.report["validation_passed"])

    def test_prediction_and_serialization(self):
        record = synthetic_idle()
        model = base_model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.json"
            model.save(path)
            restored = BasePowerModel.load(path)
            a = predict_base_from_measured_speed(record.time_s, record.values["spindle_speed_rpm"], restored)
            np.testing.assert_allclose(a["power_w"], record.values["spindle_power_w"])
            self.assertLess(a["power_w"].min(), 0)
        commands = np.where(record.time_s < 1, 0., 1000.)
        result = predict_base_from_commands(record.time_s, commands, model)
        self.assertLess(result["speed_rpm"][101], 1000.)
        self.assertEqual(result["command_source"], "estimated")

    def test_unknown_state_invalid_time_and_out_of_range(self):
        model = base_model()
        with self.assertRaises(ValueError):
            predict_base_from_measured_speed([0, .01, .04], [0, 0, 0], model)
        result = predict_base_from_measured_speed(np.arange(20) * .01, np.ones(20) * 2500, model)
        self.assertTrue(np.isnan(result["power_w"]).all())
        time = np.arange(100)*.01
        unknown = predict_base_from_measured_speed(time, time*1000, model)
        self.assertFalse(unknown["valid"].any())
        known = predict_base_from_measured_speed(time, time*1000, model, initial_dynamic_w=0.)
        self.assertTrue(known["valid"].all())
        with self.assertRaises(Cancelled):
            fit_base_power([synthetic_idle(), synthetic_idle()], cancel=lambda: True)

    def test_peak_failure_blocks_model_acceptance(self):
        records = [synthetic_idle(), synthetic_idle()]
        for record in records:
            record.values["spindle_power_w"][1199] += 1500
        model = fit_base_power(records, config=CalibrationConfig(max_nfev=80))
        self.assertFalse(model.report["peak_validation_passed"])
        self.assertEqual(model.status, "calibrated_unvalidated")

    def test_event_query_interpolation_braking_and_roundtrip(self):
        model = fit_base_power([synthetic_idle(), synthetic_idle()])
        query = predict_base_event(1500., model)
        c = query["characteristics"]
        self.assertAlmostEqual(c["steady_power_w"], 115., delta=.1)
        self.assertGreater(c["peak_power_w"], c["steady_power_w"])
        self.assertGreater(c["settling_time_s"], c["peak_time_s"])
        self.assertAlmostEqual(c["peak_time_s"], query["time_s"][np.argmax(query["power_w"])], delta=.002)
        band = max(5., .05 * abs(c["steady_power_w"]))
        selected = (query["time_s"] >= c["settling_time_s"]) & (query["time_s"] < c["settling_time_s"] + .5)
        self.assertTrue(np.all(np.abs(query["power_w"][selected] - c["steady_power_w"]) <= band + 1e-6))
        brake = predict_base_event(0., model, initial_speed_rpm=1500.)
        self.assertLess(brake["characteristics"]["trough_power_w"], 0)
        self.assertNotIn("peak_power_w", brake["characteristics"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "event.json"
            model.save(path)
            loaded = BasePowerModel.load(path)
            np.testing.assert_allclose(predict_base_event(1500., loaded)["power_w"], query["power_w"])
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["event_profiles"][0]["phase_times_s"][1] = -1
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "波形"):
                BasePowerModel.load(path)

    def test_event_domain_unknown_initial_state_and_incomplete_ramp(self):
        model = fit_base_power([synthetic_idle(), synthetic_idle()])
        for target, origin in ((3000, 0), (1500, 1000), (0, 0), (np.nan, 0), (-1, 0)):
            with self.subTest(target=target, origin=origin), self.assertRaises(ValueError):
                predict_base_event(target, model, initial_speed_rpm=origin)
        with self.assertRaisesRegex(ValueError, "旧版"):
            predict_base_event(1000, base_model())
        t = np.arange(0, 10, .01)
        r = predict_base_from_measured_speed(t, np.full(len(t), 1000.), model)
        self.assertFalse(r["valid"][0])
        self.assertTrue(r["valid"][-1])
        r = predict_base_from_measured_speed(t, np.full(len(t), 1000.), model, initial_dynamic_w=0.)
        self.assertTrue(r["valid"].all())
        short = predict_base_from_measured_speed(np.arange(100)*.01, np.arange(100)*10., model)
        self.assertFalse(short["valid"][5:].any())
        with self.assertRaises(ValueError):
            predict_base_from_measured_speed(t, np.zeros(len(t)), model, initial_dynamic_w=5.)

    def test_event_command_timeline_and_interruption(self):
        model = fit_base_power([synthetic_idle(), synthetic_idle()])
        t = np.arange(0, 12., .01)
        command = np.where(t < 2, 0., np.where(t < 7, 1000., 0.))
        r = predict_base_from_commands(t, command, model, source="nc_aligned")
        self.assertTrue(r["valid"].all())
        self.assertLess(r["power_w"].min(), 0)
        self.assertIn("assumed_zero_controller_delay", r["timing"])
        interrupted = np.where(t < 2, 0., np.where(t < 2.1, 1000., 0.))
        r = predict_base_from_commands(t, interrupted, model)
        self.assertFalse(r["valid"][t >= 2.1].any())
        self.assertTrue(any("尚未稳定" in w for w in r["warnings"]))
        with self.assertRaises(Cancelled):
            predict_base_from_commands(t, command, model, cancel=lambda: True)

    def test_event_leaveout_really_excludes_heldout_power(self):
        records = [synthetic_idle(), synthetic_idle(), synthetic_idle()]
        records[0].values["spindle_power_w"] *= 1.5
        model = fit_base_power(records)
        other = fit_base_power(records[1:])
        event = next(e for e in model.report["folds"][0]["events"] if e["kind"] == "startup" and e["level_rpm"] == 2000)
        self.assertAlmostEqual(event["predicted"]["extremum_w"], predict_base_event(2000, other)["characteristics"]["peak_power_w"])
        self.assertFalse(model.report["validation_passed"])

    def test_settling_uses_final_power_and_reports_censoring(self):
        from spindle_power.load_prediction.base_events import _settling, extract_events
        t = np.arange(0, 5, .01)
        y = np.where(t < 2, 130., 100.)
        self.assertAlmostEqual(_settling(t, y, 100., .5, CalibrationConfig()), 2.)
        self.assertIsNone(_settling(t[:220], y[:220], 100., .5, CalibrationConfig()))
        record = synthetic_idle()
        record.values["spindle_power_w"][1200:1400] = 500.
        record.values["spindle_power_w"][1400:1500] = np.linspace(500, 160, 100)
        observations, excluded, _ = extract_events(record, identify_events(record), CalibrationConfig())
        self.assertTrue(any("右删失" in e["reason"] for e in excluded))


class IdleReferenceTests(unittest.TestCase):
    def setUp(self):
        self.model = fit_base_power([synthetic_idle(), synthetic_idle()])
        self.record = synthetic_idle()
        self.record.values["spindle_power_w"] += 4 * self.record.values["spindle_speed_rpm"] / 1000.

    def reference(self, **kwargs):
        return calibrate_idle_reference(self.record, self.model, start_s=4.5, end_s=5.5,
                                        no_load_confirmed=True, **kwargs)

    def test_known_baseline_bias_and_model_is_unchanged(self):
        before = asdict(self.model)
        reference = self.reference()
        self.assertAlmostEqual(reference.offset_w_per_krpm, 4., delta=.01)
        original = predict_base_event(2000, self.model)
        corrected = predict_base_event(2000, self.model, reference=reference)
        self.assertAlmostEqual(corrected["characteristics"]["steady_power_w"] - original["characteristics"]["steady_power_w"], 8., delta=.02)
        self.assertAlmostEqual(corrected["power_w"][0], original["power_w"][0])
        self.assertAlmostEqual(corrected["characteristics"]["settling_time_s"], original["characteristics"]["settling_time_s"])
        self.assertEqual(corrected["status"], "reference_adjusted_candidate")
        self.assertEqual(corrected["validation_components"], {})
        self.assertEqual(asdict(self.model), before)
        brake = predict_base_event(0, self.model, initial_speed_rpm=2000, reference=reference)
        self.assertLess(brake["characteristics"]["trough_power_w"], 0)

    def test_reference_does_not_read_later_measurements(self):
        reference = self.reference()
        future = self.record.time_s > 5.5
        for values in self.record.values.values():
            values[future] = np.nan
        self.record.valid[future] = False
        self.assertEqual(asdict(self.reference()), asdict(reference))

    def test_unconfirmed_invalid_unstable_and_outside_reference_rejected(self):
        with self.assertRaisesRegex(ValueError, "明确确认"):
            calibrate_idle_reference(self.record, self.model, start_s=4.5, end_s=5.5)
        with self.assertRaisesRegex(ValueError, "稳速|范围"):
            calibrate_idle_reference(self.record, self.model, start_s=2, end_s=3, no_load_confirmed=True)
        self.record.values["spindle_power_w"] += 200
        with self.assertRaisesRegex(ValueError, "20%"):
            self.reference()
        with self.assertRaises(Cancelled):
            self.reference(cancel=lambda: True)
        self.record.values["spindle_power_w"] -= 200
        self.record.valid[(self.record.time_s > 5) & (self.record.time_s < 5.1)] = False
        with self.assertRaisesRegex(ValueError, "缺失"):
            self.reference()

    def test_serialization_version_and_tampered_coefficient(self):
        reference = self.reference()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            reference.save(path)
            loaded = IdleReference.load(path)
            loaded.validate(self.model)
            self.assertEqual(asdict(reference), asdict(loaded))
        with self.assertRaisesRegex(ValueError, "版本"):
            predict_base_event(2000, self.model, reference=replace(reference, model_version="different"))
        with self.assertRaisesRegex(ValueError, "偏移"):
            replace(reference, offset_w_per_krpm=123).validate(self.model)

    def test_repeatability_lower_bound_counts_files_not_windows(self):
        def e(t):
            return dict(kind="startup", level_rpm=1000., settling_time_s=t)
        self.model.report["event_observations"] = [[e(6.), e(6.)], [e(8.)], [e(7.)]]
        report = assess_repeatability(self.model)[0]
        self.assertEqual(report["independent_files"], 3)
        self.assertEqual(report["best_possible_worst_error_s"], 1.)
        self.assertTrue(report["exceeds_current_tolerance"])
        self.model.report["event_observations"] = [[e(7.)]]
        self.assertIsNone(assess_repeatability(self.model)[0]["best_possible_worst_error_s"])
        self.model.report["event_observations"] = [[e(6.)], [e(7.)]]
        self.assertFalse(assess_repeatability(self.model)[0]["exceeds_current_tolerance"])

    def bound_reference(self):
        return self.reference(session=dict(machine_id="M1", session_id="run1", time_origin_s=0.))

    def test_session_identity_and_future_reference_are_required(self):
        from spindle_power.load_prediction.base_reference import reference_use_mask
        reference = self.bound_reference()
        target = dict(machine_id="M1", session_id="run1", time_origin_s=30.)
        for session in (None, {**target, "machine_id": "M2"}, {**target, "session_id": "run2"},
                        {**target, "time_origin_s": None}, {**target, "time_origin_s": True},
                        {**target, "time_origin_s": float("nan")}, {**target, "time_origin_s": 5.5}):
            with self.subTest(session=session), self.assertRaises(ValueError):
                reference_use_mask([0., 1.], self.model, reference, session=session, record_source="target.csv")
        with self.assertRaisesRegex(ValueError, "原点"):
            reference_use_mask([0., 10.], self.model, reference, session=target, record_source=self.record.source)
        self.assertTrue(reference_use_mask([0., 1.], self.model, reference, session=target, record_source="target.csv").all())
        with self.assertRaisesRegex(ValueError, "时间原点"):
            reference_use_mask([0., 1.], self.model, self.reference(), session=target, record_source="target.csv")

    def test_measured_reference_masks_prefix_and_retains_signed_power(self):
        from spindle_power.load_prediction.base_reference import apply_idle_reference
        reference = self.bound_reference()
        before = asdict(self.model)
        original = predict_base_from_measured_speed(self.record.time_s, self.record.values["spindle_speed_rpm"], self.model)
        result = predict_base_from_measured_speed(self.record.time_s, self.record.values["spindle_speed_rpm"], self.model,
                    reference=reference, session=reference.session, record_source=self.record.source)
        prior = self.record.time_s <= reference.end_s
        self.assertFalse(result["valid"][prior].any())
        self.assertTrue(np.isnan(result["power_w"][prior]).all())
        future_steady = (self.record.time_s >= 14) & (self.record.time_s < 15)
        np.testing.assert_allclose((result["power_w"] - original["power_w"])[future_steady], 8., atol=.02)
        self.assertLess(np.nanmin(result["power_w"]), 0.)
        stopped = result["valid"] & (self.record.values["spindle_speed_rpm"] == 0)
        np.testing.assert_allclose(result["power_w"][stopped], original["power_w"][stopped])
        self.assertEqual(result["status"], "reference_adjusted_candidate")
        self.assertEqual(asdict(self.model), before)
        with self.assertRaisesRegex(ValueError, "重复"):
            apply_idle_reference(result, self.model, reference, session=reference.session, record_source=self.record.source)

    def test_commands_reference_and_bound_event_respect_timeline(self):
        reference = self.bound_reference()
        target = dict(machine_id="M1", session_id="run1", time_origin_s=30.)
        time = np.arange(0, 12., .01)
        command = np.where(time < 2, 0., np.where(time < 7, 1000., 0.))
        plain = predict_base_from_commands(time, command, self.model)
        result = predict_base_from_commands(time, command, self.model, reference=reference, session=target,
                                            record_source="nc:test")
        self.assertTrue(result["valid"].all())
        np.testing.assert_allclose((result["power_w"] - plain["power_w"])[(time > 5) & (time < 6)], 4., atol=.01)
        with self.assertRaisesRegex(ValueError, "onset_s"):
            predict_base_event(2000, self.model, reference=reference)
        with self.assertRaisesRegex(ValueError, "未来"):
            predict_base_event(2000, self.model, reference=reference, session=reference.session,
                               record_source=self.record.source, onset_s=4.)
        event = predict_base_event(2000, self.model, reference=reference, session=target,
                                   record_source="event", onset_s=0.)
        self.assertEqual(event["reference_usage"]["session"], target)
        with self.assertRaises(Cancelled):
            predict_base_from_commands(time, command, self.model, reference=reference, session=target,
                                       record_source="nc:test", cancel=lambda: True)

    def test_saved_context_and_reference_window_integrity(self):
        from spindle_power.load_prediction.base_reference import verify_reference_record
        reference = self.bound_reference()
        verify_reference_record(self.record, reference)
        self.record.values["spindle_power_w"][self.record.time_s > 6.] += 500.
        verify_reference_record(self.record, reference)
        self.assertEqual(asdict(self.bound_reference()), asdict(reference))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bound.json"
            reference.save(path)
            loaded = IdleReference.load(path)
            self.assertEqual(loaded.session, reference.session)
            legacy = asdict(reference)
            legacy.pop("session")
            path.write_text(json.dumps(legacy), encoding="utf-8")
            self.assertIsNone(IdleReference.load(path).session)
        self.record.values["spindle_power_w"][500] += 1.
        with self.assertRaisesRegex(ValueError, "窗口已变化"):
            verify_reference_record(self.record, reference)

    def test_candidate_labels_record_correction_and_never_unlock_training(self):
        reference = self.bound_reference()
        model = replace(self.model, status="validated_in_sample_domain")
        target = dict(machine_id="M1", session_id="run1", time_origin_s=30.)
        time = np.arange(0, 6., .01)
        position = np.linspace(0, 1, len(time))
        frame = pd.DataFrame(dict(line=1, execution_instance=0, segment_mm=position, block_length_mm=1.,
                                  x_mm=position, y_mm=0., z_mm=0., motion=1, feed_cmd_mm_min=600.,
                                  spindle_cmd_rpm=1000., curvature_per_mm=0.))
        plan = FeedPlan(frame, "program", "machine", True, {})
        process = pd.DataFrame(dict(line=[1, 1], execution_instance=[0, 0], segment_mm=[0., 1.],
                                   ap_mm=[1., 1.], ae_mm=[2., 2.], steady=[True, True]))
        values = dict(program_line_raw=np.ones(len(time)), spindle_speed_rpm=np.ones(len(time))*1000,
                      spindle_power_w=np.ones(len(time))*50, command_x_mm=position,
                      command_y_mm=np.zeros(len(time)), command_z_mm=np.zeros(len(time)))
        record = Measurement("target.csv", time, values, np.ones(len(time), bool), source_indices=np.arange(len(time)))
        metadata = dict(round_id=1, combination="S1T1", parameter_group="P0", feature="外形", tool_session="a",
                        diameter_mm=10., teeth=4, program_version="program", tool_healthy=True, no_alarm=True,
                        complete=True, alignment_reliable=True, position_offset_mm=[0., 0., 0.])
        sample = prepare_load_samples(process, plan, record, model, metadata, reference=reference, session=target)
        self.assertTrue(sample.frame.aligned.any())
        self.assertFalse(sample.frame.valid_main.any())
        self.assertFalse(sample.frame.valid_residual.any())
        self.assertLess(sample.frame.power_cut_w.dropna().max(), 0.)
        np.testing.assert_allclose(sample.frame.base_reference_correction_w.dropna(), 4., atol=.01)
        self.assertEqual(sample.metadata["base_reference"]["target_source"], "target.csv")
        self.assertNotEqual(sample.metadata["base_prediction_version"], model.version)


class FeedTests(unittest.TestCase):
    def run_plan(self, code, config=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.nc"
            path.write_text(code, encoding="utf-8")
            return plan_feed(path, config or machine())

    def test_continuous_short_segments_and_bounds(self):
        plan = self.run_plan("G21 G17 G90 G94\nG1 X10. F1800.\nX20.\nX30.\nM30\n")
        frame = plan.frame
        internal = frame.loc[(frame.path_mm > 5) & (frame.path_mm < 25)]
        self.assertGreater(internal.feed_cmd_mm_min.min(), 1.)
        self.assertLessEqual(frame.feed_cmd_mm_min.max(), 1800.01)
        for axis in "xyz":
            self.assertLessEqual(frame[f"v{axis}_mm_s"].abs().max(), 50.001)
            self.assertLessEqual(frame[f"a{axis}_mm_s2"].abs().max(), 500.001)
            self.assertLessEqual(frame[f"j{axis}_mm_s3"].abs().max(), 5000.001)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            plan.save(path)
            pd.testing.assert_frame_equal(plan.frame, FeedPlan.load(path).frame, check_dtype=False)

    def test_arcs_helices_corners_and_incremental(self):
        config = machine()
        config.initial_position_mm = [10., 0., 0.]
        plan = self.run_plan("G17 G90 G94\nG3 X0 Y10 Z2 I-10 J0 F600\nG91 G1 X-5\nY-5\n", config)
        np.testing.assert_allclose(plan.frame[["x_mm", "y_mm", "z_mm"]].iloc[-1], [-5, 5, 2], atol=1e-7)
        self.assertGreater(plan.frame.curvature_per_mm.max(), .05)
        self.assertTrue(all(g["contour_deviation_mm"] <= .05 for g in plan.report["groups"]))

    def test_missing_parameters_unknown_directive_and_travel(self):
        with self.assertRaisesRegex(ValueError, "max_velocity"):
            self.run_plan("G1 X1 F100\n", MachineConfig())
        with self.assertRaisesRegex(ValueError, "厂家"):
            self.run_plan("G05.1 P2 E.06\nG1 X1 F100\n")
        with self.assertRaisesRegex(ValueError, "行程"):
            self.run_plan("G1 X1100 F100\n")
        with self.assertRaisesRegex(ValueError, "G 代码"):
            self.run_plan("G18 G2 X1 Z1 I1 K0 F100\n")

    def test_macros_tool_compensation_and_stop_timing(self):
        config = machine()
        config.tool_lengths_mm = {"3": 5.}
        config.optional_stop = "unknown"
        config.fixed_rotary_degrees = {"A": 0., "C": 0.}
        plan = self.run_plan("#53620=3\nG54 G90\nG0 A0 C0\nG43 H#53620 Z10\nM01\nG1 X1 F100\n", config)
        self.assertFalse(plan.timing_complete)
        self.assertAlmostEqual(plan.frame.z_mm.iloc[-1], 15.)


class SampleMemoryTests(unittest.TestCase):
    def test_constant_off_command_spindle_is_not_a_steady_physical_label(self):
        process, plan, record, metadata = self._load_alignment_case()
        record.values["spindle_speed_rpm"][:] = 600.
        sample = prepare_load_samples(process, plan, record, base_model(), metadata)
        self.assertFalse(sample.frame.spindle_stable.any())
        self.assertFalse(sample.frame.valid_main.any())

    def test_load_prediction_boundaries_do_not_read_measurement_quality(self):
        process, plan, record, metadata = self._load_alignment_case()
        good = prepare_load_samples(process, plan, record, base_model(), metadata)
        # Changing only supervision quality must not change prospective context.
        unknown = {**metadata, "no_alarm": None}
        unknown.pop("tool_session")
        candidate = prepare_load_samples(process, plan, record, base_model(), unknown)
        pd.testing.assert_series_equal(good.frame.sequence_id, candidate.frame.sequence_id)
        self.assertTrue(good.frame.valid_residual.all())
        self.assertFalse(candidate.frame.valid_residual.any())
        self.assertGreater(candidate.frame.training_sequence_id.nunique(), good.frame.training_sequence_id.nunique())

    def _load_alignment_case(self, *, quality=True, actual_feed=True):
        count = 101
        position = np.linspace(0, 1, count)
        frame = pd.DataFrame(dict(line=1, execution_instance=0, segment_mm=position, path_mm=position,
                                 block_length_mm=1., x_mm=position, y_mm=0., z_mm=0., motion=1,
                                 feed_cmd_mm_min=600., spindle_cmd_rpm=1000., curvature_per_mm=0.))
        plan = FeedPlan(frame, "program", "machine", True, {})
        process = pd.DataFrame(dict(line=[1, 1], execution_instance=[0, 0], segment_mm=[0., 1.],
                                   ap_mm=[1., 1.], ae_mm=[2., 2.], steady=[True, True]))
        values = dict(program_line_raw=np.ones(count), spindle_speed_rpm=np.ones(count)*1000,
                      spindle_power_w=np.ones(count)*180, command_x_mm=position + 100,
                      command_y_mm=np.ones(count)*20, command_z_mm=np.zeros(count))
        record = Measurement("early-format", np.arange(count)*.001, values, np.ones(count, bool), source_indices=np.arange(count))
        metadata = dict(round_id=2, combination="S3T3", parameter_group="P0", feature="外形", tool_session="a",
                        diameter_mm=10., teeth=4, program_version="program")
        if quality:
            metadata.update(tool_healthy=True, no_alarm=True, complete=True, alignment_reliable=True)
        if actual_feed:
            values.update(actual_vx_mm_min=np.full(count, 420.), actual_vy_mm_min=np.zeros(count),
                          actual_vz_mm_min=np.zeros(count))
        return process, plan, record, metadata

    def test_alignment_quality_unknown_and_command_position_format(self):
        process, plan, record, metadata = self._load_alignment_case(quality=False, actual_feed=False)
        forward = prepare_forward_features(process, plan, metadata)
        self.assertNotIn("power_cut_w", forward.frame)
        self.assertTrue(forward.frame.feed_input_mm_min.isna().all())
        self.assertEqual(set(forward.frame.feed_source), {"missing"})
        sample = prepare_load_samples(process, plan, record, base_model(), metadata)
        self.assertTrue(sample.frame.aligned.all())
        self.assertEqual(sample.metadata["position_source"], "command")
        self.assertFalse(sample.frame.valid_main.any())
        metadata.update(tool_healthy=True, no_alarm=True, complete=True, alignment_reliable=True)
        sample = prepare_load_samples(process, plan, record, base_model(), metadata)
        self.assertFalse(sample.frame.valid_main.any())
        # Command positions and old planned feed do not supply actual-feed labels.
        count = len(record.time_s)
        record.values.update(actual_vx_mm_min=np.full(count, 420.), actual_vy_mm_min=np.zeros(count),
                             actual_vz_mm_min=np.zeros(count))
        sample = prepare_load_samples(process, plan, record, base_model(), metadata)
        # This 101-sample alignment fixture has no complete 203-sample rate support.
        self.assertFalse(sample.frame.spindle_rate_support_valid.any())
        self.assertFalse(sample.frame.valid_main.any())
        self.assertTrue(sample.frame.valid_residual.all())
        self.assertEqual(sample.metadata["position_source"], "command")
        self.assertEqual(sample.metadata["feed_label_source"], "actual_axis_velocity_norm")
        np.testing.assert_allclose(sample.frame.feed_input_mm_min, 420.)
        np.testing.assert_allclose(sample.frame.power_cut_w, 100., atol=1e-8)
        record.values["program_line_raw"] = np.ones(count)*2
        self.assertFalse(prepare_load_samples(process, plan, record, base_model(), metadata).frame.aligned.any())

    def test_actual_position_takes_priority_without_nan_fallback(self):
        process, plan, record, metadata = self._load_alignment_case()
        for axis in "xyz":
            record.values[f"actual_{axis}_mm"] = record.values[f"command_{axis}_mm"].copy()
        # A distorted command path must not displace complete actual positions.
        record.values["command_y_mm"] += np.linspace(0., 2., len(record.time_s))
        sample = prepare_load_samples(process, plan, record, base_model(), metadata)
        self.assertEqual(sample.metadata["position_source"], "actual")
        self.assertFalse(sample.frame.spindle_rate_support_valid.any())
        self.assertFalse(sample.frame.valid_main.any())
        self.assertTrue(sample.frame.valid_residual.all())
        # Even valid command coordinates cannot repair a gap in selected actual XYZ.
        record.values["command_y_mm"] = record.values["actual_y_mm"].copy()
        record.values["actual_x_mm"][50] = np.nan
        sample = prepare_load_samples(process, plan, record, base_model(),
                                      {**metadata, "position_offset_mm": [100., 20., 0.]})
        self.assertEqual(sample.metadata["position_source"], "actual")
        self.assertFalse(sample.frame.aligned.iloc[50])
        self.assertTrue(sample.frame.aligned.iloc[:49].all())
        self.assertTrue(sample.frame.aligned.iloc[52:].all())
        self.assertFalse(any("访问次数" in issue for issue in sample.metadata["issues"]))

    def test_incomplete_actual_uses_complete_command_but_no_complete_position_is_invalid(self):
        process, plan, record, metadata = self._load_alignment_case()
        record.values["actual_x_mm"] = record.values["command_x_mm"].copy()
        sample = prepare_load_samples(process, plan, record, base_model(), metadata)
        self.assertEqual(sample.metadata["position_source"], "command")
        self.assertFalse(sample.frame.spindle_rate_support_valid.any())
        self.assertFalse(sample.frame.valid_main.any())
        self.assertTrue(sample.frame.valid_residual.all())
        del record.values["command_z_mm"]
        sample = prepare_load_samples(process, plan, record, base_model(), metadata)
        self.assertIsNone(sample.metadata["position_source"])
        self.assertFalse(sample.frame.aligned.any())
        self.assertFalse(sample.frame.valid_main.any())
        self.assertFalse(sample.frame.valid_residual.any())
        self.assertTrue(any("缺少完整" in issue for issue in sample.metadata["issues"]))

    def test_command_position_never_supplies_actual_feed_derivative(self):
        process, plan, record, metadata = self._load_alignment_case(actual_feed=False)
        record.values["actual_vx_mm_min"] = np.full(len(record.time_s), 420.)
        metadata["position_derivative_verified"] = True
        sample = prepare_load_samples(process, plan, record, base_model(), metadata)
        self.assertEqual(sample.metadata["position_source"], "command")
        self.assertEqual(sample.metadata["feed_label_source"], "missing")
        self.assertTrue(sample.frame.aligned.all())
        self.assertTrue(sample.frame.feed_input_mm_min.isna().all())
        self.assertFalse(sample.frame.valid_main.any())
        self.assertFalse(sample.frame.valid_residual.any())

    def test_command_position_preserves_geometry_quality_and_base_gates(self):
        process, plan, record, metadata = self._load_alignment_case()
        for field in ("tool_healthy", "no_alarm", "complete", "alignment_reliable"):
            with self.subTest(quality=field):
                sample = prepare_load_samples(process, plan, record, base_model(), {**metadata, field: False})
                self.assertFalse(sample.frame.valid_main.any())
                self.assertFalse(sample.frame.valid_residual.any())
        candidate = replace(base_model(), status="calibrated_unvalidated")
        sample = prepare_load_samples(process, plan, record, candidate, metadata)
        self.assertFalse(sample.frame.valid_main.any())
        self.assertFalse(sample.frame.valid_residual.any())
        self.assertTrue(any("完整验证" in issue for issue in sample.metadata["issues"]))
        # An explicit machine-to-NC origin prevents re-estimating away this error.
        record.values["command_y_mm"] += 1.
        sample = prepare_load_samples(process, plan, record, base_model(),
                                      {**metadata, "position_offset_mm": [100., 20., 0.]})
        self.assertEqual(sample.metadata["position_source"], "command")
        self.assertFalse(sample.frame.aligned.any())
        self.assertFalse(sample.frame.valid_main.any())
        self.assertFalse(sample.frame.valid_residual.any())

    def test_physical_units_contact_and_geometry(self):
        physics = np.array([[2., 5., 2000., 8000., 10., 4.]])
        q = physical_quantities(physics)
        self.assertAlmostEqual(q["mrr_mm3_s"][0], 1000 / 3)
        self.assertAlmostEqual(q["equivalent_edge_mm"][0], 2.)
        expected = 1000 / 3 + 10 * (np.pi * 10 * 8000 / 60000) * 2
        self.assertAlmostEqual(physical_power(physics, 1000, 10)[0], expected)
        physics[0, 0] = 0
        self.assertEqual(physical_power(physics, 1000, 10)[0], 0)

    def test_folds_never_split_rounds(self):
        from spindle_power.load_prediction.experiment_protocol import round_spec
        metadata = [round_spec(r + 1) for r in range(90) for _ in range(5)]
        folds = grouped_folds(metadata)
        for fold in folds:
            self.assertEqual(len(fold["train_rounds"]), 80)
            self.assertEqual(len(fold["validation_rounds"]), 10)
            self.assertFalse(set(fold["train_rounds"]) & set(fold["validation_rounds"]))
        with self.assertRaisesRegex(ValueError, "完整"):
            grouped_folds(metadata[:100])

    def test_memory_chronology_gating_and_context(self):
        memory = PositionMemory()
        context = dict(program_version="p", parameter_group="P0", tool_session="t", main_version="m", residual_version="r")
        frame = pd.DataFrame(dict(line=[1, 1], segment_mm=[.01, .02], execution_instance=[0, 0]))
        self.assertEqual(memory.predict(context, 1, 0, 1), 0)
        for round_id in (1, 2):
            memory.update_round(context, frame, [20., 20.], round_id, healthy=True, no_alarm=True, complete=True, aligned=True)
        self.assertEqual(memory.predict(context, 1, 0, 3), 20)
        with self.assertRaises(ValueError):
            memory.predict(context, 1, 0, 2)
        self.assertEqual(memory.predict({**context, "tool_session": "new"}, 1, 0, 3), 0)
        self.assertEqual(memory.update_round(context, frame, [1, 1], 3, healthy=None, no_alarm=True, complete=True, aligned=True)["updated"], 0)

    def test_training_dry_run_does_not_import_torch(self):
        sample = SampleSet(pd.DataFrame({"valid_main": [False], "valid_residual": [True]}), dict(round_id=2))
        result = train_two_stage([sample])
        self.assertEqual(result["action"], "dry_run")
        self.assertEqual(result["main_samples"], 0)


class WorkbenchTests(unittest.TestCase):
    def test_reference_controls_clear_stale_samples_and_keep_old_files_readable(self):
        import tkinter as tk
        from spindle_power.load_prediction.workbench import LoadPredictionWorkbench
        root = tk.Tk()
        root.withdraw()
        try:
            temporary_output = tempfile.TemporaryDirectory()
            self.addCleanup(temporary_output.cleanup)
            panel = LoadPredictionWorkbench(root, temporary_output.name)
            model = fit_base_power([synthetic_idle(), synthetic_idle()])
            reference = calibrate_idle_reference(synthetic_idle(), model, start_s=4.5, end_s=5.5, no_load_confirmed=True,
                                                 session=dict(machine_id="M1", session_id="run1", time_origin_s=0.))
            panel.base = panel.base_candidate = model
            panel.samples = panel.last_prediction = "old"
            panel._set_idle_reference(reference)
            self.assertTrue(panel.use_idle_reference.get())
            self.assertIsNone(panel.samples)
            self.assertIsNone(panel.last_prediction)
            accepted = []
            panel._with_idle_reference(model, reference.source, lambda ref, session: accepted.append((ref, session)))
            self.assertEqual(accepted, [(reference, reference.session)])
            panel._with_idle_reference(model, "next.csv", lambda ref, session: accepted.append((ref, session)))
            dialog = next(widget for widget in panel.winfo_children() if isinstance(widget, tk.Toplevel))
            from tkinter import ttk
            entries = sorted((widget for widget in dialog.winfo_children() if isinstance(widget, ttk.Entry)),
                             key=lambda widget: int(widget.grid_info()["row"]))
            entries[-1].insert(0, "30")
            next(widget for widget in dialog.winfo_children() if isinstance(widget, ttk.Checkbutton)).invoke()
            with patch("spindle_power.load_prediction.workbench.messagebox.showerror") as error:
                next(widget for widget in dialog.winfo_children() if isinstance(widget, ttk.Button)).invoke()
                error.assert_not_called()
            self.assertEqual(accepted[-1][1]["time_origin_s"], 30.)
            panel._set_idle_reference(replace(reference, session=None))
            self.assertFalse(panel.use_idle_reference.get())
            self.assertIn("缺少会话", panel.reference_status.get())
            panel._clear_idle_reference()
            panel._with_idle_reference(model, "other", lambda ref, session: accepted.append((ref, session)))
            self.assertEqual(accepted[-1], (None, None))
            panel.destroy()
        finally:
            root.destroy()

    def test_background_completion_and_cancellation_keep_last_result(self):
        import tkinter as tk
        import time
        from spindle_power.load_prediction.workbench import LoadPredictionWorkbench
        root = tk.Tk()
        root.withdraw()
        try:
            temporary_output = tempfile.TemporaryDirectory()
            self.addCleanup(temporary_output.cleanup)
            panel = LoadPredictionWorkbench(root, temporary_output.name)
            completed = []
            panel.start("test", lambda cancel: "result", completed.append)
            for _ in range(100):
                root.update()
                if not panel.busy:
                    break
                time.sleep(.01)
            self.assertEqual(completed, ["result"])
            panel.start("cancel", lambda cancel: (time.sleep(.05), "discarded")[1], completed.append)
            panel.cancel()
            for _ in range(100):
                root.update()
                if not panel.busy:
                    break
                time.sleep(.01)
            self.assertEqual(completed, ["result"])
            self.assertEqual(len(panel.tabs.tabs()), 4)
            panel.destroy()
        finally:
            root.destroy()


try:
    from spindle_power.load_prediction.models import _runtime
    torch = _runtime().torch
    TORCH = True
except RuntimeError:
    TORCH = False


@unittest.skipUnless(TORCH, "optional torch dependency not installed")
class NeuralTests(unittest.TestCase):
    def test_main_physics_gradients_and_signed_masked_residual(self):
        bundle = build_load_models(ModelConfig(hidden=8, attention_heads=2, attention_layers=1, coefficient_widths=(8,)))
        physics = torch.tensor([[2., 5., 2000., 8000., 10., 4.], [0, 0, 0, 0, 10, 4]], requires_grad=True)
        power = bundle.main(physics)
        self.assertGreater(power[0].item(), 0)
        self.assertEqual(power[1].item(), 0)
        power.sum().backward()
        self.assertTrue(torch.isfinite(physics.grad).all())
        for p in bundle.residual.parameters():
            p.data.zero_()
        bundle.residual.output.bias.data.fill_(-2)
        result = bundle.residual(torch.ones(1, 5, 6), torch.tensor([[True, True, True, False, False]]))
        self.assertEqual(result.tolist(), [[-2, -2, -2, 0, 0]])

    def test_untrained_load_save_and_prediction_refusal(self):
        bundle = build_load_models(ModelConfig(hidden=8, attention_heads=2, attention_layers=1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "untrained.pt"
            bundle.save(path)
            loaded = load_models(path)
            self.assertFalse(loaded.main_trained)
            with self.assertRaisesRegex(ValueError, "未训练"):
                predict_cutting_power(loaded, pd.DataFrame())
        from spindle_power.load_prediction.neural import freeze
        freeze(bundle.main)
        self.assertTrue(all(not p.requires_grad for p in bundle.main.parameters()))


if __name__ == "__main__":
    unittest.main()
