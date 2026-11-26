from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spindle_power.load_prediction.base_power import CalibrationConfig, identify_events
from spindle_power.load_prediction.base_events import _settling, _interpolate_profile, _evaluate
from spindle_power.load_prediction.base_recovery import (RecoveryModel, fit_recovery,
    initialize_recovery_reference, predict_recovery_event)
from spindle_power.load_prediction.recovery_validation import compare_recovery_models
from spindle_power.load_prediction.recovery_validation import _curves
from spindle_power.load_prediction.base_events import extract_events
from spindle_power.load_prediction.greybox_validation import score_curves
from spindle_power.load_prediction.common import version_of
from spindle_power.load_prediction.base_power import BasePowerModel
from spindle_power.load_prediction.common import Cancelled
from spindle_power.load_prediction.measurement import Measurement


def records():
    """Independent sampled dynamics with four levels and run-dependent losses."""
    dt = .02
    t = np.arange(0., 120., dt)
    s, tau = np.zeros(len(t)), np.zeros(len(t))
    for j, level in enumerate((3000., 4000., 5000., 6000.)):
        use = (t >= j * 30) & (t < (j + 1) * 30)
        local = t[use] - j * 30
        s[use] = np.interp(local, [0, 2, 4, 18, 20, 30], [0, 0, level, level, 0, 0])
        tau[use] = .4 + level / 5000.
    u = s / 1000. * np.gradient(s / 1000., t)
    state = np.zeros(len(t))
    for k in range(1, len(t)):
        decay = np.exp(-dt / tau[k])
        state[k] = decay * state[k-1] + (1 - decay) * (16 * max(u[k], 0) - 9 * max(-u[k], 0))
    p = 15 + .02 * s + 20 * np.maximum(u, 0) - 10 * np.maximum(-u, 0) + state
    return [Measurement(f"recovery_{i}", t.copy(), {"spindle_speed_rpm": s.copy(), "spindle_power_w": p + shift * s / 1000},
                        np.ones(len(t), bool)) for i, shift in enumerate((-1., 0., 1.))]


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = records()
        cls.models = {method: fit_recovery(cls.records, method=method) for method in ("phase_warp", "exponential")}

    def test_joint_fit_and_shapes_preserve_peak_and_speed(self):
        for method, model in self.models.items():
            self.assertTrue(model.report["convergence"])
            for initial, target in ((0., 5000.), (5000., 0.)):
                y = predict_recovery_event(target, model, initial_speed_rpm=initial)
                t, p = y["time_s"], y["power_w"]
                profile = _interpolate_profile(BasePowerModel(**model.baseline), initial, target)
                baseline, speed = _evaluate(t, profile)
                early = t <= profile["phase_times_s"][2]
                np.testing.assert_allclose(p[early], baseline[early])
                np.testing.assert_array_equal(y["speed_rpm"], speed)
                tail = p[t >= profile["phase_times_s"][2]]
                self.assertTrue(np.all(np.diff(tail) * (1 if initial == 0 else -1) <= 1e-10))
                features = y["characteristics"]
                peak = np.argmax(p) if initial == 0 else np.argmin(p)
                self.assertEqual(features["extremum_power_w"], p[peak])
                self.assertEqual(features["peak_time_s"], t[peak])
                self.assertEqual(features["settling_time_s"], _settling(t, p, features["steady_power_w"], t[peak], CalibrationConfig()))
                if initial:
                    self.assertLess(p.min(), 0.)
                self.assertEqual(y["status"], "research_candidate_unvalidated")

    def test_future_changes_cannot_change_reference(self):
        r = self.records[0]
        changed = Measurement(r.source, r.time_s.copy(), {k: v.copy() for k, v in r.values.items()}, r.valid.copy())
        for values in changed.values.values():
            values[r.time_s > 16.] = np.nan
        changed.valid[r.time_s > 16.] = False
        for model in self.models.values():
            ref = initialize_recovery_reference(r, model, end_s=16., no_load_confirmed=True)
            again = initialize_recovery_reference(changed, model, end_s=16., no_load_confirmed=True)
            self.assertEqual(asdict(ref), asdict(again))
            self.assertAlmostEqual(ref.idle_reference["end_s"], 16.)
            self.assertTrue(all(e["completed_s"] <= 16. for e in ref.accepted_events))
            result = predict_recovery_event(6000., model, reference=ref, source=r.source, onset_s=92.)
            self.assertTrue(np.isfinite(result["power_w"]).all())

    def test_session_version_future_time_and_confirmation_guards(self):
        model, r = self.models["phase_warp"], self.records[0]
        ref = initialize_recovery_reference(r, model, end_s=16., no_load_confirmed=True)
        for bad, source, onset in ((ref, "wrong", 32.), (ref, r.source, 16.),
                                    (replace(ref, model_version="wrong"), r.source, 32.)):
            with self.subTest(source=source), self.assertRaises(ValueError):
                predict_recovery_event(4000., model, reference=bad, source=source, onset_s=onset)
        with self.assertRaises(ValueError):
            initialize_recovery_reference(r, model, end_s=16.)
        with self.assertRaises(ValueError):
            initialize_recovery_reference(r, model, end_s=3., no_load_confirmed=True)
        with self.assertRaises(ValueError):
            predict_recovery_event(6500., model)
        with self.assertRaises(ValueError):
            predict_recovery_event(6000., model, initial_speed_rpm=4000.)
        with self.assertRaises(ValueError):
            predict_recovery_event(6000., model, sample_period_s=1e-9)

    def test_save_load_and_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            for model in self.models.values():
                path = Path(directory) / "model.json"
                model.save(path)
                loaded = RecoveryModel.load(path)
                np.testing.assert_array_equal(predict_recovery_event(5000., loaded)["power_w"], predict_recovery_event(5000., model)["power_w"])
                payload = json.loads(path.read_text(encoding="utf8"))
                payload["coefficients"]["startup"][0] += .1
                path.write_text(json.dumps(payload), encoding="utf8")
                with self.assertRaisesRegex(ValueError, "版本"):
                    RecoveryModel.load(path)

    def test_cancellation_and_duplicate_records(self):
        with self.assertRaises(Cancelled):
            compare_recovery_models(self.records, cancel=lambda: True)
        count = [0]
        def cancel():
            count[0] += 1
            return count[0] > 12
        with self.assertRaises(Cancelled):
            fit_recovery(self.records, cancel=cancel)
        with self.assertRaisesRegex(ValueError, "同一数据"):
            compare_recovery_models([replace(self.records[0], source=str(i)) for i in range(3)])

    def test_short_tail_does_not_identify_session_time_scale(self):
        r = self.records[0]
        s = r.values["spindle_speed_rpm"]
        drive = s / 1000 * np.gradient(s / 1000, r.time_s)
        power = 15 + .02 * s + 20 * np.maximum(drive, 0) - 10 * np.maximum(-drive, 0)
        short = replace(r, values={"spindle_speed_rpm": s, "spindle_power_w": power})
        for model in self.models.values():
            ref = initialize_recovery_reference(short, model, end_s=16., no_load_confirmed=True)
            self.assertEqual(ref.scales["startup"], 1.)
            self.assertFalse(any(e["kind"] == "startup" for e in ref.accepted_events))
            self.assertTrue(any("0.3" in e["reason"] for e in ref.rejected_events))

    def test_recovery_beyond_record_is_right_censored(self):
        model = replace(self.models["phase_warp"], coefficients={"startup": [12., 0., 0.], "braking": [12., 0., 0.]})
        model.version = version_of(model.payload())
        r, cfg = self.records[0], CalibrationConfig()
        event_list = identify_events(r, cfg)
        obs = extract_events(r, event_list, cfg)[0][:1]
        score = score_curves(r, event_list, obs, _curves(r, event_list, obs, model), cfg)
        self.assertIsNone(score["events"][0]["settling_time_error_s"])
        self.assertFalse(score["passed"])

    def test_file_speed_splits_and_causal_comparison(self):
        result = compare_recovery_models(self.records)
        for variant in result["report"]["variants"].values():
            for fold in variant["folds"]:
                self.assertNotIn(fold["held_out"], fold["model"]["report"]["sources"])
                self.assertEqual(fold["scores"]["blind"]["event_count"], 8)
                self.assertEqual(fold["scores"]["fixed_prior"]["event_count"], 6)
                self.assertEqual(fold["scores"]["rolling_prior"]["coverage_fraction"], 1.)
                for ref in fold["references"]["rolling_prior"]:
                    self.assertNotIn("error", ref)
                    self.assertTrue(all(e["completed_s"] <= ref["end_s"] for e in ref["accepted_events"]))
            for fold in variant["crossed_speed_folds"]:
                self.assertNotIn(fold["held_out"], fold["fit"]["sources"])
                for levels in fold["fit"]["speed_levels"].values():
                    self.assertNotIn(fold["excluded_level_rpm"], levels)
            for fold in variant["unsupported_speed_folds"]:
                self.assertEqual(fold["score"]["coverage_fraction"], 0.)
                self.assertFalse(fold["score"]["passed"])
        self.assertFalse(result["report"]["decision"]["automatic_replacement"])

    def test_native_ui_query_failure_cancel_preserves_active_model(self):
        import tkinter as tk
        import time
        from spindle_power.load_prediction.workbench import LoadPredictionWorkbench
        root = tk.Tk()
        root.withdraw()
        try:
            with tempfile.TemporaryDirectory() as directory:
                panel = LoadPredictionWorkbench(root, directory)
                active = object()
                panel.base = active
                panel.records = self.records
                panel.events = [identify_events(r) for r in self.records]
                bundle = dict(models=self.models, report=dict(variants={}, decision={}, limits=[]))
                def finish():
                    limit = time.monotonic() + 8
                    while panel.busy and time.monotonic() < limit:
                        root.update()
                        time.sleep(.01)
                    self.assertFalse(panel.busy)
                with patch("spindle_power.load_prediction.workbench.compare_recovery_models", return_value=bundle):
                    panel._compare_recovery()
                    finish()
                path = panel.output_dir / "recovery_validation" / "phase_warp.json"
                self.assertTrue(path.exists())
                with patch("spindle_power.load_prediction.workbench.filedialog.askopenfilename", return_value=str(path)):
                    panel._load_recovery()
                options = dict(model="phase_warp", target_speed_rpm=6000., initial_speed_rpm=0., sample_period_s=.01,
                               reference_record_index=0, reference_end_s=16., no_load_confirmed=True, onset_s=92.)
                with patch.object(panel, "_json_editor", side_effect=lambda title, value, accept: accept(options)):
                    panel._query_recovery()
                    finish()
                self.assertIn("research_candidate_unvalidated", panel.report.get("1.0", "end"))
                self.assertEqual(len(panel.axes.lines), 1)
                old = panel.recovery_models
                with patch("spindle_power.load_prediction.workbench.compare_recovery_models", side_effect=ValueError("invalid")):
                    panel._compare_recovery()
                    finish()
                self.assertIs(panel.recovery_models, old)
                self.assertIs(panel.base, active)
                def cancelled(*args, cancel, **kwargs):
                    while not cancel():
                        time.sleep(.005)
                    raise Cancelled("cancelled")
                with patch("spindle_power.load_prediction.workbench.compare_recovery_models", side_effect=cancelled):
                    panel._compare_recovery()
                    panel.cancel()
                    finish()
                self.assertIs(panel.recovery_models, old)
                self.assertIs(panel.base, active)
                panel.destroy()
        finally:
            root.destroy()
