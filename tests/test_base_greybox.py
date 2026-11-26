from __future__ import annotations

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
from spindle_power.load_prediction.base_events import _fit, extract_events
from spindle_power.load_prediction.base_greybox import (GreyBoxConfig, GreyBoxModel, _filter,
    fit_greybox, initialize_greybox_state, predict_greybox, predict_greybox_event)
from spindle_power.load_prediction.greybox_validation import score_curves, summarize, compare_greybox_models
from spindle_power.load_prediction.common import Cancelled, version_of
from spindle_power.load_prediction.measurement import Measurement


def records():
    """Independent recurrence for two physical relaxation modes, plus run offsets."""
    t = np.arange(0., 78., .02)
    s = np.interp(t, [0, 2, 4, 18, 20, 32, 36, 55, 59, 78], [0, 0, 3000, 3000, 0, 0, 6000, 6000, 0, 0])
    u = s / 1000. * np.gradient(s / 1000., t)
    up, down = np.maximum(u, 0), np.maximum(-u, 0)
    components = np.zeros((2, len(t)))
    for j, (tau, gu, gd) in enumerate(((.08, 18., 14.), (1.2, 5., 2.))):
        decay = np.exp(-.02 / tau)
        for k in range(1, len(t)):
            components[j, k] = decay * components[j, k - 1] + (1 - decay) * (gu * up[k] - gd * down[k])
    p = 15. + .02 * s + 3. * up - 2. * down + components.sum(axis=0)
    return [Measurement(f"synthetic_{i}", t.copy(), {"spindle_speed_rpm": s.copy(),
            "spindle_power_w": p + shift * s / 1000.}, np.ones(len(t), bool)) for i, shift in enumerate((-1., 0., 1.))]


def model_for(records):
    cfg = CalibrationConfig()
    base = _fit([extract_events(r, identify_events(r, cfg), cfg) for r in records], cfg)
    model = GreyBoxModel(2, "inertial", [.08, 1.2], [3., 18., 5.], [2., 14., 2.], asdict(base), asdict(GreyBoxConfig()))
    model.version = version_of(model.payload())
    return model


class GreyBoxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = records()
        cls.model = model_for(cls.records)

    def test_filter_matches_analytic_constant_drive_and_decay(self):
        t = np.arange(0., 3., .01)
        np.testing.assert_allclose(_filter(np.full(len(t), 12.), .01, .4, -3.),
                                   12. + (-3. - 12.) * np.exp(-t / .4), atol=1e-12)
        np.testing.assert_allclose(_filter(np.zeros(len(t)), .01, 1.2, 7.), 7 * np.exp(-t / 1.2), atol=1e-12)

    def test_joint_fit_and_second_mode_improve_synthetic_tail(self):
        fits = [fit_greybox(self.records, order=k, fitting=GreyBoxConfig(max_nfev=100)) for k in (1, 2)]
        self.assertTrue(all(m.report["convergence"] for m in fits))
        self.assertLess(np.mean(fits[1].report["training_curve_rmse_w"]), np.mean(fits[0].report["training_curve_rmse_w"]))
        self.assertLess(np.mean(fits[1].report["training_curve_rmse_w"]), 3.)
        offsets = [r["offset_w_per_krpm"] for r in fits[1].report["fitted_run_states"]]
        np.testing.assert_allclose(np.diff(offsets), [1., 1.], atol=.15)
        self.assertEqual(fits[1].status, "research_candidate_unvalidated")

    def test_serialization_and_tamper_detection(self):
        r = self.records[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            self.model.save(path)
            loaded = GreyBoxModel.load(path)
            args = dict(initial_components_w=[0., 0.])
            np.testing.assert_array_equal(predict_greybox(r.time_s, r.values["spindle_speed_rpm"], self.model, **args)["power_w"],
                                          predict_greybox(r.time_s, r.values["spindle_speed_rpm"], loaded, **args)["power_w"])
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["taus_s"][0] *= 2
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "版本"):
                GreyBoxModel.load(path)

    def test_future_samples_cannot_change_prefix_state(self):
        r = self.records[0]
        end = 15.
        state = initialize_greybox_state(r, self.model, end_s=end, no_load_confirmed=True)
        changed = Measurement(r.source, r.time_s.copy(), {k: v.copy() for k, v in r.values.items()}, r.valid.copy())
        changed.values["spindle_power_w"][r.time_s > end] += 100000
        changed.values["spindle_speed_rpm"][r.time_s > end] = np.nan
        changed.valid[r.time_s > end] = False
        again = initialize_greybox_state(changed, self.model, end_s=end, no_load_confirmed=True)
        self.assertEqual(asdict(state), asdict(again))
        self.assertAlmostEqual(state.offset_w_per_krpm, -1., delta=.05)
        mask = r.time_s >= state.end_s
        prediction = predict_greybox(r.time_s[mask], r.values["spindle_speed_rpm"][mask], self.model, state=state, source=r.source)
        self.assertTrue(prediction["valid"].all())
        self.assertEqual(prediction["initial_state_source"], "prior_idle_prefix")

    def test_state_cannot_cross_version_session_or_skip_history(self):
        r = self.records[0]
        state = initialize_greybox_state(r, self.model, end_s=15., no_load_confirmed=True)
        t = np.arange(15., 16., .02)
        for bad, source, times in ((replace(state, model_version="wrong"), r.source, t),
                                   (state, "different_session", t), (state, r.source, t + 1)):
            with self.subTest(source=source, time=times[0]), self.assertRaises(ValueError):
                predict_greybox(times, np.full(len(t), 3000.), self.model, state=bad, source=source)
        with self.assertRaises(ValueError):
            initialize_greybox_state(r, self.model, end_s=15.)
        with self.assertRaises(ValueError):
            initialize_greybox_state(r, self.model, end_s=3., no_load_confirmed=True)

    def test_unknown_initial_state_and_out_of_range_masking(self):
        t = np.arange(0., 10., .02)
        y = predict_greybox(t, np.full(len(t), 3000.), self.model)
        self.assertTrue(np.isnan(y["power_w"][t < 6.]).all())
        self.assertTrue(y["valid"][t >= 6.].all())
        y = predict_greybox(t, np.full(len(t), 9000.), self.model, initial_components_w=[0, 0])
        self.assertFalse(y["valid"].any())
        with self.assertRaises(ValueError):
            predict_greybox(t, np.ones(len(t)), self.model, initial_components_w=[0])

    def test_query_and_right_censoring(self):
        y = predict_greybox_event(6000., self.model)
        self.assertGreater(y["characteristics"]["extremum_power_w"], y["characteristics"]["steady_power_w"])
        self.assertGreater(y["characteristics"]["settling_time_s"], y["characteristics"]["peak_time_s"])
        with self.assertRaises(ValueError):
            predict_greybox_event(7000., self.model)
        with self.assertRaises(ValueError):
            predict_greybox_event(6000., self.model, initial_speed_rpm=3000.)
        r, cfg = self.records[0], CalibrationConfig()
        e = identify_events(r, cfg)
        obs = extract_events(r, e, cfg)[0][:1]
        item = obs[0]
        mask = (r.time_s >= item["onset_s"]) & (r.time_s <= e[item["index"]]["end_s"])
        score = score_curves(r, e, obs, {item["index"]: (np.full(mask.sum(), 1000.), item["steady_w"])}, cfg)
        self.assertIsNone(score["events"][0]["settling_time_error_s"])
        self.assertFalse(score["passed"])
        self.assertEqual(summarize([score])["censored_or_invalid_events"], 1)

    def test_invalid_data_duplicates_and_cancellation(self):
        with self.assertRaises(ValueError):
            fit_greybox([self.records[0], self.records[0]])
        bad = replace(self.records[0], valid=self.records[0].valid.copy())
        bad.valid[100] = False
        with self.assertRaises(ValueError):
            fit_greybox([bad])
        with self.assertRaises(Cancelled):
            compare_greybox_models(self.records, cancel=lambda: True)
        calls = [0]
        def cancel():
            calls[0] += 1
            return calls[0] > 8
        with self.assertRaises(Cancelled):
            fit_greybox(self.records, cancel=cancel)
        duplicate = [replace(self.records[0], source=f"copy_{i}") for i in range(3)]
        with self.assertRaisesRegex(ValueError, "同一数据"):
            compare_greybox_models(duplicate)

    def test_neural_runtime_is_not_loaded_by_greybox(self):
        import subprocess
        result = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); import spindle_power.load_prediction; assert 'torch' not in sys.modules", str(ROOT / "src")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_whole_file_splits_and_conditional_scoring_boundaries(self):
        result = compare_greybox_models(self.records, fitting=GreyBoxConfig(max_nfev=80))
        for variant in result["report"]["variants"].values():
            for fold in variant["folds"]:
                self.assertNotIn(fold["held_out"], fold["fit"]["sources"])
                self.assertEqual(len(fold["fit"]["sources"]), 2)
                self.assertIsNone(fold["state_error"])
                end = fold["prior_state"]["end_s"]
                self.assertTrue(all(e["onset_s"] > end for e in fold["prior_idle_measured_speed"]["events"]))
                self.assertEqual(fold["blind_measured_speed"]["event_count"], 4)
                self.assertEqual(fold["prior_idle_measured_speed"]["event_count"], 2)
        self.assertFalse(result["report"]["decision"]["automatic_replacement"])
        self.assertTrue(all(m.status == "research_candidate_unvalidated" for m in result["models"].values()))

    def test_native_workbench_candidate_query_and_cancel_preserve_current_model(self):
        import tkinter as tk
        import time
        from spindle_power.load_prediction.workbench import LoadPredictionWorkbench
        root = tk.Tk()
        root.withdraw()
        try:
            with tempfile.TemporaryDirectory() as directory:
                panel = LoadPredictionWorkbench(root, directory)
                sentinel = object()
                panel.base = sentinel
                panel.records = self.records
                panel.events = [identify_events(r) for r in self.records]
                report = dict(baseline=dict(summary={}), variants={}, decision={}, limits=[])
                bundle = dict(models={"order2_inertial": self.model}, report=report)
                def finish():
                    limit = time.monotonic() + 5
                    while panel.busy and time.monotonic() < limit:
                        root.update()
                        time.sleep(.01)
                    self.assertFalse(panel.busy)
                with patch("spindle_power.load_prediction.workbench.compare_greybox_models", return_value=bundle):
                    panel._compare_greybox()
                    finish()
                self.assertIs(panel.base, sentinel)
                self.assertIn("order2_inertial", panel.greybox_models)
                path = panel.output_dir / "greybox_validation" / "order2_inertial.json"
                self.assertTrue(path.exists())
                with patch("spindle_power.load_prediction.workbench.filedialog.askopenfilename", return_value=str(path)):
                    panel._load_greybox()
                options = dict(model="order2_inertial", target_speed_rpm=6000., initial_speed_rpm=0., sample_period_s=.01)
                with patch.object(panel, "_json_editor", side_effect=lambda title, value, accept: accept(options)):
                    panel._query_greybox()
                    finish()
                self.assertIn("research_candidate_unvalidated", panel.report.get("1.0", "end"))
                self.assertEqual(len(panel.axes.lines), 2)
                previous = panel.greybox_models
                def cancelled(*args, cancel, **kwargs):
                    while not cancel():
                        time.sleep(.005)
                    raise Cancelled("cancelled")
                with patch("spindle_power.load_prediction.workbench.compare_greybox_models", side_effect=cancelled):
                    panel._compare_greybox()
                    panel.cancel()
                    finish()
                self.assertIs(panel.greybox_models, previous)
                self.assertIs(panel.base, sentinel)
                with patch("spindle_power.load_prediction.workbench.compare_greybox_models", side_effect=ValueError("invalid")):
                    panel._compare_greybox()
                    finish()
                self.assertIs(panel.greybox_models, previous)
                self.assertIn("任务失败", panel.status.get())
                panel.destroy()
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
