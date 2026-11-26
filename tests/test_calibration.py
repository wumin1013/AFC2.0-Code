"""Synthetic calibration only: no collected measurements or formal registry."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from test_condition_quantiles import frame, trained_test_bundle
from spindle_power.load_prediction.calibration import (CalibrationResult, IntervalCalibrator, _fit,
                                                _identifiability, calibrate_model)
from spindle_power.load_prediction.models import predict_cutting_power
from spindle_power.load_prediction.condition import fit_condition_spec
from spindle_power.load_prediction.neural import state_version


class CalibrationTests(unittest.TestCase):
    def test_residual_scale_preserves_existing_condition_bias(self):
        import torch
        records = [{"machine": {"rated_power_w": p}} for p in (1000., 2000.)]
        spec = fit_condition_spec(records)
        bundle = trained_test_bundle(spec)
        branch = spec["branches"]["machine"]
        field = next(f for f in branch["fields"] if f["name"] == "rated_power_w")
        network = bundle.residual.condition_effects.branches["machine"]
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.zero_()
            network[0].weight[0, field["offset"] - branch["offset"]] = 1.
            network[2].weight[1, 0] = .2
        bundle.residual_version = state_version(bundle.residual)
        context = {"condition_descriptors": records[1]}
        data = frame().iloc[:3].copy()
        baseline = predict_cutting_power(bundle, data, context=context)
        condition = torch.tensor(bundle.condition_values(context, len(data))).unsqueeze(0)
        effects = bundle.residual.condition_effects(condition, (1, len(data)), torch.ones(1, len(data), 6))
        bias = effects[0, :, 1].detach().numpy() * bundle.normalization["residual_scale_w"]
        self.assertTrue(np.all(np.abs(bias) > 0))
        calibrated = CalibrationResult(bundle.current_version(), residual_scale=2., residual_bias_w=3.)
        result = predict_cutting_power(bundle, data, context=context, calibration=calibrated)
        expected = (baseline["residual_quantiles_w"] - bias[:, None]) * 2 + bias[:, None] + 3.
        np.testing.assert_allclose(result["residual_quantiles_w"], expected, rtol=1e-6)

    def test_positive_two_stage_parameters_equal_record_weights(self):
        matrix = np.array([[1., 1.], [2., 1.], [4., 1.]])
        target = matrix @ [1.7, .8]
        _identifiability([matrix], 3000)
        first, _ = _fit([matrix], [target], kind="main", delta=1., regularization=0., max_nfev=200)
        np.testing.assert_allclose(first, [1.7, .8], rtol=1e-4)
        # Repeating every point in a long record leaves that record's weight unchanged.
        second, _ = _fit([matrix, np.repeat(matrix, 50, axis=0)], [target, np.repeat(target, 50)],
                         kind="main", delta=1., regularization=0., max_nfev=200)
        np.testing.assert_allclose(first, second, rtol=1e-4)
        with self.assertRaisesRegex(ValueError, "不可辨识"):
            _identifiability([np.ones((10, 2))], 3000)

    def test_calibration_and_inference_leave_networks_unchanged(self):
        import torch
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            bundle = trained_test_bundle()
            version = bundle.current_version()
            records = []
            for number in range(2):
                data = frame().iloc[:3].copy()
                data["feed_input_mm_min"] += number * 70
                baseline = predict_cutting_power(bundle, data)
                data["power_cut_w"] = baseline["main_w"] * 1.1 + baseline["residual_w"]
                data["valid_main"] = True
                data["valid_residual"] = True
                data["feed_source"] = "measured_actual_feed"
                metadata = {"dataset_role": "condition_calibration", "record_id": f"synthetic_{number}",
                            "provenance": "synthetic_test_only",
                            "tool_healthy": True, "no_alarm": True, "complete": True, "alignment_reliable": True}
                records.append({"frame": data, "metadata": metadata})
            result = calibrate_model(bundle, records, regularization=1e-6, huber_delta_w=1.)
            self.assertEqual(bundle.current_version(), version)
            self.assertTrue(result.report["weights_unchanged"])
            self.assertTrue(min(result.ks_scale, result.ke_scale, result.residual_scale) > 0)
            prediction = predict_cutting_power(bundle, frame(), calibration=result)
            self.assertTrue(np.all(np.diff(prediction["cutting_quantiles_w"], axis=1) >= 0))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "calibration.json"
                result.save(path)
                loaded = CalibrationResult.load(path)
                self.assertEqual(loaded.version, result.version)
                loaded.model_version = "wrong"
                with self.assertRaisesRegex(ValueError, "版本"):
                    loaded.validate_for(bundle)
            records[0]["metadata"]["dataset_role"] = "independent_test"
            with self.assertRaisesRegex(ValueError, "独立"):
                calibrate_model(bundle, records)
        finally:
            torch.set_num_threads(previous)

    def test_grouped_interval_calibration_disjointness_and_application(self):
        records = [{"group_id": str(i), "dataset_role": "interval_calibration", "lower_w": [0., 0.],
                    "upper_w": [2., 2.], "target_w": [1., 3.]} for i in range(9)]
        calibrated = IntervalCalibrator.fit(records, model_version="synthetic")
        self.assertEqual(calibrated.correction_w, 1.)
        lo, hi = calibrated.apply([0.], [2.], model_version="synthetic")
        np.testing.assert_array_equal(lo, [-1.])
        np.testing.assert_array_equal(hi, [3.])
        with self.assertRaisesRegex(ValueError, "分组不足"):
            IntervalCalibrator.fit(records[:2], model_version="synthetic")
        with self.assertRaisesRegex(ValueError, "独立"):
            IntervalCalibrator.fit(records, model_version="synthetic", excluded_group_ids=["1"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "interval.json"
            calibrated.save(path)
            self.assertEqual(IntervalCalibrator.load(path).version, calibrated.version)


if __name__ == "__main__":
    unittest.main()
