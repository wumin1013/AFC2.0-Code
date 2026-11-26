"""Synthetic checks of physical conditions, ordered quantiles and model packages."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from spindle_power.load_prediction.condition import fit_condition_spec, encode_conditions, condition_coverage
from spindle_power.load_prediction.metrics import quantile_loss, regression_metrics, aggregate_metrics
from spindle_power.load_prediction.models import ModelConfig, build_load_models, load_models, predict_cutting_power
from spindle_power.load_prediction.neural import freeze, state_version, residual_windows


def trained_test_bundle(condition_spec=None):
    torch.manual_seed(731)
    bundle = build_load_models(ModelConfig(hidden=8, attention_heads=2, attention_layers=1,
                                          coefficient_widths=(8,), condition_spec=condition_spec))
    freeze(bundle.main)
    freeze(bundle.residual)
    bundle.main_trained = bundle.residual_trained = True
    bundle.main_version = state_version(bundle.main)
    bundle.residual_version = state_version(bundle.residual)
    bundle.residual_main_version = bundle.main_version
    bundle.normalization = {"main_mean_w": 100., "main_scale_w": 50., "residual_scale_w": 10.}
    bundle.metadata = {"provenance": "synthetic_test_only", "training_status": "synthetic_test"}
    return bundle


def frame():
    return pd.DataFrame({"ap_mm": [1., 2., 1., 0.], "ae_mm": [2., 4., 3., 0.],
                         "feed_input_mm_min": [1000., 1400., 1800., 0.], "spindle_rpm": 8000.,
                         "diameter_mm": 10., "teeth": 4., "curvature_per_mm": [0., .1, .2, 0.],
                         "block_length_mm": [10., 15., 20., 5.], "sequence_id": [0, 0, 1, 2],
                         "execution_instance": 0, "line": [1, 2, 3, 4], "segment_mm": 0.,
                         "feed_source": "bilstm_actual_feed", "feed_model_version": "synthetic_feed"})


class ConditionTests(unittest.TestCase):
    def test_physical_fields_missing_masks_and_no_identity_features(self):
        records = [{"machine": {"rated_power_w": 1000., "category": "mill"}, "material": {"hardness": 100.}},
                   {"machine": {"rated_power_w": 3000., "category": "mill"}, "workpiece": {"hardness": 200.}}]
        spec = fit_condition_spec(records)
        values = encode_conditions(records + [{}], spec)
        field = next(f for f in spec["branches"]["machine"]["fields"] if f["name"] == "rated_power_w")
        np.testing.assert_allclose(values[:, field["offset"]], [-1, 1, 0])
        np.testing.assert_array_equal(values[:, field["offset"] + 1], [0, 0, 1])
        self.assertEqual(set(spec["branches"]), {"machine", "tool", "workpiece", "clamping", "cooling"})
        with self.assertRaisesRegex(ValueError, "ID"):
            fit_condition_spec([{"machine": {"machine_id": "M1"}}])
        with self.assertRaisesRegex(ValueError, "类别"):
            encode_conditions([{"machine": {"category": "lathe"}}], spec)
        changed = deepcopy(spec)
        changed["branches"]["machine"]["active"] = False
        with self.assertRaisesRegex(ValueError, "版本"):
            encode_conditions(records, changed)

    def test_single_level_freeze_reference_one_and_condition_gradients(self):
        records = [{"machine": {"rated_power_w": power}, "tool": {"material": "carbide"}} for power in (1000., 2000.)]
        spec = fit_condition_spec(records)
        bundle = build_load_models(ModelConfig(hidden=8, attention_heads=2, attention_layers=1, condition_spec=spec))
        physics = torch.tensor([[2., 4., 1200., 8000., 10., 4.]])
        reference = torch.tensor(encode_conditions(records[:1], spec))
        parts = bundle.main.components(physics, reference)
        self.assertEqual(parts["ks_multiplier"].item(), 1.)
        self.assertEqual(parts["ke_multiplier"].item(), 1.)
        self.assertTrue(all(not p.requires_grad for p in bundle.main.condition_effects.branches["tool"].parameters()))
        self.assertFalse(condition_coverage(spec)["tool"]["active"])
        other = torch.tensor(encode_conditions(records[1:], spec))
        bundle.main(physics, other).sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in bundle.main.condition_effects.branches["machine"].parameters()))
        effects = bundle.residual.condition_effects(reference, (1, 3), torch.ones(1, 3, 6))
        self.assertTrue(torch.equal(effects, torch.zeros_like(effects)))


class QuantileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_ordered_quantiles_mask_gradients_and_signed_median(self):
        bundle = build_load_models(ModelConfig(hidden=8, attention_heads=2, attention_layers=1))
        mask = torch.tensor([[True, True, False]])
        values = bundle.residual.forward_quantiles(torch.randn(1, 3, 6), valid=mask)
        self.assertTrue((torch.diff(values[mask], dim=-1) > 0).all())
        self.assertTrue(torch.equal(values[~mask], torch.zeros_like(values[~mask])))
        loss = quantile_loss(values, torch.tensor([[1., -2., float("nan")]]), valid=mask)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None for p in bundle.residual.spread.parameters()))
        for parameter in bundle.residual.parameters():
            parameter.data.zero_()
        bundle.residual.output.bias.data.fill_(-2.)
        self.assertEqual(bundle.residual(torch.ones(1, 3, 6), valid=mask).tolist(), [[-2., -2., 0.]])

    def test_direct_intervals_no_calibrator_and_safe_round_trip(self):
        bundle = trained_test_bundle()
        original = predict_cutting_power(bundle, frame())
        self.assertEqual(original["interval_kind"], "direct_neural_quantiles")
        self.assertTrue(np.all(np.diff(original["cutting_quantiles_w"], axis=1) >= 0))
        np.testing.assert_array_equal(original["cutting_quantiles_w"][-1], [0., 0., 0.])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.pt"
            bundle.save(path)
            loaded = load_models(path)
            np.testing.assert_allclose(predict_cutting_power(loaded, frame())["cutting_quantiles_w"], original["cutting_quantiles_w"])
            payload = torch.load(path, weights_only=True)
            payload["normalization"]["main_scale_w"] *= 2
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "版本"):
                load_models(path)

    def test_reappearing_sequence_ids_do_not_join_windows(self):
        data = pd.DataFrame({"sequence_id": [1, 1, 2, 1, 1]})
        self.assertEqual([i.tolist() for i in residual_windows(data, 9)], [[0, 1], [2], [3, 4]])

    def test_metrics_and_equal_record_category_round_aggregation(self):
        result = regression_metrics([1., 2., 10.], [2., 2., 8.], lower=[0., 1., 7.], upper=[3., 4., 9.],
                                    scale_w=10., high_threshold_w=9., low_threshold_w=1.5)
        self.assertEqual(result["peak_error_w"], 2.)
        self.assertEqual(result["high_load_underestimate_rate"], 1.)
        self.assertEqual(result["low_load_overestimate_rate"], 1.)
        self.assertEqual(result["high_upper_exceed_rate"], 1.)
        self.assertAlmostEqual(result["coverage"], 2 / 3)
        rows = [{"round_id": 1, "feature": "a", "mae_w": 2.}, {"round_id": 1, "feature": "a", "mae_w": 2.},
                {"round_id": 1, "feature": "b", "mae_w": 4.}, {"round_id": 2, "feature": "a", "mae_w": 9.}]
        self.assertEqual(aggregate_metrics(rows)["mae_w"], 6.)


if __name__ == "__main__":
    unittest.main()
