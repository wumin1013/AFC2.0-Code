"""Synthetic tests for actual ablation behavior and serialized model contracts."""
from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from spindle_power.load_prediction.condition import encode_conditions, fit_condition_spec
from spindle_power.load_prediction.common import json_value
from spindle_power.load_prediction.models import ModelConfig, build_load_models, load_models, predict_cutting_power
from spindle_power.load_prediction.samples import PHYSICS_COLUMNS, SampleSet
from spindle_power.load_prediction.training import TrainingConfig, _residual_loss, train_two_stage


DESCRIPTORS = [{"machine": {"rated_power_w": value}} for value in (1000., 2000.)]


def small_config(**changes):
    return ModelConfig(hidden=8, attention_heads=2, attention_layers=1,
                       coefficient_widths=(4,), dilations=(1,), window_points=8, **changes)


def samples():
    result = []
    for index, descriptors in enumerate(DESCRIPTORS):
        frame = pd.DataFrame({
            "ap_mm": 1., "ae_mm": 2., "feed_input_mm_min": np.linspace(500., 1200., 8),
            "spindle_rpm": np.linspace(4000., 6000., 8), "diameter_mm": 10., "teeth": 4.,
            "curvature_per_mm": np.linspace(0., .1, 8), "block_length_mm": 4.,
            "sequence_id": 0, "execution_instance": 0, "line": 1,
            "segment_mm": np.arange(8) * .5, "path_mm": np.arange(8) * .5,
            "feed_source": "measured_actual_feed", "valid_main": True, "valid_residual": True,
            "power_cut_w": np.linspace(50., 100., 8) + index * 3.,
        })
        result.append(SampleSet(frame, {"round_id": index + 1, "feature": "straight",
            "condition_descriptors": descriptors, "provenance": "synthetic_test_only"}))
    return result


def train(config):
    return train_two_stage(samples(), model_config=config,
        config=TrainingConfig(main_epochs=1, residual_epochs=1, main_batch_points=8, seed=17), execute=True)


class AblationModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.spec = fit_condition_spec(DESCRIPTORS)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_ablation_configuration_rejects_unknown_modes_and_nonboolean_switches(self):
        for changes in ({"physical_mode": "global"}, {"residual_output": "normal"},
                        {"use_condition_effects": 1}, {"use_residual_modulation": "false"},
                        {"schema": "afc.load.2"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                small_config(**changes).validate()

    def test_fixed_coefficients_are_constant_across_feed_speed_and_only_bias_trains(self):
        bundle = build_load_models(small_config(physical_mode="fixed", use_condition_effects=False))
        physics = torch.tensor(samples()[0].frame[PHYSICS_COLUMNS].to_numpy(), dtype=torch.float32)
        main = bundle.main
        self.assertEqual([name for name, value in main.named_parameters() if value.requires_grad],
                         ["coefficients.bias"])
        self.assertTrue(torch.equal(main.coefficients.weight, torch.zeros_like(main.coefficients.weight)))
        optimizer = torch.optim.SGD([main.coefficients.bias], lr=1e-5)
        before = main.coefficients.bias.detach().clone()
        main(physics).sum().backward()
        self.assertTrue(torch.all(main.coefficients.bias.grad.abs() > 0))
        self.assertIsNone(main.coefficients.weight.grad)
        optimizer.step()
        self.assertFalse(torch.equal(before, main.coefficients.bias))
        parts = main.components(physics)
        for name in ("ks", "ke"):
            self.assertTrue(torch.equal(parts[name], parts[name][0].expand_as(parts[name])))
        self.assertGreater(torch.unique(parts["power_w"]).numel(), 1)

    def test_local_coefficients_can_change_with_feed_and_speed(self):
        bundle = build_load_models(small_config(physical_mode="local", use_condition_effects=False))
        with torch.no_grad():
            for layer in bundle.main.coefficients:
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.fill_(.2)
                    layer.bias.fill_(.1)
        physics = torch.tensor([[1., 2., 500., 4000., 10., 4.],
                                [1., 2., 1000., 4000., 10., 4.],
                                [1., 2., 1000., 8000., 10., 4.]])
        parts = bundle.main.components(physics)
        for name in ("ks", "ke"):
            self.assertEqual(torch.unique(parts[name]).numel(), 3)
            self.assertTrue(torch.all(parts[name] > 0))

    def test_disabled_condition_effects_are_exact_unit_multipliers(self):
        bundle = build_load_models(small_config(condition_spec=self.spec, use_condition_effects=False))
        physics = torch.tensor([[1., 2., 1000., 5000., 10., 4.]]).repeat(2, 1)
        condition = torch.tensor(encode_conditions(DESCRIPTORS, self.spec))
        parts = bundle.main.components(physics, condition)
        for name in ("ks_multiplier", "ke_multiplier"):
            self.assertTrue(torch.equal(parts[name], torch.ones(2)))
        self.assertTrue(torch.equal(parts["power_w"][:1], parts["power_w"][1:]))
        self.assertEqual(len(bundle.main.condition_effects.branches), 0)

    def test_disabled_residual_modulation_ignores_conditions_in_network_and_public_prediction(self):
        config = small_config(condition_spec=self.spec, use_condition_effects=False,
                              use_residual_modulation=False)
        bundle = train(config)
        x = torch.randn(1, 5, 6)
        encoded = torch.tensor(encode_conditions(DESCRIPTORS, self.spec))
        with torch.no_grad():
            reference = bundle.residual.forward_quantiles(x, condition=encoded[:1])
            other = bundle.residual.forward_quantiles(x, condition=encoded[1:])
        self.assertTrue(torch.equal(reference, other))
        frame = samples()[0].frame
        first = predict_cutting_power(bundle, frame, feed_mode="measured", condition=DESCRIPTORS[0])
        second = predict_cutting_power(bundle, frame, feed_mode="measured", condition=DESCRIPTORS[1])
        np.testing.assert_array_equal(first["cutting_quantiles_w"], second["cutting_quantiles_w"])

    def test_point_output_preserves_negative_values_and_has_no_trainable_spread(self):
        bundle = build_load_models(small_config(residual_output="point"))
        with torch.no_grad():
            bundle.residual.output.weight.zero_()
            bundle.residual.output.bias.fill_(-2.)
        mask = torch.tensor([[True, True, False]])
        actual = bundle.residual.forward_quantiles(torch.ones(1, 3, 6), valid=mask)
        np.testing.assert_array_equal(actual.detach().numpy(), [[[-2., -2., -2.], [-2., -2., -2.], [0., 0., 0.]]])
        actual.sum().backward()
        self.assertTrue(all(not value.requires_grad and value.grad is None for value in bundle.residual.spread.parameters()))
        self.assertIsNotNone(bundle.residual.output.bias.grad)

    def test_residual_objective_and_gradients_differ_for_point_and_quantiles(self):
        class FixedResidual(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.values = torch.nn.Parameter(torch.tensor([[[-2., 0., 3.]]]))

            def forward_quantiles(self, features, condition=None):
                return self.values

        prepared = [(SampleSet(pd.DataFrame(), {"round_id": 1, "feature": "straight"}),
                     [(torch.zeros(1, 1, 6), torch.tensor([[1.]]), None)])]
        for output, expected, gradient in (("point", 1., [0., -2., 0.]),
                                           ("quantiles", .25, [-.05 / 3, -.5 / 3, .05 / 3])):
            with self.subTest(output=output):
                residual = FixedResidual()
                bundle = SimpleNamespace(residual=residual, config=small_config(residual_output=output))
                loss = _residual_loss(bundle, prepared, None, True)
                self.assertAlmostEqual(loss, expected, places=6)
                np.testing.assert_allclose(residual.values.grad.numpy().ravel(), gradient, atol=1e-7)

    def test_trained_ablation_packages_roundtrip_exactly_and_point_is_labelled_no_interval(self):
        configurations = [
            small_config(condition_spec=self.spec),
            small_config(condition_spec=self.spec, physical_mode="fixed", use_condition_effects=False,
                         use_residual_modulation=False, residual_output="point"),
            small_config(condition_spec=self.spec, physical_mode="fixed"),
            small_config(condition_spec=self.spec, use_residual_modulation=False, residual_output="point"),
        ]
        for config in configurations:
            with self.subTest(physical=config.physical_mode, condition=config.use_condition_effects,
                              modulation=config.use_residual_modulation, output=config.residual_output), \
                    tempfile.TemporaryDirectory() as directory:
                bundle = train(config)
                bundle.assert_ready()
                self.assertTrue(all(not parameter.requires_grad for module in (bundle.main, bundle.residual)
                                    for parameter in module.parameters()))
                frame = samples()[0].frame
                original = predict_cutting_power(bundle, frame, feed_mode="measured", condition=DESCRIPTORS[1])
                self.assertTrue(np.isfinite(original["cutting_quantiles_w"]).all())
                if config.residual_output == "point":
                    self.assertEqual(bundle.metadata["interval_status"], "point_regression_no_interval")
                    self.assertIsNone(bundle.metadata["quantiles"])
                    self.assertEqual(original["interval_kind"], "point_regression_comparison")
                    np.testing.assert_array_equal(original["lower_w"], original["upper_w"])
                else:
                    self.assertEqual(original["interval_kind"], "direct_neural_quantiles")
                    self.assertTrue(np.all(np.diff(original["cutting_quantiles_w"], axis=1) > 0))
                path = Path(directory) / "synthetic_ablation.pt"
                bundle.save(path)
                loaded = load_models(path)
                self.assertEqual(asdict(bundle.config), asdict(loaded.config))
                self.assertEqual(json_value(bundle.metadata), loaded.metadata)
                self.assertEqual(bundle.current_version(), loaded.current_version())
                restored = predict_cutting_power(loaded, frame, feed_mode="measured", condition=DESCRIPTORS[1])
                for name in ("main_w", "residual_w", "cutting_quantiles_w"):
                    np.testing.assert_array_equal(original[name], restored[name])

    def test_old_model_package_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old_model.pt"
            torch.save({"schema": "afc.load.2"}, path)
            with self.assertRaisesRegex(ValueError, "旧模型|不兼容"):
                load_models(path)


if __name__ == "__main__":
    unittest.main()
