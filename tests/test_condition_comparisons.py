"""Identity condition comparisons use explicit training levels without leakage."""
from __future__ import annotations

from copy import deepcopy
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
import torch

from spindle_power.load_prediction.comparison_models import fit_identity_condition_spec, encode_identity_conditions
from spindle_power.load_prediction.condition import BRANCH_FIELDS, encode_conditions, fit_condition_spec, validate_condition_spec
from spindle_power.load_prediction.models import build_load_models, load_models, predict_cutting_power
from spindle_power.load_prediction.training import TrainingConfig, train_two_stage
from test_ablation_models import samples, small_config


class IdentityConditionTests(unittest.TestCase):
    def setUp(self):
        self.records = [{"condition_id": value} for value in ("C2", "C1", "C2")]
        self.spec = fit_identity_condition_spec(self.records)

    def test_five_branch_schema_has_only_one_active_one_hot_identity_branch(self):
        self.assertIs(validate_condition_spec(self.spec), self.spec)
        self.assertEqual(self.spec["encoding_mode"], "identity_only_diagnostic")
        self.assertEqual(set(self.spec["branches"]), set(BRANCH_FIELDS))
        self.assertEqual([name for name, branch in self.spec["branches"].items() if branch["active"]], ["machine"])
        field = next(field for field in self.spec["branches"]["machine"]["fields"] if field["name"] == "category")
        self.assertEqual(field["levels"], ["C1", "C2"])
        encoded = encode_identity_conditions(self.records, self.spec)
        np.testing.assert_array_equal(encoded[:, field["offset"]:field["offset"] + field["width"]],
                                      [[0., 1.], [1., 0.], [0., 1.]])
        for name, branch in self.spec["branches"].items():
            if name != "machine":
                values = encoded[:, branch["offset"]:branch["offset"] + branch["width"]]
                np.testing.assert_array_equal(values, np.repeat(values[:1], 3, axis=0))
        np.testing.assert_array_equal(encoded[0], self.spec["reference_encoded"])
        self.assertEqual(encoded.dtype, np.float32)

    def test_program_round_physical_descriptors_and_future_rows_are_not_predictors(self):
        first = [{**row, "round_id": index, "program_id": f"P{index}",
                  "condition_descriptors": {"machine": {"rated_power_w": index * 1000.}}}
                 for index, row in enumerate(self.records)]
        second = [{**row, "round_id": 999, "program_id": "unseen_program",
                   "condition_descriptors": {"workpiece": {"grade": "unobserved"}}} for row in self.records]
        self.assertEqual(fit_identity_condition_spec(first), fit_identity_condition_spec(second))
        np.testing.assert_array_equal(encode_identity_conditions(first, self.spec),
                                      encode_identity_conditions(second, self.spec))
        with self.assertRaisesRegex(ValueError, "训练中|类别"):
            encode_identity_conditions([{"condition_id": "future_condition"}], self.spec)

    def test_missing_invalid_or_only_implicit_identity_is_rejected(self):
        invalid = [{}, {"round_id": 1, "program_id": "C1"},
                   {"condition_descriptors": {"condition_id": "C1"}},
                   {"condition_id": None}, {"condition_id": True}, {"condition_id": 1},
                   {"condition_id": ""}, {"condition_id": "  "}, "C1"]
        for record in invalid:
            with self.subTest(record=record):
                with self.assertRaisesRegex(ValueError, "condition_id"):
                    fit_identity_condition_spec([record])
                with self.assertRaisesRegex(ValueError, "condition_id"):
                    encode_identity_conditions([record], self.spec)

    def test_single_level_is_fixed_and_has_no_active_branch(self):
        spec = fit_identity_condition_spec([{"condition_id": "C1"}] * 3)
        self.assertFalse(any(branch["active"] for branch in spec["branches"].values()))
        encoded = encode_identity_conditions([{"condition_id": "C1"}], spec)
        np.testing.assert_array_equal(encoded[0], spec["reference_encoded"])
        with self.assertRaises(ValueError):
            encode_identity_conditions([{"condition_id": "C2"}], spec)

    def test_accepts_sample_metadata_and_iterables_without_mutating_spec(self):
        objects = [SimpleNamespace(metadata=row) for row in self.records]
        self.assertEqual(fit_identity_condition_spec(iter(objects)), self.spec)
        before = deepcopy(self.spec)
        np.testing.assert_array_equal(encode_identity_conditions(iter(objects), self.spec),
                                      encode_identity_conditions(self.records, self.spec))
        self.assertEqual(before, self.spec)
        self.assertEqual(encode_identity_conditions([], self.spec).shape, (0, self.spec["encoded_dim"]))
        with self.assertRaisesRegex(ValueError, "非空"):
            fit_identity_condition_spec([])

    def test_descriptor_spec_and_tampered_identity_spec_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "identity_only_diagnostic"):
            encode_identity_conditions(self.records, fit_condition_spec([{}]))
        changed = deepcopy(self.spec)
        changed["identity_field"] = "program_id"
        with self.assertRaisesRegex(ValueError, "版本|编号"):
            encode_identity_conditions(self.records, changed)

    def test_public_encoder_dispatches_and_never_permits_unknown_identity(self):
        np.testing.assert_array_equal(encode_conditions(self.records, self.spec),
                                      encode_identity_conditions(self.records, self.spec))
        for allow in (False, True):
            with self.subTest(allow_unknown=allow), self.assertRaisesRegex(ValueError, "训练中|类别"):
                encode_conditions([{"condition_id": "unseen"}], self.spec, allow_unknown=allow)


class ConditionComparisonModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_comparison_config_rejects_mismatched_encodings_and_fixed_implicit_head(self):
        identity = fit_identity_condition_spec([{"condition_id": "C1"}])
        for config in (small_config(condition_spec=identity),
                       small_config(condition_spec=fit_condition_spec([{}]), condition_input="identity_only_diagnostic"),
                       small_config(coefficient_condition="implicit", physical_mode="fixed"),
                       small_config(condition_input="program_id"), small_config(coefficient_condition="other")):
            with self.subTest(condition_input=config.condition_input, coefficient_condition=config.coefficient_condition), \
                    self.assertRaises(ValueError):
                config.validate()

    def test_implicit_condition_changes_coefficients_with_exact_unit_multipliers(self):
        records = [sample.metadata for sample in samples()]
        spec = fit_condition_spec(records)
        bundle = build_load_models(small_config(condition_spec=spec, coefficient_condition="implicit"))
        with torch.no_grad():
            for layer in bundle.main.coefficients:
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.fill_(.2)
                    layer.bias.fill_(.1)
        physics = torch.tensor([[1., 2., 1000., 5000., 10., 4.]]).repeat(2, 1)
        condition = torch.tensor(encode_conditions(records, spec))
        parts = bundle.main.components(physics, condition)
        for name in ("ks_multiplier", "ke_multiplier"):
            self.assertTrue(torch.equal(parts[name], torch.ones(2)))
        for name in ("ks", "ke"):
            self.assertGreater(parts[name][1].item(), parts[name][0].item())
            self.assertTrue(torch.all(parts[name] > 0))
        parts["power_w"].sum().backward()
        first_layer = next(layer for layer in bundle.main.coefficients if isinstance(layer, torch.nn.Linear))
        self.assertGreater(first_layer.in_features, 2)
        self.assertGreater(first_layer.weight.grad[:, 2:].abs().sum().item(), 0.)
        self.assertTrue(all(not parameter.requires_grad for parameter in bundle.main.condition_effects.parameters()))

    def test_single_identity_level_freezes_explicit_branches_and_implicit_condition_gradient(self):
        record = {"condition_id": "only_level"}
        spec = fit_identity_condition_spec([record, record])
        physics = torch.tensor([[1., 2., 1000., 5000., 10., 4.]])
        condition = torch.tensor(encode_identity_conditions([record], spec))
        for mode in ("explicit", "implicit"):
            with self.subTest(mode=mode):
                bundle = build_load_models(small_config(condition_spec=spec,
                    condition_input="identity_only_diagnostic", coefficient_condition=mode))
                self.assertTrue(all(not parameter.requires_grad for parameter in bundle.main.condition_effects.parameters()))
                self.assertTrue(all(not parameter.requires_grad for parameter in bundle.residual.condition_effects.parameters()))
                bundle.main(physics, condition).sum().backward()
                if mode == "implicit":
                    layer = next(layer for layer in bundle.main.coefficients if isinstance(layer, torch.nn.Linear))
                    self.assertTrue(torch.equal(layer.weight.grad[:, 2:], torch.zeros_like(layer.weight.grad[:, 2:])))

    def test_public_training_prediction_and_roundtrip_for_identity_and_implicit_comparisons(self):
        training = samples()
        for index, sample in enumerate(training):
            sample.metadata["condition_id"] = f"C{index}"
        for source, mode in (("identity_only_diagnostic", "explicit"), ("descriptors", "implicit"),
                             ("identity_only_diagnostic", "implicit")):
            with self.subTest(source=source, mode=mode), tempfile.TemporaryDirectory() as directory:
                spec = (fit_identity_condition_spec(training) if source == "identity_only_diagnostic"
                        else fit_condition_spec(training))
                config = small_config(condition_spec=spec, condition_input=source, coefficient_condition=mode)
                bundle = train_two_stage(training, model_config=config,
                    config=TrainingConfig(main_epochs=1, residual_epochs=1, main_batch_points=8, seed=17), execute=True)
                bundle.assert_ready()
                original = predict_cutting_power(bundle, training[0].frame, feed_mode="measured",
                                                condition=training[1].metadata)
                self.assertTrue(np.isfinite(original["cutting_quantiles_w"]).all())
                self.assertTrue(np.all(np.diff(original["cutting_quantiles_w"], axis=1) > 0))
                path = Path(directory) / "synthetic_comparison.pt"
                bundle.save(path)
                loaded = load_models(path)
                self.assertEqual(asdict(bundle.config), asdict(loaded.config))
                self.assertEqual(bundle.current_version(), loaded.current_version())
                restored = predict_cutting_power(loaded, training[0].frame, feed_mode="measured",
                                                condition=training[1].metadata)
                for name in ("main_w", "residual_w", "cutting_quantiles_w"):
                    np.testing.assert_array_equal(original[name], restored[name])
                if source == "identity_only_diagnostic":
                    with self.assertRaises(ValueError):
                        predict_cutting_power(loaded, training[0].frame, feed_mode="measured",
                                              condition={"condition_id": "unseen"})


if __name__ == "__main__":
    unittest.main()
