"""Small executable multi-condition comparisons; no collected measurements."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch

from verify_unified_load_prediction import synthetic_records, SMALL_CANDIDATE
from spindle_power.load_prediction.common import Cancelled
from spindle_power.load_prediction.condition import fit_condition_spec
from spindle_power.load_prediction.experiment_adapters import run_target_experiment, TARGET_EXPERIMENTS
from spindle_power.load_prediction.feed_model import FeedModelConfig
from spindle_power.load_prediction.feed_training import FeedTrainingConfig, train_feed_fold
from spindle_power.load_prediction.models import ModelConfig, predict_cutting_power, load_models
from spindle_power.load_prediction.training import TrainingConfig, train_two_stage


class TargetExperimentAdaptersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        all_feeds, all_loads = synthetic_records(range(1, 7), points=8)
        cls.feeds, records = [all_feeds[i * 5] for i in range(6)], [all_loads[i * 5] for i in range(6)]
        roles = ["development", "development", "target_adaptation", "target_adaptation",
                 "adaptation_validation", "independent_test"]
        for i, (feed, sample, role) in enumerate(zip(cls.feeds, records, roles)):
            for metadata in (feed.metadata, sample.metadata):
                metadata.update(dataset_role=role, provenance="synthetic_test_only")
                metadata["condition_descriptors"]["workpiece"] = {"hardness": [100., 200., 150., 150., 150., 150.][i]}
            sample.frame["ap_mm"] = np.linspace(1., 2.4, len(sample.frame))
            sample.frame["ae_mm"] = [2., 4., 1.5, 3., 2.5, 1., 4.5, 3.5]
            sample.frame["power_cut_w"] *= sample.frame.ap_mm
        cls.source, cls.adaptation, cls.validation, cls.test = records[:2], records[2:4], records[4:5], records[5:]
        cls.config = TrainingConfig(**SMALL_CANDIDATE["load_training"])
        cls.feed = train_feed_fold(cls.feeds[:2], execute=True,
                    model_config=FeedModelConfig(**SMALL_CANDIDATE["feed_model"]),
                    config=FeedTrainingConfig(**SMALL_CANDIDATE["feed_training"]))
        spec = fit_condition_spec([s.metadata for s in cls.source])
        cls.bundle = train_two_stage(cls.source, execute=True, config=cls.config,
                    model_config=ModelConfig(**SMALL_CANDIDATE["load_model"], condition_spec=spec))
        for sample in records[2:]:
            baseline = predict_cutting_power(cls.bundle, sample.frame, context=sample.metadata, feed_mode="measured")
            sample.frame["power_cut_w"] = baseline["main_w"] * 1.05 + baseline["residual_w"] * 1.1 + 2.

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def arguments(self):
        return dict(feed_model=self.feed, bundle=self.bundle, source_samples=deepcopy(self.source),
                    adaptation_samples=deepcopy(self.adaptation), validation_samples=deepcopy(self.validation),
                    test_samples=deepcopy(self.test), feed_samples=deepcopy(self.feeds),
                    training_config=self.config, synthetic=True, execute=True)

    def test_all_five_execute_and_change_only_allowed_parameters(self):
        source_version, feed_version = self.bundle.current_version(), self.feed.current_version()
        for name in TARGET_EXPERIMENTS:
            with self.subTest(name=name):
                events = []
                report = run_target_experiment(name, **self.arguments(), progress=events.append)
                self.assertEqual(report["status"], "evaluated")
                self.assertEqual(report["applied_experiment"], name)
                self.assertFalse(report["test_used_for_selection"])
                self.assertEqual(report["independent_test"]["feed_mode"], "predicted")
                self.assertEqual(report["independent_test"]["equal_weight_metrics"]["records"], 1)
                self.assertEqual(self.bundle.current_version(), source_version)
                self.assertEqual(self.feed.current_version(), feed_version)
                frozen = next(i for i, event in enumerate(events) if event.get("stage") == "target_model_frozen")
                scored = next(i for i, event in enumerate(events) if event.get("stage") == "independent_test")
                self.assertLess(frozen, scored)
                changed = report["changed_parameters"]
                if name in ("zero_shot", "four_parameter_calibration"):
                    self.assertEqual(changed, [])
                else:
                    self.assertTrue(changed)
                if name == "physical_refit":
                    self.assertTrue(all(key.startswith("main.") for key in changed))
                    self.assertTrue(all("condition_effects" not in key or ".2." in key for key in changed))
                if name == "output_finetune":
                    self.assertTrue(all(key.startswith(("residual.output.", "residual.spread.")) for key in changed))
                if name == "four_parameter_calibration":
                    self.assertTrue(report["calibration_version"])
                json.dumps(report, allow_nan=False)

    def test_test_labels_and_actual_feed_do_not_affect_fitting_or_predictions(self):
        first = run_target_experiment("output_finetune", **self.arguments())
        args = self.arguments()
        args["test_samples"][0].frame["power_cut_w"] += 5000
        args["test_samples"][0].frame["feed_input_mm_min"] *= 9
        args["feed_samples"][-1].frame["feed_actual_mm_min"] *= 9
        second = run_target_experiment("output_finetune", **args)
        self.assertEqual(first["model_version"], second["model_version"])
        self.assertEqual(first["independent_test"]["records"][0]["prediction_fingerprint"],
                         second["independent_test"]["records"][0]["prediction_fingerprint"])
        self.assertNotEqual(first["independent_test"]["equal_weight_metrics"]["mae_w"],
                            second["independent_test"]["equal_weight_metrics"]["mae_w"])

    def test_record_id_and_actual_content_hash_are_both_disjoint(self):
        for field in ("record_id", "measurement_content_sha256"):
            with self.subTest(field=field):
                args = self.arguments()
                args["test_samples"][0].metadata[field] = args["adaptation_samples"][0].metadata[field]
                with self.assertRaisesRegex(ValueError, "重复"):
                    run_target_experiment("zero_shot", **args)
        args = self.arguments()
        args["test_samples"][0].metadata["record_id"] = self.source[0].metadata["record_id"]
        with self.assertRaisesRegex(ValueError, "重复"):
            run_target_experiment("full_retrain", **args)

    def test_real_evidence_or_multicondition_absence_stays_waiting(self):
        args = self.arguments()
        args["synthetic"] = False
        result = run_target_experiment("full_retrain", **args)
        self.assertEqual(result["status"], "awaiting_data")
        self.assertIn("base_labels_verified", result["missing"])
        args = self.arguments()
        args["source_samples"][1].metadata["condition_descriptors"] = deepcopy(args["source_samples"][0].metadata["condition_descriptors"])
        self.assertEqual(run_target_experiment("zero_shot", **args)["status"], "awaiting_multicondition_data")
        args = self.arguments()
        args["adaptation_samples"][0].metadata["provenance"] = "real_measurement"
        with self.assertRaisesRegex(ValueError, "synthetic"):
            run_target_experiment("physical_refit", **args)

    def test_preview_cancellation_and_saved_frozen_weights(self):
        args = self.arguments()
        args["execute"] = False
        self.assertEqual(run_target_experiment("output_finetune", **args)["status"], "prepared_not_executed")
        with tempfile.TemporaryDirectory() as directory:
            events = []
            with self.assertRaises(Cancelled):
                run_target_experiment("output_finetune", **self.arguments(), output_dir=directory,
                    progress=events.append, cancel=lambda: any(e.get("stage") == "target_model_frozen" for e in events))
            self.assertFalse((Path(directory) / "adapted_load.pt").exists())
            result = run_target_experiment("physical_refit", **self.arguments(), output_dir=directory)
            restored = load_models(Path(directory) / "adapted_load.pt")
            self.assertEqual(restored.current_version(), result["model_version"])
            self.assertFalse(any(p.requires_grad for p in restored.main.parameters()))
            self.assertTrue((Path(directory) / "experiment.json").exists())
            args = self.arguments()
            args["bundle"] = restored
            with self.assertRaisesRegex(ValueError, "已进入"):
                run_target_experiment("zero_shot", **args)

    def test_unidentifiable_four_parameter_budget_is_waiting(self):
        args = self.arguments()
        for sample in args["adaptation_samples"]:
            for key in ("ap_mm", "ae_mm", "feed_input_mm_min"):
                sample.frame[key] = sample.frame[key].iloc[0]
        report = run_target_experiment("four_parameter_calibration", **args)
        self.assertEqual(report["status"], "awaiting_identifiable_target_data")
        self.assertFalse(report["executed"])


if __name__ == "__main__":
    unittest.main()
