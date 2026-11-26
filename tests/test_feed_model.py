"""Small synthetic CPU checks; never consume collected machining records."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from spindle_power.load_prediction.feed_model import (FeedModelConfig, build_feed_model,
    load_feed_model, predict_actual_feed, sequence_windows)
from spindle_power.load_prediction.feed_samples import FEED_FEATURE_COLUMNS, FEED_SAMPLE_SCHEMA, FeedSample
from spindle_power.load_prediction.feed_training import FeedTrainingConfig, train_feed, train_feed_fold


def synthetic_samples():
    samples = []
    for round_id in range(1, 91):
        family = (round_id - 1) % 3
        frame = pd.DataFrame(np.zeros((4, len(FEED_FEATURE_COLUMNS))), columns=FEED_FEATURE_COLUMNS)
        frame["x_mm"] = np.arange(4, dtype=float)
        frame["feed_program_mm_min"] = 1000. + family * 100.
        frame["feed_override"] = 1.
        frame["line"] = [10, 11, 12, 13]
        frame["execution_instance"] = 0
        frame["sequence_id"] = 0
        frame["path_mm"] = [0., 1., 2., 3.]
        frame["segment_mm"] = 0.
        frame["feed_cmd_mm_min"] = np.nan
        frame["feed_actual_mm_min"] = [0., np.nan, 800., 600.]
        frame["valid_label"] = [True, False, True, True]
        meta = dict(schema=FEED_SAMPLE_SCHEMA, feature_columns=list(FEED_FEATURE_COLUMNS),
                    program_id=f"p{family}", program_version=f"v{family}", round_id=round_id,
                    machine_version="synthetic_machine", machine_settings_version="synthetic_settings",
                    feed_override=1., spacing_mm=1., machining_mode="synthetic_only", complete=True,
                    round_complete=True, alignment_reliable=True, no_alarm=True,
                    dataset_role="development", provenance="synthetic_test_only")
        samples.append(FeedSample(frame, meta))
    return samples


ASSIGNMENT = {"p0": "train", "p1": "validation", "p2": "test"}
SMALL_MODEL = FeedModelConfig(hidden_size=4, num_layers=1, window_points=2, context_points=1)
SMALL_TRAINING = FeedTrainingConfig(epochs=3, batch_size=128, patience=1, min_delta=1e6, seed=31)


class FeedPreparationTests(unittest.TestCase):
    def test_prepare_and_dry_run_do_not_import_torch(self):
        code = "\n".join([
            "import sys", f"sys.path.insert(0, {str(ROOT / 'tests')!r})",
            "from test_feed_model import synthetic_samples, ASSIGNMENT, train_feed",
            "assert 'torch' not in sys.modules",
            "report = train_feed(synthetic_samples(), assignment=ASSIGNMENT)",
            "assert report['action'] == 'dry_run' and report['ready_to_train']",
            "assert 'torch' not in sys.modules",
        ])
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_windows_cross_nc_blocks_and_ignore_label_gaps(self):
        sample = synthetic_samples()[0]
        windows = list(sequence_windows(sample.frame, SMALL_MODEL))
        self.assertEqual([(w.start, w.stop, w.core_start, w.core_stop) for w in windows],
                         [(0, 3, 0, 2), (1, 4, 2, 4)])
        sample.frame["sequence_id"] = [0, 0, 1, 1]
        self.assertEqual([(w.start, w.stop) for w in sequence_windows(sample.frame, SMALL_MODEL)], [(0, 2), (2, 4)])

    def test_full_rounds_and_explicit_completeness_are_required(self):
        samples = synthetic_samples()
        report = train_feed(samples[:-1], assignment=ASSIGNMENT)
        self.assertFalse(report["ready_to_train"])
        self.assertEqual(report["missing_rounds"], [90])
        samples[0].metadata.pop("round_complete")
        with self.assertRaisesRegex(ValueError, "round_complete"):
            train_feed(samples, assignment=ASSIGNMENT, execute=True)

    def test_same_program_version_cannot_cross_splits(self):
        samples = synthetic_samples()
        samples[1].metadata["program_version"] = samples[0].metadata["program_version"]
        report = train_feed(samples, assignment=ASSIGNMENT)
        self.assertFalse(report["ready_to_train"])
        with self.assertRaisesRegex(ValueError, "program_version"):
            train_feed(samples, assignment=ASSIGNMENT, execute=True)

    def test_incomplete_assignment_is_reported_without_training(self):
        report = train_feed(synthetic_samples(), assignment={"p0": "train"})
        self.assertFalse(report["ready_to_train"])
        self.assertTrue(report["blockers"])
        self.assertEqual(report["action"], "dry_run")

    def test_same_measurement_cannot_enter_reserved_and_development_roles(self):
        samples = synthetic_samples()
        samples[0].metadata["measurement_source"] = "synthetic_record.csv"
        duplicate = deepcopy(samples[0])
        duplicate.metadata["dataset_role"] = "independent_test"
        report = train_feed(samples + [duplicate], assignment=ASSIGNMENT)
        self.assertTrue(any("measurement_source" in reason for reason in report["blockers"]))
        with self.assertRaisesRegex(ValueError, "measurement_source"):
            train_feed(samples + [duplicate], assignment=ASSIGNMENT, execute=True)

    def test_reserved_experiments_do_not_enter_development(self):
        samples = synthetic_samples()
        memory = deepcopy(samples[0])
        memory.metadata.update(dataset_role="memory_validation", round_id=91)
        external = deepcopy(samples[0])
        external.metadata.update(dataset_role="independent_test", round_id=101)
        report = train_feed(samples + [memory, external], assignment=ASSIGNMENT)
        self.assertEqual(report["reserved_records"], {"memory_validation": 1, "independent_test": 1})
        self.assertEqual(sum(split["records"] for split in report["splits"].values()), 90)


class FeedRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.samples = synthetic_samples()
        cls.events = []
        cls.model = train_feed(cls.samples, assignment=ASSIGNMENT, model_config=SMALL_MODEL,
                               config=SMALL_TRAINING, execute=True, progress=cls.events.append)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.old_threads)

    def test_real_bidirectionality_and_packing_mask(self):
        torch = self.torch
        torch.manual_seed(2)
        model = build_feed_model(SMALL_MODEL)
        self.assertTrue(model.network.lstm.bidirectional)
        x = torch.randn(2, 4, len(FEED_FEATURE_COLUMNS))
        lengths = torch.tensor([2, 4])
        model.network.eval()
        with torch.no_grad():
            expected = model.network(x[:1, :2], lengths[:1])[0]
            actual = model.network(x, lengths)
            np.testing.assert_allclose(actual[0, :2].numpy(), expected.numpy(), atol=1e-6)
            self.assertEqual(actual[0, 2:].tolist(), [0., 0.])
            changed = x[1:2].clone()
            changed[0, -1] += 3.
            future_changed = model.network(changed, lengths[1:])[0, 0]
            self.assertGreater(abs(float(future_changed - actual[1, 0])), 1e-7)

    def test_untrained_model_is_blocked(self):
        with self.assertRaisesRegex(ValueError, "尚未训练"):
            predict_actual_feed(build_feed_model(SMALL_MODEL), self.samples[0])

    def test_best_validation_checkpoint_and_train_only_normalization(self):
        model = self.model
        self.assertTrue(model.trained)
        self.assertEqual(model.metadata["best_epoch"], 1)
        self.assertEqual(len(model.metadata["history"]), 2)
        feature = list(FEED_FEATURE_COLUMNS).index("feed_program_mm_min")
        self.assertEqual(model.normalization["mean"][feature], 1000.)
        self.assertEqual(model.metadata["test_protocol"], "development_program_holdout")
        self.assertEqual(self.events[-1]["phase"], "completed")
        self.assertTrue(all(not parameter.requires_grad for parameter in model.network.parameters()))

    def test_test_labels_and_masked_labels_do_not_influence_model(self):
        samples = synthetic_samples()
        for sample in samples:
            sample.frame.loc[~sample.frame.valid_label, "feed_actual_mm_min"] = 1e30
            if sample.metadata["program_id"] == "p2":
                sample.frame.loc[sample.frame.valid_label, "feed_actual_mm_min"] *= 20.
        other = train_feed(samples, assignment=ASSIGNMENT, model_config=SMALL_MODEL,
                           config=SMALL_TRAINING, execute=True)
        self.assertEqual(other.model_version, self.model.model_version)
        self.assertNotEqual(other.metadata["test_metrics"]["mae_mm_min"], self.model.metadata["test_metrics"]["mae_mm_min"])

    def test_batching_preserves_record_weighted_objective(self):
        other = train_feed(synthetic_samples(), assignment=ASSIGNMENT, model_config=SMALL_MODEL,
                           config=replace(SMALL_TRAINING, batch_size=7), execute=True)
        np.testing.assert_allclose([item["train_loss"] for item in other.metadata["history"]],
                                   [item["train_loss"] for item in self.model.metadata["history"]], rtol=1e-5, atol=1e-6)

    def test_prediction_never_reads_measured_labels_or_invents_time(self):
        sample = deepcopy(self.samples[0])
        first = predict_actual_feed(self.model, sample)
        sample.frame["feed_actual_mm_min"] = "not a feature"
        sample.frame["valid_label"] = "not a feature"
        second = predict_actual_feed(self.model, sample)
        np.testing.assert_allclose(first.frame.feed_predicted_mm_min, second.frame.feed_predicted_mm_min)
        self.assertTrue(np.all(second.frame.feed_predicted_mm_min >= 0))
        self.assertNotIn("time_s", second.frame)
        self.assertEqual(second.metadata["source"], "bilstm_actual_feed")
        self.assertIs(second.metadata["trained"], True)
        self.assertEqual(second.metadata["machine_version"], sample.metadata["machine_version"])

    def test_incompatible_context_and_feature_schema_are_blocked(self):
        sample = deepcopy(self.samples[0])
        sample.metadata["machining_mode"] = "different_mode"
        with self.assertRaisesRegex(ValueError, "兼容域"):
            predict_actual_feed(self.model, sample)
        sample = deepcopy(self.samples[0])
        sample.metadata["spacing_mm"] = .5
        with self.assertRaisesRegex(ValueError, "空间采样间距"):
            predict_actual_feed(self.model, sample)
        sample = deepcopy(self.samples[0])
        sample.metadata["feature_columns"] = list(reversed(FEED_FEATURE_COLUMNS))
        with self.assertRaisesRegex(ValueError, "特征顺序"):
            predict_actual_feed(self.model, sample)

    def test_safe_save_load_and_normalization_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feed.pt"
            self.model.save(path)
            loaded = load_feed_model(path)
            self.assertEqual(loaded.model_version, self.model.model_version)
            np.testing.assert_allclose(predict_actual_feed(loaded, self.samples[0]).frame.feed_predicted_mm_min,
                                       predict_actual_feed(self.model, self.samples[0]).frame.feed_predicted_mm_min)
            payload = self.torch.load(path, weights_only=True, map_location="cpu")
            payload["normalization"]["target_scale_mm_min"] *= 2
            self.torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "版本不一致"):
                load_feed_model(path)

    def test_cancellation_prevents_training_completion(self):
        from spindle_power.load_prediction.common import Cancelled
        with self.assertRaises(Cancelled):
            train_feed(self.samples, assignment=ASSIGNMENT, model_config=SMALL_MODEL,
                       config=SMALL_TRAINING, execute=True, cancel=lambda: True)

    def test_round_fold_allows_same_nc_and_full_training_uses_fixed_epochs(self):
        records = synthetic_samples()
        train, validation = [records[0], records[3]], [records[6]]
        first = train_feed_fold(train, validation, model_config=SMALL_MODEL,
                                config=SMALL_TRAINING, execute=True)
        changed = deepcopy(validation)
        changed[0].frame["feed_program_mm_min"] *= 50
        second = train_feed_fold(train, changed, model_config=SMALL_MODEL,
                                 config=SMALL_TRAINING, execute=True)
        self.assertEqual(first.normalization, second.normalization)
        full = train_feed_fold(train + validation, model_config=SMALL_MODEL,
                               config=SMALL_TRAINING, execute=True)
        self.assertEqual(len(full.metadata["history"]), SMALL_TRAINING.epochs)
        self.assertEqual(full.metadata["best_epoch"], SMALL_TRAINING.epochs)
        self.assertIsNone(full.metadata["test_metrics"])
        self.assertTrue(all(e["validation_loss"] is None for e in full.metadata["history"]))

    def test_round_fold_rejects_shared_round_source_and_reserved_data(self):
        records = synthetic_samples()
        with self.assertRaisesRegex(ValueError, "计划轮次"):
            train_feed_fold([records[0]], [deepcopy(records[0])], execute=True)
        records[0].metadata["measurement_content_sha256"] = "copied"
        records[1].metadata["measurement_content_sha256"] = "copied"
        with self.assertRaisesRegex(ValueError, "实测来源"):
            train_feed_fold([records[0]], [records[1]], execute=True)
        records[0].metadata["dataset_role"] = "interval_calibration"
        with self.assertRaisesRegex(ValueError, "独立测试/校准"):
            train_feed_fold([records[0]], execute=True)


if __name__ == "__main__":
    unittest.main()
