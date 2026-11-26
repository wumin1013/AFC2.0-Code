"""Behavior-level integration checks; all data and weights are synthetic."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

for key, value in {"MKL_THREADING_LAYER": "SEQUENTIAL", "OMP_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUTF8": "1"}.items():
    os.environ.setdefault(key, value)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from verify_unified_load_prediction import synthetic_records, SMALL_CANDIDATE, check_protocol_interfaces
from spindle_power.load_prediction.common import Cancelled, save_json
from spindle_power.load_prediction.training_service import TrainingService, balanced_metrics, _predicted_frame
from spindle_power.load_prediction.training import _sample_weights


class SnapshotAndInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.service = TrainingService(self.root / "service")
        self.feeds, self.loads = synthetic_records((1, 2))
        self.saved = self.service.save_prepared(self.feeds, self.loads, source_snapshot={"synthetic": True}, synthetic=True)

    def tearDown(self):
        self.temp.cleanup()

    def test_source_mutation_does_not_change_saved_snapshot_and_csv_tamper_rejected(self):
        expected = self.feeds[0].frame.feed_actual_mm_min.iloc[0]
        self.feeds[0].frame.loc[0, "feed_actual_mm_min"] = 999999.
        restored, _, _ = self.service.load_prepared(self.saved["snapshot_id"])
        self.assertEqual(restored[0].frame.feed_actual_mm_min.iloc[0], expected)
        path = Path(self.saved["directory"]) / "feed_0000.csv"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "快照|修改|指纹"):
            self.service.load_prepared(self.saved["snapshot_id"])

    def test_metadata_tamper_invalidates_the_fixed_snapshot(self):
        path = Path(self.saved["directory"]) / "snapshot.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["records"][0]["metadata"]["dataset_role"] = "independent_test"
        save_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "快照|修改|指纹|完整"):
            self.service.load_prepared(self.saved["snapshot_id"])

    def test_http_cli_and_service_protocol_are_identical(self):
        check_protocol_interfaces(self.service, self.root)
        checked = json.loads((self.root / "protocol_http_cli.json").read_text(encoding="utf-8"))
        self.assertTrue(checked["equal"])
        self.assertEqual(len(checked["protocol"]["folds"]), 9)

    def test_round_feature_record_weighting_ignores_record_length(self):
        samples = self.loads[:2] + self.loads[5:6]
        weights = _sample_weights(samples)
        self.assertEqual(weights, [.25, .25, .5])
        records = [dict(round_id=1, feature="a", metrics={"rmse_w": 2., "count": 100000}),
                   dict(round_id=1, feature="b", metrics={"rmse_w": 4., "count": 4}),
                   dict(round_id=2, feature="a", metrics={"rmse_w": 9., "count": 4})]
        result = balanced_metrics(records)
        self.assertEqual(result["rmse_w"], 6.)
        self.assertEqual(result["effective_counts"]["count"], 100008)

    def test_real_snapshot_cannot_bypass_backend_qualification(self):
        real = self.service.save_prepared(self.feeds, self.loads, source_snapshot={}, synthetic=False)
        with self.assertRaisesRegex(ValueError, "核验|正式|450|协议"):
            self.service.preview(real["snapshot_id"], [SMALL_CANDIDATE])
        self.assertEqual(self.service.models(), [])


class NeuralStagingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from spindle_power.load_prediction.feed_model import FeedModelConfig
        from spindle_power.load_prediction.feed_training import FeedTrainingConfig, train_feed_fold
        from spindle_power.load_prediction.models import ModelConfig
        from spindle_power.load_prediction.training import TrainingConfig, train_two_stage
        torch.set_num_threads(1)
        cls.feeds, cls.loads = synthetic_records((1, 2))
        cls.feed = train_feed_fold(cls.feeds[:5], cls.feeds[5:],
                                  model_config=FeedModelConfig(**SMALL_CANDIDATE["feed_model"]),
                                  config=FeedTrainingConfig(**SMALL_CANDIDATE["feed_training"]), execute=True)
        cls.bundle = train_two_stage(cls.loads[:5], validation=cls.loads[5:],
                                     model_config=ModelConfig(**SMALL_CANDIDATE["load_model"]),
                                     config=TrainingConfig(**SMALL_CANDIDATE["load_training"]), execute=True)

    def test_all_networks_frozen_residual_binds_main_quantiles_sorted(self):
        from spindle_power.load_prediction.neural import state_version
        from spindle_power.load_prediction.models import predict_cutting_power
        self.assertFalse(any(p.requires_grad for p in self.feed.network.parameters()))
        for model in (self.bundle.main, self.bundle.residual):
            self.assertFalse(any(p.requires_grad for p in model.parameters()))
        self.assertEqual(state_version(self.bundle.main), self.bundle.residual_main_version)
        prediction = predict_cutting_power(self.bundle, _predicted_frame(self.loads[5], self.feeds[5], self.feed))
        self.assertTrue(np.isfinite(prediction["cutting_quantiles_w"]).all())
        self.assertTrue((np.diff(prediction["cutting_quantiles_w"], axis=1) >= 0).all())

    def test_reload_identical_and_target_inputs_never_leak_into_prediction(self):
        from spindle_power.load_prediction.feed_model import load_feed_model
        from spindle_power.load_prediction.models import load_models, predict_cutting_power
        changed_feed, changed_load = deepcopy(self.feeds[5]), deepcopy(self.loads[5])
        original = _predicted_frame(changed_load, changed_feed, self.feed)
        changed_feed.frame["feed_actual_mm_min"] = 1e9
        changed_feed.frame["valid_label"] = False
        changed_load.frame["power_cut_w"] = 1e9
        changed_load.frame["feed_input_mm_min"] = 1e9
        changed = _predicted_frame(changed_load, changed_feed, self.feed)
        np.testing.assert_array_equal(original.feed_input_mm_min, changed.feed_input_mm_min)
        p1 = predict_cutting_power(self.bundle, original)["cutting_quantiles_w"]
        p2 = predict_cutting_power(self.bundle, changed)["cutting_quantiles_w"]
        np.testing.assert_array_equal(p1, p2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.feed.save(path / "feed.pt")
            self.bundle.save(path / "load.pt")
            feed, bundle = load_feed_model(path / "feed.pt"), load_models(path / "load.pt")
            restored = predict_cutting_power(bundle, _predicted_frame(self.loads[5], self.feeds[5], feed))
            np.testing.assert_array_equal(restored["cutting_quantiles_w"], p1)

    def test_optimizer_cancellation_returns_no_usable_model(self):
        from spindle_power.load_prediction.training import train_two_stage
        with self.assertRaises(Cancelled):
            train_two_stage(self.loads[:5], execute=True, cancel=lambda: True)


class PersistentJobTests(unittest.TestCase):
    def setUp(self):
        from spindle_power.load_prediction.job_service import JobService
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.service = TrainingService(self.root / "service")
        feeds, loads = synthetic_records((1, 2))
        self.snapshot = self.service.save_prepared(feeds, loads, source_snapshot={}, synthetic=True)
        self.jobs = JobService(self.service)

    def tearDown(self):
        self.jobs.shutdown()
        self.temp.cleanup()

    def request(self, epochs=1):
        candidate = deepcopy(SMALL_CANDIDATE)
        candidate["feed_training"]["epochs"] = epochs
        candidate["feed_training"]["patience"] = epochs
        return {"snapshot_id": self.snapshot["snapshot_id"], "candidates": [candidate]}

    def wait(self, job_id, predicate, timeout=35):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            state = self.jobs.get(job_id)
            if predicate(state):
                return state
            time.sleep(.1)
        self.fail(f"task timeout: {self.jobs.get(job_id)}")

    def test_queued_cancel_is_persistent(self):
        with self.jobs.lock:
            state = self.jobs.create(self.request())
            result = self.jobs.cancel(state["job_id"])
        self.assertEqual(result["status"], "cancelled")
        self.jobs.shutdown()
        from spindle_power.load_prediction.job_service import JobService
        self.jobs = JobService(self.service)
        self.assertEqual(self.jobs.get(state["job_id"])["status"], "cancelled")
        self.assertEqual(self.service.models(include_synthetic=True), [])

    def test_worker_validation_failure_is_persisted_and_no_model_published(self):
        state = self.jobs.create(self.request(epochs=0))
        failed = self.wait(state["job_id"], lambda s: s["status"] in ("failed", "interrupted"))
        self.assertEqual(failed["status"], "failed")
        self.assertIn("epochs", failed["error"])
        self.assertEqual(self.service.models(include_synthetic=True), [])

    def test_unexpected_process_exit_is_interrupted_and_survives_restart(self):
        state = self.jobs.create(self.request(epochs=1000))
        self.wait(state["job_id"], lambda s: s["status"] == "running")
        with self.jobs.lock:
            self.assertIsNotNone(self.jobs.active)
            self.jobs.active[1].kill()
        interrupted = self.wait(state["job_id"], lambda s: s["status"] == "interrupted")
        self.assertIn("异常退出", interrupted["error"])
        self.jobs.shutdown()
        from spindle_power.load_prediction.job_service import JobService
        self.jobs = JobService(self.service)
        self.assertEqual(self.jobs.get(state["job_id"])["status"], "interrupted")
        self.assertEqual(self.service.models(include_synthetic=True), [])

    def test_running_cancel_persists_without_publishing_partial_model(self):
        state = self.jobs.create(self.request(epochs=1000))
        self.wait(state["job_id"], lambda s: bool(s.get("progress")), timeout=45)
        self.jobs.cancel(state["job_id"])
        cancelled = self.wait(state["job_id"], lambda s: s["status"] in ("cancelled", "failed", "interrupted"))
        self.assertEqual(cancelled["status"], "cancelled", cancelled)
        self.assertEqual(self.service.models(include_synthetic=True), [])

    def test_second_service_cannot_interrupt_a_live_worker(self):
        """A second CLI/browser owner must refuse access or preserve the live job."""
        from spindle_power.load_prediction.job_service import JobService
        state = self.jobs.create(self.request(epochs=1000))
        self.wait(state["job_id"], lambda s: s["status"] == "running")
        second = None
        try:
            try:
                second = JobService(self.service)
            except (ValueError, RuntimeError, OSError):
                self.assertEqual(self.jobs.get(state["job_id"])["status"], "running")
                return
            self.assertEqual(self.jobs.get(state["job_id"])["status"], "running")
            second.shutdown()
            second = None
            self.assertEqual(self.jobs.get(state["job_id"])["status"], "running")
        finally:
            if second is not None:
                second.shutdown()


if __name__ == "__main__":
    unittest.main()
