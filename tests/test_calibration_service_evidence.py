"""Synthetic regression tests for calibration evidence and atomic publication.

Only registry loading is replaced with a frozen synthetic bundle. Numerical
calibration, prediction, evidence files, hashing and publication run normally.
No collected measurement, formal registry or network training is used.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

for key, value in {"MKL_THREADING_LAYER": "SEQUENTIAL", "OMP_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}.items():
    os.environ.setdefault(key, value)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np
import pandas as pd
import torch

from test_condition_quantiles import frame, trained_test_bundle
from spindle_power.load_prediction.calibration import CalibrationResult, IntervalCalibrator
from spindle_power.load_prediction.common import file_hash
from spindle_power.load_prediction.memory import PositionMemory
from spindle_power.load_prediction.models import predict_cutting_power
from spindle_power.load_prediction.training_service import TrainingService


class CalibrationServiceEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="afc_synthetic_calibration_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bundle = trained_test_bundle()
        self.service = TrainingService(self.root / "service")
        mocked = patch.object(self.service, "load_model", return_value=(None, self.bundle, {"synthetic": True}))
        mocked.start()
        self.addCleanup(mocked.stop)

    def metadata(self, name, role):
        return dict(record_id=name, dataset_role=role, provenance="synthetic_test_only",
                    complete=True, alignment_reliable=True, no_alarm=True, tool_healthy=True,
                    program_version="synthetic_program", parameter_group="synthetic_parameters",
                    tool_session="synthetic_session", actual_event_order=100, round_id=100)

    def condition_record(self, number, role="condition_calibration"):
        data = frame().iloc[:3].copy()
        data["feed_input_mm_min"] += number * 70.
        baseline = predict_cutting_power(self.bundle, data)
        data["power_cut_w"] = baseline["main_w"] * 1.1 + baseline["residual_w"]
        data["valid_main"] = data["valid_residual"] = True
        data["feed_source"] = "measured_actual_feed"
        path = self.root / f"condition_{number}.csv"
        data.to_csv(path, index=False, encoding="utf-8")
        digest = file_hash(path)
        metadata = self.metadata(f"synthetic_condition_{number}", role)
        metadata["measurement_content_sha256"] = digest
        return dict(prepared_csv=str(path), prepared_csv_sha256=digest, metadata=metadata)

    def condition_request(self):
        return dict(model_id="synthetic_frozen_bundle", allow_synthetic=True, kind="condition",
                    records=[self.condition_record(0), self.condition_record(1)],
                    validation_records=[self.condition_record(3, "adaptation_validation")])

    def interval_record(self, number, role="interval_calibration", *, calibration=None, memory=None):
        data = frame().iloc[:3].copy()
        data["feed_input_mm_min"] += number * 31.
        metadata = self.metadata(f"synthetic_interval_{number}", role)
        prediction = predict_cutting_power(self.bundle, data, context=metadata, round_id=100,
                                           calibration=calibration, memory=memory)
        target = prediction["cutting_power_w"] + np.array([.2, -.3, 1. + number * .03])
        path = self.root / f"interval_{number}.csv"
        aligned = pd.DataFrame({"power_cut_w": target, "line": [1, 2, 3], "segment_mm": 0.})
        aligned.to_csv(path, index=False, encoding="utf-8")
        digest = file_hash(path)
        metadata["measurement_content_sha256"] = digest
        return dict(group_id=f"synthetic_group_{number}", record_id=metadata["record_id"],
                    dataset_role=role, metadata=metadata, measurement_content_sha256=digest,
                    aligned_measurement=str(path), aligned_measurement_sha256=digest,
                    lower_w=prediction["q05_w"].tolist(), upper_w=prediction["q95_w"].tolist(),
                    median_w=prediction["q50_w"].tolist(), target_w=target.tolist(),
                    prediction_version=prediction["prediction_version"])

    def interval_request(self, *, calibration=None, memory=None):
        request = dict(model_id="synthetic_frozen_bundle", allow_synthetic=True, kind="interval",
                       records=[self.interval_record(i, calibration=calibration, memory=memory) for i in range(9)],
                       validation_records=[self.interval_record(20, "adaptation_validation",
                                                                calibration=calibration, memory=memory)])
        if calibration is not None:
            path = self.root / "synthetic_condition.json"
            calibration.save(path)
            request["condition_calibration"] = str(path)
        if memory is not None:
            path = self.root / "synthetic_memory.json"
            memory.save(path)
            request["memory"] = str(path)
        return request

    def calibration_artifacts(self):
        directory = self.service.output_dir / "calibrations"
        return {path for name in ("condition.json", "interval.json", "calibration.json")
                for path in directory.rglob(name)}

    def assert_rejected(self, request):
        before = self.calibration_artifacts()
        try:
            report = self.service.calibrate(request)
        except ValueError:
            pass
        else:
            self.assertEqual(report.get("status"), "rejected", report)
        self.assertEqual(self.calibration_artifacts(), before, "Rejected calibration left a loadable artifact")

    def test_condition_valid_independent_evidence_can_be_calibrated(self):
        report = self.service.calibrate(self.condition_request())
        self.assertEqual(report["status"], "calibrated_with_independent_adaptation_validation")
        result = CalibrationResult.load(Path(report["directory"]) / "condition.json")
        self.assertEqual(result.model_version, self.bundle.current_version())

    def test_condition_copied_content_with_new_id_and_path_is_rejected(self):
        request = self.condition_request()
        clone = deepcopy(request["records"][0])
        copied_path = self.root / "renamed_copy.csv"
        shutil.copyfile(clone["prepared_csv"], copied_path)
        clone["prepared_csv"] = str(copied_path)
        clone["metadata"].update(record_id="renamed_copy", dataset_role="adaptation_validation")
        request["validation_records"] = [clone]
        self.assert_rejected(request)

    def test_condition_duplicate_fit_content_is_rejected(self):
        request = self.condition_request()
        clone = deepcopy(request["records"][0])
        clone["metadata"]["record_id"] = "renamed_fit"
        request["records"][1] = clone
        self.assert_rejected(request)

    def test_condition_fit_and_validation_cannot_reuse_training_hashes(self):
        for role in ("records", "validation_records"):
            for key in ("measurement_hash", "measurement_content_hash", "measurement_content_sha256"):
                with self.subTest(role=role, training_hash_key=key):
                    request = self.condition_request()
                    digest = request[role][0]["metadata"]["measurement_content_sha256"]
                    self.bundle.metadata["samples"] = [{"record_id": "synthetic_training", key: digest}]
                    self.assert_rejected(request)

    def test_condition_declared_prepared_csv_hash_tampering_is_rejected(self):
        for role in ("records", "validation_records"):
            with self.subTest(role=role):
                request = self.condition_request()
                request[role][0]["prepared_csv_sha256"] = "0" * 64
                self.assert_rejected(request)

    def test_condition_failed_validation_leaves_no_loadable_file(self):
        request = self.condition_request()
        validation = request["validation_records"][0]
        path = Path(validation["prepared_csv"])
        data = pd.read_csv(path)
        data["feed_source"] = "bilstm_actual_feed"
        data.to_csv(path, index=False, encoding="utf-8")
        validation["prepared_csv_sha256"] = file_hash(path)
        validation["metadata"]["measurement_content_sha256"] = file_hash(path)
        self.assert_rejected(request)

    def test_interval_duplicate_fit_content_with_new_group_is_rejected(self):
        request = self.interval_request()
        clone = deepcopy(request["records"][0])
        clone.update(group_id="renamed_group", record_id="renamed_interval")
        clone["metadata"]["record_id"] = "renamed_interval"
        request["records"][1] = clone
        self.assert_rejected(request)

    def test_interval_fit_validation_same_content_is_rejected(self):
        request = self.interval_request()
        clone = deepcopy(request["records"][0])
        clone.update(group_id="renamed_validation", record_id="renamed_validation",
                     dataset_role="adaptation_validation")
        clone["metadata"].update(record_id="renamed_validation", dataset_role="adaptation_validation")
        request["validation_records"] = [clone]
        self.assert_rejected(request)

    def test_interval_fit_and_validation_cannot_reuse_training_hashes(self):
        for role in ("records", "validation_records"):
            with self.subTest(role=role):
                request = self.interval_request()
                digest = request[role][0]["measurement_content_sha256"]
                self.bundle.metadata["samples"] = [{"record_id": "synthetic_training",
                                                    "measurement_content_sha256": digest}]
                self.assert_rejected(request)

    def test_interval_wrong_prediction_version_is_rejected(self):
        for role in ("records", "validation_records"):
            with self.subTest(role=role):
                request = self.interval_request()
                request[role][0]["prediction_version"] = "wrong_prediction_policy"
                self.assert_rejected(request)

    def test_interval_target_must_match_bound_measurement(self):
        request = self.interval_request()
        request["records"][0]["target_w"][0] += 10.
        self.assert_rejected(request)

    def test_interval_declared_measurement_hash_tampering_is_rejected(self):
        request = self.interval_request()
        request["records"][0]["aligned_measurement_sha256"] = "0" * 64
        self.assert_rejected(request)

    def test_interval_failed_validation_leaves_no_loadable_file(self):
        request = self.interval_request()
        request["validation_records"][0]["dataset_role"] = "independent_test"
        request["validation_records"][0]["metadata"]["dataset_role"] = "independent_test"
        self.assert_rejected(request)

    def test_interval_binds_condition_calibration_and_memory_prediction_version(self):
        condition = CalibrationResult(self.bundle.current_version(), ks_scale=1.05, residual_bias_w=2.)
        memory = PositionMemory(alpha=.4)
        request = self.interval_request(calibration=condition, memory=memory)
        expected = request["records"][0]["prediction_version"]
        self.assertNotEqual(expected, self.bundle.current_version())
        report = self.service.calibrate(request)
        self.assertEqual(report["status"], "independently_calibrated_prediction_interval")
        interval = IntervalCalibrator.load(Path(report["directory"]) / "interval.json")
        self.assertEqual(interval.model_version, expected)
        result = predict_cutting_power(self.bundle, frame().iloc[:3], context=self.metadata("use", "unused"),
                                       round_id=100, memory=memory, calibration=condition, interval_calibrator=interval)
        self.assertEqual(result["prediction_version"], expected)
        self.assertTrue(np.all(result["calibrated_lower_w"] <= result["calibrated_upper_w"]))
        with self.assertRaisesRegex(ValueError, "版本"):
            predict_cutting_power(self.bundle, frame().iloc[:3], context=self.metadata("use", "unused"),
                                  round_id=100, memory=PositionMemory(alpha=.3), calibration=condition,
                                  interval_calibrator=interval)


if __name__ == "__main__":
    unittest.main()
