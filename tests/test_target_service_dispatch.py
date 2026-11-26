"""Real snapshot/experiment dispatch with synthetic models and raw fixtures."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import test_target_experiment_adapters as target_fixtures
import test_dataset_manifest as manifest_fixtures
from spindle_power.load_prediction.dataset_manifest import inspect_manifest
from spindle_power.load_prediction.training_service import TrainingService


class TargetServiceDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = target_fixtures.TargetExperimentAdaptersTests
        cls.fixtures.setUpClass()

    @classmethod
    def tearDownClass(cls):
        cls.fixtures.tearDownClass()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.service = TrainingService(Path(temporary.name))
        fixture = self.fixtures
        self.source = self.save(fixture.feeds[:2], fixture.source)
        self.adaptation = self.save(fixture.feeds[2:4], fixture.adaptation)
        self.validation = self.save(fixture.feeds[4:5], fixture.validation)
        self.test = self.save(fixture.feeds[5:], fixture.test)

    def save(self, feeds, loads):
        return self.service.save_prepared(deepcopy(feeds), deepcopy(loads),
                    source_snapshot={"provenance": "synthetic_test_only"}, synthetic=True)["snapshot_id"]

    def request(self, name="four_parameter_calibration"):
        return dict(id=name, snapshot_id=self.source, model_id="synthetic-source",
                    target_adaptation_snapshot=self.adaptation,
                    adaptation_validation_snapshot=self.validation,
                    independent_test_snapshot=self.test,
                    target_budget={"records": 2}, execute=True,
                    training_config=asdict(self.fixtures.config))

    def registry(self):
        # Only registry lookup is substituted. Files, dispatcher and adapters run.
        return patch.object(self.service, "load_model", return_value=(
                    self.fixtures.feed, self.fixtures.bundle, {"synthetic": True}))

    def test_frozen_snapshots_reach_real_target_adapter_and_save_result(self):
        events = []
        before = self.fixtures.bundle.current_version()
        with self.registry() as registry:
            report = self.service.experiment(self.request(), progress=events.append)
        registry.assert_called_once_with("synthetic-source", allow_synthetic=True)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["result"]["applied_experiment"], "four_parameter_calibration")
        result = report["result"]
        self.assertEqual(result["counts"], {"source": 2, "adaptation": 2, "validation": 1, "test": 1})
        self.assertEqual(result["independent_test"]["feed_mode"], "predicted")
        self.assertFalse(result["test_used_for_selection"])
        self.assertTrue(result["calibration_version"])
        self.assertEqual(self.fixtures.bundle.current_version(), before)
        self.assertTrue((Path(result["artifact_directory"]) / "calibration.json").is_file())
        self.assertTrue((Path(result["artifact_directory"]) / "experiment.json").is_file())
        self.assertTrue(any(event.get("stage") == "target_model_frozen" for event in events))
        self.assertEqual(self.service.models(), [])

    def test_budget_mismatch_and_copied_content_cannot_reach_success(self):
        request = self.request()
        request["target_budget"] = {"records": 1}
        with self.registry(), self.assertRaisesRegex(ValueError, "target_budget.records"):
            self.service.experiment(request)
        copied_feed = deepcopy(self.fixtures.feeds[-1])
        copied_load = deepcopy(self.fixtures.test[0])
        copied_feed.frame = self.fixtures.feeds[2].frame.copy()
        copied_load.frame = self.fixtures.adaptation[0].frame.copy()
        digest = self.fixtures.adaptation[0].metadata["measurement_content_sha256"]
        copied_feed.metadata["measurement_content_sha256"] = digest
        copied_load.metadata["measurement_content_sha256"] = digest
        request = self.request()
        request["independent_test_snapshot"] = self.save([copied_feed], [copied_load])
        with self.registry(), self.assertRaisesRegex(ValueError, "hash重复"):
            self.service.experiment(request)
        self.assertEqual(self.service.models(), [])


class TargetManifestQualificationTests(unittest.TestCase):
    def test_selected_target_role_needs_real_files_but_not_450_development_slots(self):
        fixture = manifest_fixtures.DatasetManifestTests("test_base_claims_need_bound_label_mode_report")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        record = fixture.record(1, "外形")
        record["dataset_role"] = "target_adaptation"
        evidence = fixture.evidence()
        report = inspect_manifest([record], base_dir=fixture.root, selected_role="target_adaptation",
                                  base_validation=evidence)
        self.assertTrue(report["feed_eligible"], report["blockers"])
        self.assertTrue(report["load_eligible"], report["blockers"])
        self.assertEqual(report["development_records"], 0)
        self.assertEqual(report["missing_slots"], [])
        codes = {row["code"] for row in report["blockers"]}
        self.assertTrue({"empty_development", "missing_slots"}.isdisjoint(codes))
        unchecked = inspect_manifest([record], base_dir=fixture.root, selected_role="target_adaptation",
                                     verify_files=False, base_validation=evidence)
        self.assertFalse(unchecked["feed_eligible"])
        self.assertFalse(unchecked["load_eligible"])
        self.assertIn("files_not_verified", {row["code"] for row in unchecked["blockers"]})


if __name__ == "__main__":
    unittest.main()
