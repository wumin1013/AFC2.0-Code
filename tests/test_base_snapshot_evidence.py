"""Exercise formal snapshot gates with temporary, explicitly synthetic evidence.

The fixture is not a scientific validation or a real model certificate. Nothing
is trained or published: synthetic=False only selects the formal validation
branch so copying and rechecking its certificate cannot be skipped by the test.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from verify_unified_load_prediction import synthetic_records
from spindle_power.load_prediction.base_power import BasePowerModel, CalibrationConfig
from spindle_power.load_prediction.common import file_hash, save_json
from spindle_power.load_prediction.training_service import TrainingService


class CopiedBaseSnapshotEvidenceTests(unittest.TestCase):
    def test_formal_gate_rechecks_copied_certificate_without_rereading_original(self):
        with tempfile.TemporaryDirectory(prefix="afc_synthetic_certificate_") as temporary:
            root = Path(temporary)
            originals = root / "synthetic_sources"
            originals.mkdir()
            model_path, report_path = originals / "model.json", originals / "report.json"
            provenance = "synthetic_fixture_only_no_real_validation_qualification"
            BasePowerModel(
                [0., 12000.], [0., 100.], 0., 0., .1, .1, 1000., 1000.,
                asdict(CalibrationConfig()), version="synthetic_base_v1",
                status="validated_in_sample_domain",
            ).save(model_path)
            domain = dict(machine_id="synthetic_machine", power_channel="spindle_power_w",
                          spindle_rpm_min=6000., spindle_rpm_max=10000.)
            save_json(report_path, dict(
                model_sha256=file_hash(model_path), model_version="synthetic_base_v1",
                mode="measured_speed_labels", domain=domain, status="validated",
                eligible_for_load_training=True, provenance=provenance,
            ))
            evidence = dict(
                validated=True, applicable=True, version="synthetic_base_v1",
                model_path=str(model_path), model_sha256=file_hash(model_path),
                report_path=str(report_path), report_sha256=file_hash(report_path),
                domain=domain, mode="measured_speed_labels", provenance=provenance,
            )
            evidence_before = deepcopy(evidence)
            feeds, loads = synthetic_records()
            from spindle_power.load_prediction.sample_rules import SAMPLE_RULES_VERSION
            from afc_common.process_path import PROCESS_MAPPING_RULES_VERSION
            for sample in feeds + loads:
                sample.metadata["machine_id"] = domain["machine_id"]
                sample.metadata.update(sample_rules_version=SAMPLE_RULES_VERSION,
                                       process_mapping_rules_version=PROCESS_MAPPING_RULES_VERSION)
            # Explicit synthetic diagnostics for the versioned consumer gate.
            # These fixture constants are not machining stability evidence.
            from spindle_power.load_prediction.main_stability import VERSION
            for sample in loads:
                sample.metadata["main_stability_rules_version"] = VERSION
                sample.frame["spindle_rate_rpm_s"] = 0.
                sample.frame["spindle_rate_support_interpolated"] = 1.
                sample.frame["spindle_rate_support_valid"] = sample.frame.aligned
                sample.frame["spindle_transition_guard_interpolated"] = 0.
                sample.frame["spindle_stable_interpolated"] = sample.frame.spindle_stable.astype(float)
            service = TrainingService(root / "service")
            prepared = service.save_prepared(
                feeds, loads, source_snapshot=dict(
                    files_verified=True, base_validation=evidence, provenance=provenance,
                ), synthetic=False,
            )
            directory = Path(prepared["directory"])
            _, _, snapshot = service.load_prepared(prepared["snapshot_id"])
            copied = snapshot["source_snapshot"]["base_validation"]
            self.assertFalse(snapshot["synthetic"])
            self.assertEqual(evidence, evidence_before)
            for kind in ("model", "report"):
                path = Path(copied[kind + "_path"])
                self.assertFalse(path.is_absolute())
                self.assertEqual(path.parent, Path("base_evidence"))
                self.assertEqual(file_hash(directory / path), evidence[kind + "_sha256"])

            # Changes outside the snapshot must not invalidate the frozen copy.
            model_path.write_text("synthetic original changed", encoding="utf-8")
            report_path.write_text("synthetic original changed", encoding="utf-8")
            self.assertEqual(len(service.preview(prepared["snapshot_id"])["phases"]), 9)

            # A hash-bound manifest alone is insufficient if its evidence file
            # can change. Both copied files must be rechecked before training.
            for kind in ("model", "report"):
                with self.subTest(copied_certificate=kind):
                    path = directory / copied[kind + "_path"]
                    contents = path.read_bytes()
                    try:
                        path.write_bytes(contents + b"\n ")
                        with self.assertRaisesRegex(ValueError, "SHA-256"):
                            service.preview(prepared["snapshot_id"])
                    finally:
                        path.write_bytes(contents)
            self.assertEqual(len(service.preview(prepared["snapshot_id"])["phases"]), 9)
            self.assertEqual(service.models(include_synthetic=True), [])
            self.assertFalse((service.output_dir / "runs").exists())


if __name__ == "__main__":
    unittest.main()
