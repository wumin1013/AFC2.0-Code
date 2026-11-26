"""Synthetic certificates only; no artifact here qualifies real machine data."""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_base_timeline import fixture
from spindle_power.load_prediction.base_qualification import (
    APPLICABILITY_SCHEMA, BRIDGE_KIND, BRIDGE_SCHEMA, QUALIFICATION_SCHEMA,
    PAIRED_SETTINGS_APPLICABILITY_SCHEMA, PAIRED_SETTINGS_QUALIFICATION_SCHEMA,
    APPLICABILITY_CHECKS,
    embedded_json_artifact, verify_label_evidence)
from spindle_power.load_prediction.base_timeline import AUDITED_INTERFACE_VERSION, predict_audited_measured
from spindle_power.load_prediction.common import file_hash, save_json


class BridgeQualificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic_base_bridge_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.time, self.speed, self.model, self.timeline = fixture()
        self.raw = self.root / "synthetic_measurement.txt"
        self.raw.write_text("synthetic fixture, no actual machine data", encoding="utf-8")
        self.timeline = replace(self.timeline, record_source=str(self.raw))
        model_file = self.root / "model.json"
        self.model.save(model_file)
        self.domain = dict(machine_id="synthetic", power_channel="spindle_power_w",
                           machine_settings_version="synthetic-machine", measurement_settings_version="synthetic-scope",
                           spindle_rpm_min=3000., spindle_rpm_max=3020., initial_state="settled_with_observed_prior",
                           transitions=["settled0_to_target", "settled_target_to0"])
        self.context = dict(record_id="synthetic-record", machine_id="synthetic", dataset_role="base_validation",
                            machine_settings_version="synthetic-machine", measurement_settings_version="synthetic-scope",
                            resolved_files={"measurement": str(self.raw)}, file_sha256={"measurement": file_hash(self.raw)})
        freeze = dict(model_sha256=file_hash(model_file), development_rounds=[1, 2, 3], confirmation_rounds=[4, 5],
                      confirmation_target_power_examined=False)
        freeze_file = self.root / "freeze.json"
        save_json(freeze_file, freeze)
        confirmation = dict(frozen_manifest_sha256=file_hash(freeze_file), summary=dict(
            passed=True, event_count=2, failed_events=0, censored_or_invalid_events=0, excluded_events=0))
        confirmation_file = self.root / "confirmation.json"
        save_json(confirmation_file, confirmation)
        self.bridge = dict(schema=BRIDGE_SCHEMA, status="equivalent_in_audited_domain",
                           model_sha256=file_hash(model_file), model_version=self.model.version,
                           interface_version=AUDITED_INTERFACE_VERSION, freeze_sha256=file_hash(freeze_file),
                           confirmation_sha256=file_hash(confirmation_file),
                           production_source_sha256={name: file_hash(ROOT / "src/spindle_power/load_prediction" / name)
                                                     for name in ("base_events.py", "base_timeline.py")},
                           checks={key: True for key in (
                               "model_unchanged", "core_unchanged", "thresholds_unchanged",
                               "alignment_frozen_before_confirmation", "alignment_uses_no_power",
                               "event_curves_equivalent", "settling_grid_difference_explained",
                               "full_record_valid", "raw_inputs_unchanged")})
        self.applicability = dict(schema=APPLICABILITY_SCHEMA, status="verified", model_sha256=file_hash(model_file),
            interface_version=AUDITED_INTERFACE_VERSION, domain=self.domain,
            power_channel_identity_verified=True, machine_settings_compatible=True,
            measurement_settings_compatible=True, baseline_on_machining_idle_verified=True,
            record_audits=[dict(record_id=self.context["record_id"], measurement_sha256=file_hash(self.raw),
                initial_state_verified=True, timeline_verified=True, timeline_sha256=self.timeline.fingerprint(),
                program_version=self.timeline.program_version, time_speed_sha256=self.timeline.time_speed_sha256)])
        self.report = dict(schema=QUALIFICATION_SCHEMA, qualification_kind=BRIDGE_KIND,
            interface_version=AUDITED_INTERFACE_VERSION, new_independent_interface_confirmation=False,
            model_sha256=file_hash(model_file), model_version=self.model.version, mode="measured_speed_labels",
            domain=self.domain, status="validated", eligible_for_load_training=True,
            provenance="synthetic_fixture_only_no_real_machine_qualification",
            evidence_chain={"freeze": embedded_json_artifact(freeze_file),
                            "confirmation": embedded_json_artifact(confirmation_file)})
        self.evidence = dict(validated=True, applicable=True, version=self.model.version,
            model_path=str(model_file), model_sha256=file_hash(model_file), report_path=str(self.root / "report.json"),
            domain=self.domain, mode="measured_speed_labels")
        self.write_report()

    def write_report(self):
        for key, value in (("bridge", self.bridge), ("applicability", self.applicability)):
            path = self.root / f"{key}.json"
            save_json(path, value)
            self.report["evidence_chain"][key] = embedded_json_artifact(path)
        save_json(self.evidence["report_path"], self.report)
        self.evidence["report_sha256"] = file_hash(self.evidence["report_path"])

    def enable_paired_domains(self):
        self.domain.pop("machine_settings_version")
        self.domain.pop("measurement_settings_version")
        self.domain["settings_domains"] = [
            dict(settings_domain_id="controller-a", machine_settings_version="synthetic-machine",
                 measurement_settings_version="synthetic-scope"),
            dict(settings_domain_id="controller-b", machine_settings_version="synthetic-machine-b",
                 measurement_settings_version="synthetic-scope-b")]
        self.report["schema"] = PAIRED_SETTINGS_QUALIFICATION_SCHEMA
        self.applicability["schema"] = PAIRED_SETTINGS_APPLICABILITY_SCHEMA
        self.applicability["settings_audits"] = [
            {**value, **{key: True for key in APPLICABILITY_CHECKS}}
            for value in self.domain["settings_domains"]]
        self.applicability["record_audits"][0].update(self.domain["settings_domains"][0])
        second_raw = self.root / "synthetic_measurement_b.txt"
        second_raw.write_text("second synthetic record, no actual machine data", encoding="utf-8")
        self.second_timeline = replace(self.timeline, record_source=str(second_raw))
        self.second_context = {**self.context, **self.domain["settings_domains"][1],
            "record_id": "synthetic-record-b", "resolved_files": {"measurement": str(second_raw)},
            "file_sha256": {"measurement": file_hash(second_raw)}}
        self.applicability["record_audits"].append({**self.applicability["record_audits"][0],
            **self.domain["settings_domains"][1], "record_id": "synthetic-record-b",
            "measurement_sha256": file_hash(second_raw), "timeline_sha256": self.second_timeline.fingerprint()})
        self.write_report()

    def test_paired_domains_release_only_their_bound_records(self):
        self.enable_paired_domains()
        before = asdict(self.model)
        result = verify_label_evidence(self.evidence, [self.context, self.second_context])
        self.assertTrue(result["valid"], result["issues"])
        for context, timeline in ((self.context, self.timeline), (self.second_context, self.second_timeline)):
            prediction = predict_audited_measured(self.time, self.speed, self.model, timeline,
                qualification=self.evidence, record_context=context)
            self.assertTrue(prediction["formal_training_release"], prediction["qualification_issues"])
        self.assertEqual(asdict(self.model), before)

    def test_paired_domains_reject_unreviewed_cross_combinations(self):
        self.enable_paired_domains()
        for context in (
            {**self.context, "measurement_settings_version": "synthetic-scope-b"},
            {**self.second_context, "machine_settings_version": "synthetic-machine"},
            {**self.context, "machine_settings_version": "unknown-machine"},
        ):
            with self.subTest(context=context):
                self.assertFalse(verify_label_evidence(self.evidence, [context])["valid"])

    def test_paired_domains_require_record_and_group_specific_audits(self):
        self.enable_paired_domains()
        saved = deepcopy(self.applicability)
        mutations = [
            lambda a: a.pop("settings_audits"),
            lambda a: a["settings_audits"].pop(),
            lambda a: a["settings_audits"].append(deepcopy(a["settings_audits"][0])),
            lambda a: a["settings_audits"][1].update(machine_settings_version="synthetic-machine"),
            lambda a: a["record_audits"][1].pop("settings_domain_id"),
            lambda a: a["record_audits"][1].update(settings_domain_id="controller-a"),
            lambda a: a["record_audits"][1].pop("measurement_settings_version"),
        ]
        mutations.extend(lambda a, key=key: a["settings_audits"][1].update({key: None})
                         for key in APPLICABILITY_CHECKS)
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                self.applicability = deepcopy(saved)
                mutate(self.applicability)
                self.write_report()
                self.assertFalse(verify_label_evidence(self.evidence, [self.context, self.second_context])["valid"])

    def test_paired_domains_do_not_promote_diagnostic_or_break_evidence_chain(self):
        self.enable_paired_domains()
        saved_report, saved_applicability, saved_bridge = map(deepcopy, (self.report, self.applicability, self.bridge))
        for change in ("diagnostic", "no_confirmation", "changed_code", "unknown_initial", "changed_timeline"):
            with self.subTest(change=change):
                self.report, self.applicability, self.bridge = map(deepcopy, (saved_report, saved_applicability, saved_bridge))
                if change == "diagnostic":
                    self.applicability["status"] = "blocked"
                elif change == "no_confirmation":
                    self.report["evidence_chain"].pop("confirmation")
                elif change == "changed_code":
                    self.bridge["production_source_sha256"]["base_timeline.py"] = "0" * 64
                elif change == "unknown_initial":
                    self.applicability["record_audits"][1]["initial_state_verified"] = None
                else:
                    self.applicability["record_audits"][1]["timeline_sha256"] = "0" * 64
                self.write_report()
                result = predict_audited_measured(self.time, self.speed, self.model, self.second_timeline,
                    qualification=self.evidence, record_context=self.second_context)
                self.assertFalse(result["formal_training_release"])

    def test_paired_domain_declarations_are_explicit_and_unique(self):
        self.enable_paired_domains()
        saved = deepcopy(self.domain)
        mutations = [
            lambda d: d.update(settings_domains=[]),
            lambda d: d["settings_domains"].append(deepcopy(d["settings_domains"][0])),
            lambda d: d["settings_domains"][1].update(settings_domain_id="controller-a"),
            lambda d: d["settings_domains"][1].update(machine_settings_version="synthetic-machine",
                                                      measurement_settings_version="synthetic-scope"),
            lambda d: d["settings_domains"][0].update(measurement_settings_version="*"),
            lambda d: d["settings_domains"][0].update(machine_settings_version=["a", "b"]),
            lambda d: d.update(machine_settings_version="synthetic-machine"),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                changed = deepcopy(saved)
                mutate(changed)
                self.domain = changed
                self.evidence["domain"] = self.report["domain"] = self.applicability["domain"] = changed
                self.write_report()
                self.assertFalse(verify_label_evidence(self.evidence, [self.context, self.second_context])["valid"])

    def test_paired_domain_schema_and_applicability_domain_cannot_be_mixed(self):
        self.enable_paired_domains()
        self.report["schema"] = QUALIFICATION_SCHEMA
        self.write_report()
        self.assertFalse(verify_label_evidence(self.evidence, [self.context])["valid"])
        self.report["schema"] = PAIRED_SETTINGS_QUALIFICATION_SCHEMA
        self.applicability["schema"] = APPLICABILITY_SCHEMA
        self.write_report()
        self.assertFalse(verify_label_evidence(self.evidence, [self.context])["valid"])
        self.applicability["schema"] = PAIRED_SETTINGS_APPLICABILITY_SCHEMA
        self.applicability["domain"] = deepcopy(self.domain)
        self.applicability["domain"]["settings_domains"].pop()
        self.write_report()
        self.assertFalse(verify_label_evidence(self.evidence, [self.context])["valid"])

    def test_paired_evidence_survives_existing_snapshot_copy_and_reload(self):
        from spindle_power.load_prediction.training_service import TrainingService
        self.enable_paired_domains()
        service = TrainingService(self.root / "service")
        prepared = service.save_prepared([], [], source_snapshot={"base_validation": self.evidence})
        _, _, snapshot = service.load_prepared(prepared["snapshot_id"])
        frozen = snapshot["source_snapshot"]["base_validation"]
        for name in ("model", "report"):
            Path(self.evidence[name + "_path"]).unlink()
        result = verify_label_evidence(frozen, [self.context, self.second_context], base_dir=prepared["directory"])
        self.assertTrue(result["valid"], result["issues"])

    def test_full_evidence_qualifies_only_exact_record_and_preserves_model(self):
        before = asdict(self.model)
        self.assertTrue(verify_label_evidence(self.evidence, [self.context])["valid"])
        result = predict_audited_measured(self.time, self.speed, self.model, self.timeline,
            qualification=self.evidence, record_context=self.context)
        self.assertTrue(result["formal_training_release"], result["qualification_issues"])
        self.assertEqual(result["status"], "validated_in_sample_domain")
        self.assertEqual(asdict(self.model), before)
        default = predict_audited_measured(self.time, self.speed, self.model, self.timeline)
        self.assertFalse(default["formal_training_release"])

    def test_replay_alone_or_unreviewed_applicability_cannot_release(self):
        for change in ("baseline_on_machining_idle_verified", "machine_settings_compatible",
                       "power_channel_identity_verified", "measurement_settings_compatible"):
            with self.subTest(change=change):
                self.applicability[change] = None
                self.write_report()
                self.assertFalse(verify_label_evidence(self.evidence, [self.context])["valid"])
                self.applicability[change] = True
        self.report["evidence_chain"].pop("confirmation")
        self.write_report()
        self.assertFalse(verify_label_evidence(self.evidence, [self.context])["valid"])

    def test_engineering_status_not_a_formal_certificate_and_code_binding_required(self):
        self.bridge["status"] = "postconfirmation_production_interface_engineering_validation"
        self.write_report()
        self.assertFalse(verify_label_evidence(self.evidence, [self.context])["valid"])
        self.bridge["status"] = "equivalent_in_audited_domain"
        self.bridge["production_source_sha256"]["base_events.py"] = "0" * 64
        self.write_report()
        self.assertFalse(verify_label_evidence(self.evidence, [self.context])["valid"])

    def test_record_or_model_or_timeline_tampering_does_not_inherit_qualification(self):
        for model, timeline, context in (
            (replace(self.model, status="validated_in_sample_domain"), self.timeline, self.context),
            (self.model, replace(self.timeline, program_version="changed"), self.context),
            (self.model, self.timeline, {**self.context, "machine_settings_version": "changed"}),
        ):
            with self.subTest(context=context, model=model.version):
                result = predict_audited_measured(self.time, self.speed, model, timeline,
                    qualification=self.evidence, record_context=context)
                self.assertFalse(result["formal_training_release"])
        self.raw.write_text("changed raw fixture", encoding="utf-8")
        result = predict_audited_measured(self.time, self.speed, self.model, self.timeline,
            qualification=self.evidence, record_context=self.context)
        self.assertFalse(result["formal_training_release"])

    def test_embedded_evidence_survives_original_deletion_and_rejects_text_tampering(self):
        for name in ("freeze", "confirmation", "bridge", "applicability"):
            (self.root / f"{name}.json").unlink()
        self.assertTrue(verify_label_evidence(self.evidence, [self.context])["valid"])
        self.report["evidence_chain"]["freeze"]["json_text"] += " "
        save_json(self.evidence["report_path"], self.report)
        self.evidence["report_sha256"] = file_hash(self.evidence["report_path"])
        self.assertFalse(verify_label_evidence(self.evidence, [self.context])["valid"])


if __name__ == "__main__":
    unittest.main()
