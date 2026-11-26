"""Synthetic manifest evidence: no real machining files or neural training."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spindle_power.load_prediction.common import file_hash
from spindle_power.load_prediction.dataset_manifest import (inspect_manifest, verify_snapshot,
    measurement_content_hash, activate_round_replacement, activate_record_replacement, verify_base_evidence)
from spindle_power.load_prediction.experiment_protocol import PROGRAM_FEATURES, round_spec
from spindle_power.load_prediction.feed import MachineConfig, parse_nc


class DatasetManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.machine = MachineConfig(initial_position_mm=[0., 0., 0.], work_offsets_mm={"G54": [0., 0., 0.]})
        (self.root / "machine.json").write_text(json.dumps(asdict(self.machine)), encoding="utf-8")
        (self.root / "program.nc").write_text("G1 X1 F2000 S8000\n", encoding="utf-8")
        (self.root / "process.csv").write_text("line,execution_instance,segment_mm,ap_mm,ae_mm,steady\n1,0,0,1,1,True\n1,0,1,1,1,True\n", encoding="utf-8")
        self.program_version = parse_nc(self.root / "program.nc", self.machine)["program_version"]

    def record(self, round_id, feature, *, serial=None):
        serial = serial if serial is not None else round_id * 10 + PROGRAM_FEATURES.index(feature)
        path = self.root / f"measurement-{serial}.csv"
        channels = [("程序行号", "AX", ""), ("实际位置", "X轴", ""), ("实际位置", "Y轴", ""),
                    ("实际位置", "Z轴", ""), ("实际进给", "AX", ""),
                    ("G寄存器", "X轴", "432"), ("实际速度", "SP轴", "")]
        header = ["<SmplInfo>,<0>,<1000>"] + [f"<ChannelInfo>,<{i}>,<0>,<{s}>,<0>,<{a}>,<{e}>" for i, (s, a, e) in enumerate(channels)]
        path.write_text("\n".join(header + [f"<ChannelData>,1,0,0,0,1000,{100 + serial},8000,",
                                             f"<ChannelData>,1,1,0,0,1000,{101 + serial},8000,"]), encoding="utf-8")
        return dict(record_id=f"r{round_id}-{feature}-{serial}", **{k: v for k, v in round_spec(round_id).items() if k != "program_slots"},
                    feature=feature, program_id=feature, program_version=self.program_version,
                    nc="program.nc", measurement=path.name, machine="machine.json", process="process.csv",
                    machine_id="M", machine_settings_version="settings", machining_mode="normal", feed_override=1.,
                    dataset_role="development", complete=True, round_complete=True, tool_healthy=True,
                    alignment_reliable=True, no_alarm=True, process_aligned=True, process_version="process-v1",
                    tool_session="session-1", actual_event_order=serial, acquired_at="2026-10-09T10:00:00+09:00")

    def evidence(self):
        from spindle_power.load_prediction.base_power import BasePowerModel, CalibrationConfig
        model = self.root / "base.json"
        BasePowerModel([0., 12000.], [0., 100.], 0., 0., .1, .1, 1000., 1000.,
                       asdict(CalibrationConfig()), version="base-v1", status="validated_in_sample_domain").save(model)
        domain = dict(machine_id="M", power_channel="spindle_power_w", spindle_rpm_min=6000., spindle_rpm_max=10000.)
        report = dict(model_sha256=file_hash(model), model_version="base-v1", mode="measured_speed_labels",
                      domain=domain, status="validated", eligible_for_load_training=True)
        path = self.root / "base-report.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return dict(validated=True, applicable=True, version="base-v1", model_path=str(model), model_sha256=file_hash(model),
                    report_path=str(path), report_sha256=file_hash(path), domain=domain, mode="measured_speed_labels")

    def test_450_slots_same_nc_is_allowed_and_unknown_base_only_blocks_load(self):
        records = [self.record(r, f) for r in range(1, 91) for f in PROGRAM_FEATURES]
        report = inspect_manifest(records, base_dir=self.root)
        self.assertTrue(report["feed_eligible"], report["blockers"][:5])
        self.assertFalse(report["load_eligible"])
        self.assertEqual(report["missing_slots"], [])
        self.assertEqual(report["development_records"], 450)
        self.assertTrue(verify_snapshot(report["snapshot"], snapshot_id=report["snapshot_id"])["valid"])
        report = inspect_manifest(records, base_dir=self.root, base_validation=self.evidence())
        self.assertTrue(report["load_eligible"], report["blockers"][:5])

    def test_copied_content_with_new_header_not_an_independent_experiment(self):
        a, b = self.record(1, "外形"), self.record(1, "曲面")
        p, q = self.root / a["measurement"], self.root / b["measurement"]
        q.write_text("new export header\n" + p.read_text(encoding="utf-8"), encoding="utf-8")
        self.assertNotEqual(file_hash(p), file_hash(q))
        self.assertEqual(measurement_content_hash(p), measurement_content_hash(q))
        report = inspect_manifest([a, b], base_dir=self.root, require_complete=False)
        self.assertIn("duplicate_measurement", {i["code"] for i in report["blockers"]})

    def test_geometry_cache_does_not_mix_coordinate_modes(self):
        from unittest.mock import patch
        records = [self.record(1, f) for f in PROGRAM_FEATURES]
        records[0]["coordinate_mode"] = "workpiece_cutting"
        # This NC omits its starting workpiece coordinates: machine mode can
        # use the configured initial position, but cutting mode must reject it.
        for ordered in (records, list(reversed(records))):
            with self.subTest(first_mode=ordered[0].get("coordinate_mode", "machine")):
                with patch("spindle_power.load_prediction.feed.parse_nc", wraps=parse_nc) as parser:
                    report = inspect_manifest(ordered, base_dir=self.root, require_complete=False)
                geometry_failures = [b["record_id"] for b in report["blockers"] if b["code"] == "nc_geometry"]
                self.assertEqual(geometry_failures, [records[0]["record_id"]])
                self.assertEqual(parser.call_count, 2)

    def test_roles_and_unknown_reviews_are_never_implicitly_passed(self):
        records = [self.record(1, f) for f in PROGRAM_FEATURES]
        records[0].pop("tool_healthy")
        held = self.record(2, "外形")
        held["dataset_role"] = "independent_test"
        report = inspect_manifest(records + [held], base_dir=self.root, require_complete=False)
        self.assertFalse(report["feed_eligible"])
        codes = {i["code"] for i in report["blockers"]}
        self.assertTrue({"program_role_leakage", "quality_unknown"} <= codes)

    def test_snapshot_refuses_changed_files_or_metadata(self):
        records = [self.record(1, f) for f in PROGRAM_FEATURES]
        report = inspect_manifest(records, base_dir=self.root, require_complete=False)
        altered = deepcopy(report["snapshot"])
        altered["records"][0]["tool_session"] = "other"
        self.assertFalse(verify_snapshot(altered, snapshot_id=report["snapshot_id"])["valid"])
        with (self.root / records[0]["measurement"]).open("a", encoding="utf-8") as stream:
            stream.write("\nchanged")
        self.assertFalse(verify_snapshot(report["snapshot"])["valid"])

    def test_replacement_is_pure_and_requires_all_five_checked_slots(self):
        records = [self.record(1, f) for f in PROGRAM_FEATURES]
        replacements = [self.recollection(old, serial=1000 + i) for i, old in enumerate(records)]
        before = deepcopy(records)
        with self.assertRaisesRegex(ValueError, "五类"):
            activate_round_replacement(records, replacements[:4], base_dir=self.root)
        self.assertEqual(records, before)
        result = activate_round_replacement(records, replacements, base_dir=self.root)
        self.assertEqual(records, before)
        self.assertEqual(sum(r.get("status", "active") == "active" for r in result), 5)
        self.assertEqual(sum(r.get("status") == "superseded" for r in result), 5)
        for old, new in zip(result[:5], result[5:]):
            self.assertEqual(new["replaces"], [old["record_id"]])
            self.assertEqual(old["superseded_by"], [new["record_id"]])
            self.assertTrue(new["recollection"]["round_review_snapshot_id"])
            self.assertIsNone(new["actual_event_order"])
            self.assertFalse(new["memory_eligible"])

    def bind_numeric_receipt(self, record):
        path = self.root / (record["record_id"] + "-numeric.json")
        raw = self.root / record["measurement"]
        # Synthetic receipt: these tiny fixtures only test hash binding, not the scanner.
        path.write_text(json.dumps(dict(record_id=record["record_id"], file_sha256=file_hash(raw),
            content_sha256=measurement_content_hash(raw),
            content_hash_schema="canonical_channel_float64_sha256_v1")), encoding="utf-8")
        record["review_evidence"] = {"numeric": dict(path=str(path), sha256=file_hash(path))}
        return path

    def test_round_recollection_retains_bound_history_when_original_paths_were_overwritten(self):
        records = [self.record(74, f) for f in PROGRAM_FEATURES]
        other = self.record(73, "外形")
        replacements = [self.recollection(old, serial=2100 + i) for i, old in enumerate(records)]
        for old, new in zip(records, replacements):
            self.bind_numeric_receipt(old)
            self.bind_numeric_receipt(new)
            (self.root / old["measurement"]).write_bytes((self.root / new["measurement"]).read_bytes())
        before = deepcopy(records + [other])
        result = activate_round_replacement(records + [other], replacements, base_dir=self.root)
        self.assertEqual(records + [other], before)
        self.assertEqual(result[5], other)
        for old in result[:5]:
            evidence = old["source_availability"]
            self.assertFalse(evidence["original_bytes_available_at_path"])
            self.assertNotEqual(evidence["expected_file_sha256"], evidence["observed_file_sha256"])
            self.assertEqual(old["review_evidence"], before[result.index(old)]["review_evidence"])
        self.assertTrue(verify_snapshot(inspect_manifest(result[:5] + result[6:],
            base_dir=self.root, require_complete=False)["snapshot"])["valid"])

    def test_round_recollection_rejects_wrong_slot_nc_quality_or_inherited_date(self):
        records = [self.record(74, f) for f in PROGRAM_FEATURES]
        replacements = [self.recollection(old, serial=2200 + i) for i, old in enumerate(records)]
        wrong_nc = self.root / "other.nc"
        wrong_nc.write_text("G1 X99 F2000 S10000\n", encoding="utf-8")
        cases = [("feature", replacements[1]["feature"], "五类"),
                 ("record_id", replacements[1]["record_id"], "唯一"),
                 ("replaces", [records[1]["record_id"]], "feature"),
                 ("nc", wrong_nc.name, "预定NC"),
                 ("complete", None, "未通过"),
                 ("acquired_at", records[0]["acquired_at"], "不得继承")]
        for field, value, expected in cases:
            candidate = deepcopy(replacements)
            candidate[0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, expected):
                activate_round_replacement(records, candidate, base_dir=self.root)

    def test_round_recollection_rejects_unbound_overwrite_and_old_copied_content(self):
        records = [self.record(74, f) for f in PROGRAM_FEATURES]
        replacements = [self.recollection(old, serial=2300 + i) for i, old in enumerate(records)]
        for old, new in zip(records, replacements):
            self.bind_numeric_receipt(old)
            self.bind_numeric_receipt(new)
        old_bytes = (self.root / records[0]["measurement"]).read_bytes()
        (self.root / records[0]["measurement"]).write_bytes((self.root / replacements[0]["measurement"]).read_bytes())
        forged = deepcopy(records)
        forged[0]["review_evidence"]["numeric"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "回执哈希"):
            activate_round_replacement(forged, replacements, base_dir=self.root)
        (self.root / replacements[0]["measurement"]).write_bytes(old_bytes)
        self.bind_numeric_receipt(replacements[0])
        with self.assertRaisesRegex(ValueError, "重复"):
            activate_round_replacement(records, replacements, base_dir=self.root)

    def test_round_recollection_does_not_accept_overwrite_in_unrelated_round(self):
        records = [self.record(74, f) for f in PROGRAM_FEATURES]
        other = self.record(73, "外形")
        replacements = [self.recollection(old, serial=2400 + i) for i, old in enumerate(records)]
        for record in records + [other] + replacements:
            self.bind_numeric_receipt(record)
        with (self.root / other["measurement"]).open("a", encoding="utf-8") as stream:
            stream.write("\nchanged externally")
        with self.assertRaisesRegex(ValueError, "历史实测源已改变"):
            activate_round_replacement(records + [other], replacements, base_dir=self.root)

    def recollection(self, old, *, serial=2000):
        replacement = self.record(old["round_id"], old["feature"], serial=serial)
        evidence = self.root / f"recollection-{serial}.txt"
        evidence.write_text("Synthetic acquisition evidence for interface tests only.\n", encoding="utf-8")
        replacement.update(replaces=[old["record_id"]], acquired_at="2026-10-10T10:00:00+09:00",
                           program_slot=round_spec(old["round_id"])["program_slots"][old["feature"]],
                           recollection=dict(reason="synthetic recollection test", evidence_path=evidence.name))
        replacement.pop("actual_event_order")
        replacement.pop("tool_session")
        return replacement

    def test_single_recollection_preserves_old_facts_and_unknown_chronology(self):
        records = [self.record(74, f) for f in PROGRAM_FEATURES]
        replacement = self.recollection(records[-1])
        before, candidate_before = deepcopy(records), deepcopy(replacement)
        old_bytes = (self.root / records[-1]["measurement"]).read_bytes()
        result = activate_record_replacement(records, replacement, base_dir=self.root)
        self.assertEqual(records, before)
        self.assertEqual(replacement, candidate_before)
        self.assertEqual(result[:-2], records[:-1])
        self.assertEqual(result[-2], {**records[-1], "status": "superseded", "superseded_by": [replacement["record_id"]]})
        self.assertEqual((self.root / records[-1]["measurement"]).read_bytes(), old_bytes)
        self.assertEqual(sum(r.get("status", "active") == "active" for r in result), 5)
        new = result[-1]
        self.assertEqual(new["replaces"], [records[-1]["record_id"]])
        self.assertEqual(new["recollection"]["expected_program_slot"], "T2")
        self.assertEqual(new["recollection"]["evidence_sha256"], file_hash(self.root / replacement["recollection"]["evidence_path"]))
        self.assertIsNone(new["actual_event_order"])
        self.assertIsNone(new["tool_session"])
        self.assertFalse(new["memory_eligible"])
        report = inspect_manifest(result, base_dir=self.root, require_complete=False, base_validation=self.evidence())
        self.assertTrue(report["feed_eligible"], report["blockers"])
        self.assertTrue(report["load_eligible"], report["blockers"])
        self.assertFalse(report["memory_eligible"])

    def test_single_recollection_refuses_wrong_target_or_inherited_acquisition(self):
        records = [self.record(74, f) for f in PROGRAM_FEATURES]
        replacement = self.recollection(records[-1])
        cases = [
            ({"round_id": 73}, "round_id"), ({"feature": "外形"}, "feature"),
            ({"machine_id": "another-machine"}, "固定工况"),
            ({"program_slot": "T1"}, "program_slot"), ({"replaces": []}, "replaces"),
            ({"record_id": records[-1]["record_id"]}, "record_id"),
            ({"acquired_at": None}, "acquired_at"), ({"acquired_at": "tomorrow"}, "acquired_at"),
            ({"acquired_at": records[-1]["acquired_at"]}, "不得继承"),
            ({"acquired_at": "2026-10-09T01:00:00+00:00"}, "不得继承"),
            ({"acquired_at": "2026-10-08"}, "早于"),
            ({"tool_session": records[-1]["tool_session"]}, "不得继承"),
            ({"actual_event_order": records[-1]["actual_event_order"]}, "不得继承"),
            ({"actual_event_order": 73}, "追加"),
            ({"recollection": {"reason": "test", "evidence_path": "absent.txt"}}, "文件不存在"),
        ]
        before = deepcopy(records)
        for changes, error in cases:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, error):
                activate_record_replacement(records, {**replacement, **changes}, base_dir=self.root)
            self.assertEqual(records, before)

    def test_single_recollection_recomputes_round_complete_from_reviewed_slots(self):
        records = [self.record(74, f) for f in PROGRAM_FEATURES]
        for record in records:
            record["round_complete"] = False
        records[-1]["complete"] = False
        replacement = self.recollection(records[-1])
        replacement["round_complete"] = False
        before = deepcopy(records)
        result = activate_record_replacement(records, replacement, base_dir=self.root)
        active = [r for r in result if r.get("status", "active") == "active"]
        self.assertEqual(len(active), 5)
        self.assertTrue(all(r["round_complete"] for r in active))
        self.assertTrue(all(r["round_completeness_evidence"]["recomputed_after_replacement"] == replacement["record_id"] for r in active))
        self.assertEqual(records, before)
        # A newly acquired slot cannot repair another independently incomplete
        # record, even when a caller claims that the round is now complete.
        records[0]["complete"] = None
        replacement["round_complete"] = True
        with self.assertRaisesRegex(ValueError, "同轮五类检查"):
            activate_record_replacement(records, replacement, base_dir=self.root)

    def test_single_recollection_does_not_inherit_a_claimed_memory_release(self):
        records = [self.record(1, f) for f in PROGRAM_FEATURES]
        replacement = self.recollection(records[-1])
        replacement.update(actual_event_order=2000, tool_session="new-confirmed-session", memory_eligible=True)
        result = activate_record_replacement(records, replacement, base_dir=self.root)
        self.assertNotIn("memory_eligible", result[-1])
        unqualified = inspect_manifest(result, base_dir=self.root, require_complete=False)
        self.assertFalse(unqualified["memory_eligible"])
        qualified = inspect_manifest(result, base_dir=self.root, require_complete=False, base_validation=self.evidence())
        self.assertTrue(qualified["memory_eligible"], qualified["blockers"])

    def test_single_recollection_refuses_wrong_nc_bad_quality_or_incomplete_round(self):
        records = [self.record(74, f) for f in PROGRAM_FEATURES]
        replacement = self.recollection(records[-1])
        wrong_nc = self.root / "wrong.nc"
        wrong_nc.write_text("G1 X2 F2000 S8000\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "预定NC"):
            activate_record_replacement(records, {**replacement, "nc": wrong_nc.name}, base_dir=self.root)
        with self.assertRaisesRegex(ValueError, "同轮五类检查"):
            activate_record_replacement(records, {**replacement, "alignment_reliable": None}, base_dir=self.root)
        with self.assertRaisesRegex(ValueError, "同轮五类检查"):
            activate_record_replacement(records[1:], replacement, base_dir=self.root)
        self.assertTrue(all(r.get("status", "active") == "active" for r in records))

    def test_single_recollection_refuses_copied_history_including_superseded_content(self):
        records = [self.record(74, f) for f in PROGRAM_FEATURES]
        old_history = self.record(1, "台阶", serial=3000)
        old_history["status"] = "superseded"
        records.append(old_history)
        replacement = self.recollection(records[4])
        for old in (records[4], old_history):
            with self.subTest(old=old["record_id"]):
                (self.root / replacement["measurement"]).write_text(
                    "changed export header\n" + (self.root / old["measurement"]).read_text(encoding="utf-8"), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "重复"):
                    activate_record_replacement(records, replacement, base_dir=self.root)
        with self.assertRaisesRegex(ValueError, "原始文件"):
            activate_record_replacement(records, {**replacement, "measurement": records[4]["measurement"]}, base_dir=self.root)

    def test_base_claims_need_bound_label_mode_report(self):
        evidence = self.evidence()
        self.assertTrue(verify_base_evidence(evidence)["valid"])
        evidence["mode"] = "events"
        self.assertFalse(verify_base_evidence(evidence)["valid"])
        evidence = self.evidence()
        Path(evidence["model_path"]).write_text("changed", encoding="utf-8")
        self.assertFalse(verify_base_evidence(evidence)["valid"])
        self.assertFalse(verify_base_evidence(dict(validated=True, applicable=True, version="fake"))["valid"])

    def test_unknown_chronology_only_blocks_memory_without_fabricating_context(self):
        records = [self.record(1, feature) for feature in PROGRAM_FEATURES]
        for record in records:
            for key in ("actual_event_order", "acquired_at", "tool_session"):
                record.pop(key)
        report = inspect_manifest(records, base_dir=self.root, require_complete=False,
                                  base_validation=self.evidence())
        self.assertTrue(report["feed_eligible"], report["blockers"])
        self.assertTrue(report["load_eligible"], report["blockers"])
        self.assertFalse(report["memory_eligible"])
        self.assertEqual({b["scope"] for b in report["blockers"]}, {"memory"})
        for record in report["records"]:
            self.assertTrue(all(key not in record for key in ("actual_event_order", "acquired_at", "tool_session")))
        records[0].pop("machine_settings_version")
        report = inspect_manifest(records, base_dir=self.root, require_complete=False,
                                  base_validation=self.evidence())
        self.assertFalse(report["feed_eligible"])

    def test_recorded_command_positions_can_qualify_with_real_feed(self):
        records = [self.record(1, feature) for feature in PROGRAM_FEATURES]
        for record in records:
            path = self.root / record["measurement"]
            path.write_text(path.read_text(encoding="utf-8").replace("<实际位置>", "<指令位置>"), encoding="utf-8")
        report = inspect_manifest(records, base_dir=self.root, require_complete=False,
                                  base_validation=self.evidence())
        self.assertTrue(report["feed_eligible"], report["blockers"])
        self.assertTrue(report["load_eligible"], report["blockers"])
        self.assertEqual({record["position_source"] for record in report["snapshot"]["records"]}, {"command"})
        records[0].pop("alignment_reliable")
        unknown = inspect_manifest(records, base_dir=self.root, require_complete=False)
        self.assertFalse(unknown["feed_eligible"])
        self.assertTrue(any(item["code"] == "quality_unknown" for item in unknown["blockers"]))

    def test_explicit_command_policy_is_preserved_and_requires_command_channels(self):
        from unittest.mock import patch
        from spindle_power.load_prediction.measurement import read_measurement
        records = [self.record(1, feature) for feature in PROGRAM_FEATURES]
        for record in records:
            record["position_policy"] = "command_only_v1"
        def both_groups(path, **kwargs):
            measurement = read_measurement(path, **kwargs)
            for axis in "xyz":
                measurement.values[f"command_{axis}_mm"] = measurement.values[f"actual_{axis}_mm"].copy()
            return measurement
        with patch("spindle_power.load_prediction.measurement.read_measurement", side_effect=both_groups):
            report = inspect_manifest(records, base_dir=self.root, require_complete=False,
                                      base_validation=self.evidence())
        self.assertTrue(report["load_eligible"], report["blockers"])
        self.assertEqual({row["position_policy"] for row in report["snapshot"]["records"]}, {"command_only_v1"})
        self.assertEqual({row["position_source"] for row in report["snapshot"]["records"]}, {"command"})
        absent = inspect_manifest(records, base_dir=self.root, require_complete=False)
        self.assertFalse(absent["feed_eligible"])
        self.assertEqual(sum(row["code"] == "alignment_channels" for row in absent["blockers"]), len(records))

    def test_partial_actual_and_command_groups_cannot_be_mixed(self):
        records = [self.record(1, feature) for feature in PROGRAM_FEATURES]
        path = self.root / records[0]["measurement"]
        path.write_text(path.read_text(encoding="utf-8").replace(
            "<实际位置>,<0>,<Y轴>", "<指令位置>,<0>,<Y轴>"), encoding="utf-8")
        report = inspect_manifest(records, base_dir=self.root, require_complete=False)
        self.assertFalse(report["feed_eligible"])
        self.assertTrue(any(item["code"] == "alignment_channels" for item in report["blockers"]))


if __name__ == "__main__":
    unittest.main()
