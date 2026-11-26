"""Temporary synthetic certificates exercise qualification, never real release."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spindle_power.load_prediction.base_power import BasePowerModel, CalibrationConfig, predict_base_from_commands
from spindle_power.load_prediction.common import file_hash, save_json
from spindle_power.load_prediction.total_power import add_total_power


class TotalPowerQualificationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="afc_synthetic_total_power_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.frame = pd.DataFrame(dict(line=[10, 10, 10], execution_instance=0,
                                       segment_mm=[0., .5, 1.], spindle_rpm=2000.))
        self.result = dict(cutting_quantiles_w=np.array([[1., 3., 5.], [2., 4., 6.], [-5., -3., -1.]]),
                           lower_w=np.array([0., 1., -6.]), upper_w=np.array([6., 7., 0.]))
        self.known = dict(machine_id="synthetic_machine", power_channel="spindle_power_w",
                          machine_settings_version="synthetic_settings", spindle_stable_before_cut=True)

    def certificate(self, mode="steady_precutting", *, status="validated_in_sample_domain"):
        model = BasePowerModel([0., 1000., 2000., 10000.], [0., 10., 20., 100.],
                               .05, .08, .1, .1, 1000., 1000., asdict(CalibrationConfig()),
                               version="synthetic_base_only", status=status)
        model_path, report_path = self.root / "model.json", self.root / "report.json"
        model.save(model_path)
        domain = {k: self.known[k] for k in ("machine_id", "power_channel", "machine_settings_version")}
        domain.update(spindle_rpm_min=0., spindle_rpm_max=10000.)
        report = dict(model_sha256=file_hash(model_path), model_version=model.version, mode=mode,
                      domain=domain, status="validated", eligible_for_steady_prediction=True,
                      eligible_for_dynamic_prediction=True, provenance="synthetic_test_only_not_real_qualification")
        save_json(report_path, report)
        evidence = dict(model_path=str(model_path), model_sha256=file_hash(model_path),
                        report_path=str(report_path), report_sha256=file_hash(report_path),
                        version=model.version, domain=domain, mode=mode, validated=True, applicable=True)
        return dict(base_model_path=str(model_path), base_validation=evidence), model

    def dynamic_request(self):
        request, model = self.certificate("commanded_dynamic")
        time = np.arange(41) * .05
        timeline = pd.DataFrame(dict(time_s=time, commands_rpm=np.where(time < 1., 1000., 2000.)))
        mapping = self.frame[["line", "execution_instance", "segment_mm"]].copy()
        mapping["time_s"] = [1.1, 1.5, 1.9]
        path, mapping_path = self.root / "commands.csv", self.root / "mapping.csv"
        timeline.to_csv(path, index=False, encoding="utf-8")
        mapping.to_csv(mapping_path, index=False, encoding="utf-8")
        request.update(command_timeline=str(path), command_timeline_sha256=file_hash(path),
                       time_mapping=str(mapping_path), time_mapping_sha256=file_hash(mapping_path),
                       complete_nc_timeline_verified=True, initial_state_verified=True, initial_speed_rpm=1000.)
        return request, model, timeline, mapping

    def unavailable(self, request, *, frame=None, known=None, contains=None):
        result = add_total_power(self.result, self.frame if frame is None else frame,
                                 request, self.known if known is None else known)
        self.assertEqual(result["total_power_status"], "total_power_unavailable")
        self.assertNotIn("total_power_w", result)
        self.assertNotIn("base_power_w", result)
        np.testing.assert_array_equal(result["cutting_quantiles_w"], self.result["cutting_quantiles_w"])
        if contains:
            self.assertIn(contains, result["total_power_reason"])
        return result

    def test_default_retains_cutting_prediction_but_has_no_total_power(self):
        self.unavailable({})
        stale = {**self.result, "base_power_w": np.ones(3), "total_power_w": np.ones(3)}
        output = add_total_power(stale, self.frame, {}, self.known)
        self.assertNotIn("base_power_w", output)
        self.assertNotIn("total_power_w", output)

    def test_steady_interpolation_adds_all_quantiles_and_effective_interval(self):
        request, _ = self.certificate()
        self.result.update(calibrated_lower_w=np.array([-1., 0., -7.]),
                           calibrated_upper_w=np.array([7., 8., 1.]))
        frame_before = self.frame.copy(deep=True)
        result = add_total_power(self.result, self.frame, request, self.known)
        self.assertEqual(result["total_power_status"], "steady_total_power_estimate")
        np.testing.assert_array_equal(result["base_power_w"], [20., 20., 20.])
        np.testing.assert_array_equal(result["total_quantiles_w"], self.result["cutting_quantiles_w"] + 20.)
        np.testing.assert_array_equal(result["total_lower_w"], self.result["calibrated_lower_w"] + 20.)
        np.testing.assert_array_equal(result["total_upper_w"], self.result["calibrated_upper_w"] + 20.)
        self.assertEqual(result["total_uncertainty_scope"], "cutting_component_only_base_estimate_added")
        self.assertNotIn("total_power_w", self.result)
        pd.testing.assert_frame_equal(self.frame, frame_before)

    def test_steady_requires_verified_stable_and_constant_speed(self):
        request, _ = self.certificate()
        for stable in (False, None):
            with self.subTest(stable=stable):
                self.unavailable(request, known={**self.known, "spindle_stable_before_cut": stable}, contains="稳定")
        changed = self.frame.copy()
        changed.loc[1, "spindle_rpm"] = 1500.
        self.unavailable(request, frame=changed, contains="恒定")

    def test_domain_machine_channel_settings_and_rpm_are_enforced(self):
        request, _ = self.certificate()
        for key in ("machine_id", "power_channel", "machine_settings_version"):
            with self.subTest(domain_key=key):
                self.unavailable(request, known={**self.known, key: "other"}, contains=key)
        changed = self.frame.copy()
        changed["spindle_rpm"] = 10001.
        self.unavailable(request, frame=changed, contains="适用域")

    def test_model_report_hashes_and_version_cannot_be_changed(self):
        for field in ("model_sha256", "report_sha256", "version"):
            with self.subTest(field=field):
                request, _ = self.certificate()
                request["base_validation"][field] = "0" * 64
                self.unavailable(request)
        request, _ = self.certificate()
        alternate = self.root / "different_model.json"
        alternate.write_text("{}", encoding="utf-8")
        request["base_model_path"] = str(alternate)
        self.unavailable(request, contains="SHA-256")

    def test_event_or_research_confirmation_never_qualifies_total_prediction(self):
        for mode in ("events", "measured_speed_labels", "confirmed_response_relative_research_candidate"):
            with self.subTest(mode=mode):
                request, _ = self.certificate(mode)
                self.unavailable(request)
        for mode in ("steady_precutting", "commanded_dynamic"):
            with self.subTest(research_mode=mode):
                request, _ = self.certificate(mode, status="calibrated_unvalidated")
                self.unavailable(request, contains="研究候选")

    def test_report_must_explicitly_release_the_requested_mode(self):
        request, _ = self.certificate()
        report_path = Path(request["base_validation"]["report_path"])
        import json
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["eligible_for_steady_prediction"] = False
        save_json(report_path, report)
        request["base_validation"]["report_sha256"] = file_hash(report_path)
        self.unavailable(request, contains="尚未放行")

    def test_dynamic_uses_full_independent_timeline_and_existing_predictor(self):
        request, model, timeline, mapping = self.dynamic_request()
        with patch("spindle_power.load_prediction.total_power.predict_base_from_commands", wraps=predict_base_from_commands) as predictor:
            result = add_total_power(self.result, self.frame, request, self.known)
        self.assertEqual(result["total_power_status"], "commanded_dynamic_total_power_prediction")
        predictor.assert_called_once()
        self.assertEqual(len(predictor.call_args.args[0]), len(timeline))
        expected = predict_base_from_commands(timeline.time_s, timeline.commands_rpm, model, initial_speed_rpm=1000., source="nc_aligned")
        base = np.interp(mapping.time_s, timeline.time_s, expected["power_w"])
        np.testing.assert_allclose(result["base_power_w"], base)
        np.testing.assert_allclose(result["total_power_w"], base + self.result["cutting_quantiles_w"][:, 1])
        np.testing.assert_allclose(result["total_lower_w"], base + self.result["lower_w"])

    def test_dynamic_rejects_unknown_initial_state_or_incomplete_timeline(self):
        original, _, _, _ = self.dynamic_request()
        for field in ("initial_state_verified", "complete_nc_timeline_verified", "initial_speed_rpm",
                      "time_mapping", "command_timeline_sha256"):
            with self.subTest(missing=field):
                request = deepcopy(original)
                request.pop(field)
                self.unavailable(request)

    def test_dynamic_rejects_nonuniform_timeline(self):
        request, _, timeline, _ = self.dynamic_request()
        timeline.loc[5, "time_s"] += .01
        timeline.to_csv(request["command_timeline"], index=False, encoding="utf-8")
        request["command_timeline_sha256"] = file_hash(request["command_timeline"])
        self.unavailable(request, contains="均匀采样")

    def test_dynamic_rejects_missing_duplicate_or_outside_mapping(self):
        for kind in ("missing", "duplicate", "outside"):
            with self.subTest(mapping=kind):
                request, _, _, mapping = self.dynamic_request()
                if kind == "missing":
                    mapping = mapping.iloc[:2]
                elif kind == "duplicate":
                    mapping = pd.concat([mapping, mapping.iloc[:1]])
                else:
                    mapping.loc[2, "time_s"] = 3.
                mapping.to_csv(request["time_mapping"], index=False, encoding="utf-8")
                request["time_mapping_sha256"] = file_hash(request["time_mapping"])
                self.unavailable(request, contains="映射")

    def test_dynamic_rejects_out_of_domain_and_conflicting_command(self):
        request, _, timeline, _ = self.dynamic_request()
        timeline.loc[0, "commands_rpm"] = 11000.
        timeline.to_csv(request["command_timeline"], index=False, encoding="utf-8")
        request["command_timeline_sha256"] = file_hash(request["command_timeline"])
        self.unavailable(request, contains="适用域")
        request, _, _, _ = self.dynamic_request()
        changed = self.frame.copy()
        changed["spindle_rpm"] = 1500.
        self.unavailable(request, frame=changed, contains="不一致")


if __name__ == "__main__":
    unittest.main()
