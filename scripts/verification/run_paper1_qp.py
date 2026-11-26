"""Independent F1 Q/P comparison: shared new main, Q100/P100/Q200/P200.

The released prospective snapshot and rechecked execution profile are required.
This entry never loads K0, registers a production model, or reads event data.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

PLAN = (("Q", 100), ("P", 100), ("Q", 200), ("P", 200))
SOFT_SECONDS, HARD_SECONDS = 7.5 * 3600, 8 * 3600
WINDOW_POLICY = "prospective_v1"
SCREENING_THRESHOLDS = dict(rmse_w=-.05, mae_w=.02, sustained_absolute_bias_w=-.20,
                            dynamic_rmse_w=.05, peak_amplitude_absolute_error_w=-.10)
PROTOCOL_THRESHOLDS = dict(rmse_relative_change_max=-.05, mae_relative_change_max=.02,
                          sustained_absolute_bias_relative_change_max=-.20,
                          dynamic_rmse_relative_change_max=.05, peak_amplitude_error_relative_change_max=-.10)
REPORTING_POLICY = dict(final_selection_epoch=200, fixed_primary_reporting_epochs=[100, 200],
                        checkpoint_roles={"100": "pause_checkpoint_trajectory", "200": "predeclared_final_endpoint"},
                        best_role="supplementary_not_for_final_selection")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def now():
    return datetime.now(timezone.utc).isoformat()


def reference(path):
    from spindle_power.load_prediction.common import file_hash
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_hash(path))


def reference_fields(value):
    """Compare file identity independently of descriptive reference metadata."""
    require(isinstance(value, dict) and {"path", "sha256"} <= set(value), "Evidence lacks a file reference")
    return dict(path=str(Path(value["path"]).resolve()), sha256=value["sha256"])


def check_reference(value):
    expected = reference_fields(value)
    require(reference(expected["path"]) == expected, f"Evidence changed: {expected['path']}")


def protocol_references(value):
    """Freeze explicit file references; supplementary readiness never selects best."""
    found = []
    if isinstance(value, dict):
        if {"path", "sha256"} <= set(value):
            item = reference_fields(value)
            check_reference(item)
            found.append(item)
        for item in value.values():
            found.extend(protocol_references(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(protocol_references(item))
    return found


def supplementary_protocol(identity, entries, gate_references):
    """Bind the predeclared F1 protocol; unavailable shapes never alter scoring."""
    from paper1_prediction_metrics import ShapeProtocol
    value = identity.get("supplementary_protocol")
    if value is None:
        return dict(status="not_ready", unavailable_reasons=["supplementary_protocol_not_supplied"],
                    reference_attachments={}, shape_protocol=None)
    check_reference(value)
    protocol = read_json(value["path"])
    require(protocol.get("schema") == "paper1.qp_supplementary_protocol.1"
            and protocol.get("fold") == "F1" and reference_fields(protocol.get("snapshot")) == reference_fields(identity["snapshot"])
            and protocol.get("status") in ("ready", "not_ready"), "Supplementary protocol identity is invalid")
    snapshot = read_json(identity["snapshot"]["path"])
    require(reference_fields(protocol.get("parent_snapshot")) == reference_fields(snapshot["source_snapshot"]["support_fill_derivation"]["parent_snapshot"])
            and reference_fields(protocol.get("data_audit")) == reference_fields(gate_references["data_audit"])
            and protocol.get("baseline") == "Q" and protocol.get("candidate") == "P",
            "Supplementary protocol parent, audit, or comparison direction changed")
    train, validation = partition(entries)
    for name, selected in (("training", train), ("validation", validation)):
        expected = {entry["metadata"]["record_id"] for entry in selected}
        actual = protocol[name + "_record_ids"]
        require(len(actual) == len(expected) and set(actual) == expected,
                "Supplementary protocol must retain the complete F1 " + name + " set")
    require(protocol.get("screening_thresholds") == PROTOCOL_THRESHOLDS,
            "Supplementary screening thresholds changed")
    require(all(protocol.get(key) == expected for key, expected in REPORTING_POLICY.items()),
            "Supplementary checkpoint interpretation changed")
    attachments = protocol.get("reference_attachments", {})
    entries_by_id = {entry["metadata"]["record_id"]: entry for entry in entries}
    require(set(attachments) <= set(entries_by_id), "Unexpected reference attachment record")
    directory = Path(identity["snapshot"]["path"]).parent
    for record_id, attachment in attachments.items():
        entry = entries_by_id[record_id]
        require(reference_fields(attachment["candidate_csv"]) == dict(path=str((directory / entry["file"]).resolve()), sha256=entry["sha256"]),
                "Reference attachment does not bind this candidate record")
        receipt = read_json(attachment["receipt"]["path"])
        require(receipt.get("schema") == "afc.reference_display_candidate.2"
                and receipt.get("status") == "candidate_verified_not_formally_released"
                and receipt.get("record_id") == record_id
                and reference_fields(receipt.get("snapshot")) == reference_fields(protocol["parent_snapshot"])
                and reference_fields(receipt.get("sample")) == reference_fields(attachment["source_csv"])
                and reference_fields(receipt.get("attachment")) == reference_fields(attachment)
                and reference_fields(receipt.get("support_segments")) == reference_fields(attachment["support_segments"])
                and receipt.get("base_validity_exact") is True
                and receipt.get("label_metadata_exact_except_declared_lineage") is True,
                "Reference attachment lacks exact historical-source evidence")
    if protocol["status"] == "ready":
        require(set(attachments) == set(entries_by_id) and not protocol.get("unavailable_reasons"),
                "Ready supplementary protocol lacks complete 450-record support")
        shape = protocol["shape_protocol"]
        ShapeProtocol(**{field.name: shape[field.name] for field in fields(ShapeProtocol)}).validate()
    return protocol


def complete_macro(records, keys):
    """Never average only the available records: incomplete metrics remain null."""
    import math
    from spindle_power.load_prediction.training_service import balanced_metrics
    result = {}
    for key in keys:
        missing = [row["record_id"] for row in records
                   if not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key])]
        value = None
        if not missing:
            value = balanced_metrics([{**row, "metrics": {key: row[key]}} for row in records])[key]
        result[key] = dict(value=value, expected_records=len(records), available_records=len(records) - len(missing),
                           missing_record_ids=missing)
    return result


def shape_diagnostics(folder, evaluation, validation, protocol, ledger):
    """Attach historical support only to an evaluation copy of each full grid."""
    import numpy as np
    import pandas as pd
    from paper1_prediction_metrics import ShapeProtocol, evaluate_shapes
    expected = {sample.metadata["record_id"]: sample for sample in validation}
    rows = evaluation["records"]
    require(len(expected) == len(rows) == 50 and {row["record_id"] for row in rows} == set(expected),
            "Supplementary evaluation must retain all 50 distinct validation records")
    shape = protocol.get("shape_protocol")
    frozen = (ShapeProtocol(**{field.name: shape[field.name] for field in fields(ShapeProtocol)})
              if protocol["status"] == "ready" else None)
    records = []
    for row in rows:
        ledger.guard()
        record_id = row["record_id"]
        sample = expected[record_id]
        frame = sample.frame.copy()
        reasons = list(protocol.get("unavailable_reasons", []))
        attachment = protocol.get("reference_attachments", {}).get(record_id)
        if attachment is None:
            reasons.append("reference_attachment_unavailable")
        else:
            binding = {key: attachment[key] for key in ("path", "sha256")}
            check_reference(binding)
            support = pd.read_csv(binding["path"], float_precision="round_trip")
            require(len(support) == len(frame) and np.array_equal(support.grid_index.to_numpy(), np.arange(len(frame))),
                    "Reference attachment lost original grid positions")
            for column in ("line", "execution_instance", "segment_mm", "path_mm"):
                pd.testing.assert_series_equal(frame[column].reset_index(drop=True), support[column].reset_index(drop=True),
                                               check_names=False, check_dtype=False, check_exact=True)
            for column in ("grid_index", "reference_support_segment_id"):
                frame[column] = support[column].to_numpy()
            check_reference(binding)
        prediction = (folder / row["prediction_file"]).resolve()
        require(prediction.is_relative_to(folder.resolve()), "Prediction artifact escapes milestone directory")
        binding = dict(path=str(prediction), sha256=row["prediction_sha256"])
        check_reference(binding)
        predicted = pd.read_csv(prediction, float_precision="round_trip")
        result = evaluate_shapes(frame, predicted, frozen, record_id=record_id, unavailable_reasons=reasons)
        check_reference(binding)
        result.update(round_id=sample.metadata.get("round_id"), group_id=sample.metadata.get("group_id"),
                      feature=sample.metadata["feature"])
        records.append(result)
    summary = complete_macro(records, SCREENING_THRESHOLDS)
    by_feature = {feature: complete_macro([row for row in records if row["feature"] == feature], SCREENING_THRESHOLDS)
                  for feature in sorted({row["feature"] for row in records})}
    by_round = {str(round_id): complete_macro([row for row in records if row["round_id"] == round_id], SCREENING_THRESHOLDS)
                for round_id in sorted({row["round_id"] for row in records})}
    return dict(schema="paper1.qp_shape_diagnostics.1", status="ready" if all(
                row["shape_status"] == "ready" for row in records) else "not_ready",
                expected_records=50, checked_records=50, records=records, macro_metrics=summary,
                metrics_by_feature=by_feature, metrics_by_round=by_round,
                aggregation="complete record -> feature -> round macro average; any missing record leaves metric null",
                screening_passed=None, selection_metric_unchanged=True)


def reference_scope_summary(records, *, scope, applicability_field, comparison_field):
    """Summarize a reference-defined subset while retaining every record's status."""
    applicable = [{**row, **(row[comparison_field] or dict(baseline_w=None, candidate_w=None))}
                  for row in records if row[applicability_field] == "applicable"]
    return dict(scope=scope, all_record_ids=[row["record_id"] for row in records],
        reference_applicable_record_ids=[row["record_id"] for row in applicable],
        applicable_record_ids=[row["record_id"] for row in records if row["status"] == "ready"],
        not_applicable_record_ids=[row["record_id"] for row in records if row["status"] == "not_applicable"],
        not_ready_record_ids=[row["record_id"] for row in records if row["status"] == "not_ready"],
        macro_metrics=complete_macro(applicable, ("baseline_w", "candidate_w")) if applicable else None,
        metrics_by_feature={feature: complete_macro([row for row in applicable if row["feature"] == feature], ("baseline_w", "candidate_w"))
                            for feature in sorted({row["feature"] for row in applicable})},
        metrics_by_round={str(round_id): complete_macro([row for row in applicable if row["round_id"] == round_id], ("baseline_w", "candidate_w"))
                          for round_id in sorted({row["round_id"] for row in applicable})})


def compare_sustained_results(baseline, candidate):
    """Only the frozen reference defines eligible sustained segments, never errors."""
    import math
    for key in ("record_id", "fixed_domain_sha256", "scored_points"):
        require(baseline[key] == candidate[key], "Q/P sustained fixed domain differs: " + key)
    result = dict(status="not_ready", sustained_applicability="not_ready", reasons=[], reference_segments=[],
                  absolute_bias_comparison=None)
    if baseline["shape_status"] != "ready" or candidate["shape_status"] != "ready":
        result["reasons"].append("shape_support_or_training_protocol_not_ready")
        return result
    require(baseline["shape_protocol"] == candidate["shape_protocol"], "Q/P sustained reference protocol differs")
    definition = lambda row: [{key: segment[key] for key in ("start_index", "end_index", "points", "span_mm")}
                              for segment in row["sustained_segments"]]
    left, right = definition(baseline), definition(candidate)
    require(left == right and baseline["sustained_points"] == candidate["sustained_points"]
            == sum(segment["points"] for segment in left), "Q/P sustained reference segment definitions differ")
    result["reference_segments"] = left
    if not left:
        result.update(status="not_applicable", sustained_applicability="not_applicable")
        return result
    result["sustained_applicability"] = "applicable"
    values = [row["sustained_absolute_bias_w"] for row in (baseline, candidate)]
    biases = [segment["bias_w"] for row in (baseline, candidate) for segment in row["sustained_segments"]]
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (*values, *biases)):
        result["reasons"].append("applicable_sustained_bias_missing_or_nonfinite")
        return result
    result.update(status="ready", absolute_bias_comparison=dict(baseline_w=values[0], candidate_w=values[1]))
    return result


def scientific_screening(output, epoch, export, ledger):
    """Compare P with Q on the full frozen reference set; best remains supplementary."""
    from paper1_prediction_metrics import compare_peak_results
    diagnostics = {kind: read_json(output / kind / f"epoch_{epoch:04d}" / export / "shape_diagnostics.json")
                   for kind in ("Q", "P")}
    records = {kind: {row["record_id"]: row for row in report["records"]} for kind, report in diagnostics.items()}
    require(len(records["Q"]) == len(records["P"]) == 50 and records["Q"].keys() == records["P"].keys(),
            "Scientific screening cannot drop or mismatch validation records")
    paired, sustained = [], []
    for record_id, baseline in records["Q"].items():
        ledger.guard()
        candidate = records["P"][record_id]
        pair = compare_peak_results(baseline, candidate)
        metadata = dict(record_id=record_id, round_id=baseline.get("round_id"), group_id=baseline.get("group_id"),
                        feature=baseline["feature"])
        paired.append(dict(**metadata, **pair))
        sustained.append(dict(**metadata, **compare_sustained_results(baseline, candidate)))
    peak_scope = reference_scope_summary(paired, scope="credible_reference_interior_peaks; frozen-reference applicability",
        applicability_field="amplitude_applicability", comparison_field="full_reference_amplitude_comparison")
    sustained_scope = reference_scope_summary(sustained, scope="frozen-reference sustained valid_main AND scored segments",
        applicability_field="sustained_applicability", comparison_field="absolute_bias_comparison")
    scopes = {"peak_amplitude_absolute_error_w": peak_scope, "sustained_absolute_bias_w": sustained_scope}
    gates = {}
    for key, threshold in SCREENING_THRESHOLDS.items():
        baseline = diagnostics["Q"]["macro_metrics"][key]["value"]
        candidate = diagnostics["P"]["macro_metrics"][key]["value"]
        reasons = []
        if key in scopes:
            applicability = scopes[key]
            aggregated = applicability["macro_metrics"]
            baseline = None if aggregated is None else aggregated["baseline_w"]["value"]
            candidate = None if aggregated is None else aggregated["candidate_w"]["value"]
            if not applicability["reference_applicable_record_ids"]:
                reasons.append("no_applicable_reference_records")
            if applicability["not_ready_record_ids"]:
                reasons.append("incomplete_fixed_reference_comparison")
        if baseline is None or candidate is None:
            reasons.append("incomplete_reference_applicable_metric" if key in scopes else "incomplete_50_record_metric")
        if baseline == 0 and not (candidate == 0 and threshold >= 0):
            reasons.append("zero_baseline_relative_change_undefined")
        delta = (None if reasons else 0. if baseline == 0 else (candidate - baseline) / baseline)
        gates[key] = dict(status="not_ready" if reasons else "passed" if delta <= threshold else "failed",
                          baseline=baseline, candidate=candidate, relative_change=delta, threshold=threshold, reasons=reasons)
    status = ("not_ready" if any(row["status"] == "not_ready" for row in gates.values()) else
              "failed" if any(row["status"] == "failed" for row in gates.values()) else "passed")
    return dict(schema="paper1.qp_scientific_screening.1", epoch=epoch, export=export,
                role="primary_fixed_epoch" if export == "fixed" else "supplementary_best",
                checkpoint_role=REPORTING_POLICY["checkpoint_roles"][str(epoch)],
                baseline="Q", candidate="P", status=status, passed=status == "passed",
                decision={"passed": "P_passed_development_screen", "failed": "retain_Q", "not_ready": "undetermined"}[status],
                gates=gates, peak_comparisons=paired, peak_amplitude_applicability=peak_scope,
                sustained_comparisons=sustained, sustained_bias_applicability=sustained_scope,
                expected_records=50, independent_confirmation=False)


def screening_summary(comparisons, *, completed):
    """The predeclared 200-update fixed endpoint alone informs the final screen."""
    selected = comparisons.get("200", {}).get("fixed")
    result = dict(schema="paper1.qp_scientific_summary.1", **deepcopy(REPORTING_POLICY),
                  training_completed=completed, status="not_ready", passed=False, decision="undetermined",
                  selected_comparison=None, reason="complete_plan_and_200_fixed_endpoint_required",
                  automatic_replacement=False, independent_confirmation=False)
    if completed and selected is not None:
        require(selected.get("epoch") == 200 and selected.get("export") == "fixed", "Final screen must use the 200-update fixed endpoint")
        require(selected.get("status") in ("passed", "failed", "not_ready"), "Invalid endpoint screening status")
        status = selected["status"]
        result.update(status=status, passed=status == "passed", selected_comparison="200.fixed",
            decision={"passed": "P_passed_development_screen", "failed": "retain_Q", "not_ready": "undetermined"}[status],
            reason="predeclared_endpoint_only; 100-update trajectory and best checkpoints do not select the outcome")
    return result


class BudgetStop(RuntimeError):
    pass


class BudgetLedger:
    """Count all active work across explicit restarts; never guess crash time."""
    def __init__(self, path, identity, *, resume, clock=time.monotonic, started=None):
        from spindle_power.load_prediction.common import save_json
        from spindle_power.load_prediction.training_checkpoint import digest
        self.path, self.clock, self.save = Path(path), clock, save_json
        self.started = clock() if started is None else started
        fingerprint = digest(identity)
        if self.path.exists():
            require(resume, "Existing budget requires explicit --resume")
            self.value = read_json(self.path)
            require(self.value.get("schema") == "paper1.qp_budget.1"
                    and self.value.get("fingerprint") == fingerprint, "Budget identity changed")
            require(self.value.get("active") is None,
                    "Unresolved active budget session; audit its elapsed time before any explicit restart")
            require(self.value["spent_seconds"] < HARD_SECONDS, "Eight-hour budget is exhausted")
        else:
            require(not resume, "Cannot resume without the original budget ledger")
            self.value = dict(schema="paper1.qp_budget.1", fingerprint=fingerprint, spent_seconds=0.,
                              soft_seconds=SOFT_SECONDS, hard_seconds=HARD_SECONDS, sessions=[], active=None)
        self.value["active"] = dict(session_id=uuid.uuid4().hex, pid=os.getpid(), started_at=now(), elapsed_seconds=0.)
        self.touch()

    def elapsed(self):
        return self.value["spent_seconds"] + self.clock() - self.started

    def touch(self):
        self.value["active"].update(elapsed_seconds=self.clock() - self.started, checked_at=now())
        self.save(self.path, self.value)

    def guard(self, *, new_update=False):
        elapsed = self.elapsed()
        if elapsed >= HARD_SECONDS:
            raise BudgetStop("hard_budget_exhausted")
        if new_update and elapsed >= SOFT_SECONDS:
            raise BudgetStop("soft_budget_no_new_update")

    def close(self, status):
        session = {**self.value["active"], "elapsed_seconds": self.clock() - self.started,
                   "finished_at": now(), "status": status}
        self.value["spent_seconds"] += session["elapsed_seconds"]
        self.value["sessions"].append(session)
        self.value["active"] = None
        self.save(self.path, self.value)


class EpochControl:
    """Soft budget gates a new full update; the hard budget gates every batch."""
    def __init__(self, ledger, target, maximum, progress):
        self.ledger, self.target, self.maximum, self.progress = ledger, target, maximum, progress
        self.in_update, self.milestone = False, False

    def cancel(self):
        self.ledger.guard()
        return self.milestone

    def training(self, loss):
        def call(backward):
            self.ledger.guard(new_update=True)
            self.in_update = True
            value = loss(backward)
            self.ledger.guard()  # A final batch crossing the hard deadline must not commit an update.
            return value
        return call

    def saved(self, value):
        if value.get("phase") == "resumed":
            require(value["epoch"] <= self.target, "Checkpoint is beyond this scheduled milestone")
            self.milestone = value["epoch"] == self.target and not value["completed"]
        else:
            self.in_update = False
            self.milestone = value["epoch"] == self.target and self.target < self.maximum
        self.ledger.touch()
        if self.progress:
            self.progress(value)


def run_epochs_to(module, maximum, target, config, training, validation, stage, control, path, identity):
    """A milestone returns normally, keeping the maximum and last-state checkpoint."""
    from spindle_power.load_prediction.common import Cancelled
    from spindle_power.load_prediction.training import _epochs
    from spindle_power.load_prediction.training_checkpoint import load_checkpoint

    require(1 <= target <= maximum, "Invalid fixed epoch milestone")
    try:
        _epochs(module, maximum, config, control.training(training), validation, stage, control.cancel,
                control.saved, checkpoint_path=path, checkpoint_identity=identity, resume=Path(path).exists())
    except Cancelled:
        if not control.milestone:
            raise
    saved = load_checkpoint(path, identity, resume=True)
    require(saved["epoch"] == target and saved["optimizer_updates"] == target,
            "Training did not reach the exact scheduled update count")
    require(saved["completed"] is (target == maximum), "Milestone must not masquerade as completed training")
    return saved


def source_identity():
    from spindle_power.load_prediction.common import file_hash
    from spindle_power.load_prediction.training_checkpoint import source_fingerprints
    return dict(source=source_fingerprints(), scripts={path.name: file_hash(path)
                for path in sorted(Path(__file__).parent.glob("*.py"))})


def preflight(args, identity, ledger):
    from spindle_power.load_prediction.support_fill import SUPPORT_FILL_RULES_VERSION
    from spindle_power.load_prediction.support_fill_snapshot import (
        _snapshot, _development_entries, _verify_release, implementation_identity)
    from spindle_power.load_prediction.training import TrainingConfig

    ledger.guard()
    snapshot_path = Path(identity["snapshot"]["path"])
    snapshot = _snapshot(snapshot_path)
    entries = _development_entries(snapshot, load_only=True)
    link = snapshot["source_snapshot"].get("support_fill_derivation", {})
    require(link.get("schema") == "afc.support_fill_derivation.1"
            and link.get("rules_version") == SUPPORT_FILL_RULES_VERSION
            and link.get("implementation_sha256") == implementation_identity(), "Candidate derivation identity changed")
    source_parent = link["parent_snapshot"]
    source_parent_ref = reference_fields(source_parent)
    check_reference(source_parent_ref)
    source_parent_snapshot = _snapshot(Path(source_parent_ref["path"]))
    require(source_parent_snapshot["snapshot_id"] == source_parent.get("snapshot_id"), "Candidate parent snapshot ID changed")
    check_reference(source_parent_ref)
    release_path = snapshot_path.parent.parent.parent / "support_fill_release.json"
    release_ref = reference(release_path)
    release = _verify_release(SimpleNamespace(output_dir=release_path.parent), snapshot, snapshot_path.parent)
    check_reference(release_ref)
    profile = read_json(identity["execution_profile"]["path"])
    require(profile.get("schema") == "paper1.load_execution_profile.1"
            and profile.get("status") == "prospective_rechecked"
            and reference_fields(profile.get("candidate_snapshot")) == reference_fields(identity["snapshot"])
            and profile.get("window_policy") == WINDOW_POLICY, "New prospective execution profile is required")
    require(profile.get("source_fingerprints") == identity["source"], "Rechecked profile source differs from the runner")
    require(reference_fields(profile.get("selection_evidence")) == reference_fields(release["execution_recheck"]),
            "Execution profile and training release bind different rechecks")
    parent_ref = profile["parent_execution_profile"]
    check_reference(parent_ref)
    parent = read_json(parent_ref["path"])
    require(parent.get("schema") == "afc.paper1_execution_profile.1"
            and parent.get("status") == "stage0_engineering_accepted", "Stage-0 execution parent is not accepted")
    execution = profile["training_config"]
    expected = dict(device="cpu", cpu_threads=10, cpu_interop_threads=1, main_batch_points=32768,
                    residual_batch_windows=128, residual_batch_strategy="padded", loss_cache_mode="cpu",
                    loss_cache_max_bytes=1073741824, learning_rate=.001, gradient_clip=1., seed=42)
    require(all(execution.get(key) == value and parent["training_config"].get(key) == value
                for key, value in expected.items()), "Q/P must share the frozen and rechecked CPU configuration")
    config = TrainingConfig(**expected, main_epochs=100, residual_epochs=200, patience=201, min_delta=0.,
                            residual_window_policy=WINDOW_POLICY)
    config.validate()
    references = dict(release=release_ref, parent_snapshot=source_parent_ref, parent_execution_profile=parent_ref,
                      **{key: release[key] for key in ("data_audit", "prediction_audit", "window_audit", "execution_recheck")})
    if identity.get("supplementary_protocol") is not None:
        supplemental = identity["supplementary_protocol"]
        check_reference(supplemental)
        references["supplementary_protocol"] = supplemental
        for index, value in enumerate(protocol_references(read_json(supplemental["path"]))):
            references[f"supplementary_file_{index:04d}"] = value
    for value in (identity["snapshot"], identity["execution_profile"], *references.values()):
        check_reference(value)
    ledger.guard()
    return snapshot, entries, config, references


def read_sample(directory, entry, ledger):
    from spindle_power.load_prediction.samples import SampleSet
    from spindle_power.load_prediction.support_fill_snapshot import _frame
    ledger.guard()
    value = SampleSet(_frame(directory, entry), deepcopy(entry["metadata"]))
    ledger.guard()
    return value


def partition(entries):
    from spindle_power.load_prediction.experiment_protocol import protocol_folds
    fold = next(item for item in protocol_folds() if item["fold"] == "F1")
    train = [item for item in entries if item["metadata"]["round_id"] in fold["train_rounds"]]
    validation = [item for item in entries if item["metadata"]["round_id"] in fold["validation_rounds"]]
    require(len(train) == 400 and len(validation) == 50, "F1 requires exactly 400 training and 50 validation records")
    return train, validation


def ready_for_evaluation(bundle):
    from spindle_power.load_prediction.neural import state_version
    require(state_version(bundle.main) == bundle.main_version, "Frozen shared main changed")
    bundle.main_trained = bundle.residual_trained = True
    bundle.residual_main_version = bundle.main_version
    bundle.residual_version = state_version(bundle.residual)


def validation_rmse(bundle, samples, ledger):
    from spindle_power.load_prediction.training_service import evaluate_records
    ready_for_evaluation(bundle)
    report = evaluate_records(bundle, samples, feed_mode="programmed", memory=None, update_memory=False,
                              cancel=lambda: ledger.guard())
    expected = {sample.metadata["record_id"]: int(sample.frame.valid_residual.sum()) for sample in samples}
    require(len(report["records"]) == len(expected) == 50
            and {row["record_id"] for row in report["records"]} == set(expected)
            and all(row["metrics"]["count"] == expected[row["record_id"]] > 0 for row in report["records"]),
            "Validation must retain all 50 scored records")
    return report["equal_weight_metrics"]["rmse_w"]


def model_bundle(model_config, training_config, initial, shared=None):
    from spindle_power.load_prediction.models import build_load_models
    from spindle_power.load_prediction.neural import freeze
    bundle = build_load_models(model_config, device=training_config.device)
    bundle.main.load_state_dict(initial["main"], strict=True)
    bundle.residual.load_state_dict(initial["residual"], strict=True)
    bundle.metadata = dict(config=asdict(training_config), experiment="paper1_qp_f1",
                           residual_window_policy=WINDOW_POLICY)
    if shared is not None:
        bundle.main.load_state_dict(shared["weights"], strict=True)
        bundle.normalization = deepcopy(shared["normalization"])
        bundle.main_trained, bundle.main_version = True, shared["main_version"]
        freeze(bundle.main)
    return bundle


def initial_state(directory, identity, model_config, config):
    import torch
    from spindle_power.load_prediction.models import build_load_models
    from spindle_power.load_prediction.training_checkpoint import load_checkpoint, save_checkpoint
    path = directory / "initial.pt"
    binding = dict(run=identity, phase="shared_initialization", model=asdict(model_config))
    if path.exists():
        return load_checkpoint(path, binding, resume=True)
    torch.manual_seed(42)
    bundle = build_load_models(model_config, device=config.device)
    value = dict(epoch=0, completed=True, artifact_schema="paper1.qp_initialization.1",
                 main=deepcopy(bundle.main.state_dict()), residual=deepcopy(bundle.residual.state_dict()))
    save_checkpoint(path, binding, value)
    return value


def shared_main(directory, identity, model_config, config, initial, train_entries, validation, snapshot_dir, ledger, progress):
    import numpy as np
    import torch
    from spindle_power.load_prediction.samples import PHYSICS_COLUMNS, SampleSet, assert_feed_inputs, load_input_view
    from spindle_power.load_prediction.neural import freeze, prospective_residual_context, state_version
    from spindle_power.load_prediction.training_cache import prepare_main_loss
    from spindle_power.load_prediction.training_checkpoint import load_checkpoint, save_checkpoint

    path = directory / "shared_main.pt"
    binding = dict(run=identity, phase="shared_main_export", model=asdict(model_config), training=asdict(config))
    if path.exists():
        shared = load_checkpoint(path, binding, resume=True)
        check_reference(shared["main_checkpoint"])
        require(shared["trained_epochs"] == 100 and shared["completed"] is True, "Shared main is incomplete")
        return shared
    bundle = model_bundle(model_config, config, initial)
    needed = PHYSICS_COLUMNS + ["power_cut_w", "valid_main"]

    def compact(sample):
        measured = load_input_view(sample, "measured")
        if measured.frame.valid_main.any():
            assert_feed_inputs(measured.frame.loc[measured.frame.valid_main], "measured")
        return SampleSet(measured.frame[needed].copy(), measured.metadata)

    train = [compact(read_sample(snapshot_dir, entry, ledger)) for entry in train_entries]
    val = [compact(sample) for sample in validation]
    targets = np.concatenate([sample.frame.loc[sample.frame.valid_main, "power_cut_w"].to_numpy(float)
                              for sample in train if sample.frame.valid_main.any()])
    require(targets.size > 0 and np.isfinite(targets).all(), "Main training labels are missing or nonfinite")
    scale = max(1., float(np.std(targets)))
    control = EpochControl(ledger, 100, 100, progress)
    loss = prepare_main_loss(bundle, train, scale, config, control.cancel, cache_device=False,
                             max_cache_bytes=config.loss_cache_max_bytes)
    val_loss = prepare_main_loss(bundle, val, scale, config, control.cancel, cache_device=False,
                                 max_cache_bytes=config.loss_cache_max_bytes)
    del train, val
    main_path = directory / "main.pt"
    state = run_epochs_to(bundle.main, 100, 100, config, loss, val_loss, "main", control, main_path,
                          dict(run=identity, phase="main", model=asdict(model_config), training=asdict(config)))
    del loss, val_loss
    # _epochs returns the validation-MSE best state; epoch 100 remains archived separately.
    freeze(bundle.main)
    bundle.main_version = state_version(bundle.main)
    values, residuals = [], []
    for entry in train_entries:
        sample = read_sample(snapshot_dir, entry, ledger)
        view = load_input_view(sample, "programmed")
        selected, indices, _ = prospective_residual_context(view.frame, model_config.window_points)
        valid = selected.valid_residual.to_numpy(bool)
        require(int(valid.sum()) == int(view.frame.valid_residual.sum()), "Training score lacks prospective inputs")
        if valid.any():
            condition = bundle.condition_values(sample.metadata, len(selected))
            with torch.no_grad():
                main = bundle.main(torch.as_tensor(selected[PHYSICS_COLUMNS].to_numpy(np.float32), device=bundle.device),
                                   None if condition is None else torch.as_tensor(condition, dtype=torch.float32,
                                                                                device=bundle.device)).cpu().numpy()
            values.append(main[valid].astype(np.float64))
            residuals.append(selected.power_cut_w.to_numpy(float)[valid] - main[valid])
        ledger.guard()
    require(values and residuals, "No residual-training normalization domain")
    values, residuals = np.concatenate(values), np.concatenate(residuals)
    require(np.isfinite(values).all() and np.isfinite(residuals).all(), "Residual normalization is nonfinite")
    normalization = dict(main_mean_w=float(np.mean(values)), main_scale_w=max(1., float(np.std(values))),
                         residual_scale_w=max(1., float(np.std(residuals))), target_scale_w=scale,
                         high_load_threshold_w=float(np.quantile(targets, .9)),
                         low_load_threshold_w=float(np.quantile(targets, .1)))
    shared = dict(epoch=100, completed=True, artifact_schema="paper1.qp_shared_main.1", trained_epochs=100,
                  selected_epoch=state["best_epoch"], selection_metric="validation_main_normalized_mse",
                  weights=deepcopy(bundle.main.state_dict()), main_version=bundle.main_version,
                  main_checkpoint=reference(main_path), normalization=normalization,
                  normalization_domain="F1_training_valid_residual", normalization_points=len(values),
                  main_history=state["history"])
    ledger.guard()
    save_checkpoint(path, binding, shared)
    ledger.touch()
    return shared


def residual_inputs(bundle, train_entries, snapshot_dir, ledger):
    import pandas as pd
    from spindle_power.load_prediction.samples import SampleSet, load_input_view
    from spindle_power.load_prediction.training import _residual_data
    prepared = []
    for entry in train_entries:
        sample = load_input_view(read_sample(snapshot_dir, entry, ledger), "programmed")
        for original, windows in _residual_data(bundle, [sample]):
            # Cache weighting needs metadata only; do not retain the 90-column source frame.
            prepared.append((SampleSet(pd.DataFrame(), deepcopy(original.metadata)), windows))
        ledger.guard()
    require(len(prepared) == 400, "Residual training must retain all 400 records")
    return prepared


def export_milestone(output, kind, epoch, saved, bundle, validation, checkpoint_ref, ledger, stage_identity, supplementary):
    from spindle_power.load_prediction.common import save_json
    from spindle_power.load_prediction.neural import freeze
    from spindle_power.load_prediction.training_service import evaluate_records
    from spindle_power.load_prediction.training_checkpoint import digest

    destination = output / kind / f"epoch_{epoch:04d}"
    receipt_path = destination / "receipt.json"
    if destination.exists():
        return checked_milestone(receipt_path, kind, epoch, stage_identity, output)
    temporary = output / kind / (f".epoch_{epoch:04d}_{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    artifacts, evaluations = [], {}
    for name, weights, selected_epoch in (("fixed", saved["weights"], epoch),
                                          ("best", saved["best_state"], saved["best_epoch"])):
        ledger.guard()
        require(weights is not None, "Validation best checkpoint is missing")
        clone = deepcopy(bundle)
        clone.residual.load_state_dict(weights, strict=True)
        ready_for_evaluation(clone)
        freeze(clone.main)
        freeze(clone.residual)
        clone.metadata.update(experiment_export=name, trained_epochs=epoch, selected_epoch=selected_epoch,
                              optimization_completed=saved["completed"], register_model=False)
        folder = temporary / name
        folder.mkdir()
        clone.save(folder / "load.pt")
        evaluation = evaluate_records(clone, validation, feed_mode="programmed", memory=None, update_memory=False,
                                      output_dir=folder, cancel=lambda: ledger.guard())
        require(len(evaluation["records"]) == 50, "Milestone validation record count changed")
        diagnostic = shape_diagnostics(folder, evaluation, validation, supplementary, ledger)
        save_json(folder / "shape_diagnostics.json", diagnostic)
        evaluations[name] = dict(selected_epoch=selected_epoch, metrics=evaluation["equal_weight_metrics"],
                                shape_status=diagnostic["status"], shape_macro_metrics=diagnostic["macro_metrics"])
        for path in sorted(folder.iterdir()):
            if path.is_file():
                item = reference(path)
                item["path"] = str(destination / path.relative_to(temporary))
                artifacts.append(item)
    receipt = dict(schema="paper1.qp_milestone.1", kind=kind, epoch=epoch, checkpoint=checkpoint_ref,
                   stage_fingerprint=digest(stage_identity), run_fingerprint=digest(stage_identity["run"]),
                   window_audit=stage_identity["configuration"]["gate_references"]["window_audit"],
                   completed=saved["completed"], status="completed" if saved["completed"] else "paused_at_milestone",
                   primary="fixed_epoch", supplementary="best_validation_macro_rmse", evaluations=evaluations,
                   artifacts=artifacts, training_registered=False)
    save_json(temporary / "receipt.json", receipt)
    ledger.guard()
    temporary.rename(destination)
    ledger.touch()
    return receipt


def archived_checkpoint(path):
    audit = read_json(path.with_suffix(path.suffix + ".json"))
    value = dict(path=str((path.parent / audit["archive"]).resolve()), sha256=audit["sha256"])
    check_reference(value)
    return value


def checked_milestone(path, kind, epoch, stage_identity, output):
    from spindle_power.load_prediction.training_checkpoint import digest
    receipt = read_json(path)
    require(receipt.get("schema") == "paper1.qp_milestone.1" and receipt.get("kind") == kind
            and receipt.get("epoch") == epoch and receipt.get("completed") is (epoch == 200),
            "Existing milestone is not the scheduled fixed-epoch artifact")
    require(receipt.get("stage_fingerprint") == digest(stage_identity)
            and receipt.get("run_fingerprint") == digest(stage_identity["run"]), "Milestone belongs to another run or stage")
    expected = (output / kind / "epochs/residual" / f"epoch_{epoch:04d}.pt").resolve()
    require(Path(receipt["checkpoint"]["path"]).resolve() == expected, "Milestone checkpoint escapes its current stage")
    check_reference(receipt["checkpoint"])
    audit = read_json(expected.with_suffix(".pt.json"))
    require(audit.get("fingerprint") == digest(stage_identity)
            and audit.get("sha256") == receipt["checkpoint"]["sha256"], "Milestone checkpoint identity changed")
    folder = path.parent.resolve()
    artifacts = {Path(item["path"]).resolve(): item for item in receipt["artifacts"]}
    require(len(artifacts) == len(receipt["artifacts"]) and all(item.is_relative_to(folder) for item in artifacts),
            "Milestone artifact membership changed")
    require({folder / name / filename for name in ("fixed", "best")
             for filename in ("load.pt", "evaluation.json", "shape_diagnostics.json")}
            <= set(artifacts), "Milestone lacks fixed/best model, evaluation or shape evidence")
    for item in receipt["artifacts"]:
        check_reference(item)
    for name in ("fixed", "best"):
        evaluation = read_json(folder / name / "evaluation.json")
        diagnostic = read_json(folder / name / "shape_diagnostics.json")
        ids = {row["record_id"] for row in evaluation["records"]}
        require(len(ids) == len(evaluation["records"]) == 50 and len(diagnostic["records"]) == 50
                and {row["record_id"] for row in diagnostic["records"]} == ids, "Milestone lost validation records")
        for row in evaluation["records"]:
            prediction = (folder / name / row["prediction_file"]).resolve()
            require(prediction in artifacts and artifacts[prediction]["sha256"] == row["prediction_sha256"],
                    "Milestone lost a prediction binding")
    return receipt


def run(args):
    if not args.execute:
        return dict(status="planned_not_executed", training_started=False, fold="F1", seed=42,
                    main_updates=100, residual_maximum=200, schedule=PLAN,
                    **deepcopy(REPORTING_POLICY),
                    soft_seconds=SOFT_SECONDS, hard_seconds=HARD_SECONDS)
    require(getattr(args, "worker", False), "Execute this experiment through its supervised CLI")
    started = args.started_monotonic
    require(started is not None and 0 <= started <= time.monotonic(), "Missing supervisor budget origin")
    require(os.environ.get("MKL_THREADING_LAYER") == "SEQUENTIAL", "Set MKL_THREADING_LAYER=SEQUENTIAL before launch")
    import numpy as np
    import torch
    from spindle_power.load_prediction.common import save_json
    from spindle_power.load_prediction.condition import fit_condition_spec
    from spindle_power.load_prediction.load_device import configure_load_device
    from spindle_power.load_prediction.models import ModelConfig
    from spindle_power.load_prediction.neural import state_version
    from spindle_power.load_prediction.training_cache import prepare_residual_loss
    from spindle_power.load_prediction.training_checkpoint import digest

    output = args.output_dir.resolve()
    snapshot_path, profile_path = args.snapshot.resolve(), args.execution_profile.resolve()
    require(not output.is_relative_to(snapshot_path.parent) and not snapshot_path.is_relative_to(output),
            "Experiment output overlaps the source snapshot")
    require(output.exists() is args.resume, "Use a new output directory or explicit --resume of the original run")
    output.mkdir(parents=True, exist_ok=args.resume)
    identity = dict(schema="paper1.qp_run.1", snapshot=reference(snapshot_path), execution_profile=reference(profile_path),
                    supplementary_protocol=(reference(args.supplementary_protocol) if args.supplementary_protocol else None),
                    **source_identity(), fold="F1", seed=42, capacity="C64", window_points=256,
                    window_policy=WINDOW_POLICY, main_maximum=100, residual_maximum=200,
                    schedule=[list(item) for item in PLAN], primary="fixed_100_and_200",
                    **deepcopy(REPORTING_POLICY),
                    best_metric="balanced_validation_rmse_w", memory=False, k0_weights=False,
                    environment=dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__))
    identity_path = output / "identity.json"
    if args.resume:
        require(read_json(identity_path) == identity, "Explicit restart identity changed")
    else:
        save_json(identity_path, identity)
    ledger = BudgetLedger(output / "budget.json", identity, resume=args.resume, started=started)
    status, receipts, active_stage = "running", [], "preflight"
    screening = {}
    gate_refs, shared = {}, None

    def progress(value):
        save_json(output / "state.json", dict(status="running", stage=active_stage, progress=value,
                  elapsed_seconds=ledger.elapsed(), milestones=[(row["kind"], row["epoch"]) for row in receipts],
                  training_started=True, training_registered=False))
        print(json.dumps(dict(schedule_stage=active_stage, **value), ensure_ascii=False), flush=True)

    try:
        snapshot, entries, config, gate_refs = preflight(args, identity, ledger)
        _, execution = configure_load_device(config.device, cpu_threads=config.cpu_threads,
                                              cpu_interop_threads=config.cpu_interop_threads)
        train_entries, val_entries = partition(entries)
        supplementary = supplementary_protocol(identity, entries, gate_refs)
        model_config = ModelConfig(coefficient_widths=(64, 64), hidden=64, window_points=256,
                                   route="nominal_programmed_residual_v1",
                                   condition_spec=fit_condition_spec([item["metadata"] for item in train_entries]))
        configuration = dict(training=asdict(config), model=asdict(model_config), execution=execution,
                             gate_references=gate_refs, run_fingerprint=digest(identity))
        config_path = output / "configuration.json"
        if config_path.exists():
            require(read_json(config_path) == json.loads(json.dumps(configuration)), "Rebuilt configuration changed")
        else:
            save_json(config_path, configuration)
        checkpoint_identity = dict(run=identity, configuration=configuration)
        validation = [read_sample(snapshot_path.parent, entry, ledger) for entry in val_entries]
        common = output / "shared"
        common.mkdir(exist_ok=True)
        initial = initial_state(common, checkpoint_identity, model_config, config)
        active_stage = "main"
        shared = shared_main(common, checkpoint_identity, model_config, config, initial, train_entries,
                             validation, snapshot_path.parent, ledger, progress)
        planning_bundle = model_bundle(model_config, config, initial, shared)
        require(planning_bundle.execution_window_policy() == WINDOW_POLICY, "Prospective window interface is not active")
        active_stage = "prepare_residual"
        prepared = residual_inputs(planning_bundle, train_entries, snapshot_path.parent, ledger)
        del planning_bundle
        shared_version = shared["main_version"]
        stage_identities = {}
        for kind, target in PLAN:
            active_stage = f"{kind}{target}"
            ledger.guard()
            branch_model = replace(model_config, residual_output="quantiles" if kind == "Q" else "point")
            stage_identity = dict(**checkpoint_identity, phase=kind, model=asdict(branch_model),
                                  shared_main=reference(common / "shared_main.pt"), initialization=reference(common / "initial.pt"),
                                  normalization=shared["normalization"], validation_metric="balanced_validation_rmse_w")
            stage_identities[kind] = stage_identity
            existing = output / kind / f"epoch_{target:04d}" / "receipt.json"
            if existing.exists():
                receipts.append(checked_milestone(existing, kind, target, stage_identity, output))
            else:
                bundle = model_bundle(branch_model, config, initial, shared)
                control = EpochControl(ledger, target, 200, progress)
                loss = prepare_residual_loss(bundle, prepared, control.cancel,
                    batch_windows=config.residual_batch_windows, batch_strategy=config.residual_batch_strategy,
                    cache_device=False, max_cache_bytes=config.loss_cache_max_bytes)
                path = output / kind / "residual.pt"
                saved = run_epochs_to(bundle.residual, 200, target, config, loss,
                    lambda _back: validation_rmse(bundle, validation, ledger), f"residual_{kind}", control, path, stage_identity)
                del loss
                require(state_version(bundle.main) == shared_version, "Residual training changed the shared main")
                receipts.append(export_milestone(output, kind, target, saved, bundle, validation,
                                                 archived_checkpoint(path), ledger, stage_identity, supplementary))
                del bundle
            if kind == "P":
                active_stage = f"scientific_screening_{target}"
                screening[str(target)] = {name: scientific_screening(output, target, name, ledger) for name in ("fixed", "best")}
                save_json(output / "scientific_screening.json", dict(summary=screening_summary(screening, completed=False), comparisons=screening))
        active_stage = "final_integrity"
        for value in (identity["snapshot"], identity["execution_profile"], *gate_refs.values()):
            check_reference(value)
        require(source_identity() == {key: identity[key] for key in ("source", "scripts")}, "Training source changed")
        for entry in entries:
            ledger.guard()
            from spindle_power.load_prediction.support_fill_snapshot import _inside
            check_reference(dict(path=str(_inside(snapshot_path.parent, entry["file"])), sha256=entry["sha256"]))
        for kind, epoch in PLAN:
            checked_milestone(output / kind / f"epoch_{epoch:04d}" / "receipt.json", kind, epoch, stage_identities[kind], output)
        status = "completed"
        save_json(output / "scientific_screening.json", dict(summary=screening_summary(screening, completed=True), comparisons=screening))
        result = dict(status=status, passed=True, milestones=receipts, shared_main=reference(common / "shared_main.pt"),
                      main_selected_epoch=shared["selected_epoch"], scientific_comparison="fixed_epoch_Q_vs_P",
                      scientific_screening=screening, scientific_screening_artifact=reference(output / "scientific_screening.json"),
                      training_registered=False, independent_test_used=False)
    except BaseException as exc:
        status = "budget_stopped" if isinstance(exc, BudgetStop) else "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
        result = dict(status=status, passed=False, stage=active_stage, error_type=type(exc).__name__, error=str(exc),
                      milestones=receipts, training_registered=False, independent_test_used=False,
                      scientific_screening=dict(status="not_ready", decision="undetermined", reason="training_incomplete",
                                                completed_comparisons=screening),
                      recovery="Explicit --resume uses the last archived optimizer/RNG state; never a best-state checkpoint.")
    result.update(schema="paper1.qp_report.1", run_fingerprint=digest(identity), elapsed_seconds=ledger.elapsed(),
                  scientific_screening_summary=screening_summary(screening, completed=status == "completed"),
                  primary="fixed_100_and_200", supplementary="best_validation_macro_rmse",
                  required_schedule=[list(item) for item in PLAN], gate_references=gate_refs)
    save_json(output / "report.json", result)
    save_json(output / "state.json", dict(status=status, passed=result["passed"], stage=active_stage,
              completed_milestones=[(row["kind"], row["epoch"]) for row in receipts],
              elapsed_seconds=ledger.elapsed(), training_registered=False))
    ledger.close(status)
    return result


def supervise(args):
    """Bound only this worker; the parent can finalize an exact timeout receipt."""
    import subprocess
    from spindle_power.load_prediction.common import save_json

    started = time.monotonic()
    output = args.output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / ("." + output.name + ".qp.lock")
    token = uuid.uuid4().hex
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(dict(token=token, pid=os.getpid(), started_at=now()), stream)
    try:
        require(output.exists() is args.resume, "Use a new output directory or explicit --resume")
        budget_path = output / "budget.json"
        prior = read_json(budget_path) if args.resume else None
        require(prior is None or prior.get("active") is None, "Unresolved active budget session; explicit audit is required")
        spent = 0. if prior is None else prior["spent_seconds"]
        previous_sessions = 0 if prior is None else len(prior["sessions"])
        require(spent < HARD_SECONDS, "Eight-hour budget is exhausted")
        command = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--worker", "--execute",
                   "--snapshot", str(args.snapshot.resolve()), "--execution-profile", str(args.execution_profile.resolve()),
                   "--output-dir", str(output), "--started-monotonic", str(started)]
        if args.resume:
            command.append("--resume")
        if args.supplementary_protocol:
            command.extend(("--supplementary-protocol", str(args.supplementary_protocol.resolve())))
        timed_out = False
        try:
            result = subprocess.run(command, cwd=ROOT, timeout=max(.001, HARD_SECONDS - spent - (time.monotonic() - started)),
                                    check=False, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            code = result.returncode
        except subprocess.TimeoutExpired:
            # subprocess.run kills and waits for its own worker only.
            timed_out, code = True, 124
        elapsed = time.monotonic() - started
        if budget_path.exists() or timed_out:
            value = (read_json(budget_path) if budget_path.exists() else
                     dict(schema="paper1.qp_budget.1", fingerprint="worker_never_initialized", spent_seconds=spent,
                          soft_seconds=SOFT_SECONDS, hard_seconds=HARD_SECONDS, sessions=[], active=None))
            if timed_out:
                active = value.get("active") or dict(session_id=token, started_at=now())
                session = {**active, "elapsed_seconds": elapsed, "finished_at": now(),
                           "status": "budget_exhausted", "exit_code": 124}
                if value.get("active") is None and len(value["sessions"]) > previous_sessions:
                    value["sessions"][-1].update(session)
                else:
                    value["sessions"].append(session)
                value["active"] = None
                value["spent_seconds"] = max(HARD_SECONDS, spent + elapsed)
            elif value.get("active") is None:
                # Include parent startup and finalization, without double-counting child time.
                if len(value["sessions"]) > previous_sessions:
                    value["sessions"][-1]["elapsed_seconds"] = elapsed
                else:
                    value["sessions"].append(dict(session_id=token, elapsed_seconds=elapsed,
                                                   status="worker_preflight_failed", exit_code=code))
                value["spent_seconds"] = spent + elapsed
            # Unexpected worker crashes deliberately leave active unresolved.
            save_json(budget_path, value)
        if timed_out:
            output.mkdir(parents=True, exist_ok=True)
            partial = read_json(output / "report.json") if (output / "report.json").exists() else {}
            partial.update(schema="paper1.qp_report.1", status="budget_exhausted", passed=False,
                           error="Supervisor stopped only this worker at the eight-hour cumulative limit",
                           exit_code=124, elapsed_seconds=spent + elapsed, training_registered=False,
                           required_schedule=[list(item) for item in PLAN],
                           scientific_screening=dict(status="not_ready", decision="undetermined", reason="hard_budget_exhausted"),
                           scientific_screening_summary=screening_summary({}, completed=False))
            save_json(output / "report.json", partial)
            save_json(output / "state.json", dict(status="budget_exhausted", passed=False, exit_code=124,
                                                  elapsed_seconds=spent + elapsed, training_registered=False))
            save_json(output / "hard_budget_stop.json", dict(schema="paper1.qp_hard_stop.1", at=now(),
                      elapsed_seconds=spent + elapsed, exit_code=124, automatic_restart=False))
        return code
    finally:
        if lock.is_file() and read_json(lock).get("token") == token:
            lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "execution-profile", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--supplementary-protocol", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--started-monotonic", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.execute and not args.worker:
        return supervise(args)
    result = run(args)
    print(json.dumps({key: result[key] for key in ("status", "passed", "elapsed_seconds", "schedule") if key in result},
                     ensure_ascii=False, default=str), flush=True)
    return 0 if result["status"] in ("planned_not_executed", "completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
