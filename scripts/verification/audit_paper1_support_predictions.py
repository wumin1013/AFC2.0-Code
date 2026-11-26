"""Audit all 450 development records with their original held-out C64 models.

This runs inference only. Each original fold supplies its own frozen model and
metric settings. The parent-context compatibility policy is explicitly distinct
from a single prediction pass over the filled input. No memory is updated and no
training, model registration, event-data read, or automatic resume is provided.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_reference(path, expected=None):
    from spindle_power.load_prediction.common import file_hash

    path = Path(path).resolve()
    actual = file_hash(path)
    require(expected is None or actual == expected, f"Frozen artifact hash mismatch: {path}")
    return dict(path=str(path), sha256=actual)


def recheck_references(references):
    for reference in references:
        file_reference(reference["path"], reference["sha256"])


def frozen_archive_bindings(seal_path, snapshot_id):
    """Follow the accepted seal to its frozen, content-bound source inventory."""
    seal_ref = file_reference(seal_path)
    seal = read_json(seal_ref["path"])
    require(seal.get("schema") == "paper1.k0_seal.1" and seal.get("status") == "accepted"
            and seal.get("initial_training_complete") is True and seal.get("initial_evaluation_complete") is True,
            "Original model seal is not accepted")
    require(seal.get("snapshot_id") == snapshot_id, "Frozen seal binds a different parent snapshot")
    audit_name = "frozen_exports/audit.json"
    expected = seal.get("frozen_export_files", {}).get(audit_name)
    require(isinstance(expected, str) and len(expected) == 64, "Frozen seal lacks its audit hash")
    audit_ref = file_reference(Path(seal_ref["path"]).parent / audit_name, expected)
    audit = read_json(audit_ref["path"])
    require(audit.get("schema") == "paper1.initial_cpu_audit.1" and audit.get("status") == "k0_accepted"
            and audit.get("snapshot_id") == snapshot_id, "Frozen audit is not accepted for the parent snapshot")
    require({f"C64:F{i}" for i in range(1, 10)} <= set(audit.get("completed_folds", []))
            and audit.get("pending_folds") == [], "Frozen audit lacks all nine completed C64 folds")
    inventory = audit.get("source_artifacts")
    require(isinstance(inventory, dict) and inventory, "Frozen audit lacks source artifact bindings")
    sources = {}
    for path, digest in inventory.items():
        resolved = Path(path).resolve()
        require(resolved not in sources, "Frozen source inventory repeats a canonical path")
        require(isinstance(digest, str) and len(digest) == 64, "Invalid frozen source hash")
        sources[resolved] = digest
    references = dict(frozen_seal=seal_ref, frozen_audit=audit_ref)
    recheck_references(references.values())
    return references, sources


def sealed_reference(path, sources, *, verify=True):
    path = Path(path).resolve()
    require(path in sources, f"Artifact is absent from the frozen audit: {path}")
    return (file_reference(path, sources[path]) if verify else
            dict(path=str(path), sha256=sources[path]))


def archived_fold_records(run, parent_entries, sources):
    """Bind every record to its original held-out fold before loading a model."""
    from spindle_power.load_prediction.experiment_protocol import protocol_folds

    run = Path(run).resolve()
    mapped = {}
    for fold in protocol_folds():
        directory = run / "folds" / "C64" / fold["fold"]
        model_ref = sealed_reference(directory / "load.pt", sources)
        history_ref = sealed_reference(directory / "load_history.json", sources)
        evaluation_ref = sealed_reference(directory / "nominal_no_memory" / "evaluation.json", sources)
        history = read_json(history_ref["path"])
        require(history.get("rounds") == fold["train_rounds"], "Frozen model training rounds differ from its fold")
        report = read_json(evaluation_ref["path"])
        recheck_references((history_ref, evaluation_ref))
        require(len(report["records"]) == 50, "Each archived C64 fold must contain 50 records")
        for record in report["records"]:
            rid = record["record_id"]
            require(rid in parent_entries and rid not in mapped, "Missing or duplicated archived record")
            meta = parent_entries[rid]["metadata"]
            require(meta["round_id"] in fold["validation_rounds"], "Archived record is not held out by this fold")
            require(record.get("round_id") == meta["round_id"] and record["feature"] == meta["feature"],
                    "Archive metadata differs from the frozen parent")
            require(record.get("feed_mode") == "programmed", "Only prospective nominal archives are allowed")
            require(record.get("prediction_file") and record.get("prediction_sha256"), "Missing archived prediction binding")
            prediction_path = (directory / "nominal_no_memory" / record["prediction_file"]).resolve()
            require(prediction_path.is_relative_to(directory / "nominal_no_memory"),
                    "Archived prediction path escapes its fold")
            prediction_ref = sealed_reference(prediction_path, sources, verify=False)
            require(record["prediction_sha256"] == prediction_ref["sha256"],
                    "Archived prediction differs from its sealed binding")
            mapped[rid] = dict(fold=fold["fold"], model=directory / "load.pt",
                               directory=directory / "nominal_no_memory", record=record,
                               bindings=dict(model=model_ref, history=history_ref, evaluation=evaluation_ref,
                                             prediction=prediction_ref))
    require(set(mapped) == set(parent_entries) and len(mapped) == 450, "Exactly 450 original fold assignments required")
    return mapped


def compare_archive(parent_frame, merged, archived):
    """Use the entire old score mask, never an intersection with a new mask."""
    import numpy as np
    import pandas as pd

    keys = ["line", "execution_instance", "segment_mm", "path_mm"]
    require(len(parent_frame) == len(merged) == len(archived), "Historical full grid length changed")
    for key in keys:
        for other in (merged, archived):
            pd.testing.assert_series_equal(parent_frame[key].reset_index(drop=True), other[key].reset_index(drop=True),
                                           check_names=False, check_dtype=False, check_exact=True)
    for column in ("power_cut_w", "scored"):
        pd.testing.assert_series_equal(merged[column], archived[column], check_names=False,
                                       check_dtype=False, check_exact=True)
    old = archived.prediction_available.to_numpy(bool)
    available = merged.prediction_available.to_numpy(bool)
    require(not np.any(old & ~available), "Previously predicted historical point lost")
    require(np.isfinite(merged.loc[available, "cutting_power_w"]).all(), "Non-finite available prediction")
    reference = archived.cutting_power_w.to_numpy(float)[old]
    current = merged.cutting_power_w.to_numpy(float)[old]
    delta = np.abs(current - reference)
    # Historical hardware/batching is not the same-config reproduction bound.
    # The wrapper separately computes that bound using fresh repeated inference.
    return dict(original_predicted_points=int(old.sum()), original_scored_points=int(archived.scored.sum()),
                original_grid_and_score_mask_preserved=True,
                historical_archive_max_abs_difference_w=float(delta.max()) if delta.size else 0.,
                historical_archive_comparison="traceability_only_same_config_gate_is_separate")


def compare_previous_predictions(original, previous, current):
    """Describe three fixed domains; prior restored-context changes are not accuracy gains."""
    import numpy as np
    import pandas as pd
    from spindle_power.load_prediction.support_prediction import COMPONENTS, QUANTILES, KEYS
    require(len(original) == len(previous) == len(current), "Previous prediction grid length changed")
    for name in KEYS + ["power_cut_w", "scored"]:
        pd.testing.assert_series_equal(previous[name], current[name], check_names=False, check_dtype=False, check_exact=True)
    masks = [frame.prediction_available.to_numpy(bool) for frame in (original, previous, current)]
    old, prior, now = masks
    require(not (old & ~prior).any() and not (prior & ~now).any(), "A historical prediction domain was dropped")
    domains = dict(original_parent=old, v1_restored=prior & ~old, v2_newly_restored=now & ~prior)
    components = COMPONENTS + (QUANTILES if "q05_w" in current else [])
    changes = {}
    for domain in ("original_parent", "v1_restored"):
        mask = domains[domain]
        changes[domain] = {}
        for name in components:
            left, right = previous.loc[mask, name].to_numpy(float), current.loc[mask, name].to_numpy(float)
            require(np.isfinite(left).all() and np.isfinite(right).all(), "Prior/current available predictions must be finite")
            difference = right - left
            changes[domain][name] = dict(points=len(difference), changed_points=int(np.count_nonzero(difference)),
                max_abs_w=float(np.max(np.abs(difference))) if len(difference) else 0.,
                squared_difference_sum_w2=float(np.sum(difference ** 2)),
                prediction_difference_rmse_w=float(np.sqrt(np.mean(difference ** 2))) if len(difference) else 0.)
    for name in components:
        require(np.isfinite(current.loc[domains["v2_newly_restored"], name]).all(), "New restored predictions must be finite")
    return dict(domains={key: int(value.sum()) for key, value in domains.items()}, component_differences=changes,
        fixed_score_mask_preserved=True, interpretation="v1-restored predictions may change with candidate context; "
        "differences are not accuracy improvement; original-parent same-context acceptance remains separate")


def run(args):
    import pandas as pd
    import torch
    from spindle_power.load_prediction.common import file_hash, save_json
    from spindle_power.load_prediction.load_device import configure_load_device
    from spindle_power.load_prediction.models import load_models
    from spindle_power.load_prediction.samples import SampleSet
    from spindle_power.load_prediction.support_prediction import evaluate_support_fill_record
    from spindle_power.load_prediction.training_checkpoint import source_fingerprints
    from audit_paper1_support_fill import read_snapshot, development_records, checked_frame

    start = time.perf_counter()
    script_bindings = {name: file_reference(Path(__file__).with_name(name)) for name in
                       ("audit_paper1_support_predictions.py", "audit_paper1_support_fill.py")}
    parent_path, candidate_path = args.parent_snapshot.resolve(), args.candidate_snapshot.resolve()
    parent, candidate = read_snapshot(parent_path), read_snapshot(candidate_path)
    parents, candidates = development_records(parent), development_records(candidate)
    require(set(parents) == set(candidates), "Candidate membership changed")
    hashes = dict(parent=file_hash(parent_path), candidate=file_hash(candidate_path))
    data_audit = read_json(args.data_audit)
    require(data_audit.get("schema") == "paper1.support_fill_data_audit.1"
            and data_audit.get("status") == "data_passed_prediction_pending"
            and data_audit.get("data_gate_passed") is True and data_audit.get("checked_records") == 450
            and not data_audit.get("failures"), "Independent 450-record data audit must pass first")
    require(data_audit["parent_snapshot"]["sha256"] == hashes["parent"]
            and data_audit["candidate_snapshot"]["sha256"] == hashes["candidate"], "Data audit binds different snapshots")
    previous_report, previous_rows, previous_refs = None, {}, []
    previous_path = getattr(args, "previous_prediction_audit", None)
    if previous_path is not None:
        previous_ref = file_reference(previous_path)
        previous_refs.append(previous_ref)
        previous_report = read_json(previous_path)
        require(previous_report.get("schema") == "paper1.support_fill_prediction_audit.1"
                and previous_report.get("status") == "passed" and previous_report.get("passed") is True
                and previous_report.get("checked_records") == 450 and previous_report.get("failures") == []
                and previous_report.get("parent_snapshot") == data_audit["parent_snapshot"]
                and previous_report.get("candidate_snapshot") == data_audit.get("previous_support_snapshot")
                and previous_report.get("data_audit") == data_audit.get("previous_data_audit"),
                "Previous prediction audit is not bound to the accepted previous support data")
        previous_rows = {row["record_id"]: row for row in previous_report["records"]}
        require(len(previous_report["records"]) == 450 and set(previous_rows) == set(parents), "Previous prediction membership changed")
    profile = read_json(args.execution_profile)
    require(profile.get("schema") == "afc.paper1_execution_profile.1"
            and profile.get("status") == "stage0_engineering_accepted", "Stage-0 profile is not accepted")
    evidence = profile["selection_evidence"]
    require(file_hash(evidence["path"]) == evidence["sha256"]
            and read_json(evidence["path"])["status"] == "stage0_passed", "Stage-0 evidence changed or failed")
    execution = profile["training_config"]
    # Accept only the frozen CPU profile for this development audit.
    require(execution.get("device") == "cpu", "This audit requires the stage-0 frozen CPU profile")
    threads = execution.get("cpu_threads", execution.get("threads"))
    require(type(threads) is int and threads > 0, "Missing frozen thread count")
    require(execution.get("cpu_interop_threads") == 1, "Frozen CPU interop thread count must be one")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    _, execution_identity = configure_load_device("cpu")
    frozen_bindings, frozen_sources = frozen_archive_bindings(args.frozen_seal, parent["snapshot_id"])
    assignments = archived_fold_records(args.frozen_run, parents, frozen_sources)
    archive_bindings = {reference["path"]: reference for entry in assignments.values()
                        for reference in entry["bindings"].values()}
    destination = args.output_dir.resolve()
    require(not destination.exists(), "Refusing to overwrite or automatically resume an audit")
    require(not any(destination.is_relative_to(path.parent) or path.is_relative_to(destination)
                    for path in (parent_path, candidate_path)), "Audit output overlaps a snapshot")
    destination.mkdir(parents=True)
    code_before = source_fingerprints()
    manifest_bindings = dict(parent_snapshot=dict(path=str(parent_path), sha256=hashes["parent"]),
                             candidate_snapshot=dict(path=str(candidate_path), sha256=hashes["candidate"]),
                             data_audit=dict(path=str(args.data_audit.resolve()), sha256=file_hash(args.data_audit)),
                             execution_profile=dict(path=str(args.execution_profile.resolve()), sha256=file_hash(args.execution_profile)),
                             **frozen_bindings)
    if previous_report is not None:
        manifest_bindings["previous_prediction_audit"] = previous_refs[0]
    save_json(destination / "state.json", dict(status="running", completed_records=0, training_started=False,
                                               **manifest_bindings))
    records, failures, models = [], [], {}
    attempted_records = 0
    active_fold, bundle = None, None
    for index, rid in enumerate(sorted(parents, key=lambda name: (assignments[name]["fold"], name))):
        entry = assignments[rid]
        location = destination / "records" / f"record_{index:04d}"
        location.parent.mkdir(exist_ok=True)
        attempted_records += 1
        try:
            if active_fold != entry["fold"]:
                recheck_references((entry["bindings"]["model"],))
                bundle = load_models(entry["model"], device="cpu")
                require(bundle.execution_batch_windows() == execution["residual_batch_windows"]
                        and bundle.execution_batch_strategy() == execution["residual_batch_strategy"],
                        "Original model batching differs from the frozen execution profile; do not mutate its identity")
                active_fold = entry["fold"]
                recheck_references((entry["bindings"]["model"],))
                models[active_fold] = dict(**entry["bindings"]["model"], model_version=bundle.current_version(),
                                          history=entry["bindings"]["history"], evaluation=entry["bindings"]["evaluation"])
            require(bundle.current_version() == models[active_fold]["model_version"], "Frozen model state changed")
            old = checked_frame(parent_path, parents[rid])
            new = checked_frame(candidate_path, candidates[rid])
            archived_ref = entry["bindings"]["prediction"]
            archived_path = Path(archived_ref["path"])
            recheck_references((archived_ref,))
            archived = pd.read_csv(archived_path, float_precision="round_trip")
            recheck_references((archived_ref,))
            result = evaluate_support_fill_record(bundle, SampleSet(old, parents[rid]["metadata"]),
                SampleSet(new, candidates[rid]["metadata"]), output_dir=location,
                metric_parameters=entry["record"]["metric_parameters"])
            require(result.get("passed") is True, "Fixed-context prediction gate failed")
            merged_path = location / result["prediction_file"]
            require(file_hash(merged_path) == result["prediction_sha256"], "Merged predictions changed")
            merged = pd.read_csv(merged_path, float_precision="round_trip")
            archived_check = compare_archive(old, merged, archived)
            previous_comparison = None
            if previous_report is not None:
                prior_row = previous_rows[rid]
                require(prior_row["fold"] == entry["fold"], "Previous prediction fold changed")
                prior_model = previous_report["models"][entry["fold"]]
                require(all(prior_model[key] == entry["bindings"]["model"][key] for key in ("path", "sha256"))
                        and prior_model["model_version"] == bundle.current_version(), "Previous prediction uses a different frozen model")
                prior_record_path = (Path(previous_path).resolve().parent / prior_row["report"]).resolve()
                require(prior_record_path.is_relative_to(Path(previous_path).resolve().parent), "Previous record report escapes its audit")
                previous_refs.append(file_reference(prior_record_path, prior_row["report_sha256"]))
                prior_record = read_json(prior_record_path)
                require(prior_record.get("status") == "passed" and prior_record.get("record_id") == rid
                        and prior_record.get("model_version") == bundle.current_version(), "Previous record/model receipt mismatch")
                prior_artifact = prior_record["artifacts"]["merged"]
                require(Path(prior_artifact["path"]).resolve().is_relative_to(prior_record_path.parent), "Previous prediction escapes its record")
                previous_refs.append(file_reference(prior_artifact["path"], prior_artifact["sha256"]))
                prior_frame = pd.read_csv(prior_artifact["path"], float_precision="round_trip")
                previous_comparison = compare_previous_predictions(archived, prior_frame, merged)
                recheck_references(previous_refs[-2:])
            require(bundle.current_version() == models[active_fold]["model_version"], "Inference changed model weights")
            row = dict(record_id=rid, fold=entry["fold"], passed=True,
                       report=str((location / "report.json").relative_to(destination)),
                       report_sha256=file_hash(location / "report.json"),
                       parent_csv_sha256=parents[rid]["sha256"], candidate_csv_sha256=candidates[rid]["sha256"],
                       archived_prediction_sha256=entry["record"]["prediction_sha256"],
                       archived_prediction=archived_ref,
                       coverage=result["coverage"], historical_comparison=archived_check)
            records.append(row)
            if previous_comparison is not None:
                row["previous_prediction_comparison"] = previous_comparison
            print(json.dumps(dict(record_id=rid, fold=entry["fold"], passed=True,
                                  coverage=result["coverage"]), ensure_ascii=False), flush=True)
        except Exception as exc:
            failure = dict(scope="record", record_id=rid, fold=entry["fold"], error_type=type(exc).__name__, message=str(exc),
                           traceback=traceback.format_exc())
            failures.append(failure)
            print(json.dumps(dict(record_id=rid, passed=False, error=str(exc)), ensure_ascii=False), flush=True)
        save_json(destination / "state.json", dict(status="running", completed_records=len(records),
            failed_records=len(failures), attempted_records=attempted_records, elapsed_seconds=time.perf_counter() - start,
            training_started=False))
        if failures:
            # Do not repeat a systemic error over the remaining records. A
            # partial audit remains failed and cannot issue a training release.
            break
    try:
        recheck_references(manifest_bindings.values())
        recheck_references(script_bindings.values())
        recheck_references(archive_bindings.values())
        recheck_references((evidence,))
        recheck_references(previous_refs)
        require(source_fingerprints() == code_before, "Prediction source changed during audit")
    except Exception as exc:
        failures.append(dict(scope="integrity", error_type=type(exc).__name__, message=str(exc),
                             traceback=traceback.format_exc()))
        print(json.dumps(dict(scope="integrity", passed=False, error=str(exc)), ensure_ascii=False), flush=True)
    passed = len(records) == 450 and not failures
    totals = Counter()
    for row in records:
        for key, value in row["coverage"].items():
            if type(value) is int:
                totals[key] += value
    report = dict(schema="paper1.support_fill_prediction_audit.1", status="passed" if passed else "failed",
                  passed=passed, checked_records=len(records), failures=failures, records=records,
                  attempted_records=attempted_records, unattempted_records=450 - attempted_records,
                  failed_records=sum(item["scope"] == "record" for item in failures),
                  integrity_failures=sum(item["scope"] == "integrity" for item in failures),
                  totals=dict(totals), models=models, **manifest_bindings, source=code_before,
                  audit_scripts=script_bindings, frozen_archive_artifacts=list(archive_bindings.values()),
                  execution=execution_identity, cpu_threads=threads, cpu_interop_threads=1,
                  prediction_policy="support_fill_parent_context_v1", training_started=False,
                  training_released=False, window_unification_required=True,
                  scientific_accuracy_improvement_claimed=False, elapsed_seconds=time.perf_counter() - start)
    if previous_report is not None:
        report["three_prediction_domains"] = {key: sum(row["previous_prediction_comparison"]["domains"][key] for row in records)
                                               for key in ("original_parent", "v1_restored", "v2_newly_restored")}
        report["previous_prediction_artifacts"] = previous_refs
        report["previous_domain_interpretation"] = "All v1 input/label preservation is checked by the data audit; v1-restored prediction " \
            "differences are reported for every record without selecting a score intersection. No three-context merge or accuracy gain is claimed."
    save_json(destination / "report.json", report)
    save_json(destination / "state.json", dict(status=report["status"], passed=passed, checked_records=len(records),
        failed_records=report["failed_records"], integrity_failures=report["integrity_failures"],
        attempted_records=attempted_records, unattempted_records=450 - attempted_records, training_started=False))
    print(json.dumps({key:report[key] for key in ("status", "checked_records", "totals", "elapsed_seconds")}, ensure_ascii=False))
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-snapshot", "candidate-snapshot", "data-audit", "execution-profile", "frozen-run", "frozen-seal", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--previous-prediction-audit", type=Path,
                        help="Accepted v1 report for three-domain counts and candidate-context prediction differences")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
