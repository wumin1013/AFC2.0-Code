"""Audit prospective contexts on 450 frozen development CSVs, without training.

Every record is checked for full-grid coverage and label-independent windows.
Deterministic representatives additionally compare actual residual-forward inputs
against training/validation preparation and repeated, label-perturbed inference.
The old F1 model supplies engineering arithmetic only; it is never exported as a
new trained model or accepted as initialization for the subsequent Q/P study.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
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


def array_hash(value):
    import numpy as np
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def perturb_observed(sample):
    """Deliberately destroy all current measured supervision, not known inputs."""
    import numpy as np
    changed = deepcopy(sample)
    frame = changed.frame
    for column in ("power_cut_w", "reference_power_cut_w"):
        if column in frame:
            frame[column] = np.nan
    for column in ("valid_main", "valid_residual", "aligned", "reference_aligned"):
        if column in frame:
            frame[column] = False
    frame["time_s"] = -np.arange(len(frame), dtype=float)
    frame["training_sequence_id"] = np.arange(len(frame))
    return changed


def context_plan(sample, width):
    from spindle_power.load_prediction.neural import prospective_residual_context
    from spindle_power.load_prediction.samples import load_input_view
    selected, indices, windows = prospective_residual_context(load_input_view(sample, "programmed").frame, width)
    return selected, indices, windows


def audit_plan(sample, width):
    import numpy as np
    from spindle_power.load_prediction.samples import PHYSICS_COLUMNS, NOMINAL_GEOMETRY_COLUMNS, path_features, physical_quantities
    selected, indices, windows = context_plan(sample, width)
    require(len(indices) > 0, "A development record has no known prospective inputs")
    covered = np.concatenate(windows) if windows else np.empty(0, dtype=int)
    require(np.array_equal(covered, np.arange(len(selected))), "Complete windows must cover every known point exactly once")
    known = np.zeros(len(sample.frame), dtype=bool)
    known[indices] = True
    require(not np.any(sample.frame.valid_residual.to_numpy(bool) & ~known),
            "A scored point lacks complete prospective inputs")
    physical_quantities(selected[PHYSICS_COLUMNS].to_numpy(np.float32))
    # Validate all geometry values, not merely their finiteness.
    path_features(selected, np.zeros(len(selected), np.float32), 0., 1., 2000., route="nominal_programmed_residual_v1")
    other, changed_indices, changed_windows = context_plan(perturb_observed(sample), width)
    require(np.array_equal(indices, changed_indices) and len(windows) == len(changed_windows)
            and all(np.array_equal(a, b) for a, b in zip(windows, changed_windows)),
            "Observed fields changed prospective window context")
    inputs = PHYSICS_COLUMNS + ["curvature_per_mm", "block_length_mm"] + NOMINAL_GEOMETRY_COLUMNS
    require(np.array_equal(selected[inputs].to_numpy(float), other[inputs].to_numpy(float)),
            "Observed fields changed prospective input values")
    mask = selected.valid_residual.to_numpy(bool)
    manifest = dict(record_id=sample.metadata["record_id"], grid_points=len(sample.frame),
                    known_points=len(indices), unknown_points=len(sample.frame) - len(indices),
                    window_count=len(windows), zero_supervision_windows=sum(not mask[index].any() for index in windows),
                    partial_supervision_windows=sum(mask[index].any() and not mask[index].all() for index in windows),
                    known_indices=indices.tolist(), windows=[indices[index].tolist() for index in windows],
                    input_sha256=array_hash(selected[inputs].to_numpy(float)),
                    complete_manifest_label_invariant=True, all_known_points_covered_once=True)
    return manifest


def audit_forward_context(bundle, sample, destination):
    import numpy as np
    import torch
    from spindle_power.load_prediction.common import file_hash, save_json
    from spindle_power.load_prediction.neural import predict_cutting_power, state_version, window_batches
    from spindle_power.load_prediction.training import _residual_data

    destination.mkdir(parents=True, exist_ok=False)
    selected, source_indices, windows = context_plan(sample, bundle.config.window_points)
    prepared = _residual_data(bundle, [sample])
    validation = _residual_data(bundle, [sample], validation=True)
    train = prepared[0][1] if prepared else []
    valid = validation[0][1] if validation else []
    require(len(train) == len(valid), "Train and validation have different contexts")
    for left, right in zip(train, valid):
        require(all((a is None and b is None) or (a is not None and b is not None and torch.equal(a, b))
                    for a, b in zip(left, right)), "Train and validation tensors differ")
    groups = list(window_batches(list(enumerate(windows)), bundle.execution_batch_windows(),
        length=lambda item: len(item[1]), strategy=bundle.execution_batch_strategy()))
    observed, cursor = {}, 0
    original = bundle.residual.forward_quantiles
    def capture(x, valid=None, condition=None):
        nonlocal cursor
        require(cursor < len(groups), "Unexpected deployment batch")
        group = groups[cursor]
        cursor += 1
        require(len(x) == len(group), "Deployment batch membership differs")
        for row, (index, window) in enumerate(group):
            length = len(window)
            if valid is not None:
                expected = torch.arange(x.shape[1], device=valid.device) < length
                require(torch.equal(valid[row], expected), "Network mask is not exclusively a length mask")
            observed[index] = (x[row, :length].detach().cpu().clone(),
                               None if condition is None else condition[row, :length].detach().cpu().clone())
        return original(x, valid=valid, condition=condition)
    main_before, residual_before = state_version(bundle.main), state_version(bundle.residual)
    bundle.residual.forward_quantiles = capture
    try:
        baseline = predict_cutting_power(bundle, selected, context=sample.metadata)
    finally:
        bundle.residual.forward_quantiles = original
    require(cursor == len(groups) and len(observed) == len(windows), "Missing actual deployment contexts")
    scoring = selected.valid_residual.to_numpy(bool)
    scored_windows = [(i, index) for i, index in enumerate(windows) if scoring[index].any()]
    require(len(scored_windows) == len(train), "Loss preparation dropped or added a scored window")
    arrays, rows = {}, []
    for index, (x, condition) in observed.items():
        arrays[f"features_{index}"] = x.numpy()
        if condition is not None:
            arrays[f"condition_{index}"] = condition.numpy()
        rows.append(dict(window=index, original_indices=source_indices[windows[index]].tolist(),
                         features_sha256=array_hash(x.numpy()),
                         condition_sha256=None if condition is None else array_hash(condition.numpy()),
                         scored=bool(scoring[windows[index]].any())))
    for (index, _), (x, _, condition, _) in zip(scored_windows, train):
        actual_x, actual_condition = observed[index]
        require(torch.equal(x[0], actual_x), "Training and actual deployment feature contexts differ")
        require((condition is None and actual_condition is None) or
                (condition is not None and actual_condition is not None and torch.equal(condition[0], actual_condition)),
                "Training and actual deployment conditions differ")
    changed, _, _ = context_plan(perturb_observed(sample), bundle.config.window_points)
    perturbed = predict_cutting_power(bundle, changed, context=sample.metadata)
    for key, value in baseline.items():
        if isinstance(value, np.ndarray):
            require(np.array_equal(value, perturbed[key], equal_nan=True), "Observed fields changed actual output: " + key)
            arrays["baseline_" + key] = value
            arrays["perturbed_" + key] = perturbed[key]
    require(main_before == state_version(bundle.main) and residual_before == state_version(bundle.residual),
            "Engineering context audit changed model weights")
    np.savez_compressed(destination / "actual_contexts.npz", **arrays)
    report = dict(record_id=sample.metadata["record_id"], status="passed", passed=True,
                  complete_window_count=len(windows), scored_window_count=len(train),
                  train_validation_context_exact=True, train_deployment_context_exact=True,
                  deployment_outputs_label_invariant=True, network_mask_is_length_only=True,
                  main_weights_unchanged=True, residual_weights_unchanged=True, windows=rows,
                  tensors=dict(path=str(destination / "actual_contexts.npz"), sha256=file_hash(destination / "actual_contexts.npz")))
    save_json(destination / "report.json", report)
    return dict(path=str(destination / "report.json"), sha256=file_hash(destination / "report.json"),
                record_id=sample.metadata["record_id"], passed=True)


def run(args):
    import numpy as np
    import torch
    from spindle_power.load_prediction.common import file_hash, save_json
    from spindle_power.load_prediction.models import load_models
    from spindle_power.load_prediction.neural import state_version
    from spindle_power.load_prediction.samples import SampleSet
    from spindle_power.load_prediction.training_checkpoint import source_fingerprints
    from audit_paper1_support_fill import checked_frame, development_records, read_snapshot
    from audit_paper1_support_predictions import file_reference, read_json, recheck_references

    start = time.perf_counter()
    snapshot_path = args.candidate_snapshot.resolve()
    snapshot = read_snapshot(snapshot_path)
    entries = development_records(snapshot)
    prediction = read_json(args.prediction_audit)
    require(prediction.get("schema") == "paper1.support_fill_prediction_audit.1" and prediction.get("passed") is True
            and prediction.get("status") == "passed" and prediction.get("checked_records") == 450
            and not prediction.get("failures"), "The complete 450-record prediction audit must pass first")
    require(prediction["candidate_snapshot"]["sha256"] == file_hash(snapshot_path), "Prediction audit binds another candidate")
    profile = read_json(args.execution_profile)
    require(profile.get("schema") == "afc.paper1_execution_profile.1"
            and profile.get("status") == "stage0_engineering_accepted", "Stage-0 execution profile must be accepted")
    evidence = profile["selection_evidence"]
    require(file_hash(evidence["path"]) == evidence["sha256"]
            and read_json(evidence["path"]).get("status") == "stage0_passed", "Stage-0 evidence changed or failed")
    execution = profile["training_config"]
    require(execution.get("device") == "cpu" and execution.get("cpu_interop_threads") == 1
            and execution.get("cpu_threads", execution.get("threads")) == args.cpu_threads,
            "Window audit execution must match the frozen stage-0 profile")
    bindings = dict(candidate_snapshot=file_reference(snapshot_path), prediction_audit=file_reference(args.prediction_audit),
                    execution_profile=file_reference(args.execution_profile), selection_evidence=file_reference(evidence["path"], evidence["sha256"]),
                    frozen_f1_model=file_reference(prediction["models"]["F1"]["path"], prediction["models"]["F1"]["sha256"]),
                    audit_script=file_reference(__file__))
    helper_bindings = {name: file_reference(Path(__file__).with_name(name)) for name in
                       ("audit_paper1_support_fill.py", "audit_paper1_support_predictions.py")}
    destination = args.output_dir.resolve()
    require(not destination.exists(), "Refusing to overwrite or resume a window audit")
    require(not destination.is_relative_to(snapshot_path.parent) and not snapshot_path.is_relative_to(destination),
            "Audit output overlaps candidate snapshot")
    destination.mkdir(parents=True)
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    bundle = load_models(bindings["frozen_f1_model"]["path"], device="cpu")
    require(bundle.config.route == "nominal_programmed_residual_v1", "Engineering model must be nominal F1")
    require(bundle.execution_batch_windows() == execution["residual_batch_windows"]
            and bundle.execution_batch_strategy() == execution["residual_batch_strategy"],
            "Frozen F1 batching must match stage-0; do not mutate its execution identity")
    original_version = bundle.current_version()
    main_version, residual_version = state_version(bundle.main), state_version(bundle.residual)
    bundle.metadata = deepcopy(bundle.metadata)
    bundle.metadata.setdefault("config", {})["residual_window_policy"] = "prospective_v1"
    bundle.assert_ready()
    code = source_fingerprints()
    records, failures, representative_ids, representative_reports = [], [], set(), []
    totals = Counter()
    try:
        manifests_dir = destination / "manifests"
        manifests_dir.mkdir()
        for number, (rid, entry) in enumerate(sorted(entries.items())):
            sample = SampleSet(checked_frame(snapshot_path, entry), entry["metadata"])
            manifest = audit_plan(sample, bundle.config.window_points)
            path = manifests_dir / f"record_{number:04d}.json"
            save_json(path, manifest)
            row = {key: value for key, value in manifest.items() if key not in ("known_indices", "windows")}
            row.update(feature=sample.metadata["feature"], manifest=dict(path=str(path), sha256=file_hash(path)),
                       sample_sha256=entry["sha256"])
            records.append(row)
            for key in ("grid_points", "known_points", "unknown_points", "window_count", "zero_supervision_windows", "partial_supervision_windows"):
                totals[key] += int(row[key])
            if (number + 1) % 50 == 0:
                print(json.dumps(dict(phase="full_population_plan", checked_records=number + 1)), flush=True)
        # Each category plus the most extreme missing-input, partial-label and
        # zero-label records. Ties resolve by record ID, never by accuracy.
        features = sorted({row["feature"] for row in records})
        for feature in features:
            representative_ids.add(min(row["record_id"] for row in records if row["feature"] == feature))
        for field in ("unknown_points", "partial_supervision_windows", "zero_supervision_windows", "window_count"):
            representative_ids.add(max(records, key=lambda row: (row[field], row["record_id"]))["record_id"])
        for number, rid in enumerate(sorted(representative_ids)):
            entry = entries[rid]
            sample = SampleSet(checked_frame(snapshot_path, entry), entry["metadata"])
            representative_reports.append(audit_forward_context(bundle, sample, destination / "representatives" / f"record_{number:02d}"))
            print(json.dumps(dict(phase="actual_forward_context", record_id=rid, passed=True)), flush=True)
        require(main_version == state_version(bundle.main) and residual_version == state_version(bundle.residual), "Model weights changed")
        recheck_references(bindings.values())
        recheck_references(helper_bindings.values())
        recheck_references(row["manifest"] for row in records)
        recheck_references(representative_reports)
        recheck_references(read_json(row["path"])["tensors"] for row in representative_reports)
        recheck_references(dict(path=str((snapshot_path.parent / entry["file"]).resolve()), sha256=entry["sha256"])
                           for entry in entries.values())
        require(source_fingerprints() == code, "Source changed during window audit")
    except Exception as exc:
        failures.append(dict(error_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
    passed = len(records) == 450 and len(representative_reports) == len(representative_ids) and bool(representative_reports) and not failures
    report = dict(schema="paper1.prospective_window_audit.1", status="passed" if passed else "failed", passed=passed,
                  **bindings, checked_records=len(records), records=records, totals=dict(totals), failures=failures,
                  window_policy="prospective_v1", window_points=bundle.config.window_points,
                  representative_selection="first_record_per_feature_plus_max_unknown_partial_zero_and_window_count",
                  representative_ids=sorted(representative_ids), representative_reports=representative_reports,
                  engineering_model=dict(original_version=original_version, prospective_execution_version=bundle.current_version(),
                    original_model_sha256=bindings["frozen_f1_model"]["sha256"],
                    main_version=main_version, residual_version=residual_version, purpose="engineering_context_only",
                    accepted_as_training_initialization=False, formal_model_migration=False, exported=False),
                  training_started=False, scientific_accuracy_improvement_claimed=False, source=code,
                  helper_scripts=helper_bindings,
                  cpu_threads=args.cpu_threads, elapsed_seconds=time.perf_counter() - start)
    save_json(destination / "report.json", report)
    print(json.dumps({key: report[key] for key in ("status", "checked_records", "totals", "elapsed_seconds")}), flush=True)
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate-snapshot", "prediction-audit", "execution-profile", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, required=True)
    args = parser.parse_args()
    require(args.cpu_threads > 0, "cpu-threads must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
