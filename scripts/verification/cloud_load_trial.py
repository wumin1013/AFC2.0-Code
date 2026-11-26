"""Portable, bounded engineering trial of the shared load trainer.

This copies four complete frozen load records, never signs a formal ready, and
does not run a nine-fold experiment or measure held-out scientific accuracy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")
DEFAULT_RECORD_IDS = ["R01_外形", "R16_混合", "R52_混合", "R74_外形_recollect_20261015"]
TRIAL_SCHEMA = "afc.cloud_load_trial.1"
ROUTE = "nominal_programmed_residual_v1"


def _local_file(root, relative):
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError("Trial files must use safe relative paths")
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Trial file escapes its root")
    return resolved


def _read_frame(root, entry):
    import pandas as pd
    from spindle_power.load_prediction.common import file_hash
    path = _local_file(root, entry["file"])
    if file_hash(path) != entry["sha256"]:
        raise ValueError(f"Frozen record hash mismatch: {entry['file']}")
    frame = pd.read_csv(path, float_precision="round_trip")
    if list(frame.columns) != list(entry["dtypes"]):
        raise ValueError("Frozen record columns/order mismatch")
    for column, dtype in entry["dtypes"].items():
        if dtype == "bool" and (frame[column].isna().any() or not frame[column].isin([True, False]).all()):
            raise ValueError(f"Invalid boolean mask: {column}")
        frame[column] = frame[column].astype(dtype)
    if {k: str(v) for k, v in frame.dtypes.items()} != entry["dtypes"]:
        raise ValueError("Frozen record dtype mismatch")
    return frame


def prepare(args):
    from spindle_power.load_prediction.common import file_hash, save_json
    from spindle_power.load_prediction.training_checkpoint import digest
    source, destination = Path(args.snapshot_dir).resolve(), Path(args.output_dir).resolve()
    if destination.is_relative_to(source):
        raise ValueError("Trial output must not modify the source snapshot")
    manifest_path = source / "snapshot.json"
    parent_sha = file_hash(manifest_path)
    parent = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = digest({k: v for k, v in parent.items() if k != "snapshot_id"})
    if parent.get("snapshot_id") != identity:
        raise ValueError("Parent snapshot manifest identity mismatch")
    wanted = args.record_ids or DEFAULT_RECORD_IDS
    if len(wanted) != 4 or len(set(wanted)) != 4:
        raise ValueError("This bounded trial requires exactly four distinct explicit records")
    entries = {entry["metadata"]["record_id"]: entry for entry in parent["records"] if entry["kind"] == "load"}
    missing = set(wanted) - set(entries)
    if missing:
        raise ValueError(f"Requested record IDs are absent; select actual manifest IDs explicitly: {sorted(missing)}")
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for record_id in wanted:
        original = entries[record_id]
        if original["metadata"].get("dataset_role") != "development" or "O1203" in record_id:
            raise ValueError("Only the authorized development records may enter this trial")
        frame = _read_frame(source, original)
        relative = "data/" + Path(original["file"]).name
        target = _local_file(destination, relative)
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(_local_file(source, original["file"]), target)
        if file_hash(target) != original["sha256"]:
            raise ValueError("Trial record copy changed bytes")
        records.append({**original, "file": relative, "parent_file": original["file"],
                        "points": len(frame), "main_points": int(frame.valid_main.sum()),
                        "residual_points": int(frame.valid_residual.sum())})
    trial = dict(schema=TRIAL_SCHEMA, kind="engineering_trial_not_formal_training", route=ROUTE,
                 created_at=datetime.now(timezone.utc).isoformat(), parent_snapshot_id=identity,
                 parent_snapshot_sha256=parent_sha, records=records, full_records=True,
                 sampling_rule="fixed_explicit_record_ids_without_outcome_selection", formal_ready=False,
                 scope="CUDA execution, finite optimization, checkpoint/reload and timing only; no nine-fold accuracy")
    trial["trial_id"] = digest(trial)
    save_json(destination / "trial.json", trial)
    result = dict(trial_id=trial["trial_id"], records=wanted, points=sum(r["points"] for r in records),
                  main_points=sum(r["main_points"] for r in records), residual_points=sum(r["residual_points"] for r in records),
                  directory=str(destination), bytes=sum(_local_file(destination, r["file"]).stat().st_size for r in records))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def run(args):
    # Set the process policy before importing NumPy or Torch. No dependency changes.
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, str(args.threads))
    import numpy as np
    import torch
    from spindle_power.load_prediction.common import file_hash, save_json
    from spindle_power.load_prediction.condition import fit_condition_spec
    from spindle_power.load_prediction.load_device import configure_load_device
    from spindle_power.load_prediction.models import ModelConfig, build_load_models, load_models, predict_cutting_power
    from spindle_power.load_prediction.neural import state_version
    from spindle_power.load_prediction.sample_rules import continuous_ids
    from spindle_power.load_prediction.samples import SampleSet, load_input_view, PHYSICS_COLUMNS, NOMINAL_GEOMETRY_COLUMNS
    from spindle_power.load_prediction.training import TrainingConfig, train_two_stage
    from spindle_power.load_prediction.training_checkpoint import digest, source_fingerprints
    source, destination = Path(args.trial_dir).resolve(), Path(args.output_dir).resolve()
    if destination.is_relative_to(source):
        raise ValueError("Run output must be outside the immutable trial input")
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        if not 2 <= args.epochs <= 5 or not 1 <= args.threads <= 32:
            raise ValueError("Engineering trial is bounded to 2-5 epochs and 1-32 CPU threads")
        trial = json.loads((source / "trial.json").read_text(encoding="utf-8"))
        if trial.get("schema") != TRIAL_SCHEMA or trial.get("trial_id") != digest({k: v for k, v in trial.items() if k != "trial_id"}):
            raise ValueError("Trial manifest identity mismatch")
        if len(trial["records"]) != 4 or trial.get("formal_ready") is not False:
            raise ValueError("Expected four engineering-only records")
        samples = [SampleSet(_read_frame(source, entry), entry["metadata"]) for entry in trial["records"]]
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        _, execution = configure_load_device(args.device)
        source_identity = source_fingerprints()
        runner_sha = file_hash(__file__)
        capacity = int(args.capacity[1:])
        model_config = ModelConfig(route=ROUTE, coefficient_widths=(capacity, capacity), hidden=capacity,
                                   condition_spec=fit_condition_spec([s.metadata for s in samples]))
        config = TrainingConfig(main_epochs=args.epochs, residual_epochs=args.epochs, device=args.device, seed=42,
                                residual_batch_windows=args.batch_windows, residual_batch_strategy=args.batch_strategy)
        torch.manual_seed(config.seed)
        initial = build_load_models(model_config, device=args.device)
        initial_versions = {name: state_version(getattr(initial, name)) for name in ("main", "residual")}
        del initial
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        events, last_event = [], time.perf_counter()
        def progress(event):
            nonlocal last_event
            if args.device == "cuda":
                torch.cuda.synchronize()
            now = time.perf_counter()
            row = {**event, "seconds_since_previous_event": now - last_event,
                   "first_epoch_includes_stage_setup": event.get("epoch") == 1}
            last_event = now
            events.append(row)
            with (destination / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
        training_start = time.perf_counter()
        bundle = train_two_stage(samples, model_config=model_config, config=config, execute=True,
                                 checkpoint_dir=destination / "checkpoints", progress=progress,
                                 checkpoint_context=dict(engineering_trial=trial["trial_id"], runner_sha256=runner_sha))
        if args.device == "cuda":
            torch.cuda.synchronize()
        training_seconds = time.perf_counter() - training_start
        final_versions = {name: state_version(getattr(bundle, name)) for name in ("main", "residual")}
        if any(final_versions[name] == initial_versions[name] for name in initial_versions):
            raise ValueError("Trial did not update both physical and residual weights")
        if any(parameter.requires_grad for name in ("main", "residual") for parameter in getattr(bundle, name).parameters()):
            raise ValueError("Completed networks were not frozen")
        bundle.save(destination / "trial_model.pt")
        restored = load_models(destination / "trial_model.pt", device=args.device)
        prediction_checks = []
        for sample in samples:
            frame = load_input_view(sample, "programmed").frame
            columns = PHYSICS_COLUMNS + ["curvature_per_mm", "block_length_mm"] + NOMINAL_GEOMETRY_COLUMNS
            known = np.isfinite(frame[columns].to_numpy(float)).all(axis=1)
            frame["sequence_id"] = continuous_ids(frame, known, known_only=True)
            selected = frame.loc[known].reset_index(drop=True)
            expected = predict_cutting_power(bundle, selected, context=sample.metadata)["cutting_quantiles_w"]
            actual = predict_cutting_power(restored, selected, context=sample.metadata)["cutting_quantiles_w"]
            if not np.array_equal(expected, actual) or not np.isfinite(actual).all() or not (np.diff(actual, axis=1) >= 0).all():
                raise ValueError("Reload, finite prediction, or quantile order verification failed")
            prediction_checks.append(dict(record_id=sample.metadata["record_id"], points=len(selected),
                prediction_sha256=hashlib.sha256(actual.tobytes()).hexdigest(), reload_exact=True, quantiles_ordered=True))
        checkpoint_checks = []
        for stage in ("main", "residual"):
            path = destination / "checkpoints" / (stage + ".pt")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            state = payload["state"]
            steps = [int(item["step"]) for item in state["optimizer"]["state"].values() if "step" in item]
            gradients = [parameter.grad for parameter in getattr(bundle, stage).parameters() if parameter.grad is not None]
            finite_gradients = bool(gradients) and all(bool(torch.isfinite(value).all()) for value in gradients)
            if state["optimizer_updates"] != args.epochs or not steps or set(steps) != {args.epochs} or not finite_gradients:
                raise ValueError("Optimizer update count or finite gradient verification failed")
            if args.device == "cuda" and not state.get("cuda_rng"):
                raise ValueError("CUDA RNG missing in checkpoint")
            checkpoint_checks.append(dict(stage=stage, epoch=state["epoch"], optimizer_updates=state["optimizer_updates"],
                finite_gradients=finite_gradients, cuda_rng_saved=bool(state.get("cuda_rng")),
                sha256=file_hash(path), fingerprint=payload["fingerprint"]))
        if source_fingerprints() != source_identity or file_hash(__file__) != runner_sha:
            raise ValueError("Source changed while trial was running; result cannot be frozen")
        timings = {}
        for stage in ("main", "residual"):
            steady = [event["seconds_since_previous_event"] for event in events if event["stage"] == stage and event["epoch"] > 1]
            timings[stage] = dict(complete_epoch_seconds=steady, median_seconds=float(np.median(steady)),
                                 note="Includes immutable checkpoint/archive I/O; excludes epoch1 stage setup")
        result = dict(schema=TRIAL_SCHEMA, status="engineering_trial_passed", formal_ready=False,
            scientific_accuracy_verified=False, trial_id=trial["trial_id"], parent_snapshot_id=trial["parent_snapshot_id"],
            actual_device=str(bundle.device), capacity=args.capacity, epochs=args.epochs, execution=execution,
            residual_batch_windows=args.batch_windows,
            residual_batch_strategy=args.batch_strategy,
            python=platform.python_version(), executable=sys.executable, module_source=str(ROOT / "src"),
            runner_sha256=runner_sha, source_identity=source_identity, initial_versions=initial_versions,
            final_versions=final_versions, main_frozen_during_residual=bundle.main_version == bundle.residual_main_version,
            training_seconds=training_seconds, total_seconds=time.perf_counter() - started, timings=timings,
            maximum_cuda_memory_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
            predictions=prediction_checks, checkpoints=checkpoint_checks, model_sha256=file_hash(destination / "trial_model.pt"),
            limitation="Four fixed complete records, no validation fold; verifies execution and cost only, not prediction accuracy")
        save_json(destination / "result.json", result)
        print(json.dumps({k: result[k] for k in ("status", "actual_device", "capacity", "training_seconds", "timings")}), flush=True)
        return result
    except BaseException as exc:
        save_json(destination / "failure.json", dict(status="engineering_trial_failed", error_type=type(exc).__name__,
                  error=str(exc), elapsed_seconds=time.perf_counter() - started, formal_ready=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--snapshot-dir", required=True)
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument("--record-ids", nargs=4)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--trial-dir", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    run_parser.add_argument("--epochs", type=int, default=3)
    run_parser.add_argument("--capacity", choices=("C32", "C64"), default="C32")
    run_parser.add_argument("--threads", type=int, default=4)
    run_parser.add_argument("--batch-windows", type=int, default=64)
    run_parser.add_argument("--batch-strategy", choices=("equal_length", "padded"), default="equal_length")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
