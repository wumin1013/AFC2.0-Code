"""Prepare immutable F1 tensors for bounded CPU/CUDA engineering measurements.

This reads the existing numerical snapshot and an old F1 model only. It never
rebuilds labels, trains a model, instantiates a service, or signs formal readiness.
The local .pt artifacts contain trusted SampleSet/Pandas objects and therefore
must be loaded with weights_only=False by the engineering harness.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
SCHEMA = "afc.speed_data.1"
ROUTE = "nominal_programmed_residual_v1"


def _safe_path(root, relative):
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError("Snapshot files must use safe relative paths")
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Snapshot file escapes its directory")
    return resolved


def _read_frame(root, entry):
    import pandas as pd
    from spindle_power.load_prediction.common import file_hash

    path = _safe_path(root, entry["file"])
    if file_hash(path) != entry["sha256"]:
        raise ValueError(f"Frozen record hash mismatch: {entry['file']}")
    frame = pd.read_csv(path, float_precision="round_trip")
    if list(frame.columns) != list(entry["dtypes"]):
        raise ValueError("Frozen record column order mismatch")
    for column, dtype in entry["dtypes"].items():
        if dtype == "bool" and (frame[column].isna().any() or not frame[column].isin([True, False]).all()):
            raise ValueError(f"Invalid boolean mask: {column}")
        frame[column] = frame[column].astype(dtype)
    if {key: str(value) for key, value in frame.dtypes.items()} != entry["dtypes"]:
        raise ValueError("Frozen record dtype mismatch")
    if file_hash(path) != entry["sha256"]:
        raise ValueError("Frozen record changed while being read")
    return frame


def _subset_ids(rows):
    """Choose geometry/coverage representatives, never by prediction error."""
    import numpy as np
    from spindle_power.load_prediction.experiment_protocol import PROGRAM_FEATURES

    chosen, reasons = set(), {}
    for feature in PROGRAM_FEATURES:
        group = [row for row in rows if row["feature"] == feature]
        if not group:
            raise ValueError(f"Missing feature for benchmark subset: {feature}")
        for name, percentile in (("record_length_median", .5), ("record_length_p90", .9)):
            target = float(np.quantile([row["points"] for row in group], percentile))
            row = min(group, key=lambda value: (abs(value["points"] - target), value["record_id"]))
            chosen.add(row["record_id"])
            reasons.setdefault(row["record_id"], []).append(name)
        row = min(group, key=lambda value: (value["residual_points"] / value["points"], value["record_id"]))
        chosen.add(row["record_id"])
        reasons.setdefault(row["record_id"], []).append("sparsest_residual_supervision")
    return chosen, reasons


def prepare(args):
    # Configure the runtime before importing NumPy/Pandas/Torch.
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "4")
    import numpy as np
    import pandas as pd
    import torch
    from spindle_power.load_prediction.common import file_hash, save_json
    from spindle_power.load_prediction.condition import fit_condition_spec
    from spindle_power.load_prediction.experiment_protocol import PROGRAM_FEATURES, protocol_folds
    from spindle_power.load_prediction.load_device import configure_load_device
    from spindle_power.load_prediction.models import build_load_models, load_models
    from spindle_power.load_prediction.neural import state_version
    from spindle_power.load_prediction.samples import PHYSICS_COLUMNS, SampleSet, load_input_view
    from spindle_power.load_prediction.training import _residual_data
    from spindle_power.load_prediction.training_checkpoint import digest, source_fingerprints

    started = time.perf_counter()
    source = Path(args.snapshot_dir).resolve()
    model_path = Path(args.fold_model).resolve()
    output = Path(args.output_dir).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Output must be separate from the immutable snapshot")
    if model_path.is_relative_to(output):
        raise ValueError("Output cannot contain the source fold model")
    manifest_path = source / "snapshot.json"
    manifest_sha, model_sha = file_hash(manifest_path), file_hash(model_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("snapshot_id") != digest({key: value for key, value in manifest.items() if key != "snapshot_id"}):
        raise ValueError("Snapshot manifest identity mismatch")
    if manifest.get("schema") != "afc.training_service.1" or manifest.get("synthetic") is not False:
        raise ValueError("Expected the real frozen numerical training snapshot")
    fold = next(value for value in protocol_folds() if value["fold"] == "F1")
    saved_fold = next(value for value in manifest["protocol"]["folds"] if value["fold"] == "F1")
    if fold != saved_fold:
        raise ValueError("Frozen and current F1 split disagree")
    entries = [entry for entry in manifest["records"] if entry["kind"] == "load"]
    if len(entries) != 450 or len({entry["metadata"]["record_id"] for entry in entries}) != 450:
        raise ValueError("Expected exactly 450 distinct frozen development load records")
    groups = {"train": [], "validation": []}
    for entry in entries:
        meta = entry["metadata"]
        if (meta.get("dataset_role") != "development" or type(meta.get("round_id")) is not int
                or not 1 <= meta["round_id"] <= 90 or not meta["record_id"].startswith("R")
                or "O1203" in meta["record_id"] or meta.get("feature") not in PROGRAM_FEATURES):
            raise ValueError("Only R01-R90 development records may enter the speed trial")
        split = "train" if meta["round_id"] in fold["train_rounds"] else "validation"
        groups[split].append(entry)
    if len(groups["train"]) != 400 or len(groups["validation"]) != 50:
        raise ValueError("F1 must contain 400 training and 50 validation records")
    for split, selected in groups.items():
        wanted = {(round_id, feature) for round_id in fold[split + "_rounds"] for feature in PROGRAM_FEATURES}
        if {(entry["metadata"]["round_id"], entry["metadata"]["feature"]) for entry in selected} != wanted:
            raise ValueError("Incomplete or duplicate round/feature records in F1")

    policy = getattr(args, "window_policy", "legacy_v1")
    window_evidence = None
    if policy == "prospective_v1":
        if not args.window_audit:
            raise ValueError("Prospective measurements require the accepted 450-record window audit")
        audit_path = Path(args.window_audit).resolve()
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (audit.get("schema") != "paper1.prospective_window_audit.1" or audit.get("status") != "passed"
                or audit.get("passed") is not True or audit.get("checked_records") != 450
                or audit.get("failures") != [] or audit.get("source") != source_fingerprints()
                or audit.get("window_policy") != policy
                or audit.get("candidate_snapshot", {}).get("sha256") != manifest_sha
                or audit.get("cpu_threads") != args.threads
                or audit.get("frozen_f1_model", {}).get("sha256") != model_sha):
            raise ValueError("Prospective window evidence does not bind this data, source, model, or execution")
        window_evidence = dict(path=str(audit_path), sha256=file_hash(audit_path))
    _, execution = configure_load_device("cpu", cpu_threads=getattr(args, "threads", 4), cpu_interop_threads=1)
    frozen = load_models(model_path, device="cpu")
    config = frozen.config
    if (config.route != ROUTE or config.hidden != 64 or tuple(config.coefficient_widths) != (64, 64)
            or config.window_points != 256 or config.residual_output != "quantiles"):
        raise ValueError("The engineering source must be the frozen C64 nominal F1 model")
    train_ids = {entry["metadata"]["record_id"] for entry in groups["train"]}
    old_ids = {meta["record_id"] for meta in frozen.metadata.get("samples", [])}
    if old_ids != train_ids or len(frozen.metadata.get("samples", [])) != 400:
        raise ValueError("Source main model provenance is not the exact F1 training split")
    spec = fit_condition_spec([entry["metadata"] for entry in groups["train"]])
    if digest(spec) != digest(config.condition_spec):
        raise ValueError("Old F1 condition normalization differs from current frozen F1 training metadata")
    if policy == "prospective_v1":
        # An in-memory engineering context only: never save this as a migrated
        # scientific model or continue its weights in the Q/P pilot.
        frozen.metadata = deepcopy(frozen.metadata)
        frozen.metadata.setdefault("config", {})["residual_window_policy"] = policy
        if frozen.execution_window_policy() != policy:
            raise ValueError("Prospective source implementation is not active")
    frozen.main.eval()
    source_versions = {name: state_version(getattr(frozen, name)) for name in ("main", "residual")}
    torch.manual_seed(42)
    initial = build_load_models(config, device="cpu")
    initial_states = {name: deepcopy(getattr(initial, name).state_dict()) for name in ("main", "residual")}
    initial_versions = {name: state_version(getattr(initial, name)) for name in ("main", "residual")}
    del initial
    code_identity = source_fingerprints()
    output.mkdir(parents=True, exist_ok=False)
    try:
        main, residual, rows = {}, {}, {}
        for split, selected in groups.items():
            main[split], residual[split], rows[split] = [], [], []
            for index, entry in enumerate(selected, 1):
                sample = SampleSet(_read_frame(source, entry), entry["metadata"])
                frame = sample.frame
                measured = load_input_view(sample, "measured")
                main_frame = measured.frame.loc[measured.frame.valid_main,
                    PHYSICS_COLUMNS + ["power_cut_w", "valid_main"]].reset_index(drop=True).copy()
                if main_frame.empty or not np.isfinite(main_frame[PHYSICS_COLUMNS + ["power_cut_w"]].to_numpy()).all():
                    raise ValueError("F1 record lacks finite qualified main samples")
                main[split].append(SampleSet(main_frame, sample.metadata))
                programmed = load_input_view(sample, "programmed")
                prepared = _residual_data(frozen, [programmed], validation=split == "validation")
                if len(prepared) != 1:
                    raise ValueError("F1 record lacks prepared residual windows")
                windows = prepared[0][1]
                residual[split].append((SampleSet(pd.DataFrame(), sample.metadata), windows))
                rows[split].append(dict(record_id=sample.metadata["record_id"], feature=sample.metadata["feature"],
                    round_id=sample.metadata["round_id"], file=entry["file"], sha256=entry["sha256"],
                    points=len(frame), main_points=len(main_frame), residual_points=int(frame.valid_residual.sum()),
                    residual_windows=len(windows), shortest_window=min(window[0].shape[1] for window in windows),
                    longest_window=max(window[0].shape[1] for window in windows)))
                if index % 25 == 0 or index == len(selected):
                    print(json.dumps(dict(stage="prepare_frozen_data", split=split, records=index,
                        total_records=len(selected), elapsed_seconds=time.perf_counter() - started)), flush=True)
        # Main targets and scale use measured-input F1 training records only.
        targets = np.concatenate([sample.frame.power_cut_w.to_numpy() for sample in main["train"]])
        main_scale = max(1., float(np.std(targets)))
        if not np.isclose(main_scale, frozen.normalization["target_scale_w"], rtol=1e-12, atol=1e-12):
            raise ValueError("Frozen F1 target scale does not match its current training snapshot")
        chosen, reasons = {}, {}
        for split in groups:
            chosen[split], reasons[split] = _subset_ids(rows[split])
        provenance = dict(snapshot_id=manifest["snapshot_id"], snapshot_sha256=manifest_sha,
            snapshot_directory=str(source), fold_model=str(model_path), fold_model_sha256=model_sha,
            frozen_versions=source_versions, initial_versions=initial_versions, code_identity=code_identity,
            fold=fold, seed=42, execution=execution, source_records=rows,
            subset_selection="per-feature nearest record-length median/p90 plus sparsest residual supervision; record-id tie break; deduplicated",
            subset_reasons=reasons,
            frozen_main_usage="engineering residual feature/normalization preparation only; never a fresh scientific F1 training result",
            window_policy=policy, window_audit=window_evidence, training_performed=False,
            created_at=datetime.now(timezone.utc).isoformat())
        common = dict(schema=SCHEMA, model_config=asdict(config),
            initial_main=initial_states["main"], initial_residual=initial_states["residual"],
            normalization=deepcopy(frozen.normalization), main_scale=main_scale, provenance=provenance,
            formal_ready=False, scientific_accuracy_verified=False)
        artifacts = {}
        for name in ("full", "subset"):
            payload = dict(common, scope=name)
            for split in groups:
                wanted = train_ids if split == "train" else {row["record_id"] for row in rows[split]}
                if name == "subset":
                    wanted = chosen[split]
                payload["main_" + split] = [sample for sample in main[split] if sample.metadata["record_id"] in wanted]
                payload["residual_" + split] = [item for item in residual[split] if item[0].metadata["record_id"] in wanted]
            path = output / (name + ".pt")
            torch.save(payload, path)
            artifacts[name] = dict(file=path.name, sha256=file_hash(path), bytes=path.stat().st_size,
                train_records=len(payload["main_train"]), validation_records=len(payload["main_validation"]))
        if (file_hash(manifest_path) != manifest_sha or file_hash(model_path) != model_sha
                or source_fingerprints() != code_identity):
            raise ValueError("Input/source identity changed during preparation")
        if window_evidence is not None and file_hash(window_evidence["path"]) != window_evidence["sha256"]:
            raise ValueError("Accepted window evidence changed during preparation")
        if any(state_version(getattr(frozen, name)) != source_versions[name] for name in source_versions):
            raise ValueError("Frozen source model changed during preparation")
        report = dict(schema=SCHEMA, status="engineering_data_prepared", formal_ready=False,
            scientific_accuracy_verified=False, artifacts=artifacts, provenance=provenance,
            elapsed_seconds=time.perf_counter() - started)
        save_json(output / "data_manifest.json", report)
        print(json.dumps(dict(status=report["status"], artifacts=artifacts,
            elapsed_seconds=report["elapsed_seconds"]), ensure_ascii=False), flush=True)
        return report
    except BaseException as exc:
        save_json(output / "failure.json", dict(status="engineering_data_preparation_failed",
            error_type=type(exc).__name__, error=str(exc), elapsed_seconds=time.perf_counter() - started))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--fold-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--window-policy", choices=["legacy_v1", "prospective_v1"], default="legacy_v1")
    parser.add_argument("--window-audit")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
