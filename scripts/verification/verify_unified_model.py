"""Focused local checks; no training and no independent accuracy claim."""
from __future__ import annotations

import argparse
from copy import deepcopy
import io
import importlib.metadata
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd
import torch

from spindle_power.load_prediction.common import file_hash, save_json
from spindle_power.load_prediction.base_power import BasePowerModel, predict_base_event
from spindle_power.load_prediction.models import load_models, predict_cutting_power
from spindle_power.load_prediction.neural import prospective_residual_context
from spindle_power.load_prediction.unified import CONTEXT_DEFAULT, UnifiedSpindleModel, predict_files, prepare_inputs, read_context


def main():
    parser = argparse.ArgumentParser()
    for name in ("model", "cutting", "base", "nc", "process", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    checks = {}
    model = UnifiedSpindleModel.load(args.model, runtime_dir=out)
    checks["cutting_source_bytes_identical"] = model.payload["cutting"] == Path(args.cutting).read_bytes()
    checks["base_source_bytes_identical"] = model.payload["base"] == Path(args.base).read_bytes()
    source = load_models(args.cutting, device="cpu")
    checks["cutting_versions_identical"] = source.current_version() == model.cutting.current_version()
    checks["normalization_identical"] = source.normalization == model.cutting.normalization
    checks["base_original_status_preserved"] = model.base.status == "calibrated_unvalidated"
    for name in ("main", "residual"):
        checks[name + "_tensors_identical"] = all(torch.equal(value, getattr(model.cutting, name).state_dict()[key])
            for key, value in getattr(source, name).state_dict().items())
    corrupt = deepcopy(model.payload)
    corrupt["base"] += b" "
    try:
        UnifiedSpindleModel(corrupt, runtime_dir=out)
        checks["tampered_component_rejected"] = False
    except ValueError:
        checks["tampered_component_rejected"] = True
    corrupt = deepcopy(model.payload)
    corrupt["manifest"]["research_only"] = False
    try:
        UnifiedSpindleModel(corrupt, runtime_dir=out)
        checks["tampered_manifest_rejected"] = False
    except ValueError:
        checks["tampered_manifest_rejected"] = True
    before = model.payload["cutting"]
    updated = model.with_base(args.base, runtime_dir=out)
    checks["base_update_isolates_cutting_bytes"] = updated.payload["cutting"] == before
    checks["base_update_keeps_source_immutable"] = model.payload["cutting"] == before
    checks["base_update_new_package_version"] = updated.version != model.version
    try:
        model.save(args.model)
        checks["loaded_source_overwrite_rejected"] = False
    except ValueError:
        checks["loaded_source_overwrite_rejected"] = True
    memory = model.manifest["memory_initial_snapshot"]
    memory["entries"]["test"] = "must_not_leak"
    checks["task_snapshot_isolated"] = model.manifest["memory_initial_snapshot"]["entries"] == {}
    context_path = out / "machining_context.json"
    save_json(context_path, CONTEXT_DEFAULT)
    collision_dir = out / "collision_check"
    collision_dir.mkdir(exist_ok=True)
    collision_context = collision_dir / "prediction.json"
    save_json(collision_context, CONTEXT_DEFAULT)
    collision_hash = file_hash(collision_context)
    try:
        predict_files(model, nc_path=args.nc, process_path=args.process, context_path=collision_context,
                      output_dir=collision_dir)
        checks["input_output_collision_rejected"] = False
    except ValueError:
        checks["input_output_collision_rejected"] = file_hash(collision_context) == collision_hash
    bad = deepcopy(CONTEXT_DEFAULT)
    bad["tool"]["diameter_mm"] = 12
    bad_path = out / "outside_scope.json"
    save_json(bad_path, bad)
    try:
        read_context(bad_path)
        checks["unsupported_combination_rejected"] = False
    except ValueError:
        checks["unsupported_combination_rejected"] = True
    sample, source_table = prepare_inputs(args.nc, args.process, CONTEXT_DEFAULT, model.cutting)
    selected, _, _ = prospective_residual_context(sample.frame, model.cutting.config.window_points)
    # Compare the entire real program, not an air-cut point or one short window.
    subset = selected
    a = predict_cutting_power(source, subset, context=sample.metadata)
    b = predict_cutting_power(model.cutting, subset, context=sample.metadata)
    equal = {key: bool(np.array_equal(value, b[key])) for key, value in a.items() if isinstance(value, np.ndarray)}
    checks["source_and_packaged_predictions_byte_identical"] = all(equal.values())
    original_base = BasePowerModel.load(args.base)
    original_base_array = np.interp(selected.spindle_rpm.to_numpy(float), original_base.speeds_rpm, original_base.powers_w)
    packaged_base_array = np.interp(selected.spindle_rpm.to_numpy(float), model.base.speeds_rpm, model.base.powers_w)
    checks["whole_program_base_prediction_identical"] = np.array_equal(original_base_array, packaged_base_array)
    original_event = predict_base_event(8000, original_base, sample_period_s=.01)
    packaged_event = predict_base_event(8000, model.base, sample_period_s=.01)
    event_arrays = {key: bool(np.array_equal(value, packaged_event[key])) for key, value in original_event.items()
                    if isinstance(value, np.ndarray)}
    checks["dynamic_base_event_arrays_identical"] = bool(event_arrays) and all(event_arrays.values())
    checks["whole_program_total_quantiles_identical"] = np.array_equal(
        a["cutting_quantiles_w"] + original_base_array[:, None], b["cutting_quantiles_w"] + packaged_base_array[:, None])
    receipt = predict_files(model, nc_path=args.nc, process_path=args.process,
        context_path=context_path, output_dir=out / "end_to_end")
    result = json.loads((out / "end_to_end" / "prediction.json").read_text(encoding="utf-8"))
    checks["prediction_json_exact_process_hash"] = result["metadata"]["process_file_sha256"] == file_hash(args.process)
    checks["prediction_rows_original_order"] = [r["source_index"] for r in result["rows"]] == list(range(len(result["rows"])))
    valid = [r for r in result["rows"] if r["status"] == "research_prediction"]
    checks["finite_ordered_total_bounds"] = bool(valid) and all(
        np.isfinite([r["total_lower_w"], r["total_power_w"], r["total_upper_w"]]).all()
        and r["total_lower_w"] <= r["total_power_w"] <= r["total_upper_w"] for r in valid)
    contact_missing = [r["source_index"] for r in result["rows"] if r["status"] != "research_prediction"
        and source_table.iloc[r["source_index"]].ap_mm > 0 and source_table.iloc[r["source_index"]].ae_mm > 0
        and source_table.iloc[r["source_index"]].source_feed_mm_min > 0]
    checks["all_contact_process_rows_predicted"] = not contact_missing
    native = pd.read_csv(out / "end_to_end" / "prediction.csv", float_precision="round_trip")
    representatives = [r for r in result["rows"] if r["mapping_method"] == "source_cell_nearest_supported_grid"]
    mapping_checks = []
    for row in representatives:
        point = source_table.iloc[row["source_index"]]
        native_row = native.loc[(native.line == row["line"]) & (native.execution_instance == row["execution_instance"])
            & (native.segment_mm == row["prediction_segment_mm"])]
        mapping_checks.append(len(native_row) == 1
            and row["source_cell_start_mm"] < row["prediction_segment_mm"] <= row["source_cell_end_mm"]
            and (native_row.iloc[0].ap_mm > 0 and native_row.iloc[0].ae_mm > 0) == (point.ap_mm > 0 and point.ae_mm > 0)
            and native_row.iloc[0].spindle_power_w == row["total_power_w"])
    checks["cell_representatives_inside_same_source_cell_and_contact"] = all(mapping_checks)
    report = dict(status="passed" if all(checks.values()) else "failed", checks=checks,
        prediction_comparison={"points": len(subset), "arrays": equal, "device": "cpu", "threads": 4},
        model_sha256=file_hash(args.model), cutting_sha256=file_hash(args.cutting), base_sha256=file_hash(args.base),
        model_version=model.version, end_to_end=receipt, numerical_training_updates=0,
        source_cell_representatives=len(representatives), missing_contact_rows=contact_missing,
        dynamic_base_event_comparison=event_arrays,
        independent_accuracy_validation=False, total_power_qualification=False)
    report["runtime_project_modules"] = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
        if (name == "project" or name.startswith("project.")) and getattr(module, "__file__", None)}
    report["runtime_dependencies"] = {name: importlib.metadata.version(name) for name in
        ("numpy", "pandas", "scipy", "torch", "threadpoolctl")}
    report["gui_modules_loaded"] = [name for name in sys.modules if name == "tkinter" or name.startswith(("PySide", "PyQt"))]
    save_json(out / "verification.json", report)
    print(json.dumps({"status": report["status"], "checks": checks, "source_rows": receipt["source_rows"],
        "source_predicted_rows": receipt["source_predicted_rows"]}, ensure_ascii=False))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
