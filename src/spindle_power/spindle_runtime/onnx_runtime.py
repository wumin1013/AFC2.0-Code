"""ONNX execution of the same source-position request used by PyTorch."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .full_position import (
    DIRECT_SOURCE_POSITION_POLICY,
    IN_COLS,
    OUT_COLS,
    combine_command_power,
    combine_source_power,
    infer_grid_with,
    prepare_full_positions,
    restore_direct_source,
    uses_command_domain,
)
from .package import profile_bytes
from .startup_events import attach_startup_curves, isolated_startup_inputs


def _run_batch(session, features, valid, *, timeline=None, extrapolate):
    dynamic = timeline is not None
    if timeline is None:
        timeline = {
            "time_s": np.array([[0.0, 1.0, 2.0]], dtype=np.float64),
            "commands_rpm": np.zeros((1, 3), dtype=np.float64),
            "initial_speed_rpm": np.zeros(1, dtype=np.float64),
            "point_time_s": np.zeros(valid.shape, dtype=np.float64),
        }
    power, status = session.run(
        ["power", "status"],
        {
            "features": features,
            "valid": valid,
            "use_dynamic_base": np.array([dynamic], dtype=bool),
            "extrapolate": np.array([extrapolate], dtype=bool),
            **timeline,
        },
    )
    if status.shape != (len(valid),) or np.any(status != 0):
        raise ValueError(f"ONNX model rejected the prediction batch: {status.tolist()}")
    if (power.shape != (*valid.shape, len(OUT_COLS)) or power.dtype != np.float32
            or not np.isfinite(power[valid]).all() or np.any(power[~valid] != 0)):
        raise ValueError("ONNX power output differs from the tensor contract")
    points = power[valid]
    if ((points[:, 1] > points[:, 0]) | (points[:, 0] > points[:, 2])).any():
        raise ValueError("ONNX power output contains unordered bounds")
    return power


def _base_at_source(session, prepared, *, extrapolate):
    timeline = prepared["timeline"]
    point_times = timeline["point_time_s"]
    commands = np.asarray(prepared["timing"]["command_at_source_rpm"])
    count = len(point_times)
    base = np.empty(count, dtype=np.float64)
    shared = {key: timeline[key] for key in ("time_s", "commands_rpm", "initial_speed_rpm")}
    width = min(128, count)
    batches = 0
    for start in range(0, count, 128 * width):
        stop = min(count, start + 128 * width)
        length = stop - start
        shape = ((length + width - 1) // width, width)
        valid = np.arange(np.prod(shape)).reshape(shape) < length
        features = np.zeros((*shape, len(IN_COLS)), dtype=np.float32)
        features[..., 3][valid] = commands[start:stop]
        features[..., 4] = prepared["diameter_mm"]
        features[..., 5] = prepared["teeth"]
        times = np.full(shape, timeline["time_s"][0, 0], dtype=np.float64)
        times[valid] = point_times[start:stop]
        power = _run_batch(session, features, valid,
                           timeline=dict(shared, point_time_s=times), extrapolate=extrapolate)[valid]
        if np.any(power[:, 4] != 0) or not np.array_equal(power[:, 0], power[:, 3]):
            raise ValueError("ONNX base-only query returned cutting power")
        base[start:stop] = power[:, 3]
        batches += 1
    return base, dict(source_queries=count, batches=batches, tensor_only=True)


def _attach_onnx_startup_curves(session, model_path, prepared, metadata, profile):
    domain = profile.get("base_event_calibration", {}).get("startup")
    commands = prepared.get("spindle_start_commands", ())
    if not domain or not any(domain[0] <= row["target_rpm"] <= domain[-1] for row in commands):
        attach_startup_curves(prepared, metadata, profile, None, None, None)
        return

    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        attach_startup_curves(prepared, metadata, profile, None, None, None,
                              unavailable_reason="onnx_calibration_reader_unavailable")
        return

    names = ("dynamic.startup_levels_rpm", "dynamic.startup_phase_times_s")
    graph = onnx.load(str(model_path), load_external_data=False).graph
    initializers = {value.name: value for value in graph.initializer if value.name in names}
    if any(name not in initializers for name in names):
        raise ValueError("ONNX startup calibration initializers are missing")
    levels, phases = (numpy_helper.to_array(initializers[name]) for name in names)
    if (levels.ndim != 1 or levels.size < 2 or phases.shape != (levels.size, 5)
            or not np.isfinite(levels).all() or not np.isfinite(phases).all()
            or np.any(np.diff(levels) <= 0) or np.any(phases[:, 0] != 0)
            or np.any(np.diff(phases, axis=1) <= 0)):
        raise ValueError("Invalid ONNX startup calibration initializers")

    def evaluate(target_rpm, elapsed_s):
        valid = np.ones((len(elapsed_s), 1), dtype=bool)
        features = np.zeros((*valid.shape, len(IN_COLS)), dtype=np.float32)
        features[..., 3:6] = [target_rpm, prepared["diameter_mm"], prepared["teeth"]]
        values = _run_batch(session, features, valid,
                            timeline=isolated_startup_inputs(target_rpm, elapsed_s),
                            extrapolate=False)[:, 0]
        if np.any(values[:, 4] != 0) or not np.array_equal(values[:, 0], values[:, 3]):
            raise ValueError("ONNX startup query returned cutting power")
        return values[:, 3]

    attach_startup_curves(prepared, metadata, profile, levels, phases, evaluate)


def predict_onnx_positions(nc_path, process_path, context, model_path, *, device="cpu"):
    """Predict raw business inputs with a direct-position AFC ONNX artifact."""
    if device != "cpu":
        raise ValueError("The file-based ONNX runtime supports CPU execution")
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    session = ort.InferenceSession(str(model_path), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    artifact = session.get_modelmeta().custom_metadata_map
    if artifact.get("afc.contract") != "afc.complete_onnx.1":
        raise ValueError("The selected file does not declare the complete AFC ONNX contract")
    profile = json.loads(artifact["afc.profile"])
    if hashlib.sha256(profile_bytes(profile)).hexdigest() != artifact["afc.profile_sha256"]:
        raise ValueError("Embedded ONNX profile SHA differs")
    policy = profile["prediction_policy"]["position_policy"]
    if policy != DIRECT_SOURCE_POSITION_POLICY:
        raise ValueError("This ONNX model does not declare direct process-point prediction")
    if profile["input_columns"] != IN_COLS or profile["output_columns"] != OUT_COLS:
        raise ValueError("ONNX feature or power column order differs")
    extrapolate = bool(context.get("prediction_policy", {}).get(
        "extrapolate_base", artifact["afc.default_extrapolate"] == "true"))
    command_domain = uses_command_domain(profile)
    prepared = prepare_full_positions(nc_path, process_path, context, position_policy=policy,
                                      command_domain=command_domain)

    def cutting_batch(features, valid):
        return _run_batch(session, features, valid, extrapolate=extrapolate)

    grid, execution = infer_grid_with(prepared["grid"], profile, cutting_batch)
    result, coverage = restore_direct_source(prepared["source"], grid)
    if command_domain:
        result, metadata = combine_command_power(prepared, result, execution, coverage, profile,
                                                 extrapolate=extrapolate)
        _attach_onnx_startup_curves(session, model_path, prepared, metadata, profile)
    else:
        base, base_execution = _base_at_source(session, prepared, extrapolate=extrapolate)
        calibration = profile["base_event_calibration"]
        result, metadata = combine_source_power(
            prepared, result, base, execution, coverage, base_execution,
            curve_max=profile["base_domain"]["spindle_rpm_max"],
            domain_min=max(calibration["startup"][0], calibration["braking"][0]),
            event_domain_max=min(calibration["startup"][-1], calibration["braking"][-1]),
            extrapolate=extrapolate,
        )
    metadata.update(model_version=profile["model_version"], model_backend="onnxruntime_cpu",
                    source_pt_sha256=artifact["afc.source_pt_sha256"])
    return result, metadata
