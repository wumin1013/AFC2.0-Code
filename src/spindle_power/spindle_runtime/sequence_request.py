"""Run one entire process-point sequence against the shared spindle model runtime.

The caller supplies the geometry and point identities obtained from its three
business files. Window planning, seam repair, power calculation and startup
event curves are executed by this entry point for either tensor backend.
"""
from collections import defaultdict
from collections.abc import Mapping
import hashlib
import io
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import load_model
from .full_position import (
    COMMAND_DOMAIN, DIRECT_SOURCE_POSITION_POLICY, IN_COLS, OUT_COLS,
    SOURCE_IDENTITY_COLUMNS, combine_command_power, infer_grid_with,
    predict_prepared_positions, prepare_full_positions, restore_direct_source,
    uses_command_domain,
)
from .package import profile_bytes


REQUEST_SCHEMA = "afc.spindle_sequence_request.1"
RESULT_SCHEMA = "afc.spindle_sequence_results.1"
TRACE_SCHEMA = "afc.spindle_command_trace.1"
_INTEGER = ("source_index", "line", "point_index0", "execution_instance", "sequence_id", "motion")
_FLOAT = ("segment_mm", "path_mm", "query_path_mm", "command_rpm")
_SEQUENCE_METADATA = (
    "source_missing_reason", "geometry_audit", "spindle_start_commands",
    "diameter_mm", "teeth", "extrapolate_base",
)


def _model_identity(profile):
    if not uses_command_domain(profile):
        raise ValueError("Whole-sequence inference requires the direct command-domain spindle profile")
    if list(profile["input_columns"]) != IN_COLS or list(profile["output_columns"]) != OUT_COLS:
        raise ValueError("Whole-sequence feature or power columns differ from the spindle contract")
    digest = profile["tensor_model_sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or profile["artifact"]["sha256"] != digest:
        raise ValueError("Whole-sequence profile has an invalid model identity")
    return dict(model_sha256=digest, model_profile_sha256=hashlib.sha256(profile_bytes(profile)).hexdigest())


def prepare_sequence_request(context_path, nc_path, process_path, profile):
    """Read the three business files and return a numeric sequence plus JSON metadata."""
    identity = _model_identity(profile)
    context = json.loads(Path(context_path).read_text(encoding="utf-8-sig"))
    prepared = prepare_full_positions(
        nc_path, process_path, context, position_policy=DIRECT_SOURCE_POSITION_POLICY, command_domain=True,
    )
    source, grid = prepared["source"], prepared["grid"]
    raw = grid[IN_COLS].to_numpy(dtype=np.float64)
    known = np.isfinite(raw).all(axis=1)
    arrays = {
        "features": np.where(known[:, None], raw, 0.0),
        "feature_valid": known,
        **{key: source[key].to_numpy(dtype=np.int64) for key in ("source_index", "line", "point_index0")},
        "execution_instance": source["source_line_occurrence"].to_numpy(dtype=np.int64),
        **{key: grid[key].where(known, -1).to_numpy(dtype=np.int64)
           for key in ("sequence_id", "motion") if key in grid},
        "segment_mm": source["segment_mm"].to_numpy(dtype=np.float64),
        "path_mm": source["path_mm"].to_numpy(dtype=np.float64),
        "query_path_mm": (np.where(known, grid["path_mm"].to_numpy(dtype=np.float64), np.nan)
                          if "path_mm" in grid else np.full(len(source), np.nan)),
        "command_rpm": np.asarray(prepared["timing"]["command_at_source_rpm"], dtype=np.float64),
        "ap_ae_mm": source[["ap_mm", "ae_mm"]].to_numpy(dtype=np.float64),
    }
    for key in ("sequence_id", "motion"):
        arrays.setdefault(key, np.full(len(source), -1, dtype=np.int64))
    metadata = dict(
        schema=REQUEST_SCHEMA, **identity, source_count=len(source),
        source_missing_reason=source["missing_reason"].tolist(),
        geometry_audit=prepared["geometry_audit"],
        spindle_start_commands=prepared["spindle_start_commands"],
        diameter_mm=prepared["diameter_mm"], teeth=prepared["teeth"],
        extrapolate_base=bool(context.get("prediction_policy", {}).get("extrapolate_base", True)),
    )
    return arrays, metadata


def load_sequence_arrays(source):
    """Load an NPZ request from bytes or a file, without deserializing Python objects."""
    source = io.BytesIO(source) if isinstance(source, (bytes, bytearray, memoryview)) else source
    archive = np.load(source, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError("Whole-sequence input must be an NPZ archive")
    with archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("Whole-sequence NPZ repeats input fields")
        return {key: archive[key] for key in archive.files}


def _source_columns(arrays):
    return {
        **{key: arrays[key] for key in ("source_index", "line", "point_index0")},
        "source_line_occurrence": arrays["execution_instance"],
    }


def _validate_events(events, arrays):
    if not isinstance(events, list):
        raise ValueError("Whole-sequence spindle_start_commands must be a JSON list")
    keys = ("kind", "line", "target_rpm", "trigger", "nc_block",
            "display_before_source", "first_engaged_source")
    identities = set(zip(*(_source_columns(arrays)[key] for key in SOURCE_IDENTITY_COLUMNS)))
    for event in events:
        if (not isinstance(event, dict) or event.keys() != set(keys)
                or event["kind"] != "spindle_start"
                or type(event["line"]) is not int or event["line"] < 1
                or type(event["target_rpm"]) not in (int, float)
                or not np.isfinite(event["target_rpm"]) or event["target_rpm"] <= 0
                or event["trigger"] not in ("M03", "S")
                or not isinstance(event["nc_block"], str)):
            raise ValueError("Whole-sequence startup event is malformed")
        for key in ("display_before_source", "first_engaged_source"):
            anchor = event[key]
            if anchor is None:
                continue
            if (not isinstance(anchor, dict) or anchor.keys() != set(SOURCE_IDENTITY_COLUMNS)
                    or any(type(anchor[name]) is not int for name in SOURCE_IDENTITY_COLUMNS)
                    or tuple(anchor[name] for name in SOURCE_IDENTITY_COLUMNS) not in identities
                    or anchor["line"] < event["line"]):
                raise ValueError(f"Whole-sequence startup {key} does not identify a source point")


def _validate_sequence(arrays, metadata, profile):
    if not isinstance(metadata, dict) or metadata.get("schema") != REQUEST_SCHEMA:
        raise ValueError("Unsupported whole-sequence request schema")
    identity = _model_identity(profile)
    if any(metadata.get(name) != digest for name, digest in identity.items()):
        raise ValueError("Whole-sequence request selects a different model or profile")
    n = metadata.get("source_count")
    if type(n) is not int or n < 1:
        raise ValueError("Whole-sequence source_count must be a positive integer")
    if any(name not in metadata for name in _SEQUENCE_METADATA):
        raise ValueError("Whole-sequence request lacks required metadata")
    reasons = metadata["source_missing_reason"]
    if not isinstance(reasons, list) or len(reasons) != n or not all(isinstance(v, str) for v in reasons):
        raise ValueError("Whole-sequence missing reasons differ from the source point count")
    if not isinstance(metadata["geometry_audit"], list) or not all(isinstance(v, dict) for v in metadata["geometry_audit"]):
        raise ValueError("Whole-sequence geometry_audit must be a list of objects")
    for key in ("diameter_mm", "teeth"):
        value = metadata[key]
        if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"Whole-sequence {key} must be finite and positive")
    if type(metadata["extrapolate_base"]) is not bool:
        raise ValueError("Whole-sequence extrapolate_base must be Boolean")

    spec = {
        **{key: ((n,), np.int64) for key in _INTEGER},
        **{key: ((n,), np.float64) for key in _FLOAT},
        "features": ((n, len(IN_COLS)), np.float64), "feature_valid": ((n,), np.bool_),
        "ap_ae_mm": ((n, 2), np.float64),
    }
    if arrays.keys() != spec.keys():
        raise ValueError("Whole-sequence NPZ fields differ from the tensor contract")
    for key, (shape, dtype) in spec.items():
        value = arrays[key]
        if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != np.dtype(dtype):
            raise ValueError(f"Whole-sequence {key} must have shape {shape} and dtype {np.dtype(dtype)}")

    source_index, line = arrays["source_index"], arrays["line"]
    occurrence, index0 = arrays["execution_instance"], arrays["point_index0"]
    if ((source_index < 0).any() or len(np.unique(source_index)) != n or (line < 1).any()
            or (occurrence < 0).any() or (index0 < 0).any()):
        raise ValueError("Whole-sequence source point identities are invalid or repeated")
    same_visit = (line[1:] == line[:-1]) & (occurrence[1:] == occurrence[:-1])
    if index0[0] != 0 or not np.array_equal(index0[1:], np.where(same_visit, index0[:-1] + 1, 0)):
        raise ValueError("Whole-sequence points are not in their original visit order")
    spans = np.r_[True, ~same_visit]
    visits = list(zip(line[spans], occurrence[spans]))
    if len(visits) != len(set(visits)):
        raise ValueError("Whole-sequence source visit occurs in multiple disjoint spans")

    segment, path, query, speed = (arrays[key] for key in _FLOAT)
    if (not np.isfinite(path).all() or (np.diff(path) < 0).any()
            or np.isinf(segment).any() or np.isinf(query).any() or np.isinf(speed).any()
            or (segment < 0).any()):
        raise ValueError("Whole-sequence position or spindle-command values are invalid")
    raw, known, engagement = arrays["features"], arrays["feature_valid"], arrays["ap_ae_mm"]
    if (not np.isfinite(raw).all() or not np.isfinite(engagement).all()
            or np.any(raw[~known] != 0) or not np.array_equal(np.isfinite(query), known)
            or not np.isfinite(segment[known]).all() or (engagement[known] <= 0).any()
            or not np.array_equal(raw[known, :2], engagement[known])
            or not np.array_equal(raw[known, 3], speed[known]) or (speed[known] <= 0).any()
            or np.any(raw[known, 4] != metadata["diameter_mm"])
            or np.any(raw[known, 5] != metadata["teeth"])):
        raise ValueError("Whole-sequence features do not match the original source geometry or commands")
    sequence, motion = arrays["sequence_id"], arrays["motion"]
    if ((sequence[known] < 0).any() or not np.isin(motion[known], (1, 2, 3)).all()
            or np.any(sequence[~known] != -1) or np.any(motion[~known] != -1)):
        raise ValueError("Whole-sequence window boundary fields differ from the geometry mask")
    _validate_events(metadata["spindle_start_commands"], arrays)


def _restore_prepared(arrays, metadata):
    known = arrays["feature_valid"]
    identity = _source_columns(arrays)
    source = pd.DataFrame({
        **identity,
        "ap_mm": arrays["ap_ae_mm"][:, 0], "ae_mm": arrays["ap_ae_mm"][:, 1],
        "segment_mm": arrays["segment_mm"], "path_mm": arrays["path_mm"],
        "prediction_status": "missing_input", "missing_reason": metadata["source_missing_reason"],
    })
    grid = pd.DataFrame(np.where(known[:, None], arrays["features"], np.nan), columns=IN_COLS)
    for key in (*SOURCE_IDENTITY_COLUMNS, "ap_mm", "ae_mm", "segment_mm"):
        grid[key] = source[key]
    grid["path_mm"] = arrays["query_path_mm"]
    grid["execution_instance"] = arrays["execution_instance"]
    for key in ("sequence_id", "motion"):
        grid[key] = np.where(known, arrays[key], np.nan)
    grid["__source_query_index"] = source.index
    timing = dict(
        mode=COMMAND_DOMAIN, timing_is_measured=False, coordinate_mapping="physical_nc_line_and_process_point",
        command_at_source_rpm=[float(value) if np.isfinite(value) else None for value in arrays["command_rpm"]],
    )
    return dict(
        source=source, grid=grid, timing=timing, timeline=None, position_policy=DIRECT_SOURCE_POSITION_POLICY,
        geometry_audit=metadata["geometry_audit"], spindle_start_commands=metadata["spindle_start_commands"],
        diameter_mm=metadata["diameter_mm"], teeth=metadata["teeth"],
    )


def _open_onnx(model_path, device):
    if device != "cpu":
        raise ValueError("The sequence ONNX runtime supports CPU execution")
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    artifact = session.get_modelmeta().custom_metadata_map
    if artifact.get("afc.contract") != "afc.complete_onnx.1":
        raise ValueError("The selected file does not declare the complete AFC ONNX contract")
    profile = json.loads(artifact["afc.profile"])
    identity = _model_identity(profile)
    if (identity["model_sha256"] != artifact.get("afc.source_pt_sha256")
            or identity["model_profile_sha256"] != artifact.get("afc.profile_sha256")):
        raise ValueError("Embedded ONNX model or profile identity differs")
    return session, profile


def _predict_onnx(prepared, model_path, session, profile, *, extrapolate):
    from .onnx_runtime import _attach_onnx_startup_curves, _run_batch

    def cutting_batch(features, valid):
        return _run_batch(session, features, valid, extrapolate=extrapolate)

    grid, execution = infer_grid_with(prepared["grid"], profile, cutting_batch)
    result, coverage = restore_direct_source(prepared["source"], grid)
    result, receipt = combine_command_power(prepared, result, execution, coverage, profile, extrapolate=extrapolate)
    _attach_onnx_startup_curves(session, model_path, prepared, receipt, profile)
    return result, receipt


def _command_trace(result, power, valid, events):
    """Insert calibrated event samples before the original points they anchor to."""
    points = result[SOURCE_IDENTITY_COLUMNS].to_dict(orient="records")
    positions = {tuple(point[key] for key in SOURCE_IDENTITY_COLUMNS): i for i, point in enumerate(points)}
    startups = defaultdict(list)
    unplaced = []
    previous_position = -1
    for event_index, event in enumerate(events):
        anchor = event["display_before_source"]
        if event["status"] != "calibrated_startup_curve" or anchor is None:
            reason = event["status"] if event["status"] != "calibrated_startup_curve" else "no_source_anchor"
            unplaced.append(dict(event_index=event_index, line=event["line"], status=event["status"], reason=reason))
            continue
        position = positions[tuple(anchor[key] for key in SOURCE_IDENTITY_COLUMNS)]
        if position < previous_position:
            raise ValueError("Whole-sequence startup anchors are not in program order")
        previous_position = position
        startups[position].append((event_index, event))

    samples = []
    for position, (point, values, complete) in enumerate(zip(points, power.tolist(), valid.tolist(), strict=True)):
        for event_index, event in startups.get(position, ()):
            curve = event["curve"]
            for node, (elapsed, base) in enumerate(zip(curve["event_elapsed_s"], curve["base_power_w"], strict=True)):
                samples.append(dict(
                    sample_index=len(samples), kind="startup", line=event["line"],
                    source_index=None, source_line_occurrence=None, point_index0=None,
                    event_index=event_index, event_sample_index0=node, event_elapsed_s=elapsed, valid=True,
                    **dict(zip(OUT_COLS, (base, base, base, base, 0.0), strict=True)),
                ))
        samples.append(dict(
            sample_index=len(samples), kind="process_point", **point,
            event_index=None, event_sample_index0=None, event_elapsed_s=None, valid=complete,
            **{key: value if math.isfinite(value) else None for key, value in zip(OUT_COLS, values, strict=True)},
        ))
    return dict(schema=TRACE_SCHEMA, order="program_order", sample_count=len(samples),
                samples=samples, unplaced_startup_events=unplaced)


def predict_sequence_request(arrays, metadata, model_path, *, device="cpu"):
    """Compute all power columns and event curves in one model-side request.

    Return a float64 power matrix, a Boolean complete-row mask and a JSON-ready
    receipt in original source order. Individual columns can remain finite on
    blocked rows, for example when base power is known but cutting is unavailable.
    The receipt also includes the assembled command trace, starting each calibrated
    event before its source point without assigning a program-wide time axis.
    """
    if not isinstance(arrays, Mapping):
        arrays = load_sequence_arrays(arrays)
    model_path = Path(model_path)
    onnx = model_path.suffix.lower() == ".onnx"
    if onnx:
        model, profile = _open_onnx(model_path, device)
    else:
        model, profile = load_model(model_path, device=device)
    _validate_sequence(arrays, metadata, profile)
    prepared = _restore_prepared(arrays, metadata)
    extrapolate = metadata["extrapolate_base"]
    if onnx:
        result, receipt = _predict_onnx(prepared, model_path, model, profile, extrapolate=extrapolate)
    else:
        result, receipt = predict_prepared_positions(prepared, model, profile, extrapolate=extrapolate)
    power = result[OUT_COLS].to_numpy(dtype=np.float64)
    valid = np.isfinite(power).all(axis=1)
    receipt.update(
        prediction_schema=receipt["schema"], schema=RESULT_SCHEMA,
        **_model_identity(profile), model_version=profile["model_version"],
        model_backend="onnxruntime_cpu" if onnx else "pytorch_" + device,
        input_columns=list(IN_COLS), output_columns=list(OUT_COLS),
        source_index=result["source_index"].tolist(), row_status=result["prediction_status"].tolist(),
        row_missing_reason=result["missing_reason"].tolist(),
        row_original_missing_reason=result["original_missing_reason"].tolist(),
        row_mapping_method=result["mapping_method"].tolist(),
        row_base_is_extrapolation=result["base_is_extrapolation"].tolist(),
        command_trace=_command_trace(result, power, valid, receipt["spindle_startup_events"]),
    )
    return power, valid, receipt
