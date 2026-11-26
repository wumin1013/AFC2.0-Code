"""Silent AFO interval export. This module never loads or calls a model."""
from __future__ import annotations

import argparse
import hashlib
from inspect import signature
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .interval_process_io import read_interval_process
from .release_metadata import RELEASE_VERSION
from .segmentation import SegmentationPipeline

RG_SCHEMA = "AFC_RG_PROCESS_V1"
PREDICTION_SCHEMA = "afc.spindle_prediction.1"
OUTPUT_NAMES = ("processinfo.csv", "sampledata.rg")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".interval-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _int_value(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be an integer")
    if not math.isfinite(value) or value != int(value):
        raise ValueError(f"{name} must be a finite integer")
    return int(value)


def _bind_predictions(process, path, process_hash):
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if payload.get("schema") != PREDICTION_SCHEMA:
        raise ValueError(f"prediction.json schema must be {PREDICTION_SCHEMA}")
    metadata = payload.get("metadata", {})
    if str(metadata.get("process_file_sha256", "")).lower() != process_hash.lower():
        raise ValueError("Prediction does not belong to this process file (SHA256 mismatch)")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != len(process):
        raise ValueError("Prediction must contain exactly one record for every source process row")
    indexed = {}
    for row in rows:
        index = _int_value(row.get("source_index"), "source_index")
        if index in indexed or not 0 <= index < len(process):
            raise ValueError("Prediction source_index is duplicated or outside the process table")
        if _int_value(row.get("line"), "line") != int(process.iloc[index].line_no_raw) + 1:
            raise ValueError(f"Prediction physical line mismatch at source_index={index}")
        indexed[index] = row
    frame = pd.DataFrame([indexed[index] for index in range(len(process))])
    # The supported initial file contract contains one occurrence of each NC
    # physical line. Repeated execution cannot be encoded unambiguously in RG v1.
    lines = process.line_no_raw.to_numpy(dtype=int)
    group_starts = np.r_[True, lines[1:] != lines[:-1]]
    if len(np.unique(lines[group_starts])) != int(group_starts.sum()):
        raise ValueError("Repeated NC execution requires an execution-aware AFO mapping; RG v1 refuses it")
    if np.any(np.diff(lines) < 0):
        raise ValueError("RG process coordinates require ordered physical NC lines")
    if "status" not in frame:
        raise ValueError("Prediction rows require an explicit status")
    available = frame.status.isin(("research_prediction", "qualified_prediction"))
    power_columns = ("total_power_w", "total_lower_w", "total_upper_w", "base_power_w")
    for column in power_columns:
        if column not in frame:
            raise ValueError(f"Missing prediction column: {column}")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    values = frame[list(power_columns)].to_numpy(dtype=float)
    available &= np.isfinite(values).all(axis=1)
    cutting = (process.ap > 0) & (process.ae > 0)
    unavailable = np.flatnonzero((~available & cutting).to_numpy()).tolist()
    if unavailable:
        raise ValueError(f"Unavailable predictions at {len(unavailable)} process rows; first indices: {unavailable[:10]}")
    if np.any(values[available, 1] > values[available, 2]) or np.any(values[available, 0] < 0):
        raise ValueError("Prediction power or interval bounds are invalid")
    frame.loc[~available, list(power_columns)] = np.nan
    frame["power_available"] = available
    return frame, metadata


def generate_intervals(process_path, prediction_path, output_dir, *, optimization_ratio=2.0):
    """Validate file identity, apply existing six states, then publish both files.

    interval_status.json is the completion marker. Readers must check it and
    both output hashes; replacing two independent files is not one transaction.
    """
    process_path, prediction_path = Path(process_path).resolve(), Path(prediction_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    protected = {process_path, prediction_path}
    for name in (*OUTPUT_NAMES, "interval_status.json", ".interval_tool.lock"):
        if destination / name in protected:
            raise ValueError("Output must not overwrite an input file")
    ratio = float(optimization_ratio)
    if not math.isfinite(ratio) or not 0.1 <= ratio <= 5.0:
        raise ValueError("optimization_ratio must be between 0.1 and 5.0")
    lock_path = destination / ".interval_tool.lock"
    lock = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(lock)
    started = datetime.now(timezone.utc).isoformat()
    status_path = destination / "interval_status.json"
    common = {"schema": "afc.interval_tool.1", "release": RELEASE_VERSION, "started_at": started}
    temporary_paths = []
    try:
        _write_json(status_path, dict(common, status="running"))
        hashes = {"process_file_sha256": file_sha256(process_path),
                  "prediction_file_sha256": file_sha256(prediction_path)}
        process = read_interval_process(process_path)
        if process.empty:
            raise ValueError("Process table contains no points")
        prediction, metadata = _bind_predictions(process, prediction_path, hashes["process_file_sha256"])
        segmentation_input = process.copy()
        distance_semantics = metadata.get("process_distance_semantics")
        if distance_semantics == "cumulative_mm":
            distances = process.s.to_numpy(dtype=float)
            if np.any(np.diff(distances) < 0) or distances[-1] <= 0:
                raise ValueError("Declared cumulative process distance must be nondecreasing with positive span")
            segmentation_input["input_path_cumulative"] = distances
        elif distance_semantics != "incremental_mm":
            raise ValueError("Prediction metadata must declare process_distance_semantics as cumulative_mm or incremental_mm")
        result = SegmentationPipeline().run(segmentation_input)
        if not result.diagnostics.get("postprocess_validation_passed", False):
            raise ValueError("Six-state segmentation failed structural validation")
        if not result.diagnostics.get("path", {}).get("is_physical", False):
            raise ValueError("No physical process distance is available; refusing synthetic interval coordinates")
        labels = result.point_labels.set_index("source_index").loc[process.source_index]
        if len(labels) != len(process):
            raise ValueError("Segmentation coverage does not match process points")
        export = pd.DataFrame({
            "N": process.N_str, "S(r/min)": process.S, "ap(mm)": process.ap,
            "ae(mm)": process.ae, "F(mm/min)": process.F_program, "s(mm)": process.s,
            "MRR(mm3/s)": process.ap * process.ae * process.F_program / 60.0,
            "G": process.gcode_content, "state_code": labels.state_code.to_numpy(),
            "source_index": process.source_index, "line_index0": process.line_no_raw,
            "point_index0": labels.line_point_index.to_numpy(),
            "interval_id": labels.interval_id.to_numpy(), "segment_type": labels.segment_type.to_numpy(),
            "predicted_spindle_power_w": prediction.total_power_w,
            "lower_w": prediction.total_lower_w, "upper_w": prediction.total_upper_w,
            "prediction_status": prediction.status,
        })
        steady = export.state_code.eq(2)
        ideal = float(export.loc[steady, "predicted_spindle_power_w"].mean() * ratio) if steady.any() else None
        program = str(metadata.get("program_name", "")).strip()
        if not program or any(char in program for char in ";,\r\n"):
            raise ValueError("Prediction metadata needs a program_name without RG delimiters")
        intervals = []
        unavailable_intervals = 0
        for record in result.intervals.to_dict("records"):
            start, end = int(record["start_idx"]), int(record["end_idx"])
            indices = export.index[start:end + 1]
            complete_power = bool(prediction.loc[indices, "power_available"].all())
            mean = float(export.loc[indices, "predicted_spindle_power_w"].mean()) if complete_power else math.nan
            export.loc[indices, "interval_mean_power_w"] = mean
            # The existing ideal rule is one target per program/tool, computed
            # from all steady points. It is not silently changed to per-interval targets.
            export.loc[indices, "ideal_power_w"] = ideal if ideal is not None else np.nan
            if ideal is not None and record["segment_type"] in ("idle", "steady"):
                if not complete_power:
                    unavailable_intervals += 1
                    continue
                first, last = export.iloc[start], export.iloc[end]
                intervals.append({"count": end - start + 1, "start": start, "end": end,
                    "text": f"{int(first.line_index0)}.{int(first.point_index0)}-"
                            f"{int(last.line_index0)}.{int(last.point_index0)}:{mean:.6f}"})
        selected = sorted(sorted(intervals, key=lambda row: (-row["count"], row["start"]))[:200],
                          key=lambda row: row["start"])
        export["optimization_selected"] = False
        for entry in selected:
            export.loc[entry["start"]:entry["end"], "optimization_selected"] = True
        rg = RG_SCHEMA + "\n"
        if selected:
            rg += f"{program};{ideal:.6f};" + ",".join(item["text"] for item in selected) + ";\n"
        if file_sha256(process_path) != hashes["process_file_sha256"] or file_sha256(prediction_path) != hashes["prediction_file_sha256"]:
            raise ValueError("Input file changed during processing; outputs were not published")
        for name in OUTPUT_NAMES:
            handle, temporary = tempfile.mkstemp(dir=destination, prefix=".interval-", suffix=".tmp")
            os.close(handle)
            temporary_paths.append(Path(temporary))
        csv_parameters = signature(export.to_csv).parameters
        line_ending_option = "lineterminator" if "lineterminator" in csv_parameters else "line_terminator"
        export.to_csv(
            temporary_paths[0], index=False, encoding="utf-8", float_format="%.9g",
            **{line_ending_option: "\n"},
        )
        temporary_paths[1].write_text(rg, encoding="utf-8", newline="\n")
        outputs = {}
        for name, path in zip(OUTPUT_NAMES, temporary_paths):
            outputs[name] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
            path.replace(destination / name)
        status = dict(common, status="completed", completed_at=datetime.now(timezone.utc).isoformat(),
            **hashes, outputs=outputs, process_rows=len(export), interval_count=len(result.intervals),
            optimization_interval_count=len(selected), optimization_ratio=ratio, ideal_power_w=ideal,
            unavailable_power_rows=int((~prediction.power_available).sum()),
            skipped_optimization_intervals_missing_power=unavailable_intervals,
            ideal_rule="mean_of_all_steady_predicted_process_points_times_ratio",
            rg_schema=RG_SCHEMA, rg_coordinate_domain="process_points_v1",
            process_distance_semantics=distance_semantics,
            rg_coordinates="NC physical line minus one, within-line process row ordinal starting at zero",
            afo_integration_status="pending_consumer_validation",
            power_source="model_prediction", model_version=metadata.get("model_version"),
            total_power_status=metadata.get("total_power_status"), interval_status=metadata.get("interval_status"),
            interval_kind=metadata.get("interval_kind"), segmentation_diagnostics=result.diagnostics)
        _write_json(status_path, status)
        return status
    except Exception as exc:
        _write_json(status_path, dict(common, status="failed", error_type=type(exc).__name__, error=str(exc)))
        raise
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        lock_path.unlink(missing_ok=True)


def application_directory():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


class SilentArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def _raw_path_option(arguments, option):
    paths = _raw_path_options(arguments, option)
    return paths[-1] if paths else None


def _raw_path_options(arguments, option):
    paths = []
    for index, token in enumerate(arguments):
        if token.startswith(option + "="):
            paths.append(Path(token.split("=", 1)[1]))
        if token == option and index + 1 < len(arguments) and not arguments[index + 1].startswith("--"):
            paths.append(Path(arguments[index + 1]))
    return paths


def main(argv=None):
    parser = SilentArgumentParser(description="区间划分工具：读取工艺信息与预测功率，静默导出两类结果", allow_abbrev=False)
    parser.add_argument("--process", type=Path, help="原始工艺信息文件")
    parser.add_argument("--prediction", type=Path, help="AFO/机枢生成的 prediction.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--optimization-ratio", type=float, default=2.0)
    arguments = list(sys.argv[1:] if argv is None else argv)
    app_dir = application_directory()
    destination = (_raw_path_option(arguments, "--output-dir") or app_dir).resolve()
    source = _raw_path_option(arguments, "--process")
    prediction_path = _raw_path_option(arguments, "--prediction") or app_dir / "prediction.json"
    try:
        args = parser.parse_args(arguments)
        destination = (args.output_dir or app_dir).resolve()
        source = args.process
        prediction_path = args.prediction or app_dir / "prediction.json"
        if source is None:
            candidates = [p for p in app_dir.iterdir() if p.is_file()
                          and p.suffix.lower() in (".csv", ".txt")
                          and p.name.lower() not in ("processinfo.csv", "sha256sums.txt")]
            if len(candidates) != 1:
                raise ValueError("Specify --process when exactly one original process file cannot be identified")
            source = candidates[0]
        generate_intervals(source, prediction_path, destination,
                           optimization_ratio=args.optimization_ratio)
        return 0
    except Exception as exc:
        # No dialog, console dependency, or unhandled windowed traceback.
        inputs = {Path(path).resolve() for path in (
            [p for p in (source, prediction_path) if p is not None]
            + _raw_path_options(arguments, "--process") + _raw_path_options(arguments, "--prediction"))}
        if (not (destination / ".interval_tool.lock").exists()
                and (destination / "interval_status.json").resolve() not in inputs):
            try:
                destination.mkdir(parents=True, exist_ok=True)
                _write_json(destination / "interval_status.json", {
                    "schema": "afc.interval_tool.1", "release": RELEASE_VERSION,
                    "status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
            except OSError:
                return 3
        return 2 if isinstance(exc, (ValueError, FileNotFoundError, FileExistsError)) else 3


if __name__ == "__main__":
    raise SystemExit(main())
