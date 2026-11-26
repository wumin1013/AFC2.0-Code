"""Read-only raw-data pilot of the shared feed preparation chain.

The output is diagnostic evidence. It does not grant a training release or
replace unknown quality facts with True. Explicit metadata may be supplied in
the manifest after the corresponding evidence has been reviewed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from spindle_power.load_prediction.common import file_hash, save_json, version_of
from spindle_power.load_prediction.feed import FeedPlan, MachineConfig
from spindle_power.load_prediction.feed_samples import FeedSample, prepare_feed_features, prepare_feed_samples
from spindle_power.load_prediction.measurement import read_measurement, detect_encoding
from spindle_power.load_prediction.sample_rules import apply_process_contact, contact_mask, QUALITY_FIELDS, SAMPLE_RULES_VERSION
from spindle_power.load_prediction.samples import read_process_table
from spindle_power.load_prediction.training_checkpoint import source_fingerprints
from afc_common.process_path import PROCESS_MAPPING_RULES_VERSION


def preparation_source_fingerprints():
    fingerprints = source_fingerprints()
    modules = ("feed.py", "feed_samples.py", "measurement.py", "sample_rules.py", "samples.py")
    return {**{f"load_prediction/{name}": fingerprints[name] for name in modules},
            "process_path.py": fingerprints["shared/process_path.py"]}


def inspect_process(process, features):
    rows = []
    for (line, instance), grid in features.frame.groupby(["line", "execution_instance"], sort=False):
        source = process.loc[(process.line == line) & (process.execution_instance == instance)]
        length = float(grid.block_length_mm.iloc[0])
        rows.append(dict(line=int(line), execution_instance=int(instance), nc_length_mm=length,
                         process_points=len(source), unique_process_points=int(source.segment_mm.nunique()),
                         process_min_mm=float(source.segment_mm.min()) if len(source) else None,
                         process_max_mm=float(source.segment_mm.max()) if len(source) else None,
                         endpoint_difference_mm=float(source.segment_mm.max() - length) if len(source) else None,
                         has_conflicting_positions=bool(source.segment_mm.duplicated().any()),
                         strictly_increasing=bool(len(source) < 2 or np.all(np.diff(source.segment_mm.to_numpy(float)) > 0)),
                         endpoint_matches_nc=bool(len(source) and np.isclose(source.segment_mm.iloc[-1], length, rtol=0., atol=1e-8))))
    return rows


@lru_cache(maxsize=16)
def prepare_geometry(nc, process_path, machine_path, program_id, mode, settings, override_feed, line_offset):
    """Reuse immutable NC/process geometry inside a worker; labels are never cached."""
    features = prepare_feed_features(nc, MachineConfig.load(machine_path), program_id=program_id,
        machining_mode=mode, machine_settings_version=settings, feed_override=override_feed,
        coordinate_mode="workpiece_cutting")
    plan = FeedPlan(features.frame, features.metadata["program_version"], features.metadata["machine_version"], False,
                    {"source": "workpiece_cutting_geometry", "time_mapping": "unavailable"})
    process = read_process_table(process_path, plan, line_offset=line_offset)
    with open(process_path, encoding=detect_encoding(process_path)) as stream:
        next(stream)
        source_points = sum(bool(line.strip()) for line in stream)
    process_summary = inspect_process(process, features)
    duplicate_positions = []
    keys = ["line", "execution_instance", "segment_mm"]
    duplicates = process.loc[process.duplicated(keys, keep=False)]
    for key, group in duplicates.groupby(keys, sort=False):
        states = (group[["ap_mm", "ae_mm"]].to_numpy(float) > 0).all(axis=1)
        duplicate_positions.append(dict(line=int(key[0]), execution_instance=int(key[1]), segment_mm=float(key[2]),
            rows=len(group), contact_states_agree=bool(np.all(states == states[0])),
            geometry_values=group[["ap_mm", "ae_mm"]].to_numpy(float).tolist()))
    features.frame = apply_process_contact(features.frame, process)
    return features, process, process_summary, duplicate_positions, source_points


def prepare_one(original, *, alignment_path, output, override=None):
    meta = deepcopy(original)
    correction = deepcopy(override or {})
    if correction:
        meta.update(correction)
        meta["planned_combination"] = original["combination"]
        meta["planned_program_slot"] = original.get("program_slot", original["combination"][2:])
        meta["candidate_pairing_status"] = "proposed_not_formally_released"
    alignment = json.loads(Path(alignment_path).read_text(encoding="utf-8"))
    if alignment.get("measurement_sha256") != file_hash(meta["measurement"]):
        raise ValueError("alignment evidence does not match the measurement SHA256")
    if alignment.get("nc_sha256") != file_hash(meta["nc"]):
        raise ValueError("alignment evidence does not match the NC SHA256")
    selected = alignment["selected_candidate"]
    raw_nc = Path(meta["nc"]).read_text(encoding=detect_encoding(meta["nc"]))
    raw_settings = [line.strip() for line in raw_nc.splitlines() if "G05" in line.upper()]
    settings = meta.get("machine_settings_version") or "diagnostic_" + version_of(dict(
        controller_header=meta.get("raw_controller_header"), commands=raw_settings))
    mode = meta.get("machining_mode") or "diagnostic_recorded_settings_semantics_unverified"
    override_feed = meta.get("feed_override")
    if override_feed is None:
        override_feed = 1.
        meta["feed_override_evidence"] = "author_confirmed_cutting_override_100_percent; rapid excluded"
    cached, process, process_summary, duplicate_positions, source_points = prepare_geometry(meta["nc"], meta["process"], meta["machine"],
        meta["program_id"], mode, settings, override_feed, int(meta.get("process_line_offset", 0)))
    features = FeedSample(cached.frame, deepcopy(cached.metadata))
    derived = features.metadata.copy()
    for key in ("program_version", "machine_version", "machine_settings_version", "machining_mode", "feed_override", "coordinate_mode"):
        meta[key] = derived[key]
    meta.update(position_offset_mm=selected["position_offset_mm"],
                measurement_line_offset=int(selected["line_offset"]),
                alignment_evidence_path=str(alignment_path), alignment_evidence_sha256=file_hash(alignment_path))
    features.metadata = {**meta, **derived}
    measurement = read_measurement(meta["measurement"], sample_period_s=meta.get("sample_period_s"))
    raw_lines = measurement.values["program_line_raw"]
    mapping = {int(v): int(v) + int(selected["line_offset"]) for v in np.unique(raw_lines) if np.isfinite(v)}
    sample = prepare_feed_samples(features, measurement, meta, line_map=mapping, allow_position_derivative=False)
    frame = sample.frame
    # Diagnostic exports remain unusable as a formal-training release, even if
    # a later input manifest contains reviewed quality facts.
    qualified_points = int(frame.valid_label.sum())
    frame["quality_qualified_label"] = frame.valid_label
    frame["valid_label"] = False
    sample.metadata.update(formal_training_release=False, preparation_scope="diagnostic_only")
    numeric = np.isfinite(frame.feed_actual_mm_min.to_numpy(float))
    contact = contact_mask(frame)
    known_contact = frame.cutting_contact.notna().to_numpy()
    candidate = numeric & contact
    gaps = []
    for (line, instance), rows in frame.groupby(["line", "execution_instance"], sort=False):
        indices = rows.index.to_numpy()
        counts = dict(points=len(rows), known_process_points=int(known_contact[indices].sum()),
                      cutting_contact_points=int(contact[indices].sum()), aligned_feed_points=int(numeric[indices].sum()),
                      candidate_aligned_contact_points=int(candidate[indices].sum()), valid_label_points=int(rows.valid_label.sum()))
        if counts["known_process_points"] < len(rows) or counts["candidate_aligned_contact_points"] < counts["cutting_contact_points"]:
            gaps.append(dict(line=int(line), execution_instance=int(instance), **counts))
    directory = Path(output) / "records" / meta["record_id"]
    directory.mkdir(parents=True, exist_ok=False)
    frame.to_csv(directory / "candidate.csv", index=False, encoding="utf-8")
    process.to_csv(directory / "process.csv", index=False, encoding="utf-8")
    save_json(directory / "metadata.json", sample.metadata)
    report = dict(record_id=meta["record_id"], status="diagnostic_prepared", formal_training_release=False,
        process_mapping_rules_version=PROCESS_MAPPING_RULES_VERSION,
        source_process_points=source_points, canonical_process_rows=len(process),
        process_source_indices_preserved=bool(source_points == len(process) and "source_file_line" in process and process.source_file_line.is_unique),
        mapped_process_points=int(process.execution_instance.ge(0).sum()),
        unmapped_process_points=int(process.execution_instance.lt(0).sum()),
        process_point_order_unique=bool("point_index_in_line" in process and not process.loc[process.execution_instance.ge(0)].duplicated(
            ["line", "execution_instance", "point_index_in_line"]).any()),
        mapped_positive_segments_strictly_increasing=all(r["strictly_increasing"] for r in process_summary if r["nc_length_mm"] > 0),
        mapped_segment_endpoints_match_nc=all(r["endpoint_matches_nc"] for r in process_summary if r["process_points"] and r["nc_length_mm"] > 0),
        pairing_changes=correction or None, planned_combination=original["combination"],
        source_files={key: dict(path=meta[key], sha256=file_hash(meta[key])) for key in ("nc", "process", "measurement", "machine")},
        alignment_evidence=dict(path=str(alignment_path), sha256=file_hash(alignment_path), selected_candidate=selected),
        known_facts={key: sample.metadata.get(key) for key in QUALITY_FIELDS},
        unknown_quality_fields=[key for key in QUALITY_FIELDS if sample.metadata.get(key) is None],
        quality_pass=sample.metadata["quality_pass"], position_source=sample.metadata["position_source"],
        measured_feed_source=sample.metadata["measured_feed_source"], points=len(frame),
        known_process_points=int(known_contact.sum()), cutting_contact_points=int(contact.sum()),
        unknown_process_points=int((~known_contact).sum()), known_noncontact_points=int((known_contact & ~contact).sum()),
        aligned_feed_points=int(numeric.sum()), candidate_aligned_contact_points=int(candidate.sum()),
        candidate_contact_coverage=float(candidate.sum()/contact.sum()) if contact.any() else None,
        valid_label_points=int(frame.valid_label.sum()),
        qualified_by_current_metadata_points=qualified_points,
        max_accepted_alignment_error_mm=float(frame.alignment_error_mm[numeric].max()) if numeric.any() else None,
        process_segments=process_summary, duplicate_process_positions=duplicate_positions,
        process_geometry_conflict_positions=len(duplicate_positions),
        process_contact_conflict_positions=sum(not r["contact_states_agree"] for r in duplicate_positions),
        gaps=gaps, issues=sample.metadata["issues"],
        output_files={p.name:file_hash(p) for p in directory.glob("*") if p.is_file()},
        limitations=["Candidate aligned contact coverage is a diagnostic, not a quality release.",
                     "Unconfirmed quality fields remain unknown and cannot produce valid training labels.",
                     "Only actual feed/actual axis velocity channels provide labels; command XYZ may locate early records."])
    save_json(directory / "report.json", report)
    return report


def run_one(task):
    original, alignment_path, output, override = task
    try:
        return prepare_one(original, alignment_path=alignment_path, output=output, override=override)
    except Exception as exc:
        result = dict(record_id=original["record_id"], status="failed", formal_training_release=False,
                      error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        save_json(Path(output) / "failures" / (original["record_id"] + ".json"), result)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--alignment-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record-ids", nargs="+", default=["R01_外形", "R01_曲面", "R01_混合", "R01_斜面", "R01_台阶", "R74_台阶"])
    parser.add_argument("--pairing-overrides", type=Path)
    parser.add_argument("--all", action="store_true", help="Prepare every active development record, without qualifying it")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        raise ValueError("workers must be between 1 and 8")
    if args.output.exists() and not args.resume:
        raise FileExistsError("Use a new output directory to preserve earlier preparation evidence")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    records = {r["record_id"]:r for r in json.loads(args.manifest.read_text(encoding="utf-8"))["records"]}
    overrides = json.loads(args.pairing_overrides.read_text(encoding="utf-8")) if args.pairing_overrides else {}
    record_ids = ([key for key, value in records.items() if value.get("dataset_role") == "development" and value.get("status") == "active"]
                  if args.all else args.record_ids)
    identity = dict(manifest_sha256=file_hash(args.manifest), script_sha256=file_hash(__file__),
        record_ids=record_ids, pairing_overrides=overrides, sample_rules_version=SAMPLE_RULES_VERSION,
        process_mapping_rules_version=PROCESS_MAPPING_RULES_VERSION,
        code_fingerprints=preparation_source_fingerprints())
    plan_path = args.output / "preparation_plan.json"
    if args.resume:
        if not plan_path.is_file() or json.loads(plan_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Preparation resume identity changed; use a new output directory")
    else:
        save_json(plan_path, identity)
    results, tasks = [], []
    for record_id in record_ids:
        override = dict(overrides.get(record_id, {}))
        alignment_path = Path(override.pop("alignment_evidence", args.alignment_dir / "records" / (record_id + ".json")))
        finished = args.output / "records" / record_id / "report.json"
        if args.resume and finished.is_file():
            previous = json.loads(finished.read_text(encoding="utf-8"))
            if (previous["alignment_evidence"]["sha256"] != file_hash(alignment_path) or any(
                    file_hash(finished.parent / name) != digest for name, digest in previous["output_files"].items())):
                raise ValueError(f"Saved preparation evidence was modified: {record_id}")
            results.append(previous)
            continue
        tasks.append((records[record_id], alignment_path, args.output, override))
    def accept(result):
        results.append(result)
        event = {k:result.get(k) for k in ("record_id", "status", "points", "candidate_contact_coverage", "valid_label_points", "error")}
        print(json.dumps(dict(completed=len(results), total=len(record_ids), **event), ensure_ascii=False), flush=True)
    if args.workers == 1:
        for task in tasks:
            accept(run_one(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(run_one, task) for task in tasks]):
                accept(future.result())
    current_sources = preparation_source_fingerprints()
    source_changed = [name for name, digest in identity["code_fingerprints"].items()
                      if current_sources[name] != digest]
    save_json(args.output / "pilot_report.json", dict(records=sorted(results,key=lambda r:record_ids.index(r["record_id"])),
        formal_training_release=False, identity=identity, source_changed_during_preparation=source_changed))


if __name__ == "__main__":
    main()
