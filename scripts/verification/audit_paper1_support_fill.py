"""Audit a fill-only derivative of all 450 frozen development records.

This entry point never prepares data, instantiates a training service, loads a
model, or trains. It reads one old/new CSV pair at a time and checks the original
process source. Its success is a DATA gate, not a training release or a claim
that fixed-weight predictions have already been checked.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
KEYS = ("line", "execution_instance", "segment_mm", "path_mm")
MUTABLE = {"ap_mm", "ae_mm", "process_geometry_known", "cutting_contact"}
EXTRA_COLUMNS = {
    "reference_power_cut_w", "reference_aligned", "support_fill_restored",
    "support_fill_source_point_index", "support_fill_source_file_line",
    "support_fill_interval_start_mm", "support_fill_interval_end_mm",
}
IDENTITY_FIELDS = (
    "record_id", "round_id", "feature", "combination", "parameter_group", "program_slot",
    "dataset_role", "measurement_content_sha256", "file_sha256",
    "program_version", "machine_version", "diameter_mm", "teeth",
    "feed_override", "spacing_mm", "process_line_offset", "measurement_line_offset",
    "position_offset_mm", "sample_period_s", "steady_lines",
    "complete", "round_complete", "alignment_reliable", "no_alarm", "tool_healthy",
    "tool_session", "actual_event_order", "position_source", "feed_label_source",
    "base_version", "base_status", "formal_metrological_qualification",
)
LABEL_FIELDS = (
    "route", "label_source", "reference_kind", "evidence_grade",
    "measurement_sha256", "idle_power_w", "reference_application",
    "representativeness_assumed", "formal_metrological_qualification",
    "valid_sample_ranges", "raw_sample_count",
)


class AuditError(ValueError):
    """A fill-only contract failed; the candidate must not be released."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def json_digest(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def read_snapshot(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    contents = {key: item for key, item in value.items() if key != "snapshot_id"}
    require(value.get("snapshot_id") == json_digest(contents), "Snapshot identity mismatch")
    require(value.get("synthetic") is False, "Formal audit requires a real snapshot")
    return value


def development_records(snapshot):
    """Reject missing/duplicated slots and event data before any CSV is read."""
    from spindle_power.load_prediction.experiment_protocol import PROGRAM_FEATURES, validate_round_metadata

    entries = [item for item in snapshot["records"] if item["kind"] == "load"]
    metadata = [item["metadata"] for item in entries]
    require(len(entries) == 450, "Exactly 450 load records are required")
    require(validate_round_metadata(metadata)["valid"], "Development round metadata is invalid")
    expected = {(number, feature) for number in range(1, 91) for feature in PROGRAM_FEATURES}
    slots = [(row.get("round_id"), row.get("feature")) for row in metadata]
    require(len(set(slots)) == 450 and set(slots) == expected, "450 unique development slots required")
    require(all(row.get("dataset_role") == "development" and row.get("group_id") is None
                for row in metadata), "Event/test records are forbidden in this audit")
    ids = [row.get("record_id") for row in metadata]
    hashes = [row.get("measurement_content_sha256") for row in metadata]
    require(all(isinstance(value, str) and value for value in ids) and len(set(ids)) == 450,
            "Record IDs must be nonempty and unique")
    require(all(isinstance(value, str) and value for value in hashes) and len(set(hashes)) == 450,
            "Measured content identities must be nonempty and unique")
    return {item["metadata"]["record_id"]: item for item in entries}


def check_metadata(old, new):
    for key in IDENTITY_FIELDS:
        require(old.get(key) == new.get(key), f"Immutable metadata changed: {key}")
    for name in ("hybrid_label", "record_idle_label"):
        require((name in old) == (name in new), f"Label reference route changed: {name}")
        for key in LABEL_FIELDS:
            require(old.get(name, {}).get(key) == new.get(name, {}).get(key),
                    f"Frozen label evidence changed: {name}.{key}")


def checked_frame(snapshot_path, entry):
    directory = Path(snapshot_path).resolve().parent
    path = (directory / entry["file"]).resolve()
    require(path.is_relative_to(directory), "Snapshot CSV escapes its directory")
    require(file_hash(path) == entry["sha256"], "Snapshot CSV hash mismatch")
    frame = pd.read_csv(path, float_precision="round_trip")
    require(file_hash(path) == entry["sha256"], "Snapshot CSV changed while reading")
    return frame


def boolean(frame, column):
    values = frame[column]
    require(not values.isna().any() and values.isin([True, False]).all(),
            f"Invalid boolean column: {column}")
    return values.to_numpy(dtype=bool)


def equal_column(old, new, column, mask=None):
    left = old[column] if mask is None else old.loc[mask, column]
    right = new[column] if mask is None else new.loc[mask, column]
    try:
        pd.testing.assert_series_equal(left, right, check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise AuditError(f"Protected column changed: {column}") from exc


def source_support(frame, process, diameter):
    """Recompute support from declared source cells, never candidate flags."""
    require(np.isfinite(diameter) and diameter > 0, "Invalid tool diameter")
    count = len(frame)
    supported = np.zeros(count, dtype=bool)
    geometry = np.full((count, 2), np.nan)
    reasons = np.full(count, "no_process_row", dtype=object)
    owner_start, owner_end = np.full(count, np.nan), np.full(count, np.nan)
    source_points, source_lines = np.full(count, np.nan), np.full(count, np.nan)
    for (line, instance), target in frame.groupby(["line", "execution_instance"], sort=False):
        indices = target.index.to_numpy()
        source = process.loc[(process.line == line) & (process.execution_instance == instance)]
        if source.empty:
            continue
        if "segment_start_mm" not in source:
            reasons[indices] = "no_declared_source_interval"
            continue
        source = source.sort_values("segment_mm", kind="stable")
        starts = source.segment_start_mm.to_numpy(float)
        ends = source.segment_mm.to_numpy(float)
        require(np.isfinite(starts).all() and np.isfinite(ends).all()
                and np.all(starts >= 0) and np.all(ends > starts)
                and np.all(starts[1:] >= ends[:-1]),
                f"Malformed/overlapping source intervals: line={line}, instance={instance}")
        positions = target.segment_mm.to_numpy(float)
        owner = np.clip(np.searchsorted(ends, positions, side="left"), 0, len(ends) - 1)
        inside = np.isfinite(positions) & (positions > starts[owner]) & (positions <= ends[owner])
        origin = (starts[0] == 0.) & (positions == 0.)
        inside |= origin
        owner[origin] = 0
        values = source[["ap_mm", "ae_mm"]].to_numpy(float)
        valid = np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1) & (values[:, 1] <= diameter)
        supported[indices] = inside & valid[owner]
        geometry[indices] = values[owner]
        owner_start[indices], owner_end[indices] = starts[owner], ends[owner]
        for column, output in (("source_point_index", source_points), ("source_file_line", source_lines)):
            if column in source:
                output[indices] = source[column].to_numpy(float)[owner]
        reasons[indices] = np.where(~inside, "outside_or_missing_source_interval",
                                    np.where(valid[owner], "source_supported", "invalid_source_geometry"))
    conflict = boolean(frame, "process_geometry_conflict_support")
    supported[conflict] = False
    reasons[conflict] = "preserved_source_conflict"
    return supported, geometry, reasons, owner_start, owner_end, source_points, source_lines


def audit_frames(old, new, process, metadata):
    required = set(KEYS) | MUTABLE | {"process_geometry_conflict_support", "steady",
        "sequence_id", "training_sequence_id", "power_cut_w", "valid_main", "valid_residual"}
    require(required <= set(old), f"Parent contract missing columns: {sorted(required - set(old))}")
    require(set(old) <= set(new), "Candidate dropped parent columns")
    require(set(new) - set(old) <= EXTRA_COLUMNS, "Unreviewed new candidate columns")
    require(len(old) == len(new), "Full NC grid length changed")
    old, new = old.reset_index(drop=True), new.reset_index(drop=True)
    require(not old.duplicated(list(KEYS)).any() and not new.duplicated(list(KEYS)).any(),
            "NC grid keys are not unique")
    for column in set(old) - MUTABLE:
        equal_column(old, new, column)
    was_known, is_known = boolean(old, "process_geometry_known"), boolean(new, "process_geometry_known")
    require(np.array_equal(was_known, np.isfinite(old[["ap_mm", "ae_mm"]].to_numpy(float)).all(axis=1)),
            "Parent geometry support is inconsistent")
    require(np.array_equal(is_known, np.isfinite(new[["ap_mm", "ae_mm"]].to_numpy(float)).all(axis=1)),
            "Candidate geometry support is inconsistent")
    require(not np.any(was_known & ~is_known), "Previously known input lost")
    supported, values, reasons, starts, ends, source_points, source_lines = source_support(
        old, process, float(metadata["diameter_mm"]))
    expected = ~was_known & supported
    restored = ~was_known & is_known
    require(np.array_equal(expected, restored), "Missing or unsupported restored geometry")
    for column in ("ap_mm", "ae_mm", "cutting_contact"):
        equal_column(old, new, column, ~restored)
    require(np.array_equal(new.loc[restored, ["ap_mm", "ae_mm"]].to_numpy(float), values[restored]),
            "Restored geometry differs from its unique source cell")
    contact = boolean(new, "cutting_contact")
    require(np.array_equal(contact[restored], (values[restored] > 0).all(axis=1)),
            "Restored contact does not match source geometry")
    if "support_fill_restored" in new:
        require(np.array_equal(boolean(new, "support_fill_restored"), restored), "Restoration receipt mask differs")
    for column, source in (("support_fill_interval_start_mm", starts), ("support_fill_interval_end_mm", ends),
                           ("support_fill_source_point_index", source_points), ("support_fill_source_file_line", source_lines)):
        if column in new:
            require(np.array_equal(new.loc[restored, column].to_numpy(float), source[restored], equal_nan=True),
                    f"Restored source receipt differs: {column}")
    if {"reference_power_cut_w", "reference_aligned"} & set(new):
        require({"reference_power_cut_w", "reference_aligned"} <= set(new), "Display reference needs values and support")
        reference = new.reference_power_cut_w.to_numpy(float)
        aligned = boolean(new, "reference_aligned")
        require(np.isfinite(reference[aligned]).all(), "Display reference is nonfinite on declared support")
        old_target = old.power_cut_w.to_numpy(float)
        observed = np.isfinite(old_target)
        require(aligned[observed].all() and np.array_equal(reference[observed], old_target[observed]),
                "Display reference changed an existing observed target")
    remaining = ~is_known
    binding = pd.DataFrame({"index": np.flatnonzero(restored), "start_mm": starts[restored],
                            "end_mm": ends[restored], "ap_mm": values[restored, 0], "ae_mm": values[restored, 1]})
    return dict(grid_points=len(old), old_known_points=int(was_known.sum()), restored_points=int(restored.sum()),
                remaining_missing_points=int(remaining.sum()),
                remaining_missing_reasons=dict(Counter(reasons[remaining].tolist())),
                old_main_points=int(boolean(old, "valid_main").sum()),
                old_residual_points=int(boolean(old, "valid_residual").sum()),
                old_input_values_preserved=True, all_parent_label_columns_preserved=True,
                old_sequences_preserved=True, old_scoring_domain_preserved=True,
                source_binding_sha256=json_digest(binding.to_dict("list")))


def audit_previous_support(previous, new):
    """Preserve v1 inputs/labels, while v2 source receipts may have corrected ends."""
    require(len(previous) == len(new) and set(previous) <= set(new), "Previous support grid/columns changed")
    for column in set(previous) - MUTABLE - EXTRA_COLUMNS:
        equal_column(previous, new, column)
    prior_known, now_known = boolean(previous, "process_geometry_known"), boolean(new, "process_geometry_known")
    require(not (prior_known & ~now_known).any(), "Previous restored geometry lost")
    for column in MUTABLE:
        equal_column(previous, new, column, prior_known)
    return dict(previous_known_points=int(prior_known.sum()), newly_restored_points=int((~prior_known & now_known).sum()),
                previous_input_values_preserved=True, previous_labels_and_masks_preserved=True)


def run_audit(parent_path, candidate_path, output_dir, evidence_root, *, relocations=None,
              previous_support_snapshot=None, previous_data_audit=None):
    from spindle_power.load_prediction.artifact_paths import ArtifactResolver
    from spindle_power.load_prediction.samples import read_process_table
    from spindle_power.load_prediction.samples import SampleSet
    from spindle_power.load_prediction.support_fill_snapshot import ENDPOINT_SCHEMA, ENDPOINT_POLICY, checked_endpoint_source

    parent_path, candidate_path = Path(parent_path).resolve(), Path(candidate_path).resolve()
    output_dir, evidence_root = Path(output_dir).resolve(), Path(evidence_root).resolve()
    require(output_dir != evidence_root and output_dir.is_relative_to(evidence_root), "Output must be a new identity below evidence root")
    require(not output_dir.exists(), "Refusing to overwrite an audit directory")
    require(not any(output_dir.is_relative_to(path.parent) or path.is_relative_to(output_dir)
                    for path in (parent_path, candidate_path)), "Audit output overlaps an input snapshot")
    parent_sha, candidate_sha = file_hash(parent_path), file_hash(candidate_path)
    parent, candidate = read_snapshot(parent_path), read_snapshot(candidate_path)
    old_entries, new_entries = development_records(parent), development_records(candidate)
    require(set(old_entries) == set(new_entries), "Candidate record membership changed")
    link = candidate.get("source_snapshot", {}).get("support_fill_derivation", {})
    endpoint = link.get("schema") == ENDPOINT_SCHEMA
    if endpoint:
        require(link.get("endpoint_policy") == ENDPOINT_POLICY, "Unknown endpoint normalization policy")
        require({k: link["parent_snapshot"][k] for k in ("path", "sha256")}
                == dict(path=str(parent_path), sha256=parent_sha), "Endpoint derivation binds another parent")
    require((previous_support_snapshot is None) == (previous_data_audit is None), "Previous snapshot and accepted data audit are a pair")
    previous_entries, previous_refs = None, {}
    if previous_support_snapshot is not None:
        previous_support_snapshot, previous_data_audit = Path(previous_support_snapshot).resolve(), Path(previous_data_audit).resolve()
        previous_refs = dict(previous_support_snapshot=dict(path=str(previous_support_snapshot), sha256=file_hash(previous_support_snapshot)),
                             previous_data_audit=dict(path=str(previous_data_audit), sha256=file_hash(previous_data_audit)))
        prior_audit = json.loads(previous_data_audit.read_text(encoding="utf-8"))
        require(prior_audit.get("schema") == "paper1.support_fill_data_audit.1" and prior_audit.get("status") == "data_passed_prediction_pending"
                and prior_audit.get("data_gate_passed") is True
                and prior_audit.get("checked_records") == 450 and prior_audit.get("failures") == []
                and prior_audit.get("candidate_snapshot") == previous_refs["previous_support_snapshot"]
                and prior_audit.get("parent_snapshot") == dict(path=str(parent_path), sha256=parent_sha),
                "Previous data audit is not accepted for these parent/previous snapshots")
        previous_entries = development_records(read_snapshot(previous_support_snapshot))
        require(set(previous_entries) == set(old_entries), "Previous support membership changed")
    resolver = ArtifactResolver(relocations)
    output_dir.mkdir(parents=True)
    records, failures = [], []
    for rid, old_entry in old_entries.items():
        try:
            new_entry = new_entries[rid]
            meta = old_entry["metadata"]
            check_metadata(meta, new_entry["metadata"])
            old, new = checked_frame(parent_path, old_entry), checked_frame(candidate_path, new_entry)
            path = resolver.resolve(meta["resolved_files"]["process"])
            require(file_hash(path) == meta["file_sha256"]["process"], "Original process source hash mismatch")
            plan = SimpleNamespace(frame=old, program_version=meta["program_version"], machine_version=meta["machine_version"])
            if endpoint:
                expected, process = checked_endpoint_source(SampleSet(old, meta), path, candidate_path.parent, new_entry["metadata"])
                require(new_entry["metadata"] == expected.metadata, "Endpoint derivation changed parent metadata or receipt")
            else:
                process = read_process_table(path, plan, line_offset=meta.get("process_line_offset", 0),
                                             steady_lines=meta.get("steady_lines", []))
            result = audit_frames(old, new, process, meta)
            if previous_entries is not None:
                prior_entry = previous_entries[rid]
                check_metadata(meta, prior_entry["metadata"])
                previous = checked_frame(previous_support_snapshot, prior_entry)
                result["previous_support_comparison"] = audit_previous_support(previous, new)
                result["three_geometry_domains"] = dict(original_parent=int(boolean(old, "process_geometry_known").sum()),
                    v1_restored=int((boolean(previous, "process_geometry_known") & ~boolean(old, "process_geometry_known")).sum()),
                    v2_newly_restored=int((boolean(new, "process_geometry_known") & ~boolean(previous, "process_geometry_known")).sum()))
            require(file_hash(path) == meta["file_sha256"]["process"], "Process source changed during audit")
            records.append(dict(record_id=rid, round_id=meta["round_id"], feature=meta["feature"],
                                parent_csv_sha256=old_entry["sha256"], candidate_csv_sha256=new_entry["sha256"],
                                process_sha256=meta["file_sha256"]["process"], **result))
        except (AuditError, ValueError, KeyError, OSError) as exc:
            failures.append(dict(record_id=rid, error_type=type(exc).__name__, message=str(exc)))
    require(file_hash(parent_path) == parent_sha and file_hash(candidate_path) == candidate_sha,
            "Snapshot manifests changed during audit")
    for reference in previous_refs.values():
        require(file_hash(reference["path"]) == reference["sha256"], "Previous support evidence changed during audit")
    success = len(records) == 450 and not failures
    report = dict(schema="paper1.support_fill_data_audit.1", status="data_passed_prediction_pending" if success else "failed",
        data_gate_passed=success, training_released=False, training_started=False, model_execution_performed=False,
        fixed_weight_prediction_audit_required=True, display_reference_source_reconstruction_verified=False,
        parent_snapshot=dict(path=str(parent_path), sha256=parent_sha),
        candidate_snapshot=dict(path=str(candidate_path), sha256=candidate_sha),
        audit_source_sha256=file_hash(__file__), checked_records=len(records), failures=failures, records=records,
        candidate_derivation_schema=link.get("schema"), source_normalization_policy=link.get("endpoint_policy"),
        **previous_refs,
        totals={key: sum(row[key] for row in records) for key in
                ("grid_points", "old_known_points", "restored_points", "remaining_missing_points")})
    if previous_refs:
        report["three_geometry_domains"] = {key: sum(row["three_geometry_domains"][key] for row in records)
                                             for key in ("original_parent", "v1_restored", "v2_newly_restored")}
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-snapshot", type=Path, required=True)
    parser.add_argument("--candidate-snapshot", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--relocations-reference", type=Path,
                        help="JSON object containing the existing relocation receipt path and sha256")
    parser.add_argument("--previous-support-snapshot", type=Path)
    parser.add_argument("--previous-data-audit", type=Path)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT / "src"))
    relocations = (json.loads(args.relocations_reference.read_text(encoding="utf-8"))
                   if args.relocations_reference else None)
    report = run_audit(args.parent_snapshot, args.candidate_snapshot, args.output_dir,
                        args.evidence_root, relocations=relocations, previous_support_snapshot=args.previous_support_snapshot,
                        previous_data_audit=args.previous_data_audit)
    print(json.dumps({key: report[key] for key in ("status", "checked_records", "totals")}, ensure_ascii=False))
    return 0 if report["data_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
