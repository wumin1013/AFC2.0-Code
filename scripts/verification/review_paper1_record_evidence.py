"""Bind record-level review facts to existing full scans and masked preparation.

This does not create training samples or release a training job. Missing point
support remains masked; an observed program end does not prove every path point
has a usable label. Full manifest qualification must still run before prepare.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from spindle_power.load_prediction.common import file_hash, save_json, version_of
from spindle_power.load_prediction.experiment_protocol import PROGRAM_FEATURES
from spindle_power.load_prediction.measurement import detect_encoding
from spindle_power.load_prediction.sample_rules import SAMPLE_RULES_VERSION
from afc_common.process_path import PROCESS_MAPPING_RULES_VERSION
from prepare_paper1_feed import preparation_source_fingerprints


def record_decisions(record, numeric, aligned, prepared):
    """Evidence predicates are intentionally separate from author confirmations."""
    sources = prepared.get("source_files", {})
    bound = (numeric.get("file_sha256") == aligned.get("measurement_sha256")
             == sources.get("measurement", {}).get("sha256")
             and aligned.get("nc_sha256") == sources.get("nc", {}).get("sha256")
             and all(r.get("record_id") == record["record_id"] for r in (numeric, aligned, prepared)))
    numeric_ok = (numeric.get("scan_status") == "completed" and numeric.get("samples", 0) > 0
                  and numeric.get("invalid_samples") == 0 and numeric.get("invalid_feed_samples") == 0
                  and all(v == 0 for v in numeric.get("duplicate_conflicts", {}).values()))
    selected = aligned.get("selected_candidate", {})
    line_offset = selected.get("line_offset")
    markers = numeric.get("observed_end_marker_candidates", [])
    end_seen = bool(markers) and line_offset == 1 and all(m.get("raw_minus_one") is True for m in markers)
    intended_program = not record.get("recollection_pending", False)
    segments = prepared.get("process_segments", [])
    endpoint_differences = [abs(s["endpoint_difference_mm"]) for s in segments
                            if s.get("endpoint_difference_mm") is not None]
    # The accepted mapping uses NC length and source point order, not rounded
    # exported s. Source retention and unique uniform positions are mandatory.
    process_ok = (
        prepared.get("process_mapping_rules_version") == PROCESS_MAPPING_RULES_VERSION
        and prepared.get("process_source_indices_preserved") is True
        and prepared.get("source_process_points", 0) > 0
        and prepared["source_process_points"] == prepared.get("canonical_process_rows")
        and prepared.get("process_point_order_unique") is True
        and prepared.get("mapped_positive_segments_strictly_increasing") is True
        and prepared.get("mapped_segment_endpoints_match_nc") is True
        and prepared.get("process_geometry_conflict_positions") == 0
        and bool(endpoint_differences) and max(endpoint_differences) <= 1e-8
        and prepared.get("cutting_contact_points", 0) > 0)
    masked_alignment = (not aligned.get("parser_issues") and line_offset == 1
                        and prepared.get("candidate_aligned_contact_points", 0) > 0
                        and prepared.get("max_accepted_alignment_error_mm") is not None
                        and prepared["max_accepted_alignment_error_mm"] <= .1 + 1e-9)
    cutting_end = end_seen or record.get("post_cut_stop_observed") is True
    checks = dict(evidence_bound=bound, numeric_content_valid=numeric_ok,
                  intended_program_uncontested=intended_program, cutting_acquisition_end_observed=cutting_end,
                  process_points_preserved_and_uniformly_mapped=process_ok,
                  retained_labels_within_alignment_tolerance=masked_alignment)
    complete = bound and numeric_ok and cutting_end and intended_program and record.get("no_operator_pause") is True
    alignment_ok = bound and numeric_ok and intended_program and masked_alignment
    return dict(checks=checks, complete=complete, alignment_reliable=alignment_ok,
                process_aligned=bound and intended_program and process_ok,
                terminal_nc_marker_observed=end_seen,
                process_contact_conflict_positions=prepared.get("process_contact_conflict_positions"),
                max_process_endpoint_difference_mm=max(endpoint_differences) if endpoint_differences else None,
                failed_checks=[key for key, value in checks.items() if not value])


def bind_steady_process(record, entry, evidence, *, hash_file=file_hash):
    """Bind independently generated point labels without changing quality facts."""
    if (evidence.get("process_mapping_rules_version") != PROCESS_MAPPING_RULES_VERSION
            or evidence.get("source_points_preserved") is not True
            or evidence.get("program_version") != record.get("program_version")
            or evidence.get("nc_sha256") != hash_file(record["nc"])
            or evidence.get("source_process_sha256") != hash_file(record["process"])
            or hash_file(entry["canonical_process"]) != entry["canonical_sha256"]
            or hash_file(entry["evidence"]) != entry["evidence_sha256"]):
        raise ValueError(f"Steady evidence is not bound to the current inputs: {record['record_id']}")
    record["source_process"] = dict(path=record["process"], sha256=hash_file(record["process"]),
                                    process_version=record.get("process_version"))
    record.update(process=entry["canonical_process"], process_version=entry["canonical_sha256"],
                  process_line_offset=0, steady_lines=[],
                  segmentation_evidence_path=entry["evidence"],
                  segmentation_evidence_sha256=entry["evidence_sha256"],
                  process_mapping_rules_version=PROCESS_MAPPING_RULES_VERSION,
                  sample_rules_version=SAMPLE_RULES_VERSION)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--alignment-dir", type=Path, required=True)
    parser.add_argument("--preparation-dir", type=Path, required=True)
    parser.add_argument("--steady-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.preparation_dir / "SUPERSEDED.json").is_file():
        raise ValueError("Preparation is explicitly superseded; regenerate with the accepted process mapping")
    args.output.mkdir(parents=True, exist_ok=False)
    load = lambda p: json.loads(p.read_text(encoding="utf-8"))
    manifest = deepcopy(load(args.manifest))
    preparation = load(args.preparation_dir / "pilot_report.json")
    if preparation.get("identity", {}).get("sample_rules_version") != SAMPLE_RULES_VERSION:
        raise ValueError("Preparation sample rules do not match the current implementation")
    if preparation.get("identity", {}).get("process_mapping_rules_version") != PROCESS_MAPPING_RULES_VERSION:
        raise ValueError("Preparation process mapping rules do not match the current implementation")
    if preparation.get("source_changed_during_preparation"):
        raise ValueError("Preparation sources changed during the original run")
    current_sources = preparation_source_fingerprints()
    for name, digest in preparation["identity"]["code_fingerprints"].items():
        if current_sources.get(name) != digest:
            raise ValueError(f"Preparation implementation changed: {name}")
    steady_summary = load(args.steady_dir / "summary.json")
    steady_integrity = load(args.steady_dir / "integrity_report.json")
    if (steady_integrity.get("all_checks_passed") is not True
            or steady_integrity.get("summary_sha256") != file_hash(args.steady_dir / "summary.json")):
        raise ValueError("Steady evidence integrity was not independently verified")
    steady_config = args.steady_dir / "segmentation_config.json"
    if file_hash(steady_config) != steady_summary.get("config_sha256"):
        raise ValueError("Steady configuration hash changed")
    for name, digest in load(steady_config)["rules"].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError(f"Steady implementation changed: {name}")
    steady_entries = {}
    for entry in steady_summary["records"]:
        for rid in entry["record_ids"]:
            if rid in steady_entries:
                raise ValueError(f"Duplicate steady assignment: {rid}")
            steady_entries[rid] = entry
    if set(steady_entries) != {r["record_id"] for r in manifest["records"]}:
        raise ValueError("Steady evidence does not cover exactly the manifest members")
    review = []
    for record in manifest["records"]:
        if record.get("dataset_role") != "development" or record.get("status") != "active":
            raise ValueError("This review accepts only the current development manifest")
        rid = record["record_id"]
        paths = dict(numeric=args.audit_dir / "records" / f"{rid}.json",
                     alignment=args.alignment_dir / "records" / f"{rid}.json",
                     preparation=args.preparation_dir / "records" / rid / "report.json")
        numeric, alignment, prepared = [load(paths[name]) for name in ("numeric", "alignment", "preparation")]
        for field, source in prepared["source_files"].items():
            if Path(record[field]).resolve() != Path(source["path"]).resolve() or file_hash(source["path"]) != source["sha256"]:
                raise ValueError(f"Source changed: {rid} {field}")
        nc_lines = Path(record["nc"]).read_text(encoding=detect_encoding(record["nc"])).splitlines()
        last_cut_line = max((s["line"] for s in prepared["process_segments"]), default=0)
        post_cut_stops = [i for i, line in enumerate(nc_lines, 1)
                          if i > last_cut_line and re.search(r"(?:^|\s)M0?5(?:\s|$)", line.strip(), re.I)]
        offset = alignment["selected_candidate"]["line_offset"]
        observed_lines = {run["line"] + offset for run in numeric.get("line_runs", [])}
        record["post_cut_stop_observed"] = bool(post_cut_stops) and any(line in observed_lines for line in post_cut_stops)
        record["cutting_end_evidence"] = dict(last_cutting_nc_line=last_cut_line,
            post_cut_stop_nc_lines=post_cut_stops, post_cut_stop_observed=record["post_cut_stop_observed"],
            interpretation="Observed programmed spindle stop after the final cutting move; rapid/home completion is separate.")
        decisions = record_decisions(record, numeric, alignment, prepared)
        record.update({key: decisions[key] for key in ("complete", "alignment_reliable", "process_aligned")})
        selected = alignment["selected_candidate"]
        record.update(position_offset_mm=selected["position_offset_mm"],
                      measurement_line_offset=selected["line_offset"], coordinate_mode="workpiece_cutting")
        commands = [line.strip() for line in nc_lines
                    if "G05" in line.upper()]
        observed = dict(controller_header=record.get("raw_controller_header"), commands=commands)
        record.update(machining_mode="recorded_execution_settings_semantics_uninterpreted",
                      machine_settings_version="observed_" + version_of(observed),
                      machine_settings_evidence=observed,
                      machine_settings_interpretation="observed fingerprint; not manufacturer semantics or cross-version equivalence")
        evidence = {name: dict(path=str(path.resolve()), sha256=file_hash(path)) for name, path in paths.items()}
        record["review_evidence"] = dict(**evidence, decisions=decisions,
            alignment_scope="Only geometrically supported masked targets; unsupported boundaries remain missing.",
            completeness_scope="Full scanned file reaches M30 or the post-cut M05 after the final cutting move; not 100% label coverage or rapid/home completion.",
            process_scope="NC line/instance and within-line source point order kL/n; all source points retained. Unmapped points remain excluded.")
        entry = steady_entries[rid]
        bind_steady_process(record, entry, load(Path(entry["evidence"])))
        review.append(dict(record_id=rid, round_id=record["round_id"], feature=record["feature"], **decisions,
                           candidate_points=prepared["candidate_aligned_contact_points"],
                           formal_training_release=False, evidence=evidence))
    by_round = {}
    for record in manifest["records"]:
        by_round.setdefault(record["round_id"], []).append(record)
    for records in by_round.values():
        complete = len(records) == 5 and {r["feature"] for r in records} == set(PROGRAM_FEATURES) and all(r["complete"] for r in records)
        for record in records:
            record["round_complete"] = complete
    manifest.update(review_status="record_evidence_reviewed_pending_recollection_and_formal_preparation",
                    author_confirmations_source=str(args.manifest.resolve()),
                    formal_training_release=False)
    save_json(args.output / "reviewed_manifest.json", manifest)
    summary = dict(created_at=datetime.now(timezone.utc).isoformat(), records=len(review),
        completed_records=sum(r["complete"] for r in review),
        aligned_records=sum(r["alignment_reliable"] for r in review),
        process_mapping_records=sum(r["process_aligned"] for r in review),
        complete_rounds=sum(all(r["round_complete"] for r in rows) for rows in by_round.values()),
        failed_check_counts=dict(Counter(k for r in review for k in r["failed_checks"])),
        pending=[r["record_id"] for r in review if r["failed_checks"]],
        records_review=review, manifest_sha256=file_hash(args.output / "reviewed_manifest.json"),
        script_sha256=file_hash(__file__), formal_training_release=False,
        process_mapping_rules_version=PROCESS_MAPPING_RULES_VERSION,
        sample_rules_version=SAMPLE_RULES_VERSION,
        steady_evidence=dict(path=str((args.steady_dir / "summary.json").resolve()),
                            sha256=file_hash(args.steady_dir / "summary.json")),
        next_steps=["Audit and activate the new R74 T2 record without inheriting the old acquisition date.",
                    "Run the shared manifest inspector and regenerate a formal numerical snapshot.",
                    "Load training additionally requires qualified base-power evidence and supported steady labels."],
        limitations=["No neural training was performed.", "Unknown acquisition times and tool sessions remain unknown.",
                     "Recorded controller settings are not a proof of equivalence between controller versions."])
    save_json(args.output / "review.json", summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ("records_review",)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
