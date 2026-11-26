"""Freeze F1-only shape diagnostics before Q/P, retaining all 450 identities.

Reference attachments belong to the historical parent. Their positional support
may be joined to the independently audited support-fill child, never to training
inputs or prediction eligibility. No model or event data is loaded here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from spindle_power.load_prediction.common import file_hash, save_json
from spindle_power.load_prediction.samples import SampleSet
from spindle_power.load_prediction.support_fill_snapshot import _development_entries, _frame, _inside, _snapshot
from paper1_prediction_metrics import ShapeProtocolNotReady, fit_shape_protocol

THRESHOLDS = dict(rmse_relative_change_max=-.05, mae_relative_change_max=.02,
    sustained_absolute_bias_relative_change_max=-.20, dynamic_rmse_relative_change_max=.05,
    peak_amplitude_error_relative_change_max=-.10)
GRID_COLUMNS = ("line", "execution_instance", "segment_mm", "path_mm")
SLIM_COLUMNS = (*GRID_COLUMNS, "nc_sequence_id", "sequence_id", "training_sequence_id",
                "power_cut_w", "valid_main", "valid_residual")
FILL_COLUMNS = {"ap_mm", "ae_mm", "process_geometry_known", "cutting_contact"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_hash(path))


def checked(reference):
    value = {key: reference[key] for key in ("path", "sha256")}
    require(ref(value["path"]) == value, f"Evidence changed: {value['path']}")
    return value


def join_support(candidate, parent, attachment):
    """Require exact parent labels/boundaries and original-grid attachment keys."""
    import numpy as np
    require(len(candidate) == len(parent) == len(attachment), "Attachment lost original grid rows")
    for name in parent.columns:
        if name not in FILL_COLUMNS:
            require(name in candidate and candidate[name].equals(parent[name]),
                    f"Parent evidence changed: {name}")
    require("grid_index" in attachment and
            np.array_equal(attachment.grid_index.to_numpy(), np.arange(len(candidate))),
            "Attachment grid is reordered, duplicated, or incomplete")
    for name in GRID_COLUMNS:
        require(name in candidate and name in attachment and
                np.array_equal(candidate[name].to_numpy(), attachment[name].to_numpy(), equal_nan=True),
                f"Attachment grid differs: {name}")
    require("reference_support_segment_id" in attachment, "Attachment lacks raw reference support")
    require({"reference_power_cut_w", "reference_aligned"} <= set(attachment),
            "Attachment lacks its reference evidence")
    finite_parent = np.isfinite(parent.power_cut_w.to_numpy(float))
    reference = attachment.reference_power_cut_w.to_numpy(float)
    require(attachment.reference_aligned.isin([True, False]).all(), "Invalid reference availability flags")
    available = attachment.reference_aligned.to_numpy(bool)
    require(available[finite_parent].all() and np.isfinite(reference[finite_parent]).all()
            and np.array_equal(reference[finite_parent], parent.power_cut_w.to_numpy(float)[finite_parent]),
            "Attachment reference disagrees with finite parent labels")
    # An absent support ID remains absent and makes shape fitting not_ready;
    # neither reference availability nor support is intersected with scoring.
    result = candidate[[name for name in SLIM_COLUMNS if name in candidate]].copy()
    for name in ("grid_index", "reference_support_segment_id"):
        result[name] = attachment[name].to_numpy()
    return result


def attachment_index(manifest, expected, parent_reference):
    require(manifest.get("schema") == "afc.reference_display_candidate_manifest.2"
            and manifest.get("mode") == "development" and manifest.get("preflight_only") is False
            and manifest.get("full_attachment_generation_complete") is True,
            "A completed full development attachment manifest is required")
    require(manifest.get("development_snapshot") == parent_reference,
            "Reference attachments belong to a different parent")
    require(manifest.get("expected_records") == len(expected)
            and manifest.get("completed_records") == len(expected)
            and len(manifest.get("expected_record_ids", [])) == len(expected)
            and set(manifest["expected_record_ids"]) == set(expected), "Attachment expected identity set differs")
    records = manifest.get("records", [])
    require(len(records) == len(expected) and len({row["record_id"] for row in records}) == len(expected)
            and {row["record_id"] for row in records} == set(expected), "Attachment records are incomplete or duplicated")
    require(manifest.get("O3_read") is False and manifest.get("training_performed") is False,
            "Reference preparation scope changed")
    return {row["record_id"]: row for row in records}


def validate_receipt(receipt, record_id, parent_reference, manifest):
    """Accept only generated, exact-parent attachments from one bound execution."""
    require(receipt.get("schema") == "afc.reference_display_candidate.2"
            and receipt.get("status") == "candidate_verified_not_formally_released"
            and receipt.get("record_id") == record_id and receipt.get("snapshot") == parent_reference
            and receipt.get("mode") == "development" and receipt.get("preflight_only") is False
            and receipt.get("source_and_score_files_unchanged") is True
            and receipt.get("training_performed") is False
            and receipt.get("prediction_metrics_evaluated") is False
            and receipt.get("base_validity_exact") is True
            and receipt.get("label_metadata_exact_except_declared_lineage") is True
            and receipt.get("verification", {}).get("historical_alignment_exact") is True
            and receipt.get("verification", {}).get("finite_parent_reference_exact") is True,
            f"Reference receipt scope mismatch: {record_id}")
    require(isinstance(receipt.get("generator_sha256"), str)
            and receipt["generator_sha256"] == manifest.get("generator_sha256")
            and isinstance(receipt.get("candidate_source_sha256"), dict)
            and receipt["candidate_source_sha256"]
            and receipt["candidate_source_sha256"] == manifest.get("candidate_source_sha256"),
            f"Reference generator execution differs: {record_id}")
    require(isinstance(receipt.get("attachment"), dict) and isinstance(receipt.get("support_segments"), dict),
            f"Reference attachment/support provenance is missing: {record_id}")


def freeze(args):
    import pandas as pd
    from spindle_power.load_prediction.experiment_protocol import protocol_folds
    require(not args.output.exists(), "Use a new protocol path; never replace a frozen protocol")
    require(args.minimum_sustained_span_mm > 0 and args.span_basis.strip(), "Predeclare a positive span and its basis")
    sources = {path.name: ref(path) for path in (Path(__file__), Path(__file__).with_name("paper1_prediction_metrics.py"))}
    candidate_reference = ref(args.snapshot)
    candidate = _snapshot(args.snapshot)
    entries = _development_entries(candidate, load_only=True)
    derivation = candidate["source_snapshot"]["support_fill_derivation"]
    parent_reference = checked(derivation["parent_snapshot"])
    parent_path = Path(parent_reference["path"])
    parent = _snapshot(parent_path)
    parents = {row["metadata"]["record_id"]: row for row in _development_entries(parent)}
    data_reference, manifest_reference = ref(args.data_audit), ref(args.reference_manifest)
    audit, manifest = read(args.data_audit), read(args.reference_manifest)
    require(audit.get("schema") == "paper1.support_fill_data_audit.1"
            and audit.get("data_gate_passed") is True and audit.get("checked_records") == 450
            and not audit.get("failures") and audit.get("candidate_snapshot") == candidate_reference,
            "Candidate data audit is not complete or belongs to another snapshot")
    require(checked(audit["parent_snapshot"]) == parent_reference, "Data audit parent differs")
    ids = [row["metadata"]["record_id"] for row in entries]
    indexed = attachment_index(manifest, ids, parent_reference)
    fold = next(row for row in protocol_folds() if row["fold"] == "F1")
    training_ids = [row["metadata"]["record_id"] for row in entries if row["metadata"]["round_id"] in fold["train_rounds"]]
    validation_ids = [rid for rid in ids if rid not in training_ids]
    require(len(training_ids) == 400 and len(validation_ids) == 50, "F1 split is incomplete")
    training_set, samples, references = set(training_ids), [], {}
    for number, entry in enumerate(entries):
        metadata = entry["metadata"]
        rid = metadata["record_id"]
        receipt_ref = checked(indexed[rid]["receipt"])
        receipt = read(receipt_ref["path"])
        validate_receipt(receipt, rid, parent_reference, manifest)
        old_ref = ref(_inside(parent_path.parent, parents[rid]["file"]))
        new_ref = ref(_inside(args.snapshot.parent, entry["file"]))
        require(receipt.get("sample") == old_ref, f"Reference sample mismatch: {rid}")
        attachment_ref = checked(receipt["attachment"])
        support_ref = checked(receipt["support_segments"])
        attachment = pd.read_csv(attachment_ref["path"], float_precision="round_trip")
        new_frame = _frame(args.snapshot.parent, entry)
        old_frame = _frame(parent_path.parent, parents[rid])
        joined = join_support(new_frame, old_frame, attachment)
        checked(attachment_ref)
        if rid in training_set:
            samples.append(SampleSet(joined, metadata))
        references[rid] = dict(**attachment_ref, receipt=receipt_ref, source_csv=old_ref, candidate_csv=new_ref,
                              support_segments=support_ref)
        print(json.dumps(dict(completed_records=number + 1, record_id=rid), ensure_ascii=False), flush=True)
    reasons, protocol = [], None
    try:
        protocol = fit_shape_protocol(samples, minimum_sustained_span_mm=args.minimum_sustained_span_mm,
                                      expected_record_ids=training_ids).to_dict()
    except ShapeProtocolNotReady as exc:
        reasons.append(str(exc))
    for reference in (candidate_reference, parent_reference, data_reference, manifest_reference, *sources.values()):
        checked(reference)
    for value in references.values():
        for reference in (value, value["receipt"], value["source_csv"], value["candidate_csv"], value["support_segments"]):
            checked(reference)
    result = dict(schema="paper1.qp_supplementary_protocol.1", status="ready" if protocol else "not_ready",
        snapshot=candidate_reference, parent_snapshot=parent_reference, data_audit=data_reference,
        reference_manifest=manifest_reference, fold="F1", training_record_ids=training_ids,
        validation_record_ids=validation_ids, shape_protocol=protocol, unavailable_reasons=reasons,
        reference_attachments=references, minimum_sustained_span_mm=args.minimum_sustained_span_mm,
        span_basis=args.span_basis, screening_thresholds=THRESHOLDS, baseline="Q", candidate="P",
        final_selection_epoch=200,
        fixed_primary_reporting_epochs=[100, 200],
        checkpoint_roles={"100": "pause_checkpoint_trajectory", "200": "predeclared_final_endpoint"},
        primary_reports="fixed_100_and_200; final recommendation uses fixed_200 only after the complete schedule",
        incomplete_schedule_decision="undetermined", best_role="supplementary_not_for_final_selection",
        design_scales=dict(diameter_mm=sorted({row["metadata"]["diameter_mm"] for row in entries}),
                           spacing_mm=sorted({row["metadata"]["spacing_mm"] for row in entries})),
        fitting_scope="all 400 F1 training records; no validation labels used for threshold fitting",
        grid_evidence="all parent columns except the four audited support-fill inputs are exact",
        scientific_status="predeclared development screening; no prediction or independent effect evidence",
        source=sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "data-audit", "reference-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--minimum-sustained-span-mm", type=float, required=True)
    parser.add_argument("--span-basis", required=True)
    args = parser.parse_args()
    result = freeze(args)
    print(json.dumps(dict(status=result["status"], output=str(args.output.resolve())), ensure_ascii=False))


if __name__ == "__main__":
    main()
