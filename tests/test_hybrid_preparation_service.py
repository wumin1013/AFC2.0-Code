"""Synthetic orchestration checks; these are not data or accuracy qualification."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from spindle_power.load_prediction.base_qualification import embedded_json_artifact
from spindle_power.load_prediction.common import file_hash, save_json
from spindle_power.load_prediction.feed_identity import feed_sample_identity
from spindle_power.load_prediction.hybrid_labels import AFTER, ESTIMATED, MEASURED, MODE, label_metadata
from spindle_power.load_prediction.record_idle import MASK_VERSION, MODE as IDLE_MODE, VERSION as IDLE_VERSION, _digest
from spindle_power.load_prediction.samples import SampleSet
from spindle_power.load_prediction.training_service import (TrainingService, _inherit_hybrid_parent_records,
    evaluate_records, metrics_by_reference)
from verify_unified_load_prediction import synthetic_records, SMALL_CANDIDATE
from test_hybrid_estimated_labels import hybrid_case, idle_case


@pytest.mark.parametrize("extra", [dict(source_prepared_service_output="unused"),
    dict(source_prepared_snapshot_id="0" * 64),
    dict(source_prepared_service_output="unused", source_prepared_snapshot_id="0" * 64)])
def test_source_pair_requires_both_fields_and_hybrid_mode(tmp_path, extra):
    with pytest.raises(ValueError, match="同时指定|仅限"):
        TrainingService(tmp_path).prepare({}, base_validation={"mode": IDLE_MODE}, **extra)


def test_hybrid_pair_dispatches_without_preparing_feed(tmp_path):
    service = TrainingService(tmp_path)
    with patch.object(service, "prepare_hybrid_from_prepared", return_value={"sentinel": True}) as called:
        result = service.prepare({}, base_validation={"mode": MODE}, source_prepared_service_output="parent",
                                 source_prepared_snapshot_id="0" * 64)
    assert result == {"sentinel": True}
    assert called.call_args.kwargs["source_prepared_snapshot_id"] == "0" * 64


def test_incremental_450_feed_427_load_reused_only_23_raw_reads(tmp_path, hybrid_case):
    """Replace numerical parsing/qualification boundaries; exercise real CSV/snapshot persistence."""
    case = hybrid_case
    feeds, loads = synthetic_records()
    rows, source_meta = {}, []
    for number, (feed, load) in enumerate(zip(feeds, loads)):
        meta = {**load.metadata, "status": "active", "file_sha256": case.meta["file_sha256"],
                "process_aligned": True, "process_version": case.meta["file_sha256"]["process"],
                "resolved_files": {name: case.meta["resolved_files"].get(name, case.meta["resolved_files"]["nc"])
                                   for name in ("measurement", "nc", "machine", "process")}}
        meta.update(meta["resolved_files"])
        source_meta.append(deepcopy(meta))
        feed.metadata = deepcopy(meta)
        rid = meta["record_id"]
        retained = number < 427
        row = {**deepcopy(case.row), "record_id": rid,
               "route": MEASURED if retained else ESTIMATED,
               "original_model_status": None if retained else case.model.status,
               "reference_application": AFTER if retained else "frozen_dynamic_model"}
        rows[rid] = row
        old_label = dict(schema=MASK_VERSION, method=IDLE_MODE, version=IDLE_VERSION,
            record_id=rid, report_sha256=case.evidence["report_sha256"],
            measurement_sha256=meta["file_sha256"]["measurement"], prediction_input=False,
            status="qualified" if retained else "excluded", raw_sample_count=6000,
            valid_sample_ranges=[[2000, 6000]] if retained else [], valid_samples=4000 if retained else 0)
        old_label["mask_sha256"] = _digest(dict(valid_sample_ranges=old_label["valid_sample_ranges"],
            valid_samples=old_label["valid_samples"], raw_sample_count=6000, method=MASK_VERSION,
            report_sha256=case.evidence["report_sha256"], measurement_sha256=meta["file_sha256"]["measurement"]))
        load.metadata = {**deepcopy(meta), "record_idle_label": old_label,
                         "base_status": "validated_in_sample_domain" if retained else "excluded_record_idle_labels"}
        if not retained:
            load.frame[["valid_main", "valid_residual"]] = False
    parent_service, target = TrainingService(tmp_path / "parent"), TrainingService(tmp_path / "target")
    parent_saved = parent_service.save_prepared(feeds, loads, source_snapshot=dict(
        records=source_meta, base_validation=case.evidence, files_verified=True))
    parent_feeds, parent_loads, parent_manifest = parent_service.load_prepared(parent_saved["snapshot_id"])
    parent_load_index = {s.metadata["record_id"]: s for s in parent_loads}
    checked_report = {**case.hybrid_report, "records": rows}
    checked = dict(valid=True, issues=[], report=checked_report, files={
        kind: dict(path=case.hybrid_evidence[kind + "_path"], sha256=case.hybrid_evidence[kind + "_sha256"])
        for kind in ("model", "report")})
    manifest = dict(records=[{k: deepcopy(v) for k, v in meta.items()
                             if k not in ("file_sha256", "resolved_files")} for meta in source_meta])
    inspection = target.inspect(manifest, verify_files=False, base_validation=case.hybrid_evidence)
    assert sum(b["code"] == "contact_unknown" for b in inspection["blockers"]) == 450
    assert all("cutting_contact_confirmed" not in meta for meta in manifest["records"])
    parsed = []

    def replacement(process, plan, measurement, base, meta, **kwargs):
        parsed.append(meta["record_id"])
        sample = deepcopy(parent_load_index[meta["record_id"]])
        # A record can retain zero main labels even under authorized estimation.
        sample.frame["power_cut_w"] += 1.25
        sample.frame["valid_residual"] = True
        label = label_metadata(rows[meta["record_id"]], case.hybrid_evidence,
            valid_ranges=[[0, 6000]], sample_count=6000, original_status=case.model.status)
        sample.metadata.update(hybrid_label=label, base_formal_training_release=False)
        for key in ("label_source", "reference_kind"):
            sample.frame[key] = label[key]
        return sample

    with patch.object(TrainingService, "_validate_prepared"), \
         patch("spindle_power.load_prediction.hybrid_labels.verify_hybrid_evidence", return_value=checked), \
         patch("spindle_power.load_prediction.feed.MachineConfig.load"), \
         patch("spindle_power.load_prediction.feed_samples.prepare_feed_features",
               side_effect=lambda *a, **kw: SimpleNamespace(frame=pd.DataFrame(), metadata=dict(program_version="x", machine_version="y"))), \
         patch("spindle_power.load_prediction.training_service._merge_feature_metadata"), \
         patch("spindle_power.load_prediction.samples.read_process_table", return_value=pd.DataFrame()), \
         patch("spindle_power.load_prediction.measurement.read_measurement",
               return_value=SimpleNamespace(values={"program_line_raw": np.array([1.])})) as reader, \
         patch("spindle_power.load_prediction.feed_samples.prepare_feed_samples", side_effect=AssertionError("feed must not recompute")), \
         patch("spindle_power.load_prediction.samples.prepare_load_samples", side_effect=replacement):
        result = target.prepare(manifest, base_validation=case.hybrid_evidence,
            source_prepared_service_output=parent_service.output_dir, source_prepared_snapshot_id=parent_saved["snapshot_id"])
    new_feeds, new_loads, snapshot = target.load_prepared(result["snapshot_id"])
    assert reader.call_count == len(parsed) == result["recomputed_load_records"] == 23
    assert result["unchanged_load_records"] == 427
    assert feed_sample_identity(new_feeds) == feed_sample_identity(parent_feeds)
    old_sha = {r["metadata"]["record_id"]: r["sha256"] for r in parent_manifest["records"] if r["kind"] == "feed"}
    assert all(r["sha256"] == old_sha[r["metadata"]["record_id"]] for r in snapshot["records"] if r["kind"] == "feed")
    for original, updated in zip(parent_loads[:427], new_loads[:427]):
        pd.testing.assert_frame_equal(original.frame, updated.frame[original.frame.columns], check_exact=True)
        assert updated.metadata["record_idle_label"] == original.metadata["record_idle_label"]
    assert all(not s.frame.valid_main.any() for s in new_loads[427:])
    receipt = snapshot["source_snapshot"]["incremental_preparation"]
    assert receipt["feed_labels_recomputed"] == 0
    assert receipt["raw_measurements_parsed"] == 23
    assert len(receipt["parent_csv_sha256"]) == 900
    assert len(receipt["inherited_contact"]) == 450
    assert all("cutting_contact_confirmed" not in s.metadata for s in new_feeds)


@pytest.fixture
def parent_contact_case(tmp_path):
    """Real inspect_manifest structure with process-derived, not explicit, contact."""
    feeds, loads = synthetic_records((1,))
    paths = {}
    for kind in ("nc", "process", "measurement", "machine"):
        path = tmp_path / (kind + ".txt")
        path.write_text("synthetic " + kind, encoding="utf-8")
        paths[kind] = str(path)
    hashes = {kind: file_hash(path) for kind, path in paths.items()}
    records = []
    for feed, load in zip(feeds, loads):
        meta = {**feed.metadata, **paths, "status": "active", "process_aligned": True,
                "process_version": hashes["process"], "resolved_files": deepcopy(paths),
                "file_sha256": deepcopy(hashes), "position_source": "command"}
        records.append(deepcopy(meta))
        feed.metadata = deepcopy(meta)
        load.metadata = deepcopy(meta)
    manifest = dict(records=[{k: deepcopy(v) for k, v in meta.items()
                             if k not in ("file_sha256", "resolved_files", "position_source")} for meta in records])
    from spindle_power.load_prediction.dataset_manifest import inspect_manifest
    inspection = inspect_manifest(manifest, verify_files=False, require_complete=False)
    return SimpleNamespace(current=inspection["records"], source=dict(records=records, files_verified=True),
        feeds=feeds, loads=loads, inspection=inspection, paths=paths)


def test_real_structural_inspection_inherits_only_frozen_process_contact(parent_contact_case):
    case = parent_contact_case
    assert sum(b["code"] == "contact_unknown" for b in case.inspection["blockers"]) == 5
    before = deepcopy([s.metadata for s in case.feeds])
    receipts = _inherit_hybrid_parent_records(case.current, case.source, case.feeds, case.loads)
    assert len(receipts) == 5
    assert all(r["contact_points"] == 4 for r in receipts)
    assert all("cutting_contact_confirmed" not in meta for meta in case.current)
    assert [s.metadata for s in case.feeds] == before
    assert all(meta["position_source"] == "command" for meta in case.current)


@pytest.mark.parametrize("change", ["process_bytes", "nc_bytes", "machine_bytes", "process_path",
    "process_version", "parent_process_unaligned", "parent_source_unverified", "feed_hash",
    "load_hash", "missing_contact", "unknown_contact", "label_outside_contact", "new_context"])
def test_contact_inheritance_rejects_unbound_or_unconfirmed_parent(parent_contact_case, change):
    case = parent_contact_case
    if change.endswith("_bytes"):
        Path(case.paths[change.removesuffix("_bytes")]).write_text("changed", encoding="utf-8")
    elif change == "process_path":
        case.current[0]["resolved_files"]["process"] += ".changed"
    elif change == "process_version":
        case.current[0]["process_version"] = "changed"
    elif change == "parent_process_unaligned":
        case.source["records"][0]["process_aligned"] = False
        case.current[0]["process_aligned"] = False
    elif change == "parent_source_unverified":
        case.source["files_verified"] = False
    elif change in ("feed_hash", "load_hash"):
        samples = case.feeds if change == "feed_hash" else case.loads
        samples[0].metadata["file_sha256"]["measurement"] = "0" * 64
    elif change == "missing_contact":
        case.feeds[0].frame.drop(columns="cutting_contact", inplace=True)
    elif change == "unknown_contact":
        case.feeds[0].frame["cutting_contact"] = np.nan
    elif change == "label_outside_contact":
        case.feeds[0].frame.loc[0, "cutting_contact"] = False
    else:
        case.current[0]["new_unreviewed_context"] = "different"
    with pytest.raises(ValueError):
        _inherit_hybrid_parent_records(case.current, case.source, case.feeds, case.loads)


def test_reference_sources_and_score_mask_survive_evaluation(tmp_path):
    from spindle_power.load_prediction.models import ModelConfig
    from spindle_power.load_prediction.training import TrainingConfig, train_two_stage
    feeds, loads = synthetic_records((1,))
    for number, sample in enumerate(loads):
        source = "frozen_model_estimate" if number == 0 else "postoperative_measured_idle"
        sample.metadata["hybrid_label"] = dict(label_source=source, reference_kind=source)
    loads[0].frame.loc[0, "valid_residual"] = False
    bundle = train_two_stage(loads, model_config=ModelConfig(**SMALL_CANDIDATE["load_model"]),
                             config=TrainingConfig(**SMALL_CANDIDATE["load_training"]), execute=True)
    report = evaluate_records(bundle, loads, feed_samples=feeds, feed_mode="programmed", output_dir=tmp_path)
    for row in report["records"]:
        frame = pd.read_csv(tmp_path / row["prediction_file"])
        assert frame.label_source.eq(row["label_source"]).all()
        assert frame.reference_kind.eq(row["reference_kind"]).all()
        np.testing.assert_array_equal(frame.score_mask, frame.scored)
    assert set(report["metrics_by_reference"]) == {"frozen_model_estimate", "postoperative_measured_idle"}
    assert report["metrics_by_reference"]["frozen_model_estimate"]["effective_counts"]["count"] == 3


def test_reference_grouping_keeps_equal_record_feature_round_weights():
    rows = [dict(round_id=1, feature="a", metrics={"mae_w": 1., "count": 100}, reference_kind="a"),
            dict(round_id=2, feature="a", metrics={"mae_w": 3., "count": 1}, reference_kind="a"),
            dict(round_id=1, feature="b", metrics={"mae_w": 20., "count": 3}, reference_kind="b")]
    grouped = metrics_by_reference(rows)
    assert grouped["a"]["mae_w"] == 2.
    assert grouped["b"]["mae_w"] == 20.
