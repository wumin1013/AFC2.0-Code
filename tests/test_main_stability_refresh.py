"""Synthetic CPU service acceptance; no machining data or model execution."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from spindle_power.load_prediction.common import file_hash, save_json
from spindle_power.load_prediction.feed_identity import feed_sample_identity
from spindle_power.load_prediction.hybrid_labels import ESTIMATED, label_metadata
from spindle_power.load_prediction.main_stability_refresh import (refresh_record, verify_derivation,
    verify_parent_alignment_defaults)
from spindle_power.load_prediction.training_service import TrainingService
from verify_unified_load_prediction import synthetic_records
from test_hybrid_estimated_labels import hybrid_case, idle_case


def geometry_frame():
    positions = np.array([2.5, 3.5, 4.5])
    frame = pd.DataFrame(dict(line=1, execution_instance=0, segment_mm=positions,
        block_length_mm=6., spindle_cmd_rpm=8000., process_geometry_known=True,
        aligned=True, alignment_error_mm=0., spindle_stable=False,
        valid_main=False, valid_residual=True, steady=True, power_cut_w=150.,
        base_power_w=200., base_reference_correction_w=0., feed_input_mm_min=60.,
        feed_actual_mm_min=60., feed_source="measured_actual_feed", ap_mm=1., ae_mm=2.,
        spindle_rpm=8000., diameter_mm=10., teeth=3., curvature_per_mm=0.,
        path_mm=positions, sequence_id=0, training_sequence_id=0))
    for axis in "xyz":
        frame[f"start_{axis}_mm"] = 0.
        frame[f"end_{axis}_mm"] = 6. if axis == "x" else 0.
    for name in ("arc_radius_mm", "arc_sweep_rad", "arc_i_mm", "arc_j_mm", "arc_k_mm"):
        frame[name] = 0.
    return frame


@contextmanager
def corrupt(path):
    path = Path(path)
    before = path.read_bytes()
    path.write_bytes(before + b" ")
    try:
        yield
    finally:
        path.write_bytes(before)


def test_alignment_tolerances_are_verified_from_producer_source():
    assert verify_parent_alignment_defaults(ROOT)["position_tolerance_mm"] == .1


def test_preview_does_not_read_raw_or_create_snapshot(tmp_path):
    service = TrainingService(tmp_path)
    result = service.refresh_main_stability(parent_service_output="missing", parent_snapshot_id="none",
        producer_source="missing", decision_path="missing", training_request={}, execute=False)
    assert result["status"] == "planned_not_executed"
    assert not (tmp_path / "snapshots").exists()


def test_unified_publication_failure_keeps_staging(tmp_path):
    service = TrainingService(tmp_path)
    stage = tmp_path / "snapshots/staging"
    stage.mkdir(parents=True)
    def reject(*args):
        raise ValueError("explicit_test_rejection")
    with pytest.raises(ValueError, match="explicit_test_rejection"):
        service._finalize_prepared_snapshot(stage, {}, [], validator=reject)
    assert (stage / "snapshot.json").is_file()
    assert list((tmp_path / "snapshots").iterdir()) == [stage]


def test_streamed_refresh_true_gate_and_tampering(tmp_path, hybrid_case):
    """Exercise real 900-component persistence and qualification with synthetic raw input.

    Only raw CSV parsing is replaced by an explicit synthetic Measurement. The
    existing base-evidence verifier and TrainingService gate are NOT mocked.
    """
    case = hybrid_case
    feeds, loads = synthetic_records(points=3)
    rows = {}
    for feed, load in zip(feeds, loads):
        meta = {**load.metadata, "status": "active", "machine_id": "synthetic",
            "file_sha256": deepcopy(case.meta["file_sha256"]),
            "resolved_files": {k: case.meta["resolved_files"].get(k, case.meta["resolved_files"]["nc"])
                               for k in ("measurement", "nc", "process", "machine")},
            "sample_rules_version": "afc.sample_rules.5", "process_mapping_rules_version": "afc.process_mapping.2",
            "position_source": "actual", "position_offset_mm": [0., 0., 0.]}
        rid = meta["record_id"]
        row = {**deepcopy(case.row), "record_id": rid, "route": ESTIMATED,
            "status": "synthetic_model_estimated", "qualification_reasons": ["synthetic_only"],
            "original_model_status": case.model.status, "reference_application": "frozen_dynamic_model",
            "commanded_rpm": meta["spindle_cmd_rpm"]}
        rows[rid] = row
        feed.metadata = deepcopy(meta)
        load.metadata = deepcopy(meta)
        load.frame = geometry_frame()
    case.hybrid_report["records"] = rows
    case.publish_hybrid()
    for load in loads:
        label = label_metadata(rows[load.metadata["record_id"]], case.hybrid_evidence,
            valid_ranges=[[0, 6000]], sample_count=6000, original_status=case.model.status)
        load.metadata["hybrid_label"] = label
        load.frame["label_source"] = label["label_source"]
        load.frame["reference_kind"] = label["reference_kind"]
    parent = TrainingService(tmp_path / "parent")
    saved = parent.save_prepared(feeds, loads, source_snapshot=dict(files_verified=True,
        base_validation=case.hybrid_evidence, records=[deepcopy(f.metadata) for f in feeds]))
    parent_feeds, _, original = parent.load_prepared(saved["snapshot_id"])
    feed_identity = feed_sample_identity(parent_feeds)
    raw = deepcopy(case.record)
    raw.values["program_line_raw"] = np.ones(6000)
    raw.values["actual_x_mm"] = raw.time_s.copy()
    raw.values["actual_y_mm"] = np.zeros(6000)
    raw.values["actual_z_mm"] = np.zeros(6000)
    decision = tmp_path / "decision.json"
    save_json(decision, dict(status="fixed_before_r6_real_data_recomputation", synthetic=True))
    service = TrainingService(tmp_path / "consumer")
    with patch("spindle_power.load_prediction.main_stability_refresh.read_stability_measurement", return_value=raw) as reader, \
         patch("spindle_power.load_prediction.samples.prepare_load_samples", side_effect=AssertionError("no full prepare")), \
         patch("spindle_power.load_prediction.base_power.predict_base_from_measured_speed", side_effect=AssertionError("no base replay")):
        result = service.refresh_main_stability(parent_service_output=parent.output_dir,
            parent_snapshot_id=saved["snapshot_id"], producer_source=ROOT, decision_path=decision,
            training_request=dict(snapshot_id=saved["snapshot_id"], mode="all"), execute=True)
    assert reader.call_count == 450
    assert result["status"] == "prepared_not_launched"
    assert not result["ready_issued"] and not result["training_started"]
    current_feeds, current_loads, snapshot = service.load_prepared(result["snapshot_id"])
    assert feed_sample_identity(current_feeds) == feed_identity
    service._validate_prepared(current_feeds, current_loads, snapshot, mode="all")
    directory = Path(result["directory"])
    import json
    consumer = json.loads((directory / "base_evidence/report.json").read_text(encoding="utf-8"))
    assert consumer["production_source_role"].startswith("current_main_stability_consumer")
    assert all(s.metadata["hybrid_label"]["report_sha256"] == case.hybrid_evidence["report_sha256"] for s in current_loads)
    assert result["before_main"] == 0 and result["after_main"] == 1350
    assert not (service.output_dir / "jobs").exists()
    assert (Path(result["operation"]) / "pending_train_request.json").is_file()

    # Value changes, extra metadata, historical/source/evidence corruption and
    # permission to alter extra columns must all fail the same consumer chain.
    current_loads[0].frame.loc[0, "power_cut_w"] += 1.
    with pytest.raises(AssertionError):
        verify_derivation(snapshot, current_loads, directory=directory, consumer_report=consumer)
    current_loads[0].frame.loc[0, "power_cut_w"] -= 1.
    current_loads[0].metadata["tool_healthy"] = False
    with pytest.raises(ValueError, match="元数据"):
        verify_derivation(snapshot, current_loads, directory=directory, consumer_report=consumer)
    current_loads[0].metadata["tool_healthy"] = True
    bad_report = deepcopy(consumer)
    bad_report["records"][current_loads[0].metadata["record_id"]]["idle_power_w"] += 1.
    with pytest.raises(ValueError, match="非允许字段"):
        verify_derivation(snapshot, current_loads, directory=directory, consumer_report=bad_report)
    paths = [Path(saved["directory"]) / "snapshot.json",
             Path(saved["directory"]) / "base_evidence/report.json",
             directory / "parent_producer/samples.py",
             directory / "main_stability_receipt.json",
             directory / current_loads[0].metadata["main_stability_refresh"]["raw_audit"]["file"]]
    for path in paths:
        with corrupt(path), pytest.raises(ValueError):
            verify_derivation(snapshot, current_loads, directory=directory, consumer_report=consumer)
    bad_snapshot = deepcopy(snapshot)
    next(r for r in bad_snapshot["records"] if r["kind"] == "feed")["dtypes"]["valid_label"] = "object"
    with pytest.raises(ValueError, match="进给CSV"):
        verify_derivation(bad_snapshot, current_loads, directory=directory, consumer_report=consumer)
    bad_report = deepcopy(consumer)
    bad_report["production_source_sha256"]["samples.py"] = "0" * 64
    with pytest.raises(ValueError, match="非允许字段"):
        verify_derivation(snapshot, current_loads, directory=directory, consumer_report=bad_report)
    service._validate_prepared(current_feeds, current_loads, snapshot, mode="all")


def test_record_refresh_rejects_custom_tolerance_and_alignment_change(hybrid_case):
    from spindle_power.load_prediction.samples import SampleSet
    case = hybrid_case
    sample = SampleSet(geometry_frame(), {**case.meta, "position_offset_mm": [0., 0., 0.],
        "position_source": "actual", "position_tolerance_mm": .2,
        "hybrid_label": {"raw_sample_count": 6000, "valid_sample_ranges": [[0, 6000]]}})
    with pytest.raises(ValueError, match="容差"):
        refresh_record(sample, case.record)
    sample.metadata.pop("position_tolerance_mm")
    raw = deepcopy(case.record)
    raw.values["program_line_raw"] = np.ones(6000)
    raw.values["actual_x_mm"] = raw.time_s.copy()
    raw.values["actual_y_mm"] = np.zeros(6000)
    raw.values["actual_z_mm"] = np.zeros(6000)
    sample.frame.loc[0, "aligned"] = False
    with pytest.raises(ValueError, match="空间对齐支撑"):
        refresh_record(sample, raw)
    sample.frame.loc[0, "aligned"] = True
    sample.frame.loc[0, "alignment_error_mm"] = .01
    with pytest.raises(AssertionError):
        refresh_record(sample, raw)


def test_stability_refresh_uses_declared_command_policy_when_both_groups_exist(hybrid_case):
    from spindle_power.load_prediction.samples import SampleSet
    case = hybrid_case
    raw = deepcopy(case.record)
    raw.values["program_line_raw"] = np.ones(6000)
    for axis in "xyz":
        raw.values[f"command_{axis}_mm"] = raw.time_s.copy() if axis == "x" else np.zeros(6000)
        raw.values[f"actual_{axis}_mm"] = np.full(6000, 999.)
    sample = SampleSet(geometry_frame(), {**case.meta, "position_offset_mm": [0., 0., 0.],
        "position_policy": "command_only_v1", "position_source": "command",
        "hybrid_label": {"raw_sample_count": 6000, "valid_sample_ranges": [[0, 6000]]}})
    refreshed, _, receipt = refresh_record(sample, raw)
    assert refreshed.metadata["position_policy"] == "command_only_v1"
    assert refreshed.metadata["position_source"] == "command"
    assert receipt["aligned_exact"] and receipt["alignment_error_exact"]
    assert refreshed.frame.aligned.all()
    for axis in "xyz":
        raw.values.pop(f"command_{axis}_mm")
    with pytest.raises(ValueError, match="空间对齐支撑"):
        refresh_record(sample, raw)
