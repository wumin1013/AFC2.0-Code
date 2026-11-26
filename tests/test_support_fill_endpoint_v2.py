"""Synthetic endpoint contracts; real 450-record audits are separate commands."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spindle_power.load_prediction import support_fill_snapshot as module
from spindle_power.load_prediction.common import file_hash, save_json
from spindle_power.load_prediction.samples import SampleSet
from spindle_power.load_prediction.training_checkpoint import digest
from spindle_power.load_prediction.training_service import TrainingService
from test_paper1_support_fill_audit import audit

spec = importlib.util.spec_from_file_location("endpoint_prediction_audit", ROOT / "scripts/verification/audit_paper1_support_predictions.py")
prediction_audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prediction_audit)


def endpoint_case(length=13.163390767199763, count=26):
    bounds = np.asarray([length * i / count for i in range(count + 1)])
    process = pd.DataFrame(dict(line=1, execution_instance=0, segment_start_mm=bounds[:-1], segment_mm=bounds[1:],
        ap_mm=1., ae_mm=2., point_index_in_line=np.arange(1, count + 1), points_in_line=count,
        source_point_index=np.arange(count), source_file_line=np.arange(2, count + 2), mapping_status="mapped_uniform_nc"))
    parent = pd.DataFrame(dict(line=[1, 1, 1], execution_instance=0, segment_mm=[0., length/2, length],
        path_mm=[0., length/2, length], block_length_mm=length, ap_mm=[1.5, 1.5, np.nan], ae_mm=[2., 2., np.nan],
        process_geometry_known=[True, True, False], process_geometry_conflict_support=False,
        cutting_contact=[True, True, False], steady=[True, True, False], sequence_id=0, training_sequence_id=0,
        power_cut_w=[10., 20., np.nan], valid_main=[True, True, False], valid_residual=[True, True, False], diameter_mm=10.))
    return process, parent


def source_sample(tmp_path):
    process, frame = endpoint_case()
    source = tmp_path / "canonical.csv"
    process.to_csv(source, index=False, encoding="utf-8")
    meta = dict(record_id="synthetic", program_version="synthetic-program", machine_version="synthetic-machine",
                diameter_mm=10., file_sha256={"process": file_hash(source)}, resolved_files={"process": str(source)})
    return SampleSet(frame, meta), source


def test_only_proven_negative_terminal_changes_and_all_internal_values_are_exact():
    process, parent = endpoint_case()
    before = process.copy(deep=True)
    normalized, changes = module.normalize_source_endpoints(process, parent, diameter_mm=10.)
    assert len(changes) == 1 and changes[0]["delta_ulp"] == 1
    assert normalized.segment_mm.iloc[-1] == parent.block_length_mm.iloc[0]
    pd.testing.assert_frame_equal(process, before, check_exact=True)
    pd.testing.assert_frame_equal(normalized.iloc[:-1], before.iloc[:-1], check_exact=True)
    pd.testing.assert_frame_equal(normalized.drop(columns="segment_mm"), before.drop(columns="segment_mm"), check_exact=True)


@pytest.mark.parametrize("length,count", [(50.17331351316155, 100), (2., 4), (0., 3), (13.163390767199763, 1)])
def test_positive_exact_zero_and_single_cell_sources_are_not_expanded(length, count):
    process, parent = endpoint_case(length, count)
    normalized, changes = module.normalize_source_endpoints(process, parent, diameter_mm=10.)
    assert changes == []
    pd.testing.assert_frame_equal(normalized, process, check_exact=True)


@pytest.mark.parametrize("mutation", ["interior", "mapping", "count", "point_order", "source_identity", "geometry", "overlap"])
def test_incomplete_nonuniform_ambiguous_or_illegal_source_is_rejected(mutation):
    process, parent = endpoint_case()
    if mutation == "interior":
        process.loc[4, "segment_mm"] += .01
        process.loc[5, "segment_start_mm"] += .01
    elif mutation == "mapping":
        process.loc[4, "mapping_status"] = "unmapped_nc_group"
    elif mutation == "count":
        process.loc[4, "points_in_line"] = 25
    elif mutation == "point_order":
        process.loc[4, "point_index_in_line"] = 4
    elif mutation == "source_identity":
        process.loc[4, "source_point_index"] = 3
    elif mutation == "geometry":
        process.loc[4, "ae_mm"] = 20.
    else:
        process.loc[4, "segment_start_mm"] -= .01
    with pytest.raises(ValueError):
        module.normalize_source_endpoints(process, parent, diameter_mm=10.)


def test_missing_source_and_larger_gaps_are_not_filled(tmp_path):
    sample, source = source_sample(tmp_path)
    process = pd.read_csv(source, float_precision="round_trip")
    process["line"] = 99
    normalized, changes = module.normalize_source_endpoints(process, sample.frame, diameter_mm=10.)
    assert changes == []
    process, parent = endpoint_case()
    process.loc[25, "segment_mm"] = np.nextafter(process.segment_mm.iloc[-1], -np.inf)
    normalized, changes = module.normalize_source_endpoints(process, parent, diameter_mm=10.)
    assert changes == []
    derived, _ = module.fill_missing_process_support(parent, normalized, diameter_mm=10.)
    assert not derived.process_geometry_known.iloc[-1]


def test_endpoint_receipt_binds_original_and_normalized_sources(tmp_path):
    sample, source = source_sample(tmp_path)
    derived, _, content = module.endpoint_normalized_sample(sample, source)
    receipt = derived.metadata["support_fill"]["source_normalization"]
    assert receipt["original_source"]["sha256"] == sample.metadata["file_sha256"]["process"]
    destination = tmp_path / "snapshot"
    target = destination / receipt["normalized_source"]["file"]
    target.parent.mkdir(parents=True)
    target.write_bytes(content)
    expected, _ = module.checked_endpoint_source(sample, source, destination, derived.metadata)
    pd.testing.assert_frame_equal(derived.frame, expected.frame, check_exact=True)
    assert derived.frame.process_geometry_known.all()
    for column in sample.frame.columns:
        if column not in module.fill_missing_process_support.__globals__["MUTABLE_COLUMNS"]:
            pd.testing.assert_series_equal(sample.frame[column], derived.frame[column], check_exact=True)
    assert {k: v for k, v in derived.metadata.items() if k != "support_fill"} == sample.metadata
    modified = pd.read_csv(target, float_precision="round_trip")
    modified.loc[4, "ap_mm"] = 3.
    modified.to_csv(target, index=False)
    derived.metadata["support_fill"]["source_normalization"]["normalized_source"]["sha256"] = file_hash(target)
    with pytest.raises(ValueError, match="回执"):
        module.checked_endpoint_source(sample, source, destination, derived.metadata)


def test_changed_original_hash_fails(tmp_path):
    sample, source = source_sample(tmp_path)
    source.write_text(source.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        module.endpoint_normalized_sample(sample, source)


def test_v2_derivation_uses_original_parent_and_deduplicates_verified_sources(tmp_path, monkeypatch):
    sample, source = source_sample(tmp_path)
    parent_service = TrainingService(tmp_path / "parent")
    directory = parent_service.output_dir / "snapshots" / "preparing"
    directory.mkdir(parents=True)
    entries = []
    for number in range(2):
        name = f"load_{number}.csv"
        sample.frame.to_csv(directory / name, index=False)
        meta = deepcopy(sample.metadata)
        meta["record_id"] = f"synthetic-{number}"
        entries.append(dict(kind="load", file=name, sha256=file_hash(directory/name), metadata=meta,
                            dtypes={k: str(v) for k, v in sample.frame.dtypes.items()}))
    parent = dict(schema="test", synthetic=False, records=entries, source_snapshot=dict(files_verified=True,
        route="nominal_programmed_residual_v1", load_route_derivation=dict(historical_consumer_source="frozen")))
    parent["snapshot_id"] = digest(parent)
    save_json(directory / "snapshot.json", parent)
    fixed = directory.with_name(parent["snapshot_id"])
    directory.rename(fixed)
    # Synthetic publication test only; no claim of 450-record scientific acceptance.
    monkeypatch.setattr(module, "_development_entries", lambda snapshot, **kwargs: snapshot["records"])
    monkeypatch.setattr(TrainingService, "_verify_load_route_derivation", lambda *args, **kwargs: dict(verified=True))
    service = TrainingService(tmp_path / "v2")
    result = service.derive_support_fill(parent_service_output=str(parent_service.output_dir), parent_snapshot_id=parent["snapshot_id"],
        historical_route_source=str(tmp_path / "archive"), endpoint_policy=module.ENDPOINT_POLICY, execute=True)
    output = Path(result["directory"])
    snapshot = json.loads((output / "snapshot.json").read_text())
    assert snapshot["source_snapshot"]["support_fill_derivation"]["schema"] == module.ENDPOINT_SCHEMA
    assert len(list((output / "process_sources").glob("*.csv"))) == 1
    assert module.verify_support_fill_snapshot(service, snapshot, output, require_release=False)["verified"]
    assert not result["training_started"] and not result["training_released"]
    broken = deepcopy(snapshot)
    broken["records"][0]["metadata"]["support_fill"]["source_normalization"]["changed_cells"][0]["normalized_end_mm"] += 1.
    broken["snapshot_id"] = digest({k: v for k, v in broken.items() if k != "snapshot_id"})
    with pytest.raises(ValueError, match="回执"):
        module.verify_support_fill_snapshot(service, broken, output, require_release=False)


def test_previous_v1_inputs_and_labels_are_preserved():
    process, parent = endpoint_case()
    normalized, _ = module.normalize_source_endpoints(process, parent, diameter_mm=10.)
    previous, _ = module.fill_missing_process_support(parent, process, diameter_mm=10.)
    current, _ = module.fill_missing_process_support(parent, normalized, diameter_mm=10.)
    assert audit.audit_previous_support(previous, current)["newly_restored_points"] == 1
    current.loc[0, "power_cut_w"] = 999.
    with pytest.raises(ValueError, match="Protected"):
        audit.audit_previous_support(previous, current)


def test_three_prediction_domains_report_changes_without_shrinking_score_mask():
    from spindle_power.load_prediction.support_prediction import COMPONENTS, QUANTILES
    frames = []
    for available in ([True, True, False, False, False], [True, True, True, True, False], [True]*5):
        frame = pd.DataFrame(dict(line=1, execution_instance=0, segment_mm=np.arange(5.), path_mm=np.arange(5.),
            power_cut_w=[10., 20., np.nan, np.nan, np.nan], scored=[True, True, False, False, False],
            prediction_available=available))
        for key in COMPONENTS + QUANTILES:
            frame[key] = np.where(available, np.arange(5.), np.nan)
        frames.append(frame)
    frames[2].loc[2, "cutting_power_w"] += 3.
    report = prediction_audit.compare_previous_predictions(*frames)
    assert report["domains"] == dict(original_parent=2, v1_restored=2, v2_newly_restored=1)
    assert report["component_differences"]["v1_restored"]["cutting_power_w"]["max_abs_w"] == 3.
    assert report["component_differences"]["original_parent"]["cutting_power_w"]["max_abs_w"] == 0.
    frames[2].loc[0, "scored"] = False
    with pytest.raises(AssertionError):
        prediction_audit.compare_previous_predictions(*frames)
