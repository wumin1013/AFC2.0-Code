"""Tiny synthetic derivation/publication checks; parent scientific validation is mocked explicitly."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spindle_power.load_prediction import support_fill_snapshot as module
from spindle_power.load_prediction.common import file_hash, save_json
from spindle_power.load_prediction.training_checkpoint import digest, source_fingerprints
from spindle_power.load_prediction.training_service import TrainingService
from test_support_fill import source_sample


@pytest.fixture
def tiny_derivation(tmp_path, monkeypatch):
    sample, process = source_sample(tmp_path)
    parent_service = TrainingService(tmp_path / "parent")
    working = parent_service.output_dir / "snapshots" / "preparing"
    working.mkdir(parents=True)
    records = []
    for number in range(2):
        name = f"load_{number}.csv"
        sample.frame.to_csv(working / name, index=False, encoding="utf-8")
        metadata = deepcopy(sample.metadata)
        metadata["record_id"] = f"synthetic-{number}"
        records.append(dict(kind="load", file=name, sha256=file_hash(working / name), metadata=metadata,
                            dtypes={key: str(value) for key, value in sample.frame.dtypes.items()}))
    parent = dict(schema="test", synthetic=False, source_snapshot=dict(files_verified=True,
        route="nominal_programmed_residual_v1", load_route_derivation=dict(historical_consumer_source="frozen-r6")),
        records=records)
    parent["snapshot_id"] = digest(parent)
    save_json(working / "snapshot.json", parent)
    directory = working.with_name(parent["snapshot_id"])
    working.rename(directory)
    original_hashes = {path.name: file_hash(path) for path in directory.iterdir() if path.is_file()}
    calls = []
    def parent_proof(service, snapshot, location, **kwargs):
        calls.append(dict(snapshot_id=snapshot["snapshot_id"], directory=str(location), **kwargs))
        assert snapshot == parent
        assert Path(location) == directory
        return {"verified": True}
    # These mocks limit the fixture to two fake CSVs. No real parent evidence or
    # 450-record scientific acceptance is claimed by this unit test.
    monkeypatch.setattr(TrainingService, "_verify_load_route_derivation", parent_proof)
    monkeypatch.setattr(module, "_development_entries", lambda value, **kwargs: value["records"])
    service = TrainingService(tmp_path / "candidate")
    request = dict(parent_service_output=str(parent_service.output_dir), parent_snapshot_id=parent["snapshot_id"],
                   historical_route_source=str(tmp_path / "archived-nominal"), execute=True)
    return service, request, parent, directory, original_hashes, calls


def derive(fixture):
    service, request, *_ = fixture
    result = service.derive_support_fill(**request)
    directory = Path(result["directory"])
    snapshot = json.loads((directory / "snapshot.json").read_text(encoding="utf-8"))
    return result, snapshot, directory


def test_streamed_derivation_preserves_parent_and_verifies_historical_chain(tiny_derivation):
    service, request, parent, parent_dir, hashes, calls = tiny_derivation
    result, snapshot, directory = derive(tiny_derivation)
    assert result["status"] == "derived_data_verified_release_pending"
    assert not result["training_started"] and not result["training_released"]
    assert {path.name: file_hash(path) for path in parent_dir.iterdir()} == hashes
    assert len(calls) == 2
    assert all(call["historical_route_source"] == request["historical_route_source"] for call in calls)
    assert {key: value for key, value in snapshot["source_snapshot"].items() if key != "support_fill_derivation"} == parent["source_snapshot"]
    assert len(snapshot["records"]) == 2 and all(item["kind"] == "load" for item in snapshot["records"])
    for old, new in zip(parent["records"], snapshot["records"]):
        assert {key: value for key, value in new["metadata"].items() if key != "support_fill"} == old["metadata"]
        assert new["metadata"]["support_fill"]["source_file_verified"]
    assert module.verify_support_fill_snapshot(service, snapshot, directory, require_release=False)["verified"]


def reseal(snapshot, directory):
    snapshot["snapshot_id"] = digest({key: value for key, value in snapshot.items() if key != "snapshot_id"})
    save_json(directory / "snapshot.json", snapshot)


def test_resigning_candidate_hash_cannot_hide_changed_supervision(tiny_derivation):
    service, *_ = tiny_derivation
    _, snapshot, directory = derive(tiny_derivation)
    item = snapshot["records"][0]
    path = directory / item["file"]
    frame = pd.read_csv(path, float_precision="round_trip")
    frame.loc[2, "power_cut_w"] = 999.
    frame.to_csv(path, index=False, encoding="utf-8")
    item["sha256"] = file_hash(path)
    snapshot["source_snapshot"]["support_fill_derivation"]["record_bindings"][0]["candidate_csv_sha256"] = item["sha256"]
    reseal(snapshot, directory)
    with pytest.raises(ValueError, match="保护列"):
        module.verify_support_fill_snapshot(service, snapshot, directory, require_release=False)


def test_correct_disk_snapshot_cannot_hide_modified_in_memory_labels(tiny_derivation):
    service, *_ = tiny_derivation
    result, snapshot, directory = derive(tiny_derivation)
    _, loads, _ = service.load_prepared(result["snapshot_id"])
    assert module.verify_support_fill_snapshot(service, snapshot, directory, require_release=False, loads=loads)["verified"]
    loads[0].frame.loc[2, "power_cut_w"] = 1234.
    with pytest.raises(ValueError, match="内存补缺样本数值"):
        module.verify_support_fill_snapshot(service, snapshot, directory, require_release=False, loads=loads)


@pytest.mark.parametrize("field", ["prepared_measurement_sha256", "hybrid_label"])
def test_parent_metadata_cannot_be_resigned(tiny_derivation, field):
    service, *_ = tiny_derivation
    _, snapshot, directory = derive(tiny_derivation)
    snapshot["records"][0]["metadata"][field] = "replacement-evidence"
    reseal(snapshot, directory)
    with pytest.raises(ValueError, match="原元数据"):
        module.verify_support_fill_snapshot(service, snapshot, directory, require_release=False)


def test_parent_evidence_failure_stops_before_candidate_csv_write(tiny_derivation, monkeypatch):
    service, request, *_ = tiny_derivation
    def reject(*args, **kwargs):
        raise ValueError("old parent proof failed")
    monkeypatch.setattr(TrainingService, "_verify_load_route_derivation", reject)
    with pytest.raises(ValueError, match="old parent proof"):
        service.derive_support_fill(**request)
    assert not (service.output_dir / "snapshots").exists()


def test_existing_output_and_repeated_derivation_are_rejected(tiny_derivation):
    service, request, *_ = tiny_derivation
    derive(tiny_derivation)
    with pytest.raises(ValueError, match="拒绝覆盖"):
        service.derive_support_fill(**request)


def test_data_verification_never_implies_training_release(tiny_derivation):
    service, *_ = tiny_derivation
    _, snapshot, directory = derive(tiny_derivation)
    with pytest.raises(ValueError, match="尚无正式训练放行"):
        module.verify_support_fill_snapshot(service, snapshot, directory)
    release = dict(schema=module.RELEASE_SCHEMA, snapshot_id=snapshot["snapshot_id"],
                   snapshot_sha256=file_hash(directory / "snapshot.json"), training_released=True)
    save_json(service.output_dir / "support_fill_release.json", release)
    with pytest.raises(ValueError, match="缺少文件引用"):
        module.verify_support_fill_snapshot(service, snapshot, directory)


def test_data_and_prediction_reports_still_require_window_and_execution_gates(tiny_derivation):
    service, *_ = tiny_derivation
    _, snapshot, directory = derive(tiny_derivation)
    candidate = dict(path=str((directory / "snapshot.json").resolve()), sha256=file_hash(directory / "snapshot.json"))
    parent = snapshot["source_snapshot"]["support_fill_derivation"]["parent_snapshot"]
    parent = {key: parent[key] for key in ("path", "sha256")}
    data = dict(schema="paper1.support_fill_data_audit.1", status="data_passed_prediction_pending",
        data_gate_passed=True, checked_records=450, failures=[], parent_snapshot=parent, candidate_snapshot=candidate)
    data_path = service.output_dir / "synthetic_data_gate.json"
    save_json(data_path, data)
    data_ref = dict(path=str(data_path), sha256=file_hash(data_path))
    prediction = dict(schema="paper1.support_fill_prediction_audit.1", status="passed", passed=True,
        checked_records=450, failures=[], parent_snapshot=parent, candidate_snapshot=candidate, data_audit=data_ref)
    prediction_path = service.output_dir / "synthetic_prediction_gate.json"
    save_json(prediction_path, prediction)
    release = dict(schema=module.RELEASE_SCHEMA, snapshot_id=snapshot["snapshot_id"],
        snapshot_sha256=candidate["sha256"], training_released=True, data_audit=data_ref,
        prediction_audit=dict(path=str(prediction_path), sha256=file_hash(prediction_path)))
    save_json(service.output_dir / "support_fill_release.json", release)
    with pytest.raises(ValueError, match="缺少文件引用"):
        module.verify_support_fill_snapshot(service, snapshot, directory)


def test_archived_fingerprint_reader_matches_existing_scope():
    assert module.archived_route_fingerprints(ROOT) == source_fingerprints()


def test_declared_development_membership_rejects_events_before_csv_reads():
    from test_paper1_support_fill_audit import manifest
    snapshot = manifest()
    assert len(module._development_entries(snapshot, load_only=True)) == 450
    snapshot["records"][0]["metadata"]["group_id"] = "synthetic-event"
    with pytest.raises(ValueError, match="事件或盲测"):
        module._development_entries(snapshot, load_only=True)


def test_driver_preview_creates_no_output(tmp_path, capsys):
    path = ROOT / "scripts/verification/derive_paper1_support_fill.py"
    spec = importlib.util.spec_from_file_location("support_fill_driver", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    output = tmp_path / "preview-only"
    assert driver.main(["--parent-service-output", str(tmp_path / "parent"), "--parent-snapshot-id", "x",
        "--historical-route-source", str(tmp_path / "archive"), "--output-service", str(output)]) == 0
    assert not output.exists()
    assert json.loads(capsys.readouterr().out)["status"] == "planned_not_executed"
