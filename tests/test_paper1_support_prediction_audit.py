"""Coverage regression tests for the 450-record audit entry point."""
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import numpy as np
import pandas as pd
import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/verification/audit_paper1_support_predictions.py"
SPEC = importlib.util.spec_from_file_location("support_prediction_audit", PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)
sys.path.insert(0, str(PATH.parent))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def sha(path):
    from spindle_power.load_prediction.common import file_hash
    return file_hash(path)


def frames():
    frame = pd.DataFrame(dict(line=[1, 1, 1], execution_instance=[0, 0, 0],
        segment_mm=[0., 1., 2.], path_mm=[0., 1., 2.], power_cut_w=[10., 20., np.nan],
        prediction_available=[True, True, False], scored=[True, True, False],
        cutting_power_w=[11., 18., np.nan]))
    return frame.copy(), frame.copy(), frame.copy()


def test_added_prediction_keeps_entire_old_domain():
    parent, merged, archive = frames()
    merged.loc[2, ["prediction_available", "cutting_power_w"]] = [True, 25.]
    result = AUDIT.compare_archive(parent, merged, archive)
    assert result["original_predicted_points"] == 2
    assert result["original_scored_points"] == 2
    assert result["historical_archive_max_abs_difference_w"] == 0.


@pytest.mark.parametrize("change", ["score", "label", "grid", "loss", "nonfinite"])
def test_archive_check_rejects_domain_or_identity_change(change):
    parent, merged, archive = frames()
    if change == "score":
        merged.loc[0, "scored"] = False
    elif change == "label":
        merged.loc[0, "power_cut_w"] = 100.
    elif change == "grid":
        merged.loc[0, "segment_mm"] = .5
    elif change == "loss":
        merged.loc[0, "prediction_available"] = False
    else:
        merged.loc[0, "cutting_power_w"] = np.nan
    with pytest.raises((ValueError, AssertionError)):
        AUDIT.compare_archive(parent, merged, archive)


def archived_run(tmp_path):
    from spindle_power.load_prediction.experiment_protocol import PROGRAM_FEATURES, protocol_folds
    parents, sources = {}, {}
    for fold in protocol_folds():
        directory = tmp_path / "folds" / "C64" / fold["fold"]
        location = directory / "nominal_no_memory"
        location.mkdir(parents=True)
        model = directory / "load.pt"
        model.write_bytes(f"synthetic model {fold['fold']}".encode())
        history = directory / "load_history.json"
        write_json(history, dict(rounds=fold["train_rounds"]))
        rows = []
        for round_id in fold["validation_rounds"]:
            for feature in PROGRAM_FEATURES:
                rid = f"R{round_id:02d}_{feature}"
                meta = dict(record_id=rid, round_id=round_id, feature=feature)
                parents[rid] = dict(metadata=meta)
                prediction = location / f"record_{len(rows):04d}.csv"
                frames()[2].to_csv(prediction, index=False)
                sources[prediction.resolve()] = sha(prediction)
                rows.append(dict(**meta, feed_mode="programmed", prediction_file=prediction.name,
                                 prediction_sha256=sha(prediction), metric_parameters={}))
        evaluation = location / "evaluation.json"
        write_json(evaluation, dict(records=rows))
        for path in (model, history, evaluation):
            sources[path.resolve()] = sha(path)
    return parents, sources


def seal_chain(directory, sources, snapshot_id="parent-snapshot"):
    audit = directory / "frozen_exports/audit.json"
    write_json(audit, dict(schema="paper1.initial_cpu_audit.1", status="k0_accepted", snapshot_id=snapshot_id,
                          completed_folds=[f"C64:F{i}" for i in range(1, 10)], pending_folds=[],
                          source_artifacts={str(path): digest for path, digest in sources.items()}))
    seal = directory / "k0_seal.json"
    write_json(seal, dict(schema="paper1.k0_seal.1", status="accepted", snapshot_id=snapshot_id,
                         initial_training_complete=True, initial_evaluation_complete=True,
                         frozen_export_files={"frozen_exports/audit.json": sha(audit)}))
    return seal, audit


def test_each_development_record_uses_its_original_heldout_model(tmp_path):
    parents, sources = archived_run(tmp_path / "run")
    seal, _ = seal_chain(tmp_path / "sealed", sources)
    references, frozen_sources = AUDIT.frozen_archive_bindings(seal, "parent-snapshot")
    mapped = AUDIT.archived_fold_records(tmp_path / "run", parents, frozen_sources)
    assert len(mapped) == 450
    assert len({item["model"] for item in mapped.values()}) == 9
    assert set(references) == {"frozen_seal", "frozen_audit"}
    assert len({ref["path"] for entry in mapped.values() for ref in entry["bindings"].values()}) == 477


def test_missing_archive_record_does_not_shrink_the_evaluation(tmp_path):
    parents, sources = archived_run(tmp_path)
    path = tmp_path / "folds/C64/F1/nominal_no_memory/evaluation.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["records"].pop()
    path.write_text(json.dumps(data), encoding="utf-8")
    sources[path.resolve()] = sha(path)
    with pytest.raises(ValueError, match="50 records"):
        AUDIT.archived_fold_records(tmp_path, parents, sources)


@pytest.mark.parametrize("kind", ["model", "history", "evaluation"])
def test_changed_frozen_fold_artifact_is_rejected(tmp_path, kind):
    parents, sources = archived_run(tmp_path)
    name = {"model": "load.pt", "history": "load_history.json", "evaluation": "nominal_no_memory/evaluation.json"}[kind]
    path = tmp_path / "folds/C64/F1" / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        AUDIT.archived_fold_records(tmp_path, parents, sources)


def test_other_fold_model_cannot_be_substituted(tmp_path):
    parents, sources = archived_run(tmp_path)
    (tmp_path / "folds/C64/F1/load.pt").write_bytes((tmp_path / "folds/C64/F2/load.pt").read_bytes())
    with pytest.raises(ValueError, match="hash mismatch"):
        AUDIT.archived_fold_records(tmp_path, parents, sources)


def test_training_rounds_must_match_the_original_fold(tmp_path):
    parents, sources = archived_run(tmp_path)
    path = tmp_path / "folds/C64/F1/load_history.json"
    write_json(path, dict(rounds=list(range(1, 91))))
    sources[path.resolve()] = sha(path)
    with pytest.raises(ValueError, match="training rounds"):
        AUDIT.archived_fold_records(tmp_path, parents, sources)


@pytest.mark.parametrize("change", ["missing_source", "prediction_hash", "escape"])
def test_prediction_requires_its_own_sealed_binding(tmp_path, change):
    parents, sources = archived_run(tmp_path)
    path = tmp_path / "folds/C64/F1/nominal_no_memory/evaluation.json"
    data = AUDIT.read_json(path)
    if change == "missing_source":
        del sources[(path.parent / data["records"][0]["prediction_file"]).resolve()]
    else:
        data["records"][0]["prediction_sha256" if change == "prediction_hash" else "prediction_file"] = (
            "0" * 64 if change == "prediction_hash" else "../../outside.csv")
        write_json(path, data)
        sources[path.resolve()] = sha(path)
    with pytest.raises(ValueError, match="absent|sealed binding|escapes"):
        AUDIT.archived_fold_records(tmp_path, parents, sources)


@pytest.mark.parametrize("change", ["seal_status", "snapshot", "audit_bytes", "audit_status", "missing_fold"])
def test_unaccepted_or_changed_seal_chain_is_rejected(tmp_path, change):
    artifact = tmp_path / "source"
    artifact.write_bytes(b"synthetic")
    seal, audit = seal_chain(tmp_path / "sealed", {artifact: sha(artifact)})
    if change in ("seal_status", "snapshot"):
        value = AUDIT.read_json(seal)
        value["status" if change == "seal_status" else "snapshot_id"] = "other"
        write_json(seal, value)
    else:
        value = AUDIT.read_json(audit)
        if change == "missing_fold":
            value["completed_folds"].pop()
        else:
            value["status"] = "failed"
        write_json(audit, value)
        if change != "audit_bytes":
            value = AUDIT.read_json(seal)
            value["frozen_export_files"]["frozen_exports/audit.json"] = sha(audit)
            write_json(seal, value)
    with pytest.raises(ValueError):
        AUDIT.frozen_archive_bindings(seal, "parent-snapshot")


@pytest.fixture
def driver_case(tmp_path, monkeypatch):
    """One synthetic record replaces population validation; no model is loaded."""
    import audit_paper1_support_fill as data_audit
    from spindle_power.load_prediction import load_device, models, support_prediction, training_checkpoint
    import torch

    parents, sources = archived_run(tmp_path / "run")
    assignments = AUDIT.archived_fold_records(tmp_path / "run", parents, sources)
    rid = next(iter(assignments))
    parent = tmp_path / "parent/snapshot.json"
    candidate = tmp_path / "candidate/snapshot.json"
    write_json(parent, dict(snapshot_id="parent-snapshot"))
    write_json(candidate, dict(snapshot_id="candidate-snapshot"))
    seal, _ = seal_chain(tmp_path / "sealed", sources)
    data = tmp_path / "data_audit.json"
    write_json(data, dict(schema="paper1.support_fill_data_audit.1", status="data_passed_prediction_pending",
                         data_gate_passed=True, checked_records=450, failures=[],
                         parent_snapshot=dict(sha256=sha(parent)), candidate_snapshot=dict(sha256=sha(candidate))))
    evidence = tmp_path / "stage0.json"
    write_json(evidence, dict(status="stage0_passed"))
    profile = tmp_path / "profile.json"
    write_json(profile, dict(schema="afc.paper1_execution_profile.1", status="stage0_engineering_accepted",
                            selection_evidence=dict(path=str(evidence), sha256=sha(evidence)),
                            training_config=dict(device="cpu", cpu_threads=1, cpu_interop_threads=1,
                                                 residual_batch_windows=128, residual_batch_strategy="padded")))
    entry = parents[rid]
    entry["sha256"] = "synthetic-sample-sha"
    monkeypatch.setattr(data_audit, "read_snapshot", AUDIT.read_json)
    monkeypatch.setattr(data_audit, "development_records", lambda *_: {rid: entry})
    monkeypatch.setattr(data_audit, "checked_frame", lambda *_: frames()[0])
    monkeypatch.setattr(AUDIT, "archived_fold_records", lambda *_: {rid: assignments[rid]})
    monkeypatch.setattr(torch, "set_num_threads", lambda *_: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda *_: None)
    monkeypatch.setattr(load_device, "configure_load_device", lambda *_: ("cpu", {"device": "cpu"}))
    bundle = SimpleNamespace(execution_batch_windows=lambda: 128, execution_batch_strategy=lambda: "padded",
                             current_version=lambda: "synthetic-model")
    monkeypatch.setattr(models, "load_models", lambda *_args, **_kwargs: bundle)
    monkeypatch.setattr(training_checkpoint, "source_fingerprints", lambda: {"module.py": "stable"})
    after_prediction = []

    def predict(*_args, output_dir, **_kwargs):
        output_dir.mkdir(parents=True)
        frames()[1].to_csv(output_dir / "merged.csv", index=False)
        result = dict(passed=True, prediction_file="merged.csv", prediction_sha256=sha(output_dir / "merged.csv"),
                      coverage=dict(nc_grid_points=3))
        write_json(output_dir / "report.json", result)
        for callback in after_prediction:
            callback()
        return result

    monkeypatch.setattr(support_prediction, "evaluate_support_fill_record", predict)
    args = SimpleNamespace(parent_snapshot=parent, candidate_snapshot=candidate, data_audit=data,
                           execution_profile=profile, frozen_run=tmp_path / "run", frozen_seal=seal,
                           output_dir=tmp_path / "output")
    return args, after_prediction, assignments[rid], training_checkpoint, support_prediction


@pytest.mark.parametrize("change", ["source", "seal", "frozen_audit", "script", "prediction"])
def test_final_integrity_failure_publishes_failed_state_without_inflating_record_count(driver_case, monkeypatch, change):
    args, callbacks, assignment, training_checkpoint, _ = driver_case
    if change == "source":
        fingerprint = {"module.py": "stable"}
        monkeypatch.setattr(training_checkpoint, "source_fingerprints", lambda: dict(fingerprint))
        callbacks.append(lambda: fingerprint.update({"module.py": "changed"}))
    elif change == "script":
        from spindle_power.load_prediction import common
        original_hash = common.file_hash
        callbacks.append(lambda: monkeypatch.setattr(common, "file_hash",
            lambda path: "0" * 64 if Path(path).resolve() == PATH.resolve() else original_hash(path)))
    else:
        path = (args.frozen_seal if change == "seal" else args.frozen_seal.parent / "frozen_exports/audit.json"
                if change == "frozen_audit" else Path(assignment["bindings"]["prediction"]["path"]))
        callbacks.append(lambda: path.write_bytes(path.read_bytes() + b" "))
    assert AUDIT.run(args) == 1
    report = AUDIT.read_json(args.output_dir / "report.json")
    state = AUDIT.read_json(args.output_dir / "state.json")
    for value in (report, state):
        assert value["status"] == "failed" and value["passed"] is False
        assert value["checked_records"] == value["attempted_records"] == 1
        assert value["unattempted_records"] == 449
        assert value["failed_records"] == 0 and value["integrity_failures"] == 1
    assert report["failures"][0]["scope"] == "integrity"
    assert set(report["audit_scripts"]) == {"audit_paper1_support_predictions.py", "audit_paper1_support_fill.py"}
    assert report["frozen_seal"]["path"] == str(args.frozen_seal)


def test_record_failure_remains_one_attempt_and_stops(driver_case, monkeypatch):
    args, _, _, _, support_prediction = driver_case
    def fail(*_args, **_kwargs):
        raise ValueError("synthetic prediction failure")
    monkeypatch.setattr(support_prediction, "evaluate_support_fill_record", fail)
    assert AUDIT.run(args) == 1
    report = AUDIT.read_json(args.output_dir / "report.json")
    assert report["checked_records"] == 0 and report["attempted_records"] == 1
    assert report["failed_records"] == 1 and report["integrity_failures"] == 0
    assert report["unattempted_records"] == 449 and report["failures"][0]["scope"] == "record"


def test_cli_requires_frozen_seal(monkeypatch):
    names = ("parent-snapshot", "candidate-snapshot", "data-audit", "execution-profile", "frozen-run", "output-dir")
    monkeypatch.setattr(sys, "argv", [str(PATH)] + [value for name in names for value in ("--" + name, "dummy")])
    with pytest.raises(SystemExit) as error:
        AUDIT.main()
    assert error.value.code == 2
