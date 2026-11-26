"""Fail-closed checks for command-position data and execution release evidence."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spindle_power.load_prediction.command_alignment_snapshot import (
    checked_reference, _verify_execution_details, _verify_release, RELEASE_SCHEMA,
)
from spindle_power.load_prediction.common import file_hash
from spindle_power.load_prediction.training_checkpoint import source_fingerprints


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return dict(path=str(path.resolve()), sha256=file_hash(path))


def execution(tmp_path):
    stage = dict(passed=True, initial_weights=dict(exact=True), initial_rng_exact=True,
                 raw_gradients=dict(passed=True), one_adam_weights=dict(passed=True),
                 losses={key: dict(passed=True) for key in ("train_loss", "validation_loss")})
    checkpoint = write(tmp_path / "checkpoint.json", {"placeholder_for_reference_test": True})
    restore = dict(passed=True, bitwise_exact=True, optimizer_updates=2, weights_exact=True,
                   adam_exact=True, rng_exact=True, history_exact=True, continuous=checkpoint, resumed=checkpoint)
    processes = {name: write(tmp_path / (name + ".json"), dict(process_id=index + 1,
                 expected=True, interrupted_stage=name.replace("_interrupted", ""), checkpointed_epoch=1,
                 completed=False, checkpoint=checkpoint)) for index, name in enumerate(
                     ("continuous", "resumed", "main_interrupted", "residual_interrupted"))}
    fixed = write(tmp_path / "outputs.json", dict(passed=True,
                  stages={key: dict(bitwise_equal_outputs=True) for key in ("main", "residual")}))
    names = ("initialization_exact", "fixed_weight_outputs_exact", "weighted_loss_equivalent",
             "accumulated_gradients_equivalent", "adam_update_equivalent", "resume_weights_exact",
             "resume_adam_exact", "resume_rng_exact", "resume_history_exact", "changed_data_resume_rejected")
    return dict(checks={key: True for key in names}, losses_checked=["quantiles", "point"],
        stages_checked=["main", "residual"], updates_per_measurement=3, train_records=400, validation_records=50,
        engineering_weights_reused_for_training=False, scientific_training_started=False, window_policy="prospective_v1",
        execution_config=dict(device="cpu", cpu_threads=10, cpu_interop_threads=1, main_batch_points=32768,
            residual_batch_windows=128, residual_batch_strategy="padded", loss_cache_mode="cpu",
            loss_cache_max_bytes=1073741824, seed=42, learning_rate=.001, gradient_clip=1.),
        numerical_equivalence={key: dict(passed=True, stages={s: deepcopy(stage) for s in ("main", "residual")}) for key in ("Q", "P")},
        recovery={key: dict(passed=True, fresh_process_resume=True, changed_data_resume_rejected=True, **processes,
                    stages={s: deepcopy(restore) for s in ("main", "residual")}) for key in ("Q", "P")},
        fixed_weight_outputs={key: fixed for key in ("quantiles", "point")})


def test_reference_detects_changed_bytes(tmp_path):
    ref = write(tmp_path / "evidence.json", {"passed": True})
    assert checked_reference(ref) == {"passed": True}
    Path(ref["path"]).write_text('{"passed":false}', encoding="utf-8")
    with pytest.raises(ValueError, match="身份"):
        checked_reference(ref)


@pytest.mark.parametrize("ref", [None, {}, {"path": None, "sha256": None}, {"path": "x", "sha256": "short"}])
def test_invalid_references_fail_closed(ref):
    with pytest.raises(ValueError):
        checked_reference(ref)


def test_execution_details_positive_structure(tmp_path):
    _verify_execution_details(execution(tmp_path))


@pytest.mark.parametrize("change", ["missing_check", "subset", "p_missing", "main_missing", "adam", "rng", "resume_once", "old_identity", "cache", "outputs"])
def test_execution_summary_cannot_hide_missing_or_failed_checks(tmp_path, change):
    report = execution(tmp_path)
    if change == "missing_check":
        del report["checks"]["fixed_weight_outputs_exact"]
    elif change == "subset":
        report["train_records"] = 4
    elif change == "p_missing":
        del report["numerical_equivalence"]["P"]
    elif change == "main_missing":
        del report["numerical_equivalence"]["Q"]["stages"]["main"]
    elif change == "adam":
        report["recovery"]["Q"]["stages"]["residual"]["adam_exact"] = False
    elif change == "rng":
        report["numerical_equivalence"]["Q"]["stages"]["residual"]["initial_rng_exact"] = False
    elif change == "resume_once":
        report["recovery"]["P"]["stages"]["main"]["optimizer_updates"] = 1
    elif change == "old_identity":
        report["recovery"]["P"]["changed_data_resume_rejected"] = False
    elif change == "cache":
        report["execution_config"]["loss_cache_mode"] = "none"
    elif change == "outputs":
        Path(report["fixed_weight_outputs"]["point"]["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        _verify_execution_details(report)


def test_minimal_passed_report_never_releases_training(tmp_path):
    snapshot = dict(snapshot_id="identity", records=[])
    snapshot_ref = write(tmp_path / "snapshot.json", snapshot)
    receipt = write(tmp_path / "data.json", {})
    window = dict(schema="paper1.command_alignment_windows.1", status="passed", passed=True,
                  candidate_snapshot=snapshot_ref, source=source_fingerprints())
    write(tmp_path / "command_alignment_release.json", dict(schema=RELEASE_SCHEMA, snapshot_id="identity",
        candidate_snapshot=snapshot_ref, dataset_receipt=receipt, training_released=True,
        window_audit=write(tmp_path / "window.json", window)))
    with pytest.raises(ValueError, match="实际产物"):
        _verify_release(SimpleNamespace(output_dir=tmp_path), snapshot, tmp_path, receipt)
