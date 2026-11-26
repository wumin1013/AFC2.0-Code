"""Synthetic control-flow tests only; no development records or real training."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/verification/run_paper1_qp.py"
SPEC = importlib.util.spec_from_file_location("paper1_qp_runner", PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class Clock:
    def __init__(self):
        self.value = 0.

    def __call__(self):
        return self.value


def ledger(path, clock, *, resume=False):
    return RUNNER.BudgetLedger(path, {"synthetic": True}, resume=resume, clock=clock)


def test_dry_run_does_not_create_outputs_or_import_future_interfaces(tmp_path):
    args = SimpleNamespace(execute=False, output_dir=tmp_path / "absent")
    result = RUNNER.run(args)
    assert result["status"] == "planned_not_executed" and result["training_started"] is False
    assert result["schedule"] == (("Q", 100), ("P", 100), ("Q", 200), ("P", 200))
    assert result["final_selection_epoch"] == 200
    assert result["checkpoint_roles"]["100"] == "pause_checkpoint_trajectory"
    assert not args.output_dir.exists()


def test_file_reference_metadata_is_not_file_identity_but_content_is_checked(tmp_path):
    path = tmp_path / "parent.json"
    path.write_text('{"snapshot_id":"bound-parent"}', encoding="utf-8")
    plain = RUNNER.reference(path)
    rich = dict(**plain, snapshot_id="bound-parent", description="audited parent snapshot")
    assert RUNNER.reference_fields(rich) == plain
    RUNNER.check_reference(rich)
    assert RUNNER.protocol_references(dict(parent_snapshot=rich)) == [plain]
    path.write_text('{"snapshot_id":"changed-parent"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Evidence changed"):
        RUNNER.check_reference(rich)


def test_budget_accounts_for_preparation_resume_and_close(tmp_path):
    clock = Clock()
    first = ledger(tmp_path / "budget.json", clock)
    clock.value = 120.
    first.touch()
    first.close("paused_at_milestone")
    clock.value = 1000.  # Time outside an active session is not silently charged or reset.
    second = ledger(tmp_path / "budget.json", clock, resume=True)
    clock.value = 1030.
    assert second.elapsed() == 150.
    second.close("completed")
    saved = RUNNER.read_json(tmp_path / "budget.json")
    assert saved["spent_seconds"] == 150. and saved["active"] is None
    assert [item["elapsed_seconds"] for item in saved["sessions"]] == [120., 30.]


def test_unresolved_active_session_and_changed_identity_reject_restart(tmp_path):
    clock = Clock()
    value = ledger(tmp_path / "budget.json", clock)
    clock.value = 10.
    value.touch()
    before = (tmp_path / "budget.json").read_bytes()
    with pytest.raises(ValueError, match="Unresolved active"):
        ledger(tmp_path / "budget.json", clock, resume=True)
    assert (tmp_path / "budget.json").read_bytes() == before
    value.close("failed")
    with pytest.raises(ValueError, match="identity"):
        RUNNER.BudgetLedger(tmp_path / "budget.json", {"synthetic": False}, resume=True, clock=clock)


def test_soft_budget_blocks_new_update_but_permits_validation_and_archiving(tmp_path):
    clock = Clock()
    value = ledger(tmp_path / "budget.json", clock)
    control = RUNNER.EpochControl(value, 100, 200, None)
    calls = []
    loss = control.training(lambda back: calls.append(back) or 1.)
    clock.value = RUNNER.SOFT_SECONDS - 1
    assert loss(True) == 1.
    clock.value = RUNNER.SOFT_SECONDS + 1
    assert control.cancel() is False  # The in-flight update may validate and archive.
    control.saved(dict(stage="residual_Q", epoch=1))
    with pytest.raises(RUNNER.BudgetStop, match="soft_budget"):
        loss(True)
    assert calls == [True]
    clock.value = RUNNER.HARD_SECONDS
    with pytest.raises(RUNNER.BudgetStop, match="hard_budget"):
        control.cancel()
    value.close("budget_stopped")


def test_eight_hour_ledger_cannot_be_reset_by_explicit_resume(tmp_path):
    clock = Clock()
    value = ledger(tmp_path / "budget.json", clock)
    clock.value = RUNNER.HARD_SECONDS
    value.close("budget_stopped")
    with pytest.raises(ValueError, match="exhausted"):
        ledger(tmp_path / "budget.json", clock, resume=True)


def test_startup_time_is_part_of_budget(tmp_path):
    clock = Clock()
    clock.value = 8.
    value = RUNNER.BudgetLedger(tmp_path / "budget.json", {"synthetic": True}, resume=False, clock=clock, started=0.)
    assert value.elapsed() == 8.
    value.close("completed")


def tiny_training(module):
    def loss(backward):
        value = module.weight.square().sum()
        if backward:
            value.backward()
        return float(value.detach())
    # Deliberately worsening selection metric: best state differs from last.
    return loss, lambda _back: -float(module.weight.detach().sum())


def test_milestone_resume_uses_last_optimizer_rng_state_not_best(tmp_path):
    import torch
    from spindle_power.load_prediction.training_checkpoint import load_checkpoint
    config = SimpleNamespace(learning_rate=.01, gradient_clip=1., min_delta=0., patience=10, device="cpu")
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.)
    clock = Clock()
    budget = ledger(tmp_path / "budget.json", clock)
    path = tmp_path / "split/residual.pt"
    identity = {"test": "last_not_best", "maximum": 4}
    training, validation = tiny_training(model)
    first = RUNNER.run_epochs_to(model, 4, 2, config, training, validation, "residual", 
        RUNNER.EpochControl(budget, 2, 4, None), path, identity)
    assert first["epoch"] == 2 and first["completed"] is False and first["best_epoch"] == 1
    assert torch.equal(model.weight, first["weights"]["weight"])
    assert not torch.equal(first["weights"]["weight"], first["best_state"]["weight"])
    fixed_two = RUNNER.archived_checkpoint(path)
    second = RUNNER.run_epochs_to(model, 4, 4, config, training, validation, "residual",
        RUNNER.EpochControl(budget, 4, 4, None), path, identity)
    assert second["epoch"] == second["optimizer_updates"] == 4 and second["completed"] is True
    RUNNER.check_reference(fixed_two)  # Advancing latest must not invalidate epoch-2 evidence.
    reference_model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        reference_model.weight.fill_(1.)
    training, validation = tiny_training(reference_model)
    reference = RUNNER.run_epochs_to(reference_model, 4, 4, config, training, validation, "residual",
        RUNNER.EpochControl(budget, 4, 4, None), tmp_path / "whole/residual.pt", identity)
    assert torch.equal(second["weights"]["weight"], reference["weights"]["weight"])
    assert second["optimizer"]["state"][0]["step"] == 4
    assert load_checkpoint(path, identity, resume=True)["completed"] is True
    budget.close("completed")


def test_hard_budget_abort_keeps_last_committed_checkpoint(tmp_path):
    import torch
    from spindle_power.load_prediction.training_checkpoint import load_checkpoint
    config = SimpleNamespace(learning_rate=.01, gradient_clip=1., min_delta=0., patience=10, device="cpu")
    model = torch.nn.Linear(1, 1, bias=False)
    clock = Clock()
    budget = ledger(tmp_path / "budget.json", clock)
    control = RUNNER.EpochControl(budget, 2, 4, None)
    training, validation = tiny_training(model)
    path, identity = tmp_path / "residual.pt", {"synthetic": True}
    calls = 0

    def fail_on_second(backward):
        nonlocal calls
        calls += 1
        if calls == 2:
            clock.value = RUNNER.HARD_SECONDS
            control.cancel()
        return training(backward)

    with pytest.raises(RUNNER.BudgetStop):
        RUNNER.run_epochs_to(model, 4, 2, config, fail_on_second, validation, "residual", control, path, identity)
    saved = load_checkpoint(path, identity, resume=True)
    assert saved["epoch"] == saved["optimizer_updates"] == 1 and saved["completed"] is False
    budget.close("budget_stopped")


def test_ready_validation_keeps_trainable_residual_and_rejects_main_drift():
    import torch
    from spindle_power.load_prediction.neural import state_version
    main, residual = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
    bundle = SimpleNamespace(main=main, residual=residual, main_version=state_version(main))
    RUNNER.ready_for_evaluation(bundle)
    assert all(parameter.requires_grad for parameter in residual.parameters())
    assert bundle.residual_main_version == bundle.main_version
    with torch.no_grad():
        main.weight.add_(1.)
    with pytest.raises(ValueError, match="shared main changed"):
        RUNNER.ready_for_evaluation(bundle)


def test_paired_initialization_has_identical_common_weights_and_separate_objectives():
    import torch
    from dataclasses import replace
    from spindle_power.load_prediction.models import ModelConfig, build_load_models
    from spindle_power.load_prediction.neural import state_version
    from spindle_power.load_prediction.training import TrainingConfig
    model = ModelConfig(coefficient_widths=(4,), hidden=4, attention_heads=1, attention_layers=1,
                        condition_width=2, route="nominal_programmed_residual_v1")
    torch.manual_seed(42)
    origin = build_load_models(model)
    initial = dict(main=origin.main.state_dict(), residual=origin.residual.state_dict())
    shared = dict(weights=initial["main"], main_version=state_version(origin.main), normalization={})
    q = RUNNER.model_bundle(model, TrainingConfig(), initial, shared)
    p = RUNNER.model_bundle(replace(model, residual_output="point"), TrainingConfig(), initial, shared)
    assert all(torch.equal(q.residual.state_dict()[key], value) for key, value in p.residual.state_dict().items())
    assert state_version(q.main) == state_version(p.main)
    assert all(not value.requires_grad for value in p.residual.spread.parameters())
    assert all(value.requires_grad for value in q.residual.spread.parameters())
    assert all(not value.requires_grad for value in q.main.parameters())


def test_final_batch_crossing_hard_budget_does_not_return_to_optimizer(tmp_path):
    clock = Clock()
    budget = ledger(tmp_path / "budget.json", clock)
    control = RUNNER.EpochControl(budget, 100, 200, None)
    def last_batch(_backward):
        clock.value = RUNNER.HARD_SECONDS
        return 0.
    with pytest.raises(RUNNER.BudgetStop, match="hard_budget"):
        control.training(last_batch)(True)
    budget.close("budget_stopped")


def test_supervisor_timeout_kills_only_worker_and_finalizes_budget(tmp_path, monkeypatch):
    import subprocess
    from spindle_power.load_prediction.common import save_json
    args = SimpleNamespace(output_dir=tmp_path / "run", snapshot=tmp_path / "snapshot.json",
                           execution_profile=tmp_path / "profile.json", supplementary_protocol=None, resume=False)
    clock = Clock()
    monkeypatch.setattr(RUNNER.time, "monotonic", clock)
    calls = []
    def timed_out(command, **options):
        calls.append((command, options))
        args.output_dir.mkdir()
        save_json(args.output_dir / "budget.json", dict(schema="paper1.qp_budget.1", fingerprint="synthetic",
            spent_seconds=0., sessions=[], active=dict(session_id="worker", started_at="synthetic")))
        clock.value = RUNNER.HARD_SECONDS
        raise subprocess.TimeoutExpired(command, options["timeout"])
    monkeypatch.setattr(subprocess, "run", timed_out)
    assert RUNNER.supervise(args) == 124
    assert len(calls) == 1 and "--worker" in calls[0][0]
    assert calls[0][1]["timeout"] == RUNNER.HARD_SECONDS
    saved = RUNNER.read_json(args.output_dir / "budget.json")
    assert saved["active"] is None and saved["spent_seconds"] == RUNNER.HARD_SECONDS
    assert saved["sessions"][-1]["exit_code"] == 124
    assert RUNNER.read_json(args.output_dir / "report.json")["status"] == "budget_exhausted"
    assert not (tmp_path / ".run.qp.lock").exists()


def test_supervisor_refuses_unresolved_session_without_spawning(tmp_path, monkeypatch):
    import subprocess
    from spindle_power.load_prediction.common import save_json
    output = tmp_path / "run"
    output.mkdir()
    save_json(output / "budget.json", dict(spent_seconds=1., sessions=[], active={"pid": 1}))
    args = SimpleNamespace(output_dir=output, resume=True)
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Unexpected worker restart"))
    with pytest.raises(ValueError, match="Unresolved active"):
        RUNNER.supervise(args)
    assert not (tmp_path / ".run.qp.lock").exists()


def test_unexpected_worker_crash_keeps_active_unresolved(tmp_path, monkeypatch):
    import subprocess
    from spindle_power.load_prediction.common import save_json
    args = SimpleNamespace(output_dir=tmp_path / "run", snapshot=tmp_path / "snapshot.json",
                           execution_profile=tmp_path / "profile.json", supplementary_protocol=None, resume=False)
    def crashed(*_args, **_kwargs):
        args.output_dir.mkdir()
        save_json(args.output_dir / "budget.json", dict(schema="paper1.qp_budget.1", fingerprint="synthetic",
            spent_seconds=0., sessions=[], active=dict(session_id="unclean-worker")))
        return SimpleNamespace(returncode=99)
    monkeypatch.setattr(subprocess, "run", crashed)
    assert RUNNER.supervise(args) == 99
    assert RUNNER.read_json(args.output_dir / "budget.json")["active"] is not None


@pytest.mark.parametrize("timed_out", [False, True])
def test_supervisor_reconciles_closed_worker_session_without_double_counting(tmp_path, monkeypatch, timed_out):
    import subprocess
    from spindle_power.load_prediction.common import save_json
    output = tmp_path / "run"
    output.mkdir()
    previous = dict(session_id="previous", elapsed_seconds=120., status="paused")
    save_json(output / "budget.json", dict(schema="paper1.qp_budget.1", fingerprint="synthetic",
        spent_seconds=120., sessions=[previous], active=None))
    args = SimpleNamespace(output_dir=output, snapshot=tmp_path / "snapshot.json",
                           execution_profile=tmp_path / "profile.json", supplementary_protocol=None, resume=True)
    clock = Clock()
    monkeypatch.setattr(RUNNER.time, "monotonic", clock)

    def finished(command, **options):
        assert options["timeout"] == RUNNER.HARD_SECONDS - 120.
        elapsed = RUNNER.HARD_SECONDS - 120. if timed_out else 30.
        session = dict(session_id="worker", elapsed_seconds=elapsed - 1., status="completed")
        save_json(output / "budget.json", dict(schema="paper1.qp_budget.1", fingerprint="synthetic",
            spent_seconds=120. + elapsed - 1., sessions=[previous, session], active=None))
        clock.value = elapsed
        if timed_out:
            raise subprocess.TimeoutExpired(command, options["timeout"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", finished)
    assert RUNNER.supervise(args) == (124 if timed_out else 0)
    saved = RUNNER.read_json(output / "budget.json")
    assert len(saved["sessions"]) == 2 and saved["sessions"][0] == previous
    assert saved["spent_seconds"] == (RUNNER.HARD_SECONDS if timed_out else 150.)
    assert saved["sessions"][-1]["elapsed_seconds"] == (RUNNER.HARD_SECONDS - 120. if timed_out else 30.)
    assert saved["active"] is None


@pytest.fixture
def milestone_case(tmp_path):
    from spindle_power.load_prediction.common import save_json
    from spindle_power.load_prediction.training_checkpoint import digest
    output = tmp_path / "run"
    folder = output / "Q/epoch_0100"
    stage = dict(run=dict(snapshot="synthetic"), phase="Q")
    archive = output / "Q/epochs/residual/epoch_0100.pt"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"synthetic checkpoint")
    checkpoint = RUNNER.reference(archive)
    save_json(archive.with_suffix(".pt.json"), dict(fingerprint=digest(stage), sha256=checkpoint["sha256"]))
    artifacts = []
    for name in ("fixed", "best"):
        records = []
        directory = folder / name
        directory.mkdir(parents=True)
        model = directory / "load.pt"
        model.write_bytes(b"synthetic model")
        artifacts.append(RUNNER.reference(model))
        for index in range(50):
            prediction = directory / f"record_{index:02d}.csv"
            prediction.write_text("synthetic\n", encoding="utf-8")
            item = RUNNER.reference(prediction)
            artifacts.append(item)
            records.append(dict(record_id=f"record_{index:02d}", prediction_file=prediction.name, prediction_sha256=item["sha256"]))
        evaluation = directory / "evaluation.json"
        save_json(evaluation, dict(records=records))
        artifacts.append(RUNNER.reference(evaluation))
        diagnostic = directory / "shape_diagnostics.json"
        save_json(diagnostic, dict(records=records))
        artifacts.append(RUNNER.reference(diagnostic))
    receipt = folder / "receipt.json"
    save_json(receipt, dict(schema="paper1.qp_milestone.1", kind="Q", epoch=100, completed=False,
        stage_fingerprint=digest(stage), run_fingerprint=digest(stage["run"]), checkpoint=checkpoint, artifacts=artifacts))
    return output, receipt, stage


def test_completed_milestone_binds_current_run_archive_and_all_prediction_files(milestone_case, tmp_path):
    output, receipt, stage = milestone_case
    assert RUNNER.checked_milestone(receipt, "Q", 100, stage, output)["epoch"] == 100
    with pytest.raises(ValueError, match="another run"):
        RUNNER.checked_milestone(receipt, "Q", 100, dict(run={"snapshot": "another"}, phase="Q"), output)
    with pytest.raises(ValueError, match="escapes"):
        RUNNER.checked_milestone(receipt, "Q", 100, stage, tmp_path / "another_run")
    prediction = receipt.parent / "fixed/record_00.csv"
    prediction.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Evidence changed"):
        RUNNER.checked_milestone(receipt, "Q", 100, stage, output)


@pytest.fixture
def released_case(tmp_path, monkeypatch):
    """Only the release/profile gate is real; population and future config are stubbed."""
    from spindle_power.load_prediction.common import save_json
    from spindle_power.load_prediction import support_fill_snapshot as snapshots, training
    from spindle_power.load_prediction.support_fill import SUPPORT_FILL_RULES_VERSION
    snapshot_path = tmp_path / "service/snapshots/candidate/snapshot.json"
    snapshot_path.parent.mkdir(parents=True)
    parent = tmp_path / "parent.json"
    save_json(parent, {"parent": True, "snapshot_id": "parent-id"})
    snapshot = dict(snapshot_id="candidate", source_snapshot=dict(support_fill_derivation=dict(
        schema="afc.support_fill_derivation.1", rules_version=SUPPORT_FILL_RULES_VERSION,
        implementation_sha256=snapshots.implementation_identity(),
        parent_snapshot=dict(**RUNNER.reference(parent), snapshot_id="parent-id"))))
    save_json(snapshot_path, snapshot)
    snapshot_ref = RUNNER.reference(snapshot_path)
    references = {}
    data = dict(schema="paper1.support_fill_data_audit.1", status="data_passed_prediction_pending",
                data_gate_passed=True, checked_records=450, failures=[], candidate_snapshot=snapshot_ref,
                parent_snapshot=RUNNER.reference(parent))
    for name, value in (
        ("data_audit", data),
        ("window_audit", dict(schema="paper1.prospective_window_audit.1", status="passed", passed=True,
                              candidate_snapshot=snapshot_ref)),
        ("execution_recheck", dict(schema="paper1.load_execution_recheck.1", status="passed", passed=True,
                                   candidate_snapshot=snapshot_ref))):
        path = tmp_path / (name + ".json")
        save_json(path, value)
        references[name] = RUNNER.reference(path)
    prediction = tmp_path / "prediction_audit.json"
    save_json(prediction, dict(schema="paper1.support_fill_prediction_audit.1", status="passed", passed=True,
        checked_records=450, failures=[], candidate_snapshot=snapshot_ref, parent_snapshot=RUNNER.reference(parent),
        data_audit=references["data_audit"]))
    references["prediction_audit"] = RUNNER.reference(prediction)
    release = tmp_path / "service/support_fill_release.json"
    save_json(release, dict(schema="afc.support_fill_release.1", snapshot_id="candidate",
        snapshot_sha256=snapshot_ref["sha256"], training_released=True, **references))
    execution = dict(device="cpu", cpu_threads=10, cpu_interop_threads=1, main_batch_points=32768,
        residual_batch_windows=128, residual_batch_strategy="padded", loss_cache_mode="cpu",
        loss_cache_max_bytes=1073741824, learning_rate=.001, gradient_clip=1., seed=42)
    stage0 = tmp_path / "stage0.json"
    save_json(stage0, dict(schema="afc.paper1_execution_profile.1", status="stage0_engineering_accepted", training_config=execution))
    profile_path = tmp_path / "profile.json"
    source = RUNNER.source_identity()["source"]
    profile = dict(schema="paper1.load_execution_profile.1", status="prospective_rechecked", candidate_snapshot=snapshot_ref,
                   source_fingerprints=source, window_policy="prospective_v1", selection_evidence=references["execution_recheck"],
                   parent_execution_profile=RUNNER.reference(stage0), training_config=execution)
    save_json(profile_path, profile)
    identity = dict(snapshot=snapshot_ref, execution_profile=RUNNER.reference(profile_path), source=source)
    monkeypatch.setattr(snapshots, "_snapshot", lambda path: snapshot if Path(path) == snapshot_path else RUNNER.read_json(path))
    monkeypatch.setattr(snapshots, "_development_entries", lambda *_args, **_kwargs: [])
    class FutureConfig(SimpleNamespace):
        def validate(self):
            pass
    monkeypatch.setattr(training, "TrainingConfig", FutureConfig)
    budget = ledger(tmp_path / "budget.json", Clock())
    return identity, budget, release, profile_path


def test_four_release_receipts_and_new_profile_are_required(released_case):
    identity, budget, _, _ = released_case
    _, _, config, references = RUNNER.preflight(None, identity, budget)
    assert config.main_epochs == 100 and config.residual_epochs == 200 and config.patience == 201
    assert config.residual_window_policy == "prospective_v1"
    assert {"data_audit", "prediction_audit", "window_audit", "execution_recheck"} <= set(references)
    budget.close("completed")


def test_preflight_rejects_parent_id_not_matching_the_bound_snapshot(released_case, monkeypatch):
    from spindle_power.load_prediction import support_fill_snapshot as snapshots
    identity, budget, _, _ = released_case
    original = snapshots._snapshot
    def changed_parent(path):
        value = original(path)
        return value if str(Path(path).resolve()) == identity["snapshot"]["path"] else {**value, "snapshot_id": "wrong-parent-id"}
    monkeypatch.setattr(snapshots, "_snapshot", changed_parent)
    with pytest.raises(ValueError, match="parent snapshot ID"):
        RUNNER.preflight(None, identity, budget)
    budget.close("failed")


@pytest.mark.parametrize("missing", ["data_audit", "prediction_audit", "window_audit", "execution_recheck"])
def test_any_missing_release_gate_fails_closed(released_case, missing):
    from spindle_power.load_prediction.common import save_json
    identity, budget, release, _ = released_case
    value = RUNNER.read_json(release)
    del value[missing]
    save_json(release, value)
    with pytest.raises(ValueError):
        RUNNER.preflight(None, identity, budget)
    budget.close("failed")


@pytest.mark.parametrize("change", ["source", "legacy_profile", "selection", "window"])
def test_stage0_or_mismatched_recheck_cannot_release_new_windows(released_case, change):
    from spindle_power.load_prediction.common import save_json
    identity, budget, _, profile_path = released_case
    profile = RUNNER.read_json(profile_path)
    key = {"source": "source_fingerprints", "legacy_profile": "status", "selection": "selection_evidence", "window": "window_policy"}[change]
    profile[key] = {} if change in ("source", "selection") else "legacy"
    save_json(profile_path, profile)
    identity["execution_profile"] = RUNNER.reference(profile_path)
    with pytest.raises(ValueError):
        RUNNER.preflight(None, identity, budget)
    budget.close("failed")


def test_supplementary_not_ready_protocol_is_frozen_without_becoming_selection_metric(tmp_path):
    from spindle_power.load_prediction.common import save_json
    support = tmp_path / "support.json"
    save_json(support, {"display_only": True})
    protocol = dict(status="not_ready", supporting_file=RUNNER.reference(support))
    references = RUNNER.protocol_references(protocol)
    assert references == [RUNNER.reference(support)]
    save_json(support, {"display_only": False})
    with pytest.raises(ValueError, match="Evidence changed"):
        RUNNER.protocol_references(protocol)


def test_complete_macro_averages_record_rmse_and_preserves_missing_records():
    rows = [dict(record_id="a", round_id=1, feature="first", rmse_w=1.),
            dict(record_id="b", round_id=1, feature="first", rmse_w=9.),
            dict(record_id="c", round_id=1, feature="second", rmse_w=20.)]
    metric = RUNNER.complete_macro(rows, ["rmse_w"])["rmse_w"]
    assert metric["value"] == 12.5  # mean(mean(1, 9), 20), never sqrt(weighted MSE).
    rows[1]["rmse_w"] = None
    metric = RUNNER.complete_macro(rows, ["rmse_w"])["rmse_w"]
    assert metric["value"] is None and metric["missing_record_ids"] == ["b"]
    assert metric["expected_records"] == 3 and metric["available_records"] == 2


@pytest.mark.parametrize("parent_metadata", [{}, {"snapshot_id": "audited_parent_id"}])
def test_supplementary_protocol_binds_f1_split_parent_data_gate_and_predeclared_direction(tmp_path, parent_metadata):
    from copy import deepcopy
    from spindle_power.load_prediction.common import save_json
    entries = [dict(file=f"R{round_id:02d}_{feature}.csv", sha256="synthetic",
                    metadata=dict(record_id=f"R{round_id:02d}_{feature}", round_id=round_id, feature=str(feature)))
               for round_id in range(1, 91) for feature in range(5)]
    train, validation = RUNNER.partition(entries)
    parent = dict(path=str(tmp_path / "parent.json"), sha256="parent")
    snapshot = tmp_path / "snapshot.json"
    save_json(snapshot, dict(source_snapshot=dict(support_fill_derivation=dict(parent_snapshot={**parent, **parent_metadata}))))
    data_gate = dict(path=str(tmp_path / "data_gate.json"), sha256="gate")
    protocol_path = tmp_path / "protocol.json"
    value = dict(schema="paper1.qp_supplementary_protocol.1", fold="F1", snapshot=RUNNER.reference(snapshot),
        parent_snapshot=parent, data_audit=data_gate, status="not_ready", baseline="Q", candidate="P",
        training_record_ids=[row["metadata"]["record_id"] for row in train],
        validation_record_ids=[row["metadata"]["record_id"] for row in validation], reference_attachments={},
        shape_protocol=None, unavailable_reasons=["support_not_ready"], screening_thresholds=RUNNER.PROTOCOL_THRESHOLDS,
        **RUNNER.REPORTING_POLICY)

    def check(current):
        save_json(protocol_path, current)
        identity = dict(snapshot=RUNNER.reference(snapshot), supplementary_protocol=RUNNER.reference(protocol_path))
        return RUNNER.supplementary_protocol(identity, entries, dict(data_audit=data_gate))

    assert check(value)["status"] == "not_ready"
    for changed in (dict(baseline="P", candidate="Q"), dict(parent_snapshot={}), dict(data_audit={}),
                    dict(validation_record_ids=value["validation_record_ids"][:-1]), dict(status="ready"),
                    dict(final_selection_epoch=100), dict(best_role="primary")):
        with pytest.raises(ValueError):
            check({**deepcopy(value), **changed})


@pytest.fixture
def shape_case(tmp_path):
    import numpy as np
    import pandas as pd
    from paper1_prediction_metrics import ShapeProtocol
    x = np.arange(101) * .1
    frame = pd.DataFrame(dict(line=1, execution_instance=0, segment_mm=x, path_mm=x, sequence_id=0,
        power_cut_w=10. + 30. * np.exp(-((x - 5.) / .5) ** 2), valid_main=x < 3, valid_residual=True))
    samples = [SimpleNamespace(frame=frame.copy(), metadata=dict(record_id=f"record_{index:02d}",
        round_id=index // 5 + 1, feature=str(index % 5))) for index in range(50)]
    support = frame[["line", "execution_instance", "segment_mm", "path_mm"]].copy()
    support["grid_index"], support["reference_support_segment_id"] = np.arange(len(frame)), 0
    support_path = tmp_path / "support.csv"
    support.to_csv(support_path, index=False)
    frozen = dict(status="ready", shape_protocol=ShapeProtocol(.2, .1, 100, minimum_sustained_span_mm=.5).to_dict(),
                  unavailable_reasons=[], reference_attachments={sample.metadata["record_id"]: RUNNER.reference(support_path) for sample in samples})
    budget = ledger(tmp_path / "budget.json", Clock())

    def export(kind, error):
        folder = tmp_path / kind / "epoch_0100/fixed"
        folder.mkdir(parents=True)
        rows = []
        for sample in samples:
            prediction = sample.frame.copy()
            prediction["cutting_power_w"] = prediction.power_cut_w + error
            prediction["scored"] = prediction["prediction_available"] = True
            path = folder / (sample.metadata["record_id"] + ".csv")
            prediction.to_csv(path, index=False)
            rows.append(dict(record_id=sample.metadata["record_id"], prediction_file=path.name,
                             prediction_sha256=RUNNER.reference(path)["sha256"]))
        return folder, dict(records=rows)

    return tmp_path, samples, frozen, budget, export


def test_shape_export_keeps_all_records_and_never_changes_model_frames(shape_case):
    output, samples, protocol, budget, export = shape_case
    folder, evaluation = export("Q", 1.)
    result = RUNNER.shape_diagnostics(folder, evaluation, samples, protocol, budget)
    assert result["status"] == "ready" and len(result["records"]) == 50
    assert len(result["metrics_by_feature"]) == 5 and len(result["metrics_by_round"]) == 10
    assert result["macro_metrics"]["rmse_w"]["value"] == pytest.approx(1.)
    assert all(row["scored_points"] == 101 for row in result["records"])
    assert all("reference_support_segment_id" not in sample.frame for sample in samples)
    unavailable = dict(status="not_ready", shape_protocol=None, reference_attachments={}, unavailable_reasons=["missing_support"])
    pending = RUNNER.shape_diagnostics(folder, evaluation, samples, unavailable, budget)
    assert pending["status"] == "not_ready" and len(pending["records"]) == 50
    assert pending["macro_metrics"]["rmse_w"]["value"] == result["macro_metrics"]["rmse_w"]["value"]
    assert pending["macro_metrics"]["peak_amplitude_absolute_error_w"]["value"] is None
    assert pending["screening_passed"] is None
    with pytest.raises(ValueError, match="50 distinct"):
        RUNNER.shape_diagnostics(folder, dict(records=evaluation["records"][:-1]), samples, protocol, budget)
    budget.close("completed")


@pytest.mark.parametrize("candidate_error,expected", [(.5, "passed"), (1.1, "failed")])
def test_fixed_complete_scientific_screen_selects_p_or_retains_q(shape_case, candidate_error, expected):
    from spindle_power.load_prediction.common import save_json
    output, samples, protocol, budget, export = shape_case
    for kind, error in (("Q", 1.), ("P", candidate_error)):
        folder, evaluation = export(kind, error)
        diagnostic = RUNNER.shape_diagnostics(folder, evaluation, samples, protocol, budget)
        save_json(folder / "shape_diagnostics.json", diagnostic)
    result = RUNNER.scientific_screening(output, 100, "fixed", budget)
    assert result["status"] == expected and result["passed"] is (expected == "passed")
    assert result["decision"] == ("P_passed_development_screen" if expected == "passed" else "retain_Q")
    assert len(result["peak_comparisons"]) == 50 and result["role"] == "primary_fixed_epoch"
    budget.close("completed")


def test_missing_difficult_peak_cannot_pass_complete_macro_even_with_better_conditional_error(shape_case):
    from spindle_power.load_prediction.common import save_json
    output, samples, protocol, budget, export = shape_case
    for kind, error in (("Q", 1.), ("P", .5)):
        folder, evaluation = export(kind, error)
        diagnostic = RUNNER.shape_diagnostics(folder, evaluation, samples, protocol, budget)
        if kind == "P":
            record = diagnostic["records"][0]
            record["missed_reference_peak_ids"] = [row["peak_id"] for row in record["reference_peak_catalog"]]
            record["peak_matches"] = []
            record["peak_misses"] = len(record["missed_reference_peak_ids"])
            # A stale/conditional macro number must not override reference-set completeness.
        save_json(folder / "shape_diagnostics.json", diagnostic)
    result = RUNNER.scientific_screening(output, 100, "fixed", budget)
    assert result["status"] == "not_ready" and result["decision"] == "undetermined"
    assert result["gates"]["peak_amplitude_absolute_error_w"]["status"] == "not_ready"
    assert len(result["peak_comparisons"]) == 50
    budget.close("completed")


@pytest.mark.parametrize("zero_peak_records,expected", [(1, "passed"), (50, "not_ready")])
def test_peak_applicability_uses_reference_only_and_keeps_all_50_records(shape_case, zero_peak_records, expected):
    from spindle_power.load_prediction.common import save_json
    output, samples, protocol, budget, export = shape_case
    for sample in samples[:zero_peak_records]:
        sample.frame["power_cut_w"] = 10.
    for kind, error in (("Q", 1.), ("P", .5)):
        folder, evaluation = export(kind, error)
        diagnostic = RUNNER.shape_diagnostics(folder, evaluation, samples, protocol, budget)
        save_json(folder / "shape_diagnostics.json", diagnostic)
    result = RUNNER.scientific_screening(output, 100, "fixed", budget)
    assert result["status"] == expected
    scope = result["peak_amplitude_applicability"]
    assert len(scope["all_record_ids"]) == len(result["peak_comparisons"]) == 50
    assert len(scope["not_applicable_record_ids"]) == zero_peak_records
    assert len(scope["reference_applicable_record_ids"]) == 50 - zero_peak_records
    assert scope["not_ready_record_ids"] == []
    assert result["gates"]["rmse_w"]["status"] == "passed"
    assert result["gates"]["peak_amplitude_absolute_error_w"]["status"] == expected
    budget.close("completed")


@pytest.mark.parametrize("no_sustained_records,expected", [(1, "passed"), (50, "not_ready")])
def test_sustained_applicability_uses_fixed_reference_segments_and_keeps_50_records(shape_case, no_sustained_records, expected):
    from spindle_power.load_prediction.common import save_json
    output, samples, protocol, budget, export = shape_case
    for sample in samples[:no_sustained_records]:
        sample.frame["valid_main"] = False
    for kind, error in (("Q", 1.), ("P", .5)):
        folder, evaluation = export(kind, error)
        diagnostic = RUNNER.shape_diagnostics(folder, evaluation, samples, protocol, budget)
        save_json(folder / "shape_diagnostics.json", diagnostic)
    result = RUNNER.scientific_screening(output, 100, "fixed", budget)
    assert result["status"] == expected
    scope = result["sustained_bias_applicability"]
    assert len(scope["all_record_ids"]) == len(result["sustained_comparisons"]) == 50
    assert len(scope["not_applicable_record_ids"]) == no_sustained_records
    assert len(scope["reference_applicable_record_ids"]) == 50 - no_sustained_records
    assert scope["not_ready_record_ids"] == []
    assert result["gates"]["rmse_w"]["status"] == result["gates"]["dynamic_rmse_w"]["status"] == "passed"
    assert result["gates"]["sustained_absolute_bias_w"]["status"] == expected
    budget.close("completed")


@pytest.mark.parametrize("change", ["missing_value", "segment_definition"])
def test_sustained_applicable_evidence_cannot_be_missing_or_change_reference_segments(shape_case, change):
    from spindle_power.load_prediction.common import save_json
    output, samples, protocol, budget, export = shape_case
    for kind, error in (("Q", 1.), ("P", .5)):
        folder, evaluation = export(kind, error)
        diagnostic = RUNNER.shape_diagnostics(folder, evaluation, samples, protocol, budget)
        if kind == "P":
            record = diagnostic["records"][0]
            if change == "missing_value":
                record["sustained_absolute_bias_w"] = None
            else:
                record["sustained_segments"][0]["span_mm"] += .1
        save_json(folder / "shape_diagnostics.json", diagnostic)
    if change == "segment_definition":
        with pytest.raises(ValueError, match="segment definitions differ"):
            RUNNER.scientific_screening(output, 100, "fixed", budget)
    else:
        result = RUNNER.scientific_screening(output, 100, "fixed", budget)
        assert result["status"] == "not_ready" and result["decision"] == "undetermined"
        scope = result["sustained_bias_applicability"]
        assert len(scope["reference_applicable_record_ids"]) == 50
        assert scope["not_ready_record_ids"] == ["record_00"]
        assert scope["macro_metrics"]["candidate_w"]["value"] is None
        assert result["gates"]["sustained_absolute_bias_w"]["status"] == "not_ready"
    budget.close("completed")


@pytest.mark.parametrize("status,decision", [("passed", "P_passed_development_screen"),
                                            ("failed", "retain_Q"), ("not_ready", "undetermined")])
def test_screening_summary_uses_only_completed_200_fixed_despite_better_100_or_best(status, decision):
    comparisons = {"100": {"fixed": dict(epoch=100, export="fixed", status="passed")},
                   "200": {"fixed": dict(epoch=200, export="fixed", status=status),
                           "best": dict(epoch=200, export="best", status="passed")}}
    result = RUNNER.screening_summary(comparisons, completed=True)
    assert result["status"] == status and result["decision"] == decision
    assert result["selected_comparison"] == "200.fixed" and result["final_selection_epoch"] == 200
    assert result["fixed_primary_reporting_epochs"] == [100, 200]
    assert result["automatic_replacement"] is False


@pytest.mark.parametrize("has_200,completed", [(False, False), (False, True), (True, False)])
def test_screening_summary_never_uses_100_or_an_incomplete_plan_as_final(has_200, completed):
    comparisons = {"100": {"fixed": dict(epoch=100, export="fixed", status="passed")}}
    if has_200:
        comparisons["200"] = {"fixed": dict(epoch=200, export="fixed", status="passed")}
    result = RUNNER.screening_summary(comparisons, completed=completed)
    assert result["status"] == "not_ready" and result["decision"] == "undetermined"
    assert result["selected_comparison"] is None and result["passed"] is False


def test_screening_summary_rejects_best_or_100_relabelled_as_fixed_200():
    for endpoint in (dict(epoch=100, export="fixed", status="passed"), dict(epoch=200, export="best", status="passed")):
        with pytest.raises(ValueError, match="200-update fixed endpoint"):
            RUNNER.screening_summary({"200": {"fixed": endpoint}}, completed=True)
