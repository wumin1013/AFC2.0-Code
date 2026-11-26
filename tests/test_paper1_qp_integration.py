"""Synthetic runner integration; prospective source is required, never real data."""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from test_paper1_qp import RUNNER, Clock, ledger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture
def prospective_case(tmp_path, monkeypatch):
    from spindle_power.load_prediction.training import TrainingConfig
    if "residual_window_policy" not in TrainingConfig.__dataclass_fields__:
        pytest.skip("Formal prospective interfaces are not released yet; run in the isolated candidate")
    from spindle_power.load_prediction.condition import fit_condition_spec
    from spindle_power.load_prediction.models import ModelConfig
    from verify_unified_load_prediction import SMALL_CANDIDATE, synthetic_records
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    _, train = synthetic_records((1, 2), points=8)
    for sample in train:
        sample.frame.loc[[2, 3], "valid_residual"] = False
        sample.frame.loc[2, "power_cut_w"] += 200.
        sample.frame["nc_sequence_id"] = 0
    validation = []
    for index in range(50):
        sample = deepcopy(train[index % len(train)])
        sample.metadata["record_id"] = f"synthetic_validation_{index:02d}"
        sample.metadata["round_id"] = index // 5 + 1
        sample.frame["power_cut_w"] += index * 10.
        validation.append(sample)
    entries = [dict(metadata=sample.metadata) for sample in train]
    by_id = {sample.metadata["record_id"]: sample for sample in train}
    monkeypatch.setattr(RUNNER, "read_sample", lambda _directory, entry, _ledger: deepcopy(by_id[entry["metadata"]["record_id"]]))
    config = TrainingConfig(main_epochs=100, residual_epochs=200, patience=201, min_delta=0.,
                            residual_window_policy="prospective_v1", residual_batch_windows=128,
                            residual_batch_strategy="padded", loss_cache_mode="cpu")
    model = ModelConfig(**{**SMALL_CANDIDATE["load_model"], "route": "nominal_programmed_residual_v1",
                          "condition_spec": fit_condition_spec([sample.metadata for sample in train])})
    budget = ledger(tmp_path / "budget.json", Clock())
    common = tmp_path / "shared"
    common.mkdir()
    identity = dict(synthetic=True, model=asdict(model), config=asdict(config))
    initial = RUNNER.initial_state(common, identity, model, config)
    shared = RUNNER.shared_main(common, identity, model, config, initial, entries, validation,
                               tmp_path, budget, None)
    yield dict(directory=tmp_path, common=common, identity=identity, config=config, model=model, initial=initial,
               shared=shared, train=train, validation=validation, budget=budget)
    budget.close("completed")
    torch.set_num_threads(previous)


def test_shared_main_100_best_and_normalization_use_only_training_residual_domain(prospective_case):
    from spindle_power.load_prediction.samples import SampleSet, load_input_view
    from spindle_power.load_prediction.neural import prospective_residual_context, state_version
    from spindle_power.load_prediction.training import _main_values
    from spindle_power.load_prediction.training_checkpoint import load_checkpoint
    case = prospective_case
    shared = case["shared"]
    identity = dict(run=case["identity"], phase="main", model=asdict(case["model"]), training=asdict(case["config"]))
    saved = load_checkpoint(case["common"] / "main.pt", identity, resume=True)
    assert saved["epoch"] == saved["optimizer_updates"] == 100 and saved["completed"] is True
    assert shared["selected_epoch"] == saved["best_epoch"]
    assert all(torch.equal(value, saved["best_state"][key]) for key, value in shared["weights"].items())
    bundle = RUNNER.model_bundle(case["model"], case["config"], case["initial"], shared)
    assert bundle.execution_window_policy() == "prospective_v1"
    assert not any(parameter.requires_grad for parameter in bundle.main.parameters())
    mains, residuals = [], []
    for sample in case["train"]:
        selected, _, _ = prospective_residual_context(load_input_view(sample, "programmed").frame, case["model"].window_points)
        values = _main_values(bundle, SampleSet(selected, sample.metadata)).astype(np.float32)
        mask = selected.valid_residual.to_numpy(bool)
        mains.extend(values[mask].astype(float))
        residuals.extend(selected.power_cut_w.to_numpy(float)[mask] - values[mask])
    assert shared["normalization"]["main_mean_w"] == pytest.approx(np.mean(mains), abs=1e-10)
    assert shared["normalization"]["main_scale_w"] == pytest.approx(max(1., np.std(mains)), abs=1e-10)
    assert shared["normalization"]["residual_scale_w"] == pytest.approx(max(1., np.std(residuals)), abs=1e-10)
    assert shared["normalization_points"] == 60
    assert state_version(bundle.main) == shared["main_version"]


def test_actual_prospective_epoch_validation_uses_macro_rmse_and_keeps_paired_state_trainable(prospective_case, monkeypatch):
    from spindle_power.load_prediction import training_service
    from spindle_power.load_prediction.neural import state_version
    from spindle_power.load_prediction.training import _residual_data
    from spindle_power.load_prediction.training_cache import prepare_residual_loss
    case = prospective_case
    original = training_service.evaluate_records
    validations = []
    def audited(bundle, samples, **kwargs):
        assert bundle.execution_window_policy() == "prospective_v1"
        assert bundle.normalization == case["shared"]["normalization"]
        assert kwargs["memory"] is None and kwargs["update_memory"] is False
        assert kwargs["feed_mode"] == "programmed"
        report = original(bundle, samples, **kwargs)
        values = [row["metrics"]["rmse_w"] for row in report["records"]]
        macro = training_service.balanced_metrics(report["records"])["rmse_w"]
        assert macro != pytest.approx(float(np.sqrt(np.mean(np.square(values)))))
        validations.append(macro)
        return report
    monkeypatch.setattr(training_service, "evaluate_records", audited)
    q = RUNNER.model_bundle(case["model"], case["config"], case["initial"], case["shared"])
    p = RUNNER.model_bundle(replace(case["model"], residual_output="point"), case["config"], case["initial"], case["shared"])
    assert all(torch.equal(value, p.residual.state_dict()[key]) for key, value in q.residual.state_dict().items())
    prepared = _residual_data(q, case["train"])
    assert all(any(not bool(window[3].all()) for window in windows) for _, windows in prepared)
    for kind, bundle in (("Q", q), ("P", p)):
        path = case["directory"] / kind / "residual.pt"
        stage = dict(run={"synthetic": True}, arm=kind, maximum=4,
                     configuration=dict(gate_references=dict(window_audit={"synthetic": True})))
        for target in (2, 4):
            control = RUNNER.EpochControl(case["budget"], target, 4, None)
            loss = prepare_residual_loss(bundle, prepared, control.cancel, batch_windows=128, batch_strategy="padded")
            saved = RUNNER.run_epochs_to(bundle.residual, 4, target, case["config"], loss,
                lambda _back: RUNNER.validation_rmse(bundle, case["validation"], case["budget"]),
                "residual", control, path, stage)
            assert saved["epoch"] == saved["optimizer_updates"] == target
            assert saved["completed"] is (target == 4)
            assert saved["history"][-1]["validation_loss"] == validations[-1]
            assert state_version(bundle.main) == case["shared"]["main_version"]
            assert any(parameter.requires_grad for parameter in bundle.residual.parameters())
            if target == 2:
                receipt = RUNNER.export_milestone(case["directory"], kind, target, saved, bundle,
                    case["validation"], RUNNER.archived_checkpoint(path), case["budget"], stage,
                    dict(status="not_ready", shape_protocol=None, reference_attachments={}, unavailable_reasons=["synthetic_support_missing"]))
                assert receipt["completed"] is False
                assert all(row["shape_status"] == "not_ready" for row in receipt["evaluations"].values())
                checked = RUNNER.checked_milestone(case["directory"] / kind / "epoch_0002/receipt.json",
                                                   kind, target, stage, case["directory"])
                assert checked == receipt
                for export in ("fixed", "best"):
                    directory = case["directory"] / kind / "epoch_0002" / export
                    assert len(list(directory.glob("*.csv"))) == 50
                    diagnostics = RUNNER.read_json(directory / "shape_diagnostics.json")
                    assert len(diagnostics["records"]) == 50
    assert len(validations) == 12
