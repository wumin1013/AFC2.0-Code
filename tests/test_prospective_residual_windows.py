"""Synthetic prospective-window contracts; no machining accuracy claims."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np
import pandas as pd
import pytest
import torch

from verify_unified_load_prediction import SMALL_CANDIDATE, synthetic_records
from spindle_power.load_prediction.condition import fit_condition_spec
from spindle_power.load_prediction.models import ModelConfig, build_load_models, load_models, predict_cutting_power
from spindle_power.load_prediction.neural import prospective_residual_context, state_version
from spindle_power.load_prediction.samples import load_input_view
from spindle_power.load_prediction.training import TrainingConfig, _residual_data, _residual_loss, train_two_stage
from spindle_power.load_prediction.training_cache import prepare_residual_loss
from spindle_power.load_prediction.training_service import TrainingService, evaluate_records

POLICY = "prospective_v1"
ROUTE = "nominal_programmed_residual_v1"


@pytest.fixture
def case():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(918)
    _, loads = synthetic_records((1, 2), points=16)
    sample = loads[0]
    sample.frame["nc_sequence_id"] = sample.frame["training_sequence_id"] = 0
    sample.frame["time_s"] = np.arange(16, dtype=float)
    sample.frame.loc[[2, 3, 14, 15], "valid_residual"] = False
    sample.frame.loc[5, "direction_y"] = np.nan
    sample.frame.loc[5, ["valid_main", "valid_residual"]] = False
    config = ModelConfig(**{**SMALL_CANDIDATE["load_model"], "route": ROUTE, "window_points": 4,
                            "condition_spec": fit_condition_spec([s.metadata for s in loads])})
    bundle = build_load_models(config)
    bundle.main_trained = bundle.residual_trained = True
    bundle.main_version = bundle.residual_main_version = state_version(bundle.main)
    bundle.residual_version = state_version(bundle.residual)
    bundle.normalization = dict(main_mean_w=50., main_scale_w=20., residual_scale_w=10.)
    bundle.metadata["config"] = dict(residual_window_policy=POLICY,
                                      residual_batch_windows=3, residual_batch_strategy="padded")
    yield bundle, sample
    torch.set_num_threads(previous)


def plan(sample, width=4):
    selected, indices, windows = prospective_residual_context(load_input_view(sample, "programmed").frame, width)
    return selected, indices, windows


def manifests(sample):
    _, indices, windows = plan(sample)
    return [indices[index].tolist() for index in windows]


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert left == right


def test_complete_plan_ignores_all_observed_fields_including_zero_supervision_windows(case):
    _, sample = case
    expected = manifests(sample)
    assert expected == [[0, 1, 2, 3], [4], [6, 7, 8, 9], [10, 11, 12, 13], [14, 15]]
    changed = deepcopy(sample)
    changed.frame[["valid_main", "valid_residual"]] = False
    changed.frame["power_cut_w"] = np.nan
    changed.frame["time_s"] = np.arange(16)[::-1]
    changed.frame["training_sequence_id"] = np.arange(16)
    assert manifests(changed) == expected


@pytest.mark.parametrize("change", ["ap_mm", "direction_y", "path_mm", "sequence_id",
                                    "execution_instance", "cutting_contact", "pause", "known_break"])
def test_known_gaps_and_nc_boundaries_never_join_context(case, change):
    _, sample = case
    frame = load_input_view(sample, "programmed").frame
    frame.loc[5, "direction_y"] = 0.
    if change in ("ap_mm", "direction_y"):
        frame.loc[4, change] = np.nan
    elif change == "path_mm":
        frame.loc[4:, change] -= 100.
    elif change == "sequence_id":
        frame.loc[4:7, change] = 1  # Later zero IDs must not merge with the first zero run.
    elif change == "execution_instance":
        frame.loc[4:, change] = 1
    elif change == "cutting_contact":
        # Process geometry is authoritative when ap/ae are present.
        frame.loc[4:, "ap_mm"] = 0.
        frame.loc[4:, change] = False
    else:
        frame[change] = False
        frame.loc[4, change] = True
    _, indices, windows = prospective_residual_context(frame, 64)
    actual = [indices[index] for index in windows]
    assert not any((index < 4).any() and (index >= 4).any() for index in actual)
    if change in ("ap_mm", "direction_y"):
        assert 4 not in indices
    if change in ("pause", "known_break"):
        assert any(index.tolist() == [4] for index in actual)
    if change == "sequence_id":
        assert [index.tolist() for index in actual] == [list(range(4)), list(range(4, 8)), list(range(8, 16))]


@pytest.mark.parametrize("strategy,maximum", [("equal_length", 3), ("padded", 128)])
def test_training_validation_and_actual_deployment_share_full_feature_context(case, monkeypatch, strategy, maximum):
    bundle, sample = case
    bundle.metadata["config"].update(residual_batch_strategy=strategy, residual_batch_windows=maximum)
    train = _residual_data(bundle, [sample])[0][1]
    validation = _residual_data(bundle, [sample], validation=True)[0][1]
    assert_equal(train, validation)
    selected, _, windows = plan(sample)
    assert len(train) == len(windows) - 1  # A whole unscored window is skipped, but no retained window is shortened.
    observed = []
    original = bundle.residual.forward_quantiles

    def capture(x, valid=None, condition=None):
        for i in range(len(x)):
            keep = torch.ones(x.shape[1], dtype=torch.bool) if valid is None else valid[i].cpu()
            observed.append((x[i].cpu()[keep].clone(), condition[i].cpu()[keep].clone()))
        return original(x, valid=valid, condition=condition)

    monkeypatch.setattr(bundle.residual, "forward_quantiles", capture)
    predict_cutting_power(bundle, selected, context=sample.metadata)
    for x, _, condition, _ in train:
        assert any(torch.equal(x[0], a) and torch.equal(condition[0], b) for a, b in observed)
    assert sum(len(x) for x, _ in observed) == len(selected)
    assert any(not window[3].all() for window in train)


@pytest.mark.parametrize("column", ["ap_mm", "direction_x", "curvature_per_mm", "block_length_mm"])
def test_low_level_rejects_incomplete_inputs_instead_of_shortening_output(case, column):
    bundle, sample = case
    selected, _, _ = plan(sample)
    selected.loc[0, column] = np.nan
    with pytest.raises(ValueError, match="底层加工前预测"):
        predict_cutting_power(bundle, selected, context=sample.metadata)


def test_supervised_unknown_point_and_nonfinite_target_fail_closed(case):
    bundle, sample = case
    sample.frame.loc[5, "valid_residual"] = True
    with pytest.raises(ValueError, match="完整加工前输入"):
        _residual_data(bundle, [sample])
    sample.frame.loc[5, "valid_residual"] = False
    sample.frame.loc[0, "power_cut_w"] = np.nan
    with pytest.raises(ValueError, match="非有限目标"):
        _residual_data(bundle, [sample])


def test_service_prediction_coverage_and_values_are_label_independent(case, tmp_path):
    bundle, sample = case
    changed = deepcopy(sample)
    changed.frame.loc[[1, 4, 10], "valid_residual"] = False
    changed.frame.loc[~changed.frame.valid_residual, "power_cut_w"] = np.nan
    changed.frame["time_s"] = np.arange(16)[::-1]
    changed.frame["training_sequence_id"] = np.arange(16)
    outputs = []
    for name, record in (("before", sample), ("after", changed)):
        directory = tmp_path / name
        result = evaluate_records(bundle, [record], output_dir=directory, memory=None, update_memory=False,
            metric_parameters=dict(scale_w=100., high_threshold_w=100., low_threshold_w=30.))
        outputs.append(pd.read_csv(directory / result["records"][0]["prediction_file"], float_precision="round_trip"))
    left, right = outputs
    for column in ("prediction_available", "main_w", "residual_w", "cutting_power_w", "q05_w", "q50_w", "q95_w"):
        np.testing.assert_array_equal(left[column], right[column])
    assert left.prediction_available.sum() == 15
    assert left.scored.sum() != right.scored.sum()


def test_policy_is_saved_and_part_of_identity_without_changing_legacy(case, tmp_path):
    bundle, _ = case
    path = tmp_path / "prospective.pt"
    bundle.save(path)
    loaded = load_models(path)
    assert loaded.execution_window_policy() == POLICY and loaded.current_version() == bundle.current_version()
    assert loaded.input_definition()["residual_window_policy"] == POLICY
    legacy = deepcopy(bundle)
    del legacy.metadata["config"]["residual_window_policy"]
    identity, definition = legacy.current_version(), legacy.input_definition()
    legacy.metadata["config"]["residual_window_policy"] = "legacy_v1"
    assert identity == legacy.current_version() and definition == legacy.input_definition()
    assert identity != bundle.current_version()
    payload = torch.load(path, weights_only=True)
    for policy in (None, "legacy_v1", "unknown"):
        changed = deepcopy(payload)
        config = changed["metadata"]["config"]
        if policy is None:
            del config["residual_window_policy"]
        else:
            config["residual_window_policy"] = policy
        tampered = tmp_path / f"tampered_{policy}.pt"
        torch.save(changed, tampered)
        with pytest.raises(ValueError):
            load_models(tampered)


def test_policy_rejects_unknown_or_non_nominal_route_before_dry_run(case):
    bundle, sample = case
    with pytest.raises(ValueError, match="窗口策略"):
        TrainingConfig(residual_window_policy="unknown").validate()
    with pytest.raises(ValueError, match="仅支持"):
        train_two_stage([sample], model_config=replace(bundle.config, route="predicted_feed_v1"),
                        config=TrainingConfig(residual_window_policy=POLICY))


@pytest.mark.parametrize("output", ["quantiles", "point"])
@pytest.mark.parametrize("strategy,maximum", [("equal_length", 3), ("padded", 128)])
def test_prospective_cache_preserves_loss_raw_gradient_and_one_adam_update(case, output, strategy, maximum):
    initial, sample = case
    left = build_load_models(replace(initial.config, residual_output=output))
    left.normalization = deepcopy(initial.normalization)
    left.metadata = deepcopy(initial.metadata)
    right = deepcopy(left)
    prepared = _residual_data(left, [sample])
    arguments = dict(batch_windows=maximum, batch_strategy=strategy)
    reference = lambda backward: _residual_loss(left, prepared, None, backward, **arguments)
    cached = prepare_residual_loss(right, prepared, **arguments)
    optimizers = [torch.optim.Adam([p for p in model.residual.parameters() if p.requires_grad], lr=.001)
                  for model in (left, right)]
    for optimizer in optimizers:
        optimizer.zero_grad()
    assert reference(True) == cached(True)
    for a, b in zip(left.residual.parameters(), right.residual.parameters()):
        assert_equal(a.grad, b.grad)
    for optimizer in optimizers:
        optimizer.step()
    assert_equal(left.residual.state_dict(), right.residual.state_dict())
    assert_equal(optimizers[0].state_dict(), optimizers[1].state_dict())
    assert state_version(left.main) == state_version(right.main)


@pytest.mark.parametrize("output", ["quantiles", "point"])
@pytest.mark.parametrize("cache", ["none", "cpu"])
def test_prospective_checkpoint_resume_preserves_weights_optimizer_rng_and_best(case, tmp_path, output, cache):
    from spindle_power.load_prediction.common import Cancelled
    bundle, sample = case
    validation = deepcopy(sample)
    validation.metadata.update(record_id="synthetic_validation", round_id=2)
    options = dict(execute=True, model_config=replace(bundle.config, residual_output=output),
        config=TrainingConfig(main_epochs=2, residual_epochs=2, patience=5, seed=919,
            residual_batch_windows=128, residual_batch_strategy="padded", residual_window_policy=POLICY,
            loss_cache_mode=cache, main_batch_points=8))
    def interrupt(event):
        if event.get("stage") == "residual" and event.get("epoch") == 1:
            raise Cancelled("synthetic prospective interruption")
    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    expected = train_two_stage([sample], validation=[validation], checkpoint_dir=continuous, **options)
    with pytest.raises(Cancelled):
        train_two_stage([sample], validation=[validation], checkpoint_dir=resumed, progress=interrupt, **options)
    actual = train_two_stage([sample], validation=[validation], checkpoint_dir=resumed, resume=True, **options)
    assert_equal(expected.main.state_dict(), actual.main.state_dict())
    assert_equal(expected.residual.state_dict(), actual.residual.state_dict())
    assert_equal(expected.metadata["stages"], actual.metadata["stages"])
    for stage in ("main", "residual"):
        a = torch.load(continuous / (stage + ".pt"), weights_only=True)
        b = torch.load(resumed / (stage + ".pt"), weights_only=True)
        assert_equal(a, b)
    assert actual.main_version == actual.residual_main_version
    for change in (dict(residual_window_policy="legacy_v1"), dict(loss_cache_mode="cpu" if cache == "none" else "none")):
        with pytest.raises(ValueError, match="指纹"):
            train_two_stage([sample], validation=[validation], checkpoint_dir=resumed, resume=True,
                            **{**options, "config": replace(options["config"], **change)})
    changed = deepcopy(sample)
    changed.frame.loc[1, "valid_residual"] = False
    assert manifests(changed) == manifests(sample)
    with pytest.raises(ValueError, match="指纹"):
        train_two_stage([changed], validation=[validation], checkpoint_dir=resumed, resume=True, **options)


def test_public_service_uses_shared_eligibility_for_incomplete_curvature(case, tmp_path, monkeypatch):
    from spindle_power.load_prediction.feed import MachineConfig
    import spindle_power.load_prediction.samples as samples
    bundle, sample = case
    sample.frame.loc[7, "curvature_per_mm"] = np.nan
    service = TrainingService(tmp_path / "service")
    monkeypatch.setattr(service, "load_model", lambda *a, **kw: (None, bundle, {}))
    monkeypatch.setattr(samples, "_prepare_process_features", lambda *a, **kw: deepcopy(sample))
    machine = tmp_path / "machine.json"
    machine.write_text(json.dumps(asdict(MachineConfig(initial_position_mm=[0., 0., 0.],
        work_offsets_mm={"G54": [0., 0., 0.]}, spindle_wait_s=0., optional_stop="skip"))), encoding="utf-8")
    nc = tmp_path / "tiny.nc"
    nc.write_text("G21 G17 G90 G94\nG1 X2 F2000 S8000\nG1 Y2\n", encoding="utf-8")
    process = tmp_path / "process.csv"
    pd.DataFrame([dict(line=line, execution_instance=0, segment_mm=position, ap_mm=1., ae_mm=2.)
                  for line in (2, 3) for position in (0., 2.)]).to_csv(process, index=False)
    result = service.predict(dict(model_id="fixture", machine=str(machine), nc=str(nc), process=str(process),
                                  metadata=sample.metadata, allow_synthetic=True))
    frame = pd.DataFrame(result["columns"])
    assert len(frame) == 14 and np.isfinite(frame.cutting_power_w).all()
    assert 3.5 not in frame.path_mm.to_numpy()


def test_independent_window_audit_saves_complete_actual_context_evidence(case, tmp_path):
    import importlib.util
    path = ROOT / "scripts/verification/audit_paper1_prospective_windows.py"
    spec = importlib.util.spec_from_file_location("prospective_window_audit", path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    bundle, sample = case
    result = audit.audit_plan(sample, bundle.config.window_points)
    assert result["complete_manifest_label_invariant"] and result["zero_supervision_windows"] == 1
    assert result["known_points"] == 15 and len(result["windows"]) == 5
    actual = audit.audit_forward_context(bundle, sample, tmp_path / "context")
    assert actual["passed"]
    report = json.loads(Path(actual["path"]).read_text(encoding="utf-8"))
    assert report["complete_window_count"] == 5 and report["scored_window_count"] == 4
    assert report["deployment_outputs_label_invariant"] and report["train_deployment_context_exact"]
    tensors = np.load(report["tensors"]["path"])
    assert len([key for key in tensors if key.startswith("features_")]) == 5
