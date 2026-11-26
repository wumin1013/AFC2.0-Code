"""Synthetic load-route/device checks; never evidence of experimental accuracy."""
from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
import sys
from unittest.mock import patch

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np
import pytest
import torch

from verify_unified_load_prediction import synthetic_records, SMALL_CANDIDATE
from spindle_power.load_prediction.common import Cancelled
from spindle_power.load_prediction.models import ModelConfig, build_load_models, load_models, predict_cutting_power
from spindle_power.load_prediction.neural import state_version
from spindle_power.load_prediction.samples import load_input_view, path_feature_names
from spindle_power.load_prediction.training import TrainingConfig, train_two_stage, _residual_data, _residual_loss

ROUTE = "nominal_programmed_residual_v1"


@pytest.fixture
def records():
    torch.set_num_threads(1)
    _, loads = synthetic_records((1, 2), points=24)
    for sample in loads:
        sample.frame.loc[[5, 6], "valid_residual"] = False
        sample.frame["valid_main"] = np.arange(len(sample.frame)) % 3 == 0
    return loads[:2], loads[5:7]


def options(device="cpu", batch_windows=1, batch_strategy="equal_length"):
    return dict(model_config=ModelConfig(**SMALL_CANDIDATE["load_model"], route=ROUTE),
                config=TrainingConfig(main_epochs=3, residual_epochs=3, patience=8, seed=927, device=device,
                                      residual_batch_windows=batch_windows, residual_batch_strategy=batch_strategy),
                execute=True)


def require_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware unavailable; no GPU verification claimed")


def test_new_route_fits_measured_main_then_uses_nominal_residual(records):
    train, validation = records
    from spindle_power.load_prediction import training
    observed = []
    original = training._main_loss
    def inspect(bundle, samples, *args):
        for sample in samples:
            np.testing.assert_array_equal(sample.frame.feed_input_mm_min, sample.frame.feed_actual_mm_min)
            observed.append(len(sample.frame))
        return original(bundle, samples, *args)
    with patch.object(training, "_main_loss", side_effect=inspect):
        bundle = train_two_stage(train, validation=validation, **options())
    assert observed and bundle.residual.projection.in_features == 18
    nominal = [load_input_view(s, "programmed") for s in train]
    values, residuals = [], []
    for sample in nominal:
        main = training._main_values(bundle, sample)
        good = sample.frame.valid_residual.to_numpy(bool)
        values.extend(main[good])
        residuals.extend(sample.frame.power_cut_w.to_numpy()[good] - main[good])
    assert bundle.normalization["main_mean_w"] == float(np.mean(values))
    assert bundle.normalization["residual_scale_w"] == max(1., float(np.std(residuals)))
    assert bundle.main_version == bundle.residual_main_version == state_version(bundle.main)
    assert all(not p.requires_grad for module in (bundle.main, bundle.residual) for p in module.parameters())


def test_validation_uses_prospective_windows_then_scores_and_prediction_ignores_labels(records):
    train, validation = records
    bundle = train_two_stage(train, validation=validation, **options())
    sample = load_input_view(validation[0], "programmed")
    before = _residual_data(bundle, [sample], validation=True)[0][1]
    changed = deepcopy(sample)
    changed.frame.loc[[1, 9], "valid_residual"] = False
    changed.frame.loc[[1, 9], "power_cut_w"] = np.nan
    after = _residual_data(bundle, [changed], validation=True)[0][1]
    assert [x.shape for x, *_ in before] == [x.shape for x, *_ in after]
    assert all(torch.equal(a[0], b[0]) for a, b in zip(before, after))
    assert sum(int(w[3].sum()) for w in before) - sum(int(w[3].sum()) for w in after) == 2
    predicted = predict_cutting_power(bundle, sample.frame)["cutting_quantiles_w"]
    modified = predict_cutting_power(bundle, changed.frame)["cutting_quantiles_w"]
    np.testing.assert_array_equal(predicted, modified)
    assert np.isfinite(predicted).all() and (np.diff(predicted, axis=1) >= 0).all()
    training_windows = _residual_data(bundle, [sample])[0][1]
    assert all(bool(mask.all()) for _, _, _, mask in training_windows)
    assert sum(x.shape[1] for x, *_ in training_windows) == int(sample.frame.valid_residual.sum())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("stage", ["main", "residual"])
@pytest.mark.parametrize("batch_windows,batch_strategy", [(1, "equal_length"), (4, "equal_length"), (4, "padded")])
def test_complete_epoch_resume_matches_continuous_weights_optimizer_rng(records, tmp_path, device, stage, batch_windows, batch_strategy):
    require_device(device)
    train, validation = records
    expected_path, resumed_path = tmp_path / "continuous", tmp_path / "resumed"
    expected = train_two_stage(train, validation=validation, **options(device, batch_windows, batch_strategy), checkpoint_dir=expected_path)
    def interrupt(event):
        if event.get("stage") == stage and event.get("epoch") == 1:
            raise Cancelled("test interruption after complete immutable epoch")
    with pytest.raises(Cancelled):
        train_two_stage(train, validation=validation, **options(device, batch_windows, batch_strategy), checkpoint_dir=resumed_path, progress=interrupt)
    actual = train_two_stage(train, validation=validation, **options(device, batch_windows, batch_strategy), checkpoint_dir=resumed_path, resume=True)
    assert actual.device.type == device
    assert actual.main_version == expected.main_version
    assert actual.residual_version == expected.residual_version
    assert actual.normalization == expected.normalization
    assert actual.metadata["stages"] == expected.metadata["stages"]
    for current_stage in ("main", "residual"):
        a = torch.load(expected_path / (current_stage + ".pt"), map_location="cpu", weights_only=True)
        b = torch.load(resumed_path / (current_stage + ".pt"), map_location="cpu", weights_only=True)
        assert a["fingerprint"] == b["fingerprint"]
        assert a["state"]["optimizer_updates"] == b["state"]["optimizer_updates"] == 3
        for key in ("weights", "optimizer", "best_state", "torch_rng", "cuda_rng"):
            assert_nested_equal(a["state"].get(key), b["state"].get(key))
        assert len(list((resumed_path / "epochs" / current_stage).glob("epoch_*.pt"))) == 3
    if device == "cuda":
        assert actual.metadata["load_execution"]["device"] == "cuda"
        assert a["state"]["cuda_rng"]


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("stage", ["main", "residual"])
def test_explicit_cpu_four_threads_checkpoint_and_exact_recovery(records, tmp_path, monkeypatch, stage):
    original = options
    def four_threads(device="cpu", batch_windows=1, batch_strategy="equal_length"):
        result = original(device, batch_windows, batch_strategy)
        result["config"] = replace(result["config"], cpu_threads=4, cpu_interop_threads=1)
        return result
    monkeypatch.setitem(globals(), "options", four_threads)
    test_complete_epoch_resume_matches_continuous_weights_optimizer_rng(records, tmp_path, "cpu", stage, 128, "padded")
    state = torch.load(tmp_path / "resumed/residual.pt", map_location="cpu", weights_only=True)
    execution = state["identity"]["context"]["load_execution"]
    assert execution["cpu_threads"] == execution["requested_cpu_threads"] == 4
    assert execution["cpu_interop_threads"] == execution["requested_cpu_interop_threads"] == 1
    changed = four_threads("cpu", 128, "padded")
    changed["config"] = replace(changed["config"], cpu_threads=2)
    train, validation = records
    with pytest.raises(ValueError, match="身份|指纹"):
        train_two_stage(train, validation=validation, **changed, checkpoint_dir=tmp_path / "resumed", resume=True)


def test_gpu_reload_cpu_inference_equivalence_and_six_dimensional_rejection(records, tmp_path):
    require_device("cuda")
    train, validation = records
    bundle = train_two_stage(train, validation=validation, **options("cuda"))
    frame = load_input_view(validation[0], "programmed").frame
    expected = predict_cutting_power(bundle, frame)["cutting_quantiles_w"]
    path = tmp_path / "nominal.pt"
    bundle.save(path)
    reloaded = load_models(path, device="cuda")
    np.testing.assert_array_equal(predict_cutting_power(reloaded, frame)["cutting_quantiles_w"], expected)
    cpu = load_models(path)
    np.testing.assert_allclose(predict_cutting_power(cpu, frame)["cutting_quantiles_w"], expected, rtol=2e-5, atol=2e-4)
    with pytest.raises(ValueError, match="18"):
        cpu.residual.forward_quantiles(torch.ones(1, 8, 6))
    payload = torch.load(path, weights_only=True)
    payload["input_definition"]["path"] = path_feature_names("predicted_feed_v1")
    torch.save(payload, tmp_path / "wrong_route.pt")
    with pytest.raises(ValueError, match="输入定义"):
        load_models(tmp_path / "wrong_route.pt")


def test_explicit_cuda_refuses_unavailable_hardware():
    with patch("torch.cuda.is_available", return_value=False), pytest.raises(ValueError, match="拒绝静默回退"):
        build_load_models(ModelConfig(route=ROUTE), device="cuda")


def test_legacy_model_package_without_route_stays_readable(tmp_path):
    config = ModelConfig(**SMALL_CANDIDATE["load_model"])
    bundle = build_load_models(config)
    path = tmp_path / "legacy.pt"
    bundle.save(path)
    payload = torch.load(path, weights_only=True)
    payload["config"].pop("route")
    torch.save(payload, path)
    restored = load_models(path)
    assert restored.current_version() == bundle.current_version()
    assert restored.residual.projection.in_features == 6


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("validation", [False, True])
@pytest.mark.parametrize("batch_strategy", ["equal_length", "padded"])
def test_batch_preserves_loss_gradients_and_window_predictions(records, device, validation, batch_strategy):
    require_device(device)
    train, _ = records
    bundle = train_two_stage(train, **options(device))
    nominal = [load_input_view(sample, "programmed") for sample in train]
    # Unequal record lengths and label masks test the original per-record denominator.
    nominal[1].frame = nominal[1].frame.iloc[:17].copy()
    nominal[1].frame.loc[[1, 9], "valid_residual"] = False
    prepared = _residual_data(bundle, nominal, validation=validation)
    serial, batched = deepcopy(bundle), deepcopy(bundle)
    for current in (serial, batched):
        current.residual.train()
        for parameter in current.residual.parameters():
            parameter.requires_grad_(True)
            parameter.grad = None
    serial_loss = _residual_loss(serial, prepared, None, True)
    batched_loss = _residual_loss(batched, prepared, None, True, batch_windows=64, batch_strategy=batch_strategy)
    np.testing.assert_allclose(batched_loss, serial_loss, rtol=3e-6, atol=2e-7)
    for name, parameter in serial.residual.named_parameters():
        other = dict(batched.residual.named_parameters())[name]
        if parameter.grad is not None:
            torch.testing.assert_close(other.grad, parameter.grad, rtol=3e-4, atol=3e-6)
    frame = nominal[0].frame
    expected = predict_cutting_power(bundle, frame)["cutting_quantiles_w"]
    bundle.metadata["config"]["residual_batch_windows"] = 64
    bundle.metadata["config"]["residual_batch_strategy"] = batch_strategy
    actual = predict_cutting_power(bundle, frame)["cutting_quantiles_w"]
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-4)
    changed = frame.copy()
    changed["valid_residual"] = False
    changed["power_cut_w"] = np.nan
    np.testing.assert_array_equal(predict_cutting_power(bundle, changed)["cutting_quantiles_w"], actual)
