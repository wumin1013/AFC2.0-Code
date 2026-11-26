"""Optional loss-cache integration; small synthetic data, no accuracy evidence."""
from dataclasses import replace
import os
from pathlib import Path
import sys
import weakref

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
from spindle_power.load_prediction.common import Cancelled, file_hash
from spindle_power.load_prediction.models import ModelConfig, predict_cutting_power
from spindle_power.load_prediction.samples import load_input_view
from spindle_power.load_prediction.training import TrainingConfig, train_two_stage


EXECUTIONS = [("cpu", "cpu"), ("cuda", "cpu"), ("cuda", "device")]


@pytest.fixture
def records():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    _, samples = synthetic_records((1, 2), points=12)
    for sample in samples:
        sample.frame.loc[[2, 5], "valid_residual"] = False
        sample.frame.loc[1, "valid_main"] = False
    samples[1].frame = samples[1].frame.iloc[:9].copy()
    yield samples[:2], samples[5:7]
    torch.set_num_threads(previous)


def options(device="cpu", mode="cpu", output="quantiles"):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware unavailable; no cache integration verification claimed")
    return dict(execute=True,
        model_config=ModelConfig(**SMALL_CANDIDATE["load_model"], route="nominal_programmed_residual_v1",
                                 residual_output=output),
        config=TrainingConfig(main_epochs=3, residual_epochs=3, patience=8, seed=927, device=device,
                              main_batch_points=6, residual_batch_windows=4, residual_batch_strategy="padded",
                              loss_cache_mode=mode, loss_cache_max_bytes=1024**2))


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    else:
        assert left == right


def assert_bundle_equal(left, right, sample):
    assert_state_equal(left.main.state_dict(), right.main.state_dict())
    assert_state_equal(left.residual.state_dict(), right.residual.state_dict())
    assert left.normalization == right.normalization
    assert left.metadata["stages"] == right.metadata["stages"]
    frame = load_input_view(sample, "programmed").frame
    a = predict_cutting_power(left, frame)["cutting_quantiles_w"]
    b = predict_cutting_power(right, frame)["cutting_quantiles_w"]
    np.testing.assert_array_equal(a, b)
    if right.config.residual_output == "point":
        assert all(not parameter.requires_grad for parameter in right.residual.spread.parameters())
        assert right.metadata["interval_status"] == "point_regression_no_interval"


@pytest.mark.parametrize("device,mode", EXECUTIONS)
@pytest.mark.parametrize("output", ["quantiles", "point"])
def test_optional_cache_matches_uncached_two_stage_training(records, device, mode, output):
    train, validation = records
    kwargs = options(device, mode, output)
    expected = train_two_stage(train, validation=validation,
        **{**kwargs, "config": replace(kwargs["config"], loss_cache_mode="none")})
    actual = train_two_stage(train, validation=validation, **kwargs)
    assert_bundle_equal(expected, actual, validation[0])
    assert actual.metadata["config"]["loss_cache_mode"] == mode


@pytest.mark.parametrize("device,mode", EXECUTIONS)
@pytest.mark.parametrize("output", ["quantiles", "point"])
@pytest.mark.parametrize("stage", ["main", "residual"])
def test_two_stage_cache_resume_preserves_last_optimizer_rng_and_best(records, tmp_path, device, mode, output, stage):
    train, validation = records
    kwargs = options(device, mode, output)
    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    expected = train_two_stage(train, validation=validation, checkpoint_dir=continuous, **kwargs)

    def interrupt(event):
        if event.get("stage") == stage and event.get("epoch") == 1:
            raise Cancelled("synthetic optional-cache interruption")

    with pytest.raises(Cancelled, match="optional-cache"):
        train_two_stage(train, validation=validation, checkpoint_dir=resumed, progress=interrupt, **kwargs)
    actual = train_two_stage(train, validation=validation, checkpoint_dir=resumed, resume=True, **kwargs)
    assert_bundle_equal(expected, actual, validation[0])
    for name in ("main", "residual"):
        left = torch.load(continuous / (name + ".pt"), map_location="cpu", weights_only=True)
        right = torch.load(resumed / (name + ".pt"), map_location="cpu", weights_only=True)
        assert left["fingerprint"] == right["fingerprint"]
        assert_state_equal(left["state"], right["state"])
        config = right["identity"]["configuration"]["training"]
        assert config["loss_cache_mode"] == mode and config["loss_cache_max_bytes"] == 1024**2
        assert right["state"]["optimizer_updates"] == 3
        assert len(list((resumed / "epochs" / name).glob("epoch_*.pt"))) == 3


@pytest.mark.parametrize("changes", [dict(loss_cache_mode="other"), dict(loss_cache_mode=True),
    dict(loss_cache_max_bytes=-1), dict(loss_cache_max_bytes=1.5), dict(loss_cache_max_bytes=True),
    dict(device="cpu", loss_cache_mode="device")])
def test_invalid_cache_config_is_rejected(changes):
    with pytest.raises(ValueError):
        TrainingConfig(**changes).validate()


def test_cache_defaults_preserve_uncached_execution_and_accept_zero_budget():
    config = TrainingConfig()
    assert config.loss_cache_mode == "none" and config.loss_cache_max_bytes == 256 * 1024**2
    for mode in ("none", "cpu"):
        replace(config, loss_cache_mode=mode, loss_cache_max_bytes=0).validate()
    replace(config, device="cuda", loss_cache_mode="device", loss_cache_max_bytes=0).validate()


@pytest.mark.parametrize("changes", [dict(loss_cache_mode="none"), dict(loss_cache_max_bytes=0)])
def test_changed_cache_policy_cannot_resume_existing_checkpoint(records, tmp_path, changes):
    train, validation = records
    kwargs = options()
    train_two_stage(train, validation=validation, checkpoint_dir=tmp_path, **kwargs)
    previous = file_hash(tmp_path / "main.pt")
    with pytest.raises(ValueError, match="指纹"):
        train_two_stage(train, validation=validation, checkpoint_dir=tmp_path, resume=True,
                        **{**kwargs, "config": replace(kwargs["config"], **changes)})
    assert file_hash(tmp_path / "main.pt") == previous


def test_main_loss_caches_are_released_before_residual_preparation(records, monkeypatch):
    from spindle_power.load_prediction import training_cache
    train, validation = records
    original_main = training_cache.prepare_main_loss
    original_residual = training_cache.prepare_residual_loss
    references, residual_calls = [], []

    def main(*args, **kwargs):
        result = original_main(*args, **kwargs)
        references.append(weakref.ref(result))
        return result

    def residual(*args, **kwargs):
        assert len(references) == 2 and all(reference() is None for reference in references)
        residual_calls.append(True)
        return original_residual(*args, **kwargs)

    monkeypatch.setattr(training_cache, "prepare_main_loss", main)
    monkeypatch.setattr(training_cache, "prepare_residual_loss", residual)
    train_two_stage(train, validation=validation, **options())
    assert len(residual_calls) == 2
