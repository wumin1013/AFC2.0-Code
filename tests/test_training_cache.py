"""Synthetic CPU/CUDA equivalence checks; no experimental accuracy claims."""
from copy import deepcopy
from dataclasses import asdict, replace
import os
from pathlib import Path
import sys
from unittest.mock import patch

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np
import pytest
import torch

from verify_unified_load_prediction import synthetic_records, SMALL_CANDIDATE
from spindle_power.load_prediction.common import Cancelled
from spindle_power.load_prediction.condition import fit_condition_spec
from spindle_power.load_prediction.models import ModelConfig, build_load_models
from spindle_power.load_prediction.samples import load_input_view
from spindle_power.load_prediction.training import TrainingConfig, _epochs, _main_loss, _residual_data, _residual_loss
from spindle_power.load_prediction.training_cache import prepare_main_loss, prepare_residual_loss, _TensorCache
from spindle_power.load_prediction.training_checkpoint import training_identity


@pytest.fixture
def fixture():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(491)
    _, records = synthetic_records((1, 2), points=17)
    samples = [records[index] for index in (0, 1, 2, 5)]
    for index, sample in enumerate(samples):
        sample.frame = sample.frame.iloc[:17 - index * 2].copy()
        sample.frame.loc[[1, 4], "valid_main"] = False
        sample.frame.loc[[2, 5], "valid_residual"] = False
        sample.metadata["condition_descriptors"]["machine"]["rated_power_w"] = 1000. + 100. * index
    samples[2].metadata["feature"] = samples[0].metadata["feature"]
    samples[3].metadata["group_id"] = "synthetic-event"
    config = ModelConfig(**SMALL_CANDIDATE["load_model"], route="nominal_programmed_residual_v1",
                         condition_spec=fit_condition_spec([s.metadata for s in samples]))
    yield samples, config
    torch.set_num_threads(previous)


def make_bundle(config):
    bundle = build_load_models(config)
    bundle.normalization = dict(main_mean_w=100., main_scale_w=50., residual_scale_w=20.)
    return bundle


def compare_update(left, right, reference, cached):
    """Check gradients before clipping, then one actual Adam update and state."""
    left.train()
    right.train()
    a = torch.optim.Adam([p for p in left.parameters() if p.requires_grad], lr=.001)
    b = torch.optim.Adam([p for p in right.parameters() if p.requires_grad], lr=.001)
    a.zero_grad()
    b.zero_grad()
    assert cached(True) == pytest.approx(reference(True), rel=1e-7, abs=1e-9)
    for p, q in zip(left.parameters(), right.parameters()):
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    torch.nn.utils.clip_grad_norm_(left.parameters(), 1.)
    torch.nn.utils.clip_grad_norm_(right.parameters(), 1.)
    a.step()
    b.step()
    for p, q in zip(left.parameters(), right.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    for p, q in zip(a.state.values(), b.state.values()):
        for key in p:
            torch.testing.assert_close(p[key], q[key], rtol=0, atol=0)
    left.eval()
    right.eval()
    with torch.no_grad():
        assert cached(False) == pytest.approx(reference(False), rel=1e-7, abs=1e-9)


@pytest.mark.parametrize("condition", [False, True])
def test_main_cache_preserves_output_weighted_gradient_and_adam(fixture, condition):
    samples, config = fixture
    if not condition:
        config = replace(config, condition_spec=None)
    left = make_bundle(config)
    right = deepcopy(left)
    training = TrainingConfig(main_batch_points=6)
    reference = lambda backward: _main_loss(left, samples, 80., training, None, backward)
    cached = prepare_main_loss(right, samples, 80., training)
    predictions = [[], []]
    hooks = [module.register_forward_hook(lambda module, args, output, i=i:
                                          predictions[i].append(output.detach().clone()))
             for i, module in enumerate((left.main, right.main))]
    compare_update(left.main, right.main, reference, cached)
    for hook in hooks:
        hook.remove()
    assert predictions[0] and len(predictions[0]) == len(predictions[1])
    for a, b in zip(*predictions):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert cached.stats["batch_count"] > len(samples)
    assert cached.stats["tensor_bytes"] == cached.stats["cpu_cache_bytes"]
    assert cached.stats["device_cache_bytes"] == 0


@pytest.mark.parametrize("output", ["quantiles", "point"])
@pytest.mark.parametrize("batch_windows,strategy", [(1, "equal_length"), (3, "equal_length"), (128, "padded")])
@pytest.mark.parametrize("condition", [False, True])
def test_residual_cache_preserves_windows_loss_gradient_and_adam(fixture, output, batch_windows, strategy, condition):
    samples, config = fixture
    config = replace(config, residual_output=output, condition_spec=config.condition_spec if condition else None)
    left = make_bundle(config)
    right = deepcopy(left)
    prepared = _residual_data(left, [load_input_view(s, "programmed") for s in samples], validation=True)
    spread_before = {name: value.detach().clone() for name, value in right.residual.spread.named_parameters()}
    if output == "point":
        assert all(not parameter.requires_grad for parameter in right.residual.spread.parameters())
    arguments = dict(batch_windows=batch_windows, batch_strategy=strategy)
    reference = lambda backward: _residual_loss(left, prepared, None, backward, **arguments)
    cached = prepare_residual_loss(right, prepared, **arguments)
    observed = [[], []]
    original = [left.residual.forward_quantiles, right.residual.forward_quantiles]

    def capture(index, *args, **kwargs):
        result = original[index](*args, **kwargs)
        observed[index].append(result.detach().clone())
        return result

    with patch.object(left.residual, "forward_quantiles", side_effect=lambda *a, **k: capture(0, *a, **k)), \
         patch.object(right.residual, "forward_quantiles", side_effect=lambda *a, **k: capture(1, *a, **k)):
        compare_update(left.residual, right.residual, reference, cached)
    assert observed[0] and len(observed[0]) == len(observed[1])
    for a, b in zip(*observed):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    if output == "point":
        for name, parameter in right.residual.spread.named_parameters():
            assert parameter.grad is None
            torch.testing.assert_close(parameter, spread_before[name], rtol=0, atol=0)


def test_cache_is_input_snapshot_and_recomputes_current_weights(fixture):
    samples, config = fixture
    bundle = make_bundle(config)
    training = TrainingConfig(main_batch_points=5)
    main = prepare_main_loss(bundle, samples, 80., training)
    prepared = _residual_data(bundle, samples)
    residual = prepare_residual_loss(bundle, prepared, batch_windows=128, batch_strategy="padded")
    with torch.no_grad():
        before_main, before_residual = main(False), residual(False)
        samples[0].frame.loc[:, "power_cut_w"] += 1000
        prepared[0][1][0][0].fill_(1000)
        prepared[0][1][0][1].fill_(1000)
        assert main(False) == before_main
        assert residual(False) == before_residual
        bundle.residual.output.bias.add_(.25)
        assert residual(False) != before_residual


@pytest.mark.parametrize("stage", ["main", "residual"])
def test_callable_accumulates_without_clearing_or_stepping(fixture, stage):
    samples, config = fixture
    bundle = make_bundle(config)
    if stage == "main":
        cached = prepare_main_loss(bundle, samples, 80., TrainingConfig(main_batch_points=6))
        module = bundle.main
    else:
        cached = prepare_residual_loss(bundle, _residual_data(bundle, samples), batch_windows=3,
                                       batch_strategy="padded")
        module = bundle.residual
    weights = {name: value.detach().clone() for name, value in module.named_parameters()}
    cached(True)
    gradients = {name: parameter.grad.clone() for name, parameter in module.named_parameters()
                 if parameter.grad is not None}
    cached(True)
    assert gradients
    for name, parameter in module.named_parameters():
        torch.testing.assert_close(parameter, weights[name], rtol=0, atol=0)
        if name in gradients:
            torch.testing.assert_close(parameter.grad, 2 * gradients[name], rtol=2e-6, atol=1e-7)


def test_legacy_windows_and_unscored_nan_do_not_change_weighting(fixture):
    samples, config = fixture
    bundle = make_bundle(config)
    prepared = _residual_data(bundle, samples)
    legacy = [(s, [window[:3] for window in windows]) for s, windows in prepared]
    for windows in (prepared, legacy):
        with torch.no_grad():
            expected = _residual_loss(bundle, windows, None, False, batch_windows=3, batch_strategy="padded")
            assert prepare_residual_loss(bundle, windows, batch_windows=3, batch_strategy="padded")(False) == expected
    masked = _residual_data(bundle, [load_input_view(s, "programmed") for s in samples], validation=True)
    for _, windows in masked:
        for _, y, _, mask in windows:
            y[~mask] = float("nan")
    with torch.no_grad():
        expected = _residual_loss(bundle, masked, None, False, batch_windows=3, batch_strategy="padded")
        actual = prepare_residual_loss(bundle, masked, batch_windows=3, batch_strategy="padded")(False)
    assert np.isfinite(actual) and actual == expected


@pytest.mark.parametrize("stage", ["main", "residual"])
def test_nonfinite_and_cancel_still_fail_per_batch(fixture, stage):
    samples, config = fixture
    bundle = make_bundle(config)
    cancelled = [False]
    if stage == "main":
        cached = prepare_main_loss(bundle, samples, 80., TrainingConfig(), lambda: cancelled[0])
        module = bundle.main
    else:
        cached = prepare_residual_loss(bundle, _residual_data(bundle, samples), lambda: cancelled[0])
        module = bundle.residual
    cancelled[0] = True
    with pytest.raises(Cancelled):
        cached(True)
    assert all(parameter.grad is None for parameter in module.parameters())
    cancelled[0] = False
    with torch.no_grad():
        next(module.parameters()).fill_(float("nan"))
    with pytest.raises(ValueError, match="非有限"):
        cached(True)


def test_device_budget_is_bounded_without_using_cuda():
    # Simulated storage permits a CPU-only test of the exact cache admission rule.
    cache = _TensorCache(torch.device("cuda"), True, 8)
    with patch("spindle_power.load_prediction.training_cache._on_device", side_effect=lambda values, device: values) as move:
        cache.store((torch.ones(2), None))
        cache.store((torch.ones(1), None))
    assert move.call_count == 1
    assert cache.stats["device_cache_bytes"] == 8
    assert cache.stats["cpu_cache_bytes"] == 4
    assert cache.stats["tensor_bytes"] == 12
    with pytest.raises(ValueError, match="非负整数"):
        _TensorCache(torch.device("cpu"), False, -1)


def assert_checkpoint_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_checkpoint_state_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_checkpoint_state_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("stage,output", [("main", "quantiles"), ("residual", "quantiles"), ("residual", "point")])
def test_cached_epochs_resume_matches_continuous_last_adam_rng_and_best(fixture, tmp_path, stage, output):
    samples, config = fixture
    config = replace(config, residual_output=output)
    initial = make_bundle(config)
    # Force a retained first-epoch best while later epochs still update last.
    training = TrainingConfig(main_epochs=3, residual_epochs=3, patience=8, min_delta=1e9,
                              main_batch_points=6, residual_batch_windows=128,
                              residual_batch_strategy="padded", seed=491)
    train, validation = samples[:3], samples[3:]
    identity = training_identity("load", train, validation,
                                 dict(training=asdict(training), model=asdict(config)),
                                 context=dict(purpose="synthetic_cached_resume", stage=stage))
    initial_rng = torch.get_rng_state().clone()

    def run(bundle, path, *, resume=False, progress=None):
        module = getattr(bundle, stage)
        if stage == "main":
            losses = [prepare_main_loss(bundle, data, 80., training) for data in (train, validation)]
        else:
            losses = []
            for data, is_validation in ((train, False), (validation, True)):
                view = [load_input_view(sample, "programmed") for sample in data]
                prepared = _residual_data(bundle, view, validation=is_validation)
                losses.append(prepare_residual_loss(bundle, prepared, batch_windows=128,
                                                     batch_strategy="padded"))
        result = _epochs(module, 3, training, losses[0], losses[1], stage, None, progress,
                         checkpoint_path=path, checkpoint_identity=identity, resume=resume)
        return result, module

    continuous_path, resumed_path = tmp_path / "continuous.pt", tmp_path / "resumed.pt"
    torch.set_rng_state(initial_rng)
    expected_result, expected_module = run(deepcopy(initial), continuous_path)

    def interrupt(event):
        if event.get("epoch") == 1:
            raise Cancelled("synthetic interruption after first cached full-gradient checkpoint")

    torch.set_rng_state(initial_rng)
    with pytest.raises(Cancelled, match="first cached"):
        run(deepcopy(initial), resumed_path, progress=interrupt)
    interrupted = torch.load(resumed_path, map_location="cpu", weights_only=True)["state"]
    assert interrupted["epoch"] == interrupted["optimizer_updates"] == 1
    assert interrupted["completed"] is False

    # A fresh process can have unrelated RNG use before checkpoint restoration.
    torch.rand(7)
    events = []
    actual_result, actual_module = run(deepcopy(initial), resumed_path, resume=True, progress=events.append)
    assert events[0]["phase"] == "resumed" and events[0]["epoch"] == 1
    assert [event["epoch"] for event in events[1:]] == [2, 3]
    assert actual_result == expected_result
    assert_checkpoint_state_equal(actual_module.state_dict(), expected_module.state_dict())
    expected = torch.load(continuous_path, map_location="cpu", weights_only=True)
    actual = torch.load(resumed_path, map_location="cpu", weights_only=True)
    assert expected["fingerprint"] == actual["fingerprint"]
    assert_checkpoint_state_equal(actual["state"], expected["state"])
    state = actual["state"]
    assert state["epoch"] == state["optimizer_updates"] == 3 and state["completed"] is True
    assert state["best_epoch"] == 1 and state["stale"] == 2 and len(state["history"]) == 3
    assert torch.equal(torch.get_rng_state(), state["torch_rng"])
    assert_checkpoint_state_equal(actual_module.state_dict(), state["best_state"])
    assert any(not torch.equal(state["weights"][name], state["best_state"][name])
               for name in state["weights"])
    for path in (continuous_path, resumed_path):
        assert len(list((path.parent / "epochs" / path.stem).glob("epoch_*.pt"))) == 3


@pytest.mark.parametrize("stage,output", [("main", "quantiles"), ("residual", "quantiles"), ("residual", "point")])
@pytest.mark.parametrize("storage", ["device", "mixed", "cpu_fallback"])
def test_cuda_cache_matches_same_device_reference_and_budget(fixture, stage, output, storage):
    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware unavailable; no GPU cache verification claimed")
    # The runner supplies CUBLAS_WORKSPACE_CONFIG before CUDA initialization;
    # ModelBundle.to applies the existing deterministic, float32 CUDA policy.
    samples, config = fixture
    config = replace(config, residual_output=output)
    left = make_bundle(config).to("cuda")
    right = deepcopy(left)
    if stage == "main":
        training = TrainingConfig(main_batch_points=6, device="cuda")
        reference = lambda backward: _main_loss(left, samples, 80., training, None, backward)
        prepare = lambda **options: prepare_main_loss(right, samples, 80., training, **options)
        method = "forward"
    else:
        prepared = _residual_data(left, [load_input_view(s, "programmed") for s in samples], validation=True)
        arguments = dict(batch_windows=3, batch_strategy="padded")
        reference = lambda backward: _residual_loss(left, prepared, None, backward, **arguments)
        prepare = lambda **options: prepare_residual_loss(right, prepared, **arguments, **options)
        method = "forward_quantiles"
    # Measure packed payloads without allocating a second persistent GPU cache.
    cpu_cache = prepare()
    first_bytes = sum(t.numel() * t.element_size() for t in cpu_cache.batches[0].tensors if t is not None)
    total_bytes = cpu_cache.stats["tensor_bytes"]
    assert cpu_cache.stats["batch_count"] > 1 and total_bytes > first_bytes
    budget = {"device": total_bytes, "mixed": first_bytes, "cpu_fallback": 0}[storage]
    del cpu_cache
    cached = prepare(cache_device=True, max_cache_bytes=budget)
    assert cached.stats["device"] == "cuda:0"
    assert cached.stats["device_cache_bytes"] == budget
    assert cached.stats["cpu_cache_bytes"] == total_bytes - budget
    for batch in cached.batches:
        devices = {tensor.device.type for tensor in batch.tensors if tensor is not None}
        assert len(devices) == 1
    if storage == "mixed":
        assert 0 < cached.stats["device_cache_bytes"] < total_bytes
    modules = [getattr(left, stage), getattr(right, stage)]
    originals = [getattr(module, method) for module in modules]
    predictions = [[], []]

    def capture(index, *args, **kwargs):
        result = originals[index](*args, **kwargs)
        predictions[index].append(result.detach().clone())
        return result

    spread_before = {name: value.detach().clone() for name, value in right.residual.spread.named_parameters()}
    with patch.object(modules[0], method, side_effect=lambda *a, **k: capture(0, *a, **k)), \
         patch.object(modules[1], method, side_effect=lambda *a, **k: capture(1, *a, **k)):
        compare_update(modules[0], modules[1], reference, cached)
    assert predictions[0] and len(predictions[0]) == len(predictions[1])
    for a, b in zip(*predictions):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    if output == "point":
        for name, parameter in right.residual.spread.named_parameters():
            assert not parameter.requires_grad and parameter.grad is None
            torch.testing.assert_close(parameter, spread_before[name], rtol=0, atol=0)
