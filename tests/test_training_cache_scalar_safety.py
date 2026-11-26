"""Small CPU checks of outer scalar safety and per-batch cancellation."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import pytest
import torch

from spindle_power.load_prediction.common import Cancelled
from spindle_power.load_prediction.samples import PHYSICS_COLUMNS
from spindle_power.load_prediction.training import TrainingConfig
from spindle_power.load_prediction.training_cache import prepare_main_loss, prepare_residual_loss


class ScalarMain(torch.nn.Module):
    def __init__(self, value=0.):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(value, dtype=torch.float32))
        self.forward_calls = 0

    def forward(self, physics, condition=None):
        self.forward_calls += 1
        return self.value.expand(len(physics))


class ScalarResidual(ScalarMain):
    def forward_quantiles(self, features, valid=None, condition=None):
        self.forward_calls += 1
        self.last_prediction = self.value.expand(*features.shape[:2], 3)
        return self.last_prediction


def main_problem(value=0., *, cancel=None, records=1):
    module = ScalarMain(value)
    bundle = SimpleNamespace(main=module, device=torch.device("cpu"),
                             condition_values=lambda metadata, count=None: None)
    samples = []
    for index in range(records):
        frame = pd.DataFrame({name: [1.] for name in PHYSICS_COLUMNS})
        frame["power_cut_w"], frame["valid_main"] = 1., True
        samples.append(SimpleNamespace(frame=frame, metadata=dict(round_id=index + 1, feature="synthetic")))
    loss = prepare_main_loss(bundle, samples, 1., TrainingConfig(main_batch_points=1), cancel)
    return module, loss


def residual_problem(value=0., *, output="point", target=1., cancel=None, records=1):
    module = ScalarResidual(value)
    bundle = SimpleNamespace(residual=module, config=SimpleNamespace(residual_output=output))
    prepared = []
    for index in range(records):
        sample = SimpleNamespace(metadata=dict(round_id=index + 1, feature="synthetic"))
        window = (torch.ones(1, 1, 1), torch.full((1, 1), target, dtype=torch.float32),
                  None, torch.ones(1, 1, dtype=torch.bool))
        prepared.append((sample, [window]))
    loss = prepare_residual_loss(bundle, prepared, cancel, batch_windows=1)
    return module, loss


@pytest.mark.parametrize("stage", ["main", "point"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_outer_nonfinite_scalar_is_rejected_before_backward(stage, value):
    module, loss = main_problem(value) if stage == "main" else residual_problem(value)
    message = "主趋势出现非有限损失" if stage == "main" else "残差出现非有限损失"
    # A detached check must run before the differentiable loss reaches autograd.
    with patch("torch.autograd.backward") as backward:
        with pytest.raises(ValueError, match=f"^{message}$"):
            loss(True)
        backward.assert_not_called()
    assert module.forward_calls == 1 and module.value.grad is None


def test_finite_quantile_inputs_overflow_only_at_objective_and_outer_check_rejects():
    module, loss = residual_problem(-2e38, output="quantiles", target=2e38)
    assert torch.isfinite(module.value)
    assert torch.isfinite(torch.tensor(2e38, dtype=torch.float32))
    with patch("torch.autograd.backward") as backward:
        # Prediction/target finite checks in quantile_loss pass. Their float32
        # difference overflows, so this exact error must come from the wrapper.
        with pytest.raises(ValueError, match="^残差出现非有限损失$"):
            loss(True)
        backward.assert_not_called()
    assert torch.isfinite(module.last_prediction).all()
    assert module.forward_calls == 1 and module.value.grad is None


@pytest.mark.parametrize("stage", ["main", "point"])
def test_cancel_is_checked_before_each_batch_and_preserves_prior_gradient(stage):
    runtime = dict(active=False, calls=0)

    def cancel():
        if not runtime["active"]:
            return False
        runtime["calls"] += 1
        return runtime["calls"] == 2

    factory = main_problem if stage == "main" else residual_problem
    module, loss = factory(cancel=cancel, records=2)
    gradients = []
    handle = module.value.register_hook(lambda value: gradients.append(value.detach().clone()))
    runtime["active"] = True
    try:
        with pytest.raises(Cancelled):
            loss(True)
    finally:
        handle.remove()
    assert runtime["calls"] == 2
    assert module.forward_calls == 1 and len(gradients) == 1
    assert torch.isfinite(module.value.grad) and module.value.grad != 0
    torch.testing.assert_close(module.value.grad, gradients[0], rtol=0, atol=0)
    assert module.value.item() == 0.
