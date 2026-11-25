"""Snapshot immutable loss inputs without changing full-gradient training.

The factories return ``loss(backward: bool) -> float`` callables. They never
clear gradients, clip gradients or step an optimizer. Rebuild a callable after
changing samples, normalization, windows, condition encoding, or device; model
weights may change normally between calls. This optional execution path is
selected explicitly by TrainingConfig; it does not change training objectives.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .common import checkpoint
from .samples import PHYSICS_COLUMNS
from .training import _sample_weights


DEFAULT_DEVICE_CACHE_BYTES = 256 * 1024 * 1024


class _TensorCache:
    """Keep packed CPU inputs, optionally moving whole batches within a budget.

    The budget covers persistent tensor payloads only, not model/optimizer
    memory, activations, or allocator overhead. Each factory has its own budget.
    Batches exceeding the remaining budget stay on CPU, without changing order.
    """

    def __init__(self, device, cache_device, max_cache_bytes):
        if type(cache_device) is not bool:
            raise ValueError("cache_device必须为布尔值")
        if type(max_cache_bytes) is not int or max_cache_bytes < 0:
            raise ValueError("max_cache_bytes必须为非负整数")
        self.device = device
        self.enabled = cache_device and device.type != "cpu"
        self.stats = dict(batch_count=0, tensor_bytes=0, device_cache_bytes=0,
                          cpu_cache_bytes=0, max_cache_bytes=max_cache_bytes,
                          device=str(device), cache_device=cache_device,
                          outer_scalar_reads_per_batch=1)

    def store(self, values):
        size = sum(value.numel() * value.element_size() for value in values if value is not None)
        self.stats["batch_count"] += 1
        self.stats["tensor_bytes"] += size
        if self.enabled and self.stats["device_cache_bytes"] + size <= self.stats["max_cache_bytes"]:
            result = _on_device(values, self.device)
            self.stats["device_cache_bytes"] += size
            return result
        self.stats["cpu_cache_bytes"] += size
        return values


def _on_device(values, device):
    return tuple(None if value is None else value.to(device) for value in values)


@dataclass
class _MainBatch:
    tensors: tuple
    weight: float
    count: int


@dataclass
class _ResidualBatch:
    tensors: tuple
    weight: float
    count: int
    scored: int


class _PreparedMainLoss:
    def __init__(self, bundle, batches, scale, cancel, cache):
        self.bundle, self.batches, self.scale = bundle, batches, scale
        self.cancel, self.device, self.stats = cancel, cache.device, cache.stats

    def __call__(self, backward):
        if self.bundle.device != self.device:
            raise ValueError("主趋势设备已改变，须重新准备损失缓存")
        total = 0.
        for batch in self.batches:
            checkpoint(self.cancel)
            physics, target, condition = _on_device(batch.tensors, self.device)
            prediction = self.bundle.main(physics, condition)
            # Keep the reference operation order, including scalar weighting.
            loss = ((prediction - target) / self.scale).square().sum() * batch.weight / batch.count
            value = loss.detach().item()
            if not math.isfinite(value):
                raise ValueError("主趋势出现非有限损失")
            total += value
            if backward:
                loss.backward()
        return total


class _PreparedResidualLoss:
    def __init__(self, bundle, batches, batch_windows, batch_strategy, cancel, cache):
        self.bundle, self.batches = bundle, batches
        self.batch_windows, self.batch_strategy = batch_windows, batch_strategy
        self.output = bundle.config.residual_output
        self.cancel, self.device, self.stats = cancel, cache.device, cache.stats

    def __call__(self, backward):
        import torch
        from .metrics import quantile_loss
        if next(self.bundle.residual.parameters()).device != self.device:
            raise ValueError("残差设备已改变，须重新准备损失缓存")
        if self.bundle.config.residual_output != self.output:
            raise ValueError("残差输出定义已改变，须重新准备损失缓存")
        total = 0.
        for batch in self.batches:
            checkpoint(self.cancel)
            x, y, mask, condition, length_valid, weights = _on_device(batch.tensors, self.device)
            q = (self.bundle.residual.forward_quantiles(x, valid=length_valid, condition=condition)
                 if self.batch_strategy == "padded" else
                 self.bundle.residual.forward_quantiles(x, condition=condition))
            if self.batch_windows == 1:
                objective = (quantile_loss(q[mask], y[mask]) if self.output == "quantiles" else
                             (q[..., 1][mask] - y[mask]).square().mean())
                loss = objective * batch.weight * batch.scored / batch.count
            elif self.output == "quantiles":
                loss = quantile_loss(q, y, valid=mask, weights=weights)
            else:
                loss = torch.where(mask, (q[..., 1] - y).square(), 0.).mul(weights).sum()
            value = loss.detach().item()
            if not math.isfinite(value):
                raise ValueError("残差出现非有限损失")
            total += value
            if backward:
                loss.backward()
        return total


def prepare_main_loss(bundle, samples, scale, config, cancel=None, *, cache_device=False,
                      max_cache_bytes=DEFAULT_DEVICE_CACHE_BYTES):
    """Snapshot reference main batches; return a repeatable full-pass callable."""
    import torch
    cache = _TensorCache(bundle.device, cache_device, max_cache_bytes)
    eligible = [sample for sample in samples if sample.frame.valid_main.any()]
    batches = []
    for sample, weight in zip(eligible, _sample_weights(eligible)):
        frame = sample.frame.loc[sample.frame.valid_main]
        for start in range(0, len(frame), config.main_batch_points):
            checkpoint(cancel)
            part = frame.iloc[start:start + config.main_batch_points]
            condition = bundle.condition_values(sample.metadata, count=len(part))
            tensors = (torch.tensor(part[PHYSICS_COLUMNS].to_numpy(), dtype=torch.float32),
                       torch.tensor(part.power_cut_w.to_numpy(), dtype=torch.float32),
                       None if condition is None else torch.tensor(condition, dtype=torch.float32))
            batches.append(_MainBatch(cache.store(tensors), weight, len(frame)))
    return _PreparedMainLoss(bundle, batches, scale, cancel, cache)


def prepare_residual_loss(bundle, prepared, cancel=None, *, batch_windows=1,
                          batch_strategy="equal_length", cache_device=False,
                          max_cache_bytes=DEFAULT_DEVICE_CACHE_BYTES):
    """Pack the reference windows once, preserving masks and record weights.

    ``prepared`` is the result of ``training._residual_data``; this factory does
    not select or rebuild windows. Both legacy three-tensor windows and current
    windows carrying a separate score mask are accepted.
    """
    import torch
    from .neural import window_batches
    prepared = list(prepared)
    device = next(bundle.residual.parameters()).device
    cache = _TensorCache(device, cache_device, max_cache_bytes)
    weights = _sample_weights([sample for sample, _ in prepared])

    def weighted_windows():
        for (_, windows), weight in zip(prepared, weights):
            count = sum(int(window[3].sum()) if len(window) == 4 else window[1].numel()
                        for window in windows)
            for window in windows:
                mask = window[3] if len(window) == 4 else torch.ones_like(window[1], dtype=torch.bool)
                yield (*window[:3], mask, weight, count)

    batches = []
    for group in window_batches(weighted_windows(), batch_windows,
                                length=lambda item: item[0].shape[1], strategy=batch_strategy):
        checkpoint(cancel)
        lengths = [item[0].shape[1] for item in group]

        def packed(column):
            values = [item[column].squeeze(0).detach().cpu() for item in group]
            return torch.nn.utils.rnn.pad_sequence(values, batch_first=True)

        x, y, mask = packed(0), packed(1), packed(3)
        condition = None if group[0][2] is None else packed(2)
        length_valid = (torch.arange(x.shape[1])[None] < torch.as_tensor(lengths)[:, None]
                        if batch_strategy == "padded" else None)
        point_weights = (y.new_tensor([item[4] / item[5] for item in group])[:, None].expand_as(y)
                         if batch_windows != 1 else None)
        tensors = cache.store((x, y, mask, condition, length_valid, point_weights))
        batches.append(_ResidualBatch(tensors, group[0][4], group[0][5], int(mask.sum())))
    return _PreparedResidualLoss(bundle, batches, batch_windows, batch_strategy, cancel, cache)
