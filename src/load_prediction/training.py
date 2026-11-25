"""Staged physical/residual training; round development is in training_service."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .common import checkpoint
from .models import ModelConfig, build_load_models
from .samples import PHYSICS_COLUMNS, assert_feed_inputs, path_features, load_input_view
from .evaluation_policy import group_key


@dataclass
class TrainingConfig:
    main_epochs: int = 100
    residual_epochs: int = 100
    learning_rate: float = .001
    seed: int = 42
    main_batch_points: int = 8192
    patience: int = 10
    min_delta: float = 1e-6
    gradient_clip: float = 1.
    device: str = "cpu"
    residual_batch_windows: int = 1
    residual_batch_strategy: str = "equal_length"
    residual_window_policy: str = "legacy_v1"
    cpu_threads: int | None = None
    cpu_interop_threads: int | None = None
    loss_cache_mode: str = "none"
    loss_cache_max_bytes: int = 256 * 1024 * 1024  # Per train/validation loss callable.

    def validate(self):
        if self.device not in ("cpu", "cuda"):
            raise ValueError("负载训练device只支持显式cpu或cuda")
        from .load_device import validate_cpu_threads
        validate_cpu_threads(self.device, self.cpu_threads, self.cpu_interop_threads)
        if type(self.residual_batch_windows) is not int or not 1 <= self.residual_batch_windows <= 1024:
            raise ValueError("residual_batch_windows必须为1—1024的整数")
        if self.residual_batch_strategy not in ("equal_length", "padded"):
            raise ValueError("残差batch策略必须为equal_length或padded")
        if self.residual_window_policy not in ("legacy_v1", "prospective_v1"):
            raise ValueError("残差窗口策略必须为legacy_v1或prospective_v1")
        if self.loss_cache_mode not in ("none", "cpu", "device"):
            raise ValueError("loss_cache_mode必须为none、cpu或device")
        if type(self.loss_cache_max_bytes) is not int or self.loss_cache_max_bytes < 0:
            raise ValueError("loss_cache_max_bytes必须为非负整数")
        if self.loss_cache_mode == "device" and self.device != "cuda":
            raise ValueError("device损失缓存要求显式CUDA设备")
        for key in ("main_epochs", "residual_epochs", "main_batch_points", "patience"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} 必须为正整数")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0 or not np.isfinite(self.gradient_clip) or self.gradient_clip <= 0:
            raise ValueError("学习率和梯度裁剪必须为正有限数")
        if not np.isfinite(self.min_delta) or self.min_delta < 0 or type(self.seed) is not int or self.seed < 0:
            raise ValueError("min_delta和随机种子必须为有效非负值")


def _sample_weights(samples):
    """Each round, feature and complete record has an equal contribution."""
    grouped = {}
    for sample in samples:
        groups = grouped.setdefault(group_key(sample.metadata), {})
        feature = sample.metadata["feature"]
        groups[feature] = groups.get(feature, 0) + 1
    if not grouped:
        raise ValueError("没有通过质量筛选的训练样本")
    return [1. / len(grouped) / len(grouped[group_key(s.metadata)]) /
            grouped[group_key(s.metadata)][s.metadata["feature"]] for s in samples]


def _condition(bundle, sample, count):
    import torch
    value = bundle.condition_values(sample.metadata, count=count)
    return None if value is None else torch.as_tensor(value, dtype=torch.float32, device=bundle.device)


def _main_values(bundle, sample):
    import torch
    result = np.full(len(sample.frame), np.nan)
    good = np.isfinite(sample.frame[PHYSICS_COLUMNS]).all(axis=1).to_numpy()
    with torch.no_grad():
        if good.any():
            result[good] = bundle.main(torch.tensor(sample.frame.loc[good, PHYSICS_COLUMNS].to_numpy(), dtype=torch.float32, device=bundle.device),
                                        _condition(bundle, sample, int(good.sum()))).cpu().numpy()
    return result


def _main_loss(bundle, samples, scale, config, cancel, backward):
    import torch
    total = 0.
    eligible = [s for s in samples if s.frame.valid_main.any()]
    for sample, weight in zip(eligible, _sample_weights(eligible)):
        frame = sample.frame.loc[sample.frame.valid_main]
        for start in range(0, len(frame), config.main_batch_points):
            checkpoint(cancel)
            batch = frame.iloc[start:start + config.main_batch_points]
            pred = bundle.main(torch.tensor(batch[PHYSICS_COLUMNS].to_numpy(), dtype=torch.float32, device=bundle.device),
                               _condition(bundle, sample, len(batch)))
            target = torch.tensor(batch.power_cut_w.to_numpy(), dtype=torch.float32, device=bundle.device)
            loss = ((pred - target) / scale).square().sum() * weight / len(frame)
            if not torch.isfinite(loss):
                raise ValueError("主趋势出现非有限损失")
            total += loss.detach().item()
            if backward:
                loss.backward()
    return total


def _residual_data(bundle, samples, *, validation=False):
    if bundle.execution_window_policy() == "prospective_v1":
        return _prospective_residual_data(bundle, samples)
    import torch
    from .sample_rules import continuous_ranges
    prepared = []
    for sample in samples:
        frame = sample.frame.reset_index(drop=True)
        valid = frame.valid_residual.to_numpy(bool)
        if not valid.any():
            continue
        main = _main_values(bundle, sample)
        condition_values = bundle.condition_values(sample.metadata, count=len(frame))
        cond = None if condition_values is None else torch.as_tensor(condition_values, dtype=torch.float32)
        windows = []
        prospective = validation and bundle.config.route == "nominal_programmed_residual_v1"
        # Validation keeps deployment context: labels select the loss only.
        ranges = continuous_ranges(frame, np.isfinite(main) if prospective else valid, known_only=prospective)
        for a, b in ranges:
            for start in range(a, b, bundle.config.window_points):
                end = min(b, start + bundle.config.window_points)
                f = frame.iloc[start:end]
                x = path_features(f, main[start:end], bundle.normalization["main_mean_w"],
                                  bundle.normalization["main_scale_w"], bundle.config.feed_reference_mm_min,
                                  route=bundle.config.route)
                y = (f.power_cut_w.to_numpy() - main[start:end]) / bundle.normalization["residual_scale_w"]
                scored = valid[start:end]
                if not scored.any():
                    continue
                if not np.isfinite(y[scored]).all():
                    raise ValueError("有效残差监督含非有限目标")
                # Unscored values never enter pinball arithmetic (NaN*0 is NaN).
                y = np.where(scored, y, 0.)
                windows.append((torch.tensor(x, dtype=torch.float32)[None], torch.tensor(y, dtype=torch.float32)[None],
                                None if cond is None else cond[start:end][None], torch.as_tensor(scored)[None]))
        if windows:
            prepared.append((sample, windows))
    return prepared


def _prospective_residual_data(bundle, samples):
    """Labels select losses within complete deployment windows, never context."""
    import torch
    from .neural import prospective_residual_context
    from .samples import SampleSet, physical_quantities
    prepared = []
    for sample in samples:
        view = load_input_view(sample, "programmed")
        selected, indices, planned = prospective_residual_context(view.frame, bundle.config.window_points)
        known = np.zeros(len(view.frame), dtype=bool)
        known[indices] = True
        if np.any(view.frame.valid_residual.to_numpy(bool) & ~known):
            raise ValueError("有效残差监督缺少完整加工前输入")
        if selected.empty:
            continue
        assert_feed_inputs(selected, "programmed")
        physical_quantities(selected[PHYSICS_COLUMNS].to_numpy(np.float32))
        # Match deployment's selected tensor shape and float32 main arithmetic.
        main = _main_values(bundle, SampleSet(selected, sample.metadata)).astype(np.float32)
        features = path_features(selected, main, bundle.normalization["main_mean_w"],
            bundle.normalization["main_scale_w"], bundle.config.feed_reference_mm_min, route=bundle.config.route)
        x = torch.as_tensor(features, dtype=torch.float32)
        condition = bundle.condition_values(sample.metadata, count=len(selected))
        cond = None if condition is None else torch.as_tensor(condition, dtype=torch.float32)
        scored = selected.valid_residual.to_numpy(bool)
        target = (selected.power_cut_w.to_numpy() - main) / bundle.normalization["residual_scale_w"]
        if not np.isfinite(target[scored]).all():
            raise ValueError("有效残差监督含非有限目标")
        y = torch.as_tensor(np.where(scored, target, 0.), dtype=torch.float32)
        mask = torch.as_tensor(scored)
        windows = [(x[index][None], y[index][None], None if cond is None else cond[index][None], mask[index][None])
                   for index in planned if scored[index].any()]
        if windows:
            prepared.append((sample, windows))
    return prepared


def _residual_loss(bundle, prepared, cancel, backward, *, batch_windows=1, batch_strategy="equal_length"):
    import torch
    from .metrics import quantile_loss
    from .neural import window_batches
    total = 0.
    weights = _sample_weights([s for s, _ in prepared])
    device = next(bundle.residual.parameters()).device
    def weighted_windows():
        for (_, windows), weight in zip(prepared, weights):
            count = sum(int(window[3].sum()) if len(window) == 4 else window[1].numel() for window in windows)
            for window in windows:
                mask = window[3] if len(window) == 4 else torch.ones_like(window[1], dtype=torch.bool)
                yield (*window[:3], mask, weight, count)
    for group in window_batches(weighted_windows(), batch_windows, length=lambda item: item[0].shape[1], strategy=batch_strategy):
        checkpoint(cancel)
        lengths = [item[0].shape[1] for item in group]
        def packed(column):
            values = [item[column].squeeze(0) for item in group]
            return torch.nn.utils.rnn.pad_sequence(values, batch_first=True).to(device)
        x, y, mask = packed(0), packed(1), packed(3)
        condition = None if group[0][2] is None else packed(2)
        # Padding describes declared sequence length only; score mask is never
        # supplied to the network, including during validation early stopping.
        length_valid = torch.arange(x.shape[1], device=device)[None] < torch.as_tensor(lengths, device=device)[:, None]
        q = bundle.residual.forward_quantiles(x, valid=length_valid, condition=condition) if batch_strategy == "padded" else bundle.residual.forward_quantiles(x, condition=condition)
        if batch_windows == 1:
            # Preserve the existing serial arithmetic, including legacy tests.
            objective = quantile_loss(q[mask], y[mask]) if bundle.config.residual_output == "quantiles" else (q[..., 1][mask] - y[mask]).square().mean()
            loss = objective * group[0][4] * int(mask.sum()) / group[0][5]
        else:
            point_weights = y.new_tensor([item[4] / item[5] for item in group])[:, None].expand_as(y)
            if bundle.config.residual_output == "quantiles":
                loss = quantile_loss(q, y, valid=mask, weights=point_weights)
            else:
                loss = torch.where(mask, (q[..., 1] - y).square(), 0.).mul(point_weights).sum()
        if not torch.isfinite(loss):
            raise ValueError("残差出现非有限损失")
        total += loss.detach().item()
        if backward:
            loss.backward()
    return total


def _epochs(module, maximum, config, train_loss, validation_loss, stage, cancel, progress, *,
            checkpoint_path=None, checkpoint_identity=None, resume=False):
    import torch
    from .training_checkpoint import load_checkpoint, save_checkpoint
    from .load_device import restore_rng, rng_state
    optimizer = torch.optim.Adam([p for p in module.parameters() if p.requires_grad], lr=config.learning_rate)
    best, best_epoch, stale, state, history = float("inf"), 0, 0, None, []
    saved = load_checkpoint(checkpoint_path, checkpoint_identity, resume=resume)
    start_epoch, completed = 1, False
    if saved is not None:
        module.load_state_dict(saved["weights"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved, config.device)
        best, best_epoch, stale, state, history = (saved[k] for k in ("best_loss", "best_epoch", "stale", "best_state", "history"))
        start_epoch, completed = saved["epoch"] + 1, saved["completed"]
        if progress:
            progress(dict(stage=stage, phase="resumed", epoch=saved["epoch"], completed=completed))
    for epoch in range(start_epoch, maximum + 1) if not completed else ():
        checkpoint(cancel)
        module.train()
        optimizer.zero_grad()
        train = train_loss(True)
        gradient_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), config.gradient_clip)
        if not torch.isfinite(gradient_norm):
            raise ValueError("训练出现非有限梯度")
        optimizer.step()
        module.eval()
        with torch.no_grad():
            value = validation_loss(False) if validation_loss else train
        history.append(dict(epoch=epoch, train_loss=train, validation_loss=value if validation_loss else None))
        if validation_loss:
            if value < best - config.min_delta:
                best, best_epoch, stale, state = value, epoch, 0, deepcopy(module.state_dict())
            else:
                stale += 1
        else:
            best_epoch = epoch
        completed = epoch == maximum or (validation_loss is not None and stale >= config.patience)
        save_checkpoint(checkpoint_path, checkpoint_identity, dict(epoch=epoch, completed=completed,
            weights=module.state_dict(), optimizer=optimizer.state_dict(), **rng_state(config.device),
            optimizer_updates=epoch,
            best_loss=best, best_epoch=best_epoch, stale=stale, best_state=state, history=history))
        if progress:
            progress(dict(stage=stage, **history[-1]))
        if completed:
            break
    checkpoint(cancel)
    if state is not None:
        module.load_state_dict(state)
    return {"best_epoch": best_epoch, "history": history, "early_stopping": validation_loss is not None}


def train_two_stage(samples, *, validation=None, model_config=None, config=None, execute=False, cancel=None, progress=None,
                    main_only=False, checkpoint_dir=None, resume=False, checkpoint_context=None):
    """Identify physical coefficients, freeze them, then fit the selected residual route."""
    model_config, config = model_config or ModelConfig(), config or TrainingConfig()
    config.validate()
    model_config.validate()
    if config.residual_window_policy == "prospective_v1" and model_config.route != "nominal_programmed_residual_v1":
        raise ValueError("加工前残差窗口仅支持名义编程进给路线")
    samples, validation = list(samples), list(validation or [])
    manifest = dict(action="dry_run", main_samples=sum(int(s.frame.valid_main.sum()) for s in samples),
                    residual_samples=sum(int(s.frame.valid_residual.sum()) for s in samples),
                    rounds=sorted({s.metadata["round_id"] for s in samples if s.metadata.get("round_id") is not None}),
                    order=["main", "freeze_main"] if main_only else ["main", "freeze_main", "labels", "residual_quantiles", "freeze_residual"],
                    config=asdict(config), model_config=asdict(model_config))
    if any(s.metadata.get("group_id") is not None for s in samples):
        manifest["statistical_groups"] = sorted({group_key(s.metadata) for s in samples}, key=str)
    if not execute:
        return manifest
    import torch
    from .neural import freeze, state_version
    nominal = model_config.route == "nominal_programmed_residual_v1"
    measured, measured_validation = samples, validation
    if nominal:
        measured = [load_input_view(s, "measured") for s in samples]
        measured_validation = [load_input_view(s, "measured") for s in validation]
        samples = [load_input_view(s, "programmed") for s in samples]
        validation = [load_input_view(s, "programmed") for s in validation]
    for sample in measured + measured_validation:
        selected = sample.frame.valid_main | sample.frame.valid_residual
        if selected.any():
            assert_feed_inputs(sample.frame.loc[selected], "measured")
    if not manifest["main_samples"] or (not main_only and not manifest["residual_samples"]):
        raise ValueError("缺少合格主趋势或残差样本")
    torch.manual_seed(config.seed)
    from .load_device import configure_load_device
    _, execution_identity = configure_load_device(config.device, cpu_threads=config.cpu_threads,
                                                  cpu_interop_threads=config.cpu_interop_threads)
    bundle = build_load_models(model_config, device=config.device)
    if config.residual_window_policy == "prospective_v1":
        bundle.metadata["config"] = asdict(config)
    targets = np.concatenate([s.frame.loc[s.frame.valid_main, "power_cut_w"].to_numpy() for s in samples if s.frame.valid_main.any()])
    scale = max(1., float(np.std(targets)))
    from .training_checkpoint import training_identity
    identity = training_identity("load", samples, validation,
        dict(training=asdict(config), model=asdict(model_config), main_only=main_only),
        context=dict(caller=checkpoint_context, load_execution=execution_identity)) if checkpoint_dir else None
    if config.loss_cache_mode == "none":
        main_train_loss = lambda back: _main_loss(bundle, measured, scale, config, cancel, back)
        main_validation_loss = ((lambda back: _main_loss(bundle, measured_validation, scale, config, cancel, back))
                                if validation else None)
    else:
        # Delayed import avoids a cycle through training_cache._sample_weights.
        from .training_cache import prepare_main_loss, prepare_residual_loss
        cache_options = dict(cache_device=config.loss_cache_mode == "device",
                             max_cache_bytes=config.loss_cache_max_bytes)
        main_train_loss = prepare_main_loss(bundle, measured, scale, config, cancel, **cache_options)
        main_validation_loss = (prepare_main_loss(bundle, measured_validation, scale, config, cancel, **cache_options)
                                if validation else None)
    main_run = _epochs(bundle.main, config.main_epochs, config,
                       main_train_loss, main_validation_loss,
                       "main", cancel, progress,
                       checkpoint_path=Path(checkpoint_dir) / "main.pt" if checkpoint_dir else None,
                       checkpoint_identity=identity, resume=resume)
    del main_train_loss, main_validation_loss
    freeze(bundle.main)
    bundle.main_trained, bundle.main_version = True, state_version(bundle.main)
    values, residuals = [], []
    for sample in samples:
        main = _main_values(bundle, sample)
        good = sample.frame["valid_main" if main_only else "valid_residual"].to_numpy(bool)
        values.extend(main[good])
        residuals.extend(sample.frame.power_cut_w.to_numpy()[good] - main[good])
    bundle.normalization = dict(main_mean_w=float(np.mean(values)), main_scale_w=max(1., float(np.std(values))),
                                residual_scale_w=max(1., float(np.std(residuals))), target_scale_w=scale,
                                high_load_threshold_w=float(np.quantile(targets, .9)),
                                low_load_threshold_w=float(np.quantile(targets, .1)))
    if main_only:
        freeze(bundle.residual)
        bundle.metadata = {**manifest, "action": "trained_main_baseline", "samples": [s.metadata for s in samples],
                           "load_execution": execution_identity,
                           "stages": {"main": main_run, "residual": {"status": "not_trained", "best_epoch": 0, "history": []}},
                           "interval_status": "physical_baseline_no_interval", "quantiles": None}
        return bundle
    prepared, val_prepared = _residual_data(bundle, samples), _residual_data(bundle, validation, validation=True)
    frozen_version = bundle.main_version
    if config.loss_cache_mode == "none":
        residual_train_loss = lambda back: _residual_loss(bundle, prepared, cancel, back,
            batch_windows=config.residual_batch_windows, batch_strategy=config.residual_batch_strategy)
        residual_validation_loss = ((lambda back: _residual_loss(bundle, val_prepared, cancel, back,
            batch_windows=config.residual_batch_windows, batch_strategy=config.residual_batch_strategy))
            if validation else None)
    else:
        residual_train_loss = prepare_residual_loss(bundle, prepared, cancel,
            batch_windows=config.residual_batch_windows, batch_strategy=config.residual_batch_strategy, **cache_options)
        residual_validation_loss = (prepare_residual_loss(bundle, val_prepared, cancel,
            batch_windows=config.residual_batch_windows, batch_strategy=config.residual_batch_strategy, **cache_options)
            if validation else None)
    residual_run = _epochs(bundle.residual, config.residual_epochs, config,
                           residual_train_loss, residual_validation_loss,
                           "residual", cancel, progress,
                           checkpoint_path=Path(checkpoint_dir) / "residual.pt" if checkpoint_dir else None,
                           checkpoint_identity={**identity, "main_version": bundle.main_version,
                                                "normalization": bundle.normalization} if identity else None,
                           resume=resume)
    del residual_train_loss, residual_validation_loss
    if state_version(bundle.main) != frozen_version:
        raise RuntimeError("残差训练修改了冻结主趋势")
    freeze(bundle.residual)
    bundle.residual_trained = True
    bundle.residual_version = state_version(bundle.residual)
    bundle.residual_main_version = bundle.main_version
    bundle.metadata = {**manifest, "action": "trained", "samples": [s.metadata for s in samples],
                       "load_execution": execution_identity,
                       "stages": {"main": main_run, "residual": residual_run},
                       "interval_status": "model_prediction_interval" if model_config.residual_output == "quantiles" else "point_regression_no_interval",
                       "quantiles": [.05, .5, .95] if model_config.residual_output == "quantiles" else None}
    return bundle


def save_main_baseline(bundle, path):
    """A main-only ablation is not a trained/registrable load-model package."""
    from .training_checkpoint import save_checkpoint
    from .neural import state_version
    if not bundle.main_trained or bundle.residual_trained or bundle.main_version != state_version(bundle.main):
        raise ValueError("物理基线必须仅有冻结主趋势")
    identity = dict(kind="main_baseline", config=asdict(bundle.config), main_version=bundle.main_version)
    save_checkpoint(path, identity, dict(epoch=bundle.metadata["stages"]["main"]["best_epoch"], completed=True,
        artifact_schema="afc.main_baseline.1", weights=bundle.main.state_dict(),
        normalization=bundle.normalization, metadata=bundle.metadata))


def run_grouped_development(samples, **kwargs):
    """Compatibility entry delegates to the only official protocol implementation."""
    from .training_service import run_development
    return run_development(load_samples=samples, **kwargs)
