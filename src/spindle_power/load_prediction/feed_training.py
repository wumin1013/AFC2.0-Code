"""Independent actual-feed optimization and diagnostic program holdouts.

Formal round folds and full retraining are orchestrated by the shared service;
this low-level optimizer never selects or reads an independent test set.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np
from pandas.api.types import is_bool_dtype

from .common import checkpoint
from .feed_model import FeedModelConfig, _runtime, sequence_windows
from .feed_samples import FEED_FEATURE_COLUMNS, FEED_SAMPLE_SCHEMA, program_split
from .experiment_protocol import PROTOCOL_VERSION
from .sample_rules import SAMPLE_RULES_VERSION


@dataclass
class FeedTrainingConfig:
    epochs: int = 100
    learning_rate: float = .001
    batch_size: int = 16
    patience: int = 10
    min_delta: float = 1e-6
    gradient_clip: float = 1.
    seed: int = 42
    device: str = "cpu"

    def validate(self):
        if self.device not in ("cpu", "cuda"):
            raise ValueError("进给训练device只支持显式cpu或cuda")
        for key in ("epochs", "batch_size", "patience"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} 必须为正整数")
        for key in ("learning_rate", "gradient_clip"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} 必须为正有限数")
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("min_delta 必须为非负有限数")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed 必须为非负整数")


def _prepare(samples, assignment):
    samples = list(samples)
    development, reserved = [], {"memory_validation": [], "independent_test": []}
    blockers, source_roles = [], {}
    for sample in samples:
        role = sample.metadata.get("dataset_role", "development")
        source = sample.metadata.get("measurement_source")
        if source:
            identity = str(Path(source).resolve()).casefold()
            if identity in source_roles and source_roles[identity] != role:
                blockers.append("同一measurement_source不能跨开发、记忆验证和独立测试角色")
            source_roles[identity] = role
        if role == "development":
            development.append(sample)
        elif role in reserved:
            reserved[role].append(sample)
        else:
            raise ValueError(f"未知 dataset_role：{role}")
    try:
        groups = program_split(development, assignment)
    except ValueError as exc:
        groups = {key: [] for key in ("train", "validation", "test")}
        blockers.append(str(exc))
    if any(not groups.get(split) for split in ("train", "validation", "test")):
        blockers.append("需要按完整程序指定非空 train、validation 和 test")
    rounds = set()
    for index, sample in enumerate(development):
        meta, frame = sample.metadata, sample.frame
        round_id = meta.get("round_id")
        if type(round_id) is not int or not 1 <= round_id <= 90:
            blockers.append(f"记录{index + 1}缺少有效开发轮次1—90")
        else:
            rounds.add(round_id)
        for flag in ("round_complete", "complete", "alignment_reliable", "no_alarm"):
            if meta.get(flag) is not True:
                blockers.append(f"记录{index + 1}的{flag}未通过显式审核")
        if meta.get("tool_healthy") is False:
            blockers.append(f"记录{index + 1}刀具健康检查未通过")
        for key in ("machine_version", "machine_settings_version", "machining_mode"):
            if not isinstance(meta.get(key), str) or not meta[key].strip():
                blockers.append(f"记录{index + 1}缺少{key}")
        for key in ("feed_override", "spacing_mm"):
            value = meta.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                blockers.append(f"记录{index + 1}缺少有效{key}")
        if meta.get("schema") != FEED_SAMPLE_SCHEMA or tuple(meta.get("feature_columns", ())) != tuple(FEED_FEATURE_COLUMNS):
            blockers.append(f"记录{index + 1}的特征schema/顺序不匹配")
        if frame.empty or frame.columns.duplicated().any() or not set(FEED_FEATURE_COLUMNS) <= set(frame):
            blockers.append(f"记录{index + 1}缺少完整NC特征")
            continue
        if not np.isfinite(frame.loc[:, list(FEED_FEATURE_COLUMNS)].to_numpy(dtype=float)).all():
            blockers.append(f"记录{index + 1}存在无效NC特征")
        if "sequence_id" not in frame or frame.sequence_id.isna().any():
            blockers.append(f"记录{index + 1}缺少有效NC序列")
        if not {"feed_actual_mm_min", "valid_label"} <= set(frame) or not is_bool_dtype(frame.get("valid_label")):
            blockers.append(f"记录{index + 1}缺少实测进给标签/布尔遮罩")
        else:
            valid = frame.valid_label.fillna(False).to_numpy(dtype=bool)
            target = frame.feed_actual_mm_min.to_numpy(dtype=float)
            if not valid.any() or not np.isfinite(target[valid]).all() or np.any(target[valid] < 0):
                blockers.append(f"记录{index + 1}没有合法实测进给标签")
    missing = sorted(set(range(1, 91)) - rounds)
    if missing:
        blockers.insert(0, f"开发轮次未齐备，缺少：{missing}")
    manifest = {"action": "dry_run", "trained": False, "development_rounds": sorted(rounds),
                "missing_rounds": missing, "round_completeness": "requires_explicit_round_complete_true",
                "splits": {key: {"records": len(value), "program_ids": sorted({sample.metadata["program_id"] for sample in value})}
                           for key, value in groups.items()},
                "reserved_records": {key: len(value) for key, value in reserved.items()},
                "normalization_source": "train_only", "test_protocol": "development_program_holdout",
                "blockers": blockers, "ready_to_train": not blockers}
    return groups, manifest


def _record_weights(samples):
    groups = {}
    for sample in samples:
        meta = sample.metadata
        features = groups.setdefault(meta["round_id"], Counter())
        features[meta.get("feature", meta["program_id"])] += 1
    return [1. / len(groups) / len(groups[s.metadata["round_id"]]) /
            groups[s.metadata["round_id"]][s.metadata.get("feature", s.metadata["program_id"])] for s in samples]


def _normalization(samples):
    """Stream training records without concatenating complete experiments."""
    count, mean, m2, target_count, target_square = 0, None, None, 0, 0.
    for sample in samples:
        values = sample.frame.loc[:, list(FEED_FEATURE_COLUMNS)].to_numpy(dtype=float)
        n = len(values)
        local_mean = values.mean(axis=0)
        local_m2 = ((values - local_mean) ** 2).sum(axis=0)
        if count == 0:
            mean, m2 = local_mean, local_m2
        else:
            difference = local_mean - mean
            m2 = m2 + local_m2 + difference ** 2 * count * n / (count + n)
            mean = mean + difference * n / (count + n)
        count += n
        labels = sample.frame.loc[sample.frame.valid_label.fillna(False), "feed_actual_mm_min"].to_numpy(dtype=float)
        target_count += len(labels)
        target_square += float(np.dot(labels, labels))
    scale = np.sqrt(m2 / count)
    scale = np.where(scale > 1e-8, scale, 1.)
    return {"mean": mean.tolist(), "scale": scale.tolist(),
            "target_scale_mm_min": max(1., math.sqrt(target_square / target_count))}


def _evaluate(model, samples, runtime, cancel):
    if not samples:
        return None
    weights = _record_weights(samples)
    mae = mse = rmse = 0.
    records = []
    for sample, weight in zip(samples, weights):
        checkpoint(cancel)
        predicted = runtime.predict_values(model, sample, cancel=cancel)
        valid = sample.frame.valid_label.fillna(False).to_numpy(dtype=bool)
        error = predicted[valid] - sample.frame.feed_actual_mm_min.to_numpy(dtype=float)[valid]
        record_mae, record_mse = float(np.abs(error).mean()), float(np.square(error).mean())
        mae += weight * record_mae
        mse += weight * record_mse
        rmse += weight * math.sqrt(record_mse)
        records.append({"program_id": sample.metadata["program_id"], "round_id": sample.metadata["round_id"],
                        "mae_mm_min": record_mae, "rmse_mm_min": math.sqrt(record_mse), "valid_points": int(valid.sum())})
    return {"mae_mm_min": mae, "rmse_mm_min": rmse,
            "mse_normalized": mse / model.normalization["target_scale_mm_min"] ** 2,
            "weighting": "equal_round_then_equal_feature_then_equal_record", "records": records}


def train_feed_fold(train, validation=None, *, config=None, model_config=None,
                    execute=False, cancel=None, progress=None, checkpoint_path=None,
                    resume=False, checkpoint_context=None):
    """Train one supplied round fold, or fixed epochs when validation is None.

    The service owns 450-slot qualification. Here identity, source separation,
    labels and numerical contracts are rechecked before any optimizer update.
    Synthetic tests may intentionally contain fewer than 90 rounds.
    """
    config, model_config = config or FeedTrainingConfig(), model_config or FeedModelConfig()
    config.validate()
    model_config.validate()
    train, validation_records = list(train), [] if validation is None else list(validation)
    groups = {"train": train, "validation": validation_records, "test": []}
    blockers, identities = [], {}
    if not train or (validation is not None and not validation_records):
        blockers.append("训练组或显式验证组为空")
    train_rounds = {s.metadata.get("round_id") for s in train}
    val_rounds = {s.metadata.get("round_id") for s in validation_records}
    if train_rounds & val_rounds:
        blockers.append("同一计划轮次不能跨训练和验证集合")
    for split in ("train", "validation"):
        for sample in groups[split]:
            meta, frame = sample.metadata, sample.frame
            if meta.get("dataset_role", "development") != "development":
                blockers.append("独立测试/校准/适配/记忆验证数据禁止参与网络训练或早停")
            source = meta.get("measurement_content_sha256") or meta.get("measurement_sha256") or meta.get("measurement_source")
            if source and source in identities and identities[source] != split:
                blockers.append("同一实测来源不能跨训练与验证集合")
            if source:
                identities[source] = split
            if meta.get("schema") != FEED_SAMPLE_SCHEMA or tuple(meta.get("feature_columns", ())) != FEED_FEATURE_COLUMNS:
                blockers.append("进给样本特征版本或顺序不匹配")
            if frame.empty or not {"valid_label", "feed_actual_mm_min", "sequence_id", *FEED_FEATURE_COLUMNS} <= set(frame):
                blockers.append("缺少有效进给特征、序列或监督标签")
                continue
            if not is_bool_dtype(frame.valid_label):
                blockers.append("valid_label 必须为布尔遮罩")
                continue
            valid = frame.valid_label.fillna(False).to_numpy(bool)
            labels = frame.feed_actual_mm_min.to_numpy(float)
            if not valid.any() or not np.isfinite(labels[valid]).all() or (labels[valid] < 0).any():
                blockers.append("没有合法实测进给标签")
            if not np.isfinite(frame.loc[:, list(FEED_FEATURE_COLUMNS)].to_numpy(float)).all():
                blockers.append("进给特征包含无效数值")
    manifest = {"action": "dry_run", "trained": False, "protocol_version": PROTOCOL_VERSION,
                "sample_rules_version": SAMPLE_RULES_VERSION, "config": asdict(config),
                "model_config": asdict(model_config), "train_rounds": sorted(train_rounds, key=str),
                "validation_rounds": sorted(val_rounds, key=str), "normalization_source": "train_only",
                "training_mode": "fixed_epochs_full" if validation is None else "development_fold",
                "blockers": blockers, "ready_to_train": not blockers}
    if not execute:
        return manifest
    if blockers:
        raise ValueError("实际进给训练尚未就绪：" + "；".join(blockers[:12]))
    return _optimize(groups, manifest, config, model_config, cancel, progress,
                     checkpoint_path=checkpoint_path, resume=resume,
                     checkpoint_context=checkpoint_context)


def train_feed(samples, *, assignment, config=None, model_config=None, execute=False, cancel=None, progress=None):
    """Legacy complete-program diagnostic; formal entry points use round folds."""
    config, model_config = config or FeedTrainingConfig(), model_config or FeedModelConfig()
    config.validate()
    model_config.validate()
    groups, manifest = _prepare(samples, assignment)
    manifest.update(config=asdict(config), model_config=asdict(model_config))
    if not execute:
        return manifest
    if manifest["blockers"]:
        raise ValueError("实际进给训练尚未就绪：" + "；".join(manifest["blockers"][:12]))
    manifest.update(protocol="diagnostic_program_holdout", sample_rules_version=SAMPLE_RULES_VERSION)
    return _optimize(groups, manifest, config, model_config, cancel, progress)


def _optimize(groups, manifest, config, model_config, cancel, progress, *,
              checkpoint_path=None, resume=False, checkpoint_context=None):
    checkpoint(cancel)
    runtime = _runtime()
    import torch
    from .feed_device import configure_feed_device
    device, execution_identity = configure_feed_device(config.device)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = runtime.FeedModel(model_config)
    model.network.to(device)
    model.normalization = _normalization(groups["train"])
    for sample in groups["train"]:
        context = runtime.machine_context(sample.metadata)
        if context not in model.compatible_contexts:
            model.compatible_contexts.append(context)
    for samples_in_split in groups.values():
        for sample in samples_in_split:
            runtime.feature_values(sample)
            if runtime.machine_context(sample.metadata) not in model.compatible_contexts:
                raise ValueError("验证/测试的机床设置、倍率、模式或空间采样间距不在训练兼容域内")
    train_samples = groups["train"]
    weights = _record_weights(train_samples)
    windows = []
    for index, (sample, weight) in enumerate(zip(train_samples, weights)):
        count = int(sample.frame.valid_label.fillna(False).sum())
        for window in sequence_windows(sample.frame, model_config, sample.frame.valid_label.fillna(False).to_numpy(bool)):
            if sample.frame.valid_label.iloc[window.core_start:window.core_stop].fillna(False).any():
                windows.append((index, window, weight / count))
    optimizer = torch.optim.Adam(model.network.parameters(), lr=config.learning_rate)
    mean = np.asarray(model.normalization["mean"])
    scale = np.asarray(model.normalization["scale"])
    target_scale = model.normalization["target_scale_mm_min"]
    best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history = []
    from .training_checkpoint import training_identity, load_checkpoint, save_checkpoint
    identity = training_identity("feed", groups["train"], groups["validation"],
        dict(training=asdict(config), model=asdict(model_config)), context=checkpoint_context) if checkpoint_path else None
    saved = load_checkpoint(checkpoint_path, identity, resume=resume)
    start_epoch, completed = 1, False
    if saved is not None:
        model.network.load_state_dict(saved["weights"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            if not saved.get("cuda_rng"):
                raise ValueError("CUDA恢复缺少CUDA随机状态，拒绝静默兼容")
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        rng.bit_generator.state = saved["numpy_rng"]
        best_loss, best_epoch, best_state, stale = (saved[k] for k in ("best_loss", "best_epoch", "best_state", "stale"))
        history, start_epoch, completed = saved["history"], saved["epoch"] + 1, saved["completed"]
        if progress:
            progress(dict(phase="resumed", epoch=saved["epoch"], checkpoint=str(checkpoint_path), completed=completed))
    for epoch in range(start_epoch, config.epochs + 1) if not completed else ():
        checkpoint(cancel)
        model.network.train()
        optimizer.zero_grad(set_to_none=True)
        order = rng.permutation(len(windows))
        train_loss = 0.
        # Accumulate one equal-program objective; long files cannot dominate.
        for offset in range(0, len(order), config.batch_size):
            checkpoint(cancel)
            batch = [windows[int(item)] for item in order[offset:offset + config.batch_size]]
            arrays, labels, masks, point_weights = [], [], [], []
            for sample_index, window, weight in batch:
                frame = train_samples[sample_index].frame.iloc[window.start:window.stop]
                arrays.append((frame.loc[:, list(FEED_FEATURE_COLUMNS)].to_numpy(dtype=float) - mean) / scale)
                valid = frame.valid_label.fillna(False).to_numpy(dtype=bool)
                core = np.zeros(len(frame), dtype=bool)
                core[window.core_start - window.start:window.core_stop - window.start] = True
                valid &= core
                raw = frame.feed_actual_mm_min.to_numpy(dtype=float)
                labels.append(np.where(valid, raw / target_scale, 0.))
                masks.append(valid)
                point_weights.append(weight)
            x, lengths = runtime.padded_batch(arrays)
            y = torch.zeros(x.shape[:2], dtype=torch.float32)
            loss_weights = torch.zeros_like(y)
            for row, (label, mask, weight) in enumerate(zip(labels, masks, point_weights)):
                y[row, :len(label)] = torch.as_tensor(label, dtype=torch.float32)
                loss_weights[row, :len(mask)] = torch.as_tensor(mask, dtype=torch.float32) * weight
            x = x.to(device)
            y = y.to(device)
            loss_weights = loss_weights.to(device)
            prediction = model.network(x, lengths)
            loss = ((prediction - y).square() * loss_weights).sum()
            if not torch.isfinite(loss):
                raise ValueError("实际进给训练出现非有限损失")
            loss.backward()
            train_loss += float(loss.detach())
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.network.parameters(), config.gradient_clip)
        if not torch.isfinite(gradient_norm):
            raise ValueError("实际进给训练出现非有限梯度")
        optimizer.step()
        validation = _evaluate(model, groups["validation"], runtime, cancel)
        validation_loss = validation["mse_normalized"] if validation else None
        improved = validation is None or validation_loss < best_loss - config.min_delta
        if improved:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.network.state_dict().items()}
        else:
            stale += 1
        event = {"phase": "training", "epoch": epoch, "total_epochs": config.epochs,
                 "train_loss": train_loss, "validation_loss": validation_loss, "loss_units": "normalized_MSE",
                 "best_epoch": best_epoch, "early_stopping": validation is not None and stale >= config.patience}
        history.append(event)
        completed = epoch == config.epochs or (validation is not None and stale >= config.patience)
        save_checkpoint(checkpoint_path, identity, dict(epoch=epoch, completed=completed,
            weights=model.network.state_dict(), optimizer=optimizer.state_dict(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            numpy_rng=rng.bit_generator.state,
            best_loss=best_loss, best_epoch=best_epoch, best_state=best_state,
            stale=stale, history=history))
        if progress:
            progress(dict(event))
        if validation is not None and stale >= config.patience:
            break
    checkpoint(cancel)
    model.network.load_state_dict(best_state, strict=True)
    model.network.eval()
    for parameter in model.network.parameters():
        parameter.requires_grad_(False)
    # Held-out test is touched only after validation has selected all weights.
    test_metrics = _evaluate(model, groups["test"], runtime, cancel)
    model.metadata = {**manifest, "action": "trained", "training_status": "trained", "trained": True,
                      "best_epoch": best_epoch, "best_validation_loss": best_loss, "history": history,
                      "test_metrics": test_metrics, "device": config.device,
                      "execution_identity": execution_identity}
    model.trained = True
    model.model_version = model.current_version()
    model.assert_ready()
    if progress:
        progress({"phase": "completed", "best_epoch": best_epoch, "model_version": model.model_version,
                  "test_metrics": test_metrics})
    return model
