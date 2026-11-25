"""Executable target-condition comparisons with one data and freeze contract.

Adapters operate on copies, reuse staged training, and never publish a formal
model. Independent test records are evaluated only after adaptation freezes.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import re

import numpy as np

from .common import checkpoint, save_json, version_of
from .condition import PROVENANCE_FIELDS, descriptors, encode_conditions, fit_condition_spec
from .samples import PHYSICS_COLUMNS, assert_feed_inputs
from .training import TrainingConfig

TARGET_EXPERIMENTS = ("zero_shot", "four_parameter_calibration", "physical_refit",
                      "output_finetune", "full_retrain")
SCHEMA = "afc.target_experiment.1"
_ROLES = {"source": "development", "adaptation": "target_adaptation",
          "validation": "adaptation_validation", "test": "independent_test"}


def _physical_signature(metadata):
    value = descriptors(metadata)
    for branch, key in PROVENANCE_FIELDS:
        value[branch].pop(key, None)
    return version_of(value), any(value.values())


def _preflight(groups, *, synthetic, qualification):
    """Bad isolation is an error; absent evidence is a waiting state."""
    seen_ids, seen_hashes, missing = {}, {}, []
    source_conditions, target_conditions = set(), set()
    for group, samples in groups.items():
        for sample in samples:
            meta, frame = sample.metadata, sample.frame
            identity, digest = meta.get("record_id"), meta.get("measurement_content_sha256")
            if not isinstance(identity, str) or not identity.strip():
                missing.append(f"{group}:缺少record_id")
                continue
            if identity in seen_ids:
                raise ValueError(f"record_id在拟合/验证/测试间重复：{identity}")
            seen_ids[identity] = group
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
                missing.append(f"{identity}:缺少实际测量内容SHA-256")
            else:
                digest = digest.lower()
                if digest in seen_hashes:
                    raise ValueError(f"实际测量内容hash重复：{identity}与{seen_hashes[digest]}")
                seen_hashes[digest] = identity
            if meta.get("dataset_role") != _ROLES[group]:
                raise ValueError(f"{identity}用途必须为{_ROLES[group]}")
            if synthetic and "synthetic" not in str(meta.get("provenance", "")):
                raise ValueError("synthetic开关不能豁免真实记录的资格审核")
            if not synthetic and "synthetic" in str(meta.get("provenance", "")):
                missing.append(f"{identity}:合成记录不能作为真实工况证据")
            if any(meta.get(key) is not True for key in ("complete", "tool_healthy", "no_alarm", "alignment_reliable")):
                missing.append(f"{identity}:质量未显式通过")
            needed = set(PHYSICS_COLUMNS + ["power_cut_w", "valid_main", "valid_residual"])
            if needed - set(frame):
                missing.append(f"{identity}:缺少物理量/标签/掩码")
                continue
            residual = frame.valid_residual.to_numpy(bool)
            steady = frame.valid_main.to_numpy(bool)
            if not residual.any() or (group in ("source", "adaptation") and not steady.any()):
                missing.append(f"{identity}:缺少合格切削/稳态标签")
            if np.any(steady & ~residual):
                raise ValueError(f"{identity}:稳态必须属于有效切削")
            if residual.any():
                assert_feed_inputs(frame.loc[residual], "measured")
                if not np.isfinite(frame.loc[residual, PHYSICS_COLUMNS + ["power_cut_w"]].to_numpy(float)).all():
                    raise ValueError(f"{identity}:有效样本含非有限值")
            signature, present = _physical_signature(meta)
            if not present:
                missing.append(f"{identity}:缺少物理工况描述")
            (source_conditions if group == "source" else target_conditions).add(signature)
    if not synthetic:
        for key in ("source_data_verified", "target_data_verified", "base_labels_verified"):
            if (qualification or {}).get(key) is not True:
                missing.append(key)
    if len(source_conditions) < 2:
        return "awaiting_multicondition_data", ["源数据尚不足两个真实物理工况"] + missing
    if len(target_conditions) > 1:
        raise ValueError("一次目标工况对照只能使用一个固定目标组合")
    if source_conditions & target_conditions:
        raise ValueError("新工况对照的目标组合已存在于源拟合数据")
    return ("awaiting_data", missing) if missing else (None, [])


def _verify_training_history(bundle, target_samples):
    """Also reject target reuse hidden in the frozen package's fit history."""
    prior_ids, prior_hashes = set(), set(bundle.metadata.get("training_record_hashes", []))
    for meta in bundle.metadata.get("samples", []):
        if isinstance(meta, dict):
            if meta.get("record_id"):
                prior_ids.add(meta["record_id"])
            if meta.get("measurement_content_sha256"):
                prior_hashes.add(meta["measurement_content_sha256"].lower())
    history = bundle.metadata.get("target_adapter_history", []) + [bundle.metadata.get("target_adapter", {})]
    for previous in history:
        prior_ids.update(previous.get("fitting_records", []))
        prior_ids.update(previous.get("validation_records", []))
        prior_hashes.update(previous.get("fitting_content_hashes", []))
        prior_hashes.update(previous.get("validation_content_hashes", []))
    for sample in target_samples:
        meta = sample.metadata
        if meta.get("record_id") in prior_ids or str(meta.get("measurement_content_sha256", "")).lower() in prior_hashes:
            raise ValueError("目标适配/验证/测试记录已进入冻结源模型训练")


def _parameter_state(bundle):
    return {f"{branch}.{name}": value.detach().cpu().clone()
            for branch in ("main", "residual") for name, value in getattr(bundle, branch).named_parameters()}


def _changed(before, after):
    import torch
    return sorted(key for key in before.keys() | after.keys()
                  if key not in before or key not in after or before[key].shape != after[key].shape
                  or not torch.equal(before[key], after[key]))


def _partial_adaptation(name, model, fitted, validation, config, cancel, progress):
    """Use the same weighted losses, optimizer and early stopping as development."""
    from .neural import freeze, state_version
    from .training import _epochs, _main_loss, _residual_data, _residual_loss
    freeze(model.main)
    freeze(model.residual)
    if name == "physical_refit":
        for parameter in model.main.coefficients.parameters():
            parameter.requires_grad_(True)
        if model.config.physical_mode == "fixed":
            model.main.coefficients.weight.requires_grad_(False)
        # Encoders remain fixed; only the final physical multiplier heads adapt.
        for branch_name, network in model.main.condition_effects.branches.items():
            if model.config.condition_spec["branches"][branch_name]["active"]:
                for parameter in network[-1].parameters():
                    parameter.requires_grad_(True)
        trainable = [f"main.{key}" for key, value in model.main.named_parameters() if value.requires_grad]
        scale = model.normalization.get("target_scale_w", model.normalization["main_scale_w"])
        run = _epochs(model.main, config.main_epochs, config,
                      lambda back: _main_loss(model, fitted, scale, config, cancel, back),
                      lambda back: _main_loss(model, validation, scale, config, cancel, back),
                      "target_physical_refit", cancel, progress)
        old_main = model.main_version
        model.main_version = state_version(model.main)
        # This is a declared adapter, not a claim that old residual labels were retrained.
        model.metadata["residual_training_main_version"] = old_main
        model.metadata["adapted_main_with_frozen_residual"] = True
        model.residual_main_version = model.main_version
    else:
        for parameter in model.residual.output.parameters():
            parameter.requires_grad_(True)
        if model.config.residual_output == "quantiles":
            for parameter in model.residual.spread.parameters():
                parameter.requires_grad_(True)
        trainable = [f"residual.{key}" for key, value in model.residual.named_parameters() if value.requires_grad]
        prepared, checked = _residual_data(model, fitted), _residual_data(model, validation)
        run = _epochs(model.residual, config.residual_epochs, config,
                      lambda back: _residual_loss(model, prepared, cancel, back),
                      lambda back: _residual_loss(model, checked, cancel, back),
                      "target_output_finetune", cancel, progress)
    freeze(model.main)
    freeze(model.residual)
    model.residual_version = state_version(model.residual)
    return trainable, run


def _evaluate(model, samples, feed_index, feed, calibration, *, cancel, progress, phase):
    from .metrics import regression_metrics
    from .models import predict_cutting_power
    from .sample_rules import continuous_ids
    from .training_service import _predicted_frame, balanced_metrics
    rows = []
    for sample in samples:
        checkpoint(cancel)
        identity = sample.metadata["record_id"]
        frame = _predicted_frame(sample, feed_index[identity], feed)
        valid = frame.valid_residual.to_numpy(bool)
        frame["sequence_id"] = continuous_ids(frame, valid, known_only=True)
        frame = frame.loc[valid].reset_index(drop=True)
        result = predict_cutting_power(model, frame, context=sample.metadata, calibration=calibration)
        interval = model.config.residual_output == "quantiles"
        metrics = regression_metrics(frame.power_cut_w.to_numpy(), result["cutting_power_w"],
                    lower=result["q05_w"] if interval else None, upper=result["q95_w"] if interval else None,
                    scale_w=model.normalization.get("target_scale_w", model.normalization["main_scale_w"]),
                    high_threshold_w=model.normalization.get("high_load_threshold_w"),
                    low_threshold_w=model.normalization.get("low_load_threshold_w"))
        rows.append(dict(record_id=identity, round_id=sample.metadata["round_id"], feature=sample.metadata["feature"],
                         feed_mode="predicted", metrics=metrics,
                         prediction_fingerprint=version_of(result["cutting_quantiles_w"])))
        if progress:
            progress(dict(stage=phase, record_id=identity, completed_records=len(rows), total_records=len(samples)))
    return dict(records=rows, equal_weight_metrics=balanced_metrics(rows), feed_mode="predicted", memory_updates=0)


def run_target_experiment(name, *, feed_model, bundle, source_samples, adaptation_samples=(),
                          validation_samples=(), test_samples=(), feed_samples=(), training_config=None,
                          calibration_config=None, execute=False, synthetic=False, qualification=None,
                          cancel=None, progress=None, output_dir=None):
    """Run one isolated target comparison and return a JSON-compatible report.

    ``feed_samples`` pairs validation/test load records by record_id, content
    hash and NC position. ``full_retrain`` restarts the load networks on source
    plus target fitting records; the given feed model stays frozen. Real runs
    additionally require explicit source/target/base-label qualification.
    Optional output_dir receives frozen adapter weights and report, never a
    formal registry entry. Synthetic runs require synthetic record provenance.
    """
    if name not in TARGET_EXPERIMENTS:
        raise ValueError(f"不支持的目标工况对照：{name}")
    checkpoint(cancel)
    groups = {"source": list(source_samples), "adaptation": list(adaptation_samples),
              "validation": list(validation_samples), "test": list(test_samples)}
    report = dict(schema=SCHEMA, applied_experiment=name, status="prepared_not_executed", executed=False,
                  synthetic=bool(synthetic), real_effect_established=False,
                  counts={key: len(value) for key, value in groups.items()},
                  test_used_for_selection=False, memory_updates=0)
    if not groups["source"] or not groups["test"] or not groups["validation"]:
        return {**report, "status": "awaiting_data", "missing": ["source/独立adaptation_validation/independent_test均须非空"]}
    if name != "zero_shot" and len(groups["adaptation"]) < 2:
        return {**report, "status": "awaiting_data", "missing": ["目标适配至少两份独立记录"]}
    status, missing = _preflight(groups, synthetic=synthetic, qualification=qualification)
    if status:
        return {**report, "status": status, "missing": missing}
    target = groups["adaptation"] + groups["validation"] + groups["test"]
    bundle.assert_ready()
    feed_model.assert_ready()
    _verify_training_history(bundle, target)
    feed_index = {}
    for sample in feed_samples:
        identity = sample.metadata.get("record_id")
        if identity in feed_index:
            raise ValueError("进给配套记录ID重复")
        feed_index[identity] = sample
    for sample in groups["validation"] + groups["test"]:
        identity = sample.metadata["record_id"]
        if identity not in feed_index:
            return {**report, "status": "awaiting_data", "missing": [f"{identity}:缺少NC进给配套"]}
        paired = feed_index[identity].metadata
        if (paired.get("measurement_content_sha256") != sample.metadata["measurement_content_sha256"]
                or paired.get("dataset_role") != sample.metadata["dataset_role"]):
            raise ValueError("负载与进给配套的实际内容hash或用途不一致")
        from .feed_neural import machine_context
        if machine_context(paired) not in feed_model.compatible_contexts:
            return {**report, "status": "awaiting_compatible_feed_model", "missing": [f"{identity}:冻结BiLSTM不支持目标机床/设置域"]}
    if name not in ("four_parameter_calibration", "full_retrain") and bundle.config.condition_spec is not None:
        try:
            encode_conditions([s.metadata for s in target], bundle.config.condition_spec)
        except ValueError as exc:
            return {**report, "status": "awaiting_supported_target_descriptor", "missing": [str(exc)]}
    config = deepcopy(training_config) if isinstance(training_config, TrainingConfig) else TrainingConfig(**(training_config or {}))
    config.validate()
    report.update(training_config=asdict(config), source_model_version=bundle.current_version(),
                  feed_model_version=feed_model.current_version(),
                  fitting_records=[] if name == "zero_shot" else [s.metadata["record_id"] for s in groups["adaptation"]],
                  fitting_source_records=[s.metadata["record_id"] for s in groups["source"]] if name == "full_retrain" else [],
                  validation_records=[s.metadata["record_id"] for s in groups["validation"]],
                  independent_test_records=[s.metadata["record_id"] for s in groups["test"]],
                  test_policy="freeze_all_weights_then_evaluate_once_no_test_selection",
                  fit_feed_mode="measured_actual_feed", evaluation_feed_mode="bilstm_actual_feed",
                  equal_target_budget_reference=name != "full_retrain",
                  record_provenance={key: [{"record_id": s.metadata["record_id"],
                      "measurement_content_sha256": s.metadata["measurement_content_sha256"]} for s in values]
                      for key, values in groups.items()})
    if not execute:
        return report
    source_version, feed_version = bundle.current_version(), feed_model.current_version()
    model, feed = deepcopy(bundle), deepcopy(feed_model)
    before = _parameter_state(model)
    calibration, trainable, adaptation_run = None, [], None
    if progress:
        progress(dict(stage="target_adaptation", experiment=name, status="started"))
    try:
        checkpoint(cancel)
        if name == "four_parameter_calibration":
            from .calibration import calibrate_model
            # The small SciPy fit is atomic; cancellation is checked before/after it.
            try:
                calibration = calibrate_model(model, groups["adaptation"], **(calibration_config or {}))
            except ValueError as exc:
                if "辨识" not in str(exc):
                    raise
                checkpoint(cancel)
                return {**report, "status": "awaiting_identifiable_target_data", "missing": [str(exc)]}
            adaptation_run = calibration.report
        elif name in ("physical_refit", "output_finetune"):
            trainable, adaptation_run = _partial_adaptation(name, model, groups["adaptation"],
                    groups["validation"], config, cancel, progress)
        elif name == "full_retrain":
            from .training import train_two_stage
            fitted = deepcopy(groups["source"] + groups["adaptation"])
            # Source and target round numbers are separate acquisition namespaces.
            for sample in fitted:
                sample.metadata["round_id"] = f'{sample.metadata["dataset_role"]}:{sample.metadata["round_id"]}'
            spec = fit_condition_spec([sample.metadata for sample in fitted])
            model = train_two_stage(fitted, validation=groups["validation"],
                    model_config=replace(deepcopy(bundle.config), condition_spec=spec), config=config,
                    execute=True, cancel=cancel, progress=progress)
            adaptation_run = model.metadata["stages"]
            trainable = [f"{branch}.{key}" for branch in ("main", "residual")
                         for key, _ in getattr(model, branch).named_parameters()]
        checkpoint(cancel)
        from .neural import freeze
        freeze(model.main)
        freeze(model.residual)
        freeze(feed.network)
        if model.metadata.get("target_adapter"):
            model.metadata.setdefault("target_adapter_history", []).append(deepcopy(model.metadata["target_adapter"]))
        model.metadata["target_adapter"] = dict(experiment=name, source_model_version=source_version,
            fitting_records=report["fitting_records"], validation_records=report["validation_records"],
            fitting_content_hashes=[s.metadata["measurement_content_sha256"].lower() for s in groups["adaptation"]] if name != "zero_shot" else [],
            validation_content_hashes=[s.metadata["measurement_content_sha256"].lower() for s in groups["validation"]],
            residual_policy="frozen_residual_recomputed_with_adapted_main" if name == "physical_refit" else "unchanged_or_explicitly_trained")
        model.assert_ready()
        frozen_version = model.current_version()
        if progress:
            progress(dict(stage="target_model_frozen", experiment=name, model_version=frozen_version))
        validation = _evaluate(model, groups["validation"], feed_index, feed, calibration,
                                cancel=cancel, progress=progress, phase="adaptation_validation")
        test = _evaluate(model, groups["test"], feed_index, feed, calibration,
                         cancel=cancel, progress=progress, phase="independent_test")
        if model.current_version() != frozen_version or feed.current_version() != feed_version:
            raise RuntimeError("目标测试阶段意外改变冻结模型")
        report.update(status="evaluated", executed=True, model_version=frozen_version,
                      source_weights_unchanged=True, feed_weights_unchanged=True,
                      trainable_parameters=trainable, changed_parameters=_changed(before, _parameter_state(model)),
                      adaptation_run=adaptation_run, validation=validation, independent_test=test,
                      calibration_version=calibration.version if calibration is not None else None,
                      validation_policy="early_stopping_then_frozen_evaluation" if name in
                        ("physical_refit", "output_finetune", "full_retrain") else "frozen_evaluation_only")
        checkpoint(cancel)
        if output_dir is not None:
            directory = Path(output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            model.save(directory / "adapted_load.pt")
            if calibration is not None:
                calibration.save(directory / "calibration.json")
            report["artifact_directory"] = str(directory.resolve())
            save_json(directory / "experiment.json", report)
        return report
    finally:
        if bundle.current_version() != source_version or feed_model.current_version() != feed_version:
            raise RuntimeError("目标工况适配意外改变传入的冻结源模型")
