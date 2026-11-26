"""Prospective recovery comparison with file and crossed file/speed holdouts."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib

import numpy as np

from .base_power import BasePowerModel, CalibrationConfig, identify_events
from .base_events import extract_events
from .base_reference import IdleReference
from .base_recovery import (METHODS, fit_recovery, initialize_recovery_reference, predict_recovery_event)
from .greybox_validation import _event_curves, score_curves, summarize
from .common import checkpoint


MODES = ("blind", "fixed_prior", "rolling_prior")


def _curves(record, event_list, observations, model, reference=None, cancel=None):
    curves = {}
    for o in observations:
        checkpoint(cancel)
        origin, target = (0., o["level_rpm"]) if o["kind"] == "startup" else (o["level_rpm"], 0.)
        try:
            result = predict_recovery_event(target, model, initial_speed_rpm=origin, reference=reference,
                                             source=record.source, onset_s=o["onset_s"], cancel=cancel)
        except ValueError:
            continue
        use = (record.time_s >= o["onset_s"]) & (record.time_s <= event_list[o["index"]]["end_s"])
        t = record.time_s[use] - o["onset_s"]
        curves[o["index"]] = (np.interp(t, result["time_s"], result["power_w"]), result["characteristics"]["steady_power_w"])
    return curves


def _score(record, event_list, obs, curves, base, config, excluded=()):
    result = score_curves(record, event_list, obs, curves, config, excluded=excluded)
    squared, count = 0., 0
    for o in obs:
        use = (record.time_s >= o["onset_s"]) & (record.time_s <= o["onset_s"] + o["settling_time_s"])
        difference = (np.interp(record.values["spindle_speed_rpm"][use], base.speeds_rpm, base.powers_w)
                      - record.values["spindle_power_w"][use])
        squared += float(difference @ difference)
        count += len(difference)
    result["steady_only_transient_rmse_w"] = float(np.sqrt(squared / count)) if count else None
    result["passed"] = bool(result["passed"] and result["full_transient_rmse_w"] is not None
                             and result["full_transient_rmse_w"] < result["steady_only_transient_rmse_w"])
    return result


def _file_scores(record, event_list, extracted, model, config, cancel):
    obs, excluded, _ = extracted
    base = BasePowerModel(**model.baseline)
    scores, baseline, references = {}, {}, {}
    first = next((e for e in event_list if e["speed_rpm"] > 0), None)
    if first is None:
        raise ValueError("留出文件缺少首个正转速参考平台")
    later = [o for o in obs if o["onset_s"] > first["end_s"] and o["level_rpm"] != first["speed_rpm"]]
    for mode in MODES:
        chosen = obs if mode == "blind" else later
        candidate_curves, base_curves, mode_references = {}, {}, []
        cache = {}
        for o in chosen:
            checkpoint(cancel)
            reference = None
            if mode != "blind":
                end = first["end_s"]
                if mode == "rolling_prior":
                    # Update only before a new startup, using complete earlier
                    # positive plateaus. Both directions of that level share it;
                    # its own startup power cannot be used to predict braking.
                    onset = max(x["onset_s"] for x in obs if x["kind"] == "startup"
                                and x["level_rpm"] == o["level_rpm"] and x["onset_s"] <= o["onset_s"])
                    end = max(e["end_s"] for e in event_list if e["speed_rpm"] > 0 and e["end_s"] < onset)
                if end not in cache:
                    try:
                        cache[end] = initialize_recovery_reference(record, model, end_s=end, no_load_confirmed=True, cancel=cancel)
                        mode_references.append(asdict(cache[end]))
                    except ValueError as exc:
                        cache[end] = None
                        mode_references.append(dict(end_s=end, error=str(exc)))
                reference = cache[end]
                if reference is None:
                    continue  # Failure remains in score; never silently omitted.
            candidate_curves.update(_curves(record, event_list, [o], model, reference, cancel))
            idle = IdleReference(**reference.idle_reference) if reference is not None else None
            base_curves.update(_event_curves(record, event_list, [o], base, reference=idle, cancel=cancel))
        scores[mode] = _score(record, event_list, chosen, candidate_curves, base, config, excluded)
        baseline[mode] = _score(record, event_list, chosen, base_curves, base, config, excluded)
        references[mode] = mode_references
    return dict(scores=scores, baseline=baseline, references=references, prefix_end_s=first["end_s"])


def compare_recovery_models(records, *, config=None, events=None, cancel=None, progress=None):
    config = config or CalibrationConfig()
    config.validate()
    checkpoint(cancel)
    if len(records) < 3 or len({r.source for r in records}) != len(records):
        raise ValueError("恢复比较至少需要三份独立空载文件")
    fingerprints = []
    for r in records:
        digest = hashlib.sha256()
        for value in (r.time_s, r.values["spindle_speed_rpm"], r.values["spindle_power_w"]):
            digest.update(np.asarray(value, dtype="<f8").tobytes())
        fingerprints.append(digest.hexdigest())
    if len(set(fingerprints)) != len(records):
        raise ValueError("同一数据内容不能作为独立文件重复验证")
    events = events if events is not None else [identify_events(r, config) for r in records]
    if len(events) != len(records):
        raise ValueError("事件与文件数量不匹配")
    extracted = [extract_events(r, e, config, cancel) for r, e in zip(records, events)]
    levels = sorted({o["level_rpm"] for obs, _, _ in extracted for o in obs})
    models, variants = {}, {}
    for method in METHODS:
        folds, diagnoses, speed_folds, unsupported = [], [], [], []
        for i, record in enumerate(records):
            checkpoint(cancel)
            if progress:
                progress(f"{method}: 文件 {i + 1}/{len(records)}，单文件诊断与整文件留出")
            obs, excluded, _ = extracted[i]
            single = fit_recovery([record], method=method, config=config, events=[events[i]], cancel=cancel)
            single_base = BasePowerModel(**single.baseline)
            diagnoses.append(dict(source=record.source, fit=single.report,
                score=_score(record, events[i], obs, _curves(record, events[i], obs, single, cancel=cancel), single_base, config, excluded),
                interpretation="同文件拟合诊断，不作为留出成绩"))
            train = [r for j, r in enumerate(records) if i != j]
            train_events = [e for j, e in enumerate(events) if i != j]
            model = fit_recovery(train, method=method, config=config, events=train_events, cancel=cancel)
            fold = _file_scores(record, events[i], extracted[i], model, config, cancel)
            folds.append(dict(held_out=record.source, model=asdict(model), **fold))
            if progress:
                progress(f"{method}: 文件 {i + 1} × 留一转速档位（不向端点外推）")
            for level in levels:
                checkpoint(cancel)
                selected = [o for o in obs if o["level_rpm"] == level]
                candidate = fit_recovery(train, method=method, config=config, events=train_events, cancel=cancel, exclude_level=level)
                base = BasePowerModel(**candidate.baseline)
                score = _score(record, events[i], selected, _curves(record, events[i], selected, candidate, cancel=cancel), base, config)
                base_score = _score(record, events[i], selected, _event_curves(record, events[i], selected, base, cancel=cancel), base, config)
                row = dict(held_out=record.source, excluded_level_rpm=level, fit=candidate.report, score=score, baseline=base_score)
                if not all(candidate.domains[k][0] <= level <= candidate.domains[k][1] for k in candidate.domains):
                    unsupported.append({**row, "reason": "删去边界档位后超出训练域，明确拒绝外推"})
                else:
                    speed_folds.append(row)
        summary = {mode: summarize([f["scores"][mode] for f in folds]) for mode in MODES}
        baseline = {mode: summarize([f["baseline"][mode] for f in folds]) for mode in MODES}
        variants[method] = dict(folds=folds, single_file_diagnostics=diagnoses, summary=summary, baseline_summary=baseline,
            crossed_speed_folds=speed_folds, crossed_speed_summary=summarize([f["score"] for f in speed_folds]),
            crossed_speed_baseline=summarize([f["baseline"] for f in speed_folds]), unsupported_speed_folds=unsupported)
        models[method] = fit_recovery(records, method=method, config=config, events=events, cancel=cancel)
    eligible = []
    for method, variant in variants.items():
        checks = [models[method].report["convergence"], all(f["model"]["report"]["convergence"] for f in variant["folds"]),
                  all(f["fit"]["convergence"] for f in variant["crossed_speed_folds"])]
        comparisons = [(variant["summary"][m], variant["baseline_summary"][m]) for m in MODES]
        comparisons.append((variant["crossed_speed_summary"], variant["crossed_speed_baseline"]))
        for candidate, baseline in comparisons:
            checks.append(candidate["passed"])
            for metric in ("max_peak_relative_error", "max_peak_time_error_s", "max_settling_time_error_s",
                           "max_steady_error_w", "mean_fold_full_transient_rmse_w"):
                checks.append(candidate[metric] is not None and baseline[metric] is not None and candidate[metric] <= baseline[metric] + 1e-9)
        if all(checks):
            eligible.append(method)
        models[method].report["comparison_summary"] = variant["summary"]
    report = dict(schema="afc.recovery_comparison.1", generated_utc=datetime.now(timezone.utc).isoformat(),
        sources=[r.source for r in records], data_sha256=fingerprints, acceptance=asdict(config), variants=variants,
        decision=dict(replacement_recommended=bool(eligible), eligible=eligible, automatic_replacement=False,
                      active_model="existing_event_model", rule="三种输入模式及内部档位交叉留出均通过原阈值，且四项最大误差/完整变速RMSE不劣于同口径事件基准"),
        limits=["两种候选事先固定；三文件反复比较属探索验证，选择后仍需新增独立记录确认",
                "所有预测使用训练模板和目标转速；评分以实测响应起点对齐，真实控制器延迟未验证",
                "固定参考与滚动参考均只使用此前已完成空载；同档启动不用于预测该档制动；基准享有相同稳态参考",
                "不足0.3秒的慢尾段不调整恢复时间；0.5～2倍以外的参考比例拒绝，不截断掩盖异常",
                "留一档位同时留出整个测试文件；边界档位不外推，另列为不支持，不计入内部插值成绩",
                "不训练切削负载网络，不把空载校正解释为已辨识温度"])
    return dict(models=models, report=report)
