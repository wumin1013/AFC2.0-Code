"""File-wise grey-box comparison with explicit blind/conditional input boundaries."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np

from .base_power import CalibrationConfig, identify_events
from .base_events import _fit, _filtered, _settling, extract_events, predict_event, predict_measured
from .base_greybox import (GreyBoxConfig, fit_greybox, initialize_greybox_state,
                          predict_greybox, predict_greybox_event)
from .base_reference import calibrate_idle_reference
from .common import checkpoint, version_of


def score_curves(record, events, observations, curves, config, *, excluded=()):
    """Common score for ALL models. curves[index]=(power_on_event_grid, final_ss).

    Held-out power is used here only for scoring, never for a model input.
    Unobserved settling and unsupported events remain failures, not omissions.
    """
    rows = []
    squared_sum, squared_n, transient_sum, transient_n = 0., 0, 0., 0
    total, valid_count = 0, 0
    for item in observations:
        i = item["index"]
        mask = (record.time_s >= item["onset_s"]) & (record.time_s <= events[i]["end_s"])
        t = record.time_s[mask] - item["onset_s"]
        raw = record.values["spindle_power_w"][mask]
        total += len(t)
        row = dict(index=i, kind=item["kind"], level_rpm=item["level_rpm"], onset_s=item["onset_s"], passed=False)
        if i not in curves:
            rows.append({**row, "reason": "模型未覆盖事件"})
            continue
        power, ss = curves[i]
        if len(power) != len(t) or not np.isfinite(power).all():
            rows.append({**row, "reason": "预测含未知初态/范围外/无效数据"})
            continue
        valid_count += len(t)
        y = _filtered(power, t[1] - t[0], config)
        peak = int(np.argmax(y) if item["kind"] == "startup" else np.argmin(y))
        raw_peak = float(raw.max() if item["kind"] == "startup" else raw.min())
        peak_error = max(abs(y[peak] - item["extremum_w"]), abs(y[peak] - raw_peak)) / max(float(np.ptp(raw)), 50.)
        settling = _settling(t, power, ss, t[peak], config)
        actual_band = _settling(t, power, item["steady_w"], t[peak], config)
        time_error = abs(t[peak] - item["peak_time_s"])
        settling_error = abs(settling - item["settling_time_s"]) if settling is not None else None
        steady_error = abs(ss - item["steady_w"])
        error = (power - raw) ** 2
        squared_sum += float(error.sum())
        squared_n += len(error)
        transient_error = error[t <= item["settling_time_s"]]
        transient_sum += float(transient_error.sum())
        transient_n += len(transient_error)
        passed = (peak_error <= config.maximum_peak_relative_error and time_error <= config.maximum_peak_time_error_s
                  and settling_error is not None and settling_error <= config.maximum_settling_time_error_s
                  and steady_error <= config.maximum_steady_error_w)
        rows.append({**row, "passed": bool(passed), "peak_relative_error": float(peak_error),
            "peak_time_error_s": float(time_error), "settling_time_error_s": settling_error,
            "steady_error_w": float(steady_error), "curve_rmse_w": float(np.sqrt(error.mean())),
            "measured": {k: item[k] for k in ("extremum_w", "peak_time_s", "settling_time_s", "steady_w")},
            "predicted": dict(extremum_w=float(y[peak]), peak_time_s=float(t[peak]), settling_time_s=settling,
                              steady_w=float(ss), settling_in_actual_band_s=actual_band)})
    return dict(events=rows, excluded_events=list(excluded), event_count=len(rows),
        coverage_fraction=valid_count / total if total else 0.,
        curve_rmse_w=float(np.sqrt(squared_sum / squared_n)) if squared_n else None,
        full_transient_rmse_w=float(np.sqrt(transient_sum / transient_n)) if transient_n else None,
        passed=bool(rows and not excluded and all(r["passed"] for r in rows) and valid_count == total))


def summarize(scores):
    rows = [e for score in scores for e in score["events"]]
    result = dict(event_count=len(rows), failed_events=sum(not e["passed"] for e in rows),
                  censored_or_invalid_events=sum(e.get("settling_time_error_s") is None for e in rows),
                  excluded_events=sum(len(s["excluded_events"]) for s in scores),
                  passed=bool(scores and all(s["passed"] for s in scores)))
    for metric in ("peak_relative_error", "peak_time_error_s", "settling_time_error_s", "steady_error_w"):
        finite = [e[metric] for e in rows if e.get(metric) is not None]
        result["max_" + metric] = max(finite) if finite else None
        result["mean_" + metric] = float(np.mean(finite)) if finite else None
    for metric in ("curve_rmse_w", "full_transient_rmse_w"):
        finite = [s[metric] for s in scores if s[metric] is not None]
        result["mean_fold_" + metric] = float(np.mean(finite)) if finite else None
    return result


def _event_curves(record, events, observations, model, *, grey=False, reference=None, cancel=None):
    curves = {}
    for item in observations:
        checkpoint(cancel)
        origin, target = (0., item["level_rpm"]) if item["kind"] == "startup" else (item["level_rpm"], 0.)
        try:
            if grey:
                prediction = predict_greybox_event(target, model, initial_speed_rpm=origin, cancel=cancel)
            else:
                prediction = predict_event(target, model, initial_speed_rpm=origin, reference=reference, cancel=cancel)
        except ValueError:
            continue
        use = (record.time_s >= item["onset_s"]) & (record.time_s <= events[item["index"]]["end_s"])
        t = record.time_s[use] - item["onset_s"]
        curves[item["index"]] = (np.interp(t, prediction["time_s"], prediction["power_w"]),
                                  prediction["characteristics"]["steady_power_w"])
    return curves


def _continuous_curves(record, events, observations, prediction, model, offset=0.):
    curves = {}
    for item in observations:
        use = (prediction["time_s"] >= item["onset_s"]) & (prediction["time_s"] <= events[item["index"]]["end_s"])
        target = item["level_rpm"] if item["kind"] == "startup" else 0.
        ss = np.interp(target, model.baseline["speeds_rpm"], model.baseline["powers_w"]) + offset * target / 1000.
        curves[item["index"]] = (prediction["power_w"][use], float(ss))
    return curves


def compare_greybox_models(records, *, config=None, fitting=None, events=None, cancel=None, progress=None):
    """Four predeclared variants; single-file diagnosis, leave-one-file-out, final fit.

    Validation is exploratory model comparison on the same three recordings;
    it is not a fresh independent confirmation after model selection.
    """
    config, fitting = config or CalibrationConfig(), fitting or GreyBoxConfig()
    config.validate()
    fitting.validate()
    if len(records) < 3 or len({r.source for r in records}) != len(records):
        raise ValueError("灰箱比较需要至少三份独立空载记录")
    checkpoint(cancel)
    events = events if events is not None else [identify_events(r, config) for r in records]
    if len(events) != len(records):
        raise ValueError("事件与文件数量不匹配")
    extracted = [extract_events(r, e, config, cancel) for r, e in zip(records, events)]
    fingerprints = [version_of([r.time_s, r.values["spindle_speed_rpm"], r.values["spindle_power_w"]]) for r in records]
    if len(set(fingerprints)) != len(records):
        raise ValueError("同一数据内容不能作为独立文件重复验证")
    folds, baseline_scores, baseline_conditional = [], [], []
    for i, record in enumerate(records):
        base = _fit([x for j, x in enumerate(extracted) if i != j], config)
        obs, skipped, _ = extracted[i]
        score = score_curves(record, events[i], obs, _event_curves(record, events[i], obs, base, cancel=cancel), config, excluded=skipped)
        baseline_scores.append(score)
        # Prefix boundary depends only on speed-defined plateaus. No future
        # held-out power is inspected to choose the calibration endpoint.
        first = next((e for e in events[i] if e["speed_rpm"] > 0), None)
        if first is None:
            raise ValueError("留出记录缺少正转速参考平台")
        later = [x for x in obs if x["onset_s"] > first["end_s"] and x["level_rpm"] != first["speed_rpm"]]
        reference_error = None
        try:
            ref = calibrate_idle_reference(record, base, start_s=first["end_s"] - 1., end_s=first["end_s"], no_load_confirmed=True, cancel=cancel)
            ref_curves = _event_curves(record, events[i], later, base, reference=ref, cancel=cancel)
        except ValueError as exc:
            ref_curves, reference_error = {}, str(exc)
        conditional = score_curves(record, events[i], later, ref_curves, config, excluded=skipped)
        baseline_conditional.append(conditional)
        folds.append(dict(held_out=record.source, baseline=score, baseline_conditional=conditional,
            reference_error=reference_error, prefix_end_s=first["end_s"], excluded_reference_level_rpm=first["speed_rpm"]))
    models, variants = {}, {}
    for order in (1, 2):
        for drive in ("acceleration", "inertial"):
            name = f"order{order}_{drive}"
            diagnoses, results = [], []
            for i, record in enumerate(records):
                checkpoint(cancel)
                if progress:
                    progress(f"{name}: 单文件诊断 {i + 1}/{len(records)}")
                single = fit_greybox([record], order=order, drive=drive, config=config, fitting=fitting, events=[events[i]], cancel=cancel)
                run = single.report["fitted_run_states"][0]
                pred = predict_greybox(record.time_s, record.values["spindle_speed_rpm"], single,
                    initial_components_w=run["initial_components_w"], offset_w_per_krpm=run["offset_w_per_krpm"], cancel=cancel)
                obs, skipped, _ = extracted[i]
                diagnostic = score_curves(record, events[i], obs,
                    _continuous_curves(record, events[i], obs, pred, single, run["offset_w_per_krpm"]), config, excluded=skipped)
                diagnoses.append(dict(source=record.source, parameters=single.payload(), fit=single.report,
                                      score=diagnostic, interpretation="同文件拟合诊断，不作为验证成绩"))
                if progress:
                    progress(f"{name}: 联合拟合其余文件，留出 {i + 1}/{len(records)}")
                train = [r for j, r in enumerate(records) if i != j]
                model = fit_greybox(train, order=order, drive=drive, config=config, fitting=fitting,
                    events=[e for j, e in enumerate(events) if i != j], cancel=cancel)
                # Explicit settled initial state is a declared blind assumption,
                # never taken from held-out target-event power.
                pred = predict_greybox(record.time_s, record.values["spindle_speed_rpm"], model,
                                      initial_components_w=np.zeros(order), cancel=cancel)
                blind = score_curves(record, events[i], obs, _continuous_curves(record, events[i], obs, pred, model), config, excluded=skipped)
                prospective = score_curves(record, events[i], obs,
                    _event_curves(record, events[i], obs, model, grey=True, cancel=cancel), config, excluded=skipped)
                later = [x for x in obs if x["onset_s"] > folds[i]["prefix_end_s"]
                         and x["level_rpm"] != folds[i]["excluded_reference_level_rpm"]]
                state_error, state = None, None
                try:
                    state = initialize_greybox_state(record, model, end_s=folds[i]["prefix_end_s"], no_load_confirmed=True, cancel=cancel)
                    use = record.time_s >= state.end_s
                    pred = predict_greybox(record.time_s[use], record.values["spindle_speed_rpm"][use], model,
                                          state=state, source=record.source, cancel=cancel)
                    curves = _continuous_curves(record, events[i], later, pred, model, state.offset_w_per_krpm)
                except ValueError as exc:
                    curves, state_error = {}, str(exc)
                conditional = score_curves(record, events[i], later, curves, config, excluded=skipped)
                results.append(dict(held_out=record.source, model_version=model.version, model=model.payload(), fit=model.report,
                    blind_measured_speed=blind, prior_idle_measured_speed=conditional, planned_speed=prospective,
                    prior_state=asdict(state) if state else None, state_error=state_error))
            if progress:
                progress(f"{name}: 汇总全部文件生成候选")
            candidate = fit_greybox(records, order=order, drive=drive, config=config, fitting=fitting, events=events, cancel=cancel)
            summary = {mode: summarize([f[mode] for f in results]) for mode in
                       ("blind_measured_speed", "prior_idle_measured_speed", "planned_speed")}
            variants[name] = dict(single_file_diagnostics=diagnoses, folds=results, summary=summary,
                                  candidate_version=candidate.version)
            candidate.report["comparison_summary"] = summary
            models[name] = candidate
    base_summary, conditional_summary = summarize(baseline_scores), summarize(baseline_conditional)
    # A conservative promotion gate, declared independently of observed results.
    eligible = []
    for name, variant in variants.items():
        candidate = models[name]
        checks = [candidate.report["convergence"], all(f["fit"]["convergence"] for f in variant["folds"])]
        for mode, baseline in (("blind_measured_speed", base_summary), ("planned_speed", base_summary),
                               ("prior_idle_measured_speed", conditional_summary)):
            score = variant["summary"][mode]
            checks.append(score["passed"])
            for key in ("max_peak_relative_error", "max_peak_time_error_s", "max_settling_time_error_s",
                        "max_steady_error_w", "mean_fold_full_transient_rmse_w"):
                checks.append(score[key] is not None and baseline[key] is not None and score[key] <= baseline[key] + 1e-9)
        if all(checks):
            eligible.append(name)
    selected = min(eligible, key=lambda k: variants[k]["summary"]["planned_speed"]["mean_fold_full_transient_rmse_w"]) if eligible else None
    report = dict(schema="afc.greybox_comparison.1", generated_utc=datetime.now(timezone.utc).isoformat(),
        sources=[r.source for r in records], data_fingerprints=fingerprints, acceptance=asdict(config), fitting=asdict(fitting),
        baseline=dict(folds=folds, summary=base_summary, conditional_summary=conditional_summary), variants=variants,
        decision=dict(replacement_recommended=selected is not None, recommended_variant=selected,
                      active_model="existing_event_model", automatic_replacement=False,
                      rule="所有模式通过原阈值且四项最大误差/完整变速RMSE不劣于相应事件基准；不通过则保留现有模型"),
        evaluation="整文件留出；条件模式仅使用首个正转速平台结束前的空载前缀，评分后续不同档位；单文件拟合仅用于诊断",
        limits=["同三份记录上的候选比较，模型选择后仍需独立新记录确认",
                "blind_measured_speed使用留出实测转速，不能当作加工前预测；planned_speed只使用训练文件转速模板",
                "初态和偏移不等同于温度辨识；不引入温度、磨损等未采集变量",
                "严格恢复时间沿用原容差及持续时间；未达稳态记为右删失/失败"])
    return dict(report=report, models=models)
