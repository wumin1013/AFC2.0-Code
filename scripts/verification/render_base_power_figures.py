"""Render all saved base-power comparisons in Chinese, without refitting."""
from pathlib import Path
from dataclasses import replace
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from spindle_power.load_prediction import read_measurement, BasePowerModel, predict_base_event, predict_base_from_measured_speed
from spindle_power.load_prediction.base_reference import IdleReference
from spindle_power.load_prediction.common import file_hash, save_json
from spindle_power.load_prediction.plotting import configure_chinese_plots
from verify_base_power_events import plot_holdout
from verify_base_reference import plot_comparison
from verify_base_greybox import plot_holdouts as plot_greybox
from verify_base_recovery import plot_holdouts as plot_recovery


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def records_for_report(records, report):
    if len(records) != len(report["sources"]):
        raise ValueError("CSV数量与已保存验证报告不一致")
    # A user may supply an absolute directory although the original experiment
    # used a relative one. Rebind the stored source only after content checking.
    if "data_sha256" in report:
        fingerprints = []
        for record in records:
            digest = hashlib.sha256()
            for value in (record.time_s, record.values["spindle_speed_rpm"], record.values["spindle_power_w"]):
                digest.update(np.asarray(value, dtype="<f8").tobytes())
            fingerprints.append(digest.hexdigest())
        if fingerprints != report["data_sha256"]:
            raise ValueError("CSV内容与已保存恢复验证报告不一致，不能直接重绘")
    else:
        from spindle_power.load_prediction.common import version_of
        fingerprints = [version_of([r.time_s, r.values["spindle_speed_rpm"], r.values["spindle_power_w"]]) for r in records]
        if fingerprints != report["data_fingerprints"]:
            raise ValueError("CSV内容与已保存灰箱验证报告不一致，不能直接重绘")
    return [replace(r, source=source) for r, source in zip(records, report["sources"])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("idle_directory", type=Path)
    args = parser.parse_args()
    output = ROOT / "data" / "paper1"
    paths = sorted(args.idle_directory.glob("*.csv"))
    # All numerical results and CSVs must survive localization unchanged.
    protected = paths + list(output.glob("*.json"))
    for directory in ("event_v2", "reference_validation", "greybox_validation", "recovery_validation"):
        protected += list((output / directory).glob("*.json"))
    before = {str(p): file_hash(p) for p in protected}
    records = [read_measurement(p) for p in paths]
    report = load(output / "recovery_validation" / "comparison.json")
    # These existing fold models were fitted on files 2 and 3 only.
    fold = report["variants"]["phase_warp"]["folds"][0]
    base = BasePowerModel(**fold["model"]["baseline"])
    population = BasePowerModel.load(output / "event_v2" / "base_power_candidate.json")
    plot_holdout(records[0], population.report["event_observations"][0], base, output / "event_v2")
    reference = IdleReference(**load(output / "reference_validation" / "reference_validation.json")["folds"][0]["reference"])
    observation = max((e for e in population.report["event_observations"][0] if e["kind"] == "startup"), key=lambda e: e["level_rpm"])
    event = population.report["events"][0][observation["index"]]
    record = records[0]
    use = (record.time_s >= observation["onset_s"]) & (record.time_s <= event["end_s"])
    level = observation["level_rpm"]
    plot_comparison((record.time_s[use] - observation["onset_s"], record.values["spindle_power_w"][use],
                     predict_base_event(level, base), predict_base_event(level, base, reference=reference)), output / "reference_validation")
    greybox_report = load(output / "greybox_validation" / "comparison.json")
    plot_greybox(records_for_report(records, greybox_report), greybox_report, output / "greybox_validation")
    plot_recovery(records_for_report(records, report), report, output / "recovery_validation")
    # Retain a clearly labeled historical full-fit diagnostic for schema 1.
    legacy_path = output / "base_power_candidate.json"
    if legacy_path.exists():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        configure_chinese_plots()
        model = BasePowerModel.load(legacy_path)
        fig, axes = plt.subplots(len(records), 1, figsize=(12, 8), squeeze=False)
        for i, r in enumerate(records):
            prediction = predict_base_from_measured_speed(r.time_s, r.values["spindle_speed_rpm"], model)
            ax = axes[i, 0]
            ax.plot(r.time_s[::10], r.values["spindle_power_w"][::10], lw=.7, label="实测功率")
            ax.plot(r.time_s[::10], prediction["power_w"][::10], lw=.7, label="历史一阶候选（全文件标定）")
            ax.set(xlabel="时间（s）", ylabel="功率（W）", title=f"空载文件{i+1}")
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
        fig.suptitle("历史一阶模型拟合诊断（不作为留出验证成绩）")
        fig.tight_layout()
        fig.savefig(output / "base_power_validation.png", dpi=140)
        plt.close(fig)
    after = {str(p): file_hash(p) for p in protected}
    save_json(output / "figure_localization_integrity.json", dict(before=before, after=after, unchanged=before == after))
    if before != after:
        raise RuntimeError("绘图期间原始数据或数值结果发生变化")
    print("已生成五张中文基础功率图；原始CSV、模型及数值报告未变。")


if __name__ == "__main__":
    main()
