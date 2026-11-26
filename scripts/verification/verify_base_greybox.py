"""Reproduce non-neural grey-box comparison; raw input files remain read-only."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spindle_power.load_prediction import read_measurement
from spindle_power.load_prediction.base_greybox import GreyBoxModel, predict_greybox_event, predict_greybox
from spindle_power.load_prediction.base_power import BasePowerModel, CalibrationConfig
from spindle_power.load_prediction.base_reference import calibrate_idle_reference
from spindle_power.load_prediction.base_events import predict_event
from spindle_power.load_prediction.greybox_validation import compare_greybox_models
from spindle_power.load_prediction.common import file_hash, save_json
from spindle_power.load_prediction.plotting import configure_chinese_plots, MODEL_LABELS


def write_summary(report, output):
    lines = ["# 灰箱对比结果", "", "各方法共用原验收阈值；误差为三份文件留出结果。模型选择后仍需新增独立记录确认。", ""]
    for mode, label in (("blind_measured_speed", "实测转速输入（54事件）"),
                        ("prior_idle_measured_speed", "此前空载初始化＋实测转速（48事件）"),
                        ("planned_speed", "仅目标转速，训练文件转速模板（54事件）")):
        lines += [f"## {label}", "", "|方法|完整变速RMSE均值/W|最大峰谷误差/%|最大达峰谷时间误差/s|最大恢复时间误差/s|最大稳态偏差/W|失败事件|",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        baseline = report["baseline"]["conditional_summary" if mode == "prior_idle_measured_speed" else "summary"]
        entries = [("事件基准", baseline)] + [(k, v["summary"][mode]) for k, v in report["variants"].items()]
        for name, s in entries:
            def fmt(key, scale=1):
                return f"{s[key] * scale:.3f}" if s[key] is not None else "未观测/无效"
            lines.append(f"|{name}|{fmt('mean_fold_full_transient_rmse_w')}|{fmt('max_peak_relative_error', 100)}|"
                         f"{fmt('max_peak_time_error_s')}|{fmt('max_settling_time_error_s')}|{fmt('max_steady_error_w')}|{s['failed_events']}|")
        lines.append("")
    lines += ["## 结论", "", str(report["decision"]), "", "完整报告：comparison.json；参数候选：order*.json。", ""]
    (output / "comparison_summary.md").write_text("\n".join(lines), encoding="utf-8")


def plot_holdouts(records, report, output):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    configure_chinese_plots()
    from spindle_power.load_prediction.base_greybox import GreyBoxState
    fig, axes = plt.subplots(len(records), 2, figsize=(12, 3.1 * len(records)), squeeze=False)
    for i, record in enumerate(records):
        fold = report["variants"]["order2_inertial"]["folds"][i]
        model = GreyBoxModel(**fold["model"], version=fold["model_version"])
        state = GreyBoxState(**fold["prior_state"])
        use = record.time_s >= state.end_s
        predictions = {}
        for name in ("order1_inertial", "order2_inertial"):
            f = report["variants"][name]["folds"][i]
            m = GreyBoxModel(**f["model"], version=f["model_version"])
            st = GreyBoxState(**f["prior_state"])
            predictions[name] = predict_greybox(record.time_s[use], record.values["spindle_speed_rpm"][use], m, state=st, source=record.source)
        item = max((e for e in fold["prior_idle_measured_speed"]["events"] if e["kind"] == "startup"), key=lambda e: e["level_rpm"])
        onset, truth = item["onset_s"], item["measured"]
        base = BasePowerModel(**model.baseline)
        reference = calibrate_idle_reference(record, base, start_s=state.end_s - 1., end_s=state.end_s, no_load_confirmed=True)
        event = predict_event(item["level_rpm"], base, reference=reference)
        end = onset + max(truth["settling_time_s"] + 2., 10.)
        for j, ax in enumerate(axes[i]):
            mask = (record.time_s >= onset) & (record.time_s <= end)
            ax.plot(record.time_s[mask][::5] - onset, record.values["spindle_power_w"][mask][::5], color=".45", lw=.6, label="留出文件实测")
            ax.plot(event["time_s"], event["power_w"], color="#26836b", lw=1.1, label="事件模型＋此前空载参考")
            for name, color in (("order1_inertial", "#c78a32"), ("order2_inertial", "#305b9b")):
                pred = predictions[name]
                mask = (pred["time_s"] >= onset) & (pred["time_s"] <= end)
                ax.plot(pred["time_s"][mask][::5] - onset, pred["power_w"][mask][::5], color=color, lw=1., label=MODEL_LABELS[name])
            ax.set(xlabel="距观测响应起点的时间（s）", ylabel="功率（W）", title=f"留出文件{i+1}：{item['level_rpm']:.0f} r/min")
            ax.set_xlim(0, end - onset)
            if j:
                tol = max(5., .05 * abs(truth["steady_w"]))
                ax.axhspan(truth["steady_w"] - tol, truth["steady_w"] + tol, color=".8", alpha=.25)
                ax.set_xlim(truth["peak_time_s"], end - onset)
                ax.set_ylim(truth["steady_w"] - 30, truth["steady_w"] + 85)
            ax.grid(alpha=.2)
            if i == 0:
                ax.legend(fontsize=7)
    fig.suptitle("条件留出验证：仅用此前空载前缀初始化；灰箱曲线使用实测转速", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(output / "greybox_holdout.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("idle_directory", type=Path)
    args = parser.parse_args()
    paths = sorted(args.idle_directory.glob("*.csv"))
    hashes = {str(p): file_hash(p) for p in paths}
    records = [read_measurement(p) for p in paths]
    print(f"Loaded {len(records)} files", flush=True)
    result = compare_greybox_models(records, progress=lambda message: print(message, flush=True))
    output = ROOT / "data" / "paper1" / "greybox_validation"
    save_json(output / "comparison.json", result["report"])
    queries = {}
    for name, model in result["models"].items():
        model.save(output / f"{name}.json")
        queries[name] = [predict_greybox_event(s, model)["characteristics"] for s in model.baseline["speeds_rpm"] if s > 0]
        print(f"{name}: {result['report']['variants'][name]['summary']}", flush=True)
    save_json(output / "startup_characteristics.json", queries)
    write_summary(result["report"], output)
    plot_holdouts(records, result["report"], output)
    after = {str(p): file_hash(p) for p in paths}
    save_json(output / "raw_integrity.json", dict(before=hashes, after=after, unchanged=hashes == after))
    print(result["report"]["decision"], flush=True)
    if hashes != after:
        raise RuntimeError("原始数据指纹发生变化")


if __name__ == "__main__":
    main()
