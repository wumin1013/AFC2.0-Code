"""Compare speed-only vs a PRIOR idle reference in leave-one-file-out testing."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from scipy.optimize import least_squares
from spindle_power.load_prediction import (read_measurement, fit_base_power, predict_base_event,
                                     calibrate_idle_reference, assess_repeatability, CalibrationConfig)
from spindle_power.load_prediction.base_events import _filtered, _settling
from spindle_power.load_prediction.common import file_hash, save_json
from spindle_power.load_prediction.plotting import configure_chinese_plots


def compare(records, output):
    all_model = fit_base_power(records)
    folds, decay_records = [], []
    for i, record in enumerate(records):
        config = CalibrationConfig(**all_model.config)
        for item in all_model.report["event_observations"][i]:
            if item["kind"] != "startup":
                continue
            event = all_model.report["events"][i][item["index"]]
            use = (record.time_s >= item["onset_s"]) & (record.time_s <= event["end_s"])
            x = record.time_s[use] - item["onset_s"] - item["peak_time_s"]
            y = _filtered(record.values["spindle_power_w"][use], record.time_s[1] - record.time_s[0], config)
            positive = x >= 0
            x, y = x[positive][::10], y[positive][::10]
            ss, peak = item["steady_w"], item["extremum_w"]
            def curve(p):
                amplitude, fast, slow = p
                return ss + (peak - ss - amplitude) * np.exp(-x / fast) + amplitude * np.exp(-x / slow)
            fitted = least_squares(lambda p: curve(p) - y, [min(30., (peak - ss)/2), .025, 2.],
                                   bounds=([0., .001, .15], [peak - ss, .15, 20.]), loss="soft_l1", f_scale=2., max_nfev=120)
            decay_records.append(dict(file=i, speed=item["level_rpm"], parameters=fitted.x.tolist(),
                                      steady=ss, peak=peak, peak_time=item["peak_time_s"], settling=item["settling_time_s"]))

    comparison_plot = None
    for i, record in enumerate(records):
        model = fit_base_power([r for j, r in enumerate(records) if i != j])
        observations = all_model.report["event_observations"][i]  # Held-out power is scoring only.
        reference_event = next(e for e in observations if e["kind"] == "startup")
        window = all_model.report["events"][i][reference_event["index"]]
        reference = calibrate_idle_reference(record, model, start_s=window["end_s"] - 1., end_s=window["end_s"], no_load_confirmed=True)
        rows = []
        for item in observations:
            # The reference event and its brake are never counted as predictions.
            if item["level_rpm"] == reference_event["level_rpm"] or item["onset_s"] <= reference.end_s:
                continue
            origin, target = (0., item["level_rpm"]) if item["kind"] == "startup" else (item["level_rpm"], 0.)
            plain = predict_base_event(target, model, initial_speed_rpm=origin)
            adjusted = predict_base_event(target, model, initial_speed_rpm=origin, reference=reference)
            event = all_model.report["events"][i][item["index"]]
            mask = (record.time_s >= item["onset_s"]) & (record.time_s <= event["end_s"])
            t = record.time_s[mask] - item["onset_s"]
            actual = record.values["spindle_power_w"][mask]
            row = dict(kind=item["kind"], speed_rpm=item["level_rpm"], onset_s=item["onset_s"],
                       actual_steady_w=item["steady_w"], actual_settling_s=item["settling_time_s"])
            for label, prediction in (("speed_only", plain), ("reference", adjusted)):
                c = prediction["characteristics"]
                y = np.interp(t, prediction["time_s"], prediction["power_w"])
                row[label] = dict(steady_w=c["steady_power_w"], steady_error_w=c["steady_power_w"] - item["steady_w"],
                                  peak_error_w=c["extremum_power_w"] - item["extremum_w"],
                                  settling_error_s=c["settling_time_s"] - item["settling_time_s"],
                                  curve_rmse_w=float(np.sqrt(np.mean((y - actual) ** 2))))
            rows.append(row)
            if i == 0 and item["kind"] == "startup" and item["level_rpm"] == max(all_model.speeds_rpm):
                comparison_plot = t, actual, plain, adjusted
        startup = [r for r in rows if r["kind"] == "startup"]
        summary = {label: dict(steady_mae_w=float(np.mean([abs(r[label]["steady_error_w"]) for r in startup])),
                              steady_max_w=max(abs(r[label]["steady_error_w"]) for r in startup),
                              settling_max_s=max(abs(r[label]["settling_error_s"]) for r in rows))
                   for label in ("speed_only", "reference")}
        # Candidate double-exponential parameters use ONLY the other files.
        double_errors = []
        for row in startup:
            train = [d for d in decay_records if d["file"] != i and d["speed"] == row["speed_rpm"]]
            ss, pk, tp = [float(np.median([d[k] for d in train])) for k in ("steady", "peak", "peak_time")]
            amp, fast, slow = np.median([d["parameters"] for d in train], axis=0)
            t = np.arange(tp, tp + 30., .001)
            y = ss + (pk - ss - amp) * np.exp(-(t-tp)/fast) + amp * np.exp(-(t-tp)/slow)
            settled = _settling(t, y, ss, tp, config)
            double_errors.append(abs(settled - row["actual_settling_s"]) if settled is not None else None)
        summary["double_exponential_time_errors_s"] = double_errors
        folds.append(dict(held_out=record.source, reference=reference, predictions=rows, summary=summary))
        print(f"Fold {i+1}: {summary}", flush=True)
    report = dict(folds=folds, repeatability=assess_repeatability(all_model),
                  evaluation="条件验证：权重仅拟合其他文件；只用留出文件首个已完成空载平台的末1秒作参考；仅评分后续不同档位",
                  scope="非盲测替代。参考只调整平台功率，未解决严格恢复时间误差。只验证当前序列，不泛化到任意热状态。",
                  alternative="双指数恢复模型在原时间标准下未改善最差误差，未替换现有事件模型")
    save_json(output / "reference_validation.json", report)
    save_json(output / "double_exponential_fits.json", decay_records)
    all_model.save(output / "population_model.json")
    plot_comparison(comparison_plot, output)
    return report


def plot_comparison(comparison_plot, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    configure_chinese_plots()
    t, actual, plain, adjusted = comparison_plot
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    for ax in axes:
        ax.plot(t[::5], actual[::5], label="留出文件实测", lw=.8)
        for label, result in (("仅目标转速", plain), ("此前空载参考校正", adjusted)):
            ax.plot(t[::5], np.interp(t[::5], result["time_s"], result["power_w"]), label=label, lw=.9)
        ax.set(xlabel="距观测响应起点的时间（s）", ylabel="功率（W）")
        ax.grid(alpha=.2)
        ax.legend()
    axes[1].set_xlim(5., min(12., t[-1]))
    axes[1].set_ylim(260., 340.)
    fig.suptitle("10000 r/min；留出文件1；仅使用此前3000 r/min空载平台作参考")
    fig.tight_layout()
    fig.savefig(output / "reference_holdout.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("idle_directory", type=Path)
    args = parser.parse_args()
    paths = sorted(args.idle_directory.glob("*.csv"))
    if len(paths) < 3:
        raise ValueError("该验证脚本需要至少三份独立空载CSV")
    hashes = {str(p): file_hash(p) for p in paths}
    records = [read_measurement(p) for p in paths]
    output = ROOT / "data" / "paper1" / "reference_validation"
    output.mkdir(parents=True, exist_ok=True)
    compare(records, output)
    after = {str(p): file_hash(p) for p in paths}
    save_json(output / "raw_integrity.json", dict(before=hashes, after=after, unchanged=hashes == after))
    if hashes != after:
        raise RuntimeError("原始数据指纹发生变化")


if __name__ == "__main__":
    main()
