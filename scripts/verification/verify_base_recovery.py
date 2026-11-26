"""Reproduce recovery experiments without changing the active model or raw CSV."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spindle_power.load_prediction import read_measurement
from spindle_power.load_prediction.base_power import BasePowerModel
from spindle_power.load_prediction.base_events import predict_event
from spindle_power.load_prediction.base_reference import IdleReference
from spindle_power.load_prediction.base_recovery import RecoveryModel, RecoveryReference, predict_recovery_event
from spindle_power.load_prediction.recovery_validation import compare_recovery_models, MODES
from spindle_power.load_prediction.common import file_hash, save_json
from spindle_power.load_prediction.plotting import configure_chinese_plots, MODEL_LABELS


def write_summary(report, output):
    lines = ["# 恢复尾段候选验证", "", "所有候选共用原验收标准；三文件反复比较属于探索验证，不能代替新增独立确认。", ""]
    for mode, label in list(zip(MODES, ("仅目标转速", "首个已完成空载参考", "此前已完成空载事件滚动参考"))) + [("crossed", "交叉留出文件和内部转速档位")]:
        lines += [f"## {label}", "", "|方法|事件数|变速RMSE均值/W|最大峰谷误差/%|最大达峰时间误差/s|最大恢复误差/s|最大稳态偏差/W|失败事件|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        first = next(iter(report["variants"].values()))
        baseline = first["crossed_speed_baseline"] if mode == "crossed" else first["baseline_summary"][mode]
        rows = [("事件基准", baseline)] + [(name, v["crossed_speed_summary"] if mode == "crossed" else v["summary"][mode]) for name, v in report["variants"].items()]
        for name, row in rows:
            values = [f"{row[key] * scale:.3f}" if row[key] is not None else "未覆盖" for key, scale in
                      (("mean_fold_full_transient_rmse_w", 1), ("max_peak_relative_error", 100), ("max_peak_time_error_s", 1),
                       ("max_settling_time_error_s", 1), ("max_steady_error_w", 1))]
            lines.append(f"|{name}|{row['event_count']}|" + "|".join(values) + f"|{row['failed_events']}|")
        lines.append("")
    lines += ["## 采用决定", "", str(report["decision"]), ""] + ["- " + s for s in report["limits"]]
    (output / "comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_holdouts(records, report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    configure_chinese_plots()
    fig, axes = plt.subplots(len(records), 2, figsize=(12, 3 * len(records)), squeeze=False)
    for i, record in enumerate(records):
        for mode, ax in zip(("blind", "rolling_prior"), axes[i]):
            for method, color in (("phase_warp", "#bb7835"), ("exponential", "#315b9b")):
                fold = report["variants"][method]["folds"][i]
                available = [r for r in fold["scores"][mode]["events"] if r["kind"] == "startup" and "measured" in r]
                if not available:
                    ax.text(.1, .5, "没有有效预测", transform=ax.transAxes)
                    break
                row = max(available, key=lambda r: r["level_rpm"])
                model = RecoveryModel(**fold["model"])
                references = fold["references"][mode]
                ref = RecoveryReference(**max((r for r in references if "error" not in r and r["end_s"] < row["onset_s"]), key=lambda r: r["end_s"])) if references else None
                result = predict_recovery_event(row["level_rpm"], model, reference=ref, source=record.source, onset_s=row["onset_s"])
                ax.plot(result["time_s"][::5], result["power_w"][::5], color=color, lw=1., label=MODEL_LABELS[method])
            if not available:
                continue
            baseline = predict_event(row["level_rpm"], BasePowerModel(**model.baseline), reference=IdleReference(**ref.idle_reference) if ref else None)
            ax.plot(baseline["time_s"], baseline["power_w"], color="#288370", lw=1, label="事件基准（相同空载参考）")
            truth = row["measured"]
            end = row["onset_s"] + max(11., truth["settling_time_s"] + 2)
            mask = (record.time_s >= row["onset_s"]) & (record.time_s <= end)
            ax.plot(record.time_s[mask][::5] - row["onset_s"], record.values["spindle_power_w"][mask][::5], color=".5", lw=.5, alpha=.8, label="留出文件实测")
            tol = max(5., .05 * abs(truth["steady_w"]))
            ax.axhspan(truth["steady_w"] - tol, truth["steady_w"] + tol, color=".8", alpha=.2)
            ax.set(xlim=(truth["peak_time_s"], end - row["onset_s"]), ylim=(truth["steady_w"] - 25, truth["steady_w"] + 85),
                   xlabel="距观测响应起点的时间（s）", ylabel="功率（W）",
                   title=f"留出文件{i+1}，{row['level_rpm']:.0f} r/min，" + ("仅目标转速" if mode == "blind" else "此前空载滚动参考"))
            ax.grid(alpha=.2)
            if i == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "recovery_holdout.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("idle_directory", type=Path)
    args = parser.parse_args()
    paths = sorted(args.idle_directory.glob("*.csv"))
    before = {str(p): file_hash(p) for p in paths}
    records = [read_measurement(p) for p in paths]
    print(f"Loaded {len(records)} files", flush=True)
    result = compare_recovery_models(records, progress=lambda text: print(text, flush=True))
    output = ROOT / "data" / "paper1" / "recovery_validation"
    save_json(output / "comparison.json", result["report"])
    queries = {}
    for name, model in result["models"].items():
        model.save(output / f"{name}.json")
        queries[name] = [predict_recovery_event(s, model)["characteristics"] for s in model.baseline["speeds_rpm"] if s > 0]
        print(name, result["report"]["variants"][name]["summary"], flush=True)
    save_json(output / "startup_characteristics.json", queries)
    write_summary(result["report"], output)
    plot_holdouts(records, result["report"], output)
    after = {str(p): file_hash(p) for p in paths}
    save_json(output / "raw_integrity.json", dict(before=before, after=after, unchanged=before == after))
    if before != after:
        raise RuntimeError("原始CSV指纹发生变化")
    print(result["report"]["decision"], flush=True)


if __name__ == "__main__":
    main()
