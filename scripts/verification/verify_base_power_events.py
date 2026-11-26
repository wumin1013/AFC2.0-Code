"""Reproducible three-file idle validation. Raw CSVs are only read."""
from pathlib import Path
import argparse
import csv
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from spindle_power.load_prediction import fit_base_power, read_measurement, predict_base_event, predict_base_from_measured_speed
from spindle_power.load_prediction.common import file_hash, save_json
from spindle_power.load_prediction.plotting import configure_chinese_plots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("idle_directory", type=Path)
    args = parser.parse_args()
    output = ROOT / "data" / "paper1" / "event_v2"
    paths = sorted(args.idle_directory.glob("*.csv"))
    hashes = {str(p): file_hash(p) for p in paths}
    records = [read_measurement(p) for p in paths]
    print(f"Loaded {len(records)} idle files", flush=True)
    model = fit_base_power(records)
    model.save(output / "base_power_candidate.json")
    save_json(output / "base_validation_report.json", model.report)
    queries = []
    for speed in model.speeds_rpm:
        if speed > 0:
            result = predict_base_event(speed, model)
            queries.append({**result["characteristics"], "observed_ranges": result["observed_ranges"]})
    save_json(output / "startup_characteristics.json", queries)
    with (output / "startup_characteristics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["target_speed_rpm", "peak_power_w", "peak_time_s", "steady_power_w", "settling_time_s", "peak_to_settle_s", "observed_settle_min_s", "observed_settle_max_s", "status"])
        for q in queries:
            writer.writerow([q[k] for k in ("target_speed_rpm", "peak_power_w", "peak_time_s", "steady_power_w", "settling_time_s", "peak_to_settle_s")] + q["observed_ranges"]["settling_time_s"] + [model.status])
    save_json(output / "raw_integrity.json", {"before": hashes, "after": {str(p): file_hash(p) for p in paths}})
    print(f"Status: {model.status}", flush=True)
    for f in model.report["folds"]:
        print(f"Holdout {Path(f['held_out']).name}: steady={f['steady']}; transient={f['transient']}; coverage={f['coverage_fraction']:.5f}", flush=True)
        print(f"Failed events: {[e for e in f['events'] if not e['passed']]}; excluded={f['excluded_events']}; warnings={f['warnings']}", flush=True)
    print(f"Startup characteristics: {queries}", flush=True)
    heldout = fit_base_power(records[1:]) if len(records) > 2 else None
    if heldout is not None:
        plot_holdout(records[0], model.report["event_observations"][0], heldout, output)
    if hashes != {str(p): file_hash(p) for p in paths}:
        raise RuntimeError("Raw input integrity changed")


def plot_holdout(record, items, heldout, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    configure_chinese_plots()
    if heldout is not None:
        prediction = predict_base_from_measured_speed(record.time_s, record.values["spindle_speed_rpm"], heldout)
        fig, axes = plt.subplots(3, 1, figsize=(13, 9))
        top = max(x["level_rpm"] for x in items)
        zooms = [None] + [(x["onset_s"] - .2, x["onset_s"] + x["phase_times_s"][-1] + .5)
                          for x in items if x["level_rpm"] == top]
        for ax, zoom in zip(axes, zooms):
            use = np.ones(len(record.time_s), bool) if zoom is None else (record.time_s >= zoom[0]) & (record.time_s <= zoom[1])
            stride = max(1, int(use.sum()) // 15000)
            for label, values in (("实测功率", record.values["spindle_power_w"]), ("事件模型（留出预测）", prediction["power_w"]), ("稳态曲线", prediction["steady_w"])):
                ax.plot(record.time_s[use][::stride], values[use][::stride], label=label, lw=.9)
            ax.set(xlabel="时间（s）", ylabel="功率（W）")
            ax.grid(alpha=.2)
            ax.legend()
        fig.suptitle("留出空载文件1；事件模型仅使用文件2和文件3标定")
        fig.tight_layout()
        fig.savefig(output / "base_power_holdout.png", dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    main()
