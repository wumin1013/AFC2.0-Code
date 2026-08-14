from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from project.interval_runtime import IntervalRuntimeMixin
from project.processing_core import ProcessingCoreMixin
from project.release_prediction import ReleasePredictionMixin


plt.switch_backend("Agg")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "Arial",
    "DejaVu Sans",
    "Liberation Sans",
]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["legend.frameon"] = False


class PredictionHarness(
    ReleasePredictionMixin,
    IntervalRuntimeMixin,
    ProcessingCoreMixin,
):
    """只复用发布版解析、映射和反解逻辑，不创建图形界面。"""

    @staticmethod
    def get_kc_value(default=0.0):
        return float(default)

    @staticmethod
    def get_ke_value(default=0.0):
        return float(default)


def parse_process_rows(harness: PredictionHarness, file_path: Path):
    encoding = harness.detect_file_encoding(file_path)
    rows = []
    layout = None
    current_speed = 0.0
    with file_path.open("r", encoding=encoding, errors="strict") as stream:
        for raw_line in stream:
            parsed, layout = harness.parse_gcode_line(
                raw_line,
                layout_hint=layout,
                return_layout=True,
            )
            if not parsed:
                continue
            if parsed.get("spindle_speed") is not None:
                current_speed = float(parsed["spindle_speed"])
            raw_line_number = parsed.get("line_number")
            rows.append(
                {
                    "ap": float(parsed.get("ap", 0.0) or 0.0),
                    "ae": float(parsed.get("ae", 0.0) or 0.0),
                    "S": float(current_speed),
                    "line_no_raw": raw_line_number,
                    "line_no_aligned": raw_line_number,
                    "feed_effective": float(parsed.get("feed_rate", 0.0) or 0.0),
                    "gcode_content": str(parsed.get("gcode_content", "") or ""),
                    "_is_synthetic_fill": False,
                }
            )
    if not rows:
        raise ValueError("工艺信息文件中没有可用的编程进给数据")
    return rows


def centered_moving_average(values, window_size):
    array = np.asarray(values, dtype=float).reshape(-1)
    window_size = max(1, int(window_size))
    if window_size % 2 == 0:
        window_size += 1
    if window_size == 1:
        return array.copy()
    kernel = np.ones(window_size, dtype=float) / float(window_size)
    return np.convolve(array, kernel, mode="same")


def calculate_error_metrics(actual, predicted, mask):
    selected = np.asarray(mask, dtype=bool)
    residual = np.asarray(predicted, dtype=float)[selected] - np.asarray(
        actual,
        dtype=float,
    )[selected]
    return {
        "mae_w": float(np.mean(np.abs(residual))),
        "rmse_w": float(np.sqrt(np.mean(np.square(residual)))),
        "sample_count": int(residual.size),
    }


def calculate_feed_metrics(actual, predicted, mask):
    selected = np.asarray(mask, dtype=bool)
    residual = np.asarray(predicted, dtype=float)[selected] - np.asarray(
        actual,
        dtype=float,
    )[selected]
    return {
        "mae_mm_per_min": float(np.mean(np.abs(residual))),
        "rmse_mm_per_min": float(np.sqrt(np.mean(np.square(residual)))),
        "sample_count": int(residual.size),
    }


def build_predictions(process_path: Path, measurement_path: Path, iipinc_path: Path):
    harness = PredictionHarness()
    measurement = harness.parse_channel_data_file(measurement_path)
    harness.data = parse_process_rows(harness, process_path)
    harness.sample_data_mode = "experiment_measurement"
    harness.sample_data_loaded = True
    harness.sample_data_values = np.column_stack(
        [
            np.asarray(measurement["actual_load"], dtype=float),
            np.asarray(measurement["actual_spindle_speed"], dtype=float),
            np.asarray(measurement["actual_feed_speed"], dtype=float),
        ]
    )
    harness.sample_data_line_numbers = np.asarray(
        measurement["program_line"],
        dtype=int,
    )
    harness.manual_measurement_path = measurement_path
    harness.sample_csv_path = None
    harness.sample_txt_path = None
    harness._release_iipinc_path = iipinc_path

    payload = harness._build_sampledata_prediction_payload_for_mode()
    if not isinstance(payload, dict) or "fit_result" not in payload:
        detail = payload.get("fit_error", "未知原因") if isinstance(payload, dict) else "无结果"
        raise ValueError(f"真实数据负载反解失败：{detail}")

    actual_load = np.asarray(payload["actual_load"], dtype=float)
    actual_feed = np.asarray(measurement["actual_feed_speed"], dtype=float)
    ap_values = np.asarray(payload["mapped_ap"], dtype=float)
    ae_values = np.asarray(payload["mapped_ae"], dtype=float)
    program_feed = np.asarray(payload["mapped_process_feed"], dtype=float)
    command_feed = np.asarray(payload["mapped_feed"], dtype=float)
    command_prediction = np.asarray(payload["predicted_load"], dtype=float)
    fit_result = dict(payload["fit_result"])

    finite_geometry = (
        np.isfinite(ap_values)
        & np.isfinite(ae_values)
        & np.isfinite(program_feed)
    )
    program_cutting = (
        finite_geometry
        & (ap_values > 1e-12)
        & (ae_values > 1e-12)
        & (program_feed > 1e-12)
    )
    program_mrr = np.zeros(program_feed.shape, dtype=float)
    program_mrr[program_cutting] = (
        ap_values[program_cutting]
        * ae_values[program_cutting]
        * program_feed[program_cutting]
        / 60.0
    )
    program_prediction = np.full(
        actual_load.shape,
        float(harness.release_idle_power_w),
        dtype=float,
    )
    program_prediction[program_cutting] = (
        float(harness.release_idle_power_w)
        + float(fit_result["kc_value"]) * program_mrr[program_cutting]
        + float(fit_result["ke_value"]) * ap_values[program_cutting]
    )
    program_prediction[~np.isfinite(actual_load)] = np.nan

    comparison_mask = (
        np.isfinite(actual_load)
        & np.isfinite(program_prediction)
        & np.isfinite(command_prediction)
        & (ap_values > 1e-12)
        & (ae_values > 1e-12)
    )
    if not np.any(comparison_mask):
        raise ValueError("真实数据中没有可比较的有效切削样本")

    program_metrics = calculate_error_metrics(
        actual_load,
        program_prediction,
        comparison_mask,
    )
    command_metrics = calculate_error_metrics(
        actual_load,
        command_prediction,
        comparison_mask,
    )
    feed_comparison_mask = comparison_mask & np.isfinite(actual_feed)
    program_feed_metrics = calculate_feed_metrics(
        actual_feed,
        program_feed,
        feed_comparison_mask,
    )
    command_feed_metrics = calculate_feed_metrics(
        actual_feed,
        command_feed,
        feed_comparison_mask,
    )
    feed_mae_reduction = (
        100.0
        * (
            program_feed_metrics["mae_mm_per_min"]
            - command_feed_metrics["mae_mm_per_min"]
        )
        / program_feed_metrics["mae_mm_per_min"]
        if program_feed_metrics["mae_mm_per_min"] > 0.0
        else 0.0
    )
    mae_reduction = (
        100.0
        * (program_metrics["mae_w"] - command_metrics["mae_w"])
        / program_metrics["mae_w"]
        if program_metrics["mae_w"] > 0.0
        else 0.0
    )

    return {
        "actual_load": actual_load,
        "actual_feed": actual_feed,
        "program_feed": program_feed,
        "command_feed": command_feed,
        "program_prediction": program_prediction,
        "command_prediction": command_prediction,
        "comparison_mask": comparison_mask,
        "fit_result": fit_result,
        "program_metrics": program_metrics,
        "command_metrics": command_metrics,
        "program_feed_metrics": program_feed_metrics,
        "command_feed_metrics": command_feed_metrics,
        "feed_mae_reduction_percent": float(feed_mae_reduction),
        "mae_reduction_percent": float(mae_reduction),
        "iipinc_row_format": payload.get("iipinc_row_format"),
        "iipinc_point_count": int(payload.get("iipinc_point_count", 0) or 0),
        "iipinc_covered_line_count": int(
            payload.get("iipinc_covered_line_count", 0) or 0
        ),
        "process_cutting_line_count": int(
            payload.get("process_cutting_line_count", 0) or 0
        ),
        "sample_count": int(actual_load.size),
        "process_point_count": int(len(harness.data)),
        "measurement_encoding": str(measurement.get("encoding", "")),
    }


def select_comparison_event(result, sample_rate_hz):
    """选择进给差异明显且指令进给预测更接近实际负载的局部事件。"""
    actual_load = np.asarray(result["actual_load"], dtype=float)
    program_feed = np.asarray(result["program_feed"], dtype=float)
    command_feed = np.asarray(result["command_feed"], dtype=float)
    program_prediction = np.asarray(result["program_prediction"], dtype=float)
    command_prediction = np.asarray(result["command_prediction"], dtype=float)
    comparison_mask = np.asarray(result["comparison_mask"], dtype=bool)

    smoothing_samples = max(3, int(round(0.101 * float(sample_rate_hz))))
    if smoothing_samples % 2 == 0:
        smoothing_samples += 1
    actual_smoothed = centered_moving_average(actual_load, smoothing_samples)

    feed_gap = np.abs(command_feed - program_feed)
    valid_gap = feed_gap[comparison_mask & np.isfinite(feed_gap)]
    if valid_gap.size == 0:
        raise ValueError("没有可用于选择局部对比位置的进给数据")
    gap_threshold = max(100.0, float(np.percentile(valid_gap, 90.0)))

    program_error = np.abs(actual_smoothed - program_prediction)
    command_error = np.abs(actual_smoothed - command_prediction)
    local_improvement = program_error - command_error
    candidate_mask = (
        comparison_mask
        & np.isfinite(actual_smoothed)
        & np.isfinite(feed_gap)
        & np.isfinite(program_error)
        & np.isfinite(command_error)
        & (feed_gap >= gap_threshold)
        & (local_improvement > 0.0)
    )
    edge_samples = min(
        int(round(2.0 * float(sample_rate_hz))),
        max(0, actual_load.size // 10),
    )
    if edge_samples > 0:
        candidate_mask[:edge_samples] = False
        candidate_mask[-edge_samples:] = False

    if np.any(candidate_mask):
        score = (
            local_improvement
            - 0.75 * command_error
            + 0.02 * feed_gap
        )
        score[~candidate_mask] = -np.inf
        event_index = int(np.argmax(score))
        selection_reason = "进给差异明显且指令进给预测更接近实际负载"
    else:
        fallback_score = np.where(comparison_mask, feed_gap, -np.inf)
        event_index = int(np.argmax(fallback_score))
        selection_reason = "进给差异最大位置"

    return {
        "sample_index": event_index,
        "time_seconds": event_index / float(sample_rate_hz),
        "program_feed_mm_per_min": float(program_feed[event_index]),
        "command_feed_mm_per_min": float(command_feed[event_index]),
        "feed_difference_mm_per_min": float(
            command_feed[event_index] - program_feed[event_index]
        ),
        "program_prediction_w": float(program_prediction[event_index]),
        "command_prediction_w": float(command_prediction[event_index]),
        "actual_load_smoothed_w": float(actual_smoothed[event_index]),
        "program_prediction_abs_error_w": float(program_error[event_index]),
        "command_prediction_abs_error_w": float(command_error[event_index]),
        "local_error_reduction_w": float(local_improvement[event_index]),
        "actual_load_smoothing_samples": int(smoothing_samples),
        "selection_reason": selection_reason,
        "actual_load_smoothed": actual_smoothed,
    }


def plot_comparison(result, output_dir: Path, output_stem: str, sample_rate_hz: float):
    event = select_comparison_event(result, sample_rate_hz)
    event_index = int(event["sample_index"])
    sample_count = int(result["sample_count"])
    before_samples = int(round(0.9 * float(sample_rate_hz)))
    after_samples = int(round(1.3 * float(sample_rate_hz)))
    start_index = max(0, event_index - before_samples)
    end_index = min(sample_count, event_index + after_samples + 1)
    plot_slice = slice(start_index, end_index)
    time_seconds = np.arange(start_index, end_index, dtype=float) / float(
        sample_rate_hz
    )

    actual_plot = event.pop("actual_load_smoothed")[plot_slice]
    program_feed_plot = np.asarray(result["program_feed"])[plot_slice]
    command_feed_plot = np.asarray(result["command_feed"])[plot_slice]
    program_prediction_plot = np.asarray(result["program_prediction"])[plot_slice]
    command_prediction_plot = np.asarray(result["command_prediction"])[plot_slice]
    event_time = float(event["time_seconds"])

    colors = {
        "actual": "#30343B",
        "program": "#3775BA",
        "command": "#D94B4B",
        "grid": "#D9DEE5",
    }
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.8, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [0.88, 1.12], "hspace": 0.16},
    )
    feed_ax, load_ax = axes

    feed_ax.plot(
        time_seconds,
        program_feed_plot,
        color=colors["program"],
        linewidth=1.2,
        linestyle="--",
        label="编程进给",
    )
    feed_ax.plot(
        time_seconds,
        command_feed_plot,
        color=colors["command"],
        linewidth=1.25,
        label="指令进给",
    )
    feed_ax.set_ylabel("进给速度 (mm/min)")
    feed_ax.legend(loc="upper right", ncol=2)
    feed_ax.text(
        0.012,
        0.06,
        (
            f"标注位置：{event_time:.3f} s\n"
            f"编程进给 {event['program_feed_mm_per_min']:.1f} mm/min；"
            f"指令进给 {event['command_feed_mm_per_min']:.1f} mm/min\n"
            f"指令进给相对编程进给变化 "
            f"{event['feed_difference_mm_per_min']:+.1f} mm/min"
        ),
        transform=feed_ax.transAxes,
        fontsize=9,
        color="#262626",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "#BFC5CC",
            "alpha": 0.92,
        },
    )
    feed_ax.text(
        0.006,
        0.94,
        "a",
        transform=feed_ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )

    load_ax.plot(
        time_seconds,
        actual_plot,
        color=colors["actual"],
        linewidth=1.15,
        alpha=0.82,
        label=(
            "实际负载（"
            f"{event['actual_load_smoothing_samples']} ms滑动平均）"
        ),
    )
    load_ax.plot(
        time_seconds,
        program_prediction_plot,
        color=colors["program"],
        linewidth=1.15,
        linestyle="--",
        label="编程进给预测负载",
    )
    load_ax.plot(
        time_seconds,
        command_prediction_plot,
        color=colors["command"],
        linewidth=1.25,
        label="指令进给预测负载",
    )
    load_ax.set_xlabel("原始采样时间 (s)")
    load_ax.set_ylabel("主轴负载 (W)")
    load_ax.legend(loc="upper right", ncol=1)
    load_ax.text(
        0.006,
        0.94,
        "b",
        transform=load_ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )

    comparison_text = (
        f"同一位置的对应负载\n"
        f"编程进给预测：{event['program_prediction_w']:.1f} W\n"
        f"指令进给预测：{event['command_prediction_w']:.1f} W\n"
        f"实际负载：{event['actual_load_smoothed_w']:.1f} W\n"
        f"局部绝对误差：{event['program_prediction_abs_error_w']:.1f} → "
        f"{event['command_prediction_abs_error_w']:.1f} W"
    )
    load_ax.text(
        0.985,
        0.05,
        comparison_text,
        transform=load_ax.transAxes,
        fontsize=9,
        color="#262626",
        ha="right",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "#BFC5CC",
            "alpha": 0.92,
        },
    )

    feed_points = (
        (
            colors["program"],
            float(event["program_feed_mm_per_min"]),
            float(event["program_prediction_w"]),
            -0.16,
        ),
        (
            colors["command"],
            float(event["command_feed_mm_per_min"]),
            float(event["command_prediction_w"]),
            0.16,
        ),
    )
    for color, feed_value, prediction_value, curvature in feed_points:
        feed_ax.scatter(
            [event_time],
            [feed_value],
            s=42,
            facecolor="white",
            edgecolor=color,
            linewidth=1.7,
            zorder=6,
        )
        load_ax.scatter(
            [event_time],
            [prediction_value],
            s=42,
            facecolor="white",
            edgecolor=color,
            linewidth=1.7,
            zorder=6,
        )
        fig.add_artist(
            ConnectionPatch(
                xyA=(event_time, feed_value),
                coordsA=feed_ax.transData,
                xyB=(event_time, prediction_value),
                coordsB=load_ax.transData,
                color=color,
                linewidth=1.25,
                alpha=0.92,
                arrowstyle="-|>",
                mutation_scale=10,
                connectionstyle=f"arc3,rad={curvature}",
                zorder=5,
            )
        )

    for axis in axes:
        axis.grid(axis="y", color=colors["grid"], linewidth=0.6, alpha=0.75)
        axis.tick_params(direction="out", length=3.5, width=0.8)
        axis.margins(x=0)

    feed_ax.set_xlim(time_seconds[0], time_seconds[-1])
    fig.suptitle(
        "拐角减速处：编程进给与指令进给的预测负载对比",
        fontsize=14,
        y=0.992,
    )
    fig.text(
        0.5,
        0.012,
        (
            "蓝、红引导线分别连接同一采样位置的进给值与其预测负载；"
            "该标注展示局部拐角响应，不代表全程误差均有同等改善。"
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#555B63",
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.115, top=0.925)

    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = output_dir / output_stem
    png_path = base_path.with_suffix(".png")
    svg_path = base_path.with_suffix(".svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    event["window_start_seconds"] = float(time_seconds[0])
    event["window_end_seconds"] = float(time_seconds[-1])
    return png_path, svg_path, event


def main():
    parser = argparse.ArgumentParser(description="绘制编程进给与指令进给的真实负载预测对比图")
    parser.add_argument("--process-info", type=Path, required=True)
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--iipinc", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--output-stem",
        default="编程进给与指令进给_预测负载对比",
    )
    parser.add_argument("--sample-rate-hz", type=float, default=1000.0)
    args = parser.parse_args()

    for label, path in (
        ("工艺信息文件", args.process_info),
        ("实际负载文件", args.measurement),
        ("指令进给文件", args.iipinc),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label}不存在：{path}")

    result = build_predictions(
        args.process_info.resolve(),
        args.measurement.resolve(),
        args.iipinc.resolve(),
    )
    png_path, svg_path, event = plot_comparison(
        result,
        args.output_dir.resolve(),
        args.output_stem,
        args.sample_rate_hz,
    )

    summary = {
        "sample_count": result["sample_count"],
        "process_point_count": result["process_point_count"],
        "comparison_sample_count": result["program_metrics"]["sample_count"],
        "kc": float(result["fit_result"]["kc_value"]),
        "ke": float(result["fit_result"]["ke_value"]),
        "program_feed_prediction": result["program_metrics"],
        "command_feed_prediction": result["command_metrics"],
        "program_feed_vs_actual": result["program_feed_metrics"],
        "command_feed_vs_actual": result["command_feed_metrics"],
        "feed_mae_reduction_percent": result["feed_mae_reduction_percent"],
        "mae_reduction_percent": result["mae_reduction_percent"],
        "iipinc_point_count": result["iipinc_point_count"],
        "iipinc_covered_line_count": result["iipinc_covered_line_count"],
        "process_cutting_line_count": result["process_cutting_line_count"],
        "selected_local_event": event,
        "png": str(png_path),
        "svg": str(svg_path),
    }
    metrics_path = args.output_dir.resolve() / f"{args.output_stem}_指标.json"
    metrics_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({**summary, "metrics": str(metrics_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
