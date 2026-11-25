"""Shared Chinese labels and font setup; matplotlib stays optional until used."""

MODEL_LABELS = {
    "phase_warp": "平滑恢复时长",
    "exponential": "指数恢复尾段",
    "order1_inertial": "一阶惯性模型",
    "order2_inertial": "二阶惯性模型",
    "order1_acceleration": "一阶加速度模型",
    "order2_acceleration": "二阶加速度模型",
}


def configure_chinese_plots():
    import matplotlib
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
