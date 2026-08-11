from __future__ import annotations

import collections
import gc
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Union

os.environ.setdefault("MPLBACKEND", "TkAgg")
gc.set_threshold(10000, 10, 10)

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D as _Axes3D  # noqa: F401  # 注册 3D 投影
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoLocator, MaxNLocator

pd.options.mode.chained_assignment = None
pd.options.display.float_format = "{:.6f}".format
plt.ioff()
matplotlib.rcParams["path.simplify"] = True
matplotlib.rcParams["path.simplify_threshold"] = 1.0
matplotlib.rcParams["agg.path.chunksize"] = 10000
matplotlib.rcParams["figure.max_open_warning"] = 0
matplotlib.rcParams["savefig.dpi"] = 100
matplotlib.rcParams["figure.autolayout"] = False

_sklearn_linear = None


def _get_pandas():
    return pd


def _get_sklearn():
    global _sklearn_linear
    if _sklearn_linear is None:
        from sklearn.linear_model import HuberRegressor, LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split

        _sklearn_linear = SimpleNamespace(
            LinearRegression=LinearRegression,
            HuberRegressor=HuberRegressor,
            train_test_split=train_test_split,
            mean_squared_error=mean_squared_error,
            r2_score=r2_score,
        )
    return _sklearn_linear


UI_FONT_NORMAL = ("Microsoft YaHei UI", 11)
UI_FONT_BOLD = ("Microsoft YaHei UI", 11, "bold")
UI_FONT_LARGE = ("Microsoft YaHei UI", 13, "bold")
UI_FONT_SMALL = ("Microsoft YaHei UI", 9)
UI_FONT_TITLE = ("Microsoft YaHei UI", 14, "bold")
UI_BTN_WIDTH = 16
UI_BTN_PADX = 8
UI_BTN_PADY = 5

UI_COLOR_PRIMARY = "#1E90FF"
UI_COLOR_PRIMARY_DARK = "#0066CC"
UI_COLOR_PRIMARY_LIGHT = "#87CEEB"
UI_COLOR_SUCCESS = "#00CED1"
UI_COLOR_WARNING = "#FF8C00"
UI_COLOR_DANGER = "#DC3545"
UI_COLOR_BG_DARK = "#1B4F72"
UI_COLOR_BG_LIGHT = "#FFFFFF"
UI_COLOR_BG_PANEL = "#F0F8FF"
UI_COLOR_BORDER = "#B0C4DE"
UI_COLOR_TEXT = "#1C2833"
UI_COLOR_TEXT_MUTED = "#5D6D7E"
UI_COLOR_HEADER_BG = "#E8F4FD"

STYLE_MEASURED = {"color": "#00B4D8", "linestyle": "-", "linewidth": 1.5, "label": "实测"}
STYLE_PREDICTED = {"color": "#E63946", "linestyle": "--", "linewidth": 1.7, "label": "预测"}

PLOT_FONT_BASE = 12
PLOT_TEXT_COLOR = "#1C2833"
PLOT_GRID_COLOR = "#D6EAF8"
PLOT_SPINE_COLOR = "#85C1E9"
PLOT_FIG_BG = "#FFFFFF"
PLOT_AX_BG = "#FAFEFF"
SMIF_FIG_BG = "#FFFFFF"
SMIF_AX_BG = "#FFFFFF"
SMIF_PANE_BG = (1.0, 1.0, 1.0, 0.98)
SMIF_PANE_EDGE = (0.72, 0.76, 0.82, 0.55)
SMIF_TEXT_COLOR = "#1C2833"
SMIF_TEXT_MUTED = "#5D6D7E"
SMIF_IDLE_COLOR = "#D8E2EA"
SMIF_NONSTEADY_COLOR = "#9FD3FF"
SMIF_ANNOTATION_COLOR = "#FFD166"
SMIF_PANEL_BG = "#FFFFFF"
SMIF_PANEL_EDGE = "#C7D0D9"
SMIF_GRID_COLOR = (0.45, 0.50, 0.56, 0.12)
SMIF_METRIC_IDLE_FILL = "#8CA3B3"
SMIF_METRIC_STEADY_FILL = "#37C3FF"
SMIF_METRIC_NONSTEADY_FILL = "#6EC5FF"
SMIF_BOX_MIN_RATIO = 0.30
SMIF_BOX_ZOOM = 1.15
SMIF_MAIN_ELEV = 24.0
SMIF_MAIN_AZIM = -58.0
SMIF_AXES_RECT = (0.03, 0.05, 0.84, 0.88)
SMIF_COLORBAR_RECT = (0.90, 0.14, 0.018, 0.68)

INTERVAL_COLORS = ["#3498DB", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]
TOOL_BG_COLORS = ["#AED6F1", "#F5B7B1", "#ABEBC6", "#FAD7A0", "#D7BDE2", "#A3E4D7"]

IS_FROZEN = bool(getattr(sys, "frozen", False))

if IS_FROZEN:
    app_dir = os.path.dirname(os.path.abspath(sys.executable))
    base_dir = getattr(sys, "_MEIPASS", app_dir)
    os.chdir(app_dir)
else:
    project_dir = Path(__file__).resolve().parent
    package_root = project_dir.parent
    app_dir = str(package_root.parent)
    base_dir = app_dir

PROJECT_ROOT = Path(app_dir).resolve()
APP_DIR = PROJECT_ROOT
RESOURCE_ROOT = Path(base_dir).resolve()
CONFIG_DIR = PROJECT_ROOT / "config"
SAMPLE_DATA_DIR = PROJECT_ROOT / "data" / "sample"
RUNTIME_DATA_DIR = PROJECT_ROOT / "data" / "runtime"
PROFILE_DIR = PROJECT_ROOT / "profiles"
PROFILE_CACHE_DIR = PROFILE_DIR / "cache"
# 冻结发布版的结果直接写到 EXE 同目录；源码研究版继续集中写入项目 output/。
OUTPUT_DIR = PROJECT_ROOT if IS_FROZEN else PROJECT_ROOT / "output"

# 仅使用目标 Windows 已安装字体的回退链。发布包不携带来源不明的字体文件。
simhei_path = ""
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "SimSun",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


def _suppress_messageboxes():
    def _noop(*args, **kwargs):
        return None

    def _always_true(*args, **kwargs):
        return True

    messagebox.showinfo = _noop
    messagebox.showwarning = _noop
    messagebox.showerror = _noop
    messagebox.askyesno = _always_true


if os.environ.get("SUPPRESS_MESSAGEBOXES") == "1":
    _suppress_messageboxes()


def center_dialog_on_parent(dialog, parent):
    """将弹窗居中显示在父窗口上"""
    dialog.update_idletasks()
    dialog_width = dialog.winfo_width()
    dialog_height = dialog.winfo_height()
    parent_x = parent.winfo_rootx()
    parent_y = parent.winfo_rooty()
    parent_width = parent.winfo_width()
    parent_height = parent.winfo_height()
    x = parent_x + (parent_width - dialog_width) // 2
    y = parent_y + (parent_height - dialog_height) // 2
    screen_width = dialog.winfo_screenwidth()
    screen_height = dialog.winfo_screenheight()
    x = max(0, min(x, screen_width - dialog_width))
    y = max(0, min(y, screen_height - dialog_height))
    dialog.geometry(f"+{x}+{y}")


@dataclass
class PITEntry:
    zone_id: str
    start_idx: int
    end_idx: int
    start_line: int
    end_line: int
    start_s: float
    end_s: float
    a_p: float
    a_e: float
    F_plan: float
    p_idle: float
    p_pred: float
    K_c_hat: float
    K_c_UCB: float
    sigma_Kc: float
    sample_count: int
    start_label: str = ""
    end_label: str = ""
    process_start_label: Optional[str] = None
    process_end_label: Optional[str] = None
    sample_start_line: Optional[int] = None
    sample_end_line: Optional[int] = None
    display_start_x: float = math.nan
    display_end_x: float = math.nan
    display_start_t: float = math.nan
    display_end_t: float = math.nan
    valid_kc_count: int = 0
    gated_out_count: int = 0
    p_meas: float = 0.0
    actual_load_std: float = 0.0
    actual_load_diff_std: float = 0.0
    sigma_idle: float = 0.0
    delta_mrr: float = 0.0
    kc_source: str = ""
    start_n: Optional[str] = None
    end_n: Optional[str] = None
    tool_diameter: Optional[float] = None
    tool_radius: Optional[float] = None
    tool_material: str = ""
    blank_material: str = ""


__all__ = [
    "AutoLocator", "Dict", "FigureCanvasTkAgg", "INTERVAL_COLORS", "IS_FROZEN", "Line2D", "List", "MaxNLocator",
    "APP_DIR", "CONFIG_DIR", "NavigationToolbar2Tk", "Optional", "OUTPUT_DIR", "PITEntry", "PLOT_AX_BG",
    "PLOT_FIG_BG", "PLOT_FONT_BASE", "PROFILE_CACHE_DIR", "PROFILE_DIR", "PROJECT_ROOT",
    "PLOT_GRID_COLOR", "PLOT_SPINE_COLOR", "PLOT_TEXT_COLOR", "SMIF_ANNOTATION_COLOR", "SMIF_AX_BG",
    "SMIF_AXES_RECT", "SMIF_BOX_MIN_RATIO", "SMIF_BOX_ZOOM", "SMIF_COLORBAR_RECT", "SMIF_FIG_BG",
    "SMIF_GRID_COLOR", "SMIF_IDLE_COLOR", "SMIF_MAIN_AZIM", "SMIF_MAIN_ELEV", "SMIF_METRIC_IDLE_FILL",
    "SMIF_METRIC_NONSTEADY_FILL", "SMIF_METRIC_STEADY_FILL", "SMIF_NONSTEADY_COLOR", "SMIF_PANEL_BG",
    "SMIF_PANEL_EDGE", "SMIF_PANE_BG", "SMIF_PANE_EDGE", "SMIF_TEXT_COLOR", "SMIF_TEXT_MUTED",
    "RESOURCE_ROOT", "RUNTIME_DATA_DIR", "SAMPLE_DATA_DIR", "STYLE_MEASURED",
    "STYLE_PREDICTED", "TOOL_BG_COLORS", "Tuple", "UI_BTN_PADX", "UI_BTN_PADY", "UI_BTN_WIDTH",
    "UI_COLOR_BG_DARK", "UI_COLOR_BG_LIGHT", "UI_COLOR_BG_PANEL", "UI_COLOR_BORDER", "UI_COLOR_DANGER",
    "UI_COLOR_HEADER_BG", "UI_COLOR_PRIMARY", "UI_COLOR_PRIMARY_DARK", "UI_COLOR_PRIMARY_LIGHT",
    "UI_COLOR_SUCCESS", "UI_COLOR_TEXT", "UI_COLOR_TEXT_MUTED", "UI_COLOR_WARNING", "UI_FONT_BOLD",
    "UI_FONT_LARGE", "UI_FONT_NORMAL", "UI_FONT_SMALL", "UI_FONT_TITLE", "Union", "app_dir", "asdict",
    "base_dir", "center_dialog_on_parent", "collections", "datetime", "filedialog", "gc", "json",
    "math", "matplotlib", "messagebox", "np", "os", "pd", "plt", "re", "simhei_path", "sys", "tk",
    "ttk", "_get_pandas", "_get_sklearn",
]
