# ===== 性能优化：关键导入置顶，重型模块延迟加载 =====
import sys
import os
import gc

# 启动时立即优化GC和matplotlib后端
gc.set_threshold(10000, 10, 10)
os.environ['MPLBACKEND'] = 'TkAgg'

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import math
import re
import json
import collections
from dataclasses import dataclass, asdict
from typing import List, Tuple, Union, Dict, Optional
from datetime import datetime

# ===== 延迟导入重型模块（启动加速核心） =====
# 这些模块在首次使用时才加载
_numpy = None
_pandas = None
_matplotlib = None
_plt = None
_chardet = None
_scipy_signal = None
_sklearn_linear = None

def _get_numpy():
    global _numpy
    if _numpy is None:
        import numpy
        _numpy = numpy
    return _numpy

def _get_pandas():
    global _pandas
    if _pandas is None:
        import pandas
        _pandas = pandas
        _pandas.options.mode.chained_assignment = None
        _pandas.options.display.float_format = '{:.6f}'.format
    return _pandas

def _get_matplotlib():
    global _matplotlib, _plt
    if _matplotlib is None:
        import matplotlib
        _matplotlib = matplotlib
        _matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        _plt = plt
        # 优化matplotlib性能配置
        _matplotlib.rcParams['path.simplify'] = True
        _matplotlib.rcParams['path.simplify_threshold'] = 1.0
        _matplotlib.rcParams['agg.path.chunksize'] = 10000
        _matplotlib.rcParams['figure.max_open_warning'] = 0
        _matplotlib.rcParams['savefig.dpi'] = 100
        _matplotlib.rcParams['figure.autolayout'] = False
        _plt.ioff()  # 禁用交互模式
    return _matplotlib, _plt

def _get_chardet():
    global _chardet
    if _chardet is None:
        import chardet
        _chardet = chardet
    return _chardet

def _get_scipy_signal():
    global _scipy_signal
    if _scipy_signal is None:
        from scipy.signal import butter, filtfilt
        _scipy_signal = type('scipy_signal', (), {'butter': butter, 'filtfilt': filtfilt})()
    return _scipy_signal

def _get_sklearn():
    global _sklearn_linear
    if _sklearn_linear is None:
        from sklearn.linear_model import LinearRegression
        from sklearn.linear_model import HuberRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        _sklearn_linear = type('sklearn_linear', (), {
            'LinearRegression': LinearRegression,
            'HuberRegressor': HuberRegressor,
            'train_test_split': train_test_split,
            'mean_squared_error': mean_squared_error,
            'r2_score': r2_score
        })()
    return _sklearn_linear

# ===== 立即导入的基础模块（启动必需） =====
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, AutoLocator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd

# ===== 全局UI样式常量 - 工业科技风格 (DRY/KISS) =====
UI_FONT_NORMAL = ('Microsoft YaHei UI', 11)
UI_FONT_BOLD = ('Microsoft YaHei UI', 11, 'bold')
UI_FONT_LARGE = ('Microsoft YaHei UI', 13, 'bold')
UI_FONT_SMALL = ('Microsoft YaHei UI', 9)
UI_FONT_TITLE = ('Microsoft YaHei UI', 14, 'bold')
UI_BTN_WIDTH = 16
UI_BTN_PADX = 8
UI_BTN_PADY = 5

# 科技蓝+白色配色方案
UI_COLOR_PRIMARY = '#1E90FF'      # 主色调 - 科技蓝
UI_COLOR_PRIMARY_DARK = '#0066CC' # 深蓝色
UI_COLOR_PRIMARY_LIGHT = '#87CEEB'# 浅蓝色
UI_COLOR_SUCCESS = '#00CED1'      # 成功 - 青色
UI_COLOR_WARNING = '#FF8C00'      # 警告橙（用于特殊按钮）
UI_COLOR_DANGER = '#DC3545'       # 危险红
UI_COLOR_BG_DARK = '#1B4F72'      # 深蓝背景
UI_COLOR_BG_LIGHT = '#FFFFFF'     # 纯白背景
UI_COLOR_BG_PANEL = '#F0F8FF'     # 浅蓝面板背景
UI_COLOR_BORDER = '#B0C4DE'       # 浅蓝边框色
UI_COLOR_TEXT = '#1C2833'         # 深色文本
UI_COLOR_TEXT_MUTED = '#5D6D7E'   # 次要文本
UI_COLOR_HEADER_BG = '#E8F4FD'    # 标题背景

# 线条样式：实测/预测区分 - 科技蓝配色
STYLE_MEASURED = {"color": "#00B4D8", "linestyle": "-", "linewidth": 1.5, "label": "实测"}
STYLE_PREDICTED = {"color": "#FF6B6B", "linestyle": "-", "linewidth": 1.5, "label": "预测"}

# 图表样式 - 科技蓝白底风格
PLOT_FONT_BASE = 12
PLOT_TEXT_COLOR = "#1C2833"
PLOT_GRID_COLOR = "#D6EAF8"
PLOT_SPINE_COLOR = "#85C1E9"
PLOT_FIG_BG = "#FFFFFF"
PLOT_AX_BG = "#FAFEFF"

# 区间配色 - 高对比度区分色系
INTERVAL_COLORS = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']
TOOL_BG_COLORS = ['#AED6F1', '#F5B7B1', '#ABEBC6', '#FAD7A0', '#D7BDE2', '#A3E4D7']


# ===== 路径配置 =====
# 判断是否在打包环境中运行
if getattr(sys, 'frozen', False):
    # 打包环境 - exe所在目录（用于保存配置文件、rg文件等）
    # 使用 sys.executable 的目录作为应用目录，确保无论从哪里启动都能正确保存文件
    app_dir = os.path.dirname(os.path.abspath(sys.executable))
    # _MEIPASS临时目录（用于读取打包的资源文件）
    base_dir = getattr(sys, '_MEIPASS', app_dir)
    # 关键：将工作目录切换到exe所在目录，确保相对路径正确
    os.chdir(app_dir)
else:
    # 正常环境 - 使用脚本所在目录
    app_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = app_dir

# 设置黑体字体路径
simhei_path = os.path.join(base_dir, 'SimHei.ttf')

# 检查字体文件是否存在
if os.path.exists(simhei_path):
    # 添加字体到matplotlib
    import matplotlib.font_manager as fm
    fm.fontManager.addfont(simhei_path)
    
    # 设置matplotlib使用中文字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
else:
    print(f"警告: 字体文件 {simhei_path} 未找到，将使用系统默认字体")
# 设置matplotlib使用中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

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
    # 获取弹窗大小
    dialog_width = dialog.winfo_width()
    dialog_height = dialog.winfo_height()
    # 获取父窗口位置和大小
    parent_x = parent.winfo_rootx()
    parent_y = parent.winfo_rooty()
    parent_width = parent.winfo_width()
    parent_height = parent.winfo_height()
    # 计算居中位置
    x = parent_x + (parent_width - dialog_width) // 2
    y = parent_y + (parent_height - dialog_height) // 2
    # 确保不会超出屏幕边界
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
    start_n: Optional[str] = None
    end_n: Optional[str] = None
    tool_diameter: Optional[float] = None
    tool_radius: Optional[float] = None
    tool_material: str = ""
    blank_material: str = ""

class MillingAnalysisTool:
    def __init__(self, root):
        self.root = root
        self.root.title("🔬 铣削工艺信息分析工具 - 智能分析系统")
        
        # 配置科技感主题样式
        self.setup_tech_theme()
        
        # 获取屏幕尺寸并设置自适应窗口大小
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # 计算合适的窗口大小（屏幕的85%，但不超过最大尺寸，不小于最小尺寸）
        max_width, max_height = 1600, 1000
        min_width, min_height = 1000, 700
        
        window_width = min(max_width, max(min_width, int(screen_width * 0.85)))
        window_height = min(max_height, max(min_height, int(screen_height * 0.85)))
        
        # 计算居中位置
        center_x = (screen_width - window_width) // 2
        center_y = (screen_height - window_height) // 2
        
        # 设置窗口大小和位置
        self.root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")
        
        # 设置最小窗口大小
        self.root.minsize(min_width, min_height)
        
        # 使窗口可调整大小
        self.root.resizable(True, True)
        
        # 初始化所有变量
        self.input_file_path = tk.StringVar()
        self.input_file_paths = []
        self.input_file_count_var = tk.StringVar(value="")
        # 新流程：SampleData 导入后由用户手动绑定工艺信息表
        self.sample_bundle_path_var = tk.StringVar(value="")
        self.matched_process_file_var = tk.StringVar(value="未绑定工艺信息表")
        self.program_process_file_map = {}
        self.program_prompt_skip = {}
        self._loading_sample_data = False
        self.sample_csv_path = None
        self.sample_txt_path = None
        self.s_base = tk.DoubleVar(value=5000.0)  # 基准转速 (rpm)
        self.k_base = tk.DoubleVar(value=1.2)    # 基准转速下的扭矩系数 (N·m/(mm³/s))
        self.k_prime = tk.DoubleVar(value=1.2)   # 电流系数K' (A/(N·m))
        self.p_idle_var = tk.DoubleVar(value=0.0)  # 当前程序空载功率
        self.kc_coeff = tk.DoubleVar(value=1.2)    # 三参数模型中的K_c
        self.kc_sigma = tk.DoubleVar(value=0.0)    # K_c辨识标准差
        self.ke_coeff = tk.DoubleVar(value=0.0)    # 三参数模型中的K_e
        self.kc_beta = tk.DoubleVar(value=2.0)     # K_c保守上界系数
        self.current_program_speed = tk.DoubleVar(value=0.0)
        self.current_program_idle_power = tk.DoubleVar(value=0.0)
        self.current_program_idle_power_display = tk.StringVar(value="未计算")
        self.gcode_nc_path_var = tk.StringVar(value="")
        self.gcode_status_var = tk.StringVar(value="未导入G代码NC")
        self.no_load_csv_path_var = tk.StringVar(value="")
        self.no_load_status_var = tk.StringVar(value="未辨识空载功率")
        self.step_feed_csv_path_var = tk.StringVar(value="")
        self.step_feed_status_var = tk.StringVar(value="未辨识模型参数")
        self.idle_power_model = None
        self.idle_model_signature = ""
        self.step_feed_model_signature = ""
        self.gcode_profile = None
        self.pit_records = []
        self.data = []  # 存储处理后的数据
        self.figures = []  # 存储图表对象
        self.current_figure_index = 0  # 当前显示的图表索引
        self.figure_names = []  # 图表名称列表
        self.min_length = tk.IntVar(value=100)  # 最小区间长度
        self.batch_min_length = 5  # 添加批量处理专用的点数变量
        self.encoding_var = tk.StringVar(value="auto")  # 文件编码
        self.currents = None  # 电流数据
        self.cumulative_time = None  # 累计时间
        self.intervals = None  # 稳态区间
        self.processed_file_path = ""  # 处理后的文件路径
        self.processed_data_dir = None  # 添加这个实例变量
        self.batch_files = []  # 存储批量处理的文件列表
        self.rapid_speed_xy = tk.DoubleVar(value=4800.0)  # XY平面快速移动速度
        self.rapid_speed_z = tk.DoubleVar(value=3600.0)    # Z方向快速移动速度
        self.batch_rapid_speed_xy = tk.DoubleVar(value=4800.0)  # 批量处理XY平面快速移动速度
        self.batch_rapid_speed_z = tk.DoubleVar(value=3600.0)    # 批量处理Z方向快速移动速度

        # 机床原点（用于坐标/行程计算）
        self.origin_x = tk.DoubleVar(value=349.765)
        self.origin_y = tk.DoubleVar(value=-10.205)
        self.origin_z = tk.DoubleVar(value=-459.070)
        
        # 添加防抖动定时器，避免频繁调用resize
        self._resize_timer = None
        
        # 添加新变量（必须在创建标签页之前定义）
        self.tool_diameter = tk.StringVar(value="")  # PIT用刀具直径 (mm)
        self.tool_radius = tk.StringVar(value="")  # PIT用刀具半径 (mm)
        self.workpiece_material = tk.StringVar(value="")  # PIT用刀具材料
        self.blank_material = tk.StringVar(value="")  # PIT用毛坯材料
        self.batch_tool_diameter = tk.DoubleVar(value=10.0)  # 批量处理刀具直径
        self.batch_workpiece_material = tk.StringVar(value="硬质合金铝用铣刀")  # 批量处理刀具材料
        self.batch_blank_material = tk.StringVar(value="AL6061")  # 批量处理毛坯材料
        
        # 添加波动阈值变量
        self.threshold = tk.DoubleVar(value=0.2)  # 波动阈值 (20%)
        self.steady_threshold = tk.DoubleVar(value=0.2)  # 稳态区间划分的波动阈值
        self.actual_current_threshold = tk.DoubleVar(value=0.2)  # 实际电流稳态区间划分的波动阈值
        self.batch_threshold = tk.DoubleVar(value=0.2)  # 批量处理的波动阈值
        
        # 添加滤波相关变量
        self.cutoff_freq = tk.DoubleVar(value=0.1)  # 截止频率
        self.filter_order = tk.IntVar(value=4)  # 滤波器阶数
        
        # 预测功率稳态区间划分相关变量
        # 需求5：默认值设为1000，且不可小于1000
        self.pred_power_min_length = tk.IntVar(value=1000)  # 预测功率稳态区间最小采样点数（基于SampleData点数）
        self.enable_pred_power_steady = tk.BooleanVar(value=True)  # 是否启用预测功率稳态区间划分
        self.pred_power_intervals = []  # 存储预测功率稳态区间
        self._min_length_debounce_timer = None
        self._min_length_debounce_delay = 300
        self.adjustment_ratio = tk.DoubleVar(value=2.0)  # 优化倍率 R
        self.adjustment_ratio_display = tk.StringVar(value="2.00")
        self.filtered_data = None  # 滤波后的数据
        self.is_filtered = False  # 滤波状态标志
        
        # 实测数据对齐与联动显示
        self.sample_data_source = tk.IntVar(value=1)  # 0=电流,1=VGpro功率,2=边缘模块功率
        self.sample_display_mode = tk.StringVar(value="tool")  # 只使用tool模式（程序名+刀具号）
        self.sample_plot_mode = tk.StringVar(value="overlay")  # overlay/stacked
        self.sample_program_name = tk.StringVar()
        self.sample_tool_name = tk.StringVar()
        self.sample_avg_var = tk.StringVar(value="-")
        self.sample_ideal_var = tk.StringVar(value="-")
        self.sample_auto_status_var = tk.StringVar(value="请先点击“重新导入实测数据”")
        self._selection_change_job = None  # debounce timer handle for selection changes
        self._pending_selection_signature = None
        self._last_selection_signature = None
        self.sample_data_loaded = False
        self.sample_data_dir = None
        self.sample_programs = {}
        self.sample_data_values = None
        self.sample_data_values_raw = None
        self.sample_data_line_numbers = None
        self.sample_data_line_numbers_raw = None
        self.sample_data_point_indices = None
        self.sample_data_program_numbers = None
        self.sample_data_program_numbers_raw = None
        self.sample_data_x_positions = None
        self.sample_data_time_indices = None
        self.sample_data_base_blocks = []
        self.sample_data_valid_mask = None
        self.sample_data_valid_blocks = []
        self.process_valid_mask = None
        self.process_valid_blocks = []
        self.raw_to_aligned_line_map = {}
        self._ratio_update_lock = False
        self._ratio_dragging = False  # 滑动倍率时仅预览，不触发整图重绘
        self._ideal_line_artists = []
        self._preview_tool_key = None
        self._preview_tool_mean = None
        
        # 防抖机制：避免滑块快速移动时频繁更新
        self._ratio_debounce_timer = None
        self._ratio_debounce_delay = 150  # 毫秒，增加延迟减少刷新次数

        # 缓存按程序名切换时的处理结果，减少重复处理带来的卡顿
        self._process_cache = collections.OrderedDict()
        self._process_cache_limit = 4
        
        # 需求4：移除 trace_add，改为只在回车/失焦时触发更新
        # （不再使用 trace_add，避免输入过程中频繁刷新）
        
        # ===== ideal_store: 持久化rg存储 (核心数据结构) =====
        # 键: (program_name, tool_no)
        # 值: {"rg": float, "updated_at": str (ISO格式)}
        # 理想值 = 均值 × rg，运行时计算，不持久化
        self.ideal_store: Dict[Tuple[str, str], Dict] = {}
        # 配置文件保存到app_dir（exe所在目录），而非base_dir（临时目录）
        self.ideal_store_path = os.path.join(app_dir, "ideal_store.json")
        self.config_path = os.path.join(app_dir, "app_config.json")
        self.app_config: Dict = {}
        self._load_ideal_store()
        self._load_app_config()
        
        # 添加区间分割相关变量
        self.segment_points = []  # 存储分割点
        self.segment_lines = []  # 存储分割线对象
        self.segment_texts = []  # 存储分割点文字对象
        self.segments = []  # 存储分段数据和参数
        self.click_cid = None  # 点击事件连接ID
        
        # 添加分段参数管理
        self.current_segment_index = tk.IntVar(value=0)  # 当前选择的分段索引
        self.segment_params = {}  # 存储每个分段的参数 {segment_index: {参数字典}}
        self.segment_min_length = tk.IntVar(value=100)  # 当前分段的最小区间长度
        self.segment_threshold = tk.DoubleVar(value=0.2)  # 当前分段的波动阈值
        self.segment_abs_threshold = tk.DoubleVar(value=0.05)  # 当前分段的绝对波动阈值
        self.segment_reduce_interval = tk.BooleanVar(value=True)  # 当前分段的缩减区间边界
        
        # 图表交互功能变量
        self.original_xlim = None  # 原始x轴范围
        self.original_ylim = None  # 原始y轴范围
        self.scroll_cid = None  # 滚动事件连接ID
        self.zoom_factor = 1.2  # 缩放因子
        
        # 图表拖动功能变量
        self.is_panning = False  # 是否正在拖动
        self.pan_start = None  # 拖动起始位置
        
        # 创建标签页
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建工艺信息分析标签页
        self.data_processing_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_processing_tab, text="工艺信息分析")
        
        # 创建界面
        self.create_data_processing_tab()
        # self.create_steady_state_tab()  # 已合并到工艺信息分析页
        
        # 初始化图表
        self.init_figures()
        self.optimize_processing()  # 添加性能优化
        
        # 添加窗口大小变化监听器
        self.root.bind("<Configure>", self.on_window_resize)
        
        # 延迟调用图表大小自适应，确保所有组件都已创建完成
        self.root.after(100, self.adjust_figure_sizes)
        # 仅保留工艺信息分析页，无需实际负载页的自适应逻辑
        self.root.after(200, self.auto_load_sample_bundle)
        # 新流程不再启动时弹窗读取 SampleData；由“重新导入实测数据”触发
    




    def optimize_processing(self):
        """优化处理性能"""
        # 禁用matplotlib的交互模式
        plt.ioff()
        
        # 减少pandas内存使用
        if 'pandas' in sys.modules:
            pd.options.mode.chained_assignment = None  # 禁用链式赋值警告
            pd.options.display.float_format = '{:.6f}'.format
        
        # 优化matplotlib配置
        matplotlib.rcParams['path.simplify'] = True
        matplotlib.rcParams['path.simplify_threshold'] = 1.0
        matplotlib.rcParams['agg.path.chunksize'] = 10000
        
        # 额外性能优化
        matplotlib.rcParams['figure.max_open_warning'] = 0  # 禁用多图表警告
        matplotlib.rcParams['savefig.dpi'] = 100  # 减少保存图表时的DPI
        matplotlib.rcParams['figure.autolayout'] = False  # 禁用自动布局以提升性能
    
    def init_figures(self):
        """初始化图表 - 优化版：减少初始绘制开销"""
        # 延迟获取窗口大小，使用固定默认值加快启动
        fig_width = 14  # 固定尺寸加快初始化
        fig_height = 8
        
        # 数据处理标签页的图表 - 使用较低DPI加快启动
        self.fig_data, self.ax_data = plt.subplots(figsize=(fig_width, fig_height), dpi=80)
        
        # 设置白色背景
        self.fig_data.patch.set_facecolor('white')
        self.ax_data.set_facecolor('white')
        
        # 优化子图边距以居中对称显示，数据占2/3以上
        self.fig_data.subplots_adjust(left=0.10, right=0.90, top=0.94, bottom=0.08)
        
        # ⚠️ data_figure_frame 使用 grid 布局，这里必须用 grid，不能 pack
        # 先清空占位内容，避免 grid/pack 混用报错（必须在创建新画布之前清空）
        try:
            for w in self.data_figure_frame.winfo_children():
                w.destroy()
            self.data_figure_frame.grid_rowconfigure(0, weight=1)
            self.data_figure_frame.grid_columnconfigure(0, weight=1)
        except Exception:
            pass
        
        # 创建画布（在清空旧控件之后）
        self.canvas_data = FigureCanvasTkAgg(self.fig_data, master=self.data_figure_frame)
        canvas_widget = self.canvas_data.get_tk_widget()
        canvas_widget.grid(row=0, column=0, sticky="nsew")
        canvas_widget.configure(relief=tk.FLAT, bd=0)

        # 让图表随预览区域变化实时放缩（防抖）
        try:
            canvas_widget.bind('<Configure>', self._on_preview_canvas_configure)
        except Exception:
            pass
        
        # 为数据处理图表添加鼠标滚轮横向缩放功能
        self.canvas_data.mpl_connect('scroll_event', self.on_data_scroll_zoom)
        
        # 为数据处理图表添加鼠标左键横向拖动功能
        self.canvas_data.mpl_connect('button_press_event', self.on_data_pan_press)
        self.canvas_data.mpl_connect('button_release_event', self.on_data_pan_release)
        self.canvas_data.mpl_connect('motion_notify_event', self.on_data_pan_motion)
        
        # 显示初始提示 - 使用draw_idle减少阻塞
        self.show_initial_message()
    
    def show_initial_message(self):
        """显示初始提示信息 - 优化版"""
        self.ax_data.clear()
        self.ax_data.set_facecolor('white')
        self.ax_data.set_xlim(0, 1)
        self.ax_data.set_ylim(0, 1)
        self.ax_data.text(
            0.5,
            0.5,
            '正在自动加载数据，请稍候...',
            horizontalalignment='center',
            verticalalignment='center',
            transform=self.ax_data.transAxes,
            fontsize=20,
            fontweight='bold',
            color='#333333'
        )
        self.ax_data.set_anchor('C')
        self.ax_data.axis('off')
        # 使用draw_idle而非draw，减少阻塞
        self.canvas_data.draw_idle()
        
    def create_data_processing_tab(self):
        """工艺信息分析页面：科技蓝+白色风格，布局参考用户图片
        布局：上方当前程序 -> 中间平均功率+区间分析 -> 右侧稳态区间详情 -> 下方图表预览"""
        # 页面总体：0=主体(可伸缩)，1=进度条，2=状态栏
        self.data_processing_tab.grid_columnconfigure(0, weight=1)
        self.data_processing_tab.grid_rowconfigure(0, weight=1)
        self.data_processing_tab.grid_rowconfigure(1, weight=0)
        self.data_processing_tab.grid_rowconfigure(2, weight=0)

        # 上下可调分区（上：控件区；下：图表预览）
        paned = ttk.PanedWindow(self.data_processing_tab, orient=tk.VERTICAL)
        paned.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self._data_paned = paned

        controls = ttk.Frame(paned)
        preview = ttk.Frame(paned)
        self._data_controls = controls

        paned.add(controls, weight=0)
        paned.add(preview, weight=1)

        # ===== 控件区布局 =====
        controls.grid_columnconfigure(0, weight=3)  # 左侧控件区占更多空间
        controls.grid_columnconfigure(1, weight=1)  # 右侧稳态区间详情

        # ===== 左侧主控件区 =====
        left_frame = ttk.Frame(controls)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left_frame.grid_columnconfigure(0, weight=1)

        # ===== 第一行：当前程序 =====
        program_frame = ttk.LabelFrame(left_frame, text="📊 当前程序", padding=8, style='Tech.TLabelframe')
        program_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        program_frame.grid_columnconfigure(1, weight=1)
        program_frame.grid_columnconfigure(3, weight=1)
        program_frame.grid_columnconfigure(5, weight=1)

        # 程序名
        ttk.Label(program_frame, text="程序名:", font=UI_FONT_NORMAL).grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.sample_program_combo = ttk.Combobox(program_frame, textvariable=self.sample_program_name,
                                                 state="readonly", width=20, font=UI_FONT_NORMAL)
        self.sample_program_combo.grid(row=0, column=1, padx=(0, 16), sticky="w")
        self.sample_program_combo.bind("<<ComboboxSelected>>", self.on_sample_program_selected)

        # 刀具号
        ttk.Label(program_frame, text="刀具号:", font=UI_FONT_NORMAL).grid(row=0, column=2, sticky="w", padx=(0, 4))
        self.sample_tool_combo = ttk.Combobox(program_frame, textvariable=self.sample_tool_name,
                                              state="readonly", width=16, font=UI_FONT_NORMAL)
        self.sample_tool_combo.grid(row=0, column=3, padx=(0, 16), sticky="w")
        self.sample_tool_combo.bind("<<ComboboxSelected>>", self.on_sample_selection_change)

        # 数据源
        ttk.Label(program_frame, text="数据源:", font=UI_FONT_NORMAL).grid(row=0, column=4, sticky="w", padx=(0, 4))
        source_frame = ttk.Frame(program_frame)
        source_frame.grid(row=0, column=5, sticky="w")
        self.sample_source_buttons = []
        for txt, val in [("电流", 0), ("VGpro功率", 1), ("边缘模块功率", 2)]:
            rb = ttk.Radiobutton(source_frame, text=txt, variable=self.sample_data_source, value=val,
                                 command=self.on_sample_selection_change, style='Tech.TRadiobutton')
            rb.pack(side=tk.LEFT, padx=(0, 8))
            self.sample_source_buttons.append(rb)

        # ===== 第二行：机理辨识 =====
        model_frame = ttk.LabelFrame(left_frame, text="🧠 机理辨识", padding=8, style='Tech.TLabelframe')
        model_frame.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        model_frame.grid_columnconfigure(7, weight=1)

        self.import_nc_btn = ttk.Button(
            model_frame, text="📄 导入G代码NC", command=self.browse_nc_file, width=15, style='Tech.TButton'
        )
        self.import_nc_btn.grid(row=0, column=0, padx=(0, 6), pady=(0, 6), sticky="w")

        self.identify_idle_btn = ttk.Button(
            model_frame, text="🌀 辨识空载功率", command=self.identify_no_load_power, width=15, style='Tech.TButton'
        )
        self.identify_idle_btn.grid(row=0, column=1, padx=(0, 6), pady=(0, 6), sticky="w")

        self.identify_model_btn = ttk.Button(
            model_frame, text="📐 辨识模型参数", command=self.identify_step_feed_parameters, width=15, style='Tech.TButton'
        )
        self.identify_model_btn.grid(row=0, column=2, padx=(0, 6), pady=(0, 6), sticky="w")

        self.pit_display_btn = ttk.Button(
            model_frame, text="📋 PIT显示", command=self.show_pit_dialog, width=12, style='Secondary.TButton',
            state="disabled"
        )
        self.pit_display_btn.grid(row=0, column=3, padx=(0, 6), pady=(0, 6), sticky="w")

        ttk.Label(model_frame, text="P_idle(W):", font=UI_FONT_NORMAL).grid(row=1, column=0, sticky="w", padx=(0, 4))
        self.p_idle_entry = ttk.Entry(model_frame, textvariable=self.p_idle_var, width=10, font=UI_FONT_NORMAL)
        self.p_idle_entry.grid(row=1, column=1, sticky="w", padx=(0, 10))
        self.p_idle_entry.bind("<Return>", self.on_model_param_commit)
        self.p_idle_entry.bind("<FocusOut>", self.on_model_param_commit)

        ttk.Label(model_frame, text="K_c:", font=UI_FONT_NORMAL).grid(row=1, column=2, sticky="w", padx=(0, 4))
        self.kc_entry = ttk.Entry(model_frame, textvariable=self.kc_coeff, width=10, font=UI_FONT_NORMAL)
        self.kc_entry.grid(row=1, column=3, sticky="w", padx=(0, 10))
        self.kc_entry.bind("<Return>", self.on_model_param_commit)
        self.kc_entry.bind("<FocusOut>", self.on_model_param_commit)

        ttk.Label(model_frame, text="K_e:", font=UI_FONT_NORMAL).grid(row=1, column=4, sticky="w", padx=(0, 4))
        self.ke_entry = ttk.Entry(model_frame, textvariable=self.ke_coeff, width=10, font=UI_FONT_NORMAL)
        self.ke_entry.grid(row=1, column=5, sticky="w", padx=(0, 10))
        self.ke_entry.bind("<Return>", self.on_model_param_commit)
        self.ke_entry.bind("<FocusOut>", self.on_model_param_commit)

        ttk.Label(model_frame, text="程序空载:", font=UI_FONT_NORMAL).grid(row=1, column=6, sticky="w", padx=(0, 4))
        self.program_idle_summary_entry = ttk.Entry(
            model_frame, textvariable=self.current_program_idle_power_display,
            width=30, font=UI_FONT_NORMAL, state="readonly"
        )
        self.program_idle_summary_entry.grid(row=1, column=7, sticky="ew", padx=(0, 6))

        self.program_idle_detail_btn = ttk.Button(
            model_frame, text="📊 查看明细", command=self.show_program_idle_detail_dialog,
            width=12, style='Secondary.TButton', state="disabled"
        )
        self.program_idle_detail_btn.grid(row=1, column=8, sticky="w")

        ttk.Label(
            model_frame, textvariable=self.gcode_status_var, font=UI_FONT_SMALL,
            foreground=UI_COLOR_TEXT_MUTED
        ).grid(row=2, column=0, columnspan=9, sticky="w", pady=(2, 0))
        ttk.Label(
            model_frame, textvariable=self.no_load_status_var, font=UI_FONT_SMALL,
            foreground=UI_COLOR_TEXT_MUTED
        ).grid(row=3, column=0, columnspan=9, sticky="w", pady=(2, 0))
        ttk.Label(
            model_frame, textvariable=self.step_feed_status_var, font=UI_FONT_SMALL,
            foreground=UI_COLOR_TEXT_MUTED
        ).grid(row=4, column=0, columnspan=9, sticky="w", pady=(2, 0))

        # ===== 第三行：平均功率信息（无框头） =====
        power_frame = ttk.Frame(left_frame, padding=(8, 4))
        power_frame.grid(row=2, column=0, sticky="ew", pady=(0, 6))

        # 平均功率显示
        ttk.Label(power_frame, text="⚡ 平均功率:", font=UI_FONT_BOLD, foreground=UI_COLOR_PRIMARY).pack(
            side=tk.LEFT, padx=(0, 4))
        self.sample_avg_label = ttk.Label(power_frame, textvariable=self.sample_avg_var, 
                                          font=UI_FONT_BOLD, foreground="#0066CC")
        self.sample_avg_label.pack(side=tk.LEFT, padx=(0, 24))

        # 优化倍率滑块
        ttk.Label(power_frame, text="⚙ 优化倍率:", font=UI_FONT_BOLD, foreground=UI_COLOR_PRIMARY).pack(
            side=tk.LEFT, padx=(0, 4))
        rg_frame = ttk.Frame(power_frame)
        rg_frame.pack(side=tk.LEFT, padx=(0, 8))
        self.adjustment_ratio_scale = tk.Scale(
            rg_frame, from_=0.1, to=5.0, resolution=0.05, orient=tk.HORIZONTAL,
            variable=self.adjustment_ratio, length=140, showvalue=0,
            bg='white', troughcolor='#D6EAF8', highlightthickness=0,
            command=self._on_ratio_scale_change  # 使用 command 替代 trace_add 精确控制更新
        )
        self.adjustment_ratio_scale.pack(side=tk.LEFT)
        self.adjustment_ratio_scale.bind("<ButtonPress-1>", self._on_ratio_press)
        self.adjustment_ratio_scale.bind("<ButtonRelease-1>", self._on_ratio_release)
        self.rg_entry = ttk.Entry(rg_frame, textvariable=self.adjustment_ratio_display, width=6, font=UI_FONT_NORMAL)
        self.rg_entry.pack(side=tk.LEFT, padx=(6, 0))
        self.rg_entry.bind("<Return>", self._on_rg_entry_commit)
        self.rg_entry.bind("<FocusOut>", self._on_rg_entry_commit)

        # 理想功率显示（紧贴优化倍率）
        ttk.Label(power_frame, text="🎯 理想功率:", font=UI_FONT_BOLD, foreground=UI_COLOR_PRIMARY).pack(
            side=tk.LEFT, padx=(8, 4))
        self.sample_ideal_label = ttk.Label(power_frame, textvariable=self.sample_ideal_var,
                                            font=UI_FONT_BOLD, foreground="#E67300")
        self.sample_ideal_label.pack(side=tk.LEFT)

        # ===== 第四行：区间分析 =====
        interval_frame = ttk.LabelFrame(left_frame, text="📐 区间分析", padding=8, style='Tech.TLabelframe')
        interval_frame.grid(row=3, column=0, sticky="ew", pady=(0, 6))
        interval_frame.grid_columnconfigure(4, weight=1)

        # 第一行：最小样本点、区间数量、[弹性空间]、导入工艺信息文件、保存结果
        ttk.Label(interval_frame, text="最小样本点:", font=UI_FONT_NORMAL).grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.pred_power_min_length_entry = ttk.Entry(interval_frame, textvariable=self.pred_power_min_length,
                                                     width=10, font=UI_FONT_NORMAL)
        self.pred_power_min_length_entry.grid(row=0, column=1, padx=(0, 20), sticky="w")
        self.pred_power_min_length_entry.bind(
            "<Return>", lambda e: self._schedule_min_length_update(immediate=True)
        )
        self.pred_power_min_length_entry.bind(
            "<FocusOut>", lambda e: self._schedule_min_length_update(immediate=True)
        )

        # 区间数量显示
        ttk.Label(interval_frame, text="区间数量:", font=UI_FONT_NORMAL).grid(row=0, column=2, sticky="w", padx=(0, 4))
        self.interval_count_var = tk.StringVar(value="0")
        self.interval_count_label = ttk.Label(interval_frame, textvariable=self.interval_count_var,
                                             font=UI_FONT_BOLD, foreground="#2E7D32")
        self.interval_count_label.grid(row=0, column=3, sticky="w", padx=(0, 20))

        # 导入工艺信息文件按钮（橙色强调）- 移到最右边
        self.choose_process_btn = ttk.Button(interval_frame, text="📂 导入工艺信息文件", 
                                             command=self.choose_process_file_for_current_program, 
                                             width=18, style='Orange.TButton')
        self.choose_process_btn.grid(row=0, column=5, padx=(0, 8), sticky="e")

        # 保存结果按钮（橙色强调）- 移到最右边
        self.export_i_code_btn = ttk.Button(interval_frame, text="💾 保存结果", 
                                            command=self.save_interval_info, width=12,
                                            style='Orange.TButton',
                                            state="disabled")
        self.export_i_code_btn.grid(row=0, column=6, sticky="e")

        # 第二行：显示方式切换按钮
        plot_switch_frame = ttk.Frame(interval_frame)
        plot_switch_frame.grid(row=1, column=0, columnspan=7, pady=(8, 0), sticky="w")
        ttk.Label(plot_switch_frame, text="图表预览:", font=UI_FONT_SMALL).pack(side=tk.LEFT, padx=(0, 6))
        self.overlay_btn = ttk.Button(plot_switch_frame, text="● 叠加显示", width=12,
                                      style='Secondary.TButton', command=lambda: self._set_plot_mode("overlay"))
        self.overlay_btn.pack(side=tk.LEFT, padx=2)
        self.stacked_btn = ttk.Button(plot_switch_frame, text="● 上下显示", width=12,
                                      style='Secondary.TButton', command=lambda: self._set_plot_mode("stacked"))
        self.stacked_btn.pack(side=tk.LEFT, padx=2)

        # 默认选择叠加显示
        self.overlay_btn.state(['pressed'])
        self.stacked_btn.state(['!pressed'])

        # ===== 右侧：稳态区间详情 =====
        ideal_frame = ttk.LabelFrame(controls, text="📌 稳态区间详情", padding=6, style='Tech.TLabelframe')
        ideal_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 0))
        ideal_frame.grid_rowconfigure(0, weight=1)
        ideal_frame.grid_columnconfigure(0, weight=1)

        self.ideal_tree = ttk.Treeview(ideal_frame, show="tree", height=8)
        self.ideal_tree.grid(row=0, column=0, sticky="nsew")
        ideal_scroll = ttk.Scrollbar(ideal_frame, orient=tk.VERTICAL, command=self.ideal_tree.yview)
        ideal_scroll.grid(row=0, column=1, sticky="ns")
        self.ideal_tree.configure(yscrollcommand=ideal_scroll.set)
        self.ideal_tree.bind("<<TreeviewSelect>>", self._on_ideal_tree_select)
        self._refresh_ideal_tree()

        # 收集控件引用
        self.sample_control_widgets = []
        self.sample_control_widgets.extend(self.sample_source_buttons)
        self.sample_control_widgets.append(self.sample_program_combo)
        self.sample_control_widgets.append(self.sample_tool_combo)
        self.sample_control_widgets.append(self.choose_process_btn)
        self.sample_control_widgets.append(self.export_i_code_btn)
        
        # ===== 预览区 =====
        preview.grid_columnconfigure(0, weight=1)
        preview.grid_rowconfigure(0, weight=1)

        nav_frame = ttk.Frame(preview)
        nav_frame.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        nav_frame.grid_columnconfigure(1, weight=1)

        # 图表标题区（保留引用但不显示文字）
        self.figure_label = ttk.Label(nav_frame, text="", font=UI_FONT_LARGE, foreground=UI_COLOR_PRIMARY)

        # 保留 figure_selector_var 以兼容原有代码
        self.figure_selector_var = tk.StringVar()
        self.figure_selector = ttk.Combobox(nav_frame, textvariable=self.figure_selector_var, state="readonly")

        # 保留prev_btn/next_btn引用以兼容原有代码
        self.prev_btn = ttk.Button(nav_frame, state=tk.DISABLED)
        self.next_btn = ttk.Button(nav_frame, state=tk.DISABLED)

        # 预览容器：铺满剩余空间
        self.data_figure_frame = ttk.LabelFrame(preview, text="📈 实测负载预览（滚轮：横向缩放；Ctrl+滚轮：纵向缩放）", 
                                                padding=4, style='Tech.TLabelframe')
        self.data_figure_frame.grid(row=0, column=0, sticky="nsew")
        self.data_figure_frame.grid_rowconfigure(0, weight=1)
        self.data_figure_frame.grid_columnconfigure(0, weight=1)
        self.data_figure_frame.bind("<Configure>", self._on_preview_canvas_configure)

        # 初始化空画布占位
        placeholder = ttk.Label(self.data_figure_frame, text="请先导入 SampleData.csv / SampleData.txt 并生成图表",
                                foreground="#5D6D7E", anchor="center")
        placeholder.grid(row=0, column=0, sticky="nsew")


        # ===== 进度条与状态栏 =====
        progress_bar = ttk.Progressbar(self.data_processing_tab, orient=tk.HORIZONTAL, length=100, mode='determinate')
        progress_bar.grid(row=1, column=0, sticky="ew", padx=8)
        self.data_progress_bar = progress_bar

        if not hasattr(self, 'status_var_data'):
            default_status = self.sample_auto_status_var.get() if hasattr(self, 'sample_auto_status_var') else "就绪"
            self.status_var_data = tk.StringVar(value=default_status)
        self.sample_auto_status_var = self.status_var_data
        status_bar = ttk.Label(self.data_processing_tab, textvariable=self.status_var_data, relief=tk.SUNKEN, anchor=tk.W, background='#E8F4FD')
        status_bar.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 6))

        # 初始化分割条位置
        def _init_sash():
            try:
                total_h = paned.winfo_height()
                if total_h <= 10:
                    return
                ctrl_h = int(total_h * 0.24)
                ctrl_h = max(180, min(ctrl_h, 280))
                paned.sashpos(0, ctrl_h)
            except Exception:
                pass

        self.root.after(60, _init_sash)
        self.on_sample_display_mode_change()

    def set_input_files(self, file_paths):
        """设置导入的工艺信息表"""
        file_paths = [p for p in file_paths if p]
        self.input_file_paths = file_paths
        self.ensure_sample_data_matches_inputs(file_paths)
        self.reset_processing_state()
        if len(file_paths) == 1:
            self.input_file_path.set(file_paths[0])
            if hasattr(self, "matched_process_file_var"):
                self.matched_process_file_var.set(file_paths[0])
            self.input_file_count_var.set("")
            self.set_sample_controls_enabled(True, refresh=False)
            sample_dir = os.path.dirname(file_paths[0])
            resolved_dir, csv_path, txt_path = self.resolve_sampledata_files(sample_dir)
            sample_files_ready = bool(resolved_dir and csv_path and txt_path)
            if sample_files_ready:
                sample_dir_norm = os.path.normcase(os.path.normpath(os.path.abspath(resolved_dir)))
                current_sample_dir = ""
                if self.sample_data_dir:
                    current_sample_dir = os.path.normcase(os.path.normpath(os.path.abspath(self.sample_data_dir)))
                if (not self.sample_data_loaded) or (not current_sample_dir) or (current_sample_dir != sample_dir_norm):
                    self.load_sample_data_from_paths(csv_path, txt_path, silent=True, sample_dir=resolved_dir)
            elif self.sample_data_loaded:
                if hasattr(self, "sample_auto_status_var"):
                    self.sample_auto_status_var.set("未找到SampleData，保留已加载实测数据")
                if hasattr(self, "status_var_data"):
                    self.status_var_data.set("未找到SampleData，保留已加载实测数据")
            if self.sample_data_loaded:
                self.show_sample_preview()
            else:
                self.show_initial_message()
        elif len(file_paths) > 1:
            self.input_file_path.set(file_paths[0])
            self.input_file_count_var.set(f"已选择 {len(file_paths)} 个工艺信息表")
            if hasattr(self, "matched_process_file_var"):
                self.matched_process_file_var.set(f"多文件模式：{len(file_paths)} 个工艺信息表")
            self.show_multi_input_message(len(file_paths))
            self.set_sample_controls_enabled(False, refresh=False)
        else:
            self.input_file_path.set("")
            self.input_file_count_var.set("")
            if hasattr(self, "matched_process_file_var"):
                self.matched_process_file_var.set("未绑定工艺信息表")
            if self.sample_data_loaded:
                self.show_sample_preview()
            else:
                self.show_initial_message()
            self.set_sample_controls_enabled(True, refresh=False)

    def on_input_entry_change(self, event=None):
        """手动输入路径时同步导入工艺信息表状态"""
        entry_path = self.input_file_path.get().strip()
        if not entry_path:
            self.set_input_files([])
            return
        self.set_input_files([entry_path])
    
    def browse_input_file(self):
        """浏览工艺信息表"""
        file_paths = filedialog.askopenfilenames(
            title="选择工艺信息表",
            filetypes=(("文本文件", "*.txt"), ("所有文件", "*.*"))
        )
        if file_paths:
            self.set_input_files(list(file_paths))

    def get_input_files(self):
        """获取已导入的工艺信息表文件列表"""
        entry_path = self.input_file_path.get().strip()
        if self.input_file_paths:
            if entry_path and entry_path not in self.input_file_paths:
                return [entry_path]
            return [p for p in self.input_file_paths if p]
        return [entry_path] if entry_path else []

    def get_primary_input_file(self):
        """获取主工艺信息表文件路径"""
        files = self.get_input_files()
        return files[0] if files else ""

    def show_multi_input_message(self, count):
        """多文件模式提示"""
        if not hasattr(self, 'ax_data'):
            return
        self.ax_data.clear()
        self.ax_data.set_facecolor('white')
        self.ax_data.set_xlim(0, 1)
        self.ax_data.set_ylim(0, 1)
        self.ax_data.text(
            0.5,
            0.5,
            f'已导入 {count} 个工艺信息表\n批量处理不显示预览',
            horizontalalignment='center',
            verticalalignment='center',
            transform=self.ax_data.transAxes,
            fontsize=18,
            fontweight='bold',
            color='#333333'
        )
        self.ax_data.set_anchor('C')
        self.ax_data.axis('off')
        self.canvas_data.draw_idle()
        self.figures = []
        self.figure_names = []
        self.update_nav_buttons()

    def load_sample_bundle_from_dir(self, base_dir, silent=False):
        """按目录自动加载 SampleData"""
        resolved_dir, csv_path, txt_path = self.resolve_sampledata_files(base_dir)
        if not resolved_dir:
            if not silent:
                messagebox.showerror(
                    "文件缺失",
                    "未找到 SampleData.csv 或 SampleData.txt（可放在同目录或 SampleData 子目录）"
                )
            if hasattr(self, "sample_auto_status_var"):
                self.sample_auto_status_var.set("未找到SampleData.csv或SampleData.txt")
            if hasattr(self, "status_var_data"):
                self.status_var_data.set("未发现实测数据文件，已跳过导入")
            return False

        if not self.load_sample_data_from_paths(csv_path, txt_path, silent=silent, sample_dir=resolved_dir):
            return False

        if hasattr(self, "sample_auto_status_var"):
            self.sample_auto_status_var.set("SampleData已导入；请为程序选择工艺信息表")
        if hasattr(self, "status_var_data"):
            self.status_var_data.set("已导入SampleData；请为程序选择工艺信息表")
        return True

    def auto_load_sample_bundle(self):
        """启动后自动加载 SampleData 并自动执行处理"""
        if self.sample_data_loaded:
            return
        # 从exe所在目录加载SampleData（而非_MEIPASS临时目录）
        success = self.load_sample_bundle_from_dir(app_dir, silent=True)
        if success and self.get_input_files():
            # 自动执行处理
            self.root.after(100, self._auto_process_after_load)
    
    def _get_process_cache_key(self, input_file):
        if not input_file:
            return None
        return os.path.normcase(os.path.abspath(input_file))

    def _get_process_signature(self, input_file):
        try:
            mtime = os.path.getmtime(input_file)
        except Exception:
            mtime = None
        return (
            mtime,
            float(self.s_base.get()),
            float(self.k_base.get()),
            float(self.rapid_speed_xy.get()),
            float(self.rapid_speed_z.get()),
            float(self.origin_x.get()),
            float(self.origin_y.get()),
            float(self.origin_z.get()),
            str(self.tool_diameter.get()).strip(),
            str(self.tool_radius.get()).strip(),
            str(self.workpiece_material.get()),
            str(self.blank_material.get()),
            float(self.p_idle_var.get()),
            float(self.kc_coeff.get()),
            float(self.kc_sigma.get()),
            float(self.ke_coeff.get()),
            float(self.current_program_speed.get()),
            float(self.current_program_idle_power.get()),
            str(self.gcode_nc_path_var.get()),
            str(self.idle_model_signature),
            str(self.step_feed_model_signature),
        )

    def _load_cached_process(self, cache_key, signature):
        cached = self._process_cache.get(cache_key)
        if not cached or cached.get("signature") != signature:
            return False
        self._process_cache.move_to_end(cache_key)
        self.data = cached.get("data", [])
        self.processed_file_path = cached.get("processed_file_path", "")
        self.processed_data_dir = cached.get("processed_data_dir")
        self.raw_to_aligned_line_map = cached.get("raw_to_aligned_line_map", {})
        if self.sample_data_loaded:
            self.align_sample_data_to_processed()
        return True

    def _store_process_cache(self, cache_key, signature):
        if not cache_key:
            return
        self._process_cache[cache_key] = {
            "signature": signature,
            "data": self.data,
            "processed_file_path": self.processed_file_path,
            "processed_data_dir": self.processed_data_dir,
            "raw_to_aligned_line_map": self.raw_to_aligned_line_map,
        }
        self._process_cache.move_to_end(cache_key)
        while len(self._process_cache) > self._process_cache_limit:
            self._process_cache.popitem(last=False)

    def _has_processed_result_for(self, process_path):
        """判断指定工艺信息文件是否已处理（当前或缓存有效）。"""
        if not process_path or not os.path.exists(process_path):
            return False
        try:
            process_abs = os.path.normcase(os.path.abspath(process_path))
            if self.processed_file_path and os.path.normcase(os.path.abspath(self.processed_file_path)) == process_abs:
                return bool(self.data)
            cache_key = self._get_process_cache_key(process_path)
            signature = self._get_process_signature(process_path)
            cached = self._process_cache.get(cache_key)
            if not cached:
                return False
            return cached.get("signature") == signature and bool(cached.get("data"))
        except Exception:
            return False

    def _process_current_input_for_preview(self):
        """处理当前工艺信息表并刷新预测负载图表"""
        input_files = self.get_input_files()
        if len(input_files) != 1:
            return False
        input_file = input_files[0]
        if not input_file or not os.path.exists(input_file):
            return False
        try:
            cache_key = self._get_process_cache_key(input_file)
            signature = self._get_process_signature(input_file)
            if cache_key and self._load_cached_process(cache_key, signature):
                self.generate_plots(save=False, silent=True)
                if self.figures:
                    self.show_current_figure(0)
                # 导入工艺信息表后立即刷新“已设定理想值”视图
                self._refresh_ideal_tree()
                self._refresh_current_ideal_display()
                if hasattr(self, "status_var_data"):
                    self.status_var_data.set("已使用缓存刷新图表")
                return True
            if hasattr(self, "status_var_data"):
                self.status_var_data.set("正在自动处理...")
            try:
                self.root.update_idletasks()
            except Exception:
                pass
            success = self.process_single_file(input_file)
            if not success:
                if hasattr(self, "status_var_data"):
                    self.status_var_data.set("自动处理失败")
                return False
            self._store_process_cache(cache_key, signature)
            self.generate_plots(save=False, silent=True)
            if self.figures:
                self.show_current_figure(0)
            # 导入工艺信息表后立即刷新“已设定理想值”视图
            self._refresh_ideal_tree()
            self._refresh_current_ideal_display()
            if hasattr(self, "status_var_data"):
                self.status_var_data.set("自动处理完成，可调整优化倍率后保存")
            return True
        except Exception as e:
            if hasattr(self, "status_var_data"):
                self.status_var_data.set(f"自动处理出错: {str(e)[:50]}")
            return False

    def _auto_process_after_load(self):
        """加载数据后自动执行处理"""
        self._process_current_input_for_preview()

    def browse_sample_bundle(self):
        """导入 SampleData.csv/txt"""
        file_path = filedialog.askopenfilename(
            title="选择 SampleData.csv 或 SampleData.txt",
            filetypes=(
                ("SampleData文件", ("SampleData.csv", "SampleData.txt", "*.csv", "*.txt")),
                ("所有文件", "*.*"),
            )
        )
        if not file_path:
            return

        base_dir = os.path.dirname(file_path)
        self.load_sample_bundle_from_dir(base_dir, silent=False)

    def _columns_look_like_data(self, columns):
        suspicious = 0
        for col in columns:
            text = str(col).strip()
            if not text or text.lower().startswith("unnamed"):
                suspicious += 1
                continue
            try:
                float(text)
                suspicious += 1
            except Exception:
                pass
        return suspicious >= max(1, len(columns) // 2)

    def _read_csv_flex(self, file_path):
        """尽可能兼容不同编码/分隔符/表头形式的CSV。"""
        pd_mod = _get_pandas()
        last_error = None
        for encoding in ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'latin-1']:
            for header in ['infer', None]:
                try:
                    kwargs = {
                        "sep": None,
                        "engine": "python",
                        "encoding": encoding,
                    }
                    if header is None:
                        kwargs["header"] = None
                    df = pd_mod.read_csv(file_path, **kwargs)
                    if df is None or df.empty:
                        continue
                    if header == 'infer' and self._columns_look_like_data(df.columns):
                        continue
                    if header is None:
                        df.columns = [f"col_{idx}" for idx in range(len(df.columns))]
                    df.columns = [str(col).strip() for col in df.columns]
                    return df
                except Exception as exc:
                    last_error = exc
        raise ValueError(f"无法解析CSV文件: {last_error}")

    def _normalize_column_name(self, name):
        return re.sub(r'[\s\-_()/\\\[\]{}%]+', '', str(name).strip().lower())

    def _find_numeric_candidate_columns(self, df, min_non_na=3):
        candidates = []
        for col in df.columns:
            series = pd.to_numeric(df[col], errors='coerce')
            if int(series.notna().sum()) >= min_non_na:
                candidates.append(col)
        return candidates

    def _find_matching_column(self, df, aliases, fallback_index=None):
        normalized_map = {col: self._normalize_column_name(col) for col in df.columns}
        normalized_aliases = [self._normalize_column_name(alias) for alias in aliases]

        for col, normalized in normalized_map.items():
            if normalized in normalized_aliases:
                return col

        for col, normalized in normalized_map.items():
            for alias in normalized_aliases:
                if alias and alias in normalized:
                    return col

        numeric_cols = self._find_numeric_candidate_columns(df)
        if fallback_index is not None and len(numeric_cols) > fallback_index:
            return numeric_cols[fallback_index]
        return None

    def _parse_tagged_tokens(self, line):
        tokens = []
        for token in str(line).strip().split(','):
            token = token.strip()
            if not token:
                continue
            tokens.append(token.strip('<> ').strip())
        return tokens

    def _looks_like_channel_export(self, file_path):
        encoding = self.detect_file_encoding(file_path)
        try:
            with open(file_path, 'r', encoding=encoding, errors='ignore') as infile:
                head = infile.read(65536)
        except Exception:
            return False
        return '<ChannelInfo>' in head and ('<ChannelData>' in head or '<Scope>' in head)

    def _find_channel_index(self, channel_infos, required_tokens):
        normalized_required = [str(token).strip() for token in required_tokens if str(token).strip()]
        for idx, tokens in enumerate(channel_infos):
            normalized_tokens = [str(token).strip() for token in tokens]
            if all(any(req == token or req in token for token in normalized_tokens) for req in normalized_required):
                return idx
        return None

    def _extract_idle_points_from_plain_csv(self, file_path):
        df = self._read_csv_flex(file_path)
        speed_col = self._find_matching_column(
            df, ["spindle_speed", "speed", "rpm", "主轴转速", "转速"], fallback_index=0
        )
        power_col = self._find_matching_column(
            df, ["idle_power", "power", "空载功率", "主轴功率", "功率", "load_power"], fallback_index=1
        )
        if not speed_col or not power_col:
            raise ValueError("未识别到转速列或功率列，请检查CSV表头")

        idle_df = df[[speed_col, power_col]].copy()
        idle_df.columns = ["speed", "power"]
        idle_df["speed"] = pd.to_numeric(idle_df["speed"], errors='coerce')
        idle_df["power"] = pd.to_numeric(idle_df["power"], errors='coerce')
        idle_df = idle_df.dropna()
        idle_df = idle_df[idle_df["speed"] > 0]
        if idle_df.empty:
            raise ValueError("CSV中没有有效的转速-空载功率数据")

        idle_df["speed"] = idle_df["speed"].apply(lambda val: float(round(float(val) / 100.0) * 100.0))
        grouped = idle_df.groupby("speed", as_index=False).agg(
            power=("power", "median"),
            sample_count=("power", "size"),
        ).sort_values("speed")

        points = []
        for _, row in grouped.iterrows():
            points.append({
                "speed": float(row["speed"]),
                "power": float(row["power"]),
                "sample_count": int(row["sample_count"]),
                "segment_start": None,
                "segment_end": None,
                "stable_start": None,
                "stable_end": None,
                "power_std": 0.0,
                "sample_rate": None,
            })
        return points

    def _read_channel_export_series(self, file_path):
        encoding = self.detect_file_encoding(file_path)
        channel_infos = []
        sample_rate = 1000.0
        data_rows = []

        with open(file_path, 'r', encoding=encoding, errors='ignore') as infile:
            for raw_line in infile:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith('<SmplInfo>'):
                    tokens = self._parse_tagged_tokens(line)
                    if len(tokens) >= 3:
                        try:
                            sample_rate = float(tokens[2])
                        except Exception:
                            pass
                elif line.startswith('<ChannelInfo>'):
                    channel_infos.append(self._parse_tagged_tokens(line))
                elif line.startswith('<ChannelData>'):
                    raw_values = [val.strip() for val in line.split(',')[1:] if val.strip()]
                    if not raw_values:
                        continue
                    data_rows.append(raw_values)

        if not channel_infos:
            raise ValueError("未找到<ChannelInfo>通道定义")
        if not data_rows:
            raise ValueError("未找到<ChannelData>采样数据")

        expected_col_count = len(channel_infos)
        speed_col = self._find_channel_index(channel_infos, ["实际速度", "5", "SP轴"])
        power_col = self._find_channel_index(channel_infos, ["G寄存器", "0", "X轴", "432"])
        if speed_col is None or power_col is None:
            raise ValueError("未找到空载辨识所需的转速/功率通道")

        speeds = []
        powers = []
        for row in data_rows:
            if len(row) < expected_col_count:
                continue
            try:
                speeds.append(float(row[speed_col]))
                powers.append(float(row[power_col]))
            except Exception:
                continue

        if not speeds or not powers:
            raise ValueError("通道数据中没有可用的转速/功率样本")

        return {
            "sample_rate": float(sample_rate) if sample_rate > 0 else 1000.0,
            "speeds": np.asarray(speeds, dtype=float),
            "powers": np.asarray(powers, dtype=float),
            "speed_col": int(speed_col),
            "power_col": int(power_col),
            "channel_count": int(expected_col_count),
        }

    def _longest_true_block(self, mask):
        best_start = 0
        best_end = 0
        start = None
        for idx, flag in enumerate(mask):
            if flag and start is None:
                start = idx
            elif not flag and start is not None:
                if idx - start > best_end - best_start:
                    best_start, best_end = start, idx
                start = None
        if start is not None and len(mask) - start > best_end - best_start:
            best_start, best_end = start, len(mask)
        return best_start, best_end

    def _extract_stable_idle_points_from_series(self, speeds, powers, sample_rate):
        speeds = np.asarray(speeds, dtype=float)
        powers = np.asarray(powers, dtype=float)
        valid_mask = np.isfinite(speeds) & np.isfinite(powers)
        speeds = speeds[valid_mask]
        powers = powers[valid_mask]
        if len(speeds) == 0:
            return []

        rounded_speeds = np.where(np.abs(speeds) >= 50.0, np.rint(speeds / 100.0) * 100.0, 0.0)
        min_segment_points = max(500, int(round(float(sample_rate) * 1.5)))
        min_stable_points = max(200, int(round(float(sample_rate) * 0.8)))

        points = []
        start = 0
        current_speed = float(rounded_speeds[0])

        def _append_segment(seg_speed, seg_start, seg_end):
            seg_len = seg_end - seg_start
            if seg_speed <= 0 or seg_len < min_segment_points:
                return

            trim = max(int(round(seg_len * 0.15)), int(round(float(sample_rate) * 0.5)))
            max_trim = max(0, (seg_len - min_stable_points) // 2)
            trim = min(trim, max_trim)

            core_start = seg_start + trim
            core_end = seg_end - trim
            if core_end - core_start < min_stable_points:
                core_start = seg_start
                core_end = seg_end

            segment_power = powers[core_start:core_end]
            if len(segment_power) == 0:
                return

            median_power = float(np.median(segment_power))
            q1, q3 = np.quantile(segment_power, [0.25, 0.75])
            iqr = max(float(q3 - q1), 1.0)
            tolerance = max(1.5 * iqr, abs(median_power) * 0.03, 2.0)
            inlier_mask = np.abs(segment_power - median_power) <= tolerance
            stable_rel_start, stable_rel_end = self._longest_true_block(inlier_mask)

            if stable_rel_end - stable_rel_start < min_stable_points:
                stable_rel_start = 0
                stable_rel_end = len(segment_power)

            stable_slice = segment_power[stable_rel_start:stable_rel_end]
            if len(stable_slice) == 0:
                return

            stable_power = float(np.median(stable_slice))
            stable_std = float(np.std(stable_slice))
            stable_start = int(core_start + stable_rel_start)
            stable_end = int(core_start + stable_rel_end - 1)

            points.append({
                "speed": float(seg_speed),
                "power": stable_power,
                "sample_count": int(len(stable_slice)),
                "segment_start": int(seg_start),
                "segment_end": int(seg_end - 1),
                "stable_start": stable_start,
                "stable_end": stable_end,
                "power_std": stable_std,
                "sample_rate": float(sample_rate),
            })

        for idx in range(1, len(rounded_speeds)):
            speed_val = float(rounded_speeds[idx])
            if speed_val != current_speed:
                _append_segment(current_speed, start, idx)
                start = idx
                current_speed = speed_val
        _append_segment(current_speed, start, len(rounded_speeds))

        return points

    def _extract_idle_points_from_channel_export(self, file_path):
        parsed = self._read_channel_export_series(file_path)
        points = self._extract_stable_idle_points_from_series(
            parsed["speeds"], parsed["powers"], parsed["sample_rate"]
        )
        if not points:
            raise ValueError("未在阶梯转速数据中识别到稳定空载区间")
        return points

    def _build_idle_model_signature(self, file_paths):
        parts = []
        for path in file_paths:
            try:
                mtime = os.path.getmtime(path)
            except Exception:
                mtime = 0.0
            parts.append(f"{os.path.abspath(path)}|{mtime:.6f}")
        return "||".join(parts)

    def _build_idle_power_model_from_files(self, file_paths):
        if not file_paths:
            raise ValueError("未提供空载辨识文件")

        per_file_points = {}
        aggregated = collections.defaultdict(list)

        for file_path in file_paths:
            if self._looks_like_channel_export(file_path):
                points = self._extract_idle_points_from_channel_export(file_path)
            else:
                points = self._extract_idle_points_from_plain_csv(file_path)
            if not points:
                raise ValueError(f"{os.path.basename(file_path)} 未识别到有效空载点")
            per_file_points[file_path] = points
            for point in points:
                aggregated[float(point["speed"])].append({
                    "file_path": file_path,
                    "power": float(point["power"]),
                    "sample_count": int(point.get("sample_count", 0)),
                    "power_std": float(point.get("power_std", 0.0)),
                })

        inconsistent = []
        grouped_points = []
        file_count = len(file_paths)
        for speed in sorted(aggregated):
            entries = aggregated[speed]
            powers = np.asarray([entry["power"] for entry in entries], dtype=float)
            median_power = float(np.median(powers))
            max_rel_error = 0.0
            if len(powers) >= 2 and abs(median_power) > 1e-9:
                max_rel_error = max(abs(float(power) - median_power) / abs(median_power) for power in powers)
                if max_rel_error > 0.05:
                    inconsistent.append({
                        "speed": float(speed),
                        "median_power": median_power,
                        "max_rel_error": float(max_rel_error),
                        "entries": entries,
                    })
            grouped_points.append({
                "speed": float(speed),
                "power": median_power,
                "file_count": int(len(entries)),
                "total_files": int(file_count),
                "max_rel_error": float(max_rel_error),
            })

        if inconsistent:
            lines = []
            for item in inconsistent[:5]:
                detail = ", ".join(
                    f"{os.path.basename(entry['file_path'])}:{entry['power']:.3f}W"
                    for entry in item["entries"]
                )
                lines.append(
                    f"{item['speed']:.0f} rpm 偏差 {item['max_rel_error'] * 100:.2f}% ({detail})"
                )
            raise ValueError("相同转速的空载功率偏差超过5%:\n" + "\n".join(lines))

        speeds = [item["speed"] for item in grouped_points]
        powers = [item["power"] for item in grouped_points]
        if not speeds:
            raise ValueError("没有生成有效的空载功率-转速点")

        linear_coeff = None
        if len(speeds) >= 2:
            slope, intercept = np.polyfit(np.asarray(speeds, dtype=float), np.asarray(powers, dtype=float), 1)
            linear_coeff = (float(slope), float(intercept))

        return {
            "speeds": speeds,
            "powers": powers,
            "linear_coeff": linear_coeff,
            "source_path": file_paths[0] if len(file_paths) == 1 else "",
            "source_paths": list(file_paths),
            "per_file_points": per_file_points,
            "grouped_points": grouped_points,
        }

    def _strip_nc_comments(self, line):
        line = re.sub(r'\(.*?\)', ' ', line)
        if ';' in line:
            line = line.split(';', 1)[0]
        return line.strip()

    def _pick_representative_speed(self, speeds):
        if not speeds:
            return 0.0
        rounded = [round(float(speed), 6) for speed in speeds if float(speed) > 0]
        if not rounded:
            return 0.0
        counter = collections.Counter(rounded)
        last_seen = {value: idx for idx, value in enumerate(rounded)}
        return float(max(counter.items(), key=lambda item: (item[1], last_seen[item[0]]))[0])

    def _extract_nc_tool_geometry_from_text(self, text):
        diameter = None
        text = (text or "").strip()
        if not text:
            return diameter

        explicit_dia_patterns = [
            r'(?i)\bDIA(?:METER)?\s*=?\s*([-+]?\d*\.?\d+)\b',
            r'(?i)(?:^|[^A-Z])D\s*=\s*([-+]?\d*\.?\d+)\b',
            r'(?i)\b直径\s*[:=]?\s*([-+]?\d*\.?\d+)\b',
        ]
        for pattern in explicit_dia_patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            try:
                diameter = float(match.group(1))
                break
            except Exception:
                continue

        return diameter

    def _extract_nc_tool_geometry(self, raw_line):
        tool_diameter = None
        if not raw_line:
            return {
                "tool_diameter": None,
                "tool_radius": None,
            }

        candidate_texts = []
        stripped = raw_line.strip()
        if stripped:
            if re.search(r'(?i)\b(tool|dia|diameter)\b', stripped):
                candidate_texts.append(stripped)
            candidate_texts.extend(re.findall(r'\((.*?)\)', raw_line))
            if ';' in raw_line:
                candidate_texts.append(raw_line.split(';', 1)[1].strip())

        for text in candidate_texts:
            diameter = self._extract_nc_tool_geometry_from_text(text)
            if tool_diameter is None and diameter is not None:
                tool_diameter = diameter
            if tool_diameter is not None:
                break

        return {
            "tool_diameter": tool_diameter,
            "tool_radius": None,
        }

    def _apply_nc_tool_metadata(self, profile):
        if not profile:
            return

        tool_diameter = profile.get("tool_diameter")
        tool_radius = profile.get("tool_radius")

        if tool_diameter is not None:
            self.tool_diameter.set(self._format_optional_float(tool_diameter))
        if tool_radius is not None:
            self.tool_radius.set(self._format_optional_float(tool_radius))

        if tool_diameter is not None:
            self._sync_pit_metadata_to_records()
            self._persist_app_config()

    def _get_nc_tool_summary_text(self, profile):
        if not profile:
            return ""
        summary_parts = []
        tool_diameter = profile.get("tool_diameter")
        tool_radius = profile.get("tool_radius")
        if tool_diameter is not None:
            summary_parts.append(f"刀具直径 {float(tool_diameter):.3f} mm")
        if tool_radius is not None:
            summary_parts.append(f"刀具半径 {float(tool_radius):.3f} mm")
        return f"；{'，'.join(summary_parts)}" if summary_parts else ""

    def extract_nc_spindle_profile(self, file_path):
        """读取NC文件中的指令转速/进给轨迹，按行继承直到下次改动。"""
        if not file_path or not os.path.exists(file_path):
            raise ValueError("NC文件不存在")

        encoding = self.detect_file_encoding(file_path)
        command_speed = 0.0
        current_feed = 0.0
        spindle_on = False
        command_speeds = []
        line_speeds = []
        active_line_speeds = []
        line_feeds = []
        states = []
        state_by_line_index = {}
        state_by_n = {}
        tool_diameter = None

        with open(file_path, 'r', encoding=encoding, errors='ignore') as infile:
            for line_index, raw_line in enumerate(infile):
                if tool_diameter is None:
                    geometry_meta = self._extract_nc_tool_geometry(raw_line)
                    if tool_diameter is None and geometry_meta.get("tool_diameter") is not None:
                        tool_diameter = geometry_meta.get("tool_diameter")

                line = self._strip_nc_comments(raw_line)
                if not line:
                    continue
                matches = re.findall(r'(?<![A-Z])S([-+]?\d*\.?\d+)', line, flags=re.IGNORECASE)
                if matches:
                    try:
                        command_speed = float(matches[-1])
                    except Exception:
                        pass

                feed_matches = re.findall(r'(?<![A-Z])F([-+]?\d*\.?\d+)', line, flags=re.IGNORECASE)
                if feed_matches:
                    try:
                        current_feed = float(feed_matches[-1])
                    except Exception:
                        pass

                if re.search(r'(?<![A-Z])M0?5(?!\d)', line, flags=re.IGNORECASE):
                    spindle_on = False
                elif re.search(r'(?<![A-Z])M0?[34](?!\d)', line, flags=re.IGNORECASE):
                    spindle_on = True

                n_match = re.search(r'(?<![A-Z])N(\d+)', line, flags=re.IGNORECASE)
                n_value = int(n_match.group(1)) if n_match else None
                active_speed = command_speed if spindle_on and command_speed > 0 else 0.0

                state = {
                    "file_line_index": int(line_index),
                    "n_value": n_value,
                    "line_text": line,
                    "command_speed": float(command_speed),
                    "active_speed": float(active_speed),
                    "feed": float(current_feed),
                    "spindle_on": bool(spindle_on),
                }
                states.append(state)
                state_by_line_index[int(line_index)] = state
                if n_value is not None:
                    state_by_n[int(n_value)] = state

                if command_speed > 0:
                    command_speeds.append(float(command_speed))
                    line_speeds.append(float(command_speed))
                if active_speed > 0:
                    active_line_speeds.append(float(active_speed))
                if current_feed > 0:
                    line_feeds.append(float(current_feed))

        unique_speeds = sorted({round(speed, 6) for speed in line_speeds if speed > 0})
        unique_feeds = sorted({round(feed, 6) for feed in line_feeds if feed > 0})
        dominant_speed = self._pick_representative_speed(active_line_speeds or line_speeds)

        segments = []
        segment_start = None
        segment_state = None
        prev_state = None
        for state in states:
            state_key = (
                round(float(state["command_speed"]), 6),
                round(float(state["feed"]), 6),
                bool(state["spindle_on"]),
            )
            if segment_state is None:
                segment_state = state_key
                segment_start = state
                prev_state = state
                continue
            if state_key == segment_state:
                prev_state = state
                continue
            segments.append({
                "start_line_index": int(segment_start["file_line_index"]),
                "end_line_index": int(prev_state["file_line_index"]),
                "start_n": segment_start["n_value"],
                "end_n": prev_state["n_value"],
                "command_speed": float(segment_start["command_speed"]),
                "active_speed": float(segment_start["active_speed"]),
                "feed": float(segment_start["feed"]),
                "spindle_on": bool(segment_start["spindle_on"]),
            })
            segment_state = state_key
            segment_start = state
            prev_state = state
        if segment_start is not None and states:
            last_state = states[-1]
            segments.append({
                "start_line_index": int(segment_start["file_line_index"]),
                "end_line_index": int(last_state["file_line_index"]),
                "start_n": segment_start["n_value"],
                "end_n": last_state["n_value"],
                "command_speed": float(segment_start["command_speed"]),
                "active_speed": float(segment_start["active_speed"]),
                "feed": float(segment_start["feed"]),
                "spindle_on": bool(segment_start["spindle_on"]),
            })

        return {
            "line_speeds": line_speeds,
            "command_speeds": command_speeds,
            "active_line_speeds": active_line_speeds,
            "line_feeds": line_feeds,
            "unique_speeds": unique_speeds,
            "unique_feeds": unique_feeds,
            "dominant_speed": dominant_speed,
            "states": states,
            "state_by_line_index": state_by_line_index,
            "state_by_n": state_by_n,
            "segments": segments,
            "tool_diameter": tool_diameter,
            "tool_radius": (float(tool_diameter) / 2.0) if tool_diameter is not None else None,
        }

    def _resolve_nc_state_for_process_row(self, raw_line_number, gcode_content):
        if not self.gcode_profile:
            return None

        n_value = self.extract_n_value(gcode_content)
        n_int = self.extract_n_integer(n_value)
        if n_int is not None:
            state = self.gcode_profile.get("state_by_n", {}).get(int(n_int))
            if state:
                return state

        if raw_line_number is not None:
            state = self.gcode_profile.get("state_by_line_index", {}).get(int(raw_line_number))
            if state:
                return state
        return None

    def _get_program_idle_power_rows(self):
        if not self.gcode_profile:
            return []

        unique_speeds = [
            float(speed) for speed in self.gcode_profile.get("unique_speeds", [])
            if float(speed) > 0
        ]
        if not unique_speeds:
            return []

        segment_counter = collections.Counter()
        for segment in self.gcode_profile.get("segments", []):
            if not segment.get("spindle_on"):
                continue
            speed = float(segment.get("active_speed") or segment.get("command_speed") or 0.0)
            if speed > 0:
                segment_counter[round(speed, 6)] += 1

        rows = []
        for speed in sorted(unique_speeds):
            rows.append({
                "speed": float(speed),
                "idle_power": float(self.predict_idle_power(speed)),
                "segment_count": int(segment_counter.get(round(float(speed), 6), 0)),
            })
        return rows

    def _update_program_idle_detail_button_state(self):
        if not hasattr(self, "program_idle_detail_btn"):
            return
        has_details = bool(self.gcode_profile and self.idle_power_model and self._get_program_idle_power_rows())
        if has_details:
            self.program_idle_detail_btn.state(["!disabled"])
        else:
            self.program_idle_detail_btn.state(["disabled"])

    def _update_program_idle_summary(self):
        rows = self._get_program_idle_power_rows()

        if not self.gcode_profile:
            idle_power = float(self.current_program_idle_power.get() or self.p_idle_var.get() or 0.0)
            if idle_power > 0:
                self.current_program_idle_power_display.set(f"{idle_power:.3f} W")
            else:
                self.current_program_idle_power_display.set("未计算")
            self._update_program_idle_detail_button_state()
            return

        if not rows:
            self.current_program_idle_power_display.set("未识别到S指令")
            self._update_program_idle_detail_button_state()
            return

        if not self.idle_power_model:
            if len(rows) == 1:
                self.current_program_idle_power_display.set("已识别1档转速，待空载辨识")
            else:
                self.current_program_idle_power_display.set(f"已识别{len(rows)}档转速，待空载辨识")
            self._update_program_idle_detail_button_state()
            return

        if len(rows) == 1:
            self.current_program_idle_power_display.set(f"{rows[0]['idle_power']:.3f} W")
        else:
            powers = [row["idle_power"] for row in rows]
            self.current_program_idle_power_display.set(
                f"{len(rows)}档转速，空载 {min(powers):.1f}~{max(powers):.1f} W"
            )
        self._update_program_idle_detail_button_state()

    def show_program_idle_detail_dialog(self):
        rows = self._get_program_idle_power_rows()
        if not self.gcode_profile:
            messagebox.showwarning("提示", "请先导入G代码NC文件")
            return
        if not self.idle_power_model:
            messagebox.showwarning("提示", "请先完成空载功率辨识")
            return
        if not rows:
            messagebox.showwarning("提示", "当前G代码中没有可用的主轴转速")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("程序空载功率明细")
        dialog.geometry("520x420")
        dialog.transient(self.root)
        dialog.grab_set()

        main_frame = ttk.Frame(dialog, padding=12)
        main_frame.pack(fill=tk.BOTH, expand=True)

        gcode_name = os.path.basename(self.gcode_nc_path_var.get().strip()) if self.gcode_nc_path_var.get().strip() else "未命名NC"
        ttk.Label(
            main_frame,
            text=f"G代码: {gcode_name}",
            font=UI_FONT_BOLD,
            foreground=UI_COLOR_PRIMARY_DARK
        ).pack(anchor="w", pady=(0, 4))
        ttk.Label(
            main_frame,
            text=f"共识别 {len(rows)} 档转速，对应空载功率如下",
            font=UI_FONT_NORMAL
        ).pack(anchor="w", pady=(0, 8))

        columns = ("speed", "idle_power", "segment_count")
        tree = ttk.Treeview(main_frame, columns=columns, show="headings", height=min(max(len(rows), 6), 14))
        tree.heading("speed", text="转速 (rpm)")
        tree.heading("idle_power", text="空载功率 (W)")
        tree.heading("segment_count", text="命中段数")
        tree.column("speed", width=140, anchor="center")
        tree.column("idle_power", width=160, anchor="center")
        tree.column("segment_count", width=100, anchor="center")

        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        for row in rows:
            tree.insert(
                "",
                "end",
                values=(
                    f"{row['speed']:.1f}",
                    f"{row['idle_power']:.3f}",
                    int(row["segment_count"]),
                ),
            )

        btn_frame = ttk.Frame(dialog, padding=(0, 0, 0, 10))
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="关闭", command=dialog.destroy, width=10).pack()
        center_dialog_on_parent(dialog, self.root)

    def predict_idle_power(self, spindle_speed):
        """根据空载功率模型估算指定转速下的空载功率。"""
        try:
            speed = float(spindle_speed)
        except Exception:
            speed = 0.0

        if speed <= 0:
            return float(self.p_idle_var.get())

        model = self.idle_power_model
        if not model or not model.get("speeds") or not model.get("powers"):
            return float(self.p_idle_var.get())

        speeds = np.asarray(model["speeds"], dtype=float)
        powers = np.asarray(model["powers"], dtype=float)
        if len(speeds) == 1:
            return float(powers[0])

        if speed <= speeds[0]:
            x0, x1 = speeds[0], speeds[1]
            y0, y1 = powers[0], powers[1]
        elif speed >= speeds[-1]:
            x0, x1 = speeds[-2], speeds[-1]
            y0, y1 = powers[-2], powers[-1]
        else:
            idx = int(np.searchsorted(speeds, speed))
            x0, x1 = speeds[idx - 1], speeds[idx]
            y0, y1 = powers[idx - 1], powers[idx]

        if abs(x1 - x0) < 1e-9:
            return float(y0)
        ratio = (speed - x0) / (x1 - x0)
        return float(y0 + ratio * (y1 - y0))

    def _refresh_current_program_idle_power_from_gcode(self):
        """根据已导入NC文件刷新当前程序主转速与空载功率。"""
        gcode_path = self.gcode_nc_path_var.get().strip()
        if not gcode_path:
            self.gcode_profile = None
            self._update_program_idle_summary()
            return False

        profile = self.extract_nc_spindle_profile(gcode_path)
        self.gcode_profile = profile
        self._apply_nc_tool_metadata(profile)
        tool_summary = self._get_nc_tool_summary_text(profile)
        dominant_speed = float(profile.get("dominant_speed") or 0.0)
        if dominant_speed <= 0:
            self.current_program_speed.set(0.0)
            self.current_program_idle_power.set(float(self.p_idle_var.get()))
            self.gcode_status_var.set(
                f"NC已导入: {os.path.basename(gcode_path)}；未识别到有效S指令{tool_summary}"
            )
            self._update_program_idle_summary()
            return False

        self.current_program_speed.set(dominant_speed)
        idle_power = self.predict_idle_power(dominant_speed)
        self.current_program_idle_power.set(idle_power)
        self.p_idle_var.set(idle_power)
        self.gcode_status_var.set(
            f"NC已导入: {os.path.basename(gcode_path)}；识别到 {len(profile.get('unique_speeds', []))} 个转速、"
            f"{len(profile.get('unique_feeds', []))} 个进给，参考转速 {dominant_speed:.1f} rpm{tool_summary}"
        )
        self._update_program_idle_summary()
        return True

    def browse_nc_file(self):
        """导入G代码NC文件，用于识别当前程序主轴转速。"""
        file_path = filedialog.askopenfilename(
            title="选择G代码NC文件",
            filetypes=(("G代码文件", "*.nc;*.cnc;*.gcode;*.txt"), ("所有文件", "*.*"))
        )
        if not file_path:
            return

        try:
            self.gcode_nc_path_var.set(file_path)
            if self.idle_power_model:
                self._refresh_current_program_idle_power_from_gcode()
            else:
                profile = self.extract_nc_spindle_profile(file_path)
                self.gcode_profile = profile
                self._apply_nc_tool_metadata(profile)
                tool_summary = self._get_nc_tool_summary_text(profile)
                dominant_speed = float(profile.get("dominant_speed") or 0.0)
                self.current_program_speed.set(dominant_speed)
                if dominant_speed > 0:
                    self.gcode_status_var.set(
                        f"NC已导入: {os.path.basename(file_path)}；识别到 {len(profile.get('unique_speeds', []))} 个转速、"
                        f"{len(profile.get('unique_feeds', []))} 个进给{tool_summary}"
                    )
                else:
                    self.gcode_status_var.set(f"NC已导入: {os.path.basename(file_path)}；未识别到有效S指令{tool_summary}")
                self._update_program_idle_summary()

            self.set_status("G代码NC文件已导入", 3000)
            if self.get_primary_input_file():
                self._process_current_input_for_preview()
        except Exception as e:
            messagebox.showerror("导入失败", f"读取G代码NC文件时发生错误:\n{str(e)}")

    def identify_no_load_power(self):
        """导入空载原始文件，建立空载功率-转速关系。"""
        file_paths = list(filedialog.askopenfilenames(
            title="选择空载辨识文件",
            filetypes=(
                ("空载原始文件", "*.csv;*.txt;*.dat;*.fxt"),
                ("CSV文件", "*.csv"),
                ("文本文件", "*.txt;*.dat;*.fxt"),
                ("所有文件", "*.*"),
            )
        ))
        if not file_paths:
            return

        try:
            model = self._build_idle_power_model_from_files(file_paths)
            speeds = [float(val) for val in model["speeds"]]
            powers = [float(val) for val in model["powers"]]

            self.no_load_csv_path_var.set(" | ".join(file_paths))
            self.idle_power_model = model
            self.idle_model_signature = self._build_idle_model_signature(file_paths) + f"|{len(speeds)}"

            if self.gcode_nc_path_var.get().strip():
                self._refresh_current_program_idle_power_from_gcode()
            else:
                fallback_speed = float(self.current_program_speed.get() or self.s_base.get())
                fallback_idle = self.predict_idle_power(fallback_speed)
                self.current_program_idle_power.set(fallback_idle)
                self.p_idle_var.set(fallback_idle)
                self._update_program_idle_summary()

            speed_min = min(speeds)
            speed_max = max(speeds)
            validated_count = sum(1 for item in model.get("grouped_points", []) if int(item.get("file_count", 0)) >= 2)
            self.no_load_status_var.set(
                f"空载功率模型已辨识: {len(speeds)} 点，{len(file_paths)} 个文件，"
                f"5%一致性通过 {validated_count} 档，转速范围 {speed_min:.1f}~{speed_max:.1f} rpm"
            )
            self.set_status("空载功率辨识完成", 3000)

            if self.get_primary_input_file():
                self._process_current_input_for_preview()
        except Exception as e:
            messagebox.showerror("辨识失败", f"空载功率辨识失败:\n{str(e)}")

    def _resolve_step_feed_geometry(self, df, ap_col=None, ae_col=None):
        ap_val = None
        ae_val = None

        if ap_col:
            ap_series = pd.to_numeric(df[ap_col], errors='coerce').dropna()
            if not ap_series.empty:
                ap_val = float(ap_series.median())

        if ae_col:
            ae_series = pd.to_numeric(df[ae_col], errors='coerce').dropna()
            if not ae_series.empty:
                ae_val = float(ae_series.median())

        if ap_val is None or ae_val is None:
            for row in self.data:
                try:
                    row_ap = float(row.get('ap', 0.0))
                    row_ae = float(row.get('ae', 0.0))
                except Exception:
                    continue
                if row_ap > 0 and row_ae > 0:
                    ap_val = row_ap if ap_val is None else ap_val
                    ae_val = row_ae if ae_val is None else ae_val
                    break

        if ap_val is None or ae_val is None:
            raise ValueError("阶梯进给CSV缺少ap/ae信息，且当前未加载可推断几何参数的工艺信息表")

        return ap_val, ae_val

    def identify_step_feed_parameters(self):
        """导入阶梯进给CSV，辨识K_c与K_e。"""
        file_path = filedialog.askopenfilename(
            title="选择阶梯进给CSV",
            filetypes=(("CSV文件", "*.csv"), ("文本文件", "*.txt"), ("所有文件", "*.*"))
        )
        if not file_path:
            return

        try:
            df = self._read_csv_flex(file_path)
            feed_col = self._find_matching_column(
                df, ["feed_rate", "feed", "进给速度", "进给率", "进给"], fallback_index=0
            )
            power_col = self._find_matching_column(
                df, ["power", "主轴功率", "功率", "load_power"], fallback_index=1
            )
            ap_col = self._find_matching_column(df, ["ap", "a_p", "切深", "轴向切深"])
            ae_col = self._find_matching_column(df, ["ae", "a_e", "切宽", "径向切宽"])
            speed_col = self._find_matching_column(df, ["spindle_speed", "speed", "rpm", "主轴转速", "转速"])

            if not feed_col or not power_col:
                raise ValueError("未识别到进给列或功率列，请检查CSV表头")

            step_df = df[[feed_col, power_col]].copy()
            step_df.columns = ["feed", "power"]
            step_df["feed"] = pd.to_numeric(step_df["feed"], errors='coerce')
            step_df["power"] = pd.to_numeric(step_df["power"], errors='coerce')
            step_df = step_df.dropna()
            if len(step_df) < 2:
                raise ValueError("有效阶梯进给样本不足，至少需要2个数据点")

            ap_val, ae_val = self._resolve_step_feed_geometry(df, ap_col, ae_col)
            if ap_val <= 0 or ae_val <= 0:
                raise ValueError("阶梯进给识别所需的ap/ae必须大于0")

            x = step_df["feed"].to_numpy(dtype=float).reshape(-1, 1)
            y = step_df["power"].to_numpy(dtype=float)

            slope = 0.0
            intercept = 0.0
            try:
                sklearn_mod = _get_sklearn()
                model = sklearn_mod.HuberRegressor()
                model.fit(x, y)
                slope = float(model.coef_[0])
                intercept = float(model.intercept_)
            except Exception:
                slope, intercept = np.polyfit(step_df["feed"].to_numpy(dtype=float), y, 1)
                slope = float(slope)
                intercept = float(intercept)

            y_pred = intercept + slope * x[:, 0]
            residuals = y - y_pred
            sample_count = len(step_df)
            sxx = float(np.sum((x[:, 0] - np.mean(x[:, 0])) ** 2))
            if sample_count > 2 and sxx > 1e-12:
                residual_var = float(np.sum(residuals ** 2) / max(sample_count - 2, 1))
                slope_stderr = math.sqrt(max(residual_var, 0.0) / sxx)
            else:
                slope_stderr = 0.0

            idle_reference = float(self.current_program_idle_power.get() or self.p_idle_var.get())
            if speed_col and self.idle_power_model:
                speed_series = pd.to_numeric(df[speed_col], errors='coerce').dropna()
                if not speed_series.empty:
                    idle_reference = float(np.median([self.predict_idle_power(val) for val in speed_series]))

            kc_value = slope * 60.0 / (ap_val * ae_val)
            kc_sigma = slope_stderr * 60.0 / (ap_val * ae_val)
            ke_value = (intercept - idle_reference) / ap_val
            kc_ucb = kc_value + float(self.kc_beta.get()) * kc_sigma

            self.step_feed_csv_path_var.set(file_path)
            self.kc_coeff.set(kc_value)
            self.kc_sigma.set(kc_sigma)
            self.ke_coeff.set(ke_value)
            try:
                model_mtime = os.path.getmtime(file_path)
            except Exception:
                model_mtime = 0.0
            self.step_feed_model_signature = (
                f"{os.path.abspath(file_path)}|{model_mtime:.6f}|{kc_value:.6f}|{ke_value:.6f}|{kc_sigma:.6f}"
            )

            self.step_feed_status_var.set(
                f"模型参数已辨识: K_c={kc_value:.6f}, K_e={ke_value:.6f}, σ_Kc={kc_sigma:.6f}, K_c^UCB={kc_ucb:.6f}"
            )
            self.set_status("阶梯进给模型参数辨识完成", 3000)

            if self.get_primary_input_file():
                self._process_current_input_for_preview()
        except Exception as e:
            messagebox.showerror("辨识失败", f"模型参数辨识失败:\n{str(e)}")

    def refresh_pit_button_state(self):
        if not hasattr(self, 'pit_display_btn'):
            return
        state = "normal" if self.pit_records else "disabled"
        self.pit_display_btn.configure(state=state)

    def _format_optional_float(self, value):
        if value is None:
            return ""
        try:
            text = f"{float(value):.6f}"
        except Exception:
            return ""
        text = text.rstrip("0").rstrip(".")
        return text if text else "0"

    def _get_optional_float_value(self, value):
        raw = str(value).strip()
        if not raw:
            return None
        try:
            numeric = float(raw)
        except Exception:
            return None
        return numeric if numeric >= 0 else None

    def get_current_pit_metadata(self):
        return {
            "tool_diameter": self._get_optional_float_value(self.tool_diameter.get()),
            "tool_radius": self._get_optional_float_value(self.tool_radius.get()),
            "tool_material": str(self.workpiece_material.get()).strip(),
            "blank_material": str(self.blank_material.get()).strip(),
        }

    def _sync_pit_metadata_to_records(self):
        if not self.pred_power_intervals and not self.pit_records:
            return
        metadata = self.get_current_pit_metadata()
        for records in (self.pred_power_intervals, self.pit_records):
            for record in records:
                if isinstance(record, dict):
                    record.update(metadata)

    def on_pit_metadata_commit(self, event=None):
        invalid_fields = []
        diameter_value = None
        radius_raw = str(self.tool_radius.get()).strip()
        for label, var in (
            ("刀具直径", self.tool_diameter),
            ("刀具半径", self.tool_radius),
        ):
            raw = str(var.get()).strip()
            if not raw:
                var.set("")
                continue
            value = self._get_optional_float_value(raw)
            if value is None:
                invalid_fields.append(label)
                continue
            var.set(self._format_optional_float(value))
            if label == "刀具直径":
                diameter_value = value

        if diameter_value is None:
            diameter_value = self._get_optional_float_value(self.tool_diameter.get())
        if diameter_value is not None and not radius_raw:
            self.tool_radius.set(self._format_optional_float(diameter_value / 2.0))

        self.workpiece_material.set(str(self.workpiece_material.get()).strip())
        self.blank_material.set(str(self.blank_material.get()).strip())
        self._sync_pit_metadata_to_records()
        self._persist_app_config()
        if invalid_fields:
            self.set_status(f"{'、'.join(invalid_fields)}输入无效，已按空值处理", 4000)

    def show_pit_dialog(self):
        """显示完整PIT表。"""
        if not self.pit_records:
            messagebox.showwarning("无PIT", "请先导入工艺信息文件并生成稳态区间")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("完整PIT")
        dialog.geometry("1560x560")
        dialog.minsize(1260, 420)
        dialog.transient(self.root)
        dialog.grab_set()
        center_dialog_on_parent(dialog, self.root)

        columns = [
            ("zone_id", "Zone_ID", 90),
            ("start_line", "StartLine", 90),
            ("end_line", "EndLine", 90),
            ("start_s", "Start_s", 90),
            ("end_s", "End_s", 90),
            ("tool_diameter", "ToolDia", 90),
            ("tool_radius", "ToolR", 90),
            ("tool_material", "ToolMaterial", 150),
            ("blank_material", "BlankMaterial", 150),
            ("a_p", "a_p", 80),
            ("a_e", "a_e", 80),
            ("F_plan", "F_plan", 90),
            ("p_idle", "P_idle", 90),
            ("p_pred", "P_pred", 90),
            ("K_c_hat", "K_c_hat", 100),
            ("sigma_Kc", "sigma_Kc", 100),
            ("K_c_UCB", "K_c_UCB", 100),
            ("sample_count", "SampleCount", 100),
        ]

        outer = ttk.Frame(dialog, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(1, weight=1)

        meta_frame = ttk.LabelFrame(outer, text="PIT元数据", padding=8)
        meta_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        meta_frame.grid_columnconfigure(5, weight=1)
        meta_frame.grid_columnconfigure(7, weight=1)

        ttk.Label(meta_frame, text="刀具直径(mm):", font=UI_FONT_NORMAL).grid(row=0, column=0, sticky="w", padx=(0, 4))
        tool_diameter_entry = ttk.Entry(meta_frame, textvariable=self.tool_diameter, width=12, font=UI_FONT_NORMAL)
        tool_diameter_entry.grid(row=0, column=1, sticky="w", padx=(0, 10))

        ttk.Label(meta_frame, text="刀具半径(mm):", font=UI_FONT_NORMAL).grid(row=0, column=2, sticky="w", padx=(0, 4))
        tool_radius_entry = ttk.Entry(meta_frame, textvariable=self.tool_radius, width=12, font=UI_FONT_NORMAL)
        tool_radius_entry.grid(row=0, column=3, sticky="w", padx=(0, 10))

        ttk.Label(meta_frame, text="刀具材料:", font=UI_FONT_NORMAL).grid(row=0, column=4, sticky="w", padx=(0, 4))
        tool_material_entry = ttk.Entry(meta_frame, textvariable=self.workpiece_material, width=18, font=UI_FONT_NORMAL)
        tool_material_entry.grid(row=0, column=5, sticky="ew", padx=(0, 10))

        ttk.Label(meta_frame, text="毛坯材料:", font=UI_FONT_NORMAL).grid(row=0, column=6, sticky="w", padx=(0, 4))
        blank_material_entry = ttk.Entry(meta_frame, textvariable=self.blank_material, width=18, font=UI_FONT_NORMAL)
        blank_material_entry.grid(row=0, column=7, sticky="ew", padx=(0, 10))

        meta_hint_var = tk.StringVar(value="可手动填写或修改；导入NC时自动读取显式直径字段，并按直径一半回填刀具半径。")
        ttk.Label(
            meta_frame,
            textvariable=meta_hint_var,
            font=UI_FONT_SMALL,
            foreground=UI_COLOR_TEXT_MUTED
        ).grid(row=1, column=0, columnspan=9, sticky="w", pady=(6, 0))

        container = ttk.Frame(outer)
        container.grid(row=1, column=0, sticky="nsew")

        tree = ttk.Treeview(container, columns=[col[0] for col in columns], show="headings")
        vsb = ttk.Scrollbar(container, orient=tk.VERTICAL, command=tree.yview)
        hsb = ttk.Scrollbar(container, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        for key, heading, width in columns:
            tree.heading(key, text=heading)
            tree.column(key, width=width, anchor="center", stretch=True)

        def _fmt(value):
            if value is None:
                return ""
            if isinstance(value, float):
                return f"{value:.6f}" if abs(value) < 1000 else f"{value:.3f}"
            return value

        def _reload_tree():
            tree.delete(*tree.get_children())
            for entry in self.pit_records:
                values = [_fmt(entry.get(key)) for key, _, _ in columns]
                tree.insert("", "end", values=values)

        def _commit_and_refresh(event=None):
            self.on_pit_metadata_commit(event)
            _reload_tree()
            return None

        ttk.Button(meta_frame, text="应用到当前PIT", command=_commit_and_refresh, width=14).grid(row=0, column=8, sticky="e")

        for widget in (
            tool_diameter_entry,
            tool_radius_entry,
            tool_material_entry,
            blank_material_entry,
        ):
            widget.bind("<Return>", _commit_and_refresh)
            widget.bind("<FocusOut>", _commit_and_refresh)

        _reload_tree()

        ttk.Button(outer, text="关闭", command=dialog.destroy, width=10).grid(row=2, column=0, pady=(8, 0))

    def on_model_param_commit(self, event=None):
        """手动修改P_idle/K_c/K_e后刷新当前预览。"""
        try:
            idle_power = float(self.p_idle_var.get())
        except Exception:
            idle_power = 0.0
            self.p_idle_var.set(idle_power)

        self.current_program_idle_power.set(idle_power)
        self._update_program_idle_summary()

        if self.get_primary_input_file():
            self._process_current_input_for_preview()

    def _split_numeric_and_gcode_tokens(self, tokens):
        """拆分工艺信息行：数值列与G代码列"""
        gcode_start_idx = None
        for idx, token in enumerate(tokens):
            if re.match(r'^[A-Za-z]', token):
                gcode_start_idx = idx
                break
        if gcode_start_idx is None or gcode_start_idx == 0:
            return None, None
        return tokens[:gcode_start_idx], tokens[gcode_start_idx:]
    
    def parse_gcode_line(self, line):
        """解析G代码行"""
        tokens = line.strip().split()
        if len(tokens) < 6:
            return None
        
        # 列格式: [序号], 行号, [可选MRR], ap, ae, F, G代码...
        # MRR不参与预测计算，仅使用ap/ae/F进行推导
        numeric_tokens, gcode_tokens = self._split_numeric_and_gcode_tokens(tokens)
        if not numeric_tokens or not gcode_tokens:
            return None
        
        if len(numeric_tokens) < 3:
            return None
        
        ap = numeric_tokens[-3]
        ae = numeric_tokens[-2]
        feed_rate = numeric_tokens[-1]
        try:
            float(ap)
            float(ae)
            float(feed_rate)
        except Exception:
            return None

        line_number = None
        if len(numeric_tokens) >= 2:
            try:
                # 需求2：工艺信息文件行号减一处理（对齐SampleData行号从0开始）
                line_number = int(float(numeric_tokens[1])) - 1
            except Exception:
                line_number = None
        
        # 合并剩余的字段作为G代码内容
        gcode_content = ' '.join(gcode_tokens)
        
        # 提取转速S值（保留小数部分）
        s_value = None
        s_match = re.search(r'S(\d+\.?\d*)', gcode_content)
        if s_match:
            try:
                s_value = float(s_match.group(1))
            except ValueError:
                s_value = None
        
        return ap, ae, feed_rate, gcode_content, s_value, line_number
    
    def extract_coordinates(self, gcode_content, prev_coords):
        """提取坐标值"""
        # 默认使用上一行的坐标值
        x, y, z = prev_coords
        
        # 使用正则表达式提取坐标值
        if match := re.search(r'X([-+]?\d*\.?\d+)', gcode_content):
            x = float(match.group(1))
        if match := re.search(r'Y([-+]?\d*\.?\d+)', gcode_content):
            y = float(match.group(1))
        if match := re.search(r'Z([-+]?\d*\.?\d+)', gcode_content):
            z = float(match.group(1))
        
        return x, y, z
    
    def calculate_distance(self, prev_coords, current_coords):
        """计算距离"""
        if prev_coords is None:  # 第一行没有前一行
            return 0.0
        
        dx = current_coords[0] - prev_coords[0]
        dy = current_coords[1] - prev_coords[1]
        dz = current_coords[2] - prev_coords[2]
        
        return math.sqrt(dx**2 + dy**2 + dz**2)
    
    def extract_n_value(self, gcode_content):
        """提取N值（行号标识），保留小数部分
        当G代码中没有N前缀时返回None（而非"N0"），
        避免无N值的行被错误合并到同一对齐行号组中。
        """
        # 修改正则表达式以匹配带小数的N值
        match = re.search(r'^N\d+\.?\d*', gcode_content)
        return match.group(0) if match else None

    def extract_n_decimal_part(self, n_value):
        """提取N值的小数部分（字符串）"""
        if not n_value:
            return ""
        match = re.match(r'^N\d+(?:\.(\d+))?', n_value)
        return match.group(1) if match and match.group(1) else ""

    def extract_n_integer(self, n_value):
        """提取N值整数部分"""
        if n_value is None:
            return None
        match = re.search(r'^N(\d+)', n_value)
        if not match:
            return None
        try:
            return int(match.group(1))
        except Exception:
            return None

    def count_n_integer_occurrences(self, input_file):
        """统计输入文件中各N整数出现次数"""
        counts = {}
        if not input_file:
            return counts
        try:
            with open(input_file, 'r') as infile:
                for line in infile:
                    parsed = self.parse_gcode_line(line)
                    if not parsed:
                        continue
                    _, _, _, gcode_content, _, _ = parsed
                    n_value = self.extract_n_value(gcode_content)
                    n_int = self.extract_n_integer(n_value)
                    if n_int is None:
                        continue
                    counts[n_int] = counts.get(n_int, 0) + 1
        except Exception:
            return counts
        return counts

    def _set_widget_state(self, widget, enabled):
        state = "normal" if enabled else "disabled"
        try:
            widget.configure(state=state)
            return
        except Exception:
            pass
        try:
            if enabled:
                widget.state(["!disabled"])
            else:
                widget.state(["disabled"])
        except Exception:
            pass

    def set_sample_controls_enabled(self, enabled, refresh=True):
        """按导入工艺信息表数量切换实测数据联动控件状态"""
        if not hasattr(self, "sample_control_widgets"):
            return
        for widget in self.sample_control_widgets:
            self._set_widget_state(widget, enabled)
        if enabled:
            if refresh:
                self.on_sample_display_mode_change()
            else:
                mode = self.sample_display_mode.get()
                if mode == "program":
                    self.sample_program_combo.configure(state="readonly")
                    self.sample_tool_combo.configure(state="disabled")
                    self.sample_avg_var.set("-")
                    self.sample_ideal_var.set("-")
                else:
                    self.sample_program_combo.configure(state="readonly")
                    self.sample_tool_combo.configure(state="readonly")
                    self.sample_avg_var.set("-")
                    self.sample_ideal_var.set("-")
        else:
            self.sample_avg_var.set("多文件")
            self.sample_ideal_var.set("多文件")
            if hasattr(self, "sample_auto_status_var"):
                self.sample_auto_status_var.set("多文件模式：停用实测自动导入")

    def reset_sample_data_state(self):
        """清空已加载的实测数据状态"""
        self.sample_data_loaded = False
        self.sample_data_dir = None
        self.sample_programs = {}
        self.sample_data_values = None
        self.sample_data_values_raw = None
        self.sample_data_line_numbers = None
        self.sample_data_line_numbers_raw = None
        self.sample_data_program_numbers = None
        self.sample_data_program_numbers_raw = None
        self.sample_data_x_positions = None
        self.sample_data_point_indices = None
        self.sample_data_time_indices = None
        self.sample_data_base_blocks = []
        self.sample_data_valid_mask = None
        self.sample_data_valid_blocks = []
        self.process_valid_mask = None
        self.process_valid_blocks = []
        if hasattr(self, "sample_program_name"):
            self.sample_program_name.set("")
        if hasattr(self, "sample_tool_name"):
            self.sample_tool_name.set("")
        if hasattr(self, "sample_program_combo"):
            self.sample_program_combo["values"] = []
        if hasattr(self, "sample_tool_combo"):
            self.sample_tool_combo["values"] = []
        if hasattr(self, "sample_auto_status_var"):
            self.sample_auto_status_var.set("未导入实测数据")

    def ensure_sample_data_matches_inputs(self, file_paths):
        """导入工艺信息表变更时，多目录时重置实测数据"""
        if not self.sample_data_loaded:
            return
        if not file_paths:
            return
        base_dirs = {os.path.normcase(os.path.normpath(os.path.abspath(os.path.dirname(path))))
                     for path in file_paths if path}
        if len(base_dirs) != 1:
            self.reset_sample_data_state()
            return

    def reset_processing_state(self):
        """清理工艺信息表处理与预览状态"""
        self.data = []
        self.pred_power_intervals = []
        self.pit_records = []
        self.processed_file_path = ""
        self.processed_data_dir = None
        self.raw_to_aligned_line_map = {}
        self.process_valid_mask = None
        self.process_valid_blocks = []
        self.figures = []
        self.figure_names = []
        self.current_figure_index = 0
        if self.sample_display_mode.get() == "program":
            self.sample_avg_var.set("-")
            self.sample_ideal_var.set("-")
        else:
            self.sample_avg_var.set("-")
            self.sample_ideal_var.set("-")
        try:
            self.update_nav_buttons()
        except Exception:
            pass
        self.refresh_pit_button_state()

    def build_raw_to_aligned_line_map(self):
        """构建原始行号到重构行号的映射
        
        key: 工艺信息文件的原始第二列值（line_no_raw，已保持原样不做-1处理）
        value: 按N值整数合并后的重构行号（line_no_aligned）
        用于将SampleData的行号映射到预测负载的重构行号
        """
        mapping = {}
        for item in self.data or []:
            raw = item.get('line_no_raw')
            aligned = item.get('line_no_aligned')
            if raw is None or aligned is None:
                continue
            try:
                # 原始第二列值直接作为key（不再需要+1）
                mapping[int(raw)] = int(aligned)
            except Exception:
                continue
        self.raw_to_aligned_line_map = mapping
        return mapping

    def align_line_numbers_to_processed(self, line_numbers):
        """将行号转换为对齐行号"""
        if line_numbers is None:
            return None
        mapping = self.raw_to_aligned_line_map or {}
        aligned = []
        for ln in line_numbers:
            try:
                ln_int = int(ln)
            except Exception:
                aligned.append(ln)
                continue
            aligned.append(mapping.get(ln_int, ln_int))
        return np.asarray(aligned, dtype=int)

    def align_sample_program_ranges(self):
        """将SampleData.txt中的区间按对齐行号更新"""
        if not self.sample_programs:
            return
        mapping = self.raw_to_aligned_line_map or {}
        if not mapping:
            return
        raw_keys = np.asarray(sorted(mapping.keys()), dtype=int)
        aligned_vals = np.asarray([mapping[k] for k in raw_keys], dtype=int)
        for program_info in self.sample_programs.values():
            raw_ranges = program_info.get("tool_raw_ranges") or program_info.get("tools", {})
            aligned_tools = {}
            for tool_id, ranges in raw_ranges.items():
                aligned_ranges = []
                for start, end in ranges:
                    try:
                        start_int = int(start)
                        end_int = int(end)
                    except Exception:
                        continue
                    if start_int > end_int:
                        start_int, end_int = end_int, start_int
                    mask = (raw_keys >= start_int) & (raw_keys <= end_int)
                    if not mask.any():
                        continue
                    aligned_start = int(np.min(aligned_vals[mask]))
                    aligned_end = int(np.max(aligned_vals[mask]))
                    aligned_ranges.append((aligned_start, aligned_end))
                aligned_tools[tool_id] = self.merge_intervals(aligned_ranges)
            program_info["tools"] = aligned_tools

    def align_sample_data_to_processed(self):
        """准备实测数据的显示坐标
        
        SampleData的行号保持原样不做任何处理，
        与预测负载通过相同的行号值进行叠加对齐显示。
        """
        if self.sample_data_values is None:
            return
        # 保持原始数据不变
        if self.sample_data_values_raw is None:
            self.sample_data_values_raw = self.sample_data_values
        if self.sample_data_program_numbers_raw is None:
            self.sample_data_program_numbers_raw = self.sample_data_program_numbers
        if self.sample_data_line_numbers_raw is None:
            self.sample_data_line_numbers_raw = self.sample_data_line_numbers
        
        # SampleData行号保持原样，不做映射转换
        self.sample_data_values = np.asarray(self.sample_data_values_raw)
        self.sample_data_program_numbers = np.asarray(self.sample_data_program_numbers_raw)
        self.sample_data_line_numbers = np.asarray(self.sample_data_line_numbers_raw, dtype=int)

        self.sample_data_time_indices = np.arange(len(self.sample_data_line_numbers), dtype=int)
        self.sample_data_base_blocks = self.compute_sequence_blocks(
            self.sample_data_line_numbers,
            self.sample_data_program_numbers
        )
        self.sample_data_x_positions = np.asarray(
            self.compute_line_x_positions(self.sample_data_line_numbers, blocks=self.sample_data_base_blocks),
            dtype=float
        )
        self.sample_data_point_indices = np.asarray(
            self.compute_line_point_indices(self.sample_data_line_numbers, blocks=self.sample_data_base_blocks),
            dtype=int
        )
        self.sample_data_valid_mask = None
        self.sample_data_valid_blocks = []

        # 区间也保持原样
        # 注意：不再调用 align_sample_program_ranges()，因为区间行号也应保持原样

    def show_sample_preview(self):
        """仅显示实测数据预览"""
        if not self.sample_data_loaded or self.sample_data_values is None:
            return
        source_idx = int(self.sample_data_source.get())
        sample_values_all = np.asarray(self.sample_data_values[:, source_idx])
        sample_line_numbers_all = np.asarray(self.sample_data_line_numbers)
        sample_time_indices_all = self.get_sample_time_indices_array()
        if sample_line_numbers_all is None or sample_time_indices_all is None:
            return

        display_mode = self.sample_display_mode.get()
        program_name = self.get_current_program_key()
        tool_id = self.get_selected_tool_id() if display_mode == "tool" else None
        if display_mode == "tool":
            tool_ranges = self.get_selected_tool_ranges()
        else:
            tool_ranges = self.get_program_ranges(program_name)

        program_no = self.get_selected_program_number()
        context_mask = self.build_sample_mask(program_no, None)
        valid_mask = self.build_sample_mask(program_no, tool_ranges)
        if context_mask is None:
            return

        context_blocks = self.compute_contiguous_blocks(context_mask)
        valid_blocks = self.compute_contiguous_blocks(valid_mask) if valid_mask is not None else []
        invalid_mask = context_mask & (~valid_mask if valid_mask is not None else False)
        invalid_blocks = self.compute_contiguous_blocks(invalid_mask)
        self.sample_data_valid_mask = valid_mask
        self.sample_data_valid_blocks = valid_blocks

        fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
        fig.patch.set_facecolor(PLOT_FIG_BG)
        self.apply_plot_style(ax, grid=False)
        self.apply_tool_background(ax, program_name, None if display_mode == "program" else tool_id)

        data_source_name = self.get_sample_data_source_name()
        if context_blocks:
            if invalid_blocks:
                self.plot_series_by_blocks(
                    ax,
                    sample_time_indices_all,
                    sample_values_all,
                    invalid_blocks,
                    color="#B0BEC5",
                    linewidth=1.1,
                    alpha=0.8,
                    label="非当前有效段",
                    zorder=3
                )
            if valid_blocks:
                self.plot_series_by_blocks(
                    ax,
                    sample_time_indices_all,
                    sample_values_all,
                    valid_blocks,
                    color=STYLE_MEASURED["color"],
                    linewidth=1.6,
                    alpha=0.95,
                    label=f"实测({data_source_name})",
                    zorder=4
                )
            elif context_blocks:
                self.plot_series_by_blocks(
                    ax,
                    sample_time_indices_all,
                    sample_values_all,
                    context_blocks,
                    color="#B0BEC5",
                    linewidth=1.1,
                    alpha=0.8,
                    label=f"实测({data_source_name})",
                    zorder=3
                )
            ax.set_axisbelow(True)
            ax.grid(True, alpha=0.35, color=PLOT_GRID_COLOR, linestyle='-', linewidth=0.6, zorder=-1)
        else:
            ax.text(0.5, 0.5, "当前选择无实测数据", ha='center', va='center',
                    transform=ax.transAxes, fontsize=PLOT_FONT_BASE, color='#666666')
            ax.grid(False)

        ax.set_title('实测负载预览', fontsize=PLOT_FONT_BASE, fontweight='bold', color=PLOT_TEXT_COLOR, pad=12)
        ax.set_xlabel('时间 (ms)', fontsize=PLOT_FONT_BASE, fontweight='bold', color=PLOT_TEXT_COLOR)
        ax.set_ylabel(f"实测负载 ({data_source_name})", fontsize=PLOT_FONT_BASE, fontweight='bold', color=PLOT_TEXT_COLOR)
        ax.tick_params(labelsize=PLOT_FONT_BASE, colors=PLOT_TEXT_COLOR)

        if context_mask.any():
            context_x = sample_time_indices_all[context_mask]
            x_min = float(np.min(context_x))
            x_max = float(np.max(context_x))
            x_range = x_max - x_min if x_max > x_min else 1
            ax.set_xlim(x_min - x_range * 0.02, x_max + x_range * 0.02)
            self.apply_line_axis_on_time(ax, context_mask)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            legend = ax.legend(loc='upper right', fontsize=PLOT_FONT_BASE, framealpha=0.9, shadow=False)
            legend.get_frame().set_facecolor(PLOT_AX_BG)
            legend.get_frame().set_edgecolor(PLOT_SPINE_COLOR)
            legend.get_frame().set_linewidth(0.8)

        fig.subplots_adjust(left=0.10, right=0.90, top=0.86, bottom=0.08)
        self.figures = [fig]
        self.figure_names = ["实测负载预览"]
        self.figure_selector["values"] = self.figure_names
        if self.figure_names:
            self.figure_selector.current(0)
        self.show_current_figure(0)

    def get_sample_data_source_name(self):
        """获取实测数据源名称"""
        return {0: "电流", 1: "VGpro功率", 2: "边缘模块功率"}.get(self.sample_data_source.get(), "电流")

    def _on_ratio_scale_change(self, value):
        """Scale command 回调：滑块值变化时调用
        
        策略：command 回调只做轻量更新（预览线），完整重绘由松开事件触发。
        这样可以确保拖动时流畅，松开时才做耗时的完整重绘。
        """
        if self._ratio_update_lock:
            return
        
        # 更新显示值（立即更新，提供即时反馈）
        try:
            ratio = float(value)
            self.adjustment_ratio_display.set(f"{ratio:.2f}")
        except Exception:
            return
        
        # 取消待执行的防抖定时器
        if self._ratio_debounce_timer is not None:
            self.root.after_cancel(self._ratio_debounce_timer)
            self._ratio_debounce_timer = None
        
        # command 回调中始终只做预览更新，不做完整重绘
        # 完整重绘由 _on_ratio_release（松开）或防抖定时器（非鼠标操作）触发
        self.update_preview_ideal_lines()
        
        # 标记正在通过滑块交互（用于 release 判断是否需要完整重绘）
        self._ratio_scale_interacting = True

    def on_adjustment_ratio_change(self, *args):
        """优化倍率变更：添加防抖机制，避免频繁更新"""
        if self._ratio_update_lock:
            return
        
        # 如果正在执行 release 更新，跳过防抖调度避免重复刷新
        if getattr(self, "_ratio_release_updating", False):
            return
        
        # 更新显示值（立即更新，提供即时反馈）
        try:
            ratio = float(self.adjustment_ratio.get())
            self.adjustment_ratio_display.set(f"{ratio:.2f}")
        except Exception:
            return
        
        # 正在拖动滑块时（包括刚按下的瞬间），仅更新虚线/背景预览，不触发整图重绘
        if self._ratio_dragging:
            if self._ratio_debounce_timer is not None:
                self.root.after_cancel(self._ratio_debounce_timer)
                self._ratio_debounce_timer = None
            self.update_preview_ideal_lines()
            return
        
        # 取消之前的防抖定时器
        if self._ratio_debounce_timer is not None:
            self.root.after_cancel(self._ratio_debounce_timer)
        
        # 设置新的防抖定时器，延迟执行图表更新
        self._ratio_debounce_timer = self.root.after(
            self._ratio_debounce_delay, 
            self._apply_ratio_update
        )

    def _on_ratio_press(self, _event=None):
        """滑块按下：进入拖动预览模式，取消待触发的重绘"""
        # 设置拖动标志
        self._ratio_dragging = True
        self._ratio_scale_interacting = True  # 标记开始交互，松开时需要完整重绘
        # 取消所有待触发的防抖定时器
        if self._ratio_debounce_timer is not None:
            self.root.after_cancel(self._ratio_debounce_timer)
            self._ratio_debounce_timer = None

    def _on_ratio_release(self, _event=None):
        """滑块松开：退出预览模式并执行完整重绘"""
        self._ratio_dragging = False
        # 取消可能由松开瞬间触发的防抖定时器，避免重复刷新
        if self._ratio_debounce_timer is not None:
            self.root.after_cancel(self._ratio_debounce_timer)
            self._ratio_debounce_timer = None
        
        # 检查是否有交互（值变化），如果没有则跳过
        if not getattr(self, "_ratio_scale_interacting", False):
            return
        
        self._ratio_scale_interacting = False
        # 设置标志阻止其他回调在此期间触发重复更新
        self._ratio_release_updating = True
        try:
            # 直接应用更新，确保背景条和其它元素刷新
            self._apply_ratio_update()
        finally:
            self._ratio_release_updating = False
    
    def _apply_ratio_update(self):
        """实际应用优化倍率更新（防抖后执行）"""
        self._ratio_debounce_timer = None
        
        # 保存当前刀具的优化倍率到ideal_store
        prog = self.get_current_program_key()
        tool = self.get_selected_tool_id()
        if prog and tool:
            try:
                rg = float(self.adjustment_ratio.get())
                key = (prog, tool)
                self.ideal_store[key] = {
                    "rg": rg,
                    "updated_at": datetime.now().isoformat()
                }
                self._persist_ideal_store()
                self._refresh_ideal_tree()
            except Exception:
                pass
        
        # 同时更新理想值显示标签
        self._refresh_current_ideal_display()

        # 统一刷新图表（避免重复绘图）
        if self.data:
            try:
                # generate_plots会完整重绘，无需先调用update_preview_ideal_lines
                self.generate_plots(save=False, silent=True)
            except Exception:
                pass
        else:
            # 无数据时仅更新预览线
            self.update_preview_ideal_lines()

    def update_preview_ideal_lines(self):
        """仅更新当前刀具的理想值预览线和背景条高度，使用优化的增量更新避免整图重绘"""
        if self.sample_display_mode.get() != "tool":
            return
        if not self._ideal_line_artists or self._preview_tool_mean is None:
            return
        try:
            ratio = float(self.adjustment_ratio.get())
        except Exception:
            return
        new_val = self._preview_tool_mean * ratio
        
        # 标记是否有内容需要更新
        needs_redraw = False
        
        # 批量更新理想值预览线（减少单独的set操作）
        for line in self._ideal_line_artists:
            line.set_ydata([new_val, new_val])
            needs_redraw = True
        
        # 更新背景条高度（如果存在）- 优化版本
        if hasattr(self, '_interval_background_artists') and self._interval_background_artists:
            for artist in self._interval_background_artists:
                try:
                    # PolyCollection from fill_between: 批量更新顶点Y坐标
                    paths = artist.get_paths()
                    for path in paths:
                        vertices = path.vertices
                        # 使用numpy向量化操作提高性能
                        mask = vertices[:, 1] > 0  # 找到需要更新的Y坐标
                        vertices[mask, 1] = new_val  # 批量更新
                        needs_redraw = True
                except Exception:
                    pass
        
        # 只有确实有更新时才调用draw_idle，避免不必要的重绘
        if needs_redraw and hasattr(self, "canvas_data") and self.canvas_data:
            self.canvas_data.draw_idle()

    def get_adjustment_ratio_for_view(self):
        """获取当前视图对应的优化倍率"""
        try:
            return float(self.adjustment_ratio.get())
        except Exception:
            return 2.0

    def sync_adjustment_ratio_for_current_view(self):
        """切换视图/组合时同步优化倍率到界面"""
        if self.sample_display_mode.get() != "tool":
            return
        prog = self.get_current_program_key()
        tool = self.get_selected_tool_id()
        if not prog or not tool:
            return
        store = self.ideal_store.get((prog, tool))
        try:
            ratio = float(store.get("rg")) if store else 2.0
        except Exception:
            ratio = 2.0
        self._ratio_update_lock = True
        try:
            self.adjustment_ratio.set(ratio)
            self.adjustment_ratio_display.set(f"{ratio:.2f}")
        finally:
            self._ratio_update_lock = False

    def get_current_program_key(self):
        """获取当前程序名"""
        program_name = self.sample_program_name.get().strip()
        return program_name if program_name else None

    def get_current_tool_key(self):
        """获取当前程序名+刀具号组合键"""
        if self.sample_display_mode.get() != "tool":
            return None
        program_name = self.sample_program_name.get().strip()
        tool_id = self.get_selected_tool_id()
        if not program_name or not tool_id:
            return None
        return (program_name, tool_id)

    def get_selected_program_number(self):
        """获取当前程序号（字符串）"""
        program_name = self.sample_program_name.get().strip()
        if not program_name:
            return None
        program_info = self.sample_programs.get(program_name)
        if not program_info:
            return None
        return program_info.get("program_number")

    def get_selected_tool_id(self):
        """获取当前刀具号"""
        program_name = self.sample_program_name.get().strip()
        if not program_name:
            return None
        program_info = self.sample_programs.get(program_name, {})
        display_label = self.sample_tool_name.get().strip()
        if not display_label:
            return None
        tool_map = program_info.get("tool_display_map", {})
        if display_label in tool_map:
            return tool_map[display_label]
        return display_label.split()[0] if display_label else None

    def get_selected_tool_ranges(self):
        """获取当前刀具的对齐行号范围列表"""
        program_name = self.sample_program_name.get().strip()
        tool_id = self.get_selected_tool_id()
        if not program_name or not tool_id:
            return None
        program_info = self.sample_programs.get(program_name, {})
        return program_info.get("tools", {}).get(tool_id, None)

    def get_program_ranges(self, program_name):
        """获取程序名下所有刀具的对齐行号范围列表"""
        if not program_name:
            return None
        program_info = self.sample_programs.get(program_name, {})
        ranges = []
        for tool_ranges in program_info.get("tools", {}).values():
            ranges.extend(tool_ranges)
        return ranges if ranges else None

    def compute_contiguous_blocks(self, mask):
        """将布尔掩码转换为连续区块索引列表 [(start, end), ...]。"""
        if mask is None:
            return []
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.size == 0:
            return []

        blocks = []
        start = None
        for idx, flag in enumerate(mask_arr):
            if flag and start is None:
                start = idx
            elif not flag and start is not None:
                blocks.append((start, idx - 1))
                start = None
        if start is not None:
            blocks.append((start, len(mask_arr) - 1))
        return blocks

    def compute_sequence_blocks(self, line_numbers, group_keys=None):
        """按时间顺序切分序列块。

        满足以下任一条件时开启新区块：
        1. 分组键变化（如程序号切换）
        2. 行号下降（典型的下一次循环/下一段开始）
        """
        if line_numbers is None:
            return []

        line_arr = np.asarray(line_numbers)
        if line_arr.size == 0:
            return []

        group_arr = np.asarray(group_keys) if group_keys is not None else None
        blocks = []
        start = 0
        for idx in range(1, len(line_arr)):
            split = False
            if group_arr is not None and group_arr[idx] != group_arr[idx - 1]:
                split = True
            else:
                try:
                    split = int(line_arr[idx]) < int(line_arr[idx - 1])
                except Exception:
                    split = line_arr[idx] != line_arr[idx - 1]
            if split:
                blocks.append((start, idx - 1))
                start = idx
        blocks.append((start, len(line_arr) - 1))
        return blocks

    def _compute_line_x_positions_single_block(self, line_numbers):
        """基于行号生成可视化x坐标
        
        当多个数据点具有相同的行号时，将它们平均排布在单位区间内。
        例如：行号11有3个点，则分布在 11.0, 11.333, 11.667
        """
        if line_numbers is None:
            return []
        line_point_counts = {}
        point_indices = []
        for ln in line_numbers:
            count = line_point_counts.get(ln, 0) + 1
            line_point_counts[ln] = count
            point_indices.append(count)
        x_positions = []
        for ln, idx in zip(line_numbers, point_indices):
            total_pts = line_point_counts.get(ln, 1)
            if total_pts <= 1:
                # 只有一个点，放在行号位置
                x_positions.append(float(ln))
            else:
                # 多个点均匀分布在 [ln, ln+1) 区间内
                # 第1个点在 ln，最后一个点在 ln + (total_pts-1)/total_pts
                x_positions.append(float(ln) + (idx - 1) / total_pts)
        return x_positions

    def compute_line_x_positions(self, line_numbers, blocks=None):
        """基于行号生成可视化x坐标，可按区块独立重置。"""
        if line_numbers is None:
            return []
        line_list = list(line_numbers)
        if not line_list:
            return []
        if not blocks:
            return self._compute_line_x_positions_single_block(line_list)

        x_positions = [math.nan] * len(line_list)
        for start, end in blocks:
            if start > end:
                continue
            local_x = self._compute_line_x_positions_single_block(line_list[start:end + 1])
            x_positions[start:end + 1] = local_x
        return x_positions

    def _compute_line_segment_bounds_single_block(self, line_numbers):
        """计算每行工艺信息对应的区间边界 [start_x, end_x]
        
        每行占据一段区间而非一个点：
        - 行号11只有1行 → [11, 12)
        - 行号11有3行 → [11, 11.333), [11.333, 11.667), [11.667, 12)
        
        返回: (start_bounds, end_bounds) 两个列表
        """
        if not line_numbers:
            return [], []
        
        # 统计每个行号有多少行
        from collections import Counter
        line_counts = Counter(line_numbers)
        line_indices = {}
        
        start_bounds = []
        end_bounds = []
        
        for ln in line_numbers:
            idx = line_indices.get(ln, 0)
            total = line_counts[ln]
            
            start_x = float(ln) + idx / total
            end_x = float(ln) + (idx + 1) / total
            
            start_bounds.append(start_x)
            end_bounds.append(end_x)
            
            line_indices[ln] = idx + 1
        
        return start_bounds, end_bounds

    def compute_line_segment_bounds(self, line_numbers, blocks=None):
        """计算每行工艺信息对应的区间边界 [start_x, end_x]，支持按区块独立重置。"""
        if line_numbers is None:
            return [], []
        line_list = list(line_numbers)
        if not line_list:
            return [], []
        if not blocks:
            return self._compute_line_segment_bounds_single_block(line_list)

        start_bounds = [math.nan] * len(line_list)
        end_bounds = [math.nan] * len(line_list)
        for start, end in blocks:
            if start > end:
                continue
            local_start, local_end = self._compute_line_segment_bounds_single_block(line_list[start:end + 1])
            start_bounds[start:end + 1] = local_start
            end_bounds[start:end + 1] = local_end
        return start_bounds, end_bounds

    def compute_line_point_widths(self, line_numbers):
        """计算每个点在其行号下的宽度
        
        行号L有n个点时，每个点的宽度是1/n
        """
        if line_numbers is None:
            return []
        line_point_counts = {}
        for ln in line_numbers:
            line_point_counts[ln] = line_point_counts.get(ln, 0) + 1
        widths = []
        for ln in line_numbers:
            total_pts = line_point_counts.get(ln, 1)
            widths.append(1.0 / total_pts if total_pts > 0 else 1.0)
        return widths

    def _compute_line_point_indices_single_block(self, line_numbers):
        """计算每个点在其行号下的序号(从0开始)"""
        if line_numbers is None:
            return []
        line_point_counts = {}
        point_indices = []
        for ln in line_numbers:
            count = line_point_counts.get(ln, 0)
            point_indices.append(count)
            line_point_counts[ln] = count + 1
        return point_indices

    def compute_line_point_indices(self, line_numbers, blocks=None):
        """计算每个点在其行号下的序号(从0开始)，支持按区块独立重置。"""
        if line_numbers is None:
            return []
        line_list = list(line_numbers)
        if not line_list:
            return []
        if not blocks:
            return self._compute_line_point_indices_single_block(line_list)

        point_indices = [0] * len(line_list)
        for start, end in blocks:
            if start > end:
                continue
            local_indices = self._compute_line_point_indices_single_block(line_list[start:end + 1])
            point_indices[start:end + 1] = local_indices
        return point_indices

    def plot_series_by_blocks(self, ax, x_values, y_values, blocks, **plot_kwargs):
        """按连续区块分别绘制折线，避免跨块连线。"""
        if x_values is None or y_values is None or not blocks:
            return []

        x_arr = np.asarray(x_values)
        y_arr = np.asarray(y_values)
        artists = []
        first = True
        for start, end in blocks:
            if start > end:
                continue
            block_kwargs = dict(plot_kwargs)
            if not first:
                block_kwargs.pop("label", None)
            artist = ax.plot(x_arr[start:end + 1], y_arr[start:end + 1], **block_kwargs)
            artists.extend(artist)
            first = False
        return artists

    def get_sample_time_indices_array(self):
        """返回 SampleData 的时间轴数组（每点1ms）。"""
        if self.sample_data_line_numbers is None:
            return None
        if self.sample_data_time_indices is None or len(self.sample_data_time_indices) != len(self.sample_data_line_numbers):
            self.sample_data_time_indices = np.arange(len(self.sample_data_line_numbers), dtype=int)
        return np.asarray(self.sample_data_time_indices, dtype=float)

    def build_sample_line_time_spans(self, mask):
        """基于 SampleData 时间轴构建各程序行号的时间跨度。"""
        if self.sample_data_line_numbers is None:
            return {}, []
        time_arr = self.get_sample_time_indices_array()
        if time_arr is None:
            return {}, []

        line_arr = np.asarray(self.sample_data_line_numbers, dtype=int)
        mask_arr = np.asarray(mask, dtype=bool) if mask is not None else np.ones(len(line_arr), dtype=bool)
        if mask_arr.size != len(line_arr):
            return {}, []

        spans_by_line = collections.defaultdict(list)
        ordered_spans = []
        for block_start, block_end in self.compute_contiguous_blocks(mask_arr):
            run_start = block_start
            for idx in range(block_start + 1, block_end + 1):
                if line_arr[idx] != line_arr[idx - 1]:
                    ln = int(line_arr[run_start])
                    start_t = float(time_arr[run_start])
                    end_t = float(time_arr[idx - 1] + 1.0)
                    spans_by_line[ln].append((start_t, end_t))
                    ordered_spans.append((ln, start_t, end_t))
                    run_start = idx
            ln = int(line_arr[run_start])
            start_t = float(time_arr[run_start])
            end_t = float(time_arr[block_end] + 1.0)
            spans_by_line[ln].append((start_t, end_t))
            ordered_spans.append((ln, start_t, end_t))
        return spans_by_line, ordered_spans

    def allocate_items_to_spans(self, item_count, spans):
        """按跨度长度将若干项目分配到多个时间跨度。"""
        if item_count <= 0 or not spans:
            return []
        if len(spans) == 1:
            return [(spans[0], item_count)]

        lengths = np.asarray([max(1e-9, end - start) for start, end in spans], dtype=float)
        total = float(np.sum(lengths))
        if total <= 0:
            return [(spans[0], item_count)]

        raw = lengths / total * float(item_count)
        counts = np.floor(raw).astype(int)
        remainders = raw - counts
        assigned = int(np.sum(counts))
        remaining = item_count - assigned

        if remaining > 0:
            order = np.argsort(-remainders)
            for idx in order[:remaining]:
                counts[int(idx)] += 1

        if np.sum(counts) == 0:
            counts[0] = item_count

        allocations = []
        for idx, count in enumerate(counts):
            if count <= 0:
                continue
            allocations.append((spans[idx], int(count)))

        total_allocated = sum(count for _, count in allocations)
        if total_allocated < item_count and allocations:
            span, count = allocations[-1]
            allocations[-1] = (span, count + (item_count - total_allocated))
        return allocations

    def compute_process_time_bounds_from_sample(self, process_line_numbers, sample_mask):
        """将工艺信息行映射到 SampleData 的时间轴跨度。"""
        if process_line_numbers is None:
            return [], []

        process_lines = np.asarray(process_line_numbers, dtype=int)
        start_bounds = [math.nan] * len(process_lines)
        end_bounds = [math.nan] * len(process_lines)
        if process_lines.size == 0:
            return start_bounds, end_bounds

        spans_by_line, _ = self.build_sample_line_time_spans(sample_mask)
        if not spans_by_line:
            return start_bounds, end_bounds

        positions_by_line = collections.OrderedDict()
        for idx, ln in enumerate(process_lines):
            positions_by_line.setdefault(int(ln), []).append(idx)

        for ln, positions in positions_by_line.items():
            spans = spans_by_line.get(int(ln))
            if not spans:
                continue

            allocations = self.allocate_items_to_spans(len(positions), spans)
            pos_cursor = 0
            for span, count in allocations:
                start_t, end_t = span
                width = (end_t - start_t) / count if count > 0 else 0.0
                for local_idx in range(count):
                    if pos_cursor >= len(positions):
                        break
                    proc_idx = positions[pos_cursor]
                    start_bounds[proc_idx] = start_t + local_idx * width
                    end_bounds[proc_idx] = start_t + (local_idx + 1) * width
                    pos_cursor += 1

        return start_bounds, end_bounds

    def build_time_spans_for_ranges(self, ranges, sample_mask):
        """将刀具/程序行号范围转换为 SampleData 时间轴区间。"""
        if not ranges or self.sample_data_line_numbers is None:
            return []
        time_arr = self.get_sample_time_indices_array()
        if time_arr is None:
            return []

        line_arr = np.asarray(self.sample_data_line_numbers, dtype=int)
        base_mask = np.asarray(sample_mask, dtype=bool) if sample_mask is not None else np.ones(len(line_arr), dtype=bool)
        if base_mask.size != len(line_arr):
            return []

        range_mask = np.zeros(len(line_arr), dtype=bool)
        for start_line, end_line in ranges:
            range_mask |= (line_arr >= start_line) & (line_arr <= end_line)
        target_mask = base_mask & range_mask

        spans = []
        for block_start, block_end in self.compute_contiguous_blocks(target_mask):
            spans.append((float(time_arr[block_start]), float(time_arr[block_end] + 1.0)))
        return spans

    def remove_line_axis_on_time(self, ax):
        """移除指定坐标轴顶部的程序行号辅助轴。"""
        top_ax = getattr(ax, '_time_line_top_axis', None)
        if top_ax is None:
            return
        try:
            top_ax.remove()
        except Exception:
            pass
        ax._time_line_top_axis = None

    def get_time_line_axis_capacity(self, ax, min_ticks=8, max_ticks=60):
        """根据当前轴宽度估算顶部程序行号可容纳的刻度数。"""
        try:
            bbox = ax.get_window_extent()
            width_px = float(getattr(bbox, 'width', 0.0))
        except Exception:
            width_px = 0.0
        if width_px <= 0:
            return min_ticks
        tick_count = int(width_px // 75)
        return max(min_ticks, min(max_ticks, tick_count))

    def sample_time_line_spans(self, spans, target_count):
        """从可见跨度中抽样，尽量保留首尾和中间标签。"""
        if not spans or target_count <= 0 or len(spans) <= target_count:
            return list(spans)

        indices = sorted(set(np.linspace(0, len(spans) - 1, num=target_count, dtype=int).tolist()))
        middle_idx = len(spans) // 2
        indices.append(middle_idx)
        indices = sorted(set(indices))

        while len(indices) > target_count:
            protected = {indices[0], indices[-1], middle_idx}
            removable_positions = [
                pos for pos, idx in enumerate(indices)
                if idx not in protected
            ]
            if not removable_positions:
                removable_positions = list(range(1, max(1, len(indices) - 1)))
            if not removable_positions:
                break
            remove_pos = min(
                removable_positions,
                key=lambda pos: min(
                    indices[pos] - indices[pos - 1] if pos > 0 else float('inf'),
                    indices[pos + 1] - indices[pos] if pos < len(indices) - 1 else float('inf')
                )
            )
            indices.pop(remove_pos)

        return [spans[idx] for idx in indices]

    def on_time_line_axis_xlim_changed(self, ax):
        """横轴变化后刷新顶部程序行号轴。"""
        if getattr(ax, '_time_line_refreshing', False):
            return
        mask = getattr(ax, '_time_line_mask', None)
        if mask is None:
            return
        ax._time_line_refreshing = True
        try:
            self.apply_line_axis_on_time(ax, mask)
        finally:
            ax._time_line_refreshing = False

    def bind_time_line_axis_updates(self, ax):
        """确保横轴缩放/平移时自动刷新顶部程序行号轴。"""
        if getattr(ax, '_time_line_callback_bound', False):
            return
        try:
            ax.callbacks.connect('xlim_changed', self.on_time_line_axis_xlim_changed)
            ax._time_line_callback_bound = True
        except Exception:
            pass

    def apply_line_axis_on_time(self, ax, sample_mask, max_ticks=60):
        """在时间轴上方附加程序行号辅助坐标轴。"""
        if self.sample_data_line_numbers is None:
            return None
        time_arr = self.get_sample_time_indices_array()
        if time_arr is None:
            return None

        self.remove_line_axis_on_time(ax)
        mask_arr = np.asarray(sample_mask, dtype=bool) if sample_mask is not None else np.ones(len(self.sample_data_line_numbers), dtype=bool)
        if mask_arr.size != len(self.sample_data_line_numbers):
            return None

        _, ordered_spans = self.build_sample_line_time_spans(mask_arr)
        if not ordered_spans:
            return None

        x_min, x_max = ax.get_xlim()
        visible_spans = [
            (ln, start_t, end_t)
            for ln, start_t, end_t in ordered_spans
            if end_t >= x_min and start_t <= x_max
        ]
        if not visible_spans:
            visible_spans = ordered_spans

        adaptive_tick_count = self.get_time_line_axis_capacity(ax, min_ticks=8, max_ticks=max_ticks)
        visible_spans = self.sample_time_line_spans(visible_spans, adaptive_tick_count)

        tick_positions = []
        tick_labels = []
        for ln, start_t, end_t in visible_spans:
            span_left = max(start_t, x_min)
            span_right = min(end_t, x_max)
            tick_positions.append((span_left + span_right) / 2.0)
            tick_labels.append(str(int(ln)))

        top_ax = ax.secondary_xaxis('top')
        top_ax.set_xticks(tick_positions)
        top_ax.set_xticklabels(tick_labels, rotation=45, ha='left')
        top_ax.set_xlabel('程序行号', fontsize=PLOT_FONT_BASE, fontweight='bold', color='black')
        top_ax.tick_params(axis='x', labelsize=PLOT_FONT_BASE - 1, colors='black')
        try:
            top_ax.spines['top'].set_color('black')
        except Exception:
            pass
        ax._time_line_mask = mask_arr.copy()
        ax._time_line_top_axis = top_ax
        self.bind_time_line_axis_updates(ax)
        return top_ax

    def apply_line_axis(self, ax, line_numbers):
        """设置对齐行号刻度 - 使用自动定位器实现动态刻度调整"""
        if line_numbers is None or len(line_numbers) == 0:
            return
        unique_lines = sorted(set(line_numbers))
        if not unique_lines:
            return
        
        # 使用 MaxNLocator 实现自动刻度调整，避免刻度过密或过疏
        # nbins='auto' 会根据可用空间自动调整刻度数量
        # integer=True 确保刻度值为整数
        ax.xaxis.set_major_locator(MaxNLocator(nbins='auto', integer=True, prune='both'))
        
        # 设置刻度标签样式
        ax.tick_params(axis='x', rotation=45, labelsize=PLOT_FONT_BASE, colors=PLOT_TEXT_COLOR)
        
        # 调整刻度标签对齐方式
        for label in ax.get_xticklabels():
            label.set_ha('right')

    def apply_plot_style(self, ax, grid=True, text_color=PLOT_TEXT_COLOR, transparent=False):
        """应用工业科技感绘图样式 - 干净专业"""
        if not transparent:
            ax.set_facecolor(PLOT_AX_BG)
        
        # 简化边框 - 只显示左下边框
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color(PLOT_SPINE_COLOR)
            ax.spines[spine].set_linewidth(0.8)
        
        # 刻度样式
        ax.tick_params(labelsize=PLOT_FONT_BASE, colors=text_color, 
                      direction='out', length=4, width=0.8)
        
        # 网格样式 - 轻柔辅助
        if grid:
            ax.set_axisbelow(True)
            ax.grid(True, alpha=0.4, color=PLOT_GRID_COLOR, linestyle='-', linewidth=0.5)

    def merge_intervals(self, intervals, debug=False, merge_adjacent=False):
        """合并重叠区间（可选合并相邻区间）
        :param debug: 是否打印调试信息
        :param merge_adjacent: 是否合并相邻区间（行号差1的区间）。
                               默认False，只合并真正重叠的区间（start < last_end），
                               保持不同功率值区间之间的边界。
        """
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
        merged = [intervals[0]]
        
        for start, end in intervals[1:]:
            last_start, last_end = merged[-1]
            # 判断条件：是否合并
            # - merge_adjacent=True: start <= last_end + 1 (合并重叠和相邻)
            # - merge_adjacent=False: start < last_end (只合并真正重叠的区间，不合并边界相接的)
            if merge_adjacent:
                should_merge = start <= last_end + 1
            else:
                should_merge = start < last_end  # 严格重叠才合并，不合并 start == last_end 的情况
            
            if should_merge:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        
        return merged

    def clip_intervals_to_ranges(self, intervals, ranges):
        """将区间裁剪到指定范围"""
        if not intervals:
            return []
        if not ranges:
            return self.merge_intervals(intervals)
        clipped = []
        for start, end in intervals:
            for r_start, r_end in ranges:
                if end < r_start or start > r_end:
                    continue
                clipped_start = max(start, r_start)
                clipped_end = min(end, r_end)
                if clipped_start <= clipped_end:
                    clipped.append((clipped_start, clipped_end))
        return self.merge_intervals(clipped)

    def get_predicted_intervals_for_display(self, ranges=None):
        """获取用于显示的稳态区间（按行号）- 直接返回原始区间，不进行裁剪合并"""
        base_intervals = []
        for interval in self.pred_power_intervals:
            start_line = interval.get('start_line')
            end_line = interval.get('end_line')
            if start_line is None or end_line is None:
                continue
            base_intervals.append((int(start_line), int(end_line)))
        # 直接返回原始区间，不进行裁剪和合并
        return base_intervals

    def compute_measured_avg_in_intervals(self, values, line_numbers, intervals):
        """计算实测数据在稳态区间内的平均值"""
        if values is None or line_numbers is None or not intervals:
            return None, 0
        values = np.asarray(values)
        line_numbers = np.asarray(line_numbers)
        mask = np.zeros(len(values), dtype=bool)
        for start, end in intervals:
            mask |= (line_numbers >= start) & (line_numbers <= end)
        if not mask.any():
            return None, 0
        return float(np.mean(values[mask])), int(np.sum(mask))

    def get_program_info(self, program_name):
        """获取程序信息字典"""
        if not program_name:
            return {}
        return self.sample_programs.get(program_name, {})

    def get_program_number_by_name(self, program_name):
        """按程序名获取程序号"""
        program_info = self.get_program_info(program_name)
        return program_info.get("program_number")

    def get_tool_ranges_by_id(self, program_name, tool_id):
        """按程序名+刀具号获取对齐行号范围"""
        program_info = self.get_program_info(program_name)
        return program_info.get("tools", {}).get(tool_id, [])

    def build_sample_mask(self, program_no=None, tool_ranges=None):
        """构建实测数据筛选mask（按程序号 + 刀具区间）"""
        if self.sample_data_line_numbers is None or self.sample_data_program_numbers is None:
            return None
        mask = np.ones(len(self.sample_data_line_numbers), dtype=bool)
        if program_no is not None:
            mask &= (self.sample_data_program_numbers == program_no)
        if tool_ranges:
            range_mask = np.zeros(len(self.sample_data_line_numbers), dtype=bool)
            for r_start, r_end in tool_ranges:
                range_mask |= (self.sample_data_line_numbers >= r_start) & (self.sample_data_line_numbers <= r_end)
            mask &= range_mask
        return mask

    def build_process_mask(self, line_numbers, tool_ranges=None):
        """构建工艺信息筛选mask（按刀具区间）。"""
        if line_numbers is None:
            return None
        line_arr = np.asarray(line_numbers)
        mask = np.ones(len(line_arr), dtype=bool)
        if tool_ranges:
            mask = np.zeros(len(line_arr), dtype=bool)
            for r_start, r_end in tool_ranges:
                mask |= (line_arr >= r_start) & (line_arr <= r_end)
        return mask

    def count_points_by_blocks(self, mask, mode="sum"):
        """按连续区块统计点数。"""
        blocks = self.compute_contiguous_blocks(mask)
        if not blocks:
            return 0, []
        counts = [end - start + 1 for start, end in blocks]
        if mode == "max_block":
            return max(counts), blocks
        return sum(counts), blocks

    def compute_tool_measured_mean(self, program_name, tool_id):
        """计算指定程序+刀具在稳态区间内的实测均值"""
        if not self.sample_data_loaded or self.sample_data_values is None:
            return None, 0, []
        program_no = self.get_program_number_by_name(program_name)
        tool_ranges = self.get_tool_ranges_by_id(program_name, tool_id)
        intervals = self.get_predicted_intervals_for_display(tool_ranges)
        if not intervals:
            return None, 0, intervals
        mask = self.build_sample_mask(program_no, tool_ranges)
        if mask is None or not mask.any():
            return None, 0, intervals
        source_idx = int(self.sample_data_source.get())
        values = self.sample_data_values[:, source_idx][mask]
        line_numbers = self.sample_data_line_numbers[mask]
        mean_val, count = self.compute_measured_avg_in_intervals(values, line_numbers, intervals)
        return mean_val, count, intervals

    def format_line_point(self, line_number, point_index):
        """格式化行点：行号.点序号"""
        try:
            ln = int(line_number)
            pt = int(point_index)
        except Exception:
            return f"{line_number}.{point_index}"
        return f"{ln}.{pt}"

    def collect_line_point_intervals_for_tool(self, program_name, tool_id):
        """获取指定程序+刀具的稳态区间行点范围（包含每个区间的平均值）
        
        使用工艺信息索引的精确x坐标边界来匹配SampleData点，
        确保与图表绘制的区间边界完全一致。
        """
        if not self.sample_data_loaded or self.sample_data_line_numbers is None:
            return []
        if not self.pred_power_intervals:
            return []
        if not self.data:
            return []

        program_no = self.get_program_number_by_name(program_name)
        tool_ranges = self.get_tool_ranges_by_id(program_name, tool_id)
        if not tool_ranges:
            return []
        
        base_mask = self.build_sample_mask(program_no, tool_ranges)
        if base_mask is None or not base_mask.any():
            return []
        
        # 获取SampleData的数据
        sample_line_numbers = np.asarray(self.sample_data_line_numbers)
        sample_point_indices = np.asarray(self.sample_data_point_indices)
        sample_x_positions = self.sample_data_x_positions
        if sample_x_positions is None:
            sample_x_positions = np.asarray(
                self.compute_line_x_positions(self.sample_data_line_numbers, blocks=self.sample_data_base_blocks)
            )
        else:
            sample_x_positions = np.asarray(sample_x_positions)
        
        # 计算工艺信息数据的区间边界（与图表绘制相同）
        process_line_numbers = [d.get('line_no_aligned', i) for i, d in enumerate(self.data)]
        process_blocks = self.compute_sequence_blocks(process_line_numbers)
        process_start_bounds, process_end_bounds = self.compute_line_segment_bounds(
            process_line_numbers,
            blocks=process_blocks
        )
        
        # 获取实测数据值用于计算平均值
        data_source_idx = int(self.sample_data_source.get())
        if self.sample_data_values is not None and len(self.sample_data_values.shape) > 1:
            sample_values = self.sample_data_values[:, data_source_idx]
        else:
            sample_values = self.sample_data_values
        
        interval_strings = []
        
        for interval in self.pred_power_intervals:
            start_idx_iv = interval.get('start_idx')
            end_idx_iv = interval.get('end_idx')
            start_line = interval.get('start_line')
            end_line = interval.get('end_line')
            
            if start_idx_iv is None or end_idx_iv is None:
                continue
            if start_line is None or end_line is None:
                continue
            
            # 检查是否在当前刀具范围内
            in_range = False
            for r_start, r_end in tool_ranges:
                if not (end_line < r_start or start_line > r_end):
                    in_range = True
                    break
            if not in_range:
                continue
            
            # 获取工艺信息区间的边界（与图表绘制一致）
            if start_idx_iv >= len(process_start_bounds) or end_idx_iv >= len(process_end_bounds):
                continue
            
            interval_start_x = process_start_bounds[start_idx_iv]
            interval_end_x = process_end_bounds[end_idx_iv]
            
            # 在SampleData中找x坐标落在区间内的点
            # 使用半开区间 [start_x, end_x)
            mask = base_mask & (sample_x_positions >= interval_start_x) & (sample_x_positions < interval_end_x)
            if not mask.any():
                continue

            for block_start, block_end in self.compute_contiguous_blocks(mask):
                first_idx = int(block_start)
                last_idx = int(block_end)

                start_lp = self.format_line_point(sample_line_numbers[first_idx], sample_point_indices[first_idx])
                end_lp = self.format_line_point(sample_line_numbers[last_idx], sample_point_indices[last_idx])

                interval_values = sample_values[block_start:block_end + 1]
                finite_mask = np.isfinite(interval_values)
                if finite_mask.any():
                    avg_value = float(np.mean(interval_values[finite_mask]))
                else:
                    avg_value = 0.0

                interval_strings.append(f"{start_lp}-{end_lp}:{avg_value:.6f}")
        
        return interval_strings

    def draw_tool_mean_ideal_lines(self, ax, tool_ranges, mean_val, ideal_val, color, label_prefix=None, display_spans=None):
        """在指定刀具区间范围绘制均值/理想值线"""
        if mean_val is None or not tool_ranges:
            return []
        spans = list(display_spans) if display_spans else [(start, end + 1) for start, end in tool_ranges]
        if not spans:
            return []
        ideal_lines = []
        for start, end in spans:
            ax.plot([start, end], [mean_val, mean_val],
                    color=color, linewidth=1.2, zorder=6)
            if ideal_val is not None:
                line, = ax.plot([start, end], [ideal_val, ideal_val],
                                color=color, linestyle='--', linewidth=1.1, zorder=6)
                ideal_lines.append(line)
        if label_prefix:
            last_start, last_end = spans[-1]
            x_offset = max((last_end - last_start) * 0.03, 0.3)
            ax.text(last_end + x_offset, mean_val, f"{label_prefix}均值:{mean_val:.3f}",
                    fontsize=PLOT_FONT_BASE, color=color, va='center')
            if ideal_val is not None:
                ax.text(last_end + x_offset, ideal_val, f"{label_prefix}理想:{ideal_val:.3f}",
                        fontsize=PLOT_FONT_BASE, color=color, va='center')
        return ideal_lines

    def parse_sample_program_file(self, txt_path):
        """解析程序区间交互文件 SampleData.txt"""
        programs = {}
        with open(txt_path, 'r', encoding='utf-8') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                if line.endswith(';'):
                    line = line[:-1]
                parts = line.split(':')
                if len(parts) < 3:
                    continue
                program_name = parts[0].strip()
                program_number = parts[1].strip()
                tools_part = ":".join(parts[2:]).strip()
                tool_segments = [seg for seg in tools_part.split(';') if seg.strip()]
                if program_name not in programs:
                    programs[program_name] = {
                        'program_number': program_number,
                        'tools': {},
                        'tool_raw_ranges': {},
                        'tool_display_ranges': {},
                        'tool_display_map': {}
                    }
                program_entry = programs[program_name]
                if not program_entry.get('program_number') and program_number:
                    program_entry['program_number'] = program_number
                tools = program_entry.setdefault('tools', {})
                tool_raw_ranges = program_entry.setdefault('tool_raw_ranges', {})
                tool_display_ranges = program_entry.setdefault('tool_display_ranges', {})
                for seg in tool_segments:
                    seg = seg.strip().replace('，', ',')
                    if not seg:
                        continue
                    if ',' in seg:
                        tool_part, range_part = seg.split(',', 1)
                    elif ':' in seg:
                        tool_part, range_part = seg.split(':', 1)
                    else:
                        seg_parts = seg.split()
                        tool_part = seg_parts[0] if seg_parts else ""
                        range_part = seg_parts[1] if len(seg_parts) > 1 else ""
                    tool_id = tool_part.strip()
                    range_part = range_part.strip()
                    if '-' not in range_part:
                        continue
                    start_str, end_str = range_part.split('-', 1)
                    try:
                        start_val = int(float(start_str))
                        end_val = int(float(end_str))
                    except Exception:
                        continue
                    # 区间行号保持原样，不做-1处理
                    tools.setdefault(tool_id, []).append((start_val, end_val))
                    tool_raw_ranges.setdefault(tool_id, []).append((start_val, end_val))
                    tool_display_ranges.setdefault(tool_id, []).append((start_val, end_val))
        return programs

    def build_tool_display_label(self, tool_id, display_ranges):
        """构造刀具下拉显示文本"""
        if not display_ranges:
            return tool_id
        if len(display_ranges) == 1:
            start_val, end_val = display_ranges[0]
            return f"{tool_id} ({start_val}-{end_val})"
        range_text = ",".join([f"{start}-{end}" for start, end in display_ranges])
        return f"{tool_id} ({range_text})"

    def format_tool_label(self, tool_id):
        """Format tool label for display without duplicating the T prefix."""
        tool_str = str(tool_id).strip()
        if not tool_str:
            return ""
        if tool_str.upper().startswith("T"):
            return tool_str
        return f"T{tool_str}"

    def update_sample_program_options(self):
        """刷新程序名/刀具号下拉选项"""
        program_names = sorted(self.sample_programs.keys())
        self.sample_program_combo["values"] = program_names
        if program_names:
            if self.sample_program_name.get() not in program_names:
                self.sample_program_name.set(program_names[0])
            self.on_sample_program_selected()
        else:
            self.sample_program_name.set("")
            self.sample_tool_name.set("")
            self.sample_tool_combo["values"] = []

    def on_sample_program_selected(self, event=None):
        """程序名切换"""
        program_name = self.sample_program_name.get().strip()
        program_info = self.sample_programs.get(program_name)
        if not program_info:
            self.sample_tool_combo["values"] = []
            self.sample_tool_name.set("")
            self.on_sample_selection_change()
            return
        tool_display_map = {}
        display_labels = []
        for tool_id, display_ranges in program_info.get('tool_display_ranges', {}).items():
            label = self.build_tool_display_label(tool_id, display_ranges)
            tool_display_map[label] = tool_id
            display_labels.append(label)
        if not display_labels:
            tool_ids = list((program_info.get("tools") or program_info.get("tool_raw_ranges") or {}).keys())
            tool_ids = sorted(set(tool_ids))
            for tool_id in tool_ids:
                label = str(tool_id)
                tool_display_map[label] = tool_id
                display_labels.append(label)
        display_labels.sort()
        program_info['tool_display_map'] = tool_display_map
        self.sample_tool_combo["values"] = display_labels
        if display_labels:
            if self.sample_tool_name.get() not in display_labels:
                self.sample_tool_name.set(display_labels[0])
        else:
            self.sample_tool_name.set("")
        self.sync_adjustment_ratio_for_current_view()
        if not getattr(self, "_loading_sample_data", False):
            if self.apply_process_file_for_program(program_name):
                if self._process_current_input_for_preview():
                    return
            else:
                self.set_input_files([])
                self.prompt_process_file_for_program(program_name)
        self.on_sample_selection_change()

    def on_sample_display_mode_change(self):
        """显示模式切换"""
        mode = self.sample_display_mode.get()
        if not self.sample_programs:
            self.sample_program_combo.configure(state="disabled")
            self.sample_tool_combo.configure(state="disabled")
            if mode == "program":
                self.sample_avg_var.set("-")
                self.sample_ideal_var.set("-")
            else:
                self.sample_avg_var.set("-")
                self.sample_ideal_var.set("-")
            if hasattr(self, 'adjustment_ratio_scale'):
                self.adjustment_ratio_scale.configure(state="normal")
            self.on_sample_selection_change()
            return
        if mode == "program":
            self.sample_program_combo.configure(state="readonly")
            self.sample_tool_combo.configure(state="disabled")
            self.sample_avg_var.set("-")
            self.sample_ideal_var.set("-")
            self.sync_adjustment_ratio_for_current_view()
        else:
            self.sample_program_combo.configure(state="readonly")
            self.sample_tool_combo.configure(state="readonly")
            self.sample_avg_var.set("-")
            self.sample_ideal_var.set("-")
            self.sync_adjustment_ratio_for_current_view()
        if hasattr(self, 'adjustment_ratio_scale'):
            self.adjustment_ratio_scale.configure(state="normal")
        self.on_sample_selection_change()

    def on_sample_selection_change(self, event=None):
        """实测数据显示条件变化时刷新图表（加入去抖，防止频繁切换卡顿）"""
        try:
            # 记录当前选择签名，快速跳过重复请求
            sig = (
                self.sample_display_mode.get(),
                self.sample_program_name.get(),
                self.sample_tool_name.get(),
                self.sample_plot_mode.get(),
                self.sample_data_source.get(),
            )
            self._pending_selection_signature = sig

            # 取消上一次计划任务
            if self._selection_change_job:
                self.root.after_cancel(self._selection_change_job)
                self._selection_change_job = None

            def _apply_change(expected_sig=sig):
                self._selection_change_job = None
                # 如果在等待期间选择已改变，则等下一次定时触发
                if expected_sig != self._pending_selection_signature:
                    return
                self._last_selection_signature = expected_sig

                if self.sample_display_mode.get() in ("program", "tool"):
                    self.sync_adjustment_ratio_for_current_view()
                if not self.data:
                    if self.sample_data_loaded:
                        self.show_sample_preview()
                    self._refresh_current_ideal_display()
                    return
                self.generate_plots(save=False, silent=True)

            # 轻量去抖：120ms 内合并多次选择变更
            self._selection_change_job = self.root.after(120, _apply_change)
        except Exception:
            # 退回旧逻辑避免阻塞
            if self.sample_display_mode.get() in ("program", "tool"):
                self.sync_adjustment_ratio_for_current_view()
            if not self.data:
                if self.sample_data_loaded:
                    self.show_sample_preview()
                self._refresh_current_ideal_display()
                return
            self.generate_plots(save=False, silent=True)

    def prompt_sample_data_source(self):
        """启动时选择实测数据源并自动加载"""
        if getattr(self, "_sample_source_prompt_shown", False):
            return
        self._sample_source_prompt_shown = True

        dialog = tk.Toplevel(self.root)
        dialog.title("选择实测数据源")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        current_value = self.sample_data_source.get()
        default_value = current_value if current_value in (0, 1, 2) else 1
        source_var = tk.IntVar(value=default_value)

        ttk.Label(dialog, text="请选择实测数据源（默认VGpro功率）:").pack(padx=12, pady=(12, 6))
        options_frame = ttk.Frame(dialog)
        options_frame.pack(padx=12, pady=6, fill=tk.X)
        ttk.Radiobutton(options_frame, text="电流", variable=source_var, value=0).pack(anchor="w")
        ttk.Radiobutton(options_frame, text="VGpro功率", variable=source_var, value=1).pack(anchor="w")
        ttk.Radiobutton(options_frame, text="边缘模块功率", variable=source_var, value=2).pack(anchor="w")

        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=(6, 12))

        def _confirm():
            self.sample_data_source.set(source_var.get())
            dialog.destroy()
            self.load_sample_data(silent=False)

        ttk.Button(button_frame, text="确定", command=_confirm).pack(side=tk.LEFT, padx=6)
        dialog.protocol("WM_DELETE_WINDOW", _confirm)
        
        # 居中显示弹框
        center_dialog_on_parent(dialog, self.root)

    def _find_file_case_insensitive(self, directory, filename):
        """在目录内按文件名大小写不敏感查找文件"""
        if not directory or not filename:
            return None
        try:
            target = filename.lower()
            for name in os.listdir(directory):
                if name.lower() == target:
                    return os.path.join(directory, name)
        except Exception:
            return None
        return None

    def resolve_sampledata_files(self, base_dir):
        """
        在目录中定位 SampleData.csv / SampleData.txt。
        兼容两种放置方式：
        1) 与工艺信息表同目录
        2) 放在同目录下的 SampleData 子目录
        """
        if not base_dir:
            return None, None, None
        candidates = [base_dir, os.path.join(base_dir, "SampleData")]
        for directory in candidates:
            if not os.path.isdir(directory):
                continue
            csv_path = self._find_file_case_insensitive(directory, "SampleData.csv")
            txt_path = self._find_file_case_insensitive(directory, "SampleData.txt")
            if csv_path and txt_path and os.path.exists(csv_path) and os.path.exists(txt_path):
                return directory, csv_path, txt_path
        return None, None, None

    def load_sample_data_from_paths(self, csv_path, txt_path, silent=False, sample_dir=None):
        """从明确路径加载实测数据 SampleData.csv 与 SampleData.txt（不依赖工艺信息表路径）"""
        if not csv_path or not txt_path:
            return False
        if not os.path.exists(csv_path) or not os.path.exists(txt_path):
            if hasattr(self, "sample_auto_status_var"):
                self.sample_auto_status_var.set("未找到SampleData.csv或SampleData.txt")
            if hasattr(self, "status_var_data"):
                self.status_var_data.set("未发现实测数据文件，已跳过导入")
            return False

        if sample_dir is None:
            sample_dir = os.path.dirname(csv_path)
        self.sample_csv_path = csv_path
        self.sample_txt_path = txt_path
        if hasattr(self, "sample_bundle_path_var"):
            self.sample_bundle_path_var.set(sample_dir)

        self._loading_sample_data = True
        try:
            if hasattr(self, "sample_auto_status_var"):
                self.sample_auto_status_var.set("正在导入实测数据...")
            if hasattr(self, "status_var_data"):
                self.status_var_data.set("正在导入实测数据...")
            try:
                self.root.update_idletasks()
            except Exception:
                pass

            self.sample_programs = self.parse_sample_program_file(txt_path)
            df = pd.read_csv(csv_path, header=None, usecols=[0, 1, 2, 3, 4], dtype={4: str})
            if df.shape[1] < 5:
                if not silent:
                    messagebox.showerror("格式错误", "SampleData.csv 列数不足")
                if hasattr(self, "sample_auto_status_var"):
                    self.sample_auto_status_var.set("SampleData.csv格式错误")
                return False

            values = df.iloc[:, 0:3].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
            line_numbers = pd.to_numeric(df.iloc[:, 3], errors='coerce').to_numpy(dtype=float)
            program_numbers = df.iloc[:, 4].astype(str).to_numpy()
            valid_mask = ~np.isnan(line_numbers)
            values = values[valid_mask]
            program_numbers = program_numbers[valid_mask]
            line_numbers = line_numbers[valid_mask].astype(int)

            self.sample_data_values_raw = values
            self.sample_data_program_numbers_raw = program_numbers
            self.sample_data_line_numbers_raw = line_numbers
            self.sample_data_values = values
            self.sample_data_program_numbers = program_numbers
            self.sample_data_line_numbers = line_numbers
            self.sample_data_loaded = True
            self.sample_data_dir = sample_dir

            self.align_sample_data_to_processed()
            self.update_sample_program_options()
            self.on_sample_display_mode_change()
            self._refresh_ideal_tree()
            self._refresh_current_ideal_display()

            if self.raw_to_aligned_line_map:
                if hasattr(self, "status_var_data"):
                    self.status_var_data.set(f"实测数据已加载并对齐: {os.path.basename(csv_path)}")
                if hasattr(self, "sample_auto_status_var"):
                    self.sample_auto_status_var.set("实测数据已导入并对齐")
            else:
                if hasattr(self, "status_var_data"):
                    self.status_var_data.set(f"实测数据已加载，待处理后对齐: {os.path.basename(csv_path)}")
                if hasattr(self, "sample_auto_status_var"):
                    self.sample_auto_status_var.set("实测数据已导入，待对齐")

            if not self.data:
                self.show_sample_preview()
            return True
        except Exception as e:
            if not silent:
                messagebox.showerror("加载失败", f"读取实测数据时发生错误:\n{str(e)}")
            if hasattr(self, "sample_auto_status_var"):
                self.sample_auto_status_var.set("实测数据导入失败")
            if hasattr(self, "status_var_data"):
                self.status_var_data.set("实测数据导入失败")
            return False
        finally:
            self._loading_sample_data = False

    def resolve_process_file_for_program(self, directory, program_name):
        """按程序名匹配工艺信息表：<程序名>_工艺信息.txt"""
        if not directory or not program_name:
            return None
        program_name = str(program_name).strip()
        if not program_name:
            return None
        candidate = self._find_file_case_insensitive(directory, f"{program_name}_工艺信息.txt")
        if candidate and os.path.exists(candidate):
            return candidate
        return None

    def should_prompt_process_file(self, program_name):
        """判断是否需要提示选择工艺信息表"""
        if not program_name:
            return False
        if self.program_process_file_map.get(program_name):
            return False
        return not self.program_prompt_skip.get(program_name, False)

    def prompt_process_file_for_program(self, program_name):
        """提示为指定程序选择工艺信息表（可设置不再提醒）"""
        if not self.should_prompt_process_file(program_name):
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("未绑定工艺信息表")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text=f"程序 {program_name} 未绑定工艺信息文件。", font=UI_FONT_NORMAL).pack(padx=12, pady=(12, 4))
        ttk.Label(dialog, text="是否现在选择该程序对应的工艺信息表？", font=UI_FONT_NORMAL).pack(padx=12, pady=(0, 8))

        skip_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(dialog, text="不再提醒该程序", variable=skip_var).pack(padx=12, pady=(0, 6), anchor="w")

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=(0, 12))

        def _apply_skip():
            if skip_var.get():
                self.program_prompt_skip[program_name] = True
                self._persist_app_config()

        def _choose():
            _apply_skip()
            dialog.destroy()
            self.root.after(0, lambda: self.choose_process_file_for_program(program_name))

        def _skip():
            _apply_skip()
            dialog.destroy()

        ttk.Button(btn_frame, text="选择文件", command=_choose, width=12).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="跳过", command=_skip, width=10).pack(side=tk.LEFT, padx=6)
        
        # 居中显示弹框
        center_dialog_on_parent(dialog, self.root)

    def choose_process_file_for_current_program(self):
        """为当前程序选择工艺信息表"""
        program_name = self.sample_program_name.get().strip()
        if not program_name:
            self.set_status("请先选择程序名", 3000)
            return
        self.choose_process_file_for_program(program_name)

    def validate_process_info_file(self, file_path):
        """
        验证工艺信息表文件格式是否正确。
        返回: (is_valid, error_message)
        """
        if not file_path or not os.path.exists(file_path):
            return False, "文件不存在"
        
        try:
            # 尝试多种编码读取文件
            content = None
            for encoding in ['utf-8', 'gbk', 'gb2312', 'latin-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.readlines()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                return False, "无法读取文件，请检查文件编码"
            
            if len(content) < 3:
                return False, "文件行数过少，请检查文件内容"
            
            # 检查前几行是否符合工艺信息表格式
            valid_lines = 0
            
            for line in content[:min(100, len(content))]:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) < 6:
                    continue

                numeric_tokens, gcode_tokens = self._split_numeric_and_gcode_tokens(parts)
                if not numeric_tokens or not gcode_tokens:
                    continue

                if len(numeric_tokens) < 3:
                    continue
                try:
                    float(numeric_tokens[-1])
                    float(numeric_tokens[-2])
                    float(numeric_tokens[-3])
                except Exception:
                    continue

                # 检查是否包含N行号和G代码
                gcode_part = ' '.join(gcode_tokens)
                if 'N' in gcode_part and ('G' in gcode_part or 'M' in gcode_part or 'X' in gcode_part or 'Y' in gcode_part or 'Z' in gcode_part):
                    valid_lines += 1
            
            if valid_lines < 3:
                return False, "文件格式不正确：未检测到有效的工艺信息行（需包含ap/ae/F及G代码等）"
            
            return True, ""
            
        except Exception as e:
            return False, f"验证文件时发生错误: {str(e)}"

    def choose_process_file_for_program(self, program_name):
        """为指定程序选择并绑定工艺信息表"""
        program_name = (program_name or "").strip()
        if not program_name:
            return False
        file_path = filedialog.askopenfilename(
            title=f"选择 {program_name} 的工艺信息表",
            filetypes=(("文本文件", "*.txt"), ("Excel文件", "*.xlsx"), ("CSV文件", "*.csv"), ("所有文件", "*.*"))
        )
        if not file_path:
            return False
        
        # 验证文件格式
        is_valid, error_msg = self.validate_process_info_file(file_path)
        if not is_valid:
            messagebox.showerror("文件格式错误", f"所选文件不是有效的工艺信息表:\n{error_msg}")
            return False
        
        self.program_process_file_map[program_name] = file_path
        self._persist_app_config()
        self.set_status(f"已绑定工艺信息表: {os.path.basename(file_path)}", 4000)
        # 启用保存区间信息按钮
        if hasattr(self, 'export_i_code_btn'):
            self.export_i_code_btn.configure(state="normal")
        # 为当前程序的所有刀具设置默认优化倍率2.0
        self._set_default_rg_for_program(program_name)
        if self.apply_process_file_for_program(program_name):
            self._process_current_input_for_preview()
        return True

    def apply_process_file_for_program(self, program_name):
        """根据已绑定路径切换当前工艺信息表"""
        program_name = (program_name or "").strip()
        if not program_name:
            return False
        process_path = self.program_process_file_map.get(program_name)
        if not process_path:
            if hasattr(self, "matched_process_file_var"):
                self.matched_process_file_var.set(f"未绑定工艺信息表: {program_name}")
            return False
        if not os.path.exists(process_path):
            if hasattr(self, "matched_process_file_var"):
                self.matched_process_file_var.set(f"工艺信息表不存在: {process_path}")
            return False
        current_primary = self.get_primary_input_file()
        if os.path.normcase(os.path.abspath(current_primary)) != os.path.normcase(os.path.abspath(process_path)):
            self.set_input_files([process_path])
        if hasattr(self, "matched_process_file_var"):
            self.matched_process_file_var.set(process_path)
        return True

    def sync_process_file_for_program(self, program_name, silent=False):
        """根据程序名切换当前工艺信息表"""
        program_name = (program_name or "").strip()
        if not program_name:
            return False
        process_path = self.program_process_file_map.get(program_name)
        if not process_path:
            self.set_input_files([])
            if hasattr(self, "matched_process_file_var"):
                self.matched_process_file_var.set(f"未绑定工艺信息表: {program_name}")
            return False
        if not os.path.exists(process_path):
            self.set_input_files([])
            if hasattr(self, "matched_process_file_var"):
                self.matched_process_file_var.set(f"工艺信息表不存在: {process_path}")
            return False
        current_primary = self.get_primary_input_file()
        if os.path.normcase(os.path.abspath(current_primary)) == os.path.normcase(os.path.abspath(process_path)):
            if hasattr(self, "matched_process_file_var"):
                self.matched_process_file_var.set(process_path)
            return True
        self.set_input_files([process_path])
        if hasattr(self, "matched_process_file_var"):
            self.matched_process_file_var.set(process_path)
        return True

    def load_sample_data(self, silent=False):
        """加载实测数据 SampleData.csv 与 SampleData.txt"""
        input_files = self.get_input_files()
        if len(input_files) > 1:
            if not silent:
                messagebox.showwarning("多文件模式", "多文件模式下请先选择单个工艺信息表，再读取实测数据")
            if hasattr(self, "sample_auto_status_var"):
                self.sample_auto_status_var.set("多文件模式：停用实测自动导入")
            return False
        sample_dir = None
        primary_file = input_files[0] if input_files else ""
        if primary_file:
            sample_dir = os.path.dirname(primary_file)
        elif self.processed_file_path:
            sample_dir = os.path.dirname(self.processed_file_path)
        else:
            sample_dir = base_dir
        if not sample_dir:
            if not silent:
                messagebox.showwarning("路径缺失", "请先导入 SampleData")
            if hasattr(self, "sample_auto_status_var"):
                self.sample_auto_status_var.set("未找到工艺信息表目录")
            return False
        resolved_dir, csv_path, txt_path = self.resolve_sampledata_files(sample_dir)
        if not resolved_dir or not csv_path or not txt_path:
            if not silent:
                messagebox.showerror(
                    "文件缺失",
                    "未找到 SampleData.csv 或 SampleData.txt（可放在工艺信息表同目录，或同目录下的 SampleData 子目录）"
                )
            if hasattr(self, "sample_auto_status_var"):
                self.sample_auto_status_var.set("未找到SampleData.csv或SampleData.txt")
            if hasattr(self, "status_var_data"):
                self.status_var_data.set("未发现实测数据文件，已跳过导入")
            return False
        return self.load_sample_data_from_paths(csv_path, txt_path, silent=silent, sample_dir=resolved_dir)

    def apply_tool_background(self, ax, program_name, tool_id=None):
        """绘制当前程序内刀具背景区间 - 工业柔和配色"""
        # 已按需求禁用刀具号背景色带
        return
        if not self.sample_programs or not program_name:
            return
        program_info = self.sample_programs.get(program_name, {})
        tools = program_info.get("tools", {})
        if not tools:
            return
        # 使用全局定义的工业配色
        tool_ids = [tool_id] if tool_id else sorted(tools.keys())
        for idx, t_id in enumerate(tool_ids):
            for start, end in tools.get(t_id, []):
                ax.axvspan(start, end + 1, facecolor=TOOL_BG_COLORS[idx % len(TOOL_BG_COLORS)],
                           alpha=0.5, zorder=1, linewidth=0)

    def collect_intervals_for_tool(self, program_name, tool_id):
        """获取指定程序+刀具的稳态区间"""
        program_info = self.sample_programs.get(program_name, {})
        ranges = program_info.get("tools", {}).get(tool_id, [])
        return self.get_predicted_intervals_for_display(ranges)

    def export_sample_intervals(self):
        """导出负载区间交互文件 SampleData.rg"""
        if not self.pred_power_intervals:
            messagebox.showwarning("无区间", "请先生成稳态区间")
            return
        if not self.sample_programs:
            messagebox.showwarning("无程序信息", "请先加载 SampleData.txt")
            return
        primary_file = self.get_primary_input_file()
        # 强制保存到exe所在目录（避免散落在工作区）
        output_dir = app_dir
        if not output_dir:
            messagebox.showwarning("路径缺失", "无法确定导出目录")
            return
        output_path = os.path.join(output_dir, "SampleData.rg")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"{self.sample_data_source.get()}\n")
                for program_name in sorted(self.sample_programs.keys()):
                    program_info = self.sample_programs[program_name]
                    for tool_id in sorted(program_info.get("tools", {}).keys()):
                        intervals = self.collect_intervals_for_tool(program_name, tool_id)
                        interval_text = ",".join([f"{start}-{end}" for start, end in intervals])
                        f.write(f"{program_name};{tool_id};{interval_text};\n")
            self.status_var_data.set(f"区间文件已导出: {os.path.basename(output_path)}")
        except Exception as e:
            messagebox.showerror("导出失败", f"导出区间文件时发生错误:\n{str(e)}")
    
    def _set_default_rg_for_program(self, program_name):
        """为指定程序的所有刀具设置默认优化倍率2.0"""
        if not program_name:
            return
        program_info = self.sample_programs.get(program_name, {})
        tools = program_info.get("tools", {})
        default_rg = 2.0
        
        for tool_id in tools.keys():
            key = (program_name, tool_id)
            # 无论是否存在，都强制设置为默认值2.0
            self.ideal_store[key] = {
                "rg": default_rg,
                "updated_at": datetime.now().isoformat()
            }
        
        # 重置当前显示的优化倍率为2.0
        self._ratio_update_lock = True
        try:
            self.adjustment_ratio.set(default_rg)
            self.adjustment_ratio_display.set(f"{default_rg:.2f}")
        finally:
            self._ratio_update_lock = False
        
        self._persist_ideal_store()
        self._refresh_ideal_tree()
        self._refresh_current_ideal_display()
    
    def save_interval_info(self):
        """保存区间信息按钮的处理函数"""
        if not self.sample_programs:
            messagebox.showwarning("无程序信息", "请先加载 SampleData.txt")
            return
        if not self.pred_power_intervals:
            messagebox.showwarning("无稳态区间", "请先生成稳态区间")
            return
        
        # 收集所有可导出的刀具
        exportable_tools = []
        for program_name, program_info in self.sample_programs.items():
            tools = program_info.get("tools", {})
            for tool_id in tools.keys():
                key = (program_name, tool_id)
                store = self.ideal_store.get(key)
                if store:
                    mean_val, count, _ = self.compute_tool_measured_mean(program_name, tool_id)
                    if mean_val is not None and count > 0:
                        exportable_tools.append((program_name, tool_id, store.get("rg", 2.0)))
        
        if not exportable_tools:
            messagebox.showwarning("无可导出数据", "未找到可导出的刀具区间信息")
            return
        
        # 如果只有1把刀，直接保存
        if len(exportable_tools) == 1:
            self._do_save_interval_info(exportable_tools)
            return
        
        # 多把刀具，弹出选择对话框
        self._show_tool_selection_dialog(exportable_tools)
    
    def _show_tool_selection_dialog(self, exportable_tools):
        """显示刀具选择对话框"""
        dialog = tk.Toplevel(self.root)
        dialog.title("选择要保存的刀具")
        dialog.geometry("480x520")
        dialog.minsize(420, 420)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 居中显示
        center_dialog_on_parent(dialog, self.root)
        
        # 提示信息
        ttk.Label(dialog, text="请选择要保存区间信息的刀具：", 
                  font=UI_FONT_BOLD).pack(pady=10)
        
        # 创建带滚动条的列表框
        list_frame = ttk.Frame(dialog)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 使用Checkbutton列表
        canvas = tk.Canvas(list_frame, yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=canvas.yview)
        
        inner_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner_frame, anchor="nw")
        
        # 勾选状态字典
        check_vars = {}
        
        for prog, tool_id, rg in exportable_tools:
            key = (prog, tool_id)
            # 默认选中，但若工艺文件未导入/未处理则不选中
            var = tk.BooleanVar(value=True)
            
            process_path = self.program_process_file_map.get(prog)
            has_process_file = bool(process_path and os.path.exists(process_path))
            has_processed = self._has_processed_result_for(process_path)

            tool_label = self.format_tool_label(tool_id)
            if not has_process_file or not has_processed:
                ideal_text = "待导入工艺信息文件"
                var.set(False)
            else:
                mean_val, _, _ = self.compute_tool_measured_mean(prog, tool_id)
                ideal_val = mean_val * rg if mean_val is not None else None
                if ideal_val is not None:
                    ideal_text = f"理想值={ideal_val:.2f}"
                else:
                    ideal_text = "未计算"

            check_vars[key] = var
            cb = tk.Checkbutton(inner_frame, 
                                text=f"{prog} - {tool_label}  (override={rg:.2f}, {ideal_text})",
                                variable=var,
                                onvalue=True, offvalue=False,
                                font=UI_FONT_NORMAL,
                                anchor="w")
            cb.pack(anchor="w", pady=2)
        
        # 更新滚动区域
        inner_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))
        
        # 按钮区
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10, fill=tk.X)
        
        def select_all():
            for var in check_vars.values():
                var.set(True)
        
        def select_none():
            for var in check_vars.values():
                var.set(False)
        
        def do_save():
            selected = [(prog, tool, rg) for (prog, tool), var in check_vars.items() 
                        if var.get() 
                        for p, t, rg in exportable_tools if p == prog and t == tool]
            if not selected:
                self.set_status("未选择任何刀具", 3000)
                dialog.destroy()
                return
            dialog.destroy()
            self._do_save_interval_info(selected)
        
        ttk.Button(btn_frame, text="全选", width=10, command=select_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="全不选", width=10, command=select_none).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="保存", width=12, command=do_save).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", width=10, command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def _do_save_interval_info(self, tools_to_save):
        """实际执行保存区间信息"""
        output_dir = app_dir
        if not output_dir:
            messagebox.showwarning("路径缺失", "无法确定导出目录")
            return
        
        output_path = os.path.join(output_dir, "SampleData.rg")
        try:
            saved_lines = 0
            source_idx = int(self.sample_data_source.get())
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"{source_idx}\n")
                for program_name, tool_id, rg in tools_to_save:
                    mean_val, count, _ = self.compute_tool_measured_mean(program_name, tool_id)
                    if mean_val is None or count == 0:
                        continue
                    ideal_val = mean_val * rg
                    intervals = self.collect_line_point_intervals_for_tool(program_name, tool_id)
                    if not intervals:
                        continue
                    interval_text = ",".join(intervals)
                    f.write(f"{program_name};{ideal_val:.6f};{interval_text};\n")
                    saved_lines += 1
            
            # 文件已保存，立即弹出提示
            if saved_lines == 0:
                messagebox.showwarning("无可导出数据", "未找到有效的刀具区间信息")
            else:
                self.status_var_data.set(f"区间信息已保存: {os.path.basename(output_path)} (共{saved_lines}行)")
                
                # 需求7：保存ProcessDataPath.txt文件，只记录当前保存的工具对应的工艺信息文件路径
                saved_programs = set(prog for prog, _, _ in tools_to_save)
                self._save_process_data_paths(output_dir, saved_programs)
                
                messagebox.showinfo("保存成功", "结果已保存，请关闭该窗口")
        except Exception as e:
            messagebox.showerror("导出失败", f"保存区间信息时发生错误:\n{str(e)}")

    def _save_process_data_paths(self, output_dir, saved_programs=None):
        """需求7：保存工艺信息文件路径到ProcessDataPath.txt
        
        :param output_dir: 输出目录
        :param saved_programs: 待保存的程序名集合，如果为None则保存所有
        """
        try:
            path_file = os.path.join(output_dir, "ProcessDataPath.txt")
            # 只收集当前保存的程序对应的工艺信息文件路径
            process_paths = []
            for program_name, process_path in self.program_process_file_map.items():
                # 如果指定了程序集合，只保存该集合中的程序路径
                if saved_programs is not None and program_name not in saved_programs:
                    continue
                if process_path and os.path.exists(process_path):
                    # 使用绝对路径
                    abs_path = os.path.abspath(process_path)
                    if abs_path not in process_paths:
                        process_paths.append(abs_path)
            
            if process_paths:
                with open(path_file, 'w', encoding='utf-8') as f:
                    for path in process_paths:
                        f.write(f"{path}\n")
                print(f"[INFO] 工艺信息文件路径已保存: {path_file} (共{len(process_paths)}个文件)")
        except Exception as e:
            print(f"[WARNING] 保存ProcessDataPath.txt失败: {e}")

    def export_i_code(self):
        """保存结果文件 SampleData.rg"""
        if not self.ideal_store:
            messagebox.showwarning("未保存优化倍率", "请先设定优化倍率或批量生成理想值")
            return
        if not self.sample_programs:
            messagebox.showwarning("无程序信息", "请先加载 SampleData.txt")
            return
        if not self.pred_power_intervals:
            messagebox.showwarning("无稳态区间", "请先生成稳态区间")
            return

        # 强制保存到exe所在目录（避免散落在工作区）
        output_dir = app_dir
        if not output_dir:
            messagebox.showwarning("路径缺失", "无法确定导出目录")
            return

        output_path = os.path.join(output_dir, "SampleData.rg")
        try:
            saved_lines = 0
            source_idx = int(self.sample_data_source.get())
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"{source_idx}\n")
                for program_name, program_info in self.sample_programs.items():
                    tools = program_info.get("tools", {})
                    for tool_id in tools.keys():
                        key = (program_name, tool_id)
                        store = self.ideal_store.get(key)
                        if not store:
                            continue
                        try:
                            rg = float(store.get("rg"))
                        except Exception:
                            continue
                        mean_val, count, _ = self.compute_tool_measured_mean(program_name, tool_id)
                        if mean_val is None or count == 0:
                            continue
                        ideal_val = mean_val * rg
                        if ideal_val is None:
                            continue
                        intervals = self.collect_line_point_intervals_for_tool(program_name, tool_id)
                        if not intervals:
                            continue
                        interval_text = ",".join(intervals)
                        f.write(f"{program_name};{ideal_val:.6f};{interval_text};\n")
                        saved_lines += 1
            if saved_lines == 0:
                messagebox.showwarning("无可导出数据", "未找到已保存优化倍率的刀具区间")
                return
            self.status_var_data.set(f"结果已保存: {os.path.basename(output_path)} (共{saved_lines}行)")
            messagebox.showinfo("保存完成", f"结果已保存:\n{output_path}\n共 {saved_lines} 行")
        except Exception as e:
            messagebox.showerror("导出失败", f"保存结果时发生错误:\n{str(e)}")
    
    def calculate_additional_columns(self, ap, ae, feed_rate, s, current_s, s_base, k_base):
        """按三参数功率模型计算新增列。"""
        try:
            ap_val = float(ap)
            ae_val = float(ae)
            feed_val = float(feed_rate)
            s_val = float(s)

            if feed_val > 0:
                t_val = s_val / (feed_val / 60.0)
            else:
                t_val = 0.0

            dmrv_val = ap_val * ae_val
            mrr_val = dmrv_val * (feed_val / 60.0)

            effective_speed = float(current_s) if current_s and float(current_s) > 0 else float(self.current_program_speed.get() or s_base)
            p_idle = self.predict_idle_power(effective_speed)
            kc_value = float(self.kc_coeff.get())
            ke_value = float(self.ke_coeff.get())
            cutting_power = kc_value * mrr_val
            edge_power = ke_value * ap_val
            p_power = p_idle + cutting_power + edge_power

            angular_velocity = 2 * math.pi * current_s / 60.0
            if angular_velocity > 1e-9:
                t_torque = p_power / angular_velocity
            else:
                t_torque = 0.0

            return t_val, dmrv_val, mrr_val, kc_value, t_torque, p_power, p_idle, edge_power

        except ValueError:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    def generate_plots(self, save=False, silent=False, debug_line_range=None):
        """生成图表
        :param debug_line_range: 可选，(start_line, end_line) 用于调试特定行号范围内的划分逻辑
        """
        if not self.data:
            messagebox.showwarning("无数据", "请先处理数据以生成图表")
            return False
        
        try:
            s_values = [d['s'] for d in self.data]
            P_values = [d['P'] for d in self.data]
            n_values = [d['N_str'] for d in self.data]
            line_numbers = [d.get('line_no_aligned', idx) for idx, d in enumerate(self.data)]
            line_numbers = [int(x) if x is not None else idx for idx, x in enumerate(line_numbers)]
            cumulative_s = np.cumsum(s_values)
            
            self.figures = []
            self.pred_power_intervals = []
            self.pit_records = []
            self.refresh_pit_button_state()
            if self.enable_pred_power_steady.get():
                self.pred_power_intervals = self.partition_pred_power_steady_intervals(
                    P_values, s_values, cumulative_s, n_values, line_numbers, debug_line_range
                )
            
            # 需求6：最大区间数500限制，对原始区间按sample_count从小到大删减
            MAX_INTERVALS = 500
            if len(self.pred_power_intervals) > MAX_INTERVALS:
                original_count = len(self.pred_power_intervals)
                # 按sample_count（SampleData点数）降序排序，保留点数最多的区间
                intervals_sorted = sorted(self.pred_power_intervals, key=lambda x: x.get('sample_count', 0), reverse=True)
                # 保留前500个
                self.pred_power_intervals = intervals_sorted[:MAX_INTERVALS]
                # 恢复原始顺序（按start_line排序）
                self.pred_power_intervals = sorted(self.pred_power_intervals, key=lambda x: x.get('start_line', 0))
                self.set_status(f"区间数超过{MAX_INTERVALS}，已删减{original_count - MAX_INTERVALS}个点数较少的区间", 5000)

            self.pit_records = [dict(interval) for interval in self.pred_power_intervals]
            self.refresh_pit_button_state()
            
            line_arr_all = np.asarray(line_numbers, dtype=int)
            process_base_blocks = self.compute_sequence_blocks(line_arr_all)
            x_pred_all = np.asarray(self.compute_line_x_positions(line_arr_all, blocks=process_base_blocks), dtype=float)
            p_arr_all = np.asarray(P_values, dtype=float)
            
            display_mode = self.sample_display_mode.get()
            program_name = self.get_current_program_key()
            process_path = self.program_process_file_map.get(program_name) if program_name else None
            has_process_file = bool(process_path and os.path.exists(process_path))
            has_processed = self._has_processed_result_for(process_path)
            tool_id = self.get_selected_tool_id() if display_mode == "tool" else None
            if display_mode == "tool":
                tool_ranges = self.get_selected_tool_ranges()
            else:
                tool_ranges = self.get_program_ranges(program_name)
            
            intervals_for_display = self.get_predicted_intervals_for_display(tool_ranges)
            
            adjustment_ratio = self.get_adjustment_ratio_for_view()
            program_info = self.get_program_info(program_name) if program_name else {}
            tool_ids = sorted(program_info.get("tools", {}).keys()) if program_info else []
            # 使用全局定义的工业配色方案
            tool_line_colors = INTERVAL_COLORS
            tool_color_map = {tid: tool_line_colors[idx % len(tool_line_colors)]
                              for idx, tid in enumerate(tool_ids)}
            if tool_id and tool_id not in tool_color_map:
                tool_color_map[tool_id] = tool_line_colors[0]
            tool_mean_cache = {}
            tool_count_cache = {}
            tool_ideal_saved_map = {}

            def _get_saved_rg_for_tool(tid):
                if not program_name or not tid:
                    return None
                store = self.ideal_store.get((program_name, tid))
                if not store:
                    return None
                try:
                    return float(store.get("rg"))
                except Exception:
                    return None

            def _get_tool_stats(tid):
                if tid in tool_mean_cache:
                    return tool_mean_cache[tid], tool_count_cache[tid], tool_ideal_saved_map[tid]
                mean_val, count, _ = self.compute_tool_measured_mean(program_name, tid)
                tool_mean_cache[tid] = mean_val
                tool_count_cache[tid] = count
                rg_saved = _get_saved_rg_for_tool(tid)
                if mean_val is not None and count > 0 and rg_saved is not None:
                    ideal_val = mean_val * rg_saved
                else:
                    ideal_val = None
                tool_ideal_saved_map[tid] = ideal_val
                return mean_val, count, ideal_val

            tool_range_entries = []
            if program_info:
                for t_id, ranges in program_info.get("tools", {}).items():
                    for r_start, r_end in ranges:
                        tool_range_entries.append((r_start, r_end, t_id))
                tool_range_entries.sort(key=lambda x: (x[0], x[1]))

            def _find_tool_for_interval(start_line, end_line):
                for r_start, r_end, t_id in tool_range_entries:
                    if r_start <= start_line <= r_end:
                        return t_id
                for r_start, r_end, t_id in tool_range_entries:
                    if not (end_line < r_start or start_line > r_end):
                        return t_id
                return None
            
            pred_mask = self.build_process_mask(line_arr_all, tool_ranges)
            pred_blocks = self.compute_contiguous_blocks(pred_mask)
            pred_invalid_blocks = self.compute_contiguous_blocks((~pred_mask) if pred_mask is not None else None)
            self.process_valid_mask = pred_mask
            self.process_valid_blocks = pred_blocks
            axis_line_numbers_pred = line_arr_all[pred_mask] if pred_mask is not None and pred_mask.any() else line_arr_all

            sample_values_all = None
            sample_line_numbers_all = None
            sample_time_indices_all = None
            sample_x_positions_all = None
            sample_context_mask = None
            sample_valid_mask = None
            sample_context_blocks = []
            sample_valid_blocks = []
            sample_invalid_blocks = []
            if self.sample_data_loaded and self.sample_data_values is not None:
                source_idx = int(self.sample_data_source.get())
                sample_values_all = np.asarray(self.sample_data_values[:, source_idx])
                sample_line_numbers_all = np.asarray(self.sample_data_line_numbers)
                sample_time_indices_all = self.get_sample_time_indices_array()
                sample_x_positions_all = None if self.sample_data_x_positions is None else np.asarray(self.sample_data_x_positions, dtype=float)
                if sample_x_positions_all is None and sample_line_numbers_all is not None:
                    sample_x_positions_all = np.asarray(
                        self.compute_line_x_positions(sample_line_numbers_all, blocks=self.sample_data_base_blocks),
                        dtype=float
                    )

                program_no = self.get_selected_program_number()
                sample_context_mask = self.build_sample_mask(program_no, None)
                sample_valid_mask = self.build_sample_mask(program_no, tool_ranges or None)
                if sample_context_mask is not None:
                    finite_mask = np.isfinite(sample_values_all)
                    sample_context_mask &= finite_mask
                    if sample_valid_mask is not None:
                        sample_valid_mask &= finite_mask
                    sample_context_blocks = self.compute_contiguous_blocks(sample_context_mask)
                    sample_valid_blocks = self.compute_contiguous_blocks(sample_valid_mask) if sample_valid_mask is not None else []
                    if sample_valid_mask is not None:
                        sample_invalid_blocks = self.compute_contiguous_blocks(sample_context_mask & (~sample_valid_mask))
                    self.sample_data_valid_mask = sample_valid_mask
                    self.sample_data_valid_blocks = sample_valid_blocks

            has_sample_context = bool(sample_context_mask is not None and sample_context_mask.any())
            has_sample_valid = bool(sample_valid_mask is not None and sample_valid_mask.any())
            use_time_display = bool(has_sample_context and sample_time_indices_all is not None)
            sample_display_x_all = sample_time_indices_all if use_time_display else sample_x_positions_all

            tool_mean_val = None
            tool_count = 0
            tool_ideal_val_saved = None
            tool_ideal_val_preview = None
            tool_saved = False
            if display_mode == "tool" and program_name and tool_id:
                tool_mean_val, tool_count, tool_ideal_val_saved = _get_tool_stats(tool_id)
                if tool_mean_val is not None and tool_count > 0:
                    tool_ideal_val_preview = tool_mean_val * adjustment_ratio
                tool_saved = tool_ideal_val_saved is not None

            # 注意：不再根据SampleData行号范围过滤预测负载
            # 两者使用独立的行号体系，只是叠加显示在同一张图上
            
            interval_meta = []
            # 使用原始区间数据，包含精确的start_idx和end_idx用于绘制精确边界
            for idx, interval in enumerate(self.pred_power_intervals):
                start_line = interval.get('start_line')
                end_line = interval.get('end_line')
                start_idx_iv = interval.get('start_idx')
                end_idx_iv = interval.get('end_idx')
                if start_line is None or end_line is None:
                    continue
                # 检查区间是否完全在显示范围内（而非仅有交集）
                if tool_ranges:
                    in_range = False
                    for r_start, r_end in tool_ranges:
                        if start_line >= r_start and end_line <= r_end:
                            in_range = True
                            break
                    if not in_range:
                        continue
                if start_idx_iv is None or end_idx_iv is None:
                    continue
                if start_idx_iv < 0 or end_idx_iv >= len(p_arr_all) or start_idx_iv > end_idx_iv:
                    continue
                p_mean = float(np.mean(p_arr_all[start_idx_iv:end_idx_iv + 1]))
                p_pref = p_mean * adjustment_ratio
                ideal_val = None
                if display_mode == "tool":
                    ideal_val = tool_ideal_val_saved
                else:
                    t_id = _find_tool_for_interval(start_line, end_line)
                    if t_id:
                        _, _, ideal_val = _get_tool_stats(t_id)
                # 保存精确的idx用于计算x坐标边界
                interval_meta.append((start_line, end_line, p_pref, ideal_val, idx, start_idx_iv, end_idx_iv))

            # 使用全局定义的科技配色
            interval_colors = INTERVAL_COLORS[:2]  # 取前两种颜色交替使用
            
            # 预测负载 + 实测负载图
            plot_mode = self.sample_plot_mode.get()
            if plot_mode == "stacked":
                fig2, (ax2, ax3) = plt.subplots(2, 1, sharex=True, figsize=(16, 9), dpi=100)
                axes_pred = ax2
                axes_act = ax3
                overlay_secondary = False
            else:
                fig2, ax2 = plt.subplots(figsize=(16, 9), dpi=100)
                axes_pred = ax2
                axes_act = ax2.twinx()
                overlay_secondary = True
            fig2.patch.set_facecolor(PLOT_FIG_BG)
            self.apply_plot_style(axes_pred, grid=True)
            if overlay_secondary:
                axes_act.set_facecolor('none')
                axes_act.patch.set_visible(False)
                axes_act.grid(False)
                axes_act.set_zorder(axes_pred.get_zorder() + 1)
                self.apply_plot_style(axes_act, grid=False, text_color=STYLE_MEASURED["color"], transparent=True)
                axes_act.spines['right'].set_color(STYLE_MEASURED["color"])
                axes_act.tick_params(labelsize=PLOT_FONT_BASE, colors=STYLE_MEASURED["color"])
            
            self._interval_background_artists = []
            self.apply_tool_background(axes_pred, program_name, None if display_mode == "program" else tool_id)
            
            # 预计算工艺信息数据的区间边界（每行是一段区间而非一个点）
            all_line_numbers_for_bounds = [d.get('line_no_aligned', i) for i, d in enumerate(self.data)]
            process_line_start_bounds, process_line_end_bounds = self.compute_line_segment_bounds(
                all_line_numbers_for_bounds,
                blocks=process_base_blocks
            )
            process_display_start_bounds = np.asarray(process_line_start_bounds, dtype=float)
            process_display_end_bounds = np.asarray(process_line_end_bounds, dtype=float)
            if use_time_display:
                process_time_start_bounds, process_time_end_bounds = self.compute_process_time_bounds_from_sample(
                    line_arr_all,
                    sample_context_mask
                )
                process_time_start_bounds = np.asarray(process_time_start_bounds, dtype=float)
                process_time_end_bounds = np.asarray(process_time_end_bounds, dtype=float)
                finite_time_mask = np.isfinite(process_time_start_bounds) & np.isfinite(process_time_end_bounds)
                if finite_time_mask.any():
                    process_display_start_bounds = process_time_start_bounds
                    process_display_end_bounds = process_time_end_bounds
                else:
                    use_time_display = False
                    sample_display_x_all = sample_x_positions_all

            def _plot_process_step_blocks(ax, blocks, start_bounds, end_bounds, color, linewidth, alpha, label=None, zorder=5):
                first = True
                for block_start, block_end in blocks:
                    segment_x = []
                    segment_y = []
                    last_end_x = None
                    for global_idx in range(block_start, block_end + 1):
                        start_x = float(start_bounds[global_idx])
                        end_x = float(end_bounds[global_idx])
                        if not np.isfinite(start_x) or not np.isfinite(end_x) or end_x <= start_x:
                            if segment_x:
                                plot_kwargs = {
                                    "color": color,
                                    "linewidth": linewidth,
                                    "alpha": alpha,
                                    "zorder": zorder,
                                }
                                if first and label:
                                    plot_kwargs["label"] = label
                                ax.plot(segment_x, segment_y, **plot_kwargs)
                                first = False
                                segment_x = []
                                segment_y = []
                            last_end_x = None
                            continue
                        if last_end_x is not None and start_x > last_end_x + 1e-9 and segment_x:
                            plot_kwargs = {
                                "color": color,
                                "linewidth": linewidth,
                                "alpha": alpha,
                                "zorder": zorder,
                            }
                            if first and label:
                                plot_kwargs["label"] = label
                            ax.plot(segment_x, segment_y, **plot_kwargs)
                            first = False
                            segment_x = []
                            segment_y = []
                        p_val = p_arr_all[global_idx]
                        segment_x.extend([start_x, end_x])
                        segment_y.extend([p_val, p_val])
                        last_end_x = end_x
                    if segment_x:
                        plot_kwargs = {
                            "color": color,
                            "linewidth": linewidth,
                            "alpha": alpha,
                            "zorder": zorder,
                        }
                        if first and label:
                            plot_kwargs["label"] = label
                        ax.plot(segment_x, segment_y, **plot_kwargs)
                        first = False
            
            for item in interval_meta:
                start_line, end_line, p_pref, ideal_val, idx, start_idx_iv, end_idx_iv = item
                if plot_mode == "overlay" and ideal_val is not None:
                    bar_val = ideal_val
                else:
                    bar_val = p_pref
                if bar_val is None:
                    continue
                y0, y1 = (0.0, bar_val) if bar_val >= 0 else (bar_val, 0.0)
                # 使用区间边界：起始行的start_x到结束行的end_x
                if start_idx_iv is not None and end_idx_iv is not None and len(process_display_start_bounds) > 0:
                    start_x = process_display_start_bounds[start_idx_iv]
                    end_x = process_display_end_bounds[end_idx_iv]
                else:
                    # 回退到原来的逻辑
                    start_x = start_line
                    end_x = end_line + 1
                if not np.isfinite(start_x) or not np.isfinite(end_x) or end_x <= start_x:
                    continue
                bg_artist = axes_pred.fill_between(
                    [start_x, end_x], [y0, y0], [y1, y1],
                    alpha=0.25, facecolor=interval_colors[idx % 2],
                    edgecolor=interval_colors[idx % 2], linewidth=0.35, zorder=2
                )
                self._interval_background_artists.append(bg_artist)
            
            pred_style = STYLE_PREDICTED.copy()
            pred_style["label"] = "预测负载"
            if tool_ranges and pred_invalid_blocks:
                _plot_process_step_blocks(
                    axes_pred,
                    pred_invalid_blocks,
                    process_display_start_bounds,
                    process_display_end_bounds,
                    color="#B0BEC5",
                    linewidth=1.0,
                    alpha=0.7,
                    label="非当前有效段",
                    zorder=4
                )
            if pred_blocks:
                _plot_process_step_blocks(
                    axes_pred,
                    pred_blocks,
                    process_display_start_bounds,
                    process_display_end_bounds,
                    color=pred_style.get("color"),
                    linewidth=pred_style.get("linewidth", 1.0),
                    alpha=0.95,
                    label=pred_style.get("label"),
                    zorder=5
                )
            axes_pred.set_ylabel('预测负载 P_pred (W)', fontsize=PLOT_FONT_BASE,
                                 fontweight='bold', color=PLOT_TEXT_COLOR)
            axes_pred.tick_params(labelsize=PLOT_FONT_BASE, colors=PLOT_TEXT_COLOR)
            
            data_source_name = self.get_sample_data_source_name()
            if has_sample_context:
                measured_style = STYLE_MEASURED.copy()
                measured_style["label"] = f"实测({data_source_name})"
                if sample_invalid_blocks:
                    self.plot_series_by_blocks(
                        axes_act,
                        sample_display_x_all,
                        sample_values_all,
                        sample_invalid_blocks,
                        color="#B0BEC5",
                        linewidth=1.0,
                        alpha=0.75,
                        label="非当前有效段",
                        zorder=5
                    )
                if has_sample_valid:
                    self.plot_series_by_blocks(
                        axes_act,
                        sample_display_x_all,
                        sample_values_all,
                        sample_valid_blocks,
                        zorder=6,
                        alpha=0.95,
                        **measured_style
                    )
                    sample_plot_values = sample_values_all[sample_valid_mask]
                else:
                    self.plot_series_by_blocks(
                        axes_act,
                        sample_display_x_all,
                        sample_values_all,
                        sample_context_blocks,
                        color="#B0BEC5",
                        linewidth=1.0,
                        alpha=0.75,
                        label=f"实测({data_source_name})",
                        zorder=5
                    )
                    sample_plot_values = sample_values_all[sample_context_mask]
                if plot_mode == "stacked":
                    self.apply_plot_style(axes_act, grid=True)
                    self.apply_tool_background(axes_act, program_name, None if display_mode == "program" else tool_id)
                    y_max_act = np.nanmax(sample_plot_values) if sample_plot_values is not None and len(sample_plot_values) > 0 else 0
                    
                    # 复用已计算的x坐标和点宽度
                    for item in interval_meta:
                        start_line, end_line, p_pref, ideal_val, idx, start_idx_iv, end_idx_iv = item
                        # 优先使用理想值（基于实测均值），否则使用实测最大值，最后才用预测值（主要防止量级差异）
                        if ideal_val is not None:
                            bar_val = ideal_val
                        elif y_max_act > 0:
                            bar_val = y_max_act * 1.05
                        else:
                            bar_val = p_pref

                        if bar_val is None:
                            continue
                        y0, y1 = (0.0, bar_val) if bar_val >= 0 else (bar_val, 0.0)
                        # 使用区间边界
                        if start_idx_iv is not None and end_idx_iv is not None and len(process_display_start_bounds) > 0:
                            start_x = process_display_start_bounds[start_idx_iv]
                            end_x = process_display_end_bounds[end_idx_iv]
                        else:
                            start_x = start_line
                            end_x = end_line + 1
                        if not np.isfinite(start_x) or not np.isfinite(end_x) or end_x <= start_x:
                            continue
                        axes_act.fill_between(
                            [start_x, end_x], [y0, y0], [y1, y1],
                            alpha=0.18, facecolor=interval_colors[idx % 2],
                            edgecolor='none', zorder=2
                        )
                    axes_act.set_ylabel(f"实测负载 ({data_source_name})", fontsize=PLOT_FONT_BASE,
                                       fontweight='bold', color=PLOT_TEXT_COLOR)
                    axes_act.tick_params(labelsize=PLOT_FONT_BASE, colors=PLOT_TEXT_COLOR)
                else:
                    # 叠加模式：也在实测轴上绘制背景条（使用实测数据的Y轴范围）
                    if sample_plot_values is not None and len(sample_plot_values) > 0:
                        y_max = np.nanmax(sample_plot_values) * 1.1
                        
                        for item in interval_meta:
                            start_line, end_line, p_pref, ideal_val, idx, start_idx_iv, end_idx_iv = item
                            bar_val = ideal_val if ideal_val is not None else (y_max if y_max > 0 else p_pref)
                            if bar_val is None:
                                continue
                            # 使用区间边界
                            if start_idx_iv is not None and end_idx_iv is not None and len(process_display_start_bounds) > 0:
                                start_x = process_display_start_bounds[start_idx_iv]
                                end_x = process_display_end_bounds[end_idx_iv]
                            else:
                                start_x = start_line
                                end_x = end_line + 1
                            if not np.isfinite(start_x) or not np.isfinite(end_x) or end_x <= start_x:
                                continue
                            axes_act.fill_between(
                                [start_x, end_x], [0, 0], [bar_val, bar_val],
                                alpha=0.12, facecolor=interval_colors[idx % 2],
                                edgecolor='none', zorder=1
                            )
                    axes_act.set_ylabel(f"实测负载 ({data_source_name})", fontsize=PLOT_FONT_BASE,
                                       fontweight='bold', color=STYLE_MEASURED["color"])
            else:
                if plot_mode == "stacked":
                    self.apply_plot_style(axes_act, grid=False)
                    axes_act.text(0.5, 0.5, "未加载实测数据", ha='center', va='center',
                                  transform=axes_act.transAxes, fontsize=PLOT_FONT_BASE, color='#666666')
                    axes_act.grid(False)
                else:
                    axes_pred.text(0.5, 0.9, "未加载实测数据", ha='center', va='center',
                                   transform=axes_pred.transAxes, fontsize=PLOT_FONT_BASE, color='#666666')

            self._ideal_line_artists = []
            self._preview_tool_key = None
            self._preview_tool_mean = None
            if has_sample_valid:
                if display_mode == "program" and program_name:
                    for t_id in tool_ids:
                        mean_val, count, ideal_saved = _get_tool_stats(t_id)
                        if mean_val is None or count == 0:
                            continue
                        tool_ranges_draw = self.get_tool_ranges_by_id(program_name, t_id)
                        color = tool_color_map.get(t_id, '#333333')
                        display_spans = self.build_time_spans_for_ranges(tool_ranges_draw, sample_context_mask) if use_time_display else None
                        self.draw_tool_mean_ideal_lines(
                            axes_act, tool_ranges_draw, mean_val, ideal_saved, color, display_spans=display_spans
                        )
                elif display_mode == "tool" and program_name and tool_id:
                    if tool_mean_val is not None and tool_count > 0:
                        tool_ranges_draw = self.get_tool_ranges_by_id(program_name, tool_id)
                        color = tool_color_map.get(tool_id, '#333333')
                        display_spans = self.build_time_spans_for_ranges(tool_ranges_draw, sample_context_mask) if use_time_display else None
                        ideal_lines = self.draw_tool_mean_ideal_lines(
                            axes_act, tool_ranges_draw, tool_mean_val, tool_ideal_val_preview, color,
                            display_spans=display_spans
                        )
                        self._ideal_line_artists = ideal_lines
                        self._preview_tool_key = (program_name, tool_id)
                        self._preview_tool_mean = tool_mean_val
            
            axis_line_numbers = axis_line_numbers_pred
            if len(axis_line_numbers) == 0 and has_sample_context:
                axis_line_numbers = sample_line_numbers_all[sample_context_mask]
            
            x_candidates = []
            process_display_valid = np.isfinite(process_display_start_bounds) & np.isfinite(process_display_end_bounds)
            if pred_mask is not None and pred_mask.any():
                pred_display_mask = pred_mask & process_display_valid
                if pred_display_mask.any():
                    x_candidates.extend([
                        float(np.min(process_display_start_bounds[pred_display_mask])),
                        float(np.max(process_display_end_bounds[pred_display_mask]))
                    ])
            elif process_display_valid.any():
                x_candidates.extend([
                    float(np.min(process_display_start_bounds[process_display_valid])),
                    float(np.max(process_display_end_bounds[process_display_valid]))
                ])
            if has_sample_context:
                context_x = sample_display_x_all[sample_context_mask]
                x_candidates.extend([float(np.min(context_x)), float(np.max(context_x))])
            if x_candidates:
                x_min = min(x_candidates)
                x_max = max(x_candidates)
                x_range = x_max - x_min if x_max > x_min else 1
                axes_pred.set_xlim(x_min - x_range * 0.02, x_max + x_range * 0.02)
                if plot_mode == "stacked" or overlay_secondary:
                    axes_act.set_xlim(x_min - x_range * 0.02, x_max + x_range * 0.02)

            if use_time_display:
                if plot_mode == "stacked":
                    axes_act.set_xlabel('时间 (ms)', fontsize=PLOT_FONT_BASE, fontweight='bold', color=PLOT_TEXT_COLOR)
                else:
                    axes_pred.set_xlabel('时间 (ms)', fontsize=PLOT_FONT_BASE, fontweight='bold', color=PLOT_TEXT_COLOR)
                self.apply_line_axis_on_time(axes_pred, sample_context_mask)
            else:
                if plot_mode == "stacked":
                    axes_act.set_xlabel('对齐行号', fontsize=PLOT_FONT_BASE, fontweight='bold', color=PLOT_TEXT_COLOR)
                    self.apply_line_axis(axes_act, axis_line_numbers)
                else:
                    axes_pred.set_xlabel('对齐行号', fontsize=PLOT_FONT_BASE, fontweight='bold', color=PLOT_TEXT_COLOR)
                    self.apply_line_axis(axes_pred, axis_line_numbers)
            
            # 标题样式 - 工业简洁风格（不显示标题，通过图例区分）
            # axes_pred.set_title('', fontsize=PLOT_FONT_BASE, fontweight='bold', color=PLOT_TEXT_COLOR, pad=8)
            
            # 图例样式 - 工业科技感
            handles, labels = axes_pred.get_legend_handles_labels()
            if overlay_secondary:
                h2, l2 = axes_act.get_legend_handles_labels()
                handles = handles + h2
                labels = labels + l2
            if handles:
                legend = axes_pred.legend(handles, labels, loc='upper right', fontsize=PLOT_FONT_BASE - 1,
                                          framealpha=0.95, shadow=False, fancybox=False,
                                          edgecolor=PLOT_SPINE_COLOR, borderpad=0.6, labelspacing=0.4)
                legend.get_frame().set_facecolor('white')
                legend.get_frame().set_linewidth(0.6)
            
            if display_mode == "program":
                avg_lines = []
                ideal_lines = []
                for t_id in tool_ids:
                    mean_val, count, ideal_saved = _get_tool_stats(t_id)
                    avg_val = "-" if mean_val is None or count == 0 else f"{mean_val:.3f}"
                    avg_lines.append(f"{t_id} {avg_val}")

                    if ideal_saved is not None:
                        ideal_val = f"{ideal_saved:.3f}"
                    elif not has_process_file or not has_processed:
                        ideal_val = "待导入"
                    else:
                        ideal_val = "未设定"
                    ideal_lines.append(f"{t_id} {ideal_val}")
                if avg_lines:
                    self.sample_avg_var.set("\n".join(avg_lines))
                    self.sample_ideal_var.set("\n".join(ideal_lines))
                else:
                    self.sample_avg_var.set("-")
                    self.sample_ideal_var.set("-")
            else:
                tool_label = self.sample_tool_name.get().strip() or (self.format_tool_label(tool_id) if tool_id else "当前刀具")
                if not has_process_file:
                    if tool_mean_val is not None and tool_count > 0:
                        self.sample_avg_var.set(f"{tool_mean_val:.3f}")
                    else:
                        self.sample_avg_var.set("-")
                    self.sample_ideal_var.set("待导入")
                else:
                    if tool_mean_val is not None and tool_count > 0:
                        self.sample_avg_var.set(f"{tool_mean_val:.3f}")
                        if tool_saved and tool_ideal_val_saved is not None:
                            self.sample_ideal_var.set(f"{tool_ideal_val_saved:.3f}")
                        else:
                            self.sample_ideal_var.set("未设定")
                    else:
                        self.sample_avg_var.set("-")
                        self.sample_ideal_var.set("-")
            
            fig2.subplots_adjust(
                left=0.10,
                right=0.90,
                top=0.86 if use_time_display else 0.92,
                bottom=0.08,
                hspace=0.15
            )
            self.figures.append(fig2)
            
            self.figure_names = ["预测负载与实测负载"]
            self.figure_selector["values"] = self.figure_names
            default_index = 0
            if self.figure_names:
                self.figure_selector.current(default_index)
            
            if save:
                self.save_all_plots(silent=True)
            self.show_current_figure(default_index)
            
            # 更新区间数量显示（使用 interval_meta 长度，与图表绘制的区间数量完全一致）
            if hasattr(self, 'interval_count_var'):
                interval_count = len(interval_meta) if interval_meta else 0
                self.interval_count_var.set(str(interval_count))
            
            total_charts = len(self.figures)
            self.status_var_data.set(f"图表已生成! 共{total_charts}张图表")
            if not save and not silent:
                messagebox.showinfo("完成", f"{total_charts}张图表已成功生成! 可继续保存结果(.rg)")
            
            return True
        
        except Exception as e:
            messagebox.showerror("图表生成错误", f"生成图表时发生错误:\n{str(e)}")
            self.status_var_data.set("图表生成失败")
            return False
    
    def show_current_figure(self, index=0):
        """显示当前图表（直接渲染原始Figure，避免复制导致的元素丢失/错位）"""
        if not self.figures or index >= len(self.figures):
            return

        self.current_figure_index = index

        # 清空预览区域（销毁旧的canvas组件）
        for widget in self.data_figure_frame.winfo_children():
            widget.destroy()

        fig = self.figures[index]
        self._current_preview_fig = fig  # 供缩放等交互使用

        # 默认取主轴（缩放时会同步作用到同一Figure的其它轴）
        self.ax_data = fig.axes[0] if getattr(fig, 'axes', None) else None

        self.canvas_data = FigureCanvasTkAgg(fig, master=self.data_figure_frame)
        canvas_widget = self.canvas_data.get_tk_widget()
        canvas_widget.grid(row=0, column=0, sticky="nsew")
        canvas_widget.configure(relief=tk.FLAT, bd=0)
        canvas_widget.bind("<Enter>", lambda _e: canvas_widget.focus_set())
        canvas_widget.focus_set()
        self._preview_canvas_widget = canvas_widget
        self._ensure_preview_mousewheel_binding()

        # 绑定滚轮缩放交互
        try:
            self.canvas_data.mpl_connect('scroll_event', self.on_data_scroll_zoom)
        except Exception:
            pass

        # 绑定鼠标左键横向拖动
        try:
            self.canvas_data.mpl_connect('button_press_event', self.on_data_pan_press)
            self.canvas_data.mpl_connect('button_release_event', self.on_data_pan_release)
            self.canvas_data.mpl_connect('motion_notify_event', self.on_data_pan_motion)
        except Exception:
            pass

        self.update_nav_buttons()

        self.canvas_data.draw_idle()
        self.root.after_idle(self.adjust_figure_sizes)
        self.root.after(10, lambda: self._focus_preview_canvas(canvas_widget))

    def _on_preview_canvas_configure(self, event):
        """预览区大小变化时触发图表重排（防抖避免拖拽卡顿）"""
        try:
            if getattr(self, '_preview_resize_timer', None) is not None:
                self.root.after_cancel(self._preview_resize_timer)
            # 80ms 更快响应，同时避免频繁重绘
            self._preview_resize_timer = self.root.after(80, self._deferred_adjust_figure)
        except Exception:
            pass
    
    def _deferred_adjust_figure(self):
        """延迟调整图表尺寸 - 性能优化"""
        self._preview_resize_timer = None
        try:
            self.adjust_figure_sizes()
        except Exception:
            pass

    def on_figure_selected(self, event=None):
        """下拉选择图表"""
        if not self.figure_names:
            return
        selected = self.figure_selector.current()
        if selected >= 0:
            self.show_current_figure(selected)
    
    def save_all_plots(self, silent=False):
        """保存所有图表到exe所在目录"""
        if not self.figures:
            if not silent:
                messagebox.showwarning("无图表", "没有可保存的图表，请先生成图表")
            return False
        
        try:
            # 使用exe所在目录或sample_data_dir
            save_dir = self.sample_data_dir or app_dir
            
            # 定义图表文件名
            filenames = [
                "P_pred_actual_steady_intervals"
            ]
            
            # 保存所有图表 - 同时保存高DPI的PNG和矢量SVG格式
            for i, fig in enumerate(self.figures):
                if i < len(filenames):  # 确保不超出文件名列表范围
                    # 保存高清PNG (用于预览)
                    png_path = os.path.join(save_dir, f"{filenames[i]}.png")
                    fig.savefig(png_path, dpi=600, bbox_inches='tight', format='png')
                    
                    # 保存SVG矢量图 (可无损缩放)
                    svg_path = os.path.join(save_dir, f"{filenames[i]}.svg")
                    fig.savefig(svg_path, bbox_inches='tight', format='svg')
            
            # 如果有预测功率稳态区间，保存区间数据
            if self.pred_power_intervals:
                intervals_txt_path = os.path.join(save_dir, "P_pred_steady_intervals.txt")
                try:
                    adjustment_ratio = float(self.adjustment_ratio.get())
                except Exception:
                    adjustment_ratio = 1.0
                with open(intervals_txt_path, 'w', encoding='utf-8') as f:
                    f.write("# 预测功率稳态区间划分结果\n")
                    f.write(f"# 优化倍率 R: {adjustment_ratio:g}\n")
                    f.write(f"# 最小采样点数: {self.pred_power_min_length.get()}\n")
                    f.write(f"# 总区间数: {len(self.pred_power_intervals)}\n")
                    f.write("#" + "="*80 + "\n")
                    f.write("# 区间\t起始行号\t结束行号\t采样点数\tP_pred(W)\tP_pref(W)\n")
                    for i, interval in enumerate(self.pred_power_intervals, 1):
                        p_pred = interval['p_pred']
                        p_pref = p_pred * adjustment_ratio
                        sample_count = interval.get('sample_count', interval['end_idx'] - interval['start_idx'] + 1)
                        f.write(f"{i}\t{interval['start_n']}\t{interval['end_n']}\t"
                               f"{sample_count}\t{p_pred:.6f}\t{p_pref:.6f}\n")

            if not silent:
                self.status_var_data.set(f"所有图表已保存到: {save_dir}")
                messagebox.showinfo("保存成功", f"所有图表已自动保存到:\n{save_dir}")
            
            return True
        
        except Exception as e:
            if not silent:
                messagebox.showerror("保存错误", f"保存图表时发生错误:\n{str(e)}")
            return False
    
    def show_prev_figure(self):
        """显示上一张图表"""
        if not hasattr(self, 'current_figure_index'):
            self.current_figure_index = 0
        
        if self.current_figure_index > 0:
            self.show_current_figure(self.current_figure_index - 1)
    
    def show_next_figure(self):
        """显示下一张图表"""
        if not hasattr(self, 'current_figure_index'):
            self.current_figure_index = 0
        
        if self.current_figure_index < len(self.figures) - 1:
            self.show_current_figure(self.current_figure_index + 1)
    
    def count_sample_points_in_range(self, start_line, end_line, program_no=None):
        """计算指定行号范围内的SampleData数据点数
        
        需求3核心方法：最小样本点数指的是SampleData.csv中对应行号范围的数据点数
        :param start_line: 起始行号
        :param end_line: 结束行号
        :param program_no: 程序号（用于过滤，可选）
        :return: SampleData中该行号范围内的数据点数
        """
        if not self.sample_data_loaded or self.sample_data_line_numbers is None:
            return 0
        
        line_numbers = self.sample_data_line_numbers
        mask = (line_numbers >= start_line) & (line_numbers <= end_line)
        
        # 如果指定了程序号，进一步过滤
        if program_no is not None and self.sample_data_program_numbers is not None:
            mask &= (self.sample_data_program_numbers == program_no)

        count, blocks = self.count_points_by_blocks(mask, mode="max_block")
        self.sample_data_valid_mask = mask
        self.sample_data_valid_blocks = blocks
        return int(count)

    def count_sample_points_in_x_range(self, start_x, end_x, program_no=None):
        """计算指定x坐标范围内的SampleData数据点数
        
        使用区间边界而非行号范围，确保精确计算。
        :param start_x: 区间起始x坐标
        :param end_x: 区间结束x坐标
        :param program_no: 程序号（用于过滤，可选）
        :return: SampleData中该x坐标范围内的数据点数
        """
        if not self.sample_data_loaded or self.sample_data_line_numbers is None:
            return 0
        
        sample_x = self.sample_data_x_positions
        if sample_x is None:
            sample_x = np.asarray(
                self.compute_line_x_positions(self.sample_data_line_numbers, blocks=self.sample_data_base_blocks)
            )
        else:
            sample_x = np.asarray(sample_x)
        
        # 使用半开区间 [start_x, end_x)
        mask = (sample_x >= start_x) & (sample_x < end_x)
        
        if program_no is not None and self.sample_data_program_numbers is not None:
            mask &= (self.sample_data_program_numbers == program_no)

        count, blocks = self.count_points_by_blocks(mask, mode="max_block")
        self.sample_data_valid_mask = mask
        self.sample_data_valid_blocks = blocks
        return int(count)

    def summarize_process_interval(self, start_idx, end_idx):
        """汇总稳态区间内的几何与模型参数，用于生成PIT。"""
        if not self.data or start_idx is None or end_idx is None:
            return {
                "a_p": 0.0,
                "a_e": 0.0,
                "F_plan": 0.0,
                "p_idle": float(self.p_idle_var.get()),
            }

        segment = self.data[start_idx:end_idx + 1]
        if not segment:
            return {
                "a_p": 0.0,
                "a_e": 0.0,
                "F_plan": 0.0,
                "p_idle": float(self.p_idle_var.get()),
            }

        def _collect(key, fallback=0.0):
            values = []
            for row in segment:
                try:
                    values.append(float(row.get(key, fallback)))
                except Exception:
                    continue
            return values

        ap_values = _collect("ap")
        ae_values = _collect("ae")
        feed_values = []
        for row in segment:
            try:
                mrr_val = float(row.get("MRR", 0.0))
                ap_val = float(row.get("ap", 0.0))
                ae_val = float(row.get("ae", 0.0))
                if ap_val > 0 and ae_val > 0:
                    feed_values.append(mrr_val * 60.0 / (ap_val * ae_val))
            except Exception:
                continue
        idle_values = _collect("P_idle", float(self.p_idle_var.get()))

        return {
            "a_p": float(np.median(ap_values)) if ap_values else 0.0,
            "a_e": float(np.median(ae_values)) if ae_values else 0.0,
            "F_plan": float(np.median(feed_values)) if feed_values else 0.0,
            "p_idle": float(np.mean(idle_values)) if idle_values else float(self.p_idle_var.get()),
        }

    def partition_pred_power_steady_intervals(self, P_values, s_values, cumulative_s, n_values, line_numbers=None, debug_line_range=None):
        """
        划分预测功率稳态区间：将P_pred完全恒定的连续区域划分为稳态区间
        改进：
        1) 使用量化后的功率值，任何可观察到的阶跃都会切断区间
        2) 保留所有分段边界，再按最小采样点数过滤（需求3：基于SampleData.csv的数据点数）
        :param P_values: 预测功率P_pred值列表
        :param s_values: 各行的行程长度列表（保留用于兼容，但不再用于过滤）
        :param cumulative_s: 累计行程列表（保留用于兼容）
        :param n_values: 指令行号列表
        :param debug_line_range: 可选，(start_line, end_line) 用于调试特定行号范围内的划分逻辑
        :return: 稳态区间列表
        """
        min_sample_count = int(self.pred_power_min_length.get())  # 最小采样点数（基于SampleData）
        intervals = []
        
        if not P_values or len(P_values) == 0:
            return intervals
        
        # 将累计行程转换为列表（如果是numpy数组）
        if isinstance(cumulative_s, np.ndarray):
            cumulative_s = cumulative_s.tolist()
        
        # 计算工艺信息的区间边界（每行是一段区间而非一个点）
        process_blocks = self.compute_sequence_blocks(line_numbers)
        process_start_bounds, process_end_bounds = self.compute_line_segment_bounds(
            line_numbers,
            blocks=process_blocks
        )
        current_program_no = self.get_selected_program_number()
        
        # 量化功率，去掉浮点微抖
        quantized_P = []
        for p in P_values:
            try:
                quantized_P.append(round(float(p), 6))
            except Exception:
                quantized_P.append(0.0)

        # 判断是否需要切段：直接比较量化后的值是否相等
        def is_diff(p1, p2):
            return p1 != p2
        
        # 调试：记录所有原始分段（包括被过滤的短区间）
        all_segments = []
        debug_enabled = debug_line_range is not None
        debug_start, debug_end = debug_line_range if debug_line_range else (None, None)
        pit_metadata = self.get_current_pit_metadata()
        
        i = 0
        while i < len(quantized_P):
            current_p = quantized_P[i]
            start_idx = i
            start_s = cumulative_s[i] - s_values[i] if i > 0 else 0  # 该行起始位置（保留用于记录）
            current_line = line_numbers[i] if line_numbers is not None else i
            
            # 查找P_pred稳定的连续区域
            j = i + 1
            while j < len(quantized_P):
                # 与区间首值或前一个点只要有变化就分段
                if is_diff(quantized_P[j], current_p) or is_diff(quantized_P[j], quantized_P[j-1]):
                    break
                j += 1
            
            end_idx = j - 1
            end_s = cumulative_s[end_idx]  # 该区域结束位置（保留用于记录）
            end_line = line_numbers[end_idx] if line_numbers is not None else end_idx
            
            # 需求3：使用区间边界计算SampleData点数（而非行号范围）
            if process_start_bounds and start_idx < len(process_start_bounds) and end_idx < len(process_end_bounds):
                interval_start_x = process_start_bounds[start_idx]
                interval_end_x = process_end_bounds[end_idx]
                sample_count = self.count_sample_points_in_x_range(
                    interval_start_x,
                    interval_end_x,
                    program_no=current_program_no
                )
            else:
                # 回退到行号范围计算
                sample_count = self.count_sample_points_in_range(
                    current_line,
                    end_line,
                    program_no=current_program_no
                )
            process_row_count = end_idx - start_idx + 1  # 工艺信息文件行数（保留用于调试）
            
            # 记录所有分段用于调试
            segment_info = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_line': current_line,
                'end_line': end_line,
                'start_s': start_s,
                'end_s': end_s,
                'sample_count': sample_count,  # SampleData点数
                'process_row_count': process_row_count,  # 工艺信息行数
                'p_pred': float(P_values[start_idx]),
                'passed_min_count': sample_count >= min_sample_count
            }
            all_segments.append(segment_info)
            
            # 如果采样点数大于等于最小样本点，则保存该区间
            if sample_count >= min_sample_count:
                interval_meta = self.summarize_process_interval(start_idx, end_idx)
                kc_hat = float(self.kc_coeff.get())
                sigma_kc = float(self.kc_sigma.get())
                pit_entry = PITEntry(
                    zone_id=f"Z{len(intervals) + 1:03d}",
                    start_idx=start_idx,
                    end_idx=end_idx,
                    start_line=int(line_numbers[start_idx]) if line_numbers is not None else int(start_idx),
                    end_line=int(line_numbers[end_idx]) if line_numbers is not None else int(end_idx),
                    start_s=float(start_s),
                    end_s=float(end_s),
                    a_p=float(interval_meta["a_p"]),
                    a_e=float(interval_meta["a_e"]),
                    F_plan=float(interval_meta["F_plan"]),
                    p_idle=float(interval_meta["p_idle"]),
                    p_pred=float(np.mean(P_values[start_idx:end_idx + 1])),
                    K_c_hat=kc_hat,
                    K_c_UCB=kc_hat + float(self.kc_beta.get()) * sigma_kc,
                    sigma_Kc=sigma_kc,
                    sample_count=int(sample_count),
                    start_n=n_values[start_idx],
                    end_n=n_values[end_idx],
                    tool_diameter=pit_metadata["tool_diameter"],
                    tool_radius=pit_metadata["tool_radius"],
                    tool_material=pit_metadata["tool_material"],
                    blank_material=pit_metadata["blank_material"],
                )
                intervals.append(asdict(pit_entry))
            
            i = j
        
        # 调试输出：显示指定行号范围内的分段情况
        if debug_enabled and all_segments:
            print(f"\n{'='*80}")
            print(f"[DEBUG] 稳态区间划分调试 - 行号范围: {debug_start} ~ {debug_end}")
            print(f"[DEBUG] 最小采样点数阈值: {min_sample_count}")
            print(f"{'='*80}")
            
            debug_segments = []
            for seg in all_segments:
                seg_start = seg['start_line']
                seg_end = seg['end_line']
                # 检查是否与调试范围有交集
                if seg_end >= debug_start and seg_start <= debug_end:
                    debug_segments.append(seg)
            
            if debug_segments:
                print(f"\n[DEBUG] 在调试范围内发现 {len(debug_segments)} 个分段:")
                for idx, seg in enumerate(debug_segments):
                    status = "✓ 保留" if seg['passed_min_count'] else "✗ 过滤(点数太少)"
                    print(f"  [{idx+1}] 行号: {seg['start_line']} ~ {seg['end_line']}, "
                          f"采样点数: {seg['sample_count']}, P_pred: {seg['p_pred']:.2f}W, {status}")
                
                # 分析为何可能被合并
                print(f"\n[DEBUG] 区间合并分析:")
                passed_intervals = [seg for seg in debug_segments if seg['passed_min_count']]
                filtered_intervals = [seg for seg in debug_segments if not seg['passed_min_count']]
                
                print(f"  - 通过最小样本点阈值的区间数: {len(passed_intervals)}")
                print(f"  - 被过滤的短区间数: {len(filtered_intervals)}")
                
                if len(passed_intervals) >= 2:
                    # 检查相邻保留区间之间是否有被过滤的区间
                    for i in range(len(passed_intervals) - 1):
                        curr = passed_intervals[i]
                        next_seg = passed_intervals[i + 1]
                        gap_start = curr['end_line']
                        gap_end = next_seg['start_line']
                        
                        # 找出gap中的被过滤区间
                        gap_filtered = [seg for seg in filtered_intervals 
                                       if seg['start_line'] >= gap_start and seg['end_line'] <= gap_end]
                        
                        if gap_filtered:
                            print(f"\n  ⚠ 区间 [{curr['start_line']}~{curr['end_line']}] 与 "
                                  f"[{next_seg['start_line']}~{next_seg['end_line']}] 之间:")
                            for gap_seg in gap_filtered:
                                print(f"    - 被过滤的短区间: 行号 {gap_seg['start_line']}~{gap_seg['end_line']}, "
                                      f"采样点数 {gap_seg['sample_count']} < {min_sample_count}")
                            print(f"    → 如果这些区间行号连续，后续 merge_intervals 会将两个保留区间合并!")
            else:
                print(f"[DEBUG] 在调试范围 {debug_start}~{debug_end} 内未发现任何分段")
            
            print(f"{'='*80}\n")
        
        # 注意：MAX_INTERVALS限制已移到generate_plots中对最终显示区间做限制
        # 这样可以确保限制的是合并后的实际显示区间数
        
        return intervals
    
    def debug_interval_partition(self, start_line, end_line):
        """调试指定行号范围内的稳态区间划分
        
        使用方法（在终端中）:
            # 假设 app 是 MillingAnalysisTool 实例
            app.debug_interval_partition(26800, 26880)
        
        :param start_line: 调试范围起始行号
        :param end_line: 调试范围结束行号
        """
        if not self.data:
            print("[ERROR] 请先加载并处理数据")
            return
        
        print(f"\n{'#'*80}")
        print(f"# 调试稳态区间划分: 行号范围 {start_line} ~ {end_line}")
        print(f"{'#'*80}")
        
        # 重新生成图表并启用调试
        self.generate_plots(debug_line_range=(start_line, end_line))
        
        # 打印最终显示的区间
        print(f"\n[DEBUG] 最终显示的稳态区间 (pred_power_intervals):")
        for idx, interval in enumerate(self.pred_power_intervals):
            start_l = interval.get('start_line')
            end_l = interval.get('end_line')
            if end_l >= start_line and start_l <= end_line:
                print(f"  [{idx+1}] 行号: {start_l} ~ {end_l}, "
                      f"采样点数: {interval.get('sample_count', 0)}, "
                      f"P_pred: {interval.get('p_pred', 0):.2f}W")
        
        print(f"\n{'#'*80}\n")
    
    def update_nav_buttons(self):
        """更新导航按钮状态"""
        if not hasattr(self, 'current_figure_index'):
            self.current_figure_index = 0
        
        if not self.figures:
            self.prev_btn.config(state=tk.DISABLED)
            self.next_btn.config(state=tk.DISABLED)
            self.figure_label.config(text="")
            self.figure_selector["values"] = []
            self.figure_selector_var.set("")
            return
        
        # 更新标签
        if self.current_figure_index < len(self.figure_names):
            self.figure_label.config(text=f"{self.current_figure_index + 1}/{len(self.figures)} - {self.figure_names[self.current_figure_index]}")
        else:
            self.figure_label.config(text=f"{self.current_figure_index + 1}/{len(self.figures)}")

        # 同步下拉选择
        if self.figure_names and self.current_figure_index < len(self.figure_names):
            self.figure_selector.current(self.current_figure_index)
        
        # 更新按钮状态
        self.prev_btn.config(state=tk.NORMAL if self.current_figure_index > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.current_figure_index < len(self.figures) - 1 else tk.DISABLED)
        
    def detect_file_encoding(self,file_path):
        """使用 Python 内置方法检测文件编码"""
        # 常见编码列表，按优先级排序
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    # 尝试读取文件内容
                    f.read(1024)  # 只读取前1024字节进行测试
                return encoding
            except UnicodeDecodeError:
                continue
        
        # 如果所有编码都失败，返回默认编码
        return 'utf-8'
 
    def parse_channel_data_file(self, file_path):
        """解析包含ChannelInfo和ChannelData的文件格式"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    
        # 使用正则表达式提取ChannelInfo块
        channel_info_blocks = re.findall(r'<ChannelInfo>\s*([^<]*)', content)
        load_current_col = None
        program_line_col = None
        for i, block in enumerate(channel_info_blocks):
            lines = block.strip().split('\n')
            lines = [line.strip() for line in lines if line.strip()]
            if len(lines) >= 3:
                name = lines[2].strip('<> ')
                if name == '负载电流':
                    load_current_col = i
                elif name == '程序行号':
                    program_line_col = i
    
        if load_current_col is None or program_line_col is None:
            raise ValueError("无法找到负载电流或程序行号通道")
    
        # 解析ChannelData块
        channel_data_blocks = re.findall(r'<ChannelData>\s*([^<]*)', content)
        currents = []
        program_lines = []
        for block in channel_data_blocks:
            lines = block.strip().split('\n')
            lines = [line.strip() for line in lines if line.strip()]
            if len(lines) > max(load_current_col, program_line_col):
                try:
                    current_val = float(lines[load_current_col])
                    program_line_val = lines[program_line_col].strip()
                    currents.append(current_val)
                    program_lines.append(program_line_val)
                except ValueError:
                    continue  # 跳过无法转换的数字
    
        return currents, program_lines
        
    def process_single_file(self, input_file):
        """处理单个文件的核心逻辑 - 仅解析数据，不生成后处理文件"""
        try:
            # 获取参数
            origin = (
                self.origin_x.get(),
                self.origin_y.get(),
                self.origin_z.get()
            )
            rapid_speed_xy = self.rapid_speed_xy.get()
            rapid_speed_z = self.rapid_speed_z.get()
            
            # 只读取文件，不写入输出
            input_encoding = self.detect_file_encoding(input_file)
            with open(input_file, 'r', encoding=input_encoding, errors='ignore') as infile:
                prev_coords = origin
                current_coords = origin
                data = []
                s_base = self.s_base.get()
                k_base = self.k_base.get()
                current_s = float(self.current_program_speed.get() or s_base)
                current_feed = 0.0
                current_move_type = "rapid"  # 从机床原点开始，初始为快速移动
                
                prev_aligned_line = None
                prev_gcode_content = None  # 跟踪上一行的G代码内容（第六列之后）
                current_n_group_line = None  # 当前G代码组的重构行号
                prev_raw_line = None  # 跟踪上一行的原始行号，用于检测行号缺失
                for line_num, line in enumerate(infile):
                    parsed = self.parse_gcode_line(line)
                    if not parsed:
                        continue
                    
                    ap, ae, feed_rate, gcode_content, s_value, raw_line_number = parsed
                    nc_state = self._resolve_nc_state_for_process_row(raw_line_number, gcode_content)
                    nc_speed = float(nc_state.get("command_speed", 0.0)) if nc_state else 0.0
                    nc_feed = float(nc_state.get("feed", 0.0)) if nc_state else 0.0
                    
                    # === 行号缺失补齐：检测raw_line_number跳跃，用P=0占位 ===
                    if raw_line_number is not None and prev_raw_line is not None and raw_line_number > prev_raw_line + 1:
                        for missing_raw in range(prev_raw_line + 1, raw_line_number):
                            fill_aligned = current_n_group_line + 1 if current_n_group_line is not None else missing_raw
                            current_n_group_line = fill_aligned
                            prev_aligned_line = fill_aligned
                            prev_gcode_content = None  # 补齐行没有G代码内容
                            fill_idle = self.predict_idle_power(current_s)
                            data.append({
                                's': 0, 't': 0,
                                'ap': 0, 'ae': 0,
                                'dMRV': 0, 'MRR': 0,
                                'S': current_s, 'K': float(self.kc_coeff.get()),
                                'T': 0, 'P': fill_idle,
                                'P_idle': fill_idle,
                                'P_edge': 0.0,
                                'K_c': float(self.kc_coeff.get()),
                                'K_e': float(self.ke_coeff.get()),
                                'type': 'rapid',
                                'N_str': None,
                                'line_no_raw': missing_raw,
                                'line_no_aligned': fill_aligned
                            })
                    
                    # 更新转速
                    if s_value is not None:
                        current_s = s_value
                    elif nc_speed > 0:
                        current_s = nc_speed
                    
                    # 更新进给速度
                    if feed_rate and float(feed_rate) > 0:
                        current_feed = float(feed_rate)
                    elif nc_feed > 0:
                        current_feed = nc_feed
                    
                    n_value = self.extract_n_value(gcode_content)
                    
                    # 根据G代码内容（第六列之后所有内容）决定重构行号
                    # 规则：
                    # 1. 第一行：使用原始第二列值作为起始重构行号
                    # 2. 连续行的G代码内容完全一致 → 重构行号保持不变（同一条G代码指令的细分）
                    # 3. G代码内容变化 → 重构行号 +1（保持连续）
                    
                    if prev_aligned_line is None:
                        # 第一行：使用原始第二列值
                        aligned_line = raw_line_number if raw_line_number is not None else 0
                        current_n_group_line = aligned_line
                        prev_gcode_content = gcode_content
                    else:
                        if gcode_content == prev_gcode_content:
                            # G代码内容相同，重构行号保持不变
                            aligned_line = current_n_group_line
                        else:
                            # G代码内容变化，重构行号 +1
                            aligned_line = current_n_group_line + 1
                            current_n_group_line = aligned_line
                            prev_gcode_content = gcode_content
                    
                    prev_aligned_line = aligned_line
                    # 更新上一行的原始行号
                    if raw_line_number is not None:
                        prev_raw_line = raw_line_number
                    
                    # 提取当前坐标
                    current_coords = self.extract_coordinates(gcode_content, prev_coords)
                    
                    # 计算行程距离
                    s = self.calculate_distance(prev_coords, current_coords)
                    
                    # 确定移动类型（注意：必须先检测切削指令G01/G02/G03，再检测快速移动G0/G00）
                    # 否则 "G0" in "G01" 会误判切削为快速移动
                    if any(gcode in gcode_content for gcode in ["G1", "G01", "G2", "G02", "G3", "G03"]):
                        current_move_type = "cutting"
                    elif "G0" in gcode_content or "G00" in gcode_content:
                        current_move_type = "rapid"
                    
                    # 根据移动类型计算时间
                    if current_move_type == "rapid":
                        dx = current_coords[0] - prev_coords[0]
                        dy = current_coords[1] - prev_coords[1]
                        dz = current_coords[2] - prev_coords[2]
                        
                        dist_xy = math.sqrt(dx**2 + dy**2)
                        dist_z = abs(dz)
                        
                        if dist_xy > 0 and dist_z == 0:
                            t_val = dist_xy / (rapid_speed_xy / 60.0)
                        elif dist_xy == 0 and dist_z > 0:
                            t_val = dist_z / (rapid_speed_z / 60.0)
                        elif dist_xy > 0 and dist_z > 0:
                            t_val_xy = dist_xy / (rapid_speed_xy / 60.0)
                            t_val_z = dist_z / (rapid_speed_z / 60.0)
                            t_val = max(t_val_xy, t_val_z)
                        else:
                            t_val = 0.0
                    else:
                        t_val = s / (current_feed / 60.0) if current_feed > 0 and s > 0 else 0.0
                    
                    # 直接计算工艺参数
                    t_val, dmrv_val, mrr_val, k_val, t_torque, p_power, p_idle, p_edge = self.calculate_additional_columns(
                        ap, ae, current_feed, s, current_s, s_base, k_base
                    )
                    
                    # 收集数据（不再写入文件）
                    data.append({
                        's': s,
                        't': t_val,
                        'ap': float(ap),
                        'ae': float(ae),
                        'dMRV': dmrv_val,
                        'MRR': mrr_val,
                        'S': current_s,
                        'K': k_val,
                        'T': t_torque,
                        'P': p_power,
                        'P_idle': p_idle,
                        'P_edge': p_edge,
                        'K_c': float(self.kc_coeff.get()),
                        'K_e': float(self.ke_coeff.get()),
                        'type': current_move_type,
                        'N_str': n_value,  # 存储N列字符串值
                        'line_no_raw': raw_line_number,
                        'line_no_aligned': aligned_line
                    })
                    
                    # 更新上一行坐标
                    prev_coords = current_coords
                
                self.data = data

            self.build_raw_to_aligned_line_map()
            if self.sample_data_loaded:
                self.align_sample_data_to_processed()
            
            return True
        
        except Exception as e:
            raise Exception(f"处理文件 {input_file} 时出错: {str(e)}")
        
    def setup_tech_theme(self):
        """配置科技蓝+白色主题样式 - 专业级UI设计"""
        style = ttk.Style()
        
        # 设置主题为clam（比较容易自定义）
        try:
            style.theme_use('clam')
        except:
            pass

        # ===== 科技蓝配色方案 =====
        btn_base = UI_COLOR_PRIMARY        # 科技蓝
        btn_active = '#3399FF'             # 悬停时稍亮
        btn_pressed = '#0066CC'            # 按下时深蓝
        btn_disabled = '#87CEEB'           # 禁用时浅蓝
        btn_orange = UI_COLOR_WARNING      # 橙色按钮（保存等重要操作）
        btn_orange_active = '#FFA500'
        btn_orange_pressed = '#E67300'

        # 全局默认样式 - 白色背景
        style.configure('.', font=UI_FONT_NORMAL, background=UI_COLOR_BG_LIGHT)
        style.configure('TFrame', background=UI_COLOR_BG_LIGHT)
        style.configure('TLabel', background=UI_COLOR_BG_LIGHT, foreground=UI_COLOR_TEXT, font=UI_FONT_NORMAL)
        
        # 默认按钮 - 科技蓝
        style.configure(
            'TButton',
            font=UI_FONT_NORMAL,
            padding=(12, 6),
            background=btn_base,
            foreground='white',
            borderwidth=1,
            relief='solid',
            focuscolor=btn_base
        )
        style.map(
            'TButton',
            background=[
                ('active', btn_active),
                ('pressed', btn_pressed),
                ('disabled', btn_disabled)
            ],
            foreground=[('active', 'white'), ('pressed', 'white'), ('disabled', '#CCCCCC')],
            bordercolor=[
                ('active', btn_active),
                ('pressed', btn_pressed),
                ('!disabled', btn_base)
            ]
        )
        
        # 主要操作按钮 - 科技蓝
        style.configure('Tech.TButton',
                       background=btn_base,
                       foreground='white',
                       font=UI_FONT_BOLD,
                       borderwidth=1,
                       relief='solid',
                       bordercolor=btn_base,
                       focuscolor=btn_base,
                       padding=(16, 8))
        style.map('Tech.TButton',
                 background=[('active', btn_active), ('pressed', btn_pressed), ('disabled', btn_disabled)],
                 foreground=[('active', 'white'), ('pressed', 'white'), ('disabled', '#CCCCCC')],
                 bordercolor=[('!disabled', btn_base)])
        
        # 橙色强调按钮 - 用于保存等重要操作
        style.configure('Orange.TButton',
                       background=btn_orange,
                       foreground='white',
                       font=UI_FONT_BOLD,
                       borderwidth=1,
                       relief='solid',
                       bordercolor=btn_orange,
                       focuscolor=btn_orange,
                       padding=(16, 8))
        style.map('Orange.TButton',
                 background=[('active', btn_orange_active), ('pressed', btn_orange_pressed), ('disabled', '#FDBA74')],
                 foreground=[('active', 'white'), ('pressed', 'white'), ('disabled', '#CCCCCC')],
                 bordercolor=[('!disabled', btn_orange)])

        # 图表导航按钮 - 科技蓝
        style.configure('Nav.TButton',
                       background=btn_base,
                       foreground='white',
                       font=UI_FONT_BOLD,
                       borderwidth=1,
                       relief='solid',
                       bordercolor=btn_base,
                       focuscolor=btn_base,
                       padding=(14, 8))
        style.map('Nav.TButton',
                 background=[('active', btn_active), ('pressed', btn_pressed)],
                 foreground=[('active', 'white'), ('pressed', 'white')],
                 bordercolor=[('!disabled', btn_base)])
        
        # 成功操作按钮 - 科技蓝
        style.configure('Primary.TButton',
                       background=btn_base,
                       foreground='white',
                       font=UI_FONT_BOLD,
                       borderwidth=1,
                       relief='solid',
                       bordercolor=btn_base,
                       focuscolor=btn_base,
                       padding=(18, 10))
        style.map('Primary.TButton',
                 background=[('active', btn_active), ('pressed', btn_pressed)],
                 foreground=[('active', 'white'), ('pressed', 'white')],
                 bordercolor=[('!disabled', btn_base)])
        
        # 警告/危险操作按钮 - 红色
        style.configure('Danger.TButton',
                       background=UI_COLOR_DANGER,
                       foreground='white',
                       font=UI_FONT_BOLD,
                       borderwidth=1,
                       relief='solid',
                       bordercolor=UI_COLOR_DANGER,
                       focuscolor=UI_COLOR_DANGER,
                       padding=(16, 8))
        style.map('Danger.TButton',
                 background=[('active', '#E74C3C'), ('pressed', '#C0392B')],
                 foreground=[('active', 'white'), ('pressed', 'white')],
                 bordercolor=[('!disabled', UI_COLOR_DANGER)])
        
        # 次要按钮 - 浅蓝边框
        style.configure('Secondary.TButton',
                       background='white',
                       foreground=btn_base,
                       font=UI_FONT_NORMAL,
                       borderwidth=2,
                       relief='solid',
                       bordercolor=btn_base,
                       focuscolor=btn_base,
                       padding=(12, 6))
        style.map('Secondary.TButton',
                 background=[('active', '#E8F4FD'), ('pressed', '#D6EAF8')],
                 foreground=[('active', btn_base), ('pressed', btn_pressed)],
                 bordercolor=[('!disabled', btn_base)])
        
        # 标签框样式 - 科技蓝+白色
        style.configure('Tech.TLabelframe',
                       background=UI_COLOR_BG_LIGHT,
                       foreground=UI_COLOR_TEXT,
                       font=UI_FONT_BOLD,
                       borderwidth=1,
                       relief='solid',
                       bordercolor=UI_COLOR_BORDER)
        style.configure('Tech.TLabelframe.Label',
                       background=UI_COLOR_BG_LIGHT,
                       foreground=UI_COLOR_PRIMARY,
                       font=UI_FONT_BOLD)
        
        # 单选按钮样式 - 科技蓝
        style.configure('Tech.TRadiobutton',
                       background=UI_COLOR_BG_LIGHT,
                       foreground=UI_COLOR_TEXT,
                       font=UI_FONT_NORMAL,
                       indicatormargin=4)
        style.map('Tech.TRadiobutton',
                 background=[('active', '#E8F4FD')],
                 foreground=[('selected', UI_COLOR_PRIMARY), ('active', UI_COLOR_PRIMARY)])
        
        # 复选框样式 - 科技蓝
        style.configure('Tech.TCheckbutton',
                       background=UI_COLOR_BG_LIGHT,
                       foreground=UI_COLOR_TEXT,
                       font=UI_FONT_NORMAL)
        style.map('Tech.TCheckbutton',
                 background=[('active', '#E8F4FD')],
                 foreground=[('selected', UI_COLOR_PRIMARY)])
        
        # 下拉框样式
        style.configure('TCombobox', 
                       font=UI_FONT_NORMAL,
                       padding=4,
                       fieldbackground='white',
                       background=UI_COLOR_BG_LIGHT)
        style.map('TCombobox',
                 fieldbackground=[('readonly', 'white'), ('disabled', '#E8F4FD')],
                 background=[('active', '#E8F4FD')])
        
        # 输入框样式
        style.configure('TEntry',
                       font=UI_FONT_NORMAL,
                       padding=4,
                       fieldbackground='white')
        
        # 进度条样式 - 科技蓝
        style.configure('Tech.Horizontal.TProgressbar',
                       background=UI_COLOR_PRIMARY,
                       troughcolor='#D6EAF8',
                       borderwidth=0,
                       lightcolor=UI_COLOR_PRIMARY,
                       darkcolor=UI_COLOR_PRIMARY)
        
        # 分隔线样式 - 浅蓝色
        style.configure('TSeparator', background=UI_COLOR_BORDER)
        
        # Notebook (选项卡) 样式 - 科技蓝
        style.configure('TNotebook',
                       background=UI_COLOR_BG_LIGHT,
                       borderwidth=0,
                       tabmargins=[4, 4, 4, 0])
        style.configure('TNotebook.Tab',
                       background='#D6EAF8',
                       foreground=UI_COLOR_TEXT,
                       font=UI_FONT_NORMAL,
                       padding=[16, 8],
                       borderwidth=0)
        style.map('TNotebook.Tab',
                 background=[('selected', 'white'), ('active', '#E8F4FD')],
                 foreground=[('selected', UI_COLOR_PRIMARY), ('active', UI_COLOR_TEXT)],
                 expand=[('selected', [0, 0, 0, 2])])
        
        # Treeview 表格样式 - 科技蓝白配色
        style.configure('Treeview',
                       font=UI_FONT_NORMAL,
                       background='white',
                       foreground=UI_COLOR_TEXT,
                       fieldbackground='white',
                       rowheight=28)
        style.configure('Treeview.Heading',
                       font=UI_FONT_BOLD,
                       background='#D6EAF8',
                       foreground=UI_COLOR_TEXT,
                       borderwidth=1,
                       relief='flat')
        style.map('Treeview',
                 background=[('selected', '#D6EAF8')],
                 foreground=[('selected', UI_COLOR_PRIMARY)])
        style.map('Treeview.Heading',
                 background=[('active', '#AED6F1')])
    
    def apply_tech_theme_to_axes(self, ax):
        """应用科技蓝+白色主题到坐标轴"""
        # 设置干净的白色背景
        ax.set_facecolor(PLOT_AX_BG)
        
        # 设置坐标轴边框 - 简洁科技风格
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_edgecolor(PLOT_SPINE_COLOR)
            ax.spines[spine].set_linewidth(1.0)
        
        # 设置刻度样式
        ax.tick_params(axis='both', colors=PLOT_TEXT_COLOR, labelsize=PLOT_FONT_BASE,
                      direction='out', length=4, width=1)
        
        # 设置网格 - 轻柔的辅助线
        ax.grid(True, linestyle='-', alpha=0.4, linewidth=0.5, color=PLOT_GRID_COLOR, zorder=0)
        ax.set_axisbelow(True)
    
    def apply_tech_theme_to_figure(self, fig, ax, title):
        """应用工业科技感主题到整个图表"""
        # 设置图表背景
        fig.patch.set_facecolor(PLOT_FIG_BG)
        
        # 应用坐标轴主题
        self.apply_tech_theme_to_axes(ax)
        
        # 设置标题样式 - 简洁专业
        ax.set_title(title, fontsize=PLOT_FONT_BASE + 2, fontweight='bold', 
                    color=PLOT_TEXT_COLOR, pad=12, loc='left')
        
        # 设置轴标签颜色和大小
        ax.set_xlabel(ax.get_xlabel(), fontsize=PLOT_FONT_BASE, fontweight='medium', color=PLOT_TEXT_COLOR)
        ax.set_ylabel(ax.get_ylabel(), fontsize=PLOT_FONT_BASE, fontweight='medium', color=PLOT_TEXT_COLOR)
        
        # 调整布局以居中对称
        fig.tight_layout(pad=1.5)
    
    def on_data_scroll_zoom(self, event):
        """数据处理标签页图表缩放 - 优化流畅度：
        - 默认：横向缩放(X轴)
        - Ctrl + 滚轮：纵向缩放(Y轴)
        - Shift + 滚轮：同时缩放(X+Y)
        """
        fig = getattr(self, '_current_preview_fig', None)
        if fig is None or not getattr(fig, 'axes', None):
            return
        if event.inaxes is None or event.inaxes not in fig.axes:
            return

        # 允许无 self.data 的情况下也能缩放（只要图已经渲染出来）
        base_ax = self.ax_data if self.ax_data in fig.axes else fig.axes[0]

        key = (getattr(event, 'key', '') or '')
        key_l = key.lower()

        zoom_x = True
        zoom_y = False
        if 'control' in key_l or 'ctrl' in key_l:
            zoom_x = False
            zoom_y = True
        if 'shift' in key_l:
            zoom_x = True
            zoom_y = True

        # 缩放倍率 - 更平滑的缩放体验
        if event.button == 'up':
            scale_factor = 0.85  # 更快的放大
        elif event.button == 'down':
            scale_factor = 1.15  # 更快的缩小
        else:
            return

        # 鼠标所在数据坐标
        xdata = event.xdata
        ydata = event.ydata

        # X轴缩放
        if zoom_x and xdata is not None:
            cur_xlim = base_ax.get_xlim()
            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0]) if (cur_xlim[1] - cur_xlim[0]) != 0 else 0.5
            new_xlim = [xdata - new_width * (1 - relx), xdata + new_width * relx]
            for ax in fig.axes:
                try:
                    ax.set_xlim(new_xlim)
                except Exception:
                    pass

        # Y轴缩放（只对当前inaxes与共享y轴的轴生效通常更合理，但这里先简单对base_ax应用）
        if zoom_y and ydata is not None:
            cur_ylim = event.inaxes.get_ylim()
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
            rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0]) if (cur_ylim[1] - cur_ylim[0]) != 0 else 0.5
            new_ylim = [ydata - new_height * (1 - rely), ydata + new_height * rely]
            try:
                event.inaxes.set_ylim(new_ylim)
            except Exception:
                pass

        try:
            self.canvas_data.draw_idle()
        except Exception:
            pass

    def on_data_pan_press(self, event):
        """处理鼠标按下事件（开始横向拖动）"""
        # 只响应鼠标左键，且在图表区域内
        if event.button != 1 or event.inaxes is None:
            return
        
        fig = getattr(self, '_current_preview_fig', None)
        if fig is None or not getattr(fig, 'axes', None):
            return
        if event.inaxes not in fig.axes:
            return
        
        # 记录起始位置
        self.is_panning = True
        self.pan_start = event.xdata

    def on_data_pan_motion(self, event):
        """处理鼠标移动事件（执行横向拖动）"""
        # 如果不在拖动状态，或者鼠标不在图表区域内，则返回
        if not self.is_panning or self.pan_start is None:
            return
        
        fig = getattr(self, '_current_preview_fig', None)
        if fig is None or not getattr(fig, 'axes', None):
            return
        if event.inaxes is None or event.inaxes not in fig.axes:
            return
        
        # 如果鼠标数据坐标为None，则返回
        if event.xdata is None:
            return
        
        # 计算鼠标在X轴方向的移动距离
        dx = event.xdata - self.pan_start
        
        # 获取基准坐标轴
        base_ax = self.ax_data if self.ax_data in fig.axes else fig.axes[0]
        
        # 获取当前X轴范围
        cur_xlim = base_ax.get_xlim()
        
        # 计算新的X轴范围（反向移动，实现拖动效果）
        new_xlim = [cur_xlim[0] - dx, cur_xlim[1] - dx]
        
        # 应用新的X轴范围到所有共享X轴的子图
        for ax in fig.axes:
            try:
                ax.set_xlim(new_xlim)
            except Exception:
                pass
        
        # 重绘图表
        try:
            self.canvas_data.draw_idle()
        except Exception:
            pass

    def on_data_pan_release(self, event):
        """处理鼠标释放事件（结束横向拖动）"""
        self.is_panning = False
        self.pan_start = None

    def _focus_preview_canvas(self, canvas_widget):
        if canvas_widget is None or not canvas_widget.winfo_exists():
            return
        if getattr(self, "_preview_canvas_widget", None) is not canvas_widget:
            return
        try:
            canvas_widget.focus_set()
        except Exception:
            pass

    def _ensure_preview_mousewheel_binding(self):
        if getattr(self, "_preview_mousewheel_bound", False):
            return
        self._preview_mousewheel_bound = True
        self.root.bind_all("<MouseWheel>", self._on_preview_mousewheel, add=True)
        self.root.bind_all("<Button-4>", self._on_preview_mousewheel, add=True)
        self.root.bind_all("<Button-5>", self._on_preview_mousewheel, add=True)

    def _on_preview_mousewheel(self, event):
        canvas_widget = getattr(self, "_preview_canvas_widget", None)
        if canvas_widget is None or not canvas_widget.winfo_exists():
            return

        try:
            widget_under_pointer = self.root.winfo_containing(
                self.root.winfo_pointerx(),
                self.root.winfo_pointery()
            )
        except Exception:
            widget_under_pointer = None

        if widget_under_pointer is not canvas_widget:
            return

        fig = getattr(self, "_current_preview_fig", None)
        if fig is None or not getattr(fig, "axes", None):
            return

        try:
            x = canvas_widget.winfo_pointerx() - canvas_widget.winfo_rootx()
            y = canvas_widget.winfo_pointery() - canvas_widget.winfo_rooty()
            canvas_h = canvas_widget.winfo_height()
            y_display = canvas_h - y
        except Exception:
            return

        target_ax = None
        for ax in fig.axes:
            try:
                bbox = ax.get_window_extent()
                if bbox.x0 <= x <= bbox.x1 and bbox.y0 <= y_display <= bbox.y1:
                    target_ax = ax
                    break
            except Exception:
                continue

        if target_ax is None:
            return

        try:
            xdata, ydata = target_ax.transData.inverted().transform((x, y_display))
        except Exception:
            return

        if hasattr(event, "delta") and event.delta:
            button = "up" if event.delta > 0 else "down"
        elif hasattr(event, "num") and event.num in (4, 5):
            button = "up" if event.num == 4 else "down"
        else:
            return

        state = getattr(event, "state", 0) or 0
        key_parts = []
        if state & 0x0004:
            key_parts.append("control")
        if state & 0x0001:
            key_parts.append("shift")
        key = "+".join(key_parts)

        class _Evt:
            pass

        evt = _Evt()
        evt.inaxes = target_ax
        evt.xdata = xdata
        evt.ydata = ydata
        evt.button = button
        evt.key = key

        self.on_data_scroll_zoom(evt)
        return "break"

    def reset_chart_view(self):
        """重置图表视图到原始范围"""
        if self.original_xlim is not None and self.original_ylim is not None:
            self.ax_actual_load.set_xlim(self.original_xlim)
            self.ax_actual_load.set_ylim(self.original_ylim)
            self.canvas_actual_load.draw()
            self.status_var_actual_load.set("图表视图已重置")
        else:
            messagebox.showinfo("提示", "没有可重置的视图范围，请先加载数据")
    
    def on_window_resize(self, event):
        """处理窗口大小变化事件 - 添加防抖动机制"""
        # 只处理主窗口的resize事件，避免子组件的resize事件
        if event.widget == self.root:
            # 取消之前的定时器
            if self._resize_timer is not None:
                self.root.after_cancel(self._resize_timer)
            # 设置新的延迟调用（300ms延迟，避免拖拽时频繁调用）
            self._resize_timer = self.root.after(300, self._do_resize)
    
    def _do_resize(self):
        """实际执行resize操作"""
        self._resize_timer = None
        self.adjust_figure_sizes()
    
    def adjust_figure_sizes(self):
        """根据当前窗口大小调整图表大小 - 让图表随容器实时放缩、尽量铺满"""
        try:
            # 优先用当前图表canvas的实际像素尺寸（比LabelFrame更准）
            w = h = 0
            if hasattr(self, 'canvas_data') and self.canvas_data is not None:
                try:
                    tw = self.canvas_data.get_tk_widget()
                    w = tw.winfo_width()
                    h = tw.winfo_height()
                except Exception:
                    w = h = 0

            if (w < 50 or h < 50) and hasattr(self, 'data_figure_frame'):
                w = self.data_figure_frame.winfo_width()
                h = self.data_figure_frame.winfo_height()

            # 容器还未完全初始化
            if w < 50 or h < 50:
                return

            # 给Tk控件边框/内边距留一点点余量（太大就会“看着缩”）
            w_px = max(10, int(w) - 6)
            h_px = max(10, int(h) - 6)

            # 1) 工艺信息分析页：当前预览图（以及缓存图）统一跟随预览区域大小
            if hasattr(self, 'figures') and self.figures:
                for fig in self.figures:
                    try:
                        dpi = float(fig.get_dpi()) if fig.get_dpi() else 100.0
                    except Exception:
                        dpi = 100.0
                    fig.set_size_inches(w_px / dpi, h_px / dpi, forward=True)
                    # tight_layout 的 pad 过大会显得“没铺满”
                    try:
                        fig.tight_layout(pad=0.6)
                    except Exception:
                        pass

            # 2) 直接重绘当前预览canvas（之前只在 fig_data 存在时重绘，导致“看起来不缩放”）
            if hasattr(self, 'canvas_data') and self.canvas_data is not None:
                self.canvas_data.draw_idle()

            # 3) 其他标签页图表（保持原逻辑，但减少边距）
            if hasattr(self, 'fig_actual_load'):
                try:
                    dpi = float(self.fig_actual_load.get_dpi()) if self.fig_actual_load.get_dpi() else 100.0
                except Exception:
                    dpi = 100.0
                self.fig_actual_load.set_size_inches(w_px / dpi, h_px / dpi, forward=True)
                try:
                    self.fig_actual_load.tight_layout(pad=0.6)
                except Exception:
                    pass
                if hasattr(self, 'canvas_actual_load'):
                    self.canvas_actual_load.draw_idle()

            # 左右布局图表维持略小比例
            steady_w_px = int(w_px * 0.92)
            steady_h_px = int(h_px * 0.92)

            def _resize_other(fig_attr, canvas_attr):
                if hasattr(self, fig_attr):
                    fig = getattr(self, fig_attr)
                    if fig is None:
                        return
                    try:
                        dpi = float(fig.get_dpi()) if fig.get_dpi() else 100.0
                    except Exception:
                        dpi = 100.0
                    fig.set_size_inches(steady_w_px / dpi, steady_h_px / dpi, forward=True)
                    try:
                        fig.tight_layout(pad=0.6)
                    except Exception:
                        pass
                    if hasattr(self, canvas_attr):
                        getattr(self, canvas_attr).draw_idle()

            _resize_other('fig_steady_time', 'canvas_steady_time')
            _resize_other('fig_steady_freq', 'canvas_steady_freq')
            _resize_other('fig_steady_n', 'canvas_steady_n')

        except Exception:
            # 静默处理异常，避免影响程序运行
            pass

    # ===== ideal_store 与 config 持久化方法 =====
    def _load_ideal_store(self):
        """从 ideal_store.json 加载已保存的 rg 配置"""
        if os.path.exists(self.ideal_store_path):
            try:
                with open(self.ideal_store_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # 键格式: "program_name|tool_no" -> (program_name, tool_no)
                self.ideal_store = {tuple(k.split("|")): v for k, v in data.items()}
                # 每次启动强制回到默认优化倍率2.0，避免沿用上次调整
                now = datetime.now().isoformat()
                for key in list(self.ideal_store.keys()):
                    self.ideal_store[key] = {"rg": 2.0, "updated_at": now}
                # 写回默认值，确保下次启动仍是2.0
                self._persist_ideal_store()
            except Exception as e:
                print(f"加载 ideal_store 失败: {e}")
                self.ideal_store = {}
        else:
            self.ideal_store = {}

    def _persist_ideal_store(self):
        """将 ideal_store 保存到 ideal_store.json"""
        try:
            data = {f"{k[0]}|{k[1]}": v for k, v in self.ideal_store.items()}
            with open(self.ideal_store_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存 ideal_store 失败: {e}")

    def _load_app_config(self):
        """从 app_config.json 加载应用配置"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.app_config = json.load(f)
                # 恢复 sample_data_dir
                if "sample_data_dir" in self.app_config:
                    self.sample_data_dir = self.app_config["sample_data_dir"]
                if isinstance(self.app_config.get("program_process_file_map"), dict):
                    self.program_process_file_map = self.app_config["program_process_file_map"]
                if isinstance(self.app_config.get("program_prompt_skip"), dict):
                    self.program_prompt_skip = self.app_config["program_prompt_skip"]
                if "tool_diameter" in self.app_config:
                    self.tool_diameter.set(str(self.app_config.get("tool_diameter", "")).strip())
                if "tool_radius" in self.app_config:
                    self.tool_radius.set(str(self.app_config.get("tool_radius", "")).strip())
                if "tool_material" in self.app_config:
                    self.workpiece_material.set(str(self.app_config.get("tool_material", "")).strip())
                if "blank_material" in self.app_config:
                    self.blank_material.set(str(self.app_config.get("blank_material", "")).strip())
            except Exception as e:
                print(f"加载 app_config 失败: {e}")
                self.app_config = {}
        else:
            self.app_config = {}

    def _persist_app_config(self):
        """将应用配置保存到 app_config.json"""
        try:
            self.app_config["sample_data_dir"] = self.sample_data_dir
            self.app_config["program_process_file_map"] = self.program_process_file_map
            self.app_config["program_prompt_skip"] = self.program_prompt_skip
            self.app_config["tool_diameter"] = str(self.tool_diameter.get()).strip()
            self.app_config["tool_radius"] = str(self.tool_radius.get()).strip()
            self.app_config["tool_material"] = str(self.workpiece_material.get()).strip()
            self.app_config["blank_material"] = str(self.blank_material.get()).strip()
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.app_config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存 app_config 失败: {e}")

    def set_status(self, msg: str, duration_ms: int = 0):
        """
        更新底部状态栏信息。
        :param msg: 状态信息
        :param duration_ms: 如果 > 0，则在指定毫秒后恢复为"就绪"
        """
        if hasattr(self, 'status_var_data'):
            self.status_var_data.set(msg)
            if duration_ms > 0:
                self.root.after(duration_ms, lambda: self.status_var_data.set("就绪"))


    def _refresh_ideal_tree(self):
        """刷新右上角"已设定理想值清单"目录树"""
        if not hasattr(self, 'ideal_tree'):
            return
        # 清空现有内容
        for item in self.ideal_tree.get_children():
            self.ideal_tree.delete(item)
        
        # 按程序名分组：仅显示当前SampleData中的程序/刀具（KISS原则）
        # ideal_store中的历史数据仅用于获取已保存的rg值，不决定显示哪些程序
        programs: Dict[str, Dict[str, Optional[Dict]]] = {}
        for prog, program_info in (self.sample_programs or {}).items():
            tools = program_info.get("tools", {})
            if not tools:
                continue
            prog_tools = programs.setdefault(prog, {})
            for tool_id in tools.keys():
                # 从ideal_store获取已保存的rg值（如果有）
                store = self.ideal_store.get((prog, tool_id))
                prog_tools[tool_id] = store
        
        # 填充树
        for prog in sorted(programs.keys()):
            prog_node = self.ideal_tree.insert("", "end", text=prog, open=True)
            tool_map = programs[prog]
            
            process_path = self.program_process_file_map.get(prog)
            has_process_file = bool(process_path and os.path.exists(process_path))
            has_processed = self._has_processed_result_for(process_path)

            # 未导入或未处理，一律提示待导入，避免显示数值
            if not has_process_file or not has_processed:
                self.ideal_tree.insert(prog_node, "end", text="⚠️ 待导入工艺信息文件",
                                       values=(prog, ""))
                continue

            # 已导入，显示各刀具的理想值
            for tool in sorted(tool_map.keys()):
                store = tool_map[tool]
                tool_label = self.format_tool_label(tool)
                if store:
                    rg = store.get("rg", 1.0)
                    # 计算理想值 = 均值 × rg
                    mean_val, _, _ = self.compute_tool_measured_mean(prog, tool)
                    if mean_val is not None:
                        ideal_val = mean_val * rg
                        display = f"{tool_label}：理想值 {ideal_val:.2f}"
                    else:
                        display = f"{tool_label}：理想值 未计算"
                else:
                    display = f"{tool_label}：理想值 未设定"
                self.ideal_tree.insert(prog_node, "end", text=display,
                                       values=(prog, tool))

    def _on_ideal_tree_select(self, event=None):
        """点击目录树条目时联动到对应程序名+刀具号"""
        if not hasattr(self, 'ideal_tree'):
            return
        item_id = self.ideal_tree.focus()
        values = self.ideal_tree.item(item_id, "values")
        if not values or len(values) < 2:
            return
        prog, tool = values[0], values[1]
        if not prog:
            return

        self.sample_program_name.set(prog)
        self.on_sample_program_selected()

        program_info = self.sample_programs.get(prog, {})
        tool_map = program_info.get("tool_display_map", {})
        display_label = None
        for label, tid in tool_map.items():
            if str(tid) == str(tool):
                display_label = label
                break
        if not display_label:
            display_label = str(tool)

        self.sample_tool_name.set(display_label)
        if self.sample_display_mode.get() != "tool":
            self.sample_display_mode.set("tool")
        self.on_sample_display_mode_change()

    def _refresh_current_ideal_display(self):
        """刷新当前选中刀具的理想值显示"""
        prog = self.get_current_program_key()
        tool = self.get_selected_tool_id()
        
        if not prog or not tool:
            return
        
        tool_label = self.format_tool_label(tool)
        
        # 检查该程序是否已导入工艺信息文件
        process_path = self.program_process_file_map.get(prog)
        has_process_file = bool(process_path and os.path.exists(process_path))
        has_processed = self._has_processed_result_for(process_path)
        if not has_process_file or not has_processed:
            self.sample_ideal_var.set("待导入")
            return
        
        store = self.ideal_store.get((prog, tool))
        if store:
            rg = store.get("rg", 1.0)
            mean_val, _, _ = self.compute_tool_measured_mean(prog, tool)
            if mean_val is not None:
                ideal_val = mean_val * rg
                self.sample_ideal_var.set(f"{ideal_val:.3f}")
            else:
                self.sample_ideal_var.set("未计算")
        else:
            self.sample_ideal_var.set("未设定")

    def _on_rg_entry_commit(self, event=None):
        """rg文本框回车/失焦时同步滑条并触发完整更新"""
        try:
            val = float(self.adjustment_ratio_display.get())
            val = max(0.1, min(5.0, val))  # 范围限制
            
            # 检查值是否有变化
            current_val = self.adjustment_ratio.get()
            if abs(val - current_val) < 0.001:
                # 值没有变化，只更新显示格式
                self.adjustment_ratio_display.set(f"{val:.2f}")
                return
            
            # 设置新值（这会触发 _on_ratio_scale_change，但只做预览更新）
            self._ratio_update_lock = True
            try:
                self.adjustment_ratio.set(val)
                self.adjustment_ratio_display.set(f"{val:.2f}")
            finally:
                self._ratio_update_lock = False
            
            # 直接触发完整更新
            self._apply_ratio_update()
        except ValueError:
            # 恢复为滑条当前值
            self.adjustment_ratio_display.set(f"{self.adjustment_ratio.get():.2f}")
            self.set_status("rg 输入无效，已恢复", 3000)

    def _on_pred_power_min_length_change(self, *args):
        """最小点数变更时触发防抖更新"""
        if getattr(self, "_loading_sample_data", False):
            return
        # 如果正在执行立即更新，跳过防抖调度避免重复刷新
        if getattr(self, "_min_length_immediate_updating", False):
            return
        self._schedule_min_length_update(immediate=False)

    def _schedule_min_length_update(self, immediate=False):
        """需求4：只在回车/失焦时触发稳态区间更新（不再使用防抖）"""
        # 取消已有的防抖 timer
        if self._min_length_debounce_timer is not None:
            try:
                self.root.after_cancel(self._min_length_debounce_timer)
            except Exception:
                pass
            self._min_length_debounce_timer = None
        
        # 需求4：只有 immediate=True（回车/失焦触发）才执行更新
        if not immediate:
            return  # 非立即触发时不做任何操作
        
        # 检查是否正在处理限制，避免重复弹框
        if getattr(self, "_min_length_validating", False):
            return
        
        # 检查是否与上次立即更新的值相同，避免重复刷新
        try:
            current_val = self.pred_power_min_length.get()
        except Exception:
            return
        if hasattr(self, "_last_min_length_val") and self._last_min_length_val == current_val:
            return
        
        # 需求5：验证最小样本点数不小于1000
        if current_val < 1000:
            self._min_length_validating = True  # 设置标志位防止重复弹框
            try:
                messagebox.showwarning("参数限制", "最小样本点数不能小于1000！\n已自动调整为1000。")
                self.pred_power_min_length.set(1000)
                current_val = 1000
            finally:
                self._min_length_validating = False
        
        self._last_min_length_val = current_val
        self._min_length_immediate_updating = True
        try:
            self._apply_min_length_update()
        finally:
            self._min_length_immediate_updating = False

    def _apply_min_length_update(self):
        """根据最小行程重新划分稳态区间并刷新图表"""
        self._min_length_debounce_timer = None
        try:
            min_val = float(self.pred_power_min_length.get())
        except Exception:
            return
        if min_val <= 0:
            return
        if not self.data:
            return
        # 更新最后执行的值，防止重复刷新
        self._last_min_length_val = self.pred_power_min_length.get()
        self.generate_plots(save=False, silent=True)
        # 刷新右侧稳态区间详情
        self._refresh_ideal_tree()
        self._refresh_current_ideal_display()

    def _set_plot_mode(self, mode: str):
        """设置显示模式（叠加/上下）"""
        self.sample_plot_mode.set(mode)
        # 更新按钮视觉状态
        if hasattr(self, 'overlay_btn') and hasattr(self, 'stacked_btn'):
            if mode == "overlay":
                self.overlay_btn.state(['pressed'])
                self.stacked_btn.state(['!pressed'])
            else:
                self.overlay_btn.state(['!pressed'])
                self.stacked_btn.state(['pressed'])
        # 刷新图表
        self.on_sample_selection_change()

    def reimport_sample_data(self):
        """
        重新导入实测数据：
        1. 确定 sample_data_dir
        2. 按 mtime 选择最新 SampleData*.csv 与 SampleData*.txt
        3. 重新解析并刷新
        """
        import glob
        
        # 确定目录
        if not self.sample_data_dir:
            dir_path = filedialog.askdirectory(title="选择 SampleData 目录")
            if not dir_path:
                self.set_status("操作取消", 3000)
                return
            self.sample_data_dir = dir_path
            self._persist_app_config()
        
        self.set_status("正在查找最新 SampleData 文件...")
        
        # 按 mtime 选择最新文件
        csv_files = glob.glob(os.path.join(self.sample_data_dir, "SampleData*.csv"))
        txt_files = glob.glob(os.path.join(self.sample_data_dir, "SampleData*.txt"))
        
        if not csv_files:
            self.set_status("未找到 SampleData*.csv 文件", 5000)
            return
        if not txt_files:
            self.set_status("未找到 SampleData*.txt 文件", 5000)
            return
        
        csv_latest = max(csv_files, key=os.path.getmtime)
        txt_latest = max(txt_files, key=os.path.getmtime)
        
        self.set_status(f"导入: {os.path.basename(csv_latest)}")
        
        # 加载数据
        success = self.load_sample_data_from_paths(csv_latest, txt_latest, silent=True, 
                                                   sample_dir=self.sample_data_dir)
        if not success:
            self.set_status("导入失败，请检查文件格式", 5000)
            return

        # 对 ideal_store 中所有条目重算理想值展示
        self._recalculate_all_ideal_values()
        self._refresh_ideal_tree()
        self._refresh_current_ideal_display()
        
        # 如果有数据则刷新图表
        if self.data:
            self.generate_plots(save=False, silent=True)
        
        self.set_status(f"重导入完成: {os.path.basename(csv_latest)}；请为程序选择工艺信息表", 5000)

    def _recalculate_all_ideal_values(self):
        """对 ideal_store 中所有条目重算理想值（使用新均值×保存的rg）"""
        for (prog, tool), store in self.ideal_store.items():
            rg = store.get("rg", 1.0)
            mean_val, _, _ = self.compute_tool_measured_mean(prog, tool)
            # 理想值在运行时计算，不需要更新 store
            # 只需要刷新显示

    def open_batch_ideal_dialog(self):
        """
        打开批量生成理想值弹窗：
        - 目录树：程序名一级、刀具二级，支持勾选
        - 批量使用当前 rg 预览值
        """
        if not self.sample_programs:
            self.set_status("请先导入 SampleData", 3000)
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("批量生成理想值")
        dialog.geometry("450x550")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 居中显示弹框
        center_dialog_on_parent(dialog, self.root)
        
        # 提示信息
        rg_val = self.adjustment_ratio.get()
        info_label = ttk.Label(dialog, text=f"将使用当前 rg = {rg_val:.2f}", 
                               font=UI_FONT_BOLD, foreground="#0066cc")
        info_label.pack(pady=10)
        
        # 目录树
        tree_frame = ttk.Frame(dialog)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        tree = ttk.Treeview(tree_frame, show="tree", selectmode="none")
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # 勾选状态
        checked = {}  # {(prog, tool): bool}
        
        def toggle_item(item_id):
            """切换勾选状态"""
            values = tree.item(item_id, "values")
            if len(values) >= 2:
                prog, tool = values[0], values[1]
                key = (prog, tool)
                checked[key] = not checked.get(key, False)
                # 更新显示
                text = tree.item(item_id, "text")
                if checked[key]:
                    tree.item(item_id, text="☑ " + text.lstrip("☐☑ "))
                else:
                    tree.item(item_id, text="☐ " + text.lstrip("☐☑ "))
            else:
                # 程序名节点：全选/全不选子节点
                prog = tree.item(item_id, "text").lstrip("☐☑ ")
                children = tree.get_children(item_id)
                all_checked = all(checked.get((prog, tree.item(c, "values")[1]), False) 
                                  for c in children if tree.item(c, "values"))
                new_state = not all_checked
                for child in children:
                    vals = tree.item(child, "values")
                    if vals:
                        checked[(vals[0], vals[1])] = new_state
                        child_text = tree.item(child, "text").lstrip("☐☑ ")
                        tree.item(child, text=("☑ " if new_state else "☐ ") + child_text)
        
        tree.bind("<Button-1>", lambda e: toggle_item(tree.identify_row(e.y)) if tree.identify_row(e.y) else None)
        
        # 填充树
        for prog in sorted(self.sample_programs.keys()):
            prog_node = tree.insert("", "end", text=prog, open=True)
            program_info = self.sample_programs[prog]
            for tool_id in sorted(program_info.get("tools", {}).keys()):
                tool_label = self.format_tool_label(tool_id)
                tree.insert(prog_node, "end", text=f"☐ {tool_label}", values=(prog, tool_id))
        
        # 按钮区
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        
        def do_batch_save():
            selected = [(k[0], k[1]) for k, v in checked.items() if v]
            if not selected:
                self.set_status("未选择任何刀具", 3000)
                dialog.destroy()
                return
            rg = self.adjustment_ratio.get()
            for prog, tool in selected:
                self.ideal_store[(prog, tool)] = {
                    "rg": rg,
                    "updated_at": datetime.now().isoformat()
                }
            self._persist_ideal_store()
            self._refresh_ideal_tree()
            self._refresh_current_ideal_display()
            self.set_status(f"已批量保存 {len(selected)} 项，override={rg:.2f}", 5000)
            dialog.destroy()
            if self.data:
                self.generate_plots(save=False, silent=True)
        
        ttk.Button(btn_frame, text="全选", width=10, 
                   command=lambda: [toggle_item(c) for c in tree.get_children()]).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="生成/保存", width=12, command=do_batch_save).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", width=10, command=dialog.destroy).pack(side=tk.LEFT, padx=5)

def optimize_memory():
    """优化内存使用和性能 - 启动加速版"""
    # 减少垃圾回收频率
    gc.set_threshold(10000, 10, 10)
    
    # 注释掉stdout/stderr重定向，避免打包后无法调试
    # sys.stderr = open(os.devnull, 'w') if not sys.stderr else sys.stderr
    # sys.stdout = open(os.devnull, 'w') if not sys.stdout else sys.stdout

def _fast_startup():
    """快速启动优化 - 禁用不必要的检查"""
    import warnings
    warnings.filterwarnings('ignore')
    
    # 预配置matplotlib以减少首次绑定延迟
    plt.rcParams['figure.max_open_warning'] = 0
    plt.rcParams['agg.path.chunksize'] = 10000

def _cleanup_on_exit():
    """程序关闭时清理所有资源和后台进程"""
    try:
        # 关闭所有matplotlib图形
        plt.close('all')
        # 强制垃圾回收
        gc.collect()
    except Exception:
        pass
    finally:
        # 确保进程完全退出
        os._exit(0)

if __name__ == "__main__":
    _fast_startup()
    optimize_memory()
    root = tk.Tk()
    
    # 提前设置窗口标题和初始大小，减少后续重绘
    root.title("🔬 铣削工艺信息分析工具 - 智能分析系统")
    root.withdraw()  # 先隐藏窗口，初始化完成后再显示
    
    # 添加一些样式美化
    style = ttk.Style()
    style.configure(
        "Accent.TButton",
        font=('Arial', 10, 'bold'),
        foreground='white',
        background=UI_COLOR_WARNING,
        borderwidth=1,
        relief='solid',
        bordercolor=UI_COLOR_WARNING,
        focuscolor=UI_COLOR_WARNING
    )
    style.map(
        "Accent.TButton",
        background=[('active', '#FB923C'), ('pressed', '#F97316'), ('disabled', '#FDBA74')],
        foreground=[('active', 'white'), ('pressed', 'white'), ('disabled', 'white')],
        bordercolor=[('!disabled', UI_COLOR_WARNING)]
    )
    
    # 创建应用实例
    app = MillingAnalysisTool(root)
    
    # 绑定窗口关闭事件，确保完全退出
    root.protocol("WM_DELETE_WINDOW", _cleanup_on_exit)
    
    # 在应用创建完成后，延迟调整图表大小
    root.after(100, app.adjust_figure_sizes)
    
    # 显示窗口
    root.deiconify()
    root.mainloop()
