from __future__ import annotations

from .shared import *


class BootstrapUiMixin:
    def _report_view_refresh_error(self, context, exc):
        message = f"{context}失败: {str(exc)}"
        try:
            print(f"[UI][refresh] {message}")
        except Exception:
            pass
        if hasattr(self, "status_var_data"):
            try:
                self.status_var_data.set(message)
            except Exception:
                pass
        if hasattr(self, "set_status"):
            try:
                self.set_status(message, 8000)
            except Exception:
                pass

    def _schedule_smif_resize_and_refresh(self, refresh=False, refresh_pit=False):
        def _run():
            try:
                self.adjust_figure_sizes()
            except Exception as exc:
                self._report_view_refresh_error("调整图形尺寸", exc)
            if refresh_pit and hasattr(self, "refresh_main_pit_preview"):
                try:
                    self.refresh_main_pit_preview()
                except Exception as exc:
                    self._report_view_refresh_error("刷新PIT预览", exc)
            if refresh:
                try:
                    self.refresh_smif_view()
                except Exception as exc:
                    self._report_view_refresh_error("刷新SMIF视图", exc)

        if hasattr(self, "root"):
            try:
                self.root.after_idle(_run)
                return
            except Exception:
                pass
        _run()

    def _on_main_notebook_tab_changed(self, _event=None):
        try:
            selected = self.notebook.select()
        except Exception:
            return
        if str(selected) == str(self.smif_pit_tab):
            self._schedule_smif_resize_and_refresh(refresh=True, refresh_pit=True)

    def _on_smif_notebook_tab_changed(self, _event=None):
        try:
            selected = self.pit_smif_notebook.select()
        except Exception:
            return
        if str(selected) == str(self.smif_workspace_tab):
            self._schedule_smif_resize_and_refresh(refresh=True)
        else:
            self._schedule_smif_resize_and_refresh(refresh=False, refresh_pit=True)

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
        try:
            self.root.state("zoomed")
        except Exception:
            try:
                self.root.attributes("-zoomed", True)
            except Exception:
                self.root.geometry(f"{screen_width}x{screen_height}+0+0")
        
        # 初始化所有变量
        self.input_file_path = tk.StringVar()
        self.input_file_paths = []
        self.merged_input_file_path = ""
        self.input_file_count_var = tk.StringVar(value="")
        # 工艺信息和实际采样独立导入；工艺信息可以先行划分。
        self.sample_bundle_path_var = tk.StringVar(value="")
        self.matched_process_file_var = tk.StringVar(value="未绑定工艺信息表")
        self.program_process_file_map = {}
        self.program_prompt_skip = {}
        self._loading_sample_data = False
        self.sample_csv_path = None
        self.sample_txt_path = None
        self.manual_measurement_path = None
        self.manual_measurement_data = None
        self.s_base = tk.DoubleVar(value=5000.0)  # 基准转速 (rpm)
        self.k_base = tk.DoubleVar(value=1.2)    # 基准转速下的扭矩系数 (N·m/(mm³/s))
        self.k_prime = tk.DoubleVar(value=1.2)   # 电流系数K' (A/(N·m))
        self.p_idle_var = tk.DoubleVar(value=0.0)  # 当前程序空载功率
        self.kc_coeff = tk.StringVar(value="")     # 三参数模型中的K_c
        self.kc_sigma = tk.DoubleVar(value=0.0)    # K_c辨识标准差
        self.ke_coeff = tk.StringVar(value="")     # 三参数模型中的K_e
        self.lock_ke_during_identification = tk.BooleanVar(value=True)  # 默认锁定K_e，仅更新K_c
        self.lock_ke_check_text = tk.StringVar(value="√ 锁定K_e，仅辨识K_c")
        self.lock_idle_during_identification = tk.BooleanVar(value=True)  # 默认锁定全局P_idle
        self.lock_idle_check_text = tk.StringVar(value="√ 锁定P_idle，参数辨识不覆盖")
        self.kc_beta = tk.DoubleVar(value=2.0)     # K_c保守上界系数
        self.current_program_speed = tk.DoubleVar(value=0.0)
        self.current_program_idle_power = tk.DoubleVar(value=0.0)
        self.current_program_idle_power_display = tk.StringVar(value="未计算")
        self.idle_curve_visible = False
        self.gcode_nc_path_var = tk.StringVar(value="")
        self.gcode_status_var = tk.StringVar(value="未导入G代码NC（工艺信息分析可直接使用输入中的 s/S）")
        self.no_load_csv_path_var = tk.StringVar(value="")
        self.no_load_status_var = tk.StringVar(value="未辨识空载功率")
        self.step_feed_csv_path_var = tk.StringVar(value="")
        self.step_feed_status_var = tk.StringVar(value="未辨识模型参数")
        self.manual_kcke_pick_mode = False
        self.manual_kcke_points = []
        self.manual_kcke_marker_artists = []
        self.sigma_idle_var = tk.StringVar(value="未计算")
        self.delta_mrr_var = tk.StringVar(value="未计算")
        self.steady_gate_status_var = tk.StringVar(value="稳态门控: 未计算")
        self.segmentation_status_var = tk.StringVar(value="过程域六类划分: 未运行")
        self.sample_mapping_status_var = tk.StringVar(value="采样映射: 未导入实际采样文件")
        self.model_detail_collapsed = tk.BooleanVar(value=True)
        self.model_detail_toggle_text = tk.StringVar(value="展开详情")
        self.interval_detail_collapsed = tk.BooleanVar(value=True)
        self.interval_detail_toggle_text = tk.StringVar(value="展开显示设置")
        self._model_detail_widgets = []
        self._interval_detail_widgets = []
        self.idle_power_model = None
        self.idle_model_signature = ""
        self.step_feed_model_signature = ""
        self.saved_kc_profiles = {}
        self.saved_kc_profile_index = {}
        self.imported_kc_profile = None
        self.imported_kc_profile_path = ""
        self.runtime_identified_kc_profile = None
        self.runtime_identified_profile_case_signature = ""
        self.active_kc_profile = None
        self.active_kc_profile_path = ""
        self.profile_origin = "no_profile"
        self.prediction_source = "no_profile"
        self.measurement_case_signature = ""
        self._auto_identified_measurement_signatures = set()
        self.gcode_profile_bindings = {}
        self.kc_profile_status_var = tk.StringVar(value="未加载案例配置")
        self._force_recompute_kc_profile = False
        self._auto_identifying_model = False
        self.gcode_profile = None
        self.current_interval_records = []
        self.current_segment_records = []
        self.current_interval_point_kc_map = {}
        self._current_interval_ready = False
        self._current_interval_source = ""
        self._current_interval_context_signature = ""
        self._current_interval_prediction_source = "no_profile"
        self._current_interval_measurement_case_signature = ""
        self._profile_intervals_locked = False
        self._current_process_signature = ""
        self._current_mapping_signature = ""
        self._sample_mapping_status = "not_available"
        self._segmentation_sample_projection_records = []
        self._ideal_tree_interval_payloads = {}
        self._selected_interval_detail_item = ""
        self.pit_records = []
        self._process_model_state_version = 0
        self._last_process_application_context = ""
        self.smif_metric_var = tk.StringVar(value="K_c_hat")
        self.smif_view_mode_var = tk.StringVar(value="full")
        self.smif_scope_var = tk.StringVar(value="all")
        self.smif_status_var = tk.StringVar(value="未导入G代码NC（工艺信息分析可直接使用输入中的 s/S）")
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
        # 默认 1000；允许手动降到 10，兼顾短稳态段辨识
        self.pred_power_min_length = tk.IntVar(value=1)  # 工艺信息页面不再限制最小样本点
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
        self.sample_data_mode = "sampledata"
        self.sample_source_labels = ["电流", "VGpro功率", "边缘模块功率"]
        self.sample_display_mode = tk.StringVar(value="tool")  # 只使用tool模式（程序名+刀具号）
        self.sample_plot_mode = tk.StringVar(value="overlay")  # overlay/stacked
        self.process_axis_mode = tk.StringVar(value="时域+指令域")  # 时域+指令域/行程域+指令域
        # 长程序预览使用保留极值的抽稀上限；保存/导出仍使用全量点。
        self.preview_plot_max_points = 60000
        self.show_measured_curve_var = tk.BooleanVar(value=True)
        self.show_reconstructed_curve_var = tk.BooleanVar(value=True)
        self.show_feed_overlay_var = tk.BooleanVar(value=False)
        self.show_speed_overlay_var = tk.BooleanVar(value=False)
        self.show_ap_overlay_var = tk.BooleanVar(value=False)
        self.show_ae_overlay_var = tk.BooleanVar(value=False)
        self.sample_program_name = tk.StringVar()
        self.sample_tool_name = tk.StringVar()
        self.sample_avg_var = tk.StringVar(value="-")
        self.sample_ideal_var = tk.StringVar(value="-")
        self.sample_auto_status_var = tk.StringVar(value="请先导入 SampleData 或实验实测文件")
        self._selection_change_job = None  # debounce timer handle for selection changes
        self._pending_selection_signature = None
        self._last_selection_signature = None
        self._input_process_job = None
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
        self._optional_overlay_contexts = {}
        
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
        # 可变配置、运行状态、案例配置和输出均以项目根目录为基准分类存放。
        for directory in (CONFIG_DIR, RUNTIME_DATA_DIR, PROFILE_DIR, PROFILE_CACHE_DIR, OUTPUT_DIR):
            directory.mkdir(parents=True, exist_ok=True)
        self.ideal_store_path = str(RUNTIME_DATA_DIR / "ideal_store.json")
        self.config_path = str(CONFIG_DIR / "app_config.json")
        self.kc_profile_dir = str(PROFILE_DIR)
        self.kc_profile_cache_dir = str(PROFILE_CACHE_DIR)
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
        self._smif_pan_active = False
        self._smif_pan_state = None
        self._smif_pan_button = 1
        self._smif_pan_key = "alt"
        self._smif_alt_pressed = False

        if hasattr(self, "init_academic_workbench_state"):
            self.init_academic_workbench_state()

        # 创建标签页
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建工艺信息分析标签页
        self.data_processing_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_processing_tab, text="工艺信息分析")
        self.smif_pit_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.smif_pit_tab, text="PIT / SMIF")

        # 创建界面
        self.create_data_processing_tab()
        self.create_smif_pit_tab()
        # self.create_steady_state_tab()  # 已合并到工艺信息分析页
        
        # 初始化图表
        self.init_figures()
        self.optimize_processing()  # 添加性能优化
        
        # 添加窗口大小变化监听器
        self.root.bind("<Configure>", self.on_window_resize)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_main_notebook_tab_changed)
        
        # 延迟调用图表大小自适应，确保所有组件都已创建完成
        self.root.after(100, self.adjust_figure_sizes)
        # 仅保留工艺信息分析页，无需实际负载页的自适应逻辑
        self.root.after(200, self.auto_load_sample_bundle)

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

    def _set_grid_widgets_visible(self, widgets, visible):
        for widget in widgets or []:
            if widget is None:
                continue
            try:
                if visible:
                    widget.grid()
                else:
                    widget.grid_remove()
            except Exception:
                continue

    def _sync_data_controls_preview_ratio(self):
        paned = getattr(self, "_data_paned", None)
        controls = getattr(self, "_data_controls", None)
        if paned is None or controls is None:
            return
        try:
            self._adapt_data_processing_layout()
            paned.update_idletasks()
            total_h = int(paned.winfo_height())
            requested_h = int(controls.winfo_reqheight())
            if total_h <= 10 or requested_h <= 0:
                return
            minimum_preview_h = 260
            baseline_ratio = 0.38 if bool(self.model_detail_collapsed.get()) and bool(self.interval_detail_collapsed.get()) else 0.46
            baseline_h = int(total_h * baseline_ratio)
            target_h = max(requested_h + 12, baseline_h)
            target_h = max(220, min(target_h, max(total_h - minimum_preview_h, 220)))
            paned.sashpos(0, target_h)
            self.root.after_idle(self.adjust_figure_sizes)
        except Exception:
            pass

    def _adapt_data_processing_layout(self):
        controls = getattr(self, "_data_controls", None)
        detail_frame = getattr(self, "ideal_detail_frame", None)
        ideal_tree = getattr(self, "ideal_tree", None)
        if controls is None or detail_frame is None:
            return
        try:
            controls.update_idletasks()
            total_w = int(controls.winfo_width())
            if total_w <= 10:
                total_w = int(controls.winfo_reqwidth())
            if total_w <= 10:
                return
            if total_w >= 1850:
                detail_width = 900
            elif total_w >= 1600:
                detail_width = 820
            elif total_w >= 1360:
                detail_width = 720
            else:
                detail_width = 620
            controls.grid_columnconfigure(1, weight=0, minsize=detail_width)
            detail_frame.configure(width=detail_width)
            if ideal_tree is not None:
                tree_width = max(detail_width - 24, 560)
                ideal_tree.column("#0", width=max(int(tree_width * 0.18), 130), minwidth=110, stretch=True)
                grouped_widths = {
                    "process": max(int(tree_width * 0.24), 180),
                    "sample": max(int(tree_width * 0.24), 180),
                    "x_span": max(int(tree_width * 0.16), 120),
                    "summary": max(int(tree_width * 0.28), 200),
                }
                for key, width in grouped_widths.items():
                    try:
                        ideal_tree.column(key, width=width, minwidth=max(int(width * 0.7), 80), stretch=True)
                    except Exception:
                        pass
        except Exception:
            pass

    def _set_model_detail_collapsed(self, collapsed):
        collapsed = bool(collapsed)
        self.model_detail_collapsed.set(collapsed)
        self.model_detail_toggle_text.set("展开详情" if collapsed else "收起详情")
        self._set_grid_widgets_visible(getattr(self, "_model_detail_widgets", []), not collapsed)
        self.root.after_idle(self._sync_data_controls_preview_ratio)

    def _toggle_model_detail_section(self):
        self._set_model_detail_collapsed(not bool(self.model_detail_collapsed.get()))

    def _set_interval_detail_collapsed(self, collapsed):
        collapsed = bool(collapsed)
        self.interval_detail_collapsed.set(collapsed)
        self.interval_detail_toggle_text.set("展开显示设置" if collapsed else "收起显示设置")
        self._set_grid_widgets_visible(getattr(self, "_interval_detail_widgets", []), not collapsed)
        self.root.after_idle(self._sync_data_controls_preview_ratio)

    def _toggle_interval_detail_section(self):
        self._set_interval_detail_collapsed(not bool(self.interval_detail_collapsed.get()))

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
        controls.grid_columnconfigure(0, weight=1)  # 左侧控件区自适应
        controls.grid_columnconfigure(1, weight=0, minsize=620)  # 右侧详情区保留足够阅读宽度

        # ===== 左侧主控件区 =====
        left_frame = ttk.Frame(controls)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
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

        self.sample_source_buttons = []

        self.import_sample_btn = ttk.Button(
            program_frame, text="📦 导入SampleData", command=self.browse_sample_bundle,
            width=15, style='Tech.TButton'
        )
        self.import_sample_btn.grid(row=0, column=4, padx=(12, 6), sticky="w")

        self.import_experiment_btn = ttk.Button(
            program_frame, text="📡 导入实验实测", command=self.browse_experiment_measurement_file,
            width=15, style='Tech.TButton'
        )
        self.import_experiment_btn.grid(row=0, column=5, sticky="w")

        # ===== 第二行：机理辨识 =====
        model_frame = ttk.LabelFrame(left_frame, text="🧠 机理辨识", padding=8, style='Tech.TLabelframe')
        model_frame.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        model_frame.grid_columnconfigure(7, weight=1)
        model_frame.grid_columnconfigure(9, weight=1)

        model_action_frame = ttk.Frame(model_frame)
        model_action_frame.grid(row=0, column=0, columnspan=10, sticky="ew", pady=(0, 6))
        model_action_frame.grid_columnconfigure(4, weight=1)

        self.import_nc_btn = ttk.Button(
            model_action_frame, text="📄 导入G代码NC(可选)", command=self.browse_nc_file, width=16, style='Tech.TButton'
        )
        self.import_nc_btn.grid(row=0, column=0, padx=(0, 6), pady=(0, 6), sticky="w")

        self.identify_idle_btn = ttk.Button(
            model_action_frame, text="🌀 辨识空载功率", command=self.identify_no_load_power, width=14, style='Tech.TButton'
        )
        self.identify_idle_btn.grid(row=0, column=1, padx=(0, 6), pady=(0, 6), sticky="w")

        self.identify_model_btn = ttk.Button(
            model_action_frame, text="🔁 重新辨识", command=self.identify_model_parameters, width=13, style='Tech.TButton'
        )
        self.identify_model_btn.grid(row=0, column=2, padx=(0, 6), pady=(0, 6), sticky="w")
        self.identify_model_btn.bind(
            "<ButtonPress-1>",
            lambda _event: self._arm_model_param_commit_refresh_suppression(duration_seconds=1.2),
            add="+",
        )

        self.pit_display_btn = ttk.Button(
            model_action_frame, text="📋 PIT显示", command=self.show_pit_dialog, width=11, style='Secondary.TButton',
            state="disabled"
        )
        self.pit_display_btn.grid(row=0, column=3, padx=(0, 6), pady=(0, 6), sticky="w")

        self.model_detail_toggle_btn = ttk.Button(
            model_action_frame,
            textvariable=self.model_detail_toggle_text,
            command=self._toggle_model_detail_section,
            width=10,
            style='Secondary.TButton'
        )
        self.model_detail_toggle_btn.grid(row=0, column=5, padx=(6, 0), pady=(0, 6), sticky="e")

        model_option_frame = ttk.Frame(model_frame)
        model_option_frame.grid(row=1, column=0, columnspan=10, sticky="ew", pady=(0, 4))
        model_option_frame.grid_columnconfigure(2, weight=1)

        self.lock_ke_check = tk.Checkbutton(
            model_option_frame,
            textvariable=self.lock_ke_check_text,
            variable=self.lock_ke_during_identification,
            command=self.on_identification_mode_changed,
            font=UI_FONT_NORMAL,
            bg=UI_COLOR_BG_LIGHT,
            fg=UI_COLOR_TEXT,
            activebackground=UI_COLOR_BG_LIGHT,
            activeforeground=UI_COLOR_PRIMARY,
            selectcolor="white",
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
            anchor="w",
        )
        self.lock_ke_check.grid(row=0, column=0, padx=(0, 10), pady=(0, 6), sticky="w")
        self._refresh_lock_ke_check_text()

        self.lock_idle_check = tk.Checkbutton(
            model_option_frame,
            textvariable=self.lock_idle_check_text,
            variable=self.lock_idle_during_identification,
            command=self.on_identification_mode_changed,
            font=UI_FONT_NORMAL,
            bg=UI_COLOR_BG_LIGHT,
            fg=UI_COLOR_TEXT,
            activebackground=UI_COLOR_BG_LIGHT,
            activeforeground=UI_COLOR_PRIMARY,
            selectcolor="white",
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
            anchor="w",
        )
        self.lock_idle_check.grid(row=0, column=1, padx=(0, 10), pady=(0, 6), sticky="w")
        self._refresh_lock_idle_check_text()

        self.model_detail_hint_label = ttk.Label(
            model_option_frame,
            text="导入NC后自动匹配参数配置；辨识完成后可选择覆盖或另存",
            font=UI_FONT_SMALL,
            foreground=UI_COLOR_TEXT_MUTED,
        )
        self.model_detail_hint_label.grid(row=0, column=2, padx=(0, 0), pady=(0, 6), sticky="e")

        ttk.Label(model_frame, text="P_idle(W):", font=UI_FONT_NORMAL).grid(row=2, column=0, sticky="w", padx=(0, 4))
        self.p_idle_entry = ttk.Entry(model_frame, textvariable=self.p_idle_var, width=10, font=UI_FONT_NORMAL)
        self.p_idle_entry.grid(row=2, column=1, sticky="w", padx=(0, 10))
        self.p_idle_entry.bind("<Return>", self.on_model_param_commit)
        self.p_idle_entry.bind("<FocusOut>", self.on_model_param_commit)

        ttk.Label(model_frame, text="全局K_c:", font=UI_FONT_NORMAL).grid(row=2, column=2, sticky="w", padx=(0, 4))
        self.kc_entry = ttk.Entry(model_frame, textvariable=self.kc_coeff, width=10, font=UI_FONT_NORMAL)
        self.kc_entry.grid(row=2, column=3, sticky="w", padx=(0, 10))
        self.kc_entry.bind("<Return>", self.on_model_param_commit)
        self.kc_entry.bind("<FocusOut>", self.on_model_param_commit)

        ttk.Label(model_frame, text="全局K_e:", font=UI_FONT_NORMAL).grid(row=2, column=4, sticky="w", padx=(0, 4))
        self.ke_entry = ttk.Entry(model_frame, textvariable=self.ke_coeff, width=10, font=UI_FONT_NORMAL)
        self.ke_entry.grid(row=2, column=5, sticky="w", padx=(0, 10))
        self.ke_entry.bind("<Return>", self.on_model_param_commit)
        self.ke_entry.bind("<FocusOut>", self.on_model_param_commit)

        ttk.Label(model_frame, text="程序空载:", font=UI_FONT_NORMAL).grid(row=2, column=6, sticky="w", padx=(0, 4))
        self.program_idle_summary_entry = ttk.Entry(
            model_frame, textvariable=self.current_program_idle_power_display,
            width=30, font=UI_FONT_NORMAL, state="readonly"
        )
        self.program_idle_summary_entry.grid(row=2, column=7, sticky="ew", padx=(0, 6))

        self.program_idle_detail_btn = ttk.Button(
            model_frame, text="📊 查看明细", command=self.show_program_idle_detail_dialog,
            width=12, style='Secondary.TButton', state="disabled"
        )
        self.program_idle_detail_btn.grid(row=2, column=8, sticky="w")

        self.sigma_idle_label = ttk.Label(model_frame, text="σ_idle(W):", font=UI_FONT_NORMAL)
        self.sigma_idle_label.grid(row=3, column=0, sticky="w", padx=(0, 4))
        self.sigma_idle_entry = ttk.Entry(
            model_frame, textvariable=self.sigma_idle_var, width=14, font=UI_FONT_NORMAL, state="readonly"
        )
        self.sigma_idle_entry.grid(row=3, column=1, sticky="w", padx=(0, 10))

        self.delta_mrr_label = ttk.Label(model_frame, text="δ_MRR:", font=UI_FONT_NORMAL)
        self.delta_mrr_label.grid(row=3, column=2, sticky="w", padx=(0, 4))
        self.delta_mrr_entry = ttk.Entry(
            model_frame, textvariable=self.delta_mrr_var, width=14, font=UI_FONT_NORMAL, state="readonly"
        )
        self.delta_mrr_entry.grid(row=3, column=3, sticky="w", padx=(0, 10))

        self.steady_gate_status_label = ttk.Label(
            model_frame, textvariable=self.steady_gate_status_var, font=UI_FONT_SMALL,
            foreground=UI_COLOR_TEXT_MUTED
        )
        self.steady_gate_status_label.grid(row=3, column=4, columnspan=5, sticky="w", pady=(2, 0))

        self.gcode_status_label = ttk.Label(
            model_frame, textvariable=self.gcode_status_var, font=UI_FONT_SMALL,
            foreground=UI_COLOR_TEXT_MUTED
        )
        self.gcode_status_label.grid(row=4, column=0, columnspan=10, sticky="w", pady=(2, 0))
        self.no_load_status_label = ttk.Label(
            model_frame, textvariable=self.no_load_status_var, font=UI_FONT_SMALL,
            foreground=UI_COLOR_TEXT_MUTED
        )
        self.no_load_status_label.grid(row=5, column=0, columnspan=10, sticky="w", pady=(2, 0))
        self.step_feed_status_label = ttk.Label(
            model_frame, textvariable=self.step_feed_status_var, font=UI_FONT_SMALL,
            foreground=UI_COLOR_TEXT_MUTED
        )
        self.step_feed_status_label.grid(row=6, column=0, columnspan=10, sticky="w", pady=(2, 0))
        self.kc_profile_status_label = ttk.Label(
            model_frame, textvariable=self.kc_profile_status_var, font=UI_FONT_SMALL,
            foreground=UI_COLOR_TEXT_MUTED
        )
        self.kc_profile_status_label.grid(row=7, column=0, columnspan=10, sticky="w", pady=(2, 0))

        self._model_detail_widgets = [
            self.model_detail_hint_label,
            self.sigma_idle_label,
            self.sigma_idle_entry,
            self.delta_mrr_label,
            self.delta_mrr_entry,
            self.steady_gate_status_label,
            self.gcode_status_label,
            self.no_load_status_label,
            self.step_feed_status_label,
            self.kc_profile_status_label,
        ]

        self.idle_curve_frame = ttk.LabelFrame(
            model_frame, text="P_idle-S 图", padding=(6, 4), style='Tech.TLabelframe'
        )
        self.idle_curve_frame.grid(row=8, column=0, columnspan=10, sticky="ew", pady=(8, 0))
        self.idle_curve_frame.grid_columnconfigure(0, weight=1)
        self.idle_curve_frame.grid_rowconfigure(1, weight=1)

        self.idle_curve_hint_label = ttk.Label(
            self.idle_curve_frame,
            text="导入空载辨识文件后显示 P_idle-S 关系；导入 NC 后会叠加当前 G 代码转速",
            font=UI_FONT_SMALL,
            foreground=UI_COLOR_TEXT_MUTED,
        )
        self.idle_curve_hint_label.grid(row=0, column=0, sticky="w", pady=(0, 4))

        self.fig_idle_curve, self.ax_idle_curve = plt.subplots(figsize=(6.2, 2.35), dpi=90)
        self.fig_idle_curve.patch.set_facecolor(PLOT_FIG_BG)
        self.ax_idle_curve.set_facecolor(PLOT_AX_BG)
        self.fig_idle_curve.subplots_adjust(left=0.08, right=0.985, top=0.88, bottom=0.24)

        self.canvas_idle_curve = FigureCanvasTkAgg(self.fig_idle_curve, master=self.idle_curve_frame)
        idle_canvas_widget = self.canvas_idle_curve.get_tk_widget()
        idle_canvas_widget.grid(row=1, column=0, sticky="ew")
        idle_canvas_widget.configure(relief=tk.FLAT, bd=0)
        if not self.idle_curve_visible:
            self.idle_curve_frame.grid_remove()

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

        # 第一行：最小样本点(已取消)、区间数量、[弹性空间]、导入工艺信息文件、保存结果
        self.pred_power_min_length_label = ttk.Label(interval_frame, text="最小样本点(已取消):", font=UI_FONT_NORMAL)
        self.pred_power_min_length_label.grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.pred_power_min_length_entry = ttk.Entry(interval_frame, textvariable=self.pred_power_min_length,
                                                     width=10, font=UI_FONT_NORMAL)
        self.pred_power_min_length_entry.grid(row=0, column=1, padx=(0, 20), sticky="w")
        self.pred_power_min_length_entry.state(["disabled"])

        # 区间数量显示
        ttk.Label(interval_frame, text="区间数量:", font=UI_FONT_NORMAL).grid(row=0, column=2, sticky="w", padx=(0, 4))
        self.interval_count_var = tk.StringVar(value="0")
        self.interval_count_label = ttk.Label(interval_frame, textvariable=self.interval_count_var,
                                             font=UI_FONT_BOLD, foreground="#2E7D32")
        self.interval_count_label.grid(row=0, column=3, sticky="w", padx=(0, 20))

        self.interval_detail_toggle_btn = ttk.Button(
            interval_frame,
            textvariable=self.interval_detail_toggle_text,
            command=self._toggle_interval_detail_section,
            width=14,
            style='Secondary.TButton'
        )
        self.interval_detail_toggle_btn.grid(row=0, column=4, padx=(0, 8), sticky="e")

        self.run_segmentation_btn = ttk.Button(
            interval_frame,
            text="🧭 全行程六类划分",
            command=lambda: self.run_full_path_segmentation(),
            width=18,
            style='Tech.TButton',
        )
        self.run_segmentation_btn.grid(row=0, column=5, padx=(0, 8), sticky="e")

        # 导入工艺信息文件按钮（支持多选合并）
        self.choose_process_btn = ttk.Button(
            interval_frame, text="📂 导入工艺信息文件",
            command=self.choose_process_file_for_current_program,
            width=20, style='Orange.TButton'
        )
        self.choose_process_btn.grid(row=0, column=6, padx=(0, 8), sticky="e")

        # 保存结果按钮（橙色强调）- 移到最右边
        self.export_i_code_btn = ttk.Button(interval_frame, text="💾 保存结果", 
                                            command=self.save_interval_info, width=12,
                                            style='Orange.TButton',
                                            state="disabled")
        self.export_i_code_btn.grid(row=0, column=7, sticky="e")

        # 第二行：显示方式切换按钮
        plot_switch_frame = ttk.Frame(interval_frame)
        plot_switch_frame.grid(row=1, column=0, columnspan=8, pady=(8, 0), sticky="ew")
        plot_switch_frame.grid_columnconfigure(0, weight=1)
        plot_mode_row = ttk.Frame(plot_switch_frame)
        plot_mode_row.grid(row=0, column=0, sticky="w")
        overlay_row = ttk.Frame(plot_switch_frame)
        overlay_row.grid(row=1, column=0, sticky="w", pady=(6, 0))

        ttk.Label(plot_mode_row, text="图表预览:", font=UI_FONT_SMALL).pack(side=tk.LEFT, padx=(0, 6))
        self.overlay_btn = ttk.Button(plot_mode_row, text="● 叠加显示", width=12,
                                      style='Secondary.TButton', command=lambda: self._set_plot_mode("overlay"))
        self.overlay_btn.pack(side=tk.LEFT, padx=2)
        self.stacked_btn = ttk.Button(plot_mode_row, text="● 上下显示", width=12,
                                      style='Secondary.TButton', command=lambda: self._set_plot_mode("stacked"))
        self.stacked_btn.pack(side=tk.LEFT, padx=2)

        ttk.Label(plot_mode_row, text="横轴:", font=UI_FONT_SMALL).pack(side=tk.LEFT, padx=(12, 6))
        ttk.Radiobutton(
            plot_mode_row, text="时域+指令域", variable=self.process_axis_mode, value="时域+指令域",
            command=self.on_sample_selection_change, style='Tech.TRadiobutton'
        ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Radiobutton(
            plot_mode_row, text="行程域+指令域", variable=self.process_axis_mode, value="行程域+指令域",
            command=self.on_sample_selection_change, style='Tech.TRadiobutton'
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(overlay_row, text="附加曲线:", font=UI_FONT_SMALL).pack(side=tk.LEFT, padx=(0, 6))
        self.show_measured_curve_btn = tk.Checkbutton(
            overlay_row,
            text="显示实测",
            variable=self.show_measured_curve_var,
            command=self.on_sample_selection_change,
            font=UI_FONT_SMALL,
            bg=UI_COLOR_BG_LIGHT,
            fg=UI_COLOR_TEXT,
            activebackground=UI_COLOR_BG_LIGHT,
            activeforeground=UI_COLOR_PRIMARY,
            selectcolor="white",
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
            anchor="w",
        )
        self.show_measured_curve_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.show_reconstructed_curve_btn = tk.Checkbutton(
            overlay_row,
            text="显示预测负载",
            variable=self.show_reconstructed_curve_var,
            command=self.on_sample_selection_change,
            font=UI_FONT_SMALL,
            bg=UI_COLOR_BG_LIGHT,
            fg=UI_COLOR_TEXT,
            activebackground=UI_COLOR_BG_LIGHT,
            activeforeground=UI_COLOR_PRIMARY,
            selectcolor="white",
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
            anchor="w",
        )
        self.show_reconstructed_curve_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.show_feed_overlay_btn = tk.Checkbutton(
            overlay_row,
            text="显示F",
            variable=self.show_feed_overlay_var,
            command=self.on_optional_overlay_toggle,
            font=UI_FONT_SMALL,
            bg=UI_COLOR_BG_LIGHT,
            fg=UI_COLOR_TEXT,
            activebackground=UI_COLOR_BG_LIGHT,
            activeforeground=UI_COLOR_PRIMARY,
            selectcolor="white",
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
            anchor="w",
        )
        self.show_feed_overlay_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.show_speed_overlay_btn = tk.Checkbutton(
            overlay_row,
            text="显示S",
            variable=self.show_speed_overlay_var,
            command=self.on_optional_overlay_toggle,
            font=UI_FONT_SMALL,
            bg=UI_COLOR_BG_LIGHT,
            fg=UI_COLOR_TEXT,
            activebackground=UI_COLOR_BG_LIGHT,
            activeforeground=UI_COLOR_PRIMARY,
            selectcolor="white",
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
            anchor="w",
        )
        self.show_speed_overlay_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.show_ap_overlay_btn = tk.Checkbutton(
            overlay_row,
            text="显示ap",
            variable=self.show_ap_overlay_var,
            command=self.on_optional_overlay_toggle,
            font=UI_FONT_SMALL,
            bg=UI_COLOR_BG_LIGHT,
            fg=UI_COLOR_TEXT,
            activebackground=UI_COLOR_BG_LIGHT,
            activeforeground=UI_COLOR_PRIMARY,
            selectcolor="white",
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
            anchor="w",
        )
        self.show_ap_overlay_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.show_ae_overlay_btn = tk.Checkbutton(
            overlay_row,
            text="显示ae",
            variable=self.show_ae_overlay_var,
            command=self.on_optional_overlay_toggle,
            font=UI_FONT_SMALL,
            bg=UI_COLOR_BG_LIGHT,
            fg=UI_COLOR_TEXT,
            activebackground=UI_COLOR_BG_LIGHT,
            activeforeground=UI_COLOR_PRIMARY,
            selectcolor="white",
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
            anchor="w",
        )
        self.show_ae_overlay_btn.pack(side=tk.LEFT, padx=(0, 6))

        self.segmentation_status_label = ttk.Label(
            interval_frame,
            textvariable=self.segmentation_status_var,
            font=UI_FONT_SMALL,
            foreground=UI_COLOR_TEXT_MUTED,
        )
        self.segmentation_status_label.grid(
            row=2,
            column=0,
            columnspan=8,
            sticky="w",
            pady=(6, 0),
        )
        self.sample_mapping_status_label = ttk.Label(
            interval_frame,
            textvariable=self.sample_mapping_status_var,
            font=UI_FONT_SMALL,
            foreground=UI_COLOR_TEXT_MUTED,
        )
        self.sample_mapping_status_label.grid(
            row=3,
            column=0,
            columnspan=8,
            sticky="w",
            pady=(2, 0),
        )

        # 默认选择叠加显示
        self.overlay_btn.state(['pressed'])
        self.stacked_btn.state(['!pressed'])

        self._interval_detail_widgets = [
            self.pred_power_min_length_label,
            self.pred_power_min_length_entry,
            plot_switch_frame,
        ]

        # ===== 右侧：全行程六类区间详情 =====
        ideal_frame = ttk.LabelFrame(controls, text="📌 全行程六类区间详情", padding=6, style='Tech.TLabelframe')
        ideal_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        ideal_frame.configure(width=900)
        ideal_frame.grid_rowconfigure(1, weight=1)
        ideal_frame.grid_columnconfigure(0, weight=1)
        ideal_frame.grid_propagate(False)
        self.ideal_detail_frame = ideal_frame

        detail_toolbar = ttk.Frame(ideal_frame)
        detail_toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        detail_toolbar.grid_columnconfigure(0, weight=1)
        ttk.Label(
            detail_toolbar,
            text="按类别 / process / sample / x / Kc-P 显示；双击可查看完整详情。",
            font=UI_FONT_SMALL,
            foreground=UI_COLOR_TEXT_MUTED,
        ).grid(row=0, column=0, sticky="w")
        self.show_interval_detail_btn = ttk.Button(
            detail_toolbar,
            text="显示更多",
            command=self._show_selected_interval_detail_dialog,
            style="Secondary.TButton",
            width=12,
        )
        self.show_interval_detail_btn.grid(row=0, column=1, sticky="e")
        self.show_interval_detail_btn.state(["disabled"])

        detail_columns = ("process", "sample", "x_span", "summary")
        self.ideal_tree = ttk.Treeview(ideal_frame, columns=detail_columns, show="tree headings", height=20)
        self.ideal_tree.grid(row=1, column=0, sticky="nsew")
        self.ideal_tree.heading("#0", text="节点")
        self.ideal_tree.heading("process", text="process")
        self.ideal_tree.heading("sample", text="sample")
        self.ideal_tree.heading("x_span", text="x")
        self.ideal_tree.heading("summary", text="Kc / P")
        self.ideal_tree.column("#0", width=136, minwidth=110, stretch=True)
        self.ideal_tree.column("process", width=208, minwidth=168, stretch=True)
        self.ideal_tree.column("sample", width=208, minwidth=168, stretch=True)
        self.ideal_tree.column("x_span", width=148, minwidth=118, stretch=True, anchor="center")
        self.ideal_tree.column("summary", width=236, minwidth=188, stretch=True)
        ideal_scroll = ttk.Scrollbar(ideal_frame, orient=tk.VERTICAL, command=self.ideal_tree.yview)
        ideal_scroll.grid(row=1, column=1, sticky="ns")
        ideal_x_scroll = ttk.Scrollbar(ideal_frame, orient=tk.HORIZONTAL, command=self.ideal_tree.xview)
        ideal_x_scroll.grid(row=2, column=0, sticky="ew")
        self.ideal_tree.configure(yscrollcommand=ideal_scroll.set, xscrollcommand=ideal_x_scroll.set)
        self.ideal_tree.bind("<<TreeviewSelect>>", self._on_ideal_tree_select)
        self.ideal_tree.bind("<Double-1>", self._show_selected_interval_detail_dialog)
        self._refresh_ideal_tree()

        # 收集控件引用
        self.sample_control_widgets = []
        self.sample_control_widgets.extend(self.sample_source_buttons)
        self.sample_control_widgets.append(self.show_measured_curve_btn)
        self.sample_control_widgets.append(self.show_reconstructed_curve_btn)
        self.sample_control_widgets.append(self.show_feed_overlay_btn)
        self.sample_control_widgets.append(self.show_speed_overlay_btn)
        self.sample_control_widgets.append(self.show_ap_overlay_btn)
        self.sample_control_widgets.append(self.show_ae_overlay_btn)
        self.sample_control_widgets.append(self.sample_program_combo)
        self.sample_control_widgets.append(self.sample_tool_combo)
        # 过程域按钮不属于实测控件组，不得因未导入实测而禁用。
        if hasattr(self, "_refresh_import_order_controls"):
            self._refresh_import_order_controls()

        self._set_model_detail_collapsed(True)
        self._set_interval_detail_collapsed(True)
        self.root.after_idle(self._adapt_data_processing_layout)

        # ===== 预览区 =====
        preview.grid_columnconfigure(0, weight=1)
        preview.grid_rowconfigure(1, weight=1)

        nav_frame = ttk.Frame(preview)
        nav_frame.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        nav_frame.grid_columnconfigure(0, weight=1)

        self.figure_label = ttk.Label(nav_frame, text="", font=UI_FONT_LARGE, foreground=UI_COLOR_PRIMARY)
        self.figure_label.grid(row=0, column=0, sticky="w")

        self.data_figure_frame = ttk.LabelFrame(preview, text="📈 负载图预览（滚轮：横向缩放；Ctrl+滚轮：纵向缩放）", padding=4, style='Tech.TLabelframe')
        self.data_figure_frame.grid(row=1, column=0, sticky="nsew")
        self.data_figure_frame.grid_rowconfigure(0, weight=1)
        self.data_figure_frame.grid_columnconfigure(0, weight=1)
        self.data_figure_frame.bind("<Configure>", self._on_preview_canvas_configure)

        placeholder = ttk.Label(self.data_figure_frame, text="请先导入 SampleData 或实验实测文件，并生成图表",
                                foreground="#5D6D7E", anchor="center")
        placeholder.grid(row=0, column=0, sticky="nsew")


        # ===== 进度条与状态栏 =====
        progress_bar = ttk.Progressbar(
            self.data_processing_tab,
            orient=tk.HORIZONTAL,
            length=100,
            mode='determinate',
            style='Tech.Horizontal.TProgressbar'
        )
        progress_bar.grid(row=1, column=0, sticky="ew", padx=8)
        progress_bar.grid_remove()
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
                requested_h = int(controls.winfo_reqheight()) if controls.winfo_reqheight() > 0 else 0
                ctrl_h = max(requested_h + 12, int(total_h * 0.38))
                ctrl_h = max(220, min(ctrl_h, max(total_h - 260, 220)))
                paned.sashpos(0, ctrl_h)
                self.root.after_idle(self._sync_data_controls_preview_ratio)
            except Exception:
                pass

        self.root.after(60, _init_sash)
        self.on_sample_display_mode_change()
        if hasattr(self, "refresh_prediction_mode_controls"):
            self.refresh_prediction_mode_controls()
        if hasattr(self, "refresh_prediction_metrics_summary"):
            self.refresh_prediction_metrics_summary()
        if hasattr(self, "_refresh_idle_power_chart"):
            self.root.after_idle(self._refresh_idle_power_chart)

    def _build_pit_preview_panel(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=1)

        pit_toolbar = ttk.Frame(parent, padding=(6, 6, 6, 2))
        pit_toolbar.grid(row=0, column=0, sticky="ew")
        pit_toolbar.grid_columnconfigure(9, weight=1)

        ttk.Label(pit_toolbar, text="显示:", font=UI_FONT_SMALL).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            pit_toolbar, text="图", variable=self.pit_view_mode_var, value="plot",
            command=self._on_main_pit_config_changed, style='Tech.TRadiobutton'
        ).grid(row=0, column=1, sticky="w", padx=(4, 8))
        ttk.Radiobutton(
            pit_toolbar, text="表", variable=self.pit_view_mode_var, value="table",
            command=self._on_main_pit_config_changed, style='Tech.TRadiobutton'
        ).grid(row=0, column=2, sticky="w", padx=(0, 12))
        ttk.Label(pit_toolbar, text="范围:", font=UI_FONT_SMALL).grid(row=0, column=3, sticky="w")
        ttk.Radiobutton(
            pit_toolbar, text="全部工艺点", variable=self.pit_scope_var, value="all",
            command=self._on_main_pit_config_changed, style='Tech.TRadiobutton'
        ).grid(row=0, column=4, sticky="w", padx=(4, 8))
        ttk.Radiobutton(
            pit_toolbar, text="仅稳态区间", variable=self.pit_scope_var, value="steady",
            command=self._on_main_pit_config_changed, style='Tech.TRadiobutton'
        ).grid(row=0, column=5, sticky="w", padx=(0, 12))
        ttk.Label(pit_toolbar, text="横轴:", font=UI_FONT_SMALL).grid(row=0, column=6, sticky="w")
        ttk.Radiobutton(
            pit_toolbar, text="时域+指令域", variable=self.pit_axis_mode_var, value="时域+指令域",
            command=self._on_main_pit_config_changed, style='Tech.TRadiobutton'
        ).grid(row=0, column=7, sticky="w", padx=(4, 8))
        ttk.Radiobutton(
            pit_toolbar, text="行程域+指令域", variable=self.pit_axis_mode_var, value="行程域+指令域",
            command=self._on_main_pit_config_changed, style='Tech.TRadiobutton'
        ).grid(row=0, column=8, sticky="w", padx=(0, 12))
        ttk.Label(pit_toolbar, text="字段:", font=UI_FONT_SMALL).grid(row=0, column=9, sticky="w")
        self.pit_field_combo = ttk.Combobox(
            pit_toolbar, textvariable=self.pit_field_var, state="readonly", width=18, font=UI_FONT_SMALL
        )
        self.pit_field_combo.grid(row=0, column=10, sticky="w", padx=(4, 8))
        self.pit_field_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_main_pit_preview())
        ttk.Label(
            pit_toolbar,
            text="单字段单图显示，切换字段即切换图",
            font=UI_FONT_SMALL,
            foreground=UI_COLOR_TEXT_MUTED
        ).grid(row=0, column=11, sticky="w")
        ttk.Label(
            pit_toolbar,
            textvariable=self.pit_status_var,
            font=UI_FONT_SMALL,
            foreground=UI_COLOR_TEXT_MUTED
        ).grid(row=1, column=0, columnspan=10, sticky="w", pady=(4, 0))

        self.pit_content_stack = ttk.Frame(parent)
        self.pit_content_stack.grid(row=1, column=0, sticky="nsew")
        self.pit_content_stack.grid_columnconfigure(0, weight=1)
        self.pit_content_stack.grid_rowconfigure(0, weight=1)
        self.pit_plot_container = ttk.Frame(self.pit_content_stack)
        self.pit_table_container = ttk.Frame(self.pit_content_stack)

    def create_smif_pit_tab(self):
        """创建 SMIF / PIT 子页面。"""
        self.smif_pit_tab.grid_columnconfigure(0, weight=1)
        self.smif_pit_tab.grid_rowconfigure(0, weight=1)

        self.pit_smif_notebook = ttk.Notebook(self.smif_pit_tab)
        self.pit_smif_notebook.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.pit_smif_notebook.bind("<<NotebookTabChanged>>", self._on_smif_notebook_tab_changed)
        self.pit_workspace_tab = ttk.Frame(self.pit_smif_notebook)
        self.smif_workspace_tab = ttk.Frame(self.pit_smif_notebook)
        self.pit_smif_notebook.add(self.pit_workspace_tab, text="PIT")
        self.pit_smif_notebook.add(self.smif_workspace_tab, text="SMIF")

        self._build_pit_preview_panel(self.pit_workspace_tab)

        self.smif_workspace_tab.grid_columnconfigure(0, weight=1)
        self.smif_workspace_tab.grid_rowconfigure(1, weight=1)

        top_bar = ttk.Frame(self.smif_workspace_tab, padding=(0, 0, 0, 4))
        top_bar.grid(row=0, column=0, sticky="ew")
        top_bar.grid_columnconfigure(11, weight=1)

        ttk.Label(top_bar, text="SMIF 指标:", font=UI_FONT_NORMAL).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            top_bar, text="K_c_hat", variable=self.smif_metric_var, value="K_c_hat",
            command=self.refresh_smif_view, style='Tech.TRadiobutton'
        ).grid(row=0, column=1, sticky="w", padx=(6, 8))
        ttk.Radiobutton(
            top_bar, text="K_c_UCB", variable=self.smif_metric_var, value="K_c_UCB",
            command=self.refresh_smif_view, style='Tech.TRadiobutton'
        ).grid(row=0, column=2, sticky="w", padx=(0, 12))
        ttk.Label(top_bar, text="显示范围:", font=UI_FONT_NORMAL).grid(row=0, column=3, sticky="w")
        ttk.Radiobutton(
            top_bar, text="区间聚焦", variable=self.smif_view_mode_var, value="focus",
            command=self.refresh_smif_view, style='Tech.TRadiobutton'
        ).grid(row=0, column=4, sticky="w", padx=(6, 8))
        ttk.Radiobutton(
            top_bar, text="完整轨迹", variable=self.smif_view_mode_var, value="full",
            command=self.refresh_smif_view, style='Tech.TRadiobutton'
        ).grid(row=0, column=5, sticky="w", padx=(0, 12))
        ttk.Label(top_bar, text="显示内容:", font=UI_FONT_NORMAL).grid(row=0, column=6, sticky="w")
        ttk.Radiobutton(
            top_bar, text="全部显示", variable=self.smif_scope_var, value="all",
            command=self.refresh_smif_view, style='Tech.TRadiobutton'
        ).grid(row=0, column=7, sticky="w", padx=(6, 8))
        ttk.Radiobutton(
            top_bar, text="仅稳态", variable=self.smif_scope_var, value="steady",
            command=self.refresh_smif_view, style='Tech.TRadiobutton'
        ).grid(row=0, column=8, sticky="w", padx=(0, 12))
        ttk.Button(
            top_bar, text="刷新SMIF", command=self.refresh_smif_view,
            width=12, style='Secondary.TButton'
        ).grid(row=0, column=9, sticky="w")
        ttk.Label(
            top_bar, textvariable=self.smif_status_var, font=UI_FONT_SMALL,
            foreground=UI_COLOR_TEXT_MUTED
        ).grid(row=0, column=11, sticky="e")

        self.smif_plot_frame = ttk.LabelFrame(self.smif_workspace_tab, text="SMIF 轨迹视图（滚轮缩放）", padding=2, style='Tech.TLabelframe')
        self.smif_plot_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 0), pady=(0, 0))
        self.smif_plot_frame.grid_rowconfigure(0, weight=1)
        self.smif_plot_frame.grid_columnconfigure(0, weight=1)

        self.fig_smif = plt.figure(figsize=(8, 6), dpi=80)
        self.fig_smif.patch.set_facecolor(SMIF_FIG_BG)
        self.ax_smif = None
        self.ax_smif_xy = None
        self.ax_smif_xz = None
        self.ax_smif_metric = None
        self._smif_colorbar_ax = None
        self.canvas_smif = FigureCanvasTkAgg(self.fig_smif, master=self.smif_plot_frame)
        smif_canvas_widget = self.canvas_smif.get_tk_widget()
        try:
            smif_canvas_widget.configure(background=SMIF_FIG_BG, highlightthickness=0, borderwidth=0)
        except Exception:
            pass
        smif_canvas_widget.grid(row=0, column=0, sticky="nsew")
        smif_canvas_widget.bind("<Enter>", lambda _e: smif_canvas_widget.focus_set())
        smif_canvas_widget.bind("<FocusOut>", lambda _e: setattr(self, "_smif_alt_pressed", False))
        smif_canvas_widget.bind("<MouseWheel>", self.on_smif_widget_mousewheel)
        smif_canvas_widget.bind("<Button-4>", self.on_smif_widget_mousewheel)
        smif_canvas_widget.bind("<Button-5>", self.on_smif_widget_mousewheel)
        self.smif_plot_frame.bind(
            "<Configure>",
            lambda _e: self._schedule_smif_resize_and_refresh(refresh=False),
            add="+",
        )
        self.canvas_smif.mpl_connect('scroll_event', self.on_smif_scroll_zoom)
        self.canvas_smif.mpl_connect('button_press_event', self.on_smif_pan_press)
        self.canvas_smif.mpl_connect('motion_notify_event', self.on_smif_pan_motion)
        self.canvas_smif.mpl_connect('button_release_event', self.on_smif_pan_release)
        self.canvas_smif.mpl_connect('key_press_event', self.on_smif_key_press)
        self.canvas_smif.mpl_connect('key_release_event', self.on_smif_key_release)
        self.smif_pit_tree = None
        has_smif_source = bool(hasattr(self, "_has_smif_trajectory_source") and self._has_smif_trajectory_source())
        if not has_smif_source:
            try:
                self._draw_smif_empty_placeholder()
                self.canvas_smif.draw_idle()
            except Exception:
                pass

        initial_refresh_delay = 0 if has_smif_source else 200
        self.root.after(initial_refresh_delay, self._deferred_smif_first_refresh)
        if hasattr(self, "refresh_main_pit_preview"):
            self.refresh_main_pit_preview()

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
        ax.grid(False)
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
        self._adapt_data_processing_layout()
        self._sync_data_controls_preview_ratio()
        self.adjust_figure_sizes()

    def _deferred_smif_first_refresh(self):
        """延迟首次 SMIF 刷新，确保画布已完成布局。"""
        try:
            self._sync_smif_figure_to_canvas()
        except Exception:
            pass
        try:
            self.refresh_smif_view()
        except Exception:
            pass

    def _sync_smif_figure_to_canvas(self):
        """将 SMIF figure 尺寸同步到画布实际像素大小，并重新定位 axes。"""
        fig = getattr(self, 'fig_smif', None)
        canvas = getattr(self, 'canvas_smif', None)
        if fig is None or canvas is None:
            return
        try:
            if hasattr(self, "root") and self.root is not None:
                self.root.update_idletasks()
            widget = canvas.get_tk_widget()
            width_px = int(widget.winfo_width() or 0)
            height_px = int(widget.winfo_height() or 0)
        except Exception:
            return
        if width_px < 50 or height_px < 50:
            return
        try:
            dpi = float(fig.get_dpi()) if fig.get_dpi() else 100.0
        except Exception:
            dpi = 100.0
        current_width_px = float(fig.get_figwidth()) * dpi
        current_height_px = float(fig.get_figheight()) * dpi
        if abs(current_width_px - width_px) < 2.0 and abs(current_height_px - height_px) < 2.0:
            return
        fig.set_size_inches(width_px / dpi, height_px / dpi, forward=True)
        try:
            fig.subplots_adjust(left=0.015, right=0.955, top=0.985, bottom=0.035)
        except Exception:
            pass
        try:
            canvas.draw_idle()
        except Exception:
            pass

    def adjust_figure_sizes(self):
        """根据当前窗口大小调整图表大小 - 让图表随容器实时放缩、尽量铺满"""
        try:
            def _resize_figure_by_canvas(fig_attr, canvas_attr, pad_px=0, apply_tight_layout=True):
                if not hasattr(self, fig_attr) or not hasattr(self, canvas_attr):
                    return
                fig = getattr(self, fig_attr)
                canvas = getattr(self, canvas_attr)
                if fig is None or canvas is None:
                    return
                try:
                    widget = canvas.get_tk_widget()
                    width_px = widget.winfo_width()
                    height_px = widget.winfo_height()
                except Exception:
                    return
                if width_px < 50 or height_px < 50:
                    return
                width_px = max(10, int(width_px) - int(pad_px))
                height_px = max(10, int(height_px) - int(pad_px))
                try:
                    dpi = float(fig.get_dpi()) if fig.get_dpi() else 100.0
                except Exception:
                    dpi = 100.0
                fig.set_size_inches(width_px / dpi, height_px / dpi, forward=True)
                if apply_tight_layout:
                    try:
                        fig.tight_layout(pad=0.6)
                    except Exception:
                        pass
                canvas.draw_idle()

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

            has_main_preview_size = w >= 50 and h >= 50
            if has_main_preview_size:
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

            _resize_figure_by_canvas('fig_idle_curve', 'canvas_idle_curve', pad_px=4, apply_tight_layout=False)

            # 4) SMIF 3D 图：同步 figure 尺寸到画布并重新定位 axes
            self._sync_smif_figure_to_canvas()
            if hasattr(self, 'canvas_smif') and self.canvas_smif is not None:
                self.canvas_smif.draw_idle()

        except Exception:
            # 静默处理异常，避免影响程序运行
            pass
