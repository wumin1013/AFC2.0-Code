from __future__ import annotations

from .shared import *


class PitViewerMixin:
    """只读工艺信息查看器，不参与辨识、反解或负载预测。"""

    PIT_VIEW_FIELDS = (
        ("ap", "ap", "mm", "#1565C0"),
        ("ae", "ae", "mm", "#00897B"),
        ("feed_effective", "F", "mm/min", "#EF6C00"),
        ("MRR", "MRR", "mm³/s", "#7B1FA2"),
    )

    def init_pit_viewer_state(self):
        self.pit_status_var = tk.StringVar(value="PIT：切换到本页后加载工艺信息")
        self._pit_dataframe_cache = None
        self._pit_dataframe_cache_signature = None
        self._pit_render_signature = None
        self._pit_refresh_pending = True
        self._pit_preview_fig = None
        self._pit_preview_canvas = None
        self._pit_preview_axes = None

    def _is_pit_page_active(self):
        notebook = getattr(self, "notebook", None)
        pit_tab = getattr(self, "pit_tab", None)
        if notebook is None or pit_tab is None:
            return False
        try:
            return str(notebook.select()) == str(pit_tab)
        except Exception:
            return False

    def invalidate_pit_view(self, refresh_if_visible=True):
        """使 PIT 数据缓存失效；仅在 PIT 可见时安排重绘。"""
        self._pit_dataframe_cache_signature = None
        self._pit_render_signature = None
        self._pit_refresh_pending = True
        if not refresh_if_visible or not self._is_pit_page_active():
            return
        root = getattr(self, "root", None)
        if root is None:
            self.refresh_main_pit_preview()
            return
        try:
            root.after_idle(self.refresh_main_pit_preview)
        except Exception:
            self.refresh_main_pit_preview()

    def refresh_pit_button_state(self):
        """兼容旧调用点：PIT 已无弹窗按钮，状态变化只需使查看器失效。"""
        self.invalidate_pit_view(refresh_if_visible=True)

    def _build_pit_source_signature(self):
        data = getattr(self, "data", None)
        if not isinstance(data, list):
            data = []

        def _row_signature(row):
            if not isinstance(row, dict):
                return ()
            return tuple(
                str(row.get(key, ""))
                for key in (
                    "line_no_raw",
                    "line_no_aligned",
                    "path_start",
                    "path_end",
                    "s",
                    "ap",
                    "ae",
                    "feed_effective",
                    "MRR",
                )
            )

        return (
            id(data),
            len(data),
            int(getattr(self, "_process_model_state_version", 0) or 0),
            _row_signature(data[0]) if data else (),
            _row_signature(data[-1]) if data else (),
        )

    def _iter_pit_interval_ranges(self):
        getter = getattr(self, "_get_steady_interval_records", None)
        if not callable(getter):
            return
        for idx, interval in enumerate(getter(), 1):
            interval_id = interval.get("zone_id") or interval.get("interval_id") or f"Z{idx:03d}"
            try:
                start_idx = int(interval.get("start_idx"))
                end_idx = int(interval.get("end_idx"))
            except Exception:
                resolver = getattr(self, "_resolve_interval_process_bounds", None)
                process_bounds = resolver(interval) if callable(resolver) else None
                if not process_bounds:
                    continue
                start_idx = int(process_bounds.get("start_idx"))
                end_idx = int(process_bounds.get("end_idx"))
            if end_idx >= start_idx:
                yield interval_id, start_idx, end_idx

    @staticmethod
    def _pit_float(value, default=0.0):
        try:
            result = float(value)
        except Exception:
            return float(default)
        return result if np.isfinite(result) else float(default)

    def build_current_process_dataframe(self):
        """从当前工艺信息构建 PIT 数据；不读取实测和预测通道。"""
        data = getattr(self, "data", None)
        if not isinstance(data, list) or not data:
            return pd.DataFrame()

        interval_marks = {}
        for interval_id, start_idx, end_idx in self._iter_pit_interval_ranges() or ():
            for row_idx in range(start_idx, end_idx + 1):
                interval_marks[row_idx] = interval_id

        rows = []
        cumulative_path = 0.0
        for idx, row in enumerate(data):
            if not isinstance(row, dict) or bool(row.get("_is_synthetic_fill")):
                continue

            step_length = self._pit_float(row.get("s"), 0.0)
            path_start = self._pit_float(row.get("path_start"), cumulative_path)
            path_end = self._pit_float(row.get("path_end"), path_start + step_length)
            if path_end < path_start:
                path_end = path_start + max(step_length, 0.0)
            cumulative_path = max(cumulative_path, path_end)

            line_no = row.get("line_no_raw")
            if line_no is None:
                line_no = row.get("line_no_aligned")
            try:
                command_position = float(line_no) + 1.0
            except Exception:
                command_position = float(idx + 1)

            interval_id = interval_marks.get(idx, "")
            rows.append(
                {
                    "sample_index": int(idx),
                    "path_start": float(path_start),
                    "path_end": float(path_end),
                    "path_position": float(path_end),
                    "command_position": float(command_position),
                    "line_no_raw": row.get("line_no_raw"),
                    "line_no_aligned": row.get("line_no_aligned"),
                    "N_str": row.get("N_str"),
                    "interval_id": interval_id,
                    "is_steady": bool(interval_id),
                    "ap": self._pit_float(row.get("ap"), np.nan),
                    "ae": self._pit_float(row.get("ae"), np.nan),
                    "feed_effective": self._pit_float(row.get("feed_effective"), np.nan),
                    "MRR": self._pit_float(row.get("MRR"), np.nan),
                }
            )
        return pd.DataFrame.from_records(rows)

    def build_current_pit_dataframe(self, scope="all"):
        signature = self._build_pit_source_signature()
        if (
            signature != self._pit_dataframe_cache_signature
            or not isinstance(self._pit_dataframe_cache, pd.DataFrame)
        ):
            self._pit_dataframe_cache = self.build_current_process_dataframe()
            self._pit_dataframe_cache_signature = signature

        frame = self._pit_dataframe_cache
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return pd.DataFrame()
        if str(scope).strip().lower() == "steady":
            return frame.loc[frame["is_steady"]].reset_index(drop=True).copy()
        return frame.reset_index(drop=True).copy()

    def _get_pit_display_columns(self, pit_df):
        preferred = [
            "N_str",
            "line_no_raw",
            "path_position",
            "ap",
            "ae",
            "feed_effective",
            "MRR",
        ]
        return [column for column in preferred if column in pit_df.columns]

    def _ensure_pit_plot_canvas(self):
        if self._pit_preview_fig is not None and self._pit_preview_canvas is not None:
            return self._pit_preview_fig, self._pit_preview_axes

        container = getattr(self, "pit_plot_container", None)
        if container is None:
            return None, None
        fig, axes = plt.subplots(4, 2, figsize=(13.5, 8.0), dpi=90, squeeze=False)
        fig.patch.set_facecolor(PLOT_FIG_BG)
        fig.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.075, hspace=0.42, wspace=0.20)
        canvas = FigureCanvasTkAgg(fig, master=container)
        widget = canvas.get_tk_widget()
        widget.grid(row=0, column=0, sticky="nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self._pit_preview_fig = fig
        self._pit_preview_axes = axes
        self._pit_preview_canvas = canvas
        return fig, axes

    def _compress_pit_series(self, x_values, y_values):
        max_points = int(getattr(self, "pit_plot_max_points", 12000) or 12000)
        compressor = getattr(self, "_compress_plot_series_preserve_gaps", None)
        if callable(compressor):
            return compressor(x_values, y_values, max_points)
        if len(x_values) <= max_points:
            return x_values, y_values
        indices = np.linspace(0, len(x_values) - 1, max_points, dtype=int)
        return x_values[indices], y_values[indices]

    def _render_pit_dashboard(self, pit_df):
        fig, axes = self._ensure_pit_plot_canvas()
        if fig is None or axes is None:
            return

        path_x = pd.to_numeric(pit_df["path_position"], errors="coerce").to_numpy(dtype=float)
        command_x = pd.to_numeric(pit_df["command_position"], errors="coerce").to_numpy(dtype=float)

        for row_index, (field, label, unit, color) in enumerate(self.PIT_VIEW_FIELDS):
            values = pd.to_numeric(pit_df[field], errors="coerce").to_numpy(dtype=float)
            for column_index, (x_values, domain_title) in enumerate(
                ((path_x, "行程域"), (command_x, "指令域"))
            ):
                ax = axes[row_index][column_index]
                ax.clear()
                ax.set_axis_on()
                valid = np.isfinite(x_values) & np.isfinite(values)
                if np.any(valid):
                    plot_x, plot_y = self._compress_pit_series(x_values[valid], values[valid])
                    ax.plot(
                        plot_x,
                        plot_y,
                        color=color,
                        linewidth=1.25,
                        drawstyle="steps-post",
                        solid_capstyle="round",
                    )
                else:
                    ax.text(0.5, 0.5, "暂无有效数据", ha="center", va="center", transform=ax.transAxes)

                self.apply_plot_style(ax, grid=True)
                ax.set_ylabel(f"{label} ({unit})", fontsize=max(PLOT_FONT_BASE - 1, 9))
                if row_index == 0:
                    ax.set_title(domain_title, fontsize=PLOT_FONT_BASE + 2, fontweight="bold", color=PLOT_TEXT_COLOR)
                if row_index == len(self.PIT_VIEW_FIELDS) - 1:
                    ax.set_xlabel(
                        "累计行程 (mm)" if column_index == 0 else "程序行号 N",
                        fontsize=PLOT_FONT_BASE,
                        fontweight="bold",
                    )
                ax.margins(x=0.01)

        fig.suptitle(
            "PIT 工艺信息：ap / ae / F / MRR",
            fontsize=PLOT_FONT_BASE + 3,
            fontweight="bold",
            color=PLOT_TEXT_COLOR,
        )
        self._pit_preview_canvas.draw_idle()

    def _render_pit_empty(self):
        fig, axes = self._ensure_pit_plot_canvas()
        if fig is None or axes is None:
            return
        for ax in axes.flat:
            ax.clear()
            ax.set_axis_off()
        axes[1][0].text(
            1.0,
            0.5,
            "请先在主页面导入工艺信息文件",
            ha="center",
            va="center",
            transform=axes[1][0].transAxes,
            fontsize=PLOT_FONT_BASE + 2,
            color=UI_COLOR_TEXT_MUTED,
        )
        fig.suptitle("PIT 工艺信息查看器", fontsize=PLOT_FONT_BASE + 3, fontweight="bold")
        self._pit_preview_canvas.draw_idle()

    def refresh_main_pit_preview(self, force=False):
        if not hasattr(self, "pit_plot_container"):
            return
        if not force and not self._is_pit_page_active():
            self._pit_refresh_pending = True
            self._pit_dataframe_cache_signature = None
            return

        if force:
            self._pit_dataframe_cache_signature = None

        pit_df = self.build_current_pit_dataframe("all")
        if pit_df.empty:
            self._render_pit_empty()
            self.pit_status_var.set("PIT：请先在主页面导入工艺信息文件")
            self._pit_refresh_pending = False
            return

        render_signature = self._pit_dataframe_cache_signature
        if (
            not force
            and not self._pit_refresh_pending
            and render_signature == self._pit_render_signature
            and self._pit_preview_canvas is not None
        ):
            return

        self._render_pit_dashboard(pit_df)
        self._pit_render_signature = render_signature
        self._pit_refresh_pending = False
        self.pit_status_var.set(
            f"PIT：{len(pit_df)} 个工艺点；仅显示 ap、ae、F、MRR 的行程域与指令域图"
        )

    def resize_pit_figure_to_canvas(self):
        fig = self._pit_preview_fig
        canvas = self._pit_preview_canvas
        if fig is None or canvas is None:
            return
        try:
            widget = canvas.get_tk_widget()
            width_px = int(widget.winfo_width() or 0)
            height_px = int(widget.winfo_height() or 0)
        except Exception:
            return
        if width_px < 50 or height_px < 50:
            return
        dpi = float(fig.get_dpi() or 100.0)
        target = (width_px / dpi, height_px / dpi)
        current = fig.get_size_inches()
        if abs(current[0] - target[0]) < 0.02 and abs(current[1] - target[1]) < 0.02:
            return
        fig.set_size_inches(*target, forward=True)
        canvas.draw_idle()
