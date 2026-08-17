from __future__ import annotations

from .shared import *


class SampleManagerMixin:
    _OPTIONAL_OVERLAY_SPINE_STEP_POINTS = 56.0
    _OPTIONAL_OVERLAY_EDGE_PADDING_POINTS = 64.0

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
        show_measured_curve = bool(getattr(self, "show_measured_curve_var", tk.BooleanVar(value=True)).get())
        if context_blocks:
            drew_series = False
            if show_measured_curve and invalid_blocks:
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
                drew_series = True
            if show_measured_curve and valid_blocks:
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
                drew_series = True
            elif show_measured_curve and context_blocks:
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
                drew_series = True
            if not drew_series:
                ax.text(0.5, 0.5, "当前未启用任何主曲线显示", ha='center', va='center',
                        transform=ax.transAxes, fontsize=PLOT_FONT_BASE, color='#666666')
            ax.set_axisbelow(True)
            ax.grid(False)
        else:
            ax.text(0.5, 0.5, "当前选择无实测数据", ha='center', va='center',
                    transform=ax.transAxes, fontsize=PLOT_FONT_BASE, color='#666666')
            ax.grid(False)

        ax.set_title('实测负载预览', fontsize=PLOT_FONT_BASE, fontweight='bold', color=PLOT_TEXT_COLOR, pad=12)
        ax.set_xlabel('时间 (ms)', fontsize=PLOT_FONT_BASE, fontweight='bold', color=PLOT_TEXT_COLOR)
        ax.set_ylabel("曲线值", fontsize=PLOT_FONT_BASE, fontweight='bold', color=PLOT_TEXT_COLOR)
        ax.tick_params(labelsize=PLOT_FONT_BASE, colors=PLOT_TEXT_COLOR)

        aux_axes = self.plot_optional_measurement_overlays(ax, sample_time_indices_all, context_mask, valid_mask)

        if context_mask.any():
            context_x = sample_time_indices_all[context_mask]
            x_min = float(np.min(context_x))
            x_max = float(np.max(context_x))
            x_range = x_max - x_min if x_max > x_min else 1
            ax.set_xlim(x_min - x_range * 0.02, x_max + x_range * 0.02)
            self.apply_line_axis_on_time(ax, context_mask)

        self._apply_optional_overlay_legend(
            ax,
            [ax],
            aux_axes,
            loc='upper left',
            fontsize=PLOT_FONT_BASE,
            framealpha=0.9,
            shadow=False,
            facecolor=PLOT_AX_BG,
            edgecolor=PLOT_SPINE_COLOR,
            linewidth=0.8,
        )

        self._register_optional_overlay_context(
            fig,
            parent_ax=ax,
            x_values=sample_time_indices_all,
            context_mask=context_mask,
            valid_mask=valid_mask,
            legend_host=ax,
            legend_base_axes=[ax],
            aux_axes=aux_axes,
            legend_style={
                "loc": "upper left",
                "fontsize": PLOT_FONT_BASE,
                "framealpha": 0.9,
                "shadow": False,
                "facecolor": PLOT_AX_BG,
                "edgecolor": PLOT_SPINE_COLOR,
                "linewidth": 0.8,
            },
            subplot_adjust={
                "left": 0.06,
                "top": 0.88,
                "bottom": 0.09,
            },
        )
        self.figures = [fig]
        self.figure_names = ["实测负载预览"]
        self.show_current_figure(0)

    def get_sample_data_source_name(self):
        """获取实测数据源名称"""
        labels = list(getattr(self, "sample_source_labels", []) or [])
        default_labels = ["电流", "VGpro功率", "边缘模块功率"]
        while len(labels) < len(default_labels):
            labels.append(default_labels[len(labels)])
        source_idx = int(self.sample_data_source.get())
        if 0 <= source_idx < len(labels):
            return labels[source_idx]
        return default_labels[0]

    def _get_optional_process_geometry_values(self):
        """获取与当前样本逐点对齐的程序工艺量。"""
        if not self.sample_data_loaded or self.sample_data_line_numbers is None or not self.data:
            return {}

        raw_lines = np.asarray(self.sample_data_line_numbers, dtype=int)
        if raw_lines.size == 0:
            return {}

        try:
            process_df = self._build_aligned_process_geometry_frame(raw_lines)
        except Exception:
            return {}

        if len(process_df) != raw_lines.size:
            return {}

        overlays = {}
        for source_key, overlay_key in (
            ("ap", "ap"),
            ("ae", "ae"),
            ("feed_plan", "feed"),
            ("speed_plan", "speed"),
        ):
            if source_key not in process_df.columns:
                continue
            values = pd.to_numeric(
                process_df[source_key],
                errors="coerce",
            ).to_numpy(dtype=float)
            if values.shape[0] == raw_lines.size:
                overlays[overlay_key] = values
        return overlays

    def _optional_curve_enabled(self, variable_name, default=False):
        variable = getattr(self, variable_name, None)
        if variable is None:
            return bool(default)
        try:
            return bool(variable.get())
        except Exception:
            return bool(default)

    def get_optional_process_overlays(self, point_labels):
        """获取仅工艺信息视图可选显示的 F、S、ap 和 ae 曲线。"""
        frame = (
            point_labels.copy()
            if isinstance(point_labels, pd.DataFrame)
            else pd.DataFrame(point_labels)
        )
        if frame.empty:
            return []

        def _numeric_column(*names):
            for name in names:
                if name in frame:
                    return pd.to_numeric(
                        frame[name],
                        errors="coerce",
                    ).to_numpy(dtype=float)
            return np.full(len(frame), np.nan, dtype=float)

        speed_values = _numeric_column("S", "S_program", "spindle_speed")
        if not np.any(np.isfinite(speed_values)):
            source_indices = pd.to_numeric(
                frame.get("source_index", pd.Series(range(len(frame)))),
                errors="coerce",
            ).to_numpy(dtype=float)
            process_rows = list(getattr(self, "data", None) or [])
            speed_values = np.full(len(frame), np.nan, dtype=float)
            for output_index, source_index in enumerate(source_indices):
                if not np.isfinite(source_index):
                    continue
                source_index = int(source_index)
                if not 0 <= source_index < len(process_rows):
                    continue
                row = process_rows[source_index]
                if not isinstance(row, dict):
                    continue
                for key in ("S", "S_program", "spindle_speed"):
                    try:
                        value = float(row.get(key))
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(value):
                        speed_values[output_index] = value
                        break

        overlays = []
        feed_values = _numeric_column("F_program", "F", "feed_effective")
        if (
            self._optional_curve_enabled("show_feed_overlay_var")
            and np.any(np.isfinite(feed_values))
        ):
            overlays.append({
                "values": feed_values,
                "label": "F(程序进给)",
                "unit": "mm/min",
                "color": "#E67E22",
                "linestyle": "--",
            })
        if (
            self._optional_curve_enabled("show_speed_overlay_var")
            and np.any(np.isfinite(speed_values))
        ):
            overlays.append({
                "values": speed_values,
                "label": "S(主轴转速)",
                "unit": "r/min",
                "color": "#16A085",
                "linestyle": "-.",
            })

        for key, variable_name, label, color, linestyle in (
            ("ap", "show_ap_overlay_var", "ap(切深)", "#7E57C2", "-"),
            ("ae", "show_ae_overlay_var", "ae(切宽)", "#C0392B", "--"),
        ):
            values = _numeric_column(key)
            if (
                self._optional_curve_enabled(variable_name)
                and np.any(np.isfinite(values))
            ):
                overlays.append({
                    "values": values,
                    "label": label,
                    "unit": "mm",
                    "color": color,
                    "linestyle": linestyle,
                })
        return overlays

    def get_optional_measurement_overlays(self):
        """获取负载图上可选显示的附加曲线。"""
        overlays = []
        values = None
        if self.sample_data_values is not None:
            values = np.asarray(self.sample_data_values, dtype=float)
            if values.ndim != 2:
                values = None

        if getattr(self, "sample_data_mode", "") == "experiment_measurement" and values is not None:
            source_idx = int(self.sample_data_source.get())
            if (
                values.shape[1] > 2
                and self._optional_curve_enabled("show_feed_overlay_var")
                and source_idx != 2
            ):
                overlays.append({
                    "values": values[:, 2],
                    "label": "F(实际进给)",
                    "unit": "mm/min",
                    "color": "#E67E22",
                    "linestyle": "--",
                })
            if (
                values.shape[1] > 1
                and self._optional_curve_enabled("show_speed_overlay_var")
                and source_idx != 1
            ):
                overlays.append({
                    "values": values[:, 1],
                    "label": "S(实际转速)",
                    "unit": "rpm",
                    "color": "#16A085",
                    "linestyle": "-.",
                })

        process_overlays = self._get_optional_process_geometry_values()
        existing_labels = {str(item.get("label") or "") for item in overlays}
        if (
            self._optional_curve_enabled("show_feed_overlay_var")
            and "feed" in process_overlays
            and "F(实际进给)" not in existing_labels
        ):
            overlays.append({
                "values": process_overlays["feed"],
                "label": "F(程序进给)",
                "unit": "mm/min",
                "color": "#E67E22",
                "linestyle": "--",
            })
        if (
            self._optional_curve_enabled("show_speed_overlay_var")
            and "speed" in process_overlays
            and "S(实际转速)" not in existing_labels
        ):
            overlays.append({
                "values": process_overlays["speed"],
                "label": "S(主轴转速)",
                "unit": "r/min",
                "color": "#16A085",
                "linestyle": "-.",
            })
        if self._optional_curve_enabled("show_ap_overlay_var") and "ap" in process_overlays:
            overlays.append({
                "values": process_overlays["ap"],
                "label": "ap(切深)",
                "unit": "mm",
                "color": "#7E57C2",
                "linestyle": "-",
            })
        if self._optional_curve_enabled("show_ae_overlay_var") and "ae" in process_overlays:
            overlays.append({
                "values": process_overlays["ae"],
                "label": "ae(切宽)",
                "unit": "mm",
                "color": "#C0392B",
                "linestyle": "--",
            })
        return overlays

    def plot_optional_measurement_overlays(
        self,
        parent_ax,
        x_values,
        context_mask,
        valid_mask=None,
        overlays=None,
    ):
        """在当前图上叠加附加曲线，返回创建的辅助坐标轴。"""
        if overlays is None:
            overlays = self.get_optional_measurement_overlays()
        if not overlays or x_values is None or context_mask is None:
            return []

        x_arr = np.asarray(x_values, dtype=float)
        context_arr = np.asarray(context_mask, dtype=bool)
        valid_arr = None if valid_mask is None else np.asarray(valid_mask, dtype=bool)
        max_points = getattr(self, "_current_plot_max_points", None)
        aux_axes = []

        for idx, overlay in enumerate(overlays):
            values = np.asarray(overlay["values"], dtype=float)
            if values.shape[0] != x_arr.shape[0]:
                continue

            overlay_mask = context_arr & np.isfinite(x_arr) & np.isfinite(values)
            if valid_arr is not None and valid_arr.shape[0] == overlay_mask.shape[0] and np.any(valid_arr):
                overlay_mask &= valid_arr
            blocks = self.compute_contiguous_blocks(overlay_mask)
            if not blocks:
                continue

            aux_ax = parent_ax.twinx()
            aux_ax.set_facecolor('none')
            aux_ax.patch.set_visible(False)
            aux_ax.grid(False)
            aux_ax.set_zorder(parent_ax.get_zorder() + idx + 1)
            try:
                aux_ax.spines["right"].set_position((
                    "outward",
                    self._OPTIONAL_OVERLAY_SPINE_STEP_POINTS * idx,
                ))
            except Exception:
                pass
            self.apply_plot_style(aux_ax, grid=False, text_color=overlay["color"], transparent=True)
            aux_ax.spines['right'].set_color(overlay["color"])
            aux_ax.tick_params(labelsize=PLOT_FONT_BASE, colors=overlay["color"])
            aux_ax.set_ylabel(
                f"{overlay['label']} ({overlay['unit']})",
                fontsize=PLOT_FONT_BASE,
                fontweight='bold',
                color=overlay["color"]
            )
            aux_ax.yaxis.labelpad = 10 + idx * 2
            self.plot_series_by_blocks(
                aux_ax,
                x_arr,
                values,
                blocks,
                max_points=max_points,
                color=overlay["color"],
                linewidth=1.1,
                linestyle=overlay["linestyle"],
                alpha=0.9,
                label=overlay["label"],
                zorder=8 + idx
            )
            aux_axes.append(aux_ax)

        return aux_axes

    def _resolve_optional_overlay_right_margin(
        self,
        fig,
        overlay_count,
        base_right=0.94,
        left=0.07,
    ):
        overlay_count = max(0, int(overlay_count or 0))
        base_right = float(base_right)
        if fig is None or overlay_count <= 0:
            return base_right

        try:
            figure_width_points = float(fig.get_figwidth()) * 72.0
        except Exception:
            return base_right
        if not np.isfinite(figure_width_points) or figure_width_points <= 0:
            return base_right

        reserved_points = (
            self._OPTIONAL_OVERLAY_EDGE_PADDING_POINTS
            + self._OPTIONAL_OVERLAY_SPINE_STEP_POINTS * (overlay_count - 1)
        )
        responsive_right = 1.0 - reserved_points / figure_width_points
        minimum_right = min(base_right, float(left) + 0.20)
        return min(base_right, max(minimum_right, responsive_right))

    def _apply_optional_overlay_layout(self, fig, overlay_count, subplot_adjust=None):
        if fig is None:
            return None

        layout = dict(getattr(fig, "_optional_overlay_layout", {}) or {})
        incoming = dict(subplot_adjust or {})
        if "right" in incoming:
            layout["base_right"] = incoming.pop("right")
        layout.update(incoming)
        layout.setdefault("left", fig.subplotpars.left)
        layout.setdefault("base_right", 0.94)
        layout.setdefault("top", fig.subplotpars.top)
        layout.setdefault("bottom", fig.subplotpars.bottom)
        layout.setdefault("hspace", fig.subplotpars.hspace)
        layout["overlay_count"] = max(0, int(overlay_count or 0))

        right_margin = (
            self._resolve_optional_overlay_right_margin(
                fig,
                layout["overlay_count"],
                base_right=layout["base_right"],
                left=layout["left"],
            )
            if layout["overlay_count"] > 0
            else float(layout["base_right"])
        )
        fig.subplots_adjust(
            left=float(layout["left"]),
            right=right_margin,
            top=float(layout["top"]),
            bottom=float(layout["bottom"]),
            hspace=float(layout["hspace"]),
        )
        fig._optional_overlay_layout = layout
        return layout

    def _apply_optional_overlay_legend(self, legend_host, legend_base_axes, aux_axes, **style):
        if legend_host is None:
            return None

        existing_legend = legend_host.get_legend()
        if existing_legend is not None:
            try:
                existing_legend.remove()
            except Exception:
                pass

        handles = []
        labels = []
        for base_ax in list(legend_base_axes or []):
            if base_ax is None:
                continue
            h_base, l_base = base_ax.get_legend_handles_labels()
            handles.extend(h_base)
            labels.extend(l_base)
        for aux_ax in list(aux_axes or []):
            if aux_ax is None:
                continue
            h_aux, l_aux = aux_ax.get_legend_handles_labels()
            handles.extend(h_aux)
            labels.extend(l_aux)

        filtered = []
        seen_labels = set()
        for handle, label in zip(handles, labels):
            normalized = str(label or "").strip()
            if not normalized or normalized == "_nolegend_" or normalized in seen_labels:
                continue
            seen_labels.add(normalized)
            filtered.append((handle, normalized))
        if not filtered:
            return None

        handles, labels = zip(*filtered)
        legend_kwargs = dict(style)
        facecolor = legend_kwargs.pop("facecolor", None)
        edgecolor = legend_kwargs.pop("edgecolor", None)
        linewidth = legend_kwargs.pop("linewidth", None)

        legend = legend_host.legend(handles, labels, **legend_kwargs)
        frame = legend.get_frame()
        if facecolor is not None:
            frame.set_facecolor(facecolor)
        if edgecolor is not None:
            frame.set_edgecolor(edgecolor)
        if linewidth is not None:
            frame.set_linewidth(linewidth)
        return legend

    def _register_optional_overlay_context(
        self,
        fig,
        parent_ax,
        x_values,
        context_mask,
        valid_mask=None,
        legend_host=None,
        legend_base_axes=None,
        aux_axes=None,
        legend_style=None,
        subplot_adjust=None,
    ):
        if fig is None or parent_ax is None or x_values is None or context_mask is None:
            return

        contexts = getattr(self, "_optional_overlay_contexts", None)
        if contexts is None:
            contexts = {}
            self._optional_overlay_contexts = contexts

        contexts[id(fig)] = {
            "fig": fig,
            "parent_ax": parent_ax,
            "x_values": np.asarray(x_values, dtype=float),
            "context_mask": np.asarray(context_mask, dtype=bool),
            "valid_mask": None if valid_mask is None else np.asarray(valid_mask, dtype=bool),
            "legend_host": legend_host or parent_ax,
            "legend_base_axes": list(legend_base_axes or [parent_ax]),
            "aux_axes": list(aux_axes or []),
            "legend_style": dict(legend_style or {}),
            "subplot_adjust": dict(subplot_adjust or {}),
        }
        self._apply_optional_overlay_layout(
            fig,
            len(aux_axes or []),
            subplot_adjust,
        )

    def refresh_optional_measurement_overlays(self, fig=None, redraw=True):
        fig = fig or getattr(self, "_current_preview_fig", None)
        if fig is None:
            return False

        contexts = getattr(self, "_optional_overlay_contexts", {}) or {}
        context = contexts.get(id(fig))
        if not context:
            return False

        parent_ax = context.get("parent_ax")
        legend_host = context.get("legend_host") or parent_ax
        if parent_ax is None or legend_host is None:
            return False
        if parent_ax not in getattr(fig, "axes", []):
            return False

        for aux_ax in list(context.get("aux_axes") or []):
            if aux_ax is None:
                continue
            try:
                aux_ax.remove()
            except Exception:
                try:
                    fig.delaxes(aux_ax)
                except Exception:
                    pass

        aux_axes = self.plot_optional_measurement_overlays(
            parent_ax,
            context.get("x_values"),
            context.get("context_mask"),
            context.get("valid_mask"),
        )
        context["aux_axes"] = list(aux_axes or [])

        self._apply_optional_overlay_legend(
            legend_host,
            context.get("legend_base_axes"),
            aux_axes,
            **dict(context.get("legend_style") or {}),
        )

        adjust_style = dict(context.get("subplot_adjust") or {})
        self._apply_optional_overlay_layout(
            fig,
            len(aux_axes),
            adjust_style,
        )

        if redraw and getattr(self, "_current_preview_fig", None) is fig and hasattr(self, "canvas_data") and self.canvas_data:
            self.canvas_data.draw_idle()
        return True

    def refresh_sample_source_labels(self):
        """刷新实测数据源单选按钮名称。"""
        labels = list(getattr(self, "sample_source_labels", []) or [])
        default_labels = ["电流", "VGpro功率", "边缘模块功率"]
        while len(labels) < len(default_labels):
            labels.append(default_labels[len(labels)])
        self.sample_source_labels = labels[:3]
        if hasattr(self, "sample_source_buttons"):
            for idx, button in enumerate(self.sample_source_buttons):
                if idx >= len(self.sample_source_labels):
                    break
                try:
                    button.configure(text=self.sample_source_labels[idx])
                except Exception:
                    continue

    def _build_manual_measurement_programs(self, file_path, line_numbers):
        """为手动导入的实验实测文件构造最小程序/区间结构。"""
        line_arr = np.asarray(line_numbers, dtype=int)
        if line_arr.size == 0:
            raise ValueError("实验实测文件中没有可用的程序行号")

        program_name = os.path.splitext(os.path.basename(file_path))[0].strip() or "实验实测"
        start_line = int(np.min(line_arr))
        end_line = int(np.max(line_arr))
        tool_id = "全部数据"
        tool_range = [(start_line, end_line)]
        return {
            program_name: {
                "program_number": "__manual__",
                "tools": {tool_id: tool_range},
                "tool_raw_ranges": {tool_id: tool_range},
                "tool_display_ranges": {tool_id: tool_range},
                "tool_display_map": {self.build_tool_display_label(tool_id, tool_range): tool_id},
            }
        }

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
                interval_policy = self._get_default_interval_policy() if hasattr(self, "_get_default_interval_policy") else "fresh_or_empty"
                self.generate_plots(save=False, silent=True, interval_policy=interval_policy)
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
                if not self._has_authoritative_segmentation_state():
                    if self._process_current_input_for_preview():
                        return
                else:
                    refresher = getattr(self, "_refresh_segmentation_sample_projection", None)
                    if callable(refresher):
                        refresher(refresh_view=False, silent=True)
            else:
                if not self._has_authoritative_segmentation_state():
                    self.set_input_files([])
                    self.prompt_process_file_for_program(program_name)
                else:
                    # 工艺信息先导入时程序名尚不可知，通常暂记为“当前工艺”。
                    # 后导入实际文件后直接用已有权威过程域结果做映射，不要求
                    # 用户重复选择同一份工艺信息文件；映射自身会校验覆盖与歧义。
                    current_primary = self.get_primary_input_file()
                    if hasattr(self, "matched_process_file_var"):
                        current_name = (
                            os.path.basename(current_primary)
                            if current_primary
                            else "当前过程域结果"
                        )
                        self.matched_process_file_var.set(
                            f"使用已划分工艺信息: {current_name}"
                        )
                    refresher = getattr(
                        self,
                        "_refresh_segmentation_sample_projection",
                        None,
                    )
                    if callable(refresher):
                        refresher(refresh_view=False, silent=True)
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

    def build_sample_selection_signature(self):
        return (
            self.sample_display_mode.get(),
            self.sample_program_name.get(),
            self.sample_tool_name.get(),
            self.sample_plot_mode.get(),
            self.sample_data_source.get(),
            self._optional_curve_enabled("show_measured_curve_var", True),
            self._optional_curve_enabled("show_reconstructed_curve_var", True),
            self._optional_curve_enabled("show_feed_overlay_var"),
            self._optional_curve_enabled("show_speed_overlay_var"),
            self._optional_curve_enabled("show_ap_overlay_var"),
            self._optional_curve_enabled("show_ae_overlay_var"),
            self._optional_curve_enabled("show_interval_state_var", True),
            self._optional_curve_enabled("show_prediction_load_var"),
        )

    def _cancel_pending_sample_selection_change(self):
        if self._selection_change_job:
            try:
                self.root.after_cancel(self._selection_change_job)
            except Exception:
                pass
            self._selection_change_job = None
        try:
            sig = self.build_sample_selection_signature()
        except Exception:
            sig = None
        self._pending_selection_signature = sig
        self._last_selection_signature = sig
        return sig

    def on_optional_overlay_toggle(self):
        """附加曲线勾选时尽量只更新副轴和图例，避免整图重绘。"""
        sig = self._cancel_pending_sample_selection_change()

        current_fig = getattr(self, "_current_preview_fig", None)
        contexts = getattr(self, "_optional_overlay_contexts", {}) or {}
        visible_figs = set(getattr(self, "figures", []) or [])
        refreshed = False
        stale_keys = []

        for key, context in list(contexts.items()):
            overlay_fig = context.get("fig")
            if overlay_fig is None:
                stale_keys.append(key)
                continue
            if overlay_fig not in visible_figs and overlay_fig is not current_fig:
                stale_keys.append(key)
                continue
            refreshed = self.refresh_optional_measurement_overlays(
                fig=overlay_fig,
                redraw=(overlay_fig is current_fig),
            ) or refreshed

        for key in stale_keys:
            contexts.pop(key, None)

        if refreshed:
            return
        self.on_sample_selection_change()

    def on_sample_selection_change(self, event=None):
        """实测数据显示条件变化时刷新图表（加入去抖，防止频繁切换卡顿）"""
        if bool(getattr(self, "_loading_sample_data", False)):
            return
        try:
            # 记录当前选择签名，快速跳过重复请求
            sig = self.build_sample_selection_signature()
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
                interval_policy = self._get_default_interval_policy() if hasattr(self, "_get_default_interval_policy") else "fresh_or_empty"
                self.generate_plots(save=False, silent=True, interval_policy=interval_policy)

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
            interval_policy = self._get_default_interval_policy() if hasattr(self, "_get_default_interval_policy") else "fresh_or_empty"
            self.generate_plots(save=False, silent=True, interval_policy=interval_policy)

    def prompt_sample_data_source(self):
        """数据源选项已移除，直接使用当前默认数据源自动加载。"""
        if getattr(self, "_sample_source_prompt_shown", False):
            return
        self._sample_source_prompt_shown = True
        current_value = self.sample_data_source.get()
        default_value = current_value if current_value in (0, 1, 2) else 1
        self.sample_data_source.set(default_value)
        self.load_sample_data(silent=False)

    def _find_files_case_insensitive(self, directory, filename):
        """返回目录中与目标文件名大小写无关匹配的全部普通文件。"""
        if not directory or not filename:
            return []
        target = str(filename).casefold()
        matches = []
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.name.casefold() == target and entry.is_file():
                        matches.append(os.path.join(directory, entry.name))
        except OSError:
            return []
        return sorted(matches, key=lambda path: os.path.basename(path))

    def _find_file_case_insensitive(self, directory, filename):
        """兼容旧调用：唯一匹配时返回路径，冲突时不任意选择。"""
        matches = self._find_files_case_insensitive(directory, filename)
        return matches[0] if len(matches) == 1 else None

    def _find_file_exact(self, directory, filename):
        """仅返回文件名逐字符完全匹配的普通文件。"""
        if not directory or not filename:
            return None
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.name == filename and entry.is_file():
                        return os.path.join(directory, entry.name)
        except OSError:
            return None
        return None

    def _validate_sampledata_input_file(self, file_path, display_name):
        """在解析前确认文件存在、非空且当前可读取。"""
        if not file_path or not os.path.isfile(file_path):
            raise ValueError(f"缺少 {display_name}")
        try:
            if os.path.getsize(file_path) <= 0:
                raise ValueError(f"{display_name} 为空文件")
            with open(file_path, "rb") as stream:
                if not stream.read(1):
                    raise ValueError(f"{display_name} 为空文件")
        except ValueError:
            raise
        except OSError as exc:
            raise ValueError(f"{display_name} 当前不可读取或仍被占用：{exc}") from exc

    def resolve_sampledata_files(self, base_dir, strict_root=False):
        """
        在目录中按逐字符精确文件名定位 SampleData.csv / SampleData.txt。
        兼容两种放置方式：
        1) 与工艺信息表同目录
        2) 放在同目录下的 SampleData 子目录
        """
        self._sampledata_resolution_error = ""
        if not base_dir:
            self._sampledata_resolution_error = "未提供 SampleData 目录"
            return None, None, None
        candidates = [base_dir]
        if not strict_root:
            candidates.append(os.path.join(base_dir, "SampleData"))
        for directory in candidates:
            if not os.path.isdir(directory):
                continue
            csv_path = self._find_file_exact(directory, "SampleData.csv")
            txt_path = self._find_file_exact(directory, "SampleData.txt")
            if csv_path and txt_path:
                try:
                    self._validate_sampledata_input_file(csv_path, "SampleData.csv")
                    self._validate_sampledata_input_file(txt_path, "SampleData.txt")
                except ValueError as exc:
                    self._sampledata_resolution_error = str(exc)
                    return None, None, None
                return directory, csv_path, txt_path
            if csv_path or txt_path:
                missing = "SampleData.txt" if csv_path else "SampleData.csv"
                self._sampledata_resolution_error = f"文件对不完整：缺少 {missing}"
                if strict_root:
                    return None, None, None
        self._sampledata_resolution_error = "未找到完整的 SampleData.csv 和 SampleData.txt 文件对"
        return None, None, None

    def load_sample_data_from_paths(self, csv_path, txt_path, silent=False, sample_dir=None):
        """从明确路径加载实测数据 SampleData.csv 与 SampleData.txt（不依赖工艺信息表路径）"""
        if not csv_path or not txt_path:
            if bool(getattr(self, "release_mode", False)) and hasattr(self, "reset_sample_data_state"):
                self.reset_sample_data_state()
            return False
        if not os.path.exists(csv_path) or not os.path.exists(txt_path):
            if bool(getattr(self, "release_mode", False)) and hasattr(self, "reset_sample_data_state"):
                self.reset_sample_data_state()
            if hasattr(self, "sample_auto_status_var"):
                self.sample_auto_status_var.set("未找到SampleData.csv或SampleData.txt")
            if hasattr(self, "status_var_data"):
                self.status_var_data.set("未发现实测数据文件，已跳过导入")
            return False

        try:
            self._validate_sampledata_input_file(csv_path, "SampleData.csv")
            self._validate_sampledata_input_file(txt_path, "SampleData.txt")
        except ValueError as exc:
            try:
                self.reset_sample_data_state()
            except Exception:
                pass
            reason = str(exc)
            if not silent:
                messagebox.showerror("SampleData读取失败", reason)
            if hasattr(self, "sample_auto_status_var"):
                self.sample_auto_status_var.set(reason)
            if hasattr(self, "status_var_data"):
                self.status_var_data.set(reason)
            return False

        if sample_dir is None:
            sample_dir = os.path.dirname(csv_path)
        self.sample_csv_path = csv_path
        self.sample_txt_path = txt_path
        if hasattr(self, "_invalidate_measurement_runtime_state"):
            self._invalidate_measurement_runtime_state(keep_profile_lock=True)
        self.manual_measurement_path = None
        self.manual_measurement_data = None
        self.measurement_case_signature = ""
        if hasattr(self, "_clear_runtime_identified_profile_state"):
            self._clear_runtime_identified_profile_state(clear_active=True, reason="switch_to_sampledata")
        self.manual_kcke_pick_mode = False
        self.manual_kcke_points = []
        if hasattr(self, "_clear_manual_kcke_markers"):
            self._clear_manual_kcke_markers(clear_points=False, redraw=False)
        if hasattr(self, "_update_manual_kcke_button_text"):
            self._update_manual_kcke_button_text()
        self.sample_data_mode = "sampledata"
        if bool(getattr(self, "release_mode", False)):
            self.sample_data_source.set(1)
        self.sample_source_labels = ["电流", "VGpro功率", "边缘模块功率"]
        self.refresh_sample_source_labels()
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
            if not self.sample_programs or not any(
                bool((program_info or {}).get("tools"))
                for program_info in self.sample_programs.values()
            ):
                raise ValueError("SampleData.txt 未解析到有效程序或刀具范围")
            df = pd.read_csv(csv_path, header=None, usecols=[0, 1, 2, 3, 4], dtype={4: str})
            if df.shape[1] < 5:
                if not silent:
                    messagebox.showerror("格式错误", "SampleData.csv 列数不足")
                if hasattr(self, "sample_auto_status_var"):
                    self.sample_auto_status_var.set("SampleData.csv格式错误")
                if hasattr(self, "reset_sample_data_state"):
                    self.reset_sample_data_state()
                return False

            values = df.iloc[:, 0:3].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
            line_numbers = pd.to_numeric(df.iloc[:, 3], errors='coerce').to_numpy(dtype=float)
            program_numbers = df.iloc[:, 4].astype(str).to_numpy()
            valid_mask = ~np.isnan(line_numbers)
            values = values[valid_mask]
            program_numbers = program_numbers[valid_mask]
            line_numbers = line_numbers[valid_mask].astype(int)
            if line_numbers.size == 0:
                raise ValueError("SampleData.csv 未解析到有效采样行")
            negative_value_mask = np.isfinite(values) & (values < 0)
            corrected_count = int(np.sum(negative_value_mask))
            if corrected_count > 0:
                values = np.abs(values)

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
            projection_refresher = getattr(self, "_refresh_segmentation_sample_projection", None)
            if callable(projection_refresher):
                projection_refresher(
                    refresh_view=self._has_authoritative_segmentation_state(),
                    silent=True,
                )
            self._refresh_ideal_tree()
            self._refresh_current_ideal_display()

            if self.raw_to_aligned_line_map:
                if hasattr(self, "status_var_data"):
                    status_text = f"实测数据已加载并对齐: {os.path.basename(csv_path)}"
                    if corrected_count > 0:
                        status_text += f" | 负值已取绝对值 {corrected_count} 点"
                    self.status_var_data.set(status_text)
                if hasattr(self, "sample_auto_status_var"):
                    status_text = "实测数据已导入并对齐"
                    if corrected_count > 0:
                        status_text += f" | 负值已取绝对值 {corrected_count} 点"
                    self.sample_auto_status_var.set(status_text)
            else:
                if hasattr(self, "status_var_data"):
                    status_text = f"实测数据已加载，待处理后对齐: {os.path.basename(csv_path)}"
                    if corrected_count > 0:
                        status_text += f" | 负值已取绝对值 {corrected_count} 点"
                    self.status_var_data.set(status_text)
                if hasattr(self, "sample_auto_status_var"):
                    status_text = "实测数据已导入，待对齐"
                    if corrected_count > 0:
                        status_text += f" | 负值已取绝对值 {corrected_count} 点"
                    self.sample_auto_status_var.set(status_text)

            if not self.data:
                self.show_sample_preview()
            return True
        except Exception as e:
            try:
                self.reset_sample_data_state()
            except Exception:
                pass
            if not silent:
                messagebox.showerror("加载失败", f"读取实测数据时发生错误:\n{str(e)}")
            if hasattr(self, "sample_auto_status_var"):
                self.sample_auto_status_var.set(f"实测数据导入失败：{str(e)}")
            if hasattr(self, "status_var_data"):
                self.status_var_data.set(f"实测数据导入失败：{str(e)}；工艺信息仍可独立分析")
            return False
        finally:
            self._loading_sample_data = False

    def _clear_process_context_for_measurement_reimport(self, program_name):
        """重新导入实测时只撤销采样映射，保留工艺信息与过程域结果。"""
        self._last_process_application_context = ""

    def load_experiment_measurement_file(self, file_path, silent=False):
        """加载手动导入的实验实测文件。"""
        if not file_path or not os.path.exists(file_path):
            if hasattr(self, "reset_sample_data_state"):
                self.reset_sample_data_state()
            reason = "未提供实验实测文件" if not file_path else "实验实测文件不存在"
            if hasattr(self, "sample_auto_status_var"):
                self.sample_auto_status_var.set(f"{reason}，旧采样状态已清空")
            if hasattr(self, "status_var_data"):
                self.status_var_data.set(f"{reason}，旧采样状态已清空")
            return False

        is_measurement_reimport = bool(getattr(self, "sample_data_loaded", False))
        self._loading_sample_data = True
        try:
            if hasattr(self, "sample_auto_status_var"):
                self.sample_auto_status_var.set("正在导入实验实测文件...")
            if hasattr(self, "status_var_data"):
                self.status_var_data.set("正在导入实验实测文件...")
            try:
                self.root.update_idletasks()
            except Exception:
                pass

            measurement = self.parse_channel_data_file(file_path)
            line_numbers = np.asarray(measurement["program_line"], dtype=int)
            values = np.column_stack([
                np.asarray(measurement["actual_load"], dtype=float),
                np.asarray(measurement["actual_spindle_speed"], dtype=float),
                np.asarray(measurement["actual_feed_speed"], dtype=float),
            ])
            if len(values) != len(line_numbers) or len(line_numbers) == 0:
                raise ValueError("实验实测文件解析后的通道长度不一致")

            program_name = os.path.splitext(os.path.basename(file_path))[0].strip() or "实验实测"
            program_numbers = np.asarray(["__manual__"] * len(line_numbers), dtype=object)
            if is_measurement_reimport:
                self._clear_process_context_for_measurement_reimport(program_name)
            if hasattr(self, "_invalidate_measurement_runtime_state"):
                self._invalidate_measurement_runtime_state(keep_profile_lock=True)

            self.sample_programs = self._build_manual_measurement_programs(file_path, line_numbers)
            self.sample_csv_path = None
            self.sample_txt_path = None
            self.manual_measurement_path = file_path
            self.manual_measurement_data = measurement
            self.manual_kcke_pick_mode = False
            self.manual_kcke_points = []
            if hasattr(self, "_clear_manual_kcke_markers"):
                self._clear_manual_kcke_markers(clear_points=False, redraw=False)
            if hasattr(self, "_update_manual_kcke_button_text"):
                self._update_manual_kcke_button_text()
            self.sample_data_mode = "experiment_measurement"
            self.sample_source_labels = ["实际负载", "实际转速", "合成进给"]
            self.refresh_sample_source_labels()
            self.sample_data_source.set(0)
            if hasattr(self, "_sync_measurement_case_state"):
                self._sync_measurement_case_state(measurement, reason="load_experiment_measurement_file")

            sample_dir = os.path.dirname(file_path)
            self.sample_data_dir = sample_dir
            if hasattr(self, "sample_bundle_path_var"):
                self.sample_bundle_path_var.set(sample_dir)

            self.sample_data_values_raw = values
            self.sample_data_program_numbers_raw = program_numbers
            self.sample_data_line_numbers_raw = line_numbers
            self.sample_data_values = values
            self.sample_data_program_numbers = program_numbers
            self.sample_data_line_numbers = line_numbers
            self.sample_data_loaded = True

            self.align_sample_data_to_processed()
            current_process_path = str(self.get_primary_input_file() or "").strip()
            if current_process_path and os.path.exists(current_process_path):
                self.program_process_file_map.setdefault(program_name, current_process_path)
            self.update_sample_program_options()
            self.on_sample_display_mode_change()
            projection_refresher = getattr(self, "_refresh_segmentation_sample_projection", None)
            if callable(projection_refresher):
                projection_refresher(
                    refresh_view=self._has_authoritative_segmentation_state(),
                    silent=True,
                )
            self._refresh_ideal_tree()
            self._refresh_current_ideal_display()

            line_min = int(np.min(line_numbers))
            line_max = int(np.max(line_numbers))
            corrected_count = int(measurement.get("negative_load_corrected_count", 0) or 0)
            status_text = (
                f"实验实测已导入: {os.path.basename(file_path)} | 程序 {program_name} | "
                f"{len(line_numbers)} 点 | 行号 {line_min}-{line_max}"
            )
            if corrected_count > 0:
                status_text += f" | 负载负值已取绝对值 {corrected_count} 点"
            if is_measurement_reimport:
                status_text += " | 已保留过程域划分并重新建立采样映射"
            mapping_status = str(getattr(self, "_sample_mapping_status", "") or "")
            if mapping_status == "valid":
                status_text += " | 采样映射成功"
            elif mapping_status == "failed":
                status_text += " | 采样映射失败，过程域结果未变"
            elif self.data:
                status_text += " | 采样映射待建立"
            if hasattr(self, "sample_auto_status_var"):
                self.sample_auto_status_var.set(status_text)
            if hasattr(self, "status_var_data"):
                self.status_var_data.set(status_text)
            self._persist_app_config()

            if not self.data:
                self.show_sample_preview()
            return True
        except Exception as e:
            # 失败后不得继续暴露上一文件或半写入的采样上下文，避免后续
            # 六态投影/导出误用陈旧端点。解析和派生状态统一回到空状态。
            try:
                self.reset_sample_data_state()
            except Exception:
                self.sample_data_loaded = False
                self.sample_data_mode = "sampledata"
                self.manual_measurement_path = None
                self.manual_measurement_data = None
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
                invalidator = getattr(
                    self,
                    "_invalidate_segmentation_sample_projection",
                    None,
                )
                if callable(invalidator):
                    invalidator(reason="实验实测文件导入失败")
            if not silent:
                messagebox.showerror("加载失败", f"读取实验实测文件时发生错误:\n{str(e)}")
            if hasattr(self, "sample_auto_status_var"):
                self.sample_auto_status_var.set("实验实测文件导入失败")
            if hasattr(self, "status_var_data"):
                self.status_var_data.set("实验实测文件导入失败")
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
        """为当前程序选择工艺信息表。"""
        program_name = self.sample_program_name.get().strip() or "当前工艺"
        if hasattr(self, "_ensure_ready_for_process_info_import") and not self._ensure_ready_for_process_info_import():
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
            process_layout = None
            
            for line in content[:min(100, len(content))]:
                line = line.strip()
                if not line:
                    continue

                parsed, process_layout = self.parse_gcode_line(line, layout_hint=process_layout, return_layout=True)
                if not parsed:
                    continue

                gcode_part = str(parsed.get("gcode_content", "") or "")
                if any(code in gcode_part for code in ("G", "M", "X", "Y", "Z", "S", "F")):
                    valid_lines += 1
            
            if valid_lines < 3:
                return False, "文件格式不正确：未检测到有效的工艺信息行（需包含ap/ae/F及G代码等）"
            
            return True, ""
            
        except Exception as e:
            return False, f"验证文件时发生错误: {str(e)}"

    def choose_process_file_for_program(self, program_name, source_label="工艺信息文件"):
        """为指定程序选择并绑定工艺信息文件，支持多选后按序号合并。"""
        program_name = (program_name or "").strip()
        if not program_name:
            return False
        file_paths = list(filedialog.askopenfilenames(
            title=f"选择 {program_name} 的{source_label}",
            filetypes=(("工艺信息文件", "*.txt *.csv"), ("文本文件", "*.txt"), ("CSV文件", "*.csv"), ("所有文件", "*.*"))
        ))
        if not file_paths:
            return False

        for file_path in file_paths:
            is_valid, error_msg = self.validate_process_info_file(file_path)
            if not is_valid:
                messagebox.showerror("文件格式错误", f"所选文件不是有效的{source_label}:\n{error_msg}")
                return False

        effective_input_path = self.set_input_files(file_paths)
        if not effective_input_path:
            return False

        self.program_process_file_map[program_name] = effective_input_path
        self._persist_app_config()
        if len(file_paths) == 1:
            status_detail = os.path.basename(effective_input_path)
        else:
            status_detail = f"{len(file_paths)} 个文件已合并为 {os.path.basename(effective_input_path)}"
        self.set_status(
            f"已绑定{source_label}: {status_detail}；保留最近一次 P_idle / K_e / 全局K_c，如需刷新请点击“辨识参数”",
            5000
        )
        if hasattr(self, "_refresh_import_order_controls"):
            self._refresh_import_order_controls()
        # 为当前程序的所有刀具设置默认优化倍率2.0
        self._set_default_rg_for_program(program_name)
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
        resolved_current = os.path.normcase(os.path.abspath(current_primary)) if current_primary else ""
        resolved_target = os.path.normcase(os.path.abspath(process_path))
        apply_context = (
            self._build_process_application_context_signature(program_name, process_path)
            if hasattr(self, "_build_process_application_context_signature")
            else f"{program_name}|{resolved_target}"
        )
        previous_context = str(getattr(self, "_last_process_application_context", "") or "")
        if resolved_current != resolved_target:
            self.set_input_files([process_path])
        elif apply_context != previous_context:
            if getattr(self, "sample_data_mode", "") == "experiment_measurement":
                if hasattr(self, "_sync_measurement_case_state"):
                    self._sync_measurement_case_state(
                        getattr(self, "manual_measurement_data", None),
                        reason="apply_process_file_same_path_case_switch",
                    )
                if hasattr(self, "_invalidate_measurement_runtime_state"):
                    keep_profile_lock = bool(getattr(self, "prediction_source", "no_profile") == "imported_profile")
                    self._invalidate_measurement_runtime_state(
                        keep_profile_lock=keep_profile_lock,
                        clear_interval_state=True,
                    )
                if hasattr(self, "_invalidate_process_alignment_caches"):
                    self._invalidate_process_alignment_caches(reason="apply_process_file_same_path_case_switch")
        self._last_process_application_context = apply_context
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
            if not self._has_authoritative_segmentation_state():
                self.set_input_files([])
            else:
                invalidator = getattr(self, "_invalidate_segmentation_sample_projection", None)
                if callable(invalidator):
                    invalidator(reason=f"程序 {program_name} 尚未绑定工艺信息")
            if hasattr(self, "matched_process_file_var"):
                self.matched_process_file_var.set(f"未绑定工艺信息表: {program_name}")
            return False
        if not os.path.exists(process_path):
            if not self._has_authoritative_segmentation_state():
                self.set_input_files([])
            else:
                invalidator = getattr(self, "_invalidate_segmentation_sample_projection", None)
                if callable(invalidator):
                    invalidator(reason=f"程序 {program_name} 绑定的工艺信息表不存在")
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
            sample_dir = getattr(self, "sample_data_dir", None)
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
        """获取指定程序+刀具的稳态区间行点范围。"""
        intervals = self.collect_line_point_intervals_for_tool(program_name, tool_id)
        range_strings = []
        for interval_text in intervals:
            range_text = str(interval_text).split(":", 1)[0].strip()
            if range_text:
                range_strings.append(range_text)
        return range_strings

    def export_sample_intervals(self):
        """导出负载区间交互文件 SampleData.rg"""
        if not self._get_current_interval_records(allow_profile_fallback=False):
            messagebox.showwarning("无区间", "请先生成稳态区间")
            return
        if not self.sample_programs:
            messagebox.showwarning("无程序信息", "请先加载 SampleData.txt")
            return
        primary_file = self.get_primary_input_file()
        # 源码研究版写项目 output/；冻结发布版直接写 EXE 同目录。
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_dir = str(OUTPUT_DIR)
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
                        interval_text = ",".join(intervals)
                        f.write(f"{program_name};{tool_id};{interval_text};\n")
            self.status_var_data.set(f"区间文件已导出: {os.path.basename(output_path)}")
        except Exception as e:
            messagebox.showerror("导出失败", f"导出区间文件时发生错误:\n{str(e)}")
