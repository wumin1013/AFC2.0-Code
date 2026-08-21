from __future__ import annotations

import csv
from pathlib import Path

from matplotlib.collections import PolyCollection

from .shared import *


class AnalysisExportMixin:
    def _default_segmentation_output_dir(self):
        """发布 EXE 直接写同目录；源码研究版保留 output/segmentation。"""
        if bool(getattr(self, "release_mode", False)) and IS_FROZEN:
            return Path(OUTPUT_DIR)
        return Path(OUTPUT_DIR) / "segmentation"

    def _clear_segmentation_output_artifacts(self, output_dir=None, *, scope="all"):
        """按过程域/映射域清理固定导出，避免实测变化误删过程结果。"""
        target_dir = (
            Path(output_dir)
            if output_dir is not None
            else self._default_segmentation_output_dir()
        )
        if not target_dir.exists():
            return
        process_file_names = (
            "intervals.csv",
            "overview.png",
            "diagnostics.json",
            "point_labels.csv",
            ".intervals.tmp.csv",
            ".point_labels.tmp.csv",
            ".overview.tmp.png",
            ".diagnostics.tmp.json",
            ".intervals.bak.csv",
            ".point_labels.bak.csv",
            ".overview.bak.png",
            ".diagnostics.bak.json",
        )
        mapping_file_names = (
            "sample_projection.csv",
            "sample_overview.png",
            "sample_mapping_diagnostics.json",
            ".sample_projection.tmp.csv",
            ".sample_overview.tmp.png",
            ".sample_mapping_diagnostics.tmp.json",
            ".sample_projection.bak.csv",
            ".sample_overview.bak.png",
            ".sample_mapping_diagnostics.bak.json",
        )
        normalized_scope = str(scope or "all").strip().lower()
        if normalized_scope == "process":
            file_names = process_file_names
        elif normalized_scope == "mapping":
            file_names = mapping_file_names
        elif normalized_scope == "all":
            file_names = (*process_file_names, *mapping_file_names)
        else:
            raise ValueError(f"未知六态导出清理范围: {scope}")
        errors = []
        for file_name in file_names:
            path = target_dir / file_name
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                errors.append(f"{file_name}: {exc}")
        if errors:
            raise OSError("；".join(errors))

    def _coerce_segmentation_dataframe(self, result, attribute_name):
        payload = getattr(result, attribute_name, None)
        if isinstance(payload, pd.DataFrame):
            return payload.copy()
        if isinstance(payload, (list, tuple)):
            return pd.DataFrame(list(payload))
        if isinstance(payload, dict):
            return pd.DataFrame(payload)
        raise TypeError(f"SegmentationResult.{attribute_name} must be a pandas DataFrame")

    def _write_dataframe_csv_compat(self, frame, output_path):
        """兼容 pandas 1.4 的 line_terminator 与新版 lineterminator。"""
        kwargs = {
            "index": False,
            "encoding": "utf-8",
            "lineterminator": "\n",
        }
        try:
            frame.to_csv(output_path, **kwargs)
        except TypeError as exc:
            if "lineterminator" not in str(exc):
                raise
            kwargs.pop("lineterminator")
            kwargs["line_terminator"] = "\n"
            frame.to_csv(output_path, **kwargs)

    def _normalize_segmentation_json_value(self, value):
        if isinstance(value, dict):
            return {
                str(key): self._normalize_segmentation_json_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [self._normalize_segmentation_json_value(item) for item in value]
        if isinstance(value, np.ndarray):
            return [self._normalize_segmentation_json_value(item) for item in value.tolist()]
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            numeric = float(value)
            return numeric if np.isfinite(numeric) else None
        if isinstance(value, Path):
            return str(value)
        if value is None or isinstance(value, str):
            return value
        return str(value)

    def _save_segmentation_overview(
        self,
        sample_values,
        interval_records,
        output_path,
        *,
        background_payload=None,
        background_height_values=None,
    ):
        """保存与页面一致的实际负载及六态区间背景。"""
        actual = np.asarray(sample_values, dtype=float).reshape(-1)
        if actual.size == 0 or not np.any(np.isfinite(actual)):
            raise ValueError("实际采样值为空，无法生成 sample_overview.png")
        if background_height_values is None:
            fill_values = np.maximum(actual, 0.0)
        else:
            fill_values = np.asarray(
                background_height_values,
                dtype=float,
            ).reshape(-1)
            if fill_values.size != actual.size:
                raise ValueError("采样域填充高度与实际负载数量不一致")
        if background_payload is None:
            background_payload = self.build_segmentation_sample_background_masks(
                fill_values,
                None,
                interval_records,
                valid_mask=np.isfinite(fill_values),
            )

        x_values = np.arange(actual.size, dtype=float)
        fig, ax = plt.subplots(1, 1, figsize=(16, 5.8), dpi=120)
        try:
            fig.patch.set_facecolor(PLOT_FIG_BG)
            self.apply_plot_style(ax, grid=True)
            state_masks = dict(background_payload.get("state_masks", {}) or {})
            self.draw_segmentation_curve_background(
                ax,
                x_values,
                fill_values,
                state_masks,
                alpha=0.30,
                show_labels=True,
                zorder=1,
            )

            background_valid_mask = np.asarray(
                background_payload.get(
                    "valid_mask",
                    np.zeros(actual.size, dtype=bool),
                ),
                dtype=bool,
            )
            if background_valid_mask.size != actual.size:
                raise ValueError("六态背景有效掩码与实际负载数量不一致")
            actual_mask = np.isfinite(actual)
            if np.any(actual_mask):
                actual_x, actual_plot_values = (
                    self._compress_plot_series_preserve_gaps(
                        x_values,
                        actual,
                        int(getattr(self, "preview_plot_max_points", 60000) or 60000),
                    )
                )
                ax.plot(
                    actual_x,
                    actual_plot_values,
                    color=STYLE_MEASURED["color"],
                    linewidth=1.05,
                    label="实际负载",
                    zorder=4,
                )
            ax.set_title(
                "实际负载与区间划分",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_xlabel("实际负载采样点索引（0 基）", fontsize=11)
            ax.set_ylabel("功率 (W)", fontsize=11)
            ax.set_xlim(-0.5, float(max(actual.size, 1)) - 0.5)
            ax.set_ylim(bottom=0.0)
            ax.margins(x=0)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.18),
                    ncol=7,
                    frameon=False,
                    fontsize=9,
                )
            fig.subplots_adjust(left=0.075, right=0.985, top=0.82, bottom=0.14)
            fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
        finally:
            plt.close(fig)

    def _plot_option_enabled(self, variable_name, default=False):
        variable = getattr(self, variable_name, None)
        if variable is None:
            return bool(default)
        try:
            return bool(variable.get())
        except Exception:
            return bool(default)

    def _save_segmentation_process_overview(
        self,
        point_labels,
        intervals,
        output_path,
    ):
        """保存仅依赖 ProcessInfo 的程序 MRR 与六态背景图。"""

        fig, ax = plt.subplots(1, 1, figsize=(16, 5.8), dpi=120)
        try:
            fig.patch.set_facecolor(PLOT_FIG_BG)
            self.apply_plot_style(ax, grid=True)
            artists = self.draw_process_mrr_segmentation(
                ax,
                point_labels,
                intervals,
                show_labels=True,
            )
            if not artists:
                raise ValueError("过程域结果为空，无法生成 overview.png")
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                unique = {}
                for handle, label in zip(handles, labels):
                    if label and label not in unique:
                        unique[label] = handle
                ax.legend(
                    list(unique.values()),
                    list(unique.keys()),
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.18),
                    ncol=7,
                    frameon=False,
                    fontsize=9,
                )
            fig.subplots_adjust(left=0.075, right=0.985, top=0.82, bottom=0.14)
            fig.savefig(
                output_path,
                dpi=180,
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
            )
        finally:
            plt.close(fig)

    def _render_process_domain_segmentation_view(self, *, save=False):
        """在没有有效采样映射时显示权威过程域 MRR 视图。"""

        result = getattr(self, "_latest_segmentation_result", None)
        if result is None:
            return False
        point_labels = self._coerce_segmentation_dataframe(result, "point_labels")
        intervals = self._coerce_segmentation_dataframe(result, "intervals")
        if point_labels.empty:
            return False
        fig, ax = plt.subplots(figsize=(16, 8), dpi=100)
        fig.patch.set_facecolor(PLOT_FIG_BG)
        self.apply_plot_style(ax, grid=True)
        show_interval_states = self._plot_option_enabled(
            "show_interval_state_var",
            True,
        )
        self.draw_process_mrr_segmentation(
            ax,
            point_labels,
            intervals,
            show_labels=True,
            show_states=show_interval_states,
        )
        process_frame, process_x, _cell_left, _cell_right = (
            self._resolve_process_segmentation_coordinates(point_labels)
        )
        process_mask = np.isfinite(process_x)
        process_overlays = []
        overlay_getter = getattr(self, "get_optional_process_overlays", None)
        if callable(overlay_getter):
            process_overlays = list(overlay_getter(process_frame) or [])
        aux_axes = []
        overlay_plotter = getattr(
            self,
            "plot_optional_measurement_overlays",
            None,
        )
        if callable(overlay_plotter):
            aux_axes = list(
                overlay_plotter(
                    ax,
                    process_x,
                    process_mask,
                    process_mask,
                    overlays=process_overlays,
                )
                or []
            )

        legend_axes = [ax, *aux_axes]
        legend_count = sum(
            len(axis.get_legend_handles_labels()[0])
            for axis in legend_axes
        )
        legend_style = {
            "loc": "upper left",
            "ncol": min(7, max(legend_count, 1)),
            "framealpha": 0.95,
            "fontsize": PLOT_FONT_BASE - 1,
        }
        legend_applier = getattr(self, "_apply_optional_overlay_legend", None)
        if callable(legend_applier):
            legend_applier(ax, [ax], aux_axes, **legend_style)
        else:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                unique = {}
                for handle, label in zip(handles, labels):
                    if label and label not in unique:
                        unique[label] = handle
                ax.legend(
                    list(unique.values()),
                    list(unique.keys()),
                    **legend_style,
                )
        subplot_adjust = {
            "left": 0.07,
            "right": 0.985,
            "top": 0.92,
            "bottom": 0.11,
        }
        layout_applier = getattr(self, "_apply_optional_overlay_layout", None)
        if callable(layout_applier):
            layout_applier(fig, len(aux_axes), subplot_adjust)
        else:
            fig.subplots_adjust(**subplot_adjust)
        self.figures = [fig]
        self.figure_names = ["工艺信息与区间状态"]
        self.current_figure_index = 0
        if hasattr(self, "interval_count_var"):
            self.interval_count_var.set(str(len(intervals)))
        if hasattr(self, "_refresh_ideal_tree"):
            self._refresh_ideal_tree()
        if save:
            self.save_all_plots(silent=True)
        self.show_current_figure(0)
        return True

    def _resolve_segmentation_display_prediction(self, sample_count):
        """返回等长的显示预测曲线；必要时只刷新预测，不重跑六态分类。"""

        self._segmentation_display_prediction_error = ""

        def _read_prediction(payload):
            if not isinstance(payload, dict):
                return None
            try:
                values = np.asarray(
                    payload.get("predicted_load", []),
                    dtype=float,
                ).reshape(-1)
            except Exception:
                return None
            if values.size != int(sample_count) or not np.any(np.isfinite(values)):
                return None
            return values

        prediction_payload = None
        sample_mode = str(getattr(self, "sample_data_mode", "") or "").strip()
        if sample_mode == "experiment_measurement":
            candidate = getattr(self, "manual_measurement_data", None)
            if isinstance(candidate, dict):
                prediction_payload = candidate
            predicted_values = _read_prediction(prediction_payload)
            if predicted_values is not None:
                self._segmentation_display_prediction_error = ""
                return predicted_values

            # 工艺信息先行划分时，generate_plots 会走过程域快速路径，不再经过
            # 旧的预测刷新流程。实测文件后导入后在这里补一次显示预测即可；
            # 过程域标签和边界仍直接复用，预测结果不进入六态分类。
            refresher = getattr(self, "_refresh_manual_measurement_prediction", None)
            refresh_in_progress = bool(
                getattr(self, "_segmentation_display_prediction_refreshing", False)
            )
            if callable(refresher) and not refresh_in_progress:
                self._segmentation_display_prediction_refreshing = True
                try:
                    refresher()
                except Exception as exc:
                    self._segmentation_display_prediction_error = str(exc)
                finally:
                    self._segmentation_display_prediction_refreshing = False
                prediction_payload = getattr(
                    self,
                    "manual_measurement_data",
                    prediction_payload,
                )
            if (
                _read_prediction(prediction_payload) is None
                and bool(getattr(self, "release_mode", False))
            ):
                # 发布版不包含 PIT 的实验实测预测刷新器，改用与 SampleData
                # 相同的轻量反解。曲线是否显示由独立开关控制，计算仍会执行。
                builder = getattr(self, "_build_sampledata_prediction_payload", None)
                if callable(builder):
                    try:
                        candidate = builder()
                    except Exception as exc:
                        self._segmentation_display_prediction_error = str(exc)
                        candidate = None
                    if isinstance(candidate, dict):
                        prediction_payload = candidate
        elif sample_mode == "sampledata":
            model_ready = bool(
                self.has_prediction_model_ready()
                if hasattr(self, "has_prediction_model_ready")
                else False
            )
            builder = getattr(self, "_build_sampledata_prediction_payload", None)
            if (model_ready or bool(getattr(self, "release_mode", False))) and callable(builder):
                try:
                    candidate = builder()
                except Exception as exc:
                    self._segmentation_display_prediction_error = str(exc)
                    candidate = None
                if isinstance(candidate, dict):
                    prediction_payload = candidate

        predicted_values = _read_prediction(prediction_payload)
        if predicted_values is None:
            if not str(
                getattr(self, "_segmentation_display_prediction_error", "") or ""
            ).strip():
                self._segmentation_display_prediction_error = (
                    "预测负载未生成，或长度与实际负载不一致，或包含缺失值"
                )
        else:
            self._segmentation_display_prediction_error = ""
        return predicted_values

    def _render_segmentation_sample_overlay_view(
        self,
        mapping_records,
        *,
        save=False,
    ):
        """把 MRR 得到的六态区间投影到实际负载曲线上。"""

        result = getattr(self, "_latest_segmentation_result", None)
        values = np.asarray(getattr(self, "sample_data_values", []), dtype=float)
        if result is None or values.size == 0 or not mapping_records:
            return False
        if values.ndim == 2:
            source_var = getattr(self, "sample_data_source", None)
            try:
                source_index = int(source_var.get()) if source_var is not None else 0
            except Exception:
                source_index = 0
            if not 0 <= source_index < values.shape[1]:
                source_index = 0
            actual_values = values[:, source_index]
        else:
            actual_values = values.reshape(-1)
        actual_values = np.asarray(actual_values, dtype=float).reshape(-1)
        actual_finite = np.isfinite(actual_values)

        predicted_values = None
        show_prediction = False
        if bool(getattr(self, "release_mode", False)):
            predicted_values = self._resolve_segmentation_display_prediction(
                actual_values.size
            )
            show_prediction = self._plot_option_enabled(
                "show_prediction_load_var",
                False,
            )
            if predicted_values is not None:
                predicted_values = np.asarray(predicted_values, dtype=float).reshape(-1)
                if predicted_values.size != actual_values.size:
                    predicted_values = None

        sample_x = None
        x_label = "实际采样点索引（0 基）"
        time_getter = getattr(self, "get_sample_time_indices_array", None)
        if callable(time_getter):
            try:
                time_values = np.asarray(time_getter(), dtype=float).reshape(-1)
            except Exception:
                time_values = np.asarray([], dtype=float)
            if (
                time_values.size == actual_values.size
                and np.all(np.isfinite(time_values))
            ):
                sample_x = time_values
                x_label = "时间 (ms)"
        if sample_x is None:
            raw_positions = getattr(self, "sample_data_x_positions", None)
            if raw_positions is not None:
                try:
                    position_values = np.asarray(
                        raw_positions,
                        dtype=float,
                    ).reshape(-1)
                except Exception:
                    position_values = np.asarray([], dtype=float)
                if (
                    position_values.size == actual_values.size
                    and np.all(np.isfinite(position_values))
                ):
                    sample_x = position_values
                    x_label = "程序位置"
        if sample_x is None:
            sample_x = np.arange(actual_values.size, dtype=float)

        # 六态分类只来自过程域 MRR；显示预测时背景高度覆盖两条曲线。
        fill_values = np.maximum(actual_values, 0.0)
        if show_prediction and predicted_values is not None:
            predicted_finite = np.isfinite(predicted_values)
            fill_values[predicted_finite] = np.maximum(
                fill_values[predicted_finite],
                np.maximum(predicted_values[predicted_finite], 0.0),
            )
        background_payload = self.build_segmentation_sample_background_masks(
            fill_values,
            None,
            mapping_records,
            valid_mask=actual_finite,
        )
        projected_mask = np.asarray(
            background_payload.get(
                "process_projected_mask",
                np.zeros(actual_values.size, dtype=bool),
            ),
            dtype=bool,
        )
        overlay_context_mask = np.isfinite(sample_x)

        show_interval_states = self._plot_option_enabled(
            "show_interval_state_var",
            True,
        )

        fig, sample_ax = plt.subplots(figsize=(16, 7.2), dpi=100)
        fig.patch.set_facecolor(PLOT_FIG_BG)
        self.apply_plot_style(sample_ax, grid=True)
        if show_interval_states:
            self.draw_segmentation_curve_background(
                sample_ax,
                sample_x,
                fill_values,
                background_payload.get("state_masks", {}),
                alpha=0.20,
                show_labels=True,
                zorder=1,
            )

        preview_limit = int(getattr(self, "preview_plot_max_points", 60000) or 60000)
        plot_x, plot_values = self._compress_plot_series_preserve_gaps(
            sample_x,
            actual_values,
            preview_limit,
        )
        sample_ax.plot(
            plot_x,
            plot_values,
            color=STYLE_MEASURED["color"],
            linewidth=1.35,
            linestyle="-",
            label="实际负载",
            zorder=4,
        )
        if (
            show_prediction
            and predicted_values is not None
            and np.any(np.isfinite(predicted_values))
        ):
            predicted_x, predicted_plot_values = self._compress_plot_series_preserve_gaps(
                sample_x,
                predicted_values,
                preview_limit,
            )
            sample_ax.plot(
                predicted_x,
                predicted_plot_values,
                color=STYLE_PREDICTED["color"],
                linewidth=STYLE_PREDICTED["linewidth"],
                linestyle=STYLE_PREDICTED["linestyle"],
                label="预测负载",
                zorder=5,
            )
        aux_axes = []
        overlay_plotter = getattr(
            self,
            "plot_optional_measurement_overlays",
            None,
        )
        if callable(overlay_plotter):
            aux_axes = list(
                overlay_plotter(
                    sample_ax,
                    sample_x,
                    overlay_context_mask,
                    projected_mask,
                )
                or []
            )
        if not np.any(actual_finite):
            sample_ax.text(
                0.5,
                0.5,
                "当前实际负载没有可显示的有效数值",
                ha="center",
                va="center",
                transform=sample_ax.transAxes,
                fontsize=PLOT_FONT_BASE,
                color="#666666",
            )

        sample_ax.set_title(
            (
                "实际负载、预测负载与区间划分"
                if show_prediction and predicted_values is not None
                else "实际负载与区间划分"
            ),
            fontsize=PLOT_FONT_BASE + 1,
            fontweight="bold",
            pad=6,
        )
        sample_ax.set_xlabel(x_label)
        sample_ax.set_ylabel("功率 (W)")
        visible_x_mask = np.isfinite(sample_x) & projected_mask
        if not np.any(visible_x_mask):
            visible_x_mask = np.isfinite(sample_x)
        finite_x = sample_x[visible_x_mask]
        if finite_x.size:
            x_min = float(np.min(finite_x))
            x_max = float(np.max(finite_x))
            if x_max <= x_min:
                x_max = x_min + 1.0
            sample_ax.set_xlim(x_min, x_max)
        sample_ax.set_ylim(bottom=0.0)
        sample_ax.margins(x=0)

        if x_label == "时间 (ms)" and hasattr(self, "apply_line_axis_on_time"):
            self.apply_line_axis_on_time(sample_ax, projected_mask)
        elif hasattr(self, "apply_line_axis_on_path"):
            self.apply_line_axis_on_path(sample_ax, sample_x, projected_mask)

        legend_axes = [sample_ax, *aux_axes]
        legend_count = sum(
            len(axis.get_legend_handles_labels()[0])
            for axis in legend_axes
        )
        legend_style = {
            "loc": "upper left",
            "ncol": min(5, max(legend_count, 1)),
            "fontsize": PLOT_FONT_BASE - 1,
            "framealpha": 0.88,
            "borderpad": 0.35,
            "labelspacing": 0.3,
            "columnspacing": 0.9,
        }
        legend_applier = getattr(self, "_apply_optional_overlay_legend", None)
        if callable(legend_applier):
            legend_applier(
                sample_ax,
                [sample_ax],
                aux_axes,
                **legend_style,
            )
        else:
            sample_handles, sample_labels = sample_ax.get_legend_handles_labels()
            if sample_handles:
                unique = {}
                for handle, label in zip(sample_handles, sample_labels):
                    if label and label not in unique:
                        unique[label] = handle
                sample_ax.legend(
                    list(unique.values()),
                    list(unique.keys()),
                    **legend_style,
                )

        subplot_adjust = {
            "left": 0.06,
            "right": 0.985,
            "top": 0.94,
            "bottom": 0.09,
        }
        layout_applier = getattr(self, "_apply_optional_overlay_layout", None)
        if callable(layout_applier):
            layout_applier(fig, len(aux_axes), subplot_adjust)
        else:
            fig.subplots_adjust(**subplot_adjust)
        context_register = getattr(self, "_register_optional_overlay_context", None)
        if callable(context_register):
            context_register(
                fig,
                parent_ax=sample_ax,
                x_values=sample_x,
                context_mask=overlay_context_mask,
                valid_mask=projected_mask,
                legend_host=sample_ax,
                legend_base_axes=[sample_ax],
                aux_axes=aux_axes,
                legend_style=legend_style,
                subplot_adjust=subplot_adjust,
            )
        self.figures = [fig]
        self.figure_names = [
            "实际负载、预测负载与区间划分"
            if show_prediction and predicted_values is not None
            else "实际负载与区间划分"
        ]
        self.current_figure_index = 0
        if hasattr(self, "interval_count_var"):
            self.interval_count_var.set(str(len(mapping_records)))
        if hasattr(self, "_refresh_ideal_tree"):
            self._refresh_ideal_tree()
        if save:
            self.save_all_plots(silent=True)
        self.show_current_figure(0)
        return True

    def _finish_segmentation_fast_plot(self, rendered, message):
        """统一收尾轻量过程域图，避免进度停在“正在生成”。"""

        rendered = bool(rendered)
        if rendered:
            total_charts = len(getattr(self, "figures", []) or [])
            status_var = getattr(self, "status_var_data", None)
            if status_var is not None and hasattr(status_var, "set"):
                status_var.set(f"图表已生成! 共{total_charts}张图表")
            if hasattr(self, "set_progress"):
                self.set_progress(100, str(message or "图表已生成"))
        elif hasattr(self, "set_progress"):
            self.set_progress(0, "图表生成失败")
        return rendered

    def _replace_segmentation_bundle(self, paths, temporary_paths):
        """把同一坐标域的一组临时文件整体替换，并在失败时回滚。"""

        backup_paths = {
            key: path.with_name(f".{path.stem}.bak{path.suffix}")
            for key, path in paths.items()
        }
        backed_up_keys = []
        replaced_keys = []
        try:
            for key, target_path in paths.items():
                backup_paths[key].unlink(missing_ok=True)
                if target_path.exists():
                    target_path.replace(backup_paths[key])
                    backed_up_keys.append(key)
            for key, target_path in paths.items():
                temporary_paths[key].replace(target_path)
                replaced_keys.append(key)
        except Exception as replace_exc:
            for key in reversed(replaced_keys):
                paths[key].unlink(missing_ok=True)
            rollback_errors = []
            for key in reversed(backed_up_keys):
                try:
                    backup_paths[key].replace(paths[key])
                except Exception as rollback_exc:
                    rollback_errors.append(f"{key}: {rollback_exc}")
            if rollback_errors:
                raise RuntimeError(
                    "六态导出替换失败，且旧文件回滚不完整："
                    + "；".join(rollback_errors)
                ) from replace_exc
            raise
        else:
            for backup_path in backup_paths.values():
                backup_path.unlink(missing_ok=True)

    def export_segmentation_failure_diagnostics(self, result, output_dir=None):
        """只导出失败诊断，并确保旧区间和概览不会冒充本轮结果。"""

        if result is None:
            raise ValueError("缺少需要导出的六态失败诊断")
        target_dir = (
            Path(output_dir)
            if output_dir is not None
            else self._default_segmentation_output_dir()
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        self._clear_segmentation_output_artifacts(target_dir)
        diagnostics = dict(getattr(result, "diagnostics", {}) or {})
        config = getattr(result, "config", None)
        diagnostics["config"] = (
            config.to_dict()
            if hasattr(config, "to_dict")
            else dict(getattr(config, "__dict__", {}) or {})
        )
        diagnostics["export_status"] = "decode_failed_diagnostics_only"
        for attribute_name in ("input_schema_version", "scorer_type", "model_version"):
            diagnostics[attribute_name] = getattr(result, attribute_name, None)
        payload = self._normalize_segmentation_json_value(diagnostics)
        target_path = target_dir / "diagnostics.json"
        temporary_path = target_dir / ".diagnostics.tmp.json"
        try:
            with temporary_path.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as stream:
                json.dump(
                    payload,
                    stream,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                stream.write("\n")
            temporary_path.replace(target_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return target_path

    def _export_latest_segmentation_result_legacy_sample(self, result=None, output_dir=None):
        """独立导出最近一次六态结果，并原子更新固定导出文件。"""
        resolved_result = result or getattr(self, "_latest_segmentation_result", None)
        if resolved_result is None:
            raise ValueError("当前没有可导出的全行程六类划分结果")
        result_diagnostics = dict(
            getattr(resolved_result, "diagnostics", {}) or {}
        )
        fallback_used = bool(result_diagnostics.get("fallback_used", False))
        fallback_scope = str(result_diagnostics.get("fallback_scope") or "none")
        fallback_validated = bool(
            result_diagnostics.get("fallback_validated", not fallback_used)
        )
        if fallback_used and not (
            fallback_scope == "local_verified" and fallback_validated
        ):
            raise ValueError(
                "未验证的解码回退只能导出失败诊断，不能导出普通六态结果"
            )
        if not bool(
            result_diagnostics.get("postprocess_validation_passed", False)
        ):
            raise ValueError("六态结果未通过结构复查，不能导出普通结果")

        target_dir = (
            Path(output_dir)
            if output_dir is not None
            else self._default_segmentation_output_dir()
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        self._clear_segmentation_output_artifacts(target_dir)
        paths = {
            "intervals": target_dir / "intervals.csv",
            "overview": target_dir / "overview.png",
            "diagnostics": target_dir / "diagnostics.json",
        }

        point_labels = self._coerce_segmentation_dataframe(resolved_result, "point_labels")
        intervals = self._coerce_segmentation_dataframe(resolved_result, "intervals")
        required_point_columns = {
            "point_id", "s", "interval_id", "segment_type", "state_code",
            "review_required",
        }
        required_interval_columns = {
            "interval_id", "start_s", "end_s", "segment_type", "state_code",
            "review_required",
        }
        missing_point_columns = sorted(required_point_columns.difference(point_labels.columns))
        missing_interval_columns = sorted(required_interval_columns.difference(intervals.columns))
        if missing_point_columns or missing_interval_columns:
            details = []
            if missing_point_columns:
                details.append(f"point_labels 缺少 {missing_point_columns}")
            if missing_interval_columns:
                details.append(f"intervals 缺少 {missing_interval_columns}")
            raise ValueError("；".join(details))

        point_codes = pd.to_numeric(point_labels["state_code"], errors="coerce")
        interval_codes = pd.to_numeric(intervals["state_code"], errors="coerce")
        if not point_codes.isin(range(6)).all() or not interval_codes.isin(range(6)).all():
            raise ValueError("六态结构化输出包含 0..5 之外的 state_code")

        runtime_records = self._adapt_segmentation_interval_records(resolved_result)
        projected_records = self._materialize_segmentation_sample_bounds(runtime_records)
        prediction_payload = self._get_segmentation_prediction_payload()
        predicted_load = np.asarray(prediction_payload["predicted_load"], dtype=float)
        predicted_idle_power = np.asarray(
            prediction_payload.get("predicted_idle_power", []),
            dtype=float,
        )
        program_lines = np.asarray(prediction_payload["program_line"], dtype=int)
        if predicted_load.size != program_lines.size:
            raise ValueError("预测负载与实际采样行号数量不一致")
        sample_lines = self._get_segmentation_sample_lines()
        if program_lines.size != sample_lines.size or not np.array_equal(program_lines, sample_lines):
            raise ValueError("预测负载的程序行序列与实际采样文件不一致")
        background_payload = self.build_segmentation_sample_background_masks(
            predicted_load,
            predicted_idle_power,
            projected_records,
        )

        state_rows = {code: [] for code in range(6)}
        for record in projected_records:
            state_code = int(record.get("state_code"))
            interval_range = str(record.get("sample_interval_range") or "").strip()
            if state_code not in state_rows or not interval_range:
                raise ValueError("六态区间缺少有效的实际采样坐标")
            state_rows[state_code].append(interval_range)

        diagnostics = dict(getattr(resolved_result, "diagnostics", {}) or {})
        config = getattr(resolved_result, "config", None)
        if hasattr(config, "to_dict"):
            config_payload = config.to_dict()
        elif isinstance(config, dict):
            config_payload = dict(config)
        else:
            config_payload = dict(getattr(config, "__dict__", {}) or {})
        diagnostics["config"] = config_payload
        covered_sample_count = int(sum(
            int(record.get("sample_count", 0) or 0)
            for record in projected_records
        ))
        projection_start = int(projected_records[0]["sample_start_idx"])
        projection_end = int(projected_records[-1]["sample_end_idx"])
        diagnostics["sample_projection"] = {
            **dict(diagnostics.get("sample_projection", {}) or {}),
            "valid": True,
            "sample_count": int(predicted_load.size),
            "interval_count": int(len(projected_records)),
            "covered_sample_count": covered_sample_count,
            "projected_sample_start_idx": projection_start,
            "projected_sample_end_idx": projection_end,
            "projected_start_label": str(
                projected_records[0].get("sample_start_label") or ""
            ),
            "projected_end_label": str(
                projected_records[-1].get("sample_end_label") or ""
            ),
            "coordinate_format": "line.zero_based_point-line.zero_based_point",
            "endpoint_inclusive": True,
        }
        diagnostics["sample_visualization"] = {
            "valid_sample_count": int(background_payload["valid_sample_count"]),
            "process_projected_sample_count": int(
                background_payload["process_projected_sample_count"]
            ),
            "external_idle_sample_count": int(
                background_payload["external_idle_sample_count"]
            ),
            "external_nonsteady_sample_count": int(
                background_payload["external_nonsteady_sample_count"]
            ),
            "idle_power_tolerance": background_payload.get("idle_power_tolerance"),
            "display_only": True,
        }
        diagnostics["interval_csv"] = {
            "row_count": 6,
            "export_state_codes": {
                "1": "idle",
                "2": "entry",
                "3": "steady",
                "4": "transition",
                "5": "nonsteady",
                "6": "exit",
            },
        }
        for attribute_name in ("input_schema_version", "scorer_type", "model_version"):
            diagnostics[attribute_name] = getattr(resolved_result, attribute_name, None)
        diagnostics_payload = self._normalize_segmentation_json_value(diagnostics)

        temporary_paths = {
            key: path.with_name(f".{path.stem}.tmp{path.suffix}")
            for key, path in paths.items()
        }
        backup_paths = {
            key: path.with_name(f".{path.stem}.bak{path.suffix}")
            for key, path in paths.items()
        }
        try:
            with temporary_paths["intervals"].open(
                "w",
                encoding="utf-8",
                newline="",
            ) as stream:
                writer = csv.writer(stream, lineterminator="\n")
                for state_code in range(6):
                    writer.writerow([state_code + 1, *state_rows[state_code]])
            self._save_segmentation_overview(
                predicted_load,
                projected_records,
                temporary_paths["overview"],
                background_payload=background_payload,
                background_height_values=np.maximum(predicted_load, 0.0),
            )
            with temporary_paths["diagnostics"].open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as stream:
                json.dump(
                    diagnostics_payload,
                    stream,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                stream.write("\n")
            backed_up_keys = []
            replaced_keys = []
            try:
                for key, target_path in paths.items():
                    backup_paths[key].unlink(missing_ok=True)
                    if target_path.exists():
                        target_path.replace(backup_paths[key])
                        backed_up_keys.append(key)
                for key, target_path in paths.items():
                    temporary_paths[key].replace(target_path)
                    replaced_keys.append(key)
            except Exception as replace_exc:
                for key in reversed(replaced_keys):
                    paths[key].unlink(missing_ok=True)
                rollback_errors = []
                for key in reversed(backed_up_keys):
                    try:
                        backup_paths[key].replace(paths[key])
                    except Exception as rollback_exc:
                        rollback_errors.append(f"{key}: {rollback_exc}")
                if rollback_errors:
                    raise RuntimeError(
                        "六态导出替换失败，且旧文件回滚不完整："
                        + "；".join(rollback_errors)
                    ) from replace_exc
                raise
            else:
                for backup_path in backup_paths.values():
                    backup_path.unlink(missing_ok=True)
        finally:
            for temporary_path in temporary_paths.values():
                temporary_path.unlink(missing_ok=True)

        if hasattr(self, "segmentation_status_var"):
            self.segmentation_status_var.set(
                f"全行程六类划分: 已导出 {len(projected_records)} 段 / "
                f"覆盖 {covered_sample_count} 个实际采样点（文件共 {predicted_load.size} 点）"
            )
        return paths

    def export_latest_segmentation_result(self, result=None, output_dir=None):
        """分别导出权威过程结果和可选采样映射；前者不依赖实测文件。"""

        resolved_result = result or getattr(self, "_latest_segmentation_result", None)
        if resolved_result is None:
            raise ValueError("当前没有可导出的全行程六类划分结果")
        diagnostics = dict(getattr(resolved_result, "diagnostics", {}) or {})
        fallback_used = bool(diagnostics.get("fallback_used", False))
        fallback_scope = str(diagnostics.get("fallback_scope") or "none")
        fallback_validated = bool(
            diagnostics.get("fallback_validated", not fallback_used)
        )
        if fallback_used and not (
            fallback_scope == "local_verified" and fallback_validated
        ):
            raise ValueError("未验证的解码回退只能导出失败诊断，不能导出普通六态结果")
        if not bool(diagnostics.get("postprocess_validation_passed", False)):
            raise ValueError("六态结果未通过结构复查，不能导出普通结果")

        point_labels = self._coerce_segmentation_dataframe(
            resolved_result,
            "point_labels",
        )
        intervals = self._coerce_segmentation_dataframe(
            resolved_result,
            "intervals",
        )
        required_point_columns = {
            "point_id",
            "s",
            "interval_id",
            "segment_type",
            "state_code",
            "review_required",
        }
        required_interval_columns = {
            "interval_id",
            "start_s",
            "end_s",
            "segment_type",
            "state_code",
            "review_required",
        }
        missing_point_columns = sorted(required_point_columns.difference(point_labels.columns))
        missing_interval_columns = sorted(required_interval_columns.difference(intervals.columns))
        if missing_point_columns or missing_interval_columns:
            details = []
            if missing_point_columns:
                details.append(f"point_labels 缺少 {missing_point_columns}")
            if missing_interval_columns:
                details.append(f"intervals 缺少 {missing_interval_columns}")
            raise ValueError("；".join(details))

        point_codes = pd.to_numeric(point_labels["state_code"], errors="coerce")
        interval_codes = pd.to_numeric(intervals["state_code"], errors="coerce")
        if not point_codes.isin(range(6)).all() or not interval_codes.isin(range(6)).all():
            raise ValueError("六态结构化输出包含 0..5 之外的 state_code")

        process_signature = str(
            diagnostics.get("process_signature")
            or (
                dict(diagnostics.get("repeat_run_consistency") or {}).get(
                    "input_signature"
                )
            )
            or getattr(self, "_current_process_signature", "")
            or ""
        )
        if not process_signature:
            raise ValueError("过程域结果缺少 process_signature")

        target_dir = (
            Path(output_dir)
            if output_dir is not None
            else self._default_segmentation_output_dir()
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        process_paths = {
            "point_labels": target_dir / "point_labels.csv",
            "intervals": target_dir / "intervals.csv",
            "overview": target_dir / "overview.png",
            "diagnostics": target_dir / "diagnostics.json",
        }
        process_temporary_paths = {
            key: path.with_name(f".{path.stem}.tmp{path.suffix}")
            for key, path in process_paths.items()
        }
        process_point_labels = point_labels.copy()
        process_intervals = intervals.copy()
        for frame in (process_point_labels, process_intervals):
            frame["coordinate_domain"] = "process_info"
            frame["process_signature"] = process_signature
        process_diagnostics = dict(diagnostics)
        process_diagnostics.pop("sample_projection", None)
        process_diagnostics.pop("sample_visualization", None)
        process_diagnostics.update(
            {
                "coordinate_domain": "process_info",
                "process_signature": process_signature,
                "export_status": "process_domain_valid",
                "process_point_row_count": int(len(process_point_labels)),
                "process_interval_row_count": int(len(process_intervals)),
            }
        )
        config = getattr(resolved_result, "config", None)
        process_diagnostics["config"] = (
            config.to_dict()
            if hasattr(config, "to_dict")
            else dict(getattr(config, "__dict__", {}) or {})
        )
        for attribute_name in ("input_schema_version", "scorer_type", "model_version"):
            process_diagnostics[attribute_name] = getattr(
                resolved_result,
                attribute_name,
                None,
            )
        process_payload = self._normalize_segmentation_json_value(
            process_diagnostics
        )
        try:
            self._write_dataframe_csv_compat(
                process_point_labels,
                process_temporary_paths["point_labels"],
            )
            self._write_dataframe_csv_compat(
                process_intervals,
                process_temporary_paths["intervals"],
            )
            self._save_segmentation_process_overview(
                process_point_labels,
                process_intervals,
                process_temporary_paths["overview"],
            )
            with process_temporary_paths["diagnostics"].open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as stream:
                json.dump(
                    process_payload,
                    stream,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                stream.write("\n")
            self._replace_segmentation_bundle(
                process_paths,
                process_temporary_paths,
            )
        finally:
            for temporary_path in process_temporary_paths.values():
                temporary_path.unlink(missing_ok=True)

        exported_paths = dict(process_paths)
        mapping_records = None
        if bool(getattr(self, "sample_data_loaded", False)):
            refresher = getattr(self, "_refresh_segmentation_sample_projection", None)
            if callable(refresher):
                mapping_records = refresher(refresh_view=False, silent=True)
        else:
            self._clear_segmentation_output_artifacts(target_dir, scope="mapping")

        projection_diagnostics = dict(
            getattr(resolved_result, "diagnostics", {}).get("sample_projection", {})
            or {}
        )
        mapping_diagnostics_path = target_dir / "sample_mapping_diagnostics.json"
        mapping_diagnostics_temp = target_dir / ".sample_mapping_diagnostics.tmp.json"
        if bool(getattr(self, "sample_data_loaded", False)):
            mapping_payload = self._normalize_segmentation_json_value(
                {
                    **projection_diagnostics,
                    "coordinate_domain": "sample",
                    "process_signature": process_signature,
                    "mapping_signature": str(
                        getattr(self, "_current_mapping_signature", "") or ""
                    ),
                    "mapping_status": str(
                        getattr(self, "_sample_mapping_status", "failed") or "failed"
                    ),
                }
            )
            try:
                with mapping_diagnostics_temp.open(
                    "w",
                    encoding="utf-8",
                    newline="\n",
                ) as stream:
                    json.dump(
                        mapping_payload,
                        stream,
                        ensure_ascii=False,
                        indent=2,
                        allow_nan=False,
                    )
                    stream.write("\n")
                mapping_diagnostics_temp.replace(mapping_diagnostics_path)
                exported_paths["sample_mapping_diagnostics"] = mapping_diagnostics_path
            finally:
                mapping_diagnostics_temp.unlink(missing_ok=True)

        if mapping_records:
            values = np.asarray(getattr(self, "sample_data_values", []), dtype=float)
            if values.ndim == 2:
                source_var = getattr(self, "sample_data_source", None)
                try:
                    source_index = int(source_var.get()) if source_var is not None else 0
                except Exception:
                    source_index = 0
                if not 0 <= source_index < values.shape[1]:
                    source_index = 0
                actual_values = values[:, source_index]
            else:
                actual_values = values.reshape(-1)
            sample_lines = np.asarray(
                getattr(self, "sample_data_line_numbers", []),
                dtype=int,
            )
            if actual_values.size != sample_lines.size or actual_values.size == 0:
                raise ValueError("实际采样值与程序行号数量不一致")

            mapping_signature = str(
                getattr(self, "_current_mapping_signature", "") or ""
            )
            projection_frame = pd.DataFrame(mapping_records)
            projection_frame["coordinate_domain"] = "sample"
            projection_frame["process_signature"] = process_signature
            projection_frame["mapping_signature"] = mapping_signature
            mapping_paths = {
                "sample_projection": target_dir / "sample_projection.csv",
                "sample_overview": target_dir / "sample_overview.png",
            }
            mapping_temporary_paths = {
                key: path.with_name(f".{path.stem}.tmp{path.suffix}")
                for key, path in mapping_paths.items()
            }
            try:
                self._write_dataframe_csv_compat(
                    projection_frame,
                    mapping_temporary_paths["sample_projection"],
                )
                background_height_values = np.maximum(actual_values, 0.0)
                background_payload = self.build_segmentation_sample_background_masks(
                    background_height_values,
                    None,
                    mapping_records,
                    valid_mask=np.isfinite(background_height_values),
                )
                self._save_segmentation_overview(
                    actual_values,
                    mapping_records,
                    mapping_temporary_paths["sample_overview"],
                    background_payload=background_payload,
                    background_height_values=background_height_values,
                )
                self._replace_segmentation_bundle(
                    mapping_paths,
                    mapping_temporary_paths,
                )
                exported_paths.update(mapping_paths)
            except Exception:
                for mapping_path in mapping_paths.values():
                    mapping_path.unlink(missing_ok=True)
                raise
            finally:
                for temporary_path in mapping_temporary_paths.values():
                    temporary_path.unlink(missing_ok=True)
        else:
            for file_name in ("sample_projection.csv", "sample_overview.png"):
                (target_dir / file_name).unlink(missing_ok=True)

        if hasattr(self, "segmentation_status_var"):
            self.segmentation_status_var.set(
                f"过程域六类划分: 已导出 {len(process_point_labels)} 点 / "
                f"{len(process_intervals)} 段"
            )
        return exported_paths

    def _format_interval_point_range(self, interval):
        """按“行号.点号-行号.点号”格式返回区间边界。"""
        start_label = str(interval.get("start_label") or "").strip()
        end_label = str(interval.get("end_label") or "").strip()
        if not start_label:
            start_line = interval.get("start_line")
            if start_line is not None:
                start_label = str(start_line).strip()
        if not end_label:
            end_line = interval.get("end_line")
            if end_line is not None:
                end_label = str(end_line).strip()
        if start_label and end_label:
            return f"{start_label}-{end_label}"
        return start_label or end_label

    def _resolve_measurement_process_interval_bounds(
        self,
        seg_start,
        seg_end,
        aligned_lines,
        process_aligned_lines,
        sample_process_point_indices=None,
        process_point_indices=None,
    ):
        """将样本区间精确映射回工艺点范围，而不是按整行吞并末端程序行。"""
        sample_lines = np.asarray(aligned_lines, dtype=int)
        process_lines = np.asarray(process_aligned_lines, dtype=int)
        if sample_lines.size == 0 or process_lines.size == 0:
            return None
        try:
            start_pos = int(seg_start)
            end_pos = int(seg_end)
        except Exception:
            return None
        if start_pos < 0 or end_pos < 0 or start_pos >= sample_lines.size or end_pos >= sample_lines.size:
            return None
        if end_pos < start_pos:
            start_pos, end_pos = end_pos, start_pos

        start_line = int(sample_lines[start_pos])
        end_line = int(sample_lines[end_pos])
        lower_line = min(start_line, end_line)
        upper_line = max(start_line, end_line)

        point_arr = None
        if process_point_indices is not None:
            try:
                point_arr = np.asarray(process_point_indices, dtype=int)
            except Exception:
                point_arr = None
            if point_arr is not None and point_arr.size != process_lines.size:
                point_arr = None

        sample_point_arr = None
        if sample_process_point_indices is not None:
            try:
                sample_point_arr = np.asarray(sample_process_point_indices, dtype=int)
            except Exception:
                sample_point_arr = None
            if sample_point_arr is not None and sample_point_arr.size != sample_lines.size:
                sample_point_arr = None

        base_indices = np.flatnonzero((process_lines >= lower_line) & (process_lines <= upper_line))
        if base_indices.size == 0:
            return None

        start_candidates = np.flatnonzero(process_lines == start_line)
        end_candidates = np.flatnonzero(process_lines == end_line)
        if start_candidates.size == 0 or end_candidates.size == 0:
            return int(base_indices[0]), int(base_indices[-1])

        if sample_point_arr is not None and point_arr is not None:
            start_point = int(sample_point_arr[start_pos])
            end_point = int(sample_point_arr[end_pos])

            if start_line == end_line:
                same_line_mask = process_lines == start_line
                if start_point >= 0:
                    same_line_mask &= point_arr >= start_point
                if end_point >= 0:
                    same_line_mask &= point_arr <= end_point
                same_line_indices = np.flatnonzero(same_line_mask)
                if same_line_indices.size > 0:
                    return int(same_line_indices[0]), int(same_line_indices[-1])

            if start_point >= 0:
                start_mask = (process_lines == start_line) & (point_arr >= start_point)
                precise_start = np.flatnonzero(start_mask)
                if precise_start.size > 0:
                    start_candidates = precise_start
            if end_point >= 0:
                end_mask = (process_lines == end_line) & (point_arr <= end_point)
                precise_end = np.flatnonzero(end_mask)
                if precise_end.size > 0:
                    end_candidates = precise_end

        start_idx = int(start_candidates[0])
        end_idx = int(end_candidates[-1])
        if end_idx < start_idx:
            return int(base_indices[0]), int(base_indices[-1])
        return start_idx, end_idx

    def _split_measurement_segment_by_aligned_lines(self, seg_start, seg_end, aligned_lines):
        """当跨多条对齐程序行的候选段被整段拒绝时，回退到按 aligned line 重试。"""
        try:
            start_idx = int(seg_start)
            end_idx = int(seg_end)
        except Exception:
            return []
        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx

        aligned_arr = np.asarray(aligned_lines, dtype=int)
        if aligned_arr.size == 0 or start_idx < 0 or end_idx >= aligned_arr.size:
            return []

        segments = []
        local_start = int(start_idx)
        while local_start <= end_idx:
            local_line = int(aligned_arr[local_start])
            local_end = local_start
            while local_end + 1 <= end_idx and int(aligned_arr[local_end + 1]) == local_line:
                local_end += 1
            segments.append((int(local_start), int(local_end)))
            local_start = local_end + 1
        return segments if len(segments) > 1 else []

    def _build_block_id_array(self, blocks, size):
        """根据分块结果生成逐点 block id。"""
        block_ids = np.full(int(size), -1, dtype=int)
        for block_id, (start_idx, end_idx) in enumerate(blocks or []):
            block_ids[start_idx:end_idx + 1] = block_id
        return block_ids

    def _normalize_index_blocks(self, blocks, size):
        """标准化索引区间，裁剪到有效范围并合并重叠/相邻区间。"""
        try:
            max_size = int(size)
        except Exception:
            max_size = 0
        if max_size <= 0:
            return []

        normalized = []
        for block in blocks or []:
            if not block or len(block) < 2:
                continue
            try:
                start_idx = int(block[0])
                end_idx = int(block[1])
            except Exception:
                continue
            if end_idx < start_idx:
                start_idx, end_idx = end_idx, start_idx
            start_idx = max(0, start_idx)
            end_idx = min(max_size - 1, end_idx)
            if end_idx < start_idx:
                continue
            normalized.append((start_idx, end_idx))

        if not normalized:
            return []

        normalized.sort(key=lambda item: (item[0], item[1]))
        merged = [list(normalized[0])]
        for start_idx, end_idx in normalized[1:]:
            last_block = merged[-1]
            if start_idx <= last_block[1] + 1:
                last_block[1] = max(last_block[1], end_idx)
            else:
                merged.append([start_idx, end_idx])
        return [(int(start_idx), int(end_idx)) for start_idx, end_idx in merged]

    def _subtract_index_blocks(self, base_blocks, covered_blocks, size):
        """返回 base_blocks 去掉 covered_blocks 后的剩余索引区间。"""
        normalized_base = self._normalize_index_blocks(base_blocks, size)
        normalized_covered = self._normalize_index_blocks(covered_blocks, size)
        if not normalized_base:
            return []
        if not normalized_covered:
            return normalized_base

        remaining = []
        covered_idx = 0
        covered_count = len(normalized_covered)
        for base_start, base_end in normalized_base:
            cursor = int(base_start)
            while covered_idx < covered_count and normalized_covered[covered_idx][1] < base_start:
                covered_idx += 1
            probe_idx = covered_idx
            while probe_idx < covered_count:
                cover_start, cover_end = normalized_covered[probe_idx]
                if cover_start > base_end:
                    break
                if cover_end < cursor:
                    probe_idx += 1
                    continue
                if cover_start > cursor:
                    remaining.append((int(cursor), int(min(base_end, cover_start - 1))))
                cursor = max(cursor, cover_end + 1)
                if cursor > base_end:
                    break
                probe_idx += 1
            if cursor <= base_end:
                remaining.append((int(cursor), int(base_end)))
        return remaining

    def _build_display_spans_from_blocks(self, x_values, blocks):
        """将索引区间转换为显示坐标跨度。"""
        x_arr = np.asarray(x_values, dtype=float)
        if x_arr.size == 0:
            return []

        normalized_blocks = self._normalize_index_blocks(blocks, x_arr.size)
        spans = []
        finite_x = x_arr[np.isfinite(x_arr)]
        default_step = float(np.nanmedian(np.diff(finite_x))) if finite_x.size > 1 else 1.0
        if not np.isfinite(default_step) or default_step <= 0:
            default_step = 1.0

        for start_idx, end_idx in normalized_blocks:
            start_x = float(x_arr[start_idx])
            end_x = float(x_arr[end_idx])
            if not (np.isfinite(start_x) and np.isfinite(end_x)):
                continue

            left_step = default_step
            if start_idx > 0 and np.isfinite(x_arr[start_idx - 1]):
                delta = float(start_x - x_arr[start_idx - 1])
                if np.isfinite(delta) and delta > 0:
                    left_step = delta

            right_step = default_step
            if end_idx + 1 < x_arr.size and np.isfinite(x_arr[end_idx + 1]):
                delta = float(x_arr[end_idx + 1] - end_x)
                if np.isfinite(delta) and delta > 0:
                    right_step = delta

            span_start = float(start_x - max(left_step, 1e-9) * 0.5)
            span_end = float(end_x + max(right_step, 1e-9) * 0.5)
            if span_end <= span_start:
                span_end = float(span_start + max(default_step, 1e-9))
            spans.append((span_start, span_end))
        return spans

    def _draw_background_spans(self, ax, spans, color, alpha, label, zorder):
        """用跨度集合绘制背景带，避免逐点 fill_between 的高开销。"""
        xranges = []
        for span_start, span_end in spans or []:
            try:
                start_val = float(span_start)
                end_val = float(span_end)
            except Exception:
                continue
            if not (np.isfinite(start_val) and np.isfinite(end_val)):
                continue
            width = float(end_val - start_val)
            if width <= 0:
                continue
            xranges.append((start_val, width))
        if not xranges:
            return None

        try:
            return ax.broken_barh(
                xranges,
                (0.0, 1.0),
                facecolors=color,
                edgecolors="none",
                alpha=alpha,
                zorder=zorder,
                label=label,
                transform=ax.get_xaxis_transform(),
            )
        except Exception:
            artist = None
            first = True
            for start_val, width in xranges:
                block_kwargs = {
                    "xmin": start_val,
                    "xmax": start_val + width,
                    "ymin": 0.0,
                    "ymax": 1.0,
                    "facecolor": color,
                    "edgecolor": "none",
                    "alpha": alpha,
                    "zorder": zorder,
                }
                if first and label:
                    block_kwargs["label"] = label
                artist = ax.axvspan(**block_kwargs)
                first = False
            return artist

    def _draw_curve_background_blocks(self, ax, x_values, y_values, blocks, color, alpha, label, zorder):
        """按完整横坐标的共享单元边界绘制互斥背景块。"""
        x_arr = np.asarray(x_values, dtype=float)
        y_arr = np.asarray(y_values, dtype=float)
        if x_arr.size == 0 or y_arr.size != x_arr.size:
            return []

        normalized_blocks = self._normalize_index_blocks(blocks, x_arr.size)
        if not normalized_blocks:
            return []

        # 每个采样点占用相邻中心中点之间的单元。所有状态都从完整 x 序列
        # 计算同一组边界，因此非均匀间距和重复坐标都不会产生正宽度重叠。
        finite_indices = np.flatnonzero(np.isfinite(x_arr))
        if finite_indices.size == 0:
            return []
        finite_centers = x_arr[finite_indices]
        cell_left = np.full(x_arr.size, np.nan, dtype=float)
        cell_right = np.full(x_arr.size, np.nan, dtype=float)
        if finite_indices.size == 1:
            only_idx = int(finite_indices[0])
            cell_left[only_idx] = float(finite_centers[0])
            cell_right[only_idx] = float(finite_centers[0])
        else:
            shared_edges = finite_centers[:-1] + np.diff(finite_centers) * 0.5
            cell_right[finite_indices[:-1]] = shared_edges
            cell_left[finite_indices[1:]] = shared_edges
            first_step = float(finite_centers[1] - finite_centers[0])
            last_step = float(finite_centers[-1] - finite_centers[-2])
            cell_left[int(finite_indices[0])] = float(finite_centers[0] - first_step * 0.5)
            cell_right[int(finite_indices[-1])] = float(finite_centers[-1] + last_step * 0.5)

        polygons = []
        for start_idx, end_idx in normalized_blocks:
            valid_mask = (
                np.isfinite(x_arr[start_idx:end_idx + 1])
                & np.isfinite(y_arr[start_idx:end_idx + 1])
            )
            valid_offsets = np.flatnonzero(valid_mask)
            if valid_offsets.size == 0:
                continue

            # 异常值造成的洞必须拆开，不能跨洞连接成一个多边形。
            split_points = np.flatnonzero(np.diff(valid_offsets) > 1) + 1
            for offset_run in np.split(valid_offsets, split_points):
                run_start = int(start_idx + offset_run[0])
                run_end = int(start_idx + offset_run[-1])
                left_edge = float(cell_left[run_start])
                right_edge = float(cell_right[run_end])
                if not np.isfinite(left_edge) or not np.isfinite(right_edge):
                    continue

                segment_x = x_arr[run_start:run_end + 1]
                segment_y = np.maximum(y_arr[run_start:run_end + 1], 0.0)
                top_x = np.concatenate(([left_edge], segment_x, [right_edge]))
                top_y = np.concatenate(([segment_y[0]], segment_y, [segment_y[-1]]))
                top_vertices = np.column_stack((top_x, top_y))
                bottom_vertices = np.column_stack((top_x[::-1], np.zeros(top_x.size)))
                polygons.append(np.vstack((top_vertices, bottom_vertices)))
        if not polygons:
            return []
        collection = PolyCollection(
            polygons,
            facecolors=color,
            edgecolors="none",
            linewidths=0.0,
            alpha=alpha,
            zorder=zorder,
            label=label,
        )
        ax.add_collection(collection)
        return [collection]

    def _get_process_path_bounds(self):
        """返回工艺信息每一行的累计行程区间。"""
        if not self.data:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        start_bounds = []
        end_bounds = []
        running = 0.0
        for row in self.data:
            try:
                start_val = float(row.get("path_start"))
                end_val = float(row.get("path_end"))
            except Exception:
                start_val = running
                try:
                    running += float(row.get("s", 0.0))
                except Exception:
                    pass
                end_val = running
            else:
                running = end_val
            start_bounds.append(start_val)
            end_bounds.append(end_val)
        return np.asarray(start_bounds, dtype=float), np.asarray(end_bounds, dtype=float)

    def _build_sample_path_positions(self, sample_line_numbers, sample_point_indices):
        """将 SampleData 点映射到工艺信息累计行程轴。"""
        if sample_line_numbers is None or self.sample_data_base_blocks is None or not self.data:
            return None

        process_line_numbers = []
        for idx, row in enumerate(self.data):
            line_no = row.get("line_no_raw")
            try:
                process_line_numbers.append(int(line_no))
            except Exception:
                process_line_numbers.append(idx)
        process_line_numbers = np.asarray(process_line_numbers, dtype=int)
        process_blocks = self.compute_sequence_blocks(process_line_numbers)
        process_block_ids = self._build_block_id_array(process_blocks, len(process_line_numbers))
        process_start_bounds, process_end_bounds = self._get_process_path_bounds()

        span_map = {}
        for idx, line_no in enumerate(process_line_numbers):
            block_id = int(process_block_ids[idx])
            key = (block_id, int(line_no))
            start_val = float(process_start_bounds[idx])
            end_val = float(process_end_bounds[idx])
            if not (np.isfinite(start_val) and np.isfinite(end_val)):
                continue
            if key not in span_map:
                span_map[key] = [start_val, end_val]
            else:
                span_map[key][0] = min(span_map[key][0], start_val)
                span_map[key][1] = max(span_map[key][1], end_val)

        sample_line_arr = np.asarray(sample_line_numbers, dtype=int)
        if sample_point_indices is None or len(sample_point_indices) != len(sample_line_arr):
            sample_point_indices = np.asarray(
                self.compute_line_point_indices(sample_line_arr, blocks=self.sample_data_base_blocks),
                dtype=int
            )
        else:
            sample_point_indices = np.asarray(sample_point_indices, dtype=int)
        sample_block_ids = self._build_block_id_array(self.sample_data_base_blocks, len(sample_line_arr))

        sample_counts = collections.Counter(
            (int(block_id), int(line_no))
            for block_id, line_no in zip(sample_block_ids, sample_line_arr)
        )
        positions = np.full(len(sample_line_arr), np.nan, dtype=float)
        for idx, line_no in enumerate(sample_line_arr):
            key = (int(sample_block_ids[idx]), int(line_no))
            if key not in span_map:
                continue
            start_val, end_val = span_map[key]
            if not np.isfinite(start_val) or not np.isfinite(end_val):
                continue
            point_count = max(int(sample_counts.get(key, 1)), 1)
            point_idx = max(int(sample_point_indices[idx]), 0)
            if end_val <= start_val or point_count <= 1:
                positions[idx] = end_val if np.isfinite(end_val) else start_val
            else:
                positions[idx] = start_val + (end_val - start_val) * ((point_idx + 0.5) / point_count)
        return positions

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
        segmentation_result = getattr(self, "_latest_segmentation_result", None)
        if segmentation_result is None:
            messagebox.showwarning("无六态结果", "请先完成全行程六类划分")
            return

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_dir = str(OUTPUT_DIR)
        try:
            process_info_path = self._save_process_info_csv(
                output_dir,
                segmentation_result,
            )
        except Exception as exc:
            messagebox.showerror("ProcessInfo导出失败", str(exc))
            return
        if not process_info_path:
            messagebox.showwarning("无工艺信息", "当前没有可导出的工艺信息")
            return

        mapping_valid = bool(
            str(getattr(self, "_sample_mapping_status", "") or "") == "valid"
            and str(getattr(self, "_current_mapping_signature", "") or "")
            and getattr(self, "_segmentation_sample_projection_records", None)
        )
        if not mapping_valid:
            mapping_status = str(
                getattr(self, "_sample_mapping_status", "not_available")
                or "not_available"
            )
            mapping_note = (
                "实际采样映射失败，未生成或覆盖 SampleData.rg。"
                if mapping_status == "failed"
                else "未导入可映射的实测数据，因此未生成或覆盖 SampleData.rg。"
            )
            feed_export_note = str(
                getattr(self, "_last_processinfo_feed_export_status", "") or ""
            )
            self.status_var_data.set(
                f"工艺信息已保存: {os.path.basename(process_info_path)}；{feed_export_note}"
                if feed_export_note
                else f"工艺信息已保存: {os.path.basename(process_info_path)}"
            )
            messagebox.showinfo(
                "保存成功",
                f"已导出: {os.path.basename(process_info_path)}\n"
                + mapping_note
                + (f"\n{feed_export_note}" if feed_export_note else ""),
            )
            return
        if not self.sample_programs:
            messagebox.showwarning(
                "无程序信息",
                f"{os.path.basename(process_info_path)} 已保存；"
                "缺少程序/刀具信息，未生成 SampleData.rg",
            )
            return
        if not self._get_optimizable_interval_records():
            messagebox.showwarning(
                "无可优化区间",
                f"{os.path.basename(process_info_path)} 已保存；"
                "当前没有可映射的空载或稳态区间，未生成 SampleData.rg",
            )
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
            messagebox.showwarning(
                "无可导出数据",
                f"{os.path.basename(process_info_path)} 已保存；"
                "未找到可导出的刀具区间信息，未生成 SampleData.rg",
            )
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
        if (
            str(getattr(self, "_current_interval_source", "") or "") == "segmentation"
            and not (
                str(getattr(self, "_sample_mapping_status", "") or "") == "valid"
                and str(getattr(self, "_current_mapping_signature", "") or "")
                and getattr(self, "_segmentation_sample_projection_records", None)
            )
        ):
            messagebox.showwarning(
                "采样映射无效",
                "过程域结果仍然有效，但实际采样映射尚未成功，未覆盖已有 SampleData.rg。",
            )
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_dir = str(OUTPUT_DIR)
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
                process_info_path = self._save_process_info_csv(output_dir)
                feed_export_note = str(
                    getattr(self, "_last_processinfo_feed_export_status", "") or ""
                )
                status_text = (
                    f"区间信息已保存: {os.path.basename(output_path)} (共{saved_lines}行)"
                )
                if feed_export_note:
                    status_text += f"；{feed_export_note}"
                self.status_var_data.set(status_text)

                saved_programs = {program_name for program_name, _, _ in tools_to_save}
                self._save_process_data_paths(output_dir, saved_programs)

                detail = "结果已保存，请关闭该窗口"
                if process_info_path:
                    detail += f"\n同时导出: {os.path.basename(process_info_path)}"
                if feed_export_note:
                    detail += f"\n{feed_export_note}"
                messagebox.showinfo("保存成功", detail)
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

    def _save_process_info_csv(self, save_dir, segmentation_result=None):
        """导出当前工艺信息，并在末列写入逐点六态 state_code。"""
        if not self.data:
            return None

        resolved_result = (
            segmentation_result
            or getattr(self, "_latest_segmentation_result", None)
        )
        if resolved_result is None:
            raise ValueError("缺少六态划分结果，无法写入 state_code")
        point_labels = self._coerce_segmentation_dataframe(
            resolved_result,
            "point_labels",
        )
        required_columns = {"source_index", "state_code"}
        missing_columns = sorted(required_columns.difference(point_labels.columns))
        if missing_columns:
            raise ValueError(f"六态逐点结果缺少列: {missing_columns}")

        source_indices = pd.to_numeric(
            point_labels["source_index"],
            errors="coerce",
        )
        state_codes = pd.to_numeric(
            point_labels["state_code"],
            errors="coerce",
        )
        if source_indices.isna().any() or state_codes.isna().any():
            raise ValueError("六态逐点结果包含无效的 source_index 或 state_code")
        if not np.equal(source_indices, np.floor(source_indices)).all():
            raise ValueError("六态逐点结果的 source_index 必须为整数")
        if not np.equal(state_codes, np.floor(state_codes)).all():
            raise ValueError("六态逐点结果的 state_code 必须为整数")
        state_codes = state_codes.astype(int)
        if not state_codes.isin(range(6)).all():
            raise ValueError("六态逐点结果包含 0..5 之外的 state_code")

        state_by_source = {}
        for source_index, state_code_value in zip(
            source_indices.astype(int),
            state_codes,
        ):
            source_index_value = int(source_index)
            if source_index_value in state_by_source:
                raise ValueError(
                    f"六态逐点结果包含重复 source_index: {source_index_value}"
                )
            state_by_source[source_index_value] = int(state_code_value)

        process_rows = [
            (source_index, row)
            for source_index, row in enumerate(self.data)
            if isinstance(row, dict) and not bool(row.get("_is_synthetic_fill"))
        ]
        expected_sources = {source_index for source_index, _row in process_rows}
        actual_sources = set(state_by_source)
        if expected_sources != actual_sources:
            missing_sources = sorted(expected_sources.difference(actual_sources))
            extra_sources = sorted(actual_sources.difference(expected_sources))
            details = []
            if missing_sources:
                details.append(f"缺少工艺行 {missing_sources[:10]}")
            if extra_sources:
                details.append(f"存在未知工艺行 {extra_sources[:10]}")
            raise ValueError("六态逐点结果与工艺信息行不一致：" + "；".join(details))

        export_feeds = np.asarray(
            [
                pd.to_numeric(row.get("feed_effective"), errors="coerce")
                for _source_index, row in process_rows
            ],
            dtype=float,
        )
        if bool(getattr(self, "release_mode", False)):
            resolver = getattr(self, "_resolve_processinfo_export_feeds", None)
            if callable(resolver):
                export_feeds = np.asarray(
                    resolver([row for _source_index, row in process_rows]),
                    dtype=float,
                )
                if export_feeds.size != len(process_rows):
                    raise ValueError("ProcessInfo 指令进给映射长度与工艺信息行数不一致")

        output_path = Path(save_dir) / "ProcessInfo.csv"
        temporary_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "N",
            "S(r/min)",
            "ap(mm)",
            "ae(mm)",
            "F(mm/min)",
            "s(mm)",
            "MRR(mm3/s)",
            "G",
            "state_code",
        ]

        def _as_csv_value(value):
            if value is None:
                return ""
            if isinstance(value, float):
                if not np.isfinite(value):
                    return ""
                return f"{value:.6f}".rstrip("0").rstrip(".")
            return value

        def _export_line_number(row):
            try:
                raw_line = float(row.get("line_no_raw"))
            except (TypeError, ValueError):
                raw_line = float("nan")
            if np.isfinite(raw_line):
                return f"N{int(round(raw_line)) + 1}"
            return _as_csv_value(row.get("N_str"))

        def _calculated_mrr(row, feed_value):
            try:
                ap_value = max(float(row.get("ap", 0.0) or 0.0), 0.0)
                ae_value = max(float(row.get("ae", 0.0) or 0.0), 0.0)
                feed_value = max(float(feed_value), 0.0)
            except (TypeError, ValueError):
                return ""
            if not all(np.isfinite(value) for value in (ap_value, ae_value, feed_value)):
                return ""
            return ap_value * ae_value * feed_value / 60.0

        try:
            with temporary_path.open(
                "w",
                encoding="utf-8-sig",
                newline="",
            ) as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(header)
                for (source_index, row), feed_value in zip(process_rows, export_feeds):
                    writer.writerow([
                        _export_line_number(row),
                        _as_csv_value(row.get("S")),
                        _as_csv_value(row.get("ap")),
                        _as_csv_value(row.get("ae")),
                        _as_csv_value(float(feed_value)),
                        _as_csv_value(row.get("s")),
                        _as_csv_value(_calculated_mrr(row, feed_value)),
                        _as_csv_value(row.get("gcode_content")),
                        state_by_source[source_index],
                    ])
            temporary_path.replace(output_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return str(output_path)

    def export_i_code(self):
        """保存结果文件 SampleData.rg"""
        if not self.ideal_store:
            messagebox.showwarning("未保存优化倍率", "请先设定优化倍率或批量生成理想值")
            return
        if not self.sample_programs:
            messagebox.showwarning("无程序信息", "请先加载 SampleData.txt")
            return
        if not self._get_optimizable_interval_records():
            messagebox.showwarning(
                "无可优化区间",
                "请先生成空载或稳态区间",
            )
            return

        # 源码研究版写项目 output/；冻结发布版直接写 EXE 同目录。
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_dir = str(OUTPUT_DIR)
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
            process_info_path = self._save_process_info_csv(output_dir)
            feed_export_note = str(
                getattr(self, "_last_processinfo_feed_export_status", "") or ""
            )
            status_text = f"结果已保存: {os.path.basename(output_path)} (共{saved_lines}行)"
            if feed_export_note:
                status_text += f"；{feed_export_note}"
            self.status_var_data.set(status_text)
            detail = f"结果已保存:\n{output_path}\n共 {saved_lines} 行"
            if process_info_path:
                detail += f"\n同时导出:\n{process_info_path}"
            if feed_export_note:
                detail += f"\n{feed_export_note}"
            messagebox.showinfo("保存完成", detail)
        except Exception as e:
            messagebox.showerror("导出失败", f"保存结果时发生错误:\n{str(e)}")

    def calculate_additional_columns(
        self,
        ap,
        ae,
        feed_rate,
        s,
        current_s,
        s_base,
        k_base,
        kc_value=None,
        ke_value=None,
        idle_power=None,
        fallback_speed=None,
    ):
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

            effective_speed = float(current_s) if current_s and float(current_s) > 0 else float(
                fallback_speed if fallback_speed is not None else (self.current_program_speed.get() or s_base)
            )
            p_idle = self.predict_idle_power(effective_speed) if idle_power is None else float(idle_power)
            kc_resolved = self.get_kc_value() if kc_value is None else float(kc_value)
            ke_resolved = self.get_ke_value() if ke_value is None else float(ke_value)
            cutting_power = kc_resolved * mrr_val
            edge_power = ke_resolved * ap_val
            p_power = p_idle + cutting_power + edge_power

            angular_velocity = 2 * math.pi * current_s / 60.0
            if angular_velocity > 1e-9:
                t_torque = p_power / angular_velocity
            else:
                t_torque = 0.0

            return t_val, dmrv_val, mrr_val, kc_resolved, t_torque, p_power, p_idle, edge_power

        except ValueError:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def _resolve_prediction_band_tolerance(self, reference_value, sigma_idle=0.0):
        """预测负载稳态带宽：取相对阈值和绝对阈值中的较大者。"""
        try:
            relative_threshold = float(self.steady_threshold.get())
        except Exception:
            relative_threshold = 0.02
        if not np.isfinite(relative_threshold) or relative_threshold <= 0.0:
            relative_threshold = 0.02
        # 旧默认值 0.2 对逐点反解后的样本级曲线过宽，这里收敛到更适合稳态划分的区间。
        relative_threshold = min(max(relative_threshold, 0.005), 0.05)

        try:
            absolute_threshold = float(self.segment_abs_threshold.get())
        except Exception:
            absolute_threshold = 20.0
        if not np.isfinite(absolute_threshold) or absolute_threshold <= 0.0:
            absolute_threshold = 20.0
        # 历史配置常用 0.05 表示 0.05kW，这里统一换算到 W。
        if absolute_threshold < 1.0:
            absolute_threshold *= 1000.0
        absolute_threshold = max(absolute_threshold, 3.0 * max(float(sigma_idle or 0.0), 0.0), 10.0)

        reference = abs(float(reference_value)) if np.isfinite(reference_value) else 0.0
        return max(float(absolute_threshold), float(reference) * float(relative_threshold))

    def _resolve_measurement_steady_limits(self, sigma_idle=0.0):
        sigma_ref = max(float(sigma_idle or 0.0), 0.0)
        variance_limit = float("inf")
        diff_std_limit = float("inf")
        if sigma_ref > 0.0:
            variance_limit = float((3.0 * sigma_ref) ** 2)
            diff_std_limit = float(3.0 * math.sqrt(2.0) * sigma_ref)
        return variance_limit, diff_std_limit

    def _resolve_capped_measurement_limit(
        self,
        reference_value,
        sigma_idle=0.0,
        sigma_multiplier=3.0,
        relative_ratio=0.2,
        floor_value=40.0,
    ):
        floor_limit = max(float(floor_value or 0.0), 0.0)
        sigma_ref = max(float(sigma_idle or 0.0), 0.0)
        sigma_limit = max(floor_limit, float(sigma_multiplier) * sigma_ref)
        reference = abs(float(reference_value)) if np.isfinite(reference_value) else 0.0
        if reference > 0.0 and float(relative_ratio or 0.0) > 0.0:
            sigma_limit = min(sigma_limit, max(floor_limit, reference * float(relative_ratio)))
        return float(max(floor_limit, sigma_limit))

    def _evaluate_measurement_steady_gate(self, actual_values, sigma_idle=0.0, sample_count=None, min_sample_count=1):
        finite_actual = np.asarray(actual_values, dtype=float)
        finite_actual = finite_actual[np.isfinite(finite_actual)]
        effective_count = int(sample_count) if sample_count is not None else int(finite_actual.size)
        if effective_count < 0:
            effective_count = 0

        if finite_actual.size == 0:
            variance_limit, diff_std_limit = self._resolve_measurement_steady_limits(sigma_idle)
            return {
                "sample_count": int(effective_count),
                "p_meas": float("nan"),
                "actual_load_var": 0.0,
                "actual_load_std": 0.0,
                "actual_load_diff_std": 0.0,
                "actual_load_span": 0.0,
                "actual_load_drift": 0.0,
                "actual_load_slope": 0.0,
                "variance_limit": float(variance_limit),
                "diff_std_limit": float(diff_std_limit),
                "span_limit": 0.0,
                "drift_limit": 0.0,
                "slope_limit": 0.0,
                "steady_pass": False,
            }

        actual_load_mean = float(np.mean(finite_actual)) if finite_actual.size else float("nan")
        actual_load_var = float(np.var(finite_actual, ddof=1)) if finite_actual.size > 1 else 0.0
        actual_load_std = float(math.sqrt(max(actual_load_var, 0.0)))
        diff_values = np.diff(finite_actual) if finite_actual.size > 1 else np.asarray([], dtype=float)
        actual_load_diff_std = float(np.std(diff_values, ddof=1)) if diff_values.size > 1 else 0.0
        actual_load_span = float(np.max(finite_actual) - np.min(finite_actual)) if finite_actual.size > 0 else 0.0
        actual_load_drift = float(abs(finite_actual[-1] - finite_actual[0])) if finite_actual.size > 1 else 0.0
        if finite_actual.size > 2:
            x_axis = np.arange(finite_actual.size, dtype=float)
            try:
                actual_load_slope = float(abs(np.polyfit(x_axis, finite_actual, 1)[0]))
            except Exception:
                actual_load_slope = float(actual_load_drift / max(finite_actual.size - 1, 1))
        else:
            actual_load_slope = float(actual_load_drift / max(finite_actual.size - 1, 1)) if finite_actual.size > 1 else 0.0

        variance_limit, diff_std_limit = self._resolve_measurement_steady_limits(sigma_idle)
        span_limit = self._resolve_capped_measurement_limit(
            actual_load_mean,
            sigma_idle=sigma_idle,
            sigma_multiplier=4.0,
            relative_ratio=0.22,
            floor_value=60.0,
        )
        drift_limit = self._resolve_capped_measurement_limit(
            actual_load_mean,
            sigma_idle=sigma_idle,
            sigma_multiplier=3.0,
            relative_ratio=0.16,
            floor_value=40.0,
        )
        slope_limit = max(0.2, float(drift_limit) / max(finite_actual.size - 1, 1))
        steady_pass = effective_count >= max(int(min_sample_count), 1)
        if np.isfinite(variance_limit):
            steady_pass = steady_pass and actual_load_var <= variance_limit
        if np.isfinite(diff_std_limit):
            steady_pass = steady_pass and actual_load_diff_std <= diff_std_limit
        apply_ramp_gate = abs(actual_load_mean) >= max(300.0, 3.0 * max(float(sigma_idle or 0.0), 0.0))
        if apply_ramp_gate:
            steady_pass = steady_pass and actual_load_span <= span_limit
            steady_pass = steady_pass and actual_load_drift <= drift_limit
            steady_pass = steady_pass and actual_load_slope <= slope_limit

        return {
            "sample_count": int(effective_count),
            "p_meas": float(actual_load_mean) if np.isfinite(actual_load_mean) else float("nan"),
            "actual_load_var": float(actual_load_var),
            "actual_load_std": float(actual_load_std),
            "actual_load_diff_std": float(actual_load_diff_std),
            "actual_load_span": float(actual_load_span),
            "actual_load_drift": float(actual_load_drift),
            "actual_load_slope": float(actual_load_slope),
            "variance_limit": float(variance_limit),
            "diff_std_limit": float(diff_std_limit),
            "span_limit": float(span_limit),
            "drift_limit": float(drift_limit),
            "slope_limit": float(slope_limit),
            "steady_pass": bool(steady_pass),
        }

    def _evaluate_measurement_alignment_gate(self, actual_values, predicted_values, sigma_idle=0.0):
        actual_arr = np.asarray(actual_values, dtype=float)
        predicted_arr = np.asarray(predicted_values, dtype=float)
        finite_mask = np.isfinite(actual_arr) & np.isfinite(predicted_arr)
        if not np.any(finite_mask):
            return {
                "mean_actual": float("nan"),
                "mean_predicted": float("nan"),
                "mean_residual": float("nan"),
                "residual_rms": float("nan"),
                "residual_limit": float("nan"),
                "steady_pass": False,
            }

        actual_seg = actual_arr[finite_mask]
        predicted_seg = predicted_arr[finite_mask]
        mean_actual = float(np.mean(actual_seg))
        mean_predicted = float(np.mean(predicted_seg))
        residual_values = actual_seg - predicted_seg
        mean_residual = float(abs(mean_actual - mean_predicted))
        residual_rms = float(np.sqrt(np.mean(np.square(residual_values)))) if residual_values.size > 0 else 0.0
        reference_level = max(abs(mean_actual), abs(mean_predicted), 0.0)
        residual_limit = self._resolve_capped_measurement_limit(
            reference_level,
            sigma_idle=sigma_idle,
            sigma_multiplier=2.5,
            relative_ratio=0.18,
            floor_value=45.0,
        )
        return {
            "mean_actual": float(mean_actual),
            "mean_predicted": float(mean_predicted),
            "mean_residual": float(mean_residual),
            "residual_rms": float(residual_rms),
            "residual_limit": float(residual_limit),
            "steady_pass": bool(mean_residual <= residual_limit),
        }

    def _evaluate_curve_flatness_gate(self, curve_values, sigma_idle=0.0):
        finite_curve = np.asarray(curve_values, dtype=float)
        finite_curve = finite_curve[np.isfinite(finite_curve)]
        if finite_curve.size == 0:
            return {
                "curve_span": float("nan"),
                "curve_drift": float("nan"),
                "curve_flat_limit": float("nan"),
                "curve_drift_limit": float("nan"),
                "curve_diff_std": float("nan"),
                "curve_diff_mean_abs": float("nan"),
                "curve_jump_max": float("nan"),
                "curve_slope": float("nan"),
                "curve_diff_std_limit": float("nan"),
                "curve_diff_mean_limit": float("nan"),
                "curve_jump_limit": float("nan"),
                "curve_slope_limit": float("nan"),
                "curve_gate_mode": "reject",
                "steady_pass": False,
            }

        curve_center = float(np.median(finite_curve))
        band_tolerance = self._resolve_prediction_band_tolerance(curve_center, sigma_idle=sigma_idle)
        sample_count = int(finite_curve.size)
        curve_span = float(np.max(finite_curve) - np.min(finite_curve))
        curve_drift = float(abs(finite_curve[-1] - finite_curve[0]))
        curve_flat_limit = max(15.0, float(band_tolerance) * 0.40)
        curve_drift_limit = max(12.0, float(band_tolerance) * 0.30)
        diff_values = np.diff(finite_curve) if sample_count > 1 else np.asarray([], dtype=float)
        curve_diff_std = float(np.std(diff_values, ddof=1)) if diff_values.size > 1 else 0.0
        curve_diff_mean_abs = float(np.mean(np.abs(diff_values))) if diff_values.size > 0 else 0.0
        curve_jump_max = float(np.max(np.abs(diff_values))) if diff_values.size > 0 else 0.0
        if sample_count > 2:
            x_axis = np.arange(sample_count, dtype=float)
            try:
                curve_slope = float(abs(np.polyfit(x_axis, finite_curve, 1)[0]))
            except Exception:
                curve_slope = float(abs(finite_curve[-1] - finite_curve[0]) / max(sample_count - 1, 1))
        else:
            curve_slope = float(abs(finite_curve[-1] - finite_curve[0]) / max(sample_count - 1, 1)) if sample_count > 1 else 0.0

        curve_diff_std_limit = max(1.5, float(band_tolerance) * 0.008)
        curve_diff_mean_limit = max(1.5, float(band_tolerance) * 0.007)
        curve_jump_limit = max(4.0, float(band_tolerance) * 0.03)
        # 一般切削允许缓慢漂移，但不接受持续爬坡或下坡段。
        curve_slope_limit = max(0.12, float(band_tolerance) / 1500.0)
        flat_pass = bool(curve_span <= curve_flat_limit and curve_drift <= curve_drift_limit)
        fluctuation_pass = bool(
            sample_count >= 8
            and curve_diff_std <= curve_diff_std_limit
            and curve_diff_mean_abs <= curve_diff_mean_limit
            and curve_jump_max <= curve_jump_limit
            and curve_slope <= curve_slope_limit
        )
        gate_mode = "flat" if flat_pass else ("fluctuation" if fluctuation_pass else "reject")
        return {
            "curve_span": float(curve_span),
            "curve_drift": float(curve_drift),
            "curve_flat_limit": float(curve_flat_limit),
            "curve_drift_limit": float(curve_drift_limit),
            "curve_diff_std": float(curve_diff_std),
            "curve_diff_mean_abs": float(curve_diff_mean_abs),
            "curve_jump_max": float(curve_jump_max),
            "curve_slope": float(curve_slope),
            "curve_diff_std_limit": float(curve_diff_std_limit),
            "curve_diff_mean_limit": float(curve_diff_mean_limit),
            "curve_jump_limit": float(curve_jump_limit),
            "curve_slope_limit": float(curve_slope_limit),
            "curve_gate_mode": gate_mode,
            "steady_pass": bool(flat_pass or fluctuation_pass),
        }

    def _split_sample_block_by_curve(self, curve_values, block_start, block_end, idle_mask=None, sigma_idle=0.0):
        """按预测负载阈值带宽切分样本块。"""
        values = np.asarray(curve_values, dtype=float)
        idle_flags = None
        if idle_mask is not None:
            idle_arr = np.asarray(idle_mask, dtype=bool)
            if idle_arr.size == values.size:
                idle_flags = idle_arr

        segments = []
        seg_start = None
        seg_min = float("nan")
        seg_max = float("nan")
        seg_sum = 0.0
        seg_count = 0
        seg_idle = False

        def _flush(end_idx):
            nonlocal seg_start, seg_min, seg_max, seg_sum, seg_count, seg_idle
            if seg_start is not None and seg_start <= end_idx:
                segments.append((int(seg_start), int(end_idx)))
            seg_start = None
            seg_min = float("nan")
            seg_max = float("nan")
            seg_sum = 0.0
            seg_count = 0
            seg_idle = False

        for idx in range(int(block_start), int(block_end) + 1):
            if idx >= len(values) or not np.isfinite(values[idx]):
                if seg_start is not None and seg_start <= idx - 1:
                    _flush(idx - 1)
                continue

            value = float(values[idx])
            current_idle = bool(idle_flags[idx]) if idle_flags is not None else False
            if seg_start is None:
                seg_start = idx
                seg_min = value
                seg_max = value
                seg_sum = value
                seg_count = 1
                seg_idle = current_idle
                continue

            if current_idle != seg_idle:
                _flush(idx - 1)
                seg_start = idx
                seg_min = value
                seg_max = value
                seg_sum = value
                seg_count = 1
                seg_idle = current_idle
                continue

            next_min = min(seg_min, value)
            next_max = max(seg_max, value)
            next_sum = seg_sum + value
            next_count = seg_count + 1
            center_value = next_sum / float(max(next_count, 1))
            tolerance = self._resolve_prediction_band_tolerance(center_value, sigma_idle=sigma_idle)
            if (next_max - next_min) > tolerance:
                _flush(idx - 1)
                seg_start = idx
                seg_min = value
                seg_max = value
                seg_sum = value
                seg_count = 1
                seg_idle = current_idle
            else:
                seg_min = next_min
                seg_max = next_max
                seg_sum = next_sum
                seg_count = next_count

        if seg_start is not None and seg_start <= int(block_end):
            _flush(int(block_end))
        return segments

    def _partition_prediction_intervals_from_measurement(self, measurement):
        if not measurement:
            return []
        if self._is_imported_profile_forward_lock_active():
            active_profile = getattr(self, "imported_kc_profile", None) or getattr(self, "active_kc_profile", None)
            if isinstance(active_profile, dict):
                return self._extract_profile_interval_records(active_profile)
            return []

        predicted_load = np.asarray(measurement.get("predicted_load", []), dtype=float)
        if predicted_load.size == 0:
            return []

        expected_size = predicted_load.size
        actual_load = np.abs(np.asarray(measurement.get("actual_load", []), dtype=float))
        raw_lines = np.asarray(measurement.get("program_line", []), dtype=int)
        if raw_lines.size != expected_size:
            raw_lines = np.arange(expected_size, dtype=int)
        aligned_lines = np.asarray(measurement.get("line_no_aligned", raw_lines), dtype=int)
        if aligned_lines.size != expected_size:
            aligned_lines = raw_lines.copy()

        idle_power = np.asarray(measurement.get("predicted_idle_power", []), dtype=float)
        if idle_power.size != expected_size:
            idle_power = np.full(expected_size, np.nan, dtype=float)
        ap_values = np.asarray(measurement.get("mapped_ap", []), dtype=float)
        if ap_values.size != expected_size:
            ap_values = np.zeros(expected_size, dtype=float)
        ae_values = np.asarray(measurement.get("mapped_ae", []), dtype=float)
        if ae_values.size != expected_size:
            ae_values = np.zeros(expected_size, dtype=float)
        feed_values = np.asarray(measurement.get("mapped_feed", []), dtype=float)
        if feed_values.size != expected_size:
            feed_values = np.zeros(expected_size, dtype=float)
        mrr_values = np.asarray(measurement.get("mapped_mrr", []), dtype=float)
        if mrr_values.size != expected_size:
            mrr_values = np.zeros(expected_size, dtype=float)

        kc_points = np.asarray(measurement.get("kc_point", []), dtype=float)
        if kc_points.size != expected_size:
            kc_points = np.full(expected_size, np.nan, dtype=float)
        kc_valid_mask = np.asarray(measurement.get("kc_valid_mask", []), dtype=bool)
        if kc_valid_mask.size != expected_size:
            kc_valid_mask = np.zeros(expected_size, dtype=bool)
        sample_kc_values = np.asarray(measurement.get("sample_kc_values", []), dtype=float)
        if sample_kc_values.size != expected_size:
            sample_kc_values = np.full(expected_size, np.nan, dtype=float)
        sample_kc_valid_mask = np.asarray(measurement.get("sample_kc_valid_mask", []), dtype=bool)
        if sample_kc_valid_mask.size != expected_size:
            sample_kc_valid_mask = np.zeros(expected_size, dtype=bool)
        kc_gated_out_mask = np.asarray(measurement.get("kc_gated_out_mask", []), dtype=bool)
        if kc_gated_out_mask.size != expected_size:
            kc_gated_out_mask = np.zeros(expected_size, dtype=bool)

        idle_point_mask = np.asarray(measurement.get("idle_point_mask", []), dtype=bool)
        if idle_point_mask.size != expected_size:
            idle_point_mask = (
                (mrr_values <= 1e-12)
                | (
                    np.isfinite(predicted_load)
                    & np.isfinite(idle_power)
                    & (predicted_load <= idle_power + 1e-9)
                )
            )

        prediction_valid_mask = np.asarray(measurement.get("prediction_valid_mask", []), dtype=bool)
        if prediction_valid_mask.size != expected_size:
            prediction_valid_mask = np.ones(expected_size, dtype=bool)
        context_mask = prediction_valid_mask & np.isfinite(predicted_load)
        if not np.any(context_mask):
            return []

        sample_context = self._get_current_sample_line_point_context(raw_lines)
        sample_line_numbers = raw_lines
        sample_point_indices = None
        sample_x_positions = None
        sample_time_positions = None
        sample_point_widths = None
        if sample_context:
            sample_line_numbers = np.asarray(sample_context.get("line_numbers", raw_lines), dtype=int)
            sample_point_indices = np.asarray(sample_context.get("point_indices", []), dtype=int)
            sample_x_positions = sample_context.get("x_positions")
            sample_time_positions = sample_context.get("time_positions")
            sample_point_widths = sample_context.get("point_widths")
        if sample_point_indices is None or len(sample_point_indices) != expected_size:
            sample_point_indices = np.asarray(self.compute_line_point_indices(sample_line_numbers), dtype=int)
        if sample_x_positions is None or len(sample_x_positions) != expected_size:
            sample_x_positions = np.arange(expected_size, dtype=float)
        else:
            sample_x_positions = np.asarray(sample_x_positions, dtype=float)
        if sample_time_positions is not None and len(sample_time_positions) == expected_size:
            sample_time_positions = np.asarray(sample_time_positions, dtype=float)
        else:
            sample_time_positions = None
        if sample_point_widths is None or len(sample_point_widths) != expected_size:
            sample_point_widths = np.asarray(self.compute_line_point_widths(sample_line_numbers), dtype=float)
        else:
            sample_point_widths = np.asarray(sample_point_widths, dtype=float)

        process_aligned_lines = np.asarray(
            [int(row.get("line_no_aligned", idx)) for idx, row in enumerate(self.data or [])],
            dtype=int,
        )
        sample_process_point_indices = np.asarray(measurement.get("process_point_index", []), dtype=int)
        if sample_process_point_indices.size != expected_size:
            sample_process_point_indices = np.full(expected_size, -1, dtype=int)
        process_point_indices = np.asarray(
            [int(row.get("process_point_index", -1)) for row in (self.data or [])],
            dtype=int,
        )
        if process_point_indices.size != process_aligned_lines.size:
            process_point_indices = np.full(process_aligned_lines.shape, -1, dtype=int)
        pit_metadata = self.get_current_pit_metadata()
        sigma_idle = float(measurement.get("sigma_idle", 0.0) or 0.0)
        delta_mrr = float(measurement.get("delta_mrr", 0.0) or 0.0)
        min_sample_count = 5

        def _nearest_process_idx(line_value):
            if process_aligned_lines.size == 0:
                return None
            return int(np.argmin(np.abs(process_aligned_lines - int(line_value))))

        sample_blocks = []
        base_blocks = getattr(self, "sample_data_base_blocks", None) or [(0, expected_size - 1)]
        for base_start, base_end in base_blocks:
            if base_start > base_end or base_start >= expected_size:
                continue
            safe_end = min(int(base_end), expected_size - 1)
            local_mask = context_mask[int(base_start):safe_end + 1]
            for local_start, local_end in self.compute_contiguous_blocks(local_mask):
                sample_blocks.append((int(base_start + local_start), int(base_start + local_end)))

        intervals = []
        zone_index = 1
        for block_start, block_end in sample_blocks:
            curve_segments = self._split_sample_block_by_curve(
                predicted_load,
                block_start,
                block_end,
                idle_mask=idle_point_mask,
                sigma_idle=sigma_idle,
            )
            for curve_start, curve_end in curve_segments:
                curve_slice = slice(int(curve_start), int(curve_end) + 1)
                curve_idle_mask = idle_point_mask[curve_slice]
                is_idle_curve = bool(np.all(curve_idle_mask))
                if is_idle_curve:
                    segment_candidates = [(int(curve_start), int(curve_end))]
                else:
                    segment_candidates = self._split_sample_block_by_actual_load(actual_load, curve_start, curve_end, sigma_idle)
                    if not segment_candidates:
                        segment_candidates = [(int(curve_start), int(curve_end))]

                pending_segments = [(int(seg_start), int(seg_end)) for seg_start, seg_end in segment_candidates]
                while pending_segments:
                    seg_start, seg_end = pending_segments.pop(0)
                    if not is_idle_curve:
                        seg_start, seg_end = self._trim_segment_edges_by_actual_load(actual_load, seg_start, seg_end, sigma_idle)
                    seg_slice = slice(int(seg_start), int(seg_end) + 1)
                    seg_count = int(seg_end - seg_start + 1)
                    if seg_count <= 0:
                        continue

                    seg_pred = predicted_load[seg_slice]
                    finite_pred = seg_pred[np.isfinite(seg_pred)]
                    if finite_pred.size == 0:
                        continue

                    seg_actual = actual_load[seg_slice] if actual_load.size == expected_size else np.asarray([], dtype=float)
                    seg_idle = idle_power[seg_slice]
                    seg_ap = ap_values[seg_slice]
                    seg_ae = ae_values[seg_slice]
                    seg_feed = feed_values[seg_slice]
                    seg_idle_mask = idle_point_mask[seg_slice]
                    is_idle_interval = bool(np.all(seg_idle_mask))

                    steady_stats = self._evaluate_measurement_steady_gate(
                        seg_actual,
                        sigma_idle=sigma_idle,
                        sample_count=seg_count,
                        min_sample_count=min_sample_count,
                    )
                    curve_gate = {"steady_pass": True}
                    if not is_idle_interval:
                        curve_gate = self._evaluate_curve_flatness_gate(seg_pred, sigma_idle=sigma_idle)
                    alignment_gate = {"steady_pass": True}
                    if not is_idle_interval:
                        alignment_gate = self._evaluate_measurement_alignment_gate(seg_actual, seg_pred, sigma_idle=sigma_idle)
                    if (
                        not is_idle_interval
                        and bool(steady_stats.get("steady_pass", False))
                        and bool(alignment_gate.get("steady_pass", False))
                        and not bool(curve_gate.get("steady_pass", False))
                    ):
                        drift_limit = float(steady_stats.get("drift_limit", float("inf")))
                        actual_drift = float(steady_stats.get("actual_load_drift", float("inf")))
                        fallback_segments = self._split_measurement_segment_by_aligned_lines(
                            seg_start,
                            seg_end,
                            aligned_lines,
                        ) if (
                            np.isfinite(drift_limit)
                            and actual_drift <= max(25.0, drift_limit * 0.25)
                        ) else []
                        if fallback_segments:
                            self._debug_prediction_state_event(
                                "split_measurement_segment_by_aligned_lines",
                                segment_start=int(seg_start),
                                segment_end=int(seg_end),
                                split_count=len(fallback_segments),
                                live_display="posterior",
                            )
                            pending_segments = list(fallback_segments) + pending_segments
                            continue
                    if (
                        not bool(steady_stats.get("steady_pass", False))
                        or not bool(curve_gate.get("steady_pass", False))
                        or not bool(alignment_gate.get("steady_pass", False))
                    ):
                        continue

                    aligned_start_line = int(aligned_lines[seg_start])
                    aligned_end_line = int(aligned_lines[seg_end])
                    process_bounds = self._resolve_measurement_process_interval_bounds(
                        seg_start,
                        seg_end,
                        aligned_lines,
                        process_aligned_lines,
                        sample_process_point_indices=sample_process_point_indices,
                        process_point_indices=process_point_indices,
                    )
                    if process_bounds is not None:
                        proc_start_idx = int(process_bounds[0])
                        proc_end_idx = int(process_bounds[1])
                        interval_meta = self.summarize_process_interval(proc_start_idx, proc_end_idx)
                        start_n = self.data[proc_start_idx].get("N_str") if self.data and proc_start_idx < len(self.data) else None
                        end_n = self.data[proc_end_idx].get("N_str") if self.data and proc_end_idx < len(self.data) else None
                        try:
                            start_s = float(self.data[proc_start_idx].get("path_start", 0.0) or 0.0)
                        except Exception:
                            start_s = 0.0
                        try:
                            end_s = float(self.data[proc_end_idx].get("path_end", start_s) or start_s)
                        except Exception:
                            end_s = start_s
                    else:
                        nearest_start = _nearest_process_idx(aligned_start_line)
                        nearest_end = _nearest_process_idx(aligned_end_line)
                        proc_start_idx = int(nearest_start) if nearest_start is not None else 0
                        proc_end_idx = int(nearest_end) if nearest_end is not None else proc_start_idx
                        if proc_end_idx < proc_start_idx:
                            proc_start_idx, proc_end_idx = proc_end_idx, proc_start_idx
                        interval_meta = {
                            "a_p": float(np.nanmedian(seg_ap)) if np.any(np.isfinite(seg_ap)) else 0.0,
                            "a_e": float(np.nanmedian(seg_ae)) if np.any(np.isfinite(seg_ae)) else 0.0,
                            "F_plan": float(np.nanmedian(seg_feed)) if np.any(np.isfinite(seg_feed)) else 0.0,
                            "p_idle": float(np.nanmean(seg_idle)) if np.any(np.isfinite(seg_idle)) else float(self.p_idle_var.get()),
                        }
                        start_n = None
                        end_n = None
                        start_s = float(sample_x_positions[seg_start]) if seg_start < len(sample_x_positions) else 0.0
                        end_s = float(sample_x_positions[seg_end]) if seg_end < len(sample_x_positions) else start_s

                    resolved_start_line = int(process_aligned_lines[proc_start_idx]) if process_aligned_lines.size > proc_start_idx else int(aligned_start_line)
                    resolved_end_line = int(process_aligned_lines[proc_end_idx]) if process_aligned_lines.size > proc_end_idx else int(aligned_end_line)

                    valid_kc_values = kc_points[seg_slice][kc_valid_mask[seg_slice] & np.isfinite(kc_points[seg_slice])]
                    if valid_kc_values.size == 0:
                        valid_kc_values = sample_kc_values[seg_slice][
                            sample_kc_valid_mask[seg_slice] & np.isfinite(sample_kc_values[seg_slice])
                        ]
                    valid_kc_count = int(valid_kc_values.size)
                    gated_out_count = int(np.sum(kc_gated_out_mask[seg_slice]))
                    p_meas = float(steady_stats.get("p_meas", float("nan")))
                    actual_load_std = float(steady_stats.get("actual_load_std", 0.0))
                    actual_load_diff_std = float(steady_stats.get("actual_load_diff_std", 0.0))

                    if is_idle_interval:
                        kc_hat = 0.0
                        sigma_kc = 0.0
                        kc_source = "idle"
                    elif valid_kc_count > 0:
                        kc_hat, sigma_kc, _ = self._summarize_interval_kc_statistics(valid_kc_values)
                        kc_source = "measurement_mode"
                    else:
                        kc_hat = float("nan")
                        sigma_kc = float("nan")
                        kc_source = ""

                    sample_start_label = self.format_line_point(sample_line_numbers[seg_start], sample_point_indices[seg_start])
                    sample_end_label = self.format_line_point(sample_line_numbers[seg_end], sample_point_indices[seg_end])
                    display_start_x = float(sample_x_positions[seg_start]) if seg_start < len(sample_x_positions) else float("nan")
                    point_width = float(sample_point_widths[seg_end]) if seg_end < len(sample_point_widths) else 1.0
                    display_end_x = float(sample_x_positions[seg_end] + max(point_width, 1e-9)) if seg_end < len(sample_x_positions) else float("nan")
                    display_start_t = float("nan")
                    display_end_t = float("nan")
                    if sample_time_positions is not None:
                        display_start_t = float(sample_time_positions[seg_start])
                        display_end_t = float(sample_time_positions[seg_end] + 1.0)

                    interval_record = asdict(PITEntry(
                        zone_id=f"Z{zone_index:03d}",
                        start_idx=int(proc_start_idx),
                        end_idx=int(proc_end_idx),
                        start_line=int(resolved_start_line),
                        end_line=int(resolved_end_line),
                        start_s=float(start_s),
                        end_s=float(end_s),
                        a_p=float(interval_meta.get("a_p", 0.0) or 0.0),
                        a_e=float(interval_meta.get("a_e", 0.0) or 0.0),
                        F_plan=float(interval_meta.get("F_plan", 0.0) or 0.0),
                        p_idle=float(np.nanmean(seg_idle)) if np.any(np.isfinite(seg_idle)) else float(interval_meta.get("p_idle", self.p_idle_var.get())),
                        p_pred=float(np.mean(finite_pred)),
                        K_c_hat=float(kc_hat) if np.isfinite(kc_hat) else float("nan"),
                        K_c_UCB=0.0,
                        sigma_Kc=float(sigma_kc) if np.isfinite(sigma_kc) else float("nan"),
                        sample_count=int(seg_count),
                        start_label=sample_start_label,
                        end_label=sample_end_label,
                        process_start_label=self.format_line_point(
                            int(resolved_start_line),
                            int(process_point_indices[proc_start_idx]) if process_point_indices.size > proc_start_idx else 0,
                        ),
                        process_end_label=self.format_line_point(
                            int(resolved_end_line),
                            int(process_point_indices[proc_end_idx]) if process_point_indices.size > proc_end_idx else 0,
                        ),
                        sample_start_line=int(sample_line_numbers[seg_start]),
                        sample_end_line=int(sample_line_numbers[seg_end]),
                        display_start_x=float(display_start_x),
                        display_end_x=float(display_end_x),
                        display_start_t=float(display_start_t),
                        display_end_t=float(display_end_t),
                        valid_kc_count=int(valid_kc_count),
                        gated_out_count=int(gated_out_count),
                        p_meas=float(p_meas) if np.isfinite(p_meas) else 0.0,
                        actual_load_std=float(actual_load_std),
                        actual_load_diff_std=float(actual_load_diff_std),
                        sigma_idle=float(sigma_idle),
                        delta_mrr=float(delta_mrr),
                        kc_source=str(kc_source),
                        start_n=start_n,
                        end_n=end_n,
                        tool_diameter=pit_metadata["tool_diameter"],
                        tool_radius=pit_metadata["tool_radius"],
                        tool_material=pit_metadata["tool_material"],
                        blank_material=pit_metadata["blank_material"],
                    ))
                    interval_record["segment_type"] = "steady"
                    interval_record["steady_subtype"] = "idle" if is_idle_interval else "cutting"
                    interval_record["is_idle_interval"] = bool(is_idle_interval)
                    interval_record["steady_pass"] = True
                    interval_record["sample_start_idx"] = int(seg_start)
                    interval_record["sample_end_idx"] = int(seg_end)
                    interval_record["sample_start_label"] = str(sample_start_label)
                    interval_record["sample_end_label"] = str(sample_end_label)
                    interval_record["sample_anchor_start_idx"] = int(seg_start)
                    interval_record["sample_anchor_end_idx"] = int(seg_end)
                    interval_record["sample_anchor_start_label"] = str(sample_start_label)
                    interval_record["sample_anchor_end_label"] = str(sample_end_label)
                    interval_record["actual_load_span"] = float(steady_stats.get("actual_load_span", 0.0))
                    interval_record["actual_load_drift"] = float(steady_stats.get("actual_load_drift", 0.0))
                    interval_record["prediction_bias"] = float(alignment_gate.get("mean_residual", 0.0))
                    interval_record["prediction_bias_limit"] = float(alignment_gate.get("residual_limit", 0.0))
                    process_bounds = {
                        "start_idx": int(proc_start_idx),
                        "end_idx": int(proc_end_idx),
                        "start_line": int(resolved_start_line),
                        "end_line": int(resolved_end_line),
                    }
                    process_x_bounds = self._resolve_interval_process_x_bounds(interval_record, process_bounds=process_bounds)
                    if process_x_bounds:
                        interval_record.update(process_x_bounds)
                    intervals.append(interval_record)
                    zone_index += 1

        return self.finalize_interval_kc(intervals)

    def _split_sample_block_by_actual_load(self, actual_values, block_start, block_end, sigma_idle=0.0):
        """按实际负载波动二次细分样本块，便于只剔除不稳的局部。"""
        values = np.asarray(actual_values, dtype=float)
        segments = []
        seg_start = None
        sigma_ref = max(float(sigma_idle or 0.0), 0.0)

        for idx in range(int(block_start), int(block_end) + 1):
            if idx >= len(values) or not np.isfinite(values[idx]):
                if seg_start is not None and seg_start <= idx - 1:
                    segments.append((int(seg_start), int(idx - 1)))
                seg_start = None
                continue

            if seg_start is None:
                seg_start = idx
                continue

            seg_values = values[int(seg_start):int(idx) + 1]
            seg_values = seg_values[np.isfinite(seg_values)]
            if seg_values.size <= 1:
                continue

            seg_span = float(np.max(seg_values) - np.min(seg_values))
            seg_center = abs(float(np.median(seg_values)))
            seg_diff = np.diff(seg_values)
            max_jump = float(np.max(np.abs(seg_diff))) if seg_diff.size > 0 else 0.0

            span_tol = self._resolve_capped_measurement_limit(
                seg_center,
                sigma_idle=sigma_ref,
                sigma_multiplier=4.0,
                relative_ratio=0.22,
                floor_value=60.0,
            )
            jump_tol = self._resolve_capped_measurement_limit(
                seg_center,
                sigma_idle=sigma_ref,
                sigma_multiplier=2.5 * math.sqrt(2.0),
                relative_ratio=0.10,
                floor_value=24.0,
            )
            if seg_span > span_tol or max_jump > jump_tol:
                if seg_start <= idx - 1:
                    segments.append((int(seg_start), int(idx - 1)))
                seg_start = idx

        if seg_start is not None and seg_start <= int(block_end):
            segments.append((int(seg_start), int(block_end)))
        return segments

    def _trim_segment_edges_by_actual_load(self, actual_values, block_start, block_end, sigma_idle=0.0):
        """将区间收紧到实际负载更稳定的核心段，裁掉头尾过渡区。"""
        values = np.asarray(actual_values, dtype=float)
        if block_start > block_end or block_start < 0 or block_end >= len(values):
            return int(block_start), int(block_end)

        seg = values[int(block_start):int(block_end) + 1]
        finite_seg = seg[np.isfinite(seg)]
        if finite_seg.size < 3:
            return int(block_start), int(block_end)

        center = float(np.median(finite_seg))
        sigma_ref = max(float(sigma_idle or 0.0), 0.0)
        level_tol = self._resolve_capped_measurement_limit(
            center,
            sigma_idle=sigma_ref,
            sigma_multiplier=3.0,
            relative_ratio=0.18,
            floor_value=45.0,
        )
        jump_tol = self._resolve_capped_measurement_limit(
            center,
            sigma_idle=sigma_ref,
            sigma_multiplier=2.0 * math.sqrt(2.0),
            relative_ratio=0.08,
            floor_value=18.0,
        )
        stable_window = max(5, min(40, len(seg) // 20 if len(seg) >= 20 else len(seg)))

        def _window_stable(local_start, local_end):
            window = seg[int(local_start):int(local_end) + 1]
            window = window[np.isfinite(window)]
            if window.size < max(3, min(stable_window, len(seg)) // 2):
                return False
            window_center = float(np.median(window))
            window_span = float(np.max(window) - np.min(window))
            window_diff = np.diff(window)
            window_jump = float(np.max(np.abs(window_diff))) if window_diff.size > 0 else 0.0
            return (
                window_span <= max(level_tol, abs(window_center) * 0.03)
                and window_jump <= jump_tol
            )

        left = 0
        while left < len(seg) - 1:
            current = seg[left]
            nxt = seg[left + 1]
            right_probe = min(len(seg) - 1, left + stable_window - 1)
            if (
                np.isfinite(current)
                and np.isfinite(nxt)
                and abs(current - center) <= level_tol
                and abs(nxt - current) <= jump_tol
                and _window_stable(left, right_probe)
            ):
                break
            left += 1

        right = len(seg) - 1
        while right > left:
            current = seg[right]
            prev = seg[right - 1]
            left_probe = max(left, right - stable_window + 1)
            if (
                np.isfinite(current)
                and np.isfinite(prev)
                and abs(current - center) <= level_tol
                and abs(current - prev) <= jump_tol
                and _window_stable(left_probe, right)
            ):
                break
            right -= 1

        trimmed_start = int(block_start) + int(left)
        trimmed_end = int(block_start) + int(right)
        if trimmed_start > trimmed_end:
            return int(block_start), int(block_end)
        return int(trimmed_start), int(trimmed_end)

    def _resolve_pred_power_intervals(
        self,
        P_values,
        s_values,
        cumulative_s,
        n_values,
        line_numbers,
        *,
        model_ready=False,
        debug_line_range=None,
        interval_policy="fresh_or_empty",
        materialize_reused_current_template=True,
    ):
        sample_mode = str(getattr(self, "sample_data_mode", "") or "").strip()
        use_pred_power_steady = bool(model_ready and self.enable_pred_power_steady.get())
        imported_forward_lock = bool(
            hasattr(self, "_is_imported_profile_forward_lock_active")
            and self._is_imported_profile_forward_lock_active()
        )
        policy = "use_active_profile" if imported_forward_lock else str(interval_policy or "").strip()
        if policy not in {"use_active_profile", "reuse_current_template", "recompute_current", "fresh_or_empty"}:
            raise ValueError(f"Unsupported interval_policy: {interval_policy}")

        if policy == "use_active_profile":
            if (
                bool(getattr(self, "_current_interval_ready", False))
                and bool(getattr(self, "_profile_intervals_locked", False))
                and self._can_reuse_current_interval_template(
                    prediction_source="imported_profile",
                    measurement=getattr(self, "manual_measurement_data", None),
                )
            ):
                return self._get_current_interval_records(allow_profile_fallback=False)
            active_profile = self._get_saved_kc_profile_for_input()
            if isinstance(active_profile, dict):
                return self._extract_profile_interval_records(active_profile)
            return []

        if policy == "reuse_current_template":
            if self._can_reuse_current_interval_template(
                prediction_source=self._get_prediction_source(),
                measurement=getattr(self, "manual_measurement_data", None),
            ):
                current_records = self._get_current_interval_records(allow_profile_fallback=False)
                if current_records:
                    if bool(materialize_reused_current_template):
                        materialized = self._materialize_profile_pit_records(current_records)
                        return materialized if materialized else current_records
                    return [dict(record) for record in current_records if isinstance(record, dict)]
                return []

        if (
            not imported_forward_lock
            and
            sample_mode == "experiment_measurement"
            and getattr(self, "manual_measurement_data", None)
            and policy in {"recompute_current", "fresh_or_empty"}
            and hasattr(self, "_partition_prediction_intervals_from_measurement")
        ):
            measurement_intervals = self._partition_prediction_intervals_from_measurement(
                getattr(self, "manual_measurement_data", None)
            )
            if measurement_intervals:
                return [dict(interval) for interval in measurement_intervals if isinstance(interval, dict)]

        fresh_intervals = []
        if use_pred_power_steady and not imported_forward_lock:
            fresh_intervals = self.partition_pred_power_steady_intervals(
                P_values, s_values, cumulative_s, n_values, line_numbers, debug_line_range
            )
        if fresh_intervals:
            return [dict(interval) for interval in fresh_intervals if isinstance(interval, dict)]
        return []

    def generate_plots(
        self,
        save=False,
        silent=False,
        debug_line_range=None,
        interval_policy="fresh_or_empty",
        persist_profile=False,
        refresh_prediction=None,
    ):
        """生成图表
        :param debug_line_range: 可选，(start_line, end_line) 用于调试特定行号范围内的划分逻辑
        """
        if not self.data:
            messagebox.showwarning("无数据", "请先处理数据以生成图表")
            return False
        if persist_profile and hasattr(self, "_debug_interval_state_event"):
            self._debug_interval_state_event(
                "generate_plots_ignore_persist_profile",
                interval_policy=str(interval_policy),
            )
        
        try:
            segmentation_authoritative = bool(
                getattr(self, "_current_interval_ready", False)
                and str(getattr(self, "_current_interval_source", "") or "") == "segmentation"
            )
            if not segmentation_authoritative:
                runner = getattr(self, "run_full_path_segmentation", None)
                if not callable(runner):
                    raise RuntimeError("全行程六类划分入口不可用")
                if hasattr(self, "set_progress"):
                    self.set_progress(84, "正在执行全行程六类区间划分...")
                try:
                    self.root.update_idletasks()
                except Exception:
                    pass
                runner(
                    export_outputs=False,
                    refresh_view=False,
                    silent=True,
                )
                segmentation_authoritative = bool(
                    getattr(self, "_current_interval_ready", False)
                    and str(getattr(self, "_current_interval_source", "") or "") == "segmentation"
                )
                if not segmentation_authoritative:
                    if hasattr(self, "segmentation_status_var"):
                        self.segmentation_status_var.set("全行程六类划分: 失败，未执行旧区间划分")
                    self.status_var_data.set("全行程六类划分失败，图表未生成")
                    if not silent:
                        messagebox.showerror("区间划分失败", "未得到有效六类结果，已停止生成图表。")
                    return False
                if hasattr(self, "set_progress"):
                    self.set_progress(92, "六类区间划分完成，正在生成图表...")
                try:
                    self.root.update_idletasks()
                except Exception:
                    pass

            has_sample_values = bool(
                getattr(self, "sample_data_loaded", False)
                and getattr(self, "sample_data_values", None) is not None
            )
            mapped_records = [
                dict(record)
                for record in (
                    getattr(self, "_segmentation_sample_projection_records", []) or []
                )
                if isinstance(record, dict)
            ]
            mapping_ready = bool(
                str(getattr(self, "_sample_mapping_status", "") or "") == "valid"
                and str(getattr(self, "_current_mapping_signature", "") or "")
                and mapped_records
            )
            if segmentation_authoritative and has_sample_values and not mapping_ready:
                refresher = getattr(self, "_refresh_segmentation_sample_projection", None)
                mapped_records = (
                    refresher(refresh_view=False, silent=True)
                    if callable(refresher)
                    else None
                )
                mapping_ready = bool(mapped_records)
            if segmentation_authoritative and (not has_sample_values or not mapping_ready):
                if hasattr(self, "set_progress"):
                    self.set_progress(94, "正在生成程序 MRR 与过程域六态图...")
                rendered = self._render_process_domain_segmentation_view(save=save)
                return self._finish_segmentation_fast_plot(
                    rendered,
                    "过程域图表已生成",
                )
            if segmentation_authoritative and mapping_ready:
                if hasattr(self, "set_progress"):
                    self.set_progress(94, "正在生成过程域划分与实际负载映射图...")
                rendered = self._render_segmentation_sample_overlay_view(
                    mapped_records,
                    save=save,
                )
                if not rendered:
                    return self._finish_segmentation_fast_plot(
                        False,
                        "实际负载与区间映射图生成失败",
                    )
                return self._finish_segmentation_fast_plot(
                    rendered,
                    "过程域划分与实际负载映射图已生成",
                )

            model_ready = self.has_prediction_model_ready() if hasattr(self, "has_prediction_model_ready") else self.has_identified_kc_ke()
            self.figures = []
            if hasattr(self, "refresh_pit_button_state"):
                self.refresh_pit_button_state()
            current_sample_mode = str(getattr(self, "sample_data_mode", "") or "").strip()
            resolved_policy = str(interval_policy or "").strip()
            measurement_display_mode = None
            should_refresh_measurement_prediction = False
            imported_forward_lock = bool(
                hasattr(self, "_is_imported_profile_forward_lock_active")
                and self._is_imported_profile_forward_lock_active()
            )
            if resolved_policy not in {"use_active_profile", "reuse_current_template", "recompute_current", "fresh_or_empty"}:
                raise ValueError(f"Unsupported interval_policy: {interval_policy}")
            if imported_forward_lock and not segmentation_authoritative:
                resolved_policy = "use_active_profile"
            if (
                current_sample_mode == "experiment_measurement"
                and getattr(self, "manual_measurement_data", None)
            ):
                if imported_forward_lock:
                    measurement_display_mode = "forward"
                else:
                    measurement_display_mode = self._get_measurement_display_mode()
                should_refresh_measurement_prediction = (
                    not segmentation_authoritative
                    and (
                        True
                        if imported_forward_lock
                        else (
                            bool(refresh_prediction)
                            if refresh_prediction is not None
                            else bool(measurement_display_mode != "posterior")
                        )
                    )
                )
                self._debug_prediction_state_event(
                    "generate_plots_refresh_gate",
                    display_mode=measurement_display_mode,
                    interval_policy=resolved_policy,
                    refresh_prediction=bool(should_refresh_measurement_prediction),
                    interval_source=str(getattr(self, "_current_interval_source", "") or "none"),
                    live_display=measurement_display_mode,
                )
                if should_refresh_measurement_prediction:
                    if imported_forward_lock:
                        self._refresh_manual_measurement_prediction(
                            display_mode="forward",
                            allow_measurement_resolve=False,
                            allow_saved_sample_profile=False,
                        )
                    else:
                        self._refresh_manual_measurement_prediction(display_mode=measurement_display_mode)
                prediction_source = self._get_prediction_source()
                authoritative_profile_active = bool(
                    prediction_source in {"imported_profile", "runtime_identified_profile"}
                )
                can_reuse_current = self._can_reuse_current_interval_template(
                    prediction_source=prediction_source,
                    measurement=getattr(self, "manual_measurement_data", None),
                )
                should_rebuild_process_prediction = (
                    not segmentation_authoritative
                    and bool(measurement_display_mode != "posterior")
                    and (
                        resolved_policy in {"recompute_current", "fresh_or_empty"}
                        or (resolved_policy == "reuse_current_template" and not can_reuse_current)
                    )
                )
                if should_rebuild_process_prediction and hasattr(self, "_refresh_current_process_prediction_from_runtime"):
                    self._refresh_current_process_prediction_from_runtime(
                        allow_profile_fallback=authoritative_profile_active,
                        prefer_current_state=bool(resolved_policy == "reuse_current_template" and can_reuse_current),
                    )

            s_values = [d['s'] for d in self.data]
            P_values = [d['P'] for d in self.data]
            n_values = [d['N_str'] for d in self.data]
            line_numbers = [d.get('line_no_aligned', idx) for idx, d in enumerate(self.data)]
            line_numbers = [int(x) if x is not None else idx for idx, x in enumerate(line_numbers)]
            path_end_values = []
            use_path_end_values = True
            for row in self.data:
                try:
                    path_end_values.append(float(row.get("path_end")))
                except Exception:
                    use_path_end_values = False
                    break
            cumulative_s = np.asarray(path_end_values, dtype=float) if use_path_end_values and path_end_values else np.cumsum(s_values)

            if segmentation_authoritative:
                resolved_intervals = self._get_current_interval_records(allow_profile_fallback=False)
            else:
                resolved_intervals = self._resolve_pred_power_intervals(
                    P_values,
                    s_values,
                    cumulative_s,
                    n_values,
                    line_numbers,
                    model_ready=model_ready,
                    debug_line_range=debug_line_range,
                    interval_policy=resolved_policy,
                    materialize_reused_current_template=not bool(
                        current_sample_mode == "experiment_measurement"
                        and measurement_display_mode == "posterior"
                    ),
                )
            should_materialize_measurement_runtime_intervals = bool(
                not segmentation_authoritative
                and
                current_sample_mode == "experiment_measurement"
                and getattr(self, "manual_measurement_data", None)
                and resolved_intervals
                and measurement_display_mode != "posterior"
                and not imported_forward_lock
            )
            self._debug_prediction_state_event(
                "generate_plots_materialize_gate",
                display_mode=measurement_display_mode or self._get_measurement_display_mode(),
                interval_policy=resolved_policy,
                materialize_measurement_runtime_interval=bool(should_materialize_measurement_runtime_intervals),
                interval_source=str(getattr(self, "_current_interval_source", "") or "none"),
                live_display=measurement_display_mode or self._get_measurement_display_mode(),
            )
            if (
                should_materialize_measurement_runtime_intervals
            ):
                materialized_intervals = self._materialize_profile_pit_records(resolved_intervals)
                if materialized_intervals:
                    resolved_intervals = materialized_intervals

            # 需求6：最大区间数500限制，对原始区间按sample_count从小到大删减
            MAX_INTERVALS = 500
            if not segmentation_authoritative and len(resolved_intervals) > MAX_INTERVALS:
                original_count = len(resolved_intervals)
                # 按sample_count（SampleData点数）降序排序，保留点数最多的区间
                intervals_sorted = sorted(resolved_intervals, key=lambda x: x.get('sample_count', 0), reverse=True)
                # 保留前500个
                resolved_intervals = intervals_sorted[:MAX_INTERVALS]
                # 恢复原始顺序（优先按点级显示起点排序）
                resolved_intervals = sorted(
                    resolved_intervals,
                    key=lambda x: (
                        float(x.get('display_start_x')) if np.isfinite(x.get('display_start_x', float("nan"))) else float(x.get('start_line', 0)),
                        float(x.get('start_line', 0)),
                    )
                )
                self.set_status(f"区间数超过{MAX_INTERVALS}，已删减{original_count - MAX_INTERVALS}个点数较少的区间", 5000)

            used_compact_runtime_intervals = False
            if segmentation_authoritative:
                resolved_segments = self._get_current_segment_records(allow_profile_fallback=False)
                resolved_point_kc_map = dict(getattr(self, "current_interval_point_kc_map", {}) or {})
                current_source = "segmentation"
                profile_locked = bool(getattr(self, "_profile_intervals_locked", False))
            elif resolved_policy == "use_active_profile":
                if imported_forward_lock:
                    active_profile = getattr(self, "imported_kc_profile", None)
                    if not isinstance(active_profile, dict):
                        active_profile = getattr(self, "active_kc_profile", None)
                    resolved_intervals = self._extract_profile_interval_records(active_profile)
                    resolved_segments = self._extract_profile_segment_records(active_profile)
                    if not resolved_segments and self.data:
                        resolved_segments = self._build_profile_segment_records(interval_records=resolved_intervals)
                    resolved_point_kc_map = self._normalize_profile_point_kc_map(active_profile)
                    current_source = "imported_profile"
                    profile_locked = True
                elif (
                    bool(getattr(self, "_current_interval_ready", False))
                    and bool(getattr(self, "_profile_intervals_locked", False))
                    and self._can_reuse_current_interval_template(
                        prediction_source="imported_profile",
                        measurement=getattr(self, "manual_measurement_data", None),
                    )
                ):
                    # 导入 profile 后的预览必须直接复用 current state，不能再触发 fresh repartition。
                    resolved_segments = self._get_current_segment_records(allow_profile_fallback=False)
                    resolved_point_kc_map = dict(getattr(self, "current_interval_point_kc_map", {}) or {})
                    current_source = str(getattr(self, "_current_interval_source", "") or "imported_profile")
                else:
                    active_profile = self._get_saved_kc_profile_for_input()
                    resolved_segments = self._extract_profile_segment_records(active_profile)
                    if not resolved_segments and self.data:
                        resolved_segments = self._build_profile_segment_records(interval_records=resolved_intervals)
                    resolved_point_kc_map = self._normalize_profile_point_kc_map(active_profile) if isinstance(active_profile, dict) else {}
                    current_source = "imported_profile"
                profile_locked = True
            elif resolved_policy == "reuse_current_template":
                if self._can_reuse_current_interval_template(
                    prediction_source=self._get_prediction_source(),
                    measurement=getattr(self, "manual_measurement_data", None),
                ):
                    resolved_segments = self._get_current_segment_records(allow_profile_fallback=False)
                    if not resolved_segments and self.data:
                        resolved_segments = self._build_profile_segment_records(interval_records=resolved_intervals)
                    resolved_point_kc_map = dict(getattr(self, "current_interval_point_kc_map", {}) or {})
                    current_source = str(getattr(self, "_current_interval_source", "") or "current_template")
                    profile_locked = bool(self._get_prediction_source() == "imported_profile")
                else:
                    resolved_segments = self._build_profile_segment_records(interval_records=resolved_intervals) if self.data else []
                    resolved_point_kc_map = self._build_full_point_kc_map_from_current_state(
                        allow_profile_fallback=False,
                        prefer_current_state=False,
                    )
                    current_source = "fresh_or_empty"
                    profile_locked = False
            else:
                resolved_segments = self._build_profile_segment_records(interval_records=resolved_intervals) if self.data else []
                resolved_point_kc_map = self._build_full_point_kc_map_from_current_state(
                    allow_profile_fallback=False,
                    prefer_current_state=False,
                )
                current_source = "recompute_current" if resolved_policy == "recompute_current" else "fresh_or_empty"
                profile_locked = False
                if (
                    current_sample_mode == "experiment_measurement"
                    and resolved_segments
                    and not all(
                        isinstance(record, dict)
                        and record.get("sample_start_idx") is not None
                        and record.get("sample_end_idx") is not None
                        for record in (resolved_intervals or [])
                    )
                    and hasattr(self, "_build_compact_runtime_intervals_from_segments")
                ):
                    compact_intervals = self._build_compact_runtime_intervals_from_segments(resolved_segments)
                    if compact_intervals:
                        resolved_intervals = compact_intervals
                        used_compact_runtime_intervals = True

            if (
                not segmentation_authoritative
                and
                current_sample_mode == "experiment_measurement"
                and measurement_display_mode == "posterior"
                and resolved_intervals
                and hasattr(self, "_build_point_kc_map_from_interval_records")
            ):
                resolved_point_kc_map = self._build_point_kc_map_from_interval_records(
                    resolved_intervals,
                    base_point_kc_map=resolved_point_kc_map,
                )

            if not segmentation_authoritative and hasattr(self, "_debug_interval_state_event"):
                self._debug_interval_state_event(
                    "write_current_state",
                    write_source="generate_plots",
                    source=current_source,
                    interval_count=len(resolved_intervals),
                    segment_count=len(resolved_segments),
                )
            current_prediction_source = "imported_profile" if imported_forward_lock else self._get_prediction_source()
            if not segmentation_authoritative:
                self._set_current_interval_state(
                    interval_records=resolved_intervals,
                    segment_records=resolved_segments,
                    point_kc_map=resolved_point_kc_map,
                    source=current_source,
                    profile_locked=profile_locked,
                    context_signature=self._build_prediction_context_signature(
                        prediction_source=current_prediction_source,
                        measurement=getattr(self, "manual_measurement_data", None),
                    ),
                    prediction_source=current_prediction_source,
                    measurement_case_signature=self._get_current_measurement_case_signature(),
                )
            elif hasattr(self, "_debug_interval_state_event"):
                self._debug_interval_state_event(
                    "reuse_segmentation_state",
                    write_source="generate_plots",
                    interval_count=len(resolved_intervals),
                    segment_count=len(resolved_segments),
                )
            if (
                current_sample_mode == "experiment_measurement"
                and getattr(self, "manual_measurement_data", None)
                and measurement_display_mode == "posterior"
                and not imported_forward_lock
                and resolved_policy in {"recompute_current", "fresh_or_empty"}
                and hasattr(self, "_apply_current_interval_mode_kc_override_to_measurement")
            ):
                self._apply_current_interval_mode_kc_override_to_measurement()
            if used_compact_runtime_intervals and hasattr(self, "_apply_interval_kc_records_to_current_data"):
                self._apply_interval_kc_records_to_current_data(resolved_intervals)
                P_values = [d['P'] for d in self.data]
            if segmentation_authoritative and hasattr(
                self,
                "_refresh_authoritative_segmentation_interval_descriptors",
            ):
                self._refresh_authoritative_segmentation_interval_descriptors()
            current_intervals = self._get_current_interval_records(allow_profile_fallback=False)
            sample_background_intervals = current_intervals
            if segmentation_authoritative:
                try:
                    sample_background_intervals = self._get_authoritative_segmentation_sample_records()
                except Exception as projection_exc:
                    # None 是“投影无效”哨兵；不能用空区间降级，
                    # 否则全部采样点会被误当作过程域外补色。
                    sample_background_intervals = None
                    projection_reason = str(projection_exc)
                    latest_result = getattr(self, "_latest_segmentation_result", None)
                    latest_diagnostics = getattr(latest_result, "diagnostics", None)
                    if isinstance(latest_diagnostics, dict):
                        latest_diagnostics["sample_projection"] = {
                            "valid": False,
                            "reason": projection_reason,
                        }
                        latest_diagnostics["sample_visualization"] = {
                            "valid": False,
                            "reason": projection_reason,
                            "display_suppressed": True,
                        }
                    status_var = getattr(self, "sample_mapping_status_var", None)
                    if status_var is not None and hasattr(status_var, "set"):
                        status_var.set(
                            f"采样映射: 失败，已停止实际负载叠图（{projection_reason}）"
                        )
            if (
                current_sample_mode == "experiment_measurement"
                and getattr(self, "manual_measurement_data", None)
                and should_refresh_measurement_prediction
                and measurement_display_mode != "posterior"
            ):
                if imported_forward_lock:
                    self._refresh_manual_measurement_prediction(
                        display_mode="forward",
                        allow_measurement_resolve=False,
                        allow_saved_sample_profile=False,
                    )
                else:
                    self._refresh_manual_measurement_prediction(display_mode=measurement_display_mode)
            if hasattr(self, "refresh_pit_button_state"):
                self.refresh_pit_button_state()
            if hasattr(self, "_refresh_ideal_tree"):
                self._refresh_ideal_tree()
            
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
            sample_reconstructed_all = None
            sample_reconstructed_blocks = []
            sample_predicted_all = None
            sample_predicted_blocks = []
            sample_idle_power_all = None
            sample_sigma_idle = 0.0
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
                    sample_prediction_payload = None
                    sample_mode = str(getattr(self, "sample_data_mode", "") or "").strip()
                    if (
                        model_ready
                        and sample_mode == "experiment_measurement"
                        and source_idx == 0
                        and getattr(self, "manual_measurement_data", None)
                    ):
                        sample_prediction_payload = self.manual_measurement_data
                    elif model_ready and sample_mode == "sampledata":
                        sample_prediction_payload = self._build_sampledata_prediction_payload()

                    if sample_prediction_payload:
                        try:
                            sample_sigma_idle = float(sample_prediction_payload.get("sigma_idle", 0.0) or 0.0)
                        except Exception:
                            sample_sigma_idle = 0.0
                        idle_values = sample_prediction_payload.get("predicted_idle_power")
                        if idle_values is not None and len(idle_values) == len(sample_values_all):
                            sample_idle_power_all = np.asarray(idle_values, dtype=float)

                        predicted_values = sample_prediction_payload.get("predicted_load")
                        if predicted_values is not None and len(predicted_values) == len(sample_values_all):
                            sample_predicted_all = np.asarray(predicted_values, dtype=float)
                            predicted_mask = sample_context_mask & np.isfinite(sample_predicted_all)
                            sample_predicted_blocks = self.compute_contiguous_blocks(predicted_mask)

            has_sample_context = bool(sample_context_mask is not None and sample_context_mask.any())
            has_sample_valid = bool(sample_valid_mask is not None and sample_valid_mask.any())
            axis_mode = str(getattr(self, "process_axis_mode", tk.StringVar(value="时域+指令域")).get())
            use_path_display = axis_mode == "行程域+指令域"
            use_time_display = axis_mode == "时域+指令域" and bool(has_sample_context and sample_time_indices_all is not None)
            use_line_display = not use_time_display and not use_path_display
            sample_display_x_all = None
            if use_time_display:
                sample_display_x_all = sample_time_indices_all
            elif use_path_display and has_sample_context:
                sample_display_x_all = self._build_sample_path_positions(
                    sample_line_numbers_all,
                    getattr(self, "sample_data_point_indices", None)
                )
            else:
                sample_display_x_all = sample_x_positions_all
            if use_path_display and sample_display_x_all is not None:
                sample_display_x_all = np.asarray(sample_display_x_all, dtype=float)
                if not np.any(np.isfinite(sample_display_x_all)):
                    use_path_display = False
                    use_line_display = True
                    sample_display_x_all = sample_x_positions_all
            if sample_display_x_all is not None:
                sample_display_x_all = np.asarray(sample_display_x_all, dtype=float)

            tool_mean_val = None
            tool_count = 0
            tool_ideal_val_preview = None
            if display_mode == "tool" and program_name and tool_id:
                tool_mean_val, tool_count, _ = _get_tool_stats(tool_id)
                if tool_mean_val is not None and tool_count > 0:
                    tool_ideal_val_preview = tool_mean_val * adjustment_ratio

            # 注意：不再根据SampleData行号范围过滤预测负载
            # 两者使用独立的行号体系，只是叠加显示在同一张图上
            
            interval_meta = []
            # 使用原始区间数据，包含精确的start_idx和end_idx用于绘制精确边界
            for idx, interval in enumerate(current_intervals):
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
                # 保存精确的idx用于计算x坐标边界
                interval_meta.append((
                    start_line,
                    end_line,
                    p_mean,
                    idx,
                    start_idx_iv,
                    end_idx_iv,
                    interval.get("display_start_x"),
                    interval.get("display_end_x"),
                    interval.get("display_start_t"),
                    interval.get("display_end_t"),
                ))

            # 使用全局定义的科技配色
            steady_fill_color = "#1E88E5"
            show_base_prediction = False
            show_measured_curve = bool(getattr(self, "show_measured_curve_var", tk.BooleanVar(value=True)).get())
            show_reconstructed_curve = bool(getattr(self, "show_reconstructed_curve_var", tk.BooleanVar(value=True)).get())
            prediction_mode = self.get_effective_prediction_mode() if hasattr(self, "get_effective_prediction_mode") else "direct_prediction"
            prediction_label = self.get_prediction_curve_label(prediction_mode) if hasattr(self, "get_prediction_curve_label") else "预测负载"
            sample_prediction_curve = sample_reconstructed_all if prediction_mode == "posterior" else sample_predicted_all
            sample_prediction_blocks = sample_reconstructed_blocks if prediction_mode == "posterior" else sample_predicted_blocks
            preview_plot_max_points = None
            if not save:
                try:
                    preview_plot_max_points = int(getattr(self, "preview_plot_max_points", 0) or 0)
                except Exception:
                    preview_plot_max_points = 0
                if preview_plot_max_points <= 0:
                    preview_plot_max_points = None
            self._current_plot_max_points = preview_plot_max_points

            def _build_sample_background_blocks():
                if (
                    sample_display_x_all is None
                    or sample_context_mask is None
                    or sample_values_all is None
                    or sample_background_intervals is None
                ):
                    return None
                x_values = np.asarray(sample_display_x_all, dtype=float)
                actual_values = np.asarray(sample_values_all, dtype=float)
                if len(x_values) != len(actual_values):
                    return None
                fill_values = np.maximum(actual_values, 0.0)
                context_mask = np.asarray(sample_context_mask, dtype=bool) & np.isfinite(x_values)
                draw_mask = context_mask & np.isfinite(actual_values)
                if not np.any(draw_mask):
                    return None

                background_payload = self.build_segmentation_sample_background_masks(
                    actual_values,
                    None,
                    sample_background_intervals,
                    valid_mask=context_mask,
                )
                state_masks = background_payload["state_masks"]
                return {
                    "x_values": x_values,
                    "fill_values": fill_values,
                    "state_blocks": {
                        segment_type: self.compute_contiguous_blocks(mask)
                        for segment_type, mask in state_masks.items()
                    },
                }

            def _draw_sample_background(ax):
                payload = _build_sample_background_blocks()
                if not payload:
                    return
                artists = []
                state_blocks = payload.get("state_blocks", {})
                for segment_type in ("idle", "entry", "steady", "transition", "nonsteady", "exit"):
                    style = self.get_segmentation_state_style(segment_type)
                    state_artists = self._draw_curve_background_blocks(
                        ax,
                        payload.get("x_values"),
                        payload.get("fill_values"),
                        state_blocks.get(segment_type),
                        color=style["color"],
                        alpha=0.30,
                        label=f"{style['label']} [{style['state_code']}]",
                        zorder=1,
                    )
                    artists.extend(state_artists or [])
                self._interval_background_artists.extend(artists)
            
            # 预测负载 + 实测负载图
            # 公共页面只保留单图叠加模式；兼容旧配置中的 stacked 值但不再采用。
            plot_mode = "overlay"
            if self.sample_plot_mode.get() != plot_mode:
                self.sample_plot_mode.set(plot_mode)
            if plot_mode == "stacked":
                fig2, (axes_act, axes_pred) = plt.subplots(2, 1, sharex=True, figsize=(16, 9), dpi=100)
                overlay_secondary = False
            else:
                fig2, ax2 = plt.subplots(figsize=(16, 9), dpi=100)
                axes_pred = ax2
                if show_base_prediction:
                    axes_act = ax2.twinx()
                    overlay_secondary = True
                else:
                    axes_act = ax2
                    overlay_secondary = False
            fig2.patch.set_facecolor(PLOT_FIG_BG)
            self.apply_plot_style(axes_pred, grid=False)
            aux_axes = []
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
            if use_path_display:
                process_path_start_bounds, process_path_end_bounds = self._get_process_path_bounds()
                finite_path_mask = np.isfinite(process_path_start_bounds) & np.isfinite(process_path_end_bounds)
                if finite_path_mask.any():
                    process_display_start_bounds = process_path_start_bounds
                    process_display_end_bounds = process_path_end_bounds
                else:
                    use_path_display = False
                    use_line_display = True
                    sample_display_x_all = sample_x_positions_all
            elif use_time_display:
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
                    use_line_display = True
                    sample_display_x_all = sample_x_positions_all
            self._current_sample_display_x = None if sample_display_x_all is None else np.asarray(sample_display_x_all, dtype=float)
            self._current_sample_context_mask = None if sample_context_mask is None else np.asarray(sample_context_mask, dtype=bool)
            self._current_sample_valid_mask = None if sample_valid_mask is None else np.asarray(sample_valid_mask, dtype=bool)
            self._current_sample_use_time_display = bool(use_time_display)

            def _resolve_interval_plot_bounds(start_line, end_line, start_idx_iv, end_idx_iv,
                                              display_start_x_iv, display_end_x_iv,
                                              display_start_t_iv, display_end_t_iv):
                if (
                    use_time_display
                    and np.isfinite(display_start_t_iv)
                    and np.isfinite(display_end_t_iv)
                    and float(display_end_t_iv) > float(display_start_t_iv)
                ):
                    return float(display_start_t_iv), float(display_end_t_iv)
                if (
                    (not use_time_display)
                    and np.isfinite(display_start_x_iv)
                    and np.isfinite(display_end_x_iv)
                    and float(display_end_x_iv) > float(display_start_x_iv)
                ):
                    return float(display_start_x_iv), float(display_end_x_iv)
                if start_idx_iv is not None and end_idx_iv is not None and len(process_display_start_bounds) > 0:
                    return float(process_display_start_bounds[start_idx_iv]), float(process_display_end_bounds[end_idx_iv])
                return float(start_line), float(end_line + 1)

            def _plot_process_step_blocks(
                ax,
                blocks,
                start_bounds,
                end_bounds,
                color,
                linewidth,
                alpha,
                label=None,
                zorder=5,
                linestyle="-",
            ):
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
                                    "linestyle": linestyle,
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
                                "linestyle": linestyle,
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
                            "linestyle": linestyle,
                        }
                        if first and label:
                            plot_kwargs["label"] = label
                        ax.plot(segment_x, segment_y, **plot_kwargs)
                        first = False
            
            if model_ready:
                _draw_sample_background(axes_pred)
                if show_base_prediction:
                    pred_style = STYLE_PREDICTED.copy()
                    pred_style["label"] = "基准预测负载"
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
                            zorder=5,
                            linestyle=pred_style.get("linestyle", "--"),
                        )
                    axes_pred.set_ylabel('基准预测负载 P_base (W)', fontsize=PLOT_FONT_BASE,
                                         fontweight='bold', color=PLOT_TEXT_COLOR)
                axes_pred.tick_params(labelsize=PLOT_FONT_BASE, colors=PLOT_TEXT_COLOR)
            else:
                axes_pred.tick_params(labelsize=PLOT_FONT_BASE, colors=PLOT_TEXT_COLOR)
            
            data_source_name = self.get_sample_data_source_name()
            if has_sample_context:
                measured_style = STYLE_MEASURED.copy()
                if getattr(self, "sample_data_mode", "") == "experiment_measurement" and int(self.sample_data_source.get()) == 0:
                    measured_style["label"] = "实际负载"
                else:
                    measured_style["label"] = f"实际负载({data_source_name})"
                actual_values_for_scale = []
                prediction_values_for_scale = []
                overlay_values_for_scale = []
                prediction_axis = axes_pred if plot_mode == "stacked" else axes_act
                prediction_curve_drawn = False
                if show_measured_curve and sample_invalid_blocks:
                    self.plot_series_by_blocks(
                        axes_act,
                        sample_display_x_all,
                        sample_values_all,
                        sample_invalid_blocks,
                        max_points=preview_plot_max_points,
                        color="#B0BEC5",
                        linewidth=1.0,
                        alpha=0.75,
                        label="非当前有效段",
                        zorder=5
                    )
                if show_measured_curve and has_sample_valid:
                    self.plot_series_by_blocks(
                        axes_act,
                        sample_display_x_all,
                        sample_values_all,
                        sample_valid_blocks,
                        max_points=preview_plot_max_points,
                        zorder=6,
                        alpha=0.95,
                        **measured_style
                    )
                    actual_values_for_scale.append(sample_values_all[sample_valid_mask])
                    overlay_values_for_scale.append(sample_values_all[sample_valid_mask])
                elif show_measured_curve:
                    self.plot_series_by_blocks(
                        axes_act,
                        sample_display_x_all,
                        sample_values_all,
                        sample_context_blocks,
                        max_points=preview_plot_max_points,
                        color="#B0BEC5",
                        linewidth=1.0,
                        alpha=0.75,
                        label=f"实际负载({data_source_name})",
                        zorder=5
                    )
                    actual_values_for_scale.append(sample_values_all[sample_context_mask])
                    overlay_values_for_scale.append(sample_values_all[sample_context_mask])
                if (
                    show_reconstructed_curve
                    and model_ready
                    and sample_prediction_curve is not None
                    and sample_prediction_blocks
                ):
                    self.plot_series_by_blocks(
                        prediction_axis,
                        sample_display_x_all,
                        sample_prediction_curve,
                        sample_prediction_blocks,
                        max_points=preview_plot_max_points,
                        color=self.get_segmentation_predicted_line_color(),
                        linewidth=1.5,
                        linestyle="--",
                        alpha=0.95,
                        label=prediction_label,
                        zorder=8
                    )
                    prediction_curve_drawn = True
                    prediction_mask = sample_context_mask & np.isfinite(sample_prediction_curve)
                    if np.any(prediction_mask):
                        prediction_values_for_scale.append(sample_prediction_curve[prediction_mask])
                        if plot_mode != "stacked":
                            overlay_values_for_scale.append(sample_prediction_curve[prediction_mask])
                if actual_values_for_scale:
                    actual_plot_values = np.concatenate([
                        np.asarray(values, dtype=float) for values in actual_values_for_scale if len(values) > 0
                    ])
                else:
                    actual_plot_values = np.asarray(sample_values_all[sample_context_mask], dtype=float)
                if prediction_values_for_scale:
                    prediction_plot_values = np.concatenate([
                        np.asarray(values, dtype=float) for values in prediction_values_for_scale if len(values) > 0
                    ])
                else:
                    prediction_plot_values = np.asarray([], dtype=float)
                if overlay_values_for_scale:
                    overlay_plot_values = np.concatenate([
                        np.asarray(values, dtype=float) for values in overlay_values_for_scale if len(values) > 0
                    ])
                else:
                    overlay_plot_values = np.asarray(actual_plot_values, dtype=float)
                if (
                    show_measured_curve
                    and actual_plot_values is not None
                    and len(actual_plot_values) > 0
                    and np.any(np.isfinite(actual_plot_values))
                ):
                    avg_load = float(np.nanmean(actual_plot_values))
                    avg_target_x = sample_display_x_all[sample_valid_mask] if has_sample_valid else sample_display_x_all[sample_context_mask]
                    if avg_target_x is not None and len(avg_target_x) > 0:
                        x_min_avg = float(np.nanmin(avg_target_x))
                        x_max_avg = float(np.nanmax(avg_target_x))
                        axes_act.plot(
                            [x_min_avg, x_max_avg],
                            [avg_load, avg_load],
                            color="#90A4AE",
                            linewidth=0.9,
                            linestyle="--",
                            alpha=0.28,
                            label="_nolegend_",
                            zorder=7,
                        )
                if plot_mode == "stacked":
                    self.apply_plot_style(axes_act, grid=False)
                    self.apply_tool_background(axes_act, program_name, None if display_mode == "program" else tool_id)
                    y_max_act = np.nanmax(actual_plot_values) if actual_plot_values is not None and len(actual_plot_values) > 0 else 0
                    
                    if model_ready and axes_act is not axes_pred:
                        _draw_sample_background(axes_act)
                    axes_act.set_ylabel(measured_style["label"], fontsize=PLOT_FONT_BASE,
                                       fontweight='bold', color=PLOT_TEXT_COLOR)
                    axes_pred.set_ylabel(prediction_label, fontsize=PLOT_FONT_BASE,
                                         fontweight='bold', color=PLOT_TEXT_COLOR)
                    axes_act.tick_params(labelsize=PLOT_FONT_BASE, colors=PLOT_TEXT_COLOR)
                    axes_pred.tick_params(labelsize=PLOT_FONT_BASE, colors=PLOT_TEXT_COLOR)
                else:
                    # 叠加模式：也在实测轴上绘制背景条（使用实测数据的Y轴范围）
                    if overlay_plot_values is not None and len(overlay_plot_values) > 0:
                        y_max = np.nanmax(overlay_plot_values) * 1.1
                        
                        if model_ready and axes_act is not axes_pred:
                            _draw_sample_background(axes_act)
                    axes_act.set_ylabel("功率 (W)", fontsize=PLOT_FONT_BASE,
                                       fontweight='bold', color=PLOT_TEXT_COLOR)
                aux_axes = self.plot_optional_measurement_overlays(
                    axes_act,
                    sample_display_x_all,
                    sample_context_mask,
                    sample_valid_mask
                )
                if plot_mode == "stacked":
                    if not show_measured_curve and not aux_axes:
                        axes_act.text(
                            0.5, 0.5, "当前已隐藏实际负载",
                            ha='center', va='center', transform=axes_act.transAxes,
                            fontsize=PLOT_FONT_BASE, color='#666666'
                        )
                    if not prediction_curve_drawn:
                        if not model_ready:
                            prediction_msg = "未辨识模型参数"
                        elif not show_reconstructed_curve:
                            prediction_msg = f"当前已隐藏{prediction_label}"
                        else:
                            prediction_msg = f"当前无可显示的{prediction_label}"
                        axes_pred.text(
                            0.5, 0.5, prediction_msg,
                            ha='center', va='center', transform=axes_pred.transAxes,
                            fontsize=PLOT_FONT_BASE, color='#666666'
                        )
                elif not show_measured_curve and not show_reconstructed_curve and not aux_axes:
                    axes_act.text(0.5, 0.5, "当前未启用任何曲线显示", ha='center', va='center',
                                  transform=axes_act.transAxes, fontsize=PLOT_FONT_BASE, color='#666666')

                target_curve = np.asarray(getattr(self, "target_load_curve", []) or [], dtype=float)
                if target_curve.size == len(sample_values_all) and sample_display_x_all is not None:
                    target_mask = sample_context_mask & np.isfinite(target_curve)
                    target_blocks = self.compute_contiguous_blocks(target_mask)
                    if target_blocks:
                        self.plot_series_by_blocks(
                            axes_act,
                            sample_display_x_all,
                            target_curve,
                            target_blocks,
                            max_points=preview_plot_max_points,
                            color="#D32F2F",
                            linewidth=1.8,
                            alpha=0.95,
                            label="目标值",
                            zorder=9
                        )
            else:
                if plot_mode == "stacked":
                    self.apply_plot_style(axes_act, grid=False)
                    self.apply_plot_style(axes_pred, grid=False)
                    axes_act.text(0.5, 0.5, "未加载实测数据", ha='center', va='center',
                                  transform=axes_act.transAxes, fontsize=PLOT_FONT_BASE, color='#666666')
                    axes_pred.text(0.5, 0.5, "未加载模型曲线", ha='center', va='center',
                                   transform=axes_pred.transAxes, fontsize=PLOT_FONT_BASE, color='#666666')
                    axes_act.grid(False)
                    axes_pred.grid(False)
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
                context_x = context_x[np.isfinite(context_x)]
                if len(context_x) > 0:
                    x_candidates.extend([float(np.min(context_x)), float(np.max(context_x))])
            if x_candidates:
                x_min = min(x_candidates)
                x_max = max(x_candidates)
                x_range = x_max - x_min if x_max > x_min else 1
                axes_pred.set_xlim(x_min - x_range * 0.02, x_max + x_range * 0.02)
                if plot_mode == "stacked" or overlay_secondary:
                    axes_act.set_xlim(x_min - x_range * 0.02, x_max + x_range * 0.02)

            linked_line_axes = []
            if plot_mode == "stacked" or overlay_secondary:
                linked_line_axes.append(axes_act)
            axes_pred._time_line_linked_axes = linked_line_axes

            if use_time_display:
                if plot_mode == "stacked":
                    axes_pred.set_xlabel('时间 (ms)', fontsize=PLOT_FONT_BASE, fontweight='bold', color=PLOT_TEXT_COLOR)
                    axes_act.tick_params(labelbottom=False)
                else:
                    axes_pred.set_xlabel('时间 (ms)', fontsize=PLOT_FONT_BASE, fontweight='bold', color=PLOT_TEXT_COLOR)
                self.apply_line_axis_on_time(axes_pred, sample_context_mask)
            elif use_path_display:
                if plot_mode == "stacked":
                    axes_pred.set_xlabel('累计行程 (mm)', fontsize=PLOT_FONT_BASE, fontweight='bold', color=PLOT_TEXT_COLOR)
                    axes_act.tick_params(labelbottom=False)
                else:
                    axes_pred.set_xlabel('累计行程 (mm)', fontsize=PLOT_FONT_BASE, fontweight='bold', color=PLOT_TEXT_COLOR)
                self.apply_line_axis_on_path(axes_pred, sample_display_x_all, sample_context_mask)
            else:
                if plot_mode == "stacked":
                    axes_pred.set_xlabel('对齐行号', fontsize=PLOT_FONT_BASE, fontweight='bold', color=PLOT_TEXT_COLOR)
                    axes_act.tick_params(labelbottom=False)
                    self.apply_line_axis(axes_pred, axis_line_numbers)
                else:
                    axes_pred.set_xlabel('对齐行号', fontsize=PLOT_FONT_BASE, fontweight='bold', color=PLOT_TEXT_COLOR)
                    self.apply_line_axis(axes_pred, axis_line_numbers)
            
            # 标题样式 - 工业简洁风格（不显示标题，通过图例区分）
            # axes_pred.set_title('', fontsize=PLOT_FONT_BASE, fontweight='bold', color=PLOT_TEXT_COLOR, pad=8)
            
            # 图例样式 - 工业科技感
            legend_host = axes_pred
            legend_base_axes = [axes_pred]
            if plot_mode == "stacked":
                legend_host = axes_act
                legend_base_axes = [axes_act, axes_pred]
            elif overlay_secondary:
                legend_base_axes = [axes_pred, axes_act]
            self._apply_optional_overlay_legend(
                legend_host,
                legend_base_axes,
                aux_axes,
                loc='upper left',
                fontsize=PLOT_FONT_BASE - 1,
                framealpha=0.95,
                shadow=False,
                fancybox=False,
                edgecolor=PLOT_SPINE_COLOR,
                borderpad=0.6,
                labelspacing=0.4,
                facecolor='white',
                linewidth=0.6,
            )
            
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
                if tool_mean_val is not None and tool_count > 0:
                    self.sample_avg_var.set(f"{tool_mean_val:.3f}")
                    if tool_ideal_val_preview is not None:
                        self.sample_ideal_var.set(f"{tool_ideal_val_preview:.3f}")
                    else:
                        self.sample_ideal_var.set("-")
                else:
                    self.sample_avg_var.set("-")
                    self.sample_ideal_var.set("-")
            
            fig2.subplots_adjust(
                left=0.06,
                right=self._resolve_optional_overlay_right_margin(len(aux_axes)),
                top=0.88 if use_time_display else 0.93,
                bottom=0.13 if use_time_display else 0.10,
                hspace=0.15
            )
            if has_sample_context:
                self._register_optional_overlay_context(
                    fig2,
                    parent_ax=axes_act,
                    x_values=sample_display_x_all,
                    context_mask=sample_context_mask,
                    valid_mask=sample_valid_mask,
                    legend_host=legend_host,
                    legend_base_axes=legend_base_axes,
                    aux_axes=aux_axes,
                    legend_style={
                        "loc": "upper left",
                        "fontsize": PLOT_FONT_BASE - 1,
                        "framealpha": 0.95,
                        "shadow": False,
                        "fancybox": False,
                        "edgecolor": PLOT_SPINE_COLOR,
                        "borderpad": 0.6,
                        "labelspacing": 0.4,
                        "facecolor": "white",
                        "linewidth": 0.6,
                    },
                    subplot_adjust={
                        "left": 0.06,
                        "top": 0.88 if use_time_display else 0.93,
                        "bottom": 0.13 if use_time_display else 0.10,
                        "hspace": 0.15,
                    },
                )
            self.figures.append(fig2)
            
            if model_ready:
                self.figure_names = ["负载图"]
            else:
                self.figure_names = ["负载图"]
            default_index = 0
            
            if save:
                self.save_all_plots(silent=True)
            self.show_current_figure(default_index)
            
            # 六态结果覆盖全行程，数量显示始终使用权威区间总数。
            if hasattr(self, 'interval_count_var'):
                interval_count = len(current_intervals)
                self.interval_count_var.set(str(interval_count))
            if hasattr(self, "refresh_prediction_metrics_summary"):
                try:
                    self.root.after_idle(self.refresh_prediction_metrics_summary)
                except Exception:
                    self.refresh_prediction_metrics_summary()
            if hasattr(self, "invalidate_pit_view"):
                self.invalidate_pit_view(refresh_if_visible=True)
            
            total_charts = len(self.figures)
            self.status_var_data.set(f"图表已生成! 共{total_charts}张图表")
            if not save and not silent:
                messagebox.showinfo("完成", f"{total_charts}张图表已成功生成! 可继续保存结果(.rg)")
            if imported_forward_lock:
                active_profile = getattr(self, "imported_kc_profile", None)
                if not isinstance(active_profile, dict):
                    active_profile = getattr(self, "active_kc_profile", None)
                if hasattr(self, "_profile_has_saved_payload") and self._profile_has_saved_payload(active_profile):
                    self.prediction_source = "imported_profile"
                    self.profile_origin = "imported_profile"
            
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

        # 只替换画布；右上角的持久显示开关不得随重绘销毁。
        canvas_frame = getattr(self, "data_plot_canvas_frame", self.data_figure_frame)
        for widget in canvas_frame.winfo_children():
            widget.destroy()

        fig = self.figures[index]
        self._current_preview_fig = fig  # 供缩放等交互使用

        # 默认取主轴（缩放时会同步作用到同一Figure的其它轴）
        self.ax_data = fig.axes[0] if getattr(fig, 'axes', None) else None

        self.canvas_data = FigureCanvasTkAgg(fig, master=canvas_frame)
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

        # 新画布建立后只安排一次防抖尺寸同步；Configure 事件若紧随其后会
        # 覆盖同一个定时任务，不再出现“先画一次、再调整后重画一次”。
        self._on_preview_canvas_configure(None)
        self.root.after(10, lambda: self._focus_preview_canvas(canvas_widget))

    def _on_preview_canvas_configure(self, event):
        """预览区大小变化时触发图表重排（防抖避免拖拽卡顿）"""
        if bool(getattr(self, "_startup_layout_pending", False)):
            return
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
        self.show_current_figure(0)

    def save_all_plots(self, silent=False):
        """保存所有图表到项目输出目录。"""
        if not self.figures:
            if not silent:
                messagebox.showwarning("无图表", "没有可保存的图表，请先生成图表")
            return False
        
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            save_dir = str(OUTPUT_DIR)
            
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
            interval_records = self._get_steady_interval_records()
            if interval_records:
                intervals_txt_path = os.path.join(save_dir, "P_pred_steady_intervals.txt")
                try:
                    adjustment_ratio = float(self.adjustment_ratio.get())
                except Exception:
                    adjustment_ratio = 1.0
                with open(intervals_txt_path, 'w', encoding='utf-8') as f:
                    f.write("# 预测功率稳态区间划分结果\n")
                    f.write(f"# 优化倍率 R: {adjustment_ratio:g}\n")
                    f.write("# 最小采样点数: 已取消限制\n")
                    f.write(f"# 总区间数: {len(interval_records)}\n")
                    f.write("# 区间范围使用 SampleData 实际点级边界，格式为 行号.点号-行号.点号\n")
                    f.write("#" + "="*80 + "\n")
                    f.write("# 区间\t区间范围\t采样点数\tP_pred(W)\tP_pref(W)\n")
                    for i, interval in enumerate(interval_records, 1):
                        p_pred = float(interval.get('p_pred', 0.0) or 0.0)
                        p_pref = p_pred * adjustment_ratio
                        sample_count = interval.get('sample_count', interval['end_idx'] - interval['start_idx'] + 1)
                        interval_range = self._format_interval_point_range(interval)
                        f.write(f"{i}\t{interval_range}\t{sample_count}\t{p_pred:.6f}\t{p_pref:.6f}\n")

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

        blocks = self._get_sample_x_range_blocks(start_x, end_x, program_no=program_no)
        counts = [end - start + 1 for start, end in blocks]
        count = max(counts) if counts else 0
        self.sample_data_valid_mask = None
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
        min_sample_count = 1  # 已取消最小采样点限制
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
            try:
                start_s = float(self.data[i].get("path_start"))
            except Exception:
                start_s = cumulative_s[i] - s_values[i] if i > 0 else 0
            current_line = line_numbers[i] if line_numbers is not None else i
            interval_start_x = None
            interval_end_x = None
            measurement_summary = None
            steady_pass = True
            
            # 查找P_pred稳定的连续区域
            j = i + 1
            while j < len(quantized_P):
                # 与区间首值或前一个点只要有变化就分段
                if is_diff(quantized_P[j], current_p) or is_diff(quantized_P[j], quantized_P[j-1]):
                    break
                j += 1
            
            end_idx = j - 1
            try:
                end_s = float(self.data[end_idx].get("path_end"))
            except Exception:
                end_s = cumulative_s[end_idx]
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
            interval_rows = self.data[start_idx:end_idx + 1] if self.data else []
            interval_mrr_values = []
            interval_idle_values = []
            interval_pred_values = []
            for row in interval_rows:
                try:
                    interval_mrr_values.append(float(row.get("MRR", 0.0)))
                except Exception:
                    continue
                try:
                    interval_idle_values.append(float(row.get("P_idle", 0.0)))
                except Exception:
                    interval_idle_values.append(float("nan"))
                try:
                    interval_pred_values.append(float(row.get("P", 0.0)))
                except Exception:
                    interval_pred_values.append(float("nan"))
            is_idle_interval = True
            if interval_mrr_values:
                interval_mrr_arr = np.asarray(interval_mrr_values, dtype=float)
                finite_interval_mrr = interval_mrr_arr[np.isfinite(interval_mrr_arr)]
                if finite_interval_mrr.size > 0 and np.any(finite_interval_mrr > 1e-12):
                    interval_pred_arr = np.asarray(interval_pred_values, dtype=float)
                    interval_idle_arr = np.asarray(interval_idle_values, dtype=float)
                    finite_idle_pred = (
                        np.isfinite(interval_pred_arr)
                        & np.isfinite(interval_idle_arr)
                    )
                    if np.any(finite_idle_pred):
                        is_idle_interval = bool(np.all(interval_pred_arr[finite_idle_pred] <= interval_idle_arr[finite_idle_pred] + 1e-9))
                    else:
                        is_idle_interval = False
            interval_pass = sample_count >= min_sample_count
            if (
                interval_pass
                and interval_start_x is not None
                and interval_end_x is not None
                and getattr(self, "sample_data_mode", "") == "experiment_measurement"
                and getattr(self, "manual_measurement_data", None)
            ):
                measurement_summary = self.summarize_measurement_interval(
                    interval_start_x,
                    interval_end_x,
                    program_no=current_program_no
                )
                if measurement_summary:
                    sample_count = int(measurement_summary.get("sample_count", sample_count))
                    steady_pass = bool(measurement_summary.get("steady_pass", False))
                    if not bool(is_idle_interval):
                        interval_pass = interval_pass and steady_pass

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
                'is_idle_interval': bool(is_idle_interval),
                'passed_min_count': sample_count >= min_sample_count,
                'passed_signal_gate': bool(steady_pass),
            }
            if measurement_summary:
                segment_info['actual_load_std'] = float(measurement_summary.get("actual_load_std", 0.0))
                segment_info['actual_load_diff_std'] = float(measurement_summary.get("actual_load_diff_std", 0.0))
                segment_info['valid_kc_count'] = int(measurement_summary.get("valid_kc_count", 0))
            all_segments.append(segment_info)

            # 如果采样点数大于等于最小样本点，则保存该区间
            if interval_pass:
                interval_meta = self.summarize_process_interval(start_idx, end_idx)
                kc_hat = float("nan")
                sigma_kc = float("nan")
                kc_ucb = float("nan")
                valid_kc_count = 0
                gated_out_count = 0
                p_meas = float("nan")
                actual_load_std = 0.0
                actual_load_diff_std = 0.0
                sigma_idle = 0.0
                delta_mrr = 0.0
                if measurement_summary:
                    kc_hat = float(measurement_summary.get("kc_hat", float("nan")))
                    sigma_kc = float(measurement_summary.get("sigma_kc", float("nan")))
                    valid_kc_count = int(measurement_summary.get("valid_kc_count", 0))
                    gated_out_count = int(measurement_summary.get("gated_out_count", 0))
                    p_meas = float(measurement_summary.get("p_meas", float("nan")))
                    actual_load_std = float(measurement_summary.get("actual_load_std", 0.0))
                    actual_load_diff_std = float(measurement_summary.get("actual_load_diff_std", 0.0))
                    sigma_idle = float(measurement_summary.get("sigma_idle", 0.0))
                    delta_mrr = float(measurement_summary.get("delta_mrr", 0.0))
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
                    p_meas=float(p_meas) if np.isfinite(p_meas) else 0.0,
                    K_c_hat=kc_hat,
                    K_c_UCB=kc_ucb,
                    sigma_Kc=sigma_kc,
                    sample_count=int(sample_count),
                    valid_kc_count=int(valid_kc_count),
                    gated_out_count=int(gated_out_count),
                    actual_load_std=float(actual_load_std),
                    actual_load_diff_std=float(actual_load_diff_std),
                    sigma_idle=float(sigma_idle),
                    delta_mrr=float(delta_mrr),
                    start_n=n_values[start_idx],
                    end_n=n_values[end_idx],
                    tool_diameter=pit_metadata["tool_diameter"],
                    tool_radius=pit_metadata["tool_radius"],
                    tool_material=pit_metadata["tool_material"],
                    blank_material=pit_metadata["blank_material"],
                )
                pit_record = asdict(pit_entry)
                pit_record["segment_type"] = "steady"
                pit_record["steady_subtype"] = "idle" if is_idle_interval else "cutting"
                pit_record["is_idle_interval"] = bool(is_idle_interval)
                pit_record["steady_pass"] = bool(True if is_idle_interval else steady_pass)
                if is_idle_interval:
                    pit_record["kc_source"] = "idle"
                elif measurement_summary and valid_kc_count > 0:
                    pit_record["kc_source"] = str(measurement_summary.get("kc_source") or "measurement_mode")
                intervals.append(pit_record)
            
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
        return self.finalize_interval_kc(intervals)

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
        self.generate_plots(debug_line_range=(start_line, end_line), interval_policy="recompute_current")
        
        # 打印最终显示的区间
        interval_records = self._get_current_interval_records(allow_profile_fallback=False)
        print(f"\n[DEBUG] 最终显示的稳态区间 (current state):")
        for idx, interval in enumerate(interval_records):
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
            self.figure_label.config(text="")
            return
        
        # 更新标签
        if self.current_figure_index < len(self.figure_names):
            self.figure_label.config(text=f"{self.figure_names[self.current_figure_index]}")
        else:
            self.figure_label.config(text=f"负载图")
