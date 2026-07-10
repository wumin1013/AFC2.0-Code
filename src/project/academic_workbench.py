from __future__ import annotations

import concurrent.futures
import threading

from .academic_analysis import (
    F_COLUMN_ALIASES,
    P_COLUMN_ALIASES,
    PRED_COLUMN_ALIASES,
    build_multi_run_analysis,
    build_run_label,
    compute_error_metrics,
    export_table_to_csv,
    get_relative_error_floor,
    infer_matching_column,
    normalize_role,
    prepare_run_frame,
    role_sort_key,
)
from .shared import *


class AcademicWorkbenchMixin:
    def init_academic_workbench_state(self):
        self.prediction_mode_var = tk.StringVar(value="direct_prediction")
        self.prediction_metrics_var = tk.StringVar(value="预测摘要: 尚未生成")
        self.steady_metric_trim_ratio = 0.10
        self.steady_metric_trim_min_points = 1

        self.pit_scope_var = tk.StringVar(value="all")
        self.pit_view_mode_var = tk.StringVar(value="plot")
        self.pit_axis_mode_var = tk.StringVar(value="时域+指令域")
        self.pit_field_var = tk.StringVar(value="")
        self.pit_status_var = tk.StringVar(value="PIT 预览待生成")
        self._pit_preview_fig = None
        self._pit_preview_canvas = None
        self._pit_preview_axes = None

        self.analysis_strategy_var = tk.StringVar(value="折中策略")
        self.analysis_chart_var = tk.StringVar(value="负载图")
        self.analysis_metric_var = tk.StringVar(value="avg_actual_load")
        self.analysis_run_var = tk.StringVar(value="")
        self.analysis_interval_var = tk.StringVar(value="")
        self.analysis_status_var = tk.StringVar(value="请先在工艺信息页面生成稳态区间，再导入首次加工与优化 run")
        self.analysis_summary_var = tk.StringVar(value="暂无分析摘要")
        self.analysis_interval_source_var = tk.StringVar(value="工艺页稳态区间")
        self.analysis_runs = []
        self.analysis_result = None
        self.analysis_current_table = pd.DataFrame()
        self.analysis_current_summary = ""
        self._analysis_plot_fig = None
        self._analysis_plot_canvas = None
        self._analysis_plot_ax = None
        self.analysis_import_in_progress = False
        self.analysis_run_in_progress = False
        self._analysis_import_job_id = 0
        self._analysis_run_job_id = 0
        self._analysis_role_editor = None
        self._analysis_role_edit_item = ""
        self.target_load_curve = []
        self._cached_steady_intervals = {}

    def _on_main_prediction_config_changed(self):
        self.refresh_prediction_mode_controls()
        self.refresh_prediction_metrics_summary()
        if self.data:
            try:
                interval_policy = self._get_default_interval_policy() if hasattr(self, "_get_default_interval_policy") else "fresh_or_empty"
                self.generate_plots(silent=True, interval_policy=interval_policy)
            except Exception:
                pass

    def _on_main_pit_config_changed(self):
        self.refresh_main_pit_preview()

    def has_posterior_curve_ready(self):
        return False

    def get_effective_prediction_mode(self, mode=None):
        return "direct_prediction"

    def refresh_prediction_mode_controls(self, prefer_posterior=False):
        target_mode = self.get_effective_prediction_mode()
        if str(self.prediction_mode_var.get()).strip() != target_mode:
            self.prediction_mode_var.set(target_mode)

    def _set_data_analysis_busy_state(self):
        busy = bool(self.analysis_import_in_progress or self.analysis_run_in_progress)
        for widget_name in (
            "analysis_import_btn",
            "analysis_remove_btn",
            "analysis_run_btn",
            "analysis_export_btn",
            "analysis_apply_mapping_btn",
        ):
            widget = getattr(self, widget_name, None)
            if widget is None:
                continue
            try:
                if busy:
                    widget.state(["disabled"])
                else:
                    widget.state(["!disabled"])
            except Exception:
                continue

    def get_current_prediction_case_profile_name(self):
        active_path = str(getattr(self, "active_kc_profile_path", "") or "").strip()
        if active_path:
            return os.path.basename(active_path)

        active_profile = getattr(self, "active_kc_profile", None)
        if not isinstance(active_profile, dict) and hasattr(self, "_get_saved_kc_profile_for_input"):
            try:
                active_profile = self._get_saved_kc_profile_for_input()
            except Exception:
                active_profile = None

        if hasattr(self, "_profile_has_saved_payload"):
            if self._profile_has_saved_payload(active_profile):
                return "当前内存配置（未保存）"
        elif isinstance(active_profile, dict) and (active_profile.get("line_kc_map") or active_profile.get("pit_records")):
            return "当前内存配置（未保存）"
        return "未加载案例配置"

    def _iter_pit_interval_ranges(self):
        for idx, interval in enumerate(self._get_current_interval_records(allow_profile_fallback=False), 1):
            interval_id = interval.get("zone_id") or interval.get("interval_id") or f"Z{idx:03d}"
            try:
                start_idx = int(interval.get("start_idx"))
                end_idx = int(interval.get("end_idx"))
            except Exception:
                process_bounds = self._resolve_interval_process_bounds(interval)
                if not process_bounds:
                    continue
                start_idx = int(process_bounds.get("start_idx"))
                end_idx = int(process_bounds.get("end_idx"))
            if end_idx < start_idx:
                continue
            yield interval_id, start_idx, end_idx

    def build_current_process_dataframe(self):
        if not self.data:
            return pd.DataFrame()

        interval_marks = {}
        for interval_id, start_idx, end_idx in self._iter_pit_interval_ranges():
            for row_idx in range(start_idx, end_idx + 1):
                interval_marks[row_idx] = interval_id

        actual_feed_lookup = self._build_actual_feed_speed_lookup()

        rows = []
        cumulative_path = 0.0
        for idx, row in enumerate(self.data):
            step_length = float(row.get("s", 0.0) or 0.0)
            try:
                feed_effective = float(row.get("feed_effective"))
            except Exception:
                feed_effective = float("nan")
            try:
                path_start = float(row.get("path_start"))
                path_end = float(row.get("path_end"))
            except Exception:
                path_start = cumulative_path
                cumulative_path += step_length
                path_end = cumulative_path
            else:
                cumulative_path = path_end
            ap_val = float(row.get("ap", 0.0) or 0.0)
            ae_val = float(row.get("ae", 0.0) or 0.0)
            mrr_val = float(row.get("MRR", 0.0) or 0.0)
            feed_plan = mrr_val * 60.0 / (ap_val * ae_val) if ap_val > 1e-12 and ae_val > 1e-12 else 0.0
            try:
                raw_line_key = int(row.get("line_no_raw"))
            except Exception:
                raw_line_key = None
            rows.append({
                "sample_index": idx,
                "time_ms": idx,
                "time_s": idx / 1000.0,
                "path_start": path_start,
                "path_end": path_end,
                "path_cumulative": path_end,
                "path_position": path_end if path_end > 0 else float(idx),
                "N_str": row.get("N_str"),
                "line_no_raw": row.get("line_no_raw"),
                "line_no_aligned": row.get("line_no_aligned"),
                "interval_id": interval_marks.get(idx, ""),
                "is_steady": idx in interval_marks,
                "ap": ap_val,
                "ae": ae_val,
                "Fact": actual_feed_lookup.get(raw_line_key, np.nan),
                "feed_effective": feed_effective,
                "feed_plan": feed_plan,
                "MRR": mrr_val,
                "S": float(row.get("S", 0.0) or 0.0),
                "P": float(row.get("P", 0.0) or 0.0),
                "P_idle": float(row.get("P_idle", 0.0) or 0.0),
                "P_edge": float(row.get("P_edge", 0.0) or 0.0),
                "K_c": float(row.get("K_c", 0.0) or 0.0),
                "K_e": float(row.get("K_e", 0.0) or 0.0),
                "x": float(row.get("x", 0.0) or 0.0),
                "y": float(row.get("y", 0.0) or 0.0),
                "z": float(row.get("z", 0.0) or 0.0),
                "gcode_content": str(row.get("gcode_content", "") or ""),
            })
        return pd.DataFrame(rows)

    def _build_actual_feed_speed_lookup(self):
        measurement = getattr(self, "manual_measurement_data", None)
        if not isinstance(measurement, dict):
            return {}

        line_values = np.asarray(measurement.get("program_line", []), dtype=int)
        feed_values = np.asarray(measurement.get("actual_feed_speed", []), dtype=float)
        if line_values.size == 0 or line_values.size != feed_values.size:
            return {}

        sample_df = pd.DataFrame({
            "line_no_raw": line_values,
            "actual_feed_speed": feed_values,
        })
        sample_df["actual_feed_speed"] = pd.to_numeric(sample_df["actual_feed_speed"], errors="coerce")
        sample_df = sample_df[np.isfinite(sample_df["actual_feed_speed"].to_numpy(dtype=float))].copy()
        if sample_df.empty:
            return {}

        grouped = sample_df.groupby("line_no_raw", sort=False)["actual_feed_speed"].median()
        return {int(key): float(value) for key, value in grouped.items() if np.isfinite(value)}

    def _build_sample_level_pit_dataframe(self):
        """优先按实测采样点构建 PIT 预览数据。

        时域直接继承实测负载的采样索引：1 行 = 1 ms。
        """
        prediction_df, _ = self.build_current_prediction_dataframe()
        if prediction_df.empty:
            return pd.DataFrame()

        sample_mode = str(getattr(self, "sample_data_mode", "") or "").strip()
        sample_count = len(prediction_df)
        ke_value = float(self.get_ke_value())

        extra_df = None
        if sample_mode == "sampledata":
            payload = self._build_sampledata_prediction_payload()
            if not payload:
                return pd.DataFrame()
            actual_load = np.abs(np.asarray(payload.get("actual_load", []), dtype=float))
            if actual_load.size == 0:
                return pd.DataFrame()
            extra_df = pd.DataFrame({
                "sample_index": np.arange(actual_load.size, dtype=int),
                "ap": np.asarray(payload.get("mapped_ap", np.full(actual_load.shape, np.nan)), dtype=float),
                "ae": np.asarray(payload.get("mapped_ae", np.full(actual_load.shape, np.nan)), dtype=float),
                "feed_plan": np.asarray(payload.get("mapped_feed", np.full(actual_load.shape, np.nan)), dtype=float),
                "MRR": np.asarray(payload.get("mapped_mrr", np.full(actual_load.shape, np.nan)), dtype=float),
                "S": np.asarray(payload.get("mapped_speed", np.full(actual_load.shape, np.nan)), dtype=float),
                "P": actual_load,
                "P_idle": np.asarray(payload.get("predicted_idle_power", np.full(actual_load.shape, np.nan)), dtype=float),
                "K_c": np.asarray(payload.get("mapped_kc", np.full(actual_load.shape, np.nan)), dtype=float),
                "K_e": np.full(actual_load.shape, ke_value, dtype=float),
                "P_edge": np.full(actual_load.shape, np.nan, dtype=float),
            })
        elif sample_mode == "experiment_measurement":
            try:
                sample_df = self._build_manual_measurement_sample_frame()
            except Exception:
                return pd.DataFrame()
            if sample_df.empty:
                return pd.DataFrame()
            feed_plan = pd.to_numeric(sample_df.get("feed_plan"), errors="coerce")
            if "feed_speed" in sample_df:
                feed_plan = feed_plan.fillna(pd.to_numeric(sample_df["feed_speed"], errors="coerce"))
            speed_plan = pd.to_numeric(sample_df.get("speed_plan"), errors="coerce")
            if "spindle_speed" in sample_df:
                speed_plan = speed_plan.fillna(pd.to_numeric(sample_df["spindle_speed"], errors="coerce"))
            extra_df = pd.DataFrame({
                "sample_index": pd.to_numeric(sample_df.get("sample_index"), errors="coerce").fillna(0).astype(int),
                "ap": pd.to_numeric(sample_df.get("ap"), errors="coerce"),
                "ae": pd.to_numeric(sample_df.get("ae"), errors="coerce"),
                "feed_plan": feed_plan,
                "MRR": pd.to_numeric(sample_df.get("mrr"), errors="coerce"),
                "S": speed_plan,
                "P": pd.to_numeric(sample_df.get("actual_load"), errors="coerce"),
                "P_idle": pd.to_numeric(sample_df.get("idle_power"), errors="coerce"),
                "K_c": pd.to_numeric(sample_df.get("predicted_kc"), errors="coerce"),
                "K_e": np.full(len(sample_df), ke_value, dtype=float),
                "P_edge": np.full(len(sample_df), np.nan, dtype=float),
            })
        else:
            return pd.DataFrame()

        if extra_df is None or extra_df.empty:
            return pd.DataFrame()

        pit_df = prediction_df.merge(extra_df, on="sample_index", how="left", sort=False)
        if len(pit_df) != sample_count:
            return pd.DataFrame()
        pit_df["is_steady"] = pit_df["state"].astype(str).eq("稳态")
        return pit_df

    def build_current_pit_dataframe(self, scope="all"):
        pit_df = self.build_current_process_dataframe()
        if pit_df.empty:
            pit_df = self._build_sample_level_pit_dataframe()
        if pit_df.empty:
            return pit_df
        pit_df = self._with_pit_display_fields(pit_df)
        if str(scope) == "steady":
            pit_df = pit_df[pit_df["is_steady"]].copy()
        return pit_df.reset_index(drop=True)

    def _with_pit_display_fields(self, pit_df):
        if not isinstance(pit_df, pd.DataFrame) or pit_df.empty:
            return pit_df

        enriched = pit_df.copy()

        def first_numeric(*column_names, default=np.nan):
            result = pd.Series(np.full(len(enriched), float(default), dtype=float), index=enriched.index, dtype=float)
            for column_name in column_names:
                if column_name not in enriched.columns:
                    continue
                numeric_values = pd.to_numeric(enriched[column_name], errors="coerce")
                result = result.where(result.notna(), numeric_values)
            return result

        def format_program_line(*column_names):
            numeric_values = None
            for column_name in column_names:
                if column_name not in enriched.columns:
                    continue
                candidate = pd.to_numeric(enriched[column_name], errors="coerce")
                if numeric_values is None:
                    numeric_values = candidate
                else:
                    numeric_values = numeric_values.where(numeric_values.notna(), candidate)
            if numeric_values is None:
                return pd.Series([""] * len(enriched), index=enriched.index, dtype=object)
            return numeric_values.map(
                lambda value: "" if not np.isfinite(value) else f"N{int(round(float(value))) + 1}"
            )

        if "程序行号N" not in enriched.columns:
            display_program_line = pd.Series([""] * len(enriched), index=enriched.index, dtype=object)
            if "N_str" in enriched.columns:
                n_text = enriched["N_str"].fillna("").astype(str).str.strip()
                display_program_line = n_text.where(n_text.ne(""), "")
            fallback_program_line = format_program_line("line_no_raw", "line_no_aligned")
            enriched["程序行号N"] = display_program_line.where(display_program_line.ne(""), fallback_program_line)
        if "稳态区间号" not in enriched.columns:
            if "interval_id" in enriched.columns:
                enriched["稳态区间号"] = enriched["interval_id"].fillna("").astype(str)
            else:
                enriched["稳态区间号"] = ""
        if "切深ap" not in enriched.columns:
            enriched["切深ap"] = first_numeric("ap")
        if "切宽ae" not in enriched.columns:
            enriched["切宽ae"] = first_numeric("ae")
        if "进给F" not in enriched.columns:
            enriched["进给F"] = first_numeric("feed_effective", "feed_plan")
        if "主轴转速S" not in enriched.columns:
            enriched["主轴转速S"] = first_numeric("S", "转速S")
        if "累计行程s" not in enriched.columns:
            enriched["累计行程s"] = first_numeric("path_end", "path_position", "path_cumulative")
        if "G代码内容G" not in enriched.columns:
            if "gcode_content" in enriched.columns:
                enriched["G代码内容G"] = enriched["gcode_content"].fillna("").astype(str)
            else:
                enriched["G代码内容G"] = ""
        if "实际进给速度Fact" not in enriched.columns:
            fact_values = pd.Series(first_numeric("Fact", "actual_feed_speed"), index=enriched.index, dtype=float)
            enriched["实际进给速度Fact"] = fact_values.ffill().to_numpy(dtype=float)
        if "指令进给速度Fpred" not in enriched.columns:
            enriched["指令进给速度Fpred"] = first_numeric("Fpred", "feed_plan")
        if "kc" not in enriched.columns:
            enriched["kc"] = first_numeric("kc", "K_c")
        if "ke" not in enriched.columns:
            enriched["ke"] = first_numeric("ke", "K_e")
        if "转速S" not in enriched.columns:
            enriched["转速S"] = first_numeric("主轴转速S", "S")
        return enriched

    def _get_pit_display_columns(self, pit_df):
        preferred_columns = [
            "稳态区间号",
            "程序行号N",
            "切深ap",
            "切宽ae",
            "进给F",
            "主轴转速S",
            "累计行程s",
            "G代码内容G",
            "实际进给速度Fact",
            "kc",
            "ke",
            "P",
            "P_idle",
            "MRR",
        ]
        return [column for column in preferred_columns if column in pit_df.columns]

    def _pit_field_has_finite_values(self, pit_df, field_name):
        if not isinstance(pit_df, pd.DataFrame) or pit_df.empty or field_name not in pit_df.columns:
            return False
        values = pd.to_numeric(pit_df[field_name], errors="coerce").to_numpy(dtype=float)
        return bool(np.any(np.isfinite(values)))

    def _pit_numeric_array(self, frame, primary_key, fallback_key=None, default=np.nan):
        if not isinstance(frame, pd.DataFrame):
            return np.asarray([], dtype=float)
        source = None
        if primary_key in frame.columns:
            source = frame[primary_key]
        elif fallback_key and fallback_key in frame.columns:
            source = frame[fallback_key]
        if source is None:
            return np.full(len(frame), float(default), dtype=float)
        return pd.to_numeric(source, errors="coerce").to_numpy(dtype=float)

    def _remove_pit_line_axis(self, ax):
        top_ax = getattr(ax, "_pit_line_top_axis", None)
        if top_ax is None:
            return
        try:
            top_ax.remove()
        except Exception:
            pass
        ax._pit_line_top_axis = None

    def _build_pit_line_spans(self, pit_df, axis_mode):
        line_series = pd.to_numeric(pit_df.get("line_no_raw"), errors="coerce")
        if line_series.isna().all():
            line_series = pd.to_numeric(pit_df.get("line_no_aligned"), errors="coerce")
        line_values = line_series.to_numpy(dtype=float)
        if not np.any(np.isfinite(line_values)):
            return []

        if axis_mode == "行程域+指令域":
            start_values = self._pit_numeric_array(pit_df, "path_start")
            end_values = self._pit_numeric_array(pit_df, "path_end")
            if not np.any(np.isfinite(start_values)) or not np.any(np.isfinite(end_values)):
                position_values = self._pit_numeric_array(pit_df, "path_position", fallback_key="path_end")
                start_values = position_values.copy()
                end_values = position_values.copy()
        else:
            start_values = self._pit_numeric_array(pit_df, "time_start_ms", fallback_key="time_ms")
            end_values = self._pit_numeric_array(pit_df, "time_end_ms")
            if not np.any(np.isfinite(end_values)):
                end_values = start_values + 1.0

        valid_mask = np.isfinite(line_values) & np.isfinite(start_values) & np.isfinite(end_values)
        if not np.any(valid_mask):
            return []

        spans = []
        run_start = None
        current_line = None
        for idx, is_valid in enumerate(valid_mask):
            if not is_valid:
                if run_start is not None:
                    spans.append((
                        current_line,
                        float(start_values[run_start]),
                        float(max(end_values[idx - 1], start_values[run_start])),
                    ))
                    run_start = None
                    current_line = None
                continue
            line_no = int(line_values[idx])
            if run_start is None:
                run_start = idx
                current_line = line_no
                continue
            if line_no != current_line:
                spans.append((
                    current_line,
                    float(start_values[run_start]),
                    float(max(end_values[idx - 1], start_values[run_start])),
                ))
                run_start = idx
                current_line = line_no
        if run_start is not None:
            spans.append((
                current_line,
                float(start_values[run_start]),
                float(max(end_values[len(valid_mask) - 1], start_values[run_start])),
            ))
        return spans

    def _refresh_pit_line_axis(self, ax):
        if getattr(ax, "_pit_line_refreshing", False):
            return None
        spans = list(getattr(ax, "_pit_line_spans", []) or [])
        self._remove_pit_line_axis(ax)
        if not spans:
            return None

        ax._pit_line_refreshing = True
        x_min, x_max = ax.get_xlim()
        try:
            visible_spans = [
                (line_no, start_val, end_val)
                for line_no, start_val, end_val in spans
                if end_val >= x_min and start_val <= x_max
            ] or spans

            try:
                capacity = self.get_time_line_axis_capacity(ax, min_ticks=8, max_ticks=40)
                visible_spans = self.sample_time_line_spans(visible_spans, capacity)
            except Exception:
                pass

            tick_positions = []
            tick_labels = []
            for span in visible_spans:
                if not span or len(span) < 3:
                    continue
                try:
                    line_no = int(span[0])
                except Exception:
                    continue
                if len(span) >= 4:
                    try:
                        center_val = float(span[1])
                        start_val = float(span[2])
                        end_val = float(span[3])
                    except Exception:
                        continue
                    if np.isfinite(center_val):
                        tick_pos = min(max(center_val, x_min), x_max)
                    else:
                        tick_pos = (max(start_val, x_min) + min(end_val, x_max)) / 2.0
                else:
                    try:
                        start_val = float(span[1])
                        end_val = float(span[2])
                    except Exception:
                        continue
                    tick_pos = (max(start_val, x_min) + min(end_val, x_max)) / 2.0
                tick_positions.append(float(tick_pos))
                tick_labels.append(str(int(line_no)))
            if not tick_positions:
                return None

            top_ax = ax.secondary_xaxis("top")
            top_ax.set_xticks(tick_positions)
            top_ax.set_xticklabels(tick_labels, rotation=45, ha="left")
            top_ax.set_xlabel("程序行号", fontsize=PLOT_FONT_BASE, fontweight="bold", color=PLOT_TEXT_COLOR)
            top_ax.tick_params(axis="x", labelsize=PLOT_FONT_BASE - 1, colors=PLOT_TEXT_COLOR, pad=2)
            try:
                top_ax.spines["top"].set_color(PLOT_SPINE_COLOR)
            except Exception:
                pass
            ax._pit_line_top_axis = top_ax
            return top_ax
        finally:
            ax._pit_line_refreshing = False

    def _bind_pit_line_axis_updates(self, ax):
        if getattr(ax, "_pit_line_callback_bound", False):
            return
        try:
            ax.callbacks.connect("xlim_changed", lambda current_ax: self._refresh_pit_line_axis(current_ax))
            ax._pit_line_callback_bound = True
        except Exception:
            pass

    def _apply_pit_line_axis(self, ax, pit_df, axis_mode):
        """为 PIT 图附加顶部程序行号轴。"""
        ax._pit_line_spans = self._build_pit_line_spans(pit_df, axis_mode)
        self._bind_pit_line_axis_updates(ax)
        return self._refresh_pit_line_axis(ax)

    def _get_default_pit_fields(self, pit_df):
        defaults = []
        for field in ("切深ap", "切宽ae", "进给F", "主轴转速S", "实际进给速度Fact", "kc", "ke", "P", "P_idle", "MRR"):
            if self._pit_field_has_finite_values(pit_df, field):
                defaults.append(field)
        return defaults[:1]

    def _get_selected_pit_fields(self, pit_df):
        if hasattr(self, "pit_field_var"):
            field_items = list(getattr(self, "pit_field_combo", {}).cget("values")) if hasattr(getattr(self, "pit_field_combo", None), "cget") else []
            selected_field = str(self.pit_field_var.get()).strip()
            if selected_field and selected_field in pit_df.columns and self._pit_field_has_finite_values(pit_df, selected_field):
                return [selected_field]
            defaults = self._get_default_pit_fields(pit_df)
            default_field = defaults[0] if defaults else ""
            if default_field:
                self.pit_field_var.set(default_field)
                return [default_field]
            return []
        if not hasattr(self, "pit_field_listbox"):
            return self._get_default_pit_fields(pit_df)

        selected_indices = list(self.pit_field_listbox.curselection())
        if not selected_indices:
            defaults = self._get_default_pit_fields(pit_df)
            field_items = list(self.pit_field_listbox.get(0, tk.END))
            for field in defaults:
                if field in field_items:
                    idx = field_items.index(field)
                    self.pit_field_listbox.selection_set(idx)
                    selected_indices.append(idx)

        fields = []
        for idx in selected_indices:
            try:
                fields.append(str(self.pit_field_listbox.get(idx)))
            except Exception:
                continue
        return fields or self._get_default_pit_fields(pit_df)

    def _update_pit_field_options(self, pit_df):
        numeric_fields = []
        for column in self._get_pit_display_columns(pit_df):
            if pd.api.types.is_numeric_dtype(pit_df[column]):
                numeric_fields.append(column)

        if hasattr(self, "pit_field_combo"):
            current_items = list(self.pit_field_combo.cget("values"))
            if current_items != numeric_fields:
                self.pit_field_combo.configure(values=numeric_fields)
            selected_field = str(self.pit_field_var.get()).strip()
            if selected_field not in numeric_fields or not self._pit_field_has_finite_values(pit_df, selected_field):
                defaults = self._get_default_pit_fields(pit_df)
                self.pit_field_var.set(defaults[0] if defaults else "")
            return

        if not hasattr(self, "pit_field_listbox"):
            return

        current_items = list(self.pit_field_listbox.get(0, tk.END))
        if current_items == numeric_fields:
            return

        self.pit_field_listbox.delete(0, tk.END)
        for field in numeric_fields:
            self.pit_field_listbox.insert(tk.END, field)
        for field in self._get_default_pit_fields(pit_df):
            if field in numeric_fields:
                self.pit_field_listbox.selection_set(numeric_fields.index(field))

    def _ensure_pit_table(self):
        if hasattr(self, "pit_tree"):
            return
        tree = ttk.Treeview(self.pit_table_container, show="headings")
        y_scroll = ttk.Scrollbar(self.pit_table_container, orient=tk.VERTICAL, command=tree.yview)
        x_scroll = ttk.Scrollbar(self.pit_table_container, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.pit_table_container.grid_rowconfigure(0, weight=1)
        self.pit_table_container.grid_columnconfigure(0, weight=1)
        self.pit_tree = tree

    def _populate_treeview(self, tree, frame, max_rows=5000, width_scale=11):
        if frame is None:
            frame = pd.DataFrame()
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame)

        columns = list(frame.columns)
        width_overrides = {
            "程序行号N": 110,
            "切深ap": 90,
            "切宽ae": 90,
            "进给F": 100,
            "主轴转速S": 110,
            "累计行程s": 110,
            "G代码内容G": 420,
            "稳态区间号": 110,
            "实际进给速度Fact": 120,
            "指令进给速度Fpred": 130,
            "kc": 100,
            "ke": 100,
            "P": 90,
            "P_idle": 90,
            "MRR": 110,
        }
        anchor_overrides = {
            "G代码内容G": "w",
        }
        tree.delete(*tree.get_children())
        if getattr(tree, "_codex_columns", None) != columns:
            tree["columns"] = columns
            for column in columns:
                tree.heading(column, text=str(column))
                width = width_overrides.get(column, max(80, min(220, len(str(column)) * width_scale)))
                anchor = anchor_overrides.get(column, "center")
                tree.column(column, width=width, anchor=anchor, stretch=True)
            tree._codex_columns = columns

        if frame.empty:
            return

        for _, row in frame.head(int(max_rows)).iterrows():
            values = []
            for value in row.tolist():
                if isinstance(value, float):
                    if not np.isfinite(value):
                        values.append("")
                    elif abs(value) >= 1000:
                        values.append(f"{value:.3f}")
                    else:
                        values.append(f"{value:.6f}".rstrip("0").rstrip("."))
                else:
                    values.append("" if value is None else value)
            tree.insert("", tk.END, values=values)

    def _render_main_pit_plot(self, pit_df):
        for widget in self.pit_plot_container.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(10, 5.4), dpi=90)
        fig.patch.set_facecolor(PLOT_FIG_BG)
        ax.set_facecolor(PLOT_AX_BG)
        self.apply_plot_style(ax, grid=True)
        self._pit_preview_fig = fig
        self._pit_preview_axes = ax

        axis_mode = str(getattr(self, "pit_axis_mode_var", tk.StringVar(value="时域+指令域")).get())
        if axis_mode == "行程域+指令域":
            x_values = self._pit_numeric_array(pit_df, "path_position", fallback_key="path_end")
            if not np.any(np.isfinite(x_values)):
                x_values = pit_df["sample_index"].to_numpy(dtype=float)
                x_label = "样本序号"
            else:
                x_label = "累计行程 (mm)"
        else:
            x_values = pd.to_numeric(pit_df["time_ms"], errors="coerce").to_numpy(dtype=float)
            if not np.any(np.isfinite(x_values)):
                x_values = pit_df["sample_index"].to_numpy(dtype=float)
            x_label = "时间 (ms)"

        color_cycle = ["#1F77B4", "#E15759", "#2CA02C", "#FF9F1C", "#9467BD", "#17BECF"]
        selected_fields = self._get_selected_pit_fields(pit_df)
        selected_field = selected_fields[0] if selected_fields else ""
        if selected_field and not self._pit_field_has_finite_values(pit_df, selected_field):
            selected_field = ""
        if not selected_field:
            fallback_fields = self._get_default_pit_fields(pit_df)
            selected_field = fallback_fields[0] if fallback_fields else ""
            if selected_field and hasattr(self, "pit_field_var"):
                self.pit_field_var.set(selected_field)
        if selected_field in pit_df.columns:
            y_values = pd.to_numeric(pit_df[selected_field], errors="coerce").to_numpy(dtype=float)
            valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
            if np.any(valid_mask):
                ax.plot(
                    x_values[valid_mask],
                    y_values[valid_mask],
                    label=selected_field,
                    linewidth=1.6,
                    color=color_cycle[0],
                    alpha=0.95,
                    zorder=5,
                )

        if self.pit_scope_var.get() == "all":
            steady_df = pit_df[pit_df["is_steady"]].copy()
            if not steady_df.empty:
                for _, group in steady_df.groupby("interval_id", sort=False):
                    if axis_mode == "行程域+指令域":
                        start_x_values = self._pit_numeric_array(group, "path_start")
                        end_x_values = self._pit_numeric_array(group, "path_end")
                        if not np.any(np.isfinite(start_x_values)) or not np.any(np.isfinite(end_x_values)):
                            interval_x = self._pit_numeric_array(group, "path_position")
                            if not np.any(np.isfinite(interval_x)):
                                continue
                            start_x = float(np.nanmin(interval_x))
                            end_x = float(np.nanmax(interval_x))
                        else:
                            start_x = float(np.nanmin(start_x_values))
                            end_x = float(np.nanmax(end_x_values))
                    else:
                        start_x_values = self._pit_numeric_array(group, "time_start_ms", fallback_key="time_ms")
                        end_x_values = self._pit_numeric_array(group, "time_end_ms")
                        if not np.any(np.isfinite(end_x_values)):
                            end_x_values = start_x_values + 1.0
                        if not np.any(np.isfinite(start_x_values)) or not np.any(np.isfinite(end_x_values)):
                            continue
                        start_x = float(np.nanmin(start_x_values))
                        end_x = float(np.nanmax(end_x_values))
                    if not np.isfinite(start_x) or not np.isfinite(end_x):
                        continue
                    ax.axvline(start_x, color="#B0BEC5", linewidth=0.8, linestyle="--", alpha=0.6, zorder=2)
                    ax.axvline(end_x, color="#B0BEC5", linewidth=0.8, linestyle="--", alpha=0.6, zorder=2)

        ax.set_xlabel(x_label, fontsize=PLOT_FONT_BASE, fontweight="bold", color=PLOT_TEXT_COLOR)
        ax.set_ylabel(selected_field or "工艺参数值", fontsize=PLOT_FONT_BASE, fontweight="bold", color=PLOT_TEXT_COLOR)
        scope_text = "全部工艺点" if self.pit_scope_var.get() == "all" else "仅稳态区间工艺点"
        title_field = selected_field if selected_field else "工艺参数"
        ax.set_title(f"PIT {title_field} 点级图 ({scope_text})", fontsize=PLOT_FONT_BASE + 2, fontweight="bold", color=PLOT_TEXT_COLOR, loc="left")
        self._apply_pit_line_axis(ax, pit_df, axis_mode)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            legend = ax.legend(loc="upper right", fontsize=PLOT_FONT_BASE - 1, framealpha=0.95)
            legend.get_frame().set_facecolor("white")
            legend.get_frame().set_edgecolor(PLOT_SPINE_COLOR)
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.84)

        canvas = FigureCanvasTkAgg(fig, master=self.pit_plot_container)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.pit_plot_container.grid_rowconfigure(0, weight=1)
        self.pit_plot_container.grid_columnconfigure(0, weight=1)
        canvas.draw_idle()
        self._pit_preview_canvas = canvas

    def refresh_main_pit_preview(self):
        if not hasattr(self, "pit_content_stack"):
            return

        pit_df = self.build_current_pit_dataframe(self.pit_scope_var.get())
        if pit_df.empty and str(self.pit_scope_var.get()).strip() == "steady":
            fallback_df = self.build_current_pit_dataframe("all")
            if not fallback_df.empty:
                self.pit_scope_var.set("all")
                pit_df = fallback_df
        if pit_df.empty:
            self.pit_status_var.set("PIT 预览: 当前没有可显示的点级工艺信息")
            for child in self.pit_content_stack.winfo_children():
                child.grid_remove()
            return

        self._update_pit_field_options(pit_df)
        scope_text = "全部工艺点" if self.pit_scope_var.get() == "all" else "仅稳态区间工艺点"
        interval_ids = pit_df["interval_id"].astype(str)
        interval_count = int(interval_ids[interval_ids != ""].nunique())
        self.pit_status_var.set(f"PIT 预览: {scope_text}，共 {len(pit_df)} 点，稳态区间 {interval_count} 段")

        for child in self.pit_content_stack.winfo_children():
            child.grid_remove()

        if self.pit_view_mode_var.get() == "table":
            self._ensure_pit_table()
            self.pit_table_container.grid(row=0, column=0, sticky="nsew")
            display_columns = self._get_pit_display_columns(pit_df)
            display_df = pit_df.loc[:, display_columns].copy() if display_columns else pit_df.copy()
            self._populate_treeview(self.pit_tree, display_df)
        else:
            self.pit_plot_container.grid(row=0, column=0, sticky="nsew")
            try:
                self._render_main_pit_plot(pit_df)
            except Exception as exc:
                self.pit_status_var.set(f"PIT 预览绘制失败: {str(exc)}")
                if hasattr(self, "set_status"):
                    self.set_status(f"PIT 预览绘制失败: {str(exc)}", 8000)
                raise

    def get_prediction_curve_label(self, mode=None):
        return "预测负载"

    def _get_steady_metric_trim_count(self, sample_count):
        try:
            count = int(sample_count)
        except Exception:
            count = 0
        if count < 5:
            return 0
        trim_ratio = float(getattr(self, "steady_metric_trim_ratio", 0.10) or 0.10)
        min_points = int(getattr(self, "steady_metric_trim_min_points", 1) or 1)
        trim_count = max(min_points, int(math.floor(count * trim_ratio)))
        if trim_count * 2 >= count:
            return 0
        return int(trim_count)

    def _apply_trim_to_interval_mask(self, mask):
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.size == 0 or not np.any(mask_arr):
            return mask_arr.copy(), 0
        indices = np.flatnonzero(mask_arr)
        trim_count = self._get_steady_metric_trim_count(indices.size)
        if trim_count <= 0:
            return mask_arr.copy(), 0
        trimmed_indices = indices[trim_count:indices.size - trim_count]
        if trimmed_indices.size == 0:
            return mask_arr.copy(), 0
        trimmed_mask = np.zeros(mask_arr.shape, dtype=bool)
        trimmed_mask[trimmed_indices] = True
        return trimmed_mask, int(trim_count)

    def _get_steady_metric_trim_description(self):
        trim_ratio = float(getattr(self, "steady_metric_trim_ratio", 0.10) or 0.10)
        min_points = int(getattr(self, "steady_metric_trim_min_points", 1) or 1)
        return f"稳态误差按每段两端各裁 {trim_ratio * 100.0:.0f}%（至少{min_points}点，短段不裁）"

    def _resolve_current_prediction_view_mask(self, expected_size):
        if expected_size <= 0:
            return np.zeros(0, dtype=bool)

        try:
            program_no = self.get_selected_program_number()
            if self.sample_display_mode.get() == "tool":
                tool_ranges = self.get_selected_tool_ranges()
            else:
                tool_ranges = self.get_program_ranges(self.get_current_program_key())
            mask = self.build_sample_mask(program_no, tool_ranges)
        except Exception:
            mask = None

        if mask is None:
            return np.ones(expected_size, dtype=bool)

        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.size != expected_size:
            return np.ones(expected_size, dtype=bool)
        return mask_arr

    def _build_sampledata_prediction_payload(self):
        if getattr(self, "sample_data_mode", "") != "sampledata":
            return None
        if not self.sample_data_loaded or self.sample_data_values is None or not self.data:
            return None

        values = np.asarray(self.sample_data_values, dtype=float)
        if values.ndim != 2 or values.shape[0] == 0:
            return None

        source_idx = int(self.sample_data_source.get())
        if source_idx < 0 or source_idx >= values.shape[1]:
            return None

        raw_lines = np.asarray(getattr(self, "sample_data_line_numbers", []), dtype=int)
        if raw_lines.size != values.shape[0]:
            return None

        sample_df = pd.DataFrame({
            "sample_index": np.arange(len(raw_lines), dtype=int),
            "line_no_raw": raw_lines,
            "actual_load": np.asarray(values[:, source_idx], dtype=float),
        })
        try:
            process_df = self._build_aligned_process_geometry_frame(raw_lines)
        except Exception:
            return None
        sample_df = pd.concat([sample_df.reset_index(drop=True), process_df.reset_index(drop=True)], axis=1)

        aligned_lines = self.align_line_numbers_to_processed(raw_lines)
        if aligned_lines is None or len(aligned_lines) != len(sample_df):
            aligned_lines = raw_lines
        sample_df["line_no_aligned"] = (
            pd.to_numeric(sample_df["line_no_aligned"], errors="coerce")
            .fillna(pd.Series(aligned_lines, index=sample_df.index))
            .astype(int)
        )

        ap_values = pd.to_numeric(sample_df["ap"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        ae_values = pd.to_numeric(sample_df["ae"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        feed_values = pd.to_numeric(sample_df["feed_plan"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        speed_values = pd.to_numeric(sample_df["speed_plan"], errors="coerce").to_numpy(dtype=float)

        fallback_speed = 0.0
        for candidate in (
            getattr(self, "current_program_speed", None),
            getattr(self, "s_base", None),
        ):
            if candidate is None:
                continue
            try:
                fallback_speed = float(candidate.get())
            except Exception:
                continue
            if np.isfinite(fallback_speed) and fallback_speed > 0.0:
                break

        speed_for_idle = np.asarray(speed_values, dtype=float).copy()
        invalid_speed_mask = ~np.isfinite(speed_for_idle) | (speed_for_idle <= 0.0)
        if np.isfinite(fallback_speed) and fallback_speed > 0.0:
            speed_for_idle[invalid_speed_mask] = fallback_speed

        idle_power = np.full(len(sample_df), np.nan, dtype=float)
        positive_speed_mask = np.isfinite(speed_for_idle) & (speed_for_idle > 1e-9)
        if np.any(positive_speed_mask):
            idle_power[positive_speed_mask] = np.asarray(
                [self.predict_idle_power(speed) for speed in speed_for_idle[positive_speed_mask]],
                dtype=float,
            )

        mrr_values = ap_values * ae_values * feed_values / 60.0
        local_kc_values = pd.to_numeric(sample_df.get("process_kc"), errors="coerce").fillna(self.get_kc_value()).to_numpy(dtype=float)
        local_kc_values = np.maximum(local_kc_values, 0.0)
        ke_value = self._get_effective_ke_value_from_profile(default=self.get_ke_value()) if hasattr(self, "_get_effective_ke_value_from_profile") else self.get_ke_value()
        predicted_load = np.full(len(sample_df), np.nan, dtype=float)
        finite_idle_mask = np.isfinite(idle_power)
        if np.any(finite_idle_mask):
            predicted_load[finite_idle_mask] = (
                idle_power[finite_idle_mask]
                + local_kc_values[finite_idle_mask] * mrr_values[finite_idle_mask]
                + ke_value * ap_values[finite_idle_mask]
            )
        idle_mask = finite_idle_mask & (mrr_values <= 1e-12)
        if np.any(idle_mask):
            predicted_load[idle_mask] = idle_power[idle_mask]
        finite_pred_mask = np.isfinite(predicted_load)
        if np.any(finite_pred_mask):
            predicted_load[finite_pred_mask] = np.maximum(predicted_load[finite_pred_mask], 0.0)

        return {
            "actual_label": self.get_sample_data_source_name(),
            "actual_load": np.abs(np.asarray(values[:, source_idx], dtype=float)),
            "program_line": raw_lines,
            "line_no_aligned": sample_df["line_no_aligned"].to_numpy(dtype=int),
            "predicted_idle_power": idle_power,
            "mapped_ap": ap_values,
            "mapped_ae": ae_values,
            "mapped_feed": feed_values,
            "mapped_mrr": mrr_values,
            "mapped_speed": speed_values,
            "mapped_kc": local_kc_values,
            "predicted_load": predicted_load,
            "prediction_valid_mask": finite_idle_mask,
        }

    def build_current_prediction_dataframe(self, mode=None):
        empty_metrics = {
            "actual_label": "",
            "global": compute_error_metrics([], []),
            "steady": compute_error_metrics([], []),
            "nonsteady": compute_error_metrics([], []),
            "intervals": pd.DataFrame(),
        }
        sample_mode = str(getattr(self, "sample_data_mode", "") or "").strip()
        if sample_mode == "experiment_measurement":
            measurement = getattr(self, "manual_measurement_data", None)
            actual_label = "实际负载"
        elif sample_mode == "sampledata":
            measurement = self._build_sampledata_prediction_payload()
            actual_label = self.get_sample_data_source_name() if measurement else ""
        else:
            return pd.DataFrame(), empty_metrics

        if not measurement:
            return pd.DataFrame(), empty_metrics

        actual_load = np.abs(np.asarray(measurement.get("actual_load", []), dtype=float))
        if actual_load.size == 0:
            return pd.DataFrame(), empty_metrics

        resolved_mode = self.get_effective_prediction_mode(mode)
        prediction_label = self.get_prediction_curve_label(resolved_mode)
        actual_label = str(measurement.get("actual_label") or actual_label or "实际负载")
        case_profile_name = self.get_current_prediction_case_profile_name()
        predicted_load = measurement.get("predicted_load")

        predicted_arr = np.asarray(predicted_load if predicted_load is not None else [], dtype=float)
        if predicted_arr.size != actual_load.size:
            predicted_arr = np.full(actual_load.shape, np.nan, dtype=float)

        sample_index = np.arange(len(actual_load), dtype=int)
        time_ms = sample_index.copy()
        sample_time_indices = self.get_sample_time_indices_array()
        if sample_time_indices is not None and len(sample_time_indices) == len(actual_load):
            time_ms = np.asarray(sample_time_indices, dtype=int)
        if self.sample_data_x_positions is not None and len(self.sample_data_x_positions) == len(actual_load):
            path_position = np.asarray(self.sample_data_x_positions, dtype=float)
        else:
            path_position = np.asarray(time_ms, dtype=float)

        raw_lines = np.asarray(measurement.get("program_line", []), dtype=int)
        aligned_lines = np.asarray(measurement.get("line_no_aligned", measurement.get("program_line", [])), dtype=int)
        if raw_lines.size != len(actual_load):
            raw_lines = aligned_lines.copy()
        sample_context = self._get_current_sample_line_point_context(line_numbers=raw_lines if raw_lines.size == len(actual_load) else None)
        point_labels = np.full(len(actual_load), "", dtype=object)
        if sample_context and len(sample_context["line_numbers"]) == len(actual_load):
            point_indices = np.asarray(sample_context["point_indices"], dtype=int)
            point_labels = np.asarray(
                [self.format_line_point(raw_lines[idx], point_indices[idx]) for idx in range(len(actual_load))],
                dtype=object,
            )

        interval_ids = np.full(len(actual_load), "", dtype=object)
        steady_mask = np.zeros(len(actual_load), dtype=bool)
        steady_metric_mask = np.zeros(len(actual_load), dtype=bool)
        steady_trim_points = np.zeros(len(actual_load), dtype=int)
        interval_trim_meta = []
        for idx, interval in enumerate(self._get_current_interval_records(allow_profile_fallback=False), 1):
            interval_id = interval.get("zone_id") or interval.get("interval_id") or f"Z{idx:03d}"
            mask = self._build_interval_sample_mask(
                interval,
                len(actual_load),
                line_numbers=raw_lines if raw_lines.size == len(actual_load) else None,
            )
            if mask.size != len(actual_load) or not mask.any():
                try:
                    start_line = int(interval.get("start_line"))
                    end_line = int(interval.get("end_line"))
                except Exception:
                    continue
                mask = (aligned_lines >= start_line) & (aligned_lines <= end_line)
            interval_ids[mask] = interval_id
            steady_mask[mask] = True
            trimmed_mask, trim_count = self._apply_trim_to_interval_mask(mask)
            steady_metric_mask[trimmed_mask] = True
            if np.any(trimmed_mask):
                steady_trim_points[trimmed_mask] = int(trim_count)
            interval_trim_meta.append({
                "interval_id": str(interval_id),
                "interval_count": int(np.sum(mask)),
                "trim_count_per_side": int(trim_count),
                "metric_count": int(np.sum(trimmed_mask)),
            })

        error_arr = predicted_arr - actual_load
        relative_error = np.full(actual_load.shape, np.nan, dtype=float)
        relative_floor = get_relative_error_floor(actual_load)
        denom_mask = np.abs(actual_load) > 1e-9
        if np.isfinite(relative_floor):
            denom_mask &= np.abs(actual_load) >= float(relative_floor)
        relative_error[denom_mask] = error_arr[denom_mask] / actual_load[denom_mask] * 100.0

        mapped_mrr = np.asarray(measurement.get("mapped_mrr", []), dtype=float)
        if mapped_mrr.size != len(actual_load):
            mapped_mrr = np.full(actual_load.shape, np.nan, dtype=float)
        predicted_idle_power = np.asarray(measurement.get("predicted_idle_power", []), dtype=float)
        if predicted_idle_power.size != len(actual_load):
            predicted_idle_power = np.full(actual_load.shape, np.nan, dtype=float)

        cutting_mask = np.isfinite(mapped_mrr) & (mapped_mrr > 1e-9)
        if not np.any(cutting_mask):
            dynamic_floor = float(relative_floor) if np.isfinite(relative_floor) else 50.0
            idle_reference = predicted_idle_power.copy()
            idle_reference[~np.isfinite(idle_reference)] = 0.0
            cutting_mask = (
                np.maximum(np.abs(actual_load - idle_reference), np.abs(predicted_arr - idle_reference)) >= dynamic_floor
            )

        prediction_df = pd.DataFrame({
            "案例配置": case_profile_name,
            "sample_index": sample_index,
            "time_ms": time_ms,
            "time_s": time_ms / 1000.0,
            "path_position": path_position,
            "line_no_raw": raw_lines if len(raw_lines) == len(actual_load) else np.nan,
            "line_no_aligned": aligned_lines if len(aligned_lines) == len(actual_load) else np.nan,
            "line_point": point_labels,
            "actual_load": actual_load,
            prediction_label: predicted_arr,
            "predicted_idle_power": predicted_idle_power,
            "mapped_mrr": mapped_mrr,
            "error": error_arr,
            "relative_error_pct": relative_error,
            "relative_error_floor": np.full(actual_load.shape, float(relative_floor) if np.isfinite(relative_floor) else np.nan, dtype=float),
            "state": np.where(steady_mask, "稳态", "非稳态"),
            "metric_state": np.where(steady_metric_mask, "稳态统计", "非稳态统计"),
            "cutting_state": np.where(cutting_mask, "切削段", "非切削段"),
            "interval_id": interval_ids,
            "steady_trim_points": steady_trim_points,
        })
        target_curve = np.asarray(getattr(self, "target_load_curve", []) or [], dtype=float)
        if target_curve.size == len(prediction_df):
            prediction_df["target_load"] = target_curve
        else:
            prediction_df["target_load"] = np.nan
        view_mask = self._resolve_current_prediction_view_mask(len(prediction_df))
        if view_mask.size == len(prediction_df):
            prediction_df = prediction_df.loc[view_mask].copy()

        global_metrics = compute_error_metrics(prediction_df["actual_load"], prediction_df[prediction_label])
        steady_metrics = compute_error_metrics(
            prediction_df.loc[prediction_df["metric_state"] == "稳态统计", "actual_load"],
            prediction_df.loc[prediction_df["metric_state"] == "稳态统计", prediction_label],
        )
        nonsteady_metrics = compute_error_metrics(
            prediction_df.loc[prediction_df["metric_state"] != "稳态统计", "actual_load"],
            prediction_df.loc[prediction_df["metric_state"] != "稳态统计", prediction_label],
        )
        cutting_metrics = compute_error_metrics(
            prediction_df.loc[prediction_df["cutting_state"] == "切削段", "actual_load"],
            prediction_df.loc[prediction_df["cutting_state"] == "切削段", prediction_label],
        )

        interval_rows = []
        for interval_id, group in prediction_df[prediction_df["interval_id"] != ""].groupby("interval_id", sort=False):
            metric_group = group[group["metric_state"] == "稳态统计"]
            target_group = metric_group if not metric_group.empty else group
            metrics = compute_error_metrics(target_group["actual_load"], target_group[prediction_label])
            trim_meta = next((item for item in interval_trim_meta if item["interval_id"] == str(interval_id)), {})
            interval_rows.append({
                "interval_id": str(interval_id),
                "interval_count": int(trim_meta.get("interval_count", len(group))),
                "metric_count": int(trim_meta.get("metric_count", len(target_group))),
                "trim_count_per_side": int(trim_meta.get("trim_count_per_side", 0)),
                **metrics,
            })

        return prediction_df, {
            "actual_label": actual_label,
            "label": prediction_label,
            "case_profile_name": case_profile_name,
            "global": global_metrics,
            "steady": steady_metrics,
            "nonsteady": nonsteady_metrics,
            "cutting": cutting_metrics,
            "steady_trim_desc": self._get_steady_metric_trim_description(),
            "intervals": pd.DataFrame(interval_rows),
        }

    def refresh_prediction_metrics_summary(self):
        _, metrics = self.build_current_prediction_dataframe()
        case_profile_name = str(metrics.get("case_profile_name") or self.get_current_prediction_case_profile_name())
        if not metrics.get("label"):
            self.prediction_metrics_var.set(
                f"预测摘要 | 配置={case_profile_name}: 请先导入 SampleData/实验实测，并完成当前工艺信息处理"
            )
            return

        def _fmt(value):
            return "-" if value is None or not np.isfinite(value) else f"{value:.3f}"

        actual_label = str(metrics.get("actual_label") or "实际负载")
        mode_prefix = "预测"
        global_metrics = metrics["global"]
        steady_metrics = metrics["steady"]
        nonsteady_metrics = metrics["nonsteady"]
        cutting_metrics = metrics.get("cutting", compute_error_metrics([], []))
        trim_desc = str(metrics.get("steady_trim_desc") or "")
        floor_text = _fmt(global_metrics.get("relative_error_floor"))
        floor_suffix = "" if floor_text == "-" else f" (>{floor_text}W)"
        self.prediction_metrics_var.set(
            f"{mode_prefix}摘要 | 配置={case_profile_name} | {actual_label} vs {metrics['label']}: "
            f"稳态 MAE={_fmt(steady_metrics['mae'])}, wMAPE={_fmt(steady_metrics.get('wmape'))}% , 有效MAPE={_fmt(steady_metrics['mape'])}% | "
            f"切削段 MAE={_fmt(cutting_metrics.get('mae'))}, wMAPE={_fmt(cutting_metrics.get('wmape'))}% , 有效MAPE={_fmt(cutting_metrics.get('mape'))}% | "
            f"全局(含非稳态) MAE={_fmt(global_metrics['mae'])}, RMSE={_fmt(global_metrics['rmse'])}, wMAPE={_fmt(global_metrics.get('wmape'))}% , 有效MAPE={_fmt(global_metrics['mape'])}%{floor_suffix}, MaxAE={_fmt(global_metrics['max_abs_error'])} | "
            f"非稳态 MAE={_fmt(nonsteady_metrics['mae'])}, wMAPE={_fmt(nonsteady_metrics.get('wmape'))}% , 有效MAPE={_fmt(nonsteady_metrics['mape'])}%"
            + (f" | {trim_desc}" if trim_desc else "")
        )

    def show_prediction_results_dialog(self):
        prediction_df, metrics = self.build_current_prediction_dataframe()
        if prediction_df.empty:
            messagebox.showwarning("结果不可用", "当前视图暂无可用的逐点预测结果表")
            return

        actual_label = str(metrics.get("actual_label") or "实际负载")
        case_profile_name = str(metrics.get("case_profile_name") or self.get_current_prediction_case_profile_name())
        dialog = tk.Toplevel(self.root)
        dialog.title(f"{actual_label} vs {metrics['label']}结果表 [{case_profile_name}]")
        dialog.geometry("1480x620")
        dialog.minsize(1180, 460)
        dialog.transient(self.root)
        dialog.grab_set()
        center_dialog_on_parent(dialog, self.root)

        root_frame = ttk.Frame(dialog, padding=8)
        root_frame.pack(fill=tk.BOTH, expand=True)
        root_frame.grid_rowconfigure(1, weight=1)
        root_frame.grid_columnconfigure(0, weight=1)

        summary_frame = ttk.LabelFrame(root_frame, text="误差汇总", padding=8)
        summary_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        def _metric_text(title, metric_values):
            if metric_values.get("count", 0) <= 0:
                return f"{title}: -"
            floor_value = metric_values.get("relative_error_floor")
            floor_text = "-" if floor_value is None or not np.isfinite(floor_value) else f"{float(floor_value):.3f}"
            floor_suffix = "" if floor_text == "-" else f" (>{floor_text}W)"
            return (
                f"{title}: MAE={metric_values['mae']:.3f}, RMSE={metric_values['rmse']:.3f}, "
                f"wMAPE={metric_values.get('wmape', math.nan):.3f}% , 有效MAPE={metric_values['mape']:.3f}%{floor_suffix}, "
                f"MaxAE={metric_values['max_abs_error']:.3f}"
            )

        ttk.Label(summary_frame, text=f"当前案例配置: {case_profile_name}", font=UI_FONT_NORMAL).grid(row=0, column=0, sticky="w")
        ttk.Label(summary_frame, text=_metric_text("稳态(主指标)", metrics["steady"]), font=UI_FONT_NORMAL).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Label(summary_frame, text=_metric_text("切削段(参考主图)", metrics.get("cutting", compute_error_metrics([], []))), font=UI_FONT_NORMAL).grid(row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Label(summary_frame, text=_metric_text("全局(含非稳态/空载，仅参考)", metrics["global"]), font=UI_FONT_NORMAL).grid(row=3, column=0, sticky="w", pady=(4, 0))
        ttk.Label(summary_frame, text=_metric_text("非稳态(仅参考)", metrics["nonsteady"]), font=UI_FONT_NORMAL).grid(row=4, column=0, sticky="w", pady=(4, 0))
        ttk.Label(
            summary_frame,
            text=f"说明: {metrics.get('steady_trim_desc') or '稳态误差按区间中段统计'}；有效MAPE自动忽略低负载点，避免空切/抬刀/近零段把百分比误差异常放大。",
            font=UI_FONT_SMALL,
            foreground=UI_COLOR_TEXT_MUTED,
        ).grid(row=5, column=0, sticky="w", pady=(6, 0))

        table_frame = ttk.Frame(root_frame)
        table_frame.grid(row=1, column=0, sticky="nsew")
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        tree = ttk.Treeview(table_frame, show="headings")
        y_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        x_scroll = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        self._populate_treeview(tree, prediction_df, max_rows=8000)

        button_frame = ttk.Frame(root_frame)
        button_frame.grid(row=2, column=0, sticky="e", pady=(8, 0))

        def _export():
            file_path = filedialog.asksaveasfilename(
                title="导出预测结果表",
                defaultextension=".csv",
                filetypes=(("CSV 文件", "*.csv"), ("所有文件", "*.*")),
            )
            if not file_path:
                return
            export_table_to_csv(prediction_df, file_path)
            self.set_status(f"已导出预测结果表: {os.path.basename(file_path)}", 4000)

        ttk.Button(button_frame, text="导出CSV", command=_export, width=12).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(button_frame, text="关闭", command=dialog.destroy, width=10).pack(side=tk.LEFT)

    def create_data_analysis_tab(self):
        self.data_analysis_tab.grid_columnconfigure(0, weight=1)
        self.data_analysis_tab.grid_rowconfigure(1, weight=1)

        top_bar = ttk.Frame(self.data_analysis_tab, padding=(8, 8, 8, 4))
        top_bar.grid(row=0, column=0, sticky="ew")
        top_bar.grid_columnconfigure(12, weight=1)

        self.analysis_import_btn = ttk.Button(top_bar, text="导入多组CSV", command=self.browse_data_analysis_runs, style="Tech.TButton", width=14)
        self.analysis_import_btn.grid(row=0, column=0, padx=(0, 6))
        self.analysis_remove_btn = ttk.Button(top_bar, text="删除选中", command=self.remove_selected_data_analysis_run, style="Secondary.TButton", width=10)
        self.analysis_remove_btn.grid(row=0, column=1, padx=(0, 6))
        self.analysis_run_btn = ttk.Button(top_bar, text="分析", command=self.run_data_analysis, style="Orange.TButton", width=10)
        self.analysis_run_btn.grid(row=0, column=2, padx=(0, 12))

        ttk.Label(top_bar, text="策略:", font=UI_FONT_NORMAL).grid(row=0, column=3, sticky="w")
        self.analysis_strategy_combo = ttk.Combobox(
            top_bar,
            textvariable=self.analysis_strategy_var,
            state="readonly",
            values=["均衡优先", "效率优先", "折中策略"],
            width=12,
            font=UI_FONT_NORMAL,
        )
        self.analysis_strategy_combo.grid(row=0, column=4, padx=(4, 12))
        self.analysis_strategy_combo.bind("<<ComboboxSelected>>", lambda _e: self.run_data_analysis())
        self.analysis_apply_target_btn = ttk.Button(
            top_bar,
            text="确定该目标负载",
            command=self.apply_analysis_targets_to_main_flow,
            style="Orange.TButton",
            width=16,
        )
        self.analysis_apply_target_btn.grid(row=0, column=5, padx=(0, 12))
        self.analysis_export_btn = ttk.Button(top_bar, text="导出当前表", command=self.export_current_analysis_table, style="Secondary.TButton", width=12)
        self.analysis_export_btn.grid(row=0, column=6, padx=(0, 6))
        ttk.Label(top_bar, textvariable=self.analysis_status_var, font=UI_FONT_SMALL, foreground=UI_COLOR_TEXT_MUTED).grid(row=0, column=12, sticky="e")

        body = ttk.PanedWindow(self.data_analysis_tab, orient=tk.HORIZONTAL)
        body.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        side_panel = ttk.Frame(body)
        main_panel = ttk.Frame(body)
        body.add(side_panel, weight=0)
        body.add(main_panel, weight=1)

        side_panel.grid_columnconfigure(0, weight=1)
        side_panel.grid_rowconfigure(0, weight=1)
        side_panel.grid_rowconfigure(1, weight=0)

        run_frame = ttk.LabelFrame(side_panel, text="Run 列表", padding=6, style="Tech.TLabelframe")
        run_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        run_frame.grid_columnconfigure(0, weight=1)
        run_frame.grid_rowconfigure(0, weight=1)

        run_tree = ttk.Treeview(run_frame, columns=("run_id", "role", "file_name", "f_col", "p_col", "pred_col"), show="headings", height=18)
        for column, title, width in (
            ("run_id", "Run", 70),
            ("role", "角色", 92),
            ("file_name", "文件", 170),
            ("f_col", "F列", 90),
            ("p_col", "P列", 90),
            ("pred_col", "预测列", 90),
        ):
            run_tree.heading(column, text=title)
            run_tree.column(column, width=width, anchor="center", stretch=True)
        run_scroll = ttk.Scrollbar(run_frame, orient=tk.VERTICAL, command=run_tree.yview)
        run_tree.configure(yscrollcommand=run_scroll.set)
        run_tree.grid(row=0, column=0, sticky="nsew")
        run_scroll.grid(row=0, column=1, sticky="ns")
        run_tree.bind("<<TreeviewSelect>>", self.on_data_analysis_run_selected)
        run_tree.bind("<ButtonRelease-1>", self.on_data_analysis_run_tree_click)
        run_tree.bind("<Configure>", lambda _e: self._hide_analysis_role_editor())
        self.analysis_run_tree = run_tree

        mapping_frame = ttk.LabelFrame(side_panel, text="列映射", padding=6, style="Tech.TLabelframe")
        mapping_frame.grid(row=1, column=0, sticky="ew")
        mapping_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(mapping_frame, text="F列:", font=UI_FONT_NORMAL).grid(row=0, column=0, sticky="w", padx=(0, 6), pady=(0, 4))
        self.analysis_f_combo = ttk.Combobox(mapping_frame, state="readonly", width=18, font=UI_FONT_NORMAL)
        self.analysis_f_combo.grid(row=0, column=1, sticky="ew", pady=(0, 4))
        ttk.Label(mapping_frame, text="P列:", font=UI_FONT_NORMAL).grid(row=1, column=0, sticky="w", padx=(0, 6), pady=(0, 4))
        self.analysis_p_combo = ttk.Combobox(mapping_frame, state="readonly", width=18, font=UI_FONT_NORMAL)
        self.analysis_p_combo.grid(row=1, column=1, sticky="ew", pady=(0, 4))
        ttk.Label(mapping_frame, text="预测列:", font=UI_FONT_NORMAL).grid(row=2, column=0, sticky="w", padx=(0, 6), pady=(0, 6))
        self.analysis_pred_combo = ttk.Combobox(mapping_frame, state="readonly", width=18, font=UI_FONT_NORMAL)
        self.analysis_pred_combo.grid(row=2, column=1, sticky="ew", pady=(0, 6))
        self.analysis_apply_mapping_btn = ttk.Button(mapping_frame, text="应用映射", command=self.apply_selected_data_analysis_mapping, style="Secondary.TButton", width=12)
        self.analysis_apply_mapping_btn.grid(row=3, column=1, sticky="e")

        main_panel.grid_columnconfigure(0, weight=1)
        main_panel.grid_rowconfigure(1, weight=1)
        main_panel.grid_rowconfigure(2, weight=0)

        chart_bar = ttk.Frame(main_panel, padding=(0, 0, 0, 4))
        chart_bar.grid(row=0, column=0, sticky="ew")
        chart_bar.grid_columnconfigure(9, weight=1)

        ttk.Label(chart_bar, text="主图:", font=UI_FONT_NORMAL).grid(row=0, column=0, sticky="w")
        self.analysis_chart_combo = ttk.Combobox(
            chart_bar,
            textvariable=self.analysis_chart_var,
            state="readonly",
            values=["负载图", "区间平均负载对比图", "区间波动图", "区间调节强度图", "区间响应时序图", "区间漂移趋势图"],
            width=20,
            font=UI_FONT_NORMAL,
        )
        self.analysis_chart_combo.grid(row=0, column=1, padx=(4, 10))
        self.analysis_chart_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_data_analysis_view())

        ttk.Label(chart_bar, text="run:", font=UI_FONT_NORMAL).grid(row=0, column=2, sticky="w")
        self.analysis_run_combo = ttk.Combobox(chart_bar, textvariable=self.analysis_run_var, state="readonly", width=18, font=UI_FONT_NORMAL)
        self.analysis_run_combo.grid(row=0, column=3, padx=(4, 10))
        self.analysis_run_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_data_analysis_view())

        ttk.Label(chart_bar, text="区间:", font=UI_FONT_NORMAL).grid(row=0, column=4, sticky="w")
        self.analysis_interval_combo = ttk.Combobox(chart_bar, textvariable=self.analysis_interval_var, state="readonly", width=24, font=UI_FONT_NORMAL)
        self.analysis_interval_combo.grid(row=0, column=5, padx=(4, 10))
        self.analysis_interval_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_data_analysis_view())

        ttk.Label(chart_bar, text="指标:", font=UI_FONT_NORMAL).grid(row=0, column=6, sticky="w")
        self.analysis_metric_combo = ttk.Combobox(chart_bar, textvariable=self.analysis_metric_var, state="readonly", width=18, font=UI_FONT_NORMAL)
        self.analysis_metric_combo["values"] = ["avg_actual_load", "avg_F", "P_std", "P_peak_to_peak", "residual_rms", "delta_F", "delta_P"]
        self.analysis_metric_combo.grid(row=0, column=7, padx=(4, 10))
        self.analysis_metric_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_data_analysis_view())

        plot_frame = ttk.LabelFrame(main_panel, text="负载图 / 分析图", padding=6, style="Tech.TLabelframe")
        plot_frame.grid(row=1, column=0, sticky="nsew")
        plot_frame.grid_columnconfigure(0, weight=1)
        plot_frame.grid_rowconfigure(0, weight=1)
        self.analysis_plot_frame = plot_frame

        bottom_notebook = ttk.Notebook(main_panel)
        bottom_notebook.grid(row=2, column=0, sticky="nsew", pady=(6, 0))

        table_tab = ttk.Frame(bottom_notebook)
        summary_tab = ttk.Frame(bottom_notebook)
        bottom_notebook.add(table_tab, text="统计表")
        bottom_notebook.add(summary_tab, text="分析摘要")

        table_tab.grid_columnconfigure(0, weight=1)
        table_tab.grid_rowconfigure(0, weight=1)
        tree = ttk.Treeview(table_tab, show="headings")
        y_scroll = ttk.Scrollbar(table_tab, orient=tk.VERTICAL, command=tree.yview)
        x_scroll = ttk.Scrollbar(table_tab, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.analysis_table_tree = tree

        summary_tab.grid_columnconfigure(0, weight=1)
        summary_tab.grid_rowconfigure(0, weight=1)
        self.analysis_summary_text = tk.Text(summary_tab, wrap=tk.WORD, font=UI_FONT_NORMAL, relief=tk.FLAT, bg="white")
        self.analysis_summary_text.grid(row=0, column=0, sticky="nsew")
        self.analysis_summary_text.insert("1.0", "暂无分析摘要")
        self.analysis_summary_text.configure(state="disabled")

    def _normalize_analysis_run_role_label(self, role):
        role_text = str(role)
        if role_text == "baseline":
            return "首次加工"
        if role_text.startswith("optimization_"):
            try:
                return f"第{int(role_text.split('_')[-1])}次优化"
            except Exception:
                return role_text
        return role_text

    def _analysis_role_value_options(self, include_current=""):
        max_opt_idx = max(5, max(1, len(self.analysis_runs) - 1))
        for candidate in [include_current] + [run.get("role", "") for run in self.analysis_runs]:
            normalized = normalize_role(candidate)
            if normalized.startswith("optimization_"):
                try:
                    max_opt_idx = max(max_opt_idx, int(normalized.split("_")[-1]))
                except Exception:
                    continue
        options = ["baseline"] + [f"optimization_{idx}" for idx in range(1, max_opt_idx + 1)]
        current_role = normalize_role(include_current)
        if current_role and current_role not in options:
            options.append(current_role)
            options = sorted(set(options), key=role_sort_key)
        return options

    def _analysis_role_display_options(self, include_current=""):
        return [self._normalize_analysis_run_role_label(role) for role in self._analysis_role_value_options(include_current)]

    def _analysis_role_display_to_value(self, display_text, current_role=""):
        mapping = {
            self._normalize_analysis_run_role_label(role): role
            for role in self._analysis_role_value_options(current_role)
        }
        display = str(display_text or "").strip()
        if display in mapping:
            return mapping[display]
        normalized = normalize_role(display)
        return normalized or normalize_role(current_role) or "run"

    def _find_analysis_run(self, run_id):
        run_key = str(run_id or "").strip()
        for idx, run in enumerate(self.analysis_runs):
            if str(run.get("run_id", "")).strip() == run_key:
                return idx, run
        return None, None

    def _rebuild_analysis_run_frame(self, run):
        run["role"] = normalize_role(run.get("role"))
        run["frame"] = prepare_run_frame(
            run["raw_df"],
            run_id=run["run_id"],
            role=run["role"],
            file_name=run["file_name"],
            f_column=run["f_column"],
            p_column=run["p_column"],
            predicted_column=run["predicted_column"] or None,
            feed_column=run["f_column"],
        )
        return run["frame"]

    def _select_analysis_run_tree_item(self, run_id):
        if not hasattr(self, "analysis_run_tree"):
            return
        run_key = str(run_id or "").strip()
        if not run_key:
            return
        tree = self.analysis_run_tree
        try:
            tree.selection_set(run_key)
            tree.focus(run_key)
            tree.see(run_key)
        except Exception:
            return

    def _hide_analysis_role_editor(self):
        editor = getattr(self, "_analysis_role_editor", None)
        if editor is not None:
            try:
                editor.place_forget()
            except Exception:
                pass
        self._analysis_role_edit_item = ""

    def _ensure_analysis_role_editor(self):
        editor = getattr(self, "_analysis_role_editor", None)
        if editor is not None:
            try:
                if editor.winfo_exists():
                    return editor
            except Exception:
                pass
        editor = ttk.Combobox(self.analysis_run_tree, state="readonly", width=12, font=UI_FONT_NORMAL)
        editor.bind("<<ComboboxSelected>>", self._commit_analysis_role_edit)
        editor.bind("<Return>", self._commit_analysis_role_edit)
        editor.bind("<Escape>", lambda _e: self._hide_analysis_role_editor())
        self._analysis_role_editor = editor
        return editor

    def _begin_analysis_role_edit(self, item_id):
        tree = getattr(self, "analysis_run_tree", None)
        if tree is None:
            return
        run_id = str(item_id or "").strip()
        _, run = self._find_analysis_run(run_id)
        if run is None:
            self._hide_analysis_role_editor()
            return
        bbox = tree.bbox(run_id, "#2")
        if not bbox:
            self._hide_analysis_role_editor()
            return
        editor = self._ensure_analysis_role_editor()
        editor["values"] = self._analysis_role_display_options(run.get("role"))
        editor.set(self._normalize_analysis_run_role_label(run.get("role")))
        editor.place(x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3])
        editor.lift()
        editor.focus_set()
        try:
            def _post_role_dropdown():
                try:
                    editor.tk.call("ttk::combobox::Post", str(editor))
                    return
                except Exception:
                    pass
                try:
                    editor.event_generate("<Down>")
                except Exception:
                    pass
            self.root.after_idle(_post_role_dropdown)
        except Exception:
            pass
        self._analysis_role_edit_item = run_id

    def _commit_analysis_role_edit(self, _event=None):
        run_id = str(getattr(self, "_analysis_role_edit_item", "") or "").strip()
        editor = getattr(self, "_analysis_role_editor", None)
        if not run_id or editor is None:
            self._hide_analysis_role_editor()
            return
        _, run = self._find_analysis_run(run_id)
        current_role = run.get("role") if isinstance(run, dict) else ""
        role_value = self._analysis_role_display_to_value(editor.get(), current_role)
        self._hide_analysis_role_editor()
        if role_value:
            self._update_analysis_run_role(run_id, role_value)

    def _update_analysis_run_role(self, run_id, role_value):
        _, run = self._find_analysis_run(run_id)
        if run is None:
            return
        normalized_role = normalize_role(role_value)
        if str(run.get("role")) == normalized_role:
            self._select_analysis_run_tree_item(run_id)
            return
        run["role"] = normalized_role
        self._rebuild_analysis_run_frame(run)
        self.analysis_result = None
        self._refresh_analysis_run_tree(selected_run_id=run_id)
        self.refresh_data_analysis_view()
        self.analysis_status_var.set(
            f"已将 {run['file_name']} 设置为“{self._normalize_analysis_run_role_label(normalized_role)}”，请点击“分析”刷新结果"
        )

    def on_data_analysis_run_tree_click(self, event=None):
        tree = getattr(self, "analysis_run_tree", None)
        if tree is None or event is None:
            return
        item_id = tree.identify_row(event.y)
        column_id = tree.identify_column(event.x)
        if item_id:
            self._select_analysis_run_tree_item(item_id)
        if item_id and column_id == "#2":
            try:
                self.root.after_idle(lambda run_id=str(item_id): self._begin_analysis_role_edit(run_id))
            except Exception:
                self._begin_analysis_role_edit(item_id)
        else:
            self._hide_analysis_role_editor()

    def _selected_analysis_run_index(self):
        if not hasattr(self, "analysis_run_tree"):
            return None
        selection = self.analysis_run_tree.selection()
        if not selection:
            return None
        run_id = str(selection[0]).strip()
        idx, _ = self._find_analysis_run(run_id)
        return idx

    def _load_data_analysis_run_payload(self, file_path):
        raw_df = self._read_data_analysis_run_csv(file_path)
        f_column, p_column, predicted_column = self._infer_data_analysis_columns(raw_df)
        if not f_column or not p_column:
            raise ValueError("未能自动识别 F / P 列，请检查 CSV 列头")
        return {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "raw_df": raw_df,
            "f_column": f_column,
            "p_column": p_column,
            "predicted_column": predicted_column or "",
        }

    def _append_data_analysis_run_payload(self, payload):
        run_index = len(self.analysis_runs) + 1
        role = "baseline" if not any(run["role"] == "baseline" for run in self.analysis_runs) else f"optimization_{max(1, run_index - 1)}"
        run_id = f"run_{run_index:03d}"
        frame = prepare_run_frame(
            payload["raw_df"],
            run_id=run_id,
            role=role,
            file_name=payload["file_name"],
            f_column=payload["f_column"],
            p_column=payload["p_column"],
            predicted_column=payload["predicted_column"] or None,
            feed_column=payload["f_column"],
        )
        self.analysis_runs.append({
            "run_id": run_id,
            "file_path": payload["file_path"],
            "file_name": payload["file_name"],
            "raw_df": payload["raw_df"],
            "frame": frame,
            "role": role,
            "f_column": payload["f_column"],
            "p_column": payload["p_column"],
            "predicted_column": payload["predicted_column"] or "",
        })

    def _run_data_analysis_import_worker(self, job_id, file_paths):
        payloads = []
        errors = []
        max_workers = max(1, min(4, len(file_paths)))
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(self._load_data_analysis_run_payload, file_path): (idx, file_path)
                    for idx, file_path in enumerate(file_paths)
                }
                completed = []
                for future in concurrent.futures.as_completed(future_map):
                    idx, file_path = future_map[future]
                    try:
                        completed.append((idx, future.result()))
                    except Exception as exc:
                        errors.append((file_path, str(exc)))
                payloads = [item for _, item in sorted(completed, key=lambda entry: entry[0])]
        except Exception as exc:
            errors.append(("", str(exc)))

        try:
            self.root.after(0, lambda: self._finish_data_analysis_import(job_id, payloads, errors))
        except Exception:
            pass

    def _finish_data_analysis_import(self, job_id, payloads, errors):
        if job_id != self._analysis_import_job_id:
            return
        self.analysis_import_in_progress = False
        for payload in payloads:
            self._append_data_analysis_run_payload(payload)
        self.analysis_result = None
        self._refresh_analysis_run_tree()
        if self.analysis_runs:
            self._select_analysis_run_tree_item(self.analysis_runs[0]["run_id"])
        if self.analysis_runs:
            self.analysis_status_var.set(f"已导入 {len(self.analysis_runs)} 个 run，可直接在列表中设置角色并调整列映射后分析")
        else:
            self.analysis_status_var.set("未导入任何有效 run")
        self._set_data_analysis_busy_state()
        self.refresh_data_analysis_view()
        if errors:
            error_text = "\n".join(
                f"{os.path.basename(path) or '后台任务'}: {message}"
                for path, message in errors[:6]
            )
            messagebox.showwarning("部分导入失败", error_text)

    def _resolve_analysis_intervals_from_main_flow(self):
        return self._get_current_interval_records(allow_profile_fallback=False)

    def _format_analysis_interval_label(self, row):
        interval_id = str(row.get("interval_id", "") or "")
        start_label = str(row.get("start_label", "") or row.get("start_line", ""))
        end_label = str(row.get("end_label", "") or row.get("end_line", ""))
        if start_label and end_label:
            return f"{interval_id} | {start_label} -> {end_label}"
        return interval_id

    def _build_analysis_target_curve(self, result):
        prediction_df, metrics = self.build_current_prediction_dataframe()
        if prediction_df.empty or not isinstance(result, dict):
            return pd.DataFrame()

        prediction_label = str(metrics.get("label") or self.get_prediction_curve_label())
        actual_label = "actual_load"
        target_df = result.get("target_table")
        if not isinstance(target_df, pd.DataFrame):
            return pd.DataFrame()

        load_df = prediction_df.copy().reset_index(drop=True)
        idle_values = pd.to_numeric(load_df.get("predicted_idle_power"), errors="coerce").to_numpy(dtype=float)
        target_curve = idle_values.copy()
        target_curve[~np.isfinite(target_curve)] = 0.0
        segment_type = np.full(len(load_df), "background", dtype=object)
        sample_index_values = pd.to_numeric(load_df.get("sample_index"), errors="coerce").to_numpy(dtype=float)
        if sample_index_values.size != len(load_df) or not np.any(np.isfinite(sample_index_values)):
            sample_index_values = np.arange(len(load_df), dtype=float)

        target_lookup = {
            str(row.interval_id): {
                "target_load": float(row.target_load),
                "steady_subtype": str(getattr(row, "steady_subtype", "cutting") or "cutting"),
            }
            for row in target_df.itertuples(index=False)
            if str(getattr(row, "segment_type", "steady")) == "steady" and np.isfinite(float(getattr(row, "target_load", math.nan)))
        }

        interval_records = self._resolve_analysis_intervals_from_main_flow()
        sorted_intervals = []
        for record in interval_records:
            sample_bounds = self._get_interval_sample_index_span(record)
            if not sample_bounds:
                continue
            start_idx, end_idx = sample_bounds
            if end_idx < start_idx:
                continue
            visible_mask = np.isfinite(sample_index_values) & (sample_index_values >= start_idx) & (sample_index_values <= end_idx)
            visible_positions = np.flatnonzero(visible_mask)
            if visible_positions.size == 0:
                continue
            sorted_intervals.append({
                "sample_start_idx": start_idx,
                "sample_end_idx": end_idx,
                "start_pos": int(visible_positions[0]),
                "end_pos": int(visible_positions[-1]),
                "interval_id": str(record.get("zone_id") or record.get("interval_id") or ""),
            })
        sorted_intervals.sort(key=lambda item: item["start_pos"])

        for interval_info in sorted_intervals:
            start_pos = int(interval_info["start_pos"])
            end_pos = int(interval_info["end_pos"])
            interval_id = str(interval_info["interval_id"])
            target_item = target_lookup.get(interval_id)
            if not interval_id or not isinstance(target_item, dict):
                continue
            safe_start = max(0, start_pos)
            safe_end = min(len(load_df) - 1, end_pos)
            if safe_end < safe_start:
                continue
            target_curve[safe_start:safe_end + 1] = float(target_item["target_load"])
            segment_type[safe_start:safe_end + 1] = "steady_idle" if str(target_item.get("steady_subtype")) == "idle" else "steady_cutting"

        prev_end = -1
        nonsteady_segments = []
        if prediction_label and prediction_label in load_df.columns:
            prediction_values = pd.to_numeric(load_df.get(prediction_label), errors="coerce").to_numpy(dtype=float)
        else:
            prediction_values = np.full(len(load_df), np.nan, dtype=float)
        for order_idx, interval_info in enumerate(sorted_intervals):
            start_pos = int(interval_info["start_pos"])
            end_pos = int(interval_info["end_pos"])
            interval_id = str(interval_info["interval_id"])
            gap_start = prev_end + 1
            gap_end = start_pos - 1
            if gap_end >= gap_start:
                gap_len = gap_end - gap_start + 1
                gap_idle = idle_values[gap_start:gap_end + 1]
                gap_pred = prediction_values[gap_start:gap_end + 1]
                gap_pred_mean = float(np.nanmean(gap_pred)) if np.any(np.isfinite(gap_pred)) else 0.0
                gap_idle_mean = float(np.nanmean(gap_idle)) if np.any(np.isfinite(gap_idle)) else 0.0
                if abs(gap_pred_mean - gap_idle_mean) > max(abs(gap_idle_mean) * 0.10, 5.0):
                    left_interval_id = str(sorted_intervals[order_idx - 1]["interval_id"]) if order_idx > 0 else ""
                    left_target_item = target_lookup.get(left_interval_id) if left_interval_id else None
                    right_target_item = target_lookup.get(interval_id)
                    left_target = left_target_item.get("target_load") if isinstance(left_target_item, dict) else gap_pred_mean
                    right_target = right_target_item.get("target_load") if isinstance(right_target_item, dict) else gap_pred_mean
                    left_target = gap_pred_mean if left_target is None or not np.isfinite(left_target) else float(left_target)
                    right_target = gap_pred_mean if right_target is None or not np.isfinite(right_target) else float(right_target)
                    if gap_len < 100:
                        split_points = [(gap_start, gap_end, float((left_target + right_target) / 2.0))]
                    elif gap_len < 1000:
                        mid = gap_start + gap_len // 2
                        split_points = [
                            (gap_start, mid - 1, left_target),
                            (mid, gap_end, right_target),
                        ]
                    else:
                        third = max(1, gap_len // 3)
                        split_points = [
                            (gap_start, gap_start + third - 1, left_target),
                            (gap_start + third, gap_end - third, float((left_target + right_target) / 2.0)),
                            (gap_end - third + 1, gap_end, right_target),
                        ]
                    for seg_start, seg_end, seg_value in split_points:
                        safe_start = max(0, seg_start)
                        safe_end = min(len(load_df) - 1, seg_end)
                        if safe_end < safe_start:
                            continue
                        target_curve[safe_start:safe_end + 1] = seg_value
                        segment_type[safe_start:safe_end + 1] = "nonsteady"
                        nonsteady_segments.append({
                            "start_idx": safe_start,
                            "end_idx": safe_end,
                            "target_value": float(seg_value),
                            "segment_type": "nonsteady",
                        })
            prev_end = end_pos

        tail_start = prev_end + 1
        tail_end = len(load_df) - 1
        if tail_end >= tail_start:
            tail_idle = idle_values[tail_start:tail_end + 1]
            tail_pred = prediction_values[tail_start:tail_end + 1]
            tail_pred_mean = float(np.nanmean(tail_pred)) if np.any(np.isfinite(tail_pred)) else 0.0
            tail_idle_mean = float(np.nanmean(tail_idle)) if np.any(np.isfinite(tail_idle)) else 0.0
            if abs(tail_pred_mean - tail_idle_mean) > max(abs(tail_idle_mean) * 0.10, 5.0):
                tail_target = tail_pred_mean
                target_curve[tail_start:tail_end + 1] = tail_target
                segment_type[tail_start:tail_end + 1] = "nonsteady"
                nonsteady_segments.append({
                    "start_idx": tail_start,
                    "end_idx": tail_end,
                    "target_value": float(tail_target),
                    "segment_type": "nonsteady",
                })

        load_df["target_load"] = target_curve
        load_df["target_segment_type"] = segment_type
        result["prediction_label"] = prediction_label
        result["actual_label"] = actual_label
        result["prediction_df"] = prediction_df
        result["load_chart_df"] = load_df
        result["target_load_curve"] = target_curve.tolist()
        result["nonsteady_target_segments"] = nonsteady_segments
        return load_df

    def apply_analysis_targets_to_main_flow(self):
        if not isinstance(self.analysis_result, dict) or "target_table" not in self.analysis_result:
            messagebox.showwarning("无目标值", "请先完成数据分析并生成目标值")
            return
        target_df = self.analysis_result.get("target_table")
        if not isinstance(target_df, pd.DataFrame) or target_df.empty:
            messagebox.showwarning("无目标值", "当前没有可写回的目标值")
            return

        target_lookup = {
            str(row.interval_id): float(row.target_load)
            for row in target_df.itertuples(index=False)
            if np.isfinite(float(getattr(row, "target_load", math.nan)))
        }
        updated_intervals = self._get_current_interval_records(allow_profile_fallback=False)
        for record in updated_intervals:
            interval_id = str(record.get("zone_id") or record.get("interval_id") or "")
            if interval_id in target_lookup:
                record["target_load"] = float(target_lookup[interval_id])
                record["strategy_mode"] = str(self.analysis_strategy_var.get())

        self.target_load_curve = list(self.analysis_result.get("target_load_curve") or [])
        self._set_current_interval_state(
            interval_records=updated_intervals,
            segment_records=self._get_current_segment_records(allow_profile_fallback=False),
            point_kc_map=dict(getattr(self, "current_interval_point_kc_map", {}) or {}),
            source=str(getattr(self, "_current_interval_source", "") or ""),
            profile_locked=bool(getattr(self, "_profile_intervals_locked", False)),
        )
        if not isinstance(getattr(self, "_cached_steady_intervals", None), dict):
            self._cached_steady_intervals = {}
        self._cached_steady_intervals["pit_records"] = [dict(record) for record in updated_intervals]
        self._cached_steady_intervals["target_load_curve"] = list(self.target_load_curve)
        self._cached_steady_intervals["nonsteady_target_segments"] = list(self.analysis_result.get("nonsteady_target_segments") or [])
        imported_forward_lock = bool(
            hasattr(self, "_is_imported_profile_forward_lock_active")
            and self._is_imported_profile_forward_lock_active()
        )
        if isinstance(getattr(self, "active_kc_profile", None), dict) and not imported_forward_lock:
            self.active_kc_profile["pit_records"] = [dict(record) for record in updated_intervals]
            self.active_kc_profile["target_load_curve"] = list(self.target_load_curve)

        try:
            self.refresh_main_pit_preview()
        except Exception:
            pass
        try:
            interval_policy = self._get_default_interval_policy() if hasattr(self, "_get_default_interval_policy") else "fresh_or_empty"
            self.generate_plots(silent=True, interval_policy=interval_policy)
        except Exception:
            pass
        self.analysis_status_var.set("目标值已写回工艺信息页面，负载图已同步刷新")
        self.refresh_data_analysis_view()

    def _run_data_analysis_worker(self, job_id, frames, intervals, strategy_mode):
        result = None
        error_text = ""
        try:
            result = build_multi_run_analysis(
                frames,
                intervals=intervals,
                strategy_mode=strategy_mode,
            )
        except Exception as exc:
            error_text = str(exc)

        try:
            self.root.after(0, lambda: self._finish_data_analysis_run(job_id, result, error_text))
        except Exception:
            pass

    def _finish_data_analysis_run(self, job_id, result, error_text):
        if job_id != self._analysis_run_job_id:
            return
        self.analysis_run_in_progress = False
        self._set_data_analysis_busy_state()
        if error_text:
            messagebox.showerror("分析失败", f"数据分析表运行失败:\n{error_text}")
            return
        self._build_analysis_target_curve(result)
        self.analysis_result = result
        self._sync_analysis_selector_values()
        self.analysis_status_var.set(
            f"数据分析完成: {len(self.analysis_result['runs'])} 个 run，复用工艺页区间 {len(self.analysis_result['intervals'])} 段，策略={self.analysis_result.get('strategy_mode', self.analysis_strategy_var.get())}"
        )
        self.refresh_data_analysis_view()

    def browse_data_analysis_runs(self):
        if self.analysis_import_in_progress or self.analysis_run_in_progress:
            messagebox.showinfo("任务进行中", "数据分析页当前有后台任务，请等待完成后再继续。")
            return
        file_paths = filedialog.askopenfilenames(
            title="选择多组 run CSV",
            filetypes=(("CSV 文件", "*.csv"), ("文本文件", "*.txt"), ("所有文件", "*.*")),
        )
        if not file_paths:
            return
        self.analysis_import_in_progress = True
        self._analysis_import_job_id += 1
        self.analysis_status_var.set(f"正在后台导入 {len(file_paths)} 个 run...")
        self._set_data_analysis_busy_state()
        worker = threading.Thread(
            target=self._run_data_analysis_import_worker,
            args=(self._analysis_import_job_id, list(file_paths)),
            daemon=True,
        )
        worker.start()

    def _read_data_analysis_run_csv(self, file_path):
        """优先复用实测文件解析方式，再回退到普通 CSV / SampleData 读取。"""
        last_error = None

        try:
            measurement = self.parse_channel_data_file(file_path)
            actual_load = np.asarray(measurement["actual_load"], dtype=float)
            actual_feed_speed = np.asarray(measurement["actual_feed_speed"], dtype=float)
            actual_spindle_speed = np.asarray(measurement["actual_spindle_speed"], dtype=float)
            program_line = np.asarray(measurement["program_line"], dtype=int)
            if (
                len(actual_load) >= 3
                and len(actual_load) == len(actual_feed_speed)
                and len(actual_load) == len(actual_spindle_speed)
                and len(actual_load) == len(program_line)
            ):
                return pd.DataFrame({
                    "F": actual_feed_speed,
                    "P": actual_load,
                    "actual_spindle_speed": actual_spindle_speed,
                    "program_line": program_line,
                })
            last_error = ValueError("按实验实测文件方式读取后有效数据不足")
        except Exception as exc:
            last_error = exc

        try:
            raw_df = self._read_csv_flex(file_path)
            f_column = infer_matching_column(raw_df, F_COLUMN_ALIASES, fallback_index=0)
            p_column = infer_matching_column(raw_df, P_COLUMN_ALIASES, fallback_index=1)
            if f_column and p_column:
                return raw_df
            last_error = ValueError("普通 CSV 读取成功，但未识别到 F / P 列")
        except Exception as exc:
            last_error = exc

        try:
            df = pd.read_csv(file_path, header=None, usecols=[0, 1, 2, 3, 4], dtype={4: str})
            if df.shape[1] >= 5:
                values = df.iloc[:, 0:3].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                line_numbers = pd.to_numeric(df.iloc[:, 3], errors="coerce").to_numpy(dtype=float)
                program_numbers = df.iloc[:, 4].astype(str).to_numpy()
                valid_mask = ~np.isnan(line_numbers)
                values = values[valid_mask]
                program_numbers = program_numbers[valid_mask]
                line_numbers = line_numbers[valid_mask].astype(int)

                # SampleData 的第4列应是高离散度的程序行号，避免误把 process_info 之类的 5 列文件识别进来。
                unique_line_count = len(np.unique(line_numbers)) if len(line_numbers) > 0 else 0
                if len(values) >= 3 and unique_line_count >= 10:
                    raw_df = pd.DataFrame(values, columns=["col_0", "col_1", "col_2"])
                    raw_df["line_number"] = line_numbers
                    raw_df["program_number"] = program_numbers
                    return raw_df
                last_error = ValueError("按 SampleData 格式读取后未识别到足够有效的程序行号")
            else:
                last_error = ValueError("按 SampleData 格式读取后列数不足 5 列")
        except Exception as exc:
            last_error = exc

        if last_error is not None:
            raise last_error
        raise ValueError("无法读取 run CSV")

    def _infer_data_analysis_columns(self, raw_df):
        synthetic_columns = all(str(col).startswith("col_") for col in raw_df.columns)
        if synthetic_columns:
            f_column = infer_matching_column(raw_df, F_COLUMN_ALIASES, fallback_index=0)
            p_column = infer_matching_column(raw_df, P_COLUMN_ALIASES, fallback_index=1)
        else:
            f_column = infer_matching_column(raw_df, F_COLUMN_ALIASES)
            p_column = infer_matching_column(raw_df, P_COLUMN_ALIASES)
        predicted_column = infer_matching_column(raw_df, PRED_COLUMN_ALIASES)
        return f_column, p_column, predicted_column

    def _load_data_analysis_run(self, file_path):
        raw_df = self._read_data_analysis_run_csv(file_path)
        run_index = len(self.analysis_runs) + 1
        role = "baseline" if not any(run["role"] == "baseline" for run in self.analysis_runs) else f"optimization_{max(1, run_index - 1)}"
        f_column, p_column, predicted_column = self._infer_data_analysis_columns(raw_df)
        if not f_column or not p_column:
            raise ValueError("未能自动识别 F / P 列，请检查 CSV 列头")
        run_id = f"run_{run_index:03d}"
        frame = prepare_run_frame(
            raw_df,
            run_id=run_id,
            role=role,
            file_name=os.path.basename(file_path),
            f_column=f_column,
            p_column=p_column,
            predicted_column=predicted_column,
            feed_column=f_column,
        )
        self.analysis_runs.append({
            "run_id": run_id,
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "raw_df": raw_df,
            "frame": frame,
            "role": role,
            "f_column": f_column,
            "p_column": p_column,
            "predicted_column": predicted_column or "",
        })

    def _refresh_analysis_run_tree(self, selected_run_id=None):
        if not hasattr(self, "analysis_run_tree"):
            return
        self._hide_analysis_role_editor()
        tree = self.analysis_run_tree
        current_selection = selected_run_id
        if not current_selection:
            selection = tree.selection()
            current_selection = str(selection[0]).strip() if selection else ""
        tree.delete(*tree.get_children())
        for run in sorted(self.analysis_runs, key=lambda item: role_sort_key(item["role"])):
            tree.insert(
                "",
                tk.END,
                iid=run["run_id"],
                values=(
                    run["run_id"],
                    self._normalize_analysis_run_role_label(run["role"]),
                    run["file_name"],
                    run["f_column"],
                    run["p_column"],
                    run["predicted_column"] or "-",
                ),
            )
        if current_selection and current_selection in tree.get_children():
            self._select_analysis_run_tree_item(current_selection)
        elif self.analysis_runs:
            self._select_analysis_run_tree_item(self.analysis_runs[0]["run_id"])

    def remove_selected_data_analysis_run(self):
        idx = self._selected_analysis_run_index()
        if idx is None:
            return
        del self.analysis_runs[idx]
        self.analysis_result = None
        self._refresh_analysis_run_tree()
        self.refresh_data_analysis_view()

    def on_data_analysis_run_selected(self, event=None):
        self._hide_analysis_role_editor()
        idx = self._selected_analysis_run_index()
        if idx is None:
            return
        run = self.analysis_runs[idx]
        columns = [str(col) for col in run["raw_df"].columns]
        self.analysis_f_combo["values"] = columns
        self.analysis_p_combo["values"] = columns
        self.analysis_pred_combo["values"] = [""] + columns
        self.analysis_f_combo.set(run["f_column"])
        self.analysis_p_combo.set(run["p_column"])
        self.analysis_pred_combo.set(run["predicted_column"] or "")

    def apply_selected_data_analysis_mapping(self):
        idx = self._selected_analysis_run_index()
        if idx is None:
            return
        run = self.analysis_runs[idx]
        f_column = str(self.analysis_f_combo.get()).strip() or run["f_column"]
        p_column = str(self.analysis_p_combo.get()).strip() or run["p_column"]
        predicted_column = str(self.analysis_pred_combo.get()).strip()
        if not f_column or not p_column:
            messagebox.showwarning("映射不完整", "F列和P列不能为空")
            return
        run.update({
            "f_column": f_column,
            "p_column": p_column,
            "predicted_column": predicted_column,
        })
        self._rebuild_analysis_run_frame(run)
        self.analysis_result = None
        self._refresh_analysis_run_tree(selected_run_id=run["run_id"])
        self.refresh_data_analysis_view()
        self.analysis_status_var.set(f"已更新 {run['file_name']} 的列映射，请点击“分析”刷新结果")

    def run_data_analysis(self):
        if len(self.analysis_runs) < 2:
            messagebox.showwarning("run 不足", "至少需要导入首次加工和第一次优化两个 run")
            return
        normalized_roles = [normalize_role(run.get("role")) for run in self.analysis_runs]
        if "baseline" not in normalized_roles or not any(role.startswith("optimization") for role in normalized_roles):
            messagebox.showwarning("角色不完整", "至少需要在列表中指定 1 个“首次加工”和 1 个“优化 run”")
            return
        intervals = self._resolve_analysis_intervals_from_main_flow()
        if not intervals:
            messagebox.showwarning("无稳态区间", "请先在工艺信息页面生成稳态区间，再进行数据分析")
            return
        if self.analysis_import_in_progress or self.analysis_run_in_progress:
            messagebox.showinfo("任务进行中", "数据分析页当前有后台任务，请等待完成后再继续。")
            return
        self.analysis_run_in_progress = True
        self._analysis_run_job_id += 1
        self.analysis_status_var.set(f"正在后台分析 {len(self.analysis_runs)} 个 run，区间来源=工艺页稳态区间，策略={self.analysis_strategy_var.get()}...")
        self._set_data_analysis_busy_state()
        worker = threading.Thread(
            target=self._run_data_analysis_worker,
            args=(
                self._analysis_run_job_id,
                [run["frame"].copy() for run in self.analysis_runs],
                intervals,
                str(self.analysis_strategy_var.get()),
            ),
            daemon=True,
        )
        worker.start()

    def _sync_analysis_selector_values(self):
        if not self.analysis_result:
            return
        run_items = []
        for order_idx, frame in enumerate(self.analysis_result["runs"], 1):
            run_id = str(frame["run_id"].iloc[0])
            role = str(frame["role"].iloc[0])
            file_name = str(frame["file_name"].iloc[0])
            run_items.append(f"{run_id} | {build_run_label(role, file_name, order_idx)}")
        interval_items = [
            self._format_analysis_interval_label(row)
            for row in self.analysis_result["intervals"].to_dict("records")
        ] if not self.analysis_result["intervals"].empty else []
        self.analysis_run_combo["values"] = run_items
        self.analysis_interval_combo["values"] = interval_items
        if run_items:
            current_run = str(self.analysis_run_var.get()).strip()
            if current_run not in run_items:
                self.analysis_run_var.set(run_items[0])
        else:
            self.analysis_run_var.set("")
        if interval_items:
            current_interval = str(self.analysis_interval_var.get()).strip()
            if current_interval not in interval_items:
                self.analysis_interval_var.set(interval_items[0])
        else:
            self.analysis_interval_var.set("")

    def _apply_analysis_time_line_axis(self, ax, x_values, line_values, max_ticks=36):
        x_arr = np.asarray(x_values, dtype=float)
        line_arr = pd.to_numeric(pd.Series(line_values), errors="coerce").to_numpy(dtype=float)
        valid_mask = np.isfinite(x_arr) & np.isfinite(line_arr)
        if x_arr.size == 0 or line_arr.size != x_arr.size or not np.any(valid_mask):
            return None
        spans = []
        start_idx = None
        last_idx = None
        current_line = None
        for idx in np.flatnonzero(valid_mask):
            line_no = int(line_arr[idx])
            if start_idx is None:
                start_idx = idx
                last_idx = idx
                current_line = line_no
                continue
            if line_no != current_line or idx != last_idx + 1:
                spans.append((current_line, float(x_arr[start_idx]), float(x_arr[last_idx])))
                start_idx = idx
                current_line = line_no
            last_idx = idx
        if start_idx is not None and last_idx is not None and current_line is not None:
            spans.append((current_line, float(x_arr[start_idx]), float(x_arr[last_idx])))
        if not spans:
            return None
        tick_count = self.get_time_line_axis_capacity(ax, min_ticks=8, max_ticks=max_ticks)
        spans = self.sample_time_line_spans(spans, tick_count)
        tick_positions = [float((start_x + end_x) / 2.0) for _, start_x, end_x in spans]
        tick_labels = [str(int(line_no)) for line_no, _, _ in spans]
        if not tick_positions:
            return None
        top_ax = ax.secondary_xaxis("top")
        top_ax.set_xticks(tick_positions)
        top_ax.set_xticklabels(tick_labels, rotation=45, ha="left")
        top_ax.set_xlabel("程序行号", fontsize=PLOT_FONT_BASE, fontweight="bold", color="black")
        top_ax.tick_params(labelsize=PLOT_FONT_BASE - 1, colors="black")
        return top_ax

    def _resolve_selected_analysis_run(self):
        if not self.analysis_result or not self.analysis_result["runs"]:
            return None
        selected_text = str(self.analysis_run_var.get()).strip()
        selected_run_id = selected_text.split("|", 1)[0].strip() if selected_text else ""
        for frame in self.analysis_result["runs"]:
            if str(frame["run_id"].iloc[0]) == selected_run_id:
                return frame
        return self.analysis_result["runs"][0]

    def _resolve_selected_analysis_interval_id(self):
        selected_text = str(self.analysis_interval_var.get()).strip()
        return selected_text.split("|", 1)[0].strip() if selected_text else ""

    def _analysis_chart_key(self):
        return str(self.analysis_chart_var.get()).strip() or "区间平均负载对比图"

    def _ensure_analysis_plot_canvas(self):
        for widget in self.analysis_plot_frame.winfo_children():
            widget.destroy()
        fig, ax = plt.subplots(figsize=(10.8, 5.6), dpi=90)
        fig.patch.set_facecolor(PLOT_FIG_BG)
        ax.set_facecolor(PLOT_AX_BG)
        self.apply_plot_style(ax, grid=True)
        canvas = FigureCanvasTkAgg(fig, master=self.analysis_plot_frame)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.analysis_plot_frame.grid_rowconfigure(0, weight=1)
        self.analysis_plot_frame.grid_columnconfigure(0, weight=1)
        self._analysis_plot_fig = fig
        self._analysis_plot_canvas = canvas
        self._analysis_plot_ax = ax
        return fig, ax, canvas

    def _set_analysis_summary_text(self, text):
        if not hasattr(self, "analysis_summary_text"):
            return
        self.analysis_summary_text.configure(state="normal")
        self.analysis_summary_text.delete("1.0", tk.END)
        self.analysis_summary_text.insert("1.0", text or "暂无分析摘要")
        self.analysis_summary_text.configure(state="disabled")
        self.analysis_summary_var.set(text or "暂无分析摘要")
        self.analysis_current_summary = text or ""

    def refresh_data_analysis_view(self):
        if not getattr(self, "analysis_plot_frame", None):
            return
        fig, ax, canvas = self._ensure_analysis_plot_canvas()
        self.analysis_current_table = pd.DataFrame()
        if not self.analysis_result:
            self.analysis_run_combo["values"] = []
            self.analysis_interval_combo["values"] = []
            self.analysis_run_var.set("")
            self.analysis_interval_var.set("")
            ax.text(0.5, 0.5, "请先导入首次加工与优化 run，并点击“分析”", ha="center", va="center", transform=ax.transAxes, fontsize=PLOT_FONT_BASE, color=UI_COLOR_TEXT_MUTED)
            canvas.draw_idle()
            self._populate_treeview(self.analysis_table_tree, pd.DataFrame())
            self._set_analysis_summary_text("暂无分析摘要")
            return

        chart_key = self._analysis_chart_key()
        if chart_key == "负载图":
            table_df, summary = self._plot_analysis_load_chart(ax)
        elif chart_key == "区间平均负载对比图":
            table_df, summary = self._plot_analysis_average(ax)
        elif chart_key == "区间波动图":
            table_df, summary = self._plot_analysis_fluctuation(ax)
        elif chart_key == "区间调节强度图":
            table_df, summary = self._plot_analysis_control(ax)
        elif chart_key == "区间响应时序图":
            table_df, summary = self._plot_analysis_response(ax)
        else:
            table_df, summary = self._plot_analysis_drift(ax)

        if chart_key == "负载图":
            fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.14)
        else:
            fig.tight_layout(pad=1.0)
        canvas.draw_idle()
        self.analysis_current_table = table_df if isinstance(table_df, pd.DataFrame) else pd.DataFrame(table_df)
        self._populate_treeview(self.analysis_table_tree, self.analysis_current_table, max_rows=3000)
        self._set_analysis_summary_text(summary)

    def _plot_analysis_load_chart(self, ax):
        self.apply_plot_style(ax, grid=False)
        load_df = self.analysis_result.get("load_chart_df")
        if not isinstance(load_df, pd.DataFrame) or load_df.empty:
            ax.text(0.5, 0.5, "当前没有可显示的负载图数据", ha="center", va="center", transform=ax.transAxes, fontsize=PLOT_FONT_BASE, color=UI_COLOR_TEXT_MUTED)
            return pd.DataFrame(), "\n".join(self.analysis_result["summary_lines"])

        load_df = load_df.copy()
        if "target_load" not in load_df.columns:
            load_df["target_load"] = np.nan
        if "target_segment_type" not in load_df.columns:
            load_df["target_segment_type"] = "background"

        prediction_label = str(self.analysis_result.get("prediction_label") or self.get_prediction_curve_label())
        if prediction_label not in load_df.columns:
            pred_candidates = [col for col in load_df.columns if str(col) in {"预测负载", "后验曲线"}]
            prediction_label = pred_candidates[0] if pred_candidates else next((col for col in load_df.columns if "预测" in str(col) or "后验" in str(col)), "")
        x_series = pd.to_numeric(load_df.get("time_ms"), errors="coerce")
        x_label = "时间 (ms)"
        if x_series.isna().all():
            x_series = pd.to_numeric(load_df.get("sample_index"), errors="coerce")
            x_label = "样本序号"
        x_values = x_series.to_numpy(dtype=float)
        if x_values.size != len(load_df):
            x_values = np.arange(len(load_df), dtype=float)
            x_label = "样本序号"
        actual_values = pd.to_numeric(load_df.get("actual_load"), errors="coerce").to_numpy(dtype=float)
        predicted_values = (
            pd.to_numeric(load_df.get(prediction_label), errors="coerce").to_numpy(dtype=float)
            if prediction_label and prediction_label in load_df.columns else np.full(len(load_df), np.nan, dtype=float)
        )
        target_values = pd.to_numeric(load_df.get("target_load"), errors="coerce").to_numpy(dtype=float)
        segment_types = load_df["target_segment_type"].astype(str).to_numpy(dtype=object)

        steady_idle_blocks = self.compute_contiguous_blocks(segment_types == "steady_idle")
        steady_cutting_blocks = self.compute_contiguous_blocks(segment_types == "steady_cutting")
        nonsteady_blocks = self.compute_contiguous_blocks(segment_types == "nonsteady")
        if steady_idle_blocks:
            self._draw_curve_background_blocks(ax, x_values, predicted_values, steady_idle_blocks, color="#D3D7DC", alpha=0.28, label="空载段", zorder=1)
        if steady_cutting_blocks:
            self._draw_curve_background_blocks(ax, x_values, predicted_values, steady_cutting_blocks, color="#1E88E5", alpha=0.18, label="稳态区间", zorder=1)
        if nonsteady_blocks:
            self._draw_curve_background_blocks(ax, x_values, predicted_values, nonsteady_blocks, color="#4A4A4A", alpha=0.24, label="非稳态区间", zorder=1)

        actual_blocks = self.compute_contiguous_blocks(np.isfinite(x_values) & np.isfinite(actual_values))
        pred_blocks = self.compute_contiguous_blocks(np.isfinite(x_values) & np.isfinite(predicted_values))
        target_blocks = self.compute_contiguous_blocks(np.isfinite(x_values) & np.isfinite(target_values))
        if actual_blocks:
            self.plot_series_by_blocks(ax, x_values, actual_values, actual_blocks, color="#1F77B4", linewidth=1.6, alpha=0.95, label="实际负载", zorder=6)
            actual_plot_mask = np.isfinite(x_values) & np.isfinite(actual_values)
            if np.any(actual_plot_mask):
                avg_load = float(np.nanmean(actual_values[actual_plot_mask]))
                ax.plot([float(np.nanmin(x_values[actual_plot_mask])), float(np.nanmax(x_values[actual_plot_mask]))], [avg_load, avg_load], color="#90A4AE", linewidth=0.9, linestyle="--", alpha=0.28, label="_nolegend_", zorder=7)
        if pred_blocks:
            self.plot_series_by_blocks(ax, x_values, predicted_values, pred_blocks, color="#F97316", linewidth=1.5, linestyle="--", alpha=0.95, label=(prediction_label or "预测负载"), zorder=8)
        if target_blocks:
            self.plot_series_by_blocks(ax, x_values, target_values, target_blocks, color="#D32F2F", linewidth=1.8, alpha=0.95, label="目标值", zorder=9)

        ax.set_xlabel(x_label, fontsize=PLOT_FONT_BASE, fontweight="bold", color=PLOT_TEXT_COLOR)
        ax.set_ylabel("功率 (W)", fontsize=PLOT_FONT_BASE, fontweight="bold", color=PLOT_TEXT_COLOR)
        ax.tick_params(labelsize=PLOT_FONT_BASE, colors=PLOT_TEXT_COLOR)
        if x_label == "时间 (ms)":
            self._apply_analysis_time_line_axis(ax, x_values, load_df.get("line_no_raw"))
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            legend = ax.legend(handles, labels, loc="upper right", fontsize=PLOT_FONT_BASE - 1, framealpha=0.95, shadow=False, fancybox=False, edgecolor=PLOT_SPINE_COLOR, borderpad=0.6, labelspacing=0.4)
            legend.get_frame().set_facecolor("white")
            legend.get_frame().set_linewidth(0.6)
        table_df = load_df.loc[:, [col for col in ["sample_index", "time_ms", "line_no_raw", "interval_id", prediction_label, "actual_load", "target_load", "target_segment_type"] if col and col in load_df.columns]].copy()
        return table_df, "\n".join(self.analysis_result["summary_lines"])

    def _plot_analysis_average(self, ax):
        df = self.analysis_result["average_table"].copy()
        metric = self.analysis_metric_var.get()
        if metric not in {"avg_actual_load", "avg_F"}:
            metric = "avg_actual_load"
            self.analysis_metric_var.set(metric)
        pivot_df = df.pivot(index="interval_id", columns="run_label", values=metric)
        x = np.arange(len(pivot_df.index), dtype=float)
        color_cycle = ["#1F77B4", "#E15759", "#2CA02C", "#FF9F1C", "#9467BD", "#17BECF"]
        for idx, column in enumerate(pivot_df.columns):
            ax.plot(x, pivot_df[column].to_numpy(dtype=float), marker="o", linewidth=1.8, label=column, color=color_cycle[idx % len(color_cycle)], zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(list(pivot_df.index))
        ax.set_title("多次优化的区间平均负载对比图", fontsize=PLOT_FONT_BASE + 2, fontweight="bold", loc="left", color=PLOT_TEXT_COLOR)
        ax.set_xlabel("指令区间", fontsize=PLOT_FONT_BASE, fontweight="bold")
        ax.set_ylabel(metric, fontsize=PLOT_FONT_BASE, fontweight="bold")
        ax.legend(loc="upper right", fontsize=PLOT_FONT_BASE - 1, framealpha=0.95)
        return df, "\n".join(self.analysis_result["summary_lines"])

    def _plot_analysis_fluctuation(self, ax):
        df = self.analysis_result["fluctuation_table"].copy()
        metric = self.analysis_metric_var.get()
        if metric not in {"P_std", "P_peak_to_peak", "residual_rms", "F_std"}:
            metric = "P_std"
            self.analysis_metric_var.set(metric)
        pivot_df = df.pivot(index="interval_id", columns="run_label", values=metric).fillna(0.0)
        interval_ids = list(pivot_df.index)
        x = np.arange(len(interval_ids), dtype=float)
        width = 0.8 / max(1, len(pivot_df.columns))
        color_cycle = ["#1F77B4", "#E15759", "#2CA02C", "#FF9F1C", "#9467BD", "#17BECF"]
        for idx, column in enumerate(pivot_df.columns):
            ax.bar(x + idx * width - 0.4 + width / 2, pivot_df[column].to_numpy(dtype=float), width=width, label=column, color=color_cycle[idx % len(color_cycle)], alpha=0.9, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(interval_ids)
        ax.set_title("区间波动图", fontsize=PLOT_FONT_BASE + 2, fontweight="bold", loc="left", color=PLOT_TEXT_COLOR)
        ax.set_xlabel("指令区间", fontsize=PLOT_FONT_BASE, fontweight="bold")
        ax.set_ylabel(metric, fontsize=PLOT_FONT_BASE, fontweight="bold")
        ax.legend(loc="upper right", fontsize=PLOT_FONT_BASE - 1, framealpha=0.95)
        high_var = df.loc[df["is_high_variation"], "interval_id"].drop_duplicates().tolist()
        extra = [f"当前波动指标高风险区间: {', '.join(high_var[:8])}"] if high_var else []
        return df, "\n".join(self.analysis_result["summary_lines"] + extra)

    def _plot_analysis_control(self, ax):
        df = self.analysis_result["control_table"].copy()
        selected_run = self._resolve_selected_analysis_run()
        run_group = df[df["run_id"] == str(selected_run["run_id"].iloc[0])].copy() if selected_run is not None else df.copy()
        if run_group.empty:
            run_group = df[df["role"] != "baseline"].copy()
        interval_ids = list(run_group["interval_id"])
        x = np.arange(len(interval_ids), dtype=float)
        colors = ["#E15759" if bool(flag) else "#4E79A7" for flag in run_group["is_hard_to_reach"].tolist()]
        ax.bar(x, run_group["delta_F"].to_numpy(dtype=float), color=colors, alpha=0.85, zorder=3, label="F变化量")
        ax2 = ax.twinx()
        ax2.plot(x, run_group["delta_P"].to_numpy(dtype=float), color="#2CA02C", marker="o", linewidth=1.6, label="P变化量", zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(interval_ids)
        ax.set_title("区间调节强度图", fontsize=PLOT_FONT_BASE + 2, fontweight="bold", loc="left", color=PLOT_TEXT_COLOR)
        ax.set_xlabel("指令区间", fontsize=PLOT_FONT_BASE, fontweight="bold")
        ax.set_ylabel("ΔF", fontsize=PLOT_FONT_BASE, fontweight="bold")
        ax2.set_ylabel("ΔP", fontsize=PLOT_FONT_BASE, fontweight="bold", color="#2CA02C")
        handles, labels = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(handles + h2, labels + l2, loc="upper right", fontsize=PLOT_FONT_BASE - 1, framealpha=0.95)
        hard_ids = run_group.loc[run_group["is_hard_to_reach"], "interval_id"].tolist()
        extra = [f"当前 run 疑似难达区: {', '.join(hard_ids[:8])}"] if hard_ids else []
        return run_group, "\n".join(self.analysis_result["summary_lines"] + extra)

    def _plot_analysis_response(self, ax):
        selected_run = self._resolve_selected_analysis_run()
        if selected_run is None:
            return pd.DataFrame(), "\n".join(self.analysis_result["summary_lines"])
        selected_interval = self._resolve_selected_analysis_interval_id()
        if not selected_interval and not self.analysis_result["intervals"].empty:
            selected_interval = str(self.analysis_result["intervals"]["interval_id"].iloc[0])
            self.analysis_interval_var.set(selected_interval)
        interval_row = self.analysis_result["intervals"].loc[self.analysis_result["intervals"]["interval_id"] == selected_interval]
        if interval_row.empty:
            interval_row = self.analysis_result["intervals"].iloc[[0]]
        interval = interval_row.iloc[0]
        segment = selected_run.iloc[int(interval["start_idx"]):int(interval["end_idx"]) + 1].copy()
        local_x = np.arange(len(segment), dtype=float)
        ax.plot(local_x, pd.to_numeric(segment["P"], errors="coerce").to_numpy(dtype=float), color="#1F77B4", linewidth=1.8, label="实际负载 P", zorder=3)
        ax2 = ax.twinx()
        ax2.plot(local_x, pd.to_numeric(segment["F"], errors="coerce").to_numpy(dtype=float), color="#E15759", linewidth=1.4, linestyle="--", label="F", zorder=4)
        response_df = self.analysis_result["response_table"]
        response_row = response_df[
            (response_df["run_id"] == str(selected_run["run_id"].iloc[0]))
            & (response_df["interval_id"] == str(interval["interval_id"]))
        ]
        if not response_row.empty:
            delay = float(response_row.iloc[0]["response_delay_ms"])
            settling = float(response_row.iloc[0]["settling_time_ms"])
            if np.isfinite(delay):
                ax.axvline(delay, color="#8E44AD", linestyle="--", linewidth=1.0, alpha=0.8, label="响应延迟")
            if np.isfinite(settling):
                ax.axvline(settling, color="#2CA02C", linestyle=":", linewidth=1.0, alpha=0.85, label="稳定时间")
        ax.set_title(f"区间响应时序图: {interval['interval_id']}", fontsize=PLOT_FONT_BASE + 2, fontweight="bold", loc="left", color=PLOT_TEXT_COLOR)
        ax.set_xlabel("区间内指令点序号", fontsize=PLOT_FONT_BASE, fontweight="bold")
        ax.set_ylabel("P", fontsize=PLOT_FONT_BASE, fontweight="bold")
        ax2.set_ylabel("F", fontsize=PLOT_FONT_BASE, fontweight="bold", color="#E15759")
        handles, labels = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(handles + h2, labels + l2, loc="upper right", fontsize=PLOT_FONT_BASE - 1, framealpha=0.95)
        table_df = response_row.merge(
            self.analysis_result["boundary_table"][
                (self.analysis_result["boundary_table"]["run_id"] == str(selected_run["run_id"].iloc[0]))
                & (self.analysis_result["boundary_table"]["interval_id"] == str(interval["interval_id"]))
            ],
            on=["run_id", "run_label", "file_name", "role", "interval_id"],
            how="left",
        )
        return table_df, "\n".join(self.analysis_result["summary_lines"])

    def _plot_analysis_drift(self, ax):
        selected_interval = self._resolve_selected_analysis_interval_id()
        if not selected_interval and not self.analysis_result["intervals"].empty:
            selected_interval = str(self.analysis_result["intervals"]["interval_id"].iloc[0])
            self.analysis_interval_var.set(selected_interval)

        metric = self.analysis_metric_var.get()
        if metric in {"avg_actual_load", "avg_F"}:
            source_df = self.analysis_result["average_table"].copy()
        else:
            source_df = self.analysis_result["fluctuation_table"].copy()
            if metric not in source_df.columns:
                metric = "P_std"
                self.analysis_metric_var.set(metric)
        group = source_df[source_df["interval_id"] == selected_interval].copy()
        if group.empty:
            group = source_df.copy()
        metric_values = []
        run_labels = []
        for order_idx, frame in enumerate(self.analysis_result["runs"], 1):
            run_id = str(frame["run_id"].iloc[0])
            row = group[group["run_id"] == run_id]
            if row.empty:
                continue
            metric_values.append(float(row.iloc[0][metric]))
            run_labels.append(build_run_label(str(frame["role"].iloc[0]), str(frame["file_name"].iloc[0]), order_idx))
        x = np.arange(len(metric_values), dtype=float)
        ax.plot(x, metric_values, color="#1F77B4", linewidth=1.8, marker="o", zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(run_labels, rotation=10)
        ax.set_title(f"区间漂移趋势图: {selected_interval}", fontsize=PLOT_FONT_BASE + 2, fontweight="bold", loc="left", color=PLOT_TEXT_COLOR)
        ax.set_xlabel("run", fontsize=PLOT_FONT_BASE, fontweight="bold")
        ax.set_ylabel(metric, fontsize=PLOT_FONT_BASE, fontweight="bold")
        drift_df = self.analysis_result["drift_table"]
        table_df = drift_df[drift_df["interval_id"] == selected_interval].copy() if not drift_df.empty else pd.DataFrame()
        return table_df, "\n".join(self.analysis_result["summary_lines"])

    def export_current_analysis_table(self):
        if self.analysis_current_table is None or self.analysis_current_table.empty:
            messagebox.showwarning("暂无表格", "当前主图没有可导出的统计表")
            return
        file_path = filedialog.asksaveasfilename(
            title="导出当前统计表",
            defaultextension=".csv",
            filetypes=(("CSV 文件", "*.csv"), ("所有文件", "*.*")),
        )
        if not file_path:
            return
        export_table_to_csv(self.analysis_current_table, file_path)
        self.analysis_status_var.set(f"已导出当前统计表: {os.path.basename(file_path)}")
