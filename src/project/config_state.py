from __future__ import annotations

from .shared import *


class ConfigStateMixin:
    _SEGMENT_STATE_LABELS = {
        "idle": (0, "空载"),
        "entry": (1, "进刀"),
        "steady": (2, "稳态"),
        "transition": (3, "过渡"),
        "nonsteady": (4, "非稳态"),
        "exit": (5, "退刀"),
    }

    def _get_interval_state_display(self, record):
        """返回兼容旧记录的六态类型、编码和中文名称。"""
        segment_type = str(record.get("segment_type") or "").strip().lower()
        if segment_type not in self._SEGMENT_STATE_LABELS:
            is_idle = bool(record.get("is_idle_interval")) or str(
                record.get("kc_source", "")
            ).strip().lower() == "idle"
            segment_type = "idle" if is_idle else "steady"
        expected_code, label = self._SEGMENT_STATE_LABELS[segment_type]
        try:
            state_code = int(record.get("state_code", expected_code))
        except Exception:
            state_code = expected_code
        if state_code != expected_code:
            state_code = expected_code
        return segment_type, state_code, label

    def _parse_optional_float(self, value):
        raw = str(value).strip()
        if not raw:
            return None
        try:
            numeric = float(raw)
        except Exception:
            return None
        return numeric if np.isfinite(numeric) else None

    def _format_optional_model_param(self, value):
        numeric = self._parse_optional_float(value)
        if numeric is None:
            return ""
        text = f"{numeric:.6f}".rstrip("0").rstrip(".")
        return text if text else "0"

    def get_kc_value(self, default=0.0):
        value = self._parse_optional_float(self.kc_coeff.get())
        if value is None:
            return max(float(default), 0.0)
        return max(float(value), 0.0)

    def get_ke_value(self, default=0.0):
        value = self._parse_optional_float(self.ke_coeff.get())
        if value is None:
            return max(float(default), 0.0)
        return max(float(value), 0.0)

    def has_identified_kc_ke(self):
        return self._parse_optional_float(self.kc_coeff.get()) is not None and self._parse_optional_float(self.ke_coeff.get()) is not None

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
                if "p_idle" in self.app_config:
                    idle_val = float(self.app_config.get("p_idle", 0.0) or 0.0)
                    self.p_idle_var.set(idle_val)
                    self.current_program_idle_power.set(idle_val)
                idle_power_model = self.app_config.get("idle_power_model")
                if isinstance(idle_power_model, dict) and idle_power_model.get("speeds") and idle_power_model.get("powers"):
                    self.idle_power_model = idle_power_model
                if "idle_model_signature" in self.app_config:
                    self.idle_model_signature = str(self.app_config.get("idle_model_signature", "") or "")
                if "kc_coeff" in self.app_config:
                    self.kc_coeff.set(self._format_optional_model_param(max(float(self.app_config.get("kc_coeff", 0.0) or 0.0), 0.0)))
                if "kc_sigma" in self.app_config:
                    self.kc_sigma.set(float(self.app_config.get("kc_sigma", self.kc_sigma.get()) or 0.0))
                if "ke_coeff" in self.app_config:
                    self.ke_coeff.set(self._format_optional_model_param(max(float(self.app_config.get("ke_coeff", 0.0) or 0.0), 0.0)))
                if "lock_ke_during_identification" in self.app_config:
                    self.lock_ke_during_identification.set(
                        bool(self.app_config.get("lock_ke_during_identification", True))
                    )
                if "lock_idle_during_identification" in self.app_config:
                    self.lock_idle_during_identification.set(
                        bool(self.app_config.get("lock_idle_during_identification", True))
                    )
                if "kc_beta" in self.app_config:
                    self.kc_beta.set(float(self.app_config.get("kc_beta", self.kc_beta.get()) or 0.0))
                saved_profile_index = self.app_config.get("saved_kc_profile_index")
                if isinstance(saved_profile_index, dict):
                    self.saved_kc_profile_index = saved_profile_index
                saved_profiles = self.app_config.get("saved_kc_profiles")
                if isinstance(saved_profiles, dict) and saved_profiles:
                    if hasattr(self, "_migrate_legacy_saved_kc_profiles"):
                        try:
                            self._migrate_legacy_saved_kc_profiles(saved_profiles, persist=False)
                        except Exception:
                            pass
                gcode_bindings = self.app_config.get("gcode_profile_bindings")
                if isinstance(gcode_bindings, dict):
                    normalized_bindings = {}
                    for gcode_path, profile_paths in gcode_bindings.items():
                        if isinstance(profile_paths, str):
                            profile_list = [profile_paths]
                        elif isinstance(profile_paths, list):
                            profile_list = [str(item).strip() for item in profile_paths if str(item).strip()]
                        else:
                            continue
                        normalized_bindings[str(gcode_path)] = profile_list
                    self.gcode_profile_bindings = normalized_bindings
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
            self.app_config["p_idle"] = float(self.p_idle_var.get())
            self.app_config["idle_power_model"] = dict(getattr(self, "idle_power_model", {}) or {})
            self.app_config["idle_model_signature"] = str(getattr(self, "idle_model_signature", "") or "")
            kc_value = self._parse_optional_float(self.kc_coeff.get())
            if kc_value is None:
                self.app_config.pop("kc_coeff", None)
            else:
                self.app_config["kc_coeff"] = max(float(kc_value), 0.0)
            self.app_config["kc_sigma"] = float(self.kc_sigma.get())
            ke_value = self._parse_optional_float(self.ke_coeff.get())
            if ke_value is None:
                self.app_config.pop("ke_coeff", None)
            else:
                self.app_config["ke_coeff"] = max(float(ke_value), 0.0)
            self.app_config["lock_ke_during_identification"] = bool(self.lock_ke_during_identification.get())
            self.app_config["lock_idle_during_identification"] = bool(self.lock_idle_during_identification.get())
            self.app_config["kc_beta"] = float(self.kc_beta.get())
            if hasattr(self, "_prune_saved_kc_profile_index"):
                self._prune_saved_kc_profile_index()
            self.app_config.pop("saved_kc_profiles", None)
            self.app_config["saved_kc_profile_index"] = dict(getattr(self, "saved_kc_profile_index", {}) or {})
            self.app_config["gcode_profile_bindings"] = dict(getattr(self, "gcode_profile_bindings", {}) or {})
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

    def set_progress(self, value=None, text=None, maximum=100, mode="determinate"):
        """更新底部进度条与状态文本。"""
        bar = getattr(self, "data_progress_bar", None)
        if bar is not None:
            try:
                if not bar.winfo_ismapped():
                    bar.grid()
            except Exception:
                pass
            try:
                bar.configure(mode=mode, maximum=maximum)
                if mode == "indeterminate":
                    if value is None:
                        bar.start(10)
                    else:
                        bar.stop()
                else:
                    try:
                        bar.stop()
                    except Exception:
                        pass
                    if value is not None:
                        bar["value"] = max(0, min(float(value), float(maximum)))
            except Exception:
                pass
        if text and hasattr(self, "status_var_data"):
            progress_text = str(text)
            if mode != "indeterminate" and value is not None:
                try:
                    maximum_value = float(maximum) if float(maximum) > 0 else 100.0
                    percent = int(round(float(value) / maximum_value * 100.0))
                    percent = max(0, min(percent, 100))
                    progress_text = f"[{percent:>3d}%] {progress_text}"
                except Exception:
                    pass
            self.status_var_data.set(progress_text)
        try:
            self.root.update_idletasks()
        except Exception:
            pass

    def reset_progress(self):
        """重置底部进度条。"""
        bar = getattr(self, "data_progress_bar", None)
        if bar is None:
            return
        try:
            bar.stop()
        except Exception:
            pass
        try:
            bar.configure(mode="determinate", maximum=100)
            bar["value"] = 0
        except Exception:
            pass
        try:
            bar.grid_remove()
        except Exception:
            pass

    def _format_interval_tree_metric(self, value, digits=3, default="--"):
        try:
            numeric = float(value)
        except Exception:
            return default
        if not np.isfinite(numeric):
            return default
        return f"{numeric:.{int(digits)}f}"

    def _normalize_interval_detail_value(self, value):
        if isinstance(value, dict):
            return {str(key): self._normalize_interval_detail_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._normalize_interval_detail_value(item) for item in value]
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            numeric = float(value)
            return numeric if np.isfinite(numeric) else str(value)
        if value is None or isinstance(value, str):
            return value
        return str(value)

    def _build_interval_summary_values(self, record):
        start_label = str(record.get("sample_start_label") or "--")
        end_label = str(record.get("sample_end_label") or "--")
        try:
            sample_count = int(record.get("point_count", record.get("sample_count", 0)) or 0)
        except Exception:
            sample_count = 0
        point_count_text = str(sample_count) if sample_count > 0 else "--"
        avg_load_text = self._format_interval_tree_metric(record.get("p_meas"), digits=3)
        if avg_load_text != "--":
            avg_load_text = f"{avg_load_text} W"
        return start_label, end_label, point_count_text, avg_load_text

    def _build_interval_tree_row_values(self, record):
        segment_type, state_code, state_label = self._get_interval_state_display(record)
        process_start = str(record.get("process_start_label") or record.get("start_line_raw") or record.get("start_line") or "--")
        process_end = str(record.get("process_end_label") or record.get("end_line_raw") or record.get("end_line") or "--")
        sample_start = str(record.get("sample_start_label") or "").strip()
        sample_end = str(record.get("sample_end_label") or "").strip()
        sample_status = str(record.get("_sample_projection_status") or "").strip()
        sample_text = (
            f"{sample_start} -> {sample_end}"
            if sample_start and sample_end
            else (sample_status or "无采样投影")
        )
        x_start = self._format_interval_tree_metric(record.get("display_start_x", record.get("start_s")), digits=3)
        x_end = self._format_interval_tree_metric(record.get("display_end_x", record.get("end_s")), digits=3)
        point_count_text = self._build_interval_summary_values(record)[2]
        kc_text = self._format_interval_tree_metric(record.get("K_c_hat"), digits=6)
        p_pred_text = self._format_interval_tree_metric(record.get("p_pred"), digits=3)
        p_meas_text = self._format_interval_tree_metric(record.get("p_meas"), digits=3)
        summary_parts = [f"{state_label}[{state_code}]", f"n={point_count_text}"]
        if kc_text != "--":
            summary_parts.append(f"Kc={kc_text}")
        if p_pred_text != "--":
            summary_parts.append(f"Pp={p_pred_text}W")
        if p_meas_text != "--":
            summary_parts.append(f"Pm={p_meas_text}W")
        if bool(record.get("review_required")):
            summary_parts.append("需复核")
        return (
            f"{process_start} -> {process_end}",
            sample_text,
            f"{x_start} -> {x_end}",
            " | ".join(summary_parts),
        )

    def _build_interval_detail_text(self, interval_id, subtype_text, record):
        segment_type, state_code, state_label = self._get_interval_state_display(record)
        process_start_label = str(record.get("process_start_label") or record.get("start_line") or "--")
        process_end_label = str(record.get("process_end_label") or record.get("end_line") or "--")
        sample_start_label = str(record.get("sample_start_label") or "").strip()
        sample_end_label = str(record.get("sample_end_label") or "").strip()
        sample_projection_status = str(
            record.get("_sample_projection_status") or "无采样投影"
        ).strip()
        sample_interval_text = (
            f"{sample_start_label} -> {sample_end_label}"
            if sample_start_label and sample_end_label
            else sample_projection_status
        )
        anchor_start_label = str(record.get("sample_anchor_start_label") or "--")
        anchor_end_label = str(record.get("sample_anchor_end_label") or "--")
        kc_source = str(record.get("kc_source") or "").strip()
        if kc_source.startswith("interval"):
            kc_source = "区间中值"
        normalized_record = {
            str(key): self._normalize_interval_detail_value(value)
            for key, value in sorted(dict(record).items(), key=lambda item: str(item[0]))
        }
        lines = [
            f"{interval_id} | {subtype_text}",
            "",
            "摘要",
            f"区间编号: {interval_id}",
            f"区间类型: {segment_type} ({state_label})",
            f"state_code: {state_code}",
            f"review_required: {bool(record.get('review_required', False))}",
            f"decision_reason: {str(record.get('decision_reason') or '--')}",
            f"sample 区间: {sample_interval_text}",
            f"sample 投影状态: {sample_projection_status}",
            f"sample 投影原因: {str(record.get('_sample_projection_reason') or '--')}",
            f"anchor 区间: {anchor_start_label} -> {anchor_end_label}",
            f"process 区间: {process_start_label} -> {process_end_label}",
            f"点数: {self._build_interval_summary_values(record)[2]}",
            f"物理长度: {self._format_interval_tree_metric(record.get('length_mm'), digits=3)} mm",
            f"规则置信: {str(record.get('confidence_level') or '--')} (margin={self._format_interval_tree_metric(record.get('score_margin'), digits=6)})",
            f"平均负载 P_meas: {self._format_interval_tree_metric(record.get('p_meas'), digits=3)} W",
            f"预测负载 P_pred: {self._format_interval_tree_metric(record.get('p_pred'), digits=3)} W",
            "",
            "关键统计",
            f"K_c_hat: {self._format_interval_tree_metric(record.get('K_c_hat'), digits=6)}",
            f"K_c_UCB: {self._format_interval_tree_metric(record.get('K_c_UCB'), digits=6)}",
            f"sigma_Kc: {self._format_interval_tree_metric(record.get('sigma_Kc'), digits=6)}",
            f"kc_source: {kc_source or '--'}",
            f"valid_kc_count: {int(record.get('valid_kc_count', 0) or 0)}",
            f"gated_out_count: {int(record.get('gated_out_count', 0) or 0)}",
            f"actual_load_std: {self._format_interval_tree_metric(record.get('actual_load_std'), digits=3)}",
            f"actual_load_diff_std: {self._format_interval_tree_metric(record.get('actual_load_diff_std'), digits=3)}",
            f"sigma_idle: {self._format_interval_tree_metric(record.get('sigma_idle'), digits=3)}",
            f"delta_mrr: {self._format_interval_tree_metric(record.get('delta_mrr'), digits=6)}",
            "",
            "几何/工艺",
            f"x(sample): {self._format_interval_tree_metric(record.get('display_start_x'), digits=3)} -> {self._format_interval_tree_metric(record.get('display_end_x'), digits=3)}",
            f"x(process): {self._format_interval_tree_metric(record.get('process_start_x'), digits=3)} -> {self._format_interval_tree_metric(record.get('process_end_x'), digits=3)}",
            f"t(sample): {self._format_interval_tree_metric(record.get('display_start_t'), digits=3)} -> {self._format_interval_tree_metric(record.get('display_end_t'), digits=3)}",
            f"a_p: {self._format_interval_tree_metric(record.get('ap_mean', record.get('a_p')), digits=3)}",
            f"a_e: {self._format_interval_tree_metric(record.get('ae_mean', record.get('a_e')), digits=3)}",
            f"F_program: {self._format_interval_tree_metric(record.get('F_program_mean', record.get('F_plan')), digits=3)}",
            f"MRR_program: {self._format_interval_tree_metric(record.get('MRR_program_mean'), digits=6)}",
            f"P_idle: {self._format_interval_tree_metric(record.get('p_idle'), digits=3)} W",
            "",
            "原始记录字段",
            json.dumps(normalized_record, ensure_ascii=False, indent=2, allow_nan=True),
        ]
        return "\n".join(lines)

    def _set_interval_detail_button_enabled(self, enabled):
        button = getattr(self, "show_interval_detail_btn", None)
        if button is None:
            return
        try:
            if enabled:
                button.state(["!disabled"])
            else:
                button.state(["disabled"])
        except Exception:
            pass

    def _get_selected_interval_detail_payload(self):
        payload_map = getattr(self, "_ideal_tree_interval_payloads", {}) or {}
        if not payload_map:
            return None
        tree = getattr(self, "ideal_tree", None)
        if tree is not None:
            try:
                selection = tree.selection()
            except Exception:
                selection = ()
            if selection:
                item_id = str(selection[0])
                self._selected_interval_detail_item = item_id
                return payload_map.get(item_id)
        item_id = str(getattr(self, "_selected_interval_detail_item", "") or "")
        return payload_map.get(item_id)

    def _show_selected_interval_detail_dialog(self, event=None):
        payload = self._get_selected_interval_detail_payload()
        if not payload:
            self._set_interval_detail_button_enabled(False)
            return None

        dialog = tk.Toplevel(self.root)
        dialog.title(f"区间完整信息 - {payload['title']}")
        dialog.geometry("1100x720")
        dialog.minsize(860, 560)
        dialog.transient(self.root)
        dialog.grab_set()
        center_dialog_on_parent(dialog, self.root)

        container = ttk.Frame(dialog, padding=8)
        container.pack(fill=tk.BOTH, expand=True)
        container.grid_rowconfigure(1, weight=1)
        container.grid_columnconfigure(0, weight=1)

        ttk.Label(container, text=payload["title"], font=UI_FONT_LARGE).grid(row=0, column=0, sticky="w", pady=(0, 6))

        text_frame = ttk.Frame(container)
        text_frame.grid(row=1, column=0, sticky="nsew")
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)

        detail_text = tk.Text(text_frame, wrap="none", font=UI_FONT_NORMAL)
        detail_text.grid(row=0, column=0, sticky="nsew")
        y_scroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=detail_text.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(text_frame, orient=tk.HORIZONTAL, command=detail_text.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        detail_text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        detail_text.insert("1.0", payload["text"])
        detail_text.configure(state="disabled")

        ttk.Button(container, text="关闭", command=dialog.destroy, width=10).grid(row=2, column=0, sticky="e", pady=(8, 0))
        return "break" if event is not None else None

    def _refresh_ideal_tree(self):
        """刷新右侧全行程六类区间详情树。"""
        if not hasattr(self, 'ideal_tree'):
            return
        for item in self.ideal_tree.get_children():
            self.ideal_tree.delete(item)
        self._ideal_tree_interval_payloads = {}
        self._selected_interval_detail_item = ""
        self._set_interval_detail_button_enabled(False)

        def _row_values(*values):
            try:
                column_count = len(tuple(self.ideal_tree.cget("columns")))
            except Exception:
                column_count = 4
            padded = list(values[:column_count])
            while len(padded) < column_count:
                padded.append("")
            return tuple(padded)

        current_program = ""
        if hasattr(self, "get_current_program_key"):
            current_program = str(self.get_current_program_key() or "").strip()
        if not current_program and hasattr(self, "sample_program_name"):
            try:
                current_program = str(self.sample_program_name.get() or "").strip()
            except Exception:
                current_program = ""

        interval_records = self._get_current_interval_records(allow_profile_fallback=False)
        if bool(getattr(self, "_current_interval_ready", False)) and not interval_records:
            self.ideal_tree.insert("", "end", text="当前没有全行程划分结果", values=_row_values("当前没有全行程划分结果"))

        # 权威区间始终保留在 ProcessInfo 过程域。右侧详情只使用
        # 投影副本显示 sample 坐标，投影失败时不得回退成过程点标签。
        display_interval_records = [dict(record) for record in interval_records]
        segmentation_authoritative = bool(
            getattr(self, "_current_interval_ready", False)
            and str(getattr(self, "_current_interval_source", "") or "")
            == "segmentation"
        )
        if segmentation_authoritative and display_interval_records:
            projection_reason = ""
            try:
                projected_records = self._get_authoritative_segmentation_sample_records()
            except Exception as projection_exc:
                projected_records = []
                projection_reason = str(projection_exc)

            projected_by_id = {}
            for projected in projected_records:
                projected_id = str(
                    projected.get("zone_id") or projected.get("interval_id") or ""
                ).strip()
                if projected_id:
                    projected_by_id[projected_id] = projected

            sample_projection_keys = (
                "sample_start_idx",
                "sample_end_idx",
                "sample_start_line",
                "sample_end_line",
                "sample_start_point_index",
                "sample_end_point_index",
                "sample_start_label",
                "sample_end_label",
                "sample_interval_range",
                "sample_count",
            )
            for display_record in display_interval_records:
                for key in sample_projection_keys:
                    display_record.pop(key, None)
                interval_id = str(
                    display_record.get("zone_id")
                    or display_record.get("interval_id")
                    or ""
                ).strip()
                projected = projected_by_id.get(interval_id)
                if isinstance(projected, dict):
                    for key in sample_projection_keys:
                        if key in projected:
                            display_record[key] = projected[key]
                    display_record["_sample_projection_status"] = "已投影"
                    display_record["_sample_projection_reason"] = ""
                else:
                    display_record["_sample_projection_status"] = (
                        "无采样投影"
                        if projection_reason
                        else "无独立采样投影（已折叠或合并）"
                    )
                    display_record["_sample_projection_reason"] = (
                        projection_reason or "该过程区间没有独立的实际采样点"
                    )

        programs: Dict[str, Dict[str, Optional[Dict]]] = {}
        for prog, program_info in (self.sample_programs or {}).items():
            tools = program_info.get("tools", {})
            prog_tools = programs.setdefault(prog, {})
            for tool_id in tools.keys():
                store = self.ideal_store.get((prog, tool_id))
                prog_tools[tool_id] = store

        if not programs:
            fallback_name = current_program or "当前工艺"
            programs[fallback_name] = {}
        if not current_program and len(programs) == 1:
            current_program = next(iter(programs.keys()))

        for prog in sorted(programs.keys()):
            is_active_program = bool(current_program) and str(prog) == str(current_program)
            program_text = str(prog)
            if is_active_program and interval_records:
                program_text = f"{program_text}  [当前全行程区间 {len(interval_records)} 段]"
            prog_node = self.ideal_tree.insert("", "end", text=program_text, open=True, values=_row_values())
            tool_map = programs[prog]

            process_path = self.program_process_file_map.get(prog)
            has_process_file = bool(process_path and os.path.exists(process_path))
            has_processed = self._has_processed_result_for(process_path)
            process_name = os.path.basename(process_path) if has_process_file else "未绑定"
            status_text = "已导入并分析完成" if has_process_file and has_processed else "待导入或待分析"
            self.ideal_tree.insert(
                prog_node,
                "end",
                text=f"工艺信息：{status_text} | 文件：{process_name}",
                values=_row_values(process_name, status_text),
            )

            if is_active_program:
                kc_text = self._format_interval_tree_metric(self.get_kc_value(), digits=6)
                ke_text = self._format_interval_tree_metric(self.get_ke_value(), digits=6)
                sigma_text = self._format_interval_tree_metric(getattr(self, "kc_sigma", None).get() if hasattr(getattr(self, "kc_sigma", None), "get") else float("nan"), digits=6)
                self.ideal_tree.insert(
                    prog_node,
                    "end",
                    text=f"当前模型：K_c={kc_text} | K_e={ke_text} | σ_Kc={sigma_text}",
                    values=_row_values(f"K_c={kc_text}", f"K_e={ke_text}", "", f"σ_Kc={sigma_text}"),
                )

            if is_active_program and interval_records:
                state_counts = {state: 0 for state in self._SEGMENT_STATE_LABELS}
                for record in interval_records:
                    segment_type, _state_code, _state_label = self._get_interval_state_display(record)
                    state_counts[segment_type] += 1
                count_text = " | ".join(
                    f"{label} {state_counts[state]}"
                    for state, (_code, label) in self._SEGMENT_STATE_LABELS.items()
                )
                interval_root = self.ideal_tree.insert(
                    prog_node,
                    "end",
                    text=f"全行程区间：共 {len(interval_records)} 段 | {count_text}",
                    open=True,
                    values=_row_values("", "", str(len(interval_records)), ""),
                )
                sorted_records = sorted(
                    display_interval_records,
                    key=lambda item: (
                        int(item.get("start_idx", 0) or 0),
                        int(item.get("start_line", 0) or 0),
                    ),
                )
                for idx, record in enumerate(sorted_records, 1):
                    interval_id = str(record.get("zone_id") or record.get("interval_id") or f"Z{idx:03d}")
                    _segment_type, state_code, state_label = self._get_interval_state_display(record)
                    subtype_text = f"{state_label}[{state_code}]"
                    process_text, sample_text, x_text, summary_text = self._build_interval_tree_row_values(record)
                    interval_node = self.ideal_tree.insert(
                        interval_root,
                        "end",
                        text=f"{interval_id} | {subtype_text}",
                        open=False,
                        values=_row_values(
                            process_text,
                            sample_text,
                            x_text,
                            summary_text,
                        ),
                    )
                    self._ideal_tree_interval_payloads[interval_node] = {
                        "title": f"{interval_id} | {subtype_text}",
                        "text": self._build_interval_detail_text(interval_id, subtype_text, record),
                    }
            elif is_active_program and has_process_file and has_processed:
                self.ideal_tree.insert(
                    prog_node,
                    "end",
                    text="全行程区间：当前尚未生成，请运行六类划分",
                    values=_row_values("当前尚未生成"),
                )

            tool_root = self.ideal_tree.insert(prog_node, "end", text="刀具理想值", open=False, values=_row_values())
            if not tool_map:
                self.ideal_tree.insert(tool_root, "end", text="当前没有可显示的刀具理想值", values=_row_values("当前没有可显示的刀具理想值"))
                continue
            for tool in sorted(tool_map.keys()):
                store = tool_map[tool]
                tool_label = self.format_tool_label(tool)
                if store:
                    rg = store.get("rg", 1.0)
                    mean_val, _, _ = self.compute_tool_measured_mean(prog, tool)
                    if mean_val is not None:
                        ideal_val = mean_val * rg
                        display = f"{tool_label}：理想值 {ideal_val:.2f} | rg={rg:.2f}"
                    else:
                        display = f"{tool_label}：理想值 未计算 | rg={rg:.2f}"
                else:
                    display = f"{tool_label}：理想值 未设定"
                self.ideal_tree.insert(tool_root, "end", text=tool_label, values=_row_values(display))

    def _on_ideal_tree_select(self, event=None):
        """右侧详情树仅作展示；若选中区间摘要，则允许弹出完整详情。"""
        tree = getattr(self, "ideal_tree", None)
        if tree is None:
            self._selected_interval_detail_item = ""
            self._set_interval_detail_button_enabled(False)
            return
        try:
            selection = tree.selection()
        except Exception:
            selection = ()
        self._selected_interval_detail_item = str(selection[0]) if selection else ""
        self._set_interval_detail_button_enabled(self._selected_interval_detail_item in getattr(self, "_ideal_tree_interval_payloads", {}))
        return

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
        """工艺信息页面已取消最小样本点限制。"""
        return

    def _schedule_min_length_update(self, immediate=False):
        """工艺信息页面已取消最小样本点限制。"""
        return

    def _apply_min_length_update(self):
        """工艺信息页面已取消最小样本点限制。"""
        return

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
            interval_policy = self._get_default_interval_policy() if hasattr(self, "_get_default_interval_policy") else "fresh_or_empty"
            self.generate_plots(save=False, silent=True, interval_policy=interval_policy)
        
        self.set_status(f"重导入完成: {os.path.basename(csv_latest)}；请为程序选择工艺信息表", 5000)

    def _recalculate_all_ideal_values(self):
        """对 ideal_store 中所有条目重算理想值（使用新均值×保存的rg）"""
        for (prog, tool), store in self.ideal_store.items():
            rg = store.get("rg", 1.0)
            mean_val, _, _ = self.compute_tool_measured_mean(prog, tool)

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
                interval_policy = self._get_default_interval_policy() if hasattr(self, "_get_default_interval_policy") else "fresh_or_empty"
                self.generate_plots(save=False, silent=True, interval_policy=interval_policy)
        
        ttk.Button(btn_frame, text="全选", width=10, 
                   command=lambda: [toggle_item(c) for c in tree.get_children()]).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="生成/保存", width=12, command=do_batch_save).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", width=10, command=dialog.destroy).pack(side=tk.LEFT, padx=5)
