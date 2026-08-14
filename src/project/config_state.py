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

    def _profile_config_enabled(self):
        release_mode = bool(getattr(self, "release_mode", False))
        return bool(getattr(self, "enable_profile_config", not release_mode))

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
                if self._profile_config_enabled():
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
                else:
                    # 发布版可读取其他通用设置，但 profile 数据不进入运行态。
                    self.app_config.pop("saved_kc_profile_index", None)
                    self.app_config.pop("saved_kc_profiles", None)
                    self.app_config.pop("gcode_profile_bindings", None)
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
            if self._profile_config_enabled():
                if hasattr(self, "_prune_saved_kc_profile_index"):
                    self._prune_saved_kc_profile_index()
                self.app_config.pop("saved_kc_profiles", None)
                self.app_config["saved_kc_profile_index"] = dict(getattr(self, "saved_kc_profile_index", {}) or {})
                self.app_config["gcode_profile_bindings"] = dict(getattr(self, "gcode_profile_bindings", {}) or {})
            else:
                self.app_config.pop("saved_kc_profiles", None)
                self.app_config.pop("saved_kc_profile_index", None)
                self.app_config.pop("gcode_profile_bindings", None)
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

    @staticmethod
    def _set_overview_text(target, text):
        if target is None:
            return
        try:
            target.set(str(text))
        except Exception:
            pass

    @staticmethod
    def _read_status_text(value):
        if value is None:
            return ""
        try:
            return str(value.get() or "").strip()
        except Exception:
            return str(value or "").strip()

    @staticmethod
    def _prefixed_overview_status(prefix, value, fallback):
        text = str(value or "").strip() or str(fallback)
        normalized_prefix = str(prefix).rstrip("：:")
        if text.startswith(normalized_prefix):
            _, separator, suffix = text.partition("：")
            if not separator:
                _, separator, suffix = text.partition(":")
            text = suffix.strip() if separator else text[len(normalized_prefix):].strip()
        return f"{normalized_prefix}：{text or fallback}"

    def _refresh_ideal_tree(self):
        """刷新右侧区间划分概况；保留旧方法名以兼容既有调用点。"""
        try:
            interval_records = list(
                self._get_current_interval_records(allow_profile_fallback=False) or []
            )
        except Exception:
            interval_records = list(getattr(self, "current_interval_records", []) or [])

        ready = bool(getattr(self, "_current_interval_ready", False))
        segmentation_text = self._read_status_text(getattr(self, "segmentation_status_var", None))
        if ready:
            division_status = "成功"
        elif "失败" in segmentation_text:
            division_status = "失败"
        elif "正在" in segmentation_text:
            division_status = "进行中"
        else:
            division_status = "未运行"

        state_counts = {state: 0 for state in self._SEGMENT_STATE_LABELS}
        for record in interval_records:
            try:
                segment_type, _state_code, _state_label = self._get_interval_state_display(record)
            except Exception:
                continue
            state_counts[segment_type] = state_counts.get(segment_type, 0) + 1
        count_rows = (
            ("idle", "entry", "steady"),
            ("transition", "exit", "nonsteady"),
        )
        count_text = "\n　　　　　　".join(
            " | ".join(
                f"{self._SEGMENT_STATE_LABELS[state][1]} {state_counts.get(state, 0)}"
                for state in row
            )
            for row in count_rows
        )

        total = len(interval_records)
        if bool(getattr(self, "release_mode", False)):
            if ready:
                result_text = f"处理状态：已完成，共 {total} 个区间；可以查看图表并保存结果"
            elif division_status == "失败":
                result_text = "处理状态：未完成，请检查工艺信息文件后重试"
            elif division_status == "进行中":
                result_text = "处理状态：正在计算，请稍候"
            else:
                result_text = "处理状态：等待导入工艺信息文件"
            self._set_overview_text(
                getattr(self, "interval_overview_success_var", None), result_text
            )

            mapping_state = str(getattr(self, "_sample_mapping_status", "") or "")
            if mapping_state == "valid":
                mapping_source = ""
                for record in interval_records:
                    mapping_source = str(record.get("mapping_source", "") or "")
                    if mapping_source:
                        break
                if mapping_source == "journey_order_ratio_missing_n":
                    mapping_text = "实际负载：已匹配（工艺文件没有行号，已按加工顺序匹配）"
                elif mapping_source == "program_line_and_point_order_quantized":
                    mapping_text = "实际负载：已匹配（短区间已自动对齐到相邻采样点）"
                else:
                    mapping_text = "实际负载：已匹配，可以查看叠加图"
            elif mapping_state == "failed":
                mapping_text = "实际负载：匹配失败，请检查工艺行号和 SampleData"
            elif mapping_state == "pending":
                mapping_text = "实际负载：等待与工艺信息匹配"
            else:
                mapping_text = "实际负载：尚未读取 SampleData"
            self._set_overview_text(
                getattr(self, "interval_overview_mapping_var", None), mapping_text
            )
            self._set_overview_text(getattr(self, "interval_count_var", None), str(total))
            return

        self._set_overview_text(
            getattr(self, "interval_overview_success_var", None),
            f"当前划分：{division_status}",
        )
        self._set_overview_text(
            getattr(self, "interval_overview_total_var", None),
            f"总区间数：{total}",
        )
        self._set_overview_text(
            getattr(self, "interval_overview_counts_var", None),
            f"六类数量：{count_text}",
        )
        mapping_text = self._read_status_text(getattr(self, "sample_mapping_status_var", None))
        self._set_overview_text(
            getattr(self, "interval_overview_mapping_var", None),
            self._prefixed_overview_status(
                "采样映射", mapping_text, "未导入实际采样文件"
            ),
        )
        self._set_overview_text(getattr(self, "interval_count_var", None), str(total))

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
        """旧调用兼容入口；精简页面固定使用单图模式。"""
        self.sample_plot_mode.set("overlay")
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
