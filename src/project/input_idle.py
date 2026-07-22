from __future__ import annotations

import time
import csv
import bisect

from .shared import *


class InputIdleMixin:
    def _cancel_pending_input_process(self):
        pending_job = getattr(self, "_input_process_job", None)
        if pending_job is None:
            return
        try:
            self.root.after_cancel(pending_job)
        except Exception:
            pass
        self._input_process_job = None

    def _schedule_input_process_preview(self, delay_ms=50):
        self._cancel_pending_input_process()

        def _run():
            self._input_process_job = None
            self._process_current_input_for_preview()

        self._input_process_job = self.root.after(int(delay_ms), _run)

    def set_input_files(self, file_paths):
        """设置导入的工艺信息表"""
        file_paths = [p for p in file_paths if p]
        self.input_file_paths = file_paths
        self.merged_input_file_path = ""
        self.ensure_sample_data_matches_inputs(file_paths)
        self.reset_processing_state()
        self._refresh_import_order_controls()
        if len(file_paths) >= 1:
            effective_input_path = file_paths[0]
            if len(file_paths) > 1:
                try:
                    effective_input_path = self._build_merged_process_input_file(file_paths)
                except Exception as e:
                    self.input_file_paths = []
                    self.merged_input_file_path = ""
                    self.input_file_path.set("")
                    self.input_file_count_var.set("")
                    if hasattr(self, "matched_process_file_var"):
                        self.matched_process_file_var.set("工艺信息表合并失败")
                    self.set_sample_controls_enabled(True, refresh=False)
                    messagebox.showerror("导入失败", f"拼接工艺信息文件时发生错误:\n{str(e)}")
                    return ""

            self.input_file_path.set(effective_input_path)
            if hasattr(self, "matched_process_file_var"):
                if len(file_paths) == 1:
                    self.matched_process_file_var.set(effective_input_path)
                else:
                    self.matched_process_file_var.set(f"多文件已合并：{os.path.basename(effective_input_path)}")
            if len(file_paths) == 1:
                self.input_file_count_var.set("")
            else:
                self.input_file_count_var.set(f"已选择 {len(file_paths)} 个工艺信息表（已按序号合并）")
            self.set_sample_controls_enabled(True, refresh=False)
            sample_dir = self._resolve_input_source_dir(file_paths, effective_input_path)
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
                    self.sample_auto_status_var.set("未找到SampleData，可手动导入实验实测文件；当前保留已加载实测数据")
                if hasattr(self, "status_var_data"):
                    self.status_var_data.set("未找到SampleData，可手动导入实验实测文件；当前保留已加载实测数据")
            if self.sample_data_loaded:
                self.show_sample_preview()
            else:
                self.show_initial_message()
            if not getattr(self, "_loading_sample_data", False):
                self._schedule_input_process_preview(delay_ms=50)
            return effective_input_path
        else:
            self._cancel_pending_input_process()
            self.input_file_path.set("")
            self.input_file_count_var.set("")
            if hasattr(self, "matched_process_file_var"):
                self.matched_process_file_var.set("未绑定工艺信息表")
            if self.sample_data_loaded:
                self.show_sample_preview()
            else:
                self.show_initial_message()
            self.set_sample_controls_enabled(True, refresh=False)
            self._refresh_import_order_controls()
            return ""

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
            filetypes=(("工艺信息文件", "*.txt *.csv"), ("文本文件", "*.txt"), ("CSV文件", "*.csv"), ("所有文件", "*.*"))
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

    def _resolve_input_source_dir(self, file_paths, fallback_path=""):
        """解析工艺信息源目录，优先返回共同目录。"""
        source_dirs = []
        seen = set()
        for path in file_paths or []:
            if not path:
                continue
            try:
                abs_dir = os.path.dirname(os.path.abspath(path))
            except Exception:
                continue
            norm_dir = os.path.normcase(os.path.normpath(abs_dir))
            if norm_dir in seen:
                continue
            seen.add(norm_dir)
            source_dirs.append(abs_dir)

        if len(source_dirs) == 1:
            return source_dirs[0]
        if source_dirs:
            return source_dirs[0]
        if fallback_path:
            try:
                return os.path.dirname(os.path.abspath(fallback_path))
            except Exception:
                pass
        return ""

    def _extract_process_sequence_for_merge(self, raw_line):
        """提取工艺信息首列序号，用于多文件拼接排序。"""
        text = str(raw_line or "").strip().lstrip('\ufeff')
        if not text:
            return None

        token = ""
        if "," in text:
            try:
                parts = next(csv.reader([text]))
            except Exception:
                parts = None
            if parts:
                token = str(parts[0]).strip().lstrip('\ufeff')
        if not token:
            parts = text.split()
            if parts:
                token = str(parts[0]).strip().lstrip('\ufeff')
        if not token:
            return None
        if token.upper().startswith('N'):
            token = token[1:]
        try:
            return float(token)
        except Exception:
            return None

    def _build_merged_process_input_file(self, file_paths):
        """多工艺信息文件按序号合并为单个文本文件，仅保留一个表头。"""
        valid_paths = []
        for path in file_paths or []:
            if not path:
                continue
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                valid_paths.append(abs_path)

        if not valid_paths:
            raise ValueError("未找到可合并的工艺信息文件")
        if len(valid_paths) == 1:
            self.merged_input_file_path = ""
            return valid_paths[0]

        header_line = ""
        ordered_rows = []
        row_order = 0
        for path in valid_paths:
            encoding = self.detect_file_encoding(path)
            with open(path, 'r', encoding=encoding, errors='ignore') as infile:
                for raw_line in infile:
                    line = str(raw_line or "").strip()
                    if not line:
                        continue
                    normalized_line = line.lstrip('\ufeff')
                    seq = self._extract_process_sequence_for_merge(normalized_line)
                    if seq is None:
                        if not header_line:
                            header_line = normalized_line
                        continue
                    ordered_rows.append((seq, row_order, normalized_line))
                    row_order += 1

        if not ordered_rows:
            raise ValueError("所选工艺信息文件中未解析到有效数据行")

        ordered_rows.sort(key=lambda item: (item[0], item[1]))
        output_dir = self._resolve_input_source_dir(valid_paths) or str(OUTPUT_DIR)
        os.makedirs(output_dir, exist_ok=True)
        first_name = os.path.splitext(os.path.basename(valid_paths[0]))[0]
        safe_name = re.sub(r'[\\/:*?"<>|]+', '_', first_name) or "process_info"
        merged_path = os.path.join(output_dir, f"{safe_name}_merged.txt")

        with open(merged_path, 'w', encoding='utf-8', newline='\n') as outfile:
            if header_line:
                outfile.write(f"{header_line}\n\n")
            for _, _, line in ordered_rows:
                outfile.write(f"{line}\n")

        self.merged_input_file_path = merged_path
        return merged_path

    def load_sample_bundle_from_dir(self, base_dir, silent=False):
        """按目录自动加载 SampleData"""
        resolved_dir, csv_path, txt_path = self.resolve_sampledata_files(base_dir)
        if not resolved_dir:
            if not silent:
                messagebox.showerror(
                    "文件缺失",
                    "未找到 SampleData.csv 或 SampleData.txt（可放在同目录或 SampleData 子目录）\n可改用“导入实验实测”手动导入实验文件"
                )
            if hasattr(self, "sample_auto_status_var"):
                self.sample_auto_status_var.set("未找到SampleData.csv或SampleData.txt，可手动导入实验实测文件")
            if hasattr(self, "status_var_data"):
                self.status_var_data.set("未发现SampleData，已跳过自动导入；可手动导入实验实测文件")
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
        # 从项目示例数据目录加载 SampleData（而非依赖当前工作目录）。
        success = self.load_sample_bundle_from_dir(str(SAMPLE_DATA_DIR), silent=True)
        if success and self.get_input_files():
            # 自动执行处理
            self.root.after(100, self._auto_process_after_load)

    def _get_process_cache_key(self, input_file):
        if not input_file:
            return None
        return os.path.normcase(os.path.abspath(input_file))

    def _get_process_cache_signature(self, input_file):
        """建立工艺信息解析缓存签名。

        该签名只用于判断旧的解析行能否复用，不是六态划分的
        ``process_signature``。后者由划分管线仅基于规范化工艺字段建立。
        """
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
            float(self.get_kc_value()),
            float(self.kc_sigma.get()),
            float(self.get_ke_value()),
            float(self.current_program_speed.get()),
            float(self.current_program_idle_power.get()),
            str(self.gcode_nc_path_var.get()),
            str(self.idle_model_signature),
            str(self.step_feed_model_signature),
            str(self._get_saved_kc_profile_signature(input_file) if hasattr(self, "_get_saved_kc_profile_signature") else ""),
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
            signature = self._get_process_cache_signature(process_path)
            cached = self._process_cache.get(cache_key)
            if not cached:
                return False
            return cached.get("signature") == signature and bool(cached.get("data"))
        except Exception:
            return False

    def _process_current_input_for_preview(self):
        """处理当前工艺信息表，完成过程域划分后刷新图表。"""
        self._cancel_pending_input_process()
        input_files = self.get_input_files()
        if len(input_files) != 1:
            return False
        input_file = input_files[0]
        if not input_file or not os.path.exists(input_file):
            return False
        stage_times = {}
        try:
            total_start = time.perf_counter()
            self.set_progress(5, "正在检查缓存...")
            cache_key = self._get_process_cache_key(input_file)
            signature = self._get_process_cache_signature(input_file)
            if cache_key and self._load_cached_process(cache_key, signature):
                self.set_progress(28, "已命中工艺解析缓存，正在计算过程域特征...")
                stage_times["cache"] = time.perf_counter() - total_start
                stage_start = time.perf_counter()
                result = self.run_full_path_segmentation(
                    export_outputs=False,
                    refresh_view=False,
                    silent=True,
                )
                stage_times["segmentation"] = time.perf_counter() - stage_start
                if result is None:
                    self.set_progress(0, "过程域六类划分失败")
                    self._refresh_import_order_controls()
                    return False
                self.set_progress(88, "过程域划分完成，正在生成图表...")
                stage_start = time.perf_counter()
                self.generate_plots(
                    save=False,
                    silent=True,
                    interval_policy="reuse_current_template",
                    persist_profile=False,
                    refresh_prediction=False,
                )
                stage_times["plot"] = time.perf_counter() - stage_start
                self.set_progress(96, "正在刷新过程域预览...")
                # 导入工艺信息表后立即刷新“已设定理想值”视图
                self._refresh_ideal_tree()
                self._refresh_current_ideal_display()
                total_elapsed = time.perf_counter() - total_start
                self._refresh_import_order_controls()
                self.set_progress(100, f"已使用解析缓存完成划分（总耗时 {total_elapsed:.1f}s）")
                return True
            self.set_progress(12, "正在解析工艺信息文件...")
            try:
                self.root.update_idletasks()
            except Exception:
                pass
            stage_start = time.perf_counter()
            success = self.process_single_file(input_file)
            stage_times["parse"] = time.perf_counter() - stage_start
            if not success:
                self.set_progress(0, "自动处理失败")
                return False
            self._store_process_cache(cache_key, signature)
            self.set_progress(42, "工艺信息解析完成，正在重算 MRR 与过程域特征...")
            stage_start = time.perf_counter()
            result = self.run_full_path_segmentation(
                export_outputs=False,
                refresh_view=False,
                silent=True,
            )
            stage_times["segmentation"] = time.perf_counter() - stage_start
            if result is None:
                self.set_progress(0, "过程域六类划分失败")
                self._refresh_import_order_controls()
                return False
            self.set_progress(88, "过程域划分完成，正在生成图表...")
            stage_start = time.perf_counter()
            self.generate_plots(
                save=False,
                silent=True,
                interval_policy="reuse_current_template",
                persist_profile=False,
                refresh_prediction=False,
            )
            stage_times["plot"] = time.perf_counter() - stage_start
            self.set_progress(96, "正在刷新过程域预览...")
            # 导入工艺信息表后立即刷新“已设定理想值”视图
            self._refresh_ideal_tree()
            self._refresh_current_ideal_display()
            total_elapsed = time.perf_counter() - total_start
            summary_parts = []
            for key, label in (
                ("parse", "解析"),
                ("segmentation", "划分"),
                ("plot", "图表"),
            ):
                if key in stage_times:
                    summary_parts.append(f"{label}{stage_times[key]:.1f}s")
            summary_text = "，".join(summary_parts)
            final_text = f"自动处理完成（总耗时 {total_elapsed:.1f}s"
            if summary_text:
                final_text += f"，{summary_text}"
            final_text += "）"
            self._refresh_import_order_controls()
            self.set_progress(100, final_text)
            return True
        except Exception as e:
            self.set_progress(0, f"自动处理出错: {str(e)[:50]}")
            return False
        finally:
            self.root.after(1200, self.reset_progress)

    def _auto_process_after_load(self):
        """加载数据后自动执行处理"""
        self._process_current_input_for_preview()

    def _refresh_import_order_controls(self):
        """分别按过程域和采样域的就绪状态更新按钮。"""
        has_process_data = bool(getattr(self, "data", None))
        has_segmentation = bool(
            getattr(self, "_current_interval_ready", False)
            and str(getattr(self, "_current_interval_source", "") or "") == "segmentation"
        )

        if hasattr(self, "import_sample_btn"):
            self.import_sample_btn.configure(state="normal")
        if hasattr(self, "import_experiment_btn"):
            self.import_experiment_btn.configure(state="normal")
        if hasattr(self, "choose_process_btn"):
            self.choose_process_btn.configure(state="normal")
        if hasattr(self, "run_segmentation_btn"):
            self.run_segmentation_btn.configure(state=("normal" if has_process_data else "disabled"))
        if hasattr(self, "export_i_code_btn"):
            self.export_i_code_btn.configure(state=("normal" if has_segmentation else "disabled"))

    def _refresh_segmentation_export_controls(self):
        """映射状态更新后同步过程域操作按钮。"""
        self._refresh_import_order_controls()

    def _ensure_nc_loaded_before_measurement(self):
        return True

    def _ensure_ready_for_process_info_import(self):
        """工艺信息可以独立导入并完成过程域划分。"""
        return True

    def browse_sample_bundle(self):
        """导入 SampleData.csv/txt"""
        if not self._ensure_nc_loaded_before_measurement():
            return
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
        self._refresh_import_order_controls()

    def browse_experiment_measurement_file(self):
        """手动导入实验实测文件。"""
        if not self._ensure_nc_loaded_before_measurement():
            return
        file_path = filedialog.askopenfilename(
            title="选择实验实测文件",
            filetypes=(
                ("实验实测文件", ("*.csv", "*.txt", "*.dat", "*.fxt")),
                ("所有文件", "*.*"),
            )
        )
        if not file_path:
            return
        self.load_experiment_measurement_file(file_path, silent=False)
        self._refresh_import_order_controls()

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
        origin = (
            float(self.origin_x.get()),
            float(self.origin_y.get()),
            float(self.origin_z.get()),
        )
        prev_coords = origin
        modal_state = self._create_modal_gcode_state()
        command_speed = 0.0
        current_feed = 0.0
        spindle_on = False
        cumulative_path = 0.0
        command_speeds = []
        line_speeds = []
        active_line_speeds = []
        line_feeds = []
        states = []
        state_by_line_index = {}
        state_by_n = {}
        trajectory_segments = []
        tool_diameter = None
        seen_axes = {"x": False, "y": False, "z": False}

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
                display_ready_before = all(seen_axes.values())
                axis_updated = {
                    "x": bool(re.search(r'X([-+]?\d*\.?\d+)', line)),
                    "y": bool(re.search(r'Y([-+]?\d*\.?\d+)', line)),
                    "z": bool(re.search(r'Z([-+]?\d*\.?\d+)', line)),
                }
                motion_info = self.compute_gcode_motion_info(
                    line,
                    prev_coords=prev_coords,
                    prev_state=modal_state
                )
                modal_state = motion_info["state"]
                start_coords = tuple(motion_info["start_coords"])
                current_coords = tuple(motion_info["end_coords"])
                moved = bool(float(motion_info["segment_length"]) > 1e-9)
                path_start = float(cumulative_path)
                cumulative_path += float(motion_info["segment_length"])
                path_end = float(cumulative_path)
                for axis_name, updated in axis_updated.items():
                    if updated:
                        seen_axes[axis_name] = True

                state = {
                    "file_line_index": int(line_index),
                    "n_value": n_value,
                    "line_text": line,
                    "command_speed": float(command_speed),
                    "active_speed": float(active_speed),
                    "feed": float(current_feed),
                    "spindle_on": bool(spindle_on),
                    "x": float(current_coords[0]),
                    "y": float(current_coords[1]),
                    "z": float(current_coords[2]),
                    "start_x": float(start_coords[0]),
                    "start_y": float(start_coords[1]),
                    "start_z": float(start_coords[2]),
                    "path_start": float(path_start),
                    "path_end": float(path_end),
                    "path_length": float(path_end - path_start),
                    "motion_type": motion_info.get("motion_type"),
                }
                states.append(state)
                state_by_line_index[int(line_index)] = state
                if n_value is not None:
                    state_by_n[int(n_value)] = state

                if moved:
                    is_initial_jump = (
                        len(trajectory_segments) == 0
                        and all(abs(prev_coords[idx] - origin[idx]) <= 1e-9 for idx in range(3))
                    )
                    trajectory_segments.append({
                        "start_x": float(start_coords[0]),
                        "start_y": float(start_coords[1]),
                        "start_z": float(start_coords[2]),
                        "end_x": float(current_coords[0]),
                        "end_y": float(current_coords[1]),
                        "end_z": float(current_coords[2]),
                        "file_line_index": int(line_index),
                        "n_value": n_value,
                        "command_speed": float(command_speed),
                        "active_speed": float(active_speed),
                        "feed": float(current_feed),
                        "spindle_on": bool(spindle_on),
                        "is_initial_jump": bool(is_initial_jump),
                        "display_ok": bool(display_ready_before),
                        "path_start": float(path_start),
                        "path_end": float(path_end),
                        "path_length": float(path_end - path_start),
                        "motion_type": motion_info.get("motion_type"),
                    })

                if command_speed > 0:
                    command_speeds.append(float(command_speed))
                    line_speeds.append(float(command_speed))
                if active_speed > 0:
                    active_line_speeds.append(float(active_speed))
                if current_feed > 0:
                    line_feeds.append(float(current_feed))
                prev_coords = current_coords

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
            "sorted_line_indices": sorted(state_by_line_index.keys()),
            "sorted_n_values": sorted(state_by_n.keys()),
            "segments": segments,
            "trajectory_segments": trajectory_segments,
            "trajectory_origin": origin,
            "tool_diameter": tool_diameter,
            "tool_radius": (float(tool_diameter) / 2.0) if tool_diameter is not None else None,
            "total_path_length": float(cumulative_path),
        }

    def _resolve_nc_state_for_process_row(self, raw_line_number, gcode_content):
        if not self.gcode_profile:
            return None

        n_value = self.extract_n_value(str(gcode_content or ""))
        n_int = self.extract_n_integer(n_value)
        if n_int is not None:
            state = self.gcode_profile.get("state_by_n", {}).get(int(n_int))
            if state:
                return dict(state)

        if raw_line_number is not None:
            candidate_n = int(raw_line_number) + 1
            state = self.gcode_profile.get("state_by_n", {}).get(candidate_n)
            if state:
                return dict(state)

            sorted_n_values = self.gcode_profile.get("sorted_n_values", [])
            if sorted_n_values:
                insert_idx = bisect.bisect_right(sorted_n_values, candidate_n) - 1
                if insert_idx >= 0:
                    base_state = self.gcode_profile["state_by_n"][sorted_n_values[insert_idx]]
                    fallback_pos = float(base_state.get("path_end", 0.0) or 0.0)
                    return {
                        **dict(base_state),
                        "path_start": fallback_pos,
                        "path_end": fallback_pos,
                        "path_length": 0.0,
                        "motion_type": base_state.get("motion_type"),
                        "matched_by_previous": True,
                    }

        if raw_line_number is not None:
            state = self.gcode_profile.get("state_by_line_index", {}).get(int(raw_line_number))
            if state:
                return dict(state)
            sorted_line_indices = self.gcode_profile.get("sorted_line_indices", [])
            if sorted_line_indices:
                insert_idx = bisect.bisect_right(sorted_line_indices, int(raw_line_number)) - 1
                if insert_idx >= 0:
                    base_state = self.gcode_profile["state_by_line_index"][sorted_line_indices[insert_idx]]
                    fallback_pos = float(base_state.get("path_end", 0.0) or 0.0)
                    return {
                        **dict(base_state),
                        "path_start": fallback_pos,
                        "path_end": fallback_pos,
                        "path_length": 0.0,
                        "motion_type": base_state.get("motion_type"),
                        "matched_by_previous": True,
                    }
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

    def _get_idle_power_calibration_rows(self):
        model = self.idle_power_model if isinstance(self.idle_power_model, dict) else {}
        grouped_points = model.get("grouped_points") or []
        rows = []

        if grouped_points:
            for item in grouped_points:
                try:
                    speed = float(item.get("speed"))
                    idle_power = float(item.get("power"))
                except Exception:
                    continue
                if not np.isfinite(speed) or speed <= 0.0 or not np.isfinite(idle_power):
                    continue
                rows.append({
                    "speed": float(speed),
                    "idle_power": float(idle_power),
                    "file_count": int(item.get("file_count", 0) or 0),
                    "total_files": int(item.get("total_files", 0) or 0),
                })
        else:
            speeds = np.asarray(model.get("speeds", []), dtype=float) if model else np.empty(0, dtype=float)
            powers = np.asarray(model.get("powers", []), dtype=float) if model else np.empty(0, dtype=float)
            if speeds.size and powers.size:
                finite_mask = np.isfinite(speeds) & np.isfinite(powers)
                speeds = speeds[finite_mask]
                powers = powers[finite_mask]
                if speeds.size > 1:
                    order = np.argsort(speeds, kind="mergesort")
                    speeds = speeds[order]
                    powers = powers[order]
                for speed, idle_power in zip(speeds.tolist(), powers.tolist()):
                    if float(speed) <= 0.0:
                        continue
                    rows.append({
                        "speed": float(speed),
                        "idle_power": float(idle_power),
                        "file_count": 0,
                        "total_files": 0,
                    })

        rows.sort(key=lambda item: float(item["speed"]))
        return rows

    def _build_program_idle_detail_rows(self):
        calibration_rows = self._get_idle_power_calibration_rows()
        gcode_rows = self._get_program_idle_power_rows()
        if not calibration_rows:
            detail_rows = []
            for row in gcode_rows:
                segment_count = int(row.get("segment_count", 0) or 0)
                note = "当前未加载空载模型，沿用已保存 P_idle"
                if segment_count > 0:
                    note = f"{note}；当前 G 代码涉及 {segment_count} 段"
                detail_rows.append({
                    "speed": float(row["speed"]),
                    "idle_power": float(row["idle_power"]),
                    "source": "当前G代码转速",
                    "note": note,
                    "row_kind": "fallback",
                })
            detail_rows.sort(key=lambda item: float(item["speed"]))
            return detail_rows

        fallback_idle, lookup_speeds, lookup_powers = self._resolve_idle_power_lookup()
        del fallback_idle
        calibration_by_speed = {
            round(float(row["speed"]), 6): row
            for row in calibration_rows
        }
        gcode_by_speed = {
            round(float(row["speed"]), 6): row
            for row in gcode_rows
        }

        detail_rows = []
        for calibration in calibration_rows:
            rounded_speed = round(float(calibration["speed"]), 6)
            gcode_row = gcode_by_speed.get(rounded_speed)
            file_count = int(calibration.get("file_count", 0) or 0)
            total_files = int(calibration.get("total_files", 0) or 0)
            source = "标定值"
            note = "空载辨识标定值"
            row_kind = "calibration"

            if file_count > 0 and total_files > 0:
                note = f"{note}（{file_count}/{total_files} 个文件）"
            elif file_count > 0:
                note = f"{note}（{file_count} 个文件）"

            if gcode_row:
                source = "当前G代码转速"
                row_kind = "current_speed"
                segment_count = int(gcode_row.get("segment_count", 0) or 0)
                if segment_count > 0:
                    note = f"当前 G 代码使用该标定转速（涉及 {segment_count} 段）"
                else:
                    note = "当前 G 代码使用该标定转速"

            detail_rows.append({
                "speed": float(calibration["speed"]),
                "idle_power": float(calibration["idle_power"]),
                "source": source,
                "note": note,
                "row_kind": row_kind,
            })

        for gcode_row in gcode_rows:
            speed = float(gcode_row["speed"])
            rounded_speed = round(speed, 6)
            if rounded_speed in calibration_by_speed:
                continue

            idle_power = float(gcode_row["idle_power"])
            segment_count = int(gcode_row.get("segment_count", 0) or 0)
            note = "当前 G 代码转速"
            row_kind = "interpolated"
            source = "当前G代码转速"

            if lookup_speeds.size >= 2 and lookup_powers.size >= 2:
                if speed < float(lookup_speeds[0]):
                    low_speed = float(lookup_speeds[0])
                    high_speed = float(lookup_speeds[1])
                    row_kind = "extrapolated"
                    note = f"当前 G 代码转速超出标定范围，按 {low_speed:.0f}~{high_speed:.0f} rpm 端点外推"
                elif speed > float(lookup_speeds[-1]):
                    low_speed = float(lookup_speeds[-2])
                    high_speed = float(lookup_speeds[-1])
                    row_kind = "extrapolated"
                    note = f"当前 G 代码转速超出标定范围，按 {low_speed:.0f}~{high_speed:.0f} rpm 端点外推"
                else:
                    insert_idx = int(np.searchsorted(lookup_speeds, speed))
                    low_speed = float(lookup_speeds[insert_idx - 1])
                    high_speed = float(lookup_speeds[insert_idx])
                    note = f"当前 G 代码转速位于 {low_speed:.0f}~{high_speed:.0f} rpm 标定点之间，按线性插值计算"
            elif lookup_speeds.size == 1:
                row_kind = "fallback"
                note = f"仅有 1 个标定点，当前 G 代码沿用 {float(lookup_speeds[0]):.0f} rpm 标定值"

            if segment_count > 0:
                note = f"{note}；当前 G 代码涉及 {segment_count} 段"

            detail_rows.append({
                "speed": speed,
                "idle_power": idle_power,
                "source": source,
                "note": note,
                "row_kind": row_kind,
            })

        detail_rows.sort(
            key=lambda item: (
                float(item["speed"]),
                0 if "标定值" in str(item.get("source", "")) else 1,
            )
        )
        return detail_rows

    def _update_program_idle_detail_button_state(self):
        if not hasattr(self, "program_idle_detail_btn"):
            return
        rows = self._get_program_idle_power_rows()
        has_details = bool(
            self.gcode_profile
            and rows
            and any(float(row.get("idle_power", 0.0) or 0.0) > 0.0 for row in rows)
        )
        self._set_idle_curve_visible(False)
        if has_details:
            self.program_idle_detail_btn.state(["!disabled"])
        else:
            self.program_idle_detail_btn.state(["disabled"])

    def _set_idle_curve_visible(self, visible):
        visible = bool(visible)
        self.idle_curve_visible = visible
        frame = getattr(self, "idle_curve_frame", None)
        if frame is None:
            return
        try:
            if visible:
                frame.grid()
            else:
                frame.grid_remove()
        except Exception:
            pass

    def _draw_idle_power_chart(self, ax, fig, canvas=None, hint_label=None):
        if ax is None or fig is None:
            return

        rows = self._get_program_idle_power_rows()
        model = self.idle_power_model if isinstance(self.idle_power_model, dict) else {}

        model_speeds = np.asarray(model.get("speeds", []), dtype=float) if model else np.empty(0, dtype=float)
        model_powers = np.asarray(model.get("powers", []), dtype=float) if model else np.empty(0, dtype=float)
        if model_speeds.size and model_powers.size:
            finite_mask = np.isfinite(model_speeds) & np.isfinite(model_powers)
            model_speeds = model_speeds[finite_mask]
            model_powers = model_powers[finite_mask]
            if model_speeds.size > 1:
                order = np.argsort(model_speeds, kind="mergesort")
                model_speeds = model_speeds[order]
                model_powers = model_powers[order]

        fallback_idle = 0.0
        for candidate in (getattr(self, "current_program_idle_power", None), getattr(self, "p_idle_var", None)):
            if candidate is None:
                continue
            try:
                value = float(candidate.get())
            except Exception:
                try:
                    value = float(candidate)
                except Exception:
                    continue
            if np.isfinite(value) and value > 0.0:
                fallback_idle = float(value)
                break

        ax.clear()
        fig.patch.set_facecolor(PLOT_FIG_BG)
        ax.set_facecolor(PLOT_AX_BG)
        ax.grid(True, color=PLOT_GRID_COLOR, linestyle="--", linewidth=0.6, alpha=0.75)
        for spine in ax.spines.values():
            spine.set_color(PLOT_SPINE_COLOR)
        ax.tick_params(colors=PLOT_TEXT_COLOR, labelsize=9)
        ax.set_xlabel("S / rpm", fontsize=10, color=PLOT_TEXT_COLOR)
        ax.set_ylabel("P_idle / W", fontsize=10, color=PLOT_TEXT_COLOR)

        if hint_label is not None:
            hint_label.configure(
                text="导入空载辨识文件后显示 P_idle-S 关系；导入 NC 后会叠加当前 G 代码转速"
            )

        if model_speeds.size == 0 and not rows:
            ax.set_title("P_idle-S 关系", fontsize=10, color=PLOT_TEXT_COLOR, pad=8)
            placeholder = "导入空载辨识文件后显示 P_idle-S 关系\n导入 NC 后会在图上标出当前 G 代码对应的转速与空载功率"
            if fallback_idle > 0.0:
                placeholder += f"\n当前已保存 P_idle = {fallback_idle:.3f} W"
            ax.text(
                0.5,
                0.5,
                placeholder,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color=UI_COLOR_TEXT_MUTED,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            fig.subplots_adjust(left=0.08, right=0.985, top=0.88, bottom=0.24)
            if canvas is not None:
                canvas.draw_idle()
            return

        if model_speeds.size == 0 and rows and fallback_idle <= 0.0:
            ax.set_title("P_idle-S 关系", fontsize=10, color=PLOT_TEXT_COLOR, pad=8)
            if hint_label is not None:
                hint_label.configure(text="已识别当前 G 代码转速；请先完成空载功率辨识，再显示对应的 P_idle")
            speed_text = "、".join(f"{float(row['speed']):.0f}" for row in rows[:8])
            if len(rows) > 8:
                speed_text += " ..."
            ax.text(
                0.5,
                0.56,
                "当前 G 代码已识别到以下主轴转速",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color=PLOT_TEXT_COLOR,
            )
            ax.text(
                0.5,
                0.42,
                f"S = {speed_text} rpm",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color=UI_COLOR_PRIMARY_DARK,
            )
            ax.text(
                0.5,
                0.28,
                "待空载辨识后将在图上标出对应的 P_idle",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color=UI_COLOR_TEXT_MUTED,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            fig.subplots_adjust(left=0.08, right=0.985, top=0.88, bottom=0.24)
            if canvas is not None:
                canvas.draw_idle()
            return

        x_values = []
        y_values = []
        handles = []

        if model_speeds.size:
            curve_speeds = model_speeds.copy()
            if rows:
                row_speeds = np.asarray([float(row["speed"]) for row in rows], dtype=float)
                curve_speeds = np.unique(np.concatenate([curve_speeds, row_speeds]))
            curve_speeds = np.asarray(sorted(float(speed) for speed in curve_speeds if float(speed) > 0), dtype=float)
            curve_powers = np.asarray([self.predict_idle_power(speed) for speed in curve_speeds], dtype=float)
            line_artist, = ax.plot(
                curve_speeds,
                curve_powers,
                color=STYLE_PREDICTED["color"],
                linewidth=1.6,
                alpha=0.95,
                label="空载模型",
                zorder=2,
            )
            point_artist = ax.scatter(
                model_speeds,
                model_powers,
                s=28,
                color=STYLE_PREDICTED["color"],
                edgecolors="white",
                linewidths=0.6,
                label="辨识点",
                zorder=3,
            )
            handles.extend([line_artist, point_artist])
            x_values.extend(curve_speeds.tolist())
            y_values.extend(curve_powers.tolist())
            if hint_label is not None:
                if rows:
                    hint_label.configure(text="蓝线/蓝点为空载辨识结果，橙色标记为当前 G 代码各转速对应的 P_idle")
                else:
                    hint_label.configure(text="蓝线/蓝点为空载辨识结果；导入 NC 后会叠加当前 G 代码转速")
        elif rows and fallback_idle > 0.0:
            row_speeds = [float(row["speed"]) for row in rows]
            baseline_artist, = ax.plot(
                [min(row_speeds), max(row_speeds)],
                [fallback_idle, fallback_idle],
                color="#7F8C8D",
                linewidth=1.4,
                linestyle="--",
                label="已保存 P_idle",
                zorder=1,
            )
            handles.append(baseline_artist)
            x_values.extend(row_speeds)
            y_values.extend([fallback_idle, fallback_idle])
            if hint_label is not None:
                hint_label.configure(text="当前未加载空载模型；橙色标记为当前 G 代码转速，功率沿用已保存 P_idle")

        if rows:
            row_speeds = [float(row["speed"]) for row in rows]
            row_powers = [float(row["idle_power"]) for row in rows]
            for speed in row_speeds:
                ax.axvline(speed, color="#F39C12", linestyle="--", linewidth=0.9, alpha=0.16, zorder=0)
            row_artist = ax.scatter(
                row_speeds,
                row_powers,
                s=62,
                color="#F39C12",
                edgecolors="white",
                linewidths=0.8,
                label="当前 G 代码",
                zorder=5,
            )
            handles.append(row_artist)
            x_values.extend(row_speeds)
            y_values.extend(row_powers)

            for idx, row in enumerate(rows):
                speed = float(row["speed"])
                idle_power = float(row["idle_power"])
                direction = 1 if idx % 2 == 0 else -1
                x_offset = 10 * direction
                y_offset = 12 + 10 * (idx % 3)
                ax.annotate(
                    f"S={speed:.0f}\n{idle_power:.1f}W",
                    xy=(speed, idle_power),
                    xytext=(x_offset, y_offset),
                    textcoords="offset points",
                    ha="left" if direction > 0 else "right",
                    va="bottom",
                    fontsize=8,
                    color=PLOT_TEXT_COLOR,
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#F39C12", alpha=0.96),
                    arrowprops=dict(arrowstyle="-", color="#F39C12", lw=0.8, alpha=0.9),
                    zorder=6,
                    clip_on=False,
                )

        title = "P_idle-S 关系"
        if rows:
            title += "（橙色为当前 G 代码）"
        ax.set_title(title, fontsize=10, color=PLOT_TEXT_COLOR, pad=8)

        if x_values and y_values:
            x_min = min(x_values)
            x_max = max(x_values)
            y_min = min(y_values)
            y_max = max(y_values)

            x_span = x_max - x_min
            y_span = y_max - y_min
            x_pad = max(120.0, x_span * 0.10) if x_span > 1e-9 else max(200.0, abs(x_max) * 0.08)
            y_pad = max(4.0, y_span * 0.18) if y_span > 1e-9 else max(6.0, abs(y_max) * 0.10, 4.0)

            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=3))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))

        if handles:
            unique_handles = []
            unique_labels = []
            for handle in handles:
                label = handle.get_label()
                if not label or label == "_nolegend_" or label in unique_labels:
                    continue
                unique_handles.append(handle)
                unique_labels.append(label)
            if unique_handles:
                ax.legend(unique_handles, unique_labels, loc="best", fontsize=8, framealpha=0.95)

        fig.subplots_adjust(left=0.08, right=0.985, top=0.88, bottom=0.24)
        if canvas is not None:
            canvas.draw_idle()

    def _refresh_idle_power_chart(self):
        self._set_idle_curve_visible(False)
        ax = getattr(self, "ax_idle_curve", None)
        fig = getattr(self, "fig_idle_curve", None)
        canvas = getattr(self, "canvas_idle_curve", None)
        hint_label = getattr(self, "idle_curve_hint_label", None)
        if ax is None or fig is None or canvas is None:
            return
        self._draw_idle_power_chart(ax, fig, canvas=canvas, hint_label=hint_label)

    def _update_program_idle_summary(self):
        rows = self._get_program_idle_power_rows()

        if not self.gcode_profile:
            idle_power = float(self.current_program_idle_power.get() or self.p_idle_var.get() or 0.0)
            if idle_power > 0:
                self.current_program_idle_power_display.set(f"{idle_power:.3f} W")
            else:
                self.current_program_idle_power_display.set("未计算")
            self._update_program_idle_detail_button_state()
            self._refresh_idle_power_chart()
            return

        if not rows:
            self.current_program_idle_power_display.set("未识别到S指令")
            self._update_program_idle_detail_button_state()
            self._refresh_idle_power_chart()
            return

        if not self.idle_power_model:
            powers = [
                float(row["idle_power"])
                for row in rows
                if np.isfinite(float(row.get("idle_power", 0.0) or 0.0)) and float(row.get("idle_power", 0.0) or 0.0) > 0.0
            ]
            if powers:
                if len(rows) == 1:
                    self.current_program_idle_power_display.set(f"{powers[0]:.3f} W (沿用已保存P_idle)")
                else:
                    self.current_program_idle_power_display.set(
                        f"{len(rows)}档转速，沿用已保存P_idle {min(powers):.1f}~{max(powers):.1f} W"
                    )
            elif len(rows) == 1:
                self.current_program_idle_power_display.set("已识别1档转速，待空载辨识")
            else:
                self.current_program_idle_power_display.set(f"已识别{len(rows)}档转速，待空载辨识")
            self._update_program_idle_detail_button_state()
            self._refresh_idle_power_chart()
            return

        if len(rows) == 1:
            self.current_program_idle_power_display.set(f"{rows[0]['idle_power']:.3f} W")
        else:
            powers = [row["idle_power"] for row in rows]
            self.current_program_idle_power_display.set(
                f"{len(rows)}档转速，空载 {min(powers):.1f}~{max(powers):.1f} W"
            )
        self._update_program_idle_detail_button_state()
        self._refresh_idle_power_chart()

    def show_program_idle_detail_dialog(self):
        rows = self._get_program_idle_power_rows()
        if not self.gcode_profile:
            messagebox.showwarning("提示", "请先导入G代码NC文件")
            return
        if not rows:
            messagebox.showwarning("提示", "当前G代码中没有可用的主轴转速")
            return
        self._set_idle_curve_visible(False)
        detail_rows = self._build_program_idle_detail_rows()
        calibration_rows = self._get_idle_power_calibration_rows()

        dialog = tk.Toplevel(self.root)
        dialog.title("程序空载功率明细")
        dialog.geometry("980x720")
        dialog.minsize(760, 560)
        dialog.transient(self.root)
        dialog.grab_set()

        main_frame = ttk.Frame(dialog, padding=12)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(3, weight=1)
        main_frame.grid_rowconfigure(4, weight=1)

        gcode_name = os.path.basename(self.gcode_nc_path_var.get().strip()) if self.gcode_nc_path_var.get().strip() else "未命名NC"
        ttk.Label(
            main_frame,
            text=f"G代码: {gcode_name}",
            font=UI_FONT_BOLD,
            foreground=UI_COLOR_PRIMARY_DARK
        ).grid(row=0, column=0, sticky="w", pady=(0, 4))
        ttk.Label(
            main_frame,
            text=(
                f"下表列出 {len(calibration_rows)} 个标定值，"
                f"并按转速位置标出本次 G 代码的 {len(rows)} 档转速"
                if calibration_rows else
                f"共识别 {len(rows)} 档转速，对应空载功率如下"
            ),
            font=UI_FONT_NORMAL
        ).grid(row=1, column=0, sticky="w", pady=(0, 8))
        if not self.idle_power_model:
            ttk.Label(
                main_frame,
                text="当前未加载空载模型，以下结果沿用已保存的 P_idle 估算",
                font=UI_FONT_SMALL,
                foreground=UI_COLOR_TEXT_MUTED
            ).grid(row=2, column=0, sticky="w", pady=(0, 8))

        chart_frame = ttk.LabelFrame(main_frame, text="P_idle-S 图", padding=(6, 4), style='Tech.TLabelframe')
        chart_frame.grid(row=3, column=0, sticky="nsew", pady=(0, 10))
        chart_frame.grid_columnconfigure(0, weight=1)
        chart_frame.grid_rowconfigure(1, weight=1)

        chart_hint_label = ttk.Label(
            chart_frame,
            text="导入空载辨识文件后显示 P_idle-S 关系；导入 NC 后会叠加当前 G 代码转速",
            font=UI_FONT_SMALL,
            foreground=UI_COLOR_TEXT_MUTED,
        )
        chart_hint_label.grid(row=0, column=0, sticky="w", pady=(0, 4))

        fig_idle_detail, ax_idle_detail = plt.subplots(figsize=(7.4, 3.0), dpi=100)
        fig_idle_detail.patch.set_facecolor(PLOT_FIG_BG)
        ax_idle_detail.set_facecolor(PLOT_AX_BG)
        fig_idle_detail.subplots_adjust(left=0.08, right=0.985, top=0.88, bottom=0.24)

        canvas_idle_detail = FigureCanvasTkAgg(fig_idle_detail, master=chart_frame)
        chart_widget = canvas_idle_detail.get_tk_widget()
        chart_widget.grid(row=1, column=0, sticky="nsew")
        chart_widget.configure(relief=tk.FLAT, bd=0)
        self._draw_idle_power_chart(ax_idle_detail, fig_idle_detail, canvas=canvas_idle_detail, hint_label=chart_hint_label)

        table_frame = ttk.Frame(main_frame)
        table_frame.grid(row=4, column=0, sticky="nsew")
        table_frame.grid_columnconfigure(0, weight=1)
        table_frame.grid_rowconfigure(0, weight=1)

        columns = ("speed", "idle_power", "source", "note")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=min(max(len(detail_rows), 8), 14))
        tree.heading("speed", text="转速 (rpm)")
        tree.heading("idle_power", text="空载功率 (W)")
        tree.heading("source", text="来源")
        tree.heading("note", text="说明")
        tree.column("speed", width=120, anchor="center", stretch=False)
        tree.column("idle_power", width=138, anchor="center", stretch=False)
        tree.column("source", width=150, anchor="center", stretch=False)
        tree.column("note", width=420, anchor="w")
        tree.tag_configure("current_speed", background="#EAF7EA")
        tree.tag_configure("interpolated", background="#FFF4D8")
        tree.tag_configure("extrapolated", background="#FDECEC")
        tree.tag_configure("fallback", background="#EEF3F8")

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        for row in detail_rows:
            tree.insert(
                "",
                "end",
                values=(
                    f"{row['speed']:.1f}",
                    f"{row['idle_power']:.3f}",
                    row.get("source", ""),
                    row.get("note", ""),
                ),
                tags=(row.get("row_kind", ""),),
            )

        def _close_dialog():
            try:
                plt.close(fig_idle_detail)
            except Exception:
                pass
            dialog.destroy()

        dialog.protocol("WM_DELETE_WINDOW", _close_dialog)

        btn_frame = ttk.Frame(dialog, padding=(0, 0, 0, 10))
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="关闭", command=_close_dialog, width=10).pack()
        center_dialog_on_parent(dialog, self.root)

    def _resolve_idle_power_lookup(self):
        """解析空载功率预测所需的回退值与插值表。"""
        fallback_idle = 0.0
        for candidate in (
            getattr(self, "current_program_idle_power", None),
            getattr(self, "p_idle_var", None),
        ):
            if candidate is None:
                continue
            try:
                value = float(candidate.get())
            except Exception:
                try:
                    value = float(candidate)
                except Exception:
                    continue
            if np.isfinite(value) and value > 0:
                fallback_idle = float(value)
                break

        model = self.idle_power_model
        if not model or not model.get("speeds") or not model.get("powers"):
            return float(fallback_idle), np.empty(0, dtype=float), np.empty(0, dtype=float)

        speeds = np.asarray(model["speeds"], dtype=float)
        powers = np.asarray(model["powers"], dtype=float)
        finite_mask = np.isfinite(speeds) & np.isfinite(powers)
        speeds = speeds[finite_mask]
        powers = powers[finite_mask]
        if speeds.size == 0 or powers.size == 0:
            return float(fallback_idle), np.empty(0, dtype=float), np.empty(0, dtype=float)
        if speeds.size > 1:
            order = np.argsort(speeds, kind="mergesort")
            speeds = speeds[order]
            powers = powers[order]
        return float(fallback_idle), speeds, powers

    def _predict_idle_power_from_lookup(self, spindle_speed, fallback_idle, speeds, powers, cache=None):
        """基于已预处理的插值表预测空载功率。"""
        if cache is not None:
            try:
                cache_key = round(float(spindle_speed), 6)
            except Exception:
                cache_key = None
            if cache_key is not None and cache_key in cache:
                return cache[cache_key]

        try:
            speed = float(spindle_speed)
        except Exception:
            speed = float("nan")

        if not np.isfinite(speed) or speed <= 0:
            return float(fallback_idle)

        if speeds.size == 0 or powers.size == 0:
            return float(fallback_idle)
        if len(speeds) == 1:
            predicted = float(powers[0])
            if cache is not None:
                cache[round(speed, 6)] = predicted
            return predicted

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
            predicted = float(y0)
            if cache is not None:
                cache[round(speed, 6)] = predicted
            return predicted
        ratio = (speed - x0) / (x1 - x0)
        predicted = float(y0 + ratio * (y1 - y0))
        if not np.isfinite(predicted) or predicted <= 0:
            return float(fallback_idle)
        if cache is not None:
            cache[round(speed, 6)] = predicted
        return float(predicted)

    def _create_idle_power_predictor(self):
        """为单次批量处理构建带缓存的空载功率预测器。"""
        fallback_idle, speeds, powers = self._resolve_idle_power_lookup()
        cache = {}

        def _predict(spindle_speed):
            return self._predict_idle_power_from_lookup(
                spindle_speed,
                fallback_idle,
                speeds,
                powers,
                cache=cache,
            )

        return _predict

    def predict_idle_power(self, spindle_speed):
        """根据空载功率模型估算指定转速下的空载功率。"""
        fallback_idle, speeds, powers = self._resolve_idle_power_lookup()
        return self._predict_idle_power_from_lookup(spindle_speed, fallback_idle, speeds, powers)

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
        if self.idle_power_model and self.idle_power_model.get("speeds") and self.idle_power_model.get("powers"):
            idle_note = f"，当前程序空载 {idle_power:.3f} W"
        elif idle_power > 0:
            idle_note = f"，沿用已保存P_idle {idle_power:.3f} W"
        else:
            idle_note = "，当前尚无可用P_idle"
        self.gcode_status_var.set(
            f"NC已导入: {os.path.basename(gcode_path)}；识别到 {len(profile.get('unique_speeds', []))} 个转速、"
            f"{len(profile.get('unique_feeds', []))} 个进给，累计行程 {float(profile.get('total_path_length', 0.0) or 0.0):.3f} mm，"
            f"参考转速 {dominant_speed:.1f} rpm{tool_summary}{idle_note}"
        )
        self._update_program_idle_summary()
        if hasattr(self, "refresh_mechanism_status_summary"):
            self.refresh_mechanism_status_summary()
        if hasattr(self, "_schedule_smif_refresh"):
            self._schedule_smif_refresh(delay_ms=0)
        else:
            self.refresh_smif_view()
        return True

    def browse_nc_file(self):
        """可选导入 G 代码 NC，用于辅助主轴转速/空载明细与轨迹展示。"""
        file_path = filedialog.askopenfilename(
            title="选择G代码NC文件",
            filetypes=(("G代码文件", "*.nc;*.cnc;*.gcode;*.txt"), ("所有文件", "*.*"))
        )
        if not file_path:
            return

        try:
            self.gcode_nc_path_var.set(file_path)
            self._refresh_current_program_idle_power_from_gcode()
            status_handled = False
            if hasattr(self, "_handle_kc_profile_after_gcode_import"):
                self._handle_kc_profile_after_gcode_import(file_path)
                status_handled = True
            if not status_handled:
                self.set_status("G代码NC文件已导入", 3000)
            if hasattr(self, "_schedule_smif_refresh"):
                self._schedule_smif_refresh(delay_ms=0)
            else:
                self.refresh_smif_view()
            self._refresh_import_order_controls()
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
                self._update_program_idle_summary()

            committed_idle = float(self.current_program_idle_power.get() or 0.0)
            if committed_idle > 0.0:
                self.p_idle_var.set(committed_idle)

            speed_min = min(speeds)
            speed_max = max(speeds)
            validated_count = sum(1 for item in model.get("grouped_points", []) if int(item.get("file_count", 0)) >= 2)
            self.no_load_status_var.set(
                f"空载功率模型已辨识: {len(speeds)} 点，{len(file_paths)} 个文件，"
                f"5%一致性通过 {validated_count} 档，转速范围 {speed_min:.1f}~{speed_max:.1f} rpm"
            )
            if hasattr(self, "refresh_mechanism_status_summary"):
                self.refresh_mechanism_status_summary()
            self._persist_app_config()
            self.set_status("空载功率辨识完成", 3000)

            if self.get_primary_input_file():
                self._process_current_input_for_preview()
        except Exception as e:
            messagebox.showerror("辨识失败", f"空载功率辨识失败:\n{str(e)}")
