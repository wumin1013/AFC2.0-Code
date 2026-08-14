from __future__ import annotations

import hashlib

from .segmentation import RuleSegmentScorer, SegmentationConfig, SegmentationPipeline
from .prediction_metrics import (
    compute_error_metrics,
    export_table_to_csv,
    get_relative_error_floor,
)
from .shared import *


class AcademicWorkbenchMixin:
    def init_academic_workbench_state(self):
        self.prediction_mode_var = tk.StringVar(value="direct_prediction")
        self.prediction_metrics_var = tk.StringVar(value="预测摘要: 尚未生成")
        self.steady_metric_trim_ratio = 0.10
        self.steady_metric_trim_min_points = 1

        self._cached_steady_intervals = {}
        self._segmentation_config = SegmentationConfig()
        self.segmentation_config = self._segmentation_config
        self.segmentation_pipeline = SegmentationPipeline(self._segmentation_config)
        self._latest_segmentation_result = None
        self._segmentation_running = False

    def _build_segmentation_input_frame(self):
        """将当前工艺行转换为六态模块的最小标准输入。"""
        if not getattr(self, "data", None):
            return pd.DataFrame()
        if hasattr(self, "_ensure_process_point_metadata"):
            self._ensure_process_point_metadata()

        rows = []
        for source_index, row in enumerate(self.data):
            if not isinstance(row, dict) or bool(row.get("_is_synthetic_fill")):
                continue
            input_cumulative = row.get("_input_path_cumulative")
            if input_cumulative is None:
                input_cumulative = row.get("input_path_cumulative")
            path_source = str(row.get("path_source") or "")
            if input_cumulative is None and path_source in {"input_cumulative", "input_incremental"}:
                input_cumulative = row.get("path_end")
            line_id = row.get("line_no_raw")
            if line_id is None:
                line_id = row.get("line_no_aligned")
            if line_id is None:
                line_id = source_index
            rows.append(
                {
                    "point_id": len(rows) + 1,
                    "source_index": int(source_index),
                    "input_path_cumulative": input_cumulative,
                    "path_start": row.get("path_start"),
                    "path_end": row.get("path_end"),
                    "path_source": row.get("path_source", ""),
                    "path_is_physical": row.get("path_is_physical", False),
                    "line_id": line_id,
                    "line_no_raw": row.get("line_no_raw"),
                    "N_str": row.get("N_str"),
                    "ap": row.get("ap"),
                    "ae": row.get("ae"),
                    "F_program": row.get("feed_effective"),
                }
            )
        return pd.DataFrame.from_records(rows)

    def _adapt_segmentation_interval_records(self, result):
        """把完整六态结果适配到现有运行时区间字段，但不改变导出 DataFrame。"""
        intervals = getattr(result, "intervals", None)
        point_labels = getattr(result, "point_labels", None)
        if not isinstance(intervals, pd.DataFrame) or not isinstance(point_labels, pd.DataFrame):
            raise TypeError("SegmentationResult 必须提供 point_labels/intervals DataFrame")
        if intervals.empty and not point_labels.empty:
            raise ValueError("六态划分未生成任何连续区间")

        display_start_bounds = None
        display_end_bounds = None
        if getattr(self, "data", None) and hasattr(self, "compute_line_segment_bounds"):
            process_lines = [
                int(record.get("line_no_aligned", index))
                if record.get("line_no_aligned") is not None else int(index)
                for index, record in enumerate(self.data)
            ]
            blocks = self.compute_sequence_blocks(process_lines) if hasattr(self, "compute_sequence_blocks") else None
            display_start_bounds, display_end_bounds = self.compute_line_segment_bounds(
                process_lines,
                blocks=blocks,
            )

        def _resolve_display_range(start_source_index, end_source_index, start_point, end_point):
            if (
                display_start_bounds is not None
                and display_end_bounds is not None
                and 0 <= start_source_index < len(display_start_bounds)
                and 0 <= end_source_index < len(display_end_bounds)
            ):
                return (
                    float(display_start_bounds[start_source_index]),
                    float(display_end_bounds[end_source_index]),
                )
            return (
                float(start_point.get("line_id", start_source_index)),
                float(end_point.get("line_id", end_source_index)) + 1.0,
            )

        adapted = []
        for row in intervals.to_dict(orient="records"):
            start_pos = int(row.get("start_idx", 0))
            end_pos = int(row.get("end_idx", start_pos))
            start_point = point_labels.iloc[start_pos]
            end_point = point_labels.iloc[end_pos]

            start_source_index = int(row.get("start_source_index", start_point["source_index"]))
            end_source_index = int(row.get("end_source_index", end_point["source_index"]))
            segment_type = str(row.get("segment_type") or "nonsteady")
            start_label = str(row.get("start_label") or start_point["point_label"])
            end_label = str(row.get("end_label") or end_point["point_label"])
            display_start_x, display_end_x = _resolve_display_range(
                start_source_index,
                end_source_index,
                start_point,
                end_point,
            )
            current = dict(row)
            current.update(
                {
                    "zone_id": str(row.get("interval_id") or f"SEG{len(adapted) + 1:04d}"),
                    "start_idx": start_source_index,
                    "end_idx": end_source_index,
                    "start_line": int(row.get("start_line_id", start_point["line_id"])),
                    "end_line": int(row.get("end_line_id", end_point["line_id"])),
                    "start_point_index": int(start_point.get("line_point_index", 0)),
                    "end_point_index": int(end_point.get("line_point_index", 0)),
                    "process_start_label": start_label,
                    "process_end_label": end_label,
                    "start_label": start_label,
                    "end_label": end_label,
                    "interval_range": str(row.get("boundary_label") or f"{start_label}-{end_label}"),
                    "path_start": float(row.get("start_s", start_point["path_start"])),
                    "path_end": float(row.get("end_s", end_point["path_end"])),
                    "display_start_x": display_start_x,
                    "display_end_x": display_end_x,
                    "sample_count": int(row.get("point_count", end_pos - start_pos + 1)),
                    # 仅补齐旧运行时消费者需要的字段。过程域六态结果不再
                    # 携带预测模型或实测辨识量，避免显示层误把它们当作判据。
                    "a_p": float(row.get("ap_mean", 0.0) or 0.0),
                    "a_e": float(row.get("ae_mean", 0.0) or 0.0),
                    "F_plan": float(row.get("F_program_mean", 0.0) or 0.0),
                    "p_idle": float("nan"),
                    "p_pred": float("nan"),
                    "segment_type": segment_type,
                    "state_code": int(row.get("state_code")),
                    "steady_pass": segment_type == "steady",
                    "is_idle_interval": segment_type == "idle",
                    "kc_source": "idle" if segment_type == "idle" else "",
                }
            )
            adapted.append(current)
        return adapted

    def _get_segmentation_sample_lines(self):
        """返回当前实际负载的原始程序行号；没有真实采样序列时拒绝投影。"""
        if not bool(getattr(self, "sample_data_loaded", False)):
            raise ValueError("尚未加载实际负载文件，无法生成实际采样点区间")
        sample_lines = np.asarray(getattr(self, "sample_data_line_numbers", []), dtype=int)
        if sample_lines.size == 0:
            raise ValueError("实际负载文件没有可用的程序行号通道")
        return sample_lines

    @staticmethod
    def _update_segmentation_digest_with_array(digest, name, values):
        """把已解析数组稳定写入签名，避免使用文件路径或对象地址。"""

        array = np.asarray(values)
        digest.update(str(name).encode("utf-8"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii", errors="ignore"))
        if array.dtype.kind == "O":
            hashed = pd.util.hash_array(
                array.reshape(-1),
                categorize=True,
            ).astype(np.uint64, copy=False)
            digest.update(memoryview(np.ascontiguousarray(hashed)).cast("B"))
            return
        contiguous = np.ascontiguousarray(array)
        digest.update(memoryview(contiguous).cast("B"))

    def _get_current_segmentation_process_signature(self):
        signature = str(getattr(self, "_current_process_signature", "") or "")
        if signature:
            return signature
        result = getattr(self, "_latest_segmentation_result", None)
        diagnostics = getattr(result, "diagnostics", None)
        if not isinstance(diagnostics, dict):
            return ""
        signature = str(diagnostics.get("process_signature") or "")
        if not signature:
            repeat = dict(diagnostics.get("repeat_run_consistency") or {})
            signature = str(repeat.get("input_signature") or "")
        return signature

    def _build_segmentation_mapping_signature(self):
        """建立只属于采样投影的签名，不反向污染过程域签名。"""

        process_signature = self._get_current_segmentation_process_signature()
        if not process_signature:
            return ""
        if not bool(getattr(self, "sample_data_loaded", False)):
            return ""
        sample_lines = np.asarray(
            getattr(self, "sample_data_line_numbers_raw", None)
            if getattr(self, "sample_data_line_numbers_raw", None) is not None
            else getattr(self, "sample_data_line_numbers", []),
        )
        sample_values = np.asarray(
            getattr(self, "sample_data_values_raw", None)
            if getattr(self, "sample_data_values_raw", None) is not None
            else getattr(self, "sample_data_values", []),
        )
        program_numbers = np.asarray(
            getattr(self, "sample_data_program_numbers_raw", None)
            if getattr(self, "sample_data_program_numbers_raw", None) is not None
            else getattr(self, "sample_data_program_numbers", []),
        )
        if sample_lines.size == 0:
            return ""
        digest = hashlib.sha256()
        digest.update(b"process-to-sample-line-point-v2")
        digest.update(process_signature.encode("ascii", errors="ignore"))
        digest.update(str(getattr(self, "sample_data_mode", "") or "").encode("utf-8"))
        self._update_segmentation_digest_with_array(digest, "program_line", sample_lines)
        self._update_segmentation_digest_with_array(digest, "program_number", program_numbers)
        self._update_segmentation_digest_with_array(digest, "sample_values", sample_values)
        return digest.hexdigest()

    def _set_segmentation_mapping_status(
        self,
        status,
        *,
        reason="",
        signature="",
        projected_records=None,
    ):
        status = str(status or "pending")
        reason = str(reason or "")
        self._sample_mapping_status = status
        self._current_mapping_signature = str(signature or "")
        records = [
            dict(record)
            for record in (projected_records or [])
            if isinstance(record, dict)
        ]
        if status == "valid":
            self._segmentation_sample_projection_records = records
        else:
            self._segmentation_sample_projection_records = []

        mapping_source = str(
            records[0].get("mapping_source", "")
            if records
            else ""
        ).strip()
        if mapping_source == "journey_order_ratio_missing_n":
            valid_status_text = (
                f"采样映射: 成功（{len(records)} 段；"
                "按行程顺序比例映射，N列缺失）"
            )
        elif mapping_source == "program_line_and_point_order_quantized":
            valid_status_text = (
                f"采样映射: 成功（{len(records)} 段；"
                "短区间按相邻采样点量化）"
            )
        else:
            valid_status_text = f"采样映射: 成功（{len(records)} 段）"
        status_text = {
            "not_available": "采样映射: 未导入实际采样文件",
            "pending": "采样映射: 待建立",
            "valid": valid_status_text,
            "failed": f"采样映射: 失败（{reason or '未知原因'}）",
        }.get(status, f"采样映射: {status}")
        status_var = getattr(self, "sample_mapping_status_var", None)
        if status_var is not None and hasattr(status_var, "set"):
            status_var.set(status_text)

        result = getattr(self, "_latest_segmentation_result", None)
        diagnostics = getattr(result, "diagnostics", None)
        if isinstance(diagnostics, dict):
            if status == "valid" and records:
                covered_count = int(sum(
                    int(record.get("sample_count", 0) or 0)
                    for record in records
                ))
                sample_lines = getattr(self, "sample_data_line_numbers", None)
                sample_count = int(len(sample_lines)) if sample_lines is not None else 0
                diagnostics["sample_projection"] = {
                    "valid": True,
                    "status": status,
                    "coordinate_domain": "sample",
                    "process_signature": self._get_current_segmentation_process_signature(),
                    "mapping_signature": self._current_mapping_signature,
                    "sample_count": sample_count,
                    "projected_interval_count": int(len(records)),
                    "covered_sample_count": covered_count,
                    "coverage_rate": (
                        float(covered_count / sample_count) if sample_count else 0.0
                    ),
                    "projected_sample_start_idx": int(records[0]["sample_start_idx"]),
                    "projected_sample_end_idx": int(records[-1]["sample_end_idx"]),
                    "mapping_source": (
                        mapping_source or "program_line_and_point_order"
                    ),
                }
            else:
                diagnostics["sample_projection"] = {
                    "valid": False,
                    "status": status,
                    "coordinate_domain": "sample",
                    "process_signature": self._get_current_segmentation_process_signature(),
                    "mapping_signature": self._current_mapping_signature,
                    "reason": reason,
                }
        refresh_export_state = getattr(self, "_refresh_segmentation_export_controls", None)
        if callable(refresh_export_state):
            refresh_export_state()
        return status

    def _get_current_sample_projection_indices(self, sample_lines):
        """返回当前程序/刀具用于区间投影的连续采样点索引。"""

        sample_count = int(np.asarray(sample_lines).size)
        if sample_count <= 0:
            raise ValueError("实际负载文件没有可用采样点")
        selected_mask = np.ones(sample_count, dtype=bool)
        mask_builder = getattr(self, "build_sample_mask", None)
        if callable(mask_builder):
            try:
                program_getter = getattr(self, "get_selected_program_number", None)
                program_no = program_getter() if callable(program_getter) else None
                range_getter = getattr(self, "get_selected_tool_ranges", None)
                tool_ranges = range_getter() if callable(range_getter) else None
                candidate = mask_builder(program_no, tool_ranges)
                if candidate is not None:
                    candidate = np.asarray(candidate, dtype=bool).reshape(-1)
                    if candidate.size != sample_count:
                        raise ValueError("当前程序/刀具掩码与实际采样数量不一致")
                    selected_mask = candidate
            except ValueError:
                raise
            except Exception:
                selected_mask = np.ones(sample_count, dtype=bool)

        selected_indices = np.flatnonzero(selected_mask)
        if selected_indices.size == 0:
            raise ValueError("当前程序/刀具在实际负载中没有有效采样点")
        if selected_indices.size > 1 and np.any(np.diff(selected_indices) != 1):
            raise ValueError(
                "当前程序/刀具的实际采样点不是单一连续行程，无法按顺序比例映射"
            )
        return selected_indices.astype(int, copy=False)

    def _materialize_segmentation_sample_bounds_by_sequence(
        self,
        process_records,
        sample_lines,
    ):
        """N 列缺失时，按过程点和当前采样行程的相对顺序投影。"""

        selected_indices = self._get_current_sample_projection_indices(sample_lines)
        process_bounds = []
        previous_end = None
        for record in process_records:
            interval_id = record.get("interval_id") or record.get("zone_id") or "未知区间"
            bounds = self._resolve_interval_process_bounds(record)
            if not bounds:
                raise ValueError(f"区间 {interval_id} 缺少有效的过程边界")
            start_idx = int(bounds["start_idx"])
            end_idx = int(bounds["end_idx"])
            if end_idx < start_idx:
                raise ValueError(f"区间 {interval_id} 的过程边界顺序无效")
            if previous_end is not None and start_idx != previous_end + 1:
                raise ValueError("六态过程区间存在空洞或重叠，无法按行程顺序投影")
            process_bounds.append((start_idx, end_idx))
            previous_end = end_idx

        if not process_bounds:
            return []
        process_start = int(process_bounds[0][0])
        process_end_exclusive = int(process_bounds[-1][1]) + 1
        process_count = process_end_exclusive - process_start
        sample_count = int(selected_indices.size)
        if process_count <= 0:
            raise ValueError("六态过程范围为空，无法按行程顺序投影")

        process_boundaries = np.asarray(
            [start for start, _end in process_bounds] + [process_end_exclusive],
            dtype=float,
        )
        relative_boundaries = process_boundaries - float(process_start)
        sample_cuts = np.floor(
            relative_boundaries * float(sample_count) / float(process_count)
        ).astype(int)
        sample_cuts[0] = 0
        sample_cuts[-1] = sample_count
        sample_cuts = np.clip(sample_cuts, 0, sample_count)
        if np.any(np.diff(sample_cuts) <= 0):
            raise ValueError(
                "实际采样点少于区间顺序投影所需点数，无法保证每个区间均有采样点"
            )

        point_indices = np.asarray(
            getattr(self, "sample_data_point_indices", []),
            dtype=int,
        ).reshape(-1)
        if point_indices.size != sample_lines.size:
            blocks = list(getattr(self, "sample_data_base_blocks", None) or [])
            point_builder = getattr(self, "compute_line_point_indices", None)
            if callable(point_builder):
                point_indices = np.asarray(
                    point_builder(sample_lines, blocks=blocks),
                    dtype=int,
                ).reshape(-1)
        if point_indices.size != sample_lines.size:
            point_indices = np.zeros(sample_lines.size, dtype=int)

        materialized = []
        for record, cut_start, cut_end in zip(
            process_records,
            sample_cuts[:-1],
            sample_cuts[1:],
        ):
            start_idx = int(selected_indices[int(cut_start)])
            end_idx = int(selected_indices[int(cut_end) - 1])
            start_line = int(sample_lines[start_idx])
            end_line = int(sample_lines[end_idx])
            start_point = int(point_indices[start_idx])
            end_point = int(point_indices[end_idx])
            start_label = self.format_rg_line_point(start_line, start_point)
            end_label = self.format_rg_line_point(end_line, end_point)
            current = dict(record)
            current.update(
                {
                    "sample_start_idx": start_idx,
                    "sample_end_idx": end_idx,
                    "sample_start_line": start_line,
                    "sample_end_line": end_line,
                    "sample_start_point_index": start_point,
                    "sample_end_point_index": end_point,
                    "sample_start_label": start_label,
                    "sample_end_label": end_label,
                    "sample_interval_range": f"{start_label}-{end_label}",
                    "sample_count": end_idx - start_idx + 1,
                    "mapping_source": "journey_order_ratio_missing_n",
                    "mapping_description": "按行程顺序比例映射（N列缺失）",
                }
            )
            materialized.append(current)

        expected_start = int(selected_indices[0])
        expected_end = int(selected_indices[-1])
        next_start = expected_start
        for record in materialized:
            start_idx = int(record["sample_start_idx"])
            end_idx = int(record["sample_end_idx"])
            if start_idx != next_start or end_idx < start_idx:
                raise ValueError("按行程顺序投影后的采样区间存在空洞或重叠")
            next_start = end_idx + 1
        if next_start - 1 != expected_end:
            raise ValueError("按行程顺序投影未完整覆盖当前程序/刀具实际负载")
        return materialized

    def _materialize_segmentation_sample_bounds(self, records):
        """按过程边界把六态结果投影为实际负载的零基采样区间。"""
        sample_lines = self._get_segmentation_sample_lines()
        if not getattr(self, "data", None):
            raise ValueError("尚未处理工艺信息，无法投影实际采样点区间")
        process_records = [dict(record) for record in records or [] if isinstance(record, dict)]
        if not process_records:
            return []
        process_row_indices = [
            idx
            for idx, row in enumerate(self.data)
            if isinstance(row, dict) and not bool(row.get("_is_synthetic_fill"))
        ]
        missing_raw = [
            idx
            for idx in process_row_indices
            if self.data[idx].get("line_no_raw") is None
        ]
        if process_row_indices and len(missing_raw) == len(process_row_indices):
            return self._materialize_segmentation_sample_bounds_by_sequence(
                process_records,
                sample_lines,
            )
        if missing_raw:
            raise ValueError(
                "工艺信息 N 行号部分缺失；为保留已有行号的优先对齐语义，"
                "请补齐 N 列或绑定原始 NC 文件后重试"
            )

        context = self._get_current_sample_line_point_context(line_numbers=sample_lines)
        if not context:
            raise ValueError("无法建立实际负载的行号/点号上下文")
        point_indices = np.asarray(context.get("point_indices", []), dtype=int)
        sample_x = np.asarray(context.get("x_positions", []), dtype=float)
        if point_indices.size != sample_lines.size or sample_x.size != sample_lines.size:
            raise ValueError("实际负载的点号或坐标数量与采样数量不一致")

        process_starts = []
        previous_process_end = -1
        for record in process_records:
            interval_id = record.get("interval_id") or record.get("zone_id") or "未知区间"
            process_bounds = self._resolve_interval_process_bounds(record)
            if not process_bounds:
                raise ValueError(f"区间 {interval_id} 缺少有效的过程边界")
            process_x_bounds = self._resolve_interval_process_x_bounds(
                record,
                process_bounds=process_bounds,
            )
            if not process_x_bounds:
                raise ValueError(f"区间 {interval_id} 缺少有效的过程坐标")
            start_x = float(process_x_bounds.get("process_start_x", float("nan")))
            if not np.isfinite(start_x):
                raise ValueError(f"区间 {interval_id} 的过程起点无效")
            process_start_idx = int(process_bounds["start_idx"])
            process_end_idx = int(process_bounds["end_idx"])
            if process_start_idx <= previous_process_end:
                raise ValueError("六态过程区间顺序重叠，无法安全投影")
            if process_starts and start_x <= process_starts[-1]:
                raise ValueError("六态过程坐标未严格递增，无法安全投影")
            process_starts.append(start_x)
            previous_process_end = process_end_idx

        last_x_bounds = self._resolve_interval_process_x_bounds(process_records[-1])
        process_domain_end = float(
            last_x_bounds.get("process_display_end_x", float("nan"))
        ) if last_x_bounds else float("nan")
        process_domain_start = float(process_starts[0])
        if not np.isfinite(process_domain_end) or process_domain_end <= process_starts[-1]:
            raise ValueError("六态过程终点无效，无法安全投影")

        sequence_blocks = list(getattr(self, "sample_data_base_blocks", None) or [])
        if not sequence_blocks:
            group_keys = getattr(self, "sample_data_program_numbers", None)
            if group_keys is not None and len(group_keys) != sample_lines.size:
                group_keys = None
            sequence_blocks = self.compute_sequence_blocks(sample_lines, group_keys=group_keys)

        normalized_blocks = []
        expected_block_start = 0
        for raw_start, raw_end in sequence_blocks:
            block_start = int(raw_start)
            block_end = int(raw_end)
            if (
                block_start != expected_block_start
                or block_end < block_start
                or block_end >= sample_lines.size
            ):
                raise ValueError("实际负载序列块无效或不连续")
            block_x = sample_x[block_start:block_end + 1]
            if not np.all(np.isfinite(block_x)):
                raise ValueError("实际负载序列块包含无效坐标")
            if block_x.size > 1 and np.any(np.diff(block_x) <= 0.0):
                raise ValueError("实际负载序列块坐标未严格递增")
            normalized_blocks.append((block_start, block_end, block_x))
            expected_block_start = block_end + 1
        if expected_block_start != sample_lines.size:
            raise ValueError("实际负载序列块未覆盖全部采样点")

        matching_blocks = []
        for block_start, block_end, block_x in normalized_blocks:
            block_domain_start = float(block_x[0])
            # 行内样本坐标位于 [line, line + 1) 内；末行的右边界
            # 是该行号 + 1，不能只用最后一个样本中心判断覆盖。
            block_domain_end = float(sample_lines[block_end]) + 1.0
            if (
                block_domain_start > process_domain_start + 1e-9
                or block_domain_end < process_domain_end - 1e-9
            ):
                continue
            local_start = int(np.searchsorted(block_x, process_domain_start, side="left"))
            local_end = int(np.searchsorted(block_x, process_domain_end, side="left"))
            if local_end > local_start:
                matching_blocks.append(
                    (block_start, block_end, block_x, local_start, local_end)
                )
        if not matching_blocks:
            raise ValueError("实际负载中没有完整覆盖六态过程范围的单一序列块")
        if len(matching_blocks) != 1:
            raise ValueError("多个实际负载序列块匹配六态过程范围，旧 .rg 坐标无法唯一表示")

        block_start, _block_end, block_x, local_domain_start, local_domain_end = matching_blocks[0]
        process_boundaries = np.asarray([*process_starts, process_domain_end], dtype=float)
        local_cuts = np.searchsorted(block_x, process_boundaries, side="left")
        local_cuts = np.clip(local_cuts, local_domain_start, local_domain_end).astype(int)
        if np.any(np.diff(local_cuts) < 0):
            raise ValueError("六态采样切分点未保持单调")
        interval_count = len(process_records)
        available_sample_count = int(local_domain_end - local_domain_start)
        if available_sample_count < interval_count:
            raise ValueError(
                "六态过程区间数多于可用实际采样点，无法保证每个区间至少对应一个采样点"
            )

        # 过程域短区间可能恰好落在两个实际采样点之间，导致相邻切分点重合。
        # 在总体采样数充足时，将内部边界确定性地量化到相邻采样切点，既保持
        # 全覆盖和区间顺序，也避免因为一个无落点的微小区间放弃整条实际曲线。
        original_cuts = local_cuts.copy()
        local_cuts[0] = int(local_domain_start)
        local_cuts[-1] = int(local_domain_end)
        for boundary_index in range(1, interval_count):
            minimum_cut = int(local_cuts[boundary_index - 1]) + 1
            remaining_intervals = interval_count - boundary_index
            maximum_cut = int(local_domain_end) - remaining_intervals
            local_cuts[boundary_index] = min(
                max(int(original_cuts[boundary_index]), minimum_cut),
                maximum_cut,
            )
        quantized_boundary_count = int(np.count_nonzero(local_cuts != original_cuts))
        mapping_source = (
            "program_line_and_point_order_quantized"
            if quantized_boundary_count
            else "program_line_and_point_order"
        )
        mapping_description = (
            "按程序行号与点内顺序映射（短区间按相邻采样点量化）"
            if quantized_boundary_count
            else "按程序行号与点内顺序精确映射"
        )

        pair_counts = {}
        for line_no, point_idx in zip(sample_lines, point_indices):
            key = (int(line_no), int(point_idx))
            pair_counts[key] = pair_counts.get(key, 0) + 1

        materialized = []
        for record, local_start, local_end in zip(
            process_records,
            local_cuts[:-1],
            local_cuts[1:],
        ):
            if local_end <= local_start:
                interval_id = record.get("interval_id") or record.get("zone_id") or "未知区间"
                raise ValueError(
                    f"过程区间 {interval_id} 没有对应的实际采样点，映射不完整"
                )
            start_idx = block_start + int(local_start)
            end_idx = block_start + int(local_end) - 1
            if not (0 <= start_idx <= end_idx < sample_lines.size):
                raise ValueError("实际负载采样区间越界")

            start_line = int(sample_lines[start_idx])
            end_line = int(sample_lines[end_idx])
            start_point = int(point_indices[start_idx])
            end_point = int(point_indices[end_idx])
            start_label = self.format_rg_line_point(start_line, start_point)
            end_label = self.format_rg_line_point(end_line, end_point)

            current = dict(record)
            current.update(
                {
                    "sample_start_idx": start_idx,
                    "sample_end_idx": end_idx,
                    "sample_start_line": start_line,
                    "sample_end_line": end_line,
                    "sample_start_point_index": start_point,
                    "sample_end_point_index": end_point,
                    "sample_start_label": start_label,
                    "sample_end_label": end_label,
                    "sample_interval_range": f"{start_label}-{end_label}",
                    "sample_count": end_idx - start_idx + 1,
                    "mapping_source": mapping_source,
                    "mapping_description": mapping_description,
                    "quantized_boundary_count": quantized_boundary_count,
                }
            )

            materialized.append(current)

        if not materialized:
            raise ValueError("六态过程范围内没有可导出的实际负载采样区间")
        if len(materialized) != len(process_records):
            raise ValueError("实际采样映射没有保持过程区间一一对应")

        expected_start = block_start + int(local_domain_start)
        expected_end = block_start + int(local_domain_end) - 1
        next_start = expected_start
        for record in materialized:
            start_idx = int(record["sample_start_idx"])
            end_idx = int(record["sample_end_idx"])
            if start_idx != next_start or end_idx < start_idx:
                raise ValueError("实际采样区间存在空洞或重叠")
            next_start = end_idx + 1
            for key in (
                (int(record["sample_start_line"]), int(record["sample_start_point_index"])),
                (int(record["sample_end_line"]), int(record["sample_end_point_index"])),
            ):
                if pair_counts.get(key, 0) != 1:
                    raise ValueError(
                        f"实际采样坐标 {key[0]}.{key[1]} 在重复程序块中不唯一"
                    )
        if next_start - 1 != expected_end:
            raise ValueError("实际采样区间未完整覆盖匹配的过程范围")
        return materialized

    def _refresh_segmentation_sample_projection(self, *, refresh_view=False, silent=True):
        """只刷新采样投影；无论成功与否都不重新运行过程域分类。"""

        authoritative = bool(
            getattr(self, "_current_interval_ready", False)
            and str(getattr(self, "_current_interval_source", "") or "") == "segmentation"
        )
        if not authoritative:
            self._set_segmentation_mapping_status(
                "pending",
                reason="尚无有效过程域划分",
            )
            return None
        if not bool(getattr(self, "sample_data_loaded", False)):
            self._set_segmentation_mapping_status(
                "not_available",
                reason="尚未导入实际采样文件",
            )
            return None

        mapping_signature = self._build_segmentation_mapping_signature()
        if not mapping_signature:
            self._set_segmentation_mapping_status(
                "failed",
                reason="无法建立采样内容签名",
            )
            return None
        if (
            str(getattr(self, "_sample_mapping_status", "") or "") == "valid"
            and str(getattr(self, "_current_mapping_signature", "") or "")
            == mapping_signature
        ):
            cached_records = [
                dict(record)
                for record in (
                    getattr(self, "_segmentation_sample_projection_records", []) or []
                )
                if isinstance(record, dict)
            ]
            if cached_records:
                return cached_records

        runtime_records = self._get_current_interval_records(allow_profile_fallback=False)
        try:
            projected_records = self._materialize_segmentation_sample_bounds(runtime_records)
        except Exception as exc:
            self._authoritative_segmentation_sample_lookup_cache = None
            self._set_segmentation_mapping_status(
                "failed",
                reason=str(exc),
                signature=mapping_signature,
            )
            if not silent:
                messagebox.showwarning("采样映射失败", str(exc))
            return None

        self._authoritative_segmentation_sample_lookup_cache = None
        self._set_segmentation_mapping_status(
            "valid",
            signature=mapping_signature,
            projected_records=projected_records,
        )
        if refresh_view and hasattr(self, "generate_plots"):
            try:
                self.generate_plots(
                    save=False,
                    silent=silent,
                    interval_policy="reuse_current_template",
                    refresh_prediction=False,
                )
            except Exception as exc:
                if not silent:
                    messagebox.showwarning("实际负载叠图失败", str(exc))
        return [dict(record) for record in projected_records]

    def _get_authoritative_segmentation_sample_records(self):
        """为样本级消费者返回当前过程结果的采样投影副本。"""
        records = self._refresh_segmentation_sample_projection(
            refresh_view=False,
            silent=True,
        )
        if records is None:
            result = getattr(self, "_latest_segmentation_result", None)
            diagnostics = getattr(result, "diagnostics", None)
            projection = (
                dict(diagnostics.get("sample_projection") or {})
                if isinstance(diagnostics, dict)
                else {}
            )
            reason = str(projection.get("reason") or "采样映射无效")
            raise ValueError(reason)
        return [dict(record) for record in records]

    def _validate_segmentation_prediction_payload_arrays(self, payload):
        """验证样本预测数组及其程序行序列，并返回统一的一维数组。"""
        if not isinstance(payload, dict):
            raise ValueError("当前实际负载模式没有可用的样本级预测负载")
        try:
            predicted = np.asarray(payload.get("predicted_load", []), dtype=float)
            predicted_idle = np.asarray(
                payload.get("predicted_idle_power", []),
                dtype=float,
            )
            program_line = np.asarray(payload.get("program_line", []), dtype=int)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("样本级预测数组或程序行号格式无效") from exc
        if predicted.ndim != 1 or predicted_idle.ndim != 1 or program_line.ndim != 1:
            raise ValueError("样本级预测数组和程序行号必须是一维序列")
        if predicted.size == 0 or predicted.size != program_line.size:
            raise ValueError("预测负载与实际采样行号数量不一致")
        if predicted_idle.size != predicted.size:
            raise ValueError("预测空载功率与预测负载数量不一致")
        invalid_predicted_count = int(np.sum(~np.isfinite(predicted)))
        invalid_idle_count = int(np.sum(~np.isfinite(predicted_idle)))
        if invalid_predicted_count or invalid_idle_count:
            raise ValueError(
                "样本级 P_pred/P_idle 必须全部有限，"
                f"当前无效点分别为 {invalid_predicted_count}/{invalid_idle_count}"
            )
        sample_lines = self._get_segmentation_sample_lines()
        if (
            program_line.size != sample_lines.size
            or not np.array_equal(program_line, sample_lines)
        ):
            raise ValueError("预测负载的程序行序列与实际采样文件不一致")
        return predicted, predicted_idle, program_line

    def _resolve_segmentation_prediction_policy(self, payload=None):
        """判定六态分段可使用的样本预测，并保留其真实来源。"""
        mode = str(getattr(self, "sample_data_mode", "") or "").strip()
        if mode != "experiment_measurement":
            return {
                "allowed": True,
                "source": "non_measurement",
                "independent": True,
                "temporary_measurement_mode": False,
                "reason": "非实验实测模式",
            }
        source_payload = payload if isinstance(payload, dict) else {}
        try:
            self._validate_segmentation_prediction_payload_arrays(source_payload)
        except ValueError as exc:
            return {
                "allowed": False,
                "source": str(
                    source_payload.get("segmentation_prediction_source")
                    or source_payload.get("prediction_source")
                    or "unknown"
                ),
                "independent": False,
                "temporary_measurement_mode": False,
                "reason": str(exc),
            }
        payload_source = str(source_payload.get("prediction_source") or "").strip()
        declared_source = str(
            source_payload.get("segmentation_prediction_source") or ""
        ).strip()
        active_source = str(
            self._get_prediction_source()
            if hasattr(self, "_get_prediction_source")
            else "no_profile"
        ).strip()
        if active_source == "runtime_identified_profile":
            source_profile = getattr(self, "runtime_identified_kc_profile", None)
        else:
            source_profile = getattr(self, "imported_kc_profile", None)
        if not isinstance(source_profile, dict):
            source_profile = getattr(self, "active_kc_profile", None)
        independence_checker = getattr(
            self,
            "_profile_is_independent_from_current_measurement",
            None,
        )
        profile_is_independent = bool(
            isinstance(source_profile, dict)
            and callable(independence_checker)
            and independence_checker(
                source_profile,
                measurement=getattr(self, "manual_measurement_data", None),
            )
        )
        declared_reverse_matches_context = bool(
            declared_source == "measurement_reverse"
            and (
                (
                    payload_source == "no_profile"
                    and active_source in {"no_profile", "runtime_identified_profile"}
                )
                or (
                    payload_source == "runtime_identified_profile"
                    and active_source == "runtime_identified_profile"
                )
            )
        )

        if declared_reverse_matches_context:
            source = "measurement_reverse"
            independent = False
            temporary_mode = True
            reason = "当前实测反向辨识预测（仅用于本次区间划分）"
        elif (
            active_source == "runtime_identified_profile"
            and isinstance(source_profile, dict)
            and payload_source in {"no_profile", "runtime_identified_profile"}
        ):
            source = "measurement_reverse"
            independent = False
            temporary_mode = True
            reason = "当前实测绑定的运行时 profile（仅用于本次区间划分）"
        elif (
            active_source == "imported_profile"
            and payload_source == "imported_profile"
            and profile_is_independent
        ):
            source = "independent_profile"
            independent = True
            temporary_mode = False
            reason = "独立导入 profile 前向预测"
        elif (
            (
                declared_source == "same_measurement_profile"
                and active_source == "imported_profile"
                and isinstance(source_profile, dict)
                and not profile_is_independent
            )
            or (
                active_source == "imported_profile"
                and payload_source == "imported_profile"
                and isinstance(source_profile, dict)
                and not profile_is_independent
            )
        ):
            source = "same_measurement_profile"
            independent = False
            temporary_mode = True
            reason = "与当前实测同源的 profile（仅用于本次区间划分）"
        else:
            return {
                "allowed": False,
                "source": declared_source or payload_source or active_source or "unknown",
                "independent": False,
                "temporary_measurement_mode": False,
                "reason": "没有可追溯的样本级预测来源",
            }
        return {
            "allowed": True,
            "source": source,
            "independent": bool(independent),
            "temporary_measurement_mode": bool(temporary_mode),
            "reason": reason,
        }

    def _is_segmentation_sample_prediction_allowed(self, payload=None):
        """返回六态分段阶段是否允许使用该预测。"""
        policy = self._resolve_segmentation_prediction_policy(payload)
        return bool(policy.get("allowed", False))

    def _has_independent_segmentation_sample_prediction(self, payload=None):
        """兼容旧导出调用；此名称已弃用，结果表示 allowed 而非 independent。"""
        return self._is_segmentation_sample_prediction_allowed(payload)

    def _get_segmentation_prediction_payload(self):
        """取得与实际采样序列等长的权威预测负载。"""
        mode = str(getattr(self, "sample_data_mode", "") or "").strip()
        if mode == "sampledata":
            payload = self._build_sampledata_prediction_payload()
        elif mode == "experiment_measurement":
            measurement = getattr(self, "manual_measurement_data", None)
            if isinstance(measurement, dict):
                predicted = np.asarray(measurement.get("predicted_load", []), dtype=float)
                program_line = np.asarray(measurement.get("program_line", []), dtype=int)
                if predicted.size != program_line.size and hasattr(self, "_refresh_manual_measurement_prediction"):
                    self._refresh_manual_measurement_prediction()
                    measurement = getattr(self, "manual_measurement_data", None)
            payload = measurement
        else:
            payload = None
        if not isinstance(payload, dict):
            raise ValueError("当前实际负载模式没有可用的样本级预测负载")
        predicted, predicted_idle, program_line = (
            self._validate_segmentation_prediction_payload_arrays(payload)
        )
        prediction_policy = self._resolve_segmentation_prediction_policy(payload)
        if not bool(prediction_policy.get("allowed", False)):
            raise ValueError(
                "六态样本预测缺少可追溯来源；允许独立 profile，或明确标记为"
                " measurement_reverse / same_measurement_profile 的临时区间划分预测"
            )

        return {
            **payload,
            "predicted_load": predicted,
            "predicted_idle_power": predicted_idle,
            "program_line": program_line,
            "segmentation_prediction_policy": dict(prediction_policy),
            "segmentation_prediction_source": str(
                prediction_policy.get("source") or "unknown"
            ),
            "segmentation_prediction_independent": bool(
                prediction_policy.get("independent", False)
            ),
            "segmentation_temporary_measurement_mode": bool(
                prediction_policy.get("temporary_measurement_mode", False)
            ),
        }

    def _prepare_segmentation_prediction_context(self):
        """在解码前从同一来源重建样本域与过程域预测。"""
        mode = str(getattr(self, "sample_data_mode", "") or "").strip()
        if mode != "experiment_measurement":
            return self._get_segmentation_prediction_payload(), {
                "success": True,
                "source": "non_measurement",
                "context_signature": "",
                "prediction_context": "",
                "row_count": 0,
            }

        measurement = getattr(self, "manual_measurement_data", None)
        refresher = getattr(self, "_refresh_manual_measurement_prediction", None)
        process_synchronizer = getattr(
            self,
            "_refresh_segmentation_process_prediction",
            None,
        )
        if not isinstance(measurement, dict) or not callable(refresher):
            raise ValueError("当前实际负载无法重建样本级预测")
        if not callable(process_synchronizer):
            raise ValueError("缺少过程域预测同步器")

        last_reason = ""
        for _attempt in range(2):
            active_source = str(
                self._get_prediction_source()
                if hasattr(self, "_get_prediction_source")
                else "no_profile"
            ).strip()
            uses_forward_profile = active_source in {
                "imported_profile",
                "runtime_identified_profile",
            }
            refreshed = refresher(
                allow_saved_sample_profile=False,
                allow_measurement_resolve=not uses_forward_profile,
                display_mode="forward" if uses_forward_profile else "posterior",
            )
            if refreshed is None:
                raise ValueError("样本级预测重建失败")

            payload = self._get_segmentation_prediction_payload()
            process_context = process_synchronizer(payload)
            if not isinstance(process_context, dict) or not bool(
                process_context.get("success", False)
            ):
                reason = (
                    str((process_context or {}).get("reason") or "").strip()
                    if isinstance(process_context, dict)
                    else ""
                )
                if (
                    _attempt == 0
                    and reason == "样本预测与过程预测不属于同一模型上下文"
                ):
                    # 同步器刚应用了 profile 中的模型参数；
                    # 再生成一次样本预测后必须严格一致。
                    last_reason = reason
                    continue
                raise ValueError(reason or "过程域预测同步失败")

            policy = dict(payload.get("segmentation_prediction_policy") or {})
            if str(process_context.get("source") or "") != str(
                policy.get("source") or ""
            ):
                raise ValueError("样本域与过程域的预测来源不一致")
            if bool(process_context.get("independent", False)) != bool(
                policy.get("independent", False)
            ) or bool(
                process_context.get("temporary_measurement_mode", False)
            ) != bool(policy.get("temporary_measurement_mode", False)):
                raise ValueError("样本域与过程域的来源标记不一致")

            sample_context_signature = str(
                payload.get("segmentation_sample_prediction_context_signature")
                or measurement.get("segmentation_sample_prediction_context_signature")
                or ""
            )
            process_prediction_context = str(
                process_context.get("prediction_context") or ""
            )
            if (
                sample_context_signature
                and process_prediction_context
                and sample_context_signature == process_prediction_context
            ):
                return payload, process_context
            last_reason = "样本域与过程域的预测内容签名不一致"

        raise ValueError(last_reason or "无法建立同源预测上下文")

    def _invalidate_segmentation_sample_projection(self, reason=""):
        """实际采样上下文变化时只清除映射，不触碰过程域划分。"""
        self._authoritative_segmentation_sample_lookup_cache = None
        self._segmentation_sample_projection_records = []
        self._current_mapping_signature = ""
        result = getattr(self, "_latest_segmentation_result", None)
        diagnostics = getattr(result, "diagnostics", None)
        if isinstance(diagnostics, dict):
            diagnostics.pop("sample_projection", None)
            diagnostics.pop("sample_visualization", None)
        cleaner = getattr(self, "_clear_segmentation_output_artifacts", None)
        if callable(cleaner):
            try:
                cleaner(scope="mapping")
            except TypeError:
                # 兼容尚未拆分 scope 的外部 mixin；此时宁可保留旧文件，
                # 也不能误删仍然有效的过程域导出。
                pass
            except OSError as exc:
                status_var = getattr(self, "sample_mapping_status_var", None)
                if status_var is not None and hasattr(status_var, "set"):
                    status_var.set(f"采样映射已失效，旧映射导出清理失败（{exc}）")
                return False
        next_status = (
            "pending" if bool(getattr(self, "sample_data_loaded", False))
            else "not_available"
        )
        self._set_segmentation_mapping_status(next_status, reason=str(reason or ""))
        return True

    def run_full_path_segmentation(
        self,
        *,
        export_outputs=True,
        refresh_view=True,
        silent=False,
    ):
        """运行确定性六态 Semi-Markov 划分并写入唯一当前区间状态。"""
        if bool(getattr(self, "_segmentation_running", False)):
            return getattr(self, "_latest_segmentation_result", None)
        if not getattr(self, "data", None):
            if hasattr(self, "segmentation_status_var"):
                self.segmentation_status_var.set("全行程六类划分: 请先导入工艺信息")
            if not silent:
                messagebox.showwarning("无工艺信息", "请先导入并处理工艺信息文件。")
            return None
        # 每次重算先撤销上一批权威状态；本轮失败时不得继续沿用旧边界或旧导出。
        if hasattr(self, "_clear_current_interval_state"):
            self._clear_current_interval_state(keep_profile_lock=False)
        self._latest_segmentation_result = None
        cleaner = getattr(self, "_clear_segmentation_output_artifacts", None)
        if callable(cleaner):
            try:
                cleaner()
            except OSError as exc:
                if hasattr(self, "segmentation_status_var"):
                    self.segmentation_status_var.set(f"全行程六类划分: 旧导出清理失败（{exc}）")
                if not silent:
                    messagebox.showerror("六类划分失败", str(exc))
                return None

        self._segmentation_running = True
        try:
            if hasattr(self, "segmentation_status_var"):
                self.segmentation_status_var.set("过程域六类划分: 正在重算 MRR 与特征…")
            if hasattr(self, "set_progress"):
                self.set_progress(62, "正在重算程序 MRR 与过程域特征...")
            input_frame = self._build_segmentation_input_frame()
            if input_frame.empty:
                raise ValueError("没有可用于六态划分的原始工艺点")
            config = (
                getattr(self, "segmentation_config", None)
                or getattr(self, "_segmentation_config", None)
                or SegmentationConfig()
            )
            if not isinstance(config, SegmentationConfig):
                raise TypeError("segmentation_config 必须是 SegmentationConfig")
            self._segmentation_config = config
            pipeline = getattr(self, "segmentation_pipeline", None)
            if not isinstance(pipeline, SegmentationPipeline) or pipeline.config is not config:
                pipeline = SegmentationPipeline(config)
                self.segmentation_pipeline = pipeline
            if hasattr(self, "set_progress"):
                self.set_progress(72, "正在执行六态过程域划分...")
            result = pipeline.run(
                input_frame,
                scorer=RuleSegmentScorer(config),
            )
            if len(result.point_labels) != len(input_frame):
                raise ValueError(
                    f"逐点结果数量不一致: 输入 {len(input_frame)}，输出 {len(result.point_labels)}"
                )
            process_input_diagnostics = dict(
                getattr(self, "process_input_diagnostics", {}) or {}
            )
            line_matching = dict(
                getattr(self, "process_line_number_diagnostics", {}) or {}
            )
            line_number_source = str(
                line_matching.get("line_number_source") or "missing"
            )
            group_count = int(line_matching.get("nc_process_group_count", 0) or 0)
            matched_group_count = int(
                line_matching.get("nc_matched_group_count", 0) or 0
            )
            if line_number_source == "input":
                unique_line_match = True
                line_match_status = "not_required_input_lines"
            elif line_number_source in {
                "nc_profile_line_text",
                "input_with_nc_completion",
            } and group_count > 0 and matched_group_count == group_count:
                unique_line_match = True
                line_match_status = "unique"
            else:
                unique_line_match = False
                line_match_status = "failed_or_ambiguous"
            line_matching.update(
                {
                    "unique_match": bool(unique_line_match),
                    "match_status": line_match_status,
                }
            )
            result.diagnostics.update(
                {
                    "raw_input_row_count": int(
                        process_input_diagnostics.get(
                            "raw_data_row_count", len(input_frame)
                        )
                    ),
                    "valid_process_point_count": int(len(input_frame)),
                    "process_input": process_input_diagnostics,
                    "mrr_formula": "MRR_program = ap * ae * F_program / 60",
                    "input_mrr_columns_ignored": True,
                    "program_line_matching": line_matching,
                    "classification_domain": "process_info",
                    "classification_uses_actual_sample": False,
                    "classification_uses_prediction_profile": False,
                }
            )
            diagnostics = dict(result.diagnostics or {})
            fallback_used = bool(diagnostics.get("fallback_used", False))
            fallback_scope = str(diagnostics.get("fallback_scope") or "none")
            fallback_validated = bool(
                diagnostics.get("fallback_validated", not fallback_used)
            )
            if fallback_used and not (
                fallback_scope == "local_verified" and fallback_validated
            ):
                fallback_reason = str(
                    diagnostics.get("fallback_reason") or "未知解码错误"
                )
                if export_outputs and hasattr(
                    self,
                    "export_segmentation_failure_diagnostics",
                ):
                    try:
                        self.export_segmentation_failure_diagnostics(result)
                    except Exception as export_exc:
                        result.diagnostics[
                            "failure_diagnostics_export_error"
                        ] = str(export_exc)
                raise ValueError(
                    f"解码失败/已回退（{fallback_scope}）：{fallback_reason}"
                )
            if (
                float(diagnostics.get("coverage_rate", 0.0)) != 1.0
                or int(diagnostics.get("gap_count", -1)) != 0
                or int(diagnostics.get("overlap_count", -1)) != 0
                or int(diagnostics.get("illegal_transition_count", -1)) != 0
                or not bool(diagnostics.get("postprocess_validation_passed", False))
            ):
                raise ValueError("六态划分未满足全覆盖、工艺空载门控或状态结构约束")

            runtime_records = self._adapt_segmentation_interval_records(result)
            result_diagnostics = dict(result.diagnostics or {})
            process_signature = str(result_diagnostics.get("process_signature") or "")
            if not process_signature:
                repeat_diagnostics = dict(
                    result_diagnostics.get("repeat_run_consistency") or {}
                )
                process_signature = str(
                    repeat_diagnostics.get("input_signature") or ""
                )
            if not process_signature:
                raise ValueError("过程域划分缺少可追溯的 process_signature")
            result.diagnostics["process_signature"] = process_signature
            result.diagnostics["coordinate_domain"] = "process_info"
            self._current_process_signature = process_signature
            self._set_current_interval_state(
                interval_records=runtime_records,
                segment_records=[dict(record) for record in runtime_records],
                point_kc_map={},
                source="segmentation",
                profile_locked=False,
                context_signature=process_signature,
                prediction_source="process_info",
                measurement_case_signature="",
            )
            self._latest_segmentation_result = result
            self._invalidate_segmentation_sample_projection(reason="过程域划分已更新")
            if bool(getattr(self, "sample_data_loaded", False)):
                if hasattr(self, "set_progress"):
                    self.set_progress(86, "过程域划分完成，正在建立实际采样映射...")
                self._refresh_segmentation_sample_projection(
                    refresh_view=False,
                    silent=True,
                )
            self.target_load_curve = []
            diagnostics_path = diagnostics.get("path", {})
            path_source = str(diagnostics_path.get("source") or "unknown")
            if hasattr(self, "segmentation_status_var"):
                fallback_suffix = (
                    "，已验证局部回退"
                    if fallback_used and fallback_scope == "local_verified"
                    else ""
                )
                self.segmentation_status_var.set(
                    f"过程域六类划分: {len(result.point_labels)} 点 / "
                    f"{len(result.intervals)} 段，行程来源 {path_source}"
                    f"{fallback_suffix}"
                )
        except Exception as exc:
            if hasattr(self, "segmentation_status_var"):
                self.segmentation_status_var.set(f"全行程六类划分: 失败（{exc}）")
            if not silent:
                messagebox.showerror("六类划分失败", str(exc))
            return None
        finally:
            self._segmentation_running = False

        if export_outputs and hasattr(self, "export_latest_segmentation_result"):
            try:
                self.export_latest_segmentation_result(result)
            except Exception as exc:
                if hasattr(self, "segmentation_status_var"):
                    self.segmentation_status_var.set(f"六类划分完成，但结构化导出失败（{exc}）")
                if not silent:
                    messagebox.showerror("六类结果导出失败", str(exc))

        if refresh_view and hasattr(self, "generate_plots"):
            try:
                if hasattr(self, "set_progress"):
                    self.set_progress(94, "正在生成过程域图表与可用的实际负载叠图...")
                self.generate_plots(
                    save=False,
                    silent=silent,
                    interval_policy="reuse_current_template",
                    refresh_prediction=False,
                )
            except Exception as exc:
                if not silent:
                    messagebox.showwarning("六类结果刷新失败", str(exc))
        return result

    def _on_main_prediction_config_changed(self):
        self.refresh_prediction_mode_controls()
        self.refresh_prediction_metrics_summary()
        if self.data:
            try:
                interval_policy = self._get_default_interval_policy() if hasattr(self, "_get_default_interval_policy") else "fresh_or_empty"
                self.generate_plots(silent=True, interval_policy=interval_policy)
            except Exception:
                pass

    def has_posterior_curve_ready(self):
        return False

    def get_effective_prediction_mode(self, mode=None):
        return "direct_prediction"

    def refresh_prediction_mode_controls(self, prefer_posterior=False):
        target_mode = self.get_effective_prediction_mode()
        if str(self.prediction_mode_var.get()).strip() != target_mode:
            self.prediction_mode_var.set(target_mode)

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

    def get_prediction_curve_label(self, mode=None):
        return "预测负载"

    @staticmethod
    def _populate_treeview(tree, frame, max_rows=5000, width_scale=11):
        """填充通用只读表格，供预测结果弹窗复用。"""
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame)
        columns = list(frame.columns)
        tree.delete(*tree.get_children())
        tree.configure(columns=columns)
        for column in columns:
            tree.heading(column, text=str(column))
            width = max(80, min(320, max(len(str(column)) * width_scale, 80)))
            tree.column(column, width=width, anchor="center", stretch=True)
        for values in frame.head(max_rows).itertuples(index=False, name=None):
            formatted = []
            for value in values:
                if isinstance(value, (float, np.floating)):
                    formatted.append("" if not np.isfinite(value) else f"{float(value):.6g}")
                else:
                    formatted.append("" if value is None else str(value))
            tree.insert("", tk.END, values=formatted)

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
        """按应用模式选择 SampleData 预测策略。"""
        strategy = getattr(self, "_build_sampledata_prediction_payload_for_mode", None)
        if callable(strategy):
            return strategy()
        return self._build_forward_sampledata_prediction_payload()

    def _build_forward_sampledata_prediction_payload(self):
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
            if not self._record_represents_steady_interval(interval):
                continue
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
