from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .segmentation import STATE_CODE_BY_TYPE


class IntervalRuntimeMixin:
    """PIT/SMIF 与发布版共用的区间状态、边界和工艺点对齐层。"""

    @staticmethod
    def _normalize_runtime_prediction_source(source) -> str:
        normalized = str(source or "no_profile").strip().lower()
        if normalized in {"imported_profile", "runtime_identified_profile", "no_profile", "process_info"}:
            return normalized
        return "no_profile" if not normalized else normalized

    def _debug_interval_state_event(self, event, **fields):
        profile_origin_getter = getattr(self, "_get_profile_origin", None)
        prediction_source_getter = getattr(self, "_get_prediction_source", None)
        display_mode_getter = getattr(self, "_get_measurement_display_mode", None)
        payload = {
            "interval_source": str(getattr(self, "_current_interval_source", "") or "none"),
            "profile_origin": profile_origin_getter() if callable(profile_origin_getter) else "no_profile",
            "prediction_source": prediction_source_getter() if callable(prediction_source_getter) else "no_profile",
            "display_mode": display_mode_getter() if callable(display_mode_getter) else "inverse",
        }
        payload.update(fields)
        parts = [f"{key}={value}" for key, value in payload.items()]
        try:
            print(f"[DEBUG][interval-state] {event}: {', '.join(parts)}")
        except Exception:
            pass

    def _clear_current_interval_state(self, keep_profile_lock: bool = False):
        previous_ready = bool(getattr(self, "_current_interval_ready", False))
        previous_source = str(getattr(self, "_current_interval_source", "") or "")
        previous_locked = bool(getattr(self, "_profile_intervals_locked", False))
        previous_context_signature = str(getattr(self, "_current_interval_context_signature", "") or "")
        previous_prediction_source = self._normalize_runtime_prediction_source(
            getattr(self, "_current_interval_prediction_source", "no_profile")
        )
        previous_case_signature = str(getattr(self, "_current_interval_measurement_case_signature", "") or "")

        empty_intervals = []
        self.current_interval_records = empty_intervals
        self.current_segment_records = []
        self.current_interval_point_kc_map = {}
        self._current_interval_ready = False
        self._current_interval_source = ""
        self._current_interval_context_signature = ""
        self._current_interval_prediction_source = "no_profile"
        self._current_interval_measurement_case_signature = ""
        self._authoritative_segmentation_sample_lookup_cache = None
        if previous_source == "segmentation":
            self._current_process_signature = ""
            self._current_mapping_signature = ""
            self._segmentation_sample_projection_records = []
            mapping_status = "pending" if bool(getattr(self, "sample_data_loaded", False)) else "not_available"
            status_setter = getattr(self, "_set_segmentation_mapping_status", None)
            if callable(status_setter):
                status_setter(mapping_status, reason="过程域划分已清除")
            else:
                self._sample_mapping_status = mapping_status
        if not keep_profile_lock:
            self._profile_intervals_locked = False

        # 保留旧消费端的只读别名。
        self.pred_power_intervals = empty_intervals
        self.pit_records = empty_intervals
        self._cached_steady_intervals = {}
        if previous_ready or previous_source or previous_locked:
            self._debug_interval_state_event(
                "clear_current_state",
                previous_ready=previous_ready,
                previous_source=previous_source or "none",
                previous_profile_locked=previous_locked,
                previous_context_signature=previous_context_signature or "none",
                previous_prediction_source=previous_prediction_source,
                previous_case_signature=previous_case_signature or "none",
                keep_profile_lock=bool(keep_profile_lock),
            )

    def _set_current_interval_state(
        self,
        interval_records,
        *,
        segment_records=None,
        point_kc_map=None,
        source="",
        profile_locked=False,
        context_signature="",
        prediction_source="no_profile",
        measurement_case_signature="",
    ):
        if interval_records is not None and not isinstance(interval_records, list):
            raise TypeError("interval_records must be a list of dict records")
        if segment_records is not None and not isinstance(segment_records, list):
            raise TypeError("segment_records must be a list of dict records")
        if point_kc_map is not None and not isinstance(point_kc_map, dict):
            raise TypeError("point_kc_map must be a dict")

        source_text = str(source or "")
        has_authoritative_segmentation = bool(
            getattr(self, "_current_interval_ready", False)
            and str(getattr(self, "_current_interval_source", "") or "") == "segmentation"
        )
        if has_authoritative_segmentation and source_text != "segmentation":
            self._debug_interval_state_event(
                "skip_non_authoritative_interval_write",
                attempted_source=source_text or "none",
                preserved_source="segmentation",
            )
            return False

        previous_source = str(getattr(self, "_current_interval_source", "") or "")
        previous_locked = bool(getattr(self, "_profile_intervals_locked", False))
        previous_ready = bool(getattr(self, "_current_interval_ready", False))
        normalized_intervals = [dict(record) for record in (interval_records or []) if isinstance(record, dict)]
        normalized_segments = [dict(record) for record in (segment_records or []) if isinstance(record, dict)]
        normalized_point_map = dict(point_kc_map or {})

        self.current_interval_records = normalized_intervals
        self.current_segment_records = normalized_segments
        self.current_interval_point_kc_map = normalized_point_map
        self._current_interval_ready = True
        self._current_interval_source = source_text
        self._profile_intervals_locked = bool(profile_locked)
        if source_text == "segmentation":
            process_signature = str(
                context_signature or getattr(self, "_current_process_signature", "") or ""
            )
            self._current_process_signature = process_signature
            self._current_interval_context_signature = process_signature
            self._current_interval_prediction_source = "no_profile"
            self._current_interval_measurement_case_signature = ""
        else:
            self._current_interval_context_signature = str(context_signature or "")
            self._current_interval_prediction_source = self._normalize_runtime_prediction_source(prediction_source)
            signature_getter = getattr(self, "_get_current_measurement_case_signature", None)
            fallback_signature = signature_getter() if callable(signature_getter) else ""
            self._current_interval_measurement_case_signature = str(
                measurement_case_signature or fallback_signature or ""
            )
        self._authoritative_segmentation_sample_lookup_cache = None

        self.pred_power_intervals = self.current_interval_records
        self.pit_records = self.current_interval_records
        self._cached_steady_intervals = {
            "pit_records": [dict(record) for record in normalized_intervals],
            "segment_records": [dict(record) for record in normalized_segments],
            "point_kc_map": dict(normalized_point_map),
            "target_load_curve": list(getattr(self, "target_load_curve", []) or []),
        }
        self._debug_interval_state_event(
            "set_current_state",
            previous_ready=previous_ready,
            previous_source=previous_source or "none",
            previous_profile_locked=previous_locked,
            source=source_text or "none",
            profile_locked=bool(profile_locked),
            context_signature=self._current_interval_context_signature or "none",
            prediction_source=self._current_interval_prediction_source,
            measurement_case_signature=self._current_interval_measurement_case_signature or "none",
            interval_count=len(normalized_intervals),
            segment_count=len(normalized_segments),
            point_kc_count=len(normalized_point_map),
        )
        return True

    def _get_current_interval_records(self, allow_profile_fallback: bool = False):
        if bool(getattr(self, "_current_interval_ready", False)):
            return [
                dict(record)
                for record in (getattr(self, "current_interval_records", None) or [])
                if isinstance(record, dict)
            ]
        if allow_profile_fallback:
            resolver = getattr(self, "_resolve_forward_prediction_profile", None)
            extractor = getattr(self, "_extract_profile_interval_records", None)
            if callable(resolver) and callable(extractor):
                origin, profile = resolver(
                    measurement=getattr(self, "manual_measurement_data", None),
                    process_path=(
                        self._get_primary_input_file_or_empty()
                        if hasattr(self, "_get_primary_input_file_or_empty")
                        else ""
                    ),
                    allow_autoload_imported=(
                        self._should_allow_imported_profile_autoload()
                        if hasattr(self, "_should_allow_imported_profile_autoload")
                        else False
                    ),
                )
                if origin and isinstance(profile, dict):
                    return extractor(profile)
        return []

    def _get_current_segment_records(self, allow_profile_fallback: bool = False):
        if bool(getattr(self, "_current_interval_ready", False)):
            return [
                dict(record)
                for record in (getattr(self, "current_segment_records", None) or [])
                if isinstance(record, dict)
            ]
        if allow_profile_fallback:
            resolver = getattr(self, "_resolve_forward_prediction_profile", None)
            extractor = getattr(self, "_extract_profile_segment_records", None)
            if callable(resolver) and callable(extractor):
                _origin, profile = resolver(
                    measurement=getattr(self, "manual_measurement_data", None),
                    process_path=(
                        self._get_primary_input_file_or_empty()
                        if hasattr(self, "_get_primary_input_file_or_empty")
                        else ""
                    ),
                    allow_autoload_imported=(
                        self._should_allow_imported_profile_autoload()
                        if hasattr(self, "_should_allow_imported_profile_autoload")
                        else False
                    ),
                )
                if isinstance(profile, dict):
                    return extractor(profile)
        return []

    def _has_authoritative_segmentation_state(self):
        return bool(
            getattr(self, "_current_interval_ready", False)
            and str(getattr(self, "_current_interval_source", "") or "") == "segmentation"
        )

    @staticmethod
    def _get_process_row_sample_line(row, fallback=0):
        value = row.get("line_no_raw") if isinstance(row, dict) else None
        if value is None and isinstance(row, dict):
            value = row.get("line_no_aligned")
        try:
            return int(value)
        except Exception:
            return int(fallback)

    @staticmethod
    def _parse_line_point_label(label):
        text = str(label or "").strip()
        match = re.match(r"^\s*(-?\d+)\s*\.\s*(\d+)\s*$", text)
        if not match:
            return None, None
        line_no = int(match.group(1))
        point_no = int(match.group(2))
        return line_no, point_no if point_no > 0 else None

    def _resolve_profile_segment_state_code(self, record):
        segment_type = str(record.get("segment_type") or "").strip().lower()
        fixed_codes = dict(STATE_CODE_BY_TYPE)
        fixed_codes["steady_cutting"] = fixed_codes["steady"]
        if segment_type in fixed_codes:
            return int(fixed_codes[segment_type])
        try:
            return int(record.get("state_code"))
        except Exception:
            steady_subtype = str(record.get("steady_subtype") or "").strip().lower()
            is_idle = bool(record.get("is_idle_interval")) or str(record.get("kc_source", "")).strip().lower() == "idle"
            if segment_type == "nonsteady" or steady_subtype == "nonsteady":
                return 0
            if segment_type == "idle" or steady_subtype == "idle" or is_idle:
                return 1
            return 2

    def _resolve_smif_state_code(self, record):
        if not isinstance(record, dict):
            return 0
        segment_type = str(record.get("segment_type") or "").strip().lower()
        if segment_type in {"steady", "steady_cutting"}:
            return 2
        if segment_type == "idle":
            return 1
        if segment_type in {"entry", "transition", "nonsteady", "exit"}:
            return 0
        state_code = int(self._resolve_profile_segment_state_code(record))
        return state_code if state_code in {0, 1, 2} else 0

    def _record_represents_steady_interval(self, record):
        if not isinstance(record, dict):
            return False
        segment_type = str(record.get("segment_type") or "").strip().lower()
        is_idle = bool(record.get("is_idle_interval")) or str(record.get("kc_source", "")).strip().lower() == "idle"
        if is_idle:
            return False
        if segment_type:
            return segment_type in {"steady", "steady_cutting"}
        return int(self._resolve_profile_segment_state_code(record)) == 2

    def _refresh_interval_process_descriptors(self, record):
        current = dict(record) if isinstance(record, dict) else {}
        rows = getattr(self, "data", None) or []
        if not current or not rows:
            return current
        try:
            start_idx = int(current.get("start_idx"))
            end_idx = int(current.get("end_idx"))
        except (TypeError, ValueError):
            return current
        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx
        if start_idx < 0 or end_idx >= len(rows):
            return current
        interval_rows = [row for row in rows[start_idx:end_idx + 1] if isinstance(row, dict)]

        def _mean_value(*keys):
            values = []
            for row in interval_rows:
                value = next((row.get(key) for key in keys if row.get(key) is not None), None)
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    values.append(value)
            return float(np.mean(values)) if values else None

        for target_key, source_keys in {
            "a_p": ("ap", "a_p"),
            "a_e": ("ae", "a_e"),
            "F_plan": ("feed_effective", "F_program", "F_plan"),
            "p_idle": ("P_idle",),
            "p_pred": ("P",),
        }.items():
            value = _mean_value(*source_keys)
            if value is not None:
                current[target_key] = value
        return current

    def _get_steady_interval_records(self, records=None):
        if isinstance(records, list):
            source_records = records
        else:
            segment_records = self._get_current_segment_records(allow_profile_fallback=False)
            source_records = segment_records or self._get_current_interval_records(allow_profile_fallback=False)
        steady_records = []
        for record in source_records or []:
            if not isinstance(record, dict) or not self._record_represents_steady_interval(record):
                continue
            refreshed = self._refresh_interval_process_descriptors(record)
            if refreshed:
                steady_records.append(refreshed)
        return steady_records

    def _refresh_authoritative_segmentation_interval_descriptors(self):
        if not self._has_authoritative_segmentation_state() or not getattr(self, "data", None):
            return False
        return self._set_current_interval_state(
            interval_records=[
                self._refresh_interval_process_descriptors(record)
                for record in self._get_current_interval_records(allow_profile_fallback=False)
            ],
            segment_records=self._get_current_segment_records(allow_profile_fallback=False),
            point_kc_map=dict(getattr(self, "current_interval_point_kc_map", {}) or {}),
            source="segmentation",
            profile_locked=bool(getattr(self, "_profile_intervals_locked", False)),
            context_signature=str(getattr(self, "_current_interval_context_signature", "") or ""),
            prediction_source=str(getattr(self, "_current_interval_prediction_source", "no_profile") or "no_profile"),
            measurement_case_signature=str(getattr(self, "_current_interval_measurement_case_signature", "") or ""),
        )

    def _ensure_process_point_metadata(self):
        rows = getattr(self, "data", None) or []
        if not rows:
            return
        cache_key = (id(rows), len(rows))
        if getattr(self, "_process_point_metadata_cache_key", None) == cache_key:
            return
        complete = True
        for row in rows:
            try:
                if int(row.get("process_point_index", -1)) < 0 or int(row.get("process_point_count", 0)) <= 0:
                    complete = False
                    break
            except Exception:
                complete = False
                break
        if complete:
            self._process_point_metadata_cache_key = cache_key
            return

        point_counts = {}
        for row_index, row in enumerate(rows):
            raw_key = self._get_process_row_sample_line(row, fallback=row_index)
            point_counts[raw_key] = point_counts.get(raw_key, 0) + 1
        point_offsets = {}
        for row_index, row in enumerate(rows):
            raw_key = self._get_process_row_sample_line(row, fallback=row_index)
            point_idx = int(point_offsets.get(raw_key, 0))
            point_offsets[raw_key] = point_idx + 1
            row["process_point_index"] = point_idx
            row["process_point_count"] = int(point_counts.get(raw_key, point_idx + 1))
        self._process_point_metadata_cache_key = cache_key

    def _build_process_point_lookup(self):
        self._ensure_process_point_metadata()
        cache_key = (
            id(getattr(self, "data", None)),
            len(getattr(self, "data", None) or []),
            int(getattr(self, "_process_model_state_version", 0) or 0),
            str(self._get_prediction_source() if hasattr(self, "_get_prediction_source") else "no_profile"),
            str(getattr(self, "step_feed_model_signature", "") or ""),
            float(self.get_kc_value()),
            float(self.get_ke_value()),
        )
        if getattr(self, "_process_point_lookup_cache_key", None) == cache_key:
            cached = getattr(self, "_process_point_lookup_cache", None)
            if cached:
                return cached

        def _safe_float(value):
            try:
                numeric = float(value)
            except Exception:
                return float("nan")
            return numeric if np.isfinite(numeric) else float("nan")

        lookup = {}
        for row_idx, row in enumerate(getattr(self, "data", None) or []):
            raw_key = self._get_process_row_sample_line(row, fallback=row_idx)
            ap_val = _safe_float(row.get("ap"))
            ae_val = _safe_float(row.get("ae"))
            aligned_val = _safe_float(row.get("line_no_aligned", raw_key))
            speed_val = _safe_float(row.get("S"))
            feed_value = row.get("F_program")
            if feed_value is None:
                feed_value = row.get("feed_effective", row.get("F_plan"))
            feed_plan = _safe_float(feed_value)
            kc_val = _safe_float(row.get("K_c", row.get("K", self.get_kc_value())))
            if not np.isfinite(feed_plan) or feed_plan < 0.0:
                feed_plan = 0.0
            process_mrr = 0.0
            if (
                np.isfinite(ap_val)
                and np.isfinite(ae_val)
                and ap_val > 1e-12
                and ae_val > 1e-12
                and feed_plan > 1e-12
            ):
                process_mrr = float(ap_val * ae_val * feed_plan / 60.0)

            bucket = lookup.setdefault(
                raw_key,
                {
                    "line_no_aligned": [],
                    "ap": [],
                    "ae": [],
                    "feed_plan": [],
                    "speed_plan": [],
                    "process_mrr": [],
                    "process_kc": [],
                    "process_row_index": [],
                    "process_point_index": [],
                },
            )
            bucket["line_no_aligned"].append(int(round(aligned_val)) if np.isfinite(aligned_val) else int(raw_key))
            bucket["ap"].append(float(ap_val) if np.isfinite(ap_val) else np.nan)
            bucket["ae"].append(float(ae_val) if np.isfinite(ae_val) else np.nan)
            bucket["feed_plan"].append(float(feed_plan) if np.isfinite(feed_plan) else np.nan)
            bucket["speed_plan"].append(float(speed_val) if np.isfinite(speed_val) else np.nan)
            bucket["process_mrr"].append(float(process_mrr) if np.isfinite(process_mrr) else np.nan)
            bucket["process_kc"].append(float(kc_val) if np.isfinite(kc_val) else float(self.get_kc_value()))
            bucket["process_row_index"].append(int(row_idx))
            bucket["process_point_index"].append(
                int(row.get("process_point_index", len(bucket["line_no_aligned"]) - 1) or 0)
            )
        if not lookup:
            raise ValueError("当前工艺信息文件中未找到可映射的工艺点")

        for raw_line, bucket in lookup.items():
            for key in (
                "line_no_aligned",
                "ap",
                "ae",
                "feed_plan",
                "speed_plan",
                "process_mrr",
                "process_kc",
                "process_row_index",
                "process_point_index",
            ):
                bucket[key] = np.asarray(bucket[key])
            bucket["point_count"] = int(len(bucket["ap"]))
            point_count = max(int(bucket["point_count"]), 1)
            bucket["process_anchor_x"] = float(raw_line) + bucket["process_point_index"].astype(float) / float(point_count)
        self._process_point_lookup_cache = lookup
        self._process_point_lookup_cache_key = cache_key
        return lookup

    def _build_aligned_process_geometry_frame(self, raw_line_numbers):
        raw_lines = np.asarray(raw_line_numbers, dtype=int)
        columns = {
            "line_no_aligned": (float, np.nan),
            "ap": (float, np.nan),
            "ae": (float, np.nan),
            "feed_plan": (float, np.nan),
            "speed_plan": (float, np.nan),
            "process_mrr": (float, np.nan),
            "process_kc": (float, np.nan),
            "process_point_index": (int, -1),
            "process_point_count": (int, 0),
            "process_row_index": (int, -1),
            "process_point_anchor_x": (float, np.nan),
            "sample_anchor_x": (float, np.nan),
            "process_anchor_mask": (bool, False),
            "process_anchor_index": (int, -1),
        }
        if raw_lines.size == 0:
            return pd.DataFrame({name: np.asarray([], dtype=dtype) for name, (dtype, _fill) in columns.items()})

        values = {
            name: np.full(len(raw_lines), fill, dtype=dtype)
            for name, (dtype, fill) in columns.items()
        }
        point_lookup = self._build_process_point_lookup()
        start = 0
        while start < len(raw_lines):
            raw_line = int(raw_lines[start])
            end = start + 1
            while end < len(raw_lines) and int(raw_lines[end]) == raw_line:
                end += 1
            bucket = point_lookup.get(raw_line)
            target_slice = slice(start, end)
            if bucket and int(bucket.get("point_count", 0) or 0) > 0:
                point_count = int(bucket["point_count"])
                local_count = end - start
                mapped_index = np.floor(
                    (np.arange(local_count, dtype=float) + 0.5) * point_count / float(local_count)
                ).astype(int)
                mapped_index = np.clip(mapped_index, 0, point_count - 1)
                for target, source in (
                    ("line_no_aligned", "line_no_aligned"),
                    ("ap", "ap"),
                    ("ae", "ae"),
                    ("feed_plan", "feed_plan"),
                    ("speed_plan", "speed_plan"),
                    ("process_mrr", "process_mrr"),
                    ("process_kc", "process_kc"),
                    ("process_point_index", "process_point_index"),
                    ("process_row_index", "process_row_index"),
                    ("process_point_anchor_x", "process_anchor_x"),
                ):
                    values[target][target_slice] = bucket[source][mapped_index]
                values["process_point_count"][target_slice] = point_count
                values["sample_anchor_x"][target_slice] = raw_line + np.arange(local_count, dtype=float) / float(local_count)
                if local_count >= point_count:
                    anchor_offsets = np.floor(np.arange(point_count) * local_count / float(point_count)).astype(int)
                    anchor_indices = np.arange(point_count, dtype=int)
                else:
                    anchor_offsets = np.arange(local_count, dtype=int)
                    anchor_indices = np.floor(
                        (np.arange(local_count, dtype=float) + 0.5) * point_count / float(local_count)
                    ).astype(int)
                    anchor_indices = np.clip(anchor_indices, 0, point_count - 1)
                anchor_pairs = {}
                for local_offset, anchor_idx in zip(anchor_offsets, anchor_indices):
                    anchor_pairs.setdefault(int(local_offset), int(anchor_idx))
                for local_offset in sorted(anchor_pairs):
                    absolute_idx = start + local_offset
                    values["process_anchor_mask"][absolute_idx] = True
                    values["process_anchor_index"][absolute_idx] = anchor_pairs[local_offset]
            else:
                values["line_no_aligned"][target_slice] = float(raw_line)
                values["ap"][target_slice] = 0.0
                values["ae"][target_slice] = 0.0
                values["feed_plan"][target_slice] = 0.0
                values["process_mrr"][target_slice] = 0.0
                values["process_kc"][target_slice] = float(self.get_kc_value())
            start = end
        return pd.DataFrame(values)

    def _get_current_sample_line_point_context(self, line_numbers=None):
        if line_numbers is None:
            sample_lines = np.asarray(getattr(self, "sample_data_line_numbers", []), dtype=int)
            use_global_context = True
            blocks = getattr(self, "sample_data_base_blocks", None)
        else:
            sample_lines = np.asarray(line_numbers, dtype=int)
            global_lines = np.asarray(getattr(self, "sample_data_line_numbers", []), dtype=int)
            use_global_context = global_lines.size == sample_lines.size and np.array_equal(global_lines, sample_lines)
            blocks = getattr(self, "sample_data_base_blocks", None) if use_global_context else None
        if sample_lines.size == 0:
            return None

        point_indices = getattr(self, "sample_data_point_indices", None)
        if use_global_context and point_indices is not None and len(point_indices) == len(sample_lines):
            sample_points = np.asarray(point_indices, dtype=int)
        else:
            sample_points = np.asarray(self.compute_line_point_indices(sample_lines, blocks=blocks), dtype=int)

        x_positions = getattr(self, "sample_data_x_positions", None)
        if use_global_context and x_positions is not None and len(x_positions) == len(sample_lines):
            x_positions = np.asarray(x_positions, dtype=float)
        else:
            try:
                local_blocks = self.compute_sequence_blocks(sample_lines) if hasattr(self, "compute_sequence_blocks") else None
                x_positions = np.asarray(self.compute_line_x_positions(sample_lines, blocks=local_blocks), dtype=float)
            except Exception:
                x_positions = np.asarray(
                    sample_lines.astype(float) + sample_points.astype(float) / np.maximum(
                        np.bincount(np.unique(sample_lines, return_inverse=True)[1])[np.unique(sample_lines, return_inverse=True)[1]],
                        1,
                    ),
                    dtype=float,
                )

        time_getter = getattr(self, "get_sample_time_indices_array", None)
        time_positions = time_getter() if callable(time_getter) else None
        if time_positions is None or len(time_positions) != len(sample_lines):
            time_positions = np.arange(len(sample_lines), dtype=float)
        else:
            time_positions = np.asarray(time_positions, dtype=float)

        point_widths = getattr(self, "sample_data_point_widths", None)
        if use_global_context and point_widths is not None and len(point_widths) == len(sample_lines):
            point_widths = np.asarray(point_widths, dtype=float)
        else:
            width_builder = getattr(self, "compute_line_point_widths", None)
            point_widths = (
                np.asarray(width_builder(sample_lines), dtype=float)
                if callable(width_builder)
                else np.ones(len(sample_lines), dtype=float)
            )

        sort_order = np.argsort(sample_lines, kind="stable")
        sorted_lines = sample_lines[sort_order]
        unique_lines, first_positions = np.unique(sorted_lines, return_index=True)
        end_positions = np.append(first_positions[1:], sorted_lines.size)
        return {
            "line_numbers": sample_lines,
            "point_indices": sample_points,
            "point_numbers": sample_points + 1,
            "x_positions": x_positions,
            "time_positions": time_positions,
            "point_widths": point_widths,
            "line_index_lookup": {
                int(line_no): sort_order[int(start_pos):int(end_pos)]
                for line_no, start_pos, end_pos in zip(unique_lines, first_positions, end_positions)
            },
        }

    def _get_process_point_anchor_x(self, line_no, point_no, process_rows=None):
        try:
            line_value = int(line_no)
        except Exception:
            return float("nan")
        if point_no is None:
            return float(line_value)
        rows = process_rows if process_rows is not None else (getattr(self, "data", None) or [])
        rows = rows if isinstance(rows, list) else list(rows)
        if not rows:
            return float("nan")
        target_point_idx = max(int(point_no) - 1, 0)
        same_line_rows = []
        for row_idx, row in enumerate(rows):
            if self._get_process_row_sample_line(row, fallback=row_idx) != line_value:
                continue
            point_idx = max(int(row.get("process_point_index", len(same_line_rows)) or 0), 0)
            point_count = max(int(row.get("process_point_count", 0) or 0), 0)
            same_line_rows.append((point_idx, point_count))
            if point_idx == target_point_idx and point_count > 0:
                return float(line_value) + float(point_idx) / float(point_count)
        if not same_line_rows:
            return float("nan")
        total_points = max(
            max((item[1] for item in same_line_rows), default=0),
            max((item[0] for item in same_line_rows), default=0) + 1,
        )
        target_point_idx = max(0, min(target_point_idx, total_points - 1))
        return float(line_value) + float(target_point_idx) / float(max(total_points, 1))

    def _resolve_interval_process_x_bounds(self, interval, process_bounds=None, process_rows=None):
        resolved = process_bounds if isinstance(process_bounds, dict) else self._resolve_interval_process_bounds(
            interval,
            process_rows=process_rows,
        )
        if not resolved:
            return None
        rows = process_rows if process_rows is not None else (getattr(self, "data", None) or [])
        rows = rows if isinstance(rows, list) else list(rows)
        try:
            start_idx = int(resolved["start_idx"])
            end_idx = int(resolved["end_idx"])
            start_line = int(resolved["start_line"])
            end_line = int(resolved["end_line"])
            start_point_idx = int(resolved.get("start_point_index", 0))
            end_point_idx = int(resolved.get("end_point_index", 0))
        except (KeyError, TypeError, ValueError):
            return None

        def _anchor(row_idx, line_no, point_idx):
            point_count = int(rows[row_idx].get("process_point_count", 0) or 0)
            if point_count > 0:
                return float(line_no) + float(max(point_idx, 0)) / float(point_count)
            return self._get_process_point_anchor_x(line_no, int(point_idx) + 1, process_rows=rows)

        start_x = _anchor(start_idx, start_line, start_point_idx)
        end_x = _anchor(end_idx, end_line, end_point_idx)
        display_end_x = float("nan")
        if end_idx + 1 < len(rows):
            next_row = rows[end_idx + 1]
            next_line = self._get_process_row_sample_line(next_row, fallback=end_line)
            next_point_idx = int(next_row.get("process_point_index", end_point_idx + 1) or 0)
            display_end_x = _anchor(end_idx + 1, next_line, next_point_idx)
        if not np.isfinite(display_end_x):
            point_count = int(rows[end_idx].get("process_point_count", 0) or 0)
            if point_count > 0:
                display_end_x = float(end_line) + float(end_point_idx + 1) / float(point_count)
        if not np.isfinite(display_end_x):
            display_end_x = float(end_x)
        if np.isfinite(end_x) and display_end_x <= end_x:
            display_end_x = float(end_x) + 1e-9
        return {
            "process_start_x": float(start_x),
            "process_end_x": float(end_x),
            "process_display_end_x": float(display_end_x),
        }

    def _resolve_interval_sample_bounds(self, interval, line_numbers=None):
        if not isinstance(interval, dict):
            return None
        if self._has_authoritative_segmentation_state():
            getter = getattr(self, "_get_authoritative_segmentation_sample_records", None)
            if callable(getter):
                interval_id = str(interval.get("interval_id") or interval.get("zone_id") or "").strip()
                try:
                    projected_records = getter()
                except Exception:
                    projected_records = []
                for projected in projected_records or []:
                    projected_id = str(projected.get("interval_id") or projected.get("zone_id") or "").strip()
                    if projected_id == interval_id:
                        return dict(projected)

        sample_lines = np.asarray(
            line_numbers if line_numbers is not None else getattr(self, "sample_data_line_numbers", []),
            dtype=int,
        )
        sample_size = int(sample_lines.size)
        if sample_size == 0:
            return None
        try:
            start_idx = int(interval.get("sample_start_idx"))
            end_idx = int(interval.get("sample_end_idx"))
        except (TypeError, ValueError):
            start_idx = end_idx = None
        if start_idx is None or end_idx is None or not (0 <= start_idx < sample_size and 0 <= end_idx < sample_size):
            start_line_from_label, _start_point = self._parse_line_point_label(interval.get("start_label"))
            end_line_from_label, _end_point = self._parse_line_point_label(interval.get("end_label"))
            try:
                start_line = int(start_line_from_label if start_line_from_label is not None else interval.get("start_line"))
                end_line = int(end_line_from_label if end_line_from_label is not None else interval.get("end_line"))
            except (TypeError, ValueError):
                return None
            matching = np.flatnonzero(
                (sample_lines >= min(start_line, end_line)) & (sample_lines <= max(start_line, end_line))
            )
            if matching.size == 0:
                return None
            start_idx, end_idx = int(matching[0]), int(matching[-1])
        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx
        point_indices = None
        if hasattr(self, "compute_line_point_indices"):
            try:
                point_indices = np.asarray(self.compute_line_point_indices(sample_lines), dtype=int)
            except Exception:
                point_indices = None
        if point_indices is None or point_indices.size != sample_size:
            point_indices = np.zeros(sample_size, dtype=int)
        formatter = getattr(self, "format_line_point", None)
        start_label = formatter(sample_lines[start_idx], point_indices[start_idx]) if callable(formatter) else f"{sample_lines[start_idx]}.{point_indices[start_idx] + 1}"
        end_label = formatter(sample_lines[end_idx], point_indices[end_idx]) if callable(formatter) else f"{sample_lines[end_idx]}.{point_indices[end_idx] + 1}"
        return {
            "sample_start_idx": int(start_idx),
            "sample_end_idx": int(end_idx),
            "sample_start_line": int(sample_lines[start_idx]),
            "sample_end_line": int(sample_lines[end_idx]),
            "sample_start_label": str(start_label),
            "sample_end_label": str(end_label),
            "start_label": str(start_label),
            "end_label": str(end_label),
            "sample_count": int(end_idx - start_idx + 1),
        }

    def _build_interval_sample_mask(self, interval, sample_size, line_numbers=None):
        size = int(sample_size)
        mask = np.zeros(max(size, 0), dtype=bool)
        if size <= 0:
            return mask
        bounds = self._resolve_interval_sample_bounds(interval, line_numbers=line_numbers)
        if not bounds:
            return mask
        start_idx = int(bounds["sample_start_idx"])
        end_idx = int(bounds["sample_end_idx"])
        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx
        if start_idx < 0 or end_idx >= size:
            return mask
        mask[start_idx:end_idx + 1] = True
        return mask

    def _get_interval_sample_index_span(self, interval, line_numbers=None):
        bounds = self._resolve_interval_sample_bounds(interval, line_numbers=line_numbers)
        if not bounds:
            return None
        return int(bounds["sample_start_idx"]), int(bounds["sample_end_idx"])

    def _resolve_interval_process_bounds(self, interval, process_rows=None):
        rows = process_rows if process_rows is not None else (getattr(self, "data", None) or [])
        rows = rows if isinstance(rows, list) else list(rows)
        if not rows:
            return None
        if process_rows is None:
            self._ensure_process_point_metadata()
            rows = getattr(self, "data", None) or []
        try:
            start_idx = int(interval.get("start_idx"))
            end_idx = int(interval.get("end_idx"))
        except (TypeError, ValueError):
            try:
                start_line = int(interval.get("start_line"))
                end_line = int(interval.get("end_line"))
            except (TypeError, ValueError):
                return None
            matching = [
                idx
                for idx, row in enumerate(rows)
                if min(start_line, end_line) <= self._get_process_row_sample_line(row, idx) <= max(start_line, end_line)
            ]
            if not matching:
                return None
            start_idx, end_idx = matching[0], matching[-1]
        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx
        if start_idx < 0 or end_idx >= len(rows):
            return None
        start_row, end_row = rows[start_idx], rows[end_idx]
        start_line = self._get_process_row_sample_line(start_row, start_idx)
        end_line = self._get_process_row_sample_line(end_row, end_idx)
        start_point = max(int(start_row.get("process_point_index", 0) or 0), 0)
        end_point = max(int(end_row.get("process_point_index", 0) or 0), 0)
        formatter = getattr(self, "format_line_point", None)
        return {
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "start_line": int(start_line),
            "end_line": int(end_line),
            "start_point_index": int(start_point),
            "end_point_index": int(end_point),
            "process_start_label": formatter(start_line, start_point) if callable(formatter) else f"{start_line}.{start_point + 1}",
            "process_end_label": formatter(end_line, end_point) if callable(formatter) else f"{end_line}.{end_point + 1}",
        }


__all__ = ["IntervalRuntimeMixin"]
