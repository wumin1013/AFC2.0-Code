from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .prediction_support import (
    append_inverse_prediction_channels,
    estimate_idle_noise_and_mrr_gate,
    fit_nonnegative_kc_ke,
    robust_sigma,
)
from .shared import PROJECT_ROOT


RELEASE_IDLE_POWER_W = 250.0
IIPINC_FEED_SCALE = 0.6e-4
IIPINC_SINGLE_FEED_SCALE = 0.6


class IipincFormatError(ValueError):
    """``iipinc.txt`` 不满足发布版成对指令进给契约。"""


def _parse_iipinc_numeric_prefix(tokens, physical_line_number):
    if len(tokens) < 8:
        raise IipincFormatError(
            f"iipinc.txt 第 {physical_line_number} 行少于 8 列"
        )
    try:
        values = np.asarray([float(value) for value in tokens[:8]], dtype=float)
    except (TypeError, ValueError) as exc:
        raise IipincFormatError(
            f"iipinc.txt 第 {physical_line_number} 行前 8 列包含非法数值"
        ) from exc
    if not np.all(np.isfinite(values)):
        raise IipincFormatError(
            f"iipinc.txt 第 {physical_line_number} 行前 8 列包含非有限数值"
        )
    return values


def parse_iipinc_rows(rows):
    """自动识别单行或相邻双行格式，并按零基物理行号分组。"""
    feeds_by_line = {}
    pending_values = None
    paired_format = True
    physical_row_count = 0
    declared_period_count = None
    for source_line_number, raw_row in enumerate(rows, start=1):
        if isinstance(raw_row, str):
            tokens = raw_row.strip().split()
        else:
            tokens = [str(value) for value in raw_row]
        if not tokens:
            continue
        if (
            len(tokens) == 3
            and tokens[0].casefold() == "total"
            and tokens[1].casefold() == "periods"
        ):
            if declared_period_count is not None:
                raise IipincFormatError("iipinc.txt 包含重复的 Total periods 汇总行")
            try:
                declared_period_count = int(tokens[2])
            except ValueError as exc:
                raise IipincFormatError(
                    f"iipinc.txt 第 {source_line_number} 行 Total periods 数量不是整数"
                ) from exc
            if declared_period_count < 0:
                raise IipincFormatError(
                    f"iipinc.txt 第 {source_line_number} 行 Total periods 数量为负数"
                )
            continue
        if declared_period_count is not None:
            raise IipincFormatError("iipinc.txt 的 Total periods 汇总行必须位于文件末尾")
        physical_row_count += 1
        values = _parse_iipinc_numeric_prefix(tokens, source_line_number)

        one_based_line = float(values[0])
        rounded_line = int(round(one_based_line))
        if rounded_line < 1 or abs(one_based_line - rounded_line) > 1e-9:
            raise IipincFormatError(
                f"iipinc.txt 第 {source_line_number} 行第一列不是正的一基整数行号"
            )
        raw_feed = float(values[7])
        if raw_feed < 0.0:
            raise IipincFormatError(
                f"iipinc.txt 第 {source_line_number} 行第八列指令进给为负数"
            )
        zero_based_line = rounded_line - 1
        feeds_by_line.setdefault(zero_based_line, []).append(raw_feed)

        if pending_values is None:
            pending_values = values
            continue
        if not np.array_equal(pending_values, values):
            paired_format = False
        pending_values = None

    if physical_row_count == 0:
        raise IipincFormatError("iipinc.txt 为空")
    if (
        declared_period_count is not None
        and declared_period_count != physical_row_count
    ):
        raise IipincFormatError(
            "iipinc.txt 的 Total periods 数量与数据行数不一致："
            f"声明 {declared_period_count}，实际 {physical_row_count}"
        )

    paired_format = paired_format and pending_values is None
    if paired_format:
        feeds_by_line = {
            line_number: values[::2]
            for line_number, values in feeds_by_line.items()
        }
    feed_scale = IIPINC_FEED_SCALE if paired_format else IIPINC_SINGLE_FEED_SCALE
    normalized = {
        int(line_number): np.asarray(values, dtype=float) * feed_scale
        for line_number, values in feeds_by_line.items()
    }
    point_count = sum(len(values) for values in normalized.values())
    return {
        "feeds_by_line": normalized,
        "physical_row_count": int(physical_row_count),
        "deduplicated_point_count": int(point_count),
        "declared_period_count": declared_period_count,
        "feed_scale": float(feed_scale),
        "row_format": "paired" if paired_format else "single",
        "unique_line_count": int(len(normalized)),
    }


def read_iipinc_file(path):
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8-sig", errors="strict") as stream:
        payload = parse_iipinc_rows(stream)
    payload["path"] = str(file_path.resolve())
    return payload


def map_iipinc_feed_to_samples(raw_line_numbers, feeds_by_line):
    """用与工艺映射相同的中点规则，把每行指令点均匀覆盖到样本。"""
    raw_lines = np.asarray(raw_line_numbers, dtype=int).reshape(-1)
    mapped_feed = np.full(raw_lines.shape, np.nan, dtype=float)
    covered_mask = np.zeros(raw_lines.shape, dtype=bool)
    start = 0
    while start < raw_lines.size:
        raw_line = int(raw_lines[start])
        end = start + 1
        while end < raw_lines.size and int(raw_lines[end]) == raw_line:
            end += 1
        line_feed = feeds_by_line.get(raw_line)
        if line_feed is not None:
            line_feed = np.asarray(line_feed, dtype=float).reshape(-1)
            if line_feed.size and np.all(np.isfinite(line_feed)):
                local_count = end - start
                mapped_index = np.floor(
                    (np.arange(local_count, dtype=float) + 0.5)
                    * line_feed.size
                    / float(local_count)
                ).astype(int)
                mapped_index = np.clip(mapped_index, 0, line_feed.size - 1)
                mapped_feed[start:end] = line_feed[mapped_index]
                covered_mask[start:end] = True
        start = end
    return mapped_feed, covered_mask


def _file_signature(path):
    try:
        file_path = Path(path).resolve()
        stat = file_path.stat()
    except (OSError, TypeError, ValueError):
        return (str(path or ""), False, 0, 0)
    return (str(file_path), True, int(stat.st_size), int(stat.st_mtime_ns))


class ReleasePredictionMixin:
    """AFC2.0.2alpha 发布版的会话内反解与负载预测。"""

    release_idle_power_w = RELEASE_IDLE_POWER_W

    def _set_inverse_prediction_status(self, text):
        message = str(text or "")
        self._release_prediction_status = message
        variable = getattr(self, "inverse_prediction_status_var", None)
        if variable is not None and hasattr(variable, "set"):
            variable.set(message)

    def invalidate_release_prediction_cache(self):
        self._release_prediction_cache_key = None
        self._release_prediction_cache_payload = None

    def _resolve_release_iipinc_path(self):
        override = getattr(self, "_release_iipinc_path", None)
        if override:
            return Path(override)
        return Path(PROJECT_ROOT) / "iipinc.txt"

    def _get_cached_iipinc_payload(self, file_path):
        signature = _file_signature(file_path)
        cached = getattr(self, "_release_iipinc_parse_cache", None)
        if isinstance(cached, dict) and cached.get("signature") == signature:
            return cached.get("payload"), cached.get("error", ""), signature

        if not signature[1]:
            payload = None
            error = "未找到 EXE 同目录的 iipinc.txt"
        else:
            try:
                payload = read_iipinc_file(file_path)
                error = ""
            except (OSError, UnicodeError, IipincFormatError) as exc:
                payload = None
                error = str(exc)
        self._release_iipinc_parse_cache = {
            "signature": signature,
            "payload": payload,
            "error": error,
        }
        return payload, error, signature

    @staticmethod
    def _process_row_feed_value(row):
        try:
            value = float(row.get("feed_effective"))
        except (AttributeError, TypeError, ValueError):
            return float("nan")
        return value if np.isfinite(value) else float("nan")

    @staticmethod
    def _process_row_raw_line(row, missing_value):
        try:
            value = float(row.get("line_no_raw"))
        except (AttributeError, TypeError, ValueError):
            return int(missing_value), False
        rounded = int(round(value)) if np.isfinite(value) else int(missing_value)
        if not np.isfinite(value) or abs(value - rounded) > 1e-9 or rounded < 0:
            return int(missing_value), False
        return rounded, True

    def _resolve_processinfo_export_feeds(self, process_rows):
        """发布版导出 ProcessInfo 时优先使用逐点指令进给。"""
        rows = list(process_rows or [])
        original_feed = np.asarray(
            [self._process_row_feed_value(row) for row in rows], dtype=float
        )
        raw_lines = []
        valid_raw_line = np.zeros(len(rows), dtype=bool)
        for index, row in enumerate(rows):
            raw_line, is_valid = self._process_row_raw_line(row, -index - 1)
            raw_lines.append(raw_line)
            valid_raw_line[index] = is_valid

        iip_path = self._resolve_release_iipinc_path()
        iip_payload, iip_error, _signature = self._get_cached_iipinc_payload(iip_path)
        exported_feed = original_feed.copy()
        covered_mask = np.zeros(len(rows), dtype=bool)
        if iip_payload is not None and rows:
            command_feed, covered_mask = map_iipinc_feed_to_samples(
                raw_lines, iip_payload["feeds_by_line"]
            )
            covered_mask = np.asarray(covered_mask, dtype=bool) & np.isfinite(command_feed)
            exported_feed[covered_mask] = command_feed[covered_mask]

        replaced_count = int(np.sum(covered_mask))
        total_count = len(rows)
        valid_lines = set(
            int(raw_line)
            for raw_line, is_valid in zip(raw_lines, valid_raw_line)
            if is_valid
        )
        covered_lines = set(
            int(raw_line)
            for raw_line, is_covered in zip(raw_lines, covered_mask)
            if is_covered
        )
        if iip_payload is None:
            status_text = (
                "ProcessInfo 进给：未读取到可用的 iipinc.txt，F 保留编程进给"
            )
        elif replaced_count == 0:
            status_text = (
                "ProcessInfo 进给：iipinc.txt 没有匹配到当前工艺行，F 保留编程进给"
            )
        elif replaced_count == total_count:
            status_text = (
                f"ProcessInfo 进给：已用指令进给替换全部 {total_count} 个工艺点"
            )
        else:
            status_text = (
                f"ProcessInfo 进给：已用指令进给替换 {replaced_count}/{total_count} "
                f"个工艺点；其余 {total_count - replaced_count} 个点保留编程进给"
            )
        self._last_processinfo_feed_export_status = status_text
        self._last_processinfo_feed_export = {
            "iipinc_valid": iip_payload is not None,
            "iipinc_error": str(iip_error or ""),
            "process_point_count": int(total_count),
            "replaced_point_count": replaced_count,
            "process_line_count": int(len(valid_lines)),
            "covered_line_count": int(len(covered_lines)),
            "fallback_line_count": int(len(valid_lines - covered_lines)),
            "status_text": status_text,
        }
        return exported_feed

    def _release_process_content_signature(self):
        rows = getattr(self, "data", None) or []
        signature_rows = []
        for row in rows:
            if not isinstance(row, dict):
                signature_rows.append((str(row),))
                continue
            feed = row.get("F_program")
            if feed is None:
                feed = row.get("feed_effective", row.get("F_plan"))
            signature_rows.append(
                (
                    row.get("line_no_raw"),
                    row.get("line_no_aligned"),
                    row.get("process_point_index"),
                    row.get("ap"),
                    row.get("ae"),
                    feed,
                )
            )
        return tuple(tuple(str(value) for value in row) for row in signature_rows)

    def _release_prediction_signature(self, values, raw_lines, iip_signature):
        sample_mode = str(getattr(self, "sample_data_mode", "") or "").strip()
        measurement_path = (
            getattr(self, "manual_measurement_path", None)
            if sample_mode == "experiment_measurement"
            else getattr(self, "sample_csv_path", None)
        )
        return (
            sample_mode,
            _file_signature(measurement_path),
            _file_signature(getattr(self, "sample_txt_path", None)),
            id(getattr(self, "sample_data_values", values)),
            tuple(values.shape),
            id(getattr(self, "sample_data_line_numbers", raw_lines)),
            tuple(raw_lines.shape),
            self._release_process_content_signature(),
            tuple(iip_signature),
            float(self.release_idle_power_w),
        )

    def _update_release_fit_variables(self, fit_result):
        setters = (
            (getattr(self, "p_idle_var", None), self.release_idle_power_w),
            (getattr(self, "kc_coeff", None), f"{fit_result['kc_value']:.12g}"),
            (getattr(self, "ke_coeff", None), f"{fit_result['ke_value']:.12g}"),
        )
        for variable, value in setters:
            if variable is not None and hasattr(variable, "set"):
                variable.set(value)

    def has_prediction_model_ready(self, process_path=None):
        del process_path
        return bool(
            getattr(self, "sample_data_loaded", False)
            and getattr(self, "sample_data_values", None) is not None
            and getattr(self, "data", None)
        )

    def get_prediction_curve_label(self, mode=None):
        del mode
        return "预测负载"

    def _build_sampledata_prediction_payload_for_mode(self):
        sample_mode = str(getattr(self, "sample_data_mode", "") or "").strip()
        if sample_mode == "sampledata":
            actual_column = 1
            actual_label = "VGpro功率"
        elif sample_mode == "experiment_measurement":
            actual_column = 0
            actual_label = "实际负载"
        else:
            return None
        if not self.has_prediction_model_ready():
            self._set_inverse_prediction_status("预测负载：等待自动生成")
            return None

        values = np.asarray(self.sample_data_values, dtype=float)
        raw_lines = np.asarray(
            getattr(self, "sample_data_line_numbers", []), dtype=int
        ).reshape(-1)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] <= actual_column:
            self._set_inverse_prediction_status(f"预测负载：未生成；实测数据缺少{actual_label}列")
            return None
        if raw_lines.size != values.shape[0]:
            self._set_inverse_prediction_status("预测负载：未生成；采样行号与功率点数不一致")
            return None

        iip_path = self._resolve_release_iipinc_path()
        iip_payload, iip_error, iip_signature = self._get_cached_iipinc_payload(iip_path)
        cache_key = self._release_prediction_signature(values, raw_lines, iip_signature)
        if cache_key == getattr(self, "_release_prediction_cache_key", None):
            cached_payload = getattr(self, "_release_prediction_cache_payload", None)
            if isinstance(cached_payload, dict):
                self._set_inverse_prediction_status(cached_payload.get("status_text", ""))
                return cached_payload

        actual_load = np.abs(np.asarray(values[:, actual_column], dtype=float))
        try:
            process_frame = self._build_aligned_process_geometry_frame(raw_lines)
        except Exception as exc:
            self._set_inverse_prediction_status(f"预测负载：未生成；工艺匹配失败（{exc}）")
            return None
        if len(process_frame) != raw_lines.size:
            self._set_inverse_prediction_status("预测负载：未生成；工艺匹配长度不一致")
            return None

        ap_values = pd.to_numeric(process_frame["ap"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        ae_values = pd.to_numeric(process_frame["ae"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        process_feed = pd.to_numeric(process_frame["feed_plan"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        process_mrr = ap_values * ae_values * process_feed / 60.0
        fixed_idle = np.full(raw_lines.size, float(self.release_idle_power_w), dtype=float)
        prediction_valid = np.isfinite(actual_load)
        sample_frame = pd.DataFrame(
            {
                "sample_index": np.arange(raw_lines.size, dtype=int),
                "line_no_raw": raw_lines,
                "actual_load": actual_load,
                "idle_power": fixed_idle,
                "ap": ap_values,
                "ae": ae_values,
                "feed_speed": process_feed,
                "mrr": process_mrr,
                "prediction_valid": prediction_valid,
                "process_anchor_mask": np.asarray(
                    process_frame["process_anchor_mask"], dtype=bool
                ),
            }
        )
        sigma_idle, delta_mrr, idle_sample_count, idle_mask = (
            estimate_idle_noise_and_mrr_gate(sample_frame, kc_reference=1.0)
        )
        # 固定 250 W 时，大量未映射工艺行不能作为空载模型样本，否则实际
        # 加工功率波动会被误当成空载噪声并抬高 MRR 门限。只用工艺锚点中
        # 的非切削点估计突变容差；本发布版不建立空载模型，故不启用由
        # Kc 参考值推导的 MRR 门限。
        process_anchor_mask = np.asarray(
            process_frame["process_anchor_mask"], dtype=bool
        )
        anchor_idle_mask = (
            process_anchor_mask
            & prediction_valid
            & (
                (process_mrr <= 1e-12)
                | (ap_values <= 1e-12)
                | (ae_values <= 1e-12)
                | (process_feed <= 1e-12)
            )
        )
        anchor_idle_residuals = (
            actual_load[anchor_idle_mask] - float(self.release_idle_power_w)
        )
        if anchor_idle_residuals.size >= 2:
            sigma_idle = robust_sigma(anchor_idle_residuals)
            if not np.isfinite(sigma_idle) or sigma_idle < 0.0:
                sigma_idle = 0.0
        else:
            sigma_idle = 0.0
        idle_sample_count = int(anchor_idle_residuals.size)
        delta_mrr = 0.0
        sample_frame = append_inverse_prediction_channels(
            sample_frame,
            sigma_idle=sigma_idle,
            delta_mrr=delta_mrr,
            idle_mask=idle_mask,
            ke_value=0.0,
        )
        fit_mask = np.asarray(sample_frame["kc_valid"], dtype=bool)
        try:
            fit_result = fit_nonnegative_kc_ke(
                process_mrr[fit_mask],
                ap_values[fit_mask],
                actual_load[fit_mask] - float(self.release_idle_power_w),
                source_label="发布版有效切削锚点",
            )
        except ValueError as exc:
            predicted_load = np.full(actual_load.shape, np.nan, dtype=float)
            status_text = f"预测负载：未生成（{exc}）；实际负载和区间结果仍可使用"
            payload = {
                "actual_label": actual_label,
                "actual_load": actual_load,
                "program_line": raw_lines,
                "predicted_load": predicted_load,
                "prediction_valid_mask": prediction_valid,
                "fit_error": str(exc),
                "status_text": status_text,
                "sample_frame": sample_frame,
            }
            self._release_prediction_cache_key = cache_key
            self._release_prediction_cache_payload = payload
            self._set_inverse_prediction_status(status_text)
            return payload

        if iip_payload is not None:
            iip_feed, iip_covered_mask = map_iipinc_feed_to_samples(
                raw_lines,
                iip_payload["feeds_by_line"],
            )
        else:
            iip_feed = np.full(raw_lines.shape, np.nan, dtype=float)
            iip_covered_mask = np.zeros(raw_lines.shape, dtype=bool)
        command_feed = process_feed.copy()
        command_feed[iip_covered_mask] = iip_feed[iip_covered_mask]

        finite_geometry = (
            np.isfinite(ap_values)
            & np.isfinite(ae_values)
            & np.isfinite(command_feed)
        )
        cutting_mask = (
            finite_geometry
            & (ap_values > 1e-12)
            & (ae_values > 1e-12)
            & (command_feed > 1e-12)
        )
        command_mrr = np.zeros(raw_lines.shape, dtype=float)
        command_mrr[cutting_mask] = (
            ap_values[cutting_mask]
            * ae_values[cutting_mask]
            * command_feed[cutting_mask]
            / 60.0
        )
        predicted_load = np.full(raw_lines.shape, float(self.release_idle_power_w), dtype=float)
        predicted_load[cutting_mask] = (
            float(self.release_idle_power_w)
            + float(fit_result["kc_value"]) * command_mrr[cutting_mask]
            + float(fit_result["ke_value"]) * ap_values[cutting_mask]
        )
        predicted_load[~prediction_valid] = np.nan

        process_cutting_mask = (
            np.isfinite(ap_values)
            & np.isfinite(ae_values)
            & np.isfinite(process_feed)
            & (ap_values > 1e-12)
            & (ae_values > 1e-12)
            & (process_feed > 1e-12)
        )
        cutting_lines = set(int(value) for value in raw_lines[process_cutting_mask])
        covered_lines = set(
            int(value)
            for value in raw_lines[process_cutting_mask & iip_covered_mask]
        )
        fallback_lines = cutting_lines - covered_lines
        fallback_point_count = int(
            np.sum(process_cutting_mask & ~iip_covered_mask)
        )

        if iip_payload is None:
            feed_source_text = "未读取到可用指令进给，已使用编程进给"
        elif fallback_lines:
            feed_source_text = (
                f"指令进给覆盖 {len(covered_lines)}/{len(cutting_lines)} 个加工行，"
                f"其余 {len(fallback_lines)} 行使用编程进给"
            )
        else:
            feed_source_text = (
                f"指令进给已覆盖全部 {len(cutting_lines)} 个加工行"
            )
        status_text = (
            f"预测负载：已生成（空载按 {self.release_idle_power_w:.0f} W）；"
            f"{feed_source_text}"
        )
        self._update_release_fit_variables(fit_result)
        self._set_inverse_prediction_status(status_text)

        sample_frame["process_feed"] = process_feed
        sample_frame["command_feed"] = command_feed
        sample_frame["command_feed_from_iip"] = iip_covered_mask
        sample_frame["command_mrr"] = command_mrr
        sample_frame["predicted_load"] = predicted_load
        payload = {
            "actual_label": actual_label,
            "actual_load": actual_load,
            "program_line": raw_lines,
            "line_no_aligned": pd.to_numeric(
                process_frame["line_no_aligned"], errors="coerce"
            ).fillna(pd.Series(raw_lines)).to_numpy(dtype=int),
            "predicted_idle_power": fixed_idle,
            "mapped_ap": ap_values,
            "mapped_ae": ae_values,
            "mapped_feed": command_feed,
            "mapped_process_feed": process_feed,
            "mapped_mrr": command_mrr,
            "predicted_load": predicted_load,
            "prediction_valid_mask": prediction_valid,
            "fit_result": dict(fit_result),
            "sigma_idle": float(sigma_idle),
            "delta_mrr": float(delta_mrr),
            "idle_sample_count": int(idle_sample_count),
            "iipinc_valid": iip_payload is not None,
            "iipinc_error": str(iip_error or ""),
            "iipinc_physical_row_count": int(
                iip_payload.get("physical_row_count", 0) if iip_payload else 0
            ),
            "iipinc_point_count": int(
                iip_payload.get("deduplicated_point_count", 0) if iip_payload else 0
            ),
            "iipinc_row_format": str(
                iip_payload.get("row_format", "") if iip_payload else ""
            ),
            "iipinc_feed_scale": float(
                iip_payload.get("feed_scale", 0.0) if iip_payload else 0.0
            ),
            "iipinc_covered_line_count": int(len(covered_lines)),
            "process_cutting_line_count": int(len(cutting_lines)),
            "fallback_line_count": int(len(fallback_lines)),
            "fallback_point_count": fallback_point_count,
            "status_text": status_text,
            "sample_frame": sample_frame,
        }
        self._release_prediction_cache_key = cache_key
        self._release_prediction_cache_payload = payload
        return payload


__all__ = [
    "IIPINC_FEED_SCALE",
    "IIPINC_SINGLE_FEED_SCALE",
    "IipincFormatError",
    "RELEASE_IDLE_POWER_W",
    "ReleasePredictionMixin",
    "map_iipinc_feed_to_samples",
    "parse_iipinc_rows",
    "read_iipinc_file",
]
