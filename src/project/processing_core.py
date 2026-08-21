from __future__ import annotations

import csv

from .shared import *

PROCESS_LAYOUT_DIRECT_SEQ = "direct_seq"
PROCESS_LAYOUT_DIRECT_NO_SEQ = "direct_no_seq"
PROCESS_LAYOUT_EXPORT_SEQ = "export_seq"
PROCESS_LAYOUT_EXPORT_NO_SEQ = "export_no_seq"
PROCESS_LAYOUT_DIRECT_SEQ_NO_MRR = "direct_seq_no_mrr"
PROCESS_LAYOUT_DIRECT_NO_SEQ_NO_MRR = "direct_no_seq_no_mrr"
PROCESS_LAYOUT_EXPORT_SEQ_NO_MRR = "export_seq_no_mrr"
PROCESS_LAYOUT_EXPORT_NO_SEQ_NO_MRR = "export_no_seq_no_mrr"
PROCESS_LAYOUT_DIRECT_SEQ_NO_S = "direct_seq_no_s"
PROCESS_LAYOUT_DIRECT_NO_SEQ_NO_S = "direct_no_seq_no_s"
PROCESS_LAYOUT_EXPORT_SEQ_NO_S = "export_seq_no_s"
PROCESS_LAYOUT_EXPORT_NO_SEQ_NO_S = "export_no_seq_no_s"
PROCESS_LAYOUT_DIRECT_SEQ_NO_MRR_NO_S = "direct_seq_no_mrr_no_s"
PROCESS_LAYOUT_DIRECT_NO_SEQ_NO_MRR_NO_S = "direct_no_seq_no_mrr_no_s"
PROCESS_LAYOUT_EXPORT_SEQ_NO_MRR_NO_S = "export_seq_no_mrr_no_s"
PROCESS_LAYOUT_EXPORT_NO_SEQ_NO_MRR_NO_S = "export_no_seq_no_mrr_no_s"
PROCESS_LAYOUT_LEGACY_SEQ = "legacy_seq"
PROCESS_LAYOUT_LEGACY_NO_SEQ = "legacy_no_seq"
PROCESS_LAYOUT_STATE_SUFFIX = "__with_state_code"

PROCESS_LAYOUT_GCODE_INDEX = {
    PROCESS_LAYOUT_DIRECT_SEQ: 8,
    PROCESS_LAYOUT_DIRECT_NO_SEQ: 7,
    PROCESS_LAYOUT_EXPORT_SEQ: 8,
    PROCESS_LAYOUT_EXPORT_NO_SEQ: 7,
    PROCESS_LAYOUT_DIRECT_SEQ_NO_MRR: 7,
    PROCESS_LAYOUT_DIRECT_NO_SEQ_NO_MRR: 6,
    PROCESS_LAYOUT_EXPORT_SEQ_NO_MRR: 7,
    PROCESS_LAYOUT_EXPORT_NO_SEQ_NO_MRR: 6,
    PROCESS_LAYOUT_DIRECT_SEQ_NO_S: 7,
    PROCESS_LAYOUT_DIRECT_NO_SEQ_NO_S: 6,
    PROCESS_LAYOUT_EXPORT_SEQ_NO_S: 7,
    PROCESS_LAYOUT_EXPORT_NO_SEQ_NO_S: 6,
    PROCESS_LAYOUT_DIRECT_SEQ_NO_MRR_NO_S: 6,
    PROCESS_LAYOUT_DIRECT_NO_SEQ_NO_MRR_NO_S: 5,
    PROCESS_LAYOUT_EXPORT_SEQ_NO_MRR_NO_S: 6,
    PROCESS_LAYOUT_EXPORT_NO_SEQ_NO_MRR_NO_S: 5,
    PROCESS_LAYOUT_LEGACY_SEQ: 6,
    PROCESS_LAYOUT_LEGACY_NO_SEQ: 5,
}

PROCESS_HEADER_MAP = {
    ("序号", "N", "ap(mm)", "ae(mm)", "F(mm/min)", "MRR(mm3/min)", "s(mm)", "S(r/min)", "G"): PROCESS_LAYOUT_DIRECT_SEQ,
    ("N", "ap(mm)", "ae(mm)", "F(mm/min)", "MRR(mm3/min)", "s(mm)", "S(r/min)", "G"): PROCESS_LAYOUT_DIRECT_NO_SEQ,
    ("序号", "N", "S(r/min)", "ap(mm)", "ae(mm)", "F(mm/min)", "s(mm)", "MRR(mm3/min)", "G"): PROCESS_LAYOUT_EXPORT_SEQ,
    ("N", "S(r/min)", "ap(mm)", "ae(mm)", "F(mm/min)", "s(mm)", "MRR(mm3/min)", "G"): PROCESS_LAYOUT_EXPORT_NO_SEQ,
    ("序号", "N", "ap(mm)", "ae(mm)", "F(mm/min)", "s(mm)", "S(r/min)", "G"): PROCESS_LAYOUT_DIRECT_SEQ_NO_MRR,
    ("N", "ap(mm)", "ae(mm)", "F(mm/min)", "s(mm)", "S(r/min)", "G"): PROCESS_LAYOUT_DIRECT_NO_SEQ_NO_MRR,
    ("序号", "N", "S(r/min)", "ap(mm)", "ae(mm)", "F(mm/min)", "s(mm)", "G"): PROCESS_LAYOUT_EXPORT_SEQ_NO_MRR,
    ("N", "S(r/min)", "ap(mm)", "ae(mm)", "F(mm/min)", "s(mm)", "G"): PROCESS_LAYOUT_EXPORT_NO_SEQ_NO_MRR,
    ("序号", "N", "ap(mm)", "ae(mm)", "F(mm/min)", "MRR(mm3/min)", "S(r/min)", "G"): PROCESS_LAYOUT_DIRECT_SEQ_NO_S,
    ("N", "ap(mm)", "ae(mm)", "F(mm/min)", "MRR(mm3/min)", "S(r/min)", "G"): PROCESS_LAYOUT_DIRECT_NO_SEQ_NO_S,
    ("序号", "N", "S(r/min)", "ap(mm)", "ae(mm)", "F(mm/min)", "MRR(mm3/min)", "G"): PROCESS_LAYOUT_EXPORT_SEQ_NO_S,
    ("N", "S(r/min)", "ap(mm)", "ae(mm)", "F(mm/min)", "MRR(mm3/min)", "G"): PROCESS_LAYOUT_EXPORT_NO_SEQ_NO_S,
    ("序号", "N", "ap(mm)", "ae(mm)", "F(mm/min)", "S(r/min)", "G"): PROCESS_LAYOUT_DIRECT_SEQ_NO_MRR_NO_S,
    ("N", "ap(mm)", "ae(mm)", "F(mm/min)", "S(r/min)", "G"): PROCESS_LAYOUT_DIRECT_NO_SEQ_NO_MRR_NO_S,
    ("序号", "N", "S(r/min)", "ap(mm)", "ae(mm)", "F(mm/min)", "G"): PROCESS_LAYOUT_EXPORT_SEQ_NO_MRR_NO_S,
    ("N", "S(r/min)", "ap(mm)", "ae(mm)", "F(mm/min)", "G"): PROCESS_LAYOUT_EXPORT_NO_SEQ_NO_MRR_NO_S,
    ("序号", "N", "ap(mm)", "ae(mm)", "F(mm/min)", "MRR(mm3/min)", "G"): PROCESS_LAYOUT_LEGACY_SEQ,
    ("N", "ap(mm)", "ae(mm)", "F(mm/min)", "MRR(mm3/min)", "G"): PROCESS_LAYOUT_LEGACY_NO_SEQ,
}


class ProcessingCoreMixin:
    def _canonicalize_process_header_token(self, token):
        """将不同版本表头归一到现有布局关键字。"""
        raw = str(token or "").strip().lstrip("\ufeff")
        if not raw:
            return ""
        normalized = raw.replace("（", "(").replace("）", ")").replace(" ", "")
        lowered = normalized.lower()

        if lowered in {"序号", "index", "idx", "no"}:
            return "序号"
        if lowered in {"n", "程序行号", "行号", "line", "lineno", "line_no", "linenumber"}:
            return "N"
        if lowered in {"ap", "a_p", "ap(mm)", "a_p(mm)", "切深", "轴向切深"}:
            return "ap(mm)"
        if lowered in {"ae", "a_e", "ae(mm)", "a_e(mm)", "切宽", "径向切宽"}:
            return "ae(mm)"
        if lowered in {"f", "f(mm/min)", "feed", "feedrate", "feed_rate", "进给", "进给率", "进给速度"}:
            return "F(mm/min)"
        if lowered in {
            "mrr",
            "mrr(mm3/min)",
            "mrr(mm^3/min)",
            "mrr(mm3/s)",
            "mrr(mm^3/s)",
            "材料去除率",
            "materialremovalrate",
        }:
            # 布局键保留旧名仅为兼容；该输入列不参与 MRR 计算。
            return "MRR(mm3/min)"
        if normalized == "S" or "rpm" in lowered or "r/min" in lowered or "转速" in normalized:
            return "S(r/min)"
        if normalized == "s" or lowered in {
            "s(mm)",
            "path",
            "pathlength",
            "input_path_cumulative",
            "inputpathcumulative",
            "s_cumulative",
            "scumulative",
            "path_cumulative",
            "pathcumulative",
            "行程",
            "累计行程",
            "累计路径",
            "cumulativepath",
            "cumulativedistance",
            "accumulatedpath",
            "增量行程",
            "逐行行程",
            "当前行行程",
            "incrementalpath",
            "perrowpath",
            "路径长度",
        }:
            return "s(mm)"
        if lowered in {"g", "gcode", "nc", "程序段", "代码"}:
            return "G"
        if lowered in {
            "state",
            "statecode",
            "state_code",
            "segmentstate",
            "segment_state",
            "区间状态",
            "六态",
            "六态标识",
        }:
            return "state_code"
        return normalized

    def _parse_prefixed_numeric_token(self, token, prefix):
        """解析带前缀的数值标记，例如 N12 / S5000。"""
        text = str(token or "").strip()
        if not text:
            return None
        if text.upper().startswith(prefix.upper()):
            text = text[len(prefix):].strip()
        try:
            return float(text)
        except Exception:
            return None

    def _detect_process_header_layout(self, tokens):
        """根据表头识别工艺信息列布局。"""
        if not tokens:
            return None
        normalized = tuple(
            self._canonicalize_process_header_token(token)
            for token in tokens
            if str(token).strip()
        )
        if normalized and normalized[-1] == "state_code":
            base_layout = PROCESS_HEADER_MAP.get(normalized[:-1])
            if base_layout:
                return f"{base_layout}{PROCESS_LAYOUT_STATE_SUFFIX}"
        return PROCESS_HEADER_MAP.get(normalized)

    def _is_process_header_row(self, tokens):
        """判断是否为工艺信息表头行。"""
        return self._detect_process_header_layout(tokens) is not None

    def _detect_process_path_semantics_hint(self, tokens):
        """按表头契约区分逐行行程与显式累计行程。"""
        for token in tokens or []:
            if self._canonicalize_process_header_token(token) != "s(mm)":
                continue
            normalized = str(token or "").strip().lower().replace(" ", "")
            if "累计" in normalized or "cumulative" in normalized or "accumulated" in normalized:
                return "cumulative"
            if (
                "增量" in normalized
                or "逐行" in normalized
                or "当前行" in normalized
                or "increment" in normalized
                or "perrow" in normalized
            ):
                return "incremental"
            # 普通 s / s(mm) 的固定契约是当前行增量，不再根据数值
            # 是否恰好单调来猜测累计语义。
            return "incremental"
        return None

    def _parse_process_numeric(self, token, prefix=""):
        return self._parse_prefixed_numeric_token(token, prefix)

    def _build_process_parse_result(
        self,
        *,
        layout_name,
        line_token,
        ap_token,
        ae_token,
        feed_token,
        gcode_tokens,
        s_token=None,
        spindle_token=None,
        mrr_token=None,
    ):
        gcode_content = " ".join(gcode_tokens).replace(",", " ").strip()

        line_no_val = self._parse_process_numeric(line_token, "N")
        ap_val = self._parse_process_numeric(ap_token)
        ae_val = self._parse_process_numeric(ae_token)
        feed_val = self._parse_process_numeric(feed_token)
        if ap_val is None or ae_val is None or feed_val is None:
            return None
        if ap_val < 0 or ae_val < 0 or feed_val < 0:
            return None

        path_value = None
        if s_token is not None and str(s_token).strip():
            path_value = self._parse_process_numeric(s_token)

        spindle_val = None
        if spindle_token is not None and str(spindle_token).strip():
            spindle_val = self._parse_process_numeric(spindle_token, "S")
            if spindle_val is None:
                spindle_val = self._parse_process_numeric(spindle_token)
            if spindle_val is None or spindle_val < 0:
                return None

        line_number = None
        if line_no_val is not None:
            try:
                numeric_line = float(line_no_val)
                rounded_line = int(round(numeric_line))
                if (
                    math.isfinite(numeric_line)
                    and rounded_line >= 1
                    and abs(numeric_line - rounded_line) <= 1e-9
                ):
                    line_number = rounded_line - 1
            except Exception:
                line_number = None

        score = 0.0
        if line_number is not None and line_number >= 0:
            score += 3.0
        if ap_val <= 100:
            score += 1.0
        else:
            score -= 3.0
        if ae_val <= 100:
            score += 1.0
        else:
            score -= 3.0
        if feed_val > 0:
            score += 1.0
        if spindle_val is not None and spindle_val >= 100:
            score += 1.0
        if path_value is not None and path_value > 0:
            score += 1.0

        return {
            "layout": layout_name,
            "line_number": line_number,
            "ap": float(ap_val),
            "ae": float(ae_val),
            "feed_rate": float(feed_val),
            "gcode_content": gcode_content,
            "spindle_speed": float(spindle_val) if spindle_val is not None else None,
            # 普通 s(mm) 固定是逐行增量；只有显式 cumulative 表头
            # 才使用累计语义。这里先保留原始数值，稍后统一生成路径。
            "path_value": float(path_value) if path_value is not None else None,
            "path_column_present": s_token is not None,
            # MRR 列只参与布局定位。其单元格既不解析、也不校验，业务 MRR
            # 始终在派生列回填阶段由 ap * ae * F / 60 重新计算。
            "score": float(score),
        }

    def _infer_process_layout_record(self, numeric_tokens, gcode_tokens):
        """在没有表头提示时，根据列值形态推断工艺信息布局。"""
        candidates = []
        token_count = len(numeric_tokens)

        if token_count >= 8:
            candidates.append(
                self._build_process_parse_result(
                    layout_name=PROCESS_LAYOUT_DIRECT_SEQ,
                    line_token=numeric_tokens[1],
                    ap_token=numeric_tokens[2],
                    ae_token=numeric_tokens[3],
                    feed_token=numeric_tokens[4],
                    mrr_token=numeric_tokens[5],
                    s_token=numeric_tokens[6],
                    spindle_token=numeric_tokens[7],
                    gcode_tokens=gcode_tokens,
                )
            )
            candidates.append(
                self._build_process_parse_result(
                    layout_name=PROCESS_LAYOUT_EXPORT_SEQ,
                    line_token=numeric_tokens[1],
                    spindle_token=numeric_tokens[2],
                    ap_token=numeric_tokens[3],
                    ae_token=numeric_tokens[4],
                    feed_token=numeric_tokens[5],
                    s_token=numeric_tokens[6],
                    mrr_token=numeric_tokens[7],
                    gcode_tokens=gcode_tokens,
                )
            )

        if token_count >= 7:
            candidates.append(
                self._build_process_parse_result(
                    layout_name=PROCESS_LAYOUT_DIRECT_NO_SEQ,
                    line_token=numeric_tokens[0],
                    ap_token=numeric_tokens[1],
                    ae_token=numeric_tokens[2],
                    feed_token=numeric_tokens[3],
                    mrr_token=numeric_tokens[4],
                    s_token=numeric_tokens[5],
                    spindle_token=numeric_tokens[6],
                    gcode_tokens=gcode_tokens,
                )
            )
            candidates.append(
                self._build_process_parse_result(
                    layout_name=PROCESS_LAYOUT_EXPORT_NO_SEQ,
                    line_token=numeric_tokens[0],
                    spindle_token=numeric_tokens[1],
                    ap_token=numeric_tokens[2],
                    ae_token=numeric_tokens[3],
                    feed_token=numeric_tokens[4],
                    s_token=numeric_tokens[5],
                    mrr_token=numeric_tokens[6],
                    gcode_tokens=gcode_tokens,
                )
            )

        if token_count >= 5:
            candidates.append(
                self._build_process_parse_result(
                    layout_name=PROCESS_LAYOUT_LEGACY_SEQ,
                    line_token=numeric_tokens[1],
                    ap_token=numeric_tokens[2],
                    ae_token=numeric_tokens[3],
                    feed_token=numeric_tokens[4],
                    gcode_tokens=gcode_tokens,
                )
            )
            candidates.append(
                self._build_process_parse_result(
                    layout_name=PROCESS_LAYOUT_LEGACY_NO_SEQ,
                    line_token=numeric_tokens[0],
                    ap_token=numeric_tokens[1],
                    ae_token=numeric_tokens[2],
                    feed_token=numeric_tokens[3],
                    gcode_tokens=gcode_tokens,
                )
            )

        valid_candidates = [candidate for candidate in candidates if candidate]
        if not valid_candidates:
            return None
        return max(valid_candidates, key=lambda item: float(item.get("score", 0.0)))

    def _parse_process_tokens(self, tokens, layout_hint=None):
        """兼容旧格式与新格式工艺信息行。"""
        # CSV 空单元格具有列定位语义，尤其是允许为空的 N、s 和 G 列，不能删除。
        cleaned = [str(token).strip() for token in tokens]
        if len(cleaned) < 5:
            return None, layout_hint

        detected_layout = self._detect_process_header_layout(cleaned)
        if detected_layout:
            return None, detected_layout

        reported_layout = layout_hint
        if (
            isinstance(layout_hint, str)
            and layout_hint.endswith(PROCESS_LAYOUT_STATE_SUFFIX)
        ):
            if not cleaned:
                return None, reported_layout
            cleaned = cleaned[:-1]
            layout_hint = layout_hint[:-len(PROCESS_LAYOUT_STATE_SUFFIX)]

        numeric_tokens, gcode_tokens = self._split_numeric_and_gcode_tokens(
            cleaned,
            layout_hint=layout_hint,
        )
        if not numeric_tokens or not gcode_tokens or len(numeric_tokens) < 4:
            return None, reported_layout or layout_hint

        result = None
        effective_layout = reported_layout or layout_hint

        if layout_hint == PROCESS_LAYOUT_DIRECT_SEQ and len(numeric_tokens) >= 8:
            result = self._build_process_parse_result(
                layout_name=layout_hint,
                line_token=numeric_tokens[1],
                ap_token=numeric_tokens[2],
                ae_token=numeric_tokens[3],
                feed_token=numeric_tokens[4],
                mrr_token=numeric_tokens[5],
                s_token=numeric_tokens[6],
                spindle_token=numeric_tokens[7],
                gcode_tokens=gcode_tokens,
            )
        elif layout_hint == PROCESS_LAYOUT_DIRECT_NO_SEQ and len(numeric_tokens) >= 7:
            result = self._build_process_parse_result(
                layout_name=layout_hint,
                line_token=numeric_tokens[0],
                ap_token=numeric_tokens[1],
                ae_token=numeric_tokens[2],
                feed_token=numeric_tokens[3],
                mrr_token=numeric_tokens[4],
                s_token=numeric_tokens[5],
                spindle_token=numeric_tokens[6],
                gcode_tokens=gcode_tokens,
            )
        elif layout_hint == PROCESS_LAYOUT_EXPORT_SEQ and len(numeric_tokens) >= 8:
            result = self._build_process_parse_result(
                layout_name=layout_hint,
                line_token=numeric_tokens[1],
                spindle_token=numeric_tokens[2],
                ap_token=numeric_tokens[3],
                ae_token=numeric_tokens[4],
                feed_token=numeric_tokens[5],
                s_token=numeric_tokens[6],
                mrr_token=numeric_tokens[7],
                gcode_tokens=gcode_tokens,
            )
        elif layout_hint == PROCESS_LAYOUT_EXPORT_NO_SEQ and len(numeric_tokens) >= 7:
            result = self._build_process_parse_result(
                layout_name=layout_hint,
                line_token=numeric_tokens[0],
                spindle_token=numeric_tokens[1],
                ap_token=numeric_tokens[2],
                ae_token=numeric_tokens[3],
                feed_token=numeric_tokens[4],
                s_token=numeric_tokens[5],
                mrr_token=numeric_tokens[6],
                gcode_tokens=gcode_tokens,
            )
        elif layout_hint == PROCESS_LAYOUT_DIRECT_SEQ_NO_MRR and len(numeric_tokens) >= 7:
            result = self._build_process_parse_result(
                layout_name=layout_hint,
                line_token=numeric_tokens[1],
                ap_token=numeric_tokens[2],
                ae_token=numeric_tokens[3],
                feed_token=numeric_tokens[4],
                s_token=numeric_tokens[5],
                spindle_token=numeric_tokens[6],
                gcode_tokens=gcode_tokens,
            )
        elif layout_hint == PROCESS_LAYOUT_DIRECT_NO_SEQ_NO_MRR and len(numeric_tokens) >= 6:
            result = self._build_process_parse_result(
                layout_name=layout_hint,
                line_token=numeric_tokens[0],
                ap_token=numeric_tokens[1],
                ae_token=numeric_tokens[2],
                feed_token=numeric_tokens[3],
                s_token=numeric_tokens[4],
                spindle_token=numeric_tokens[5],
                gcode_tokens=gcode_tokens,
            )
        elif layout_hint == PROCESS_LAYOUT_EXPORT_SEQ_NO_MRR and len(numeric_tokens) >= 7:
            result = self._build_process_parse_result(
                layout_name=layout_hint,
                line_token=numeric_tokens[1],
                spindle_token=numeric_tokens[2],
                ap_token=numeric_tokens[3],
                ae_token=numeric_tokens[4],
                feed_token=numeric_tokens[5],
                s_token=numeric_tokens[6],
                gcode_tokens=gcode_tokens,
            )
        elif layout_hint == PROCESS_LAYOUT_EXPORT_NO_SEQ_NO_MRR and len(numeric_tokens) >= 6:
            result = self._build_process_parse_result(
                layout_name=layout_hint,
                line_token=numeric_tokens[0],
                spindle_token=numeric_tokens[1],
                ap_token=numeric_tokens[2],
                ae_token=numeric_tokens[3],
                feed_token=numeric_tokens[4],
                s_token=numeric_tokens[5],
                gcode_tokens=gcode_tokens,
            )
        elif layout_hint == PROCESS_LAYOUT_DIRECT_SEQ_NO_S and len(numeric_tokens) >= 7:
            result = self._build_process_parse_result(
                layout_name=layout_hint,
                line_token=numeric_tokens[1],
                ap_token=numeric_tokens[2],
                ae_token=numeric_tokens[3],
                feed_token=numeric_tokens[4],
                mrr_token=numeric_tokens[5],
                spindle_token=numeric_tokens[6],
                gcode_tokens=gcode_tokens,
            )
        elif layout_hint == PROCESS_LAYOUT_DIRECT_NO_SEQ_NO_S and len(numeric_tokens) >= 6:
            result = self._build_process_parse_result(
                layout_name=layout_hint,
                line_token=numeric_tokens[0],
                ap_token=numeric_tokens[1],
                ae_token=numeric_tokens[2],
                feed_token=numeric_tokens[3],
                mrr_token=numeric_tokens[4],
                spindle_token=numeric_tokens[5],
                gcode_tokens=gcode_tokens,
            )
        elif layout_hint == PROCESS_LAYOUT_EXPORT_SEQ_NO_S and len(numeric_tokens) >= 7:
            result = self._build_process_parse_result(
                layout_name=layout_hint,
                line_token=numeric_tokens[1],
                spindle_token=numeric_tokens[2],
                ap_token=numeric_tokens[3],
                ae_token=numeric_tokens[4],
                feed_token=numeric_tokens[5],
                mrr_token=numeric_tokens[6],
                gcode_tokens=gcode_tokens,
            )
        elif layout_hint == PROCESS_LAYOUT_EXPORT_NO_SEQ_NO_S and len(numeric_tokens) >= 6:
            result = self._build_process_parse_result(
                layout_name=layout_hint,
                line_token=numeric_tokens[0],
                spindle_token=numeric_tokens[1],
                ap_token=numeric_tokens[2],
                ae_token=numeric_tokens[3],
                feed_token=numeric_tokens[4],
                mrr_token=numeric_tokens[5],
                gcode_tokens=gcode_tokens,
            )
        elif layout_hint == PROCESS_LAYOUT_DIRECT_SEQ_NO_MRR_NO_S and len(numeric_tokens) >= 6:
            result = self._build_process_parse_result(
                layout_name=layout_hint,
                line_token=numeric_tokens[1],
                ap_token=numeric_tokens[2],
                ae_token=numeric_tokens[3],
                feed_token=numeric_tokens[4],
                spindle_token=numeric_tokens[5],
                gcode_tokens=gcode_tokens,
            )
        elif layout_hint == PROCESS_LAYOUT_DIRECT_NO_SEQ_NO_MRR_NO_S and len(numeric_tokens) >= 5:
            result = self._build_process_parse_result(
                layout_name=layout_hint,
                line_token=numeric_tokens[0],
                ap_token=numeric_tokens[1],
                ae_token=numeric_tokens[2],
                feed_token=numeric_tokens[3],
                spindle_token=numeric_tokens[4],
                gcode_tokens=gcode_tokens,
            )
        elif layout_hint == PROCESS_LAYOUT_EXPORT_SEQ_NO_MRR_NO_S and len(numeric_tokens) >= 6:
            result = self._build_process_parse_result(
                layout_name=layout_hint,
                line_token=numeric_tokens[1],
                spindle_token=numeric_tokens[2],
                ap_token=numeric_tokens[3],
                ae_token=numeric_tokens[4],
                feed_token=numeric_tokens[5],
                gcode_tokens=gcode_tokens,
            )
        elif layout_hint == PROCESS_LAYOUT_EXPORT_NO_SEQ_NO_MRR_NO_S and len(numeric_tokens) >= 5:
            result = self._build_process_parse_result(
                layout_name=layout_hint,
                line_token=numeric_tokens[0],
                spindle_token=numeric_tokens[1],
                ap_token=numeric_tokens[2],
                ae_token=numeric_tokens[3],
                feed_token=numeric_tokens[4],
                gcode_tokens=gcode_tokens,
            )
        elif layout_hint == PROCESS_LAYOUT_LEGACY_SEQ and len(numeric_tokens) >= 5:
            result = self._build_process_parse_result(
                layout_name=layout_hint,
                line_token=numeric_tokens[1],
                ap_token=numeric_tokens[2],
                ae_token=numeric_tokens[3],
                feed_token=numeric_tokens[4],
                gcode_tokens=gcode_tokens,
            )
        elif layout_hint == PROCESS_LAYOUT_LEGACY_NO_SEQ and len(numeric_tokens) >= 4:
            result = self._build_process_parse_result(
                layout_name=layout_hint,
                line_token=numeric_tokens[0],
                ap_token=numeric_tokens[1],
                ae_token=numeric_tokens[2],
                feed_token=numeric_tokens[3],
                gcode_tokens=gcode_tokens,
            )

        # 已有表头时列语义是确定的；该布局下缺少必要字段的行应跳过，
        # 不能再用其他布局猜测，否则会把 S=5000 等值错读为 ap。
        if result is None and layout_hint is None:
            result = self._infer_process_layout_record(numeric_tokens, gcode_tokens)
            if result:
                effective_layout = result.get("layout")

        if not result:
            return None, effective_layout

        result.pop("score", None)
        return result, effective_layout

    def _choose_process_geometry_layout(self, numeric_tokens, gcode_tokens):
        """工艺信息统一按 [序号, 行号, ap, ae, F, ...] 读取。"""
        if len(numeric_tokens) < 5:
            return None
        return numeric_tokens[2], numeric_tokens[3], numeric_tokens[4]

    def _split_numeric_and_gcode_tokens(self, tokens, layout_hint=None):
        """拆分工艺信息行：数值列与G代码列"""
        gcode_column = PROCESS_LAYOUT_GCODE_INDEX.get(layout_hint)
        if gcode_column is not None and len(tokens) >= gcode_column:
            return tokens[:gcode_column], tokens[gcode_column:]

        gcode_start_idx = None
        for idx, token in enumerate(tokens):
            # 无表头时只把“字母地址 + 数值”识别为 G 指令起点；这样 MRR
            # 占位单元格即使是任意非数值文本，也不会改变列拆分结果。
            if re.match(
                r'^(?:N\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*)?'
                r'(?:G|M|X|Y|Z|I|J|K|F|S|T|L|R|P|Q)\s*'
                r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)',
                str(token).strip(),
                flags=re.IGNORECASE,
            ):
                gcode_start_idx = idx
                break
        if gcode_start_idx is None and tokens and not str(tokens[-1]).strip():
            gcode_start_idx = len(tokens) - 1
        if gcode_start_idx is None or gcode_start_idx == 0:
            return None, None
        return tokens[:gcode_start_idx], tokens[gcode_start_idx:]

    def _parse_simulation_csv_line(self, line, layout_hint=None):
        """解析仿真导出的 CSV 行。

        格式：
        序号, 程序行号, 切深, 切宽, 进给, G代码
        首行表头会自动跳过。
        """
        raw_line = str(line or "").strip().lstrip("\ufeff")
        if not raw_line or "," not in raw_line:
            return None, layout_hint

        try:
            parts = next(csv.reader([raw_line]))
        except Exception:
            return None, layout_hint

        return self._parse_process_tokens(parts, layout_hint=layout_hint)

    def parse_gcode_line(self, line, layout_hint=None, return_layout=False):
        """解析G代码行"""
        parsed, updated_layout = self._parse_simulation_csv_line(line, layout_hint=layout_hint)
        if parsed is None and "," not in str(line or ""):
            tokens = str(line or "").strip().split()
            parsed, updated_layout = self._parse_process_tokens(tokens, layout_hint=layout_hint)
        if return_layout:
            return parsed, updated_layout
        return parsed

    def _create_modal_gcode_state(self):
        """创建 G 代码模态状态。"""
        return {
            "motion": None,
            "plane": "G17",
            "distance_mode": "G90",
            "arc_center_mode": "incremental",
        }

    def _strip_gcode_comments(self, gcode_content):
        """移除括号注释与分号注释。"""
        text = str(gcode_content or "")
        text = re.sub(r"\([^)]*\)", " ", text)
        text = re.sub(r";.*$", " ", text)
        return text.strip()

    def _parse_gcode_words(self, gcode_content):
        """解析 G 代码字地址，兼容无空格写法。"""
        text = self._strip_gcode_comments(gcode_content)
        return [
            (match.group(1).upper(), (match.group(2) or "").strip())
            for match in re.finditer(r"([A-Za-z])\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)?)?", text)
        ]

    def _resolve_modal_gcode_state(self, gcode_content, prev_state):
        """结合当前行更新 G 代码模态状态。"""
        state = dict(prev_state or self._create_modal_gcode_state())
        word_values = {}
        for letter, raw_value in self._parse_gcode_words(gcode_content):
            if letter == "G" and raw_value:
                try:
                    g_value = float(raw_value)
                except ValueError:
                    continue
                if abs(g_value) < 1e-9:
                    state["motion"] = "G0"
                elif abs(g_value - 1.0) < 1e-9:
                    state["motion"] = "G1"
                elif abs(g_value - 2.0) < 1e-9:
                    state["motion"] = "G2"
                elif abs(g_value - 3.0) < 1e-9:
                    state["motion"] = "G3"
                elif abs(g_value - 17.0) < 1e-9:
                    state["plane"] = "G17"
                elif abs(g_value - 18.0) < 1e-9:
                    state["plane"] = "G18"
                elif abs(g_value - 19.0) < 1e-9:
                    state["plane"] = "G19"
                elif abs(g_value - 90.0) < 1e-9:
                    state["distance_mode"] = "G90"
                elif abs(g_value - 91.0) < 1e-9:
                    state["distance_mode"] = "G91"
                elif abs(g_value - 90.1) < 1e-9:
                    state["arc_center_mode"] = "absolute"
                elif abs(g_value - 91.1) < 1e-9:
                    state["arc_center_mode"] = "incremental"
            elif raw_value:
                try:
                    word_values[letter] = float(raw_value)
                except ValueError:
                    continue
        return state, word_values

    def _resolve_gcode_end_coords(self, prev_coords, word_values, distance_mode):
        """结合坐标字地址求当前行终点坐标。"""
        coords = list(prev_coords)
        for axis, idx in (("X", 0), ("Y", 1), ("Z", 2)):
            if axis not in word_values:
                continue
            axis_value = float(word_values[axis])
            if distance_mode == "G91":
                coords[idx] += axis_value
            else:
                coords[idx] = axis_value
        return tuple(coords)

    def _calculate_linear_distance(self, prev_coords, current_coords):
        """计算直线长度。"""
        if prev_coords is None:
            return 0.0
        dx = current_coords[0] - prev_coords[0]
        dy = current_coords[1] - prev_coords[1]
        dz = current_coords[2] - prev_coords[2]
        return math.sqrt(dx**2 + dy**2 + dz**2)

    def _calculate_arc_distance(self, start_coords, end_coords, word_values, state):
        """按活动平面与 IJK 参数计算圆弧/螺旋线长度。"""
        plane = state.get("plane", "G17")
        if plane == "G18":
            plane_axes = (0, 2)
            center_axes = ("I", "K")
            helix_axis = 1
        elif plane == "G19":
            plane_axes = (1, 2)
            center_axes = ("J", "K")
            helix_axis = 0
        else:
            plane_axes = (0, 1)
            center_axes = ("I", "J")
            helix_axis = 2

        if not any(axis in word_values for axis in center_axes):
            return self._calculate_linear_distance(start_coords, end_coords)

        start_a, start_b = start_coords[plane_axes[0]], start_coords[plane_axes[1]]
        end_a, end_b = end_coords[plane_axes[0]], end_coords[plane_axes[1]]

        if state.get("arc_center_mode") == "absolute":
            center_a = float(word_values.get(center_axes[0], start_a))
            center_b = float(word_values.get(center_axes[1], start_b))
        else:
            center_a = start_a + float(word_values.get(center_axes[0], 0.0))
            center_b = start_b + float(word_values.get(center_axes[1], 0.0))

        start_radius = math.hypot(start_a - center_a, start_b - center_b)
        end_radius = math.hypot(end_a - center_a, end_b - center_b)
        radius = (start_radius + end_radius) / 2.0 if start_radius > 1e-9 and end_radius > 1e-9 else max(start_radius, end_radius)
        if radius <= 1e-9:
            return self._calculate_linear_distance(start_coords, end_coords)

        start_angle = math.atan2(start_b - center_b, start_a - center_a)
        end_angle = math.atan2(end_b - center_b, end_a - center_a)
        same_endpoint = abs(start_a - end_a) <= 1e-9 and abs(start_b - end_b) <= 1e-9
        if same_endpoint:
            sweep_angle = 2.0 * math.pi
        else:
            delta_angle = end_angle - start_angle
            if state.get("motion") == "G2":
                if delta_angle >= 0:
                    delta_angle -= 2.0 * math.pi
                sweep_angle = abs(delta_angle)
            else:
                if delta_angle <= 0:
                    delta_angle += 2.0 * math.pi
                sweep_angle = delta_angle

        planar_length = radius * sweep_angle
        helix_delta = end_coords[helix_axis] - start_coords[helix_axis]
        return math.hypot(planar_length, helix_delta)

    def compute_gcode_motion_info(
        self,
        gcode_content,
        prev_coords=None,
        prev_state=None,
        calculate_segment_length=True,
    ):
        """计算当前行的模态状态、终点坐标和运动长度。"""
        state, word_values = self._resolve_modal_gcode_state(gcode_content, prev_state)
        has_prev_coords = prev_coords is not None
        start_coords = tuple(prev_coords if prev_coords is not None else (0.0, 0.0, 0.0))
        end_coords = self._resolve_gcode_end_coords(start_coords, word_values, state.get("distance_mode", "G90"))
        has_motion_words = any(axis in word_values for axis in ("X", "Y", "Z", "I", "J", "K"))

        if not calculate_segment_length or not has_prev_coords or not has_motion_words or state.get("motion") is None:
            segment_length = 0.0
        elif state["motion"] in {"G2", "G3"}:
            segment_length = self._calculate_arc_distance(start_coords, end_coords, word_values, state)
        else:
            segment_length = self._calculate_linear_distance(start_coords, end_coords)

        motion_type = None
        if state.get("motion") == "G0":
            motion_type = "rapid"
        elif state.get("motion") in {"G1", "G2", "G3"}:
            motion_type = "cutting"

        return {
            "state": state,
            "start_coords": start_coords,
            "end_coords": end_coords,
            "segment_length": max(float(segment_length), 0.0),
            "motion_type": motion_type,
        }

    def _process_instruction_group_key(self, row):
        """返回连续指令分组键，优先使用真实文件行号。"""
        raw_line = row.get("line_no_raw")
        if raw_line is not None:
            return "raw", raw_line
        aligned_line = row.get("line_no_aligned")
        if aligned_line is not None:
            return "aligned", aligned_line
        return "gcode", str(row.get("gcode_content", "") or "")

    def _normalize_gcode_for_nc_line_match(self, gcode_content):
        """规范化普通字地址指令，用于 ProcessInfo 与 NC 原文的严格顺序匹配。"""
        text = self._strip_gcode_comments(gcode_content).upper().strip()
        if not text:
            return ""
        text = re.sub(r"^N\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*", "", text)
        word_pattern = re.compile(r"([A-Z])\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))")
        words = list(word_pattern.finditer(text))
        residual = word_pattern.sub("", text)
        if words and not re.sub(r"[\s,]", "", residual):
            normalized_words = []
            for match in words:
                number = float(match.group(2))
                if abs(number) <= 1e-15:
                    number = 0.0
                normalized_words.append(f"{match.group(1)}{number:.15g}")
            return "".join(normalized_words)
        return re.sub(r"[\s,]", "", text)

    def _collect_process_gcode_groups(self, rows):
        """按规范化后的连续 G 指令收集工艺点组。"""
        groups = []
        for row in rows:
            if bool(row.get("_is_synthetic_fill")):
                continue
            normalized = self._normalize_gcode_for_nc_line_match(row.get("gcode_content", ""))
            if not groups or normalized != groups[-1]["normalized"]:
                groups.append({"normalized": normalized, "rows": [row]})
            else:
                groups[-1]["rows"].append(row)
        return groups

    def _restore_missing_line_numbers_from_nc_profile(self, rows):
        """按已有 N 锚点和 NC 指令顺序补全缺失的真实文件行号。"""
        process_rows = [row for row in rows if not bool(row.get("_is_synthetic_fill"))]
        groups = self._collect_process_gcode_groups(process_rows)
        diagnostics = {
            "line_number_source": "missing",
            "nc_process_group_count": len(groups),
            "nc_matched_group_count": 0,
        }
        if not process_rows:
            self.process_line_number_diagnostics = diagnostics
            return diagnostics

        missing_flags = [row.get("line_no_raw") is None for row in process_rows]
        if not any(missing_flags):
            diagnostics["line_number_source"] = "input"
            for row in process_rows:
                row["line_number_source"] = "input"
            self.process_line_number_diagnostics = diagnostics
            return diagnostics

        # AfoMilling 会把同一条 NC 指令细分为多个工艺点。部分细分点的 N
        # 为空时，优先使用最后一列 G 代码把它归入同一连续指令组，并继承该组
        # 已存在的物理行号；这比丢弃已有 N 后整体做比例映射更可靠。
        group_known_lines = []
        completed_group_row_count = 0
        for group in groups:
            known_lines = {
                int(row.get("line_no_raw"))
                for row in group["rows"]
                if row.get("line_no_raw") is not None
            }
            if len(known_lines) > 1:
                diagnostics["line_number_source"] = "partial_gcode_group_anchor_conflict"
                for row in process_rows:
                    row["line_number_source"] = diagnostics["line_number_source"]
                self.process_line_number_diagnostics = diagnostics
                return diagnostics
            known_line = next(iter(known_lines), None)
            group_known_lines.append(known_line)
            if known_line is None:
                continue
            for row in group["rows"]:
                if row.get("line_no_raw") is None:
                    row["line_no_raw"] = int(known_line)
                    row["line_number_source"] = "gcode_group_anchor"
                    completed_group_row_count += 1

        diagnostics["gcode_group_completed_row_count"] = int(completed_group_row_count)
        missing_flags = [row.get("line_no_raw") is None for row in process_rows]
        if not any(missing_flags):
            diagnostics["line_number_source"] = "input_with_gcode_group_completion"
            for row in process_rows:
                row.setdefault("line_number_source", "input")
            self.process_line_number_diagnostics = diagnostics
            return diagnostics

        profile = getattr(self, "gcode_profile", None) or {}
        states = sorted(
            (
                state for state in profile.get("states", [])
                if state.get("file_line_index") is not None
            ),
            key=lambda state: int(state.get("file_line_index")),
        )
        if not states or not groups or any(not group["normalized"] for group in groups):
            diagnostics["line_number_source"] = "missing_nc_match_unavailable"
            for row in process_rows:
                row["line_number_source"] = diagnostics["line_number_source"]
            self.process_line_number_diagnostics = diagnostics
            return diagnostics

        normalized_states = [
            (self._normalize_gcode_for_nc_line_match(state.get("line_text", "")), state)
            for state in states
        ]
        state_position_by_line = {
            int(state.get("file_line_index")): index
            for index, (_normalized, state) in enumerate(normalized_states)
        }
        candidate_positions = []
        for group_index, group in enumerate(groups):
            known_line = group_known_lines[group_index]
            if known_line is not None:
                candidate_index = state_position_by_line.get(int(known_line))
                candidates = []
                if (
                    candidate_index is not None
                    and normalized_states[candidate_index][0] == group["normalized"]
                ):
                    candidates = [int(candidate_index)]
            else:
                candidates = [
                    state_index
                    for state_index, (normalized, _state) in enumerate(normalized_states)
                    if normalized == group["normalized"]
                ]
            if not candidates:
                source_prefix = "partial" if not all(missing_flags) else "missing"
                diagnostics["line_number_source"] = f"{source_prefix}_nc_match_failed"
                for row in process_rows:
                    row["line_number_source"] = diagnostics["line_number_source"]
                self.process_line_number_diagnostics = diagnostics
                return diagnostics
            candidate_positions.append(candidates)

        earliest_positions = []
        previous_position = -1
        for candidates in candidate_positions:
            match_index = next(
                (candidate for candidate in candidates if candidate > previous_position),
                None,
            )
            if match_index is None:
                break
            earliest_positions.append(match_index)
            previous_position = match_index

        latest_positions = []
        next_position = len(normalized_states)
        for candidates in reversed(candidate_positions):
            match_index = next(
                (candidate for candidate in reversed(candidates) if candidate < next_position),
                None,
            )
            if match_index is None:
                break
            latest_positions.append(match_index)
            next_position = match_index
        latest_positions.reverse()

        source_prefix = "partial" if not all(missing_flags) else "missing"
        if (
            len(earliest_positions) != len(groups)
            or len(latest_positions) != len(groups)
        ):
            diagnostics["line_number_source"] = f"{source_prefix}_nc_match_failed"
            for row in process_rows:
                row["line_number_source"] = diagnostics["line_number_source"]
            self.process_line_number_diagnostics = diagnostics
            return diagnostics
        if earliest_positions != latest_positions:
            diagnostics["line_number_source"] = f"{source_prefix}_nc_match_ambiguous"
            for row in process_rows:
                row["line_number_source"] = diagnostics["line_number_source"]
            self.process_line_number_diagnostics = diagnostics
            return diagnostics

        assignments = [
            (group, normalized_states[match_index][1])
            for group, match_index in zip(groups, earliest_positions)
        ]

        # 只有所有连续指令组均匹配成功后才一次性写回，避免部分匹配污染行号。
        completed_partial = not all(missing_flags)
        overall_source = "input_with_nc_completion" if completed_partial else "nc_profile_line_text"
        for group, state in assignments:
            file_line_index = int(state.get("file_line_index"))
            for row in group["rows"]:
                had_input_line = row.get("line_no_raw") is not None
                row["line_no_raw"] = file_line_index
                row["line_number_source"] = "input" if had_input_line else "nc_profile_line_text"
        diagnostics.update({
            "line_number_source": overall_source,
            "nc_matched_group_count": len(assignments),
        })
        self.process_line_number_diagnostics = diagnostics
        return diagnostics

    def _get_segmentation_config_value(self, name, default):
        """读取当前集中分割配置；尚未初始化界面时使用配置类默认值。"""
        config = getattr(self, "segmentation_config", None)
        if config is None:
            config = getattr(self, "_segmentation_config", None)
        if config is None:
            pipeline = getattr(self, "segmentation_pipeline", None)
            config = getattr(pipeline, "config", None)
        if config is None:
            try:
                from .segmentation.schemas import SegmentationConfig

                config = SegmentationConfig()
            except Exception:
                config = None
        if isinstance(config, dict):
            return config.get(name, default)
        return getattr(config, name, default)

    def _validate_input_path(self, rows):
        """按列契约全局验证行程；普通 s 默认是逐点增量。"""
        try:
            path_tolerance = max(float(self._get_segmentation_config_value("path_tolerance_mm", 1e-8)), 0.0)
        except (TypeError, ValueError):
            path_tolerance = 1e-8
        process_rows = [row for row in rows if not bool(row.get("_is_synthetic_fill"))]
        path_present = bool(process_rows) and any(
            bool(row.get("_input_path_column_present")) for row in process_rows
        )
        result = {
            "present": path_present,
            "valid": False,
            "reason": "missing",
            "span": 0.0,
            "semantics": "unknown",
        }
        if not path_present:
            return result
        if not all(bool(row.get("_input_path_column_present")) for row in process_rows):
            result["reason"] = "missing_or_invalid_value"
            return result

        values = []
        for row in process_rows:
            try:
                value = float(row.get("_input_path_value"))
            except (TypeError, ValueError):
                result["reason"] = "missing_or_invalid_value"
                return result
            if not math.isfinite(value):
                result["reason"] = "non_finite"
                return result
            values.append(value)

        semantics_hints = {
            str(row.get("_input_path_semantics_hint") or "").strip().lower()
            for row in process_rows
            if str(row.get("_input_path_semantics_hint") or "").strip()
        }
        if len(semantics_hints) > 1:
            result["reason"] = "conflicting_semantics_hints"
            return result
        semantics_hint = next(iter(semantics_hints), "")

        if any(value < -path_tolerance for value in values):
            result["reason"] = "negative_value"
            return result

        cumulative_monotonic = all(
            current >= previous - path_tolerance * max(1.0, abs(previous), abs(current))
            for previous, current in zip(values, values[1:])
        )
        cumulative_span = float(max(values[-1], 0.0)) if values else 0.0
        incremental_values = [max(value, 0.0) for value in values]
        incremental_total = float(sum(incremental_values))

        if semantics_hint == "cumulative":
            if not cumulative_monotonic or cumulative_span <= path_tolerance:
                result["reason"] = "invalid_cumulative"
                return result
            result.update({
                "valid": True,
                "reason": "valid_cumulative",
                "span": cumulative_span,
                "semantics": "cumulative",
            })
            return result

        if semantics_hint not in {"", "incremental"}:
            result["reason"] = "unknown_semantics_hint"
            return result
        result["span"] = incremental_total
        if incremental_total <= path_tolerance:
            result["reason"] = "no_positive_span"
            return result

        result.update({
            "valid": True,
            "reason": "valid_incremental",
            "span": incremental_total,
            "semantics": "incremental",
        })
        return result

    def _path_positions_are_physical(self, rows):
        """检查已回填的逐点行程是否有限、非回退且具有正物理长度。"""
        try:
            path_tolerance = max(float(self._get_segmentation_config_value("path_tolerance_mm", 1e-8)), 0.0)
        except (TypeError, ValueError):
            path_tolerance = 1e-8
        previous_end = None
        total_length = 0.0
        for row in rows:
            try:
                start_value = float(row.get("path_start"))
                end_value = float(row.get("path_end"))
            except (TypeError, ValueError):
                return False
            if not math.isfinite(start_value) or not math.isfinite(end_value):
                return False
            bound_tolerance = path_tolerance * max(1.0, abs(start_value), abs(end_value))
            if end_value < start_value - bound_tolerance:
                return False
            if previous_end is not None:
                sequence_tolerance = path_tolerance * max(1.0, abs(previous_end), abs(start_value))
            else:
                sequence_tolerance = path_tolerance
            if previous_end is not None and start_value < previous_end - sequence_tolerance:
                return False
            total_length += max(end_value - start_value, 0.0)
            previous_end = end_value
        return total_length > path_tolerance

    def _apply_gcode_geometry_positions(self, rows, origin, calculate_path=True):
        """每个连续指令只计算一次几何，并确定性分配到该指令的工艺点。"""
        previous_coords = tuple(origin)
        modal_state = self._create_modal_gcode_state()
        cumulative_total = 0.0
        row_index = 0
        while row_index < len(rows):
            group_key = self._process_instruction_group_key(rows[row_index])
            group_end = row_index + 1
            while (
                group_end < len(rows)
                and self._process_instruction_group_key(rows[group_end]) == group_key
            ):
                group_end += 1

            group_rows = rows[row_index:group_end]
            motion_info = self.compute_gcode_motion_info(
                group_rows[0].get("gcode_content", ""),
                prev_coords=previous_coords,
                prev_state=modal_state,
                calculate_segment_length=calculate_path,
            )
            modal_state = motion_info["state"]
            start_coords = tuple(motion_info["start_coords"])
            end_coords = tuple(motion_info["end_coords"])
            group_length = float(motion_info["segment_length"]) if calculate_path else 0.0
            group_count = len(group_rows)
            group_step = group_length / group_count if group_count else 0.0

            for offset, row in enumerate(group_rows):
                ratio = (offset + 1) / group_count if group_count else 1.0
                row["x"] = start_coords[0] + (end_coords[0] - start_coords[0]) * ratio
                row["y"] = start_coords[1] + (end_coords[1] - start_coords[1]) * ratio
                row["z"] = start_coords[2] + (end_coords[2] - start_coords[2]) * ratio
                if motion_info["motion_type"] is not None:
                    row["type"] = motion_info["motion_type"]
                if calculate_path:
                    row["s"] = group_step
                    row["path_start"] = cumulative_total
                    cumulative_total += group_step
                    row["path_end"] = cumulative_total
                    row["path_cumulative"] = cumulative_total
            previous_coords = end_coords
            row_index = group_end

    def _assign_group_distributed_path_positions(self, rows):
        """基于连续相同行号，将累计行程均匀分布到每一行。"""
        cumulative_total = 0.0
        row_index = 0
        while row_index < len(rows):
            group_key = self._process_instruction_group_key(rows[row_index])
            group_end = row_index
            group_total = 0.0
            while (
                group_end < len(rows)
                and self._process_instruction_group_key(rows[group_end]) == group_key
            ):
                try:
                    group_total += float(rows[group_end].get("s", 0.0))
                except Exception:
                    pass
                group_end += 1
            group_count = group_end - row_index
            group_step = group_total / group_count if group_count else 0.0
            for offset in range(row_index, group_end):
                rows[offset]["s"] = group_step
                rows[offset]["path_start"] = cumulative_total
                cumulative_total += group_step
                rows[offset]["path_end"] = cumulative_total
                rows[offset]["path_cumulative"] = cumulative_total
            row_index = group_end

    def _apply_input_path_positions(self, rows, semantics):
        """按已判定的累计/增量语义生成统一累计 path_start/path_end。"""
        if semantics == "incremental":
            for row in rows:
                if bool(row.get("_is_synthetic_fill")):
                    row["s"] = 0.0
                    continue
                try:
                    segment_length = float(row.get("_input_path_value", 0.0) or 0.0)
                except Exception:
                    segment_length = 0.0
                row["s"] = max(segment_length, 0.0)
            # 同一指令含多个工艺点时，把该指令总行程确定性地均分到各点。
            self._assign_group_distributed_path_positions(rows)
            return

        previous_end = 0.0
        for row in rows:
            if bool(row.get("_is_synthetic_fill")):
                end_val = previous_end
            else:
                try:
                    end_val = float(row.get("_input_path_value"))
                except (TypeError, ValueError):
                    end_val = previous_end
                if end_val < previous_end:
                    end_val = previous_end
            row["s"] = max(end_val - previous_end, 0.0)
            previous_end = end_val
        # 累计值先转换为逐点增量，再按同一指令的全部工艺点均分。
        self._assign_group_distributed_path_positions(rows)

    def _resolve_exact_nc_state_for_process_row(self, raw_line_number, gcode_content):
        """以已验证文件行号为锚点，并用显式 N 做一致性校验。"""
        profile = getattr(self, "gcode_profile", None)
        if not isinstance(profile, dict) or not profile or raw_line_number is None:
            return None

        try:
            line_index = int(raw_line_number)
        except (TypeError, ValueError):
            return None
        line_state = profile.get("state_by_line_index", {}).get(line_index)
        if not line_state:
            return None

        n_value = self.extract_n_value(str(gcode_content or ""))
        n_int = self.extract_n_integer(n_value)
        if n_int is not None:
            n_state = profile.get("state_by_n", {}).get(int(n_int))
            if not n_state:
                return None
            try:
                n_line_index = int(n_state.get("file_line_index"))
                exact_line_index = int(line_state.get("file_line_index"))
            except (TypeError, ValueError):
                return None
            if n_line_index != exact_line_index:
                return None
        return dict(line_state)

    def _apply_bound_nc_profile_positions(self, rows):
        """尝试使用已绑定 NC profile；不完整或无正跨度时返回 False。"""
        try:
            path_tolerance = max(float(self._get_segmentation_config_value("path_tolerance_mm", 1e-8)), 0.0)
        except (TypeError, ValueError):
            path_tolerance = 1e-8
        cumulative_total = 0.0
        all_required_groups_matched = True
        row_index = 0
        while row_index < len(rows):
            group_key = self._process_instruction_group_key(rows[row_index])
            group_end = row_index + 1
            while (
                group_end < len(rows)
                and self._process_instruction_group_key(rows[group_end]) == group_key
            ):
                group_end += 1

            group_rows = rows[row_index:group_end]
            anchor_state = self._resolve_exact_nc_state_for_process_row(
                group_rows[0].get("line_no_raw"),
                group_rows[0].get("gcode_content"),
            )
            requires_match = any(
                not bool(row.get("_is_synthetic_fill"))
                and bool(str(row.get("gcode_content", "") or "").strip())
                for row in group_rows
            )
            if anchor_state is None and requires_match:
                all_required_groups_matched = False

            if anchor_state:
                try:
                    anchor_start = float(anchor_state.get("path_start", 0.0) or 0.0)
                    anchor_end = float(anchor_state.get("path_end", anchor_start) or anchor_start)
                except (TypeError, ValueError):
                    anchor_start = 0.0
                    anchor_end = 0.0
                    all_required_groups_matched = False
                if not math.isfinite(anchor_start) or not math.isfinite(anchor_end):
                    anchor_start = 0.0
                    anchor_end = 0.0
                    all_required_groups_matched = False
                anchor_tolerance = path_tolerance * max(1.0, abs(anchor_start), abs(anchor_end))
                if anchor_end < anchor_start - anchor_tolerance:
                    all_required_groups_matched = False
                group_length = max(anchor_end - anchor_start, 0.0)
                speed_value = float(anchor_state.get("command_speed", 0.0) or 0.0)
                motion_type = anchor_state.get("motion_type")
                start_coords = (
                    float(anchor_state.get("start_x", anchor_state.get("x", 0.0)) or 0.0),
                    float(anchor_state.get("start_y", anchor_state.get("y", 0.0)) or 0.0),
                    float(anchor_state.get("start_z", anchor_state.get("z", 0.0)) or 0.0),
                )
                end_coords = (
                    float(anchor_state.get("x", start_coords[0]) or start_coords[0]),
                    float(anchor_state.get("y", start_coords[1]) or start_coords[1]),
                    float(anchor_state.get("z", start_coords[2]) or start_coords[2]),
                )
            else:
                group_length = 0.0
                speed_value = 0.0
                motion_type = None
                if row_index > 0:
                    previous = rows[row_index - 1]
                    start_coords = (
                        float(previous.get("x", 0.0) or 0.0),
                        float(previous.get("y", 0.0) or 0.0),
                        float(previous.get("z", 0.0) or 0.0),
                    )
                else:
                    start_coords = (0.0, 0.0, 0.0)
                end_coords = start_coords

            group_count = len(group_rows)
            group_step = group_length / group_count if group_count else 0.0
            for offset, row in enumerate(group_rows):
                row["path_start"] = cumulative_total
                cumulative_total += group_step
                row["path_end"] = cumulative_total
                row["path_cumulative"] = cumulative_total
                row["s"] = group_step
                if speed_value > 0 and not bool(row.get("_has_input_spindle_speed")):
                    row["S"] = speed_value
                if motion_type:
                    row["type"] = motion_type
                ratio = (offset + 1) / group_count if group_count else 1.0
                row["x"] = start_coords[0] + (end_coords[0] - start_coords[0]) * ratio
                row["y"] = start_coords[1] + (end_coords[1] - start_coords[1]) * ratio
                row["z"] = start_coords[2] + (end_coords[2] - start_coords[2]) * ratio
            row_index = group_end

        return all_required_groups_matched and self._path_positions_are_physical(rows)

    def _get_sequential_fallback_step(self):
        """从分割配置读取非物理顺序回退步长，未初始化时使用配置默认值。"""
        step = self._get_segmentation_config_value("sequential_fallback_step_mm", 1.0)
        try:
            step = float(step)
        except (TypeError, ValueError):
            step = 1.0
        return step if math.isfinite(step) and step > 0.0 else 1.0

    def _apply_sequential_fallback_positions(self, rows):
        """所有物理来源无效时生成确定性顺序轴，并明确标记为非物理。"""
        step = self._get_sequential_fallback_step()
        cumulative_total = 0.0
        for row in rows:
            row["s"] = step
            row["path_start"] = cumulative_total
            cumulative_total += step
            row["path_end"] = cumulative_total
            row["path_cumulative"] = cumulative_total

    def _mark_process_path_source(self, rows, source, is_physical, input_validation):
        for row in rows:
            row["path_source"] = source
            row["path_is_physical"] = bool(is_physical)
            row["input_path_valid"] = bool(input_validation.get("valid"))
            row["input_path_validity_reason"] = input_validation.get("reason", "unknown")
        diagnostics = {
            "path_source": source,
            "path_is_physical": bool(is_physical),
            "used_nonphysical_fallback": not bool(is_physical),
            "input_path_present": bool(input_validation.get("present")),
            "input_path_valid": bool(input_validation.get("valid")),
            "input_path_validity_reason": input_validation.get("reason", "unknown"),
            "input_path_span": float(input_validation.get("span", 0.0) or 0.0),
            "input_path_semantics": input_validation.get("semantics", "unknown"),
        }
        diagnostics.update(dict(getattr(self, "process_input_diagnostics", {}) or {}))
        diagnostics.update(dict(getattr(self, "process_line_number_diagnostics", {}) or {}))
        self.process_path_diagnostics = diagnostics
        self.process_path_source = source
        self.process_path_is_physical = bool(is_physical)

    def _apply_nc_profile_to_process_rows(self, rows, origin=None):
        """按输入行程、NC、G 几何、顺序回退的优先级统一建立行程轴。"""
        if not rows:
            return
        if origin is None:
            try:
                origin = (self.origin_x.get(), self.origin_y.get(), self.origin_z.get())
            except Exception:
                origin = (0.0, 0.0, 0.0)

        # 行号补齐完成后才按精确文件行回填缺失转速；不使用旧的最近行匹配。
        for row in rows:
            if bool(row.get("_has_input_spindle_speed")):
                continue
            exact_state = self._resolve_exact_nc_state_for_process_row(
                row.get("line_no_raw"),
                row.get("gcode_content"),
            )
            if not exact_state:
                continue
            try:
                exact_speed = float(exact_state.get("command_speed", 0.0) or 0.0)
            except (TypeError, ValueError):
                exact_speed = 0.0
            if math.isfinite(exact_speed) and exact_speed > 0.0:
                row["S"] = exact_speed

        input_validation = self._validate_input_path(rows)
        if input_validation["valid"]:
            # 有效输入行程绝对优先；这里只解析坐标/模态，不计算 G 几何长度。
            self._apply_gcode_geometry_positions(rows, origin, calculate_path=False)
            path_semantics = str(input_validation.get("semantics") or "unknown")
            self._apply_input_path_positions(rows, path_semantics)
            path_source = f"input_{path_semantics}"
            path_is_physical = True
        else:
            used_nc_profile = bool(getattr(self, "gcode_profile", None)) and self._apply_bound_nc_profile_positions(rows)
            if used_nc_profile:
                path_source = "nc_profile"
                path_is_physical = True
            else:
                self._apply_gcode_geometry_positions(rows, origin, calculate_path=True)
                if self._path_positions_are_physical(rows):
                    path_source = "gcode_geometry"
                    path_is_physical = True
                else:
                    self._apply_sequential_fallback_positions(rows)
                    path_source = "sequential_fallback"
                    path_is_physical = False

        self._mark_process_path_source(rows, path_source, path_is_physical, input_validation)

        s_base = self.s_base.get()
        k_base = self.k_base.get()
        fallback_speed = float(self.current_program_speed.get() or s_base)
        kc_value = self.get_kc_value()
        ke_value = self.get_ke_value()
        idle_power_predict = self._create_idle_power_predictor() if hasattr(self, "_create_idle_power_predictor") else self.predict_idle_power
        for row in rows:
            effective_feed = float(row.get("feed_effective", 0.0) or 0.0)
            try:
                speed_value = float(row.get("S", 0.0) or 0.0)
            except Exception:
                speed_value = 0.0
            effective_speed = speed_value if speed_value > 0.0 else fallback_speed
            idle_power = idle_power_predict(effective_speed)
            t_val, dmrv_val, mrr_val, k_val, t_torque, p_power, p_idle, p_edge = self.calculate_additional_columns(
                row.get("ap", 0.0),
                row.get("ae", 0.0),
                effective_feed,
                row.get("s", 0.0),
                row.get("S", 0.0),
                s_base,
                k_base,
                kc_value=kc_value,
                ke_value=ke_value,
                idle_power=idle_power,
                fallback_speed=fallback_speed,
            )
            row["t"] = t_val
            row["dMRV"] = dmrv_val
            row["MRR"] = mrr_val
            row["K"] = k_val
            row["T"] = t_torque
            row["P"] = p_power
            row["P_idle"] = p_idle
            row["P_edge"] = p_edge
            row["K_c"] = kc_value
            row["K_e"] = ke_value

    def extract_coordinates(self, gcode_content, prev_coords):
        """提取坐标值"""
        # 默认使用上一行的坐标值
        x, y, z = prev_coords
        
        # 使用正则表达式提取坐标值
        if match := re.search(r'X([-+]?\d*\.?\d+)', gcode_content):
            x = float(match.group(1))
        if match := re.search(r'Y([-+]?\d*\.?\d+)', gcode_content):
            y = float(match.group(1))
        if match := re.search(r'Z([-+]?\d*\.?\d+)', gcode_content):
            z = float(match.group(1))
        
        return x, y, z

    def calculate_distance(self, prev_coords, current_coords):
        """计算距离"""
        return self._calculate_linear_distance(prev_coords, current_coords)

    def extract_n_value(self, gcode_content):
        """提取N值（行号标识），保留小数部分
        当G代码中没有N前缀时返回None（而非"N0"），
        避免无N值的行被错误合并到同一对齐行号组中。
        """
        # 修改正则表达式以匹配带小数的N值
        match = re.search(r'^N\d+\.?\d*', gcode_content)
        return match.group(0) if match else None

    def extract_n_decimal_part(self, n_value):
        """提取N值的小数部分（字符串）"""
        if not n_value:
            return ""
        match = re.match(r'^N\d+(?:\.(\d+))?', n_value)
        return match.group(1) if match and match.group(1) else ""

    def extract_n_integer(self, n_value):
        """提取N值整数部分"""
        if n_value is None:
            return None
        match = re.search(r'^N(\d+)', n_value)
        if not match:
            return None
        try:
            return int(match.group(1))
        except Exception:
            return None

    def count_n_integer_occurrences(self, input_file):
        """统计输入文件中各N整数出现次数"""
        counts = {}
        if not input_file:
            return counts
        try:
            process_layout = None
            input_encoding = self.detect_file_encoding(input_file)
            with open(input_file, 'r', encoding=input_encoding, errors='ignore') as infile:
                for line in infile:
                    parsed, process_layout = self.parse_gcode_line(line, layout_hint=process_layout, return_layout=True)
                    if not parsed:
                        continue
                    gcode_content = parsed.get("gcode_content", "")
                    n_value = self.extract_n_value(gcode_content)
                    n_int = self.extract_n_integer(n_value)
                    if n_int is None:
                        continue
                    counts[n_int] = counts.get(n_int, 0) + 1
        except Exception:
            return counts
        return counts

    def _set_widget_state(self, widget, enabled):
        state = "normal" if enabled else "disabled"
        try:
            widget.configure(state=state)
            return
        except Exception:
            pass
        try:
            if enabled:
                widget.state(["!disabled"])
            else:
                widget.state(["disabled"])
        except Exception:
            pass

    def set_sample_controls_enabled(self, enabled, refresh=True):
        """按导入工艺信息表数量切换实测数据联动控件状态"""
        if not hasattr(self, "sample_control_widgets"):
            return
        for widget in self.sample_control_widgets:
            self._set_widget_state(widget, enabled)
        if enabled:
            if refresh:
                self.on_sample_display_mode_change()
            else:
                mode = self.sample_display_mode.get()
                if mode == "program":
                    self.sample_program_combo.configure(state="readonly")
                    self.sample_tool_combo.configure(state="disabled")
                    self.sample_avg_var.set("-")
                    self.sample_ideal_var.set("-")
                else:
                    self.sample_program_combo.configure(state="readonly")
                    self.sample_tool_combo.configure(state="readonly")
                    self.sample_avg_var.set("-")
                    self.sample_ideal_var.set("-")
        else:
            self.sample_avg_var.set("多文件")
            self.sample_ideal_var.set("多文件")
            if hasattr(self, "sample_auto_status_var"):
                self.sample_auto_status_var.set("多文件模式：停用实测自动导入")

    def reset_sample_data_state(self):
        """清空已加载的实测数据状态"""
        self._current_display_power_mean = None
        if hasattr(self, "_invalidate_measurement_runtime_state"):
            self._invalidate_measurement_runtime_state(keep_profile_lock=False)
        self.sample_data_loaded = False
        self.sample_data_dir = None
        self.manual_measurement_path = None
        self.manual_measurement_data = None
        self.sample_data_mode = "sampledata"
        self.sample_source_labels = ["电流", "VGpro功率", "边缘模块功率"]
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
        self.sample_data_valid_mask = None
        self.sample_data_valid_blocks = []
        self.process_valid_mask = None
        self.process_valid_blocks = []
        if hasattr(self, "sample_program_name"):
            self.sample_program_name.set("")
        if hasattr(self, "sample_tool_name"):
            self.sample_tool_name.set("")
        if hasattr(self, "sample_program_combo"):
            self.sample_program_combo["values"] = []
        if hasattr(self, "sample_tool_combo"):
            self.sample_tool_combo["values"] = []
        if hasattr(self, "sample_auto_status_var"):
            self.sample_auto_status_var.set("未导入实测数据")
        if hasattr(self, "refresh_sample_source_labels"):
            self.refresh_sample_source_labels()
        invalidator = getattr(self, "_invalidate_segmentation_sample_projection", None)
        if callable(invalidator):
            invalidator(reason="实际采样文件已移除")
        if hasattr(self, "_refresh_import_order_controls"):
            self._refresh_import_order_controls()

    def ensure_sample_data_matches_inputs(self, file_paths):
        """导入工艺信息表变更时，多目录时重置实测数据"""
        if not self.sample_data_loaded:
            return
        if not file_paths:
            return
        base_dirs = {os.path.normcase(os.path.normpath(os.path.abspath(os.path.dirname(path))))
                     for path in file_paths if path}
        if len(base_dirs) != 1:
            self.reset_sample_data_state()
            return

    def reset_processing_state(self):
        """清理工艺信息表处理与预览状态"""
        self.data = []
        interval_state_clearer = getattr(self, "_clear_current_interval_state", None)
        if callable(interval_state_clearer):
            interval_state_clearer()
        self._latest_segmentation_result = None
        self._current_process_signature = ""
        self._current_mapping_signature = ""
        self._segmentation_sample_projection_records = []
        mapping_status = (
            "pending" if bool(getattr(self, "sample_data_loaded", False)) else "not_available"
        )
        status_setter = getattr(self, "_set_segmentation_mapping_status", None)
        if callable(status_setter):
            status_setter(mapping_status, reason="工艺信息已变更，等待新的过程域划分")
        else:
            self._sample_mapping_status = mapping_status
        cleaner = getattr(self, "_clear_segmentation_output_artifacts", None)
        if callable(cleaner):
            try:
                cleaner()
            except OSError as exc:
                if hasattr(self, "segmentation_status_var"):
                    self.segmentation_status_var.set(f"全行程六类划分: 旧导出清理失败（{exc}）")
        if hasattr(self, "segmentation_status_var"):
            self.segmentation_status_var.set("过程域六类划分: 未运行")
        if hasattr(self, "_clear_runtime_identified_profile_state"):
            self._clear_runtime_identified_profile_state(clear_active=True, reason="reset_processing_state")
        self.processed_file_path = ""
        self.processed_data_dir = None
        self.raw_to_aligned_line_map = {}
        self.process_valid_mask = None
        self.process_valid_blocks = []
        self.figures = []
        self.figure_names = []
        self.current_figure_index = 0
        self._process_point_lookup_cache = None
        self._process_point_lookup_cache_key = None
        self._process_point_metadata_cache_key = None
        self._sample_line_point_context_cache = None
        self._authoritative_segmentation_sample_lookup_cache = None
        self._last_process_application_context = ""
        if self.sample_display_mode.get() == "program":
            self.sample_avg_var.set("-")
            self.sample_ideal_var.set("-")
        else:
            self.sample_avg_var.set("-")
            self.sample_ideal_var.set("-")
        try:
            self.update_nav_buttons()
        except Exception:
            pass
        if hasattr(self, "invalidate_pit_view"):
            self.invalidate_pit_view(refresh_if_visible=True)
        if hasattr(self, "refresh_prediction_metrics_summary"):
            self.refresh_prediction_metrics_summary()
        if hasattr(self, "_refresh_import_order_controls"):
            self._refresh_import_order_controls()

    def build_raw_to_aligned_line_map(self):
        """构建原始行号到重构行号的映射
        
        key: 工艺信息文件的第二列行号（line_no_raw，已在解析阶段做过-1处理）
        value: 按N值整数合并后的重构行号（line_no_aligned）
        用于将SampleData的行号映射到预测负载的重构行号
        """
        mapping = {}
        for item in self.data or []:
            raw = item.get('line_no_raw')
            aligned = item.get('line_no_aligned')
            if raw is None or aligned is None:
                continue
            try:
                # 这里的 raw 已是解析后的 0 基行号，直接作为 key 使用
                mapping[int(raw)] = int(aligned)
            except Exception:
                continue
        self.raw_to_aligned_line_map = mapping
        return mapping

    def align_line_numbers_to_processed(self, line_numbers):
        """将行号转换为对齐行号"""
        if line_numbers is None:
            return None
        mapping = self.raw_to_aligned_line_map or {}
        aligned = []
        for ln in line_numbers:
            try:
                ln_int = int(ln)
            except Exception:
                aligned.append(ln)
                continue
            aligned.append(mapping.get(ln_int, ln_int))
        return np.asarray(aligned, dtype=int)

    def align_sample_program_ranges(self):
        """将SampleData.txt中的区间按对齐行号更新"""
        if not self.sample_programs:
            return
        mapping = self.raw_to_aligned_line_map or {}
        if not mapping:
            return
        raw_keys = np.asarray(sorted(mapping.keys()), dtype=int)
        aligned_vals = np.asarray([mapping[k] for k in raw_keys], dtype=int)
        for program_info in self.sample_programs.values():
            raw_ranges = program_info.get("tool_raw_ranges") or program_info.get("tools", {})
            aligned_tools = {}
            for tool_id, ranges in raw_ranges.items():
                aligned_ranges = []
                for start, end in ranges:
                    try:
                        start_int = int(start)
                        end_int = int(end)
                    except Exception:
                        continue
                    if start_int > end_int:
                        start_int, end_int = end_int, start_int
                    mask = (raw_keys >= start_int) & (raw_keys <= end_int)
                    if not mask.any():
                        continue
                    aligned_start = int(np.min(aligned_vals[mask]))
                    aligned_end = int(np.max(aligned_vals[mask]))
                    aligned_ranges.append((aligned_start, aligned_end))
                aligned_tools[tool_id] = self.merge_intervals(aligned_ranges)
            program_info["tools"] = aligned_tools

    def align_sample_data_to_processed(self):
        """准备实测数据的显示坐标
        
        SampleData的行号保持原样不做任何处理，
        与预测负载通过相同的行号值进行叠加对齐显示。
        """
        if self.sample_data_values is None:
            return
        # 保持原始数据不变
        if self.sample_data_values_raw is None:
            self.sample_data_values_raw = self.sample_data_values
        if self.sample_data_program_numbers_raw is None:
            self.sample_data_program_numbers_raw = self.sample_data_program_numbers
        if self.sample_data_line_numbers_raw is None:
            self.sample_data_line_numbers_raw = self.sample_data_line_numbers
        
        # SampleData行号保持原样，不做映射转换
        self.sample_data_values = np.asarray(self.sample_data_values_raw)
        self.sample_data_program_numbers = np.asarray(self.sample_data_program_numbers_raw)
        self.sample_data_line_numbers = np.asarray(self.sample_data_line_numbers_raw, dtype=int)

        self.sample_data_time_indices = np.arange(len(self.sample_data_line_numbers), dtype=int)
        self.sample_data_base_blocks = self.compute_sequence_blocks(
            self.sample_data_line_numbers,
            self.sample_data_program_numbers
        )
        self.sample_data_x_positions = np.asarray(
            self.compute_line_x_positions(self.sample_data_line_numbers, blocks=self.sample_data_base_blocks),
            dtype=float
        )
        self.sample_data_point_indices = np.asarray(
            self.compute_line_point_indices(self.sample_data_line_numbers, blocks=self.sample_data_base_blocks),
            dtype=int
        )
        self.sample_data_valid_mask = None
        self.sample_data_valid_blocks = []
        self._sample_line_point_context_cache = None
        self._authoritative_segmentation_sample_lookup_cache = None
        if (
            bool(getattr(self, "_current_interval_ready", False))
            and str(getattr(self, "_current_interval_source", "") or "") == "segmentation"
            and (
                bool(str(getattr(self, "_current_mapping_signature", "") or ""))
                or str(getattr(self, "_sample_mapping_status", "") or "") == "valid"
            )
        ):
            invalidator = getattr(self, "_invalidate_segmentation_sample_projection", None)
            if callable(invalidator):
                invalidator(reason="实际采样坐标已重新对齐")

    def detect_file_encoding(self,file_path):
        """使用 Python 内置方法检测文件编码"""
        # 常见编码列表，按优先级排序
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    # 尝试读取文件内容
                    f.read(1024)  # 只读取前1024字节进行测试
                return encoding
            except UnicodeDecodeError:
                continue
        
        # 如果所有编码都失败，返回默认编码
        return 'utf-8'

    def _parse_tagged_csv_tokens(self, line):
        """解析形如 <Tag>,<Value>... 的文本行，保留列顺序。"""
        parts = [part.strip() for part in str(line).strip().split(',')]
        if parts and parts[-1] == "":
            parts = parts[:-1]
        tokens = []
        for part in parts:
            token = part
            if token.startswith('<') and token.endswith('>'):
                token = token[1:-1]
            tokens.append(token.strip())
        return tokens

    def _match_channel_info_key(self, tokens):
        """根据 ChannelInfo 描述识别关注的通道。"""
        if len(tokens) < 6 or tokens[0] != "ChannelInfo":
            return None

        signal_name = str(tokens[3]).strip()
        axis_name = str(tokens[5]).strip()
        extra_code = str(tokens[6]).strip() if len(tokens) > 6 else ""

        if signal_name == "实际速度" and axis_name == "SP轴":
            return "actual_spindle_speed"
        if signal_name == "G寄存器" and axis_name == "X轴" and extra_code == "432":
            return "actual_load"
        if signal_name == "实际速度" and axis_name in {"X轴", "Y轴", "Z轴"}:
            return f"axis_feed_{axis_name[0].lower()}"
        if signal_name == "程序行号" and axis_name == "AX":
            return "program_line"
        return None

    def parse_channel_data_file(self, file_path):
        """解析实验实测文件，提取实际负载、实际转速、合成进给与程序行号。"""
        encoding = self.detect_file_encoding(file_path)
        channel_columns = {}
        channel_data_rows = []

        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            channel_info_index = 0
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("<ChannelInfo>"):
                    tokens = self._parse_tagged_csv_tokens(line)
                    channel_key = self._match_channel_info_key(tokens)
                    if channel_key:
                        channel_columns[channel_key] = channel_info_index
                    channel_info_index += 1
                    continue
                if line.startswith("<ChannelData>"):
                    tokens = self._parse_tagged_csv_tokens(line)
                    if len(tokens) > 1:
                        channel_data_rows.append(tokens[1:])

        required_keys = ("actual_spindle_speed", "actual_load", "program_line")
        missing_keys = [key for key in required_keys if key not in channel_columns]
        feed_axis_keys = [key for key in ("axis_feed_x", "axis_feed_y", "axis_feed_z") if key in channel_columns]
        if not feed_axis_keys:
            missing_keys.append("axis_feed_[x|y|z]")
        if missing_keys:
            missing_text = ", ".join(missing_keys)
            raise ValueError(f"实验实测文件缺少必要通道: {missing_text}")

        max_col_index = max(channel_columns.values())
        actual_load = []
        actual_spindle_speed = []
        actual_feed_speed = []
        program_line = []
        negative_load_corrected_count = 0

        def _safe_float(row, idx):
            if idx is None or idx >= len(row):
                return np.nan
            text = str(row[idx]).strip()
            if not text:
                return np.nan
            try:
                return float(text)
            except Exception:
                return np.nan

        for row in channel_data_rows:
            if len(row) <= max_col_index:
                continue

            line_val = _safe_float(row, channel_columns["program_line"])
            if np.isnan(line_val):
                continue

            load_val = _safe_float(row, channel_columns["actual_load"])
            if np.isfinite(load_val) and load_val < 0:
                load_val = abs(load_val)
                negative_load_corrected_count += 1
            spindle_val = _safe_float(row, channel_columns["actual_spindle_speed"])

            feed_components = []
            for axis_key in ("axis_feed_x", "axis_feed_y", "axis_feed_z"):
                if axis_key not in channel_columns:
                    continue
                component = _safe_float(row, channel_columns[axis_key])
                if not np.isnan(component):
                    feed_components.append(component)
            feed_val = float(math.sqrt(sum(component * component for component in feed_components))) if feed_components else np.nan

            actual_load.append(load_val)
            actual_spindle_speed.append(spindle_val)
            actual_feed_speed.append(feed_val)
            program_line.append(int(line_val))

        if not program_line:
            raise ValueError("未在实验实测文件中解析到有效的程序行号数据")

        return {
            "source_file": file_path,
            "encoding": encoding,
            "sample_count": len(program_line),
            "channel_columns": dict(channel_columns),
            "negative_load_corrected_count": int(negative_load_corrected_count),
            "actual_load": np.asarray(actual_load, dtype=float),
            "actual_spindle_speed": np.asarray(actual_spindle_speed, dtype=float),
            "actual_feed_speed": np.asarray(actual_feed_speed, dtype=float),
            "program_line": np.asarray(program_line, dtype=int),
        }

    def process_single_file(self, input_file):
        """处理单个文件的核心逻辑 - 仅解析数据，不生成后处理文件"""
        try:
            # 获取参数
            origin = (
                self.origin_x.get(),
                self.origin_y.get(),
                self.origin_z.get()
            )
            rapid_speed_xy = self.rapid_speed_xy.get()
            rapid_speed_z = self.rapid_speed_z.get()
            
            # 只读取文件，不写入输出
            input_encoding = self.detect_file_encoding(input_file)
            with open(input_file, 'r', encoding=input_encoding, errors='ignore') as infile:
                current_coords = origin
                data = []
                s_base = self.s_base.get()
                k_base = self.k_base.get()
                current_s = float(self.current_program_speed.get() or s_base)
                fallback_speed = float(current_s if current_s > 0 else s_base)
                current_feed = 0.0
                process_layout = None
                process_path_semantics_hint = None
                kc_value = self.get_kc_value()
                ke_value = self.get_ke_value()
                idle_power_predict = self._create_idle_power_predictor() if hasattr(self, "_create_idle_power_predictor") else self.predict_idle_power
                source_file_line_count = 0
                raw_nonempty_row_count = 0
                header_row_count = 0
                 
                prev_aligned_line = None
                prev_gcode_content = None  # 跟踪上一行的G代码内容（第六列之后）
                current_n_group_line = None  # 当前G代码组的重构行号
                for line_num, line in enumerate(infile):
                    source_file_line_count += 1
                    if str(line).strip():
                        raw_nonempty_row_count += 1
                    raw_csv_line = str(line or "").strip().lstrip("\ufeff")
                    if raw_csv_line and "," in raw_csv_line:
                        try:
                            header_tokens = next(csv.reader([raw_csv_line]))
                        except Exception:
                            header_tokens = []
                        if self._detect_process_header_layout(header_tokens):
                            process_path_semantics_hint = (
                                self._detect_process_path_semantics_hint(header_tokens)
                            )
                    previous_layout = process_layout
                    parsed, process_layout = self.parse_gcode_line(line, layout_hint=process_layout, return_layout=True)
                    if not parsed:
                        if previous_layout is None and process_layout is not None:
                            header_row_count += 1
                        continue
                    
                    ap = float(parsed.get("ap", 0.0) or 0.0)
                    ae = float(parsed.get("ae", 0.0) or 0.0)
                    feed_rate = float(parsed.get("feed_rate", 0.0) or 0.0)
                    gcode_content = str(parsed.get("gcode_content", "") or "")
                    spindle_speed = parsed.get("spindle_speed")
                    input_path_value = parsed.get("path_value")
                    raw_line_number = parsed.get("line_number")

                    # 更新转速
                    if spindle_speed is not None:
                        current_s = float(spindle_speed)

                    # ProcessInfo 的 F 是当前行原始编程进给；显式 0 不能沿用上一行。
                    current_feed = float(feed_rate)
                    
                    n_value = self.extract_n_value(gcode_content)
                    
                    # 根据G代码内容（第六列之后所有内容）决定重构行号
                    # 规则：
                    # 1. 第一行：使用原始第二列值作为起始重构行号
                    # 2. 连续行的G代码内容完全一致 → 重构行号保持不变（同一条G代码指令的细分）
                    # 3. G代码内容变化 → 重构行号 +1（保持连续）
                    
                    if prev_aligned_line is None:
                        # 第一行：使用原始第二列值
                        aligned_line = raw_line_number if raw_line_number is not None else 0
                        current_n_group_line = aligned_line
                        prev_gcode_content = gcode_content
                    else:
                        if gcode_content == prev_gcode_content:
                            # G代码内容相同，重构行号保持不变
                            aligned_line = current_n_group_line
                        else:
                            # G代码内容变化，重构行号 +1
                            aligned_line = current_n_group_line + 1
                            current_n_group_line = aligned_line
                            prev_gcode_content = gcode_content
                    
                    prev_aligned_line = aligned_line
                    
                    effective_speed = current_s if current_s > 0 else fallback_speed
                    p_idle = idle_power_predict(effective_speed)
                    
                    # 收集基础数据，派生功率列在最终回填阶段统一计算，避免重复计算两遍
                    data.append({
                        's': 0.0,
                        't': 0.0,
                        'ap': ap,
                        'ae': ae,
                        'dMRV': 0.0,
                        'MRR': 0.0,
                        'S': current_s,
                        'K': kc_value,
                        'T': 0.0,
                        'P': p_idle,
                        'P_idle': p_idle,
                        'P_edge': 0.0,
                        'K_c': kc_value,
                        'K_e': ke_value,
                        'type': 'rapid',
                        'N_str': n_value,  # 存储N列字符串值
                        'x': float(current_coords[0]),
                        'y': float(current_coords[1]),
                        'z': float(current_coords[2]),
                        'line_no_raw': raw_line_number,
                        'line_no_aligned': aligned_line,
                        'gcode_content': gcode_content,
                        'feed_effective': float(current_feed),
                        '_has_input_path_bounds': False,
                        '_input_path_column_present': bool(parsed.get('path_column_present')),
                        '_input_path_value': input_path_value,
                        '_input_path_semantics_hint': process_path_semantics_hint,
                        '_has_input_spindle_speed': spindle_speed is not None,
                        '_is_synthetic_fill': False,
                    })

                raw_data_row_count = max(raw_nonempty_row_count - header_row_count, 0)
                self.process_input_diagnostics = {
                    "source_file_line_count": int(source_file_line_count),
                    "raw_nonempty_row_count": int(raw_nonempty_row_count),
                    "header_row_count": int(header_row_count),
                    "raw_data_row_count": int(raw_data_row_count),
                    "valid_process_point_count": int(len(data)),
                    "discarded_data_row_count": int(max(raw_data_row_count - len(data), 0)),
                }
                self._restore_missing_line_numbers_from_nc_profile(data)
                self._apply_nc_profile_to_process_rows(data, origin=origin)
                self.data = data
                self._process_point_lookup_cache = None
                self._process_point_lookup_cache_key = None
                self._process_point_metadata_cache_key = None
                self._authoritative_segmentation_sample_lookup_cache = None

            self.build_raw_to_aligned_line_map()
            if self.sample_data_loaded:
                self.align_sample_data_to_processed()
            
            return True
        
        except Exception as e:
            raise Exception(f"处理文件 {input_file} 时出错: {str(e)}")
