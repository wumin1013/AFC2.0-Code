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
PROCESS_LAYOUT_LEGACY_SEQ = "legacy_seq"
PROCESS_LAYOUT_LEGACY_NO_SEQ = "legacy_no_seq"

PROCESS_HEADER_MAP = {
    ("序号", "N", "ap(mm)", "ae(mm)", "F(mm/min)", "MRR(mm3/min)", "s(mm)", "S(r/min)", "G"): PROCESS_LAYOUT_DIRECT_SEQ,
    ("N", "ap(mm)", "ae(mm)", "F(mm/min)", "MRR(mm3/min)", "s(mm)", "S(r/min)", "G"): PROCESS_LAYOUT_DIRECT_NO_SEQ,
    ("序号", "N", "S(r/min)", "ap(mm)", "ae(mm)", "F(mm/min)", "s(mm)", "MRR(mm3/min)", "G"): PROCESS_LAYOUT_EXPORT_SEQ,
    ("N", "S(r/min)", "ap(mm)", "ae(mm)", "F(mm/min)", "s(mm)", "MRR(mm3/min)", "G"): PROCESS_LAYOUT_EXPORT_NO_SEQ,
    ("序号", "N", "ap(mm)", "ae(mm)", "F(mm/min)", "s(mm)", "S(r/min)", "G"): PROCESS_LAYOUT_DIRECT_SEQ_NO_MRR,
    ("N", "ap(mm)", "ae(mm)", "F(mm/min)", "s(mm)", "S(r/min)", "G"): PROCESS_LAYOUT_DIRECT_NO_SEQ_NO_MRR,
    ("序号", "N", "S(r/min)", "ap(mm)", "ae(mm)", "F(mm/min)", "s(mm)", "G"): PROCESS_LAYOUT_EXPORT_SEQ_NO_MRR,
    ("N", "S(r/min)", "ap(mm)", "ae(mm)", "F(mm/min)", "s(mm)", "G"): PROCESS_LAYOUT_EXPORT_NO_SEQ_NO_MRR,
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
        if lowered in {"mrr", "mrr(mm3/min)", "mrr(mm^3/min)", "材料去除率", "materialremovalrate"}:
            return "MRR(mm3/min)"
        if normalized == "S" or "rpm" in lowered or "r/min" in lowered or "转速" in normalized:
            return "S(r/min)"
        if normalized == "s" or lowered in {"s(mm)", "path", "pathlength", "行程", "累计行程", "路径长度"}:
            return "s(mm)"
        if lowered in {"g", "gcode", "nc", "程序段", "代码"}:
            return "G"
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
        return PROCESS_HEADER_MAP.get(normalized)

    def _is_process_header_row(self, tokens):
        """判断是否为工艺信息表头行。"""
        return self._detect_process_header_layout(tokens) is not None

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
        if not gcode_content:
            return None

        line_no_val = self._parse_process_numeric(line_token, "N")
        ap_val = self._parse_process_numeric(ap_token)
        ae_val = self._parse_process_numeric(ae_token)
        feed_val = self._parse_process_numeric(feed_token)
        if line_no_val is None or ap_val is None or ae_val is None or feed_val is None:
            return None
        if ap_val < 0 or ae_val < 0 or feed_val < 0:
            return None

        path_length_val = None
        if s_token is not None:
            path_length_val = self._parse_process_numeric(s_token)
            if path_length_val is None or path_length_val < 0:
                return None

        spindle_val = None
        if spindle_token is not None:
            spindle_val = self._parse_process_numeric(spindle_token, "S")
            if spindle_val is None:
                spindle_val = self._parse_process_numeric(spindle_token)
            if spindle_val is None or spindle_val < 0:
                return None

        mrr_val = None
        if mrr_token is not None:
            mrr_val = self._parse_process_numeric(mrr_token)
            if mrr_val is None:
                return None

        try:
            line_number = int(round(float(line_no_val))) - 1
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
        if path_length_val is not None and path_length_val > 0:
            score += 1.0
        if mrr_val is not None and mrr_val >= 0:
            score += 0.5

        return {
            "layout": layout_name,
            "line_number": line_number,
            "ap": float(ap_val),
            "ae": float(ae_val),
            "feed_rate": float(feed_val),
            "gcode_content": gcode_content,
            "spindle_speed": float(spindle_val) if spindle_val is not None else None,
            "path_cumulative": float(path_length_val) if path_length_val is not None else None,
            "mrr": float(mrr_val) if mrr_val is not None else None,
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
        cleaned = [str(token).strip() for token in tokens if str(token).strip()]
        if len(cleaned) < 5:
            return None, layout_hint

        detected_layout = self._detect_process_header_layout(cleaned)
        if detected_layout:
            return None, detected_layout

        numeric_tokens, gcode_tokens = self._split_numeric_and_gcode_tokens(cleaned)
        if not numeric_tokens or not gcode_tokens or len(numeric_tokens) < 4:
            return None, layout_hint

        result = None
        effective_layout = layout_hint

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

        if result is None:
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

    def _split_numeric_and_gcode_tokens(self, tokens):
        """拆分工艺信息行：数值列与G代码列"""
        gcode_start_idx = None
        for idx, token in enumerate(tokens):
            if re.match(r'^[A-Za-z]', token):
                gcode_start_idx = idx
                break
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

    def compute_gcode_motion_info(self, gcode_content, prev_coords=None, prev_state=None):
        """计算当前行的模态状态、终点坐标和运动长度。"""
        state, word_values = self._resolve_modal_gcode_state(gcode_content, prev_state)
        has_prev_coords = prev_coords is not None
        start_coords = tuple(prev_coords if prev_coords is not None else (0.0, 0.0, 0.0))
        end_coords = self._resolve_gcode_end_coords(start_coords, word_values, state.get("distance_mode", "G90"))
        has_motion_words = any(axis in word_values for axis in ("X", "Y", "Z", "I", "J", "K"))

        if not has_prev_coords or not has_motion_words or state.get("motion") is None:
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

    def _assign_group_distributed_path_positions(self, rows):
        """基于连续相同行号，将累计行程均匀分布到每一行。"""
        cumulative_total = 0.0
        row_index = 0
        while row_index < len(rows):
            group_key = rows[row_index].get("line_no_raw")
            group_end = row_index
            group_total = 0.0
            while group_end < len(rows) and rows[group_end].get("line_no_raw") == group_key:
                try:
                    group_total += float(rows[group_end].get("s", 0.0))
                except Exception:
                    pass
                group_end += 1
            group_count = group_end - row_index
            group_step = group_total / group_count if group_count else 0.0
            for offset in range(row_index, group_end):
                rows[offset]["path_start"] = cumulative_total
                cumulative_total += group_step
                rows[offset]["path_end"] = cumulative_total
                rows[offset]["path_cumulative"] = cumulative_total
            row_index = group_end

    def _apply_input_path_positions(self, rows):
        """直接使用输入文件中的累计行程生成 path_start/path_end。"""
        previous_end = 0.0
        for row in rows:
            if bool(row.get("_has_input_path_bounds")):
                try:
                    start_val = float(row.get("_input_path_start", previous_end) or previous_end)
                except Exception:
                    start_val = previous_end
                try:
                    end_val = float(row.get("_input_path_end", start_val) or start_val)
                except Exception:
                    end_val = start_val
                if end_val < start_val:
                    end_val = start_val
            else:
                start_val = previous_end
                try:
                    segment_length = float(row.get("s", 0.0) or 0.0)
                except Exception:
                    segment_length = 0.0
                if segment_length < 0:
                    segment_length = 0.0
                end_val = start_val + segment_length
            row["path_start"] = start_val
            row["path_end"] = end_val
            row["path_cumulative"] = end_val
            previous_end = end_val

    def _apply_nc_profile_to_process_rows(self, rows):
        """导入 NC 后，按 NC 状态回填每行转速与累计行程。"""
        if not rows:
            return
        has_input_path_bounds = any(bool(row.get("_has_input_path_bounds")) for row in rows)
        if has_input_path_bounds:
            self._apply_input_path_positions(rows)
        elif not getattr(self, "gcode_profile", None):
            self._assign_group_distributed_path_positions(rows)
        else:
            row_index = 0
            while row_index < len(rows):
                group_key = rows[row_index].get("line_no_raw")
                group_end = row_index
                while group_end < len(rows) and rows[group_end].get("line_no_raw") == group_key:
                    group_end += 1

                group_rows = rows[row_index:group_end]
                anchor_state = self._resolve_nc_state_for_process_row(
                    group_rows[0].get("line_no_raw"),
                    group_rows[0].get("gcode_content"),
                )
                if anchor_state:
                    group_start = float(anchor_state.get("path_start", 0.0) or 0.0)
                    group_end_pos = float(anchor_state.get("path_end", group_start) or group_start)
                    group_count = len(group_rows)
                    group_step = (group_end_pos - group_start) / group_count if group_count else 0.0
                    speed_value = float(anchor_state.get("command_speed", 0.0) or 0.0)
                    feed_value = float(anchor_state.get("feed", 0.0) or 0.0)
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
                    for offset, row in enumerate(group_rows):
                        row["path_start"] = group_start + group_step * offset
                        row["path_end"] = group_start + group_step * (offset + 1)
                        row["path_cumulative"] = row["path_end"]
                        row["s"] = group_step
                        if speed_value > 0 and not bool(row.get("_has_input_spindle_speed")):
                            row["S"] = speed_value
                        if (float(row.get("feed_effective", 0.0) or 0.0) <= 0.0) and feed_value > 0:
                            row["feed_effective"] = feed_value
                        if motion_type:
                            row["type"] = motion_type
                        if group_count > 0:
                            interp_ratio = (offset + 1) / group_count
                        else:
                            interp_ratio = 1.0
                        row["x"] = start_coords[0] + (end_coords[0] - start_coords[0]) * interp_ratio
                        row["y"] = start_coords[1] + (end_coords[1] - start_coords[1]) * interp_ratio
                        row["z"] = start_coords[2] + (end_coords[2] - start_coords[2]) * interp_ratio
                else:
                    previous_end = 0.0
                    if row_index > 0:
                        previous_end = float(rows[row_index - 1].get("path_end", 0.0) or 0.0)
                    for row in group_rows:
                        row["s"] = 0.0
                        row["path_start"] = previous_end
                        row["path_end"] = previous_end
                        row["path_cumulative"] = previous_end
                row_index = group_end

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
        self._clear_current_interval_state()
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
        self._sample_line_point_context_cache = None
        self._smif_source_cache = None
        self._smif_dashboard_payload = None
        self._smif_focus_bounds = None
        self._smif_profile_process_rows_cache = None
        self._last_process_application_context = ""
        self._reset_smif_runtime_cache()
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
        self.refresh_pit_button_state()
        self.refresh_smif_view()
        if hasattr(self, "refresh_main_pit_preview"):
            self.refresh_main_pit_preview()
        if hasattr(self, "refresh_prediction_metrics_summary"):
            self.refresh_prediction_metrics_summary()

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
                prev_coords = origin
                current_coords = origin
                modal_gcode_state = self._create_modal_gcode_state()
                data = []
                s_base = self.s_base.get()
                k_base = self.k_base.get()
                current_s = float(self.current_program_speed.get() or s_base)
                fallback_speed = float(current_s if current_s > 0 else s_base)
                current_feed = 0.0
                current_move_type = "rapid"  # 从机床原点开始，初始为快速移动
                process_layout = None
                prev_input_path_cumulative = None
                kc_value = self.get_kc_value()
                ke_value = self.get_ke_value()
                idle_power_predict = self._create_idle_power_predictor() if hasattr(self, "_create_idle_power_predictor") else self.predict_idle_power
                 
                prev_aligned_line = None
                prev_gcode_content = None  # 跟踪上一行的G代码内容（第六列之后）
                current_n_group_line = None  # 当前G代码组的重构行号
                prev_raw_line = None  # 跟踪上一行的原始行号，用于检测行号缺失
                for line_num, line in enumerate(infile):
                    parsed, process_layout = self.parse_gcode_line(line, layout_hint=process_layout, return_layout=True)
                    if not parsed:
                        continue
                    
                    ap = float(parsed.get("ap", 0.0) or 0.0)
                    ae = float(parsed.get("ae", 0.0) or 0.0)
                    feed_rate = float(parsed.get("feed_rate", 0.0) or 0.0)
                    gcode_content = str(parsed.get("gcode_content", "") or "")
                    spindle_speed = parsed.get("spindle_speed")
                    input_path_cumulative = parsed.get("path_cumulative")
                    raw_line_number = parsed.get("line_number")
                    direct_path_length = None
                    input_path_start = None
                    input_path_end = None
                    if input_path_cumulative is not None:
                        try:
                            input_path_end = float(input_path_cumulative)
                        except Exception:
                            input_path_end = None
                        if input_path_end is not None:
                            if prev_input_path_cumulative is None:
                                input_path_start = 0.0
                            else:
                                input_path_start = float(prev_input_path_cumulative)
                            if input_path_end < input_path_start:
                                input_path_end = input_path_start
                            direct_path_length = input_path_end - input_path_start
                            prev_input_path_cumulative = input_path_end
                    else:
                        prev_input_path_cumulative = None
                    nc_state = self._resolve_nc_state_for_process_row(raw_line_number, gcode_content)
                    nc_speed = float(nc_state.get("command_speed", 0.0)) if nc_state else 0.0
                    nc_feed = float(nc_state.get("feed", 0.0)) if nc_state else 0.0
                    
                    # === 行号缺失补齐：检测raw_line_number跳跃，用P=0占位 ===
                    if raw_line_number is not None and prev_raw_line is not None and raw_line_number > prev_raw_line + 1:
                        for missing_raw in range(prev_raw_line + 1, raw_line_number):
                            fill_aligned = current_n_group_line + 1 if current_n_group_line is not None else missing_raw
                            current_n_group_line = fill_aligned
                            prev_aligned_line = fill_aligned
                            prev_gcode_content = None  # 补齐行没有G代码内容
                            fill_speed = current_s if current_s > 0 else fallback_speed
                            fill_idle = idle_power_predict(fill_speed)
                            data.append({
                                's': 0, 't': 0,
                                'ap': 0, 'ae': 0,
                                'dMRV': 0, 'MRR': 0,
                                'S': current_s, 'K': kc_value,
                                'T': 0, 'P': fill_idle,
                                'P_idle': fill_idle,
                                'P_edge': 0.0,
                                'K_c': kc_value,
                                'K_e': ke_value,
                                'type': 'rapid',
                                'N_str': None,
                                'x': float(current_coords[0]),
                                'y': float(current_coords[1]),
                                'z': float(current_coords[2]),
                                'line_no_raw': missing_raw,
                                'line_no_aligned': fill_aligned
                            })
                    
                    # 更新转速
                    if spindle_speed is not None:
                        current_s = float(spindle_speed)
                    elif nc_speed > 0:
                        current_s = nc_speed
                    
                    # 更新进给速度
                    if feed_rate > 0:
                        current_feed = feed_rate
                    elif nc_feed > 0:
                        current_feed = nc_feed
                    
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
                    # 更新上一行的原始行号
                    if raw_line_number is not None:
                        prev_raw_line = raw_line_number
                    
                    motion_info = self.compute_gcode_motion_info(
                        gcode_content,
                        prev_coords=prev_coords,
                        prev_state=modal_gcode_state
                    )
                    modal_gcode_state = motion_info["state"]
                    current_coords = motion_info["end_coords"]
                    computed_path_length = motion_info["segment_length"]
                    if direct_path_length is not None:
                        s = float(direct_path_length)
                    else:
                        s = float(computed_path_length)
                    if motion_info["motion_type"] is not None:
                        current_move_type = motion_info["motion_type"]
                    effective_speed = current_s if current_s > 0 else fallback_speed
                    p_idle = idle_power_predict(effective_speed)
                    
                    # 收集基础数据，派生功率列在最终回填阶段统一计算，避免重复计算两遍
                    data.append({
                        's': s,
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
                        'type': current_move_type,
                        'N_str': n_value,  # 存储N列字符串值
                        'x': float(current_coords[0]),
                        'y': float(current_coords[1]),
                        'z': float(current_coords[2]),
                        'line_no_raw': raw_line_number,
                        'line_no_aligned': aligned_line,
                        'gcode_content': gcode_content,
                        'feed_effective': float(current_feed),
                        '_has_input_path_bounds': input_path_start is not None and input_path_end is not None,
                        '_input_path_start': input_path_start,
                        '_input_path_end': input_path_end,
                        '_has_input_spindle_speed': spindle_speed is not None,
                    })
                    
                    # 更新上一行坐标
                    prev_coords = current_coords
                
                self._apply_nc_profile_to_process_rows(data)
                self.data = data
                self._process_point_lookup_cache = None
                self._process_point_lookup_cache_key = None

            self.build_raw_to_aligned_line_map()
            if self.sample_data_loaded:
                self.align_sample_data_to_processed()
            
            return True
        
        except Exception as e:
            raise Exception(f"处理文件 {input_file} 时出错: {str(e)}")
