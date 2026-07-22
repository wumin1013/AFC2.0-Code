from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .features import build_atomic_segments, compute_point_features, standardize_input
from .schemas import (
    INPUT_SCHEMA_VERSION,
    RULE_SCORER_VERSION,
    AtomicSegment,
    DecodeResult,
    DecodedSegment,
    PathDiagnostics,
    SegmentState,
    SegmentationConfig,
    SegmentationResult,
    state_code,
)
from .scorers import RuleSegmentScorer, SegmentScorer
from .semi_markov import (
    SemiMarkovDecoder,
    count_entry_without_idle_predecessor,
    count_exit_without_idle_successor,
    count_illegal_transitions,
    normalize_steady_segments,
    normalize_transition_segments,
)


POINT_LABEL_COLUMNS: Tuple[str, ...] = (
    "point_id",
    "source_index",
    "s",
    "path_start",
    "path_end",
    "path_source",
    "path_is_physical",
    "line_id",
    "line_no_raw",
    "N_str",
    "line_point_index",
    "point_label",
    "ap",
    "ae",
    "F_program",
    "MRR_program",
    "interval_id",
    "segment_type",
    "state_code",
    "review_required",
)


INTERVAL_COLUMNS: Tuple[str, ...] = (
    "interval_id",
    "start_idx",
    "end_idx",
    "start_point_id",
    "end_point_id",
    "start_source_index",
    "end_source_index",
    "start_s",
    "end_s",
    "start_line_id",
    "end_line_id",
    "start_line_raw",
    "end_line_raw",
    "start_label",
    "end_label",
    "boundary_label",
    "segment_type",
    "state_code",
    "length_mm",
    "point_count",
    "ap_mean",
    "ap_std",
    "ae_mean",
    "ae_std",
    "F_program_mean",
    "F_program_std",
    "MRR_program_mean",
    "MRR_program_std",
    "optimal_rule_score",
    "second_high_score",
    "score_margin",
    "confidence_type",
    "confidence_level",
    "review_required",
    "decision_reason",
    "input_schema_version",
    "scorer_type",
    "model_version",
    "class_confidence",
    "boundary_confidence",
)


def _safe_raw_line(value, fallback: int) -> int:
    try:
        numeric = float(value)
    except Exception:
        numeric = float("nan")
    return int(round(numeric)) if np.isfinite(numeric) else int(fallback)


def _confidence_level(margin: float, config: SegmentationConfig) -> str:
    if margin >= float(config.confidence_high_margin):
        return "high"
    if margin >= float(config.confidence_medium_margin):
        return "medium"
    return "low"


def _frame_signature(frame: pd.DataFrame, config: SegmentationConfig) -> str:
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64).tobytes())
    digest.update(json.dumps(config.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()


def _result_signature(intervals: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    if not intervals.empty:
        columns = ["start_idx", "end_idx", "segment_type", "state_code"]
        digest.update(pd.util.hash_pandas_object(intervals[columns], index=False).to_numpy(dtype=np.uint64).tobytes())
    return digest.hexdigest()


def _decoded_coverage_valid(
    segments: Sequence[DecodedSegment],
    atoms: Sequence[AtomicSegment],
    point_count: int,
) -> bool:
    """复查规范化结果是否连续、无重叠地覆盖全部原子段和过程点。"""

    segments = tuple(segments)
    atoms = tuple(atoms)
    if point_count == 0:
        return not segments and not atoms
    if not segments or not atoms:
        return False
    if segments[0].start_atom != 0 or segments[0].start_idx != 0:
        return False
    if segments[-1].end_atom != len(atoms) - 1:
        return False
    if segments[-1].end_idx != point_count - 1:
        return False

    previous_end_atom = -1
    previous_end_idx = -1
    for segment in segments:
        if segment.start_atom != previous_end_atom + 1:
            return False
        if segment.start_idx != previous_end_idx + 1:
            return False
        if not (0 <= segment.start_atom <= segment.end_atom < len(atoms)):
            return False
        if segment.start_idx != atoms[segment.start_atom].start_idx:
            return False
        if segment.end_idx != atoms[segment.end_atom].end_idx:
            return False
        previous_end_atom = segment.end_atom
        previous_end_idx = segment.end_idx
    return True


class SegmentationPipeline:
    """标准化、特征、原子段、评分、Semi-Markov 解码和全行程展开的唯一入口。"""

    def __init__(self, config: Optional[SegmentationConfig] = None):
        self.config = config or SegmentationConfig()
        self._last_input_signature: Optional[str] = None
        self._last_result_signature: Optional[str] = None

    def _empty_result(self, path: PathDiagnostics, scorer: SegmentScorer) -> SegmentationResult:
        point_labels = pd.DataFrame(columns=POINT_LABEL_COLUMNS)
        intervals = pd.DataFrame(columns=INTERVAL_COLUMNS)
        diagnostics = {
            "input_point_count": 0,
            "atomic_segment_count": 0,
            "decoded_interval_count_before_normalization": 0,
            "final_interval_count": 0,
            "coverage_rate": 1.0,
            "gap_count": 0,
            "overlap_count": 0,
            "illegal_transition_count": 0,
            "postprocess_illegal_transition_count": 0,
            "transition_interval_count_before_normalization": 0,
            "transition_reclassified_count": 0,
            "transition_neighbor_reclassified_count": 0,
            "transition_edge_reclassified_count": 0,
            "transition_length_reclassified_count": 0,
            "transition_edge_violation_count": 0,
            "transition_length_violation_count": 0,
            "transition_point_count_violation_count": 0,
            "transition_point_ratio_violation_count": 0,
            "steady_parent_interval_count": 0,
            "transition_trim_candidate_count": 0,
            "transition_carved_interval_count": 0,
            "transition_carved_atom_count": 0,
            "transition_carved_point_count": 0,
            "transition_carving_skipped_due_to_fallback": 0,
            "transition_generic_mrr_gate_rejection_count": 0,
            "transition_generic_mrr_gate_violation_count": 0,
            "transition_ratio_context_reclassified_count": 0,
            "transition_ratio_boundary_reclassified_count": 0,
            "transition_ratio_core_reclassified_count": 0,
            "transition_ratio_short_transition_reclassified_count": 0,
            "transition_point_atom_granularity_reclassified_count": 0,
            "transition_point_core_short_mm_reclassified_count": 0,
            "transition_point_core_short_point_reclassified_count": 0,
            "transition_point_core_score_reclassified_count": 0,
            "transition_point_target_each_side_total": 0.0,
            "transition_point_actual_each_side_total": 0,
            "transition_ratio_unit": "process_info_point",
            "transition_point_rounding_rule": (
                "nearest_integer_half_up_with_minimum"
            ),
            "transition_ratio_parent": "strict_steady_platform",
            "transition_semantics": "strict_platform_inner_5pct_v1",
            "transition_point_carving_records": [],
            "transition_outside_steady_parent_count": 0,
            "transition_outside_strict_platform_count": 0,
            "provisional_nonsteady_to_transition_count": 0,
            "base_steady_candidate_interval_count": 0,
            "strict_steady_platform_interval_count": 0,
            "strict_platform_refinement_records": [],
            "restored_outer_interval_count": 0,
            "restored_outer_point_count": 0,
            "restored_outer_state_counts": {},
            "restored_outer_invalid_score_count": 0,
            "restored_outer_to_transition_count": 0,
            "strict_platform_partition_violation_count": 0,
            "strict_transition_semantics_valid": True,
            "steady_anchor_candidate_point_count": 0,
            "steady_platform_slope_rejected_point_count": 0,
            "steady_interval_count_before_normalization": 0,
            "steady_reclassified_count": 0,
            "steady_short_reclassified_count": 0,
            "steady_core_recovered_count": 0,
            "steady_core_recovered_point_count": 0,
            "steady_core_candidate_count": 0,
            "steady_without_core_reclassified_count": 0,
            "provisional_steady_recovered_count": 0,
            "provisional_steady_recovered_point_count": 0,
            "provisional_steady_candidate_count": 0,
            "provisional_steady_without_candidate_reclassified_count": 0,
            "steady_normalization_split_count": 0,
            "steady_normalization_merge_count": 0,
            "steady_threshold_violation_count": 0,
            "steady_point_count_violation_count": 0,
            "normalization_merge_count": 0,
            "transition_without_steady_neighbor_count": 0,
            "entry_without_idle_predecessor_count": 0,
            "exit_without_idle_successor_count": 0,
            "entry_without_qualified_idle_predecessor_count": 0,
            "exit_without_qualified_idle_successor_count": 0,
            "qualified_idle_boundary_violation_count": 0,
            "postprocess_coverage_valid": True,
            "idle_gate_mismatch_count": 0,
            "power_gate_invalid_point_count": 0,
            "selected_state_score_invalid_count": 0,
            "postprocess_validation_passed": True,
            "short_steady_count": 0,
            "mrr_formula": "MRR_program = ap * ae * F_program / 60",
            "input_mrr_columns_ignored": True,
            "input_schema_version": self.config.input_schema_version,
            "config": self.config.to_dict(),
            "path": path.to_dict(),
            "fallback_used": False,
            "repeat_run_consistency": {
                "deterministic": True,
                "comparable_to_previous": False,
                "matches_previous": None,
                "input_signature": None,
                "result_signature": _result_signature(intervals),
            },
        }
        return SegmentationResult(
            point_labels=point_labels,
            intervals=intervals,
            diagnostics=diagnostics,
            config=self.config,
            input_schema_version=self.config.input_schema_version,
            scorer_type=str(getattr(scorer, "scorer_type", scorer.__class__.__name__)),
            model_version=getattr(scorer, "model_version", None),
        )

    def _interval_rows(
        self,
        features: pd.DataFrame,
        atoms: Sequence[AtomicSegment],
        decoded: DecodeResult,
        scorer: SegmentScorer,
        path: PathDiagnostics,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        rows: List[Dict] = []
        point_assignment = np.zeros(len(features), dtype=np.int16)
        point_interval = np.full(len(features), "", dtype=object)
        scorer_type = str(getattr(scorer, "scorer_type", scorer.__class__.__name__))
        model_version = getattr(scorer, "model_version", None)

        column_values = {
            name: features[name].to_numpy(copy=False)
            for name in (
                "point_id",
                "source_index",
                "path_start",
                "path_end",
                "line_id",
                "line_no_raw",
                "point_label",
                "input_invalid",
            )
        }
        source_numeric = pd.to_numeric(
            features["source_index"],
            errors="coerce",
        ).to_numpy(dtype=float)
        metric_series = {
            name: features[name]
            for name in ("ap", "ae", "F_program", "MRR_program")
        }

        def _range_mean_std(name: str, start_idx: int, end_idx: int) -> tuple[float, float]:
            values = metric_series[name].iloc[int(start_idx):int(end_idx) + 1]
            return float(values.mean()), float(values.std(ddof=0))

        for number, segment in enumerate(decoded.segments, 1):
            interval_id = f"SEG{number:04d}"
            start_idx = int(segment.start_idx)
            end_idx = int(segment.end_idx)
            start_s = float(column_values["path_start"][start_idx])
            end_s = float(column_values["path_end"][end_idx])
            length_mm = max(end_s - start_s, 0.0)

            all_scores = scorer.score_final_all(
                segment.start_atom,
                segment.end_atom,
            )
            selected = all_scores[segment.state]
            alternatives = [
                (float(value.score), state_code(state))
                for state, value in all_scores.items()
                if state is not segment.state
            ]
            second_high = max(alternatives, key=lambda item: (item[0], -item[1]))[0] if alternatives else float("nan")
            margin = float(selected.score - second_high) if np.isfinite(second_high) else float("inf")
            confidence = _confidence_level(margin, self.config)

            invalid_process_value = bool(
                np.asarray(
                    column_values["input_invalid"][start_idx:end_idx + 1],
                    dtype=bool,
                ).any()
            )
            source_indices = source_numeric[start_idx:end_idx + 1]
            source_index_contiguous = bool(
                np.all(np.isfinite(source_indices))
                and np.allclose(source_indices, np.rint(source_indices), rtol=0.0, atol=0.0)
                and (
                    len(source_indices) <= 1
                    or np.all(np.diff(source_indices) == 1.0)
                )
            )
            review_required = bool(
                confidence == "low"
                or not path.is_physical
                or path.used_nonphysical_fallback
                or decoded.used_fallback
                or invalid_process_value
                or not source_index_contiguous
            )

            reason_parts = [selected.reason]
            if decoded.used_fallback:
                reason_parts.append(decoded.failure_reason)
            if not path.is_physical:
                reason_parts.append("使用非物理顺序行程，持续长度需复核")
            if invalid_process_value:
                reason_parts.append("区间含无效 ap、ae 或程序进给值，计算 MRR 需复核")
            if not source_index_contiguous:
                reason_parts.append("区间源索引不连续，内部可能含被排除的合成占位或无效行")
            if confidence == "low":
                reason_parts.append("规则分数边际低，需复核")

            start_line_id = int(column_values["line_id"][start_idx])
            end_line_id = int(column_values["line_id"][end_idx])
            start_line_raw = _safe_raw_line(
                column_values["line_no_raw"][start_idx],
                start_line_id,
            )
            end_line_raw = _safe_raw_line(
                column_values["line_no_raw"][end_idx],
                end_line_id,
            )
            start_label = str(column_values["point_label"][start_idx])
            end_label = str(column_values["point_label"][end_idx])
            ap_mean, ap_std = _range_mean_std("ap", start_idx, end_idx)
            ae_mean, ae_std = _range_mean_std("ae", start_idx, end_idx)
            feed_mean, feed_std = _range_mean_std("F_program", start_idx, end_idx)
            mrr_mean, mrr_std = _range_mean_std("MRR_program", start_idx, end_idx)

            rows.append(
                {
                    "interval_id": interval_id,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "start_point_id": column_values["point_id"][start_idx],
                    "end_point_id": column_values["point_id"][end_idx],
                    "start_source_index": int(column_values["source_index"][start_idx]),
                    "end_source_index": int(column_values["source_index"][end_idx]),
                    "start_s": start_s,
                    "end_s": end_s,
                    "start_line_id": start_line_id,
                    "end_line_id": end_line_id,
                    "start_line_raw": start_line_raw,
                    "end_line_raw": end_line_raw,
                    "start_label": start_label,
                    "end_label": end_label,
                    "boundary_label": f"{start_label}-{end_label}",
                    "segment_type": segment.state.value,
                    "state_code": state_code(segment.state),
                    "length_mm": length_mm,
                    "point_count": int(end_idx - start_idx + 1),
                    "ap_mean": ap_mean,
                    "ap_std": ap_std,
                    "ae_mean": ae_mean,
                    "ae_std": ae_std,
                    "F_program_mean": feed_mean,
                    "F_program_std": feed_std,
                    "MRR_program_mean": mrr_mean,
                    "MRR_program_std": mrr_std,
                    "optimal_rule_score": float(selected.score),
                    "second_high_score": float(second_high),
                    "score_margin": margin,
                    "confidence_type": "rule_margin",
                    "confidence_level": confidence,
                    "review_required": review_required,
                    "decision_reason": "；".join(reason_parts),
                    "input_schema_version": self.config.input_schema_version,
                    "scorer_type": scorer_type,
                    "model_version": model_version,
                    "class_confidence": None,
                    "boundary_confidence": None,
                }
            )
            point_assignment[start_idx:end_idx + 1] += 1
            point_interval[start_idx:end_idx + 1] = interval_id

        intervals = pd.DataFrame(rows, columns=INTERVAL_COLUMNS)
        return intervals, point_assignment, point_interval

    def run(
        self,
        input_frame,
        *,
        scorer: Optional[SegmentScorer] = None,
    ) -> SegmentationResult:
        selected_scorer = scorer or RuleSegmentScorer(self.config)
        standardized, path_diagnostics = standardize_input(input_frame, self.config)
        if standardized.empty:
            return self._empty_result(path_diagnostics, selected_scorer)

        features = compute_point_features(standardized, self.config)
        atoms = build_atomic_segments(features, self.config)
        selected_scorer.prepare(features, atoms)
        non_idle_column = "is_non_idle" if "is_non_idle" in features else "is_effective_cutting"
        fallback_state = (
            SegmentState.NONSTEADY
            if bool(features[non_idle_column].astype(bool).any())
            else SegmentState.IDLE
        )
        raw_decoded = SemiMarkovDecoder(self.config).decode(
            atoms,
            selected_scorer,
            fallback_state=fallback_state,
        )
        steady_normalized, steady_diagnostics = normalize_steady_segments(
            raw_decoded.segments,
            selected_scorer,
            atoms,
            self.config,
        )
        normalized_segments, transition_diagnostics = normalize_transition_segments(
            steady_normalized,
            selected_scorer,
            atoms,
            self.config,
            # 固定点数裁边是最终 steady 的结构定义，不依赖 Viterbi 是否
            # 使用回退；回退结果也不得留下没有双侧 transition 的 steady。
            allow_carving=True,
        )
        normalization_diagnostics = {
            **steady_diagnostics,
            **transition_diagnostics,
            "decoded_interval_count_before_normalization": int(len(raw_decoded.segments)),
            "normalization_merge_count": int(
                steady_diagnostics["steady_normalization_merge_count"]
                + transition_diagnostics["normalization_merge_count"]
            ),
        }
        normalization_diagnostics.setdefault(
            "base_steady_candidate_interval_count",
            int(sum(
                segment.state is SegmentState.STEADY
                for segment in steady_normalized
            )),
        )
        # v7 将 transition 的唯一来源收紧为严格绿色平台 P。缺少任一
        # 新结构诊断时使用失败哨兵，避免旧后处理被静默当作 v7 结果。
        for diagnostic_name in (
            "strict_steady_platform_interval_count",
            "restored_outer_interval_count",
            "restored_outer_point_count",
            "restored_outer_invalid_score_count",
            "restored_outer_to_transition_count",
            "strict_platform_partition_violation_count",
            "transition_outside_strict_platform_count",
        ):
            normalization_diagnostics.setdefault(diagnostic_name, -1)
        normalization_diagnostics.setdefault(
            "strict_platform_refinement_records",
            [],
        )
        normalization_diagnostics.setdefault("restored_outer_state_counts", {})
        normalization_diagnostics.setdefault(
            "transition_ratio_parent",
            "strict_steady_platform",
        )
        normalization_diagnostics.setdefault(
            "transition_semantics",
            "strict_platform_inner_5pct_v1",
        )
        normalization_diagnostics.setdefault(
            "provisional_steady_recovered_count",
            int(steady_diagnostics.get("steady_core_recovered_count", 0)),
        )
        normalization_diagnostics.setdefault(
            "provisional_steady_recovered_point_count",
            int(steady_diagnostics.get("steady_core_recovered_point_count", 0)),
        )
        normalization_diagnostics.setdefault(
            "provisional_steady_candidate_count",
            int(steady_diagnostics.get("steady_core_candidate_count", 0)),
        )
        normalization_diagnostics.setdefault(
            "provisional_steady_without_candidate_reclassified_count",
            int(steady_diagnostics.get("steady_without_core_reclassified_count", 0)),
        )
        decoded = DecodeResult(
            segments=normalized_segments,
            total_score=raw_decoded.total_score,
            used_fallback=raw_decoded.used_fallback,
            failure_reason=raw_decoded.failure_reason,
        )
        structural_coverage_valid = _decoded_coverage_valid(
            decoded.segments,
            atoms,
            len(features),
        )
        intervals, assignment, point_interval = self._interval_rows(
            features,
            atoms,
            decoded,
            selected_scorer,
            path_diagnostics,
        )

        interval_ids = intervals["interval_id"].to_numpy(dtype=object)
        segment_type_lookup = dict(zip(
            interval_ids,
            intervals["segment_type"].to_numpy(dtype=object),
        ))
        state_code_lookup = dict(zip(
            interval_ids,
            intervals["state_code"].to_numpy(dtype=np.int8),
        ))
        review_lookup = dict(zip(
            interval_ids,
            intervals["review_required"].to_numpy(dtype=bool),
        ))
        segment_types = np.fromiter(
            (str(segment_type_lookup[value]) for value in point_interval),
            dtype=object,
            count=len(point_interval),
        )
        state_codes = np.fromiter(
            (int(state_code_lookup[value]) for value in point_interval),
            dtype=np.int8,
            count=len(point_interval),
        )
        review_required = np.fromiter(
            (bool(review_lookup[value]) for value in point_interval),
            dtype=bool,
            count=len(point_interval),
        )
        point_labels = features.loc[:, [
            "point_id",
            "source_index",
            "s",
            "path_start",
            "path_end",
            "path_source",
            "path_is_physical",
            "line_id",
            "line_no_raw",
            "N_str",
            "line_point_index",
            "point_label",
            "ap",
            "ae",
            "F_program",
            "MRR_program",
        ]].copy()
        point_labels["interval_id"] = point_interval
        point_labels["segment_type"] = segment_types
        point_labels["state_code"] = state_codes
        point_labels["review_required"] = review_required
        point_labels = point_labels.loc[:, POINT_LABEL_COLUMNS]

        input_signature = _frame_signature(standardized, self.config)
        result_signature = _result_signature(intervals)
        comparable = self._last_input_signature == input_signature
        matches_previous = (
            self._last_result_signature == result_signature
            if comparable and self._last_result_signature is not None
            else None
        )
        self._last_input_signature = input_signature
        self._last_result_signature = result_signature

        gap_count = int(np.sum(assignment == 0))
        overlap_count = int(np.sum(assignment > 1))
        covered_count = int(np.sum(assignment == 1))
        final_illegal_transition_count = count_illegal_transitions(decoded.segments)
        idle_gate = (
            features["is_idle_gate"].astype(bool).to_numpy()
            if "is_idle_gate" in features
            else ~features["is_effective_cutting"].astype(bool).to_numpy()
        )
        idle_label = segment_types == SegmentState.IDLE.value
        idle_gate_mismatch_count = int(np.sum(idle_label != idle_gate))
        power_gate_invalid_point_count = (
            int((~features["power_gate_valid"].astype(bool)).sum())
            if "power_gate_valid" in features
            else 0
        )
        postprocess_coverage_valid = bool(
            structural_coverage_valid
            and gap_count == 0
            and overlap_count == 0
            and covered_count == len(features)
        )
        transition_without_steady = int(
            normalization_diagnostics["transition_without_steady_neighbor_count"]
        )
        transition_edge_violation_count = int(
            normalization_diagnostics["transition_edge_violation_count"]
        )
        transition_length_violation_count = int(
            normalization_diagnostics["transition_length_violation_count"]
        )
        transition_point_ratio_violation_count = int(
            normalization_diagnostics.get(
                "transition_point_ratio_violation_count",
                0,
            )
        )
        transition_outside_steady_parent_count = int(
            normalization_diagnostics["transition_outside_steady_parent_count"]
        )
        transition_outside_strict_platform_count = int(
            normalization_diagnostics["transition_outside_strict_platform_count"]
        )
        transition_generic_mrr_gate_violation_count = int(
            normalization_diagnostics[
                "transition_generic_mrr_gate_violation_count"
            ]
        )
        provisional_nonsteady_to_transition_count = int(
            normalization_diagnostics["provisional_nonsteady_to_transition_count"]
        )
        strict_steady_platform_interval_count = int(
            normalization_diagnostics["strict_steady_platform_interval_count"]
        )
        restored_outer_interval_count = int(
            normalization_diagnostics["restored_outer_interval_count"]
        )
        restored_outer_point_count = int(
            normalization_diagnostics["restored_outer_point_count"]
        )
        restored_outer_invalid_score_count = int(
            normalization_diagnostics["restored_outer_invalid_score_count"]
        )
        restored_outer_to_transition_count = int(
            normalization_diagnostics["restored_outer_to_transition_count"]
        )
        strict_platform_partition_violation_count = int(
            normalization_diagnostics[
                "strict_platform_partition_violation_count"
            ]
        )
        strict_transition_semantics_valid = bool(
            str(normalization_diagnostics.get("transition_ratio_parent") or "")
            == "strict_steady_platform"
            and str(normalization_diagnostics.get("transition_semantics") or "")
            == "strict_platform_inner_5pct_v1"
        )
        entry_without_idle_predecessor_count = count_entry_without_idle_predecessor(
            decoded.segments
        )
        exit_without_idle_successor_count = count_exit_without_idle_successor(
            decoded.segments
        )
        entry_boundary_values = (
            features["entry_boundary_candidate"].astype(bool).to_numpy()
            if "entry_boundary_candidate" in features
            else np.zeros(len(features), dtype=bool)
        )
        exit_boundary_values = (
            features["exit_boundary_candidate"].astype(bool).to_numpy()
            if "exit_boundary_candidate" in features
            else np.zeros(len(features), dtype=bool)
        )
        entry_without_qualified_idle_predecessor_count = int(sum(
            segment.state is SegmentState.ENTRY
            and not bool(entry_boundary_values[segment.start_idx])
            for segment in decoded.segments
        ))
        exit_without_qualified_idle_successor_count = int(sum(
            segment.state is SegmentState.EXIT
            and not bool(exit_boundary_values[segment.end_idx])
            for segment in decoded.segments
        ))
        qualified_idle_boundary_violation_count = int(
            entry_without_qualified_idle_predecessor_count
            + exit_without_qualified_idle_successor_count
        )
        selected_state_score_invalid_count = int(sum(
            not np.isfinite(float(selected_scorer.score_final_segment(
                segment.start_atom,
                segment.end_atom,
                segment.state,
            ).score))
            for segment in decoded.segments
        ))
        postprocess_validation_passed = bool(
            postprocess_coverage_valid
            and final_illegal_transition_count == 0
            and transition_without_steady == 0
            and transition_edge_violation_count == 0
            and transition_length_violation_count == 0
            and transition_point_ratio_violation_count == 0
            and transition_generic_mrr_gate_violation_count == 0
            and transition_outside_steady_parent_count == 0
            and transition_outside_strict_platform_count == 0
            and provisional_nonsteady_to_transition_count == 0
            and strict_steady_platform_interval_count >= 0
            and restored_outer_interval_count >= 0
            and restored_outer_point_count >= 0
            and restored_outer_invalid_score_count == 0
            and restored_outer_to_transition_count == 0
            and strict_platform_partition_violation_count == 0
            and strict_transition_semantics_valid
            and entry_without_idle_predecessor_count == 0
            and exit_without_idle_successor_count == 0
            and qualified_idle_boundary_violation_count == 0
            and idle_gate_mismatch_count == 0
            and selected_state_score_invalid_count == 0
        )
        short_steady_count = int(
            (
                (intervals["segment_type"] == SegmentState.STEADY.value)
                & (intervals["length_mm"] < float(self.config.min_steady_mm))
            ).sum()
        ) if not intervals.empty else 0
        steady_threshold_violation_count = 0
        steady_point_count_violation_count = 0
        if not intervals.empty:
            steady_rows = intervals["segment_type"] == SegmentState.STEADY.value
            steady_means = intervals.loc[steady_rows, "MRR_program_mean"].abs()
            steady_relative_std = (
                intervals.loc[steady_rows, "MRR_program_std"]
                / steady_means.clip(lower=max(float(self.config.mrr_cutting_epsilon), 1e-12))
            )
            steady_threshold_violation_count = int(
                (
                    steady_relative_std
                    > float(self.config.steady_mrr_relative_std_max)
                    + float(self.config.tie_epsilon)
                ).sum()
            )
            steady_point_count_violation_count = int(
                (
                    steady_rows
                    & (
                        intervals["point_count"]
                        < int(self.config.steady_min_plateau_points)
                    )
                ).sum()
            )
        postprocess_validation_passed = bool(
            postprocess_validation_passed
            and steady_threshold_violation_count == 0
            and steady_point_count_violation_count == 0
            and short_steady_count == 0
        )
        state_counts = {
            state.value: int((intervals["segment_type"] == state.value).sum())
            for state in SegmentState
        }
        diagnostics = {
            "input_point_count": int(len(features)),
            "atomic_segment_count": int(len(atoms)),
            **normalization_diagnostics,
            "final_interval_count": int(len(intervals)),
            "coverage_rate": float(covered_count / len(features)),
            "gap_count": gap_count,
            "overlap_count": overlap_count,
            "illegal_transition_count": final_illegal_transition_count,
            "postprocess_illegal_transition_count": final_illegal_transition_count,
            "postprocess_coverage_valid": postprocess_coverage_valid,
            "entry_without_idle_predecessor_count": (
                entry_without_idle_predecessor_count
            ),
            "exit_without_idle_successor_count": exit_without_idle_successor_count,
            "entry_without_qualified_idle_predecessor_count": (
                entry_without_qualified_idle_predecessor_count
            ),
            "exit_without_qualified_idle_successor_count": (
                exit_without_qualified_idle_successor_count
            ),
            "qualified_idle_boundary_violation_count": (
                qualified_idle_boundary_violation_count
            ),
            "idle_gate_mismatch_count": idle_gate_mismatch_count,
            "power_gate_invalid_point_count": power_gate_invalid_point_count,
            "selected_state_score_invalid_count": (
                selected_state_score_invalid_count
            ),
            "postprocess_validation_passed": postprocess_validation_passed,
            "strict_transition_semantics_valid": (
                strict_transition_semantics_valid
            ),
            "short_steady_count": short_steady_count,
            "steady_threshold_violation_count": steady_threshold_violation_count,
            "steady_point_count_violation_count": (
                steady_point_count_violation_count
            ),
            "steady_anchor_candidate_point_count": int(
                getattr(
                    selected_scorer,
                    "steady_anchor_candidate_point_count",
                    0,
                )
                or 0
            ),
            "steady_platform_slope_rejected_point_count": int(
                getattr(
                    selected_scorer,
                    "steady_platform_slope_rejected_point_count",
                    0,
                )
                or 0
            ),
            "state_interval_counts": state_counts,
            "mrr_formula": "MRR_program = ap * ae * F_program / 60",
            "input_mrr_columns_ignored": True,
            "input_schema_version": self.config.input_schema_version,
            "config": self.config.to_dict(),
            "path": path_diagnostics.to_dict(),
            "fallback_used": bool(raw_decoded.used_fallback),
            "fallback_reason": raw_decoded.failure_reason,
            "decoder_total_score": float(raw_decoded.total_score),
            "scorer_type": str(getattr(selected_scorer, "scorer_type", selected_scorer.__class__.__name__)),
            "model_version": getattr(selected_scorer, "model_version", None),
            "repeat_run_consistency": {
                "deterministic": True,
                "tie_break_rule": "分数优先，并列时优先更早起点与固定状态顺序",
                "comparable_to_previous": bool(comparable),
                "matches_previous": matches_previous,
                "input_signature": input_signature,
                "result_signature": result_signature,
            },
        }
        return SegmentationResult(
            point_labels=point_labels,
            intervals=intervals,
            diagnostics=diagnostics,
            config=self.config,
            input_schema_version=self.config.input_schema_version,
            scorer_type=str(getattr(selected_scorer, "scorer_type", RULE_SCORER_VERSION)),
            model_version=getattr(selected_scorer, "model_version", None),
            atomic_segments=tuple(atoms),
        )
