from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .schemas import (
    RULE_SCORER_VERSION,
    AtomicSegment,
    SegmentScore,
    SegmentState,
    SegmentationConfig,
)


class SegmentScorer(ABC):
    """区间评分器稳定接口；后续模型只需替换该层。"""

    scorer_type = "abstract"
    model_version = None

    @abstractmethod
    def prepare(self, point_features: pd.DataFrame, atoms: Sequence[AtomicSegment]) -> None:
        """为当前输入准备可复用的区间统计。"""

    @abstractmethod
    def score_segment(self, start_atom: int, end_atom: int, state: SegmentState) -> SegmentScore:
        """返回包含首尾原子段的指定状态分数。"""

    def score_all(self, start_atom: int, end_atom: int) -> Dict[SegmentState, SegmentScore]:
        return {
            state: self.score_segment(start_atom, end_atom, state)
            for state in SegmentState
        }

    def score_final_segment(
        self,
        start_atom: int,
        end_atom: int,
        state: SegmentState,
    ) -> SegmentScore:
        """评价后处理后的最终区间，并支持固定比例 transition。"""

        if state is SegmentState.TRANSITION:
            return getattr(self, "_registered_transition_trims", {}).get(
                (int(start_atom), int(end_atom)),
                SegmentScore(
                    score=float("-inf"),
                    reason="该 transition 未由固定点数裁边登记",
                ),
            )
        return self.score_segment(start_atom, end_atom, state)

    def score_final_all(
        self,
        start_atom: int,
        end_atom: int,
    ) -> Dict[SegmentState, SegmentScore]:
        return {
            state: self.score_final_segment(start_atom, end_atom, state)
            for state in SegmentState
        }

    def score_values(self, start_atom: int, end_atom: int) -> Dict[SegmentState, float]:
        """为解码器提供无文本构造开销的分数快速入口。"""

        return {
            state: float(result.score)
            for state, result in self.score_all(start_atom, end_atom).items()
        }

    def score_candidates(
        self,
        start_atoms: Sequence[int],
        end_atom: int,
    ) -> Dict[SegmentState, np.ndarray]:
        """批量评分同一终点的候选起点；默认实现保持自定义评分器兼容。"""

        rows = [self.score_values(int(start), int(end_atom)) for start in start_atoms]
        return {
            state: np.asarray([row[state] for row in rows], dtype=float)
            for state in SegmentState
        }

    def score_steady_continuation_candidates(
        self,
        start_atoms: Sequence[int],
        end_atom: int,
    ) -> np.ndarray:
        """评价只能直接续接已有 steady 的同态计算块。"""

        return np.asarray(
            self.score_candidates(start_atoms, end_atom)[SegmentState.STEADY],
            dtype=float,
        )

    def structural_candidate_starts(
        self,
        end_atom: int,
        state: SegmentState,
    ) -> np.ndarray:
        """返回由业务边界直接确定、不能被普通窗口剪掉的候选起点。"""

        return np.asarray([], dtype=np.int32)

    def structural_candidates_are_complete(self, state: SegmentState) -> bool:
        """说明结构候选是否已完整覆盖该状态，供解码器安全剪枝。"""

        return False

    def register_transition_trims(
        self,
        trims: Mapping[Tuple[int, int], SegmentScore],
    ) -> None:
        """登记已证明位于严格 steady 平台 P 内的精确最终裁边。"""

        self._registered_transition_trims = {
            (int(start_atom), int(end_atom)): score
            for (start_atom, end_atom), score in trims.items()
            if np.isfinite(float(score.score))
        }


def _prefix(values: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(values, dtype=float)))


class RuleSegmentScorer(SegmentScorer):
    """六态只按工艺字段与重算 MRR 评分。"""

    scorer_type = RULE_SCORER_VERSION
    model_version = None
    _hard_fraction_tolerance = 1e-12

    def __init__(self, config: SegmentationConfig):
        self.config = config
        self._prepared = False
        self._atoms: Sequence[AtomicSegment] = tuple()
        self._frame = pd.DataFrame()
        self._prefixes: Dict[str, np.ndarray] = {}
        self._square_prefixes: Dict[str, np.ndarray] = {}
        self._mrr_values = np.asarray([], dtype=float)
        self._non_idle_values = np.asarray([], dtype=bool)
        self._steady_candidate_values = np.asarray([], dtype=bool)
        self._steady_anchor_values = np.asarray([], dtype=bool)
        self._machining_segment_ids = np.asarray([], dtype=np.int32)
        self._machining_segment_end_indices = np.asarray([], dtype=np.int32)
        self._machining_segment_last_peak_indices = np.asarray([], dtype=np.int32)
        self._machining_entry_start_indices = np.asarray([], dtype=np.int32)
        self._machining_entry_peak_indices = np.asarray([], dtype=np.int32)
        self._machining_entry_peak_methods = np.asarray([], dtype=object)
        self._selected_entry_end_indices = np.asarray([], dtype=np.int32)
        self._entry_required_values = np.asarray([], dtype=bool)
        self._entry_boundary_values = np.asarray([], dtype=bool)
        self._exit_boundary_values = np.asarray([], dtype=bool)
        self._atom_start_indices = np.asarray([], dtype=np.int32)
        self._atom_end_indices = np.asarray([], dtype=np.int32)
        self._atom_start_s = np.asarray([], dtype=float)
        self._atom_end_s = np.asarray([], dtype=float)
        self._single_atom_score_values: Dict[SegmentState, np.ndarray] = {}
        self._single_atom_continuation_values = np.asarray([], dtype=float)
        self._path_center_values = np.asarray([], dtype=float)
        self._transition_trim_scores: Dict[Tuple[int, int], SegmentScore] = {}
        self._steady_structural_starts_by_end: Dict[int, np.ndarray] = {}
        self.steady_anchor_candidate_point_count = 0
        self.steady_anchor_candidate_interval_count = 0
        self.steady_candidate_batch_count = 0
        self.steady_anchor_run_records = []
        self.steady_anchor_local_fallback_records = []
        self.entry_precedence_anchor_excluded_point_count = 0
        self.steady_precedence_entry_clipped_point_count = 0
        self.entry_boundary_records = []
        self.steady_platform_slope_rejected_point_count = 0

    def prepare(self, point_features: pd.DataFrame, atoms: Sequence[AtomicSegment]) -> None:
        self._frame = point_features.reset_index(drop=True)
        self._atoms = tuple(atoms)
        self._prefixes = {}
        self._square_prefixes = {}
        for name in (
            "MRR_program",
            "MRR_program_local_trend",
            "machining_segment_relative_position",
        ):
            values = self._frame[name].to_numpy(dtype=float)
            self._prefixes[name] = _prefix(values)
            self._square_prefixes[name] = _prefix(values * values)
        for name in (
            "is_effective_cutting",
            "is_idle_gate",
            "is_non_idle",
            "steady_point_candidate",
            "entry_required",
            "entry_phase_eligible",
            "exit_phase_eligible",
        ):
            values = self._frame[name].to_numpy(dtype=bool)
            self._prefixes[name] = _prefix(values.astype(float))
        self._mrr_values = self._frame["MRR_program"].to_numpy(dtype=float)
        self._path_center_values = (
            self._frame["path_start"].to_numpy(dtype=float)
            + self._frame["path_end"].to_numpy(dtype=float)
        ) / 2.0
        self._prefixes["path_center"] = _prefix(self._path_center_values)
        self._square_prefixes["path_center"] = _prefix(
            self._path_center_values * self._path_center_values
        )
        self._prefixes["path_mrr_product"] = _prefix(
            self._path_center_values * self._mrr_values
        )
        self._non_idle_values = self._frame["is_non_idle"].to_numpy(dtype=bool)
        self._steady_candidate_values = self._frame[
            "steady_point_candidate"
        ].to_numpy(dtype=bool)
        local_relative_std = self._frame[
            "MRR_program_local_relative_std"
        ].to_numpy(dtype=float)
        local_relative_slope = self._frame[
            "MRR_program_local_relative_slope"
        ].to_numpy(dtype=float)
        locally_low_variation = (
            self._non_idle_values
            & self._frame["is_effective_cutting"].to_numpy(dtype=bool)
            & (
                local_relative_std
                <= float(self.config.steady_mrr_relative_std_max)
            )
        )
        self.steady_platform_slope_rejected_point_count = int(np.sum(
            locally_low_variation
            & (
                local_relative_slope
                > float(self.config.steady_mrr_relative_slope_max)
            )
        ))
        self._steady_anchor_values = np.zeros(len(self._frame), dtype=bool)
        self._machining_segment_ids = self._frame["machining_segment_id"].to_numpy(dtype=np.int32)
        self._machining_segment_end_indices = self._frame[
            "machining_segment_end_idx"
        ].to_numpy(dtype=np.int32)
        self._machining_segment_last_peak_indices = self._frame[
            "machining_segment_last_peak_idx"
        ].to_numpy(dtype=np.int32)
        self._machining_entry_start_indices = self._frame[
            "machining_entry_start_idx"
        ].to_numpy(dtype=np.int32)
        self._machining_entry_peak_indices = self._frame[
            "machining_entry_peak_idx"
        ].to_numpy(dtype=np.int32)
        self._machining_entry_peak_methods = self._frame[
            "machining_entry_peak_method"
        ].fillna("").astype(str).to_numpy(dtype=object)
        self._selected_entry_end_indices = self._frame[
            "selected_entry_end_idx"
        ].to_numpy(dtype=np.int32)
        self._entry_required_values = self._frame["entry_required"].to_numpy(
            dtype=bool
        )
        if "entry_boundary_candidate" in self._frame:
            self._entry_boundary_values = self._frame[
                "entry_boundary_candidate"
            ].to_numpy(dtype=bool)
        else:
            # 有效 idle reset 元数据缺失时 fail closed：普通短 idle 不得
            # 被猜作阶段重置边界，也就不能重新触发 entry。
            self._entry_boundary_values = np.zeros(len(self._frame), dtype=bool)
        if "exit_boundary_candidate" in self._frame:
            self._exit_boundary_values = self._frame[
                "exit_boundary_candidate"
            ].to_numpy(dtype=bool)
        else:
            self._exit_boundary_values = np.zeros(len(self._frame), dtype=bool)
        self._atom_start_indices = np.fromiter(
            (atom.start_idx for atom in self._atoms),
            dtype=np.int32,
            count=len(self._atoms),
        )
        self._atom_end_indices = np.fromiter(
            (atom.end_idx for atom in self._atoms),
            dtype=np.int32,
            count=len(self._atoms),
        )
        self._atom_start_s = np.fromiter(
            (atom.start_s for atom in self._atoms),
            dtype=float,
            count=len(self._atoms),
        )
        self._atom_end_s = np.fromiter(
            (atom.end_s for atom in self._atoms),
            dtype=float,
            count=len(self._atoms),
        )
        self._prepared = True
        self._transition_trim_scores = {}
        self._summary.cache_clear()
        self._steady_anchor_values = self._build_steady_anchor_values()
        self.steady_anchor_candidate_point_count = int(
            np.sum(self._steady_anchor_values)
        )
        self._prefixes["steady_anchor_candidate"] = _prefix(
            self._steady_anchor_values.astype(float)
        )
        # 仅保存到评分器内部帧，避免把“可形成合法 steady 候选”误当成
        # 无上下文的普通点特征。
        self._frame["steady_anchor_candidate"] = self._steady_anchor_values
        self._rebuild_phase_eligibility_from_steady_anchors()
        self._sync_phase_features(point_features)
        self._build_single_atom_score_cache()
        self._summary.cache_clear()

    def _sync_phase_features(self, target: pd.DataFrame) -> None:
        """将评分器中需要稳态上下文的特征回写到 Pipeline。"""

        for name in (
            "steady_anchor_candidate",
            "machining_first_steady_idx",
            "machining_last_steady_idx",
            "selected_entry_end_idx",
            "machining_selected_entry_end_idx",
            "entry_required",
            "entry_phase_eligible",
            "exit_phase_eligible",
        ):
            target[name] = self._frame[name].to_numpy(copy=True)

    def _build_steady_anchor_values(self) -> np.ndarray:
        """标记确实能参与合法稳态候选的点，而非仅局部看似稳定的点。"""

        anchor_delta = np.zeros(len(self._frame) + 1, dtype=np.int32)
        if not self._atoms:
            return anchor_delta[:-1].astype(bool)
        minimum, maximum = self.config.duration_bounds(SegmentState.STEADY)
        tolerance = float(self.config.path_tolerance_mm)
        batch_size = int(self.config.max_segment_atoms)
        hard_tolerance = self._hard_fraction_tolerance
        mrr_epsilon = float(self.config.mrr_cutting_epsilon)
        variation_limit = float(self.config.steady_mrr_relative_std_max)
        slope_limit = float(self.config.steady_mrr_relative_slope_max)
        minimum_point_count = int(self.config.steady_min_plateau_points)
        variation_scale = max(variation_limit, 1e-12)
        mrr_prefix = self._prefixes["MRR_program"]
        mrr_square_prefix = self._square_prefixes["MRR_program"]
        non_idle_prefix = self._prefixes["is_non_idle"]
        steady_prefix = self._prefixes["steady_point_candidate"]
        cutting_prefix = self._prefixes["is_effective_cutting"]
        trend_prefix = self._prefixes["MRR_program_local_trend"]
        cfg = self.config
        atom_run_starts = np.zeros(len(self._atoms), dtype=np.int32)
        previous_run_start = 0
        previous_segment_id = 0
        for atom_index, atom in enumerate(self._atoms):
            start_idx = int(atom.start_idx)
            end_idx = int(atom.end_idx)
            point_count = max(end_idx - start_idx + 1, 1)
            segment_id = int(self._machining_segment_ids[start_idx])
            locally_eligible = bool(
                segment_id > 0
                and (
                    non_idle_prefix[end_idx + 1] - non_idle_prefix[start_idx]
                    >= point_count - hard_tolerance
                )
                and (
                    steady_prefix[end_idx + 1] - steady_prefix[start_idx]
                    >= point_count - hard_tolerance
                )
            )
            if not locally_eligible:
                atom_run_starts[atom_index] = atom_index + 1
                previous_run_start = atom_index + 1
                previous_segment_id = 0
                continue
            if atom_index == 0 or previous_segment_id != segment_id:
                previous_run_start = atom_index
            atom_run_starts[atom_index] = previous_run_start
            previous_segment_id = segment_id

        candidate_interval_count = 0
        candidate_batch_count = 0
        self._steady_structural_starts_by_end = {}
        for end_atom in range(len(self._atoms)):
            run_start = int(atom_run_starts[end_atom])
            if run_start > end_atom:
                continue
            starts = np.arange(run_start, end_atom + 1, dtype=np.int32)
            lengths = np.maximum(
                self._atom_end_s[end_atom] - self._atom_start_s[starts],
                0.0,
            )
            start_indices = self._atom_start_indices[starts]
            end_idx = int(self._atom_end_indices[end_atom])
            counts = (end_idx - start_indices + 1).astype(float)
            valid_length = (
                (lengths >= minimum - tolerance)
                & (lengths <= maximum + tolerance)
                & (counts >= minimum_point_count)
            )
            if not np.any(valid_length):
                continue
            starts = starts[valid_length]
            lengths = lengths[valid_length]
            earliest_structural_start: int | None = None
            for batch_start in range(0, len(starts), batch_size):
                batch_end = min(batch_start + batch_size, len(starts))
                batch_starts = starts[batch_start:batch_end]
                batch_lengths = lengths[batch_start:batch_end]
                start_indices = self._atom_start_indices[batch_starts]
                counts = (end_idx - start_indices + 1).astype(float)
                candidate_batch_count += 1

                def means(prefix: np.ndarray) -> np.ndarray:
                    return (prefix[end_idx + 1] - prefix[start_indices]) / counts

                mean_mrr = means(mrr_prefix)
                mean_mrr_square = means(mrr_square_prefix)
                std_mrr = np.sqrt(
                    np.maximum(mean_mrr_square - mean_mrr * mean_mrr, 0.0)
                )
                variation = std_mrr / np.maximum(
                    np.abs(mean_mrr),
                    max(mrr_epsilon, 1e-12),
                )
                relative_slope = self._relative_path_slopes(
                    start_indices,
                    end_idx,
                    batch_lengths,
                    mean_mrr,
                )
                non_idle_fraction = means(non_idle_prefix)
                steady_point_fraction = means(steady_prefix)
                start_segment_ids = self._machining_segment_ids[start_indices]
                end_segment_id = int(self._machining_segment_ids[end_idx])
                steady_eligible = (
                    (start_segment_ids > 0)
                    & (start_segment_ids == end_segment_id)
                    & (non_idle_fraction >= 1.0 - hard_tolerance)
                    & (steady_point_fraction >= 1.0 - hard_tolerance)
                    & (mean_mrr > mrr_epsilon)
                    & (variation <= variation_limit)
                    & (relative_slope <= slope_limit)
                    & (counts >= minimum_point_count)
                )
                if not np.any(steady_eligible):
                    continue

                start_mrr = self._mrr_values[start_indices]
                end_mrr = float(self._mrr_values[end_idx])
                mrr_scale = np.maximum(
                    np.maximum(np.abs(start_mrr), abs(end_mrr)),
                    np.maximum(np.abs(mean_mrr), 1e-9),
                )
                trend_relative = np.clip(
                    means(trend_prefix) / mrr_scale,
                    -2.0,
                    2.0,
                )
                platform_trend = np.maximum(np.abs(trend_relative), relative_slope)
                stability = np.maximum(1.0 - variation / variation_scale, 0.0)
                steady_score = (
                    cfg.cutting_match_weight * means(cutting_prefix)
                    + cfg.stability_weight * (1.0 + stability)
                    - cfg.variation_weight * np.minimum(variation, 2.0)
                    - cfg.trend_weight * platform_trend
                )
                eligible_mask = steady_eligible & np.isfinite(steady_score)
                eligible_starts = start_indices[eligible_mask]
                if eligible_starts.size:
                    # 解码器对同一局部平台最终只消费最早的
                    # 合法原子起点，无需把所有候选再转成 Python
                    # 整数存入列表。合法点起点仍全部计入锚点
                    # 覆盖和诊断，只精简最小值的保存方式。
                    batch_earliest = int(batch_starts[eligible_mask][0])
                    if (
                        earliest_structural_start is None
                        or batch_earliest < earliest_structural_start
                    ):
                        earliest_structural_start = batch_earliest
                    candidate_interval_count += int(eligible_starts.size)
                    # 同一批的原子起点严格唯一，可以直接索引
                    # 累加，不需要 np.add.at 的重复索引语义。
                    anchor_delta[eligible_starts] += 1
                    anchor_delta[end_idx + 1] -= int(eligible_starts.size)
            if earliest_structural_start is not None:
                # 同一局部平台只需保留最早的合法起点；其后的点由单原子
                # steady 续块覆盖。该确定性剪枝与计算批量大小无关。
                self._steady_structural_starts_by_end[int(end_atom)] = np.asarray(
                    [earliest_structural_start],
                    dtype=np.int32,
                )

        anchors = np.cumsum(anchor_delta[:-1]) > 0
        self.steady_anchor_candidate_interval_count = int(candidate_interval_count)
        self.steady_candidate_batch_count = int(candidate_batch_count)
        # 稳态候选的构建与 provisional entry 完全无关，因此不再存在
        # “进刀先占导致稳态锚点被排除”的点。
        self.entry_precedence_anchor_excluded_point_count = 0
        run_starts = np.flatnonzero(
            anchors & np.concatenate(([True], ~anchors[:-1]))
        )
        run_ends = np.flatnonzero(
            anchors & np.concatenate((~anchors[1:], [True]))
        )
        self.steady_anchor_run_records = [
            {
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "point_count": int(end_idx - start_idx + 1),
                "machining_segment_id": int(
                    self._machining_segment_ids[int(start_idx)]
                ),
                "coverage_reachable": True,
                "first_unreachable_idx": None,
            }
            for start_idx, end_idx in zip(run_starts, run_ends)
        ]
        self.steady_anchor_local_fallback_records = []
        return anchors

    def _rebuild_phase_eligibility_from_steady_anchors(self) -> None:
        """在稳态父平台已独立确定后，生成最终进刀边界。

        对每个由合格 idle 开启的加工阶段，比较首个局部峰值 ``p``
        与首个合法稳态起点 ``s``。峰值先到时峰值点归 entry；
        稳态先到或两者重合时，entry 只到 ``s - 1``。若两者都不
        存在，才使用特征层记录的阶段首次最大 MRR 点。
        """

        count = len(self._frame)
        entry_eligible = np.zeros(count, dtype=bool)
        entry_required = np.zeros(count, dtype=bool)
        selected_entry_end_idx = np.full(count, -1, dtype=np.int32)
        exit_eligible = np.zeros(count, dtype=bool)
        first_steady_idx = np.full(count, -1, dtype=np.int32)
        last_steady_idx = np.full(count, -1, dtype=np.int32)
        records = []
        clipped_point_count = 0
        if count:
            segment_ids = self._machining_segment_ids
            starts = np.flatnonzero(
                (segment_ids > 0)
                & np.concatenate(([True], segment_ids[1:] != segment_ids[:-1]))
            )
            for segment_start in starts:
                segment_start = int(segment_start)
                segment_id = int(segment_ids[segment_start])
                segment_end = int(
                    self._machining_segment_end_indices[segment_start]
                )
                sample = slice(segment_start, segment_end + 1)
                anchor_offsets = np.flatnonzero(
                    self._steady_anchor_values[sample]
                )
                steady_start = None
                steady_end = None
                last_peak_idx = int(
                    self._machining_segment_last_peak_indices[segment_start]
                )
                if anchor_offsets.size:
                    steady_start = segment_start + int(anchor_offsets[0])
                    steady_end = segment_start + int(anchor_offsets[-1])
                    first_steady_idx[sample] = steady_start
                    last_steady_idx[sample] = steady_end
                    exit_start = max(last_peak_idx, steady_end + 1)
                else:
                    exit_start = last_peak_idx
                if exit_start <= segment_end:
                    exit_eligible[exit_start:segment_end + 1] = True

                entry_start = int(
                    self._machining_entry_start_indices[segment_start]
                )
                peak_candidate = int(
                    self._machining_entry_peak_indices[segment_start]
                )
                peak_method = str(
                    self._machining_entry_peak_methods[segment_start]
                )
                local_peak = (
                    peak_candidate
                    if peak_method == "local_turning_point"
                    else None
                )
                fallback_peak = (
                    peak_candidate
                    if peak_method == "stage_first_max_fallback"
                    else None
                )
                selected_end = None
                decision = "no_entry_boundary"
                if (
                    entry_start >= 0
                    and bool(self._entry_boundary_values[entry_start])
                ):
                    if local_peak is not None and steady_start is not None:
                        if local_peak < steady_start:
                            selected_end = local_peak
                            decision = "local_peak_before_steady"
                        elif steady_start < local_peak:
                            selected_end = steady_start - 1
                            decision = "steady_before_local_peak"
                        else:
                            selected_end = steady_start - 1
                            decision = "local_peak_equals_steady_start"
                    elif local_peak is not None:
                        selected_end = local_peak
                        decision = "local_peak_without_steady"
                    elif steady_start is not None:
                        selected_end = steady_start - 1
                        decision = "steady_without_local_peak"
                    elif fallback_peak is not None:
                        selected_end = fallback_peak
                        decision = "stage_first_max_fallback"
                    else:
                        decision = "no_positive_mrr_boundary"

                provisional_end = peak_candidate if peak_candidate >= 0 else None
                if (
                    provisional_end is not None
                    and selected_end is not None
                    and selected_end < provisional_end
                ):
                    clipped_point_count += provisional_end - selected_end

                entry_is_continuous = bool(
                    selected_end is not None
                    and selected_end >= entry_start
                    and np.all(
                        self._non_idle_values[entry_start:selected_end + 1]
                    )
                )
                if entry_is_continuous:
                    entry_slice = slice(entry_start, selected_end + 1)
                    entry_eligible[entry_slice] = True
                    entry_required[entry_slice] = True
                elif selected_end is not None and selected_end >= entry_start:
                    # 进刀不能跨过中间 idle 硬标签。不猜测新边界，
                    # 保留已确定的边界供诊断，并将该段交给其他合法状态。
                    decision = f"{decision}_discontinuous"

                stored_end = (
                    int(selected_end) if selected_end is not None else -1
                )
                selected_entry_end_idx[sample] = stored_end
                records.append({
                    "machining_segment_id": segment_id,
                    "segment_start_idx": segment_start,
                    "segment_end_idx": segment_end,
                    "entry_start_idx": entry_start if entry_start >= 0 else None,
                    "local_peak_idx": local_peak,
                    "peak_candidate_idx": (
                        peak_candidate if peak_candidate >= 0 else None
                    ),
                    "peak_method": peak_method,
                    "first_steady_start_idx": steady_start,
                    "selected_entry_end_idx": (
                        stored_end if selected_end is not None else None
                    ),
                    "entry_required": entry_is_continuous,
                    "entry_is_empty": bool(
                        selected_end is not None and selected_end < entry_start
                    ),
                    "decision": decision,
                })

        self._frame["machining_first_steady_idx"] = first_steady_idx
        self._frame["machining_last_steady_idx"] = last_steady_idx
        self._frame["selected_entry_end_idx"] = selected_entry_end_idx
        self._frame["machining_selected_entry_end_idx"] = (
            selected_entry_end_idx
        )
        self._frame["entry_required"] = entry_required
        self._frame["entry_phase_eligible"] = entry_eligible
        self._frame["exit_phase_eligible"] = exit_eligible
        self._selected_entry_end_indices = selected_entry_end_idx
        self._entry_required_values = entry_required
        self._prefixes["entry_required"] = _prefix(entry_required.astype(float))
        self._prefixes["entry_phase_eligible"] = _prefix(
            entry_eligible.astype(float)
        )
        self._prefixes["exit_phase_eligible"] = _prefix(
            exit_eligible.astype(float)
        )
        self.steady_precedence_entry_clipped_point_count = int(
            clipped_point_count
        )
        self.entry_boundary_records = records

    def _build_single_atom_score_cache(self) -> None:
        """一次性计算单点原子的状态分数，避免在 DP 中重复建临时数组。"""

        self._single_atom_score_values = {}
        self._single_atom_continuation_values = np.asarray([], dtype=float)
        atom_count = len(self._atoms)
        if (
            atom_count == 0
            or not np.array_equal(
                self._atom_start_indices,
                self._atom_end_indices,
            )
        ):
            return

        indices = self._atom_start_indices
        cfg = self.config
        tolerance = self._hard_fraction_tolerance

        def point_values(name: str) -> np.ndarray:
            prefix = self._prefixes[name]
            return prefix[indices + 1] - prefix[indices]

        cutting = point_values("is_effective_cutting")
        idle_fraction = point_values("is_idle_gate")
        non_idle_fraction = point_values("is_non_idle")
        steady_point_fraction = point_values("steady_point_candidate")
        steady_anchor_fraction = self._steady_anchor_values[indices].astype(float)
        entry_required_fraction = self._entry_required_values[indices].astype(float)
        mean_mrr = point_values("MRR_program")
        mean_mrr_square = (
            self._square_prefixes["MRR_program"][indices + 1]
            - self._square_prefixes["MRR_program"][indices]
        )
        std_mrr = np.sqrt(
            np.maximum(mean_mrr_square - mean_mrr * mean_mrr, 0.0)
        )
        variation = std_mrr / np.maximum(
            np.abs(mean_mrr),
            max(float(cfg.mrr_cutting_epsilon), 1e-12),
        )
        stability = np.maximum(
            1.0
            - variation
            / max(float(cfg.steady_mrr_relative_std_max), 1e-12),
            0.0,
        )
        relative_slope = self._relative_path_slopes(
            indices,
            indices,
            np.maximum(self._atom_end_s - self._atom_start_s, 0.0),
            mean_mrr,
        )
        raw_mrr = self._mrr_values[indices]
        mrr_scale = np.maximum(
            np.maximum(np.abs(raw_mrr), np.abs(mean_mrr)),
            1e-9,
        )
        trend_relative = np.clip(
            point_values("MRR_program_local_trend") / mrr_scale,
            -2.0,
            2.0,
        )
        relative_position = point_values("machining_segment_relative_position")
        entry_fraction = point_values("entry_phase_eligible")
        exit_fraction = point_values("exit_phase_eligible")

        before_non_idle = np.zeros(atom_count, dtype=float)
        has_before = indices > 0
        before_non_idle[has_before] = self._non_idle_values[
            indices[has_before] - 1
        ].astype(float)
        after_non_idle = np.zeros(atom_count, dtype=float)
        has_after = indices + 1 < len(self._non_idle_values)
        after_non_idle[has_after] = self._non_idle_values[
            indices[has_after] + 1
        ].astype(float)

        segment_ids = self._machining_segment_ids[indices]
        same_machining_segment = (
            (segment_ids > 0)
            & (non_idle_fraction >= 1.0 - tolerance)
        )
        entry_allowed = (
            same_machining_segment
            & (entry_fraction >= 1.0 - tolerance)
            & (entry_required_fraction >= 1.0 - tolerance)
            & (indices == self._machining_entry_start_indices[indices])
            & (indices == self._selected_entry_end_indices[indices])
            & self._entry_boundary_values[indices]
        )
        # 单点原子不满足至少两个点的 steady 定义，因此只能作为已有
        # steady 的续块；nonsteady/exit 仍按相同锚点优先级判定。
        unstable_non_idle = (
            same_machining_segment
            & (entry_required_fraction <= tolerance)
            & (steady_anchor_fraction <= tolerance)
        )
        exit_allowed = (
            unstable_non_idle
            & (exit_fraction >= 1.0 - tolerance)
            & (indices == self._machining_segment_end_indices[indices])
            & self._exit_boundary_values[indices]
        )

        rising = np.maximum(trend_relative, 0.0)
        falling = np.maximum(-trend_relative, 0.0)
        variation_limited = np.minimum(variation, 2.0)
        platform_trend = np.maximum(np.abs(trend_relative), relative_slope)
        entry_raw = (
            cfg.cutting_match_weight * (0.25 + 0.75 * cutting)
            + cfg.boundary_context_weight * (1.0 - before_non_idle)
            + cfg.trend_weight * rising
            + cfg.variation_weight * variation_limited * 0.25
            + cfg.boundary_context_weight * (1.0 - relative_position) * 0.25
        )
        nonsteady_raw = (
            cfg.cutting_match_weight * cutting * 0.65
            + cfg.variation_weight * variation_limited
            + cfg.trend_weight * platform_trend * 0.25
        )
        exit_raw = (
            cfg.cutting_match_weight * (0.25 + 0.75 * cutting)
            + cfg.boundary_context_weight * (1.0 - after_non_idle)
            + cfg.trend_weight * falling
            + cfg.variation_weight * variation_limited * 0.25
            + cfg.boundary_context_weight * relative_position * 0.25
        )
        negative = np.full(atom_count, -np.inf, dtype=float)
        self._single_atom_score_values = {
            SegmentState.IDLE: np.where(
                idle_fraction >= 1.0 - tolerance,
                np.full(atom_count, float(cfg.idle_match_weight)),
                negative,
            ),
            SegmentState.ENTRY: np.where(entry_allowed, entry_raw, negative),
            SegmentState.STEADY: negative.copy(),
            SegmentState.TRANSITION: negative.copy(),
            SegmentState.NONSTEADY: np.where(
                unstable_non_idle,
                nonsteady_raw,
                negative,
            ),
            SegmentState.EXIT: np.where(exit_allowed, exit_raw, negative),
        }

        continuation_allowed = (
            same_machining_segment
            & (entry_required_fraction <= tolerance)
            & (steady_point_fraction >= 1.0 - tolerance)
            & (mean_mrr > float(cfg.mrr_cutting_epsilon))
            & (variation <= float(cfg.steady_mrr_relative_std_max))
            & (relative_slope <= float(cfg.steady_mrr_relative_slope_max))
        )
        continuation_raw = (
            cfg.cutting_match_weight * cutting
            + cfg.stability_weight * (1.0 + stability)
            - cfg.variation_weight * variation_limited
            - cfg.trend_weight * platform_trend
        )
        self._single_atom_continuation_values = np.where(
            continuation_allowed,
            continuation_raw,
            negative,
        )

    def _point_bounds(self, start_atom: int, end_atom: int) -> tuple[int, int]:
        if not self._prepared:
            raise RuntimeError("评分器尚未 prepare")
        if start_atom < 0 or end_atom < start_atom or end_atom >= len(self._atoms):
            raise IndexError("原子段范围越界")
        return self._atoms[start_atom].start_idx, self._atoms[end_atom].end_idx

    def _mean(self, name: str, start_idx: int, end_idx: int) -> float:
        count = end_idx - start_idx + 1
        values = self._prefixes[name]
        return float((values[end_idx + 1] - values[start_idx]) / max(count, 1))

    def _std(self, name: str, start_idx: int, end_idx: int) -> float:
        count = end_idx - start_idx + 1
        total = self._prefixes[name][end_idx + 1] - self._prefixes[name][start_idx]
        total2 = self._square_prefixes[name][end_idx + 1] - self._square_prefixes[name][start_idx]
        mean = total / max(count, 1)
        return float(np.sqrt(max(total2 / max(count, 1) - mean * mean, 0.0)))

    def _relative_path_slopes(
        self,
        start_indices: np.ndarray,
        end_idx: int | np.ndarray,
        lengths: np.ndarray,
        mean_mrr: np.ndarray,
    ) -> np.ndarray:
        """按实际行程拟合 MRR，并返回候选全跨度上的相对漂移。"""

        starts = np.asarray(start_indices, dtype=int)
        ends = np.asarray(end_idx, dtype=int)
        candidate_lengths = np.asarray(lengths, dtype=float)
        means = np.asarray(mean_mrr, dtype=float)
        x_prefix = self._prefixes["path_center"]
        x_square_prefix = self._square_prefixes["path_center"]
        xy_prefix = self._prefixes["path_mrr_product"]
        y_prefix = self._prefixes["MRR_program"]
        if ends.ndim == 0:
            scalar_end = int(ends)
            counts = (scalar_end - starts + 1).astype(float)
            sum_x = x_prefix[scalar_end + 1] - x_prefix[starts]
            sum_x2 = x_square_prefix[scalar_end + 1] - x_square_prefix[starts]
            sum_xy = xy_prefix[scalar_end + 1] - xy_prefix[starts]
            sum_y = y_prefix[scalar_end + 1] - y_prefix[starts]
        else:
            ends = np.broadcast_to(ends, starts.shape)
            counts = (ends - starts + 1).astype(float)
            sum_x = x_prefix[ends + 1] - x_prefix[starts]
            sum_x2 = x_square_prefix[ends + 1] - x_square_prefix[starts]
            sum_xy = xy_prefix[ends + 1] - xy_prefix[starts]
            sum_y = y_prefix[ends + 1] - y_prefix[starts]
        denominator = counts * sum_x2 - sum_x * sum_x
        slopes = np.zeros(len(starts), dtype=float)
        valid = np.abs(denominator) > max(
            float(self.config.path_tolerance_mm),
            np.finfo(float).eps,
        )
        slopes[valid] = (
            counts[valid] * sum_xy[valid] - sum_x[valid] * sum_y[valid]
        ) / denominator[valid]
        scale = np.maximum(
            np.abs(means),
            max(float(self.config.mrr_cutting_epsilon), 1e-12),
        )
        return np.abs(slopes) * np.maximum(candidate_lengths, 0.0) / scale

    @lru_cache(maxsize=4096)
    def _summary(self, start_atom: int, end_atom: int) -> Dict[str, float]:
        start_idx, end_idx = self._point_bounds(start_atom, end_atom)
        point_count = int(end_idx - start_idx + 1)
        length_mm = max(
            float(self._atoms[end_atom].end_s)
            - float(self._atoms[start_atom].start_s),
            0.0,
        )
        cutting = self._mean("is_effective_cutting", start_idx, end_idx)
        idle_fraction = self._mean("is_idle_gate", start_idx, end_idx)
        non_idle_fraction = self._mean("is_non_idle", start_idx, end_idx)
        steady_point_fraction = self._mean(
            "steady_point_candidate", start_idx, end_idx
        )
        steady_anchor_fraction = self._mean(
            "steady_anchor_candidate", start_idx, end_idx
        )
        entry_required_fraction = self._mean(
            "entry_required", start_idx, end_idx
        )
        mean_mrr = self._mean("MRR_program", start_idx, end_idx)
        std_mrr = self._std("MRR_program", start_idx, end_idx)
        variation = float(
            std_mrr
            / max(abs(mean_mrr), float(self.config.mrr_cutting_epsilon), 1e-12)
        )
        threshold = max(float(self.config.steady_mrr_relative_std_max), 1e-12)
        stability = max(1.0 - variation / threshold, 0.0)
        relative_slope = float(self._relative_path_slopes(
            np.asarray([start_idx], dtype=int),
            end_idx,
            np.asarray([length_mm], dtype=float),
            np.asarray([mean_mrr], dtype=float),
        )[0])

        start_mrr = float(self._mrr_values[start_idx])
        end_mrr = float(self._mrr_values[end_idx])
        mrr_scale = max(abs(start_mrr), abs(end_mrr), abs(mean_mrr), 1e-9)
        signed_change = float(np.clip((end_mrr - start_mrr) / mrr_scale, -2.0, 2.0))
        trend = self._mean("MRR_program_local_trend", start_idx, end_idx)
        trend_relative = float(np.clip(trend / mrr_scale, -2.0, 2.0))
        before_non_idle = float(self._non_idle_values[start_idx - 1]) if start_idx > 0 else 0.0
        after_non_idle = (
            float(self._non_idle_values[end_idx + 1])
            if end_idx + 1 < len(self._non_idle_values)
            else 0.0
        )
        segment_id = int(self._machining_segment_ids[start_idx])
        same_machining_segment = bool(
            segment_id > 0
            and segment_id == int(self._machining_segment_ids[end_idx])
            and non_idle_fraction >= 1.0 - self._hard_fraction_tolerance
        )
        return {
            "cutting": cutting,
            "idle_fraction": idle_fraction,
            "non_idle_fraction": non_idle_fraction,
            "steady_point_fraction": steady_point_fraction,
            "steady_anchor_fraction": steady_anchor_fraction,
            "entry_required_fraction": entry_required_fraction,
            "same_machining_segment": float(same_machining_segment),
            "mean_mrr": mean_mrr,
            "variation": variation,
            "stability": stability,
            "steady_eligible": float(
                same_machining_segment
                and entry_required_fraction <= self._hard_fraction_tolerance
                and steady_point_fraction >= 1.0 - self._hard_fraction_tolerance
                and mean_mrr > float(self.config.mrr_cutting_epsilon)
                and variation <= float(self.config.steady_mrr_relative_std_max)
                and relative_slope
                <= float(self.config.steady_mrr_relative_slope_max)
                and point_count >= int(self.config.steady_min_plateau_points)
            ),
            "steady_chunk_eligible": float(
                same_machining_segment
                and entry_required_fraction <= self._hard_fraction_tolerance
                and steady_point_fraction >= 1.0 - self._hard_fraction_tolerance
                and mean_mrr > float(self.config.mrr_cutting_epsilon)
                and variation <= float(self.config.steady_mrr_relative_std_max)
                and relative_slope
                <= float(self.config.steady_mrr_relative_slope_max)
            ),
            "signed_change": signed_change,
            "trend_relative": trend_relative,
            "relative_slope": relative_slope,
            "point_count": float(point_count),
            "before_non_idle": before_non_idle,
            "after_non_idle": after_non_idle,
            "relative_position": self._mean(
                "machining_segment_relative_position", start_idx, end_idx
            ),
            "entry_phase_fraction": self._mean(
                "entry_phase_eligible", start_idx, end_idx
            ),
            "exit_phase_fraction": self._mean(
                "exit_phase_eligible", start_idx, end_idx
            ),
            "entry_starts_segment": float(
                start_idx == int(self._machining_entry_start_indices[start_idx])
            ),
            "entry_ends_at_selected_boundary": float(
                end_idx == int(self._selected_entry_end_indices[end_idx])
            ),
            "entry_follows_idle": float(self._entry_boundary_values[start_idx]),
            "exit_ends_segment": float(
                end_idx == int(self._machining_segment_end_indices[end_idx])
            ),
            "exit_precedes_idle": float(self._exit_boundary_values[end_idx]),
            "length_mm": length_mm,
        }

    def _state_allowed(
        self,
        m: Dict[str, float],
        state: SegmentState,
        *,
        enforce_anchor_priority: bool = True,
    ) -> bool:
        tolerance = self._hard_fraction_tolerance
        all_idle = m["idle_fraction"] >= 1.0 - tolerance
        if state is SegmentState.IDLE:
            return bool(all_idle)
        all_non_idle = (
            m["non_idle_fraction"] >= 1.0 - tolerance
            and bool(m["same_machining_segment"])
        )
        if not all_non_idle:
            return False
        if state is SegmentState.ENTRY:
            return bool(
                m["entry_starts_segment"]
                and m["entry_ends_at_selected_boundary"]
                and m["entry_phase_fraction"] >= 1.0 - tolerance
                and m["entry_required_fraction"] >= 1.0 - tolerance
                and m["entry_follows_idle"]
            )
        # 这里的 entry_required 已经是稳态父平台先确定后生成的
        # 最终范围，因此可以作为其他非 idle 状态的硬排除区。
        if m["entry_required_fraction"] > tolerance:
            return False
        steady_eligible = bool(m["steady_eligible"])
        if state is SegmentState.STEADY:
            return steady_eligible
        # transition 只由基础划分完成后确定的严格 steady 平台 P
        # 内部裁边产生，不再作为解码候选从 P 外侧抢占。
        if state is SegmentState.TRANSITION:
            return False
        # steady 采用硬优先级；满足稳定阈值的候选不能被其他非 idle 状态抢占。
        if steady_eligible and not (
            state is SegmentState.NONSTEADY and not enforce_anchor_priority
        ):
            return False
        if enforce_anchor_priority and m["steady_anchor_fraction"] > tolerance:
            return False
        if state is SegmentState.EXIT:
            return bool(
                m["exit_ends_segment"]
                and m["exit_precedes_idle"]
                and m["exit_phase_fraction"] >= 1.0 - tolerance
                and m["signed_change"] <= float(self.config.tie_epsilon)
            )
        return True

    def _score_value(
        self,
        m: Dict[str, float],
        state: SegmentState,
        *,
        enforce_anchor_priority: bool = True,
    ) -> float:
        if not self._state_allowed(
            m,
            state,
            enforce_anchor_priority=enforce_anchor_priority,
        ):
            return float("-inf")
        cfg = self.config
        cutting = m["cutting"]
        variation = min(m["variation"], 2.0)
        stability = m["stability"]
        platform_trend = max(
            abs(m["trend_relative"]),
            m["relative_slope"],
        )
        rising = max(m["signed_change"], m["trend_relative"], 0.0)
        falling = max(-m["signed_change"], -m["trend_relative"], 0.0)

        if state is SegmentState.IDLE:
            score = cfg.idle_match_weight
        elif state is SegmentState.ENTRY:
            entry_context = 1.0 - m["before_non_idle"]
            score = (
                cfg.cutting_match_weight * (0.25 + 0.75 * cutting)
                + cfg.boundary_context_weight * entry_context
                + cfg.trend_weight * rising
                + cfg.variation_weight * variation * 0.25
                + cfg.boundary_context_weight * (1.0 - m["relative_position"]) * 0.25
            )
        elif state is SegmentState.STEADY:
            score = (
                cfg.cutting_match_weight * cutting
                + cfg.stability_weight * (1.0 + stability)
                - cfg.variation_weight * variation
                - cfg.trend_weight * platform_trend
            )
        elif state is SegmentState.NONSTEADY:
            score = (
                cfg.cutting_match_weight * cutting * 0.65
                + cfg.variation_weight * variation
                + cfg.trend_weight * platform_trend * 0.25
            )
        else:
            exit_context = 1.0 - m["after_non_idle"]
            score = (
                cfg.cutting_match_weight * (0.25 + 0.75 * cutting)
                + cfg.boundary_context_weight * exit_context
                + cfg.trend_weight * falling
                + cfg.variation_weight * variation * 0.25
                + cfg.boundary_context_weight * m["relative_position"] * 0.25
            )
        return float(score)

    def score_values(self, start_atom: int, end_atom: int) -> Dict[SegmentState, float]:
        summary = self._summary(start_atom, end_atom)
        return {state: self._score_value(summary, state) for state in SegmentState}

    def score_candidates(
        self,
        start_atoms: Sequence[int],
        end_atom: int,
    ) -> Dict[SegmentState, np.ndarray]:
        """向量化候选评分，并以 ``-inf`` 表示违反硬判据的状态。"""

        starts = np.asarray(start_atoms, dtype=int)
        if starts.size == 0:
            return {state: np.asarray([], dtype=float) for state in SegmentState}
        if (
            self._prepared
            and starts.size == 1
            and int(starts[0]) == int(end_atom)
            and 0 <= int(end_atom) < len(self._atoms)
            and self._single_atom_score_values
        ):
            position = int(end_atom)
            return {
                state: np.asarray([values[position]], dtype=float)
                for state, values in self._single_atom_score_values.items()
            }
        if (
            not self._prepared
            or (starts < 0).any()
            or int(end_atom) < 0
            or int(end_atom) >= len(self._atoms)
            or (starts > int(end_atom)).any()
        ):
            raise IndexError("原子段范围越界或评分器尚未 prepare")
        if (
            starts.size > 1
            and int(starts[-1]) == int(end_atom)
            and self._single_atom_score_values
        ):
            # 解码候选按起点升序排列，末项通常是当前单点原子。复用
            # 单点缓存，仅对真正的跨点区间执行完整统计。
            interval_values = self.score_candidates(starts[:-1], end_atom)
            position = int(end_atom)
            return {
                state: np.concatenate((
                    np.asarray(interval_values[state], dtype=float),
                    np.asarray(
                        [self._single_atom_score_values[state][position]],
                        dtype=float,
                    ),
                ))
                for state in SegmentState
            }
        if starts.size == 1:
            summary = self._summary(int(starts[0]), int(end_atom))
            return {
                state: np.asarray([self._score_value(summary, state)], dtype=float)
                for state in SegmentState
            }
        start_indices = self._atom_start_indices[starts]
        end_idx = int(self._atom_end_indices[int(end_atom)])
        counts = (end_idx - start_indices + 1).astype(float)
        lengths = np.maximum(
            self._atom_end_s[int(end_atom)] - self._atom_start_s[starts],
            0.0,
        )

        def means(name: str) -> np.ndarray:
            prefix = self._prefixes[name]
            return (prefix[end_idx + 1] - prefix[start_indices]) / counts

        def stds(name: str) -> np.ndarray:
            prefix = self._prefixes[name]
            square = self._square_prefixes[name]
            mean = (prefix[end_idx + 1] - prefix[start_indices]) / counts
            mean2 = (square[end_idx + 1] - square[start_indices]) / counts
            return np.sqrt(np.maximum(mean2 - mean * mean, 0.0))

        cfg = self.config
        tolerance = self._hard_fraction_tolerance
        cutting = means("is_effective_cutting")
        idle_fraction = means("is_idle_gate")
        non_idle_fraction = means("is_non_idle")
        steady_point_fraction = means("steady_point_candidate")
        steady_anchor_fraction = means("steady_anchor_candidate")
        entry_required_fraction = means("entry_required")
        mean_mrr = means("MRR_program")
        variation = stds("MRR_program") / np.maximum(
            np.abs(mean_mrr),
            max(float(cfg.mrr_cutting_epsilon), 1e-12),
        )
        relative_slope = self._relative_path_slopes(
            start_indices,
            end_idx,
            lengths,
            mean_mrr,
        )
        variation_limited = np.minimum(variation, 2.0)
        stability = np.maximum(
            1.0 - variation / max(float(cfg.steady_mrr_relative_std_max), 1e-12),
            0.0,
        )
        start_mrr = self._mrr_values[start_indices]
        end_mrr = float(self._mrr_values[end_idx])
        mrr_scale = np.maximum(
            np.maximum(np.abs(start_mrr), abs(end_mrr)),
            np.maximum(np.abs(mean_mrr), 1e-9),
        )
        signed_change = np.clip((end_mrr - start_mrr) / mrr_scale, -2.0, 2.0)
        trend_relative = np.clip(
            means("MRR_program_local_trend") / mrr_scale, -2.0, 2.0
        )
        relative_position = means("machining_segment_relative_position")
        entry_fraction = means("entry_phase_eligible")
        exit_fraction = means("exit_phase_eligible")
        before_non_idle = np.zeros(len(starts), dtype=float)
        has_before = start_indices > 0
        before_non_idle[has_before] = self._non_idle_values[
            start_indices[has_before] - 1
        ].astype(float)
        after_non_idle = (
            float(self._non_idle_values[end_idx + 1])
            if end_idx + 1 < len(self._non_idle_values)
            else 0.0
        )
        start_segment_ids = self._machining_segment_ids[start_indices]
        end_segment_id = int(self._machining_segment_ids[end_idx])
        same_machining_segment = (
            (start_segment_ids > 0)
            & (start_segment_ids == end_segment_id)
            & (non_idle_fraction >= 1.0 - tolerance)
        )
        steady_eligible = (
            same_machining_segment
            & (entry_required_fraction <= tolerance)
            & (steady_point_fraction >= 1.0 - tolerance)
            & (mean_mrr > float(cfg.mrr_cutting_epsilon))
            & (variation <= float(cfg.steady_mrr_relative_std_max))
            & (
                relative_slope
                <= float(cfg.steady_mrr_relative_slope_max)
            )
            & (counts >= int(cfg.steady_min_plateau_points))
        )
        # 基础 Viterbi 不允许较长的非稳态候选跨过合法 steady 锚点；
        # 最终裁边若因核心过短而把母区间保守降为 nonsteady，仍可由
        # score_final_segment 复核，所以该约束只施加在基础候选阶段。
        unstable_non_idle = (
            same_machining_segment
            & ~steady_eligible
            & (steady_anchor_fraction <= tolerance)
            & (entry_required_fraction <= tolerance)
        )
        frame_entry_starts = self._machining_entry_start_indices
        frame_entry_ends = self._selected_entry_end_indices
        frame_segment_ends = self._machining_segment_end_indices
        entry_allowed = (
            same_machining_segment
            & (entry_fraction >= 1.0 - tolerance)
            & (entry_required_fraction >= 1.0 - tolerance)
            & (start_indices == frame_entry_starts[start_indices])
            & (end_idx == int(frame_entry_ends[end_idx]))
            & self._entry_boundary_values[start_indices]
        )
        exit_allowed = (
            unstable_non_idle
            & (exit_fraction >= 1.0 - tolerance)
            & (end_idx == int(frame_segment_ends[end_idx]))
            & bool(self._exit_boundary_values[end_idx])
            & (signed_change <= float(cfg.tie_epsilon))
        )
        idle_allowed = idle_fraction >= 1.0 - tolerance

        rising = np.maximum(np.maximum(signed_change, trend_relative), 0.0)
        falling = np.maximum(np.maximum(-signed_change, -trend_relative), 0.0)
        platform_trend = np.maximum(np.abs(trend_relative), relative_slope)
        entry_raw = (
            cfg.cutting_match_weight * (0.25 + 0.75 * cutting)
            + cfg.boundary_context_weight * (1.0 - before_non_idle)
            + cfg.trend_weight * rising
            + cfg.variation_weight * variation_limited * 0.25
            + cfg.boundary_context_weight * (1.0 - relative_position) * 0.25
        )
        steady_raw = (
            cfg.cutting_match_weight * cutting
            + cfg.stability_weight * (1.0 + stability)
            - cfg.variation_weight * variation_limited
            - cfg.trend_weight * platform_trend
        )
        nonsteady_raw = (
            cfg.cutting_match_weight * cutting * 0.65
            + cfg.variation_weight * variation_limited
            + cfg.trend_weight * platform_trend * 0.25
        )
        exit_raw = (
            cfg.cutting_match_weight * (0.25 + 0.75 * cutting)
            + cfg.boundary_context_weight * (1.0 - after_non_idle)
            + cfg.trend_weight * falling
            + cfg.variation_weight * variation_limited * 0.25
            + cfg.boundary_context_weight * relative_position * 0.25
        )
        negative_infinity = np.full(len(starts), -np.inf, dtype=float)
        return {
            SegmentState.IDLE: np.where(
                idle_allowed,
                np.full(len(starts), float(cfg.idle_match_weight)),
                negative_infinity,
            ),
            SegmentState.ENTRY: np.where(entry_allowed, entry_raw, negative_infinity),
            SegmentState.STEADY: np.where(steady_eligible, steady_raw, negative_infinity),
            SegmentState.TRANSITION: negative_infinity.copy(),
            SegmentState.NONSTEADY: np.where(
                unstable_non_idle,
                nonsteady_raw,
                negative_infinity,
            ),
            SegmentState.EXIT: np.where(exit_allowed, exit_raw, negative_infinity),
        }

    def score_steady_continuation_candidates(
        self,
        start_atoms: Sequence[int],
        end_atom: int,
    ) -> np.ndarray:
        """评价 steady 后续同态计算块，不重复施加整段最小点数。"""

        starts = np.asarray(start_atoms, dtype=int)
        if (
            starts.size == 1
            and int(starts[0]) == int(end_atom)
            and self._single_atom_continuation_values.size == len(self._atoms)
        ):
            return np.asarray(
                [self._single_atom_continuation_values[int(end_atom)]],
                dtype=float,
            )
        values = np.full(len(start_atoms), -np.inf, dtype=float)
        cfg = self.config
        for position, start_atom in enumerate(start_atoms):
            summary = self._summary(int(start_atom), int(end_atom))
            if not bool(summary["steady_chunk_eligible"]):
                continue
            variation = min(summary["variation"], 2.0)
            platform_trend = max(
                abs(summary["trend_relative"]),
                summary["relative_slope"],
            )
            values[position] = float(
                cfg.cutting_match_weight * summary["cutting"]
                + cfg.stability_weight * (1.0 + summary["stability"])
                - cfg.variation_weight * variation
                - cfg.trend_weight * platform_trend
            )
        return values

    def structural_candidate_starts(
        self,
        end_atom: int,
        state: SegmentState,
    ) -> np.ndarray:
        if not self._prepared:
            return np.asarray([], dtype=np.int32)
        if state is SegmentState.STEADY:
            return self._steady_structural_starts_by_end.get(
                int(end_atom),
                np.asarray([], dtype=np.int32),
            )
        if state is SegmentState.EXIT:
            end_atom = int(end_atom)
            end_idx = int(self._atom_end_indices[end_atom])
            if (
                end_idx != int(self._machining_segment_end_indices[end_idx])
                or not bool(self._exit_boundary_values[end_idx])
            ):
                return np.asarray([], dtype=np.int32)
            _, maximum = self.config.duration_bounds(SegmentState.EXIT)
            first = 0
            if np.isfinite(maximum):
                first = int(np.searchsorted(
                    self._atom_start_s,
                    float(self._atom_end_s[end_atom])
                    - float(maximum)
                    - float(self.config.path_tolerance_mm),
                    side="left",
                ))
            return np.arange(max(first, 0), end_atom + 1, dtype=np.int32)
        if state is not SegmentState.ENTRY:
            return np.asarray([], dtype=np.int32)
        end_idx = int(self._atom_end_indices[int(end_atom)])
        if end_idx != int(self._selected_entry_end_indices[end_idx]):
            return np.asarray([], dtype=np.int32)
        entry_start_idx = int(self._machining_entry_start_indices[end_idx])
        if entry_start_idx < 0:
            return np.asarray([], dtype=np.int32)
        position = int(np.searchsorted(
            self._atom_start_indices,
            entry_start_idx,
            side="left",
        ))
        if (
            position >= len(self._atom_start_indices)
            or int(self._atom_start_indices[position]) != entry_start_idx
            or position > int(end_atom)
        ):
            return np.asarray([], dtype=np.int32)
        return np.asarray([position], dtype=np.int32)

    def structural_candidates_are_complete(self, state: SegmentState) -> bool:
        # RuleSegmentScorer 的 exit 终点受切削段边界硬约束；普通点
        # 不可能产生合法退刀候选，因此无需展开每个点的毫米窗口。
        return state is SegmentState.EXIT

    def _score_segment(
        self,
        start_atom: int,
        end_atom: int,
        state: SegmentState,
        *,
        enforce_anchor_priority: bool = True,
    ) -> SegmentScore:
        m = self._summary(start_atom, end_atom)
        score = self._score_value(
            m,
            state,
            enforce_anchor_priority=enforce_anchor_priority,
        )
        if not np.isfinite(score):
            reason = (
                "候选违反硬判据："
                f"idle比例={m['idle_fraction']:.3f}，"
                f"非idle比例={m['non_idle_fraction']:.3f}，"
                f"局部稳态候选比例={m['steady_point_fraction']:.3f}，"
                f"合法稳态锚点比例={m['steady_anchor_fraction']:.3f}，"
                f"候选整体MRR相对标准差={m['variation']:.3f}，"
                f"候选整体MRR行程拟合漂移={m['relative_slope']:.3f}，"
                f"候选点数={int(m['point_count'])}/"
                f"{int(self.config.steady_min_plateau_points)}，"
                f"entry前接有效idle={bool(m['entry_follows_idle'])}，"
                f"exit后接有效idle={bool(m['exit_precedes_idle'])}"
            )
        elif state is SegmentState.IDLE:
            reason = "全部点均不满足 ap>0、ae>0、F_program>0 与重算 MRR 门限"
        elif state is SegmentState.ENTRY:
            rising = max(m["signed_change"], m["trend_relative"], 0.0)
            reason = (
                "直接位于有效idle之后，并精确结束于局部峰值"
                f"或稳态前一点，MRR上升={rising:.3f}"
            )
        elif state is SegmentState.STEADY:
            reason = (
                f"MRR均值={m['mean_mrr']:.6g}，"
                f"局部稳态候选比例={m['steady_point_fraction']:.3f}，"
                f"候选整体MRR相对标准差={m['variation']:.3f}，"
                f"候选整体MRR行程拟合漂移={m['relative_slope']:.3f}，"
                f"候选点数={int(m['point_count'])}"
            )
        elif state is SegmentState.NONSTEADY:
            reason = (
                f"MRR相对标准差={m['variation']:.3f}，"
                f"MRR行程拟合漂移={m['relative_slope']:.3f}，"
                f"候选点数={int(m['point_count'])}，归入剩余非稳态"
            )
        else:
            falling = max(-m["signed_change"], -m["trend_relative"], 0.0)
            reason = f"直接连接后续有效idle的加工段后阶段，MRR下降={falling:.3f}"
        return SegmentScore(score=score, reason=reason)

    def score_segment(
        self,
        start_atom: int,
        end_atom: int,
        state: SegmentState,
    ) -> SegmentScore:
        return self._score_segment(start_atom, end_atom, state)

    def register_transition_trims(
        self,
        trims: Mapping[Tuple[int, int], SegmentScore],
    ) -> None:
        """登记已经由解码后处理证明位于严格 steady 平台 P 内的最终裁边。

        该接口只接受最终 transition 的精确原子范围，不搜索、扩张或修正
        边界；每次调用都会替换旧登记，避免上一轮平台边界继续生效。
        """
        self._transition_trim_scores = {
            (int(start_atom), int(end_atom)): score
            for (start_atom, end_atom), score in trims.items()
            if np.isfinite(float(score.score))
        }

    def score_final_segment(
        self,
        start_atom: int,
        end_atom: int,
        state: SegmentState,
    ) -> SegmentScore:
        if state is SegmentState.TRANSITION:
            return self._transition_trim_scores.get(
                (int(start_atom), int(end_atom)),
                SegmentScore(
                    score=float("-inf"),
                    reason="该 transition 不是从严格 steady 平台 P 内部登记的裁边",
                ),
            )
        return self._score_segment(
            start_atom,
            end_atom,
            state,
            # 严格平台无法形成双侧 transition 和合法最终核心时，会按业务
            # 规则保守降为 nonsteady；最终复核必须允许这一明确后处理。
            enforce_anchor_priority=state is not SegmentState.NONSTEADY,
        )
