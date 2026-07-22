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
    """功率只做 idle 硬门控，其余五态只按重算 MRR 评分。"""

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
        self._machining_segment_start_indices = np.asarray([], dtype=np.int32)
        self._machining_segment_end_indices = np.asarray([], dtype=np.int32)
        self._machining_segment_first_peak_indices = np.asarray([], dtype=np.int32)
        self._machining_segment_last_peak_indices = np.asarray([], dtype=np.int32)
        self._entry_boundary_values = np.asarray([], dtype=bool)
        self._exit_boundary_values = np.asarray([], dtype=bool)
        self._atom_start_indices = np.asarray([], dtype=np.int32)
        self._atom_end_indices = np.asarray([], dtype=np.int32)
        self._atom_start_s = np.asarray([], dtype=float)
        self._atom_end_s = np.asarray([], dtype=float)
        self._path_center_values = np.asarray([], dtype=float)
        self._transition_trim_scores: Dict[Tuple[int, int], SegmentScore] = {}
        self.steady_anchor_candidate_point_count = 0
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
        self._machining_segment_start_indices = self._frame[
            "machining_segment_start_idx"
        ].to_numpy(dtype=np.int32)
        self._machining_segment_end_indices = self._frame[
            "machining_segment_end_idx"
        ].to_numpy(dtype=np.int32)
        self._machining_segment_first_peak_indices = self._frame[
            "machining_segment_first_peak_idx"
        ].to_numpy(dtype=np.int32)
        self._machining_segment_last_peak_indices = self._frame[
            "machining_segment_last_peak_idx"
        ].to_numpy(dtype=np.int32)
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
        self._summary.cache_clear()

    def _build_steady_anchor_values(self) -> np.ndarray:
        """标记确实能参与合法稳态候选的点，而非仅局部看似稳定的点。"""

        anchor_delta = np.zeros(len(self._frame) + 1, dtype=np.int32)
        if not self._atoms:
            return anchor_delta[:-1].astype(bool)
        minimum, maximum = self.config.duration_bounds(SegmentState.STEADY)
        tolerance = float(self.config.path_tolerance_mm)
        max_atoms = int(self.config.max_segment_atoms)
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
        for end_atom in range(len(self._atoms)):
            first_start = max(0, end_atom - max_atoms + 1)
            start_groups = [
                np.arange(first_start, end_atom + 1, dtype=np.int32)
            ]
            latest_minimum_start = int(np.searchsorted(
                self._atom_start_s,
                float(self._atom_end_s[end_atom]) - minimum + tolerance,
                side="right",
            )) - 1
            if latest_minimum_start >= 0:
                latest_minimum_start = min(latest_minimum_start, end_atom)
                physical_window_start = max(
                    0,
                    latest_minimum_start - max_atoms + 1,
                )
                start_groups.append(np.arange(
                    physical_window_start,
                    latest_minimum_start + 1,
                    dtype=np.int32,
                ))
            starts = np.unique(np.concatenate(start_groups))
            lengths = np.maximum(
                self._atom_end_s[end_atom] - self._atom_start_s[starts],
                0.0,
            )
            valid_length = (
                (lengths >= minimum - tolerance)
                & (lengths <= maximum + tolerance)
            )
            if not np.any(valid_length):
                continue
            starts = starts[valid_length]
            lengths = lengths[valid_length]
            start_indices = self._atom_start_indices[starts]
            end_idx = int(self._atom_end_indices[end_atom])
            counts = (end_idx - start_indices + 1).astype(float)

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
                lengths,
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
            mrr_scale = np.maximum.reduce(
                (
                    np.abs(start_mrr),
                    np.full(len(starts), abs(end_mrr)),
                    np.abs(mean_mrr),
                    np.full(len(starts), 1e-9),
                )
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
            eligible_starts = start_indices[
                steady_eligible & np.isfinite(steady_score)
            ]
            if eligible_starts.size:
                np.add.at(anchor_delta, eligible_starts, 1)
                anchor_delta[end_idx + 1] -= int(eligible_starts.size)
        return np.cumsum(anchor_delta[:-1]) > 0

    def _rebuild_phase_eligibility_from_steady_anchors(self) -> None:
        """只用合法 steady 锚点收紧每个加工段的进刀/退刀阶段。"""

        count = len(self._frame)
        entry_eligible = np.zeros(count, dtype=bool)
        exit_eligible = np.zeros(count, dtype=bool)
        first_steady_idx = np.full(count, -1, dtype=np.int32)
        last_steady_idx = np.full(count, -1, dtype=np.int32)
        if count:
            segment_ids = self._machining_segment_ids
            starts = np.flatnonzero(
                (segment_ids > 0)
                & np.concatenate(([True], segment_ids[1:] != segment_ids[:-1]))
            )
            for start_idx in starts:
                start_idx = int(start_idx)
                end_idx = int(self._machining_segment_end_indices[start_idx])
                anchor_indices = np.flatnonzero(
                    self._steady_anchor_values[start_idx:end_idx + 1]
                )
                first_peak_idx = int(self._machining_segment_first_peak_indices[start_idx])
                last_peak_idx = int(self._machining_segment_last_peak_indices[start_idx])
                if anchor_indices.size:
                    first_anchor = start_idx + int(anchor_indices[0])
                    last_anchor = start_idx + int(anchor_indices[-1])
                    first_steady_idx[start_idx:end_idx + 1] = first_anchor
                    last_steady_idx[start_idx:end_idx + 1] = last_anchor
                    entry_end = min(first_peak_idx, first_anchor - 1)
                    exit_start = max(last_peak_idx, last_anchor + 1)
                else:
                    entry_end = first_peak_idx
                    exit_start = last_peak_idx
                if entry_end >= start_idx:
                    entry_eligible[start_idx:entry_end + 1] = True
                if exit_start <= end_idx:
                    exit_eligible[exit_start:end_idx + 1] = True

        self._frame["machining_first_steady_idx"] = first_steady_idx
        self._frame["machining_last_steady_idx"] = last_steady_idx
        self._frame["entry_phase_eligible"] = entry_eligible
        self._frame["exit_phase_eligible"] = exit_eligible
        self._prefixes["entry_phase_eligible"] = _prefix(
            entry_eligible.astype(float)
        )
        self._prefixes["exit_phase_eligible"] = _prefix(
            exit_eligible.astype(float)
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
        end_idx: int,
        lengths: np.ndarray,
        mean_mrr: np.ndarray,
    ) -> np.ndarray:
        """按实际行程拟合 MRR，并返回候选全跨度上的相对漂移。"""

        starts = np.asarray(start_indices, dtype=int)
        candidate_lengths = np.asarray(lengths, dtype=float)
        means = np.asarray(mean_mrr, dtype=float)
        counts = (int(end_idx) - starts + 1).astype(float)
        x_prefix = self._prefixes["path_center"]
        x_square_prefix = self._square_prefixes["path_center"]
        xy_prefix = self._prefixes["path_mrr_product"]
        y_prefix = self._prefixes["MRR_program"]
        sum_x = x_prefix[end_idx + 1] - x_prefix[starts]
        sum_x2 = x_square_prefix[end_idx + 1] - x_square_prefix[starts]
        sum_xy = xy_prefix[end_idx + 1] - xy_prefix[starts]
        sum_y = y_prefix[end_idx + 1] - y_prefix[starts]
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
            "same_machining_segment": float(same_machining_segment),
            "mean_mrr": mean_mrr,
            "variation": variation,
            "stability": stability,
            "steady_eligible": float(
                same_machining_segment
                and steady_point_fraction >= 1.0 - self._hard_fraction_tolerance
                and mean_mrr > float(self.config.mrr_cutting_epsilon)
                and variation <= float(self.config.steady_mrr_relative_std_max)
                and relative_slope
                <= float(self.config.steady_mrr_relative_slope_max)
                and point_count >= int(self.config.steady_min_plateau_points)
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
                start_idx == int(self._machining_segment_start_indices[start_idx])
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
        if state is SegmentState.ENTRY:
            return bool(
                m["entry_starts_segment"]
                and m["entry_follows_idle"]
                and m["entry_phase_fraction"] >= 1.0 - tolerance
                and m["signed_change"] >= -float(self.config.tie_epsilon)
            )
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
            not self._prepared
            or np.any(starts < 0)
            or int(end_atom) < 0
            or int(end_atom) >= len(self._atoms)
            or np.any(starts > int(end_atom))
        ):
            raise IndexError("原子段范围越界或评分器尚未 prepare")
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
        mrr_scale = np.maximum.reduce(
            (
                np.abs(start_mrr),
                np.full(len(starts), abs(end_mrr)),
                np.abs(mean_mrr),
                np.full(len(starts), 1e-9),
            )
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
        )
        frame_segment_starts = self._machining_segment_start_indices
        frame_segment_ends = self._machining_segment_end_indices
        entry_allowed = (
            unstable_non_idle
            & (entry_fraction >= 1.0 - tolerance)
            & (start_indices == frame_segment_starts[start_indices])
            & self._entry_boundary_values[start_indices]
            & (signed_change >= -float(cfg.tie_epsilon))
        )
        exit_allowed = (
            unstable_non_idle
            & (exit_fraction >= 1.0 - tolerance)
            & (end_idx == int(frame_segment_ends[end_idx]))
            & bool(self._exit_boundary_values[end_idx])
            & (signed_change <= float(cfg.tie_epsilon))
        )
        idle_allowed = idle_fraction >= 1.0 - tolerance

        rising = np.maximum.reduce((signed_change, trend_relative, np.zeros(len(starts))))
        falling = np.maximum.reduce((-signed_change, -trend_relative, np.zeros(len(starts))))
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
            reason = "全部点满足 P_pred <= P_idle + idle_power_tolerance"
        elif state is SegmentState.ENTRY:
            rising = max(m["signed_change"], m["trend_relative"], 0.0)
            reason = f"直接位于有效idle之后的加工段前阶段，MRR上升={rising:.3f}"
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
