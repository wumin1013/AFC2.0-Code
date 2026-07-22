from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .schemas import (
    ALLOWED_TRANSITIONS,
    AtomicSegment,
    DecodeResult,
    DecodedSegment,
    SegmentScore,
    SegmentState,
    SegmentationConfig,
    is_transition_allowed,
)
from .scorers import SegmentScorer


STATE_ORDER: Tuple[SegmentState, ...] = tuple(SegmentState)
STATE_INDEX: Dict[SegmentState, int] = {state: idx for idx, state in enumerate(STATE_ORDER)}


def _merge_adjacent_segments(
    segments: Sequence[DecodedSegment],
) -> Tuple[DecodedSegment, ...]:
    """合并连续同态分块，同时保留首尾原子段和过程点边界。"""

    merged: List[DecodedSegment] = []
    for segment in segments:
        if merged and merged[-1].state is segment.state:
            previous = merged[-1]
            merged[-1] = DecodedSegment(
                start_atom=previous.start_atom,
                end_atom=segment.end_atom,
                start_idx=previous.start_idx,
                end_idx=segment.end_idx,
                state=segment.state,
            )
        else:
            merged.append(segment)
    return tuple(merged)


def count_transition_without_steady_neighbor(
    segments: Sequence[DecodedSegment],
) -> int:
    """统计没有直接邻接 steady 的 transition 连续区间。"""

    count = 0
    for index, segment in enumerate(segments):
        if segment.state is not SegmentState.TRANSITION:
            continue
        left_is_steady = index > 0 and segments[index - 1].state is SegmentState.STEADY
        right_is_steady = (
            index + 1 < len(segments)
            and segments[index + 1].state is SegmentState.STEADY
        )
        if not (left_is_steady or right_is_steady):
            count += 1
    return int(count)


def count_entry_without_idle_predecessor(
    segments: Sequence[DecodedSegment],
) -> int:
    """统计没有直接位于 idle 之后的 entry 连续区间。"""

    return int(sum(
        segment.state is SegmentState.ENTRY
        and (index == 0 or segments[index - 1].state is not SegmentState.IDLE)
        for index, segment in enumerate(segments)
    ))


def count_exit_without_idle_successor(
    segments: Sequence[DecodedSegment],
) -> int:
    """统计没有直接连接后续 idle 的 exit 连续区间。"""

    return int(sum(
        segment.state is SegmentState.EXIT
        and (
            index + 1 >= len(segments)
            or segments[index + 1].state is not SegmentState.IDLE
        )
        for index, segment in enumerate(segments)
    ))


def normalize_steady_segments(
    segments: Sequence[DecodedSegment],
    scorer: SegmentScorer,
    atoms: Sequence[AtomicSegment],
    config: SegmentationConfig,
) -> Tuple[Tuple[DecodedSegment, ...], Dict[str, int]]:
    """复查 steady，并从非法合并段中至多恢复一个合法稳态核心。"""

    source = tuple(segments)
    atoms = tuple(atoms)
    reclassified: List[DecodedSegment] = []
    steady_count = 0
    reclassified_count = 0
    short_reclassified_count = 0
    recovered_core_count = 0
    recovered_core_point_count = 0
    core_candidate_count = 0
    without_core_reclassified_count = 0
    split_count = 0
    tolerance = float(config.path_tolerance_mm)

    def decoded_segment(
        start_atom: int,
        end_atom: int,
        state: SegmentState,
    ) -> DecodedSegment:
        return DecodedSegment(
            start_atom=start_atom,
            end_atom=end_atom,
            start_idx=atoms[start_atom].start_idx,
            end_idx=atoms[end_atom].end_idx,
            state=state,
        )

    def select_core(
        segment: DecodedSegment,
    ) -> Tuple[int, int] | None:
        nonlocal core_candidate_count
        best: Tuple[int, int] | None = None
        best_length = float("-inf")
        best_score = float("-inf")
        minimum, maximum = config.duration_bounds(SegmentState.STEADY)
        for end_atom in range(segment.start_atom, segment.end_atom + 1):
            starts = np.arange(segment.start_atom, end_atom + 1, dtype=int)
            lengths = np.maximum(
                float(atoms[end_atom].end_s)
                - np.asarray([atoms[start].start_s for start in starts], dtype=float),
                0.0,
            )
            valid_length = (
                (lengths >= minimum - tolerance)
                & (lengths <= maximum + tolerance)
            )
            if not np.any(valid_length):
                continue
            scores = np.asarray(
                scorer.score_candidates(starts, end_atom)[SegmentState.STEADY],
                dtype=float,
            )
            candidates = np.flatnonzero(valid_length & np.isfinite(scores))
            core_candidate_count += int(len(candidates))
            for position in candidates:
                start_atom = int(starts[position])
                if start_atom > segment.start_atom and not np.isfinite(float(
                    scorer.score_segment(
                        segment.start_atom,
                        start_atom - 1,
                        SegmentState.NONSTEADY,
                    ).score
                )):
                    continue
                if end_atom < segment.end_atom and not np.isfinite(float(
                    scorer.score_segment(
                        end_atom + 1,
                        segment.end_atom,
                        SegmentState.NONSTEADY,
                    ).score
                )):
                    continue
                length = float(lengths[position])
                score = float(scores[position])
                if (
                    length > best_length + tolerance
                    or (
                        abs(length - best_length) <= tolerance
                        and (
                            score > best_score + float(config.tie_epsilon)
                            or (
                                abs(score - best_score)
                                <= float(config.tie_epsilon)
                                and (best is None or start_atom < best[0])
                            )
                        )
                    )
                ):
                    best = (start_atom, end_atom)
                    best_length = length
                    best_score = score
        return best

    for segment in source:
        state = segment.state
        if state is SegmentState.STEADY:
            steady_count += 1
            score = scorer.score_segment(
                segment.start_atom,
                segment.end_atom,
                SegmentState.STEADY,
            ).score
            length_mm = max(
                float(atoms[segment.end_atom].end_s)
                - float(atoms[segment.start_atom].start_s),
                0.0,
            )
            length_valid = bool(
                length_mm >= float(config.min_steady_mm) - tolerance
            )
            if not length_valid:
                short_reclassified_count += 1
            if not np.isfinite(float(score)) or not length_valid:
                reclassified_count += 1
                core = select_core(segment) if length_valid else None
                if core is None:
                    without_core_reclassified_count += 1
                    reclassified.append(decoded_segment(
                        segment.start_atom,
                        segment.end_atom,
                        SegmentState.NONSTEADY,
                    ))
                    continue
                core_start, core_end = core
                recovered_core_count += 1
                recovered_core_point_count += int(
                    atoms[core_end].end_idx - atoms[core_start].start_idx + 1
                )
                pieces_before = len(reclassified)
                if core_start > segment.start_atom:
                    reclassified.append(decoded_segment(
                        segment.start_atom,
                        core_start - 1,
                        SegmentState.NONSTEADY,
                    ))
                reclassified.append(decoded_segment(
                    core_start,
                    core_end,
                    SegmentState.STEADY,
                ))
                if core_end < segment.end_atom:
                    reclassified.append(decoded_segment(
                        core_end + 1,
                        segment.end_atom,
                        SegmentState.NONSTEADY,
                    ))
                split_count += max(len(reclassified) - pieces_before - 1, 0)
                continue
        reclassified.append(decoded_segment(
            segment.start_atom,
            segment.end_atom,
            state,
        ))

    normalized = _merge_adjacent_segments(reclassified)
    diagnostics = {
        "steady_interval_count_before_normalization": int(steady_count),
        "steady_reclassified_count": int(reclassified_count),
        "steady_short_reclassified_count": int(short_reclassified_count),
        "steady_core_recovered_count": int(recovered_core_count),
        "steady_core_recovered_point_count": int(recovered_core_point_count),
        "steady_core_candidate_count": int(core_candidate_count),
        "steady_without_core_reclassified_count": int(
            without_core_reclassified_count
        ),
        "steady_normalization_split_count": int(split_count),
        "steady_normalization_merge_count": int(len(reclassified) - len(normalized)),
    }
    return normalized, diagnostics


def normalize_transition_segments(
    segments: Sequence[DecodedSegment],
    scorer: SegmentScorer,
    atoms: Sequence[AtomicSegment],
    config: SegmentationConfig,
    *,
    allow_carving: bool = True,
) -> Tuple[Tuple[DecodedSegment, ...], Dict[str, object]]:
    """先确定绿色 seed steady，再从 seed 内部按点数比例裁 transition。"""

    source = tuple(segments)
    atoms = tuple(atoms)
    tolerance = float(config.path_tolerance_mm)
    transition_ratio = float(config.transition_ratio)
    minimum_transition_points = int(config.min_transition_points)

    def decoded_segment(
        start_atom: int,
        end_atom: int,
        state: SegmentState,
    ) -> DecodedSegment:
        return DecodedSegment(
            start_atom=start_atom,
            end_atom=end_atom,
            start_idx=atoms[start_atom].start_idx,
            end_idx=atoms[end_atom].end_idx,
            state=state,
        )

    def length_mm(start_atom: int, end_atom: int) -> float:
        return max(
            float(atoms[end_atom].end_s) - float(atoms[start_atom].start_s),
            0.0,
        )

    def rounded_transition_points(parent_point_count: int) -> Tuple[float, int]:
        target = float(parent_point_count) * transition_ratio
        # Python round 使用银行家舍入；这里明确采用 0.5 向上。
        rounded = int(np.floor(target + 0.5))
        return target, max(rounded, minimum_transition_points)

    # 新语义下基础解码不应包含 transition。对自定义/旧解码结果仍做
    # 保守兼容：先把外部 transition 归为 nonsteady，再开始内部裁边。
    base_segments: List[DecodedSegment] = []
    decoded_transition_count = 0
    for segment in source:
        state = segment.state
        if state is SegmentState.TRANSITION:
            state = SegmentState.NONSTEADY
            decoded_transition_count += 1
        base_segments.append(
            DecodedSegment(
                start_atom=segment.start_atom,
                end_atom=segment.end_atom,
                start_idx=segment.start_idx,
                end_idx=segment.end_idx,
                state=state,
            )
        )
    base = _merge_adjacent_segments(base_segments)
    provisional_nonsteady_atoms = {
        atom_index
        for segment in base
        if segment.state is SegmentState.NONSTEADY
        for atom_index in range(segment.start_atom, segment.end_atom + 1)
    }
    steady_parent_atoms: set[int] = set()
    restored_outer_atoms: set[int] = set()

    carved: List[DecodedSegment] = []
    registered: Dict[Tuple[int, int], SegmentScore] = {}
    candidate_ranges: set[Tuple[int, int]] = set()
    point_carving_records: List[Dict[str, object]] = []
    base_steady_candidate_count = int(sum(
        segment.state is SegmentState.STEADY
        for segment in base
    ))
    steady_parent_count = 0
    restored_outer_interval_count = 0
    restored_outer_point_count = 0
    restored_outer_state_counts: Dict[str, int] = {}
    context_reclassified_count = 0
    atom_granularity_reclassified_count = 0
    core_reclassified_count = 0
    short_core_mm_reclassified_count = 0
    short_core_point_reclassified_count = 0
    invalid_core_score_reclassified_count = 0
    initial_states = set(config.initial_states)
    terminal_states = set(config.terminal_states)

    def restored_boundary_state(
        segment_index: int,
        restore_start: int,
        restore_end: int,
        *,
        side: str,
    ) -> Tuple[SegmentState, str, str, str]:
        """把旧 transition 恢复到合法外邻态，否则保守归为 nonsteady。"""

        neighbor_index = segment_index - 1 if side == "left" else segment_index + 1
        if neighbor_index < 0 or neighbor_index >= len(base):
            return (
                SegmentState.NONSTEADY,
                "",
                "",
                "没有对应侧外邻基础状态",
            )
        neighbor = base[neighbor_index]
        if neighbor.state is SegmentState.NONSTEADY:
            return (
                SegmentState.NONSTEADY,
                neighbor.state.value,
                SegmentState.NONSTEADY.value,
                "外邻基础状态本身为 nonsteady",
            )
        directional_state = (
            SegmentState.ENTRY if side == "left" else SegmentState.EXIT
        )
        if neighbor.state is not directional_state:
            return (
                SegmentState.NONSTEADY,
                neighbor.state.value,
                "",
                "外邻基础状态不允许向该方向继承",
            )
        merged_start = neighbor.start_atom if side == "left" else restore_start
        merged_end = restore_end if side == "left" else neighbor.end_atom
        merged_score = scorer.score_final_segment(
            merged_start,
            merged_end,
            directional_state,
        ).score
        if np.isfinite(float(merged_score)):
            return (
                directional_state,
                neighbor.state.value,
                directional_state.value,
                "",
            )
        return (
            SegmentState.NONSTEADY,
            neighbor.state.value,
            directional_state.value,
            "继承外邻状态后的合并区间评分非法，保守回退 nonsteady",
        )

    def append_failed_seed(
        left_start: int,
        left_end: int,
        left_state: SegmentState,
        seed_start: int,
        seed_end: int,
        right_start: int,
        right_end: int,
        right_state: SegmentState,
    ) -> None:
        carved.extend((
            decoded_segment(left_start, left_end, left_state),
            decoded_segment(seed_start, seed_end, SegmentState.NONSTEADY),
            decoded_segment(right_start, right_end, right_state),
        ))

    for segment_index, segment in enumerate(base):
        if segment.state is not SegmentState.STEADY:
            carved.append(segment)
            continue
        parent_atoms = atoms[segment.start_atom:segment.end_atom + 1]
        parent_point_count = int(sum(
            max(int(atom.point_count), 0) for atom in parent_atoms
        ))
        seed_target_point_count, seed_transition_point_count = rounded_transition_points(
            parent_point_count
        )
        seed_parent_point_count = max(
            parent_point_count - 2 * seed_transition_point_count,
            0,
        )
        carving_record: Dict[str, object] = {
            "provisional_parent_start_idx": int(segment.start_idx),
            "provisional_parent_end_idx": int(segment.end_idx),
            "provisional_parent_point_count": int(parent_point_count),
            "seed_extraction_target_each_side_point_count": float(
                round(seed_target_point_count, 12)
            ),
            "seed_extraction_actual_each_side_point_count": int(
                seed_transition_point_count
            ),
            "seed_parent_start_idx": None,
            "seed_parent_end_idx": None,
            "seed_parent_point_count": int(seed_parent_point_count),
            "restored_left_start_idx": None,
            "restored_left_end_idx": None,
            "restored_left_state": "",
            "restored_left_adjacent_state": "",
            "restored_left_attempted_state": "",
            "restored_left_fallback_reason": "",
            "restored_right_start_idx": None,
            "restored_right_end_idx": None,
            "restored_right_state": "",
            "restored_right_adjacent_state": "",
            "restored_right_attempted_state": "",
            "restored_right_fallback_reason": "",
            "parent_start_idx": int(segment.start_idx),
            "parent_end_idx": int(segment.end_idx),
            "parent_point_count": 0,
            "target_each_side_point_count": 0.0,
            "actual_each_side_point_count": 0,
            "rounding_error_point_count": 0.0,
            "core_point_count": 0,
            "status": "pending",
        }
        point_carving_records.append(carving_record)

        # 点数比例只有在每个原子严格对应一个 ProcessInfo 点时才能
        # 无歧义地落到原子边界。异常输入不猜测内部边界。
        single_point_atoms = bool(parent_atoms) and all(
            int(atom.point_count) == 1
            and int(atom.start_idx) == int(atom.end_idx)
            for atom in parent_atoms
        )
        if not single_point_atoms:
            atom_granularity_reclassified_count += 1
            carving_record["status"] = "atom_not_single_process_point"
            carved.append(decoded_segment(
                segment.start_atom,
                segment.end_atom,
                SegmentState.NONSTEADY,
            ))
            continue
        if not allow_carving:
            carving_record["status"] = "fallback_carving_skipped"
            # 调用方显式禁止裁边时也不能留下没有双侧 transition 的
            # steady；按业务定义保守降为 nonsteady。
            carved.append(decoded_segment(
                segment.start_atom,
                segment.end_atom,
                SegmentState.NONSTEADY,
            ))
            continue
        left_context_valid = bool(
            (
                segment_index == 0
                and SegmentState.TRANSITION.value in initial_states
            )
            or (
                segment_index > 0
                and is_transition_allowed(
                    base[segment_index - 1].state,
                    SegmentState.TRANSITION,
                )
            )
        )
        right_context_valid = bool(
            (
                segment_index + 1 == len(base)
                and SegmentState.TRANSITION.value in terminal_states
            )
            or (
                segment_index + 1 < len(base)
                and is_transition_allowed(
                    SegmentState.TRANSITION,
                    base[segment_index + 1].state,
                )
            )
        )
        if not (left_context_valid and right_context_valid):
            context_reclassified_count += 1
            carving_record["status"] = "invalid_transition_context"
            carved.append(decoded_segment(
                segment.start_atom,
                segment.end_atom,
                SegmentState.NONSTEADY,
            ))
            continue

        if seed_parent_point_count < int(config.steady_min_plateau_points):
            core_reclassified_count += 1
            short_core_point_reclassified_count += 1
            carving_record["status"] = "seed_parent_too_short_in_points"
            carved.append(decoded_segment(
                segment.start_atom,
                segment.end_atom,
                SegmentState.NONSTEADY,
            ))
            continue

        old_left_end = segment.start_atom + seed_transition_point_count - 1
        old_right_start = segment.end_atom - seed_transition_point_count + 1
        seed_start = old_left_end + 1
        seed_end = old_right_start - 1
        if seed_start > seed_end:
            core_reclassified_count += 1
            short_core_point_reclassified_count += 1
            carving_record["status"] = "seed_parent_too_short_in_points"
            carved.append(decoded_segment(
                segment.start_atom,
                segment.end_atom,
                SegmentState.NONSTEADY,
            ))
            continue

        seed_length_mm = length_mm(seed_start, seed_end)
        if seed_length_mm < float(config.min_steady_mm) - tolerance:
            core_reclassified_count += 1
            short_core_mm_reclassified_count += 1
            carving_record["status"] = "seed_parent_too_short_in_mm"
            carved.append(decoded_segment(
                segment.start_atom,
                segment.end_atom,
                SegmentState.NONSTEADY,
            ))
            continue
        if not np.isfinite(float(scorer.score_segment(
            seed_start,
            seed_end,
            SegmentState.STEADY,
        ).score)):
            core_reclassified_count += 1
            invalid_core_score_reclassified_count += 1
            carving_record["status"] = "seed_parent_score_invalid"
            carved.append(decoded_segment(
                segment.start_atom,
                segment.end_atom,
                SegmentState.NONSTEADY,
            ))
            continue

        steady_parent_count += 1
        steady_parent_atoms.update(range(seed_start, seed_end + 1))
        (
            restored_left_state,
            restored_left_adjacent_state,
            restored_left_attempted_state,
            restored_left_fallback_reason,
        ) = restored_boundary_state(
            segment_index,
            segment.start_atom,
            old_left_end,
            side="left",
        )
        (
            restored_right_state,
            restored_right_adjacent_state,
            restored_right_attempted_state,
            restored_right_fallback_reason,
        ) = restored_boundary_state(
            segment_index,
            old_right_start,
            segment.end_atom,
            side="right",
        )
        restored_outer_interval_count += 2
        restored_outer_point_count += int(2 * seed_transition_point_count)
        restored_outer_atoms.update(
            range(segment.start_atom, old_left_end + 1)
        )
        restored_outer_atoms.update(
            range(old_right_start, segment.end_atom + 1)
        )
        for restored_state in (restored_left_state, restored_right_state):
            state_name = restored_state.value
            restored_outer_state_counts[state_name] = int(
                restored_outer_state_counts.get(state_name, 0) + 1
            )
        new_target_point_count, new_transition_point_count = (
            rounded_transition_points(seed_parent_point_count)
        )
        final_core_point_count = max(
            seed_parent_point_count - 2 * new_transition_point_count,
            0,
        )
        carving_record.update({
            "seed_parent_start_idx": int(atoms[seed_start].start_idx),
            "seed_parent_end_idx": int(atoms[seed_end].end_idx),
            "restored_left_start_idx": int(segment.start_idx),
            "restored_left_end_idx": int(atoms[old_left_end].end_idx),
            "restored_left_state": restored_left_state.value,
            "restored_left_adjacent_state": restored_left_adjacent_state,
            "restored_left_attempted_state": restored_left_attempted_state,
            "restored_left_fallback_reason": restored_left_fallback_reason,
            "restored_right_start_idx": int(atoms[old_right_start].start_idx),
            "restored_right_end_idx": int(segment.end_idx),
            "restored_right_state": restored_right_state.value,
            "restored_right_adjacent_state": restored_right_adjacent_state,
            "restored_right_attempted_state": restored_right_attempted_state,
            "restored_right_fallback_reason": restored_right_fallback_reason,
            # 兼容既有比例复核字段；新语义下 parent 即绿色 seed steady。
            "parent_start_idx": int(atoms[seed_start].start_idx),
            "parent_end_idx": int(atoms[seed_end].end_idx),
            "parent_point_count": int(seed_parent_point_count),
            "target_each_side_point_count": float(
                round(new_target_point_count, 12)
            ),
            "actual_each_side_point_count": int(new_transition_point_count),
            "rounding_error_point_count": float(round(
                abs(new_transition_point_count - new_target_point_count),
                12,
            )),
            "core_point_count": int(final_core_point_count),
        })
        if final_core_point_count < int(config.steady_min_plateau_points):
            core_reclassified_count += 1
            short_core_point_reclassified_count += 1
            carving_record["status"] = "core_too_short_in_points"
            append_failed_seed(
                segment.start_atom,
                old_left_end,
                restored_left_state,
                seed_start,
                seed_end,
                old_right_start,
                segment.end_atom,
                restored_right_state,
            )
            continue

        new_left_end = seed_start + new_transition_point_count - 1
        new_right_start = seed_end - new_transition_point_count + 1
        left_key = (seed_start, new_left_end)
        right_key = (new_right_start, seed_end)
        candidate_ranges.update((left_key, right_key))
        core_start = new_left_end + 1
        core_end = new_right_start - 1
        if core_start > core_end:
            core_reclassified_count += 1
            short_core_point_reclassified_count += 1
            carving_record["status"] = "core_too_short_in_points"
            append_failed_seed(
                segment.start_atom,
                old_left_end,
                restored_left_state,
                seed_start,
                seed_end,
                old_right_start,
                segment.end_atom,
                restored_right_state,
            )
            continue

        core_length_mm = length_mm(core_start, core_end)
        if core_length_mm < float(config.min_steady_mm) - tolerance:
            core_reclassified_count += 1
            short_core_mm_reclassified_count += 1
            carving_record["status"] = "core_too_short_in_mm"
            append_failed_seed(
                segment.start_atom,
                old_left_end,
                restored_left_state,
                seed_start,
                seed_end,
                old_right_start,
                segment.end_atom,
                restored_right_state,
            )
            continue
        if not np.isfinite(float(scorer.score_segment(
            core_start,
            core_end,
            SegmentState.STEADY,
        ).score)):
            core_reclassified_count += 1
            invalid_core_score_reclassified_count += 1
            carving_record["status"] = "core_score_invalid"
            append_failed_seed(
                segment.start_atom,
                old_left_end,
                restored_left_state,
                seed_start,
                seed_end,
                old_right_start,
                segment.end_atom,
                restored_right_state,
            )
            continue

        carving_record["status"] = "carved"
        registered_score = float(config.boundary_context_weight)
        registered[left_key] = SegmentScore(
            score=registered_score,
            reason=(
                "按绿色 seed steady ProcessInfo 点数的固定比例裁出左侧 transition："
                f"目标={new_target_point_count:.6g} 点，实际={new_transition_point_count} 点"
            ),
        )
        registered[right_key] = SegmentScore(
            score=registered_score,
            reason=(
                "按绿色 seed steady ProcessInfo 点数的固定比例裁出右侧 transition："
                f"目标={new_target_point_count:.6g} 点，实际={new_transition_point_count} 点"
            ),
        )
        carved.extend((
            decoded_segment(
                segment.start_atom,
                old_left_end,
                restored_left_state,
            ),
            decoded_segment(left_key[0], left_key[1], SegmentState.TRANSITION),
            decoded_segment(core_start, core_end, SegmentState.STEADY),
            decoded_segment(right_key[0], right_key[1], SegmentState.TRANSITION),
            decoded_segment(
                old_right_start,
                segment.end_atom,
                restored_right_state,
            ),
        ))

    normalized = _merge_adjacent_segments(carved)
    scorer.register_transition_trims(registered)
    normalized_ranges = {
        (int(segment.start_idx), int(segment.end_idx), segment.state)
        for segment in normalized
    }
    final_point_ratio_violations = 0
    for record in point_carving_records:
        if str(record.get("status")) != "carved":
            continue
        parent_start = int(record["parent_start_idx"])
        parent_end = int(record["parent_end_idx"])
        parent_count = int(record["parent_point_count"])
        expected_target, expected_points = rounded_transition_points(parent_count)
        actual_points = int(record["actual_each_side_point_count"])
        core_count = int(record["core_point_count"])
        expected_ranges = {
            (
                parent_start,
                parent_start + expected_points - 1,
                SegmentState.TRANSITION,
            ),
            (
                parent_start + expected_points,
                parent_end - expected_points,
                SegmentState.STEADY,
            ),
            (
                parent_end - expected_points + 1,
                parent_end,
                SegmentState.TRANSITION,
            ),
        }
        if (
            actual_points != expected_points
            or core_count != parent_count - 2 * expected_points
            or abs(
                float(record["target_each_side_point_count"])
                - expected_target
            ) > max(tolerance, 1e-12)
            or not expected_ranges.issubset(normalized_ranges)
        ):
            final_point_ratio_violations += 1
    transition_atoms = {
        atom_index
        for segment in normalized
        if segment.state is SegmentState.TRANSITION
        for atom_index in range(segment.start_atom, segment.end_atom + 1)
    }
    restored_outer_invalid_score_count = int(sum(
        bool(
            restored_outer_atoms.intersection(
                range(segment.start_atom, segment.end_atom + 1)
            )
        )
        and not np.isfinite(float(scorer.score_final_segment(
            segment.start_atom,
            segment.end_atom,
            segment.state,
        ).score))
        for segment in normalized
    ))
    transition_outside_strict_platform_count = int(
        len(transition_atoms - steady_parent_atoms)
    )
    restored_outer_to_transition_count = int(
        len(restored_outer_atoms & transition_atoms)
    )
    final_point_count_violations = 0
    final_edge_violations = 0
    for segment in normalized:
        if segment.state is not SegmentState.TRANSITION:
            continue
        segment_point_count = int(sum(
            atoms[atom_index].point_count
            for atom_index in range(segment.start_atom, segment.end_atom + 1)
        ))
        if segment_point_count < minimum_transition_points:
            final_point_count_violations += 1
        if not np.isfinite(float(scorer.score_final_segment(
            segment.start_atom,
            segment.end_atom,
            SegmentState.TRANSITION,
        ).score)):
            final_edge_violations += 1

    carved_point_count = int(sum(
        sum(
            atoms[atom_index].point_count
            for atom_index in range(segment.start_atom, segment.end_atom + 1)
        )
        for segment in normalized
        if segment.state is SegmentState.TRANSITION
    ))
    diagnostics = {
        "decoded_interval_count_before_normalization": int(len(source)),
        "transition_interval_count_before_normalization": int(decoded_transition_count),
        "transition_reclassified_count": int(decoded_transition_count),
        "transition_neighbor_reclassified_count": int(context_reclassified_count),
        "transition_edge_reclassified_count": int(
            atom_granularity_reclassified_count
        ),
        "transition_length_reclassified_count": 0,
        "normalization_merge_count": int(
            len(base_segments) - len(base) + len(carved) - len(normalized)
        ),
        "steady_parent_interval_count": int(steady_parent_count),
        "base_steady_candidate_interval_count": int(
            base_steady_candidate_count
        ),
        "strict_steady_platform_interval_count": int(steady_parent_count),
        "strict_platform_refinement_records": [
            dict(record) for record in point_carving_records
        ],
        "restored_outer_interval_count": int(restored_outer_interval_count),
        "restored_outer_point_count": int(restored_outer_point_count),
        "restored_outer_state_counts": dict(restored_outer_state_counts),
        "restored_outer_invalid_score_count": int(
            restored_outer_invalid_score_count
        ),
        "restored_outer_to_transition_count": int(
            restored_outer_to_transition_count
        ),
        "transition_trim_candidate_count": int(len(candidate_ranges)),
        "transition_carved_interval_count": int(len(registered)),
        "transition_carved_atom_count": int(len(transition_atoms)),
        "transition_carved_point_count": carved_point_count,
        "transition_carving_skipped_due_to_fallback": int(not allow_carving),
        # 兼容旧报告字段：固定比例裁边不再使用 MRR 边缘证据门控。
        "transition_generic_mrr_gate_rejection_count": 0,
        "transition_generic_mrr_gate_violation_count": 0,
        "transition_ratio_context_reclassified_count": int(
            context_reclassified_count
        ),
        "transition_ratio_boundary_reclassified_count": int(
            atom_granularity_reclassified_count
        ),
        "transition_ratio_core_reclassified_count": int(core_reclassified_count),
        "transition_ratio_short_transition_reclassified_count": 0,
        "transition_point_atom_granularity_reclassified_count": int(
            atom_granularity_reclassified_count
        ),
        "transition_point_core_short_mm_reclassified_count": int(
            short_core_mm_reclassified_count
        ),
        "transition_point_core_short_point_reclassified_count": int(
            short_core_point_reclassified_count
        ),
        "transition_point_core_score_reclassified_count": int(
            invalid_core_score_reclassified_count
        ),
        "transition_point_target_each_side_total": float(sum(
            float(record["target_each_side_point_count"])
            for record in point_carving_records
            if str(record.get("status")) == "carved"
        )),
        "transition_point_actual_each_side_total": int(sum(
            int(record["actual_each_side_point_count"])
            for record in point_carving_records
            if str(record.get("status")) == "carved"
        )),
        "transition_ratio_unit": "process_info_point",
        "transition_ratio_parent": "strict_steady_platform",
        "transition_semantics": "strict_platform_inner_5pct_v1",
        "transition_point_rounding_rule": "nearest_integer_half_up_with_minimum",
        "transition_point_carving_records": point_carving_records,
        "transition_outside_steady_parent_count": int(
            transition_outside_strict_platform_count
        ),
        "transition_outside_strict_platform_count": int(
            transition_outside_strict_platform_count
        ),
        "strict_platform_partition_violation_count": int(
            final_point_ratio_violations
        ),
        "provisional_nonsteady_to_transition_count": int(
            len(transition_atoms & provisional_nonsteady_atoms)
        ),
        "transition_without_steady_neighbor_count": (
            count_transition_without_steady_neighbor(normalized)
        ),
        "transition_edge_violation_count": int(final_edge_violations),
        # 旧字段名兼容保留；新语义校验的是点数，不再检查毫米长度。
        "transition_length_violation_count": int(final_point_count_violations),
        "transition_point_count_violation_count": int(
            final_point_count_violations
        ),
        "transition_point_ratio_violation_count": int(
            final_point_ratio_violations
        ),
    }
    return normalized, diagnostics


class SemiMarkovDecoder:
    """固定工艺语法、毫米持续长度和确定性回溯的 Semi-Markov 解码器。"""

    def __init__(self, config: SegmentationConfig):
        self.config = config
        predecessors: Dict[SegmentState, List[SegmentState]] = {state: [] for state in STATE_ORDER}
        for previous, current_states in ALLOWED_TRANSITIONS.items():
            for current in current_states:
                predecessors[current].append(previous)
        self._predecessors = {
            state: tuple(sorted(values, key=lambda item: STATE_INDEX[item]))
            for state, values in predecessors.items()
        }

    def _duration_adjustment(self, state: SegmentState, duration_mm: float) -> float | None:
        minimum, maximum = self.config.duration_bounds(state)
        tolerance = float(self.config.path_tolerance_mm)
        if duration_mm > maximum + tolerance:
            return None
        if minimum <= tolerance or duration_mm >= minimum - tolerance:
            return 0.0
        shortage = (minimum - max(duration_mm, 0.0)) / minimum
        return -float(self.config.short_duration_penalty) * float(shortage)

    def _is_better(
        self,
        candidate_score: float,
        candidate_key: tuple,
        best_score: float,
        best_key: tuple | None,
    ) -> bool:
        epsilon = float(self.config.tie_epsilon)
        if candidate_score > best_score + epsilon:
            return True
        if abs(candidate_score - best_score) <= epsilon:
            return best_key is None or candidate_key < best_key
        return False

    def _fallback_result(
        self,
        atoms: Sequence[AtomicSegment],
        scorer: SegmentScorer,
        fallback_state: SegmentState,
        reason: str,
    ) -> DecodeResult:
        """按原子段功率门控生成确定性的完整覆盖安全回退。"""

        atom_states: List[SegmentState] = []
        tolerance = max(float(self.config.tie_epsilon), 1e-12)
        has_power_gate = all(hasattr(atom, "idle_fraction") for atom in atoms)
        for atom_index, atom in enumerate(atoms):
            if has_power_gate:
                idle_fraction = float(getattr(atom, "idle_fraction"))
                if idle_fraction >= 1.0 - tolerance:
                    state = SegmentState.IDLE
                else:
                    # 回退不再猜测 entry/exit/transition。逐原子段优先保留
                    # 满足稳定硬判据的 steady，其余加工点保守归 nonsteady；
                    # 这样既不改变 idle 门控，也不会绕过评分器的硬约束。
                    atom_scores = scorer.score_all(atom_index, atom_index)
                    if np.isfinite(float(atom_scores[SegmentState.STEADY].score)):
                        state = SegmentState.STEADY
                    elif np.isfinite(float(atom_scores[SegmentState.NONSTEADY].score)):
                        state = SegmentState.NONSTEADY
                    else:
                        state = (
                            fallback_state
                            if fallback_state is not SegmentState.IDLE
                            else SegmentState.NONSTEADY
                        )
            else:
                state = fallback_state
            atom_states.append(state)

        chunks = [
            DecodedSegment(
                start_atom=atom_index,
                end_atom=atom_index,
                start_idx=atom.start_idx,
                end_idx=atom.end_idx,
                state=atom_states[atom_index],
            )
            for atom_index, atom in enumerate(atoms)
        ]
        segments = _merge_adjacent_segments(chunks)
        segment_scores = [
            float(scorer.score_segment(segment.start_atom, segment.end_atom, segment.state).score)
            for segment in segments
        ]
        total_score = float(sum(segment_scores)) if segment_scores else 0.0
        gate_note = "，已按功率门控逐原子段回退" if has_power_gate else ""
        return DecodeResult(
            segments=segments,
            total_score=total_score,
            used_fallback=True,
            failure_reason=f"{reason}{gate_note}",
        )

    def decode(
        self,
        atoms: Sequence[AtomicSegment],
        scorer: SegmentScorer,
        *,
        fallback_state: SegmentState = SegmentState.NONSTEADY,
    ) -> DecodeResult:
        atoms = tuple(atoms)
        if not atoms:
            return DecodeResult(segments=tuple(), total_score=0.0)

        atom_count = len(atoms)
        state_count = len(STATE_ORDER)
        scores = np.full((atom_count, state_count), -np.inf, dtype=float)
        back_start = np.full((atom_count, state_count), -1, dtype=np.int32)
        back_state = np.full((atom_count, state_count), -1, dtype=np.int8)
        initial_states = {SegmentState(value) for value in self.config.initial_states}

        max_atoms = int(self.config.max_segment_atoms)
        atom_starts = np.asarray([atom.start_s for atom in atoms], dtype=float)
        epsilon = float(self.config.tie_epsilon)
        tolerance = float(self.config.path_tolerance_mm)
        state_change_penalty = float(self.config.state_change_penalty)
        state_plans = []
        for state in STATE_ORDER:
            if state is SegmentState.TRANSITION:
                continue
            state_idx = STATE_INDEX[state]
            minimum, maximum = self.config.duration_bounds(state)
            chunk_limited = state in {
                SegmentState.IDLE,
                SegmentState.STEADY,
                SegmentState.NONSTEADY,
            }
            if chunk_limited:
                previous_states = tuple(
                    sorted(
                        {state, *self._predecessors[state]},
                        key=lambda item: STATE_INDEX[item],
                    )
                )
            else:
                previous_states = self._predecessors[state]
            previous_indices = np.asarray(
                [STATE_INDEX[value] for value in previous_states],
                dtype=int,
            )
            previous_penalties = np.asarray(
                [0.0 if previous is state else state_change_penalty for previous in previous_states],
                dtype=float,
            )
            state_plans.append(
                (
                    state,
                    state_idx,
                    float(minimum),
                    float(maximum),
                    chunk_limited,
                    previous_indices,
                    previous_penalties,
                    state in initial_states,
                )
            )
        for end_atom in range(atom_count):
            # max_segment_atoms 只限制允许同态计算分块的长状态。entry/exit
            # 不允许靠同态分块延长，必须按各自物理毫米上限完整回溯候选，
            # steady 则额外保留一组跨过最短毫米长度的候选；否则细化原子
            # 会错误改变业务边界或让合法 steady 完全无法形成。
            chunk_earliest = max(0, end_atom - max_atoms + 1)
            end_s = float(atoms[end_atom].end_s)
            start_groups = [
                np.arange(chunk_earliest, end_atom + 1, dtype=int)
            ]
            for bounded_state in (SegmentState.ENTRY, SegmentState.EXIT):
                _, maximum = self.config.duration_bounds(bounded_state)
                if np.isfinite(maximum):
                    first = int(np.searchsorted(
                        atom_starts,
                        end_s - maximum - tolerance,
                        side="left",
                    ))
                    if first <= end_atom:
                        start_groups.append(
                            np.arange(max(first, 0), end_atom + 1, dtype=int)
                        )
            steady_minimum, _ = self.config.duration_bounds(SegmentState.STEADY)
            steady_latest = int(np.searchsorted(
                atom_starts,
                end_s - steady_minimum + tolerance,
                side="right",
            )) - 1
            steady_window_start = end_atom + 1
            if steady_latest >= 0:
                steady_latest = min(steady_latest, end_atom)
                steady_window_start = max(0, steady_latest - max_atoms + 1)
                start_groups.append(
                    np.arange(
                        steady_window_start,
                        steady_latest + 1,
                        dtype=int,
                    )
                )
            starts = np.unique(np.concatenate(start_groups))
            durations = np.maximum(end_s - atom_starts[starts], 0.0)
            evidence_lengths = np.maximum(durations, max(tolerance, 1e-6))
            # 同一终点的六类候选共享一次向量化区间统计。
            local_values = scorer.score_candidates(starts, end_atom)
            for (
                state,
                state_idx,
                minimum,
                maximum,
                chunk_limited,
                previous_indices,
                previous_penalties,
                is_initial_state,
            ) in state_plans:
                duration_adjustments = np.zeros(len(starts), dtype=float)
                short = durations < minimum - tolerance
                if minimum > tolerance:
                    duration_adjustments[short] = (
                        -float(self.config.short_duration_penalty)
                        * (minimum - durations[short])
                        / minimum
                    )
                valid_duration = durations <= maximum + tolerance
                if state is SegmentState.STEADY:
                    steady_window = (
                        (starts >= steady_window_start)
                        & (starts <= steady_latest)
                    )
                    valid_duration &= (
                        (starts >= chunk_earliest) | steady_window
                    )
                elif chunk_limited:
                    valid_duration &= starts >= chunk_earliest
                if state is SegmentState.STEADY:
                    # steady 是 transition 的结构锚点，最短长度必须在 DP 中
                    # 直接成为硬约束，不能等回溯后再删除伪稳态。
                    valid_duration &= durations >= minimum - tolerance
                segment_scores = (
                    np.asarray(local_values[state], dtype=float) * evidence_lengths
                    + duration_adjustments
                )

                base_scores = np.full(len(starts), -np.inf, dtype=float)
                previous_choice = np.full(len(starts), -1, dtype=np.int8)
                if starts[0] == 0 and is_initial_state:
                    base_scores[0] = 0.0

                positive_offset = 1 if starts[0] == 0 else 0
                positive_count = len(starts) - positive_offset
                if positive_count > 0 and previous_indices.size:
                    previous_end_rows = starts[positive_offset:] - 1
                    previous_values = scores[
                        previous_end_rows[:, None],
                        previous_indices[None, :],
                    ].copy()
                    previous_values -= previous_penalties[None, :]
                    row_max = np.max(previous_values, axis=1)
                    chosen_column = np.full(positive_count, -1, dtype=int)
                    for column in range(previous_indices.size):
                        choose = (
                            (chosen_column < 0)
                            & np.isfinite(previous_values[:, column])
                            & (previous_values[:, column] >= row_max - epsilon)
                        )
                        chosen_column[choose] = column
                    usable = chosen_column >= 0
                    if np.any(usable):
                        target_positions = np.flatnonzero(usable) + positive_offset
                        chosen = chosen_column[usable]
                        base_scores[target_positions] = previous_values[usable, chosen]
                        previous_choice[target_positions] = previous_indices[chosen].astype(np.int8)

                candidate_scores = base_scores + segment_scores
                candidate_scores[~valid_duration] = -np.inf
                if not np.any(np.isfinite(candidate_scores)):
                    continue
                maximum_score = float(np.max(candidate_scores))
                tied = np.flatnonzero(
                    np.isfinite(candidate_scores)
                    & (candidate_scores >= maximum_score - epsilon)
                )
                # starts 升序：并列时固定优先更早起点。
                chosen_position = int(tied[0])
                scores[end_atom, state_idx] = float(candidate_scores[chosen_position])
                back_start[end_atom, state_idx] = int(starts[chosen_position])
                back_state[end_atom, state_idx] = int(previous_choice[chosen_position])

        terminal_states = [SegmentState(value) for value in self.config.terminal_states]
        final_state = None
        final_score = -np.inf
        final_key = None
        for state in terminal_states:
            candidate = float(scores[-1, STATE_INDEX[state]])
            key = (STATE_INDEX[state],)
            if np.isfinite(candidate) and self._is_better(candidate, key, final_score, final_key):
                final_score = candidate
                final_state = state
                final_key = key

        if final_state is None:
            return self._fallback_result(
                atoms,
                scorer,
                fallback_state,
                "无合法全行程最优划分结果，已保守全覆盖回退",
            )

        chunks: List[DecodedSegment] = []
        end_atom = atom_count - 1
        current_state = final_state
        while end_atom >= 0:
            state_idx = STATE_INDEX[current_state]
            start_atom = int(back_start[end_atom, state_idx])
            if start_atom < 0:
                return self._fallback_result(
                    atoms,
                    scorer,
                    fallback_state,
                    "回溯指针不完整，已保守全覆盖回退",
                )
            chunks.append(
                DecodedSegment(
                    start_atom=start_atom,
                    end_atom=end_atom,
                    start_idx=atoms[start_atom].start_idx,
                    end_idx=atoms[end_atom].end_idx,
                    state=current_state,
                )
            )
            previous_idx = int(back_state[end_atom, state_idx])
            end_atom = start_atom - 1
            if end_atom >= 0:
                if previous_idx < 0:
                    break
                current_state = STATE_ORDER[previous_idx]

        if end_atom >= 0:
            return self._fallback_result(
                atoms,
                scorer,
                fallback_state,
                "回溯未覆盖全部原子段，已保守全覆盖回退",
            )

        chunks.reverse()
        return DecodeResult(
            segments=_merge_adjacent_segments(chunks),
            total_score=float(final_score),
        )


def count_illegal_transitions(segments: Sequence[DecodedSegment]) -> int:
    count = 0
    for previous, current in zip(segments, segments[1:]):
        if previous.state is current.state:
            continue
        if not is_transition_allowed(previous.state, current.state):
            count += 1
    return int(count)
