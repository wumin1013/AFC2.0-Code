from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple


INPUT_SCHEMA_VERSION = "process-info-mrr-v3"
CONFIG_SCHEMA_VERSION = "segmentation-config-mrr-v9"
RULE_SCORER_VERSION = "rule-mrr-v9"


class SegmentState(str, Enum):
    """全行程六类状态。"""

    IDLE = "idle"
    ENTRY = "entry"
    STEADY = "steady"
    TRANSITION = "transition"
    NONSTEADY = "nonsteady"
    EXIT = "exit"


SEGMENT_TYPES: Tuple[str, ...] = tuple(state.value for state in SegmentState)
STATE_CODE_BY_TYPE: Dict[str, int] = {
    SegmentState.IDLE.value: 0,
    SegmentState.ENTRY.value: 1,
    SegmentState.STEADY.value: 2,
    SegmentState.TRANSITION.value: 3,
    SegmentState.NONSTEADY.value: 4,
    SegmentState.EXIT.value: 5,
}
TYPE_BY_STATE_CODE: Dict[int, str] = {code: name for name, code in STATE_CODE_BY_TYPE.items()}

# 只描述最终连续区间之间的跨状态转移。解码器内部允许同态分块，
# 但回溯后会合并，因此不会在输出中形成额外的自转移。
ALLOWED_TRANSITIONS: Dict[SegmentState, Tuple[SegmentState, ...]] = {
    # entry / exit 可以为空。这里保留 idle 与 transition 的结构兼容边，
    # 供旧结果或自定义评分器接受统一复核；v7 内置后处理会先恢复 B\P
    # 外缘，因此正常生成的 transition 仍只位于严格平台 P 内。transition
    # 的稳态邻接由解码结果校验负责，因为“左右至少一侧为 steady”无法
    # 只用一阶转移表表达。
    SegmentState.IDLE: (
        SegmentState.ENTRY,
        SegmentState.TRANSITION,
        SegmentState.STEADY,
        SegmentState.NONSTEADY,
    ),
    SegmentState.ENTRY: (
        SegmentState.TRANSITION,
        SegmentState.STEADY,
        SegmentState.NONSTEADY,
        SegmentState.EXIT,
        SegmentState.IDLE,
    ),
    SegmentState.STEADY: (
        SegmentState.TRANSITION,
        SegmentState.NONSTEADY,
        SegmentState.EXIT,
        SegmentState.IDLE,
    ),
    SegmentState.TRANSITION: (
        SegmentState.STEADY,
        SegmentState.NONSTEADY,
        SegmentState.EXIT,
        SegmentState.IDLE,
    ),
    SegmentState.NONSTEADY: (
        SegmentState.TRANSITION,
        SegmentState.STEADY,
        SegmentState.EXIT,
        SegmentState.IDLE,
    ),
    SegmentState.EXIT: (SegmentState.IDLE,),
}


def state_code(segment_type: str | SegmentState) -> int:
    value = segment_type.value if isinstance(segment_type, SegmentState) else str(segment_type)
    return STATE_CODE_BY_TYPE[value]


def is_transition_allowed(previous: str | SegmentState, current: str | SegmentState) -> bool:
    previous_state = previous if isinstance(previous, SegmentState) else SegmentState(str(previous))
    current_state = current if isinstance(current, SegmentState) else SegmentState(str(current))
    return current_state in ALLOWED_TRANSITIONS[previous_state]


@dataclass(frozen=True)
class SegmentationConfig:
    """初步规则划分的集中、可序列化配置。"""

    config_schema_version: str = CONFIG_SCHEMA_VERSION
    input_schema_version: str = INPUT_SCHEMA_VERSION

    path_tolerance_mm: float = 1e-8
    sequential_fallback_step_mm: float = 1.0
    mrr_cutting_epsilon: float = 1e-9
    # 仅为旧配置兼容保留；六态分类只使用工艺表重算 MRR，
    # 该字段不再参与 idle 判定或过程域签名。
    idle_power_tolerance: float = 1e-9
    steady_mrr_relative_std_max: float = 0.08
    # 规则基线只接受真正的平台：候选区间的 MRR 线性拟合总漂移
    # 相对于均值不得超过该比例。后续学习型 scorer 仍须遵守此安全门控。
    steady_mrr_relative_slope_max: float = 0.02
    # 防止长趋势中的少量离散点或短暂平肩被当成完整稳态平台。
    steady_min_plateau_points: int = 24

    local_window_points: int = 9
    # entry 的峰值候选为第一个 MRR 局部峰值。窗口只用于确认峰后
    # 短期内不再上升，容差用于合并数值上等高的连续峰顶。最终进刀终点还需
    # 与首个合法稳态起点比较，且稳态优先。
    entry_peak_window_points: int = 3
    entry_peak_relative_tolerance: float = 1e-6

    relative_change_threshold: float = 0.08
    mrr_change_abs_mm3_min: float = 1e-6
    mrr_trend_relative_per_point: float = 0.03

    # idle 硬标签与“重置进/退刀阶段”分离：短于该物理长度的
    # 内部 idle 脉冲仍是 idle，但不会开启新的 entry/exit 阶段。
    min_idle_reset_mm: float = 0.10
    # 仅为旧配置兼容字段；当前每个有效工艺点固定对应一个原子。
    max_atomic_length_mm: float = 1.0

    min_idle_mm: float = 0.0
    min_entry_mm: float = 0.10
    min_steady_mm: float = 0.50
    min_transition_mm: float = 0.10
    min_nonsteady_mm: float = 0.0
    min_exit_mm: float = 0.10
    max_idle_mm: float = 1.0e12
    max_entry_mm: float = 40.0
    max_steady_mm: float = 1.0e12
    # 先确定严格 steady 平台 P，再按 P 的 ProcessInfo 点数在其内部
    # 两端各裁该比例作为 transition。该比例不得用于向 P 外扩张，
    # 也不按物理行程或实际负载采样点数计算。
    transition_ratio: float = 0.05
    min_transition_points: int = 1
    # 仅为旧配置兼容字段，点数比例裁边不再使用物理长度上限。
    max_transition_mm: float = 10.0
    max_nonsteady_mm: float = 1.0e12
    max_exit_mm: float = 40.0
    short_duration_penalty: float = 2.0
    # 仅是 DP 候选窗口的计算上限；持续长度约束始终使用毫米。
    max_segment_atoms: int = 32

    idle_match_weight: float = 7.0
    cutting_match_weight: float = 5.0
    stability_weight: float = 4.0
    variation_weight: float = 4.0
    boundary_context_weight: float = 3.0
    trend_weight: float = 2.0
    state_change_penalty: float = 0.20

    confidence_medium_margin: float = 0.75
    confidence_high_margin: float = 2.0
    tie_epsilon: float = 1e-10

    # entry 必须直接位于 idle 之后，因此不能作为全行程初态；exit 必须
    # 直接连接后续 idle，因此不能作为全行程终态。
    initial_states: Tuple[str, ...] = (
        "idle",
        "transition",
        "steady",
        "nonsteady",
    )
    terminal_states: Tuple[str, ...] = (
        "idle",
        "entry",
        "steady",
        "transition",
        "nonsteady",
    )

    def __post_init__(self) -> None:
        nonnegative = (
            "path_tolerance_mm",
            "mrr_cutting_epsilon",
            "idle_power_tolerance",
            "steady_mrr_relative_std_max",
            "steady_mrr_relative_slope_max",
            "entry_peak_relative_tolerance",
            "relative_change_threshold",
            "mrr_change_abs_mm3_min",
            "mrr_trend_relative_per_point",
            "min_idle_reset_mm",
            "min_idle_mm",
            "min_entry_mm",
            "min_steady_mm",
            "min_transition_mm",
            "min_nonsteady_mm",
            "min_exit_mm",
            "max_idle_mm",
            "max_entry_mm",
            "max_steady_mm",
            "max_transition_mm",
            "max_nonsteady_mm",
            "max_exit_mm",
            "short_duration_penalty",
            "idle_match_weight",
            "cutting_match_weight",
            "stability_weight",
            "variation_weight",
            "boundary_context_weight",
            "trend_weight",
            "state_change_penalty",
            "confidence_medium_margin",
            "confidence_high_margin",
            "tie_epsilon",
        )
        for name in nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} 必须是有限非负数")
        fallback_step = float(self.sequential_fallback_step_mm)
        if not math.isfinite(fallback_step) or fallback_step <= 0.0:
            raise ValueError("sequential_fallback_step_mm 必须大于 0")
        atomic_length = float(self.max_atomic_length_mm)
        if not math.isfinite(atomic_length) or atomic_length <= 0.0:
            raise ValueError("max_atomic_length_mm 必须是有限正数")
        if int(self.max_segment_atoms) < 1:
            raise ValueError("max_segment_atoms 必须大于 0")
        if int(self.local_window_points) < 1:
            raise ValueError("local_window_points 必须大于 0")
        entry_peak_window_points = int(self.entry_peak_window_points)
        if (
            entry_peak_window_points < 1
            or entry_peak_window_points != self.entry_peak_window_points
        ):
            raise ValueError("entry_peak_window_points 必须是大于等于 1 的整数")
        plateau_points = int(self.steady_min_plateau_points)
        if plateau_points < 2 or plateau_points != self.steady_min_plateau_points:
            raise ValueError("steady_min_plateau_points 必须是大于等于 2 的整数")
        minimum_transition_points = int(self.min_transition_points)
        if (
            minimum_transition_points < 1
            or minimum_transition_points != self.min_transition_points
        ):
            raise ValueError("min_transition_points 必须是大于等于 1 的整数")
        transition_ratio = float(self.transition_ratio)
        if not math.isfinite(transition_ratio) or not 0.0 < transition_ratio < 0.5:
            raise ValueError("transition_ratio 必须位于 (0, 0.5) 区间")
        for state in SegmentState:
            minimum, maximum = self.duration_bounds(state)
            if minimum > maximum:
                raise ValueError(f"{state.value} 的最短持续长度不得大于最长持续长度")
        if float(self.confidence_medium_margin) > float(self.confidence_high_margin):
            raise ValueError("confidence_medium_margin 不得大于 confidence_high_margin")
        for value in self.initial_states + self.terminal_states:
            if value not in STATE_CODE_BY_TYPE:
                raise ValueError(f"未知六类状态: {value}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def duration_bounds(self, state: SegmentState) -> Tuple[float, float]:
        return (
            float(getattr(self, f"min_{state.value}_mm")),
            float(getattr(self, f"max_{state.value}_mm")),
        )


@dataclass(frozen=True)
class PathDiagnostics:
    source: str
    is_valid: bool
    is_physical: bool
    used_nonphysical_fallback: bool
    span_mm: float
    input_cumulative_valid: bool = False
    input_incremental_valid: bool = False
    input_bounds_valid: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AtomicSegment:
    atom_id: int
    start_idx: int
    end_idx: int
    start_s: float
    end_s: float
    length_mm: float
    start_line_id: int
    end_line_id: int
    point_count: int
    cutting_fraction: float
    mrr_mean: float
    mrr_std: float
    mrr_trend_sign: int
    idle_fraction: float = 0.0
    non_idle_fraction: float = 1.0
    machining_segment_id: int = 0


@dataclass(frozen=True)
class SegmentScore:
    score: float
    reason: str


@dataclass(frozen=True)
class DecodedSegment:
    start_atom: int
    end_atom: int
    start_idx: int
    end_idx: int
    state: SegmentState


@dataclass(frozen=True)
class DecodeResult:
    segments: Tuple[DecodedSegment, ...]
    total_score: float
    used_fallback: bool = False
    failure_reason: str = ""
    fallback_scope: str = "none"
    fallback_validated: bool = True
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentationResult:
    """全行程六类划分结果；表属性可直接导出。"""

    point_labels: Any
    intervals: Any
    diagnostics: Dict[str, Any]
    config: SegmentationConfig
    input_schema_version: str = INPUT_SCHEMA_VERSION
    scorer_type: str = RULE_SCORER_VERSION
    model_version: Optional[str] = None
    atomic_segments: Tuple[AtomicSegment, ...] = field(default_factory=tuple)

    def config_dict(self) -> Dict[str, Any]:
        return self.config.to_dict()


def normalize_metadata(metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """将可选元数据复制为普通字典，避免结果持有可变外部引用。"""

    return dict(metadata or {})
