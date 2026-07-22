"""基于工艺信息的全行程六类 Semi-Markov 划分稳定接口。"""

from .pipeline import INTERVAL_COLUMNS, POINT_LABEL_COLUMNS, SegmentationPipeline
from .schemas import (
    ALLOWED_TRANSITIONS,
    INPUT_SCHEMA_VERSION,
    SEGMENT_TYPES,
    STATE_CODE_BY_TYPE,
    TYPE_BY_STATE_CODE,
    AtomicSegment,
    DecodeResult,
    DecodedSegment,
    PathDiagnostics,
    SegmentScore,
    SegmentState,
    SegmentationConfig,
    SegmentationResult,
    is_transition_allowed,
    state_code,
)
from .scorers import RuleSegmentScorer, SegmentScorer
from .semi_markov import SemiMarkovDecoder

__all__ = [
    "ALLOWED_TRANSITIONS",
    "AtomicSegment",
    "DecodeResult",
    "DecodedSegment",
    "INPUT_SCHEMA_VERSION",
    "INTERVAL_COLUMNS",
    "POINT_LABEL_COLUMNS",
    "PathDiagnostics",
    "RuleSegmentScorer",
    "SEGMENT_TYPES",
    "STATE_CODE_BY_TYPE",
    "SegmentScore",
    "SegmentScorer",
    "SegmentState",
    "SegmentationConfig",
    "SegmentationPipeline",
    "SegmentationResult",
    "SemiMarkovDecoder",
    "TYPE_BY_STATE_CODE",
    "is_transition_allowed",
    "state_code",
]
