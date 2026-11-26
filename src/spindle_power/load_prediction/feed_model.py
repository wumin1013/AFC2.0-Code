"""Lazy public API and shared sequence windows for actual-feed prediction."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class FeedModelConfig:
    hidden_size: int = 32
    num_layers: int = 2
    dropout: float = 0.
    window_points: int = 256
    context_points: int = 64
    schema: str = "afc.feed_model.2"

    def validate(self):
        for name in ("hidden_size", "num_layers", "window_points", "context_points"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "context_points" else 1):
                raise ValueError(f"{name} 必须为有效整数")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout 必须在 [0, 1) 内")
        if self.schema != "afc.feed_model.2":
            raise ValueError("不支持的实际进给模型版本")


@dataclass(frozen=True)
class FeedWindow:
    start: int
    stop: int
    core_start: int
    core_stop: int


def sequence_windows(frame, config, valid_mask=None):
    """Common window rule; supervised gaps may further censor training input.

    Inference never inspects measured labels. Both paths retain the same
    NC-known G0/contact/execution boundaries, then apply identical context.
    """
    config.validate()
    if "sequence_id" not in frame or frame.sequence_id.isna().any():
        raise ValueError("实际进给输入缺少有效 sequence_id")
    from .sample_rules import continuous_ranges
    for first, last in continuous_ranges(frame, valid_mask, known_only=True):
        for core_start in range(first, last, config.window_points):
            core_stop = min(last, core_start + config.window_points)
            yield FeedWindow(max(first, core_start - config.context_points),
                             min(last, core_stop + config.context_points), core_start, core_stop)


def _runtime():
    import sys
    if sys.platform == "win32":
        from threadpoolctl import threadpool_info
        if any(item.get("internal_api") == "mkl" and item.get("threading_layer") == "intel"
               for item in threadpool_info()):
            raise RuntimeError("当前进程已加载Intel线程版MKL。请设置 MKL_THREADING_LAYER=SEQUENTIAL 后重启，再加载进给模型。")
    try:
        from . import feed_neural
        return feed_neural
    except (ImportError, OSError, AttributeError) as exc:
        raise RuntimeError("实际进给模型需要可用的 CPU PyTorch，请检查 AFC 环境。") from exc


def build_feed_model(config=None):
    config = config or FeedModelConfig()
    config.validate()
    return _runtime().FeedModel(config)


def load_feed_model(path):
    return _runtime().FeedModel.load(path)


def predict_actual_feed(model, features):
    """Return a FeedSample; never consume measured labels or invent time."""
    return _runtime().predict_actual_feed(model, features)
