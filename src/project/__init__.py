from __future__ import annotations

from .main import _cleanup_on_exit, _fast_startup, main, optimize_memory
from .shared import PITEntry

__all__ = [
    "MillingAnalysisTool",
    "PITEntry",
    "main",
    "optimize_memory",
    "_fast_startup",
    "_cleanup_on_exit",
]


def __getattr__(name):
    """延迟导入研究版主类，避免发布入口加载 PIT/SMIF 与 sklearn。"""
    if name == "MillingAnalysisTool":
        from .app import MillingAnalysisTool

        return MillingAnalysisTool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
