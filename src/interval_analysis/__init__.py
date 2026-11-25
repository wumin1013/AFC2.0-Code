from __future__ import annotations

__all__ = [
    "MillingAnalysisTool",
    "PITEntry",
    "main",
    "optimize_memory",
    "_fast_startup",
    "_cleanup_on_exit",
]


def __getattr__(name):
    """Keep data/model commands independent of the desktop runtime."""
    if name in {"main", "optimize_memory", "_fast_startup", "_cleanup_on_exit"}:
        from importlib import import_module

        module = import_module(".main", __name__)
        for public in ("main", "optimize_memory", "_fast_startup", "_cleanup_on_exit"):
            globals()[public] = getattr(module, public)
        return globals()[name]
    if name == "PITEntry":
        from .shared import PITEntry

        globals()[name] = PITEntry
        return PITEntry
    if name == "MillingAnalysisTool":
        from .app import MillingAnalysisTool

        return MillingAnalysisTool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
