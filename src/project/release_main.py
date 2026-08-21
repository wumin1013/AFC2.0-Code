from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback
from pathlib import Path


_RUNTIME_LOAD_RETRY_ATTEMPTS = 12
_RUNTIME_LOAD_RETRY_DELAY_SECONDS = 0.5


def _load_runtime_components():
    """延迟加载发布运行组件，使导入阶段错误也能被主入口记录。"""
    if __package__ in (None, ""):
        package_root = Path(__file__).resolve().parent.parent
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
        from project.main import (
            _cleanup_on_exit,
            _configure_accent_button_style,
            _fast_startup,
            optimize_memory,
        )
        from project.release_app import AFCReleaseApplication
        from project.shared import messagebox, tk, ttk
    else:
        from .main import _cleanup_on_exit, _configure_accent_button_style, _fast_startup, optimize_memory
        from .release_app import AFCReleaseApplication
        from .shared import messagebox, tk, ttk
    return {
        "application": AFCReleaseApplication,
        "cleanup": _cleanup_on_exit,
        "configure_style": _configure_accent_button_style,
        "fast_startup": _fast_startup,
        "messagebox": messagebox,
        "optimize_memory": optimize_memory,
        "tk": tk,
        "ttk": ttk,
    }


def _load_runtime_components_with_retry():
    """冻结发布版遇到同步软件的瞬时文件锁时短暂重试。"""
    attempts = _RUNTIME_LOAD_RETRY_ATTEMPTS if getattr(sys, "frozen", False) else 1
    for attempt in range(1, attempts + 1):
        try:
            return _load_runtime_components()
        except PermissionError:
            if attempt >= attempts:
                raise
            time.sleep(_RUNTIME_LOAD_RETRY_DELAY_SECONDS)
    raise RuntimeError("发布运行组件加载重试意外结束")


def _startup_log_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def _write_startup_error(exc: BaseException) -> str:
    """尽最大努力记录 windowed EXE 启动错误，并返回实际日志路径。"""
    content = "AFC2.0.2alpha 启动失败\n\n" + "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    )
    primary_log = (
        _startup_log_root() / "startup-error.log"
        if getattr(sys, "frozen", False)
        else _startup_log_root() / "output" / "startup-error.log"
    )
    candidates = (
        primary_log,
        Path(tempfile.gettempdir()) / "AFC2.0.2alpha-startup-error.log",
    )
    for path in candidates:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return str(path)
        except Exception:
            continue
    return ""


def _show_startup_error(components, exc: BaseException, log_path: str) -> None:
    if os.environ.get("SUPPRESS_MESSAGEBOXES") == "1":
        return
    detail = f"AFC2.0.2alpha 无法启动：\n{exc}"
    if log_path:
        detail += f"\n\n错误日志：\n{log_path}"
    messagebox = components.get("messagebox") if isinstance(components, dict) else None
    if messagebox is not None:
        try:
            messagebox.showerror("启动失败", detail)
            return
        except Exception:
            pass
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, detail, "AFC2.0.2alpha 启动失败", 0x10)
    except Exception:
        pass


def main() -> int:
    components = {}
    root = None
    try:
        components = _load_runtime_components_with_retry()
        components["fast_startup"]()
        components["optimize_memory"]()
        root = components["tk"].Tk()
        root.withdraw()
        root.title("AFC2.0.2alpha - 稳态区间划分")
        style = components["ttk"].Style()
        components["configure_style"](style)
        app = components["application"](root)
        if os.environ.get("AFC_RELEASE_SMOKE_TEST") == "1":
            root.update_idletasks()
            return 0
        root.protocol("WM_DELETE_WINDOW", components["cleanup"])
        root.deiconify()
        try:
            root.state("zoomed")
        except Exception:
            pass
        root.mainloop()
        return 0
    except Exception as exc:
        log_path = _write_startup_error(exc)
        _show_startup_error(components, exc, log_path)
        return 1
    finally:
        if root is not None and os.environ.get("AFC_RELEASE_SMOKE_TEST") == "1":
            try:
                root.destroy()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
