from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

if __package__ in (None, ""):
    _PACKAGE_ROOT = Path(__file__).resolve().parent.parent
    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))
    from interval_analysis.release_metadata import RELEASE_VERSION
else:
    from .release_metadata import RELEASE_VERSION


_RUNTIME_LOAD_RETRY_ATTEMPTS = 12
_RUNTIME_LOAD_RETRY_DELAY_SECONDS = 0.5


def _configure_frozen_tk_data_paths() -> None:
    """用 Unicode 路径显式定位冻结包内的 Tcl/Tk 数据目录。"""
    if not getattr(sys, "frozen", False):
        return
    internal_root = Path(
        getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent / "_internal")
    ).resolve()
    for variable_name, directory_name in (
        ("TCL_LIBRARY", "_tcl_data"),
        ("TK_LIBRARY", "_tk_data"),
    ):
        data_path = internal_root / directory_name
        if data_path.is_dir():
            os.environ[variable_name] = str(data_path)


def _load_runtime_components():
    """延迟加载发布运行组件，使导入阶段错误也能被主入口记录。"""
    if __package__ in (None, ""):
        package_root = Path(__file__).resolve().parent.parent
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
        from interval_analysis.main import (
            _cleanup_on_exit,
            _configure_accent_button_style,
            _fast_startup,
            optimize_memory,
        )
        from interval_analysis.release_app import AFCReleaseApplication
        from interval_analysis.shared import messagebox, tk, ttk
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
    content = f"{RELEASE_VERSION} 启动失败\n\n" + "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    )
    primary_log = (
        _startup_log_root() / "startup-error.log"
        if getattr(sys, "frozen", False)
        else _startup_log_root() / "output" / "startup-error.log"
    )
    candidates = (
        primary_log,
        Path(tempfile.gettempdir()) / f"{RELEASE_VERSION}-startup-error.log",
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
    detail = f"{RELEASE_VERSION} 无法启动：\n{exc}"
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

        ctypes.windll.user32.MessageBoxW(None, detail, f"{RELEASE_VERSION} 启动失败", 0x10)
    except Exception:
        pass


def main() -> int:
    components = {}
    root = None
    try:
        _configure_frozen_tk_data_paths()
        components = _load_runtime_components_with_retry()
        components["fast_startup"]()
        components["optimize_memory"]()
        root = components["tk"].Tk()
        root.withdraw()
        root.title(f"{RELEASE_VERSION} - 稳态区间划分")
        style = components["ttk"].Style()
        components["configure_style"](style)
        app = components["application"](root)
        if os.environ.get("AFC_RELEASE_SMOKE_TEST") == "1":
            # 冒烟模式只验证冻结依赖、Tk 和发布应用能完成构造。
            # update_idletasks() 会无界排空 TkAgg 的 idle 队列，特定
            # Windows/Tk/DPI 组合下可能陷入 Configure/draw_idle 反馈；
            # root.update() 还会误执行自动加载定时器，因此这里直接退出。
            return 0
        root.protocol("WM_DELETE_WINDOW", components["cleanup"])
        root.deiconify()
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
