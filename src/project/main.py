from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from project.shared import *
else:
    from .shared import *


def optimize_memory():
    """优化内存使用和性能。"""
    gc.set_threshold(10000, 10, 10)


def _fast_startup():
    """快速启动优化。"""
    import warnings

    warnings.filterwarnings("ignore")
    plt.rcParams["figure.max_open_warning"] = 0
    plt.rcParams["agg.path.chunksize"] = 10000


def _cleanup_on_exit():
    """程序关闭时清理所有资源和后台进程。"""
    try:
        plt.close("all")
        gc.collect()
    except Exception:
        pass
    finally:
        os._exit(0)


def _configure_accent_button_style(style):
    style.configure(
        "Accent.TButton",
        font=("Arial", 10, "bold"),
        foreground="white",
        background=UI_COLOR_WARNING,
        borderwidth=1,
        relief="solid",
        bordercolor=UI_COLOR_WARNING,
        focuscolor=UI_COLOR_WARNING,
    )
    style.map(
        "Accent.TButton",
        background=[("active", "#FB923C"), ("pressed", "#F97316"), ("disabled", "#FDBA74")],
        foreground=[("active", "white"), ("pressed", "white"), ("disabled", "white")],
        bordercolor=[("!disabled", UI_COLOR_WARNING)],
    )


def main() -> int:
    if __package__ in (None, ""):
        from project.app import MillingAnalysisTool
    else:
        from .app import MillingAnalysisTool

    _fast_startup()
    optimize_memory()
    root = tk.Tk()
    root.title("🔬 铣削工艺信息分析工具 - 智能分析系统")
    root.withdraw()

    style = ttk.Style()
    _configure_accent_button_style(style)

    app = MillingAnalysisTool(root)
    root.protocol("WM_DELETE_WINDOW", _cleanup_on_exit)
    root.after(100, app.adjust_figure_sizes)
    root.deiconify()
    def _maximize_root():
        try:
            root.state("zoomed")
        except Exception:
            try:
                root.attributes("-zoomed", True)
            except Exception:
                pass
    root.after(0, _maximize_root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
