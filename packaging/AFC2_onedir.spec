# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import ast
import os
import sys


DIAGNOSTIC_CONSOLE = os.environ.get("AFC_PYINSTALLER_DIAGNOSTIC_CONSOLE") == "1"
# PyInstaller 提供的 SPECPATH 已是 spec 所在目录（packaging/）。
PROJECT_ROOT = Path(SPECPATH).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
ENTRY_POINT = SOURCE_ROOT / "interval_analysis" / "release_main.py"
VERSION_SOURCE = SOURCE_ROOT / "interval_analysis" / "release_metadata.py"


def read_release_version(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "RELEASE_VERSION" for target in statement.targets):
            continue
        value = ast.literal_eval(statement.value)
        if isinstance(value, str) and value:
            return value
    raise ValueError(f"未在版本文件中找到有效 RELEASE_VERSION: {path}")


PRODUCT_NAME = read_release_version(VERSION_SOURCE)

if not ENTRY_POINT.is_file():
    raise FileNotFoundError(f"发布入口不存在: {ENTRY_POINT}")

# 发布入口必须在导入图层面与研究应用隔离。这里的 excludes 是第二道保护，
# scripts/verification/verify_release_build.ps1 还会检查最终 CArchive/PYZ。
EXCLUDED_MODULES = [
    "interval_analysis.app",
    "interval_analysis.pit_model",
    "sklearn",
    "scipy",
    "joblib",
    "threadpoolctl",
    "pkg_resources",
    # Tk 发布应用不使用交互式 Python 补全；排除其测试和源码数据载荷。
    "IPython",
    "jedi",
]

# Conda 的 NumPy/Pandas、ctypes 与 Pillow 会在运行时动态加载这些 DLL；
# 仅显式指定 Python 路径而未激活 Conda 时，PyInstaller 无法从 PATH
# 解析它们，缺失会使 windowed EXE 在显示界面前直接退出。
CONDA_LIBRARY_BIN = Path(sys.prefix) / "Library" / "bin"
CONDA_RUNTIME_NAMES = [
    "mkl_rt.2.dll",
    "mkl_core.2.dll",
    "mkl_intel_thread.2.dll",
    # 通用调度回退保证不同 x64 CPU 都能启动，不依赖本机构建时的 AVX 等级。
    "mkl_def.2.dll",
    "mkl_vml_def.2.dll",
    "libiomp5md.dll",
    "ffi.dll",
    "libsharpyuv.dll",
    "libwebp.dll",
    "libwebpdemux.dll",
    "libwebpmux.dll",
    "tcl86t.dll",
    "tk86t.dll",
]
CONDA_RUNTIME_BINARIES = []
for runtime_name in CONDA_RUNTIME_NAMES:
    runtime_path = CONDA_LIBRARY_BIN / runtime_name
    if not runtime_path.is_file():
        raise FileNotFoundError(f"缺少发布版所需 Conda 运行库: {runtime_path}")
    CONDA_RUNTIME_BINARIES.append((str(runtime_path), "."))

analysis = Analysis(
    [str(ENTRY_POINT)],
    pathex=[str(SOURCE_ROOT)],
    binaries=CONDA_RUNTIME_BINARIES,
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDED_MODULES,
    noarchive=False,
    optimize=0,
)

python_archive = PYZ(analysis.pure)

executable = EXE(
    python_archive,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name=PRODUCT_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=DIAGNOSTIC_CONSOLE,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    contents_directory="_internal",
)

distribution = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=PRODUCT_NAME,
)
