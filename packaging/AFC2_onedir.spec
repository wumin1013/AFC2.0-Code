# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import os
import sys


PRODUCT_NAME = "AFC2.0.2alpha"
DIAGNOSTIC_CONSOLE = os.environ.get("AFC_PYINSTALLER_DIAGNOSTIC_CONSOLE") == "1"
# PyInstaller 提供的 SPECPATH 已是 spec 所在目录（packaging/）。
PROJECT_ROOT = Path(SPECPATH).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
ENTRY_POINT = SOURCE_ROOT / "project" / "release_main.py"

if not ENTRY_POINT.is_file():
    raise FileNotFoundError(f"发布入口不存在: {ENTRY_POINT}")

# 发布入口必须在导入图层面与研究应用隔离。这里的 excludes 是第二道保护，
# scripts/verification/verify_release_build.ps1 还会检查最终 CArchive/PYZ。
EXCLUDED_MODULES = [
    "project.app",
    "project.pit_model",
    "project.prediction_support",
    "project.release_prediction",
    "sklearn",
    "scipy",
    "joblib",
    "threadpoolctl",
]

# Conda 的 NumPy/Pandas 通过 mkl_rt 在运行时动态加载这些 DLL；
# PyInstaller 的静态依赖分析无法发现，缺失时 windowed EXE 会在显示界面前直接退出。
CONDA_LIBRARY_BIN = Path(sys.prefix) / "Library" / "bin"
MKL_RUNTIME_NAMES = [
    "mkl_core.2.dll",
    "mkl_intel_thread.2.dll",
    # 通用调度回退保证不同 x64 CPU 都能启动，不依赖本机构建时的 AVX 等级。
    "mkl_def.2.dll",
    "mkl_vml_def.2.dll",
    "libiomp5md.dll",
]
MKL_RUNTIME_BINARIES = []
for runtime_name in MKL_RUNTIME_NAMES:
    runtime_path = CONDA_LIBRARY_BIN / runtime_name
    if not runtime_path.is_file():
        raise FileNotFoundError(f"缺少 NumPy/Pandas 所需 MKL 运行库: {runtime_path}")
    MKL_RUNTIME_BINARIES.append((str(runtime_path), "."))

analysis = Analysis(
    [str(ENTRY_POINT)],
    pathex=[str(SOURCE_ROOT)],
    binaries=MKL_RUNTIME_BINARIES,
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
