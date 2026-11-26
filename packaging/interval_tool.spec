# -*- mode: python ; coding: utf-8 -*-
"""Windowless interval tool; version remains in release_metadata.py."""
from pathlib import Path
import ast
import sys

ROOT = Path(SPECPATH).resolve().parent
SOURCE = ROOT / "src"
tree = ast.parse((SOURCE / "interval_analysis/release_metadata.py").read_text(encoding="utf-8"))
version = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "RELEASE_VERSION" for target in node.targets))
runtime = Path(sys.prefix) / "Library/bin"
binaries = [(str(runtime / name), ".") for name in (
    "mkl_rt.2.dll", "mkl_core.2.dll", "mkl_intel_thread.2.dll",
    "mkl_def.2.dll", "mkl_vml_def.2.dll", "libiomp5md.dll",
) if (runtime / name).is_file()]
a = Analysis([str(SOURCE / "interval_analysis/interval_main.py")], pathex=[str(SOURCE)],
    binaries=binaries, datas=[], hiddenimports=[],
    excludes=["tkinter", "matplotlib", "torch", "scipy", "sklearn", "IPython", "jedi", "pytest", "pkg_resources"],
    noarchive=False, optimize=0)
# Build-machine package provenance and Python source are not runtime payload.
a.datas = [entry for entry in a.datas
           if Path(entry[0]).name != "direct_url.json" and Path(entry[0]).suffix.lower() != ".py"]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="区间划分工具", debug=False,
    bootloader_ignore_signals=False, strip=False, upx=False, console=False,
    disable_windowed_traceback=True, contents_directory="_internal")
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=version)
