"""Verify a clean windowless candidate, ZIP and independent debug directory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import time
import zipfile

from PyInstaller.archive.readers import CArchiveReader


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_tool(exe, output, *, process=None, prediction=None, extra=()):
    args = [str(exe)]
    if process is not None:
        args += ["--process", str(process), "--prediction", str(prediction)]
    if output is not None:
        args += ["--output-dir", str(output)]
    started = time.monotonic()
    result = subprocess.run(args + list(extra), capture_output=True, timeout=60,
                            cwd=exe.parent, creationflags=subprocess.CREATE_NO_WINDOW)
    directory = output or exe.parent
    status = json.loads((directory / "interval_status.json").read_text(encoding="utf-8"))
    return {"exit_code": result.returncode, "seconds": time.monotonic() - started,
            "stdout_empty": not result.stdout, "stderr_empty": not result.stderr,
            "status": status}


def main():
    parser = argparse.ArgumentParser()
    for name in ("candidate", "output", "process", "prediction", "readme"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    candidate, output = args.candidate.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    exe = candidate / "区间划分工具.exe"
    binary = exe.read_bytes()
    pe = struct.unpack_from("<I", binary, 0x3c)[0]
    subsystem = struct.unpack_from("<H", binary, pe + 24 + 68)[0]
    assert subsystem == 2, "PE must use WINDOWS_GUI subsystem, without a console"
    reader = CArchiveReader(str(exe))
    pyz_name = next(name for name in reader.toc if name.endswith(".pyz"))
    modules = reader.open_embedded_archive(pyz_name).toc
    forbidden = [name for name in modules if name.split(".")[0] in ("tkinter", "torch", "matplotlib", "scipy", "sklearn")]
    assert not forbidden, forbidden
    assert "interval_analysis.interval_tool" in modules
    original_inputs = {str(p.resolve()): sha(p) for p in (args.process, args.prediction)}
    runs = {"candidate": run_tool(exe, output / "candidate_run", process=args.process, prediction=args.prediction)}
    assert runs["candidate"]["exit_code"] == 0
    expected = runs["candidate"]["status"]["outputs"]
    # Candidate payload contains no debug inputs, outputs, source or machine configuration.
    assert not any(p.suffix.lower() in (".csv", ".nc", ".pt", ".py") for p in candidate.rglob("*") if p.is_file())
    assert not any(p.name in ("prediction.json", "interval_status.json", "machining_context.json", "direct_url.json")
                   for p in candidate.rglob("*"))
    shutil.copyfile(args.readme, candidate / "README.md")
    files = {p.relative_to(candidate).as_posix(): sha(p) for p in candidate.rglob("*")
             if p.is_file() and p.name not in ("SHA256SUMS.txt", "build_manifest.json")}
    manifest = {"schema": "afc.interval_release.1", "release": candidate.name,
                "entry": exe.name, "requires_internal_directory": True, "console": False,
                "model_runtime_included": False, "gui_toolkit_included": False,
                "integration_status": "local_candidate_not_deployed_to_AFO", "files": files}
    (candidate / "build_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    files["build_manifest.json"] = sha(candidate / "build_manifest.json")
    (candidate / "SHA256SUMS.txt").write_text("\n".join(f"{value}  {name}" for name, value in sorted(files.items())) + "\n", encoding="utf-8")
    archive = output / (candidate.name + ".zip")
    if archive.exists():
        raise FileExistsError(f"Preserve the previous ZIP: {archive}")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in sorted(candidate.rglob("*")):
            if path.is_file():
                zipped.write(path, path.relative_to(candidate.parent))
    extracted = output / "zip_readback"
    if extracted.exists():
        raise FileExistsError(extracted)
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        zipped.extractall(extracted)
    readback = extracted / candidate.name
    assert all(sha(readback / name) == value for name, value in files.items())
    runs["zip_readback"] = run_tool(readback / exe.name, output / "zip_readback_run",
                                   process=args.process, prediction=args.prediction)
    debug = output / (candidate.name.replace("beta", "Beta") + "-debug")
    if debug.exists():
        raise FileExistsError(debug)
    shutil.copytree(candidate, debug)
    shutil.copyfile(args.process, debug / "original_process.txt")
    shutil.copyfile(args.prediction, debug / "prediction.json")
    runs["debug_auto_import"] = run_tool(debug / exe.name, None)
    for name, run in runs.items():
        assert run["exit_code"] == 0 and run["status"]["status"] == "completed", name
        assert run["stdout_empty"] and run["stderr_empty"], name
        assert run["status"]["outputs"] == expected, name
    negative = run_tool(readback / exe.name, output / "invalid_cli", extra=("--unknown-flag",))
    assert negative["exit_code"] == 2 and negative["status"]["status"] == "failed"
    assert negative["stdout_empty"] and negative["stderr_empty"]
    assert original_inputs == {path: sha(path) for path in original_inputs}
    receipt = {"status": "passed", "pe_subsystem": subsystem, "no_gui_or_model_modules": True,
               "original_inputs_unchanged": True, "output_hashes_identical": expected,
               "runs": {name: {key: value for key, value in run.items() if key != "status"} for name, run in runs.items()},
               "invalid_cli_checked": True, "archive": str(archive), "archive_sha256": sha(archive),
               "archive_bytes": archive.stat().st_size, "exe_sha256": sha(exe), "debug_directory": str(debug)}
    (output / "delivery_verification.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
