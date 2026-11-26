"""Package the current shared source and an explicit engineering trial only.

No training, uploads, credentials, cloud purchases or formal-ready release are
performed here. All package members are relative and independently hashed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manual", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    trial = args.trial_dir.resolve(strict=True)
    if not any(trial.glob("*.json")):
        parser.error("trial-dir must contain the prepared trial manifest")
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        parser.error("refusing to overwrite an existing cloud trial bundle")
    files = [(p, p.relative_to(root).as_posix()) for p in sorted((root / "src").rglob("*.py"))]
    files += [(p, p.relative_to(root).as_posix()) for p in (
        root / "scripts/verification/cloud_load_trial.py",
        root / "scripts/verification/build_cloud_trial_bundle.py",
        root / "scripts/verification/verify_cloud_trial_bundle.py",
        root / "scripts/verify_unified_load_prediction.py",
        root / "requirements-cloud.txt",
    )]
    for name in ("test_load_gpu_route.py", "test_nominal_path_features.py", "test_artifact_paths.py",
                 "test_portable_derivation.py", "test_hybrid_estimated_labels.py",
                 "test_record_idle_labels.py", "test_base_timeline.py"):
        path = root / "tests" / name
        if path.is_file():
            files.append((path, "tests/" + name))
    files += [(p, "trial/" + p.relative_to(trial).as_posix()) for p in sorted(trial.rglob("*"))
              if p.is_file() and "__pycache__" not in p.parts]
    if args.manual:
        files.append((args.manual.resolve(strict=True), "云端GPU试跑说明.md"))
    if any(p.is_symlink() for p, _ in files):
        raise ValueError("bundle input may not contain symlinks")
    entries = [{"path": name, "sha256": sha256(path), "bytes": path.stat().st_size} for path, name in files]
    if len({entry["path"] for entry in entries}) != len(entries):
        raise ValueError("duplicate archive member")
    manifest = dict(schema="afc.cloud_trial_bundle.1", purpose="engineering_smoke_and_timing_only",
                    formal_training_ready=False, entries=entries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation leaves an incomplete archive visible on failure, never
    # silently claiming that a partially built package is ready.
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, name in files:
            archive.write(path, name)
        archive.writestr("bundle_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(args.output) as archive:
        for item in entries:
            content = archive.read(item["path"])
            if len(content) != item["bytes"] or hashlib.sha256(content).hexdigest() != item["sha256"]:
                raise ValueError("cloud bundle verification failed: " + item["path"])
    manifest.update(archive_sha256=sha256(args.output), archive_bytes=args.output.stat().st_size)
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"archive": str(args.output.resolve()), "sha256": manifest["archive_sha256"],
                      "bytes": manifest["archive_bytes"], "files": len(entries)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
