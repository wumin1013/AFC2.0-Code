"""Verify an extracted engineering bundle before installing or running it."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def verify(root):
    root = Path(root).resolve(strict=True)
    manifest = json.loads((root / "bundle_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "afc.cloud_trial_bundle.1" or manifest.get("formal_training_ready") is not False:
        raise ValueError("Expected an engineering-only cloud trial manifest")
    seen = set()
    for entry in manifest["entries"]:
        name = entry["path"]
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name or name in seen:
            raise ValueError("Unsafe or duplicate member: " + name)
        seen.add(name)
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Bundle member escaped root: " + name)
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        if path.stat().st_size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
            raise ValueError("Bundle member changed: " + name)
    return {"status": "bundle_bytes_verified", "files": len(seen), "formal_training_ready": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(verify(args.root)))
