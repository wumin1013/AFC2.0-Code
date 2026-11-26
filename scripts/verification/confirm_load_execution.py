"""Confirm coarse finalists on complete F1 without starting formal training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    root = Path(args.output_root).resolve()
    coarse = json.loads((root / "coarse_screen.json").read_text(encoding="utf-8"))
    script = Path(__file__).with_name("benchmark_load_execution.py")
    configs = [("full_cpu4_original", dict(device="cpu", threads=4, batch_windows=128, main_points=8192, cache="none"))]
    for device, candidate in coarse["finalists"].items():
        configs.extend([(f"full_{device}_optimized", candidate), (f"full_{device}_control", {**candidate, "cache":"none"})])
    status = []
    for name, config in configs:
        result_file = root / "runs" / name / "result.json"
        if not result_file.exists():
            command = [sys.executable, "-X", "utf8", str(script), "run", "--output-root", str(root),
                       "--name", name, "--data", str(root / "data" / "full.pt"), "--device", config["device"],
                       "--threads", str(config["threads"]), "--batch-windows", str(config["batch_windows"]),
                       "--main-points", str(config["main_points"]), "--cache", config["cache"]]
            code = subprocess.run(command, check=False).returncode
        else:
            code = 0
        status.append(dict(name=name, exit_code=code, completed=result_file.exists()))
        print(json.dumps(status[-1]), flush=True)
        if code != 0:
            raise RuntimeError(f"Full F1 confirmation failed: {name}; inspect without silent fallback")
    (root / "full_confirmation.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
