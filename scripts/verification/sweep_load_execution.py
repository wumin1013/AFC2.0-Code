"""Sequential, budgeted coarse screening. Every candidate is a fresh process."""
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
    script = Path(__file__).with_name("benchmark_load_execution.py")
    data = root / "data" / "subset.pt"
    reports = []

    def run(name, *, device="cpu", threads=4, batch=128, points=8192, cache="none", stage="both"):
        target = root / "runs" / name / "result.json"
        if not target.exists():
            command = [sys.executable, "-X", "utf8", str(script), "run", "--output-root", str(root),
                       "--name", name, "--data", str(data), "--device", device, "--threads", str(threads),
                       "--batch-windows", str(batch), "--main-points", str(points), "--cache", cache, "--stage", stage]
            code = subprocess.run(command, check=False).returncode
            if code != 0:
                print(json.dumps(dict(name=name, status="failed", exit_code=code)), flush=True)
                return None
        result = json.loads(target.read_text(encoding="utf-8"))
        reports.append(result)
        return result

    # Historical actual configuration comes first, with identical work thereafter.
    baseline = run("subset_cpu4_original")
    if baseline is None:
        raise RuntimeError("Cannot screen without a valid historical CPU baseline")
    original_cpu = [baseline]
    for threads in (1, 2, 8, 10):
        result = run(f"subset_cpu{threads}_original", threads=threads)
        if result:
            original_cpu.append(result)
    best_threads = min(original_cpu, key=lambda r: sum(r["medians"].values()))["args"]["threads"]
    finalists = {}
    for device in ("cpu", "cuda"):
        rows = []
        for batch in (32, 64, 128, 256):
            row = run(f"subset_{device}_cached_b{batch}", device=device, threads=best_threads if device=="cpu" else 4,
                      batch=batch, cache="cpu" if device=="cpu" else "device")
            if row:
                rows.append(row)
        if not rows:
            continue
        winner = min(rows, key=lambda r: r["medians"]["residual"])
        control = run(f"subset_{device}_original_b{winner['args']['batch_windows']}", device=device,
                      threads=best_threads if device=="cpu" else 4, batch=winner["args"]["batch_windows"])
        large = run(f"subset_{device}_main32768", device=device, threads=best_threads if device=="cpu" else 4,
                    points=32768, cache="cpu" if device=="cpu" else "device", stage="main")
        points = 32768 if large and large["medians"]["main"] < winner["medians"]["main"] else 8192
        finalists[device] = dict(device=device, threads=best_threads if device=="cpu" else 4,
                                batch_windows=winner["args"]["batch_windows"], main_points=points,
                                cache="cpu" if device=="cpu" else "device",
                                needs_full_f1_and_equivalence=True,
                                control_run=control["args"]["name"] if control else None)
    result = dict(status="coarse_screen_only", scientific_accuracy_verified=False,
                  finalists=finalists, measured_runs=[r["args"]["name"] for r in reports])
    (root / "coarse_screen.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
