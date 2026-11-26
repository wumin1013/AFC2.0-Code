"""Bounded F1 execution benchmark; never launches or registers a research model."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
BUDGET_SECONDS = 90 * 60


def peak_memory_bytes():
    """Read the OS working-set peak without adding a benchmark dependency."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("peak", "working", "peak_paged", "paged",
                    "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        api = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        api.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not api(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.peak
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def bounded_child(args):
    """Charge preparation and every subprocess, including failures, across restarts."""
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "budget.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    ledger_path = root / "budget.json"
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else dict(
            schema="afc.speed_budget.1", limit_seconds=BUDGET_SECONDS, spent_seconds=0., runs=[], active=None)
        if ledger.get("active"):
            raise RuntimeError("Unresolved prior benchmark: audit its process and charge elapsed time before continuing")
        remaining = ledger["limit_seconds"] - ledger["spent_seconds"]
        if remaining <= 60:
            raise RuntimeError("Stage 0 compute budget exhausted; no later stage may start")
        started = time.perf_counter()
        ledger["active"] = dict(name=args.name, started_at=datetime.now(timezone.utc).isoformat(), parent_pid=os.getpid())
        write_json(ledger_path, ledger)
        env = dict(os.environ, MKL_THREADING_LAYER="SEQUENTIAL", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                   OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                   OPENBLAS_NUM_THREADS=str(args.threads), MPLBACKEND="Agg", PYTHONUTF8="1")
        command = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
        code = 1
        try:
            result = subprocess.run(command, env=env, timeout=remaining, check=False)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
            print("Stage 0 budget reached; benchmark worker terminated", flush=True)
        finally:
            elapsed = time.perf_counter() - started
            ledger["spent_seconds"] += elapsed
            ledger["runs"].append({**ledger["active"], "elapsed_seconds": elapsed, "exit_code": code})
            ledger["active"] = None
            write_json(ledger_path, ledger)
        return code
    finally:
        lock.unlink(missing_ok=True)


def prepare_worker(args):
    import importlib.util
    started = time.perf_counter()
    spec = importlib.util.spec_from_file_location("prepare_speed", Path(__file__).with_name("prepare_load_speed_data.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.argv = [str(spec.origin), "--snapshot-dir", args.snapshot_dir, "--fold-model", args.fold_model,
                "--output-dir", args.data_output or str(Path(args.output_root) / "data"),
                "--threads", str(args.threads), "--window-policy", args.window_policy]
    if args.window_audit:
        sys.argv.extend(["--window-audit", args.window_audit])
    module.main()
    write_json(Path(args.output_root) / (args.name + "_prepare_stats.json"),
        dict(status="prepared", elapsed_seconds=time.perf_counter() - started,
             memory_peak_bytes=peak_memory_bytes(), memory_measurement="OS peak working set where available"))


def run_worker(args):
    import numpy as np
    import torch
    from spindle_power.load_prediction.common import file_hash
    from spindle_power.load_prediction.load_device import configure_load_device, rng_state, restore_rng
    from spindle_power.load_prediction.models import ModelConfig, build_load_models
    from spindle_power.load_prediction.training import TrainingConfig, _main_loss, _residual_loss
    from spindle_power.load_prediction.training_checkpoint import save_checkpoint, source_fingerprints

    config = TrainingConfig(device=args.device, main_batch_points=args.main_points,
                            residual_batch_windows=args.batch_windows, residual_batch_strategy=args.batch_strategy,
                            loss_cache_mode=args.cache, loss_cache_max_bytes=args.cache_max_bytes)
    config.validate()
    destination = Path(args.output_root) / "runs" / args.name
    destination.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    _, execution = configure_load_device(args.device)
    data_path = Path(args.data).resolve()
    data_manifest = json.loads((data_path.parent / "data_manifest.json").read_text(encoding="utf-8"))
    data_sha = file_hash(data_path)
    expected = data_manifest["artifacts"][data_path.stem]
    if expected["file"] != data_path.name or expected["sha256"] != data_sha:
        raise ValueError("Prepared benchmark data hash mismatch")
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    if data.get("schema") != "afc.speed_data.1":
        raise ValueError("Expected locally prepared performance-only F1 data")
    model_config = ModelConfig(**data["model_config"])
    model_config.residual_output = args.loss
    config.residual_window_policy = data["provenance"].get("window_policy", "legacy_v1")
    config.validate()
    torch.manual_seed(42)
    bundle = build_load_models(model_config, device=args.device)
    bundle.main.load_state_dict(data["initial_main"])
    bundle.residual.load_state_dict(data["initial_residual"])
    bundle.normalization = data["normalization"]
    if config.residual_window_policy != "legacy_v1":
        bundle.metadata["config"] = dict(residual_window_policy=config.residual_window_policy,
            residual_batch_windows=args.batch_windows, residual_batch_strategy=args.batch_strategy)
    peak_rss = peak_memory_bytes()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()
    def rss():
        nonlocal peak_rss
        peak_rss = max(peak_rss, peak_memory_bytes())
    setup = time.perf_counter() - begin
    results = []
    states = {}
    preparation_costs = {}
    cache_stats = {}
    for stage in (["main", "residual"] if args.stage == "both" else [args.stage]):
        module = getattr(bundle, stage)
        initial = deepcopy(module.state_dict())
        initial_rng = rng_state(args.device)
        prepare_start = time.perf_counter()
        if args.cache != "none":
            from spindle_power.load_prediction.training_cache import prepare_main_loss, prepare_residual_loss
            cache_options = dict(cache_device=args.cache == "device", max_cache_bytes=args.cache_max_bytes)
            if stage == "main":
                train = prepare_main_loss(bundle, data["main_train"], data["normalization"]["target_scale_w"], config, **cache_options)
                validation = prepare_main_loss(bundle, data["main_validation"], data["normalization"]["target_scale_w"], config, **cache_options)
            else:
                train = prepare_residual_loss(bundle, data["residual_train"], batch_windows=args.batch_windows,
                                              batch_strategy=args.batch_strategy, **cache_options)
                validation = prepare_residual_loss(bundle, data["residual_validation"], batch_windows=args.batch_windows,
                                                   batch_strategy=args.batch_strategy, **cache_options)
        elif stage == "main":
            train = lambda back: _main_loss(bundle, data["main_train"], data["normalization"]["target_scale_w"], config, None, back)
            validation = lambda back: _main_loss(bundle, data["main_validation"], data["normalization"]["target_scale_w"], config, None, back)
        else:
            train = lambda back: _residual_loss(bundle, data["residual_train"], None, back,
                                               batch_windows=args.batch_windows, batch_strategy=args.batch_strategy)
            validation = lambda back: _residual_loss(bundle, data["residual_validation"], None, back,
                                                    batch_windows=args.batch_windows, batch_strategy=args.batch_strategy)
        sync()
        preparation = time.perf_counter() - prepare_start
        cache_stats[stage] = dict(train=getattr(train, "stats", None), validation=getattr(validation, "stats", None))
        # Warm the actual complete pass, then discard all updates and RNG changes.
        warm_start = time.perf_counter()
        module.train()
        warm_optimizer = torch.optim.Adam([p for p in module.parameters() if p.requires_grad], lr=.001)
        warm_optimizer.zero_grad()
        train(True)
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.)
        warm_optimizer.step()
        module.eval()
        with torch.no_grad():
            validation(False)
        sync()
        warmup = time.perf_counter() - warm_start
        module.load_state_dict(initial)
        restore_rng(initial_rng, args.device)
        del warm_optimizer
        optimizer = torch.optim.Adam([p for p in module.parameters() if p.requires_grad], lr=.001)
        best, best_state, best_epoch, stale, history = float("inf"), None, 0, 0, []
        identity = dict(kind="engineering_speed_only", data_sha256=data_sha,
                        model_config=data["model_config"], stage=stage, execution=execution,
                        source=source_fingerprints(), batch_windows=args.batch_windows,
                        batch_strategy=args.batch_strategy, main_points=args.main_points, cache=args.cache,
                        cache_max_bytes=args.cache_max_bytes, loss=args.loss)
        for epoch in range(1, args.iterations + 1):
            sync()
            start = time.perf_counter()
            module.train()
            optimizer.zero_grad()
            loss = train(True)
            diagnostic_seconds = 0.
            if epoch == 1:
                sync()
                diagnostic_start = time.perf_counter()
                gradients = {key: p.grad.detach().cpu().clone() for key, p in module.named_parameters() if p.grad is not None}
                sync()
                diagnostic_seconds = time.perf_counter() - diagnostic_start
            norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 1.)
            if not bool(torch.isfinite(norm)):
                raise ValueError("Non-finite benchmark gradient")
            optimizer.step()
            sync()
            train_seconds = time.perf_counter() - start - diagnostic_seconds
            start = time.perf_counter()
            module.eval()
            with torch.no_grad():
                value = validation(False)
            if value < best - config.min_delta:
                best, best_state, best_epoch, stale = value, deepcopy(module.state_dict()), epoch, 0
            else:
                stale += 1
            sync()
            val_seconds = time.perf_counter() - start
            history.append(dict(epoch=epoch, train_loss=loss, validation_loss=value))
            start = time.perf_counter()
            save_checkpoint(destination / (stage + ".pt"), identity,
                            dict(epoch=epoch, completed=epoch == args.iterations, weights=module.state_dict(),
                                 optimizer=optimizer.state_dict(), **rng_state(args.device), optimizer_updates=epoch,
                                 best_loss=best, best_state=best_state, best_epoch=best_epoch, stale=stale, history=history))
            sync()
            save_seconds = time.perf_counter() - start
            rss()
            row = dict(stage=stage, epoch=epoch, train_seconds=train_seconds, validation_seconds=val_seconds,
                       checkpoint_seconds=save_seconds, total_seconds=train_seconds+val_seconds+save_seconds,
                       diagnostic_seconds=diagnostic_seconds, train_loss=loss, validation_loss=value)
            results.append(row)
            print(json.dumps({"run":args.name, **row}), flush=True)
            if epoch == 1:
                states[stage] = dict(gradients=gradients, weights={k:v.detach().cpu().clone() for k,v in module.state_dict().items()},
                                     train_loss=loss, validation_loss=value,
                                     initial_weights={k:v.detach().cpu().clone() for k,v in initial.items()},
                                     initial_rng=initial_rng)
        torch.save(states, destination / "first_update.pt")
        write_json(destination / (stage + "_setup.json"), dict(preparation_seconds=preparation, warmup_seconds=warmup))
        preparation_costs[stage] = preparation
        del train, validation, optimizer
    sync()
    report = dict(status="measured", scientific_accuracy_verified=False, args=vars(args), execution=execution,
                  data_sha256=data_sha, setup_seconds=setup, total_process_seconds=time.perf_counter()-begin,
                  memory_peak_bytes=peak_rss, memory_measurement="OS peak working set where available",
                  warmup_optimizer_released_before_timing=True,
                  cuda_peak_bytes=torch.cuda.max_memory_allocated() if args.device=="cuda" else None,
                  preparation_seconds=preparation_costs,
                  cache_stats=cache_stats,
                  epochs=results, medians={stage:float(np.median([r["total_seconds"] for r in results if r["stage"]==stage]))
                                          for stage in {r["stage"] for r in results}})
    report["projected_qp_compute_seconds"] = (setup + sum(preparation_costs.values()) +
        sum(report["medians"].get(stage, 0.) * updates for stage, updates in (("main", 100), ("residual", 400))))
    report["projection_scope"] = "100 main + 400 residual Q updates, amortized measured setup; excludes raw-data rebuild; P must be rechecked"
    write_json(destination / "result.json", report)
    print(json.dumps({"run":args.name, "medians":report["medians"], "total_seconds":report["total_process_seconds"]}), flush=True)


def check_prospective_worker(args):
    """Fresh formal-source JUnit receipt, charged by the existing parent budget."""
    import pytest
    import tempfile
    from spindle_power.load_prediction.common import file_hash
    from spindle_power.load_prediction.training_checkpoint import source_fingerprints
    from spindle_power.load_prediction import neural, training, training_service

    test = ROOT / "tests" / "test_prospective_residual_windows.py"
    xml = Path(args.output_root) / (args.name + ".xml")
    receipt = Path(args.output_root) / (args.name + "_checks.json")
    if any(path.exists() for path in (xml, receipt)):
        raise FileExistsError("Refusing to overwrite a prospective test run")
    imports = {module.__name__: str(Path(module.__file__).resolve()) for module in (neural, training, training_service)}
    if any(not Path(path).is_relative_to((ROOT / "src").resolve()) for path in imports.values()):
        raise ValueError("Prospective tests must import the formal workspace source, not a candidate mirror")
    source, test_sha = source_fingerprints(), file_hash(test)
    # Keep nested service exports below Windows path limits; retain test evidence.
    temporary_root = Path(tempfile.mkdtemp(prefix="afc-prospective-"))
    temporary = temporary_root / "pytest"
    started = time.perf_counter()
    code = pytest.main([str(test), "-q", "--basetemp", str(temporary), "--junitxml", str(xml)])
    unchanged = source == source_fingerprints() and test_sha == file_hash(test)
    value = dict(schema="paper1.prospective_checks.1", status="passed" if code == 0 and unchanged else "failed",
        passed=code == 0 and unchanged, return_code=int(code), source=source, imported_modules=imports,
        formal_root=str(ROOT.resolve()), candidate_source_used=False,
        suite=dict(path=str(test.resolve()), sha256=test_sha),
        temporary_root=str(temporary_root.resolve()), temporary_directory=str(temporary.resolve()),
        junit=None if not xml.is_file() else dict(path=str(xml.resolve()), sha256=file_hash(xml)),
        elapsed_seconds=time.perf_counter() - started, source_unchanged=unchanged,
        scientific_training_started=False)
    write_json(receipt, value)
    return int(code) if code else (0 if unchanged else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "check", "check-prospective"])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--snapshot-dir")
    parser.add_argument("--fold-model")
    parser.add_argument("--data")
    parser.add_argument("--data-output", help="New preparation directory; never overwrites stage-0 data")
    parser.add_argument("--window-policy", choices=["legacy_v1", "prospective_v1"], default="legacy_v1")
    parser.add_argument("--window-audit")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-windows", type=int, default=128)
    parser.add_argument("--batch-strategy", choices=["padded", "equal_length"], default="padded")
    parser.add_argument("--main-points", type=int, default=8192)
    parser.add_argument("--cache", choices=["none", "cpu", "device"], default="none")
    parser.add_argument("--cache-max-bytes", type=int, default=1024**3,
                        help="Persistent device tensor budget per train/validation loss callable")
    parser.add_argument("--stage", choices=["both", "main", "residual"], default="both")
    parser.add_argument("--loss", choices=["quantiles", "point"], default="quantiles")
    parser.add_argument("--iterations", type=int, choices=[3, 4, 5], default=3)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.worker:
        return bounded_child(args)
    try:
        if args.action == "prepare":
            prepare_worker(args)
        elif args.action == "check-prospective":
            return check_prospective_worker(args)
        elif args.action == "check":
            import pytest
            return pytest.main([str(ROOT / "tests" / "test_training_cache.py"),
                                str(ROOT / "tests" / "test_training_cache_scalar_safety.py"),
                                str(ROOT / "tests" / "test_training_cache_integration.py"),
                                str(ROOT / "tests" / "test_load_gpu_route.py"), "-q",
                                "--basetemp", str(Path(args.output_root) / (args.name + "_tmp")),
                                "--junitxml", str(Path(args.output_root) / (args.name + ".xml"))])
        else:
            run_worker(args)
        return 0
    except Exception:
        failure = dict(status="failed", traceback=traceback.format_exc(), name=args.name)
        output = Path(args.output_root) / (args.name + "_failure.json")
        write_json(output, failure)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
