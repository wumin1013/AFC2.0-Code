"""Recheck prospective Q/P execution under the ORIGINAL stage-0 budget.

Only --execute starts work. Every child uses benchmark_load_execution's bounded
parent and existing budget.json. This does not start the scientific Q/P study.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
POLICY = "prospective_v1"
RUNS = (("Q_none", "quantiles", "none"), ("Q_cpu", "quantiles", "cpu"),
        ("P_none", "point", "none"), ("P_cpu", "point", "cpu"))
EXPECTED = dict(device="cpu", cpu_threads=10, cpu_interop_threads=1, main_batch_points=32768,
                residual_batch_windows=128, residual_batch_strategy="padded", loss_cache_mode="cpu",
                loss_cache_max_bytes=1073741824, learning_rate=.001, gradient_clip=1., seed=42)
SCRIPTS = ("recheck_paper1_execution.py", "benchmark_load_execution.py", "prepare_load_speed_data.py",
           "summarize_load_speed.py", "audit_paper1_prospective_windows.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def reference(path):
    from spindle_power.load_prediction.common import file_hash
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_hash(path))


def recheck(value):
    require(reference(value["path"])["sha256"] == value["sha256"], "Bound evidence changed: " + value["path"])


def existing_budget(stage0):
    path = stage0 / "budget.json"
    require(path.is_file(), "The original stage-0 budget must exist; never initialize a new budget")
    value = read_json(path)
    require(value.get("schema") == "afc.speed_budget.1" and value.get("limit_seconds") == 5400,
            "Expected the original 5400-second budget")
    require(value.get("active") is None and not (stage0 / "budget.lock").exists(), "Stage-0 budget is active or unresolved")
    require(isinstance(value.get("spent_seconds"), (int, float)) and math.isfinite(value["spent_seconds"])
            and 0 < value["spent_seconds"] < 5340 and isinstance(value.get("runs"), list) and value["runs"],
            "Stage-0 cumulative budget is invalid or exhausted")
    return value


def preflight(args):
    from spindle_power.load_prediction.training_checkpoint import source_fingerprints
    from audit_paper1_support_fill import read_snapshot, development_records
    stage0, snapshot_path = args.stage0_root.resolve(), args.candidate_snapshot.resolve()
    budget = existing_budget(stage0)
    parent = read_json(args.execution_profile)
    require(parent.get("schema") == "afc.paper1_execution_profile.1"
            and parent.get("status") == "stage0_engineering_accepted", "Accepted stage-0 execution profile required")
    require(all(parent["training_config"].get(key) == value for key, value in EXPECTED.items()),
            "Recheck must preserve CPU10/b128/padded/main32768 and the original optimizer/cache choices")
    require(Path(parent["stage0_budget"]["ledger"]).resolve() == stage0 / "budget.json"
            and budget["spent_seconds"] >= parent["stage0_budget"]["spent_seconds_before_artifact_freeze"],
            "Budget is not the original profile's cumulative ledger")
    recheck(parent["selection_evidence"])
    require(read_json(parent["selection_evidence"]["path"]).get("status") == "stage0_passed", "Stage-0 selection evidence failed")
    snapshot = read_snapshot(snapshot_path)
    entries = development_records(snapshot)
    source = source_fingerprints()
    window = read_json(args.window_audit)
    require(window.get("schema") == "paper1.prospective_window_audit.1" and window.get("status") == "passed"
            and window.get("passed") is True and window.get("checked_records") == 450 and window.get("failures") == []
            and window.get("candidate_snapshot") == reference(snapshot_path) and window.get("source") == source
            and window.get("window_policy") == POLICY and window.get("cpu_threads") == EXPECTED["cpu_threads"],
            "A new passed formal-source 450-record prospective window audit is required")
    require(window.get("execution_profile") == reference(args.execution_profile), "Window audit and recheck use different stage-0 profiles")
    require(window.get("representative_reports") and len(window.get("records", [])) == 450,
            "Window audit lacks population or actual-forward evidence")
    require({row.get("record_id") for row in window["records"]} == set(entries),
            "Window audit record membership differs from the frozen 450 records")
    window_refs = [window["frozen_f1_model"], *window["representative_reports"],
                   *(row["manifest"] for row in window["records"])]
    for item in window_refs:
        recheck(item)
    for item in window["representative_reports"]:
        report = read_json(item["path"])
        require(report.get("passed") is True and report.get("train_deployment_context_exact") is True
                and report.get("deployment_outputs_label_invariant") is True, "Incomplete actual-forward window evidence")
        window_refs.append(report["tensors"])
        recheck(report["tensors"])
    bindings = dict(candidate_snapshot=reference(snapshot_path), window_audit=reference(args.window_audit),
                    parent_execution_profile=reference(args.execution_profile), selection_evidence=reference(parent["selection_evidence"]["path"]),
                    engineering_fold_model=window["frozen_f1_model"])
    scripts = {name: reference(Path(__file__).with_name(name)) for name in SCRIPTS}
    test = reference(ROOT / "tests" / "test_prospective_residual_windows.py")
    return dict(stage0=stage0, snapshot=snapshot, entries=entries, budget=budget, source=source,
                parent=parent, bindings=bindings, scripts=scripts, test=test, window_refs=window_refs)


def child_command(state, output, prefix, suffix, action, *options):
    return [sys.executable, "-X", "utf8", str(Path(__file__).with_name("benchmark_load_execution.py")), action,
            "--output-root", str(state["stage0"]), "--name", prefix + "_" + suffix, "--threads", "10", *map(str, options)]


def execute_child(command, stage0, log_path, previous):
    require(read_json(stage0 / "budget.json") == previous, "Budget changed outside this sequential recheck")
    name = command[command.index("--name") + 1]
    require(not any(row.get("name") == name for row in previous["runs"]), "Recheck run name already used")
    require(not (stage0 / "runs" / name).exists() and not (stage0 / (name + ".xml")).exists(), "Recheck output already exists")
    print(json.dumps(dict(phase=name, status="starting")), flush=True)
    env = dict(os.environ, MKL_THREADING_LAYER="SEQUENTIAL", PYTHONUTF8="1")
    with log_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                check=False, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    current = read_json(stage0 / "budget.json")
    require(current.get("active") is None and current["runs"][:-1] == previous["runs"]
            and current["runs"][-1].get("name") == name, "Child did not resolve and append exactly one original-budget entry")
    require(current["spent_seconds"] >= previous["spent_seconds"], "Budget counter decreased")
    require(result.returncode == 0 and current["runs"][-1].get("exit_code") == 0,
            f"Bounded child failed ({result.returncode}): {name}; see {log_path}")
    require(current["spent_seconds"] < current["limit_seconds"], "Original speed budget exhausted")
    print(json.dumps(dict(phase=name, status="completed", cumulative_seconds=current["spent_seconds"])), flush=True)
    return current


def formal_checks(stage0, prefix, source, test):
    from summarize_load_speed import checks_evidence
    receipt_path = stage0 / (prefix + "_checks_checks.json")
    receipt = read_json(receipt_path)
    require(receipt.get("schema") == "paper1.prospective_checks.1" and receipt.get("passed") is True
            and receipt.get("status") == "passed" and receipt.get("return_code") == 0
            and receipt.get("source") == source and receipt.get("source_unchanged") is True
            and receipt.get("candidate_source_used") is False and receipt.get("formal_root") == str(ROOT.resolve())
            and receipt.get("suite") == test, "Fresh tests must bind the current formal source and suite")
    expected_modules = {"spindle_power.load_prediction." + name: str((ROOT / "src/spindle_power/load_prediction" / (name + ".py")).resolve())
                        for name in ("neural", "training", "training_service")}
    require(receipt.get("imported_modules") == expected_modules, "Candidate or other workspace source was tested")
    recheck(receipt["junit"])
    checks = checks_evidence(Path(receipt["junit"]["path"]), 29, ["test_prospective_residual_windows"])
    require(checks["passed"], "Fresh formal prospective 29-test JUnit suite must pass without skips")
    return dict(receipt=reference(receipt_path), junit=receipt["junit"], checks=checks)


def prepared_data(output, state):
    manifest_path = output / "data/data_manifest.json"
    manifest = read_json(manifest_path)
    provenance = manifest["provenance"]
    require(manifest.get("status") == "engineering_data_prepared" and manifest.get("formal_ready") is False
            and provenance.get("window_policy") == POLICY and provenance.get("code_identity") == state["source"]
            and provenance.get("snapshot_sha256") == state["bindings"]["candidate_snapshot"]["sha256"]
            and provenance.get("fold_model_sha256") == state["bindings"]["engineering_fold_model"]["sha256"]
            and provenance.get("window_audit") == state["bindings"]["window_audit"]
            and provenance.get("seed") == 42, "Prepared prospective data provenance mismatch")
    full = manifest["artifacts"]["full"]
    require(full["file"] == "full.pt" and full["train_records"] == 400 and full["validation_records"] == 50,
            "Recheck requires complete F1 data, not subset data")
    artifact = dict(path=str((output / "data/full.pt").resolve()), sha256=full["sha256"])
    recheck(artifact)
    return manifest, dict(manifest=reference(manifest_path), full=artifact)


def measure_evidence(stage0, name, loss, cache, data_sha, source):
    import torch
    from summarize_load_speed import TOLERANCES, mapping_check, metric
    directory = stage0 / "runs" / name
    raw = read_json(directory / "result.json")
    args = raw["args"]
    required = dict(name=name, device="cpu", threads=10, main_points=32768, batch_windows=128,
                    batch_strategy="padded", cache=cache, cache_max_bytes=1073741824, stage="both", loss=loss, iterations=3,
                    window_policy=POLICY)
    require(all(args.get(key) == value for key, value in required.items()) and raw.get("status") == "measured"
            and raw.get("scientific_accuracy_verified") is False and raw.get("data_sha256") == data_sha,
            "Incomplete or mismatched prospective full-F1 measurement")
    execution = raw["execution"]
    require(execution.get("device") == "cpu" and execution.get("cpu_threads") == 10
            and execution.get("cpu_interop_threads") == 1, "Measured execution differs from the frozen profile")
    states = torch.load(directory / "first_update.pt", map_location="cpu", weights_only=True)
    artifacts = [reference(directory / "result.json"), reference(directory / "first_update.pt")]
    stages = {}
    for stage in ("main", "residual"):
        rows = [row for row in raw["epochs"] if row["stage"] == stage]
        require([row["epoch"] for row in rows] == [1, 2, 3], "Each stage requires exactly three complete updates")
        require(all(math.isfinite(row[key]) for row in rows for key in ("train_loss", "validation_loss")), "Nonfinite measured losses")
        require(all(states[stage][key] == rows[0][key] for key in ("train_loss", "validation_loss")), "First update disagrees with timing report")
        for key in ("gradients", "weights", "initial_weights"):
            tolerance = TOLERANCES["gradients" if key == "gradients" else "weights"]
            require(mapping_check(states[stage][key], states[stage][key], tolerance)["passed"], "Missing or nonfinite update tensors")
        require(set(states[stage]["initial_rng"]) == {"torch_rng"}, "Missing CPU initialization RNG state")
        rng = states[stage]["initial_rng"]["torch_rng"]
        require(torch.is_tensor(rng) and rng.dtype == torch.uint8 and rng.ndim == 1 and rng.numel() > 0, "Invalid initialization RNG")
        for split in ("train", "validation"):
            stat = raw["cache_stats"][stage][split]
            require(stat is None if cache == "none" else isinstance(stat, dict) and stat.get("max_cache_bytes") == 1073741824
                    and stat.get("device_cache_bytes") == 0 and stat.get("batch_count", 0) > 0, "Cache measurement does not match requested mode")
        setup_path = directory / (stage + "_setup.json")
        setup = read_json(setup_path)
        require(setup["preparation_seconds"] == raw["preparation_seconds"][stage], "Preparation cost mismatch")
        stages[stage] = dict(updates=3, preparation=metric([setup["preparation_seconds"]]), warmup=metric([setup["warmup_seconds"]]),
            **{key: metric([row[key] for row in rows]) for key in ("train_seconds", "validation_seconds", "checkpoint_seconds", "total_seconds")})
        artifacts.append(reference(setup_path))
        for epoch in (1, 2, 3):
            path = directory / "epochs" / stage / f"epoch_{epoch:04d}.pt"
            saved = torch.load(path, map_location="cpu", weights_only=True)
            require(saved["identity"].get("source") == source and saved["identity"].get("data_sha256") == data_sha
                    and saved["identity"].get("loss") == loss and saved["identity"].get("cache") == cache
                    and saved["state"].get("optimizer_updates") == epoch and saved["state"].get("epoch") == epoch
                    and saved["state"].get("completed") is (epoch == 3), "Archived optimizer update provenance mismatch")
            require(saved["state"].get("optimizer", {}).get("state"), "Missing actual Adam state")
            if epoch == 1:
                require(mapping_check(states[stage]["weights"], saved["state"]["weights"], TOLERANCES["weights"])["exact"],
                        "Reported one-Adam weights disagree with immutable epoch archive")
            artifacts.append(reference(path))
    require(raw.get("memory_peak_bytes", 0) > 0, "Missing OS peak memory")
    metric([raw["setup_seconds"], raw["total_process_seconds"]])
    return dict(states=states, summary=dict(name=name, loss=loss, cache=cache, stages=stages, execution=execution,
        setup_seconds=raw["setup_seconds"], total_process_seconds=raw["total_process_seconds"],
        memory_peak_bytes=raw["memory_peak_bytes"], artifacts=artifacts))


def compare_pair(control, cached):
    import torch
    from summarize_load_speed import TOLERANCES, mapping_check, tensor_check
    stages = {}
    for stage in ("main", "residual"):
        left, right = cached["states"][stage], control["states"][stage]
        # Existing diagnostics divide by the allowance; retain their positive
        # fixed tolerances, but require exact initialization rather than merely
        # a tolerable difference (zero allowance would serialize a NaN ratio).
        initial = mapping_check(left["initial_weights"], right["initial_weights"], TOLERANCES["weights"])
        initial.update(passed=initial["exact"], requirement="bitwise_exact")
        rng = torch.equal(left["initial_rng"]["torch_rng"], right["initial_rng"]["torch_rng"])
        losses = {key: tensor_check(torch.tensor(left[key], dtype=torch.float64), torch.tensor(right[key], dtype=torch.float64), TOLERANCES["loss"])
                  for key in ("train_loss", "validation_loss")}
        gradients = mapping_check(left["gradients"], right["gradients"], TOLERANCES["gradients"])
        weights = mapping_check(left["weights"], right["weights"], TOLERANCES["weights"])
        stages[stage] = dict(passed=initial["passed"] and rng and all(item["passed"] for item in losses.values())
                            and gradients["passed"] and weights["passed"], initial_weights=initial, initial_rng_exact=rng,
                            losses=losses, raw_gradients=gradients, one_adam_weights=weights)
    return dict(passed=all(item["passed"] for item in stages.values()), tolerances=TOLERANCES, stages=stages)


def qp_projection(measurements, prepare_seconds, cache="cpu"):
    q, p = measurements["Q_" + cache], measurements["P_" + cache]
    setup = prepare_seconds + q["setup_seconds"] + p["setup_seconds"]
    # The pilot rebuilds residual caches at each 100/200 milestone.
    setup += q["stages"]["main"]["preparation"]["median"]
    setup += 2 * sum(item["stages"]["residual"]["preparation"]["median"] for item in (q, p))
    estimates = {key: setup + 100 * q["stages"]["main"]["total_seconds"][key]
                 + 200 * q["stages"]["residual"]["total_seconds"][key]
                 + 200 * p["stages"]["residual"]["total_seconds"][key] for key in ("minimum", "median", "maximum")}
    return dict(seconds=estimates, setup_seconds=setup, main_updates=100, q_updates=200, p_updates=200,
                measured_validation="full_F1_loss_validation", macro_rmse_and_export_cost_measured=False,
                limitation="Pilot macro-RMSE inference includes all known windows; additional cost and milestone exports are unmeasured. Calibrate at epoch 100; 7.5/8-hour gates override this estimate.")


def charge_overhead(stage0, before, prefix, started, child_names):
    from spindle_power.load_prediction.common import save_json
    lock = stage0 / "budget.lock"
    with lock.open("x", encoding="utf-8"):
        pass
    try:
        value = read_json(stage0 / "budget.json")
        require(value.get("active") is None and value["runs"][:len(before["runs"])] == before["runs"], "Budget changed or remains unresolved")
        added = value["runs"][len(before["runs"]):]
        children = [row["name"] for row in added if row["name"] in child_names]
        require(children == child_names[:len(children)] and all(row["name"] in child_names or
                row["name"].startswith(prefix + "_orchestration_") for row in added)
                and len({row["name"] for row in added}) == len(added), "Unexpected concurrent budget entries")
        # Called before every child as well as at the end. Subtract ALL work
        # already charged, including earlier orchestration entries.
        elapsed = max(0., time.perf_counter() - started - (value["spent_seconds"] - before["spent_seconds"]))
        value["runs"].append(dict(name=prefix + f"_orchestration_{len(added):02d}", elapsed_seconds=elapsed, exit_code=0,
                                   started_at=datetime.now(timezone.utc).isoformat(), action="recheck_preflight_summary_and_evidence"))
        value["spent_seconds"] += elapsed
        save_json(stage0 / "budget.json", value)
        return value
    finally:
        lock.unlink(missing_ok=True)


def run(args):
    if not args.execute:
        return dict(status="planned_not_executed", scientific_training_started=False, window_policy=POLICY,
                    sequence=["new_prospective_data", "formal_29_tests", *[item[0] for item in RUNS]],
                    original_budget_required=True, iterations=3, training_config=EXPECTED)
    started = time.perf_counter()
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    from spindle_power.load_prediction.common import save_json
    from spindle_power.load_prediction.training_checkpoint import source_fingerprints
    output = args.output_dir.resolve()
    require(not output.exists(), "New recheck output directory required; never overwrite prior evidence")
    require(not output.is_relative_to(args.candidate_snapshot.resolve().parent), "Recheck output overlaps candidate snapshot")
    state = preflight(args)
    output.mkdir(parents=True)
    prefix = "prospective_" + uuid.uuid4().hex[:12]
    commands = [child_command(state, output, prefix, "prepare", "prepare", "--snapshot-dir", args.candidate_snapshot.resolve().parent,
        "--fold-model", state["bindings"]["engineering_fold_model"]["path"], "--data-output", output / "data",
        "--window-policy", POLICY, "--window-audit", args.window_audit.resolve()),
        child_command(state, output, prefix, "checks", "check-prospective")]
    commands.extend(child_command(state, output, prefix, name, "run", "--data", output / "data/full.pt",
        "--main-points", "32768", "--batch-windows", "128", "--batch-strategy", "padded", "--cache", cache,
        "--cache-max-bytes", "1073741824", "--stage", "both", "--loss", loss, "--iterations", "3", "--window-policy", POLICY)
        for name, loss, cache in RUNS)
    child_names = [command[command.index("--name") + 1] for command in commands]
    save_json(output / "plan.json", dict(commands=commands, prefix=prefix, budget_before=state["budget"],
        bindings=state["bindings"], scripts=state["scripts"], source=state["source"], test_suite=state["test"]))
    result, current = {}, state["budget"]
    def deadline():
        require(time.perf_counter() - started < 5400 - state["budget"]["spent_seconds"], "Original cumulative speed deadline reached")
    try:
        for index, command in enumerate(commands):
            deadline()
            current = charge_overhead(state["stage0"], state["budget"], prefix, started, child_names)
            require(current["spent_seconds"] < 5340, "Insufficient remaining original budget for another child")
            current = execute_child(command, state["stage0"], output / f"child_{index:02d}.log", current)
            if index == 0:
                data, data_refs = prepared_data(output, state)
                preparation_stats_path = state["stage0"] / (prefix + "_prepare_prepare_stats.json")
                preparation_stats = read_json(preparation_stats_path)
                require(preparation_stats.get("status") == "prepared" and preparation_stats.get("memory_peak_bytes", 0) > 0
                        and math.isfinite(preparation_stats["elapsed_seconds"]) and preparation_stats["elapsed_seconds"] >= 0,
                        "Missing preparation wall-time/peak-memory evidence")
                data_refs["preparation_stats"] = reference(preparation_stats_path)
            elif index == 1:
                tests = formal_checks(state["stage0"], prefix, state["source"], state["test"])
        evidence = {}
        for name, loss, cache in RUNS:
            deadline()
            evidence[name] = measure_evidence(state["stage0"], prefix + "_" + name, loss, cache,
                data_refs["full"]["sha256"], state["source"])
        comparisons = {kind: compare_pair(evidence[kind + "_none"], evidence[kind + "_cpu"]) for kind in ("Q", "P")}
        require(all(item["passed"] for item in comparisons.values()), "New protocol cache equivalence failed; do not release")
        measurements = {name: item["summary"] for name, item in evidence.items()}
        for item in (*state["bindings"].values(), *state["scripts"].values(), state["test"],
                     *data_refs.values(), tests["receipt"], tests["junit"], *state["window_refs"],
                     *(item for run in measurements.values() for item in run["artifacts"])):
            deadline()
            recheck(item)
        require(source_fingerprints() == state["source"], "Formal source changed during prospective recheck")
        result = dict(status="passed", passed=True, measurements=measurements, comparisons=comparisons,
            formal_checks=tests, prepared_data=data_refs, preparation_seconds=data["elapsed_seconds"],
            preparation_cost=preparation_stats,
            initial_versions=data["provenance"]["initial_versions"], qp_projection=qp_projection(measurements, data["elapsed_seconds"]),
            uncached_qp_projection=qp_projection(measurements, data["elapsed_seconds"], "none"),
            cache_speedups={kind: {stage: measurements[kind + "_none"]["stages"][stage]["total_seconds"]["median"] /
                                  measurements[kind + "_cpu"]["stages"][stage]["total_seconds"]["median"]
                                  for stage in ("main", "residual")} for kind in ("Q", "P")})
    except Exception as exc:
        result = dict(status="failed", passed=False, error_type=type(exc).__name__, error=str(exc))
    try:
        budget = charge_overhead(state["stage0"], state["budget"], prefix, started, child_names)
        require(budget["spent_seconds"] < budget["limit_seconds"], "Cumulative stage-0 budget exhausted")
    except Exception as exc:
        budget = read_json(state["stage0"] / "budget.json")
        result.update(status="failed", passed=False, budget_error=str(exc))
    save_json(output / "budget_after.json", budget)
    result.update(schema="paper1.load_execution_recheck.1", **state["bindings"], source=state["source"],
        scripts=state["scripts"], test_suite=state["test"], window_policy=POLICY, training_config={**EXPECTED, "residual_window_policy": POLICY},
        budget_before_seconds=state["budget"]["spent_seconds"], budget_after=reference(output / "budget_after.json"),
        original_budget_path=str(state["stage0"] / "budget.json"), elapsed_seconds=time.perf_counter() - started,
        scientific_training_started=False, scientific_accuracy_improvement_claimed=False, engineering_weights_reused_for_training=False)
    save_json(output / "report.json", result)
    if result["passed"]:
        profile = dict(schema="paper1.load_execution_profile.1", status="prospective_rechecked", window_policy=POLICY,
            training_config=result["training_config"], candidate_snapshot=state["bindings"]["candidate_snapshot"],
            window_audit=state["bindings"]["window_audit"], parent_execution_profile=state["bindings"]["parent_execution_profile"],
            selection_evidence=reference(output / "report.json"), source_fingerprints=state["source"],
            scientific_accuracy_verified=False, automatic_scientific_training_start=False, reuse_engineering_weights=False)
        save_json(output / "execution_profile.json", profile)
    print(json.dumps({key: result[key] for key in ("status", "passed", "elapsed_seconds")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage0-root", "candidate-snapshot", "window-audit", "execution-profile", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = run(args)
    if not args.execute:
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] in ("planned_not_executed", "passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
