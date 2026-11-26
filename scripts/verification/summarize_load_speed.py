"""Audit saved stage-0 evidence without training or recomputing predictions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
NAMES = ("full_cpu4_original", "full_cpu_optimized", "full_cpu_control",
         "full_cuda_optimized", "full_cuda_control")
SCALAR_NAMES = ("full_cpu_scalar_optimized", "full_cuda_scalar_optimized")
REQUIRED_SUITES = ("test_training_cache", "test_load_gpu_route")
TOLERANCES = dict(loss=dict(rtol=3e-6, atol=2e-7),
                  gradients=dict(rtol=3e-4, atol=3e-6),
                  weights=dict(rtol=2e-5, atol=2e-6))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def tensor_check(actual, reference, tolerance):
    import torch
    actual, reference = torch.as_tensor(actual).detach().cpu(), torch.as_tensor(reference).detach().cpu()
    if actual.shape != reference.shape:
        return dict(passed=False, exact=False, reason="shape_mismatch", actual=list(actual.shape), reference=list(reference.shape))
    if not bool(torch.isfinite(actual).all() and torch.isfinite(reference).all()):
        return dict(passed=False, exact=False, reason="non_finite")
    difference = (actual.to(torch.float64) - reference.to(torch.float64)).abs()
    allowance = tolerance["atol"] + tolerance["rtol"] * reference.to(torch.float64).abs()
    exact = bool(torch.equal(actual, reference))
    return dict(passed=exact or bool(torch.all(difference <= allowance)), exact=exact,
        count=actual.numel(), failed_elements=int((difference > allowance).sum()),
        max_abs_error=float(difference.max()) if difference.numel() else 0.,
        max_tolerance_ratio=float((difference / allowance).max()) if difference.numel() else 0.)


def mapping_check(actual, reference, tolerance):
    if set(actual) != set(reference) or not reference:
        return dict(passed=False, exact=False, reason="missing_or_different_tensor_keys",
            actual_keys=sorted(actual), reference_keys=sorted(reference))
    checks = {name: tensor_check(actual[name], reference[name], tolerance) for name in reference}
    return dict(passed=all(value["passed"] for value in checks.values()),
        exact=all(value["exact"] for value in checks.values()), tensors=len(checks),
        max_abs_error=max(value.get("max_abs_error", 0.) for value in checks.values()),
        max_tolerance_ratio=max(value.get("max_tolerance_ratio", 0.) for value in checks.values()),
        failures={name: value for name, value in checks.items() if not value["passed"]},
        worst_tensor=max(checks, key=lambda name: checks[name].get("max_tolerance_ratio", 0.)))


def pair_check(actual_name, reference_name, evidence, *, same_execution=False):
    import torch
    result = dict(actual=actual_name, reference=reference_name, tolerances=TOLERANCES,
                  scientific_accuracy_comparison=False, passed=False)
    if actual_name not in evidence or reference_name not in evidence:
        return dict(result, reason="missing_full_result")
    actual, reference = evidence[actual_name], evidence[reference_name]
    if not actual["valid"] or not reference["valid"]:
        return dict(result, reason="invalid_full_result")
    if actual["raw"]["data_sha256"] != reference["raw"]["data_sha256"]:
        return dict(result, reason="different_initial_data_identity")
    fields = ("device", "threads", "main_points", "batch_windows", "batch_strategy", "loss", "iterations")
    if same_execution and any(actual["raw"]["args"].get(key) != reference["raw"]["args"].get(key) for key in fields):
        return dict(result, reason="optimized_control_execution_mismatch")
    left, right = actual["states"], reference["states"]
    stages = {}
    for stage in ("main", "residual"):
        # Scalars are kept in float64, without weakening the recorded loss test.
        losses = {key: tensor_check(torch.tensor(left[stage][key], dtype=torch.float64),
                                   torch.tensor(right[stage][key], dtype=torch.float64), TOLERANCES["loss"])
                  for key in ("train_loss", "validation_loss")}
        gradients = mapping_check(left[stage]["gradients"], right[stage]["gradients"], TOLERANCES["gradients"])
        weights = mapping_check(left[stage]["weights"], right[stage]["weights"], TOLERANCES["weights"])
        stages[stage] = dict(losses=losses, raw_accumulated_gradients=gradients, one_adam_weights=weights,
            passed=all(value["passed"] for value in losses.values()) and gradients["passed"] and weights["passed"],
            exact=all(value["exact"] for value in losses.values()) and gradients["exact"] and weights["exact"])
    return dict(result, stages=stages, passed=all(value["passed"] for value in stages.values()),
        exact=all(value["exact"] for value in stages.values()),
        comparison="exact_first_then_fixed_tolerances", initial_identity=actual["raw"]["data_sha256"])


def metric(values):
    if not values or not all(math.isfinite(float(value)) and value >= 0 for value in values):
        raise ValueError("Missing, negative or non-finite timing values")
    return dict(median=float(median(values)), minimum=float(min(values)), maximum=float(max(values)), values=values)


def run_evidence(root, name, expected_sha):
    import torch
    directory = root / "runs" / name
    result_path, state_path = directory / "result.json", directory / "first_update.pt"
    result = dict(name=name, valid=False, errors=[])
    if not result_path.is_file() or not state_path.is_file():
        result["errors"].append("missing_result_or_first_update")
        return result
    try:
        raw = read_json(result_path)
        result["raw"] = raw
        result["result_sha256"], result["first_update_sha256"] = sha256(result_path), sha256(state_path)
        args = raw["args"]
        if (raw["status"] != "measured" or raw.get("scientific_accuracy_verified") is not False
                or raw["data_sha256"] != expected_sha or Path(args["data"]).name != "full.pt"
                or args["name"] != name or args["stage"] != "both" or args["loss"] != "quantiles"):
            raise ValueError("Not a complete, common-data F1 engineering Q measurement")
        expected_device = "cuda" if name.startswith("full_cuda") else "cpu"
        if args["device"] != expected_device or raw["execution"]["device"] != expected_device:
            raise ValueError("Run name, requested device and recorded device disagree")
        if (name.endswith("control") or name == "full_cpu4_original") and args["cache"] != "none":
            raise ValueError("Original/control measurement unexpectedly used a cache")
        if name in SCALAR_NAMES and args["cache"] == "none":
            raise ValueError("Scalar-optimized measurement did not use the training cache")
        if name == "full_cpu4_original" and (args["threads"], args["batch_windows"], args["main_points"], args["batch_strategy"]) != (4, 128, 8192, "padded"):
            raise ValueError("Historical CPU4 baseline configuration changed")
        cache_rows = [raw["cache_stats"][stage][split]
                      for stage in ("main", "residual") for split in ("train", "validation")]
        if args["cache"] == "none":
            if any(row is not None for row in cache_rows):
                raise ValueError("Uncached measurement unexpectedly reported cache statistics")
            max_cache_bytes = None
        else:
            cache_limits = [row["max_cache_bytes"] for row in cache_rows]
            if (any(type(value) is not int or value < 0 for value in cache_limits)
                    or len(set(cache_limits)) != 1):
                raise ValueError("Train/validation cache limits must agree across both stages")
            max_cache_bytes = cache_limits[0]
        states = torch.load(state_path, map_location="cpu", weights_only=True)
        stage_results, setup = {}, float(raw["setup_seconds"])
        preparation = raw["preparation_seconds"]
        if not math.isfinite(setup) or setup < 0:
            raise ValueError("Invalid setup cost")
        for stage in ("main", "residual"):
            rows = [row for row in raw["epochs"] if row["stage"] == stage]
            if len(rows) < 3 or [row["epoch"] for row in rows] != list(range(1, len(rows) + 1)):
                raise ValueError("Full F1 requires at least three consecutive updates per stage")
            if len(rows) != args["iterations"]:
                raise ValueError("Measured updates do not match requested updates")
            prep = float(preparation[stage])
            if not math.isfinite(prep) or prep < 0:
                raise ValueError("Invalid preparation cost")
            if not all(math.isfinite(row[key]) for row in rows for key in ("train_loss", "validation_loss")):
                raise ValueError("Non-finite full-pass loss")
            for key in ("gradients", "weights"):
                if not mapping_check(states[stage][key], states[stage][key], TOLERANCES[key])["passed"]:
                    raise ValueError("Missing or non-finite first-update tensors")
            if any(states[stage][key] != rows[0][key] for key in ("train_loss", "validation_loss")):
                raise ValueError("First-update loss and timing report disagree")
            measured_setup = read_json(directory / (stage + "_setup.json"))
            if measured_setup["preparation_seconds"] != prep:
                raise ValueError("Stage preparation and result report disagree")
            stage_results[stage] = dict(updates=len(rows), preparation_seconds=prep,
                warmup_seconds=measured_setup["warmup_seconds"],
                **{key: metric([row[key] for row in rows]) for key in
                   ("train_seconds", "validation_seconds", "checkpoint_seconds", "total_seconds")})
        constant = setup + sum(float(value) for value in preparation.values())
        projection = {key: constant + sum(count * stage_results[stage]["total_seconds"][key]
                     for stage, count in (("main", 100), ("residual", 400)))
                     for key in ("median", "minimum", "maximum")}
        result.update(valid=True, states=states, stages=stage_results,
            projection_seconds=projection, projected_setup_seconds=constant,
            execution=dict({key: args.get(key) for key in ("device", "threads", "batch_windows", "main_points", "batch_strategy", "cache", "loss")},
                           loss_cache_max_bytes=max_cache_bytes),
            setup_seconds=setup, total_process_seconds=raw["total_process_seconds"],
            peak_ram_bytes=raw["memory_peak_bytes"], peak_vram_bytes=raw.get("cuda_peak_bytes"))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    return result


def checks_evidence(path, expected_tests, required_suites):
    result = dict(file=str(path), passed=False, independent_of_full_f1_numerics=True,
                  expected_tests=expected_tests, required_suites=required_suites,
                  missing_suites=list(required_suites))
    if not path.is_file():
        return dict(result, reason="missing_junit_xml")
    try:
        root = ET.parse(path).getroot()
        cases = list(root.iter("testcase"))
        counts = dict(tests=len(cases), failures=sum(case.find("failure") is not None for case in cases),
            errors=sum(case.find("error") is not None for case in cases),
            skipped=sum(case.find("skipped") is not None for case in cases))
        suites = sorted({case.get("classname", "") for case in cases})
        missing_suites = [required for required in required_suites
                          if not any(suite == required or suite.endswith("." + required) for suite in suites)]
        return dict(result, sha256=sha256(path), counts=counts,
            passed=counts == dict(tests=expected_tests, failures=0, errors=0, skipped=0) and not missing_suites,
            suites=suites, missing_suites=missing_suites,
            note=f"单份{expected_tests}项缓存、设备与恢复检查，必须覆盖指定套件；不与其他历史测试相加，不替代完整F1数值验收")
    except (OSError, ET.ParseError) as exc:
        return dict(result, reason=str(exc))


def number(value, precision=3):
    return "—" if value is None else f"{value:.{precision}f}"


def markdown(report):
    lines = ["# 训练设备与提速验收报告", "", f"状态：**{report['status']}**。仅评价执行成本与数值一致性，不评价预测精度。", ""]
    if report["blockers"]:
        lines.extend(["尚未满足的验收条件：", "", *[f"- {value}" for value in report["blockers"]], ""])
    choice = report["selection"]
    lines.extend([f"设备建议：{choice.get('selected_run') or '尚不能冻结'}。{choice['reason']}", "",
        f"实现身份规则：{choice['implementation_policy']}。", "",
        "## 完整F1计时", "", "每个完整更新包含训练、完整验证与检查点存档；数值诊断拷贝和预热不计入稳态耗时。", "",
        "|配置|设备/线程/窗口批量/主趋势块|缓存/每损失设备预算(MiB)|准备及加载(s)|主趋势中位(s)|残差中位(s)|统一工作量估计(h)|相对原CPU4速度|峰值RAM/VRAM(GiB)|",
        "|---|---|---|---:|---:|---:|---:|---:|---:|"])
    for name, value in report["full_runs"].items():
        if not value["valid"]:
            lines.append(f"|{name}|缺失或无效|—|—|—|—|—|—|—|")
            continue
        cfg, stages = value["execution"], value["stages"]
        setting = f"{cfg['device']}/{cfg['threads']}/{cfg['batch_windows']}/{cfg['main_points']}"
        vram = None if value["peak_vram_bytes"] is None else value["peak_vram_bytes"] / 1024**3
        cache_limit = None if cfg["loss_cache_max_bytes"] is None else cfg["loss_cache_max_bytes"] / 1024**2
        lines.append(f"|{name}|{setting}|{cfg['cache']}/{number(cache_limit, 0)}|{number(value['projected_setup_seconds'])}|"
            f"{number(stages['main']['total_seconds']['median'])}|{number(stages['residual']['total_seconds']['median'])}|"
            f"{number(value['projection_seconds']['median']/3600)}|{number(value.get('speedup_vs_original'))}×|"
            f"{number(value['peak_ram_bytes']/1024**3)}/{number(vram)}|")
    lines.extend(["", "|配置/阶段|缓存准备(s)|训练中位(s)|验证中位(s)|存档中位(s)|完整更新中位[min,max](s)|预热(s，已排除)|",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for name, value in report["full_runs"].items():
        if value["valid"]:
            for stage, row in value["stages"].items():
                total = row["total_seconds"]
                lines.append(f"|{name}/{stage}|{number(row['preparation_seconds'])}|{number(row['train_seconds']['median'])}|"
                    f"{number(row['validation_seconds']['median'])}|{number(row['checkpoint_seconds']['median'])}|"
                    f"{number(total['median'])} [{number(total['minimum'])}, {number(total['maximum'])}]|{number(row['warmup_seconds'])}|")
    lines.extend(["", "## 数值与独立测试证据", "", "|比较|严格一致|预定容差内通过|", "|---|---|---|"])
    for value in report["numeric_checks"].values():
        lines.append(f"|{value['actual']} 对 {value['reference']}|{value.get('exact', False)}|{value['passed']}|")
    lines.extend(["", "先检查逐元素严格一致，再报告固定容差。损失 rtol=3e-6、atol=2e-7；原始累积梯度 rtol=3e-4、atol=3e-6；一次Adam后权重 rtol=2e-5、atol=2e-6。没有自动放宽。", "",
        f"独立缓存/设备/恢复检查：{report['unit_checks'].get('counts', '缺失')}；通过={report['unit_checks']['passed']}。这份检查与完整F1数值比较分别验收。", "",
        f"要求测试数：{report['unit_checks']['expected_tests']}；要求套件：{'、'.join(report['unit_checks']['required_suites'])}；缺少套件：{'、'.join(report['unit_checks'].get('missing_suites', [])) or '无'}。", "",
        f"标量同步优化候选强制验收：{report['require_scalar']}；本次纳入：{'、'.join(report['scalar_runs']) or '无'}。候选一旦存在，缺失或不完整的证据均阻止冻结。", "",
        "## 预算与解释边界", "", f"累计预算：{number(report['budget'].get('spent_seconds', 0.)/60)} / {number(report['budget'].get('limit_seconds', 5400)/60)} 分钟。", "",
        "投影公式为一次已测加载与缓存准备 + 100次主趋势更新 + 400次残差Q更新。每次更新均含完整验证和同等存档。全部预热、首轮梯度诊断复制不计为正式训练成本。", "",
        "该工作量仅用于统一设备比较，不包含新数据构建、实际P损失训练、完整主趋势重新拟合后的残差特征准备、Q/P分段启动及额外导出。后续真实P训练和新窗口仍须短测，不能把此估计当作完整Q/P墙钟预算保证。", "",
        "GPU需数值验收通过、投影中位耗时至少低5%，且GPU投影最大值低于CPU投影最小值才被选择。min/max来自至少3次更新，仅描述观测波动，不是统计置信区间。", "",
        "缓存预算从每个阶段训练及验证的实际cache_stats提取，四者必须一致；该预算仅限制每份损失缓存中的设备张量，CPU缓存不受此设备预算限制，也不代表整个进程的内存上限。", "",
        "CPU内存为操作系统工作集峰值；GPU内存为PyTorch峰值已分配显存，未包含全部驱动/外部分配。本次原始benchmark在计时阶段仍保留预热Adam状态，因此峰值为该测速进程真实峰值，相对单optimizer正式训练为保守估计，不能作为精确正式训练内存。失败候选和预算已消耗时间均保留。子集筛选不代替完整F1确认。", "",
        "本报告不证明预测断点、窗口协议或预测低估得到解决，也不产生可注册的正式模型。", ""])
    return "\n".join(lines)


def summarize(args):
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    import torch
    torch.set_num_threads(1)
    root = Path(args.output_root).resolve()
    blockers, evidence = [], {}
    loaded = {}
    for key, relative in (("budget", "budget.json"), ("coarse", "coarse_screen.json"),
                          ("confirmation", "full_confirmation.json"), ("data", "data/data_manifest.json")):
        try:
            loaded[key] = read_json(root / relative)
        except (OSError, ValueError) as exc:
            loaded[key] = {}
            blockers.append(f"缺少或无法读取 {relative}：{type(exc).__name__}")
    budget = loaded["budget"]
    if budget.get("active") or (root / "budget.lock").exists():
        blockers.append("预算仍有活动进程或锁，不能冻结阶段0结论")
    if (budget.get("spent_seconds", 5400) >= budget.get("limit_seconds", 5400)
            or any(row.get("exit_code") == 124 for row in budget.get("runs", []))):
        blockers.append("累计预算已耗尽或存在超时任务")
    confirmation = loaded["confirmation"]
    declared_runs = {row.get("name") for row in confirmation} if isinstance(confirmation, list) else set()
    scalar_names = [name for name in SCALAR_NAMES if args.require_scalar or name in declared_runs
                    or (root / "runs" / name).exists() or (root / (name + "_failure.json")).exists()
                    or any(row.get("name") == name for row in budget.get("runs", []))]
    run_names = (*NAMES, *scalar_names)
    required_suites = list(dict.fromkeys((*REQUIRED_SUITES,
        *(("test_training_cache_scalar_safety",) if scalar_names else ()), *args.required_suite)))
    xml = Path(args.checks_xml).resolve() if args.checks_xml else root / "cache_device_recovery_checks.xml"
    unit_checks = checks_evidence(xml, args.expected_tests, required_suites)
    if not unit_checks["passed"]:
        blockers.append(f"独立{args.expected_tests}项缓存、设备及恢复检查未满足数量、套件覆盖或零失败/错误/跳过要求")
    if set(loaded["coarse"].get("finalists", {})) != {"cpu", "cuda"}:
        blockers.append("子集筛选未给出CPU和CUDA两份完整候选")
    confirmed = {row["name"] for row in confirmation if row.get("completed") and row.get("exit_code") == 0} if isinstance(confirmation, list) else set()
    if not set(NAMES).issubset(confirmed):
        blockers.append("完整F1确认清单缺少必需的五个成功配置")
    for name in scalar_names:
        ledger_rows = [row for row in budget.get("runs", []) if row.get("name") == name]
        if not ledger_rows or ledger_rows[-1].get("exit_code") != 0:
            blockers.append(f"{name}缺少预算账本中的成功终态")
    expected_sha = loaded["data"].get("artifacts", {}).get("full", {}).get("sha256")
    full_artifact = loaded["data"].get("artifacts", {}).get("full", {})
    provenance = loaded["data"].get("provenance", {})
    if not expected_sha or not provenance.get("initial_versions"):
        blockers.append("缺少共同完整F1数据及CPU初始化身份")
    if (full_artifact.get("train_records"), full_artifact.get("validation_records")) != (400, 50):
        blockers.append("共同数据不是完整F1的400条训练记录和50条验证记录")
    for name in run_names:
        evidence[name] = run_evidence(root, name, expected_sha)
        if not evidence[name]["valid"]:
            blockers.append(f"{name}缺失或验收失败：{'；'.join(evidence[name]['errors'])}")
    numeric = {
        "cpu_cache": pair_check("full_cpu_optimized", "full_cpu_control", evidence, same_execution=True),
        "cuda_cache": pair_check("full_cuda_optimized", "full_cuda_control", evidence, same_execution=True),
        "cpu_configuration": pair_check("full_cpu_control", "full_cpu4_original", evidence),
        "cuda_configuration": pair_check("full_cuda_control", "full_cpu4_original", evidence),
        "cross_device_optimized": pair_check("full_cuda_optimized", "full_cpu_optimized", evidence),
    }
    for name in scalar_names:
        device = "cuda" if name.startswith("full_cuda") else "cpu"
        numeric[device + "_scalar"] = pair_check(name, f"full_{device}_control", evidence, same_execution=True)
    if set(SCALAR_NAMES).issubset(scalar_names):
        numeric["cross_device_scalar"] = pair_check("full_cuda_scalar_optimized", "full_cpu_scalar_optimized", evidence)
    eligible = []
    if evidence["full_cpu4_original"]["valid"]:
        eligible.append("full_cpu4_original")
    if numeric["cpu_configuration"]["passed"]:
        eligible.append("full_cpu_control")
        if not args.require_scalar and numeric["cpu_cache"]["passed"]:
            eligible.append("full_cpu_optimized")
        if numeric.get("cpu_scalar", {}).get("passed"):
            eligible.append("full_cpu_scalar_optimized")
    cpu = min(eligible, key=lambda name: evidence[name]["projection_seconds"]["median"]) if eligible else None
    selection = dict(selected_run=None, provisional_cpu=cpu, reason="必需证据尚未齐全，不能冻结设备。",
        implementation_policy="强制scalar验收：缓存候选仅采用当前scalar实现的测量；旧缓存结果仅作历史对照，无缓存配置仍可候选"
            if args.require_scalar else "采用通过验收的已测候选，scalar候选存在时一并比较",
        eligible_cpu_runs=eligible,
        gpu_minimum_reduction=.05, noise_rule="gpu_projected_maximum < cpu_projected_minimum")
    if cpu:
        cuda_options = []
        cuda_names = ["full_cuda_control"]
        if not args.require_scalar:
            cuda_names.append("full_cuda_optimized")
        if "full_cuda_scalar_optimized" in scalar_names:
            cuda_names.append("full_cuda_scalar_optimized")
        for name in cuda_names:
            check = pair_check(name, cpu, evidence)
            numeric[name + "_vs_selected_cpu"] = check
            optimization_check = "cuda_scalar" if name in SCALAR_NAMES else "cuda_cache"
            if check["passed"] and (name.endswith("control") or numeric[optimization_check]["passed"]):
                cuda_options.append(name)
        gpu = min(cuda_options, key=lambda name: evidence[name]["projection_seconds"]["median"]) if cuda_options else None
        proposed = cpu
        selection.update(provisional_gpu=gpu, eligible_cuda_runs=cuda_options)
        if gpu:
            cpu_time, gpu_time = evidence[cpu]["projection_seconds"], evidence[gpu]["projection_seconds"]
            gain = 1. - gpu_time["median"] / cpu_time["median"]
            exceeds_noise = gpu_time["maximum"] < cpu_time["minimum"]
            selection.update(gpu_reduction=gain, gpu_exceeds_observed_variation=exceeds_noise)
            if gain >= .05 and exceeds_noise:
                proposed = gpu
        if not blockers:
            selection.update(selected_run=proposed, configuration=evidence[proposed]["execution"],
                reason="GPU通过数值、至少5%收益及观测波动门槛。" if proposed.startswith("full_cuda") else
                       "采用通过数值验收的最低投影成本CPU配置；GPU未同时满足全部选择门槛。")
            if args.require_scalar:
                selection["reason"] += "所选结果来自当前scalar实现或仍可复现的无缓存执行，不使用旧缓存实现冻结配置。"
    else:
        blockers.append("没有可用的CPU基准或通过验收的CPU配置")
    baseline = evidence["full_cpu4_original"]
    if baseline["valid"]:
        for value in evidence.values():
            if value["valid"]:
                value["speedup_vs_original"] = baseline["projection_seconds"]["median"] / value["projection_seconds"]["median"]
    # Tensor values and the repeated raw result payload stay in source evidence.
    serializable = {name: {key: value for key, value in row.items() if key not in ("states", "raw")}
                    for name, row in evidence.items()}
    report = dict(schema="afc.load_speed_summary.2", status="incomplete" if blockers else "stage0_passed",
        created_at=datetime.now(timezone.utc).isoformat(), scientific_accuracy_verified=False, formal_model_registered=False,
        blockers=blockers, selection=selection, full_runs=serializable, numeric_checks=numeric,
        required_runs=list(run_names), scalar_runs=scalar_names, require_scalar=args.require_scalar,
        cache_budget_scope="Per-loss device tensors only; CPU caches are not bounded by this device budget",
        memory_interpretation="Measured benchmark-process peak includes retained warmup Adam state; conservative relative to single-optimizer formal training, not its exact memory requirement",
        unit_checks=unit_checks, budget=budget, coarse_screen=loaded["coarse"], full_confirmation=confirmation,
        common_input=dict(full_data_sha256=expected_sha, snapshot_id=provenance.get("snapshot_id"),
                          initial_versions=provenance.get("initial_versions")),
        projection=dict(main_updates=100, residual_quantile_updates=400, true_point_training_measured=False,
            excludes=["warmup", "diagnostic_gradient_copy", "new_data_preparation", "real_P_training",
                      "post_main_residual_feature_rebuild", "Q_P_restart_and_export_overhead"]))
    write_json(root / "stage0_summary.json", report)
    (root / "训练设备与提速验收报告.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(dict(status=report["status"], selected_run=selection["selected_run"],
        blockers=blockers, summary=str(root / "stage0_summary.json")), ensure_ascii=False), flush=True)
    return 0 if report["status"] == "stage0_passed" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checks-xml")
    parser.add_argument("--require-scalar", action="store_true",
                        help="Require complete CPU and CUDA scalar-optimized confirmation runs")
    parser.add_argument("--expected-tests", type=int, default=60,
                        help="Exact number of test cases required in the single JUnit XML (default: 60)")
    parser.add_argument("--required-suite", action="append", default=[],
                        help="Additional required testcase classname or module suffix; may be repeated")
    args = parser.parse_args()
    if args.expected_tests < 1:
        parser.error("--expected-tests must be positive")
    return summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
