"""Synthetic receipts and budget contracts; never launches a real benchmark."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/verification")]
import pytest
import torch

SPEC = importlib.util.spec_from_file_location("execution_recheck", ROOT / "scripts/verification/recheck_paper1_execution.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def budget(directory):
    value = dict(schema="afc.speed_budget.1", limit_seconds=5400, spent_seconds=2496.89,
                 active=None, runs=[dict(name="old_frozen_stage0", elapsed_seconds=2496.89, exit_code=0)])
    write(directory / "budget.json", value)
    return value


def test_missing_original_budget_is_never_created(tmp_path):
    with pytest.raises(ValueError, match="original"):
        AUDIT.existing_budget(tmp_path)
    assert not (tmp_path / "budget.json").exists()


@pytest.mark.parametrize("change", [dict(active={"name": "unresolved"}), dict(limit_seconds=9000),
                                     dict(spent_seconds=5400), dict(spent_seconds=0)])
def test_invalid_or_exhausted_parent_budget_fails_closed(tmp_path, change):
    value = budget(tmp_path)
    value.update(change)
    write(tmp_path / "budget.json", value)
    with pytest.raises(ValueError):
        AUDIT.existing_budget(tmp_path)


def test_plan_does_not_read_data_create_outputs_or_start_children(tmp_path, monkeypatch):
    monkeypatch.setattr(AUDIT, "preflight", lambda *_: pytest.fail("planning must not read the real snapshot"))
    monkeypatch.setattr(AUDIT.subprocess, "run", lambda *_a, **_kw: pytest.fail("planning must not run"))
    result = AUDIT.run(SimpleNamespace(execute=False))
    assert result["status"] == "planned_not_executed" and result["iterations"] == 3
    assert result["sequence"] == ["new_prospective_data", "formal_29_tests", "Q_none", "Q_cpu", "P_none", "P_cpu"]


@pytest.mark.parametrize("outcome", ["ok", "exit", "missing_charge", "unresolved"])
def test_child_requires_resolved_original_budget_charge(tmp_path, monkeypatch, outcome):
    previous = budget(tmp_path)
    command = ["python", "benchmark.py", "run", "--name", "new_trial"]
    def run(args, **kwargs):
        assert args == command and kwargs["check"] is False
        current = deepcopy(previous)
        if outcome != "missing_charge":
            current["spent_seconds"] += 10.
            current["runs"].append(dict(name="new_trial", elapsed_seconds=10., exit_code=0 if outcome != "exit" else 1))
        if outcome == "unresolved":
            current["active"] = {"name": "new_trial"}
        write(tmp_path / "budget.json", current)
        return SimpleNamespace(returncode=1 if outcome == "exit" else 0)
    monkeypatch.setattr(AUDIT.subprocess, "run", run)
    if outcome == "ok":
        result = AUDIT.execute_child(command, tmp_path, tmp_path / "log.txt", previous)
        assert result["spent_seconds"] == 2506.89 and result["runs"][0] == previous["runs"][0]
    else:
        with pytest.raises(ValueError):
            AUDIT.execute_child(command, tmp_path, tmp_path / "log.txt", previous)


def test_orchestration_cost_does_not_double_count_children(tmp_path, monkeypatch):
    before = budget(tmp_path)
    current = deepcopy(before)
    current["runs"].append(dict(name="prefix_prepare", elapsed_seconds=10., exit_code=0))
    current["spent_seconds"] += 10.
    write(tmp_path / "budget.json", current)
    monkeypatch.setattr(AUDIT.time, "perf_counter", lambda: 113.)
    result = AUDIT.charge_overhead(tmp_path, before, "prefix", 100., ["prefix_prepare"])
    assert result["spent_seconds"] == before["spent_seconds"] + 13.
    assert result["runs"][-1]["elapsed_seconds"] == 3.
    assert result["runs"][0] == before["runs"][0] and not (tmp_path / "budget.lock").exists()
    monkeypatch.setattr(AUDIT.time, "perf_counter", lambda: 115.)
    again = AUDIT.charge_overhead(tmp_path, before, "prefix", 100., ["prefix_prepare"])
    assert again["spent_seconds"] == before["spent_seconds"] + 15.
    assert again["runs"][-1]["elapsed_seconds"] == 2.


def test_orchestration_rejects_unrelated_concurrent_budget_entries(tmp_path, monkeypatch):
    before = budget(tmp_path)
    current = deepcopy(before)
    current["runs"].append(dict(name="unrelated", elapsed_seconds=3., exit_code=0))
    write(tmp_path / "budget.json", current)
    with pytest.raises(ValueError, match="concurrent"):
        AUDIT.charge_overhead(tmp_path, before, "prefix", 0., ["prefix_prepare"])
    assert AUDIT.read_json(tmp_path / "budget.json") == current


def checks_fixture(tmp_path):
    suite = tmp_path / "suite.py"
    suite.write_text("# synthetic suite evidence\n", encoding="utf-8")
    test = AUDIT.reference(suite)
    xml = tmp_path / "trial_checks.xml"
    xml.write_text('<testsuites><testsuite>' + ''.join(
        f'<testcase classname="test_prospective_residual_windows" name="case_{i}" />' for i in range(29)) +
        '</testsuite></testsuites>', encoding="utf-8")
    value = dict(schema="paper1.prospective_checks.1", status="passed", passed=True, return_code=0,
        source={"fixture.py": "source"}, source_unchanged=True, candidate_source_used=False,
        formal_root=str(ROOT.resolve()), suite=test, junit=AUDIT.reference(xml),
        imported_modules={"spindle_power.load_prediction." + name: str((ROOT / "src/spindle_power/load_prediction" / (name + ".py")).resolve())
                          for name in ("neural", "training", "training_service")})
    receipt = tmp_path / "trial_checks_checks.json"
    write(receipt, value)
    return value, test, receipt, xml


def test_fresh_formal_29_test_junit_is_accepted(tmp_path):
    value, test, _, _ = checks_fixture(tmp_path)
    result = AUDIT.formal_checks(tmp_path, "trial", value["source"], test)
    assert result["checks"]["counts"] == dict(tests=29, failures=0, errors=0, skipped=0)


@pytest.mark.parametrize("change", ["candidate", "source", "import", "skipped", "missing_test", "old_xml"])
def test_candidate_or_stale_junit_cannot_release_execution(tmp_path, change):
    value, test, receipt, xml = checks_fixture(tmp_path)
    if change == "candidate":
        value["candidate_source_used"] = True
    elif change == "source":
        value["source"] = {"fixture.py": "old source"}
    elif change == "import":
        value["imported_modules"]["spindle_power.load_prediction.neural"] = "D:/candidate/neural.py"
    else:
        contents = xml.read_text(encoding="utf-8")
        if change == "skipped":
            contents = contents.replace('name="case_0" />', 'name="case_0"><skipped /></testcase>')
        elif change == "missing_test":
            contents = contents.replace('<testcase classname="test_prospective_residual_windows" name="case_0" />', '')
        else:
            contents += " "
        xml.write_text(contents, encoding="utf-8")
        if change != "old_xml":
            value["junit"] = AUDIT.reference(xml)
    write(receipt, value)
    with pytest.raises(ValueError):
        AUDIT.formal_checks(tmp_path, "trial", {"fixture.py": "source"}, test)


def measurement(tmp_path, name, loss="quantiles", cache="none"):
    directory = tmp_path / "runs" / name
    directory.mkdir(parents=True)
    source = {"fixture.py": "source"}
    states, rows = {}, []
    for stage in ("main", "residual"):
        initial = {"weight": torch.tensor([1., 2.])}
        first = {"weight": torch.tensor([.9, 1.9])}
        states[stage] = dict(initial_weights=initial, initial_rng=dict(torch_rng=torch.get_rng_state()),
                             gradients={"weight": torch.tensor([.1, .1])}, weights=first, train_loss=2., validation_loss=1.)
        for epoch in (1, 2, 3):
            path = directory / "epochs" / stage / f"epoch_{epoch:04d}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            saved = dict(identity=dict(source=source, data_sha256="data", loss=loss, cache=cache),
                state=dict(epoch=epoch, optimizer_updates=epoch, completed=epoch == 3, weights=first,
                           optimizer={"state": {0: {"step": torch.tensor(epoch), "exp_avg": torch.tensor([.1, .1])}}}))
            torch.save(saved, path)
            rows.append(dict(stage=stage, epoch=epoch, train_loss=2., validation_loss=1., train_seconds=2.,
                             validation_seconds=1., checkpoint_seconds=.5, total_seconds=3.5))
        write(directory / (stage + "_setup.json"), dict(preparation_seconds=1., warmup_seconds=2.))
    torch.save(states, directory / "first_update.pt")
    stat = None if cache == "none" else dict(max_cache_bytes=1073741824, device_cache_bytes=0, batch_count=2)
    value = dict(status="measured", scientific_accuracy_verified=False, data_sha256="data",
        args=dict(name=name, device="cpu", threads=10, main_points=32768, batch_windows=128,
                  batch_strategy="padded", cache=cache, cache_max_bytes=1073741824, stage="both", loss=loss, iterations=3,
                  window_policy="prospective_v1"),
        execution=dict(device="cpu", cpu_threads=10, cpu_interop_threads=1), cache_stats={
            stage: dict(train=stat, validation=stat) for stage in ("main", "residual")}, epochs=rows,
        preparation_seconds=dict(main=1., residual=1.), setup_seconds=2., total_process_seconds=30., memory_peak_bytes=1024)
    write(directory / "result.json", value)
    return directory, value, states, source


@pytest.mark.parametrize("loss", ["quantiles", "point"])
def test_same_loss_same_data_rng_and_actual_archived_adam_are_checked(tmp_path, loss):
    _, _, _, source = measurement(tmp_path, "control", loss, "none")
    measurement(tmp_path, "cached", loss, "cpu")
    control = AUDIT.measure_evidence(tmp_path, "control", loss, "none", "data", source)
    cached = AUDIT.measure_evidence(tmp_path, "cached", loss, "cpu", "data", source)
    pair = AUDIT.compare_pair(control, cached)
    assert pair["passed"]
    json.dumps(pair, allow_nan=False)
    assert all(row["initial_rng_exact"] and row["one_adam_weights"]["exact"] for row in pair["stages"].values())


@pytest.mark.parametrize("change", ["loss", "data", "source", "updates", "adam", "reported_weights"])
def test_wrong_or_partial_measurement_fails_closed(tmp_path, change):
    directory, raw, states, source = measurement(tmp_path, "test")
    if change in ("loss", "data"):
        if change == "loss":
            raw["args"]["loss"] = "point"
        else:
            raw["data_sha256"] = "other"
        write(directory / "result.json", raw)
    elif change == "reported_weights":
        states["main"]["weights"]["weight"] += 1.
        torch.save(states, directory / "first_update.pt")
    else:
        path = directory / "epochs/main/epoch_0001.pt"
        saved = torch.load(path, weights_only=True)
        if change == "source":
            saved["identity"]["source"] = {}
        elif change == "updates":
            saved["state"]["optimizer_updates"] = 0
        else:
            saved["state"]["optimizer"] = {}
        torch.save(saved, path)
    with pytest.raises(ValueError):
        AUDIT.measure_evidence(tmp_path, "test", "quantiles", "none", "data", source)


@pytest.mark.parametrize("field", ["initial_rng", "initial_weights", "gradients", "weights", "train_loss"])
def test_equivalence_uses_original_tolerances_and_rejects_changed_initialization(tmp_path, field):
    _, _, states, _ = measurement(tmp_path, "test")
    changed = deepcopy(states)
    if field == "initial_rng":
        changed["main"][field]["torch_rng"][0] ^= 1
    elif field == "train_loss":
        changed["main"][field] += 1.
    else:
        changed["main"][field]["weight"] += 1.
    result = AUDIT.compare_pair(dict(states=states), dict(states=changed))
    assert not result["passed"]
    from summarize_load_speed import TOLERANCES
    assert result["tolerances"] == TOLERANCES


@pytest.mark.parametrize("equivalent", [True, False])
def test_profile_requires_new_checks_and_all_six_bounded_children(tmp_path, monkeypatch, equivalent):
    from spindle_power.load_prediction import training_checkpoint
    stage0, output = tmp_path / "stage0", tmp_path / "new_recheck"
    before = budget(stage0)
    evidence_path = tmp_path / "evidence.json"
    write(evidence_path, {"synthetic": True})
    ref = AUDIT.reference(evidence_path)
    source = {"fixture.py": "source"}
    state = dict(stage0=stage0, budget=before, source=source, test=ref, window_refs=[],
        bindings=dict(candidate_snapshot=ref, window_audit=ref, parent_execution_profile=ref, engineering_fold_model=ref),
        scripts={"synthetic": ref})
    monkeypatch.setattr(AUDIT, "preflight", lambda _: state)
    monkeypatch.setattr(training_checkpoint, "source_fingerprints", lambda: source)
    calls = []
    def child(command, root, log, previous):
        name = command[command.index("--name") + 1]
        calls.append(command[4])
        current = deepcopy(previous)
        current["runs"].append(dict(name=name, elapsed_seconds=.01, exit_code=0))
        current["spent_seconds"] += .01
        write(root / "budget.json", current)
        if name.endswith("_prepare"):
            write(root / (name + "_prepare_stats.json"), dict(status="prepared", memory_peak_bytes=1024, elapsed_seconds=.01))
        return current
    monkeypatch.setattr(AUDIT, "execute_child", child)
    monkeypatch.setattr(AUDIT, "prepared_data", lambda *_: (dict(elapsed_seconds=.01, provenance=dict(initial_versions={})),
                                                           dict(manifest=ref, full=ref)))
    tests_called = []
    def checks(*_args):
        tests_called.append(True)
        assert calls == ["prepare", "check-prospective"]
        return dict(receipt=ref, junit=ref, checks=dict(passed=True))
    monkeypatch.setattr(AUDIT, "formal_checks", checks)
    cost = dict(median=2., minimum=1., maximum=3.)
    summary = dict(setup_seconds=1., stages={stage: dict(preparation=cost, total_seconds=cost)
                                           for stage in ("main", "residual")}, artifacts=[])
    monkeypatch.setattr(AUDIT, "measure_evidence", lambda *_: dict(summary=summary))
    monkeypatch.setattr(AUDIT, "compare_pair", lambda *_: dict(passed=equivalent))
    args = SimpleNamespace(execute=True, output_dir=output, candidate_snapshot=tmp_path / "snapshot/snapshot.json",
                           window_audit=evidence_path, execution_profile=evidence_path, stage0_root=stage0)
    result = AUDIT.run(args)
    assert tests_called == [True] and calls == ["prepare", "check-prospective", "run", "run", "run", "run"]
    assert result["passed"] is equivalent
    after = AUDIT.read_json(stage0 / "budget.json")
    assert after["runs"][0] == before["runs"][0] and after["spent_seconds"] >= before["spent_seconds"]
    profile_path = output / "execution_profile.json"
    if equivalent:
        profile = AUDIT.read_json(profile_path)
        assert profile["schema"] == "paper1.load_execution_profile.1" and profile["status"] == "prospective_rechecked"
        assert profile["selection_evidence"] == AUDIT.reference(output / "report.json")
        assert profile["source_fingerprints"] == source
        assert profile["training_config"]["residual_window_policy"] == "prospective_v1"
    else:
        assert not profile_path.exists()


@pytest.mark.parametrize("mode", ["formal", "source_changed", "candidate"])
def test_bounded_check_worker_receipt_cannot_claim_candidate_or_changed_source(tmp_path, monkeypatch, mode):
    import benchmark_load_execution as benchmark
    from spindle_power.load_prediction import neural, training_checkpoint
    calls = []
    def fingerprint():
        calls.append(True)
        return {"fixture.py": "changed" if mode == "source_changed" and len(calls) > 1 else "source"}
    monkeypatch.setattr(training_checkpoint, "source_fingerprints", fingerprint)
    def pytest_run(arguments):
        path = Path(arguments[arguments.index("--junitxml") + 1])
        path.write_text("<testsuites />", encoding="utf-8")
        return 0
    monkeypatch.setattr(pytest, "main", pytest_run)
    args = SimpleNamespace(output_root=str(tmp_path), name="fresh")
    if mode == "candidate":
        monkeypatch.setattr(neural, "__file__", "D:/candidate/src/spindle_power/load_prediction/neural.py")
        with pytest.raises(ValueError, match="formal workspace"):
            benchmark.check_prospective_worker(args)
        assert not (tmp_path / "fresh_checks.json").exists()
    else:
        code = benchmark.check_prospective_worker(args)
        receipt = AUDIT.read_json(tmp_path / "fresh_checks.json")
        assert (code == 0) is (mode == "formal") and receipt["passed"] is (mode == "formal")
        assert receipt["candidate_source_used"] is False
        assert receipt["junit"] == AUDIT.reference(tmp_path / "fresh.xml")


def test_check_worker_keeps_service_export_paths_short(tmp_path, monkeypatch):
    import benchmark_load_execution as benchmark
    output = tmp_path / ("deep_" * 16)
    output.mkdir()
    observed = {}
    def pytest_run(arguments):
        temporary = Path(arguments[arguments.index("--basetemp") + 1])
        export = temporary / "test_public_service_uses_share0/service/predictions" / ("a" * 32) / "prediction.csv"
        export.parent.mkdir(parents=True)
        export.write_text("value\n1\n", encoding="utf-8")
        observed.update(temporary=temporary, export=export)
        Path(arguments[arguments.index("--junitxml") + 1]).write_text("<testsuites />", encoding="utf-8")
        return 0
    monkeypatch.setattr(pytest, "main", pytest_run)
    assert benchmark.check_prospective_worker(SimpleNamespace(output_root=str(output), name="long_path")) == 0
    receipt = AUDIT.read_json(output / "long_path_checks.json")
    assert Path(receipt["temporary_directory"]) == observed["temporary"].resolve()
    assert observed["temporary"].parent == Path(receipt["temporary_root"])
    assert not observed["temporary"].is_relative_to(output)
    assert observed["export"].read_text(encoding="utf-8") == "value\n1\n"
    if os.name == "nt":
        assert len(str(observed["export"]).encode("utf-16-le")) // 2 < 260
