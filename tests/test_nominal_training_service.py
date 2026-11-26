"""Small synthetic acceptance; these scores are never machining results."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from verify_unified_load_prediction import synthetic_records, SMALL_CANDIDATE
from spindle_power.load_prediction import training_service as module
from spindle_power.load_prediction.training_service import TrainingService, NOMINAL_ROUTE, protocol


def candidate():
    result = {key: deepcopy(SMALL_CANDIDATE[key]) for key in ("id", "load_model", "load_training")}
    result["load_model"]["route"] = NOMINAL_ROUTE
    result["load_training"]["device"] = "cpu"
    return result


@pytest.fixture
def tiny(monkeypatch):
    torch.set_num_threads(1)
    _, loads = synthetic_records((1, 2), points=4)
    for sample in loads:
        sample.frame["entry_angle_rad"] = 0.
        sample.frame["exit_angle_rad"] = 0.
    folds = [dict(fold="F1", train_rounds=[1], validation_rounds=[2]),
             dict(fold="F2", train_rounds=[2], validation_rounds=[1])]
    monkeypatch.setattr(module, "protocol_folds", lambda: folds)
    def forbidden(*args, **kwargs):
        raise AssertionError("nominal route must not create, optimize or infer a feed network")
    monkeypatch.setattr(module, "train_feed_cached", forbidden)
    monkeypatch.setattr("spindle_power.load_prediction.feed_model.predict_actual_feed", forbidden)
    monkeypatch.setattr("spindle_power.load_prediction.feed_training.train_feed_fold", forbidden)
    return loads


def test_nominal_protocol_has_no_feed_and_18_features():
    result = protocol(NOMINAL_ROUTE)
    assert result["feed_columns"] == []
    assert len(result["path_columns"]) == 18
    assert "feed" not in result["stage_order"]


def test_nominal_development_full_publication_without_feed(tiny, tmp_path):
    service = TrainingService(tmp_path / "service")
    prepared = service.save_prepared([], tiny, source_snapshot={"route": NOMINAL_ROUTE}, synthetic=True)
    request = dict(job_id="nominal_acceptance", mode="load", snapshot_id=prepared["snapshot_id"],
                   candidates=[candidate()], analysis_mode="numbered_round_pooled_tool", analysis_tool_group="synthetic_only")
    preview = service.preview(prepared["snapshot_id"], [candidate()], mode="load")
    assert preview["feed_network_used"] is False
    result = service.execute(request)
    assert result["status"] == "frozen"
    destination = Path(result["directory"])
    assert not (destination / "feed.pt").exists()
    feed, bundle, meta = service.load_model(result["model_id"], allow_synthetic=True)
    assert feed is None and bundle.config.route == NOMINAL_ROUTE
    assert len(meta["protocol"]["path_columns"]) == 18
    development = json.loads(Path(result["development"]).read_text(encoding="utf-8"))
    assert development["feed_network_used"] is False
    assert set(development["full_epochs"]) == {"main", "residual"}
    for fold in development["candidates"][0]["folds"]:
        assert len(fold["common_comparison"]["comparisons"]) == 5
        assert fold["fixed_physical"]["nominal_coverage"] is None
        assert fold["with_memory"]["memory_analysis"]["initial_memory"] == "empty"
        assert len(fold["without_memory"]["records"]) == 5
    assert not list(service.output_dir.rglob("feed.pt"))


def test_nominal_mode_route_and_feed_source_must_match(tiny, tmp_path):
    service = TrainingService(tmp_path / "service")
    prepared = service.save_prepared([], tiny, source_snapshot={}, synthetic=True)
    request = dict(job_id="bad", mode="load", snapshot_id=prepared["snapshot_id"], candidates=[candidate()])
    with pytest.raises(ValueError, match="feed"):
        service.execute({**request, "source_feed_job_id": "legacy"})
    wrong = candidate()
    wrong["load_model"].pop("route")
    with pytest.raises(ValueError, match="路线"):
        service.execute({**request, "candidates": [wrong]})


def test_nominal_preview_never_needs_feed_samples(tiny):
    result = module.run_development(tiny, candidates=[candidate()], execute=False)
    assert result["route"] == NOMINAL_ROUTE
    assert all(row["stages"] == ["main", "residual"] for row in result["phases"])


def test_public_nominal_prediction_reads_nc_and_process_without_feed(tiny, tmp_path, monkeypatch):
    from spindle_power.load_prediction.feed import MachineConfig
    from spindle_power.load_prediction.models import ModelConfig
    from spindle_power.load_prediction.training import TrainingConfig, train_two_stage
    config = candidate()
    bundle = train_two_stage(tiny[:5], model_config=ModelConfig(**config["load_model"]),
        config=TrainingConfig(**config["load_training"]), execute=True)
    service = TrainingService(tmp_path / "predict_service")
    monkeypatch.setattr(service, "load_model", lambda *a, **kw: (None, bundle, {}))
    machine = tmp_path / "machine.json"
    machine.write_text(json.dumps(asdict(MachineConfig(initial_position_mm=[0., 0., 0.],
        work_offsets_mm={"G54": [0., 0., 0.]}, spindle_wait_s=0., optional_stop="skip"))), encoding="utf-8")
    nc = tmp_path / "tiny.nc"
    nc.write_text("G21 G17 G90 G94\nG1 X2 F2000 S8000\nG1 Y2\n", encoding="utf-8")
    process = tmp_path / "process.csv"
    pd.DataFrame([dict(line=line, execution_instance=0, segment_mm=position, ap_mm=1., ae_mm=2.)
                  for line in (2, 3) for position in (0., 2.)]).to_csv(process, index=False)
    known = {**tiny[0].metadata, "program_id": "tiny", "feed_override": 1.,
             "machining_mode": "synthetic", "machine_settings_version": "fixture"}
    result = service.predict(dict(model_id="fixture", machine=str(machine), nc=str(nc),
                                  process=str(process), metadata=known, allow_synthetic=True))
    frame = pd.DataFrame(result["columns"])
    assert len(frame) > 0 and np.isfinite(frame.cutting_power_w).all()
    assert frame.feed_source.eq("nc_programmed_feed").all()
    assert np.allclose(frame.feed_input_mm_min, frame.feed_program_mm_min)
    assert "feed_actual_mm_min" not in frame


def launcher_module():
    spec = importlib.util.spec_from_file_location("nominal_launch", ROOT / "scripts/verification/prepare_paper1_launch.py")
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    return launch


def test_nominal_ready_binds_device_identity_and_refuses_changed_device(tmp_path, monkeypatch):
    launch = launcher_module()
    identity = {"device": "cuda", "gpu": "synthetic_A"}
    service = SimpleNamespace(output_dir=tmp_path / "service", _run_identity=lambda _: deepcopy(identity))
    monkeypatch.setattr(launch, "runtime_identity", lambda: {})
    monkeypatch.setattr(launch, "validate_numerical", lambda *a: {"trained": False})
    monkeypatch.setattr(launch, "TrainingService", lambda *_: service)
    monkeypatch.setattr(launch.subprocess, "Popen", lambda *a, **kw: pytest.fail("must reject before spawn"))
    result = launch.save_ready(service, {"mode": "load", "snapshot_id": "a" * 64}, tmp_path / "ready")
    assert launch.read_json(result["ready"])["run_identity"] == identity
    identity["gpu"] = "synthetic_B"
    with pytest.raises(ValueError, match="执行身份"):
        launch.launch(result["ready"])


def test_nominal_plan_resolves_parent_and_historical_source_relative_to_plan(tmp_path):
    launch = launcher_module()
    launch.write_new(tmp_path / "manifest.json", {"records": []})
    launch.write_new(tmp_path / "launch_plan.json", dict(schema="paper1.launch_plan.1", manifest="manifest.json",
        source_prepared_service_output="parent_service", historical_consumer_source="frozen_source"))
    plan = launch.load_plan(tmp_path / "launch_plan.json")
    assert Path(plan["source_prepared_service_output"]) == tmp_path / "parent_service"
    assert Path(plan["historical_consumer_source"]) == tmp_path / "frozen_source"


def test_equal_historical_source_names_do_not_require_raw_transfer(tmp_path, monkeypatch):
    launch = launcher_module()
    records = [dict(record_id=f"record_{i:03d}", status="active", **{
        key: f"D:/untransferred_originals/{i}/{key}.dat" for key in ("nc", "measurement", "machine", "process")})
        for i in range(450)]
    manifest = tmp_path / "manifest.json"
    launch.write_new(manifest, {"records": records})
    snapshot_id = "a" * 64
    parent = tmp_path / "parent"
    historical = tmp_path / "historical"
    snapshot = dict(snapshot_id=snapshot_id, source_snapshot={"records": records})
    launch.write_new(parent / "snapshots" / snapshot_id / "snapshot.json", snapshot)
    def resolve(value):
        if str(value).startswith("D:/untransferred_originals/"):
            pytest.fail("equal provenance paths must not require raw-data relocation")
        return Path(value)
    monkeypatch.setattr(launch, "artifact_resolver", lambda _: SimpleNamespace(resolve=resolve, reference=None))
    service = SimpleNamespace(output_dir=parent, load_prepared=lambda _: ([], [], snapshot),
                              _validate_prepared=lambda *a, **kw: None)
    monkeypatch.setattr(launch, "TrainingService", lambda _: service)
    result = launch.verify_nominal_parent(dict(manifest=str(manifest), source_prepared_service_output=str(parent),
        source_prepared_snapshot_id=snapshot_id, historical_consumer_source=str(historical)))
    assert result["records_verified"] == 450 and result["raw_data_read"] is False


def test_formal_nominal_cpu_gate_accepts_explicit_device_and_rejects_missing(tiny, tmp_path, monkeypatch):
    service = TrainingService(tmp_path / "cpu_gate")
    monkeypatch.setattr(service, "load_prepared", lambda _: ([], tiny, {"synthetic": False}))
    monkeypatch.setattr(service, "_validate_prepared", lambda *a, **kw: None)
    class ReachedRun(Exception):
        pass
    monkeypatch.setattr(service, "_prepare_run", lambda _: (_ for _ in ()).throw(ReachedRun()))
    request = dict(mode="load", snapshot_id="fixture", candidates=[candidate()])
    with pytest.raises(ReachedRun):
        service.execute(request)
    request["candidates"][0]["load_training"].pop("device")
    with pytest.raises(ValueError, match="显式"):
        service.execute(request)


def test_nominal_cpu_numerical_gate_preserves_explicit_device_requirement(tiny, monkeypatch):
    launch = launcher_module()
    for sample in tiny:
        sample.metadata["sample_rules_version"] = launch.SAMPLE_RULES_VERSION
    service = SimpleNamespace(load_prepared=lambda _: ([], tiny, {"synthetic": False}),
        _validate_prepared=lambda *a, **kw: None)
    monkeypatch.setattr(launch, "protocol_folds", lambda: [dict(fold="F1", train_rounds=[1], validation_rounds=[2])])
    monkeypatch.setattr(launch, "validate_paper1_fixed_conditions", lambda _: {"fixture": True})
    request = dict(mode="load", snapshot_id="fixture", candidates=[candidate()])
    report = launch.validate_nominal_numerical(service, request)
    assert report["trained"] is False and report["load_records"] == 10
    request["candidates"][0]["load_training"].pop("device")
    with pytest.raises(ValueError, match="显式"):
        launch.validate_nominal_numerical(service, request)


def test_cpu_thread_identity_and_recovery_bind_explicit_configuration(tmp_path):
    from spindle_power.load_prediction.common import save_json
    from spindle_power.load_prediction.load_device import configure_load_device, load_execution_identity, request_cpu_threads
    from spindle_power.load_prediction.training import TrainingConfig
    old_threads = torch.get_num_threads()
    try:
        config = candidate()
        config["load_training"]["cpu_threads"] = 4
        request = dict(mode="load", snapshot_id="fixture", candidates=[config])
        service = TrainingService(tmp_path / "thread_identity")
        identity = service._run_identity(request)
        assert identity["environment"]["load_execution"]["cpu"]["cpu_threads"] == 4
        assert torch.get_num_threads() == 4
        torch.set_num_threads(2)
        with pytest.raises(ValueError, match="实际执行线程"):
            load_execution_identity("cpu", cpu_threads=4)
        assert configure_load_device("cpu", cpu_threads=4)[1]["cpu_threads"] == 4
        assert request_cpu_threads({**request, "mode": "feed"}) == (None, None)
        save_json(service.output_dir / "jobs/source/state.json", {"status": "interrupted"})
        save_json(service.output_dir / "runs/source/run_fingerprint.json", identity)
        assert service.validate_recovery({**request, "resume_from_job_id": "source"}) is not None
        changed = deepcopy(request)
        changed["candidates"][0]["load_training"]["cpu_threads"] = 2
        with pytest.raises(ValueError, match="恢复指纹不一致"):
            service.validate_recovery({**changed, "resume_from_job_id": "source"})
        for invalid in (0, True, 1.5):
            with pytest.raises(ValueError, match="cpu_threads"):
                TrainingConfig(cpu_threads=invalid).validate()
        with pytest.raises(ValueError, match="CPU设备"):
            TrainingConfig(device="cuda", cpu_threads=4).validate()
    finally:
        torch.set_num_threads(old_threads)


def test_real_background_worker_uses_requested_cpu_threads_without_training(tmp_path):
    """A genuine worker must prove its runtime before a missing snapshot stops it."""
    import time
    from spindle_power.load_prediction.common import save_json
    from spindle_power.load_prediction.job_service import JobService
    service = TrainingService(tmp_path / "actual_worker")
    config = candidate()
    config["load_training"].update(cpu_threads=4, cpu_interop_threads=1)
    request = dict(job_id="cputhreadprobe", mode="load", snapshot_id="0" * 64,
                   candidates=[config], service_output=str(service.output_dir))
    jobs = JobService(service)
    directory = jobs.directory / request["job_id"]
    try:
        save_json(directory / "request.json", request)
        save_json(directory / "state.json", dict(job_id=request["job_id"], status="queued", created_at="synthetic_probe"))
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            state = jobs.get(request["job_id"])
            if state["status"] == "failed":
                break
            time.sleep(.1)
        assert state["status"] == "failed", state
        execution = state["load_execution"]["cpu"]
        assert execution["device"] == "cpu"
        assert execution["cpu_threads"] == execution["requested_cpu_threads"] == 4
        assert execution["cpu_interop_threads"] == execution["requested_cpu_interop_threads"] == 1
        assert "load_prepared" in state["traceback"]
        assert not list((service.output_dir / "runs").glob("*"))
        save_json(tmp_path / "actual_worker_thread_receipt.json", dict(
            schema="afc.cpu_worker_thread_probe.1", actual_subprocess=True,
            training_started=False, expected_stop="missing_synthetic_snapshot_before_training", state=state))
    finally:
        jobs.shutdown()
