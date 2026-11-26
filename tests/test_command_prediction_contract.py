"""Service deployment must honor the geometry used by command-position models."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spindle_power.load_prediction.common import file_hash
from spindle_power.load_prediction.feed import FeedPlan, MachineConfig
from spindle_power.load_prediction.feed_samples import prepare_feed_features
from spindle_power.load_prediction.samples import _prepare_process_features
from spindle_power.load_prediction.training_service import TrainingService, NOMINAL_ROUTE


@pytest.fixture
def command_case(tmp_path, monkeypatch):
    machine = tmp_path / "machine.json"
    machine.write_text(json.dumps(asdict(MachineConfig(initial_position_mm=[0., 0., 0.]))), encoding="utf-8")
    nc = tmp_path / "tiny.nc"
    nc.write_text("G1 X2 F2000 S8000\n", encoding="utf-8")
    process_path = tmp_path / "process.csv"
    process = pd.DataFrame(dict(line=[1, 1], execution_instance=0, segment_mm=[0., 2.], ap_mm=1., ae_mm=2.))
    process.to_csv(process_path, index=False)
    contract = dict(position_policy="command_only_v1", process_mapping_rules_version="afc.data_supported_process_coordinates.2")
    known = dict(program_id="tiny", feed_override=1., machining_mode="synthetic", machine_settings_version="fixture",
                 diameter_mm=10., teeth=4)
    features = prepare_feed_features(nc, MachineConfig.load(machine), program_id="tiny", feed_override=1.,
        machining_mode="synthetic", machine_settings_version="fixture")
    plan = FeedPlan(features.frame, features.metadata["program_version"], features.metadata["machine_version"], False, {})
    prepared = _prepare_process_features(process, plan, {**known, "program_version": plan.program_version})
    prepared.frame["ap_mm"] = 3.
    prepared.metadata.update(contract)
    prepared.metadata["file_sha256"] = {key: file_hash(path) for key, path in
                                       (("nc", nc), ("process", process_path), ("machine", machine))}
    frame = tmp_path / "geometry.csv"
    prepared.frame.to_csv(frame, index=False)
    artifact = dict(frame=dict(path=str(frame), sha256=file_hash(frame)), metadata=prepared.metadata)
    request = dict(model_id="fixture", machine=str(machine), nc=str(nc), process=str(process_path), metadata=known)
    bundle = SimpleNamespace(config=SimpleNamespace(route=NOMINAL_ROUTE, window_points=256), metadata={"samples": [contract]},
                             execution_window_policy=lambda: "prospective_v1")
    registry = {"input_contract": contract}
    service = TrainingService(tmp_path / "service")
    monkeypatch.setattr(service, "load_model", lambda *a, **kw: (None, bundle, registry))
    return service, request, artifact, bundle, registry


def test_new_model_cannot_omit_or_forge_deployment_geometry(command_case):
    service, request, artifact, _, registry = command_case
    with pytest.raises(ValueError, match="不能退回旧工艺映射"):
        service.predict(request)
    registry.clear()  # Serialized training provenance still prevents a bypass.
    with pytest.raises(ValueError, match="不能退回旧工艺映射"):
        service.predict(request)
    wrong = deepcopy(artifact)
    wrong["frame"].pop("sha256")
    with pytest.raises(ValueError, match="SHA-256"):
        service.predict({**request, "prepared_geometry": wrong})
    wrong = deepcopy(artifact)
    wrong["metadata"]["file_sha256"]["process"] = "wrong"
    with pytest.raises(ValueError, match="未绑定当前"):
        service.predict({**request, "prepared_geometry": wrong})


def test_new_model_reaches_prediction_with_the_verified_geometry(command_case, monkeypatch):
    service, request, artifact, _, _ = command_case
    def capture(bundle, selected, **kwargs):
        assert selected.ap_mm.eq(3.).all()
        assert selected.feed_source.eq("nc_programmed_feed").all()
        assert "power_cut_w" not in selected
        raise RuntimeError("verified_geometry_reached_prediction")
    monkeypatch.setattr("spindle_power.load_prediction.models.predict_cutting_power", capture)
    with pytest.raises(RuntimeError, match="verified_geometry_reached_prediction"):
        service.predict({**request, "prepared_geometry": artifact})
    assert not list(service.output_dir.rglob("prediction.csv"))


def test_legacy_model_retains_historical_geometry(command_case, monkeypatch):
    service, request, _, bundle, registry = command_case
    registry.clear()
    bundle.metadata.clear()
    def capture(bundle, selected, **kwargs):
        assert selected.ap_mm.eq(1.).all()
        raise RuntimeError("historical_geometry_reached_prediction")
    monkeypatch.setattr("spindle_power.load_prediction.models.predict_cutting_power", capture)
    with pytest.raises(RuntimeError, match="historical_geometry_reached_prediction"):
        service.predict(request)


def test_model_version_keeps_legacy_hash_and_binds_uniform_new_contract(monkeypatch):
    from spindle_power.load_prediction.models import ModelConfig, build_load_models
    from spindle_power.load_prediction import neural
    bundle = build_load_models(ModelConfig(hidden=8, attention_heads=2, coefficient_widths=(8, 8)))
    before = bundle.current_version()
    bundle.metadata["samples"] = [{"process_mapping_rules_version": "afc.process_mapping.2"}]
    assert bundle.current_version() == before
    contract = dict(position_policy="command_only_v1", process_mapping_rules_version="afc.data_supported_process_coordinates.2")
    bundle.metadata["samples"] = [contract, dict(contract)]
    assert bundle.current_version() != before
    with monkeypatch.context() as patcher:
        patcher.setattr(neural, "version_of", lambda value: value)
        rules = bundle.current_version()["input_rules"]
        assert rules["process"] == contract["process_mapping_rules_version"]
        assert rules["position_policy"] == "command_only_v1"
    for wrong in ({}, {"process_mapping_rules_version": "afc.process_mapping.2"},
                  {**contract, "position_policy": "actual_preferred_v1"},
                  {**contract, "process_mapping_rules_version": "afc.data_supported_process_coordinates.1"}):
        bundle.metadata["samples"] = [contract, wrong]
        with pytest.raises(ValueError, match="混合或缺少"):
            bundle.current_version()
