"""Synthetic interface regressions only; no real record qualification/training."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spindle_power.load_prediction.base_qualification import embedded_json_artifact, verify_label_evidence
from spindle_power.load_prediction.base_reference import idle_platform_quality
from spindle_power.load_prediction.common import file_hash, save_json
from spindle_power.load_prediction.feed_identity import feed_sample_identity, feed_source_fingerprints
from spindle_power.load_prediction.measurement import Measurement
from spindle_power.load_prediction.record_idle import (LABEL_SOURCES, MASK_VERSION, MODE, POLICY_VALUES, SCHEMA, VERSION,
    prepare_record_idle_power, verify_prepared_record_idle)
from spindle_power.load_prediction.samples import SampleSet, prepare_load_samples
from spindle_power.load_prediction.training_service import TrainingService


@pytest.fixture
def case(tmp_path):
    raw, nc = tmp_path / "synthetic.csv", tmp_path / "synthetic.nc"
    raw.write_text("synthetic fixture only, no machine acquisition", encoding="utf-8")
    nc.write_text("G0 X0 S8000\nG1 X10 F600\n", encoding="utf-8")
    meta = dict(record_id="R01_synthetic", round_id=1, machine_id="synthetic",
                dataset_role="development", resolved_files=dict(measurement=str(raw), nc=str(nc)),
                file_sha256=dict(measurement=file_hash(raw), nc=file_hash(nc),
                                 process=file_hash(nc), machine=file_hash(nc)))
    selection_path = tmp_path / "selection.json"
    selection = dict(record_id=meta["record_id"], measurement_sha256=file_hash(raw),
                     platform_ranges=[[1000, 2000]], commanded_rpm=8000.)
    save_json(selection_path, selection)
    artifact = embedded_json_artifact(selection_path)
    audit = dict(record_id=meta["record_id"], file_sha256=meta["file_sha256"], status="qualified", sample_count=6000,
        commanded_rpm=8000., idle_power_w=200., platform_ranges=selection["platform_ranges"],
        derivative_context_ranges=[[0, 2000]],
        no_contact_verified=True, same_speed_verified=True, continuity_verified=True,
        selection_artifact=artifact, selection_sha256=artifact["sha256"], qualification_reasons=[])
    domain = dict(machine_id="synthetic", power_channel="spindle_power_w", scope="record_local_constant_speed_only")
    report = dict(schema=SCHEMA, mode=MODE, version=VERSION, domain=domain, status="qualified", interface_version=MASK_VERSION,
        eligible_for_load_training=True, provenance="synthetic_fixture_only", record_audits=[audit],
        production_source_sha256={name: file_hash(ROOT / "src/spindle_power/load_prediction" / name) for name in LABEL_SOURCES},
        selection_policy=dict(version="synthetic_frozen_rule", uses_power_for_selection=False, **POLICY_VALUES))
    evidence = dict(mode=MODE, version=VERSION, domain=domain, validated=True, applicable=True,
                    report_path=str(tmp_path / "report.json"))
    time = np.arange(6000) * .001
    record = Measurement(str(raw), time, dict(spindle_speed_rpm=np.full(len(time), 8000.),
        spindle_power_w=np.where(time < 2., 200., 350.)), np.ones(len(time), bool),
        dict(sample_period_s=.001), source_indices=np.arange(len(time)))

    def publish():
        policy_path = tmp_path / "policy.json"
        save_json(policy_path, report["selection_policy"])
        report["selection_policy_artifact"] = embedded_json_artifact(policy_path)
        save_json(evidence["report_path"], report)
        evidence["report_sha256"] = file_hash(evidence["report_path"])

    publish()
    return SimpleNamespace(record=record, meta=meta, evidence=evidence, report=report, audit=audit, publish=publish)


def test_record_route_needs_no_frozen_dynamic_model(case):
    assert verify_label_evidence(case.evidence, [case.meta])["valid"]
    with patch("spindle_power.load_prediction.base_power.predict_base_from_measured_speed", side_effect=AssertionError("dynamic model must not run")):
        result = prepare_record_idle_power(case.record, case.evidence, case.meta)
    assert result["valid"].sum() == 4000
    assert not result["valid"][:2000].any()
    np.testing.assert_array_equal(case.record.values["spindle_power_w"][result["valid"]] - result["power_w"][result["valid"]], 150.)
    assert result["label_metadata"]["prediction_input"] is False


def test_pending_certificate_cannot_be_upgraded_by_prepare(case):
    case.audit["status"] = "pending"
    case.publish()
    assert not verify_label_evidence(case.evidence, [case.meta])["valid"]
    with pytest.raises(ValueError, match="尚未完成"):
        prepare_record_idle_power(case.record, case.evidence, case.meta)


def test_excluded_record_retains_explicit_empty_mask(case):
    case.audit.update(status="excluded", qualification_reasons=["no_same_speed_platform"])
    for key in ("idle_power_w", "platform_ranges", "selection_artifact", "selection_sha256"):
        case.audit.pop(key)
    case.publish()
    assert verify_label_evidence(case.evidence, [case.meta])["valid"]
    result = prepare_record_idle_power(case.record, case.evidence, case.meta)
    assert not result["valid"].any()
    assert result["formal_training_release"] is True  # A complete certificate includes its exclusions.
    assert np.isnan(result["power_w"]).all()
    sample = SampleSet(pd.DataFrame(dict(valid_main=[False], valid_residual=[False])),
                       {**case.meta, "record_idle_label": result["label_metadata"]})
    verify_prepared_record_idle(sample, case.evidence)
    sample.frame.loc[0, "valid_residual"] = True
    with pytest.raises(ValueError, match="排除记录"):
        verify_prepared_record_idle(sample, case.evidence)


def test_exclusion_requires_reason_and_exact_measurement_binding(case):
    case.audit["status"] = "excluded"
    case.publish()
    assert not verify_label_evidence(case.evidence, [case.meta])["valid"]
    case.audit["qualification_reasons"] = ["missing_platform"]
    case.audit["file_sha256"] = {**case.audit["file_sha256"], "measurement": "0" * 64}
    case.publish()
    assert not verify_label_evidence(case.evidence, [case.meta])["valid"]


def test_embedded_selection_and_report_hashes_are_enforced(case):
    case.audit["selection_artifact"]["json_text"] += " "
    case.publish()
    assert not verify_label_evidence(case.evidence, [case.meta])["valid"]
    Path(case.evidence["report_path"]).write_text("{}", encoding="utf-8")
    assert "SHA-256" in " ".join(verify_label_evidence(case.evidence, [case.meta])["issues"])


@pytest.mark.parametrize("kind", ["process", "machine"])
def test_all_four_source_hashes_are_bound(case, kind):
    case.audit["file_sha256"] = {**case.audit["file_sha256"], kind: "0" * 64}
    case.publish()
    assert not verify_label_evidence(case.evidence, [case.meta])["valid"]


def test_policy_and_source_interface_are_bound(case):
    case.report["production_source_sha256"]["record_idle.py"] = "0" * 64
    case.publish()
    assert not verify_label_evidence(case.evidence, [case.meta])["valid"]


def test_full_sample_count_and_platform_context_are_rechecked(case):
    case.audit["sample_count"] -= 1
    case.publish()
    with pytest.raises(ValueError, match="样本数量"):
        prepare_record_idle_power(case.record, case.evidence, case.meta)
    case.audit["sample_count"] += 1
    case.record.values["spindle_speed_rpm"][:2000] += np.linspace(-79., 79., 2000)
    case.publish()
    # Entire platform remains inside the allowed RPM band; a faster ramp fails
    # the independently required smoothed derivative check.
    case.record.values["spindle_speed_rpm"][1000:2000] = 8000. + np.linspace(-79., 79., 1000)
    with pytest.raises(ValueError, match="平滑转速导数"):
        prepare_record_idle_power(case.record, case.evidence, case.meta)


def test_two_sample_tail_is_excluded_without_filter_crash(case):
    case.record.valid[-3] = False
    result = prepare_record_idle_power(case.record, case.evidence, case.meta)
    assert not result["valid"][-3:].any()


def test_platform_value_and_quality_are_recomputed_without_power_selection(case):
    case.audit["idle_power_w"] = 199.
    case.publish()
    with pytest.raises(ValueError, match="中位功率"):
        prepare_record_idle_power(case.record, case.evidence, case.meta)
    case.audit["idle_power_w"] = 200.
    case.publish()
    case.record.values["spindle_power_w"][1000:2000] = np.linspace(100., 300., 1000)
    with pytest.raises(ValueError, match="恢复|波动"):
        prepare_record_idle_power(case.record, case.evidence, case.meta)


def test_constant_speed_mask_excludes_speed_change_gaps_and_hold(case):
    case.record.values["spindle_speed_rpm"][3000:4000] = 6000.
    case.record.valid[5000:5010] = False
    result = prepare_record_idle_power(case.record, case.evidence, case.meta)
    assert result["valid"][2500]
    assert not result["valid"][3000:4000].any()
    assert not result["valid"][4000:4500].any()
    assert not result["valid"][5000:5510].any()
    assert result["valid"][5800]
    assert result["label_metadata"]["schema"] == MASK_VERSION


def test_source_index_gap_never_bridges_supervision(case):
    case.record.source_indices[4000:] += 5
    result = prepare_record_idle_power(case.record, case.evidence, case.meta)
    assert not result["valid"][4000:4500].any()
    assert result["valid"][4800]


def test_short_platform_and_wrong_nominal_are_rejected(case):
    case.record.valid[1500] = False
    with pytest.raises(ValueError, match="缺失|无效"):
        prepare_record_idle_power(case.record, case.evidence, case.meta)
    case.record.valid[1500] = True
    case.record.values["spindle_speed_rpm"][1000:2000] = 6000.
    with pytest.raises(ValueError, match="转速"):
        prepare_record_idle_power(case.record, case.evidence, case.meta)


def test_new_label_metadata_does_not_change_feed_reuse_identity(case):
    from spindle_power.load_prediction.feed_samples import FEED_FEATURE_COLUMNS
    frame = pd.DataFrame({key: [1., 2.] for key in FEED_FEATURE_COLUMNS})
    sample = SimpleNamespace(frame=frame, metadata=deepcopy(case.meta))
    before = feed_sample_identity([sample])
    sample.metadata["record_idle_label"] = prepare_record_idle_power(case.record, case.evidence, case.meta)["label_metadata"]
    sample.frame["power_cut_w"] = [123., 456.]
    sample.frame["base_power_w"] = 200.
    assert feed_sample_identity([sample]) == before
    dependencies = feed_source_fingerprints()
    assert "record_idle.py" not in dependencies and "base_qualification.py" not in dependencies


def test_frozen_report_is_copied_and_rechecked_independently(case, tmp_path):
    result = prepare_record_idle_power(case.record, case.evidence, case.meta)
    sample = SampleSet(pd.DataFrame(dict(valid_main=[True], valid_residual=[True], power_cut_w=[150.])),
                       {**case.meta, "record_idle_label": result["label_metadata"]})
    service = TrainingService(tmp_path / "new_service")
    prepared = service.save_prepared([], [sample], source_snapshot=dict(base_validation=case.evidence), synthetic=True)
    _, loads, snapshot = service.load_prepared(prepared["snapshot_id"])
    evidence = snapshot["source_snapshot"]["base_validation"]
    assert not Path(evidence["report_path"]).is_absolute()
    Path(case.evidence["report_path"]).write_text("original changed", encoding="utf-8")
    assert verify_label_evidence(evidence, [case.meta], base_dir=prepared["directory"])["valid"]
    verify_prepared_record_idle(loads[0], evidence)
    loads[0].metadata["record_idle_label"]["valid_samples"] += 1
    with pytest.raises(ValueError, match="掩码指纹"):
        verify_prepared_record_idle(loads[0], evidence)
    (Path(prepared["directory"]) / evidence["report_path"]).write_text("{}", encoding="utf-8")
    assert not verify_label_evidence(evidence, [case.meta], base_dir=prepared["directory"])["valid"]


def test_quality_refactor_preserves_existing_numerical_definition():
    time = np.arange(1000) * .001
    power = 200. + 2. * np.sin(time * 31.)
    value = idle_platform_quality(time, power)
    groups = np.array_split(power, 4)
    expected = abs(np.median(groups[-1]) - np.median(groups[0])) / (.75 * (time[-1] - time[0]))
    assert value["drift_w_per_s"] == expected
    assert value["ripple_p95_w"] == np.quantile(np.abs(power - np.median(power)), .95)


def test_unified_sample_route_keeps_nc_grid_and_excluded_record(case, tmp_path):
    from test_shared_spatial_alignment import _case
    features, plan, record, meta, process, _ = _case(tmp_path)
    # This synthetic adapter test isolates the new route from spatial math.
    result = dict(power_w=np.full(len(record.time_s), np.nan), valid=np.zeros(len(record.time_s), bool),
        status="excluded_record_idle_labels", formal_training_release=True, model_version=VERSION,
        label_metadata=dict(status="excluded", prediction_input=False))
    with patch("spindle_power.load_prediction.record_idle.prepare_record_idle_power", return_value=result), patch(
            "spindle_power.load_prediction.samples.predict_base_from_measured_speed", side_effect=AssertionError("dynamic path called")):
        load = prepare_load_samples(process, plan, record, None, meta, base_qualification=dict(mode=MODE))
    assert len(load.frame) == len(features.frame)
    assert not load.frame.valid_main.any() and not load.frame.valid_residual.any()
    assert load.metadata["record_idle_label"]["status"] == "excluded"
    pd.testing.assert_series_equal(load.frame.sequence_id, features.frame.sequence_id)
