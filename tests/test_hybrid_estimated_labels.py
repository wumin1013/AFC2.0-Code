"""Synthetic hybrid-protocol regressions; not machining-domain qualification."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spindle_power.load_prediction.base_power import predict_base_from_measured_speed
from spindle_power.load_prediction.base_qualification import embedded_json_artifact, verify_label_evidence
from spindle_power.load_prediction.common import file_hash, save_json
from spindle_power.load_prediction.feed_identity import feed_sample_identity, feed_source_fingerprints
from spindle_power.load_prediction.hybrid_labels import (
    AFTER, ENTIRE, ESTIMATED, LABEL_SOURCES, MEASURED, MODE, SCHEMA,
    SOURCE_NAMES, STATUS, VERSION, label_metadata, prepare_hybrid_power,
    verify_hybrid_evidence, verify_prepared_hybrid,
)
from spindle_power.load_prediction.record_idle import _digest, prepare_record_idle_power
from spindle_power.load_prediction.samples import SampleSet
from test_base_timeline import fixture as event_fixture
from test_record_idle_labels import case as idle_case


@pytest.fixture
def hybrid_case(idle_case, tmp_path):
    """Bind an explicitly synthetic frozen model and unchanged parent audit."""
    model = deepcopy(event_fixture()[2])
    model.speeds_rpm = [0., 6000., 8000.]
    model.version = "synthetic_hybrid_frozen_model"
    for profile in model.event_profiles:
        level = 6000. if profile["level_rpm"] == 3000. else 8000.
        profile["speed_rpm"] = (np.asarray(profile["speed_rpm"]) * level / profile["level_rpm"]).tolist()
        profile["level_rpm"] = level
    model_path = tmp_path / "frozen_model.json"
    model.save(model_path)
    parent_artifact = embedded_json_artifact(idle_case.evidence["report_path"])
    row = deepcopy(idle_case.audit)
    row.update(route=MEASURED, reference_application=AFTER)
    required = dict(mode=MODE, protocol_version=VERSION, status=STATUS,
                    authorized_for_training=True, formal_metrological_qualification=False)
    domain = dict(machine_id="synthetic", power_channel="spindle_power_w",
                  scope="record_local_hybrid_estimation")
    report = dict(**required, schema=SCHEMA, domain=domain,
        model=dict(version=model.version, sha256=file_hash(model_path), original_status=model.status),
        selection_policy=deepcopy(idle_case.report["selection_policy"]),
        selection_policy_artifact=deepcopy(idle_case.report["selection_policy_artifact"]),
        parent_record_idle_report=parent_artifact,
        parent_source_sha256=deepcopy(idle_case.report["production_source_sha256"]),
        records={row["record_id"]: row})
    evidence = dict(**required, domain=domain, model_path=str(model_path),
        model_sha256=file_hash(model_path), model_version=model.version,
        report_path=str(tmp_path / "hybrid_report.json"))

    def publish():
        report["production_source_sha256"] = {
            name: file_hash(ROOT / "src/spindle_power/load_prediction" / name) for name in LABEL_SOURCES}
        save_json(evidence["report_path"], report)
        evidence["report_sha256"] = file_hash(evidence["report_path"])

    def estimate():
        row.update(route=ESTIMATED, status="model_estimated",
                   qualification_reasons=["synthetic_missing_same_record_platform"],
                   original_model_status=model.status, reference_application="frozen_dynamic_model")
        publish()

    def late_platform():
        row.update(reference_application=ENTIRE, representativeness_assumed=True,
                   platform_ranges=[[5000, 6000]], derivative_context_ranges=[[0, 6000]])
        selection_path = tmp_path / "late_selection.json"
        save_json(selection_path, dict(record_id=row["record_id"],
            measurement_sha256=row["file_sha256"]["measurement"],
            platform_ranges=row["platform_ranges"], commanded_rpm=row["commanded_rpm"]))
        row["selection_artifact"] = embedded_json_artifact(selection_path)
        row["selection_sha256"] = row["selection_artifact"]["sha256"]
        idle_case.record.values["spindle_speed_rpm"][:1000] = 6000.
        idle_case.record.values["spindle_power_w"][:] = 350.
        idle_case.record.values["spindle_power_w"][5000:] = 200.
        publish()

    publish()
    return SimpleNamespace(**vars(idle_case), row=row, model=model, hybrid_report=report,
        hybrid_evidence=evidence, publish_hybrid=publish, estimate=estimate, late_platform=late_platform)


def prepare(case):
    return prepare_hybrid_power(case.record, case.model, case.hybrid_evidence, case.meta)


def prepared_sample(case, result):
    label = result["hybrid_label_metadata"]
    return SampleSet(pd.DataFrame(dict(valid_main=[False, False], valid_residual=[False, False],
        label_source=label["label_source"], reference_kind=label["reference_kind"])),
        {**case.meta, "hybrid_label": deepcopy(label)})


def test_parent_platform_numbers_and_mask_remain_exact(hybrid_case):
    case = hybrid_case
    old = prepare_record_idle_power(case.record, case.evidence, case.meta)
    with patch("spindle_power.load_prediction.base_power.predict_base_from_measured_speed",
               side_effect=AssertionError("platform route must not use the dynamic model")):
        new = prepare(case)
    np.testing.assert_array_equal(new["power_w"], old["power_w"])
    np.testing.assert_array_equal(new["valid"], old["valid"])
    assert new["label_metadata"]["valid_sample_ranges"] == old["label_metadata"]["valid_sample_ranges"]
    assert new["status"] == STATUS
    assert new["authorized_for_training"] is True
    assert new["formal_training_release"] is False
    assert new["formal_metrological_qualification"] is False
    assert new["hybrid_label_metadata"]["label_source"] == SOURCE_NAMES[MEASURED]


@pytest.mark.parametrize("key,value", [
    ("authorized_for_training", False), ("formal_metrological_qualification", True),
    ("status", "qualified"), ("protocol_version", "obsolete"),
])
def test_estimation_requires_explicit_current_authorization(hybrid_case, key, value):
    case = hybrid_case
    case.hybrid_evidence[key] = value
    assert not verify_label_evidence(case.hybrid_evidence, [case.meta])["valid"]
    with pytest.raises(ValueError, match="授权"):
        prepare(case)


def test_authorization_cannot_be_added_only_to_evidence(hybrid_case):
    case = hybrid_case
    case.hybrid_report["authorized_for_training"] = False
    case.publish_hybrid()
    assert not verify_hybrid_evidence(case.hybrid_evidence, [case.meta])["valid"]


@pytest.mark.parametrize("kind", ["model", "report"])
def test_exact_artifact_hashes_are_required(hybrid_case, kind):
    case = hybrid_case
    Path(case.hybrid_evidence[kind + "_path"]).write_text("{}", encoding="utf-8")
    check = verify_hybrid_evidence(case.hybrid_evidence, [case.meta])
    assert not check["valid"] and "SHA-256" in " ".join(check["issues"])


def test_current_source_identity_is_enforced(hybrid_case):
    case = hybrid_case
    case.hybrid_report["production_source_sha256"]["hybrid_labels.py"] = "0" * 64
    save_json(case.hybrid_evidence["report_path"], case.hybrid_report)
    case.hybrid_evidence["report_sha256"] = file_hash(case.hybrid_evidence["report_path"])
    check = verify_hybrid_evidence(case.hybrid_evidence, [case.meta])
    assert not check["valid"] and "当前源码" in " ".join(check["issues"])


def test_parent_source_is_preserved_as_historical_identity(hybrid_case, tmp_path):
    case = hybrid_case
    parent = deepcopy(case.report)
    parent["production_source_sha256"] = {k: "a" * 64 for k in parent["production_source_sha256"]}
    path = tmp_path / "historical_parent.json"
    save_json(path, parent)
    case.hybrid_report["parent_record_idle_report"] = embedded_json_artifact(path)
    case.hybrid_report["parent_source_sha256"] = parent["production_source_sha256"]
    case.publish_hybrid()
    assert verify_hybrid_evidence(case.hybrid_evidence, [case.meta])["valid"]
    case.hybrid_report["parent_source_sha256"] = {}
    case.publish_hybrid()
    assert not verify_hybrid_evidence(case.hybrid_evidence, [case.meta])["valid"]


@pytest.mark.parametrize("key", ["measurement", "nc", "process", "machine"])
def test_each_source_file_is_bound(hybrid_case, key):
    case = hybrid_case
    case.row["file_sha256"][key] = "0" * 64
    case.publish_hybrid()
    assert not verify_hybrid_evidence(case.hybrid_evidence, [case.meta])["valid"]


def test_parent_platform_cannot_be_silently_recalculated(hybrid_case):
    case = hybrid_case
    case.row["idle_power_w"] += 1.
    case.publish_hybrid()
    check = verify_hybrid_evidence(case.hybrid_evidence, [case.meta])
    assert not check["valid"] and "父平台数值" in " ".join(check["issues"])


def test_late_platform_covers_earlier_same_speed_with_explicit_assumption(hybrid_case):
    case = hybrid_case
    case.late_platform()
    result = prepare(case)
    assert not result["valid"][:1000].any()
    assert result["valid"][2500] and result["valid"][5500]
    np.testing.assert_array_equal(result["power_w"], 200.)
    label = result["hybrid_label_metadata"]
    assert label["representativeness_assumed"] is True
    assert label["reference_application"] == ENTIRE
    assert any("假设" in warning for warning in result["warnings"])
    case.row["representativeness_assumed"] = False
    case.publish_hybrid()
    assert not verify_hybrid_evidence(case.hybrid_evidence, [case.meta])["valid"]


def test_dynamic_route_is_exact_existing_predictor_without_platform_hold(hybrid_case):
    case = hybrid_case
    case.estimate()
    time = case.record.time_s
    speed = np.interp(time, [0., 1., 1.3, 4., 4.3, 6.], [0., 0., 8000., 8000., 0., 0.])
    case.record.values["spindle_speed_rpm"][:] = speed
    expected = predict_base_from_measured_speed(time, speed, case.model)
    before = deepcopy(asdict(case.model))
    with patch("spindle_power.load_prediction.hybrid_labels._record_idle_power",
               side_effect=AssertionError("estimated route must not apply platform hold")):
        result = prepare(case)
    np.testing.assert_array_equal(result["power_w"], expected["power_w"])
    np.testing.assert_array_equal(result["valid"], expected["valid"])
    assert result["valid"][:500].all()  # Existing stopped-state domain has no new 0.5 s hold.
    assert result["warnings"] == ["[0,6000): " + w for w in expected["warnings"]]
    assert asdict(case.model) == before
    assert result["original_model_status"] == "calibrated_unvalidated"
    assert result["hybrid_label_metadata"]["evidence_grade"] == "model_estimated"
    assert result["formal_metrological_qualification"] is False


def test_dynamic_unknown_initial_warning_and_mask_are_preserved(hybrid_case):
    case = hybrid_case
    case.estimate()
    expected = predict_base_from_measured_speed(case.record.time_s,
        case.record.values["spindle_speed_rpm"], case.model)
    assert expected["warnings"]
    result = prepare(case)
    np.testing.assert_array_equal(result["valid"], expected["valid"])
    assert result["warnings"] == ["[0,6000): " + w for w in expected["warnings"]]


def test_dynamic_continuous_blocks_do_not_bridge_measurement_or_index_gaps(hybrid_case):
    case = hybrid_case
    case.estimate()
    case.record.valid[2000] = False
    case.record.source_indices[4000:] += 5
    result = prepare(case)
    assert not result["valid"][2000] and np.isnan(result["power_w"][2000])
    for first, last in [(0, 2000), (2001, 4000), (4000, 6000)]:
        expected = predict_base_from_measured_speed(case.record.time_s[first:last],
            case.record.values["spindle_speed_rpm"][first:last], case.model)
        np.testing.assert_array_equal(result["power_w"][first:last], expected["power_w"])
        np.testing.assert_array_equal(result["valid"][first:last], expected["valid"])


@pytest.mark.parametrize("change", ["claimed_qualified", "changed_original_status", "missing_reason"])
def test_estimated_record_does_not_upgrade_old_metrological_status(hybrid_case, change):
    case = hybrid_case
    case.estimate()
    if change == "claimed_qualified":
        case.row["status"] = "qualified"
    elif change == "changed_original_status":
        case.row["original_model_status"] = "validated_in_sample_domain"
    else:
        case.row["qualification_reasons"] = []
    case.publish_hybrid()
    assert not verify_hybrid_evidence(case.hybrid_evidence, [case.meta])["valid"]


def test_changed_in_memory_model_is_rejected(hybrid_case):
    case = hybrid_case
    case.model.powers_w[-1] += 10.
    with pytest.raises(ValueError, match="内存基础模型"):
        prepare(case)


def test_new_source_metadata_cannot_change_feed_identity(hybrid_case):
    from spindle_power.load_prediction.feed_samples import FEED_FEATURE_COLUMNS
    case = hybrid_case
    case.estimate()
    frame = pd.DataFrame({key: [1., 2.] for key in FEED_FEATURE_COLUMNS})
    sample = SimpleNamespace(frame=frame, metadata=deepcopy(case.meta))
    before = feed_sample_identity([sample])
    label = prepare(case)["hybrid_label_metadata"]
    sample.metadata["hybrid_label"] = label
    for name in ("label_source", "reference_kind", "original_model_status"):
        sample.frame[name] = label[name]
    sample.frame["power_cut_w"] = [123., 456.]
    assert feed_sample_identity([sample]) == before
    dependencies = feed_source_fingerprints()
    assert "hybrid_labels.py" not in dependencies
    assert "record_idle.py" not in dependencies


def test_prepared_receipt_enforces_point_source_and_mask_identity(hybrid_case):
    case = hybrid_case
    sample = prepared_sample(case, prepare(case))
    verify_prepared_hybrid(sample, case.hybrid_evidence)
    sample.frame.loc[0, "label_source"] = SOURCE_NAMES[ESTIMATED]
    with pytest.raises(ValueError, match="逐点标签来源"):
        verify_prepared_hybrid(sample, case.hybrid_evidence)
    sample.frame["label_source"] = SOURCE_NAMES[MEASURED]
    sample.metadata["hybrid_label"]["valid_samples"] += 1
    with pytest.raises(ValueError, match="掩码回执"):
        verify_prepared_hybrid(sample, case.hybrid_evidence)


def test_empty_dynamic_validity_never_creates_training_labels(hybrid_case):
    case = hybrid_case
    case.estimate()
    case.record.valid[:] = False
    result = prepare(case)
    assert not result["valid"].any()
    sample = prepared_sample(case, result)
    verify_prepared_hybrid(sample, case.hybrid_evidence)
    sample.frame.loc[0, "valid_main"] = True
    with pytest.raises(ValueError, match="空原始有效域"):
        verify_prepared_hybrid(sample, case.hybrid_evidence)


@pytest.mark.parametrize("kind", ["fractional", "negative", "duplicate"])
def test_estimated_route_rejects_invalid_original_indices(hybrid_case, kind):
    case = hybrid_case
    case.estimate()
    if kind == "fractional":
        case.record.source_indices = case.record.source_indices.astype(float) + .5
    elif kind == "negative":
        case.record.source_indices -= 1
    else:
        case.record.source_indices[1] = 0
    with pytest.raises(ValueError, match="采样索引"):
        prepare(case)


def test_prepared_receipt_cannot_claim_inconsistent_valid_range_count(hybrid_case):
    case = hybrid_case
    sample = prepared_sample(case, prepare(case))
    label = sample.metadata["hybrid_label"]
    label["valid_sample_ranges"] = [[2000, 6000]]
    label["valid_samples"] = 1
    label["mask_sha256"] = _digest({k: v for k, v in label.items() if k != "mask_sha256"})
    with pytest.raises(ValueError):
        verify_prepared_hybrid(sample, case.hybrid_evidence)
