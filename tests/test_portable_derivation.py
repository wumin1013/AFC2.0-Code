"""Portable historical-consumer verification; no training or real data is used."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from spindle_power.load_prediction.base_qualification import embedded_json_artifact
from spindle_power.load_prediction.common import file_hash, save_json
from spindle_power.load_prediction.hybrid_labels import LABEL_SOURCES
from spindle_power.load_prediction.main_stability_refresh import (
    POLICY, ROLE, RULE_FILES, SCHEMA, VERSION, _inside, verify_derivation,
)
from spindle_power.load_prediction.training_checkpoint import digest
from test_hybrid_estimated_labels import hybrid_case, idle_case


@pytest.fixture
def portable_chain(tmp_path):
    """A zero-row structural fixture isolates lookup from scientific mask tests."""
    parent, child, historical = (tmp_path / name for name in ("parent", "child", "historical"))
    for directory in (parent / "base_evidence", child / "parent_producer",
                      historical / "src/load_prediction"):
        directory.mkdir(parents=True)
    source_dir = historical / "src/load_prediction"
    for name in set(RULE_FILES) | set(LABEL_SOURCES):
        (source_dir / name).write_text(f"# Immutable historical fixture: {name}\n", encoding="utf-8")
    source_hashes = {name: file_hash(source_dir / name) for name in LABEL_SOURCES}
    for name in LABEL_SOURCES:
        (child / "parent_producer" / name).write_bytes((source_dir / name).read_bytes())
    producer = dict(production_source_sha256=source_hashes, records={})
    producer_file = parent / "base_evidence/report.json"
    save_json(producer_file, producer)
    evidence = dict(report_path=r"base_evidence\report.json", report_sha256=file_hash(producer_file))
    source = dict(files_verified=True, base_validation=evidence)
    parent_snapshot = dict(synthetic=False, source_snapshot=source, records=[])
    parent_snapshot["snapshot_id"] = digest(parent_snapshot)
    save_json(parent / "snapshot.json", parent_snapshot)
    parent_ref = dict(directory=r"D:\frozen\parent", snapshot_id=parent_snapshot["snapshot_id"],
                      sha256=file_hash(parent / "snapshot.json"))
    decision = tmp_path / "decision.json"
    save_json(decision, dict(status="synthetic_path_test"))
    rule_sources = {name: file_hash(source_dir / name) for name in RULE_FILES}
    receipt = dict(schema=SCHEMA, policy=POLICY, rule_sources=rule_sources,
                   parent_snapshot=parent_ref, decision=embedded_json_artifact(decision), records=[])
    save_json(child / "main_stability_receipt.json", receipt)
    link = dict(schema=SCHEMA, rule_version=VERSION, receipt_file="main_stability_receipt.json",
                receipt_sha256=file_hash(child / "main_stability_receipt.json"))
    consumer = {**deepcopy(producer), "production_source_role": ROLE,
        "main_stability_derivation": dict(schema=SCHEMA, producer_report=embedded_json_artifact(producer_file),
            parent_snapshot=parent_ref, rule_sources=rule_sources, receipt_sha256=link["receipt_sha256"],
            decision=receipt["decision"])}
    snapshot = dict(source_snapshot={**deepcopy(source), "main_stability_refresh": link}, records=[])
    moves = tmp_path / "moves.json"
    save_json(moves, [dict(source=r"D:\frozen\parent", destination=str(parent), status="moved_verified"),
                      dict(source="D:/frozen/source", destination=str(historical), status="moved_verified")])
    reference = dict(path=str(moves), sha256=file_hash(moves))
    return dict(snapshot=snapshot, samples=[], directory=child, consumer_report=consumer,
                artifact_relocations=reference, historical_consumer_source="D:/frozen/source")


@pytest.mark.parametrize("name", ["base_evidence/report.json", r"base_evidence\report.json"])
def test_inside_resolves_both_historical_relative_separators(tmp_path, name):
    evidence = tmp_path / "base_evidence/report.json"
    evidence.parent.mkdir()
    evidence.write_text("{}", encoding="utf-8")
    assert _inside(tmp_path, name) == evidence.resolve()


@pytest.mark.parametrize("name", ["../outside.json", r"..\outside.json",
                                  "base_evidence/../../outside.json",
                                  r"base_evidence\..\..\outside.json"])
def test_inside_rejects_parent_traversal(tmp_path, name):
    with pytest.raises(ValueError):
        _inside(tmp_path, name)


def test_inside_rejects_external_absolute_paths(tmp_path):
    root = tmp_path / "snapshot"
    root.mkdir()
    with pytest.raises(ValueError, match="越界"):
        _inside(root, str(tmp_path / "outside.json"))


def test_historical_windows_parent_and_consumer_are_verified_without_rewriting(portable_chain):
    before = json.dumps(portable_chain["snapshot"], sort_keys=True)
    result = verify_derivation(**portable_chain)
    assert result["valid"]
    assert result["verified_rule_sources"] == portable_chain["consumer_report"]["main_stability_derivation"]["rule_sources"]
    assert json.dumps(portable_chain["snapshot"], sort_keys=True) == before
    # The default remains strict against the currently imported source.
    omitted = {k: v for k, v in portable_chain.items() if k != "historical_consumer_source"}
    with pytest.raises(ValueError, match="源码改变"):
        verify_derivation(**omitted)


@pytest.mark.parametrize("name", list(dict.fromkeys((*RULE_FILES, *LABEL_SOURCES))))
def test_each_historical_source_hash_remains_mandatory(portable_chain, name):
    base = Path(portable_chain["directory"]).parent / "historical/src/load_prediction"
    with (base / name).open("a", encoding="utf-8") as stream:
        stream.write("# changed\n")
    with pytest.raises(ValueError):
        verify_derivation(**portable_chain)


def test_relocated_parent_and_receipt_tampering_remain_rejected(portable_chain):
    root = Path(portable_chain["directory"]).parent
    path = root / "parent/snapshot.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="证据文件改变"):
        verify_derivation(**portable_chain)


def test_relocation_manifest_hash_is_mandatory(portable_chain):
    path = Path(portable_chain["artifact_relocations"]["path"])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_derivation(**portable_chain)


@pytest.mark.parametrize("mode", ["hybrid", "record_idle"])
def test_qualification_dispatch_verifies_explicit_historical_sources(hybrid_case, tmp_path, mode):
    """Exercise the real qualification wrappers, model and record evidence."""
    from spindle_power.load_prediction.dataset_manifest import verify_base_evidence
    from spindle_power.load_prediction.record_idle import LABEL_SOURCES as IDLE_SOURCES
    case = hybrid_case
    evidence = deepcopy(case.hybrid_evidence if mode == "hybrid" else case.evidence)
    report = deepcopy(case.hybrid_report if mode == "hybrid" else case.report)
    names = LABEL_SOURCES if mode == "hybrid" else IDLE_SOURCES
    current = Path(__file__).resolve().parents[1] / "src/spindle_power/load_prediction"
    historical = tmp_path / "historical"
    historical_module = historical / "src/load_prediction"
    historical_module.mkdir(parents=True)
    for name in names:
        (historical_module / name).write_bytes((current / name).read_bytes() + b"\n# historical fixture\n")
    report["production_source_sha256"] = {n: file_hash(historical_module / n) for n in names}
    report_path = tmp_path / "historical_report.json"
    save_json(report_path, report)
    evidence_directory = tmp_path / "portable_evidence"
    evidence_directory.mkdir()
    (evidence_directory / report_path.name).write_bytes(report_path.read_bytes())
    evidence["report_path"] = "D:/evidence/historical_report.json"
    evidence["report_sha256"] = file_hash(report_path)
    if mode == "hybrid":
        model_path = Path(evidence["model_path"])
        (evidence_directory / model_path.name).write_bytes(model_path.read_bytes())
        evidence["model_path"] = "D:/evidence/" + Path(evidence["model_path"]).name
    moves = tmp_path / "qualification_moves.json"
    save_json(moves, [dict(source="D:/evidence", destination=str(evidence_directory), status="moved_verified"),
                      dict(source="D:/source", destination=str(historical), status="moved_verified")])
    relocations = dict(path=str(moves), sha256=file_hash(moves))
    kwargs = dict(artifact_relocations=relocations)
    assert not verify_base_evidence(evidence, [case.meta], **kwargs)["valid"]
    kwargs["historical_consumer_source"] = "D:/source"
    check = verify_base_evidence(evidence, [case.meta], **kwargs)
    assert check["valid"], check["issues"]
    assert check["report"]["production_source_sha256"] == report["production_source_sha256"]
    # Every frozen source remains required, including sources unchanged by this task.
    for name in names:
        file = historical_module / name
        raw = file.read_bytes()
        file.write_bytes(raw + b"# tampered\n")
        check = verify_base_evidence(evidence, [case.meta], **kwargs)
        assert not check["valid"] and name in " ".join(check["issues"])
        file.write_bytes(raw)
    assert verify_base_evidence(evidence, [case.meta], **kwargs)["valid"]


def test_historical_source_option_does_not_bypass_other_label_modes(tmp_path):
    from spindle_power.load_prediction.dataset_manifest import verify_base_evidence
    check = verify_base_evidence(dict(mode="measured_speed_labels"), historical_consumer_source=tmp_path)
    assert not check["valid"] and "仅适用于" in " ".join(check["issues"])
