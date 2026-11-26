"""A new TXT binding must preserve provenance and fail on changed inputs."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spindle_power.load_prediction.common import file_hash
from spindle_power.load_prediction.process_library import (BINDING_SCHEMA, attach_process_binding,
    relocated_process_path, verify_process_library_binding)


@pytest.fixture
def bound(tmp_path):
    paths = {}
    for key in ("nc", "process", "derived", "source"):
        p = tmp_path / (key + ".txt")
        p.write_text(key, encoding="utf-8")
        paths[key] = p
    record = dict(record_id="R01_外形", round_id=1, feature="外形", parameter_group="P0",
        status="active", dataset_role="development", nc=str(paths["nc"]), process=str(paths["process"]),
        source_process={"path": str(paths["source"]), "sha256": file_hash(paths["source"])})
    assignment = dict(record_id=record["record_id"], round_id=1, feature="外形", combination_key="P0:nc",
        process_txt=str(paths["derived"]), process_txt_sha256=file_hash(paths["derived"]))
    receipt = dict(schema=BINDING_SCHEMA, assignments=[assignment], combinations={"P0:nc": dict(
        parameter_group="P0", feature="外形", nc_file=str(paths["nc"]), canonical_process=str(paths["process"]))},
        source_hashes={str(p): file_hash(p) for p in paths.values()})
    p = tmp_path / "binding.json"
    p.write_text(json.dumps(receipt, ensure_ascii=False), encoding="utf-8")
    return record, attach_process_binding([record], p), dict(path=str(p), sha256=file_hash(p)), paths


def test_new_binding_preserves_canonical_and_original_source(bound, tmp_path):
    original, records, reference, _ = bound
    assert "process_library_binding" not in original
    assert records[0]["source_process"] == original["source_process"]
    assert records[0]["process"] == original["process"]
    assert verify_process_library_binding(records, manifest_base=tmp_path, reference=reference)["records_verified"] == 1


def test_changed_derived_txt_cannot_reuse_old_binding(bound, tmp_path):
    _, records, reference, paths = bound
    paths["derived"].write_text("different F or geometry", encoding="utf-8")
    with pytest.raises(ValueError, match="哈希"):
        verify_process_library_binding(records, manifest_base=tmp_path, reference=reference)


def test_changed_record_combination_rejected(bound, tmp_path):
    _, records, reference, _ = bound
    records[0]["parameter_group"] = "P1"
    with pytest.raises(ValueError, match="组合身份"):
        verify_process_library_binding(records, manifest_base=tmp_path, reference=reference)


def test_binding_receipt_must_not_be_rewritten(bound, tmp_path):
    _, records, reference, _ = bound
    Path(reference["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="哈希"):
        verify_process_library_binding(records, manifest_base=tmp_path, reference=reference)


def test_relocation_only_removes_known_wrapper(tmp_path):
    assert relocated_process_path(str(Path("工艺信息") / "P0" / "x.txt"), tmp_path) == tmp_path / "P0" / "x.txt"
    with pytest.raises(ValueError):
        relocated_process_path(str(Path("工艺信息") / ".." / "x.txt"), tmp_path)
    with pytest.raises(ValueError):
        relocated_process_path("unknown/x.txt", tmp_path)
