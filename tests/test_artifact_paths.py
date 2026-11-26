import json
import os
from pathlib import Path

import pytest

from spindle_power.load_prediction.artifact_paths import ArtifactResolver
from spindle_power.load_prediction.common import file_hash


def receipt(tmp_path, rows):
    path = tmp_path / "moves.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return {"path": str(path), "sha256": file_hash(path)}


def test_explicit_relocation_preserves_bytes_and_does_not_rewrite_source(tmp_path):
    old, target = tmp_path / "old", tmp_path / "external"
    target.mkdir()
    artifact = target / "audit.json"
    raw = b'{"original_path":"immutable-origin","verified":true}\n'
    artifact.write_bytes(raw)
    ref = receipt(tmp_path, [dict(source=str(old), destination=str(target), status="moved_verified")])
    resolver = ArtifactResolver(ref)
    assert resolver.resolve(old / artifact.name) == artifact
    assert artifact.read_bytes() == raw
    assert not old.exists()
    assert resolver.resolve(target / artifact.name) == artifact
    assert resolver.resolve(tmp_path / "old-sibling" / artifact.name) == tmp_path / "old-sibling" / artifact.name


def test_changed_or_unverified_receipt_is_rejected(tmp_path):
    ref = receipt(tmp_path, [dict(source=str(tmp_path / "old"), destination=str(tmp_path / "new"), status="pending")])
    with pytest.raises(ValueError, match="核验"):
        ArtifactResolver(ref)
    Path(ref["path"]).write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        ArtifactResolver(ref)


def test_nested_migration_uses_most_specific_path(tmp_path):
    old = tmp_path / "old"
    ref = receipt(tmp_path, [
        dict(source=str(old), destination=str(tmp_path / "new"), status="moved_verified"),
        dict(source=str(old / "special"), destination=str(tmp_path / "other"), status="moved_verified")])
    assert ArtifactResolver(ref).resolve(old / "special" / "x.json") == tmp_path / "other" / "x.json"


def test_duplicate_or_recursive_destination_is_rejected(tmp_path):
    row = dict(source=str(tmp_path / "old"), destination=str(tmp_path / "new"), status="moved_verified")
    with pytest.raises(ValueError):
        ArtifactResolver(receipt(tmp_path, [row, row]))
    row["destination"] = str(tmp_path / "old" / "inside")
    with pytest.raises(ValueError):
        ArtifactResolver(receipt(tmp_path, [row]))


def test_relative_evidence_requires_explicit_base(tmp_path):
    with pytest.raises(ValueError):
        ArtifactResolver().resolve("x.json")
    assert ArtifactResolver(base_dir=tmp_path).resolve("x.json") == tmp_path / "x.json"


def test_binding_accepts_relocated_historical_paths_but_rejects_changed_bytes(tmp_path):
    from spindle_power.load_prediction.process_library import attach_process_binding, verify_process_library_binding
    old, target = tmp_path / "old", tmp_path / "new"
    target.mkdir()
    for name in ("path.nc", "canonical.csv"):
        (target / name).write_text("unchanged", encoding="utf-8")
    binding = dict(schema="paper1.process_library_binding.1", source_hashes={
        str(old / name): file_hash(target / name) for name in ("path.nc", "canonical.csv")},
        combinations={"key": dict(parameter_group="P0", feature="外形", nc_file=str(old / "path.nc"),
                                  canonical_process=str(old / "canonical.csv"))},
        assignments=[dict(record_id="R01", round_id=1, feature="外形", combination_key="key")])
    path = target / "binding.json"
    path.write_text(json.dumps(binding), encoding="utf-8")
    records = attach_process_binding([dict(record_id="R01", status="active", dataset_role="development",
        parameter_group="P0", feature="外形", nc=str(old / "path.nc"), process=str(old / "canonical.csv"))], path)
    records[0]["process_library_binding"]["path"] = str(old / "binding.json")
    reference = dict(path=str(old / "binding.json"), sha256=file_hash(path))
    moves = receipt(tmp_path, [dict(source=str(old), destination=str(target), status="moved_verified")])
    result = verify_process_library_binding(records, manifest_base=tmp_path, reference=reference, artifact_relocations=moves)
    assert result["records_verified"] == 1
    (target / "canonical.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="哈希"):
        verify_process_library_binding(records, manifest_base=tmp_path, reference=reference, artifact_relocations=moves)


@pytest.mark.parametrize("value", [r"D:\archive\数据\x.json", "d:/ARCHIVE/数据/x.json"])
def test_windows_source_is_resolved_lexically_on_any_host(tmp_path, value):
    target = tmp_path / "portable"
    ref = receipt(tmp_path, [dict(source="D:/archive", destination=str(target), status="moved_verified")])
    assert ArtifactResolver(ref).resolve(value) == target / "数据" / "x.json"


def test_relative_path_can_use_a_historical_windows_base(tmp_path):
    target = tmp_path / "portable"
    ref = receipt(tmp_path, [dict(source="D:/archive", destination=str(target), status="moved_verified")])
    resolver = ArtifactResolver(ref, base_dir=r"D:\archive\record")
    assert resolver.resolve(r"evidence\x.json") == target / "record/evidence/x.json"


def test_duplicate_windows_source_spellings_and_conflicting_targets_are_rejected(tmp_path):
    rows = [dict(source="D:/Archive", destination=str(tmp_path / "a"), status="moved_verified"),
            dict(source=r"d:\archive", destination=str(tmp_path / "b"), status="moved_verified")]
    with pytest.raises(ValueError, match="重复"):
        ArtifactResolver(receipt(tmp_path, rows))
    rows[1] = dict(source="E:/other", destination=str(tmp_path / "a"), status="moved_verified")
    with pytest.raises(ValueError, match="冲突"):
        ArtifactResolver(receipt(tmp_path, rows))


@pytest.mark.parametrize("value", [r"D:\archive\..\escape.json", "D:/archive/a/../../escape.json",
                                    "D:archive/file.json", r"\rooted_without_drive\file.json"])
def test_ambiguous_or_traversing_historical_paths_are_rejected(tmp_path, value):
    ref = receipt(tmp_path, [dict(source="D:/archive", destination=str(tmp_path / "a"), status="moved_verified")])
    with pytest.raises(ValueError):
        ArtifactResolver(ref).resolve(value)


@pytest.mark.skipif(os.name == "nt", reason="Requires a POSIX host to test foreign Windows path rejection")
def test_unmapped_windows_path_is_not_a_posix_relative_path():
    with pytest.raises(ValueError, match="未登记"):
        ArtifactResolver().resolve("D:/archive/file.json")


def test_posix_source_is_explicitly_relocated_to_local_target(tmp_path):
    target = tmp_path / "portable"
    ref = receipt(tmp_path, [dict(source="/historic/records", destination=str(target), status="moved_verified")])
    assert ArtifactResolver(ref).resolve("/historic/records/x.json") == target / "x.json"


def test_overlapping_destinations_require_consistent_source_hierarchy(tmp_path):
    rows = [dict(source="D:/old", destination=str(tmp_path / "target"), status="moved_verified"),
            dict(source="E:/unrelated", destination=str(tmp_path / "target/sub"), status="moved_verified")]
    with pytest.raises(ValueError, match="冲突"):
        ArtifactResolver(receipt(tmp_path, rows))
    with pytest.raises(ValueError, match="冲突"):
        ArtifactResolver(receipt(tmp_path, list(reversed(rows))))
    rows[1]["source"] = "D:/old/sub"
    resolver = ArtifactResolver(receipt(tmp_path, rows))
    assert resolver.resolve("d:/old/sub/x.json") == tmp_path / "target/sub/x.json"
