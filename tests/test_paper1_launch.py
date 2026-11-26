"""Launch orchestration checks; no raw-data inspection or neural training."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paper1_launch", ROOT / "scripts/verification/prepare_paper1_launch.py")
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)
from spindle_power.load_prediction.experiment_protocol import PROGRAM_FEATURES, round_spec


@pytest.fixture
def plan(tmp_path):
    records = []
    for round_id in range(1, 91):
        details = round_spec(round_id)
        for feature in PROGRAM_FEATURES:
            records.append(dict(record_id=f"R{round_id:02d}_{feature}", round_id=round_id, feature=feature,
                combination=details["combination"], parameter_group=details["parameter_group"],
                dataset_role="development", status="active", complete=True, round_complete=True,
                sample_rules_version=launch.SAMPLE_RULES_VERSION, process_mapping_rules_version="fixture_only",
                alignment_reliable=True, process_aligned=True, no_alarm=True, tool_healthy=True))
    manifest = tmp_path / "manifest.json"
    launch.write_new(manifest, {"records": records})
    candidates = tmp_path / "candidates.json"
    candidate = dict(id="C32", feed_model=asdict(launch.FeedModelConfig()),
                     feed_training=asdict(launch.FeedTrainingConfig()),
                     load_model=asdict(launch.ModelConfig()), load_training=asdict(launch.TrainingConfig()))
    second = deepcopy(candidate)
    second["id"] = "C64"
    second["load_model"].update(hidden=64, coefficient_widths=[64, 64])
    launch.write_new(candidates, {"candidates": [candidate, second]})
    return dict(schema="paper1.launch_plan.1", manifest=str(manifest), candidates=str(candidates),
                service_output=str(tmp_path / "service"), required_replacement="R74_台阶",
                input_contract={"status": "frozen", "manifest_sha256": launch.file_hash(manifest),
                    "required_metadata": {"sample_rules_version": launch.SAMPLE_RULES_VERSION, "process_mapping_rules_version": "fixture_only"}},
                python_executable=sys.executable, base_evidence=None)


def replace_manifest(plan, transform):
    path = Path(plan["manifest"])
    data = launch.read_json(path)
    transform(data["records"])
    path.write_text(__import__("json").dumps(data, ensure_ascii=False), encoding="utf-8")
    plan["input_contract"]["manifest_sha256"] = launch.file_hash(path)


def replace_r74(records, directory):
    old = next(r for r in records if r["record_id"] == "R74_台阶")
    new = {**old, "record_id": "R74_台阶_补采", "replaces": [old["record_id"]],
           "recollection": {"round_review_snapshot_id": "test-fixture-only"},
           "actual_event_order": None, "tool_session": None}
    attach_review(new, directory)
    new["recollection"].update(measurement_sha256=launch.file_hash(new["measurement"]),
                               expected_nc_sha256=launch.file_hash(new["nc"]))
    old.update(status="superseded", superseded_by=[new["record_id"]])
    records.append(new)


def attach_review(record, directory):
    """Tiny source files exercise provenance checks, never actual supervision."""
    root = directory / "audit_fixture" / record["record_id"]
    root.mkdir(parents=True)
    sources = {}
    for field in ("measurement", "nc", "machine", "process"):
        path = root / (field + ".txt")
        path.write_text(f"fixture only: {field}", encoding="utf-8")
        sources[field] = {"path": str(path), "sha256": launch.file_hash(path)}
        record[field] = str(path)
    canonical = root / "canonical_process.csv"
    canonical.write_text("fixture only: canonical", encoding="utf-8")
    record.update(source_process=sources["process"], process=str(canonical),
                  process_version=launch.file_hash(canonical), program_version="fixture_program")
    segmentation = root / "evidence.json"
    launch.write_new(segmentation, {"nc_path": record["nc"], "nc_sha256": sources["nc"]["sha256"],
        "source_process_path": sources["process"]["path"], "source_process_sha256": sources["process"]["sha256"],
        "program_version": record["program_version"], "process_mapping_rules_version": record["process_mapping_rules_version"],
        "source_points_preserved": True, "output_sha256": {"canonical_process.csv": record["process_version"]}})
    record.update(segmentation_evidence_path=str(segmentation), segmentation_evidence_sha256=launch.file_hash(segmentation))
    reports = {
        "numeric": {"record_id": record["record_id"], "file_sha256": sources["measurement"]["sha256"]},
        "alignment": {"record_id": record["record_id"], "measurement_sha256": sources["measurement"]["sha256"],
                      "nc_sha256": sources["nc"]["sha256"]},
        "preparation": {"record_id": record["record_id"], "source_files": sources}}
    record["review_evidence"] = {"decisions": {"complete": True, "alignment_reliable": True, "process_aligned": True}}
    for kind, value in reports.items():
        path = root / (kind + ".json")
        launch.write_new(path, value)
        record["review_evidence"][kind] = {"path": str(path), "sha256": launch.file_hash(path)}


def attach_all_reviews(plan):
    def attach(rows):
        for row in rows:
            if row.get("status") == "active" and not row.get("review_evidence"):
                attach_review(row, Path(plan["manifest"]).parent)
    replace_manifest(plan, attach)


def test_unreplaced_old_r74_cannot_prepare_even_if_flags_true(plan, monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(launch, "TrainingService", lambda *_: called.append("service"))
    with pytest.raises(ValueError, match="activate_record_replacement"):
        launch.prepare(plan, "feed", tmp_path / "ready")
    assert not called


def test_missing_slot_cannot_prepare(plan, monkeypatch, tmp_path):
    replace_manifest(plan, lambda rows: rows.pop())
    monkeypatch.setattr(launch, "TrainingService", lambda *_: pytest.fail("must fail before service"))
    with pytest.raises(ValueError, match="450"):
        launch.prepare(plan, "feed", tmp_path / "ready")


def test_structural_readiness_does_not_grant_formal_qualification(plan):
    replace_manifest(plan, lambda rows: replace_r74(rows, Path(plan["manifest"]).parent))
    report = launch.preflight(plan, "feed")
    assert report["structurally_ready_for_strict_prepare"]
    assert report["formal_training_release"] is False
    assert report["raw_data_read"] is False
    assert report["current_source_content_verified"] is False
    assert report["candidate_ids"] == ["C32"]
    assert any("标签模式" in issue for issue in launch.preflight(plan, "all")["blockers"])


def replace_entire_r74(records, directory):
    for old in list(records):
        if old["round_id"] != 74:
            continue
        new = {**old, "record_id": old["record_id"] + "_补采", "replaces": [old["record_id"]],
               "recollection": {"round_review_snapshot_id": "fixture-round-checked"},
               "actual_event_order": None, "tool_session": None}
        attach_review(new, directory)
        new["recollection"].update(measurement_sha256=launch.file_hash(new["measurement"]),
                                   expected_nc_sha256=launch.file_hash(new["nc"]))
        old.update(status="superseded", superseded_by=[new["record_id"]])
        records.append(new)


def test_round_replacement_preserves_required_slot_gate_and_checks_every_recollection(plan):
    replace_manifest(plan, lambda rows: replace_entire_r74(rows, Path(plan["manifest"]).parent))
    assert launch.preflight(plan)["structurally_ready_for_strict_prepare"]
    replace_manifest(plan, lambda rows: next(r for r in rows if r["record_id"] == "R74_外形_补采")["review_evidence"].pop("alignment"))
    assert any("独立审核回执" in message for message in launch.preflight(plan)["blockers"])


def test_round_replacement_cannot_claim_all_old_slots_as_one_replacement(plan):
    replace_manifest(plan, lambda rows: replace_entire_r74(rows, Path(plan["manifest"]).parent))
    replace_manifest(plan, lambda rows: next(r for r in rows if r["record_id"] == "R74_台阶_补采").update(
        replaces=["R74_台阶", "R74_外形"]))
    assert any("逐槽唯一" in message for message in launch.preflight(plan)["blockers"])


def test_round_activation_uses_shared_transaction_and_keeps_old_manifest(plan, tmp_path, monkeypatch):
    import spindle_power.load_prediction.dataset_manifest as manifest_module
    source = launch.read_json(plan["manifest"])
    updated = deepcopy(source["records"])
    replace_entire_r74(updated, tmp_path)
    replacements = [row for row in updated if row.get("replaces")]
    payload = tmp_path / "round.json"
    launch.write_new(payload, {"replacements": replacements})
    called = []

    def transaction(records, candidates, *, base_dir):
        called.append((records, candidates, base_dir))
        return deepcopy(updated)

    monkeypatch.setattr(manifest_module, "activate_round_replacement", transaction)
    result = launch.activate(plan, payload, tmp_path / "activated", whole_round=True)
    assert len(called) == 1 and called[0][1] == replacements
    assert launch.read_json(plan["manifest"]) == source
    assert launch.read_json(result["plan"])["input_contract"]["status"] == "awaiting_refreeze"
    assert len(launch.read_json(result["manifest"])["records"]) == 455
    assert result["training_started"] is False
    # A missing receipt for any of the other four must reject before writing an activation.
    updated[-5]["review_evidence"].pop("numeric")
    with pytest.raises(ValueError, match="独立审核证据"):
        launch.activate(plan, payload, tmp_path / "rejected", whole_round=True)
    assert not (tmp_path / "rejected").exists()


def test_reserved_o1203_never_passes_development_preflight(plan):
    replace_manifest(plan, lambda rows: replace_r74(rows, Path(plan["manifest"]).parent))
    replace_manifest(plan, lambda rows: rows[0].update(nc="O1203.NC"))
    assert any("O1203" in issue for issue in launch.preflight(plan)["blockers"])


def test_old_process_mapping_or_unfrozen_contract_blocks(plan):
    replace_manifest(plan, lambda rows: replace_r74(rows, Path(plan["manifest"]).parent))
    plan["input_contract"]["status"] = "awaiting_refreeze"
    assert any("旧累计s" in issue for issue in launch.preflight(plan)["blockers"])
    plan["input_contract"]["status"] = "frozen"
    replace_manifest(plan, lambda rows: rows[0].update(process_mapping_rules_version="old"))
    assert any("未绑定新工艺" in issue for issue in launch.preflight(plan)["blockers"])


def test_explicitly_superseded_manifest_blocks_even_when_rule_versions_match(plan, tmp_path, monkeypatch):
    replace_manifest(plan, lambda rows: replace_r74(rows, Path(plan["manifest"]).parent))
    launch.write_new(Path(plan["manifest"]).parent / "SUPERSEDED.json", {"reason": "old float parsing"})
    assert any("SUPERSEDED" in issue for issue in launch.preflight(plan)["blockers"])
    monkeypatch.setattr(launch, "TrainingService", lambda *_: pytest.fail("must not prepare old artifact"))
    with pytest.raises(ValueError, match="SUPERSEDED"):
        launch.refreeze(plan, tmp_path / "frozen.json")
    with pytest.raises(ValueError, match="SUPERSEDED"):
        launch.prepare(plan, "feed", tmp_path / "ready")


def test_nearest_steady_batch_marker_does_not_poison_new_nested_batch(plan, tmp_path):
    container = tmp_path / "steady_process"
    batch = container / "uniform_new"
    canonical = batch / "combinations" / "pair" / "canonical_process.csv"
    launch.write_new(container / "SUPERSEDED.json", {"scope": "old root artifact only"})
    launch.write_new(batch / "summary.json", {"fixture": True})
    records = [{"process": str(canonical), "segmentation_evidence_path": str(canonical.with_name("evidence.json"))}]
    assert launch.superseded_input_batches(plan, records) == []
    launch.write_new(batch / "SUPERSEDED.json", {"scope": "this old batch"})
    assert launch.superseded_input_batches(plan, records) == [str(batch / "SUPERSEDED.json")]


def test_refreeze_never_bypasses_missing_recollection(plan, tmp_path):
    with pytest.raises(ValueError, match="activate_record_replacement"):
        launch.refreeze(plan, tmp_path / "final_plan.json")
    assert not (tmp_path / "final_plan.json").exists()


def test_refreeze_binds_manifest_without_granting_training_qualification(plan, tmp_path):
    replace_manifest(plan, lambda rows: replace_r74(rows, Path(plan["manifest"]).parent))
    attach_all_reviews(plan)
    plan["input_contract"].update(status="awaiting_refreeze", manifest_sha256=None)
    original = deepcopy(plan)
    result = launch.refreeze(plan, tmp_path / "final_plan.json")
    assert plan == original
    saved = launch.read_json(result["plan"])
    assert saved["input_contract"]["status"] == "frozen"
    assert saved["input_contract"]["manifest_sha256"] == launch.file_hash(plan["manifest"])
    assert result["formal_training_release"] is False and result["training_started"] is False
    records = launch.read_json(saved["manifest"])["records"]
    assert len(records) == 451
    assert sum(r["status"] == "active" for r in records) == 450
    old = next(r for r in records if r["record_id"] == "R74_台阶")
    new = next(r for r in records if r["record_id"] == "R74_台阶_补采")
    assert old["status"] == "superseded"
    assert new["actual_event_order"] is None and new["tool_session"] is None


def test_unreviewed_recollection_cannot_refreeze_or_prepare(plan, tmp_path, monkeypatch):
    replace_manifest(plan, lambda rows: replace_r74(rows, Path(plan["manifest"]).parent))
    replace_manifest(plan, lambda rows: rows[-1].pop("review_evidence"))
    monkeypatch.setattr(launch, "TrainingService", lambda *_: pytest.fail("must stop before strict prepare"))
    with pytest.raises(ValueError, match="独立审核证据"):
        launch.refreeze(plan, tmp_path / "blocked.json")
    with pytest.raises(ValueError, match="独立审核证据"):
        launch.prepare(plan, "feed", tmp_path / "blocked_ready")


def test_recollection_cannot_reuse_old_record_audit_receipt(plan):
    replace_manifest(plan, lambda rows: replace_r74(rows, Path(plan["manifest"]).parent))
    def corrupt(rows):
        ref = rows[-1]["review_evidence"]["numeric"]
        path = Path(ref["path"])
        report = launch.read_json(path)
        report["record_id"] = "R74_台阶"
        path.write_text(__import__("json").dumps(report), encoding="utf-8")
        ref["sha256"] = launch.file_hash(path)
    replace_manifest(plan, corrupt)
    assert any("独立审核证据" in issue for issue in launch.preflight(plan)["blockers"])


def test_candidates_validate_real_dataclasses_and_deduplicate_feed(plan):
    assert [item["id"] for item in launch.candidate_list(plan["candidates"], "all")] == ["C32", "C64"]
    path = Path(plan["candidates"])
    data = launch.read_json(path)
    data["candidates"][0]["feed_training"]["epochs"] = 0
    path.write_text(__import__("json").dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="epochs"):
        launch.candidate_list(path, "feed")


def test_empty_real_snapshot_is_rejected_before_network(tmp_path, plan):
    service = launch.TrainingService(tmp_path / "empty_service")
    prepared = service.save_prepared([], [], source_snapshot={"files_verified": True})
    with pytest.raises(ValueError, match="缺少任务所需监督"):
        launch.validate_numerical(service, {"snapshot_id": prepared["snapshot_id"], "mode": "feed",
                                          "candidates": launch.candidate_list(plan["candidates"], "feed")})


def test_old_prepared_sample_rules_are_rejected_without_training(plan):
    old = SimpleNamespace(metadata={"sample_rules_version": "afc.sample_rules.old"})
    service = SimpleNamespace(output_dir=Path(plan["service_output"]), preview=lambda *args: None,
        load_prepared=lambda *_: ([old], [], {"synthetic": False}))
    with pytest.raises(ValueError, match="禁止静默兼容"):
        launch.validate_numerical(service, {"snapshot_id": "a" * 64, "mode": "feed", "candidates": []})


def test_prepare_uses_strict_shared_service_and_never_execute(plan, monkeypatch, tmp_path):
    replace_manifest(plan, lambda rows: replace_r74(rows, Path(plan["manifest"]).parent))
    attach_all_reviews(plan)
    calls = []
    class Service:
        def __init__(self, output):
            self.output_dir = Path(output)
        def prepare_feed(self, manifest):
            calls.append(("prepare_feed", manifest))
            return {"snapshot_id": "a" * 64}
    monkeypatch.setattr(launch, "TrainingService", Service)
    monkeypatch.setattr(launch, "runtime_identity", lambda: {"test": "fixed"})
    monkeypatch.setattr(launch, "validate_numerical", lambda *args: {"trained": False})
    result = launch.prepare(plan, "feed", tmp_path / "ready")
    ready = launch.read_json(result["ready"])
    request = launch.read_json(ready["request"])
    assert calls == [("prepare_feed", plan["manifest"])]
    assert result["training_started"] is False
    assert request["mode"] == "feed" and len(request["candidates"]) == 1
    assert ready["request_sha256"] == launch.file_hash(ready["request"])
    assert ready["provenance"]["source_verification_before_prepare"]["records_verified"] == 450
    assert ready["provenance"]["source_verification_after_prepare"]["records_verified"] == 450


def test_incremental_prepare_delegates_parent_snapshot_to_shared_service(plan, monkeypatch, tmp_path):
    replace_manifest(plan, lambda rows: replace_r74(rows, Path(plan["manifest"]).parent))
    attach_all_reviews(plan)
    evidence_path = tmp_path / "hybrid.json"
    launch.write_new(evidence_path, {"mode": "hybrid_fixture", "model_path": "frozen_model.json"})
    plan["base_evidence"] = str(evidence_path)
    calls = []
    service = SimpleNamespace(output_dir=Path(plan["service_output"]))
    def prepare_shared(manifest, **kwargs):
        calls.append((manifest, kwargs))
        return {"snapshot_id": "c" * 64}
    service.prepare = prepare_shared
    monkeypatch.setattr(launch, "TrainingService", lambda *_: service)
    monkeypatch.setattr(launch, "runtime_identity", lambda: {"test": "fixed"})
    monkeypatch.setattr(launch, "save_ready", lambda svc, req, dest, **kwargs: req)
    source_service = tmp_path / "previous_service"
    request = launch.prepare(plan, "all", tmp_path / "ready", source_feed_job_id="feed123",
                             source_feed_service_output=tmp_path / "feed_service",
                             source_prepared_service_output=source_service, source_prepared_snapshot_id="b" * 64)
    assert calls == [(plan["manifest"], dict(base_model_path="frozen_model.json",
        base_validation=launch.read_json(evidence_path), base_event_timelines=None,
        source_prepared_service_output=str(source_service.resolve()), source_prepared_snapshot_id="b" * 64))]
    assert request["snapshot_id"] == "c" * 64
    assert request["source_feed_job_id"] == "feed123"
    assert request["source_feed_service_output"] == str((tmp_path / "feed_service").resolve())


@pytest.mark.parametrize("mode,has_output,has_id", [("all", True, False), ("all", False, True), ("feed", True, True)])
def test_incremental_prepare_rejects_incomplete_or_feed_request_before_io(plan, monkeypatch, tmp_path, mode, has_output, has_id):
    monkeypatch.setattr(launch, "assert_runtime", lambda _: None)
    monkeypatch.setattr(launch, "preflight", lambda *a: pytest.fail("must reject before source checks"))
    with pytest.raises(ValueError, match="增量"):
        launch.prepare(plan, mode, tmp_path / "ready",
                       source_prepared_service_output=tmp_path / "previous" if has_output else None,
                       source_prepared_snapshot_id="b" * 64 if has_id else None)


def test_activated_measurement_overwritten_at_same_path_cannot_refreeze_or_prepare(plan, tmp_path, monkeypatch):
    replace_manifest(plan, lambda rows: replace_r74(rows, Path(plan["manifest"]).parent))
    attach_all_reviews(plan)
    record = launch.read_json(plan["manifest"])["records"][-1]
    Path(record["measurement"]).write_text("newly exported contents", encoding="utf-8")
    # Fast preflight intentionally does not read source bytes or grant release.
    assert launch.preflight(plan)["current_source_content_verified"] is False
    monkeypatch.setattr(launch, "TrainingService", lambda *_: pytest.fail("must stop before shared service"))
    with pytest.raises(ValueError, match="当前文件与审核冻结源不一致"):
        launch.refreeze(plan, tmp_path / "blocked.json")
    with pytest.raises(ValueError, match="当前文件与审核冻结源不一致"):
        launch.prepare(plan, "feed", tmp_path / "blocked_ready")
    assert not (tmp_path / "blocked.json").exists()
    assert not (tmp_path / "blocked_ready").exists()


@pytest.mark.parametrize("field", ["measurement", "nc", "machine", "process", "source_process", "segmentation_evidence_path"])
def test_existing_record_sources_are_rehashed_not_only_replacement(plan, field):
    record = launch.read_json(plan["manifest"])["records"][0]
    attach_review(record, Path(plan["manifest"]).parent)
    path = Path(plan["manifest"]).with_name("single_audit.json")
    launch.write_new(path, {"records": [record]})
    audited_plan = {**plan, "manifest": str(path)}
    assert launch.verify_review_sources(audited_plan)["records_verified"] == 1
    value = record[field]["path"] if field == "source_process" else record[field]
    Path(value).write_text("changed bytes", encoding="utf-8")
    with pytest.raises(ValueError, match="当前源内容校验失败"):
        launch.verify_review_sources(audited_plan)


def test_source_changed_during_service_prepare_retains_rejected_snapshot_without_ready(plan, tmp_path, monkeypatch):
    replace_manifest(plan, lambda rows: replace_r74(rows, Path(plan["manifest"]).parent))
    attach_all_reviews(plan)
    record = launch.read_json(plan["manifest"])["records"][0]
    snapshot_id = "a" * 64
    class Service:
        def __init__(self, output):
            self.output_dir = Path(output)
        def prepare_feed(self, _):
            launch.write_new(self.output_dir / "snapshots" / snapshot_id / "snapshot.json", {"fixture": "retain"})
            Path(record["measurement"]).write_text("sync changed during preparation", encoding="utf-8")
            return {"snapshot_id": snapshot_id}
    monkeypatch.setattr(launch, "TrainingService", Service)
    monkeypatch.setattr(launch, "runtime_identity", lambda: {"test": "fixed"})
    with pytest.raises(ValueError, match="不生成ready"):
        launch.prepare(plan, "feed", tmp_path / "ready")
    snapshot = Path(plan["service_output"]) / "snapshots" / snapshot_id
    assert launch.read_json(snapshot / "snapshot.json") == {"fixture": "retain"}
    assert launch.read_json(snapshot / "launcher_rejected.json")["training_started"] is False
    assert not (tmp_path / "ready").exists()
    with pytest.raises(ValueError, match="准备后源内容校验中被拒绝"):
        launch.validate_numerical(Service(plan["service_output"]), {"snapshot_id": snapshot_id})


def test_changed_request_is_rejected_without_spawning(tmp_path, monkeypatch):
    request = tmp_path / "request.json"
    launch.write_new(request, {"snapshot_id": "a" * 64})
    ready = tmp_path / "ready.json"
    launch.write_new(ready, dict(schema="paper1.ready_request.1", runtime={}, request=str(request), request_sha256="wrong"))
    monkeypatch.setattr(launch, "runtime_identity", lambda: {})
    monkeypatch.setattr(launch.subprocess, "Popen", lambda *a, **k: pytest.fail("must not spawn"))
    with pytest.raises(ValueError, match="训练请求已改变"):
        launch.launch(ready)


def test_background_launch_calls_existing_cli_without_shell_or_window(tmp_path, monkeypatch):
    request = tmp_path / "request.json"
    launch.write_new(request, {"snapshot_id": "a" * 64, "mode": "feed", "candidates": []})
    ready = tmp_path / "ready.json"
    launch.write_new(ready, dict(schema="paper1.ready_request.1", runtime={}, request=str(request),
        request_sha256=launch.file_hash(request), service_output=str(tmp_path / "service")))
    monkeypatch.setattr(launch, "runtime_identity", lambda: {})
    monkeypatch.setattr(launch, "TrainingService", lambda *_: SimpleNamespace(validate_recovery=lambda _: None))
    monkeypatch.setattr(launch, "validate_numerical", lambda *a: {"trained": False})
    seen = []
    def spawn(command, **kwargs):
        seen.append((command, kwargs))
        return SimpleNamespace(pid=123)
    monkeypatch.setattr(launch.subprocess, "Popen", spawn)
    receipt = launch.launch(ready)
    command, kwargs = seen[0]
    assert command == launch.cli_command(tmp_path / "service", request)
    assert command[-2] == "train" and Path(command[2]).name == "load_prediction.py"
    assert not kwargs.get("shell", False)
    if sys.platform == "win32":
        assert kwargs["creationflags"] == launch.subprocess.CREATE_NO_WINDOW
    assert receipt["pid"] == 123
    with pytest.raises(FileExistsError):
        launch.launch(ready)
    assert len(seen) == 1


def test_resume_preserves_request_and_delegates_identity_check(plan, tmp_path, monkeypatch):
    source = Path(plan["service_output"]) / "jobs" / "abc123" / "request.json"
    request = dict(snapshot_id="a" * 64, mode="feed", candidates=[{"id": "C32"}],
                   job_id="abc123", service_output=plan["service_output"])
    launch.write_new(source, request)
    captured = []
    service = SimpleNamespace(output_dir=Path(plan["service_output"]), validate_recovery=lambda req: captured.append(deepcopy(req)))
    monkeypatch.setattr(launch, "TrainingService", lambda *_: service)
    monkeypatch.setattr(launch, "save_ready", lambda svc, req, dest, **kw: req)
    result = launch.resume(plan, "abc123", tmp_path / "resume")
    assert result == {k: v for k, v in request.items() if k not in ("job_id", "service_output")} | {"resume_from_job_id": "abc123"}
    assert captured == [result]
    assert launch.read_json(source) == request


@pytest.mark.parametrize("external_feed", [False, True])
def test_fixed_baseline_uses_selected_configuration_and_original_snapshot(plan, tmp_path, monkeypatch, external_feed):
    root = Path(plan["service_output"])
    candidate = launch.candidate_list(plan["candidates"], "all")[1]
    request = dict(mode="all", snapshot_id="b" * 64, candidates=launch.candidate_list(plan["candidates"], "all"))
    source_fields = dict(source_feed_job_id="feed123", source_feed_service_output=str(tmp_path / "old_service"),
                         artifact_relocations={"path": "migration.json", "sha256": "fixture"}) if external_feed else {}
    request.update(source_fields)
    launch.write_new(root / "jobs/source123/request.json", request)
    launch.write_new(root / "jobs/source123/state.json", {"status": "succeeded", "result": {"status": "frozen"}})
    launch.write_new(root / "runs/source123/development.json", {"status": "developed", "selected": candidate})
    launch.write_new(root / "runs/source123/run_fingerprint.json", {"test": "fixed"})
    service = SimpleNamespace(output_dir=root, _run_identity=lambda _: {"test": "fixed"})
    monkeypatch.setattr(launch, "TrainingService", lambda *_: service)
    monkeypatch.setattr(launch, "save_ready", lambda svc, req, dest, **kw: {"request": req, **kw})
    result = launch.prepare_baseline(plan, "source123", tmp_path / "baseline")
    baseline = result["request"]
    assert baseline["snapshot_id"] == request["snapshot_id"]
    assert baseline["id"] == "fixed_physical" and baseline["mode"] == "experiment" and baseline["execute"] is True
    assert baseline["model_config"]["hidden"] == 64
    assert baseline["model_config"]["physical_mode"] == "fixed"
    assert baseline["model_config"]["use_condition_effects"] is False
    assert baseline["feed_training"] == candidate["feed_training"]
    assert baseline["training_config"] == candidate["load_training"]
    assert result["provenance"]["residual_training"] is False
    for key, value in source_fields.items():
        assert baseline[key] == value


def test_fixed_baseline_rejects_incomplete_selection(plan, tmp_path):
    root = Path(plan["service_output"])
    launch.write_new(root / "jobs/source123/request.json", {"mode": "all"})
    launch.write_new(root / "jobs/source123/state.json", {"status": "running"})
    with pytest.raises(ValueError, match="等待完整all任务成功"):
        launch.prepare_baseline(plan, "source123", tmp_path / "baseline")


def test_fixed_baseline_public_preview_and_nine_fold_main_only_dispatch(plan, tmp_path, monkeypatch):
    """Exercise real service preview and real development planner without training."""
    from spindle_power.load_prediction.training_service import run_development
    service = launch.TrainingService(tmp_path / "service")
    samples = [SimpleNamespace(metadata={"condition_descriptors": {}})]
    monkeypatch.setattr(service, "load_prepared", lambda _: (samples, samples, {"synthetic": False}))
    monkeypatch.setattr(service, "_validate_prepared", lambda *args, **kwargs: None)
    candidate = launch.candidate_list(plan["candidates"], "all")[0]
    request = dict(id="fixed_physical", snapshot_id="a" * 64, execute=False,
                   model_config=candidate["load_model"], training_config=candidate["load_training"])
    preview = service.experiment(request)
    assert preview["status"] == "prepared_not_executed"
    assert preview["request"]["ablation"] == {"main": "fixed_coefficients", "residual": False, "memory": False}
    planned = run_development([], feed_samples=[], candidates=[candidate], main_only=True, execute=False)
    assert len(planned["folds"]) == 9 and len(planned["phases"]) == 9
    for fold, phase in zip(planned["folds"], planned["phases"]):
        assert len(fold["train_rounds"]) == 80 and len(fold["validation_rounds"]) == 10
        assert not set(fold["train_rounds"]) & set(fold["validation_rounds"])
        assert phase["stages"] == ["feed", "main"]


def test_fixed_baseline_command_uses_original_experiment_cli(tmp_path):
    command = launch.cli_command(tmp_path / "service", tmp_path / "baseline.json", "experiment")
    assert command[-2:] == ["experiment", str((tmp_path / "baseline.json").resolve())]


def test_baseline_requires_original_fold_feed_cache(plan, tmp_path, monkeypatch):
    from spindle_power.load_prediction import training_checkpoint, feed_model, feed_identity
    root = tmp_path / "service"
    chosen = {"id": "C32"}
    fold = launch.protocol_folds()[0]
    source = root / "runs/source123/development.json"
    launch.write_new(source, {"selected": chosen, "candidates": [{"candidate": chosen, "folds": [
        {"fold": fold["fold"], "validation_rounds": fold["validation_rounds"], "feed_metrics": {"model_version": "original_version"}}
    ]}]})
    candidate = launch.candidate_list(plan["candidates"], "feed")[0]
    request = {**candidate, "source_development_job_id": "source123", "source_development_sha256": launch.file_hash(source)}
    service = SimpleNamespace(output_dir=root)
    monkeypatch.setattr(feed_identity, "feed_training_identity", lambda *a, **k: {"frozen": "identity"})
    monkeypatch.setattr(training_checkpoint, "digest", lambda *_: "cachekey")
    monkeypatch.setattr(feed_model, "load_feed_model", lambda _: SimpleNamespace(model_version="original_version"))
    with pytest.raises(FileNotFoundError):
        launch.verify_baseline_feed_cache(service, request, fold, [], [])
    weights = root / "feed_cache/feed_reuse_v1/cachekey/feed.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"fixture_not_network_weights")
    launch.write_new(weights.parent / "model.json", {"identity": {"frozen": "identity"}, "sha256": launch.file_hash(weights)})
    assert launch.verify_baseline_feed_cache(service, request, fold, [], [])["model_version"] == "original_version"
    monkeypatch.setattr(feed_model, "load_feed_model", lambda _: SimpleNamespace(model_version="different_version"))
    with pytest.raises(ValueError, match="模型版本不一致"):
        launch.verify_baseline_feed_cache(service, request, fold, [], [])


def test_baseline_external_feed_uses_verified_fold_reference_without_local_cache(plan, tmp_path, monkeypatch):
    from spindle_power.load_prediction import training_service
    root = tmp_path / "service"
    chosen = {"id": "C32"}
    fold = launch.protocol_folds()[0]
    source = root / "runs/source123/development.json"
    launch.write_new(source, {"selected": chosen, "candidates": [{"candidate": chosen, "folds": [
        {"fold": fold["fold"], "validation_rounds": fold["validation_rounds"],
         "feed_metrics": {"model_version": "original_version"}}
    ]}]})
    candidate = launch.candidate_list(plan["candidates"], "feed")[0]
    relocation = {"path": "migration.json", "sha256": "fixture"}
    request = {**candidate, "source_development_job_id": "source123",
               "source_development_sha256": launch.file_hash(source), "artifact_relocations": relocation}
    reference = {"path": "original/feed.pt", "sha256": "immutable"}
    feed_source = {"source_feed_job_id": "feed123", "folds": {fold["fold"]: reference}}
    calls = []
    def load_reference(ref, train, validation, training_config, model_config, **kwargs):
        calls.append((ref, kwargs))
        return SimpleNamespace(model_version="original_version")
    monkeypatch.setattr(training_service, "_load_feed_reference", load_reference)
    service = SimpleNamespace(output_dir=root)
    result = launch.verify_baseline_feed_cache(service, request, fold, [], [], feed_source=feed_source)
    assert result["model_version"] == "original_version"
    assert calls == [(reference, {"artifact_relocations": relocation})]
    assert not (root / "feed_cache").exists()
    monkeypatch.setattr(training_service, "_load_feed_reference", lambda *a, **kw: SimpleNamespace(model_version="wrong"))
    with pytest.raises(ValueError, match="模型版本不一致"):
        launch.verify_baseline_feed_cache(service, request, fold, [], [], feed_source=feed_source)


@pytest.mark.parametrize("mode,source_id", [("feed", "feed123"), ("all", None)])
def test_source_service_without_valid_mode_and_job_rejected_before_prepare(plan, tmp_path, monkeypatch, mode, source_id):
    monkeypatch.setattr(launch, "assert_runtime", lambda _: None)
    monkeypatch.setattr(launch, "TrainingService", lambda *_: pytest.fail("must reject before reading numerical data"))
    with pytest.raises(ValueError, match="source_feed_job_id"):
        launch.prepare(plan, mode, tmp_path / "ready", source_feed_job_id=source_id,
                       source_feed_service_output=tmp_path / "old_service")
