"""Small synthetic regressions for stage contracts, never accuracy evidence."""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np
import pandas as pd
import pytest
import torch

from verify_unified_load_prediction import synthetic_records, SMALL_CANDIDATE
from spindle_power.load_prediction.common import file_hash, save_json
from spindle_power.load_prediction.feed_identity import feed_training_identity
from spindle_power.load_prediction.feed_model import FeedModelConfig, load_feed_model
from spindle_power.load_prediction.feed_training import FeedTrainingConfig, train_feed_fold
from spindle_power.load_prediction.models import ModelConfig
from spindle_power.load_prediction.training import TrainingConfig, train_two_stage
from spindle_power.load_prediction.training_checkpoint import load_checkpoint, save_checkpoint
from spindle_power.load_prediction.training_service import (TrainingService, evaluate_records,
    compare_evaluation_artifacts, validate_paper1_fixed_conditions)
from spindle_power.load_prediction.memory import PositionMemory
from spindle_power.load_prediction.metrics import regression_metrics


@pytest.fixture
def small():
    torch.set_num_threads(1)
    feeds, loads = synthetic_records((1, 11, 26), points=8)
    feed = train_feed_fold(feeds[:5], config=FeedTrainingConfig(epochs=1),
        model_config=FeedModelConfig(**SMALL_CANDIDATE["feed_model"]), execute=True)
    load = train_two_stage(loads[:5], config=TrainingConfig(main_epochs=1, residual_epochs=1),
        model_config=ModelConfig(**SMALL_CANDIDATE["load_model"]), execute=True)
    return feeds, loads, feed, load


def test_feed_identity_ignores_added_power_qualification_but_binds_feed_inputs():
    feeds, _ = synthetic_records((1, 2), points=4)
    options = dict(training_config=FeedTrainingConfig(epochs=1), model_config=FeedModelConfig())
    original = feed_training_identity(feeds[:5], feeds[5:], **options)
    changed = deepcopy(feeds)
    for sample in changed:
        sample.frame["power_cut_w"] = 42.
        sample.frame["valid_residual"] = False
        sample.metadata.update(base_qualification={"new": "evidence"}, base_formal_training_release=True)
    assert feed_training_identity(changed[:5], changed[5:], **options) == original
    assert not any(name.startswith("base_") or name in ("training.py", "neural.py", "training_service.py") for name in original["code"])
    changed[0].frame.loc[0, "feed_actual_mm_min"] += 1
    assert feed_training_identity(changed[:5], changed[5:], **options) != original
    assert feed_training_identity(feeds[:5], feeds[5:], **{**options, "training_config": FeedTrainingConfig(epochs=2)}) != original


def test_all_false_score_mask_preserves_predictions_and_grid(small, tmp_path):
    feeds, loads, feed, bundle = small
    first = evaluate_records(bundle, loads[:1], feed_samples=feeds, feed_model=feed, output_dir=tmp_path / "original")
    changed = deepcopy(loads[:1])
    changed[0].frame["valid_residual"] = False
    result = evaluate_records(bundle, changed, feed_samples=feeds, feed_model=feed,
                              memory=PositionMemory(), output_dir=tmp_path / "empty")
    row, original = result["records"][0], first["records"][0]
    assert row["prediction_fingerprint"] == original["prediction_fingerprint"]
    assert row["predicted_points"] == original["predicted_points"]
    assert row["scored_points"] == 0 and row["metrics"]["mae_w"] is None
    assert row["memory_update"] == {"status": "paused_no_scored_labels", "updated": 0}
    data = pd.read_csv(tmp_path / "empty" / row["prediction_file"])
    assert len(data) == len(loads[0].frame) and data.prediction_available.all() and not data.scored.any()
    assert result["records"][0]["paired_memory_comparison"]["mae_delta_w"] is None


def test_missing_known_inputs_keep_unpredictable_record(small, tmp_path):
    feeds, loads, feed, bundle = small
    samples = deepcopy(loads[:1])
    samples[0].frame["ap_mm"] = np.nan
    samples[0].frame["valid_residual"] = False
    result = evaluate_records(bundle, samples, feed_samples=feeds, feed_model=feed, output_dir=tmp_path)
    row = result["records"][0]
    assert row["status"] == "unpredictable_known_inputs"
    assert row["coverage"]["nc_grid_points"] == 8 and row["predicted_points"] == 0
    frame = pd.read_csv(tmp_path / row["prediction_file"])
    assert len(frame) == 8 and not frame.prediction_available.any() and frame.cutting_power_w.isna().all()


def test_recollection_and_ineligible_memory_never_read_or_update(small, tmp_path):
    feeds, loads, feed, bundle = small
    changed = deepcopy(loads)
    changed[5].metadata.update(replaces=["old"], recollection={"actual_date": "2026-10-15"}, memory_eligible=False)
    changed[6].metadata["memory_eligible"] = False
    memory = PositionMemory()
    result = evaluate_records(bundle, changed, feed_samples=feeds, feed_model=feed, memory=memory,
        analysis_mode="numbered_round_pooled_tool", analysis_tool_group="same_spec_assumption", output_dir=tmp_path)
    assert result["records"][5]["memory_update"]["status"] == "paused_recollection_not_in_historical_order"
    assert result["records"][6]["memory_update"]["status"] == "paused_memory_ineligible"
    for index in (5, 6):
        data = pd.read_csv(tmp_path / result["records"][index]["prediction_file"])
        assert data.memory_w.eq(0).all() and data.memory_history_count_before_update.eq(0).all()
    # Their later same-program records have only the first round as history.
    assert result["records"][10]["paired_memory_comparison"]["prior_history_count_max"] == 1
    assert result["records"][11]["paired_memory_comparison"]["prior_history_count_max"] == 1


def test_common_scoring_preserves_native_coverage_and_null_intervals(small, tmp_path):
    feeds, loads, feed, bundle = small
    paths = {name: tmp_path / name for name in ("full", "physical")}
    evaluate_records(bundle, loads[:1], feed_samples=feeds, feed_model=feed, output_dir=paths["full"])
    changed = deepcopy(loads[:1])
    changed[0].frame.loc[:2, "valid_residual"] = False
    evaluate_records(bundle, changed, feed_samples=feeds, feed_model=feed, main_only=True, output_dir=paths["physical"])
    result = compare_evaluation_artifacts(paths, output_dir=tmp_path / "comparison")
    a = result["comparisons"]["full"]["records"][0]
    b = result["comparisons"]["physical"]["records"][0]
    assert a["native_metrics"]["count"] == 8 and a["metrics"]["count"] == b["metrics"]["count"] == 5
    assert a["coverage"]["excluded_from_native_for_common"] == 3
    assert b["metrics"]["coverage"] is None
    assert len(result["paired_differences"]) == 1
    original = pd.read_csv(paths["full"] / "record_0000.csv")
    assert original.native_scored.all()  # No source artifact is rewritten.


@pytest.mark.parametrize("restriction, expected", [
    ({"recollection": {"date": "2026-10-15"}}, "paused_recollection_not_in_historical_order"),
    ({"memory_eligible": False}, "paused_memory_ineligible"),
])
def test_public_evaluation_refuses_memory_without_reading_file(small, tmp_path, restriction, expected):
    _, _, _, bundle = small
    service = TrainingService(tmp_path / "service")
    directory = service.output_dir / "predictions/example"
    directory.mkdir(parents=True)
    frame = pd.DataFrame(dict(line=[1, 1], execution_instance=[1, 1], segment_mm=[0., .5],
        cutting_power_w=[100., 110.], q05_w=[90., 100.], q95_w=[110., 120.],
        main_w=[80., 90.], residual_w=[20., 20.]))
    frame.to_csv(directory / "prediction.csv", index=False)
    save_json(directory / "prediction.json", dict(prediction_id="example", model_id="model",
        metadata={"round_id": 1, **restriction}, frame_sha256=file_hash(directory / "prediction.csv")))
    actual = frame[["line", "execution_instance", "segment_mm"]].copy()
    actual["power_cut_w"] = [102., 108.]
    actual_path = tmp_path / "actual.csv"
    actual.to_csv(actual_path, index=False)
    with patch.object(service, "load_model", return_value=(None, bundle, {})):
        result = service.evaluate(dict(prediction_id="example", aligned_measurement=str(actual_path),
            update_memory=True, memory=str(tmp_path / "does_not_exist.json")))
    assert result["memory_update"] == {"status": expected, "updated": 0}
    assert "memory_file" not in result and not list(directory.glob("memory_*.json"))
    assert result["metrics"]["mae_w"] == 2.


def test_low_load_signed_bias_and_fixed_condition_gate(small):
    feeds, _, _, _ = small
    result = regression_metrics([10., 20., 100.], [15., 18., 80.], low_threshold_w=20.)
    assert result["low_load_bias_w"] == 1.5 and result["low_count"] == 2
    assert validate_paper1_fixed_conditions(feeds)["interactions"] is False
    changed = deepcopy(feeds)
    changed[0].metadata["condition_descriptors"]["machine"]["category"] = "other_machine"
    with pytest.raises(ValueError, match="五分支"):
        validate_paper1_fixed_conditions(changed)


def test_epoch_archives_preserve_each_state_and_detect_tampering(tmp_path):
    path, identity = tmp_path / "feed.pt", {"case": "synthetic"}
    for epoch in (1, 2):
        save_checkpoint(path, identity, {"epoch": epoch, "completed": epoch == 2, "weights": torch.tensor([epoch])})
    archive = tmp_path / "epochs" / "feed"
    index = json.loads((archive / "index.json").read_text())
    assert [row["epoch"] for row in index["epochs"]] == [1, 2]
    assert load_checkpoint(path, identity, resume=True)["epoch"] == 2
    with (archive / "epoch_0001.pt").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="归档"):
        load_checkpoint(path, identity, resume=True)


@pytest.mark.parametrize("remaining", ["companion", "archive"])
def test_missing_latest_never_silently_restarts_started_stage(tmp_path, remaining):
    path, identity = tmp_path / "feed.pt", {"case": "synthetic"}
    assert load_checkpoint(path, identity, resume=True) is None
    if remaining == "companion":
        save_json(path.with_suffix(".pt.json"), {"epoch": 1})
    else:
        (tmp_path / "epochs/feed").mkdir(parents=True)
    with pytest.raises(ValueError, match="latest.*缺失"):
        load_checkpoint(path, identity, resume=True)


@pytest.mark.parametrize("source_location", ["same_service", "external_service", "relocated_service"])
def test_explicit_feed_source_reuses_folds_and_full_without_optimizer(tmp_path, source_location):
    torch.set_num_threads(1)
    feeds, loads = synthetic_records((1, 2), points=4)
    service = TrainingService(tmp_path / "service")
    source_service = service if source_location == "same_service" else TrainingService(tmp_path / "source_service")
    feed_snapshot = source_service.save_prepared(feeds, [], source_snapshot={}, synthetic=True)
    power_feeds = deepcopy(feeds)
    for sample in power_feeds:
        sample.metadata["base_qualification"] = {"added_after_feed": True}
    all_snapshot = service.save_prepared(power_feeds, loads, source_snapshot={}, synthetic=True)
    folds = [dict(fold="F1", train_rounds=[1], validation_rounds=[2]), dict(fold="F2", train_rounds=[2], validation_rounds=[1])]
    source_request = dict(job_id="source_feed", mode="feed", snapshot_id=feed_snapshot["snapshot_id"], candidates=[SMALL_CANDIDATE])
    with patch("spindle_power.load_prediction.training_service.protocol_folds", return_value=folds):
        result = source_service.execute(source_request)
        save_json(source_service.output_dir / "jobs/source_feed/request.json", source_request)
        save_json(source_service.output_dir / "jobs/source_feed/state.json", dict(status="succeeded", result=result))
        source_options = {} if source_service is service else dict(source_feed_service_output=str(source_service.output_dir))
        preserved = ["runs/source_feed/feed_reuse.json", "runs/source_feed/run_fingerprint.json",
                     "jobs/source_feed/request.json", "runs/source_feed/development.json"]
        original_hashes = {name: file_hash(source_service.output_dir / name) for name in preserved}
        resolved_source = source_service.output_dir
        if source_location == "relocated_service":
            resolved_source = tmp_path / "relocated_source_service"
            assert source_service.output_dir.parent == resolved_source.parent == tmp_path
            source_service.output_dir.rename(resolved_source)
            migration = tmp_path / "artifact_relocations.json"
            save_json(migration, [dict(source=str(source_service.output_dir), destination=str(resolved_source), status="moved_verified")])
            source_options["artifact_relocations"] = dict(path=str(migration), sha256=file_hash(migration))
            with pytest.raises(FileNotFoundError):
                service.validate_feed_source_job("source_feed", all_snapshot["snapshot_id"], [SMALL_CANDIDATE],
                    source_feed_service_output=str(source_service.output_dir))
            with pytest.raises(ValueError, match="SHA-256"):
                service.validate_feed_source_job("source_feed", all_snapshot["snapshot_id"], [SMALL_CANDIDATE],
                    **{**source_options, "artifact_relocations": {"path": str(migration), "sha256": "incorrect"}})
        validated = service.validate_feed_source_job("source_feed", all_snapshot["snapshot_id"], [SMALL_CANDIDATE], **source_options)
        assert validated["full_epochs"] == 1 and len(validated["folds"]) == 2
        assert validated["receipt_sha256"] == original_hashes["runs/source_feed/feed_reuse.json"]
        assert validated["artifact_relocations"] == source_options.get("artifact_relocations")
        save_json(resolved_source / "jobs/source_feed/state.json", dict(status="running", result=result))
        with pytest.raises(ValueError, match="成功冻结"):
            service.validate_feed_source_job("source_feed", all_snapshot["snapshot_id"], [SMALL_CANDIDATE], **source_options)
        save_json(resolved_source / "jobs/source_feed/state.json", dict(status="succeeded", result=result))
        with patch("spindle_power.load_prediction.feed_training._optimize", side_effect=AssertionError("feed must not train again")):
            complete = service.execute(dict(job_id="all_load", mode="all", snapshot_id=all_snapshot["snapshot_id"],
                candidates=[SMALL_CANDIDATE], source_feed_job_id="source_feed", **source_options))
            baseline = service.execute(dict(job_id="fixed_baseline", mode="experiment", id="fixed_physical",
                snapshot_id=all_snapshot["snapshot_id"], model_config=SMALL_CANDIDATE["load_model"],
                training_config=SMALL_CANDIDATE["load_training"], feed_model=SMALL_CANDIDATE["feed_model"],
                feed_training=SMALL_CANDIDATE["feed_training"], source_feed_job_id="source_feed", **source_options))
        from spindle_power.load_prediction.artifact_paths import ArtifactResolver
        resolver = ArtifactResolver(source_options.get("artifact_relocations"))
        source = load_feed_model(resolver.resolve(Path(result["directory"]) / "feed.pt"))
        reused = load_feed_model(Path(complete["directory"]) / "feed.pt")
        assert reused.model_version == source.model_version
        assert not (service.output_dir / "runs/all_load/full_checkpoints/feed.pt").exists()
        assert baseline["status"] == "completed"
        for fold in baseline["result"]["development"]["candidates"][0]["folds"]:
            assert fold["feed_metrics"]["model_version"] == validated["folds"][fold["fold"]]["model_version"]
            assert fold["best_epochs"]["residual"] == 0
        baseline_reference = json.loads((service.output_dir / "runs/fixed_baseline/experiment/feed_source_reference.json").read_text())
        assert baseline_reference["receipt_sha256"] == validated["receipt_sha256"]
        if source_service is not service:
            assert not (service.output_dir / "feed_cache").exists()
        assert {name: file_hash(resolved_source / name) for name in preserved} == original_hashes
        history = json.loads((service.output_dir / "runs/all_load/folds/synthetic_1epoch_only/F1/with_memory/evaluation.json").read_text())
        assert history["memory_analysis"]["initial_memory"] == "empty"
        assert {row["round_id"] for row in history["records"]} == {2}
        assert history["memory_analysis"]["history_source"] == "current_fold_validation_only"
        with pytest.raises(ValueError, match="配置"):
            service.validate_feed_source_job("source_feed", all_snapshot["snapshot_id"], [{**SMALL_CANDIDATE, "feed_training": {"epochs": 2}}], **source_options)
        changed_feeds = deepcopy(power_feeds)
        changed_feeds[0].frame.loc[0, "feed_actual_mm_min"] += 1
        changed_snapshot = service.save_prepared(changed_feeds, loads, source_snapshot={}, synthetic=True)
        with pytest.raises(ValueError, match="数值"):
            service.validate_feed_source_job("source_feed", changed_snapshot["snapshot_id"], [SMALL_CANDIDATE], **source_options)
        # Missing/changed explicit source is an error; cache availability cannot rescue it.
        with resolver.resolve(validated["full"]["path"]).open("ab") as stream:
            stream.write(b"changed")
        with pytest.raises(ValueError, match="哈希"):
            service.validate_feed_source_job("source_feed", all_snapshot["snapshot_id"], [SMALL_CANDIDATE], **source_options)
