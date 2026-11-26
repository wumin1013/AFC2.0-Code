"""Synthetic recovery/evaluation contracts; these are not real accuracy evidence."""
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
from spindle_power.load_prediction.common import Cancelled, save_json
from spindle_power.load_prediction.feed_model import FeedModelConfig
from spindle_power.load_prediction.feed_training import FeedTrainingConfig, train_feed_fold
from spindle_power.load_prediction.models import ModelConfig
from spindle_power.load_prediction.memory import PositionMemory
from spindle_power.load_prediction.neural import state_version
from spindle_power.load_prediction.training import TrainingConfig, train_two_stage, save_main_baseline
from spindle_power.load_prediction.training_checkpoint import training_identity, save_checkpoint, load_checkpoint
from spindle_power.load_prediction.training_service import TrainingService, evaluate_records, train_feed_cached, run_development


@pytest.fixture
def data():
    torch.set_num_threads(1)
    return synthetic_records((1, 2), points=12)


def feed_options():
    return dict(model_config=FeedModelConfig(**SMALL_CANDIDATE["feed_model"]),
                config=FeedTrainingConfig(epochs=4, batch_size=4, patience=8, seed=17), execute=True)


def test_snapshot_round_trip_preserves_exact_positions_labels_and_feed(data, tmp_path):
    feeds, loads = deepcopy(data)
    for sample in feeds + loads:
        values = np.resize([.49005016579937954, .9801592778727329, 1.3223992664849737], len(sample.frame))
        sample.frame["segment_mm"] = values
        sample.frame["path_mm"] = values + 100.12345678901234
        sample.frame["feed_actual_mm_min"] = values * 2000.123456789
        if "power_cut_w" in sample.frame:
            sample.frame["power_cut_w"] = values * 300.123456789
    service = TrainingService(tmp_path / "service")
    saved = service.save_prepared(feeds, loads, source_snapshot={}, synthetic=True)
    restored_feeds, restored_loads, _ = service.load_prepared(saved["snapshot_id"])
    for original, restored in zip(feeds + loads, restored_feeds + restored_loads):
        columns = ["segment_mm", "path_mm", "feed_actual_mm_min"]
        if "power_cut_w" in original.frame:
            columns.append("power_cut_w")
        pd.testing.assert_frame_equal(original.frame[columns], restored.frame[columns], check_exact=True)


def load_options():
    return dict(model_config=ModelConfig(**SMALL_CANDIDATE["load_model"]),
                config=TrainingConfig(main_epochs=4, residual_epochs=4, patience=8, seed=17), execute=True)


def test_feed_resume_matches_uninterrupted_weights_history_and_normalization(data, tmp_path):
    feeds, _ = data
    options = feed_options()
    options["model_config"] = replace(options["model_config"], num_layers=2, dropout=.2)
    expected = train_feed_fold(feeds[:5], feeds[5:], **options)
    path = tmp_path / "feed.pt"
    def interrupt(event):
        if event.get("epoch") == 2:
            raise Cancelled("synthetic interruption after saved epoch")
    with pytest.raises(Cancelled):
        train_feed_fold(feeds[:5], feeds[5:], **options, checkpoint_path=path, progress=interrupt)
    actual = train_feed_fold(feeds[:5], feeds[5:], **options, checkpoint_path=path, resume=True)
    assert actual.model_version == expected.model_version
    assert actual.normalization == expected.normalization
    assert actual.metadata["history"] == expected.metadata["history"]
    assert all(torch.equal(actual.network.state_dict()[k], v) for k, v in expected.network.state_dict().items())


@pytest.mark.parametrize("stage", ["main", "residual"])
def test_load_resume_matches_both_stages_and_best_state(data, tmp_path, stage):
    _, loads = data
    expected = train_two_stage(loads[:5], validation=loads[5:], **load_options())
    def interrupt(event):
        if event.get("stage") == stage and event.get("epoch") == 2:
            raise Cancelled("synthetic interruption after saved epoch")
    with pytest.raises(Cancelled):
        train_two_stage(loads[:5], validation=loads[5:], **load_options(), checkpoint_dir=tmp_path, progress=interrupt)
    actual = train_two_stage(loads[:5], validation=loads[5:], **load_options(), checkpoint_dir=tmp_path, resume=True)
    assert (actual.main_version, actual.residual_version) == (expected.main_version, expected.residual_version)
    assert actual.metadata["stages"] == expected.metadata["stages"]
    assert actual.normalization == expected.normalization


def test_checkpoint_rejects_changed_identity_and_corrupt_bytes(data, tmp_path):
    feeds, _ = data
    identity = training_identity("test", feeds[:5], feeds[5:], {"seed": 17})
    path = tmp_path / "state.pt"
    save_checkpoint(path, identity, {"epoch": 1, "completed": False, "weight": torch.ones(2)})
    with pytest.raises(FileExistsError):
        load_checkpoint(path, identity)
    for field in ("configuration", "code", "train"):
        changed = deepcopy(identity)
        changed[field] = "changed"
        with pytest.raises(ValueError, match="指纹"):
            load_checkpoint(path, changed, resume=True)
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="哈希"):
        load_checkpoint(path, identity, resume=True)


def test_main_only_never_optimizes_residual_and_saves_distinct_artifact(data, tmp_path):
    _, loads = data
    options = load_options()
    options["model_config"] = replace(options["model_config"], physical_mode="fixed")
    for sample in loads:
        sample.frame["valid_residual"] = False
    with patch("spindle_power.load_prediction.training._residual_data", side_effect=AssertionError("residual must not train")):
        model = train_two_stage(loads, **options, main_only=True)
    assert model.main_trained and not model.residual_trained
    assert model.metadata["stages"]["residual"]["best_epoch"] == 0
    assert np.isfinite(list(model.normalization.values())).all()
    save_main_baseline(model, tmp_path / "main.pt")
    payload = torch.load(tmp_path / "main.pt", weights_only=True)
    assert payload["state"]["artifact_schema"] == "afc.main_baseline.1"
    with pytest.raises(ValueError):
        model.save(tmp_path / "invalid_load.pt")


def test_observed_label_gaps_do_not_change_prospective_predictions(data, tmp_path):
    feeds, loads = data
    feed = train_feed_fold(feeds[:5], **feed_options())
    bundle = train_two_stage(loads[:5], **load_options())
    for mode in ("predicted", "programmed"):
        reference = evaluate_records(bundle, loads[5:6], feed_samples=feeds[5:6], feed_model=feed,
                                     feed_mode=mode, output_dir=tmp_path / (mode + "_all"))
        changed = deepcopy(loads[5:6])
        changed[0].frame.loc[[2, 5, 8], "valid_residual"] = False
        changed[0].frame.loc[[2, 5, 8], "power_cut_w"] = np.nan
        actual = evaluate_records(bundle, changed, feed_samples=feeds[5:6], feed_model=feed,
                                  feed_mode=mode, output_dir=tmp_path / (mode + "_gap"))
        assert actual["records"][0]["prediction_fingerprint"] == reference["records"][0]["prediction_fingerprint"]
        assert actual["records"][0]["predicted_points"] == reference["records"][0]["predicted_points"]
        assert actual["records"][0]["scored_points"] == reference["records"][0]["scored_points"] - 3


def test_numbered_memory_mode_preserves_unknown_provenance_and_rejects_unknown_quality(data):
    feeds, loads = data
    feed = train_feed_fold(feeds[:5], **feed_options())
    bundle = train_two_stage(loads[:5], **load_options())
    for sample in loads:
        sample.metadata.update(tool_session=None, actual_event_order=None)
    loads[0].metadata["tool_healthy"] = None
    original = deepcopy([s.metadata for s in loads])
    memory = PositionMemory()
    result = evaluate_records(bundle, list(reversed(loads)), feed_samples=feeds, feed_model=feed, memory=memory,
                              analysis_mode="numbered_round_pooled_tool", analysis_tool_group="assumed_same_tool")
    assert [s.metadata for s in loads] == original
    assert [r["record_id"] for r in result["records"]] == [s.metadata["record_id"] for s in loads]
    assert result["memory_analysis"]["rejected_records"] == 1
    assert result["memory_analysis"]["rejection_counts"] == {"paused_quality": 1}
    assert result["memory_analysis"]["assumption"]["interpretation"] == "sensitivity_analysis_only"
    assert result["records"][0]["memory_update"]["rejected_fields"] == ["tool_healthy"]
    assert all(r["acquisition"] == dict(tool_session=None, actual_event_order=None) for r in result["records"])
    assert all(r["memory_analysis"]["event_order"] == i for i, r in enumerate(result["records"]))
    cold = evaluate_records(bundle, loads, feed_samples=feeds, feed_model=feed, memory=PositionMemory())
    assert cold["memory_analysis"]["rejected_records"] == 10
    with pytest.raises(ValueError, match="analysis_tool_group"):
        evaluate_records(bundle, loads, feed_samples=feeds, feed_model=feed, analysis_mode="numbered_round_pooled_tool")


def test_completed_feed_cache_reuses_identical_model_only(data, tmp_path):
    feeds, _ = data
    options = feed_options()
    config = options.pop("config")
    options.pop("execute")
    expected = train_feed_cached(feeds[:5], feeds[5:], training_config=config, cache_dir=tmp_path, **options)
    def dry_run_only(*args, **kwargs):
        assert kwargs.get("execute") is not True, "cache miss must not invoke an optimizer"
        return train_feed_fold(*args, **kwargs)
    with patch("spindle_power.load_prediction.feed_training.train_feed_fold", side_effect=dry_run_only):
        actual = train_feed_cached(feeds[:5], feeds[5:], training_config=config, cache_dir=tmp_path, **options)
    assert actual.model_version == expected.model_version


def test_service_explicit_recovery_copies_history_and_rejects_changed_request(tmp_path):
    service = TrainingService(tmp_path)
    request = dict(snapshot_id="f" * 64, mode="feed", job_id="source_job", candidates=[SMALL_CANDIDATE])
    run, resume = service._prepare_run(request)
    assert not resume
    (run / "preserved.txt").write_text("original", encoding="utf-8")
    save_json(tmp_path / "jobs/source_job/state.json", dict(status="interrupted"))
    recovery = {**request, "job_id": "new_job", "resume_from_job_id": "source_job"}
    new, resume = service._prepare_run(recovery)
    assert resume and (new / "preserved.txt").read_text(encoding="utf-8") == "original"
    assert (run / "preserved.txt").read_text(encoding="utf-8") == "original"
    with pytest.raises(ValueError, match="指纹"):
        service.validate_recovery({**recovery, "mode": "all"})


def test_feed_service_persists_all_folds_and_correct_units(tmp_path):
    feeds, _ = synthetic_records(points=4)
    service = TrainingService(tmp_path / "service")
    snapshot = service.save_prepared(feeds, [], source_snapshot={}, synthetic=True)
    request = dict(snapshot_id=snapshot["snapshot_id"], mode="feed", job_id="feed_artifacts", candidates=[SMALL_CANDIDATE])
    result = service.execute(request)
    development = json.loads(Path(result["development"]).read_text(encoding="utf-8"))
    assert development["units"] == "mm/min"
    assert "rmse_mm_min" in development["candidates"][0]["metrics"]
    assert "rmse_w" not in development["candidates"][0]["metrics"]
    folds = Path(result["development"]).parent / "folds" / SMALL_CANDIDATE["id"]
    assert len(list(folds.glob("*/feed.pt"))) == 9
    assert len(list(folds.glob("*/history.json"))) == 9
    assert len(list(folds.glob("*/predictions/record_*.csv"))) == 450
    assert len(list(folds.glob("*/checkpoints/feed.pt.json"))) == 9


def test_development_three_feed_modes_share_fold_models(data, tmp_path):
    feeds, loads = data
    folds = [dict(fold="test_fold", train_rounds=[1], validation_rounds=[2])]
    with patch("spindle_power.load_prediction.training_service.protocol_folds", return_value=folds), patch(
            "spindle_power.load_prediction.training.train_two_stage", wraps=train_two_stage) as optimize_load:
        result = run_development(loads, feed_samples=feeds, candidates=[SMALL_CANDIDATE], execute=True, output_dir=tmp_path)
    assert optimize_load.call_count == 1
    fold = result["candidates"][0]["folds"][0]
    assert fold["physical_reused_main"]["comparison"]["network_retrained"] is False
    assert fold["physical_reused_main"]["comparison"]["validation_rounds"] == [2]
    assert {r["record_id"] for r in fold["physical_reused_main"]["records"]} == {r["record_id"] for r in fold["without_memory"]["records"]}
    physical_files = list(tmp_path.glob("*/*/physical_reused_main/record_*.csv"))
    assert len(physical_files) == 5
    for artifact in physical_files:
        physical = pd.read_csv(artifact)
        complete = pd.read_csv(artifact.parent.parent / "predicted_feed" / artifact.name)
        np.testing.assert_allclose(physical.cutting_power_w, complete.main_w)
        np.testing.assert_array_equal(physical.scored, complete.scored)
    assert {r["record_id"] for r in fold["without_memory"]["records"]} == {r["record_id"] for r in fold["programmed_feed"]["records"]}
    assert fold["measured_feed_diagnostic"]["interpretation"] == "postmachining_diagnostic"
    assert len(list(tmp_path.glob("*/*/load.pt"))) == 1
    assert len(list(tmp_path.glob("*/*/feed.pt"))) == 1
    from spindle_power.load_prediction.training_service import train_full
    with patch("spindle_power.load_prediction.training_service.train_feed_cached", wraps=train_feed_cached) as train:
        feed, load = train_full(loads, feeds, result, output_dir=tmp_path / "full",
                               checkpoint_dir=tmp_path / "full_checkpoints", feed_cache_dir=tmp_path / "unused_full_cache")
    assert train.call_args.kwargs["cache_dir"] is None
    assert feed.metadata["training_mode"] == "fixed_epochs_full"
    assert feed.metadata["best_epoch"] == 1
    assert load.metadata["stages"]["main"]["best_epoch"] == 1
    assert load.metadata["stages"]["residual"]["best_epoch"] == 1
    assert (tmp_path / "full/feed.pt").is_file() and (tmp_path / "full/load.pt").is_file()


def test_feed_coordinate_mode_is_frozen_and_cannot_mix_at_prediction(data, tmp_path):
    from spindle_power.load_prediction.feed_neural import machine_context, predict_values, FeedModel
    feeds, _ = data
    assert machine_context(feeds[0].metadata)["coordinate_mode"] == "machine"
    for sample in feeds:
        sample.metadata["coordinate_mode"] = "workpiece_cutting"
    model = train_feed_fold(feeds[:5], feeds[5:], **feed_options())
    assert all(c["coordinate_mode"] == "workpiece_cutting" for c in model.compatible_contexts)
    model.save(tmp_path / "feed.pt")
    restored = FeedModel.load(tmp_path / "feed.pt")
    np.testing.assert_array_equal(predict_values(model, feeds[5]), predict_values(restored, feeds[5]))
    incompatible = deepcopy(feeds[5])
    incompatible.metadata.pop("coordinate_mode")
    with pytest.raises(ValueError, match="坐标模式.*兼容域"):
        predict_values(restored, incompatible)
    with pytest.raises(ValueError, match="坐标模式"):
        machine_context({**feeds[0].metadata, "coordinate_mode": "assumed"})


def test_memory_learning_curve_artifacts_score_same_points_before_update(tmp_path):
    import pandas as pd
    feeds, loads = synthetic_records((1, 11, 26), points=8)
    feed = train_feed_fold(feeds[:5], **feed_options())
    bundle = train_two_stage(loads[:5], **load_options())
    report = evaluate_records(bundle, loads, feed_samples=feeds, feed_model=feed,
                              memory=PositionMemory(), output_dir=tmp_path)
    for index, record in enumerate(report["records"]):
        comparison = record["paired_memory_comparison"]
        assert comparison["prior_history_count_min"] == index // 5
        assert comparison["prior_history_count_max"] == index // 5
        assert not comparison["improvement_guaranteed"]
        frame = pd.read_csv(tmp_path / record["prediction_file"])
        valid = frame.scored.to_numpy(bool)
        np.testing.assert_allclose(frame.memory_absolute_error_delta_w[valid],
            np.abs(frame.with_memory_error_w[valid]) - np.abs(frame.without_memory_error_w[valid]))
        assert comparison["mae_delta_w"] == pytest.approx(frame.memory_absolute_error_delta_w[valid].mean())
        assert frame.memory_history_count_before_update.eq(index // 5).all()
    assert report["memory_analysis"]["update_order"] == "score_before_update"
