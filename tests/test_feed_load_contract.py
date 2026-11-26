"""Actual-feed provenance at the feed/load boundary; no production training."""
from __future__ import annotations

import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from dataclasses import asdict
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spindle_power.load_prediction.feed import FeedPlan, MachineConfig
from spindle_power.load_prediction.models import ModelConfig
from spindle_power.load_prediction.samples import (
    PHYSICS_COLUMNS, SampleSet, assert_feed_inputs, path_features,
    prepare_forward_features, prepare_predicted_load_features)
from spindle_power.load_prediction.training import run_grouped_development, train_two_stage


class FeedLoadContractTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(dict(
            line=[2, 2, 2], execution_instance=[0, 0, 0],
            path_mm=[0., .5, 1.], segment_mm=[0., .5, 1.],
            block_length_mm=1., motion=1, x_mm=[0., .5, 1.], y_mm=0., z_mm=0.,
            feed_cmd_mm_min=999., spindle_cmd_rpm=1000., curvature_per_mm=0.))
        self.process = pd.DataFrame(dict(line=[2, 2], execution_instance=[0, 0],
            segment_mm=[0., 1.], ap_mm=[1., 1.], ae_mm=[2., 2.]))
        self.plan = FeedPlan(self.frame, "nc-a", "machine-a", False, {})
        self.meta = dict(program_version="nc-a", diameter_mm=10., teeth=4)

    def prediction(self):
        frame = self.frame.copy()
        frame["feed_predicted_mm_min"] = [310., 390., 420.]
        return SimpleNamespace(frame=frame, metadata=dict(program_version="nc-a", machine_version="machine-a",
            source="bilstm_actual_feed", model_version="feed-v1", trained=True))

    def test_planner_only_is_geometry_preview(self):
        sample = prepare_forward_features(self.process, self.plan, self.meta)
        self.assertTrue(sample.frame.feed_input_mm_min.isna().all())
        self.assertIn("feed_input_mm_min", PHYSICS_COLUMNS)
        self.assertNotIn("feed_cmd_mm_min", PHYSICS_COLUMNS)
        with self.assertRaisesRegex(ValueError, "来源"):
            assert_feed_inputs(sample.frame, "predicted")

    def test_join_uses_program_positions_not_array_order(self):
        prediction = self.prediction()
        prediction.frame = prediction.frame.iloc[::-1]
        sample = prepare_forward_features(self.process, self.plan, self.meta, feed_prediction=prediction)
        np.testing.assert_allclose(sample.frame.feed_input_mm_min, [310., 390., 420.])
        assert_feed_inputs(sample.frame, "predicted")
        self.assertEqual(sample.metadata["feed_model_version"], "feed-v1")
        values = path_features(sample.frame, np.ones(3), 0., 1., 1000.)
        np.testing.assert_allclose(values[:, 3], [.31, .39, .42])
        with self.assertRaisesRegex(ValueError, "来源"):
            assert_feed_inputs(sample.frame, "measured")

    def test_direct_prediction_grid_and_wrong_versions(self):
        prediction = self.prediction()
        result = prepare_predicted_load_features(self.process, prediction, self.meta)
        self.assertEqual(len(result.frame), 3)
        prediction.metadata["program_version"] = "different"
        with self.assertRaisesRegex(ValueError, "版本"):
            prepare_forward_features(self.process, self.plan, self.meta, feed_prediction=prediction)

    def test_untrained_or_ambiguous_prediction_is_rejected(self):
        prediction = self.prediction()
        prediction.metadata["trained"] = False
        with self.assertRaisesRegex(ValueError, "已训练"):
            prepare_predicted_load_features(self.process, prediction, self.meta)
        prediction = self.prediction()
        prediction.frame = pd.concat([prediction.frame, prediction.frame.iloc[:1]])
        with self.assertRaisesRegex(ValueError, "重复"):
            prepare_predicted_load_features(self.process, prediction, self.meta)

    def test_data_supported_forward_reuses_training_geometry_and_requires_it(self):
        from copy import deepcopy
        from unittest.mock import patch
        from spindle_power.load_prediction.samples import _prepare_process_features
        prepared = _prepare_process_features(self.process, self.plan, self.meta)
        prepared.frame["ap_mm"] = 3.
        prepared.metadata["process_mapping_rules_version"] = "afc.data_supported_process_coordinates.2"
        meta = {**self.meta, "position_policy": "command_only_v1",
                "process_mapping_rules_version": "afc.data_supported_process_coordinates.2"}
        before = deepcopy(prepared)
        for prediction in (None, self.prediction()):
            with patch("spindle_power.load_prediction.samples._prepare_process_features", side_effect=AssertionError("no old mapping")):
                result = prepare_forward_features(self.process, self.plan, meta,
                    feed_prediction=prediction, prepared_geometry=prepared)
            self.assertTrue(result.frame.ap_mm.eq(3.).all())
            self.assertEqual(result.metadata["position_policy"], "command_only_v1")
            self.assertEqual(result.metadata["process_mapping_rules_version"], "afc.data_supported_process_coordinates.2")
        direct = prepare_predicted_load_features(self.process, self.prediction(), meta, prepared_geometry=prepared)
        np.testing.assert_allclose(direct.frame.feed_input_mm_min, [310., 390., 420.])
        self.assertTrue(direct.frame.ap_mm.eq(3.).all())
        pd.testing.assert_frame_equal(before.frame, prepared.frame)
        self.assertEqual(before.metadata, prepared.metadata)
        wrong = deepcopy(prepared)
        wrong.metadata["process_mapping_rules_version"] = "afc.process_mapping.2"
        with self.assertRaisesRegex(ValueError, "规则版本"):
            prepare_forward_features(self.process, self.plan, meta, prepared_geometry=wrong)
        with self.assertRaisesRegex(ValueError, "不能退回旧工艺映射"):
            prepare_forward_features(self.process, self.plan, meta)
        with self.assertRaisesRegex(ValueError, "不能退回旧工艺映射"):
            prepare_predicted_load_features(self.process, self.prediction(), meta)

    def test_load_training_rejects_predicted_feed(self):
        sample = prepare_predicted_load_features(self.process, self.prediction(), self.meta)
        sample.frame["valid_main"] = True
        sample.frame["valid_residual"] = True
        sample.metadata["round_id"] = 1
        with self.assertRaisesRegex(ValueError, "来源"):
            train_two_stage([sample], execute=True)

    def test_old_feed_checkpoint_schema_is_not_reinterpreted(self):
        with self.assertRaisesRegex(ValueError, "重新训练"):
            ModelConfig(schema="afc.load.1").validate()

    def test_round_training_requires_independent_feed_supervision(self):
        with self.assertRaisesRegex(ValueError, "实际进给监督"):
            run_grouped_development([], execute=True)

    def test_independent_import_does_not_load_desktop_or_torch(self):
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "MKL_THREADING_LAYER": "SEQUENTIAL"}
        code = ("import sys; import spindle_power.load_prediction.feed_training; "
                "assert 'torch' not in sys.modules; assert 'tkinter' not in sys.modules; "
                "assert 'interval_analysis.main' not in sys.modules")
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_reads_files_and_blocks_incomplete_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "machine.json").write_text(json.dumps(asdict(MachineConfig(
                initial_position_mm=[0., 0., 0.]))), encoding="utf-8")
            channels = [("程序行号", "AX"), ("实际位置", "X轴"), ("实际位置", "Y轴"),
                        ("实际位置", "Z轴"), ("实际速度", "X轴"), ("实际速度", "Y轴"), ("实际速度", "Z轴")]
            header = ["<SmplInfo>,<0>,<1000>"] + [
                f"<ChannelInfo>,<{i}>,<0>,<{signal}>,<0>,<{axis}>"
                for i, (signal, axis) in enumerate(channels)]
            rows = [f"<ChannelData>,1,{x},0,0,600,0,0," for x in [0., .5, 1.]]
            records = []
            for index, name in enumerate(("A", "B", "C")):
                (root / f"{name}.nc").write_text(f"G1 X1 F{600 + index}\n", encoding="utf-8")
                (root / f"{name}.csv").write_text("\n".join(header + rows), encoding="utf-8")
                records.append(dict(program_id=name, nc=f"{name}.nc", measurement=f"{name}.csv",
                    machine="machine.json", feed_override=1., machining_mode="synthetic",
                    machine_settings_version="test-only", metadata=dict(round_id=1, complete=True,
                    alignment_reliable=True, no_alarm=True, round_complete=False)))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(dict(schema="afc.feed_manifest.1", records=records,
                assignment=dict(A="train", B="validation", C="test"))), encoding="utf-8")
            env = {**os.environ, "MKL_THREADING_LAYER": "SEQUENTIAL", "PYTHONIOENCODING": "utf-8"}
            command = [sys.executable, str(ROOT / "scripts" / "actual_feed.py")]
            output = root / "output"
            result = subprocess.run(command + ["check", str(manifest), "--output", str(output)],
                cwd=root, env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((output / "preparation.json").read_text(encoding="utf-8"))
            self.assertFalse(report["ready_to_train"])
            self.assertFalse(report["trained"])
            self.assertIn(90, report["missing_rounds"])
            result = subprocess.run(command + ["train", str(manifest), "--execute", "--output", str(output)],
                cwd=root, env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("未就绪", result.stderr)
            self.assertFalse((output / "feed_model.pt").exists())


if __name__ == "__main__":
    unittest.main()
