"""Boundary, actual-event ordering and strict reader regression checks."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spindle_power.load_prediction.sample_rules import continuous_ranges, task_masks
from spindle_power.load_prediction.feed_model import FeedModelConfig, sequence_windows
from spindle_power.load_prediction.memory import PositionMemory
from spindle_power.load_prediction.measurement import read_measurement, iter_measurement


class SampleRuleTests(unittest.TestCase):
    def test_rapid_contact_gap_execution_and_repeated_ids_break(self):
        frame = pd.DataFrame({"sequence_id": [0, 0, 0, 1, 1, 0, 0, 0],
                              "motion": [0, 1, 1, 1, 1, 1, 1, 1],
                              "execution_instance": [0, 0, 0, 0, 0, 0, 1, 1],
                              "cutting_contact": [False, True, True, True, False, True, True, True]})
        valid = np.array([True, True, False, True, True, True, True, True])
        self.assertEqual(list(continuous_ranges(frame, valid)), [(0, 1), (1, 2), (3, 4), (4, 5), (5, 6), (6, 8)])
        windows = list(sequence_windows(frame, FeedModelConfig(window_points=20, context_points=20), valid))
        self.assertEqual([(w.start, w.stop) for w in windows], list(continuous_ranges(frame, valid)))

    def test_supervised_gaps_do_not_become_prospective_inputs(self):
        frame = pd.DataFrame({"sequence_id": [0] * 4, "execution_instance": [0] * 4,
                              "valid_label": [True, False, True, True]})
        cfg = FeedModelConfig(window_points=20)
        self.assertEqual([(w.start, w.stop) for w in sequence_windows(frame, cfg)], [(0, 4)])
        self.assertEqual([(w.start, w.stop) for w in sequence_windows(frame, cfg, frame.valid_label)], [(0, 1), (2, 4)])

    def test_contact_and_quality_are_explicit(self):
        frame = pd.DataFrame(dict(motion=[0, 1, 1], feed_actual_mm_min=[200., 200., 200.],
                                  power_cut_w=[0., 20., 30.], steady=True, spindle_stable=True))
        meta = dict(complete=True, no_alarm=True, tool_healthy=True, alignment_reliable=True)
        self.assertFalse(task_masks(frame, meta)["valid_feed"].any())
        frame["ap_mm"], frame["ae_mm"] = [1., 0., 1.], [1., 1., 1.]
        self.assertEqual(task_masks(frame, meta)["valid_main"].tolist(), [False, False, True])
        meta.pop("tool_healthy")
        self.assertFalse(task_masks(frame, meta)["valid_main"].any())

    def test_measured_time_gap_breaks_only_observed_sequence(self):
        frame = pd.DataFrame(dict(sequence_id=0, time_s=[0., .001, .01, .011]))
        self.assertEqual(list(continuous_ranges(frame)), [(0, 2), (2, 4)])
        self.assertEqual(list(continuous_ranges(frame, known_only=True)), [(0, 4)])

    def test_recollection_uses_actual_order_and_unknown_session_pauses(self):
        context = dict(program_version="p", parameter_group="P0", tool_session="t",
                       main_version="m", residual_version="r", protocol_version="test")
        frame = pd.DataFrame(dict(line=[1], segment_mm=[0.], execution_instance=[0]))
        quality = dict(healthy=True, no_alarm=True, complete=True, aligned=True)
        memory = PositionMemory(minimum_repeats=1)
        memory.update_round(context, frame, [10.], 20, event_order=100, **quality)
        self.assertEqual(memory.predict(context, 1, 0., 3, event_order=101), 10.)
        memory.update_round(context, frame, [20.], 3, event_order=101, **quality)
        with self.assertRaisesRegex(ValueError, "当前|顺序"):
            memory.predict(context, 99, 0., 21, event_order=100)
        self.assertEqual(memory.update_round({**context, "tool_session": None}, frame, [20.], 4,
                                            event_order=102, **quality)["status"], "paused_unknown_session")
        with self.assertRaisesRegex(ValueError, "实际采集"):
            memory.predict(context, 1, 0., 999)

    def test_reader_actual_feed_and_original_indices_survive_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("<SmplInfo>,<0>,<1000>\n<ChannelInfo>,<0>,<0>,<实际进给>,<0>,<AX>\n"
                            "<ChannelData>,10,\ncomment\n<ChannelData>,bad,\n<ChannelData>,30,\n", encoding="utf-8")
            chunks = list(iter_measurement(path, chunk_size=2))
            self.assertEqual(chunks[1].source_indices.tolist(), [2])
            record = read_measurement(path)
            self.assertEqual(record.source_indices.tolist(), [0, 1, 2])
            self.assertEqual(record.source_lines.tolist(), [3, 5, 6])
            self.assertEqual(record.valid.tolist(), [True, False, True])
            self.assertIn("actual_feed_mm_min", record.values)
            path.write_bytes(b"\xff\xfe" + b"x")
            with self.assertRaises(UnicodeError):
                read_measurement(path)


if __name__ == "__main__":
    unittest.main()
