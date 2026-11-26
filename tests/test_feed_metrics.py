from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spindle_power.load_prediction.feed_metrics import FeedMetricConfig, evaluate_feed_record, aggregate_feed_metrics


class FeedMetricTests(unittest.TestCase):
    def trajectory(self):
        position = np.arange(12, dtype=float)
        target = 1000. - 100. * np.clip(position - 4., 0., 3.)
        predicted = 1000. - 100. * np.clip(position - 5., 0., 3.)
        frame = pd.DataFrame(dict(path_mm=position, segment_mm=position, block_length_mm=20.,
                                  line=1, execution_instance=0, sequence_id=0, motion=1,
                                  steady=True, valid_label=True, feed_actual_mm_min=target,
                                  entry_angle_rad=np.pi / 4, exit_angle_rad=0.))
        return frame, predicted

    def test_known_one_mm_delayed_deceleration_and_minimum_feed(self):
        frame, predicted = self.trajectory()
        result = evaluate_feed_record(frame, predicted)
        self.assertEqual(result["minimum_feed_error_mm_min"], 0.)
        self.assertEqual(result["deceleration_onset_error_mm"], 1.)
        self.assertEqual(result["actual_events"][0]["path_mm"], 4.)
        self.assertEqual(result["predicted_events"][0]["path_mm"], 5.)
        self.assertEqual(result["missed_deceleration_events"], 0)
        self.assertEqual(result["groups"]["corners"]["points"], 3)
        self.assertEqual(result["groups"]["deceleration"]["points"], 4)
        self.assertGreater(result["groups"]["steady"]["points"], 0)

    def test_missed_deceleration_not_reported_as_zero_error(self):
        frame, predicted = self.trajectory()
        result = evaluate_feed_record(frame, np.full(len(frame), 1000.))
        self.assertEqual(result["missed_deceleration_events"], 1)
        self.assertIsNone(result["deceleration_onset_error_mm"])

    def test_gaps_prevent_inventing_continuous_deceleration(self):
        frame, predicted = self.trajectory()
        frame.loc[5, "valid_label"] = False
        result = evaluate_feed_record(frame, predicted)
        self.assertEqual(result["actual_deceleration_events"], 0)
        self.assertIsNone(result["deceleration_onset_error_mm"])

    def test_complete_record_aggregation_not_point_count_weighted(self):
        records = []
        for index, (count, error) in enumerate(((1000, 1.), (2, 9.))):
            frame = pd.DataFrame(dict(path_mm=np.arange(count, dtype=float), sequence_id=0,
                                      valid_label=True, feed_actual_mm_min=500.))
            metrics = evaluate_feed_record(frame, np.full(count, 500. + error))
            records.append(dict(round_id=index + 1, feature="外形", metrics=metrics))
        report = aggregate_feed_metrics(records)
        self.assertEqual(report["mae_mm_min"], 5.)
        self.assertEqual(report["rmse_mm_min"], 5.)
        self.assertEqual(report["minimum_feed_error_mm_min"], 5.)
        self.assertEqual(report["mae_mm_min_effective_records"], 2)

    def test_metric_rules_are_explicit_and_invalid_prediction_is_not_dropped(self):
        frame, predicted = self.trajectory()
        cfg = replace(FeedMetricConfig(), gradient_threshold_mm_min_per_mm=200.)
        result = evaluate_feed_record(frame, predicted, config=cfg)
        self.assertEqual(result["actual_deceleration_events"], 0)
        self.assertEqual(result["config"]["gradient_threshold_mm_min_per_mm"], 200.)
        predicted[4] = np.nan
        with self.assertRaisesRegex(ValueError, "必须有限"):
            evaluate_feed_record(frame, predicted)


if __name__ == "__main__":
    unittest.main()
