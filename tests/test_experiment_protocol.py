"""Exact archived experiment mapping, independent of implementation loops."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spindle_power.load_prediction.experiment_protocol import (DEVELOPMENT_ROUNDS, PARAMETER_GROUPS,
    PROGRAM_FEATURES, protocol_folds, round_spec, validate_round_metadata)
from spindle_power.load_prediction.samples import grouped_folds


class ExperimentProtocolTests(unittest.TestCase):
    def test_special_rounds_and_first_last_folds(self):
        self.assertEqual({r: round_spec(r)["combination"] for r in (10, 15, 25, 34, 39)},
                         {10: "S2T2", 15: "S3T3", 25: "S1T3", 34: "S2T3", 39: "S2T2"})
        folds = protocol_folds()
        self.assertEqual(folds[0]["validation_rounds"], [1, 11, 26, 32, 40, 46, 55, 64, 73, 82])
        self.assertEqual(folds[-1]["validation_rounds"], [2, 15, 19, 31, 45, 54, 63, 72, 81, 90])

    def test_ninety_rounds_450_slots_and_parameter_balance(self):
        rounds = [r for f in protocol_folds() for r in f["validation_rounds"]]
        self.assertEqual(sorted(rounds), list(DEVELOPMENT_ROUNDS))
        self.assertEqual(len(rounds), len(set(rounds)))
        self.assertEqual(len(rounds) * len(PROGRAM_FEATURES), 450)
        for fold in protocol_folds():
            self.assertEqual(len(fold["train_rounds"]), 80)
            self.assertFalse(set(fold["train_rounds"]) & set(fold["validation_rounds"]))
            groups = [round_spec(r)["parameter_group"] for r in fold["validation_rounds"]]
            self.assertEqual(groups.count("P0"), 5)
            self.assertEqual({p: groups.count(p) for p in list(PARAMETER_GROUPS)[1:]}, {f"P{i}": 1 for i in range(1, 6)})

    def test_wrong_mapping_cannot_pass_by_preserving_counts(self):
        records = [round_spec(r) for r in DEVELOPMENT_ROUNDS]
        self.assertTrue(validate_round_metadata(records)["valid"])
        records[0]["combination"], records[1]["combination"] = records[1]["combination"], records[0]["combination"]
        with self.assertRaisesRegex(ValueError, "combination"):
            grouped_folds(records)

    def test_returned_definitions_do_not_mutate_protocol(self):
        fold = protocol_folds()
        fold[0]["validation_rounds"].clear()
        self.assertEqual(len(protocol_folds()[0]["validation_rounds"]), 10)
        with self.assertRaises(ValueError):
            round_spec(True)


if __name__ == "__main__":
    unittest.main()
