from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spindle_power.load_prediction.common import Cancelled
from spindle_power.load_prediction.experiments import list_experiments, run_experiment


class ExperimentRegistryTests(unittest.TestCase):
    def context(self):
        return dict(development_data_ready=True, snapshot_id="fixed-snapshot", seed=42)

    def test_complete_registry_and_no_multicondition_claim_from_p_groups(self):
        items = {s["id"]: s for s in list_experiments(self.context())}
        self.assertEqual(len(items), 21)
        for name in ("descriptor_condition", "id_condition", "implicit_condition", "explicit_multipliers",
                     "no_residual_modulation", "residual_modulation", "zero_shot", "four_parameter_calibration",
                     "physical_refit", "output_finetune", "full_retrain"):
            self.assertEqual(items[name]["status"], "awaiting_multicondition_data")
        self.assertEqual(items["fixed_physical"]["status"], "dispatch_ready")
        self.assertTrue(all(s["reports_real_effect"] is False for s in items.values()))

    def test_pending_data_and_dry_run_never_call_runner(self):
        def forbidden(*args, **kwargs):
            self.fail("不应调用执行器")
        self.assertEqual(run_experiment("explicit_multipliers", forbidden, self.context(), execute=True)["status"],
                         "awaiting_multicondition_data")
        self.assertEqual(run_experiment("fixed_physical", forbidden, self.context())["status"], "prepared_not_executed")

    def test_runner_receives_fixed_protocol_and_exact_variant(self):
        received = []
        def runner(request, **kwargs):
            received.append(request)
            return dict(status="evaluated", applied_experiment=request["experiment"], metrics={"mae_w": 1.})
        result = run_experiment("fixed_physical", runner, self.context(), execute=True)
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["real_effect_established"])
        self.assertEqual(received[0]["snapshot_id"], "fixed-snapshot")
        self.assertEqual(received[0]["ablation"]["main"], "fixed_coefficients")
        self.assertIs(received[0]["freeze_test_roles"], True)

    def test_unimplemented_or_wrong_variant_never_becomes_success(self):
        def missing(*args, **kwargs):
            raise NotImplementedError("待共享服务实现")
        self.assertEqual(run_experiment("fixed_physical", missing, self.context(), execute=True)["status"],
                         "awaiting_service_implementation")
        with self.assertRaisesRegex(ValueError, "applied_experiment"):
            run_experiment("fixed_physical", lambda *a, **kw: {"status": "completed"}, self.context(), execute=True)
        with self.assertRaises(Cancelled):
            run_experiment("fixed_physical", missing, self.context(), execute=True, cancel=lambda: True)


if __name__ == "__main__":
    unittest.main()
