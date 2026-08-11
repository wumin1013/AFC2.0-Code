from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection

from project.academic_workbench import AcademicWorkbenchMixin
from project.analysis_export import AnalysisExportMixin
from project.plot_support import PlotSupportMixin
from project.processing_core import ProcessingCoreMixin
from project.sample_manager import SampleManagerMixin
from project.segmentation import SegmentationConfig, SegmentationPipeline
from project.ui_bootstrap import BootstrapUiMixin


def _process_frame(mrr_values: list[float], *, step_mm: float = 0.1) -> pd.DataFrame:
    point_count = len(mrr_values)
    return pd.DataFrame(
        {
            "source_index": range(point_count),
            "line_id": range(100, 100 + point_count),
            "line_no_raw": range(100, 100 + point_count),
            "ap": [0.0 if value == 0.0 else 1.0 for value in mrr_values],
            "ae": mrr_values,
            "F_program": [60.0] * point_count,
            "s": [step_mm] * point_count,
        }
    )


def _test_config(**overrides) -> SegmentationConfig:
    values = {
        "local_window_points": 1,
        "steady_min_plateau_points": 6,
        "min_steady_mm": 0.3,
        "min_entry_mm": 0.0,
        "min_transition_mm": 0.0,
        "min_exit_mm": 0.0,
        "transition_ratio": 0.1,
        "max_segment_atoms": 32,
    }
    values.update(overrides)
    return SegmentationConfig(**values)


def _run(mrr_values: list[float], config: SegmentationConfig | None = None):
    return SegmentationPipeline(config or _test_config()).run(
        _process_frame(mrr_values)
    )


def _entry_record(result) -> dict:
    records = list(result.diagnostics.get("entry_peak_records") or [])
    if len(records) != 1:
        raise AssertionError(f"预期一个加工阶段，实际为 {len(records)} 个")
    return dict(records[0])


def _process_signature(result) -> str:
    repeat = dict(result.diagnostics.get("repeat_run_consistency") or {})
    return str(
        result.diagnostics.get("process_signature")
        or repeat.get("process_signature")
        or repeat.get("input_signature")
        or ""
    )


def _result_shape(result) -> tuple[pd.DataFrame, pd.DataFrame]:
    points = result.point_labels[
        ["source_index", "segment_type", "state_code", "interval_id"]
    ].reset_index(drop=True)
    intervals = result.intervals[
        ["start_idx", "end_idx", "segment_type", "state_code"]
    ].reset_index(drop=True)
    return points, intervals


class _MappingHarness(AcademicWorkbenchMixin):
    """只提供采样投影状态机需要的上下文，不创建 Tk 页面。"""

    def __init__(self, process_result):
        self._latest_segmentation_result = process_result
        self._current_interval_ready = True
        self._current_interval_source = "segmentation"
        self._current_process_signature = _process_signature(process_result)
        self._sample_mapping_status = "pending"
        self._current_mapping_signature = ""
        self._segmentation_sample_projection_records = []
        self._authoritative_segmentation_sample_lookup_cache = None
        self.sample_data_loaded = True
        self.sample_data_line_numbers = [100, 101, 102, 103]
        self.mapping_signature = "mapping-success"
        self.fail_mapping = False
        self.process_records = [
            {
                "interval_id": "SEG0001",
                "start_idx": 0,
                "end_idx": 3,
                "segment_type": "idle",
                "state_code": 0,
            }
        ]

    def _build_segmentation_mapping_signature(self):
        return self.mapping_signature

    def _get_current_segmentation_process_signature(self):
        return self._current_process_signature

    def _get_current_interval_records(self, *, allow_profile_fallback=False):
        self.assert_no_profile_fallback = not allow_profile_fallback
        return [dict(record) for record in self.process_records]

    def _materialize_segmentation_sample_bounds(self, records):
        if self.fail_mapping:
            raise ValueError("程序行映射不唯一")
        return [
            {
                **dict(records[0]),
                "sample_start_idx": 0,
                "sample_end_idx": 3,
                "sample_count": 4,
            }
        ]


class _SequenceProjectionHarness(AcademicWorkbenchMixin):
    """模拟工艺信息 N 列为空、但实际负载行程连续的发布场景。"""

    def __init__(self):
        self.data = [
            {"line_no_raw": None, "_is_synthetic_fill": False}
            for _index in range(6)
        ]
        self.sample_data_loaded = True
        self.sample_data_line_numbers = np.arange(10, 22, dtype=int)
        self.sample_data_point_indices = np.zeros(12, dtype=int)
        self.sample_data_base_blocks = [(0, 11)]

    @staticmethod
    def _resolve_interval_process_bounds(record, process_rows=None):
        return {
            "start_idx": int(record["start_idx"]),
            "end_idx": int(record["end_idx"]),
        }

    @staticmethod
    def format_rg_line_point(line_number, point_index):
        return f"{int(line_number)}.{int(point_index)}"

    @staticmethod
    def get_selected_program_number():
        return "P1"

    @staticmethod
    def get_selected_tool_ranges():
        return [(12, 19)]

    def build_sample_mask(self, program_no=None, tool_ranges=None):
        mask = np.zeros(12, dtype=bool)
        mask[2:10] = True
        return mask


class _QuantizedProjectionHarness(AcademicWorkbenchMixin):
    """模拟短过程区间落在两个实际采样点之间的精确行号映射。"""

    def __init__(self):
        self.data = [
            {"line_no_raw": 20, "_is_synthetic_fill": False}
            for _index in range(3)
        ]
        self.sample_data_line_numbers = np.full(6, 20, dtype=int)
        self.sample_data_base_blocks = [(0, 5)]

    def _get_segmentation_sample_lines(self):
        return self.sample_data_line_numbers

    def _get_current_sample_line_point_context(self, line_numbers=None):
        return {
            "point_indices": np.arange(6, dtype=int),
            "x_positions": 20.0 + np.arange(6, dtype=float) / 6.0,
        }

    @staticmethod
    def _resolve_interval_process_bounds(record, process_rows=None):
        return {
            "start_idx": int(record["start_idx"]),
            "end_idx": int(record["end_idx"]),
        }

    @staticmethod
    def _resolve_interval_process_x_bounds(record, process_bounds=None):
        return {
            "process_start_x": float(record["process_start_x"]),
            "process_display_end_x": float(record["process_display_end_x"]),
        }

    @staticmethod
    def format_rg_line_point(line_number, point_index):
        return f"{int(line_number)}.{int(point_index)}"


class _ValueVar:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _ImmediateRoot:
    def after(self, _delay, callback):
        callback()
        return None

    def after_cancel(self, _job):
        return None


class _CanvasStub:
    def __init__(self):
        self.draw_count = 0

    def draw_idle(self):
        self.draw_count += 1


class _SizedWidgetStub:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def winfo_width(self):
        return self.width

    def winfo_height(self):
        return self.height


class _SizedCanvasStub(_CanvasStub):
    def __init__(self, width, height):
        super().__init__()
        self.widget = _SizedWidgetStub(width, height)

    def get_tk_widget(self):
        return self.widget


class _SampleSelectionHarness(SampleManagerMixin):
    def __init__(self):
        self.sample_program_name = _ValueVar("PHONE")
        self.sample_tool_name = _ValueVar("")
        self.sample_programs = {"PHONE": {"tools": {"T1": []}}}
        self.sample_tool_combo = {}
        self._loading_sample_data = False
        self.matched_process_file_var = _ValueVar("")
        self.mapping_refresh_count = 0
        self.prompt_count = 0

    def build_tool_display_label(self, tool_id, display_ranges):
        return str(tool_id)

    def sync_adjustment_ratio_for_current_view(self):
        return None

    def apply_process_file_for_program(self, program_name):
        return False

    def _has_authoritative_segmentation_state(self):
        return True

    def get_primary_input_file(self):
        return "process-info.txt"

    def _refresh_segmentation_sample_projection(self, **kwargs):
        self.mapping_refresh_count += 1
        return []

    def prompt_process_file_for_program(self, program_name):
        self.prompt_count += 1

    def on_sample_selection_change(self, event=None):
        return None


class _VisualizationHarness(AnalysisExportMixin, PlotSupportMixin):
    def __init__(self, process_result):
        self.data = [{}]
        self._latest_segmentation_result = process_result
        self._current_interval_ready = True
        self._current_interval_source = "segmentation"
        self._current_process_signature = _process_signature(process_result)
        self._sample_mapping_status = "not_available"
        self._current_mapping_signature = ""
        self._segmentation_sample_projection_records = []
        self.sample_data_loaded = False
        self.sample_data_values = None
        self.sample_data_line_numbers = None
        self.sample_data_time_indices = None
        self.sample_data_x_positions = None
        self.sample_data_source = _ValueVar(0)
        self.sample_data_mode = ""
        self.manual_measurement_data = None
        self.preview_plot_max_points = 60000
        self.status_var_data = _ValueVar("")
        self.figures = []
        self.figure_names = []
        self.progress = []
        self.display_prediction_to_generate = None
        self.prediction_refresh_count = 0

    def set_progress(self, value, text):
        self.progress.append((value, text))

    def show_current_figure(self, index=0):
        return None

    def apply_line_axis_on_time(self, ax, sample_mask, max_ticks=60):
        return None

    def apply_line_axis_on_path(self, ax, sample_positions, sample_mask, max_ticks=60):
        return None

    def _refresh_manual_measurement_prediction(self):
        self.prediction_refresh_count += 1
        if self.display_prediction_to_generate is not None:
            self.manual_measurement_data["predicted_load"] = np.asarray(
                self.display_prediction_to_generate,
                dtype=float,
            )
        return self.manual_measurement_data


class _OptionalOverlayVisualizationHarness(
    _VisualizationHarness,
    SampleManagerMixin,
):
    def __init__(self, process_result):
        super().__init__(process_result)
        self.show_measured_curve_var = _ValueVar(True)
        self.show_reconstructed_curve_var = _ValueVar(True)
        self.show_feed_overlay_var = _ValueVar(False)
        self.show_speed_overlay_var = _ValueVar(False)
        self.show_ap_overlay_var = _ValueVar(False)
        self.show_ae_overlay_var = _ValueVar(False)
        self.show_interval_state_var = _ValueVar(True)
        self._optional_overlay_contexts = {}
        self.sample_display_mode = _ValueVar("none")
        self.sample_program_name = _ValueVar("")
        self.sample_tool_name = _ValueVar("")
        self.sample_plot_mode = _ValueVar("overlay")
        self.root = _ImmediateRoot()
        self._selection_change_job = None
        self._pending_selection_signature = None
        self._last_selection_signature = None
        self._loading_sample_data = False

    def _build_aligned_process_geometry_frame(self, raw_line_numbers):
        point_count = len(raw_line_numbers)
        return pd.DataFrame({
            "ap": np.linspace(0.5, 1.5, point_count),
            "ae": np.linspace(1.0, 2.0, point_count),
            "feed_plan": np.linspace(500.0, 800.0, point_count),
            "speed_plan": np.linspace(3500.0, 4500.0, point_count),
        })


class _ProcessExportHarness(AnalysisExportMixin):
    def __init__(self, process_result):
        self._latest_segmentation_result = process_result
        self._sample_mapping_status = "not_available"
        self._current_mapping_signature = ""
        self._segmentation_sample_projection_records = []
        self.sample_programs = {}
        self.status_var_data = _ValueVar("")
        self.data = []
        for source_index in range(len(process_result.point_labels)):
            self.data.append({
                "line_no_raw": source_index,
                "N_str": f"N{source_index + 1}",
                "S": 5000.0,
                "ap": 1.0,
                "ae": float(source_index + 1),
                "feed_effective": 60.0,
                "s": 0.1,
                "gcode_content": f"G1 X{source_index + 1}",
                "_is_synthetic_fill": False,
            })


class _ProcessPathHarness(ProcessingCoreMixin):
    def __init__(self):
        self.gcode_profile = {}
        self.process_input_diagnostics = {}
        self.process_line_number_diagnostics = {}
        self.s_base = _ValueVar(5000.0)
        self.k_base = _ValueVar(1.0)
        self.current_program_speed = _ValueVar(5000.0)

    def get_kc_value(self):
        return 1.0

    def get_ke_value(self):
        return 0.0

    def predict_idle_power(self, speed):
        return 100.0

    def calculate_additional_columns(
        self,
        ap,
        ae,
        feed_rate,
        s,
        current_s,
        s_base,
        k_base,
        **kwargs,
    ):
        mrr = float(ap) * float(ae) * float(feed_rate) / 60.0
        return 0.0, float(ap) * float(ae), mrr, 1.0, 0.0, 100.0 + mrr, 100.0, 0.0


def _fill_heights_at_x(collections, x_values) -> np.ndarray:
    """从测试图元中还原各曲线采样点的背景上沿。"""

    heights = np.full(len(x_values), np.nan, dtype=float)
    for index, x_value in enumerate(np.asarray(x_values, dtype=float)):
        candidates = []
        for collection in collections:
            for path in collection.get_paths():
                vertices = np.asarray(path.vertices, dtype=float)
                matched = np.isclose(vertices[:, 0], x_value, rtol=0.0, atol=1e-10)
                if np.any(matched):
                    candidates.extend(vertices[matched, 1].tolist())
        if candidates:
            heights[index] = float(np.max(candidates))
    return heights


class CurrentRequirementRegressionTests(unittest.TestCase):
    def assert_valid_result(self, result) -> None:
        diagnostics = result.diagnostics
        self.assertEqual(1.0, float(diagnostics["coverage_rate"]))
        self.assertEqual(0, int(diagnostics["gap_count"]))
        self.assertEqual(0, int(diagnostics["overlap_count"]))
        self.assertEqual(0, int(diagnostics["illegal_transition_count"]))
        self.assertTrue(bool(diagnostics["postprocess_validation_passed"]))

    def test_01_entry_endpoint_uses_first_event_and_deterministic_fallback(self):
        cases = {
            "peak_before_steady": {
                "mrr": [0.0] * 3
                + [1.0, 2.0, 3.0, 2.0, 1.5, 2.5]
                + [4.0] * 12
                + [3.0, 2.0, 1.0]
                + [0.0] * 3,
                "decision": "local_peak_before_steady",
                "entry_start": 3,
                "peak": 5,
                "steady_start": 9,
                "entry_end": 5,
            },
            "steady_before_peak": {
                "mrr": [0.0] * 3
                + [3.0, 7.0]
                + [10.0 + index * 0.01 for index in range(12)]
                + [11.0, 12.0, 13.0, 12.0]
                + [0.0] * 3,
                "decision": "steady_before_local_peak",
                "entry_start": 3,
                "peak": 19,
                "steady_start": 5,
                "entry_end": 4,
            },
            "peak_equals_steady_and_entry_is_empty": {
                "mrr": [0.0] * 3 + [10.0] * 12 + [9.0, 8.0, 7.0] + [0.0] * 3,
                "decision": "local_peak_equals_steady_start",
                "entry_start": 3,
                "peak": 3,
                "steady_start": 3,
                "entry_end": 2,
            },
        }
        for name, case in cases.items():
            with self.subTest(name=name):
                result = _run(case["mrr"])
                self.assert_valid_result(result)
                record = _entry_record(result)
                self.assertEqual(case["decision"], record["boundary_decision"])
                self.assertEqual(case["entry_start"], record["entry_start_idx"])
                self.assertEqual(case["peak"], record["local_peak_idx"])
                self.assertEqual(case["steady_start"], record["first_steady_start_idx"])
                self.assertEqual(case["entry_end"], record["selected_entry_end_idx"])
                labels = result.point_labels["segment_type"].astype(str)
                expected_entry = set(
                    range(case["entry_start"], case["entry_end"] + 1)
                )
                actual_entry = set(labels.index[labels.eq("entry")])
                self.assertEqual(expected_entry, actual_entry)

        equal_peak = _run(
            [0.0] * 3
            + [2.0, 5.0, 10.0, 10.0, 10.0, 9.0, 8.0, 7.0]
            + [0.0] * 3
        )
        equal_peak_record = _entry_record(equal_peak)
        self.assertEqual("local_turning_point", equal_peak_record["local_peak_method"])
        self.assertEqual(5, equal_peak_record["local_peak_idx"])
        self.assertEqual(5, equal_peak_record["peak_plateau_start_idx"])
        self.assertEqual(7, equal_peak_record["peak_plateau_end_idx"])
        self.assertEqual(5, equal_peak_record["selected_entry_end_idx"])

        fallback = _run([0.0] * 3 + [1.0, 2.0, 3.0] + [0.0] * 3)
        fallback_record = _entry_record(fallback)
        self.assertIsNone(fallback_record["local_peak_idx"])
        self.assertEqual(5, fallback_record["fallback_peak_idx"])
        self.assertEqual("stage_first_max_fallback", fallback_record["local_peak_method"])
        self.assertEqual("stage_first_max_fallback", fallback_record["boundary_decision"])
        self.assertEqual(5, fallback_record["selected_entry_end_idx"])

    def test_02_steady_parent_and_inner_transitions_are_never_entry(self):
        mrr = (
            [0.0] * 3
            + [3.0, 7.0]
            + [10.0 + index * 0.01 for index in range(12)]
            + [11.0, 12.0, 13.0, 12.0]
            + [0.0] * 3
        )
        result = _run(mrr)
        self.assert_valid_result(result)
        record = _entry_record(result)
        self.assertEqual("steady_before_local_peak", record["boundary_decision"])
        self.assertLess(
            int(record["selected_entry_end_idx"]),
            int(record["first_steady_start_idx"]),
        )

        parent_records = list(result.diagnostics.get("steady_anchor_run_records") or [])
        self.assertTrue(parent_records)
        labels = result.point_labels["segment_type"].astype(str)
        parent = parent_records[0]
        parent_labels = labels.iloc[
            int(parent["start_idx"]):int(parent["end_idx"]) + 1
        ]
        self.assertFalse(parent_labels.eq("entry").any())
        self.assertTrue(parent_labels.eq("steady").any())
        self.assertTrue(parent_labels.eq("transition").any())
        transition_indices = set(labels.index[labels.eq("transition")])
        entry_indices = set(labels.index[labels.eq("entry")])
        self.assertTrue(transition_indices)
        self.assertTrue(transition_indices.isdisjoint(entry_indices))

    def test_03_only_process_fields_affect_signature_and_labels(self):
        mrr = (
            [0.0] * 3
            + [1.0, 2.0, 3.0, 2.0, 1.5, 2.5]
            + [4.0] * 12
            + [3.0, 2.0, 1.0]
            + [0.0] * 3
        )
        frame = _process_frame(mrr)
        config = _test_config()
        baseline = SegmentationPipeline(config).run(frame)
        self.assertTrue(_process_signature(baseline))

        derived = frame.copy()
        derived["MRR_program"] = [-9999.0 + index for index in range(len(frame))]
        derived["P_pred"] = [1000.0 + index * 7.0 for index in range(len(frame))]
        derived["P_idle"] = [5000.0 - index * 11.0 for index in range(len(frame))]
        derived["actual_load"] = [index**2 for index in range(len(frame))]
        derived["actual_speed"] = [8000.0 + index for index in range(len(frame))]
        derived["actual_feed"] = [0.1 * index for index in range(len(frame))]
        derived["Kc"] = [123.0] * len(frame)
        derived["Ke"] = [456.0] * len(frame)
        derived["profile_id"] = [f"profile-{index}" for index in range(len(frame))]
        changed = SegmentationPipeline(
            replace(config, idle_power_tolerance=1.0e9)
        ).run(derived)

        self.assertEqual(_process_signature(baseline), _process_signature(changed))
        baseline_points, baseline_intervals = _result_shape(baseline)
        changed_points, changed_intervals = _result_shape(changed)
        pd.testing.assert_frame_equal(baseline_points, changed_points)
        pd.testing.assert_frame_equal(baseline_intervals, changed_intervals)
        expected_mrr = frame["ap"] * frame["ae"] * frame["F_program"] / 60.0
        pd.testing.assert_series_equal(
            changed.point_labels["MRR_program"].reset_index(drop=True),
            expected_mrr.rename("MRR_program").reset_index(drop=True),
            check_dtype=False,
        )

    def test_04_max_segment_atoms_is_only_a_batch_size(self):
        mrr = (
            [0.0] * 4
            + [1.0, 2.0, 3.0, 2.0]
            + [4.0] * 80
            + [3.0, 2.0, 1.0]
            + [0.0] * 4
            + [2.0, 4.0, 6.0, 5.0]
            + [8.0] * 80
            + [6.0, 4.0, 2.0]
            + [0.0] * 4
        )
        frame = _process_frame(mrr)
        results = {
            batch_size: SegmentationPipeline(
                _test_config(max_segment_atoms=batch_size)
            ).run(frame)
            for batch_size in (32, 48, 64)
        }
        for result in results.values():
            self.assert_valid_result(result)
            self.assertEqual(
                "candidate_scoring_batch_size",
                result.diagnostics["max_segment_atoms_role"],
            )
        reference_points, reference_intervals = _result_shape(results[32])
        reference_signature = _process_signature(results[32])
        for batch_size in (48, 64):
            with self.subTest(max_segment_atoms=batch_size):
                points, intervals = _result_shape(results[batch_size])
                pd.testing.assert_frame_equal(reference_points, points)
                pd.testing.assert_frame_equal(reference_intervals, intervals)
                self.assertEqual(reference_signature, _process_signature(results[batch_size]))

    def test_05_sample_mapping_success_or_failure_preserves_process_result(self):
        process_result = _run(
            [0.0] * 3
            + [1.0, 2.0, 3.0, 2.0]
            + [4.0] * 12
            + [3.0, 2.0, 1.0]
            + [0.0] * 3
        )
        before_points, before_intervals = _result_shape(process_result)
        before_signature = _process_signature(process_result)
        harness = _MappingHarness(process_result)

        projected = harness._refresh_segmentation_sample_projection()
        self.assertIsNotNone(projected)
        self.assertEqual("valid", harness._sample_mapping_status)
        self.assertEqual("mapping-success", harness._current_mapping_signature)
        self.assertTrue(process_result.diagnostics["sample_projection"]["valid"])
        self.assertEqual(before_signature, _process_signature(process_result))
        after_success_points, after_success_intervals = _result_shape(process_result)
        pd.testing.assert_frame_equal(before_points, after_success_points)
        pd.testing.assert_frame_equal(before_intervals, after_success_intervals)

        harness.fail_mapping = True
        harness.mapping_signature = "mapping-failure"
        harness._sample_mapping_status = "pending"
        harness._current_mapping_signature = ""
        projected = harness._refresh_segmentation_sample_projection()
        self.assertIsNone(projected)
        self.assertEqual("failed", harness._sample_mapping_status)
        self.assertEqual("mapping-failure", harness._current_mapping_signature)
        projection = process_result.diagnostics["sample_projection"]
        self.assertFalse(projection["valid"])
        self.assertIn("不唯一", projection["reason"])
        self.assertEqual(before_signature, _process_signature(process_result))
        after_failure_points, after_failure_intervals = _result_shape(process_result)
        pd.testing.assert_frame_equal(before_points, after_failure_points)
        pd.testing.assert_frame_equal(before_intervals, after_failure_intervals)

        sequence_harness = _SequenceProjectionHarness()
        sequence_records = [
            {"interval_id": "SEG0001", "start_idx": 0, "end_idx": 1, "segment_type": "idle"},
            {"interval_id": "SEG0002", "start_idx": 2, "end_idx": 3, "segment_type": "steady"},
            {"interval_id": "SEG0003", "start_idx": 4, "end_idx": 5, "segment_type": "exit"},
        ]
        sequence_projection = sequence_harness._materialize_segmentation_sample_bounds(
            sequence_records
        )
        self.assertEqual(
            [(2, 3), (4, 6), (7, 9)],
            [
                (record["sample_start_idx"], record["sample_end_idx"])
                for record in sequence_projection
            ],
        )
        self.assertTrue(all(
            record["mapping_source"] == "journey_order_ratio_missing_n"
            for record in sequence_projection
        ))
        self.assertEqual(
            list(range(2, 10)),
            [
                index
                for record in sequence_projection
                for index in range(
                    record["sample_start_idx"],
                    record["sample_end_idx"] + 1,
                )
            ],
        )

        partial_n_harness = _SequenceProjectionHarness()
        partial_n_harness.data[0]["line_no_raw"] = 10
        with self.assertRaisesRegex(ValueError, "N 行号部分缺失"):
            partial_n_harness._materialize_segmentation_sample_bounds(sequence_records)

        quantized_harness = _QuantizedProjectionHarness()
        quantized_records = [
            {
                "interval_id": "SEG0001",
                "start_idx": 0,
                "end_idx": 0,
                "segment_type": "idle",
                "process_start_x": 20.0,
                "process_display_end_x": 20.2,
            },
            {
                "interval_id": "SEG0002",
                "start_idx": 1,
                "end_idx": 1,
                "segment_type": "entry",
                "process_start_x": 20.2,
                "process_display_end_x": 20.21,
            },
            {
                "interval_id": "SEG0003",
                "start_idx": 2,
                "end_idx": 2,
                "segment_type": "steady",
                "process_start_x": 20.21,
                "process_display_end_x": 21.0,
            },
        ]
        quantized_projection = quantized_harness._materialize_segmentation_sample_bounds(
            quantized_records
        )
        self.assertEqual(3, len(quantized_projection))
        self.assertTrue(all(
            record["mapping_source"] == "program_line_and_point_order_quantized"
            for record in quantized_projection
        ))
        self.assertTrue(all(
            int(record["sample_count"]) >= 1
            for record in quantized_projection
        ))
        self.assertEqual(
            list(range(6)),
            [
                index
                for record in quantized_projection
                for index in range(
                    record["sample_start_idx"],
                    record["sample_end_idx"] + 1,
                )
            ],
        )

        plotter = PlotSupportMixin()
        figure, axis = plt.subplots()
        try:
            states = ("idle", "entry", "steady", "transition", "nonsteady", "exit")
            records = [
                {
                    "display_start_x": float(index),
                    "display_end_x": float(index + 1),
                    "segment_type": states[index % len(states)],
                }
                for index in range(600)
            ]
            plotter.draw_full_path_segmentation_background(
                axis,
                records,
                show_labels=True,
                mark_boundaries=True,
            )
            self.assertEqual(0, len(axis.patches))
            self.assertLessEqual(len(axis.collections), len(states) + 1)
        finally:
            plt.close(figure)

        export_harness = AnalysisExportMixin()
        export_harness._current_interval_source = "segmentation"
        export_harness._sample_mapping_status = "failed"
        export_harness._current_mapping_signature = "mapping-failure"
        export_harness._segmentation_sample_projection_records = []
        with (
            patch("builtins.open") as mocked_open,
            patch("project.analysis_export.messagebox.showwarning") as warning,
        ):
            export_harness._do_save_interval_info([])
        warning.assert_called_once()
        mocked_open.assert_not_called()

        export_harness.figures = [object()]
        export_harness.status_var_data = _ValueVar("")
        progress = []
        export_harness.set_progress = lambda value, text: progress.append((value, text))
        self.assertTrue(
            export_harness._finish_segmentation_fast_plot(True, "过程域图表已生成")
        )
        self.assertEqual(100, progress[-1][0])
        self.assertIn("图表已生成", export_harness.status_var_data.get())

        sample_harness = _SampleSelectionHarness()
        sample_harness.on_sample_program_selected()
        self.assertEqual(1, sample_harness.mapping_refresh_count)
        self.assertEqual(0, sample_harness.prompt_count)
        self.assertIn("使用已划分工艺信息", sample_harness.matched_process_file_var.get())

        visualization = _VisualizationHarness(process_result)
        self.assertTrue(visualization.generate_plots(silent=True))
        process_figure = visualization.figures[0]
        process_axis = process_figure.axes[0]
        try:
            self.assertEqual(1, len(process_figure.axes))
            self.assertIn(
                "程序 MRR",
                [line.get_label() for line in process_axis.lines],
            )
            process_fills = [
                artist
                for artist in process_axis.collections
                if isinstance(artist, PolyCollection)
            ]
            self.assertLessEqual(len(process_fills), 6)
            process_fill_y = np.concatenate([
                path.vertices[:, 1]
                for artist in process_fills
                for path in artist.get_paths()
            ])
            self.assertAlmostEqual(
                float(process_result.point_labels["MRR_program"].max()),
                float(np.max(process_fill_y)),
            )
            np.testing.assert_allclose(
                _fill_heights_at_x(
                    process_fills,
                    (
                        process_result.point_labels["path_start"].to_numpy(dtype=float)
                        + process_result.point_labels["path_end"].to_numpy(dtype=float)
                    )
                    * 0.5,
                ),
                np.maximum(
                    process_result.point_labels["MRR_program"].to_numpy(dtype=float),
                    0.0,
                ),
            )
            self.assertEqual(0.0, float(np.min(process_fill_y)))
            self.assertEqual(0, len(process_axis.patches))
            self.assertEqual(100, visualization.progress[-1][0])
        finally:
            plt.close(process_figure)

        point_count = len(process_result.point_labels)
        predicted_load = np.linspace(10.0, 40.0, point_count)
        actual_load = np.linspace(50.0, 80.0, point_count)
        actual_load[5] = np.nan
        mapping_records = []
        for record in process_result.intervals.to_dict(orient="records"):
            mapping_records.append({
                **record,
                "sample_start_idx": int(record["start_idx"]),
                "sample_end_idx": int(record["end_idx"]),
            })
        visualization.sample_data_loaded = True
        visualization.sample_data_values = actual_load.reshape(-1, 1)
        visualization.sample_data_line_numbers = np.arange(
            100,
            100 + point_count,
            dtype=int,
        )
        visualization.sample_data_time_indices = np.arange(point_count, dtype=float)
        visualization.sample_data_mode = "experiment_measurement"
        visualization.manual_measurement_data = {
            "actual_load": actual_load,
        }
        visualization.display_prediction_to_generate = predicted_load
        visualization._sample_mapping_status = "valid"
        visualization._current_mapping_signature = "mapping-valid"
        visualization._segmentation_sample_projection_records = mapping_records
        visualization.progress = []
        self.assertTrue(visualization.generate_plots(silent=True))
        sample_figure = visualization.figures[0]
        sample_axis = sample_figure.axes[0]
        try:
            self.assertEqual(1, len(sample_figure.axes))
            line_labels = [line.get_label() for line in sample_axis.lines]
            self.assertIn("实际负载", line_labels)
            self.assertNotIn("预测负载", line_labels)
            self.assertNotIn("程序 MRR", line_labels)
            self.assertEqual(0, visualization.prediction_refresh_count)
            actual_line = next(
                line for line in sample_axis.lines if line.get_label() == "实际负载"
            )
            self.assertTrue(np.isnan(np.asarray(actual_line.get_ydata(), dtype=float)).any())
            sample_fills = [
                artist
                for artist in sample_axis.collections
                if isinstance(artist, PolyCollection)
            ]
            self.assertLessEqual(len(sample_fills), 6)
            sample_fill_y = np.concatenate([
                path.vertices[:, 1]
                for artist in sample_fills
                for path in artist.get_paths()
            ])
            self.assertAlmostEqual(
                float(np.nanmax(actual_load)),
                float(np.max(sample_fill_y)),
            )
            np.testing.assert_allclose(
                _fill_heights_at_x(
                    sample_fills,
                    visualization.sample_data_time_indices,
                ),
                np.maximum(actual_load, 0.0),
                equal_nan=True,
            )
            self.assertEqual(0.0, float(np.min(sample_fill_y)))
            self.assertEqual(0, len(sample_axis.patches))
            self.assertEqual(100, visualization.progress[-1][0])
        finally:
            plt.close(sample_figure)

        visualization.progress = []
        self.assertTrue(visualization.generate_plots(silent=True))
        self.assertEqual(0, visualization.prediction_refresh_count)
        plt.close(visualization.figures[0])

        visualization.manual_measurement_data = {"actual_load": actual_load}
        incomplete_prediction = predicted_load.copy()
        incomplete_prediction[5] = np.nan
        visualization.display_prediction_to_generate = incomplete_prediction
        visualization.progress = []
        self.assertTrue(visualization.generate_plots(silent=True))
        fallback_figure = visualization.figures[0]
        fallback_axis = fallback_figure.axes[0]
        try:
            self.assertEqual(1, len(fallback_figure.axes))
            fallback_labels = [line.get_label() for line in fallback_axis.lines]
            self.assertIn("实际负载", fallback_labels)
            self.assertNotIn("预测负载", fallback_labels)
            self.assertNotIn("程序 MRR", fallback_labels)
            self.assertEqual(0, len(fallback_axis.patches))
            self.assertEqual(100, visualization.progress[-1][0])
        finally:
            plt.close(fallback_figure)

        visualization._refresh_segmentation_sample_projection = (
            lambda **kwargs: [dict(record) for record in mapping_records]
        )
        with TemporaryDirectory() as temporary_dir:
            export_dir = Path(temporary_dir)
            stale_projection = export_dir / "sample_projection.csv"
            stale_overview = export_dir / "sample_overview.png"
            stale_projection.write_text("旧投影", encoding="utf-8")
            stale_overview.write_bytes(b"old-overview")
            visualization.export_latest_segmentation_result(
                process_result,
                export_dir,
            )
            self.assertTrue(stale_projection.exists())
            self.assertTrue(stale_overview.exists())
            self.assertTrue((export_dir / "point_labels.csv").exists())
            self.assertTrue((export_dir / "intervals.csv").exists())
            self.assertTrue((export_dir / "overview.png").exists())
            self.assertTrue(
                (export_dir / "sample_mapping_diagnostics.json").exists()
            )

        after_visualization_points, after_visualization_intervals = _result_shape(
            process_result
        )
        self.assertEqual(before_signature, _process_signature(process_result))
        pd.testing.assert_frame_equal(before_points, after_visualization_points)
        pd.testing.assert_frame_equal(before_intervals, after_visualization_intervals)

        boundary_figure, boundary_axis = plt.subplots()
        try:
            boundary_points = pd.DataFrame({
                "path_start": [0.0, 1.0],
                "path_end": [1.0, 11.0],
                "s": [1.0, 11.0],
                "MRR_program": [2.0, 4.0],
                "segment_type": ["idle", "steady"],
            })
            plotter.draw_process_mrr_segmentation(
                boundary_axis,
                boundary_points,
                [],
            )
            fills_by_label = {
                artist.get_label(): artist
                for artist in boundary_axis.collections
                if isinstance(artist, PolyCollection)
            }
            idle_fill = next(
                artist for label, artist in fills_by_label.items() if "[0]" in label
            )
            steady_fill = next(
                artist for label, artist in fills_by_label.items() if "[2]" in label
            )
            self.assertAlmostEqual(
                1.0,
                float(np.max(idle_fill.get_paths()[0].vertices[:, 0])),
            )
            self.assertAlmostEqual(
                1.0,
                float(np.min(steady_fill.get_paths()[0].vertices[:, 0])),
            )
        finally:
            plt.close(boundary_figure)

        gap_figure, gap_axis = plt.subplots()
        try:
            gap_artists = plotter.draw_segmentation_curve_background(
                gap_axis,
                [0.0, 1.0, 2.0],
                [1.0, np.nan, 1.0],
                {"steady": np.ones(3, dtype=bool)},
            )
            gap_paths = gap_artists[0].get_paths()
            self.assertEqual(2, len(gap_paths))
            self.assertAlmostEqual(
                0.5,
                float(np.max(gap_paths[0].vertices[:, 0])),
            )
            self.assertAlmostEqual(
                1.5,
                float(np.min(gap_paths[1].vertices[:, 0])),
            )
        finally:
            plt.close(gap_figure)

    def test_06_process_only_save_exports_state_codes_without_rg(self):
        process_result = _run(
            [0.0] * 3
            + [1.0, 2.0, 3.0, 2.0]
            + [4.0] * 12
            + [3.0, 2.0, 1.0]
            + [0.0] * 3
        )
        harness = _ProcessExportHarness(process_result)

        with (
            TemporaryDirectory() as temporary_dir,
            patch("project.analysis_export.OUTPUT_DIR", Path(temporary_dir)),
            patch("project.analysis_export.messagebox.showinfo") as showinfo,
        ):
            harness.save_interval_info()
            process_info_path = Path(temporary_dir) / "ProcessInfo.csv"
            self.assertTrue(process_info_path.exists())
            self.assertFalse((Path(temporary_dir) / "SampleData.rg").exists())
            self.assertFalse((Path(temporary_dir) / "segmentation").exists())
            exported = pd.read_csv(process_info_path, encoding="utf-8-sig")

        self.assertEqual("state_code", exported.columns[-1])
        expected_codes = (
            process_result.point_labels
            .sort_values("source_index")["state_code"]
            .astype(int)
            .tolist()
        )
        self.assertEqual(expected_codes, exported["state_code"].astype(int).tolist())
        self.assertTrue(exported["state_code"].isin(range(6)).all())
        showinfo.assert_called_once()
        self.assertIn("未生成或覆盖 SampleData.rg", showinfo.call_args.args[1])

    def test_07_process_file_without_s_uses_gcode_geometry_or_sequence_fallback(self):
        harness = _ProcessPathHarness()
        partial_n_rows = [
            {"line_no_raw": 14, "gcode_content": "G0 X-35 Y-40", "_is_synthetic_fill": False},
            {"line_no_raw": None, "gcode_content": "G0 X-35 Y-40", "_is_synthetic_fill": False},
            {"line_no_raw": 19, "gcode_content": "G1 Z-0.5 F100", "_is_synthetic_fill": False},
            {"line_no_raw": None, "gcode_content": "G1 Z-0.5 F100", "_is_synthetic_fill": False},
        ]
        line_diagnostics = harness._restore_missing_line_numbers_from_nc_profile(
            partial_n_rows
        )
        self.assertEqual([14, 14, 19, 19], [row["line_no_raw"] for row in partial_n_rows])
        self.assertEqual(
            "input_with_gcode_group_completion",
            line_diagnostics["line_number_source"],
        )
        self.assertEqual(2, line_diagnostics["gcode_group_completed_row_count"])

        header = "N,S(r/min),ap(mm),ae(mm),F(mm/min),MRR(mm3/min),G"
        parsed_header, layout = harness.parse_gcode_line(
            header,
            return_layout=True,
        )
        self.assertIsNone(parsed_header)
        self.assertEqual("export_no_seq_no_s", layout)

        state_header = f"{header},state_code"
        parsed_header, state_layout = harness.parse_gcode_line(
            state_header,
            return_layout=True,
        )
        self.assertIsNone(parsed_header)
        self.assertEqual(
            "export_no_seq_no_s__with_state_code",
            state_layout,
        )
        parsed_with_state, state_layout = harness.parse_gcode_line(
            "N1,5000,1,1,60,1,G1 X1,2",
            layout_hint=state_layout,
            return_layout=True,
        )
        self.assertEqual("G1 X1", parsed_with_state["gcode_content"])
        self.assertEqual(
            "export_no_seq_no_s__with_state_code",
            state_layout,
        )

        rows = []
        for line in (
            "N1,5000,1,1,60,1,G1 X1",
            "N2,5000,1,1,60,1,G1 X3",
        ):
            parsed, layout = harness.parse_gcode_line(
                line,
                layout_hint=layout,
                return_layout=True,
            )
            self.assertIsNotNone(parsed)
            self.assertFalse(parsed["path_column_present"])
            self.assertEqual(5000.0, parsed["spindle_speed"])
            rows.append({
                "line_no_raw": parsed["line_number"],
                "gcode_content": parsed["gcode_content"],
                "ap": parsed["ap"],
                "ae": parsed["ae"],
                "feed_effective": parsed["feed_rate"],
                "S": parsed["spindle_speed"],
                "s": 0.0,
                "_input_path_column_present": parsed["path_column_present"],
                "_input_path_value": parsed["path_value"],
                "_input_path_semantics_hint": None,
                "_has_input_spindle_speed": True,
                "_has_input_path_bounds": False,
                "_is_synthetic_fill": False,
            })

        harness._apply_nc_profile_to_process_rows(rows, origin=(0.0, 0.0, 0.0))
        self.assertEqual("gcode_geometry", harness.process_path_source)
        self.assertTrue(harness.process_path_is_physical)
        np.testing.assert_allclose([row["s"] for row in rows], [1.0, 2.0])
        np.testing.assert_allclose([row["path_end"] for row in rows], [1.0, 3.0])

        no_geometry_rows = [
            {
                **dict(row),
                "gcode_content": "M3",
                "s": 0.0,
                "path_start": 0.0,
                "path_end": 0.0,
                "path_cumulative": 0.0,
            }
            for row in rows
        ]
        harness._apply_nc_profile_to_process_rows(
            no_geometry_rows,
            origin=(0.0, 0.0, 0.0),
        )
        self.assertEqual("sequential_fallback", harness.process_path_source)
        self.assertFalse(harness.process_path_is_physical)
        np.testing.assert_allclose(
            [row["s"] for row in no_geometry_rows],
            [1.0, 1.0],
        )

    def test_08_process_preview_supports_all_optional_curves_and_state_toggle(self):
        process_result = _run(
            [0.0] * 3
            + [1.0, 2.0, 3.0, 2.0]
            + [4.0] * 12
            + [3.0, 2.0, 1.0]
            + [0.0] * 3
        )
        visualization = _OptionalOverlayVisualizationHarness(process_result)
        point_count = len(process_result.point_labels)
        visualization.data = [
            {"S": 4000.0 + index * 10.0}
            for index in range(point_count)
        ]

        expected_labels = {
            "show_feed_overlay_var": "F(程序进给)",
            "show_speed_overlay_var": "S(主轴转速)",
            "show_ap_overlay_var": "ap(切深)",
            "show_ae_overlay_var": "ae(切宽)",
        }
        curve_variables = [
            visualization.show_feed_overlay_var,
            visualization.show_speed_overlay_var,
            visualization.show_ap_overlay_var,
            visualization.show_ae_overlay_var,
        ]
        for variable_name, expected_label in expected_labels.items():
            for variable in curve_variables:
                variable.set(False)
            getattr(visualization, variable_name).set(True)
            overlays = visualization.get_optional_process_overlays(
                process_result.point_labels
            )
            self.assertEqual([expected_label], [item["label"] for item in overlays])

        for variable in curve_variables:
            variable.set(True)
        self.assertTrue(visualization.generate_plots(silent=True))
        process_figure = visualization.figures[0]
        process_axis = process_figure.axes[0]
        try:
            self.assertEqual(5, len(process_figure.axes))
            self.assertEqual("工艺信息与区间状态", process_axis.get_title())
            self.assertEqual("行程", process_axis.get_xlabel())
            self.assertIn(r"\mathrm{mm^3/s}", process_axis.get_ylabel())
            overlay_labels = {
                line.get_label()
                for axis in process_figure.axes[1:]
                for line in axis.lines
            }
            self.assertEqual(set(expected_labels.values()), overlay_labels)
            process_figure.canvas.draw()
        finally:
            plt.close(process_figure)

        for variable in curve_variables:
            variable.set(False)
        visualization.show_interval_state_var.set(False)
        self.assertTrue(visualization.generate_plots(silent=True))
        states_hidden_figure = visualization.figures[0]
        try:
            state_fills = [
                artist
                for artist in states_hidden_figure.axes[0].collections
                if isinstance(artist, PolyCollection)
            ]
            self.assertEqual([], state_fills)
        finally:
            plt.close(states_hidden_figure)

        visualization.show_interval_state_var.set(True)
        visualization.show_feed_overlay_var.set(True)
        visualization.on_optional_overlay_toggle()
        callback_figure = visualization.figures[0]
        try:
            self.assertIsNot(states_hidden_figure, callback_figure)
            self.assertEqual(2, len(callback_figure.axes))
            self.assertIn(
                "F(程序进给)",
                [line.get_label() for line in callback_figure.axes[-1].lines],
            )
        finally:
            plt.close(callback_figure)

    def test_09_mapped_preview_respects_curve_and_state_toggles(self):
        process_result = _run(
            [0.0] * 3
            + [1.0, 2.0, 3.0, 2.0]
            + [4.0] * 12
            + [3.0, 2.0, 1.0]
            + [0.0] * 3
        )
        point_count = len(process_result.point_labels)
        actual_load = np.linspace(50.0, 80.0, point_count)
        predicted_load = np.linspace(10.0, 40.0, point_count)
        actual_speed = np.linspace(4000.0, 5000.0, point_count)
        actual_feed = np.linspace(600.0, 900.0, point_count)
        mapping_records = [
            {
                **record,
                "sample_start_idx": int(record["start_idx"]),
                "sample_end_idx": int(record["end_idx"]),
            }
            for record in process_result.intervals.to_dict(orient="records")
        ]

        visualization = _OptionalOverlayVisualizationHarness(process_result)
        visualization.data = [
            {
                "ap": float(process_result.point_labels.iloc[index]["ap"]),
                "ae": float(process_result.point_labels.iloc[index]["ae"]),
                "S": float(actual_speed[index]),
            }
            for index in range(point_count)
        ]
        visualization.sample_data_loaded = True
        visualization.sample_data_values = np.column_stack(
            [actual_load, actual_speed, actual_feed]
        )
        visualization.sample_data_line_numbers = np.arange(
            100,
            100 + point_count,
            dtype=int,
        )
        visualization.sample_data_time_indices = np.arange(
            point_count,
            dtype=float,
        )
        visualization.sample_data_mode = "experiment_measurement"
        visualization.manual_measurement_data = {
            "actual_load": actual_load,
            "predicted_load": predicted_load,
        }
        visualization._sample_mapping_status = "valid"
        visualization._current_mapping_signature = "mapping-valid"
        visualization._segmentation_sample_projection_records = mapping_records

        self.assertTrue(visualization.generate_plots(silent=True))
        figure = visualization.figures[0]
        try:
            self.assertEqual("实际负载与区间划分", figure.axes[0].get_title())
            self.assertIn(id(figure), visualization._optional_overlay_contexts)
            self.assertEqual(1, len(figure.axes))
            visualization._current_preview_fig = figure
            visualization.canvas_data = _CanvasStub()

            visualization.show_feed_overlay_var.set(True)
            visualization.show_speed_overlay_var.set(True)
            visualization.show_ap_overlay_var.set(True)
            visualization.show_ae_overlay_var.set(True)
            visualization.on_optional_overlay_toggle()
            self.assertEqual(5, len(figure.axes))
            self.assertEqual(1, visualization.canvas_data.draw_count)
            overlay_labels = {
                line.get_label()
                for axis in figure.axes[1:]
                for line in axis.lines
            }
            self.assertEqual(
                {
                    "F(实际进给)",
                    "S(实际转速)",
                    "ap(切深)",
                    "ae(切宽)",
                },
                overlay_labels,
            )

            visualization.show_feed_overlay_var.set(False)
            visualization.show_speed_overlay_var.set(False)
            visualization.show_ap_overlay_var.set(False)
            visualization.show_ae_overlay_var.set(False)
            visualization.on_optional_overlay_toggle()
            self.assertEqual(1, len(figure.axes))
            self.assertEqual(2, visualization.canvas_data.draw_count)
        finally:
            plt.close(figure)

        visualization.show_measured_curve_var.set(False)
        visualization.show_interval_state_var.set(False)
        self.assertTrue(visualization.generate_plots(silent=True))
        toggled_figure = visualization.figures[0]
        try:
            labels = [
                line.get_label()
                for line in toggled_figure.axes[0].lines
            ]
            self.assertIn("实际负载", labels)
            self.assertNotIn("预测负载", labels)
            state_fills = [
                artist
                for artist in toggled_figure.axes[0].collections
                if isinstance(artist, PolyCollection)
            ]
            self.assertEqual([], state_fills)
        finally:
            plt.close(toggled_figure)

        visualization.show_measured_curve_var.set(True)
        visualization.show_reconstructed_curve_var.set(False)
        self.assertTrue(visualization.generate_plots(silent=True))
        prediction_hidden_figure = visualization.figures[0]
        try:
            labels = [
                line.get_label()
                for line in prediction_hidden_figure.axes[0].lines
            ]
            self.assertIn("实际负载", labels)
            self.assertNotIn("预测负载", labels)
        finally:
            plt.close(prediction_hidden_figure)

        visualization.manual_measurement_data = {"actual_load": actual_load}
        visualization.display_prediction_to_generate = None
        self.assertTrue(visualization.generate_plots(silent=True))
        actual_only_figure = visualization.figures[0]
        try:
            labels = [line.get_label() for line in actual_only_figure.axes[0].lines]
            self.assertIn("实际负载", labels)
            self.assertNotIn("程序 MRR", labels)
        finally:
            plt.close(actual_only_figure)

        visualization.sample_data_mode = "sampledata"
        visualization.show_feed_overlay_var.set(True)
        visualization.show_speed_overlay_var.set(True)
        visualization.show_ap_overlay_var.set(True)
        visualization.show_ae_overlay_var.set(True)
        sampledata_labels = {
            item["label"]
            for item in visualization.get_optional_measurement_overlays()
        }
        self.assertEqual(
            {
                "F(程序进给)",
                "S(主轴转速)",
                "ap(切深)",
                "ae(切宽)",
            },
            sampledata_labels,
        )

    def test_10_optional_axes_stay_compact_and_do_not_overlap_after_resize(self):
        process_result = _run(
            [0.0] * 3
            + [1.0, 2.0, 3.0, 2.0]
            + [4.0] * 12
            + [3.0, 2.0, 1.0]
            + [0.0] * 3
        )
        visualization = _OptionalOverlayVisualizationHarness(process_result)
        visualization.data = [
            {"S": 4000.0 + index * 10.0}
            for index in range(len(process_result.point_labels))
        ]
        visualization.show_ap_overlay_var.set(True)
        visualization.show_ae_overlay_var.set(True)

        self.assertTrue(visualization.generate_plots(silent=True))
        figure = visualization.figures[0]
        visualization.canvas_data = _SizedCanvasStub(700, 520)
        try:
            BootstrapUiMixin.adjust_figure_sizes(visualization)
            figure.canvas.draw()

            self.assertGreater(figure.subplotpars.right, 0.75)
            ap_axis, ae_axis = figure.axes[1:]
            renderer = figure.canvas.get_renderer()
            ap_label_bounds = ap_axis.yaxis.label.get_window_extent(renderer)
            ae_tick_bounds = [
                tick_label.get_window_extent(renderer)
                for tick_label in ae_axis.get_yticklabels()
                if tick_label.get_visible() and tick_label.get_text()
            ]
            self.assertTrue(ae_tick_bounds)
            self.assertFalse(
                any(
                    ap_label_bounds.overlaps(tick_bounds)
                    for tick_bounds in ae_tick_bounds
                )
            )
            ae_label_bounds = ae_axis.yaxis.label.get_window_extent(renderer)
            self.assertLessEqual(ae_label_bounds.x1, figure.bbox.x1)
        finally:
            plt.close(figure)

        visualization.show_feed_overlay_var.set(True)
        visualization.show_speed_overlay_var.set(True)
        self.assertTrue(visualization.generate_plots(silent=True))
        wide_figure = visualization.figures[0]
        visualization.canvas_data = _SizedCanvasStub(2048, 520)
        try:
            BootstrapUiMixin.adjust_figure_sizes(visualization)
            wide_figure.canvas.draw()

            self.assertEqual(5, len(wide_figure.axes))
            self.assertGreater(wide_figure.subplotpars.right, 0.84)
            self.assertGreater(wide_figure.axes[0].get_position().width, 0.75)
            renderer = wide_figure.canvas.get_renderer()
            aux_axes = wide_figure.axes[1:]
            for index, axis in enumerate(aux_axes):
                self.assertEqual(
                    (
                        "outward",
                        visualization._OPTIONAL_OVERLAY_SPINE_STEP_POINTS * index,
                    ),
                    axis.spines["right"].get_position(),
                )
            for inner_axis, outer_axis in zip(aux_axes, aux_axes[1:]):
                inner_label_bounds = inner_axis.yaxis.label.get_window_extent(
                    renderer
                )
                outer_tick_bounds = [
                    tick_label.get_window_extent(renderer)
                    for tick_label in outer_axis.get_yticklabels()
                    if tick_label.get_visible() and tick_label.get_text()
                ]
                self.assertFalse(
                    any(
                        inner_label_bounds.overlaps(tick_bounds)
                        for tick_bounds in outer_tick_bounds
                    )
                )
            outer_label_bounds = aux_axes[-1].yaxis.label.get_window_extent(
                renderer
            )
            self.assertLessEqual(outer_label_bounds.x1, wide_figure.bbox.x1)
        finally:
            plt.close(wide_figure)


if __name__ == "__main__":
    unittest.main()
