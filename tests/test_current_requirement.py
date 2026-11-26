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

from interval_analysis.academic_workbench import AcademicWorkbenchMixin
from interval_analysis.analysis_export import AnalysisExportMixin
from interval_analysis.app import MillingAnalysisTool
from interval_analysis.pit_model import PitModelMixin
from interval_analysis.pit_viewer import PitViewerMixin
from interval_analysis.plot_support import PlotSupportMixin
from interval_analysis.prediction_runtime import PredictionRuntimeMixin
from interval_analysis.processing_core import ProcessingCoreMixin
from interval_analysis.sample_manager import SampleManagerMixin
from interval_analysis.segmentation import SegmentationConfig, SegmentationPipeline
from interval_analysis.segmentation.features import (
    build_atomic_segments,
    compute_point_features,
    standardize_input,
)
from interval_analysis.segmentation.scorers import RuleSegmentScorer
from interval_analysis.ui_bootstrap import BootstrapUiMixin


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


class _QuantizedProjectionHarness(AcademicWorkbenchMixin, PlotSupportMixin):
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


class _PanedHeightStub:
    def __init__(self, height):
        self.height = int(height)
        self.sash_height = None

    def update_idletasks(self):
        return None

    def winfo_height(self):
        return self.height

    def sashpos(self, _index, value):
        self.sash_height = int(value)


class _RequestedHeightStub:
    def __init__(self, requested_height):
        self.requested_height = int(requested_height)

    def winfo_reqheight(self):
        return self.requested_height


class _GridVisibilityStub:
    def __init__(self):
        self.remove_count = 0
        self.restore_count = 0

    def grid_remove(self):
        self.remove_count += 1

    def grid(self):
        self.restore_count += 1


class _ImmediateIdleRoot:
    def after_idle(self, callback):
        callback()
        return None


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
    def test_experiment_channel_parser_accepts_program_line_axis_aliases(self):
        parser = ProcessingCoreMixin()
        parser.detect_file_encoding = lambda _path: "gbk"
        headers = [
            "<ChannelInfo>,<False>,<5>,<实际速度>,<5>,<SP轴>,<0>,<1>,<1>,<10>",
            "<ChannelInfo>,<True>,<107>,<G寄存器>,<0>,<X轴>,<432>",
            "<ChannelInfo>,<False>,<1>,<指令位置>,<0>,<X轴>,<0>",
            "<ChannelInfo>,<False>,<5>,<实际速度>,<0>,<X轴>,<0>",
            "<ChannelInfo>,<False>,<5>,<实际速度>,<1>,<Y轴>,<0>",
            "<ChannelInfo>,<False>,<5>,<实际速度>,<2>,<Z轴>,<0>",
            "<ChannelInfo>,<False>,<12>,<程序行号>,<0>,<{axis}>,<0>",
        ]
        rows = [
            [8000, -25, -210.5, 3, 4, 0, 0],
            [288000, 6, -211.5, 0, 5, 12, 651],
        ]
        with TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "measurement.csv"
            for axis in ("AX", "X轴"):
                for order in (list(range(7)), [1, 6, 2, 3, 4, 5, 0]):
                    with self.subTest(axis=axis, order=order):
                        payload = ["<Scope>"]
                        payload.extend(headers[index].format(axis=axis) for index in order)
                        payload.extend(
                            "<ChannelData>," + ",".join(str(row[index]) for index in order) + ","
                            for row in rows
                        )
                        data_path.write_text("\n".join(payload), encoding="gbk")
                        result = parser.parse_channel_data_file(data_path)

                        self.assertEqual(2, result["sample_count"])
                        self.assertEqual(order.index(6), result["channel_columns"]["program_line"])
                        np.testing.assert_array_equal(result["program_line"], [0, 651])
                        np.testing.assert_allclose(result["actual_load"], [25, 6])
                        np.testing.assert_allclose(result["actual_feed_speed"], [5, 13])
                        # 通道头倍率不同也保留采样转速原值，由采集端修正。
                        np.testing.assert_array_equal(result["actual_spindle_speed"], [8000, 288000])

    def test_experiment_program_line_alias_does_not_match_axis_position(self):
        parser = ProcessingCoreMixin()
        for signal_name in ("指令位置", "实际位置"):
            tokens = ["ChannelInfo", "False", "1", signal_name, "0", "X轴", "0"]
            with self.subTest(signal_name=signal_name):
                self.assertIsNone(parser._match_channel_info_key(tokens))
        self.assertIsNone(parser._match_channel_info_key(
            ["ChannelInfo", "False", "12", "程序行号", "0", "Y轴", "0"]
        ))

    def test_experiment_channel_parser_batch_reads_only_required_columns(self):
        payload = "\n".join((
            "<Scope>",
            "<ChannelInfo>,<True>,<5>,<实际速度>,<5>,<SP轴>",
            "<ChannelInfo>,<True>,<5>,<G寄存器>,<5>,<X轴>,<432>",
            "<ChannelInfo>,<True>,<5>,<实际速度>,<5>,<X轴>",
            "<ChannelInfo>,<True>,<5>,<实际速度>,<5>,<Y轴>",
            "<ChannelInfo>,<True>,<5>,<实际速度>,<5>,<Z轴>",
            "<ChannelInfo>,<True>,<5>,<程序行号>,<5>,<AX>",
            "<ChannelData>,1,2,",
            "<ChannelData>,1000,-25,3,4,,10,",
            "<ChannelData>,<2000>,<6>,<0>,<5>,<12>,<11>,",
            "<ChannelData>,3000,7,1,2,2,bad,",
            "<OtherRecord>,4000,8,6,8,0,12,",
        ))
        parser = ProcessingCoreMixin()
        parser.detect_file_encoding = lambda _path: "utf-8"

        with TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "measurement.csv"
            data_path.write_text(payload, encoding="utf-8")
            with patch(
                "interval_analysis.processing_core.pd.read_csv",
                wraps=pd.read_csv,
            ) as batch_reader:
                result = parser.parse_channel_data_file(data_path)

        self.assertEqual(1, batch_reader.call_count)
        read_options = batch_reader.call_args.kwargs
        self.assertEqual("c", read_options["engine"])
        self.assertEqual([0, 1, 2, 3, 4, 5, 6], read_options["usecols"])
        self.assertEqual(2, result["sample_count"])
        self.assertEqual(1, result["negative_load_corrected_count"])
        np.testing.assert_allclose(result["actual_load"], [25.0, 6.0])
        np.testing.assert_allclose(result["actual_spindle_speed"], [1000.0, 2000.0])
        np.testing.assert_allclose(result["actual_feed_speed"], [5.0, 13.0])
        np.testing.assert_array_equal(result["program_line"], [10, 11])

    def test_sample_sequence_layout_fast_path_matches_legacy_coordinates(self):
        harness = PlotSupportMixin()
        line_numbers = np.asarray(
            [10, 10, 11, 11, 9, 9, 9, 10, 10, 10],
            dtype=int,
        )
        group_keys = np.asarray(
            ["A", "A", "A", "A", "A", "A", "B", "B", "B", "B"],
            dtype=object,
        )

        blocks, x_positions, point_indices = (
            harness.compute_sample_sequence_layout(line_numbers, group_keys)
        )
        expected_blocks = harness.compute_sequence_blocks(
            line_numbers,
            group_keys,
        )
        expected_x = harness.compute_line_x_positions(
            line_numbers,
            blocks=expected_blocks,
        )
        expected_points = harness.compute_line_point_indices(
            line_numbers,
            blocks=expected_blocks,
        )

        self.assertEqual(expected_blocks, blocks)
        np.testing.assert_array_equal(point_indices, expected_points)
        np.testing.assert_allclose(x_positions, expected_x, rtol=0.0, atol=0.0)
        empty = harness.compute_sample_sequence_layout(
            np.asarray([], dtype=int),
            np.asarray([], dtype=object),
        )
        self.assertEqual([], empty[0])
        self.assertEqual(0, empty[1].size)
        self.assertEqual(0, empty[2].size)

    def test_process_header_detection_skips_normal_rows_after_layout_is_known(self):
        parser = ProcessingCoreMixin()
        header = [
            "序号",
            "N",
            "ap(mm)",
            "ae(mm)",
            "F(mm/min)",
            "MRR(mm3/s)",
            "s(mm)",
            "S(r/min)",
            "G代码",
        ]
        layout = parser._detect_process_header_layout(header)
        self.assertIsNotNone(layout)
        data_row = ["1", "N10", "1", "2", "600", "20", "0.5", "8000", "G1 X1"]

        with patch.object(
            parser,
            "_detect_process_header_layout",
            wraps=parser._detect_process_header_layout,
        ) as detector:
            for _index in range(200):
                parsed, updated_layout = parser._parse_process_tokens(
                    data_row,
                    layout_hint=layout,
                )
                self.assertIsNotNone(parsed)
                self.assertEqual(layout, updated_layout)
            repeated_header, repeated_layout = parser._parse_process_tokens(
                header,
                layout_hint=layout,
            )

        self.assertIsNone(repeated_header)
        self.assertEqual(layout, repeated_layout)
        self.assertEqual(1, detector.call_count)

    def test_main_page_prediction_and_pit_viewer_are_separate_components(self):
        mro = list(MillingAnalysisTool.__mro__)

        self.assertLess(mro.index(PredictionRuntimeMixin), mro.index(PitModelMixin))
        self.assertLess(mro.index(PitViewerMixin), mro.index(PitModelMixin))
        self.assertIs(
            PredictionRuntimeMixin._refresh_manual_measurement_prediction,
            MillingAnalysisTool._refresh_manual_measurement_prediction,
        )
        self.assertNotIn("_refresh_manual_measurement_prediction", PitViewerMixin.__dict__)
        self.assertNotIn("has_prediction_model_ready", PitModelMixin.__dict__)

    def test_default_preview_is_majority_and_consistent_across_both_apps(self):
        total_height = 480
        observed = {}
        for label, research_enabled, requested_height in (
            ("release", False, 129),
            ("main_collapsed", True, 140),
        ):
            with self.subTest(label=label):
                harness = BootstrapUiMixin.__new__(BootstrapUiMixin)
                harness.enable_research_features = research_enabled
                harness.model_detail_collapsed = _ValueVar(True)
                harness._data_paned = _PanedHeightStub(total_height)
                harness._data_controls = _RequestedHeightStub(requested_height)
                harness._adapt_data_processing_layout = lambda: None

                harness._sync_data_controls_preview_ratio()

                controls_height = harness._data_paned.sash_height
                self.assertIsNotNone(controls_height)
                self.assertGreaterEqual(controls_height, requested_height)
                preview_ratio = (
                    total_height - controls_height
                ) / float(total_height)
                self.assertGreaterEqual(preview_ratio, 0.68)
                observed[label] = preview_ratio

        self.assertLessEqual(
            abs(observed["release"] - observed["main_collapsed"]),
            0.04,
        )

    def test_research_detail_toggle_hides_and_restores_whole_model_frame(self):
        harness = BootstrapUiMixin.__new__(BootstrapUiMixin)
        harness.model_detail_collapsed = _ValueVar(False)
        harness.model_detail_toggle_text = _ValueVar("")
        harness._model_detail_frame = _GridVisibilityStub()
        harness.root = _ImmediateIdleRoot()
        sync_count = []
        harness._sync_data_controls_preview_ratio = lambda: sync_count.append(1)
        harness.model_detail_toggle_btn = object()

        harness._set_model_detail_collapsed(True)

        self.assertTrue(harness.model_detail_collapsed.get())
        self.assertEqual("展开详情", harness.model_detail_toggle_text.get())
        self.assertEqual(1, harness._model_detail_frame.remove_count)
        self.assertEqual(0, harness._model_detail_frame.restore_count)
        self.assertIsNotNone(harness.model_detail_toggle_btn)

        harness._toggle_model_detail_section()

        self.assertFalse(harness.model_detail_collapsed.get())
        self.assertEqual("收起详情", harness.model_detail_toggle_text.get())
        self.assertEqual(1, harness._model_detail_frame.remove_count)
        self.assertEqual(1, harness._model_detail_frame.restore_count)
        self.assertEqual(2, len(sync_count))

    def test_main_ui_removes_smif_and_data_analysis_pages(self):
        ui_source = (SRC_ROOT / "interval_analysis" / "ui_bootstrap.py").read_text(encoding="utf-8")
        workbench_source = (SRC_ROOT / "interval_analysis" / "academic_workbench.py").read_text(encoding="utf-8")

        self.assertIn('self.notebook.add(self.data_processing_tab, text="主页面")', ui_source)
        self.assertIn('self.notebook.add(self.pit_tab, text="PIT")', ui_source)
        self.assertNotIn("create_smif_pit_tab", ui_source)
        self.assertNotIn("PIT / SMIF", ui_source)
        self.assertNotIn("create_data_analysis_tab", workbench_source)
        self.assertNotIn("run_data_analysis", workbench_source)

    def test_pit_viewer_is_process_only_fixed_four_field_dashboard(self):
        harness = PitViewerMixin.__new__(PitViewerMixin)
        harness.data = [
            {
                "line_no_raw": 9,
                "path_start": 0.0,
                "path_end": 1.0,
                "ap": 1.5,
                "ae": 2.5,
                "feed_effective": 600.0,
                "MRR": 37.5,
            }
        ]
        harness._process_model_state_version = 1
        harness._pit_dataframe_cache = None
        harness._pit_dataframe_cache_signature = None
        harness._pit_render_signature = None
        harness._pit_refresh_pending = True

        first = harness.build_current_pit_dataframe()
        cached_frame_id = id(harness._pit_dataframe_cache)
        second = harness.build_current_pit_dataframe()

        self.assertEqual(
            ("ap", "ae", "feed_effective", "MRR"),
            tuple(item[0] for item in PitViewerMixin.PIT_VIEW_FIELDS),
        )
        self.assertEqual(cached_frame_id, id(harness._pit_dataframe_cache))
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(10.0, first.loc[0, "command_position"])
        self.assertNotIn("actual_load", first.columns)
        self.assertNotIn("predicted_load", first.columns)

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
        self.assertEqual(512, SegmentationConfig().max_segment_atoms)
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
            for batch_size in (32, 48, 64, 512)
        }
        for result in results.values():
            self.assert_valid_result(result)
            self.assertEqual(
                "candidate_scoring_batch_size",
                result.diagnostics["max_segment_atoms_role"],
            )
        reference_points, reference_intervals = _result_shape(results[32])
        reference_signature = _process_signature(results[32])
        for batch_size in (48, 64, 512):
            with self.subTest(max_segment_atoms=batch_size):
                points, intervals = _result_shape(results[batch_size])
                pd.testing.assert_frame_equal(reference_points, points)
                pd.testing.assert_frame_equal(reference_intervals, intervals)
                self.assertEqual(reference_signature, _process_signature(results[batch_size]))

    def test_interval_statistics_remain_stable_after_large_idle_baseline_jump(self):
        point_count = 21
        ap_values = np.concatenate((
            np.asarray([0.0]),
            1.0e12 + np.arange(point_count - 1, dtype=float) % 2.0,
        ))
        frame = pd.DataFrame(
            {
                "source_index": range(point_count),
                "line_id": range(100, 100 + point_count),
                "line_no_raw": range(100, 100 + point_count),
                "ap": ap_values,
                "ae": np.ones(point_count, dtype=float),
                "F_program": np.full(point_count, 60.0, dtype=float),
                "s": np.full(point_count, 0.1, dtype=float),
            }
        )

        result = SegmentationPipeline(_test_config()).run(frame)
        observed_small_variation = False
        for interval in result.intervals.to_dict(orient="records"):
            start_idx = int(interval["start_idx"])
            end_idx = int(interval["end_idx"])
            for metric_name in ("ap", "ae", "F_program", "MRR_program"):
                values = result.point_labels[metric_name].to_numpy(
                    dtype=float,
                    copy=False,
                )[start_idx:end_idx + 1]
                self.assertEqual(
                    float(np.mean(values)),
                    float(interval[f"{metric_name}_mean"]),
                )
                self.assertEqual(
                    float(np.std(values, ddof=0)),
                    float(interval[f"{metric_name}_std"]),
                )
            if start_idx > 0 and end_idx > start_idx:
                observed_small_variation |= bool(
                    np.isclose(
                        float(interval["ap_std"]),
                        0.5,
                        rtol=0.0,
                        atol=0.0,
                    )
                )

        self.assertTrue(observed_small_variation)

    def test_single_point_steady_continuation_has_zero_slope_at_large_path_offset(self):
        point_count = 3000
        frame = _process_frame([10.0] * point_count)
        # 模拟 AfoMilling trace 中随序号递增的 s(mm) 被现有契约作为
        # 每点增量累计后形成的大横坐标，覆盖本次多文件导入断链场景。
        frame["s"] = np.linspace(0.0, 1500.0, point_count)
        config = _test_config()
        standardized, _ = standardize_input(frame, config)
        features = compute_point_features(standardized, config)
        atoms = build_atomic_segments(features, config)
        scorer = RuleSegmentScorer(config)
        scorer.prepare(features, atoms)

        target = 2760
        duration = float(atoms[target].end_s - atoms[target].start_s)
        slope = scorer._relative_path_slopes(
            np.asarray([target], dtype=int),
            target,
            np.asarray([duration], dtype=float),
            np.asarray([10.0], dtype=float),
        )
        continuation = scorer.score_steady_continuation_candidates(
            (target,),
            target,
        )

        np.testing.assert_array_equal(np.asarray([0.0]), slope)
        self.assertTrue(np.isfinite(float(continuation[0])))

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
        # 真实对象初始化后这两个缓存字段可能明确为 None；
        # 快路径必须回退到现有上下文计算，而不是在 asarray 处报错。
        quantized_harness.sample_data_point_indices = None
        quantized_harness.sample_data_x_positions = None
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

        quantized_harness.sample_data_loaded = True
        quantized_harness.sample_data_point_indices = np.arange(6, dtype=int)
        quantized_harness.sample_data_x_positions = (
            20.0 + np.arange(6, dtype=float) / 6.0
        )
        quantized_harness.sample_data_program_numbers = np.ones(6, dtype=int)
        quantized_harness.sample_data_values = np.asarray(
            [[10.0], [20.0], [30.0], [40.0], [50.0], [60.0]],
            dtype=float,
        )
        quantized_harness.sample_data_source = _ValueVar(0)
        optimizable_records = [
            {
                **dict(quantized_records[index]),
                "start_line": 20,
                "end_line": 20,
                "display_start_x": float(quantized_records[index]["process_start_x"]),
                "display_end_x": float(
                    quantized_records[index]["process_display_end_x"]
                ),
            }
            for index in (0, 2)
        ]
        quantized_harness._get_optimizable_interval_records = (
            lambda: [dict(record) for record in optimizable_records]
        )
        quantized_harness._get_authoritative_segmentation_sample_records = (
            lambda: [dict(record) for record in quantized_projection]
        )
        quantized_harness.get_program_number_by_name = lambda _name: 1
        quantized_harness.get_tool_ranges_by_id = lambda _name, _tool: [(20, 20)]

        self.assertEqual(
            [
                "20.0-20.1:15.000000",
                "20.3-20.5:50.000000",
            ],
            quantized_harness.collect_line_point_intervals_for_tool("P1", "T0"),
        )

        typed_intervals = quantized_harness.collect_line_point_intervals_for_tool(
            "P1", "T0", include_types=True
        )
        self.assertEqual(["idle", "steady"], [item["segment_type"] for item in typed_intervals])
        self.assertEqual(
            ["20.0-20.1:15.000000", "20.3-20.5:50.000000"],
            [item["text"] for item in typed_intervals],
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
            patch("interval_analysis.analysis_export.messagebox.showwarning") as warning,
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
            process_line_axis = getattr(
                process_axis,
                "_time_line_top_axis",
                None,
            )
            self.assertIsNotNone(process_line_axis)
            self.assertEqual("程序行号", process_line_axis.get_xlabel())
            displayed_line_numbers = {
                int(label.get_text())
                for label in process_line_axis.get_xticklabels()
                if label.get_text()
            }
            expected_line_numbers = set(
                process_result.point_labels["line_no_raw"].astype(int)
            )
            self.assertTrue(displayed_line_numbers)
            self.assertTrue(displayed_line_numbers.issubset(expected_line_numbers))
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
            patch("interval_analysis.analysis_export.OUTPUT_DIR", Path(temporary_dir)),
            patch("interval_analysis.analysis_export.messagebox.showinfo") as showinfo,
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
        visualization.apply_line_axis_on_time = (
            PlotSupportMixin.apply_line_axis_on_time.__get__(visualization)
        )

        self.assertTrue(visualization.generate_plots(silent=True))
        figure = visualization.figures[0]
        try:
            self.assertEqual("实际负载与区间划分", figure.axes[0].get_title())
            self.assertIn(id(figure), visualization._optional_overlay_contexts)
            self.assertEqual(1, len(figure.axes))
            visualization._current_preview_fig = figure
            visualization.canvas_data = _CanvasStub()
            original_canvas = visualization.canvas_data
            actual_line = next(
                line
                for line in figure.axes[0].lines
                if line.get_label() == "实际负载"
            )
            original_ylim = figure.axes[0].get_ylim()
            original_title = figure.axes[0].get_title()
            original_line_axis_mask = np.asarray(
                figure._measurement_curve_context["line_axis_mask"],
                dtype=bool,
            ).copy()
            original_top_axis = figure.axes[0]._time_line_top_axis
            zoomed_xlim = (5.0, float(point_count - 6))
            figure.axes[0].set_xlim(*zoomed_xlim)

            self.assertTrue(visualization.generate_plots(silent=True))
            self.assertIs(figure, visualization.figures[0])
            self.assertIs(original_canvas, visualization.canvas_data)
            refreshed_actual_line = next(
                line
                for line in figure.axes[0].lines
                if line.get_label() == "实际负载"
            )
            self.assertIs(actual_line, refreshed_actual_line)
            np.testing.assert_allclose(
                zoomed_xlim,
                figure.axes[0].get_xlim(),
                rtol=0.0,
                atol=0.0,
            )
            self.assertEqual(original_ylim, figure.axes[0].get_ylim())
            self.assertEqual(original_title, figure.axes[0].get_title())
            self.assertEqual("时间 (ms)", figure.axes[0].get_xlabel())
            self.assertEqual("功率 (W)", figure.axes[0].get_ylabel())
            np.testing.assert_array_equal(
                original_line_axis_mask,
                figure._measurement_curve_context["line_axis_mask"],
            )
            np.testing.assert_array_equal(
                original_line_axis_mask,
                figure.axes[0]._time_line_mask,
            )
            self.assertIs(original_top_axis, figure.axes[0]._time_line_top_axis)
            self.assertEqual(1, visualization.canvas_data.draw_count)
            self.assertLessEqual(
                len([
                    artist
                    for artist in figure.axes[0].collections
                    if isinstance(artist, PolyCollection)
                ]),
                6,
            )

            visualization.show_feed_overlay_var.set(True)
            visualization.show_speed_overlay_var.set(True)
            visualization.show_ap_overlay_var.set(True)
            visualization.show_ae_overlay_var.set(True)
            visualization.on_optional_overlay_toggle()
            self.assertEqual(5, len(figure.axes))
            self.assertEqual(2, visualization.canvas_data.draw_count)
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
            self.assertEqual(3, visualization.canvas_data.draw_count)
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

    def test_mapped_preview_reuses_objects_and_refreshes_mapping_context(self):
        process_result = _run(
            [0.0] * 3
            + [1.0, 2.0, 3.0, 2.0]
            + [4.0] * 12
            + [3.0, 2.0, 1.0]
            + [0.0] * 3
        )
        point_count = len(process_result.point_labels)
        sample_x = np.arange(point_count, dtype=float)
        actual_load = np.linspace(50.0, 80.0, point_count)
        projected_mask = np.zeros(point_count, dtype=bool)
        projected_mask[2:point_count - 2] = True

        visualization = _OptionalOverlayVisualizationHarness(process_result)
        visualization.sample_data_loaded = True
        visualization.sample_data_values = actual_load.reshape(-1, 1)
        visualization.sample_data_line_numbers = np.arange(
            100,
            100 + point_count,
            dtype=int,
        )
        visualization.sample_data_time_indices = sample_x
        visualization.sample_data_mode = "experiment_measurement"
        visualization.manual_measurement_data = {"actual_load": actual_load}
        visualization._sample_mapping_status = "valid"
        visualization._current_mapping_signature = "mapping-trimmed"
        visualization._segmentation_sample_projection_records = [{
            "segment_type": "steady",
            "sample_start_idx": 2,
            "sample_end_idx": point_count - 3,
        }]
        visualization.apply_line_axis_on_time = (
            PlotSupportMixin.apply_line_axis_on_time.__get__(visualization)
        )

        figure, axis = plt.subplots(figsize=(9.0, 3.0), dpi=100)
        actual_line, = axis.plot(
            sample_x,
            actual_load,
            label="实际负载",
        )
        axis.set_xlim(-5.0, float(point_count + 4))
        axis.set_xlabel("时间 (ms)")
        axis.set_ylabel("曲线值")
        preview_mask = np.ones(point_count, dtype=bool)
        top_axis = visualization.apply_line_axis_on_time(axis, preview_mask)
        figure._measurement_curve_context = {
            "axis": axis,
            "sample_count": point_count,
            "source_index": 0,
            "data_values_id": id(visualization.sample_data_values),
            "x_values": sample_x.copy(),
            "line_axis_mask": preview_mask.copy(),
            "actual_lines": [actual_line],
            "segmentation_artists": [],
            "prediction_lines": [],
            "message_artists": [],
        }
        visualization.figures = [figure]
        visualization.figure_names = ["实测负载预览"]
        visualization._current_preview_fig = figure
        visualization.canvas_data = _CanvasStub()
        original_canvas = visualization.canvas_data
        original_x = np.asarray(actual_line.get_xdata(), dtype=float).copy()
        original_y = np.asarray(actual_line.get_ydata(), dtype=float).copy()

        try:
            self.assertTrue(visualization.generate_plots(silent=True))

            self.assertIs(figure, visualization.figures[0])
            self.assertIs(original_canvas, visualization.canvas_data)
            self.assertIs(actual_line, next(
                line
                for line in axis.lines
                if line.get_label() == "实际负载"
            ))
            self.assertIs(top_axis, axis._time_line_top_axis)
            np.testing.assert_array_equal(original_x, actual_line.get_xdata())
            np.testing.assert_array_equal(original_y, actual_line.get_ydata())
            np.testing.assert_allclose(
                (2.0, float(point_count - 3)),
                axis.get_xlim(),
                rtol=0.0,
                atol=0.0,
            )
            self.assertGreater(axis.get_xlim()[0], sample_x[0])
            self.assertLess(axis.get_xlim()[1], sample_x[-1])
            np.testing.assert_array_equal(
                projected_mask,
                figure._measurement_curve_context["line_axis_mask"],
            )
            np.testing.assert_array_equal(projected_mask, axis._time_line_mask)
            self.assertEqual(
                "mapping-trimmed",
                figure._measurement_curve_context["mapping_signature"],
            )
            self.assertEqual("时间 (ms)", axis.get_xlabel())
            self.assertEqual("功率 (W)", axis.get_ylabel())
            final_xlim = axis.get_xlim()
            self.assertTrue(all(
                final_xlim[0] <= float(tick) <= final_xlim[1]
                for tick in axis.get_xticks()
            ))
            self.assertEqual(1, visualization.canvas_data.draw_count)
        finally:
            plt.close(figure)

    def test_compact_line_axis_layout_fits_long_labels_without_drift(self):
        class _LayoutHarness(
            BootstrapUiMixin,
            SampleManagerMixin,
            PlotSupportMixin,
        ):
            pass

        line_numbers = np.repeat(
            np.asarray([123456, -123456, 1234567890], dtype=np.int64),
            25,
        )
        expected_labels = {str(value) for value in np.unique(line_numbers)}

        for dpi in (100, 150):
            with self.subTest(dpi=dpi):
                harness = _LayoutHarness.__new__(_LayoutHarness)
                harness.sample_data_line_numbers = line_numbers.copy()
                harness.sample_data_time_indices = np.arange(
                    line_numbers.size,
                    dtype=float,
                )
                figure, axis = plt.subplots(figsize=(9.0, 2.1), dpi=dpi)
                axis.plot(
                    harness.sample_data_time_indices,
                    np.linspace(100.0, 500.0, line_numbers.size),
                )
                axis.set_xlim(
                    0.0,
                    float(harness.sample_data_time_indices[-1]),
                )
                axis.set_title("实际负载与区间划分")
                axis.set_xlabel("时间 (ms)")
                original_mask = np.ones(line_numbers.size, dtype=bool)
                top_axis = harness.apply_line_axis_on_time(axis, original_mask)
                harness._apply_optional_overlay_layout(
                    figure,
                    0,
                    {
                        "left": 0.06,
                        "right": 0.985,
                        "top": 0.94,
                        "bottom": 0.09,
                    },
                )

                try:
                    observed_margins = []
                    for _iteration in range(2):
                        self.assertTrue(
                            harness._layout_data_figure_for_viewport(
                                figure,
                                900,
                                210,
                            )
                        )
                        figure.canvas.draw()
                        observed_margins.append(np.asarray([
                            figure.subplotpars.left,
                            figure.subplotpars.right,
                            figure.subplotpars.bottom,
                            figure.subplotpars.top,
                        ]))

                    np.testing.assert_allclose(
                        observed_margins[0],
                        observed_margins[1],
                        rtol=0.0,
                        atol=1.0e-9,
                    )
                    self.assertIs(top_axis, axis._time_line_top_axis)
                    np.testing.assert_array_equal(
                        original_mask,
                        axis._time_line_mask,
                    )
                    self.assertGreaterEqual(
                        figure.subplotpars.top - figure.subplotpars.bottom,
                        0.25 - 1.0e-9,
                    )

                    visible_tick_labels = [
                        label
                        for label in top_axis.get_xticklabels()
                        if label.get_visible() and label.get_text()
                    ]
                    self.assertTrue(expected_labels.issubset({
                        label.get_text()
                        for label in visible_tick_labels
                    }))
                    renderer = figure.canvas.get_renderer()
                    figure_bounds = figure.bbox
                    visible_artists = [
                        axis.xaxis.label,
                        top_axis.xaxis.label,
                        *visible_tick_labels,
                    ]
                    for artist in visible_artists:
                        bounds = artist.get_window_extent(renderer)
                        self.assertGreaterEqual(bounds.x0, figure_bounds.x0 - 2.0)
                        self.assertLessEqual(bounds.x1, figure_bounds.x1 + 2.0)
                        self.assertGreaterEqual(bounds.y0, figure_bounds.y0 - 2.0)
                        self.assertLessEqual(bounds.y1, figure_bounds.y1 + 2.0)
                finally:
                    plt.close(figure)

    def test_10_optional_axes_stay_compact_and_do_not_overlap_after_resize(self):
        process_result = _run(
            [0.0] * 3
            + [1.0, 2.0, 3.0, 2.0]
            + [4.0] * 12
            + [3.0, 2.0, 1.0]
            + [0.0] * 3
        )
        visualization = _OptionalOverlayVisualizationHarness(process_result)
        # 本测试需要验证 Bootstrap 的视口布局，但不改变其他
        # 可视化测试的精简 harness MRO，避免把真实 Tk 生命周期引入无头测试。
        visualization._apply_data_figure_typography = (
            BootstrapUiMixin._apply_data_figure_typography.__get__(visualization)
        )
        visualization._layout_data_figure_for_viewport = (
            BootstrapUiMixin._layout_data_figure_for_viewport.__get__(visualization)
        )
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

        compact_figure, compact_axis = plt.subplots(figsize=(7.0, 2.6), dpi=100)
        compact_axis.set_title("实际负载与区间划分")
        compact_axis.set_xlabel("时间 (ms)")
        compact_axis._time_line_top_axis = object()
        try:
            BootstrapUiMixin._apply_data_figure_typography(
                visualization,
                compact_figure,
                260,
            )
            visualization._apply_optional_overlay_layout(
                compact_figure,
                0,
                {
                    "left": 0.06,
                    "right": 0.985,
                    "top": 0.94,
                    "bottom": 0.09,
                },
            )
            compact_figure.canvas.draw()
            renderer = compact_figure.canvas.get_renderer()
            xlabel_bounds = compact_axis.xaxis.label.get_window_extent(renderer)

            self.assertGreaterEqual(xlabel_bounds.y0, compact_figure.bbox.y0)
            self.assertGreater(
                compact_figure.subplotpars.bottom,
                0.09,
            )
            self.assertLess(
                compact_figure.subplotpars.top,
                0.94,
            )
            self.assertGreater(
                compact_figure.subplotpars.top - compact_figure.subplotpars.bottom,
                0.5,
            )
            self.assertFalse(compact_axis.title.get_visible())
            self.assertLessEqual(compact_axis.xaxis.label.get_fontsize(), 8)
        finally:
            plt.close(compact_figure)

        class _EmbeddedAxisHarness(
            BootstrapUiMixin,
            SampleManagerMixin,
            PlotSupportMixin,
        ):
            pass

        axis_harness = _EmbeddedAxisHarness.__new__(_EmbeddedAxisHarness)
        axis_harness.sample_data_line_numbers = np.repeat(
            np.arange(17, 25, dtype=int),
            25,
        )
        axis_harness.sample_data_time_indices = np.arange(
            axis_harness.sample_data_line_numbers.size,
            dtype=float,
        )
        embedded_figure, embedded_axis = plt.subplots(figsize=(9.0, 2.1), dpi=100)
        embedded_axis.plot(
            axis_harness.sample_data_time_indices,
            np.linspace(100.0, 500.0, axis_harness.sample_data_time_indices.size),
        )
        embedded_axis.set_xlim(
            0.0,
            float(axis_harness.sample_data_time_indices[-1]),
        )
        embedded_axis.set_title("实际负载与区间划分")
        embedded_axis.set_xlabel("时间 (ms)")
        original_mask = np.ones(
            axis_harness.sample_data_line_numbers.size,
            dtype=bool,
        )
        top_axis = axis_harness.apply_line_axis_on_time(
            embedded_axis,
            original_mask,
        )
        top_axis_id = id(top_axis)
        axis_harness._apply_optional_overlay_layout(
            embedded_figure,
            0,
            {
                "left": 0.06,
                "right": 0.985,
                "top": 0.94,
                "bottom": 0.09,
            },
        )
        try:
            self.assertTrue(
                axis_harness._layout_data_figure_for_viewport(
                    embedded_figure,
                    900,
                    210,
                )
            )
            embedded_figure.canvas.draw()
            self.assertEqual(top_axis_id, id(embedded_axis._time_line_top_axis))
            np.testing.assert_array_equal(
                original_mask,
                embedded_axis._time_line_mask,
            )
            self.assertFalse(embedded_axis.title.get_visible())

            renderer = embedded_figure.canvas.get_renderer()
            figure_bounds = embedded_figure.bbox
            visible_artists = [
                embedded_axis.xaxis.label,
                embedded_axis._time_line_top_axis.xaxis.label,
                *[
                    label
                    for label in embedded_axis._time_line_top_axis.get_xticklabels()
                    if label.get_visible() and label.get_text()
                ],
            ]
            for artist in visible_artists:
                bounds = artist.get_window_extent(renderer)
                self.assertGreaterEqual(bounds.x0, figure_bounds.x0 - 2.0)
                self.assertLessEqual(bounds.x1, figure_bounds.x1 + 2.0)
                self.assertGreaterEqual(bounds.y0, figure_bounds.y0 - 2.0)
                self.assertLessEqual(bounds.y1, figure_bounds.y1 + 2.0)
        finally:
            plt.close(embedded_figure)

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
