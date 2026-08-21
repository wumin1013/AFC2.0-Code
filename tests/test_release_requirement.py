from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from project.input_idle import InputIdleMixin
from project.analysis_export import AnalysisExportMixin
from project.config_state import ConfigStateMixin
from project.interval_runtime import IntervalRuntimeMixin
from project.plot_support import PlotSupportMixin
from project.prediction_support import (
    append_inverse_prediction_channels,
    apply_steady_representative_prediction,
    fit_nonnegative_kc_ke,
    summarize_interval_kc_mode_statistics,
)
from project.release_prediction import (
    IIPINC_FEED_SCALE,
    IIPINC_SINGLE_FEED_SCALE,
    IipincFormatError,
    ReleasePredictionMixin,
    map_iipinc_feed_to_samples,
    parse_iipinc_rows,
)
from project.sample_manager import SampleManagerMixin


class _ReleaseStartupHarness(InputIdleMixin):
    release_mode = True

    def __init__(self, sample_success=False, experiment_path=None, experiment_success=True):
        self.sample_data_loaded = False
        self.calls = []
        self.sample_success = bool(sample_success)
        self.experiment_path = experiment_path
        self.experiment_success = bool(experiment_success)
        self.experiment_resolve_calls = []
        self.experiment_load_calls = []

    def load_sample_bundle_from_dir(self, base_dir, **kwargs):
        self.calls.append((str(base_dir), dict(kwargs)))
        return self.sample_success

    def resolve_experiment_measurement_file(self, base_dir):
        self.experiment_resolve_calls.append(str(base_dir))
        if self.experiment_path is None:
            self._experiment_measurement_resolution_error = "未找到可识别的实验实测 CSV"
        return self.experiment_path

    def load_experiment_measurement_file(self, file_path, silent=False):
        self.experiment_load_calls.append((str(file_path), bool(silent)))
        return self.experiment_success

    def get_input_files(self):
        return []


class _ValueVar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _PowerDisplayHarness(ConfigStateMixin):
    def __init__(self):
        self.sample_avg_var = _ValueVar("-")
        self.sample_ideal_var = _ValueVar("-")
        self.adjustment_ratio = _ValueVar(2.5)

    @staticmethod
    def get_current_program_key():
        return "P1"

    @staticmethod
    def get_selected_tool_id():
        return "T0"

    @staticmethod
    def compute_tool_measured_mean(_program_name, _tool_id):
        return 200.0, 3, [(10, 12)]


class _RgExportHarness(AnalysisExportMixin):
    def __init__(self):
        self._current_interval_source = "segmentation"
        self._sample_mapping_status = "valid"
        self._current_mapping_signature = "mapping"
        self._segmentation_sample_projection_records = [{"interval_id": "SEG0001"}]
        self.sample_data_source = _ValueVar(1)
        self.status_var_data = _ValueVar("")

    def compute_tool_measured_mean(self, _program_name, _tool_id):
        return 10.0, 2, []

    def collect_line_point_intervals_for_tool(self, _program_name, _tool_id):
        return ["10.0-11.0:10.000000"]

    def _save_process_info_csv(self, _output_dir):
        return None

    def _save_process_data_paths(self, _output_dir, _saved_programs=None):
        return None


class _RootStub:
    def update_idletasks(self):
        return None


class _LoadFailureHarness(SampleManagerMixin):
    release_mode = True

    def __init__(self):
        self.root = _RootStub()
        self.sample_data_loaded = True
        self.sample_data_source = _ValueVar(0)
        self.sample_auto_status_var = _ValueVar("")
        self.status_var_data = _ValueVar("")
        self.reset_count = 0

    def refresh_sample_source_labels(self):
        return None

    def reset_sample_data_state(self):
        self.reset_count += 1
        self.sample_data_loaded = False


class _ReleasePredictionHarness(ReleasePredictionMixin):
    release_mode = True

    def __init__(self, iip_path):
        self._release_iipinc_path = Path(iip_path)
        self.sample_data_mode = "sampledata"
        self.sample_data_loaded = True
        self.sample_csv_path = None
        self.sample_txt_path = None
        self.sample_data_source = _ValueVar(1)
        self.inverse_prediction_status_var = _ValueVar("")
        self.p_idle_var = _ValueVar(0.0)
        self.kc_coeff = _ValueVar("")
        self.ke_coeff = _ValueVar("")
        self.ap = np.asarray([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])
        self.ae = np.asarray([1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 2.0, 5.0])
        self.process_feed = np.full(8, 60.0)
        self.sample_data_line_numbers = np.arange(8, dtype=int)
        process_mrr = self.ap * self.ae * self.process_feed / 60.0
        actual = 250.0 + 2.0 * process_mrr + 3.0 * self.ap
        self.sample_data_values = np.column_stack(
            [np.zeros(8), actual, np.zeros(8)]
        )
        self.data = [
            {
                "line_no_raw": int(index),
                "line_no_aligned": int(index),
                "process_point_index": 0,
                "ap": float(self.ap[index]),
                "ae": float(self.ae[index]),
                "feed_effective": 60.0,
            }
            for index in range(8)
        ]

    def _build_aligned_process_geometry_frame(self, raw_line_numbers):
        point_count = len(raw_line_numbers)
        return pd.DataFrame(
            {
                "line_no_aligned": np.asarray(raw_line_numbers, dtype=int),
                "ap": self.ap[:point_count],
                "ae": self.ae[:point_count],
                "feed_plan": self.process_feed[:point_count],
                "process_anchor_mask": np.ones(point_count, dtype=bool),
            }
        )


class _ReleaseProcessExportHarness(ReleasePredictionMixin, AnalysisExportMixin):
    release_mode = True

    def __init__(self, iip_path):
        self._release_iipinc_path = Path(iip_path)
        self.data = [
            {
                "line_no_raw": 0,
                "S": 5000.0,
                "ap": 1.0,
                "ae": 2.0,
                "feed_effective": 60.0,
                "s": 1.0,
                "gcode_content": "G1 X1 F60",
            },
            {
                "line_no_raw": 0,
                "S": 5000.0,
                "ap": 1.0,
                "ae": 2.0,
                "feed_effective": 60.0,
                "s": 1.0,
                "gcode_content": "G1 X2 F60",
            },
            {
                "line_no_raw": 1,
                "S": 5000.0,
                "ap": 1.0,
                "ae": 2.0,
                "feed_effective": 60.0,
                "s": 1.0,
                "gcode_content": "G1 X3 F60",
            },
        ]
        self._latest_segmentation_result = SimpleNamespace(
            point_labels=pd.DataFrame(
                {"source_index": [0, 1, 2], "state_code": [2, 2, 3]}
            )
        )


class _ReleasePlotHarness(AnalysisExportMixin, PlotSupportMixin):
    release_mode = True

    def __init__(self):
        self._latest_segmentation_result = object()
        self.sample_data_values = np.column_stack(
            [np.zeros(5), np.asarray([250.0, 260.0, np.nan, 270.0, 250.0])]
        )
        self.sample_data_source = _ValueVar(1)
        self.sample_data_mode = "sampledata"
        self.show_prediction_load_var = _ValueVar(False)
        self.preview_plot_max_points = 60000
        self.figures = []
        self.figure_names = []
        self.predicted = np.asarray([250.0, 255.0, np.nan, 275.0, 250.0])
        self.prediction_build_count = 0

    def has_prediction_model_ready(self):
        return True

    def _build_sampledata_prediction_payload(self):
        self.prediction_build_count += 1
        return {"predicted_load": self.predicted.copy()}

    @staticmethod
    def get_sample_time_indices_array():
        return np.arange(5, dtype=float)

    @staticmethod
    def show_current_figure(index=0):
        return None

    @staticmethod
    def apply_line_axis_on_time(ax, sample_mask, max_ticks=60):
        return None


class ReleaseRequirementTests(unittest.TestCase):
    def test_frozen_release_output_root_is_executable_directory(self):
        with TemporaryDirectory() as temp_dir:
            release_root = Path(temp_dir).resolve()
            internal_root = release_root / "_internal"
            internal_root.mkdir()
            code = """
import sys
from pathlib import Path
sys.frozen = True
sys.executable = str(Path(r'%s') / 'AFC2.0.2alpha.exe')
sys._MEIPASS = str(Path(r'%s'))
sys.path.insert(0, str(Path(r'%s')))
import project.shared as shared
expected = Path(r'%s').resolve()
if shared.OUTPUT_DIR != expected:
    raise SystemExit(f'OUTPUT_DIR={shared.OUTPUT_DIR}, expected={expected}')
if (expected / 'output').exists():
    raise SystemExit('frozen import created output directory')
""" % (release_root, internal_root, SRC_ROOT, release_root)
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(0, completed.returncode, completed.stderr or completed.stdout)

    def test_release_runtime_files_do_not_create_output_subdirectory(self):
        import project.analysis_export as export_module
        import project.release_main as release_main

        with TemporaryDirectory() as temp_dir:
            release_root = Path(temp_dir).resolve()
            harness = AnalysisExportMixin()
            harness.release_mode = True
            with patch.object(export_module, "OUTPUT_DIR", release_root), patch.object(
                export_module,
                "IS_FROZEN",
                True,
            ):
                self.assertEqual(
                    release_root,
                    harness._default_segmentation_output_dir(),
                )

            with patch.object(sys, "frozen", True, create=True), patch.object(
                sys,
                "executable",
                str(release_root / "AFC2.0.2alpha.exe"),
            ):
                log_path = Path(release_main._write_startup_error(RuntimeError("test")))
            self.assertEqual(release_root / "startup-error.log", log_path)
            self.assertTrue(log_path.is_file())
            self.assertFalse((release_root / "output").exists())

    def test_release_import_loads_only_lightweight_prediction(self):
        code = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(r'%s')))
import project.release_app
forbidden = [
    name for name in sys.modules
    if name in {
        'project.app',
        'project.pit_model',
        'sklearn',
    }
    or name.startswith('sklearn.')
]
if forbidden:
    raise SystemExit('forbidden imports: ' + ','.join(sorted(forbidden)))
required = {'project.prediction_support', 'project.release_prediction'}
missing = required.difference(sys.modules)
if missing:
    raise SystemExit('missing imports: ' + ','.join(sorted(missing)))
""" % str(SRC_ROOT)
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(PROJECT_ROOT.parent),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(0, completed.returncode, completed.stderr or completed.stdout)

    def test_release_mro_and_research_ui_are_isolated(self):
        from project.release_app import AFCReleaseApplication

        mro_names = {base.__name__ for base in AFCReleaseApplication.__mro__}
        self.assertNotIn("PitModelMixin", mro_names)
        self.assertIn("ReleasePredictionMixin", mro_names)
        self.assertTrue(AFCReleaseApplication.release_mode)
        self.assertFalse(AFCReleaseApplication.enable_research_features)
        self.assertFalse(AFCReleaseApplication.enable_profile_config)

    def test_release_main_smoke_mode_initializes_without_entering_mainloop(self):
        import project.release_main as release_main

        events = []

        class _SmokeRoot:
            def withdraw(self):
                events.append("withdraw")

            def title(self, _value):
                events.append("title")

            def update_idletasks(self):
                events.append("update_idletasks")

            def destroy(self):
                events.append("destroy")

        class _SmokeTk:
            @staticmethod
            def Tk():
                return _SmokeRoot()

        class _SmokeTtk:
            @staticmethod
            def Style():
                return object()

        class _SmokeApplication:
            def __init__(self, _root):
                events.append("application")

        components = {
            "application": _SmokeApplication,
            "cleanup": lambda: None,
            "configure_style": lambda _style: events.append("style"),
            "fast_startup": lambda: events.append("fast_startup"),
            "messagebox": None,
            "optimize_memory": lambda: events.append("optimize_memory"),
            "tk": _SmokeTk,
            "ttk": _SmokeTtk,
        }
        with patch.dict(
            os.environ,
            {"AFC_RELEASE_SMOKE_TEST": "1", "SUPPRESS_MESSAGEBOXES": "1"},
            clear=False,
        ), patch.object(
            release_main,
            "_load_runtime_components",
            return_value=components,
        ):
            self.assertEqual(0, release_main.main())

        self.assertIn("application", events)
        self.assertIn("update_idletasks", events)
        self.assertIn("destroy", events)

    def test_release_main_retries_transient_frozen_executable_lock(self):
        import project.release_main as release_main

        components = {
            "application": lambda _root: None,
            "cleanup": lambda: None,
            "configure_style": lambda _style: None,
            "fast_startup": lambda: None,
            "messagebox": None,
            "optimize_memory": lambda: None,
            "tk": SimpleNamespace(
                Tk=lambda: SimpleNamespace(
                    withdraw=lambda: None,
                    title=lambda _value: None,
                    update_idletasks=lambda: None,
                    destroy=lambda: None,
                )
            ),
            "ttk": SimpleNamespace(Style=lambda: object()),
        }
        transient_error = PermissionError(13, "Permission denied", "AFC2.0.2alpha.exe")
        with patch.dict(
            os.environ,
            {"AFC_RELEASE_SMOKE_TEST": "1", "SUPPRESS_MESSAGEBOXES": "1"},
            clear=False,
        ), patch.object(
            sys,
            "frozen",
            True,
            create=True,
        ), patch.object(
            release_main,
            "_load_runtime_components",
            side_effect=[transient_error, transient_error, components],
        ) as loader, patch.object(release_main.time, "sleep") as sleeper:
            self.assertEqual(0, release_main.main())

        self.assertEqual(3, loader.call_count)
        self.assertEqual(2, sleeper.call_count)

    def test_packaging_spec_collects_dynamic_mkl_runtime(self):
        spec_text = (PROJECT_ROOT / "packaging" / "AFC2_onedir.spec").read_text(
            encoding="utf-8"
        )
        for runtime_name in (
            "mkl_rt.2.dll",
            "mkl_core.2.dll",
            "mkl_intel_thread.2.dll",
            "mkl_def.2.dll",
            "mkl_vml_def.2.dll",
            "libiomp5md.dll",
        ):
            self.assertIn(runtime_name, spec_text)

        self.assertNotIn('"project.prediction_support",', spec_text)
        self.assertNotIn('"project.release_prediction",', spec_text)
        self.assertIn('"IPython",', spec_text)
        self.assertIn('"jedi",', spec_text)
        self.assertIn('"pkg_resources",', spec_text)
        verifier_text = (
            PROJECT_ROOT / "scripts" / "verification" / "verify_release_build.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn('$RequiredArchiveModules = @(', verifier_text)
        self.assertIn('"project.prediction_support"', verifier_text)
        self.assertIn('"project.release_prediction"', verifier_text)
        self.assertIn('"SampleData.rg"', verifier_text)
        self.assertIn('"ProcessDataPath.txt"', verifier_text)
        self.assertIn('$RuntimeInputPatterns = @("*.csv", "SampleData*.txt")', verifier_text)

    def test_build_script_uses_dedicated_release_versions_directory(self):
        build_script = (PROJECT_ROOT / "scripts" / "build_release.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn('Join-Path $ProjectRoot "release_versions"', build_script)
        self.assertIn('$FinalReleaseDirectory = Join-Path $OutputRoot $ProductName', build_script)
        self.assertNotIn('Join-Path $ProjectOutputRoot "release"', build_script)
        for runtime_name in (
            "SampleData.csv",
            "SampleData.txt",
            "iipinc.txt",
            "SampleData.rg",
            "ProcessDataPath.txt",
        ):
            self.assertIn(runtime_name, build_script)
        self.assertIn('$RuntimeInputPatterns = @("*.csv", "SampleData*.txt")', build_script)
        self.assertIn("Get-FileSha256WithRetry -Path $restoredPath", build_script)

    def test_public_plot_toolbar_has_default_hidden_prediction_option(self):
        ui_source = (SRC_ROOT / "project" / "ui_bootstrap.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('"显示实测"', ui_source)
        self.assertIn('self.show_prediction_load_var = tk.BooleanVar(value=False)', ui_source)
        self.assertIn('self.data_plot_toolbar, "预测负载", self.show_prediction_load_var', ui_source)
        self.assertNotIn("show_measured_curve_btn", ui_source)
        self.assertNotIn("show_reconstructed_curve_btn", ui_source)
        self.assertNotIn("baseline_ratio = 0.30", ui_source)
        self.assertNotIn("ideal_frame.grid_propagate(False)", ui_source)

    def test_strict_sampledata_pair_resolution_and_validation(self):
        manager = SampleManagerMixin()
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            wrong_csv_path = root / "sampledata.CSV"
            wrong_txt_path = root / "SAMPLEDATA.txt"
            wrong_csv_path.write_text("1,2,3,4,P1\n", encoding="utf-8")
            wrong_txt_path.write_text("P1:1:T0:0-4;\n", encoding="utf-8")

            self.assertEqual(
                (None, None, None),
                manager.resolve_sampledata_files(root, strict_root=True),
            )
            self.assertIn("未找到完整", manager._sampledata_resolution_error)

            wrong_csv_path.unlink()
            wrong_txt_path.unlink()
            csv_path = root / "SampleData.csv"
            txt_path = root / "SampleData.txt"
            csv_path.write_text("1,2,3,4,P1\n", encoding="utf-8")
            txt_path.write_text("P1:1:T0:0-4;\n", encoding="utf-8")

            resolved_dir, resolved_csv, resolved_txt = manager.resolve_sampledata_files(
                root,
                strict_root=True,
            )
            self.assertEqual(str(root), os.fspath(resolved_dir))
            self.assertEqual(csv_path, Path(resolved_csv))
            self.assertEqual(txt_path, Path(resolved_txt))

            txt_path.unlink()
            self.assertEqual(
                (None, None, None),
                manager.resolve_sampledata_files(root, strict_root=True),
            )
            self.assertIn("缺少 SampleData.txt", manager._sampledata_resolution_error)

            txt_path.write_text("", encoding="utf-8")
            self.assertEqual(
                (None, None, None),
                manager.resolve_sampledata_files(root, strict_root=True),
            )
            self.assertIn("为空文件", manager._sampledata_resolution_error)

            txt_path.unlink()
            csv_path.unlink()
            nested = root / "SampleData"
            nested.mkdir()
            (nested / "SampleData.csv").write_text("1,2,3,4,P1\n", encoding="utf-8")
            (nested / "SampleData.txt").write_text("P1:1:T0:0-4;\n", encoding="utf-8")
            self.assertEqual(
                (None, None, None),
                manager.resolve_sampledata_files(root, strict_root=True),
            )
            nested_result = manager.resolve_sampledata_files(root, strict_root=False)
            self.assertEqual(nested, Path(nested_result[0]))

    def test_sample_program_txt_accepts_gb18030_chinese_program_name(self):
        manager = SampleManagerMixin()
        with TemporaryDirectory() as temp_dir:
            txt_path = Path(temp_dir) / "SampleData.txt"
            txt_path.write_bytes(
                "1D10-外形.NC:613448256:T0:23-643;\r\n".encode("gb18030")
            )
            programs = manager.parse_sample_program_file(txt_path)

        self.assertEqual("gb18030", manager._sample_program_file_encoding)
        self.assertEqual("613448256", programs["1D10-外形.NC"]["program_number"])
        self.assertEqual([(23, 643)], programs["1D10-外形.NC"]["tools"]["T0"])

    def test_sample_program_txt_accepts_common_unicode_and_legacy_encodings(self):
        cases = (
            ("utf-8-sig", "外形"),
            ("utf-16", "外形"),
            ("utf-16-le", "外形"),
            ("utf-16-be", "外形"),
            ("utf-32", "外形"),
            ("utf-32-le", "外形"),
            ("utf-32-be", "外形"),
            ("big5", "外形"),
            ("shift_jis", "外形テスト"),
            ("euc_jp", "外形テスト"),
            ("euc_kr", "외형"),
            ("cp1252", "Contouré"),
        )
        manager = SampleManagerMixin()
        with TemporaryDirectory() as temp_dir:
            txt_path = Path(temp_dir) / "SampleData.txt"
            for source_encoding, program_name in cases:
                with self.subTest(source_encoding=source_encoding):
                    txt_path.write_bytes(
                        f"{program_name}:123:T0:1-9;\r\n".encode(source_encoding)
                    )
                    programs = manager.parse_sample_program_file(txt_path)
                    self.assertIn(program_name, programs)
                    self.assertEqual([(1, 9)], programs[program_name]["tools"]["T0"])

    def test_sampledata_csv_encoding_detection_accepts_utf16_and_gb18030(self):
        manager = SampleManagerMixin()
        cases = (
            ("utf-16", "程序一"),
            ("utf-16-le", "程序一"),
            ("utf-16-be", "程序一"),
            ("utf-32", "程序一"),
            ("gb18030", "程序一"),
        )
        with TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "SampleData.csv"
            for source_encoding, program_number in cases:
                with self.subTest(source_encoding=source_encoding):
                    csv_path.write_bytes(
                        f"1,2,3,4,{program_number}\r\n".encode(source_encoding)
                    )
                    detected = manager._detect_sampledata_csv_encoding(csv_path)
                    decoded = pd.read_csv(
                        csv_path,
                        header=None,
                        encoding=detected,
                        dtype={4: str},
                    )
                    self.assertEqual(program_number, decoded.iloc[0, 4])

    def test_manual_sampledata_import_uses_selected_csv_and_txt_pair(self):
        manager = InputIdleMixin()
        manager._ensure_nc_loaded_before_measurement = lambda: True
        manager._refresh_import_order_controls = lambda: None
        loaded = []
        manager.load_sample_data_from_paths = lambda *args, **kwargs: loaded.append(
            (args, kwargs)
        )
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "SampleData_1.csv"
            txt_path = root / "SampleData_1.txt"
            csv_path.write_text("1,2,3,4,P1\n", encoding="utf-8")
            txt_path.write_text("P1:1:T0:0-4;\n", encoding="utf-8")
            with patch(
                "project.input_idle.filedialog.askopenfilenames",
                return_value=(str(csv_path), str(txt_path)),
            ):
                manager.browse_sample_bundle()

        self.assertEqual(1, len(loaded))
        args, kwargs = loaded[0]
        self.assertEqual(csv_path, Path(args[0]))
        self.assertEqual(txt_path, Path(args[1]))
        self.assertFalse(kwargs["silent"])
        self.assertEqual(root, Path(kwargs["sample_dir"]))

    def test_manual_sampledata_import_rejects_single_file_selection(self):
        manager = InputIdleMixin()
        manager._ensure_nc_loaded_before_measurement = lambda: True
        manager._refresh_import_order_controls = lambda: None
        manager.load_sample_data_from_paths = lambda *_args, **_kwargs: self.fail(
            "不完整文件对不应进入加载"
        )
        with TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "SampleData.csv"
            csv_path.write_text("1,2,3,4,P1\n", encoding="utf-8")
            with patch(
                "project.input_idle.filedialog.askopenfilenames",
                return_value=(str(csv_path),),
            ), patch("project.input_idle.messagebox.showwarning") as warning:
                manager.browse_sample_bundle()

        warning.assert_called_once()
        self.assertIn("同时选择", warning.call_args.args[1])

    def test_strict_sampledata_exact_name_is_selected_without_casefold_lookup(self):
        manager = SampleManagerMixin()
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            expected_csv = str(root / "SampleData.csv")
            expected_txt = str(root / "SampleData.txt")

            def _exact(_directory, filename):
                return expected_csv if filename == "SampleData.csv" else expected_txt

            with patch.object(manager, "_find_file_exact", side_effect=_exact), patch.object(
                manager,
                "_validate_sampledata_input_file",
            ):
                resolved = manager.resolve_sampledata_files(root, strict_root=True)
            self.assertEqual((root, expected_csv, expected_txt), resolved)

    @unittest.skipUnless(os.name == "nt", "文件共享占用检查仅适用于 Windows")
    def test_strict_sampledata_exclusive_writer_is_reported(self):
        import ctypes
        from ctypes import wintypes

        manager = SampleManagerMixin()
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "SampleData.csv"
            txt_path = root / "SampleData.txt"
            csv_path.write_text("1,2,3,4,P1\n", encoding="utf-8")
            txt_path.write_text("P1:1:T0:0-4;\n", encoding="utf-8")

            create_file = ctypes.windll.kernel32.CreateFileW
            create_file.argtypes = [
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.LPVOID,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.HANDLE,
            ]
            create_file.restype = wintypes.HANDLE
            close_handle = ctypes.windll.kernel32.CloseHandle
            close_handle.argtypes = [wintypes.HANDLE]
            close_handle.restype = wintypes.BOOL
            handle = create_file(
                str(csv_path),
                0x80000000,
                0,
                None,
                3,
                0x80,
                None,
            )
            invalid_handle = ctypes.c_void_p(-1).value
            self.assertNotEqual(invalid_handle, handle)
            try:
                self.assertEqual(
                    (None, None, None),
                    manager.resolve_sampledata_files(root, strict_root=True),
                )
                self.assertIn("不可读取或仍被占用", manager._sampledata_resolution_error)
            finally:
                close_handle(handle)

    def test_release_startup_attempts_exact_root_only_once(self):
        harness = _ReleaseStartupHarness()
        harness.auto_load_sample_bundle()
        harness.auto_load_sample_bundle()
        self.assertEqual(1, len(harness.calls))
        _root, kwargs = harness.calls[0]
        self.assertTrue(kwargs["strict_root"])
        self.assertTrue(kwargs["clear_on_failure"])
        self.assertEqual(1, len(harness.experiment_resolve_calls))
        self.assertEqual([], harness.experiment_load_calls)
        self.assertTrue(harness._sampledata_startup_attempted)

    def test_release_startup_prefers_sampledata_over_experiment_csv(self):
        harness = _ReleaseStartupHarness(
            sample_success=True,
            experiment_path="ignored.csv",
        )
        harness.auto_load_sample_bundle()
        self.assertEqual([], harness.experiment_resolve_calls)
        self.assertEqual([], harness.experiment_load_calls)

    def test_release_startup_falls_back_to_experiment_csv(self):
        harness = _ReleaseStartupHarness(experiment_path="measurement.csv")
        harness.auto_load_sample_bundle()
        self.assertEqual(1, len(harness.experiment_resolve_calls))
        self.assertEqual([("measurement.csv", True)], harness.experiment_load_calls)

    def test_experiment_csv_resolution_requires_one_channel_export(self):
        manager = InputIdleMixin()
        manager.detect_file_encoding = lambda _path: "utf-8"
        channel_payload = (
            "<Scope>\n"
            "<ChannelInfo>,<True>,<5>,<实际速度>,<5>,<SP轴>\n"
            "<ChannelData>,<1>,<2>\n"
        )
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            measurement = root / "measurement.csv"
            measurement.write_text(channel_payload, encoding="utf-8")
            (root / "ProcessInfo.csv").write_text("N,S,F\n1,2,3\n", encoding="utf-8")
            (root / "SampleData.csv").write_text(channel_payload, encoding="utf-8")

            self.assertEqual(
                measurement,
                Path(manager.resolve_experiment_measurement_file(root)),
            )

            (root / "second.csv").write_text(channel_payload, encoding="utf-8")
            self.assertIsNone(manager.resolve_experiment_measurement_file(root))
            self.assertIn("找到多个实验实测 CSV", manager._experiment_measurement_resolution_error)

    def test_release_parse_failure_clears_stale_sample_state(self):
        harness = _LoadFailureHarness()
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "SampleData.csv"
            txt_path = root / "SampleData.txt"
            csv_path.write_text("1,2,3,4,P1\n", encoding="utf-8")
            txt_path.write_text("这不是有效的程序范围\n", encoding="utf-8")
            loaded = harness.load_sample_data_from_paths(
                str(csv_path),
                str(txt_path),
                silent=True,
                sample_dir=str(root),
            )
        self.assertFalse(loaded)
        self.assertFalse(harness.sample_data_loaded)
        self.assertGreaterEqual(harness.reset_count, 1)
        self.assertIn("未解析到有效程序", harness.sample_auto_status_var.get())

    def test_inverse_prediction_ke_idle_and_zero_mrr_match_baseline_rules(self):
        frame = pd.DataFrame(
            {
                "actual_load": [10.0, 16.0, 20.0, 18.0],
                "idle_power": [10.0, 10.0, 10.0, 10.0],
                "ap": [0.0, 1.0, 1.0, 1.0],
                "mrr": [0.0, 2.0, 2.0, 2.0],
                "prediction_valid": [True, True, True, True],
                "process_anchor_mask": [True, True, True, True],
            }
        )
        result = append_inverse_prediction_channels(
            frame,
            sigma_idle=0.0,
            delta_mrr=0.0,
            idle_mask=np.array([True, False, False, False]),
            ke_value=2.0,
        )

        self.assertTrue(np.isnan(result.loc[0, "sample_kc"]))
        self.assertEqual(10.0, result.loc[0, "predicted_load"])
        np.testing.assert_allclose(
            result.loc[1:, "sample_kc"].to_numpy(dtype=float),
            np.array([2.0, 4.0, 3.0]),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.loc[1:, "predicted_load"].to_numpy(dtype=float),
            np.array([16.0, 20.0, 18.0]),
            rtol=0.0,
            atol=1e-12,
        )

    def test_iipinc_pair_deduplication_scaling_and_midpoint_mapping(self):
        rows = [
            "1 10 20 30 40 50 60 1000000 extra",
            "1 10 20 30 40 50 60 1000000 extra",
            "1 11 21 31 41 51 61 2000000",
            "1 11 21 31 41 51 61 2000000",
            "3 12 22 32 42 52 62 3000000",
            "3 12 22 32 42 52 62 3000000",
        ]
        parsed = parse_iipinc_rows(rows)
        self.assertEqual(6, parsed["physical_row_count"])
        self.assertEqual(3, parsed["deduplicated_point_count"])
        self.assertEqual("paired", parsed["row_format"])
        self.assertEqual({0, 2}, set(parsed["feeds_by_line"]))
        np.testing.assert_allclose(
            parsed["feeds_by_line"][0],
            np.asarray([1000000.0, 2000000.0]) * IIPINC_FEED_SCALE,
        )

        mapped, covered = map_iipinc_feed_to_samples(
            np.asarray([0, 0, 0, 0, 1, 2, 2]),
            parsed["feeds_by_line"],
        )
        np.testing.assert_allclose(mapped[:4], [60.0, 60.0, 120.0, 120.0])
        self.assertTrue(np.isnan(mapped[4]))
        np.testing.assert_allclose(mapped[5:], [180.0, 180.0])
        np.testing.assert_array_equal(covered, [True, True, True, True, False, True, True])

    def test_iipinc_single_rows_are_used_directly(self):
        parsed = parse_iipinc_rows(
            [
                "1 0 0 0 0 0 0 1000000",
                "1 0 0 0 0 0 0 2000000",
                "3 0 0 0 0 0 0 3000000",
                "Total periods 3",
            ]
        )

        self.assertEqual(3, parsed["physical_row_count"])
        self.assertEqual(3, parsed["deduplicated_point_count"])
        self.assertEqual(3, parsed["declared_period_count"])
        self.assertEqual("single", parsed["row_format"])
        np.testing.assert_allclose(
            parsed["feeds_by_line"][0],
            np.asarray([1000000.0, 2000000.0]) * IIPINC_SINGLE_FEED_SCALE,
        )
        np.testing.assert_allclose(
            parsed["feeds_by_line"][2],
            np.asarray([3000000.0]) * IIPINC_SINGLE_FEED_SCALE,
        )

    def test_iipinc_rejects_illegal_rows(self):
        with self.assertRaisesRegex(IipincFormatError, "非法数值"):
            parse_iipinc_rows(
                [
                    "1 0 0 0 0 bad 0 1",
                    "1 0 0 0 0 bad 0 1",
                ]
            )
        with self.assertRaisesRegex(IipincFormatError, "数量与数据行数不一致"):
            parse_iipinc_rows(
                [
                    "1 0 0 0 0 0 0 1",
                    "Total periods 2",
                ]
            )

    def test_nonnegative_fit_recovers_known_kc_ke(self):
        ap_values = np.asarray([1.0, 1.0, 2.0, 2.0, 3.0, 4.0])
        mrr_values = np.asarray([1.0, 2.0, 2.0, 6.0, 3.0, 8.0])
        target = 2.5 * mrr_values + 4.0 * ap_values
        result = fit_nonnegative_kc_ke(mrr_values, ap_values, target)
        self.assertAlmostEqual(2.5, result["kc_value"], places=10)
        self.assertAlmostEqual(4.0, result["ke_value"], places=10)
        self.assertEqual(2, result["matrix_rank"])
        self.assertEqual(6, result["sample_count"])

    @staticmethod
    def _write_iipinc(path, line_numbers, command_feeds, *, duplicate_rows=True):
        rows = []
        feed_scale = IIPINC_FEED_SCALE if duplicate_rows else IIPINC_SINGLE_FEED_SCALE
        for line_number, command_feed in zip(line_numbers, command_feeds):
            raw_feed = float(command_feed) / feed_scale
            row = f"{int(line_number) + 1} 0 0 0 0 0 0 {raw_feed:.12g}"
            rows.extend([row, row] if duplicate_rows else [row])
        Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")

    def test_release_fit_uses_process_feed_but_prediction_uses_iip_feed(self):
        with TemporaryDirectory() as temp_dir:
            iip_path = Path(temp_dir) / "iipinc.txt"
            self._write_iipinc(iip_path, range(8), [0.0] + [120.0] * 7)
            harness = _ReleasePredictionHarness(iip_path)
            payload = harness._build_sampledata_prediction_payload_for_mode()

        self.assertAlmostEqual(2.0, payload["fit_result"]["kc_value"], places=9)
        self.assertAlmostEqual(3.0, payload["fit_result"]["ke_value"], places=9)
        np.testing.assert_allclose(payload["mapped_process_feed"], 60.0)
        np.testing.assert_allclose(payload["mapped_feed"], [0.0] + [120.0] * 7)
        expected = 250.0 + 2.0 * (harness.ap * harness.ae * 120.0 / 60.0) + 3.0 * harness.ap
        expected[0] = 250.0
        np.testing.assert_allclose(payload["predicted_load"], expected, atol=1e-9)
        self.assertEqual(250.0, harness.p_idle_var.get())
        self.assertEqual(0, payload["fallback_line_count"])
        self.assertIn("空载按 250 W", payload["status_text"])

    def test_release_prediction_uses_experiment_actual_load_column(self):
        with TemporaryDirectory() as temp_dir:
            iip_path = Path(temp_dir) / "iipinc.txt"
            self._write_iipinc(iip_path, range(8), [120.0] * 8)
            harness = _ReleasePredictionHarness(iip_path)
            actual_load = harness.sample_data_values[:, 1].copy()
            harness.sample_data_mode = "experiment_measurement"
            harness.manual_measurement_path = Path(temp_dir) / "measurement.csv"
            harness.sample_data_values = np.column_stack(
                [actual_load, np.zeros(8), np.zeros(8)]
            )
            payload = harness._build_sampledata_prediction_payload_for_mode()

        self.assertEqual("实际负载", payload["actual_label"])
        np.testing.assert_allclose(payload["actual_load"], actual_load)
        self.assertAlmostEqual(2.0, payload["fit_result"]["kc_value"], places=9)
        self.assertAlmostEqual(3.0, payload["fit_result"]["ke_value"], places=9)

    def test_process_info_export_replaces_f_and_mrr_with_command_feed(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            iip_path = root / "iipinc.txt"
            self._write_iipinc(
                iip_path,
                [0, 0],
                [120.0, 180.0],
                duplicate_rows=False,
            )
            harness = _ReleaseProcessExportHarness(iip_path)
            output_path = harness._save_process_info_csv(root)
            exported = pd.read_csv(output_path, encoding="utf-8-sig")

        np.testing.assert_allclose(exported["F(mm/min)"], [120.0, 180.0, 60.0])
        np.testing.assert_allclose(exported["MRR(mm3/s)"], [4.0, 6.0, 2.0])
        self.assertEqual(2, harness._last_processinfo_feed_export["replaced_point_count"])
        self.assertEqual(1, harness._last_processinfo_feed_export["fallback_line_count"])
        self.assertIn("替换 2/3", harness._last_processinfo_feed_export_status)

    def test_process_info_export_without_iip_keeps_process_feed(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            harness = _ReleaseProcessExportHarness(root / "iipinc.txt")
            output_path = harness._save_process_info_csv(root)
            exported = pd.read_csv(output_path, encoding="utf-8-sig")

        np.testing.assert_allclose(exported["F(mm/min)"], [60.0, 60.0, 60.0])
        np.testing.assert_allclose(exported["MRR(mm3/s)"], [2.0, 2.0, 2.0])
        self.assertEqual(0, harness._last_processinfo_feed_export["replaced_point_count"])
        self.assertIn("F 保留编程进给", harness._last_processinfo_feed_export_status)

    def test_iipinc_missing_and_partial_lines_fall_back_to_process_feed(self):
        with TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "iipinc.txt"
            missing_harness = _ReleasePredictionHarness(missing_path)
            missing = missing_harness._build_sampledata_prediction_payload_for_mode()
            np.testing.assert_allclose(missing["mapped_feed"], 60.0)
            self.assertFalse(missing["iipinc_valid"])
            self.assertEqual(8, missing["fallback_line_count"])

            missing_path.write_text(
                "1 0 0 0 0 bad 0 1\n",
                encoding="utf-8",
            )
            invalid_harness = _ReleasePredictionHarness(missing_path)
            invalid = invalid_harness._build_sampledata_prediction_payload_for_mode()
            np.testing.assert_allclose(invalid["mapped_feed"], 60.0)
            self.assertFalse(invalid["iipinc_valid"])
            self.assertIn("非法数值", invalid["iipinc_error"])

            self._write_iipinc(missing_path, [0, 2, 4, 6], [120.0] * 4)
            partial_harness = _ReleasePredictionHarness(missing_path)
            partial = partial_harness._build_sampledata_prediction_payload_for_mode()
            np.testing.assert_allclose(
                partial["mapped_feed"],
                [120.0, 60.0, 120.0, 60.0, 120.0, 60.0, 120.0, 60.0],
            )
            self.assertTrue(partial["iipinc_valid"])
            self.assertEqual(4, partial["fallback_line_count"])

    def test_prediction_cache_uses_iip_file_signature_and_preserves_nan_gap(self):
        with TemporaryDirectory() as temp_dir:
            iip_path = Path(temp_dir) / "iipinc.txt"
            self._write_iipinc(iip_path, range(8), [90.0] * 8)
            harness = _ReleasePredictionHarness(iip_path)
            harness.sample_data_values[3, 1] = np.nan
            first = harness._build_sampledata_prediction_payload_for_mode()
            second = harness._build_sampledata_prediction_payload_for_mode()
            self.assertIs(first, second)
            self.assertTrue(np.isnan(first["predicted_load"][3]))

            self._write_iipinc(iip_path, range(8), [150.0] * 8)
            third = harness._build_sampledata_prediction_payload_for_mode()
            self.assertIsNot(first, third)
            self.assertFalse(np.allclose(first["mapped_feed"], third["mapped_feed"]))

    def test_release_plot_calculates_prediction_but_hides_it_until_selected(self):
        harness = _ReleasePlotHarness()
        mapping_records = [
            {
                "interval_id": "SEG0001",
                "segment_type": "steady",
                "state_code": 2,
                "sample_start_idx": 0,
                "sample_end_idx": 4,
            }
        ]
        before = harness.build_segmentation_sample_background_masks(
            np.asarray([250.0, 260.0, np.nan, 270.0, 250.0]),
            None,
            mapping_records,
            valid_mask=np.asarray([True, True, False, True, True]),
        )["state_masks"]
        self.assertTrue(harness._render_segmentation_sample_overlay_view(mapping_records))
        figure = harness.figures[0]
        try:
            axis = figure.axes[0]
            labels = [line.get_label() for line in axis.lines]
            self.assertIn("实际负载", labels)
            self.assertNotIn("预测负载", labels)
            self.assertEqual("实际负载与区间划分", axis.get_title())
            self.assertEqual(1, harness.prediction_build_count)
        finally:
            plt.close(figure)

        harness.show_prediction_load_var.set(True)
        self.assertTrue(harness._render_segmentation_sample_overlay_view(mapping_records))
        figure = harness.figures[0]
        try:
            axis = figure.axes[0]
            labels = [line.get_label() for line in axis.lines]
            self.assertIn("预测负载", labels)
            self.assertEqual("实际负载、预测负载与区间划分", axis.get_title())
            predicted_line = next(line for line in axis.lines if line.get_label() == "预测负载")
            self.assertTrue(np.isnan(np.asarray(predicted_line.get_ydata(), dtype=float)).any())
            after = harness.build_segmentation_sample_background_masks(
                np.asarray([250.0, 260.0, np.nan, 275.0, 250.0]),
                None,
                mapping_records,
                valid_mask=np.asarray([True, True, False, True, True]),
            )["state_masks"]
            for state_name in before:
                np.testing.assert_array_equal(before[state_name], after[state_name])
        finally:
            plt.close(figure)

    def test_release_experiment_display_resolver_runs_lightweight_inverse(self):
        harness = _ReleasePlotHarness()
        harness.sample_data_mode = "experiment_measurement"
        harness.sample_data_source = _ValueVar(0)
        harness.sample_data_values = np.column_stack(
            [np.asarray([250.0, 260.0, np.nan, 270.0, 250.0]), np.zeros(5), np.zeros(5)]
        )
        harness.manual_measurement_data = {
            "actual_load": harness.sample_data_values[:, 0].copy()
        }

        predicted = harness._resolve_segmentation_display_prediction(5)

        np.testing.assert_allclose(predicted, harness.predicted, equal_nan=True)
        self.assertEqual(1, harness.prediction_build_count)

    def test_steady_mode_is_deterministic_and_nonsteady_remains_pointwise(self):
        kc_hat, _sigma, _valid = summarize_interval_kc_mode_statistics([1.0, 3.0])
        self.assertEqual(1.0, kc_hat)

        frame = pd.DataFrame(
            {
                "predicted_kc": [2.0, 4.0, 2.0, 5.0],
                "predicted_load": [16.0, 20.0, 16.0, 22.0],
                "predicted_kc_source": ["measurement_point_kc"] * 4,
                "kc_point": [2.0, 4.0, 2.0, 5.0],
                "sample_kc": [2.0, 4.0, 2.0, 5.0],
                "kc_valid": [True] * 4,
                "sample_kc_valid": [True] * 4,
                "prediction_valid": [True] * 4,
                "is_idle_point": [False] * 4,
                "idle_power": [10.0] * 4,
                "ap": [1.0] * 4,
                "mrr": [2.0] * 4,
            }
        )
        result = apply_steady_representative_prediction(
            frame,
            [{"sample_start_idx": 0, "sample_end_idx": 2}],
            ke_value=2.0,
        )
        np.testing.assert_array_equal(
            result["predicted_kc"].to_numpy(dtype=float),
            np.array([2.0, 2.0, 2.0, 5.0]),
        )
        np.testing.assert_array_equal(
            result["predicted_load"].to_numpy(dtype=float),
            np.array([16.0, 16.0, 16.0, 22.0]),
        )
        self.assertEqual("measurement_point_kc", result.loc[3, "predicted_kc_source"])

    def test_steady_average_and_live_ratio_refresh(self):
        harness = _PowerDisplayHarness()

        harness._refresh_current_ideal_display()

        self.assertEqual("200.000", harness.sample_avg_var.get())
        self.assertEqual("500.000", harness.sample_ideal_var.get())
        self.assertEqual(200.0, harness._current_display_power_mean)

    def test_tool_average_uses_only_finite_cutting_steady_points(self):
        harness = PlotSupportMixin()
        harness.sample_data_loaded = True
        harness.sample_data_source = _ValueVar(1)
        harness.sample_data_values = np.asarray(
            [
                [0.0, 100.0],
                [0.0, np.nan],
                [0.0, 300.0],
                [0.0, 900.0],
            ]
        )
        harness.sample_data_line_numbers = np.asarray([10, 10, 11, 12])
        harness.sample_data_program_numbers = np.asarray(["1", "1", "1", "1"])
        harness.sample_programs = {
            "P1": {
                "program_number": "1",
                "tools": {"T0": [(10, 11)]},
            }
        }
        steady_record = {
            "segment_type": "steady",
            "start_line": 10,
            "end_line": 11,
        }
        harness._get_current_interval_records = lambda **_kwargs: [steady_record]
        harness._get_steady_interval_records = lambda records=None: list(records or [])
        harness._build_interval_sample_mask = (
            lambda _interval, _size, line_numbers=None: np.asarray(
                [True, False, True, False]
            )
        )

        mean_value, count, ranges = harness.compute_tool_measured_mean("P1", "T0")

        self.assertEqual(200.0, mean_value)
        self.assertEqual(2, count)
        self.assertEqual([(10, 11)], ranges)

    def test_tool_average_is_empty_without_cutting_steady_intervals(self):
        harness = PlotSupportMixin()
        harness.sample_data_loaded = True
        harness.sample_data_source = _ValueVar(1)
        harness.sample_data_values = np.asarray([[0.0, 100.0]])
        harness.sample_data_line_numbers = np.asarray([10])
        harness.sample_data_program_numbers = np.asarray(["1"])
        harness.sample_programs = {
            "P1": {
                "program_number": "1",
                "tools": {"T0": [(10, 10)]},
            }
        }
        harness._get_current_interval_records = lambda **_kwargs: []
        harness._get_steady_interval_records = lambda records=None: []

        mean_value, count, ranges = harness.compute_tool_measured_mean("P1", "T0")

        self.assertIsNone(mean_value)
        self.assertEqual(0, count)
        self.assertEqual([], ranges)

    def test_ratio_preview_updates_ideal_power_without_rescanning_samples(self):
        harness = SampleManagerMixin()
        harness._current_display_power_mean = 200.0
        harness.sample_ideal_var = _ValueVar("-")

        harness._update_ideal_power_preview(1.85)

        self.assertEqual("370.000", harness.sample_ideal_var.get())

    def test_selection_change_refreshes_power_display_with_processed_data(self):
        harness = SampleManagerMixin()
        harness._loading_sample_data = False
        harness._selection_change_job = None
        harness._pending_selection_signature = None
        harness._last_selection_signature = None
        harness._current_display_power_mean = 200.0
        harness.sample_display_mode = _ValueVar("tool")
        harness.sample_data_loaded = True
        harness.data = [{}]
        harness.build_sample_selection_signature = lambda: ("tool", "P1", "T0")
        harness.sync_adjustment_ratio_for_current_view = lambda: None
        refresh_calls = []
        plot_calls = []
        harness._refresh_current_ideal_display = lambda: refresh_calls.append(True)
        harness.generate_plots = lambda **kwargs: plot_calls.append(kwargs)
        harness.root = SimpleNamespace(
            after=lambda _delay, callback: callback(),
            after_cancel=lambda _job: None,
        )

        harness.on_sample_selection_change()

        self.assertIsNone(harness._current_display_power_mean)
        self.assertEqual([True], refresh_calls)
        self.assertEqual(1, len(plot_calls))

    def test_rg_optimizable_intervals_include_idle_and_cutting_steady(self):
        harness = IntervalRuntimeMixin()
        records = [
            {"segment_type": "idle", "start_idx": 0, "end_idx": 1},
            {"segment_type": "entry", "start_idx": 2, "end_idx": 2},
            {"segment_type": "steady", "start_idx": 3, "end_idx": 5},
            {"segment_type": "exit", "start_idx": 6, "end_idx": 6},
        ]

        optimizable = harness._get_optimizable_interval_records(records)
        steady_only = harness._get_steady_interval_records(records)

        self.assertEqual(
            ["idle", "steady"],
            [record["segment_type"] for record in optimizable],
        )
        self.assertEqual(
            ["steady"],
            [record["segment_type"] for record in steady_only],
        )

    def test_rg_output_contract_does_not_gain_average_or_ratio_columns(self):
        import project.analysis_export as export_module

        harness = _RgExportHarness()
        with TemporaryDirectory() as temp_dir, patch.object(
            export_module,
            "OUTPUT_DIR",
            Path(temp_dir),
        ), patch.object(export_module.messagebox, "showinfo"), patch.object(
            export_module.messagebox,
            "showwarning",
        ):
            harness._do_save_interval_info([("P1", "T0", 2.0)])
            raw = (Path(temp_dir) / "SampleData.rg").read_bytes()
        self.assertEqual(
            b"1\r\nP1;20.000000;10.0-11.0:10.000000;\r\n",
            raw,
        )


if __name__ == "__main__":
    unittest.main()
