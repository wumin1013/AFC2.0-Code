from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from project.input_idle import InputIdleMixin
from project.analysis_export import AnalysisExportMixin
from project.prediction_support import (
    append_inverse_prediction_channels,
    apply_steady_representative_prediction,
    summarize_interval_kc_mode_statistics,
)
from project.sample_manager import SampleManagerMixin


class _ReleaseStartupHarness(InputIdleMixin):
    release_mode = True

    def __init__(self):
        self.sample_data_loaded = False
        self.calls = []

    def load_sample_bundle_from_dir(self, base_dir, **kwargs):
        self.calls.append((str(base_dir), dict(kwargs)))
        return False

    def get_input_files(self):
        return []


class _ValueVar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


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

    def test_release_import_does_not_load_research_application(self):
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
        'project.prediction_support',
        'project.release_prediction',
        'sklearn',
    }
    or name.startswith('sklearn.')
]
if forbidden:
    raise SystemExit('forbidden imports: ' + ','.join(sorted(forbidden)))
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
        self.assertNotIn("ReleasePredictionMixin", mro_names)
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

    def test_packaging_spec_collects_dynamic_mkl_runtime(self):
        spec_text = (PROJECT_ROOT / "packaging" / "AFC2_onedir.spec").read_text(
            encoding="utf-8"
        )
        for runtime_name in (
            "mkl_core.2.dll",
            "mkl_intel_thread.2.dll",
            "mkl_def.2.dll",
            "mkl_vml_def.2.dll",
            "libiomp5md.dll",
        ):
            self.assertIn(runtime_name, spec_text)

    def test_build_script_uses_dedicated_release_versions_directory(self):
        build_script = (PROJECT_ROOT / "scripts" / "build_release.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn('Join-Path $ProjectRoot "release_versions"', build_script)
        self.assertIn('$FinalReleaseDirectory = Join-Path $OutputRoot $ProductName', build_script)
        self.assertNotIn('Join-Path $ProjectOutputRoot "release"', build_script)

    def test_public_plot_toolbar_keeps_only_optional_overlays(self):
        ui_source = (SRC_ROOT / "project" / "ui_bootstrap.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('"显示实测"', ui_source)
        self.assertNotIn('"显示预测负载"', ui_source)
        self.assertNotIn("show_measured_curve_btn", ui_source)
        self.assertNotIn("show_reconstructed_curve_btn", ui_source)
        self.assertNotIn("baseline_ratio = 0.30", ui_source)
        self.assertNotIn("ideal_frame.grid_propagate(False)", ui_source)

    def test_strict_sampledata_pair_resolution_and_validation(self):
        manager = SampleManagerMixin()
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "sampledata.CSV"
            txt_path = root / "SAMPLEDATA.txt"
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

    def test_strict_sampledata_case_conflict_is_not_silently_selected(self):
        manager = SampleManagerMixin()
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake_csv = [str(root / "SampleData.csv"), str(root / "sampledata.CSV")]
            fake_txt = [str(root / "SampleData.txt")]

            def _matches(_directory, filename):
                return fake_csv if filename.casefold().endswith(".csv") else fake_txt

            with patch.object(manager, "_find_files_case_insensitive", side_effect=_matches):
                self.assertEqual(
                    (None, None, None),
                    manager.resolve_sampledata_files(root, strict_root=True),
                )
            self.assertIn("文件名冲突", manager._sampledata_resolution_error)

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
        self.assertTrue(harness._sampledata_startup_attempted)

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
