"""Plot generation must not initialize a second Tk application."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("backend", ["Agg", "TkAgg"])
def test_embedded_and_export_figures_do_not_create_tk_roots(tmp_path, backend):
    root = Path(__file__).resolve().parents[1]
    script = r'''
import sys
from pathlib import Path
from unittest.mock import patch
sys.path[:0] = [str(Path.cwd() / "src"), str(Path.cwd() / "tests")]
import tkinter
with patch.object(tkinter, "Tk", side_effect=AssertionError("unexpected Tk root")):
    from test_current_requirement import _VisualizationHarness, _process_frame, _test_config
    from interval_analysis.segmentation import SegmentationPipeline
    from interval_analysis.shared import matplotlib, plt
    from matplotlib._pylab_helpers import Gcf
    assert matplotlib.get_backend().lower() == sys.argv[1].lower()
    result = SegmentationPipeline(_test_config()).run(_process_frame([0.] * 5 + [10.] * 40 + [0.] * 5))
    view = _VisualizationHarness(result)
    with patch.object(plt, "subplots", side_effect=AssertionError("unexpected pyplot manager")):
        for _ in range(3):
            assert view.generate_plots(silent=True)
            assert view.figures[0].axes
        view.figures[0].savefig(Path(sys.argv[2]) / "preview.png")
        view._save_segmentation_process_overview(result.point_labels, result.intervals,
                                                 Path(sys.argv[2]) / "overview.png")
    assert not Gcf.get_all_fig_managers()
    assert tkinter._default_root is None
    for name in ("preview.png", "overview.png"):
        assert (Path(sys.argv[2]) / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
'''
    env = {**os.environ, "MPLBACKEND": backend, "MKL_THREADING_LAYER": "SEQUENTIAL",
           "PYTHONUTF8": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    completed = subprocess.run([sys.executable, "-c", script, backend, str(tmp_path)],
                               cwd=root, env=env, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
