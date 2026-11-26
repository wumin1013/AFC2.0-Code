"""Shared point-position convention without importing the desktop runtime."""
import ast
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from afc_common.process_path import uniform_path_boundaries


@pytest.mark.parametrize("length,count,direction", [
    (13.163390767199763, 26, -1),  # R01 step, NC34.
    (6.290897932251239, 12, -1),   # R01 outer, NC72.
    (15.141499999999994, 30, -1),  # R01 surface, NC90.
    (91.6666, 183, -1),           # R02 slope, NC245.
    (50.17331351316155, 100, 1),  # R03 slope, NC41.
])
def test_real_nc_endpoints_are_exact_without_changing_internal_boundaries(length, count, direction):
    legacy = tuple(length * index / count for index in range(count + 1))
    assert legacy[-1] == np.nextafter(length, direction * np.inf)
    bounds = uniform_path_boundaries(length, count)
    assert len(bounds) == count + 1
    assert bounds[0] == 0.0 and bounds[-1] == length
    assert bounds[1:-1] == legacy[1:-1]


@pytest.mark.parametrize("length,count", [(0., 1), (0., 7), (13.163390767199763, 1)])
def test_exact_endpoints_preserve_single_cell_and_zero_length(length, count):
    bounds = uniform_path_boundaries(length, count)
    assert len(bounds) == count + 1
    assert bounds[0] == 0.0 and bounds[-1] == length
    assert bounds == ((0.,) * (count + 1) if length == 0. else (0., length))


def test_zero_length_and_noninteger_spacing_preserve_all_source_points():
    assert uniform_path_boundaries(0., 4) == (0.,) * 5
    bounds = uniform_path_boundaries(1., 3)
    np.testing.assert_allclose(bounds, [0., 1/3, 2/3, 1.])
    assert bounds[-1] == 1.
    for length, count in [(-1., 2), (float("nan"), 2), (1., 0), (1., True)]:
        with pytest.raises(ValueError):
            uniform_path_boundaries(length, count)


def test_existing_instruction_group_method_uses_same_endpoint_convention():
    # Execute the real pure method's AST, avoiding the shared GUI import graph.
    tree = ast.parse((ROOT / "src/interval_analysis/processing_core.py").read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ProcessingCoreMixin")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_assign_group_distributed_path_positions")
    namespace = {"uniform_path_boundaries": uniform_path_boundaries}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "processing_core_pure_method", "exec"), namespace)
    fake = type("Core", (), {"_process_instruction_group_key": staticmethod(lambda row: row["line"])})()
    rows = [dict(line=21, s=value) for value in (0., 0., 1.5, .5)] + [dict(line=22, s=3.)]
    namespace[method.name](fake, rows)
    np.testing.assert_allclose([row["path_end"] for row in rows], [.5, 1., 1.5, 2., 5.])
    np.testing.assert_allclose([row["path_start"] for row in rows], [0., .5, 1., 1.5, 2.])
    assert len(rows) == 5


def test_training_identity_includes_shared_mapping_and_segmentation_sources():
    from spindle_power.load_prediction.training_checkpoint import source_fingerprints
    fingerprints = source_fingerprints()
    assert "shared/process_path.py" in fingerprints
    assert "shared/processing_core.py" in fingerprints
    assert "shared/segmentation/pipeline.py" in fingerprints
    assert all(len(value) == 64 for value in fingerprints.values())
