"""Integration tests for input identity, export coordinates and silent failure."""
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from interval_analysis.interval_tool import file_sha256, generate_intervals, main, RG_SCHEMA


def fixture_files(root):
    root.mkdir(parents=True, exist_ok=True)
    process = root / "input.csv"
    with process.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["N", "S(r/min)", "ap(mm)", "ae(mm)", "F(mm/min)", "s(mm)", "MRR(mm3/s)", "G"])
        for index in range(100):
            ap = 0 if index < 10 or index >= 90 else 1
            writer.writerow([f"N{2 + index // 10}", 6000, ap, 2, 600, 0.1, 999, "G01 X1"])
    prediction = root / "prediction.json"
    payload = {"schema": "afc.spindle_prediction.1", "metadata": {
        "program_name": "O123", "process_file_sha256": file_sha256(process),
        "model_version": "test_fixture", "total_power_status": "research_estimate",
        "process_distance_semantics": "incremental_mm",
        "interval_status": "uncalibrated_total", "interval_kind": "cutting_interval_plus_base_point"},
        "rows": [{"source_index": i, "line": 2 + i // 10, "execution_instance": 0,
                  "status": "research_prediction", "total_power_w": 100 + i,
                  "total_lower_w": 90 + i, "total_upper_w": 110 + i,
                  "base_power_w": 50} for i in range(100)]}
    prediction.write_text(json.dumps(payload), encoding="utf-8")
    return process, prediction, payload


def test_complete_export(tmp_path):
    process, prediction, _ = fixture_files(tmp_path / "input")
    out = tmp_path / "out"
    status = generate_intervals(process, prediction, out)
    frame = pd.read_csv(out / "processinfo.csv")
    assert status["status"] == "completed"
    assert status["rg_coordinate_domain"] == "process_points_v1"
    assert frame.state_code.eq(2).any()
    expected = frame.loc[frame.state_code.eq(2), "predicted_spindle_power_w"].mean() * 2
    assert status["ideal_power_w"] == pytest.approx(expected)
    assert frame["MRR(mm3/s)"].max() == 20
    assert frame.point_index0.tolist() == list(range(10)) * 10
    assert (out / "sampledata.rg").read_text(encoding="utf-8").startswith(RG_SCHEMA + "\nO123;")
    assert not (out / "processinfo.csv").read_bytes().startswith(b"\xef\xbb\xbf")
    for name, reference in status["outputs"].items():
        assert reference["sha256"] == file_sha256(out / name)


@pytest.mark.parametrize("mutation", ["hash", "duplicate", "line", "nan", "bounds", "missing"])
def test_invalid_prediction_preserves_prior_outputs(tmp_path, mutation):
    process, prediction, payload = fixture_files(tmp_path / "input")
    out = tmp_path / "out"
    generate_intervals(process, prediction, out)
    before = {name: (out / name).read_bytes() for name in ("processinfo.csv", "sampledata.rg")}
    row = payload["rows"][30]
    if mutation == "hash":
        payload["metadata"]["process_file_sha256"] = "0" * 64
    elif mutation == "duplicate":
        row["source_index"] = 31
    elif mutation == "line":
        row["line"] += 1
    elif mutation == "nan":
        row["total_power_w"] = None
    elif mutation == "bounds":
        row["total_lower_w"] = 1e6
    else:
        payload["rows"].pop()
    prediction.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        generate_intervals(process, prediction, out)
    assert json.loads((out / "interval_status.json").read_text(encoding="utf-8"))["status"] == "failed"
    assert all((out / name).read_bytes() == value for name, value in before.items())


def test_missing_noncutting_power_is_not_fabricated(tmp_path):
    process, prediction, payload = fixture_files(tmp_path / "input")
    payload["rows"][0].update(status="unavailable_input", total_power_w=None)
    prediction.write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "out"
    status = generate_intervals(process, prediction, out)
    frame = pd.read_csv(out / "processinfo.csv")
    assert status["unavailable_power_rows"] == 1
    assert status["skipped_optimization_intervals_missing_power"] >= 1
    assert np.isnan(frame.iloc[0].predicted_spindle_power_w)


def test_no_input_overwrite_and_silent_error(tmp_path, capsys):
    process, prediction, _ = fixture_files(tmp_path)
    process.rename(tmp_path / "processinfo.csv")
    original = (tmp_path / "processinfo.csv").read_bytes()
    code = main(["--process", str(tmp_path / "processinfo.csv"), "--prediction", str(prediction), "--output-dir", str(tmp_path)])
    assert code == 2
    assert (tmp_path / "processinfo.csv").read_bytes() == original
    assert capsys.readouterr().out == ""


def test_busy_directory_does_not_change_other_status(tmp_path):
    process, prediction, _ = fixture_files(tmp_path / "input")
    out = tmp_path / "out"
    out.mkdir()
    (out / ".interval_tool.lock").write_text("occupied", encoding="utf-8")
    (out / "interval_status.json").write_text("existing", encoding="utf-8")
    assert main(["--process", str(process), "--prediction", str(prediction), "--output-dir", str(out)]) == 2
    assert (out / "interval_status.json").read_text(encoding="utf-8") == "existing"


def test_error_receipt_never_overwrites_input(tmp_path):
    process, prediction, _ = fixture_files(tmp_path)
    prediction.rename(tmp_path / "interval_status.json")
    original = (tmp_path / "interval_status.json").read_bytes()
    assert main(["--process", str(process), "--prediction", str(tmp_path / "interval_status.json"),
                 "--output-dir", str(tmp_path)]) == 2
    assert (tmp_path / "interval_status.json").read_bytes() == original


def test_explicit_cumulative_distance_is_not_summed_twice(tmp_path):
    process, prediction, payload = fixture_files(tmp_path / "input")
    frame = pd.read_csv(process)
    frame["s(mm)"] = np.arange(1, len(frame) + 1) * .1
    frame.to_csv(process, index=False, encoding="utf-8")
    payload["metadata"].update(process_file_sha256=file_sha256(process), process_distance_semantics="cumulative_mm")
    prediction.write_text(json.dumps(payload), encoding="utf-8")
    status = generate_intervals(process, prediction, tmp_path / "out")
    assert status["segmentation_diagnostics"]["path"]["span_mm"] == pytest.approx(10.)


def test_invalid_cli_replaces_completion_with_error(tmp_path, capsys):
    assert main(["--output-dir", str(tmp_path), "--unknown"]) == 2
    assert json.loads((tmp_path / "interval_status.json").read_text(encoding="utf-8"))["status"] == "failed"
    assert capsys.readouterr() == ("", "")


def test_duplicate_paths_do_not_bypass_input_protection(tmp_path):
    protected = tmp_path / "interval_status.json"
    protected.write_text("input must survive", encoding="utf-8")
    assert main(["--output-dir", str(tmp_path), "--prediction", str(tmp_path / "other.json"),
                 "--prediction", str(protected), "--unknown"]) == 2
    assert protected.read_text(encoding="utf-8") == "input must survive"
