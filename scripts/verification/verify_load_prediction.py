"""Read-only real-data checks; writes reports only under data/paper1."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spindle_power.load_prediction import MachineConfig, fit_base_power, inspect_dataset, parse_nc, plan_feed, read_measurement
from spindle_power.load_prediction.common import file_hash, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--test-nc", action="store_true", help="NC software validation with synthetic machine parameters only")
    args = parser.parse_args()
    output = ROOT / "data" / "paper1"
    report = inspect_dataset(args.data_root, deep=args.deep)
    save_json(output / ("real_dataset_report.json" if args.deep else "real_dataset_header_report.json"), report)
    print(f"CSV={report['csv_count']}; rounds={report['round_count']}; inspection={report['inspection']}", flush=True)
    print(f"missing features={report['missing_features']}", flush=True)
    audits = []
    for path in args.data_root.rglob("*"):
        if path.suffix.lower() == ".nc":
            parsed = parse_nc(path, MachineConfig())
            audits.append(dict(path=str(path), issues=parsed["issues"], segments=len(parsed["segments"])))
    save_json(output / "nc_configuration_audit.json", audits)
    print(f"NC configuration audits={len(audits)}", flush=True)
    if args.test_nc:
        config = MachineConfig(name="软件验证专用，非真实机床参数", max_velocity_mm_min=[6000.] * 3,
            max_acceleration_mm_s2=[1000.] * 3, max_jerk_mm_s3=[10000.] * 3,
            interpolation_period_s=.004, contour_tolerance_mm=.03, corner_tolerance_mm=.03,
            initial_position_mm=[0.] * 3, travel_min_mm=[-1000.] * 3, travel_max_mm=[1000.] * 3,
            home_position_mm=[0.] * 3, work_offsets_mm={"G54": [0.] * 3},
            tool_lengths_mm={str(i): 0. for i in range(21)}, fixed_rotary_degrees={"A": 0., "B": 0., "C": 0.},
            manufacturer_semantics={"G05.1": "contour_tolerance", "G43.4": "fixed_axis_length_compensation"},
            spindle_wait_s=0., tool_change_s=0., optional_stop="skip", provenance="synthetic_test_only")
        syntax = []
        for audit in audits:
            parsed = parse_nc(audit["path"], config)
            syntax.append(dict(path=audit["path"], issues=parsed["issues"], segments=len(parsed["segments"])))
        save_json(output / "nc_synthetic_configuration_audit.json", syntax)
        print(f"TEST config NC syntax: {len(syntax)} checked, {sum(bool(x['issues']) for x in syntax)} blocked", flush=True)
        plans = []
        for name in ("1D10-外形.NC", "2D10-曲面-中.NC"):
            path = args.data_root / "负载预测实验G代码及工件模型" / "P5_R82-R90" / name
            plan = plan_feed(path, config)
            plans.append(dict(path=str(path), points=len(plan.frame), report=plan.report))
            print(f"TEST plan {name}: {len(plan.frame)} points, {plan.report['total_automatic_time_s']:.2f} s; not a real machine estimate", flush=True)
        save_json(output / "feed_synthetic_validation.json", dict(machine=config, plans=plans))
    if args.calibrate:
        paths = sorted((args.data_root / "空载数据采集").glob("*.csv"))
        before = {str(path): file_hash(path) for path in paths}
        records = [read_measurement(path) for path in paths]
        print(f"Idle records loaded: {[len(r.time_s) for r in records]}", flush=True)
        model = fit_base_power(records)
        calibration_output = output / "event_v2"
        model.save(calibration_output / "base_power_candidate.json")
        save_json(calibration_output / "base_validation_report.json", model.report)
        after = {str(path): file_hash(path) for path in paths}
        save_json(calibration_output / "raw_integrity.json", dict(before=before, after=after, unchanged=before == after))
        print(f"Base status={model.status}; schema={model.schema}; event profiles={len(model.event_profiles)}; raw unchanged={before == after}", flush=True)
        for fold in model.report["folds"]:
            print(f"Holdout {Path(fold['held_out']).name}: {fold['transient']}", flush=True)


if __name__ == "__main__":
    main()
