"""Freeze audited inputs and honest prerequisite status without inventing settings."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from spindle_power.load_prediction.common import file_hash,version_of
from spindle_power.load_prediction.feed import MachineConfig
from spindle_power.load_prediction.measurement import detect_encoding,tokens
from spindle_power.load_prediction.experiment_protocol import protocol_folds,PROTOCOL_VERSION
from spindle_power.load_prediction.dataset_manifest import inspect_manifest
from audit_paper1_data import save


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--inventory",type=Path,required=True)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--reserved-nc-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    machine=MachineConfig(name="200C-2",provenance="machine identity author-confirmed; geometry/settings unknown; not training-qualified")
    machine_path=a.output/"machine_200C2_unqualified.json"
    machine.save(machine_path)
    condition=dict(machine={"power_channel":"G432 spindle power W"},tool={"type":"flat_end_mill"},workpiece={},clamping={},cooling={})
    save(a.output/"condition_description.json",dict(schema="afc.phase_condition_evidence.1",condition_descriptors=condition,
        confirmed={"machine_id":"200C-2","diameter_mm":10.,"teeth":3,"tool_type":"flat_end_mill","tool_healthy":True},
        confirmed_source="author statements supplied with approved implementation plan",
        actual_facts={"multiple_physical_tools":True,"individual_tool_sessions":None,"actual_acquisition_order":None},
        analysis_assumptions={"order":"round_number","tool_group":"same_specification_pooled","source":"explicit_author_instruction"},
        unknown=["tool substrate/coating","material grade/hardness","alarm log","feed override","G54","H3","initial/home positions","G05.1 semantics","settings equivalence"],
        single_level_factors="freeze; no independent machine/tool/material effects claimed"))
    inv=json.loads(a.inventory.read_text(encoding="utf-8"))
    records=[]
    process_root=a.data_root/"工艺信息文件"
    for r in inv["records"]:
        feature=r["feature"]
        process={"外形":process_root/"外形/外形_工艺信息文件.txt","曲面":process_root/"曲面.txt","混合":process_root/"混合路径.txt"}.get(feature)
        if process is None:
            level=r["combination"][:2] if feature=="斜面" else r["combination"][2:]
            process=process_root/(feature+level+".txt")
        path=Path(r["measurement"])
        system=None
        with path.open(encoding=detect_encoding(path),newline="") as stream:
            for row in csv.reader(stream):
                row=tokens(row)
                if row and row[0]=="SysInfo": system=row;break
        nc=Path(r["nc"])
        records.append(dict(record_id=r["record_id"],dataset_role="development",status="active",round_id=r["round_id"],
            feature=feature,combination=r["combination"],parameter_group=r["parameter_group"],program_id=feature,
            nc=str(nc.resolve()),measurement=str(path.resolve()),process=str(process.resolve()),machine=str(machine_path.resolve()),
            program_version=version_of(nc.read_text(encoding=detect_encoding(nc))),process_version=file_hash(process),
            machine_id="200C-2",machine_version=version_of(asdict(machine)),diameter_mm=10.,teeth=3,
            tool_healthy=True,no_alarm=None,complete=None,round_complete=None,alignment_reliable=None,process_aligned=None,
            feed_override=None,machining_mode=None,machine_settings_version=None,
            measurement_settings_version=version_of(system),raw_controller_header=system,
            actual_event_order=None,acquired_at=None,tool_session=None,spacing_mm=.5,
            condition_descriptors=condition,numeric_audit=dict(path=str(a.inventory.resolve()),
                inventory_sha256=file_hash(a.inventory),content_sha256=r.get("content_sha256"),samples=r.get("samples")),
            source_facts={"multiple_physical_tools":True,"health":"author_confirmed","no_alarm":"unknown"},
            analysis_mode="numbered_round_pooled_tool",analysis_tool_group="200C2_D10_Z3_pooled",analysis_assumption_source="explicit_author_instruction"))
    # Reserved records receive only registration and hashes. No sample reader,
    # model, scaler, score or memory update is invoked on O1203.
    reserved=[]
    sources=list((a.data_root/"实际加工采样数据/O1203").glob("*.csv"))
    sources += [a.reserved_nc_root/"O1203",a.reserved_nc_root/"O1203工艺信息.txt"]
    for path in sources:
        reserved.append(dict(path=str(path.resolve()),sha256=file_hash(path),bytes=path.stat().st_size,
            dataset_role="future_independent_test",excluded_from=["training","selection","early_stopping","normalization","memory","calibration","prediction","scoring"]))
    save(a.output/"reserved_O1203.json",dict(schema="afc.reserved_sources.1",sources=reserved))
    manifest=dict(schema="afc.dataset_manifest.1",records=records,base_validation={},reserved_registry="reserved_O1203.json")
    save(a.output/"development_manifest.json",manifest)
    save(a.output/"ninefold_protocol.json",dict(protocol_version=PROTOCOL_VERSION,folds=protocol_folds()))
    candidates=[]
    for width in (32,64):
        candidates.append(dict(id=f"C{width}",feed_model=dict(hidden_size=32,num_layers=2,dropout=0.,window_points=256,context_points=64),
            feed_training=dict(epochs=100,learning_rate=.001,patience=10,min_delta=1e-6,seed=42,gradient_clip=1.,batch_size=16),
            load_model=dict(coefficient_widths=[width,width],hidden=width,window_points=256),
            load_training=dict(main_epochs=100,residual_epochs=100,learning_rate=.001,patience=10,min_delta=1e-6,seed=42,gradient_clip=1.,main_batch_points=8192)))
    save(a.output/"preregistered_design.json",dict(status="design_registered_not_launched",candidates=candidates,
        note="This is a preregistered design, not a run log.",
        selection=["predicted_feed_no_memory_equal_weight_RMSE","MAE","high_load_underestimation"],
        full_training="fresh initialization; ceil(median fold best epochs) per stage; no training-set early stopping"))
    print("Registered 450 development records and isolated O1203. Starting actual manifest verification.",flush=True)
    report=inspect_manifest(a.output/"development_manifest.json",verify_files=True)
    save(a.output/"qualification_report.json",report)
    save(a.output/"fixed_source_snapshot.json",dict(snapshot_id=report["snapshot_id"],snapshot=report["snapshot"],
        feed_eligible=report["feed_eligible"],load_eligible=report["load_eligible"],
        note="Fixed source snapshot, not a prepared numerical training snapshot."))
    print(json.dumps(dict(feed_eligible=report["feed_eligible"],load_eligible=report["load_eligible"],
        blocker_counts=dict(Counter(b["code"] for b in report["blockers"]))),ensure_ascii=False),flush=True)


if __name__=="__main__":main()
