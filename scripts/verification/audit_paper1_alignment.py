"""Diagnose one global line map/translation per machining record; never qualify it silently."""
from __future__ import annotations
import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from spindle_power.load_prediction.common import file_hash
from spindle_power.load_prediction.feed import MachineConfig,parse_nc
from spindle_power.load_prediction.measurement import read_measurement,select_position_channels,detect_encoding
from audit_paper1_data import save


def projection(segment, xyz):
    if segment.center is None:
        delta=segment.end-segment.start
        u=np.sum((xyz-segment.start)*delta,axis=1)/np.dot(delta,delta)
    else:
        initial=np.arctan2(segment.start[1]-segment.center[1],segment.start[0]-segment.center[0])
        signed=(np.arctan2(xyz[:,1]-segment.center[1],xyz[:,0]-segment.center[0])-initial)*np.sign(segment.sweep)
        signed=np.mod(signed,2*np.pi)
        signed=np.where(signed>2*np.pi-.02,signed-2*np.pi,signed)
        u=signed/abs(segment.sweep)
    expected=segment.points(np.clip(u,0,1))
    return u,np.linalg.norm(xyz-expected,axis=1),xyz-expected


def audit(task):
    rec=read_measurement(task["measurement"])
    source,columns=select_position_channels(rec.values)
    xyz=np.column_stack([rec.values[k] for k in columns])
    raw=rec.values["program_line_raw"]
    # Zero here defines a diagnostic NC coordinate frame, NOT a machine fact.
    config=MachineConfig(initial_position_mm=[0.,0.,0.],work_offsets_mm={"G54":[0.,0.,0.]},
        tool_lengths_mm={"3":0.},home_position_mm=[0.,0.,0.],fixed_rotary_degrees={"A":0.,"C":0.},
        manufacturer_semantics={"G05.1":"contour_tolerance","G04":"seconds"},provenance="diagnostic_assumptions_not_formal_machine_configuration")
    parsed=parse_nc(task["nc"],config)
    segments={s.line:s for s in parsed["segments"] if s.motion in (1,2,3)}
    boundaries=np.flatnonzero(np.r_[True,raw[1:]!=raw[:-1],True])
    visits={}
    for a,b in zip(boundaries[:-1],boundaries[1:]):
        if np.isfinite(raw[a]) and raw[a]==round(raw[a]):
            visits.setdefault(int(raw[a]),[]).append((int(a),int(b)))
    candidates=[]
    for line_offset in (-1,0,1,2):
        pairs=[]
        for line,segment in segments.items():
            ranges=visits.get(line-line_offset,[])
            if len(ranges)!=1: continue
            a,b=ranges[0]
            if b-a<5: continue
            ids=np.unique(np.linspace(a,b-1,min(7,b-a)).astype(int))
            points=xyz[ids]
            if np.isfinite(points).all(): pairs.append((segment,points))
        if len(pairs)<5: continue
        shifts=np.array([points[len(points)//2]-s.points(.5) for s,points in pairs if s.length>=1.])
        if not len(shifts): continue
        initial=np.median(shifts,axis=0)
        # Robust fit is a single translation; it cannot bend individual blocks.
        anchors=[(s,p[1:-1]) for s,p in pairs if s.length>=5. and len(p)>=5]
        if len(anchors)<5: anchors=pairs
        def residual(shift):
            return np.concatenate([projection(s,p-shift)[2].ravel() for s,p in anchors])
        fit=least_squares(residual,initial,loss="soft_l1",f_scale=.1,max_nfev=25)
        errors=np.concatenate([projection(s,p-fit.x)[1] for s,p in pairs])
        candidates.append(dict(line_offset=line_offset,position_offset_mm=fit.x.tolist(),
            sampled_points=len(errors),median_error_mm=float(np.median(errors)),p95_error_mm=float(np.quantile(errors,.95)),
            within_01_fraction=float(np.mean(errors<=.1))))
    if not candidates:
        return dict(record_id=task["record_id"],status="no_alignment_candidate",formal_training_release=False)
    # Median alone can favor the wrong block in a densely segmented curve.
    # The tail exposes those gross line-map mismatches across the whole record.
    best=min(candidates,key=lambda x:(x["p95_error_mm"],x["median_error_mm"]))
    shift=np.array(best["position_offset_mm"])
    errors,covered,missing,backtracks=[],[],[],0
    for line,segment in segments.items():
        ranges=visits.get(line-best["line_offset"],[])
        if len(ranges)!=1:
            missing.append(line)
            continue
        a,b=ranges[0]
        u,error,_=projection(segment,xyz[a:b]-shift)
        errors.append(error[np.isfinite(error)])
        valid=np.isfinite(u)&np.isfinite(error)&(error<=.1)&(u>=0)&(u<=1)
        covered.append(dict(line=line,execution_instances=len(ranges),samples=b-a,within_01_samples=int(valid.sum()),
            segment_mm=segment.length,observed_min_fraction=float(np.nanmin(u)),observed_max_fraction=float(np.nanmax(u)),
            median_error_mm=float(np.nanmedian(error))))
        backtracks+=int(np.sum(np.diff(u)<-1e-6))
    errors=np.concatenate(errors) if errors else np.array([np.inf])
    # NC-declared startup preceding the first G1/G2/G3 move is a geometric idle
    # candidate. Power is inspected only after this selection is fixed.
    lines=Path(task["nc"]).read_text(encoding=detect_encoding(task["nc"])).splitlines()
    starts=[i+1 for i,x in enumerate(lines) if re.search(r"\bM0?3\b",x)]
    first_cut=min(segments)
    idle_mask=(raw+best["line_offset"]>=min(starts))&(raw+best["line_offset"]<first_cut) if starts else np.zeros(len(raw),bool)
    idle_ids=np.flatnonzero(idle_mask)
    speed=rec.values["spindle_speed_rpm"]
    stable=idle_mask&(np.abs(speed-task["spindle_cmd_rpm"])<=.01*task["spindle_cmd_rpm"])
    # Fixed last second of the NC pre-cut interval, no power-based selection.
    if len(idle_ids): stable &= np.arange(len(raw)) >= idle_ids[-1]-999
    model=json.loads(Path(task["base_model"]).read_text(encoding="utf-8"))
    base=float(np.interp(task["spindle_cmd_rpm"],model["speeds_rpm"],model["powers_w"]))
    idle=dict(selection="NC startup to first cutting motion; last 1s at commanded speed +/-1%; power not used for selection",
        nc_first_cut_physical_line=first_cut,samples=int(idle_mask.sum()),stable_tail_samples=int(stable.sum()),
        first_sample=int(idle_ids[0]) if len(idle_ids) else None,last_sample=int(idle_ids[-1]) if len(idle_ids) else None,
        frozen_steady_power_w=base,formal_applicability=False,
        limitation="pre-cut contact, machine settings, and settled initial state require domain review")
    if stable.any():
        p=rec.values["spindle_power_w"][stable]
        idle.update(mean_rpm=float(speed[stable].mean()),mean_power_w=float(p.mean()),median_power_w=float(np.median(p)),
            bias_to_frozen_w=float(p.mean()-base),rmse_to_frozen_w=float(np.sqrt(np.mean((p-base)**2))))
    output=dict(record_id=task["record_id"],round_id=task["round_id"],feature=task["feature"],status="diagnostic_completed",
        measurement_sha256=file_hash(task["measurement"]),nc_sha256=file_hash(task["nc"]),position_source=source,
        coordinate_semantics="canonical_NC_translation_diagnostic_not_verified_machine_frame",parser_issues=parsed["issues"],
        assumptions=["diagnostic G54/H3 origin zero; one inferred global translation","initial/home positions unverified",
            "G05.1 semantics unverified; geometry diagnostic only","A/C zero follows explicit NC; machine configuration pending"],
        candidates=candidates,selected_candidate=best,cutting_segments=len(segments),unobserved_or_revisited_cutting_lines=missing,
        aligned_sample_fraction=float(np.mean(errors<=.1)),median_error_mm=float(np.median(errors)),p95_error_mm=float(np.quantile(errors,.95)),
        backwards_projected_steps=backtracks,segments=covered,idle_candidate=idle,
        formal_training_release=False,quality_updates={"alignment_reliable":None,"complete":None})
    if task.get("trace_dir") and len(idle_ids):
        trace=Path(task["trace_dir"])/(task["record_id"]+"_precut_idle.npz")
        trace.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(trace,source_indices=idle_ids,time_s=rec.time_s[idle_ids],rpm=speed[idle_ids],
            power_w=rec.values["spindle_power_w"][idle_ids],raw_line=raw[idle_ids],xyz=xyz[idle_ids])
        output["idle_candidate"].update(trace_path=str(trace),trace_sha256=file_hash(trace))
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--base-model",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--limit",type=int)
    p.add_argument("--workers",type=int,default=2)
    a=p.parse_args()
    tasks=json.loads(a.plan.read_text(encoding="utf-8"))["records"][:a.limit]
    identity=dict(script_sha256=file_hash(__file__),parser_sha256=file_hash(ROOT/"src/spindle_power/load_prediction/feed.py"),
        plan_sha256=file_hash(a.plan),base_model_sha256=file_hash(a.base_model))
    pending,results=[],[]
    for task in tasks:
        target=a.output/"records"/(task["record_id"]+".json")
        if target.exists():
            r=json.loads(target.read_text(encoding="utf-8"))
            if r.get("identity")==identity: results.append(r);continue
        pending.append({**task,"base_model":str(a.base_model),"trace_dir":str(a.output/"idle_traces")})
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(audit,t):t for t in pending}
        for future in as_completed(futures):
            task=futures[future]
            try: r=future.result()
            except Exception as exc: r=dict(record_id=task["record_id"],status="failed",error=f"{type(exc).__name__}: {exc}")
            r["identity"]=identity
            save(a.output/"records"/(task["record_id"]+".json"),r)
            results.append(r)
            print(json.dumps(dict(completed=len(results),record_id=task["record_id"],status=r["status"],alignment=r.get("aligned_sample_fraction")),ensure_ascii=False),flush=True)
    save(a.output/"alignment_report.json",dict(identity=identity,records=results,formal_training_release=False))


if __name__=="__main__": main()
