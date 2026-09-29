#!/usr/bin/env python3
"""
diagnose_task39_replay_mismatch_v1.py

READ-ONLY diagnostic for the failed Task-39/P020 replay-equivalence gate.

It does NOT rerun MuJoCo.

Inputs already present on the workstation:
  1) final corrected396 P020 Task-39 canonical trial
  2) outputs/task39_p020_exact_pose_replay_v2_headless
  3) current project source tree and patch backups

Goals:
  - distinguish coordinate-frame mismatch from true motion mismatch;
  - quantify canonical vs replay duration/phase differences;
  - test constant-offset and time-shift hypotheses;
  - compare canonical/replay metadata;
  - identify all scenario39 source variants;
  - identify campaign/patch scripts that generated corrected396;
  - produce exact excerpts for the next replay fix.

No scientific gate is relaxed.
"""

from __future__ import annotations
import argparse, hashlib, json, math, os, re, shutil, zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT=Path("/mnt/hdd16T/ToqeerHomeBackup/mujoco_project")
CANON_ROOT=PROJECT/"outputs/_highrate_overnight/campaign_highrate_truth_v3_rne_corrected396"
REPLAY_ROOT=PROJECT/"outputs/task39_p020_exact_pose_replay_v2_headless"
PHASE_ROOT=PROJECT/"outputs/task39_phase_and_replay_evidence_v1"

PROFILE="P020"

def sha256(p:Path, block=1024*1024):
    h=hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b=f.read(block)
            if not b: break
            h.update(b)
    return h.hexdigest()

def read_csv(p):
    return pd.read_csv(p,comment="#",low_memory=False)

def preamble(p):
    d={}
    with p.open("r",encoding="ascii",errors="replace") as f:
        for ln in f:
            if not ln.startswith("#"): break
            s=ln[1:].strip()
            if ":" in s:
                k,v=s.split(":",1); d[k.strip()]=v.strip()
    return d

def canonical_files():
    runs=sorted((CANON_ROOT/"runs"/PROFILE/"task_39").glob("scenario39_*"))
    if len(runs)!=1: raise RuntimeError(f"canonical runs={len(runs)}")
    r=runs[0]
    main=[p for p in r.glob("fall_scenario39_*.csv") if "highrate_truth" not in p.name]
    truth=list(r.glob("*_highrate_truth.csv"))
    return r,main[0],truth[0] if truth else None,r/"run_manifest.json"

def replay_main():
    raw=REPLAY_ROOT/"01_REPLAY_RAW"
    mains=sorted([p for p in raw.glob("fall_forward_height_age49_*.csv")
                  if "highrate_truth" not in p.name and not any(x in p.name for x in [
                      "_markers","_segments","_joints","_dynamics","_contacts"
                  ])])
    if not mains:
        # fallback from replay inventory
        inv=REPLAY_ROOT/"04_AUDIT/replay_export_inventory.json"
        if inv.exists():
            obj=json.loads(inv.read_text())
            p=obj.get("main")
            if p and Path(p).exists(): return Path(p)
    if not mains: raise RuntimeError("replay main CSV not found")
    return mains[-1]

def interp_on_ref(ref,rep,col,shift=0.0):
    tr=pd.to_numeric(ref["timestamp"],errors="coerce").to_numpy(float)
    tp=pd.to_numeric(rep["timestamp"],errors="coerce").to_numpy(float)+shift
    yr=pd.to_numeric(ref[col],errors="coerce").to_numpy(float)
    yp=pd.to_numeric(rep[col],errors="coerce").to_numpy(float)
    g=np.isfinite(tp)&np.isfinite(yp)
    if g.sum()<3: return None
    yi=np.interp(tr,tp[g],yp[g],left=np.nan,right=np.nan)
    m=np.isfinite(yr)&np.isfinite(yi)&(tr>=np.nanmin(tp[g]))&(tr<=np.nanmax(tp[g]))
    if m.sum()<10: return None
    return tr[m],yr[m],yi[m]

def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
def mae(a,b): return float(np.mean(np.abs(a-b)))

def channel_metrics(ref,rep,col,shift=0.0,offset=None):
    x=interp_on_ref(ref,rep,col,shift)
    if x is None: return None
    t,a,b=x
    if offset is None: offset=0.0
    bb=b+offset
    return {
        "n":int(len(a)),
        "rmse":rmse(a,bb),
        "mae":mae(a,bb),
        "median_error":float(np.median(bb-a)),
        "mean_error":float(np.mean(bb-a)),
        "offset_applied":float(offset),
    }

def best_constant_offset(ref,rep,col,shift=0.0):
    x=interp_on_ref(ref,rep,col,shift)
    if x is None: return None
    _,a,b=x
    # least-squares constant offset b + c ~= a
    return float(np.mean(a-b))

def best_shift(ref,rep,cols,lo=-2.0,hi=2.0,step=0.01,offset_correct=True):
    best=None
    for s in np.arange(lo,hi+0.5*step,step):
        vals=[]
        for c in cols:
            if c not in ref.columns or c not in rep.columns: continue
            off=best_constant_offset(ref,rep,c,s) if offset_correct else 0.0
            m=channel_metrics(ref,rep,c,s,off)
            if m: vals.append(m["rmse"]**2)
        if not vals: continue
        score=float(np.sqrt(np.mean(vals)))
        if best is None or score<best["score"]:
            best={"shift_s":float(s),"score":score}
    return best

def initial_stats(df, seconds=1.0):
    t=pd.to_numeric(df["timestamp"],errors="coerce")
    m=t<=float(t.dropna().iloc[0])+seconds
    cols=["pelvis_height","sensor_pos_x","sensor_pos_y","sensor_pos_z"]
    out={}
    for c in cols:
        if c in df:
            out[c]=float(pd.to_numeric(df.loc[m,c],errors="coerce").median())
    return out

def final_stats(df, seconds=1.0):
    t=pd.to_numeric(df["timestamp"],errors="coerce")
    m=t>=float(t.dropna().iloc[-1])-seconds
    cols=["pelvis_height","sensor_pos_x","sensor_pos_y","sensor_pos_z"]
    out={}
    for c in cols:
        if c in df:
            out[c]=float(pd.to_numeric(df.loc[m,c],errors="coerce").median())
    return out

def event_times(df, canonical_phase=None):
    t=pd.to_numeric(df["timestamp"],errors="coerce").to_numpy(float)
    out={"start_s":float(t[0]),"end_s":float(t[-1]),"duration_s":float(t[-1]-t[0])}
    if canonical_phase is not None:
        for _,r in canonical_phase.iterrows():
            out[str(r["phase"])]=float(r["time_s"])
    else:
        # replay: report first fall_detected, peak impact magnitude/force
        if "fall_detected" in df:
            x=pd.to_numeric(df["fall_detected"],errors="coerce").fillna(0).to_numpy()
            ii=np.where(x>0)[0]
            out["first_fall_detected_s"]=float(t[ii[0]]) if len(ii) else None
        for c in ["impact_magnitude","impact_force"]:
            if c in df:
                x=pd.to_numeric(df[c],errors="coerce").to_numpy(float)
                if np.isfinite(x).any():
                    out[f"peak_{c}_s"]=float(t[int(np.nanargmax(x))])
                    out[f"peak_{c}_value"]=float(np.nanmax(x))
    return out

def flatten(obj,prefix=""):
    out={}
    if isinstance(obj,dict):
        for k,v in obj.items():
            kk=f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten(v,kk))
    elif isinstance(obj,list):
        out[prefix]=json.dumps(obj,sort_keys=True)
    else:
        out[prefix]=obj
    return out

def compare_json(a,b):
    fa,fb=flatten(a),flatten(b)
    keys=sorted(set(fa)|set(fb))
    rows=[]
    for k in keys:
        va,vb=fa.get(k,"<MISSING>"),fb.get(k,"<MISSING>")
        if str(va)!=str(vb):
            rows.append({"key":k,"canonical":str(va)[:500],"replay":str(vb)[:500]})
    return pd.DataFrame(rows)

def scan_scenario_versions(outdir):
    rows=[]
    excerpts=[]
    for p in PROJECT.rglob("scenario39*.py"):
        if not p.is_file(): continue
        if any(x in p.parts for x in [".venv",".venv-mjpc","site-packages","outputs"]): continue
        if p.stat().st_size>2_000_000: continue
        try: txt=p.read_text(errors="replace")
        except: continue
        banner=""
        m=re.search(r"Scenario\s*39\s*-\s*Forward Fall From Height\s*\[([^\]]+)\]",txt,re.I)
        if m: banner=m.group(1)
        # capture phase/total-step signals statically
        nums={}
        for key in ["TASK39_STAND_STEPS","TASK39_STEP_STEPS","TASK39_FALL_STEPS","TOTAL_STEPS"]:
            mm=re.search(rf"(?m)^\s*{key}\s*=\s*([^\n#]+)",txt)
            if mm: nums[key]=mm.group(1).strip()
        rows.append({
            "path":str(p.relative_to(PROJECT)),
            "bytes":p.stat().st_size,
            "mtime":p.stat().st_mtime,
            "sha256":sha256(p),
            "banner_version":banner,
            "phase_constants":json.dumps(nums,sort_keys=True),
            "mentions_run_with_subject": "run_with_subject" in txt,
            "mentions_highrate_sidecar": "HighRate" in txt or "high_rate_imu_sidecar" in txt,
        })
        hits=[]
        patterns=[
            r"Scenario\s*39\s*-",r"Subject-adaptive total steps",r"phases\s*=",
            r"TOTAL_STEPS",r"TASK39_.*STEPS",r"def .*phase",r"run_with_subject",
            r"high_rate_imu_sidecar",r"HighRate",r"mj_rnePostConstraint",
        ]
        lines=txt.splitlines()
        hitset=set()
        for pat in patterns:
            for i,line in enumerate(lines):
                if re.search(pat,line,re.I):
                    for j in range(max(0,i-3),min(len(lines),i+5)): hitset.add(j)
        excerpts.append("="*120)
        excerpts.append(str(p.relative_to(PROJECT)))
        excerpts.append("="*120)
        for j in sorted(hitset)[:500]:
            excerpts.append(f"{j+1:06d}: {lines[j]}")
        excerpts.append("")
    d=pd.DataFrame(rows)
    if not d.empty: d=d.sort_values(["mtime","path"],ascending=[False,True])
    d.to_csv(outdir/"scenario39_source_versions.csv",index=False)
    (outdir/"scenario39_source_excerpts.txt").write_text("\n".join(excerpts))
    return d

def scan_campaign_code(outdir):
    keys=[
        "campaign_highrate_truth_v3_rne_corrected396",
        "campaign_highrate_truth_v3_rne",
        "corrected396",
        "rne_corrected",
        "_highrate_overnight",
        "high_rate_imu_sidecar",
        "mj_rnePostConstraint",
    ]
    rows=[]; excerpts=[]
    allowed={".py",".sh",".txt",".md",".json",".yaml",".yml"}
    for p in PROJECT.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in allowed: continue
        if any(x in p.parts for x in [".venv",".venv-mjpc","site-packages"]): continue
        if p.stat().st_size>3_000_000: continue
        try: txt=p.read_text(errors="replace")
        except: continue
        hitkeys={k:txt.count(k) for k in keys if k in txt}
        if not hitkeys: continue
        score=sum(hitkeys.values())
        rows.append({
            "path":str(p.relative_to(PROJECT)),
            "bytes":p.stat().st_size,
            "mtime":p.stat().st_mtime,
            "sha256":sha256(p),
            "score":score,
            "hitkeys":json.dumps(hitkeys,sort_keys=True),
        })
    d=pd.DataFrame(rows)
    if not d.empty: d=d.sort_values(["score","mtime"],ascending=[False,False])
    d.to_csv(outdir/"campaign_code_hits.csv",index=False)

    for _,r in d.head(40).iterrows():
        p=PROJECT/r["path"]
        txt=p.read_text(errors="replace"); lines=txt.splitlines()
        hitset=set()
        for i,line in enumerate(lines):
            if any(k in line for k in keys) or re.search(r"scenario39|task_39|run_with_subject|subprocess|P0\d\d",line,re.I):
                for j in range(max(0,i-4),min(len(lines),i+6)): hitset.add(j)
        excerpts.append("="*120); excerpts.append(str(r["path"])); excerpts.append("="*120)
        for j in sorted(hitset)[:600]:
            excerpts.append(f"{j+1:06d}: {lines[j]}")
        excerpts.append("")
    (outdir/"campaign_code_excerpts.txt").write_text("\n".join(excerpts))
    return d

def zipfolder(folder,zpath):
    if zpath.exists(): zpath.unlink()
    with zipfile.ZipFile(zpath,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file(): z.write(p,arcname=str(p.relative_to(folder.parent)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-dir",type=Path,default=PROJECT/"outputs/task39_replay_mismatch_diagnostic_v1")
    ap.add_argument("--zip-path",type=Path,default=PROJECT/"outputs/task39_replay_mismatch_diagnostic_v1.zip")
    args=ap.parse_args()

    if not REPLAY_ROOT.exists():
        raise SystemExit(f"ERROR: missing replay folder {REPLAY_ROOT}")

    out=args.output_dir
    if out.exists(): shutil.rmtree(out)
    for s in ["00_SIGNAL_DIAG","01_METADATA","02_SOURCE_VERSIONS","03_CAMPAIGN_PROVENANCE"]:
        (out/s).mkdir(parents=True,exist_ok=True)

    crun,cmain,ctruth,cmanifest=canonical_files()
    rmain=replay_main()
    ref=read_csv(cmain); rep=read_csv(rmain)

    phasefile=PHASE_ROOT/"03_PHASES/phase_timestamps.csv"
    phase=pd.read_csv(phasefile) if phasefile.exists() else None

    # Basic identity
    identity={
        "canonical_run_dir":str(crun),
        "canonical_main":str(cmain),
        "canonical_main_sha256":sha256(cmain),
        "replay_main":str(rmain),
        "replay_main_sha256":sha256(rmain),
        "canonical_rows":len(ref),
        "replay_rows":len(rep),
        "canonical_columns":list(ref.columns),
        "replay_columns":list(rep.columns),
        "canonical_preamble":preamble(cmain),
        "replay_preamble":preamble(rmain),
    }
    (out/"00_SIGNAL_DIAG/identity.json").write_text(json.dumps(identity,indent=2))

    # Duration and event timing
    events={
        "canonical":event_times(ref,phase),
        "replay":event_times(rep,None),
    }
    (out/"00_SIGNAL_DIAG/event_timing.json").write_text(json.dumps(events,indent=2))

    # Initial/final coordinate evidence
    init={"canonical":initial_stats(ref),"replay":initial_stats(rep)}
    fin={"canonical":final_stats(ref),"replay":final_stats(rep)}
    (out/"00_SIGNAL_DIAG/initial_final_coordinates.json").write_text(json.dumps({"initial":init,"final":fin},indent=2))

    common=[c for c in [
        "pelvis_height","sensor_pos_x","sensor_pos_y","sensor_pos_z",
        "sensor_vel_x","sensor_vel_y","sensor_vel_z",
        "accel_true_x","accel_true_y","accel_true_z"
    ] if c in ref.columns and c in rep.columns]

    rows=[]
    for c in common:
        raw=channel_metrics(ref,rep,c)
        off=best_constant_offset(ref,rep,c)
        corr=channel_metrics(ref,rep,c,offset=off) if off is not None else None

        # shape after subtracting each stream's own first-second median
        ref0=initial_stats(ref).get(c)
        rep0=initial_stats(rep).get(c)
        shape=None
        if ref0 is not None and rep0 is not None:
            shape=channel_metrics(ref,rep,c,offset=(ref0-rep0))

        rec={"channel":c}
        if raw:
            rec.update({f"raw_{k}":v for k,v in raw.items() if k!="n"})
            rec["n"]=raw["n"]
        rec["best_constant_offset"]=off
        if corr:
            rec.update({f"offset_corrected_{k}":v for k,v in corr.items() if k not in {"n","offset_applied"}})
        if shape:
            rec["initial_aligned_rmse"]=shape["rmse"]
        rows.append(rec)
    pd.DataFrame(rows).to_csv(out/"00_SIGNAL_DIAG/channel_mismatch_metrics.csv",index=False)

    # Search best time shifts, with and without coordinate offsets
    shift_offset=best_shift(ref,rep,[c for c in ["pelvis_height","sensor_pos_x","sensor_pos_y","sensor_pos_z"] if c in common],offset_correct=True)
    shift_raw=best_shift(ref,rep,[c for c in ["pelvis_height","sensor_pos_x","sensor_pos_y","sensor_pos_z"] if c in common],offset_correct=False)
    (out/"00_SIGNAL_DIAG/time_shift_search.json").write_text(json.dumps({
        "best_with_per_channel_constant_offset":shift_offset,
        "best_without_offset":shift_raw
    },indent=2))

    # Canonical vs replay metadata if both manifests exist
    canon_obj=json.loads(cmanifest.read_text()) if cmanifest.exists() else {}
    replay_manifest_candidates=list((REPLAY_ROOT/"01_REPLAY_RAW").glob("run_manifest.json"))
    if replay_manifest_candidates:
        replay_obj=json.loads(replay_manifest_candidates[0].read_text())
        compare_json(canon_obj,replay_obj).to_csv(out/"01_METADATA/canonical_vs_replay_manifest_diff.csv",index=False)
        shutil.copy2(replay_manifest_candidates[0],out/"01_METADATA/replay_run_manifest.json")
    shutil.copy2(cmanifest,out/"01_METADATA/canonical_run_manifest.json")
    shutil.copy2(cmain,out/"01_METADATA/canonical_main.csv")
    shutil.copy2(rmain,out/"01_METADATA/replay_main.csv")

    # Existing V2 gate/audits
    for name in ["REPLAY_EQUIVALENCE.json","source_identity.json","headless_viewer_patch.json","replay_export_inventory.json","render_capture_log.json"]:
        p=REPLAY_ROOT/"04_AUDIT"/name
        if p.exists(): shutil.copy2(p,out/"01_METADATA"/name)

    sv=scan_scenario_versions(out/"02_SOURCE_VERSIONS")
    cc=scan_campaign_code(out/"03_CAMPAIGN_PROVENANCE")

    # Classification
    dur_can=events["canonical"]["duration_s"]; dur_rep=events["replay"]["duration_s"]
    duration_delta=abs(dur_can-dur_rep)

    md=pd.read_csv(out/"00_SIGNAL_DIAG/channel_mismatch_metrics.csv")
    def val(ch,key):
        x=md[md["channel"]==ch]
        if x.empty or key not in x: return None
        v=x.iloc[0][key]
        return None if pd.isna(v) else float(v)

    raw_p=val("pelvis_height","raw_rmse")
    corr_p=val("pelvis_height","offset_corrected_rmse")
    raw_z=val("sensor_pos_z","raw_rmse")
    corr_z=val("sensor_pos_z","offset_corrected_rmse")

    coordinate_mismatch=bool(
        raw_p is not None and corr_p is not None and raw_p>0.2 and corr_p<0.08
    )
    timing_version_mismatch=duration_delta>0.5
    dynamic_mismatch=bool(
        (corr_p is not None and corr_p>0.08) or
        (corr_z is not None and corr_z>0.10)
    )

    classification={
        "duration_canonical_s":dur_can,
        "duration_replay_s":dur_rep,
        "duration_delta_s":duration_delta,
        "raw_pelvis_rmse_m":raw_p,
        "offset_corrected_pelvis_rmse_m":corr_p,
        "raw_sensor_z_rmse_m":raw_z,
        "offset_corrected_sensor_z_rmse_m":corr_z,
        "coordinate_frame_offset_is_major_factor":coordinate_mismatch,
        "timing_or_source_version_mismatch_is_major_factor":timing_version_mismatch,
        "true_dynamic_divergence_remains_after_offset_correction":dynamic_mismatch,
        "best_time_shift_with_offset":shift_offset,
        "conclusion":(
            "SOURCE_OR_RUNTIME_VERSION_MISMATCH"
            if timing_version_mismatch else
            "COORDINATE_DEFINITION_MISMATCH"
            if coordinate_mismatch and not dynamic_mismatch else
            "DYNAMIC_DIVERGENCE_OR_STOCHASTIC_RUNTIME_MISMATCH"
        ),
        "paper_use": "BLOCK_REMAINS",
        "next_action": "Identify the exact campaign runtime/source invocation before any third replay."
    }
    (out/"ROOT_CAUSE_CLASSIFICATION.json").write_text(json.dumps(classification,indent=2))

    summary=[]
    summary.append("TASK 39 REPLAY MISMATCH DIAGNOSTIC")
    summary.append("="*80)
    summary.append(json.dumps(classification,indent=2))
    summary.append("")
    summary.append("Top scenario39 source versions:")
    if not sv.empty:
        summary.append(sv.head(20).to_string(index=False))
    summary.append("")
    summary.append("Top corrected396 campaign provenance files:")
    if not cc.empty:
        summary.append(cc.head(30).to_string(index=False))
    (out/"DIAGNOSTIC_SUMMARY.txt").write_text("\n".join(summary))

    zipfolder(out,args.zip_path)
    zsha=sha256(args.zip_path)
    (args.zip_path.parent/f"{args.zip_path.name}.sha256").write_text(f"{zsha}  {args.zip_path}\n")

    print("="*100)
    print("TASK39 REPLAY MISMATCH DIAGNOSTIC: PASS")
    print("="*100)
    print("Canonical duration [s]:",dur_can)
    print("Replay duration [s]:",dur_rep)
    print("Duration delta [s]:",duration_delta)
    print("Raw pelvis RMSE [m]:",raw_p)
    print("Offset-corrected pelvis RMSE [m]:",corr_p)
    print("Raw sensor-z RMSE [m]:",raw_z)
    print("Offset-corrected sensor-z RMSE [m]:",corr_z)
    print("Best time shift + offsets:",shift_offset)
    print("ROOT CAUSE CLASS:",classification["conclusion"])
    print("Paper gate: BLOCK_REMAINS")
    print("Scenario39 source variants found:",len(sv))
    print("Campaign provenance files found:",len(cc))
    print("ZIP:",args.zip_path)
    print("ZIP SHA256:",zsha)

if __name__=="__main__":
    main()
