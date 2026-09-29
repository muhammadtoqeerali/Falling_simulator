#!/usr/bin/env python3
"""
Read-only reconciliation of the FINAL physical detector dataset.

Purpose:
  1) explain why final Protechto K-fold root has 71 subject identities,
  2) separate KFall / UniVrFall laboratory / UniVrFall worksite records,
  3) reconcile 6,325 event recordings with the event evaluator,
  4) find any fall-task recordings that contain zero Falling segments,
  5) quantify those cases by fold before paper revision.

No training. No raw-signal copying. Reads only folder names + labels.npy + final result CSV.
"""
from pathlib import Path
import argparse, json
import numpy as np, pandas as pd
from sklearn.model_selection import KFold, train_test_split

FALL_TASKS=set(list(range(20,35))+list(range(37,43)))
ACTIVITY_TASKS=set(list(range(1,20))+[35,36,43,44,88])

def source_of_subject(s):
    n=int(s)
    if 100 <= n < 1000:
        return "KFall", n-100, "KFall"
    if n >= 1000:
        original=n-1000
        # UniVrFall v1: 29 lab participants SA09..SA37;
        # 10 real-worksite worker identities are 1..8, 109, 110.
        if 9 <= original <= 37:
            subgroup="UniVrFall_lab"
        elif original in set(range(1,9)) | {109,110}:
            subgroup="UniVrFall_worksite"
        else:
            subgroup="UniVrFall_unresolved"
        return "UniVrFall", original, subgroup
    return "UNRESOLVED", n, "UNRESOLVED"

def labels01(a):
    a=np.asarray(a).reshape(-1)
    if np.issubdtype(a.dtype,np.number):
        z=a.astype(int)
        if not np.all(np.isin(z,[0,1])): raise RuntimeError(f"bad labels {np.unique(z)}")
        return z
    s=np.char.lower(np.char.strip(a.astype(str)))
    z=np.full(len(s),-1,int)
    z[np.isin(s,["activity","0"])]=0
    z[np.isin(s,["falling","fall","1"])]=1
    if (z<0).any(): raise RuntimeError(f"bad labels {np.unique(s[z<0])}")
    return z

ap=argparse.ArgumentParser()
ap.add_argument("--project",type=Path,default=Path("/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"))
ap.add_argument("--protechto-root",type=Path,default=Path("/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master"))
a=ap.parse_args()
P=a.project.resolve(); R=a.protechto_root.resolve()
ROOT=R/"data/dataset/segments/300ms_50ov_npseg_filt_binary"
CAM=P/"outputs/protechto_exact_full_campaign_v1"
POST=CAM/"final_results_posthoc"
OUT=P/"outputs/final_physical_provenance_reconciliation_2026_09_25"
OUT.mkdir(parents=True,exist_ok=True)
if not ROOT.exists(): raise RuntimeError(f"missing {ROOT}")

subjects=np.array([p.name for p in sorted(ROOT.iterdir()) if p.is_dir() and p.name not in {"999","1000"}],dtype=object)
rows=[]
for s in subjects:
    dataset,orig,subgroup=source_of_subject(s)
    sd=ROOT/s
    for td in sorted([x for x in sd.iterdir() if x.is_dir()],key=lambda x:x.name):
        task=int(td.name)
        if task not in FALL_TASKS|ACTIVITY_TASKS:
            event_class="UNRESOLVED_TASK"
        else:
            event_class="Falling" if task in FALL_TASKS else "Activity"
        for rd in sorted([x for x in td.iterdir() if x.is_dir()],key=lambda x:x.name):
            yp=rd/"labels.npy"
            if not yp.exists(): continue
            y=labels01(np.load(yp,allow_pickle=True))
            A=int((y==0).sum()); F=int((y==1).sum())
            segment_presence="Falling" if F>0 else "Activity_only"
            mismatch=""
            if event_class=="Falling" and F==0: mismatch="FALL_TASK_WITH_ZERO_FALLING_WINDOWS"
            elif event_class=="Activity" and F>0: mismatch="ACTIVITY_TASK_WITH_FALLING_WINDOWS"
            rows.append(dict(
                final_subject_id=str(s),source_dataset=dataset,source_subject_id=orig,
                source_subgroup=subgroup,task_id=task,trial_id=rd.name,
                event_class_by_task=event_class,activity_windows=A,falling_windows=F,
                total_windows=A+F,segment_presence=segment_presence,mismatch=mismatch,
                trial_dir=str(rd)
            ))
df=pd.DataFrame(rows)
df.to_csv(OUT/"physical_trial_reconciliation.csv",index=False)

# Subject inventory
subj=pd.DataFrame([dict(final_subject_id=str(s),source_dataset=source_of_subject(s)[0],
                        source_subject_id=source_of_subject(s)[1],source_subgroup=source_of_subject(s)[2])
                   for s in subjects])
subj.to_csv(OUT/"physical_subject_provenance.csv",index=False)

# Source summary
src=[]
for key,g in df.groupby(["source_dataset","source_subgroup"],dropna=False):
    src.append(dict(
        source_dataset=key[0],source_subgroup=key[1],
        subjects=g.final_subject_id.nunique(),recordings=len(g),
        activity_events_by_task=int((g.event_class_by_task=="Activity").sum()),
        fall_events_by_task=int((g.event_class_by_task=="Falling").sum()),
        activity_windows=int(g.activity_windows.sum()),falling_windows=int(g.falling_windows.sum()),
        total_windows=int(g.total_windows.sum()),
        fall_task_zero_falling_windows=int((g.mismatch=="FALL_TASK_WITH_ZERO_FALLING_WINDOWS").sum()),
        activity_task_with_falling_windows=int((g.mismatch=="ACTIVITY_TASK_WITH_FALLING_WINDOWS").sum()),
    ))
SRC=pd.DataFrame(src)
SRC.to_csv(OUT/"physical_source_summary_final_root.csv",index=False)

# Event-vs-segment reconciliation
rec=pd.DataFrame([{
    "total_recordings":len(df),
    "event_activity_by_task":int((df.event_class_by_task=="Activity").sum()),
    "event_falling_by_task":int((df.event_class_by_task=="Falling").sum()),
    "recordings_with_zero_falling_windows":int((df.falling_windows==0).sum()),
    "recordings_with_at_least_one_falling_window":int((df.falling_windows>0).sum()),
    "fall_task_zero_falling_windows":int((df.mismatch=="FALL_TASK_WITH_ZERO_FALLING_WINDOWS").sum()),
    "activity_task_with_falling_windows":int((df.mismatch=="ACTIVITY_TASK_WITH_FALLING_WINDOWS").sum()),
}])
rec.to_csv(OUT/"event_segment_reconciliation.csv",index=False)

# Reconstruct fold role and locate mismatches.
kf=KFold(5,shuffle=True,random_state=42)
foldrows=[]; badfold=[]
for f,(tv,te) in enumerate(kf.split(subjects),1):
    tr,va=train_test_split(tv,test_size=.2,random_state=42)
    roles={"train":subjects[tr],"val":subjects[va],"test":subjects[te]}
    for role,ss in roles.items():
        q=df[df.final_subject_id.isin(map(str,ss))]
        foldrows.append(dict(fold=f,role=role,subjects=len(ss),recordings=len(q),
                             activity_events_by_task=int((q.event_class_by_task=="Activity").sum()),
                             fall_events_by_task=int((q.event_class_by_task=="Falling").sum()),
                             activity_windows=int(q.activity_windows.sum()),falling_windows=int(q.falling_windows.sum()),
                             fall_task_zero_falling_windows=int((q.mismatch=="FALL_TASK_WITH_ZERO_FALLING_WINDOWS").sum())))
        b=q[q.mismatch!=""].copy()
        if len(b):
            b.insert(0,"fold",f);b.insert(1,"role",role);badfold.append(b)
F=pd.DataFrame(foldrows);F.to_csv(OUT/"physical_fold_event_reconciliation.csv",index=False)
BAD=pd.concat(badfold,ignore_index=True) if badfold else pd.DataFrame()
BAD.to_csv(OUT/"mismatched_event_trials_by_fold.csv",index=False)

# Check pooled final results event supports.
pooled_path=POST/"POOLED_ALL_RESULTS.csv"
result_check={}
if pooled_path.exists():
    po=pd.read_csv(pooled_path)
    q=po[(po.condition=="EXP01_REAL_ONLY")&(po.level.str.lower()=="event")]
    if len(q)==1:
        r=q.iloc[0]
        result_check={
            "pooled_event_n":int(r["n"]),
            "pooled_event_support_activity":int(r["support_activity"]),
            "pooled_event_support_fall":int(r["support_fall"]),
            "directory_event_activity_by_task":int((df.event_class_by_task=="Activity").sum()),
            "directory_event_fall_by_task":int((df.event_class_by_task=="Falling").sum()),
        }
        result_check["supports_match"]=(
            result_check["pooled_event_n"]==len(df) and
            result_check["pooled_event_support_activity"]==result_check["directory_event_activity_by_task"] and
            result_check["pooled_event_support_fall"]==result_check["directory_event_fall_by_task"]
        )
pd.DataFrame([result_check]).to_csv(OUT/"pooled_event_support_check.csv",index=False)

# Human record.
lines=[
"FINAL PHYSICAL DATASET PROVENANCE RECONCILIATION","="*100,"",
"SUBJECT PROVENANCE",subj.groupby(["source_dataset","source_subgroup"]).size().rename("subjects").reset_index().to_string(index=False),"",
"SOURCE SUMMARY",SRC.to_string(index=False),"",
"EVENT / SEGMENT RECONCILIATION",rec.to_string(index=False),"",
"FOLD RECONCILIATION",F.to_string(index=False),"",
"POOLED RESULT SUPPORT CHECK",json.dumps(result_check,indent=2),"",
"MISMATCHED RECORDINGS",BAD[["fold","role","final_subject_id","source_dataset","source_subgroup","task_id","trial_id","activity_windows","falling_windows","mismatch"]].to_string(index=False) if len(BAD) else "NONE","",
"INTERPRETATION NOTES",
"- Event ground truth in the final Protechto evaluator is task-ID based.",
"- Segment ground truth is labels.npy based.",
"- Any fall-task recording with zero Falling windows must be explicitly reviewed because event and segment semantics differ.",
"- This script does not alter the dataset or rerun the CNN."
]
(OUT/"MASTER_PHYSICAL_PROVENANCE_RECONCILIATION.txt").write_text("\n".join(lines)+"\n")
print("OUTPUT",OUT)
print("Subjects",len(subjects))
print(subj.groupby(["source_dataset","source_subgroup"]).size().rename("subjects").to_string())
print("Recordings",len(df))
print("Event Activity/Falling by task",int((df.event_class_by_task=="Activity").sum()),int((df.event_class_by_task=="Falling").sum()))
print("Fall-task recordings with zero Falling windows",int((df.mismatch=="FALL_TASK_WITH_ZERO_FALLING_WINDOWS").sum()))
print("Activity-task recordings with Falling windows",int((df.mismatch=="ACTIVITY_TASK_WITH_FALLING_WINDOWS").sum()))
print("Pooled supports match",result_check.get("supports_match","N/A"))
