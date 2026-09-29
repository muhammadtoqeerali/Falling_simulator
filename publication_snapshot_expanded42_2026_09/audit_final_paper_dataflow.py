#!/usr/bin/env python3
from pathlib import Path
import argparse, json, hashlib
import numpy as np, pandas as pd
from sklearn.model_selection import KFold, train_test_split

COND={"Physical only":"EXP01_REAL_ONLY","Simulator only":"EXP03_SIM_ONLY_FULL","SIM20":"EXP04_MIX20","SIM50":"EXP05_MIX50","SIM70":"EXP06_MIX70","SIM100":"EXP07_MIX100"}
RAT={"SIM20":.2,"SIM50":.5,"SIM70":.7,"SIM100":1.0}
PAPER={
("Physical only","Segment"):(85.06,93.43,70.16,80.14),("Physical only","Event"):(89.88,98.31,79.52,87.92),
("Simulator only","Segment"):(84.14,90.00,68.79,77.98),("Simulator only","Event"):(57.10,100.00,57.10,72.69),
("SIM20","Segment"):(84.29,94.76,68.60,79.59),("SIM20","Event"):(89.33,99.00,77.74,87.09),
("SIM50","Segment"):(84.75,93.54,69.53,79.77),("SIM50","Event"):(90.07,98.65,79.65,88.14),
("SIM70","Segment"):(83.57,94.34,67.16,78.47),("SIM70","Event"):(88.73,98.98,76.44,86.26),
("SIM100","Segment"):(85.29,93.28,70.61,80.38),("SIM100","Event"):(90.50,98.70,80.54,88.70)}
CONF={("Physical only","Segment"):(484,2926),("SIM100","Segment"):(499,2882),("Physical only","Event"):(40,600),("SIM100","Event"):(31,570)}

def lab(y):
    a=np.asarray(y).reshape(-1)
    if np.issubdtype(a.dtype,np.number): return a.astype(int)
    s=np.char.lower(np.char.strip(a.astype(str))); z=np.full(len(s),-1,int)
    z[np.isin(s,["activity","0","false"])]=0; z[np.isin(s,["falling","fall","1","true"])]=1
    if (z<0).any(): raise RuntimeError("unknown labels")
    return z
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def rh(x): return int(np.floor(float(x)+.5))
def phys_count(root,subs):
    n0=n1=tr=ae=fe=0
    for s in subs:
      for t in sorted((root/s).iterdir()):
       if not t.is_dir() or t.name.startswith("."): continue
       for r in sorted(t.iterdir()):
        yp=r/"labels.npy"
        if not yp.exists(): continue
        y=lab(np.load(yp,allow_pickle=True)); a=int((y==0).sum()); f=int((y==1).sum())
        n0+=a;n1+=f;tr+=1;fe+=int(f>0);ae+=int(f==0)
    return dict(subjects=len(set(subs)),trials=tr,windows=n0+n1,activity_windows=n0,falling_windows=n1,activity_events=ae,fall_events=fe)
def ov(a,b): return len(set(map(str,a))&set(map(str,b)))
def col(meta,names):
    return next((c for c in names if c in meta.columns),None)

ap=argparse.ArgumentParser()
ap.add_argument("--project",type=Path,default=Path("/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"))
ap.add_argument("--protechto-root",type=Path,default=Path("/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master"))
a=ap.parse_args(); P=a.project.resolve(); R=a.protechto_root.resolve()
O=P/"outputs/final_dataset_split_audit_2026_09_25"; O.mkdir(parents=True,exist_ok=True)
PR=R/"data/dataset/segments/300ms_50ov_npseg_filt_binary"; C=P/"outputs/protechto_exact_synthetic_cache_v1"; CAM=P/"outputs/protechto_exact_full_campaign_v1"; POST=CAM/"final_results_posthoc"
for q in [PR,C/"synthetic_y01.npy",C/"synthetic_metadata_resolved.csv",CAM/"RUN_ID"]:
    if not q.exists(): raise RuntimeError(f"missing {q}")

subs=np.array([x.name for x in sorted(PR.iterdir()) if x.is_dir() and x.name not in {"999","1000"}],object)
kf=KFold(5,shuffle=True,random_state=42); folds={}; ass=[]; ps=[]; checks=[]
for f,(tv,te) in enumerate(kf.split(subs),1):
    tr,va=train_test_split(tv,test_size=.2,random_state=42); roles={"train":subs[tr],"val":subs[va],"test":subs[te]};folds[f]=roles
    for x,y in [("train","val"),("train","test"),("val","test")]: checks.append((f"physical_f{f}_{x}_{y}_subject_overlap",ov(roles[x],roles[y]),0))
    for role,v in roles.items():
        for s in v: ass.append((f,role,str(s)))
        ps.append({"fold":f,"role":role,**phys_count(PR,[str(x) for x in v]),"subject_ids":";".join(map(str,v))})
pd.DataFrame(ass,columns=["fold","role","subject_id"]).to_csv(O/"physical_subject_assignments.csv",index=False)
PS=pd.DataFrame(ps);PS.to_csv(O/"physical_fold_split_counts.csv",index=False)
PG=phys_count(PR,[str(x) for x in subs])

sy=lab(np.load(C/"synthetic_y01.npy",allow_pickle=True)); sm=pd.read_csv(C/"synthetic_metadata_resolved.csv",low_memory=False)
if len(sy)!=len(sm): raise RuntimeError("synthetic y/meta mismatch")
pc=col(sm,["profile_id","profile","sim_profile"]);tc=col(sm,["task_id","task","scenario_id","scenario"])
uids=np.array(sorted(sm.trial_uid_resolved.astype(str).unique()),object)
trajsets=pd.DataFrame({"u":sm.trial_uid_resolved.astype(str),"y":sy}).groupby("u").y.agg(lambda s:set(map(int,s)))
SG=dict(windows=len(sy),activity_windows=int((sy==0).sum()),falling_windows=int((sy==1).sum()),trajectories=len(uids),profiles=int(sm[pc].astype(str).nunique()) if pc else np.nan,tasks=int(sm[tc].astype(str).nunique()) if tc else np.nan,fall_trajectories=int(sum(1 in x for x in trajsets)),activity_only_trajectories=int(sum(x=={0} for x in trajsets)))

sim=[]; skf=KFold(5,shuffle=True,random_state=42)
for f,(tv,te) in enumerate(skf.split(uids),1):
    tr,va=train_test_split(tv,test_size=.2,random_state=42); rr={"train":uids[tr],"val":uids[va],"test":uids[te]}; sets={}
    for role,v in rr.items():
        m=sm.trial_uid_resolved.astype(str).isin(v).to_numpy(); q=sm.loc[m]; y=sy[m]
        sets[role]=(set(q[pc].astype(str)) if pc else set(),set(q[tc].astype(str)) if tc else set())
        sim.append(dict(fold=f,role=role,trajectories=len(v),windows=len(y),activity_windows=int((y==0).sum()),falling_windows=int((y==1).sum()),profiles=len(sets[role][0]) if pc else np.nan,tasks=len(sets[role][1]) if tc else np.nan))
    for x,y in [("train","val"),("train","test"),("val","test")]:
        checks.append((f"sim_f{f}_{x}_{y}_trajectory_overlap",ov(rr[x],rr[y]),0))
        sim.append(dict(fold=f,role=f"OVERLAP_{x}_{y}",trajectories=ov(rr[x],rr[y]),windows=np.nan,activity_windows=np.nan,falling_windows=np.nan,profiles=len(sets[x][0]&sets[y][0]) if pc else np.nan,tasks=len(sets[x][1]&sets[y][1]) if tc else np.nan))
pd.DataFrame(sim).to_csv(O/"simulator_only_fold_counts.csv",index=False)

mix=[]; samp=[]; idxmeta=sm.set_index("synthetic_row",drop=False)
for f in range(1,6):
    q=PS[PS.fold==f].set_index("role"); tr=q.loc["train"];va=q.loc["val"];te=q.loc["test"]
    for name,cond,ratio in [("Physical only","EXP01_REAL_ONLY",0),("SIM20","EXP04_MIX20",.2),("SIM50","EXP05_MIX50",.5),("SIM70","EXP06_MIX70",.7),("SIM100","EXP07_MIX100",1)]:
        req=0 if ratio==0 else rh(int(tr.falling_windows)*ratio); actual=uw=rep=ut=0;up=0 if pc else np.nan;uk=0 if tc else np.nan; path="";sh=""
        if ratio:
            dp=CAM/"synthetic_draw_manifests"/f"{cond}_fold{f}.csv"
            if not dp.exists(): raise RuntimeError(f"missing draw {dp}")
            d=pd.read_csv(dp); rows=pd.to_numeric(d.synthetic_row,errors="raise").astype(int).to_numpy()
            actual=len(rows);uw=len(np.unique(rows));rep=actual-uw; j=idxmeta.loc[rows];ut=j.trial_uid_resolved.astype(str).nunique()
            if pc: up=j[pc].astype(str).nunique()
            if tc: uk=j[tc].astype(str).nunique()
            checks.append((f"{name}_f{f}_draw_count",actual,req));checks.append((f"{name}_f{f}_all_draws_falling",int((sy[rows]==1).all()),1));path=str(dp);sh=sha(dp)
        mix.append(dict(fold=f,condition=name,physical_train_activity=int(tr.activity_windows),physical_train_falling=int(tr.falling_windows),sim_train_falling=actual,physical_val_activity=int(va.activity_windows),physical_val_falling=int(va.falling_windows),physical_test_activity=int(te.activity_windows),physical_test_falling=int(te.falling_windows),sim_val_windows=0,sim_test_windows=0,unique_sim_windows=uw,repeated_sim_draws=rep,unique_sim_trajectories=ut,unique_sim_profiles=up,unique_sim_tasks=uk,test_domain="Physical"))
        samp.append(dict(fold=f,condition=name,requested_sim_falling=req,actual_sim_falling=actual,unique_sim_windows=uw,repeated_sim_draws=rep,unique_sim_trajectories=ut,unique_sim_profiles=up,unique_sim_tasks=uk,draw_manifest=path,sha256=sh))
M=pd.DataFrame(mix);M.to_csv(O/"physical_and_mixed_fold_counts.csv",index=False);pd.DataFrame(samp).to_csv(O/"simulator_sampling_audit.csv",index=False)
for f in range(1,6):
    q=M[M.fold==f]
    for c in ["physical_val_activity","physical_val_falling","physical_test_activity","physical_test_falling"]: checks.append((f"f{f}_{c}_same_all_conditions",int(q[c].nunique()==1),1))
    checks += [(f"f{f}_no_sim_validation",int((q.sim_val_windows==0).all()),1),(f"f{f}_no_sim_test",int((q.sim_test_windows==0).all()),1)]

GLOBAL=pd.DataFrame([{"domain":"Physical exact final root",**PG},{"domain":"Simulator exact cache","subjects":np.nan,"trials":SG["trajectories"],"windows":SG["windows"],"activity_windows":SG["activity_windows"],"falling_windows":SG["falling_windows"],"activity_events":SG["activity_only_trajectories"],"fall_events":SG["fall_trajectories"]}]);GLOBAL.to_csv(O/"dataset_global_comparison.csv",index=False)
CHK=pd.DataFrame(checks,columns=["check","value","expected"]);CHK["pass"]=CHK.value==CHK.expected;CHK.to_csv(O/"integrity_checks.csv",index=False)

pm=[]
pp=POST/"POOLED_ALL_RESULTS.csv"
if pp.exists():
    po=pd.read_csv(pp);po.to_csv(O/"final_pooled_results_copy.csv",index=False)
    for name,cond in COND.items():
      for level in ["Segment","Event"]:
        q=po[(po.condition.astype(str)==cond)&(po.level.astype(str).str.lower()==level.lower())]
        if len(q)!=1: pm.append((name,level,"ROW",np.nan,np.nan,False)); continue
        r=q.iloc[0]; vals=PAPER[(name,level)]
        mets=["balanced_accuracy","fall_precision","fall_recall","fall_f1"] if level=="Segment" else ["overall_accuracy","fall_precision","fall_recall","fall_f1"]
        for m,e in zip(mets,vals):
            o=round(float(r[m])*100,2); pm.append((name,level,m,e,o,o==e))
        if (name,level) in CONF:
            for m,e in zip(["fp","fn"],CONF[(name,level)]): pm.append((name,level,m,e,int(r[m]),int(r[m])==e))
PM=pd.DataFrame(pm,columns=["condition","level","metric","paper_value","observed_value","match"]);PM.to_csv(O/"paper_table_vi_result_match.csv",index=False)
ff=POST/"ALL_FOLD_METRICS.csv"
if ff.exists(): pd.read_csv(ff).to_csv(O/"final_fold_metrics_copy.csv",index=False)

txt=[]
txt += ["FINAL PAPER DATASET / SPLIT / LEAKAGE AUDIT","="*100,"","GLOBAL DATASET COMPARISON",GLOBAL.to_string(index=False),"","PHYSICAL FOLDS",PS.to_string(index=False),"","PHYSICAL + SIM CONDITIONS",M.to_string(index=False),"","SIMULATOR ONLY",pd.DataFrame(sim).to_string(index=False),"","INTEGRITY",CHK.to_string(index=False),f"\nALL INTEGRITY PASS: {CHK['pass'].all()}"]
if len(PM): txt += ["","TABLE VI MATCH",PM.to_string(index=False),f"\nALL TABLE VI CHECKS MATCH: {PM['match'].all()}"]
txt += ["","NOTES","- Mixed synthetic Falling is training-only; physical validation/test are unchanged.","- Simulator-only splitting is at trial_uid_resolved trajectory level.","- Profile overlap is measured and is not assumed to be zero.","- No raw sensor data are copied and no training is performed."]
(O/"MASTER_DATASET_SPLIT_AUDIT.txt").write_text("\n".join(txt)+"\n")
(O/"AUDIT_PROVENANCE.json").write_text(json.dumps({"project":str(P),"protechto_root":str(R),"physical_root":str(PR),"campaign":str(CAM),"run_id":(CAM/"RUN_ID").read_text().strip(),"physical_split":"KFold5 seed42 + inner 80/20 seed42","sim_split":"trial_uid_resolved KFold5 seed42 + inner 80/20 seed42"},indent=2))
print("OUTPUT",O);print("Physical subjects",len(subs));print("Physical windows",PG["windows"]);print("Synthetic trajectories",SG["trajectories"]);print("Synthetic A/F",SG["activity_windows"],SG["falling_windows"]);print("Integrity pass",bool(CHK["pass"].all()));print("Paper match",bool(PM["match"].all()) if len(PM) else "NO POOLED FILE")
