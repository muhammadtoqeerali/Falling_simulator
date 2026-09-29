#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, re, shutil, zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

PROJECT = Path('/mnt/hdd16T/ToqeerHomeBackup/mujoco_project')
CORRECTED = PROJECT/'outputs/_highrate_overnight/campaign_highrate_truth_v3_rne_corrected396'
EVENT_MANIFEST = PROJECT/'outputs/phase2_publication_gate_audit_20260826/final_synthetic_event_policy_v2/canonical_synthetic_event_manifest_v2.csv'
DEFAULT_OUT = PROJECT/'outputs/paper_task39_height_fall_evidence_v1'
DEFAULT_ZIP = PROJECT/'outputs/paper_task39_height_fall_evidence_v1.zip'

TASK_ALIASES = ['task_id','task','taskid','activity_id','scenario_id','scenario']
PROFILE_ALIASES = ['profile_id','profile','profileid','subject_id','subject','avatar_id']
TIME_ALIASES = ['time','time_s','timestamp','timestamp_s','t','sim_time','time_sec','seconds']
ONSET_ALIASES = ['fall_onset_time','onset_time','fall_start_time','start_fall_time','t_onset','fall_onset','onset']
IMPACT_ALIASES = ['impact_time','peak_impact_time','recovered_impact_time','t_impact','impact']
REST_ALIASES = ['rest_time','settle_time','settled_time','t_rest']


def norm(s): return re.sub(r'[^a-z0-9]+','',str(s).lower())

def find_col(cols, aliases):
    m={norm(c):c for c in cols}
    for a in aliases:
        if norm(a) in m: return m[norm(a)]
    for c in cols:
        nc=norm(c)
        for a in aliases:
            na=norm(a)
            if len(na)>=5 and (na in nc or nc in na): return c
    return None

def sha256(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def read_csv(p,nrows=None):
    try: return pd.read_csv(p,nrows=nrows,low_memory=False)
    except Exception:
        try: return pd.read_csv(p,nrows=nrows,engine='python')
        except Exception: return None

def path_task_profile(p:Path):
    s=str(p).lower(); task=None; profile=None
    for pat in [r'(?:task|t)[_\- ]*0*(\d{1,3})(?:\D|$)', r'scenario[_\- ]*0*(\d{1,3})(?:\D|$)']:
        m=re.search(pat,s)
        if m:
            try: task=int(m.group(1)); break
            except: pass
    for pat in [r'(?:profile|prof|p)[_\- ]*([a-z0-9]+)(?:\D|$)',r'(?:subject|subj)[_\- ]*([a-z0-9]+)(?:\D|$)']:
        m=re.search(pat,s)
        if m: profile=m.group(1); break
    return task,profile

def header_task_profile(p:Path):
    if p.suffix.lower()!='.csv': return None,None,[]
    d=read_csv(p,5)
    if d is None: return None,None,[]
    tc=find_col(d.columns,TASK_ALIASES); pc=find_col(d.columns,PROFILE_ALIASES)
    task=profile=None
    if tc:
        v=pd.to_numeric(d[tc],errors='coerce').dropna()
        if len(v): task=int(v.iloc[0])
    if pc and len(d[pc].dropna()): profile=str(d[pc].dropna().iloc[0])
    return task,profile,list(d.columns)

def kind(cols,name):
    s=name.lower(); nc=[norm(c) for c in cols]
    if 'validation' in s or 'report' in s: return 'validation'
    if any(('acc' in c or 'gyro' in c or 'imu' in c) for c in nc):
        return 'highrate_truth' if 'truth' in s or any('truth' in c for c in nc) else 'imu_timeseries'
    if any(any(k in c for k in ['pelvis','trunk','grf','contact','impact','force']) for c in nc): return 'physics_timeseries'
    if Path(name).suffix.lower() in {'.npz','.npy','.pkl','.pickle'}: return 'state'
    if Path(name).suffix.lower() in {'.png','.jpg','.jpeg','.pdf','.svg'}: return 'figure'
    return 'other'

def discover(root:Path,task_id:int):
    rows=[]
    keep={'.csv','.json','.npz','.npy','.pkl','.pickle','.png','.jpg','.jpeg','.pdf','.svg'}
    for p in root.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in keep: continue
        pt,pp=path_task_profile(p); ht=hp=None; cols=[]
        if p.suffix.lower()=='.csv' and (pt is None or pp is None or pt==task_id or 'validation' in p.name.lower()):
            ht,hp,cols=header_task_profile(p)
        task=ht if ht is not None else pt; profile=hp or pp
        if task!=task_id:
            if not (p.suffix.lower() in {'.png','.jpg','.jpeg','.pdf','.svg'} and re.search(rf'(?<!\d){task_id}(?!\d)',str(p))):
                continue
        if not cols and p.suffix.lower()=='.csv':
            try: cols=list(pd.read_csv(p,nrows=0).columns)
            except: cols=[]
        rows.append(dict(path=str(p),name=p.name,bytes=p.stat().st_size,task_id=task_id,
                         profile_id='' if profile is None else str(profile),kind=kind(cols,p.name),
                         n_columns=len(cols),columns=' | '.join(cols[:120])))
    return pd.DataFrame(rows)

def load_manifest(p:Path,task_id:int):
    if not p.exists(): return pd.DataFrame()
    d=read_csv(p)
    if d is None or d.empty:return pd.DataFrame()
    tc=find_col(d.columns,TASK_ALIASES)
    if not tc:return pd.DataFrame()
    return d[pd.to_numeric(d[tc],errors='coerce')==task_id].copy()

def candidate_table(idx,manifest):
    prof=set(idx.profile_id.astype(str)) if len(idx) else set(); prof.discard('')
    pc=find_col(manifest.columns,PROFILE_ALIASES) if len(manifest) else None
    if pc: prof.update(manifest[pc].astype(str).unique())
    rows=[]
    for p in sorted(prof):
        f=idx[idx.profile_id.astype(str)==p] if len(idx) else pd.DataFrame()
        kv=f.kind.value_counts().to_dict() if len(f) else {}
        eligible=bool(pc and (manifest[pc].astype(str)==p).any())
        score=(100 if eligible else 0)+8*kv.get('highrate_truth',0)+5*kv.get('imu_timeseries',0)+4*kv.get('physics_timeseries',0)+2*kv.get('state',0)
        rows.append(dict(profile_id=p,eligible_in_event_manifest=eligible,score=score,n_files=len(f),
                         n_highrate_truth=kv.get('highrate_truth',0),n_imu=kv.get('imu_timeseries',0),n_physics=kv.get('physics_timeseries',0),n_state=kv.get('state',0)))
    return pd.DataFrame(rows).sort_values(['score','eligible_in_event_manifest','profile_id'],ascending=[False,False,True]).reset_index(drop=True) if rows else pd.DataFrame()

def best_timeseries(files):
    cand=[]
    for _,r in files.iterrows():
        p=Path(r.path)
        if p.suffix.lower()!='.csv': continue
        try: cols=list(pd.read_csv(p,nrows=0).columns)
        except: continue
        if not find_col(cols,TIME_ALIASES): continue
        richness=sum(any(k in norm(c) for k in ['pelvis','trunk','impact','contact','grf','force','acc','gyro','head']) for c in cols)
        bonus={'highrate_truth':20,'physics_timeseries':12,'imu_timeseries':8}.get(r.kind,0)
        cand.append((bonus+richness+.2*len(cols)+math.log10(max(10,p.stat().st_size)),p))
    return max(cand,key=lambda x:x[0])[1] if cand else None

def numeric_df(p):
    d=read_csv(p); tc=find_col(d.columns,TIME_ALIASES) if d is not None else None
    if d is None or not tc:return None,None
    x=pd.to_numeric(d[tc],errors='coerce'); d=d[x.notna()].copy(); d[tc]=x[x.notna()].astype(float)
    return d.sort_values(tc).drop_duplicates(tc).reset_index(drop=True),tc

def profile_manifest(m,profile):
    if m.empty:return m
    pc=find_col(m.columns,PROFILE_ALIASES)
    return m[m[pc].astype(str)==str(profile)].copy() if pc else m

def event_value(d,aliases):
    if d is None or d.empty:return None,None
    c=find_col(d.columns,aliases)
    if not c:return None,None
    v=pd.to_numeric(d[c],errors='coerce').dropna()
    return (float(v.iloc[0]),c) if len(v) else (None,None)

def pelvis_col(cols):
    c=find_col(cols,['pelvis_height','pelvis_z','pelvis_pos_z','pelvis_position_z'])
    if c:return c
    return next((c for c in cols if 'pelvis' in norm(c) and ('height' in norm(c) or norm(c).endswith('z') or 'posz' in norm(c))),None)

def contact_envelope(d,tc):
    cs=[c for c in d.columns if c!=tc and any(k in norm(c) for k in ['contactforce','grf','groundreaction','normalforce','impactmagnitude','contactload'])]
    arr=[]; used=[]
    for c in cs:
        x=pd.to_numeric(d[c],errors='coerce').abs().to_numpy(float)
        if np.isfinite(x).sum()>10: arr.append(np.nan_to_num(x));used.append(c)
    return (np.max(np.vstack(arr),axis=0),used) if arr else (None,[])

def event_times(manifest,profile,d,tc):
    m=profile_manifest(manifest,profile); sources={}
    onset,oc=event_value(m,ONSET_ALIASES); impact,ic=event_value(m,IMPACT_ALIASES); rest,rc=event_value(m,REST_ALIASES)
    if onset is not None:sources['onset']='event_manifest:'+oc
    if impact is not None:sources['impact']='event_manifest:'+ic
    if rest is not None:sources['rest']='event_manifest:'+rc
    tmin,tmax=float(d[tc].min()),float(d[tc].max())
    if impact is None:
        icols=[c for c in d.columns if 'impact' in norm(c) and c!=tc]
        if onset is not None and icols:
            w=d[(d[tc]>=onset)&(d[tc]<=onset+3)]
            best=None
            for c in icols:
                x=pd.to_numeric(w[c],errors='coerce').abs()
                if x.notna().any():
                    j=x.idxmax(); v=float(x.loc[j]); tt=float(w.loc[j,tc])
                    if best is None or v>best[0]:best=(v,tt,c)
            if best: impact=best[1];sources['impact']='derived_peak:'+best[2]
    if onset is None and impact is not None:
        pc=pelvis_col(d.columns)
        if pc:
            t=d[tc].to_numpy(float);x=pd.to_numeric(d[pc],errors='coerce').to_numpy(float);mask=np.isfinite(x)&(t<impact)&(t>=max(tmin,impact-4))
            if mask.sum()>20:
                tt=t[mask];xx=x[mask];v=np.gradient(xx,tt);thr=np.nanpercentile(v,10);ii=np.where(v<=thr)[0]
                if len(ii):onset=float(tt[ii[0]]);sources['onset']='derived_pelvis_descent:'+pc
    if impact is None: impact=tmin+.65*(tmax-tmin);sources['impact']='fallback_fraction'
    if onset is None: onset=max(tmin,impact-1.2);sources['onset']='fallback_impact_minus_1.2s'
    if rest is None: rest=tmax;sources['rest']='trial_end'
    return float(np.clip(onset,tmin,tmax)),float(np.clip(impact,onset,tmax)),float(np.clip(rest,impact,tmax)),sources

def phases(d,tc,onset,impact,rest):
    t=d[tc].to_numpy(float);tmin,tmax=float(t.min()),float(t.max()); out=[]
    out.append(('SETUP',max(tmin,onset-min(1.5,max(.5,.25*(onset-tmin)))),'pre-onset setup'))
    out.append(('FALL_ONSET',onset,'recorded/derived onset'))
    pc=pelvis_col(d.columns); dt=onset+.55*max(.1,impact-onset);reason='midpoint onset-impact'
    if pc:
        x=pd.to_numeric(d[pc],errors='coerce').to_numpy(float);mask=np.isfinite(x)&(t>=onset)&(t<=impact)
        if mask.sum()>=5:
            tt=t[mask];xx=x[mask];v=np.gradient(xx,tt);j=int(np.argmin(v));dt=float(tt[j]);reason='maximum negative pelvis vertical velocity'
    out.append(('MAX_DESCENT',dt,reason))
    env,used=contact_envelope(d,tc);ct=max(onset,impact-.15);creason='pre-impact fallback'
    if env is not None:
        mask=(t>=onset)&(t<=min(tmax,impact+.5));vals=env[mask];tt=t[mask]
        if len(vals):
            base=np.nanmedian(env[t<onset]) if np.any(t<onset) else 0.; peak=float(np.nanmax(vals));thr=base+.1*max(0,peak-base);ii=np.where(vals>thr)[0]
            if len(ii):ct=float(tt[ii[0]]);creason='first substantial contact/load crossing: '+', '.join(used[:4])
    out.append(('FIRST_CONTACT',ct,creason));out.append(('PEAK_IMPACT',impact,'recorded/derived peak impact'));out.append(('POST_IMPACT',min(tmax,impact+.25),'0.25 s after peak impact'));out.append(('REST',rest,'recorded rest/trial end'))
    clean=[];last=tmin
    for n,tt,r in out:
        tt=max(last,min(float(tt),tmax));clean.append((n,tt,r));last=tt
    return clean

def marker_triplets(cols):
    axes=defaultdict(dict)
    for c in cols:
        m=re.match(r'^(.*?)[_\-. ]([xyzXYZ])$',str(c))
        if m: axes[m.group(1).strip()][m.group(2).lower()]=c
    return {b:a for b,a in axes.items() if all(k in a for k in 'xyz')}

def nearest(d,tc,tt):
    t=d[tc].to_numpy(float);i=int(np.argmin(np.abs(t-tt)));return d.iloc[i]

def export(d,tc,ph,out):
    # exact rows at selected phase times
    rows=[]
    for n,tt,r in ph:
        s=nearest(d,tc,tt); rec={'phase':n,'requested_time_s':tt,'sample_time_s':float(s[tc]),'derivation':r}
        for c in d.columns:
            if c==tc or pd.isna(s[c]):continue
            try:rec[c]=float(s[c])
            except:rec[c]=str(s[c])
        rows.append(rec)
    pd.DataFrame(rows).to_csv(out/'03_PHASES/phase_snapshot_rows.csv',index=False)
    trips=marker_triplets(d.columns); mr=[]
    for n,tt,_ in ph:
        s=nearest(d,tc,tt)
        for m,a in sorted(trips.items()):
            vals=[pd.to_numeric(pd.Series([s[a[k]]]),errors='coerce').iloc[0] for k in 'xyz']
            if all(pd.notna(v) for v in vals):mr.append({'phase':n,'requested_time_s':tt,'sample_time_s':float(s[tc]),'marker':m,'x':float(vals[0]),'y':float(vals[1]),'z':float(vals[2])})
    pd.DataFrame(mr).to_csv(out/'04_MARKER_SKELETON/phase_marker_positions_long.csv',index=False)
    pd.DataFrame({'marker':sorted(trips)}).to_csv(out/'04_MARKER_SKELETON/detected_markers.csv',index=False)
    # figure-ready compact timeseries
    groups={
      'kinematics_timeseries.csv':['pelvis','trunk','torso','com','head','lean','height'],
      'contact_dynamics_timeseries.csv':['contact','grf','force','impact','load','normal','tangent','friction'],
      'imu_timeseries.csv':['acc','gyro','angularvel','angvel','imu']}
    for fn,keys in groups.items():
        cs=[tc]+[c for c in d.columns if c!=tc and any(k in norm(c) for k in keys)];d[cs].to_csv(out/'05_TIMESERIES'/fn,index=False)
    return len(trips)

def previews(d,tc,ph,out):
    if plt is None:return
    pc=pelvis_col(d.columns); env,_=contact_envelope(d,tc);t=d[tc].to_numpy(float)
    series=[]
    if pc:series.append(('Pelvis height',pd.to_numeric(d[pc],errors='coerce').to_numpy(float)))
    if env is not None:series.append(('Contact/load envelope',env))
    if series:
        fig,axs=plt.subplots(len(series),1,figsize=(12,3*len(series)),sharex=True);axs=[axs] if len(series)==1 else axs
        for ax,(lab,y) in zip(axs,series):
            ax.plot(t,y);ax.set_ylabel(lab);ax.grid(alpha=.2)
            for n,tt,_ in ph:ax.axvline(tt,ls='--',lw=.8,alpha=.7)
        axs[-1].set_xlabel('Time [s]');fig.suptitle('Diagnostic only — exact extracted phase timeline');fig.tight_layout();fig.savefig(out/'06_DIAGNOSTIC_PREVIEWS/phase_timeline.png',dpi=180,bbox_inches='tight');plt.close(fig)
    trips=marker_triplets(d.columns)
    if len(trips)>=6:
        fig,axs=plt.subplots(1,len(ph),figsize=(2.7*len(ph),4));axs=[axs] if len(ph)==1 else axs
        for ax,(n,tt,_) in zip(axs,ph):
            s=nearest(d,tc,tt);pts=[]
            for m,a in trips.items():
                v=[pd.to_numeric(pd.Series([s[a[k]]]),errors='coerce').iloc[0] for k in 'xyz']
                if all(pd.notna(x) for x in v):pts.append((v[0],v[2]))
            if pts:
                ar=np.array(pts);ax.scatter(ar[:,0],ar[:,1],s=12)
            ax.set_title(f'{n}\n{tt:.3f}s',fontsize=8);ax.grid(alpha=.2);ax.set_aspect('equal',adjustable='datalim')
        fig.suptitle('Diagnostic marker phases — exact data, no anatomical edges invented');fig.tight_layout();fig.savefig(out/'06_DIAGNOSTIC_PREVIEWS/marker_phase_points.png',dpi=180,bbox_inches='tight');plt.close(fig)

def zipdir(src,zp):
    if zp.exists():zp.unlink()
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(src.rglob('*')):
            if p.is_file():z.write(p,arcname=str(p.relative_to(src.parent)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--task-id',type=int,default=39,help='39 = forward fall from height; 40 = backward fall from height')
    ap.add_argument('--profile-id',default='AUTO')
    ap.add_argument('--top-n',type=int,default=8)
    ap.add_argument('--corrected-root',type=Path,default=CORRECTED)
    ap.add_argument('--event-manifest',type=Path,default=EVENT_MANIFEST)
    ap.add_argument('--output-dir',type=Path,default=DEFAULT_OUT)
    ap.add_argument('--zip-path',type=Path,default=DEFAULT_ZIP)
    a=ap.parse_args();out=a.output_dir
    print('='*110);print('HEIGHT-FALL PAPER EVIDENCE EXTRACTOR');print('='*110);print('Task',a.task_id);print('Root',a.corrected_root)
    if not a.corrected_root.exists():raise SystemExit('ERROR corrected root missing: '+str(a.corrected_root))
    if out.exists():shutil.rmtree(out)
    for s in ['00_AUDIT','01_SELECTION','02_EXACT_SOURCE_FILES','03_PHASES','04_MARKER_SKELETON','05_TIMESERIES','06_DIAGNOSTIC_PREVIEWS','07_EXISTING_RELATED_FIGURES']:(out/s).mkdir(parents=True,exist_ok=True)
    idx=discover(a.corrected_root,a.task_id);idx.to_csv(out/'00_AUDIT/task_file_index.csv',index=False);print('Task files',len(idx))
    man=load_manifest(a.event_manifest,a.task_id)
    if len(man):man.to_csv(out/'00_AUDIT/task_event_manifest_rows.csv',index=False)
    cand=candidate_table(idx,man);cand.to_csv(out/'01_SELECTION/candidate_profiles.csv',index=False)
    if cand.empty:raise SystemExit('ERROR no profiles resolved; upload 00_AUDIT/task_file_index.csv')
    print('\nTOP CANDIDATES\n',cand.head(a.top_n).to_string(index=False))
    if a.profile_id.upper()=='AUTO':
        e=cand[cand.eligible_in_event_manifest==True];sel=e.iloc[0] if len(e) else cand.iloc[0]
    else:
        h=cand[cand.profile_id.astype(str)==str(a.profile_id)]
        if h.empty:raise SystemExit('ERROR profile not found: '+str(a.profile_id))
        sel=h.iloc[0]
    prof=str(sel.profile_id);print('\nSELECTED task/profile:',a.task_id,prof)
    files=idx[idx.profile_id.astype(str)==prof].copy();files.to_csv(out/'01_SELECTION/selected_trial_file_index.csv',index=False)
    (out/'01_SELECTION/SELECTED_TRIAL.txt').write_text(f'task_id={a.task_id}\nprofile_id={prof}\nselection_mode={"AUTO" if a.profile_id.upper()=="AUTO" else "USER"}\n')
    sm=[]
    for _,r in files.iterrows():
        src=Path(r.path);dst=out/'02_EXACT_SOURCE_FILES'/src.name
        if dst.exists():dst=out/'02_EXACT_SOURCE_FILES'/(src.parent.name+'__'+src.name)
        try:shutil.copy2(src,dst);sm.append({'source':str(src),'copy':str(dst),'kind':r.kind,'bytes':src.stat().st_size,'sha256':sha256(src)})
        except Exception as e:sm.append({'source':str(src),'copy':'','kind':r.kind,'bytes':src.stat().st_size,'sha256':'','error':repr(e)})
    pd.DataFrame(sm).to_csv(out/'02_EXACT_SOURCE_FILES/source_manifest_sha256.csv',index=False)
    ts=best_timeseries(files)
    if ts is None:raise SystemExit('ERROR no usable timeseries; upload AUDIT + SELECTION files')
    d,tc=numeric_df(ts)
    if d is None:raise SystemExit('ERROR selected timeseries lacks numeric time')
    shutil.copy2(ts,out/'02_EXACT_SOURCE_FILES/PRIMARY_TIMESERIES_EXACT.csv');(out/'02_EXACT_SOURCE_FILES/PRIMARY_TIMESERIES.txt').write_text(str(ts)+'\n')
    onset,impact,rest,sources=event_times(man,prof,d,tc);ph=phases(d,tc,onset,impact,rest);pdf=pd.DataFrame(ph,columns=['phase','time_s','derivation']);pdf.to_csv(out/'03_PHASES/phase_timestamps.csv',index=False)
    meta={'task_id':a.task_id,'profile_id':prof,'primary_timeseries':str(ts),'time_column':tc,'trial_start_s':float(d[tc].min()),'trial_end_s':float(d[tc].max()),'fall_onset_s':onset,'peak_impact_s':impact,'rest_s':rest,'event_sources':sources,'phases':[{'name':n,'time_s':tt,'derivation':r} for n,tt,r in ph]};(out/'03_PHASES/phase_metadata.json').write_text(json.dumps(meta,indent=2))
    nm=export(d,tc,ph,out);previews(d,tc,ph,out)
    figs=files[files.kind=='figure']
    for _,r in figs.iterrows():
        src=Path(r.path);dst=out/'07_EXISTING_RELATED_FIGURES'/src.name
        if not dst.exists():shutil.copy2(src,dst)
    (out/'00_AUDIT/primary_timeseries_schema.txt').write_text('Primary timeseries: '+str(ts)+'\nRows: '+str(len(d))+'\nColumns: '+str(len(d.columns))+'\nTime column: '+tc+'\n\n'+'\n'.join(map(str,d.columns))+'\n')
    readme=f'''PAPER FIGURE INPUT PACKAGE\nTask: {a.task_id}\nProfile: {prof}\n\nExact phase sequence:\n{pdf.to_string(index=False)}\n\nRecommended final publication figure to create AFTER uploading this ZIP:\n(A) Exact humanoid/skeleton stages: SETUP -> FALL_ONSET -> MAX_DESCENT -> FIRST_CONTACT -> PEAK_IMPACT -> POST_IMPACT -> REST.\n(B) One detailed marker-skeleton projection at loss-of-balance/impact.\n(C) Pelvis height + trunk lean with phase markers.\n(D) Contact/impact dynamics with phase markers.\n(E) Lower-back acceleration/gyroscope around onset and impact.\n\nUse exact files in 02_EXACT_SOURCE_FILES and exported figure inputs in 03-05.\nDiagnostic PNGs are checks only, NOT final paper artwork.\nNO simulator rerun, CNN training, or inference was performed.\n''';(out/'PAPER_FIGURE_README.txt').write_text(readme)
    summary={'gate':'PASS','task_id':a.task_id,'profile_id':prof,'primary_timeseries':str(ts),'primary_rows':len(d),'primary_columns':len(d.columns),'marker_triplets_detected':nm,'no_simulator_rerun':True,'no_training':True,'no_inference':True};(out/'EXTRACTION_SUMMARY.json').write_text(json.dumps(summary,indent=2))
    zipdir(out,a.zip_path);zs=sha256(a.zip_path);Path(str(a.zip_path)+'.sha256').write_text(f'{zs}  {a.zip_path}\n')
    print('\n'+'='*110);print('HEIGHT-FALL PAPER EVIDENCE EXTRACTION: PASS');print('='*110);print('Selected Task/Profile:',a.task_id,prof);print('Markers detected:',nm);print('ZIP:',a.zip_path);print('SHA256:',zs);print('\nUPLOAD:');print(a.zip_path);print(str(a.zip_path)+'.sha256');print('\nNO SIMULATOR RERUN, TRAINING, OR INFERENCE WAS PERFORMED.')

if __name__=='__main__':main()
