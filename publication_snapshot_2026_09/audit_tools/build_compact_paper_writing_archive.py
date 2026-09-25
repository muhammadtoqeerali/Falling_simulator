#!/usr/bin/env python3
from pathlib import Path
import argparse, hashlib, json, re, shutil, zipfile
import numpy as np
import pandas as pd

PROJECT=Path('/mnt/hdd16T/ToqeerHomeBackup/mujoco_project')
STAGE=PROJECT/'outputs/paper_evidence_archive_v1'
POST=PROJECT/'outputs/protechto_exact_full_campaign_v1/final_results_posthoc'
PUB=PROJECT/'outputs/phase2_publication_gate_audit_20260826'
EVENT=PUB/'final_synthetic_event_policy_v2'
CORRECTED=PROJECT/'outputs/_highrate_overnight/campaign_highrate_truth_v3_rne_corrected396'
OUT_DEFAULT=PROJECT/'outputs/paper_writing_evidence_COMPACT'
ZIP_DEFAULT=PROJECT/'outputs/paper_writing_evidence_COMPACT.zip'
MAX=25*1024*1024
FIG={'.png','.pdf','.svg'}
SMALL={'.csv','.json','.txt','.md','.tex','.yaml','.yml'}
PRIMARY=['EXP01_REAL_ONLY','EXP03_SIM_ONLY_FULL','EXP04_MIX20','EXP05_MIX50','EXP06_MIX70','EXP07_MIX100']
DISPLAY={'EXP01_REAL_ONLY':'REAL_ONLY','EXP03_SIM_ONLY_FULL':'SIM_ACTIVITY + SIM_FALLING','EXP04_MIX20':'REAL + SIM20','EXP05_MIX50':'REAL + SIM50','EXP06_MIX70':'REAL + SIM70','EXP07_MIX100':'REAL + SIM100'}
RESULT_RE=re.compile(r'(classification|confusion|metric|report|statistics|event|segment|simulation|threshold|summary|precision|recall|f1|accuracy|result)',re.I)
VALID_RE=re.compile(r'(audit|summary|metric|event|impact|validation|profile|task|manifest|coverage|count|distribution|confidence|eligibility|window|recovered)',re.I)
EXCLUDE_RE=re.compile(r'(checkpoint|\.ckpt$|y_true|y_pred|probabilit|prediction_array|synthetic_[xy]\.npy|physical_[xy]\.npy|highrate_truth\.csv$|fall_scenario.*\.csv$)',re.I)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def cp(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)

def profile_meta(path):
    s=str(path)
    def g(p,flags=0):
        m=re.search(p,s,flags); return m
    pm=g(r'/(P\d{3})/'); tm=g(r'/task_(\d+)/'); am=g(r'(?:^|[_/])age(\d{1,3})(?:[_/]|$)'); sm=g(r'(?:^|[_/])sex_(male|female)(?:[_/]|$)',re.I); hm=g(r'(?:^|[_/])h(\d+)p(\d+)(?:[_/]|$)',re.I); wm=g(r'(?:^|[_/])w(\d+)p(\d+)(?:[_/]|$)',re.I)
    dec=lambda m: float(f'{int(m.group(1))}.{m.group(2)}') if m else np.nan
    return {'profile_id':pm.group(1) if pm else '', 'task_id':int(tm.group(1)) if tm else np.nan, 'age_years':int(am.group(1)) if am else np.nan, 'sex':sm.group(1).lower() if sm else '', 'height_m':dec(hm), 'weight_kg':dec(wm)}

def write_method_facts(p):
    p.write_text('''PAPER-WRITING ESSENTIAL METHOD FACTS\n====================================\n\nACCEPTED/FROZEN PIPELINE — DO NOT ALTER WHEN DRAFTING\n\nPHYSICAL PROTECHTO PIPELINE\n---------------------------\nCanonical physical segmented dataset: data/dataset/segments/300ms_50ov_npseg_filt_binary\nSampling: 100 Hz\nWindow: 300 ms = 30 samples\nOverlap: 50%; stride = 15 samples\nFilter: first-order Butterworth low-pass, 5 Hz, fs=100 Hz, scipy filtfilt per 30-sample window and feature\nFeature order: AccX AccY AccZ GyrX GyrY GyrZ EulerX EulerY EulerZ\nCNN uses acceleration + gyroscope; Euler is discarded internally.\nPhysical input units entering IMUNormalizer: acceleration mg; gyroscope mdps.\nNormalizer: acceleration /4000; gyroscope /1000 (mdps->deg/s), then /1800.\nLabels: Activity and Falling. Regions are sliced before windowing, so windows do not cross the class boundary.\nSplit: 5-fold KFold, shuffle=True, random_state=42; inner validation split test_size=0.2, random_state=42.\nTraining: lightweight CNN, batch size 64, AdamW, max 100 epochs, early stopping patience 20; no active augmentation in accepted KFold loader.\nDecision bias: 0.65. Event rule: 2 consecutive predicted Falling windows.\n\nCRITICAL 150-ms FACT\n--------------------\nThere is NO active manual removal of the final 150 ms in the accepted npseg physical pipeline.\nThe old '-15 samples' / '-150 ms' lines in preprocessing/windowing.py are COMMENTED OUT.\nA separate helper creates a different non-npseg dataset and is NOT the accepted pipeline.\nTherefore the final manuscript must NOT claim that the accepted real data removes the final 150 ms.\nThe accepted synthetic representation likewise does NOT apply a manual terminal 150-ms trim.\n\nSYNTHETIC -> PROTECHTO REPRESENTATION\n------------------------------------\nCorrected physics source: outputs/_highrate_overnight/campaign_highrate_truth_v3_rne_corrected396\nCorrected396: 396 simulations, 22 simulated profiles, 18 tasks; ages observed 20-78.\nClassifier synthetic source: outputs/phase2_corrected_event_dataset_v1\nSynthetic windows: total 20,080 = Activity 18,824 + Falling 1,256.\nSynthetic source trial/profile groups represented in classifier metadata: 310.\nAll 310 groups contain pre-onset Activity windows and Falling windows; no independent Activity-only simulated event trajectories exist in this set.\nRepresentation: 100 Hz; 30 samples / 300 ms; stride 15; regions windowed separately; same first-order 5-Hz Butterworth + filtfilt per window.\nSynthetic conversion: m/s^2 -> mg by dividing by 0.00980665; deg/s -> mdps by multiplying by 1000.\n9-column Protechto compatibility is preserved; Euler is later discarded by the CNN.\nSynthetic split/addition is by source trial/profile groups, never individual windows.\n\nPRIMARY PAPER CONDITIONS\n------------------------\nUse: REAL_ONLY; SIM_ACTIVITY + SIM_FALLING (synthetic-domain diagnostic); REAL + SIM20; REAL + SIM50; REAL + SIM70; REAL + SIM100.\nDo not use REAL_ACTIVITY + SIM_FALLING as a primary manuscript table row.\n\nEVENT-LEVEL REPORTING\n---------------------\nPhysical event table: REAL_ONLY, MIX20, MIX50, MIX70, MIX100; report Fall Precision, Fall Recall/Detection, Fall F1.\nSIM-only event diagnostic: 177/310 detected = 57.10% sensitivity; 133 missed.\nDo not present the synthetic-only event set as a normal balanced binary event test because it has no independent Activity-only event trajectories.\n\nCORRECTED396 VALIDATION\n-----------------------\n396 reports; 22 profiles; 18 tasks. Confidence mean 90.10%; median 93.20%; range 68.30%-98.30%. HIGH_CONFIDENCE 352; MODERATE_CONFIDENCE 44.\nPass rates: head_velocity_safe 100.00%; duration_realistic 89.65%; jerk_realistic 43.94%; protective_response 89.39%; imu_quality 100.00%; contact_pattern 86.87%.\nThe 43.94% jerk-realism pass rate is a real limitation and must not be hidden.\n\nFIGURE INTERPRETATION\n---------------------\nFinal acceleration/gyroscope truth figures compare simulator sensor-model output against internal physics-truth output. They are not physical-real vs synthetic waveform comparisons.\nDownstream real-world usefulness is assessed with held-out physical Protechto evaluation.\n\nPHYSICAL SUBJECT AGES\n---------------------\nNo reliable physical subject-age records were recovered in the paper evidence archive. Do not attribute simulated profile ages to KFall/UniVR participants.\n''',encoding='utf-8')

def write_synth_details(p):
    p.write_text('''SYNTHETIC DATASET / SENSOR / LABEL DETAILS\n========================================\n\nCorrected standard simulator schema (396/396 files):\ntimestamp; accel_x/y/z; accel_raw_x/y/z; gyro_x/y/z; sensor_pos_x/y/z; sensor_vel_x/y/z; pelvis_height; impact_force; impact_magnitude; jerk_mag; fall_detected; accel_true_x/y/z; sensor_error_mag; soft_tissue_artifact; sensor_confidence.\n\nHigh-rate physics-truth schema (396/396 files):\ntimestamp; accel_true_x/y/z; gyro_true_x/y/z; accel_true_mag; gyro_true_mag; sensor_pos_x/y/z; sensor_vel_x/y/z; pelvis_height.\n\nClassifier-ready synthetic representation:\n100 Hz; 30 samples/300 ms; 50% overlap/15-sample stride; 9 columns AccX AccY AccZ GyrX GyrY GyrZ EulerX EulerY EulerZ.\nUnits entering Protechto: acceleration mg; gyroscope mdps. Conversion: acc_mg = acc_ms2 / 0.00980665; gyro_mdps = gyro_deg_s * 1000.\nFilter: first-order 5-Hz Butterworth + filtfilt per window.\nLabels: pre-onset Activity; accepted onset-to-impact/event-region Falling. Activity and Falling are windowed separately.\nNo active terminal 150-ms trim is used in the accepted physical or synthetic pipeline.\nCounts: Activity 18,824; Falling 1,256; total 20,080.\nEvent limitation: all 310 simulated source trajectories are fall trajectories at event level; SIM-only event sensitivity = 177/310 = 57.10%.\n''',encoding='utf-8')

def profile_details(dst):
    rows=[]
    for p in sorted((CORRECTED/'runs').rglob('*.csv')):
        if p.name.endswith('_highrate_truth.csv'): continue
        m=profile_meta(p)
        if m['profile_id']: rows.append(m)
    df=pd.DataFrame(rows).drop_duplicates(['profile_id','age_years','sex','height_m','weight_kg']).sort_values('profile_id')
    df.to_csv(dst,index=False); return df

def task_metadata(dst):
    candidates=[EVENT/'task_summary_v2.csv',EVENT/'canonical_synthetic_event_manifest_v2.csv',PUB/'rne_corrected396_event_window_audit/task_event_window_summary.csv',PUB/'rne_corrected396_recovered_impact_audit/recovered_impact_by_task.csv']
    frames=[]
    for p in candidates:
        if not p.exists(): continue
        try: d=pd.read_csv(p,comment='#',low_memory=False)
        except Exception: continue
        tcs=[c for c in d.columns if str(c).lower() in {'task','task_id'}]
        if not tcs: continue
        d=d.rename(columns={tcs[0]:'task_id'}); d['task_id']=pd.to_numeric(d['task_id'],errors='coerce'); d=d[d.task_id.notna()].copy(); d['task_id']=d.task_id.astype(int)
        keep=['task_id']+[c for c in d.columns if c!='task_id' and any(k in str(c).lower() for k in ('name','label','scenario','fall','direction','description','type','event','impact','onset','count','window','eligible'))]
        out=[]
        for task,g in d[keep].groupby('task_id'):
            row={'task_id':task}
            for c in keep[1:]:
                vals=[x for x in g[c].dropna().astype(str).unique() if x!='']
                row[c]=' | '.join(vals[:5]) if len(vals)<=5 else f'{len(vals)} unique values'
            out.append(row)
        frames.append(pd.DataFrame(out))
    if not frames:
        df=pd.DataFrame(columns=['task_id']); df.to_csv(dst,index=False); return df
    out=frames[0]
    for i,f in enumerate(frames[1:],2):
        overlap=[c for c in f.columns if c!='task_id' and c in out.columns]; f=f.rename(columns={c:f'{c}_source{i}' for c in overlap}); out=out.merge(f,on='task_id',how='outer')
    out=out.sort_values('task_id'); out.to_csv(dst,index=False); return out

def copy_tree(src,dst,keyword=None):
    n=0
    if not src.exists(): return n
    for p in src.rglob('*'):
        if not p.is_file() or p.stat().st_size>MAX: continue
        if p.suffix.lower() not in FIG|SMALL: continue
        if EXCLUDE_RE.search(p.name): continue
        if keyword and not keyword.search(p.name): continue
        cp(p,dst/p.relative_to(src)); n+=1
    return n

def zipdir(root,zpath):
    if zpath.exists(): zpath.unlink()
    with zipfile.ZipFile(zpath,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9,allowZip64=True) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file(): z.write(p,arcname=str(Path(root.name)/p.relative_to(root)))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',default=str(OUT_DEFAULT)); ap.add_argument('--zip',dest='zip_path',default=str(ZIP_DEFAULT)); a=ap.parse_args(); out=Path(a.output_dir); zpath=Path(a.zip_path)
    print('='*110); print('BUILD COMPACT PAPER-WRITING EVIDENCE ARCHIVE'); print('='*110)
    if not STAGE.exists(): raise RuntimeError(f'Missing {STAGE}')
    if not (STAGE/'14_FINAL_CANONICAL_PUBLICATION_FIGURES').exists(): raise RuntimeError('Missing final canonical publication figures')
    if out.exists(): shutil.rmtree(out)
    facts=out/'00_ESSENTIAL_METHOD_AND_DATA_DETAILS'; facts.mkdir(parents=True)
    write_method_facts(facts/'ESSENTIAL_METHOD_FACTS.txt'); write_synth_details(facts/'SYNTHETIC_DATASET_SENSOR_LABEL_DETAILS.txt')
    profiles=profile_details(facts/'SIMULATED_PROFILE_DETAILS_CORRECTED396.csv'); tasks=task_metadata(facts/'TASK_SCENARIO_METADATA_FROM_CANONICAL_FILES.csv')
    for name in ['canonical_synthetic_event_manifest_v2.csv','canonical_synthetic_event_facts_v2.json','profile_summary_v2.csv','task_summary_v2.csv']:
        p=EVENT/name
        if p.exists(): cp(p,facts/name)
    # final paper tables/plots and canonical figures
    for rel in ['04_PAPER_TABLES','05_FIGURES_AND_PLOT_DATA','14_FINAL_CANONICAL_PUBLICATION_FIGURES']:
        copy_tree(STAGE/rel,out/'01_RESULTS_AND_FIGURES'/rel)
    # posthoc summaries
    if POST.exists():
        for p in POST.iterdir():
            if p.is_file() and p.stat().st_size<=MAX and p.suffix.lower() in {'.csv','.txt','.md','.json'}: cp(p,out/'01_RESULTS_AND_FIGURES/final_posthoc_summaries'/p.name)
    # original small classifier reports/confusion matrices for primary conditions
    srcroot=STAGE/'03_CAMPAIGN_RESULTS/conditions'; idx=[]
    for cond in PRIMARY:
        src=srcroot/cond
        if not src.exists(): continue
        for p in src.rglob('*'):
            if not p.is_file() or p.stat().st_size>MAX or p.suffix.lower() not in FIG|SMALL or EXCLUDE_RE.search(p.name) or not RESULT_RE.search(p.name): continue
            dst=out/'02_ORIGINAL_CLASSIFIER_REPORTS'/cond/p.relative_to(src); cp(p,dst); idx.append({'condition':DISPLAY[cond],'archive_path':str(dst.relative_to(out)),'original_path':str(p),'size_bytes':p.stat().st_size,'sha256':sha(p)})
    pd.DataFrame(idx).to_csv(out/'02_ORIGINAL_CLASSIFIER_REPORTS/ORIGINAL_REPORT_INDEX.csv',index=False)
    # canonical validation summaries only
    compact=STAGE/'13_CANONICAL_PAPER_EVIDENCE_V3/compact_canonical_audit'; copy_tree(compact,out/'03_SIMULATOR_VALIDATION/compact_canonical_audit')
    vidx=[]
    for p in PUB.rglob('*'):
        if not p.is_file() or p.stat().st_size>MAX or p.suffix.lower() not in SMALL or not VALID_RE.search(p.name): continue
        if 'stage1_raw_data_audit' in p.parts and p.stat().st_size>2*1024*1024: continue
        dst=out/'03_SIMULATOR_VALIDATION/publication_gate_aggregate'/p.relative_to(PUB); cp(p,dst); vidx.append({'archive_path':str(dst.relative_to(out)),'original_path':str(p),'size_bytes':p.stat().st_size,'sha256':sha(p)})
    pd.DataFrame(vidx).to_csv(out/'03_SIMULATOR_VALIDATION/VALIDATION_DATA_INDEX.csv',index=False)
    # small provenance
    prov=out/'04_PROVENANCE_AND_AUDIT'
    for p in [STAGE/'00_README_AND_MANIFESTS/REPORTING_POLICY.md',STAGE/'06_VALIDATION_AND_AUDITS/protechto_clean_lightweight_pipeline_audit.txt',STAGE/'06_VALIDATION_AND_AUDITS/POSTHOC_AGGREGATION_RECOVERY.md',STAGE/'07_ENVIRONMENT_AND_REPRODUCIBILITY/CAMPAIGN_AUDIT.json',STAGE/'12_CLAIM_TO_EVIDENCE_MAP/CLAIM_TO_EVIDENCE_MAP.csv',STAGE/'13_CANONICAL_PAPER_EVIDENCE_V3/LEGACY_OR_SUPERSEDED_DO_NOT_USE_PRIMARY.csv']:
        if p.exists() and p.stat().st_size<=MAX: cp(p,prov/p.name)
    (out/'README_COMPACT_ARCHIVE.txt').write_text('''COMPACT MANUSCRIPT-WRITING ARCHIVE\n==================================\nKeeps final tables, figures, classifier reports/confusion matrices, simulator validation summaries, canonical event metadata, profile/task details, units, labels, windowing/filtering facts, and audit notes.\n\nIntentionally excludes source code, checkpoints, raw physical datasets, raw 396 simulator trial CSVs, high-rate truth CSVs, NPY tensors, large per-window y_true/y_pred/probability arrays, and broad legacy galleries.\n\nThe larger ~2.6GB paper_evidence_archive_FINAL.zip is a reproducibility backup, not the manuscript-writing package.\n''',encoding='utf-8')
    rows=[]
    for p in sorted(out.rglob('*')):
        if p.is_file(): rows.append({'path':str(p.relative_to(out)),'size_bytes':p.stat().st_size,'sha256':sha(p)})
    man=pd.DataFrame(rows); man.to_csv(out/'FILE_MANIFEST_SHA256.csv',index=False)
    summary={'files':len(man),'uncompressed_bytes':int(man.size_bytes.sum()) if len(man) else 0,'simulated_profiles':len(profiles),'task_rows':len(tasks),'code_included':False,'checkpoints_included':False,'raw_trial_csv_included':False,'raw_numpy_arrays_included':False,'final_150ms_trim_active':False,'final_150ms_note':'No active terminal 150-ms trim in accepted physical npseg or synthetic pipeline.'}
    (out/'ARCHIVE_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('Files staged:',len(man)); print('Uncompressed bytes:',summary['uncompressed_bytes']); print('Profiles:',len(profiles)); print('Task metadata rows:',len(tasks)); print('Code/checkpoints/raw trials: NO')
    zipdir(out,zpath); h=sha(zpath); side=Path(str(zpath)+'.sha256'); side.write_text(f'{h}  {zpath.name}\n',encoding='utf-8')
    print('ZIP:',zpath); print('ZIP bytes:',zpath.stat().st_size); print('ZIP SHA256:',h); print('SHA sidecar:',side); print('='*110); print('COMPACT PAPER-WRITING ARCHIVE GATE: PASS'); print('='*110)
if __name__=='__main__': main()
