#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, re, shutil, subprocess, sys
from pathlib import Path
import pandas as pd

EXCLUDE = {".git",".venv","venv","__pycache__",".idea",".vscode","node_modules"}
TEXT = {".py",".txt",".md",".json",".jsonl",".yaml",".yml",".csv",".tsv",".log",".toml",".ini",".cfg"}
COPY = TEXT | {".npy",".npz",".pkl",".pickle"}
PATH_HINTS = [
    "fold","split","sample","sampling","sim20","sim50","sim70","sim100",
    "cnn","eval","evaluation","result","metric","confusion","trajectory",
    "window","dataset","manifest","train","valid","test","physical",
    "simulator","kfall","univr"
]
KEYWORDS = [
    "sim20","sim50","sim70","sim100","simulator_only","simulator only",
    "physical_only","physical only","kfold","groupkfold","fold","train",
    "validation","test","subject","participant","trajectory","profile","task",
    "sampling_list","sampling list","replacement","replace=true","replace = true",
    "falling","activity","window","balanced accuracy","precision","recall","f1",
    "confusion","event","random_state=42","random_state = 42","early stopping"
]
ALIASES = {
    "fold":["fold","fold_id","cv_fold","outer_fold"],
    "condition":["condition","experiment","mode","training_condition","train_condition","variant"],
    "split":["split","subset","partition","set","dataset_split"],
    "label":["label","class","target","y","class_name"],
    "domain":["domain","source","data_source","dataset","dataset_name"],
    "subject":["subject","subject_id","participant","participant_id","person_id"],
    "trajectory":["trajectory","trajectory_id","source_trajectory","source_id","group_id","profile_task","sim_group"],
    "profile":["profile","profile_id","sim_profile"],
    "task":["task","task_id","scenario","scenario_id","fall_task"],
}
MAX_COPY = 25*1024*1024

def norm(x):
    return re.sub(r"[^a-z0-9]+","_",str(x).strip().lower()).strip("_")

def rel(p,root):
    try: return p.resolve().relative_to(root.resolve())
    except Exception: return Path(p.name)

def relevant(p):
    s=str(p).lower()
    return any(k in s for k in PATH_HINTS)

def files(root):
    for dp,dns,fns in os.walk(root):
        dns[:] = [d for d in dns if d not in EXCLUDE]
        for fn in fns:
            p=Path(dp)/fn
            if any(part in EXCLUDE for part in p.parts): continue
            yield p

def find_col(cols,names):
    m={norm(c):c for c in cols}
    for n in names:
        if norm(n) in m: return m[norm(n)]
    return None

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def git(root,cmd):
    try:
        r=subprocess.run(cmd,cwd=root,capture_output=True,text=True,check=False)
        return (r.stdout or r.stderr).strip()
    except Exception as e:
        return repr(e)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=Path.cwd())
    ap.add_argument("--out",type=Path,default=None)
    a=ap.parse_args()
    root=a.root.resolve()
    out=(a.out.resolve() if a.out else root/"outputs"/"paper_dataset_split_audit_stage1")
    inv=out/"inventory"; art=out/"artifacts"; scripts=out/"scripts"
    for d in [inv,art,scripts]: d.mkdir(parents=True,exist_ok=True)

    rows=[]
    cand=[]
    for p in files(root):
        try: st=p.stat()
        except: continue
        r={"relative_path":str(rel(p,root)),"absolute_path":str(p),"size_bytes":st.st_size,
           "suffix":p.suffix.lower(),"relevant_path_hint":relevant(p)}
        rows.append(r)
        if r["relevant_path_hint"] or p.suffix.lower()==".py": cand.append(r)
    pd.DataFrame(rows).to_csv(inv/"FILE_INVENTORY.csv",index=False)
    pd.DataFrame(cand).to_csv(inv/"CANDIDATE_FILES.csv",index=False)

    gs={
        "branch":git(root,["git","branch","--show-current"]),
        "commit":git(root,["git","rev-parse","HEAD"]),
        "last_commit":git(root,["git","log","-1","--oneline"]),
        "status":git(root,["git","status","--short"]),
        "remote":git(root,["git","remote","-v"]),
    }
    (inv/"GIT_SNAPSHOT.json").write_text(json.dumps(gs,indent=2),encoding="utf-8")
    (inv/"ENVIRONMENT.json").write_text(json.dumps({
        "python":sys.version,"executable":sys.executable,"pandas":pd.__version__},indent=2),encoding="utf-8")

    excerpts=[]
    csvinfo=[]
    jsoninfo=[]
    copied=[]

    for r in cand:
        p=Path(r["absolute_path"]); suf=p.suffix.lower(); size=r["size_bytes"]

        if suf in TEXT and size <= 15*1024*1024:
            try: txt=p.read_text(encoding="utf-8",errors="replace")
            except: txt=""
            hits=[k for k in KEYWORDS if k in txt.lower()]
            if hits:
                excerpts += ["\n"+"="*100,f"FILE: {rel(p,root)}",f"HITS: {', '.join(hits)}","="*100]
                lines=txt.splitlines()
                seen=set()
                for i,line in enumerate(lines):
                    if any(k in line.lower() for k in KEYWORDS):
                        s=max(0,i-3); e=min(len(lines),i+4)
                        if (s,e) in seen: continue
                        seen.add((s,e))
                        excerpts.append(f"\n--- lines {s+1}-{e} ---")
                        excerpts += [f"{j+1:06d}: {lines[j]}" for j in range(s,e)]
                if suf==".py":
                    dst=scripts/rel(p,root); dst.parent.mkdir(parents=True,exist_ok=True)
                    try: shutil.copy2(p,dst); copied.append(dst)
                    except: pass

        if suf in {".csv",".tsv"} and size <= 2*1024*1024*1024:
            info={"relative_path":str(rel(p,root)),"size_bytes":size}
            try:
                header=pd.read_csv(p,nrows=0,comment="#")
                cols=list(header.columns)
                info["columns"]=cols
                roles={}
                for role,names in ALIASES.items():
                    c=find_col(cols,names)
                    if c: roles[role]=c
                info["detected_roles"]=roles
                if size <= 200*1024*1024:
                    df=pd.read_csv(p,comment="#",low_memory=False)
                    info["rows"]=len(df)
                    info["distinct_counts"]={role:int(df[col].nunique(dropna=True)) for role,col in roles.items()}
                    gcols=[roles[x] for x in ["fold","condition","split","domain","label"] if x in roles]
                    if gcols:
                        gp=df.groupby(gcols,dropna=False).size().reset_index(name="row_count")
                        info["grouped_preview"]=gp.head(300).to_dict("records")
                else:
                    info["rows"]="not_loaded_large_file"
            except Exception as e:
                info["error"]=repr(e)
            csvinfo.append(info)

        if suf==".json" and size<=MAX_COPY:
            ji={"relative_path":str(rel(p,root))}
            try:
                x=json.loads(p.read_text(encoding="utf-8"))
                ji["type"]=type(x).__name__
                if isinstance(x,dict): ji["keys"]=list(x.keys())[:200]
                elif isinstance(x,list):
                    ji["length"]=len(x)
                    if x and isinstance(x[0],dict): ji["first_item_keys"]=list(x[0].keys())[:200]
            except Exception as e: ji["error"]=repr(e)
            jsoninfo.append(ji)

        if suf in COPY and size<=MAX_COPY and relevant(p) and suf!=".py":
            dst=art/rel(p,root); dst.parent.mkdir(parents=True,exist_ok=True)
            try: shutil.copy2(p,dst); copied.append(dst)
            except: pass

    (inv/"SCRIPT_AND_TEXT_KEYWORD_EXCERPTS.txt").write_text("\n".join(excerpts),encoding="utf-8")
    (inv/"CSV_SCHEMA_AND_COUNTS.json").write_text(json.dumps(csvinfo,indent=2,default=str),encoding="utf-8")
    (inv/"JSON_STRUCTURE_REPORT.json").write_text(json.dumps(jsoninfo,indent=2,default=str),encoding="utf-8")

    human=[]
    for x in csvinfo:
        human += ["\n"+"="*100,f"FILE: {x.get('relative_path')}",
                  f"SIZE: {x.get('size_bytes')}",f"ROWS: {x.get('rows')}",
                  f"COLUMNS: {x.get('columns')}",f"ROLES: {x.get('detected_roles')}",
                  f"DISTINCT: {x.get('distinct_counts')}"]
        if x.get("grouped_preview"):
            human.append("GROUPED PREVIEW:")
            human += ["  "+json.dumps(y,default=str) for y in x["grouped_preview"]]
        if x.get("error"): human.append("ERROR: "+x["error"])
    (inv/"CSV_SCHEMA_AND_COUNTS.txt").write_text("\n".join(human),encoding="utf-8")

    req = "\n".join([
        "FINAL REQUIRED RECORD",
        "=====================",
        "",
        "For each physical/mixed outer fold and each condition Physical-only/SIM20/SIM50/SIM70/SIM100:",
        "  physical train/validation/test subject IDs and counts",
        "  physical train Activity/Falling windows",
        "  physical validation Activity/Falling windows",
        "  physical test Activity/Falling windows",
        "  physical train/validation/test recording-event counts",
        "  requested and actual simulator Falling additions",
        "  unique simulated windows, repeated simulator draws",
        "  unique simulator trajectories/profiles/tasks",
        "  prove zero simulator examples in physical validation/test",
        "  prove physical validation and test are identical across mixed conditions",
        "  prove zero physical subject leakage across train/validation/test",
        "",
        "For each simulator-only fold:",
        "  train/validation/test trajectory counts",
        "  train/validation/test Activity/Falling windows",
        "  unique profile/task counts per split",
        "  trajectory overlap train-val/train-test/val-test MUST be zero",
        "  profile overlap must be measured, not assumed zero",
        "  task overlap must be measured",
        "",
        "For each fixed simulator sampling list:",
        "  artifact path + checksum, fold, condition, target length, actual length",
        "  unique windows, duplicate draws, unique trajectories/profiles/tasks",
        "  whether replacement occurred",
        "",
        "Connect each fold/condition to outcomes:",
        "  best epoch, epochs, validation loss",
        "  segment confusion counts and metrics",
        "  event TP/FP/TN/FN and metrics",
        "",
        "Global integrity checks:",
        "  no physical subject leakage",
        "  no simulator trajectory leakage",
        "  no simulator data in physical val/test",
        "  same held-out physical folds across Physical/SIM20/SIM50/SIM70/SIM100",
        "  fixed CNN/preprocessing/hyperparameters across conditions",
        "  reconcile pooled physical test events with 6,325",
        "  reconcile simulator eligible trajectories with 310",
        "  reconcile simulator windows with 18,824 Activity + 1,256 Falling",
    ])
    (out/"FINAL_AUDIT_REQUIREMENTS.txt").write_text(req+"\n",encoding="utf-8")

    manifests=[]
    for p in out.rglob("*"):
        if p.is_file():
            try: manifests.append({"relative_path":str(p.relative_to(out)),"size_bytes":p.stat().st_size,"sha256":sha256(p)})
            except: pass
    pd.DataFrame(manifests).to_csv(inv/"AUDIT_BUNDLE_CHECKSUMS.csv",index=False)

    (out/"README.txt").write_text(
        "READ-ONLY stage-1 audit of CNN dataset splitting and simulator sampling.\n"
        "No training or repository files are modified.\n"
        "Upload the resulting ZIP for exact fold-table reconstruction.\n",encoding="utf-8"
    )

    print("\nDONE")
    print("Audit directory:",out)
    print("\nZip with:")
    print(f'cd "{root}" && zip -r cnn_dataset_split_audit_stage1.zip "{out.relative_to(root)}"')

if __name__=="__main__":
    main()
