#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, subprocess
from pathlib import Path
import pandas as pd

EXCLUDE_DIRS={".git",".venv","venv","__pycache__","node_modules",".idea",".vscode",".pytest_cache"}
PATH_TERMS=["cnn","fold","split","sample","sampling","sim20","sim50","sim70","sim100","physical","simulator","evaluation","eval","result","metric","confusion","window","trajectory","subject","participant","kfall","univr","manifest"]
CODE_TERMS=["SIM20","SIM50","SIM70","SIM100","simulator_only","simulator only","physical_only","physical only","KFold","GroupKFold","train_test_split","random_state","sampling_list","sampling list","replacement","replace=True","replace = True","trajectory","profile","subject","Falling","Activity","event","confusion","balanced_accuracy","precision","recall","f1"]

def relevant(p):
    s=str(p).lower()
    return any(t in s for t in PATH_TERMS)

def git(root,args):
    r=subprocess.run(["git",*args],cwd=root,text=True,capture_output=True,check=False)
    return (r.stdout or r.stderr).strip()

def iter_files(root):
    for dp,dns,fns in os.walk(root):
        dns[:]=[d for d in dns if d not in EXCLUDE_DIRS]
        for fn in fns:
            p=Path(dp)/fn
            if any(part in EXCLUDE_DIRS for part in p.parts): continue
            yield p

def rel(p,root):
    try:return str(p.resolve().relative_to(root.resolve()))
    except:return str(p)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=Path.cwd())
    ap.add_argument("--out",type=Path,default=None)
    a=ap.parse_args()
    root=a.root.resolve()
    out=(a.out or root/"outputs"/"paper_dataset_split_locator.txt").resolve()
    out.parent.mkdir(parents=True,exist_ok=True)
    L=[]; add=L.append
    add("DATASET / SPLIT / SAMPLING EVIDENCE LOCATOR")
    add("="*100); add(f"ROOT: {root}")
    add(f"BRANCH: {git(root,['branch','--show-current'])}")
    add(f"COMMIT: {git(root,['rev-parse','HEAD'])}")
    add(f"LAST: {git(root,['log','-1','--oneline'])}")
    add("STATUS:"); add(git(root,['status','--short'])); add("")
    files=[]
    for p in iter_files(root):
        try:size=p.stat().st_size
        except:continue
        if relevant(p) or p.suffix.lower()==".py": files.append((p,size))
    files.sort(key=lambda x:rel(x[0],root).lower())

    add("RELEVANT FILE PATHS"); add("-"*100)
    for p,size in files: add(f"{size:12d}  {rel(p,root)}")

    add("\nCSV / TSV HEADERS ONLY"); add("-"*100)
    for p,size in files:
        if p.suffix.lower() not in {".csv",".tsv"}: continue
        low=str(p).lower()
        if not any(t in low for t in ["fold","split","sample","sampling","result","metric","confusion","manifest","window","trajectory"]): continue
        try:
            sep="\t" if p.suffix.lower()==".tsv" else ","
            h=pd.read_csv(p,sep=sep,nrows=0,comment="#")
            add(f"\nFILE: {rel(p,root)}")
            add(f"SIZE: {size}")
            add("COLUMNS: "+" | ".join(map(str,h.columns)))
        except Exception as e:
            add(f"\nFILE: {rel(p,root)}")
            add(f"HEADER ERROR: {e!r}")

    add("\nJSON STRUCTURE ONLY"); add("-"*100)
    for p,size in files:
        if p.suffix.lower()!=".json" or size>20*1024*1024: continue
        try:
            obj=json.loads(p.read_text(encoding="utf-8",errors="replace"))
            add(f"\nFILE: {rel(p,root)}")
            if isinstance(obj,dict): add(f"TYPE: dict KEYS: {list(obj)[:120]}")
            elif isinstance(obj,list):
                add(f"TYPE: list LEN: {len(obj)}")
                if obj and isinstance(obj[0],dict): add(f"FIRST_KEYS: {list(obj[0])[:120]}")
            else:add(f"TYPE: {type(obj).__name__}")
        except Exception as e:
            add(f"\nFILE: {rel(p,root)}"); add(f"JSON ERROR: {e!r}")

    add("\nFOCUSED CODE / TEXT EXCERPTS"); add("-"*100)
    for p,size in files:
        if p.suffix.lower() not in {".py",".txt",".md",".yaml",".yml",".json"} or size>8*1024*1024: continue
        try:src=p.read_text(encoding="utf-8",errors="replace").splitlines()
        except:continue
        hits=[i for i,line in enumerate(src) if any(t.lower() in line.lower() for t in CODE_TERMS)]
        if not hits: continue
        spans=[]
        for i in hits:
            s,e=max(0,i-3),min(len(src),i+4)
            if spans and s<=spans[-1][1]+2: spans[-1]=(spans[-1][0],max(spans[-1][1],e))
            else: spans.append((s,e))
        add("\n"+"="*100); add("FILE: "+rel(p,root)); add("="*100)
        for s,e in spans[:30]:
            add(f"\n--- lines {s+1}-{e} ---")
            for j in range(s,e): add(f"{j+1:06d}: {src[j]}")
        if len(spans)>30:add(f"[TRUNCATED {len(spans)-30} additional spans]")

    add("\nNEXT STEP")
    add("-"*100)
    add("Upload ONLY this TXT file. A second targeted script can then compute the exact fold-by-fold counts.")
    out.write_text("\n".join(L)+"\n",encoding="utf-8")
    print(f"Created: {out}")
    print(f"Size: {out.stat().st_size/1024:.1f} KB")

if __name__=="__main__":
    main()
