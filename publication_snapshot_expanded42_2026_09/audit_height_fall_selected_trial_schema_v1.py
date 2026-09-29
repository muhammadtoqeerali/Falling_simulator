#!/usr/bin/env python3
"""
audit_height_fall_selected_trial_schema_v1.py

Purpose:
  Inspect the exact files selected by V5 for Task 39 / P020 (or whatever V5 selected)
  and determine why no usable timeseries/time axis was recognized.

This is an AUDIT ONLY.
It does not modify any project file, rerun MuJoCo, train a model, or infer events.

It creates a compact ZIP containing:
  - exact selected file index
  - per-file schema summaries
  - first/last rows (small samples only)
  - JSON key summaries
  - candidate time/frame/sample columns
  - candidate kinematic/contact/IMU/state columns
  - a nearby-file search for the same task/profile under _highrate_overnight
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import zipfile
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path("/mnt/hdd16T/ToqeerHomeBackup/mujoco_project")
DEFAULT_V5 = PROJECT / "outputs/paper_task39_height_fall_evidence_v5"
DEFAULT_SEARCH_ROOT = PROJECT / "outputs/_highrate_overnight"
DEFAULT_OUT = PROJECT / "outputs/task39_selected_trial_schema_audit_v1"
DEFAULT_ZIP = PROJECT / "outputs/task39_selected_trial_schema_audit_v1.zip"

def norm(x):
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())

def sha256(path: Path, block=1024*1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def safe_read_csv(path: Path, nrows=None):
    try:
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    except Exception:
        try:
            return pd.read_csv(path, nrows=nrows, engine="python")
        except Exception:
            return None

def safe_read_json(path: Path):
    try:
        return json.loads(path.read_text(errors="replace"))
    except Exception:
        return None

def flatten_json(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            kk = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_json(v, kk))
    elif isinstance(obj, list):
        if len(obj) <= 50 and all(not isinstance(v, (dict, list)) for v in obj):
            out[prefix] = obj
        else:
            out[prefix] = f"<list len={len(obj)}>"
    else:
        out[prefix] = obj
    return out

def detect_numeric_axis(df: pd.DataFrame):
    rows = []
    n = len(df)
    if n < 2:
        return pd.DataFrame()

    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        valid = s.dropna()
        if len(valid) < max(3, int(0.8*n)):
            continue

        arr = valid.to_numpy(dtype=float)
        dif = np.diff(arr)
        finite = np.isfinite(dif)
        if finite.sum() == 0:
            continue
        dif = dif[finite]

        monotonic_inc = bool(np.all(dif >= 0))
        strictly_inc = bool(np.all(dif > 0))
        unique_ratio = float(valid.nunique() / max(len(valid), 1))
        median_step = float(np.median(dif)) if len(dif) else np.nan
        q05 = float(np.percentile(dif, 5)) if len(dif) else np.nan
        q95 = float(np.percentile(dif, 95)) if len(dif) else np.nan

        nc = norm(c)
        lexical = 0
        for token in ["time", "timestamp", "sec", "second", "frame", "sample", "index", "step", "tick"]:
            if token in nc:
                lexical += 10

        score = lexical
        score += 20 if strictly_inc else (10 if monotonic_inc else 0)
        score += 10 if unique_ratio > 0.95 else 0
        if np.isfinite(median_step) and median_step > 0:
            score += 5

        rows.append({
            "column": c,
            "score": score,
            "monotonic_increasing": monotonic_inc,
            "strictly_increasing": strictly_inc,
            "unique_ratio": unique_ratio,
            "first": float(valid.iloc[0]),
            "last": float(valid.iloc[-1]),
            "median_step": median_step,
            "step_p05": q05,
            "step_p95": q95,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["score", "column"], ascending=[False, True])

def classify_columns(columns):
    groups = {
        "time_or_index": [],
        "imu": [],
        "kinematics": [],
        "contact_dynamics": [],
        "state_qpos_qvel": [],
        "events": [],
        "markers_xyz_like": [],
    }

    for c in columns:
        nc = norm(c)
        if any(k in nc for k in ["time", "timestamp", "frame", "sample", "index", "step", "tick"]):
            groups["time_or_index"].append(c)
        if any(k in nc for k in ["acc", "gyro", "imu", "angularvel", "angvel"]):
            groups["imu"].append(c)
        if any(k in nc for k in ["pelvis", "trunk", "torso", "com", "head", "height", "lean", "pitch", "roll", "yaw"]):
            groups["kinematics"].append(c)
        if any(k in nc for k in ["contact", "grf", "force", "impact", "load", "normal", "tangent", "friction"]):
            groups["contact_dynamics"].append(c)
        if any(k in nc for k in ["qpos", "qvel", "jointpos", "jointvel", "jointangle"]):
            groups["state_qpos_qvel"].append(c)
        if any(k in nc for k in ["onset", "impact", "fallstart", "fallend", "rest", "event"]):
            groups["events"].append(c)

        # broad xyz marker-like detector
        if re.search(r"(?:_|\.|-)[xyz]$", str(c), re.I):
            groups["markers_xyz_like"].append(c)
    return groups

def profile_task_tokens_from_selected(v5_dir: Path):
    sel = v5_dir / "01_SELECTION/SELECTED_TRIAL.txt"
    task = None
    profile = None
    if sel.exists():
        text = sel.read_text(errors="replace")
        mt = re.search(r"task_id\s*=\s*(\d+)", text)
        mp = re.search(r"profile_id\s*=\s*([A-Za-z0-9_-]+)", text)
        if mt:
            task = int(mt.group(1))
        if mp:
            profile = mp.group(1)

    # V5 may not have written SELECTED_TRIAL.txt before failing. Fall back to log/index.
    idx = v5_dir / "01_SELECTION/selected_trial_file_index.csv"
    if idx.exists():
        d = safe_read_csv(idx)
        if d is not None and not d.empty:
            if task is None and "task_id" in d.columns:
                vals = pd.to_numeric(d["task_id"], errors="coerce").dropna()
                if len(vals):
                    task = int(vals.iloc[0])
            if profile is None and "profile_id" in d.columns and len(d["profile_id"].dropna()):
                profile = str(d["profile_id"].dropna().iloc[0])
    return task, profile

def summarize_csv(path: Path, out_dir: Path):
    full = safe_read_csv(path)
    if full is None:
        return {"status": "READ_FAIL"}

    schema = []
    for c in full.columns:
        s = full[c]
        num = pd.to_numeric(s, errors="coerce")
        schema.append({
            "column": c,
            "dtype_read": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "numeric_non_null": int(num.notna().sum()),
            "n_unique": int(s.nunique(dropna=True)),
            "example_first": "" if len(s)==0 else str(s.iloc[0])[:120],
            "example_last": "" if len(s)==0 else str(s.iloc[-1])[:120],
        })
    pd.DataFrame(schema).to_csv(out_dir / f"{path.name}__schema.csv", index=False)

    axis = detect_numeric_axis(full)
    axis.to_csv(out_dir / f"{path.name}__candidate_axes.csv", index=False)

    groups = classify_columns(full.columns)
    (out_dir / f"{path.name}__column_groups.json").write_text(json.dumps(groups, indent=2))

    # Small samples only.
    full.head(8).to_csv(out_dir / f"{path.name}__head8.csv", index=False)
    full.tail(8).to_csv(out_dir / f"{path.name}__tail8.csv", index=False)

    return {
        "status": "PASS",
        "rows": int(len(full)),
        "columns": int(len(full.columns)),
        "best_axis": "" if axis.empty else str(axis.iloc[0]["column"]),
        "best_axis_score": np.nan if axis.empty else float(axis.iloc[0]["score"]),
        "imu_cols": len(groups["imu"]),
        "kinematic_cols": len(groups["kinematics"]),
        "contact_cols": len(groups["contact_dynamics"]),
        "state_cols": len(groups["state_qpos_qvel"]),
        "marker_xyz_like_cols": len(groups["markers_xyz_like"]),
        "event_cols": len(groups["events"]),
    }

def summarize_json(path: Path, out_dir: Path):
    obj = safe_read_json(path)
    if obj is None:
        return {"status": "READ_FAIL"}
    flat = flatten_json(obj)
    rows = [{"key": k, "value": str(v)[:500]} for k, v in flat.items()]
    pd.DataFrame(rows).to_csv(out_dir / f"{path.name}__json_keys.csv", index=False)

    groups = classify_columns(list(flat.keys()))
    (out_dir / f"{path.name}__json_groups.json").write_text(json.dumps(groups, indent=2))

    return {
        "status": "PASS",
        "keys": len(flat),
        "imu_keys": len(groups["imu"]),
        "kinematic_keys": len(groups["kinematics"]),
        "contact_keys": len(groups["contact_dynamics"]),
        "state_keys": len(groups["state_qpos_qvel"]),
        "event_keys": len(groups["events"]),
    }

def likely_match(path: Path, task: int, profile: str):
    s = str(path)
    ns = norm(s)
    pnum = re.sub(r"\D", "", profile or "")
    task_hit = bool(
        re.search(rf"(?i)(?:task|scenario|T)[_\- .]*0*{task}(?!\d)", s)
        or re.search(rf"(?<!\d){task}(?!\d)", path.name)
    )
    profile_hit = False
    if profile:
        profile_hit = bool(
            re.search(rf"(?i)(?<![A-Za-z0-9]){re.escape(profile)}(?![A-Za-z0-9])", s)
            or (pnum and re.search(rf"(?i)(?:profile|prof|subject|subj|P)[_\- .]*0*{int(pnum)}(?!\d)", s))
        )
    return task_hit and profile_hit

def search_nearby(search_root: Path, task: int, profile: str):
    rows = []
    exts = {".csv", ".json", ".npz", ".npy", ".pkl", ".pickle", ".png", ".jpg", ".jpeg"}
    for p in search_root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        if likely_match(p, task, profile):
            rows.append({
                "path": str(p),
                "suffix": p.suffix.lower(),
                "bytes": p.stat().st_size,
                "sha256": sha256(p) if p.stat().st_size < 500_000_000 else "",
            })
    return pd.DataFrame(rows)

def zip_folder(folder: Path, zip_path: Path):
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(folder.parent)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5-dir", type=Path, default=DEFAULT_V5)
    ap.add_argument("--search-root", type=Path, default=DEFAULT_SEARCH_ROOT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP)
    args = ap.parse_args()

    if not args.v5_dir.exists():
        raise SystemExit(f"ERROR: V5 audit directory not found: {args.v5_dir}")

    out = args.output_dir
    if out.exists():
        shutil.rmtree(out)
    (out / "selected_files").mkdir(parents=True, exist_ok=True)
    (out / "schemas").mkdir(parents=True, exist_ok=True)
    (out / "nearby_search").mkdir(parents=True, exist_ok=True)

    task, profile = profile_task_tokens_from_selected(args.v5_dir)
    if task is None or profile is None:
        raise SystemExit(
            "ERROR: could not resolve selected task/profile from V5 selected_trial_file_index.csv"
        )

    print("="*110)
    print("TASK/PROFILE SCHEMA AUDIT")
    print("="*110)
    print("Selected task:", task)
    print("Selected profile:", profile)

    src_index = args.v5_dir / "01_SELECTION/selected_trial_file_index.csv"
    d = safe_read_csv(src_index)
    if d is None or d.empty:
        raise SystemExit(f"ERROR: cannot read {src_index}")

    shutil.copy2(src_index, out / "selected_files/selected_trial_file_index.csv")

    summaries = []
    print("\nEXACT SELECTED FILES")
    for _, r in d.iterrows():
        p = Path(r["path"])
        print("-", p)
        rec = {
            "path": str(p),
            "exists": p.exists(),
            "suffix": p.suffix.lower(),
            "bytes": p.stat().st_size if p.exists() else -1,
            "kind_from_v5": r.get("kind", ""),
        }
        if not p.exists():
            summaries.append(rec)
            continue

        if p.suffix.lower() == ".csv":
            rec.update(summarize_csv(p, out / "schemas"))
        elif p.suffix.lower() == ".json":
            rec.update(summarize_json(p, out / "schemas"))
        else:
            rec["status"] = "BINARY_OR_IMAGE"

        summaries.append(rec)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out / "selected_files/file_schema_summary.csv", index=False)

    print("\nSCHEMA SUMMARY")
    cols = [c for c in [
        "path", "status", "rows", "columns", "best_axis", "best_axis_score",
        "imu_cols", "kinematic_cols", "contact_cols", "state_cols",
        "marker_xyz_like_cols", "event_cols"
    ] if c in summary_df.columns]
    print(summary_df[cols].to_string(index=False))

    print("\nSEARCHING NEARBY _highrate_overnight FOR SAME TASK/PROFILE...")
    near = search_nearby(args.search_root, task, profile)
    near.to_csv(out / "nearby_search/task_profile_nearby_files.csv", index=False)
    print("Nearby matching files:", len(near))
    if len(near):
        print(near[["suffix", "bytes", "path"]].to_string(index=False))

    # Produce one compact diagnosis text.
    diagnosis = []
    diagnosis.append(f"task_id={task}")
    diagnosis.append(f"profile_id={profile}")
    diagnosis.append(f"selected_files={len(d)}")
    diagnosis.append(f"nearby_matches={len(near)}")
    diagnosis.append("")
    diagnosis.append("Selected-file schema summary:")
    diagnosis.append(summary_df.to_string(index=False))
    diagnosis.append("")
    diagnosis.append("Interpretation:")
    if "best_axis" in summary_df.columns and summary_df["best_axis"].fillna("").astype(str).str.len().gt(0).any():
        diagnosis.append("- At least one selected CSV has a strong monotonic numeric axis; V6 can use/convert it.")
    else:
        diagnosis.append("- No strong explicit/implicit numeric axis detected in selected CSVs.")
    if "kinematic_cols" in summary_df.columns and summary_df["kinematic_cols"].fillna(0).sum() > 0:
        diagnosis.append("- Selected files contain kinematic-like columns.")
    else:
        diagnosis.append("- Selected files do not expose obvious pelvis/trunk/kinematic columns.")
    if "contact_cols" in summary_df.columns and summary_df["contact_cols"].fillna(0).sum() > 0:
        diagnosis.append("- Selected files contain contact/dynamics-like columns.")
    else:
        diagnosis.append("- Selected files do not expose obvious contact/dynamics columns.")
    if "state_cols" in summary_df.columns and summary_df["state_cols"].fillna(0).sum() > 0:
        diagnosis.append("- Selected files expose state/qpos/qvel-like columns.")
    else:
        diagnosis.append("- Selected CSVs do not expose obvious qpos/qvel columns; nearby binary/state files may still exist.")
    (out / "DIAGNOSIS.txt").write_text("\n".join(diagnosis) + "\n")

    zip_folder(out, args.zip_path)
    zsha = sha256(args.zip_path)
    sha_path = args.zip_path.parent / f"{args.zip_path.name}.sha256"
    sha_path.write_text(f"{zsha}  {args.zip_path}\n")

    print("\n" + "="*110)
    print("SCHEMA AUDIT: PASS")
    print("="*110)
    print("ZIP:", args.zip_path)
    print("ZIP SHA256:", zsha)

if __name__ == "__main__":
    main()
