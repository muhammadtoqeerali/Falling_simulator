#!/usr/bin/env python3
"""
prepare_height_fall_paper_evidence_v5.py

Standalone, self-contained extractor for one corrected396 height-fall trial.
This REPLACES the earlier patch-based approach.

Default:
    Task 39 = forward fall from height

It does NOT modify prepare_height_fall_paper_evidence.py.
It does NOT rerun MuJoCo, CNN training, or inference.

Outputs:
- exact selected source files
- Task-39 file/profile audit
- candidate profile ranking
- canonical event-manifest rows
- exact primary timeseries
- phase timestamps:
    SETUP
    FALL_ONSET
    MAX_DESCENT
    FIRST_CONTACT
    PEAK_IMPACT
    POST_IMPACT
    REST
- phase snapshot rows
- marker/skeleton xyz positions when present
- compact kinematic/contact/IMU timeseries
- diagnostic previews
- one ZIP for upload
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

PROJECT = Path("/mnt/hdd16T/ToqeerHomeBackup/mujoco_project")
DEFAULT_ROOT = PROJECT / "outputs/_highrate_overnight/campaign_highrate_truth_v3_rne_corrected396"
DEFAULT_EVENT_MANIFEST = PROJECT / "outputs/phase2_publication_gate_audit_20260826/final_synthetic_event_policy_v2/canonical_synthetic_event_manifest_v2.csv"

TEXT_EXT = {".csv", ".json", ".txt"}
STATE_EXT = {".npz", ".npy", ".pkl", ".pickle"}
FIG_EXT = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}

TASK_ALIASES = ["task_id", "task", "taskid", "activity_id", "scenario_id", "scenario", "task_number"]
PROFILE_ALIASES = ["profile_id", "profile", "profileid", "subject_id", "subject", "avatar_id", "human_id"]
TIME_ALIASES = ["time", "time_s", "timestamp", "timestamp_s", "t", "sim_time", "time_sec", "seconds"]
ONSET_ALIASES = ["fall_onset_time", "onset_time", "fall_start_time", "start_fall_time", "t_onset", "fall_onset", "onset"]
IMPACT_ALIASES = ["impact_time", "peak_impact_time", "recovered_impact_time", "t_impact", "impact"]
REST_ALIASES = ["rest_time", "settle_time", "settled_time", "t_rest"]

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

def read_csv(path: Path, nrows=None):
    try:
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    except Exception:
        try:
            return pd.read_csv(path, nrows=nrows, engine="python")
        except Exception:
            return None

def read_json(path: Path):
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
        if len(obj) <= 32 and all(not isinstance(v, (dict, list)) for v in obj):
            out[prefix] = obj
        else:
            out[prefix] = json.dumps(obj)
    else:
        out[prefix] = obj
    return out

def find_col(columns, aliases):
    cmap = {norm(c): c for c in columns}
    for a in aliases:
        if norm(a) in cmap:
            return cmap[norm(a)]
    for c in columns:
        nc = norm(c)
        for a in aliases:
            na = norm(a)
            if len(na) >= 5 and (na in nc or nc in na):
                return c
    return None

def canonical_profile_id(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None

    pats = [
        r"(?i)^P[_\- .]*0*(\d{1,4})$",
        r"(?i)^(?:profile|prof)[_\- .]*(?:P[_\- .]*)?0*(\d{1,4})$",
        r"(?i)^(?:subject|subj)[_\- .]*(?:P[_\- .]*)?0*(\d{1,4})$",
        r"^0*(\d{1,4})$",
    ]
    for pat in pats:
        m = re.match(pat, s)
        if m:
            return f"P{int(m.group(1)):03d}"
    return s

def path_task_profile(path: Path):
    """
    Strict parser. It cannot interpret 'mujoco_project' as profile 'roject'.
    """
    near = [path.name] + list(reversed(path.parts[-8:-1]))

    profile = None
    ppats = [
        re.compile(r"(?<![A-Za-z0-9])P[_\- .]*0*(\d{1,4})(?![A-Za-z0-9])", re.I),
        re.compile(r"(?<![A-Za-z0-9])(?:profile|prof)[_\- .]*(?:P[_\- .]*)?0*(\d{1,4})(?![A-Za-z0-9])", re.I),
        re.compile(r"(?<![A-Za-z0-9])(?:subject|subj)[_\- .]*(?:P[_\- .]*)?0*(\d{1,4})(?![A-Za-z0-9])", re.I),
    ]
    for part in near:
        for pat in ppats:
            m = pat.search(str(part))
            if m:
                profile = f"P{int(m.group(1)):03d}"
                break
        if profile:
            break

    task = None
    tpats = [
        re.compile(r"(?<![A-Za-z0-9])task[_\- .]*0*(\d{1,3})(?!\d)", re.I),
        re.compile(r"(?<![A-Za-z0-9])scenario[_\- .]*0*(\d{1,3})(?!\d)", re.I),
        re.compile(r"(?<![A-Za-z0-9])T[_\- .]*0*(\d{1,3})(?!\d)", re.I),
    ]
    for part in near:
        for pat in tpats:
            m = pat.search(str(part))
            if m:
                task = int(m.group(1))
                break
        if task is not None:
            break

    return task, profile

def header_task_profile(path: Path):
    task = None
    profile = None
    cols = []

    if path.suffix.lower() == ".csv":
        d = read_csv(path, 5)
        if d is None:
            return task, profile, cols
        cols = list(d.columns)
        tc = find_col(cols, TASK_ALIASES)
        pc = find_col(cols, PROFILE_ALIASES)

        if tc:
            vals = pd.to_numeric(d[tc], errors="coerce").dropna()
            if not vals.empty:
                task = int(vals.iloc[0])

        if pc and not d[pc].dropna().empty:
            profile = canonical_profile_id(d[pc].dropna().iloc[0])

    elif path.suffix.lower() == ".json":
        obj = read_json(path)
        if obj is None:
            return task, profile, cols
        flat = flatten_json(obj)
        cols = list(flat)
        tc = find_col(cols, TASK_ALIASES)
        pc = find_col(cols, PROFILE_ALIASES)
        if tc is not None:
            try:
                task = int(float(flat[tc]))
            except Exception:
                pass
        if pc is not None:
            profile = canonical_profile_id(flat[pc])

    return task, profile, cols

def classify_kind(path: Path, columns):
    name = path.name.lower()
    ncols = [norm(c) for c in columns]

    if "validation" in name or "report" in name or any("validation" in c for c in ncols):
        return "validation"
    if any(("acc" in c or "gyro" in c or "imu" in c) for c in ncols):
        if "truth" in name or any("truth" in c for c in ncols):
            return "highrate_truth"
        return "imu_timeseries"
    if any(("pelvis" in c or "trunk" in c or "grf" in c or "contact" in c or "impact" in c or "force" in c) for c in ncols):
        return "physics_timeseries"
    if path.suffix.lower() in STATE_EXT:
        return "state"
    if path.suffix.lower() in FIG_EXT:
        return "figure"
    return "other"

def discover_task_files(root: Path, task_id: int):
    rows = []
    exts = TEXT_EXT | STATE_EXT | FIG_EXT

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue

        path_task, path_profile = path_task_profile(path)
        head_task = head_profile = None
        cols = []

        # Inspect CSV/JSON headers when useful.
        if path.suffix.lower() in {".csv", ".json"}:
            head_task, head_profile, cols = header_task_profile(path)

        task = head_task if head_task is not None else path_task
        profile = head_profile if head_profile is not None else path_profile
        profile = canonical_profile_id(profile)

        if task != task_id:
            continue

        if not cols and path.suffix.lower() == ".csv":
            try:
                cols = list(pd.read_csv(path, nrows=0).columns)
            except Exception:
                cols = []

        rows.append({
            "path": str(path),
            "name": path.name,
            "suffix": path.suffix.lower(),
            "bytes": path.stat().st_size,
            "task_id": task_id,
            "profile_id": "" if profile is None else profile,
            "profile_from_path": "" if path_profile is None else path_profile,
            "profile_from_header": "" if head_profile is None else head_profile,
            "kind": classify_kind(path, cols),
            "n_columns": len(cols),
            "columns": " | ".join(cols[:120]),
        })

    return pd.DataFrame(rows)

def load_task_manifest(path: Path, task_id: int):
    if not path.exists():
        return pd.DataFrame()
    d = read_csv(path)
    if d is None or d.empty:
        return pd.DataFrame()
    tc = find_col(d.columns, TASK_ALIASES)
    if tc is None:
        return pd.DataFrame()
    tv = pd.to_numeric(d[tc], errors="coerce")
    out = d[tv == task_id].copy()
    pc = find_col(out.columns, PROFILE_ALIASES)
    if pc:
        out["_canonical_profile_id"] = out[pc].map(canonical_profile_id)
    return out

def score_candidates(index_df, manifest_df):
    profiles = set()
    if not index_df.empty:
        profiles.update(p for p in index_df["profile_id"].astype(str) if p)
    if not manifest_df.empty and "_canonical_profile_id" in manifest_df.columns:
        profiles.update(p for p in manifest_df["_canonical_profile_id"].astype(str) if p and p != "nan")

    rows = []
    for p in sorted(profiles):
        f = index_df[index_df["profile_id"].astype(str) == p] if not index_df.empty else pd.DataFrame()
        kinds = f["kind"].value_counts().to_dict() if not f.empty else {}
        eligible = False
        if not manifest_df.empty and "_canonical_profile_id" in manifest_df.columns:
            eligible = bool((manifest_df["_canonical_profile_id"].astype(str) == p).any())

        score = 0.0
        score += 100.0 if eligible else 0.0
        score += 20.0 * kinds.get("highrate_truth", 0)
        score += 12.0 * kinds.get("physics_timeseries", 0)
        score += 10.0 * kinds.get("imu_timeseries", 0)
        score += 4.0 * kinds.get("state", 0)
        score += 2.0 * kinds.get("validation", 0)
        score += min(len(f), 10)

        rows.append({
            "profile_id": p,
            "eligible_in_event_manifest": eligible,
            "score": score,
            "n_files": len(f),
            "n_highrate_truth": kinds.get("highrate_truth", 0),
            "n_imu": kinds.get("imu_timeseries", 0),
            "n_physics": kinds.get("physics_timeseries", 0),
            "n_state": kinds.get("state", 0),
            "n_validation": kinds.get("validation", 0),
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["eligible_in_event_manifest", "n_files", "score", "profile_id"],
        ascending=[False, False, False, True]
    ).reset_index(drop=True)

def choose_profile(candidates, requested):
    if candidates.empty:
        raise RuntimeError("No candidate profiles were resolved.")

    if str(requested).upper() != "AUTO":
        p = canonical_profile_id(requested)
        hit = candidates[candidates["profile_id"] == p]
        if hit.empty:
            raise RuntimeError(f"Requested profile {p} not found.")
        if int(hit.iloc[0]["n_files"]) <= 0:
            raise RuntimeError(f"Requested profile {p} has no exact Task files.")
        return p

    # Strong preference: event eligible AND has exact files.
    good = candidates[
        (candidates["eligible_in_event_manifest"] == True) &
        (candidates["n_files"] > 0)
    ]
    if not good.empty:
        return str(good.iloc[0]["profile_id"])

    # Next-best: exact file-backed profile.
    backed = candidates[candidates["n_files"] > 0]
    if not backed.empty:
        return str(backed.iloc[0]["profile_id"])

    raise RuntimeError("Manifest profiles exist, but none have exact Task files.")

def choose_primary_timeseries(files_df):
    ranked = []
    for _, r in files_df.iterrows():
        p = Path(r["path"])
        if p.suffix.lower() != ".csv":
            continue
        d = read_csv(p, 5)
        if d is None:
            continue
        tc = find_col(d.columns, TIME_ALIASES)
        if tc is None:
            continue

        richness = 0
        for c in d.columns:
            nc = norm(c)
            if any(k in nc for k in [
                "pelvis", "trunk", "torso", "contact", "grf", "force",
                "impact", "acc", "gyro", "imu", "head", "lean"
            ]):
                richness += 1

        kind_bonus = {
            "highrate_truth": 100,
            "physics_timeseries": 70,
            "imu_timeseries": 50,
            "other": 0,
        }.get(r["kind"], 0)

        ranked.append((kind_bonus + richness + math.log10(max(10, p.stat().st_size)), p))

    if not ranked:
        return None
    ranked.sort(reverse=True, key=lambda x: x[0])
    return ranked[0][1]

def load_numeric_timeseries(path):
    d = read_csv(path)
    if d is None or d.empty:
        return None, None
    tc = find_col(d.columns, TIME_ALIASES)
    if tc is None:
        return None, None
    t = pd.to_numeric(d[tc], errors="coerce")
    mask = t.notna()
    d = d.loc[mask].copy()
    d[tc] = t[mask].astype(float)
    d = d.sort_values(tc).drop_duplicates(tc).reset_index(drop=True)
    return d, tc

def manifest_profile_rows(manifest, profile):
    if manifest.empty:
        return manifest
    if "_canonical_profile_id" not in manifest.columns:
        return pd.DataFrame()
    return manifest[manifest["_canonical_profile_id"].astype(str) == profile].copy()

def get_scalar_from_df(d, aliases):
    if d is None or d.empty:
        return None, None
    c = find_col(d.columns, aliases)
    if c is None:
        return None, None
    vals = pd.to_numeric(d[c], errors="coerce").dropna()
    if vals.empty:
        return None, None
    return float(vals.iloc[0]), c

def pelvis_height_col(columns):
    for aliases in [
        ["pelvis_height"],
        ["pelvis_z"],
        ["pelvis_pos_z"],
        ["pelvis_position_z"],
    ]:
        c = find_col(columns, aliases)
        if c:
            return c
    for c in columns:
        nc = norm(c)
        if "pelvis" in nc and ("height" in nc or nc.endswith("z") or "posz" in nc):
            return c
    return None

def trunk_lean_col(columns):
    for aliases in [
        ["trunk_lean"],
        ["trunk_lean_deg"],
        ["torso_lean"],
        ["trunk_angle"],
        ["torso_pitch"],
    ]:
        c = find_col(columns, aliases)
        if c:
            return c
    for c in columns:
        nc = norm(c)
        if ("trunk" in nc or "torso" in nc) and any(k in nc for k in ["lean", "pitch", "angle"]):
            return c
    return None

def marker_triplets(columns):
    axes = defaultdict(dict)
    for c in columns:
        s = str(c)
        for pat in [
            re.compile(r"^(.*?)[_\- ]([xyzXYZ])$"),
            re.compile(r"^(.*?)\.([xyzXYZ])$"),
        ]:
            m = pat.match(s)
            if m:
                axes[m.group(1).strip()][m.group(2).lower()] = c
                break
    return {b: a for b, a in axes.items() if all(k in a for k in "xyz")}

def contact_envelope(d, time_col):
    cols = []
    for c in d.columns:
        if c == time_col:
            continue
        nc = norm(c)
        if any(k in nc for k in [
            "contactforce", "grf", "groundreaction", "normalforce",
            "impactmagnitude", "contactload", "primaryimpact"
        ]):
            cols.append(c)

    arr = []
    used = []
    for c in cols:
        x = pd.to_numeric(d[c], errors="coerce").abs().to_numpy(dtype=float)
        if np.isfinite(x).sum() >= 10:
            arr.append(np.nan_to_num(x, nan=0.0))
            used.append(c)
    if not arr:
        return None, []
    return np.max(np.vstack(arr), axis=0), used

def resolve_events(manifest_rows, d, time_col):
    onset, onset_col = get_scalar_from_df(manifest_rows, ONSET_ALIASES)
    impact, impact_col = get_scalar_from_df(manifest_rows, IMPACT_ALIASES)
    rest, rest_col = get_scalar_from_df(manifest_rows, REST_ALIASES)

    sources = {}
    if onset is not None:
        sources["onset"] = f"event_manifest:{onset_col}"
    if impact is not None:
        sources["impact"] = f"event_manifest:{impact_col}"
    if rest is not None:
        sources["rest"] = f"event_manifest:{rest_col}"

    # Try constant/sparse event columns from timeseries when missing.
    if onset is None:
        onset, c = get_scalar_from_df(d, ONSET_ALIASES)
        if onset is not None:
            sources["onset"] = f"timeseries:{c}"
    if impact is None:
        impact, c = get_scalar_from_df(d, IMPACT_ALIASES)
        if impact is not None:
            sources["impact"] = f"timeseries:{c}"
    if rest is None:
        rest, c = get_scalar_from_df(d, REST_ALIASES)
        if rest is not None:
            sources["rest"] = f"timeseries:{c}"

    t = d[time_col].to_numpy(dtype=float)
    t0, t1 = float(t.min()), float(t.max())

    # Impact fallback from contact/load peak.
    env, used = contact_envelope(d, time_col)
    if impact is None and env is not None:
        j = int(np.nanargmax(env))
        impact = float(t[j])
        sources["impact"] = "derived_peak_contact:" + ",".join(used[:5])

    # Onset fallback from strongest pelvis descent before impact.
    pcol = pelvis_height_col(d.columns)
    if onset is None and impact is not None and pcol is not None:
        x = pd.to_numeric(d[pcol], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & (t < impact) & (t >= max(t0, impact - 4.0))
        if mask.sum() >= 10:
            tt = t[mask]
            xx = x[mask]
            vel = np.gradient(xx, tt)
            threshold = np.nanpercentile(vel, 10)
            jj = np.where(vel <= threshold)[0]
            if len(jj):
                onset = float(tt[jj[0]])
                sources["onset"] = f"derived_pelvis_descent:{pcol}"

    # Conservative last-resort fallbacks, always flagged.
    if impact is None:
        impact = t0 + 0.65 * (t1 - t0)
        sources["impact"] = "fallback_fraction_trial"
    if onset is None:
        onset = max(t0, impact - 1.2)
        sources["onset"] = "fallback_impact_minus_1.2s"
    if rest is None:
        rest = t1
        sources["rest"] = "fallback_trial_end"

    onset = float(np.clip(onset, t0, t1))
    impact = float(np.clip(impact, onset, t1))
    rest = float(np.clip(rest, impact, t1))
    return onset, impact, rest, sources

def derive_phases(d, time_col, onset, impact, rest):
    t = d[time_col].to_numpy(dtype=float)
    t0, t1 = float(t.min()), float(t.max())

    setup = max(t0, onset - min(1.5, max(0.5, 0.25 * max(onset - t0, 0.1))))

    descent = onset + 0.55 * max(impact - onset, 0.1)
    descent_reason = "midpoint onset-impact"
    pcol = pelvis_height_col(d.columns)
    if pcol is not None:
        x = pd.to_numeric(d[pcol], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & (t >= onset) & (t <= impact)
        if mask.sum() >= 5:
            tt = t[mask]
            xx = x[mask]
            vel = np.gradient(xx, tt)
            j = int(np.argmin(vel))
            descent = float(tt[j])
            descent_reason = f"max negative pelvis vertical velocity:{pcol}"

    first_contact = max(onset, impact - 0.15)
    contact_reason = "pre-impact fallback"
    env, used = contact_envelope(d, time_col)
    if env is not None:
        mask = (t >= onset) & (t <= min(t1, impact + 0.5))
        tt = t[mask]
        vv = env[mask]
        if len(vv):
            base = np.nanmedian(env[t < onset]) if np.any(t < onset) else 0.0
            peak = float(np.nanmax(vv))
            thr = base + 0.10 * max(peak - base, 0.0)
            idx = np.where(vv > thr)[0]
            if len(idx):
                first_contact = float(tt[idx[0]])
                contact_reason = "first 10% contact/load dynamic-range crossing:" + ",".join(used[:5])

    post = min(t1, impact + 0.25)

    phases = [
        ("SETUP", setup, "pre-onset stable/setup"),
        ("FALL_ONSET", onset, "canonical/derived fall onset"),
        ("MAX_DESCENT", descent, descent_reason),
        ("FIRST_CONTACT", first_contact, contact_reason),
        ("PEAK_IMPACT", impact, "canonical/derived peak impact"),
        ("POST_IMPACT", post, "0.25 s after impact"),
        ("REST", rest, "canonical rest / trial end"),
    ]

    # Enforce monotonic requested times.
    out = []
    last = t0
    for name, tt, reason in phases:
        tt = float(np.clip(max(last, tt), t0, t1))
        out.append((name, tt, reason))
        last = tt
    return out

def nearest_index(d, time_col, target):
    t = d[time_col].to_numpy(dtype=float)
    return int(np.argmin(np.abs(t - target)))

def export_phase_rows(d, time_col, phases, path):
    rows = []
    for name, target, reason in phases:
        i = nearest_index(d, time_col, target)
        row = d.iloc[i]
        rec = {
            "phase": name,
            "requested_time_s": target,
            "sample_time_s": float(row[time_col]),
            "derivation": reason,
        }
        for c in d.columns:
            if c == time_col:
                continue
            v = row[c]
            if pd.isna(v):
                continue
            try:
                rec[c] = float(v)
            except Exception:
                rec[c] = str(v)
        rows.append(rec)
    pd.DataFrame(rows).to_csv(path, index=False)

def export_markers(d, time_col, phases, path):
    trip = marker_triplets(d.columns)
    rows = []
    for phase, target, _ in phases:
        i = nearest_index(d, time_col, target)
        row = d.iloc[i]
        st = float(row[time_col])
        for marker, axes in sorted(trip.items()):
            vals = []
            for a in "xyz":
                vals.append(pd.to_numeric(pd.Series([row[axes[a]]]), errors="coerce").iloc[0])
            if all(pd.notna(v) for v in vals):
                rows.append({
                    "phase": phase,
                    "requested_time_s": target,
                    "sample_time_s": st,
                    "marker": marker,
                    "x": float(vals[0]),
                    "y": float(vals[1]),
                    "z": float(vals[2]),
                })
    pd.DataFrame(rows).to_csv(path, index=False)
    return trip

def export_compact_timeseries(d, time_col, outdir):
    outdir.mkdir(parents=True, exist_ok=True)

    groups = {
        "kinematics_timeseries.csv": ["pelvis", "trunk", "torso", "com", "head", "lean", "height"],
        "contact_dynamics_timeseries.csv": ["contact", "grf", "force", "impact", "load", "normal", "tangent", "friction"],
        "imu_timeseries.csv": ["acc", "gyro", "imu", "angularvel", "angvel"],
    }

    for filename, keys in groups.items():
        cols = [time_col]
        for c in d.columns:
            if c == time_col:
                continue
            nc = norm(c)
            if any(k in nc for k in keys):
                cols.append(c)
        d[cols].to_csv(outdir / filename, index=False)

def copy_selected_files(files_df, dest):
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    used = set()
    for _, r in files_df.iterrows():
        src = Path(r["path"])
        name = src.name
        if name in used:
            name = f"{src.parent.name}__{name}"
        used.add(name)
        dst = dest / name
        try:
            shutil.copy2(src, dst)
            rows.append({
                "source": str(src),
                "copied": str(dst),
                "kind": r["kind"],
                "bytes": src.stat().st_size,
                "sha256": sha256(src),
            })
        except Exception as e:
            rows.append({
                "source": str(src),
                "copied": "",
                "kind": r["kind"],
                "bytes": src.stat().st_size if src.exists() else -1,
                "sha256": "",
                "error": repr(e),
            })
    return pd.DataFrame(rows)

def diagnostic_plot(d, time_col, phases, path):
    if plt is None:
        return False

    series = []
    p = pelvis_height_col(d.columns)
    tr = trunk_lean_col(d.columns)
    if p:
        series.append(("Pelvis height", pd.to_numeric(d[p], errors="coerce").to_numpy(dtype=float)))
    if tr:
        series.append(("Trunk lean", pd.to_numeric(d[tr], errors="coerce").to_numpy(dtype=float)))

    env, _ = contact_envelope(d, time_col)
    if env is not None:
        series.append(("Contact/load envelope", env))

    if not series:
        return False

    t = d[time_col].to_numpy(dtype=float)
    n = len(series)
    fig, axes = plt.subplots(n, 1, figsize=(12, max(3.2, 2.6*n)), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (label, y) in zip(axes, series):
        ax.plot(t, y, lw=1.2)
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        for name, tt, _ in phases:
            ax.axvline(tt, ls="--", lw=0.8, alpha=0.65)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Diagnostic Task-39 phase extraction — not final paper artwork")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True

def zip_folder(folder, zip_path):
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(folder.parent)))

def parser_self_test():
    tests = [
        (Path("/a/mujoco_project/outputs/task_39/P014/trial.csv"), (39, "P014")),
        (Path("/a/profile-P015/task39_data.csv"), (39, "P015")),
        (Path("/a/P016/T39_truth.csv"), (39, "P016")),
        (Path("/a/mujoco_project/outputs/task_39/trial.csv"), (39, None)),
    ]
    for p, expected in tests:
        got = path_task_profile(p)
        if got != expected:
            raise RuntimeError(f"Parser self-test failed: {p} -> {got}, expected {expected}")
    if canonical_profile_id("14") != "P014":
        raise RuntimeError("Profile canonicalization self-test failed.")
    print("PARSER SELF-TEST: PASS")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=39)
    ap.add_argument("--profile-id", default="AUTO")
    ap.add_argument("--top-n", type=int, default=22)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--event-manifest", type=Path, default=DEFAULT_EVENT_MANIFEST)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--zip-path", type=Path, required=True)
    args = ap.parse_args()

    parser_self_test()

    print("="*118)
    print("HEIGHT-FALL PAPER EVIDENCE EXTRACTOR V5 — STANDALONE")
    print("="*118)
    print("Task:", args.task_id)
    print("Root:", args.root)
    print("Event manifest:", args.event_manifest)

    if not args.root.exists():
        raise SystemExit(f"ERROR: root not found: {args.root}")

    out = args.output_dir
    if out.exists():
        shutil.rmtree(out)
    for sub in [
        "00_AUDIT",
        "01_SELECTION",
        "02_EXACT_SOURCE_FILES",
        "03_PHASES",
        "04_MARKER_SKELETON",
        "05_TIMESERIES",
        "06_DIAGNOSTIC_PREVIEWS",
    ]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    print("\n[1/7] Discovering exact Task files...")
    index_df = discover_task_files(args.root, args.task_id)
    index_df.to_csv(out / "00_AUDIT/task_file_index.csv", index=False)
    print("Task files:", len(index_df))

    if index_df.empty:
        raise SystemExit("ERROR: zero Task files found.")

    bad = index_df[index_df["profile_id"].astype(str).str.lower().eq("roject")]
    if not bad.empty:
        raise SystemExit("ERROR: regression detected: fake profile `roject` still present.")

    unresolved = index_df[index_df["profile_id"].astype(str).eq("")]
    print("Files with unresolved profile:", len(unresolved))

    print("\n[2/7] Loading canonical Task event-manifest rows...")
    manifest = load_task_manifest(args.event_manifest, args.task_id)
    if not manifest.empty:
        manifest.to_csv(out / "00_AUDIT/task_event_manifest_rows.csv", index=False)
    print("Task manifest rows:", len(manifest))

    print("\n[3/7] Ranking file-backed candidate profiles...")
    candidates = score_candidates(index_df, manifest)
    candidates.to_csv(out / "01_SELECTION/candidate_profiles.csv", index=False)
    print("\nTOP CANDIDATES")
    print(candidates.head(args.top_n).to_string(index=False))

    selected = choose_profile(candidates, args.profile_id)
    selected_files = index_df[index_df["profile_id"].astype(str) == selected].copy()
    selected_files.to_csv(out / "01_SELECTION/selected_trial_file_index.csv", index=False)

    print("\nSELECTED task/profile:", args.task_id, selected)
    print("Selected exact files:", len(selected_files))

    if selected_files.empty:
        raise SystemExit(
            "ERROR: selected profile has zero exact files. "
            "Upload 00_AUDIT/task_file_index.csv and 01_SELECTION/candidate_profiles.csv."
        )

    print("\n[4/7] Copying exact selected-trial source files...")
    copied = copy_selected_files(selected_files, out / "02_EXACT_SOURCE_FILES")
    copied.to_csv(out / "02_EXACT_SOURCE_FILES/source_manifest_sha256.csv", index=False)

    primary = choose_primary_timeseries(selected_files)
    if primary is None:
        raise SystemExit(
            "ERROR: selected profile has files but no usable CSV timeseries with a recognized time column. "
            "Upload 00_AUDIT/task_file_index.csv and 01_SELECTION/selected_trial_file_index.csv."
        )
    print("Primary timeseries:", primary)

    shutil.copy2(primary, out / "02_EXACT_SOURCE_FILES/PRIMARY_TIMESERIES_EXACT.csv")
    (out / "02_EXACT_SOURCE_FILES/PRIMARY_TIMESERIES.txt").write_text(str(primary) + "\n")

    d, time_col = load_numeric_timeseries(primary)
    if d is None:
        raise SystemExit("ERROR: failed to load primary timeseries.")

    schema = [
        f"Primary timeseries: {primary}",
        f"Rows: {len(d)}",
        f"Columns: {len(d.columns)}",
        f"Time column: {time_col}",
        "",
        "COLUMNS",
        *[str(c) for c in d.columns],
    ]
    (out / "00_AUDIT/primary_timeseries_schema.txt").write_text("\n".join(schema) + "\n")

    print("\n[5/7] Resolving fall event phases...")
    mrows = manifest_profile_rows(manifest, selected)
    onset, impact, rest, event_sources = resolve_events(mrows, d, time_col)
    phases = derive_phases(d, time_col, onset, impact, rest)

    phase_df = pd.DataFrame(phases, columns=["phase", "time_s", "derivation"])
    phase_df.to_csv(out / "03_PHASES/phase_timestamps.csv", index=False)

    meta = {
        "task_id": args.task_id,
        "profile_id": selected,
        "primary_timeseries": str(primary),
        "time_column": time_col,
        "trial_start_s": float(d[time_col].min()),
        "trial_end_s": float(d[time_col].max()),
        "fall_onset_s": onset,
        "peak_impact_s": impact,
        "rest_s": rest,
        "event_sources": event_sources,
        "phases": [
            {"phase": name, "time_s": tt, "derivation": reason}
            for name, tt, reason in phases
        ],
    }
    (out / "03_PHASES/phase_metadata.json").write_text(json.dumps(meta, indent=2))

    export_phase_rows(d, time_col, phases, out / "03_PHASES/phase_snapshot_rows.csv")
    triplets = export_markers(d, time_col, phases, out / "04_MARKER_SKELETON/phase_marker_positions_long.csv")
    pd.DataFrame({"marker": sorted(triplets)}).to_csv(
        out / "04_MARKER_SKELETON/detected_markers.csv", index=False
    )

    print("Fall onset:", onset)
    print("Peak impact:", impact)
    print("Rest:", rest)
    print("Detected marker triplets:", len(triplets))
    print(phase_df.to_string(index=False))

    print("\n[6/7] Exporting compact figure-ready timeseries...")
    export_compact_timeseries(d, time_col, out / "05_TIMESERIES")
    diag = diagnostic_plot(d, time_col, phases, out / "06_DIAGNOSTIC_PREVIEWS/diagnostic_phase_timeline.png")

    readme = f"""TASK-{args.task_id} HEIGHT-FALL PAPER EVIDENCE PACKAGE

Selected profile: {selected}
Primary timeseries: {primary}

Scientific phase sequence:
SETUP
FALL_ONSET
MAX_DESCENT
FIRST_CONTACT
PEAK_IMPACT
POST_IMPACT
REST

Exact phase timestamps:
{phase_df.to_string(index=False)}

Key files:
00_AUDIT/task_file_index.csv
01_SELECTION/candidate_profiles.csv
01_SELECTION/selected_trial_file_index.csv
02_EXACT_SOURCE_FILES/source_manifest_sha256.csv
02_EXACT_SOURCE_FILES/PRIMARY_TIMESERIES_EXACT.csv
03_PHASES/phase_metadata.json
03_PHASES/phase_timestamps.csv
03_PHASES/phase_snapshot_rows.csv
04_MARKER_SKELETON/phase_marker_positions_long.csv
05_TIMESERIES/kinematics_timeseries.csv
05_TIMESERIES/contact_dynamics_timeseries.csv
05_TIMESERIES/imu_timeseries.csv

Use later to construct the final paper figure:
(A) exact fall-phase humanoid/skeleton sequence,
(B) sagittal/frontal internal marker skeleton,
(C) pelvis/trunk kinematics,
(D) contact/impact dynamics,
(E) lower-back IMU around onset/contact/impact.

The diagnostic PNG is NOT final publication artwork.

NO simulator rerun.
NO CNN training.
NO model inference.
"""
    (out / "PAPER_FIGURE_README.txt").write_text(readme)

    summary = {
        "gate": "PASS",
        "task_id": args.task_id,
        "profile_id": selected,
        "task_files": int(len(index_df)),
        "unresolved_profile_files": int(len(unresolved)),
        "selected_files": int(len(selected_files)),
        "primary_rows": int(len(d)),
        "primary_columns": int(len(d.columns)),
        "marker_triplets": int(len(triplets)),
        "diagnostic_created": bool(diag),
        "no_simulator_rerun": True,
        "no_training": True,
        "no_inference": True,
    }
    (out / "EXTRACTION_SUMMARY.json").write_text(json.dumps(summary, indent=2))

    print("\n[7/7] Building upload ZIP...")
    zip_folder(out, args.zip_path)
    zsha = sha256(args.zip_path)
    sha_path = args.zip_path.parent / f"{args.zip_path.name}.sha256"
    sha_path.write_text(f"{zsha}  {args.zip_path}\n")

    print("\n" + "="*118)
    print("HEIGHT-FALL PAPER EVIDENCE EXTRACTION V5: PASS")
    print("="*118)
    print("Selected:", f"Task {args.task_id}, {selected}")
    print("ZIP:", args.zip_path)
    print("ZIP SHA256:", zsha)
    print("NO SIMULATOR RERUN, TRAINING, OR INFERENCE WAS PERFORMED.")

if __name__ == "__main__":
    main()
