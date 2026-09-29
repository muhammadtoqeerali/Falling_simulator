#!/usr/bin/env python3
"""
collect_task39_phase_and_replay_evidence_v1.py

Goal
----
Build the exact evidence package needed before creating a publication figure for
Task 39 (forward fall from height).

This script does TWO things:

A) Recorded-data phase evidence
   - reads all final corrected396 Task-39 CSVs correctly with comment='#'
   - summarizes all 22 profiles
   - selects a transparent representative profile (not simply alphabetical)
   - extracts exact/derived stages:
       SETUP
       FALL_ONSET
       MAX_DESCENT
       FIRST_CONTACT
       PEAK_IMPACT
       POST_IMPACT
       REST
   - saves main 100-Hz-like stream + high-rate physics-truth stream
   - saves exact values at each phase
   - saves pelvis / impact / IMU curves and diagnostic plots

B) Exact replay-source discovery
   - audits the project source tree for the actual simulator runner / Task-39 logic
   - saves ranked source-file hits and relevant code excerpts
   - copies the most relevant small source files into the evidence package

Why B is required
-----------------
The currently recorded corrected396 CSVs contain pelvis height, sensor position/
velocity, IMU, impact force/magnitude, jerk, and physics-truth acceleration/gyro,
but they do NOT expose full joint/body pose (qpos/qvel/marker/body xyz) in the
columns observed so far.

Therefore a scientifically exact skeleton/whole-body phase figure will require
either:
  1) another already-existing pose/state artifact found by this audit, or
  2) one instrumented deterministic replay of the exact selected scenario that
     records qpos/body positions/contact details.

This script DOES NOT rerun the simulator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
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
ROOT = PROJECT / "outputs/_highrate_overnight/campaign_highrate_truth_v3_rne_corrected396"
EVENT_MANIFEST = PROJECT / "outputs/phase2_publication_gate_audit_20260826/final_synthetic_event_policy_v2/canonical_synthetic_event_manifest_v2.csv"

TASK = 39

TASK_ALIASES = ["task_id", "task", "scenario_id", "scenario", "task_number"]
PROFILE_ALIASES = ["profile_id", "profile", "subject_id", "subject"]
ONSET_ALIASES = ["fall_onset_time", "onset_time", "fall_start_time", "t_onset", "fall_onset", "onset"]
IMPACT_ALIASES = ["impact_time", "peak_impact_time", "recovered_impact_time", "t_impact", "impact"]
REST_ALIASES = ["rest_time", "settle_time", "settled_time", "t_rest"]

def norm(x):
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())

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

def canonical_profile(v):
    if v is None:
        return None
    s = str(v).strip()
    m = re.search(r"(?i)(?:^|[^A-Za-z0-9])P[_\- ]*0*(\d{1,4})(?:$|[^A-Za-z0-9])", s)
    if m:
        return f"P{int(m.group(1)):03d}"
    try:
        return f"P{int(float(s)):03d}"
    except Exception:
        return s

def sha256(path: Path, block=1024*1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def read_comment_csv(path: Path):
    return pd.read_csv(path, comment="#", low_memory=False)

def read_preamble(path: Path):
    meta = {}
    with path.open("r", encoding="ascii", errors="replace") as f:
        for line in f:
            if not line.startswith("#"):
                break
            txt = line[1:].strip()
            if ":" in txt:
                k, v = txt.split(":", 1)
                meta[k.strip()] = v.strip()
    return meta

def path_profile(path: Path):
    for part in reversed(path.parts):
        m = re.fullmatch(r"P(\d{1,4})", part, re.I)
        if m:
            return f"P{int(m.group(1)):03d}"
    return None

def task39_run_dirs():
    dirs = []
    for p in ROOT.glob("runs/P*/task_39/*"):
        if p.is_dir():
            dirs.append(p)
    return sorted(dirs)

def load_manifest():
    d = pd.read_csv(EVENT_MANIFEST, low_memory=False)
    tc = find_col(d.columns, TASK_ALIASES)
    pc = find_col(d.columns, PROFILE_ALIASES)
    if tc is None:
        return pd.DataFrame()
    tv = pd.to_numeric(d[tc], errors="coerce")
    d = d[tv == TASK].copy()
    if pc:
        d["_profile"] = d[pc].map(canonical_profile)
    return d

def manifest_row_for(manifest, profile):
    if manifest.empty:
        return pd.DataFrame()
    if "_profile" in manifest.columns:
        return manifest[manifest["_profile"] == profile].copy()
    return manifest.copy()

def scalar(df, aliases):
    if df is None or df.empty:
        return None, None
    c = find_col(df.columns, aliases)
    if c is None:
        return None, None
    v = pd.to_numeric(df[c], errors="coerce").dropna()
    if v.empty:
        return None, None
    return float(v.iloc[0]), c

def validation_parse(path: Path):
    out = {
        "validation_confidence": np.nan,
        "validation_classification": "",
        "validation_passes": 0,
        "validation_fails": 0,
    }
    if not path.exists():
        return out
    text = path.read_text(errors="replace")
    m = re.search(r"Overall\s+Confidence\s*:\s*([0-9.]+)\s*%?", text, re.I)
    if m:
        out["validation_confidence"] = float(m.group(1))
    m = re.search(r"Classification\s*:\s*([A-Z_]+)", text, re.I)
    if m:
        out["validation_classification"] = m.group(1)
    out["validation_passes"] = len(re.findall(r"\bPASS\b", text, re.I))
    out["validation_fails"] = len(re.findall(r"\bFAIL\b", text, re.I))
    return out

def velocity_mag(df):
    cols = ["sensor_vel_x", "sensor_vel_y", "sensor_vel_z"]
    if all(c in df.columns for c in cols):
        arr = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        return np.linalg.norm(arr, axis=1)
    return np.full(len(df), np.nan)

def accel_mag(df, true=False):
    if true:
        cols = ["accel_true_x", "accel_true_y", "accel_true_z"]
    else:
        cols = ["accel_x", "accel_y", "accel_z"]
    if all(c in df.columns for c in cols):
        arr = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        return np.linalg.norm(arr, axis=1)
    return np.full(len(df), np.nan)

def gyro_mag(df, true=False):
    if true:
        cols = ["gyro_true_x", "gyro_true_y", "gyro_true_z"]
    else:
        cols = ["gyro_x", "gyro_y", "gyro_z"]
    if all(c in df.columns for c in cols):
        arr = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        return np.linalg.norm(arr, axis=1)
    return np.full(len(df), np.nan)

def first_stable_rest_time(df, impact):
    t = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(float)
    vm = velocity_mag(df)
    if not np.isfinite(vm).any():
        return float(t[-1]), "trial_end_no_velocity"

    dt = np.nanmedian(np.diff(t))
    win = max(3, int(round(0.75 / max(dt, 1e-3))))
    start = int(np.searchsorted(t, impact + 0.4))

    # 5 cm/s sustained sensor-speed criterion for 0.75 s.
    for i in range(start, max(start, len(t)-win)):
        x = vm[i:i+win]
        if len(x) == win and np.nanmax(x) <= 0.05:
            return float(t[i]), "derived_sensor_speed_below_0.05mps_for_0.75s"
    return float(t[-1]), "trial_end_no_sustained_rest_found"

def derive_events(df, manifest_row):
    t = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(float)
    t0, t1 = float(t[0]), float(t[-1])
    sources = {}

    onset, oc = scalar(manifest_row, ONSET_ALIASES)
    impact, ic = scalar(manifest_row, IMPACT_ALIASES)
    rest, rc = scalar(manifest_row, REST_ALIASES)

    if onset is not None and t0 <= onset <= t1:
        sources["onset"] = f"canonical_manifest:{oc}"
    else:
        onset = None

    if impact is not None and t0 <= impact <= t1:
        sources["impact"] = f"canonical_manifest:{ic}"
    else:
        impact = None

    if rest is not None and t0 <= rest <= t1:
        sources["rest"] = f"canonical_manifest:{rc}"
    else:
        rest = None

    # Onset fallback: first fall_detected = 1, otherwise strong pelvis descent.
    if onset is None and "fall_detected" in df.columns:
        fd = pd.to_numeric(df["fall_detected"], errors="coerce").fillna(0).to_numpy()
        idx = np.where(fd > 0)[0]
        if len(idx):
            onset = float(t[idx[0]])
            sources["onset"] = "main_csv:first_fall_detected"

    if onset is None and "pelvis_height" in df.columns:
        h = pd.to_numeric(df["pelvis_height"], errors="coerce").to_numpy(float)
        vel = np.gradient(h, t)
        # Use first point below the 5th percentile of vertical velocity after 1 s.
        mask = t >= min(t1, t0+1.0)
        threshold = np.nanpercentile(vel[mask], 5)
        idx = np.where(mask & (vel <= threshold))[0]
        if len(idx):
            onset = float(t[idx[0]])
            sources["onset"] = "derived:first_strong_pelvis_descent"

    # Manuscript's accepted impact construction: peak impact magnitude within
    # 3 s after onset.
    if impact is None:
        if onset is None:
            onset = float(t0)
            sources["onset"] = "fallback_trial_start"
        window = (t >= onset) & (t <= min(t1, onset+3.0))
        if "impact_magnitude" in df.columns and window.any():
            x = pd.to_numeric(df["impact_magnitude"], errors="coerce").to_numpy(float)
            ids = np.where(window)[0]
            j = ids[int(np.nanargmax(x[ids]))]
            impact = float(t[j])
            sources["impact"] = "main_csv:peak_impact_magnitude_within_3s_after_onset"
        elif "impact_force" in df.columns and window.any():
            x = pd.to_numeric(df["impact_force"], errors="coerce").to_numpy(float)
            ids = np.where(window)[0]
            j = ids[int(np.nanargmax(x[ids]))]
            impact = float(t[j])
            sources["impact"] = "main_csv:peak_impact_force_within_3s_after_onset"

    if rest is None:
        rest, reason = first_stable_rest_time(df, impact)
        sources["rest"] = reason

    return float(onset), float(impact), float(rest), sources

def phase_times(df, onset, impact, rest):
    t = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(float)
    h = pd.to_numeric(df["pelvis_height"], errors="coerce").to_numpy(float)
    hf = pd.to_numeric(df["impact_force"], errors="coerce").fillna(0).to_numpy(float)

    setup = max(float(t[0]), onset - 1.0)

    # Max descent
    mask = (t >= onset) & (t <= impact)
    ids = np.where(mask)[0]
    if len(ids) >= 3:
        vel = np.gradient(h, t)
        j = ids[int(np.argmin(vel[ids]))]
        max_descent = float(t[j])
        desc_reason = "maximum_negative_pelvis_vertical_velocity"
    else:
        max_descent = onset
        desc_reason = "onset_fallback"

    # First meaningful contact after onset:
    # threshold based on 5% of the onset->impact maximum force, but at least
    # 1 N. This is recorded as derived, not claimed as a native contact flag.
    preimpact = (t >= onset) & (t <= impact)
    pids = np.where(preimpact)[0]
    if len(pids):
        mx = float(np.nanmax(hf[pids]))
        thr = max(1.0, 0.05*mx)
        cids = pids[hf[pids] >= thr]
        if len(cids):
            first_contact = float(t[cids[0]])
            contact_reason = f"impact_force_first_crossing_{thr:.6g}N"
        else:
            first_contact = max(onset, impact-0.1)
            contact_reason = "no_preimpact_force_crossing_fallback"
    else:
        first_contact = onset
        contact_reason = "empty_onset_impact_window"

    post = min(float(t[-1]), impact + 0.25)

    phases = [
        ("SETUP", setup, "1.0s_before_onset"),
        ("FALL_ONSET", onset, "accepted_event_onset"),
        ("MAX_DESCENT", max_descent, desc_reason),
        ("FIRST_CONTACT", first_contact, contact_reason),
        ("PEAK_IMPACT", impact, "accepted_peak_impact"),
        ("POST_IMPACT", post, "0.25s_after_peak_impact"),
        ("REST", rest, "derived_or_manifest_rest"),
    ]

    # Make monotonic for presentation only.
    out = []
    last = float(t[0])
    for n, tt, reason in phases:
        tt = max(last, min(float(tt), float(t[-1])))
        out.append((n, tt, reason))
        last = tt
    return out

def nearest_row(df, tt):
    t = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(float)
    return int(np.nanargmin(np.abs(t-tt)))

def summarize_run(run_dir, manifest):
    profile = path_profile(run_dir)
    main_files = [
        p for p in run_dir.glob("fall_scenario*.csv")
        if "highrate_truth" not in p.name
    ]
    truth_files = list(run_dir.glob("*_highrate_truth.csv"))
    validations = list(run_dir.glob("*_validation.txt"))
    manifests = list(run_dir.glob("run_manifest.json"))

    if len(main_files) != 1:
        return None

    main = main_files[0]
    df = read_comment_csv(main)
    if "timestamp" not in df.columns or "pelvis_height" not in df.columns:
        return None

    mrow = manifest_row_for(manifest, profile)
    onset, impact, rest, sources = derive_events(df, mrow)
    phases = phase_times(df, onset, impact, rest)

    t = pd.to_numeric(df["timestamp"], errors="coerce")
    first1 = df[t <= float(t.iloc[0])+1.0]
    last1 = df[t >= float(t.iloc[-1])-1.0]

    initial_h = float(pd.to_numeric(first1["pelvis_height"], errors="coerce").median())
    final_h = float(pd.to_numeric(last1["pelvis_height"], errors="coerce").median())
    drop = initial_h-final_h

    imp = pd.to_numeric(df["impact_magnitude"], errors="coerce") if "impact_magnitude" in df.columns else pd.Series(dtype=float)
    force = pd.to_numeric(df["impact_force"], errors="coerce") if "impact_force" in df.columns else pd.Series(dtype=float)
    jerk = pd.to_numeric(df["jerk_mag"], errors="coerce") if "jerk_mag" in df.columns else pd.Series(dtype=float)
    conf = pd.to_numeric(df["sensor_confidence"], errors="coerce") if "sensor_confidence" in df.columns else pd.Series(dtype=float)

    meta = read_preamble(main)
    val = validation_parse(validations[0]) if validations else validation_parse(Path("/nonexistent"))

    return {
        "profile_id": profile,
        "run_dir": str(run_dir),
        "main_csv": str(main),
        "truth_csv": str(truth_files[0]) if truth_files else "",
        "run_manifest": str(manifests[0]) if manifests else "",
        "validation_txt": str(validations[0]) if validations else "",
        "age": pd.to_numeric(pd.Series([meta.get("age")]), errors="coerce").iloc[0],
        "height_m": pd.to_numeric(pd.Series([meta.get("height_m")]), errors="coerce").iloc[0],
        "sex": meta.get("sex", ""),
        "weight_kg": pd.to_numeric(pd.Series([meta.get("weight_kg")]), errors="coerce").iloc[0],
        "duration_s": float(t.iloc[-1]-t.iloc[0]),
        "onset_s": onset,
        "impact_s": impact,
        "rest_s": rest,
        "onset_source": sources.get("onset",""),
        "impact_source": sources.get("impact",""),
        "rest_source": sources.get("rest",""),
        "initial_pelvis_height_m": initial_h,
        "final_pelvis_height_m": final_h,
        "pelvis_drop_m": drop,
        "max_impact_magnitude": float(imp.max()) if len(imp) else np.nan,
        "max_impact_force": float(force.max()) if len(force) else np.nan,
        "max_jerk": float(jerk.max()) if len(jerk) else np.nan,
        "median_sensor_confidence": float(conf.median()) if len(conf) else np.nan,
        **val,
    }

def choose_representative(summary_df):
    """
    Transparent representative selection:
    - require a clear fall (pelvis drop > 0.75 m)
    - require finite impact
    - prefer runs with >=5 validation PASS tokens and no more than 1 FAIL token
    - among eligible runs, choose the one closest to the median of normalized:
        age, height, weight, impact magnitude
    This avoids selecting the largest impact merely because it looks dramatic.
    """
    d = summary_df.copy()
    eligible = d[
        (d["pelvis_drop_m"] > 0.75) &
        d["max_impact_magnitude"].notna()
    ].copy()

    strict = eligible[
        (eligible["validation_passes"] >= 5) &
        (eligible["validation_fails"] <= 1)
    ]
    if len(strict):
        eligible = strict

    cols = ["age", "height_m", "weight_kg", "max_impact_magnitude"]
    score = np.zeros(len(eligible), dtype=float)
    details = {}
    for c in cols:
        x = pd.to_numeric(eligible[c], errors="coerce").to_numpy(float)
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x-med))
        scale = mad if np.isfinite(mad) and mad > 1e-12 else np.nanstd(x)
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        z = np.abs(x-med)/scale
        score += np.nan_to_num(z, nan=10.0)
        details[c] = {"median": float(med), "scale": float(scale)}
    eligible = eligible.copy()
    eligible["representativeness_score"] = score
    eligible = eligible.sort_values(["representativeness_score","profile_id"])
    return eligible.iloc[0], eligible, details

def export_selected(summary_row, manifest, out):
    run_dir = Path(summary_row["run_dir"])
    main = Path(summary_row["main_csv"])
    truth = Path(summary_row["truth_csv"])
    rm = Path(summary_row["run_manifest"])
    val = Path(summary_row["validation_txt"])

    exact = out/"02_SELECTED_EXACT_FILES"
    exact.mkdir(parents=True, exist_ok=True)
    hashes = []
    for p in [main, truth, rm, val]:
        if p.exists():
            dst = exact/p.name
            shutil.copy2(p, dst)
            hashes.append({
                "source": str(p),
                "copied": str(dst),
                "bytes": p.stat().st_size,
                "sha256": sha256(p),
            })
    pd.DataFrame(hashes).to_csv(exact/"source_manifest_sha256.csv", index=False)

    df = read_comment_csv(main)
    tdf = read_comment_csv(truth) if truth.exists() else pd.DataFrame()

    profile = summary_row["profile_id"]
    mrow = manifest_row_for(manifest, profile)
    onset, impact, rest, sources = derive_events(df, mrow)
    phases = phase_times(df, onset, impact, rest)
    pdf = pd.DataFrame(phases, columns=["phase","time_s","derivation"])
    pdf.to_csv(out/"03_PHASES/phase_timestamps.csv", index=False)

    # Per-phase main + nearest truth rows
    rows = []
    for phase, tt, reason in phases:
        i = nearest_row(df, tt)
        rec = {
            "phase": phase,
            "requested_time_s": tt,
            "main_sample_time_s": float(df.iloc[i]["timestamp"]),
            "derivation": reason,
        }
        for c in df.columns:
            if c == "timestamp":
                continue
            v = df.iloc[i][c]
            try:
                rec[f"main__{c}"] = float(v)
            except Exception:
                rec[f"main__{c}"] = str(v)

        if not tdf.empty and "timestamp" in tdf.columns:
            j = int(np.nanargmin(np.abs(
                pd.to_numeric(tdf["timestamp"], errors="coerce").to_numpy(float)-tt
            )))
            rec["truth_sample_time_s"] = float(tdf.iloc[j]["timestamp"])
            for c in tdf.columns:
                if c == "timestamp":
                    continue
                v = tdf.iloc[j][c]
                try:
                    rec[f"truth__{c}"] = float(v)
                except Exception:
                    rec[f"truth__{c}"] = str(v)
        rows.append(rec)
    pd.DataFrame(rows).to_csv(out/"03_PHASES/phase_snapshot_values.csv", index=False)

    # Compact curve exports
    curve = pd.DataFrame({
        "timestamp": pd.to_numeric(df["timestamp"], errors="coerce"),
        "pelvis_height": pd.to_numeric(df["pelvis_height"], errors="coerce"),
        "impact_force": pd.to_numeric(df["impact_force"], errors="coerce"),
        "impact_magnitude": pd.to_numeric(df["impact_magnitude"], errors="coerce"),
        "jerk_mag": pd.to_numeric(df["jerk_mag"], errors="coerce"),
        "sensor_vel_mag": velocity_mag(df),
        "accel_mag": accel_mag(df, false) if False else accel_mag(df, False),
        "accel_true_mag_main": accel_mag(df, True),
        "gyro_mag": gyro_mag(df, False),
    })
    curve.to_csv(out/"04_RECORDED_CURVES/main_figure_curves.csv", index=False)

    if not tdf.empty:
        truth_curve = tdf.copy()
        truth_curve.to_csv(out/"04_RECORDED_CURVES/highrate_truth_full.csv", index=False)

    phase_meta = {
        "profile_id": profile,
        "task_id": TASK,
        "event_sources": sources,
        "phases": [
            {"phase": p, "time_s": t, "derivation": r}
            for p,t,r in phases
        ],
    }
    (out/"03_PHASES/phase_metadata.json").write_text(json.dumps(phase_meta, indent=2))

    # Diagnostic plot
    if plt is not None:
        tt = curve["timestamp"].to_numpy(float)
        fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
        axes[0].plot(tt, curve["pelvis_height"])
        axes[0].set_ylabel("Pelvis height [m]")
        axes[1].plot(tt, curve["impact_force"])
        axes[1].set_ylabel("Impact force")
        axes[2].plot(tt, curve["accel_true_mag_main"])
        axes[2].set_ylabel("Accel true mag")
        axes[3].plot(tt, curve["gyro_mag"])
        axes[3].set_ylabel("Gyro mag")
        axes[3].set_xlabel("Time [s]")
        for ax in axes:
            ax.grid(alpha=0.2)
            for p,t,r in phases:
                ax.axvline(t, ls="--", lw=0.8)
        fig.suptitle(f"Task 39 {profile} recorded phase diagnostic — not final paper figure")
        fig.tight_layout()
        fig.savefig(out/"05_DIAGNOSTIC_PREVIEWS/recorded_phase_diagnostic.png",
                    dpi=180, bbox_inches="tight")
        plt.close(fig)

    return df, tdf, phases

SOURCE_KEYWORDS = {
    "task39": [r"\b39\b", r"task_39", r"scenario39", r"scenario_39"],
    "runner": [r"fall_scenario", r"run_manifest", r"campaign_highrate", r"corrected396"],
    "mujoco_step": [r"mj_step\(", r"mj_step1\(", r"mj_step2\("],
    "corrected_accel": [r"mj_rnePostConstraint", r"mj_objectAcceleration"],
    "state": [r"\bqpos\b", r"\bqvel\b", r"xpos", r"xquat", r"body.*position", r"joint"],
    "contact": [r"\bncon\b", r"mj_contactForce", r"contact", r"geom1", r"geom2"],
    "render": [r"Renderer", r"mujoco\.viewer", r"render\(", r"camera"],
    "controller": [r"Meta.?Motivo", r"HumEnv", r"motivo", r"policy"],
    "height_scene": [r"height", r"platform", r"elevation", r"ladder"],
}

def source_audit(project, out):
    audit = out/"06_REPLAY_SOURCE_AUDIT"
    audit.mkdir(parents=True, exist_ok=True)

    exclude_parts = {
        ".git", ".venv", ".venv-mjpc", "__pycache__", "outputs",
        "node_modules", "site-packages",
    }

    candidates = []
    for p in project.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".py",".sh",".json",".yaml",".yml"}:
            continue
        if any(part in exclude_parts for part in p.parts):
            continue
        if p.stat().st_size > 2_000_000:
            continue
        try:
            text = p.read_text(errors="replace")
        except Exception:
            continue

        category_hits = {}
        score = 0
        for cat, pats in SOURCE_KEYWORDS.items():
            hits = 0
            for pat in pats:
                hits += len(re.findall(pat, text, re.I))
            if hits:
                category_hits[cat] = hits
                # emphasize exact runtime evidence
                weights = {
                    "task39": 8, "runner": 7, "mujoco_step": 10,
                    "corrected_accel": 12, "state": 5, "contact": 6,
                    "render": 3, "controller": 7, "height_scene": 3,
                }
                score += min(hits,10)*weights[cat]
        if score:
            candidates.append({
                "path": str(p.relative_to(project)),
                "abs_path": str(p),
                "bytes": p.stat().st_size,
                "score": score,
                **{f"hits_{k}": category_hits.get(k,0) for k in SOURCE_KEYWORDS},
            })

    cdf = pd.DataFrame(candidates)
    if cdf.empty:
        cdf.to_csv(audit/"ranked_source_files.csv", index=False)
        return cdf

    cdf = cdf.sort_values(["score","path"], ascending=[False,True]).reset_index(drop=True)
    cdf.to_csv(audit/"ranked_source_files.csv", index=False)

    excerpt_lines = []
    copydir = audit/"top_source_files"
    copydir.mkdir(parents=True, exist_ok=True)

    for rank, row in cdf.head(20).iterrows():
        p = Path(row["abs_path"])
        text = p.read_text(errors="replace")
        lines = text.splitlines()

        excerpt_lines.append("="*120)
        excerpt_lines.append(f"RANK {rank+1} SCORE {row['score']} FILE {row['path']}")
        excerpt_lines.append("="*120)

        hit_lines = set()
        for pats in SOURCE_KEYWORDS.values():
            for pat in pats:
                for i,line in enumerate(lines):
                    if re.search(pat, line, re.I):
                        for j in range(max(0,i-4), min(len(lines),i+5)):
                            hit_lines.add(j)

        # collapse and limit excerpts
        for j in sorted(hit_lines)[:500]:
            excerpt_lines.append(f"{j+1:06d}: {lines[j]}")
        excerpt_lines.append("")

        if p.stat().st_size <= 500_000:
            safe_name = str(row["path"]).replace("/", "__")
            shutil.copy2(p, copydir/safe_name)

    (audit/"source_excerpts.txt").write_text("\n".join(excerpt_lines))
    return cdf

def zip_folder(folder, zpath):
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(zpath,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(folder.parent)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path,
                    default=PROJECT/"outputs/task39_phase_and_replay_evidence_v1")
    ap.add_argument("--zip-path", type=Path,
                    default=PROJECT/"outputs/task39_phase_and_replay_evidence_v1.zip")
    args = ap.parse_args()

    out = args.output_dir
    if out.exists():
        shutil.rmtree(out)
    for sub in [
        "00_AUDIT","01_ALL_PROFILES","02_SELECTED_EXACT_FILES","03_PHASES",
        "04_RECORDED_CURVES","05_DIAGNOSTIC_PREVIEWS","06_REPLAY_SOURCE_AUDIT"
    ]:
        (out/sub).mkdir(parents=True, exist_ok=True)

    print("="*120)
    print("TASK 39 PHASE + EXACT REPLAY DISCOVERY")
    print("="*120)

    manifest = load_manifest()
    print("Canonical Task-39 manifest rows:", len(manifest))

    summaries = []
    runs = task39_run_dirs()
    print("Corrected396 Task-39 run directories:", len(runs))

    for run in runs:
        try:
            s = summarize_run(run, manifest)
            if s is not None:
                summaries.append(s)
        except Exception as e:
            summaries.append({
                "profile_id": path_profile(run),
                "run_dir": str(run),
                "summary_error": repr(e),
            })

    sdf = pd.DataFrame(summaries)
    sdf.to_csv(out/"01_ALL_PROFILES/task39_profile_summary.csv", index=False)

    good = sdf[sdf.get("summary_error", pd.Series("",index=sdf.index)).fillna("").eq("")].copy()
    print("Successfully parsed profiles:", len(good))
    if good.empty:
        raise SystemExit("ERROR: no Task-39 profile parsed successfully.")

    selected, ranked, selection_meta = choose_representative(good)
    ranked.to_csv(out/"01_ALL_PROFILES/representative_ranking.csv", index=False)
    (out/"01_ALL_PROFILES/selection_method.json").write_text(json.dumps({
        "method": "closest_to_median_age_height_weight_impact_among_clear_valid_falls",
        "details": selection_meta,
        "selected_profile": selected["profile_id"],
    }, indent=2))

    print("\nSELECTED REPRESENTATIVE PROFILE")
    print(selected.to_string())

    df, tdf, phases = export_selected(selected, manifest, out)

    # Explicitly document currently observed recorded-state limitations.
    recorded_cols = list(df.columns)
    missing = {
        "qpos_present": any("qpos" in norm(c) for c in recorded_cols),
        "qvel_present": any("qvel" in norm(c) for c in recorded_cols),
        "joint_columns_present": any("joint" in norm(c) for c in recorded_cols),
        "body_marker_xyz_present": any(
            ("body" in norm(c) or "marker" in norm(c) or "knee" in norm(c) or "ankle" in norm(c))
            and re.search(r"(?:_|\.|-)[xyz]$", str(c), re.I)
            for c in recorded_cols
        ),
        "pelvis_height_present": "pelvis_height" in df.columns,
        "impact_force_present": "impact_force" in df.columns,
        "impact_magnitude_present": "impact_magnitude" in df.columns,
        "sensor_position_present": all(c in df.columns for c in ["sensor_pos_x","sensor_pos_y","sensor_pos_z"]),
        "sensor_velocity_present": all(c in df.columns for c in ["sensor_vel_x","sensor_vel_y","sensor_vel_z"]),
    }
    (out/"00_AUDIT/recorded_pose_state_inventory.json").write_text(json.dumps(missing, indent=2))

    print("\nRECORDED POSE/STATE INVENTORY")
    print(json.dumps(missing, indent=2))

    print("\nScanning project source for exact replay entrypoint/state/contact hooks...")
    source_df = source_audit(PROJECT, out)
    print("Relevant source files found:", len(source_df))
    if len(source_df):
        print(source_df.head(15)[[
            "score","path","hits_task39","hits_runner","hits_mujoco_step",
            "hits_corrected_accel","hits_state","hits_contact","hits_controller"
        ]].to_string(index=False))

    readiness = {
        "recorded_data_sufficient_for_kinematic_contact_imu_panel": bool(
            missing["pelvis_height_present"] and
            missing["impact_force_present"] and
            missing["sensor_position_present"]
        ),
        "recorded_data_sufficient_for_exact_whole_body_skeleton": bool(
            missing["qpos_present"] or
            missing["joint_columns_present"] or
            missing["body_marker_xyz_present"]
        ),
        "next_step": (
            "DIRECT_FIGURE_BUILD"
            if (
                missing["qpos_present"] or
                missing["joint_columns_present"] or
                missing["body_marker_xyz_present"]
            )
            else "BUILD_INSTRUMENTED_REPLAY_FROM_SOURCE_AUDIT"
        )
    }
    (out/"FIGURE_READINESS.json").write_text(json.dumps(readiness, indent=2))

    readme = f"""TASK 39 PAPER FIGURE EVIDENCE

Selected representative profile:
{selected["profile_id"]}

Selection:
transparent central/representative trial selection, not maximum-impact cherry-picking.
See 01_ALL_PROFILES/selection_method.json and representative_ranking.csv.

Recorded phases:
{pd.DataFrame(phases, columns=["phase","time_s","derivation"]).to_string(index=False)}

Recorded pose/state inventory:
{json.dumps(missing, indent=2)}

Figure readiness:
{json.dumps(readiness, indent=2)}

Interpretation:
The corrected CSVs are sufficient for pelvis trajectory, impact/contact proxy,
sensor motion, virtual IMU, jerk, and phase timing. They are NOT sufficient for
an exact anatomical skeleton unless qpos/joint/body-marker data are found.

06_REPLAY_SOURCE_AUDIT/ contains the project-source evidence needed to build one
exact instrumented replay script that records qpos/body xyz/contact identities
and renders the selected phase snapshots without inventing body motion.

THIS SCRIPT DOES NOT RERUN THE SIMULATOR.
"""
    (out/"README_FOR_FIGURE_BUILD.txt").write_text(readme)

    zip_folder(out,args.zip_path)
    zsha=sha256(args.zip_path)
    (args.zip_path.parent/f"{args.zip_path.name}.sha256").write_text(
        f"{zsha}  {args.zip_path}\n"
    )

    print("\nFIGURE READINESS")
    print(json.dumps(readiness, indent=2))
    print("\n"+"="*120)
    print("TASK 39 PHASE + REPLAY DISCOVERY: PASS")
    print("="*120)
    print("ZIP:", args.zip_path)
    print("ZIP SHA256:", zsha)

if __name__ == "__main__":
    main()
