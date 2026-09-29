#!/usr/bin/env python3
"""
prepare_height_fall_paper_evidence_v6.py

Robust standalone extractor for the final corrected396 Task-39 height fall.

Key fixes over V5
-----------------
1. PATH-AUTHORITATIVE profile/task matching:
   if the directory is .../runs/P020/task_39/..., P020 and Task 39 win.
   JSON/CSV metadata may supplement the path but never override it.

2. ROBUST CSV READER:
   probes encoding/delimiter/quoting and records exact parser diagnostics.
   It does not assume ordinary comma-separated pandas CSV.

3. FILENAME-AUTHORITATIVE artifact typing:
   *_highrate_truth.csv is classified as high-rate truth even before parsing.

4. IMPLICIT AXIS SUPPORT:
   if no named time column exists, a monotonic frame/sample/step/index column
   can be converted to time using manifest sample-rate/dt metadata.
   The project-protocol 450-Hz high-rate value is used only as a clearly
   recorded last-resort for *_highrate_truth.csv.

5. FIGURE-READINESS AUDIT:
   exports exact event/phase timestamps, kinematics, contact/impact channels,
   IMU channels, marker xyz triplets, qpos/qvel-like state information, and
   declares whether exact skeleton rendering is possible from recorded data.

This script DOES NOT rerun MuJoCo, train the CNN, or run inference.
"""

from __future__ import annotations

import argparse
import ast
import csv
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

TASK_ALIASES = [
    "task_id", "task", "taskid", "activity_id", "scenario_id", "scenario",
    "task_number"
]
PROFILE_ALIASES = [
    "profile_id", "profile", "profileid", "subject_id", "subject", "avatar_id",
    "human_id"
]
TIME_ALIASES = [
    "time", "time_s", "timestamp", "timestamp_s", "sim_time", "sim_time_s",
    "time_sec", "seconds", "elapsed_time", "elapsed_s", "t"
]
FRAME_ALIASES = [
    "frame", "frame_id", "frame_idx", "frame_index", "sample", "sample_id",
    "sample_idx", "sample_index", "step", "step_id", "physics_step",
    "sensor_step", "tick", "index"
]
ONSET_ALIASES = [
    "fall_onset_time", "onset_time", "fall_start_time", "start_fall_time",
    "t_onset", "fall_onset", "onset"
]
IMPACT_ALIASES = [
    "impact_time", "peak_impact_time", "recovered_impact_time", "t_impact",
    "impact"
]
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

def canonical_profile_id(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    for pat in [
        r"(?i)^P[_\- .]*0*(\d{1,4})$",
        r"(?i)^(?:profile|prof)[_\- .]*(?:P[_\- .]*)?0*(\d{1,4})$",
        r"(?i)^(?:subject|subj)[_\- .]*(?:P[_\- .]*)?0*(\d{1,4})$",
        r"^0*(\d{1,4})$",
    ]:
        m = re.match(pat, s)
        if m:
            return f"P{int(m.group(1)):03d}"
    return s

def path_task_profile(path: Path):
    """Strict path parser; explicit P### and task_### tokens only."""
    parts = [path.name] + list(reversed(path.parts[-9:-1]))
    profile = None
    task = None

    ppats = [
        re.compile(r"(?<![A-Za-z0-9])P[_\- .]*0*(\d{1,4})(?![A-Za-z0-9])", re.I),
        re.compile(r"(?<![A-Za-z0-9])(?:profile|prof)[_\- .]*(?:P[_\- .]*)?0*(\d{1,4})(?![A-Za-z0-9])", re.I),
    ]
    tpats = [
        re.compile(r"(?<![A-Za-z0-9])task[_\- .]*0*(\d{1,3})(?!\d)", re.I),
        re.compile(r"(?<![A-Za-z0-9])scenario[_\- .]*0*(\d{1,3})(?!\d)", re.I),
        re.compile(r"(?<![A-Za-z0-9])T[_\- .]*0*(\d{1,3})(?!\d)", re.I),
    ]

    for part in parts:
        s = str(part)
        for pat in ppats:
            m = pat.search(s)
            if m:
                profile = f"P{int(m.group(1)):03d}"
                break
        if profile:
            break

    for part in parts:
        s = str(part)
        for pat in tpats:
            m = pat.search(s)
            if m:
                task = int(m.group(1))
                break
        if task is not None:
            break

    return task, profile

def flatten_json(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            kk = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_json(v, kk))
    elif isinstance(obj, list):
        if len(obj) <= 100 and all(not isinstance(v, (dict, list)) for v in obj):
            out[prefix] = obj
        else:
            out[prefix] = f"<list len={len(obj)}>"
    else:
        out[prefix] = obj
    return out

def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None

def find_col(columns, aliases):
    cmap = {norm(c): c for c in columns}
    for a in aliases:
        na = norm(a)
        if na in cmap:
            return cmap[na]
    for c in columns:
        nc = norm(c)
        for a in aliases:
            na = norm(a)
            if len(na) >= 5 and (na in nc or nc in na):
                return c
    return None

def raw_probe(path: Path):
    b = path.read_bytes()[:131072]
    null_ratio = (b.count(b"\x00") / max(len(b), 1))
    encodings = ["utf-8-sig", "utf-8", "latin-1"]
    decoded = None
    encoding = None
    for enc in encodings:
        try:
            decoded = b.decode(enc)
            encoding = enc
            break
        except Exception:
            continue
    if decoded is None:
        decoded = b.decode("latin-1", errors="replace")
        encoding = "latin-1-replace"

    lines = [ln for ln in decoded.splitlines() if ln.strip()][:20]
    counts = {}
    for sep_name, sep in [("comma", ","), ("semicolon", ";"), ("tab", "\t"), ("pipe", "|")]:
        if lines:
            counts[sep_name] = {
                "median": float(np.median([ln.count(sep) for ln in lines])),
                "min": int(min(ln.count(sep) for ln in lines)),
                "max": int(max(ln.count(sep) for ln in lines)),
            }
        else:
            counts[sep_name] = {"median": 0, "min": 0, "max": 0}

    return {
        "bytes_probed": len(b),
        "null_ratio": null_ratio,
        "encoding_guess": encoding,
        "first_lines": lines[:8],
        "delimiter_counts": counts,
    }

def robust_read_table(path: Path, diag_dir: Path):
    """
    Return (df, parser_metadata). Never hides the parser attempts.
    """
    diag_dir.mkdir(parents=True, exist_ok=True)
    probe = raw_probe(path)
    (diag_dir / f"{path.name}__raw_probe.json").write_text(json.dumps(probe, indent=2))

    attempts = []
    candidates = []

    # Delimiter ordering from raw evidence.
    delim_map = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}
    ranked = sorted(
        delim_map,
        key=lambda s: probe["delimiter_counts"][delim_map[s]]["median"],
        reverse=True
    )

    # Also allow whitespace and pandas sniffer.
    sep_specs = [(s, f"literal:{repr(s)}") for s in ranked]
    sep_specs += [(None, "python_sniffer"), (r"\s+", "regex_whitespace")]

    encodings = []
    if probe["encoding_guess"] and "replace" not in probe["encoding_guess"]:
        encodings.append(probe["encoding_guess"])
    encodings += ["utf-8-sig", "utf-8", "latin-1"]
    encodings = list(dict.fromkeys(encodings))

    for enc in encodings:
        for sep, label in sep_specs:
            for quoting_mode, quoting_value in [
                ("normal", csv.QUOTE_MINIMAL),
                ("no_quote", csv.QUOTE_NONE),
            ]:
                kwargs = dict(
                    filepath_or_buffer=path,
                    engine="python",
                    encoding=enc,
                    on_bad_lines="skip",
                    dtype_backend=None,
                )
                if sep is not None:
                    kwargs["sep"] = sep
                else:
                    kwargs["sep"] = None
                if quoting_mode == "no_quote":
                    kwargs["quoting"] = quoting_value
                    kwargs["escapechar"] = "\\"

                try:
                    # dtype_backend=None is not accepted by older pandas; remove if needed.
                    try:
                        df = pd.read_csv(**kwargs)
                    except TypeError:
                        kwargs.pop("dtype_backend", None)
                        df = pd.read_csv(**kwargs)

                    rec = {
                        "encoding": enc,
                        "separator": label,
                        "quoting": quoting_mode,
                        "status": "PASS",
                        "rows": int(len(df)),
                        "columns": int(len(df.columns)),
                        "column_names": [str(c) for c in df.columns[:80]],
                    }
                    attempts.append(rec)

                    # Require more than one meaningful column unless the raw file itself
                    # clearly looks single-column.
                    score = len(df.columns) * 100000 + len(df)
                    if len(df.columns) > 1:
                        candidates.append((score, df, rec))
                except Exception as e:
                    attempts.append({
                        "encoding": enc,
                        "separator": label,
                        "quoting": quoting_mode,
                        "status": "FAIL",
                        "error": repr(e)[:1000],
                    })

    (diag_dir / f"{path.name}__parser_attempts.json").write_text(json.dumps(attempts, indent=2))

    if not candidates:
        return None, {
            "status": "READ_FAIL",
            "raw_probe": probe,
            "attempts_file": str(diag_dir / f"{path.name}__parser_attempts.json"),
        }

    candidates.sort(reverse=True, key=lambda x: x[0])
    df, meta = candidates[0][1], candidates[0][2]
    meta = dict(meta)
    meta["status"] = "PASS"
    return df, meta

def classify_kind_by_name(path: Path):
    n = path.name.lower()
    if "highrate_truth" in n:
        return "highrate_truth"
    if "validation" in n:
        return "validation"
    if "manifest" in n or path.suffix.lower() == ".json":
        return "manifest"
    if path.suffix.lower() == ".csv":
        return "timeseries"
    if path.suffix.lower() in STATE_EXT:
        return "state"
    if path.suffix.lower() in FIG_EXT:
        return "figure"
    return "other"

def discover_exact_task_files(root: Path, task_id: int):
    rows = []
    exts = TEXT_EXT | STATE_EXT | FIG_EXT
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        task, profile = path_task_profile(p)
        if task != task_id or profile is None:
            continue
        rows.append({
            "path": str(p),
            "task_id": task,
            "profile_id": profile,
            "suffix": p.suffix.lower(),
            "bytes": p.stat().st_size,
            "kind": classify_kind_by_name(p),
        })
    return pd.DataFrame(rows)

def load_event_manifest(path: Path, task_id: int):
    # canonical manifest is ordinary CSV in the final pipeline; still use a small
    # robust fallback if needed.
    try:
        d = pd.read_csv(path, low_memory=False)
    except Exception:
        d, _ = robust_read_table(path, path.parent / "_tmp_v6_manifest_diag")
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

def candidate_table(index_df, manifest):
    profiles = sorted(index_df["profile_id"].unique().tolist())
    rows = []
    for p in profiles:
        f = index_df[index_df["profile_id"] == p]
        kinds = f["kind"].value_counts().to_dict()
        eligible = False
        if not manifest.empty and "_canonical_profile_id" in manifest.columns:
            eligible = bool((manifest["_canonical_profile_id"] == p).any())
        score = (
            (100 if eligible else 0)
            + 30*kinds.get("highrate_truth", 0)
            + 15*kinds.get("timeseries", 0)
            + 5*kinds.get("manifest", 0)
            + 3*kinds.get("validation", 0)
            + 2*kinds.get("state", 0)
            + len(f)
        )
        rows.append({
            "profile_id": p,
            "eligible_in_event_manifest": eligible,
            "score": score,
            "n_files": len(f),
            "n_highrate_truth": kinds.get("highrate_truth", 0),
            "n_timeseries": kinds.get("timeseries", 0),
            "n_manifest": kinds.get("manifest", 0),
            "n_validation": kinds.get("validation", 0),
            "n_state": kinds.get("state", 0),
        })
    return pd.DataFrame(rows).sort_values(
        ["eligible_in_event_manifest", "n_highrate_truth", "n_timeseries", "score", "profile_id"],
        ascending=[False, False, False, False, True]
    ).reset_index(drop=True)

def parse_numeric(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except Exception:
        return None

def manifest_rate_metadata(manifest_path: Path):
    obj = read_json(manifest_path)
    if obj is None:
        return {}
    flat = flatten_json(obj)
    out = {}
    for k, v in flat.items():
        nk = norm(k)
        x = parse_numeric(v)
        if x is None:
            continue
        if any(tok in nk for tok in ["highratehz", "physicshz", "physicsrate", "samplerate", "samplingrate", "sensorhz", "ratehz", "frequency"]):
            out[k] = x
        if any(tok in nk for tok in ["timestep", "dt", "physicsdt"]):
            out[k] = x
    return out

def choose_rate(rate_meta, highrate):
    # Prefer high-rate/physics metadata for the highrate truth file.
    best = None
    source = None
    for k, v in rate_meta.items():
        nk = norm(k)
        if highrate and any(tok in nk for tok in ["highrate", "physics"]):
            if v > 1:
                return float(v), f"run_manifest:{k}"
            if 0 < v < 1:
                return float(1.0/v), f"run_manifest_inverse_dt:{k}"

    for k, v in rate_meta.items():
        nk = norm(k)
        if any(tok in nk for tok in ["samplerate", "samplingrate", "sensorhz", "ratehz", "frequency"]):
            if v > 1:
                best, source = float(v), f"run_manifest:{k}"
                break
        if any(tok in nk for tok in ["timestep", "dt"]):
            if 0 < v < 1:
                best, source = float(1.0/v), f"run_manifest_inverse_dt:{k}"
                break

    if best is not None:
        return best, source

    if highrate:
        return 450.0, "final_project_protocol_fallback:approx_450Hz"
    return 100.0, "final_project_protocol_fallback:100Hz"

def numeric_axis_candidates(df):
    rows = []
    if len(df) < 3:
        return pd.DataFrame()
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        valid = s.dropna()
        if len(valid) < max(3, int(0.9*len(df))):
            continue
        arr = valid.to_numpy(dtype=float)
        diff = np.diff(arr)
        if len(diff) == 0:
            continue
        inc = bool(np.all(diff >= 0))
        strict = bool(np.all(diff > 0))
        uniq = float(valid.nunique()/max(len(valid),1))
        lexical = 0
        nc = norm(c)
        if find_col([c], TIME_ALIASES):
            lexical += 100
        if find_col([c], FRAME_ALIASES):
            lexical += 80
        score = lexical + (25 if strict else (10 if inc else 0)) + (10 if uniq > 0.95 else 0)
        rows.append({
            "column": c,
            "score": score,
            "strictly_increasing": strict,
            "monotonic": inc,
            "unique_ratio": uniq,
            "first": float(valid.iloc[0]),
            "last": float(valid.iloc[-1]),
            "median_step": float(np.median(diff)),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["score", "column"], ascending=[False, True])

def ensure_time_axis(df, rate_hz, rate_source):
    """
    Returns df, time_col, axis_metadata.
    """
    exact = find_col(df.columns, TIME_ALIASES)
    if exact is not None:
        s = pd.to_numeric(df[exact], errors="coerce")
        if s.notna().sum() >= max(3, int(0.9*len(df))):
            out = df.loc[s.notna()].copy()
            out[exact] = s[s.notna()].astype(float)
            out = out.sort_values(exact).drop_duplicates(exact).reset_index(drop=True)
            return out, exact, {
                "axis_type": "explicit_time",
                "source_column": exact,
                "rate_hz": None,
                "rate_source": None,
            }

    candidates = numeric_axis_candidates(df)
    # Prefer lexical frame/sample/index candidates.
    if not candidates.empty:
        for _, r in candidates.iterrows():
            c = str(r["column"])
            if find_col([c], FRAME_ALIASES) is not None and bool(r["strictly_increasing"]):
                axis = pd.to_numeric(df[c], errors="coerce")
                mask = axis.notna()
                out = df.loc[mask].copy()
                vals = axis[mask].astype(float)
                out["time_s_derived"] = (vals - float(vals.iloc[0])) / float(rate_hz)
                out = out.sort_values("time_s_derived").reset_index(drop=True)
                return out, "time_s_derived", {
                    "axis_type": "derived_from_monotonic_frame_sample_step",
                    "source_column": c,
                    "rate_hz": rate_hz,
                    "rate_source": rate_source,
                }

    # Final implicit row-index fallback, explicitly flagged. This is acceptable
    # for extracting exact sample states at known event-frame fractions only, but
    # not ideal. We record it prominently.
    out = df.copy().reset_index(drop=True)
    out["time_s_derived"] = np.arange(len(out), dtype=float) / float(rate_hz)
    return out, "time_s_derived", {
        "axis_type": "derived_from_row_index",
        "source_column": "<row_index>",
        "rate_hz": rate_hz,
        "rate_source": rate_source,
        "warning": "No explicit time/frame/sample column was recognized; time was reconstructed from row order."
    }

def parse_vector(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        try:
            return [float(x) for x in v]
        except Exception:
            return None
    if not isinstance(v, str):
        return None
    s = v.strip()
    if not s:
        return None
    for method in ("ast", "json"):
        try:
            arr = ast.literal_eval(s) if method == "ast" else json.loads(s)
            if isinstance(arr, (list, tuple)) and len(arr) >= 2:
                return [float(x) for x in arr]
        except Exception:
            pass
    # whitespace/bracket fallback
    ss = s.strip("[]()")
    parts = [p for p in re.split(r"[\s,;]+", ss) if p]
    if len(parts) >= 2:
        try:
            return [float(x) for x in parts]
        except Exception:
            return None
    return None

def inspect_array_columns(df, max_rows=20):
    rows = []
    for c in df.columns:
        if df[c].dtype != object:
            continue
        lengths = []
        examples = []
        for v in df[c].dropna().head(max_rows):
            arr = parse_vector(v)
            if arr is not None:
                lengths.append(len(arr))
                if not examples:
                    examples = arr[:10]
        if lengths:
            rows.append({
                "column": c,
                "parsed_rows": len(lengths),
                "modal_length": int(pd.Series(lengths).mode().iloc[0]),
                "min_length": int(min(lengths)),
                "max_length": int(max(lengths)),
                "example": json.dumps(examples),
            })
    return pd.DataFrame(rows)

def expand_vector3_columns(df):
    out = df.copy()
    vector_map = {}
    for c in list(df.columns):
        if df[c].dtype != object:
            continue
        parsed = []
        ok = 0
        for v in df[c].head(min(len(df), 50)):
            arr = parse_vector(v)
            parsed.append(arr)
            if arr is not None and len(arr) == 3:
                ok += 1
        if ok < max(3, int(0.7*max(1, len(parsed)))):
            continue

        all_arr = [parse_vector(v) for v in df[c]]
        xyz = np.full((len(df), 3), np.nan, dtype=float)
        for i, arr in enumerate(all_arr):
            if arr is not None and len(arr) == 3:
                xyz[i] = arr
        for j, ax in enumerate("xyz"):
            newc = f"{c}_{ax}"
            out[newc] = xyz[:, j]
        vector_map[c] = [f"{c}_{a}" for a in "xyz"]
    return out, vector_map

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
    return {base: ax for base, ax in axes.items() if all(a in ax for a in "xyz")}

def pelvis_height_col(columns):
    c = find_col(columns, ["pelvis_height", "pelvis_z", "pelvis_pos_z", "pelvis_position_z"])
    if c:
        return c
    for col in columns:
        nc = norm(col)
        if "pelvis" in nc and ("height" in nc or nc.endswith("z") or "posz" in nc):
            return col
    return None

def trunk_lean_col(columns):
    c = find_col(columns, ["trunk_lean", "trunk_lean_deg", "torso_lean", "trunk_angle", "torso_pitch"])
    if c:
        return c
    for col in columns:
        nc = norm(col)
        if ("trunk" in nc or "torso" in nc) and any(k in nc for k in ["lean", "pitch", "angle"]):
            return col
    return None

def classified_columns(columns):
    groups = {
        "imu": [],
        "kinematics": [],
        "contact_dynamics": [],
        "qpos_qvel_state": [],
        "events": [],
        "marker_xyz_like": [],
    }
    for c in columns:
        nc = norm(c)
        if any(k in nc for k in ["acc", "gyro", "imu", "angularvel", "angvel"]):
            groups["imu"].append(c)
        if any(k in nc for k in ["pelvis", "trunk", "torso", "com", "head", "height", "lean", "pitch", "roll", "yaw"]):
            groups["kinematics"].append(c)
        if any(k in nc for k in ["contact", "grf", "force", "impact", "load", "normal", "tangent", "friction", "touch"]):
            groups["contact_dynamics"].append(c)
        if any(k in nc for k in ["qpos", "qvel", "jointpos", "jointvel", "jointangle", "bodypos", "bodyquat"]):
            groups["qpos_qvel_state"].append(c)
        if any(k in nc for k in ["onset", "impact", "fallstart", "fallend", "rest", "event"]):
            groups["events"].append(c)
        if re.search(r"(?:_|\.|-)[xyz]$", str(c), re.I):
            groups["marker_xyz_like"].append(c)
    return groups

def scalar_from_df(df, aliases):
    if df is None or df.empty:
        return None, None
    c = find_col(df.columns, aliases)
    if c is None:
        return None, None
    vals = pd.to_numeric(df[c], errors="coerce").dropna()
    if vals.empty:
        return None, None
    return float(vals.iloc[0]), c

def event_rows_for_profile(manifest, profile):
    if manifest.empty:
        return manifest
    if "_canonical_profile_id" in manifest.columns:
        return manifest[manifest["_canonical_profile_id"] == profile].copy()
    return manifest.copy()

def contact_envelope(df, time_col):
    cols = []
    for c in df.columns:
        if c == time_col:
            continue
        nc = norm(c)
        if any(k in nc for k in ["contact", "grf", "groundreaction", "normalforce", "impactmagnitude", "contactload", "primaryimpact", "force", "touch"]):
            cols.append(c)
    arrs, used = [], []
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").abs().to_numpy(dtype=float)
        if np.isfinite(x).sum() >= 10:
            arrs.append(np.nan_to_num(x, nan=0.0))
            used.append(c)
    if not arrs:
        return None, []
    return np.max(np.vstack(arrs), axis=0), used

def resolve_events(manifest_rows, df, time_col):
    onset, oc = scalar_from_df(manifest_rows, ONSET_ALIASES)
    impact, ic = scalar_from_df(manifest_rows, IMPACT_ALIASES)
    rest, rc = scalar_from_df(manifest_rows, REST_ALIASES)
    sources = {}
    if onset is not None:
        sources["onset"] = f"canonical_event_manifest:{oc}"
    if impact is not None:
        sources["impact"] = f"canonical_event_manifest:{ic}"
    if rest is not None:
        sources["rest"] = f"canonical_event_manifest:{rc}"

    if onset is None:
        onset, c = scalar_from_df(df, ONSET_ALIASES)
        if onset is not None:
            sources["onset"] = f"timeseries:{c}"
    if impact is None:
        impact, c = scalar_from_df(df, IMPACT_ALIASES)
        if impact is not None:
            sources["impact"] = f"timeseries:{c}"
    if rest is None:
        rest, c = scalar_from_df(df, REST_ALIASES)
        if rest is not None:
            sources["rest"] = f"timeseries:{c}"

    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    t0, t1 = float(np.nanmin(t)), float(np.nanmax(t))

    # If event times are outside current reconstructed time range (possible if
    # manifest uses absolute timestamps), use relative offsets when plausible.
    for key, val in list([("onset", onset), ("impact", impact), ("rest", rest)]):
        if val is not None and (val < t0 - 1e-6 or val > t1 + 1e-6):
            # Do not silently alter; source will be marked and fallback derivation
            # below will be used if required.
            sources[key + "_warning"] = f"value {val} outside timeseries range [{t0},{t1}]"

    if impact is None or not (t0 <= impact <= t1):
        env, used = contact_envelope(df, time_col)
        if env is not None:
            j = int(np.nanargmax(env))
            impact = float(t[j])
            sources["impact"] = "derived_peak_contact:" + ",".join(used[:6])

    if onset is None or not (t0 <= onset <= t1):
        pcol = pelvis_height_col(df.columns)
        if pcol is not None and impact is not None:
            x = pd.to_numeric(df[pcol], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & (t < impact) & (t >= max(t0, impact-4.0))
            if mask.sum() >= 10:
                tt = t[mask]
                xx = x[mask]
                vel = np.gradient(xx, tt)
                thr = np.nanpercentile(vel, 10)
                idx = np.where(vel <= thr)[0]
                if len(idx):
                    onset = float(tt[idx[0]])
                    sources["onset"] = f"derived_pelvis_descent:{pcol}"

    if impact is None:
        impact = t0 + 0.65*(t1-t0)
        sources["impact"] = "FALLBACK_fraction_trial"
    if onset is None:
        onset = max(t0, impact-1.2)
        sources["onset"] = "FALLBACK_impact_minus_1.2s"
    if rest is None or not (t0 <= rest <= t1):
        rest = t1
        sources["rest"] = "FALLBACK_trial_end"

    onset = float(np.clip(onset, t0, t1))
    impact = float(np.clip(impact, onset, t1))
    rest = float(np.clip(rest, impact, t1))
    return onset, impact, rest, sources

def derive_phases(df, time_col, onset, impact, rest):
    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    t0, t1 = float(np.nanmin(t)), float(np.nanmax(t))

    setup = max(t0, onset - min(1.5, max(0.5, 0.25*max(onset-t0, 0.1))))

    descent = onset + 0.55*max(impact-onset, 0.1)
    descent_reason = "midpoint onset-impact"
    pcol = pelvis_height_col(df.columns)
    if pcol:
        x = pd.to_numeric(df[pcol], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & (t >= onset) & (t <= impact)
        if mask.sum() >= 5:
            tt = t[mask]
            xx = x[mask]
            vel = np.gradient(xx, tt)
            j = int(np.argmin(vel))
            descent = float(tt[j])
            descent_reason = f"max_negative_pelvis_vertical_velocity:{pcol}"

    first_contact = max(onset, impact-0.15)
    contact_reason = "FALLBACK_preimpact_0.15s"
    env, used = contact_envelope(df, time_col)
    if env is not None:
        mask = (t >= onset) & (t <= min(t1, impact+0.5))
        tt, vv = t[mask], env[mask]
        if len(vv):
            base = np.nanmedian(env[t < onset]) if np.any(t < onset) else 0.0
            peak = float(np.nanmax(vv))
            thr = base + 0.10*max(peak-base, 0.0)
            idx = np.where(vv > thr)[0]
            if len(idx):
                first_contact = float(tt[idx[0]])
                contact_reason = "first_contact_envelope_crossing:" + ",".join(used[:6])

    post = min(t1, impact+0.25)

    phases = [
        ("SETUP", setup, "pre_onset_setup"),
        ("FALL_ONSET", onset, "canonical_or_derived_onset"),
        ("MAX_DESCENT", descent, descent_reason),
        ("FIRST_CONTACT", first_contact, contact_reason),
        ("PEAK_IMPACT", impact, "canonical_or_derived_impact"),
        ("POST_IMPACT", post, "impact_plus_0.25s"),
        ("REST", rest, "canonical_rest_or_trial_end"),
    ]

    out, last = [], t0
    for n, tt, why in phases:
        tt = float(np.clip(max(last, tt), t0, t1))
        out.append((n, tt, why))
        last = tt
    return out

def nearest_idx(df, time_col, tt):
    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    return int(np.nanargmin(np.abs(t-tt)))

def serialize_value(v):
    if pd.isna(v) if np.isscalar(v) else False:
        return None
    arr = parse_vector(v)
    if arr is not None:
        return arr
    try:
        x = float(v)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return str(v)

def export_phase_snapshots(df, time_col, phases, csv_path, json_path):
    rows = []
    jrows = []
    for name, tt, why in phases:
        i = nearest_idx(df, time_col, tt)
        row = df.iloc[i]
        flat = {
            "phase": name,
            "requested_time_s": tt,
            "sample_time_s": float(row[time_col]),
            "derivation": why,
        }
        rich = dict(flat)
        for c in df.columns:
            if c == time_col:
                continue
            v = row[c]
            arr = parse_vector(v)
            if arr is not None:
                rich[c] = arr
                flat[c] = json.dumps(arr)
            else:
                try:
                    if pd.isna(v):
                        continue
                except Exception:
                    pass
                try:
                    flat[c] = float(v)
                    rich[c] = float(v)
                except Exception:
                    flat[c] = str(v)
                    rich[c] = str(v)
        rows.append(flat)
        jrows.append(rich)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(jrows, indent=2))

def export_markers(df, time_col, phases, path):
    trip = marker_triplets(df.columns)
    rows = []
    for phase, tt, _ in phases:
        i = nearest_idx(df, time_col, tt)
        row = df.iloc[i]
        for marker, ax in sorted(trip.items()):
            vals = [pd.to_numeric(pd.Series([row[ax[a]]]), errors="coerce").iloc[0] for a in "xyz"]
            if all(pd.notna(v) for v in vals):
                rows.append({
                    "phase": phase,
                    "sample_time_s": float(row[time_col]),
                    "marker": marker,
                    "x": float(vals[0]),
                    "y": float(vals[1]),
                    "z": float(vals[2]),
                })
    pd.DataFrame(rows).to_csv(path, index=False)
    return trip

def export_groups(df, time_col, groups, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "kinematics_timeseries.csv": "kinematics",
        "contact_dynamics_timeseries.csv": "contact_dynamics",
        "imu_timeseries.csv": "imu",
        "state_timeseries.csv": "qpos_qvel_state",
    }
    for fn, g in mapping.items():
        cols = [time_col] + [c for c in groups[g] if c != time_col]
        cols = list(dict.fromkeys(cols))
        df[cols].to_csv(outdir/fn, index=False)

def copy_files(files_df, dest):
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in files_df.iterrows():
        src = Path(r["path"])
        dst = dest/src.name
        shutil.copy2(src, dst)
        rows.append({
            "source": str(src),
            "copied": str(dst),
            "kind": r["kind"],
            "bytes": src.stat().st_size,
            "sha256": sha256(src),
        })
    return pd.DataFrame(rows)

def diagnostic_plot(df, time_col, phases, path):
    if plt is None:
        return False
    series = []
    pc = pelvis_height_col(df.columns)
    tc = trunk_lean_col(df.columns)
    if pc:
        series.append(("Pelvis height", pd.to_numeric(df[pc], errors="coerce").to_numpy(dtype=float)))
    if tc:
        series.append(("Trunk lean", pd.to_numeric(df[tc], errors="coerce").to_numpy(dtype=float)))
    env, _ = contact_envelope(df, time_col)
    if env is not None:
        series.append(("Contact/impact envelope", env))
    if not series:
        return False

    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    fig, axes = plt.subplots(len(series), 1, figsize=(12, max(3.5, 2.6*len(series))), sharex=True)
    if len(series) == 1:
        axes = [axes]
    for ax, (label, y) in zip(axes, series):
        ax.plot(t, y, lw=1.2)
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        for name, tt, _ in phases:
            ax.axvline(tt, ls="--", lw=0.8, alpha=0.6)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Task 39 phase diagnostic — not final publication artwork")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True

def zip_folder(folder, zpath):
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
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
    for p, exp in tests:
        got = path_task_profile(p)
        if got != exp:
            raise RuntimeError(f"parser self-test failed: {p} -> {got}, expected {exp}")
    print("PARSER SELF-TEST: PASS")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=39)
    ap.add_argument("--profile-id", default="AUTO")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--event-manifest", type=Path, default=DEFAULT_EVENT_MANIFEST)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--zip-path", type=Path, required=True)
    args = ap.parse_args()

    parser_self_test()

    out = args.output_dir
    if out.exists():
        shutil.rmtree(out)
    for sub in [
        "00_AUDIT", "01_SELECTION", "02_EXACT_SOURCE_FILES", "03_PHASES",
        "04_MARKER_SKELETON", "05_TIMESERIES", "06_DIAGNOSTIC_PREVIEWS"
    ]:
        (out/sub).mkdir(parents=True, exist_ok=True)

    print("="*120)
    print("HEIGHT-FALL PAPER EVIDENCE EXTRACTOR V6 — ROBUST")
    print("="*120)
    print("Task:", args.task_id)
    print("Root:", args.root)

    print("\n[1/8] Path-authoritative discovery...")
    index_df = discover_exact_task_files(args.root, args.task_id)
    index_df.to_csv(out/"00_AUDIT/task_file_index.csv", index=False)
    print("Task files:", len(index_df))
    print("Profiles:", index_df["profile_id"].nunique())

    if index_df.empty:
        raise SystemExit("ERROR: no exact task files found.")

    print("\n[2/8] Canonical event manifest...")
    manifest = load_event_manifest(args.event_manifest, args.task_id)
    manifest.to_csv(out/"00_AUDIT/task_event_manifest_rows.csv", index=False)
    print("Task manifest rows:", len(manifest))

    cand = candidate_table(index_df, manifest)
    cand.to_csv(out/"01_SELECTION/candidate_profiles.csv", index=False)
    print("\nTOP CANDIDATES")
    print(cand.head(22).to_string(index=False))

    if str(args.profile_id).upper() == "AUTO":
        good = cand[
            (cand["eligible_in_event_manifest"] == True) &
            (cand["n_highrate_truth"] > 0) &
            (cand["n_timeseries"] > 0)
        ]
        if good.empty:
            good = cand[cand["n_highrate_truth"] > 0]
        if good.empty:
            raise SystemExit("ERROR: no file-backed high-rate candidate.")
        profile = str(good.iloc[0]["profile_id"])
    else:
        profile = canonical_profile_id(args.profile_id)

    files = index_df[index_df["profile_id"] == profile].copy()
    files.to_csv(out/"01_SELECTION/selected_trial_file_index.csv", index=False)
    print("\nSELECTED task/profile:", args.task_id, profile)
    print("Selected exact files:", len(files))
    print(files[["kind","bytes","path"]].to_string(index=False))

    print("\n[3/8] Exact source copy + hashes...")
    copied = copy_files(files, out/"02_EXACT_SOURCE_FILES")
    copied.to_csv(out/"02_EXACT_SOURCE_FILES/source_manifest_sha256.csv", index=False)

    # Same-directory manifest, path authoritative.
    manifests = [Path(p) for p in files[files["kind"]=="manifest"]["path"].tolist()]
    manifest_path = manifests[0] if manifests else None
    rate_meta = manifest_rate_metadata(manifest_path) if manifest_path else {}
    (out/"00_AUDIT/run_manifest_rate_metadata.json").write_text(json.dumps(rate_meta, indent=2))

    print("\n[4/8] Robust CSV parsing...")
    parsed = []
    for _, r in files[files["suffix"]==".csv"].iterrows():
        p = Path(r["path"])
        df, meta = robust_read_table(p, out/"00_AUDIT/parser_diagnostics")
        meta["path"] = str(p)
        meta["kind"] = r["kind"]
        parsed.append((p, r["kind"], df, meta))
        print(f"- {p.name}: {meta['status']}", end="")
        if df is not None:
            print(f" rows={len(df)} cols={len(df.columns)} sep={meta.get('separator')} enc={meta.get('encoding')}")
        else:
            print()

    parser_summary = [
        {k:v for k,v in meta.items() if k not in {"raw_probe"}}
        for _,_,_,meta in parsed
    ]
    (out/"00_AUDIT/parser_summary.json").write_text(json.dumps(parser_summary, indent=2))

    usable = [(p,k,d,m) for p,k,d,m in parsed if d is not None and len(d)>2 and len(d.columns)>1]
    if not usable:
        raise SystemExit(
            "ERROR: robust parser still could not decode either selected CSV. "
            "Upload V6 00_AUDIT/parser_diagnostics."
        )

    # Prefer high-rate truth by filename classification.
    high = [x for x in usable if x[1] == "highrate_truth"]
    primary_p, primary_kind, primary_df, primary_meta = high[0] if high else usable[0]
    print("Primary timeseries:", primary_p)
    print("Primary kind:", primary_kind)

    shutil.copy2(primary_p, out/"02_EXACT_SOURCE_FILES/PRIMARY_TIMESERIES_EXACT.csv")

    rate_hz, rate_source = choose_rate(rate_meta, highrate=(primary_kind=="highrate_truth"))
    primary_df, time_col, axis_meta = ensure_time_axis(primary_df, rate_hz, rate_source)
    axis_meta["primary_file"] = str(primary_p)
    (out/"00_AUDIT/time_axis_metadata.json").write_text(json.dumps(axis_meta, indent=2))
    print("Time axis:", time_col)
    print("Axis type:", axis_meta["axis_type"])
    if axis_meta.get("rate_hz"):
        print("Rate:", axis_meta["rate_hz"], "Hz via", axis_meta["rate_source"])

    print("\n[5/8] Expanding recorded vector3 columns + inventory...")
    primary_df, vector_map = expand_vector3_columns(primary_df)
    (out/"00_AUDIT/vector3_expansion_map.json").write_text(json.dumps(vector_map, indent=2))
    arr_cols = inspect_array_columns(primary_df)
    arr_cols.to_csv(out/"00_AUDIT/array_valued_columns.csv", index=False)

    groups = classified_columns(primary_df.columns)
    (out/"00_AUDIT/column_groups.json").write_text(json.dumps(groups, indent=2))
    pd.DataFrame({"column": list(primary_df.columns)}).to_csv(out/"00_AUDIT/primary_columns.csv", index=False)

    print("IMU columns:", len(groups["imu"]))
    print("Kinematic columns:", len(groups["kinematics"]))
    print("Contact/dynamics columns:", len(groups["contact_dynamics"]))
    print("State/qpos/qvel-like columns:", len(groups["qpos_qvel_state"]))
    print("Marker xyz-like columns:", len(groups["marker_xyz_like"]))

    print("\n[6/8] Resolving exact fall phases...")
    mrows = event_rows_for_profile(manifest, profile)
    onset, impact, rest, event_sources = resolve_events(mrows, primary_df, time_col)
    phases = derive_phases(primary_df, time_col, onset, impact, rest)
    phase_df = pd.DataFrame(phases, columns=["phase","time_s","derivation"])
    phase_df.to_csv(out/"03_PHASES/phase_timestamps.csv", index=False)

    export_phase_snapshots(
        primary_df, time_col, phases,
        out/"03_PHASES/phase_snapshot_rows.csv",
        out/"03_PHASES/phase_snapshot_rows.json"
    )
    triplets = export_markers(
        primary_df, time_col, phases,
        out/"04_MARKER_SKELETON/phase_marker_positions_long.csv"
    )
    pd.DataFrame({"marker": sorted(triplets)}).to_csv(
        out/"04_MARKER_SKELETON/detected_markers.csv", index=False
    )

    phase_meta = {
        "task_id": args.task_id,
        "profile_id": profile,
        "primary_timeseries": str(primary_p),
        "primary_kind": primary_kind,
        "time_axis": axis_meta,
        "fall_onset_s": onset,
        "peak_impact_s": impact,
        "rest_s": rest,
        "event_sources": event_sources,
        "phases": [
            {"phase": n, "time_s": t, "derivation": why}
            for n,t,why in phases
        ],
    }
    (out/"03_PHASES/phase_metadata.json").write_text(json.dumps(phase_meta, indent=2))

    print("Fall onset:", onset, "source:", event_sources.get("onset"))
    print("Peak impact:", impact, "source:", event_sources.get("impact"))
    print("Rest:", rest, "source:", event_sources.get("rest"))
    print("Detected marker triplets:", len(triplets))
    print(phase_df.to_string(index=False))

    print("\n[7/8] Figure-ready channel exports...")
    export_groups(primary_df, time_col, groups, out/"05_TIMESERIES")
    diag = diagnostic_plot(
        primary_df, time_col, phases,
        out/"06_DIAGNOSTIC_PREVIEWS/diagnostic_phase_timeline.png"
    )

    # Determine whether exact recorded state is sufficient for direct skeleton or MuJoCo render.
    state_ready = (
        len(triplets) >= 6
        or len(groups["qpos_qvel_state"]) > 0
        or (not arr_cols.empty and arr_cols["column"].astype(str).str.contains("qpos|body|joint|marker", case=False, regex=True).any())
    )
    contact_ready = len(groups["contact_dynamics"]) > 0
    kinematic_ready = len(groups["kinematics"]) > 0
    imu_ready = len(groups["imu"]) > 0

    critical_fallbacks = [
        k for k,v in event_sources.items()
        if isinstance(v, str) and "FALLBACK" in v
    ]
    phase_fallback = any("FALLBACK" in why for _,_,why in phases)

    readiness = {
        "exact_recorded_skeleton_or_state_available": bool(state_ready),
        "kinematic_channels_available": bool(kinematic_ready),
        "contact_dynamics_available": bool(contact_ready),
        "imu_channels_available": bool(imu_ready),
        "critical_event_fallbacks": critical_fallbacks,
        "phase_fallback_present": bool(phase_fallback),
        "recommended_next_step": (
            "CREATE_PAPER_FIGURE_FROM_EXTRACTED_DATA"
            if state_ready and contact_ready and kinematic_ready
            else
            "INSTRUMENTED_REPLAY_NEEDED_FOR_EXACT_POSE_OR_CONTACT_DETAIL"
        ),
    }
    (out/"FIGURE_READINESS.json").write_text(json.dumps(readiness, indent=2))

    readme = f"""TASK {args.task_id} HEIGHT-FALL EVIDENCE PACKAGE V6

Selected profile: {profile}
Primary recorded source: {primary_p}
Primary artifact: {primary_kind}

Time-axis audit:
{json.dumps(axis_meta, indent=2)}

Event/phase sources:
{json.dumps(event_sources, indent=2)}

Phase timestamps:
{phase_df.to_string(index=False)}

Recorded-data readiness:
{json.dumps(readiness, indent=2)}

Use this package to decide the final publication figure.
Do NOT invent skeleton geometry. If exact marker/qpos/body state is absent,
perform an instrumented deterministic replay using the exact scenario manifest
before constructing the final pose sequence.

NO SIMULATOR RERUN WAS PERFORMED BY THIS EXTRACTOR.
NO CNN TRAINING OR INFERENCE WAS PERFORMED.
"""
    (out/"PAPER_FIGURE_README.txt").write_text(readme)

    print("\nFIGURE READINESS")
    print(json.dumps(readiness, indent=2))

    print("\n[8/8] Building upload ZIP...")
    zip_folder(out, args.zip_path)
    zsha = sha256(args.zip_path)
    sha_path = args.zip_path.parent/f"{args.zip_path.name}.sha256"
    sha_path.write_text(f"{zsha}  {args.zip_path}\n")

    print("\n"+"="*120)
    print("HEIGHT-FALL PAPER EVIDENCE EXTRACTION V6: PASS")
    print("="*120)
    print("ZIP:", args.zip_path)
    print("ZIP SHA256:", zsha)

if __name__ == "__main__":
    main()
