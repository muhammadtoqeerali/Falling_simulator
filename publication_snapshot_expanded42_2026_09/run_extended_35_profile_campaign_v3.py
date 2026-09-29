#!/usr/bin/env python3
"""
run_extended_35_profile_campaign.py

Purpose
-------
Extend the existing 13-profile humanoid fall campaign to a designed total cohort
of 35 unique parameterized profiles by running 22 additional profiles across all
26 semantic fall scenarios.

The script:
  * preserves the existing 13 profiles as reference profiles;
  * adds 22 deliberately stratified profiles;
  * balances the combined cohort to 18 male / 17 female profiles;
  * covers the current profile-code age bands approximately evenly:
        under_35: 8
        35_to_55: 9
        55_to_70: 9
        70_plus:  9
  * audits the profile-derived gait/control parameters before simulation;
  * runs all 26 current semantic scenarios for the 22 new profiles;
  * uses the existing fall_core.run_simulation() path, which in the current
    codebase uses highrate_env_step() and therefore writes the sibling
    *_highrate_truth.csv stream;
  * disables the interactive viewer;
  * supports resume after interruption;
  * moves completed run folders into one dedicated campaign directory;
  * writes profile, progress, run-manifest, QC, and failure tables continuously.

IMPORTANT
---------
This script does not alter biofidelic_profile.py, fall_core.py, scenario code,
or the original 13-profile campaign. It only orchestrates additional runs.

By default --full-exports is OFF to reduce overnight runtime and storage.
This still keeps the legacy IMU CSV, high-rate truth CSV, run manifest, event
summary, and validation report produced by run_simulation(). Use
--full-exports only if you specifically want all per-run biomechanics plots
and bundles for every new profile.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Existing 13 profiles from the frozen publication handoff.
# These are NOT rerun by this script.
# ---------------------------------------------------------------------------
EXISTING_13 = [
    {"profile_id": "P001", "age": 25, "height": 1.62, "sex": "female", "weight": 58.0},
    {"profile_id": "P002", "age": 28, "height": 1.65, "sex": "male",   "weight": 60.0},
    {"profile_id": "P003", "age": 32, "height": 1.70, "sex": "male",   "weight": 78.0},
    {"profile_id": "P004", "age": 33, "height": 1.65, "sex": "male",   "weight": 77.0},
    {"profile_id": "P005", "age": 35, "height": 1.68, "sex": "male",   "weight": 65.0},
    {"profile_id": "P006", "age": 37, "height": 1.67, "sex": "male",   "weight": 75.0},
    {"profile_id": "P007", "age": 38, "height": 1.67, "sex": "male",   "weight": 75.0},
    {"profile_id": "P008", "age": 41, "height": 1.70, "sex": "male",   "weight": 80.0},
    {"profile_id": "P009", "age": 46, "height": 1.72, "sex": "male",   "weight": 80.0},
    {"profile_id": "P010", "age": 55, "height": 1.75, "sex": "male",   "weight": 82.0},
    {"profile_id": "P011", "age": 60, "height": 1.64, "sex": "male",   "weight": 80.0},
    {"profile_id": "P012", "age": 62, "height": 1.65, "sex": "male",   "weight": 75.0},
    {"profile_id": "P013", "age": 65, "height": 1.70, "sex": "male",   "weight": 80.0},
]


# ---------------------------------------------------------------------------
# 22 NEW profiles.
#
# Design logic:
#   Combined total = 35 profiles
#   Combined sex   = 18 male / 17 female
#   Combined current code age bands = 8 / 9 / 9 / 9
#   Age span       = 20--78 years
#   Height span    = 1.50--1.90 m
#   Weight span    = 48--96 kg
#
# The 70+ group is kept at 70--78 rather than extending far beyond 80 because
# several profile functions start approaching clipping/saturation at advanced
# ages. This gives an older group while retaining more parameter variation.
# ---------------------------------------------------------------------------
NEW_22 = [
    # under_35: add 4 -> combined band total = 8
    {"profile_id": "P014", "age": 20, "height": 1.52, "sex": "female", "weight": 48.0},
    {"profile_id": "P015", "age": 23, "height": 1.74, "sex": "female", "weight": 68.0},
    {"profile_id": "P016", "age": 30, "height": 1.60, "sex": "female", "weight": 72.0},
    {"profile_id": "P017", "age": 34, "height": 1.88, "sex": "male",   "weight": 95.0},

    # 35_to_55: add 4 -> combined band total = 9
    {"profile_id": "P018", "age": 36, "height": 1.55, "sex": "female", "weight": 55.0},
    {"profile_id": "P019", "age": 42, "height": 1.76, "sex": "female", "weight": 78.0},
    {"profile_id": "P020", "age": 49, "height": 1.63, "sex": "female", "weight": 73.0},
    {"profile_id": "P021", "age": 53, "height": 1.90, "sex": "male",   "weight": 96.0},

    # 55_to_70: add 5 -> combined band total = 9
    {"profile_id": "P022", "age": 56, "height": 1.53, "sex": "female", "weight": 61.0},
    {"profile_id": "P023", "age": 59, "height": 1.71, "sex": "female", "weight": 69.0},
    {"profile_id": "P024", "age": 63, "height": 1.60, "sex": "female", "weight": 76.0},
    {"profile_id": "P025", "age": 67, "height": 1.74, "sex": "female", "weight": 70.0},
    {"profile_id": "P026", "age": 69, "height": 1.84, "sex": "male",   "weight": 90.0},

    # 70_plus: add 9 -> combined band total = 9
    {"profile_id": "P027", "age": 70, "height": 1.50, "sex": "female", "weight": 58.0},
    {"profile_id": "P028", "age": 71, "height": 1.86, "sex": "male",   "weight": 88.0},
    {"profile_id": "P029", "age": 72, "height": 1.66, "sex": "female", "weight": 75.0},
    {"profile_id": "P030", "age": 73, "height": 1.58, "sex": "female", "weight": 64.0},
    {"profile_id": "P031", "age": 74, "height": 1.78, "sex": "male",   "weight": 84.0},
    {"profile_id": "P032", "age": 75, "height": 1.75, "sex": "female", "weight": 80.0},
    {"profile_id": "P033", "age": 76, "height": 1.54, "sex": "female", "weight": 67.0},
    {"profile_id": "P034", "age": 77, "height": 1.68, "sex": "male",   "weight": 74.0},
    {"profile_id": "P035", "age": 78, "height": 1.70, "sex": "female", "weight": 72.0},
]

SCENARIO_IDS = [
    20, 21, 22, 23, 24,
    25, 26, 27, 28, 29,
    30, 31, 32, 33, 34,
    37, 38, 39, 40, 41,
    42, 43, 44, 250, 290, 291,
]


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _age_band(age: int) -> str:
    if age < 35:
        return "under_35"
    if age < 55:
        return "35_to_55"
    if age < 70:
        return "55_to_70"
    return "70_plus"


def _atomic_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _replace_path_strings(obj, old: str, new: str):
    if isinstance(obj, dict):
        return {k: _replace_path_strings(v, old, new) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_path_strings(v, old, new) for v in obj]
    if isinstance(obj, str):
        return obj.replace(old, new)
    return obj


def _find_highrate_truth(run_dir: Path):
    files = sorted(run_dir.glob("*_highrate_truth.csv"))
    return files[-1] if files else None


def _truth_qc(path: Path | None):
    result = {
        "truth_exists": False,
        "truth_rows": 0,
        "truth_rate_hz": np.nan,
        "truth_columns_ok": False,
        "truth_qc_ok": False,
    }
    if path is None or not path.exists():
        return result

    result["truth_exists"] = True
    try:
        df = pd.read_csv(path, comment="#")
        result["truth_rows"] = int(len(df))

        required = {
            "timestamp",
            "accel_true_x", "accel_true_y", "accel_true_z",
            "gyro_true_x", "gyro_true_y", "gyro_true_z",
        }
        result["truth_columns_ok"] = required.issubset(set(df.columns))

        if "timestamp" in df.columns and len(df) >= 3:
            t = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(float)
            dt = np.diff(t)
            dt = dt[np.isfinite(dt) & (dt > 0)]
            if len(dt):
                result["truth_rate_hz"] = float(1.0 / np.median(dt))

        rate_ok = (
            np.isfinite(result["truth_rate_hz"])
            and 99.0 <= result["truth_rate_hz"] <= 101.0
        )
        result["truth_qc_ok"] = bool(
            result["truth_rows"] > 10
            and result["truth_columns_ok"]
            and rate_ok
        )
    except Exception:
        pass

    return result


def build_profile_audit(get_age_style_v2):
    rows = []
    all_profiles = [
        (p, "existing_reference") for p in EXISTING_13
    ] + [
        (p, "new_run") for p in NEW_22
    ]

    for p, cohort_role in all_profiles:
        d = get_age_style_v2(
            p["age"], p["height"], p["sex"], p["weight"]
        )
        bmi = p["weight"] / (p["height"] ** 2)

        rows.append({
            **p,
            "cohort_role": cohort_role,
            "age_band": _age_band(p["age"]),
            "bmi": bmi,
            "target_walk_speed_mps": d["target_walk_speed"],
            "double_support_fraction": d["expected_double_support"],
            "muscle_strength_factor": d["muscle_strength_factor"],
            "balance_impairment": d["balance_impairment"],
            "reaction_delay_s": d["reaction_delay_s"],
            "stand_stoop_deg": d["stand_stoop_target_deg"],
            "walk_stoop_deg": d["walk_stoop_target_deg"],
            "arm_gain": d["arm_gain"],
            "proprioception_scale": d["proprioception_scale"],
            "profile_code_label": d["label"],
        })

    df = pd.DataFrame(rows)

    # Nearest-neighbor audit in normalized input + profile-output feature space.
    feature_cols = [
        "age", "height", "bmi",
        "target_walk_speed_mps",
        "double_support_fraction",
        "muscle_strength_factor",
        "balance_impairment",
        "reaction_delay_s",
        "walk_stoop_deg",
        "arm_gain",
        "proprioception_scale",
    ]

    X = df[feature_cols].to_numpy(float)
    mins = np.nanmin(X, axis=0)
    spans = np.nanmax(X, axis=0) - mins
    spans[spans == 0] = 1.0
    Z = (X - mins) / spans

    nn_id = []
    nn_dist = []

    for i in range(len(df)):
        delta = Z - Z[i]
        dist = np.sqrt(np.nanmean(delta * delta, axis=1))

        # Add a sex-separation penalty in the profile-space distance.
        sex_mismatch = (
            df["sex"].astype(str).str.lower().to_numpy()
            != str(df.loc[i, "sex"]).lower()
        )
        dist = np.sqrt(dist * dist + 0.15 * sex_mismatch.astype(float))
        dist[i] = np.inf

        j = int(np.argmin(dist))
        nn_id.append(df.loc[j, "profile_id"])
        nn_dist.append(float(dist[j]))

    df["nearest_profile_id"] = nn_id
    df["nearest_profile_distance"] = nn_dist
    return df


def cohort_summary(profile_df: pd.DataFrame):
    print("\n" + "=" * 78)
    print("COMBINED 35-PROFILE DESIGN")
    print("=" * 78)
    print(f"Total profiles : {len(profile_df)}")
    print("Sex counts:")
    print(profile_df["sex"].value_counts().to_string())
    print("\nAge-band counts:")
    print(profile_df["age_band"].value_counts().reindex(
        ["under_35", "35_to_55", "55_to_70", "70_plus"]
    ).to_string())
    print(
        f"\nAge range      : {profile_df.age.min()}--{profile_df.age.max()} yr"
    )
    print(
        f"Height range   : {profile_df.height.min():.2f}--"
        f"{profile_df.height.max():.2f} m"
    )
    print(
        f"Weight range   : {profile_df.weight.min():.1f}--"
        f"{profile_df.weight.max():.1f} kg"
    )
    print(
        f"BMI range      : {profile_df.bmi.min():.1f}--"
        f"{profile_df.bmi.max():.1f} kg/m^2"
    )
    print("=" * 78)


def move_completed_run(old_dir: Path, campaign_root: Path, profile_id: str, scenario_id: int):
    target_parent = campaign_root / "runs" / profile_id / f"task_{scenario_id}"
    target_parent.mkdir(parents=True, exist_ok=True)
    target_dir = target_parent / old_dir.name

    if target_dir.exists():
        raise RuntimeError(f"Target run directory already exists: {target_dir}")

    shutil.move(str(old_dir), str(target_dir))

    # Repair path strings in the moved JSON run manifest.
    manifest = target_dir / "run_manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            data = _replace_path_strings(data, str(old_dir), str(target_dir))
            manifest.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"[WARN] Could not rewrite moved manifest paths: {exc}", flush=True)

    return target_dir



def _acquire_campaign_lock(campaign_root: Path):
    """
    Prevent two campaign runners from writing the same progress/manifests at once.
    The returned file handle must remain alive for the entire process.
    """
    lock_path = campaign_root / ".campaign.lock"
    fh = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.seek(0)
        holder = fh.read().strip()
        raise SystemExit(
            "Another campaign process is already running for this campaign.\n"
            f"Lock file: {lock_path}\n"
            f"Recorded holder: {holder or 'unknown'}\n"
            "Do not start a second writer. Check the existing PID/log first."
        )
    fh.seek(0)
    fh.truncate()
    fh.write(f"pid={os.getpid()} started={_now()}\n")
    fh.flush()
    return fh


def _coerce_progress_dtypes(progress: pd.DataFrame) -> pd.DataFrame:
    """
    CSV reload can infer all-empty text columns as float64/NaN.
    Force the mutable text columns back to object/string-compatible dtype
    before assigning timestamps, paths, statuses, or error text.
    """
    text_cols = [
        "profile_id",
        "status",
        "started_at",
        "finished_at",
        "output_dir",
        "highrate_truth_csv",
        "error",
    ]
    for col in text_cols:
        if col not in progress.columns:
            progress[col] = ""
        progress[col] = progress[col].fillna("").astype(object)

    bool_cols = ["truth_qc_ok"]
    for col in bool_cols:
        if col in progress.columns:
            # Preserve booleans if possible; NaN becomes False.
            progress[col] = progress[col].fillna(False).astype(bool)

    return progress


def _reconcile_completed_qc_warnings(progress: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Re-check prior completed_truth_qc_warning rows using the correct parser for
    high-rate truth files (which contain leading '#' provenance comments).

    Valid prior rows are promoted to 'completed' so a resumed campaign does not
    rerun them. Invalid/missing rows remain warnings for later inspection.
    """
    repaired = 0

    if "status" not in progress.columns:
        return progress, repaired

    warning_mask = progress["status"].astype(str).eq(
        "completed_truth_qc_warning"
    )

    for idx in progress.index[warning_mask]:
        path_str = str(progress.at[idx, "highrate_truth_csv"] or "").strip()
        if not path_str:
            continue

        qc = _truth_qc(Path(path_str))

        progress.at[idx, "truth_rows"] = qc["truth_rows"]
        progress.at[idx, "truth_rate_hz"] = qc["truth_rate_hz"]
        progress.at[idx, "truth_qc_ok"] = qc["truth_qc_ok"]

        if qc["truth_qc_ok"]:
            progress.at[idx, "status"] = "completed"
            repaired += 1

    return progress, repaired

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Simulator repository root (default: current directory)",
    )
    ap.add_argument(
        "--campaign-name",
        default="campaign_highrate_truth_v2_extended35",
        help="Dedicated output campaign directory name",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only generate/audit the 35-profile design; do not simulate",
    )
    ap.add_argument(
        "--full-exports",
        action="store_true",
        help=(
            "Keep per-run biomechanics/plot bundles. Default is faster IMU-focused "
            "overnight generation without those heavy exports."
        ),
    )
    ap.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry rows previously marked failed when resuming",
    )
    ap.add_argument(
        "--only-profile",
        nargs="*",
        default=None,
        help="Optional new profile IDs to run, e.g. P014 P015",
    )
    ap.add_argument(
        "--only-task",
        nargs="*",
        type=int,
        default=None,
        help="Optional scenario IDs to run",
    )
    args = ap.parse_args()

    root = args.root.resolve()
    if not (root / "fall_core.py").exists():
        raise SystemExit(
            f"fall_core.py not found under {root}. "
            "Run this script from /mnt/hdd16T/ToqeerHomeBackup/mujoco_project "
            "or pass --root."
        )

    sys.path.insert(0, str(root))

    # Force noninteractive plotting.
    os.environ.setdefault("MPLBACKEND", "Agg")

    from biofidelic_profile import get_age_style_v2
    from fall_core import run_simulation
    from fall_scenario_library import get_scenario

    campaign_root = (
        root / "outputs" / "_highrate_overnight" / args.campaign_name
    )
    campaign_root.mkdir(parents=True, exist_ok=True)
    (campaign_root / "logs").mkdir(exist_ok=True)
    (campaign_root / "runs").mkdir(exist_ok=True)

    # Hold an exclusive lock for the entire campaign process.
    campaign_lock = _acquire_campaign_lock(campaign_root)

    profile_df = build_profile_audit(get_age_style_v2)
    cohort_summary(profile_df)

    _atomic_csv(profile_df, campaign_root / "combined_35_profile_design.csv")
    _atomic_csv(
        profile_df[profile_df["cohort_role"] == "new_run"].copy(),
        campaign_root / "new_22_profile_design.csv",
    )

    # Simple human-readable cohort summary.
    summary_txt = [
        "EXTENDED 35-PROFILE CAMPAIGN DESIGN",
        "===================================",
        f"Generated: {_now()}",
        "",
        "Existing profiles retained: 13",
        "New profiles to run: 22",
        "Combined profiles: 35",
        "Combined sex balance: 18 male / 17 female",
        "Combined age-band design:",
        "  under_35 : 8",
        "  35_to_55 : 9",
        "  55_to_70 : 9",
        "  70_plus  : 9",
        "",
        "Age range: 20--78 years",
        "The 70+ group is intentionally concentrated at 70--78 because",
        "several current profile functions begin approaching clipping/saturation",
        "at more advanced ages. This preserves an older cohort while retaining",
        "more variation in the derived control/profile parameters.",
        "",
        "This campaign runs only P014--P035. The original 13-profile campaign",
        "remains untouched and should later be combined at the dataset/manifest level.",
        "",
        "IMPORTANT: these are parameterized synthetic profiles, not matched real subjects",
        "and not validated person-specific digital twins.",
    ]
    (campaign_root / "README_CAMPAIGN.txt").write_text(
        "\n".join(summary_txt) + "\n", encoding="utf-8"
    )

    # Build plan.
    selected_profiles = NEW_22
    if args.only_profile:
        allowed = set(args.only_profile)
        selected_profiles = [p for p in selected_profiles if p["profile_id"] in allowed]

    selected_tasks = SCENARIO_IDS
    if args.only_task:
        selected_tasks = [t for t in SCENARIO_IDS if t in set(args.only_task)]

    plan_rows = []
    for p in selected_profiles:
        for sid in selected_tasks:
            plan_rows.append({
                "profile_id": p["profile_id"],
                "age": p["age"],
                "height": p["height"],
                "sex": p["sex"],
                "weight": p["weight"],
                "scenario_id": sid,
            })

    plan_df = pd.DataFrame(plan_rows)
    _atomic_csv(plan_df, campaign_root / "campaign_plan.csv")

    print(f"\nCampaign root : {campaign_root}")
    print(f"New profiles  : {len(selected_profiles)}")
    print(f"Tasks/profile : {len(selected_tasks)}")
    print(f"Planned runs  : {len(plan_df)}")
    print(f"Full exports  : {args.full_exports}")

    if args.dry_run:
        print("\nDRY RUN COMPLETE. No simulations were started.")
        print(
            "Inspect combined_35_profile_design.csv and new_22_profile_design.csv "
            "before launching the overnight run."
        )
        return

    progress_path = campaign_root / "campaign_progress.csv"
    if progress_path.exists():
        progress = pd.read_csv(progress_path)
        progress = _coerce_progress_dtypes(progress)
    else:
        progress = plan_df.copy()
        progress["status"] = "pending"
        progress["started_at"] = ""
        progress["finished_at"] = ""
        progress["elapsed_s"] = np.nan
        progress["output_dir"] = ""
        progress["highrate_truth_csv"] = ""
        progress["truth_rows"] = 0
        progress["truth_rate_hz"] = np.nan
        progress["truth_qc_ok"] = False
        progress["error"] = ""
        progress = _coerce_progress_dtypes(progress)
        _atomic_csv(progress, progress_path)

    # Repair legacy QC warnings produced by the earlier parser that did not
    # ignore the leading '#' provenance-comment lines.
    progress, repaired_qc = _reconcile_completed_qc_warnings(progress)
    if repaired_qc:
        print(
            f"[QC] Reclassified {repaired_qc} prior truth-QC warning rows "
            "as completed after parsing provenance comments correctly.",
            flush=True,
        )
        _atomic_csv(progress, progress_path)

    # Ensure any newly filtered plan rows exist in progress.
    key_cols = ["profile_id", "scenario_id"]
    existing_keys = {
        (str(r.profile_id), int(r.scenario_id))
        for r in progress.itertuples()
    }
    missing = []
    for r in plan_df.itertuples():
        key = (str(r.profile_id), int(r.scenario_id))
        if key not in existing_keys:
            missing.append({
                **{c: getattr(r, c) for c in plan_df.columns},
                "status": "pending",
                "started_at": "",
                "finished_at": "",
                "elapsed_s": np.nan,
                "output_dir": "",
                "highrate_truth_csv": "",
                "truth_rows": 0,
                "truth_rate_hz": np.nan,
                "truth_qc_ok": False,
                "error": "",
            })
    if missing:
        progress = pd.concat([progress, pd.DataFrame(missing)], ignore_index=True)
        progress = _coerce_progress_dtypes(progress)
        _atomic_csv(progress, progress_path)

    # Run manifest is a richer one-row-per-run table.
    run_manifest_path = campaign_root / "campaign_run_manifest.csv"
    if run_manifest_path.exists():
        run_manifest = pd.read_csv(run_manifest_path)
    else:
        run_manifest = pd.DataFrame()

    total_selected = len(plan_df)
    completed_before = 0

    # Write live status immediately so monitoring works before the first run finishes.
    initial_live = {
        "updated_at": _now(),
        "campaign_root": str(campaign_root),
        "selected_profiles": len(selected_profiles),
        "selected_tasks": len(selected_tasks),
        "planned_runs": total_selected,
        "status_counts": progress["status"].value_counts().to_dict(),
    }
    (campaign_root / "live_status.json").write_text(
        json.dumps(initial_live, indent=2), encoding="utf-8"
    )

    for plan_idx, row in plan_df.iterrows():
        pid = str(row["profile_id"])
        sid = int(row["scenario_id"])

        mask = (
            progress["profile_id"].astype(str).eq(pid)
            & pd.to_numeric(progress["scenario_id"], errors="coerce").eq(sid)
        )
        if not mask.any():
            raise RuntimeError(f"Progress row missing for {pid} task {sid}")

        current_status = str(progress.loc[mask, "status"].iloc[0])

        if current_status in {"completed", "completed_truth_qc_warning"}:
            completed_before += 1
            print(f"[RESUME] Skip completed {pid} task {sid} ({current_status})", flush=True)
            continue

        if current_status == "failed" and not args.retry_failed:
            print(
                f"[RESUME] Skip previously failed {pid} task {sid}; "
                "use --retry-failed to retry.",
                flush=True,
            )
            continue

        p = next(x for x in NEW_22 if x["profile_id"] == pid)
        scenario = get_scenario(sid)

        # No interactive viewer during overnight batch.
        scenario.viewer_enabled = False

        # Heavy biomechanics image/bundle export is optional.
        scenario.save_biomechanics = bool(args.full_exports)
        scenario.save_imu_csv = True

        progress.loc[mask, "status"] = "running"
        progress.loc[mask, "started_at"] = _now()
        progress.loc[mask, "error"] = ""
        _atomic_csv(progress, progress_path)

        running_live = {
            "updated_at": _now(),
            "campaign_root": str(campaign_root),
            "selected_profiles": len(selected_profiles),
            "selected_tasks": len(selected_tasks),
            "planned_runs": total_selected,
            "current_profile_id": pid,
            "current_scenario_id": sid,
            "status_counts": progress["status"].value_counts().to_dict(),
        }
        (campaign_root / "live_status.json").write_text(
            json.dumps(running_live, indent=2), encoding="utf-8"
        )

        t0 = time.time()

        print("\n" + "#" * 90, flush=True)
        print(
            f"[{_now()}] START {pid} | task {sid} | "
            f"age={p['age']} height={p['height']} sex={p['sex']} weight={p['weight']}",
            flush=True,
        )
        print("#" * 90, flush=True)

        try:
            summary = run_simulation(
                scenario,
                {
                    "age": p["age"],
                    "height": p["height"],
                    "sex": p["sex"],
                    "weight": p["weight"],
                },
            )

            old_dir = Path(summary["output_dir"]).resolve()
            if not old_dir.exists():
                raise RuntimeError(
                    f"run_simulation returned missing output_dir: {old_dir}"
                )

            new_dir = move_completed_run(
                old_dir, campaign_root, pid, sid
            ).resolve()

            truth = _find_highrate_truth(new_dir)
            qc = _truth_qc(truth)

            elapsed = time.time() - t0

            # Update progress.
            progress.loc[mask, "status"] = (
                "completed" if qc["truth_qc_ok"] else "completed_truth_qc_warning"
            )
            progress.loc[mask, "finished_at"] = _now()
            progress.loc[mask, "elapsed_s"] = elapsed
            progress.loc[mask, "output_dir"] = str(new_dir)
            progress.loc[mask, "highrate_truth_csv"] = str(truth) if truth else ""
            progress.loc[mask, "truth_rows"] = qc["truth_rows"]
            progress.loc[mask, "truth_rate_hz"] = qc["truth_rate_hz"]
            progress.loc[mask, "truth_qc_ok"] = qc["truth_qc_ok"]
            progress.loc[mask, "error"] = ""
            _atomic_csv(progress, progress_path)

            # Add/replace one row in run manifest.
            rec = {
                "profile_id": pid,
                "age": p["age"],
                "height": p["height"],
                "sex": p["sex"],
                "weight": p["weight"],
                "scenario_id": sid,
                "description": summary.get("description", ""),
                "classification": summary.get("classification", ""),
                "overall_score": summary.get("overall_score", np.nan),
                "authenticity": summary.get("authenticity", np.nan),
                "sisfall_compliant": summary.get("sisfall_compliant", False),
                "kfall_compliant": summary.get("kfall_compliant", False),
                "output_dir": str(new_dir),
                "highrate_truth_csv": str(truth) if truth else "",
                "truth_rows": qc["truth_rows"],
                "truth_rate_hz": qc["truth_rate_hz"],
                "truth_qc_ok": qc["truth_qc_ok"],
                "elapsed_s": elapsed,
                "finished_at": _now(),
            }

            if len(run_manifest):
                keep = ~(
                    run_manifest["profile_id"].astype(str).eq(pid)
                    & pd.to_numeric(
                        run_manifest["scenario_id"], errors="coerce"
                    ).eq(sid)
                )
                run_manifest = run_manifest.loc[keep].copy()

            run_manifest = pd.concat(
                [run_manifest, pd.DataFrame([rec])],
                ignore_index=True,
            )
            _atomic_csv(run_manifest, run_manifest_path)

            print(
                f"[{_now()}] DONE {pid} task {sid} | "
                f"{elapsed/60:.1f} min | "
                f"truth={qc['truth_rows']} rows @ "
                f"{qc['truth_rate_hz']:.3f} Hz | "
                f"QC={'OK' if qc['truth_qc_ok'] else 'WARN'}",
                flush=True,
            )

        except Exception as exc:
            elapsed = time.time() - t0
            err = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )

            progress.loc[mask, "status"] = "failed"
            progress.loc[mask, "finished_at"] = _now()
            progress.loc[mask, "elapsed_s"] = elapsed
            progress.loc[mask, "error"] = str(exc)[:1000]
            _atomic_csv(progress, progress_path)

            fail_dir = campaign_root / "logs" / "failures"
            fail_dir.mkdir(parents=True, exist_ok=True)
            (fail_dir / f"{pid}_task{sid}.txt").write_text(
                err, encoding="utf-8"
            )

            print(
                f"[{_now()}] FAILED {pid} task {sid}: {exc}",
                flush=True,
            )
            # Continue with the remaining runs.
            continue

        # Continuously write a small live summary.
        status_counts = progress["status"].value_counts().to_dict()
        live = {
            "updated_at": _now(),
            "campaign_root": str(campaign_root),
            "selected_profiles": len(selected_profiles),
            "selected_tasks": len(selected_tasks),
            "planned_runs": total_selected,
            "status_counts": status_counts,
        }
        (campaign_root / "live_status.json").write_text(
            json.dumps(live, indent=2), encoding="utf-8"
        )

    # Final status summary.
    final_counts = progress["status"].value_counts().rename_axis(
        "status"
    ).reset_index(name="count")
    _atomic_csv(final_counts, campaign_root / "campaign_status_summary.csv")

    # Truth-only QC table for all rows that have a path.
    qc_cols = [
        "profile_id", "age", "height", "sex", "weight", "scenario_id",
        "status", "output_dir", "highrate_truth_csv", "truth_rows",
        "truth_rate_hz", "truth_qc_ok", "elapsed_s", "error",
    ]
    _atomic_csv(
        progress[qc_cols].copy(),
        campaign_root / "highrate_truth_qc_manifest.csv",
    )

    print("\n" + "=" * 90)
    print("CAMPAIGN FINISHED")
    print("=" * 90)
    print(final_counts.to_string(index=False))
    print(f"\nCampaign root:\n  {campaign_root}")
    print(
        "\nIf any rows are failed, rerun the same command with --retry-failed. "
        "Completed rows will be skipped."
    )


if __name__ == "__main__":
    main()
