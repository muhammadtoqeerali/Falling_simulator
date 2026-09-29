#!/usr/bin/env python3
"""
build_expanded42_canonical_event_manifest_v2.py

V2 canonical synthetic-event builder for the expanded 42-profile campaign.

Design:
  * NEVER modifies the frozen 396-row publication manifest.
  * First regression-validates the reconstructed event policy against all
    396 frozen publication rows.
  * For the new P036--P055 runs, applies the same frozen event policy:
        recovered impact =
        peak legacy `impact_magnitude` within 3 s after fall onset.
  * Uses the corrected ~100-Hz high-rate truth stream to determine the
    onset-to-impact sample count and 300-ms / 50%-overlap window eligibility.
  * Combines the frozen 396 rows with 360 newly generated rows.
  * Requires exactly 756 profile-task source simulations.
  * Does NOT assume the final eligible-trial/window counts in advance.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import math

import numpy as np
import pandas as pd


PROJECT = Path("/mnt/hdd16T/ToqeerHomeBackup/mujoco_project")

OLD_CANON = (
    PROJECT
    / "outputs"
    / "phase2_publication_gate_audit_20260826"
    / "final_synthetic_event_policy_v2"
    / "canonical_synthetic_event_manifest_v2.csv"
)

EXPECTED_OLD_SHA256 = (
    "5f0dd36d7d24e12805ed49f57778c5ceaf5e5ac3582347f3ed5525d2f2d61c2a"
)

NEW_CAMPAIGN = (
    PROJECT
    / "outputs"
    / "_highrate_overnight"
    / "campaign_highrate_truth_v4_expanded42_new20"
)

NEW_RUN_MANIFEST = NEW_CAMPAIGN / "campaign_run_manifest.csv"
NEW_PROGRESS = NEW_CAMPAIGN / "campaign_progress.csv"

OUT = (
    PROJECT
    / "outputs"
    / "phase2_publication_gate_expanded42_v2"
    / "final_synthetic_event_policy_v2"
)

OUT.mkdir(parents=True, exist_ok=True)

COMBINED_MANIFEST = OUT / "canonical_synthetic_event_manifest_expanded42_v2.csv"
FACTS_PATH = OUT / "canonical_synthetic_event_facts_expanded42_v2.json"
REGRESSION_PATH = OUT / "old396_event_policy_regression.csv"
PROFILE_SUMMARY = OUT / "profile_summary_expanded42_v2.csv"
TASK_SUMMARY = OUT / "task_summary_expanded42_v2.csv"

TASKS = [
    20, 21, 22, 23, 24,
    28, 29, 30, 31, 32, 33, 34,
    37, 38, 39, 40, 41, 42,
]

OLD_PROFILES = [f"P{i:03d}" for i in range(14, 36)]
NEW_PROFILES = [f"P{i:03d}" for i in range(36, 56)]
ALL_PROFILES = OLD_PROFILES + NEW_PROFILES

FS_HZ = 100
WINDOW = 30
STRIDE = 15

EVENT_POLICY_VERSION = (
    "repository_peak_impact_magnitude_within_3s_after_fall_onset_v2"
)

ELIGIBILITY_POLICY = (
    ">=30 actual samples on corrected 100Hz grid and "
    ">=1 complete 30-sample window"
)

REQUIRED_COLUMNS = [
    "profile_id",
    "task_id",
    "source_run_dir",
    "truth_csv",
    "fall_onset_time_s",
    "original_manifest_impact_time_s",
    "cnn_recovered_impact_time_s",
    "cnn_lead_time_ms",
    "fall_samples_100hz",
    "raw_falling_window_count",
    "raw_activity_window_count",
    "cnn_event_order_valid",
    "cnn_event_inside_truth",
    "cnn_fall_eligible",
    "impact_recovery_source",
    "repository_methods_agree",
    "recovery_status",
    "event_policy_version",
    "eligibility_policy",
    "sampling_rate_hz",
    "window_samples",
    "stride_samples",
    "nominal_window_ms",
    "window_overlap_percent",
    "cnn_exclusion_reason",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def finite_float(x):
    try:
        y = float(x)
    except Exception:
        return None
    return y if math.isfinite(y) else None


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def n_windows(n_samples: int) -> int:
    n_samples = int(n_samples)
    if n_samples < WINDOW:
        return 0
    return 1 + (n_samples - WINDOW) // STRIDE


def find_legacy_signal_csv(run_dir: Path) -> Path:
    candidates = []

    for p in sorted(run_dir.glob("*.csv")):
        if p.name.endswith("_highrate_truth.csv"):
            continue

        try:
            h = pd.read_csv(p, comment="#", nrows=4)
        except Exception:
            continue

        if "impact_magnitude" not in h.columns:
            continue

        time_cols = [
            c for c in ["timestamp", "time", "time_s", "simulation_time_s"]
            if c in h.columns
        ]

        if not time_cols:
            continue

        candidates.append(p)

    if len(candidates) != 1:
        raise RuntimeError(
            f"{run_dir}: expected exactly one legacy CSV containing "
            f"impact_magnitude + time; found {candidates}"
        )

    return candidates[0]


def read_legacy_signal(path: Path):
    q = pd.read_csv(path, comment="#")

    if "impact_magnitude" not in q.columns:
        raise RuntimeError(f"{path}: impact_magnitude missing")

    time_col = next(
        (
            c
            for c in ["timestamp", "time", "time_s", "simulation_time_s"]
            if c in q.columns
        ),
        None,
    )

    if time_col is None:
        raise RuntimeError(f"{path}: no recognized time column")

    t = pd.to_numeric(q[time_col], errors="coerce").to_numpy(float)
    impact = pd.to_numeric(
        q["impact_magnitude"],
        errors="coerce",
    ).to_numpy(float)

    return q, time_col, t, impact


def recover_peak_impact(run_dir: Path, onset: float):
    """
    Two independently expressed implementations of the frozen rule are used
    as an internal agreement check.

    Rule:
      peak `impact_magnitude` in [fall onset, fall onset + 3.0 s].
    """

    source = find_legacy_signal_csv(run_dir)
    q, time_col, t, v = read_legacy_signal(source)

    valid = (
        np.isfinite(t)
        & np.isfinite(v)
        & (t >= float(onset))
        & (t <= float(onset) + 3.0)
    )

    idx = np.flatnonzero(valid)

    if not len(idx):
        raise RuntimeError(
            f"{source}: no finite impact_magnitude samples within 3 s of onset"
        )

    # Method A: NumPy.
    j_np = int(idx[np.argmax(v[idx])])
    t_np = float(t[j_np])

    # Method B: pandas idxmax on the same policy interval.
    sub = q.loc[valid, [time_col, "impact_magnitude"]].copy()
    sub[time_col] = pd.to_numeric(sub[time_col], errors="coerce")
    sub["impact_magnitude"] = pd.to_numeric(
        sub["impact_magnitude"],
        errors="coerce",
    )
    sub = sub.dropna()

    if sub.empty:
        raise RuntimeError(f"{source}: empty pandas impact interval")

    j_pd = sub["impact_magnitude"].idxmax()
    t_pd = float(sub.loc[j_pd, time_col])

    methods_agree = abs(t_np - t_pd) <= 1e-9

    if not methods_agree:
        raise RuntimeError(
            f"{source}: independent impact implementations disagree: "
            f"{t_np} vs {t_pd}"
        )

    return {
        "impact_s": t_np,
        "legacy_csv": str(source),
        "time_column": time_col,
        "methods_agree": True,
    }


def read_event_summary(run_dir: Path):
    p = run_dir / "run_manifest.json"

    if not p.exists():
        raise RuntimeError(f"Missing run_manifest.json: {p}")

    obj = json.loads(p.read_text(encoding="utf-8"))

    try:
        ev = obj["summary"]["event_summary"]
    except Exception as exc:
        raise RuntimeError(
            f"{p}: summary.event_summary missing"
        ) from exc

    onset = finite_float(ev.get("onset_time"))
    impact = finite_float(ev.get("impact_time"))

    if onset is None:
        raise RuntimeError(f"{p}: finite onset_time missing")

    if impact is None:
        raise RuntimeError(f"{p}: finite impact_time missing")

    # IMPORTANT:
    # original manifest impact is retained only as provenance.
    #
    # In the frozen publication cohort some original impact timestamps can
    # equal the fall onset (e.g. P014/task22). Those values were NOT used as
    # the final CNN event endpoint. The canonical policy independently
    # recovers the impact as the peak legacy `impact_magnitude` within
    # 3 seconds after onset.
    #
    # Therefore original impact <= onset is permitted here. Event-order
    # validity is assessed later using the RECOVERED impact.
    return onset, impact


def truth_counts(
    truth_csv: Path,
    onset: float,
    recovered_impact: float,
):
    q = pd.read_csv(
        truth_csv,
        comment="#",
        usecols=["timestamp"],
    )

    t = pd.to_numeric(
        q["timestamp"],
        errors="coerce",
    ).to_numpy(float)

    if not np.isfinite(t).all():
        raise RuntimeError(f"{truth_csv}: nonfinite timestamp")

    if len(t) < 2:
        raise RuntimeError(f"{truth_csv}: insufficient truth samples")

    activity_samples = int(
        np.sum(t < float(onset))
    )

    falling_samples = int(
        np.sum(
            (t >= float(onset))
            & (t < float(recovered_impact))
        )
    )

    inside = bool(
        float(onset) >= float(np.min(t)) - 1e-9
        and
        float(recovered_impact) <= float(np.max(t)) + 1e-9
    )

    return {
        "activity_samples": activity_samples,
        "falling_samples": falling_samples,
        "activity_windows": n_windows(activity_samples),
        "falling_windows": n_windows(falling_samples),
        "inside_truth": inside,
        "truth_rows": int(len(t)),
    }


def build_policy_row(
    *,
    profile_id: str,
    task_id: int,
    run_dir: Path,
    truth_csv: Path,
):
    onset, original_impact = read_event_summary(run_dir)

    rec = recover_peak_impact(
        run_dir,
        onset,
    )

    recovered = float(rec["impact_s"])

    counts = truth_counts(
        truth_csv,
        onset,
        recovered,
    )

    order_valid = bool(recovered > onset)
    eligible = bool(
        order_valid
        and counts["inside_truth"]
        and counts["falling_samples"] >= WINDOW
        and counts["falling_windows"] >= 1
    )

    exclusion = (
        ""
        if eligible
        else "recovered_interval_shorter_than_30_samples"
    )

    return {
        "profile_id": str(profile_id),
        "task_id": int(task_id),
        "source_run_dir": str(run_dir.resolve()),
        "truth_csv": str(truth_csv.resolve()),
        "fall_onset_time_s": float(onset),
        "original_manifest_impact_time_s": float(original_impact),
        "cnn_recovered_impact_time_s": float(recovered),
        "cnn_lead_time_ms": float((recovered - onset) * 1000.0),
        "fall_samples_100hz": int(counts["falling_samples"]),
        "raw_falling_window_count": int(counts["falling_windows"]),
        "raw_activity_window_count": int(counts["activity_windows"]),
        "cnn_event_order_valid": bool(order_valid),
        "cnn_event_inside_truth": bool(counts["inside_truth"]),
        "cnn_fall_eligible": bool(eligible),
        "impact_recovery_source":
            "peak_impact_magnitude_within_3s_after_fall_onset",
        "repository_methods_agree": bool(rec["methods_agree"]),
        "recovery_status": "ok",
        "event_policy_version": EVENT_POLICY_VERSION,
        "eligibility_policy": ELIGIBILITY_POLICY,
        "sampling_rate_hz": 100,
        "window_samples": 30,
        "stride_samples": 15,
        "nominal_window_ms": 300,
        "window_overlap_percent": 50,
        "cnn_exclusion_reason": exclusion,
    }


def validate_old396():
    print("=" * 104)
    print("REGRESSION: RECONSTRUCT FROZEN 396-ROW EVENT POLICY")
    print("=" * 104)

    if not OLD_CANON.exists():
        raise RuntimeError(f"Missing frozen canonical manifest: {OLD_CANON}")

    actual_sha = sha256_file(OLD_CANON)

    print("Frozen manifest:", OLD_CANON)
    print("Expected SHA256:", EXPECTED_OLD_SHA256)
    print("Actual SHA256  :", actual_sha)

    if actual_sha != EXPECTED_OLD_SHA256:
        raise RuntimeError(
            "Frozen 396 canonical manifest SHA256 mismatch"
        )

    old = pd.read_csv(OLD_CANON)

    if list(old.columns) != REQUIRED_COLUMNS:
        raise RuntimeError(
            "Frozen canonical manifest column contract changed.\n"
            f"Expected: {REQUIRED_COLUMNS}\n"
            f"Actual  : {list(old.columns)}"
        )

    if len(old) != 396:
        raise RuntimeError(f"Expected 396 frozen rows, got {len(old)}")

    if sorted(old["profile_id"].astype(str).unique()) != OLD_PROFILES:
        raise RuntimeError("Frozen profile set does not equal P014--P035")

    if sorted(pd.to_numeric(old["task_id"]).astype(int).unique()) != TASKS:
        raise RuntimeError("Frozen task set does not equal the 18 paper tasks")

    regression = []

    for n, row in old.iterrows():
        profile = str(row["profile_id"])
        task = int(row["task_id"])
        run_dir = Path(str(row["source_run_dir"]))
        truth_csv = Path(str(row["truth_csv"]))

        rebuilt = build_policy_row(
            profile_id=profile,
            task_id=task,
            run_dir=run_dir,
            truth_csv=truth_csv,
        )

        onset_err = abs(
            float(rebuilt["fall_onset_time_s"])
            - float(row["fall_onset_time_s"])
        )

        original_impact_err = abs(
            float(rebuilt["original_manifest_impact_time_s"])
            - float(row["original_manifest_impact_time_s"])
        )

        recovered_impact_err = abs(
            float(rebuilt["cnn_recovered_impact_time_s"])
            - float(row["cnn_recovered_impact_time_s"])
        )

        sample_match = (
            int(rebuilt["fall_samples_100hz"])
            == int(row["fall_samples_100hz"])
        )

        fall_window_match = (
            int(rebuilt["raw_falling_window_count"])
            == int(row["raw_falling_window_count"])
        )

        activity_window_match = (
            int(rebuilt["raw_activity_window_count"])
            == int(row["raw_activity_window_count"])
        )

        eligible_match = (
            bool(rebuilt["cnn_fall_eligible"])
            == bool(row["cnn_fall_eligible"])
        )

        exclusion_old = (
            ""
            if pd.isna(row["cnn_exclusion_reason"])
            else str(row["cnn_exclusion_reason"])
        )

        exclusion_match = (
            str(rebuilt["cnn_exclusion_reason"])
            == exclusion_old
        )

        ok = bool(
            onset_err <= 1e-6
            and original_impact_err <= 1e-6
            and recovered_impact_err <= 0.001
            and sample_match
            and fall_window_match
            and activity_window_match
            and eligible_match
            and exclusion_match
        )

        regression.append({
            "profile_id": profile,
            "task_id": task,
            "onset_abs_error_s": onset_err,
            "original_impact_abs_error_s": original_impact_err,
            "recovered_impact_abs_error_s": recovered_impact_err,
            "fall_samples_match": sample_match,
            "fall_windows_match": fall_window_match,
            "activity_windows_match": activity_window_match,
            "eligible_match": eligible_match,
            "exclusion_match": exclusion_match,
            "regression_pass": ok,
        })

        if (n + 1) % 50 == 0:
            print(f"  reconstructed {n + 1}/396")

    reg = pd.DataFrame(regression)
    reg.to_csv(REGRESSION_PATH, index=False)

    n_pass = int(reg["regression_pass"].sum())

    print()
    print("Rows reconstructed:", len(reg))
    print("Rows passing      :", n_pass)
    print(
        "Max recovered-impact error [ms]:",
        float(reg["recovered_impact_abs_error_s"].max()) * 1000.0,
    )

    if n_pass != 396:
        bad = reg.loc[
            ~reg["regression_pass"],
            [
                "profile_id",
                "task_id",
                "onset_abs_error_s",
                "original_impact_abs_error_s",
                "recovered_impact_abs_error_s",
                "fall_samples_match",
                "fall_windows_match",
                "activity_windows_match",
                "eligible_match",
                "exclusion_match",
            ],
        ]

        print("\nFIRST FAILURES:")
        print(bad.head(25).to_string(index=False))

        raise RuntimeError(
            f"OLD-396 EVENT-POLICY REGRESSION FAILED: "
            f"{396 - n_pass} rows differ"
        )

    print("OLD-396 EVENT-POLICY REGRESSION: PASS")

    return old


def validate_new_campaign():
    if not NEW_RUN_MANIFEST.exists():
        raise RuntimeError(
            f"New 360-run campaign is not complete/missing manifest: "
            f"{NEW_RUN_MANIFEST}"
        )

    if not NEW_PROGRESS.exists():
        raise RuntimeError(
            f"Missing new campaign progress table: {NEW_PROGRESS}"
        )

    progress = pd.read_csv(NEW_PROGRESS)

    if len(progress) != 360:
        raise RuntimeError(
            f"Expected 360 progress rows, got {len(progress)}"
        )

    if progress.duplicated(
        ["profile_id", "scenario_id"]
    ).any():
        raise RuntimeError(
            "Duplicate profile/task pair in V2 progress table"
        )

    statuses = progress["status"].astype(str)

    if not statuses.eq("completed").all():
        print(progress["status"].value_counts(dropna=False).to_string())
        raise RuntimeError(
            "Not every V2 simulation has status='completed'"
        )

    if not bool_series(progress["truth_qc_ok"]).all():
        raise RuntimeError(
            "Not every V2 simulation passed high-rate truth QC"
        )

    rm = pd.read_csv(NEW_RUN_MANIFEST)

    if len(rm) != 360:
        raise RuntimeError(
            f"Expected 360 run-manifest rows, got {len(rm)}"
        )

    if rm.duplicated(
        ["profile_id", "scenario_id"]
    ).any():
        raise RuntimeError(
            "Duplicate profile/task pair in V2 run manifest"
        )

    if sorted(rm["profile_id"].astype(str).unique()) != NEW_PROFILES:
        raise RuntimeError(
            "V2 run manifest does not contain exactly P036--P055"
        )

    if sorted(
        pd.to_numeric(rm["scenario_id"]).astype(int).unique()
    ) != TASKS:
        raise RuntimeError(
            "V2 run manifest task set differs from frozen 18 tasks"
        )

    per_profile = rm.groupby("profile_id").size()

    if not (per_profile == 18).all():
        raise RuntimeError(
            f"Every new profile must have 18 runs:\n{per_profile}"
        )

    if not bool_series(rm["truth_qc_ok"]).all():
        raise RuntimeError(
            "New run manifest contains failed truth-QC rows"
        )

    return rm.sort_values(
        ["profile_id", "scenario_id"]
    ).reset_index(drop=True)


def build_new360(rm: pd.DataFrame):
    rows = []

    print()
    print("=" * 104)
    print("BUILD NEW 360 EVENT-POLICY ROWS")
    print("=" * 104)

    for n, r in rm.iterrows():
        profile = str(r["profile_id"])
        task = int(r["scenario_id"])
        run_dir = Path(str(r["output_dir"]))
        truth_csv = Path(str(r["highrate_truth_csv"]))

        if not run_dir.exists():
            raise RuntimeError(f"Missing run directory: {run_dir}")

        if not truth_csv.exists():
            raise RuntimeError(f"Missing truth CSV: {truth_csv}")

        rows.append(
            build_policy_row(
                profile_id=profile,
                task_id=task,
                run_dir=run_dir,
                truth_csv=truth_csv,
            )
        )

        if (n + 1) % 50 == 0:
            print(f"  built {n + 1}/360")

    new = pd.DataFrame(rows)

    if list(new.columns) != REQUIRED_COLUMNS:
        raise RuntimeError(
            "New event rows do not match frozen 25-column contract"
        )

    if len(new) != 360:
        raise RuntimeError(
            f"Expected 360 new event rows, got {len(new)}"
        )

    return new


def final_gate(old: pd.DataFrame, new: pd.DataFrame):
    combined = pd.concat(
        [old[REQUIRED_COLUMNS], new[REQUIRED_COLUMNS]],
        ignore_index=True,
    )

    combined = combined.sort_values(
        ["profile_id", "task_id"]
    ).reset_index(drop=True)

    checks = {}

    checks["rows_756"] = len(combined) == 756

    checks["profiles_42"] = (
        sorted(combined["profile_id"].astype(str).unique())
        == ALL_PROFILES
    )

    checks["tasks_18"] = (
        sorted(
            pd.to_numeric(combined["task_id"]).astype(int).unique()
        )
        == TASKS
    )

    checks["profile_task_unique"] = (
        not combined.duplicated(
            ["profile_id", "task_id"]
        ).any()
    )

    counts = (
        combined.groupby("profile_id")
        .size()
        .reindex(ALL_PROFILES)
    )

    checks["18_tasks_per_profile"] = bool(
        counts.notna().all()
        and (counts == 18).all()
    )

    checks["status_ok_756"] = bool(
        combined["recovery_status"]
        .astype(str)
        .eq("ok")
        .all()
    )

    checks["repository_agreement_756"] = bool(
        bool_series(
            combined["repository_methods_agree"]
        ).all()
    )

    checks["truth_files_exist_756"] = bool(
        combined["truth_csv"]
        .astype(str)
        .map(lambda x: Path(x).exists())
        .all()
    )

    checks["event_policy_consistent"] = bool(
        combined["event_policy_version"]
        .astype(str)
        .eq(EVENT_POLICY_VERSION)
        .all()
    )

    eligible = bool_series(
        combined["cnn_fall_eligible"]
    )

    checks["eligible_have_valid_event_order"] = bool(
        bool_series(
            combined.loc[
                eligible,
                "cnn_event_order_valid",
            ]
        ).all()
    )

    checks["eligible_events_inside_truth"] = bool(
        bool_series(
            combined.loc[
                eligible,
                "cnn_event_inside_truth",
            ]
        ).all()
    )

    checks["eligible_have_at_least_30_samples"] = bool(
        (
            pd.to_numeric(
                combined.loc[
                    eligible,
                    "fall_samples_100hz",
                ]
            )
            >= 30
        ).all()
    )

    checks["eligible_have_complete_window"] = bool(
        (
            pd.to_numeric(
                combined.loc[
                    eligible,
                    "raw_falling_window_count",
                ]
            )
            >= 1
        ).all()
    )

    checks["ineligible_have_zero_complete_windows"] = bool(
        (
            pd.to_numeric(
                combined.loc[
                    ~eligible,
                    "raw_falling_window_count",
                ]
            )
            == 0
        ).all()
    )

    eligible_profiles = set(
        combined.loc[
            eligible,
            "profile_id",
        ].astype(str)
    )

    eligible_tasks = set(
        pd.to_numeric(
            combined.loc[
                eligible,
                "task_id",
            ]
        ).astype(int)
    )

    checks["every_profile_has_eligible_trial"] = (
        eligible_profiles == set(ALL_PROFILES)
    )

    checks["every_task_has_eligible_trial"] = (
        eligible_tasks == set(TASKS)
    )

    # Critical preservation gate:
    # the old 396 semantic values must be unchanged after concatenation.
    old_recovered = (
        combined[
            combined["profile_id"].isin(OLD_PROFILES)
        ]
        .sort_values(["profile_id", "task_id"])
        .reset_index(drop=True)
    )

    frozen_sorted = (
        old[REQUIRED_COLUMNS]
        .sort_values(["profile_id", "task_id"])
        .reset_index(drop=True)
    )

    try:
        pd.testing.assert_frame_equal(
            old_recovered[REQUIRED_COLUMNS],
            frozen_sorted[REQUIRED_COLUMNS],
            check_dtype=False,
            check_exact=False,
            rtol=0,
            atol=1e-12,
        )
        checks["frozen_old396_values_preserved"] = True
    except AssertionError:
        checks["frozen_old396_values_preserved"] = False

    gate_pass = bool(all(checks.values()))

    combined.to_csv(
        COMBINED_MANIFEST,
        index=False,
    )

    profile_summary = (
        combined.groupby("profile_id")
        .agg(
            source_trials=("task_id", "size"),
            eligible_trials=("cnn_fall_eligible", lambda s: int(bool_series(s).sum())),
            falling_windows=("raw_falling_window_count", "sum"),
            activity_windows=("raw_activity_window_count", "sum"),
        )
        .reset_index()
    )

    task_summary = (
        combined.groupby("task_id")
        .agg(
            source_trials=("profile_id", "size"),
            eligible_trials=("cnn_fall_eligible", lambda s: int(bool_series(s).sum())),
            falling_windows=("raw_falling_window_count", "sum"),
            activity_windows=("raw_activity_window_count", "sum"),
        )
        .reset_index()
    )

    profile_summary.to_csv(
        PROFILE_SUMMARY,
        index=False,
    )

    task_summary.to_csv(
        TASK_SUMMARY,
        index=False,
    )

    eligible_df = combined.loc[eligible].copy()

    facts = {
        "dataset_generation": "expanded_42_profile_v2",
        "frozen_old_source_simulations": 396,
        "new_source_simulations": 360,
        "source_simulations": int(len(combined)),
        "profiles": int(combined["profile_id"].nunique()),
        "female_profiles": 20,
        "male_profiles": 22,
        "tasks": int(combined["task_id"].nunique()),
        "eligible_trials": int(eligible.sum()),
        "excluded_trials": int((~eligible).sum()),
        "raw_falling_windows": int(
            pd.to_numeric(
                eligible_df["raw_falling_window_count"]
            ).sum()
        ),
        "eligible_activity_windows": int(
            pd.to_numeric(
                eligible_df["raw_activity_window_count"]
            ).sum()
        ),
        "minimum_eligible_lead_ms": float(
            pd.to_numeric(
                eligible_df["cnn_lead_time_ms"]
            ).min()
        ),
        "eligible_below_299ms": int(
            (
                pd.to_numeric(
                    eligible_df["cnn_lead_time_ms"]
                )
                < 299.0
            ).sum()
        ),
        "sampling_rate_hz": FS_HZ,
        "window_samples": WINDOW,
        "stride_samples": STRIDE,
        "nominal_window_ms": 300,
        "window_overlap_percent": 50,
        "eligibility_definition": (
            "at least 30 actual corrected-100Hz samples between "
            "onset inclusive and recovered impact exclusive, yielding "
            "at least one complete 30-sample window"
        ),
        "event_policy": (
            "peak impact_magnitude within 3 seconds after fall onset"
        ),
        "frozen_old_manifest": str(OLD_CANON),
        "frozen_old_manifest_sha256": EXPECTED_OLD_SHA256,
        "combined_manifest": str(COMBINED_MANIFEST),
        "combined_manifest_sha256": sha256_file(COMBINED_MANIFEST),
        "checks": checks,
        "gate_pass": gate_pass,
    }

    FACTS_PATH.write_text(
        json.dumps(
            facts,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 104)
    print("EXPANDED 42-PROFILE EVENT-POLICY GATE")
    print("=" * 104)

    for k, v in checks.items():
        print(f"{k:44s}: {v}")

    print()
    print("Source simulations :", len(combined))
    print("Profiles           :", combined["profile_id"].nunique())
    print("Tasks              :", combined["task_id"].nunique())
    print("Eligible trials    :", int(eligible.sum()))
    print("Excluded trials    :", int((~eligible).sum()))
    print(
        "Falling windows    :",
        facts["raw_falling_windows"],
    )
    print(
        "Activity windows   :",
        facts["eligible_activity_windows"],
    )

    if not gate_pass:
        raise RuntimeError(
            "EXPANDED 42-PROFILE EVENT-POLICY GATE FAILED"
        )

    (OUT / "COMPLETE").write_text(
        "EXPANDED42 CANONICAL EVENT POLICY COMPLETE\n",
        encoding="utf-8",
    )

    print()
    print("Combined manifest :", COMBINED_MANIFEST)
    print("Facts             :", FACTS_PATH)
    print("EXPANDED 42-PROFILE EVENT-POLICY GATE: PASS")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--validate-old-only",
        action="store_true",
        help=(
            "Regression-test the reconstructed event policy against "
            "the frozen 396 publication rows, then stop."
        ),
    )

    args = ap.parse_args()

    old = validate_old396()

    if args.validate_old_only:
        print()
        print(
            "OLD-396 REGRESSION COMPLETE. "
            "No new simulations or combined manifest were required."
        )
        return

    rm = validate_new_campaign()
    new = build_new360(rm)
    final_gate(old, new)


if __name__ == "__main__":
    main()
