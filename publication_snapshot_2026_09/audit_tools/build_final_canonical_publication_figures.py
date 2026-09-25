#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import shutil
import zipfile

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path("/mnt/hdd16T/ToqeerHomeBackup/mujoco_project")
CORRECTED = (
    PROJECT
    / "outputs/_highrate_overnight/"
      "campaign_highrate_truth_v3_rne_corrected396"
)
RUNS = CORRECTED / "runs"
PUB = PROJECT / "outputs/phase2_publication_gate_audit_20260826"

STAGE_DEFAULT = PROJECT / "outputs/paper_evidence_archive_v1"
ZIP_DEFAULT = PROJECT / "outputs/paper_evidence_archive_FINAL.zip"

OUT_REL = Path("14_FINAL_CANONICAL_PUBLICATION_FIGURES")

CHECKS = [
    "head_velocity_safe",
    "duration_realistic",
    "jerk_realistic",
    "protective_response",
    "imu_quality",
    "contact_pattern",
]

REP_TARGET_AGES = [20, 30, 53, 63, 74, 78]
REP_TASK = 21


def sha256_file(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_sim_csv(path: Path, nrows=None) -> pd.DataFrame:
    return pd.read_csv(
        path,
        comment="#",
        nrows=nrows,
        low_memory=False,
    )


def parse_path(path: Path | str) -> dict:
    s = str(path)

    def m(pattern, flags=0):
        x = re.search(pattern, s, flags)
        return x

    profile = m(r"/(P\d{3})/")
    task = m(r"/task_(\d+)/")
    age = m(r"(?:^|[_/])age(\d{1,3})(?:[_/]|$)")
    sex = m(r"(?:^|[_/])sex_(male|female)(?:[_/]|$)", re.I)
    height = m(r"(?:^|[_/])h(\d+)p(\d+)(?:[_/]|$)", re.I)
    weight = m(r"(?:^|[_/])w(\d+)p(\d+)(?:[_/]|$)", re.I)

    def dec(mm):
        if not mm:
            return np.nan
        return float(f"{int(mm.group(1))}.{mm.group(2)}")

    return {
        "profile": profile.group(1) if profile else "",
        "task": int(task.group(1)) if task else np.nan,
        "age": int(age.group(1)) if age else np.nan,
        "sex": sex.group(1).lower() if sex else "",
        "height_m": dec(height),
        "weight_kg": dec(weight),
    }


def age_band(age: float) -> str:
    if age < 30:
        return "<30"
    if age < 50:
        return "30-49"
    if age < 65:
        return "50-64"
    return "65+"


def magnitude(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    a = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return np.sqrt(np.nansum(a * a, axis=1))


def save_fig(fig, base: Path):
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(base.with_suffix("." + ext), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def discover_trials():
    standard = []
    for p in sorted(RUNS.rglob("*.csv")):
        if p.name.endswith("_highrate_truth.csv"):
            continue
        meta = parse_path(p)
        if pd.isna(meta["age"]) or pd.isna(meta["task"]):
            continue
        standard.append({**meta, "file": str(p)})

    return pd.DataFrame(standard)


def discover_validation_reports():
    rows = []

    for p in sorted(RUNS.rglob("*_validation.txt")):
        text = p.read_text(encoding="utf-8", errors="replace")
        meta = parse_path(p)

        conf = re.search(r"Overall Confidence:\s*([0-9.]+)%", text)
        cls = re.search(r"Classification:\s*([A-Z_]+)", text)

        row = {
            **meta,
            "confidence": float(conf.group(1)) if conf else np.nan,
            "classification": cls.group(1) if cls else "",
            "file": str(p),
        }

        for c in CHECKS:
            mm = re.search(rf"{re.escape(c)}\s+(PASS|FAIL)", text)
            row[c] = mm.group(1) if mm else ""

        rows.append(row)

    return pd.DataFrame(rows)


def select_representatives(trials: pd.DataFrame):
    reps = []
    for target in REP_TARGET_AGES:
        x = trials.copy()
        x["age_distance"] = (x["age"] - target).abs()
        x["task_preference"] = (x["task"] != REP_TASK).astype(int)
        x = x.sort_values(
            [
                "age_distance",
                "task_preference",
                "profile",
                "task",
                "file",
            ]
        )
        reps.append(x.iloc[0].to_dict())
    return pd.DataFrame(reps)


def matching_truth(standard_path: Path) -> Path:
    p = standard_path.with_name(
        standard_path.stem + "_highrate_truth.csv"
    )
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def aligned_gyro(standard: pd.DataFrame, truth: pd.DataFrame):
    st = pd.to_numeric(standard["timestamp"], errors="coerce").to_numpy(float)
    tt = pd.to_numeric(truth["timestamp"], errors="coerce").to_numpy(float)

    valid_t = np.isfinite(tt)
    tt = tt[valid_t]

    out = {"timestamp": st}

    for a, b in zip(
        ["gyro_x", "gyro_y", "gyro_z"],
        ["gyro_true_x", "gyro_true_y", "gyro_true_z"],
    ):
        y = pd.to_numeric(truth[b], errors="coerce").to_numpy(float)[valid_t]
        ok = np.isfinite(tt) & np.isfinite(y)
        if ok.sum() < 2:
            out[b] = np.full_like(st, np.nan)
        else:
            out[b] = np.interp(st, tt[ok], y[ok], left=np.nan, right=np.nan)

        out[a] = pd.to_numeric(
            standard[a], errors="coerce"
        ).to_numpy(float)

    return pd.DataFrame(out)


def generate_validation_summary(vr: pd.DataFrame, outdir: Path, index_rows: list):
    summary_rows = []

    for c in CHECKS:
        valid = vr[vr[c].isin(["PASS", "FAIL"])]
        passed = int(valid[c].eq("PASS").sum())
        total = int(len(valid))
        summary_rows.append({
            "check": c,
            "pass_count": passed,
            "total": total,
            "pass_rate_percent": 100.0 * passed / total if total else np.nan,
        })

    check_df = pd.DataFrame(summary_rows)
    write_csv(check_df, outdir / "data/validation_check_pass_rates.csv")

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(check_df))
    y = check_df["pass_rate_percent"].to_numpy(float)
    ax.bar(x, y)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [s.replace("_", " ") for s in check_df["check"]],
        rotation=20,
        ha="right",
    )
    ax.set_ylim(0, 105)
    ax.set_ylabel("Pass rate (%)")
    ax.set_title("Physical-plausibility validation across 396 corrected simulations")
    ax.grid(axis="y", alpha=0.25)

    for i, (yy, p, n) in enumerate(
        zip(y, check_df["pass_count"], check_df["total"])
    ):
        ax.text(i, yy + 1.5, f"{yy:.1f}%\n({p}/{n})", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    save_fig(fig, outdir / "figures/FIG_S01_validation_check_pass_rates")

    index_rows.append({
        "figure_id": "FIG_S01",
        "title": "Physical-plausibility validation pass rates",
        "use": "Aggregate simulator validation",
        "population": "All 396 corrected simulations",
        "selection_rule": "All corrected396 validation reports; no trial selection",
        "source": str(RUNS),
        "data_file": "data/validation_check_pass_rates.csv",
        "important_note": (
            "Jerk realism pass rate is intentionally shown, including failures; "
            "do not describe all plausibility checks as uniformly high."
        ),
    })

    # Confidence distribution
    conf = pd.to_numeric(vr["confidence"], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(7.3, 4.6))
    ax.hist(conf, bins=np.arange(65, 101, 2.5))
    ax.axvline(conf.mean(), linestyle="--", linewidth=1.2, label=f"Mean {conf.mean():.1f}%")
    ax.axvline(conf.median(), linestyle=":", linewidth=1.2, label=f"Median {conf.median():.1f}%")
    ax.set_xlabel("Validation confidence (%)")
    ax.set_ylabel("Number of simulations")
    ax.set_title("Distribution of validation confidence")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_fig(fig, outdir / "figures/FIG_S02_validation_confidence_distribution")

    write_csv(
        vr[
            [
                "profile", "task", "age", "sex",
                "confidence", "classification", "file"
            ]
        ],
        outdir / "data/validation_confidence_all396.csv",
    )

    index_rows.append({
        "figure_id": "FIG_S02",
        "title": "Validation confidence distribution",
        "use": "Shows full-campaign variability, not only best trials",
        "population": "All 396 corrected simulations",
        "selection_rule": "All corrected396 validation reports",
        "source": str(RUNS),
        "data_file": "data/validation_confidence_all396.csv",
        "important_note": "Report the observed range and HIGH/MODERATE counts transparently.",
    })

    # Task heatmap
    task_rows = []
    for task, g in vr.groupby("task"):
        r = {"task": int(task), "confidence_mean": pd.to_numeric(g["confidence"], errors="coerce").mean()}
        for c in CHECKS:
            valid = g[g[c].isin(["PASS", "FAIL"])]
            r[c] = 100.0 * valid[c].eq("PASS").mean() if len(valid) else np.nan
        task_rows.append(r)

    task_df = pd.DataFrame(task_rows).sort_values("task")
    write_csv(task_df, outdir / "data/validation_by_task.csv")

    hm_cols = CHECKS
    mat = task_df[hm_cols].to_numpy(float)

    fig, ax = plt.subplots(figsize=(10.0, 6.8))
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(hm_cols)))
    ax.set_xticklabels([c.replace("_", " ") for c in hm_cols], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(task_df)))
    ax.set_yticklabels(task_df["task"].astype(int).astype(str))
    ax.set_xlabel("Validation check")
    ax.set_ylabel("Task ID")
    ax.set_title("Task-wise physical-plausibility pass rates (%)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Pass rate (%)")
    fig.tight_layout()
    save_fig(fig, outdir / "figures/FIG_S03_taskwise_validation_heatmap")

    index_rows.append({
        "figure_id": "FIG_S03",
        "title": "Task-wise physical-plausibility pass rates",
        "use": "Shows which checks/tasks drive validation limitations",
        "population": "All 396 corrected simulations grouped by task",
        "selection_rule": "All corrected396 validation reports",
        "source": str(RUNS),
        "data_file": "data/validation_by_task.csv",
        "important_note": "Useful as main or supplementary figure; exposes task-dependent failure patterns.",
    })


def generate_profile_coverage(trials: pd.DataFrame, outdir: Path, index_rows: list):
    profiles = trials[
        ["profile", "age", "sex", "height_m", "weight_kg"]
    ].drop_duplicates("profile").sort_values("age")

    profiles["age_band"] = profiles["age"].map(age_band)
    write_csv(profiles, outdir / "data/corrected396_profile_demographics.csv")

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for sex, g in profiles.groupby("sex"):
        ax.scatter(
            g["age"],
            g["weight_kg"],
            label=sex if sex else "unspecified",
            s=55,
        )
    ax.set_xlabel("Simulated profile age (years)")
    ax.set_ylabel("Simulated body mass (kg)")
    ax.set_title("Demographic parameter coverage of corrected simulation profiles")
    ax.legend(title="Simulated sex")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_fig(fig, outdir / "figures/FIG_S04_corrected396_profile_coverage")

    index_rows.append({
        "figure_id": "FIG_S04",
        "title": "Corrected396 simulated profile coverage",
        "use": "Simulator parameterization / diversity",
        "population": f"{profiles['profile'].nunique()} corrected simulation profiles",
        "selection_rule": "One demographic record per corrected396 profile",
        "source": str(RUNS),
        "data_file": "data/corrected396_profile_demographics.csv",
        "important_note": "These are simulated demographic profiles, not physical participant demographics.",
    })


def representative_signal_data(rep: pd.Series):
    p = Path(rep["file"])
    standard = read_sim_csv(p)
    truth = read_sim_csv(matching_truth(p))

    standard = standard.copy()
    truth = truth.copy()

    for c in standard.columns:
        if c != "timestamp":
            standard[c] = pd.to_numeric(standard[c], errors="coerce")
    for c in truth.columns:
        if c != "timestamp":
            truth[c] = pd.to_numeric(truth[c], errors="coerce")

    return p, standard, truth


def generate_representative_waveforms(reps: pd.DataFrame, outdir: Path, index_rows: list):
    rep_index_rows = []
    for _, r in reps.iterrows():
        p = Path(r["file"])
        rep_index_rows.append({
            "target_age": int(r["target_age"]) if "target_age" in r else int(r["age"]),
            "actual_age": int(r["age"]),
            "profile": r["profile"],
            "sex": r["sex"],
            "height_m": r["height_m"],
            "weight_kg": r["weight_kg"],
            "task": int(r["task"]),
            "standard_file": str(p),
            "truth_file": str(matching_truth(p)),
            "standard_sha256": sha256_file(p),
            "truth_sha256": sha256_file(matching_truth(p)),
        })

    rep_index = pd.DataFrame(rep_index_rows)
    write_csv(rep_index, outdir / "data/representative_trials_predefined.csv")

    n = len(reps)
    ncols = 3
    nrows = math.ceil(n / ncols)

    # A. acceleration magnitudes
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.2, 3.6*nrows), squeeze=False)

    for ax, (_, r) in zip(axes.ravel(), reps.iterrows()):
        p, d, truth = representative_signal_data(r)

        t = pd.to_numeric(d["timestamp"], errors="coerce").to_numpy(float)
        am = magnitude(d, ["accel_x", "accel_y", "accel_z"])
        atm = magnitude(d, ["accel_true_x", "accel_true_y", "accel_true_z"])

        ax.plot(t, am, label="Sensor-model acceleration")
        ax.plot(t, atm, label="Physics-truth acceleration", linewidth=1.0)
        ax.set_title(
            f"{r['profile']} | age {int(r['age'])} | {r['sex']} | task {int(r['task'])}"
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("|a|")
        ax.grid(alpha=0.2)

    for ax in axes.ravel()[n:]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(
        "Representative corrected simulations: sensor-model vs physics-truth acceleration",
        y=0.995,
    )
    fig.tight_layout(rect=(0,0,1,0.965))
    save_fig(fig, outdir / "figures/FIG_S05_representative_acceleration_truth_comparison")

    index_rows.append({
        "figure_id": "FIG_S05",
        "title": "Representative acceleration: sensor model vs physics truth",
        "use": "Internal simulator signal-generation validation across demographic profiles",
        "population": "Predefined age-target representatives from corrected396, common task 21 where available",
        "selection_rule": (
            "Ages targeted before plotting: 20, 30, 53, 63, 74, 78; "
            "nearest corrected396 profile, task 21 preferred; never selected by validation score."
        ),
        "source": "Corrected396 standard CSVs",
        "data_file": "data/representative_trials_predefined.csv",
        "important_note": "This is simulator-internal truth/sensor-model agreement, not physical-vs-synthetic agreement.",
    })

    # B. gyroscope magnitudes, standard vs high-rate truth
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.2, 3.6*nrows), squeeze=False)

    for ax, (_, r) in zip(axes.ravel(), reps.iterrows()):
        p, d, truth = representative_signal_data(r)
        aligned = aligned_gyro(d, truth)

        t = aligned["timestamp"].to_numpy(float)
        gm = magnitude(aligned, ["gyro_x", "gyro_y", "gyro_z"])
        gtm = magnitude(aligned, ["gyro_true_x", "gyro_true_y", "gyro_true_z"])

        ax.plot(t, gm, label="Sensor-model gyroscope")
        ax.plot(t, gtm, label="High-rate physics truth", linewidth=1.0)
        ax.set_title(
            f"{r['profile']} | age {int(r['age'])} | {r['sex']} | task {int(r['task'])}"
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("|ω|")
        ax.grid(alpha=0.2)

    for ax in axes.ravel()[n:]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(
        "Representative corrected simulations: sensor-model vs physics-truth angular velocity",
        y=0.995,
    )
    fig.tight_layout(rect=(0,0,1,0.965))
    save_fig(fig, outdir / "figures/FIG_S06_representative_gyroscope_truth_comparison")

    index_rows.append({
        "figure_id": "FIG_S06",
        "title": "Representative gyroscope: sensor model vs high-rate physics truth",
        "use": "Internal simulator rotational-signal validation across demographic profiles",
        "population": "Same predefined representatives as FIG_S05",
        "selection_rule": "Same fixed representative set as FIG_S05",
        "source": "Corrected396 standard + high-rate truth CSVs",
        "data_file": "data/representative_trials_predefined.csv",
        "important_note": "Timestamp interpolation is used only to align high-rate truth to standard-output timestamps for visualization.",
    })

    # C. event/impact dynamics
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.2, 3.8*nrows), squeeze=False)

    for ax, (_, r) in zip(axes.ravel(), reps.iterrows()):
        p, d, truth = representative_signal_data(r)
        t = pd.to_numeric(d["timestamp"], errors="coerce").to_numpy(float)

        acc_mag = magnitude(d, ["accel_x", "accel_y", "accel_z"])
        impact = pd.to_numeric(d["impact_magnitude"], errors="coerce").to_numpy(float)
        fall = pd.to_numeric(d["fall_detected"], errors="coerce").fillna(0).to_numpy(float)

        # Scale impact only for shared-axis visualization, preserving raw data separately.
        impact_scale = np.nanmax(np.abs(impact))
        acc_scale = np.nanmax(np.abs(acc_mag))
        if np.isfinite(impact_scale) and impact_scale > 0 and np.isfinite(acc_scale):
            impact_plot = impact / impact_scale * acc_scale
        else:
            impact_plot = impact

        ax.plot(t, acc_mag, label="Acceleration magnitude")
        ax.plot(t, impact_plot, label="Impact magnitude (display-scaled)", linewidth=1.0)

        detected_idx = np.where(fall > 0.5)[0]
        if len(detected_idx):
            ax.axvline(t[detected_idx[0]], linestyle="--", linewidth=1.0, label="Fall detected")

        if np.isfinite(impact).any():
            ii = int(np.nanargmax(impact))
            ax.axvline(t[ii], linestyle=":", linewidth=1.0, label="Peak impact")

        ax.set_title(
            f"{r['profile']} | age {int(r['age'])} | {r['sex']} | task {int(r['task'])}"
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Signal magnitude")
        ax.grid(alpha=0.2)

    for ax in axes.ravel()[n:]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    # remove duplicates while preserving order
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    fig.legend(list(seen.values()), list(seen.keys()), loc="upper center", ncol=4)
    fig.suptitle(
        "Representative corrected simulations: fall-detection and impact timing",
        y=0.995,
    )
    fig.tight_layout(rect=(0,0,1,0.955))
    save_fig(fig, outdir / "figures/FIG_S07_representative_event_impact_timing")

    index_rows.append({
        "figure_id": "FIG_S07",
        "title": "Representative fall-detection and impact timing",
        "use": "Visualizes event timing for the same predefined representative cases",
        "population": "Same predefined representatives as FIG_S05",
        "selection_rule": "Same fixed representative set as FIG_S05",
        "source": "Corrected396 standard CSVs",
        "data_file": "data/representative_trials_predefined.csv",
        "important_note": "Impact magnitude is display-scaled within each panel; event times come from original data.",
    })


def generate_impact_aggregate(outdir: Path, index_rows: list):
    p = (
        PUB
        / "rne_corrected396_recovered_impact_audit"
        / "recovered_impact_by_task.csv"
    )
    if not p.exists():
        return

    df = pd.read_csv(p, comment="#", low_memory=False)
    write_csv(df, outdir / "data/recovered_impact_by_task.csv")

    task_candidates = [c for c in df.columns if str(c).lower() in {"task", "task_id"}]
    task_col = task_candidates[0] if task_candidates else None

    numeric = []
    for c in df.columns:
        if c == task_col:
            continue
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().sum() >= max(2, len(df)//2):
            numeric.append(c)

    if task_col is None or not numeric:
        return

    # Prefer metric-like columns and cap to 5.
    preferred_words = ("impact", "time", "duration", "interval", "peak", "median", "mean", "rate", "eligible")
    ranked = sorted(
        numeric,
        key=lambda c: (
            0 if any(w in str(c).lower() for w in preferred_words) else 1,
            str(c),
        ),
    )[:5]

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    x = pd.to_numeric(df[task_col], errors="coerce")
    for c in ranked:
        y = pd.to_numeric(df[c], errors="coerce")
        ax.plot(x, y, marker="o", label=str(c))
    ax.set_xlabel("Task ID")
    ax.set_ylabel("Recovered-impact summary value")
    ax.set_title("Recovered impact/event summary by task")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_fig(fig, outdir / "figures/FIG_S08_recovered_impact_by_task")

    index_rows.append({
        "figure_id": "FIG_S08",
        "title": "Recovered impact/event summary by task",
        "use": "Aggregate event/impact validation",
        "population": "Corrected396 publication-gate recovered-impact audit",
        "selection_rule": "All task rows in canonical recovered_impact_by_task.csv",
        "source": str(p),
        "data_file": "data/recovered_impact_by_task.csv",
        "important_note": "The plotted numeric fields are listed in the data table and figure index; interpret using their source column names.",
    })


def generate_event_policy_summary(outdir: Path, index_rows: list):
    p = (
        PUB
        / "final_synthetic_event_policy_v2"
        / "task_summary_v2.csv"
    )
    if not p.exists():
        return

    df = pd.read_csv(p, comment="#", low_memory=False)
    write_csv(df, outdir / "data/canonical_event_policy_task_summary_v2.csv")

    task_candidates = [c for c in df.columns if str(c).lower() in {"task", "task_id"}]
    task_col = task_candidates[0] if task_candidates else None

    numeric = []
    for c in df.columns:
        if c == task_col:
            continue
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().sum() >= max(2, len(df)//2):
            numeric.append(c)

    if task_col is None or not numeric:
        return

    # Show up to 4 most useful event-window/count columns by keyword.
    words = ("event", "window", "eligible", "count", "duration", "onset", "impact", "interval")
    ranked = sorted(
        numeric,
        key=lambda c: (
            -sum(w in str(c).lower() for w in words),
            str(c),
        ),
    )[:4]

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    x = pd.to_numeric(df[task_col], errors="coerce")
    for c in ranked:
        y = pd.to_numeric(df[c], errors="coerce")
        ax.plot(x, y, marker="o", label=str(c))
    ax.set_xlabel("Task ID")
    ax.set_ylabel("Canonical event-policy summary")
    ax.set_title("Canonical synthetic event-policy summary by task")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_fig(fig, outdir / "figures/FIG_S09_canonical_event_policy_by_task")

    index_rows.append({
        "figure_id": "FIG_S09",
        "title": "Canonical synthetic event-policy summary by task",
        "use": "Documents event-window/onset-impact policy behavior",
        "population": "Canonical event-policy v2 task summary",
        "selection_rule": "All rows in task_summary_v2.csv",
        "source": str(p),
        "data_file": "data/canonical_event_policy_task_summary_v2.csv",
        "important_note": "Use source column names in captions; this figure is generated from the frozen v2 canonical event policy.",
    })


def generate_readme(vr, trials, reps, outdir):
    conf = pd.to_numeric(vr["confidence"], errors="coerce").dropna()

    high = int(vr["classification"].eq("HIGH_CONFIDENCE").sum())
    moderate = int(vr["classification"].eq("MODERATE_CONFIDENCE").sum())

    rates = {}
    for c in CHECKS:
        valid = vr[vr[c].isin(["PASS", "FAIL"])]
        rates[c] = 100.0 * valid[c].eq("PASS").mean() if len(valid) else np.nan

    text = f"""# Final Canonical Publication Figures

This directory contains paper-facing simulator figures generated **only from the frozen
corrected396 campaign and canonical publication-gate outputs**.

No model training, inference, or simulator rerun is performed.

## Corrected396 audit basis

- Corrected simulations: {len(vr)}
- Profiles: {vr['profile'].nunique()}
- Tasks: {vr['task'].nunique()}
- Profile age range: {int(trials['age'].min())}–{int(trials['age'].max())}
- Validation confidence: mean {conf.mean():.2f}%, median {conf.median():.2f}%,
  range {conf.min():.2f}%–{conf.max():.2f}%
- HIGH_CONFIDENCE: {high}
- MODERATE_CONFIDENCE: {moderate}

Validation pass rates:
- head velocity safe: {rates['head_velocity_safe']:.2f}%
- duration realistic: {rates['duration_realistic']:.2f}%
- jerk realistic: {rates['jerk_realistic']:.2f}%
- protective response: {rates['protective_response']:.2f}%
- IMU quality: {rates['imu_quality']:.2f}%
- contact pattern: {rates['contact_pattern']:.2f}%

## Representative-case policy

Representative waveform figures are not selected using confidence, error, correlation,
or visual attractiveness.

The target ages are fixed at:
{", ".join(map(str, REP_TARGET_AGES))}

Task {REP_TASK} is preferred so demographic/profile variation can be shown under a common
fall scenario. If an exact target age does not exist, the nearest corrected396 age is used.

The selected cases and exact source SHA256 hashes are stored in:
`data/representative_trials_predefined.csv`.

## Important interpretation

FIG_S05 and FIG_S06 compare the simulator's sensor-model output against its internal
physics-truth output. They validate signal-processing consistency inside the simulator;
they are **not** real-vs-synthetic waveform comparisons.

The downstream Protechto physical-test segment/event results remain in the existing
paper tables and classifier-figure directories of the evidence archive.

The low jerk-realism pass rate is retained explicitly. Do not suppress it; describe it as
a simulator limitation or as evidence that jerk is a stricter plausibility criterion.
"""
    (outdir / "README.md").write_text(text, encoding="utf-8")


def update_archive_manifest(stage: Path):
    d00 = stage / "00_README_AND_MANIFESTS"
    rows = []
    for p in sorted(stage.rglob("*")):
        if not p.is_file():
            continue
        if p.name == "FILE_MANIFEST_SHA256.csv":
            continue
        rows.append({
            "path": str(p.relative_to(stage)),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })
    pd.DataFrame(rows).to_csv(d00 / "FILE_MANIFEST_SHA256.csv", index=False)


def zip_stage(stage: Path, zip_path: Path):
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as zf:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                arc = Path(stage.name) / p.relative_to(stage)
                zf.write(p, arcname=str(arc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=str(STAGE_DEFAULT))
    ap.add_argument("--zip", dest="zip_path", default=str(ZIP_DEFAULT))
    ap.add_argument(
        "--no-zip",
        action="store_true",
        help="generate figures and manifests but do not rebuild the large final ZIP",
    )
    args = ap.parse_args()

    stage = Path(args.stage)
    zip_path = Path(args.zip_path)
    outdir = stage / OUT_REL

    if not stage.exists():
        raise SystemExit(f"Evidence archive stage does not exist: {stage}")
    if not RUNS.exists():
        raise SystemExit(f"Corrected396 runs do not exist: {RUNS}")

    if outdir.exists():
        shutil.rmtree(outdir)
    (outdir / "figures").mkdir(parents=True)
    (outdir / "data").mkdir(parents=True)

    print("=" * 118)
    print("FINAL CANONICAL PUBLICATION FIGURE GENERATOR")
    print("=" * 118)

    trials = discover_trials()
    vr = discover_validation_reports()

    if len(trials) != 396:
        raise RuntimeError(f"Expected 396 standard corrected trials; found {len(trials)}")
    if len(vr) != 396:
        raise RuntimeError(f"Expected 396 validation reports; found {len(vr)}")

    reps = select_representatives(trials)

    # Preserve target ages explicitly.
    reps = reps.reset_index(drop=True)
    reps.insert(0, "target_age", REP_TARGET_AGES)

    print("Corrected trials        :", len(trials))
    print("Validation reports      :", len(vr))
    print("Profiles                :", trials["profile"].nunique())
    print("Tasks                   :", trials["task"].nunique())
    print("Representative cases    :", len(reps))

    index_rows = []

    generate_validation_summary(vr, outdir, index_rows)
    generate_profile_coverage(trials, outdir, index_rows)
    generate_representative_waveforms(reps, outdir, index_rows)
    generate_impact_aggregate(outdir, index_rows)
    generate_event_policy_summary(outdir, index_rows)

    figure_index = pd.DataFrame(index_rows)
    write_csv(figure_index, outdir / "FIGURE_INDEX.csv")

    # Copy the compact audit into the final figure package if available.
    compact = (
        stage
        / "13_CANONICAL_PAPER_EVIDENCE_V3/"
          "compact_canonical_audit"
    )
    if compact.exists():
        dst = outdir / "compact_audit_snapshot"
        shutil.copytree(compact, dst)

    generate_readme(vr, trials, reps, outdir)

    # Local manifest for this final section.
    rows = []
    for p in sorted(outdir.rglob("*")):
        if p.is_file():
            rows.append({
                "path": str(p.relative_to(stage)),
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            })
    pd.DataFrame(rows).to_csv(
        outdir / "FINAL_FIGURE_PACKAGE_SHA256.csv",
        index=False,
    )

    completion = {
        "corrected_trials": len(trials),
        "validation_reports": len(vr),
        "profiles": int(trials["profile"].nunique()),
        "tasks": int(trials["task"].nunique()),
        "representative_cases": len(reps),
        "figures_indexed": len(figure_index),
        "output_directory": str(outdir),
        "zip_requested": not args.no_zip,
    }

    (outdir / "FINAL_CANONICAL_FIGURES_COMPLETE.json").write_text(
        json.dumps(completion, indent=2),
        encoding="utf-8",
    )

    update_archive_manifest(stage)

    print()
    print("Generated figure IDs:")
    for x in figure_index["figure_id"].tolist():
        print(" ", x)

    if args.no_zip:
        print()
        print("ZIP rebuild skipped by --no-zip")
    else:
        print()
        print("Rebuilding final master ZIP (this can take several minutes)...")
        zip_stage(stage, zip_path)
        print("ZIP:", zip_path)
        print("ZIP bytes:", zip_path.stat().st_size)
        print("ZIP SHA256:", sha256_file(zip_path))

    print()
    print("=" * 118)
    print("FINAL CANONICAL PUBLICATION FIGURE GATE: PASS")
    print("=" * 118)
    print("NO TRAINING, MODEL INFERENCE, OR SIMULATOR RERUN WAS PERFORMED.")


if __name__ == "__main__":
    main()
