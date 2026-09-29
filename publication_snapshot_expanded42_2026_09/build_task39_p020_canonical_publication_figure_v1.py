#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PROJECT = Path("/mnt/hdd16T/ToqeerHomeBackup/mujoco_project")
RUN = PROJECT / (
    "outputs/_highrate_overnight/campaign_highrate_truth_v3_rne_corrected396/"
    "runs/P020/task_39/scenario39_age49_h1p63_sex_female_w73p0_20260826_144048"
)

MAIN = RUN / "fall_scenario39_age49_20260826_144048.csv"
HR = RUN / "fall_scenario39_age49_20260826_144048_highrate_truth.csv"

OUT = PROJECT / "outputs/task39_p020_canonical_publication_figure_v1"

PHASES = [
    ("SETUP",         4.633333333333462),
    ("FALL ONSET",    5.633333333333462),
    ("MAX DESCENT",   6.0133),
    ("FIRST CONTACT", 6.1433),
    ("PEAK IMPACT",   6.166666666666878),
    ("POST-IMPACT",   6.416666666666878),
    ("REST",          7.4833),
]

PHASE_SHORT = ["Setup", "Onset", "Max\ndescent", "First\ncontact", "Peak\nimpact", "Post-\nimpact", "Rest"]

def read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, comment="#", low_memory=False)

def nearest_row(df: pd.DataFrame, t: float) -> pd.Series:
    idx = (df["timestamp"] - t).abs().idxmin()
    return df.loc[idx]

def mag(df: pd.DataFrame, cols) -> np.ndarray:
    return np.sqrt(np.sum(np.column_stack([df[c].to_numpy(float) for c in cols])**2, axis=1))

def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)
    ax.grid(True, axis="y", linewidth=0.35, alpha=0.35)

def add_phase_lines(ax, ymin=None, ymax=None, labels=False):
    phase_colors = plt.cm.tab10(np.linspace(0, 0.8, len(PHASES)))
    for i, ((name, t), c) in enumerate(zip(PHASES, phase_colors), start=1):
        ax.axvline(t, color=c, linewidth=0.85, alpha=0.8, zorder=0)
        if labels:
            lo, hi = ax.get_ylim() if ymin is None or ymax is None else (ymin, ymax)
            ax.text(t, hi, str(i), ha="center", va="bottom", fontsize=6.8,
                    color=c, fontweight="bold")

def interpolate_hr_to_main(main: pd.DataFrame, hr: pd.DataFrame, col: str) -> np.ndarray:
    return np.interp(
        main["timestamp"].to_numpy(float),
        hr["timestamp"].to_numpy(float),
        hr[col].to_numpy(float),
    )

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    if not MAIN.exists():
        raise SystemExit(f"ERROR: canonical main CSV missing: {MAIN}")
    if not HR.exists():
        raise SystemExit(f"ERROR: canonical high-rate truth CSV missing: {HR}")

    df = read_csv(MAIN)
    hr = read_csv(HR)

    required_main = {
        "timestamp", "sensor_pos_x", "sensor_pos_y", "sensor_pos_z",
        "sensor_vel_x", "sensor_vel_y", "sensor_vel_z",
        "pelvis_height", "impact_force", "impact_magnitude", "jerk_mag"
    }
    missing = sorted(required_main - set(df.columns))
    if missing:
        raise SystemExit(f"ERROR: canonical main CSV missing columns: {missing}")

    # Prefer physics-truth high-rate acceleration/gyro where available.
    for c in ["accel_true_x", "accel_true_y", "accel_true_z",
              "gyro_true_x", "gyro_true_y", "gyro_true_z"]:
        if c not in hr.columns:
            raise SystemExit(f"ERROR: canonical high-rate truth missing: {c}")

    # Exact derived magnitudes.
    sensor_speed = mag(df, ["sensor_vel_x", "sensor_vel_y", "sensor_vel_z"])
    accel_mag_hr = mag(hr, ["accel_true_x", "accel_true_y", "accel_true_z"])
    gyro_mag_hr = mag(hr, ["gyro_true_x", "gyro_true_y", "gyro_true_z"])

    # Locked display window around the canonical event.
    t0 = PHASES[0][1] - 0.35
    t1 = PHASES[-1][1] + 0.45

    m_main = (df["timestamp"] >= t0) & (df["timestamp"] <= t1)
    m_hr = (hr["timestamp"] >= t0) & (hr["timestamp"] <= t1)

    # Choose the horizontal world coordinate with the larger event excursion.
    x = df.loc[m_main, "sensor_pos_x"].to_numpy(float)
    y = df.loc[m_main, "sensor_pos_y"].to_numpy(float)
    xr = float(np.nanmax(x)-np.nanmin(x)) if len(x) else 0
    yr = float(np.nanmax(y)-np.nanmin(y)) if len(y) else 0
    horiz_col = "sensor_pos_y" if yr >= xr else "sensor_pos_x"
    horiz_label = "Sensor world y [m]" if horiz_col.endswith("_y") else "Sensor world x [m]"

    # Phase summary from exact nearest canonical samples.
    phase_rows = []
    setup = nearest_row(df, PHASES[0][1])
    setup_pelvis = float(setup["pelvis_height"])
    for i, (name, t) in enumerate(PHASES, start=1):
        r = nearest_row(df, t)
        hrr = nearest_row(hr, t)
        phase_rows.append({
            "phase_index": i,
            "phase": name,
            "target_time_s": t,
            "sample_time_main_s": float(r["timestamp"]),
            "sample_time_highrate_s": float(hrr["timestamp"]),
            "pelvis_height_m": float(r["pelvis_height"]),
            "pelvis_drop_from_setup_m": setup_pelvis - float(r["pelvis_height"]),
            "sensor_x_m": float(r["sensor_pos_x"]),
            "sensor_y_m": float(r["sensor_pos_y"]),
            "sensor_z_m": float(r["sensor_pos_z"]),
            "sensor_speed_mps": float(np.sqrt(r["sensor_vel_x"]**2+r["sensor_vel_y"]**2+r["sensor_vel_z"]**2)),
            "impact_force_N": float(r["impact_force"]),
            "impact_magnitude_source_native": float(r["impact_magnitude"]),
            "jerk_mag": float(r["jerk_mag"]),
            "accel_true_mag_mps2": float(np.sqrt(hrr["accel_true_x"]**2+hrr["accel_true_y"]**2+hrr["accel_true_z"]**2)),
            "gyro_true_mag_rps": float(np.sqrt(hrr["gyro_true_x"]**2+hrr["gyro_true_y"]**2+hrr["gyro_true_z"]**2)),
        })
    ps = pd.DataFrame(phase_rows)
    ps.to_csv(OUT / "task39_p020_phase_summary.csv", index=False)

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8.0,
        "axes.labelsize": 8.2,
        "axes.titlesize": 9.0,
        "legend.fontsize": 6.8,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(7.15, 8.35), constrained_layout=False)
    gs = GridSpec(
        4, 2, figure=fig,
        height_ratios=[1.15, 1.0, 1.0, 0.92],
        hspace=0.58, wspace=0.38
    )

    # (a) Exact sensor trajectory with seven synchronized phase points.
    ax = fig.add_subplot(gs[0, :])
    sl = df.loc[m_main]
    ax.plot(sl[horiz_col], sl["sensor_pos_z"], linewidth=1.35, label="Canonical sensor trajectory")
    phase_colors = plt.cm.tab10(np.linspace(0, 0.8, len(PHASES)))
    for i, (row, c) in enumerate(zip(phase_rows, phase_colors), start=1):
        xx = row["sensor_y_m"] if horiz_col.endswith("_y") else row["sensor_x_m"]
        zz = row["sensor_z_m"]
        ax.scatter([xx], [zz], s=34, color=c, edgecolor="white", linewidth=0.6, zorder=4)
        ax.text(xx, zz, str(i), ha="center", va="center", fontsize=6.4,
                color="white", fontweight="bold", zorder=5)
    ax.set_xlabel(horiz_label)
    ax.set_ylabel("Sensor world z [m]")
    clean_axes(ax)
    ax.set_title("(a) Canonical lower-back sensor trajectory and locked Task-39 phases", loc="left", fontweight="bold")
    # Phase key under trajectory.
    key = "   ".join(f"{i} {s.replace(chr(10),' ')}" for i, s in enumerate(PHASE_SHORT, 1))
    ax.text(0.5, -0.28, key, transform=ax.transAxes, ha="center", va="top", fontsize=6.5)

    # (b) Pelvis height + sensor speed
    ax = fig.add_subplot(gs[1, 0])
    t = df["timestamp"].to_numpy(float)
    ax.plot(t[m_main], df.loc[m_main, "pelvis_height"], linewidth=1.3, label="Pelvis height [m]")
    ax2 = ax.twinx()
    ax2.plot(t[m_main], sensor_speed[m_main.to_numpy()], linewidth=1.0, linestyle="--", label="Sensor speed [m/s]")
    ax.set_xlim(t0, t1)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Pelvis height [m]")
    ax2.set_ylabel("Sensor speed [m/s]")
    add_phase_lines(ax, labels=True)
    clean_axes(ax)
    ax2.spines["top"].set_visible(False)
    ax.set_title("(b) Descent and post-impact settling", loc="left", fontweight="bold")
    lines = ax.get_lines()[:1] + ax2.get_lines()[:1]
    ax.legend(lines, [l.get_label() for l in lines], frameon=False, loc="best")

    # (c) Impact/contact proxy and jerk
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t[m_main], df.loc[m_main, "impact_force"], linewidth=1.25, label="Impact force [N]")
    ax2 = ax.twinx()
    ax2.plot(t[m_main], df.loc[m_main, "jerk_mag"], linewidth=1.0, linestyle="--", label="Jerk magnitude")
    ax.set_xlim(t0, t1)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Impact force [N]")
    ax2.set_ylabel("Jerk magnitude")
    add_phase_lines(ax, labels=True)
    clean_axes(ax)
    ax2.spines["top"].set_visible(False)
    ax.set_title("(c) Ground-interaction / impact dynamics", loc="left", fontweight="bold")
    lines = ax.get_lines()[:1] + ax2.get_lines()[:1]
    ax.legend(lines, [l.get_label() for l in lines], frameon=False, loc="best")

    # (d) Physics-truth acceleration
    ax = fig.add_subplot(gs[2, 0])
    th = hr["timestamp"].to_numpy(float)
    for c, lab in [("accel_true_x","a$_x$"),("accel_true_y","a$_y$"),("accel_true_z","a$_z$")]:
        ax.plot(th[m_hr], hr.loc[m_hr, c], linewidth=0.85, label=lab)
    ax.plot(th[m_hr], accel_mag_hr[m_hr.to_numpy()], linewidth=1.35, label="|a|")
    ax.set_xlim(t0, t1)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Acceleration [m/s²]")
    add_phase_lines(ax, labels=True)
    clean_axes(ax)
    ax.set_title("(d) Corrected physics-truth lower-back acceleration", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=4, loc="best")

    # (e) Physics-truth angular velocity
    ax = fig.add_subplot(gs[2, 1])
    for c, lab in [("gyro_true_x","ω$_x$"),("gyro_true_y","ω$_y$"),("gyro_true_z","ω$_z$")]:
        ax.plot(th[m_hr], hr.loc[m_hr, c], linewidth=0.85, label=lab)
    ax.plot(th[m_hr], gyro_mag_hr[m_hr.to_numpy()], linewidth=1.35, label="|ω|")
    ax.set_xlim(t0, t1)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angular velocity [rad/s]")
    add_phase_lines(ax, labels=True)
    clean_axes(ax)
    ax.set_title("(e) Corrected physics-truth lower-back angular velocity", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=4, loc="best")

    # (f) Phase-state matrix; normalized by row, exact values annotated.
    ax = fig.add_subplot(gs[3, :])
    metrics = [
        ("Pelvis drop", ps["pelvis_drop_from_setup_m"].to_numpy(), "m", 3),
        ("Sensor speed", ps["sensor_speed_mps"].to_numpy(), "m/s", 2),
        ("Impact force", ps["impact_force_N"].to_numpy(), "N", 0),
        ("Accel |a|", ps["accel_true_mag_mps2"].to_numpy(), "m/s²", 1),
        ("Gyro |ω|", ps["gyro_true_mag_rps"].to_numpy(), "rad/s", 2),
        ("Jerk", ps["jerk_mag"].to_numpy(), "", 0),
    ]
    raw = np.vstack([v for _, v, _, _ in metrics]).astype(float)
    norm = np.zeros_like(raw)
    for i in range(raw.shape[0]):
        mn, mx = np.nanmin(raw[i]), np.nanmax(raw[i])
        norm[i] = 0 if mx == mn else (raw[i]-mn)/(mx-mn)
    ax.imshow(norm, aspect="auto", cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(range(len(PHASE_SHORT)), PHASE_SHORT)
    ax.set_yticks(range(len(metrics)), [m[0] for m in metrics])
    for i, (_, vals, unit, nd) in enumerate(metrics):
        for j, val in enumerate(vals):
            txtv = f"{val:.{nd}f}" + (f" {unit}" if unit else "")
            ax.text(j, i, txtv, ha="center", va="center",
                    fontsize=6.2, color=("white" if norm[i,j] > 0.58 else "black"))
    ax.set_title("(f) Exact phase-state summary (row shading normalized only for visualization)", loc="left", fontweight="bold")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    fig.subplots_adjust(left=0.10, right=0.92, top=0.975, bottom=0.075)

    base = OUT / "fig_task39_p020_canonical_multiphysics_v1"
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    caption = (
        "Canonical Task-39/P020 forward-fall-from-height evidence from the final corrected396 campaign. "
        "(a) Lower-back virtual-sensor world trajectory with seven locked phases: Setup, Fall onset, "
        "Maximum descent, First contact, Peak impact, Post-impact, and Rest. "
        "(b) Pelvis height and sensor speed. (c) Impact force and jerk. "
        "(d,e) Corrected high-rate physics-truth lower-back acceleration and angular velocity. "
        "(f) Exact phase-state values; grayscale shading is normalized independently within each metric row. "
        "All curves and phase samples are derived from the canonical corrected396 P020 Task-39 files. "
        "No replay-derived whole-body pose, skeleton, contact identity, or render is shown because the "
        "historical-source replay failed the strict trajectory-equivalence gate."
    )
    (OUT / "FIGURE_CAPTION.txt").write_text(caption + "\n", encoding="utf-8")

    audit = {
        "canonical_main": str(MAIN),
        "canonical_highrate_truth": str(HR),
        "phase_times_s": {k: v for k, v in PHASES},
        "horizontal_trajectory_coordinate": horiz_col,
        "replay_pose_used": False,
        "replay_skeleton_used": False,
        "replay_contact_identity_used": False,
        "figure_status": "PAPER_SAFE_CANONICAL_ONLY",
        "note": (
            "V8 historical-source replay failed strict equivalence; therefore the figure intentionally "
            "uses only final corrected396 canonical signal evidence."
        ),
    }
    (OUT / "FIGURE_PROVENANCE.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    readme = f"""TASK39 P020 CANONICAL PUBLICATION FIGURE V1
============================================

STATUS
------
PAPER_SAFE_CANONICAL_ONLY

Generated from:
  {MAIN}
  {HR}

Outputs:
  {base.with_suffix('.png')}
  {base.with_suffix('.pdf')}
  {base.with_suffix('.svg')}
  {OUT / 'task39_p020_phase_summary.csv'}
  {OUT / 'FIGURE_CAPTION.txt'}
  {OUT / 'FIGURE_PROVENANCE.json'}

The figure uses NO failed replay pose/skeleton/contact-identity/render products.
"""
    (OUT / "README.txt").write_text(readme, encoding="utf-8")

    print("="*92)
    print("TASK39 P020 CANONICAL PUBLICATION FIGURE V1: COMPLETE")
    print("="*92)
    print("Status: PAPER_SAFE_CANONICAL_ONLY")
    print("Replay pose/skeleton used: NO")
    print("Canonical main:", MAIN)
    print("Canonical high-rate truth:", HR)
    print("PNG:", base.with_suffix(".png"))
    print("PDF:", base.with_suffix(".pdf"))
    print("SVG:", base.with_suffix(".svg"))
    print("Phase summary:", OUT / "task39_p020_phase_summary.csv")
    print("Caption:", OUT / "FIGURE_CAPTION.txt")
    print("="*92)

if __name__ == "__main__":
    main()
