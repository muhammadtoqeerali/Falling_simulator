#!/usr/bin/env python3
"""
Generate publication-ready quantitative tables and figures from the final
high-rate fall-simulator validation outputs.

This script does NOT rerun the simulator and does NOT modify any existing
validation outputs. It only reads the frozen publication tables and sensor CSVs.

Default repository root:
    current working directory

Example:
    cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project
    ./.venv/bin/python /path/to/generate_publication_results.py

Outputs:
    outputs/publication_results_v1/
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FS = 100
PRE_N = 50
POST_N = 200
TIME = np.arange(-PRE_N, POST_N + 1) / FS

ACC_FEATURES = [
    "peak_acc_g",
    "p95_acc_g",
    "rms_dynamic_acc_g",
    "min_acc_g",
    "peak_jerk_gps",
    "dynamic_acc_area_gs",
]
GYRO_FEATURES = [
    "peak_gyro_dps",
    "rms_gyro_dps",
    "gyro_area_deg",
]
DURATION_FEATURES = ["fall_duration_s"]

FEATURE_LABELS = {
    "fall_duration_s": "Fall duration",
    "peak_acc_g": "Peak AccMag",
    "p95_acc_g": "95th-pct AccMag",
    "rms_dynamic_acc_g": "RMS dynamic AccMag",
    "min_acc_g": "Minimum AccMag",
    "peak_gyro_dps": "Peak GyroMag",
    "rms_gyro_dps": "RMS GyroMag",
    "peak_jerk_gps": "Peak jerk",
    "gyro_area_deg": "GyroMag area",
    "dynamic_acc_area_gs": "Dynamic AccMag area",
}

FAMILY_MAP = {
    28: "Fainting/collapse",
    29: "Protective response",
    30: "Trip",
    31: "Trip",
    32: "Slip",
    33: "Slip",
    34: "Slip",
    37: "Backward motion",
    38: "Backward motion",
}


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def numeric(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").to_numpy(float)


def read_sensor(path: str | Path):
    """Match validate_simulator_against_real_v2.py channel semantics exactly."""
    p = Path(path)
    df = pd.read_csv(p)
    req = ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise RuntimeError(f"{p}: missing columns {missing}")

    ax = numeric(df, "AccX") / 1000.0
    ay = numeric(df, "AccY") / 1000.0
    az = numeric(df, "AccZ") / 1000.0
    gx = numeric(df, "GyrX") / 1000.0
    gy = numeric(df, "GyrY") / 1000.0
    gz = numeric(df, "GyrZ") / 1000.0

    amag = np.sqrt(ax * ax + ay * ay + az * az)
    gmag = np.sqrt(gx * gx + gy * gy + gz * gz)
    return amag, gmag


def fixed_curve(signal: np.ndarray, onset: int) -> np.ndarray:
    rel = np.arange(-PRE_N, POST_N + 1)
    pos = int(onset) + rel
    out = np.full(len(rel), np.nan, float)
    good = (pos >= 0) & (pos < len(signal))
    out[good] = signal[pos[good]]
    return out


def zcurve(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    idx = np.arange(len(x))
    good = np.isfinite(x)
    if good.sum() < 2:
        return np.full_like(x, np.nan)
    y = np.interp(idx, idx[good], x[good])
    sd = np.std(y)
    if sd < 1e-12:
        return y * 0.0
    return (y - np.mean(y)) / sd


def domain_for_feature(feature: str) -> str:
    if feature in ACC_FEATURES:
        return "Acceleration-related"
    if feature in GYRO_FEATURES:
        return "Gyroscope-related"
    if feature in DURATION_FEATURES:
        return "Event timing"
    return "Other"


def save_table(df: pd.DataFrame, out: Path, name: str):
    p = out / name
    df.to_csv(p, index=False)
    print("[table]", p)


def build_summary_tables(tables: Path, out: Path):
    dist = pd.read_csv(require(tables / "task_distribution_metrics.csv"))
    wave = pd.read_csv(require(tables / "task_median_waveform_similarity.csv"))
    disc_trials = pd.read_csv(require(tables / "matched_vs_wrong_task_trials.csv"))
    disc_summary = pd.read_csv(require(tables / "matched_vs_wrong_task_summary.csv"))
    qc = pd.read_csv(require(tables / "dataset_qc_summary.csv"))

    primary_dist = dist[dist["validation_status"] == "primary"].copy()
    primary_wave = wave[wave["validation_status"] == "primary"].copy()
    primary_dist["domain"] = primary_dist["feature"].map(domain_for_feature)

    # Main validation summary.
    rows = []
    for ds in ["UniVrFall", "KFall"]:
        d = primary_dist[primary_dist["real_dataset"] == ds]
        w = primary_wave[primary_wave["real_dataset"] == ds]
        s = disc_summary[disc_summary["real_dataset"] == ds].iloc[0]
        rows.append({
            "real_dataset": ds,
            "n_primary_tasks": int(w["task"].nunique()),
            "n_sim_trials_discriminability": int(s["n_sim_trials"]),
            "median_nwd": d["normalized_wasserstein_real_iqr"].median(),
            "nwd_q25": d["normalized_wasserstein_real_iqr"].quantile(0.25),
            "nwd_q75": d["normalized_wasserstein_real_iqr"].quantile(0.75),
            "median_coverage_5_95": d["fraction_sim_inside_real_5_95"].median(),
            "mean_coverage_5_95": d["fraction_sim_inside_real_5_95"].mean(),
            "median_acc_corr": w["acc_best_lag_corr"].median(),
            "median_gyro_corr": w["gyro_best_lag_corr"].median(),
            "median_acc_dtw": w["acc_dtw"].median(),
            "median_gyro_dtw": w["gyro_dtw"].median(),
            "top1_accuracy": s["top1_accuracy"],
            "top3_accuracy": s["top3_accuracy"],
            "median_true_task_rank": s["median_true_task_rank"],
            "mean_true_task_rank": s["mean_true_task_rank"],
        })
    main = pd.DataFrame(rows)
    save_table(main, out, "table_main_validation_summary.csv")

    # Feature-domain summary (particularly useful for acceleration vs gyro story).
    domain = (
        primary_dist.groupby(["real_dataset", "domain"])
        .agg(
            n_comparisons=("feature", "size"),
            n_tasks=("task", "nunique"),
            median_nwd=("normalized_wasserstein_real_iqr", "median"),
            q25_nwd=("normalized_wasserstein_real_iqr", lambda x: x.quantile(0.25)),
            q75_nwd=("normalized_wasserstein_real_iqr", lambda x: x.quantile(0.75)),
            median_coverage=("fraction_sim_inside_real_5_95", "median"),
            mean_coverage=("fraction_sim_inside_real_5_95", "mean"),
            median_abs_cliffs_delta=("cliffs_delta_sim_vs_real", lambda x: np.median(np.abs(x))),
        )
        .reset_index()
    )
    save_table(domain, out, "table_feature_domain_summary.csv")

    # Per-feature summary across primary tasks.
    feat = (
        primary_dist.groupby(["real_dataset", "feature"])
        .agg(
            n_tasks=("task", "nunique"),
            median_nwd=("normalized_wasserstein_real_iqr", "median"),
            q25_nwd=("normalized_wasserstein_real_iqr", lambda x: x.quantile(0.25)),
            q75_nwd=("normalized_wasserstein_real_iqr", lambda x: x.quantile(0.75)),
            median_coverage=("fraction_sim_inside_real_5_95", "median"),
            mean_coverage=("fraction_sim_inside_real_5_95", "mean"),
            median_abs_cliffs_delta=("cliffs_delta_sim_vs_real", lambda x: np.median(np.abs(x))),
            median_sim_over_real=("median_ratio_sim_over_real", "median"),
        )
        .reset_index()
    )
    feat["feature_label"] = feat["feature"].map(FEATURE_LABELS)
    feat["domain"] = feat["feature"].map(domain_for_feature)
    save_table(feat, out, "table_feature_level_fidelity.csv")

    # Task-level distribution summary.
    task_rows = []
    for (ds, task), g in primary_dist.groupby(["real_dataset", "task"]):
        ga = g[g["feature"].isin(ACC_FEATURES)]
        gg = g[g["feature"].isin(GYRO_FEATURES)]
        gt = g[g["feature"].isin(DURATION_FEATURES)]
        task_rows.append({
            "real_dataset": ds,
            "task": int(task),
            "family": FAMILY_MAP.get(int(task), "Other"),
            "overall_median_nwd": g["normalized_wasserstein_real_iqr"].median(),
            "overall_median_coverage": g["fraction_sim_inside_real_5_95"].median(),
            "acc_median_nwd": ga["normalized_wasserstein_real_iqr"].median(),
            "acc_median_coverage": ga["fraction_sim_inside_real_5_95"].median(),
            "gyro_median_nwd": gg["normalized_wasserstein_real_iqr"].median(),
            "gyro_median_coverage": gg["fraction_sim_inside_real_5_95"].median(),
            "timing_nwd": gt["normalized_wasserstein_real_iqr"].median() if len(gt) else np.nan,
        })
    task_dist = pd.DataFrame(task_rows)

    # Merge waveform and task-discriminability metrics.
    task_disc = (
        disc_trials.groupby(["real_dataset", "true_task"])
        .agg(
            n_discriminability_trials=("true_task_rank", "size"),
            top1_accuracy=("top1_correct", "mean"),
            top3_accuracy=("top3_correct", "mean"),
            median_true_task_rank=("true_task_rank", "median"),
            mean_true_task_rank=("true_task_rank", "mean"),
        )
        .reset_index()
        .rename(columns={"true_task": "task"})
    )

    task = task_dist.merge(
        primary_wave.drop(columns=["validation_status"]),
        on=["real_dataset", "task"], how="left"
    ).merge(task_disc, on=["real_dataset", "task"], how="left")
    save_table(task, out, "table_task_level_validation.csv")

    # Family summary.
    family = (
        task.groupby(["real_dataset", "family"])
        .agg(
            n_tasks=("task", "nunique"),
            median_overall_nwd=("overall_median_nwd", "median"),
            median_acc_nwd=("acc_median_nwd", "median"),
            median_gyro_nwd=("gyro_median_nwd", "median"),
            median_acc_corr=("acc_best_lag_corr", "median"),
            median_gyro_corr=("gyro_best_lag_corr", "median"),
            median_acc_dtw=("acc_dtw", "median"),
            median_gyro_dtw=("gyro_dtw", "median"),
            median_task_rank=("median_true_task_rank", "median"),
        )
        .reset_index()
    )
    save_table(family, out, "table_family_level_validation.csv")

    save_table(qc, out, "table_dataset_qc.csv")
    save_table(task_disc, out, "table_task_discriminability.csv")

    return dist, wave, task, feat, domain


def build_acquisition_table(root: Path, out: Path):
    p = root / "outputs/validation_v2/truth_sampling_rich221_ab/truth_sampling_feature_summary.csv"
    if not p.exists():
        print("[skip] acquisition table: missing", p)
        return None
    df = pd.read_csv(p).copy()
    # Publication-safe labels: this is an acquisition-method sensitivity check,
    # not an old-vs-new simulator comparison.
    df = df.rename(columns={
        "legacy30_median": "control_step_acquisition_median",
        "highrate450_median": "physics_substep_acquisition_median",
    })
    df["feature_label"] = df["feature"].map(FEATURE_LABELS)
    save_table(df, out, "table_acquisition_sensitivity.csv")
    return df


def plot_feature_nwd(feat: pd.DataFrame, out: Path):
    pivot = feat.pivot(index="feature", columns="real_dataset", values="median_nwd")
    ordered = [f for f in ACC_FEATURES + GYRO_FEATURES + DURATION_FEATURES if f in pivot.index]
    pivot = pivot.loc[ordered]

    fig, ax = plt.subplots(figsize=(7.3, 5.4))
    y = np.arange(len(pivot))
    for ds in [c for c in ["UniVrFall", "KFall"] if c in pivot.columns]:
        ax.plot(pivot[ds].to_numpy(), y, marker="o", linewidth=1.2, label=ds)
    ax.set_yticks(y)
    ax.set_yticklabels([FEATURE_LABELS.get(f, f) for f in pivot.index])
    ax.invert_yaxis()
    ax.set_xlabel("Median normalized Wasserstein distance (lower is better)")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    p = out / "fig_feature_fidelity_nwd.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[figure]", p)


def annotated_metric_table(task: pd.DataFrame, dataset: str, out: Path):
    d = task[task["real_dataset"] == dataset].sort_values("task").copy()
    cols = [
        "overall_median_nwd",
        "acc_best_lag_corr",
        "gyro_best_lag_corr",
        "acc_dtw",
        "gyro_dtw",
    ]
    labels = ["NWD", "Acc corr", "Gyro corr", "Acc DTW", "Gyro DTW"]
    M = d[cols].to_numpy(float)

    # Column-wise min-max only for background visualization; the annotated
    # numbers remain the actual metrics. Do not interpret the background as a
    # composite score because metric direction differs by column.
    Z = np.zeros_like(M)
    for j in range(M.shape[1]):
        col = M[:, j]
        lo, hi = np.nanmin(col), np.nanmax(col)
        Z[:, j] = 0.5 if hi - lo < 1e-12 else (col - lo) / (hi - lo)

    fig_h = max(3.1, 0.42 * len(d) + 1.6)
    fig, ax = plt.subplots(figsize=(7.0, fig_h))
    im = ax.imshow(Z, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels([f"Task {int(t)}" for t in d["task"]])
    ax.set_title(f"{dataset}: primary task-level validation metrics")

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M[i, j]
            txt = "--" if not np.isfinite(val) else f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)

    ax.set_xlabel("Annotated values are raw metrics; column shading is within-column only")
    fig.tight_layout()
    p = out / f"fig_task_metric_matrix_{dataset.lower()}.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[figure]", p)


def plot_acquisition_sensitivity(acq: pd.DataFrame | None, out: Path):
    if acq is None:
        return
    d = acq.copy()
    ordered = [f for f in ACC_FEATURES + GYRO_FEATURES if f in set(d["feature"])]
    d = d.set_index("feature").loc[ordered].reset_index()
    vals = 100.0 * d["median_relative_delta"].to_numpy(float)
    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.barh(y, vals)
    ax.axvline(0, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([FEATURE_LABELS.get(f, f) for f in d["feature"]])
    ax.invert_yaxis()
    ax.set_xlabel("Median relative change with physics-substep acquisition (%)")
    ax.set_title("Acquisition-method sensitivity across 221 matched trajectories")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    p = out / "fig_acquisition_sensitivity.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[figure]", p)


def collect_curves(features: pd.DataFrame, dataset: str, task: int):
    g = features[(features["dataset"] == dataset) & (features["canonical_task_id"] == task)]
    acc, gyr = [], []
    failures = []
    for r in g.itertuples(index=False):
        try:
            amag, gmag = read_sensor(r.sensor_csv)
            onset = int(r.fall_onset_frame)
            acc.append(fixed_curve(amag, onset))
            gyr.append(fixed_curve(gmag, onset))
        except Exception as e:
            failures.append((str(r.sensor_csv), repr(e)))
    return np.asarray(acc, float), np.asarray(gyr, float), failures


def median_iqr(curves: np.ndarray):
    if curves.size == 0:
        return None
    return (
        np.nanmedian(curves, axis=0),
        np.nanpercentile(curves, 25, axis=0),
        np.nanpercentile(curves, 75, axis=0),
    )


def plot_waveforms(root: Path, out: Path, selected_tasks=(29, 32)):
    feat_path = root / "outputs/validation_v2/results_highrate_truth_rich221_publication/tables/features_all_trials_v2.csv"
    if not feat_path.exists():
        print("[skip] waveform figures: missing", feat_path)
        return
    features = pd.read_csv(feat_path)

    failures_all = []
    curve_rows = []

    for real_dataset in ["UniVrFall", "KFall"]:
        available = sorted(set(features.loc[features["dataset"] == real_dataset, "canonical_task_id"].astype(int)))
        tasks = [t for t in selected_tasks if t in available]
        if not tasks:
            continue

        fig, axes = plt.subplots(len(tasks), 2, figsize=(8.0, 3.0 * len(tasks)), sharex=True)
        if len(tasks) == 1:
            axes = np.asarray([axes])

        for i, task in enumerate(tasks):
            r_acc, r_gyr, f1 = collect_curves(features, real_dataset, task)
            s_acc, s_gyr, f2 = collect_curves(features, "Simulated", task)
            failures_all.extend(f1 + f2)

            for j, (real_curves, sim_curves, ylabel) in enumerate([
                (r_acc, s_acc, "AccMag (g)"),
                (r_gyr, s_gyr, "GyroMag (deg/s)"),
            ]):
                ax = axes[i, j]
                rs = median_iqr(real_curves)
                ss = median_iqr(sim_curves)
                if rs is None or ss is None:
                    ax.text(0.5, 0.5, "Missing curves", ha="center", va="center", transform=ax.transAxes)
                    continue

                rm, rq1, rq3 = rs
                sm, sq1, sq3 = ss
                lr, = ax.plot(TIME, rm, linewidth=1.5, label=f"{real_dataset} median")
                ax.fill_between(TIME, rq1, rq3, alpha=0.18, color=lr.get_color(), label=f"{real_dataset} IQR")
                ls, = ax.plot(TIME, sm, linewidth=1.5, label="Synthetic median")
                ax.fill_between(TIME, sq1, sq3, alpha=0.18, color=ls.get_color(), label="Synthetic IQR")
                ax.axvline(0, linestyle="--", linewidth=0.8)
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.20)
                if i == 0:
                    ax.set_title("Acceleration magnitude" if j == 0 else "Angular-velocity magnitude")
                if j == 0:
                    ax.text(0.02, 0.95, f"Task {task}", transform=ax.transAxes, va="top", fontweight="bold")
                if i == len(tasks) - 1:
                    ax.set_xlabel("Time from fall onset (s)")

                # Export curve summaries for exact reproducibility.
                for k, t in enumerate(TIME):
                    curve_rows.append({
                        "real_dataset": real_dataset,
                        "task": task,
                        "modality": "AccMag" if j == 0 else "GyroMag",
                        "time_s": t,
                        "real_median": rm[k],
                        "real_q25": rq1[k],
                        "real_q75": rq3[k],
                        "sim_median": sm[k],
                        "sim_q25": sq1[k],
                        "sim_q75": sq3[k],
                    })

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
        fig.suptitle(f"{real_dataset}: real vs synthetic onset-centered waveforms", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        p = out / f"fig_waveform_real_vs_synthetic_{real_dataset.lower()}_tasks_{'_'.join(map(str,tasks))}.png"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("[figure]", p)

    if curve_rows:
        save_table(pd.DataFrame(curve_rows), out, "table_waveform_curve_summaries.csv")
    if failures_all:
        pd.DataFrame(failures_all, columns=["sensor_csv", "error"]).to_csv(out / "waveform_load_failures.csv", index=False)
        print("[warning] waveform load failures written to", out / "waveform_load_failures.csv")


def make_scenario_panel(root: Path, out: Path):
    """Combine representative final-campaign sagittal snapshot images if available."""
    selected_root = root / "outputs/paper_results_inventory/simulator_images_selected"
    if not selected_root.exists():
        print("[skip] scenario panel: missing", selected_root)
        return

    tasks = [20, 28, 30, 33, 39, 41]
    titles = {
        20: "Sit-down transition",
        28: "Walking faint/collapse",
        30: "Walking trip",
        33: "Lateral slip",
        39: "Elevated forward fall",
        41: "Ladder/elevation scenario",
    }
    paths = []
    for task in tasks:
        folder_matches = sorted(selected_root.glob(f"task_{task}_age_*"))
        if not folder_matches:
            continue
        matches = sorted(folder_matches[0].glob("*_sagittal_snapshots.png"))
        if matches:
            paths.append((task, matches[0]))

    if len(paths) < 3:
        print("[skip] scenario panel: fewer than 3 representative images found")
        return

    ncols = 2
    nrows = math.ceil(len(paths) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.0, 2.7 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (task, pth) in zip(axes, paths):
        img = plt.imread(pth)
        ax.imshow(img)
        ax.set_title(f"Task {task}: {titles.get(task, '')}", fontsize=9)
        ax.axis("off")
    for ax in axes[len(paths):]:
        ax.axis("off")
    fig.tight_layout()
    p = out / "fig_representative_scenario_kinematics.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[figure]", p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Simulator repository root")
    parser.add_argument("--out", type=Path, default=None, help="Output directory")
    parser.add_argument("--waveform-tasks", nargs="*", type=int, default=[29, 32], help="Representative direct-overlap tasks")
    args = parser.parse_args()

    root = args.root.resolve()
    out = args.out.resolve() if args.out else root / "outputs/publication_results_v1"
    out.mkdir(parents=True, exist_ok=True)

    tables = root / "outputs/validation_v2/results_highrate_truth_rich221_publication/tables"
    require(tables)

    print("Repository root:", root)
    print("Output directory:", out)

    dist, wave, task, feat, domain = build_summary_tables(tables, out)
    acq = build_acquisition_table(root, out)

    plot_feature_nwd(feat, out)
    annotated_metric_table(task, "UniVrFall", out)
    annotated_metric_table(task, "KFall", out)
    plot_acquisition_sensitivity(acq, out)
    plot_waveforms(root, out, tuple(args.waveform_tasks))
    make_scenario_panel(root, out)

    print("\nDONE")
    print("Please inspect:", out)
    print("Then zip it with:")
    print(f"  cd {root}")
    try:
        rel_out = out.relative_to(root)
    except ValueError:
        rel_out = out
    print(f"  zip -r publication_results_v1.zip {rel_out}")


if __name__ == "__main__":
    main()
