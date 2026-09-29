from pathlib import Path
import argparse
import json
import math
import re
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

REAL_ORDER = [20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,37,38,39,40,41,42]

SIGNALS = [
    "AccX_g", "AccY_g", "AccZ_g",
    "GyrX_dps", "GyrY_dps", "GyrZ_dps",
    "AccMag_g", "GyrMag_dps",
]

def find_lab_root(root):
    root = Path(root)

    if (root / "laboratory" / "sensors_data").exists():
        return root / "laboratory"

    if (root / "sensors_data").exists():
        return root

    matches = list(root.rglob("sensors_data"))
    for m in matches:
        lab = m.parent
        if (lab / "labels_data").exists():
            return lab

    raise FileNotFoundError(f"Could not find sensors_data/labels_data inside {root}")

def task_from_code(x):
    s = str(x)

    m = re.search(r"\((\d+)\)", s)
    if m:
        return int(m.group(1))

    m = re.search(r"F\s*0*(\d+)", s, re.IGNORECASE)
    if m:
        fnum = int(m.group(1))
        if 1 <= fnum <= len(REAL_ORDER):
            return REAL_ORDER[fnum - 1]
        return fnum

    nums = re.findall(r"\d+", s)
    return int(nums[-1]) if nums else None

def fnum_from_code(x):
    s = str(x)
    m = re.search(r"F\s*0*(\d+)", s, re.IGNORECASE)
    return int(m.group(1)) if m else None

def col_like(df, options):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for opt in options:
        key = str(opt).strip().lower()
        if key in lookup:
            return lookup[key]
    return None

def find_sensor_file(sensors_root, subject_id, task_id, trial_id, fnum=None):
    subject_folder = sensors_root / f"SA{subject_id:02d}"
    search_roots = [subject_folder] if subject_folder.exists() else [sensors_root]

    candidates = []
    task_candidates = [task_id]
    if fnum is not None and fnum not in task_candidates:
        task_candidates.append(fnum)

    for root in search_roots:
        for t in task_candidates:
            patterns = [
                f"S{subject_id:02d}T{t:02d}R{trial_id:02d}.csv",
                f"S{subject_id:02d}T{t}R{trial_id:02d}.csv",
                f"*T{t:02d}R{trial_id:02d}.csv",
                f"*T{t}R{trial_id:02d}.csv",
            ]
            for pat in patterns:
                candidates.extend(root.rglob(pat))

    if candidates:
        return sorted(candidates)[0]

    return None

def load_dataset(root, name):
    lab = find_lab_root(root)
    sensors_root = lab / "sensors_data"
    labels_root = lab / "labels_data"

    records = []

    for label_file in sorted(labels_root.glob("*.xlsx")):
        m = re.search(r"SA(\d+)", label_file.name, re.IGNORECASE)
        subject_id = int(m.group(1)) if m else None

        labels = pd.read_excel(label_file)

        task_col = col_like(labels, ["Task Code (Task ID)", "Task Code", "Task", "Task ID"])
        desc_col = col_like(labels, ["Description", "Task Description"])
        trial_col = col_like(labels, ["Trial ID", "Trial", "Repetition"])
        onset_col = col_like(labels, ["Fall_onset_frame", "Fall Onset Frame", "Fall_onset"])
        impact_col = col_like(labels, ["Fall_impact_frame", "Fall Impact Frame", "Fall_impact"])

        if task_col is None or trial_col is None or onset_col is None or impact_col is None:
            raise ValueError(f"Label file has unexpected columns: {label_file} -> {labels.columns.tolist()}")

        if subject_id is None:
            first_sensor_folder = sorted(sensors_root.glob("SA*"))[0]
            subject_id = int(re.search(r"SA(\d+)", first_sensor_folder.name).group(1))

        for _, row in labels.iterrows():
            task_code = row[task_col]
            task_id = task_from_code(task_code)
            fnum = fnum_from_code(task_code)
            trial_id = int(row[trial_col])

            sensor_file = find_sensor_file(
                sensors_root=sensors_root,
                subject_id=subject_id,
                task_id=task_id,
                trial_id=trial_id,
                fnum=fnum,
            )

            if sensor_file is None:
                print(f"[WARN] missing sensor file: dataset={name}, subject={subject_id}, task={task_id}, trial={trial_id}")
                continue

            records.append({
                "dataset": name,
                "subject_id": subject_id,
                "task_id": int(task_id),
                "trial_id": int(trial_id),
                "description": str(row[desc_col]) if desc_col is not None else "",
                "label_file": str(label_file),
                "sensor_file": str(sensor_file),
                "fall_onset_frame": int(row[onset_col]),
                "fall_impact_frame": int(row[impact_col]),
            })

    return pd.DataFrame(records)

def read_sensor(record):
    df = pd.read_csv(record["sensor_file"])

    # Standard UniVr-style scale:
    # Acc = mg, Gyro = mdps.
    for c in ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"]:
        if c not in df.columns:
            raise ValueError(f"Missing {c} in {record['sensor_file']}")

    df = df.copy()
    df["FrameCounter"] = pd.to_numeric(df["FrameCounter"], errors="coerce").fillna(0).astype(int)

    # Time column: first unnamed column is usually timestamp in ms.
    first_col = df.columns[0]
    if str(first_col).startswith("Unnamed"):
        ms = pd.to_numeric(df[first_col], errors="coerce").fillna(0).to_numpy(float)
        if len(ms) > 1 and np.nanmedian(np.diff(ms)) > 0:
            df["Time_s"] = (ms - ms[0]) / 1000.0
        else:
            df["Time_s"] = np.arange(len(df)) / 100.0
    else:
        df["Time_s"] = np.arange(len(df)) / 100.0

    df["AccX_g"] = pd.to_numeric(df["AccX"], errors="coerce").fillna(0) / 1000.0
    df["AccY_g"] = pd.to_numeric(df["AccY"], errors="coerce").fillna(0) / 1000.0
    df["AccZ_g"] = pd.to_numeric(df["AccZ"], errors="coerce").fillna(0) / 1000.0

    df["GyrX_dps"] = pd.to_numeric(df["GyrX"], errors="coerce").fillna(0) / 1000.0
    df["GyrY_dps"] = pd.to_numeric(df["GyrY"], errors="coerce").fillna(0) / 1000.0
    df["GyrZ_dps"] = pd.to_numeric(df["GyrZ"], errors="coerce").fillna(0) / 1000.0

    df["AccMag_g"] = np.sqrt(df["AccX_g"]**2 + df["AccY_g"]**2 + df["AccZ_g"]**2)
    df["GyrMag_dps"] = np.sqrt(df["GyrX_dps"]**2 + df["GyrY_dps"]**2 + df["GyrZ_dps"]**2)

    return df

def pos_from_frame(df, frame_value):
    fc = df["FrameCounter"].to_numpy()
    hits = np.where(fc == int(frame_value))[0]
    if len(hits):
        return int(hits[0])

    if 0 <= int(frame_value) < len(df):
        return int(frame_value)

    if 1 <= int(frame_value) <= len(df):
        return int(frame_value) - 1

    return max(0, min(int(frame_value), len(df)-1))

def window_df(df, onset_pos, impact_pos, pre_s=2.0, post_s=2.0, hz=100):
    start = max(0, onset_pos - int(pre_s * hz))
    end = min(len(df)-1, impact_pos + int(post_s * hz))

    w = df.iloc[start:end+1].copy()
    onset_time = df.iloc[onset_pos]["Time_s"]
    w["RelTime_s"] = w["Time_s"] - onset_time
    w["Phase"] = "pre"
    w.loc[w.index >= onset_pos, "Phase"] = "fall_to_impact"
    w.loc[w.index > impact_pos, "Phase"] = "post_impact"
    return w

def features_for_record(record):
    df = read_sensor(record)
    onset = pos_from_frame(df, record["fall_onset_frame"])
    impact = pos_from_frame(df, record["fall_impact_frame"])
    if impact < onset:
        impact = onset

    fall = df.iloc[onset:impact+1].copy()
    w = window_df(df, onset, impact)

    duration_s = (impact - onset) / 100.0

    if len(fall) < 1:
        fall = df.iloc[[onset]].copy()

    accmag = fall["AccMag_g"].to_numpy(float)
    gyrmag = fall["GyrMag_dps"].to_numpy(float)

    jerk = np.diff(df["AccMag_g"].to_numpy(float)) * 100.0
    if len(jerk) == 0:
        jerk_peak = 0.0
    else:
        jstart = max(0, onset-1)
        jend = min(len(jerk), impact+1)
        jerk_peak = float(np.nanmax(np.abs(jerk[jstart:jend]))) if jend > jstart else 0.0

    first = df.iloc[:min(100, len(df))]

    out = {
        **record,
        "n_frames": len(df),
        "onset_pos": onset,
        "impact_pos": impact,
        "fall_duration_s": duration_s,
        "peak_acc_g": float(np.nanmax(accmag)),
        "mean_acc_g": float(np.nanmean(accmag)),
        "rms_acc_g": float(np.sqrt(np.nanmean(accmag**2))),
        "peak_gyr_dps": float(np.nanmax(gyrmag)),
        "mean_gyr_dps": float(np.nanmean(gyrmag)),
        "rms_gyr_dps": float(np.sqrt(np.nanmean(gyrmag**2))),
        "peak_jerk_gps": jerk_peak,
        "baseline_AccX_g": float(first["AccX_g"].median()),
        "baseline_AccY_g": float(first["AccY_g"].median()),
        "baseline_AccZ_g": float(first["AccZ_g"].median()),
        "baseline_AccMag_g": float(first["AccMag_g"].median()),
    }

    baseline_abs = {
        "X": abs(out["baseline_AccX_g"]),
        "Y": abs(out["baseline_AccY_g"]),
        "Z": abs(out["baseline_AccZ_g"]),
    }
    out["dominant_gravity_axis_first_1s"] = max(baseline_abs, key=baseline_abs.get)

    return out, df, w

def normalized_signal(df, onset_pos, impact_pos, signal, n=101):
    if impact_pos <= onset_pos:
        return None

    seg = df.iloc[onset_pos:impact_pos+1]
    y = seg[signal].to_numpy(float)

    if len(y) < 2:
        return None

    x_old = np.linspace(0, 1, len(y))
    x_new = np.linspace(0, 1, n)
    return np.interp(x_new, x_old, y)

def corr_safe(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    if np.nanstd(a) < 1e-9 or np.nanstd(b) < 1e-9:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])

def rmse(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.sqrt(np.nanmean((a-b)**2)))

def zrmse(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    az = (a - np.nanmean(a)) / (np.nanstd(a) + 1e-9)
    bz = (b - np.nanmean(b)) / (np.nanstd(b) + 1e-9)

    return float(np.sqrt(np.nanmean((az-bz)**2)))

def ensure(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

def plot_per_task(task_id, real_rec, sim_rec, real_df, sim_df, out_dir):
    r_on = real_rec["onset_pos"]
    r_imp = real_rec["impact_pos"]
    s_on = sim_rec["onset_pos"]
    s_imp = sim_rec["impact_pos"]

    rw = window_df(real_df, r_on, r_imp)
    sw = window_df(sim_df, s_on, s_imp)

    # Magnitude overlay
    plt.figure(figsize=(12, 6))
    plt.plot(rw["RelTime_s"], rw["AccMag_g"], label="Real AccMag")
    plt.plot(sw["RelTime_s"], sw["AccMag_g"], label="Sim AccMag")
    plt.axvline(0, linestyle="--", label="Fall onset")
    plt.axvline((r_imp-r_on)/100.0, linestyle=":", label="Real impact")
    plt.axvline((s_imp-s_on)/100.0, linestyle="-.", label="Sim impact")
    plt.xlabel("Time relative to fall onset (s)")
    plt.ylabel("Acceleration magnitude (g)")
    plt.title(f"Task {task_id}: Acceleration magnitude around fall")
    plt.legend()
    savefig(out_dir / f"task_{task_id}_accmag_overlay.png")

    plt.figure(figsize=(12, 6))
    plt.plot(rw["RelTime_s"], rw["GyrMag_dps"], label="Real GyrMag")
    plt.plot(sw["RelTime_s"], sw["GyrMag_dps"], label="Sim GyrMag")
    plt.axvline(0, linestyle="--", label="Fall onset")
    plt.axvline((r_imp-r_on)/100.0, linestyle=":", label="Real impact")
    plt.axvline((s_imp-s_on)/100.0, linestyle="-.", label="Sim impact")
    plt.xlabel("Time relative to fall onset (s)")
    plt.ylabel("Gyroscope magnitude (deg/s)")
    plt.title(f"Task {task_id}: Gyroscope magnitude around fall")
    plt.legend()
    savefig(out_dir / f"task_{task_id}_gyrmag_overlay.png")

    # Axis overlay
    for group, cols, ylabel in [
        ("acc_axes", ["AccX_g", "AccY_g", "AccZ_g"], "Acceleration (g)"),
        ("gyr_axes", ["GyrX_dps", "GyrY_dps", "GyrZ_dps"], "Gyroscope (deg/s)"),
    ]:
        fig, axes = plt.subplots(len(cols), 1, figsize=(12, 9), sharex=True)
        for ax, c in zip(axes, cols):
            ax.plot(rw["RelTime_s"], rw[c], label=f"Real {c}")
            ax.plot(sw["RelTime_s"], sw[c], label=f"Sim {c}")
            ax.axvline(0, linestyle="--")
            ax.set_ylabel(ylabel)
            ax.legend()
        axes[-1].set_xlabel("Time relative to fall onset (s)")
        fig.suptitle(f"Task {task_id}: {group}")
        savefig(out_dir / f"task_{task_id}_{group}_overlay.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--physical", default="/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/outputs/testing/physical_dataset")
    ap.add_argument("--simulated", default="/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/outputs/testing/simulated")
    ap.add_argument("--out", default="/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/outputs/testing/outcomes_real_vs_sim")
    args = ap.parse_args()

    out = Path(args.out)
    tables = out / "tables"
    plots = out / "plots"
    per_task = plots / "per_task"
    distributions = plots / "distributions"
    features_plot = plots / "features"
    phase_plot = plots / "normalized_phase"

    for p in [tables, per_task, distributions, features_plot, phase_plot]:
        ensure(p)

    print("Loading physical dataset...")
    real_records = load_dataset(args.physical, "real")
    print("Loading simulated dataset...")
    sim_records = load_dataset(args.simulated, "simulated")

    inventory = pd.concat([real_records, sim_records], ignore_index=True)
    inventory.to_csv(tables / "dataset_inventory.csv", index=False)

    all_features = []
    data_cache = {}

    print("Extracting features...")
    for _, rec in inventory.iterrows():
        rec_dict = rec.to_dict()
        feat, df, w = features_for_record(rec_dict)
        all_features.append(feat)
        key = (feat["dataset"], feat["task_id"], feat["trial_id"])
        data_cache[key] = {"df": df, "window": w, "features": feat}

    features = pd.DataFrame(all_features)
    features.to_csv(tables / "features_all_trials.csv", index=False)

    axis_cols = [
        "dataset", "subject_id", "task_id", "trial_id",
        "baseline_AccX_g", "baseline_AccY_g", "baseline_AccZ_g",
        "baseline_AccMag_g", "dominant_gravity_axis_first_1s",
    ]
    features[axis_cols].to_csv(tables / "axis_unit_baseline_checks.csv", index=False)

    # Label comparison table
    label_summary = features[[
        "dataset", "subject_id", "task_id", "trial_id",
        "fall_onset_frame", "fall_impact_frame",
        "onset_pos", "impact_pos", "fall_duration_s",
    ]].copy()
    label_summary.to_csv(tables / "label_window_summary.csv", index=False)

    # Similarity only where task exists in both real and simulated.
    similarity_rows = []

    common_tasks = sorted(set(real_records["task_id"]).intersection(set(sim_records["task_id"])))

    for task in common_tasks:
        real_task = features[(features["dataset"] == "real") & (features["task_id"] == task)]
        sim_task = features[(features["dataset"] == "simulated") & (features["task_id"] == task)]

        if len(real_task) == 0 or len(sim_task) == 0:
            continue

        # Compare first real and first simulated trial for this task.
        rr = real_task.iloc[0].to_dict()
        sr = sim_task.iloc[0].to_dict()

        rkey = ("real", int(rr["task_id"]), int(rr["trial_id"]))
        skey = ("simulated", int(sr["task_id"]), int(sr["trial_id"]))

        rdf = data_cache[rkey]["df"]
        sdf = data_cache[skey]["df"]

        plot_per_task(task, rr, sr, rdf, sdf, per_task)

        row = {
            "task_id": task,
            "real_trial_id": rr["trial_id"],
            "sim_trial_id": sr["trial_id"],
            "real_duration_s": rr["fall_duration_s"],
            "sim_duration_s": sr["fall_duration_s"],
            "duration_diff_s": sr["fall_duration_s"] - rr["fall_duration_s"],
            "real_peak_acc_g": rr["peak_acc_g"],
            "sim_peak_acc_g": sr["peak_acc_g"],
            "peak_acc_ratio_sim_over_real": sr["peak_acc_g"] / (rr["peak_acc_g"] + 1e-9),
            "real_peak_gyr_dps": rr["peak_gyr_dps"],
            "sim_peak_gyr_dps": sr["peak_gyr_dps"],
            "peak_gyr_ratio_sim_over_real": sr["peak_gyr_dps"] / (rr["peak_gyr_dps"] + 1e-9),
        }

        for sig in SIGNALS:
            a = normalized_signal(rdf, int(rr["onset_pos"]), int(rr["impact_pos"]), sig)
            b = normalized_signal(sdf, int(sr["onset_pos"]), int(sr["impact_pos"]), sig)

            if a is None or b is None:
                row[f"{sig}_corr"] = np.nan
                row[f"{sig}_rmse"] = np.nan
                row[f"{sig}_zrmse"] = np.nan
            else:
                row[f"{sig}_corr"] = corr_safe(a, b)
                row[f"{sig}_rmse"] = rmse(a, b)
                row[f"{sig}_zrmse"] = zrmse(a, b)

        similarity_rows.append(row)

    similarity = pd.DataFrame(similarity_rows)
    similarity.to_csv(tables / "similarity_by_task.csv", index=False)

    # Plot duration by task
    plt.figure(figsize=(14, 6))
    for ds in ["real", "simulated"]:
        sub = features[features["dataset"] == ds].sort_values("task_id")
        plt.plot(sub["task_id"], sub["fall_duration_s"], marker="o", label=ds)
    plt.xlabel("Task ID")
    plt.ylabel("Onset-to-impact duration (s)")
    plt.title("Fall duration comparison")
    plt.legend()
    savefig(features_plot / "fall_duration_by_task.png")

    # Peak acceleration by task
    plt.figure(figsize=(14, 6))
    for ds in ["real", "simulated"]:
        sub = features[features["dataset"] == ds].sort_values("task_id")
        plt.plot(sub["task_id"], sub["peak_acc_g"], marker="o", label=ds)
    plt.xlabel("Task ID")
    plt.ylabel("Peak acceleration during fall window (g)")
    plt.title("Peak acceleration comparison")
    plt.legend()
    savefig(features_plot / "peak_acc_by_task.png")

    # Peak gyro by task
    plt.figure(figsize=(14, 6))
    for ds in ["real", "simulated"]:
        sub = features[features["dataset"] == ds].sort_values("task_id")
        plt.plot(sub["task_id"], sub["peak_gyr_dps"], marker="o", label=ds)
    plt.xlabel("Task ID")
    plt.ylabel("Peak gyroscope magnitude during fall window (deg/s)")
    plt.title("Peak gyroscope comparison")
    plt.legend()
    savefig(features_plot / "peak_gyr_by_task.png")

    # Feature boxplots
    for metric in ["fall_duration_s", "peak_acc_g", "peak_gyr_dps", "rms_acc_g", "rms_gyr_dps", "peak_jerk_gps"]:
        plt.figure(figsize=(8, 6))
        data = [
            features[features["dataset"] == "real"][metric].dropna(),
            features[features["dataset"] == "simulated"][metric].dropna(),
        ]
        plt.boxplot(data, labels=["real", "simulated"])
        plt.ylabel(metric)
        plt.title(f"Distribution comparison: {metric}")
        savefig(distributions / f"boxplot_{metric}.png")

    # Baseline gravity axis comparison
    plt.figure(figsize=(10, 6))
    for ds in ["real", "simulated"]:
        sub = features[features["dataset"] == ds]
        vals = [
            sub["baseline_AccX_g"].median(),
            sub["baseline_AccY_g"].median(),
            sub["baseline_AccZ_g"].median(),
        ]
        plt.plot(["AccX", "AccY", "AccZ"], vals, marker="o", label=ds)
    plt.axhline(1.0, linestyle="--")
    plt.axhline(-1.0, linestyle="--")
    plt.ylabel("Median first-second acceleration (g)")
    plt.title("Axis / gravity baseline comparison")
    plt.legend()
    savefig(distributions / "axis_gravity_baseline_comparison.png")

    # Similarity plots
    if len(similarity):
        plt.figure(figsize=(14, 6))
        plt.plot(similarity["task_id"], similarity["AccMag_g_corr"], marker="o", label="AccMag correlation")
        plt.plot(similarity["task_id"], similarity["GyrMag_dps_corr"], marker="o", label="GyrMag correlation")
        plt.xlabel("Task ID")
        plt.ylabel("Correlation after onset-impact phase normalization")
        plt.ylim(-1.05, 1.05)
        plt.title("Real vs simulated phase-shape similarity")
        plt.legend()
        savefig(features_plot / "similarity_correlation_by_task.png")

        plt.figure(figsize=(14, 6))
        plt.plot(similarity["task_id"], similarity["AccMag_g_zrmse"], marker="o", label="AccMag zRMSE")
        plt.plot(similarity["task_id"], similarity["GyrMag_dps_zrmse"], marker="o", label="GyrMag zRMSE")
        plt.xlabel("Task ID")
        plt.ylabel("z-normalized RMSE")
        plt.title("Real vs simulated normalized-shape error")
        plt.legend()
        savefig(features_plot / "similarity_zrmse_by_task.png")

    # Normalized phase mean curves across common tasks
    for sig in ["AccMag_g", "GyrMag_dps", "AccX_g", "AccY_g", "AccZ_g"]:
        plt.figure(figsize=(10, 6))
        x = np.linspace(0, 1, 101)

        for ds in ["real", "simulated"]:
            curves = []
            for _, fr in features[features["dataset"] == ds].iterrows():
                task = int(fr["task_id"])
                if task not in common_tasks:
                    continue
                key = (ds, task, int(fr["trial_id"]))
                df = data_cache[key]["df"]
                y = normalized_signal(df, int(fr["onset_pos"]), int(fr["impact_pos"]), sig)
                if y is not None:
                    curves.append(y)
            if curves:
                curves = np.vstack(curves)
                plt.plot(x, np.nanmean(curves, axis=0), label=ds)

        plt.xlabel("Normalized fall phase: onset=0, impact=1")
        plt.ylabel(sig)
        plt.title(f"Average normalized fall phase: {sig}")
        plt.legend()
        savefig(phase_plot / f"normalized_phase_mean_{sig}.png")

    # Optional PCA feature space
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        pca_features = [
            "fall_duration_s", "peak_acc_g", "mean_acc_g", "rms_acc_g",
            "peak_gyr_dps", "mean_gyr_dps", "rms_gyr_dps", "peak_jerk_gps",
            "baseline_AccX_g", "baseline_AccY_g", "baseline_AccZ_g",
        ]
        clean = features.dropna(subset=pca_features).copy()
        X = StandardScaler().fit_transform(clean[pca_features].to_numpy(float))
        p = PCA(n_components=2).fit_transform(X)
        clean["PC1"] = p[:, 0]
        clean["PC2"] = p[:, 1]
        clean.to_csv(tables / "pca_feature_space.csv", index=False)

        plt.figure(figsize=(9, 7))
        for ds in ["real", "simulated"]:
            sub = clean[clean["dataset"] == ds]
            plt.scatter(sub["PC1"], sub["PC2"], label=ds)
            for _, r in sub.iterrows():
                plt.text(r["PC1"], r["PC2"], str(int(r["task_id"])), fontsize=8)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Feature-space comparison: real vs simulated")
        plt.legend()
        savefig(features_plot / "pca_feature_space.png")
    except Exception as e:
        print("[WARN] PCA skipped:", e)

    summary = {
        "physical_root": str(args.physical),
        "simulated_root": str(args.simulated),
        "outcomes_folder": str(out),
        "n_real_records": int(len(real_records)),
        "n_simulated_records": int(len(sim_records)),
        "n_common_tasks": int(len(common_tasks)),
        "common_tasks": common_tasks,
        "tables": {
            "inventory": str(tables / "dataset_inventory.csv"),
            "features": str(tables / "features_all_trials.csv"),
            "similarity": str(tables / "similarity_by_task.csv"),
            "axis_checks": str(tables / "axis_unit_baseline_checks.csv"),
            "labels": str(tables / "label_window_summary.csv"),
        },
        "plots_folder": str(plots),
    }

    (out / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    readme = out / "README_RESULTS.txt"
    readme.write_text(
        "Real vs simulated UniVrFall-style analysis outcomes\n\n"
        "Main folders:\n"
        f"- Tables: {tables}\n"
        f"- Per-task plots: {per_task}\n"
        f"- Feature plots: {features_plot}\n"
        f"- Distribution plots: {distributions}\n"
        f"- Normalized phase plots: {phase_plot}\n\n"
        "Important tables:\n"
        "- features_all_trials.csv: extracted fall-window features\n"
        "- similarity_by_task.csv: real-vs-sim task similarity using normalized onset-impact phase\n"
        "- axis_unit_baseline_checks.csv: checks axis/gravity/unit baseline\n"
        "- label_window_summary.csv: onset/impact frame windows used\n\n"
        "Signal units used for analysis:\n"
        "- AccX/AccY/AccZ converted from mg to g\n"
        "- GyrX/GyrY/GyrZ converted from mdps to deg/s\n\n"
        "Alignment rule:\n"
        "- Fall onset and impact are read from each dataset label file.\n"
        "- Time series are aligned at fall onset.\n"
        "- Similarity curves are normalized from onset=0 to impact=1.\n",
        encoding="utf-8",
    )

    print("\n" + "="*90)
    print("REAL VS SIMULATED ANALYSIS COMPLETE")
    print("="*90)
    print("Real records     :", len(real_records))
    print("Sim records      :", len(sim_records))
    print("Common tasks     :", common_tasks)
    print("Outcomes folder  :", out)
    print("Tables           :", tables)
    print("Plots            :", plots)
    print("Summary JSON     :", out / "analysis_summary.json")
    print("="*90)

if __name__ == "__main__":
    main()
