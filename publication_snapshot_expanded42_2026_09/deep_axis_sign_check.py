from pathlib import Path
import itertools
import json
import numpy as np
import pandas as pd

from analyze_real_vs_sim_univr import (
    load_dataset,
    read_sensor,
    pos_from_frame,
    normalized_signal,
    corr_safe,
)

PHYSICAL = "/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/outputs/testing/physical_dataset"
SIMULATED = "/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/outputs/testing/simulated"
OUT = Path("/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/outputs/testing/outcomes_real_vs_sim/deep_axis_sign_check")
OUT.mkdir(parents=True, exist_ok=True)

AXES_ACC = ["AccX_g", "AccY_g", "AccZ_g"]
AXES_GYR = ["GyrX_dps", "GyrY_dps", "GyrZ_dps"]

def load_first_trials():
    real = load_dataset(PHYSICAL, "real")
    sim = load_dataset(SIMULATED, "simulated")

    common = sorted(set(real["task_id"]).intersection(set(sim["task_id"])))

    pairs = []
    for task in common:
        r = real[real["task_id"] == task].iloc[0].to_dict()
        s = sim[sim["task_id"] == task].iloc[0].to_dict()
        pairs.append((task, r, s))

    return pairs

def baseline_report(pairs):
    rows = []

    for task, r, s in pairs:
        for name, rec in [("real", r), ("simulated", s)]:
            df = read_sensor(rec)
            first = df.iloc[:min(100, len(df))]
            rows.append({
                "dataset": name,
                "task_id": task,
                "baseline_AccX_g": float(first["AccX_g"].median()),
                "baseline_AccY_g": float(first["AccY_g"].median()),
                "baseline_AccZ_g": float(first["AccZ_g"].median()),
                "baseline_AccMag_g": float(first["AccMag_g"].median()),
                "dominant_axis": max(
                    ["X", "Y", "Z"],
                    key=lambda a: abs(float(first[f"Acc{a}_g"].median()))
                ),
            })

    return pd.DataFrame(rows)

def get_curve(df, rec, signal):
    onset = pos_from_frame(df, rec["fall_onset_frame"])
    impact = pos_from_frame(df, rec["fall_impact_frame"])
    if impact <= onset:
        return None
    return normalized_signal(df, onset, impact, signal, n=101)

def score_mapping(pairs, axes):
    perms = list(itertools.permutations(range(3)))
    signs = list(itertools.product([-1, 1], repeat=3))

    results = []

    for perm in perms:
        for sign in signs:
            corrs = []
            per_task = []

            for task, r, s in pairs:
                rdf = read_sensor(r)
                sdf = read_sensor(s)

                task_corrs = []

                for real_axis_i, sim_axis_i in enumerate(perm):
                    real_sig = axes[real_axis_i]
                    sim_sig = axes[sim_axis_i]

                    a = get_curve(rdf, r, real_sig)
                    b = get_curve(sdf, s, sim_sig)

                    if a is None or b is None:
                        continue

                    b = sign[real_axis_i] * b
                    c = corr_safe(a, b)

                    if not np.isnan(c):
                        corrs.append(c)
                        task_corrs.append(c)

                per_task.append({
                    "task_id": task,
                    "mean_corr": float(np.nanmean(task_corrs)) if task_corrs else np.nan,
                })

            mapping = []
            for real_axis_i, sim_axis_i in enumerate(perm):
                real_name = axes[real_axis_i]
                sim_name = axes[sim_axis_i]
                sg = "+" if sign[real_axis_i] > 0 else "-"
                mapping.append(f"{real_name} <- {sg}{sim_name}")

            results.append({
                "mapping": "; ".join(mapping),
                "mean_corr": float(np.nanmean(corrs)) if corrs else np.nan,
                "median_corr": float(np.nanmedian(corrs)) if corrs else np.nan,
                "n_corrs": len(corrs),
                "per_task": per_task,
            })

    results = sorted(results, key=lambda x: -999 if np.isnan(x["mean_corr"]) else x["mean_corr"], reverse=True)
    return results

def main():
    pairs = load_first_trials()

    base = baseline_report(pairs)
    base.to_csv(OUT / "baseline_axis_report.csv", index=False)

    acc_results = score_mapping(pairs, AXES_ACC)
    gyr_results = score_mapping(pairs, AXES_GYR)

    pd.DataFrame([
        {k: v for k, v in r.items() if k != "per_task"}
        for r in acc_results
    ]).to_csv(OUT / "acc_axis_mapping_candidates.csv", index=False)

    pd.DataFrame([
        {k: v for k, v in r.items() if k != "per_task"}
        for r in gyr_results
    ]).to_csv(OUT / "gyro_axis_mapping_candidates.csv", index=False)

    report = {
        "interpretation": {
            "expected_acc_mapping": "AccX_g <- +AccX_g; AccY_g <- +AccY_g; AccZ_g <- +AccZ_g",
            "expected_gyr_mapping": "GyrX_dps <- +GyrX_dps; GyrY_dps <- +GyrY_dps; GyrZ_dps <- +GyrZ_dps",
            "note": "If a sign-flipped or permuted mapping is much better, check axis/sign conversion. If all scores are low, dynamics differ too much to infer signs from correlation alone."
        },
        "top_acc_candidates": [{k: v for k, v in r.items() if k != "per_task"} for r in acc_results[:10]],
        "top_gyro_candidates": [{k: v for k, v in r.items() if k != "per_task"} for r in gyr_results[:10]],
    }

    (OUT / "deep_axis_sign_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nDEEP AXIS/SIGN CHECK COMPLETE")
    print("Folder:", OUT)
    print("\nTop acceleration mapping:")
    print(report["top_acc_candidates"][0])
    print("\nTop gyro mapping:")
    print(report["top_gyro_candidates"][0])
    print("\nExpected current mapping:")
    print("AccX<-+AccX, AccY<-+AccY, AccZ<-+AccZ")
    print("GyrX<-+GyrX, GyrY<-+GyrY, GyrZ<-+GyrZ")

if __name__ == "__main__":
    main()
