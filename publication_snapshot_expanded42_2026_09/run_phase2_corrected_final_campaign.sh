#!/usr/bin/env bash
set -euo pipefail

cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project

ROOT="outputs/phase2_corrected_final_campaign_v1"
mkdir -p "$ROOT"

EXPERIMENTS=(
"EXP01_PHYSICAL_ONLY"
"EXP02_SIM_ONLY_CLOSED_DOMAIN"
"EXP03_MIX20"
"EXP04_MIX50"
"EXP05_MIX70"
"EXP06_MIX100"
)

echo "================================================================================"
echo "CORRECTED PHASE-2 FINAL CNN CAMPAIGN"
echo "================================================================================"
echo "Order: condition first, then all 5 folds"
echo "Total trainings: 30"
echo "Started: $(date)"
echo "================================================================================"

for EXPERIMENT in "${EXPERIMENTS[@]}"
do
    echo
    echo "################################################################################"
    echo "STARTING COMPLETE EXPERIMENT: ${EXPERIMENT}"
    echo "################################################################################"

    EXP_START=$(date +%s)

    for FOLD in 0 1 2 3 4
    do
        echo
        echo "================================================================================"
        echo "${EXPERIMENT} | STARTING FOLD ${FOLD}"
        echo "Time: $(date)"
        echo "================================================================================"

        FOLD_START=$(date +%s)

        PHASE2_EXPERIMENT="$EXPERIMENT" \
        PHASE2_FOLD="$FOLD" \
        CUDA_VISIBLE_DEVICES="${CAMPAIGN_GPU:-1}" \
        nice -n 5 \
        ./.venv/bin/python -u \
        run_phase2_corrected_final_condition_fold.py

        FOLD_END=$(date +%s)

        echo
        echo "${EXPERIMENT} | FOLD ${FOLD} COMPLETE"
        echo "Fold elapsed seconds: $((FOLD_END - FOLD_START))"
    done

    EXP_END=$(date +%s)

    echo
    echo "################################################################################"
    echo "ALL 5 FOLDS COMPLETE: ${EXPERIMENT}"
    echo "Experiment elapsed seconds: $((EXP_END - EXP_START))"
    echo "################################################################################"
done

echo
echo "================================================================================"
echo "ALL 30 CNN TRAININGS FINISHED"
echo "================================================================================"

./.venv/bin/python - <<'PY'
from pathlib import Path
import json
import pandas as pd

root = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/"
    "outputs/phase2_corrected_final_campaign_v1"
)

experiments = [
    "EXP01_PHYSICAL_ONLY",
    "EXP02_SIM_ONLY_CLOSED_DOMAIN",
    "EXP03_MIX20",
    "EXP04_MIX50",
    "EXP05_MIX70",
    "EXP06_MIX100",
]

required = [
    "best_model.pth",
    "metrics.json",
    "confusion_matrix.csv",
    "predictions.csv",
    "training_history.csv",
    "run_config.json",
    "training_window_manifest.csv",
]

rows = []
missing = []

for experiment in experiments:
    for fold in range(5):
        d = root / experiment / f"fold_{fold}"

        for filename in required:
            p = d / filename
            if not p.exists():
                missing.append(str(p))

        mp = d / "metrics.json"

        if mp.exists():
            with open(mp) as f:
                m = json.load(f)

            m["experiment"] = experiment
            m["fold"] = fold
            rows.append(m)

if missing:
    print("FINAL CAMPAIGN ARTIFACT GATE: FAIL")
    for p in missing:
        print(p)
    raise SystemExit(1)

df = pd.DataFrame(rows)

if len(df) != 30:
    raise RuntimeError(
        f"Expected 30 completed trainings, found {len(df)}"
    )

df = df.sort_values(
    ["experiment", "fold"]
).reset_index(drop=True)

df.to_csv(
    root / "full_campaign_summary_30runs.csv",
    index=False,
)

metrics = [
    "accuracy",
    "balanced_accuracy",
    "precision_fall",
    "recall_fall",
    "f1_fall",
    "specificity",
    "roc_auc",
    "average_precision",
]

metrics = [m for m in metrics if m in df.columns]

summary = (
    df.groupby("experiment")[metrics]
    .agg(["mean", "std"])
)

summary.columns = [
    f"{metric}_{stat}"
    for metric, stat in summary.columns
]

summary = summary.reset_index()

summary.to_csv(
    root / "condition_summary_mean_std.csv",
    index=False,
)

print()
print("=" * 110)
print("FINAL 30-RUN RESULTS")
print("=" * 110)

print(
    df[
        [
            "experiment",
            "fold",
            "f1_fall",
            "precision_fall",
            "recall_fall",
            "balanced_accuracy",
            "roc_auc",
            "average_precision",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print()
print("=" * 110)
print("FINAL 5-FOLD MEAN ± STD")
print("=" * 110)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print()
print("FINAL CAMPAIGN ARTIFACT GATE: PASS")
print("Total completed CNN trainings:", len(df))
PY

echo
echo "================================================================================"
echo "CORRECTED PHASE-2 FINAL CAMPAIGN COMPLETE"
echo "Finished: $(date)"
echo "================================================================================"
