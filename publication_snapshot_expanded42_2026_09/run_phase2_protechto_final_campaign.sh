#!/usr/bin/env bash
set -euo pipefail

cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project

ROOT="outputs/phase2_protechto_final_campaign_v1"
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
echo "FINAL PROTECHTO-PARITY PHASE-2 CAMPAIGN"
echo "================================================================================"
echo "Condition-major order"
echo "6 conditions x 5 folds = 30 trainings"
echo "Started: $(date)"
echo "================================================================================"

for EXPERIMENT in "${EXPERIMENTS[@]}"
do
    echo
    echo "################################################################################"
    echo "STARTING COMPLETE EXPERIMENT: ${EXPERIMENT}"
    echo "################################################################################"

    for FOLD in 0 1 2 3 4
    do
        echo
        echo "================================================================================"
        echo "${EXPERIMENT} | STARTING FOLD ${FOLD}"
        echo "Time: $(date)"
        echo "================================================================================"

        START=$(date +%s)

        PHASE2_EXPERIMENT="$EXPERIMENT" \
        PHASE2_FOLD="$FOLD" \
        CUDA_VISIBLE_DEVICES="${CAMPAIGN_GPU:-1}" \
        nice -n 5 \
        ./.venv/bin/python -u \
        run_phase2_protechto_final_condition_fold.py

        echo
        echo "${EXPERIMENT} | TRAINING COMPLETE | FOLD ${FOLD}"

        PHASE2_EXPERIMENT="$EXPERIMENT" \
        PHASE2_FOLD="$FOLD" \
        ./.venv/bin/python -u \
        evaluate_protechto_final_condition_fold_events.py

        END=$(date +%s)

        echo
        echo "${EXPERIMENT} | FOLD ${FOLD} COMPLETE"
        echo "Elapsed seconds: $((END - START))"
    done

    echo
    echo "ALL 5 FOLDS COMPLETE: ${EXPERIMENT}"
done

echo
echo "================================================================================"
echo "ALL 30 TRAININGS + EVENT EVALUATIONS FINISHED"
echo "================================================================================"

./.venv/bin/python - <<'PY'
from pathlib import Path
import json
import pandas as pd

root = Path(
    "outputs/phase2_protechto_final_campaign_v1"
)

experiments = [
    "EXP01_PHYSICAL_ONLY",
    "EXP02_SIM_ONLY_CLOSED_DOMAIN",
    "EXP03_MIX20",
    "EXP04_MIX50",
    "EXP05_MIX70",
    "EXP06_MIX100",
]

segment_rows = []
event_rows = []
missing = []

segment_required = [
    "best_model.pth",
    "metrics.json",
    "confusion_matrix.csv",
    "predictions.csv",
    "training_history.csv",
    "run_config.json",
    "training_window_manifest.csv",
]

event_required = [
    "event_predictions.csv",
    "event_metrics.csv",
    "event_confusion_matrix.csv",
]

for exp in experiments:
    for fold in range(5):

        d = root / exp / f"fold_{fold}"

        for name in segment_required:
            if not (d / name).exists():
                missing.append(str(d / name))

        ed = d / "protechto_event_evaluation"

        for name in event_required:
            if not (ed / name).exists():
                missing.append(str(ed / name))

        mp = d / "metrics.json"

        if mp.exists():
            with open(mp) as f:
                row = json.load(f)

            row["experiment"] = exp
            row["fold"] = fold
            segment_rows.append(row)

        ep = ed / "event_metrics.csv"

        if ep.exists():
            x = pd.read_csv(ep)

            if len(x) != 1:
                raise RuntimeError(
                    f"Expected one row in {ep}"
                )

            row = x.iloc[0].to_dict()
            row["experiment"] = exp
            row["fold"] = fold
            event_rows.append(row)

if missing:
    print(
        "FINAL PROTECHTO CAMPAIGN ARTIFACT GATE: FAIL"
    )

    for p in missing:
        print("MISSING:", p)

    raise SystemExit(1)

segment = pd.DataFrame(segment_rows)
event = pd.DataFrame(event_rows)

if len(segment) != 30:
    raise RuntimeError(
        f"Expected 30 segment results, got {len(segment)}"
    )

if len(event) != 30:
    raise RuntimeError(
        f"Expected 30 event results, got {len(event)}"
    )

segment = segment.sort_values(
    ["experiment", "fold"]
)

event = event.sort_values(
    ["experiment", "fold"]
)

segment.to_csv(
    root / "final_segment_results_30runs.csv",
    index=False,
)

event.to_csv(
    root / "final_event_results_30runs.csv",
    index=False,
)

segment_metrics = [
    "accuracy",
    "balanced_accuracy",
    "precision_fall",
    "recall_fall",
    "f1_fall",
    "specificity",
    "roc_auc",
    "average_precision",
]

event_metrics = [
    "accuracy",
    "balanced_accuracy",
    "precision_fall",
    "recall_fall",
    "f1_fall",
]

def summarize(df, metrics):

    cols = [
        c for c in metrics
        if c in df.columns
    ]

    out = (
        df.groupby("experiment")[cols]
        .agg(["mean", "std"])
    )

    out.columns = [
        f"{a}_{b}"
        for a, b in out.columns
    ]

    return out.reset_index()

segment_summary = summarize(
    segment,
    segment_metrics,
)

event_summary = summarize(
    event,
    event_metrics,
)

segment_summary.to_csv(
    root / "final_segment_summary_mean_std.csv",
    index=False,
)

event_summary.to_csv(
    root / "final_event_summary_mean_std.csv",
    index=False,
)

print()
print("=" * 120)
print("FINAL EVENT-LEVEL 5-FOLD SUMMARY")
print("=" * 120)

print(
    event_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print()
print(
    "FINAL PROTECHTO CAMPAIGN ARTIFACT GATE: PASS"
)
print(
    "Completed trainings:",
    len(segment),
)
print(
    "Completed event evaluations:",
    len(event),
)
PY

echo
echo "================================================================================"
echo "FINAL PROTECHTO-PARITY CAMPAIGN COMPLETE"
echo "Finished: $(date)"
echo "================================================================================"
