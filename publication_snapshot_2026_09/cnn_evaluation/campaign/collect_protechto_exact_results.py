#!/usr/bin/env python3

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

# ======================================================================================
# Frozen campaign locations
# ======================================================================================

PROJECT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"
)

ROOT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master"
)

CAMPAIGN = (
    PROJECT
    / "outputs/protechto_exact_full_campaign_v1"
)

RUN_ID = (
    CAMPAIGN
    / "RUN_ID"
).read_text().strip()

RESULT_ROOT = (
    ROOT
    / "results/CNN/300ms"
    / RUN_ID
)

POST = (
    CAMPAIGN
    / "final_results_posthoc"
)

POST.mkdir(
    parents=True,
    exist_ok=True,
)

sys.path.insert(
    0,
    str(ROOT),
)

import constants as const


PHYSICAL_CONDITIONS = [
    "EXP01_REAL_ONLY",
    "EXP02_SIM_FALL_SUBSTITUTION",
    "EXP04_MIX20",
    "EXP05_MIX50",
    "EXP06_MIX70",
    "EXP07_MIX100",
]

ALL_CONDITIONS = [
    "EXP01_REAL_ONLY",
    "EXP02_SIM_FALL_SUBSTITUTION",
    "EXP03_SIM_ONLY_FULL",
    "EXP04_MIX20",
    "EXP05_MIX50",
    "EXP06_MIX70",
    "EXP07_MIX100",
]

DISPLAY = {
    "EXP01_REAL_ONLY":
        "REAL_ONLY",

    "EXP02_SIM_FALL_SUBSTITUTION":
        "REAL_ACTIVITY + SIM_FALLING",

    "EXP03_SIM_ONLY_FULL":
        "SIM_ACTIVITY + SIM_FALLING",

    "EXP04_MIX20":
        "REAL + SIM20",

    "EXP05_MIX50":
        "REAL + SIM50",

    "EXP06_MIX70":
        "REAL + SIM70",

    "EXP07_MIX100":
        "REAL + SIM100",
}


# ======================================================================================
# Metric helpers
# ======================================================================================

def div(a, b):
    if b == 0:
        return np.nan
    return float(a) / float(b)


def f1(p, r):
    if np.isnan(p) or np.isnan(r):
        return np.nan

    if p + r == 0:
        return 0.0

    return (
        2.0
        * p
        * r
        / (p + r)
    )


def metrics_from_counts(
    tn,
    fp,
    fn,
    tp,
):

    tn = int(tn)
    fp = int(fp)
    fn = int(fn)
    tp = int(tp)

    n = (
        tn
        + fp
        + fn
        + tp
    )

    support_activity = (
        tn + fp
    )

    support_fall = (
        tp + fn
    )

    activity_precision = div(
        tn,
        tn + fn,
    )

    activity_recall = div(
        tn,
        tn + fp,
    )

    activity_f1 = f1(
        activity_precision,
        activity_recall,
    )

    fall_precision = div(
        tp,
        tp + fp,
    )

    fall_recall = div(
        tp,
        tp + fn,
    )

    fall_f1 = f1(
        fall_precision,
        fall_recall,
    )

    overall_accuracy = div(
        tn + tp,
        n,
    )

    if (
        support_activity > 0
        and support_fall > 0
    ):
        balanced_accuracy = (
            activity_recall
            + fall_recall
        ) / 2.0
    else:
        balanced_accuracy = np.nan

    return {
        "n": n,

        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,

        "support_activity":
            support_activity,

        "support_fall":
            support_fall,

        "overall_accuracy":
            overall_accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "activity_precision":
            activity_precision,

        "activity_recall":
            activity_recall,

        "activity_f1":
            activity_f1,

        "fall_precision":
            fall_precision,

        "fall_recall":
            fall_recall,

        "fall_f1":
            fall_f1,
    }


def metrics_from_arrays(
    y_true,
    y_pred,
):

    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.int64,
    )

    tn = int(
        np.sum(
            (y_true == 0)
            & (y_pred == 0)
        )
    )

    fp = int(
        np.sum(
            (y_true == 0)
            & (y_pred == 1)
        )
    )

    fn = int(
        np.sum(
            (y_true == 1)
            & (y_pred == 0)
        )
    )

    tp = int(
        np.sum(
            (y_true == 1)
            & (y_pred == 1)
        )
    )

    return metrics_from_counts(
        tn,
        fp,
        fn,
        tp,
    )


# ======================================================================================
# Physical event parser
#
# Important fix:
# event_stats contains task="Overall".
# We intentionally discard every non-numeric task row.
# ======================================================================================

def physical_event_counts(
    condition,
    fold,
):

    p = (
        RESULT_ROOT
        / condition
        / str(fold)
        / "event_stats__bias_0.65.csv"
    )

    if not p.exists():
        raise RuntimeError(
            f"Missing event file: {p}"
        )

    df = pd.read_csv(p)

    required = {
        "task",
        "passed_simulations",
        "missed_simulations",
    }

    if not required.issubset(
        df.columns
    ):
        raise RuntimeError(
            f"Unexpected columns in {p}: "
            f"{df.columns.tolist()}"
        )

    task_numeric = pd.to_numeric(
        df["task"],
        errors="coerce",
    )

    # Removes:
    #     Overall
    # or any other report-only summary row.
    keep = task_numeric.notna()

    df = df.loc[keep].copy()

    df["task_numeric"] = (
        task_numeric.loc[keep]
        .astype(int)
    )

    df["passed_simulations"] = (
        pd.to_numeric(
            df["passed_simulations"],
            errors="raise",
        )
        .astype(int)
    )

    df["missed_simulations"] = (
        pd.to_numeric(
            df["missed_simulations"],
            errors="raise",
        )
        .astype(int)
    )

    tn = fp = fn = tp = 0

    for r in df.itertuples():

        task = int(
            r.task_numeric
        )

        passed = int(
            r.passed_simulations
        )

        missed = int(
            r.missed_simulations
        )

        if task in const.FALL_TASKS:
            tp += passed
            fn += missed

        else:
            tn += passed
            fp += missed

    return (
        tn,
        fp,
        fn,
        tp,
    )


# ======================================================================================
# Pre-aggregation artifact gate
# ======================================================================================

print(
    "=" * 126
)

print(
    "PROTECHTO EXACT CAMPAIGN — FINAL POST-HOC RESULTS"
)

print(
    "=" * 126
)

print(
    "RUN_ID:",
    RUN_ID,
)

print(
    "Result root:",
    RESULT_ROOT,
)

print(
    "Post-hoc output:",
    POST,
)

markers = list(
    CAMPAIGN.glob(
        "conditions/*/folds/"
        "fold_*.COMPLETE"
    )
)

print()
print(
    "Completed fold markers:",
    len(markers),
)

if len(markers) != 35:
    raise RuntimeError(
        "Expected exactly 35 completed folds."
    )


for condition in PHYSICAL_CONDITIONS:

    d = (
        RESULT_ROOT
        / condition
    )

    for fold in range(
        1,
        6,
    ):

        required = [
            d / str(fold) / "y_true.npy",
            d / str(fold) / "y_pred.npy",
            d / str(fold)
            / "event_stats__bias_0.65.csv",
        ]

        for p in required:
            if not p.exists():
                raise RuntimeError(
                    f"Missing physical result artifact: {p}"
                )


for fold in range(
    1,
    6,
):

    d = (
        CAMPAIGN
        / "conditions"
        / "EXP03_SIM_ONLY_FULL"
        / f"fold_{fold}"
    )

    required = [
        d / "segment_predictions.npz",
        d / "event_predictions.csv",
    ]

    for p in required:
        if not p.exists():
            raise RuntimeError(
                f"Missing simulation-only result: {p}"
            )


print(
    "35-fold result artifact gate: PASS"
)


# ======================================================================================
# Fold-level metrics
# ======================================================================================

fold_rows = []


# --------------------------------------------------------------------------------------
# Six conditions tested on held-out PHYSICAL data
# --------------------------------------------------------------------------------------

for condition in PHYSICAL_CONDITIONS:

    result_dir = (
        RESULT_ROOT
        / condition
    )

    for fold in range(
        1,
        6,
    ):

        yt = np.load(
            result_dir
            / str(fold)
            / "y_true.npy"
        ).astype(
            np.int64
        )

        yp = np.load(
            result_dir
            / str(fold)
            / "y_pred.npy"
        ).astype(
            np.int64
        )

        sm = metrics_from_arrays(
            yt,
            yp,
        )

        fold_rows.append({
            "condition":
                condition,

            "condition_display":
                DISPLAY[condition],

            "evaluation_domain":
                "Held-out physical",

            "level":
                "Segment",

            "fold":
                fold,

            **sm,
        })

        tn, fp, fn, tp = (
            physical_event_counts(
                condition,
                fold,
            )
        )

        em = metrics_from_counts(
            tn,
            fp,
            fn,
            tp,
        )

        fold_rows.append({
            "condition":
                condition,

            "condition_display":
                DISPLAY[condition],

            "evaluation_domain":
                "Held-out physical",

            "level":
                "Event",

            "fold":
                fold,

            **em,
        })


# --------------------------------------------------------------------------------------
# Simulation-only held-out complete source trials
# --------------------------------------------------------------------------------------

condition = (
    "EXP03_SIM_ONLY_FULL"
)

for fold in range(
    1,
    6,
):

    d = (
        CAMPAIGN
        / "conditions"
        / condition
        / f"fold_{fold}"
    )

    z = np.load(
        d / "segment_predictions.npz"
    )

    sm = metrics_from_arrays(
        z["y_true"],
        z["y_pred"],
    )

    fold_rows.append({
        "condition":
            condition,

        "condition_display":
            DISPLAY[condition],

        "evaluation_domain":
            "Held-out simulated trials",

        "level":
            "Segment",

        "fold":
            fold,

        **sm,
    })

    ev = pd.read_csv(
        d / "event_predictions.csv"
    )

    em = metrics_from_arrays(
        ev["true_event"].to_numpy(
            dtype=np.int64
        ),
        ev["pred_event"].to_numpy(
            dtype=np.int64
        ),
    )

    fold_rows.append({
        "condition":
            condition,

        "condition_display":
            DISPLAY[condition],

        "evaluation_domain":
            "Held-out simulated trials",

        "level":
            "Event",

        "fold":
            fold,

        **em,
    })


fold_df = pd.DataFrame(
    fold_rows
)

fold_df.to_csv(
    POST
    / "ALL_FOLD_METRICS.csv",
    index=False,
)


# ======================================================================================
# Pooled five-fold metrics
#
# Pooled values are the main descriptive result because the physical folds have very
# different numbers of windows. We concatenate/sum predictions across all held-out folds.
# ======================================================================================

pooled_rows = []


for condition in PHYSICAL_CONDITIONS:

    result_dir = (
        RESULT_ROOT
        / condition
    )

    y_true = []
    y_pred = []

    etn = efp = efn = etp = 0

    for fold in range(
        1,
        6,
    ):

        y_true.append(
            np.load(
                result_dir
                / str(fold)
                / "y_true.npy"
            ).astype(
                np.int64
            )
        )

        y_pred.append(
            np.load(
                result_dir
                / str(fold)
                / "y_pred.npy"
            ).astype(
                np.int64
            )
        )

        tn, fp, fn, tp = (
            physical_event_counts(
                condition,
                fold,
            )
        )

        etn += tn
        efp += fp
        efn += fn
        etp += tp

    sm = metrics_from_arrays(
        np.concatenate(y_true),
        np.concatenate(y_pred),
    )

    pooled_rows.append({
        "condition":
            condition,

        "condition_display":
            DISPLAY[condition],

        "evaluation_domain":
            "Held-out physical",

        "level":
            "Segment",

        **sm,
    })

    em = metrics_from_counts(
        etn,
        efp,
        efn,
        etp,
    )

    pooled_rows.append({
        "condition":
            condition,

        "condition_display":
            DISPLAY[condition],

        "evaluation_domain":
            "Held-out physical",

        "level":
            "Event",

        **em,
    })


# Simulation only pooled
condition = (
    "EXP03_SIM_ONLY_FULL"
)

segment_true = []
segment_pred = []

event_true = []
event_pred = []

for fold in range(
    1,
    6,
):

    d = (
        CAMPAIGN
        / "conditions"
        / condition
        / f"fold_{fold}"
    )

    z = np.load(
        d / "segment_predictions.npz"
    )

    segment_true.append(
        z["y_true"].astype(
            np.int64
        )
    )

    segment_pred.append(
        z["y_pred"].astype(
            np.int64
        )
    )

    ev = pd.read_csv(
        d / "event_predictions.csv"
    )

    event_true.append(
        ev["true_event"].to_numpy(
            dtype=np.int64
        )
    )

    event_pred.append(
        ev["pred_event"].to_numpy(
            dtype=np.int64
        )
    )


sm = metrics_from_arrays(
    np.concatenate(
        segment_true
    ),
    np.concatenate(
        segment_pred
    ),
)

pooled_rows.append({
    "condition":
        condition,

    "condition_display":
        DISPLAY[condition],

    "evaluation_domain":
        "Held-out simulated trials",

    "level":
        "Segment",

    **sm,
})


em = metrics_from_arrays(
    np.concatenate(
        event_true
    ),
    np.concatenate(
        event_pred
    ),
)

pooled_rows.append({
    "condition":
        condition,

    "condition_display":
        DISPLAY[condition],

    "evaluation_domain":
        "Held-out simulated trials",

    "level":
        "Event",

    **em,
})


pooled = pd.DataFrame(
    pooled_rows
)

pooled.to_csv(
    POST
    / "POOLED_ALL_RESULTS.csv",
    index=False,
)


# ======================================================================================
# Fold mean ± SD
# ======================================================================================

metric_cols = [
    "overall_accuracy",
    "balanced_accuracy",

    "activity_precision",
    "activity_recall",
    "activity_f1",

    "fall_precision",
    "fall_recall",
    "fall_f1",
]


stats_rows = []

for (
    condition,
    display,
    domain,
    level,
), g in fold_df.groupby([
    "condition",
    "condition_display",
    "evaluation_domain",
    "level",
], sort=False):

    row = {
        "condition":
            condition,

        "condition_display":
            display,

        "evaluation_domain":
            domain,

        "level":
            level,
    }

    for col in metric_cols:

        values = g[col].to_numpy(
            dtype=float
        )

        valid = values[
            np.isfinite(values)
        ]

        if len(valid) == 0:

            row[
                f"{col}_mean"
            ] = np.nan

            row[
                f"{col}_std"
            ] = np.nan

        else:

            row[
                f"{col}_mean"
            ] = float(
                np.mean(valid)
            )

            row[
                f"{col}_std"
            ] = float(
                np.std(
                    valid,
                    ddof=1,
                )
            ) if len(valid) > 1 else 0.0

    stats_rows.append(
        row
    )


stats = pd.DataFrame(
    stats_rows
)

stats.to_csv(
    POST
    / "FOLD_MEAN_STD.csv",
    index=False,
)


# ======================================================================================
# Human/paper-ready percentage tables
# ======================================================================================

def pct(x):
    if pd.isna(x):
        return "N/A"

    return (
        f"{100.0 * float(x):.2f}"
    )


def mean_sd(
    mean,
    sd,
):
    if (
        pd.isna(mean)
        or pd.isna(sd)
    ):
        return "N/A"

    return (
        f"{100*float(mean):.2f}"
        f" ± "
        f"{100*float(sd):.2f}"
    )


order = {
    c: i
    for i, c in enumerate(
        ALL_CONDITIONS
    )
}


# --------------------------------------------------------------------------------------
# POOLED SEGMENT TABLE
# --------------------------------------------------------------------------------------

segment = pooled[
    pooled["level"] == "Segment"
].copy()

segment["order"] = (
    segment["condition"]
    .map(order)
)

segment = segment.sort_values(
    "order"
)

segment_table = pd.DataFrame({
    "Condition":
        segment[
            "condition_display"
        ],

    "Evaluation":
        segment[
            "evaluation_domain"
        ],

    "N windows":
        segment[
            "n"
        ].astype(int),

    "Overall Acc (%)":
        segment[
            "overall_accuracy"
        ].map(pct),

    "Balanced Acc (%)":
        segment[
            "balanced_accuracy"
        ].map(pct),

    "Activity Precision (%)":
        segment[
            "activity_precision"
        ].map(pct),

    "Activity Recall/Acc (%)":
        segment[
            "activity_recall"
        ].map(pct),

    "Activity F1 (%)":
        segment[
            "activity_f1"
        ].map(pct),

    "Falling Precision (%)":
        segment[
            "fall_precision"
        ].map(pct),

    "Falling Recall/Acc (%)":
        segment[
            "fall_recall"
        ].map(pct),

    "Falling F1 (%)":
        segment[
            "fall_f1"
        ].map(pct),

    "TN":
        segment[
            "tn"
        ].astype(int),

    "FP":
        segment[
            "fp"
        ].astype(int),

    "FN":
        segment[
            "fn"
        ].astype(int),

    "TP":
        segment[
            "tp"
        ].astype(int),
})

segment_table.to_csv(
    POST
    / "SEGMENT_LEVEL_POOLED.csv",
    index=False,
)


# --------------------------------------------------------------------------------------
# POOLED EVENT TABLE
# --------------------------------------------------------------------------------------

event = pooled[
    pooled["level"] == "Event"
].copy()

event["order"] = (
    event["condition"]
    .map(order)
)

event = event.sort_values(
    "order"
)

event_rows = []

for r in event.itertuples():

    sim_only = (
        r.condition
        == "EXP03_SIM_ONLY_FULL"
    )

    row = {
        "Condition":
            r.condition_display,

        "Evaluation":
            r.evaluation_domain,

        "N trials/events":
            int(r.n),

        "Overall Acc (%)":
            pct(
                r.overall_accuracy
            ),

        "Balanced Acc (%)":
            (
                "N/A"
                if sim_only
                else pct(
                    r.balanced_accuracy
                )
            ),

        "Activity Precision (%)":
            (
                "N/A"
                if sim_only
                else pct(
                    r.activity_precision
                )
            ),

        "Activity Recall/Specificity (%)":
            (
                "N/A"
                if sim_only
                else pct(
                    r.activity_recall
                )
            ),

        "Activity F1 (%)":
            (
                "N/A"
                if sim_only
                else pct(
                    r.activity_f1
                )
            ),

        "Falling Precision (%)":
            pct(
                r.fall_precision
            ),

        "Falling Recall / Detection (%)":
            pct(
                r.fall_recall
            ),

        "Falling F1 (%)":
            pct(
                r.fall_f1
            ),

        "TN":
            int(r.tn),

        "FP":
            int(r.fp),

        "FN":
            int(r.fn),

        "TP":
            int(r.tp),
    }

    event_rows.append(
        row
    )


event_table = pd.DataFrame(
    event_rows
)

event_table.to_csv(
    POST
    / "EVENT_LEVEL_POOLED.csv",
    index=False,
)


# --------------------------------------------------------------------------------------
# SEGMENT mean ± SD across five folds
# --------------------------------------------------------------------------------------

seg_stats = stats[
    stats["level"] == "Segment"
].copy()

seg_stats["order"] = (
    seg_stats[
        "condition"
    ].map(order)
)

seg_stats = seg_stats.sort_values(
    "order"
)

seg_mean_table = pd.DataFrame({
    "Condition":
        seg_stats[
            "condition_display"
        ],

    "Overall Acc (%)":
        [
            mean_sd(m, s)
            for m, s in zip(
                seg_stats[
                    "overall_accuracy_mean"
                ],
                seg_stats[
                    "overall_accuracy_std"
                ],
            )
        ],

    "Activity Precision (%)":
        [
            mean_sd(m, s)
            for m, s in zip(
                seg_stats[
                    "activity_precision_mean"
                ],
                seg_stats[
                    "activity_precision_std"
                ],
            )
        ],

    "Activity Recall (%)":
        [
            mean_sd(m, s)
            for m, s in zip(
                seg_stats[
                    "activity_recall_mean"
                ],
                seg_stats[
                    "activity_recall_std"
                ],
            )
        ],

    "Activity F1 (%)":
        [
            mean_sd(m, s)
            for m, s in zip(
                seg_stats[
                    "activity_f1_mean"
                ],
                seg_stats[
                    "activity_f1_std"
                ],
            )
        ],

    "Falling Precision (%)":
        [
            mean_sd(m, s)
            for m, s in zip(
                seg_stats[
                    "fall_precision_mean"
                ],
                seg_stats[
                    "fall_precision_std"
                ],
            )
        ],

    "Falling Recall (%)":
        [
            mean_sd(m, s)
            for m, s in zip(
                seg_stats[
                    "fall_recall_mean"
                ],
                seg_stats[
                    "fall_recall_std"
                ],
            )
        ],

    "Falling F1 (%)":
        [
            mean_sd(m, s)
            for m, s in zip(
                seg_stats[
                    "fall_f1_mean"
                ],
                seg_stats[
                    "fall_f1_std"
                ],
            )
        ],
})

seg_mean_table.to_csv(
    POST
    / "SEGMENT_LEVEL_MEAN_STD.csv",
    index=False,
)


# --------------------------------------------------------------------------------------
# EVENT mean ± SD across five folds
# --------------------------------------------------------------------------------------

event_stats = stats[
    stats["level"] == "Event"
].copy()

event_stats["order"] = (
    event_stats[
        "condition"
    ].map(order)
)

event_stats = event_stats.sort_values(
    "order"
)

event_mean_rows = []

for r in event_stats.itertuples():

    sim_only = (
        r.condition
        == "EXP03_SIM_ONLY_FULL"
    )

    event_mean_rows.append({
        "Condition":
            r.condition_display,

        "Overall Acc (%)":
            mean_sd(
                r.overall_accuracy_mean,
                r.overall_accuracy_std,
            ),

        "Activity Precision (%)":
            (
                "N/A"
                if sim_only
                else mean_sd(
                    r.activity_precision_mean,
                    r.activity_precision_std,
                )
            ),

        "Activity Recall/Specificity (%)":
            (
                "N/A"
                if sim_only
                else mean_sd(
                    r.activity_recall_mean,
                    r.activity_recall_std,
                )
            ),

        "Activity F1 (%)":
            (
                "N/A"
                if sim_only
                else mean_sd(
                    r.activity_f1_mean,
                    r.activity_f1_std,
                )
            ),

        "Falling Precision (%)":
            mean_sd(
                r.fall_precision_mean,
                r.fall_precision_std,
            ),

        "Falling Recall/Detection (%)":
            mean_sd(
                r.fall_recall_mean,
                r.fall_recall_std,
            ),

        "Falling F1 (%)":
            mean_sd(
                r.fall_f1_mean,
                r.fall_f1_std,
            ),
    })


event_mean_table = pd.DataFrame(
    event_mean_rows
)

event_mean_table.to_csv(
    POST
    / "EVENT_LEVEL_MEAN_STD.csv",
    index=False,
)


# ======================================================================================
# Terminal report
# ======================================================================================

SEP = "=" * 170

print()
print(SEP)
print(
    "SEGMENT LEVEL — POOLED HELD-OUT RESULTS ACROSS ALL FIVE FOLDS"
)
print(SEP)

print(
    segment_table.to_string(
        index=False,
    )
)

print()
print(SEP)
print(
    "SEGMENT LEVEL — FIVE-FOLD MEAN ± SD"
)
print(SEP)

print(
    seg_mean_table.to_string(
        index=False,
    )
)

print()
print(SEP)
print(
    "EVENT LEVEL — POOLED HELD-OUT RESULTS ACROSS ALL FIVE FOLDS"
)
print(SEP)

print(
    event_table.to_string(
        index=False,
    )
)

print()
print(SEP)
print(
    "EVENT LEVEL — FIVE-FOLD MEAN ± SD"
)
print(SEP)

print(
    event_mean_table.to_string(
        index=False,
    )
)

print()
print(SEP)
print(
    "IMPORTANT INTERPRETATION"
)
print(SEP)

print(
    """
1. REAL_ONLY, SIM_FALL_SUBSTITUTION and MIX20/50/70/100 are evaluated
   on the SAME held-out physical folds.

2. SIM_ACTIVITY + SIM_FALLING is evaluated on held-out complete
   simulated source trials.

3. Every SIM_ONLY source trial contains Falling windows, therefore its
   EVENT-level test has no true Activity events. For that condition,
   Activity-event metrics and balanced event accuracy are intentionally
   reported as N/A. Its Falling Recall is the fall-event detection rate.

4. Segment-level SIM_ONLY remains a proper binary Activity/Falling
   evaluation because each simulated trial contains both pre-onset
   Activity windows and Falling windows.

5. Pooled values concatenate/sum all held-out folds. Mean ± SD reports
   fold-to-fold variability. Both are retained because physical fold
   sizes are unequal.
"""
)

print(SEP)

report = POST / "RESULTS_REPORT.txt"

with report.open(
    "w"
) as f:

    f.write(
        "RUN_ID: "
        + RUN_ID
        + "\n\n"
    )

    f.write(
        "SEGMENT LEVEL — POOLED\n"
    )

    f.write(
        segment_table.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "SEGMENT LEVEL — MEAN ± SD\n"
    )

    f.write(
        seg_mean_table.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "EVENT LEVEL — POOLED\n"
    )

    f.write(
        event_table.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "EVENT LEVEL — MEAN ± SD\n"
    )

    f.write(
        event_mean_table.to_string(
            index=False
        )
    )

    f.write(
        "\n"
    )


(CAMPAIGN / "POSTHOC_AGGREGATION_COMPLETE").write_text(
    "COMPLETE\n"
)

(CAMPAIGN / "FINAL_CAMPAIGN_COMPLETE").write_text(
    "35 TRAINING/EVALUATION FOLDS COMPLETE; "
    "POST-HOC AGGREGATION COMPLETE\n"
)

print()
print(
    "Saved:"
)

for p in [
    POST / "ALL_FOLD_METRICS.csv",
    POST / "POOLED_ALL_RESULTS.csv",
    POST / "SEGMENT_LEVEL_POOLED.csv",
    POST / "SEGMENT_LEVEL_MEAN_STD.csv",
    POST / "EVENT_LEVEL_POOLED.csv",
    POST / "EVENT_LEVEL_MEAN_STD.csv",
    POST / "RESULTS_REPORT.txt",
]:

    print(
        " ",
        p
    )

print()
print(
    "FINAL POST-HOC RESULT GATE: PASS"
)

print(
    "NO TRAINING OR INFERENCE WAS RE-RUN."
)
