from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"
)

BASE = (
    ROOT
    / "outputs/validation_v2"
)

LEG = (
    BASE
    / "results_legacy_truth_rich221_publication"
    / "tables"
)

HR = (
    BASE
    / "results_highrate_truth_rich221_publication"
    / "tables"
)

OUT = (
    BASE
    / "truth_sampling_rich221_ab"
)


if OUT.exists():
    raise SystemExit(
        f"STOP: output already exists:\n{OUT}"
    )

OUT.mkdir(
    parents=True,
    exist_ok=False,
)


# ============================================================
# Distribution metrics
# ============================================================

ld = pd.read_csv(
    LEG / "task_distribution_metrics.csv"
)

hd = pd.read_csv(
    HR / "task_distribution_metrics.csv"
)


keys = [
    "real_dataset",
    "task",
    "validation_status",
    "feature",
]


d = ld.merge(
    hd,
    on=keys,
    suffixes=(
        "_legacy30",
        "_highrate450",
    ),
    validate="one_to_one",
)


d[
    "delta_nwd_highrate_minus_legacy"
] = (
    d[
        "normalized_wasserstein_real_iqr_highrate450"
    ]
    -
    d[
        "normalized_wasserstein_real_iqr_legacy30"
    ]
)


d[
    "delta_coverage_highrate_minus_legacy"
] = (
    d[
        "fraction_sim_inside_real_5_95_highrate450"
    ]
    -
    d[
        "fraction_sim_inside_real_5_95_legacy30"
    ]
)


d[
    "delta_abs_cliffs_highrate_minus_legacy"
] = (
    np.abs(
        d[
            "cliffs_delta_sim_vs_real_highrate450"
        ]
    )
    -
    np.abs(
        d[
            "cliffs_delta_sim_vs_real_legacy30"
        ]
    )
)


d.to_csv(
    OUT
    / "distribution_metric_delta.csv",
    index=False,
)


# ============================================================
# Waveform morphology
# ============================================================

lw = pd.read_csv(
    LEG
    / "task_median_waveform_similarity.csv"
)

hw = pd.read_csv(
    HR
    / "task_median_waveform_similarity.csv"
)


wkeys = [
    "real_dataset",
    "task",
    "validation_status",
]


w = lw.merge(
    hw,
    on=wkeys,
    suffixes=(
        "_legacy30",
        "_highrate450",
    ),
    validate="one_to_one",
)


wave_metrics = [
    "acc_best_lag_corr",
    "gyro_best_lag_corr",

    "acc_zrmse",
    "gyro_zrmse",

    "acc_dtw",
    "gyro_dtw",
]


for metric in wave_metrics:

    w[
        f"delta_{metric}_highrate_minus_legacy"
    ] = (
        w[
            f"{metric}_highrate450"
        ]
        -
        w[
            f"{metric}_legacy30"
        ]
    )


w.to_csv(
    OUT
    / "waveform_metric_delta.csv",
    index=False,
)


# ============================================================
# Direct paired synthetic-feature comparison
# ============================================================

lf = pd.read_csv(
    LEG
    / "features_all_trials_v2.csv"
)

hf = pd.read_csv(
    HR
    / "features_all_trials_v2.csv"
)


lf = lf[
    lf["dataset"] == "Simulated"
].copy()

hf = hf[
    hf["dataset"] == "Simulated"
].copy()


pkeys = [
    "subject_id",
    "canonical_task_id",
    "trial_id",
]


p = lf.merge(
    hf,
    on=pkeys,
    suffixes=(
        "_legacy30",
        "_highrate450",
    ),
    validate="one_to_one",
)


if len(p) != 221:
    raise RuntimeError(
        f"Expected 221 paired synthetic records, got {len(p)}"
    )


paired_features = [
    "peak_acc_g",
    "p95_acc_g",
    "rms_dynamic_acc_g",
    "min_acc_g",

    "peak_gyro_dps",
    "rms_gyro_dps",

    "peak_jerk_gps",

    "gyro_area_deg",
    "dynamic_acc_area_gs",
]


for feat in paired_features:

    p[
        f"delta_{feat}"
    ] = (
        p[
            f"{feat}_highrate450"
        ]
        -
        p[
            f"{feat}_legacy30"
        ]
    )

    den = np.abs(
        p[
            f"{feat}_legacy30"
        ]
    )

    p[
        f"relative_delta_{feat}"
    ] = np.where(
        den > 1e-12,

        p[
            f"delta_{feat}"
        ]
        / den,

        np.nan,
    )


p.to_csv(
    OUT
    / "paired_truth_feature_delta.csv",
    index=False,
)


# ============================================================
# Publication-primary empirical comparison
#
# - primary tasks only
# - exclude fall_duration_s because it is an annotation/event
#   quantity rather than an inertial sampling-fidelity metric
# ============================================================

dp = d[
    (
        d[
            "validation_status"
        ] == "primary"
    )
    &
    (
        d[
            "feature"
        ] != "fall_duration_s"
    )
].copy()


wp = w[
    w[
        "validation_status"
    ] == "primary"
].copy()


summary_rows = []


for real_dataset in [
    "UniVrFall",
    "KFall",
]:

    x = dp[
        dp[
            "real_dataset"
        ] == real_dataset
    ].copy()

    y = wp[
        wp[
            "real_dataset"
        ] == real_dataset
    ].copy()


    summary_rows.append({
        "real_dataset":
            real_dataset,

        "primary_tasks_distribution":
            x[
                "task"
            ].nunique(),

        "distribution_rows":
            len(x),

        "legacy30_median_nwd":
            x[
                "normalized_wasserstein_real_iqr_legacy30"
            ].median(),

        "highrate450_median_nwd":
            x[
                "normalized_wasserstein_real_iqr_highrate450"
            ].median(),

        "median_nwd_delta":
            x[
                "delta_nwd_highrate_minus_legacy"
            ].median(),

        "fraction_nwd_rows_improved":
            float(
                (
                    x[
                        "delta_nwd_highrate_minus_legacy"
                    ] < 0
                ).mean()
            ),

        "legacy30_mean_coverage":
            x[
                "fraction_sim_inside_real_5_95_legacy30"
            ].mean(),

        "highrate450_mean_coverage":
            x[
                "fraction_sim_inside_real_5_95_highrate450"
            ].mean(),

        "mean_coverage_delta":
            x[
                "delta_coverage_highrate_minus_legacy"
            ].mean(),

        "primary_waveform_tasks":
            y[
                "task"
            ].nunique(),

        "legacy30_median_acc_corr":
            y[
                "acc_best_lag_corr_legacy30"
            ].median(),

        "highrate450_median_acc_corr":
            y[
                "acc_best_lag_corr_highrate450"
            ].median(),

        "median_acc_corr_delta":
            y[
                "delta_acc_best_lag_corr_highrate_minus_legacy"
            ].median(),

        "legacy30_median_gyro_corr":
            y[
                "gyro_best_lag_corr_legacy30"
            ].median(),

        "highrate450_median_gyro_corr":
            y[
                "gyro_best_lag_corr_highrate450"
            ].median(),

        "median_gyro_corr_delta":
            y[
                "delta_gyro_best_lag_corr_highrate_minus_legacy"
            ].median(),

        "legacy30_median_acc_dtw":
            y[
                "acc_dtw_legacy30"
            ].median(),

        "highrate450_median_acc_dtw":
            y[
                "acc_dtw_highrate450"
            ].median(),

        "median_acc_dtw_delta":
            y[
                "delta_acc_dtw_highrate_minus_legacy"
            ].median(),

        "legacy30_median_gyro_dtw":
            y[
                "gyro_dtw_legacy30"
            ].median(),

        "highrate450_median_gyro_dtw":
            y[
                "gyro_dtw_highrate450"
            ].median(),

        "median_gyro_dtw_delta":
            y[
                "delta_gyro_dtw_highrate_minus_legacy"
            ].median(),
    })


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUT
    / "truth_sampling_primary_summary.csv",
    index=False,
)


# ============================================================
# Direct sampling effect summary
# ============================================================

feature_rows = []


for feat in paired_features:

    delta = p[
        f"delta_{feat}"
    ]

    rel = (
        p[
            f"relative_delta_{feat}"
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )


    feature_rows.append({
        "feature":
            feat,

        "n":
            int(
                delta.notna().sum()
            ),

        "legacy30_median":
            p[
                f"{feat}_legacy30"
            ].median(),

        "highrate450_median":
            p[
                f"{feat}_highrate450"
            ].median(),

        "median_absolute_delta":
            delta.median(),

        "median_relative_delta":
            rel.median(),

        "q25_relative_delta":
            rel.quantile(
                0.25
            ),

        "q75_relative_delta":
            rel.quantile(
                0.75
            ),
    })


fs = pd.DataFrame(
    feature_rows
)

fs.to_csv(
    OUT
    / "truth_sampling_feature_summary.csv",
    index=False,
)


# ============================================================
# Task-level sampling effects
# ============================================================

task_rows = []


for task, g in p.groupby(
    "canonical_task_id"
):

    row = {
        "task":
            int(task),

        "n_profiles":
            len(g),
    }

    for feat in paired_features:

        rel = (
            g[
                f"relative_delta_{feat}"
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

        row[
            f"median_relative_delta_{feat}"
        ] = rel.median()

    task_rows.append(
        row
    )


pd.DataFrame(
    task_rows
).to_csv(
    OUT
    / "truth_sampling_effect_by_task.csv",
    index=False,
)


# ============================================================
# Console summary
# ============================================================

print("=" * 118)
print("TRUTH-ONLY SAMPLING A/B: 30-Hz+INTERP vs 450-Hz+AA")
print("=" * 118)


print()
print(
    "Paired synthetic records:",
    len(p),
)

print(
    "Semantic tasks:",
    p[
        "canonical_task_id"
    ].nunique(),
)

print(
    "Age profiles:",
    p[
        "subject_id"
    ].nunique(),
)


print()
print("PRIMARY REAL-vs-SIM SUMMARY")

print(
    summary.to_string(
        index=False
    )
)


print()
print(
    "DIRECT TRUTH-SAMPLING FEATURE CHANGE"
)

print(
    fs.to_string(
        index=False
    )
)


print()
print(
    "NWD improvement counts "
    "(negative delta = 450-Hz truth closer to physical data)"
)

print(
    dp.groupby(
        "real_dataset"
    )[
        "delta_nwd_highrate_minus_legacy"
    ]
    .apply(
        lambda x:
        f"{int((x < 0).sum())}/{len(x)} "
        f"({(x < 0).mean():.1%})"
    )
    .to_string()
)


print()
print(
    "Waveform correlation improvement counts"
)

for ds in [
    "UniVrFall",
    "KFall",
]:

    q = wp[
        wp[
            "real_dataset"
        ] == ds
    ]

    print(
        ds
    )

    print(
        "  AccMag :",
        f"{int((q['delta_acc_best_lag_corr_highrate_minus_legacy'] > 0).sum())}"
        f"/{len(q)}"
    )

    print(
        "  GyroMag:",
        f"{int((q['delta_gyro_best_lag_corr_highrate_minus_legacy'] > 0).sum())}"
        f"/{len(q)}"
    )


print()
print(
    "TRUTH_SAMPLING_RICH221_AB_OK"
)

print(
    "Output:",
    OUT,
)
