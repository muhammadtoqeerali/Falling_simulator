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
    / "results_paired_legacy_publication/tables"
)

HR = (
    BASE
    / "results_highrate_truth_publication/tables"
)

OUT = (
    BASE
    / "acquisition_ab_publication"
)

OUT.mkdir(
    parents=True,
    exist_ok=False,
)


# ============================================================
# Distribution metrics
# ============================================================

ld = pd.read_csv(
    LEG
    / "task_distribution_metrics.csv"
)

hd = pd.read_csv(
    HR
    / "task_distribution_metrics.csv"
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
        "_legacy",
        "_highrate",
    ),
    validate="one_to_one",
)

d[
    "delta_nwd_highrate_minus_legacy"
] = (
    d[
        "normalized_wasserstein_real_iqr_highrate"
    ]
    -
    d[
        "normalized_wasserstein_real_iqr_legacy"
    ]
)

d[
    "delta_coverage_highrate_minus_legacy"
] = (
    d[
        "fraction_sim_inside_real_5_95_highrate"
    ]
    -
    d[
        "fraction_sim_inside_real_5_95_legacy"
    ]
)

d[
    "delta_abs_cliffs_highrate_minus_legacy"
] = (
    np.abs(
        d[
            "cliffs_delta_sim_vs_real_highrate"
        ]
    )
    -
    np.abs(
        d[
            "cliffs_delta_sim_vs_real_legacy"
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
        "_legacy",
        "_highrate",
    ),
    validate="one_to_one",
)

for metric in [
    "acc_best_lag_corr",
    "gyro_best_lag_corr",
    "acc_zrmse",
    "gyro_zrmse",
    "acc_dtw",
    "gyro_dtw",
]:
    w[
        f"delta_{metric}_highrate_minus_legacy"
    ] = (
        w[
            f"{metric}_highrate"
        ]
        -
        w[
            f"{metric}_legacy"
        ]
    )

w.to_csv(
    OUT
    / "waveform_metric_delta.csv",
    index=False,
)


# ============================================================
# Direct paired synthetic feature change
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
    lf[
        "dataset"
    ] == "Simulated"
].copy()

hf = hf[
    hf[
        "dataset"
    ] == "Simulated"
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
        "_legacy",
        "_highrate",
    ),
    validate="one_to_one",
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
            f"{feat}_highrate"
        ]
        -
        p[
            f"{feat}_legacy"
        ]
    )

    den = np.abs(
        p[
            f"{feat}_legacy"
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
    / "paired_synthetic_feature_delta.csv",
    index=False,
)


# ============================================================
# Publication-primary aggregates.
#
# Exclude:
# - protocol-review rows
# - fall_duration_s from acquisition-quality aggregate
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
    ]

    y = wp[
        wp[
            "real_dataset"
        ] == real_dataset
    ]

    summary_rows.append({
        "real_dataset":
            real_dataset,

        "distribution_rows":
            len(x),

        "legacy_median_nwd":
            x[
                "normalized_wasserstein_real_iqr_legacy"
            ].median(),

        "highrate_median_nwd":
            x[
                "normalized_wasserstein_real_iqr_highrate"
            ].median(),

        "median_nwd_delta":
            x[
                "delta_nwd_highrate_minus_legacy"
            ].median(),

        "fraction_distribution_rows_nwd_improved":
            float(
                (
                    x[
                        "delta_nwd_highrate_minus_legacy"
                    ] < 0
                ).mean()
            ),

        "legacy_mean_coverage":
            x[
                "fraction_sim_inside_real_5_95_legacy"
            ].mean(),

        "highrate_mean_coverage":
            x[
                "fraction_sim_inside_real_5_95_highrate"
            ].mean(),

        "mean_coverage_delta":
            x[
                "delta_coverage_highrate_minus_legacy"
            ].mean(),

        "waveform_tasks":
            len(y),

        "legacy_median_acc_corr":
            y[
                "acc_best_lag_corr_legacy"
            ].median(),

        "highrate_median_acc_corr":
            y[
                "acc_best_lag_corr_highrate"
            ].median(),

        "median_acc_corr_delta":
            y[
                "delta_acc_best_lag_corr_highrate_minus_legacy"
            ].median(),

        "legacy_median_gyro_corr":
            y[
                "gyro_best_lag_corr_legacy"
            ].median(),

        "highrate_median_gyro_corr":
            y[
                "gyro_best_lag_corr_highrate"
            ].median(),

        "median_gyro_corr_delta":
            y[
                "delta_gyro_best_lag_corr_highrate_minus_legacy"
            ].median(),

        "legacy_median_acc_dtw":
            y[
                "acc_dtw_legacy"
            ].median(),

        "highrate_median_acc_dtw":
            y[
                "acc_dtw_highrate"
            ].median(),

        "median_acc_dtw_delta":
            y[
                "delta_acc_dtw_highrate_minus_legacy"
            ].median(),

        "legacy_median_gyro_dtw":
            y[
                "gyro_dtw_legacy"
            ].median(),

        "highrate_median_gyro_dtw":
            y[
                "gyro_dtw_highrate"
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
    / "acquisition_primary_summary.csv",
    index=False,
)


# ============================================================
# Direct acquisition feature summary
# ============================================================

feature_rows = []

for feat in paired_features:

    delta = p[
        f"delta_{feat}"
    ]

    rel = p[
        f"relative_delta_{feat}"
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    feature_rows.append({
        "feature":
            feat,

        "n":
            int(
                delta.notna().sum()
            ),

        "legacy_median":
            p[
                f"{feat}_legacy"
            ].median(),

        "highrate_median":
            p[
                f"{feat}_highrate"
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
    / "paired_feature_summary.csv",
    index=False,
)


print("=" * 110)
print("LEGACY 30->100 vs EXACT 450->100 ACQUISITION")
print("=" * 110)

print()
print("PRIMARY REAL-vs-SIM SUMMARY")
print(
    summary.to_string(
        index=False
    )
)

print()
print("DIRECT PAIRED SYNTHETIC FEATURE CHANGE")
print(
    fs.to_string(
        index=False
    )
)

print()
print(
    "Rows where normalized Wasserstein improved "
    "(negative delta = high-rate closer):"
)

print(
    dp.groupby(
        "real_dataset"
    )[
        "delta_nwd_highrate_minus_legacy"
    ]
    .apply(
        lambda x:
        f"{(x < 0).sum()}/{len(x)} "
        f"({(x < 0).mean():.1%})"
    )
    .to_string()
)

print()
print("ACQUISITION_AB_COMPARISON_OK")
print("Output:", OUT)
