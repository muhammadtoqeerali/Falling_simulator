from pathlib import Path
from collections import defaultdict
import math
import numpy as np
import pandas as pd

try:
    from scipy.stats import wasserstein_distance
except ImportError:
    raise SystemExit("scipy is required: pip install scipy")

# ============================================================
# CONFIGURATION
# ============================================================

FS = 100.0

# Signal window used for sensor-level comparison.
# This is NOT total task/recording duration.
PRE_S  = 0.50
POST_S = 2.00

PRE_N  = int(PRE_S * FS)
POST_N = int(POST_S * FS)

BOOTSTRAPS = 500
RNG = np.random.default_rng(20260817)

ROOT = Path("/mnt/hdd16T/ToqeerHomeBackup/mujoco_project")

REAL_INV = ROOT / "outputs/validation_v2/real_inventory_canonical.csv"

SIM_REPORT = (
    ROOT /
    "outputs/simulated_dataset_all_subjects/"
    "conversion_reports/conversion_report_all_subjects.csv"
)

OUT = ROOT / "outputs/validation_v2/results_core"
TABLES = OUT / "tables"

TABLES.mkdir(parents=True, exist_ok=True)

# Tasks 41/42 are calculated, but kept out of primary aggregate
# until stairs-vs-ladder protocol equivalence is confirmed.
PROTOCOL_REVIEW = {41, 42}

FEATURES = [
    "fall_duration_s",
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

DISCRIM_FEATURES = [
    "peak_acc_g",
    "rms_dynamic_acc_g",
    "peak_gyro_dps",
    "rms_gyro_dps",
    "peak_jerk_gps",
    "gyro_area_deg",
    "dynamic_acc_area_gs",
]


# ============================================================
# HELPERS
# ============================================================

def numeric(df, col):
    return pd.to_numeric(df[col], errors="coerce").to_numpy(float)


def read_sensor(path):
    df = pd.read_csv(path)

    req = [
        "AccX", "AccY", "AccZ",
        "GyrX", "GyrY", "GyrZ",
    ]

    missing = [c for c in req if c not in df.columns]

    if missing:
        raise RuntimeError(
            f"{path}: missing columns {missing}"
        )

    # Physical + converted synthetic convention:
    # acceleration stored in mg
    # gyro stored in mdps
    ax = numeric(df, "AccX") / 1000.0
    ay = numeric(df, "AccY") / 1000.0
    az = numeric(df, "AccZ") / 1000.0

    gx = numeric(df, "GyrX") / 1000.0
    gy = numeric(df, "GyrY") / 1000.0
    gz = numeric(df, "GyrZ") / 1000.0

    amag = np.sqrt(ax*ax + ay*ay + az*az)
    gmag = np.sqrt(gx*gx + gy*gy + gz*gz)

    return amag, gmag


def finite(x):
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


def fixed_curve(signal, onset):
    """
    Fixed time axis:
       onset-0.5s ... onset ... onset+2.0s

    No time warping here.
    """
    rel = np.arange(-PRE_N, POST_N + 1)
    pos = onset + rel

    out = np.full(len(rel), np.nan, float)

    good = (pos >= 0) & (pos < len(signal))
    out[good] = signal[pos[good]]

    return out


def extract_features(amag, gmag, onset, impact):

    n = len(amag)

    if not (0 <= onset < n):
        return None

    pre0 = max(0, onset - PRE_N)
    event1 = min(n, onset + POST_N + 1)

    pre_acc = finite(amag[pre0:onset])
    pre_gyr = finite(gmag[pre0:onset])

    ev_acc = finite(amag[onset:event1])
    ev_gyr = finite(gmag[onset:event1])

    if len(ev_acc) < 3 or len(ev_gyr) < 3:
        return None

    jerk = np.abs(np.diff(ev_acc)) * FS

    # We do NOT assume all trials have the same duration.
    if (
        impact is not None
        and np.isfinite(impact)
        and impact > onset
        and impact < n
    ):
        duration = (impact - onset) / FS
        valid_event = True
    else:
        duration = np.nan
        valid_event = False

    dynamic_acc = np.abs(ev_acc - 1.0)

    return {
        "valid_event_label": valid_event,

        "fall_duration_s": duration,

        "baseline_acc_g":
            np.nanmedian(pre_acc) if len(pre_acc) else np.nan,

        "baseline_gyro_dps":
            np.nanmedian(pre_gyr) if len(pre_gyr) else np.nan,

        "peak_acc_g":
            np.nanmax(ev_acc),

        "p95_acc_g":
            np.nanpercentile(ev_acc, 95),

        "rms_dynamic_acc_g":
            np.sqrt(np.nanmean((ev_acc - 1.0)**2)),

        "min_acc_g":
            np.nanmin(ev_acc),

        "peak_gyro_dps":
            np.nanmax(ev_gyr),

        "rms_gyro_dps":
            np.sqrt(np.nanmean(ev_gyr**2)),

        "peak_jerk_gps":
            np.nanmax(jerk) if len(jerk) else np.nan,

        "gyro_area_deg":
            np.trapezoid(ev_gyr, dx=1.0/FS),

        "dynamic_acc_area_gs":
            np.trapezoid(dynamic_acc, dx=1.0/FS),
    }


def cliffs_delta(sim, real):
    sim = finite(sim)
    real = finite(real)

    if len(sim) == 0 or len(real) == 0:
        return np.nan

    d = sim[:, None] - real[None, :]

    return (
        np.sum(d > 0) - np.sum(d < 0)
    ) / d.size


def bootstrap_median_difference(sim, real, nboot=BOOTSTRAPS):
    sim = finite(sim)
    real = finite(real)

    if len(sim) < 2 or len(real) < 2:
        return np.nan, np.nan

    vals = np.empty(nboot, float)

    for i in range(nboot):

        s = RNG.choice(
            sim, size=len(sim), replace=True
        )

        r = RNG.choice(
            real, size=len(real), replace=True
        )

        vals[i] = np.median(s) - np.median(r)

    return (
        np.percentile(vals, 2.5),
        np.percentile(vals, 97.5)
    )


def zcurve(x):
    x = np.asarray(x, float)

    idx = np.arange(len(x))
    good = np.isfinite(x)

    if good.sum() < 3:
        return np.full_like(x, np.nan)

    # interpolate unavailable edge samples only for morphology metric
    y = np.interp(idx, idx[good], x[good])

    sd = np.std(y)

    if sd < 1e-12:
        return y * 0

    return (y - np.mean(y)) / sd


def best_lag_corr(a, b, max_lag=50):
    a = zcurve(a)
    b = zcurve(b)

    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return np.nan, np.nan

    best_corr = -np.inf
    best_lag = 0

    for lag in range(-max_lag, max_lag + 1):

        if lag < 0:
            x = a[-lag:]
            y = b[:len(b)+lag]

        elif lag > 0:
            x = a[:-lag]
            y = b[lag:]

        else:
            x = a
            y = b

        if len(x) < 20:
            continue

        sx = np.std(x)
        sy = np.std(y)

        if sx < 1e-12 or sy < 1e-12:
            continue

        c = np.corrcoef(x, y)[0, 1]

        if np.isfinite(c) and c > best_corr:
            best_corr = c
            best_lag = lag

    if best_corr == -np.inf:
        return np.nan, np.nan

    return best_corr, best_lag / FS


def zrmse(a, b):
    a = zcurve(a)
    b = zcurve(b)

    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return np.nan

    return np.sqrt(np.mean((a-b)**2))


def dtw_distance(a, b):
    """
    Normalized DTW on task-level median z-normalized waveform.
    """
    a = zcurve(a)
    b = zcurve(b)

    n = len(a)
    m = len(b)

    D = np.full((n+1, m+1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, n+1):
        for j in range(1, m+1):

            cost = abs(a[i-1] - b[j-1])

            D[i, j] = cost + min(
                D[i-1, j],
                D[i, j-1],
                D[i-1, j-1]
            )

    return D[n, m] / (n + m)


# ============================================================
# LOAD REAL INVENTORY
# ============================================================

real = pd.read_csv(REAL_INV)

real["fall_onset_frame"] = pd.to_numeric(
    real["fall_onset_frame"],
    errors="coerce"
)

real["fall_impact_frame"] = pd.to_numeric(
    real["fall_impact_frame"],
    errors="coerce"
)

records = []

# Curves grouped by dataset/task.
acc_curves = defaultdict(list)
gyr_curves = defaultdict(list)

load_failures = []


def process_record(
    dataset,
    subject_id,
    task,
    trial,
    description,
    sensor_csv,
    onset,
    impact,
    source_kind,
):

    try:
        onset = int(onset)
    except Exception:
        return

    try:
        impact_i = int(impact) if np.isfinite(impact) else None
    except Exception:
        impact_i = None

    try:
        amag, gmag = read_sensor(sensor_csv)
    except Exception as e:

        load_failures.append({
            "dataset": dataset,
            "subject_id": subject_id,
            "task": task,
            "trial": trial,
            "sensor_csv": str(sensor_csv),
            "error": repr(e),
        })

        return

    f = extract_features(
        amag,
        gmag,
        onset,
        impact_i
    )

    if f is None:
        return

    row = {
        "dataset": dataset,
        "source_kind": source_kind,
        "subject_id": subject_id,
        "canonical_task_id": int(task),
        "trial_id": int(trial),
        "description": description,
        "sensor_csv": str(sensor_csv),
        "fall_onset_frame": onset,
        "fall_impact_frame": impact_i,
        **f,
    }

    records.append(row)

    acc_curves[
        (dataset, int(task))
    ].append(
        fixed_curve(amag, onset)
    )

    gyr_curves[
        (dataset, int(task))
    ].append(
        fixed_curve(gmag, onset)
    )


for r in real.itertuples(index=False):

    process_record(
        dataset=r.dataset,
        subject_id=int(r.subject_id),
        task=int(r.canonical_task_id),
        trial=int(r.trial_id),
        description=str(r.description),
        sensor_csv=Path(r.sensor_csv),
        onset=r.fall_onset_frame,
        impact=r.fall_impact_frame,
        source_kind="physical",
    )


# ============================================================
# LOAD SIMULATED INVENTORY
# ============================================================

sim = pd.read_csv(SIM_REPORT)

for r in sim.itertuples(index=False):

    process_record(
        dataset="Simulated",
        # Important: this is a synthetic profile identifier;
        # current dataset builder uses AGE as SA number.
        subject_id=int(r.subject_id),
        task=int(r.task_id),
        trial=int(r.trial_id),
        description="simulated",
        sensor_csv=Path(r.output_csv),
        onset=r.fall_onset_frame,
        impact=r.fall_impact_frame,
        source_kind="synthetic_profile",
    )


features = pd.DataFrame(records)

features.to_csv(
    TABLES / "features_all_trials_v2.csv",
    index=False
)

if load_failures:
    pd.DataFrame(load_failures).to_csv(
        TABLES / "load_failures.csv",
        index=False
    )


# ============================================================
# QC TABLE
# ============================================================

qc = (
    features.groupby("dataset")
    .agg(
        records=("canonical_task_id", "size"),
        subjects=("subject_id", "nunique"),
        tasks=("canonical_task_id", "nunique"),
        valid_event_labels=("valid_event_label", "sum"),
    )
    .reset_index()
)

qc["invalid_event_labels"] = (
    qc["records"] - qc["valid_event_labels"]
)

qc.to_csv(
    TABLES / "dataset_qc_summary.csv",
    index=False
)


# ============================================================
# TASK / FEATURE SUMMARY
# ============================================================

summary_rows = []

for (dataset, task), g in features.groupby(
    ["dataset", "canonical_task_id"]
):

    for feat in FEATURES:

        vals = finite(g[feat])

        if len(vals) == 0:
            continue

        summary_rows.append({
            "dataset": dataset,
            "task": int(task),
            "feature": feat,
            "n": len(vals),
            "mean": np.mean(vals),
            "std": np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
            "median": np.median(vals),
            "q25": np.percentile(vals, 25),
            "q75": np.percentile(vals, 75),
            "min": np.min(vals),
            "max": np.max(vals),
        })

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    TABLES / "task_feature_summary.csv",
    index=False
)


# ============================================================
# REAL VS SIM DISTRIBUTION COMPARISON
# ============================================================

comparison_rows = []

real_sets = ["UniVrFall", "KFall"]

sim_tasks = set(
    features[
        features.dataset == "Simulated"
    ].canonical_task_id.unique()
)

for real_dataset in real_sets:

    real_tasks = set(
        features[
            features.dataset == real_dataset
        ].canonical_task_id.unique()
    )

    common = sorted(real_tasks & sim_tasks)

    for task in common:

        gr = features[
            (features.dataset == real_dataset) &
            (features.canonical_task_id == task)
        ]

        gs = features[
            (features.dataset == "Simulated") &
            (features.canonical_task_id == task)
        ]

        status = (
            "protocol_review"
            if real_dataset == "UniVrFall"
            and task in PROTOCOL_REVIEW
            else "primary"
        )

        for feat in FEATURES:

            rv = finite(gr[feat])
            sv = finite(gs[feat])

            if len(rv) == 0 or len(sv) == 0:
                continue

            rmed = np.median(rv)
            smed = np.median(sv)

            rq25 = np.percentile(rv, 25)
            rq75 = np.percentile(rv, 75)
            riqr = rq75 - rq25

            if riqr <= 1e-12:
                scale = np.std(rv)
            else:
                scale = riqr

            wd = wasserstein_distance(rv, sv)

            nwd = (
                wd / scale
                if scale > 1e-12
                else np.nan
            )

            r05, r95 = np.percentile(
                rv, [5, 95]
            )

            coverage = np.mean(
                (sv >= r05) &
                (sv <= r95)
            )

            lo, hi = bootstrap_median_difference(
                sv,
                rv
            )

            comparison_rows.append({
                "real_dataset": real_dataset,
                "task": task,
                "validation_status": status,
                "feature": feat,

                "n_real": len(rv),
                "n_sim": len(sv),

                "real_median": rmed,
                "sim_median": smed,

                "median_difference_sim_minus_real":
                    smed - rmed,

                "median_ratio_sim_over_real":
                    (
                        smed / rmed
                        if abs(rmed) > 1e-12
                        else np.nan
                    ),

                "bootstrap_median_diff_ci_low": lo,
                "bootstrap_median_diff_ci_high": hi,

                "wasserstein": wd,
                "normalized_wasserstein_real_iqr": nwd,

                "cliffs_delta_sim_vs_real":
                    cliffs_delta(sv, rv),

                "real_p05": r05,
                "real_p95": r95,

                "fraction_sim_inside_real_5_95":
                    coverage,
            })


comparison = pd.DataFrame(comparison_rows)

comparison.to_csv(
    TABLES / "task_distribution_metrics.csv",
    index=False
)


# ============================================================
# TASK-LEVEL MEDIAN WAVEFORM MORPHOLOGY
# ============================================================

wave_rows = []

for real_dataset in real_sets:

    real_tasks = {
        k[1]
        for k in acc_curves
        if k[0] == real_dataset
    }

    sim_wave_tasks = {
        k[1]
        for k in acc_curves
        if k[0] == "Simulated"
    }

    common = sorted(
        real_tasks & sim_wave_tasks
    )

    for task in common:

        status = (
            "protocol_review"
            if real_dataset == "UniVrFall"
            and task in PROTOCOL_REVIEW
            else "primary"
        )

        ra = np.nanmedian(
            np.vstack(
                acc_curves[
                    (real_dataset, task)
                ]
            ),
            axis=0
        )

        sa = np.nanmedian(
            np.vstack(
                acc_curves[
                    ("Simulated", task)
                ]
            ),
            axis=0
        )

        rg = np.nanmedian(
            np.vstack(
                gyr_curves[
                    (real_dataset, task)
                ]
            ),
            axis=0
        )

        sg = np.nanmedian(
            np.vstack(
                gyr_curves[
                    ("Simulated", task)
                ]
            ),
            axis=0
        )

        acorr, alag = best_lag_corr(
            ra, sa
        )

        gcorr, glag = best_lag_corr(
            rg, sg
        )

        wave_rows.append({
            "real_dataset": real_dataset,
            "task": task,
            "validation_status": status,

            "n_real":
                len(acc_curves[
                    (real_dataset, task)
                ]),

            "n_sim":
                len(acc_curves[
                    ("Simulated", task)
                ]),

            "acc_best_lag_corr": acorr,
            "acc_best_lag_s": alag,
            "acc_zrmse": zrmse(ra, sa),
            "acc_dtw": dtw_distance(ra, sa),

            "gyro_best_lag_corr": gcorr,
            "gyro_best_lag_s": glag,
            "gyro_zrmse": zrmse(rg, sg),
            "gyro_dtw": dtw_distance(rg, sg),
        })


wave = pd.DataFrame(wave_rows)

wave.to_csv(
    TABLES / "task_median_waveform_similarity.csv",
    index=False
)


# ============================================================
# MATCHED-TASK VS WRONG-TASK DISCRIMINABILITY
# ============================================================

disc_rows = []
disc_summary = []

for real_dataset in real_sets:

    real_df = features[
        features.dataset == real_dataset
    ].copy()

    sim_df = features[
        features.dataset == "Simulated"
    ].copy()

    common_tasks = sorted(
        set(real_df.canonical_task_id) &
        set(sim_df.canonical_task_id)
    )

    # For primary discriminability, omit protocol-review tasks.
    if real_dataset == "UniVrFall":
        common_tasks = [
            t for t in common_tasks
            if t not in PROTOCOL_REVIEW
        ]

    real_df = real_df[
        real_df.canonical_task_id.isin(
            common_tasks
        )
    ]

    sim_df = sim_df[
        sim_df.canonical_task_id.isin(
            common_tasks
        )
    ]

    # Positive inertial features are log transformed to reduce
    # domination by large impact values.
    real_mat = np.log1p(
        real_df[DISCRIM_FEATURES]
        .astype(float)
        .to_numpy()
    )

    pooled_scale = np.nanstd(
        real_mat,
        axis=0
    )

    pooled_scale[
        pooled_scale < 1e-9
    ] = 1.0

    centroids = {}

    for task in common_tasks:

        g = real_df[
            real_df.canonical_task_id == task
        ]

        m = np.log1p(
            g[DISCRIM_FEATURES]
            .astype(float)
            .to_numpy()
        )

        centroids[task] = (
            np.nanmedian(m, axis=0)
        )

    for r in sim_df.itertuples(index=False):

        raw = np.array(
            [
                getattr(r, f)
                for f in DISCRIM_FEATURES
            ],
            float
        )

        if not np.isfinite(raw).all():
            continue

        v = np.log1p(raw)

        distances = {}

        for task, c in centroids.items():

            distances[task] = np.sqrt(
                np.mean(
                    ((v-c)/pooled_scale)**2
                )
            )

        ranked = sorted(
            distances,
            key=distances.get
        )

        true_task = int(
            r.canonical_task_id
        )

        if true_task not in ranked:
            continue

        rank = ranked.index(
            true_task
        ) + 1

        disc_rows.append({
            "real_dataset": real_dataset,
            "sim_profile_id": r.subject_id,
            "sim_trial_id": r.trial_id,
            "true_task": true_task,
            "nearest_real_task": ranked[0],
            "true_task_rank": rank,
            "true_task_distance":
                distances[true_task],
            "nearest_task_distance":
                distances[ranked[0]],
            "top1_correct":
                rank == 1,
            "top3_correct":
                rank <= 3,
        })

    temp = pd.DataFrame(
        [
            x for x in disc_rows
            if x["real_dataset"] == real_dataset
        ]
    )

    if len(temp):

        disc_summary.append({
            "real_dataset": real_dataset,
            "n_sim_trials": len(temp),
            "top1_accuracy":
                temp.top1_correct.mean(),
            "top3_accuracy":
                temp.top3_correct.mean(),
            "median_true_task_rank":
                temp.true_task_rank.median(),
            "mean_true_task_rank":
                temp.true_task_rank.mean(),
        })


pd.DataFrame(disc_rows).to_csv(
    TABLES / "matched_vs_wrong_task_trials.csv",
    index=False
)

pd.DataFrame(disc_summary).to_csv(
    TABLES / "matched_vs_wrong_task_summary.csv",
    index=False
)


# ============================================================
# SIMULATOR-ONLY EXTENSION SUMMARY
# ============================================================

extensions = features[
    (features.dataset == "Simulated") &
    (~features.canonical_task_id.isin(
        set(
            features[
                features.dataset == "UniVrFall"
            ].canonical_task_id
        )
    ))
]

extensions.to_csv(
    TABLES / "simulator_only_extension_trials.csv",
    index=False
)


# ============================================================
# PRINT COMPACT SUMMARY
# ============================================================

print("\n" + "="*100)
print("VALIDATION V2 CORE COMPLETE")
print("="*100)

print("\nDataset QC:")
print(qc.to_string(index=False))

print("\nPhysical comparison scope:")
print(
    "KFall     : Tasks 20-34"
)

print(
    "UniVrFall : Tasks 20-40 primary; "
    "41-42 calculated but protocol_review"
)

print(
    "Simulator-only extensions remain outside "
    "empirical fidelity claims."
)

print("\nPrimary distribution summary:")

p = comparison[
    comparison.validation_status == "primary"
]

for ds in ["UniVrFall", "KFall"]:

    q = p[
        p.real_dataset == ds
    ]

    if len(q) == 0:
        continue

    print(f"\n{ds}")

    print(
        " median normalized Wasserstein =",
        round(
            q.normalized_wasserstein_real_iqr
             .median(),
            3
        )
    )

    print(
        " median fraction synthetic inside "
        "real 5-95% interval =",
        round(
            q.fraction_sim_inside_real_5_95
             .median(),
            3
        )
    )


print("\nTask-median waveform summary:")

wp = wave[
    wave.validation_status == "primary"
]

for ds in ["UniVrFall", "KFall"]:

    q = wp[
        wp.real_dataset == ds
    ]

    if len(q) == 0:
        continue

    print(f"\n{ds}")

    print(
        " median AccMag lag-correlation =",
        round(
            q.acc_best_lag_corr.median(),
            3
        )
    )

    print(
        " median GyroMag lag-correlation =",
        round(
            q.gyro_best_lag_corr.median(),
            3
        )
    )

    print(
        " median AccMag DTW =",
        round(
            q.acc_dtw.median(),
            3
        )
    )

    print(
        " median GyroMag DTW =",
        round(
            q.gyro_dtw.median(),
            3
        )
    )


print("\nMatched-task vs wrong-task:")
dsum = pd.DataFrame(disc_summary)

if len(dsum):
    print(
        dsum.round(3).to_string(
            index=False
        )
    )


print("\nTables written to:")
print(TABLES)

print("\nImportant:")
print(
    "- No simulator amplitudes were scaled."
)
print(
    "- No nearest real trial was cherry-picked."
)
print(
    "- All available KFall repetitions were retained."
)
print(
    "- Total recording/task duration was NOT compared."
)
print(
    "- Fall duration means onset-to-impact only and is "
    "reported as a distribution, not a required match."
)
print(
    "- Sensor waveform analysis uses the same fixed "
    "onset-centered window for real and synthetic data."
)
