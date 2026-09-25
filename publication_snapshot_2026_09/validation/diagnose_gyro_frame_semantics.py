from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"
)

CAMPAIGN = (
    ROOT
    / "outputs/_highrate_overnight/"
      "campaign_highrate_truth_v1"
)

MANIFEST = (
    CAMPAIGN
    / "highrate_event_manifest_PUBLICATION_QC.csv"
)

OUT = (
    ROOT
    / "outputs/validation_v2/"
      "gyro_frame_semantics_diagnostic.csv"
)

RICH_TASKS = {
    28,29,30,31,32,33,34,
    37,38,39,40,41,42,43,44,
    290,291,
}


def normpath(x):
    p = Path(str(x))
    return p if p.is_absolute() else ROOT / p


def derive_segments(p):
    endings = [
        "_highrate_truth.unlabeled_backup.csv",
        "_highrate_truth.csv",
    ]

    for ending in endings:
        if p.name.endswith(ending):
            stem = p.name[:-len(ending)]
            q = p.with_name(
                stem + "_segments.csv"
            )
            if not q.exists():
                raise RuntimeError(
                    f"missing segments: {q}"
                )
            return q

    raise RuntimeError(
        f"unexpected truth filename: {p}"
    )


def quat_wxyz_to_R(q):
    q = np.asarray(q, dtype=float)

    n = np.linalg.norm(q)
    if n <= 1e-12:
        raise RuntimeError(
            "invalid quaternion"
        )

    w, x, y, z = q / n

    # Standard wxyz rotation expression.
    return np.array([
        [
            1 - 2*(y*y + z*z),
            2*(x*y - z*w),
            2*(x*z + y*w),
        ],
        [
            2*(x*y + z*w),
            1 - 2*(x*x + z*z),
            2*(y*z - x*w),
        ],
        [
            2*(x*z - y*w),
            2*(y*z + x*w),
            1 - 2*(x*x + y*y),
        ],
    ])


def interp3(t_src, xyz, t_dst):
    out = np.zeros(
        (len(t_dst), 3),
        dtype=float,
    )

    for j in range(3):
        out[:, j] = np.interp(
            t_dst,
            t_src,
            xyz[:, j],
        )

    return out


def safe_corr(a, b):
    a = np.asarray(a, float).reshape(-1)
    b = np.asarray(b, float).reshape(-1)

    good = (
        np.isfinite(a)
        & np.isfinite(b)
    )

    a = a[good]
    b = b[good]

    if (
        len(a) < 5
        or np.std(a) < 1e-12
        or np.std(b) < 1e-12
    ):
        return np.nan

    return float(
        np.corrcoef(a, b)[0, 1]
    )


def score(candidate, reference):
    d = candidate - reference

    return {
        "corr":
            safe_corr(
                candidate,
                reference,
            ),

        "rmse":
            float(
                np.sqrt(
                    np.mean(
                        d*d
                    )
                )
            ),

        "mae":
            float(
                np.mean(
                    np.abs(d)
                )
            ),

        # Rotation does not change vector magnitude.
        "mag_corr":
            safe_corr(
                np.linalg.norm(
                    candidate,
                    axis=1,
                ),
                np.linalg.norm(
                    reference,
                    axis=1,
                ),
            ),

        "mag_rmse":
            float(
                np.sqrt(
                    np.mean(
                        (
                            np.linalg.norm(
                                candidate,
                                axis=1,
                            )
                            -
                            np.linalg.norm(
                                reference,
                                axis=1,
                            )
                        ) ** 2
                    )
                )
            ),
    }


m = pd.read_csv(MANIFEST)

m = m[
    m["scenario_id"]
    .astype(int)
    .isin(RICH_TASKS)
].copy()


# Physically motivated timing tests only:
#
#   whole 30-Hz-frame offsets:
#       -2, -1, 0, +1, +2 frames
#
# plus the 450-Hz physics phase:
#       -1, 0, +1 physics substep
#
# This is diagnostic, not fitting.
shifts = sorted({
    frame / 30.0 + sub / 450.0
    for frame in range(-2, 3)
    for sub in (-1, 0, 1)
})


rows = []


for _, rec in m.iterrows():

    task = int(
        rec["scenario_id"]
    )

    age = int(
        rec["sim_age_profile"]
    )

    truth_path = normpath(
        rec["pristine_truth_source"]
    )

    seg_path = derive_segments(
        truth_path
    )

    truth = pd.read_csv(
        truth_path,
        comment="#",
    )

    seg = pd.read_csv(
        seg_path
    )


    th = pd.to_numeric(
        truth["timestamp"],
        errors="coerce",
    ).to_numpy(float)

    gh = (
        truth[
            [
                "gyro_true_x",
                "gyro_true_y",
                "gyro_true_z",
            ]
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(float)
    )


    ts = pd.to_numeric(
        seg["time"],
        errors="coerce",
    ).to_numpy(float)

    q = (
        seg[
            [
                "torso_qw",
                "torso_qx",
                "torso_qy",
                "torso_qz",
            ]
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(float)
    )

    w = (
        seg[
            [
                "torso_wx",
                "torso_wy",
                "torso_wz",
            ]
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(float)
    )


    # Three frame hypotheses.
    raw = w.copy()
    R_w = np.zeros_like(w)
    RT_w = np.zeros_like(w)

    for i in range(len(w)):
        R = quat_wxyz_to_R(
            q[i]
        )

        R_w[i] = R @ w[i]
        RT_w[i] = R.T @ w[i]


    candidates = {
        "raw_segment_omega":
            raw,

        "R_times_segment_omega":
            R_w,

        "RT_times_segment_omega":
            RT_w,
    }


    for shift in shifts:

        t_query = ts + shift

        valid = (
            (t_query >= th[0])
            & (t_query <= th[-1])
        )

        if valid.sum() < 20:
            continue

        ref = interp3(
            th,
            gh,
            t_query[valid],
        )


        for method, cand_all in candidates.items():

            cand = cand_all[
                valid
            ]

            s = score(
                cand,
                ref,
            )

            rows.append({
                "scenario_id":
                    task,

                "sim_age_profile":
                    age,

                "method":
                    method,

                "time_shift_s":
                    shift,

                "time_shift_30hz_frames":
                    shift * 30.0,

                "n":
                    int(valid.sum()),

                **s,
            })


out = pd.DataFrame(rows)

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

out.to_csv(
    OUT,
    index=False,
)


print("=" * 110)
print("GYRO FRAME / TIMING SEMANTICS DIAGNOSTIC")
print("=" * 110)

print(
    "records:",
    out[
        [
            "scenario_id",
            "sim_age_profile",
        ]
    ]
    .drop_duplicates()
    .shape[0]
)


# ------------------------------------------------------------
# Zero-shift comparison
# ------------------------------------------------------------

zero = out[
    np.isclose(
        out["time_shift_s"],
        0.0,
        atol=1e-12,
    )
].copy()


print()
print(
    "ZERO-SHIFT TRANSFORM SUMMARY"
)

print(
    zero.groupby(
        "method"
    )[
        [
            "corr",
            "rmse",
            "mag_corr",
            "mag_rmse",
        ]
    ]
    .median()
    .to_string()
)


# ------------------------------------------------------------
# Magnitude test.
#
# raw / R*w / R.T*w all have identical magnitude.
# If this is poor, orientation conversion cannot be the
# primary explanation.
# ------------------------------------------------------------

zraw = zero[
    zero["method"]
    == "raw_segment_omega"
].copy()

print()
print(
    "ZERO-SHIFT MAGNITUDE CORRELATION"
)

print(
    zraw[
        "mag_corr"
    ]
    .describe()
    .to_string()
)


# ------------------------------------------------------------
# Best transform at zero shift, per recording.
# ------------------------------------------------------------

best_zero = (
    zero.sort_values(
        "corr",
        ascending=False,
    )
    .groupby(
        [
            "scenario_id",
            "sim_age_profile",
        ],
        as_index=False,
    )
    .first()
)

print()
print(
    "BEST ZERO-SHIFT TRANSFORM COUNTS"
)

print(
    best_zero[
        "method"
    ]
    .value_counts()
    .to_string()
)

print()
print(
    "BEST ZERO-SHIFT CORRELATION"
)

print(
    best_zero[
        "corr"
    ]
    .describe()
    .to_string()
)


# ------------------------------------------------------------
# Best transform + physically plausible timing shift.
# ------------------------------------------------------------

best_all = (
    out.sort_values(
        "corr",
        ascending=False,
    )
    .groupby(
        [
            "scenario_id",
            "sim_age_profile",
        ],
        as_index=False,
    )
    .first()
)

print()
print(
    "BEST TRANSFORM + SHIFT METHOD COUNTS"
)

print(
    best_all[
        "method"
    ]
    .value_counts()
    .to_string()
)

print()
print(
    "BEST TRANSFORM + SHIFT CORRELATION"
)

print(
    best_all[
        "corr"
    ]
    .describe()
    .to_string()
)

print()
print(
    "BEST TIME SHIFT [s]"
)

print(
    best_all[
        "time_shift_s"
    ]
    .describe()
    .to_string()
)

print()
print(
    "BEST TIME SHIFT COUNTS"
)

print(
    best_all[
        "time_shift_s"
    ]
    .round(6)
    .value_counts()
    .sort_index()
    .to_string()
)


# ------------------------------------------------------------
# Magnitude-only best shift.
# This tells us whether a timing offset can explain disagreement
# without any coordinate-system assumptions.
# ------------------------------------------------------------

raw_only = out[
    out["method"]
    == "raw_segment_omega"
].copy()

best_mag = (
    raw_only.sort_values(
        "mag_corr",
        ascending=False,
    )
    .groupby(
        [
            "scenario_id",
            "sim_age_profile",
        ],
        as_index=False,
    )
    .first()
)

print()
print(
    "BEST MAGNITUDE-ONLY CORRELATION"
)

print(
    best_mag[
        "mag_corr"
    ]
    .describe()
    .to_string()
)

print()
print(
    "BEST MAGNITUDE-ONLY SHIFT COUNTS"
)

print(
    best_mag[
        "time_shift_s"
    ]
    .round(6)
    .value_counts()
    .sort_index()
    .to_string()
)


print()
print(
    "BEST RESULT BY TASK"
)

print(
    best_all.groupby(
        "scenario_id"
    )[
        [
            "corr",
            "rmse",
            "mag_corr",
            "time_shift_s",
        ]
    ]
    .median()
    .to_string()
)


print()
print(
    "Interpretation:"
)

med_mag0 = float(
    zraw["mag_corr"].median()
)

med_magbest = float(
    best_mag["mag_corr"].median()
)

med_vecbest = float(
    best_all["corr"].median()
)

print(
    f"  median magnitude corr, zero shift : {med_mag0:.6f}"
)
print(
    f"  median magnitude corr, best shift : {med_magbest:.6f}"
)
print(
    f"  median vector corr, best candidate : {med_vecbest:.6f}"
)

if med_mag0 > 0.95:
    print(
        "  -> Magnitudes already agree strongly."
    )
    print(
        "     Coordinate-frame conversion is the leading issue."
    )

elif med_magbest > 0.95:
    print(
        "  -> Magnitudes agree after a small timing shift."
    )
    print(
        "     Timing/state phase is the leading issue."
    )

else:
    print(
        "  -> Magnitude agreement itself remains weak."
    )
    print(
        "     A simple R/R.T frame conversion cannot explain the discrepancy."
    )
    print(
        "     Next inspect segment body identity / saved-state semantics."
    )


print()
print(
    "Diagnostic CSV:",
    OUT,
)

print(
    "GYRO_FRAME_SEMANTICS_DIAGNOSTIC_DONE"
)
