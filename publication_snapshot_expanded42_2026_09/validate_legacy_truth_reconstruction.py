from pathlib import Path
import math

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
      "legacy_truth_reconstruction_audit.csv"
)


RICH_TASKS = {
    28,29,30,31,32,33,34,
    37,38,39,40,41,42,43,44,
    290,291,
}


def norm(x):
    p = Path(str(x))
    return p if p.is_absolute() else ROOT / p


def pristine_read(path):
    return pd.read_csv(
        path,
        comment="#",
    )


def quat_wxyz_to_rot(q):
    """
    MuJoCo xquat convention: [w, x, y, z].
    Returns body-local -> world rotation matrix.
    """

    q = np.asarray(
        q,
        dtype=float,
    )

    n = np.linalg.norm(q)

    if n <= 1e-12:
        raise RuntimeError(
            "invalid quaternion"
        )

    w, x, y, z = q / n

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


def derive_segments(
    highrate_path,
):
    # Pristine truth is either:
    #   *_highrate_truth.csv
    # for untouched files, or
    #   *_highrate_truth.unlabeled_backup.csv
    # when the post-run labeler rewrote the working truth CSV.
    endings = [
        "_highrate_truth.unlabeled_backup.csv",
        "_highrate_truth.csv",
    ]

    stem = None

    for ending in endings:
        if highrate_path.name.endswith(ending):
            stem = highrate_path.name[:-len(ending)]
            break

    if stem is None:
        raise RuntimeError(
            f"unexpected truth name: {highrate_path}"
        )

    seg = highrate_path.with_name(
        stem + "_segments.csv"
    )

    if not seg.exists():
        raise RuntimeError(
            f"segments file not found: {seg}"
        )

    return seg


def interp_xyz(
    source_t,
    source_xyz,
    target_t,
):
    out = np.zeros(
        (
            len(target_t),
            3,
        ),
        dtype=float,
    )

    for k in range(3):
        out[:, k] = np.interp(
            target_t,
            source_t,
            source_xyz[:, k],
        )

    return out


m = pd.read_csv(
    MANIFEST
)

m = m[
    m[
        "scenario_id"
    ].astype(int)
    .isin(
        RICH_TASKS
    )
].copy()


rows = []


for _, r in m.iterrows():

    task = int(
        r[
            "scenario_id"
        ]
    )

    age = int(
        r[
            "sim_age_profile"
        ]
    )

    legacy_path = norm(
        r[
            "legacy_csv"
        ]
    )

    truth_path = norm(
        r[
            "pristine_truth_source"
        ]
    )

    seg_path = derive_segments(
        truth_path
    )

    result = {
        "scenario_id":
            task,

        "sim_age_profile":
            age,

        "status":
            "FAIL",

        "n_segment_frames":
            0,

        "segment_rate_hz":
            np.nan,

        "max_segment_to_truth_time_error_s":
            np.nan,

        "acc_true_rmse_mps2":
            np.nan,

        "acc_true_mae_mps2":
            np.nan,

        "acc_true_corr":
            np.nan,

        "gyro_segment_local_rmse_rads":
            np.nan,

        "gyro_segment_local_mae_rads":
            np.nan,

        "gyro_segment_local_corr":
            np.nan,

        "gyro_segment_local_max_abs_rads":
            np.nan,

        "error":
            "",
    }

    try:

        legacy = pd.read_csv(
            legacy_path,
            comment="#",
        )

        truth = pristine_read(
            truth_path
        )

        seg = pd.read_csv(
            seg_path
        )


        # --------------------------------------------------------
        # Required rich legacy acceleration truth.
        # --------------------------------------------------------

        acc_cols = [
            "accel_true_x",
            "accel_true_y",
            "accel_true_z",
        ]

        if not all(
            c in legacy.columns
            for c in acc_cols
        ):
            raise RuntimeError(
                "legacy accel_true missing"
            )


        # --------------------------------------------------------
        # Legacy output timeline
        # --------------------------------------------------------

        if "timestamp" not in legacy.columns:
            raise RuntimeError(
                "legacy timestamp missing"
            )

        tl = pd.to_numeric(
            legacy[
                "timestamp"
            ],
            errors="coerce",
        ).to_numpy(float)


        legacy_acc = (
            legacy[
                acc_cols
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .to_numpy(float)
        )


        # --------------------------------------------------------
        # Exact high-rate physics truth.
        # --------------------------------------------------------

        th = pd.to_numeric(
            truth[
                "timestamp"
            ],
            errors="coerce",
        ).to_numpy(float)

        high_acc = (
            truth[
                [
                    "accel_true_x",
                    "accel_true_y",
                    "accel_true_z",
                ]
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .to_numpy(float)
        )

        high_gyro = (
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


        # Compare acceleration truth on the legacy 100-Hz
        # timestamps. This is the actual:
        #
        #    30-Hz truth -> interpolation to 100 Hz
        #
        # versus
        #
        #    450-Hz truth -> AA 100 Hz
        #
        high_acc_at_legacy = interp_xyz(
            th,
            high_acc,
            tl,
        )

        da = (
            legacy_acc
            - high_acc_at_legacy
        )

        result[
            "acc_true_rmse_mps2"
        ] = float(
            np.sqrt(
                np.mean(
                    da ** 2
                )
            )
        )

        result[
            "acc_true_mae_mps2"
        ] = float(
            np.mean(
                np.abs(
                    da
                )
            )
        )

        result[
            "acc_true_corr"
        ] = float(
            np.corrcoef(
                legacy_acc.reshape(-1),
                high_acc_at_legacy.reshape(-1),
            )[0, 1]
        )


        # --------------------------------------------------------
        # Reconstruct native legacy gyro_true from segment data.
        #
        # Segment exporter used mj_objectVelocity(..., 0),
        # i.e. world-frame angular velocity.
        #
        # IMU gyro_true used omega_local.
        # --------------------------------------------------------

        required_seg = [
            "time",

            "torso_qw",
            "torso_qx",
            "torso_qy",
            "torso_qz",

            "torso_wx",
            "torso_wy",
            "torso_wz",
        ]

        missing = [
            c for c in required_seg
            if c not in seg.columns
        ]

        if missing:
            raise RuntimeError(
                f"segment columns missing: {missing}"
            )


        ts = pd.to_numeric(
            seg[
                "time"
            ],
            errors="coerce",
        ).to_numpy(float)

        qs = (
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

        ww = (
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


        local_gyro = np.zeros_like(
            ww
        )

        for i in range(
            len(ts)
        ):
            R = quat_wxyz_to_rot(
                qs[i]
            )

            local_gyro[i] = (
                R.T
                @ ww[i]
            )


        if len(ts) > 1:
            seg_hz = (
                1.0
                / float(
                    np.median(
                        np.diff(ts)
                    )
                )
            )
        else:
            seg_hz = np.nan


        result[
            "n_segment_frames"
        ] = len(ts)

        result[
            "segment_rate_hz"
        ] = seg_hz


        # Exact high-rate gyro at native segment timestamps.
        high_gyro_at_seg = interp_xyz(
            th,
            high_gyro,
            ts,
        )


        dg = (
            local_gyro
            - high_gyro_at_seg
        )


        result[
            "gyro_segment_local_rmse_rads"
        ] = float(
            np.sqrt(
                np.mean(
                    dg ** 2
                )
            )
        )

        result[
            "gyro_segment_local_mae_rads"
        ] = float(
            np.mean(
                np.abs(
                    dg
                )
            )
        )

        result[
            "gyro_segment_local_max_abs_rads"
        ] = float(
            np.max(
                np.abs(
                    dg
                )
            )
        )

        result[
            "gyro_segment_local_corr"
        ] = float(
            np.corrcoef(
                local_gyro.reshape(-1),
                high_gyro_at_seg.reshape(-1),
            )[0, 1]
        )


        # Distance from each segment timestamp to nearest
        # high-rate truth timestamp.
        nearest_err = []

        for t in ts:
            nearest_err.append(
                float(
                    np.min(
                        np.abs(
                            th - t
                        )
                    )
                )
            )

        result[
            "max_segment_to_truth_time_error_s"
        ] = max(
            nearest_err
        )


        result[
            "status"
        ] = "OK"


    except Exception as exc:

        result[
            "error"
        ] = str(
            exc
        )


    rows.append(
        result
    )


out = pd.DataFrame(
    rows
)

out.to_csv(
    OUT,
    index=False,
)


print("=" * 110)
print("LEGACY TRUTH-CHANNEL RECONSTRUCTION VALIDATION")
print("=" * 110)

print(
    "records:",
    len(out),
)

print(
    "OK:",
    int(
        (
            out.status
            == "OK"
        ).sum()
    ),
)

print(
    "FAIL:",
    int(
        (
            out.status
            != "OK"
        ).sum()
    ),
)


ok = out[
    out.status
    == "OK"
].copy()


print()
print("Segment sampling:")
print(
    ok[
        "segment_rate_hz"
    ]
    .describe()
    .to_string()
)


print()
print("LEGACY accel_true 30->100 vs HIGH-RATE truth 450->100")
print(
    ok[
        [
            "acc_true_rmse_mps2",
            "acc_true_mae_mps2",
            "acc_true_corr",
        ]
    ]
    .describe()
    .to_string()
)


print()
print("RECONSTRUCTED native gyro_true vs HIGH-RATE truth at native times")
print(
    ok[
        [
            "gyro_segment_local_rmse_rads",
            "gyro_segment_local_mae_rads",
            "gyro_segment_local_corr",
            "gyro_segment_local_max_abs_rads",
        ]
    ]
    .describe()
    .to_string()
)


print()
print("Gyro reconstruction by task:")
print(
    ok.groupby(
        "scenario_id"
    )[
        [
            "gyro_segment_local_rmse_rads",
            "gyro_segment_local_corr",
        ]
    ]
    .median()
    .to_string()
)


if len(
    out[
        out.status != "OK"
    ]
):
    print()
    print("FAILURES:")
    print(
        out[
            out.status != "OK"
        ][
            [
                "scenario_id",
                "sim_age_profile",
                "error",
            ]
        ].to_string(
            index=False
        )
    )


# This gate is deliberately strict for gyro because at a
# matching native simulation state it should be essentially
# the same physical angular velocity, allowing only CSV
# rounding / timestamp interpolation effects.
gyro_ok = (
    len(ok) == 221
    and ok[
        "gyro_segment_local_corr"
    ].median() > 0.999
    and ok[
        "gyro_segment_local_rmse_rads"
    ].median() < 0.02
)

print()
print(
    "GYRO_TRUE_RECONSTRUCTION_VALIDATED:",
    gyro_ok,
)

print()
print(
    "Audit:",
    OUT,
)

print(
    "LEGACY_TRUTH_RECONSTRUCTION_AUDIT_COMPLETE"
)
