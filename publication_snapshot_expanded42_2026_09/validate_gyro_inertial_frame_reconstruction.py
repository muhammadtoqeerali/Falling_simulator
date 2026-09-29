from pathlib import Path

import mujoco
import numpy as np
import pandas as pd
from humenv import make_humenv


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
      "gyro_inertial_frame_reconstruction_audit.csv"
)

RICH_TASKS = {
    28,29,30,31,32,33,34,
    37,38,39,40,41,42,43,44,
    290,291,
}


# ============================================================
# Helpers
# ============================================================

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

            seg = p.with_name(
                stem + "_segments.csv"
            )

            if not seg.exists():
                raise RuntimeError(
                    f"missing segments: {seg}"
                )

            return seg

    raise RuntimeError(
        f"unexpected truth filename: {p}"
    )


def quat_wxyz_to_R(q):
    q = np.asarray(
        q,
        dtype=float,
    )

    n = float(
        np.linalg.norm(q)
    )

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


def interp3(
    t_src,
    xyz,
    t_dst,
):
    out = np.zeros(
        (len(t_dst), 3),
        dtype=float,
    )

    for k in range(3):
        out[:, k] = np.interp(
            t_dst,
            t_src,
            xyz[:, k],
        )

    return out


def corr_flat(
    a,
    b,
):
    a = np.asarray(
        a,
        dtype=float,
    ).reshape(-1)

    b = np.asarray(
        b,
        dtype=float,
    ).reshape(-1)

    good = (
        np.isfinite(a)
        & np.isfinite(b)
    )

    a = a[good]
    b = b[good]

    if (
        len(a) < 10
        or np.std(a) < 1e-12
        or np.std(b) < 1e-12
    ):
        return np.nan

    return float(
        np.corrcoef(a, b)[0, 1]
    )


def metrics(
    candidate,
    reference,
):
    d = (
        candidate
        - reference
    )

    cmag = np.linalg.norm(
        candidate,
        axis=1,
    )

    rmag = np.linalg.norm(
        reference,
        axis=1,
    )

    return {
        "corr":
            corr_flat(
                candidate,
                reference,
            ),

        "rmse_rads":
            float(
                np.sqrt(
                    np.mean(
                        d ** 2
                    )
                )
            ),

        "mae_rads":
            float(
                np.mean(
                    np.abs(d)
                )
            ),

        "max_abs_rads":
            float(
                np.max(
                    np.abs(d)
                )
            ),

        "mag_corr":
            corr_flat(
                cmag,
                rmag,
            ),

        "mag_rmse_rads":
            float(
                np.sqrt(
                    np.mean(
                        (
                            cmag
                            - rmag
                        ) ** 2
                    )
                )
            ),
    }


# ============================================================
# Load model ONCE.
#
# This does not execute a fall scenario.
# We only need the static body_iquat.
# ============================================================

print("=" * 110)
print("STATIC MUJOCO BODY-INERTIAL FRAME")
print("=" * 110)

env, _ = make_humenv(
    task="move-ego-0-0"
)

model = env.unwrapped.model

torso_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "Torso",
)

if torso_id < 0:
    env.close()
    raise RuntimeError(
        "Torso body not found"
    )


torso_name = mujoco.mj_id2name(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    torso_id,
)

body_iquat = np.asarray(
    model.body_iquat[
        torso_id
    ],
    dtype=float,
).copy()

body_ipos = np.asarray(
    model.body_ipos[
        torso_id
    ],
    dtype=float,
).copy()

R_i = quat_wxyz_to_R(
    body_iquat
)


print(
    "Torso body id    :",
    torso_id,
)

print(
    "Torso body name  :",
    torso_name,
)

print(
    "Torso body_iquat :",
    body_iquat,
)

print(
    "Torso body_ipos  :",
    body_ipos,
)

print()
print(
    "R(body_iquat):"
)

print(
    R_i
)


identity_angle = float(
    2.0
    * np.degrees(
        np.arccos(
            np.clip(
                abs(
                    body_iquat[0]
                    / max(
                        np.linalg.norm(
                            body_iquat
                        ),
                        1e-12,
                    )
                ),
                -1.0,
                1.0,
            )
        )
    )
)

print()
print(
    "inertial-frame rotation angle [deg]:",
    identity_angle,
)

env.close()


# ============================================================
# Existing 221-record audit
# ============================================================

m = pd.read_csv(
    MANIFEST
)

m = m[
    m[
        "scenario_id"
    ]
    .astype(int)
    .isin(
        RICH_TASKS
    )
].copy()


rows = []

shifts = [
    -1.0 / 450.0,
    0.0,
    +1.0 / 450.0,
]


for _, rec in m.iterrows():

    task = int(
        rec[
            "scenario_id"
        ]
    )

    age = int(
        rec[
            "sim_age_profile"
        ]
    )

    truth_path = normpath(
        rec[
            "pristine_truth_source"
        ]
    )

    seg_path = derive_segments(
        truth_path
    )


    truth = pd.read_csv(
        truth_path,
        comment="#",
    )

    seg = pd.read_csv(
        seg_path,
    )


    th = pd.to_numeric(
        truth[
            "timestamp"
        ],
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
        seg[
            "time"
        ],
        errors="coerce",
    ).to_numpy(float)

    q_body_global = (
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

    omega_world = (
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


    if not (
        len(ts)
        == len(q_body_global)
        == len(omega_world)
    ):
        raise RuntimeError(
            f"segment length mismatch "
            f"task={task} age={age}"
        )


    old_xquat_only = np.zeros_like(
        omega_world
    )

    inertial_corrected = np.zeros_like(
        omega_world
    )


    for i in range(
        len(ts)
    ):

        R_body = quat_wxyz_to_R(
            q_body_global[i]
        )


        # Previous hypothesis:
        #
        # BODY kinematic frame only.
        old_xquat_only[i] = (
            R_body.T
            @ omega_world[i]
        )


        # MuJoCo mjOBJ_BODY local orientation:
        #
        # q_global_inertial =
        #     xquat * body_iquat
        #
        # Therefore:
        #
        # R_global_inertial =
        #     R(xquat) @ R(body_iquat)
        #
        R_global_inertial = (
            R_body
            @ R_i
        )


        inertial_corrected[i] = (
            R_global_inertial.T
            @ omega_world[i]
        )


    # --------------------------------------------------------
    # Zero-shift metrics
    # --------------------------------------------------------

    valid0 = (
        (ts >= th[0])
        & (ts <= th[-1])
    )

    ref0 = interp3(
        th,
        gh,
        ts[
            valid0
        ],
    )


    old0 = metrics(
        old_xquat_only[
            valid0
        ],
        ref0,
    )

    new0 = metrics(
        inertial_corrected[
            valid0
        ],
        ref0,
    )


    # --------------------------------------------------------
    # Only test ± one physics substep for timestamp phase.
    # This is not fitting; it tests the known 450-Hz sampling
    # phase ambiguity.
    # --------------------------------------------------------

    best = None

    for shift in shifts:

        tq = (
            ts
            + shift
        )

        valid = (
            (tq >= th[0])
            & (tq <= th[-1])
        )

        if int(
            valid.sum()
        ) < 20:
            continue


        ref = interp3(
            th,
            gh,
            tq[
                valid
            ],
        )

        candidate = (
            inertial_corrected[
                valid
            ]
        )

        mm = metrics(
            candidate,
            ref,
        )

        cand = {
            "shift_s":
                shift,
            **mm,
        }

        if (
            best is None
            or cand[
                "corr"
            ] > best[
                "corr"
            ]
        ):
            best = cand


    rows.append({
        "scenario_id":
            task,

        "sim_age_profile":
            age,

        "body_iquat_w":
            body_iquat[0],

        "body_iquat_x":
            body_iquat[1],

        "body_iquat_y":
            body_iquat[2],

        "body_iquat_z":
            body_iquat[3],

        "inertial_rotation_deg":
            identity_angle,

        "old_xquat_only_corr":
            old0[
                "corr"
            ],

        "old_xquat_only_rmse_rads":
            old0[
                "rmse_rads"
            ],

        "corrected_zero_corr":
            new0[
                "corr"
            ],

        "corrected_zero_rmse_rads":
            new0[
                "rmse_rads"
            ],

        "corrected_zero_mae_rads":
            new0[
                "mae_rads"
            ],

        "corrected_zero_mag_corr":
            new0[
                "mag_corr"
            ],

        "corrected_zero_mag_rmse_rads":
            new0[
                "mag_rmse_rads"
            ],

        "corrected_best_corr":
            best[
                "corr"
            ],

        "corrected_best_rmse_rads":
            best[
                "rmse_rads"
            ],

        "corrected_best_mag_corr":
            best[
                "mag_corr"
            ],

        "corrected_best_shift_s":
            best[
                "shift_s"
            ],
    })


out = pd.DataFrame(
    rows
)

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

out.to_csv(
    OUT,
    index=False,
)


# ============================================================
# Results
# ============================================================

print()
print("=" * 110)
print("MUJOCO BODY-INERTIAL GYRO RECONSTRUCTION")
print("=" * 110)

print(
    "records:",
    len(out),
)

print()
print(
    "PREVIOUS xquat-only reconstruction:"
)

print(
    out[
        [
            "old_xquat_only_corr",
            "old_xquat_only_rmse_rads",
        ]
    ]
    .describe()
    .to_string()
)


print()
print(
    "INERTIAL-CORRECTED reconstruction, zero shift:"
)

print(
    out[
        [
            "corrected_zero_corr",
            "corrected_zero_rmse_rads",
            "corrected_zero_mae_rads",
            "corrected_zero_mag_corr",
            "corrected_zero_mag_rmse_rads",
        ]
    ]
    .describe()
    .to_string()
)


print()
print(
    "INERTIAL-CORRECTED, best ±1 physics substep:"
)

print(
    out[
        [
            "corrected_best_corr",
            "corrected_best_rmse_rads",
            "corrected_best_mag_corr",
            "corrected_best_shift_s",
        ]
    ]
    .describe()
    .to_string()
)


print()
print(
    "Best shift counts:"
)

print(
    out[
        "corrected_best_shift_s"
    ]
    .round(9)
    .value_counts()
    .sort_index()
    .to_string()
)


print()
print(
    "Corrected zero-shift by task:"
)

print(
    out.groupby(
        "scenario_id"
    )[
        [
            "corrected_zero_corr",
            "corrected_zero_rmse_rads",
            "corrected_zero_mag_corr",
        ]
    ]
    .median()
    .to_string()
)


old_med = float(
    out[
        "old_xquat_only_corr"
    ].median()
)

new_med = float(
    out[
        "corrected_zero_corr"
    ].median()
)

mag_med = float(
    out[
        "corrected_zero_mag_corr"
    ].median()
)

best_med = float(
    out[
        "corrected_best_corr"
    ].median()
)


print()
print(
    "Median vector corr:"
)

print(
    "  previous xquat-only :",
    old_med,
)

print(
    "  corrected zero      :",
    new_med,
)

print(
    "  corrected best      :",
    best_med,
)

print(
    "  magnitude zero      :",
    mag_med,
)


# Validation criterion:
#
# Corrected vector agreement should become close to the
# already-observed magnitude agreement.
#
# RMSE need not be near zero because we are comparing native
# 30-Hz segment states with the high-rate stream after
# 450->100-Hz anti-aliased resampling.
validated = bool(
    len(out) == 221
    and new_med > 0.95
    and best_med > 0.97
    and (
        mag_med
        - best_med
    ) < 0.04
)


print()
print(
    "BODY_INERTIAL_FRAME_HYPOTHESIS_VALIDATED:",
    validated,
)

print()
print(
    "Audit:",
    OUT,
)

print(
    "GYRO_INERTIAL_FRAME_RECONSTRUCTION_DONE"
)
