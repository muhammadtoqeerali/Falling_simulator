from pathlib import Path
import math
import shutil

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

GYRO_AUDIT = (
    ROOT
    / "outputs/validation_v2/"
      "gyro_inertial_frame_reconstruction_audit.csv"
)

HIGH_FULL_REPORT = (
    ROOT
    / "outputs/simulated_dataset_highrate_truth_v1/"
      "conversion_reports/conversion_report_all_subjects.csv"
)

LEGACY_TARGET = (
    ROOT
    / "outputs/simulated_dataset_legacy_truth_rich221_v1"
)

HIGH_TARGET = (
    ROOT
    / "outputs/simulated_dataset_highrate_truth_rich221_v1"
)

LEGACY_TMP = (
    ROOT
    / "outputs/_building_simulated_dataset_legacy_truth_rich221_v1"
)

HIGH_TMP = (
    ROOT
    / "outputs/_building_simulated_dataset_highrate_truth_rich221_v1"
)


RICH_TASKS = {
    28,29,30,31,32,33,34,
    37,38,39,40,41,42,43,44,
    290,291,
}

EXPECTED_AGES = {
    25,28,32,33,35,37,38,
    41,46,55,60,62,65,
}

G = 9.80665
RAD_TO_MDPS = 180.0 / math.pi * 1000.0


# ============================================================
# Helpers
# ============================================================

def as_bool(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)

    s = str(x).strip().lower()

    if s in {
        "1", "true", "yes", "y", "on"
    }:
        return True

    if s in {
        "0", "false", "no", "n", "off",
        "", "nan", "none"
    }:
        return False

    raise ValueError(
        f"cannot parse boolean: {x!r}"
    )


def normpath(x):
    p = Path(str(x))

    return (
        p
        if p.is_absolute()
        else ROOT / p
    )


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
                    f"missing segment file: {q}"
                )

            return q

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


def nearest_index(
    t,
    x,
):
    return int(
        np.argmin(
            np.abs(
                t - float(x)
            )
        )
    )


def convert_univr(
    t,
    acc,
    gyro,
):
    """
    Historical mapping retained exactly:

       UniVr X/down = simulator Y
       UniVr Y/right = simulator X
       UniVr Z       = simulator Z

    signs = +1,+1,+1
    """

    accX = np.rint(
        acc[:, 1]
        / G
        * 1000.0
    ).astype(int)

    accY = np.rint(
        acc[:, 0]
        / G
        * 1000.0
    ).astype(int)

    accZ = np.rint(
        acc[:, 2]
        / G
        * 1000.0
    ).astype(int)

    gyrX = np.rint(
        gyro[:, 1]
        * RAD_TO_MDPS
    ).astype(int)

    gyrY = np.rint(
        gyro[:, 0]
        * RAD_TO_MDPS
    ).astype(int)

    gyrZ = np.rint(
        gyro[:, 2]
        * RAD_TO_MDPS
    ).astype(int)

    return pd.DataFrame({
        "":
            np.rint(
                t * 1000.0
            ).astype(int),

        "FrameCounter":
            np.arange(
                len(t),
                dtype=int,
            ),

        "AccX":
            accX,

        "AccY":
            accY,

        "AccZ":
            accZ,

        "GyrX":
            gyrX,

        "GyrY":
            gyrY,

        "GyrZ":
            gyrZ,

        "EulerX":
            np.zeros(len(t)),

        "EulerY":
            np.zeros(len(t)),

        "EulerZ":
            np.zeros(len(t)),
    })


def write_labels(
    root,
    labels_by_age,
):
    labels_root = (
        root
        / "laboratory/labels_data"
    )

    labels_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for age, rows in sorted(
        labels_by_age.items()
    ):
        pd.DataFrame(
            rows
        ).sort_values(
            [
                "Task Code (Task ID)",
                "Trial ID",
            ]
        ).to_excel(
            labels_root
            / f"SA{age:02d}_label.xlsx",
            index=False,
        )


def label_row(
    task,
    onset,
    impact,
    r,
):
    return {
        "Task Code (Task ID)":
            f"F{task:02d} ({task})",

        "Description":
            f"Simulated task {task}",

        "Trial ID":
            1,

        "Fall_onset_frame":
            int(onset),

        "Fall_impact_frame":
            int(impact),

        "Impact_Validation_Status":
            str(
                r[
                    "impact_validation_status"
                ]
            ),

        "Primary_Onset_Analysis":
            as_bool(
                r[
                    "primary_onset_analysis"
                ]
            ),

        "Secondary_Impact_Analysis":
            as_bool(
                r[
                    "secondary_impact_analysis"
                ]
            ),

        "Physical_Overlap_Role":
            str(
                r[
                    "physical_overlap_role"
                ]
            ),
    }


# ============================================================
# Preflight
# ============================================================

print("=" * 110)
print("BUILD SIX-AXIS TRUTH-ONLY RICH-IMU PAIR")
print("=" * 110)


for p in [
    MANIFEST,
    GYRO_AUDIT,
    HIGH_FULL_REPORT,
]:
    if not p.exists():
        raise SystemExit(
            f"missing prerequisite: {p}"
        )


for p in [
    LEGACY_TARGET,
    HIGH_TARGET,
    LEGACY_TMP,
    HIGH_TMP,
]:
    if p.exists():
        raise SystemExit(
            f"STOP: output already exists:\n{p}"
        )


m = pd.read_csv(
    MANIFEST
)

m[
    "scenario_id"
] = pd.to_numeric(
    m[
        "scenario_id"
    ],
    errors="raise",
).astype(int)

m[
    "sim_age_profile"
] = pd.to_numeric(
    m[
        "sim_age_profile"
    ],
    errors="raise",
).astype(int)

m = m[
    m[
        "scenario_id"
    ].isin(
        RICH_TASKS
    )
].copy()


assert len(m) == 221
assert set(
    m[
        "scenario_id"
    ].unique()
) == RICH_TASKS

assert set(
    m[
        "sim_age_profile"
    ].unique()
) == EXPECTED_AGES

assert (
    m.groupby(
        [
            "sim_age_profile",
            "scenario_id",
        ]
    )
    .size()
    .eq(1)
    .all()
)


audit = pd.read_csv(
    GYRO_AUDIT
)

audit[
    "scenario_id"
] = pd.to_numeric(
    audit[
        "scenario_id"
    ]
).astype(int)

audit[
    "sim_age_profile"
] = pd.to_numeric(
    audit[
        "sim_age_profile"
    ]
).astype(int)


if len(audit) != 221:
    raise SystemExit(
        f"expected 221 gyro audit rows, got {len(audit)}"
    )


m = m.merge(
    audit[
        [
            "scenario_id",
            "sim_age_profile",
            "corrected_zero_corr",
            "corrected_zero_rmse_rads",
            "corrected_zero_mag_corr",
        ]
    ],
    on=[
        "scenario_id",
        "sim_age_profile",
    ],
    how="left",
    validate="one_to_one",
)


if m[
    "corrected_zero_corr"
].isna().any():
    raise SystemExit(
        "missing gyro reconstruction QC rows"
    )


print(
    "records        :",
    len(m),
)

print(
    "age profiles   :",
    m[
        "sim_age_profile"
    ].nunique(),
)

print(
    "semantic tasks :",
    m[
        "scenario_id"
    ].nunique(),
)

print(
    "gyro QC median corr:",
    float(
        m[
            "corrected_zero_corr"
        ].median()
    ),
)


# ============================================================
# Static MuJoCo Torso inertial-frame transform.
#
# This creates an environment only to read the model constant.
# It does NOT execute a fall scenario.
# ============================================================

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
        "Torso not found"
    )


body_iquat = np.asarray(
    model.body_iquat[
        torso_id
    ],
    dtype=float,
).copy()

R_i = quat_wxyz_to_R(
    body_iquat
)

env.close()


print(
    "Torso body_iquat:",
    body_iquat,
)


# ============================================================
# Create directory structure
# ============================================================

for root in [
    LEGACY_TMP,
    HIGH_TMP,
]:

    (
        root
        / "laboratory/sensors_data"
    ).mkdir(
        parents=True,
        exist_ok=False,
    )

    (
        root
        / "conversion_reports"
    ).mkdir(
        parents=True,
        exist_ok=False,
    )


high_full = pd.read_csv(
    HIGH_FULL_REPORT
)

high_full[
    "subject_id"
] = pd.to_numeric(
    high_full[
        "subject_id"
    ]
).astype(int)

high_full[
    "task_id"
] = pd.to_numeric(
    high_full[
        "task_id"
    ]
).astype(int)


legacy_report_rows = []
high_report_rows = []

legacy_labels = {}
high_labels = {}

native_rates = []
legacy_output_rates = []

start_errors = []
end_errors = []


# ============================================================
# Build paired records
# ============================================================

for _, r in m.sort_values(
    [
        "sim_age_profile",
        "scenario_id",
    ]
).iterrows():

    age = int(
        r[
            "sim_age_profile"
        ]
    )

    task = int(
        r[
            "scenario_id"
        ]
    )


    # --------------------------------------------------------
    # Legacy truth:
    #
    # Acceleration:
    #   use saved legacy accel_true, already produced by
    #   legacy 30-Hz -> linear 100-Hz interpolation.
    #
    # Gyro:
    #   reconstruct the omitted native gyro_true from the
    #   segment world's omega + BODY inertial-frame rotation,
    #   then reproduce the legacy np.interp -> 100 Hz.
    #
    # No timing optimization or per-record shift is applied.
    # --------------------------------------------------------

    legacy_path = normpath(
        r[
            "legacy_csv"
        ]
    )

    truth_path = normpath(
        r[
            "pristine_truth_source"
        ]
    )

    seg_path = derive_segments(
        truth_path
    )


    legacy = pd.read_csv(
        legacy_path,
        comment="#",
    )


    required_legacy = [
        "timestamp",
        "accel_true_x",
        "accel_true_y",
        "accel_true_z",
    ]

    missing = [
        c
        for c in required_legacy
        if c not in legacy.columns
    ]

    if missing:
        raise RuntimeError(
            f"age={age} task={task}: "
            f"legacy truth missing {missing}"
        )


    t100 = pd.to_numeric(
        legacy[
            "timestamp"
        ],
        errors="coerce",
    ).to_numpy(float)

    acc100 = (
        legacy[
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


    if (
        not np.isfinite(t100).all()
        or not np.isfinite(acc100).all()
    ):
        raise RuntimeError(
            f"age={age} task={task}: "
            "non-finite legacy truth"
        )


    hz100 = (
        1.0
        / float(
            np.median(
                np.diff(
                    t100
                )
            )
        )
    )

    legacy_output_rates.append(
        hz100
    )

    if not (
        99.5
        <= hz100
        <= 100.5
    ):
        raise RuntimeError(
            f"age={age} task={task}: "
            f"legacy truth output rate {hz100}"
        )


    seg = pd.read_csv(
        seg_path
    )


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
        c
        for c in required_seg
        if c not in seg.columns
    ]

    if missing:
        raise RuntimeError(
            f"age={age} task={task}: "
            f"segment columns missing {missing}"
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


    native_hz = (
        1.0
        / float(
            np.median(
                np.diff(
                    ts
                )
            )
        )
    )

    native_rates.append(
        native_hz
    )


    if not (
        29.5
        <= native_hz
        <= 30.5
    ):
        raise RuntimeError(
            f"age={age} task={task}: "
            f"segment rate {native_hz}"
        )


    start_err = abs(
        float(
            t100[0]
            - ts[0]
        )
    )

    end_err = abs(
        float(
            t100[-1]
            - ts[-1]
        )
    )

    start_errors.append(
        start_err
    )

    end_errors.append(
        end_err
    )


    # One native frame is ~33 ms.
    # Timeline endpoints should be much closer than that.
    if start_err > 0.04:
        raise RuntimeError(
            f"age={age} task={task}: "
            f"start timeline mismatch {start_err}"
        )

    if end_err > 0.04:
        raise RuntimeError(
            f"age={age} task={task}: "
            f"end timeline mismatch {end_err}"
        )


    gyro_native = np.zeros_like(
        omega_world
    )


    for i in range(
        len(ts)
    ):

        R_body = quat_wxyz_to_R(
            qs[i]
        )

        # MJOBJ_BODY local orientation includes
        # the body's static inertial-frame quaternion.
        R_global_inertial = (
            R_body
            @ R_i
        )

        gyro_native[i] = (
            R_global_inertial.T
            @ omega_world[i]
        )


    # Exact legacy resampling semantic:
    # component-wise linear np.interp to legacy t100.
    gyro100 = interp3(
        ts,
        gyro_native,
        t100,
    )


    converted_legacy = convert_univr(
        t100,
        acc100,
        gyro100,
    )


    subj = f"SA{age:02d}"

    filename = (
        f"S{age:02d}"
        f"T{task:02d}"
        f"R01.csv"
    )


    legacy_tmp_csv = (
        LEGACY_TMP
        / "laboratory/sensors_data"
        / subj
        / filename
    )

    legacy_final_csv = (
        LEGACY_TARGET
        / "laboratory/sensors_data"
        / subj
        / filename
    )

    legacy_tmp_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    converted_legacy.to_csv(
        legacy_tmp_csv,
        index=False,
    )


    onset_t = float(
        r[
            "fall_onset_time_s"
        ]
    )

    impact_t = float(
        r[
            "main_impact_time_s"
        ]
    )

    legacy_onset = nearest_index(
        t100,
        onset_t,
    )

    legacy_impact = nearest_index(
        t100,
        impact_t,
    )


    if legacy_impact < legacy_onset:
        raise RuntimeError(
            f"age={age} task={task}: "
            "legacy impact before onset"
        )


    secondary = as_bool(
        r[
            "secondary_impact_analysis"
        ]
    )

    primary = as_bool(
        r[
            "primary_onset_analysis"
        ]
    )


    legacy_labels.setdefault(
        age,
        [],
    ).append(
        label_row(
            task,
            legacy_onset,
            legacy_impact,
            r,
        )
    )


    legacy_report_rows.append({
        "status":
            "converted",

        "subject_id":
            age,

        "subject_folder":
            subj,

        "task_id":
            task,

        "trial_id":
            1,

        "age":
            age,

        "output_csv":
            str(
                legacy_final_csv
            ),

        "rows":
            len(
                converted_legacy
            ),

        "source_sampling_hz":
            hz100,

        "native_truth_sampling_hz":
            native_hz,

        "acquisition_variant":
            "legacy_truth_30Hz_linear_100Hz",

        "acceleration_truth_source":
            "saved_legacy_accel_true",

        "gyro_truth_source":
            "reconstructed_MJOBJ_BODY_inertial_local",

        "gyro_resampling":
            "np.interp_native_segment_to_legacy_100Hz",

        "timing_shift_applied_s":
            0.0,

        "source_legacy_csv":
            str(
                legacy_path
            ),

        "source_segments_csv":
            str(
                seg_path
            ),

        "paired_pristine_highrate_truth":
            str(
                truth_path
            ),

        "fall_onset_time_s":
            onset_t,

        "fall_impact_time_s":
            impact_t,

        "fall_onset_frame":
            legacy_onset,

        "fall_impact_frame":
            legacy_impact,

        "fall_impact_frame_secondary":
            (
                legacy_impact
                if secondary
                else np.nan
            ),

        "primary_onset_analysis":
            primary,

        "secondary_impact_analysis":
            secondary,

        "impact_validation_status":
            str(
                r[
                    "impact_validation_status"
                ]
            ),

        "physical_overlap_role":
            str(
                r[
                    "physical_overlap_role"
                ]
            ),

        "gyro_reconstruction_zero_corr":
            float(
                r[
                    "corrected_zero_corr"
                ]
            ),

        "gyro_reconstruction_zero_rmse_rads":
            float(
                r[
                    "corrected_zero_rmse_rads"
                ]
            ),

        "gyro_reconstruction_zero_mag_corr":
            float(
                r[
                    "corrected_zero_mag_corr"
                ]
            ),
    })


    # --------------------------------------------------------
    # High-rate truth subset.
    #
    # Reuse the already validated converted high-rate signal
    # for the exact same age/task run.
    # --------------------------------------------------------

    h = high_full[
        (
            high_full[
                "subject_id"
            ] == age
        )
        &
        (
            high_full[
                "task_id"
            ] == task
        )
    ]


    if len(h) != 1:
        raise RuntimeError(
            f"age={age} task={task}: "
            f"expected one high-rate record, got {len(h)}"
        )


    h = h.iloc[0]

    source_high_csv = normpath(
        h[
            "output_csv"
        ]
    )

    if not source_high_csv.exists():
        raise RuntimeError(
            f"missing high-rate converted CSV: "
            f"{source_high_csv}"
        )


    high_tmp_csv = (
        HIGH_TMP
        / "laboratory/sensors_data"
        / subj
        / filename
    )

    high_final_csv = (
        HIGH_TARGET
        / "laboratory/sensors_data"
        / subj
        / filename
    )


    high_tmp_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source_high_csv,
        high_tmp_csv,
    )


    high_onset = int(
        h[
            "fall_onset_frame"
        ]
    )

    high_impact = int(
        h[
            "fall_impact_frame"
        ]
    )


    high_labels.setdefault(
        age,
        [],
    ).append(
        label_row(
            task,
            high_onset,
            high_impact,
            r,
        )
    )


    high_row = (
        h.to_dict()
    )

    high_row[
        "output_csv"
    ] = str(
        high_final_csv
    )

    high_row[
        "acquisition_variant"
    ] = (
        "highrate_truth_450Hz_polyphase_AA_100Hz"
    )

    high_row[
        "truth_only_rich221_subset"
    ] = True

    high_row[
        "paired_legacy_truth_dataset"
    ] = str(
        legacy_final_csv
    )

    high_report_rows.append(
        high_row
    )


    print(
        f"[OK] age {age:02d} "
        f"task {task:03d}"
    )


# ============================================================
# Write labels + reports
# ============================================================

write_labels(
    LEGACY_TMP,
    legacy_labels,
)

write_labels(
    HIGH_TMP,
    high_labels,
)


legacy_report = pd.DataFrame(
    legacy_report_rows
)

high_report = pd.DataFrame(
    high_report_rows
)


legacy_report.to_csv(
    LEGACY_TMP
    / "conversion_reports/"
      "conversion_report_all_subjects.csv",
    index=False,
)

high_report.to_csv(
    HIGH_TMP
    / "conversion_reports/"
      "conversion_report_all_subjects.csv",
    index=False,
)


for root, report in [
    (
        LEGACY_TMP,
        legacy_report,
    ),
    (
        HIGH_TMP,
        high_report,
    ),
]:

    summary = (
        report.groupby(
            "subject_id"
        )
        .agg(
            converted_files=(
                "status",
                "size",
            ),

            unique_tasks=(
                "task_id",
                "nunique",
            ),

            first_task=(
                "task_id",
                "min",
            ),

            last_task=(
                "task_id",
                "max",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        root
        / "conversion_reports/"
          "subject_summary.csv",
        index=False,
    )


# ============================================================
# Provenance README
# ============================================================

(
    LEGACY_TMP
    / "README_truth_only_sampling_pair.txt"
).write_text(
    "\n".join([
        "Legacy truth-only rich-IMU acquisition dataset.",
        "",
        "Scope:",
        "  221 records = 13 age profiles x 17 rich-IMU tasks.",
        "  Tasks: 28-34, 37-44, 290, 291.",
        "",
        "Acceleration:",
        "  saved legacy accel_true.",
        "  Native simulator states were ~30 Hz and the",
        "  legacy IMU linearly interpolated truth to 100 Hz.",
        "",
        "Gyroscope:",
        "  legacy gyro_true was computed internally but omitted",
        "  from the legacy CSV.",
        "  It is reconstructed from saved Torso world angular",
        "  velocity plus MuJoCo MJOBJ_BODY inertial orientation",
        "  using xquat * body_iquat.",
        "  Native reconstructed samples are linearly interpolated",
        "  with numpy.interp to the existing legacy 100-Hz timeline.",
        "",
        "Timing policy:",
        "  ZERO additional timing shift.",
        "  Per-record best shifts from diagnostic work are NOT used.",
        "",
        "Sensor noise, bias, STA, clipping and LP filtering:",
        "  excluded from this truth-only comparison.",
        "",
        "No simulator fall task was rerun.",
    ])
    + "\n",
    encoding="utf-8",
)


(
    HIGH_TMP
    / "README_truth_only_sampling_pair.txt"
).write_text(
    "\n".join([
        "High-rate truth-only rich-IMU acquisition dataset.",
        "",
        "Scope:",
        "  exact same 221 records as paired legacy truth dataset.",
        "",
        "Acquisition:",
        "  exact 450-Hz MuJoCo substep truth",
        "  -> anti-aliased polyphase 100-Hz truth.",
        "",
        "Sensor noise, bias, STA, clipping and LP filtering:",
        "  excluded.",
        "",
        "No simulator fall task was rerun.",
    ])
    + "\n",
    encoding="utf-8",
)


# ============================================================
# Structural gates before publication
# ============================================================

for name, report in [
    (
        "legacy truth",
        legacy_report,
    ),
    (
        "highrate truth",
        high_report,
    ),
]:

    assert len(
        report
    ) == 221

    assert report[
        "subject_id"
    ].nunique() == 13

    assert report[
        "task_id"
    ].nunique() == 17

    assert set(
        report[
            "task_id"
        ].astype(int)
    ) == RICH_TASKS

    assert (
        report.groupby(
            [
                "subject_id",
                "task_id",
            ]
        )
        .size()
        .eq(1)
        .all()
    )


print()
print("=" * 110)
print("PRE-PUBLISH VALIDATION")
print("=" * 110)

print(
    "legacy records        :",
    len(
        legacy_report
    ),
)

print(
    "high-rate records     :",
    len(
        high_report
    ),
)

print(
    "tasks                 :",
    legacy_report[
        "task_id"
    ].nunique(),
)

print(
    "age profiles          :",
    legacy_report[
        "subject_id"
    ].nunique(),
)

print(
    "legacy native Hz median:",
    float(
        np.median(
            native_rates
        )
    ),
)

print(
    "legacy output Hz median:",
    float(
        np.median(
            legacy_output_rates
        )
    ),
)

print(
    "max start-time mismatch [s]:",
    max(
        start_errors
    ),
)

print(
    "max end-time mismatch [s]  :",
    max(
        end_errors
    ),
)

print(
    "gyro reconstruction zero-shift median corr:",
    float(
        legacy_report[
            "gyro_reconstruction_zero_corr"
        ].median()
    ),
)


# ============================================================
# Publish both completed temporary datasets.
# Reports already contain FINAL paths, avoiding the previous
# stale _building_ path issue.
# ============================================================

LEGACY_TMP.rename(
    LEGACY_TARGET
)

HIGH_TMP.rename(
    HIGH_TARGET
)


# Read-back path verification.
for report_path in [
    LEGACY_TARGET
    / "conversion_reports/"
      "conversion_report_all_subjects.csv",

    HIGH_TARGET
    / "conversion_reports/"
      "conversion_report_all_subjects.csv",
]:

    check = pd.read_csv(
        report_path
    )

    assert len(
        check
    ) == 221

    missing = [
        p
        for p in check[
            "output_csv"
        ]
        if not Path(
            p
        ).exists()
    ]

    if missing:
        raise RuntimeError(
            f"published report has "
            f"{len(missing)} missing output paths"
        )


print()
print(
    "TRUTH_ONLY_RICH221_PAIR_OK"
)

print(
    "Legacy:",
    LEGACY_TARGET,
)

print(
    "Highrate:",
    HIGH_TARGET,
)

print(
    "No simulator fall task was rerun."
)
