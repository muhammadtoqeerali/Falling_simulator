from pathlib import Path
from collections import defaultdict
import json
import math
import re
import sys

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

TARGET = (
    ROOT
    / "outputs/simulated_dataset_highrate_truth_v1"
)

TMP = (
    ROOT
    / "outputs/_building_simulated_dataset_highrate_truth_v1"
)


# Exact historical constants from
# build_simulated_univr_dataset_all_subjects.py
G = 9.80665
RAD_TO_MDPS = 180.0 / math.pi * 1000.0


TASK_DESC = {
    20:  "Forward fall when trying to sit down",
    21:  "Backward fall when trying to sit down",
    22:  "Lateral fall when trying to sit down",
    23:  "Forward fall when trying to get up",
    24:  "Lateral fall when trying to get up",
    25:  "Forward fall while sitting, caused by fainting",
    26:  "Lateral fall while sitting, caused by fainting",
    27:  "Backward fall while sitting, caused by fainting",
    28:  "Vertical/forward fall while walking caused by fainting",
    29:  "Fall while walking, hands used to dampen",
    30:  "Forward fall while walking caused by a trip",
    31:  "Forward fall while jogging caused by a trip",
    32:  "Forward fall while walking caused by a slip",
    33:  "Lateral fall while walking caused by a slip",
    34:  "Backward fall while walking caused by a slip",
    37:  "Backward fall while slowly moving back",
    38:  "Backward fall while quickly moving back",
    39:  "Forward fall from height",
    40:  "Backward fall from height",
    41:  "Backward fall while climbing up the ladder",
    42:  "Backward fall while climbing down the ladder",
    43:  "Forward fall while climbing up the ladder",
    44:  "Vertical fall while climbing up the ladder caused by a slip",
    250: "Forward fall while standing, caused by fainting",
    290: "Fall backward while walking, hands used to dampen",
    291: "Fall lateral while walking, hands used to dampen",
}


EXPECTED_TASKS = {
    20,21,22,23,24,25,26,27,
    28,29,30,31,32,33,34,
    37,38,39,40,41,42,43,44,
    250,290,291,
}

EXPECTED_AGES = {
    25,28,32,33,35,37,38,
    41,46,55,60,62,65,
}


def norm_path(x):
    p = Path(str(x))

    if not p.is_absolute():
        p = ROOT / p

    return p


def finite_float(x):
    try:
        x = float(x)

        if math.isfinite(x):
            return x

    except Exception:
        pass

    return None


def parse_profile_from_path(path):
    """
    Profile metadata fallback from simulator output-folder name.

    Example:
      scenario20_age25_h1p62_sex_female_w58p0_...
    """

    name = path.parent.name

    def grab(pattern, default=""):
        m = re.search(
            pattern,
            name,
        )

        return (
            m.group(1)
            if m
            else default
        )

    return {
        "height_token":
            grab(r"_h([0-9p]+)"),

        "sex":
            grab(r"_sex_([^_]+)"),

        "weight_token":
            grab(r"_w([^_]+)"),
    }


def read_truth(path):
    df = pd.read_csv(
        path,
        comment="#",
    )

    required = [
        "timestamp",

        "accel_true_x",
        "accel_true_y",
        "accel_true_z",

        "gyro_true_x",
        "gyro_true_y",
        "gyro_true_z",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{path}: missing columns {missing}"
        )

    if df.empty:
        raise RuntimeError(
            f"{path}: empty truth file"
        )

    for c in required:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    if df[required].isna().any().any():
        raise RuntimeError(
            f"{path}: non-finite required truth channel"
        )

    t = df[
        "timestamp"
    ].to_numpy(float)

    dt = np.diff(t)

    if len(dt) < 2:
        raise RuntimeError(
            f"{path}: insufficient timestamps"
        )

    med_dt = float(
        np.median(dt)
    )

    hz = 1.0 / med_dt

    if not (
        99.5 <= hz <= 100.5
    ):
        raise RuntimeError(
            f"{path}: expected ~100 Hz, got {hz}"
        )

    if np.any(dt <= 0):
        raise RuntimeError(
            f"{path}: non-monotonic timestamp"
        )

    return df, hz


def nearest_index(
    t,
    event_time,
):
    event_time = finite_float(
        event_time
    )

    if event_time is None:
        return None

    return int(
        np.argmin(
            np.abs(
                t - event_time
            )
        )
    )


def convert_truth(df):
    """
    Historical UniVr-oriented conversion, unchanged:

      AccX = sim Y
      AccY = sim X
      AccZ = sim Z

      GyrX = sim Y
      GyrY = sim X
      GyrZ = sim Z

    All historical signs remain +1.
    """

    t = df[
        "timestamp"
    ].to_numpy(float)

    sim_x = df[
        "accel_true_x"
    ].to_numpy(float)

    sim_y = df[
        "accel_true_y"
    ].to_numpy(float)

    sim_z = df[
        "accel_true_z"
    ].to_numpy(float)

    gx = df[
        "gyro_true_x"
    ].to_numpy(float)

    gy = df[
        "gyro_true_y"
    ].to_numpy(float)

    gz = df[
        "gyro_true_z"
    ].to_numpy(float)

    accX = np.rint(
        sim_y / G * 1000.0
    ).astype(int)

    accY = np.rint(
        sim_x / G * 1000.0
    ).astype(int)

    accZ = np.rint(
        sim_z / G * 1000.0
    ).astype(int)

    gyrX = np.rint(
        gy * RAD_TO_MDPS
    ).astype(int)

    gyrY = np.rint(
        gx * RAD_TO_MDPS
    ).astype(int)

    gyrZ = np.rint(
        gz * RAD_TO_MDPS
    ).astype(int)

    out = pd.DataFrame({
        # Preserve historical converter convention.
        "":
            np.rint(
                t * 1000.0
            ).astype(int),

        "FrameCounter":
            np.arange(
                len(df),
                dtype=int,
            ),

        "AccX": accX,
        "AccY": accY,
        "AccZ": accZ,

        "GyrX": gyrX,
        "GyrY": gyrY,
        "GyrZ": gyrZ,

        "EulerX":
            np.zeros(
                len(df)
            ),

        "EulerY":
            np.zeros(
                len(df)
            ),

        "EulerZ":
            np.zeros(
                len(df)
            ),
    })

    return out


# ============================================================
# PREFLIGHT
# ============================================================

print("=" * 100)
print("HIGH-RATE TRUTH -> UNIVR VALIDATION DATASET")
print("=" * 100)

if not MANIFEST.exists():
    raise SystemExit(
        f"Missing publication manifest:\n{MANIFEST}"
    )

# Never silently replace either a finished dataset
# or a partial build.
if TARGET.exists():
    raise SystemExit(
        "STOP: target dataset already exists:\n"
        f"{TARGET}\n"
        "Nothing was modified."
    )

if TMP.exists():
    raise SystemExit(
        "STOP: temporary build directory already exists:\n"
        f"{TMP}\n"
        "Inspect it before retrying."
    )


m = pd.read_csv(
    MANIFEST
)

required_manifest = [
    "sim_age_profile",
    "scenario_id",
    "pristine_truth_source",
    "fall_onset_time_s",
    "main_impact_time_s",
    "impact_validation_status",
    "primary_onset_analysis",
    "secondary_impact_analysis",
    "physical_overlap_role",
]

missing = [
    c for c in required_manifest
    if c not in m.columns
]

if missing:
    raise SystemExit(
        "Publication manifest is missing columns:\n"
        + "\n".join(missing)
    )


m[
    "sim_age_profile"
] = pd.to_numeric(
    m[
        "sim_age_profile"
    ],
    errors="raise",
).astype(int)

m[
    "scenario_id"
] = pd.to_numeric(
    m[
        "scenario_id"
    ],
    errors="raise",
).astype(int)


print(
    "manifest records:",
    len(m),
)

print(
    "age profiles:",
    m[
        "sim_age_profile"
    ].nunique(),
)

print(
    "semantic tasks:",
    m[
        "scenario_id"
    ].nunique(),
)


if len(m) != 338:
    raise SystemExit(
        f"Expected 338 records, got {len(m)}"
    )

if set(
    m[
        "sim_age_profile"
    ].unique()
) != EXPECTED_AGES:
    raise SystemExit(
        "Age-profile set mismatch"
    )

if set(
    m[
        "scenario_id"
    ].unique()
) != EXPECTED_TASKS:
    raise SystemExit(
        "Semantic task set mismatch"
    )


counts = (
    m.groupby(
        [
            "sim_age_profile",
            "scenario_id",
        ]
    )
    .size()
)

if not (
    counts == 1
).all():
    print(
        counts[
            counts != 1
        ]
    )

    raise SystemExit(
        "Expected exactly one recording per age profile / semantic task"
    )


# Crucial Task250 safety assertion.
task250 = m[
    m[
        "scenario_id"
    ] == 250
]

if len(task250) != 13:
    raise SystemExit(
        "Expected exactly 13 semantic Task250 records"
    )


# ============================================================
# OUTPUT STRUCTURE
# ============================================================

sensors_root = (
    TMP
    / "laboratory/sensors_data"
)

labels_root = (
    TMP
    / "laboratory/labels_data"
)

reports_root = (
    TMP
    / "conversion_reports"
)

sensors_root.mkdir(
    parents=True,
    exist_ok=False,
)

labels_root.mkdir(
    parents=True,
    exist_ok=False,
)

reports_root.mkdir(
    parents=True,
    exist_ok=False,
)


report_rows = []
label_rows = defaultdict(list)

source_rates = []
onset_errors = []
impact_errors = []

baseline_rows = []


# ============================================================
# CONVERSION
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

    pristine = norm_path(
        r[
            "pristine_truth_source"
        ]
    )

    if not pristine.exists():
        raise RuntimeError(
            f"Missing pristine truth:\n{pristine}"
        )

    df, hz = read_truth(
        pristine
    )

    source_rates.append(
        hz
    )

    t = df[
        "timestamp"
    ].to_numpy(float)

    onset_t = finite_float(
        r[
            "fall_onset_time_s"
        ]
    )

    impact_t = finite_float(
        r[
            "main_impact_time_s"
        ]
    )

    if onset_t is None:
        raise RuntimeError(
            f"Missing onset age={age} task={task}"
        )

    if impact_t is None:
        raise RuntimeError(
            f"Missing impact age={age} task={task}"
        )

    onset_idx = nearest_index(
        t,
        onset_t,
    )

    impact_idx = nearest_index(
        t,
        impact_t,
    )

    onset_err = abs(
        float(
            t[onset_idx]
        )
        - onset_t
    )

    impact_err = abs(
        float(
            t[impact_idx]
        )
        - impact_t
    )

    onset_errors.append(
        onset_err
    )

    impact_errors.append(
        impact_err
    )

    if onset_err > 0.006:
        raise RuntimeError(
            f"Onset alignment >6ms age={age} task={task}: {onset_err}"
        )

    if impact_err > 0.006:
        raise RuntimeError(
            f"Impact alignment >6ms age={age} task={task}: {impact_err}"
        )

    if impact_idx < onset_idx:
        raise RuntimeError(
            f"Impact frame before onset age={age} task={task}"
        )

    converted = convert_truth(
        df
    )

    subj_folder = (
        f"SA{age:02d}"
    )

    # One semantic recording per profile/task.
    trial = 1

    out_name = (
        f"S{age:02d}"
        f"T{task:02d}"
        f"R{trial:02d}.csv"
    )

    out_csv = (
        sensors_root
        / subj_folder
        / out_name
    )

    out_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    converted.to_csv(
        out_csv,
        index=False,
    )

    secondary_ok = bool(
        r[
            "secondary_impact_analysis"
        ]
    )

    profile = parse_profile_from_path(
        pristine
    )

    # Try manifest fields first, then folder-derived metadata.
    sex = str(
        r.get(
            "sex",
            "",
        )
    )

    if (
        not sex
        or sex.lower() == "nan"
    ):
        sex = profile[
            "sex"
        ]

    height_token = str(
        r.get(
            "height_token",
            "",
        )
    )

    if (
        not height_token
        or height_token.lower() == "nan"
    ):
        height_token = profile[
            "height_token"
        ]

    weight_token = str(
        r.get(
            "weight_token",
            "",
        )
    )

    if (
        not weight_token
        or weight_token.lower() == "nan"
    ):
        weight_token = profile[
            "weight_token"
        ]

    label_rows[
        age
    ].append({
        "Task Code (Task ID)":
            f"F{task:02d} ({task})",

        "Description":
            TASK_DESC.get(
                task,
                f"Simulated task {task}",
            ),

        "Trial ID":
            trial,

        "Fall_onset_frame":
            onset_idx,

        # Kept for structural compatibility.
        # Publication QC controls whether this frame
        # may be used in impact-aligned analyses.
        "Fall_impact_frame":
            impact_idx,

        "Impact_Validation_Status":
            r[
                "impact_validation_status"
            ],

        "Primary_Onset_Analysis":
            bool(
                r[
                    "primary_onset_analysis"
                ]
            ),

        "Secondary_Impact_Analysis":
            secondary_ok,

        "Physical_Overlap_Role":
            r[
                "physical_overlap_role"
            ],
    })


    # Baseline QC in final oriented units.
    n0 = min(
        100,
        len(converted),
    )

    baseline_rows.append({
        "subject_id":
            age,

        "task_id":
            task,

        "baseline_AccX_g":
            float(
                np.median(
                    converted[
                        "AccX"
                    ].iloc[:n0]
                )
                / 1000.0
            ),

        "baseline_AccY_g":
            float(
                np.median(
                    converted[
                        "AccY"
                    ].iloc[:n0]
                )
                / 1000.0
            ),

        "baseline_AccZ_g":
            float(
                np.median(
                    converted[
                        "AccZ"
                    ].iloc[:n0]
                )
                / 1000.0
            ),
    })


    report_rows.append({
        "status":
            "converted",

        # Compatibility with V2:
        "subject_id":
            age,

        "subject_folder":
            subj_folder,

        # IMPORTANT:
        # semantic scenario ID, NOT path-derived ID.
        "task_id":
            task,

        "trial_id":
            trial,

        "age":
            age,

        "height_token":
            height_token,

        "sex":
            sex,

        "weight_token":
            weight_token,

        "input_csv":
            str(
                pristine
            ),

        "source_type":
            r.get(
                "source_type",
                "",
            ),

        "output_csv":
            str(
                out_csv
            ),

        "rows":
            len(converted),

        "source_sampling_hz":
            hz,

        "fall_onset_time_s":
            onset_t,

        "fall_impact_time_s":
            impact_t,

        "fall_onset_frame":
            onset_idx,

        "fall_impact_frame":
            impact_idx,

        # Used by publication-aware validator.
        "fall_impact_frame_secondary":
            (
                impact_idx
                if secondary_ok
                else np.nan
            ),

        "onset_frame_time_s":
            float(
                t[onset_idx]
            ),

        "impact_frame_time_s":
            float(
                t[impact_idx]
            ),

        "onset_alignment_error_s":
            onset_err,

        "impact_alignment_error_s":
            impact_err,

        "impact_validation_status":
            r[
                "impact_validation_status"
            ],

        "primary_onset_analysis":
            bool(
                r[
                    "primary_onset_analysis"
                ]
            ),

        "secondary_impact_analysis":
            secondary_ok,

        "physical_overlap_role":
            r[
                "physical_overlap_role"
            ],

        "final_onset_source":
            r.get(
                "final_onset_source",
                "",
            ),

        "final_impact_source":
            r.get(
                "final_impact_source",
                "",
            ),
    })

    print(
        f"[OK] profile age {age:02d} "
        f"task {task:03d} -> "
        f"{out_csv.name}"
    )


# ============================================================
# LABEL FILES
# ============================================================

for age, rows in sorted(
    label_rows.items()
):

    labels = pd.DataFrame(
        rows
    ).sort_values(
        [
            "Task Code (Task ID)",
            "Trial ID",
        ]
    )

    label_path = (
        labels_root
        / f"SA{age:02d}_label.xlsx"
    )

    labels.to_excel(
        label_path,
        index=False,
    )


# ============================================================
# REPORTS
# ============================================================

report = pd.DataFrame(
    report_rows
)

report_csv = (
    reports_root
    / "conversion_report_all_subjects.csv"
)

report.to_csv(
    report_csv,
    index=False,
)

report.to_json(
    reports_root
    / "conversion_report_all_subjects.json",
    orient="records",
    indent=2,
)


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
    reports_root
    / "subject_summary.csv",
    index=False,
)


baseline = pd.DataFrame(
    baseline_rows
)

baseline.to_csv(
    reports_root
    / "axis_baseline_qc.csv",
    index=False,
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 100)
print("DERIVED DATASET VALIDATION")
print("=" * 100)

print(
    "converted records:",
    len(report),
)

print(
    "age profiles:",
    report[
        "subject_id"
    ].nunique(),
)

print(
    "semantic tasks:",
    report[
        "task_id"
    ].nunique(),
)

print(
    "Task250 records:",
    int(
        (
            report[
                "task_id"
            ] == 250
        ).sum()
    ),
)

print(
    "primary onset eligible:",
    int(
        report[
            "primary_onset_analysis"
        ].sum()
    ),
)

print(
    "secondary impact eligible:",
    int(
        report[
            "secondary_impact_analysis"
        ].sum()
    ),
)

print()
print("Sampling Hz:")
print(
    pd.Series(
        source_rates
    )
    .describe()
    .to_string()
)

print()
print(
    "max onset frame alignment error [s]:",
    max(
        onset_errors
    ),
)

print(
    "max impact frame alignment error [s]:",
    max(
        impact_errors
    ),
)


print()
print("Baseline oriented acceleration:")
print(
    baseline[
        [
            "baseline_AccX_g",
            "baseline_AccY_g",
            "baseline_AccZ_g",
        ]
    ]
    .median()
    .to_string()
)


# Final structural assertions.
assert len(report) == 338

assert (
    report[
        "subject_id"
    ].nunique()
    == 13
)

assert (
    report[
        "task_id"
    ].nunique()
    == 26
)

assert int(
    (
        report[
            "task_id"
        ] == 250
    ).sum()
) == 13

assert int(
    report[
        "primary_onset_analysis"
    ].sum()
) == 338

assert int(
    report[
        "secondary_impact_analysis"
    ].sum()
) == 286

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

assert max(
    onset_errors
) <= 0.006

assert max(
    impact_errors
) <= 0.006


readme = (
    TMP
    / "README_highrate_truth_conversion.txt"
)

readme.write_text(
    "\n".join([
        "High-rate simulated UniVr-style validation dataset",
        "",
        "Source:",
        "  pristine 100-Hz physics truth generated by exact 450-Hz",
        "  MuJoCo substep capture followed by 450->100 Hz",
        "  anti-aliased polyphase resampling.",
        "",
        "Synthetic profile identifier:",
        "  SAxx / Sxx uses simulator AGE PROFILE number.",
        "  These are not matched physical subjects or digital twins.",
        "",
        "Semantic task identity:",
        "  task_id comes from the publication-QC semantic manifest.",
        "  It is never inferred from scenario folder/filename.",
        "  This preserves Task 250 separately from Task 25.",
        "",
        "Historical axis mapping retained:",
        "  UniVr X/down = simulator Y",
        "  UniVr Y/right = simulator X",
        "  UniVr Z/board-normal = simulator Z",
        "  signs = +1,+1,+1",
        "",
        "Units:",
        "  acceleration m/s^2 -> mg integer",
        "  gyroscope rad/s -> mdps integer",
        "",
        "Event policy:",
        "  338/338 usable for onset-centered primary analysis.",
        "  286/338 usable for impact-centered secondary analysis.",
        "  Tasks 25,26,27,250 are explicitly excluded from",
        "  impact-aligned secondary analysis.",
        "",
        "Original simulator outputs were not modified.",
    ])
    + "\n",
    encoding="utf-8",
)


# Only expose final target after every check succeeds.
TMP.rename(
    TARGET
)


print()
print(
    "HIGH_RATE_DERIVED_DATASET_OK"
)

print(
    "Dataset:",
    TARGET,
)

print(
    "V2 report:",
    TARGET
    / "conversion_reports/"
      "conversion_report_all_subjects.csv",
)

print(
    "No simulator rerun was performed."
)
