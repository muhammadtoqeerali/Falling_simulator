from pathlib import Path
import math
import re

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
    / "outputs/simulated_dataset_legacy_paired_v1"
)

TMP = (
    ROOT
    / "outputs/_building_simulated_dataset_legacy_paired_v1"
)

G = 9.80665
RAD_TO_MDPS = 180.0 / math.pi * 1000.0


EXPECTED_TASKS = {
    20,21,22,23,24,25,26,27,
    28,29,30,31,32,33,34,
    37,38,39,40,41,42,43,44,
    250,290,291,
}


def norm_path(x):
    p = Path(str(x))
    return p if p.is_absolute() else ROOT / p


def get_col(df, names):
    lookup = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for n in names:
        if str(n).lower() in lookup:
            return lookup[str(n).lower()]

    return None


def vec(df, names):
    c = get_col(df, names)

    if c is None:
        raise RuntimeError(
            f"none of columns found: {names}"
        )

    return (
        pd.to_numeric(
            df[c],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(float)
    )


def time_vector(df):
    c = get_col(
        df,
        [
            "Time_s_standard",
            "t",
            "time_s",
            "timestamp",
            "Time",
        ],
    )

    if c is None:
        return (
            np.arange(
                len(df),
                dtype=float,
            )
            / 100.0
        )

    t = (
        pd.to_numeric(
            df[c],
            errors="coerce",
        )
        .to_numpy(float)
    )

    if not np.isfinite(t).all():
        raise RuntimeError(
            "non-finite legacy timestamps"
        )

    # Historical converter behavior.
    if len(t) and t[0] > 100:
        t = t - t[0]

    return t


def nearest(t, value):
    return int(
        np.argmin(
            np.abs(
                t - float(value)
            )
        )
    )


def convert_legacy(df, t):
    # Exact old-converter signal preference.
    sim_x = vec(
        df,
        ["accel_raw_x", "accel_x", "ax"],
    )
    sim_y = vec(
        df,
        ["accel_raw_y", "accel_y", "ay"],
    )
    sim_z = vec(
        df,
        ["accel_raw_z", "accel_z", "az"],
    )

    gx = vec(
        df,
        ["gyro_x", "gyr_x", "gx"],
    )
    gy = vec(
        df,
        ["gyro_y", "gyr_y", "gy"],
    )
    gz = vec(
        df,
        ["gyro_z", "gyr_z", "gz"],
    )

    # Exact historical axis/sign mapping.
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

    return pd.DataFrame({
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

        "EulerX": np.zeros(len(df)),
        "EulerY": np.zeros(len(df)),
        "EulerZ": np.zeros(len(df)),
    })


print("=" * 100)
print("PAIRED LEGACY ACQUISITION -> UNIVR DATASET")
print("=" * 100)

if TARGET.exists():
    raise SystemExit(
        f"STOP: target already exists:\n{TARGET}"
    )

if TMP.exists():
    raise SystemExit(
        f"STOP: temporary directory exists:\n{TMP}"
    )

m = pd.read_csv(MANIFEST)

assert len(m) == 338
assert set(m["scenario_id"].astype(int)) == EXPECTED_TASKS

sensors = TMP / "laboratory/sensors_data"
labels = TMP / "laboratory/labels_data"
reports = TMP / "conversion_reports"

sensors.mkdir(parents=True)
labels.mkdir(parents=True)
reports.mkdir(parents=True)

report_rows = []
label_rows = {}

rates = []
onset_errors = []
impact_errors = []
baseline_rows = []


for _, r in m.sort_values(
    ["sim_age_profile", "scenario_id"]
).iterrows():

    age = int(r["sim_age_profile"])
    task = int(r["scenario_id"])

    legacy = norm_path(
        r["legacy_csv"]
    )

    if not legacy.exists():
        raise RuntimeError(
            f"missing legacy CSV:\n{legacy}"
        )

    df = pd.read_csv(
        legacy,
        comment="#",
    )

    if df.empty:
        raise RuntimeError(
            f"empty legacy CSV: {legacy}"
        )

    t = time_vector(df)

    if len(t) < 3:
        raise RuntimeError(
            f"too few samples: {legacy}"
        )

    dt = np.diff(t)

    if np.any(dt <= 0):
        raise RuntimeError(
            f"non-monotonic time: {legacy}"
        )

    hz = 1.0 / float(
        np.median(dt)
    )

    rates.append(hz)

    if not (95 <= hz <= 105):
        raise RuntimeError(
            f"legacy export is not ~100 Hz: "
            f"age={age} task={task} hz={hz}"
        )

    onset_t = float(
        r["fall_onset_time_s"]
    )

    impact_t = float(
        r["main_impact_time_s"]
    )

    onset_i = nearest(
        t,
        onset_t,
    )

    impact_i = nearest(
        t,
        impact_t,
    )

    onset_err = abs(
        t[onset_i] - onset_t
    )

    impact_err = abs(
        t[impact_i] - impact_t
    )

    onset_errors.append(onset_err)
    impact_errors.append(impact_err)

    # A 100-Hz legacy export should be within about one sample.
    if onset_err > 0.011:
        raise RuntimeError(
            f"onset alignment >11ms "
            f"age={age} task={task}: {onset_err}"
        )

    if impact_err > 0.011:
        raise RuntimeError(
            f"impact alignment >11ms "
            f"age={age} task={task}: {impact_err}"
        )

    converted = convert_legacy(
        df,
        t,
    )

    subj = f"SA{age:02d}"

    out_csv = (
        sensors
        / subj
        / f"S{age:02d}T{task:02d}R01.csv"
    )

    out_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    converted.to_csv(
        out_csv,
        index=False,
    )

    secondary = bool(
        r["secondary_impact_analysis"]
    )

    label_rows.setdefault(
        age,
        [],
    ).append({
        "Task Code (Task ID)":
            f"F{task:02d} ({task})",

        "Description":
            f"Simulated task {task}",

        "Trial ID":
            1,

        "Fall_onset_frame":
            onset_i,

        "Fall_impact_frame":
            impact_i,

        "Impact_Validation_Status":
            r["impact_validation_status"],

        "Primary_Onset_Analysis":
            True,

        "Secondary_Impact_Analysis":
            secondary,

        "Physical_Overlap_Role":
            r["physical_overlap_role"],
    })

    report_rows.append({
        "status":
            "converted",

        "subject_id":
            age,

        "subject_folder":
            subj,

        # Semantic task ID from reconciled manifest.
        "task_id":
            task,

        "trial_id":
            1,

        "age":
            age,

        "input_csv":
            str(legacy),

        "output_csv":
            str(out_csv),

        "rows":
            len(converted),

        "source_sampling_hz":
            hz,

        "acquisition_variant":
            "legacy_30Hz_state_interpolated_to_100Hz",

        "fall_onset_time_s":
            onset_t,

        "fall_impact_time_s":
            impact_t,

        "fall_onset_frame":
            onset_i,

        "fall_impact_frame":
            impact_i,

        "fall_impact_frame_secondary":
            impact_i if secondary else np.nan,

        "onset_alignment_error_s":
            onset_err,

        "impact_alignment_error_s":
            impact_err,

        "impact_validation_status":
            r["impact_validation_status"],

        "primary_onset_analysis":
            True,

        "secondary_impact_analysis":
            secondary,

        "physical_overlap_role":
            r["physical_overlap_role"],

        "paired_pristine_highrate_truth":
            str(
                norm_path(
                    r["pristine_truth_source"]
                )
            ),
    })

    n0 = min(
        100,
        len(converted),
    )

    baseline_rows.append({
        "age":
            age,
        "task":
            task,
        "AccX_g":
            converted["AccX"]
            .iloc[:n0]
            .median() / 1000.0,
        "AccY_g":
            converted["AccY"]
            .iloc[:n0]
            .median() / 1000.0,
        "AccZ_g":
            converted["AccZ"]
            .iloc[:n0]
            .median() / 1000.0,
    })

    print(
        f"[OK] age {age:02d} "
        f"task {task:03d} -> "
        f"{out_csv.name}"
    )


for age, rows in sorted(
    label_rows.items()
):
    pd.DataFrame(
        rows
    ).to_excel(
        labels
        / f"SA{age:02d}_label.xlsx",
        index=False,
    )


report = pd.DataFrame(
    report_rows
)

report.to_csv(
    reports
    / "conversion_report_all_subjects.csv",
    index=False,
)

pd.DataFrame(
    baseline_rows
).to_csv(
    reports
    / "axis_baseline_qc.csv",
    index=False,
)


print()
print("=" * 100)
print("PAIRED LEGACY DATASET VALIDATION")
print("=" * 100)

print(
    "records:",
    len(report),
)

print(
    "age profiles:",
    report.subject_id.nunique(),
)

print(
    "semantic tasks:",
    report.task_id.nunique(),
)

print(
    "Task250:",
    int(
        (
            report.task_id
            == 250
        ).sum()
    ),
)

print(
    "onset eligible:",
    int(
        report.primary_onset_analysis.sum()
    ),
)

print(
    "impact eligible:",
    int(
        report.secondary_impact_analysis.sum()
    ),
)

print()
print("Sampling Hz:")
print(
    pd.Series(
        rates
    )
    .describe()
    .to_string()
)

print()
print(
    "max onset alignment error [s]:",
    max(onset_errors),
)

print(
    "max impact alignment error [s]:",
    max(impact_errors),
)

base = pd.DataFrame(
    baseline_rows
)

print()
print("Baseline:")
print(
    base[
        ["AccX_g","AccY_g","AccZ_g"]
    ]
    .median()
    .to_string()
)


assert len(report) == 338
assert report.subject_id.nunique() == 13
assert report.task_id.nunique() == 26
assert int((report.task_id == 250).sum()) == 13
assert report.groupby(
    ["subject_id","task_id"]
).size().eq(1).all()

assert int(
    report.primary_onset_analysis.sum()
) == 338

assert int(
    report.secondary_impact_analysis.sum()
) == 286


(TMP / "README_paired_legacy.txt").write_text(
    "\n".join([
        "Paired legacy acquisition validation dataset.",
        "",
        "Each record corresponds to the exact simulation run",
        "used for the high-rate truth dataset.",
        "",
        "Legacy acquisition:",
        "  simulator state observed at env.step (~30 Hz)",
        "  then legacy interpolation/export at ~100 Hz.",
        "",
        "Semantic task IDs and event times come from the",
        "same publication-QC manifest as the high-rate dataset.",
        "",
        "This dataset is intended for controlled acquisition",
        "A/B comparison against simulated_dataset_highrate_truth_v1.",
        "",
        "No simulator rerun was performed.",
    ]) + "\n",
    encoding="utf-8",
)


TMP.rename(
    TARGET
)


print()
print("PAIRED_LEGACY_DATASET_OK")
print("Dataset:", TARGET)
print(
    "Report:",
    TARGET
    / "conversion_reports/"
      "conversion_report_all_subjects.csv"
)
print("No simulator rerun was performed.")
