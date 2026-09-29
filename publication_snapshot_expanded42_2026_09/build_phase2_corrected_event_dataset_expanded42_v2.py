from pathlib import Path
import hashlib
import json
import math
import re

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


PROJECT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"
)

PHYSICAL_ROOT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/toqeer/"
    "uniVr-dataset"
)

DATASET_ROOTS = {
    "KFALL":
        PHYSICAL_ROOT / "KFall_oriented",

    "UNIVR":
        PHYSICAL_ROOT / "UniVrFall_Dataset",
}

FALL_TASKS = {
    "KFALL":
        set(range(20, 35)),

    "UNIVR":
        set(range(20, 35))
        | set(range(37, 43)),
}

SYNTHETIC_MANIFEST = (
    PROJECT
    / "outputs"
    / "phase2_publication_gate_expanded42_v2"
    / "final_synthetic_event_policy_v2"
    / "canonical_synthetic_event_manifest_expanded42_v2.csv"
)

OUT = (
    PROJECT
    / "outputs"
    / "phase2_corrected_event_dataset_expanded42_v2"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

FS_HZ = 100.0
WINDOW = 30
STRIDE = 15
CUTOFF_HZ = 5.0

ACC_MG_TO_MPS2 = 0.00980665
GYRO_MDPS_TO_DPS = 0.001

B, A = butter(
    1,
    CUTOFF_HZ / (0.5 * FS_HZ),
    btype="low",
)

SENSOR_RE = re.compile(
    r"^S(\d+)T(\d+)R0?(\d+)\.csv$",
    re.I,
)

SUBJECT_RE = re.compile(
    r"SA(\d+)_label",
    re.I,
)

EXPECTED = {
    "KFALL_total_csv":
        5075,

    "UNIVR_total_csv":
        1234,

    "KFALL_fall_csv":
        2346,

    "UNIVR_fall_csv":
        583,

    "physical_missing_fall_annotations":
        9,

    "physical_invalid_event_annotations":
        1,

    "synthetic_source_trials":
        396,

    "synthetic_eligible_trials":
        310,

    "synthetic_falling_windows":
        1256,

    "synthetic_activity_windows":
        18824,
}


def to_bool_series(s):
    if s.dtype == bool:
        return s

    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
            ]
        )
    )


def filter_window(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    if x.shape != (
        WINDOW,
        9,
    ):
        raise RuntimeError(
            f"Unexpected window shape: {x.shape}"
        )

    y = filtfilt(
        B,
        A,
        x,
        axis=0,
    )

    return y.astype(
        np.float32
    )


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def parse_physical_annotations():

    records = []

    for dataset, root in DATASET_ROOTS.items():

        label_dir = (
            root
            / "labels_data"
        )

        for workbook in sorted(
            label_dir.glob("*.xlsx")
        ):

            sm = SUBJECT_RE.search(
                workbook.stem
            )

            if not sm:
                continue

            subject = int(
                sm.group(1)
            )

            df = pd.read_excel(
                workbook,
                sheet_name=0,
                header=0,
            )

            required = [
                "Task Code (Task ID)",
                "Trial ID",
                "Fall_onset_frame",
                "Fall_impact_frame",
            ]

            missing = [
                c
                for c in required
                if c not in df.columns
            ]

            if missing:
                raise RuntimeError(
                    f"{workbook}: "
                    f"missing columns {missing}"
                )

            task_text = (
                df[
                    "Task Code (Task ID)"
                ]
                .ffill()
                .astype(str)
            )

            task = pd.to_numeric(
                task_text.str.extract(
                    r"\((\d+)\)",
                    expand=False,
                ),
                errors="coerce",
            )

            trial = pd.to_numeric(
                df["Trial ID"],
                errors="coerce",
            )

            onset = pd.to_numeric(
                df[
                    "Fall_onset_frame"
                ],
                errors="coerce",
            )

            impact = pd.to_numeric(
                df[
                    "Fall_impact_frame"
                ],
                errors="coerce",
            )

            for i in range(
                len(df)
            ):

                if (
                    not np.isfinite(
                        task.iloc[i]
                    )
                    or
                    not np.isfinite(
                        trial.iloc[i]
                    )
                ):
                    continue

                task_id = int(
                    task.iloc[i]
                )

                trial_id = int(
                    trial.iloc[i]
                )

                if (
                    task_id
                    not in FALL_TASKS[
                        dataset
                    ]
                ):
                    continue

                records.append({
                    "dataset":
                        dataset,

                    "subject":
                        subject,

                    "task":
                        task_id,

                    "trial":
                        trial_id,

                    "onset_frame":
                        onset.iloc[i],

                    "impact_frame":
                        impact.iloc[i],

                    "workbook":
                        str(workbook),

                    "workbook_row":
                        int(i),
                })

    ann = pd.DataFrame(
        records
    )

    key = [
        "dataset",
        "subject",
        "task",
        "trial",
    ]

    dup = ann.duplicated(
        key,
        keep=False,
    )

    if dup.any():
        raise RuntimeError(
            "Duplicate physical "
            "annotation keys detected."
        )

    return ann


def physical_file_inventory():

    records = []

    for dataset, root in DATASET_ROOTS.items():

        for path in sorted(
            root.rglob("*.csv")
        ):

            m = SENSOR_RE.match(
                path.name
            )

            if not m:
                continue

            records.append({
                "dataset":
                    dataset,

                "subject":
                    int(m.group(1)),

                "task":
                    int(m.group(2)),

                "trial":
                    int(m.group(3)),

                "filename":
                    path.name,

                "path":
                    str(path),
            })

    return pd.DataFrame(
        records
    )


def read_physical_sensor(path):

    required = [
        "FrameCounter",
        "AccX",
        "AccY",
        "AccZ",
        "GyrX",
        "GyrY",
        "GyrZ",
    ]

    q = pd.read_csv(
        path,
        usecols=required,
    )

    for c in required:
        q[c] = pd.to_numeric(
            q[c],
            errors="coerce",
        )

    if q[required].isna().any().any():
        raise RuntimeError(
            "nonfinite_or_nonnumeric_sensor_values"
        )

    frames = q[
        "FrameCounter"
    ].to_numpy(
        dtype=float
    )

    if (
        len(frames) == 0
        or
        not np.isfinite(
            frames
        ).all()
    ):
        raise RuntimeError(
            "invalid_framecounter"
        )

    acc = (
        q[
            [
                "AccX",
                "AccY",
                "AccZ",
            ]
        ]
        .to_numpy(
            dtype=float
        )
        * ACC_MG_TO_MPS2
    )

    gyro = (
        q[
            [
                "GyrX",
                "GyrY",
                "GyrZ",
            ]
        ]
        .to_numpy(
            dtype=float
        )
        * GYRO_MDPS_TO_DPS
    )

    zeros = np.zeros(
        (
            len(q),
            3,
        ),
        dtype=float,
    )

    x = np.concatenate(
        [
            acc,
            gyro,
            zeros,
        ],
        axis=1,
    )

    if not np.isfinite(
        x
    ).all():
        raise RuntimeError(
            "nonfinite_converted_sensor"
        )

    return x, frames


physical_X = []
physical_y = []
physical_meta = []

physical_skips = []
physical_sources = []

annotations = (
    parse_physical_annotations()
)

files = (
    physical_file_inventory()
)

KEY = [
    "dataset",
    "subject",
    "task",
    "trial",
]

ann_map = {}

for _, r in annotations.iterrows():

    key = (
        r["dataset"],
        int(r["subject"]),
        int(r["task"]),
        int(r["trial"]),
    )

    ann_map[key] = r


def emit_physical_segment(
    raw9,
    frames,
    row_indices,
    *,
    dataset,
    subject,
    task,
    trial,
    source_path,
    segment,
    label,
    onset_frame,
    impact_frame,
):

    row_indices = np.asarray(
        row_indices,
        dtype=int,
    )

    if len(row_indices) < WINDOW:
        return 0

    count = 0

    for start in range(
        0,
        len(row_indices)
        - WINDOW
        + 1,
        STRIDE,
    ):

        idx = row_indices[
            start:
            start + WINDOW
        ]

        window = raw9[
            idx
        ]

        filtered = filter_window(
            window
        )

        physical_X.append(
            filtered
        )

        physical_y.append(
            int(label)
        )

        physical_meta.append({
            "dataset":
                dataset,

            "group_id":
                f"{dataset}_SA"
                f"{subject:02d}",

            "subject_id":
                f"SA{subject:02d}",

            "subject_numeric":
                int(subject),

            "task_id":
                int(task),

            "trial_id":
                int(trial),

            "source_path":
                str(source_path),

            "segment":
                segment,

            "label":
                int(label),

            "label_name":
                (
                    "Falling"
                    if label == 1
                    else "Activity"
                ),

            "source_row_start":
                int(idx[0]),

            "source_row_end":
                int(idx[-1]),

            "frame_start":
                float(
                    frames[
                        idx[0]
                    ]
                ),

            "frame_end":
                float(
                    frames[
                        idx[-1]
                    ]
                ),

            "onset_frame":
                (
                    float(onset_frame)
                    if np.isfinite(
                        onset_frame
                    )
                    else np.nan
                ),

            "impact_frame":
                (
                    float(impact_frame)
                    if np.isfinite(
                        impact_frame
                    )
                    else np.nan
                ),

            "sampling_hz":
                100,

            "window_samples":
                WINDOW,

            "stride_samples":
                STRIDE,

            "filter_hz":
                CUTOFF_HZ,

            "acc_unit":
                "m/s^2",

            "gyro_unit":
                "deg/s",
        })

        count += 1

    return count


print(
    "=" * 88
)

print(
    "BUILDING PHYSICAL DATASET"
)

print(
    "=" * 88
)

fall_file_count = {
    "KFALL": 0,
    "UNIVR": 0,
}

missing_annotation_count = 0
invalid_annotation_count = 0

for n, row in files.iterrows():

    dataset = str(
        row["dataset"]
    )

    subject = int(
        row["subject"]
    )

    task = int(
        row["task"]
    )

    trial = int(
        row["trial"]
    )

    path = Path(
        row["path"]
    )

    key = (
        dataset,
        subject,
        task,
        trial,
    )

    is_fall_task = (
        task
        in FALL_TASKS[
            dataset
        ]
    )

    if is_fall_task:
        fall_file_count[
            dataset
        ] += 1

    try:
        raw9, frames = (
            read_physical_sensor(
                path
            )
        )

    except Exception as exc:

        physical_skips.append({
            "dataset":
                dataset,

            "subject":
                subject,

            "task":
                task,

            "trial":
                trial,

            "path":
                str(path),

            "reason":
                f"sensor_read_error:"
                f"{exc!r}",
        })

        continue

    activity_windows = 0
    falling_windows = 0

    if is_fall_task:

        if key not in ann_map:

            missing_annotation_count += 1

            physical_skips.append({
                "dataset":
                    dataset,

                "subject":
                    subject,

                "task":
                    task,

                "trial":
                    trial,

                "path":
                    str(path),

                "reason":
                    "missing_fall_annotation",
            })

            physical_sources.append({
                "dataset":
                    dataset,

                "group_id":
                    f"{dataset}_SA"
                    f"{subject:02d}",

                "task_id":
                    task,

                "trial_id":
                    trial,

                "source_path":
                    str(path),

                "source_type":
                    "fall_task",

                "activity_windows":
                    0,

                "falling_windows":
                    0,

                "status":
                    "skipped_missing_annotation",
            })

            continue

        a = ann_map[
            key
        ]

        onset = float(
            a["onset_frame"]
        )

        impact = float(
            a["impact_frame"]
        )

        if (
            not np.isfinite(onset)
            or
            not np.isfinite(impact)
            or
            impact <= onset
        ):

            invalid_annotation_count += 1

            physical_skips.append({
                "dataset":
                    dataset,

                "subject":
                    subject,

                "task":
                    task,

                "trial":
                    trial,

                "path":
                    str(path),

                "reason":
                    "invalid_fall_annotation:"
                    "impact_not_after_onset",

                "onset_frame":
                    onset,

                "impact_frame":
                    impact,
            })

            physical_sources.append({
                "dataset":
                    dataset,

                "group_id":
                    f"{dataset}_SA"
                    f"{subject:02d}",

                "task_id":
                    task,

                "trial_id":
                    trial,

                "source_path":
                    str(path),

                "source_type":
                    "fall_task",

                "activity_windows":
                    0,

                "falling_windows":
                    0,

                "status":
                    "skipped_invalid_annotation",
            })

            continue

        activity_idx = np.flatnonzero(
            frames < onset
        )

        falling_idx = np.flatnonzero(
            (frames >= onset)
            &
            (frames < impact)
        )

        activity_windows = (
            emit_physical_segment(
                raw9,
                frames,
                activity_idx,
                dataset=dataset,
                subject=subject,
                task=task,
                trial=trial,
                source_path=path,
                segment="pre_onset_activity",
                label=0,
                onset_frame=onset,
                impact_frame=impact,
            )
        )

        falling_windows = (
            emit_physical_segment(
                raw9,
                frames,
                falling_idx,
                dataset=dataset,
                subject=subject,
                task=task,
                trial=trial,
                source_path=path,
                segment="falling_preimpact",
                label=1,
                onset_frame=onset,
                impact_frame=impact,
            )
        )

        status = (
            "used_annotated_fall"
        )

    else:

        all_idx = np.arange(
            len(frames),
            dtype=int,
        )

        activity_windows = (
            emit_physical_segment(
                raw9,
                frames,
                all_idx,
                dataset=dataset,
                subject=subject,
                task=task,
                trial=trial,
                source_path=path,
                segment="full_activity",
                label=0,
                onset_frame=np.nan,
                impact_frame=np.nan,
            )
        )

        falling_windows = 0

        status = (
            "used_activity"
        )

    physical_sources.append({
        "dataset":
            dataset,

        "group_id":
            f"{dataset}_SA"
            f"{subject:02d}",

        "task_id":
            task,

        "trial_id":
            trial,

        "source_path":
            str(path),

        "source_type":
            (
                "fall_task"
                if is_fall_task
                else "activity_task"
            ),

        "activity_windows":
            int(
                activity_windows
            ),

        "falling_windows":
            int(
                falling_windows
            ),

        "status":
            status,
    })

    if (
        (n + 1) % 500
        == 0
    ):
        print(
            f"Physical files processed: "
            f"{n + 1} / {len(files)}"
        )


if not physical_X:
    raise RuntimeError(
        "No physical windows generated."
    )

physical_X = np.stack(
    physical_X
).astype(
    np.float32
)

physical_y = np.asarray(
    physical_y,
    dtype=np.int64,
)

physical_meta = pd.DataFrame(
    physical_meta
)

physical_skips = pd.DataFrame(
    physical_skips
)

physical_sources = pd.DataFrame(
    physical_sources
)


print()
print(
    "=" * 88
)

print(
    "BUILDING SYNTHETIC DATASET"
)

print(
    "=" * 88
)

if not SYNTHETIC_MANIFEST.exists():
    raise RuntimeError(
        f"Missing synthetic manifest: "
        f"{SYNTHETIC_MANIFEST}"
    )

syn_manifest = pd.read_csv(
    SYNTHETIC_MANIFEST
)

syn_eligible = to_bool_series(
    syn_manifest[
        "cnn_fall_eligible"
    ]
)

# ==================================================================
# EXPANDED-42 V2 MANIFEST GATE
#
# The old publication builder had exact synthetic counts hard-coded
# for the frozen 22-profile cohort. For V2, the raw source dimensions
# are fixed (42 profiles x 18 tasks = 756), but eligibility and window
# counts are deliberately MEASURED rather than guessed.
#
# The canonical V2 manifest is authoritative for those measured counts.
# ==================================================================

if len(syn_manifest) != 756:
    raise RuntimeError(
        f"Expanded42 manifest must contain exactly 756 source trials; "
        f"got {len(syn_manifest)}"
    )

if syn_manifest["profile_id"].astype(str).nunique() != 42:
    raise RuntimeError(
        "Expanded42 manifest must contain exactly 42 profiles"
    )

if pd.to_numeric(
    syn_manifest["task_id"],
    errors="raise",
).astype(int).nunique() != 18:
    raise RuntimeError(
        "Expanded42 manifest must contain exactly 18 tasks"
    )

eligible_rows_v2 = syn_manifest.loc[syn_eligible].copy()

EXPECTED["synthetic_source_trials"] = int(len(syn_manifest))

EXPECTED["synthetic_eligible_trials"] = int(
    syn_eligible.sum()
)

EXPECTED["synthetic_falling_windows"] = int(
    pd.to_numeric(
        eligible_rows_v2["raw_falling_window_count"],
        errors="raise",
    ).sum()
)

EXPECTED["synthetic_activity_windows"] = int(
    pd.to_numeric(
        eligible_rows_v2["raw_activity_window_count"],
        errors="raise",
    ).sum()
)

print(
    "Expanded42 measured synthetic expectations:",
    {
        "source_trials": EXPECTED["synthetic_source_trials"],
        "eligible_trials": EXPECTED["synthetic_eligible_trials"],
        "activity_windows": EXPECTED["synthetic_activity_windows"],
        "falling_windows": EXPECTED["synthetic_falling_windows"],
    },
)

synthetic_X = []
synthetic_y = []
synthetic_meta = []
synthetic_sources = []
synthetic_skips = []


def emit_synthetic_segment(
    raw9,
    t,
    row_indices,
    *,
    profile,
    task,
    source_path,
    segment,
    label,
    onset,
    impact,
    original_impact,
):

    row_indices = np.asarray(
        row_indices,
        dtype=int,
    )

    if len(row_indices) < WINDOW:
        return 0

    count = 0

    for start in range(
        0,
        len(row_indices)
        - WINDOW
        + 1,
        STRIDE,
    ):

        idx = row_indices[
            start:
            start + WINDOW
        ]

        filtered = filter_window(
            raw9[
                idx
            ]
        )

        synthetic_X.append(
            filtered
        )

        synthetic_y.append(
            int(label)
        )

        synthetic_meta.append({
            "dataset":
                "SIM",

            "group_id":
                f"SIM_{profile}",

            "profile_id":
                profile,

            "task_id":
                int(task),

            "source_path":
                str(source_path),

            "segment":
                segment,

            "label":
                int(label),

            "label_name":
                (
                    "Falling"
                    if label == 1
                    else "Activity"
                ),

            "source_row_start":
                int(idx[0]),

            "source_row_end":
                int(idx[-1]),

            "window_start_time_s":
                float(
                    t[
                        idx[0]
                    ]
                ),

            "window_end_time_s":
                float(
                    t[
                        idx[-1]
                    ]
                ),

            "fall_onset_time_s":
                float(onset),

            "cnn_recovered_impact_time_s":
                float(impact),

            "original_manifest_impact_time_s":
                float(original_impact),

            "sampling_hz":
                100,

            "window_samples":
                WINDOW,

            "stride_samples":
                STRIDE,

            "filter_hz":
                CUTOFF_HZ,

            "acc_unit":
                "m/s^2",

            "gyro_unit":
                "deg/s",

            "axis_policy":
                "acc=[y,x,z];gyro=[y,x,z]",
        })

        count += 1

    return count


eligible_manifest = (
    syn_manifest[
        syn_eligible
    ]
    .sort_values(
        [
            "profile_id",
            "task_id",
        ]
    )
)

for _, row in eligible_manifest.iterrows():

    profile = str(
        row["profile_id"]
    )

    task = int(
        row["task_id"]
    )

    path = Path(
        str(
            row["truth_csv"]
        )
    )

    onset = float(
        row[
            "fall_onset_time_s"
        ]
    )

    impact = float(
        row[
            "cnn_recovered_impact_time_s"
        ]
    )

    original_impact = float(
        row[
            "original_manifest_impact_time_s"
        ]
    )

    expected_fall = int(
        row[
            "raw_falling_window_count"
        ]
    )

    expected_activity = int(
        row[
            "raw_activity_window_count"
        ]
    )

    try:
        q = pd.read_csv(
            path,
            comment="#",
            usecols=[
                "timestamp",
                "accel_true_x",
                "accel_true_y",
                "accel_true_z",
                "gyro_true_x",
                "gyro_true_y",
                "gyro_true_z",
            ],
        )

        vals = q.to_numpy(
            dtype=float
        )

        if not np.isfinite(
            vals
        ).all():
            raise RuntimeError(
                "nonfinite_truth_values"
            )

        t = q[
            "timestamp"
        ].to_numpy(
            dtype=float
        )

        acc = np.column_stack(
            [
                q[
                    "accel_true_y"
                ].to_numpy(float),

                q[
                    "accel_true_x"
                ].to_numpy(float),

                q[
                    "accel_true_z"
                ].to_numpy(float),
            ]
        )

        gyro = np.column_stack(
            [
                q[
                    "gyro_true_y"
                ].to_numpy(float),

                q[
                    "gyro_true_x"
                ].to_numpy(float),

                q[
                    "gyro_true_z"
                ].to_numpy(float),
            ]
        )

        gyro = np.degrees(
            gyro
        )

        zeros = np.zeros(
            (
                len(q),
                3,
            ),
            dtype=float,
        )

        raw9 = np.concatenate(
            [
                acc,
                gyro,
                zeros,
            ],
            axis=1,
        )

        activity_idx = np.flatnonzero(
            t < onset
        )

        falling_idx = np.flatnonzero(
            (t >= onset)
            &
            (t < impact)
        )

        activity_count = (
            emit_synthetic_segment(
                raw9,
                t,
                activity_idx,
                profile=profile,
                task=task,
                source_path=path,
                segment="pre_onset_activity",
                label=0,
                onset=onset,
                impact=impact,
                original_impact=original_impact,
            )
        )

        fall_count = (
            emit_synthetic_segment(
                raw9,
                t,
                falling_idx,
                profile=profile,
                task=task,
                source_path=path,
                segment="falling_preimpact",
                label=1,
                onset=onset,
                impact=impact,
                original_impact=original_impact,
            )
        )

        if (
            activity_count
            != expected_activity
        ):
            raise RuntimeError(
                f"activity_window_count_mismatch:"
                f"generated={activity_count},"
                f"expected={expected_activity}"
            )

        if (
            fall_count
            != expected_fall
        ):
            raise RuntimeError(
                f"fall_window_count_mismatch:"
                f"generated={fall_count},"
                f"expected={expected_fall}"
            )

        synthetic_sources.append({
            "profile_id":
                profile,

            "task_id":
                task,

            "source_path":
                str(path),

            "activity_windows":
                int(
                    activity_count
                ),

            "falling_windows":
                int(
                    fall_count
                ),

            "status":
                "used",
        })

    except Exception as exc:

        synthetic_skips.append({
            "profile_id":
                profile,

            "task_id":
                task,

            "source_path":
                str(path),

            "reason":
                repr(exc),
        })


if synthetic_skips:
    raise RuntimeError(
        "Synthetic dataset generation had "
        f"{len(synthetic_skips)} failures. "
        "Inspect synthetic_skips.csv."
    )

synthetic_X = np.stack(
    synthetic_X
).astype(
    np.float32
)

synthetic_y = np.asarray(
    synthetic_y,
    dtype=np.int64,
)

synthetic_meta = pd.DataFrame(
    synthetic_meta
)

synthetic_sources = pd.DataFrame(
    synthetic_sources
)

synthetic_skips = pd.DataFrame(
    synthetic_skips
)


print()
print(
    "=" * 88
)

print(
    "SAVING DATASET ARTIFACTS"
)

print(
    "=" * 88
)

np.save(
    OUT / "physical_X.npy",
    physical_X,
)

np.save(
    OUT / "physical_y.npy",
    physical_y,
)

physical_meta.to_csv(
    OUT / "physical_metadata.csv",
    index=False,
)

physical_sources.to_csv(
    OUT / "physical_source_summary.csv",
    index=False,
)

physical_skips.to_csv(
    OUT / "physical_skips.csv",
    index=False,
)

np.save(
    OUT / "synthetic_X.npy",
    synthetic_X,
)

np.save(
    OUT / "synthetic_y.npy",
    synthetic_y,
)

synthetic_meta.to_csv(
    OUT / "synthetic_metadata.csv",
    index=False,
)

synthetic_sources.to_csv(
    OUT / "synthetic_source_summary.csv",
    index=False,
)

synthetic_skips.to_csv(
    OUT / "synthetic_skips.csv",
    index=False,
)

np.save(
    OUT / "synthetic_activity_indices.npy",
    np.flatnonzero(
        synthetic_y == 0
    ).astype(
        np.int64
    ),
)

np.save(
    OUT / "synthetic_falling_indices.npy",
    np.flatnonzero(
        synthetic_y == 1
    ).astype(
        np.int64
    ),
)


physical_summary = (
    physical_meta
    .groupby(
        [
            "dataset",
            "label_name",
            "segment",
        ],
        as_index=False,
    )
    .size()
    .rename(
        columns={
            "size":
                "windows"
        }
    )
)

physical_summary.to_csv(
    OUT / "physical_window_summary.csv",
    index=False,
)

physical_subject_summary = (
    physical_meta
    .groupby(
        [
            "dataset",
            "group_id",
            "label_name",
        ],
        as_index=False,
    )
    .size()
    .rename(
        columns={
            "size":
                "windows"
        }
    )
)

physical_subject_summary.to_csv(
    OUT / "physical_subject_summary.csv",
    index=False,
)

synthetic_profile_summary = (
    synthetic_meta
    .groupby(
        [
            "profile_id",
            "label_name",
        ],
        as_index=False,
    )
    .size()
    .rename(
        columns={
            "size":
                "windows"
        }
    )
)

synthetic_profile_summary.to_csv(
    OUT / "synthetic_profile_summary.csv",
    index=False,
)

synthetic_task_summary = (
    synthetic_meta
    .groupby(
        [
            "task_id",
            "label_name",
        ],
        as_index=False,
    )
    .size()
    .rename(
        columns={
            "size":
                "windows"
        }
    )
)

synthetic_task_summary.to_csv(
    OUT / "synthetic_task_summary.csv",
    index=False,
)


physical_counts = (
    files.groupby(
        "dataset"
    )
    .size()
    .to_dict()
)

physical_fall_counts = {}

for dataset in [
    "KFALL",
    "UNIVR",
]:
    physical_fall_counts[
        dataset
    ] = int(
        files[
            (files["dataset"] == dataset)
            &
            files["task"].isin(
                FALL_TASKS[
                    dataset
                ]
            )
        ].shape[0]
    )


checks = {
    "physical_shape_Nx30x9":
        (
            physical_X.ndim == 3
            and
            physical_X.shape[1:]
            == (30, 9)
        ),

    "synthetic_shape_Nx30x9":
        (
            synthetic_X.ndim == 3
            and
            synthetic_X.shape[1:]
            == (30, 9)
        ),

    "physical_labels_match":
        (
            len(physical_X)
            ==
            len(physical_y)
            ==
            len(physical_meta)
        ),

    "synthetic_labels_match":
        (
            len(synthetic_X)
            ==
            len(synthetic_y)
            ==
            len(synthetic_meta)
        ),

    "physical_all_finite":
        bool(
            np.isfinite(
                physical_X
            ).all()
        ),

    "synthetic_all_finite":
        bool(
            np.isfinite(
                synthetic_X
            ).all()
        ),

    "physical_binary_labels":
        set(
            np.unique(
                physical_y
            ).tolist()
        )
        <= {0, 1},

    "synthetic_binary_labels":
        set(
            np.unique(
                synthetic_y
            ).tolist()
        )
        == {0, 1},

    "physical_group_namespaced":
        physical_meta[
            "group_id"
        ].str.startswith(
            (
                "KFALL_",
                "UNIVR_",
            )
        ).all(),

    "kfall_total_files_5075":
        int(
            physical_counts.get(
                "KFALL",
                0,
            )
        )
        ==
        EXPECTED[
            "KFALL_total_csv"
        ],

    "univr_total_files_1234":
        int(
            physical_counts.get(
                "UNIVR",
                0,
            )
        )
        ==
        EXPECTED[
            "UNIVR_total_csv"
        ],

    "kfall_fall_files_2346":
        physical_fall_counts[
            "KFALL"
        ]
        ==
        EXPECTED[
            "KFALL_fall_csv"
        ],

    "univr_fall_files_583":
        physical_fall_counts[
            "UNIVR"
        ]
        ==
        EXPECTED[
            "UNIVR_fall_csv"
        ],

    "physical_missing_annotations_9":
        missing_annotation_count
        ==
        EXPECTED[
            "physical_missing_fall_annotations"
        ],

    "physical_invalid_annotations_1":
        invalid_annotation_count
        ==
        EXPECTED[
            "physical_invalid_event_annotations"
        ],

    "kfall_subjects_32":
        physical_meta.loc[
            physical_meta[
                "dataset"
            ] == "KFALL",
            "group_id",
        ].nunique()
        == 32,

    "univr_subjects_29":
        physical_meta.loc[
            physical_meta[
                "dataset"
            ] == "UNIVR",
            "group_id",
        ].nunique()
        == 29,

    "synthetic_manifest_rows_match_expanded42":
        len(
            syn_manifest
        )
        ==
        EXPECTED[
            "synthetic_source_trials"
        ],

    "synthetic_eligible_trials_match_expanded42":
        len(
            eligible_manifest
        )
        ==
        EXPECTED[
            "synthetic_eligible_trials"
        ],

    "synthetic_profiles_42":
        synthetic_meta[
            "profile_id"
        ].nunique()
        == 42,

    "synthetic_tasks_18":
        synthetic_meta[
            "task_id"
        ].nunique()
        == 18,

    "synthetic_falling_windows_match_manifest":
        int(
            (
                synthetic_y == 1
            ).sum()
        )
        ==
        EXPECTED[
            "synthetic_falling_windows"
        ],

    "synthetic_activity_windows_match_manifest":
        int(
            (
                synthetic_y == 0
            ).sum()
        )
        ==
        EXPECTED[
            "synthetic_activity_windows"
        ],

    "physical_has_activity_and_falling":
        set(
            np.unique(
                physical_y
            ).tolist()
        )
        == {0, 1},
}


facts = {
    "dataset_version":
        "phase2_corrected_event_dataset_expanded42_v2",

    "physical_total_windows":
        int(
            len(
                physical_y
            )
        ),

    "physical_activity_windows":
        int(
            (
                physical_y == 0
            ).sum()
        ),

    "physical_falling_windows":
        int(
            (
                physical_y == 1
            ).sum()
        ),

    "physical_kfall_windows":
        int(
            (
                physical_meta[
                    "dataset"
                ] == "KFALL"
            ).sum()
        ),

    "physical_univr_windows":
        int(
            (
                physical_meta[
                    "dataset"
                ] == "UNIVR"
            ).sum()
        ),

    "physical_groups":
        int(
            physical_meta[
                "group_id"
            ].nunique()
        ),

    "physical_missing_fall_annotations":
        int(
            missing_annotation_count
        ),

    "physical_invalid_event_annotations":
        int(
            invalid_annotation_count
        ),

    "synthetic_total_windows":
        int(
            len(
                synthetic_y
            )
        ),

    "synthetic_activity_windows":
        int(
            (
                synthetic_y == 0
            ).sum()
        ),

    "synthetic_falling_windows":
        int(
            (
                synthetic_y == 1
            ).sum()
        ),

    "synthetic_profiles":
        int(
            synthetic_meta[
                "profile_id"
            ].nunique()
        ),

    "synthetic_tasks":
        int(
            synthetic_meta[
                "task_id"
            ].nunique()
        ),

    "physical_units": {
        "source_acc":
            "mg",

        "cnn_acc":
            "m/s^2",

        "acc_conversion":
            "x * 0.00980665",

        "source_gyro":
            "mdps",

        "cnn_gyro":
            "deg/s",

        "gyro_conversion":
            "x * 0.001",
    },

    "synthetic_units": {
        "source_acc":
            "m/s^2",

        "cnn_acc":
            "m/s^2",

        "source_gyro":
            "rad/s",

        "cnn_gyro":
            "deg/s",

        "gyro_conversion":
            "rad/s * 180/pi",

        "axis_mapping":
            "acc=[y,x,z], gyro=[y,x,z]",
    },

    "windowing": {
        "sampling_hz":
            100,

        "window_samples":
            30,

        "stride_samples":
            15,

        "overlap_percent":
            50,

        "lowpass_hz":
            5,

        "filter":
            "first-order Butterworth + filtfilt per window/channel",
    },

    "physical_label_policy": {
        "annotated_fall_pre_onset":
            "Activity=0",

        "annotated_fall_onset_to_impact":
            "Falling=1",

        "postimpact":
            "excluded",

        "known_fall_without_annotation":
            "skip",

        "nonfall_task":
            "full Activity=0",

        "event_coordinate":
            "FrameCounter",
    },

    "synthetic_label_policy": {
        "activity":
            "pre-onset from CNN-eligible trials only",

        "falling":
            "onset inclusive to recovered impact exclusive",

        "impact_policy":
            "peak impact_magnitude within 3 s after fall onset",
    },

    "checks": {
        k: bool(v)
        for k, v
        in checks.items()
    },

    "gate_pass":
        bool(
            all(
                checks.values()
            )
        ),
}


with open(
    OUT
    / "dataset_facts.json",
    "w",
) as f:

    json.dump(
        facts,
        f,
        indent=2,
    )


hash_targets = [
    "physical_X.npy",
    "physical_y.npy",
    "physical_metadata.csv",
    "synthetic_X.npy",
    "synthetic_y.npy",
    "synthetic_metadata.csv",
    "dataset_facts.json",
]

with open(
    OUT / "SHA256SUMS.txt",
    "w",
) as f:

    for name in hash_targets:

        path = (
            OUT / name
        )

        f.write(
            f"{sha256_file(path)}  "
            f"{name}\n"
        )


print()
print(
    "=" * 88
)

print(
    "FINAL CORRECTED DATASET FACTS"
)

print(
    "=" * 88
)

print(
    "Physical windows :",
    len(
        physical_y
    )
)

print(
    "  Activity        :",
    int(
        (
            physical_y == 0
        ).sum()
    )
)

print(
    "  Falling         :",
    int(
        (
            physical_y == 1
        ).sum()
    )
)

print(
    "  KFall windows   :",
    int(
        (
            physical_meta[
                "dataset"
            ] == "KFALL"
        ).sum()
    )
)

print(
    "  UniVR windows   :",
    int(
        (
            physical_meta[
                "dataset"
            ] == "UNIVR"
        ).sum()
    )
)

print(
    "  Subject groups  :",
    physical_meta[
        "group_id"
    ].nunique()
)

print()
print(
    "Synthetic windows:",
    len(
        synthetic_y
    )
)

print(
    "  Activity        :",
    int(
        (
            synthetic_y == 0
        ).sum()
    )
)

print(
    "  Falling         :",
    int(
        (
            synthetic_y == 1
        ).sum()
    )
)

print(
    "  Profiles        :",
    synthetic_meta[
        "profile_id"
    ].nunique()
)

print(
    "  Tasks           :",
    synthetic_meta[
        "task_id"
    ].nunique()
)

print()
print(
    "=" * 88
)

print(
    "PHYSICAL WINDOW SUMMARY"
)

print(
    "=" * 88
)

print(
    physical_summary.to_string(
        index=False
    )
)

print()
print(
    "=" * 88
)

print(
    "GATE CHECKS"
)

print(
    "=" * 88
)

for k, v in checks.items():

    print(
        f"{k:46s}",
        "PASS"
        if v
        else "FAIL"
    )

print()
print(
    "=" * 88
)

if all(
    checks.values()
):

    print(
        "FINAL CORRECTED DATASET GATE: PASS"
    )

else:

    print(
        "FINAL CORRECTED DATASET GATE: INVESTIGATE"
    )

print(
    "=" * 88
)

print()
print(
    "Dataset directory:"
)

print(
    OUT
)
