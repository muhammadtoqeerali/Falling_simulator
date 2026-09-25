from pathlib import Path
import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    KFold,
    train_test_split,
)

ROOT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"
)

PHYSICAL_SOURCE = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/toqeer/"
    "Protechto_master/data/dataset/segments/"
    "300ms_50ov_npseg_filt_binary"
)

AUDIT_PATH = (
    ROOT
    / "outputs/protechto_300ms_150ms_audit.csv"
)

SYN_DATA = (
    ROOT
    / "outputs/phase2_corrected_event_dataset_v1"
)

OUT = (
    ROOT
    / "outputs/phase2_lightweight_paper_aligned_inputs_v1"
)

PHYSICAL_OUT = (
    OUT
    / "physical_segments_300ms_50ov_preimpact150"
)

SPLIT_OUT = (
    OUT
    / "split_manifests"
)

DRAW_OUT = (
    OUT
    / "synthetic_draw_manifests"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

PHYSICAL_OUT.mkdir(
    parents=True,
    exist_ok=True,
)

SPLIT_OUT.mkdir(
    parents=True,
    exist_ok=True,
)

DRAW_OUT.mkdir(
    parents=True,
    exist_ok=True,
)


def norm_id(x):
    return str(int(float(x)))


def make_link(src, dst):
    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    os.symlink(
        str(src.resolve()),
        str(dst),
    )


print("=" * 100)
print("1. LOAD 150-ms AUDIT")
print("=" * 100)

audit = pd.read_csv(
    AUDIT_PATH
)

required = {
    "subject",
    "task",
    "trial",
    "observed",
    "expected_cut150",
    "status",
}

missing = required - set(
    audit.columns
)

if missing:
    raise RuntimeError(
        f"Audit columns missing: {missing}"
    )

audit = audit.copy()

audit["subject_key"] = (
    audit["subject"]
    .map(norm_id)
)

audit["task_key"] = (
    audit["task"]
    .map(norm_id)
)

audit["trial_key"] = (
    audit["trial"]
    .map(norm_id)
)

keys = [
    "subject_key",
    "task_key",
    "trial_key",
]

if audit.duplicated(keys).any():
    raise RuntimeError(
        "Duplicate subject/task/trial rows "
        "in 150-ms audit."
    )

audit_map = {
    (
        row.subject_key,
        row.task_key,
        row.trial_key,
    ): row
    for row in audit.itertuples()
}

print(
    "Audit trials:",
    len(audit_map),
)


print()
print("=" * 100)
print("2. BUILD NON-DESTRUCTIVE PHYSICAL 150-ms VIEW")
print("=" * 100)

trial_rows = []

for subject_dir in sorted(
    p
    for p in PHYSICAL_SOURCE.iterdir()
    if p.is_dir()
):

    subject = subject_dir.name

    (
        PHYSICAL_OUT
        / subject
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    for task_dir in sorted(
        p
        for p in subject_dir.iterdir()
        if p.is_dir()
    ):

        for trial_dir in sorted(
            p
            for p in task_dir.iterdir()
            if p.is_dir()
        ):

            sp = (
                trial_dir
                / "segments.npy"
            )

            lp = (
                trial_dir
                / "labels.npy"
            )

            if not (
                sp.exists()
                and lp.exists()
            ):
                raise RuntimeError(
                    f"Missing trial arrays: {trial_dir}"
                )

            y = np.load(
                lp,
                allow_pickle=True,
            ).astype(str)

            n_activity = int(
                np.sum(
                    y == "Activity"
                )
            )

            n_falling = int(
                np.sum(
                    y == "Falling"
                )
            )

            task = norm_id(
                task_dir.name
            )

            trial = norm_id(
                trial_dir.name
            )

            key = (
                norm_id(subject),
                task,
                trial,
            )

            target_dir = (
                PHYSICAL_OUT
                / subject
                / task_dir.name
                / trial_dir.name
            )

            if key in audit_map:

                row = audit_map[key]

                target_falling = int(
                    float(
                        row.expected_cut150
                    )
                )

                if target_falling < 0:
                    raise RuntimeError(
                        f"Negative target: {key}"
                    )

                if n_falling < target_falling:
                    raise RuntimeError(
                        f"Stored Falling count smaller "
                        f"than target for {key}: "
                        f"{n_falling} < {target_falling}"
                    )

                if target_falling == 0:

                    action = (
                        "skip_no_valid_preimpact_window"
                    )

                    kept_activity = 0
                    kept_falling = 0
                    kept_windows = 0

                else:

                    fall_pos = np.flatnonzero(
                        y == "Falling"
                    )

                    keep = (
                        y != "Falling"
                    )

                    keep[
                        fall_pos[
                            :target_falling
                        ]
                    ] = True

                    x = np.load(
                        sp,
                        allow_pickle=True,
                    )

                    if len(x) != len(y):
                        raise RuntimeError(
                            f"X/Y mismatch: {trial_dir}"
                        )

                    target_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    np.save(
                        target_dir
                        / "segments.npy",
                        x[keep],
                    )

                    np.save(
                        target_dir
                        / "labels.npy",
                        y[keep],
                    )

                    kept_activity = int(
                        np.sum(
                            y[keep]
                            == "Activity"
                        )
                    )

                    kept_falling = int(
                        np.sum(
                            y[keep]
                            == "Falling"
                        )
                    )

                    kept_windows = int(
                        np.sum(keep)
                    )

                    if (
                        n_falling
                        > target_falling
                    ):
                        action = (
                            "trimmed_to_preimpact150"
                        )
                    else:
                        action = (
                            "already_preimpact150"
                        )

            else:

                if n_falling != 0:
                    raise RuntimeError(
                        "Found unmapped Falling trial: "
                        f"{key}"
                    )

                make_link(
                    sp,
                    target_dir
                    / "segments.npy",
                )

                make_link(
                    lp,
                    target_dir
                    / "labels.npy",
                )

                action = (
                    "activity_symlink"
                )

                kept_activity = (
                    n_activity
                )

                kept_falling = 0

                kept_windows = len(y)

            trial_rows.append(
                {
                    "subject_id":
                        subject,
                    "task_id":
                        task,
                    "trial_id":
                        trial,
                    "source_dir":
                        str(trial_dir),
                    "output_dir":
                        (
                            str(target_dir)
                            if kept_windows
                            else ""
                        ),
                    "original_windows":
                        len(y),
                    "original_activity":
                        n_activity,
                    "original_falling":
                        n_falling,
                    "kept_windows":
                        kept_windows,
                    "kept_activity":
                        kept_activity,
                    "kept_falling":
                        kept_falling,
                    "action":
                        action,
                }
            )


trials = pd.DataFrame(
    trial_rows
)

trials.to_csv(
    OUT
    / "physical_trial_preimpact150_manifest.csv",
    index=False,
)

print(
    trials["action"]
    .value_counts()
    .to_string()
)

print()
print(
    "Original Activity:",
    int(
        trials[
            "original_activity"
        ].sum()
    ),
)

print(
    "Original Falling :",
    int(
        trials[
            "original_falling"
        ].sum()
    ),
)

print(
    "Aligned Activity :",
    int(
        trials[
            "kept_activity"
        ].sum()
    ),
)

print(
    "Aligned Falling  :",
    int(
        trials[
            "kept_falling"
        ].sum()
    ),
)


print()
print("=" * 100)
print("3. EXACT ORIGINAL PROTECHTO 5-FOLD SUBJECT SPLIT")
print("=" * 100)

all_subjects = sorted(
    p.name
    for p in PHYSICAL_OUT.iterdir()
    if p.is_dir()
)

excluded = {
    "999",
    "1000",
}

subjects = np.array(
    [
        s
        for s in all_subjects
        if s not in excluded
    ]
)

print(
    "All subject directories:",
    len(all_subjects),
)

print(
    "Excluded augmentation subjects:",
    sorted(excluded),
)

print(
    "KFold subjects:",
    len(subjects),
)

if len(all_subjects) != 73:
    raise RuntimeError(
        f"Expected 73 directories, "
        f"got {len(all_subjects)}"
    )

if len(subjects) != 71:
    raise RuntimeError(
        f"Expected 71 KFold subjects, "
        f"got {len(subjects)}"
    )

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

assignment_rows = []

for fold, (
    trainval_idx,
    test_idx,
) in enumerate(
    kf.split(subjects)
):

    train_idx, val_idx = (
        train_test_split(
            trainval_idx,
            test_size=0.2,
            random_state=42,
        )
    )

    role_sets = {
        "train":
            subjects[
                train_idx
            ],
        "val":
            subjects[
                val_idx
            ],
        "test":
            subjects[
                test_idx
            ],
    }

    seen = set()

    for role, vals in (
        role_sets.items()
    ):

        for subject in vals:

            if subject in seen:
                raise RuntimeError(
                    "Subject overlap in "
                    f"fold {fold}: "
                    f"{subject}"
                )

            seen.add(subject)

            assignment_rows.append(
                {
                    "outer_fold":
                        fold,
                    "subject_id":
                        subject,
                    "role":
                        role,
                }
            )

    if seen != set(subjects):
        raise RuntimeError(
            f"Incomplete fold {fold}"
        )

    print(
        f"fold={fold}",
        f"train={len(role_sets['train'])}",
        f"val={len(role_sets['val'])}",
        f"test={len(role_sets['test'])}",
    )


assignments = pd.DataFrame(
    assignment_rows
)

assignments.to_csv(
    SPLIT_OUT
    / "physical_subject_kfold_assignments.csv",
    index=False,
)


print()
print("=" * 100)
print("4. FOLD WINDOW COUNTS AFTER 150-ms ALIGNMENT")
print("=" * 100)

fold_rows = []

eligible_trials = trials[
    trials[
        "subject_id"
    ].isin(
        subjects
    )
].copy()

for fold in range(5):

    a = assignments[
        assignments[
            "outer_fold"
        ] == fold
    ]

    role_map = dict(
        zip(
            a[
                "subject_id"
            ],
            a[
                "role"
            ],
        )
    )

    m = (
        eligible_trials
        .copy()
    )

    m["role"] = (
        m[
            "subject_id"
        ].map(
            role_map
        )
    )

    if m[
        "role"
    ].isna().any():
        raise RuntimeError(
            f"Missing role fold {fold}"
        )

    for role in [
        "train",
        "val",
        "test",
    ]:

        q = m[
            m["role"]
            == role
        ]

        fold_rows.append(
            {
                "fold":
                    fold,
                "role":
                    role,
                "subjects":
                    q[
                        "subject_id"
                    ].nunique(),
                "trials":
                    int(
                        (
                            q[
                                "kept_windows"
                            ] > 0
                        ).sum()
                    ),
                "windows":
                    int(
                        q[
                            "kept_windows"
                        ].sum()
                    ),
                "activity_windows":
                    int(
                        q[
                            "kept_activity"
                        ].sum()
                    ),
                "falling_windows":
                    int(
                        q[
                            "kept_falling"
                        ].sum()
                    ),
            }
        )


fold_summary = pd.DataFrame(
    fold_rows
)

fold_summary.to_csv(
    SPLIT_OUT
    / "physical_fold_window_summary.csv",
    index=False,
)

print(
    fold_summary.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("5. BUILD SYNTHETIC FALLING POOL WITH SAME 150-ms RULE")
print("=" * 100)

syn_meta = pd.read_csv(
    SYN_DATA
    / "synthetic_metadata.csv",
    low_memory=False,
)

syn_y = np.load(
    SYN_DATA
    / "synthetic_y.npy",
    mmap_mode="r",
)

syn_x = np.load(
    SYN_DATA
    / "synthetic_X.npy",
    mmap_mode="r",
)

if not (
    len(syn_meta)
    == len(syn_y)
    == len(syn_x)
):
    raise RuntimeError(
        "Synthetic X/Y/metadata mismatch"
    )

fall_indices = np.flatnonzero(
    np.asarray(
        syn_y
    ) == 1
)

fall_meta = (
    syn_meta
    .iloc[
        fall_indices
    ]
    .copy()
)

fall_meta[
    "original_index"
] = fall_indices

keep_original_indices = []

source_rows = []

for source_path, g in (
    fall_meta
    .groupby(
        "source_path",
        sort=True,
    )
):

    g = g.sort_values(
        [
            "source_row_start",
            "source_row_end",
        ]
    )

    original_count = len(g)

    target_count = max(
        0,
        original_count - 1,
    )

    kept = (
        g.iloc[
            :target_count
        ]
    )

    keep_original_indices.extend(
        kept[
            "original_index"
        ].astype(int).tolist()
    )

    source_rows.append(
        {
            "source_path":
                source_path,
            "profile_id":
                g.iloc[0][
                    "profile_id"
                ],
            "task_id":
                g.iloc[0][
                    "task_id"
                ],
            "original_falling":
                original_count,
            "preimpact150_falling":
                target_count,
            "dropped_final_window":
                int(
                    original_count > 0
                ),
        }
    )


keep_original_indices = np.array(
    keep_original_indices,
    dtype=np.int64,
)

syn_pool_x = np.asarray(
    syn_x[
        keep_original_indices
    ],
    dtype=np.float32,
)

syn_pool_meta = (
    syn_meta
    .iloc[
        keep_original_indices
    ]
    .copy()
    .reset_index(
        drop=True
    )
)

syn_pool_meta[
    "pool_index"
] = np.arange(
    len(
        syn_pool_meta
    ),
    dtype=np.int64,
)

if not (
    syn_pool_meta[
        "label"
    ].astype(int)
    == 1
).all():
    raise RuntimeError(
        "Synthetic pool contains "
        "non-Falling windows."
    )

np.save(
    OUT
    / "synthetic_falling_preimpact150_X.npy",
    syn_pool_x,
)

np.save(
    OUT
    / "synthetic_falling_preimpact150_y.npy",
    np.ones(
        len(syn_pool_x),
        dtype=np.int64,
    ),
)

np.save(
    OUT
    / "synthetic_falling_original_indices.npy",
    keep_original_indices,
)

syn_pool_meta.to_csv(
    OUT
    / "synthetic_falling_preimpact150_metadata.csv",
    index=False,
)

source_summary = pd.DataFrame(
    source_rows
)

source_summary.to_csv(
    OUT
    / "synthetic_source_preimpact150_summary.csv",
    index=False,
)

print(
    "Original synthetic Falling:",
    len(fall_indices),
)

print(
    "Synthetic source trials:",
    len(source_summary),
)

print(
    "Aligned synthetic Falling:",
    len(syn_pool_x),
)

print(
    "Sources retaining >=1 window:",
    int(
        (
            source_summary[
                "preimpact150_falling"
            ] > 0
        ).sum()
    ),
)


print()
print("=" * 100)
print("6. BUILD FOLD-SPECIFIC SYNTHETIC DRAW ORDERS")
print("=" * 100)

if len(
    syn_pool_meta
) == 0:
    raise RuntimeError(
        "Synthetic pool is empty."
    )

groups = {
    source: (
        g[
            "pool_index"
        ]
        .astype(int)
        .to_numpy()
    )
    for source, g
    in syn_pool_meta.groupby(
        "source_path",
        sort=True,
    )
}

mix_ratios = {
    "MIX20": 0.20,
    "MIX50": 0.50,
    "MIX70": 0.70,
    "MIX100": 1.00,
}

draw_summary = []

for fold in range(5):

    q = fold_summary[
        (
            fold_summary[
                "fold"
            ] == fold
        )
        & (
            fold_summary[
                "role"
            ] == "train"
        )
    ]

    if len(q) != 1:
        raise RuntimeError(
            f"Bad fold summary {fold}"
        )

    real_train_falls = int(
        q.iloc[0][
            "falling_windows"
        ]
    )

    rng = np.random.default_rng(
        42
        + fold * 1000
    )

    source_names = np.array(
        sorted(
            groups
        ),
        dtype=object,
    )

    working = {}

    for source in source_names:

        arr = groups[
            source
        ].copy()

        rng.shuffle(
            arr
        )

        working[source] = {
            "arr": arr,
            "pos": 0,
        }

    draw_order = []

    while len(
        draw_order
    ) < real_train_falls:

        cycle = (
            source_names
            .copy()
        )

        rng.shuffle(
            cycle
        )

        for source in cycle:

            state = working[
                source
            ]

            if (
                state[
                    "pos"
                ]
                >= len(
                    state[
                        "arr"
                    ]
                )
            ):

                arr = groups[
                    source
                ].copy()

                rng.shuffle(
                    arr
                )

                state[
                    "arr"
                ] = arr

                state[
                    "pos"
                ] = 0

            idx = int(
                state[
                    "arr"
                ][
                    state[
                        "pos"
                    ]
                ]
            )

            state[
                "pos"
            ] += 1

            draw_order.append(
                idx
            )

            if len(
                draw_order
            ) >= real_train_falls:
                break

    draw_order = np.array(
        draw_order,
        dtype=np.int64,
    )

    conditions = {
        "SIM_FALL_SUBSTITUTION":
            1.00,
        **mix_ratios,
    }

    for name, ratio in (
        conditions.items()
    ):

        target = int(
            np.floor(
                real_train_falls
                * ratio
                + 0.5
            )
        )

        picks = (
            draw_order[
                :target
            ]
        )

        d = (
            syn_pool_meta
            .iloc[
                picks
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        d.insert(
            0,
            "draw_position",
            np.arange(
                len(d),
                dtype=np.int64,
            ),
        )

        d[
            "fold"
        ] = fold

        d[
            "condition"
        ] = name

        d.to_csv(
            DRAW_OUT
            / (
                f"fold_{fold}_"
                f"{name}_"
                "synthetic_draws.csv"
            ),
            index=False,
        )

        draw_summary.append(
            {
                "fold":
                    fold,
                "condition":
                    name,
                "real_train_falling":
                    real_train_falls,
                "synthetic_draws":
                    target,
                "unique_pool_windows":
                    len(
                        np.unique(
                            picks
                        )
                    ),
                "unique_source_trials":
                    d[
                        "source_path"
                    ].nunique(),
                "reuse_factor":
                    (
                        target
                        / len(
                            syn_pool_meta
                        )
                    ),
            }
        )


draw_summary = pd.DataFrame(
    draw_summary
)

draw_summary.to_csv(
    OUT
    / "synthetic_draw_summary.csv",
    index=False,
)

print(
    draw_summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.3f}",
    )
)


print()
print("=" * 100)
print("7. RAW-UNIT SCALE SANITY CHECK")
print("=" * 100)

physical_falls = []

for row in trials.itertuples():

    if (
        row.kept_falling <= 0
        or not row.output_dir
    ):
        continue

    d = Path(
        row.output_dir
    )

    y = np.load(
        d
        / "labels.npy",
        allow_pickle=True,
    ).astype(str)

    x = np.load(
        d
        / "segments.npy",
        allow_pickle=True,
    )

    q = x[
        y == "Falling"
    ]

    if len(q):
        physical_falls.append(
            q
        )

    if sum(
        len(a)
        for a in physical_falls
    ) >= 1000:
        break

physical_sample = (
    np.concatenate(
        physical_falls,
        axis=0,
    )[:1000]
)

synthetic_sample = (
    syn_pool_x[
        :min(
            1000,
            len(
                syn_pool_x
            ),
        )
    ]
    .copy()
)

synthetic_raw = (
    synthetic_sample
    .copy()
)

synthetic_raw[
    :,
    :,
    0:3
] /= np.float32(
    0.00980665
)

synthetic_raw[
    :,
    :,
    3:6
] *= np.float32(
    1000.0
)


def report_scale(name, x):

    acc = np.abs(
        x[
            :,
            :,
            0:3
        ]
    ).ravel()

    gyro = np.abs(
        x[
            :,
            :,
            3:6
        ]
    ).ravel()

    print()
    print(name)

    print(
        "  |ACC| p50/p95/p99:",
        np.percentile(
            acc,
            [50, 95, 99],
        ),
    )

    print(
        "  |GYRO| p50/p95/p99:",
        np.percentile(
            gyro,
            [50, 95, 99],
        ),
    )


report_scale(
    "PHYSICAL STORED RAW",
    physical_sample,
)

report_scale(
    "SYNTHETIC AFTER SI -> mg/mdps",
    synthetic_raw,
)


print()
print("=" * 100)
print("8. FINAL INPUT GATES")
print("=" * 100)

draw_files = list(
    DRAW_OUT.glob(
        "fold_*_synthetic_draws.csv"
    )
)

if len(
    draw_files
) != 25:
    raise RuntimeError(
        f"Expected 25 draw manifests, "
        f"got {len(draw_files)}"
    )

if (
    trials[
        "kept_falling"
    ].sum()
    <= 0
):
    raise RuntimeError(
        "No physical Falling remains."
    )

if (
    trials[
        "kept_activity"
    ].sum()
    <= 0
):
    raise RuntimeError(
        "No physical Activity remains."
    )

if len(
    syn_pool_x
) <= 0:
    raise RuntimeError(
        "No synthetic Falling remains."
    )

facts = {
    "physical_source_root":
        str(
            PHYSICAL_SOURCE
        ),
    "physical_aligned_root":
        str(
            PHYSICAL_OUT
        ),
    "physical_total_subject_dirs":
        len(
            all_subjects
        ),
    "physical_kfold_subjects":
        len(
            subjects
        ),
    "excluded_subjects":
        sorted(
            excluded
        ),
    "physical_aligned_activity":
        int(
            trials[
                "kept_activity"
            ].sum()
        ),
    "physical_aligned_falling":
        int(
            trials[
                "kept_falling"
            ].sum()
        ),
    "synthetic_original_falling":
        int(
            len(
                fall_indices
            )
        ),
    "synthetic_aligned_falling":
        int(
            len(
                syn_pool_x
            )
        ),
    "synthetic_acc_source_unit":
        sorted(
            syn_pool_meta[
                "acc_unit"
            ].dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
    "synthetic_gyro_source_unit":
        sorted(
            syn_pool_meta[
                "gyro_unit"
            ].dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
    "window_samples": 30,
    "stride_samples": 15,
    "sampling_hz": 100,
    "preimpact_exclusion_samples": 15,
    "preimpact_exclusion_ms": 150,
}

with open(
    OUT
    / "aligned_input_facts.json",
    "w",
) as f:
    json.dump(
        facts,
        f,
        indent=2,
    )

print(
    "Physical subject split gate : PASS"
)

print(
    "Physical pre-impact gate    : PASS"
)

print(
    "Synthetic pre-impact gate   : PASS"
)

print(
    "Synthetic draw gate         : PASS"
)

print()
print(
    "FINAL PAPER-ALIGNED INPUT GATE: PASS"
)

print()
print(
    "Output:",
    OUT,
)
