from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


DATA = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/"
    "outputs/phase2_corrected_event_dataset_v1"
)

SPLITS = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/"
    "outputs/phase2_protechto_original_kfold_manifests_v1"
)

OUT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/"
    "outputs/phase2_protechto_original_kfold_augmentation_manifests_v1"
)
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260827

RATIOS = [
    ("MIX20", 0.20),
    ("MIX50", 0.50),
    ("MIX70", 0.70),
    ("MIX100", 1.00),
]

meta = pd.read_csv(
    DATA / "synthetic_metadata.csv"
)

fold_summary = pd.read_csv(
    SPLITS / "physical_fold_window_summary.csv"
)

meta["label"] = pd.to_numeric(
    meta["label"],
    errors="raise",
).astype(int)

meta["task_id"] = pd.to_numeric(
    meta["task_id"],
    errors="raise",
).astype(int)

meta["synthetic_window_index"] = np.arange(
    len(meta),
    dtype=np.int64,
)

fall = meta[
    meta["label"] == 1
].copy()

fall["source_trial_id"] = (
    fall["profile_id"].astype(str)
    + "_T"
    + fall["task_id"].astype(str)
)

trial_groups = {
    trial:
        q[
            "synthetic_window_index"
        ].to_numpy(
            dtype=np.int64
        )
    for trial, q
    in fall.groupby(
        "source_trial_id",
        sort=True,
    )
}

trial_ids = sorted(
    trial_groups
)

print("=" * 92)
print("SYNTHETIC FALLING SOURCE POOL")
print("=" * 92)

print(
    "Falling windows:",
    len(fall)
)

print(
    "Source trials  :",
    len(trial_ids)
)

print(
    "Profiles       :",
    fall["profile_id"].nunique()
)

print(
    "Tasks          :",
    fall["task_id"].nunique()
)

if len(fall) != 1256:
    raise RuntimeError(
        f"Expected 1256 Falling windows, "
        f"found {len(fall)}"
    )

if len(trial_ids) != 310:
    raise RuntimeError(
        f"Expected 310 source trials, "
        f"found {len(trial_ids)}"
    )


def make_balanced_stream(
    target,
    seed,
):
    rng = np.random.default_rng(
        seed
    )

    queues = {}
    positions = {}
    queue_cycles = {}

    for trial in trial_ids:

        arr = trial_groups[
            trial
        ].copy()

        queues[trial] = rng.permutation(
            arr
        )

        positions[trial] = 0
        queue_cycles[trial] = 0

    window_use_count = {
        int(i): 0
        for i in fall[
            "synthetic_window_index"
        ].to_numpy()
    }

    trial_draw_count = {
        trial: 0
        for trial in trial_ids
    }

    records = []

    draw_position = 0
    round_id = 0

    while draw_position < target:

        order = rng.permutation(
            trial_ids
        )

        for trial in order:

            if draw_position >= target:
                break

            pos = positions[
                trial
            ]

            queue = queues[
                trial
            ]

            if pos >= len(queue):

                queue_cycles[
                    trial
                ] += 1

                queue = rng.permutation(
                    trial_groups[
                        trial
                    ]
                )

                queues[
                    trial
                ] = queue

                positions[
                    trial
                ] = 0

                pos = 0

            syn_idx = int(
                queue[pos]
            )

            positions[
                trial
            ] += 1

            window_use_count[
                syn_idx
            ] += 1

            trial_draw_count[
                trial
            ] += 1

            source_row = meta.iloc[
                syn_idx
            ]

            records.append({
                "draw_position":
                    int(draw_position),

                "round_id":
                    int(round_id),

                "synthetic_window_index":
                    syn_idx,

                "window_use_number":
                    int(
                        window_use_count[
                            syn_idx
                        ]
                    ),

                "is_repeated_window_draw":
                    bool(
                        window_use_count[
                            syn_idx
                        ] > 1
                    ),

                "source_trial_id":
                    trial,

                "source_trial_draw_number":
                    int(
                        trial_draw_count[
                            trial
                        ]
                    ),

                "source_trial_queue_cycle":
                    int(
                        queue_cycles[
                            trial
                        ]
                    ),

                "profile_id":
                    str(
                        source_row[
                            "profile_id"
                        ]
                    ),

                "task_id":
                    int(
                        source_row[
                            "task_id"
                        ]
                    ),

                "source_path":
                    str(
                        source_row[
                            "source_path"
                        ]
                    ),

                "source_row_start":
                    int(
                        source_row[
                            "source_row_start"
                        ]
                    ),

                "source_row_end":
                    int(
                        source_row[
                            "source_row_end"
                        ]
                    ),

                "window_start_time_s":
                    float(
                        source_row[
                            "window_start_time_s"
                        ]
                    ),

                "window_end_time_s":
                    float(
                        source_row[
                            "window_end_time_s"
                        ]
                    ),
            })

            draw_position += 1

        round_id += 1

    return pd.DataFrame(
        records
    )


all_checks = {}
summary_rows = []

for fold in range(5):

    row = fold_summary[
        (fold_summary["fold"] == fold)
        &
        (fold_summary["role"] == "train")
        &
        (fold_summary["dataset"] == "ALL")
    ]

    if len(row) != 1:
        raise RuntimeError(
            f"Could not uniquely resolve "
            f"physical train summary for fold {fold}"
        )

    physical_falls = int(
        row.iloc[0][
            "falling_windows"
        ]
    )

    targets = {
        name:
            int(
                round(
                    physical_falls
                    * ratio
                )
            )
        for name, ratio
        in RATIOS
    }

    max_target = targets[
        "MIX100"
    ]

    stream = make_balanced_stream(
        max_target,
        SEED + 10000 + fold,
    )

    stream["fold"] = fold

    master_path = (
        OUT /
        f"fold_{fold}_master_"
        f"synthetic_draw_stream.csv"
    )

    stream.to_csv(
        master_path,
        index=False,
    )

    print()
    print("=" * 92)
    print(
        f"FOLD {fold}"
    )
    print("=" * 92)

    print(
        "Physical train Falling:",
        physical_falls
    )

    previous_positions = None

    for name, ratio in RATIOS:

        target = targets[
            name
        ]

        q = stream.iloc[
            :target
        ].copy()

        q[
            "experiment"
        ] = name

        q[
            "mix_ratio"
        ] = ratio

        ratio_path = (
            OUT /
            f"fold_{fold}_{name}_"
            f"synthetic_draws.csv"
        )

        q.to_csv(
            ratio_path,
            index=False,
        )

        unique_windows = int(
            q[
                "synthetic_window_index"
            ].nunique()
        )

        repeated_draws = int(
            target
            - unique_windows
        )

        unique_trials = int(
            q[
                "source_trial_id"
            ].nunique()
        )

        profiles = int(
            q[
                "profile_id"
            ].nunique()
        )

        tasks = int(
            q[
                "task_id"
            ].nunique()
        )

        window_use = (
            q.groupby(
                "synthetic_window_index"
            )
            .size()
        )

        trial_use = (
            q.groupby(
                "source_trial_id"
            )
            .size()
        )

        task_use = (
            q.groupby(
                "task_id"
            )
            .size()
            .sort_values(
                ascending=False
            )
        )

        max_task_pct = float(
            100.0
            * task_use.iloc[0]
            / target
        )

        current_positions = (
            q[
                "synthetic_window_index"
            ]
            .to_numpy()
        )

        exact_target = (
            len(q) == target
        )

        all_falling = bool(
            meta.iloc[
                current_positions
            ]["label"]
            .eq(1)
            .all()
        )

        all_trials = (
            unique_trials == 310
        )

        all_profiles = (
            profiles == 22
        )

        all_tasks = (
            tasks == 18
        )

        trial_balance = (
            int(
                trial_use.max()
            )
            -
            int(
                trial_use.min()
            )
            <= 1
        )

        prefix_ok = True

        if previous_positions is not None:
            prefix_ok = bool(
                np.array_equal(
                    previous_positions,
                    current_positions[
                        :len(
                            previous_positions
                        )
                    ],
                )
            )

        checks = {
            "exact_target":
                exact_target,

            "all_draws_are_falling":
                all_falling,

            "all_310_trials_represented":
                all_trials,

            "all_22_profiles_represented":
                all_profiles,

            "all_18_tasks_represented":
                all_tasks,

            "source_trial_draw_balance_le1":
                trial_balance,

            "nested_prefix":
                prefix_ok,
        }

        for k, v in checks.items():

            all_checks[
                f"fold{fold}_{name}_{k}"
            ] = bool(v)

        summary_rows.append({
            "fold":
                fold,

            "experiment":
                name,

            "ratio":
                ratio,

            "physical_train_falling":
                physical_falls,

            "target_draws":
                target,

            "unique_windows":
                unique_windows,

            "repeated_draws":
                repeated_draws,

            "unique_window_fraction":
                unique_windows
                / target,

            "source_trials":
                unique_trials,

            "profiles":
                profiles,

            "tasks":
                tasks,

            "min_draws_per_trial":
                int(
                    trial_use.min()
                ),

            "max_draws_per_trial":
                int(
                    trial_use.max()
                ),

            "max_single_window_uses":
                int(
                    window_use.max()
                ),

            "max_task_draw_percent":
                max_task_pct,
        })

        print()
        print(name)

        print(
            "  target draws            :",
            target
        )

        print(
            "  unique windows          :",
            unique_windows
        )

        print(
            "  repeated draws          :",
            repeated_draws
        )

        print(
            "  unique fraction         :",
            f"{unique_windows / target:.3f}"
        )

        print(
            "  source trials           :",
            unique_trials
        )

        print(
            "  draws/trial min-max     :",
            int(
                trial_use.min()
            ),
            "-",
            int(
                trial_use.max()
            ),
        )

        print(
            "  max uses of one window  :",
            int(
                window_use.max()
            )
        )

        print(
            "  largest task share      :",
            f"{max_task_pct:.2f}%"
        )

        print(
            "  checks                  :",
            (
                "PASS"
                if all(
                    checks.values()
                )
                else "FAIL"
            )
        )

        previous_positions = (
            current_positions.copy()
        )


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUT /
    "augmentation_sampling_summary.csv",
    index=False,
)

failed = [
    k
    for k, v
    in all_checks.items()
    if not v
]

facts = {
    "version":
        "phase2_protechto_original_kfold_augmentation_manifests_v1",

    "seed":
        SEED,

    "synthetic_falling_windows":
        int(len(fall)),

    "synthetic_source_trials":
        int(len(trial_ids)),

    "synthetic_profiles":
        int(
            fall[
                "profile_id"
            ].nunique()
        ),

    "synthetic_tasks":
        int(
            fall[
                "task_id"
            ].nunique()
        ),

    "sampling_policy":
        (
            "hierarchical source-trial-balanced "
            "deterministic draw stream; each round "
            "visits every eligible simulated trial "
            "once in shuffled order; windows within "
            "each trial are shuffled and cycled; "
            "MIX20/MIX50/MIX70/MIX100 use nested "
            "prefixes of the same fold-specific stream"
        ),

    "interpretation":
        (
            "augmentation ratios denote synthetic "
            "Falling training draws relative to "
            "physical Falling training windows, "
            "not numbers of unique simulations"
        ),

    "checks": {
        k: bool(v)
        for k, v
        in all_checks.items()
    },

    "gate_pass":
        len(failed) == 0,
}

facts_path = (
    OUT /
    "augmentation_sampling_facts.json"
)

with open(
    facts_path,
    "w",
) as f:

    json.dump(
        facts,
        f,
        indent=2,
    )


hash_rows = []

for path in sorted(
    OUT.glob("*.csv")
):

    h = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    hash_rows.append({
        "file":
            path.name,

        "sha256":
            h,
    })

pd.DataFrame(
    hash_rows
).to_csv(
    OUT /
    "augmentation_manifest_sha256.csv",
    index=False,
)


print()
print("=" * 92)
print("AUGMENTATION SAMPLING SUMMARY")
print("=" * 92)

print(
    summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.3f}",
    )
)

print()
print("=" * 92)
print("FINAL SAMPLING GATE")
print("=" * 92)

if failed:

    print(
        "FINAL AUGMENTATION SAMPLING GATE: "
        "INVESTIGATE"
    )

    for x in failed:
        print(
            "FAIL:",
            x
        )

else:

    print(
        "FINAL AUGMENTATION SAMPLING GATE: PASS"
    )

print()
print(
    "No physical or synthetic source "
    "file was modified."
)

print()
print(
    "Artifacts:"
)

print(OUT)
