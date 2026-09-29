from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/"
    "outputs/phase2_corrected_event_dataset_v1"
)

OUT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/"
    "outputs/phase2_corrected_split_manifests_v1"
)
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260827

physical = pd.read_csv(
    ROOT / "physical_metadata.csv"
)

synthetic = pd.read_csv(
    ROOT / "synthetic_metadata.csv"
)

physical["label"] = pd.to_numeric(
    physical["label"],
    errors="raise",
).astype(int)

synthetic["label"] = pd.to_numeric(
    synthetic["label"],
    errors="raise",
).astype(int)


print("=" * 88)
print("SOURCE DATA")
print("=" * 88)

print(
    "Physical windows:",
    len(physical)
)

print(
    "Physical groups :",
    physical["group_id"].nunique()
)

print(
    "Synthetic windows:",
    len(synthetic)
)

print(
    "Synthetic profiles:",
    synthetic["profile_id"].nunique()
)


def make_group_folds(
    df,
    group_col,
    *,
    n_splits,
    seed,
):
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    fold_for_group = {}

    X_dummy = np.zeros(
        (len(df), 1),
        dtype=np.float32,
    )

    y = df["label"].to_numpy()
    groups = df[group_col].astype(str).to_numpy()

    for fold, (_, test_idx) in enumerate(
        splitter.split(
            X_dummy,
            y,
            groups,
        )
    ):
        test_groups = sorted(
            set(groups[test_idx])
        )

        for g in test_groups:
            if g in fold_for_group:
                raise RuntimeError(
                    f"Group assigned twice: {g}"
                )

            fold_for_group[g] = fold

    expected = set(
        df[group_col].astype(str).unique()
    )

    if set(fold_for_group) != expected:
        raise RuntimeError(
            "Not every group received a fold."
        )

    return fold_for_group


physical_assignments = []

for dataset in [
    "KFALL",
    "UNIVR",
]:
    sub = physical[
        physical["dataset"] == dataset
    ].copy()

    outer_map = make_group_folds(
        sub,
        "group_id",
        n_splits=5,
        seed=SEED,
    )

    subject_summary = (
        sub.groupby(
            "group_id",
            as_index=False,
        )
        .agg(
            total_windows=(
                "label",
                "size",
            ),
            activity_windows=(
                "label",
                lambda x:
                    int((x == 0).sum()),
            ),
            falling_windows=(
                "label",
                lambda x:
                    int((x == 1).sum()),
            ),
        )
    )

    for _, r in subject_summary.iterrows():
        physical_assignments.append({
            "dataset":
                dataset,

            "group_id":
                r["group_id"],

            "outer_test_fold":
                int(
                    outer_map[
                        r["group_id"]
                    ]
                ),

            "total_windows":
                int(
                    r["total_windows"]
                ),

            "activity_windows":
                int(
                    r["activity_windows"]
                ),

            "falling_windows":
                int(
                    r["falling_windows"]
                ),
        })


physical_assignments = pd.DataFrame(
    physical_assignments
)

physical_assignments.to_csv(
    OUT /
    "physical_outer_fold_assignments.csv",
    index=False,
)


nested_rows = []

for outer_fold in range(5):

    for dataset in [
        "KFALL",
        "UNIVR",
    ]:
        dataset_groups = physical_assignments[
            physical_assignments[
                "dataset"
            ] == dataset
        ]

        outer_test_groups = set(
            dataset_groups.loc[
                dataset_groups[
                    "outer_test_fold"
                ] == outer_fold,
                "group_id",
            ].astype(str)
        )

        sub = physical[
            (physical["dataset"] == dataset)
            &
            (~physical["group_id"].isin(
                outer_test_groups
            ))
        ].copy()

        inner_map = make_group_folds(
            sub,
            "group_id",
            n_splits=4,
            seed=SEED + 100 + outer_fold,
        )

        val_inner_fold = 0

        for g in sorted(
            sub["group_id"]
            .astype(str)
            .unique()
        ):
            role = (
                "val"
                if inner_map[g]
                == val_inner_fold
                else "train"
            )

            nested_rows.append({
                "outer_fold":
                    outer_fold,

                "dataset":
                    dataset,

                "group_id":
                    g,

                "role":
                    role,

                "inner_fold":
                    int(inner_map[g]),
            })

        for g in sorted(
            outer_test_groups
        ):
            nested_rows.append({
                "outer_fold":
                    outer_fold,

                "dataset":
                    dataset,

                "group_id":
                    g,

                "role":
                    "test",

                "inner_fold":
                    -1,
            })


physical_nested = pd.DataFrame(
    nested_rows
)

physical_nested.to_csv(
    OUT /
    "physical_nested_split_assignments.csv",
    index=False,
)


syn_outer_map = make_group_folds(
    synthetic,
    "profile_id",
    n_splits=5,
    seed=SEED + 1000,
)

syn_profile_summary = (
    synthetic.groupby(
        "profile_id",
        as_index=False,
    )
    .agg(
        total_windows=(
            "label",
            "size",
        ),
        activity_windows=(
            "label",
            lambda x:
                int((x == 0).sum()),
        ),
        falling_windows=(
            "label",
            lambda x:
                int((x == 1).sum()),
        ),
    )
)

syn_profile_summary[
    "outer_test_fold"
] = (
    syn_profile_summary[
        "profile_id"
    ]
    .map(syn_outer_map)
    .astype(int)
)

syn_profile_summary.to_csv(
    OUT /
    "synthetic_outer_fold_assignments.csv",
    index=False,
)


syn_nested_rows = []

for outer_fold in range(5):

    outer_test_profiles = set(
        syn_profile_summary.loc[
            syn_profile_summary[
                "outer_test_fold"
            ] == outer_fold,
            "profile_id",
        ].astype(str)
    )

    sub = synthetic[
        ~synthetic[
            "profile_id"
        ].isin(
            outer_test_profiles
        )
    ].copy()

    inner_map = make_group_folds(
        sub,
        "profile_id",
        n_splits=4,
        seed=SEED + 2000 + outer_fold,
    )

    for p in sorted(
        sub["profile_id"]
        .astype(str)
        .unique()
    ):
        syn_nested_rows.append({
            "outer_fold":
                outer_fold,

            "profile_id":
                p,

            "role":
                (
                    "val"
                    if inner_map[p] == 0
                    else "train"
                ),

            "inner_fold":
                int(inner_map[p]),
        })

    for p in sorted(
        outer_test_profiles
    ):
        syn_nested_rows.append({
            "outer_fold":
                outer_fold,

            "profile_id":
                p,

            "role":
                "test",

            "inner_fold":
                -1,
        })


synthetic_nested = pd.DataFrame(
    syn_nested_rows
)

synthetic_nested.to_csv(
    OUT /
    "synthetic_nested_split_assignments.csv",
    index=False,
)


physical_fold_rows = []
mix_rows = []

unique_syn_falling = int(
    (
        synthetic["label"] == 1
    ).sum()
)

for fold in range(5):

    a = physical_nested[
        physical_nested[
            "outer_fold"
        ] == fold
    ]

    role_map = dict(
        zip(
            a["group_id"].astype(str),
            a["role"].astype(str),
        )
    )

    tmp = physical.copy()

    tmp["role"] = (
        tmp["group_id"]
        .astype(str)
        .map(role_map)
    )

    if tmp["role"].isna().any():
        raise RuntimeError(
            f"Fold {fold}: missing role."
        )

    for role in [
        "train",
        "val",
        "test",
    ]:
        r = tmp[
            tmp["role"] == role
        ]

        for dataset in [
            "ALL",
            "KFALL",
            "UNIVR",
        ]:
            q = (
                r
                if dataset == "ALL"
                else r[
                    r["dataset"]
                    == dataset
                ]
            )

            physical_fold_rows.append({
                "fold":
                    fold,

                "role":
                    role,

                "dataset":
                    dataset,

                "groups":
                    int(
                        q[
                            "group_id"
                        ].nunique()
                    ),

                "windows":
                    int(len(q)),

                "activity_windows":
                    int(
                        (
                            q["label"] == 0
                        ).sum()
                    ),

                "falling_windows":
                    int(
                        (
                            q["label"] == 1
                        ).sum()
                    ),
            })

    train = tmp[
        tmp["role"] == "train"
    ]

    physical_train_falls = int(
        (
            train["label"] == 1
        ).sum()
    )

    for ratio in [
        0.20,
        0.50,
        0.70,
        1.00,
    ]:
        target = int(
            round(
                physical_train_falls
                * ratio
            )
        )

        mix_rows.append({
            "fold":
                fold,

            "mix_ratio":
                ratio,

            "physical_train_falling_windows":
                physical_train_falls,

            "target_synthetic_falling_windows":
                target,

            "unique_synthetic_falling_available":
                unique_syn_falling,

            "unique_sampling_feasible":
                bool(
                    target
                    <= unique_syn_falling
                ),

            "minimum_reuse_factor":
                float(
                    target
                    / unique_syn_falling
                ),
        })


physical_fold_summary = pd.DataFrame(
    physical_fold_rows
)

mix_feasibility = pd.DataFrame(
    mix_rows
)

physical_fold_summary.to_csv(
    OUT /
    "physical_fold_window_summary.csv",
    index=False,
)

mix_feasibility.to_csv(
    OUT /
    "mix_ratio_feasibility.csv",
    index=False,
)


synthetic_fold_rows = []

for fold in range(5):

    a = synthetic_nested[
        synthetic_nested[
            "outer_fold"
        ] == fold
    ]

    role_map = dict(
        zip(
            a["profile_id"].astype(str),
            a["role"].astype(str),
        )
    )

    tmp = synthetic.copy()

    tmp["role"] = (
        tmp["profile_id"]
        .astype(str)
        .map(role_map)
    )

    for role in [
        "train",
        "val",
        "test",
    ]:
        r = tmp[
            tmp["role"] == role
        ]

        synthetic_fold_rows.append({
            "fold":
                fold,

            "role":
                role,

            "profiles":
                int(
                    r[
                        "profile_id"
                    ].nunique()
                ),

            "windows":
                int(len(r)),

            "activity_windows":
                int(
                    (
                        r["label"] == 0
                    ).sum()
                ),

            "falling_windows":
                int(
                    (
                        r["label"] == 1
                    ).sum()
                ),
        })


synthetic_fold_summary = pd.DataFrame(
    synthetic_fold_rows
)

synthetic_fold_summary.to_csv(
    OUT /
    "synthetic_fold_window_summary.csv",
    index=False,
)


checks = {}


for fold in range(5):

    a = physical_nested[
        physical_nested[
            "outer_fold"
        ] == fold
    ]

    train = set(
        a.loc[
            a["role"] == "train",
            "group_id",
        ].astype(str)
    )

    val = set(
        a.loc[
            a["role"] == "val",
            "group_id",
        ].astype(str)
    )

    test = set(
        a.loc[
            a["role"] == "test",
            "group_id",
        ].astype(str)
    )

    checks[
        f"physical_fold{fold}_train_val_disjoint"
    ] = train.isdisjoint(val)

    checks[
        f"physical_fold{fold}_train_test_disjoint"
    ] = train.isdisjoint(test)

    checks[
        f"physical_fold{fold}_val_test_disjoint"
    ] = val.isdisjoint(test)

    for role in [
        "train",
        "val",
        "test",
    ]:
        rr = a[
            a["role"] == role
        ]

        datasets = set(
            rr["dataset"].astype(str)
        )

        checks[
            f"physical_fold{fold}_{role}_both_datasets"
        ] = (
            datasets
            ==
            {"KFALL", "UNIVR"}
        )

        groups = set(
            rr["group_id"].astype(str)
        )

        q = physical[
            physical[
                "group_id"
            ].isin(groups)
        ]

        checks[
            f"physical_fold{fold}_{role}_both_classes"
        ] = (
            set(
                q["label"]
                .astype(int)
                .unique()
            )
            ==
            {0, 1}
        )


for fold in range(5):

    a = synthetic_nested[
        synthetic_nested[
            "outer_fold"
        ] == fold
    ]

    train = set(
        a.loc[
            a["role"] == "train",
            "profile_id",
        ].astype(str)
    )

    val = set(
        a.loc[
            a["role"] == "val",
            "profile_id",
        ].astype(str)
    )

    test = set(
        a.loc[
            a["role"] == "test",
            "profile_id",
        ].astype(str)
    )

    checks[
        f"synthetic_fold{fold}_train_val_disjoint"
    ] = train.isdisjoint(val)

    checks[
        f"synthetic_fold{fold}_train_test_disjoint"
    ] = train.isdisjoint(test)

    checks[
        f"synthetic_fold{fold}_val_test_disjoint"
    ] = val.isdisjoint(test)

    for role in [
        "train",
        "val",
        "test",
    ]:
        profiles = set(
            a.loc[
                a["role"] == role,
                "profile_id",
            ].astype(str)
        )

        q = synthetic[
            synthetic[
                "profile_id"
            ].isin(profiles)
        ]

        checks[
            f"synthetic_fold{fold}_{role}_both_classes"
        ] = (
            set(
                q["label"]
                .astype(int)
                .unique()
            )
            ==
            {0, 1}
        )


checks[
    "all_61_physical_groups_assigned"
] = (
    physical_assignments[
        "group_id"
    ].nunique()
    == 61
)

checks[
    "all_22_synthetic_profiles_assigned"
] = (
    syn_profile_summary[
        "profile_id"
    ].nunique()
    == 22
)


facts = {
    "seed":
        SEED,

    "physical_outer_split":
        (
            "5-fold StratifiedGroupKFold "
            "performed separately within "
            "KFall and UniVR, then fold IDs merged"
        ),

    "physical_inner_validation":
        (
            "4-fold StratifiedGroupKFold on "
            "outer-training subjects separately "
            "within each physical dataset; "
            "inner fold 0 used as validation"
        ),

    "synthetic_sim_only_split":
        (
            "5-fold profile-grouped outer split; "
            "4-fold profile-grouped inner validation"
        ),

    "unique_synthetic_falling_windows":
        unique_syn_falling,

    "checks": {
        k: bool(v)
        for k, v in checks.items()
    },

    "gate_pass":
        bool(
            all(
                checks.values()
            )
        ),
}

with open(
    OUT / "split_facts.json",
    "w",
) as f:
    json.dump(
        facts,
        f,
        indent=2,
    )


print()
print("=" * 88)
print("PHYSICAL 5-FOLD SUMMARY")
print("=" * 88)

print(
    physical_fold_summary.to_string(
        index=False
    )
)

print()
print("=" * 88)
print("SIM-ONLY 5-FOLD SUMMARY")
print("=" * 88)

print(
    synthetic_fold_summary.to_string(
        index=False
    )
)

print()
print("=" * 88)
print("MIX RATIO FEASIBILITY")
print("=" * 88)

print(
    mix_feasibility.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.3f}",
    )
)

print()
print("=" * 88)
print("LEAKAGE / COVERAGE CHECKS")
print("=" * 88)

failed = []

for k, v in checks.items():
    print(
        f"{k:58s}",
        "PASS" if v else "FAIL"
    )

    if not v:
        failed.append(k)

print()
print("=" * 88)

if not failed:
    print(
        "FINAL SPLIT MANIFEST GATE: PASS"
    )
else:
    print(
        "FINAL SPLIT MANIFEST GATE: INVESTIGATE"
    )

    print(
        "Failed checks:"
    )

    for x in failed:
        print(
            " -",
            x,
        )

print("=" * 88)

print()
print(
    "Split directory:"
)
print(OUT)
