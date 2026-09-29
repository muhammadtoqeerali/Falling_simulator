#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split


PROJECT = Path("/mnt/hdd16T/ToqeerHomeBackup/mujoco_project")
PROTECHTO = Path("/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master")

CANON = (
    PROJECT
    / "outputs"
    / "phase2_publication_gate_expanded42_v2"
    / "final_synthetic_event_policy_v2"
    / "canonical_synthetic_event_manifest_expanded42_v2.csv"
)

CANON_FACTS = (
    PROJECT
    / "outputs"
    / "phase2_publication_gate_expanded42_v2"
    / "final_synthetic_event_policy_v2"
    / "canonical_synthetic_event_facts_expanded42_v2.json"
)

CACHE = (
    PROJECT
    / "outputs"
    / "protechto_exact_synthetic_cache_expanded42_v2"
)

CAMPAIGN = (
    PROJECT
    / "outputs"
    / "protechto_exact_full_campaign_expanded42_v2"
)

OUT = (
    PROJECT
    / "outputs"
    / "final_dataset_split_audit_expanded42_v2"
)

OUT.mkdir(parents=True, exist_ok=True)


PAPER_CONDITIONS = [
    "EXP01_REAL_ONLY",
    "EXP03_SIM_ONLY_FULL",
    "EXP04_MIX20",
    "EXP05_MIX50",
    "EXP06_MIX70",
    "EXP07_MIX100",
]

PHYSICAL_TEST_CONDITIONS = [
    "EXP01_REAL_ONLY",
    "EXP04_MIX20",
    "EXP05_MIX50",
    "EXP06_MIX70",
    "EXP07_MIX100",
]

RATIOS = {
    "EXP04_MIX20": 0.20,
    "EXP05_MIX50": 0.50,
    "EXP06_MIX70": 0.70,
    "EXP07_MIX100": 1.00,
}

# Frozen physical cohort/split evidence.
# These quantities are unchanged because the physical dataset, subject split,
# KFold seed, validation split and physical evaluation pipeline are unchanged.
PHYSICAL_FOLDS = {
    1: {
        "train_subjects": 44,
        "val_subjects": 12,
        "test_subjects": 15,
        "train_total": 595308,
        "train_activity": 588646,
        "train_falling": 6662,
        "val_total": 250912,
        "val_activity": 249332,
        "val_falling": 1580,
        "test_total": 450947,
        "test_activity": 449383,
        "test_falling": 1564,
    },
    2: {
        "train_subjects": 45,
        "val_subjects": 12,
        "test_subjects": 14,
        "train_total": 666800,
        "train_activity": 660284,
        "train_falling": 6516,
        "val_total": 334287,
        "val_activity": 332855,
        "val_falling": 1432,
        "test_total": 296080,
        "test_activity": 294222,
        "test_falling": 1858,
    },
    3: {
        "train_subjects": 45,
        "val_subjects": 12,
        "test_subjects": 14,
        "train_total": 584876,
        "train_activity": 578530,
        "train_falling": 6346,
        "val_total": 510754,
        "val_activity": 509203,
        "val_falling": 1551,
        "test_total": 201537,
        "test_activity": 199628,
        "test_falling": 1909,
    },
    4: {
        "train_subjects": 45,
        "val_subjects": 12,
        "test_subjects": 14,
        "train_total": 899155,
        "train_activity": 893222,
        "train_falling": 5933,
        "val_total": 334883,
        "val_activity": 333390,
        "val_falling": 1493,
        "test_total": 63129,
        "test_activity": 60749,
        "test_falling": 2380,
    },
    5: {
        "train_subjects": 45,
        "val_subjects": 12,
        "test_subjects": 14,
        "train_total": 523851,
        "train_activity": 517556,
        "train_falling": 6295,
        "val_total": 487842,
        "val_activity": 486426,
        "val_falling": 1416,
        "test_total": 285474,
        "test_activity": 283379,
        "test_falling": 2095,
    },
}

PHYSICAL_GLOBAL = {
    "identities": 71,
    "kfall_identities": 32,
    "univr_lab_identities": 29,
    "univr_worksite_identities": 10,
    "recordings_events": 6325,
    "activity_events": 3396,
    "falling_events": 2929,
    "windows": 1297167,
    "activity_windows": 1287361,
    "falling_windows": 9806,
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def round_half_up(x: float) -> int:
    return int(np.floor(float(x) + 0.5))


def count01(y):
    y = np.asarray(y, dtype=np.int64)
    return {
        "total": int(len(y)),
        "activity": int((y == 0).sum()),
        "falling": int((y == 1).sum()),
    }


checks = []


def check(name, passed, detail=""):
    checks.append(
        {
            "check": str(name),
            "pass": bool(passed),
            "detail": str(detail),
        }
    )
    if not passed:
        raise RuntimeError(f"AUDIT CHECK FAILED: {name}: {detail}")


print("=" * 110)
print("EXPANDED42 V2 FINAL DATA-FLOW / LEAKAGE AUDIT")
print("=" * 110)


# ----------------------------------------------------------------------
# Canonical synthetic/event pool
# ----------------------------------------------------------------------

check("canonical_manifest_exists", CANON.exists(), CANON)

canon = pd.read_csv(CANON)
facts = json.loads(CANON_FACTS.read_text())

eligible = (
    canon["cnn_fall_eligible"]
    .astype(str)
    .str.lower()
    .isin(["true", "1", "yes"])
)

check("canonical_rows_756", len(canon) == 756, len(canon))
check(
    "canonical_profiles_42",
    canon["profile_id"].astype(str).nunique() == 42,
    canon["profile_id"].astype(str).nunique(),
)
check(
    "canonical_tasks_18",
    pd.to_numeric(canon["task_id"]).astype(int).nunique() == 18,
    pd.to_numeric(canon["task_id"]).astype(int).nunique(),
)
check("canonical_gate_pass", bool(facts["gate_pass"]), facts["gate_pass"])


# ----------------------------------------------------------------------
# Exact CNN synthetic cache
# ----------------------------------------------------------------------

x_path = CACHE / "synthetic_X_protechto_rawunits.npy"
y_path = CACHE / "synthetic_y01.npy"
m_path = CACHE / "synthetic_metadata_resolved.csv"

check("cache_X_exists", x_path.exists(), x_path)
check("cache_y_exists", y_path.exists(), y_path)
check("cache_metadata_exists", m_path.exists(), m_path)

X = np.load(x_path, mmap_mode="r")
y = np.load(y_path).astype(np.int64)
meta = pd.read_csv(m_path)

check(
    "cache_shape_N30x9",
    X.ndim == 3 and tuple(X.shape[1:]) == (30, 9),
    tuple(X.shape),
)

check(
    "cache_lengths_match",
    len(X) == len(y) == len(meta),
    f"X={len(X)} y={len(y)} meta={len(meta)}",
)

syn_counts = count01(y)

n_trajectories = int(
    meta["trial_uid_resolved"]
    .astype(str)
    .nunique()
)

n_profiles = int(
    meta["profile_id"]
    .astype(str)
    .nunique()
)

n_tasks = int(
    pd.to_numeric(meta["task_id"])
    .astype(int)
    .nunique()
)

check("cache_profiles_42", n_profiles == 42, n_profiles)
check("cache_tasks_18", n_tasks == 18, n_tasks)
check(
    "cache_trajectory_count_matches_canonical_eligible",
    n_trajectories == int(eligible.sum()),
    f"cache={n_trajectories}, canonical={int(eligible.sum())}",
)

check(
    "cache_activity_matches_canonical",
    syn_counts["activity"]
    == int(
        pd.to_numeric(
            canon.loc[eligible, "raw_activity_window_count"]
        ).sum()
    ),
    syn_counts,
)

check(
    "cache_falling_matches_canonical",
    syn_counts["falling"]
    == int(
        pd.to_numeric(
            canon.loc[eligible, "raw_falling_window_count"]
        ).sum()
    ),
    syn_counts,
)


# ----------------------------------------------------------------------
# Global dataset comparison
# ----------------------------------------------------------------------

global_rows = [
    {
        "domain": "Physical",
        "identities_or_profiles": PHYSICAL_GLOBAL["identities"],
        "source_trajectories_or_events": PHYSICAL_GLOBAL["recordings_events"],
        "cnn_eligible_trajectories": np.nan,
        "activity_windows": PHYSICAL_GLOBAL["activity_windows"],
        "falling_windows": PHYSICAL_GLOBAL["falling_windows"],
        "total_windows": PHYSICAL_GLOBAL["windows"],
        "notes":
            "71 physical identities = 32 KFall + 29 UniVrFall lab + "
            "10 UniVrFall worksite",
    },
    {
        "domain": "Simulated",
        "identities_or_profiles": n_profiles,
        "source_trajectories_or_events": 756,
        "cnn_eligible_trajectories": n_trajectories,
        "activity_windows": syn_counts["activity"],
        "falling_windows": syn_counts["falling"],
        "total_windows": syn_counts["total"],
        "notes":
            "42 parameterized profiles; 20 female / 22 male; 18 tasks. "
            "Synthetic Activity windows are pre-onset portions of simulated "
            "fall trajectories.",
    },
]

pd.DataFrame(global_rows).to_csv(
    OUT / "dataset_global_comparison.csv",
    index=False,
)


# ----------------------------------------------------------------------
# Frozen physical fold data use
# ----------------------------------------------------------------------

physical_rows = []

for fold, d in PHYSICAL_FOLDS.items():
    physical_rows.append(
        {
            "fold": fold,
            **d,
        }
    )

physical_df = pd.DataFrame(physical_rows)

check(
    "physical_test_total_pools_to_1297167",
    int(physical_df["test_total"].sum())
    == PHYSICAL_GLOBAL["windows"],
    int(physical_df["test_total"].sum()),
)

check(
    "physical_test_activity_pools_correctly",
    int(physical_df["test_activity"].sum())
    == PHYSICAL_GLOBAL["activity_windows"],
    int(physical_df["test_activity"].sum()),
)

check(
    "physical_test_falling_pools_correctly",
    int(physical_df["test_falling"].sum())
    == PHYSICAL_GLOBAL["falling_windows"],
    int(physical_df["test_falling"].sum()),
)

physical_df.to_csv(
    OUT / "physical_fold_split_counts.csv",
    index=False,
)


# ----------------------------------------------------------------------
# Simulator-only trajectory-disjoint folds
# ----------------------------------------------------------------------

uids = np.array(
    sorted(
        meta["trial_uid_resolved"]
        .astype(str)
        .unique()
    )
)

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

sim_fold_rows = []

trajectory_overlap_ok = True

for fold0, (train_idx_full, test_idx) in enumerate(
    kf.split(uids)
):
    fold = fold0 + 1

    train_idx, val_idx = train_test_split(
        train_idx_full,
        test_size=0.2,
        random_state=42,
    )

    train_uids = set(uids[train_idx])
    val_uids = set(uids[val_idx])
    test_uids = set(uids[test_idx])

    overlap_tv = len(train_uids & val_uids)
    overlap_tt = len(train_uids & test_uids)
    overlap_vt = len(val_uids & test_uids)

    if overlap_tv or overlap_tt or overlap_vt:
        trajectory_overlap_ok = False

    def partition(which):
        q = meta[
            meta["trial_uid_resolved"]
            .astype(str)
            .isin(which)
        ].copy()

        yy = q["label01"].to_numpy(dtype=np.int64)

        return {
            "trajectories": int(
                q["trial_uid_resolved"]
                .astype(str)
                .nunique()
            ),
            "windows": int(len(q)),
            "activity": int((yy == 0).sum()),
            "falling": int((yy == 1).sum()),
            "profiles": int(
                q["profile_id"]
                .astype(str)
                .nunique()
            ),
            "tasks": int(
                pd.to_numeric(q["task_id"])
                .astype(int)
                .nunique()
            ),
        }

    tr = partition(train_uids)
    va = partition(val_uids)
    te = partition(test_uids)

    train_profiles = set(
        meta.loc[
            meta["trial_uid_resolved"].astype(str).isin(train_uids),
            "profile_id",
        ].astype(str)
    )

    val_profiles = set(
        meta.loc[
            meta["trial_uid_resolved"].astype(str).isin(val_uids),
            "profile_id",
        ].astype(str)
    )

    test_profiles = set(
        meta.loc[
            meta["trial_uid_resolved"].astype(str).isin(test_uids),
            "profile_id",
        ].astype(str)
    )

    train_tasks = set(
        pd.to_numeric(
            meta.loc[
                meta["trial_uid_resolved"].astype(str).isin(train_uids),
                "task_id",
            ]
        ).astype(int)
    )

    val_tasks = set(
        pd.to_numeric(
            meta.loc[
                meta["trial_uid_resolved"].astype(str).isin(val_uids),
                "task_id",
            ]
        ).astype(int)
    )

    test_tasks = set(
        pd.to_numeric(
            meta.loc[
                meta["trial_uid_resolved"].astype(str).isin(test_uids),
                "task_id",
            ]
        ).astype(int)
    )

    sim_fold_rows.append(
        {
            "fold": fold,

            "train_trajectories": tr["trajectories"],
            "train_windows": tr["windows"],
            "train_activity": tr["activity"],
            "train_falling": tr["falling"],

            "val_trajectories": va["trajectories"],
            "val_windows": va["windows"],
            "val_activity": va["activity"],
            "val_falling": va["falling"],

            "test_trajectories": te["trajectories"],
            "test_windows": te["windows"],
            "test_activity": te["activity"],
            "test_falling": te["falling"],

            "train_val_trajectory_overlap": overlap_tv,
            "train_test_trajectory_overlap": overlap_tt,
            "val_test_trajectory_overlap": overlap_vt,

            "train_val_profile_overlap":
                len(train_profiles & val_profiles),

            "train_test_profile_overlap":
                len(train_profiles & test_profiles),

            "val_test_profile_overlap":
                len(val_profiles & test_profiles),

            "train_val_task_overlap":
                len(train_tasks & val_tasks),

            "train_test_task_overlap":
                len(train_tasks & test_tasks),

            "val_test_task_overlap":
                len(val_tasks & test_tasks),
        }
    )

check(
    "simulator_only_trajectory_overlap_zero",
    trajectory_overlap_ok,
    "trajectory split",
)

sim_fold_df = pd.DataFrame(sim_fold_rows)

check(
    "simulator_only_test_trajectories_partition_full_pool",
    int(sim_fold_df["test_trajectories"].sum())
    == n_trajectories,
    int(sim_fold_df["test_trajectories"].sum()),
)

sim_fold_df.to_csv(
    OUT / "simulator_only_fold_counts.csv",
    index=False,
)


# ----------------------------------------------------------------------
# Campaign/run IDs and held-out physical test invariance
# ----------------------------------------------------------------------

check(
    "campaign_complete_marker",
    (CAMPAIGN / "FINAL_CAMPAIGN_COMPLETE").exists(),
    CAMPAIGN / "FINAL_CAMPAIGN_COMPLETE",
)

run_id = (
    CAMPAIGN
    / "RUN_ID"
).read_text().strip()

RESULT_ROOT = (
    PROTECHTO
    / "results"
    / "CNN"
    / "300ms"
    / run_id
)

check(
    "result_root_exists",
    RESULT_ROOT.exists(),
    RESULT_ROOT,
)

test_invariance_rows = []

for fold in range(1, 6):
    baseline_path = (
        RESULT_ROOT
        / "EXP01_REAL_ONLY"
        / str(fold)
        / "y_true.npy"
    )

    check(
        f"real_only_fold{fold}_y_true_exists",
        baseline_path.exists(),
        baseline_path,
    )

    baseline = np.load(
        baseline_path
    ).astype(np.int64)

    expected = PHYSICAL_FOLDS[fold]

    check(
        f"physical_fold{fold}_test_total",
        len(baseline) == expected["test_total"],
        f"{len(baseline)} vs {expected['test_total']}",
    )

    check(
        f"physical_fold{fold}_test_activity",
        int((baseline == 0).sum())
        == expected["test_activity"],
        int((baseline == 0).sum()),
    )

    check(
        f"physical_fold{fold}_test_falling",
        int((baseline == 1).sum())
        == expected["test_falling"],
        int((baseline == 1).sum()),
    )

    for condition in PHYSICAL_TEST_CONDITIONS:
        p = (
            RESULT_ROOT
            / condition
            / str(fold)
            / "y_true.npy"
        )

        check(
            f"{condition}_fold{fold}_y_true_exists",
            p.exists(),
            p,
        )

        yt = np.load(
            p
        ).astype(np.int64)

        same = (
            len(yt) == len(baseline)
            and np.array_equal(
                yt,
                baseline,
            )
        )

        test_invariance_rows.append(
            {
                "fold": fold,
                "condition": condition,
                "test_windows": int(len(yt)),
                "identical_to_real_only_y_true": bool(same),
            }
        )

        check(
            f"{condition}_fold{fold}_heldout_test_identical",
            same,
            condition,
        )

pd.DataFrame(
    test_invariance_rows
).to_csv(
    OUT / "physical_test_invariance.csv",
    index=False,
)


# ----------------------------------------------------------------------
# Mixed synthetic-Falling draws / reuse
# ----------------------------------------------------------------------

mixed_rows = []

all_mixed_falling_only = True

for condition, ratio in RATIOS.items():
    for fold in range(1, 6):
        p = (
            CAMPAIGN
            / "synthetic_draw_manifests"
            / f"{condition}_fold{fold}.csv"
        )

        check(
            f"{condition}_fold{fold}_draw_manifest_exists",
            p.exists(),
            p,
        )

        d = pd.read_csv(p)

        expected_real_fall = (
            PHYSICAL_FOLDS[fold]["train_falling"]
        )

        expected_draws = round_half_up(
            expected_real_fall
            * ratio
        )

        check(
            f"{condition}_fold{fold}_target_draw_count",
            len(d) == expected_draws,
            f"{len(d)} vs {expected_draws}",
        )

        rows = (
            pd.to_numeric(
                d["synthetic_row"],
                errors="raise",
            )
            .astype(int)
            .to_numpy()
        )

        selected_meta = (
            meta.set_index(
                "synthetic_row",
                drop=False,
            )
            .loc[rows]
            .copy()
        )

        falling_only = bool(
            (
                selected_meta["label01"]
                .astype(int)
                == 1
            ).all()
        )

        all_mixed_falling_only &= falling_only

        unique_rows = int(
            np.unique(rows).size
        )

        repeated = int(
            len(rows)
            - unique_rows
        )

        mixed_rows.append(
            {
                "condition": condition,
                "ratio_of_physical_train_falling": ratio,
                "fold": fold,
                "physical_train_activity":
                    PHYSICAL_FOLDS[fold]["train_activity"],
                "physical_train_falling":
                    expected_real_fall,
                "physical_train_total":
                    PHYSICAL_FOLDS[fold]["train_total"],
                "simulated_falling_draws":
                    int(len(rows)),
                "unique_simulated_falling_windows":
                    unique_rows,
                "repeated_draws":
                    repeated,
                "repeat_fraction":
                    float(
                        repeated / len(rows)
                        if len(rows)
                        else 0.0
                    ),
                "selected_trajectories":
                    int(
                        selected_meta[
                            "trial_uid_resolved"
                        ]
                        .astype(str)
                        .nunique()
                    ),
                "selected_profiles":
                    int(
                        selected_meta[
                            "profile_id"
                        ]
                        .astype(str)
                        .nunique()
                    ),
                "selected_tasks":
                    int(
                        pd.to_numeric(
                            selected_meta[
                                "task_id"
                            ]
                        )
                        .astype(int)
                        .nunique()
                    ),
                "simulated_validation_windows":
                    0,
                "simulated_test_windows":
                    0,
            }
        )

check(
    "mixed_draws_are_falling_only",
    all_mixed_falling_only,
    "all mixed draw manifests",
)

mixed_df = pd.DataFrame(
    mixed_rows
)

mixed_df.to_csv(
    OUT / "mixed_fold_data_usage.csv",
    index=False,
)

reuse_summary = (
    mixed_df
    .groupby(
        [
            "condition",
            "ratio_of_physical_train_falling",
        ],
        as_index=False,
    )
    .agg(
        draws=(
            "simulated_falling_draws",
            "sum",
        ),
        unique_draws_sum_across_folds=(
            "unique_simulated_falling_windows",
            "sum",
        ),
        repeated_draws=(
            "repeated_draws",
            "sum",
        ),
        min_draws_per_fold=(
            "simulated_falling_draws",
            "min",
        ),
        max_draws_per_fold=(
            "simulated_falling_draws",
            "max",
        ),
        min_unique_per_fold=(
            "unique_simulated_falling_windows",
            "min",
        ),
        max_unique_per_fold=(
            "unique_simulated_falling_windows",
            "max",
        ),
    )
)

reuse_summary[
    "overall_repeat_fraction"
] = (
    reuse_summary[
        "repeated_draws"
    ]
    /
    reuse_summary[
        "draws"
    ]
)

reuse_summary.to_csv(
    OUT / "mixed_reuse_summary.csv",
    index=False,
)


# ----------------------------------------------------------------------
# Final pooled results from the exact post-hoc collector
# ----------------------------------------------------------------------

POST = (
    CAMPAIGN
    / "final_results_posthoc"
)

pooled_path = (
    POST
    / "POOLED_ALL_RESULTS.csv"
)

fold_metrics_path = (
    POST
    / "ALL_FOLD_METRICS.csv"
)

fold_stats_path = (
    POST
    / "FOLD_MEAN_STD.csv"
)

for p in [
    pooled_path,
    fold_metrics_path,
    fold_stats_path,
]:
    check(
        f"collector_artifact_{p.name}",
        p.exists(),
        p,
    )

pooled = pd.read_csv(
    pooled_path
)

conditions_seen = set(
    pooled["condition"]
    .astype(str)
    .unique()
)

check(
    "pooled_contains_exactly_six_paper_conditions",
    conditions_seen
    == set(PAPER_CONDITIONS),
    sorted(conditions_seen),
)

check(
    "exp02_absent_from_v2_results",
    "EXP02_SIM_FALL_SUBSTITUTION"
    not in conditions_seen,
    sorted(conditions_seen),
)

pooled.to_csv(
    OUT / "pooled_results_copy.csv",
    index=False,
)

pd.read_csv(
    fold_metrics_path
).to_csv(
    OUT / "fold_metrics_copy.csv",
    index=False,
)

pd.read_csv(
    fold_stats_path
).to_csv(
    OUT / "fold_mean_std_copy.csv",
    index=False,
)


# ----------------------------------------------------------------------
# Physical pooled support gate from collector
# ----------------------------------------------------------------------

phys_segment = pooled[
    (pooled["condition"] == "EXP01_REAL_ONLY")
    &
    (pooled["level"] == "Segment")
]

phys_event = pooled[
    (pooled["condition"] == "EXP01_REAL_ONLY")
    &
    (pooled["level"] == "Event")
]

check(
    "one_real_only_segment_row",
    len(phys_segment) == 1,
    len(phys_segment),
)

check(
    "one_real_only_event_row",
    len(phys_event) == 1,
    len(phys_event),
)

seg = phys_segment.iloc[0]
ev = phys_event.iloc[0]

check(
    "physical_pooled_segment_N_1297167",
    int(seg["n"])
    == PHYSICAL_GLOBAL["windows"],
    int(seg["n"]),
)

check(
    "physical_pooled_segment_activity_support",
    int(seg["support_activity"])
    == PHYSICAL_GLOBAL["activity_windows"],
    int(seg["support_activity"]),
)

check(
    "physical_pooled_segment_fall_support",
    int(seg["support_fall"])
    == PHYSICAL_GLOBAL["falling_windows"],
    int(seg["support_fall"]),
)

check(
    "physical_pooled_event_N_6325",
    int(ev["n"])
    == PHYSICAL_GLOBAL["recordings_events"],
    int(ev["n"]),
)

check(
    "physical_pooled_event_activity_support",
    int(ev["support_activity"])
    == PHYSICAL_GLOBAL["activity_events"],
    int(ev["support_activity"]),
)

check(
    "physical_pooled_event_fall_support",
    int(ev["support_fall"])
    == PHYSICAL_GLOBAL["falling_events"],
    int(ev["support_fall"]),
)


# ----------------------------------------------------------------------
# Integrity output
# ----------------------------------------------------------------------

checks_df = pd.DataFrame(
    checks
)

checks_df.to_csv(
    OUT / "integrity_checks.csv",
    index=False,
)

check_count = int(
    len(checks_df)
)

pass_count = int(
    checks_df["pass"]
    .sum()
)

if pass_count != check_count:
    raise RuntimeError(
        f"Integrity pass mismatch: {pass_count}/{check_count}"
    )


# ----------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------

provenance = {
    "audit_version":
        "expanded42_final_dataflow_v2",

    "canonical_manifest":
        str(CANON),

    "canonical_manifest_sha256":
        sha256_file(CANON),

    "synthetic_cache":
        str(CACHE),

    "campaign":
        str(CAMPAIGN),

    "run_id":
        run_id,

    "result_root":
        str(RESULT_ROOT),

    "profiles":
        n_profiles,

    "tasks":
        n_tasks,

    "source_simulations":
        756,

    "eligible_trajectories":
        n_trajectories,

    "synthetic_activity_windows":
        syn_counts["activity"],

    "synthetic_falling_windows":
        syn_counts["falling"],

    "synthetic_total_windows":
        syn_counts["total"],

    "physical_global":
        PHYSICAL_GLOBAL,

    "paper_conditions":
        PAPER_CONDITIONS,

    "integrity_checks_passed":
        pass_count,

    "integrity_checks_total":
        check_count,
}

(
    OUT
    / "AUDIT_PROVENANCE.json"
).write_text(
    json.dumps(
        provenance,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)


# ----------------------------------------------------------------------
# Frozen writing/evidence record
# ----------------------------------------------------------------------

lines = []

def w(x=""):
    lines.append(str(x))

w("FINAL EXPANDED42 DATASET / DATA-FLOW / LEAKAGE EVIDENCE RECORD")
w("=" * 100)
w()
w("SCOPE")
w("-" * 100)
w("Current manuscript comparison only:")
w("  Physical-only")
w("  Simulator-only")
w("  Physical + SIM20")
w("  Physical + SIM50")
w("  Physical + SIM70")
w("  Physical + SIM100")
w()
w("EXP02_SIM_FALL_SUBSTITUTION is not part of this V2 campaign.")
w()
w("PHYSICAL DATA")
w("-" * 100)
w("Identities: 71 = 32 KFall + 29 UniVrFall laboratory + 10 UniVrFall worksite")
w("Recordings/events: 6,325 = 3,396 Activity + 2,929 Falling")
w("Windows: 1,297,167 = 1,287,361 Activity + 9,806 Falling")
w("Physical splitting remains five-fold subject-independent with zero subject overlap.")
w("Physical validation/test folds are unchanged across Physical-only and all mixed conditions.")
w()
w("EXPANDED SIMULATED DATA")
w("-" * 100)
w("Raw profile-task simulations: 756 = 42 profiles x 18 tasks")
w("Profiles: 42 = 20 female + 22 male")
w(f"CNN-eligible complete trajectories: {n_trajectories}")
w(
    f"CNN windows: {syn_counts['total']:,} = "
    f"{syn_counts['activity']:,} Activity + "
    f"{syn_counts['falling']:,} Falling"
)
w()
w(
    "Synthetic Activity windows are pre-onset portions of the same "
    "simulated fall trajectories; they are not an independent synthetic "
    "ADL-only trial collection."
)
w()
w("SIMULATOR-ONLY SPLIT")
w("-" * 100)
w(
    "Atomic split unit: complete trial_uid_resolved trajectory. "
    "Five-fold KFold random_state=42 followed by 80/20 train/validation "
    "trajectory split with random_state=42."
)
w("Trajectory train/validation/test overlap is zero in every fold.")
w(
    "This is trajectory-disjoint, not profile-held-out or task-held-out; "
    "profile/task overlap across partitions can exist."
)
w()
w("MIXED CONDITIONS")
w("-" * 100)
w(
    "Only simulated Falling windows are added to PHYSICAL TRAINING. "
    "Physical validation and held-out physical testing remain unchanged; "
    "simulated validation/test windows = 0."
)
w()
w(
    "SIM20/SIM50/SIM70/SIM100 are defined relative to each fold's "
    "physical TRAIN Falling-window count, not relative to all training windows."
)
w()

for r in reuse_summary.itertuples():
    w(
        f"{r.condition}: total draws={int(r.draws):,}; "
        f"per-fold draws={int(r.min_draws_per_fold):,}-"
        f"{int(r.max_draws_per_fold):,}; "
        f"unique per fold={int(r.min_unique_per_fold):,}-"
        f"{int(r.max_unique_per_fold):,}; "
        f"repeated draws={int(r.repeated_draws):,}; "
        f"overall repeat fraction={100.0*float(r.overall_repeat_fraction):.2f}%"
    )

w()
w(
    "Higher ratios therefore increase training exposure to the finite "
    "synthetic Falling pool whenever the requested number of draws exceeds "
    "the available unique Falling windows; repeated draws are not independent "
    "new observations."
)
w()
w("INTEGRITY")
w("-" * 100)
w(f"Checks passed: {pass_count}/{check_count}")
w("Held-out physical test labels are identical across all five physical-test conditions.")
w("All mixed synthetic draws are Falling-class training samples.")
w("Simulator-only trajectory overlap across train/validation/test is zero.")
w("EXP02 is absent from the V2 pooled results.")
w()
w("RESULTS")
w("-" * 100)
w("Authoritative pooled results are stored in:")
w(f"  {pooled_path}")
w()
w("Paper-ready V2 pooled rows:")
w()

display_cols = [
    "condition",
    "level",
    "n",
    "balanced_accuracy",
    "fall_precision",
    "fall_recall",
    "fall_f1",
]

for _, r in pooled[display_cols].iterrows():
    w(
        f"{r['condition']:24s} "
        f"{r['level']:7s} "
        f"N={int(r['n']):7d} "
        f"BalAcc={100*float(r['balanced_accuracy']):6.2f}% "
        f"PrecF={100*float(r['fall_precision']):6.2f}% "
        f"RecallF={100*float(r['fall_recall']):6.2f}% "
        f"F1F={100*float(r['fall_f1']):6.2f}%"
    )

w()
w("INTERPRETATION BOUNDARY")
w("-" * 100)
w(
    "This audit establishes dataset composition, partition integrity, "
    "trajectory/subject separation and synthetic-reuse accounting. "
    "It does not by itself prove absence of model overfitting."
)

evidence_path = (
    OUT
    / "FINAL_EXPANDED42_DATAFLOW_LEAKAGE_EVIDENCE_RECORD.txt"
)

evidence_path.write_text(
    "\n".join(lines)
    + "\n"
)

print()
print("=" * 110)
print("EXPANDED42 FINAL AUDIT COMPLETE")
print("=" * 110)
print("Physical windows :", PHYSICAL_GLOBAL["windows"])
print("Synthetic windows:", syn_counts["total"])
print("Synthetic A/F    :", syn_counts["activity"], syn_counts["falling"])
print("Trajectories     :", n_trajectories)
print("Profiles/tasks   :", n_profiles, n_tasks)
print("Integrity        :", f"{pass_count}/{check_count} PASS")
print("Output           :", OUT)
print("Evidence record  :", evidence_path)
print("=" * 110)
