from pathlib import Path
import json
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

INPUT = (
    CAMPAIGN
    / "highrate_semantic_event_manifest_v2.csv"
)

FINAL = (
    CAMPAIGN
    / "highrate_event_manifest_FINAL.csv"
)

AUDIT = (
    CAMPAIGN
    / "generic_onset_finalization_audit.csv"
)


EARLY_TASKS = {
    20, 21, 22, 23, 24,
    25, 26, 27, 250,
}


def finite(x):
    try:
        x = float(x)

        if math.isfinite(x):
            return x

    except Exception:
        pass

    return None


def norm_path(x):
    p = Path(str(x))

    if not p.is_absolute():
        p = ROOT / p

    return p


def read_truth(path):
    return pd.read_csv(
        path,
        comment="#",
    )


def metadata_onset(current):
    p = current.with_name(
        current.stem
        + ".label_metadata.json"
    )

    if not p.exists():
        return None

    try:
        obj = json.loads(
            p.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        )
    except Exception:
        return None

    hits = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():

                if (
                    str(k).lower()
                    == "fall_onset_time_s"
                ):
                    q = finite(v)

                    if q is not None:
                        hits.append(q)

                walk(v)

        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)

    return hits[0] if hits else None


def generic_onset_exact(pristine):
    """
    Exact generic pelvis-height fallback shown in
    imu_pipeline_labels.py.

    Relevant source behavior:

      early = time 0.2 .. min(3.0, recording end)

      h_ref = median pelvis_height[early]

      threshold =
          h_ref
          - max(
                0.18,
                0.18 * max(h_ref, 1.0)
            )

      search = time >= 1.0

      onset = first pelvis_height <= threshold

    The high-rate pristine stream has no fall_detected
    channel, so pelvis-height is the applicable generic
    fallback for these recordings.
    """

    df = read_truth(
        pristine
    )

    if "timestamp" not in df.columns:
        raise RuntimeError(
            "timestamp missing"
        )

    if "pelvis_height" not in df.columns:
        raise RuntimeError(
            "pelvis_height missing"
        )

    t = pd.to_numeric(
        df["timestamp"],
        errors="coerce",
    ).to_numpy(float)

    h = pd.to_numeric(
        df["pelvis_height"],
        errors="coerce",
    ).to_numpy(float)

    valid_t = t[
        np.isfinite(t)
    ]

    if not len(valid_t):
        raise RuntimeError(
            "no finite timestamps"
        )

    tmax = float(
        np.nanmax(valid_t)
    )

    early = (
        (t >= 0.2)
        &
        (
            t
            <= min(
                3.0,
                tmax,
            )
        )
    )

    if not np.any(
        early
        &
        np.isfinite(h)
    ):
        raise RuntimeError(
            "generic baseline window unavailable"
        )

    h_ref = float(
        np.nanmedian(
            h[
                early
                &
                np.isfinite(h)
            ]
        )
    )

    threshold = (
        h_ref
        -
        max(
            0.18,
            0.18
            * max(
                h_ref,
                1.0,
            ),
        )
    )

    hits = np.where(
        (t >= 1.0)
        &
        np.isfinite(h)
        &
        (h <= threshold)
    )[0]

    if not len(hits):
        raise RuntimeError(
            "generic pelvis-height onset not found"
        )

    idx = int(
        hits[0]
    )

    return {
        "onset_s": float(
            t[idx]
        ),
        "index": idx,
        "h_ref_m": h_ref,
        "threshold_m": float(
            threshold
        ),
        "pelvis_height_at_onset_m":
            float(h[idx]),
        "source":
            (
                "generic_pelvis_height_drop_"
                f"h_ref={h_ref:.3f}_"
                f"threshold={threshold:.3f}"
            ),
    }


def dynamics_path(current):
    ending = "_highrate_truth.csv"

    if not current.name.endswith(
        ending
    ):
        raise RuntimeError(
            "unexpected high-rate filename"
        )

    stem = current.name[
        :-len(ending)
    ]

    return current.with_name(
        stem
        + "_dynamics_frames.csv"
    )


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c

    lower = {
        str(c).lower(): c
        for c in df.columns
    }

    for c in candidates:
        if c.lower() in lower:
            return lower[
                c.lower()
            ]

    return None


def dynamics_impact(
    dyn_path,
    onset,
    profile_weight,
):
    """
    Previously validated DynamicsLayer2 rule:

      after onset, first primary impact body load > 0.25 BW.

    On 221 known records this reproduced the saved
    main-impact time with maximum discrepancy <0.35 ms.
    """

    df = pd.read_csv(
        dyn_path
    )

    time_col = find_col(
        df,
        [
            "time",
            "timestamp",
            "time_s",
        ],
    )

    force_col = find_col(
        df,
        [
            "primary_impact_body_load_n_filt",
            "primary_impact_body_load_n",
        ],
    )

    support_col = find_col(
        df,
        [
            "support_vertical_n",
        ],
    )

    if time_col is None:
        raise RuntimeError(
            "dynamics time column missing"
        )

    if force_col is None:
        raise RuntimeError(
            "primary impact load missing"
        )

    t = pd.to_numeric(
        df[time_col],
        errors="coerce",
    ).to_numpy(float)

    force = (
        pd.to_numeric(
            df[force_col],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(float)
    )

    mass = finite(
        profile_weight
    )

    if mass is None:
        mass = 70.0

    mass_source = (
        "profile_weight"
    )

    if support_col is not None:

        support = (
            pd.to_numeric(
                df[support_col],
                errors="coerce",
            )
            .fillna(0)
            .to_numpy(float)
        )

        early = support[
            :min(
                len(support),
                30,
            )
        ]

        positive = early[
            early > 0
        ]

        if len(positive):
            mass = max(
                1.0,
                float(
                    np.median(
                        positive
                    )
                    / 9.81
                ),
            )

            mass_source = (
                "early_support_vertical_n"
            )

    BW = max(
        1.0,
        mass * 9.81,
    )

    imp_bw = (
        force / BW
    )

    post = np.where(
        np.isfinite(t)
        &
        (t >= onset)
    )[0]

    if not len(post):
        raise RuntimeError(
            "no post-onset dynamics frames"
        )

    j0 = int(
        post[0]
    )

    crossing = np.where(
        imp_bw[
            j0:
        ] > 0.25
    )[0]

    if len(crossing):

        j = (
            j0
            + int(
                crossing[0]
            )
        )

        method = (
            "first_primary_body_"
            "impact_gt_0p25BW"
        )

    else:

        segment = imp_bw[
            j0:
        ]

        if not len(segment):
            raise RuntimeError(
                "empty post-onset impact segment"
            )

        j = (
            j0
            + int(
                np.nanargmax(
                    segment
                )
            )
        )

        method = (
            "fallback_postonset_"
            "max_primary_body_impact"
        )

    return {
        "impact_s":
            float(t[j]),

        "impact_index":
            int(j),

        "impact_force_n":
            float(force[j]),

        "impact_bw":
            float(imp_bw[j]),

        "mass_kg":
            float(mass),

        "mass_source":
            mass_source,

        "source":
            method,
    }


if not INPUT.exists():
    raise SystemExit(
        f"Missing:\n{INPUT}"
    )


m = pd.read_csv(
    INPUT
)

audit_rows = []


# ============================================================
# STAGE 1:
# Validate exact generic rule against Task 20 metadata.
# ============================================================

for _, r in m[
    m["scenario_id"] == 20
].iterrows():

    current = norm_path(
        r["current_truth_csv"]
    )

    pristine = norm_path(
        r["pristine_truth_source"]
    )

    expected = metadata_onset(
        current
    )

    result = {
        "sim_age_profile":
            int(
                r[
                    "sim_age_profile"
                ]
            ),

        "expected_task20_onset_s":
            expected,

        "candidate_generic_onset_s":
            np.nan,

        "abs_error_s":
            np.nan,

        "h_ref_m":
            np.nan,

        "threshold_m":
            np.nan,

        "status":
            "FAIL",

        "error":
            "",
    }

    try:
        if expected is None:
            raise RuntimeError(
                "Task20 reference metadata onset missing"
            )

        rec = generic_onset_exact(
            pristine
        )

        result[
            "candidate_generic_onset_s"
        ] = rec[
            "onset_s"
        ]

        result[
            "h_ref_m"
        ] = rec[
            "h_ref_m"
        ]

        result[
            "threshold_m"
        ] = rec[
            "threshold_m"
        ]

        result[
            "abs_error_s"
        ] = abs(
            rec[
                "onset_s"
            ]
            - expected
        )

        result[
            "status"
        ] = "OK"

    except Exception as exc:
        result[
            "error"
        ] = str(exc)

    audit_rows.append(
        result
    )


audit = pd.DataFrame(
    audit_rows
)

audit.to_csv(
    AUDIT,
    index=False,
)


print("=" * 100)
print("GENERIC ONSET TASK-20 VALIDATION")
print("=" * 100)

print(
    "Task20 records:",
    len(audit),
)

print(
    "reconstructed:",
    int(
        (
            audit.status
            == "OK"
        ).sum()
    ),
)

print()

print(
    audit[
        [
            "sim_age_profile",
            "expected_task20_onset_s",
            "candidate_generic_onset_s",
            "abs_error_s",
            "h_ref_m",
            "threshold_m",
        ]
    ].to_string(
        index=False
    )
)


if len(audit):

    errors = audit[
        "abs_error_s"
    ].dropna()

    print()

    print(
        "median abs error [s]:",
        errors.median()
        if len(errors)
        else np.nan,
    )

    print(
        "max abs error [s]   :",
        errors.max()
        if len(errors)
        else np.nan,
    )


# 100-Hz generic detector:
# accept <= one sample + small floating rounding.
generic_validated = (
    len(audit) == 13
    and bool(
        (
            audit.status
            == "OK"
        ).all()
    )
    and float(
        audit[
            "abs_error_s"
        ].max()
    ) <= 0.011
)


print()
print(
    "GENERIC_ONSET_RULE_VALIDATED:",
    generic_validated,
)


if not generic_validated:

    print()
    print(
        "STOP: final manifest was NOT written."
    )

    print(
        "Review generic onset validation first."
    )

    raise SystemExit(2)


# ============================================================
# STAGE 2:
# Create final event manifest.
#
# 221 known:
#   preserve onset + impact.
#
# Task20:
#   preserve metadata onset,
#   recover missing impact.
#
# Tasks21-27/250:
#   exact generic onset,
#   validated DynamicsLayer2 impact.
# ============================================================

rows = []


for _, r in m.iterrows():

    x = r.to_dict()

    task = int(
        r[
            "scenario_id"
        ]
    )

    current = norm_path(
        r[
            "current_truth_csv"
        ]
    )

    pristine = norm_path(
        r[
            "pristine_truth_source"
        ]
    )

    old_onset = finite(
        r.get(
            "fall_onset_time_s"
        )
    )

    old_impact = finite(
        r.get(
            "main_impact_time_s"
        )
    )

    x[
        "original_fall_onset_time_s"
    ] = old_onset

    x[
        "original_main_impact_time_s"
    ] = old_impact

    x[
        "onset_recovery_method"
    ] = "preserved_existing"

    x[
        "impact_recovery_method"
    ] = "preserved_existing"

    x[
        "event_recovery_status"
    ] = "FAIL"

    x[
        "event_recovery_error"
    ] = ""

    try:

        # ----------------------------------------------------
        # ONSET
        # ----------------------------------------------------

        if old_onset is not None:

            onset = old_onset

            onset_source = str(
                r.get(
                    "event_source",
                    "preserved_existing",
                )
            )

        elif task == 20:

            # Should normally not be reached because Task20
            # reference onset exists in metadata, but keep
            # deterministic recovery available.
            onset = metadata_onset(
                current
            )

            if onset is None:
                raise RuntimeError(
                    "Task20 metadata onset missing"
                )

            onset_source = (
                "preserved_task20_"
                "generic_metadata_onset"
            )

            x[
                "onset_recovery_method"
            ] = onset_source

        elif task in EARLY_TASKS:

            rec = generic_onset_exact(
                pristine
            )

            onset = rec[
                "onset_s"
            ]

            onset_source = (
                "offline_exact_existing_"
                + rec[
                    "source"
                ]
            )

            x[
                "onset_recovery_method"
            ] = onset_source

        else:
            raise RuntimeError(
                "unexpected missing onset "
                f"for scenario {task}"
            )

        # ----------------------------------------------------
        # IMPACT
        # ----------------------------------------------------

        if old_impact is not None:

            impact = old_impact

            impact_source = str(
                r.get(
                    "event_source",
                    "preserved_existing",
                )
            )

        else:

            dyn = dynamics_path(
                current
            )

            if not dyn.exists():
                raise RuntimeError(
                    "dynamics CSV missing"
                )

            imp = dynamics_impact(
                dyn,
                onset,
                r.get(
                    "weight_kg",
                    np.nan,
                ),
            )

            impact = imp[
                "impact_s"
            ]

            impact_source = (
                "offline_validated_"
                + imp[
                    "source"
                ]
            )

            x[
                "impact_recovery_method"
            ] = impact_source

            x[
                "recovered_impact_force_n"
            ] = imp[
                "impact_force_n"
            ]

            x[
                "recovered_impact_bw"
            ] = imp[
                "impact_bw"
            ]

        # ----------------------------------------------------
        # Basic physical/event consistency
        # ----------------------------------------------------

        if impact < onset:
            raise RuntimeError(
                f"impact {impact} < onset {onset}"
            )

        truth = read_truth(
            pristine
        )

        tt = pd.to_numeric(
            truth["timestamp"],
            errors="coerce",
        ).to_numpy(float)

        tt = tt[
            np.isfinite(tt)
        ]

        t0 = float(
            np.min(tt)
        )

        t1 = float(
            np.max(tt)
        )

        tol = 0.05

        if not (
            t0 - tol
            <= onset
            <= t1 + tol
        ):
            raise RuntimeError(
                "onset outside truth timeline"
            )

        if not (
            t0 - tol
            <= impact
            <= t1 + tol
        ):
            raise RuntimeError(
                "impact outside truth timeline"
            )

        x[
            "fall_onset_time_s"
        ] = onset

        x[
            "main_impact_time_s"
        ] = impact

        x[
            "final_onset_source"
        ] = onset_source

        x[
            "final_impact_source"
        ] = impact_source

        x[
            "onset_to_impact_s"
        ] = (
            impact
            - onset
        )

        x[
            "event_valid"
        ] = True

        x[
            "status"
        ] = "OK"

        x[
            "event_recovery_status"
        ] = "OK"

    except Exception as exc:

        x[
            "event_valid"
        ] = False

        x[
            "status"
        ] = "FAIL"

        x[
            "event_recovery_error"
        ] = str(exc)

    rows.append(
        x
    )


final = pd.DataFrame(
    rows
)


# ============================================================
# FINAL VALIDATION
# ============================================================

preserved_both = final[
    final[
        "original_fall_onset_time_s"
    ].notna()
    &
    final[
        "original_main_impact_time_s"
    ].notna()
]

task20 = final[
    final[
        "scenario_id"
    ] == 20
]

generic_recovered = final[
    final[
        "onset_recovery_method"
    ].astype(str)
    .str.startswith(
        "offline_exact_existing_"
    )
]

impact_recovered = final[
    final[
        "impact_recovery_method"
    ].astype(str)
    .str.startswith(
        "offline_validated_"
    )
]


print()
print("=" * 100)
print("FINAL EVENT MANIFEST VALIDATION")
print("=" * 100)

print(
    "records:",
    len(final),
)

print(
    "OK:",
    int(
        (
            final[
                "event_recovery_status"
            ]
            == "OK"
        ).sum()
    ),
)

print(
    "FAIL:",
    int(
        (
            final[
                "event_recovery_status"
            ]
            != "OK"
        ).sum()
    ),
)

print()

print(
    "preserved onset+impact records:",
    len(
        preserved_both
    ),
)

print(
    "generic onsets recovered:",
    len(
        generic_recovered
    ),
)

print(
    "impacts recovered:",
    len(
        impact_recovered
    ),
)

print()
print("Final onset source:")
print(
    final[
        "final_onset_source"
    ]
    .value_counts()
    .to_string()
)

print()
print("Final impact source:")
print(
    final[
        "final_impact_source"
    ]
    .value_counts()
    .to_string()
)

print()
print("Onset->impact duration by scenario:")

print(
    final.groupby(
        "scenario_id"
    )[
        "onset_to_impact_s"
    ]
    .agg(
        [
            "count",
            "median",
            "min",
            "max",
        ]
    )
    .to_string()
)


print()
print(
    "Scenario counts:"
)

print(
    final.groupby(
        "scenario_id"
    )
    .size()
    .to_string()
)


# Check preserved 221 were not changed.
preserve_exact = True

if len(preserved_both):

    onset_same = np.allclose(
        preserved_both[
            "fall_onset_time_s"
        ],
        preserved_both[
            "original_fall_onset_time_s"
        ],
        rtol=0,
        atol=0,
    )

    impact_same = np.allclose(
        preserved_both[
            "main_impact_time_s"
        ],
        preserved_both[
            "original_main_impact_time_s"
        ],
        rtol=0,
        atol=0,
    )

    preserve_exact = bool(
        onset_same
        and impact_same
    )


counts = (
    final.groupby(
        "scenario_id"
    )
    .size()
)

final_ok = (
    len(final) == 338
    and bool(
        (
            final[
                "event_recovery_status"
            ]
            == "OK"
        ).all()
    )
    and len(counts) == 26
    and bool(
        (counts == 13).all()
    )
    and len(
        preserved_both
    ) == 221
    and preserve_exact
    and len(
        generic_recovered
    ) == 104
    and len(
        impact_recovered
    ) == 117
)


print()
print(
    "PRESERVED_221_EXACTLY:",
    preserve_exact,
)

print(
    "FINAL_EVENT_MANIFEST_OK:",
    final_ok,
)


if not final_ok:

    print()
    print(
        "STOP: final manifest was NOT written."
    )

    bad = final[
        final[
            "event_recovery_status"
        ]
        != "OK"
    ]

    if len(bad):
        print()
        print(
            bad[
                [
                    "scenario_id",
                    "sim_age_profile",
                    "event_recovery_error",
                ]
            ].to_string(
                index=False
            )
        )

    raise SystemExit(3)


# Only write after every validation gate passes.
final.to_csv(
    FINAL,
    index=False,
)


print()
print(
    "FINAL_EVENT_RECOVERY_OK"
)

print(
    "All 338 recordings now have "
    "validated onset and impact annotations."
)

print(
    "No simulator rerun is required."
)

print()
print(
    "Final manifest:",
    FINAL,
)

print(
    "Generic validation audit:",
    AUDIT,
)
